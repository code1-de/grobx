//! Sub-item 3: the call's parameters (`OBJECTIVES.md`, _Call parameters_)
//! through the Rust API: the order, the threads, the seed, the checker,
//! which certifies the results it accepts, and the cofactors.

use std::num::{NonZeroU32, NonZeroUsize};

use grobx::{CheckerMode, Error, MonomialOrder, Params, Threads};

use crate::common::{P, basis, dense, params, random_system, run};

const TINY: &str = "x,y\n65521\nx^2+y,\nx*y+1\n";

#[test]
fn params_new_leaves_every_option_absent() {
    let p = Params::new(MonomialOrder::Grevlex, Threads::All, 7);
    assert_eq!(p.order, MonomialOrder::Grevlex);
    assert_eq!(p.threads, Threads::All);
    assert_eq!(p.seed, 7);
    assert_eq!(p.deadline, None);
    assert!(p.cancel.is_none());
    assert_eq!(p.memory, None);
    assert_eq!(p.checker, None);
    assert!(!p.cofactors);
}

#[test]
fn unknown_orders_are_refused() {
    for name in ["deglex", "degrevlex", "drl", "", "GREVLEX"] {
        assert!(
            matches!(name.parse::<MonomialOrder>(), Err(Error::Order { .. })),
            "{name:?}"
        );
    }
}

#[test]
fn threads_bound_but_do_not_change_the_basis() {
    let text = dense(4, 3, P, 1);
    let want = basis(&text);
    for threads in [Threads::All, Threads::Count(NonZeroUsize::new(4).unwrap())] {
        let mut p = params(1);
        p.threads = threads;
        assert_eq!(run(&text, &p).basis.unwrap(), want);
    }
}

#[test]
fn seeds_change_no_basis() {
    let text = dense(4, 3, P, 2);
    let want = basis(&text);
    for seed in [0, 1, 2, u64::MAX, 0x5eed] {
        assert_eq!(
            run(&text, &params(seed)).basis.unwrap(),
            want,
            "seed {seed}"
        );
    }
    assert_eq!(run(&text, &params(9)), run(&text, &params(9)));
}

fn certified(text: &str, mode: CheckerMode, cofactors: bool) -> grobx::Outcome {
    let mut p = params(3);
    p.checker = Some(mode);
    p.cofactors = cofactors;
    run(text, &p)
}

#[test]
fn the_checker_certifies_in_both_modes() {
    let six = CheckerMode::Cheap {
        points: NonZeroU32::new(6).unwrap(),
    };
    for text in [
        TINY.to_string(),
        dense(3, 3, P, 1),
        dense(4, 3, P, 2),
        random_system(7, 11),
        "x,y\n65521\n0\n".to_string(),
        "x,y\n65521\nx*y-1,\nx\n".to_string(),
    ] {
        for mode in [CheckerMode::Exact, six] {
            let outcome = certified(&text, mode, false);
            assert_eq!(
                outcome.certified,
                Some(true),
                "{mode:?} on\n{text}\n{outcome:?}"
            );
            assert_eq!(outcome.flaw, None);
            assert_eq!(outcome.cofactors, None, "cofactors were not asked for");
            assert_eq!(outcome.basis.as_deref(), Some(basis(&text).as_str()));
        }
    }
    let outcome = certified(TINY, CheckerMode::Exact, true);
    assert_eq!(outcome.certified, Some(true));
    assert!(outcome.cofactors.is_some());
}

#[test]
fn without_the_checker_nothing_is_certified() {
    let outcome = run(TINY, &params(1));
    assert_eq!(outcome.certified, None);
    assert_eq!(outcome.flaw, None);
}
