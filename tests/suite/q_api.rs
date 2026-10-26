//! q-lift, the item's test code: the call over Q through the public API,
//! with its bounds, its seed, its trace, the checker parameter, and what it
//! refuses (plan.md, _Edge cases_).

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use grobx::id::RowId;
use grobx::trace::{NoTrace, RowSource, Trace};
use grobx::{Cancel, CheckerMode, Error, Stop};

use crate::common::{dense, params, run};

/// A trace that counts the matrices it hears of.
#[derive(Default)]
struct Count(usize);

impl Trace for Count {
    fn matrix(&mut self, _: &[RowSource]) {
        self.0 += 1;
    }

    fn counted(&mut self, _: &[RowId]) {}
}

/// A dense cubic system on five variables over Q runs far past 50 ms.
#[test]
fn bounds_stop_a_call_over_q() {
    let text = dense(5, 3, 0, 1);
    let mut p = params(1);
    p.deadline = Some(Duration::from_millis(50));
    let start = Instant::now();
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Deadline));
    assert_eq!(outcome.basis, None);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{elapsed:?}",
        elapsed = start.elapsed()
    );

    let mut p = params(1);
    let cancel = Cancel::new();
    cancel.set();
    p.cancel = Some(cancel);
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel));

    let mut p = params(1);
    p.memory = Some(1);
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Memory));
    assert_eq!(outcome.basis, None);
}

#[test]
fn the_seed_decides() {
    let text = dense(3, 3, 0, 1);
    let a = run(&text, &params(7));
    let b = run(&text, &params(7));
    assert_eq!(a, b);
    for seed in 1..=3 {
        assert_eq!(run(&text, &params(seed)).basis, a.basis, "seed {seed}");
    }
}

/// q-lift.30, q-lift.35: learn, replay, and the test code's reader refuse
/// Q.
#[test]
fn learn_replay_and_the_reader_refuse_q() {
    let q = "x,y\n0\nx^2+y,\nx*y+1\n";
    let gfp = "x,y\n65521\nx^2+y,\nx*y+1\n";
    assert!(matches!(
        grobx::learn(q, &params(1), &mut NoTrace),
        Err(Error::Characteristic { .. })
    ));
    let (_, record) = grobx::learn(gfp, &params(1), &mut NoTrace).unwrap();
    let record = record.unwrap();
    assert!(matches!(
        grobx::replay(q, &params(1), &record, &mut NoTrace),
        Err(Error::Characteristic { .. })
    ));
    assert!(matches!(
        grobx::text::generators(q),
        Err(Error::Characteristic { .. })
    ));
}

/// q-lift.31: a call over Q tells its trace the matrices of every run, more
/// than one call over one prime tells.
#[test]
fn the_trace_hears_every_run() {
    let text = dense(3, 3, 0, 1);
    let mut over_q = Count::default();
    let outcome = grobx::groebner_with(&text, &params(1), &mut over_q).unwrap();
    assert!(outcome.basis.is_some());
    let one_prime = text.replacen("\n0\n", "\n1073741827\n", 1);
    let mut over_p = Count::default();
    grobx::groebner_with(&one_prime, &params(1), &mut over_p).unwrap();
    assert!(over_p.0 > 0);
    assert!(
        over_q.0 > over_p.0,
        "{matrices_q} matrices over Q, {matrices_p} over one prime",
        matrices_q = over_q.0,
        matrices_p = over_p.0
    );
}

#[test]
fn the_checker_parameter_over_q() {
    let text = dense(3, 3, 0, 1);
    let basis = run(&text, &params(3)).basis;
    for mode in [
        CheckerMode::Exact,
        CheckerMode::Cheap {
            points: NonZeroU32::new(3).unwrap(),
        },
    ] {
        let mut p = params(3);
        p.checker = Some(mode);
        let outcome = run(&text, &p);
        assert_eq!(
            outcome.certified,
            Some(true),
            "{mode:?}: {flaw:?}",
            flaw = outcome.flaw
        );
        assert_eq!(outcome.flaw, None);
        assert_eq!(outcome.basis, basis, "{mode:?}");
        assert_eq!(outcome.cofactors, None, "{mode:?}");
    }
}

#[test]
fn a_characteristic_of_zeros_is_q() {
    assert_eq!(
        run("x\n000\nx-1/2\n", &params(1)).basis.as_deref(),
        Some("[2*x-1]:\n")
    );
}
