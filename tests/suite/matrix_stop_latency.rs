//! matrix-stop-latency, the item's test code: calls whose round matrices
//! reach a full check of the budget within their work
//! (matrix-stop-latency.1 to .9). Square systems of `random_square` stop
//! soon past their deadline in `block:1,…,1`; calls with every bound
//! generous return the bases of the calls without bounds; a cancel and a
//! memory ceiling are read at the same checks; and the edge cases of
//! `OBJECTIVES.md` (plan.md, _Test code_, 4 to 7).

use std::thread;
use std::time::{Duration, Instant};

use grobx::{Cancel, CheckerMode, Params, Stop};
use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted, dense, params, params_in, random_square, run, with_cofactors,
    with_cofactors_in,
};
use crate::orders::{accepted_in, assert_canonical, lex_as_blocks, names};
use crate::q::accepted_q;

/// Every bound set, none of them reached.
fn generous(mut p: Params) -> Params {
    p.deadline = Some(Duration::from_secs(3600));
    p.cancel = Some(Cancel::new());
    p.memory = Some(1 << 40);
    p
}

/// Test 4, .1 to .6: the square systems over 2^31 − 1 whose plain run
/// went 11.0 s between two full checks in the test profile (log 0055-2),
/// seeds 3 and 7, in `block:1,1,1` as `fglm_random.rs` ran them, with a
/// deadline of 300 ms: each stops with `Deadline` within 1 s past it.
#[test]
fn the_square_systems_stop_soon() {
    const DEADLINE: Duration = Duration::from_millis(300);
    for seed in [3, 7] {
        let text = random_square(P31, seed);
        let blocks = lex_as_blocks(names(&text).len());
        assert_eq!(blocks, "block:1,1,1", "seed {seed}: four variables");
        let mut p = params_in(&blocks, 1);
        p.deadline = Some(DEADLINE);
        let start = Instant::now();
        let outcome = grobx::groebner(&text, &p).expect("the call runs");
        let took = start.elapsed();
        assert_eq!(outcome.stopped, Some(Stop::Deadline), "seed {seed}");
        assert_eq!(outcome.basis, None, "seed {seed}");
        assert!(
            took < DEADLINE + Duration::from_secs(1),
            "seed {seed}: returned {late:?} past the deadline",
            late = took.saturating_sub(DEADLINE)
        );
    }
}

/// Test 5, .1 to .9: with a deadline of an hour, a cancel flag never
/// raised, and a memory ceiling of 2^40 bytes, under seeds 1 and 2, each
/// call over GF(65521), GF(2^31 − 1), and ℚ returns the basis of the call
/// without bounds.
#[test]
fn generous_bounds_change_no_basis() {
    for text in [
        dense(4, 3, P, 1),
        random_square(P31, 1),
        dense(3, 3, 0, 1),
        random_square(0, 1),
    ] {
        for seed in [1, 2] {
            let bounded = run(&text, &generous(params(seed)));
            let free = run(&text, &params(seed));
            assert_eq!(bounded.basis, free.basis, "seed {seed}\n{text}");
        }
    }
}

/// Test 6, .1: a cancel raised from another thread and a memory ceiling
/// are read at the same checks as the deadline; with cofactors and the
/// exact checker, generous bounds change nothing.
#[test]
fn every_bound_is_read_at_the_checks() {
    let long = dense(8, 3, P, 1);
    let cancel = Cancel::new();
    let mut p = params(1);
    p.cancel = Some(cancel.clone());
    let stopper = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        cancel.set();
        Instant::now()
    });
    let outcome = grobx::groebner(&long, &p).expect("the call runs");
    let ended = Instant::now();
    let raised = stopper.join().unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel));
    assert!(
        ended.duration_since(raised) < Duration::from_secs(1),
        "{took:?} after the raise",
        took = ended.duration_since(raised)
    );

    let mut p = params(1);
    p.memory = Some(1);
    let outcome = grobx::groebner(&long, &p).expect("the call runs");
    assert_eq!(outcome.stopped, Some(Stop::Memory));

    let text = dense(3, 3, P, 1);
    let mut p = generous(with_cofactors(1));
    p.checker = Some(CheckerMode::Exact);
    let outcome = run(&text, &p);
    assert_eq!(
        outcome.certified,
        Some(true),
        "{flaw:?}",
        flaw = outcome.flaw
    );
    assert_eq!(outcome.basis, run(&text, &params(1)).basis);
    assert_accepted("cubic-3", &text, &outcome, Mode::Exact);
}

/// The edge cases of `OBJECTIVES.md` on x, y, z: name and generators.
const EDGE: [(&str, &str); 7] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// Test 7, .1: the edge cases over GF(65521), GF(2^31 − 1), and ℚ, in
/// grevlex, lex, and `block:1`. With generous bounds each basis is the one
/// without bounds, canonical, and accepted with cofactors by the exact
/// checker; with a deadline of zero each call stops with `Deadline`.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        for p in [P, P31, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                let name = format!("{case} over {p} in {order}");
                let free = run(&text, &params_in(order, 0x0d6e)).basis.unwrap();
                let bounded = run(&text, &generous(params_in(order, 0x0d6e)));
                assert_eq!(bounded.basis.as_deref(), Some(free.as_str()), "{name}");
                if p == 0 {
                    accepted_q(&name, &text, order);
                } else {
                    assert_canonical(&name, &text, &free, order);
                    accepted_in(&name, &text, order, Mode::Exact);
                }
                let with = run(&text, &generous(with_cofactors_in(order, 0x0d6e)));
                assert_eq!(with.basis.as_deref(), Some(free.as_str()), "{name}");

                let mut zero = params_in(order, 0x0d6e);
                zero.deadline = Some(Duration::ZERO);
                let outcome = grobx::groebner(&text, &zero).expect("the call runs");
                assert_eq!(outcome.stopped, Some(Stop::Deadline), "{name}");
                assert_eq!(outcome.basis, None, "{name}");
            }
        }
    }
}
