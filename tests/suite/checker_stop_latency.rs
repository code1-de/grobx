//! checker-stop-latency, the item's test code: calls with the checker whose
//! check reaches a full check of the budget within its work
//! (checker-stop-latency.1 to .12). Calls whose deadline falls inside the
//! check stop soon past it; a cancel is read inside it; and generous
//! bounds change no verdict (plan.md, _Test code_, 5 to 7).

use std::num::NonZeroU32;
use std::thread;
use std::time::{Duration, Instant};

use grobx::{Cancel, CheckerMode, Params, Stop};

use crate::common::{P, dense, random_square, run, with_cofactors};

/// A call with cofactors and the checker in `mode`, under `seed`.
fn checked(mode: CheckerMode, seed: u64) -> Params {
    let mut p = with_cofactors(seed);
    p.checker = Some(mode);
    p
}

fn cheap(points: u32) -> CheckerMode {
    CheckerMode::Cheap {
        points: NonZeroU32::new(points).unwrap(),
    }
}

/// Every bound set, none of them reached.
fn generous(mut p: Params) -> Params {
    p.deadline = Some(Duration::from_secs(3600));
    p.cancel = Some(Cancel::new());
    p.memory = Some(1 << 40);
    p
}

/// The time of an unbounded call under seed 1, which the checker
/// certifies, and its basis.
fn unbounded(case: &str, text: &str, mode: CheckerMode) -> (Duration, String) {
    let start = Instant::now();
    let outcome = run(text, &checked(mode, 1));
    let took = start.elapsed();
    assert_eq!(
        outcome.certified,
        Some(true),
        "{case}: {flaw:?}",
        flaw = outcome.flaw
    );
    (took, outcome.basis.unwrap())
}

/// The calls of test 5, dense cubic systems whose check takes much of
/// their time; test 6 takes the first.
fn long() -> [(&'static str, String, CheckerMode); 3] {
    [
        ("cubic-3 over ℚ", dense(3, 3, 0, 1), CheckerMode::Exact),
        ("cubic-4 over GF(65521)", dense(4, 3, P, 1), CheckerMode::Exact),
        ("cubic-3 over ℚ", dense(3, 3, 0, 1), cheap(100)),
    ]
}

/// Test 5, .1 to .12: a call with the checker with a deadline at 0.6,
/// 0.8, or 0.95 of its time stops with `Deadline`, with no basis and no
/// verdict, or returns the unbounded call's basis certified, within 0.5 s
/// past the deadline.
#[test]
fn calls_with_the_checker_stop_soon_past_the_deadline() {
    for (case, text, mode) in long() {
        let (took, basis) = unbounded(case, &text, mode);
        for share in [0.6, 0.8, 0.95] {
            let deadline = took.mul_f64(share);
            let mut p = checked(mode, 1);
            p.deadline = Some(deadline);
            let start = Instant::now();
            let outcome = grobx::groebner(&text, &p).expect("the call runs");
            let ended = start.elapsed();
            if outcome.basis.is_some() {
                assert_eq!(outcome.basis, Some(basis.clone()), "{case} {mode:?} at {share}");
                assert_eq!(outcome.certified, Some(true), "{case} {mode:?} at {share}");
            } else {
                assert_eq!(
                    outcome.stopped,
                    Some(Stop::Deadline),
                    "{case} {mode:?} at {share}"
                );
                assert_eq!(outcome.certified, None, "{case} {mode:?} at {share}");
            }
            assert!(
                ended < deadline + Duration::from_millis(500),
                "{case} {mode:?} at {share}: returned {late:?} past the deadline",
                late = ended.saturating_sub(deadline)
            );
        }
    }
}

/// Test 6, .1: a cancel raised from another thread at 0.8 of the
/// unbounded call's time, inside the check, stops the call with `Cancel`
/// within 0.5 s of the raise, or the call has returned the unbounded
/// call's basis certified before it.
#[test]
fn a_cancel_is_read_inside_the_check() {
    let [(case, text, mode), ..] = long();
    let (took, basis) = unbounded(case, &text, mode);
    let at = took.mul_f64(0.8);
    let cancel = Cancel::new();
    let mut p = checked(mode, 1);
    p.cancel = Some(cancel.clone());
    let stopper = thread::spawn(move || {
        thread::sleep(at);
        cancel.set();
        Instant::now()
    });
    let outcome = grobx::groebner(&text, &p).expect("the call runs");
    let ended = Instant::now();
    let raised = stopper.join().unwrap();
    if outcome.basis.is_some() {
        assert_eq!(outcome.basis, Some(basis));
        assert_eq!(outcome.certified, Some(true));
        assert!(
            ended <= raised,
            "{case}: returned its basis after the raise"
        );
    } else {
        assert_eq!(outcome.stopped, Some(Stop::Cancel));
        assert!(
            ended.saturating_duration_since(raised) < Duration::from_millis(500),
            "{case}: returned {took:?} after the raise",
            took = ended.saturating_duration_since(raised)
        );
    }
}

/// Test 7, .1 to .12: with a deadline of an hour, a cancel flag never
/// raised, and a memory ceiling of 2^40 bytes, under seeds 1 and 2, each
/// call with the checker, exact and at 50 points, returns the basis and
/// the cofactor text of the same call without bounds, both calls certified.
#[test]
fn generous_bounds_change_no_verdict() {
    for text in [
        dense(3, 3, P, 1),
        random_square(P, 2),
        dense(2, 3, 0, 1),
        random_square(0, 1),
    ] {
        for mode in [CheckerMode::Exact, cheap(50)] {
            for seed in [1, 2] {
                let bounded = run(&text, &generous(checked(mode, seed)));
                let free = run(&text, &checked(mode, seed));
                let case = format!("{mode:?}, seed {seed}\n{text}");
                assert_eq!(bounded.basis, free.basis, "{case}");
                assert_eq!(bounded.cofactors, free.cofactors, "{case}");
                assert_eq!(
                    bounded.certified,
                    Some(true),
                    "{case}: {flaw:?}",
                    flaw = bounded.flaw
                );
                assert_eq!(free.certified, Some(true), "{case}");
            }
        }
    }
}
