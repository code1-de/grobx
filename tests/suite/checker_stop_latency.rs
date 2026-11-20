//! checker-stop-latency, the item's test code: calls with the checker whose
//! check reaches a full check of the budget within its work
//! (checker-stop-latency.1 to .12). Calls whose deadline falls inside the
//! check stop soon past it, and generous bounds change no verdict
//! (plan.md, _Test code_, 5 and 7). Test 6, a cancel read inside the
//! check, is `cancel_check_race.rs`' test 1.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use grobx::{CheckerMode, Params, Stop};

use crate::common::{P, dense, generous, random_square, run, with_cofactors};

/// How long past its deadline or its cancel a call with the checker may
/// return.
pub(crate) const LATENCY: Duration = Duration::from_millis(500);

/// A call with cofactors and the checker in `mode`, under `seed`.
pub(crate) fn checked(mode: CheckerMode, seed: u64) -> Params {
    let mut p = with_cofactors(seed);
    p.checker = Some(mode);
    p
}

pub(crate) fn cheap(points: u32) -> CheckerMode {
    CheckerMode::Cheap {
        points: NonZeroU32::new(points).unwrap(),
    }
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
/// their time; `cancel_check_race.rs`' test 1 takes them too.
pub(crate) fn long() -> [(&'static str, String, CheckerMode); 3] {
    [
        ("cubic-3 over ℚ", dense(3, 3, 0, 1), CheckerMode::Exact),
        (
            "cubic-4 over GF(65521)",
            dense(4, 3, P, 1),
            CheckerMode::Exact,
        ),
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
                assert_eq!(
                    outcome.basis,
                    Some(basis.clone()),
                    "{case} {mode:?} at {share}"
                );
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
                ended < deadline + LATENCY,
                "{case} {mode:?} at {share}: returned {late:?} past the deadline",
                late = ended.saturating_sub(deadline)
            );
        }
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
