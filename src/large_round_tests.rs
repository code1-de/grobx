//! cyclic-9-unsolved, the crate's test code: the large-round test of the
//! handover over the rounds of a signature run, and calls whose
//! large-round test the test profile lowers to every round that made an
//! element (`iterations/0093-cyclic-9-unsolved/plan.md`, _Test code_, 1
//! to 5).

use std::cell::Cell;
use std::num::NonZeroUsize;

use crate::f4::{Handover, LARGE, LARGE_DIVISIBLE, LARGE_LEAST, LARGE_ZEROS};
use crate::report_tests::lines;
use crate::shared_tests::systems;
use crate::{CheckerMode, MonomialOrder, Params, Threads};

/// The least count of `made` elements that reaches `LARGE_DIVISIBLE`.
fn least_divisible(made: usize) -> usize {
    (made * LARGE_DIVISIBLE.n).div_ceil(LARGE_DIVISIBLE.d)
}

/// The most rows reduced to zero that lie below `LARGE_ZEROS` of `made`
/// elements, for `made` of at least `LARGE_ZEROS.d`.
fn most_zeros(made: usize) -> usize {
    (made * LARGE_ZEROS.n - 1) / LARGE_ZEROS.d
}

/// `after_round` of a round that is neither the early round nor the gap
/// round, with a `divisible` that returns `n` and counts its calls in
/// `calls`.
fn after(h: &mut Handover, made: usize, zeros: usize, n: usize, calls: &Cell<u32>) -> bool {
    h.after_round(false, false, made, zeros, || {
        calls.set(calls.get() + 1);
        Ok(n)
    })
    .unwrap()
}

/// Test 1: cyclic-9-unsolved.1 to .4. A round of `LARGE` elements or more
/// hands over where `LARGE_DIVISIBLE` or more of them are divisible and it
/// reduced fewer rows to zero than `LARGE_ZEROS` of them; a round one
/// element short is not read; the count runs once.
#[test]
fn the_large_round_reads_its_shares() {
    let (made, d, z) = (LARGE, least_divisible(LARGE), most_zeros(LARGE));
    for combines in [true, false] {
        let calls = Cell::new(0);
        let mut h = Handover::new(combines);
        assert!(after(&mut h, made, z, d, &calls), "combines {combines}");
        assert_eq!(h.tests().large, Some(true), "combines {combines}");
        assert_eq!(h.tests().outgrown, Some(false), "combines {combines}");
        assert_eq!(h.tests().divisible, Some(d), "combines {combines}");
        assert_eq!(calls.get(), 1, "the count runs once");
        let cases = [
            ("one divisible short", made, z, d - 1),
            ("one zero more", made, z + 1, d),
        ];
        for (case, made, zeros, n) in cases {
            let mut h = Handover::new(combines);
            assert!(!after(&mut h, made, zeros, n, &calls), "{case}");
            assert_eq!(h.tests().large, Some(false), "{case}");
        }
        let short = LARGE - 1;
        let mut h = Handover::new(combines);
        let n = least_divisible(short);
        assert!(!after(&mut h, short, 0, n, &calls), "one element short");
        assert_eq!(h.tests().large, None, "one element short");
        assert_eq!(h.tests().outgrown, Some(false), "one element short");
    }
}

/// Test 2: cyclic-9-unsolved.5. The large-round test reads every round of
/// `LARGE` elements or more: rounds that keep the run come before the one
/// that hands it over, and each reads the count once.
#[test]
fn the_large_round_reads_every_round() {
    let calls = Cell::new(0);
    let mut h = Handover::new(true);
    let made = 2 * LARGE;
    // Too many zeros, then too few divisible elements: the run stays.
    assert!(!after(&mut h, made, most_zeros(made) + 1, made / 2, &calls));
    assert_eq!(h.tests().large, Some(false));
    assert!(!after(&mut h, made, 0, least_divisible(made) - 1, &calls));
    assert_eq!(h.tests().large, Some(false));
    // A small round between them is not read.
    assert!(!after(&mut h, 10, 0, 5, &calls));
    assert_eq!(h.tests().large, None);
    let made = 3 * LARGE;
    let d = least_divisible(made);
    assert!(after(&mut h, made, most_zeros(made), d, &calls));
    assert_eq!(h.tests().large, Some(true));
    assert_eq!(calls.get(), 4, "each round read the count once");
}

/// Test 3: cyclic-9-unsolved.6. The early, the gap, and the outgrown tests
/// read a round first, and where one of them hands over, the large-round
/// test does not read it; where the large-round test keeps the run, the
/// pair-work test reads the round, and where it hands over, no pair-work
/// verdict stands.
#[test]
fn the_earlier_tests_go_first() {
    let calls = Cell::new(0);
    let made = LARGE;
    let (d, z) = (least_divisible(made), most_zeros(made));
    // The early round, which reduced fewer rows to zero than a quarter.
    let mut h = Handover::new(true);
    assert!(h.after_round(true, false, made, z, || Ok(d)).unwrap());
    assert_eq!(h.tests().early, Some(true));
    assert_eq!(h.tests().large, None);
    // The gap round, which reduced fewer rows to zero than half.
    let mut h = Handover::new(true);
    assert!(h.after_round(false, true, made, z, || Ok(d)).unwrap());
    assert_eq!(h.tests().gap, Some(true));
    assert_eq!(h.tests().large, None);
    // An outgrown round: 8 in 10 of its elements divisible.
    let mut h = Handover::new(true);
    assert!(after(&mut h, made, z, made * 8 / 10, &calls));
    assert_eq!(h.tests().outgrown, Some(true));
    assert_eq!(h.tests().large, None);
    // The large-round test hands over, and the run asks no pair-work test.
    let mut h = Handover::new(true);
    assert!(after(&mut h, made, z, d, &calls));
    assert_eq!(h.tests().pair_work, None);
    // It keeps the run, and the pair-work test reads the round.
    let mut h = Handover::new(true);
    assert!(!after(&mut h, made, z + 1, d, &calls));
    assert!(h.after_pairs(u64::MAX, 1));
    assert_eq!(h.tests().large, Some(false));
    assert_eq!(h.tests().pair_work, Some(true));
}

/// The parameters of a call in grevlex on one thread under `seed`, with
/// its report.
fn reported(seed: u64) -> Params {
    let mut params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    params.report = true;
    params
}

/// The basis text of the call of `text` and its report's handover lines
/// that hold a large-round verdict, with the large-round test's least as
/// `least` sets it; None where an error ended the call.
fn call(text: &str, params: &Params, least: Option<usize>) -> Option<(String, Vec<String>)> {
    LARGE_LEAST.set(least);
    let outcome = crate::groebner(text, params);
    LARGE_LEAST.set(None);
    let outcome = outcome.ok()?;
    let report = outcome.report.unwrap();
    let large = lines(&report)
        .into_iter()
        .filter(|l| l.kind == "handover")
        .filter_map(|l| l.get("large").map(str::to_string))
        .collect();
    Some((outcome.basis.unwrap(), large))
}

/// Test 4: cyclic-9-unsolved.1 and .5 with the large-round test lowered to
/// every round that made an element: over 7, 65521, 2^31 − 1, and ℚ, and
/// with cofactors and the exact checker over 65521, every call returns the
/// basis of the call under `LARGE`, and some calls hand over by the
/// lowered test.
#[test]
fn lowered_large_rounds_leave_every_basis() {
    let mut handed = 0;
    for p in [7, 65521, 2_147_483_647, 0] {
        for (i, text) in systems(p, 12, 4).iter().enumerate() {
            let params = reported(0x1a7e ^ i as u64);
            let Some((want, _)) = call(text, &params, None) else {
                continue;
            };
            let (basis, large) = call(text, &params, Some(1)).unwrap();
            assert_eq!(basis, want, "p {p}, system {i}\n{text}");
            handed += large.iter().filter(|v| *v == "hand-over").count();
        }
    }
    assert!(handed > 0, "no call handed over by the lowered test");
    for (i, text) in systems(65521, 12, 4).iter().enumerate() {
        let mut params = reported(0x1a7f ^ i as u64);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        let Some((want, _)) = call(text, &params, None) else {
            continue;
        };
        let (basis, _) = call(text, &params, Some(1)).unwrap();
        assert_eq!(basis, want, "cofactors, system {i}\n{text}");
    }
}

/// Test 5: cyclic-9-unsolved.7. With the large-round test lowered, a
/// handover line that holds its verdict names it between the outgrown
/// test's and the pair-work test's, before `made`; under `LARGE` the
/// small systems' lines hold none.
#[test]
fn a_large_round_writes_its_verdict() {
    let mut seen = false;
    for (i, text) in systems(65521, 12, 4).iter().enumerate() {
        let params = reported(0x1a80 ^ i as u64);
        for least in [Some(1), None] {
            LARGE_LEAST.set(least);
            let outcome = crate::groebner(text, &params);
            LARGE_LEAST.set(None);
            let Ok(outcome) = outcome else {
                continue;
            };
            for line in lines(&outcome.report.unwrap()) {
                if line.kind != "handover" {
                    continue;
                }
                let keys: Vec<&str> = line.fields.iter().map(|(k, _)| k.as_str()).collect();
                let Some(at) = keys.iter().position(|&k| k == "large") else {
                    continue;
                };
                assert!(least.is_some(), "system {i}: {line:?}");
                seen = true;
                assert_eq!(keys[at - 1], "outgrown", "{line:?}");
                assert!(["pair-work", "made"].contains(&keys[at + 1]), "{line:?}");
                assert!(
                    ["stay", "hand-over"].contains(&line.get("large").unwrap()),
                    "{line:?}"
                );
            }
        }
    }
    assert!(seen, "no handover line held a large-round verdict");
}
