//! zero-rows-time, the crate's test code: a learn's plain run that rules
//! its rows zero by tests (learn-zero-time.1 to .6, zero-rows-time.2 to
//! .4) against each row reduced on its own and against a reference of the
//! tests' groups; the primes that take no test or few; and a deadline
//! inside a test (`iterations/0078-zero-rows-time/plan.md`, _Test code_, 5
//! to 7).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::Stop;
use crate::cyclic_gap_tests::{Random, alone, long, rows, short, with_rest};
use crate::f4::Halt;
use crate::field::Gf31;
use crate::matrix::{
    BLOCK, Combine, FIRST, GROUP_ROWS, Pivot, Reduction, STREAK, zero_combinations,
};
use crate::plain_band_tests::passed_after;

/// The primes that take tests of two combinations: below 2^16 and the
/// largest below 2^31.
const TESTED: [u64; 2] = [65521, 2_147_483_647];

/// What the reference of the tests' groups predicts for a matrix, and
/// which of the rule's paths it took.
#[derive(Debug, Default)]
struct Groups {
    reductions: u64,
    zeros: u64,
    /// Tests of the rows to the matrix's end that passed and failed.
    whole_passed: usize,
    whole_failed: usize,
    /// Tests of a block after a failed test that passed and failed.
    block_passed: usize,
    block_failed: usize,
    /// Blocks of `BLOCK` rows that passed, the doubling's cap.
    capped: usize,
    /// Groups of rows left untested, fewer than 2s.
    short: usize,
}

/// The reference of the tests (plan, _Algorithms_): the rows to reduce in
/// the order of cyclic-gap.1, each zero or not as the test's own
/// elimination finds it on its own (`cyclic_gap_tests::alone`). A row is
/// reduced on its own; once `STREAK` rows have reduced to zero one after
/// another, groups of the rows after them are tested, the first the rows
/// to the end, or `FIRST` rows where a test has failed in the matrix, each
/// next group twice the rows of the one before that passed, at most
/// `BLOCK`, until a group fails or holds fewer than 2s rows. A group
/// passes where every row of it reduces to zero, at s reductions and s
/// zeros, and fails at its first combination, one reduction, save with
/// probability about 1/p.
fn groups(m: &Random, s: usize) -> Groups {
    let (zero, _) = alone(m);
    let (_, rest) = m.split();
    let flags: Vec<bool> = rest.iter().map(|&r| zero.contains(&(r as u32))).collect();
    let mut g = Groups::default();
    let (mut streak, mut failed) = (0, false);
    let mut next = 0;
    while next < flags.len() {
        g.reductions += 1;
        let is_zero = flags[next];
        next += 1;
        if !is_zero {
            streak = 0;
            continue;
        }
        g.zeros += 1;
        streak += 1;
        if streak < STREAK {
            continue;
        }
        streak = 0;
        let mut block = if failed { FIRST } else { usize::MAX };
        loop {
            let len = block.min(flags.len() - next);
            if len < GROUP_ROWS * s {
                g.short += usize::from(len > 0);
                break;
            }
            let whole = block == usize::MAX;
            if flags[next..next + len].iter().all(|&z| z) {
                g.reductions += s as u64;
                g.zeros += s as u64;
                next += len;
                g.capped += usize::from(len == BLOCK);
                if whole {
                    g.whole_passed += 1;
                } else {
                    g.block_passed += 1;
                }
                block = (2 * len).min(BLOCK);
            } else {
                g.reductions += 1;
                failed = true;
                if whole {
                    g.whole_failed += 1;
                } else {
                    g.block_failed += 1;
                }
                break;
            }
        }
    }
    g
}

/// The pivot of every column, as kinds and indices.
fn pivots(reduction: &Reduction<Gf31>) -> Vec<Option<(bool, usize)>> {
    reduction
        .pivot
        .iter()
        .map(|p| match p {
            Pivot::None => None,
            Pivot::Row(r) => Some((false, r.index())),
            Pivot::New(k) => Some((true, k.index())),
        })
        .collect()
}

/// `m` reduced with the tests drawn from `seed` against without: the same
/// made rows in the same order, each with its row, columns, and
/// coefficients; the same pivot of every column; and the same rows reduced
/// to zero in the same order. The reduction with the tests.
fn assert_as_alone(name: &str, m: &Random, seed: u64) -> Reduction<Gf31> {
    let alone = m.reduce(None);
    let tested = m
        .reduce_with(
            None,
            Some(&mut Combine::new(seed)),
            &mut crate::cyclic_gap_tests::budget(),
        )
        .unwrap();
    let order = |r: &Reduction<Gf31>| r.made.iter().map(|x| x.row).collect::<Vec<_>>();
    assert_eq!(
        order(&tested),
        order(&alone),
        "{name}: the made rows' order"
    );
    assert_eq!(rows(&tested), rows(&alone), "{name}: the made rows");
    assert_eq!(pivots(&tested), pivots(&alone), "{name}: the pivots");
    assert_eq!(tested.zero, alone.zero, "{name}: the rows reduced to zero");
    tested
}

/// The shape of case `seed`: short rows, long rows, or short rows whose
/// rows to reduce are mostly copies of one element.
fn case(p: u64, seed: u64, kind: u64) -> Option<Random> {
    let k = seed as usize;
    match kind {
        0 => Some(Random::new(
            p,
            0x7e00 + seed,
            &short(8 + k % 12, 40 + 4 * k, [0, 1, 2, 3][k % 4], 6 + k % 20),
        )),
        1 => Some(Random::new(
            p,
            0x7e40 + seed,
            &long(8 + k % 6, 30 + 3 * k, [0, 2, 3][k % 3], 4 + k % 12),
        )),
        _ => with_rest(
            p,
            0x7e80 + seed,
            &short(6 + k % 6, 6 + k % 10, 0, 0),
            40 + 9 * k,
        ),
    }
}

/// Test 5: learn-zero-time.1, .2, zero-rows-time.2 to .4.
/// `matrix::reduce` with the learn's tests against without, at 65521 and
/// 2^31 − 1, on matrices of short rows, of long rows, and of rows to
/// reduce that are mostly copies of one element: the same made rows,
/// pivots, and rows reduced to zero (`assert_as_alone`); and the
/// reductions and zeros the reference of the groups predicts. Over the
/// cases every path of the rule is taken.
#[test]
fn the_tests_rule_rows_zero() {
    assert_eq!((STREAK, FIRST, BLOCK), (2, 8, 64));
    let mut paths = Groups::default();
    let mut cases = 0;
    for p in TESTED {
        let s = zero_combinations(p).unwrap();
        assert_eq!(s, 2, "p {p}");
        for seed in 0..40u64 {
            for kind in 0..3 {
                let Some(m) = case(p, seed, kind) else {
                    continue;
                };
                let name = format!("p {p}, seed {seed}, kind {kind}");
                let tested = assert_as_alone(&name, &m, seed);
                let g = groups(&m, s);
                assert_eq!(
                    (tested.reductions, tested.zeros),
                    (g.reductions, g.zeros),
                    "{name}: {g:?}"
                );
                paths.whole_passed += g.whole_passed;
                paths.whole_failed += g.whole_failed;
                paths.block_passed += g.block_passed;
                paths.block_failed += g.block_failed;
                paths.capped += g.capped;
                paths.short += g.short;
                cases += 1;
            }
        }
    }
    assert!(cases >= 200, "{cases} matrices");
    for (path, n) in [
        ("whole tests passed", paths.whole_passed),
        ("whole tests failed", paths.whole_failed),
        ("block tests passed", paths.block_passed),
        ("block tests failed", paths.block_failed),
        ("blocks capped at BLOCK", paths.capped),
        ("short groups untested", paths.short),
    ] {
        assert!(n >= 20, "{n} {path}: {paths:?}");
    }
}

/// Test 6: signature-zero-time.8 standing. At 2 and 3
/// the tests change nothing, the counts included. At 7, s is 12: the
/// reductions that are no zero are the reference's with groups of 24 rows
/// or more, its zeros and up to s − 1 more for each failed test, and some
/// group of fewer rows stands untested.
#[test]
fn the_small_primes() {
    for p in [2, 3] {
        assert_eq!(zero_combinations(p), None);
        for seed in 0..12u64 {
            for kind in 0..3 {
                let Some(m) = case(p, seed, kind) else {
                    continue;
                };
                let name = format!("p {p}, seed {seed}, kind {kind}");
                let alone = m.reduce(None);
                let tested = assert_as_alone(&name, &m, seed);
                assert_eq!(
                    (tested.reductions, tested.zeros),
                    (alone.reductions, alone.zeros),
                    "{name}"
                );
            }
        }
    }
    let s = zero_combinations(7).unwrap();
    assert_eq!(s, 12);
    let (mut passed, mut short) = (0, 0);
    for seed in 0..40u64 {
        for kind in 0..3 {
            let Some(m) = case(7, seed, kind) else {
                continue;
            };
            let name = format!("p 7, seed {seed}, kind {kind}");
            let tested = assert_as_alone(&name, &m, seed);
            let g = groups(&m, s);
            // At 7 a failed test's combinations reduce to zero by chance,
            // 1 in 6 each, before the one that does not: each failed test
            // still takes one reduction that is no zero.
            let failed = (g.whole_failed + g.block_failed) as u64;
            assert_eq!(
                tested.reductions - tested.zeros,
                g.reductions - g.zeros,
                "{name}: {g:?}"
            );
            assert!(
                (g.zeros..=g.zeros + failed * (s as u64 - 1)).contains(&tested.zeros),
                "{name}: {zeros} zeros, {g:?}",
                zeros = tested.zeros
            );
            passed += g.whole_passed + g.block_passed;
            short += g.short;
        }
    }
    assert!(passed >= 5, "{passed} tests passed at 7");
    assert!(short >= 5, "{short} short groups at 7");
}

/// Test 7: a deadline inside the tests. The matrices of test 5 at 65521
/// whose reference runs a test; a deadline passed and unchecked, the
/// setup's units and `T` more between two full checks, for `T` one to
/// seven eighths of the units past the setup's that the reduction with the
/// tests spends unbounded: the reduction stops with `Deadline` after one
/// full check and at most `T` + W units past the setup's, W the matrix's
/// width, which bounds the length of every row added or subtracted.
#[test]
fn a_deadline_stops_the_tests() {
    let mut cases = 0;
    for seed in 0..40u64 {
        for kind in 0..3 {
            let Some(m) = case(65521, seed, kind) else {
                continue;
            };
            let g = groups(&m, 2);
            if g.whole_passed + g.whole_failed + g.block_passed + g.block_failed == 0 {
                continue;
            }
            let setup = (m.matrix.rows.len() + m.matrix.columns()) as u64;
            let width = m.matrix.columns() as u64;
            let mut b = crate::cyclic_gap_tests::budget();
            m.reduce_with(None, Some(&mut Combine::new(seed)), &mut b)
                .unwrap();
            let past = b.spent() - setup;
            for eighth in 1..8 {
                let t = past * eighth / 8;
                let name = format!("seed {seed}, kind {kind}, T {t}");
                let mut b = passed_after(setup + t);
                let got = m.reduce_with(None, Some(&mut Combine::new(seed)), &mut b);
                assert!(
                    matches!(got, Err(Halt::Stop(Stop::Deadline))),
                    "{name}: not stopped by the deadline"
                );
                assert_eq!(b.checks(), 1, "{name}");
                assert!(
                    b.spent() <= setup + t + width,
                    "{name}: {spent} spent before the stop, {past} past the setup in all",
                    spent = b.spent()
                );
            }
            cases += 1;
        }
    }
    assert!(cases >= 30, "{cases} matrices with a test");
}
