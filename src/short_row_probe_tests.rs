//! short-row-reduce-time, sub-item 2, the crate's test code: the probe
//! decides a round's mode on the binomial systems, whose rows are short,
//! and at the matrix level the made rows are the same under the probe and
//! under each forced mode
//! (`iterations/0103-short-row-reduce-time/plan.md`, _Test code_, 7 and 8).

use crate::MonomialOrder;
use crate::cofactors::NoCofactors;
use crate::cyclic_gap_tests::{Random, led};
use crate::matrix::{self, BLOCK, Buffers, Combine, MODE_FORCED, Modes, PROBE, STREAK};
use crate::module::Rank1;
use crate::monomial::Dense16;
use crate::rebuilt_reducer_tests::{call, reported};
use crate::report::RoundMode;
use crate::report_tests::Line;
use crate::shared_tests::{plain_rounds, reported_budget, rows};
use crate::systems::random_binomials;

fn count(l: &Line, key: &str) -> u64 {
    l.get(key)
        .unwrap_or_else(|| panic!("a round line holds {key}"))
        .parse()
        .unwrap()
}

/// Test 7: short-row-reduce-time.2, .3, .4, .8. On the binomial systems,
/// seeds 1 to 4, with no mode forced, some plain round takes the rows'
/// mode; the first round of `BLOCK + PROBE` given rows or more is the
/// first that names a mode the probe decided, and every round of fewer
/// rows names `blocks`; in a round in the rows' mode every reduction makes
/// a pivot or is a zero but a failed test's combination, the probe rows
/// and the tests' combinations among the reductions. A failed test follows
/// `STREAK`, 2, rows reduced to zero one after another, so the failed
/// tests number at most half the round's `zero-rows`.
#[test]
fn the_probe_decides_on_short_rows() {
    let params = reported(MonomialOrder::Grevlex);
    let mut decided_rows = 0;
    for s in 1..=4 {
        MODE_FORCED.set(None);
        let (basis, lines) = call(&random_binomials(65521, s), &params);
        assert!(basis.is_some());
        let mut probed = false;
        for l in plain_rounds(&lines) {
            let given = usize::try_from(count(l, "given")).unwrap();
            let mode = l.get("mode").expect("a plain round line names its mode");
            let least = BLOCK + PROBE;
            if given < least && !probed {
                assert_eq!(
                    mode, "blocks",
                    "seed {s}: no probe before a round of {least} rows"
                );
            } else {
                probed = true;
            }
            if mode == "rows" {
                decided_rows += 1;
                assert!(
                    l.get("zero-rows").is_some(),
                    "seed {s}: zero-rows in the rows' mode"
                );
                let failed = count(l, "reductions") - count(l, "pivots") - count(l, "zero");
                assert!(
                    failed <= count(l, "zero-rows") / 2,
                    "seed {s}: in the rows' mode every reduction makes a pivot or is a zero but a failed test's combination, one per {STREAK} zero rows at most, {failed} here"
                );
            }
        }
        let least = BLOCK + PROBE;
        assert!(probed, "seed {s}: some round holds {least} rows or more");
    }
    assert!(
        decided_rows > 0,
        "the probe decides the rows' mode on some binomial round"
    );
}

/// Reduces `m` under `modes` where Some, else as a run without modes,
/// with the forced mode `forced`.
fn reduce(
    m: &Random,
    modes: Option<&mut Modes>,
    forced: Option<RoundMode>,
    seed: u64,
) -> matrix::Reduction<crate::field::Gf31> {
    MODE_FORCED.set(forced);
    let out = matrix::reduce::<crate::field::Gf31, Dense16, Rank1, NoCofactors>(
        &m.field,
        &m.basis,
        &m.matrix,
        &mut NoCofactors,
        &mut reported_budget(),
        true,
        Some(&mut Combine::new(seed)),
        &mut Buffers::new(),
        None,
        modes,
    )
    .unwrap();
    MODE_FORCED.set(None);
    out
}

/// Test 8: short-row-reduce-time.1 to .3. A random matrix of short rows
/// whose rows to reduce number `BLOCK + PROBE + 4` or more, at 65521:
/// the made rows after the back-substitution are the same under the
/// probe, under the blocks forced, and under the rows forced, and the
/// `Modes` after the probed reduction holds a decision.
#[test]
fn the_made_rows_are_the_same_in_every_mode() {
    for seed in 1..=6u64 {
        let m = Random::new(65521, seed, &led(40, BLOCK + PROBE + 4));
        let (_, rest) = m.split();
        let to_reduce = rest.len();
        assert!(
            to_reduce >= BLOCK + PROBE + 4,
            "seed {seed}: enough rows to reduce, {to_reduce} here"
        );
        let mut modes = Modes::new();
        let probed = reduce(&m, Some(&mut modes), None, 0x5a0f ^ seed);
        assert!(modes.decision.is_some(), "seed {seed}: the probe decided");
        let blocks = reduce(&m, None, Some(RoundMode::Blocks), 0x5a0f ^ seed);
        let alone = reduce(&m, None, Some(RoundMode::Rows), 0x5a0f ^ seed);
        assert_eq!(
            rows(&probed),
            rows(&blocks),
            "seed {seed}: probe against blocks"
        );
        assert_eq!(
            rows(&probed),
            rows(&alone),
            "seed {seed}: probe against rows"
        );
    }
}
