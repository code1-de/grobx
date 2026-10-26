//! outgrown-round-size, the crate's test code: the early round's tests
//! read a round of `EARLY_ROUND` elements or more where the plain run would
//! combine rows and of `EARLY_ROUND_ALONE` where it would reduce each row
//! alone, and the outgrown test reads every round that made an element
//! (plan.md, _Test code_, 1 to 3).

use std::cell::Cell;

use crate::cofactors::{NoCofactors, Tracked};
use crate::early_handover_tests::after;
use crate::f4::{
    DIVISIBLE, EARLY_ROUND, EARLY_ROUND_ALONE, EARLY_ZEROS, Handover, OUTGROWN, Tests,
    plain_combines,
};
use crate::field::Gf31;
use crate::monomial::Dense16;
use crate::trace::{NoLearner, NoTrace, Recording};

/// Test 1, outgrown-round-size.1 to .3: the early round's tests read a
/// round of `EARLY_ROUND` elements or more where the plain run would
/// combine rows, and of `EARLY_ROUND_ALONE` or more where it would not.
/// Below the least the early test stays, and the outgrown test reads the
/// count once; at the least
/// the zero test and the share test decide as before; the early round
/// decides once.
#[test]
fn the_early_round_reads_by_the_plain_runs_mode() {
    assert_eq!((EARLY_ROUND, EARLY_ROUND_ALONE), (25, 100));
    let calls = Cell::new(0);
    for (combines, least) in [(true, EARLY_ROUND), (false, EARLY_ROUND_ALONE)] {
        let mode = format!("combines {combines}");

        // One element fewer than the least: the early test stays, the run
        // stays, and the count is read once.
        calls.set(0);
        let mut h = Handover::new(combines);
        assert!(
            !after(&mut h, true, false, (least - 1, 0, 0), &calls),
            "{mode}"
        );
        assert_eq!(h.tests().early, Some(false), "{mode}");
        assert_eq!(calls.get(), 1, "{mode}: the outgrown test's count");

        // The least with no zero: the zero test hands over before the
        // count.
        calls.set(0);
        let mut h = Handover::new(combines);
        assert!(after(&mut h, true, false, (least, 0, 0), &calls), "{mode}");
        assert_eq!(h.tests().early, Some(true), "{mode}");
        assert_eq!(calls.get(), 0, "{mode}: no count read");

        // The fewest zeros not below a quarter, and 6 in 10 divisible
        // exactly: the run stays; one divisible more hands over.
        let quarter = (EARLY_ZEROS.n * least).div_ceil(EARLY_ZEROS.d);
        let share = DIVISIBLE.n * least / DIVISIBLE.d;
        assert!(
            !after(
                &mut Handover::new(combines),
                true,
                false,
                (least, quarter, share),
                &calls
            ),
            "{mode}: {quarter} zeros, {share} divisible"
        );
        assert!(
            after(
                &mut Handover::new(combines),
                true,
                false,
                (least, quarter, share + 1),
                &calls
            ),
            "{mode}: {quarter} zeros, {divisible} divisible",
            divisible = share + 1
        );

        // An early round below the least, then one at the least with no
        // zero: the early round decided at the first, and the second stays.
        let mut h = Handover::new(combines);
        assert!(
            !after(&mut h, true, false, (least - 1, 0, 0), &calls),
            "{mode}"
        );
        assert!(
            !after(&mut h, true, false, (least, 0, 0), &calls),
            "{mode}: once"
        );
        assert_eq!(
            h.tests().early,
            None,
            "{mode}: the early tests read no second round"
        );
    }
}

/// Test 2, outgrown-round-size.4: the outgrown test reads every round that
/// made an element, under both modes, and hands over where 8 in 10 of its
/// elements or more are divisible; a round that made none no test reads.
#[test]
fn the_outgrown_test_reads_every_round_that_made_one() {
    let calls = Cell::new(0);
    for combines in [true, false] {
        calls.set(0);
        let mut h = Handover::new(combines);
        assert!(!after(&mut h, false, false, (0, 0, 0), &calls));
        assert_eq!(
            calls.get(),
            0,
            "combines {combines}: no count for no element"
        );
        assert_eq!(h.tests(), Tests::default(), "combines {combines}");
        for made in [1_usize, 4, 5, 10, 99] {
            let least = (OUTGROWN.n * made).div_ceil(OUTGROWN.d);
            for (divisible, hands) in [(least, true), (least - 1, false)] {
                let case = format!("combines {combines}: {divisible} of {made} divisible");
                calls.set(0);
                let mut h = Handover::new(combines);
                assert_eq!(
                    after(&mut h, false, false, (made, made, divisible), &calls),
                    hands,
                    "{case}"
                );
                assert_eq!(calls.get(), 1, "{case}: the count read once");
                assert_eq!(h.tests().outgrown, Some(hands), "{case}");
            }
        }
    }
}

/// Test 3, outgrown-round-size.5: the plain run combines rows at 65521
/// where neither the trace nor the learner listens and no cofactors are
/// tracked, and not at 7, nor where the learner listens, nor where
/// cofactors are tracked.
#[test]
fn the_plain_runs_mode() {
    type C = NoCofactors;
    assert!(plain_combines::<Gf31, Dense16, C, NoTrace, NoLearner>(
        65521
    ));
    assert!(!plain_combines::<Gf31, Dense16, C, NoTrace, NoLearner>(7));
    assert!(!plain_combines::<Gf31, Dense16, C, NoTrace, Recording>(
        65521
    ));
    assert!(!plain_combines::<
        Gf31,
        Dense16,
        Tracked<Gf31, Dense16>,
        NoTrace,
        NoLearner,
    >(65521));
}
