//! pair-work-ratio, the crate's test code: the pair-work test hands over
//! where the S-pairs a round's update would form reach the entries its row
//! operations subtracted, and over rounds of the tuning set's signature
//! runs (`iterations/0080-pair-work-ratio/plan.md`, _Test code_, 1 and 2).

use crate::f4::{Handover, PAIR_WORK, PAIRS, pairs_formed};

/// The multiple pair-work-handover.3 named, withdrawn: the rounds of test
/// 2 that hand over read less than it.
const WITHDRAWN: u64 = 4;

/// A round of one element, none divisible, that the outgrown test reads
/// and keeps (outgrown-round-size.4), then the pair-work test on `pairs`
/// and `subtracted`: whether the run hands over, and what the test
/// recorded.
fn pair_work(h: &mut Handover, pairs: u64, subtracted: u64) -> (bool, Option<bool>, Option<u64>) {
    assert!(
        !h.after_round(false, false, 1, 0, || 0),
        "no early, gap, or outgrown test hands over"
    );
    let handed = h.after_pairs(pairs, subtracted);
    let t = h.tests();
    (handed, t.pair_work, t.pairs)
}

/// Test 1, pair-work-ratio.1 and .2: `PAIR_WORK` is 1 and `PAIRS` 1,000.
/// A round of `PAIRS` S-pairs or more hands over where they reach the
/// entries subtracted and stays where they fall short by one, at the
/// least the test reads and at the largest counts; a round of fewer
/// S-pairs is not read.
#[test]
fn the_pair_work_test_hands_over_at_the_entries() {
    assert_eq!(PAIR_WORK, 1);
    assert_eq!(PAIRS, 1_000);
    let mut h = Handover::new(true);
    // .2: 999 S-pairs are too few to read, against any entries.
    assert_eq!(pair_work(&mut h, 999, 0), (false, None, None));
    assert_eq!(pair_work(&mut h, 999, 998), (false, None, None));
    // .1: 1,000 S-pairs against 1,000 entries hand over, against 1,001
    // stay, and against 999 or none hand over.
    for (subtracted, hands) in [(1_000, true), (1_001, false), (999, true), (0, true)] {
        assert_eq!(
            pair_work(&mut h, 1_000, subtracted),
            (hands, Some(hands), Some(1_000)),
            "1,000 S-pairs against {subtracted} entries"
        );
    }
    // Between the entries and 4 times them, where the withdrawn multiple
    // stayed, the test hands over.
    for (pairs, subtracted) in [
        (4_000, 1_001),
        (4_000, 2_000),
        (4_000, 3_999),
        (1 << 40, (1 << 38) + 1),
    ] {
        assert!(
            pairs < WITHDRAWN * subtracted,
            "{pairs} against {subtracted}"
        );
        assert_eq!(
            pair_work(&mut h, pairs, subtracted),
            (true, Some(true), Some(pairs)),
            "{pairs} against {subtracted}"
        );
    }
    // The largest counts compare without overflow.
    assert_eq!(
        pair_work(&mut h, u64::MAX, u64::MAX),
        (true, Some(true), Some(u64::MAX))
    );
    assert_eq!(
        pair_work(&mut h, u64::MAX - 1, u64::MAX),
        (false, Some(false), Some(u64::MAX - 1))
    );
}

/// Test 2, pair-work-ratio.1: rounds of the tuning set's signature runs,
/// seed 1, read off the call reports of session 0080-2 (log 0080-2, _Step
/// 1_): made, the basis before, and the entries subtracted. The rounds
/// where the plain run from the start takes less time and the withdrawn
/// multiple stayed hand over; the rounds of the calls the signature run
/// takes in less time stay.
#[test]
fn rounds_of_the_tuning_set() {
    for (stem, made, basis, subtracted) in [
        ("colour-g18m76s1-k4-p0, round 1", 146, 94, 6_621),
        ("sat-pret60-25-p65521, round 1", 107, 220, 8_553),
        ("Geometry.Chou.459_1-p0, round 4", 64, 51, 3_122),
        ("Geometry.Chou.460_1-p65521, round 2", 46, 24, 1_270),
        ("colour-paley17-k4-p0, round 1", 87, 85, 7_197),
        ("phc-i1-p0, round 4", 42, 37, 1_906),
    ] {
        let pairs = pairs_formed(made, basis);
        assert!(pairs < WITHDRAWN * subtracted, "{stem}: {pairs}");
        assert_eq!(
            pair_work(&mut Handover::new(true), pairs, subtracted),
            (true, Some(true), Some(pairs)),
            "{stem}"
        );
    }
    for (stem, made, basis, subtracted) in [
        ("noon-9-p0, round 8", 607, 785, 801_307),
        ("noon-9-p0, round 9", 754, 1_392, 1_507_875),
        ("noon-8-p0, round 3", 22, 38, 1_371),
        ("noon-8-p0, round 8", 284, 564, 285_665),
        ("chandra-11-p0, round 2", 166, 66, 45_645),
        ("latin-n4-p0, round 1", 35, 64, 6_491),
        ("colour-g18m76s1-k4-p0, round 2", 570, 240, 330_541),
        ("Geometry.Chou.460_1-p65521, round 4", 198, 178, 89_956),
    ] {
        let pairs = pairs_formed(made, basis);
        assert!(pairs >= PAIRS, "{stem}: {pairs}");
        assert_eq!(
            pair_work(&mut Handover::new(true), pairs, subtracted),
            (false, Some(false), Some(pairs)),
            "{stem}"
        );
    }
}
