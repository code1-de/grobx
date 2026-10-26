//! pair-work-handover, the crate's test code: the S-pairs an update forms,
//! and the pair-work test over the rounds of a signature run
//! (`iterations/0067-pair-work-handover/plan.md`, _Test code_, 1 to 3).

use crate::f4::{Handover, PAIRS, pairs_formed};

/// The S-pairs an update forms by counting them: each new element with
/// every element before it.
fn counted(made: usize, basis: usize) -> u64 {
    (0..made).map(|k| (basis + k) as u64).sum()
}

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
    assert_eq!(
        (t.early, t.gap, t.outgrown, t.divisible),
        (None, None, Some(false), Some(0)),
        "the outgrown test alone reads the round, and keeps the run"
    );
    (handed, t.pair_work, t.pairs)
}

/// Test 1, pair-work-handover.1: the S-pairs an update forms where `made`
/// elements enter a basis, made·basis + made·(made − 1)/2, as counted
/// one by one, as session 0067-1 read them off the reports, and
/// `u64::MAX` past it.
#[test]
fn the_pairs_an_update_forms() {
    for made in 0..60 {
        for basis in 0..60 {
            assert_eq!(
                pairs_formed(made, basis),
                counted(made, basis),
                "{made} into {basis}"
            );
        }
    }
    // satlib.uuf50_8-p65521: its 268 inputs entered one by one, the
    // `considered` of the run's `ran` line where it hands over after round
    // 1, and its rounds 1 to 3.
    assert_eq!(pairs_formed(268, 0), 35_778);
    assert_eq!(pairs_formed(112, 268), 36_232);
    assert_eq!(pairs_formed(2051, 380), 2_881_655);
    assert_eq!(pairs_formed(7103, 2431), 42_490_146);
    // No element made forms no pair, and the first element none.
    assert_eq!(pairs_formed(0, 1 << 20), 0);
    assert_eq!(pairs_formed(1, 0), 0);
    // Past u64, the count stops at its largest value.
    let k = u32::MAX as usize;
    assert_eq!(
        u128::from(pairs_formed(k, 0)),
        u128::from(u32::MAX) * (u128::from(u32::MAX) - 1) / 2
    );
    assert_eq!(pairs_formed(1 << 32, 1 << 32), u64::MAX);
    assert_eq!(pairs_formed(usize::MAX, usize::MAX), u64::MAX);
}

/// Test 2, pair-work-handover.4: the pair-work test reads a round whose
/// update would form `PAIRS` S-pairs or more. Its multiple, `PAIR_WORK`,
/// is pair-work-ratio.1's, whose test code asserts it.
#[test]
fn the_pair_work_test_reads_a_round_of_pairs() {
    assert_eq!(PAIRS, 1_000);
    let mut h = Handover::new(true);
    // .4: 999 S-pairs are too few to read, however little was subtracted.
    assert_eq!(pair_work(&mut h, 999, 0), (false, None, None));
    // A round that subtracted nothing hands over where the test reads it.
    assert_eq!(pair_work(&mut h, 1_000, 0), (true, Some(true), Some(1_000)));
    // The next round's `after_round` clears what the test recorded.
    assert!(!h.after_round(false, false, 1, 0, || 0));
    assert_eq!((h.tests().pair_work, h.tests().pairs), (None, None));
}

/// Test 3, pair-work-handover.2 and .6: the pair-work test reads every
/// round of its size, before and after the early and the gap round have
/// decided, and reads a round after the outgrown test has read it and
/// stayed. The counts are session 0067-1's: made, the basis, and the
/// entries subtracted, of rounds of the tuning set's signature runs.
#[test]
fn the_pair_work_test_reads_every_round() {
    // Rounds that hand over: the S-pairs reach 4 times the entries, and
    // so the entries.
    for (stem, made, basis, subtracted) in [
        ("satlib.uuf50_8-p65521, round 1", 112, 268, 5_617),
        ("satlib.uuf50_8-p65521, round 2", 2_051, 380, 49_373),
        ("brent-223-r11-fix9s1-p65521, round 1", 936, 144, 1_905),
        ("sat-pret60-25-p65521, round 1", 120, 220, 387),
        ("sian-hiv2-p65521, round 1", 28, 59, 105),
        ("distgeo-fw-laman10-s1-p65521, round 2", 31, 33, 101),
        ("biomd-333-p65521, round 5", 851, 1_068, 21_435),
        ("Gonnet-83_1-p65521, round 7", 1_078, 2_496, 515_110),
    ] {
        let pairs = pairs_formed(made, basis);
        assert!(
            pair_work(&mut Handover::new(true), pairs, subtracted).0,
            "{stem}"
        );
    }
    // Rounds that stay: the entries outweigh the S-pairs, or the test does
    // not read the round.
    for (stem, made, basis, subtracted) in [
        ("noon-9-p0, round 9", 754, 1_392, 1_507_875),
        ("noon-9-p0, round 10", 677, 2_146, 2_622_092),
        ("chandra-12-p65521, round 6", 1_002, 2_685, 66_836_936),
        ("colour-g18m76s1-k4-p0, round 5", 5_073, 3_243, 152_392_540),
        ("minrank-sm10r3k6-p0, round 3", 2_572, 2_981, 48_623_377),
        ("Geometry.Chou.459_1-p0, round 1, 75 S-pairs", 5, 13, 16),
        ("sian-daisymamil4-p65521, round 1, 649 S-pairs", 11, 54, 140),
        ("noon-8-p0, round 2, 268 S-pairs", 8, 30, 150),
    ] {
        let pairs = pairs_formed(made, basis);
        assert!(
            !pair_work(&mut Handover::new(true), pairs, subtracted).0,
            "{stem}"
        );
    }

    // .2: one run's rounds, each read: stay, then hand over.
    let mut h = Handover::new(true);
    assert!(!pair_work(&mut h, 1_000, 1_001).0);
    assert!(pair_work(&mut h, 4_000, 1_000).0);
    // After its early and its gap round stayed, the run's test still reads.
    let mut h = Handover::new(true);
    assert!(
        !h.after_round(true, true, 100, 50, || 0),
        "early and gap rounds that stay"
    );
    assert_eq!((h.tests().early, h.tests().gap), (Some(false), Some(false)));
    assert!(!h.after_pairs(1_000, 1_001));
    assert_eq!(h.tests().pair_work, Some(false));
    assert!(
        !h.after_round(true, true, 1, 0, || 0),
        "neither decides again"
    );
    assert!(h.after_pairs(1_000, 250));
    assert_eq!(h.tests().pair_work, Some(true));
    // .6: after the outgrown test read a round and stayed, the pair-work
    // test reads it and hands over; the line keeps both verdicts.
    let mut h = Handover::new(true);
    assert!(
        !h.after_round(false, false, 100, 0, || 79),
        "79 in 100 stay"
    );
    assert!(h.after_pairs(pairs_formed(100, 400), 100));
    let t = h.tests();
    assert_eq!(
        (t.outgrown, t.divisible, t.pair_work, t.pairs),
        (Some(false), Some(79), Some(true), Some(44_950))
    );
}
