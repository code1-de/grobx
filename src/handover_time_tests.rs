//! handover-time, the crate's test code: the handover's decision over the
//! rounds of a signature run (plan.md, _Test code_, 1).

use std::cell::Cell;

use crate::f4::{EARLY_ROUND_ALONE, Handover};

/// `after_round` with a `divisible` that returns `n` and counts its calls
/// in `calls`. A gap round is an early round as well. Every `Handover`
/// here is one whose plain run would reduce each row alone, so that its
/// early round's least is `EARLY_ROUND_ALONE`.
fn after(
    h: &mut Handover,
    gap: bool,
    made: usize,
    zeros: usize,
    n: usize,
    calls: &Cell<u32>,
) -> bool {
    h.after_round(gap, gap, made, zeros, || {
        calls.set(calls.get() + 1);
        n
    })
}

/// Test 1, handover-time.1 to .4: the gap round hands over where it
/// reduced fewer rows to zero than half the elements it made, or where
/// more than 6 in 10 of them are divisible, and decides once; the count
/// runs only where a test reads it.
#[test]
fn the_gap_round_decides_once() {
    let calls = Cell::new(0);
    // A round of no element: no test reads the count. A small round
    // before the gap that the outgrown test keeps: the gap test does not
    // read it (outgrown-round-size.4).
    let mut h = Handover::new(false);
    assert!(!after(&mut h, false, 0, 0, 0, &calls));
    assert_eq!(calls.get(), 0, "no test read the count");
    assert!(!after(&mut h, false, 10, 0, 7, &calls));
    assert_eq!(
        h.tests().gap,
        None,
        "the gap test read no round before the gap"
    );
    assert_eq!(calls.get(), 1, "the outgrown test read the count");
    // 5 zeros of 10 and 6 divisible: the gap round keeps the run.
    calls.set(0);
    assert!(!after(&mut h, true, 10, 5, 6, &calls));
    assert_eq!(
        calls.get(),
        1,
        "the share and outgrown tests read the count once"
    );
    // Later gap rounds that its tests would hand over keep it, where the
    // outgrown test keeps it.
    assert!(!after(&mut h, true, 10, 0, 7, &calls));
    assert!(!after(&mut h, true, 99, 0, 79, &calls));
    assert_eq!(
        h.tests().gap,
        None,
        "the gap test read no round past the gap round"
    );
    assert_eq!(calls.get(), 3, "the outgrown test read each count once");

    // One zero fewer: the zero test hands over, before the count.
    calls.set(0);
    assert!(after(&mut Handover::new(false), true, 10, 4, 0, &calls));
    assert_eq!(calls.get(), 0, "the zero test hands over before the count");
    // One divisible element more: the share test hands over.
    assert!(after(&mut Handover::new(false), true, 10, 5, 7, &calls));
    assert_eq!(calls.get(), 1);
    // An odd count of elements: 2·5 < 11 hands over, 2·6 ≥ 11 keeps.
    assert!(after(&mut Handover::new(false), true, 11, 5, 0, &calls));
    assert!(!after(&mut Handover::new(false), true, 11, 6, 0, &calls));
    // Exactly 6 in 10 keeps; past it hands over.
    assert!(!after(&mut Handover::new(false), true, 20, 10, 12, &calls));
    assert!(after(&mut Handover::new(false), true, 20, 10, 13, &calls));
    // A gap round with more zeros than elements keeps the run.
    assert!(!after(&mut Handover::new(false), true, 3, 9, 0, &calls));
}

/// Test 1, handover-time.5 and .7, and outgrown-round-size.4: a round
/// that made an element hands over where 8 in 10 or more of them are
/// divisible, before the gap round and after a gap round that kept the
/// run.
#[test]
fn an_outgrown_round_hands_over() {
    assert_eq!(EARLY_ROUND_ALONE, 100);
    let calls = Cell::new(0);
    // 99 elements with 80 divisible hand over, 79 keep.
    assert!(after(&mut Handover::new(false), false, 99, 0, 80, &calls));
    assert!(!after(&mut Handover::new(false), false, 99, 0, 79, &calls));
    assert_eq!(calls.get(), 2, "the outgrown test read each count once");
    // 100 with 80 divisible hands over, 79 keeps.
    assert!(after(&mut Handover::new(false), false, 100, 0, 80, &calls));
    assert!(!after(&mut Handover::new(false), false, 100, 0, 79, &calls));
    assert!(after(&mut Handover::new(false), false, 125, 0, 100, &calls));
    assert!(!after(&mut Handover::new(false), false, 125, 0, 99, &calls));
    // At a gap round that is also the early round, past the zero tests, the
    // early round's share test hands over first, and the count runs once.
    calls.set(0);
    assert!(after(&mut Handover::new(false), true, 100, 50, 80, &calls));
    assert_eq!(calls.get(), 1, "one count a round");
    // At a gap round that keeps the run, a round of the early round's
    // least or more reads the count once for the early round's, the gap round's, and the outgrown
    // tests.
    calls.set(0);
    let mut h = Handover::new(false);
    assert!(!after(&mut h, true, 100, 50, 60, &calls));
    assert_eq!(calls.get(), 1, "one count a round");
    // After it, an outgrown round hands over, and one short of it keeps.
    assert!(!after(&mut h, true, 100, 0, 79, &calls));
    assert!(!after(&mut h, false, 100, 0, 79, &calls));
    assert!(after(&mut h, false, 100, 0, 80, &calls));
    let mut h = Handover::new(false);
    assert!(!after(&mut h, true, 10, 5, 0, &calls));
    assert!(after(&mut h, true, 200, 0, 160, &calls));
}
