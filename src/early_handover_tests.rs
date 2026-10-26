//! early-handover-time, the crate's test code: the early round's decision
//! over the rounds of a signature run (plan.md, _Test code_, 1).

use std::cell::Cell;

use crate::f4::{EARLY, EARLY_ROUND_ALONE, Handover};

/// `after_round` with a `divisible` that returns `n` and counts its calls
/// in `calls`.
pub(crate) fn after(
    h: &mut Handover,
    early: bool,
    gap: bool,
    (made, zeros, n): (usize, usize, usize),
    calls: &Cell<u32>,
) -> bool {
    h.after_round(early, gap, made, zeros, || {
        calls.set(calls.get() + 1);
        n
    })
}

/// `after` on a new `Handover` whose plain run would reduce each row
/// alone, so that its early round's least is `EARLY_ROUND_ALONE`, at an
/// early round that is no gap round, the count's calls set to 0 first.
fn early_round(round: (usize, usize, usize), calls: &Cell<u32>) -> bool {
    calls.set(0);
    after(&mut Handover::new(false), true, false, round, calls)
}

/// Test 1, early-handover-time.1, .2, and .5 to .7, and
/// input-redundancy-time.7: the early round hands over where it made
/// its least elements or more, `EARLY_ROUND_ALONE` here
/// (outgrown-round-size.3), and reduced fewer rows to zero than a
/// quarter of its elements, none among them, or had more than 6 in 10 of
/// them divisible; it decides once, before the gap round's tests;
/// the count runs only where a test reads it, once a round.
#[test]
fn the_early_round_decides_once() {
    assert_eq!(EARLY, 2);
    assert_eq!(EARLY_ROUND_ALONE, 100);
    let calls = Cell::new(0);

    // outgrown-round-size.3: 99 elements are too few for the early tests;
    // the outgrown test reads the count and hands over
    // (outgrown-round-size.4).
    calls.set(0);
    let mut h = Handover::new(false);
    assert!(after(&mut h, true, false, (99, 1, 99), &calls));
    assert_eq!(
        (h.tests().early, h.tests().outgrown),
        (Some(false), Some(true))
    );
    assert_eq!(calls.get(), 1, "the outgrown test read the count");

    // input-redundancy-time.7, in place of .4: no zero, and the zero test
    // hands over before the count.
    assert!(early_round((100, 0, 61), &calls));
    assert_eq!(calls.get(), 0, "the zero test hands over before the count");
    assert!(early_round((100, 0, 80), &calls), "the zero test");

    // .5: fewer zeros than a quarter hand over before the count.
    assert!(early_round((100, 24, 100), &calls));
    assert_eq!(calls.get(), 0, "the zero test hands over before the count");
    // A quarter exactly, and exactly 6 in 10 divisible: the run goes on,
    // and the count is read once for the early and the outgrown tests.
    assert!(!early_round((100, 25, 60), &calls));
    assert_eq!(calls.get(), 1, "one count a round");
    // .6: one divisible element more hands over.
    assert!(early_round((100, 25, 61), &calls));
    assert_eq!(calls.get(), 1);

    // The early rounds of session 0054-1's counters: elements, zeros, and
    // divisible elements.
    for (stem, round) in [("biomd-103", (148, 15, 82)), ("biomd-365", (412, 567, 260))] {
        assert!(early_round(round, &calls), "{stem}: a handover");
    }
    for (stem, round) in [
        ("biomd-200", (97, 53, 25)),
        ("biomd-315", (191, 115, 82)),
        ("biomd-333", (290, 303, 88)),
        ("biomd-147", (472, 283, 270)),
    ] {
        assert!(!early_round(round, &calls), "{stem}: the run goes on");
    }

    // .2: the early round decides once. A later early round that the
    // tests would hand over keeps the run; the outgrown test reads the
    // count. The gap round after them decides by its own tests.
    let mut h = Handover::new(false);
    assert!(!after(&mut h, true, false, (100, 25, 60), &calls));
    calls.set(0);
    assert!(!after(&mut h, true, false, (200, 1, 0), &calls));
    assert_eq!(calls.get(), 1, "the outgrown test read the count once");
    assert!(
        after(&mut h, true, true, (10, 4, 0), &calls),
        "the gap round"
    );

    // An early round that is the gap round: the early tests first, then
    // the gap round's on the same counts, the count read once.
    calls.set(0);
    assert!(after(
        &mut Handover::new(false),
        true,
        true,
        (100, 30, 0),
        &calls
    ));
    assert_eq!(calls.get(), 1, "one count a round");
    calls.set(0);
    assert!(after(
        &mut Handover::new(false),
        true,
        true,
        (40, 10, 0),
        &calls
    ));
    assert_eq!(calls.get(), 0, "the gap round's zero test needs no count");
    // An early round that kept the run, and a gap round past it that
    // keeps it as well.
    let mut h = Handover::new(false);
    assert!(!after(&mut h, true, false, (100, 25, 60), &calls));
    assert!(!after(&mut h, true, true, (10, 5, 6), &calls));

    // A round of neither: the outgrown test of handover-time.5 alone.
    assert!(after(
        &mut Handover::new(false),
        false,
        false,
        (100, 0, 80),
        &calls
    ));
    assert!(!after(
        &mut Handover::new(false),
        false,
        false,
        (100, 0, 79),
        &calls
    ));
}
