//! gf2-wide-round-time, the item's test code: the selection splits the
//! last split's sugar again where the note of its last piece has the piece
//! make elements from fewer than half its pairs and the sugar holds at
//! least half the set's live pairs, and takes it whole otherwise; a note
//! belongs to its split; calls over GF(2) under lowered splits return the
//! basis the exact checker certifies; and GF(2) alone splits again
//! (`iterations/0154-gf2-wide-round-time/plan.md`, _Test code_, 1 to 5).

use crate::field::{Field, Gf2, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::gf2_bound_losses_tests::{certified, systems};
use crate::pairs::{SPLIT_LEAST, SPLIT_TAKE};
use crate::split_piece_time_tests::{D, World, monomials, outcome, reported, step};
use crate::systems::cubics;
use crate::{CheckerMode, MonomialOrder};

/// The lowered splits of test 4: the most pairs a selection takes whole,
/// and the pairs a split takes.
const LOWERED: [(usize, usize); 4] = [(1, 1), (2, 1), (3, 2), (8, 4)];

/// A set whose sugar 5 holds `3 · SPLIT_LEAST` pairs, `above` pairs of
/// sugar 6 beside them, after the selection that splits sugar 5.
fn split_world(above: usize, seed: u64) -> World {
    let lcms = monomials(D);
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let mut w = World::new(&[], seed);
    w.push_all(&lcms, 3 * l, 5);
    w.push_all(&lcms, above, 6);
    step(&mut w, "the first split of 5", 5, t, 3 * l - t);
    w
}

/// Under notes of pieces of no element, the selections of sugar 5 in
/// `w`, `left` of its pairs left after a split, each a piece while the
/// sugar holds more than `SPLIT_TAKE` pairs (gf2-split-rest-time.1); the
/// pairs left then.
fn split_down(w: &mut World, mut left: usize) -> usize {
    let t = SPLIT_TAKE;
    while left > t {
        w.set.note_piece(t, 0, 0);
        step(w, &format!("{left} pairs left"), 5, t, left - t);
        left -= t;
    }
    left
}

/// Test 1: gf2-wide-round-time.1 and .4. After a split of sugar 5, a note
/// of its piece's pairs and of fewer elements than half of them has the
/// next selection of 5 split it again, and one of half or more has it take
/// the rest whole. Under notes of no element the sugar splits again at
/// every selection while it holds more than `SPLIT_TAKE` pairs
/// (gf2-split-rest-time.1), and the selection that follows takes the rest
/// whole.
#[test]
fn a_piece_of_few_elements_splits_its_sugar_again() {
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    for (made, again) in [(0, true), (t / 2 - 1, true), (t / 2, false), (t, false)] {
        let mut w = split_world(0, 0x154a ^ made as u64);
        w.set.note_piece(t, made, 0);
        let case = format!("a piece of {made} elements of {t} pairs");
        if again {
            step(&mut w, &case, 5, t, 3 * l - 2 * t);
        } else {
            step(&mut w, &case, 5, 3 * l - t, 0);
        }
    }
    let mut w = split_world(0, 0x154b);
    let left = split_down(&mut w, 3 * l - t);
    w.set.note_piece(t, 0, 0);
    step(&mut w, "the rest within the bound", 5, left, 0);
    assert!(w.set.is_empty());
}

/// Test 2: gf2-wide-round-time.2. After a split of sugar 5 and a note of a
/// piece of no element, the next selection of 5 splits it again where its
/// pairs are at least half the set's live pairs, as many of sugar 6 beside
/// them as of 5, and takes it whole where one pair of 6 more stands beside
/// them.
#[test]
fn a_sugar_splits_again_where_it_holds_half_the_set() {
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let left = 3 * l - t;
    let mut w = split_world(left, 0x154c);
    w.set.note_piece(t, 0, 0);
    step(&mut w, "half the set", 5, t, left - t);
    let mut w = split_world(left + 1, 0x154d);
    w.set.note_piece(t, 0, 0);
    step(&mut w, "less than half the set", 5, left, 0);
}

/// Test 3: gf2-wide-round-time.1. A note belongs to its split. A lower
/// sugar taken whole between two pieces of sugar 5 keeps the note of 5.
/// A split of a lower sugar 4 in between takes the split's place, and the
/// next selection of 4, which holds half the set's live pairs, reads no
/// note of a piece of 5 and takes 4 whole; so does a split of a higher
/// sugar 6 after the last piece of 5.
#[test]
fn a_note_belongs_to_its_split() {
    let lcms = monomials(D);
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let mut w = split_world(0, 0x154e);
    w.set.note_piece(t, 0, 0);
    w.push_all(&lcms, l + 1, 4);
    step(&mut w, "4 between the pieces", 4, l + 1, 0);
    step(&mut w, "5 after 4", 5, t, 3 * l - 2 * t);
    let mut w = split_world(0, 0x154f);
    w.set.note_piece(t, 0, 0);
    w.push_all(&lcms, l + 1, 4);
    step(&mut w, "4 right after the split", 4, l + 1, 0);
    w.push_all(&lcms, 3 * l, 4);
    step(&mut w, "a split of 4", 4, t, 3 * l - t);
    step(&mut w, "the rest of 4, no note of its own", 4, 3 * l - t, 0);
    let mut w = split_world(0, 0x1550);
    let left = split_down(&mut w, 3 * l - t);
    w.set.note_piece(t, 0, 0);
    step(&mut w, "the rest of 5", 5, left, 0);
    w.push_all(&lcms, 3 * l, 6);
    step(&mut w, "a split of 6", 6, t, 3 * l - t);
    step(&mut w, "the rest of 6, no note of its own", 6, 3 * l - t, 0);
}

/// Test 4: gf2-wide-round-time.1 to .3 in calls. Over GF(2), in grevlex,
/// the boolean systems and the edge cases of `gf2_bound_losses_tests` and
/// cubic systems on 8 to 10 variables, with the signatures off and on, and
/// with cofactors and the exact checker: every call under each split of
/// `LOWERED` returns the basis the exact checker certifies, and with the
/// checker on it is certified.
#[test]
fn lowered_splits_over_gf2_leave_every_basis() {
    let mut texts = systems(true);
    texts.extend((8..=10usize).map(|n| {
        let m = 2 * n;
        (format!("cubics({n}, {m})"), cubics(n, m, 0x1551 + n as u64))
    }));
    for (i, (name, text)) in texts.iter().enumerate() {
        let seed = 0x1552 ^ i as u64;
        let want = certified(name, text, seed);
        for bounds in LOWERED {
            for signatures in [false, true] {
                let mut params = reported(MonomialOrder::Grevlex, seed);
                params.signatures = signatures;
                let got = outcome(text, &params, Some(bounds))
                    .unwrap_or_else(|| panic!("{name}, {bounds:?}: the call failed"));
                assert_eq!(
                    got.basis.as_deref(),
                    Some(want.as_str()),
                    "{name}, {bounds:?}, signatures {signatures}"
                );
            }
            let mut params = reported(MonomialOrder::Grevlex, seed);
            params.cofactors = true;
            params.checker = Some(CheckerMode::Exact);
            let got = outcome(text, &params, Some(bounds))
                .unwrap_or_else(|| panic!("{name}, {bounds:?}, cofactors: the call failed"));
            assert_eq!(got.certified, Some(true), "{name}, {bounds:?}: certified");
            assert_eq!(
                got.basis.as_deref(),
                Some(want.as_str()),
                "{name}, {bounds:?}, cofactors"
            );
        }
    }
}

/// Test 5: gf2-wide-round-time.3. Of the fields, `Gf2` alone splits a
/// sugar again.
#[test]
fn gf2_alone_splits_again() {
    let fields = [
        ("Gf2", <Gf2 as Field>::SPLITS_AGAIN),
        ("Gf31", <Gf31 as Field>::SPLITS_AGAIN),
        ("Gf64", <Gf64 as Field>::SPLITS_AGAIN),
        ("Gf31Pair", <Gf31Pair as Field>::SPLITS_AGAIN),
        ("Gf64Pair", <Gf64Pair as Field>::SPLITS_AGAIN),
    ];
    let again: Vec<&str> = fields
        .iter()
        .filter(|&&(_, splits)| splits)
        .map(|&(name, _)| name)
        .collect();
    assert_eq!(again, ["Gf2"]);
}
