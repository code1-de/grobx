//! gf2-split-rest-time, the item's test code: where the repeat of a split
//! holds, the selection splits the split sugar past `SPLIT_TAKE` live
//! pairs, the first split keeping its least `SPLIT_LEAST`, and calls over
//! GF(2) under lowered splits that reach the repeat below the first
//! split's least return the basis the exact checker certifies
//! (`iterations/0156-gf2-split-rest-time/plan.md`, _Test code_, 1 to 4).

use crate::gf2_bound_losses_tests::{certified, systems};
use crate::pairs::{SPLIT_LEAST, SPLIT_TAKE};
use crate::split_piece_time_tests::{D, World, monomials, outcome, reported, step};
use crate::systems::cubics;
use crate::{CheckerMode, MonomialOrder};

/// The lowered splits of test 4, the most pairs a selection takes whole
/// and the pairs a split takes, each with room for a repeat between them.
const LOWERED: [(usize, usize); 4] = [(8, 2), (16, 2), (16, 4), (64, 8)];

/// A set whose sugar 5 holds `3 · SPLIT_LEAST` pairs, `above` pairs of
/// sugar 6 beside them, after the selection that splits sugar 5 and a
/// note of a piece of `made` elements; of the pairs of 5 it leaves, all
/// but `rest` dropped.
fn rest_world(rest: usize, above: usize, made: usize, seed: u64) -> World {
    let lcms = monomials(D);
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    assert!(
        rest <= 3 * l - t,
        "a rest of at most the pairs the split leaves"
    );
    let mut w = World::new(&[], seed);
    w.push_all(&lcms, 3 * l, 5);
    w.push_all(&lcms, above, 6);
    step(&mut w, "the first split of 5", 5, t, 3 * l - t);
    // The pairs of 5 the split left come first in the set's order, at
    // places the split's holes leave scattered (pair-index-upkeep-time.1).
    for &k in &w.set.places()[..3 * l - t - rest] {
        w.set.drop_at(k);
    }
    w.set.note_piece(t, made, 0);
    w
}

/// Test 1: gf2-split-rest-time.1 and .2. After a split of sugar 5, under
/// notes of pieces of no element, every selection of 5 takes `SPLIT_TAKE`
/// pairs while the sugar holds more, within `SPLIT_LEAST` as past it, and
/// the selection that follows takes the last `SPLIT_TAKE` whole.
#[test]
fn a_rest_after_few_elements_goes_in_pieces() {
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let mut w = rest_world(3 * l - t, 0, 0, 0x156a);
    let mut left = 3 * l - t;
    let mut within = 0;
    while left > t {
        within += usize::from(left <= l);
        step(&mut w, &format!("{left} pairs left"), 5, t, left - t);
        left -= t;
        w.set.note_piece(t, 0, 0);
    }
    assert!(within > 1, "pieces within the first split's least");
    step(&mut w, "the last pieces' pairs", 5, left, 0);
    assert!(w.set.is_empty());
}

/// Test 2: gf2-split-rest-time.1 and .2. After a split of sugar 5 and a
/// note of a piece of no element, a rest of `SPLIT_TAKE` + 1, of
/// `SPLIT_LEAST`, and of `SPLIT_LEAST` + 1 pairs splits again, and one of
/// `SPLIT_TAKE` is taken whole. After a note of half the piece's pairs a
/// rest of `SPLIT_LEAST` is taken whole, and so is one beside a pair more
/// of sugar 6 than it holds itself.
#[test]
fn the_repeat_takes_the_piece_as_its_least() {
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    for (k, rest) in [t + 1, l, l + 1].into_iter().enumerate() {
        let mut w = rest_world(rest, 0, 0, 0x156b + k as u64);
        step(&mut w, &format!("a rest of {rest}"), 5, t, rest - t);
    }
    let mut w = rest_world(t, 0, 0, 0x156e);
    step(&mut w, "a rest of the piece's size", 5, t, 0);
    let mut w = rest_world(l, 0, t / 2, 0x156f);
    step(&mut w, "a rest after a piece of half its pairs", 5, l, 0);
    let mut w = rest_world(l, l, 0, 0x1570);
    step(&mut w, "a rest of half the set", 5, t, l - t);
    let mut w = rest_world(l, l + 1, 0, 0x1571);
    step(&mut w, "a rest of less than half the set", 5, l, 0);
}

/// Test 3: gf2-split-rest-time.1. The first split keeps its least: a sugar
/// of `SPLIT_LEAST` pairs that no selection split is taken whole, after
/// another sugar's split and its note too. A lower sugar taken whole
/// between two pieces keeps the note, and the rest within `SPLIT_LEAST`
/// splits again after it.
#[test]
fn the_first_split_keeps_its_least() {
    let lcms = monomials(D);
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let mut w = World::new(&[], 0x1572);
    w.push_all(&lcms, l, 5);
    step(&mut w, "a sugar of the least, unsplit", 5, l, 0);
    let mut w = rest_world(0, 0, 0, 0x1573);
    w.push_all(&lcms, l, 6);
    step(&mut w, "6 of the least after a split of 5", 6, l, 0);
    let mut w = rest_world(l, 0, 0, 0x1574);
    w.push_all(&lcms, l, 4);
    step(&mut w, "4 between the pieces", 4, l, 0);
    step(&mut w, "5 within the least after 4", 5, t, l - t);
}

/// Test 4: gf2-split-rest-time.1 and .2 in calls. Over GF(2), in grevlex,
/// the boolean systems and the edge cases of `gf2_bound_losses_tests` and
/// cubic systems on 8 to 10 variables, with the signatures off and on, and
/// with cofactors and the exact checker: every call under each split of
/// `LOWERED` returns the basis the exact checker certifies, and with the
/// checker on it is certified.
#[test]
fn lowered_rests_over_gf2_leave_every_basis() {
    let mut texts = systems(true);
    texts.extend((8..=10usize).map(|n| {
        let m = 2 * n;
        (format!("cubics({n}, {m})"), cubics(n, m, 0x1575 + n as u64))
    }));
    for (i, (name, text)) in texts.iter().enumerate() {
        let seed = 0x1576 ^ i as u64;
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
