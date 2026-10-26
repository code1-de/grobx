//! plain-update-time, the crate's test code: the pair set against
//! joux-2011's Algorithm 2 written out on exponent vectors, with
//! selections between the updates, a dropped pair left out of the
//! selection, the masks of an lcm, and the new pairs whatever their
//! degrees span (`iterations/0071-plain-update-time/plan.md`, _Test code_,
//! 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::divisibility_tests::{
    Engine, RefPair, Reference, below, degree, lcm, random_lanes, rng, written,
};
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::{Grevlex, Lex, Order};
use crate::pairs;
use crate::table::Table;

/// The reference's selection: the pairs of least sugar, in its order,
/// taken out of `pairs`, the rest left in its order.
fn select_reference(pairs: &mut Vec<RefPair>) -> Vec<RefPair> {
    let Some(d) = pairs.iter().map(|p| p.sugar).min() else {
        return Vec::new();
    };
    let (selected, rest): (Vec<RefPair>, Vec<RefPair>) =
        pairs.drain(..).partition(|p| p.sugar == d);
    *pairs = rest;
    selected
}

/// The engine's pair set against the reference: the pairs in order, their
/// count, and the emptiness.
fn assert_same<M: Format>(engine: &Engine<M>, reference: &Reference, case: &str) {
    assert_eq!(engine.written(), reference.pairs, "{case}: the pairs");
    assert_eq!(
        engine.pairs.len(),
        reference.pairs.len(),
        "{case}: the count"
    );
    assert_eq!(
        engine.pairs.is_empty(),
        reference.pairs.is_empty(),
        "{case}: the emptiness"
    );
    assert!(
        engine.pairs.held() >= engine.pairs.len(),
        "{case}: the set holds its live pairs"
    );
}

/// A selection on both sides: the same pairs in the same order, the same
/// pairs left, and no dropped pair held after it.
fn select_both<M: Format>(engine: &mut Engine<M>, reference: &mut Reference, case: &str) {
    let got = pairs::select(&mut engine.pairs);
    let want = select_reference(&mut reference.pairs);
    assert_eq!(written(&engine.table, &got), want, "{case}: the selection");
    assert_same(engine, reference, case);
    assert_eq!(
        engine.pairs.held(),
        engine.pairs.len(),
        "{case}: the selection leaves no dropped pair"
    );
}

/// Test 1 in one format and order: random leading monomials entered one
/// at a time into the engine and the reference, on `variables(world)`
/// variables in each of 40 worlds, a selection on both sides after about
/// a quarter of the updates.
fn check_with_selections<M: Format, O: Order<M>>(
    order: &O,
    seed: u64,
    variables: impl Fn(u64) -> usize,
) {
    let mut rng = rng(seed);
    for world in 0..40 {
        let n = variables(world);
        let mut engine: Engine<M> = Engine::new(n, seed ^ world);
        let mut reference = Reference::new(O::GRADED, M::MAX_DEGREE);
        let elements = 10 + below(&mut rng, 40) as usize;
        for e in 0..elements {
            // Small degrees on few variables: many coprime pairs, many
            // equal lcms, and many pairs the B criterion drops.
            let most = 1 + below(&mut rng, 5) as u32;
            let lead = random_lanes(&mut rng, n, 3, Some(most));
            let sugar = degree(&lead) + below(&mut rng, 4) as u32;
            engine.enter(order, &lead, sugar).0.unwrap();
            reference.update(lead.clone(), sugar).unwrap();
            let case = format!("world {world}, element {e}, lead {lead:?}");
            assert_same(&engine, &reference, &case);
            if below(&mut rng, 4) == 0 {
                select_both(&mut engine, &mut reference, &case);
            }
        }
        while !reference.pairs.is_empty() {
            select_both(
                &mut engine,
                &mut reference,
                &format!("world {world}, at the end"),
            );
        }
        assert!(engine.pairs.is_empty(), "world {world}: the set is empty");
    }
}

/// Test 1: plain-update-time.1 to .6, in each format, grevlex and lex, on
/// 2 to 6 variables, where a mask holds several bits a variable, and on 33
/// to 72, where a bit stands for several variables.
#[test]
fn the_pair_set_follows_algorithm_2() {
    let few = |world: u64| 2 + (world % 5) as usize;
    let many = |world: u64| 33 + (world % 40) as usize;
    check_with_selections::<Dense8, _>(&Grevlex, 0x9a17_0001, few);
    check_with_selections::<Dense8, _>(&Lex, 0x9a17_0002, many);
    check_with_selections::<Dense16, _>(&Grevlex, 0x9a17_0003, many);
    check_with_selections::<Dense16, _>(&Lex, 0x9a17_0004, few);
    check_with_selections::<Dense32, _>(&Grevlex, 0x9a17_0005, few);
    check_with_selections::<Dense32, _>(&Lex, 0x9a17_0006, many);
}

/// x·y and y·z, then y, with the sugars given, in `order`: the pair of
/// x·y and y·z, of lcm x·y·z, goes, and the selection returns the two new
/// pairs, of lcm x·y and y·z, and leaves the set empty.
fn dropped_then_selected<O: Order<Dense16>>(order: &O, sugars: [u32; 3], new_sugar: u32) {
    let mut e: Engine<Dense16> = Engine::new(3, 7);
    e.enter(order, &[1, 1, 0], sugars[0]).0.unwrap();
    e.enter(order, &[0, 1, 1], sugars[1]).0.unwrap();
    assert_eq!(e.pairs.len(), 1, "the pair of x·y and y·z");
    e.enter(order, &[0, 1, 0], sugars[2]).0.unwrap();
    assert_eq!(e.pairs.len(), 2, "the pair of lcm x·y·z goes");
    assert_eq!(e.pairs.held(), 3, "the dropped pair stays held");
    let want = vec![
        RefPair {
            i: 0,
            j: 2,
            lcm: vec![1, 1, 0],
            degree: 2,
            sugar: new_sugar,
        },
        RefPair {
            i: 1,
            j: 2,
            lcm: vec![0, 1, 1],
            degree: 2,
            sugar: new_sugar,
        },
    ];
    assert_eq!(e.written(), want, "the two new pairs");
    let got = pairs::select(&mut e.pairs);
    assert_eq!(written(&e.table, &got), want, "the selection");
    assert!(e.pairs.is_empty(), "the set is empty");
    assert_eq!(e.pairs.held(), 0, "the dropped pair is gone");
    assert!(
        pairs::select(&mut e.pairs).is_empty(),
        "an empty set selects nothing"
    );
}

/// Test 2: plain-update-time.3, by hand on x, y, z. In lex with sugars 2,
/// 2, and 10 the dropped pair's sugar, 3, is the least of the set, and the
/// new pairs' is 11.
#[test]
fn a_dropped_pair_is_left_out() {
    dropped_then_selected(&Grevlex, [2, 2, 1], 2);
    dropped_then_selected(&Lex, [2, 2, 10], 11);
}

/// Test 3 in one format: random exponents within the format on each of
/// `ns` variables, 200 pairs each.
fn check_lcm_masks<M: Format>(seed: u64, ns: &[usize]) {
    let mut rng = rng(seed);
    for &n in ns {
        let format = M::new(n);
        let mut table = Table::new(format, &mut rng);
        let half = M::MAX_DEGREE / 2;
        let w = format.words();
        let (mut wa, mut wb, mut wl) = (
            vec![M::Word::default(); w],
            vec![M::Word::default(); w],
            vec![M::Word::default(); w],
        );
        for _ in 0..200 {
            let a = random_lanes(&mut rng, n, half, Some(half));
            let b = random_lanes(&mut rng, n, half, Some(half));
            let l = lcm(&a, &b);
            format.encode(&a, &mut wa).unwrap();
            format.encode(&b, &mut wb).unwrap();
            format.lcm(&wa, &wb, &mut wl).unwrap();
            assert_eq!(
                format.mask(&wl),
                format.mask(&wa) | format.mask(&wb),
                "n {n}: the words' masks of {a:?} and {b:?}"
            );
            let (ia, ib, il) = (
                table.intern(&a).unwrap(),
                table.intern(&b).unwrap(),
                table.intern(&l).unwrap(),
            );
            assert_eq!(
                table.mask(il),
                table.mask(ia) | table.mask(ib),
                "n {n}: the table's masks of {a:?} and {b:?}"
            );
        }
    }
}

/// Test 3: plain-update-time.4, in each format on 1 to 140 variables.
#[test]
fn lcm_masks_are_the_masks_ored() {
    let ns = [
        1, 2, 3, 4, 5, 7, 8, 10, 11, 16, 17, 31, 32, 33, 40, 63, 64, 65, 100, 140,
    ];
    check_lcm_masks::<Dense8>(0x9a17_0011, &ns);
    check_lcm_masks::<Dense16>(0x9a17_0012, &ns);
    check_lcm_masks::<Dense32>(0x9a17_0013, &ns);
}

/// The leading monomials `leads`, each of sugar its degree, entered one
/// at a time into the engine and the reference in grevlex: the same pairs
/// after each.
fn agree_by_hand<M: Format>(case: &str, leads: &[Vec<u32>]) {
    let n = leads[0].len();
    let mut engine: Engine<M> = Engine::new(n, 11);
    let mut reference = Reference::new(true, M::MAX_DEGREE);
    for (k, lead) in leads.iter().enumerate() {
        engine.enter(&Grevlex, lead, degree(lead)).0.unwrap();
        reference.update(lead.clone(), degree(lead)).unwrap();
        assert_same(&engine, &reference, &format!("{case}, element {k}"));
    }
}

/// Test 4: plain-update-time.5, .6, by hand: the last element's new pairs
/// of one degree; of degrees spanning as many values as the pairs, and
/// one more; of equal lcms, the later pair standing; and two pairs whose
/// degrees span 2^30 − 1 values in `Dense32`.
#[test]
fn new_pairs_whatever_their_degrees_span() {
    // x1², x2², x3², then x1·x2·x3·x4: three lcms of degree 5.
    let one = [
        vec![2, 0, 0, 0],
        vec![0, 2, 0, 0],
        vec![0, 0, 2, 0],
        vec![1, 1, 1, 1],
    ];
    agree_by_hand::<Dense16>("one degree", &one);
    // x1², x2³, x3⁴, then x1·x2·x3: lcms of degrees 4, 5, and 6, three
    // values for three pairs.
    let as_many = [vec![2, 0, 0], vec![0, 3, 0], vec![0, 0, 4], vec![1, 1, 1]];
    agree_by_hand::<Dense16>("as many values as pairs", &as_many);
    // x1², x2⁴, then x1·x2·x3: lcms of degrees 4 and 6, three values for
    // two pairs.
    let one_more = [vec![2, 0, 0], vec![0, 4, 0], vec![1, 1, 1]];
    agree_by_hand::<Dense16>("one value more than pairs", &one_more);
    // x²y, x²z, then yz: two lcms x²yz, and the later pair stands.
    let equal = [vec![2, 1, 0], vec![2, 0, 1], vec![0, 1, 1]];
    agree_by_hand::<Dense8>("equal lcms", &equal);
    // x^(2^30), y², then x·y·z: lcms of degrees 2^30 + 2 and 4.
    let wide = [vec![1 << 30, 0, 0], vec![0, 2, 0], vec![1, 1, 1]];
    agree_by_hand::<Dense32>("a span of 2^30 − 1", &wide);
}
