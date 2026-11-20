//! pair-index-upkeep-time, the item's test code: the pair set with holes
//! and `G_min` with tombstones keep the pairs of the reference's Algorithm 2
//! and take its selections, and keep and take what the set and `G_min`
//! without them do; the members that leave `G_min` by hand; the index kept
//! by `Bitmap::unset` against a rebuild; a call returns what it returns
//! without them; and stops in the selection and the update
//! (`iterations/0171-pair-index-upkeep-time/plan.md`, _Test code_, 1 to
//! 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::bitmap::Bitmap;
use crate::budget::{Budget, Site};
use crate::call::built;
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::ccode_newton_gap_tests::call;
use crate::divisibility_tests::{Engine, RefPair, Reference, degree, exponents};
use crate::gf2_chain_criteria_time_tests::{lead_for, mono};
use crate::gf2_update_rest_time_tests::{pairs_of, spread, systems_on};
use crate::linear_substitution_time_tests::untimed;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::order::{Grevlex, Lex, Order};
use crate::pairs::{self, HOLES, INDEX_FORCED, Pair, SPLIT_BOUNDS, TOMBS};
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::{below, rng};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{boolean, placed};
use crate::{MonomialOrder, Outcome, Params};

/// The item's two switches set to `holes` and `tombs`, and back to their
/// default, both on, after `f`.
fn with<T>(holes: bool, tombs: bool, f: impl FnOnce() -> T) -> T {
    HOLES.set(holes);
    TOMBS.set(tombs);
    let got = f();
    HOLES.set(true);
    TOMBS.set(true);
    got
}

/// The holes of `e`'s pair set: its places less its live pairs.
fn holes<M: Format>(e: &Engine<M>) -> usize {
    e.pairs.held() - e.pairs.len()
}

/// The selection of `e` under `bounds`, the split's least and take, or
/// under bounds that split no sugar where None.
fn select<M: Format, O: Order<M>>(
    e: &mut Engine<M>,
    order: &O,
    bounds: Option<(usize, usize)>,
) -> Vec<Pair> {
    SPLIT_BOUNDS.set(Some(bounds.unwrap_or((usize::MAX, 1))));
    let got = pairs::select(&mut e.pairs, &mut e.table, order, &mut Budget::unbounded());
    SPLIT_BOUNDS.set(None);
    got.unwrap_or_else(|limit| panic!("the selection: {limit:?}"))
}

/// The pairs a selection of `e` took, as the reference writes them, their
/// lcms read from the table that interned them.
fn written_taken<M: Format>(e: &Engine<M>, taken: &[Pair]) -> Vec<RefPair> {
    taken
        .iter()
        .map(|p| RefPair {
            i: p.i.index() as u32,
            j: p.j.index() as u32,
            lcm: exponents(&e.table, p.lcm),
            degree: p.degree,
            sugar: p.sugar,
        })
        .collect()
}

/// The reference's selection, which splits no sugar: its pairs of the
/// least sugar, in its order, taken out of its set.
fn reference_select(reference: &mut Reference) -> Vec<RefPair> {
    let Some(d) = reference.pairs.iter().map(|p| p.sugar).min() else {
        return Vec::new();
    };
    let (taken, left): (Vec<RefPair>, Vec<RefPair>) =
        reference.pairs.drain(..).partition(|p| p.sugar == d);
    reference.pairs = left;
    taken
}

/// `G_min` of `e` as indices.
fn members<M: Format>(e: &Engine<M>) -> Vec<u32> {
    e.min_members().iter().map(|i| i.index() as u32).collect()
}

/// Test 1's run in format M on n variables: `steps` leads of degree at most
/// `top`, exponents at most `most` (`lead_for`), the constant tenth from
/// the last, entered into an engine with the item's switches on, one with
/// them off, and the reference, with a selection after every ninth step
/// that splits no sugar: after every step the pair set is the reference's
/// pairs in their order, the two engines' sets, `G_min`s, and arrays
/// agree, and the selections take the reference's pairs in its order.
/// `forced` fixes the indexes (`INDEX_FORCED`). The steps after which the
/// engine with the switches held fewer holes than before, and those that
/// made members of `G_min` redundant.
#[allow(
    clippy::too_many_arguments,
    reason = "the order, the variables, the seed, the steps, the degree and exponent bounds, and the index's choice, each a dimension of the comparison"
)]
fn alike<M: Format, O: Order<M>>(
    order: &O,
    n: usize,
    seed: u64,
    steps: usize,
    top: u32,
    most: u32,
    forced: Option<bool>,
) -> (usize, usize) {
    let mut on: Engine<M> = Engine::new(n, seed);
    let mut off: Engine<M> = Engine::new(n, seed);
    let mut reference = Reference::new(O::GRADED, M::MAX_DEGREE);
    let mut r = rng(seed ^ 0x401e);
    let mut entered: Vec<Vec<u32>> = Vec::new();
    let (mut filled, mut left) = (0, 0);
    let hot = n.min(12);
    for step in 0..steps {
        let lead = if step + 10 == steps {
            vec![0u32; n]
        } else {
            lead_for(&mut r, n, hot, top, most, &entered)
        };
        if degree(&lead) > M::MAX_DEGREE / 2 {
            continue;
        }
        let sugar = degree(&lead) + below(&mut r, 2) as u32;
        let case = format!("{n} variables, seed {seed:#x}, step {step}: {lead:?}");
        let before = (holes(&on), reference.min_members().to_vec());
        INDEX_FORCED.set(forced);
        let a = with(true, true, || on.enter(order, &lead, sugar).0);
        let b = with(false, false, || off.enter(order, &lead, sugar).0);
        INDEX_FORCED.set(None);
        reference
            .update(lead.clone(), sugar)
            .unwrap_or_else(|e| panic!("{case}: the reference: {e:?}"));
        assert_eq!(a, Ok(()), "{case}: the update");
        assert_eq!(b, Ok(()), "{case}: the update without the switches");
        if holes(&on) < before.0 {
            filled += 1;
        }
        if before
            .1
            .iter()
            .any(|m| !reference.min_members().contains(m))
        {
            left += 1;
        }
        assert_eq!(
            on.written(),
            reference.pairs,
            "{case}: the reference's pairs"
        );
        assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the pair sets");
        assert_eq!(on.min_arrays(), off.min_arrays(), "{case}: G_min's arrays");
        assert_eq!(members(&on), reference.min_members(), "{case}: G_min");
        entered.push(lead);
        if step % 9 == 8 {
            let x = with(true, true, || select(&mut on, order, None));
            let y = with(false, false, || select(&mut off, order, None));
            let want = reference_select(&mut reference);
            assert_eq!(written_taken(&on, &x), want, "{case}: the selection");
            assert_eq!(
                written_taken(&off, &y),
                want,
                "{case}: the selection without the switches"
            );
            assert_eq!(on.written(), reference.pairs, "{case}: the pairs left");
            assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the sets left");
        }
    }
    (filled, left)
}

/// Test 1: pair-index-upkeep-time.1 to .6. The update and the selection
/// with holes and tombstones keep the pairs of the reference's Algorithm 2
/// and take its selections, and keep and take what they keep and take
/// without them, in every regime of the index: up to 32 variables without
/// it and forced on, 33 to 128 with it, and past 128 with folded supports;
/// in `Dense4`, `Dense8`, `Dense16`, and `Dense32`, in grevlex and lex; on
/// leads of distinct variables beside squares and on exponents to 3. A
/// constant among the leads leaves `G_min` the constant alone. Each run of
/// `Dense8` fills holes and makes members of `G_min` redundant.
#[test]
fn the_set_keeps_the_reference_pairs() {
    for n in [6, 16, 20, 32, 33, 60, 128, 129, 300] {
        for (most, top) in [(1, 4), (3, 3)] {
            for forced in [None, Some(n <= 32)] {
                let seed = 0x171 + n as u64 + u64::from(most) * 0x100;
                let (filled, left) = alike::<Dense8, _>(&Grevlex, n, seed, 220, top, most, forced);
                let case = format!("Dense8, {n} variables, most {most}, forced {forced:?}");
                assert!(filled > 0, "{case}: no hole filled");
                assert!(left > 0, "{case}: no member left G_min");
            }
        }
    }
    for n in [12, 20, 40, 130] {
        alike::<Dense4, _>(&Grevlex, n, 0x4f + n as u64, 200, 4, 1, None);
        alike::<Dense16, _>(&Grevlex, n, 0x16f + n as u64, 160, 3, 3, None);
        alike::<Dense32, _>(&Grevlex, n, 0x32f + n as u64, 120, 3, 2, None);
        alike::<Dense8, _>(&Lex, n, 0x1f + n as u64, 160, 3, 2, None);
    }
}

/// Test 2's run in format M on n variables: `steps` drawn leads entered
/// into an engine with the item's switches on and one with them off, a
/// selection after every eighth under the split bounds `bounds`, and the
/// sets cleared halfway: the updates' results, the sets in their order,
/// `G_min`'s arrays, the selections, and `deferred` agree, and with the
/// switches on the holes number at most the live pairs after each
/// selection, and the set grows at an update only where it then holds no
/// hole. The selections that split their sugar.
fn against<M: Format, O: Order<M>>(
    order: &O,
    n: usize,
    seed: u64,
    steps: usize,
    bounds: (usize, usize),
) -> usize {
    let mut on: Engine<M> = Engine::new(n, seed);
    let mut off: Engine<M> = Engine::new(n, seed);
    let mut r = rng(seed ^ 0xa9a1);
    let mut entered: Vec<Vec<u32>> = Vec::new();
    let mut split = 0;
    let hot = n.min(10);
    for step in 0..steps {
        let most = 1 + below(&mut r, 2) as u32;
        let lead = lead_for(&mut r, n, hot, 4, most, &entered);
        if degree(&lead) > M::MAX_DEGREE / 2 {
            continue;
        }
        let sugar = degree(&lead) + below(&mut r, 2) as u32;
        let case = format!("{n} variables, seed {seed:#x}, step {step}: {lead:?}");
        let held = on.pairs.held();
        let a = with(true, true, || on.enter(order, &lead, sugar).0);
        let b = with(false, false, || off.enter(order, &lead, sugar).0);
        assert_eq!(a, b, "{case}: the update");
        assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the pair sets");
        assert_eq!(on.min_arrays(), off.min_arrays(), "{case}: G_min's arrays");
        assert!(
            on.pairs.held() <= held || holes(&on) == 0,
            "{case}: the set grew from {held} to {now} places with {h} holes",
            now = on.pairs.held(),
            h = holes(&on)
        );
        entered.push(lead);
        if step % 8 == 7 {
            let x = with(true, true, || select(&mut on, order, Some(bounds)));
            let y = with(false, false, || select(&mut off, order, Some(bounds)));
            assert_eq!(x, y, "{case}: the selection");
            assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the sets left");
            assert_eq!(
                on.pairs.deferred(),
                off.pairs.deferred(),
                "{case}: deferred"
            );
            assert!(
                holes(&on) <= on.pairs.len(),
                "{case}: {h} holes beside {live} live pairs after the selection",
                h = holes(&on),
                live = on.pairs.len()
            );
            if on.pairs.deferred() > 0 {
                split += 1;
            }
        }
        if step == steps / 2 {
            on.pairs.clear();
            off.pairs.clear();
        }
    }
    split
}

/// Test 2: pair-index-upkeep-time.1 to .3, .8. With holes and tombstones
/// the update and the selection leave the set, `G_min`'s arrays, the pairs
/// taken, and `deferred` as they leave them without, under split bounds
/// that split the selections, the sets cleared halfway, on 20 to 140
/// variables in `Dense4` and `Dense8`, in grevlex and lex. Each run of
/// `Dense8` in grevlex splits a selection. After a selection the holes
/// number at most the live pairs, and an update grows the set only where
/// it fills every hole.
#[test]
fn holes_and_tombs_against_the_set_without_them() {
    for n in [20, 33, 60, 129, 140] {
        for bounds in [(4, 2), (16, 4)] {
            let seed = 0x2a + n as u64 + bounds.0 as u64;
            let s = against::<Dense8, _>(&Grevlex, n, seed, 600, bounds);
            assert!(s > 0, "Dense8, {n} variables, bounds {bounds:?}: no split");
            against::<Dense4, _>(&Grevlex, n, seed ^ 0x4, 300, bounds);
            against::<Dense8, _>(&Lex, n, seed ^ 0x1e, 300, bounds);
        }
    }
}

/// Test 3 on n variables, the index as `forced`: the leads of the plan's
/// test 3 entered one at a time into the engine and the reference, each at
/// its degree's sugar: after each the pairs of the set are the reference's,
/// `G_min` is the reference's members, and the arrays' live members are
/// `G_min`. x0·x1·x8 and x2·x9 keep their one pair, with x0·x1 and with
/// x2, and x3 after the constant keeps none.
fn tombs_on<M: Format>(n: usize, forced: Option<bool>) {
    let leads = [
        mono(n, &[(0, 1), (1, 1), (2, 1)]),
        mono(n, &[(0, 1), (1, 1), (3, 1)]),
        mono(n, &[(4, 1), (5, 1)]),
        mono(n, &[(2, 1), (6, 1)]),
        mono(n, &[(7, 1)]),
        mono(n, &[(0, 1), (1, 1)]),
        mono(n, &[(0, 1), (1, 1), (8, 1)]),
        mono(n, &[(4, 1)]),
        mono(n, &[(2, 1)]),
        mono(n, &[(2, 1), (9, 1)]),
        vec![0u32; n],
        mono(n, &[(3, 1)]),
    ];
    let mut e: Engine<M> = Engine::new(n, 0x70b5);
    let mut reference = Reference::new(true, M::MAX_DEGREE);
    let mut news: Vec<Vec<(u32, u32)>> = Vec::new();
    for (t, lead) in leads.iter().enumerate() {
        let case = format!("{n} variables, forced {forced:?}, lead {t}: {lead:?}");
        INDEX_FORCED.set(forced);
        let got = with(true, true, || e.enter(&Grevlex, lead, degree(lead)).0);
        INDEX_FORCED.set(None);
        assert_eq!(got, Ok(()), "{case}: the update");
        reference.update(lead.clone(), degree(lead)).unwrap();
        assert_eq!(e.written(), reference.pairs, "{case}: the pairs");
        assert_eq!(members(&e), reference.min_members(), "{case}: G_min");
        assert_eq!(
            e.min_arrays().0,
            e.min_members(),
            "{case}: the arrays' live members"
        );
        news.push(
            e.written()
                .into_iter()
                .filter(|p| p.j == t as u32)
                .map(|p| (p.i, p.j))
                .collect(),
        );
    }
    assert_eq!(news[6], [(5, 6)], "{n} variables: x0·x1·x8's one pair");
    assert_eq!(news[9], [(8, 9)], "{n} variables: x2·x9's one pair");
    assert!(news[11].is_empty(), "{n} variables: x3 after the constant");
    assert_eq!(members(&e), [10], "{n} variables: G_min the constant");
}

/// Test 3: pair-index-upkeep-time.4 to .6. Members that leave `G_min` by
/// hand: x0·x1 makes two members redundant, which stay as tombstones; x4
/// brings the tombstones to the live members, and the arrays are rebuilt;
/// x2 makes a member redundant after the rebuild; the constant makes every
/// member redundant. The divided elements keep their pair with a live
/// member. On 40, 70, and 140 variables, the last past `SUPPORT_BITS`, and
/// on 10 with the index forced on.
#[test]
fn members_that_leave_stay_as_tombstones() {
    tombs_on::<Dense8>(40, None);
    tombs_on::<Dense4>(70, None);
    tombs_on::<Dense8>(140, None);
    tombs_on::<Dense8>(10, Some(true));
}

/// Test 4: pair-index-upkeep-time.1. An index kept by `Bitmap::set` and
/// `Bitmap::unset` at random places, a place set again after its unset
/// among them, on 1 to 128 bits over 1 to 400 places: `holding`,
/// `meeting`, and `meeting_last` return the places an index rebuilt over
/// the places set and not unset returns, and those the supports name.
#[test]
fn the_index_kept_against_a_rebuild() {
    let mut r = rng(0x4b17);
    let budget = &mut Budget::unbounded();
    for bits in [1usize, 2, 7, 64, 100, 128] {
        let top = if bits == 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        for places in [1usize, 63, 64, 65, 200, 400] {
            let mut kept = Bitmap::new(bits);
            let mut supports: Vec<Option<u128>> = vec![None; places];
            for _ in 0..3 * places {
                let k = below(&mut r, places as u64) as usize;
                if let Some(s) = supports[k] {
                    kept.unset(k, s);
                    supports[k] = None;
                } else {
                    let s = (0..3).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64)) & top;
                    kept.set(k, s);
                    supports[k] = Some(s);
                }
            }
            let mut rebuilt = Bitmap::new(bits);
            rebuilt
                .rebuild(
                    places,
                    |k| supports[k].unwrap_or(0),
                    1,
                    budget,
                    Site::Update,
                )
                .unwrap();
            let mut buffer = Vec::new();
            for _ in 0..16 {
                let query = (0..=below(&mut r, 2))
                    .fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64))
                    & top;
                let case = format!("{bits} bits, {places} places, query {query:#x}");
                let holds: Vec<usize> = (0..places)
                    .filter(|&k| supports[k].is_some_and(|s| s & query == query))
                    .collect();
                let meets: Vec<usize> = (0..places)
                    .filter(|&k| supports[k].is_some_and(|s| s & query != 0))
                    .collect();
                for (name, index) in [("kept", &kept), ("rebuilt", &rebuilt)] {
                    let mut got = Vec::new();
                    index
                        .holding(query, places, &mut buffer, 1, budget, Site::Update, |k| {
                            got.push(k);
                        })
                        .unwrap();
                    assert_eq!(got, holds, "{case}, {name}: holding");
                    let mut got = Vec::new();
                    index
                        .meeting(query, 1, budget, Site::Update, |k| got.push(k))
                        .unwrap();
                    assert_eq!(got, meets, "{case}, {name}: meeting");
                    let last = index
                        .meeting_last(query, 1, budget, Site::Update, |_| true)
                        .unwrap();
                    assert_eq!(last, meets.last().copied(), "{case}, {name}: meeting_last");
                }
            }
        }
    }
}

/// The call of `text` with the item's switches as given.
fn switched(text: &str, params: &Params, holes: bool, tombs: bool) -> Outcome {
    with(holes, tombs, || call(text, params, true))
}

/// Test 5: pair-index-upkeep-time.1 to .8. Calls over GF(2), GF(7),
/// GF(65521), and ℚ on 10 to 140 variables, each regime of the index among
/// them, on gf2-update-rest-time's systems (`systems_on`), and the edge
/// cases of `OBJECTIVES.md` over each field, in grevlex and lex, the
/// signatures off and on: with the holes, the tombstones, or both, a call
/// returns the basis and the report, but its times, that it returns without
/// them.
#[test]
fn calls_return_what_they_return_without_them() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for p in [2u64, 7, 65521, 0] {
        for (k, g) in EDGE.iter().enumerate() {
            texts.push((format!("edge {k} over {p}"), format!("x,y,z\n{p}\n{g}\n")));
            let text = placed(&format!("x,y,z\n{p}\n{g}\n"), &spread(3, 40), 40);
            texts.push((format!("edge {k} over {p} on 40"), text));
        }
    }
    for n in [10, 16, 20, 32, 33, 60, 129, 140] {
        texts.extend(systems_on(n));
    }
    for (name, text) in &texts {
        for order in [MonomialOrder::Grevlex, MonomialOrder::Lex] {
            for signatures in [false, true] {
                let mut params = reported(order.clone());
                params.signatures = signatures;
                if built(&params).is_err() {
                    continue;
                }
                let off = switched(text, &params, false, false);
                let name = format!("{name}, {order:?}, signatures {signatures}");
                for (holes, tombs) in [(true, false), (false, true), (true, true)] {
                    let on = switched(text, &params, holes, tombs);
                    let case = format!("{name}, holes {holes}, tombs {tombs}");
                    assert_eq!(on.basis, off.basis, "{case}: the basis");
                    assert_eq!(untimed(&on), untimed(&off), "{case}: the report");
                }
            }
        }
    }
}

/// Test 6: pair-index-upkeep-time.1, .4, .6, .7. Calls over GF(2) on
/// boolean systems of 12 variables placed on 12, 24, and 60 variables,
/// whose selections vacate places and whose updates fill holes and make
/// members of `G_min` redundant, stop at the first check at `Site::Select`
/// and at `Site::Update` and return what they return without the forcing,
/// with the switches on and off.
#[test]
fn stops_in_the_selection_and_the_update() {
    for n in [12, 24, 60] {
        let text = placed(&boolean(12, 20, 0x5707), &spread(12, n), n);
        for site in [Site::Select, Site::Update] {
            for on in [true, false] {
                let got = with(on, on, || {
                    stops_at(site, &Case::new(text.clone(), "grevlex", false))
                });
                got.unwrap_or_else(|e| panic!("{n} variables, {site:?}, switches {on}: {e}"));
            }
        }
    }
}
