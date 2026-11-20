//! gf2-chain-criteria-time, the item's test code: the plain run's update
//! keeps the pairs of the reference's Algorithm 2, and the pairs it keeps
//! without a divided element's one pair and the colon masks, in every key
//! regime; a divided element's one pair and the colons of one and two
//! variables by hand; the index's search from the end against a scan; the
//! divisor mask's bit per variable; a call returns what it returns without
//! them; and a stop in the update
//! (`iterations/0170-gf2-chain-criteria-time/plan.md`, _Test code_, 1 to
//! 7).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::bitmap::Bitmap;
use crate::budget::{Budget, Site};
use crate::call::built;
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::ccode_newton_gap_tests::call;
use crate::divisibility_tests::{Engine, RefPair, Reference, degree, lcm};
use crate::gf2_update_rest_time_tests::{pairs_of, spread, systems_on};
use crate::linear_substitution_time_tests::untimed;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format, SUPPORT_BITS};
use crate::order::{Grevlex, Lex, Order};
use crate::pairs::{COLON_MASKS, INDEX_FORCED, ONE_PAIR};
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::{below, rng};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{boolean, placed};
use crate::{MonomialOrder, Outcome, Params};

/// The two switches of the item, set to `on`, and back to their default,
/// both on, after `f`.
fn with<T>(one_pair: bool, masks: bool, f: impl FnOnce() -> T) -> T {
    ONE_PAIR.set(one_pair);
    COLON_MASKS.set(masks);
    let got = f();
    ONE_PAIR.set(true);
    COLON_MASKS.set(true);
    got
}

/// A leading monomial on n variables for step `step` of a draw over the
/// leads `entered`: in a run of `most` 1 a square of one variable at one
/// step in eight, as the field equations' leads stand, else a product of
/// distinct variables; elsewhere exponents up to `most`. At one step in
/// three it is a multiple of an earlier lead, by a monomial of degree 0 to
/// 2, so that a member of `G_min` may divide it. Half the variables are
/// drawn from the first `hot`, so that the leads share variables.
#[allow(
    clippy::too_many_arguments,
    reason = "the draw, the variables, their hot part, the degree and exponent bounds, and the leads so far, each a dimension of the draw"
)]
pub(crate) fn lead_for(
    r: &mut Xoshiro256PlusPlus,
    n: usize,
    hot: usize,
    top: u32,
    most: u32,
    entered: &[Vec<u32>],
) -> Vec<u32> {
    let pick = |r: &mut Xoshiro256PlusPlus| {
        if below(r, 2) == 0 {
            below(r, hot as u64) as usize
        } else {
            below(r, n as u64) as usize
        }
    };
    let cap = most.max(1);
    if most == 1 && below(r, 8) == 0 {
        let mut e = vec![0u32; n];
        e[pick(r)] = 2;
        return e;
    }
    let mut e = if !entered.is_empty() && below(r, 3) == 0 {
        entered[below(r, entered.len() as u64) as usize].clone()
    } else {
        vec![0u32; n]
    };
    let grow = if e.iter().any(|&k| k > 0) {
        below(r, 3) as u32
    } else {
        1 + below(r, u64::from(top)) as u32
    };
    let mut left = grow;
    let mut tries = 0;
    while left > 0 && tries < 64 {
        tries += 1;
        let v = pick(r);
        if e[v] < cap {
            e[v] += 1;
            left -= 1;
        }
    }
    if e.iter().all(|&k| k == 0) {
        e[pick(r)] = 1;
    }
    e
}

/// Test 1's run in format M on n variables: `steps` leads of degree at most
/// `top`, exponents at most `most` (`lead_for`), the constant tenth from
/// the last, entered into the engine
/// with the item's switches on, into one with them off, and into the
/// reference: after every step the pair set equals the reference's pairs,
/// the two engines' sets agree, words and all, and so do their `G_min`s.
/// `forced` fixes the indexes (`INDEX_FORCED`). The steps that entered an
/// element a member of `G_min` divided.
#[allow(
    clippy::too_many_arguments,
    reason = "the order, the variables, the seed, the steps, the degree and exponent bounds, the index's choice, and the covered sugar, each a dimension of the comparison"
)]
fn alike<M: Format, O: Order<M>>(
    order: &O,
    n: usize,
    seed: u64,
    steps: usize,
    top: u32,
    most: u32,
    forced: Option<bool>,
) -> usize {
    let mut on: Engine<M> = Engine::new(n, seed);
    let mut off: Engine<M> = Engine::new(n, seed);
    let mut reference = Reference::new(O::GRADED, M::MAX_DEGREE);
    let mut r = rng(seed ^ 0xc4a1);
    let mut entered: Vec<Vec<u32>> = Vec::new();
    let mut divided = 0;
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
        if reference
            .min_members()
            .iter()
            .any(|&i| entered[i as usize].iter().zip(&lead).all(|(a, b)| a <= b))
        {
            divided += 1;
        }
        INDEX_FORCED.set(forced);
        let a = with(true, true, || on.enter(order, &lead, sugar).0);
        let b = with(false, false, || off.enter(order, &lead, sugar).0);
        INDEX_FORCED.set(None);
        reference
            .update(lead.clone(), sugar)
            .unwrap_or_else(|e| panic!("{case}: the reference: {e:?}"));
        assert_eq!(a, Ok(()), "{case}: the update");
        assert_eq!(b, Ok(()), "{case}: the update without the switches");
        assert_eq!(
            on.written(),
            reference.pairs,
            "{case}: the reference's pairs"
        );
        assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the pair sets");
        assert_eq!(on.min_arrays(), off.min_arrays(), "{case}: G_min's arrays");
        let min: Vec<u32> = on.min_members().iter().map(|i| i.index() as u32).collect();
        assert_eq!(min, reference.min_members(), "{case}: G_min");
        entered.push(lead);
    }
    divided
}

/// Test 1: gf2-chain-criteria-time.1 to .6, .8. The update with a divided
/// element's one pair and the colon masks keeps the pairs of the
/// reference's Algorithm 2 and those it keeps without them, in every key
/// regime: up to 16 variables, where the divisor mask holds two bits or
/// more a variable; 17 to 32, one bit; 33 to `SUPPORT_BITS`, the index's
/// supports; and past `SUPPORT_BITS`, folded supports and no masks; with
/// the index forced on below 33 variables and off above 32; in `Dense4`,
/// `Dense8`, `Dense16`, and `Dense32`, in grevlex and lex; on leads of
/// distinct variables beside squares, as over GF(2), and on exponents to 3.
/// A constant among the leads leaves `G_min` the constant alone. Each
/// longer run enters divided elements.
#[test]
fn the_update_keeps_the_reference_pairs() {
    for n in [6, 16, 17, 23, 32, 33, 60, 128, 129, 300] {
        for (most, top) in [(1, 4), (3, 3)] {
            for forced in [None, Some(n <= 32)] {
                let seed = 0x17 + n as u64 + u64::from(most) * 0x100;
                let d = alike::<Dense8, _>(&Grevlex, n, seed, 220, top, most, forced);
                assert!(
                    d > 0,
                    "Dense8, {n} variables, most {most}: no divided element"
                );
            }
        }
    }
    for n in [12, 20, 40, 130] {
        alike::<Dense4, _>(&Grevlex, n, 0x4e + n as u64, 200, 4, 1, None);
        alike::<Dense16, _>(&Grevlex, n, 0x16e + n as u64, 160, 3, 3, None);
        alike::<Dense32, _>(&Grevlex, n, 0x32e + n as u64, 120, 3, 2, None);
        alike::<Dense8, _>(&Lex, n, 0x1e + n as u64, 160, 3, 2, None);
    }
}

/// The new pairs of the element just entered, as the reference writes
/// them.
fn last_pairs<M: Format>(e: &Engine<M>, t: u32) -> Vec<RefPair> {
    e.written().into_iter().filter(|p| p.j == t).collect()
}

/// The exponents of `vars` on n variables, each to its power.
pub(crate) fn mono(n: usize, vars: &[(usize, u32)]) -> Vec<u32> {
    let mut e = vec![0u32; n];
    for &(v, k) in vars {
        e[v] = k;
    }
    e
}

/// Test 2 on n variables, the index as `forced`: leads x0·x1, x2, x3·x4,
/// x2·x5·x6, then h = x0·x1·x2·x7, which x0·x1 and x2 divide, and then x3·x4
/// again, which the member x3·x4 divides with its own lead: each takes the
/// one pair with its last divisor in `G_min`'s order, of lcm its lead, and
/// joins no `G_min`. After x0 and the constant, x0·x3 keeps no pair, the
/// constant's pair the product criterion's, and `G_min` stays the constant.
fn divided_on<M: Format>(n: usize, forced: Option<bool>) {
    let mut e: Engine<M> = Engine::new(n, 0xd1d);
    let leads = [
        mono(n, &[(0, 1), (1, 1)]),
        mono(n, &[(2, 1)]),
        mono(n, &[(3, 1), (4, 1)]),
        mono(n, &[(2, 1), (5, 1), (6, 1)]),
    ];
    INDEX_FORCED.set(forced);
    for lead in &leads {
        e.enter(&Grevlex, lead, degree(lead)).0.unwrap();
    }
    let members = e.min_len();
    let h = mono(n, &[(0, 1), (1, 1), (2, 1), (7, 1)]);
    e.enter(&Grevlex, &h, 4).0.unwrap();
    let case = format!("{n} variables, forced {forced:?}");
    assert_eq!(
        last_pairs(&e, 4),
        vec![RefPair {
            i: 1,
            j: 4,
            lcm: h.clone(),
            degree: 4,
            sugar: 4,
        }],
        "{case}: x0·x1·x2·x7 keeps its pair with x2, the later divisor"
    );
    assert_eq!(e.min_len(), members, "{case}: h joins no G_min");
    let again = mono(n, &[(3, 1), (4, 1)]);
    e.enter(&Grevlex, &again, 2).0.unwrap();
    INDEX_FORCED.set(None);
    assert_eq!(
        last_pairs(&e, 5),
        vec![RefPair {
            i: 2,
            j: 5,
            lcm: again,
            degree: 2,
            sugar: 2,
        }],
        "{case}: a lead equal to a member's keeps its pair with it"
    );
    assert_eq!(
        e.min_len(),
        members,
        "{case}: the equal lead joins no G_min"
    );
    let mut unit: Engine<M> = Engine::new(n, 0x1d1);
    INDEX_FORCED.set(forced);
    for lead in [
        mono(n, &[(0, 1)]),
        vec![0u32; n],
        mono(n, &[(0, 1), (3, 1)]),
    ] {
        unit.enter(&Grevlex, &lead, degree(&lead)).0.unwrap();
    }
    INDEX_FORCED.set(None);
    assert!(
        last_pairs(&unit, 2).is_empty(),
        "{case}: after the constant, x0·x3 keeps no pair"
    );
    assert_eq!(unit.min_len(), 1, "{case}: G_min the constant alone");
}

/// Test 2: gf2-chain-criteria-time.1 to .3. A divided element keeps the
/// one pair with its last divisor, by hand: on 8 variables, the mask
/// holding four bits a variable, with the index off and forced on; on 20,
/// one bit; on 40, the index; and on 300, where x0 and x128 share a
/// folded support bit.
#[test]
fn a_divided_element_keeps_its_last_divisors_pair() {
    divided_on::<Dense8>(8, None);
    divided_on::<Dense8>(8, Some(true));
    divided_on::<Dense8>(20, None);
    divided_on::<Dense4>(40, None);
    divided_on::<Dense8>(300, None);
}

/// Test 3 on n variables, the index as `forced`: `G_min`'s leads x0·x2,
/// x1·x2, x0·x3·x4, x1·x3·x4, x0², x5·x6·x7, x1·x5·x8, x0·x5·x8·x9,
/// x1·x6·x7·x9, and x2·x6·x9, then h = x0·x1. The colons lcm(i, h)/h are
/// x2 twice, x3·x4 twice, x0, a variable of h's lead, none for the coprime
/// x5·x6·x7, x5·x8, x5·x8·x9, x6·x7·x9, and x2·x6·x9. h keeps the pairs
/// the reference keeps: of each equal colon the later, the colon inside the
/// lead, x5·x8, and x6·x7·x9; x5·x8 drops x5·x8·x9, and x2 drops x2·x6·x9.
fn colons_on<M: Format>(n: usize, forced: Option<bool>) {
    let leads = [
        mono(n, &[(0, 1), (2, 1)]),
        mono(n, &[(1, 1), (2, 1)]),
        mono(n, &[(0, 1), (3, 1), (4, 1)]),
        mono(n, &[(1, 1), (3, 1), (4, 1)]),
        mono(n, &[(0, 2)]),
        mono(n, &[(5, 1), (6, 1), (7, 1)]),
        mono(n, &[(1, 1), (5, 1), (8, 1)]),
        mono(n, &[(0, 1), (5, 1), (8, 1), (9, 1)]),
        mono(n, &[(1, 1), (6, 1), (7, 1), (9, 1)]),
        mono(n, &[(2, 1), (6, 1), (9, 1)]),
    ];
    let h = mono(n, &[(0, 1), (1, 1)]);
    let mut e: Engine<M> = Engine::new(n, 0xc010);
    let mut reference = Reference::new(true, M::MAX_DEGREE);
    INDEX_FORCED.set(forced);
    for lead in leads.iter().chain([&h]) {
        let sugar = degree(lead);
        e.enter(&Grevlex, lead, sugar).0.unwrap();
        reference.update(lead.clone(), sugar).unwrap();
    }
    INDEX_FORCED.set(None);
    let case = format!("{n} variables, forced {forced:?}");
    let t = leads.len() as u32;
    assert_eq!(
        e.min_len(),
        leads.len() + 1,
        "{case}: every lead joins G_min"
    );
    let want: Vec<RefPair> = reference
        .pairs
        .iter()
        .filter(|p| p.j == t)
        .cloned()
        .collect();
    assert_eq!(last_pairs(&e, t), want, "{case}: the pairs of h");
    let partners: Vec<u32> = want.iter().map(|p| p.i).collect();
    assert_eq!(partners, vec![1, 3, 4, 6, 8], "{case}: h's partners");
}

/// Test 3: gf2-chain-criteria-time.4, .5. The colons of one and two
/// variables outside h's lead, equal colons of each, a colon inside the
/// lead, and colons of three, by hand, against the reference, in each key
/// regime.
#[test]
fn colons_of_one_and_two_variables() {
    colons_on::<Dense8>(10, None);
    colons_on::<Dense8>(10, Some(true));
    colons_on::<Dense16>(24, None);
    colons_on::<Dense4>(70, None);
    colons_on::<Dense8>(140, None);
}

/// Test 4: gf2-chain-criteria-time.2. `Bitmap::meeting_last` returns the
/// last place whose support shares a bit with the query and passes the
/// test, as a scan from the last place finds it, on indexes of 1 to
/// `SUPPORT_BITS` bits over 1 to 300 places, set place by place and
/// rebuilt, with tests that pass every place, none, and one place in three.
#[test]
fn the_search_from_the_end_against_a_scan() {
    let mut r = rng(0x1a57);
    for bits in [1, 7, 64, SUPPORT_BITS] {
        for places in [1, 63, 64, 65, 300] {
            let top = if bits == SUPPORT_BITS {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            let supports: Vec<u128> = (0..places)
                .map(|_| match below(&mut r, 4) {
                    0 => 0,
                    _ => (0..3).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64)) & top,
                })
                .collect();
            let mut set = Bitmap::new(bits);
            for (k, &s) in supports.iter().enumerate() {
                set.set(k, s);
            }
            let mut rebuilt = Bitmap::new(bits);
            rebuilt
                .rebuild(
                    places,
                    |k| supports[k],
                    1,
                    &mut Budget::unbounded(),
                    Site::Update,
                )
                .unwrap();
            for _ in 0..12 {
                let query = (0..2).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64)) & top;
                for pass in [0u64, 1, 3] {
                    let test = |k: usize| match pass {
                        0 => false,
                        1 => true,
                        _ => k.is_multiple_of(3),
                    };
                    let want = (0..places)
                        .rev()
                        .find(|&k| supports[k] & query != 0 && test(k));
                    for (name, index) in [("set", &set), ("rebuilt", &rebuilt)] {
                        let got = index
                            .meeting_last(query, 1, &mut Budget::unbounded(), Site::Update, test)
                            .unwrap();
                        assert_eq!(
                            got, want,
                            "{name}: {bits} bits, {places} places, {query:#x}, pass {pass}"
                        );
                    }
                }
            }
        }
    }
}

/// Test 5's check in format M on n variables: the bits of each variable's
/// powers to 3 are no other variable's, and for drawn exponents a and h to
/// 3, the bits of mask(lcm(a, h)) outside mask(h) that are a variable u's
/// number at most u's exponent in lcm(a, h)/h.
fn mask_bits_in<M: Format>(n: usize, seed: u64) {
    let f = M::new(n);
    let mut words = vec![M::Word::default(); f.words()];
    let mut mask = |e: &[u32]| {
        f.pack(e, &mut words);
        f.mask(&words)
    };
    let own: Vec<u32> = (0..n).map(|v| mask(&mono(n, &[(v, 3)]))).collect();
    let mut taken = 0u32;
    for (v, &bits) in own.iter().enumerate() {
        assert!(bits != 0, "{n} variables: x{v} sets a bit");
        assert_eq!(
            bits & taken,
            0,
            "{n} variables: x{v}'s bits {bits:#x} are its own"
        );
        taken |= bits;
    }
    let mut r = rng(seed);
    for _ in 0..200 {
        let a: Vec<u32> = (0..n).map(|_| below(&mut r, 4) as u32).collect();
        let h: Vec<u32> = (0..n).map(|_| below(&mut r, 4) as u32).collect();
        let l = lcm(&a, &h);
        let outside = mask(&l) & !mask(&h);
        for (v, &bits) in own.iter().enumerate() {
            assert!(
                (outside & bits).count_ones() <= l[v] - h[v],
                "{n} variables: x{v} in lcm({a:?}, {h:?}) over {h:?}"
            );
        }
    }
}

/// Test 5: gf2-chain-criteria-time.4, .5. Up to 32 variables each bit of
/// the divisor mask stands for one variable's exponent reaching a
/// threshold, and the bits of an lcm's mask outside the lead's that are a
/// variable's number at most its exponent in the colon: what the colon
/// masks read from the divisor masks.
#[test]
fn the_mask_bits_stand_for_one_variable_to_32() {
    for n in [1, 2, 5, 10, 16, 17, 31, 32] {
        let seed = 0x5a5 + n as u64;
        mask_bits_in::<Dense4>(n, seed);
        mask_bits_in::<Dense8>(n, seed);
        mask_bits_in::<Dense16>(n, seed);
        mask_bits_in::<Dense32>(n, seed);
    }
}

/// The call of `text` with the item's switches as given.
fn switched(text: &str, params: &Params, one_pair: bool, masks: bool) -> Outcome {
    with(one_pair, masks, || call(text, params, true))
}

/// Test 6: gf2-chain-criteria-time.1 to .8. Calls over GF(2), GF(7),
/// GF(65521), and ℚ on 10 to 140 variables, each key regime among them, on
/// gf2-update-rest-time's systems (`gf2_update_rest_time_tests::systems_on`),
/// and the edge cases of `OBJECTIVES.md` over each field, in grevlex and
/// lex, the signatures off and on: with a divided element's one pair, the
/// colon masks, or both, a call returns the basis and the report, but its
/// times, that it returns without them.
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
                for (one_pair, masks) in [(true, false), (false, true), (true, true)] {
                    let on = switched(text, &params, one_pair, masks);
                    let case = format!("{name}, one pair {one_pair}, masks {masks}");
                    assert_eq!(on.basis, off.basis, "{case}: the basis");
                    assert_eq!(untimed(&on), untimed(&off), "{case}: the report");
                }
            }
        }
    }
}

/// Test 7: gf2-chain-criteria-time.1, .2, .4. Calls over GF(2) on boolean
/// systems of 12 variables placed on 12, 24, and 60 variables, whose
/// updates enter divided elements and drop pairs by the colon masks, stop
/// at the first check at `Site::Update` and return what they return without
/// the forcing, with the switches on and off.
#[test]
fn stops_in_the_update() {
    for n in [12, 24, 60] {
        let text = placed(&boolean(12, 20, 0x5707), &spread(12, n), n);
        for on in [true, false] {
            let got = with(on, on, || {
                stops_at(Site::Update, &Case::new(text.clone(), "grevlex", false))
            });
            got.unwrap_or_else(|e| panic!("{n} variables, switches {on}: {e}"));
        }
    }
}
