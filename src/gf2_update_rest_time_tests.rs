//! gf2-update-rest-time, the item's test code: a support holds the
//! variables of nonzero exponent, the bitmap index names the places a scan
//! of the supports names, the plain run's update and selection leave the
//! pair set, `G_min`, and the pairs taken as they leave them without the
//! indexes, also over a basis the set did not enter, and a call returns
//! what it returns without them
//! (`iterations/0137-gf2-update-rest-time/plan.md`, _Test code_, 1 to 9).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::bitmap::Bitmap;
use crate::budget::{Budget, Site};
use crate::call::{built, replay_in};
use crate::divisibility_tests::Engine;
use crate::format_tests::{call, heard, params};
use crate::gf2_start_over_tests::forcing;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format, SUPPORT_BITS, WIDTHS};
use crate::order::{Grevlex, Lex, Order};
use crate::packed_tests::mono;
use crate::pairs::{self, INDEX_FORCED, Pair, PairSet, SPLIT_BOUNDS};
use crate::shared_tests::{CASE_DEADLINE, below, rng};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{boolean, cubics, placed, random_system};
use crate::trace::NoTrace;
use crate::{Error, Outcome, Params};

/// The support of exponents `e` as `Format::support` states it: bit v mod
/// `SUPPORT_BITS` for each variable v of nonzero exponent.
pub(crate) fn support_of(e: &[u32]) -> u128 {
    e.iter()
        .enumerate()
        .filter(|&(_, &k)| k > 0)
        .fold(0, |s, (v, _)| s | 1 << (v % SUPPORT_BITS))
}

/// Exponents on n variables of degree at most `degree`, each variable's
/// at most `most`.
fn draw(r: &mut Xoshiro256PlusPlus, n: usize, degree: u32, most: u32) -> Vec<u32> {
    let mut e = vec![0u32; n];
    let mut left = below(r, u64::from(degree) + 1) as u32;
    while left > 0 {
        let v = below(r, n as u64) as usize;
        if e[v] < most {
            e[v] += 1;
            left -= 1;
        } else if e.iter().all(|&k| k >= most) {
            break;
        }
    }
    e
}

/// Test 1's checks in format M on n variables: the support of drawn
/// monomials, of their lcms, and of a divisor's multiple.
fn supports_in<M: Format>(n: usize, seed: u64) {
    let f = M::new(n);
    let degree = M::MAX_DEGREE.min(9) / 2;
    let mut r = rng(seed);
    let mut lcm = vec![M::Word::default(); f.words()];
    for _ in 0..300 {
        let (ea, eb) = (draw(&mut r, n, degree, 3), draw(&mut r, n, degree, 3));
        let ((a, _), (b, _)) = (mono(&f, &ea), mono(&f, &eb));
        let (sa, sb) = (f.support(&a), f.support(&b));
        assert_eq!(sa, support_of(&ea), "{n} variables: {ea:?}");
        f.lcm_words_n::<{ crate::monomial::ANY }>(&a, &b, &mut lcm);
        assert_eq!(
            f.support(&lcm),
            sa | sb,
            "{n} variables: lcm of {ea:?}, {eb:?}"
        );
        let product: Vec<u32> = ea.iter().zip(&eb).map(|(x, y)| x + y).collect();
        let (c, _) = mono(&f, &product);
        assert_eq!(
            f.support(&a) & !f.support(&c),
            0,
            "{ea:?} divides {product:?}"
        );
        if n <= SUPPORT_BITS {
            let shared = ea.iter().zip(&eb).any(|(&x, &y)| x > 0 && y > 0);
            assert_eq!(sa & sb != 0, shared, "{n} variables: {ea:?}, {eb:?}");
        }
    }
}

/// Test 1: gf2-update-rest-time.2. In every format a support holds bit v
/// mod 128 for each variable v of nonzero exponent, from one variable to
/// 300, so that an lcm's support is its factors' union and a divisor's
/// lies in its multiple's; to 128 variables two monomials share a bit
/// exactly where they share a variable.
#[test]
fn supports_agree_with_the_exponents() {
    for n in [1, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 200, 300] {
        let seed = 0x5a99 + n as u64;
        supports_in::<Dense4>(n, seed);
        supports_in::<Dense8>(n, seed);
        supports_in::<Dense16>(n, seed);
        supports_in::<Dense32>(n, seed);
    }
}

/// The places below `places` whose support holds every bit of `query`,
/// and those whose support shares a bit with it, by a scan of `supports`.
fn scanned(supports: &[u128], places: usize, query: u128) -> (Vec<usize>, Vec<usize>) {
    let holding = (0..places)
        .filter(|&k| supports[k] & query == query)
        .collect();
    let meeting = (0..supports.len())
        .filter(|&k| supports[k] & query != 0)
        .collect();
    (holding, meeting)
}

/// Test 2: gf2-update-rest-time.1, .5, .6. The bitmap index over 0 to
/// 40,000 places, every chunk boundary among them, of supports of 1 to 128
/// bits, set place by place with places left empty, rebuilt, and cleared:
/// the places it names as holding a query's every bit, below a bound, and
/// as sharing a bit with it are those a scan of the supports names, in
/// increasing order; a query of no bit names every place below the bound.
#[test]
fn the_bitmap_names_what_a_scan_names() {
    let mut r = rng(0xb17a);
    let mut buffer = Vec::new();
    for bits in [1, 7, 64, 127, 128] {
        for places in [0, 1, 63, 64, 65, 16_383, 16_384, 16_385, 40_000] {
            let top: u128 = if bits == SUPPORT_BITS {
                u128::MAX
            } else {
                (1 << bits) - 1
            };
            let supports: Vec<u128> = (0..places)
                .map(|_| {
                    if below(&mut r, 4) == 0 {
                        return 0;
                    }
                    let x = (0..3).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64));
                    x & top
                })
                .collect();
            let mut set = Bitmap::new(bits);
            for (k, &s) in supports.iter().enumerate() {
                if s != 0 || k + 1 == places {
                    set.set(k, s);
                }
            }
            let mut rebuilt = Bitmap::new(bits);
            rebuilt.set(3, top);
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
                let query = match below(&mut r, 6) {
                    0 => 0,
                    1 => 1 << below(&mut r, bits as u64),
                    _ => (0..2).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64)),
                };
                let bound = match below(&mut r, 3) {
                    0 => places / 2,
                    _ => places,
                };
                let (holding, meeting) = scanned(&supports, bound, query);
                for (name, index) in [("set", &set), ("rebuilt", &rebuilt)] {
                    let mut got = Vec::new();
                    index
                        .holding(
                            query,
                            bound,
                            &mut buffer,
                            1,
                            &mut Budget::unbounded(),
                            Site::Update,
                            |k| {
                                got.push(k);
                            },
                        )
                        .unwrap();
                    let case = format!("{name}: {bits} bits, {places} places, {query:#x}");
                    assert_eq!(got, holding, "{case}: holding, below {bound}");
                    if query != 0 {
                        let mut got = Vec::new();
                        index
                            .meeting(query, 1, &mut Budget::unbounded(), Site::Update, |k| {
                                got.push(k);
                            })
                            .unwrap();
                        assert_eq!(got, meeting, "{case}: meeting");
                    }
                }
            }
            set.clear();
            let mut got = Vec::new();
            set.holding(
                top & 1,
                places,
                &mut buffer,
                1,
                &mut Budget::unbounded(),
                Site::Update,
                |k| {
                    got.push(k);
                },
            )
            .unwrap();
            assert!(got.is_empty(), "{bits} bits, {places} places: cleared");
            set.set(70, top & 1);
            got.clear();
            set.holding(
                top & 1,
                71,
                &mut buffer,
                1,
                &mut Budget::unbounded(),
                Site::Update,
                |k| {
                    got.push(k);
                },
            )
            .unwrap();
            assert_eq!(got, vec![70], "{bits} bits: set again after the clear");
        }
    }
}

/// Test 3: gf2-update-rest-time.3. The indexes run past 32 variables, and
/// `INDEX_FORCED` decides where it holds a choice.
#[test]
fn the_indexes_run_past_32_variables() {
    for (n, want) in [
        (1, false),
        (32, false),
        (33, true),
        (128, true),
        (423, true),
    ] {
        assert_eq!(pairs::indexed(n), want, "{n} variables");
    }
    for forced in [true, false] {
        INDEX_FORCED.set(Some(forced));
        let (low, high) = (pairs::indexed(2), pairs::indexed(300));
        INDEX_FORCED.set(None);
        assert_eq!((low, high), (forced, forced), "forced {forced}");
    }
}

/// The pairs of an engine's set, live, in order, with their lcms' words.
pub(crate) fn pairs_of<M: Format>(e: &Engine<M>) -> Vec<(Pair, Vec<M::Word>)> {
    e.pairs
        .with_words()
        .map(|(p, w)| (*p, w.to_vec()))
        .collect()
}

/// Enters `steps` drawn leading monomials of degree at most `degree`,
/// exponents at most `most`, into two engines on n variables in format M,
/// one whose pair set keeps its indexes and one whose does not, with a
/// selection after every 16th under the split bounds `bounds`, and the
/// sets cleared halfway: the updates' results, the pair sets, `G_min`'s
/// arrays, and the pairs taken agree after every step. The leads draw half
/// their variables from the first 24, so that their pairs share variables.
/// The pairs the B criterion dropped, summed over the selections.
#[allow(
    clippy::too_many_arguments,
    reason = "the order, the format's variables, the draw's seed, steps, degree, and exponent bound, the covered sugar, and the split bounds, each a dimension of the comparison"
)]
fn alike<M: Format, O: Order<M>>(
    order: &O,
    n: usize,
    seed: u64,
    steps: usize,
    degree: u32,
    most: u32,
    covered: u32,
    bounds: Option<(usize, usize)>,
) -> usize {
    let mut on: Engine<M> = Engine::new(n, seed);
    let mut off: Engine<M> = Engine::new(n, seed);
    let mut r = rng(seed ^ 0x1dea);
    let mut dropped = 0;
    let hot = n.min(24);
    for step in 0..steps {
        let mut lead = vec![0u32; n];
        let d = 1 + below(&mut r, u64::from(degree)) as u32;
        let mut left = d;
        while left > 0 {
            let v = if below(&mut r, 2) == 0 {
                below(&mut r, hot as u64)
            } else {
                below(&mut r, n as u64)
            } as usize;
            if lead[v] < most {
                lead[v] += 1;
                left -= 1;
            }
        }
        let sugar = d + below(&mut r, 2) as u32;
        let case = format!("{n} variables, seed {seed:#x}, step {step}: {lead:?}");
        INDEX_FORCED.set(Some(true));
        let a = on.enter_with(order, &lead, sugar, covered);
        INDEX_FORCED.set(Some(false));
        let b = off.enter_with(order, &lead, sugar, covered);
        INDEX_FORCED.set(None);
        assert_eq!(a, b, "{case}: the update");
        assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the pair set");
        assert_eq!(on.min_arrays(), off.min_arrays(), "{case}: G_min");
        if step % 16 == 15 {
            dropped += on.pairs.held() - on.pairs.len();
            SPLIT_BOUNDS.set(bounds);
            let a = pairs::select(
                &mut on.pairs,
                &mut on.table,
                order,
                &mut Budget::unbounded(),
            );
            let b = pairs::select(
                &mut off.pairs,
                &mut off.table,
                order,
                &mut Budget::unbounded(),
            );
            SPLIT_BOUNDS.set(None);
            assert_eq!(a, b, "{case}: the selection");
            assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the set left");
            assert_eq!(
                on.pairs.deferred(),
                off.pairs.deferred(),
                "{case}: deferred"
            );
        }
        if step == steps / 2 {
            on.pairs.clear();
            off.pairs.clear();
        }
    }
    dropped
}

/// Test 4: gf2-update-rest-time.1, .4, .7 to .9. The plain run's update and
/// selection with the indexes leave the pair set, `G_min`, and the pairs
/// taken as they leave them without: in `Dense4` on 33 to 140 variables,
/// in `Dense8` on 33 to 200 with exponents to 3, past `SUPPORT_BITS` where
/// the supports fold, in `Dense16` and `Dense32`, in grevlex and lex, with
/// a covered sugar, and under split bounds that split the selections. The
/// B criterion drops pairs in every one of the longer runs.
#[test]
fn the_update_with_and_without_the_indexes() {
    for n in [33, 64, 65, 129, 140] {
        let d = alike::<Dense4, _>(&Grevlex, n, 0x4a + n as u64, 400, 3, 1, 0, None);
        assert!(d > 0, "Dense4, {n} variables: the B criterion drops pairs");
    }
    for n in [33, 100, 128, 129, 200] {
        let d = alike::<Dense8, _>(&Grevlex, n, 0x8a + n as u64, 400, 4, 3, 0, None);
        assert!(d > 0, "Dense8, {n} variables: the B criterion drops pairs");
    }
    alike::<Dense8, _>(&Lex, 100, 0x1e, 300, 4, 2, 0, None);
    alike::<Dense16, _>(&Grevlex, 40, 0x16, 300, 4, 3, 0, None);
    alike::<Dense32, _>(&Grevlex, 40, 0x32, 200, 4, 3, 0, None);
    alike::<Dense8, _>(&Grevlex, 64, 0xc0, 300, 4, 2, 4, None);
    alike::<Dense4, _>(&Grevlex, 65, 0x5b, 400, 3, 1, 0, Some((8, 3)));
    alike::<Dense8, _>(&Grevlex, 129, 0x5c, 400, 4, 2, 0, Some((4, 1)));
}

/// The outcome of the call in `WIDTHS` with the indexes forced on, which
/// the call with them forced off and the call they leave to the variables'
/// count return too; with cofactors certified. None where the deadline
/// stopped a call.
pub(crate) fn as_without(name: &str, text: &str, p: &Params) -> Option<Outcome> {
    let mut got: Vec<Result<Outcome, Error>> = [Some(true), Some(false), None]
        .into_iter()
        .map(|forced| {
            INDEX_FORCED.set(forced);
            let o = call(text, p, WIDTHS.0, WIDTHS.1);
            INDEX_FORCED.set(None);
            o
        })
        .collect();
    let stopped = |o: &Result<Outcome, Error>| match o {
        Ok(o) => o.stopped.is_some(),
        Err(_) => false,
    };
    if got.iter().any(stopped) {
        return None;
    }
    let order = &p.order;
    let on = got.remove(0);
    for (forced, other) in ["off", "by the count"].iter().zip(got) {
        assert_eq!(other, on, "{name} in {order}: on and {forced}\n{text}");
    }
    let on = on.unwrap_or_else(|e| panic!("{name} in {order}: {e:?}\n{text}"));
    if p.cofactors {
        assert_eq!(
            on.certified,
            Some(true),
            "{name} in {order}: {flaw:?}",
            flaw = on.flaw
        );
    }
    Some(on)
}

/// The places of a system of k variables on n: spread over the n, the
/// first and the last among them.
pub(crate) fn spread(k: usize, n: usize) -> Vec<usize> {
    if k == 1 {
        return vec![n - 1];
    }
    (0..k).map(|i| i * (n - 1) / (k - 1)).collect()
}

/// The systems of test 5 on n variables: over GF(2) boolean systems of 6
/// and 10 variables and dense cubics of 6, over GF(7), GF(65521), and ℚ
/// random systems, each placed across the n.
pub(crate) fn systems_on(n: usize) -> Vec<(String, String)> {
    let mut cases = Vec::new();
    for (k, m, seed) in [(6, 8, 0x60), (10, 12, 0x61)] {
        let text = placed(&boolean(k, m, seed), &spread(k, n), n);
        cases.push((format!("boolean({k}, {m}, {seed:#x}) on {n}"), text));
    }
    let text = placed(&cubics(6, 8, 0x62), &spread(6, n), n);
    cases.push((format!("cubics(6, 8, 0x62) on {n}"), text));
    for p in [7, 65521, 0] {
        for seed in 0..3 {
            let small = random_system(p, seed);
            let k = small
                .split_once('\n')
                .map_or(0, |(names, _)| names.split(',').count());
            let text = placed(&small, &spread(k, n), n);
            cases.push((format!("random_system({p}, {seed}) on {n}"), text));
        }
    }
    cases
}

/// Test 5: gf2-update-rest-time.1, .3, .7 to .9. On 33 to 140 variables,
/// over GF(2), GF(7), GF(65521), and ℚ, in grevlex, lex, and `block:1`,
/// with and without cofactors, signatures on and off: the call with the
/// indexes returns the outcome of the call without them, certified where
/// it tracks cofactors; compared on at least nine calls in ten.
#[test]
fn calls_with_and_without_the_indexes() {
    let (mut compared, mut total) = (0, 0);
    for n in [33, 64, 65, 128, 129, 140] {
        for (name, text) in systems_on(n) {
            for order in ["grevlex", "lex", "block:1"] {
                for cofactors in [false, true] {
                    for signatures in [true, false] {
                        let mut p = params(order, cofactors);
                        if built(&p).is_err() {
                            continue;
                        }
                        p.signatures = signatures;
                        p.deadline = Some(CASE_DEADLINE);
                        total += 1;
                        let case =
                            format!("{name}, cofactors {cofactors}, signatures {signatures}");
                        if as_without(&case, &text, &p).is_some() {
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(10 * compared >= 9 * total, "{compared} of {total} compared");
}

/// Test 6: the edge cases of `OBJECTIVES.md` past 32 variables: the zero
/// ideal, F = (0), no generator, the unit ideal, an input already a
/// Gröbner basis, the zero polynomial among the generators, and an
/// eliminable variable, on x, y, z placed among 40 and 140 variables, over
/// GF(2), GF(65521), and ℚ: with the indexes as without, and the zero
/// ideal's basis `[]:` and the unit ideal's `[1]:`.
#[test]
fn the_edge_cases_past_32_variables() {
    for n in [40, 140] {
        for p in [2, 65521, 0] {
            for (k, gens) in EDGE.iter().enumerate() {
                let text = placed(&format!("x,y,z\n{p}\n{gens}\n"), &spread(3, n), n);
                for cofactors in [false, true] {
                    let params = params("grevlex", cofactors);
                    if built(&params).is_err() {
                        continue;
                    }
                    let case = format!("edge case {k} over {p} on {n}, cofactors {cofactors}");
                    let got = as_without(&case, &text, &params)
                        .unwrap_or_else(|| panic!("{case}: no deadline stops it"));
                    let basis = got.basis.as_deref();
                    match k {
                        0..=2 => assert_eq!(basis, Some("[]:\n"), "{case}"),
                        3 => assert_eq!(basis, Some("[1]:\n"), "{case}"),
                        _ => assert!(basis.is_some_and(|b| b != "[1]:\n"), "{case}: {basis:?}"),
                    }
                }
            }
        }
    }
}

/// Test 7: gf2-update-rest-time.4, .7. Over GF(2), boolean systems placed
/// across 48 and 129 variables under bounds that start the call over in
/// the signature run, whose basis a carried plain run takes up, return
/// with the indexes the outcome without them; and over GF(65521) a learn's
/// record replays with the indexes as without, with the learn's basis.
#[test]
fn a_carried_run_and_a_replay_with_the_indexes() {
    for n in [48, 129] {
        for (k, m, seed) in [(9, 11, 0x70), (11, 13, 0x71), (12, 24, 0x72)] {
            let text = placed(&boolean(k, m, seed), &spread(k, n), n);
            let name = format!("boolean({k}, {m}, {seed:#x}) on {n}");
            let p = params("grevlex", false);
            for forced in [Some(1), Some(1 << 10), None] {
                let got = forcing(forced, || {
                    as_without(&format!("{name}, forced {forced:?}"), &text, &p)
                });
                assert!(got.is_some(), "{name}, forced {forced:?}: no deadline");
            }
        }
        for seed in 0..3 {
            let small = random_system(65521, seed);
            let k = small
                .split_once('\n')
                .map_or(0, |(names, _)| names.split(',').count());
            let text = placed(&small, &spread(k, n), n);
            let name = format!("random_system(65521, {seed}) on {n}");
            let p = params("grevlex", false);
            let (learned, _, record) = heard(&text, &p, WIDTHS.0, WIDTHS.1);
            let learned = learned.unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let mut replays = [Some(true), Some(false)].into_iter().map(|forced| {
                INDEX_FORCED.set(forced);
                let got = replay_in(&text, &p, &record, WIDTHS.0, WIDTHS.1, &mut NoTrace);
                INDEX_FORCED.set(None);
                got.unwrap_or_else(|e| panic!("{name}, forced {forced:?}: {e:?}"))
            });
            let (on, off) = (replays.next().unwrap(), replays.next().unwrap());
            assert_eq!(on, off, "{name}: the replays");
            assert_eq!(on.basis, learned.basis, "{name}: the learn's basis");
        }
    }
}

/// Test 8: gf2-update-rest-time.1, .7, .9. Over GF(2) a lead that a member
/// of `G_min` divides, a lead of a square, and leads past `SUPPORT_BITS`
/// whose variables fold onto one support bit: the update with the indexes
/// leaves the pair set and `G_min` as without, and keeps the pair of two
/// leads that share a folded bit and no variable out of P1.
#[test]
fn folded_and_divided_leads() {
    let n = 300;
    let unit = |pairs: &[(usize, u32)]| {
        let mut e = vec![0u32; n];
        for &(v, k) in pairs {
            e[v] = k;
        }
        e
    };
    let leads = [
        unit(&[(0, 1), (128, 1)]),
        unit(&[(256, 1), (1, 1)]),
        unit(&[(0, 2)]),
        unit(&[(128, 1), (129, 1)]),
        unit(&[(0, 1)]),
        unit(&[(0, 1), (1, 1), (2, 1)]),
        unit(&[(2, 1), (130, 1)]),
    ];
    let mut on: Engine<Dense8> = Engine::new(n, 0xf01d);
    let mut off: Engine<Dense8> = Engine::new(n, 0xf01d);
    for (step, lead) in leads.iter().enumerate() {
        let sugar = lead.iter().sum::<u32>();
        INDEX_FORCED.set(Some(true));
        let a = on.enter(&Grevlex, lead, sugar);
        INDEX_FORCED.set(Some(false));
        let b = off.enter(&Grevlex, lead, sugar);
        INDEX_FORCED.set(None);
        assert_eq!(a, b, "step {step}");
        assert_eq!(pairs_of(&on), pairs_of(&off), "step {step}: the pair set");
        assert_eq!(on.min_arrays(), off.min_arrays(), "step {step}: G_min");
    }
    // x_0·x_128 and x_256·x_1 share bit 0 of their folded supports and no
    // variable: their pair is P0's, which no set holds.
    assert!(
        on.pairs
            .iter()
            .all(|p| (p.i.index(), p.j.index()) != (0, 1)),
        "the coprime pair of a folded bit"
    );
}

/// Test 9: gf2-update-rest-time.4, .7. A pair set that starts over a basis
/// of elements it did not enter, a fresh set in place of the engine's
/// after 60 updates, takes those elements' supports from the table: the
/// updates after it with the indexes leave the pair set, `G_min`, and the
/// pairs taken as they leave them without, in `Dense4` on 65 variables and
/// in `Dense8` on 140.
#[test]
fn a_fresh_set_over_a_basis_it_did_not_enter() {
    fresh_set::<Dense4>(65, 0x9a, 3, 1);
    fresh_set::<Dense8>(140, 0x9b, 4, 2);
}

/// Test 9 in format M on n variables, leads of degree at most `degree`
/// and exponents at most `most`.
fn fresh_set<M: Format>(n: usize, seed: u64, degree: u32, most: u32) {
    let mut on: Engine<M> = Engine::new(n, seed);
    let mut off: Engine<M> = Engine::new(n, seed);
    let mut r = rng(seed ^ 0xf5e5);
    for step in 0..260 {
        if step == 60 {
            on.pairs = PairSet::new(on.table.format().words());
            off.pairs = PairSet::new(off.table.format().words());
        }
        let lead = draw(&mut r, n, degree, most);
        if lead.iter().all(|&k| k == 0) {
            continue;
        }
        let sugar = lead.iter().sum::<u32>();
        let case = format!("{n} variables, step {step}: {lead:?}");
        INDEX_FORCED.set(Some(true));
        let a = on.enter(&Grevlex, &lead, sugar);
        INDEX_FORCED.set(Some(false));
        let b = off.enter(&Grevlex, &lead, sugar);
        INDEX_FORCED.set(None);
        assert_eq!(a, b, "{case}: the update");
        assert_eq!(pairs_of(&on), pairs_of(&off), "{case}: the pair set");
        assert_eq!(on.min_arrays(), off.min_arrays(), "{case}: G_min");
        if step % 16 == 15 {
            let a = pairs::select(
                &mut on.pairs,
                &mut on.table,
                &Grevlex,
                &mut Budget::unbounded(),
            );
            let b = pairs::select(
                &mut off.pairs,
                &mut off.table,
                &Grevlex,
                &mut Budget::unbounded(),
            );
            assert_eq!(a, b, "{case}: the selection");
        }
    }
}
