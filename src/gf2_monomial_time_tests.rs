//! gf2-monomial-time, the item's test code: over GF(2) a call in lanes of
//! 4 bits returns what it returns in lanes of 8, a degree past the narrow
//! bound starts the call over in `W8`, a pair whose lcm passes the bound
//! halts no run whose rounds do not take it, a call over every other field
//! starts in `W8`, and `Dense4`'s word operations agree with the exponents
//! (`iterations/0136-gf2-monomial-time/plan.md`, _Test code_, 1 to 9).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::budget::Budget;
use crate::build_tests::{agree_in, at_the_bound};
use crate::call::{built, replay_in};
use crate::divisibility_tests::Engine;
use crate::format_tests::{alone, call, heard, params};
use crate::gf2_bound_losses_tests::systems;
use crate::gf2_start_over_tests::forcing;
use crate::id::{ElementId, MonoId};
use crate::key_tests::{key, orders};
use crate::monomial::{Dense4, Dense8, Format, Overflow, WIDTHS, Width};
use crate::order::Grevlex;
use crate::packed_tests::{bound, infinity, mono, operations, table_agrees};
use crate::pairs::{self, Pair, PairSet};
use crate::shared_tests::{CASE_DEADLINE, below, rng};
use crate::systems::{boolean, cubics, placed, random_system};
use crate::table::{Limit, Table};
use crate::trace::NoTrace;
use crate::{Error, Outcome, Params};

/// `boolean(k, k + 2, seed)` on n variables: its k variables are those of
/// the places 0, 15, 16, 31, 32, and n − 1 below n, the first and last
/// lanes of `Dense4`'s words, and the other variables appear in no
/// generator.
fn straddling(n: usize, seed: u64) -> String {
    let mut places: Vec<usize> = [0, 15, 16, 31, 32, n - 1]
        .into_iter()
        .filter(|&i| i < n)
        .collect();
    places.sort_unstable();
    places.dedup();
    placed(&boolean(places.len(), places.len() + 2, seed), &places, n)
}

/// The systems over GF(2) of test 1: the edge cases and the boolean
/// systems of `gf2-bound-losses`'s tests, boolean systems whose variables
/// straddle `Dense4`'s words on 16 to 129 variables, one word to nine,
/// and dense cubics whose rounds rule rows zero by sketches.
fn cases() -> Vec<(String, String)> {
    let mut cases = systems(true);
    for n in [16, 17, 32, 33, 48, 129] {
        let seed = 0x4b17 + n as u64;
        cases.push((format!("straddling({n}, {seed:#x})"), straddling(n, seed)));
    }
    for seed in 0..3 {
        cases.push((format!("cubics(8, 12, {seed})"), cubics(8, 12, seed)));
    }
    cases
}

/// The outcome of the call in `W8` alone, which `W4` alone gives, or
/// `Error::Overflow`, and the call in `WIDTHS` gives too; None where the
/// deadline stopped a call.
fn as_in_w8(name: &str, text: &str, p: &Params) -> Option<Outcome> {
    let stopped = |o: &Result<Outcome, Error>| match o {
        Ok(o) => o.stopped.is_some(),
        Err(_) => false,
    };
    let w8 = alone(text, p, Width::W8);
    let w4 = alone(text, p, Width::W4);
    let all = call(text, p, WIDTHS.0, WIDTHS.1);
    if stopped(&w8) || stopped(&w4) || stopped(&all) {
        return None;
    }
    let order = &p.order;
    let w8 = w8.unwrap_or_else(|e| panic!("{name} in {order}: W8: {e:?}\n{text}"));
    match w4 {
        Ok(o) => assert_eq!(o, w8, "{name} in {order}: W4 and W8\n{text}"),
        Err(Error::Overflow { .. }) => {}
        Err(e) => panic!("{name} in {order}: W4: {e:?}\n{text}"),
    }
    assert_eq!(
        all.ok(),
        Some(w8.clone()),
        "{name} in {order}: WIDTHS\n{text}"
    );
    Some(w8)
}

/// Test 1: gf2-monomial-time.1, .2, .4 to .7. Over GF(2), every case in
/// grevlex, lex, and `block:1`, with and without cofactors, signatures on
/// and off: the call in `W4` alone returns the outcome of the call in `W8`
/// alone, or `Error::Overflow`, and the call in `WIDTHS` returns it too;
/// compared on at least nine calls in ten. In grevlex the two widths
/// record one record.
#[test]
fn every_case_in_w4_as_in_w8() {
    let (mut compared, mut total) = (0, 0);
    for (name, text) in cases() {
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
                    let case = format!("{name}, cofactors {cofactors}, signatures {signatures}");
                    if as_in_w8(&case, &text, &p).is_some() {
                        compared += 1;
                    }
                }
            }
        }
        for signatures in [true, false] {
            let mut p = params("grevlex", false);
            p.signatures = signatures;
            p.deadline = Some(CASE_DEADLINE);
            let (w8, _, r8) = heard(&text, &p, Width::W8, Width::W8);
            let (w4, _, r4) = heard(&text, &p, Width::W4, Width::W4);
            if let (Ok(w8), Ok(w4)) = (w8, w4)
                && w8.stopped.is_none()
                && w4.stopped.is_none()
            {
                assert_eq!(w4, w8, "{name}, signatures {signatures}");
                assert_eq!(r4, r8, "{name}, signatures {signatures}: the record");
            }
        }
    }
    assert!(10 * compared >= 9 * total, "{compared} of {total} compared");
}

/// The cases of test 2: an input of degree 8, which `W4` reads to its
/// line 3, and a run whose one S-pair has an lcm of degree 8.
const PAST: [(&str, Option<usize>); 2] = [
    ("x,y\n2\nx^8+y,\ny^2+x\n", Some(3)),
    ("x,y\n2\nx^4*y^3+x,\nx^3*y^4+y\n", None),
];

/// Test 2: gf2-monomial-time.3, monomial-formats.8. Over GF(2), a degree
/// past 7, in the input or in a round, is `Error::Overflow` in `W4` alone,
/// at the input's line where the input passes it; the call in `WIDTHS`
/// returns the outcome of `W8` alone, with a basis, certified where it
/// tracks cofactors, and hears the matrices of the run in `W4` before
/// those of the run in `W8`.
#[test]
fn a_degree_past_seven_starts_the_call_over() {
    for (text, line) in PAST {
        for order in ["grevlex", "lex", "block:1"] {
            for cofactors in [false, true] {
                for signatures in [true, false] {
                    let mut p = params(order, cofactors);
                    if built(&p).is_err() {
                        continue;
                    }
                    p.signatures = signatures;
                    let case = format!(
                        "{text:?} in {order}, cofactors {cofactors}, signatures {signatures}"
                    );
                    let got = alone(text, &p, Width::W4);
                    assert_eq!(got, Err(Error::Overflow { line }), "{case}");
                    let w8 = alone(text, &p, Width::W8).unwrap();
                    assert!(w8.basis.is_some(), "{case}: {w8:?}");
                    if cofactors {
                        assert_eq!(w8.certified, Some(true), "{case}: {flaw:?}", flaw = w8.flaw);
                    }
                    let (all, all_trace, _) = heard(text, &p, WIDTHS.0, WIDTHS.1);
                    assert_eq!(all.unwrap(), w8, "{case}");
                    let (_, trace4, _) = heard(text, &p, Width::W4, Width::W4);
                    let (_, trace8, _) = heard(text, &p, Width::W8, Width::W8);
                    let mut want = trace4.0;
                    want.extend(trace8.0);
                    assert_eq!(all_trace.0, want, "{case}: the matrices heard");
                }
            }
        }
    }
}

/// The cases of test 3 over GF(2), each with a pair whose lcm passes degree
/// 7, both of whose leading monomials stand in `G_min`: the criterion M drops
/// the pair of a·b·c·d and d·e·f·g·h, of lcm degree 8, for the lcm with
/// a·e divides it, and the basis is the three monomials; and a pair of lcm
/// degree 8 lies in the set while a round of degree 2 makes the unit
/// ideal's 1.
const LEFT: [(&str, Option<&str>); 2] = [
    ("a,b,c,d,e,f,g,h\n2\na*e,\na*b*c*d,\nd*e*f*g*h\n", None),
    (
        "a,b,c,d,e,f,g,h,x,y\n2\na*b*c*d+x,\nd*e*f*g*h+y,\nx*y,\nx+1,\ny+1\n",
        Some("[1]:\n"),
    ),
];

/// Test 3: gf2-monomial-time.3. Over GF(2), a plain run whose update forms
/// a pair of lcm degree 8 that no round takes returns in `W4` alone the
/// outcome of `W8` alone: the unit ideal's basis where the case names it,
/// a basis of three elements else, each certified where the call tracks
/// cofactors. A call in grevlex or lex that tracks cofactors with
/// signatures on runs the signature run in grevlex, which reduces every
/// S-pair its criteria leave, and is left out.
#[test]
fn a_pair_past_seven_that_no_round_takes() {
    for (text, basis) in LEFT {
        for order in ["grevlex", "lex", "block:1"] {
            for cofactors in [false, true] {
                for signatures in [true, false] {
                    if order != "block:1" && cofactors && signatures {
                        continue;
                    }
                    let mut p = params(order, cofactors);
                    if built(&p).is_err() {
                        continue;
                    }
                    p.signatures = signatures;
                    let case = format!(
                        "{text:?} in {order}, cofactors {cofactors}, signatures {signatures}"
                    );
                    let w4 = alone(text, &p, Width::W4).unwrap_or_else(|e| panic!("{case}: {e:?}"));
                    assert_eq!(w4, alone(text, &p, Width::W8).unwrap(), "{case}");
                    let got = w4.basis.as_deref().unwrap();
                    match basis {
                        Some(want) => assert_eq!(got, want, "{case}"),
                        None => assert_eq!(got.matches(",\n").count(), 2, "{case}: {got}"),
                    }
                    if cofactors {
                        assert_eq!(w4.certified, Some(true), "{case}: {flaw:?}", flaw = w4.flaw);
                    }
                }
            }
        }
    }
}

/// Test 4: gf2-monomial-time.2. A call over GF(7) and GF(65521) asked to
/// start in `W4` starts in `W8`: `W4` alone returns the outcome of `W8`
/// alone, on random systems and on an input of degree 10, which `W4`
/// cannot read.
#[test]
fn every_other_field_starts_in_w8() {
    for prime in [7, 65521] {
        let mut texts: Vec<String> = (0..4).map(|seed| random_system(prime, seed)).collect();
        texts.push(format!("x,y\n{prime}\nx^10-y,\ny^2-1\n"));
        for text in &texts {
            for cofactors in [false, true] {
                let mut p = params("grevlex", cofactors);
                if built(&p).is_err() {
                    continue;
                }
                p.deadline = Some(CASE_DEADLINE);
                let w8 = alone(text, &p, Width::W8).unwrap();
                if w8.stopped.is_some() {
                    continue;
                }
                assert_eq!(alone(text, &p, Width::W4).unwrap(), w8, "{text}");
            }
        }
    }
}

/// Test 5: gf2-monomial-time.1, .4, .5, .7. `Dense4`'s word operations
/// against the exponents, at its degree bound, under its borrows, in the
/// table, at the base divisor's infinity, in a row's products, and in the
/// grevlex key, as for the other formats.
#[test]
fn dense4_agrees_with_the_exponents() {
    assert_eq!(Dense4::MAX_DEGREE, 7);
    operations::<Dense4>(4);
    bound::<Dense4>();
    table_agrees::<Dense4>(14);
    infinity::<Dense4>(24);
    agree_in::<Dense4>(&[1, 16, 17, 48, 64, 65, 112, 128], 0x0b14);
    for narrow in [false, true] {
        for n in [1, 16, 17, 33, 49, 65, 97, 113, 129, 145] {
            at_the_bound::<Dense4>(n, narrow);
        }
    }
    for n in [1, 2, 3, 5, 7] {
        orders::<Dense4>(n, 4);
    }
    for n in [8, 15, 16, 17, 23, 24, 25, 33, 129] {
        key_orders(n, 0x6b34 + n as u64);
    }
    // x^2·y^3·z: the degree 6, then ~1, ~3, ~2 in a lane each.
    assert_eq!(key::<Dense4>(&[2, 3, 1]), 0x6ECD << 112);
    // x^7 at the bound, then y^0 and x^7 complemented.
    assert_eq!(key::<Dense4>(&[7, 0]), 0x7F8 << 116);
    // 25 variables, x_i of exponent 1 at i = 0 and i = 24: the degree 2,
    // ~1 for the last, then ~0 for 22 lanes; the first variable is cut.
    let mut e = [0u32; 25];
    e[0] = 1;
    e[24] = 1;
    assert_eq!(key::<Dense4>(&e), (0x2E << 88 | ((1u128 << 88) - 1)) << 32);
}

/// For drawn pairs of monomials of degree at most 7 on n variables in
/// `Dense4`, a variable's exponent 0 to 3: the smaller grevlex key is the
/// smaller monomial in grevlex, and where the key holds every variable but
/// the first, n ≤ 24, two distinct monomials' keys differ.
fn key_orders(n: usize, seed: u64) {
    let mut r = rng(seed);
    let draw = |r: &mut rand_xoshiro::Xoshiro256PlusPlus| {
        let mut e = vec![0u32; n];
        let mut left = below(r, 8);
        while left > 0 {
            let i = below(r, n as u64) as usize;
            let k = below(r, left.min(3) + 1);
            e[i] += k as u32;
            left -= k;
        }
        e
    };
    let grevlex = |a: &[u32], b: &[u32]| {
        let degree = |e: &[u32]| e.iter().sum::<u32>();
        degree(a).cmp(&degree(b)).then_with(|| {
            (0..n)
                .rev()
                .find(|&i| a[i] != b[i])
                .map_or(std::cmp::Ordering::Equal, |i| b[i].cmp(&a[i]))
        })
    };
    for _ in 0..2000 {
        let (a, b) = (draw(&mut r), draw(&mut r));
        let (ka, kb) = (key::<Dense4>(&a), key::<Dense4>(&b));
        let want = grevlex(&a, &b);
        match ka.cmp(&kb) {
            std::cmp::Ordering::Equal => assert!(
                n > 24 || want.is_eq(),
                "{n} variables: {a:?} and {b:?} share a key"
            ),
            got @ (std::cmp::Ordering::Less | std::cmp::Ordering::Greater) => {
                assert_eq!(got, want, "{n} variables: {a:?} against {b:?}");
            }
        }
    }
}

/// Test 6: gf2-monomial-time.1. In `Dense4` a lane below its divisor's
/// fails the divisibility test whatever its neighbours hold, and a borrow
/// from the lane below does not make a passing lane fail.
#[test]
fn a_borrow_in_dense4() {
    let n = 33;
    let f = Dense4::new(n);
    let unit = |pairs: &[(usize, u32)]| {
        let mut v = vec![0u32; n];
        for &(i, e) in pairs {
            v[i] = e;
        }
        v
    };
    for i in 0..n {
        let (a, _) = mono(&f, &unit(&[(i, 1)]));
        let mut around: Vec<(usize, u32)> = Vec::new();
        for j in [i.wrapping_sub(1), i + 1] {
            if j < n {
                around.push((j, 3));
            }
        }
        let (b, _) = mono(&f, &unit(&around));
        assert!(!f.divides(&a, &b), "lane {i} fails, its neighbours pass");
        around.push((i, 1));
        let (b, _) = mono(&f, &unit(&around));
        assert!(f.divides(&a, &b), "lane {i}: every lane passes");
        if i > 0 {
            let (a, _) = mono(&f, &unit(&[(i - 1, 1), (i, 1)]));
            let (c, _) = mono(&f, &unit(&[(i, 2)]));
            assert!(
                !f.divides(&a, &c),
                "lane {lane} fails below lane {i}",
                lane = i - 1
            );
        }
    }
    let mut out = vec![0u64; f.words()];
    let (x7, d7) = mono(&f, &unit(&[(0, 7)]));
    let (y1, d1) = mono(&f, &unit(&[(n - 1, 1)]));
    assert_eq!(f.mul(&x7, d7, &y1, d1, &mut out), Err(Overflow));
    assert_eq!(
        f.lcm_words_n::<{ crate::monomial::ANY }>(&x7, &y1, &mut out),
        8
    );
    assert_eq!(out, unit_words(&f, &unit(&[(0, 7), (n - 1, 1)])));
}

/// The words of exponents `e` in `f`, whatever their degree: the lcm of
/// test 6 past the bound.
fn unit_words(f: &Dense4, e: &[u32]) -> Vec<u64> {
    let mut w = vec![0u64; f.words()];
    f.pack(e, &mut w);
    w
}

/// Test 7: gf2-monomial-time.1, .2. Over GF(2), test 1's boolean systems
/// under bounds that start the call over in the signature run return in
/// `W4` alone the outcome of `W8` alone, or `Error::Overflow`; and a
/// learn's record replays in `W4` alone with the outcome it replays with
/// in `W8` alone and in `WIDTHS`, and the learn's basis.
#[test]
fn the_signature_run_and_a_replay_in_w4() {
    for (name, text) in systems(false) {
        let p = params("grevlex", false);
        for forced in [Some(1), Some(1 << 10)] {
            let (w8, w4) = forcing(forced, || {
                (alone(&text, &p, Width::W8), alone(&text, &p, Width::W4))
            });
            let w8 = w8.unwrap_or_else(|e| panic!("{name}, forced {forced:?}: {e:?}"));
            match w4 {
                Ok(o) => assert_eq!(o, w8, "{name}, forced {forced:?}"),
                Err(Error::Overflow { .. }) => {}
                Err(e) => panic!("{name}, forced {forced:?}: W4: {e:?}"),
            }
        }
        let (learned, _, record) = heard(&text, &p, WIDTHS.0, WIDTHS.1);
        let learned = learned.unwrap();
        let all = replay_in(&text, &p, &record, WIDTHS.0, WIDTHS.1, &mut NoTrace).unwrap();
        assert_eq!(all.basis, learned.basis, "{name}");
        for w in [Width::W4, Width::W8] {
            let got = replay_in(&text, &p, &record, w, w, &mut NoTrace);
            assert_eq!(got.unwrap(), all, "{name}: {w:?}");
        }
    }
}

/// Test 8: gf2-monomial-time.3. A selection in `Dense4` that takes a pair
/// of lcm degree 8 is `Limit::Overflow`, and the table interns nothing; a
/// pair of lcm degree 7 is taken, its lcm interned.
#[test]
fn a_selection_past_seven() {
    let format = Dense4::new(2);
    for (lanes, degree) in [([4, 4], 8), ([4, 3], 7)] {
        let mut table = Table::new(format, &mut rng(0x5e1));
        let mut words = vec![0u64; format.words()];
        format.pack(&lanes, &mut words);
        let mut set = PairSet::new(format.words());
        let pair = Pair {
            i: ElementId::at(0),
            j: ElementId::at(1),
            lcm: MonoId::NONE,
            degree,
            sugar: degree,
        };
        set.push(pair, format.mask(&words), &words, None);
        let before = table.len();
        let got = pairs::select(&mut set, &mut table, &Grevlex, &mut Budget::unbounded());
        if degree > Dense4::MAX_DEGREE {
            assert_eq!(got, Err(Limit::Overflow), "degree {degree}");
            assert_eq!(table.len(), before, "degree {degree}: nothing interned");
        } else {
            let got = got.unwrap();
            assert_eq!(got.len(), 1, "degree {degree}");
            assert_eq!(
                table.monomial(got[0].lcm),
                words.as_slice(),
                "degree {degree}"
            );
        }
    }
}

/// Test 9: gf2-monomial-time.3, in place of the first case of
/// `divisibility_tests.rs`'s test 6. In `Dense8`, whose degree bound is
/// 127, the update of x^100 and x^5·y^30 keeps their pair, of lcm degree
/// 130, in the set, and the selection that takes it is `Limit::Overflow`;
/// a coprime pair past the bound is dropped.
#[test]
fn an_lcm_past_the_format_waits_for_its_selection() {
    let mut e: Engine<Dense8> = Engine::new(2, 4);
    e.enter(&Grevlex, &[100, 0], 100).0.unwrap();
    let (got, _) = e.enter(&Grevlex, &[5, 30], 35);
    assert_eq!(got, Ok(()), "x^100 and x^5·y^30");
    let degrees: Vec<u32> = e.pairs.iter().map(|p| p.degree).collect();
    assert_eq!(degrees, vec![130], "the pair of lcm degree 130");
    assert_eq!(
        pairs::select(
            &mut e.pairs,
            &mut e.table,
            &Grevlex,
            &mut Budget::unbounded()
        ),
        Err(Limit::Overflow),
        "its selection"
    );

    let mut e: Engine<Dense8> = Engine::new(2, 5);
    e.enter(&Grevlex, &[100, 0], 100).0.unwrap();
    let (got, _) = e.enter(&Grevlex, &[0, 120], 120);
    assert_eq!(got, Ok(()), "x^100 and y^120: coprime");
    assert!(e.pairs.is_empty(), "the coprime pair is dropped");
}
