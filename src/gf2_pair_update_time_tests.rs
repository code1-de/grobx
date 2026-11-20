//! gf2-pair-update-time, the item's test code: a visit of `G_min`'s index
//! with places left out and added back names the places a scan of the
//! supports names, less those and with these; the update keeps the pairs of
//! joux-2011's Algorithm 2 with the kept colons of one variable found first
//! at every update, at none, and as the count of the update before rules, in
//! formats of one to eight words and past eight, with the index and without;
//! and a call returns with the colons found first the outcome it returns
//! without them, certified where it tracks cofactors, on `OBJECTIVES.md`'s
//! edge cases too (`iterations/0180-gf2-pair-update-time/plan.md`, _Test
//! code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::bitmap::Bitmap;
use crate::budget::{Budget, Site};
use crate::call::built;
use crate::format_tests::{call, params};
use crate::gf2_pair_queries_time_tests::{against_algorithm_2, spans};
use crate::gf2_update_rest_time_tests::{spread, systems_on};
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format, SUPPORT_BITS, WIDTHS};
use crate::pairs::COLONS_FIRST;
use crate::shared_tests::{CASE_DEADLINE, below, rng};
use crate::split_piece_time_tests::EDGE;
use crate::systems::placed;
use crate::{Error, Outcome, Params};

/// The places `meeting_less` visits on `set` for `query`, less those of
/// `less`, with `extra` added back.
fn visited(
    set: &Bitmap,
    query: u128,
    less: u128,
    extra: &[usize],
    buffer: &mut Vec<u64>,
) -> Vec<usize> {
    let mut got = Vec::new();
    set.meeting_less(
        query,
        less,
        extra,
        buffer,
        1,
        &mut Budget::unbounded(),
        Site::Update,
        |k| got.push(k),
    )
    .unwrap();
    got
}

/// Up to two bits below `bits`, drawn.
fn two(r: &mut Xoshiro256PlusPlus, bits: usize) -> u128 {
    (0..2).fold(0u128, |s, _| s | 1 << below(r, bits as u64))
}

/// Test 1: gf2-pair-update-time.2. The index over 1 to 20,000 places,
/// every boundary of a word and of a chunk of 256 words among them, its
/// supports of 1 to 128 bits drawn sparse, or every one all the index's
/// bits: the places `meeting_less` visits for a query, less the places
/// whose support holds a bit of `less`, with drawn places added back, are
/// those a scan of the supports names, increasing, each once: a place the
/// query names and `less` leaves out, one it names and keeps, one it does
/// not name, and the first and the last place among those added back. With
/// no bit left out and no place added, the places `meeting` visits; with no
/// bit in the query, the places added back alone.
#[test]
fn meeting_less_names_what_a_scan_names_and_added_back() {
    let mut r = rng(0x9a2f);
    let mut buffer = Vec::new();
    for bits in [1, 5, 64, 127, 128] {
        let top: u128 = if bits == SUPPORT_BITS {
            u128::MAX
        } else {
            (1 << bits) - 1
        };
        for covered in [
            1, 2, 63, 64, 65, 127, 128, 129, 16_383, 16_384, 16_385, 16_447, 16_448, 16_449, 20_000,
        ] {
            for full in [false, true] {
                let supports: Vec<u128> = (0..covered)
                    .map(|_| {
                        if full {
                            return top;
                        }
                        if below(&mut r, 3) == 0 {
                            return 0;
                        }
                        (0..3).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64)) & top
                    })
                    .collect();
                let mut set = Bitmap::new(bits);
                for (k, &s) in supports.iter().enumerate() {
                    if s != 0 || k + 1 == covered {
                        set.set(k, s);
                    }
                }
                let mut queries = vec![(0, 0), (1, 0), (top, 0), (1, top)];
                for _ in 0..3 {
                    let q = supports[below(&mut r, covered as u64) as usize] | two(&mut r, bits);
                    queries.push((q, 0));
                    queries.push((q, two(&mut r, bits) & !q));
                    queries.push((q, two(&mut r, bits)));
                }
                for &(query, less) in &queries {
                    let named: Vec<usize> = (0..covered)
                        .filter(|&k| supports[k] & query != 0 && supports[k] & less == 0)
                        .collect();
                    let mut extra: Vec<usize> =
                        (0..covered).filter(|_| below(&mut r, 16) == 0).collect();
                    if covered > 1 {
                        extra.extend([0, covered - 1]);
                    }
                    extra.sort_unstable();
                    extra.dedup();
                    let case = format!(
                        "{bits} bits, {covered} places, full {full}, {query:#x} less {less:#x}"
                    );
                    let got = visited(&set, query, less, &[], &mut buffer);
                    assert_eq!(got, named, "{case}, none added");
                    if less == 0 {
                        let mut met = Vec::new();
                        set.meeting(query, 1, &mut Budget::unbounded(), Site::Update, |k| {
                            met.push(k);
                        })
                        .unwrap();
                        assert_eq!(got, met, "{case}: as meeting");
                    }
                    let mut want = named.clone();
                    want.extend(&extra);
                    want.sort_unstable();
                    want.dedup();
                    let got = visited(&set, query, less, &extra, &mut buffer);
                    assert_eq!(got, want, "{case}, {n} added", n = extra.len());
                }
            }
        }
    }
}

/// Test 2 in format M: one to nine words, on the fewest and the most
/// variables of each count, with the index and without, the kept colons
/// found first at every update and at none over 60 leads, and as the count
/// of the update before rules over 200, where P1 may pass `FIRST_LEAST`.
fn every_word_count<M: Format>(seed: u64, most: u32) {
    for (first, steps) in [(Some(true), 60), (Some(false), 60), (None, 200)] {
        COLONS_FIRST.set(first);
        for w in 1..=9 {
            let (lo, hi) = spans::<M>(w);
            for n in [lo, hi] {
                for index in [true, false] {
                    let held =
                        against_algorithm_2::<M>(n, index, seed ^ ((n as u64) << 8), steps, most);
                    assert!(
                        held > 0,
                        "{n} variables in {w} words, index {index}, first {first:?}: pairs held"
                    );
                }
            }
        }
        COLONS_FIRST.set(None);
    }
}

/// Test 2: gf2-pair-update-time.1, .3 to .5. The update keeps the pairs of
/// joux-2011's Algorithm 2 (`divisibility_tests::Reference`) and its
/// `G_min` in `Dense4`, `Dense8`, `Dense16`, and `Dense32`, in every count
/// of words from one to eight, which the update's copies take, and in nine,
/// which the copy for any count takes, on the fewest and the most variables
/// of each count, past `SUPPORT_BITS` where the supports fold and the keys
/// hold no colons, with the pair set's index and without, on leads of one
/// exponent to a variable, where a colon's one bit outside the lead is a
/// variable of exponent one, and of squares, where a colon of one bit may
/// have a degree past one; the kept colons of one variable found first at
/// every update whose keys hold colons, at none, and as `FIRST_LEAST` rules.
#[test]
fn the_update_keeps_algorithm_2s_pairs_with_the_colons_first() {
    every_word_count::<Dense4>(0x4c, 1);
    every_word_count::<Dense4>(0x4d, 2);
    every_word_count::<Dense8>(0x8c, 2);
    every_word_count::<Dense16>(0x16c, 3);
    every_word_count::<Dense32>(0x32c, 3);
}

/// The outcome of the call in `WIDTHS` with the kept colons of one
/// variable found first at every update, which the call that finds them
/// first at none and the call that leaves them to the count of the update
/// before return too; with cofactors certified. None where the deadline
/// stopped a call.
fn as_without(name: &str, text: &str, p: &Params) -> Option<Outcome> {
    let mut got: Vec<Result<Outcome, Error>> = [Some(true), Some(false), None]
        .into_iter()
        .map(|forced| {
            COLONS_FIRST.set(forced);
            let o = call(text, p, WIDTHS.0, WIDTHS.1);
            COLONS_FIRST.set(None);
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
    for (forced, other) in ["at none", "by the count"].iter().zip(got) {
        assert_eq!(other, on, "{name} in {order}: first and {forced}\n{text}");
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

/// Test 3: gf2-pair-update-time.1 to .5. Systems over GF(2), GF(7),
/// GF(65521), and ℚ placed on 10, 20, 40, 70, and 129 variables, which the
/// call's formats take in one to nine words, where the pair set indexes
/// past 32 and its supports fold past `SUPPORT_BITS`, in grevlex and lex,
/// with and without cofactors, the signatures on and off; and
/// `OBJECTIVES.md`'s edge cases, the zero ideal, F = (0), no generator, the
/// unit ideal, an input already a Gröbner basis, the zero polynomial among
/// the generators, and an eliminable variable, on x, y, z placed among 3,
/// 20, 70, and 129 variables over the same fields: the call with the kept
/// colons found first at every update returns the outcome of the call that
/// finds them first at none and of the call that leaves them to the count,
/// certified by the exact checker where it tracks cofactors; the edge
/// cases' bases are `[]:` for the zero ideal and `[1]:` for the unit ideal
/// alone. At least nine calls in ten of the systems are compared.
#[test]
fn calls_with_and_without_the_colons_first() {
    let (mut compared, mut total) = (0, 0);
    for n in [10, 20, 40, 70, 129] {
        for (name, text) in systems_on(n) {
            for order in ["grevlex", "lex"] {
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
    for n in [3, 20, 70, 129] {
        for p in [2, 7, 65521, 0] {
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
