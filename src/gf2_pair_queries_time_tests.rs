//! gf2-pair-queries-time, the item's test code: the intersection of the
//! pair set's index visits the places a scan of the supports names, under
//! bounds at every place of the last word and past the index's words; the
//! update keeps the pairs of joux-2011's Algorithm 2 in formats of one to
//! eight words and past eight, with the index and without; and a call in
//! those formats returns with the index the outcome it returns without it,
//! certified where it tracks cofactors, on `OBJECTIVES.md`'s edge cases too
//! (`iterations/0179-gf2-pair-queries-time/plan.md`, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::bitmap::Bitmap;
use crate::budget::{Budget, Site};
use crate::call::built;
use crate::divisibility_tests::{Engine, Reference};
use crate::format_tests::params;
use crate::gf2_update_rest_time_tests::{as_without, spread, systems_on};
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format, SUPPORT_BITS};
use crate::order::Grevlex;
use crate::pairs::INDEX_FORCED;
use crate::shared_tests::{CASE_DEADLINE, below, rng};
use crate::split_piece_time_tests::EDGE;
use crate::systems::placed;

/// The places `holding` visits on `set` for `query` below `bound`.
fn visited(set: &Bitmap, query: u128, bound: usize, buffer: &mut Vec<u64>) -> Vec<usize> {
    let mut got = Vec::new();
    set.holding(
        query,
        bound,
        buffer,
        1,
        &mut Budget::unbounded(),
        Site::Update,
        |k| got.push(k),
    )
    .unwrap();
    got
}

/// The bounds test 1 visits below on an index over `covered` places: the
/// first places, every place of the last word the index covers and of the
/// word after it, and bounds past the index's words, a word's places apart
/// and not.
fn bounds(covered: usize) -> Vec<usize> {
    let last = (covered - 1) / 64 * 64;
    let mut b: Vec<usize> = [0, 1, 63, 64, 65].into_iter().collect();
    b.extend(last..=last + 128);
    b.extend([covered + 200, covered + 4_096, covered + 16_449]);
    b.sort_unstable();
    b.dedup();
    b
}

/// Test 1: gf2-pair-queries-time.1. The index over 1 to 20,000 places,
/// every boundary of a word and of a chunk of 256 words among them, its
/// supports of 1 to 128 bits drawn sparse, or every one all the index's
/// bits so that each word of the intersection is full: the places the
/// intersection visits for a query below a bound are those a scan of the
/// supports names, increasing, under every bound of the last word the index
/// covers and of the word after it, and under bounds past the index's
/// words, which no place there holds; a query of no bit visits every place
/// below the bound.
#[test]
fn the_intersection_visits_what_a_scan_names() {
    let mut r = rng(0x9a1e);
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
                let mut queries = vec![0, 1, top];
                for _ in 0..usize::from(!full) * 3 {
                    queries.push(supports[below(&mut r, covered as u64) as usize]);
                    queries.push((0..2).fold(0u128, |s, _| s | 1 << below(&mut r, bits as u64)));
                }
                queries.sort_unstable();
                queries.dedup();
                for &query in &queries {
                    let holding: Vec<usize> = (0..covered)
                        .filter(|&k| supports[k] & query == query)
                        .collect();
                    for bound in bounds(covered) {
                        let want: Vec<usize> = if query == 0 {
                            (0..bound).collect()
                        } else {
                            holding.iter().copied().take_while(|&k| k < bound).collect()
                        };
                        let got = visited(&set, query, bound, &mut buffer);
                        assert_eq!(
                            got, want,
                            "{bits} bits, {covered} places, full {full}, {query:#x}, below {bound}"
                        );
                    }
                }
            }
        }
    }
}

/// The first and the last count of variables on which format M takes w
/// words, between 1 and 400.
pub(crate) fn spans<M: Format>(w: usize) -> (usize, usize) {
    let ns: Vec<usize> = (1..=400).filter(|&n| M::new(n).words() == w).collect();
    (ns[0], ns[ns.len() - 1])
}

/// Enters `steps` drawn leading monomials of degree 1 to 4, at most what n
/// variables hold, and exponents at most `most` into an engine on n variables in format M, its pair set's
/// index forced on or off by `index`, and into joux-2011's Algorithm 2
/// (`Reference`): after every step the pair set and `G_min` are the
/// reference's. Half a lead's variables come from the first 12, so that the
/// leads share variables. The pairs the set holds at the end.
pub(crate) fn against_algorithm_2<M: Format>(
    n: usize,
    index: bool,
    seed: u64,
    steps: usize,
    most: u32,
) -> usize {
    let mut engine: Engine<M> = Engine::new(n, seed);
    let mut reference = Reference::new(true, M::MAX_DEGREE);
    let mut r = rng(seed ^ 0x2a1);
    let hot = n.min(12);
    for step in 0..steps {
        let mut lead = vec![0u32; n];
        // A degree the n variables hold at `most` each.
        let d = 1 + below(&mut r, (n as u64 * u64::from(most)).min(4)) as u32;
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
        let case = format!("{n} variables, index {index}, seed {seed:#x}, step {step}: {lead:?}");
        INDEX_FORCED.set(Some(index));
        let (got, _) = engine.enter(&Grevlex, &lead, sugar);
        INDEX_FORCED.set(None);
        got.unwrap_or_else(|e| panic!("{case}: {e:?}"));
        reference
            .update(lead.clone(), sugar)
            .unwrap_or_else(|e| panic!("{case}: the reference, {e:?}"));
        assert_eq!(engine.written(), reference.pairs, "{case}: the pair set");
        let min: Vec<u32> = engine
            .min_members()
            .iter()
            .map(|i| i.index() as u32)
            .collect();
        assert_eq!(min, reference.min_members(), "{case}: G_min");
    }
    reference.pairs.len()
}

/// Test 2 in format M: one to nine words, on the fewest and the most
/// variables of each count, with the index and without.
fn every_word_count<M: Format>(seed: u64, most: u32) {
    for w in 1..=9 {
        let (lo, hi) = spans::<M>(w);
        for n in [lo, hi] {
            for index in [true, false] {
                let held = against_algorithm_2::<M>(n, index, seed ^ ((n as u64) << 8), 60, most);
                assert!(
                    held > 0,
                    "{n} variables in {w} words, index {index}: the set holds pairs"
                );
            }
        }
    }
}

/// Test 2: gf2-pair-queries-time.2, .3. The update keeps the pairs of
/// joux-2011's Algorithm 2 (`divisibility_tests::Reference`) and its
/// `G_min` in `Dense4`, `Dense8`, `Dense16`, and `Dense32`, in every count
/// of words from one to eight, which the update's copies take, and in
/// nine, which the copy for any count takes, on the fewest and the most
/// variables of each count, past `SUPPORT_BITS` where the supports fold,
/// with the pair set's index and without, on leads of one exponent to a
/// variable and of squares.
#[test]
fn the_update_keeps_algorithm_2s_pairs_in_every_word_count() {
    every_word_count::<Dense4>(0x4a, 1);
    every_word_count::<Dense4>(0x4b, 2);
    every_word_count::<Dense8>(0x8a, 2);
    every_word_count::<Dense16>(0x16a, 3);
    every_word_count::<Dense32>(0x32a, 3);
}

/// Test 3: gf2-pair-queries-time.1 to .3. Systems over GF(2), GF(7),
/// GF(65521), and ℚ placed on 10, 20, 40, 70, and 129 variables, which
/// the call's formats take in one to nine words, in grevlex and lex, with
/// and without cofactors, the signatures on and off; and `OBJECTIVES.md`'s
/// edge cases, the zero ideal, F = (0), no generator, the unit ideal, an
/// input already a Gröbner basis, the zero polynomial among the generators,
/// and an eliminable variable, on x, y, z placed among 3, 20, 70, and 129
/// variables over the same fields: the call with the index returns the
/// outcome of the call without it, certified by the exact checker where it
/// tracks cofactors; the edge cases' bases are `[]:` for the zero ideal and
/// `[1]:` for the unit ideal alone. At least nine calls in ten of the
/// systems are compared.
#[test]
fn calls_with_and_without_the_index_in_every_word_count() {
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
