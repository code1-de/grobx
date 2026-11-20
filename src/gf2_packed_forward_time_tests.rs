//! gf2-packed-forward-time, the item's test code: plain rounds in packed
//! rows whose K columns take one word to 24 words, their rows from sparse
//! to every column, give the outcome the reduction in sparse rows gives;
//! and calls over GF(2) whose rounds reduce in packed rows, in sparse rows,
//! or as the bounds decide, with the signatures on and off, return the
//! basis the exact checker certifies
//! (`iterations/0160-gf2-packed-forward-time/plan.md`, _Test code_, 1 and
//! 2).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::gf2_bound_losses_tests::{certified, systems};
use crate::gf2_start_over_tests::{call, reported};
use crate::packed_reduction_tests::{Plain, forced, outcome};
use crate::shared_tests::budget;
use crate::systems::cubics;

/// Test 1's K columns: a word less one, a word, and a word and one; two
/// words less one, two, and two and one; and 10, about 16, and 24 words.
const SPANS: [usize; 9] = [63, 64, 65, 127, 128, 129, 640, 1000, 1536];

/// Test 1, gf2-packed-forward-time.1: `matrix::reduce` over `Gf2` with
/// `PACK_FORCED` on and off, on plain rounds of each K of `SPANS`, of
/// sparse tails, of tails of half the columns, and of tails of every
/// column, whose reduced rows and packed rows then reach from the K
/// columns' first word to their last, with the back-substitution and
/// without: one outcome.
#[test]
fn rows_of_every_span() {
    for (i, &k) in SPANS.iter().enumerate() {
        for seed in 0..3u64 {
            for density in [1, 4, 8] {
                let m = Plain::new(
                    0x0160_f000 + 16 * i as u64 + seed,
                    40 + 7 * seed as usize,
                    k,
                    24,
                    density,
                );
                for full in [false, true] {
                    let case = format!("k {k}, seed {seed}, density {density}, full {full}");
                    let sparse = forced(false, || m.reduce(full, &mut budget())).unwrap();
                    let packed = forced(true, || m.reduce(full, &mut budget())).unwrap();
                    assert!(
                        sparse.packed.is_none() && packed.packed.is_some(),
                        "{case}: the round packs where it is forced to"
                    );
                    assert_eq!(outcome(&packed), outcome(&sparse), "{case}");
                }
            }
        }
    }
}

/// Test 2, gf2-packed-forward-time.1: over GF(2), boolean systems of 9 to
/// 13 variables, the edge cases, and cubic systems of 8 to 10 variables,
/// each call with every round in packed rows, with none, and as the bounds
/// decide, with the signatures on and off, returns the basis that the
/// exact checker certifies on the call with cofactors.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn calls_in_packed_rows_return_every_basis() {
    let mut texts = systems(true);
    texts.extend((8..=10usize).map(|n| {
        let m = 2 * n;
        (format!("cubics({n}, {m})"), cubics(n, m, 0x0160 + n as u64))
    }));
    for (i, (name, text)) in texts.iter().enumerate() {
        let seed = 0x0161 ^ i as u64;
        let want = certified(name, text, seed);
        for signatures in [false, true] {
            for packs in [Some(true), Some(false), None] {
                let case = format!("{name}, signatures {signatures}, packs {packs:?}");
                let params = reported(seed, signatures);
                let (basis, _) = match packs {
                    Some(p) => forced(p, || call(text, &params, None)),
                    None => call(text, &params, None),
                };
                assert_eq!(basis, want, "{case}");
            }
        }
    }
}
