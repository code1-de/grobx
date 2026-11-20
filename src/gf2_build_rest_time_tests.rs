//! gf2-build-rest-time, the item's test code: a row's products by
//! `Table::row_products` where every other row's multiplier is the unit
//! monomial against the lookup per term on a twin table; and calls over
//! GF(2), GF(65521), and ℚ, whose rounds take rows by the unit monomial,
//! return the basis the exact checker certifies
//! (`iterations/0161-gf2-build-rest-time/plan.md`, _Test code_, 1 and 2).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::build_tests::agree_units;
use crate::gf2_bound_losses_tests::{certified, systems};
use crate::gf2_start_over_tests::{call, reported};
use crate::monomial::{Dense4, Dense8, Dense16, Dense32};
use crate::shared_tests::P;
use crate::split_piece_time_tests::EDGE;
use crate::systems::{cubics, dense, random_sparse, random_square, random_system};

/// Test 1, gf2-build-rest-time.1: a row's products by `row_products`,
/// every other row by the unit monomial, are those of the lookup per term
/// on a twin table, in `Dense4`, `Dense8`, `Dense16`, and `Dense32`, in one
/// word, in eight, which the last copy of the lookup takes, and past eight,
/// which the lookup per term takes, whatever the hashes, into columns of 4
/// bytes and of 2.
#[test]
fn unit_rows_agree_with_the_lookup_per_term() {
    // Dense4: 16 lanes a word, 1 to 27 words.
    agree_units::<Dense4>(&[1, 16, 17, 128, 129, 424], 0x0161_0b01, true);
    // Dense8: 8 lanes a word, 1 to 75 words.
    agree_units::<Dense8>(&[1, 8, 9, 64, 65, 600], 0x0161_0b02, true);
    // Dense16: 4 lanes a word, 1 to 75 words.
    agree_units::<Dense16>(&[1, 4, 5, 32, 33, 300], 0x0161_0b03, true);
    // Dense32: 2 lanes a word, 1 to 70 words.
    agree_units::<Dense32>(&[1, 2, 3, 16, 17, 140], 0x0161_0b04, true);
}

/// The systems of test 2 over `p`, 65521 or 0 for ℚ: random systems and
/// random square systems, seeds 0 to 5, dense cubics on three variables,
/// seeds 0 and 1, random sparse systems of 3 and 4 variables, and the edge
/// cases.
fn others(p: u64) -> Vec<(String, String)> {
    let mut texts = Vec::new();
    for seed in 0..6 {
        texts.push((
            format!("random_system({p}, {seed})"),
            random_system(p, seed),
        ));
        texts.push((
            format!("random_square({p}, {seed})"),
            random_square(p, seed),
        ));
    }
    for seed in 0..2 {
        texts.push((format!("dense(3, 3, {p}, {seed})"), dense(3, 3, p, seed)));
    }
    for n in [3, 4] {
        texts.push((
            format!("random_sparse({n}, {p})"),
            random_sparse(n, p, 0x0161 + n as u64),
        ));
    }
    for (k, gens) in EDGE.iter().enumerate() {
        texts.push((
            format!("edge case {k} over {p}"),
            format!("x,y,z\n{p}\n{gens}\n"),
        ));
    }
    texts
}

/// Test 2, gf2-build-rest-time.1 and .2: over GF(2), boolean systems of 9
/// to 13 variables, the edge cases, and cubic systems of 8 to 10
/// variables; over GF(65521) and ℚ, the systems of `others`: each call,
/// with the signatures on and off, returns the basis that the exact checker
/// certifies on the call with cofactors.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn calls_with_unit_rows_return_every_basis() {
    let mut texts = systems(true);
    texts.extend((8..=10usize).map(|n| {
        let m = 2 * n;
        (format!("cubics({n}, {m})"), cubics(n, m, 0x0161 + n as u64))
    }));
    texts.extend(others(P));
    texts.extend(others(0));
    for (i, (name, text)) in texts.iter().enumerate() {
        let seed = 0x0161_0c00 ^ i as u64;
        let want = certified(name, text, seed);
        for signatures in [false, true] {
            let (basis, _) = call(text, &reported(seed, signatures), None);
            assert_eq!(basis, want, "{name}, signatures {signatures}");
        }
    }
}
