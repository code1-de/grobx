//! column-index-peak, sub-item 1, the crate's test code: the field
//! contract's row operations, its band operations, and the marks read
//! columns of 2 bytes as they read `MonoId` columns
//! (`iterations/0115-column-index-peak/plan.md`, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow columns below 2^16 and draws"
)]

use std::collections::BTreeSet;

use crate::field::{BAND, Field, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::id::MonoId;
use crate::matrix;
use crate::systems::Rng;

/// The dense rows' width: every column a `u16` holds.
const WIDTH: usize = 1 << 16;

/// The rows of a case: the first loaded, the others subtracted.
const ROWS: usize = 3;

/// The cases a field takes.
const CASES: usize = 20;

/// The most columns of a row.
const MOST: u64 = 300;

/// `len` distinct columns below `WIDTH`, increasing; where `edges` holds,
/// the first and the last column of the width among them, and `len` at
/// least 2.
fn columns(rng: &mut Rng, len: usize, edges: bool) -> Vec<usize> {
    let mut set = BTreeSet::new();
    if edges {
        set.insert(0);
        set.insert(WIDTH - 1);
    }
    while set.len() < len {
        set.insert(rng.below(WIDTH as u64) as usize);
    }
    set.into_iter().collect()
}

/// A case's rows: their columns as `u16`, as `MonoId`, and their
/// coefficients over `field`, every fourth case with the edges' columns.
#[allow(
    clippy::type_complexity,
    reason = "the three parallel vectors of a case's rows"
)]
fn rows<F: Field>(
    field: &F,
    rng: &mut Rng,
    case: usize,
) -> (Vec<Vec<u16>>, Vec<Vec<MonoId>>, Vec<Vec<F::Coef>>) {
    let edges = case.is_multiple_of(4);
    let (mut narrow, mut wide, mut coefs) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ROWS {
        let len = usize::from(edges) + 1 + rng.below(MOST) as usize;
        let cols = columns(rng, len, edges);
        narrow.push(cols.iter().map(|&c| c as u16).collect());
        wide.push(cols.iter().map(|&c| MonoId::at(c)).collect());
        coefs.push(cols.iter().map(|_| field.coef(rng.next())).collect());
    }
    (narrow, wide, coefs)
}

/// Over `field`, `CASES` cases: the first row loaded and the others
/// subtracted by `sub_mul`, each with one multiplier, through `u16`
/// columns into one dense row and through `MonoId` columns into another.
/// The two are equal at every column, read by `take`.
fn the_ops_agree<F: Field>(name: &str, field: &F, seed: u64) {
    let mut rng = Rng::new(seed);
    for case in 0..CASES {
        let (narrow, wide, coefs) = rows(field, &mut rng, case);
        let mut a = vec![F::Acc::default(); WIDTH];
        let mut b = vec![F::Acc::default(); WIDTH];
        // SAFETY: every column lies below `WIDTH`, the dense rows' length,
        // and a row's columns and coefficients have one length.
        unsafe {
            field.load(&mut a, &narrow[0], &coefs[0]);
            field.load(&mut b, &wide[0], &coefs[0]);
            for k in 1..ROWS {
                let c = field.coef(rng.next());
                field.sub_mul(&mut a, &narrow[k], &coefs[k], c);
                field.sub_mul(&mut b, &wide[k], &coefs[k], c);
            }
        }
        for column in 0..WIDTH {
            assert_eq!(
                field.take(&mut a[column]),
                field.take(&mut b[column]),
                "{name}, case {case}, column {column}"
            );
        }
    }
}

#[test]
fn the_row_operations_read_two_byte_columns() {
    the_ops_agree("Gf31 at 65521", &Gf31::new(65521).unwrap(), 1);
    the_ops_agree("Gf31 at 2^31 - 1", &Gf31::new(2_147_483_647).unwrap(), 2);
    the_ops_agree("Gf64 past 2^31", &Gf64::new(2_147_483_659).unwrap(), 3);
    the_ops_agree(
        "Gf64 past 2^62",
        &Gf64::new(9_223_372_036_854_775_783).unwrap(),
        4,
    );
    the_ops_agree(
        "Gf64 past 2^63",
        &Gf64::new(18_446_744_073_709_551_557).unwrap(),
        5,
    );
    the_ops_agree(
        "Gf31Pair",
        &Gf31Pair::primes(2_147_483_647, 2_147_483_629),
        6,
    );
    the_ops_agree(
        "Gf64Pair",
        &Gf64Pair::primes(4_611_686_018_427_387_847, 4_611_686_018_427_387_817),
        7,
    );
}

/// Over `field`, `CASES` cases: every band row loaded by `load_band`, then
/// a row subtracted from every band row by `sub_mul_band` and one from a
/// single band row by `sub_mul_band_row`, through `u16` columns into one
/// band and through `MonoId` columns into another. The two are equal at
/// every column and band row, read by `take`.
fn the_band_ops_agree(name: &str, field: &Gf31, seed: u64) {
    let mut rng = Rng::new(seed);
    for case in 0..CASES {
        let mut a = vec![[0i64; BAND]; WIDTH];
        let mut b = vec![[0i64; BAND]; WIDTH];
        for k in 0..BAND {
            let (narrow, wide, coefs) = rows(field, &mut rng, case);
            // SAFETY: every column lies below `WIDTH`, the bands' length,
            // a row's columns and coefficients have one length, and k <
            // BAND.
            unsafe {
                field.load_band(&mut a, k, &narrow[0], &coefs[0]);
                field.load_band(&mut b, k, &wide[0], &coefs[0]);
            }
        }
        let (narrow, wide, coefs) = rows(field, &mut rng, case);
        let c: [u32; BAND] = std::array::from_fn(|_| field.coef(rng.next()));
        let k = rng.below(BAND as u64) as usize;
        let single = field.coef(rng.next());
        // SAFETY: as above.
        unsafe {
            field.sub_mul_band(&mut a, &narrow[1], &coefs[1], &c);
            field.sub_mul_band(&mut b, &wide[1], &coefs[1], &c);
            field.sub_mul_band_row(&mut a, k, &narrow[2], &coefs[2], single);
            field.sub_mul_band_row(&mut b, k, &wide[2], &coefs[2], single);
        }
        for column in 0..WIDTH {
            for row in 0..BAND {
                assert_eq!(
                    field.take(&mut a[column][row]),
                    field.take(&mut b[column][row]),
                    "{name}, case {case}, column {column}, band row {row}"
                );
            }
        }
    }
}

#[test]
fn the_band_operations_read_two_byte_columns() {
    the_band_ops_agree("Gf31 at 65521", &Gf31::new(65521).unwrap(), 11);
    the_band_ops_agree("Gf31 at 2^31 - 1", &Gf31::new(2_147_483_647).unwrap(), 12);
}

#[test]
fn the_marks_read_two_byte_columns() {
    let field = Gf31::new(65521).unwrap();
    let mut rng = Rng::new(21);
    for case in 0..CASES {
        let (narrow, wide, _) = rows(&field, &mut rng, case);
        let mut a = vec![0u64; WIDTH / 64];
        let mut b = vec![0u64; WIDTH / 64];
        for k in 0..ROWS {
            matrix::mark(&mut a, &narrow[k]);
            matrix::mark(&mut b, &wide[k]);
        }
        assert_eq!(a, b, "case {case}: the marks");
    }
}
