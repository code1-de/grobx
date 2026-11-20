//! gf2-reduction-time, sub-item 1's test code: `Gf2`'s arithmetic, its row
//! operations and band operations against sums mod 2, and its reductions
//! against `Gf31`'s at p = 2 (`iterations/0125-gf2-reduction-time/plan.md`,
//! _Test code_, 5 and 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::band_tests::Random;
use crate::cofactors::NoCofactors;
use crate::field::{BAND, Field, Gf2, Gf31};
use crate::id::{Column, MonoId};
use crate::matrix::{self, Buffers, Reduction};
use crate::module::Rank1;
use crate::monomial::Dense16;
use crate::packed::PACK_FORCED;
use crate::shared_tests::budget;
use crate::signature_zero_tests::{Made, Round, made};
use crate::systems::Rng;

/// The cases of each part of test 5.
const CASES: u64 = 20;

/// The operations of each case of test 5.
const STEPS: u64 = 200;

/// The widest dense row of test 5.
const WIDTH: u64 = 300;

/// A random sparse row of width `width` over GF(2): increasing columns,
/// each kept with probability 1/4, every coefficient 1.
fn sparse<I: Column + Copy>(
    rng: &mut Rng,
    width: usize,
    at: impl Fn(usize) -> I,
) -> (Vec<I>, Vec<u8>) {
    let cols: Vec<I> = (0..width).filter(|_| rng.below(4) == 0).map(at).collect();
    let coefs = vec![1u8; cols.len()];
    (cols, coefs)
}

/// Test 5, part 1: `Gf2`'s arithmetic over {0, 1}, the residues of 0 to 9,
/// and of two decimals of 40 digits.
#[test]
fn the_field_of_two_elements() {
    let f = Gf2::new();
    assert_eq!(f.characteristic(), 2);
    for v in 0..10u64 {
        assert_eq!(f.value(f.coef(v)), v % 2, "coef({v})");
    }
    assert_eq!(
        f.value(f.decimal("1234567890123456789012345678901234567891")),
        1
    );
    assert_eq!(
        f.value(f.decimal("1234567890123456789012345678901234567890")),
        0
    );
    for a in [0u8, 1] {
        assert_eq!(f.neg(a), a, "neg({a})");
        assert_eq!(f.inv(a), (a == 1).then_some(1), "inv({a})");
        for b in [0u8, 1] {
            assert_eq!(f.add(a, b), a ^ b, "{a} + {b}");
            assert_eq!(f.sub(a, b), a ^ b, "{a} − {b}");
            assert_eq!(f.mul(a, b), a & b, "{a} · {b}");
        }
    }
}

/// Test 5, part 2: `load`, `sub_mul`, and `take` on a dense row against a
/// row of sums, for columns of `MonoId` and of `u16`.
#[test]
fn the_row_operations_add_mod_2() {
    fn check<I: Column + Copy>(seed: u64, at: impl Fn(usize) -> I + Copy) {
        let f = Gf2::new();
        let mut rng = Rng::new(seed);
        for case in 0..CASES {
            let width = 1 + rng.below(WIDTH) as usize;
            let mut dense = vec![0u8; width];
            let mut sums = vec![0u64; width];
            // Whether the dense row holds nothing since its last read.
            let mut fresh = true;
            for step in 0..STEPS {
                let (cols, coefs) = sparse(&mut rng, width, at);
                match rng.below(3) {
                    0 if fresh => {
                        // SAFETY: every column lies below the width.
                        unsafe { f.load(&mut dense, &cols, &coefs) };
                        for c in &cols {
                            sums[c.column()] += 1;
                        }
                        fresh = false;
                    }
                    0 | 1 => {
                        let c = rng.below(2) as u8;
                        // SAFETY: as above.
                        unsafe { f.sub_mul(&mut dense, &cols, &coefs, c) };
                        for col in &cols {
                            sums[col.column()] += u64::from(c);
                        }
                        fresh = false;
                    }
                    _ => {
                        for (k, e) in dense.iter_mut().enumerate() {
                            let v = f.take(e);
                            assert_eq!(
                                u64::from(v),
                                sums[k] % 2,
                                "case {case}, step {step}, column {k}"
                            );
                            assert_eq!(*e, 0, "case {case}, step {step}, column {k}");
                            sums[k] = 0;
                        }
                        fresh = true;
                    }
                }
            }
        }
    }
    check(0x9f70, MonoId::at);
    check(0x9f71, |c| c as u16);
}

/// Test 5, part 3: the band operations against four dense rows of their
/// own, as `band_tests` checks `Gf31`'s.
#[test]
fn the_band_operations_add_mod_2() {
    let f = Gf2::new();
    let mut rng = Rng::new(0x9f72);
    for case in 0..CASES {
        let width = 1 + rng.below(WIDTH) as usize;
        let mut band = vec![[0u8; BAND]; width];
        let mut dense: [Vec<u8>; BAND] = std::array::from_fn(|_| vec![0u8; width]);
        let mut fresh = [true; BAND];
        let read = |band: &mut [[u8; BAND]], dense: &mut [Vec<u8>; BAND]| {
            for (c, entry) in band.iter_mut().enumerate() {
                for k in 0..BAND {
                    assert_eq!(
                        f.take(&mut entry[k]),
                        f.take(&mut dense[k][c]),
                        "case {case}, column {c}, band row {k}"
                    );
                }
            }
        };
        for _ in 0..STEPS {
            let (cols, coefs) = sparse(&mut rng, width, MonoId::at);
            match rng.below(4) {
                0 => {
                    let k = rng.below(BAND as u64) as usize;
                    if fresh[k] {
                        // SAFETY: every column lies below the width.
                        unsafe {
                            f.load_band(&mut band, k, &cols, &coefs);
                            f.load(&mut dense[k], &cols, &coefs);
                        }
                        fresh[k] = false;
                    }
                }
                1 => {
                    let c: [u8; BAND] = std::array::from_fn(|_| rng.below(2) as u8);
                    // SAFETY: as above.
                    unsafe {
                        f.sub_mul_band(&mut band, &cols, &coefs, &c);
                        for k in 0..BAND {
                            f.sub_mul(&mut dense[k], &cols, &coefs, c[k]);
                        }
                    }
                    fresh = [false; BAND];
                }
                2 => {
                    let k = rng.below(BAND as u64) as usize;
                    let c = rng.below(2) as u8;
                    // SAFETY: as above.
                    unsafe {
                        f.sub_mul_band_row(&mut band, k, &cols, &coefs, c);
                        f.sub_mul(&mut dense[k], &cols, &coefs, c);
                    }
                    fresh[k] = false;
                }
                _ => {
                    read(&mut band, &mut dense);
                    fresh = [true; BAND];
                }
            }
        }
        read(&mut band, &mut dense);
    }
}

/// A reduction's rows made, rows that reduced to zero, pivots, and order,
/// each made row's coefficients as their values, and its reductions and
/// zeros.
type Outcome = (Made<u64>, u64, u64);

fn outcome<F: Field>(field: &F, reduction: &Reduction<F>) -> Outcome {
    let (rows, zero, pivots, taken) = made(reduction);
    let rows = rows
        .into_iter()
        .map(|(r, cols, coefs)| (r, cols, coefs.into_iter().map(|c| field.value(c)).collect()))
        .collect();
    (
        (rows, zero, pivots, taken),
        reduction.reductions,
        reduction.zeros,
    )
}

/// `matrix::reduce` of `m` under a fresh budget and buffers, with the
/// back-substitution where `full` holds.
fn plain<F: Field>(m: &Random<F>, full: bool) -> Outcome {
    let reduction = matrix::reduce::<F, Dense16, Rank1, NoCofactors>(
        &m.field,
        &m.basis,
        &m.matrix,
        &mut NoCofactors,
        &mut budget(),
        full,
        None,
        &mut Buffers::new(),
        None,
        None,
    )
    .unwrap();
    outcome(&m.field, &reduction)
}

/// Test 6: `matrix::reduce` of random matrices and `matrix::reduce_sig` of
/// random signature rounds over `Gf2` against the same over `Gf31` at
/// p = 2, the signature rounds of long rows taking bands: one outcome. The
/// rounds over `Gf2` take no packed rows, so that its bands and its sparse
/// reduction are compared, and then once more every one packed
/// (`PACK_FORCED`).
#[test]
fn gf2_reduces_as_gf31_at_two() {
    for pack in [Some(false), Some(true)] {
        PACK_FORCED.set(pack);
        compare_with_gf31();
    }
    PACK_FORCED.set(None);
}

/// Test 6's comparisons under the packed rows `PACK_FORCED` chooses.
fn compare_with_gf31() {
    let gf31 = Gf31::new(2).unwrap();
    // The signature rounds that took bands, and those that did not.
    let mut banded = [0usize; 2];
    for seed in 0..12u64 {
        for (elements, pairs, twice) in [(6, 20, 0), (12, 60, 3), (20, 120, 5)] {
            for (most, degree) in [(3, 4), (7, 6)] {
                let shape = ((elements, pairs, twice), (most, degree));
                let a = Random::of(Gf2::new(), 0x9f80 + seed, shape.0, shape.1);
                let b = Random::of(gf31, 0x9f80 + seed, shape.0, shape.1);
                for full in [false, true] {
                    assert_eq!(
                        plain(&a, full),
                        plain(&b, full),
                        "seed {seed}, shape {shape:?}, full {full}"
                    );
                }
            }
        }
        for long in [false, true] {
            let a = Round::random(Gf2::new(), 0x9f90 + seed, 12, 24, 16, 6, long);
            let b = Round::random(gf31, 0x9f90 + seed, 12, 24, 16, 6, long);
            let bands = matrix::takes_bands::<Gf2, Dense16, NoCofactors>(&a.matrix);
            assert_eq!(
                bands,
                matrix::takes_bands::<Gf31, Dense16, NoCofactors>(&b.matrix),
                "seed {seed}, long {long}: bands"
            );
            banded[usize::from(bands)] += 1;
            assert_eq!(
                outcome(&a.field, &a.reduce(None)),
                outcome(&b.field, &b.reduce(None)),
                "seed {seed}, long {long}"
            );
        }
    }
    assert!(
        banded[0] > 0 && banded[1] > 0,
        "{banded:?} signature rounds without bands and with"
    );
}
