//! back-substitution-time, the crate's test code: the back-substitution
//! and the final inter-reduction in bands against one row at a time, the
//! given rows reduced already taken as they stand, and a stop in a band's
//! sweep (`iterations/0091-back-substitution-time/plan.md`, _Test code_, 1
//! to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stop;
use crate::band_tests::Random;
use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::field::{BAND, Field, Gf31};
use crate::halt::Halt;
use crate::id::{MonoId, RowId, RowSource};
use crate::matrix::{self, Buffers, Matrix, Reduced};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::shared_tests::{GF31_PRIMES, budget, passed_after};
use crate::table::Table;

/// A row of a result: its row, columns, and coefficients.
type Row = (RowId, Vec<MonoId>, Vec<u32>);

fn rows_of(made: &[Reduced<Gf31>]) -> Vec<Row> {
    made.iter()
        .map(|r| (r.row, r.cols.clone(), r.coefs.clone()))
        .collect()
}

/// `matrix::reduce` of `m`'s matrix, each row on its own, no tests, no
/// cofactors, under `b`, from `bufs` with their bands as `bands` says;
/// the back-substitution where `full`. The made rows.
pub(crate) fn reduce(
    m: &Random,
    full: bool,
    bands: bool,
    b: &mut Budget,
    bufs: &mut Buffers<Gf31>,
) -> Result<Vec<Reduced<Gf31>>, Halt> {
    bufs.bands = bands;
    matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
        &m.field,
        &m.basis,
        &m.matrix,
        &mut NoCofactors,
        b,
        full,
        None,
        bufs,
        None,
        None,
    )
    .map(|r| r.made)
}

/// The made rows of a forward reduction that the back-substitution takes,
/// in its order, by decreasing leading column: those whose tail holds the
/// leading column of another made row.
fn needing(made: &[Reduced<Gf31>]) -> Vec<&Reduced<Gf31>> {
    let leads: Vec<MonoId> = made.iter().map(|r| r.cols[0]).collect();
    let mut need: Vec<&Reduced<Gf31>> = made
        .iter()
        .filter(|r| r.cols[1..].iter().any(|c| leads.contains(c)))
        .collect();
    need.sort_by_key(|r| std::cmp::Reverse(r.cols[0]));
    need
}

/// Test 1: back-substitution-time.1 to .4, .9. The back-substitution in
/// bands makes the rows one row at a time makes, in the order made, on
/// random matrices. Over all cases some round back-substitutes a group of
/// two, of three, and of four rows, and some band row holds another band
/// row's leading column in its tail before the sweep.
#[test]
fn bands_back_substitute_as_one_row_at_a_time() {
    // Groups of two, three, and four rows, and band rows that meet another
    // band row's lead.
    let (mut groups, mut met) = ([0usize; BAND + 1], 0);
    for p in GF31_PRIMES {
        for seed in 0..200u64 {
            let shape = (6 + (seed % 11) as usize, 8 + (seed % 29) as usize, 0);
            let terms = [8, 30, 80][(seed % 3) as usize];
            let m = Random::new(p, seed, shape, (terms, 5));
            let name = format!("p {p}, seed {seed}");
            let forward = reduce(&m, false, true, &mut budget(), &mut Buffers::new()).unwrap();
            let need = needing(&forward);
            for group in need.chunks(BAND) {
                groups[group.len()] += 1;
                let leads: Vec<MonoId> = group.iter().map(|r| r.cols[0]).collect();
                if group.len() >= 2
                    && group
                        .iter()
                        .any(|r| r.cols[1..].iter().any(|c| leads.contains(c)))
                {
                    met += 1;
                }
            }
            let banded = reduce(&m, true, true, &mut budget(), &mut Buffers::new()).unwrap();
            let alone = reduce(&m, true, false, &mut budget(), &mut Buffers::new()).unwrap();
            assert_eq!(rows_of(&banded), rows_of(&alone), "{name}");
        }
    }
    assert!(groups[2] > 0, "no group of two rows");
    assert!(groups[3] > 0, "no group of three rows");
    assert!(groups[4] > 0, "no group of four rows");
    assert!(met > 0, "no band row held another band row's lead");
}

/// A random basis of `m`'s, and its inter-reduction's matrix as
/// `f4::inter_matrix` builds it: `G_min`'s elements in the order made,
/// multiplier one, and the reducers symbolic preprocessing adds, in round
/// 2 of `m`'s table.
fn inter_matrix(m: &mut Random) -> Matrix {
    let one = m.table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
    let rows = m
        .basis
        .min()
        .iter()
        .map(|&g| RowSource {
            element: g,
            multiplier: one,
        })
        .collect();
    matrix::build(
        &mut m.table,
        &m.basis,
        &Grevlex,
        rows,
        &mut Buffers::new(),
        &mut budget(),
    )
    .unwrap()
}

/// `matrix::inter_reduce` of `matrix` over `m`'s basis, no cofactors,
/// with its bands as `bands` says: every given row's result, by row.
fn inter_reduce(m: &Random, matrix: &Matrix, bands: bool) -> Vec<Row> {
    let mut bufs = Buffers::new();
    bufs.bands = bands;
    let made = matrix::inter_reduce::<Gf31, Dense16, Rank1, NoCofactors>(
        &m.field,
        &m.basis,
        matrix,
        &mut NoCofactors,
        &mut budget(),
        &mut bufs,
    )
    .unwrap();
    let mut rows = rows_of(&made);
    rows.sort_by_key(|r| r.0);
    rows
}

/// Test 2: back-substitution-time.5 to .7. The inter-reduction in bands
/// makes every given row's result one row at a time makes, on random
/// bases. Over all cases some inter-reduction had four given rows or more
/// whose tails hold a pivot's column, as a band takes them, and some had
/// such rows in a matrix wider than eight times its mean row, where the
/// counts may choose the marked scan.
#[test]
fn bands_inter_reduce_as_one_row_at_a_time() {
    let (mut dense, mut wide) = (0, 0);
    for p in GF31_PRIMES {
        for seed in 0..200u64 {
            let shape = (6 + (seed % 17) as usize, 0, 0);
            let terms = [8, 30, 80][(seed % 3) as usize];
            let mut m = Random::new(p, seed, shape, (terms, 6));
            let name = format!("p {p}, seed {seed}");
            let matrix = inter_matrix(&mut m);
            let mut lead = vec![false; matrix.columns()];
            for q in (0..matrix.rows.len()).map(RowId::at) {
                lead[matrix.row(q).lead().index()] = true;
            }
            let working = (0..matrix.given)
                .map(RowId::at)
                .filter(|&r| {
                    matrix
                        .row(r)
                        .tail()
                        .to_wide()
                        .iter()
                        .any(|c| lead[c.index()])
                })
                .count();
            if working >= BAND {
                let mean = matrix.entries() / matrix.rows.len().max(1);
                if 8 * mean < matrix.columns() {
                    wide += 1;
                } else {
                    dense += 1;
                }
            }
            assert_eq!(
                inter_reduce(&m, &matrix, true),
                inter_reduce(&m, &matrix, false),
                "{name}"
            );
        }
    }
    assert!(dense > 0, "no dense inter-reduction of four rows to reduce");
    assert!(wide > 0, "no wide inter-reduction of four rows to reduce");
}

/// Test 3: back-substitution-time.8. A given row whose tail holds no
/// pivot's column comes back as it stands: every row of a basis whose
/// tails are constants, and beside them the rows of a basis that adds an
/// element whose tail holds another element's leading monomial times a
/// variable, below its own leading monomial, with bands on and off.
#[test]
fn a_reduced_row_comes_back_as_it_stands() {
    for p in GF31_PRIMES {
        let field = Gf31::new(p).unwrap();
        let mut table = Table::new(Dense16::new(4), &mut Xoshiro256PlusPlus::seed_from_u64(p));
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        // x0^2 + 3, x1^3 + 5, x2^2 x3 + 6, x3^4 + 4, and x2^3 + 2 x0^2 x3 + 9,
        // whose tail holds x0^2 times x3, below x2^3 in grevlex.
        let elements: [&[([u32; 4], u64)]; 5] = [
            &[([2, 0, 0, 0], 1), ([0, 0, 0, 0], 3)],
            &[([0, 3, 0, 0], 1), ([0, 0, 0, 0], 5)],
            &[([0, 0, 2, 1], 1), ([0, 0, 0, 0], 6)],
            &[([0, 0, 0, 4], 1), ([0, 0, 0, 0], 4)],
            &[([0, 0, 3, 0], 1), ([2, 0, 0, 1], 2), ([0, 0, 0, 0], 9)],
        ];
        for (k, element) in elements.iter().enumerate() {
            let mut terms: Vec<(MonoId, u32)> = element
                .iter()
                .map(|(e, c)| {
                    (
                        table.intern(e, &mut Budget::unbounded()).unwrap(),
                        field.coef(*c),
                    )
                })
                .collect();
            let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
            basis.push_min(&table, g, &mut Budget::unbounded()).unwrap();
            if k < 3 {
                continue;
            }
            let one = table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
            let rows = basis
                .min()
                .iter()
                .map(|&g| RowSource {
                    element: g,
                    multiplier: one,
                })
                .collect();
            let matrix = matrix::build(
                &mut table,
                &basis,
                &Grevlex,
                rows,
                &mut Buffers::new(),
                &mut budget(),
            )
            .unwrap();
            for bands in [true, false] {
                let mut bufs = Buffers::new();
                bufs.bands = bands;
                let made = matrix::inter_reduce::<Gf31, Dense16, Rank1, NoCofactors>(
                    &field,
                    &basis,
                    &matrix,
                    &mut NoCofactors,
                    &mut budget(),
                    &mut bufs,
                )
                .unwrap();
                let name = format!("p {p}, {n} elements, bands {bands}", n = k + 1);
                assert_eq!(made.len(), matrix.given, "{name}");
                for row in &made {
                    let source = matrix.rows[row.row.index()];
                    let coefs = &basis.element(source.element).coefs;
                    // The elements whose tails are constants stand; the
                    // one with x0^2 x3 in its tail does not.
                    let tail = &basis.element(source.element).monomials[1..];
                    let reduced_already = tail.iter().all(|&m| table.degree(m) == 0);
                    if reduced_already {
                        assert_eq!(row.cols, matrix.row(row.row).to_wide(), "{name}");
                        assert_eq!(&row.coefs, coefs, "{name}");
                    } else {
                        assert_ne!(row.cols, matrix.row(row.row).to_wide(), "{name}");
                    }
                }
            }
        }
    }
}

/// Test 4. A reduction whose back-substitution takes a band, stopped by a
/// deadline once the forward reduction's units are spent, returns
/// `Stop::Deadline`; the same buffers then reduce the matrix again with no
/// bound and make the rows the back-substitution one row at a time makes.
#[test]
fn a_band_back_substitution_stops_at_its_deadline() {
    let mut stopped = 0;
    for p in GF31_PRIMES {
        for seed in 0..60u64 {
            let shape = (6 + (seed % 11) as usize, 8 + (seed % 29) as usize, 0);
            let m = Random::new(p, seed, shape, (80, 5));
            let name = format!("p {p}, seed {seed}");
            let mut forward_budget = budget();
            let forward =
                reduce(&m, false, true, &mut forward_budget, &mut Buffers::new()).unwrap();
            if needing(&forward).len() < 2 {
                continue;
            }
            let mut bufs = Buffers::new();
            let mut b = passed_after(forward_budget.spent() + 1);
            match reduce(&m, true, true, &mut b, &mut bufs) {
                Err(Halt::Stop(Stop::Deadline)) => stopped += 1,
                Err(e) => panic!("{name}: {e:?}"),
                Ok(_) => panic!("{name}: no stop"),
            }
            let again = reduce(&m, true, true, &mut budget(), &mut bufs).unwrap();
            let alone = reduce(&m, true, false, &mut budget(), &mut Buffers::new()).unwrap();
            assert_eq!(rows_of(&again), rows_of(&alone), "{name}");
        }
    }
    assert!(stopped > 0, "no reduction stopped");
}
