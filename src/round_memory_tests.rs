//! round-memory-time, the crate's test code: the rows a reduction makes at
//! their length, one run's buffers over a sequence of matrices, a matrix
//! past `RESERVE_PAST` from kept buffers, the polynomial that takes a row's
//! column vector, and the buffers given back (plan.md, _Test code_, 1 to
//! 5).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{
    self, Buffers, Combine, Matrix, Pivot, RESERVATION, RESERVE_PAST, Reduced, Reduction,
};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::table::Table;
use crate::trace::RowSource;
use crate::{MonomialOrder, Params, Threads};

fn budget() -> Budget {
    let p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    Budget::new(Instant::now(), &p).unwrap()
}

/// A random system over GF(p) on four variables: a basis of `elements`
/// random monic polynomials, and the rows of the random S-pairs of its
/// `G_min` drawn `pairs` times, a draw of one element twice skipped, every
/// third draw's rows given twice.
struct System {
    field: Gf31,
    table: Table<Dense16>,
    basis: Basis<Gf31, Rank1>,
    rows: Vec<RowSource>,
}

impl System {
    fn new(p: u64, seed: u64, elements: usize, pairs: usize) -> System {
        let field = Gf31::new(p).unwrap();
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut draw = Xoshiro256PlusPlus::seed_from_u64(seed ^ p.rotate_left(23) ^ 0x6d0e);
        let mut below = |n: u64| draw.next_u64() % n;
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        while basis.len() < elements {
            let mut terms = Vec::new();
            for _ in 0..2 + below(7) {
                let mut e = [0u32; 4];
                let mut degree = 1 + below(5) as u32;
                for x in &mut e {
                    let k = below(u64::from(degree) + 1) as u32;
                    *x = k;
                    degree -= k;
                }
                let m = table.intern(&e).unwrap();
                terms.push((m, field.coef(1 + below(p - 1))));
            }
            let mut g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
            if g.is_zero() || table.degree(g.lead()) == 0 {
                continue;
            }
            g.make_monic(&field);
            basis.push_min(&table, g).unwrap();
        }
        let min = basis.min().to_vec();
        let mut rows = Vec::new();
        for k in 0..pairs {
            let i = min[below(min.len() as u64) as usize];
            let j = min[below(min.len() as u64) as usize];
            if i == j {
                continue;
            }
            let lcm = table.lcm(basis.lead(i), basis.lead(j)).unwrap();
            for g in [i, j] {
                let row = RowSource {
                    element: g,
                    multiplier: table.div(lcm, basis.lead(g)).unwrap(),
                };
                rows.push(row);
                if k % 3 == 0 {
                    rows.push(row);
                }
            }
        }
        System {
            field,
            table,
            basis,
            rows,
        }
    }

    /// The matrix of the system's rows in round `round`, its columns in
    /// `cols`. Each matrix of a table takes a round of its own.
    fn matrix(&mut self, round: u32, cols: Vec<MonoId>) -> Matrix {
        let rows = self.rows.clone();
        matrix::build(
            &mut self.table,
            &self.basis,
            &Grevlex,
            round,
            rows,
            cols,
            &mut budget(),
        )
        .unwrap()
    }

    /// The inter-reduction's matrix of `G_min` in round `round`.
    fn inter(&mut self, round: u32) -> Matrix {
        let one = self.table.intern(&[0; 4]).unwrap();
        let rows = self
            .basis
            .min()
            .iter()
            .map(|&i| RowSource {
                element: i,
                multiplier: one,
            })
            .collect();
        matrix::build(
            &mut self.table,
            &self.basis,
            &Grevlex,
            round,
            rows,
            Vec::new(),
            &mut budget(),
        )
        .unwrap()
    }

    fn reduce(
        &self,
        m: &Matrix,
        seed: Option<u64>,
        full: bool,
        bufs: &mut Buffers<Gf31>,
    ) -> Reduction<Gf31> {
        let mut combine = seed.map(Combine::new);
        matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
            &self.field,
            &self.basis,
            m,
            &mut NoCofactors,
            &mut budget(),
            full,
            combine.as_mut(),
            bufs,
            None,
        )
        .unwrap()
    }
}

/// The made rows, each its row, columns, and coefficients, in the order
/// made.
fn made(reduction: &Reduction<Gf31>) -> Vec<(RowId, Vec<MonoId>, Vec<u32>)> {
    reduction
        .made
        .iter()
        .map(|r| (r.row, r.cols.clone(), r.coefs.clone()))
        .collect()
}

/// The pivot of every column, as a row, a made row, or none.
fn pivots(reduction: &Reduction<Gf31>) -> Vec<(u8, usize)> {
    reduction
        .pivot
        .iter()
        .map(|p| match *p {
            Pivot::None => (0, 0),
            Pivot::Row(r) => (1, r.index()),
            Pivot::New(k) => (2, k.index()),
        })
        .collect()
}

/// Every made row holds its entries at its length.
fn at_length(rows: &[Reduced<Gf31>], case: &str) {
    for r in rows {
        assert_eq!(
            r.cols.capacity(),
            r.cols.len(),
            "{case}: row {row}",
            row = r.row
        );
        assert_eq!(
            r.coefs.capacity(),
            r.coefs.len(),
            "{case}: row {row}",
            row = r.row
        );
    }
}

/// The columns, the rows, and the monomials of two matrices agree.
fn same_matrix(a: &Matrix, b: &Matrix, case: &str) {
    assert_eq!(a.given, b.given, "{case}");
    assert_eq!(a.rows, b.rows, "{case}");
    assert_eq!(a.monomials, b.monomials, "{case}");
    for r in (0..a.rows.len()).map(RowId::at) {
        assert_eq!(a.row(r), b.row(r), "{case}: row {r}");
    }
}

/// Test 1, round-memory-time.1, .2: the rows `reduce` makes, on its own and
/// in combinations, with and without the back-substitution, and those
/// `inter_reduce` makes, each at its length; and `reduce`'s rows the same
/// as with fresh buffers.
#[test]
fn the_rows_a_reduction_makes_hold_their_entries_at_their_length() {
    let mut rows = 0;
    for p in [65521, 46349, 2_147_483_647, 101] {
        for seed in 0..12 {
            let mut s = System::new(p, seed, 4 + seed as usize % 10, 20 + 8 * seed as usize);
            let m = s.matrix(1, Vec::new());
            let mut bufs = Buffers::new();
            for (combine, full) in [
                (None, false),
                (None, true),
                (Some(seed), false),
                (Some(seed), true),
            ] {
                let case = format!("p {p}, seed {seed}, combine {combine:?}, full {full}");
                let red = s.reduce(&m, combine, full, &mut bufs);
                at_length(&red.made, &case);
                let fresh = s.reduce(&m, combine, full, &mut Buffers::new());
                assert_eq!(made(&red), made(&fresh), "{case}");
                rows += red.made.len();
            }
            let inter = s.inter(2);
            let reduced = matrix::inter_reduce::<Gf31, Dense16, Rank1, NoCofactors>(
                &s.field,
                &s.basis,
                &inter,
                &mut NoCofactors,
                &mut budget(),
                &mut bufs,
            )
            .unwrap();
            at_length(&reduced, &format!("p {p}, seed {seed}, inter_reduce"));
            rows += reduced.len();
        }
    }
    assert!(rows > 1000, "{rows} rows made");
}

/// Test 2, round-memory-time.4, .5, .6, .12: one run's buffers over the
/// matrices of 20 systems in turn, wider and narrower, each built from the
/// kept column vector and given back after its reduction: the matrices and
/// reductions of fresh buffers, a kept vector's capacity handed on, and a
/// soiled dense row that changes nothing.
#[test]
fn one_runs_buffers_over_a_sequence_of_matrices() {
    let mut bufs: Buffers<Gf31> = Buffers::new();
    let mut widths = Vec::new();
    for k in 0..20u64 {
        let p = [65521, 2_147_483_647][k as usize % 2];
        let mut s = System::new(
            p,
            100 + k,
            3 + (k as usize * 7) % 13,
            8 + (k as usize * 37) % 140,
        );
        let case = format!("system {k}");
        let kept = bufs.capacities()[0];
        let cols = bufs.take_cols();
        assert!(cols.is_empty(), "{case}: a column vector taken empty");
        assert_eq!(cols.capacity(), kept, "{case}: the kept capacity handed on");
        let m = s.matrix(1, cols);
        let fresh = s.matrix(2, Vec::new());
        same_matrix(&m, &fresh, &case);
        widths.push(m.columns());
        if k % 3 == 1 {
            let len = (m.columns() + 37) * (1 + (k as usize / 3) % 2) / 2;
            bufs.soil(len, 0x7fff_ffff_0000_0001);
        }
        let seed = (k % 2 == 0).then_some(k);
        let full = k % 4 < 2;
        let red = s.reduce(&m, seed, full, &mut bufs);
        let other = s.reduce(&fresh, seed, full, &mut Buffers::new());
        assert_eq!(made(&red), made(&other), "{case}");
        assert_eq!(red.zero, other.zero, "{case}");
        assert_eq!(pivots(&red), pivots(&other), "{case}");
        assert_eq!(
            (red.reductions, red.zeros),
            (other.reductions, other.zeros),
            "{case}"
        );
        at_length(&red.made, &case);
        let capacity = m.capacity();
        bufs.keep_cols(m);
        assert_eq!(bufs.capacities()[0], capacity, "{case}: the vector kept");
    }
    let (least, most) = (widths.iter().min(), widths.iter().max());
    assert!(least < most, "widths {widths:?}");
}

/// A table on 6 variables in `Dense8`, and a basis of one monic element:
/// every monomial of degree 9 or less, 5,005 terms.
fn element() -> (Table<Dense8>, Basis<Gf31, Rank1>) {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense8::new(6), &mut Xoshiro256PlusPlus::seed_from_u64(43));
    let mut draw = Xoshiro256PlusPlus::seed_from_u64(0x51ad);
    let mut terms = Vec::new();
    for e in exponents(6, 9) {
        let m = table.intern(&e).unwrap();
        terms.push((m, field.coef(1 + draw.next_u64() % 65520)));
    }
    let mut g: Poly<Gf31, Rank1> = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
    assert_eq!(g.len(), 5005);
    g.make_monic(&field);
    let mut basis = Basis::new();
    basis.push_min(&table, g).unwrap();
    (table, basis)
}

/// Every exponent vector on n variables of degree at most d.
fn exponents(n: usize, d: u32) -> Vec<Vec<u32>> {
    fn fill(e: &mut [u32], i: usize, left: u32, out: &mut Vec<Vec<u32>>) {
        if i + 1 == e.len() {
            e[i] = left;
            out.push(e.to_vec());
            return;
        }
        for k in (0..=left).rev() {
            e[i] = k;
            fill(e, i + 1, left - k, out);
        }
        e[i] = 0;
    }
    let mut out = Vec::new();
    for degree in 0..=d {
        fill(&mut vec![0u32; n], 0, degree, &mut out);
    }
    out
}

/// The matrix of round `round` of `count` multiples of the element, its
/// multipliers the first `count` monomials of degree 1 to 3, by
/// `build_rows` into `cols`.
fn multiples(
    table: &mut Table<Dense8>,
    basis: &Basis<Gf31, Rank1>,
    count: usize,
    round: u32,
    cols: Vec<MonoId>,
) -> Matrix {
    let rows = exponents(6, 3)
        .into_iter()
        .filter(|e| e.iter().sum::<u32>() > 0)
        .take(count)
        .map(|e| RowSource {
            element: ElementId::at(0),
            multiplier: table.intern(&e).unwrap(),
        })
        .collect();
    matrix::build_rows(table, basis, &Grevlex, round, rows, cols, &mut budget()).unwrap()
}

/// Test 3, round-memory-time.6, matrix-slack.1: a matrix past
/// `RESERVE_PAST` built after one that is not, and after a small one that
/// holds the reservation of one that is, from the kept column vector: the
/// rows of a matrix built from `Vec::new()`,
/// and a capacity of at least `RESERVATION`; a small matrix after it keeps
/// that capacity.
#[test]
fn a_matrix_past_the_bound_from_kept_buffers() {
    let (mut table, basis) = element();
    let mut bufs: Buffers<Gf31> = Buffers::new();
    let mut round = 0;
    let mut next = || {
        round += 1;
        round
    };
    let small = multiples(&mut table, &basis, 10, next(), bufs.take_cols());
    assert!(small.capacity() < RESERVATION, "50,050 entries");
    bufs.keep_cols(small);
    for k in 0..2 {
        let big = multiples(&mut table, &basis, 60, next(), bufs.take_cols());
        let fresh = multiples(&mut table, &basis, 60, next(), Vec::new());
        assert!(
            big.entries() > RESERVE_PAST,
            "{entries} entries",
            entries = big.entries()
        );
        assert!(
            big.capacity() >= RESERVATION,
            "pass {k}: {capacity}",
            capacity = big.capacity()
        );
        same_matrix(&big, &fresh, &format!("pass {k}"));
        let capacity = big.capacity();
        bufs.keep_cols(big);
        let small = multiples(&mut table, &basis, 10, next(), bufs.take_cols());
        let fresh = multiples(&mut table, &basis, 10, next(), Vec::new());
        assert_eq!(
            small.capacity(),
            capacity,
            "pass {k}: the small matrix after"
        );
        same_matrix(&small, &fresh, &format!("pass {k}, small"));
        bufs.keep_cols(small);
    }
}

/// Test 4, round-memory-time.3: the polynomial of a made row takes the
/// row's column vector, the same allocation, and holds each column's
/// monomial.
#[test]
fn the_polynomial_takes_the_rows_column_vector() {
    let mut polys = 0;
    for seed in 0..12 {
        let mut s = System::new(65521, 300 + seed, 6, 60);
        let m = s.matrix(1, Vec::new());
        let red = s.reduce(&m, None, true, &mut Buffers::new());
        for r in red.made {
            let pointer = r.cols.as_ptr();
            let want: Vec<MonoId> = r.cols.iter().map(|&c| m.monomials[c.index()]).collect();
            let coefs = r.coefs.clone();
            let p: Poly<Gf31, Rank1> = r.poly(&m);
            assert_eq!(p.monomials.as_ptr(), pointer, "seed {seed}");
            assert_eq!(p.monomials, want, "seed {seed}");
            assert_eq!(p.coefs, coefs, "seed {seed}");
            polys += 1;
        }
    }
    assert!(polys >= 30, "{polys} polynomials");
}

/// Test 5: `release` gives back the column vector, the dense row, and the
/// two work vectors (`band_tests.rs`, test 9, the band's), and the next
/// matrix built and reduced from released buffers makes what fresh buffers
/// make.
#[test]
fn release_gives_everything_back() {
    let mut s = System::new(65521, 500, 8, 80);
    let mut bufs: Buffers<Gf31> = Buffers::new();
    let m = s.matrix(1, bufs.take_cols());
    let red = s.reduce(&m, None, true, &mut bufs);
    assert!(!red.made.is_empty());
    bufs.keep_cols(m);
    let [cols, dense, row_cols, row_coefs] = bufs.capacities();
    assert!(cols > 0 && dense > 0 && row_cols > 0 && row_coefs > 0);
    bufs.release();
    assert_eq!(bufs.capacities(), [0; 4]);
    let m = s.matrix(2, bufs.take_cols());
    let again = s.reduce(&m, None, true, &mut bufs);
    let fresh = s.reduce(&m, None, true, &mut Buffers::new());
    assert_eq!(made(&again), made(&fresh));
    assert_eq!(made(&again), made(&red));
}
