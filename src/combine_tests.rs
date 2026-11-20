//! gfp-unsolved, sub-item 1: the plain run's linear algebra: the random
//! combinations of `matrix::reduce` against each row on its own, the plain
//! run with and without a listener, and the checker on the cofactors of a
//! plain run.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stats;
use crate::budget::Budget;
use crate::cofactors::{Cofactors, NoCofactors, Tracked};
use crate::f4;
use crate::field::{Field, Gf31};
use crate::id::{RowId, RowSource};
use crate::matrix::{self, Buffers, Combine};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Elements, Poly};
use crate::shared_tests::{Random, budget, rows};
use crate::systems::{Rng, dense, random_system};
use crate::table::Table;
use crate::text::{Header, Unread, write_basis};
use crate::trace::{NoLearner, NoTrace, Trace};

/// A trace that listens and keeps nothing.
struct Listening;

impl Trace for Listening {
    fn matrix(&mut self, _: &[RowSource]) {}
    fn counted(&mut self, _: &[RowId]) {}
}

impl Random {
    /// A random matrix over GF(p) on four variables: a basis of `elements` random monic
    /// polynomials, and given rows that are the halves of random S-pairs of the basis, with every
    /// `twice`-th draw's rows given twice. Symbolic preprocessing adds the reducers.
    fn drawn(p: u64, seed: u64, elements: usize, pairs: usize, twice: usize) -> Random {
        let field = Gf31::new(p).unwrap();
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut rng = Rng::new(seed ^ p.rotate_left(17) ^ 0xc0b1);
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        while basis.len() < elements {
            let mut terms = Vec::new();
            for _ in 0..2 + rng.below(6) {
                let mut e = [0u32; 4];
                let mut degree = 1 + rng.below(4) as u32;
                for x in &mut e {
                    let k = rng.below(u64::from(degree) + 1) as u32;
                    *x = k;
                    degree -= k;
                }
                let m = table.intern(&e, &mut Budget::unbounded()).unwrap();
                terms.push((m, field.coef(1 + rng.below(p - 1))));
            }
            let mut g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
            if g.is_zero() || table.degree(g.lead()) == 0 {
                continue;
            }
            g.make_monic(&field);
            basis.push_min(&table, g, &mut Budget::unbounded()).unwrap();
        }
        let min = basis.min().to_vec();
        let mut rows = Vec::new();
        for k in 0..pairs {
            let i = min[rng.below(min.len() as u64) as usize];
            let j = min[rng.below(min.len() as u64) as usize];
            if i == j {
                continue;
            }
            let lcm = table
                .lcm(basis.lead(i), basis.lead(j), &mut Budget::unbounded())
                .unwrap();
            for g in [i, j] {
                let row = RowSource {
                    element: g,
                    multiplier: table
                        .div(lcm, basis.lead(g), &mut Budget::unbounded())
                        .unwrap(),
                };
                rows.push(row);
                if twice > 0 && k % twice == 0 {
                    rows.push(row);
                }
            }
        }
        let matrix = matrix::build(
            &mut table,
            &basis,
            &Grevlex,
            rows,
            &mut Buffers::new(),
            &mut budget(),
        )
        .unwrap();
        Random {
            field,
            basis,
            matrix,
        }
    }
}

/// gfp-unsolved.2, .5: the made rows are the reduced row echelon form of
/// the rows to reduce modulo the pivots, with a `Combine` and without one,
/// below 46341 and above.
#[test]
fn combinations_make_what_rows_make() {
    for p in [7, 46337, 65521, 46349, 2_147_483_647] {
        for seed in 0..12 {
            let m = Random::drawn(
                p,
                seed,
                4 + (seed as usize % 12),
                20 + 10 * seed as usize,
                3,
            );
            let exact = m.reduce(None);
            let combined = m.reduce(Some(&mut Combine::new(seed)));
            assert_eq!(rows(&combined), rows(&exact), "p {p}, seed {seed}");
        }
    }
}

/// A matrix with no row to reduce: nothing made, combined or not.
#[test]
fn a_matrix_with_no_row_to_reduce() {
    let m = Random::drawn(65521, 3, 6, 0, 0);
    for combine in [None, Some(Combine::new(1))] {
        let mut combine = combine;
        let got = m.reduce(combine.as_mut());
        assert!(got.made.is_empty());
    }
}

/// The parts of a call and the reduced basis of `f4::run` on them.
fn plain_run<C: Cofactors<Gf31, Dense16>, T: Trace>(
    text: &str,
    cof: &mut C,
    trace: &mut T,
    seed: u64,
) -> (Table<Dense16>, Elements<Gf31, Rank1, C::Element>) {
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    let field = header.gf31().unwrap();
    let mut table = Table::new(
        Dense16::new(header.names.len()),
        &mut Xoshiro256PlusPlus::seed_from_u64(1),
    );
    let generators = header
        .generators(&field, &mut table, &Grevlex, &mut Budget::unbounded())
        .map_err(Unread::unbounded)
        .unwrap();
    let elements = f4::run(
        &field,
        &mut table,
        &Grevlex,
        generators,
        cof,
        trace,
        &mut budget(),
        &mut Stats::default(),
        &mut NoLearner,
        seed,
    )
    .unwrap();
    (table, elements)
}

fn basis_text<E>(
    text: &str,
    table: &Table<Dense16>,
    elements: &Elements<Gf31, Rank1, E>,
) -> String {
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    let field = header.gf31().unwrap();
    let mut out = String::new();
    write_basis(
        &mut out,
        &header.names,
        table,
        &field,
        elements.iter().map(|(g, _)| g),
        &mut Budget::unbounded(),
    )
    .expect("a budget without bounds never stops");
    out
}

/// Random systems over two primes and dense cubic systems over GF(65521).
fn cases() -> Vec<(String, String)> {
    let mut cases: Vec<(String, String)> = Vec::new();
    for p in [65521, 2_147_483_647] {
        for seed in 0..4 {
            cases.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
    }
    for m in 3..=4 {
        cases.push((format!("cubic-{m}"), dense(m, 3, 65521, 1)));
    }
    cases
}

/// gfp-unsolved.6: the plain run gives one basis without a listener and
/// with a listening trace.
#[test]
fn a_listener_leaves_the_basis() {
    for (name, text) in &cases() {
        let (t1, e1) = plain_run(text, &mut NoCofactors, &mut NoTrace, 5);
        let (t2, e2) = plain_run(text, &mut NoCofactors, &mut Listening, 5);
        assert_eq!(
            basis_text(text, &t1, &e1),
            basis_text(text, &t2, &e2),
            "{name}"
        );
    }
}

/// gfp-unsolved.8: the checker accepts the cofactors of a plain run that
/// tracks them in the exact mode.
#[test]
fn the_plain_runs_cofactors_are_accepted() {
    for (name, text) in &cases() {
        let header = Header::read(text, &mut Budget::unbounded()).unwrap();
        let field = header.gf31().unwrap();
        let n = header.names.len();
        let mut scratch = Table::new(Dense16::new(n), &mut Xoshiro256PlusPlus::seed_from_u64(1));
        let generators: Vec<Poly<Gf31, Rank1>> = header
            .generators(&field, &mut scratch, &Grevlex, &mut Budget::unbounded())
            .map_err(Unread::unbounded)
            .unwrap();
        let generators = generators.len();
        let mut cof = Tracked::new(
            field,
            Dense16::new(n),
            generators,
            &mut Xoshiro256PlusPlus::seed_from_u64(2),
        );
        let (table, elements) = plain_run(text, &mut cof, &mut NoTrace, 9);
        let basis = basis_text(text, &table, &elements);
        let rows: Vec<_> = elements.into_iter().map(|(_, row)| row).collect();
        let mut cofactors = String::new();
        cof.write(
            &mut cofactors,
            &header.names,
            &Grevlex,
            &rows,
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        let verdict = grobx_checker::check(
            text,
            &basis,
            &cofactors,
            &grobx_checker::Order::Grevlex,
            grobx_checker::Mode::Exact,
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}\n{basis}\n{cofactors}"));
        assert_eq!(verdict.groebner, Ok(()), "{name}\n{text}\n{basis}");
        assert_eq!(verdict.reduced, Ok(()), "{name}\n{text}\n{basis}");
    }
}
