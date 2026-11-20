//! column-index-peak, sub-item 2, the crate's test code: the rows of a
//! built matrix name their products through its columns, in grevlex and
//! lex; two builds through one run's buffers, the second taking the
//! first's rows, make the matrix a build from fresh buffers makes; and the
//! table's stamp and column read back as set
//! (`iterations/0115-column-index-peak/plan.md`, _Test code_, 4 to 6).
//!
//! `Random::with_order` is `cyclic_gap_tests::Random`'s construction in a
//! given order, which keeps the table it built in: sub-item 2 adds it, as
//! `iterations/0115-column-index-peak/instrument/random-with-order.diff`
//! writes it.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cyclic_gap_tests::{Random, long, short};
use crate::field::Gf31;
use crate::id::{MonoId, RowId};
use crate::matrix::{self, Buffers, Matrix};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Grevlex, Lex, Order};
use crate::poly::Basis;
use crate::rebuilt_reducer_tests::same;
use crate::shared_tests::budget;
use crate::systems::Rng;
use crate::table::Table;

/// The primes of the tests: one past the combinations' bound and one
/// below it.
const PRIMES: [u64; 2] = [65521, 7];

/// Asserts that the monomials of `m`'s columns decrease in `order`, and
/// that the monomials at each row's columns are its multiplier's products
/// with its element's monomials, in the element's order.
fn names_its_products<O: Order<Dense16>>(
    name: &str,
    table: &mut Table<Dense16>,
    basis: &Basis<Gf31, Rank1>,
    order: &O,
    m: &Matrix,
) {
    for w in m.monomials.windows(2) {
        assert!(
            table.cmp(order, w[0], w[1]).is_gt(),
            "{name}: the columns' monomials decrease"
        );
    }
    for r in 0..m.rows.len() {
        let row = m.rows[r];
        let want: Vec<MonoId> = basis
            .element(row.element)
            .monomials
            .iter()
            .map(|&u| {
                table
                    .mul(row.multiplier, u, &mut Budget::unbounded())
                    .unwrap()
            })
            .collect();
        let got: Vec<MonoId> = m
            .row(RowId::at(r))
            .to_wide()
            .iter()
            .map(|&c| m.monomials[c.index()])
            .collect();
        assert_eq!(got, want, "{name}: row {r}'s monomials");
    }
}

/// Test 4 in one order: random matrices of short and long rows over each
/// prime of `PRIMES`, seeds 1 to 4.
fn the_rows_name_their_products_in<O: Order<Dense16> + Copy>(order: O, order_name: &str) {
    for p in PRIMES {
        for seed in 1..=4 {
            for (shape_name, shape) in [("short", short(12, 30, 3, 2)), ("long", long(6, 8, 0, 1))]
            {
                let mut m = Random::with_order(p, seed, &shape, &order);
                let name = format!("{order_name}, {shape_name}, p {p}, seed {seed}");
                names_its_products(&name, &mut m.table, &m.basis, &order, &m.matrix);
            }
        }
    }
}

#[test]
fn the_rows_name_their_products() {
    the_rows_name_their_products_in(Grevlex, "grevlex");
    the_rows_name_their_products_in(Lex, "lex");
}

/// Test 5 in one order: a random matrix's given rows, built in round 1 by
/// `Random`, built in round 2 into a run's buffers and kept, then built
/// again in round 3 through them, against the same build from fresh
/// buffers in round 4: the two matrices are the same, some reducer row was
/// taken from the matrix before, and the matrix names its products. The
/// table numbers the rounds (`Table::open`).
fn kept_rows_and_the_merge_in<O: Order<Dense16> + Copy>(order: O, order_name: &str) {
    for p in PRIMES {
        for seed in 1..=4 {
            let mut m = Random::with_order(p, seed, &short(12, 30, 3, 2), &order);
            let given: Vec<_> = m.matrix.rows[..m.matrix.given].to_vec();
            let name = format!("{order_name}, p {p}, seed {seed}");
            let mut bufs: Buffers<Gf31> = Buffers::new();
            let first = matrix::build(
                &mut m.table,
                &m.basis,
                &order,
                given.clone(),
                &mut bufs,
                &mut budget(),
            )
            .unwrap();
            bufs.keep(first);
            let second = matrix::build(
                &mut m.table,
                &m.basis,
                &order,
                given.clone(),
                &mut bufs,
                &mut budget(),
            )
            .unwrap();
            let fresh = matrix::build(
                &mut m.table,
                &m.basis,
                &order,
                given,
                &mut Buffers::new(),
                &mut budget(),
            )
            .unwrap();
            same(&name, &fresh, &second);
            assert!(
                second.rows.len() == second.given || second.kept() > 0,
                "{name}: a reducer row taken from the matrix before"
            );
            names_its_products(&name, &mut m.table, &m.basis, &order, &second);
        }
    }
}

#[test]
fn kept_rows_and_the_merge() {
    // Grevlex numbers the columns by the merge with the matrix before,
    // lex by the sort, whose taken rows map through the matrix before's
    // monomials.
    kept_rows_and_the_merge_in(Grevlex, "grevlex");
    kept_rows_and_the_merge_in(Lex, "lex");
}

#[test]
fn the_table_pairs_stamp_and_column() {
    let format = Dense16::new(3);
    let mut table = Table::new(format, &mut Xoshiro256PlusPlus::seed_from_u64(3));
    let mut ids = Vec::new();
    for x in 0..10 {
        for y in 0..10 {
            for z in 0..10 {
                ids.push(table.intern(&[x, y, z], &mut Budget::unbounded()).unwrap());
            }
        }
    }
    for &id in &ids {
        assert_eq!(
            (table.stamp(id), table.column(id)),
            (0, MonoId::NONE),
            "id {id}: a new monomial's stamp and column"
        );
    }
    // Stamps and columns set apart, in two interleaved passes, each pass
    // over every id in an order of its own.
    let mut rng = Rng::new(5);
    let mut want: Vec<(u32, MonoId)> = vec![(0, MonoId::NONE); ids.len()];
    for pass in 0..4 {
        for _ in 0..ids.len() {
            let k = rng.below(ids.len() as u64) as usize;
            if pass % 2 == 0 {
                let stamp = rng.below(1 << 20) as u32;
                table.set_stamp(ids[k], stamp);
                want[k].0 = stamp;
            } else {
                let column = MonoId::at(rng.below(1 << 20) as usize);
                table.set_column(ids[k], column);
                want[k].1 = column;
            }
        }
        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(
                (table.stamp(id), table.column(id)),
                want[k],
                "pass {pass}, id {id}: the stamp and the column as set"
            );
        }
    }
}
