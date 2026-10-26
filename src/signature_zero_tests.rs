//! signature-zero-time, the unit tests: the signature round's rows ruled
//! zero by random combinations against each row reduced on its own, the
//! number of combinations by prime, a failed test, a band that completes
//! the streak, and random banded rounds
//! (`iterations/0049-signature-zero-time/plan.md`, _Test code_, 1 to 3,
//! and `iterations/0061-signature-zero-time/plan.md`, _Test code_, 10 and
//! 11).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::field::{Field, Gf31, Gf64};
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{self, Buffers, Combine, Matrix, Pivot, Reduction, zero_combinations};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::sig::Sig;
use crate::systems::Rng;
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

/// A signature round over a field: a basis, the matrix of its rows, the
/// reductants first, and each row's signature. Every signature has the
/// unit monomial, so its index alone orders it.
pub(crate) struct Round<F: Field> {
    pub(crate) field: F,
    pub(crate) table: Table<Dense16>,
    pub(crate) basis: Basis<F, Rank1>,
    pub(crate) matrix: Matrix,
    pub(crate) sigs: Vec<Sig>,
}

impl<F: Field> Round<F> {
    /// The round of the reducers `reducers` and the reductants
    /// `reductants`, each a row and its signature's index.
    pub(crate) fn of(
        field: F,
        table: Table<Dense16>,
        basis: Basis<F, Rank1>,
        reductants: Vec<(RowSource, u32)>,
        reducers: Vec<(RowSource, u32)>,
    ) -> Round<F> {
        let mut table = table;
        let one = table.intern(&[0; 4]).unwrap();
        let given = reductants.len();
        let (rows, indices): (Vec<RowSource>, Vec<u32>) =
            reductants.into_iter().chain(reducers).unzip();
        let mut matrix = matrix::build_rows(
            &mut table,
            &basis,
            &Grevlex,
            1,
            rows,
            Vec::new(),
            &mut budget(),
        )
        .unwrap();
        matrix.given = given;
        let sigs = indices
            .into_iter()
            .map(|index| Sig {
                monomial: one,
                index,
            })
            .collect();
        Round {
            field,
            table,
            basis,
            matrix,
            sigs,
        }
    }

    /// A random round on four variables: `elements` random monic elements,
    /// each the sum of 2 to 7 drawn terms in degrees up to 4, or where `long`
    /// is true each
    /// monomial of degree up to 6 a term with probability 1/2, about 105
    /// terms; up to `reducers` reducer rows of distinct leading monomials; and
    /// `reductants` reductants, each leading at a reducer's leading
    /// monomial, some a reducer's copy and some the reductant before them
    /// again, then `tail` copies of rows before them. Each reducer's
    /// signature lies below those of the reductants that lead at its
    /// column, some reducers share a reductant's signature, and the tail's
    /// signatures lie past every reducer's.
    pub(crate) fn random(
        field: F,
        seed: u64,
        elements: usize,
        reducers: usize,
        reductants: usize,
        tail: usize,
        long: bool,
    ) -> Round<F> {
        let p = field.characteristic();
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut rng = Rng::new(seed ^ p.rotate_left(13) ^ 0x5157);
        let mut basis: Basis<F, Rank1> = Basis::new();
        while basis.len() < elements {
            let mut terms = Vec::new();
            if long {
                for e in exponents(6) {
                    if rng.below(2) == 0 {
                        let m = table.intern(&e).unwrap();
                        terms.push((m, field.coef(1 + rng.below(p - 1))));
                    }
                }
            }
            for _ in 0..if long { 0 } else { 2 + rng.below(6) } {
                let mut e = [0u32; 4];
                let mut degree = 1 + rng.below(4) as u32;
                for x in &mut e {
                    let k = rng.below(u64::from(degree) + 1) as u32;
                    *x = k;
                    degree -= k;
                }
                let m = table.intern(&e).unwrap();
                terms.push((m, field.coef(1 + rng.below(p - 1))));
            }
            let mut g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
            if g.is_zero() || table.degree(g.lead()) == 0 {
                continue;
            }
            g.make_monic(&field);
            basis.push(g, false).unwrap();
        }
        // Reducers: distinct leading monomials, each at a time drawn for it.
        let span = 4 * reducers as u64 + 4;
        let mut leads: Vec<MonoId> = Vec::new();
        let mut reducer_rows: Vec<(RowSource, u32)> = Vec::new();
        for _ in 0..reducers {
            let g = ElementId::at(rng.below(elements as u64) as usize);
            let mut e = [0u32; 4];
            let mut degree = rng.below(3) as u32;
            for x in &mut e {
                let k = rng.below(u64::from(degree) + 1) as u32;
                *x = k;
                degree -= k;
            }
            let t = table.intern(&e).unwrap();
            let lead = table.mul(t, basis.lead(g)).unwrap();
            if leads.contains(&lead) {
                continue;
            }
            leads.push(lead);
            let row = RowSource {
                element: g,
                multiplier: t,
            };
            reducer_rows.push((row, rng.below(span) as u32));
        }
        // Reductants: each leads at a reducer's leading monomial, after it.
        let mut reductant_rows: Vec<(RowSource, u32)> = Vec::new();
        for _ in 0..reductants {
            let k = rng.below(reducer_rows.len() as u64) as usize;
            let (reducer, time) = reducer_rows[k];
            let lead = leads[k];
            let index = time + 1 + rng.below(span) as u32;
            let row = match rng.below(4) {
                0 => reducer,
                1 if !reductant_rows.is_empty() => reductant_rows[reductant_rows.len() - 1].0,
                _ => {
                    let divisors: Vec<ElementId> = (0..elements)
                        .map(ElementId::at)
                        .filter(|&g| table.divides(basis.lead(g), lead))
                        .collect();
                    let g = divisors[rng.below(divisors.len() as u64) as usize];
                    RowSource {
                        element: g,
                        multiplier: table.div(lead, basis.lead(g)).unwrap(),
                    }
                }
            };
            reductant_rows.push((row, index));
            // A reducer at the reductant's signature, where it has none.
            if rng.below(8) == 0 {
                let r = rng.below(reducer_rows.len() as u64) as usize;
                if reducer_rows[r].1 < index && r != k {
                    reducer_rows[r].1 = index;
                }
            }
        }
        // A reducer moved past a reductant that leads at its column breaks
        // the order the reductant needs: such reductants take its time + 1.
        for (row, index) in &mut reductant_rows {
            let lead = table.mul(row.multiplier, basis.lead(row.element)).unwrap();
            let k = leads.iter().position(|&l| l == lead).unwrap();
            if *index <= reducer_rows[k].1 {
                *index = reducer_rows[k].1 + 1;
            }
        }
        // The tail: copies of rows before it, past every row's signature.
        let last = 1 + reducer_rows
            .iter()
            .chain(&reductant_rows)
            .map(|&(_, index)| index)
            .max()
            .unwrap();
        for j in 0..tail {
            let copy = if reductant_rows.is_empty() || rng.below(2) == 0 {
                reducer_rows[rng.below(reducer_rows.len() as u64) as usize].0
            } else {
                reductant_rows[rng.below(reductant_rows.len() as u64) as usize].0
            };
            reductant_rows.push((copy, last + j as u32));
        }
        Round::of(field, table, basis, reductant_rows, reducer_rows)
    }

    /// The reduction from fresh buffers under an unbounded budget.
    pub(crate) fn reduce(&self, combine: Option<&mut Combine>) -> Reduction<F> {
        self.reduce_in(combine, &mut Buffers::new())
    }

    /// The reduction from `bufs` under an unbounded budget.
    pub(crate) fn reduce_in(
        &self,
        combine: Option<&mut Combine>,
        bufs: &mut Buffers<F>,
    ) -> Reduction<F> {
        matrix::reduce_sig::<F, Dense16, Grevlex, Rank1, NoCofactors>(
            &self.field,
            &self.table,
            &Grevlex,
            &self.basis,
            &self.matrix,
            &self.sigs,
            &mut NoCofactors,
            &mut budget(),
            bufs,
            combine,
        )
        .unwrap()
    }

    /// Whether the round's reductants take bands.
    fn takes_bands(&self) -> bool {
        matrix::takes_bands::<F, Dense16, NoCofactors>(&self.matrix)
    }
}

/// The exponents of every monomial on four variables of degree up to `d`.
fn exponents(d: u32) -> Vec<[u32; 4]> {
    let mut out = Vec::new();
    for a in 0..=d {
        for b in 0..=d - a {
            for c in 0..=d - a - b {
                for e in 0..=d - a - b - c {
                    out.push([a, b, c, e]);
                }
            }
        }
    }
    out
}

/// What a reduction made and which rows it ruled zero, in the order
/// made: each made row's source row, columns, and coefficients; the zero
/// rows; every column's pivot; and the order the rows were taken in.
pub(crate) type Made<C> = (
    Vec<(RowId, Vec<MonoId>, Vec<C>)>,
    Vec<RowId>,
    Vec<(u8, usize)>,
    Vec<RowId>,
);

pub(crate) fn made<F: Field>(reduction: &Reduction<F>) -> Made<F::Coef> {
    let rows = reduction
        .made
        .iter()
        .map(|r| (r.row, r.cols.clone(), r.coefs.clone()))
        .collect();
    let pivots = reduction
        .pivot
        .iter()
        .map(|p| match *p {
            Pivot::None => (0, 0),
            Pivot::Row(q) => (1, q.index()),
            Pivot::New(k) => (2, k.index()),
        })
        .collect();
    (
        rows,
        reduction.zero.clone(),
        pivots,
        reduction.taken.clone(),
    )
}

/// Compares the reduction with tests against the one without on random
/// rounds over `field`, of long elements where `long` is true: the rows'
/// outcome, the order, and the pivots are the same.
fn compare<F: Field>(field: F, seeds: u64, long: bool) {
    let p = field.characteristic();
    for seed in 0..seeds {
        let tail = [0, 6, 20, 80][seed as usize % 4];
        let round = Round::random(
            field,
            seed,
            4 + seed as usize % 10,
            6 + 3 * (seed as usize % 9),
            10 + 7 * (seed as usize % 11),
            tail,
            long,
        );
        assert_eq!(round.takes_bands(), long, "p {p}, seed {seed}: bands");
        let exact = round.reduce(None);
        let tested = round.reduce(Some(&mut Combine::new(seed ^ 0x2e70)));
        assert_eq!(made(&tested), made(&exact), "p {p}, seed {seed}");
    }
}

/// Test 1: the tests rule the rows zero that reduce to zero, and only
/// those, over primes where s is 16, 12, 4, 3, and 2, and in `Gf64`, where
/// it is 1.
#[test]
fn the_tests_rule_the_rows_zero_that_reduce_to_zero() {
    for p in [5, 7, 251, 1297, 46337, 46349, 65521, 2_147_483_647] {
        compare(Gf31::new(p).unwrap(), 48, false);
    }
    for p in [(1u64 << 31) + 11, 18_446_744_073_709_551_557] {
        compare(Gf64::new(p).unwrap(), 48, false);
    }
}

/// Test 1: at p = 2 and p = 3, where `zero_combinations` is None, the
/// reduction with a `Combine` makes what the one without makes.
#[test]
fn two_and_three_take_no_test() {
    for p in [2, 3] {
        let field = Gf31::new(p).unwrap();
        for seed in 0..24 {
            let round = Round::random(field, seed, 5, 10, 30, 80, false);
            let exact = round.reduce(None);
            let tested = round.reduce(Some(&mut Combine::new(seed)));
            assert_eq!(made(&tested), made(&exact), "p {p}, seed {seed}");
        }
    }
}

/// Test 2: the number of combinations a test takes.
#[test]
fn the_number_of_combinations() {
    let want: [(u64, Option<usize>); 16] = [
        (2, None),
        (3, None),
        (5, Some(16)),
        (7, Some(12)),
        (223, Some(4)),
        (1291, Some(4)),
        (1297, Some(3)),
        (46337, Some(3)),
        (46349, Some(2)),
        (65521, Some(2)),
        (2_147_483_647, Some(2)),
        ((1 << 31) + 11, Some(1)),
        ((1 << 61) - 1, Some(1)),
        (18_446_744_073_709_551_557, Some(1)),
        (211, Some(5)),
        (13, Some(9)),
    ];
    for (p, s) in want {
        assert_eq!(zero_combinations(p), s, "p {p}");
    }
    let bound = 1u128 << 31;
    for p in [
        5u64, 7, 11, 13, 101, 211, 223, 1291, 1297, 32003, 46337, 46349, 65521,
    ] {
        let s = zero_combinations(p).unwrap() as u32;
        let q = u128::from(p - 1);
        assert!(q.pow(s) >= bound, "p {p}: (p − 1)^{s} below 2^31");
        assert!(
            q.pow(s - 1) < bound,
            "p {p}: (p − 1)^{exponent} reaches 2^31",
            exponent = s - 1
        );
    }
}

/// The round of test 3 over `field` on x, y, z, w: the reducer x + 1 at
/// signature 0, and as reductants, at signatures 1 on, `copies` copies of
/// x + 1, then x + 2 where `last` is true, which reduces to the constant
/// 1, the matrix's last column.
pub(crate) fn failing<F: Field>(field: F, copies: usize, last: bool) -> Round<F> {
    let mut table = Table::new(Dense16::new(4), &mut Xoshiro256PlusPlus::seed_from_u64(9));
    let x = table.intern(&[1, 0, 0, 0]).unwrap();
    let one = table.intern(&[0; 4]).unwrap();
    let mut basis: Basis<F, Rank1> = Basis::new();
    for c in [1, 2] {
        let mut terms = vec![(x, field.coef(1)), (one, field.coef(c))];
        let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
        basis.push(g, false).unwrap();
    }
    let row = |k: usize| RowSource {
        element: ElementId::at(k),
        multiplier: one,
    };
    let mut reductants: Vec<(RowSource, u32)> =
        (0..copies).map(|k| (row(0), 1 + k as u32)).collect();
    if last {
        reductants.push((row(1), 1 + copies as u32));
    }
    Round::of(field, table, basis, reductants, vec![(row(0), 0)])
}

/// Test 3: where the round's last row reduces to the last column, the
/// reduction with tests makes that row and rules the copies before it
/// zero, as the reduction without does; where every row reduces to zero,
/// both rule every row zero.
#[test]
fn a_failed_test() {
    let field = Gf31::new(65521).unwrap();
    let round = failing(field, 8, true);
    let exact = round.reduce(None);
    let tested = round.reduce(Some(&mut Combine::new(5)));
    assert_eq!(made(&tested), made(&exact));
    assert_eq!(exact.made.len(), 1);
    let last = MonoId::at(round.matrix.columns() - 1);
    assert_eq!(exact.made[0].cols, vec![last], "x + 2 reduces to 1");
    assert_eq!(exact.zero, (0..8).map(RowId::at).collect::<Vec<_>>());

    let round = failing(field, 10, false);
    let exact = round.reduce(None);
    let tested = round.reduce(Some(&mut Combine::new(6)));
    assert_eq!(made(&tested), made(&exact));
    assert!(exact.made.is_empty());

    let field = Gf64::new((1 << 61) - 1).unwrap();
    let round = failing(field, 10, false);
    let tested = round.reduce(Some(&mut Combine::new(7)));
    assert_eq!(tested.zero, (0..10).map(RowId::at).collect::<Vec<_>>());
}

/// The rounds of test 10 over GF(65521) on x, y, z, w: an element g of 80
/// terms of degree up to 6, the constant not among them, and h = g + 1.
/// The reducer g at signature 0, and as reductants, at signatures 1 on,
/// `copies` copies of g, then h where `last` is true, which reduces to the
/// constant 1, the matrix's last column.
fn banded(copies: usize, last: bool) -> Round<Gf31> {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(4), &mut Xoshiro256PlusPlus::seed_from_u64(10));
    let mut rng = Rng::new(0x5e7a);
    let mut monomials: Vec<[u32; 4]> = exponents(6).into_iter().filter(|e| *e != [0; 4]).collect();
    // A Fisher–Yates shuffle; g takes the first 80.
    for k in (1..monomials.len()).rev() {
        monomials.swap(k, rng.below(k as u64 + 1) as usize);
    }
    let mut terms: Vec<(MonoId, u32)> = monomials[..80]
        .iter()
        .map(|e| (table.intern(e).unwrap(), field.coef(1 + rng.below(65520))))
        .collect();
    let mut g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
    g.make_monic(&field);
    let one = table.intern(&[0; 4]).unwrap();
    let mut terms: Vec<(MonoId, u32)> = g.monomials.iter().copied().zip(g.coefs.clone()).collect();
    terms.push((one, field.coef(1)));
    let h = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    basis.push(g, false).unwrap();
    basis.push(h, false).unwrap();
    let row = |k: usize| RowSource {
        element: ElementId::at(k),
        multiplier: one,
    };
    let mut reductants: Vec<(RowSource, u32)> =
        (0..copies).map(|k| (row(0), 1 + k as u32)).collect();
    if last {
        reductants.push((row(1), 1 + copies as u32));
    }
    Round::of(field, table, basis, reductants, vec![(row(0), 0)])
}

/// Test 10: signature-zero-time.13. On rows that take bands, the
/// reduction with tests rules every copy of g zero, and where h follows
/// the copies it makes h's row, as the reduction without does.
#[test]
fn a_band_completes_the_streak() {
    let round = banded(10, false);
    assert!(round.takes_bands(), "80 terms a row take bands");
    let exact = round.reduce(None);
    let tested = round.reduce(Some(&mut Combine::new(10)));
    assert_eq!(made(&tested), made(&exact));
    assert!(exact.made.is_empty());
    assert_eq!(tested.zero, (0..10).map(RowId::at).collect::<Vec<_>>());

    let round = banded(8, true);
    assert!(round.takes_bands(), "80 terms a row take bands");
    let exact = round.reduce(None);
    let tested = round.reduce(Some(&mut Combine::new(11)));
    assert_eq!(made(&tested), made(&exact));
    assert_eq!(exact.made.len(), 1);
    let last = MonoId::at(round.matrix.columns() - 1);
    assert_eq!(exact.made[0].cols, vec![last], "h reduces to 1");
}

/// Test 11: signature-zero-time.13 over random rounds whose rows take
/// bands, below 2^16 and from it: the tests rule the rows zero that reduce
/// to zero, and only those.
#[test]
fn random_banded_rounds() {
    for p in [65521, 2_147_483_647] {
        compare(Gf31::new(p).unwrap(), 24, true);
    }
}
