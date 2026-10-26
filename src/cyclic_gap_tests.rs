//! cyclic-gap, the crate's test code: the rows to reduce taken by
//! decreasing leading column, each on its own and in blocks of
//! combinations, against a reference elimination of the test's own; the
//! edges of the order; and a record learned in that order, replayed
//! (`iterations/0066-cyclic-gap/plan.md`, _Test code_, 1 to 4, test 1
//! without the plan's stats, and test 4 on a dense cubic system of four
//! variables in place of cyclic-6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Reverse;
use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::f4::Halt;
use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{self, BLOCK, Buffers, Combine, Matrix, Reduced, Reduction};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::systems::{Rng, dense, random_system};
use crate::table::Table;
use crate::trace::{NoTrace, RowSource};
use crate::{MonomialOrder, Params, Threads};

/// The primes of the combinations: below 2^16 and the largest below 2^31,
/// each past the combinations' bound of 46341.
const COMBINING: [u64; 2] = [65521, 2_147_483_647];

/// The primes of each row on its own: 7, below the combinations' bound,
/// and the two of `COMBINING`.
const ALONE: [u64; 3] = [7, 65521, 2_147_483_647];

fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

pub(crate) fn budget() -> Budget {
    Budget::new(Instant::now(), &params(1)).unwrap()
}

/// The shape of a random matrix, as `plain_band_tests.rs`'s: `elements`
/// basis elements, each of `least` to `most` terms of degree 1 to
/// `degree`, led by a monomial of degree `lead` where it is Some; the
/// halves of `pairs` random S-pairs as given rows, every `twice`-th
/// draw's rows given twice; and the first `doubles` elements of `G_min`
/// given twice with multiplier 1, one copy a row to reduce that the other
/// reduces to zero.
#[derive(Clone)]
pub(crate) struct Shape {
    pub(crate) elements: usize,
    pub(crate) pairs: usize,
    pub(crate) twice: usize,
    pub(crate) doubles: usize,
    pub(crate) least: u64,
    pub(crate) most: u64,
    pub(crate) degree: u64,
    pub(crate) lead: Option<u32>,
}

/// A random matrix over GF(p) on four variables: a basis of random monic
/// polynomials, given rows of `Shape`, and the reducers symbolic
/// preprocessing adds.
pub(crate) struct Random {
    pub(crate) field: Gf31,
    pub(crate) basis: Basis<Gf31, Rank1>,
    pub(crate) matrix: Matrix,
}

impl Random {
    pub(crate) fn new(p: u64, seed: u64, shape: &Shape) -> Random {
        Random::with_rows(p, seed, shape, |_, _, rows| rows)
    }

    /// As `new`, the given rows passed through `edit` before the build:
    /// it gets the table, the basis, and the rows `new` would give.
    pub(crate) fn with_rows(
        p: u64,
        seed: u64,
        shape: &Shape,
        edit: impl FnOnce(&mut Table<Dense16>, &Basis<Gf31, Rank1>, Vec<RowSource>) -> Vec<RowSource>,
    ) -> Random {
        let field = Gf31::new(p).unwrap();
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut rng = Rng::new(seed ^ p.rotate_left(29) ^ 0xc6a9);
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        while basis.len() < shape.elements {
            let mut terms = Vec::new();
            if let Some(d) = shape.lead {
                let mut e = [0u32; 4];
                let mut degree = d;
                for x in &mut e[..3] {
                    let k = rng.below(u64::from(degree) + 1) as u32;
                    *x = k;
                    degree -= k;
                }
                e[3] = degree;
                let m = table.intern(&e).unwrap();
                terms.push((m, field.coef(1 + rng.below(p - 1))));
            }
            for _ in 0..shape.least + rng.below(shape.most - shape.least + 1) {
                let mut e = [0u32; 4];
                let mut degree = 1 + rng.below(shape.degree) as u32;
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
            basis.push_min(&table, g).unwrap();
        }
        let min = basis.min().to_vec();
        let one = table.intern(&[0; 4]).unwrap();
        let mut rows = Vec::new();
        for k in 0..shape.pairs {
            let i = min[rng.below(min.len() as u64) as usize];
            let j = min[rng.below(min.len() as u64) as usize];
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
                if shape.twice > 0 && k % shape.twice == 0 {
                    rows.push(row);
                }
            }
        }
        for &g in min.iter().take(shape.doubles) {
            let row = RowSource {
                element: g,
                multiplier: one,
            };
            rows.push(row);
            rows.push(row);
        }
        let rows = edit(&mut table, &basis, rows);
        let matrix = matrix::build(
            &mut table,
            &basis,
            &Grevlex,
            1,
            rows,
            Vec::new(),
            &mut budget(),
        )
        .unwrap();
        Random {
            field,
            basis,
            matrix,
        }
    }

    pub(crate) fn reduce(&self, combine: Option<&mut Combine>) -> Reduction<Gf31> {
        self.reduce_in(combine, &mut budget()).unwrap()
    }

    /// `matrix::reduce` of the matrix under the budget `b`, full.
    pub(crate) fn reduce_in(
        &self,
        combine: Option<&mut Combine>,
        b: &mut Budget,
    ) -> Result<Reduction<Gf31>, Halt> {
        self.reduce_with(combine, None, b)
    }

    /// `matrix::reduce` of the matrix under the budget `b`, full, its rows
    /// each on its own tested by `tests` where that is Some.
    pub(crate) fn reduce_with(
        &self,
        combine: Option<&mut Combine>,
        tests: Option<&mut Combine>,
        b: &mut Budget,
    ) -> Result<Reduction<Gf31>, Halt> {
        matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
            &self.field,
            &self.basis,
            &self.matrix,
            &mut NoCofactors,
            b,
            true,
            combine,
            &mut Buffers::new(),
            tests,
        )
    }

    pub(crate) fn lead(&self, r: usize) -> u32 {
        self.matrix.row(RowId::at(r))[0].raw()
    }

    /// Row r as a dense vector over the matrix's columns, mod p.
    pub(crate) fn dense(&self, r: usize) -> Vec<u64> {
        let mut v = vec![0u64; self.matrix.columns()];
        let coefs = &self.basis.element(self.matrix.rows[r].element).coefs;
        for (&c, &a) in self.matrix.row(RowId::at(r)).iter().zip(coefs) {
            v[c.index()] = u64::from(a);
        }
        v
    }

    /// The rows to reduce, as `matrix::reduce` sets the pivots
    /// (f4-gfp.5): every reducer row the pivot of its leading column, then
    /// the given rows in row order, each the pivot of its column where it
    /// has none, and in place of a pivot row of more terms, which goes to
    /// the rows to reduce. The pivot rows, and the rows to reduce in the
    /// order of cyclic-gap.1: by decreasing leading column, those of one
    /// leading column by decreasing row index.
    pub(crate) fn split(&self) -> (Vec<usize>, Vec<usize>) {
        let m = &self.matrix;
        let mut pivot: Vec<Option<usize>> = vec![None; m.columns()];
        for r in m.given..m.rows.len() {
            pivot[self.lead(r) as usize] = Some(r);
        }
        let mut rest = Vec::new();
        for r in 0..m.given {
            let c = self.lead(r) as usize;
            match pivot[c] {
                Some(q) if m.row(RowId::at(q)).len() > m.row(RowId::at(r)).len() => {
                    rest.push(q);
                    pivot[c] = Some(r);
                }
                Some(_) => rest.push(r),
                None => pivot[c] = Some(r),
            }
        }
        rest.sort_unstable_by_key(|&r| Reverse((self.lead(r), r)));
        (pivot.into_iter().flatten().collect(), rest)
    }
}

/// a^(p − 2) mod p.
fn inverse(a: u64, p: u64) -> u64 {
    let (mut base, mut e, mut out) = (a % p, p - 2, 1u64);
    while e > 0 {
        if e & 1 == 1 {
            out = (u128::from(out) * u128::from(base) % u128::from(p)) as u64;
        }
        base = (u128::from(base) * u128::from(base) % u128::from(p)) as u64;
        e >>= 1;
    }
    out
}

/// The test's own row echelon form mod p: per column, the row that leads
/// there. A vector reduces to zero where it lies in the span of the rows
/// held.
pub(crate) struct Echelon {
    p: u64,
    rows: Vec<Option<Vec<u64>>>,
}

impl Echelon {
    pub(crate) fn new(p: u64, columns: usize) -> Echelon {
        Echelon {
            p,
            rows: vec![None; columns],
        }
    }

    /// Reduces `v` by the rows held, and holds it where it is not zero:
    /// whether it was.
    pub(crate) fn insert(&mut self, mut v: Vec<u64>) -> bool {
        let p = self.p;
        for c in 0..v.len() {
            if v[c] == 0 {
                continue;
            }
            if let Some(row) = &self.rows[c] {
                let k = v[c];
                for (x, &y) in v.iter_mut().zip(row) {
                    *x = (*x + p - (u128::from(k) * u128::from(y) % u128::from(p)) as u64) % p;
                }
            } else {
                let s = inverse(v[c], p);
                for x in &mut v {
                    *x = (u128::from(*x) * u128::from(s) % u128::from(p)) as u64;
                }
                self.rows[c] = Some(v);
                return false;
            }
        }
        true
    }
}

/// The reference of each row on its own: the pivot rows held, then the
/// rows to reduce in the order of cyclic-gap.1, each zero or a new pivot.
/// The rows that reduce to zero and the rows that make a new pivot, each
/// in that order.
pub(crate) fn alone(m: &Random) -> (Vec<u32>, Vec<u32>) {
    let (pivots, rest) = m.split();
    let mut e = Echelon::new(m.field.characteristic(), m.matrix.columns());
    for &r in &pivots {
        assert!(!e.insert(m.dense(r)), "a pivot row is not zero");
    }
    let (mut zero, mut made) = (Vec::new(), Vec::new());
    for r in rest {
        if e.insert(m.dense(r)) {
            zero.push(r as u32);
        } else {
            made.push(r as u32);
        }
    }
    (zero, made)
}

/// The made rows, each as its columns and coefficients, by leading column.
pub(crate) fn rows(reduction: &Reduction<Gf31>) -> Vec<(Vec<MonoId>, Vec<u32>)> {
    let mut made: Vec<&Reduced<Gf31>> = reduction.made.iter().collect();
    made.sort_by_key(|r| r.cols[0]);
    made.iter()
        .map(|r| (r.cols.clone(), r.coefs.clone()))
        .collect()
}

/// Each row on its own: `reduction`'s zero rows and made rows, each in its
/// order, are the reference's.
fn assert_alone(name: &str, m: &Random, reduction: &Reduction<Gf31>) {
    let (zero, made) = alone(m);
    let got: Vec<u32> = reduction
        .made
        .iter()
        .map(|r| r.row.index() as u32)
        .collect();
    let got_zero: Vec<u32> = reduction.zero.iter().map(|r| r.index() as u32).collect();
    assert_eq!(got_zero, zero, "{name}: the rows that reduced to zero");
    assert_eq!(got, made, "{name}: the rows that made new pivots");
}

/// The combinations: every made row names the first row of a block of
/// the item `plain-zero-time`'s rule, which cuts the blocks of cyclic-gap.2,
/// as many as the block's pivots, in block order (`plain_zero_tests::blocks`).
fn assert_blocks(name: &str, m: &Random, reduction: &Reduction<Gf31>, seed: u64) {
    crate::plain_zero_tests::assert_blocks(
        name,
        reduction,
        &crate::plain_zero_tests::blocks(m, seed),
    );
}

/// The short rows of `combine_tests.rs`'s matrices.
pub(crate) fn short(elements: usize, pairs: usize, twice: usize, doubles: usize) -> Shape {
    Shape {
        elements,
        pairs,
        twice,
        doubles,
        least: 2,
        most: 7,
        degree: 4,
        lead: None,
    }
}

/// Short rows whose elements all lead at degree 4 over tails of degree 3
/// at most, so that `G_min` holds every element whose leading monomial no
/// earlier element has.
pub(crate) fn led(elements: usize, pairs: usize) -> Shape {
    Shape {
        elements,
        pairs,
        twice: 0,
        doubles: 0,
        least: 2,
        most: 7,
        degree: 3,
        lead: Some(4),
    }
}

/// Long rows, as `plain_band_tests.rs`'s: elements that lead with a
/// monomial of degree 7 and draw 200 to 300 tail terms of degree at most
/// 6.
pub(crate) fn long(elements: usize, pairs: usize, twice: usize, doubles: usize) -> Shape {
    Shape {
        elements,
        pairs,
        twice,
        doubles,
        least: 200,
        most: 300,
        degree: 6,
        lead: Some(7),
    }
}

/// Test 1: cyclic-gap.1, each row on its own. `matrix::reduce` without a
/// `Combine` at 7, 65521, and 2^31 − 1, and with one at 7, where p is
/// below the combinations' bound: the rows that reduce to zero and the
/// rows that make new pivots, each in its order, are the reference's, the
/// rows to reduce taken by decreasing leading column.
#[test]
fn rows_alone_go_from_the_right() {
    let mut cases = 0;
    for p in ALONE {
        for seed in 0..24u64 {
            let shape = short(
                5 + (seed % 8) as usize,
                4 + (seed % 24) as usize,
                [0, 1, 2, 5][seed as usize % 4],
                (seed % 3) as usize,
            );
            let m = Random::new(p, seed, &shape);
            let name = format!("p {p}, seed {seed}");
            assert_alone(&name, &m, &m.reduce(None));
            if !matrix::combines_rows(p) {
                let mut combine = Combine::new(seed);
                assert_alone(
                    &format!("{name}, a Combine"),
                    &m,
                    &m.reduce(Some(&mut combine)),
                );
            }
            cases += 1;
        }
    }
    assert_eq!(cases, 72);
}

/// Test 2: cyclic-gap.1, .2, the combinations. `matrix::reduce` with a
/// `Combine` at 65521 and 2^31 − 1 on matrices of `2·BLOCK` + 1 rows to
/// reduce or more, of short rows and of long rows: every made row names
/// the first row of a block of the order, cut by `plain-zero-time`'s rule,
/// as many as the block's pivots, in block order (`assert_blocks`); and the
/// made rows are those of each row on its own.
#[test]
fn blocks_go_from_the_right() {
    let (mut short_cases, mut long_cases, mut banded) = (0, 0, 0);
    for p in COMBINING {
        for seed in 0..12u64 {
            for is_long in [false, true] {
                let shape = if is_long {
                    long(12 + (seed % 6) as usize, 160 + 8 * seed as usize, 3, 12)
                } else {
                    short(30 + (seed % 10) as usize, 180 + 6 * seed as usize, 4, 20)
                };
                let m = Random::new(p, 0x6a90 + seed, &shape);
                let (_, rest) = m.split();
                if rest.len() < 2 * BLOCK + 1 {
                    continue;
                }
                let name = format!("p {p}, seed {seed}, long {is_long}");
                let mut combine = Combine::new(0x6a91 ^ seed);
                let combined = m.reduce(Some(&mut combine));
                assert_blocks(&name, &m, &combined, seed);
                assert_eq!(rows(&combined), rows(&m.reduce(None)), "{name}");
                if is_long {
                    long_cases += 1;
                    banded +=
                        usize::from(matrix::takes_bands::<Gf31, Dense16, NoCofactors>(&m.matrix));
                } else {
                    short_cases += 1;
                }
            }
        }
    }
    assert!(short_cases >= 12, "{short_cases} matrices of short rows");
    assert!(long_cases >= 6, "{long_cases} matrices of long rows");
    assert!(banded >= 6, "{banded} matrices took bands");
}

/// A matrix of `shape` at p whose rows to reduce number `n`: the first
/// element of `G_min` given with multiplier 1 as many times more as the
/// shape's rows fall short of `n`, every copy but the first a row to
/// reduce. None where the shape's rows pass `n`.
pub(crate) fn with_rest(p: u64, seed: u64, shape: &Shape, n: usize) -> Option<Random> {
    let base = Random::new(p, seed, shape).split().1.len();
    if base > n {
        return None;
    }
    let m = Random::with_rows(p, seed, shape, |table, basis, mut rows| {
        let one = table.intern(&[0; 4]).unwrap();
        let g = basis.min()[0];
        for _ in 0..=n - base {
            rows.push(RowSource {
                element: g,
                multiplier: one,
            });
        }
        rows
    });
    (m.split().1.len() == n).then_some(m)
}

/// Test 3: the edges of the order, at 65521, each with and without a
/// `Combine`: no row to reduce; one; rows of one leading column only,
/// which go by decreasing row index; and exactly `BLOCK` and `BLOCK` + 1
/// rows to reduce, one block and two, the second the one row that leads
/// farthest left. As tests 1 and 2 predict, and the made rows those of
/// each row on its own.
#[test]
fn the_edges_of_the_order() {
    let p = 65521;
    let mut cases: Vec<(String, Random)> = Vec::new();
    for seed in 0..4u64 {
        // Every element of G_min given once: every row a pivot.
        let m = Random::with_rows(p, seed, &led(5, 0), |table, basis, _| {
            let one = table.intern(&[0; 4]).unwrap();
            let min = basis.min();
            min.iter()
                .map(|&g| RowSource {
                    element: g,
                    multiplier: one,
                })
                .collect()
        });
        assert!(m.split().1.is_empty(), "seed {seed}: a row to reduce");
        cases.push((format!("no row, seed {seed}"), m));
        let one = with_rest(p, seed, &led(4, 0), 1).expect("one row to reduce");
        cases.push((format!("one row, seed {seed}"), one));
    }
    for seed in 0..8u64 {
        let m = one_column(p, seed);
        let (_, rest) = m.split();
        let leads: Vec<u32> = rest.iter().map(|&r| m.lead(r)).collect();
        assert!(
            rest.len() >= 3,
            "seed {seed}: {rows} rows to reduce",
            rows = rest.len()
        );
        assert!(
            leads.windows(2).all(|w| w[0] == w[1]),
            "seed {seed}: {leads:?}"
        );
        assert!(
            rest.windows(2).all(|w| w[0] > w[1]),
            "seed {seed}: {rest:?}"
        );
        cases.push((format!("one leading column, seed {seed}"), m));
    }
    for n in [BLOCK, BLOCK + 1] {
        let found = (0..40u64)
            .find_map(|seed| with_rest(p, 0x6a92 + seed, &led(12, 30), n).map(|m| (seed, m)));
        let (seed, m) = found.unwrap_or_else(|| panic!("no matrix of {n} rows to reduce"));
        if n == BLOCK + 1 {
            let (_, rest) = m.split();
            let least = rest.iter().map(|&r| m.lead(r)).min().unwrap();
            assert_eq!(
                m.lead(rest[BLOCK]),
                least,
                "the second block leads farthest left"
            );
        }
        cases.push((format!("{n} rows, seed {seed}"), m));
    }
    assert!(cases.len() >= 12, "{len} cases", len = cases.len());
    for (name, m) in &cases {
        let alone_reduced = m.reduce(None);
        assert_alone(name, m, &alone_reduced);
        let mut combine = Combine::new(0x6a93);
        let combined = m.reduce(Some(&mut combine));
        assert_blocks(name, m, &combined, 0x6a93);
        assert_eq!(rows(&combined), rows(&alone_reduced), "{name}");
    }
}

/// A matrix whose rows to reduce all lead at one column: the two halves
/// of one S-pair of `G_min`, each given three times, interleaved.
fn one_column(p: u64, seed: u64) -> Random {
    Random::with_rows(p, seed, &led(6, 0), |table, basis, _| {
        let min = basis.min();
        let (i, j) = (min[0], min[1 + seed as usize % (min.len() - 1)]);
        let lcm = table.lcm(basis.lead(i), basis.lead(j)).unwrap();
        let mut half = |g: ElementId| RowSource {
            element: g,
            multiplier: table.div(lcm, basis.lead(g)).unwrap(),
        };
        let (a, b) = (half(i), half(j));
        vec![a, b, a, b, a, b]
    })
}

/// Test 4: the record. A learn run, `crate::learn`, with signatures off
/// and on, of random systems over 65521 and of a dense cubic system on four
/// variables over 65521: the record's replay at 2^31 − 1 and at a prime of
/// [2^61, 2^62) returns the basis a call at that prime returns.
#[test]
fn a_record_learned_from_the_right_replays() {
    const PRIMES: [u64; 2] = [2_147_483_647, 2_536_775_269_993_504_451];
    let mut texts: Vec<(String, String)> = (0..16u64)
        .map(|seed| (format!("random {seed}"), random_system(65521, seed)))
        .collect();
    texts.push(("cubic-4".to_string(), dense(4, 3, 65521, 1)));
    let mut replayed = 0;
    for (name, text) in &texts {
        for signatures in [false, true] {
            let mut learn_params = params(0x6a94);
            learn_params.signatures = signatures;
            let (outcome, record) = crate::learn(text, &learn_params, &mut NoTrace)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let Some(record) = record else {
                assert!(outcome.basis.is_none(), "{name}: a basis and no record");
                continue;
            };
            for q in PRIMES {
                let at_q = over(text, q);
                let want = crate::groebner(&at_q, &params(0x6a95))
                    .unwrap_or_else(|e| panic!("{name} at {q}: {e:?}"))
                    .basis;
                let got = crate::replay(&at_q, &params(0x6a95), &record, &mut NoTrace)
                    .unwrap_or_else(|e| panic!("{name} at {q}, signatures {signatures}: {e:?}"))
                    .basis;
                assert_eq!(got, want, "{name} at {q}, signatures {signatures}");
                replayed += 1;
            }
        }
    }
    assert!(replayed >= 40, "{replayed} replays");
}

/// `text` with its characteristic, its second line, set to `p`.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let variables = lines.next().unwrap();
    lines.next().unwrap();
    format!(
        "{variables}\n{p}\n{rest}",
        rest = lines.next().unwrap_or("")
    )
}
