//! plain-reduction-time, the crate's test code: a band's combinations
//! formed by `Reducer::add_band`, the plain run's combinations in bands
//! against the reduction of each row on its own, the bands' spends under a
//! deadline, and the reduction at primes below the combinations' bound
//! (`iterations/0062-plain-reduction-time/plan.md`, _Test code_, 1, 2
//! without its check that `zero` is empty, 5, and 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stop;
use crate::band_tests::setup_units;
use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::field::{BAND, Field, Gf31};
use crate::halt::Halt;
use crate::id::{RowId, RowSource};
use crate::matrix::{self, BLOCK, Buffers, Combine, Reducer};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::shared_tests::{Random, budget, passed_after, read_all, rows};
use crate::systems::Rng;
use crate::table::Table;

/// The primes of the band: below 2^16, from 2^16, and the largest below
/// 2^31, each past the combinations' bound of 46341.
const PRIMES: [u64; 3] = [65521, 65537, 2_147_483_647];

/// The shape of a random matrix: basis elements, S-pairs drawn, every
/// `twice`-th draw's rows given twice, `doubles` basis elements given
/// twice as they are, `least` to `most` terms drawn an element, and terms
/// of degree at most `degree`. Where `lead` is Some(d), each element leads
/// with a monomial of degree d drawn on its own, d past `degree`, so that
/// no element's leading monomial divides another's unless the two are
/// equal, where `push_min` leaves the later out of `G_min`, and every drawn
/// term stands in its tail.
#[derive(Clone, Copy)]
struct Shape {
    elements: usize,
    pairs: usize,
    twice: usize,
    doubles: usize,
    least: u64,
    most: u64,
    degree: u64,
    lead: Option<u32>,
}

impl Random {
    /// A random matrix over GF(p) on four variables, as `band_tests.rs` builds one: a basis of
    /// random monic polynomials, given rows that are the halves of random S-pairs of the basis and
    /// basis elements given twice, and the reducers symbolic preprocessing adds. A basis element
    /// given twice is a row to reduce that the other copy reduces to zero.
    fn shaped(p: u64, seed: u64, shape: Shape) -> Random {
        let field = Gf31::new(p).unwrap();
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut rng = Rng::new(seed ^ p.rotate_left(19) ^ 0x9a1b);
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
                let m = table.intern(&e, &mut Budget::unbounded()).unwrap();
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
        let one = table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
        let mut rows = Vec::new();
        for k in 0..shape.pairs {
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

    /// Whether the plain run's reduction of this matrix takes bands.
    fn long(&self) -> bool {
        matrix::takes_bands::<Gf31, Dense16, NoCofactors>(&self.matrix)
    }

    /// Row r's coefficients, its element's.
    fn coefs(&self, r: RowId) -> &[u32] {
        &self
            .basis
            .element(self.matrix.rows[r.index()].element)
            .coefs
    }

    /// The longest row's entries.
    fn longest(&self) -> usize {
        (0..self.matrix.rows.len())
            .map(|r| self.matrix.row(RowId::at(r)).len())
            .max()
            .unwrap_or(0)
    }
}

/// A matrix of long rows, the shape varied by `seed`: elements that lead
/// with a monomial of degree 7 and draw 200 to 300 tail terms of degree at
/// most 6, of the 210 monomials there.
fn long(p: u64, seed: u64) -> Random {
    let shape = Shape {
        elements: 4 + (seed % 9) as usize,
        pairs: 6 + (seed % 23) as usize,
        twice: 3,
        doubles: (seed % 4) as usize,
        least: 200,
        most: 300,
        degree: 6,
        lead: Some(7),
    };
    Random::shaped(p, seed, shape)
}

/// How many given rows the reduction must reduce: those that are not the
/// pivot of their leading column.
fn to_reduce(m: &Random) -> usize {
    let mut leads: Vec<u32> = (0..m.matrix.rows.len())
        .map(|r| m.matrix.row(RowId::at(r)).lead().raw())
        .collect();
    let given = m.matrix.given;
    let reducers: std::collections::HashSet<u32> = leads[given..].iter().copied().collect();
    leads.truncate(given);
    leads.sort_unstable();
    leads.dedup();
    let pivots = leads.iter().filter(|c| !reducers.contains(c)).count();
    given - pivots
}

/// Test 1: plain-reduction-time.2, .6. `Reducer::add_band` against
/// four dense rows: the band, empty from `Reducer::new`, takes the
/// matrix's width at the first row added; each row added with four
/// coefficients, some 0, agrees with `sub_mul` of the negated coefficient
/// into a dense row of its own; and each row spends `BAND` times its
/// length, beside the band's fill, `BAND` units a column
/// (call-stop-latency.53).
#[test]
fn a_band_forms_four_combinations_in_one_pass() {
    let mut cases = 0;
    for p in PRIMES {
        for seed in 0..20u64 {
            let m = long(p, seed);
            let f = &m.field;
            let name = format!("p {p}, seed {seed}");
            let width = m.matrix.columns();
            let mut bufs = Buffers::new();
            let mut b = budget();
            let mut red = Reducer::new(f, &m.basis, &m.matrix, &mut bufs, &b);
            assert!(red.band.is_empty(), "{name}: a band before the first row");
            let mut dense: [Vec<i64>; BAND] = std::array::from_fn(|_| vec![0i64; width]);
            let mut rng = Rng::new(seed ^ p.rotate_left(3));
            let mut want = (BAND * width) as u64;
            for _ in 0..=rng.below(BLOCK as u64) {
                let r = RowId::at(rng.below(m.matrix.rows.len() as u64) as usize);
                let k: [u32; BAND] = std::array::from_fn(|_| {
                    if rng.below(5) == 0 {
                        0
                    } else {
                        f.coef(1 + rng.below(p - 1))
                    }
                });
                red.add_band(r, &k, &mut b).unwrap();
                want += (BAND * m.matrix.row(r).len()) as u64;
                for j in 0..BAND {
                    // SAFETY: every column of the matrix lies below its
                    // width.
                    unsafe {
                        f.sub_mul(
                            &mut dense[j],
                            &m.matrix.row(r).to_wide(),
                            m.coefs(r),
                            f.neg(k[j]),
                        );
                    };
                }
            }
            assert_eq!(red.band.len(), width, "{name}: the band's width");
            assert_eq!(b.spent(), want, "{name}: units spent");
            read_all(f, &mut red.band, &mut dense, &name);
            cases += 1;
        }
    }
    assert_eq!(cases, 60);
}

/// Test 2: plain-reduction-time.1, .4. On random matrices of long rows the
/// plain run's combinations, in bands where the matrix takes them, make
/// the rows that the reduction of each row on its own makes, by leading
/// column; and some of the matrices take bands.
#[test]
fn banded_combinations_make_what_rows_make() {
    let mut took = 0;
    for p in PRIMES {
        for seed in 0..40u64 {
            let m = long(p, seed);
            let name = format!("p {p}, seed {seed}");
            let exact = m.reduce(None);
            let combined = m.reduce(Some(&mut Combine::new(seed)));
            assert_eq!(rows(&combined), rows(&exact), "{name}");
            if m.long() && to_reduce(&m) >= 2 {
                took += 1;
            }
        }
    }
    assert!(took >= 30, "{took} matrices took bands");
}

/// Test 5: plain-reduction-time.6. A matrix of long rows whose first block
/// takes a band, with a deadline passed and unchecked, the setup's units
/// (`band_tests::setup_units`) and the band's fill, `BAND` units a column,
/// and `T` more between two full checks: the reduction stops with
/// `Deadline` after one full check and at most `T` + `BAND`·L units past
/// the setup's and the fill's, L the matrix's longest row, where unbounded
/// it spends more than four times that.
#[test]
fn a_deadline_stops_a_banded_block() {
    const T: u64 = 256;
    let mut cases = 0;
    for p in PRIMES {
        for seed in 0..100u64 {
            let m = long(p, seed);
            if !m.long() || to_reduce(&m) < 2 {
                continue;
            }
            let setup = setup_units(&m.matrix) + (BAND * m.matrix.columns()) as u64;
            let bound = T + (BAND * m.longest()) as u64;
            let mut b = budget();
            m.reduce_in(Some(&mut Combine::new(seed)), &mut b).unwrap();
            let total = b.spent();
            if total <= setup + 4 * bound {
                continue;
            }
            cases += 1;
            let name = format!("p {p}, seed {seed}");
            let mut b = passed_after(setup + T);
            let got = m.reduce_in(Some(&mut Combine::new(seed)), &mut b);
            assert!(
                matches!(got, Err(Halt::Stop(Stop::Deadline))),
                "{name}: not stopped by the deadline"
            );
            assert_eq!(b.checks(), 1, "{name}");
            assert!(
                b.spent() <= setup + bound,
                "{name}: {spent} spent before the stop, {total} in all",
                spent = b.spent()
            );
        }
    }
    assert!(
        cases >= 10,
        "{cases} reductions spend enough to stop inside"
    );
}

/// Test 6: below 46341 a reduction given a `Combine` makes the rows that
/// the reduction without one makes, on matrices of long rows too.
#[test]
fn a_small_prime_makes_what_rows_make() {
    for p in [7, 46337] {
        for seed in 0..20u64 {
            let m = long(p, seed);
            let name = format!("p {p}, seed {seed}");
            let exact = m.reduce(None);
            let given = m.reduce(Some(&mut Combine::new(seed)));
            assert_eq!(rows(&given), rows(&exact), "{name}");
        }
    }
}
