//! matrix-stop-latency, the crate's test code: the budget's units of work,
//! a round's matrix that stops inside its work, and the sorts in runs
//! (plan.md, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, WORK};
use crate::cofactors::NoCofactors;
use crate::f4::Halt;
use crate::field::{Field, Gf31};
use crate::id::{MonoId, RowId};
use crate::matrix::{
    self, Buffers, Combine, Matrix, RUN, Reduced, Reduction, by_signature, sort_decreasing,
    sort_runs,
};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::poly::{Basis, Poly};
use crate::sig::Sig;
use crate::table::Table;
use crate::trace::RowSource;
use crate::{Cancel, MonomialOrder, Params, Stop, Threads};

fn params() -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    )
}

/// A budget with no bound.
fn generous() -> Budget {
    Budget::new(Instant::now(), &params()).unwrap()
}

/// A budget whose deadline passed before the test began, and which has not
/// been checked.
fn passed() -> Budget {
    let mut p = params();
    p.deadline = Some(Duration::from_millis(1));
    Budget::new(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        &p,
    )
    .unwrap()
}

/// `passed` with `units` of work between two full checks.
fn passed_after(units: u64) -> Budget {
    let mut b = passed();
    b.set_work(units);
    b
}

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// Test 1, matrix-stop-latency.1 and .2: a full check comes once the units
/// spent since the last reach the units per check, or 64 ticks have passed;
/// one check resets both; a spend far past what is left checks once.
#[test]
fn the_budget_spends() {
    assert_eq!(WORK, 1 << 23);
    let cancel = Cancel::new();
    let mut p = params();
    p.cancel = Some(cancel.clone());
    let mut b = Budget::new(Instant::now(), &p).unwrap();
    b.set_work(100);

    for units in [1, 0, 98] {
        b.spend(units).unwrap();
    }
    assert_eq!(b.checks(), 0, "99 units make no check");
    b.spend(1).unwrap();
    assert_eq!(b.checks(), 1, "the 100th unit checks, and the flag is down");
    b.spend(99).unwrap();
    assert_eq!(b.checks(), 1, "99 more units make none");
    b.spend(1).unwrap();
    assert_eq!(b.checks(), 2);
    b.spend(1000).unwrap();
    assert_eq!(b.checks(), 3, "a spend far past what is left checks once");

    // Ticks and units share their check, and one check resets both.
    for _ in 0..63 {
        b.tick().unwrap();
    }
    b.spend(99).unwrap();
    assert_eq!(b.checks(), 3, "63 ticks and 99 units");
    b.tick().unwrap();
    assert_eq!(b.checks(), 4, "the 64th tick");
    b.spend(99).unwrap();
    for _ in 0..63 {
        b.tick().unwrap();
    }
    assert_eq!(b.checks(), 4, "the tick's check reset the units");

    // With the flag raised, the 100th unit stops the budget.
    cancel.set();
    assert_eq!(b.spend(1), Err(Stop::Cancel));
    assert_eq!(b.spent(), 1 + 98 + 1 + 99 + 1 + 1000 + 99 + 99 + 1);

    // A budget from `new` checks at `WORK` units, and not one before.
    let mut b = generous();
    b.spend(WORK as usize - 1).unwrap();
    assert_eq!(b.checks(), 0);
    b.spend(1).unwrap();
    assert_eq!(b.checks(), 1);
}

/// The degree of test 2's leading monomials.
const D: u32 = 30;

/// Test 2's matrix over GF(65521) on x and y. `G_min` holds `g_0` to `g_30`:
/// `g_a` is x^a·y^(30−a), plus x^b·y^(30−b) for every b < a, plus every
/// monomial of degree below 30, each with a coefficient drawn in [1, p).
/// The given rows are `x·g_a` and y·g_(a+1), the halves of the S-pairs of
/// neighbours, a = 0 to 29.
struct Dense {
    field: Gf31,
    table: Table<Dense16>,
    basis: Basis<Gf31, Rank1>,
    given: Vec<RowSource>,
}

impl Dense {
    fn new() -> Dense {
        let field = Gf31::new(65521).unwrap();
        let mut table = Table::new(Dense16::new(2), &mut rng(0x6d73));
        let mut r = rng(0x006d_736c);
        let mut lower = Vec::new();
        for d in 0..D {
            for a in 0..=d {
                lower.push(table.intern(&[a, d - a]).unwrap());
            }
        }
        let leads: Vec<MonoId> = (0..=D)
            .map(|a| table.intern(&[a, D - a]).unwrap())
            .collect();
        let mut basis = Basis::new();
        let mut elements = Vec::new();
        for a in 0..=D as usize {
            let mut terms = vec![(leads[a], 1u32)];
            for &u in leads[..a].iter().chain(&lower) {
                terms.push((u, field.coef(1 + r.next_u64() % (65521 - 1))));
            }
            let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
            assert_eq!(
                g.lead(),
                leads[a],
                "g_{a} leads with x^{a}·y^{b}",
                b = D - a as u32
            );
            elements.push(basis.push_min(&table, g).unwrap());
        }
        assert_eq!(basis.min().len(), D as usize + 1);
        let x = table.intern(&[1, 0]).unwrap();
        let y = table.intern(&[0, 1]).unwrap();
        let mut given = Vec::new();
        for a in 0..D as usize {
            given.push(RowSource {
                element: elements[a],
                multiplier: x,
            });
            given.push(RowSource {
                element: elements[a + 1],
                multiplier: y,
            });
        }
        Dense {
            field,
            table,
            basis,
            given,
        }
    }

    fn build(&mut self, budget: &mut Budget) -> Result<Matrix, Halt> {
        matrix::build(
            &mut self.table,
            &self.basis,
            &Grevlex,
            1,
            self.given.clone(),
            Vec::new(),
            budget,
        )
    }

    fn reduce(
        &self,
        m: &Matrix,
        full: bool,
        combine: Option<&mut Combine>,
        budget: &mut Budget,
    ) -> Result<Reduction<Gf31>, Halt> {
        matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
            &self.field,
            &self.basis,
            m,
            &mut NoCofactors,
            budget,
            full,
            combine,
            &mut Buffers::new(),
            None,
        )
    }

    /// The inter-reduction's matrix: `G_min`'s elements as given rows.
    fn inter_matrix(&mut self, budget: &mut Budget) -> Result<Matrix, Halt> {
        let one = self.table.intern(&[0, 0]).unwrap();
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
            2,
            rows,
            Vec::new(),
            budget,
        )
    }

    fn inter_reduce(&self, m: &Matrix, budget: &mut Budget) -> Result<Vec<Reduced<Gf31>>, Stop> {
        matrix::inter_reduce::<Gf31, Dense16, Rank1, NoCofactors>(
            &self.field,
            &self.basis,
            m,
            &mut NoCofactors,
            budget,
            &mut Buffers::new(),
        )
    }
}

/// The made rows, each as its columns and coefficients, by leading column.
fn rows(made: &[Reduced<Gf31>]) -> Vec<(Vec<MonoId>, Vec<u32>)> {
    let mut made: Vec<&Reduced<Gf31>> = made.iter().collect();
    made.sort_by_key(|r| r.cols[0]);
    made.iter()
        .map(|r| (r.cols.clone(), r.coefs.clone()))
        .collect()
}

/// The longest row of a matrix, in entries.
fn longest(m: &Matrix) -> u64 {
    (0..m.rows.len())
        .map(|r| m.row(RowId::at(r)).len())
        .max()
        .unwrap() as u64
}

/// Test 2, matrix-stop-latency.1, .3, .5, and .8: with a deadline passed
/// and unchecked, and 4,096 units a check, the build stops at its first
/// row's products, and each reduction inside the first row it reduces;
/// the back-substitution and the inter-reduction stop inside their work.
/// Generous, each returns its result.
#[test]
fn a_round_matrix_stops_inside_its_work() {
    const T: u64 = 4096;
    let mut dense = Dense::new();

    // The build: fewer than 64 reducer rows, so no tick checks.
    let mut b = generous();
    let m = dense.build(&mut b).unwrap();
    let reducers = m.rows.len() - m.given;
    assert_eq!(m.given, 2 * D as usize);
    assert!(
        reducers < 64,
        "{reducers} reducer rows tick fewer than 64 times"
    );
    let products = dense.basis.element(dense.given[0].element).monomials.len() as u64;
    let mut b = passed_after(T);
    assert_eq!(dense.build(&mut b).err(), Some(Halt::Stop(Stop::Deadline)));
    assert_eq!(b.checks(), 1);
    assert!(
        b.spent() <= T + products * matrix::PRODUCT as u64,
        "the build spent {spent} before it stopped",
        spent = b.spent()
    );

    // The reductions, with combinations and without.
    let mut b = generous();
    let exact = dense.reduce(&m, true, None, &mut b).unwrap();
    let exact_spent = b.spent();
    let mut b = generous();
    let combined = dense
        .reduce(&m, true, Some(&mut Combine::new(1)), &mut b)
        .unwrap();
    let combined_spent = b.spent();
    assert_eq!(rows(&combined.made), rows(&exact.made));
    assert!(!exact.made.is_empty());
    let bound = T + longest(&m);
    for (name, total, combine) in [
        ("row by row", exact_spent, None),
        ("combinations", combined_spent, Some(Combine::new(1))),
    ] {
        let mut combine = combine;
        let mut b = passed_after(T);
        assert_eq!(
            dense.reduce(&m, true, combine.as_mut(), &mut b).err(),
            Some(Halt::Stop(Stop::Deadline)),
            "{name}"
        );
        assert_eq!(b.checks(), 1, "{name}");
        assert!(
            b.spent() <= bound && 4 * bound < total,
            "{name}: {spent} spent before the stop, {total} in all",
            spent = b.spent()
        );
    }

    // The back-substitution: the forward reduction spends F, and with F + 64
    // units a check the stop comes after it.
    let mut b = generous();
    let forward = dense.reduce(&m, false, None, &mut b).unwrap();
    let f = b.spent();
    assert!(exact_spent > f + 64, "the back-substitution spends");
    assert_eq!(rows(&forward.made).len(), exact.made.len());
    let mut b = passed_after(f + 64);
    assert_eq!(
        dense.reduce(&m, true, None, &mut b).err(),
        Some(Halt::Stop(Stop::Deadline))
    );
    assert!(b.spent() > f, "the stop came in the back-substitution");
    assert!(b.spent() <= f + 64 + longest(&m));

    // The inter-reduction.
    let mut b = generous();
    let inter = dense.inter_matrix(&mut b).unwrap();
    let mut b = generous();
    let reduced = dense.inter_reduce(&inter, &mut b).unwrap();
    assert_eq!(reduced.len(), D as usize + 1);
    let total = b.spent();
    let mut b = passed_after(T);
    assert_eq!(
        dense.inter_reduce(&inter, &mut b).err(),
        Some(Stop::Deadline)
    );
    assert!(b.spent() <= T + longest(&inter) && 4 * T < total);
}

/// `n` distinct values in a random order.
fn shuffled(n: usize, seed: u64) -> Vec<u32> {
    let mut r = rng(seed);
    let mut v: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
    for i in (1..n).rev() {
        let j = (r.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// `n` distinct monomials of `Dense16` on six variables, of degree 24 or
/// less, in the random order drawn, in a table of their own.
fn monomials(n: usize, seed: u64) -> (Table<Dense16>, Vec<MonoId>) {
    let mut t = Table::new(Dense16::new(6), &mut rng(seed));
    let mut r = rng(seed ^ 0x6d6f);
    let mut taken = HashSet::new();
    let mut ids = Vec::with_capacity(n);
    while ids.len() < n {
        let mut e = [0u32; 6];
        let mut d = (r.next_u64() % 25) as u32;
        for x in &mut e {
            let k = (r.next_u64() % (u64::from(d) + 1)) as u32;
            *x = k;
            d -= k;
        }
        let id = t.intern(&e).unwrap();
        if taken.insert(id) {
            ids.push(id);
        }
    }
    (t, ids)
}

/// `sort_decreasing` returns the comparison sort's order.
fn sorts_as_compared<O: Order<Dense16>>(order: &O, t: &Table<Dense16>, ids: &[MonoId]) {
    let mut got = ids.to_vec();
    let mut b = generous();
    sort_decreasing(t, order, &mut got, &mut b).unwrap();
    let mut want = ids.to_vec();
    want.sort_by(|&a, &b| t.cmp(order, b, a));
    assert_eq!(got, want, "{order:?}, {len} ids", len = ids.len());
    if !O::KEYED && ids.len() > RUN {
        assert!(
            b.checks() >= ids.len().div_ceil(RUN) as u64,
            "a check a run"
        );
    }
}

/// Test 3, matrix-stop-latency.6 and .7: the sorts in runs return the
/// comparison sort's order, check at least once a run, and stop at their
/// first run where the deadline has passed; the keyed sorts return their
/// order as before.
#[test]
fn the_sorts_in_runs() {
    for (k, n) in [0, 1, RUN - 1, RUN, RUN + 1, 3 * RUN + 7]
        .into_iter()
        .enumerate()
    {
        let v = shuffled(n, k as u64);
        let mut got = v.clone();
        let mut b = generous();
        sort_runs(&mut got, |a: &u32, b: &u32| b.cmp(a), &mut b).unwrap();
        let mut want = v;
        want.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(got, want, "{n} values");
        if n > RUN {
            assert!(b.checks() >= n.div_ceil(RUN) as u64, "{n} values");
        }
    }
    let mut v = shuffled(3 * RUN + 7, 9);
    let mut b = passed();
    assert_eq!(
        sort_runs(&mut v, |a: &u32, b: &u32| a.cmp(b), &mut b),
        Err(Stop::Deadline)
    );
    assert_eq!(b.checks(), 1, "the first run's check");

    let (t, ids) = monomials(3 * RUN + 7, 0x736f);
    for len in [0, 1, RUN, ids.len()] {
        sorts_as_compared(&Grevlex, &t, &ids[..len]);
        sorts_as_compared(&Lex, &t, &ids[..len]);
        sorts_as_compared(&Block::new(&[2], 6).unwrap(), &t, &ids[..len]);
    }
    let mut got = ids.clone();
    assert_eq!(
        sort_decreasing(&t, &Lex, &mut got, &mut passed()),
        Err(Stop::Deadline)
    );
    let mut got = ids.clone();
    assert_eq!(
        sort_decreasing(&t, &Grevlex, &mut got, &mut passed_after(1)),
        Err(Stop::Deadline)
    );

    // Signatures over a few hundred monomials and three indices: repeated
    // monomials, repeated indices, and equal signatures.
    let mut r = rng(0x7369);
    let sigs: Vec<Sig> = (0..3 * RUN + 7)
        .map(|_| Sig {
            monomial: ids[(r.next_u64() % 300) as usize],
            index: (r.next_u64() % 3) as u32,
        })
        .collect();
    let got = by_signature(&t, &Grevlex, &sigs, &mut generous()).unwrap();
    let mut want: Vec<RowId> = (0..sigs.len()).map(RowId::at).collect();
    want.sort_by(|&a, &b| {
        sigs[a.index()]
            .cmp(sigs[b.index()], &t, &Grevlex)
            .then(a.cmp(&b))
    });
    assert_eq!(got, want);
    assert_eq!(
        by_signature(&t, &Grevlex, &sigs, &mut passed_after(1)).err(),
        Some(Stop::Deadline)
    );
}
