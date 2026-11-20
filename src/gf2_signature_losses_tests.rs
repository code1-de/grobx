//! gf2-signature-losses, sub-item 1's test code: the budget's row work,
//! its bound, and a plain round over `Gf2` that halts past the bound
//! (`iterations/0129-gf2-signature-losses/plan.md`, _Test code_, 4 to 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, Site};
use crate::cofactors::NoCofactors;
use crate::cyclic_gap_tests::{Random, long, short};
use crate::field::{BAND, Gf2, Gf31};
use crate::halt::Halt;
use crate::id::{ElementId, MonoId, RowSource};
use crate::matrix::{self, Buffers, Combine};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::packed_reduction_tests::{K, Plain, forced, outcome};
use crate::poly::{Basis, Poly};
use crate::shared_tests::{budget, rows};
use crate::table::Table;

/// The random plain rounds of tests 4 and 6: per K of `K`, seeds 0 to 3,
/// and sparse and dense tails, as `packed_reduction_tests`' test 7 draws
/// them, with a name and K each.
fn rounds() -> Vec<(String, usize, Plain)> {
    let mut out = Vec::new();
    for (i, &k) in K.iter().enumerate() {
        for seed in 0..4u64 {
            for density in [1, 4] {
                let name = format!("k {k}, seed {seed}, density {density}");
                let m = Plain::new(
                    0xa290 + 16 * i as u64 + seed,
                    40 + 7 * seed as usize,
                    k,
                    24,
                    density,
                );
                out.push((name, k, m));
            }
        }
    }
    out
}

/// Test 6's plain rounds of one row to reduce, per K of `K`, and a round
/// of no row to reduce and no K column, with a name and K each.
fn few_rounds() -> Vec<(String, usize, Plain)> {
    let mut out: Vec<(String, usize, Plain)> = K
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            let m = Plain::new(0xa2b0 + i as u64, 40, k, 1, 4);
            (format!("k {k}, one row to reduce"), k, m)
        })
        .collect();
    let none = Plain::new(0xa2c0, 40, 0, 0, 4);
    out.push(("no row to reduce".to_owned(), 0, none));
    out
}

/// Test 4: on random plain rounds over `Gf2` reduced in sparse rows, the
/// row work a reduction adds is the entries it subtracted; on a
/// hand-built round reduced in packed rows, whose one packed row added
/// holds two entries in one word, the row work is that word and the
/// subtracted entries are its two entries.
#[test]
fn the_row_work() {
    for (name, _, m) in rounds() {
        let mut b = budget();
        let (work, subtracted) = (b.row_work(), b.counts().0);
        forced(false, || m.reduce(true, &mut b)).unwrap();
        assert_eq!(
            b.row_work() - work,
            b.counts().0 - subtracted,
            "{name}: the sparse reduction's row work"
        );
    }
    // Columns c0 > c1 > c2 > c3. The reducer row c0 + c2 + c3 leads c0, an
    // early pivot whose reduced row is c2 + c3 over the K columns c1, c2,
    // and c3, one word. The given row c0 + c1 + c2 + c3, longer than the
    // reducer row, stays the row to reduce: it adds c0's reduced row, two
    // entries in one word, and ends c1, a made row.
    let field = Gf2::new();
    let mut table = Table::new(
        Dense16::new(4),
        &mut Xoshiro256PlusPlus::seed_from_u64(0xa2a0),
    );
    let mut c: Vec<MonoId> = [[3, 0, 0, 0], [2, 1, 0, 0], [2, 0, 1, 0], [1, 1, 1, 0]]
        .iter()
        .map(|e| table.intern(e, &mut Budget::unbounded()).unwrap())
        .collect();
    c.sort_by(|&u, &v| table.cmp(&Grevlex, v, u));
    let one = table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
    let mut basis: Basis<Gf2, Rank1> = Basis::new();
    let row = |cols: &[usize], basis: &mut Basis<Gf2, Rank1>| {
        let mut terms: Vec<(MonoId, u8)> = cols.iter().map(|&i| (c[i], 1)).collect();
        basis
            .push(
                Poly::from_terms(&mut terms, &field, &table, &Grevlex, ()),
                false,
            )
            .unwrap();
        RowSource {
            element: ElementId::at(basis.len() - 1),
            multiplier: one,
        }
    };
    let rows = vec![row(&[0, 1, 2, 3], &mut basis), row(&[0, 2, 3], &mut basis)];
    let mut matrix = matrix::build_rows(
        &mut table,
        &basis,
        &Grevlex,
        rows,
        &mut Buffers::new(),
        &mut budget(),
    )
    .unwrap();
    matrix.given = 1;
    let mut b = budget();
    let (work, subtracted) = (b.row_work(), b.counts().0);
    let reduction = forced(true, || {
        matrix::reduce::<Gf2, Dense16, Rank1, NoCofactors>(
            &field,
            &basis,
            &matrix,
            &mut NoCofactors,
            &mut b,
            false,
            None,
            &mut Buffers::new(),
            None,
            None,
        )
    })
    .unwrap();
    assert_eq!(reduction.packed, Some(1), "three K columns, a word");
    let made: Vec<Vec<MonoId>> = reduction
        .made
        .iter()
        .map(|r| r.cols.iter().map(|x| matrix.monomials[x.index()]).collect())
        .collect();
    assert_eq!(made, vec![vec![c[1]]], "the made row");
    assert_eq!(b.row_work() - work, 1, "the packed row's one word");
    assert_eq!(b.counts().0 - subtracted, 2, "the packed row's two entries");
}

/// Test 5: a budget without a bound is never outworked; bounded by b at
/// row work w, it is not outworked at w + b and is outworked at
/// w + b + 1; unbound, it is not outworked; a bound of `u64::MAX`
/// saturates and is never passed.
#[test]
fn the_work_bound() {
    let mut b = budget();
    assert!(!b.outworked(), "no bound");
    b.spend_subtracted(1_000, Site::Row).unwrap();
    assert!(!b.outworked(), "no bound after a row operation");
    for bound in [0u64, 1, 7, 1 << 20] {
        let w = b.row_work();
        b.bound(bound);
        let mut done = 0;
        while done < bound {
            let step = (bound - done).min(1 << 16);
            b.spend_subtracted(step as usize, Site::Row).unwrap();
            done += step;
        }
        assert_eq!(b.row_work(), w + bound, "bound {bound}");
        assert!(!b.outworked(), "bound {bound}: at the bound");
        b.spend_subtracted(1, Site::Row).unwrap();
        assert!(b.outworked(), "bound {bound}: one past the bound");
        b.unbound();
        assert!(!b.outworked(), "bound {bound}: unbound");
    }
    b.bound(u64::MAX);
    b.spend_subtracted(1 << 20, Site::Row).unwrap();
    assert!(!b.outworked(), "a bound of u64::MAX");
    b.unbound();
}

/// Test 6: `matrix::reduce` over `Gf2` on random plain rounds, sparse and
/// packed, and on rounds of one row to reduce or none. Bounded by
/// `u64::MAX`, and by the row work the unbounded reduction spent: the
/// unbounded reduction's outcome. Bounded by 0: in sparse rows,
/// `Halt::Outworked` where the unbounded reduction reduced two rows or
/// more, as every row to reduce there cancels its leading entry, and the
/// unbounded outcome on a round of one row to reduce and on one of none,
/// each present; in packed rows, `Halt::Outworked` or the unbounded
/// outcome, and `Halt::Outworked` on some round. A packed round of no K
/// column, whose packed rows hold no word, spends no row work and halts
/// at no bound.
#[test]
fn a_bounded_plain_round() {
    let mut halted_packed = 0;
    // The sparse rounds of no reduction and of one that ran past a bound
    // of 0.
    let mut few = [false; 2];
    for (name, k, m) in rounds().into_iter().chain(few_rounds()) {
        for packs in [false, true] {
            let case = format!("{name}, packed {packs}");
            let mut free = budget();
            let unbounded = forced(packs, || m.reduce(true, &mut free)).unwrap();
            let spent = free.row_work();
            if packs && k == 0 {
                assert_eq!(spent, 0, "{case}: packed rows of no word");
            }
            for bound in [u64::MAX, spent] {
                let mut b = budget();
                b.bound(bound);
                let bounded = forced(packs, || m.reduce(true, &mut b)).unwrap();
                assert_eq!(
                    outcome(&bounded),
                    outcome(&unbounded),
                    "{case}, bound {bound}"
                );
            }
            let mut b = budget();
            b.bound(0);
            let at_zero = forced(packs, || m.reduce(true, &mut b));
            match at_zero {
                Err(Halt::Outworked) => halted_packed += usize::from(packs),
                Ok(reduction) => {
                    assert!(
                        packs || unbounded.reductions < 2,
                        "{case}: a sparse round of {r} reductions ran past a bound of 0",
                        r = unbounded.reductions
                    );
                    if !packs {
                        few[unbounded.reductions as usize] = true;
                    }
                    assert_eq!(outcome(&reduction), outcome(&unbounded), "{case}, bound 0");
                }
                Err(other) => panic!("{case}: bound 0 halted with {other:?}"),
            }
        }
    }
    assert!(halted_packed > 0, "no packed round halted at a bound of 0");
    assert_eq!(few, [true; 2], "a sparse round of no reduction, and of one");
}

/// The prime of `a_bounded_round_of_combinations`, past the combinations'
/// bound of 46341 (`matrix::combines_rows`).
const COMBINES: u64 = 65521;

/// Test 6 on the rounds of combinations, which no round over `Gf2` takes:
/// `matrix::reduce` over GF(65521) with a `Combine`, on random matrices
/// of short rows, which combine one combination at a time, and of long
/// rows, which take bands. Bounded by `u64::MAX`, and by the row work the
/// unbounded reduction spent: the unbounded reduction's made rows,
/// reductions, and zeros. Bounded by 0: `Halt::Outworked` where the
/// unbounded reduction made more than one reduction in short rows, or
/// more than `BAND`, a band's, in long rows, and the unbounded outcome
/// otherwise; `Halt::Outworked` on some matrix of each kind.
#[test]
fn a_bounded_round_of_combinations() {
    let mut halted = [false; 2];
    for seed in 0..12u64 {
        for is_long in [false, true] {
            let i = seed as usize;
            let shape = if is_long {
                long(4 + i % 5, 6 + i * 7 % 24, 3, i % 4)
            } else {
                short(8 + i % 7, 12 + i * 11 % 48, 3, i % 4)
            };
            let m = Random::new(COMBINES, 0xa2d0 + seed, &shape);
            let case = format!("seed {seed}, long {is_long}");
            assert_eq!(
                matrix::takes_bands::<Gf31, Dense16, NoCofactors>(&m.matrix),
                is_long,
                "{case}: bands where the rows are long"
            );
            let reduce = |b: &mut Budget| m.reduce_in(Some(&mut Combine::new(0xa2e0 ^ seed)), b);
            let mut free = budget();
            let unbounded = reduce(&mut free).unwrap();
            let want = (rows(&unbounded), unbounded.reductions, unbounded.zeros);
            for bound in [u64::MAX, free.row_work()] {
                let mut b = budget();
                b.bound(bound);
                let got = reduce(&mut b).unwrap();
                let got = (rows(&got), got.reductions, got.zeros);
                assert_eq!(got, want, "{case}, bound {bound}");
            }
            let most = if is_long { BAND as u64 } else { 1 };
            let mut b = budget();
            b.bound(0);
            match reduce(&mut b) {
                Err(Halt::Outworked) => halted[usize::from(is_long)] = true,
                Ok(got) => {
                    assert!(
                        unbounded.reductions <= most,
                        "{case}: {r} reductions ran past a bound of 0",
                        r = unbounded.reductions
                    );
                    let got = (rows(&got), got.reductions, got.zeros);
                    assert_eq!(got, want, "{case}, bound 0");
                }
                Err(other) => panic!("{case}: bound 0 halted with {other:?}"),
            }
        }
    }
    assert_eq!(
        halted, [true; 2],
        "a round of short rows halted, and one of long rows"
    );
}
