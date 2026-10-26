//! row-scan-time, the crate's test code: the marked scan against the dense
//! scan on random matrices, in signature rounds over the four fields,
//! along a chain across the words of the marks, under the chosen scan as
//! the rows turn from few entries to many, after a stopped reduction, and
//! from past a row's lead (`iterations/0072-row-scan-time/plan.md`,
//! _Test code_, 1 to 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::band_tests::{PRIMES, Random, budget, passed_after};
use crate::cofactors::NoCofactors;
use crate::f4::Halt;
use crate::field::{Field, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{self, Buffers, Combine, Reduction, Scan};
use crate::module::Rank1;
use crate::modulus_tests::{L1, L2, W1, W2};
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::signature_zero_tests::{Made, Round, failing, made};
use crate::table::Table;
use crate::trace::RowSource;

/// The three scans; the first, the dense scan of every column, is what
/// the others are compared with.
const SCANS: [Scan; 3] = [Scan::Dense, Scan::Marked, Scan::Chosen];

/// Fresh buffers whose reducers take `scan`.
fn buffers<F: Field>(scan: Scan) -> Buffers<F> {
    let mut bufs = Buffers::new();
    bufs.scan = scan;
    bufs
}

/// What a reduction returns, to compare: `made`'s rows, zero rows,
/// pivots, and order, the reductions, and the zeros; or its halt.
type Outcome<C> = Result<(Made<C>, u64, u64), Halt>;

fn outcome<F: Field>(reduction: Result<Reduction<F>, Halt>) -> Outcome<F::Coef> {
    reduction.map(|r| (made(&r), r.reductions, r.zeros))
}

/// `matrix::reduce` of `m` in each scan, with combinations under the seed
/// 0x5ca7 where `combine` is true and the prime takes them, the back-substitution where `full` is: each
/// outcome is the dense scan's.
fn assert_plain_scans_agree(m: &Random, combine: bool, full: bool, case: &str) {
    let mut outcomes = SCANS.map(|scan| {
        let mut combination = Combine::new(0x5ca7);
        outcome(matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
            &m.field,
            &m.basis,
            &m.matrix,
            &mut NoCofactors,
            &mut budget(),
            full,
            combine.then_some(&mut combination),
            &mut buffers(scan),
            None,
        ))
    });
    let dense = outcomes[0].clone();
    for (scan, got) in SCANS.iter().zip(&mut outcomes).skip(1) {
        assert_eq!(*got, dense, "{case}: {scan:?} against the dense scan");
    }
}

/// Test 1: row-scan-time.1, .2, .5, .6, .10. `matrix::reduce` of random
/// matrices over `Gf31` at every prime of the band tests, of elements of
/// two terms and of up to eight, degrees 2 to 8 on four variables, with
/// combinations where the prime takes them, from 46341 on, and without,
/// with the back-substitution and without, in
/// the three scans: one outcome.
#[test]
fn the_marked_scan_reduces_as_the_dense_scan() {
    for p in PRIMES {
        for seed in 0..24u64 {
            let shape = (4 + (seed % 9) as usize, 6 + (seed % 23) as usize, 3);
            let most = [1, 3, 7][(seed % 3) as usize];
            let degree = 2 + seed % 7;
            let m = Random::new(p, seed, shape, (most, degree));
            for combine in [false, true] {
                for full in [false, true] {
                    let case = format!(
                        "p {p}, seed {seed}, combinations {combine}, back-substitution {full}"
                    );
                    assert_plain_scans_agree(&m, combine, full, &case);
                }
            }
        }
    }
}

/// The signature round of `field` under `seed`, `long` elements or short,
/// as `signature_zero_tests::compare` draws it.
fn round<F: Field>(field: F, seed: u64, long: bool) -> Round<F> {
    let tail = [0, 6, 20, 80][(seed % 4) as usize];
    Round::random(
        field,
        seed,
        4 + (seed % 10) as usize,
        6 + 3 * (seed % 9) as usize,
        10 + 7 * (seed % 11) as usize,
        tail,
        long,
    )
}

/// The rounds of test 2 over `field`: `reduce_sig` with tests where
/// `tests` is true and without, and `matrix::reduce` of the round's
/// matrix, in the three scans: one outcome each.
fn assert_round_scans_agree<F: Field>(field: F, seeds: u64, tests: bool) {
    let p = field.characteristic();
    for seed in 0..seeds {
        for long in [false, true] {
            let round = round(field, seed, long);
            let case = format!("p {p}, seed {seed}, long {long}");
            for test in [false, tests] {
                let outcomes = SCANS.map(|scan| {
                    let mut combination = Combine::new(seed ^ 0x2e70);
                    made(&round.reduce_in(test.then_some(&mut combination), &mut buffers(scan)))
                });
                assert_eq!(outcomes[1], outcomes[0], "{case}, tests {test}: marked");
                assert_eq!(outcomes[2], outcomes[0], "{case}, tests {test}: chosen");
            }
            let outcomes = SCANS.map(|scan| {
                outcome(matrix::reduce::<F, Dense16, Rank1, NoCofactors>(
                    &round.field,
                    &round.basis,
                    &round.matrix,
                    &mut NoCofactors,
                    &mut budget(),
                    true,
                    None,
                    &mut buffers(scan),
                    None,
                ))
            });
            assert_eq!(outcomes[1], outcomes[0], "{case}, matrix::reduce: marked");
            assert_eq!(outcomes[2], outcomes[0], "{case}, matrix::reduce: chosen");
        }
    }
}

/// Test 2: row-scan-time.1, .9. Random signature rounds over `Gf31` at
/// 65521, `Gf64` at 2^62 − 57, `Gf31Pair`, and `Gf64Pair`, short and long
/// elements, tails of 0 to 80 copies: `reduce_sig` with the signature
/// run's tests where the field has one lane and without, and
/// `matrix::reduce`, in the three scans: one outcome. A failed test in
/// the marked scan: the dense scan's outcome.
#[test]
fn the_scans_agree_in_every_field() {
    assert_round_scans_agree(Gf31::new(65521).unwrap(), 24, true);
    assert_round_scans_agree(Gf64::new(4_611_686_018_427_387_847).unwrap(), 24, true);
    // The first lane's prime is the smaller, so that a coefficient drawn
    // below it is a unit at both lanes.
    assert_round_scans_agree(Gf31Pair::primes(L2, L1), 24, false);
    assert_round_scans_agree(Gf64Pair::primes(W1, W2), 24, false);
    for (copies, last) in [(8, true), (10, false)] {
        let round = failing(Gf31::new(65521).unwrap(), copies, last);
        let outcomes = SCANS
            .map(|scan| made(&round.reduce_in(Some(&mut Combine::new(5)), &mut buffers(scan))));
        assert_eq!(outcomes[1], outcomes[0], "{copies} copies, last {last}");
        assert_eq!(outcomes[2], outcomes[0], "{copies} copies, last {last}");
    }
    let round = failing(Gf64::new((1 << 61) - 1).unwrap(), 10, false);
    let outcomes =
        SCANS.map(|scan| made(&round.reduce_in(Some(&mut Combine::new(7)), &mut buffers(scan))));
    assert_eq!(outcomes[1], outcomes[0], "Gf64");
    assert_eq!(outcomes[2], outcomes[0], "Gf64");
}

/// A matrix on one variable x over GF(65521): the rows x^(W−1)·g and h,
/// g = x − c, both leading at x^W, and the reducers x^k·g that symbolic
/// preprocessing adds down to x^0, W + 1 columns.
fn chain(w: u32, c: u64, h: &[(u32, u64)]) -> (Gf31, Basis<Gf31, Rank1>, matrix::Matrix) {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(1), &mut Xoshiro256PlusPlus::seed_from_u64(72));
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    let mut poly = |terms: &[(u32, u64)]| {
        let mut terms: Vec<(MonoId, u32)> = terms
            .iter()
            .map(|&(e, v)| (table.intern(&[e]).unwrap(), field.coef(v)))
            .collect();
        Poly::from_terms(&mut terms, &field, &table, &Grevlex, ())
    };
    let g = poly(&[(1, 1), (0, 65521 - c)]);
    let h = poly(h);
    basis.push(g, false).unwrap();
    basis.push(h, true).unwrap();
    let rows = vec![
        RowSource {
            element: ElementId::at(0),
            multiplier: table.intern(&[w - 1]).unwrap(),
        },
        RowSource {
            element: ElementId::at(1),
            multiplier: table.intern(&[0]).unwrap(),
        },
    ];
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
    assert_eq!(matrix.columns(), w as usize + 1, "W = {w}: x^W down to x^0");
    (field, basis, matrix)
}

/// The outcome of `matrix::reduce` of a chain's matrix in `scan`.
fn reduce_chain(
    (field, basis, matrix): &(Gf31, Basis<Gf31, Rank1>, matrix::Matrix),
    scan: Scan,
) -> Outcome<u32> {
    outcome(matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
        field,
        basis,
        matrix,
        &mut NoCofactors,
        &mut budget(),
        false,
        None,
        &mut buffers(scan),
        None,
    ))
}

/// c^e mod 65521.
fn power(c: u64, e: u32) -> u64 {
    (0..e).fold(1, |x, _| x * c % 65521)
}

/// Test 3: row-scan-time.1, .6. The chain of reductions from x^W down to
/// x^0, W + 1 columns: 2, 63 to 65, 127 to 129, and 1001. h = x^W − d
/// reduces to zero exactly where d = c^W; h = x^W − c^(W−k)·x^k + b,
/// k = ⌊W/2⌋, cancels at x^k and reduces to b at x^0. The marked and the
/// dense scan make one reduction.
#[test]
fn a_chain_across_the_words() {
    let c = 3;
    for w in [1, 62, 63, 64, 126, 127, 128, 1000] {
        for d in [power(c, w), power(c, w) + 1] {
            let m = chain(w, c, &[(w, 1), (0, 65521 - d % 65521)]);
            let dense = reduce_chain(&m, Scan::Dense);
            assert_eq!(reduce_chain(&m, Scan::Marked), dense, "W = {w}, d = {d}");
            assert_eq!(reduce_chain(&m, Scan::Chosen), dense, "W = {w}, d = {d}");
            let (rows, zero, _, _) = &dense.as_ref().unwrap().0;
            assert_eq!(
                zero.is_empty(),
                d % 65521 != power(c, w),
                "W = {w}, d = {d}: zero where d = c^W"
            );
            assert_eq!(rows.len(), usize::from(zero.is_empty()), "W = {w}, d = {d}");
        }
        if w >= 2 {
            let (k, b) = (w / 2, 5);
            let a = 65521 - power(c, w - k);
            let m = chain(w, c, &[(w, 1), (k, a), (0, b)]);
            let dense = reduce_chain(&m, Scan::Dense);
            assert_eq!(reduce_chain(&m, Scan::Marked), dense, "W = {w}, cancelled");
            assert_eq!(reduce_chain(&m, Scan::Chosen), dense, "W = {w}, cancelled");
            let (rows, _, _, _) = &dense.as_ref().unwrap().0;
            assert_eq!(rows.len(), 1, "W = {w}: the row is not zero");
            assert_eq!(
                rows[0].1,
                vec![MonoId::at(w as usize)],
                "W = {w}: one entry, at x^0"
            );
        }
    }
}

/// Test 4: row-scan-time.2, .3. Matrices of elements of 2 to 81 terms,
/// the given rows reduced one at a time in an order the test chooses: 100
/// rows of elements of four terms or fewer, then every row of a longer
/// element, then the rest. Under the chosen scan each row's result is the
/// dense scan's.
#[test]
fn the_chosen_scan_turns() {
    let mut turned = 0;
    for seed in 0..24u64 {
        let m = Random::new(65521, seed, (24, 400, 0), (80, 6));
        let terms = |r: RowId| m.basis.element(m.matrix.rows[r.index()].element).len();
        let mut chosen = buffers::<Gf31>(Scan::Chosen);
        let mut dense = buffers::<Gf31>(Scan::Dense);
        let (mut a, rest) = m.reducer_in(&mut chosen);
        let (mut b, _) = m.reducer_in(&mut dense);
        let short: Vec<RowId> = rest.iter().copied().filter(|&r| terms(r) <= 4).collect();
        let long: Vec<RowId> = rest.iter().copied().filter(|&r| terms(r) > 4).collect();
        if short.len() < 100 || long.is_empty() {
            continue;
        }
        turned += 1;
        let order = short[..100].iter().chain(&long).chain(&short[100..]);
        let (mut ba, mut bb) = (budget(), budget());
        for &r in order {
            let lead = m.matrix.row(r)[0];
            let x = a.reduce::<Dense16, NoCofactors>(r, lead, &mut NoCofactors, &mut ba);
            let y = b.reduce::<Dense16, NoCofactors>(r, lead, &mut NoCofactors, &mut bb);
            let (x, y) = (x.unwrap(), y.unwrap());
            assert_eq!(
                (&x.cols, &x.coefs),
                (&y.cols, &y.coefs),
                "seed {seed}, row {r}"
            );
            if !x.cols.is_empty() {
                assert!(a.keep::<Dense16, NoCofactors>(x, &mut NoCofactors));
                assert!(b.keep::<Dense16, NoCofactors>(y, &mut NoCofactors));
            }
        }
    }
    assert!(turned >= 8, "{turned} matrices hold both kinds of rows");
}

/// Test 5: row-scan-time.8. A reduction in the marked scan that a bound
/// stops 1 to 400 units of work past its setup's spend, the matrix's rows
/// and columns; then a reduction of the same matrix from the same buffers
/// under an unbounded budget: the dense scan's outcome from fresh
/// buffers.
#[test]
fn a_stopped_marked_reduction_leaves_nothing_behind() {
    let mut stopped = 0;
    for seed in 0..12u64 {
        let m = Random::new(65521, seed, (8, 20, 3), (1, 6));
        let setup = (m.matrix.rows.len() + m.matrix.columns()) as u64;
        let mut bufs = buffers::<Gf31>(Scan::Marked);
        for extra in [1, 3, 10, 40, 100, 400] {
            let units = setup + extra;
            let halted = matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
                &m.field,
                &m.basis,
                &m.matrix,
                &mut NoCofactors,
                &mut passed_after(units),
                true,
                None,
                &mut bufs,
                None,
            );
            if halted.is_err() {
                stopped += 1;
            }
            let after = outcome(matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
                &m.field,
                &m.basis,
                &m.matrix,
                &mut NoCofactors,
                &mut budget(),
                true,
                None,
                &mut bufs,
                None,
            ));
            let dense = outcome(matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
                &m.field,
                &m.basis,
                &m.matrix,
                &mut NoCofactors,
                &mut budget(),
                true,
                None,
                &mut buffers(Scan::Dense),
                None,
            ));
            assert_eq!(after, dense, "seed {seed}, {units} units");
        }
    }
    assert!(stopped > 0, "no reduction stopped");
}

/// Test 6: row-scan-time.1. Every given row of random matrices reduced
/// from one past its leading column, as `inter_reduce` reduces a row, its
/// lead taken outside the scan: the marked and the dense scan make one
/// row.
#[test]
fn the_lead_before_the_start() {
    for seed in 0..24u64 {
        let shape = (4 + (seed % 9) as usize, 6 + (seed % 23) as usize, 3);
        let m = Random::new(65521, seed, shape, ([1, 3, 7][(seed % 3) as usize], 6));
        let mut marked = buffers::<Gf31>(Scan::Marked);
        let mut dense = buffers::<Gf31>(Scan::Dense);
        let (mut a, _) = m.reducer_in(&mut marked);
        let (mut b, _) = m.reducer_in(&mut dense);
        let (mut ba, mut bb) = (budget(), budget());
        for r in (0..m.matrix.given).map(RowId::at) {
            let after = MonoId::at(m.matrix.row(r)[0].index() + 1);
            let x = a
                .reduce::<Dense16, NoCofactors>(r, after, &mut NoCofactors, &mut ba)
                .unwrap();
            let y = b
                .reduce::<Dense16, NoCofactors>(r, after, &mut NoCofactors, &mut bb)
                .unwrap();
            assert_eq!(
                (&x.cols, &x.coefs),
                (&y.cols, &y.coefs),
                "seed {seed}, row {r}"
            );
            assert_eq!(
                x.cols[0],
                m.matrix.row(r)[0],
                "seed {seed}, row {r}: the lead kept"
            );
        }
    }
}
