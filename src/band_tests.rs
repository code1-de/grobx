//! row-traffic-time, the crate's test code: the band's row operations of
//! `Gf31` against the one-row operations, the band reduction against the
//! reduction of one row at a time, the bands' ends, the rounds that take
//! bands, the band from the buffers, and the band's spends
//! (`iterations/0044-row-traffic-time/plan.md`, _Test code_, 1 to 4, and
//! `iterations/0060-row-traffic-time/plan.md`, _Test code_, 2, 9, and 10).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::{NoCofactors, Tracked};
use crate::field::{BAND, Field, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::id::{KeepId, MonoId, RowId};
use crate::matrix::{self, BAND_ENTRIES, Buffers, Matrix, Pivot, Reduced, Reducer};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::systems::Rng;
use crate::table::Table;
use crate::trace::RowSource;
use crate::{MonomialOrder, Params, Stop, Threads};

/// The primes of test 1: the smallest, below 2^16 on both sides of the
/// combinations' bound, from 2^16, and the largest below 2^31.
pub(crate) const PRIMES: [u64; 7] = [2, 3, 7, 32003, 65521, 65537, 2_147_483_647];

pub(crate) fn params() -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    )
}

/// A budget with no bound.
pub(crate) fn budget() -> Budget {
    Budget::new(Instant::now(), &params()).unwrap()
}

/// A budget whose deadline passed before the test began, unchecked, with
/// `units` of work between two full checks.
pub(crate) fn passed_after(units: u64) -> Budget {
    let mut p = params();
    p.deadline = Some(Duration::from_millis(1));
    let mut b = Budget::new(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        &p,
    )
    .unwrap();
    b.set_work(units);
    b
}

/// A random sparse row of width `width`: increasing columns, each kept
/// with probability about `density` in 1/8, and nonzero coefficients.
fn sparse(f: &Gf31, rng: &mut Rng, width: usize, density: u64) -> (Vec<MonoId>, Vec<u32>) {
    let p = f.characteristic();
    let (mut cols, mut coefs) = (Vec::new(), Vec::new());
    for c in 0..width {
        if rng.below(8) < density {
            cols.push(MonoId::at(c));
            coefs.push(f.coef(1 + rng.below(p - 1)));
        }
    }
    (cols, coefs)
}

/// A random multiplier of [0, p), zero one time in four or more.
fn multiplier(f: &Gf31, rng: &mut Rng) -> u32 {
    if rng.below(4) == 0 {
        0
    } else {
        f.coef(rng.below(f.characteristic()))
    }
}

/// Every entry of the band and of the dense rows read by `take`: the
/// values agree, and both hold zeros after.
fn read_all(f: &Gf31, band: &mut [[i64; BAND]], dense: &mut [Vec<i64>; BAND], case: &str) {
    for (c, entry) in band.iter_mut().enumerate() {
        for k in 0..BAND {
            let a = f.take(&mut entry[k]);
            let b = f.take(&mut dense[k][c]);
            assert_eq!(a, b, "{case}: column {c}, band row {k}");
        }
    }
    assert!(band.iter().all(|e| e.iter().all(|&x| x == 0)), "{case}");
}

/// Test 1: row-traffic-time.2, .5, .6. The band's row operations of
/// `Gf31` against `load` and `sub_mul` on four dense rows of their own.
#[test]
fn the_band_operations_agree_with_four_dense_rows() {
    for p in PRIMES {
        let f = Gf31::new(p).unwrap();
        let mut rng = Rng::new(p.rotate_left(29) ^ 0xba4d);
        for case in 0..20 {
            let width = 1 + rng.below(300) as usize;
            let name = format!("p {p}, case {case}, width {width}");
            let mut band = vec![[0i64; BAND]; width];
            let mut dense: [Vec<i64>; BAND] = std::array::from_fn(|_| vec![0i64; width]);
            // Whether a band row holds nothing since its last read.
            let mut fresh = [true; BAND];
            for _ in 0..200 {
                let density = 1 + rng.below(8);
                let (cols, coefs) = sparse(&f, &mut rng, width, density);
                match rng.below(4) {
                    0 => {
                        let k = rng.below(BAND as u64) as usize;
                        if fresh[k] {
                            // SAFETY: every column lies below the width.
                            unsafe {
                                f.load_band(&mut band, k, &cols, &coefs);
                                f.load(&mut dense[k], &cols, &coefs);
                            }
                            fresh[k] = false;
                        }
                    }
                    1 => {
                        let c: [u32; BAND] = std::array::from_fn(|_| multiplier(&f, &mut rng));
                        // SAFETY: as above.
                        unsafe {
                            f.sub_mul_band(&mut band, &cols, &coefs, &c);
                            for k in 0..BAND {
                                f.sub_mul(&mut dense[k], &cols, &coefs, c[k]);
                            }
                        }
                        fresh = [false; BAND];
                    }
                    2 => {
                        let k = rng.below(BAND as u64) as usize;
                        let c = multiplier(&f, &mut rng);
                        // SAFETY: as above.
                        unsafe {
                            f.sub_mul_band_row(&mut band, k, &cols, &coefs, c);
                            f.sub_mul(&mut dense[k], &cols, &coefs, c);
                        }
                        fresh[k] = false;
                    }
                    _ => {
                        read_all(&f, &mut band, &mut dense, &name);
                        fresh = [true; BAND];
                    }
                }
            }
            read_all(&f, &mut band, &mut dense, &name);
        }
    }
    // Below 2^16 an entry takes products unreduced: 2^20 subtractions of
    // (p − 1)² into one entry of each band row, from a load of p − 1.
    let f = Gf31::new(65521).unwrap();
    let mut band = vec![[0i64; BAND]; 1];
    let mut dense: [Vec<i64>; BAND] = std::array::from_fn(|_| vec![0i64; 1]);
    let top = f.coef(65520);
    // SAFETY: column 0 lies below the width, 1.
    unsafe {
        for (k, row) in dense.iter_mut().enumerate() {
            f.load_band(&mut band, k, &[MonoId::at(0)], &[top]);
            f.load(row, &[MonoId::at(0)], &[top]);
        }
        for _ in 0..1 << 20 {
            f.sub_mul_band(&mut band, &[MonoId::at(0)], &[top], &[top; BAND]);
            for row in &mut dense {
                f.sub_mul(row, &[MonoId::at(0)], &[top], top);
            }
        }
    }
    read_all(&f, &mut band, &mut dense, "2^20 subtractions over 65521");
}

/// A random matrix over GF(p) on four variables, as `combine_tests` builds
/// one: a basis of `elements` random monic polynomials of 2 to `most` + 1
/// terms of degree at most `degree`, and given rows that are the halves of
/// random S-pairs of the basis, every `twice`-th draw's rows given twice.
/// Symbolic preprocessing adds the reducers.
pub(crate) struct Random {
    pub(crate) field: Gf31,
    pub(crate) basis: Basis<Gf31, Rank1>,
    pub(crate) matrix: Matrix,
}

impl Random {
    pub(crate) fn new(
        p: u64,
        seed: u64,
        (elements, pairs, twice): (usize, usize, usize),
        (most, degree): (u64, u64),
    ) -> Random {
        let field = Gf31::new(p).unwrap();
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut rng = Rng::new(seed ^ p.rotate_left(17) ^ 0xba4e);
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        while basis.len() < elements {
            let mut terms = Vec::new();
            for _ in 0..2 + rng.below(most) {
                let mut e = [0u32; 4];
                let mut degree = 1 + rng.below(degree) as u32;
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
        let mut rows = Vec::new();
        for k in 0..pairs {
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
                if twice > 0 && k % twice == 0 {
                    rows.push(row);
                }
            }
        }
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

    /// A reducer from fresh buffers, as `reducer_in` makes it.
    pub(crate) fn reducer(&self) -> (Reducer<'_, Gf31, Rank1>, Vec<RowId>) {
        self.reducer_in(&mut Buffers::new())
    }

    /// A reducer from `bufs` with the pivots `matrix::reduce` sets before
    /// it reduces: every reducer row, and of the given rows of one leading
    /// column the sparsest; and the rows it then reduces, here by
    /// increasing leading column, where `matrix::reduce` takes them by
    /// decreasing leading column (cyclic-gap.1). The reducer's counts start
    /// at a fresh budget's, zero, as those of the fresh budgets its callers
    /// reduce under, save test 2's two reducers, which share one budget, so
    /// that each counts the other's entries written as well.
    pub(crate) fn reducer_in(
        &self,
        bufs: &mut Buffers<Gf31>,
    ) -> (Reducer<'_, Gf31, Rank1>, Vec<RowId>) {
        let m = &self.matrix;
        let mut red = Reducer::new(&self.field, &self.basis, m, bufs, &budget());
        let lead = |r: RowId| m.row(r)[0].index();
        for r in (m.given..m.rows.len()).map(RowId::at) {
            red.pivot[lead(r)] = Pivot::Row(r);
        }
        let mut rest = Vec::new();
        for r in (0..m.given).map(RowId::at) {
            match red.pivot[lead(r)] {
                Pivot::Row(q) if m.row(q).len() > m.row(r).len() => {
                    rest.push(q);
                    red.pivot[lead(r)] = Pivot::Row(r);
                }
                Pivot::Row(_) => rest.push(r),
                Pivot::None | Pivot::New(_) => red.pivot[lead(r)] = Pivot::Row(r),
            }
        }
        rest.sort_by_key(|&r| lead(r));
        (red, rest)
    }
}

/// A pivot as a pair, to compare two pivot arrays.
fn pivots(red: &Reducer<'_, Gf31, Rank1>) -> Vec<(u8, usize)> {
    red.pivot
        .iter()
        .map(|p| match *p {
            Pivot::None => (0, 0),
            Pivot::Row(r) => (1, r.index()),
            Pivot::New(k) => (2, k.index()),
        })
        .collect()
}

/// Reduces the run `rows` with `red`, as `reduce_sig` reduces a run of
/// reductants: in one band where it holds two rows or more, else alone; the
/// nonzero rows kept in order. The rows reduced to zero.
fn reduce_run(
    red: &mut Reducer<'_, Gf31, Rank1>,
    m: &Matrix,
    rows: &[RowId],
    b: &mut Budget,
) -> Vec<RowId> {
    let made = if rows.len() >= 2 {
        red.reduce_band(rows, b).unwrap()
    } else {
        let r = rows[0];
        let row = red
            .reduce::<Dense16, NoCofactors>(r, m.row(r)[0], &mut NoCofactors, b)
            .unwrap();
        vec![row]
    };
    let mut zero = Vec::new();
    for row in made {
        if row.cols.is_empty() {
            zero.push(row.row);
            continue;
        }
        assert!(red.keep::<Dense16, NoCofactors>(row, &mut NoCofactors));
    }
    zero
}

/// A made row as its row, columns, coefficients, and keeping.
fn made(red: &Reducer<'_, Gf31, Rank1>) -> Vec<(RowId, Vec<MonoId>, Vec<u32>, KeepId)> {
    red.made
        .iter()
        .map(|r: &Reduced<Gf31>| (r.row, r.cols.clone(), r.coefs.clone(), r.kept))
        .collect()
}

/// Test 2: row-traffic-time.1, .4. `Reducer::reduce_band` against the
/// reduction of one row at a time, on random matrices: after every run of
/// rows the rows made, the rows reduced to zero, and the pivots agree.
#[test]
fn a_band_makes_what_its_rows_make_one_at_a_time() {
    // Runs that held a row twice whose first copy made a row, so that the
    // second met its lead in the sweep; and band rows that reduced to zero.
    let (mut met, mut zeros) = (0, 0);
    for p in [7, 65521, 2_147_483_647] {
        for seed in 0..300u64 {
            let shape = (4 + (seed % 9) as usize, 6 + (seed % 23) as usize, 3);
            let terms = [8, 30, 80][(seed % 3) as usize];
            let m = Random::new(p, seed, shape, (terms, 5));
            let name = format!("p {p}, seed {seed}");
            let (mut band, rest) = m.reducer();
            let (mut one, _) = m.reducer();
            let mut rng = Rng::new(seed.rotate_left(7) ^ p);
            let mut b = budget();
            let (mut zero_band, mut zero_one) = (Vec::new(), Vec::new());
            let mut k = 0;
            while k < rest.len() {
                let end = (k + 1 + rng.below(BAND as u64) as usize).min(rest.len());
                let run = &rest[k..end];
                zero_band.extend(reduce_run(&mut band, &m.matrix, run, &mut b));
                for (i, &r) in run.iter().enumerate() {
                    let row = one
                        .reduce::<Dense16, NoCofactors>(
                            r,
                            m.matrix.row(r)[0],
                            &mut NoCofactors,
                            &mut b,
                        )
                        .unwrap();
                    if row.cols.is_empty() {
                        zero_one.push(row.row);
                        if run.len() >= 2 {
                            zeros += 1;
                        }
                        let twin = m.matrix.rows[r.index()];
                        let first = run[..i]
                            .iter()
                            .any(|&q| m.matrix.rows[q.index()] == twin && !zero_one.contains(&q));
                        if run.len() >= 2 && first {
                            met += 1;
                        }
                        continue;
                    }
                    assert!(one.keep::<Dense16, NoCofactors>(row, &mut NoCofactors));
                }
                assert_eq!(made(&band), made(&one), "{name}: rows made, run at {k}");
                assert_eq!(zero_band, zero_one, "{name}: zero rows, run at {k}");
                assert_eq!(pivots(&band), pivots(&one), "{name}: pivots, run at {k}");
                k = end;
            }
            assert!(
                band.band.iter().all(|e| e.iter().all(|&x| x == 0)),
                "{name}: the band holds zeros"
            );
        }
    }
    assert!(met > 0, "no band row met the lead of a band row before it");
    assert!(zeros > 0, "no band row reduced to zero");
}

/// Test 3: row-traffic-time.1. The bands' ends: runs of reductants, below
/// `given`, between reducer rows.
#[test]
fn bands_end_at_four_rows_and_at_reducer_rows() {
    let given = 100;
    // Reductant ids below 100, reducer rows from 100; a run of n
    // reductants after a reducer row.
    for n in [1usize, 2, 3, 4, 5, 8, 9] {
        let mut rows: Vec<RowId> = vec![RowId::at(100)];
        rows.extend((0..n).map(RowId::at));
        rows.push(RowId::at(101));
        rows.push(RowId::at(n));
        let mut k = 1;
        let mut ends = Vec::new();
        while k < 1 + n {
            let j = matrix::band_end(&rows, given, k);
            ends.push(j - k);
            k = j;
        }
        let want: Vec<usize> = match n {
            1 => vec![1],
            2 => vec![2],
            3 => vec![3],
            4 => vec![4],
            5 => vec![4, 1],
            8 => vec![4, 4],
            _ => vec![4, 4, 1],
        };
        assert_eq!(ends, want, "a run of {n}");
        // The reducer row after the run ends the last band, and a run of
        // one at the rows' end ends there.
        assert_eq!(k, 1 + n, "a run of {n}");
        assert_eq!(
            matrix::band_end(&rows, given, rows.len() - 1),
            rows.len(),
            "the last row"
        );
    }
}

/// Test 4: row-traffic-time.7 to .9. The rows long enough for bands, and
/// the rounds that take them.
#[test]
fn long_rows_take_bands_where_the_field_has_them_and_no_cofactors() {
    for r in [1usize, 7, 1000] {
        assert!(!matrix::long_rows(BAND_ENTRIES * r - 1, r), "{r} rows");
        assert!(matrix::long_rows(BAND_ENTRIES * r, r), "{r} rows");
    }
    assert!(matrix::long_rows(0, 0), "no row");
    // A random matrix of 12 elements of up to 151 terms of degree at most
    // 6 and 40 pairs drawn, whose rows hold 64 entries or more in the mean.
    let m = (0..u64::MAX)
        .map(|seed| Random::new(2_147_483_647, seed, (12, 40, 3), (150, 6)))
        .find(|m| matrix::long_rows(m.matrix.entries(), m.matrix.rows.len()))
        .unwrap();
    let m = &m.matrix;
    assert!(matrix::takes_bands::<Gf31, Dense16, NoCofactors>(m));
    assert!(!matrix::takes_bands::<Gf31, Dense16, Tracked<Gf31, Dense16>>(m));
    assert!(!matrix::takes_bands::<Gf64, Dense16, NoCofactors>(m));
    assert!(!matrix::takes_bands::<Gf64Pair, Dense16, NoCofactors>(m));
    // row-traffic-time.13: the pair learn's field takes none.
    assert!(!matrix::takes_bands::<Gf31Pair, Dense16, NoCofactors>(m));
}

/// The random matrix of test 2 under `seed`.
fn random(p: u64, seed: u64) -> Random {
    let shape = (4 + (seed % 9) as usize, 6 + (seed % 23) as usize, 3);
    let terms = [8, 30, 80][(seed % 3) as usize];
    Random::new(p, seed, shape, (terms, 5))
}

/// Reduces every row of `rest` with `red`, in runs of 2 to `BAND` rows,
/// the last perhaps of one, their lengths drawn under `seed`. The rows
/// reduced to zero, and the rows the bands made that are not zero.
fn reduce_all(
    red: &mut Reducer<'_, Gf31, Rank1>,
    m: &Matrix,
    rest: &[RowId],
    seed: u64,
) -> (Vec<RowId>, usize) {
    let mut rng = Rng::new(seed ^ 0xb0f);
    let mut b = budget();
    let (mut zero, mut made) = (Vec::new(), 0);
    let mut k = 0;
    while k < rest.len() {
        let end = (k + 2 + rng.below(BAND as u64 - 1) as usize).min(rest.len());
        let zeros = reduce_run(red, m, &rest[k..end], &mut b);
        if end - k >= 2 {
            made += end - k - zeros.len();
        }
        zero.extend(zeros);
        k = end;
    }
    (zero, made)
}

/// Test 9: row-traffic-time.10, .11. The band and the band rows' work
/// vectors come from the buffers and go back to them. A reducer from
/// buffers whose band holds entries that are not zero, past the matrix's
/// width, makes the rows, the rows reduced to zero, and the pivots of one
/// from fresh buffers. A band given back keeps its capacity for a
/// narrower matrix, and `release` leaves no capacity.
#[test]
fn the_band_comes_from_the_buffers() {
    let (mut banded, mut made_by_bands) = (0, 0);
    for p in [65521, 2_147_483_647] {
        let mut bufs: Buffers<Gf31> = Buffers::new();
        for seed in 0..60u64 {
            let m = random(p, seed);
            let name = format!("p {p}, seed {seed}");
            let width = m.matrix.columns();
            bufs.soil_band(width + 7, 3 * p as i64 + 1);
            let (mut soiled, rest) = m.reducer_in(&mut bufs);
            let (mut fresh, _) = m.reducer();
            let (zero_soiled, made_soiled) = reduce_all(&mut soiled, &m.matrix, &rest, seed);
            let (zero_fresh, _) = reduce_all(&mut fresh, &m.matrix, &rest, seed);
            let took = rest.len() >= 2;
            banded += usize::from(took);
            made_by_bands += made_soiled;
            assert_eq!(made(&soiled), made(&fresh), "{name}: rows made");
            assert_eq!(zero_soiled, zero_fresh, "{name}: zero rows");
            assert_eq!(pivots(&soiled), pivots(&fresh), "{name}: pivots");
            assert!(
                soiled.band.iter().all(|e| e.iter().all(|&x| x == 0)),
                "{name}: the band holds zeros"
            );
            soiled.give_back(&mut bufs);
            if took {
                assert!(
                    bufs.band_capacities()[0] >= width,
                    "{name}: the band given back"
                );
            }
        }
    }
    assert!(banded > 0, "no run took a band");
    assert!(made_by_bands > 0, "no band made a row");

    // Of the seeds over 65521 whose bands make a row, the widest matrix
    // and the narrowest: the band the widest gave back serves the
    // narrowest, its capacity kept.
    let all: Vec<Random> = (0..60u64)
        .map(|seed| random(65521, seed))
        .filter(|m| {
            let (mut red, rest) = m.reducer();
            reduce_all(&mut red, &m.matrix, &rest, 1).1 > 0
        })
        .collect();
    let by_width = |m: &&Random| m.matrix.columns();
    let wide = all.iter().max_by_key(by_width).unwrap();
    let narrow = all.iter().min_by_key(by_width).unwrap();
    assert!(narrow.matrix.columns() < wide.matrix.columns());
    let mut bufs: Buffers<Gf31> = Buffers::new();
    let (mut red, rest) = wide.reducer_in(&mut bufs);
    reduce_all(&mut red, &wide.matrix, &rest, 1);
    red.give_back(&mut bufs);
    let [band, cols, coefs] = bufs.band_capacities();
    assert!(band >= wide.matrix.columns(), "the band given back");
    assert!(
        cols > 0 && coefs > 0,
        "the band rows' work vectors given back"
    );
    let (mut red, rest) = narrow.reducer_in(&mut bufs);
    reduce_all(&mut red, &narrow.matrix, &rest, 1);
    red.give_back(&mut bufs);
    assert_eq!(
        bufs.band_capacities()[0],
        band,
        "the band kept its capacity"
    );
    bufs.release();
    assert_eq!(bufs.band_capacities(), [0; 3]);
}

/// Test 10: row-traffic-time.12. A band spends each pivot row's length
/// once per band row it is subtracted from. On test 2's matrices over
/// 65521 whose first run of up to `BAND` rows spends more than
/// 4·(T + 4·L) units in its band, L the matrix's longest row: unbounded,
/// the band spends at least what `reduce` spends on the run's first row
/// alone; with a deadline passed and unchecked and T units a check, it
/// stops with `Deadline` after one check and at most T + 4·L units.
#[test]
fn a_band_spends_its_row_operations() {
    const T: u64 = 256;
    let mut cases = 0;
    for seed in 0..300u64 {
        let m = random(65521, seed);
        let (_, rest) = m.reducer();
        if rest.len() < 2 {
            continue;
        }
        let run = &rest[..BAND.min(rest.len())];
        let longest = (0..m.matrix.rows.len())
            .map(|r| m.matrix.row(RowId::at(r)).len())
            .max()
            .unwrap() as u64;
        let bound = T + BAND as u64 * longest;
        let (mut red, _) = m.reducer();
        let mut b = budget();
        red.reduce_band(run, &mut b).unwrap();
        let total = b.spent();
        if total <= 4 * bound {
            continue;
        }
        cases += 1;
        let (mut one, _) = m.reducer();
        let mut b = budget();
        let r = run[0];
        one.reduce::<Dense16, NoCofactors>(r, m.matrix.row(r)[0], &mut NoCofactors, &mut b)
            .unwrap();
        assert!(
            total >= b.spent(),
            "seed {seed}: the band spent {total}, its first row alone {spent}",
            spent = b.spent()
        );
        let (mut red, _) = m.reducer();
        let mut b = passed_after(T);
        assert_eq!(
            red.reduce_band(run, &mut b).err(),
            Some(Stop::Deadline),
            "seed {seed}"
        );
        assert_eq!(b.checks(), 1, "seed {seed}");
        assert!(
            b.spent() <= bound,
            "seed {seed}: {spent} spent before the stop, {total} in all",
            spent = b.spent()
        );
    }
    assert!(cases >= 10, "{cases} bands spend enough to stop inside");
}
