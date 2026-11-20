//! column-index-peak, sub-item 3, the crate's test code: a matrix whose
//! columns are held in 2 bytes reduces as its twin in 4 bytes, each row on
//! its own, with the zero tests, in combinations, in bands, in the bands
//! of the signature run's reductants, and in the inter-reduction; and a
//! column vector of 2-byte entries widens in place
//! to the same entries (`iterations/0115-column-index-peak/plan.md`, _Test
//! code_, 7 and 8). The signature run's reduction and the replays over ℚ
//! read narrow matrices in the calls of the item's tests 9 to 13.
//!
//! `Matrix::narrowed` and `Columns::of` are the test profile's: sub-item 3
//! adds them. Sub-item 3 adds a test beside the plan's:
//! `a_narrow_build_makes_the_wide_matrix`, a build into the narrow column
//! vector a build without the matrix before makes. Sub-item 4 adds
//! `a_build_widens_at_its_bound`, the rows whose widening no call of the
//! item's tests reaches. Every wide matrix here is built under
//! `NARROW_BOUND` at Some(0).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::band_tests::{reduce_run, reducer_of};
use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::columns::{self, Columns};
use crate::cyclic_gap_tests::{Random, long, short};
use crate::f4;
use crate::field::{BAND, Gf31};
use crate::id::{RowId, RowSource};
use crate::matrix::{
    self, Buffers, Combine, Matrix, Modes, NARROW_BOUND, Pivot, Reduced, Reduction,
};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Grevlex, Lex, Order};
use crate::poly::Basis;
use crate::rebuilt_reducer_tests::{element, same};
use crate::shared_tests::budget;
use crate::systems::Rng;
use crate::table::Table;

/// A made row as the test compares it: the row it was made from, its
/// columns, and its coefficients.
type Row = (usize, Vec<u32>, Vec<u32>);

/// The made rows, in the order made.
fn made(rows: &[Reduced<Gf31>]) -> Vec<Row> {
    rows.iter()
        .map(|r| {
            (
                r.row.index(),
                r.cols.iter().map(|c| c.raw()).collect(),
                r.coefs.clone(),
            )
        })
        .collect()
}

/// A pivot as the test compares it: 0 for none, 1 and the row for a row
/// of the matrix, 2 and the index for a made row.
fn pivot(p: Pivot) -> (u8, usize) {
    match p {
        Pivot::None => (0, 0),
        Pivot::Row(r) => (1, r.index()),
        Pivot::New(k) => (2, k.index()),
    }
}

/// `build` with every build on this thread wide (`NARROW_BOUND` at
/// Some(0)).
pub(crate) fn wide<T>(build: impl FnOnce() -> T) -> T {
    NARROW_BOUND.set(Some(0));
    let out = build();
    NARROW_BOUND.set(None);
    out
}

/// Asserts the two reductions agree: the made rows, the zero rows, the
/// pivots, the counts, and the mode.
fn agree(name: &str, wide: &Reduction<Gf31>, narrow: &Reduction<Gf31>) {
    assert_eq!(
        made(&wide.made),
        made(&narrow.made),
        "{name}: the made rows"
    );
    let zero = |r: &Reduction<Gf31>| r.zero.iter().map(|z| z.index()).collect::<Vec<_>>();
    assert_eq!(zero(wide), zero(narrow), "{name}: the zero rows");
    let pivots = |r: &Reduction<Gf31>| r.pivot.iter().map(|&p| pivot(p)).collect::<Vec<_>>();
    assert_eq!(pivots(wide), pivots(narrow), "{name}: the pivots");
    assert_eq!(
        (wide.reductions, wide.zeros),
        (narrow.reductions, narrow.zeros),
        "{name}: the counts"
    );
    // `RoundMode` compares by its name.
    let mode = |r: &Reduction<Gf31>| format!("{mode:?}", mode = r.mode);
    assert_eq!(mode(wide), mode(narrow), "{name}: the mode");
}

/// `matrix::reduce` of `matrix`, the matrix of `m` or its twin, full, with
/// combinations drawn from seed `combine` and tests from seed `tests` where
/// they are Some, and the run's modes where `modes` holds.
fn reduce(
    m: &Random,
    matrix: &Matrix,
    combine: Option<u64>,
    tests: Option<u64>,
    modes: bool,
) -> Reduction<Gf31> {
    let mut combine = combine.map(Combine::new);
    let mut tests = tests.map(Combine::new);
    let mut modes = modes.then(Modes::new);
    matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
        &m.field,
        &m.basis,
        matrix,
        &mut NoCofactors,
        &mut budget(),
        true,
        combine.as_mut(),
        &mut Buffers::new(),
        tests.as_mut(),
        modes.as_mut(),
    )
    .unwrap()
}

#[test]
fn a_narrow_matrix_reduces_as_its_wide_twin() {
    for p in [65521, 7, 2_147_483_647] {
        for seed in 1..=4 {
            for (shape_name, shape) in [("short", short(12, 40, 3, 2)), ("long", long(6, 8, 0, 1))]
            {
                let mut m = wide(|| Random::new(p, seed, &shape));
                let twin = m
                    .matrix
                    .narrowed()
                    .expect("a test matrix holds fewer than 2^16 columns");
                let name = format!("{shape_name}, p {p}, seed {seed}");
                assert_eq!(m.matrix.index(), 4, "{name}: the matrix wide");
                // Each row on its own, with and without the tests; in
                // combinations, which a long matrix takes in bands, with
                // and without the modes.
                for (case, combine, tests, modes) in [
                    ("alone", None, None, false),
                    ("tests", None, Some(9), false),
                    ("combinations", Some(7), None, false),
                    ("modes", Some(7), None, true),
                ] {
                    let wide = reduce(&m, &m.matrix, combine, tests, modes);
                    let narrow = reduce(&m, &twin, combine, tests, modes);
                    agree(&format!("{name}, {case}"), &wide, &narrow);
                }
                // The signature run's bands (`Reducer::reduce_band`): the rows
                // to reduce in runs of up to BAND, each run in one band, as
                // `reduce_sig` reduces a run of reductants.
                let bands = |matrix: &Matrix| {
                    let mut bufs = Buffers::new();
                    let (mut red, rest) = reducer_of(&m.field, &m.basis, matrix, &mut bufs);
                    let mut b = budget();
                    let zero: Vec<usize> = rest
                        .chunks(BAND)
                        .flat_map(|run| reduce_run(&mut red, matrix, run, &mut b))
                        .map(RowId::index)
                        .collect();
                    (made(&red.made), zero)
                };
                assert_eq!(
                    bands(&m.matrix),
                    bands(&twin),
                    "{name}, the bands: the made and the zero rows"
                );
                // The inter-reduction runs on the matrix that inter-reduces
                // the basis's `G_min`, one row a leading column, as the run
                // builds it.
                let inter_wide =
                    wide(|| f4::inter_matrix(&mut m.table, &Grevlex, &m.basis, &mut budget()))
                        .unwrap();
                assert_eq!(
                    inter_wide.index(),
                    4,
                    "{name}: the inter-reduction's matrix wide"
                );
                let inter_twin = inter_wide
                    .narrowed()
                    .expect("a test matrix holds fewer than 2^16 columns");
                let inter = |matrix: &Matrix| {
                    matrix::inter_reduce::<Gf31, Dense16, Rank1, NoCofactors>(
                        &m.field,
                        &m.basis,
                        matrix,
                        &mut NoCofactors,
                        &mut budget(),
                        &mut Buffers::new(),
                    )
                    .unwrap()
                };
                assert_eq!(
                    made(&inter(&inter_wide)),
                    made(&inter(&inter_twin)),
                    "{name}, the inter-reduction: the rows"
                );
            }
        }
    }
}

#[test]
fn a_narrow_vector_widens_in_place() {
    let mut rng = Rng::new(31);
    let mut lens: Vec<usize> = vec![0, 1, 2, 3, 7, 8, 999, 1000];
    lens.extend((0..20).map(|_| rng.below(1001) as usize));
    for len in lens {
        let entries: Vec<usize> = (0..len).map(|_| rng.below(1 << 16) as usize).collect();
        let narrow = Columns::of(true, &entries);
        let mut widened = Columns::of(true, &entries);
        // A chunk of 7 entries, so that the widening crosses chunks.
        widened.widen(7, |_| Ok::<(), ()>(())).unwrap();
        let wide = Columns::of(false, &entries);
        let read = |c: &Columns| (0..c.len()).map(|i| c.entry(i)).collect::<Vec<usize>>();
        assert_eq!(read(&narrow), entries, "{len} entries narrow");
        assert_eq!(read(&widened), entries, "{len} entries widened");
        assert!(
            !widened.is_narrow(),
            "{len} entries: wide after the widening"
        );
        assert_eq!(widened.words(), wide.words(), "{len} entries: the words");
        if len < 2 {
            continue;
        }
        // A copy within each vector and a truncation give the same entries
        // in both widths.
        let (from, to) = (rng.below(len as u64 / 2) as usize, len / 2);
        let at = rng.below((len - (to - from)) as u64) as usize;
        let mut want = entries.clone();
        want.copy_within(from..to, at);
        let keep = rng.below(len as u64) as usize;
        want.truncate(keep);
        for (width, mut c) in [("narrow", narrow), ("wide", wide)] {
            c.copy_within(from..to, at);
            c.truncate(keep);
            assert_eq!(
                read(&c),
                want,
                "{len} entries {width}: copied and truncated"
            );
        }
    }
}

/// Whether every row of `m` holds its columns in 2 bytes.
fn is_narrow(m: &Matrix) -> bool {
    (0..m.rows.len()).all(|r| match m.row(RowId::at(r)) {
        columns::Row::Narrow(_) => true,
        columns::Row::Wide(_) => false,
    })
}

/// The test beside the plan's in one order: a random matrix's given rows
/// built from buffers that hand on a wide column vector and no matrix, and
/// built again through the same buffers, which hold the first matrix and
/// hand its reducer rows on, against the wide build from fresh buffers,
/// made first:
/// each narrow matrix is the wide one row for row, it holds 2-byte
/// columns, the second took a row from the first where it has a reducer
/// row, and the second's reduction is the wide matrix's.
fn narrow_builds_in<O: Order<Dense16> + Copy>(order: O, order_name: &str) {
    for p in [65521, 7] {
        for seed in 1..=4 {
            for (shape_name, shape) in [("short", short(12, 30, 3, 2)), ("long", long(6, 8, 0, 1))]
            {
                let mut m = Random::with_order(p, seed, &shape, &order);
                let given: Vec<RowSource> = m.matrix.rows[..m.matrix.given].to_vec();
                let name = format!("{order_name}, {shape_name}, p {p}, seed {seed}");
                let build = |m: &mut Random, bufs: &mut Buffers<Gf31>| {
                    matrix::build(
                        &mut m.table,
                        &m.basis,
                        &order,
                        given.clone(),
                        bufs,
                        &mut budget(),
                    )
                    .unwrap()
                };
                let wide = wide(|| build(&mut m, &mut Buffers::new()));
                assert_eq!(wide.index(), 4, "{name}: the matrix wide");
                let mut bufs = Buffers::of_cols(Columns::new());
                let first = build(&mut m, &mut bufs);
                same(&format!("{name}, the first"), &wide, &first);
                assert!(is_narrow(&first), "{name}: the first matrix narrow");
                bufs.keep(first);
                let second = build(&mut m, &mut bufs);
                same(&format!("{name}, the second"), &wide, &second);
                assert!(is_narrow(&second), "{name}: the second matrix narrow");
                assert!(
                    second.rows.len() == second.given || second.kept() > 0,
                    "{name}: a reducer row taken from the first matrix"
                );
                agree(
                    &name,
                    &reduce(&m, &wide, Some(7), None, false),
                    &reduce(&m, &second, Some(7), None, false),
                );
            }
        }
    }
}

#[test]
fn a_narrow_build_makes_the_wide_matrix() {
    // Grevlex numbers the second matrix's columns by the merge with the
    // first's, lex by the sort, whose taken rows map through the first
    // matrix's monomials.
    narrow_builds_in(Grevlex, "grevlex");
    narrow_builds_in(Lex, "lex");
}

/// Asserts `m` is the wide build `wide` row for row and holds 2-byte
/// columns only where its columns lie within `bound`.
fn within(name: &str, wide: &Matrix, m: &Matrix, bound: usize) {
    same(name, wide, m);
    assert!(
        m.index() == 4 || m.columns() <= bound,
        "{name}: a matrix of {columns} columns in 2 bytes",
        columns = m.columns()
    );
}

/// The test beside the plan's at the bound, column-index-peak.7: under
/// every bound from 1 to 24, a build is the wide build row for row and is
/// narrow only where its columns lie within the bound. The system is
/// written out so that the rows that pass the bound are those whose
/// widening no call of the item's tests reaches: in `build`, a row taken
/// from a narrow kept matrix after the build's last product, and in
/// `build_rows`, as a replay builds, a row whose every term is new to the
/// round, which reaches one past the bound. In `build` every row enters a
/// monomial the round holds, its given row's lead or its reducer's.
#[test]
fn a_build_widens_at_its_bound() {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(4), &mut Xoshiro256PlusPlus::seed_from_u64(5));
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    // g = x^2 and ten terms of degree 2 or less; h1 = z^5 + x^2; h2 = y^5 +
    // x^2 and ten terms of degree 3 and 4. No lead, x^2, z^5, or y^5,
    // divides a term of a tail but x^2.
    let tail_g: [[u32; 4]; 10] = [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
        [1, 0, 0, 0],
    ];
    let tail_h2: [[u32; 4]; 10] = [
        [0, 3, 0, 0],
        [0, 2, 1, 0],
        [0, 1, 2, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [0, 2, 0, 1],
        [0, 0, 2, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
    ];
    let terms = |lead: [u32; 4], tail: &[[u32; 4]]| {
        let mut terms = vec![(lead, 1)];
        terms.extend(tail.iter().zip(2..).map(|(&e, c)| (e, c)));
        terms
    };
    let mut push = |terms: &[([u32; 4], u64)]| {
        let g = element(&mut table, &field, terms);
        basis
            .push_min(&table, g, &mut Budget::unbounded())
            .unwrap()
            .id
    };
    let g = push(&terms([2, 0, 0, 0], &tail_g));
    let h1 = push(&terms([0, 0, 5, 0], &[[2, 0, 0, 0]]));
    let mut h2_tail = vec![[2, 0, 0, 0]];
    h2_tail.extend(tail_h2);
    let h2 = push(&terms([0, 5, 0, 0], &h2_tail));
    let one = table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
    let row = |element| RowSource {
        element,
        multiplier: one,
    };
    let build = |table: &mut Table<Dense16>, given: RowSource, bufs: &mut Buffers<Gf31>| {
        matrix::build(table, &basis, &Grevlex, vec![given], bufs, &mut budget()).unwrap()
    };
    let rows = |table: &mut Table<Dense16>| {
        matrix::build_rows(
            table,
            &basis,
            &Grevlex,
            vec![row(g)],
            &mut Buffers::new(),
            &mut budget(),
        )
        .unwrap()
    };
    // The first matrix, 12 columns, takes g's row as a reducer; the second
    // takes it from the first after its given row's 12 columns, and has 22.
    let first_wide = wide(|| build(&mut table, row(h1), &mut Buffers::new()));
    let second_wide = wide(|| build(&mut table, row(h2), &mut Buffers::new()));
    let rows_wide = wide(|| rows(&mut table));
    assert_eq!(
        (
            first_wide.columns(),
            second_wide.columns(),
            rows_wide.columns()
        ),
        (12, 22, 11),
        "the matrices' columns"
    );
    for bound in 1..=24 {
        let name = format!("at {bound}");
        NARROW_BOUND.set(Some(bound));
        let mut bufs = Buffers::new();
        let first = build(&mut table, row(h1), &mut bufs);
        within(&format!("{name}, the first"), &first_wide, &first, bound);
        bufs.keep(first);
        let second = build(&mut table, row(h2), &mut bufs);
        within(&format!("{name}, the second"), &second_wide, &second, bound);
        assert_eq!(second.kept(), 1, "{name}: g's row taken from the first");
        let alone = rows(&mut table);
        within(&format!("{name}, build_rows"), &rows_wide, &alone, bound);
        NARROW_BOUND.set(None);
    }
}
