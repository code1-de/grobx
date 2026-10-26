//! matrix-slack, the crate's test code: `matrix::make_room` against its
//! rule, the columns of a matrix in their reservation from the row that
//! passes `RESERVE_PAST` on, a kept matrix shrunk, and the reservation's
//! unwritten pages outside the resident set (plan.md, _Test code_, 1 to
//! 5).

use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::child_tests;
use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{self, Matrix, RESERVATION, RESERVE_PAST, make_room};
use crate::module::Rank1;
use crate::monomial::{Dense8, Format};
use crate::order::Grevlex;
use crate::os::resident;
use crate::poly::{Basis, Poly};
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

/// A vector of `len` entries, each its index, grown by pushes.
fn columns(len: usize) -> Vec<MonoId> {
    let mut cols = Vec::new();
    for k in 0..len {
        cols.push(MonoId::at(k));
    }
    cols
}

/// Whether the entries are still each its index.
fn intact(cols: &[MonoId]) -> bool {
    cols.iter().enumerate().all(|(k, &c)| c == MonoId::at(k))
}

/// Test 1, matrix-slack.1, .2, .7: columns that stay at `RESERVE_PAST`
/// entries or below keep their capacity; columns that pass take
/// `RESERVATION`, their entries unchanged; a row longer than `RESERVATION`
/// finds room; columns that hold a reservation keep it as it is.
#[test]
fn make_room_reserves_past_its_bound_alone() {
    let mut empty: Vec<MonoId> = Vec::new();
    make_room(&mut empty, 10);
    assert_eq!(empty.capacity(), 0, "10 entries of none");

    let mut below = columns(RESERVE_PAST - 6);
    let capacity = below.capacity();
    make_room(&mut below, 6);
    assert_eq!(below.capacity(), capacity, "RESERVE_PAST entries exactly");
    assert!(intact(&below));

    let mut past = columns(RESERVE_PAST - 6);
    make_room(&mut past, 7);
    assert!(
        past.capacity() >= RESERVATION,
        "one entry past RESERVE_PAST"
    );
    assert_eq!(past.len(), RESERVE_PAST - 6);
    assert!(intact(&past), "the entries moved unchanged");

    let mut first: Vec<MonoId> = Vec::new();
    make_room(&mut first, RESERVE_PAST + 1);
    assert!(
        first.capacity() >= RESERVATION,
        "a first row that passes alone"
    );
    assert_eq!(first.len(), 0, "the reservation holds no column");

    let mut long: Vec<MonoId> = Vec::new();
    make_room(&mut long, RESERVATION + 1);
    assert!(
        long.capacity() > RESERVATION,
        "a row longer than RESERVATION"
    );

    let (pointer, capacity) = (past.as_ptr(), past.capacity());
    make_room(&mut past, 1000);
    make_room(&mut past, RESERVE_PAST);
    assert_eq!(past.as_ptr(), pointer, "a reservation held");
    assert_eq!(past.capacity(), capacity, "a reservation held");
}

/// Test 2, matrix-slack.1, .4: rows appended as `multiply` appends them,
/// `make_room` and then the row, never move the columns from the row that
/// passes `RESERVE_PAST` up to `RESERVATION` entries; a row past
/// `RESERVATION` is appended with every entry intact.
#[test]
fn the_reservation_holds_its_columns_in_place() {
    const ROW: usize = 5005;
    let mut cols: Vec<MonoId> = Vec::new();
    let mut pointer = None;
    while cols.len() + ROW <= RESERVATION {
        make_room(&mut cols, ROW);
        let start = cols.len();
        cols.extend((start..start + ROW).map(MonoId::at));
        if cols.len() > RESERVE_PAST {
            let at = *pointer.get_or_insert(cols.as_ptr());
            assert_eq!(
                cols.as_ptr(),
                at,
                "the columns moved at {len}",
                len = cols.len()
            );
        }
    }
    assert!(pointer.is_some(), "the rows passed RESERVE_PAST");
    assert!(intact(&cols));
    make_room(&mut cols, ROW);
    let start = cols.len();
    cols.extend((start..start + ROW).map(MonoId::at));
    assert!(cols.len() > RESERVATION);
    assert!(cols.capacity() >= cols.len());
    assert!(intact(&cols), "the columns past RESERVATION");
}

/// A table on 6 variables in `Dense8`, and a basis of one monic element:
/// every monomial of degree 9 or less, 5,005 terms, its coefficients
/// drawn below 65521.
fn element() -> (Gf31, Table<Dense8>, Basis<Gf31, Rank1>) {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense8::new(6), &mut Xoshiro256PlusPlus::seed_from_u64(41));
    let mut draw = Xoshiro256PlusPlus::seed_from_u64(0x51ac);
    let mut terms = Vec::new();
    for e in exponents(6, 9) {
        let m = table.intern(&e).unwrap();
        let c = 1 + draw.next_u64() % 65520;
        terms.push((m, field.coef(c)));
    }
    let mut g: Poly<Gf31, Rank1> = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
    assert_eq!(g.len(), 5005);
    g.make_monic(&field);
    let mut basis = Basis::new();
    basis.push_min(&table, g).unwrap();
    (field, table, basis)
}

/// Every exponent vector on n variables of degree at most d, by
/// increasing degree.
fn exponents(n: usize, d: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for degree in 0..=d {
        fill(&mut vec![0u32; n], 0, degree, &mut out);
    }
    out
}

/// Every vector of `e` with the entries before i as they are and the
/// rest summing to `left`, pushed to `out`.
fn fill(e: &mut [u32], i: usize, left: u32, out: &mut Vec<Vec<u32>>) {
    if i + 1 == e.len() {
        e[i] = left;
        out.push(e.to_vec());
        return;
    }
    for k in (0..=left).rev() {
        e[i] = k;
        fill(e, i + 1, left - k, out);
    }
    e[i] = 0;
}

/// The matrix of round `round` of `count` multiples of the basis's
/// element, its multipliers the first `count` monomials of degree 1 to 3,
/// by `build_rows`, and the multipliers. Each matrix of a table takes a
/// round of its own.
fn multiples(
    table: &mut Table<Dense8>,
    basis: &Basis<Gf31, Rank1>,
    count: usize,
    round: u32,
) -> (Matrix, Vec<MonoId>) {
    let multipliers: Vec<MonoId> = exponents(6, 3)
        .into_iter()
        .filter(|e| e.iter().sum::<u32>() > 0)
        .take(count)
        .map(|e| table.intern(&e).unwrap())
        .collect();
    assert_eq!(multipliers.len(), count);
    let rows = multipliers
        .iter()
        .map(|&t| RowSource {
            element: ElementId::at(0),
            multiplier: t,
        })
        .collect();
    let m = matrix::build_rows(
        table,
        basis,
        &Grevlex,
        round,
        rows,
        Vec::new(),
        &mut budget(),
    )
    .unwrap();
    (m, multipliers)
}

/// Test 3, matrix-slack.1: 60 multiples, 300,300 entries, sit in a
/// reservation, and every row's columns are its multiplier's products with
/// the element's monomials, in order; 10 multiples, 50,050 entries, take
/// none.
#[test]
fn a_matrix_past_the_bound_keeps_its_rows() {
    let (_, mut table, basis) = element();
    let (m, multipliers) = multiples(&mut table, &basis, 60, 1);
    assert_eq!(m.entries(), 60 * 5005);
    assert!(
        m.capacity() >= RESERVATION,
        "a capacity of {capacity} for {entries} entries",
        capacity = m.capacity(),
        entries = m.entries()
    );
    let g = basis.element(ElementId::at(0));
    for (r, &t) in multipliers.iter().enumerate() {
        let row = m.row(RowId::at(r));
        assert!(row.windows(2).all(|w| w[0] < w[1]), "row {r} increasing");
        let got: Vec<MonoId> = row.iter().map(|&c| m.monomials[c.index()]).collect();
        let want: Vec<MonoId> = g
            .monomials
            .iter()
            .map(|&u| table.mul(t, u).unwrap())
            .collect();
        assert_eq!(got, want, "row {r}");
    }

    let (small, _) = multiples(&mut table, &basis, 10, 2);
    assert_eq!(small.entries(), 10 * 5005);
    assert!(small.capacity() < RESERVATION, "50,050 entries");
}

/// Test 4, q-replay-time.14 with matrix-slack.1: the matrix of 300,300
/// entries, shrunk as a replay keeps it, gives its reservation back.
#[test]
fn a_kept_matrix_gives_its_reservation_back() {
    let (_, mut table, basis) = element();
    let (mut m, _) = multiples(&mut table, &basis, 60, 1);
    assert!(m.capacity() >= RESERVATION);
    m.shrink_to_fit();
    assert!(m.capacity() >= m.entries());
    assert!(
        m.capacity() < RESERVATION,
        "a capacity of {capacity} for {entries} entries",
        capacity = m.capacity(),
        entries = m.entries()
    );
}

/// Test 5, matrix-slack.3, .4: in a child process of this test binary,
/// which runs this test alone, building the matrix of 300,300 entries,
/// 1.2 MB of columns, raises the resident set by less than 16 MiB, where
/// its reservation spans 32 MiB. The parent runs the child.
#[test]
fn the_reservation_is_resident_as_written() {
    if child_tests::running() {
        let (_, mut table, basis) = element();
        let before = resident();
        let (m, _) = multiples(&mut table, &basis, 60, 1);
        let after = resident();
        assert!(m.capacity() >= RESERVATION);
        let grown = after.saturating_sub(before);
        assert!(
            grown < 16 << 20,
            "the resident set grew by {grown} bytes for {entries} entries",
            entries = m.entries()
        );
        return;
    }
    child_tests::run("slack_tests::the_reservation_is_resident_as_written");
}
