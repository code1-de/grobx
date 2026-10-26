//! table-read-time, the unit tests: the table's reads the item takes
//! unchecked give what was stored, at every index, and a read one past
//! the end of the table, the basis, or a polynomial panics in the test profile, whose debug assertions
//! check every index (plan.md, _Test code_, 1 and 2).

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::table::Table;

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// Drawn exponents of n variables of degree at most `bound`.
fn drawn(r: &mut impl Rng, n: usize, bound: u32) -> Vec<u32> {
    let mut e = vec![0u32; n];
    let mut left = bound;
    for x in &mut e {
        let want = r.next_u32() % 5;
        *x = want.min(left);
        left -= *x;
    }
    e
}

/// A table of the format `M` on n variables holding the unit monomial,
/// each variable alone at the degree bound, and drawn monomials, with the
/// exponents of each id in order.
fn table<M: Format>(n: usize, seed: u64) -> (Table<M>, Vec<Vec<u32>>) {
    let mut r = rng(seed);
    let mut table = Table::new(M::new(n), &mut r);
    let mut all = vec![vec![0; n]];
    for i in 0..n {
        let mut e = vec![0; n];
        e[i] = M::MAX_DEGREE;
        all.push(e);
    }
    for _ in 0..200 {
        all.push(drawn(&mut r, n, M::MAX_DEGREE));
    }
    let mut by_id: Vec<Vec<u32>> = Vec::new();
    for e in &all {
        let id = table.intern(e).unwrap().index();
        if id == by_id.len() {
            by_id.push(e.clone());
        }
        assert_eq!(by_id[id], *e, "id {id}");
    }
    assert_eq!(by_id.len(), table.len());
    (table, by_id)
}

/// Test 1 for one format: every id's words unpack to its exponents, its
/// degree is their sum, its mask is the format's mask of its words, and
/// its column and stamp give back what was set, the first id and the last
/// among them.
fn reads_agree<M: Format>(n: usize) {
    let (mut table, by_id) = table::<M>(n, 0x7ab1 + n as u64);
    let mut out = vec![0u32; n];
    for (id, e) in by_id.iter().enumerate() {
        let id = MonoId::at(id);
        let case = format!("{n} variables, id {id}");
        table.format().unpack(table.monomial(id), &mut out);
        assert_eq!(out, *e, "{case}: monomial");
        assert_eq!(table.degree(id), e.iter().sum::<u32>(), "{case}: degree");
        assert_eq!(
            table.mask(id),
            table.format().mask(table.monomial(id)),
            "{case}: mask"
        );
        table.set_column(id, MonoId::at(3 * id.index() + 1));
        table.set_stamp(id, id.raw() ^ 0x55);
    }
    for id in (0..table.len()).map(MonoId::at) {
        let column = MonoId::at(3 * id.index() + 1);
        assert_eq!(table.column(id), column, "{n} variables, id {id}");
        assert_eq!(table.stamp(id), id.raw() ^ 0x55, "{n} variables, id {id}");
    }
    // Interning again finds each id. Each id's hash is read in range, and
    // its value is not compared.
    for (id, e) in by_id.iter().enumerate() {
        assert_eq!(table.intern(e).unwrap().index(), id, "{n} variables");
        let _ = table.hash(MonoId::at(id));
    }
}

/// Test 1: decision table-read-time.1, at 1, 2, 3, 5, and 7 words in
/// `Dense8`, and at 1 to 3 words in `Dense16` and `Dense32`.
#[test]
fn reads_in_range_give_what_was_stored() {
    for n in [1, 2, 7, 8, 9, 16, 17, 33, 54] {
        reads_agree::<Dense8>(n);
    }
    for n in [1, 3, 4, 5, 9] {
        reads_agree::<Dense16>(n);
    }
    for n in [1, 2, 3, 5] {
        reads_agree::<Dense32>(n);
    }
}

/// A table of three variables and its length.
fn small() -> (Table<Dense8>, MonoId) {
    let (table, _) = table::<Dense8>(3, 0x7ab0);
    let len = MonoId::at(table.len());
    (table, len)
}

// Test 2: a read or a write one past the end panics in the test profile
// (table-read-time.1, .3, .4).

#[test]
#[should_panic(expected = "past the table's")]
fn monomial_past_the_end() {
    let (table, len) = small();
    let _ = table.monomial(len);
}

#[test]
#[should_panic(expected = "degree of id")]
fn degree_past_the_end() {
    let (table, len) = small();
    let _ = table.degree(len);
}

#[test]
#[should_panic(expected = "hash of id")]
fn hash_past_the_end() {
    let (table, len) = small();
    let _ = table.hash(len);
}

#[test]
#[should_panic(expected = "mask of id")]
fn mask_past_the_end() {
    let (table, len) = small();
    let _ = table.mask(len);
}

#[test]
#[should_panic(expected = "column of id")]
fn column_past_the_end() {
    let (table, len) = small();
    let _ = table.column(len);
}

#[test]
#[should_panic(expected = "column of id")]
fn set_column_past_the_end() {
    let (mut table, len) = small();
    table.set_column(len, MonoId::at(0));
}

#[test]
#[should_panic(expected = "stamp of id")]
fn stamp_past_the_end() {
    let (table, len) = small();
    let _ = table.stamp(len);
}

#[test]
#[should_panic(expected = "stamp of id")]
fn set_stamp_past_the_end() {
    let (mut table, len) = small();
    table.set_stamp(len, 1);
}

/// A basis of one element, x + 1 over GF(65521), in a table of x.
fn one_element() -> Basis<Gf31, Rank1> {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense8::new(1), &mut rng(1));
    let x = table.intern(&[1]).unwrap();
    let one = table.intern(&[0]).unwrap();
    let mut terms = vec![(x, field.coef(1)), (one, field.coef(1))];
    let poly = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
    let mut basis = Basis::new();
    basis.push(poly, false).unwrap();
    basis
}

#[test]
#[should_panic(expected = "past the basis's")]
fn element_past_the_end() {
    let basis = one_element();
    let _ = basis.element(ElementId::at(1));
}

#[test]
#[should_panic(expected = "the lead of element")]
fn lead_past_the_end() {
    let basis = one_element();
    let _ = basis.lead(ElementId::at(1));
}

#[test]
#[should_panic(expected = "the length of element")]
fn length_past_the_end() {
    let basis = one_element();
    let _ = basis.length(ElementId::at(1));
}

#[test]
#[should_panic(expected = "the lead of the zero polynomial")]
fn lead_of_the_zero_polynomial() {
    let field = Gf31::new(65521).unwrap();
    let table = Table::new(Dense8::new(1), &mut rng(1));
    let zero: Poly<Gf31, Rank1> = Poly::from_terms(&mut Vec::new(), &field, &table, &Grevlex, ());
    let _ = zero.lead();
}
