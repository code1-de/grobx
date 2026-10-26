//! matrix-build-time, the crate's test code: a row's products by
//! `Table::row_products` against the lookup per term on a twin table, and
//! the two reducer searches, their masks tested eight at a time, against
//! plain scans (plan.md, _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;
use std::convert::Infallible;

use rand_core::{Rng, SeedableRng, TryRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId};
use crate::matrix::Divisors;
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::sig::{Sig, Signatures};
use crate::table::{Limit, Table};

pub(crate) fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub(crate) fn below(r: &mut Xoshiro256PlusPlus, n: u64) -> u64 {
    r.next_u64() % n
}

/// Hands out the given words in turn, then 0: the hash multipliers of a
/// table, which `Table::new` draws one per variable, in order.
struct Given(std::vec::IntoIter<u64>);

impl TryRng for Given {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.try_next_u64()? as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.0.next().unwrap_or(0))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        dst.fill(0);
        Ok(())
    }
}

/// The hashes of `lookup_tests.rs`, by name.
const HASHES: [&str; 4] = [
    "drawn",
    // Every hash's low 32 bits are 0: every slot's tag agrees.
    "tags that agree",
    // Every hash is 0: one probe chain from slot 0.
    "one chain",
    // Every hash is 2^64 minus the degree: every chain from a home slot
    // past the unit monomial's wraps past the end.
    "chains that wrap",
];

/// The n multipliers of the hashes `name`.
fn multipliers(name: &str, n: usize, seed: u64) -> Vec<u64> {
    let mut r = rng(seed ^ n as u64);
    (0..n as u64)
        .map(|i| match name {
            "drawn" => r.next_u64(),
            "tags that agree" => (i + 1) << 32,
            "one chain" => 0,
            "chains that wrap" => u64::MAX,
            _ => unreachable!("{name}"),
        })
        .collect()
}

/// A table on `n` variables of format `M` whose hash multipliers are
/// `multipliers`.
fn table<M: Format>(n: usize, multipliers: Vec<u64>) -> Table<M> {
    Table::new(M::new(n), &mut Given(multipliers.into_iter()))
}

pub(crate) fn exponents<M: Format>(t: &Table<M>, id: MonoId) -> Vec<u32> {
    (0..t.format().variables())
        .map(|i| t.exponent(id, i))
        .collect()
}

/// Exponents of n variables, each drawn in [0, top], their sum at most
/// `cap`.
fn draw(r: &mut Xoshiro256PlusPlus, n: usize, top: u32, cap: u32) -> Vec<u32> {
    let mut left = cap;
    (0..n)
        .map(|_| {
            let e = (below(r, u64::from(top) + 1) as u32).min(left);
            left -= e;
            e
        })
        .collect()
}

/// The slot array's capacity at the start, `table.rs`'s `SLOTS`: the
/// array doubles where a monomial makes the count pass half of it.
const SLOTS: usize = 1 << 10;

/// Whether the table's slot array doubled between `before` and `after`
/// monomials with a monomial stored after the doubling: some count
/// SLOTS·2^k/2 + 1 lies in (before, after).
fn doubled_inside(before: usize, after: usize) -> bool {
    let mut half = SLOTS / 2;
    while half < after {
        if before < half + 1 && half + 1 < after {
            return true;
        }
        half *= 2;
    }
    false
}

/// Twin tables, the same 600 drawn monomials interned in each, fewer
/// distinct ones on one or two variables, then 150 rows of
/// 1 to 40 distinct monomials and a multiplier, all drawn from the
/// interned ones: on one twin by `row_products`, on the other by
/// `Table::mul` per term with the stamp tested as `matrix::multiply`
/// tested it. A third of the rows start a round of their own. Each row's
/// columns, monomials new to the round, stamps, and the table's length
/// agree; at the end every id names one monomial in both twins.
fn agree<M: Format>(n: usize, name: &str, seed: u64) {
    let case = format!(
        "{words} words of degree bound {max_degree} on {n} variables, {name}",
        words = M::new(n).words(),
        max_degree = M::MAX_DEGREE
    );
    let mut a = table::<M>(n, multipliers(name, n, seed));
    let mut b = table::<M>(n, multipliers(name, n, seed));
    let cap = M::MAX_DEGREE / 2;
    let top = (cap / n as u32).clamp(1, 40);
    let mut r = rng(seed ^ 0xb11d);
    let mut ids = Vec::new();
    for _ in 0..600 {
        let e = draw(&mut r, n, top, cap);
        let x = a.intern(&e).unwrap();
        assert_eq!(b.intern(&e), Ok(x), "{case}: the twins intern alike");
        if !ids.contains(&x) {
            ids.push(x);
        }
    }
    let (mut round, mut inside) = (1, 0);
    // Columns and new monomials of rows before, as a matrix holds them.
    let (mut cols, mut fresh) = (vec![MonoId::at(7), MonoId::at(3)], vec![MonoId::at(9)]);
    let (mut want_cols, mut want_fresh) = (cols.clone(), fresh.clone());
    for row in 0..150 {
        if below(&mut r, 3) == 0 {
            round += 1;
        }
        let t = ids[below(&mut r, ids.len() as u64) as usize];
        let len = 1 + below(&mut r, 40) as usize;
        let mut ms: Vec<MonoId> = Vec::new();
        while ms.len() < len.min(ids.len()) {
            let u = ids[below(&mut r, ids.len() as u64) as usize];
            if !ms.contains(&u) {
                ms.push(u);
            }
        }
        let before = a.len();
        // SAFETY: t and every u are ids both twins returned by `intern`.
        unsafe { a.row_products(t, &ms, round, &mut cols, &mut fresh) }
            .unwrap_or_else(|e| panic!("{case}, row {row}: {e:?}"));
        for &u in &ms {
            let p = b.mul(t, u).unwrap();
            want_cols.push(p);
            if b.stamp(p) != round {
                b.set_stamp(p, round);
                want_fresh.push(p);
            }
        }
        let name = format!("{case}, row {row} in round {round}");
        assert_eq!(cols, want_cols, "{name}: the columns");
        assert_eq!(fresh, want_fresh, "{name}: the monomials new to the round");
        assert_eq!(a.len(), b.len(), "{name}: the monomials stored");
        for &p in &cols[cols.len() - ms.len()..] {
            assert_eq!(a.stamp(p), round, "{name}: the stamp of {p}");
            assert_eq!(b.stamp(p), round, "{name}: the twin's stamp of {p}");
        }
        inside += u32::from(doubled_inside(before, a.len()));
    }
    // One variable holds too few monomials to grow the slot array.
    if n >= 2 {
        assert!(
            inside > 0,
            "{case}: no row grew the slot array before its last product"
        );
    }
    for id in (0..a.len()).map(MonoId::at) {
        assert_eq!(
            exponents(&a, id),
            exponents(&b, id),
            "{case}: id {id} names one monomial"
        );
        assert_eq!(a.stamp(id), b.stamp(id), "{case}: the stamp of {id}");
    }
}

fn agree_in<M: Format>(counts: &[usize], seed: u64) {
    for &n in counts {
        for name in HASHES {
            agree::<M>(n, name, seed ^ n as u64);
        }
    }
}

/// matrix-build-time.1 to .6, test 1: a row's products by `row_products`
/// are those of the lookup per term, for formats of one to six words, the
/// copies for one to four and the per-term lookup past four, whatever the
/// hashes, across the slot array's doublings inside a row.
#[test]
fn a_rows_products_agree_with_the_lookup_per_term() {
    // Dense8: 8 lanes a word.
    agree_in::<Dense8>(&[1, 8, 9, 24, 32, 40, 48], 0x0b11);
    // Dense16: 4 lanes a word.
    agree_in::<Dense16>(&[1, 4, 5, 12, 16, 20], 0x0b12);
    // Dense32: 2 lanes a word.
    agree_in::<Dense32>(&[1, 2, 3, 6, 8, 10], 0x0b13);
}

/// A row whose products reach the degree bound and pass it at the third
/// term: Overflow, the columns and new monomials of the first two terms
/// kept, and the table holding what the twin holds after them.
fn at_the_bound<M: Format>(n: usize) {
    let max = M::MAX_DEGREE;
    let case = format!(
        "{words} words of degree bound {max} on {n} variables",
        words = M::new(n).words()
    );
    let mut a = table::<M>(n, multipliers("drawn", n, 3));
    let mut b = table::<M>(n, multipliers("drawn", n, 3));
    let unit = |i: usize, e: u32| {
        let mut v = vec![0u32; n];
        v[i] = e;
        v
    };
    // x1^2·x2 of degree 3, or x1^2 on one variable.
    let mut two = vec![0u32; n];
    two[0] = 2;
    if n >= 2 {
        two[1] = 1;
    }
    let monomials = [
        unit(n - 1, max - 3),
        unit(0, 3),
        two,
        unit(0, 4),
        unit(0, 1),
    ];
    let mut ids = Vec::new();
    for e in &monomials {
        let x = a.intern(e).unwrap();
        assert_eq!(b.intern(e), Ok(x), "{case}");
        ids.push(x);
    }
    let t = ids[0];
    let row = &ids[1..];
    let (mut cols, mut fresh) = (vec![MonoId::at(1)], Vec::new());
    // SAFETY: t and the row's ids are ids `intern` returned.
    let got = unsafe { a.row_products(t, row, 1, &mut cols, &mut fresh) };
    assert_eq!(
        got,
        Err(Limit::Overflow),
        "{case}: the third product passes {max}"
    );
    let p = b.mul(t, row[0]).unwrap();
    let q = b.mul(t, row[1]).unwrap();
    assert_eq!(exponents(&b, p).iter().sum::<u32>(), max, "{case}");
    assert_eq!(b.mul(t, row[2]), Err(Limit::Overflow), "{case}");
    assert_eq!(
        cols,
        vec![MonoId::at(1), p, q],
        "{case}: the first two terms' columns"
    );
    assert_eq!(fresh, vec![p, q], "{case}: the first two terms are new");
    assert_eq!(a.len(), b.len(), "{case}: nothing stored past the bound");
}

/// matrix-build-time.1, .3, test 2: the degree bound of each format inside
/// a row, in every copy and in the per-term lookup.
#[test]
fn a_row_at_the_degree_bound() {
    for n in [1, 8, 9, 17, 25, 33, 41] {
        at_the_bound::<Dense8>(n);
    }
    for n in [1, 4, 5, 9, 13, 17] {
        at_the_bound::<Dense16>(n);
    }
    for n in [1, 2, 3, 5, 7, 9] {
        at_the_bound::<Dense32>(n);
    }
}

// --- the reducer searches ---

/// A monomial of n variables of degree at most `degree`, with exponents up
/// to `degree`.
fn monomial(r: &mut Xoshiro256PlusPlus, n: usize, degree: u64) -> Vec<u32> {
    let mut left = below(r, degree + 1);
    let mut e = vec![0u32; n];
    for _ in 0..n {
        let i = below(r, n as u64) as usize;
        let x = below(r, left + 1);
        e[i] += x as u32;
        left -= x;
    }
    e
}

pub(crate) fn divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

/// Grevlex on exponent vectors of any sign: the degree first, then the last
/// variable where they differ, the smaller exponent the larger monomial.
fn grevlex(a: &[i64], b: &[i64]) -> Ordering {
    let (da, db): (i64, i64) = (a.iter().sum(), b.iter().sum());
    da.cmp(&db).then_with(|| {
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return b[i].cmp(&a[i]);
            }
        }
        Ordering::Equal
    })
}

/// A monic polynomial of `terms` terms whose leading monomial has exponents
/// `lead`, its other terms of lower degree. `lead` has degree 2 or more,
/// and n is 3 or more, so that the n + 1 monomials of degree at most 1 lie
/// below it.
pub(crate) fn poly<M: Format>(
    r: &mut Xoshiro256PlusPlus,
    t: &mut Table<M>,
    field: &Gf31,
    lead: &[u32],
    terms: usize,
) -> Poly<Gf31, Rank1> {
    let n = lead.len();
    let mut monomials = vec![t.intern(lead).unwrap()];
    while monomials.len() < terms {
        let mut e = vec![0u32; n];
        let i = below(r, n as u64 + 1) as usize;
        if i < n {
            e[i] = 1;
        }
        let m = t.intern(&e).unwrap();
        if !monomials.contains(&m) {
            monomials.push(m);
        }
    }
    monomials[1..].sort_by(|&x, &y| t.cmp(&Grevlex, y, x));
    let coefs = (0..terms)
        .map(|k| field.coef(if k == 0 { 1 } else { 2 + k as u64 }))
        .collect();
    Poly {
        monomials,
        coefs,
        position: (),
    }
}

/// A lead of degree 2 to 12 on n variables; one in three has 4 to 6 added
/// to one variable's exponent, which a mask of 3 bits per variable
/// saturates.
pub(crate) fn lead(r: &mut Xoshiro256PlusPlus, n: usize) -> Vec<u32> {
    loop {
        let mut e = monomial(r, n, 6);
        if below(r, 3) == 0 {
            e[below(r, n as u64) as usize] += 4 + below(r, 3) as u32;
        }
        if e.iter().sum::<u32>() >= 2 {
            return e;
        }
    }
}

/// The queries of a search on n variables: every monomial of degree at
/// most 6 on 3 variables, 400 drawn ones of degree at most 12 on more.
pub(crate) fn queries(r: &mut Xoshiro256PlusPlus, n: usize) -> Vec<Vec<u32>> {
    if n == 3 {
        let mut all = Vec::new();
        for a in 0..=6u32 {
            for b in 0..=6 - a {
                for c in 0..=6 - a - b {
                    all.push(vec![a, b, c]);
                }
            }
        }
        return all;
    }
    (0..400)
        .map(|_| {
            let mut e = monomial(r, n, 9);
            if below(r, 2) == 0 {
                e[below(r, n as u64) as usize] += 3;
            }
            e
        })
        .collect()
}

/// The sizes of plan.md, _Edge cases_: empty, below a block of eight, at
/// it, one past it, and more.
pub(crate) const SIZES: [usize; 9] = [0, 1, 7, 8, 9, 15, 16, 17, 40];

/// matrix-build-time.7, test 3: `Signatures::reducer` returns the first
/// element of the reducer order, (sig-lead ratio, index, terms, element),
/// whose leading monomial divides the query, as a plain scan finds it.
pub(crate) fn reducer_agrees<M: Format>(n: usize, seed: u64) {
    let field = Gf31::new(65521).unwrap();
    let (mut passed_alone, mut later) = (0, 0);
    for count in SIZES {
        let case = format!(
            "{words} words on {n} variables, {count} elements",
            words = M::new(n).words()
        );
        let mut r = rng(seed ^ (count as u64) << 8);
        let mut t: Table<M> = Table::new(M::new(n), &mut rng(seed ^ 0x5157));
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        let mut sigs = Signatures::new(2);
        // Per element: its ratio, index, terms, and lead's exponents.
        let mut keys: Vec<(Vec<i64>, u32, usize, Vec<u32>)> = Vec::new();
        for _ in 0..count {
            let l = lead(&mut r, n);
            let terms = 1 + below(&mut r, 3) as usize;
            let g = poly(&mut r, &mut t, &field, &l, terms);
            let s = monomial(&mut r, n, 8);
            let sig = Sig {
                monomial: t.intern(&s).unwrap(),
                index: below(&mut r, 2) as u32,
            };
            basis.push(g, false).unwrap();
            sigs.push(&t, &Grevlex, &basis, sig).unwrap();
            let ratio: Vec<i64> = s
                .iter()
                .zip(&l)
                .map(|(&x, &y)| i64::from(x) - i64::from(y))
                .collect();
            keys.push((ratio, sig.index, terms, l));
        }
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&x, &y| {
            let (kx, ky) = (&keys[x], &keys[y]);
            grevlex(&kx.0, &ky.0)
                .then(kx.1.cmp(&ky.1))
                .then(kx.2.cmp(&ky.2))
                .then(x.cmp(&y))
        });
        for q in queries(&mut r, n) {
            let m = t.intern(&q).unwrap();
            let want = order.iter().copied().find(|&e| divides(&keys[e].3, &q));
            let got = sigs.reducer(&t, m).map(ElementId::index);
            assert_eq!(got, want, "{case}: {q:?}");
            let not_m = !t.mask(m);
            let pass = |e: usize| t.mask(basis.lead(ElementId::at(e))) & not_m == 0;
            let first = order.iter().copied().find(|&e| pass(e));
            passed_alone += u32::from(first.is_some() && want.is_none());
            later += u32::from(first.is_some() && want.is_some() && first != want);
        }
    }
    // On 3 variables a mask holds 10 bits a variable and decides every
    // query here; on 9 it holds 3, and exponents past 3 pass it.
    if n >= 9 {
        assert!(
            passed_alone > 0,
            "{n} variables: no mask passed without a divisor"
        );
        assert!(
            later > 0,
            "{n} variables: no divisor after a candidate that fails"
        );
    }
}

#[test]
fn the_reducer_search_finds_the_first_divisor() {
    // Dense16: one word on 3 variables, three on 9.
    reducer_agrees::<Dense16>(3, 0x0b14);
    reducer_agrees::<Dense16>(9, 0x0b15);
    reducer_agrees::<Dense8>(9, 0x0b16);
}

/// matrix-build-time.8, test 4: `Divisors::find` returns the element of
/// `G_min` of fewest terms whose leading monomial divides the query, ties to
/// the element made first, as a plain scan finds it.
fn divisors_agree<M: Format>(n: usize, seed: u64) {
    let field = Gf31::new(65521).unwrap();
    let mut ties = 0;
    for count in SIZES {
        let case = format!(
            "{words} words on {n} variables, {count} elements",
            words = M::new(n).words()
        );
        let mut r = rng(seed ^ (count as u64) << 8);
        let mut t: Table<M> = Table::new(M::new(n), &mut rng(seed ^ 0xd17));
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        let mut elements: Vec<(Vec<u32>, usize)> = Vec::new();
        for _ in 0..count {
            let l = lead(&mut r, n);
            // Two lengths alone, so that divisors of equal terms tie, some in
            // different groups of eight in the order made.
            let terms = 1 + 2 * below(&mut r, 2) as usize;
            basis
                .push(poly(&mut r, &mut t, &field, &l, terms), false)
                .unwrap();
            elements.push((l, terms));
        }
        let divisors = Divisors::new(&t, &basis);
        for q in queries(&mut r, n) {
            let u = t.intern(&q).unwrap();
            let mut want: Option<usize> = None;
            for (e, (l, terms)) in elements.iter().enumerate() {
                if divides(l, &q) && want.is_none_or(|b| elements[b].1 > *terms) {
                    want = Some(e);
                }
            }
            let rivals = elements
                .iter()
                .enumerate()
                .filter(|(e, (l, terms))| {
                    divides(l, &q)
                        && want.is_some_and(|b| b / 8 != e / 8 && *terms == elements[b].1)
                })
                .count();
            ties += u32::from(rivals > 0);
            let got = divisors.find(&t, u).map(ElementId::index);
            assert_eq!(got, want, "{case}: {q:?}");
        }
    }
    assert!(ties > 0, "{n} variables: no tie across groups of eight");
}

#[test]
fn the_divisor_search_finds_the_fewest_terms_first() {
    divisors_agree::<Dense16>(3, 0x0b17);
    divisors_agree::<Dense16>(9, 0x0b18);
    divisors_agree::<Dense8>(9, 0x0b19);
}
