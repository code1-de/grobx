//! product-lookup-time, the table's test code: the product lookup through
//! both doors, `Table::mul` and `Table::mul_unchecked`, agrees with
//! interning whatever the hashes, at the degree bound, and refuses an id
//! past the table (plan.md, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use rand_core::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::id::MonoId;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::shared_tests::{HASHES, exponents, multipliers, rng, table};
use crate::table::{Limit, Table};

/// Exponents of n variables, each drawn in [0, top], their sum at most
/// `cap`.
pub(crate) fn draw(r: &mut Xoshiro256PlusPlus, n: usize, top: u32, cap: u32) -> Vec<u32> {
    let mut left = cap;
    (0..n)
        .map(|_| {
            let e = ((r.next_u64() % (u64::from(top) + 1)) as u32).min(left);
            left -= e;
            e
        })
        .collect()
}

/// `agree_by` on the monomials `draw` draws on n variables, each exponent
/// at most `top` and their sum at most half the degree bound.
pub(crate) fn agree<M: Format>(n: usize, t: Table<M>, seed: u64, case: &str) {
    let cap = M::MAX_DEGREE / 2;
    let top = (cap / n as u32).clamp(1, 40);
    agree_by(t, seed, case, |r| draw(r, n, top, cap));
}

/// One table: 600 monomials `next` draws interned, then 900 products of
/// drawn pairs, each looked up through one door and its commuted product
/// through the other. Each product's exponents are the factors' sum, and
/// interning the sum returns its id; at the end every id names its
/// monomial, one id per monomial. `next` draws exponents whose pairwise
/// sums the format holds.
pub(crate) fn agree_by<M: Format>(
    mut t: Table<M>,
    seed: u64,
    case: &str,
    mut next: impl FnMut(&mut Xoshiro256PlusPlus) -> Vec<u32>,
) {
    let mut r = rng(seed);
    let mut interned: HashSet<Vec<u32>> = HashSet::new();
    let mut ids = Vec::new();
    for _ in 0..600 {
        let e = next(&mut r);
        let id = t.intern(&e, &mut Budget::unbounded()).unwrap();
        interned.insert(e.clone());
        ids.push((e, id));
    }
    for k in 0..900 {
        let (a, x) = ids[(r.next_u64() % ids.len() as u64) as usize].clone();
        let (b, y) = ids[(r.next_u64() % ids.len() as u64) as usize].clone();
        let sum: Vec<u32> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        let name = format!("{case}, product {k}: {a:?} {b:?}");
        // SAFETY: x and y are ids `intern` returned from this table.
        let (p, q) = if k % 2 == 0 {
            (
                t.mul(x, y, &mut Budget::unbounded()).unwrap(),
                unsafe { t.mul_unchecked(y, x, &mut Budget::unbounded()) }.unwrap(),
            )
        } else {
            (
                unsafe { t.mul_unchecked(x, y, &mut Budget::unbounded()) }.unwrap(),
                t.mul(y, x, &mut Budget::unbounded()).unwrap(),
            )
        };
        assert_eq!(p, q, "{name}: the two doors");
        assert_eq!(exponents(&t, p), sum, "{name}");
        interned.insert(sum.clone());
        assert_eq!(
            t.intern(&sum, &mut Budget::unbounded()),
            Ok(p),
            "{name}: interning"
        );
    }
    assert_eq!(t.len(), interned.len(), "{case}: one id per monomial");
    for (e, id) in &ids {
        assert_eq!(
            &exponents(&t, *id),
            e,
            "{case}: an id still names its monomial"
        );
    }
}

fn agree_in<M: Format>(counts: &[usize], seed: u64) {
    for &n in counts {
        for name in HASHES {
            let case = format!(
                "degree bound {max_degree} on {n} variables, {name}",
                max_degree = M::MAX_DEGREE
            );
            let t = table::<M>(n, multipliers(name, n, seed));
            agree::<M>(n, t, seed ^ n as u64, &case);
        }
    }
}

/// product-lookup-time.1, .5, .6: products through both doors return the
/// id that interning returns, for hashes drawn, tags that all agree, one
/// probe chain, and chains that wrap; across the slot array's doublings;
/// in monomials of one word and of several.
#[test]
fn products_agree_with_interning_whatever_the_hash() {
    agree_in::<Dense8>(&[1, 3, 8, 9, 17], 0x100c);
    agree_in::<Dense16>(&[3, 5], 0x100d);
    agree_in::<Dense32>(&[3], 0x100e);
}

/// A product at the degree bound, through both doors, and one past it,
/// which stores nothing.
fn at_the_bound<M: Format>() {
    let max = M::MAX_DEGREE;
    let mut t = table::<M>(2, multipliers("drawn", 2, 3));
    let x = t.intern(&[max - 3, 0], &mut Budget::unbounded()).unwrap();
    let y = t.intern(&[0, 3], &mut Budget::unbounded()).unwrap();
    let z = t.intern(&[0, 4], &mut Budget::unbounded()).unwrap();
    let p = t.mul(x, y, &mut Budget::unbounded()).unwrap();
    assert_eq!(exponents(&t, p), [max - 3, 3], "{max}");
    // SAFETY: x, y, and z are ids `intern` returned from this table.
    assert_eq!(
        unsafe { t.mul_unchecked(y, x, &mut Budget::unbounded()) },
        Ok(p),
        "{max}"
    );
    let len = t.len();
    assert_eq!(
        t.mul(x, z, &mut Budget::unbounded()),
        Err(Limit::Overflow),
        "{max}"
    );
    // SAFETY: as above.
    assert_eq!(
        unsafe { t.mul_unchecked(z, x, &mut Budget::unbounded()) },
        Err(Limit::Overflow),
        "{max}"
    );
    assert_eq!(t.len(), len, "{max}: an overflow stores nothing");
}

/// The degree bound of each format, 127, 32767, and 2^31 − 1.
#[test]
fn products_at_the_degree_bound() {
    at_the_bound::<Dense8>();
    at_the_bound::<Dense16>();
    at_the_bound::<Dense32>();
}

/// product-lookup-time.4: the safe door panics where an id lies at or past
/// the table's length, and stores nothing.
#[test]
fn an_id_past_the_table_panics() {
    let mut t = table::<Dense8>(2, vec![5, 7]);
    let x = t.intern(&[1, 0], &mut Budget::unbounded()).unwrap();
    let y = t.intern(&[0, 1], &mut Budget::unbounded()).unwrap();
    let len = t.len();
    let past = MonoId::at(len);
    for (a, b) in [(past, x), (y, past), (MonoId::NONE, x)] {
        let r = catch_unwind(AssertUnwindSafe(|| t.mul(a, b, &mut Budget::unbounded())));
        assert!(r.is_err(), "ids {a} and {b} past {len}");
        assert_eq!(t.len(), len, "ids {a} and {b}: nothing stored");
    }
}
