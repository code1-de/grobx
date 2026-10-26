//! product-lookup-time, the table's test code: the product lookup through
//! both doors, `Table::mul` and `Table::mul_unchecked`, agrees with
//! interning whatever the hashes, at the degree bound, and refuses an id
//! past the table (plan.md, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::HashSet;
use std::convert::Infallible;
use std::panic::{AssertUnwindSafe, catch_unwind};

use rand_core::{Rng, SeedableRng, TryRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::id::MonoId;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::table::{Limit, Table};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
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

/// The hashes of plan.md, _Edge cases_, by name.
const HASHES: [&str; 4] = [
    "drawn",
    // Every hash's low 32 bits are 0: every slot's tag agrees.
    "tags that agree",
    // Every hash is 0: one probe chain from slot 0.
    "one chain",
    // Every hash is 2^64 minus the degree, the unit monomial's 0: every
    // other home slot is the last, and every chain from there wraps past
    // the end.
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

fn exponents<M: Format>(t: &Table<M>, id: MonoId) -> Vec<u32> {
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
            let e = ((r.next_u64() % (u64::from(top) + 1)) as u32).min(left);
            left -= e;
            e
        })
        .collect()
}

/// One table: 600 drawn monomials interned, then 900 products of drawn
/// pairs, each looked up through one door and its commuted product through
/// the other.
fn agree<M: Format>(n: usize, mut t: Table<M>, seed: u64, case: &str) {
    let cap = M::MAX_DEGREE / 2;
    let top = (cap / n as u32).clamp(1, 40);
    let mut r = rng(seed);
    let mut interned: HashSet<Vec<u32>> = HashSet::new();
    let mut ids = Vec::new();
    for _ in 0..600 {
        let e = draw(&mut r, n, top, cap);
        let id = t.intern(&e).unwrap();
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
                t.mul(x, y).unwrap(),
                unsafe { t.mul_unchecked(y, x) }.unwrap(),
            )
        } else {
            (
                unsafe { t.mul_unchecked(x, y) }.unwrap(),
                t.mul(y, x).unwrap(),
            )
        };
        assert_eq!(p, q, "{name}: the two doors");
        assert_eq!(exponents(&t, p), sum, "{name}");
        interned.insert(sum.clone());
        assert_eq!(t.intern(&sum), Ok(p), "{name}: interning");
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
    let x = t.intern(&[max - 3, 0]).unwrap();
    let y = t.intern(&[0, 3]).unwrap();
    let z = t.intern(&[0, 4]).unwrap();
    let p = t.mul(x, y).unwrap();
    assert_eq!(exponents(&t, p), [max - 3, 3], "{max}");
    // SAFETY: x, y, and z are ids `intern` returned from this table.
    assert_eq!(unsafe { t.mul_unchecked(y, x) }, Ok(p), "{max}");
    let len = t.len();
    assert_eq!(t.mul(x, z), Err(Limit::Overflow), "{max}");
    // SAFETY: as above.
    assert_eq!(
        unsafe { t.mul_unchecked(z, x) },
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
    let x = t.intern(&[1, 0]).unwrap();
    let y = t.intern(&[0, 1]).unwrap();
    let len = t.len();
    let past = MonoId::at(len);
    for (a, b) in [(past, x), (y, past), (MonoId::NONE, x)] {
        let r = catch_unwind(AssertUnwindSafe(|| t.mul(a, b)));
        assert!(r.is_err(), "ids {a} and {b} past {len}");
        assert_eq!(t.len(), len, "ids {a} and {b}: nothing stored");
    }
}
