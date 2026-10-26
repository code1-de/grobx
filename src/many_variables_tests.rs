//! many-variables-time, the crate's test code: the divisor masks past 32
//! variables against masks built from the exponents, coprimality against
//! the exponents, and the reducer searches and the pair update on more
//! than 32 variables against their references
//! (`iterations/0070-many-variables-time/plan.md`, _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::build_tests::reducer_agrees;
use crate::divisibility_tests::{check_masks_on, check_updates_on};
use crate::field::{Field, Gf31};
use crate::id::ElementId;
use crate::matrix::Divisors;
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::{Grevlex, Lex};
use crate::poly::{Basis, Poly};
use crate::table::Table;

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn below(rng: &mut Xoshiro256PlusPlus, n: u64) -> u64 {
    rng.next_u64() % n
}

/// Test 1, many-variables-time.1, .2: the masks on 71 to 140 variables,
/// in every format, are `mask_of`'s, and a divisor's mask lies in its
/// multiple's. `divisibility_tests.rs` test 1 covers 0 to 70.
#[test]
fn masks_past_32_variables() {
    check_masks_on::<Dense8>(0x3a71_0001, 71..=140, 300);
    check_masks_on::<Dense16>(0x3a71_0002, 71..=140, 300);
    check_masks_on::<Dense32>(0x3a71_0003, 71..=140, 300);
}

/// Whether no variable is positive in both.
fn coprime_by_exponents(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(&x, &y)| x == 0 || y == 0)
}

/// The exponents of one to four variables of n, each 1 to 3, the others 0.
/// Half the draws take their variables from those of residues 0 to 3 mod
/// 32, which share the masks' first four bits past 32 variables.
fn sparse(rng: &mut Xoshiro256PlusPlus, n: usize) -> Vec<u32> {
    let mut e = vec![0u32; n];
    let aligned = below(rng, 2) == 0;
    for _ in 0..=below(rng, 4) {
        let i = if aligned {
            let r = below(rng, 4.min(n as u64)) as usize;
            let k = below(rng, (n - r).div_ceil(32) as u64) as usize;
            r + 32 * k
        } else {
            below(rng, n as u64) as usize
        };
        e[i] = 1 + below(rng, 3) as u32;
    }
    e
}

/// Test 2 for one format: `Table::coprime` and `Format::coprime` against
/// the exponents, on random monomials and on monomials built of variables
/// 32 apart.
fn check_coprime<M: Format>(seed: u64) {
    let mut r = rng(seed);
    for n in [31usize, 32, 33, 40, 63, 64, 65, 100, 140] {
        let mut table: Table<M> = Table::new(M::new(n), &mut rng(seed ^ n as u64));
        let agree = |table: &mut Table<M>, a: &[u32], b: &[u32]| {
            let (ia, ib) = (table.intern(a).unwrap(), table.intern(b).unwrap());
            let want = coprime_by_exponents(a, b);
            assert_eq!(table.coprime(ia, ib), want, "n {n}: {a:?}, {b:?}");
            assert_eq!(
                table
                    .format()
                    .coprime(table.monomial(ia), table.monomial(ib)),
                want,
                "n {n}, the format: {a:?}, {b:?}"
            );
        };
        for _ in 0..400 {
            let (a, b) = (sparse(&mut r, n), sparse(&mut r, n));
            agree(&mut table, &a, &b);
        }
        let single = |i: usize, e: u32| {
            let mut v = vec![0u32; n];
            v[i] = e;
            v
        };
        for i in 0..n.saturating_sub(32) {
            let a = single(i, 1 + (i % 3) as u32);
            let b = single(i + 32, 2);
            // x_i^a and x_{i+32}^2: coprime, their masks one bit past 32.
            agree(&mut table, &a, &b);
            // x_i^a·x_{i+32} and x_{i+32}^2: not coprime.
            let mut c = a.clone();
            c[i + 32] = 1;
            agree(&mut table, &c, &b);
        }
        if n >= 64 {
            // x1⋯x32 and x33⋯x64: every bit in both masks, and coprime.
            let a: Vec<u32> = (0..n).map(|i| u32::from(i < 32)).collect();
            let b: Vec<u32> = (0..n).map(|i| u32::from((32..64).contains(&i))).collect();
            agree(&mut table, &a, &b);
        }
    }
}

/// Test 2, many-variables-time.4: coprimality past 32 variables, and on
/// 31 and 32, in every format.
#[test]
fn coprimality_past_32_variables() {
    check_coprime::<Dense8>(0x3a71_0004);
    check_coprime::<Dense16>(0x3a71_0005);
    check_coprime::<Dense32>(0x3a71_0006);
}

/// A leading monomial of degree 2 to 12 on n variables: one to four of
/// them, each of exponent 1 to 3, of residues 0 to 3 mod 32 in half the
/// draws.
fn lead(rng: &mut Xoshiro256PlusPlus, n: usize) -> Vec<u32> {
    loop {
        let e = sparse(rng, n);
        if e.iter().sum::<u32>() >= 2 {
            return e;
        }
    }
}

/// Test 3 for one format and n: `Divisors::find` on 40 elements of one or
/// three terms whose leads `lead` draws, and 400 queries, half of them a
/// lead's multiple: the element of fewest terms whose leading monomial
/// divides the query, ties to the element made first, as a scan of the
/// exponents finds it.
fn check_divisors<M: Format>(n: usize, seed: u64) {
    let field = Gf31::new(65521).unwrap();
    let mut r = rng(seed);
    let mut table: Table<M> = Table::new(M::new(n), &mut rng(seed ^ 0xd17));
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    let mut elements: Vec<(Vec<u32>, usize)> = Vec::new();
    let one = table.intern(&vec![0; n]).unwrap();
    for _ in 0..40 {
        let l = lead(&mut r, n);
        let terms = 1 + 2 * below(&mut r, 2) as usize;
        let mut monomials = vec![table.intern(&l).unwrap()];
        if terms == 3 {
            // A variable of the lead alone, then 1: both below it.
            let k = l.iter().position(|&e| e > 0).unwrap();
            let mut x = vec![0u32; n];
            x[k] = 1;
            monomials.push(table.intern(&x).unwrap());
            monomials.push(one);
        }
        let coefs = vec![field.coef(1); monomials.len()];
        let poly = Poly {
            monomials,
            coefs,
            position: (),
        };
        basis.push(poly, false).unwrap();
        elements.push((l, terms));
    }
    let divisors = Divisors::new(&table, &basis);
    let mut found = 0;
    for _ in 0..400 {
        let mut q = sparse(&mut r, n);
        if below(&mut r, 2) == 0 {
            let (l, _) = &elements[below(&mut r, elements.len() as u64) as usize];
            for (x, y) in q.iter_mut().zip(l) {
                *x += y;
            }
        }
        let mut want: Option<usize> = None;
        for (e, (l, terms)) in elements.iter().enumerate() {
            let divides = l.iter().zip(&q).all(|(x, y)| x <= y);
            if divides && want.is_none_or(|b| elements[b].1 > *terms) {
                want = Some(e);
            }
        }
        found += u32::from(want.is_some());
        let u = table.intern(&q).unwrap();
        let got = divisors.find(&table, u).map(ElementId::index);
        assert_eq!(got, want, "n {n}: {q:?}");
    }
    assert!(found > 100, "n {n}: {found} of 400 queries had a divisor");
}

/// Test 3, many-variables-time.1: the signature run's reducer and the
/// plain run's divisor of fewest terms on 33 to 65 variables, as plain
/// scans of the exponents find them.
#[test]
fn reducer_searches_past_32_variables() {
    for n in [33usize, 40, 64, 65] {
        let salt = n as u64;
        reducer_agrees::<Dense8>(n, 0x3a71_0100 ^ salt);
        reducer_agrees::<Dense16>(n, 0x3a71_0200 ^ salt);
        check_divisors::<Dense8>(n, 0x3a71_0300 ^ salt);
        check_divisors::<Dense16>(n, 0x3a71_0400 ^ salt);
    }
}

/// Test 4, many-variables-time.1, .4: the pairs the update keeps on 33 to
/// 72 variables, in a graded order and in one that is not, are joux-2011's
/// Algorithm 2's.
#[test]
fn pair_updates_past_32_variables() {
    let many = |world: u64| 33 + (world % 40) as usize;
    check_updates_on(&Grevlex, 0x3a71_0007, many);
    check_updates_on(&Lex, 0x3a71_0008, many);
}
