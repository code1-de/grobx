//! q-reconstruct-time, sub-item 2: the lift's basis on the case where
//! Wang's reconstruction keeps a wrong rational past a later coefficient
//! that fails, which the chain's MQRR does not, and of a
//! polynomial whose coefficients grow to 3000 bits (plan.md, _Test code_).

use std::fmt::Write;
use std::num::NonZeroUsize;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lift::{Report, lift_text};
use crate::rational::{Rational, bound, reconstruct};
use crate::{MonomialOrder, Outcome, Params, Threads};

/// The learn prime of the forced cases, lucky for them.
const L1: u64 = 2_147_483_647;

fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

/// The lift of `text` with the primes `forced` first: its outcome, which
/// must hold a basis.
fn lift(text: &str, params: &Params, forced: &[u64]) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text(text, params, forced.to_vec(), &mut report)
        .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    outcome
}

/// A monic polynomial in x of 40 terms over Q whose coefficients grow from
/// 1 to 3000 bits, the first term past the leading one the smallest: its
/// text and its canonical basis text.
fn growing() -> (String, String) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9e0);
    let mut poly = String::from("x^39");
    for k in 1..=39u64 {
        let bits = 1 + 2999 * (k - 1) / 38;
        let mut c = BigUint::zero();
        for _ in 0..bits.div_ceil(64) {
            c = (c << 64u32) + BigUint::from(rng.next_u64());
        }
        c >>= bits.div_ceil(64) * 64 - bits;
        c |= BigUint::one() << (bits - 1);
        poly.push(if k % 2 == 0 { '+' } else { '-' });
        let e = 39 - k;
        if !c.is_one() || e == 0 {
            poly.push_str(&c.to_string());
            if e > 0 {
                poly.push('*');
            }
        }
        match e {
            0 => {}
            1 => poly.push('x'),
            _ => write!(poly, "x^{e}").unwrap(),
        }
    }
    (format!("x\n0\n{poly}\n"), format!("[{poly}]:\n"))
}

/// The polynomial of `growing`: the lift returns it as its basis.
#[test]
fn a_polynomial_of_growing_coefficients_lifts() {
    let (text, basis) = growing();
    let outcome = lift(&text, &params(7), &[]);
    assert_eq!(outcome.basis.as_deref(), Some(basis.as_str()));
}

/// The primes from 101 below `limit`, by trial division.
fn small_primes(limit: u64) -> Vec<u64> {
    (101..limit)
        .filter(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0))
        .collect()
}

/// a mod m, in [0, m).
fn image(a: i64, m: u64) -> u64 {
    a.rem_euclid(m as i64) as u64
}

/// Wang's reconstruction of a mod m.
fn wang(a: i64, m: u64) -> Option<Rational> {
    let big = BigUint::from(m);
    reconstruct(&BigUint::from(image(a, m)), &big, &bound(&big))
}

/// q-lift.40's search: an integer c and primes p₁ ≠ p₂ below 1200, neither
/// dividing c, such that Wang's reconstruction of −c mod p₁ is a rational
/// r ≠ −c with r ≡ −c mod p₂. Then an integer b, |b| ≤ bound(p₁p₂), whose
/// reconstruction mod p₁ is None or does not fit p₂, and a prime p₃ below
/// 1200 at which r does not fit, none of p₁, p₂, p₃ dividing b, and p₃ not
/// dividing c. Returns (c, b, p₁, p₂, p₃).
fn completion_case() -> (i64, i64, u64, u64, u64) {
    let primes = small_primes(1200);
    for c in 1_000_003i64.. {
        for &p1 in &primes {
            if (c as u64).is_multiple_of(p1) {
                continue;
            }
            let Some(r) = wang(-c, p1) else {
                continue;
            };
            if r == Rational::new(BigInt::from(-c), BigInt::from(1)).unwrap() {
                continue;
            }
            for &p2 in &primes {
                if p2 == p1 || (c as u64).is_multiple_of(p2) || r.residue(p2) != Some(image(-c, p2))
                {
                    continue;
                }
                let n = bound(&BigUint::from(p1 * p2)).to_i64().unwrap();
                let b = (2..=n).rev().find(|&b| {
                    [p1, p2].iter().all(|&p| !(b as u64).is_multiple_of(p))
                        && wang(b, p1).is_none_or(|s| s.residue(p2) != Some(image(b, p2)))
                });
                let Some(b) = b else {
                    continue;
                };
                let p3 = primes.iter().copied().find(|&p3| {
                    p3 != p1
                        && p3 != p2
                        && !(b as u64).is_multiple_of(p3)
                        && !(c as u64).is_multiple_of(p3)
                        && r.residue(p3) != Some(image(-c, p3))
                });
                if let Some(p3) = p3 {
                    return (c, b, p1, p2, p3);
                }
            }
        }
    }
    unreachable!()
}

/// The case of q-reconstruct-time.9: x² − c·x + b with p₁, p₂, and p₃
/// forced after L1, where at p₂ Wang's reconstruction keeps a wrong
/// rational for −c and fails at b. The chain's MQRR takes no rational from
/// a modulus below 2^11. The lift returns the polynomial as its basis.
#[test]
fn an_early_rational_is_cleared_at_completion() {
    let (c, b, p1, p2, p3) = completion_case();
    let text = format!("x\n0\nx^2-{c}*x+{b}\n");
    let outcome = lift(&text, &params(1), &[L1, p1, p2, p3]);
    assert_eq!(
        outcome.basis.as_deref(),
        Some(format!("[x^2-{c}*x+{b}]:\n").as_str()),
        "c = {c}, b = {b}, {p1}, {p2}, {p3}"
    );
}
