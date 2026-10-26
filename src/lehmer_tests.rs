//! q-reconstruct-time, sub-item 1: Wang's reconstruction by Lehmer's
//! Euclid on limbs, `lehmer::Wang`, against `rational::reconstruct`, and
//! the check `lehmer::coprime` against num-bigint's `gcd` (plan.md, _Test
//! code_).

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::Gf31;
use crate::lehmer::{Wang, coprime};
use crate::rational::{Rational, bound, reconstruct};

/// The numbers of primes the moduli are products of: one prime, below
/// 128 bits, past it, and up to 100.
const COUNTS: [usize; 9] = [1, 2, 3, 4, 5, 8, 20, 44, 100];

/// k distinct random primes of [2^30, 2^31), by `Gf31::new`'s test.
fn primes(rng: &mut Xoshiro256PlusPlus, k: usize) -> Vec<u64> {
    let mut used: Vec<u64> = Vec::with_capacity(k);
    while used.len() < k {
        let mut c = (1u64 << 30) + (rng.next_u64() >> 34);
        while c < 1 << 31 && Gf31::new(c).is_none() {
            c += 1;
        }
        if c < 1 << 31 && !used.contains(&c) {
            used.push(c);
        }
    }
    used
}

fn product(primes: &[u64]) -> BigUint {
    primes.iter().fold(BigUint::one(), |m, &p| m * p)
}

/// A random integer below 2^bits.
fn integer(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let mut x = BigUint::zero();
    for _ in 0..bits.div_ceil(64) {
        x = (x << 64u32) + BigUint::from(rng.next_u64());
    }
    x >> (bits.div_ceil(64) * 64 - bits)
}

/// n·d⁻¹ mod m; None where d is no unit mod m.
fn image(n: &BigInt, d: &BigUint, m: &BigUint) -> Option<BigUint> {
    let m = BigInt::from(m.clone());
    let e = BigInt::from(d.clone()).extended_gcd(&m);
    if !e.gcd.is_one() {
        return None;
    }
    (n * e.x).mod_floor(&m).to_biguint()
}

/// `Wang` modulo m reconstructs u as `rational::reconstruct` does.
fn agree(wang: &Wang, m: &BigUint, u: &BigUint, case: &str) -> Option<Rational> {
    let expected = reconstruct(u, m, &bound(m));
    assert_eq!(
        wang.reconstruct(&u.to_u64_digits()),
        expected,
        "{case}: u = {u} modulo m = {m}"
    );
    expected
}

/// Wang over each modulus of `COUNTS`: rationals of the bound's size and
/// smaller, of both signs, by their residues, and uniform residues.
#[test]
fn agrees_with_wang_on_random_residues() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1e4e);
    for &k in &COUNTS {
        let m = product(&primes(&mut rng, k));
        let wang = Wang::new(&m);
        assert_eq!(*wang.bound(), bound(&m), "{k} primes");
        let half = bound(&m).bits();
        let inputs = (6_000 / k).clamp(40, 200);
        let (mut found, mut none) = (0, 0);
        for i in 0..inputs {
            let u = if i % 3 == 2 {
                integer(&mut rng, m.bits()) % &m
            } else {
                // Numerator and denominator of up to the bound's length,
                // some shorter.
                let (nbits, dbits) = (
                    half - rng.next_u64() % half.min(40),
                    half - rng.next_u64() % half.min(40),
                );
                let n = integer(&mut rng, nbits);
                let d = integer(&mut rng, dbits).max(BigUint::one());
                let n = if i % 2 == 0 {
                    BigInt::from(n)
                } else {
                    -BigInt::from(n)
                };
                let Some(u) = image(&n, &d, &m) else {
                    continue;
                };
                u
            };
            match agree(&wang, &m, &u, &format!("{k} primes, input {i}")) {
                Some(_) => found += 1,
                None => none += 1,
            }
        }
        // Both outcomes occur at every size.
        assert!(
            found > 0 && none > 0,
            "{k} primes: {found} found, {none} none"
        );
    }
}

/// u = 0, 1, M − 1, M − 2, N, N + 1, and n/d at the bound: |n| = N,
/// d = N, and d = N + 1.
#[test]
fn agrees_at_the_edges() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xed9e);
    for k in [1, 4, 5, 44] {
        let m = product(&primes(&mut rng, k));
        let wang = Wang::new(&m);
        let n = bound(&m);
        let one = BigUint::one();
        let case = |what: &str| format!("{k} primes, {what}");
        assert_eq!(
            agree(&wang, &m, &BigUint::zero(), &case("u = 0")),
            Some(Rational::new(BigInt::zero(), BigInt::one()).unwrap())
        );
        assert_eq!(
            agree(&wang, &m, &one, &case("u = 1")),
            Some(Rational::new(BigInt::one(), BigInt::one()).unwrap())
        );
        assert_eq!(
            agree(&wang, &m, &(&m - 1u32), &case("u = M - 1")),
            Some(Rational::new(-BigInt::one(), BigInt::one()).unwrap())
        );
        agree(&wang, &m, &(&m - 2u32), &case("u = M - 2"));
        agree(&wang, &m, &n, &case("u = N"));
        agree(&wang, &m, &(&n + 1u32), &case("u = N + 1"));
        // n/d with |n| = N or d = N, and d = N + 1, where d is a unit.
        for (num, den, what) in [
            (BigInt::from(n.clone()), &n - 1u32, "N/(N - 1)"),
            (-BigInt::from(n.clone()), &n - 2u32, "-N/(N - 2)"),
            (BigInt::from(&n - 1u32), n.clone(), "(N - 1)/N"),
            (BigInt::from(3u32), n.clone(), "3/N"),
            (BigInt::one(), &n + 1u32, "1/(N + 1)"),
            (BigInt::from(&n - 1u32), &n + 1u32, "(N - 1)/(N + 1)"),
        ] {
            if den.is_zero() {
                continue;
            }
            if let Some(u) = image(&num, &den, &m) {
                agree(&wang, &m, &u, &case(what));
            }
        }
    }
}

/// A first quotient past 2^64: N < u ≤ M/2^70, where M has room for it.
#[test]
fn agrees_where_a_quotient_passes_a_word() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9a55);
    for k in [20, 100] {
        let m = product(&primes(&mut rng, k));
        let wang = Wang::new(&m);
        let n = bound(&m);
        let top = &m >> 70u32;
        assert!(top > n, "{k} primes leave no room past N");
        for i in 0..200 {
            let u = &n + 1u32 + integer(&mut rng, top.bits()) % (&top - &n);
            agree(&wang, &m, &u, &format!("{k} primes, input {i}"));
        }
    }
}

/// Every quotient 1, or nearly: u = ⌊M·(√5 − 1)/2⌋ and its neighbours.
#[test]
fn agrees_where_every_quotient_is_one() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xf1b0);
    for k in [20, 100, 290] {
        let m = product(&primes(&mut rng, k));
        let wang = Wang::new(&m);
        // (√5 − 1)/2 = (√(5·M²) − M)/(2M), so the u is ⌊(√(5·M²) − M)/2⌋.
        let golden = ((&m * &m * 5u32).sqrt() - &m) >> 1u32;
        for delta in 0u32..20 {
            for u in [&golden + delta, &golden - delta] {
                agree(&wang, &m, &u, &format!("{k} primes, golden ± {delta}"));
            }
        }
    }
}

/// The primes from 101 below `limit`, by trial division.
fn small_primes(limit: u64) -> Vec<u64> {
    (101..limit)
        .filter(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0))
        .collect()
}

/// collins-1995 §2's case: the Euclidean algorithm stops at a d that is
/// no unit mod M, and `reconstruct` returns None (q-lift.14). Found by
/// search over M = p₁·p₂ of two primes below 1200.
#[test]
fn refuses_a_denominator_not_prime_to_m() {
    let primes = small_primes(1200);
    let mut found = 0;
    'search: for (i, &p1) in primes.iter().enumerate() {
        for &p2 in &primes[i + 1..] {
            let m = BigUint::from(p1 * p2);
            let n = bound(&m);
            let wang = Wang::new(&m);
            for u in 0..p1 * p2 {
                let u = BigUint::from(u);
                // Wang's stopping point, by the plain Euclid.
                let (mut r0, mut r1) = (m.clone(), u.clone());
                let (mut t0, mut t1) = (BigInt::zero(), BigInt::one());
                while r1 > n {
                    let (q, r) = r0.div_rem(&r1);
                    (r0, r1) = (r1, r);
                    let t = &t0 - BigInt::from(q) * &t1;
                    (t0, t1) = (t1, t);
                }
                let d = t1.magnitude().clone();
                if d.is_zero() || d > n || d.gcd(&m).is_one() {
                    continue;
                }
                assert_eq!(reconstruct(&u, &m, &n), None, "u = {u} modulo {m}");
                assert_eq!(
                    wang.reconstruct(&u.to_u64_digits()),
                    None,
                    "u = {u} modulo {m}"
                );
                found += 1;
                if found == 50 {
                    break 'search;
                }
            }
        }
    }
    assert_eq!(found, 50, "the search finds 50 cases");
}

/// Consecutive Fibonacci numbers of up to about `bits` bits.
fn fibonacci(bits: u64) -> (BigUint, BigUint) {
    let (mut a, mut b) = (BigUint::one(), BigUint::one());
    while b.bits() < bits {
        (a, b) = (b.clone(), a + b);
    }
    (b, a)
}

#[test]
fn coprime_agrees_with_gcd() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc0e1);
    let mut pairs: Vec<(BigUint, BigUint, String)> = Vec::new();
    for limbs in 1..=40u64 {
        for _ in 0..20 {
            let ybits = 64 * (1 + rng.next_u64() % limbs);
            let x = integer(&mut rng, 64 * limbs);
            let y = integer(&mut rng, ybits);
            pairs.push((x.clone(), y.clone(), format!("random, {limbs} limbs")));
            // A planted common factor.
            let fbits = 1 + rng.next_u64() % 200;
            let f = integer(&mut rng, fbits) + 2u32;
            pairs.push((&x * &f, &y * &f, format!("factor, {limbs} limbs")));
        }
    }
    for bits in [64, 127, 128, 129, 1000, 4000] {
        let (a, b) = fibonacci(bits);
        pairs.push((a.clone(), b.clone(), format!("Fibonacci, {bits} bits")));
        pairs.push((
            &a * 6u32,
            &b * 4u32,
            format!("Fibonacci times 6 and 4, {bits} bits"),
        ));
    }
    for e in [1u32, 63, 64, 65, 200, 2000] {
        let x = BigUint::one() << e;
        pairs.push((x.clone(), BigUint::from(2u32), format!("2^{e} and 2")));
        pairs.push((x.clone(), BigUint::from(3u32), format!("2^{e} and 3")));
        pairs.push((x.clone(), &x - 1u32, format!("2^{e} and 2^{e} - 1")));
    }
    let big = integer(&mut rng, 64 * 40);
    for y in [1u64, 2, 3, 0xffff_ffff_ffff_fffb, u64::MAX] {
        pairs.push((big.clone(), BigUint::from(y), format!("40 limbs and {y}")));
        pairs.push((
            &big * y,
            BigUint::from(y),
            format!("40 limbs times {y} and {y}"),
        ));
    }
    let y = integer(&mut rng, 300);
    pairs.push((BigUint::zero(), BigUint::one(), "0 and 1".into()));
    pairs.push((BigUint::zero(), y.clone(), "0 and y".into()));
    pairs.push((y.clone(), BigUint::zero(), "y and 0".into()));
    pairs.push((BigUint::one(), y.clone(), "1 and y".into()));
    pairs.push((y.clone(), y.clone(), "y and y".into()));
    for (x, y, case) in &pairs {
        assert_eq!(coprime(x, y), x.gcd(y).is_one(), "{case}: x = {x}, y = {y}");
        assert_eq!(coprime(y, x), coprime(x, y), "{case}, swapped");
    }
}
