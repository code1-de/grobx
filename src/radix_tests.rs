//! q-crt-time, the sub-item: residues modulo a prime by the powers of
//! 2^64, the Chinese remainder step, and the check of a rational without an
//! inverse (plan.md, _Test code_). The steps of a catch-up went with the
//! catch-up (result-time.13 to .16), and the step on limbs went with the
//! residues modulo M: the images an attempt converts stand in its place
//! (q-lift-growth-time.1).

use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::Gf31;
use crate::mixed_radix_tests::converted;
use crate::rational::{Crt, Radix, Rational, residue};

/// The primes residues are taken modulo: a small one, GF(65521)'s, and
/// three near 2^30 and 2^31.
const PRIMES: [u64; 5] = [101, 65521, 1_073_741_827, 2_147_483_629, 2_147_483_647];

/// A random prime of [2^30, 2^31) not among `used`, by `Gf31::new`'s test.
fn prime(rng: &mut Xoshiro256PlusPlus, used: &[u64]) -> u64 {
    loop {
        let mut c = (1u64 << 30) + (rng.next_u64() >> 34);
        while c < 1 << 31 && Gf31::new(c).is_none() {
            c += 1;
        }
        if c < 1 << 31 && !used.contains(&c) {
            return c;
        }
    }
}

/// A random integer below 2^bits.
fn integer(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let mut x = BigUint::zero();
    for _ in 0..bits.div_ceil(64) {
        x = (x << 64u32) + BigUint::from(rng.next_u64());
    }
    x >> (bits.div_ceil(64) * 64 - bits)
}

/// x on limbs, the least significant first, no zero limb at the top.
fn limbs(x: &BigUint) -> Vec<u64> {
    x.to_u64_digits()
}

/// q-crt-time.4: the sum of the limbs times the powers of 2^64, reduced
/// once, is the remainder Horner's rule takes limb by limb.
#[test]
fn radix_residues_agree_with_horner() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7ad1);
    for &p in &PRIMES {
        let radix = Radix::new(p, 40);
        assert_eq!(radix.prime(), p);
        let mut xs = vec![
            BigUint::zero(),
            BigUint::one(),
            BigUint::from(u64::MAX),
            BigUint::from(p - 1),
            BigUint::from(p),
            BigUint::from(p) * 7u32,
            (BigUint::one() << (64 * 40u32)) - 1u32,
        ];
        for k in 0..=40 {
            xs.push(integer(&mut rng, 64 * k));
        }
        for _ in 0..100 {
            let bits = rng.next_u64() % (64 * 40) + 1;
            xs.push(integer(&mut rng, bits));
        }
        for x in &xs {
            let want = residue(x, p);
            assert_eq!(radix.residue(&limbs(x)), want, "{x} mod {p}, on limbs");
            assert_eq!(radix.residue_big(x), want, "{x} mod {p}");
        }
    }
}

/// q-crt-time.6, q-lift-growth-time.1: the images at the primes of each
/// step convert to the residue modulo M·p, with no zero limb at the top,
/// as `rational_tests.rs`'s `combines_by_chinese_remainders` asserts it of
/// every step.
#[test]
fn the_limb_step_combines_by_chinese_remainders() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7ad2);
    for case in 0..18 {
        let bits = [1, 31, 64, 200, 1000, 2000][case % 6];
        let xs: Vec<BigUint> = (0..5).map(|_| integer(&mut rng, bits)).collect();
        let mut crt = Crt::new();
        let mut images: Vec<Vec<u64>> = vec![Vec::new(); xs.len()];
        let mut used = Vec::new();
        for k in 0..70 {
            let p = prime(&mut rng, &used);
            used.push(p);
            let step = crt.step(p, 0);
            assert_eq!(step.prime(), p);
            assert_eq!(step.radix().prime(), p);
            for (x, images) in xs.iter().zip(&mut images) {
                images.push((x % BigUint::from(p)).to_u64().unwrap());
            }
            crt.push(step);
            let m = crt.modulus().clone();
            for (x, images) in xs.iter().zip(&images) {
                let r = converted(&crt, images);
                assert!(r < m, "case {case}, step {k}: the residue passes M");
                assert_eq!(r, x % &m, "case {case}, step {k}");
            }
        }
    }
}

/// q-crt-time.8: n ≡ c·d mod p with d a unit agrees with n·d⁻¹ ≡ c, and
/// `rational` is `Rational::residue`.
#[test]
fn fits_agrees_with_the_inverse() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7ad4);
    for &p in &PRIMES {
        // 300 bits take five limbs.
        let radix = Radix::new(p, 5);
        for case in 0..400 {
            let n_bits = rng.next_u64() % 301;
            let d_bits = rng.next_u64() % 300 + 1;
            let mut n = BigInt::from(integer(&mut rng, n_bits));
            if rng.next_u64() % 2 == 1 {
                n = -n;
            }
            let mut d = integer(&mut rng, d_bits);
            if d.is_zero() {
                d = BigUint::one();
            }
            if case % 7 == 0 && d.bits() < 280 {
                d *= p;
            }
            let r = Rational::new(n, BigInt::from(d)).unwrap();
            let image = r.residue(p);
            assert_eq!(radix.rational(&r), image, "{r:?} mod {p}");
            let c = image.unwrap_or(rng.next_u64() % p);
            for c in [c, (c + 1) % p] {
                assert_eq!(radix.fits(&r, c), image == Some(c), "{r:?} at {p}, c = {c}");
            }
        }
    }
}
