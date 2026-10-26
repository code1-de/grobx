//! q-rational-time, sub-item 1: `lehmer::gcd` is the gcd (plan.md, _Test
//! code_). The tests of `Wang::shortcut` went with it in iteration 0076
//! (its plan.md, _The test code of earlier items_).

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lehmer::gcd;

pub(crate) fn product(primes: &[u64]) -> BigUint {
    primes.iter().fold(BigUint::one(), |m, &p| m * p)
}

/// A random integer below 2^bits.
pub(crate) fn integer(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let mut x = BigUint::zero();
    for _ in 0..bits.div_ceil(64) {
        x = (x << 64u32) + BigUint::from(rng.next_u64());
    }
    x >> (bits.div_ceil(64) * 64 - bits)
}

/// A random integer of 1 to `bits` bits, at least 1.
fn positive(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let b = 1 + rng.next_u64() % bits.max(1);
    integer(rng, b).max(BigUint::one())
}

/// n·d⁻¹ mod m; None where d is no unit mod m.
pub(crate) fn image(n: &BigInt, d: &BigUint, m: &BigUint) -> Option<BigUint> {
    let m = BigInt::from(m.clone());
    let e = BigInt::from(d.clone()).extended_gcd(&m);
    if !e.gcd.is_one() {
        return None;
    }
    (n * e.x).mod_floor(&m).to_biguint()
}

/// q-rational-time.9: Lehmer's gcd on limbs equals num-integer's.
#[test]
fn lehmer_gcd_agrees_with_num_integer() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6cd);
    let check = |x: &BigUint, y: &BigUint| {
        assert_eq!(gcd(x, y), x.gcd(y), "gcd({x}, {y})");
        assert_eq!(gcd(y, x), x.gcd(y), "gcd({y}, {x})");
    };
    let zero = BigUint::zero();
    let one = BigUint::one();
    check(&zero, &zero);
    check(&zero, &one);
    check(&one, &one);
    let big = integer(&mut rng, 3000) | (BigUint::one() << 2999u32);
    check(&zero, &big);
    check(&one, &big);
    check(&big, &big);
    for e in [1u32, 63, 64, 65, 127, 128, 1000] {
        let p = BigUint::one() << e;
        check(&p, &(&p << 3u32));
        check(&p, &(&big << 5u32));
        check(&(&p - 1u32), &p);
    }
    for (x, y) in [
        (u64::MAX, 1u64),
        (u64::MAX, u64::MAX - 1),
        (1 << 63, 3 << 62),
    ] {
        check(&BigUint::from(x), &BigUint::from(y));
        check(&BigUint::from(x), &big);
    }
    for i in 0..400 {
        let g = positive(&mut rng, 1 + (i % 1000) as u64);
        let (bx, by) = (1 + rng.next_u64() % 2000, 1 + rng.next_u64() % 2000);
        let x = &g * positive(&mut rng, bx);
        let y = &g * positive(&mut rng, by);
        check(&x, &y);
        let lone = BigUint::from(rng.next_u64() | 1);
        check(&x, &lone);
    }
}
