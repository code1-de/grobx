//! q-lift, sub-item 1: the integers of the lift, the Chinese remainder
//! steps, rational reconstruction, the scaling to primitive integers, and
//! the texts over Q (plan.md, _Test code_).

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::field::Gf31;
use crate::mixed_radix_tests::converted;
use crate::rational::{
    Crt, Rational, bound, reconstruct, residue, scale, write_cofactor_q, write_poly_q,
};

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

fn big(digits: &str) -> BigUint {
    BigUint::parse_bytes(digits.as_bytes(), 10).unwrap()
}

fn int(v: i64) -> BigInt {
    BigInt::from(v)
}

fn q(n: i64, d: i64) -> Rational {
    Rational::new(int(n), int(d)).unwrap()
}

/// u⁻¹ mod m, for u coprime to m.
fn inverse(u: &BigInt, m: &BigInt) -> BigInt {
    let e = u.extended_gcd(m);
    assert!(e.gcd.is_one(), "{u} is no unit mod {m}");
    e.x.mod_floor(m)
}

#[test]
fn combines_by_chinese_remainders() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    for case in 0..18 {
        let bits = [1, 31, 64, 200, 1000, 2000][case % 6];
        let xs: Vec<BigUint> = (0..5).map(|_| integer(&mut rng, bits)).collect();
        let mut crt = Crt::new();
        assert!(crt.modulus().is_one());
        assert_eq!(crt.primes().len(), 0, "no prime combined");
        let mut images: Vec<Vec<u64>> = vec![Vec::new(); xs.len()];
        let mut used = Vec::new();
        for k in 0..70 {
            let p = prime(&mut rng, &used);
            used.push(p);
            let step = crt.step(p, 0);
            assert_eq!(step.prime(), p);
            for (x, images) in xs.iter().zip(&mut images) {
                images.push((x % BigUint::from(p)).to_u64().unwrap());
            }
            crt.push(step);
            let m = crt.modulus().clone();
            assert_eq!(crt.primes(), &used[..]);
            assert_eq!(
                m,
                used.iter().map(|&p| BigUint::from(p)).product::<BigUint>()
            );
            for (x, images) in xs.iter().zip(&images) {
                let r = converted(&crt, images);
                assert!(r < m, "case {case}, step {k}: the residue passes M");
                assert_eq!(r, x % &m, "case {case}, step {k}");
                if k == 0 {
                    assert_eq!(r, x % BigUint::from(p), "the first step sets x = r");
                }
            }
        }
    }
}

#[test]
fn residues_of_integers() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
    let primes = [
        2u64,
        3,
        65521,
        1_073_741_827,
        2_147_483_647,
        (1 << 61) - 1,
        4_611_686_018_427_387_847,
    ];
    let mut xs = vec![
        BigUint::zero(),
        BigUint::one(),
        BigUint::from(u64::MAX),
        BigUint::one() << 64u32,
        (BigUint::one() << 64u32) + BigUint::one(),
    ];
    for _ in 0..20 {
        xs.push(integer(&mut rng, 1000));
    }
    for &p in &primes {
        let mut all = xs.clone();
        all.extend([
            BigUint::from(p - 1),
            BigUint::from(p),
            BigUint::from(p) * 7u32,
        ]);
        for x in &all {
            let want = (x % BigUint::from(p)).to_u64().unwrap();
            assert_eq!(residue(x, p), want, "{x} mod {p}");
        }
    }
}

#[test]
fn reconstructs_small_rationals() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
    for primes in 1..=12 {
        let mut used = Vec::new();
        for _ in 0..primes {
            let p = prime(&mut rng, &used);
            used.push(p);
        }
        let m: BigUint = used.iter().map(|&p| BigUint::from(p)).product();
        let n_bound = bound(&m);
        let mi = BigInt::from(m.clone());
        let nb = BigInt::from(n_bound.clone());
        let mut cases: Vec<(BigInt, BigInt)> = vec![
            (BigInt::zero(), BigInt::one()),
            (nb.clone(), BigInt::one()),
            (-nb.clone(), BigInt::one()),
            (BigInt::one(), nb.clone()),
            (-BigInt::one(), nb.clone()),
        ];
        for _ in 0..40 {
            let bits = n_bound.bits().max(1);
            let n = BigInt::from(integer(&mut rng, bits) % (&n_bound + 1u32));
            let d = BigInt::from(integer(&mut rng, bits) % &n_bound) + BigInt::one();
            let n = if rng.next_u64() % 2 == 0 { n } else { -n };
            cases.push((n, d));
        }
        for (n, d) in cases {
            let g = n.gcd(&d);
            let (n, d) = (&n / &g, &d / &g);
            let u = (&n * inverse(&d, &mi)).mod_floor(&mi);
            let u = u.to_biguint().unwrap();
            let got = reconstruct(&u, &m, &n_bound)
                .unwrap_or_else(|| panic!("{n}/{d} mod a product of {primes} primes"));
            assert_eq!(got.num, n, "{n}/{d}");
            assert_eq!(BigInt::from(got.den.clone()), d, "{n}/{d}");
        }
        // 0 is 0/1.
        let zero = reconstruct(&BigUint::zero(), &m, &n_bound).unwrap();
        assert!(zero.is_zero());
        assert!(zero.den.is_one());
    }
}

// q-lift.14: monagan-2004 §2's table for Wang's reconstruction, the test
// code's oracle, and collins-1995 §2's counterexample.
#[test]
fn refuses_what_wang_refuses() {
    let m = BigUint::from(19u32);
    let n = BigUint::from(3u32);
    assert_eq!(bound(&m), n);
    let table: [Option<(i64, i64)>; 18] = [
        Some((1, 1)),
        Some((2, 1)),
        Some((3, 1)),
        None,
        None,
        Some((-1, 3)),
        Some((2, 3)),
        Some((-3, 2)),
        Some((-1, 2)),
        Some((1, 2)),
        Some((3, 2)),
        Some((-2, 3)),
        Some((1, 3)),
        None,
        None,
        Some((-3, 1)),
        Some((-2, 1)),
        Some((-1, 1)),
    ];
    for (u, want) in (1u32..=18).zip(table) {
        let got = reconstruct(&BigUint::from(u), &m, &n);
        assert_eq!(got, want.map(|(a, b)| q(a, b)), "u = {u} mod 19");
    }
    let got = reconstruct(
        &BigUint::from(5u32),
        &BigUint::from(12u32),
        &BigUint::from(2u32),
    );
    assert_eq!(got, None, "M = 12, U = 5: (−2, 2) has gcd 2");
}

#[test]
fn bound_is_the_square_root() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
    let mut ms: Vec<BigUint> = [2u64, 3, 19, 1 << 61]
        .iter()
        .map(|&m| BigUint::from(m))
        .collect();
    let mut used = Vec::new();
    let mut m = BigUint::one();
    for _ in 0..30 {
        let p = prime(&mut rng, &used);
        used.push(p);
        m *= p;
        ms.push(m.clone());
    }
    for m in ms {
        let b = bound(&m);
        let half = &m / 2u32;
        assert_eq!(b, half.sqrt(), "bound of {m}");
        assert!(&b * &b <= half);
        assert!((&b + 1u32) * (&b + 1u32) > half);
    }
}

#[test]
#[allow(
    clippy::cast_possible_truncation,
    reason = "test code: a residue modulo a prime of a u64 fits a u64"
)]
fn rationals_reduce() {
    assert_eq!(Rational::new(int(2), int(4)), Some(q(1, 2)));
    assert_eq!(Rational::new(int(3), int(-6)), Some(q(-1, 2)));
    let minus = Rational::new(int(3), int(-6)).unwrap();
    assert_eq!(minus.num, int(-1));
    assert_eq!(minus.den, BigUint::from(2u32));
    let zero = Rational::new(int(0), int(5)).unwrap();
    assert!(zero.is_zero());
    assert!(zero.den.is_one());
    assert_eq!(Rational::new(int(1), int(0)), None);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
    let primes = [2u64, 3, 5, 65521, 1_073_741_827, 2_147_483_647];
    for _ in 0..300 {
        let n = (rng.next_u64() >> 2) as i64 * if rng.next_u64() % 2 == 0 { 1 } else { -1 };
        let d = 1 + (rng.next_u64() >> 34) as i64;
        let r = Rational::new(int(n), int(d)).unwrap();
        for &p in &primes {
            let (num, den) = (r.num.clone(), BigInt::from(r.den.clone()));
            let dp = den.mod_floor(&BigInt::from(p)).to_u128().unwrap();
            if dp == 0 {
                assert_eq!(r.residue(p), None, "{n}/{d} mod {p}");
                continue;
            }
            let np = num.mod_floor(&BigInt::from(p)).to_u128().unwrap();
            // d^(p−2) mod p by square and multiply.
            let p128 = u128::from(p);
            let (mut inv, mut b, mut e) = (1u128, dp, p128 - 2);
            while e > 0 {
                if e & 1 == 1 {
                    inv = inv * b % p128;
                }
                b = b * b % p128;
                e >>= 1;
            }
            let want = (np * inv % p128) as u64;
            assert_eq!(r.residue(p), Some(want), "{n}/{d} mod {p}");
        }
    }
}

/// The coefficients as (n, d), L, and the integers.
type ScaleCase<'a> = (&'a [(i64, i64)], i64, &'a [i64]);

#[test]
fn scales_to_primitive_integers() {
    let cases: [ScaleCase; 3] = [
        (&[(1, 1), (-1, 2), (3, 4)], 4, &[4, -2, 3]),
        (&[(1, 1)], 1, &[1]),
        (&[(1, 1), (2, 3), (5, 6), (-7, 10)], 30, &[30, 20, 25, -21]),
    ];
    for (coefs, l, want) in cases {
        let coefs: Vec<Rational> = coefs.iter().map(|&(n, d)| q(n, d)).collect();
        let (got_l, got) = scale(&coefs);
        assert_eq!(got_l, BigUint::from(l as u64));
        assert_eq!(got, want.iter().map(|&v| int(v)).collect::<Vec<_>>());
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(6);
    for _ in 0..200 {
        let mut coefs = vec![q(1, 1)];
        for _ in 0..=(rng.next_u64() % 8) {
            let n = (rng.next_u64() % 2001) as i64 - 1000;
            let d = 1 + (rng.next_u64() % 360) as i64;
            coefs.push(q(n, d));
        }
        let (l, ints) = scale(&coefs);
        assert!(ints[0].is_positive());
        assert_eq!(ints[0], BigInt::from(l.clone()));
        let g = ints.iter().fold(BigInt::zero(), |g, c| g.gcd(c));
        assert!(g.is_one(), "{coefs:?} scaled to {ints:?}");
        for (c, i) in coefs.iter().zip(&ints) {
            assert_eq!(
                BigInt::from(l.clone()) * &c.num,
                i * BigInt::from(c.den.clone())
            );
        }
    }
}

#[test]
fn writes_over_q() {
    let xy = ["x", "y"];
    let poly = |exponents: &[u32], coefs: &[BigInt]| {
        let mut out = String::new();
        write_poly_q(&mut out, &xy, exponents, coefs, &mut Budget::unbounded())
            .expect("a budget without bounds never stops");
        out
    };
    assert_eq!(poly(&[1, 0, 0, 0], &[int(1), int(-1)]), "x-1");
    assert_eq!(
        poly(&[2, 0, 0, 1, 0, 0], &[int(2), int(-3), int(1)]),
        "2*x^2-3*y+1"
    );
    assert_eq!(poly(&[1, 3, 0, 0], &[int(1), int(-1)]), "x*y^3-1");
    assert_eq!(poly(&[0, 0], &[int(-7)]), "-7");
    assert_eq!(poly(&[1, 0, 0, 1], &[int(-1), int(1)]), "-x+y");
    let forty = BigInt::from(big("10000000000000000000000000000000000000001"));
    assert_eq!(
        poly(&[1, 0, 0, 0], &[int(1), forty.clone()]),
        "x+10000000000000000000000000000000000000001"
    );
    assert_eq!(
        poly(&[0, 2, 0, 0], &[-forty, int(1)]),
        "-10000000000000000000000000000000000000001*y^2+1"
    );

    let cofactor = |names: &[&str], exponents: &[u32], coefs: &[Rational]| {
        let mut out = String::new();
        write_cofactor_q(&mut out, names, exponents, coefs, &mut Budget::unbounded())
            .expect("a budget without bounds never stops");
        out
    };
    assert_eq!(
        cofactor(&["x2", "x3"], &[3, 1], &[q(-465, 14)]),
        "-465/14*x2^3*x3"
    );
    assert_eq!(cofactor(&xy, &[0, 0], &[q(1, 2)]), "1/2");
    assert_eq!(cofactor(&xy, &[0, 1], &[q(-1, 1)]), "-y");
    assert_eq!(cofactor(&xy, &[], &[]), "0");
    assert_eq!(cofactor(&xy, &[1, 0], &[q(3, 1)]), "3*x");
    assert_eq!(cofactor(&xy, &[0, 0], &[q(1, 1)]), "1");
    assert_eq!(cofactor(&xy, &[0, 0], &[q(-1, 1)]), "-1");
    assert_eq!(
        cofactor(&xy, &[2, 0, 0, 1, 0, 0], &[q(1, 1), q(-1, 3), q(2, 1)]),
        "x^2-1/3*y+2"
    );
    assert_eq!(
        cofactor(&xy, &[1, 1, 0, 0], &[q(-1, 7), q(5, 2)]),
        "-1/7*x*y+5/2"
    );
}
