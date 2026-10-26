//! q-prime-count-time, sub-item 1: `lehmer::Mqrr`'s maximal quotient
//! rational reconstruction returns what the plain algorithm of
//! monagan-2004 §3 returns, the scaled residue `Mqrr::classify` gives from
//! the converted scaled images is u·l mod M in the symmetric range, small
//! where |v|·2^(T+1) < M, and the threshold grows with the modulus
//! (plan.md, _Test code_).

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::is_prime_64;
use crate::lehmer::{MARGIN, Mqrr, Scaled, threshold};
use crate::mixed_radix_tests::{classify, crt_of};
use crate::rational::Rational;
use crate::shortcut_tests::{image, integer, product};

/// The numbers of primes the moduli are products of: one, two, and four,
/// past 64 and 128 bits, and 13 to 290, counts of primes a call over ℚ
/// combines (log 0076-1, _Step 1_).
const COUNTS: [usize; 7] = [1, 2, 4, 13, 44, 100, 290];

/// k distinct random primes of [2^`low`, 2^(`low` + 1)), `low` 30 or 61.
pub(crate) fn primes(rng: &mut Xoshiro256PlusPlus, k: usize, low: u32) -> Vec<u64> {
    let mut used: Vec<u64> = Vec::with_capacity(k);
    while used.len() < k {
        let mut c = (1u64 << low) | (rng.next_u64() >> (64 - low)) | 1;
        while c < 1 << (low + 1) && !is_prime_64(c) {
            c += 2;
        }
        if c < 1 << (low + 1) && !used.contains(&c) {
            used.push(c);
        }
    }
    used
}

/// A random integer of exactly `bits` bits, `bits` at least 1.
pub(crate) fn of_bits(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    integer(rng, bits - 1) | BigUint::one() << (bits - 1)
}

pub(crate) fn limbs(x: &BigUint) -> Vec<u64> {
    x.to_u64_digits()
}

/// The test code's oracle, monagan-2004 §3's algorithm MQRR as written, on
/// the whole numbers: the extended Euclidean algorithm on (m, v) to its
/// end, and the rational r₁/t₁ before the largest quotient q = ⌊r₀/r₁⌋,
/// the first of equal ones, where q is at least 2^t, d > 0; None
/// where no quotient reaches 2^t or gcd(r₁, t₁) ≠ 1. v in [1, m).
fn oracle(v: &BigUint, m: &BigUint, t: u64) -> Option<Rational> {
    let floor = BigUint::one() << t;
    let (mut r0, mut r1) = (BigInt::from(m.clone()), BigInt::from(v.clone()));
    let (mut t0, mut t1) = (BigInt::zero(), BigInt::one());
    let mut best: Option<(BigUint, BigInt, BigInt)> = None;
    while !r1.is_zero() {
        let q = &r0 / &r1;
        let qu = q.to_biguint().expect("a quotient is at least 0");
        if qu >= floor && best.as_ref().is_none_or(|(b, _, _)| qu > *b) {
            best = Some((qu, r1.clone(), t1.clone()));
        }
        let r = &r0 - &q * &r1;
        (r0, r1) = (r1, r);
        let t = &t0 - &q * &t1;
        (t0, t1) = (t1, t);
    }
    let (_, n, d) = best?;
    let (n, d) = match d.sign() {
        Sign::Minus => (-n, -d),
        Sign::NoSign | Sign::Plus => (n, d),
    };
    if !n.gcd(&d).is_one() {
        return None;
    }
    Some(Rational {
        num: n,
        den: d.to_biguint().expect("d > 0"),
    })
}

/// `Mqrr::with_threshold(m, t).reconstruct(v)` is the oracle's answer.
fn agrees(m: &BigUint, v: &BigUint, t: u64, case: &str) -> Option<Rational> {
    let want = oracle(v, m, t);
    let got = Mqrr::with_threshold(m, t).reconstruct(&limbs(v));
    assert_eq!(got, want, "{case}: v = {v}, M = {m}, t = {t}");
    got
}

/// The residue of a random rational n/d mod m whose |n|·d has about
/// `bits` bits, its numerator's share of them `share` in [0, 1], and a
/// random sign; None where d is no unit mod m.
fn shaped(rng: &mut Xoshiro256PlusPlus, m: &BigUint, bits: u64, share: f64) -> Option<BigUint> {
    let bits = bits.max(2);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a share of a bit count below 2^32"
    )]
    let bn = ((bits as f64 * share) as u64).clamp(1, bits - 1);
    let n = BigInt::from(of_bits(rng, bn));
    let n = if rng.next_u64().is_multiple_of(2) {
        n
    } else {
        -n
    };
    let d = of_bits(rng, bits - bn);
    image(&n, &d, m)
}

/// q-prime-count-time.3, .4: the threshold is ⌈log₂ L(M)⌉ + `MARGIN`, L the
/// bit length, and `MARGIN` is 10.
#[test]
fn the_threshold_grows_with_the_modulus() {
    assert_eq!(MARGIN, 10);
    for (bits, log) in [
        (31u64, 5u64),
        (32, 5),
        (33, 6),
        (62, 6),
        (63, 6),
        (64, 6),
        (65, 7),
        (124, 7),
        (1000, 10),
        (1024, 10),
        (1025, 11),
        (61_800, 16),
    ] {
        let m = (BigUint::one() << (bits - 1)) | BigUint::one();
        assert_eq!(m.bits(), bits);
        assert_eq!(threshold(&m), log + MARGIN, "{bits} bits");
        assert_eq!(Mqrr::new(&m).threshold(), log + MARGIN, "{bits} bits");
    }
}

/// q-prime-count-time.1, .5, .6, .7: residues of rationals of every shape
/// near the bound |n|·d·2^t ≈ M, and uniform residues, modulo products of
/// primes of [2^30, 2^31) and [2^61, 2^62): the reconstruction is the
/// plain algorithm's, at the call's threshold and at 1, 64, and the
/// threshold plus 20.
#[test]
fn the_reconstruction_is_the_plain_algorithms() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x3a77);
    for low in [30, 61] {
        for k in COUNTS {
            let ps = primes(&mut rng, k, low);
            let m = product(&ps);
            let own = threshold(&m);
            let mut found = 0;
            for i in 0..24 {
                let t = [own, 1, 64, own + 20][i % 4];
                let case = format!("{k} primes of 2^{low}, case {i}");
                let share = [0.5, 0.05, 0.95, 0.3, 0.7][i % 5];
                let slack = rng.next_u64() % 9;
                let bits = (m.bits() + 4).saturating_sub(t + slack);
                if let Some(v) = shaped(&mut rng, &m, bits, share)
                    && !v.is_zero()
                    && agrees(&m, &v, t, &case).is_some()
                {
                    found += 1;
                }
                let v = integer(&mut rng, m.bits()) % &m;
                if !v.is_zero() {
                    agrees(&m, &v, t, &format!("{case}, uniform"));
                }
            }
            assert!(found > 0, "{k} primes of 2^{low}: no rational found");
        }
    }
}

/// q-prime-count-time.1: v in [1, M) at steps of ⌊M/3000⌋, every v where
/// M < 6000, modulo products of small odd primes, where gcd(r, t) ≠ 1
/// and the leading parts are the numbers
/// themselves: the reconstruction is the plain algorithm's at t from 1 to
/// 4.
#[test]
fn small_moduli_exhaustively() {
    for m in [3u64, 15, 105, 1155, 3 * 5 * 7 * 11 * 13, 65_537, 3 * 65_537] {
        let mb = BigUint::from(m);
        let step = (m / 3000).max(1);
        for t in 1..=4 {
            let mut v = 1;
            while v < m {
                agrees(&mb, &BigUint::from(v), t, &format!("M = {m}"));
                v += step;
            }
        }
    }
}

/// q-prime-count-time.1, monagan-2004 §3, Lemma 2: where 9·n²·d² < M, the
/// largest quotient is n/d's and passes √M, so the reconstruction returns
/// n/d at the call's threshold, whatever the shape.
#[test]
fn a_rational_well_inside_the_bound_returns() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1e33);
    for k in COUNTS {
        let ps = primes(&mut rng, k, 61);
        let m = product(&ps);
        let mqrr = Mqrr::new(&m);
        assert!(m.bits() >= 2 * mqrr.threshold() + 2);
        for i in 0..40 {
            // n·d < 2^room gives 9·n²·d² < 2^(L(M) − 1) ≤ M.
            let room = (m.bits() - 5) / 2;
            let bn = 1 + rng.next_u64() % (room - 1);
            let n = BigInt::from(integer(&mut rng, bn));
            let n = if i % 2 == 0 { n } else { -n };
            let d = integer(&mut rng, room - bn).max(BigUint::one());
            let Some(r) = Rational::new(n, BigInt::from(d)) else {
                continue;
            };
            let Some(v) = image(&r.num, &r.den, &m) else {
                continue;
            };
            if v.is_zero() {
                continue;
            }
            assert_eq!(
                mqrr.reconstruct(&limbs(&v)),
                Some(r.clone()),
                "{k} primes, case {i}: {num}/{den}",
                num = r.num,
                den = r.den
            );
        }
    }
}

/// The test code's oracle of the scaled residue: v = u·l mod M in
/// (−M/2, M/2], Small(v) where |v|·2^(t+1) < M, else Residue(u·l mod M)
/// on limbs.
pub(crate) fn scale_oracle(u: &BigUint, l: &BigUint, m: &BigUint, t: u64) -> Scaled {
    let v = u * l % m;
    let s = if &v + &v > *m {
        BigInt::from(v.clone()) - BigInt::from(m.clone())
    } else {
        BigInt::from(v.clone())
    };
    if (s.magnitude() << (t + 1)) < *m {
        Scaled::Small(s)
    } else {
        Scaled::Residue(limbs(&v))
    }
}

/// q-prime-count-time.2, q-lift-growth-time.4: the scaled residue of
/// random residues by l = 1, small l, l of M's size, and l past M, and at
/// the bound: s and s + 1 for the largest s with s·2^(T+1) < M, each of
/// both signs; each the classification of the converted scaled images.
#[test]
fn the_scaled_residue() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5ca1);
    for k in COUNTS {
        let ps = primes(&mut rng, k, 61);
        let m = product(&ps);
        let crt = crt_of(&ps);
        let mqrr = Mqrr::new(&m);
        let t = mqrr.threshold();
        let case = |what: &str| format!("{k} primes: {what}");
        for i in 0..60 {
            let u = integer(&mut rng, m.bits()) % &m;
            let l = match i % 4 {
                0 => BigUint::one(),
                1 => integer(&mut rng, 40).max(BigUint::one()),
                2 => integer(&mut rng, m.bits()).max(BigUint::one()),
                _ => integer(&mut rng, 2 * m.bits()).max(BigUint::one()),
            };
            assert_eq!(
                classify(&crt, &mqrr, &u, &l),
                scale_oracle(&u, &l, &m, t),
                "{case}",
                case = case(&format!("u = {u}, l = {l}"))
            );
        }
        // The largest s with s·2^(T+1) < M.
        let s = (&m - BigUint::one()) >> (t + 1);
        let l = loop {
            let l = integer(&mut rng, 64).max(BigUint::from(3u32));
            if l.gcd(&m).is_one() {
                break l;
            }
        };
        for (v, small) in [
            (BigInt::from(s.clone()), true),
            (-BigInt::from(s.clone()), true),
            (BigInt::from(&s + 1u32), false),
            (-BigInt::from(&s + 1u32), false),
        ] {
            let u = image(&v, &l, &m).expect("l is a unit mod M");
            let got = classify(&crt, &mqrr, &u, &l);
            assert_eq!(
                got,
                scale_oracle(&u, &l, &m, t),
                "{case}",
                case = case(&format!("v = {v}"))
            );
            assert_eq!(
                matches!(got, Scaled::Small(_)),
                small,
                "{case}",
                case = case(&format!("v = {v} at the bound"))
            );
        }
        assert_eq!(
            classify(&crt, &mqrr, &BigUint::zero(), &BigUint::from(7u32)),
            Scaled::Small(BigInt::zero()),
            "{case}",
            case = case("u = 0")
        );
        let last = &m - BigUint::one();
        assert_eq!(
            classify(&crt, &mqrr, &last, &BigUint::one()),
            Scaled::Small(-BigInt::one()),
            "{case}",
            case = case("u = M − 1")
        );
    }
}

/// v and M with v/M = [0; a₁, …, aₙ], so that the Euclidean algorithm on
/// (M, v) takes the quotients a₁ to aₙ; aₙ ≥ 2 is raised by one until M is
/// odd.
pub(crate) fn continued(quotients: &[BigUint]) -> (BigUint, BigUint) {
    let mut a = quotients.to_vec();
    loop {
        let (mut h1, mut h2) = (BigUint::zero(), BigUint::one());
        let (mut k1, mut k2) = (BigUint::one(), BigUint::zero());
        for q in &a {
            (h1, h2) = (q * &h1 + &h2, h1);
            (k1, k2) = (q * &k1 + &k2, k1);
        }
        if k1.bit(0) {
            return (h1, k1);
        }
        *a.last_mut().expect("a quotient") += 1u32;
    }
}

/// q-prime-count-time.5, .6, .7: chosen quotients, the large ones apart by
/// 300 small ones, so that Lehmer's runs hold them in runs of their own:
/// two equal largest, and the first is taken; a larger after a large one,
/// and the larger is taken; the largest past 2^64, a full step's; and the
/// largest one below the threshold, which finds nothing.
#[test]
fn chosen_quotients() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xcf00);
    let small = |rng: &mut Xoshiro256PlusPlus| -> Vec<BigUint> {
        (0..300)
            .map(|_| BigUint::from(1 + rng.next_u64() % 3))
            .collect()
    };
    let big = |bits: u64| BigUint::one() << bits;
    for (case, large, t) in [
        ("two equal", [big(40), big(40)], 30),
        ("a larger later", [big(30), big(45)], 20),
        ("past 2^64", [big(70), big(90)], 32),
        ("below the threshold", [big(40), big(41)], 50),
    ] {
        let mut a = small(&mut rng);
        a.push(large[0].clone());
        a.extend(small(&mut rng));
        a.push(large[1].clone());
        a.extend(small(&mut rng));
        a.push(BigUint::from(2u32));
        let (v, m) = continued(&a);
        let got = agrees(&m, &v, t, case);
        assert_eq!(got.is_some(), case != "below the threshold", "{case}");
    }
}
