//! q-lift-growth-time, sub-item 1: `Crt::convert` turns the images of an
//! integer at the primes combined into the integer modulo their product
//! by its mixed-radix digits, in the form its top digit names; a prime's
//! step holds the residues of the products of the primes before it; the
//! scaled images classify as `mqrr_tests.rs`'s `scale_oracle` classifies
//! u·l mod M; and the conversion counts its products (plan.md, _Test
//! code_).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::shoup_factor;
use crate::lehmer::{Mqrr, Scaled, threshold};
use crate::mqrr_tests::{primes, scale_oracle};
use crate::rational::{Converted, Crt, residue, scale_images};
use crate::shortcut_tests::{image, integer, product};

/// x mod p as a word.
pub(crate) fn word(x: &BigUint, p: u64) -> u64 {
    (x % BigUint::from(p)).to_u64().unwrap()
}

/// The integer of limbs `x`, the least significant first.
pub(crate) fn from_limbs(x: &[u64]) -> BigUint {
    x.iter()
        .rev()
        .fold(BigUint::zero(), |a, &l| (a << 64u32) + BigUint::from(l))
}

/// k distinct primes, below 2^31 and of [2^61, 2^62) in turn.
pub(crate) fn mixed(rng: &mut Xoshiro256PlusPlus, k: usize) -> Vec<u64> {
    let narrow = primes(rng, k.div_ceil(2), 30);
    let wide = primes(rng, k / 2, 61);
    (0..k)
        .map(|i| {
            if i % 2 == 0 {
                narrow[i / 2]
            } else {
                wide[i / 2]
            }
        })
        .collect()
}

/// A `Crt` with `ps` pushed, each by its step.
pub(crate) fn crt_of(ps: &[u64]) -> Crt {
    let mut crt = Crt::new();
    for &p in ps {
        let step = crt.step(p, 1);
        crt.push(step);
    }
    crt
}

/// The value a conversion's form and limbs name, modulo M.
fn value(form: Converted, limbs: &[u64], m: &BigUint) -> BigUint {
    match form {
        Converted::Low | Converted::Any => from_limbs(limbs),
        Converted::High => m - BigUint::one() - from_limbs(limbs),
    }
}

/// The x in [0, M) that the conversion of `images`, one at each prime of
/// `crt`, gives, M `crt`'s modulus: by the form it names, from its limbs,
/// which have no zero at the top.
pub(crate) fn converted(crt: &Crt, images: &[u64]) -> BigUint {
    assert_eq!(images.len(), crt.primes().len(), "an image a prime");
    let (mut digits, mut out) = (Vec::new(), Vec::new());
    let (form, _) = crt.convert(images, &mut digits, &mut out);
    assert_ne!(out.last(), Some(&0), "a zero limb at the top");
    value(form, &out, crt.modulus())
}

/// The form x's top digit ⌊x/P⌋ names, P the product of the primes but
/// the last, `last` the last prime.
fn form_of(x: &BigUint, below_last: &BigUint, last: u64) -> Converted {
    let top = x / below_last;
    if top.is_zero() {
        Converted::Low
    } else if top == BigUint::from(last - 1) {
        Converted::High
    } else {
        Converted::Any
    }
}

/// Checks one conversion of x's images at the first k primes of `crt`:
/// the value, the form, the limbs without a zero at the top, and the
/// digits, each below its prime, whose mixed-radix sum is x mod M.
fn check(crt: &Crt, x: &BigUint, case: &str) {
    let ps = crt.primes();
    let k = ps.len();
    let m = crt.modulus();
    let images: Vec<u64> = ps.iter().map(|&p| word(x, p)).collect();
    let (mut digits, mut out) = (Vec::new(), Vec::new());
    let (form, _) = crt.convert(&images, &mut digits, &mut out);
    let want = x % m;
    assert_ne!(out.last(), Some(&0), "{case}: a zero limb at the top");
    assert_eq!(value(form, &out, m), want, "{case}: the value");
    let below_last = product(&ps[..k - 1]);
    assert_eq!(
        form,
        form_of(&want, &below_last, ps[k - 1]),
        "{case}: the form"
    );
    match form {
        Converted::Low | Converted::High => {
            assert!(from_limbs(&out) < below_last, "{case}: below p₀⋯p_(k−2)");
        }
        Converted::Any => {}
    }
    assert_eq!(digits.len(), k, "{case}: a digit a prime");
    let mut sum = BigUint::zero();
    let mut weight = BigUint::one();
    for (&d, &p) in digits.iter().zip(ps) {
        assert!(d < p, "{case}: the digit {d} past {p}");
        sum += &weight * d;
        weight *= p;
    }
    assert_eq!(sum, want, "{case}: the digits' sum");
}

/// Test 1, q-lift-growth-time.1 to .3, .6, .7: integers of 1 to 20,000
/// bits converted from their images after every push up to 70 primes and
/// after every 37th up to 400, at primes below 2^31 and of [2^61, 2^62)
/// in turn.
#[test]
fn converts_images_to_the_residue() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6a7e);
    for case in 0..8 {
        let bits = [1, 31, 64, 200, 1000, 2000, 6000, 20_000][case];
        let xs: Vec<BigUint> = (0..3).map(|_| integer(&mut rng, bits)).collect();
        let ps = mixed(&mut rng, 400);
        let mut crt = Crt::new();
        for (k, &p) in ps.iter().enumerate() {
            let step = crt.step(p, 1);
            crt.push(step);
            if k < 70 || k % 37 == 0 || k == ps.len() - 1 {
                for x in &xs {
                    check(
                        &crt,
                        x,
                        &format!("{bits} bits, {primes} primes", primes = k + 1),
                    );
                }
            }
        }
    }
}

/// Test 2, q-lift-growth-time.2, .7: at 1, 2, 3, and 70 primes, x = 0,
/// 1, M − 1, M − 2, P − 1, P, M − P − 1, and M − P, P the product of the
/// primes but the last: each value back, in the form its top digit names.
#[test]
fn converts_the_extremes() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xe873);
    for k in [1, 2, 3, 70] {
        let ps = mixed(&mut rng, k);
        let crt = crt_of(&ps);
        let m = crt.modulus().clone();
        let p = product(&ps[..k - 1]);
        let one = BigUint::one();
        for x in [
            BigUint::zero(),
            one.clone(),
            &m - &one,
            &m - 2u32,
            &p - &one,
            p.clone(),
            &m - &p - &one,
            &m - &p,
        ] {
            check(&crt, &x, &format!("{k} primes, x = {x}"));
        }
    }
}

/// Test 3, q-lift-growth-time.3: each step's row holds the residues of
/// the products of the primes before it, (p₀⋯p_(j−1)) mod p, and its
/// inverse is (M mod p)⁻¹ with Shoup's factor ⌊inverse·2^64/p⌋.
#[test]
fn the_step_holds_the_products_residues() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x57e9);
    let ps = mixed(&mut rng, 120);
    let mut crt = Crt::new();
    for (k, &p) in ps.iter().enumerate() {
        let step = crt.step(p, 1);
        assert_eq!(step.row().len(), k, "{k} primes before");
        for (j, &r) in step.row().iter().enumerate() {
            assert_eq!(r, word(&product(&ps[..j]), p), "row {k}, entry {j}");
        }
        let m = word(crt.modulus(), p);
        let inverse = step.inverse();
        assert_eq!(
            u128::from(inverse) * u128::from(m) % u128::from(p),
            1 % u128::from(p),
            "the inverse at {k}"
        );
        let factor = ((u128::from(inverse) << 64) / u128::from(p)) as u64;
        assert_eq!(step.factor(), factor, "the factor at {k}");
        assert_eq!(shoup_factor(inverse, p), factor);
        crt.push(step);
    }
}

/// An l of `bits` bits prime to M, at least 1.
fn unit(rng: &mut Xoshiro256PlusPlus, bits: u64, m: &BigUint) -> BigUint {
    loop {
        let l = integer(rng, bits).max(BigUint::one());
        if l.gcd(m).is_one() {
            return l;
        }
    }
}

/// `Mqrr::classify` of u's images scaled by l's residues and converted,
/// at the first k primes of `crt`, under `mqrr`.
pub(crate) fn classify(crt: &Crt, mqrr: &Mqrr, u: &BigUint, l: &BigUint) -> Scaled {
    let ps = crt.primes();
    let images: Vec<u64> = ps.iter().map(|&p| word(u, p)).collect();
    let residues: Vec<u64> = ps.iter().map(|&p| residue(l, p)).collect();
    let factors: Vec<u64> = residues
        .iter()
        .zip(ps)
        .map(|(&r, &p)| shoup_factor(r, p))
        .collect();
    let mut scaled = Vec::new();
    scale_images(&images, &residues, &factors, ps, &mut scaled);
    let (mut digits, mut out) = (Vec::new(), Vec::new());
    let (form, _) = crt.convert(&scaled, &mut digits, &mut out);
    mqrr.classify(form, &out)
}

/// Test 4, q-lift-growth-time.4, .6, .7: u below M and l prime to M,
/// at 1 to 60 primes of [2^61, 2^62) and below 2^31: the classification
/// of the converted scaled images is `scale_oracle`'s of u·l mod M at
/// M's threshold, and at the bound, s and s + 1 of both signs for the
/// largest s with s·2^(T+1) < M, where s passes p₀⋯p_(k−2) and its top
/// digit is neither 0 nor p_(k−1) − 1. At a threshold whose s lies below
/// p₀⋯p_(k−2), the bound and 0, 1, and 2 of both signs.
#[test]
fn classifies_as_the_scaling_did() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc1a5);
    for (k, low) in [(1, 61), (2, 61), (4, 30), (13, 61), (44, 30), (60, 61)] {
        let ps = primes(&mut rng, k, low);
        let crt = crt_of(&ps);
        let m = crt.modulus().clone();
        let mqrr = Mqrr::new(&m);
        let t = threshold(&m);
        for i in 0..40 {
            let u = integer(&mut rng, m.bits()) % &m;
            let bits = [1, 40, m.bits(), 2 * m.bits()][i % 4];
            let l = unit(&mut rng, bits, &m);
            assert_eq!(
                classify(&crt, &mqrr, &u, &l),
                scale_oracle(&u, &l, &m, t),
                "{k} primes: u = {u}, l = {l}"
            );
        }
        let s = (&m - BigUint::one()) >> (t + 1);
        let l = unit(&mut rng, 64, &m);
        for v in [
            BigInt::from(s.clone()),
            -BigInt::from(s.clone()),
            BigInt::from(&s + 1u32),
            -BigInt::from(&s + 1u32),
        ] {
            let u = image(&v, &l, &m).expect("l is a unit mod M");
            assert_eq!(
                classify(&crt, &mqrr, &u, &l),
                scale_oracle(&u, &l, &m, t),
                "{k} primes: v = {v} at the bound"
            );
        }
        // At a threshold past the last prime's bits, s lies below
        // p₀⋯p_(k−2): the bound falls in the forms `Low` and `High`, and so
        // do 0, 1, 2, and their negatives.
        let forced = u64::from(ps[k - 1].bit_width()) + 1;
        let mqrr = Mqrr::with_threshold(&m, forced);
        let s = (&m - BigUint::one()) >> (forced + 1);
        assert!(s < product(&ps[..k - 1]), "{k} primes: s = {s}");
        for v in [
            BigUint::zero(),
            BigUint::one(),
            BigUint::from(2u32),
            s.clone(),
            &s + 1u32,
        ] {
            for v in [BigInt::from(v.clone()), -BigInt::from(v)] {
                let u = image(&v, &l, &m).expect("l is a unit mod M");
                assert_eq!(
                    classify(&crt, &mqrr, &u, &l),
                    scale_oracle(&u, &l, &m, forced),
                    "{k} primes: v = {v} at the threshold {forced}"
                );
            }
        }
    }
}

/// Test 5, q-lift-growth-time.6: `scale_images` multiplies each image by
/// its residue modulo its prime, p − 1, 0, and 1 among both.
#[test]
fn scales_by_shoups_product() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b0d);
    let ps = mixed(&mut rng, 64);
    for round in 0..20 {
        let draw = |rng: &mut Xoshiro256PlusPlus, p: u64, i: usize| match (i + round) % 7 {
            0 => p - 1,
            1 => 0,
            2 => 1,
            _ => rng.next_u64() % p,
        };
        let images: Vec<u64> = ps
            .iter()
            .enumerate()
            .map(|(i, &p)| draw(&mut rng, p, i))
            .collect();
        let residues: Vec<u64> = ps
            .iter()
            .enumerate()
            .map(|(i, &p)| draw(&mut rng, p, i + 3))
            .collect();
        let factors: Vec<u64> = residues
            .iter()
            .zip(&ps)
            .map(|(&r, &p)| shoup_factor(r, p))
            .collect();
        let mut out = vec![u64::MAX; 3];
        scale_images(&images, &residues, &factors, &ps, &mut out);
        assert_eq!(out.len(), ps.len());
        for (i, &p) in ps.iter().enumerate() {
            let want = (u128::from(images[i]) * u128::from(residues[i]) % u128::from(p)) as u64;
            assert_eq!(out[i], want, "round {round}, prime {i}");
        }
    }
}

/// Test 6, q-lift-growth-time.10: a conversion of k images returns
/// k(k − 1)/2 + s(s − 1)/2 multiply-adds, s the digits its Horner's rule
/// reads: up to the highest digit not 0 of x, or of M − 1 − x where the
/// form is `High`, and every digit where it is `Any`.
#[test]
fn the_conversion_counts_its_products() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc0de);
    for k in [1, 2, 5, 33, 90] {
        let ps = mixed(&mut rng, k);
        let crt = crt_of(&ps);
        let m = crt.modulus().clone();
        let p = product(&ps[..k - 1]);
        for x in [
            BigUint::zero(),
            BigUint::one(),
            integer(&mut rng, 64) % &m,
            &p - 1u32,
            &m - 1u32,
            integer(&mut rng, m.bits()) % &m,
        ] {
            let images: Vec<u64> = ps.iter().map(|&q| word(&x, q)).collect();
            let (mut digits, mut out) = (Vec::new(), Vec::new());
            let (form, ops) = crt.convert(&images, &mut digits, &mut out);
            let read: Vec<u64> = match form {
                Converted::High => digits.iter().zip(&ps).map(|(&d, &q)| q - 1 - d).collect(),
                Converted::Low | Converted::Any => digits.clone(),
            };
            let s = match form {
                Converted::Any => k,
                Converted::Low | Converted::High => {
                    read.iter().rposition(|&d| d != 0).map_or(0, |i| i + 1)
                }
            };
            assert_eq!(
                ops,
                k * (k - 1) / 2 + s * s.saturating_sub(1) / 2,
                "{k} primes, x = {x}"
            );
        }
    }
}
