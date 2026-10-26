//! q-attempt-time, the item: an attempt's conversion ends at its first
//! digit 0 or p − 1 below the top and converts in full where its value
//! does not fit, MQRR tries the rational before its first quotient of at
//! least 2^T first, and a part's chain grows by the denominator MQRR
//! finds, with every outcome the lift returned before (plan.md, _Test
//! code_).

use std::fmt::Write;
use std::time::Duration;

use grobx_checker::Order;
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Params;
use crate::catch_up_tests::L1;
use crate::chain_tests::{assert_accepted, basis, both_reports, params, with_cofactors};
use crate::lehmer::{First, Mqrr, threshold};
use crate::lift_growth_tests::{coprime, primes_for, wide};
use crate::mixed_radix_tests::{crt_of, from_limbs, mixed, word};
use crate::mqrr_tests::{continued, limbs, of_bits, primes};
use crate::rational::{Converted, Crt, Rational};
use crate::shortcut_tests::{image, integer, product};

/// x = Σ dᵢ·p₀⋯pᵢ₋₁ over the digits given, one a prime of `ps`.
fn from_digits(digits: &[u64], ps: &[u64]) -> BigUint {
    let mut x = BigUint::zero();
    let mut weight = BigUint::one();
    for (&d, &p) in digits.iter().zip(ps) {
        x += &weight * d;
        weight *= p;
    }
    x
}

/// A digit at p that is neither 0 nor p − 1.
fn inner(rng: &mut Xoshiro256PlusPlus, p: u64) -> u64 {
    1 + rng.next_u64() % (p - 2)
}

/// One conversion's outcome: its form, its limbs, its digits, and its
/// products.
type Conversion = (Converted, Vec<u64>, Vec<u64>, usize);

/// The ended conversion of x's images at `crt`'s primes, and the full
/// one; neither has a zero limb at the top.
fn conversions(crt: &Crt, x: &BigUint) -> (Conversion, Conversion) {
    let images: Vec<u64> = crt.primes().iter().map(|&p| word(x, p)).collect();
    let run = |ended: bool| -> Conversion {
        let (mut digits, mut out) = (Vec::new(), Vec::new());
        let (form, products) = if ended {
            crt.convert_small(&images, &mut digits, &mut out)
        } else {
            crt.convert(&images, &mut digits, &mut out)
        };
        assert_ne!(out.last(), Some(&0), "a zero limb at the top of {x}");
        (form, out, digits, products)
    };
    (run(true), run(false))
}

/// Test 1, q-attempt-time.4: at 1 to 300 primes, below 2^31 and of
/// [2^61, 2^62) in turn. x of s digits that are neither 0 nor p − 1, and
/// 0 past them, and M − 1 − x: below the top the ended conversion stops
/// at digit s, `Low` or `High` with x's limbs, after s digits and s²
/// products; at the top it is the full one. A digit 0 or pⱼ − 1 below
/// the top of a value that is not small ends it at j, with the limbs the
/// digits before j name. A value with no such digit converts in full.
#[test]
fn the_ended_conversion() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0e4d);
    for k in [1, 2, 3, 40, 300] {
        let ps = mixed(&mut rng, k);
        let crt = crt_of(&ps);
        let m = crt.modulus().clone();
        let counts: Vec<usize> = if k <= 40 {
            (0..=k).collect()
        } else {
            (0..k - 2).step_by(23).chain(k - 2..=k).collect()
        };
        for s in counts {
            let digits: Vec<u64> = ps[..s].iter().map(|&p| inner(&mut rng, p)).collect();
            let x = from_digits(&digits, &ps);
            for (value, form) in [
                (x.clone(), Converted::Low),
                (&m - 1u32 - &x, Converted::High),
            ] {
                let case = format!("{k} primes, {s} digits, {form:?}");
                let (ended, full) = conversions(&crt, &value);
                if s + 1 < k {
                    assert_eq!(ended.0, form, "{case}: the form");
                    assert_eq!(from_limbs(&ended.1), x, "{case}: the limbs");
                    assert_eq!(ended.2.len(), s, "{case}: the digits taken");
                    assert_eq!(ended.3, s * s, "{case}: the products");
                } else {
                    assert_eq!(ended, full, "{case}: the full conversion");
                }
            }
        }
        for j in [0, 1, k / 2, k.saturating_sub(2)] {
            if j + 1 >= k {
                continue;
            }
            for end in [0, 1] {
                let mut digits: Vec<u64> = ps.iter().map(|&p| inner(&mut rng, p)).collect();
                digits[j] = if end == 0 { 0 } else { ps[j] - 1 };
                let x = from_digits(&digits, &ps);
                let which = ["0", "p − 1"][end];
                let case = format!("{k} primes, digit {j} {which}");
                let (form, below) = if end == 0 {
                    (Converted::Low, from_digits(&digits[..j], &ps))
                } else {
                    let complement: Vec<u64> = digits[..j]
                        .iter()
                        .zip(&ps)
                        .map(|(&d, &p)| p - 1 - d)
                        .collect();
                    (Converted::High, from_digits(&complement, &ps))
                };
                let (ended, full) = conversions(&crt, &x);
                assert_eq!(ended.0, form, "{case}: the form");
                assert_eq!(from_limbs(&ended.1), below, "{case}: the limbs");
                assert_eq!(ended.2.len(), j, "{case}: the digits taken");
                assert_eq!(ended.3, j * j, "{case}: the products");
                assert_eq!(full.0, Converted::Any, "{case}: the full form");
                assert_eq!(from_limbs(&full.1), x, "{case}: the full value");
                assert_eq!(full.2, digits, "{case}: every digit");
            }
        }
        for i in 0..8 {
            let x = integer(&mut rng, m.bits()) % &m;
            let (ended, full) = conversions(&crt, &x);
            let edge = full.2[..k - 1]
                .iter()
                .zip(&ps)
                .any(|(&d, &p)| d == 0 || d == p - 1);
            if !edge {
                assert_eq!(ended, full, "{k} primes, uniform {i}: x = {x}");
            }
        }
    }
}

/// The test code's oracle of `Mqrr::first`: the plain Euclidean algorithm
/// on (m, v) to its first quotient q = ⌊r₀/r₁⌋ of at least 2^t, and the
/// rational r₁/t₁ before it, its denominator made positive: `Nothing`
/// where no quotient reaches 2^t, `Shared` where gcd(r₁, t₁) ≠ 1. v in
/// [1, m).
fn first_oracle(v: &BigUint, m: &BigUint, t: u64) -> First {
    let floor = BigInt::one() << t;
    let (mut r0, mut r1) = (BigInt::from(m.clone()), BigInt::from(v.clone()));
    let (mut t0, mut t1) = (BigInt::zero(), BigInt::one());
    while !r1.is_zero() {
        let q = &r0 / &r1;
        if q >= floor {
            let (n, d) = match t1.sign() {
                Sign::Minus => (-r1, -t1),
                Sign::NoSign | Sign::Plus => (r1, t1),
            };
            if !n.gcd(&d).is_one() {
                return First::Shared;
            }
            return First::Rational(Rational {
                num: n,
                den: d.to_biguint().expect("d > 0"),
            });
        }
        let r = &r0 - &q * &r1;
        (r0, r1) = (r1, r);
        let s = &t0 - &q * &t1;
        (t0, t1) = (t1, s);
    }
    First::Nothing
}

/// `Mqrr::with_threshold(m, t).first(v)` is the oracle's answer, and where
/// it is `Nothing`, `reconstruct` finds nothing either.
pub(crate) fn agrees(m: &BigUint, v: &BigUint, t: u64, case: &str) -> First {
    let mqrr = Mqrr::with_threshold(m, t);
    let got = mqrr.first(&limbs(v));
    assert_eq!(
        got,
        first_oracle(v, m, t),
        "{case}: v = {v}, M = {m}, t = {t}"
    );
    match got {
        First::Nothing => assert_eq!(
            mqrr.reconstruct(&limbs(v)),
            None,
            "{case}: v = {v}, M = {m}, t = {t}"
        ),
        First::Rational(_) | First::Shared => {}
    }
    got
}

/// Quotients 1 to 3, `n` of them.
pub(crate) fn small(rng: &mut Xoshiro256PlusPlus, n: usize) -> Vec<BigUint> {
    (0..n)
        .map(|_| BigUint::from(1 + rng.next_u64() % 3))
        .collect()
}

/// Test 2, q-attempt-time.3: chosen quotients, the large ones apart by
/// 300 small ones or by one, against MQRR's largest: two equal, the same
/// rational; a larger after a large one, past 2^64, and within one run,
/// the rational before the first; both below the threshold, `Nothing`. A
/// first large quotient whose rational's terms share a factor, `Shared`.
/// Residues of rationals and uniform ones modulo products of 1 to 290
/// primes, and every v below small odd moduli: the oracle's answer.
#[test]
fn the_first_large_quotient() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xf125);
    let big = |bits: u64| BigUint::one() << bits;
    for (case, large, apart, t) in [
        ("two equal", [big(40), big(40)], 300, 30),
        ("a larger later", [big(30), big(45)], 300, 20),
        ("past 2^64", [big(70), big(90)], 300, 32),
        ("below the threshold", [big(40), big(41)], 300, 50),
        ("within one run", [big(12), big(14)], 1, 10),
    ] {
        let mut a = small(&mut rng, 300);
        a.push(large[0].clone());
        a.extend(small(&mut rng, apart));
        a.push(large[1].clone());
        a.extend(small(&mut rng, 300));
        a.push(BigUint::from(2u32));
        let (v, m) = continued(&a);
        let first = agrees(&m, &v, t, case);
        let largest = Mqrr::with_threshold(&m, t).reconstruct(&limbs(&v));
        match case {
            "below the threshold" => assert_eq!(first, First::Nothing, "{case}"),
            "two equal" => assert_eq!(first, First::Rational(largest.unwrap()), "{case}"),
            _ => match first {
                First::Rational(r) => assert_ne!(Some(r), largest, "{case}"),
                First::Nothing | First::Shared => panic!("{case}: {first:?}"),
            },
        }
    }

    let mut shared = false;
    for _ in 0..1000 {
        let mut a = small(&mut rng, 40);
        a.push(big(30));
        a.extend(small(&mut rng, 40));
        a.push(BigUint::from(2u32));
        let (v, m) = continued(&a);
        match first_oracle(&v, &m, 20) {
            First::Shared => {
                assert_eq!(agrees(&m, &v, 20, "shared"), First::Shared);
                shared = true;
                break;
            }
            First::Nothing | First::Rational(_) => {}
        }
    }
    assert!(
        shared,
        "no first large quotient whose rational's terms share a factor"
    );

    for low in [30, 61] {
        for k in [1, 2, 4, 13, 44, 100, 290] {
            let ps = primes(&mut rng, k, low);
            let m = product(&ps);
            let own = threshold(&m);
            for i in 0..8 {
                let t = [own, 1, 64, own + 20][i % 4];
                let case = format!("{k} primes of 2^{low}, case {i}");
                let bits = (m.bits() + 4).saturating_sub(t + rng.next_u64() % 9).max(2);
                let bn = 1 + rng.next_u64() % (bits - 1);
                let n = BigInt::from(of_bits(&mut rng, bn));
                let n = if i % 2 == 0 { n } else { -n };
                let d = of_bits(&mut rng, bits - bn);
                if let Some(v) = image(&n, &d, &m)
                    && !v.is_zero()
                {
                    agrees(&m, &v, t, &case);
                }
                let v = integer(&mut rng, m.bits()) % &m;
                if !v.is_zero() {
                    agrees(&m, &v, t, &format!("{case}, uniform"));
                }
            }
        }
    }

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

/// The text of lead·y + Σ cᵢ·xᵢ over ℚ in the variables y, x1, …, xₙ,
/// and its basis text where its integers have no common factor and lead
/// is positive.
fn linear(lead: &BigUint, coefs: &[BigInt]) -> (String, String) {
    let mut poly = format!("{lead}*y");
    for (i, c) in coefs.iter().enumerate() {
        let sign = if c.is_negative() { '-' } else { '+' };
        let (magnitude, index) = (c.magnitude(), i + 1);
        write!(poly, "{sign}{magnitude}*x{index}").expect("a String takes any text");
    }
    let vars: Vec<String> = (1..=coefs.len()).map(|i| format!("x{i}")).collect();
    (
        format!("y,{names}\n0\n{poly}\n", names = vars.join(",")),
        format!("[{poly}]:\n"),
    )
}

/// Test 3, q-attempt-time.1 to .3: B·y + Σ aᵢ·(B/bᵢ)·xᵢ, B the lcm of the
/// bᵢ, whose basis has the coefficients aᵢ/bᵢ: each brings its chain a
/// new denominator, prime to the chain's last or sharing 2, 3, 5, or 7
/// with it. The basis through both lifts, every coefficient tried by
/// MQRR, at most two primes past the most MQRR asks for at |aᵢ|·B, and
/// with cofactors the checker accepts. Beside it, f = 1: at the attempt
/// at w₃, over M = w₀·w₁·w₂, v = n·b lies in (M/2^(T+1), M/2^T], so MQRR
/// takes n·b over b's chain, and the lift combines w₀ to w₃ alone. n·b
/// is drawn in that range: at its edge v would be about n″/2^(T+1) for a
/// small n″, a rational whose larger quotient MQRR would take.
#[test]
fn a_chain_that_grows_at_each_coefficient() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc4a1);
    let mut q: Vec<BigUint> = Vec::new();
    for _ in 0..4 {
        let mut others: Vec<&BigUint> = q.iter().collect();
        let small_primes = BigUint::from(105u32);
        others.push(&small_primes);
        let next = coprime(&mut rng, 300, &others);
        q.push(next);
    }
    let sharing: Vec<BigUint> = q
        .iter()
        .zip([6u32, 10, 15, 21])
        .map(|(x, g)| x * g)
        .collect();
    for (case, dens) in [
        ("prime to each other", q.clone()),
        ("sharing factors", sharing),
    ] {
        let b = dens.iter().fold(BigUint::one(), |l, d| l.lcm(d));
        let nums: Vec<BigInt> = dens
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let a = BigInt::from(coprime(&mut rng, 300, &[d]));
                if i % 2 == 1 { -a } else { a }
            })
            .collect();
        let coefs: Vec<BigInt> = nums
            .iter()
            .zip(&dens)
            .map(|(a, d)| a * BigInt::from(&b / d))
            .collect();
        let (text, want) = linear(&b, &coefs);
        let bits = nums.iter().map(BigInt::bits).max().unwrap() + b.bits();
        let (_, most) = primes_for(bits);
        for params in [params(21), with_cofactors(21)] {
            let cofactors = params.cofactors;
            let case = format!("{case}, cofactors {cofactors}");
            let (outcome, chain, _) = both_reports(&case, &text, &params, &[]);
            assert_eq!(basis(&outcome), want, "{case}");
            assert_eq!(chain.reconstructed, chain.euclid, "{case}: {chain:?}");
            assert!(chain.combined.len() <= most + 2, "{case}: {chain:?}");
            if params.cofactors {
                assert_accepted(&case, &text, &outcome, &Order::Grevlex);
            }
        }
    }

    let ws = wide(4, 31_000);
    let m = product(&ws[..3]);
    let t = threshold(&m);
    let room = m.bits() - t - 20;
    let b = coprime(&mut rng, room / 2, &[]);
    let a = BigInt::from(coprime(&mut rng, room - room / 2 - 1, &[&b]));
    let (low, high) = (&m / (&b << (t + 1)) + 1u32, &m / (&b << t));
    let n = &low + integer(&mut rng, high.bits()) % (&high - &low);
    let (text, want) = linear(&b, &[a, BigInt::from(&n * &b)]);
    let forced = [L1, ws[0], ws[1], ws[2], ws[3]];
    for params in [params(22), with_cofactors(22)] {
        let cofactors = params.cofactors;
        let case = format!("f = 1, cofactors {cofactors}");
        let (outcome, chain, _) = both_reports(&case, &text, &params, &forced);
        assert_eq!(basis(&outcome), want, "{case}");
        assert_eq!(chain.combined, ws, "{case}: {chain:?}");
        if params.cofactors {
            assert_accepted(&case, &text, &outcome, &Order::Grevlex);
        }
    }
}

/// `p` with a deadline of 20 s.
fn within_20_s(mut p: Params) -> Params {
    p.deadline = Some(Duration::from_secs(20));
    p
}

/// Test 4, q-attempt-time.4, .5: y ± c·x with the wide primes w₀ to w₇
/// forced: c's digit 1 is 0, or w₁ − 1, its digit 0 neither, and c of
/// about 392 bits is small at seven primes and not reached at six. The
/// attempt at w₇ ends its conversion at digit 1, its value does not fit,
/// and the full conversion keeps c: the basis through both lifts within
/// 20 s, the lift combines w₀ to w₇, and with cofactors the checker
/// accepts.
#[test]
fn a_digit_below_the_top() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xd161);
    let ws = wide(8, 47_000);
    let (w0, w1) = (BigUint::from(ws[0]), BigUint::from(ws[1]));
    let e = of_bits(&mut rng, 270);
    let r = BigUint::from(2 + rng.next_u64() % (ws[0] - 4));
    let zero = &r + &w0 * &w1 * &e;
    let top = &r + (&w1 - 1u32) * &w0 + &w0 * &w1 * &e;
    let mut forced = vec![L1];
    forced.extend(&ws);
    for (case, c) in [("digit 1 is 0", zero), ("digit 1 is w₁ − 1", top)] {
        for sign in ['+', '-'] {
            let text = format!("y,x\n0\ny{sign}{c}*x\n");
            let want = format!("[y{sign}{c}*x]:\n");
            for params in [within_20_s(params(23)), within_20_s(with_cofactors(23))] {
                let cofactors = params.cofactors;
                let case = format!("{case}, {sign}, cofactors {cofactors}");
                let (outcome, chain, _) = both_reports(&case, &text, &params, &forced);
                assert_eq!(basis(&outcome), want, "{case}");
                assert_eq!(chain.combined, ws, "{case}: {chain:?}");
                if params.cofactors {
                    assert_accepted(&case, &text, &outcome, &Order::Grevlex);
                }
            }
        }
    }
}
