//! The item's test code, `gfp-large-primes`: the field `Gf64` for the
//! primes from 2^31 to 2^64 (gfp-large-primes.1, .4 to .10). Primality is
//! checked against GNU `factor` (gfp-large-primes.11), arithmetic and row
//! operations against the same sums in `u128`.

use grobx::field::{Field, Gf64};
use grobx::id::MonoId;

use crate::common::Rng;
use crate::large::{LARGE, LARGEST, LEAST, factor_says_prime};

fn field(p: u64) -> Gf64 {
    Gf64::new(p).unwrap_or_else(|| panic!("GF({p}) is a field the engine takes"))
}

/// `Gf64::new(n)` is Some exactly where `factor` names n prime and
/// n ≥ 2^31, and its field has the characteristic n.
fn assert_takes_as_factor_says(numbers: &[u64]) {
    let primes = factor_says_prime(numbers);
    for (&n, prime) in numbers.iter().zip(primes) {
        let want = prime && n >= 1 << 31;
        let got = Gf64::new(n);
        assert_eq!(
            got.is_some(),
            want,
            "Gf64::new({n}): factor says prime {prime}"
        );
        if let Some(f) = got {
            assert_eq!(f.characteristic(), n);
        }
    }
}

#[test]
#[ignore = "calls factor"]
fn takes_exactly_the_primes_from_2_31_to_2_64() {
    let mut numbers: Vec<u64> = vec![
        0,
        1,
        2,
        3,
        4,
        65521,
        2_147_483_647,
        // 2^31, 2^32 + 1 = 641·6700417, and 2^64 − 1
        1 << 31,
        (1 << 32) + 1,
        u64::MAX,
        // squares of primes
        65537 * 65537,
        4_294_967_291 * 4_294_967_291,
        // two primes below 2^32, 2^32 − 5 and 2^32 − 17
        4_294_967_291 * 4_294_967_279,
        // Carmichael numbers (6k + 1)(12k + 1)(18k + 1): 727·1453·2179 and
        // 916837·1833673·2750509
        2_301_745_249,
        4_624_098_664_067_171_209,
        // strong pseudoprimes to the prime bases 2 to 7, 2 to 11, 2 to 13,
        // 2 to 19, and 2 to 31, which the next prime base, 11, 13, 17, 23,
        // and 37, tells from a prime
        3_215_031_751,
        2_152_302_898_747,
        3_474_749_660_383,
        341_550_071_728_321,
        3_825_123_056_546_413_051,
        // the Mersenne prime 2^61 − 1
        (1 << 61) - 1,
    ];
    numbers.extend(LARGE);
    for base in [1u64 << 31, 1 << 32, 1 << 62, 1 << 63] {
        numbers.extend((0..200).map(|k| base - 100 + k));
    }
    numbers.extend((0..200).map(|k| u64::MAX - k));
    assert_takes_as_factor_says(&numbers);
    for p in LARGE {
        assert!(
            Gf64::new(p).is_some(),
            "{p} is one of the test code's primes"
        );
    }
}

#[test]
#[ignore = "calls factor"]
fn agrees_with_factor_on_random_integers() {
    let mut rng = Rng::new(0xfac7);
    let numbers: Vec<u64> = (0..4000)
        .map(|k| {
            let n = (1u64 << 31) + rng.below(u64::MAX - (1 << 31));
            if k % 2 == 0 { n | 1 } else { n }
        })
        .collect();
    let primes = factor_says_prime(&numbers);
    assert!(
        primes.iter().filter(|&&p| p).count() >= 20,
        "the draw holds primes"
    );
    assert_takes_as_factor_says(&numbers);
}

#[test]
fn coefficients_reduce_and_read_back() {
    for p in LARGE {
        let f = field(p);
        for v in [0, 1, p - 1, p, p.saturating_add(1), u64::MAX] {
            assert_eq!(f.value(f.coef(v)), v % p, "GF({p}): {v}");
        }
    }
}

/// Random residues, and 0, 1, p − 2, and p − 1 among them.
fn residues(rng: &mut Rng, p: u64, k: usize) -> Vec<u64> {
    let mut out = vec![0, 1, p - 2, p - 1];
    out.extend((0..k).map(|_| rng.below(p)));
    out
}

#[test]
fn arithmetic_agrees_with_u128() {
    let mut rng = Rng::new(0x6f64);
    for p in LARGE {
        let f = field(p);
        let q = u128::from(p);
        let values = residues(&mut rng, p, 60);
        for &a in &values {
            for &b in &values {
                let (ca, cb) = (f.coef(a), f.coef(b));
                let (a, b) = (u128::from(a), u128::from(b));
                assert_eq!(u128::from(f.value(f.add(ca, cb))), (a + b) % q, "GF({p})");
                assert_eq!(
                    u128::from(f.value(f.sub(ca, cb))),
                    (a + q - b) % q,
                    "GF({p})"
                );
                assert_eq!(u128::from(f.value(f.mul(ca, cb))), a * b % q, "GF({p})");
            }
            assert_eq!(
                u128::from(f.value(f.neg(f.coef(a)))),
                (q - u128::from(a)) % q
            );
        }
    }
}

#[test]
fn every_unit_has_its_inverse() {
    let mut rng = Rng::new(0x1a764);
    for p in LARGE {
        let f = field(p);
        assert_eq!(f.inv(f.coef(0)), None);
        assert_eq!(f.inv(f.coef(p)), None);
        let mut units = vec![1, 2, p - 1];
        units.extend((0..500).map(|_| 1 + rng.below(p - 1)));
        for a in units {
            let i = f.inv(f.coef(a)).expect("a unit");
            assert!(f.value(i) < p);
            assert_eq!(f.value(f.mul(f.coef(a), i)), 1, "{a} over GF({p})");
        }
    }
}

/// A dense row of `width` columns loaded with one sparse row, minus up to
/// 39 others times multipliers of 0, 1, p − 1, and random ones, against the
/// same sums in `u128`.
#[test]
fn row_operations_agree_with_u128() {
    let mut rng = Rng::new(0xd3e64);
    for p in LARGE {
        let f = field(p);
        let q = u128::from(p);
        for round in 0..200 {
            let width = 1 + rng.below(60) as usize;
            let sparse = |rng: &mut Rng| -> (Vec<MonoId>, Vec<u64>) {
                let mut cols = Vec::new();
                let mut vals = Vec::new();
                for c in 0..width {
                    if rng.below(3) == 0 {
                        cols.push(MonoId::from_raw(c as u32));
                        // Extremes as often as random residues.
                        vals.push(match rng.below(4) {
                            0 => p - 1,
                            1 => 1,
                            _ => rng.below(p),
                        });
                    }
                }
                (cols, vals)
            };
            let mut want = vec![0u128; width];
            let mut dense = vec![<Gf64 as Field>::Acc::default(); width];
            let (cols, vals) = sparse(&mut rng);
            let coefs: Vec<_> = vals.iter().map(|&v| f.coef(v)).collect();
            // SAFETY: `sparse` draws its columns below `width`, the dense
            // row's length, one coefficient for each.
            unsafe { f.load(&mut dense, &cols, &coefs) };
            for (&c, &v) in cols.iter().zip(&vals) {
                want[c.index()] = u128::from(v);
            }
            for k in 0..rng.below(40) {
                let (cols, vals) = sparse(&mut rng);
                let coefs: Vec<_> = vals.iter().map(|&v| f.coef(v)).collect();
                let m = match (round + k) % 5 {
                    0 => 0,
                    1 => 1,
                    2 => p - 1,
                    _ => rng.below(p),
                };
                // SAFETY: as above.
                unsafe { f.sub_mul(&mut dense, &cols, &coefs, f.coef(m)) };
                for (&c, &v) in cols.iter().zip(&vals) {
                    let product = u128::from(m) * u128::from(v) % q;
                    want[c.index()] = (want[c.index()] + q - product) % q;
                }
            }
            for c in 0..width {
                assert_eq!(
                    u128::from(f.value(f.take(&mut dense[c]))),
                    want[c],
                    "GF({p}), column {c}"
                );
            }
            // take leaves every entry zero
            for entry in &mut dense {
                assert_eq!(f.value(f.take(entry)), 0);
            }
        }
    }
}

/// A million subtractions of (p − 1)·(p − 1) from one entry stay exact on
/// either side of 2^63 and at the ends of the range (gfp-large-primes.4).
#[test]
fn long_accumulations_stay_exact() {
    for p in [
        LEAST,
        9_223_372_036_854_775_783,
        9_223_372_036_854_775_837,
        LARGEST,
    ] {
        let f = field(p);
        let top = f.coef(p - 1);
        let mut dense = vec![<Gf64 as Field>::Acc::default(); 1];
        // SAFETY: column 0 is the dense row's one entry.
        unsafe { f.load(&mut dense, &[MonoId::from_raw(0)], &[top]) };
        for _ in 0..1_000_000 {
            // SAFETY: as above.
            unsafe { f.sub_mul(&mut dense, &[MonoId::from_raw(0)], &[top], top) };
        }
        // (p − 1) − 10^6·(p − 1)² ≡ −1 − 10^6 mod p
        let q = u128::from(p);
        let want = (q - 1 + q - 1_000_000 % q) % q;
        assert_eq!(u128::from(f.value(f.take(&mut dense[0]))), want, "GF({p})");
    }
}
