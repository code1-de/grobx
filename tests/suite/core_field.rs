//! Sub-item 1: the field contract with `Gf31`, its implementation for the
//! primes below 2^31 (f4-gfp.8, f4-gfp.9). Row operations are compared
//! with arithmetic in `u128`.

use grobx::field::{Field, Gf31};
use grobx::id::MonoId;

use crate::common::{P, P31, Rng};

fn field(p: u64) -> Gf31 {
    Gf31::new(p).unwrap_or_else(|| panic!("GF({p}) is a field the engine takes"))
}

#[test]
fn takes_exactly_the_primes_below_2_31() {
    let cases = [
        (0, false),
        (1, false),
        (2, true),
        (3, true),
        (4, false),
        (7, true),
        (P, true),
        (65535, false),
        (P31, true),
        (1u64 << 31, false),
        // the least prime above 2^32
        ((1u64 << 32) + 15, false),
        (u64::MAX, false),
    ];
    for (p, ok) in cases {
        assert_eq!(Gf31::new(p).is_some(), ok, "Gf31::new({p})");
    }
    assert_eq!(field(P).characteristic(), P);
}

#[test]
fn coefficients_reduce_and_read_back() {
    for p in [2, 7, P, P31] {
        let f = field(p);
        for v in [0, 1, p - 1, p, p + 1, 3 * p + 2, u64::MAX] {
            assert_eq!(f.value(f.coef(v)), v % p, "GF({p}): {v}");
        }
    }
}

#[test]
fn arithmetic_agrees_with_u128() {
    let mut rng = Rng::new(0xf1e1d);
    for p in [2, 3, 7, P, P31] {
        let f = field(p);
        for _ in 0..2000 {
            let (a, b) = (rng.below(p), rng.below(p));
            let (ca, cb) = (f.coef(a), f.coef(b));
            let (a, b, q) = (u128::from(a), u128::from(b), u128::from(p));
            assert_eq!(u128::from(f.value(f.add(ca, cb))), (a + b) % q);
            assert_eq!(u128::from(f.value(f.sub(ca, cb))), (a + q - b) % q);
            assert_eq!(u128::from(f.value(f.neg(ca))), (q - a) % q);
            assert_eq!(u128::from(f.value(f.mul(ca, cb))), a * b % q);
        }
    }
}

#[test]
fn every_unit_has_its_inverse() {
    let f = field(7);
    assert_eq!(f.inv(f.coef(0)), None);
    for a in 1..7 {
        let i = f.inv(f.coef(a)).expect("a unit");
        assert_eq!(f.value(f.mul(f.coef(a), i)), 1, "{a}·{a}⁻¹ over GF(7)");
    }
    let mut rng = Rng::new(0x1a7);
    for p in [P, P31] {
        let f = field(p);
        assert_eq!(f.inv(f.coef(p)), None);
        for _ in 0..1000 {
            let a = 1 + rng.below(p - 1);
            let i = f.inv(f.coef(a)).expect("a unit");
            assert_eq!(f.value(f.mul(f.coef(a), i)), 1, "{a} over GF({p})");
        }
    }
}

/// A dense row of `width` columns loaded with one sparse row, minus several
/// others times random multipliers, against the same sums in `u128`.
#[test]
fn row_operations_agree_with_u128() {
    let mut rng = Rng::new(0xd3e5e);
    for p in [2, 7, P, P31] {
        let f = field(p);
        for _ in 0..200 {
            let width = 1 + rng.below(60) as usize;
            let sparse = |rng: &mut Rng| -> (Vec<MonoId>, Vec<u64>) {
                let mut cols = Vec::new();
                let mut vals = Vec::new();
                for c in 0..width {
                    if rng.below(3) == 0 {
                        cols.push(MonoId::from_raw(c as u32));
                        vals.push(rng.below(p));
                    }
                }
                (cols, vals)
            };
            let mut want = vec![0u128; width];
            let mut dense = vec![<Gf31 as Field>::Acc::default(); width];
            let (cols, vals) = sparse(&mut rng);
            let coefs: Vec<_> = vals.iter().map(|&v| f.coef(v)).collect();
            // SAFETY: `sparse` draws its columns below `width`, the dense
            // row's length, one coefficient for each.
            unsafe { f.load(&mut dense, &cols, &coefs) };
            for (&c, &v) in cols.iter().zip(&vals) {
                want[c.index()] = u128::from(v);
            }
            for _ in 0..rng.below(40) {
                let (cols, vals) = sparse(&mut rng);
                let coefs: Vec<_> = vals.iter().map(|&v| f.coef(v)).collect();
                let m = rng.below(p);
                // SAFETY: as above.
                unsafe { f.sub_mul(&mut dense, &cols, &coefs, f.coef(m)) };
                for (&c, &v) in cols.iter().zip(&vals) {
                    let q = u128::from(p);
                    want[c.index()] =
                        (want[c.index()] + q * q - (u128::from(m) * u128::from(v)) % (q * q)) % q;
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

/// A million subtractions of (p − 1)·(p − 1) from one entry at the largest
/// prime the field takes do not overflow it (f4-gfp.8).
#[test]
fn the_dense_row_does_not_overflow() {
    let f = field(P31);
    let top = f.coef(P31 - 1);
    let mut dense = vec![<Gf31 as Field>::Acc::default(); 1];
    // SAFETY: column 0 is the dense row's one entry.
    unsafe { f.load(&mut dense, &[MonoId::from_raw(0)], &[top]) };
    for _ in 0..1_000_000 {
        // SAFETY: as above.
        unsafe { f.sub_mul(&mut dense, &[MonoId::from_raw(0)], &[top], top) };
    }
    // (p − 1) − 10^6·(p − 1)² ≡ −1 − 10^6 mod p
    let q = u128::from(P31);
    let want = (q - 1 + q * q - 1_000_000 % q) % q;
    assert_eq!(u128::from(f.value(f.take(&mut dense[0]))), want);
}
