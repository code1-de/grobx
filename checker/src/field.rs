//! GF(p) arithmetic on residues in [0, p), for a prime p below 2^64, and
//! the test that a characteristic is such a prime.

use crate::narrow::Narrow;

/// The decimal digits a chunk of a number holds as the checker reads it:
/// the most a u64 holds whatever they are, as 10^19 − 1 lies below 2^64.
pub(crate) const DIGITS: usize = 19;

// checker-gfp.4 cited: dumas-2008 §2.1.1 -- multiplication is followed by machine remaindering while division is performed via the extended gcd algorithm
#[derive(Clone, Copy, Debug)]
pub struct Field {
    p: u64,
}

impl Field {
    pub fn new(p: u64) -> Field {
        debug_assert!(p >= 2);
        Field { p }
    }

    pub fn add(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.p && b < self.p, "a, b < p");
        let (s, carry) = a.overflowing_add(b);
        if carry || s >= self.p {
            s.wrapping_sub(self.p)
        } else {
            s
        }
    }

    pub fn neg(self, a: u64) -> u64 {
        debug_assert!(a < self.p, "a < p");
        if a == 0 { 0 } else { self.p - a }
    }

    pub fn mul(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.p && b < self.p, "a, b < p");
        ((u128::from(a) * u128::from(b)) % u128::from(self.p)).narrow()
    }

    /// a·10^k + d for a < p, the step of reading k more decimal digits d.
    pub fn shift_add(self, a: u64, k: u32, d: u64) -> u64 {
        debug_assert!(a < self.p, "a < p");
        debug_assert!(
            10u64.checked_pow(k).is_some_and(|ten| d < ten),
            "d < 10^k, which a u64 holds"
        );
        ((u128::from(a) * 10u128.pow(k) + u128::from(d)) % u128::from(self.p)).narrow()
    }

    /// The integer whose decimal digits are `digits`, reduced mod p `DIGITS`
    /// digits at a time.
    pub fn decimal(self, digits: &[u8]) -> u64 {
        let mut acc = 0;
        for chunk in digits.chunks(DIGITS) {
            let d = chunk.iter().fold(0u64, |d, b| d * 10 + u64::from(b - b'0'));
            acc = self.shift_add(acc, chunk.len().narrow(), d);
        }
        acc
    }

    /// The natural number whose 64-bit digits are `digits`, the most
    /// significant first, reduced mod p.
    pub fn limbs(self, digits: impl Iterator<Item = u64>) -> u64 {
        digits.fold(0, |acc, d| {
            ((u128::from(acc) << 64 | u128::from(d)) % u128::from(self.p)).narrow()
        })
    }

    /// a^e by squaring.
    pub(crate) fn pow(self, mut a: u64, mut e: u64) -> u64 {
        debug_assert!(a < self.p, "a < p");
        let mut x = 1;
        while e > 0 {
            if e & 1 == 1 {
                x = self.mul(x, a);
            }
            a = self.mul(a, a);
            e >>= 1;
        }
        x
    }

    pub fn modulus(self) -> u64 {
        self.p
    }

    /// The inverse of a, or None where gcd(a, p) ≠ 1.
    pub fn inv(self, a: u64) -> Option<u64> {
        debug_assert!(a < self.p, "a < p");
        let (mut r0, mut r1) = (i128::from(self.p), i128::from(a));
        let (mut t0, mut t1) = (0i128, 1i128);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        if r0 != 1 {
            return None;
        }
        Some(t0.rem_euclid(i128::from(self.p)).narrow())
    }
}

/// The first twelve primes: the trial divisors and the bases of `is_prime`.
const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

// instance-grammar-agreement.4 cited: sorenson-2017 §1, Theorem 1.1 -- ψ₁₂ = 318665857834031151167461, the least composite that is a strong pseudoprime to each of the first 12 prime bases, lies past 2^64; miller-1976 _Tests for primality_, the algorithm A_f -- a base a with a^(n−1) ≢ 1 mod n, or with ((a^((n−1)/2^k) mod n) − 1, n) ≠ 1, n for some k, shows n composite
/// Whether n is prime: no prime of `BASES` divides n but n itself, and n
/// is a strong probable prime to each of them as a base.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if let Some(&q) = BASES.iter().find(|&&q| n.is_multiple_of(q)) {
        return n == q;
    }
    let field = Field { p: n };
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    BASES.iter().all(|&a| {
        let mut x = field.pow(a, d);
        if x == 1 || x == n - 1 {
            return true;
        }
        for _ in 1..s {
            x = field.mul(x, x);
            if x == n - 1 {
                return true;
            }
        }
        false
    })
}
