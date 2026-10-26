//! The integers of the lift over Q: the Chinese remainder steps, each of
//! which makes a prime's constants of the conversion of an integer's
//! images at the primes to its residue modulo M by its mixed-radix
//! digits, that conversion, the residues of integers and rationals modulo
//! a prime, rational numbers, and the texts over Q. Wang's bound and the test
//! code's oracle for Wang's reconstruction are here too, both the test
//! code's, and so is the scaling of a polynomial over Q to integers with no
//! common factor, which the lift's basis text holds.

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::Stop;
use crate::budget::Budget;
use crate::field::{inverse_mod, mul_mod, shoup_factor, shoup_mul};
use crate::narrow::Narrow;
use crate::text::{
    TERM_MARKS, decimal_units, digits_bound, monomial_longest, push_big, push_monomial,
};

// q-lift.9 cited: arnold-2003 §6.3 -- apply the Chinese remainder algorithm to the coefficients of G_k and G_p, and G^(kp) = G_kp; idrees-2011 §2 -- lift GP to G_N by the Chinese remainder algorithm, then pull back by the Farey rational map
/// The product M of the primes combined so far, those primes pᵢ, and
/// the constants of their mixed-radix conversion.
pub(crate) struct Crt {
    modulus: BigUint,
    primes: Vec<u64>,
    // q-lift-growth-time.3 design -- each prime's step makes the residues of the products of the primes before it at that prime, and the lifting keeps them as a triangle, not the row made at each conversion (not measured)
    /// Row i, at offset i(i − 1)/2, holds (p₀⋯pⱼ₋₁) mod pᵢ for j < i,
    /// its entry 0 equal to 1: the rows of the steps pushed.
    rows: Vec<u64>,
    /// Per prime, the constants of its digit.
    digits: Vec<Digit>,
}

/// The constants of the mixed-radix digit at a prime pᵢ.
#[derive(Clone, Copy)]
struct Digit {
    /// (p₀⋯pᵢ₋₁)⁻¹ mod pᵢ.
    inverse: u64,
    /// The inverse's Shoup factor, `shoup_factor(inverse, pᵢ)`.
    factor: u64,
    /// 2^128 mod pᵢ.
    top: u64,
}

/// The form in which a conversion gives the x in [0, M) the images name,
/// M the product of the primes converted at, by the mixed-radix digit dⱼ
/// that ended it, P = p₀⋯pⱼ₋₁ the product of the primes before that
/// digit's: the top digit for `Crt::convert`, and for `Crt::convert_small`
/// the first below the top that is 0 or pⱼ − 1, or else the top digit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Converted {
    /// dⱼ is 0: the limbs hold x mod P, which is x where every digit from
    /// j on is 0, as the top digit's is for `Crt::convert`.
    Low,
    /// dⱼ is pⱼ − 1: the limbs hold (M − 1 − x) mod P, which is M − 1 − x
    /// where every digit from j on is the prime less 1.
    High,
    /// The top digit ended it and is neither: the limbs hold x.
    Any,
}

impl Crt {
    /// M = 1, no prime.
    pub(crate) fn new() -> Crt {
        Crt {
            modulus: BigUint::one(),
            primes: Vec::new(),
            rows: Vec::new(),
            digits: Vec::new(),
        }
    }

    pub(crate) fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    pub(crate) fn primes(&self) -> &[u64] {
        &self.primes
    }

    /// The step from M to M·p, for a prime p below 2^62 not among `primes`,
    /// whose radix reduces numbers of at most `width` limbs besides those
    /// below M.
    pub(crate) fn step(&self, p: u64, width: usize) -> CrtStep {
        debug_assert!(!self.primes.contains(&p), "{p} is combined already");
        CrtStep::new(self, p, width)
    }

    /// M becomes M·p, `step`'s prime: the images of every open element at
    /// p have been appended. The step's row and constants join the
    /// conversion's.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the step is spent: taken by value, it cannot be pushed twice"
    )]
    pub(crate) fn push(&mut self, step: CrtStep) {
        debug_assert_eq!(step.row.len(), self.primes.len(), "the step is from M");
        self.modulus *= step.prime;
        self.primes.push(step.prime);
        self.rows.extend_from_slice(&step.row);
        self.digits.push(Digit {
            inverse: step.inverse,
            factor: step.factor,
            top: step.radix.top,
        });
    }

    // q-lift-growth-time.2 cited: garner-1959 §Conversion from a residue code to a normal number representation, (26) to (41) -- the digits of the mixed-radix form come by residue arithmetic alone, each from its residue less the digits before it times their weights, times the inverse of its own weight, and S = a_t + m_t(d_t + m_{t−1}(d_{t−1} + ⋯))
    /// The x in [0, Mₖ) with x ≡ vᵢ mod pᵢ, Mₖ the product of the first k
    /// primes, k = |v|, 1 ≤ k ≤ the primes, each vᵢ below its prime: x's
    /// mixed-radix digits dᵢ, x = d₀ + p₀·(d₁ + p₁·(d₂ + ⋯)), each
    /// below its prime, to `digits`, and the limbs the form names to
    /// `limbs`, with no zero limb at the top. Beside the form, k(k − 1)/2 +
    /// s(s − 1)/2 for the s digits Horner's rule reads: the products of the
    /// digits, and a bound on those of the rule, whose number before digit
    /// j holds at most s − 1 − j limbs.
    pub(crate) fn convert(
        &self,
        v: &[u64],
        digits: &mut Vec<u64>,
        limbs: &mut Vec<u64>,
    ) -> (Converted, usize) {
        self.mixed_radix(v, digits, limbs, false)
    }

    // q-attempt-time.4 invented -- an attempt's conversion ends at the first digit below the top that is 0 or p_j − 1, and Horner's rule reads the digits before it as Low and High read them, not every digit computed, nor every digit computed with the sums of a run of top digits 0 or p_j − 1 cut by a telescoping sum (log 0085-1)
    /// As `convert`, but it ends at the first digit dⱼ, j < k − 1, that is
    /// 0 or pⱼ − 1: `Low` with the limbs of x mod p₀⋯pⱼ₋₁ where dⱼ = 0,
    /// and `High` with those of (M − 1 − x) mod p₀⋯pⱼ₋₁ where dⱼ = pⱼ − 1
    /// (`Converted`). Where no digit below the top is 0 or pᵢ − 1, what
    /// `convert` returns. `digits` holds the digits before the one that
    /// ended it, so it ended below the top exactly where it holds fewer
    /// than k. Beside the form, j(j + 1)/2 + s(s − 1)/2 for an end at j <
    /// k − 1, s the digits Horner's rule reads, and what `convert` counts
    /// at the top.
    pub(crate) fn convert_small(
        &self,
        v: &[u64],
        digits: &mut Vec<u64>,
        limbs: &mut Vec<u64>,
    ) -> (Converted, usize) {
        self.mixed_radix(v, digits, limbs, true)
    }

    /// `convert`, and where `edge`, `convert_small`.
    fn mixed_radix(
        &self,
        v: &[u64],
        digits: &mut Vec<u64>,
        limbs: &mut Vec<u64>,
        edge: bool,
    ) -> (Converted, usize) {
        let k = v.len();
        debug_assert!((1..=self.primes.len()).contains(&k), "{k} images");
        digits.clear();
        // dᵢ = (vᵢ − Σ_{j<i} dⱼ·(p₀⋯pⱼ₋₁))·(p₀⋯pᵢ₋₁)⁻¹ mod pᵢ: x mod pᵢ
        // is the sum over the digits up to i, as pᵢ divides the weight of
        // every later one. d₀ = v₀.
        let mut offset = 0;
        // The form of an end below the top, at a digit 0 or pᵢ − 1.
        let mut ended = None;
        for (i, (&image, &p)) in v.iter().zip(&self.primes).enumerate() {
            debug_assert!(image < p, "the image {image} at {p}");
            let row = &self.rows[offset..offset + i];
            offset += i;
            let Digit {
                inverse,
                factor,
                top,
            } = self.digits[i];
            // A digit and a row's entry lie below 2^62, their product below
            // 2^124.
            let s = wrapped_sum(digits.iter().copied().zip(row.iter().copied()), p, top);
            let t = if image >= s { image - s } else { image + p - s };
            // q-lift-growth-time.6 cited: vanderhoeven-2016 §2.2.2, Function 8 and Proposition 4 -- for a fixed multiplicand y precompute ψ := ⌊2^n y/p⌋, c = (xψ) >> n, d = xy − cp, one correction
            let digit = shoup_mul(t, inverse, factor, p);
            if edge && i + 1 < k && (digit == 0 || digit == p - 1) {
                ended = Some(if digit == 0 {
                    Converted::Low
                } else {
                    Converted::High
                });
                break;
            }
            digits.push(digit);
        }
        // The digits computed: those taken, and the one that ended the
        // conversion below the top.
        let computed = digits.len() + usize::from(ended.is_some());
        let form = ended.unwrap_or_else(|| match digits[k - 1] {
            0 => Converted::Low,
            d if d == self.primes[k - 1] - 1 => Converted::High,
            _ => Converted::Any,
        });
        let primes = &self.primes[..digits.len()];
        // q-lift-growth-time.7 design -- Horner's rule runs from the highest digit that is not 0 where the top digit is 0, and over the complement's digits p_i − 1 − d_i where the top digit is the last prime less 1, so a value below p_0⋯p_{k−2}, or within it of M, reads only its own digits, not Horner's rule over every digit, nor the digits' sums cut at a run of top digits 0 or p_j − 1 by a telescoping sum (log 0083-1)
        // Σ (pⱼ − 1)·p₀⋯pⱼ₋₁ = M − 1, so the digits pⱼ − 1 − dⱼ are those
        // of M − 1 − x.
        let read = match form {
            Converted::Low | Converted::Any => horner(primes, |j| digits[j], limbs),
            Converted::High => horner(primes, |j| primes[j] - 1 - digits[j], limbs),
        };
        (
            form,
            computed * (computed - 1) / 2 + read * read.saturating_sub(1) / 2,
        )
    }
}

/// Σ eⱼ·(p₀⋯pⱼ₋₁) over the digits eⱼ = `digit(j)`, j below the
/// primes given, into `out` on limbs with no zero limb at the top, by
/// Horner's rule from the highest digit that is not 0; the count of
/// digits read, that digit's index plus 1, or 0 where every digit is 0.
/// Each digit lies below its prime, and each prime below 2^62.
fn horner(primes: &[u64], digit: impl Fn(usize) -> u64, out: &mut Vec<u64>) -> usize {
    out.clear();
    let Some(s) = (0..primes.len()).rev().find(|&j| digit(j) != 0) else {
        return 0;
    };
    out.push(digit(s));
    for j in (0..s).rev() {
        // x·pⱼ + eⱼ limb by limb: a limb's product and carry stay below
        // 2^64·pⱼ, and the carry below pⱼ.
        let p = u128::from(primes[j]);
        let mut carry = u128::from(digit(j));
        for limb in out.iter_mut() {
            let t = u128::from(*limb) * p + carry;
            *limb = (t & u128::from(u64::MAX)).narrow();
            carry = t >> 64;
        }
        // x > 0 and pⱼ ≥ 2, so a carry of 0 leaves a top limb that is not 0.
        if carry != 0 {
            out.push(carry.narrow());
        }
    }
    s + 1
}

// q-lift-growth-time.4 design -- the scaling v = u·l mod M multiplies each image of u by l's residue at its prime before the conversion, which then gives v, not u converted and then num-bigint's product u·l and its remainder by M (q-rational-time.7, withdrawn) (log 0050-1, 0083-1)
// q-lift-growth-time.6 cited: vanderhoeven-2016 §2.2.2, Function 8 and Proposition 4 -- for a fixed multiplicand y precompute ψ := ⌊2^n y/p⌋, c = (xψ) >> n, d = xy − cp, one correction
/// outᵢ = imagesᵢ·lᵢ mod pᵢ for each image, lᵢ = `residues[i]` < pᵢ
/// with its Shoup factor `factors[i]`, pᵢ = `primes[i]` < 2^63: the images
/// of u·l at the primes from those of u and l. Past the images the
/// residues, factors, and primes are not read.
pub(crate) fn scale_images(
    images: &[u64],
    residues: &[u64],
    factors: &[u64],
    primes: &[u64],
    out: &mut Vec<u64>,
) {
    let k = images.len();
    debug_assert!(
        residues.len() >= k && factors.len() >= k && primes.len() >= k,
        "{k} images, {residues} residues, {factors} factors, {primes} primes",
        residues = residues.len(),
        factors = factors.len(),
        primes = primes.len()
    );
    out.clear();
    out.extend(
        images
            .iter()
            .zip(residues)
            .zip(factors)
            .zip(primes)
            .map(|(((&x, &l), &f), &p)| shoup_mul(x, l, f, p)),
    );
}

/// One Chinese remainder step, M to M·p.
pub(crate) struct CrtStep {
    prime: u64,
    /// (p₀⋯pⱼ₋₁) mod p for each of the k primes pⱼ of M, entry 0
    /// equal to 1.
    row: Vec<u64>,
    /// M⁻¹ mod p.
    inverse: u64,
    /// The inverse's Shoup factor, `shoup_factor(inverse, p)`.
    factor: u64,
    // q-crt-time.5 design -- the powers 2^(64i) mod a prime are computed once, as many as M has limbs or the lifting's width, and the prime's step holds them for the attempts at that prime, not recomputed per residue nor one table for the whole call (not measured)
    /// The powers 2^(64i) mod p, as many as M has limbs or the lifting's
    /// width, whichever is more.
    radix: Radix,
}

impl CrtStep {
    /// The step from `crt`'s M to M·p, p a prime below 2^62 that does not
    /// divide M. The radix has as many powers as M has limbs or `width`,
    /// whichever is more.
    fn new(crt: &Crt, p: u64, width: usize) -> CrtStep {
        let m = &crt.modulus;
        // Entry j is (p₀⋯pⱼ₋₁) mod p, and past the last prime m_mod_p is M
        // mod p. p ≥ 2, so entry 0 is 1.
        let mut row = Vec::with_capacity(crt.primes.len());
        let mut m_mod_p = 1;
        for &q in &crt.primes {
            row.push(m_mod_p);
            m_mod_p = mul_mod(m_mod_p, q, p);
        }
        // M is a product of primes other than p, so M mod p is a unit. The
        // products and the inverse take any p below 2^63.
        let inverse = inverse_mod(m_mod_p, p);
        let radix = Radix::new(p, m.iter_u64_digits().len().max(width).max(1));
        CrtStep {
            prime: p,
            row,
            inverse,
            factor: shoup_factor(inverse, p),
            radix,
        }
    }

    /// The row, (p₀⋯pⱼ₋₁) mod p for each prime pⱼ of M. The crate's
    /// unit tests read it.
    #[cfg(test)]
    pub(crate) fn row(&self) -> &[u64] {
        &self.row
    }

    /// M⁻¹ mod p. The crate's unit tests read it.
    #[cfg(test)]
    pub(crate) fn inverse(&self) -> u64 {
        self.inverse
    }

    /// The inverse's Shoup factor. The crate's unit tests read it.
    #[cfg(test)]
    pub(crate) fn factor(&self) -> u64 {
        self.factor
    }

    /// The prime p. The crate's unit tests read it.
    #[cfg(test)]
    pub(crate) fn prime(&self) -> u64 {
        self.prime
    }

    /// The powers 2^(64i) mod p, for every number the step reduces modulo p.
    pub(crate) fn radix(&self) -> &Radix {
        &self.radix
    }
}

/// The powers 2^(64i) mod p, i below a length, for one prime p below
/// 2^62: the residue modulo p of an integer of at most that many limbs.
pub(crate) struct Radix {
    p: u64,
    powers: Vec<u64>,
    /// 2^128 mod p.
    top: u64,
}

impl Radix {
    /// p a prime below 2^62, `len` at least 1 and below 2^32.
    pub(crate) fn new(p: u64, len: usize) -> Radix {
        debug_assert!((2..1 << 62).contains(&p) && (1..1 << 32).contains(&len));
        let b: u64 = ((1u128 << 64) % u128::from(p)).narrow();
        let mut powers = Vec::with_capacity(len);
        let mut w = 1;
        for _ in 0..len {
            powers.push(w);
            w = mul_mod(w, b, p);
        }
        // 2^128 = (2^128 − 1) + 1.
        let below: u64 = (u128::MAX % u128::from(p)).narrow();
        let top = (below + 1) % p;
        Radix { p, powers, top }
    }

    /// The residue modulo p of the sum of the products of `limbs` and the
    /// powers. A product lies below 2^64·2^62 = 2^126.
    fn sum(&self, limbs: impl Iterator<Item = u64>) -> u64 {
        wrapped_sum(limbs.zip(self.powers.iter().copied()), self.p, self.top)
    }

    /// The prime p. The crate's unit tests read it.
    #[cfg(test)]
    pub(crate) fn prime(&self) -> u64 {
        self.p
    }

    // q-crt-time.4 cited: dumas-2008 §2.1.2 -- successive multiplications and accumulations without any division
    /// x mod p, x on limbs, at most `len` of them. The crate's unit tests
    /// read it.
    #[cfg(test)]
    pub(crate) fn residue(&self, x: &[u64]) -> u64 {
        debug_assert!(x.len() <= self.powers.len(), "{len} limbs", len = x.len());
        self.sum(x.iter().copied())
    }

    // q-crt-time.4 cited: dumas-2008 §2.1.2 -- successive multiplications and accumulations without any division
    /// x mod p: by the powers where x has at most `len` limbs, and past
    /// them by the free function `residue`'s Horner rule.
    pub(crate) fn residue_big(&self, x: &BigUint) -> u64 {
        let digits = x.iter_u64_digits();
        // A chain's last and a numerator over it may pass the step's width
        // within the attempt that makes them (log 0076-1, _Step 3_).
        if digits.len() > self.powers.len() {
            return residue(x, self.p);
        }
        self.sum(digits)
    }

    // q-crt-time.8 design -- a rational n/d fits c at p where d mod p ≠ 0 and n mod p = c·(d mod p) mod p, not n·d⁻¹ mod p by an extended Euclid compared with c (log 0023-1)
    /// Whether r reduces to c at p: d mod p ≠ 0 and n ≡ c·d mod p. c < p.
    pub(crate) fn fits(&self, r: &Rational, c: u64) -> bool {
        self.fits_over(&r.num, self.residue_big(&r.den), c)
    }

    // q-crt-time.8 design -- a rational n/d fits c at p where d mod p ≠ 0 and n mod p = c·(d mod p) mod p, not n·d⁻¹ mod p by an extended Euclid compared with c (log 0023-1)
    /// Whether n/d reduces to c at p, given d's residue dr = d mod p:
    /// dr ≠ 0 and n ≡ c·dr mod p. c < p.
    pub(crate) fn fits_over(&self, n: &BigInt, dr: u64, c: u64) -> bool {
        debug_assert!(c < self.p && dr < self.p);
        dr != 0 && self.signed(n) == mul_mod(c, dr, self.p)
    }

    /// a·b mod p, a and b below p.
    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.p && b < self.p);
        mul_mod(a, b, self.p)
    }

    /// d⁻¹ mod p; None where p divides d.
    pub(crate) fn inverse(&self, d: &BigUint) -> Option<u64> {
        let d = self.residue_big(d);
        (d != 0).then(|| inverse_mod(d, self.p))
    }

    /// n·di mod p, given a denominator's inverse di mod p.
    pub(crate) fn over(&self, n: &BigInt, di: u64) -> u64 {
        mul_mod(self.signed(n), di, self.p)
    }

    /// n·d⁻¹ mod p; None where p divides d. The crate's unit tests read
    /// it.
    #[cfg(test)]
    pub(crate) fn rational(&self, r: &Rational) -> Option<u64> {
        self.inverse(&r.den).map(|di| self.over(&r.num, di))
    }

    /// n mod p, in [0, p).
    fn signed(&self, n: &BigInt) -> u64 {
        let r = self.residue_big(n.magnitude());
        match n.sign() {
            Sign::Minus if r != 0 => self.p - r,
            Sign::Minus | Sign::NoSign | Sign::Plus => r,
        }
    }
}

// q-modulus-time.6 design -- the residue is the u128 sum of the limbs' products by the powers, its wrapped additions counted, for every prime, not each limb split in halves of 32 bits with two products a limb (log 0036-1)
// q-modulus-time.7 design -- the counted sum is reduced once, after its last limb, as (c·(2^128 mod p) + s mod p) mod p, not a remainder taken every four products before the sum can wrap (not measured)
/// The residue modulo p < 2^62 of the sum of the products of the pairs,
/// given `top` = 2^128 mod p: the residue of an integer from its limbs and
/// the powers 2^(64i) mod p, and a mixed-radix digit's sum. Each addition
/// that wraps the `u128` sum s counts one 2^128, so the sum is c·2^128 + s
/// for the count c, and c·(2^128 mod p) + s mod p < 2^64·2^62 + 2^62.
#[inline]
fn wrapped_sum(pairs: impl Iterator<Item = (u64, u64)>, p: u64, top: u64) -> u64 {
    debug_assert!(top < p && p < 1 << 62, "2^128 mod {p} = {top}");
    let (mut s, mut c) = (0u128, 0u64);
    for (a, b) in pairs {
        let (t, wrapped) = s.overflowing_add(u128::from(a) * u128::from(b));
        s = t;
        c += u64::from(wrapped);
    }
    ((u128::from(c) * u128::from(top) + s % u128::from(p)) % u128::from(p)).narrow()
}

/// x mod p, for p < 2^63.
pub(crate) fn residue(x: &BigUint, p: u64) -> u64 {
    debug_assert!(p > 0 && p < 1 << 63);
    // Horner's rule over the 64-bit digits from the most significant: r < p
    // < 2^63 keeps r·2^64 + digit below 2^128.
    x.iter_u64_digits().rev().fold(0u64, |r, digit| {
        (((u128::from(r) << 64) | u128::from(digit)) % u128::from(p)).narrow()
    })
}

/// A rational number n/d in lowest terms, d > 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rational {
    pub(crate) num: BigInt,
    pub(crate) den: BigUint,
}

impl Rational {
    /// n/d in lowest terms; None for d = 0.
    pub(crate) fn new(num: BigInt, den: BigInt) -> Option<Rational> {
        if den.is_zero() {
            return None;
        }
        // The gcd is positive, and |d| where n = 0, which makes d 1.
        let g = num.gcd(&den);
        let (mut num, den) = (num / &g, den / &g);
        let (sign, den) = den.into_parts();
        match sign {
            Sign::Minus => num = -num,
            Sign::NoSign | Sign::Plus => {}
        }
        Some(Rational { num, den })
    }

    /// Whether n is 0. The crate's unit tests read it.
    #[cfg(test)]
    pub(crate) fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// n·d⁻¹ mod p, None where p divides d; p < 2^63 a prime. The crate's
    /// unit tests read it.
    #[cfg(test)]
    pub(crate) fn residue(&self, p: u64) -> Option<u64> {
        let d = residue(&self.den, p);
        if d == 0 {
            return None;
        }
        let mut n = residue(self.num.magnitude(), p);
        match self.num.sign() {
            Sign::Minus if n != 0 => n = p - n,
            Sign::Minus | Sign::NoSign | Sign::Plus => {}
        }
        Some(mul_mod(n, inverse_mod(d, p), p))
    }
}

/// Wang's bound for the modulus m, ⌊√(m/2)⌋, of the test code's lift of
/// Wang's reconstruction alone.
#[cfg(test)]
pub(crate) fn bound(m: &BigUint) -> BigUint {
    // ⌊√⌊m/2⌋⌋ = ⌊√(m/2)⌋.
    (m >> 1u32).sqrt()
}

/// The rational n/d with |n| ≤ N, 0 < d ≤ N, gcd(n, d) = 1, and
/// n ≡ d·u mod m, where N = `bound`, m > 1, and u < m; None where there
/// is none. monagan-2004 §2's algorithm RR with N = D. The test code's
/// oracle for `lehmer::Wang`, which the test code's lift of Wang's
/// reconstruction alone runs.
#[cfg(test)]
pub(crate) fn reconstruct(u: &BigUint, m: &BigUint, bound: &BigUint) -> Option<Rational> {
    debug_assert!(u < m);
    let (mut r0, mut r1) = (m.clone(), u.clone());
    let (mut t0, mut t1) = (BigInt::zero(), BigInt::one());
    // r1 falls each turn, so the loop ends; r1 > N ≥ 0 keeps the divisor
    // nonzero.
    while r1 > *bound {
        let (q, r) = r0.div_rem(&r1);
        (r0, r1) = (r1, r);
        let t = t0 - BigInt::from(q) * &t1;
        (t0, t1) = (t1, t);
    }
    // n = r1 and d = t1, both negated where t1 < 0. Every quotient is at
    // least 1, since u < m, so the t after the first alternate in sign and
    // grow, and t1 is nonzero.
    let (sign, den) = t1.into_parts();
    if den.is_zero() || den > *bound {
        return None;
    }
    // q-lift.14 cited: collins-1995 §2 -- RATCONVERT applied to M = 12 and U = 5 outputs (−2, 2), since it does not check gcd(B, M) = 1; §4.3 -- gcd(A, B) = gcd(B, M), and A and B are at most about half the length of M
    if !r1.gcd(&den).is_one() {
        return None;
    }
    let sign = match sign {
        Sign::Minus => Sign::Minus,
        Sign::NoSign | Sign::Plus => Sign::Plus,
    };
    Some(Rational {
        num: BigInt::from_biguint(sign, r1),
        den,
    })
}

/// The coefficients of a polynomial over Q whose first coefficient is 1,
/// multiplied by the lcm L of their denominators: L and the integers. The
/// first integer is L, and the integers have no common factor: a prime
/// that divides L divides some denominator to its full power in L, and
/// neither that numerator nor L over that denominator. The scaling the
/// lift's basis text holds (q-rational-time.6), tested on its own.
#[cfg(test)]
pub(crate) fn scale(coefs: &[Rational]) -> (BigUint, Vec<BigInt>) {
    debug_assert!(
        coefs
            .first()
            .is_some_and(|c| c.num.is_one() && c.den.is_one())
    );
    let l = coefs.iter().fold(BigUint::one(), |l, c| l.lcm(&c.den));
    let ints = coefs
        .iter()
        .map(|c| &c.num * BigInt::from(&l / &c.den))
        .collect();
    (l, ints)
}

/// At least the text of a term over Q whose integer has `bits` bits: a
/// sign, the integer's digits, a `*`, and the monomial's longest text.
pub(crate) fn term_bound_q(names: &[&str], bits: u64) -> usize {
    TERM_MARKS + digits_bound(bits) + monomial_longest(names)
}

/// Writes a polynomial over Q with integer coefficients in the canonical
/// spelling: terms joined by `+`, or by `-` before a negative coefficient;
/// the coefficient's absolute value, then `*` and the monomial, where an
/// absolute value 1 and its `*` are left out before a monomial, and a
/// constant term is its coefficient's absolute value alone; the
/// monomial's variables joined by `*`, each `^e` where e > 1. A term whose
/// coefficient is 0 is left out, and a polynomial with no other term is
/// `0`. `exponents` holds `names.len()` exponents per term. Each term
/// spends what `write_terms` says; a stop where the budget's check stops
/// the call, the text then cut short.
pub(crate) fn write_poly_q(
    out: &mut String,
    names: &[&str],
    exponents: &[u32],
    coefs: &[BigInt],
    budget: &mut Budget,
) -> Result<(), Stop> {
    debug_assert_eq!(exponents.len(), names.len() * coefs.len());
    let terms = coefs.iter().map(|c| (c.sign(), c.magnitude(), None));
    write_terms(out, names, exponents, terms, budget)
}

/// Writes a polynomial over Q with rational coefficients as
/// `write_poly_q` writes one, each coefficient whose denominator is not 1
/// as `|n|/d`, which the `*` and the monomial follow unless the monomial
/// is 1; `0` where it has no
/// term. Each term spends what `write_terms` says; a stop where the
/// budget's check stops the call, the text then cut short.
pub(crate) fn write_cofactor_q(
    out: &mut String,
    names: &[&str],
    exponents: &[u32],
    coefs: &[Rational],
    budget: &mut Budget,
) -> Result<(), Stop> {
    debug_assert_eq!(exponents.len(), names.len() * coefs.len());
    let terms = coefs.iter().map(|c| {
        (
            c.num.sign(),
            c.num.magnitude(),
            (!c.den.is_one()).then_some(&c.den),
        )
    });
    write_terms(out, names, exponents, terms, budget)
}

// text-stop-latency.1 design -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
/// Writes the terms, each its sign, the absolute value of its numerator,
/// and its denominator where it has one, as `write_poly_q` and
/// `write_cofactor_q` say. Right after each term it writes it spends its
/// bytes, its sign or `+` among them, and `decimal_units` of its numerator
/// and of its denominator where it has one; a term of coefficient 0 and
/// the `0` of no term spend nothing. A stop where the budget's check stops
/// the call, the text then cut short.
fn write_terms<'c>(
    out: &mut String,
    names: &[&str],
    exponents: &[u32],
    terms: impl Iterator<Item = (Sign, &'c BigUint, Option<&'c BigUint>)>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let n = names.len();
    // result-time.3 design -- the text is written as bytes into the vector under the String, ASCII and each name whole, not characters pushed into the String or a vector checked by `String::from_utf8` at the end (log 0042-1)
    // SAFETY: every byte written is ASCII or a byte of a name, and each name
    // is a `&str` written whole, so the String stays UTF-8.
    let out = unsafe { out.as_mut_vec() };
    let mut written = false;
    for (k, (sign, abs, den)) in terms.enumerate() {
        let mark = out.len();
        match sign {
            Sign::NoSign => continue,
            Sign::Minus => out.push(b'-'),
            Sign::Plus if written => out.push(b'+'),
            Sign::Plus => {}
        }
        written = true;
        let monomial = &exponents[k * n..(k + 1) * n];
        let constant = monomial.iter().all(|&e| e == 0);
        if constant || den.is_some() || !abs.is_one() {
            push_big(out, abs);
            if let Some(d) = den {
                out.push(b'/');
                push_big(out, d);
            }
            if !constant {
                out.push(b'*');
            }
        }
        if !constant {
            // text-time.8 design -- over Q each monomial is written from its exponents at each term, not kept in a hash map keyed by its exponents (log 0048-1)
            push_monomial(out, names, monomial);
        }
        // text-stop-latency.3 design -- over Q each term spends, right after it, its bytes plus the square of the limbs of its numerator and of its denominator, not its bytes alone nor a spend per polynomial (log 0057-1)
        let digits = decimal_units(abs) + den.map_or(0, decimal_units);
        budget.spend(out.len() - mark + digits)?;
    }
    if !written {
        out.push(b'0');
    }
    Ok(())
}
