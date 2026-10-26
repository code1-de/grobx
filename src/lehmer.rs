//! The maximal quotient rational reconstruction of monagan-2004 §3, MQRR,
//! on the scaled residue u·l mod M, the check gcd(n, d) = 1, and the gcd,
//! by Lehmer's multiprecision Euclidean algorithm on 64-bit limbs of the
//! module's own: collins-1995's ISMC. Wang's rational reconstruction, which
//! the test code's lift runs, takes the same Euclidean algorithm. A number
//! is its limbs, the least significant first, with no zero limb at the
//! top, so 0 has none.

use std::cmp::Ordering;
use std::mem;

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;

use crate::narrow::Narrow;
#[cfg(test)]
use crate::rational::bound;
use crate::rational::{Converted, Rational};

// q-reconstruct-time.7 cited: collins-1995 §4.2 -- ISMC is applied several times to the same modulus, so ⌊√(M/2)⌋ and its bit length are inputs that are not recomputed each time
/// Wang's reconstruction modulo one M, prepared once for every residue
/// reconstructed modulo it: the reconstruction of the test code's lift of
/// Wang's reconstruction alone (`lift::Reconstruct::Wang`).
#[cfg(test)]
pub(crate) struct Wang {
    m: Vec<u64>,
    /// N = ⌊√(M/2)⌋.
    bound: BigUint,
    /// N's limbs.
    n: Vec<u64>,
    /// L(N), N's bit length.
    n_bits: u64,
}

#[cfg(test)]
impl Wang {
    /// M > 1.
    pub(crate) fn new(m: &BigUint) -> Wang {
        debug_assert!(m.bits() > 1, "M = {m}");
        let bound = bound(m);
        Wang {
            m: limbs(m),
            n: limbs(&bound),
            n_bits: bound.bits(),
            bound,
        }
    }

    /// N = ⌊√(M/2)⌋, `rational::bound(M)`.
    pub(crate) fn bound(&self) -> &BigUint {
        &self.bound
    }

    /// What `rational::reconstruct(u, M, N)` returns: the rational n/d
    /// with |n| ≤ N, 0 < d ≤ N, gcd(n, d) = 1, and n ≡ d·u mod M; None
    /// where there is none. u < M, on limbs.
    pub(crate) fn reconstruct(&self, u: &[u64]) -> Option<Rational> {
        debug_assert_ne!(u.last(), Some(&0), "a zero limb at the top of u");
        let mut e = Euclid::new(self.m.clone(), u.to_vec());
        debug_assert_eq!(cmp(&e.a2, &e.a1), Ordering::Less, "u = {u:?} ≥ M");
        e.v2.push(1);
        // Every quotient taken is a true one, so the state is that of the
        // plain Euclidean algorithm on (M, u) at some index. No run passes
        // the first remainder at most N, so the loop stops there, where
        // `rational::reconstruct` stops.
        while cmp(&e.a2, &self.n).is_gt() {
            // |t| only grows, so d would pass N.
            if cmp(&e.v2, &self.n).is_gt() {
                return None;
            }
            // q-reconstruct-time.3 cited: collins-1995 §4.1, (4.1) and Figure 3 -- |V_{k+2}| ≤ 2|v_{k+2}V₂|, so L(v_{k+2}) < ⌊log₂ √(M/2)⌋ − L(V₂) − 1 keeps it below √(M/2); DDRPCC ensures |v₂| < m
            let (bound_bits, v2_bits): (i64, i64) = (self.n_bits.narrow(), bits(&e.v2).narrow());
            let c = bound_bits - v2_bits - 1;
            // The limit 2^(c−1) bounds |v_{k+2}| below 2^h, h the right side
            // of (4.1): L(v_{k+2}) ≤ h, one bit past (4.1), and |V_{k+2}| <
            // 2^(h+1+L(V₂)) = 2^⌊log₂ N⌋ ≤ N all the same.
            // Where c < 2 the run takes one step at most. A limit past 2^64
            // binds no cosequence element.
            let run = e.run(1 << (c - 1).clamp(0, 64), NO_CAP, NO_FLOOR);
            if run.k == 0 {
                e.full_step(true);
            } else {
                e.apply(&run, true);
            }
        }
        if cmp(&e.v2, &self.n).is_gt() {
            return None;
        }
        e.rational()
    }

    /// What `reconstruct` returns for the u in [0, M) a conversion of its
    /// images names (`rational::Crt::convert`): x on limbs with no zero
    /// limb at the top is u where `converted` is `Low` or `Any`, and M − 1
    /// − u where it is `High`.
    pub(crate) fn reconstruct_converted(
        &self,
        converted: Converted,
        x: &[u64],
    ) -> Option<Rational> {
        match converted {
            Converted::Low | Converted::Any => self.reconstruct(x),
            Converted::High => self.reconstruct(&complement(&big(&self.m), x)),
        }
    }
}

/// M − 1 − x on limbs with no zero limb at the top, x < M on limbs.
fn complement(m: &BigUint, x: &[u64]) -> Vec<u64> {
    limbs(&(m - 1u32 - big(x)))
}

// q-prime-count-time.4 tuned -- the threshold asks for 10 bits past the median of the largest quotient, not 6, which took the same time and, as the count of largest quotients halves per bit past the median (monagan-2004 §3, Table 4), lets 2^4 times as many false candidates through, or 16, which took more time (log 0076-1, _Step 3_)
/// The bits past the median of the largest quotient that MQRR's threshold
/// asks for.
pub(crate) const MARGIN: u64 = 10;

// q-prime-count-time.3 cited: monagan-2004 §3 -- the T used by algorithm MQRR must increase as m increases, log₂ T = ⌈log₂ log₂ m⌉ + c; §3.1 -- the median largest quotient's log₂ is about log₂ log₂ m + 0.2820
/// MQRR's threshold T for M > 1: ⌈log₂ L(M)⌉ + `MARGIN`, L the bit length.
pub(crate) fn threshold(m: &BigUint) -> u64 {
    debug_assert!(m.bits() > 1, "M = {m}");
    u64::from((m.bits() - 1).bit_width()) + MARGIN
}

/// What `Mqrr::first` returns for v in [1, M).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum First {
    /// No quotient of the Euclidean algorithm on (M, v) reaches 2^T, so
    /// MQRR finds no rational either.
    Nothing,
    /// The rational `r_i/t_i` before the first quotient `q_{i+1}` that
    /// reaches 2^T, `|t_i|` its denominator, `r_i` and `t_i` coprime.
    Rational(Rational),
    /// That rational's `r_i` and `t_i` share a factor.
    Shared,
}

/// What `Mqrr::classify` returns for the conversion of the images of u
/// scaled by those of l: v = u·l mod M.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Scaled {
    /// v in (−M/2, M/2], where |v|·2^(T+1) < M.
    Small(BigInt),
    /// v in [1, M) on limbs, where v is not small.
    Residue(Vec<u64>),
}

// q-prime-count-time.1 cited: monagan-2004 §3 -- MQRR outputs n/d with high probability once m is only a modest number of bits longer than 2|n|d, where N = D = ⌊√(m/2)⌋ requires up to twice as many primes as are necessary; traverso-1989 §Reconstruction algorithms -- the coefficients are lifted one at a time, and the others are multiplied by the denominator found
/// The maximal quotient rational reconstruction modulo one odd M > 1 at its
/// threshold T, prepared once for every residue reconstructed modulo it.
pub(crate) struct Mqrr {
    m: Vec<u64>,
    /// M.
    modulus: BigUint,
    /// T: a quotient MQRR takes is at least 2^T.
    t: u64,
    // q-lift-growth-time.12 design -- MQRR's bounds `small` and M − `small` are held on limbs, and a converted x is compared with them on limbs, not made num-bigint's BigUint first and compared (not measured)
    /// ⌊(M − 1)/2^(T+1)⌋ on limbs, the largest |v| with |v|·2^(T+1) < M.
    /// Up to it v/1 has a quotient of at least 2^T in MQRR on (M, v mod M),
    /// of either sign: ⌊M/v⌋ for v > 0, and ⌊(M − |v|)/|v|⌋ after the
    /// quotient 1 for v < 0.
    small: Vec<u64>,
    /// M − `small` on limbs: a residue v at least this has M − v ≤
    /// `small`.
    upper: Vec<u64>,
}

impl Mqrr {
    /// M odd, M > 1, at T = `threshold(M)`.
    pub(crate) fn new(m: &BigUint) -> Mqrr {
        Mqrr::prepare(m, threshold(m))
    }

    /// M odd, M > 1, at T = t.
    #[cfg(test)]
    pub(crate) fn with_threshold(m: &BigUint, t: u64) -> Mqrr {
        Mqrr::prepare(m, t)
    }

    fn prepare(m: &BigUint, t: u64) -> Mqrr {
        debug_assert!(m.bits() > 1 && m.bit(0), "M = {m} is no odd number past 1");
        let small = (m - 1u32) >> (t + 1);
        Mqrr {
            m: limbs(m),
            modulus: m.clone(),
            t,
            upper: limbs(&(m - &small)),
            small: limbs(&small),
        }
    }

    /// T.
    pub(crate) fn threshold(&self) -> u64 {
        self.t
    }

    // q-prime-count-time.2 cited: traverso-1989 §Reconstruction algorithms -- the coefficients are lifted one at a time and the others multiplied by the denominator found, and the remaining coefficients that fall below the bound are lifted with denominator 1
    /// v = u·l mod M from the conversion of the images of u·l at M's primes
    /// (`rational::Crt::convert`): `Small(v)`, v in (−M/2, M/2], where
    /// |v|·2^(T+1) < M, and `Residue(v)`, v in [1, M), where it is not. x
    /// on limbs with no zero limb at the top is v where `converted` is
    /// `Low` or `Any`, and M − 1 − v where it is `High`. A conversion that
    /// ended below the top (`rational::Crt::convert_small`) gives v so
    /// where v is small; elsewhere its x names another v, which the caller
    /// checks.
    pub(crate) fn classify(&self, converted: Converted, x: &[u64]) -> Scaled {
        debug_assert!(x.last() != Some(&0), "a zero limb at the top of {x:?}");
        debug_assert_eq!(cmp(x, &self.m), Ordering::Less, "x = {x:?} ≥ M");
        match converted {
            // `small` < M/2 < `upper`, so v and M − v are not both small. A
            // `Low` v lies below M/3, short of `upper`.
            Converted::Low | Converted::Any => {
                if cmp(x, &self.small).is_le() {
                    return Scaled::Small(BigInt::from(big(x)));
                }
                if cmp(x, &self.upper).is_ge() {
                    return Scaled::Small(-BigInt::from(&self.modulus - big(x)));
                }
                Scaled::Residue(x.to_vec())
            }
            // M − v = x + 1, which is at most `small` where x is below it.
            // x lies below M/3, so v lies past M/2 and is not small.
            Converted::High => {
                if cmp(x, &self.small).is_lt() {
                    return Scaled::Small(-BigInt::from(big(x) + 1u32));
                }
                Scaled::Residue(complement(&self.modulus, x))
            }
        }
    }

    // q-prime-count-time.5 design -- MQRR runs in two passes, the remainders alone to find the step of the first largest quotient and then the remainders and the cosequence to that step, not one pass with the cosequence throughout as monagan-2004 §3 writes MQRR (not measured)
    /// MQRR on (M, v), monagan-2004 §3: the rational n/d = `r_i/t_i` of the
    /// Euclidean algorithm on (M, v) before its first largest quotient
    /// `q_{i+1}`, where `q_{i+1}` ≥ 2^T and `gcd(r_i, t_i)` = 1, d = `|t_i|`;
    /// None where no quotient reaches 2^T or the gcd is not 1. n ≡ d·v mod
    /// M. v in [1, M), on limbs.
    pub(crate) fn reconstruct(&self, v: &[u64]) -> Option<Rational> {
        debug_assert!(!v.is_empty() && v.last() != Some(&0), "v = {v:?}");
        debug_assert_eq!(cmp(v, &self.m), Ordering::Less, "v = {v:?} ≥ M");
        let steps = largest_quotient(&self.m, v, self.t, Until::Largest)?;
        self.rational_at(v, steps)
    }

    // q-attempt-time.3 invented -- MQRR's first pass ends at the first quotient, in the order of the steps, that reaches 2^T, and the attempt takes the rational before it where it fits at p, MQRR running to its largest quotient only where it does not, not the first pass to the remainder below the largest quotient at every coefficient, nor the pass ended at a quotient past √M (log 0085-1)
    /// The rational `r_i/t_i` of the Euclidean algorithm on (M, v) before
    /// its first quotient `q_{i+1}` of at least 2^T, in the order of the
    /// steps, d = `|t_i|`, as `First` says. Where it is `Nothing`,
    /// `reconstruct` returns None. v in [1, M), on limbs.
    pub(crate) fn first(&self, v: &[u64]) -> First {
        debug_assert!(!v.is_empty() && v.last() != Some(&0), "v = {v:?}");
        debug_assert_eq!(cmp(v, &self.m), Ordering::Less, "v = {v:?} ≥ M");
        let Some(steps) = largest_quotient(&self.m, v, self.t, Until::First) else {
            return First::Nothing;
        };
        match self.rational_at(v, steps) {
            Some(r) => First::Rational(r),
            None => First::Shared,
        }
    }

    /// MQRR's second pass, the remainders and the cosequence: the rational
    /// A₂/V₂ of the Euclidean algorithm on (M, v) after `steps` steps, in
    /// the sign A₂'s index gives it, where A₂ and V₂ are coprime; None
    /// where they are not. v in [1, M), on limbs.
    fn rational_at(&self, v: &[u64], steps: u64) -> Option<Rational> {
        let mut e = Euclid::new(self.m.clone(), v.to_vec());
        e.v2.push(1);
        // q-prime-count-time.6 cited: monagan-2004 §4 -- the efficiency improvements made to Wang's algorithm by Collins and Encarnación should be applicable to MQRR, both being based on the Euclidean algorithm; collins-1995 §4.1, Figure 4 -- ISMC: the quotients of a run come from the leading digits, and the run is applied to the remainders and the cosequence in one linear combination each
        // Every quotient taken is a true one, so after `steps` steps A₂ and
        // V₂ are `r_i` and `|t_i|`, whatever the runs.
        let mut taken = 0;
        while taken < steps {
            let run = e.run(UNLIMITED, steps - taken, NO_FLOOR);
            if run.k == 0 {
                e.full_step(true);
                taken += 1;
            } else {
                e.apply(&run, true);
                taken += u64::from(run.k);
            }
        }
        e.rational()
    }
}

/// Where MQRR's first pass ends.
#[derive(Clone, Copy)]
enum Until {
    /// At the first quotient of at least 2^t, in the order of the steps.
    First,
    /// At the first largest quotient, where it is at least 2^t.
    Largest,
}

// q-prime-count-time.6 cited: monagan-2004 §4 -- the efficiency improvements made to Wang's algorithm by Collins and Encarnación should be applicable to MQRR, both being based on the Euclidean algorithm; collins-1995 §4.1, Figure 4 -- ISMC: the quotients of a run come from the leading digits, and the run is applied to the remainders and the cosequence in one linear combination each
/// MQRR's first pass on (M, v), the remainders alone: the count of steps of
/// the Euclidean algorithm on (M, v) before the quotient `until` names,
/// where that quotient is at least 2^t; None where none is. v in [1, M).
fn largest_quotient(m: &[u64], v: &[u64], t: u64, until: Until) -> Option<u64> {
    debug_assert!(!v.is_empty() && cmp(v, m).is_lt(), "v = {v:?}");
    let mut euclid = Euclid::new(m.to_vec(), v.to_vec());
    let mut steps = 0;
    // The largest quotient found of at least 2^t, the first of equal ones,
    // and the count of steps before it.
    let mut best: Option<(Vec<u64>, u64)> = None;
    // A run's quotients lie below 2^64 (`cosequence`), so a floor past
    // them ends no run, and a quotient past 2^64 is a full step's.
    let floor = match until {
        Until::First if t < 64 => 1 << t,
        Until::First | Until::Largest => NO_FLOOR,
    };
    let first = match until {
        Until::First => true,
        Until::Largest => false,
    };
    // q-prime-count-time.7 cited: monagan-2004 §3, algorithm MQRR -- the loop runs while r₁ ≠ 0 and r₀ > T, and T becomes each quotient larger than it
    while !euclid.a2.is_empty()
        && bits(&euclid.a1) > t
        && best.as_ref().is_none_or(|(largest, _)| match until {
            Until::First => false,
            Until::Largest => cmp(&euclid.a1, largest).is_gt(),
        })
    {
        let run = euclid.run(UNLIMITED, NO_CAP, floor);
        // q-lift-rest-time.4 design -- MQRR's first pass ends at a step its Lehmer runs do not give where the remainders' bit lengths differ by more than T, as ⌊A₁/A₂⌋ ≥ 2^(L(A₁)−L(A₂)−1) reaches 2^T, not num-bigint's division of the step (q-reconstruct-time.6) for a quotient of which it reads only that (log 0088-1)
        if run.k == 0 && first && bits(&euclid.a1) - bits(&euclid.a2) > t {
            return Some(steps);
        }
        let found = if run.k == 0 {
            let quotient = euclid.full_step(false);
            let found = (quotient.bits() > t).then(|| (limbs(&quotient), steps));
            steps += 1;
            found
        } else {
            euclid.apply(&run, false);
            let found = (u64::from(run.largest.bit_width()) > t)
                .then(|| (wide(run.largest), steps + u64::from(run.at) - 1));
            steps += u64::from(run.k);
            found
        };
        if let Some((quotient, before)) = found
            && best
                .as_ref()
                .is_none_or(|(largest, _)| cmp(&quotient, largest).is_gt())
        {
            best = Some((quotient, before));
        }
    }
    best.map(|(_, before)| before)
}

/// Whether gcd(x, y) = 1, by Lehmer's Euclid on limbs. gcd(0, y) = y.
#[cfg(test)]
pub(crate) fn coprime(x: &BigUint, y: &BigUint) -> bool {
    coprime_limbs(limbs(x), limbs(y))
}

// q-reconstruct-time.4 cited: collins-1995 §4.3 -- method I, gcd(A, B) computed by the algorithm of Section 3, used throughout
/// As `coprime`, on limbs.
fn coprime_limbs(x: Vec<u64>, y: Vec<u64>) -> bool {
    let e = to_a_word(x, y);
    match e.a2.first() {
        Some(&d) => word_gcd(remainder(&e.a1, d), d) == 1,
        None => e.a1 == [1],
    }
}

// q-rational-time.9 cited: collins-1995 §4.3 -- method I, gcd(A, B) computed by the algorithm of Section 3, used throughout
/// gcd(x, y) by Lehmer's Euclid on limbs, as `coprime` runs it. gcd(0, y)
/// = y.
pub(crate) fn gcd(x: &BigUint, y: &BigUint) -> BigUint {
    let e = to_a_word(limbs(x), limbs(y));
    match e.a2.first() {
        Some(&d) => BigUint::from(word_gcd(remainder(&e.a1, d), d)),
        None => big(&e.a1),
    }
}

/// The Euclidean algorithm on x and y without the cosequence, run until
/// the smaller remainder has at most one limb. Its two remainders have
/// the gcd of x and y.
fn to_a_word(x: Vec<u64>, y: Vec<u64>) -> Euclid {
    let (a1, a2) = if cmp(&x, &y).is_lt() { (y, x) } else { (x, y) };
    let mut e = Euclid::new(a1, a2);
    while e.a2.len() >= 2 {
        let run = e.run(UNLIMITED, NO_CAP, NO_FLOOR);
        if run.k == 0 {
            e.full_step(false);
        } else {
            e.apply(&run, false);
        }
    }
    e
}

/// No limit on a run's cosequence but the word: 2^64.
const UNLIMITED: u128 = 1 << 64;

/// No cap on a run's steps.
const NO_CAP: u64 = u64::MAX;

/// No floor on a run's quotients: none reaches it.
const NO_FLOOR: u128 = u128::MAX;

// q-reconstruct-time.1 cited: lehmer-1938 §1, via collins-1995 §3 -- the first partial quotients come from approximations of a few figures, and formula (6) recovers the large remainders from the convergents; jebelean-1993 §2, via collins-1995 §3 -- the partial cosequence is computed on pairs of double digits; collins-1995 §3 attributes its condition (3.3) to Collins 1980, and §4.1, Figure 4 -- ISMC: the quotients of a run come from the leading digits, and the run is applied to the remainders and the cosequence in one linear combination each; monagan-2004 §2.1 -- the Euclidean algorithm may be accelerated by a constant factor using Lehmer's algorithm, 3.6 at 1550 bits
/// The Euclidean algorithm on A₁ ≥ A₂ under way: the two remainders, the
/// magnitudes |t| of the cosequence they belong to where it is tracked,
/// and two scratch buffers the combinations are written to and swapped in.
/// In the extended Euclidean algorithm on (M, u), A₁ = M index 1 and A₂ =
/// u index 2, t₁ = 0 and t₂ = 1, and `t_i` is positive for even i and
/// negative for odd i ≥ 3. So `|t_{i+2}| = |t_i| + Q_i·|t_{i+1}|`.
struct Euclid {
    a1: Vec<u64>,
    a2: Vec<u64>,
    v1: Vec<u64>,
    v2: Vec<u64>,
    /// Whether A₂'s index is odd.
    odd: bool,
    s1: Vec<u64>,
    s2: Vec<u64>,
}

impl Euclid {
    /// A₁ ≥ A₂, A₂ at index 2, and no cosequence. The other buffers are
    /// allocated where a combination first writes them.
    fn new(a1: Vec<u64>, a2: Vec<u64>) -> Euclid {
        debug_assert!(cmp(&a1, &a2).is_ge(), "A₁ ≥ A₂");
        debug_assert!(
            a1.last() != Some(&0) && a2.last() != Some(&0),
            "no zero limb at the top"
        );
        Euclid {
            a1,
            a2,
            v1: Vec::new(),
            v2: Vec::new(),
            odd: false,
            s1: Vec::new(),
            s2: Vec::new(),
        }
    }

    /// The partial cosequence of the leading parts of A₁ and A₂, at most
    /// `cap` ≥ 1 steps, ended after the first whose quotient reaches
    /// `floor`: its first element below 2^64, and every later one below
    /// `limit` ≤ 2^64.
    fn run(&self, limit: u128, cap: u64, floor: u128) -> Run {
        // q-reconstruct-time.2 design -- the leading parts are 128 bits at every size, and a number of at most 128 bits is its own leading part, whose quotients are exact, not ISMC's single digits for numbers of four words or less (not measured)
        let h = bits(&self.a1).saturating_sub(128);
        cosequence(
            top(&self.a1, h),
            top(&self.a2, h),
            h == 0,
            limit,
            cap,
            floor,
        )
    }

    /// A₁ and A₂ become A_{k+1} and A_{k+2} of the remainder sequence from
    /// A₁ at index 1, and the cosequence follows where `cosequence` says.
    fn apply(&mut self, run: &Run, cosequence: bool) {
        let Run {
            k, u1, w1, u2, w2, ..
        } = *run;
        debug_assert!(k >= 1);
        // A_i = u·A₁ − w·A₂ for odd i, w·A₂ − u·A₁ for even i.
        if k % 2 == 0 {
            lin_sub(&mut self.s1, &self.a1, u1, &self.a2, w1);
            lin_sub(&mut self.s2, &self.a2, w2, &self.a1, u2);
        } else {
            lin_sub(&mut self.s1, &self.a2, w1, &self.a1, u1);
            lin_sub(&mut self.s2, &self.a1, u2, &self.a2, w2);
            self.odd = !self.odd;
        }
        mem::swap(&mut self.a1, &mut self.s1);
        mem::swap(&mut self.a2, &mut self.s2);
        if cosequence {
            // |t_i| = u·|t₁| + w·|t₂| for every relative index i.
            lin_add(&mut self.s1, &self.v1, u1, &self.v2, w1);
            lin_add(&mut self.s2, &self.v1, u2, &self.v2, w2);
            mem::swap(&mut self.v1, &mut self.s1);
            mem::swap(&mut self.v2, &mut self.s2);
        }
    }

    /// One step of the Euclidean algorithm on the whole numbers, A₂ > 0,
    /// and its quotient.
    fn full_step(&mut self, cosequence: bool) -> BigUint {
        // q-reconstruct-time.6 design -- a step whose quotient the leading parts do not give divides through num-bigint, not by a long division on the module's limbs (log 0018-1)
        let (q, r) = big(&self.a1).div_rem(&big(&self.a2));
        mem::swap(&mut self.a1, &mut self.a2);
        set(&mut self.a2, &r);
        if cosequence {
            let v = big(&self.v1) + &q * big(&self.v2);
            mem::swap(&mut self.v1, &mut self.v2);
            set(&mut self.v2, &v);
        }
        self.odd = !self.odd;
        q
    }

    /// The rational A₂/V₂ in the sign A₂'s index gives it, where gcd(A₂,
    /// V₂) = 1; None where it is not.
    fn rational(self) -> Option<Rational> {
        let sign = if self.odd { Sign::Minus } else { Sign::Plus };
        let (num, den) = (big(&self.a2), big(&self.v2));
        // q-lift.14 cited: collins-1995 §2 -- RATCONVERT applied to M = 12 and U = 5 outputs (−2, 2), since it does not check gcd(B, M) = 1; §4.3 -- gcd(A, B) = gcd(B, M), and A and B are at most about half the length of M
        if !coprime_limbs(self.a2, self.v2) {
            return None;
        }
        Some(Rational {
            num: BigInt::from_biguint(sign, num),
            den,
        })
    }
}

/// k steps of the Euclidean algorithm on leading parts, and the
/// magnitudes of their cosequences at relative indices k + 1, (u₁, w₁),
/// and k + 2, (u₂, w₂): `A_i = u_i·A₁ − w_i·A₂` for odd i and the negative
/// for even i, A₁ index 1 and A₂ index 2.
#[derive(Clone, Copy)]
struct Run {
    k: u32,
    u1: u64,
    w1: u64,
    u2: u64,
    w2: u64,
    /// The largest quotient of the k steps, the first where two are equal,
    /// and its step, 1 to k; 0 and 0 where k = 0.
    largest: u128,
    at: u32,
}

impl Run {
    const NONE: Run = Run {
        k: 0,
        u1: 1,
        w1: 0,
        u2: 0,
        w2: 1,
        largest: 0,
        at: 0,
    };
}

/// The partial cosequence of a ≥ b: the steps of the Euclidean algorithm
/// on (a, b) whose quotients are those of the numbers a and b lead, at
/// most `cap` ≥ 1 of them, ended after the first whose quotient reaches
/// `floor`. The first step's cosequence element stays below 2^64, and
/// every later one's below `limit` ≤ 2^64, so every quotient taken lies
/// below 2^64. Where `exact`, a and b are the numbers.
fn cosequence(mut a: u128, mut b: u128, exact: bool, limit: u128, cap: u64, floor: u128) -> Run {
    debug_assert!(a >= b && limit <= UNLIMITED && cap >= 1);
    let mut run = Run::NONE;
    while b > 0 && u64::from(run.k) < cap {
        let (q, r) = divide(a, b);
        // w₃ < 2^64, past the first step w₃ < limit ≤ 2^64, and u₃ ≤ w₃.
        let Some(w3) = q
            .checked_mul(u128::from(run.w2))
            .and_then(|x| x.checked_add(u128::from(run.w1)))
        else {
            break;
        };
        // q-reconstruct-time.13 cited: collins-1995 §4.1, Figure 3 -- the restriction |v_{k+2}| < m need not prevent DDRPCC from doing one step, hence "|v₂| < m or v₁ ≤ 1" in its output specifications
        if w3 >= UNLIMITED || run.k > 0 && w3 >= limit {
            break;
        }
        // Collins's condition (3.3): the quotient of the leading parts is
        // the true one where the remainder and its distance to b are at
        // least the cosequence's.
        if !exact && (r < w3 || b - r < u128::from(run.w2) + w3) {
            break;
        }
        let u3 = u128::from(run.u1) + q * u128::from(run.u2);
        let (largest, at) = if q > run.largest {
            (q, run.k + 1)
        } else {
            (run.largest, run.at)
        };
        (a, b) = (b, r);
        run = Run {
            k: run.k + 1,
            u1: run.u2,
            w1: run.w2,
            u2: u3.narrow(),
            w2: w3.narrow(),
            largest,
            at,
        };
        if q >= floor {
            break;
        }
    }
    run
}

// q-reconstruct-time.12 cited: collins-1995 §3 -- lines 3, 4, and 5 of DDPCC should not be implemented in a straightforward way, bit shifting and subtraction should be used instead
/// ⌊a/b⌋ and a mod b for a ≥ b > 0: a quotient below 4 by subtraction,
/// any other by a 64-bit division where a fits 64 bits, else by a 128-bit
/// one.
fn divide(a: u128, b: u128) -> (u128, u128) {
    debug_assert!(b > 0 && a >= b, "a ≥ b > 0");
    let mut r = a - b;
    for q in 1..4 {
        if r < b {
            return (q, r);
        }
        r -= b;
    }
    if a >> 64 == 0 {
        let (a, b): (u64, u64) = (a.narrow(), b.narrow());
        (u128::from(a / b), u128::from(a % b))
    } else {
        (a / b, a % b)
    }
}

// q-reconstruct-time.5 cited: collins-1995 §4.2 -- the linear combinations can be formed in one pass through the digits since the signs of the results are known in advance
/// a·x − b·y, known nonnegative and no longer than the longer of x and y,
/// into `out`, in one pass over the limbs.
fn lin_sub(out: &mut Vec<u64>, x: &[u64], a: u64, y: &[u64], b: u64) {
    out.clear();
    out.reserve(x.len().max(y.len()) + 2);
    let (mut cx, mut cy, mut borrow) = (0u64, 0u64, false);
    for limb_index in 0..x.len().max(y.len()) {
        let px = u128::from(a) * u128::from(limb(x, limb_index)) + u128::from(cx);
        let py = u128::from(b) * u128::from(limb(y, limb_index)) + u128::from(cy);
        (cx, cy) = ((px >> 64) as u64, (py >> 64) as u64);
        let (lx, ly): (u64, u64) = (
            (px & u128::from(u64::MAX)).narrow(),
            (py & u128::from(u64::MAX)).narrow(),
        );
        let (difference, b1) = lx.overflowing_sub(ly);
        let (difference, b2) = difference.overflowing_sub(u64::from(borrow));
        borrow = b1 || b2;
        out.push(difference);
    }
    debug_assert_eq!(
        u128::from(cx),
        u128::from(cy) + u128::from(borrow),
        "a·x < b·y"
    );
    trim(out);
}

// q-reconstruct-time.5 cited: collins-1995 §4.2 -- the linear combinations can be formed in one pass through the digits since the signs of the results are known in advance
/// a·x + b·y into `out`, in one pass over the limbs.
fn lin_add(out: &mut Vec<u64>, x: &[u64], a: u64, y: &[u64], b: u64) {
    out.clear();
    out.reserve(x.len().max(y.len()) + 2);
    let (mut cx, mut cy, mut carry) = (0u64, 0u64, false);
    for limb_index in 0..x.len().max(y.len()) {
        let px = u128::from(a) * u128::from(limb(x, limb_index)) + u128::from(cx);
        let py = u128::from(b) * u128::from(limb(y, limb_index)) + u128::from(cy);
        (cx, cy) = ((px >> 64) as u64, (py >> 64) as u64);
        // At most one of the two carries.
        let (lx, ly): (u64, u64) = (
            (px & u128::from(u64::MAX)).narrow(),
            (py & u128::from(u64::MAX)).narrow(),
        );
        let (sum, c1) = lx.overflowing_add(ly);
        let (sum, c2) = sum.overflowing_add(u64::from(carry));
        carry = c1 || c2;
        out.push(sum);
    }
    let high = u128::from(cx) + u128::from(cy) + u128::from(carry);
    out.extend::<[u64; 2]>([
        (high & u128::from(u64::MAX)).narrow(),
        (high >> 64).narrow(),
    ]);
    trim(out);
}

fn limb(x: &[u64], i: usize) -> u64 {
    x.get(i).copied().unwrap_or(0)
}

fn limbs(x: &BigUint) -> Vec<u64> {
    let mut out = Vec::new();
    set(&mut out, x);
    out
}

/// x on limbs.
fn wide(x: u128) -> Vec<u64> {
    let mut out = vec![(x & u128::from(u64::MAX)).narrow(), (x >> 64).narrow()];
    trim(&mut out);
    out
}

fn set(out: &mut Vec<u64>, x: &BigUint) {
    out.clear();
    out.extend(x.iter_u64_digits());
    trim(out);
}

fn big(x: &[u64]) -> BigUint {
    if x.len() <= 2 {
        return BigUint::from(u128::from(limb(x, 0)) | u128::from(limb(x, 1)) << 64);
    }
    BigUint::new(
        x.iter()
            .flat_map(|&l| -> [u32; 2] { [(l & u64::from(u32::MAX)).narrow(), (l >> 32).narrow()] })
            .collect(),
    )
}

fn trim(x: &mut Vec<u64>) {
    while x.last() == Some(&0) {
        x.pop();
    }
}

/// L(x), the bit length.
fn bits(x: &[u64]) -> u64 {
    x.last()
        .map_or(0, |&l| 64 * x.len() as u64 - u64::from(l.leading_zeros()))
}

/// x against y, each on limbs, least significant first, with no zero limb
/// at the top.
pub(crate) fn cmp(x: &[u64], y: &[u64]) -> Ordering {
    x.len()
        .cmp(&y.len())
        .then_with(|| x.iter().rev().cmp(y.iter().rev()))
}

/// ⌊x/2^h⌋, which is below 2^128.
fn top(x: &[u64], h: u64) -> u128 {
    let (i, s) = ((h / 64) as usize, (h % 64) as u32);
    let low = u128::from(limb(x, i)) | u128::from(limb(x, i + 1)) << 64;
    if s == 0 {
        low
    } else {
        low >> s | u128::from(limb(x, i + 2)) << (128 - s)
    }
}

/// x mod d, d > 0.
fn remainder(x: &[u64], d: u64) -> u64 {
    debug_assert!(d > 0, "d > 0");
    x.iter().rev().fold(0, |r, &l| {
        (((u128::from(r) << 64) | u128::from(l)) % u128::from(d)).narrow()
    })
}

fn word_gcd(mut x: u64, mut y: u64) -> u64 {
    while y != 0 {
        (x, y) = (y, x % y);
    }
    x
}
