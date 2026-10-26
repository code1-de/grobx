//! The field contract: coefficients, their arithmetic, and the row format of
//! the reduction matrix with its row operations. GF(p) has two
//! implementations here: `Gf31` for the primes below 2^31, and `Gf64` for
//! those from 2^31 to 2^64. `Gf31Pair` and `Gf64Pair` compute over two
//! primes at once, one lane each.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _mm_add_epi64, _mm_and_si128, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_mul_epu32,
    _mm_set_epi64x, _mm_set1_epi64x, _mm_shuffle_epi32, _mm_srai_epi32, _mm_storeu_si128,
    _mm_sub_epi64,
};
use std::fmt::Debug;

use crate::id::MonoId;
use crate::narrow::Narrow;

/// A coefficient field. A sparse row is its column indices, increasing, and
/// its coefficients in the same order. A dense row is a slice of `Acc`, one
/// entry per column, zero where nothing is loaded.
pub trait Field: Copy + Debug + Send + Sync + 'static {
    type Coef: Copy + Eq + Ord + Default + Debug + Send + Sync + 'static;
    type Acc: Copy + Default + Debug + Send + Sync + 'static;

    fn characteristic(&self) -> u64;
    /// v mod p.
    fn coef(&self, v: u64) -> Self::Coef;
    /// The coefficient's value in [0, p).
    fn value(&self, c: Self::Coef) -> u64;
    fn add(&self, a: Self::Coef, b: Self::Coef) -> Self::Coef;
    fn sub(&self, a: Self::Coef, b: Self::Coef) -> Self::Coef;
    fn neg(&self, a: Self::Coef) -> Self::Coef;
    fn mul(&self, a: Self::Coef, b: Self::Coef) -> Self::Coef;
    /// The inverse, or None for 0.
    fn inv(&self, a: Self::Coef) -> Option<Self::Coef>;

    // reduction-time.3 design -- the two row operations are unsafe fns of the trait, whose callers guarantee the columns, not a second, unchecked method beside each checked one (not measured)
    /// Sets the dense row's entries at `cols` to `coefs`. The entries must be
    /// zero before.
    ///
    /// # Safety
    ///
    /// Every column of `cols` lies below `dense.len()`, and `cols` and
    /// `coefs` have one length.
    unsafe fn load(&self, dense: &mut [Self::Acc], cols: &[MonoId], coefs: &[Self::Coef]);
    /// dense −= c · row, for the sparse row `cols`, `coefs`.
    ///
    /// # Safety
    ///
    /// As `load`.
    unsafe fn sub_mul(
        &self,
        dense: &mut [Self::Acc],
        cols: &[MonoId],
        coefs: &[Self::Coef],
        c: Self::Coef,
    );
    /// The entry's value in [0, p), and the entry set to zero.
    fn take(&self, entry: &mut Self::Acc) -> Self::Coef;

    // q-later-runs-time.1 cited: graebe-1993 §4, via demin-2026 §4.2 -- a multimodular coefficient arithmetic over the base ring of the product of Z/p_i Z, the computations at the primes carried out in common; dekleine-2001 §1 and §3, via demin-2026 §4.2 -- Buchberger's algorithm over Z_p1 × … × Z_pb, the polynomial operations done simultaneously on b primes to amortize the monomial cost; demin-2024 §4.4, independently -- the apply stage glues the polynomial coefficients together and computes over N-tuples of primes
    /// How many primes the field computes over at once: a coefficient
    /// holds a residue at each, its lane.
    const LANES: usize = 1;

    /// The coefficient's residue at lane k, k < `LANES`, in [0, `p_k`).
    fn lane(&self, c: Self::Coef, k: usize) -> u64 {
        debug_assert_eq!(k, 0, "a field of one lane");
        self.value(c)
    }

    /// Whether the coefficient is nonzero at every lane.
    fn is_unit(&self, c: Self::Coef) -> bool {
        c != Self::Coef::default()
    }

    /// The integer the decimal digits `digits` write, at every lane.
    fn decimal(&self, digits: &str) -> Self::Coef {
        self.coef(decimal_residue(digits, self.characteristic()))
    }

    // row-traffic-time.8 design -- `Gf31` alone reduces bands, not `Gf64` and `Gf64Pair` too, whose row operation multiplies 64-bit residues by Shoup's product and which run the replays over ℚ (log 0044-1)
    // row-traffic-time.13 design -- `Gf31Pair`, the field of a pair learn over ℚ, takes no band, not bands of two `i64` lanes an entry, 64 bytes a column (log 0046-2)
    /// Whether the field reduces bands, and has `load_band`,
    /// `sub_mul_band`, and `sub_mul_band_row`: `Gf31` alone. A field
    /// without bands keeps the three methods' default bodies, which are
    /// never called.
    const BANDS: bool = false;

    /// Sets band row k's entries at `cols` to `coefs`. The entries must be
    /// zero before.
    ///
    /// # Safety
    ///
    /// Every column of `cols` lies below `band.len()`, `cols` and `coefs`
    /// have one length, and k < BAND.
    #[allow(
        clippy::unreachable,
        reason = "a band is reduced only where `matrix::takes_bands` holds, which needs `BANDS`, and every method of `Reducer` that calls a band operation debug-asserts it"
    )]
    unsafe fn load_band(
        &self,
        _: &mut [[Self::Acc; BAND]],
        _: usize,
        _: &[MonoId],
        _: &[Self::Coef],
    ) {
        unreachable!(
            "{field} reduces no band",
            field = std::any::type_name::<Self>()
        )
    }

    /// band[col][k] −= c[k] · v for every entry (col, v) of the sparse row
    /// `cols`, `coefs` and every band row k.
    ///
    /// # Safety
    ///
    /// Every column of `cols` lies below `band.len()`, and `cols` and
    /// `coefs` have one length.
    #[allow(
        clippy::unreachable,
        reason = "a band is reduced only where `matrix::takes_bands` holds, which needs `BANDS`, and every method of `Reducer` that calls a band operation debug-asserts it"
    )]
    unsafe fn sub_mul_band(
        &self,
        _: &mut [[Self::Acc; BAND]],
        _: &[MonoId],
        _: &[Self::Coef],
        _: &[Self::Coef; BAND],
    ) {
        unreachable!(
            "{field} reduces no band",
            field = std::any::type_name::<Self>()
        )
    }

    /// band[col][k] −= c · v for every entry (col, v) of the sparse row,
    /// band row k alone.
    ///
    /// # Safety
    ///
    /// As `load_band`.
    #[allow(
        clippy::unreachable,
        reason = "a band is reduced only where `matrix::takes_bands` holds, which needs `BANDS`, and every method of `Reducer` that calls a band operation debug-asserts it"
    )]
    unsafe fn sub_mul_band_row(
        &self,
        _: &mut [[Self::Acc; BAND]],
        _: usize,
        _: &[MonoId],
        _: &[Self::Coef],
        _: Self::Coef,
    ) {
        unreachable!(
            "{field} reduces no band",
            field = std::any::type_name::<Self>()
        )
    }
}

// row-traffic-time.3 tuned -- a band holds four rows, not two or eight (log 0044-1)
/// The rows of a band, reduced in one sweep over a matrix's columns
/// (`matrix::Reducer::reduce_band`).
pub const BAND: usize = 4;

/// The most decimal digits a u64 holds whatever they are: 10^19 − 1 lies
/// below 2^64.
pub(crate) const U64_DIGITS: usize = 19;

/// The integer the decimal digits `digits`, at most `U64_DIGITS`, write.
pub(crate) fn decimal_value(digits: &[u8]) -> u64 {
    debug_assert!(
        digits.len() <= U64_DIGITS,
        "{len} digits past a u64's {U64_DIGITS}",
        len = digits.len()
    );
    digits
        .iter()
        .fold(0u64, |d, b| d * 10 + u64::from(b - b'0'))
}

/// The integer the decimal digits `digits` write, mod p, read `U64_DIGITS`
/// digits at a time.
pub(crate) fn decimal_residue(digits: &str, p: u64) -> u64 {
    digits
        .as_bytes()
        .chunks(U64_DIGITS)
        .fold(0u64, |acc, chunk| {
            let d = decimal_value(chunk);
            ((u128::from(acc) * 10u128.pow(chunk.len().narrow()) + u128::from(d)) % u128::from(p))
                .narrow()
        })
}

// reduction-time.1 design -- `load` and `sub_mul` write the dense row with no bounds check per entry, the callers' guarantee in its place, not a check per entry (log 0030-1)
// reduction-time.2 design -- the write goes through `get_unchecked_mut`, not the dense row padded to a power of two and indexed by the column masked with its length less one (log 0030-1)
/// The dense row's entry at column `col`, with no bounds check outside the
/// test profile, whose debug assertions check it.
///
/// # Safety
///
/// `col` lies below `dense.len()`.
#[inline(always)]
unsafe fn entry<A>(dense: &mut [A], col: MonoId) -> &mut A {
    debug_assert!(
        col.index() < dense.len(),
        "column {col} past the dense row's {len} entries",
        len = dense.len()
    );
    // SAFETY: the caller's guarantee.
    unsafe { dense.get_unchecked_mut(col.index()) }
}

/// Shoup's factor ψ = ⌊c·2^64/p⌋ of a fixed multiplicand c < p, below
/// 2^64 since c < p (gfp-large-primes.5).
#[inline(always)]
pub(crate) fn shoup_factor(c: u64, p: u64) -> u64 {
    debug_assert!(c < p, "{c} ≥ {p}");
    ((u128::from(c) << 64) / u128::from(p)).narrow()
}

/// Shoup's product of any v below 2^64 by c < p, ψ = `shoup_factor(c,
/// p)`: v·c − q·p for q = ⌊v·ψ/2^64⌋, which lies in [0, 2p) and is v·c mod
/// p or that plus p. The wrapping difference is exact where 2p < 2^64.
#[inline(always)]
pub(crate) fn shoup_redundant(v: u64, c: u64, psi: u64, p: u64) -> u64 {
    let q = ((u128::from(v) * u128::from(psi)) >> 64) as u64;
    v.wrapping_mul(c).wrapping_sub(q.wrapping_mul(p))
}

/// v·c mod p by Shoup's product, for any v below 2^64, c < p < 2^63, and
/// ψ = `shoup_factor(c, p)`.
#[inline(always)]
pub(crate) fn shoup_mul(v: u64, c: u64, psi: u64, p: u64) -> u64 {
    debug_assert!(p < 1 << 63, "{p} ≥ 2^63");
    let d = shoup_redundant(v, c, psi, p);
    // d − p wraps past d unless d ≥ p.
    d.min(d.wrapping_sub(p))
}

/// The primes below which `Gf31`'s row operations take every product
/// unreduced and reduce an entry once, by `take` (gfp-unsolved.10): c·v <
/// 2^32 for c, v < p, and fewer than 2^31 subtractions keep an entry above
/// −2^63.
const DELAY_BOUND: u32 = 1 << 16;

/// Whether the row operations over GF(p) delay their reduction.
#[inline(always)]
fn delays(p: u32) -> bool {
    p < DELAY_BOUND
}

// f4-gfp.9 cited: berthomieu-2021 §4 -- msolve's F4, its linear algebra, and fglm run over prime fields with characteristic below 2^31
/// GF(p) for a prime p < 2^31, coefficients in [0, p) as `u32`.
#[derive(Clone, Copy, Debug)]
pub struct Gf31 {
    p: u32,
    /// p² as the dense row's bound.
    p2: i64,
}

impl Gf31 {
    /// GF(p), or None unless p is a prime below 2^31.
    #[must_use]
    pub fn new(p: u64) -> Option<Gf31> {
        if p >= 1 << 31 || !is_prime(p) {
            return None;
        }
        Some(Gf31 {
            p: p.narrow(),
            p2: (p * p).narrow(),
        })
    }

    // q-lift.8 design -- the lift tests a candidate prime by gfp-large-primes.10's strong probable-prime test and builds its field without testing it again, not Gf31::new's trial division at each candidate and each run (log 0016-2)
    /// GF(p) for a prime p < 2^31 the caller has tested; p is not tested
    /// again.
    pub(crate) fn prime(p: u64) -> Gf31 {
        debug_assert!(p < 1 << 31 && is_prime_64(p), "{p} is a prime below 2^31");
        Gf31 {
            p: p.narrow(),
            p2: (p * p).narrow(),
        }
    }
}

// f4-gfp.10 practice: trial division -- the characteristic is tested for primality by trial division by every integer from 2 up to √p
fn is_prime(p: u64) -> bool {
    if p < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= p {
        if p.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

impl Field for Gf31 {
    type Coef = u32;
    /// A dense entry is reduced mod p once, by `take`. For p ≥ 2^16 it lies
    /// in [0, p²); below, it may be any value above −2^63.
    type Acc = i64;

    fn characteristic(&self) -> u64 {
        u64::from(self.p)
    }

    fn coef(&self, v: u64) -> u32 {
        (v % u64::from(self.p)).narrow()
    }

    fn value(&self, c: u32) -> u64 {
        u64::from(c)
    }

    fn add(&self, a: u32, b: u32) -> u32 {
        // a + b < 2^32, since both are below p < 2^31.
        let s = a + b;
        if s >= self.p { s - self.p } else { s }
    }

    fn sub(&self, a: u32, b: u32) -> u32 {
        if a >= b { a - b } else { a + self.p - b }
    }

    fn neg(&self, a: u32) -> u32 {
        if a == 0 { 0 } else { self.p - a }
    }

    fn mul(&self, a: u32, b: u32) -> u32 {
        (u64::from(a) * u64::from(b) % u64::from(self.p)).narrow()
    }

    fn inv(&self, a: u32) -> Option<u32> {
        if a == 0 {
            return None;
        }
        // The extended Euclidean algorithm on p and a.
        let (mut r0, mut r1) = (i64::from(self.p), i64::from(a));
        let (mut t0, mut t1) = (0i64, 1i64);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        debug_assert_eq!(r0, 1, "p is prime");
        Some(t0.rem_euclid(i64::from(self.p)).narrow())
    }

    unsafe fn load(&self, dense: &mut [i64], cols: &[MonoId], coefs: &[u32]) {
        debug_assert_eq!(cols.len(), coefs.len());
        for (&c, &v) in cols.iter().zip(coefs) {
            // SAFETY: `load`'s condition, c < dense.len().
            unsafe { *entry(dense, c) = i64::from(v) };
        }
    }

    unsafe fn sub_mul(&self, dense: &mut [i64], cols: &[MonoId], coefs: &[u32], c: u32) {
        debug_assert_eq!(cols.len(), coefs.len());
        let c = i64::from(c);
        // reduction-time.5 cited: monagan-2015 §2.2 -- unrolling the innermost loop below saved up to 34%; vanderhoeven-2016 §4.2, independently -- four by four loop unrolling as a variant of a vector operation
        // Whole chunks of four entries, then a tail of up to three; `cols`
        // and `coefs` split alike, as they have one length.
        let (cc, ct) = cols.as_chunks::<4>();
        let (vc, vt) = coefs.as_chunks::<4>();
        // gfp-unsolved.10 cited: demin-2024 §4.2 -- the delayed modular arithmetic exploits the spare leading zero bits and delays the modulo operation to just until the overflow is about to happen; dumas-2008 §2.1.2, independently -- λ(p−1)² < 2^γ products are accumulated before a reduction
        if delays(self.p) {
            // c·v < 2^32, and an entry takes fewer than 2^31 subtractions
            // between two `take`s, so it stays above −2^63.
            // SAFETY: `sub_mul`'s condition, every column below
            // dense.len().
            unsafe {
                for (k, w) in cc.iter().zip(vc) {
                    *entry(dense, k[0]) -= c * i64::from(w[0]);
                    *entry(dense, k[1]) -= c * i64::from(w[1]);
                    *entry(dense, k[2]) -= c * i64::from(w[2]);
                    *entry(dense, k[3]) -= c * i64::from(w[3]);
                }
                for (&col, &v) in ct.iter().zip(vt) {
                    *entry(dense, col) -= c * i64::from(v);
                }
            }
            return;
        }
        // f4-gfp.8 cited: monagan-2015 §2.2 -- 64-bit buffer, rows of 32-bit words, a 31-bit prime, and p² added when the value becomes negative
        // The entry is in [0, p²) and c·v ≤ (p − 1)², so the difference is
        // above −p², and one addition of p² brings it back.
        let p2 = self.p2;
        let step = |e: &mut i64, v: u32| {
            *e -= c * i64::from(v);
            if *e < 0 {
                *e += p2;
            }
        };
        // SAFETY: as above.
        unsafe {
            for (k, w) in cc.iter().zip(vc) {
                step(entry(dense, k[0]), w[0]);
                step(entry(dense, k[1]), w[1]);
                step(entry(dense, k[2]), w[2]);
                step(entry(dense, k[3]), w[3]);
            }
            for (&col, &v) in ct.iter().zip(vt) {
                step(entry(dense, col), v);
            }
        }
    }

    fn take(&self, entry: &mut i64) -> u32 {
        // Most entries of a dense row are zero, and they cost no division.
        if *entry == 0 {
            return 0;
        }
        // The residue whatever the entry's sign.
        let v = entry.rem_euclid(i64::from(self.p)).narrow();
        *entry = 0;
        v
    }

    const BANDS: bool = true;

    unsafe fn load_band(&self, band: &mut [[i64; BAND]], k: usize, cols: &[MonoId], coefs: &[u32]) {
        debug_assert_eq!(cols.len(), coefs.len());
        debug_assert!(k < BAND, "band row {k} of {BAND}");
        for (&c, &v) in cols.iter().zip(coefs) {
            // SAFETY: `load_band`'s condition, c < band.len() and k < BAND.
            unsafe { *entry(band, c).get_unchecked_mut(k) = i64::from(v) };
        }
    }

    // row-traffic-time.6 cited: berthomieu-2021 §4.1 (5) -- CPU intrinsics make the basic operations on 32-bit coefficients more efficient: four multiplications and subtractions at a time into 64-bit integers, and p² added where a value is negative
    unsafe fn sub_mul_band(
        &self,
        band: &mut [[i64; BAND]],
        cols: &[MonoId],
        coefs: &[u32],
        c: &[u32; BAND],
    ) {
        debug_assert_eq!(cols.len(), coefs.len());
        // SAFETY: `sub_mul_band`'s condition.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sub_mul_band_sse2(band, cols, coefs, c, self.p, self.p2);
        }
        // SAFETY: as above.
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            sub_mul_band_scalar(band, cols, coefs, c, self.p, self.p2);
        }
    }

    // row-traffic-time.5 cited: boyer-2016 §4.2 -- where only one of a multiline's rows has an entry, a specialized AXPY reduces one row, and lightweight tests before the loop run it only on nonzero multipliers
    unsafe fn sub_mul_band_row(
        &self,
        band: &mut [[i64; BAND]],
        k: usize,
        cols: &[MonoId],
        coefs: &[u32],
        c: u32,
    ) {
        debug_assert_eq!(cols.len(), coefs.len());
        debug_assert!(k < BAND, "band row {k} of {BAND}");
        let c = i64::from(c);
        // As `sub_mul` steps an entry, one entry per turn.
        if delays(self.p) {
            for (&col, &v) in cols.iter().zip(coefs) {
                // SAFETY: `sub_mul_band_row`'s condition, col < band.len()
                // and k < BAND.
                unsafe { *entry(band, col).get_unchecked_mut(k) -= c * i64::from(v) };
            }
            return;
        }
        let p2 = self.p2;
        for (&col, &v) in cols.iter().zip(coefs) {
            // SAFETY: as above.
            let e = unsafe { entry(band, col).get_unchecked_mut(k) };
            *e -= c * i64::from(v);
            if *e < 0 {
                *e += p2;
            }
        }
    }
}

/// `Gf31::sub_mul_band` over GF(p), p² = `p2`, by SSE2: band rows 0 and 1
/// in one 128-bit register, 2 and 3 in another, each register's two
/// products by one `pmuludq`. Below 2^16 an entry takes the products
/// unreduced, as `Gf31::sub_mul` takes them; from 2^16 it gains p² where
/// the subtraction leaves it negative.
///
/// # Safety
///
/// As `Field::sub_mul_band`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "the entry pointer is read and written by _mm_loadu_si128 and _mm_storeu_si128 alone, which take any alignment"
)]
unsafe fn sub_mul_band_sse2(
    band: &mut [[i64; BAND]],
    cols: &[MonoId],
    coefs: &[u32],
    c: &[u32; BAND],
    p: u32,
    p2: i64,
) {
    // SAFETY: SSE2 is part of baseline x86-64, which every x86-64 target
    // enables. Every column of `cols` lies below band.len(), the caller's
    // guarantee, and an entry of the band is 32 bytes, two unaligned
    // 128-bit halves.
    unsafe {
        // The multipliers in the low 32 bits of each 64-bit lane: c[0] and
        // c[1], c[2] and c[3].
        let m01 = _mm_set_epi64x(i64::from(c[1]), i64::from(c[0]));
        let m23 = _mm_set_epi64x(i64::from(c[3]), i64::from(c[2]));
        // v in the low 32 bits of both 64-bit lanes.
        let spread =
            |v: u32| _mm_shuffle_epi32::<0b01_00_01_00>(_mm_cvtsi32_si128(v.cast_signed()));
        if delays(p) {
            // c·v < 2^32, and an entry takes fewer than 2^31 subtractions
            // between two `take`s, so it stays above −2^63.
            for (&col, &v) in cols.iter().zip(coefs) {
                let v_lanes = spread(v);
                let entry_ptr = entry(band, col).as_mut_ptr().cast::<__m128i>();
                let rows01 = _mm_sub_epi64(_mm_loadu_si128(entry_ptr), _mm_mul_epu32(m01, v_lanes));
                let rows23 = _mm_sub_epi64(
                    _mm_loadu_si128(entry_ptr.add(1)),
                    _mm_mul_epu32(m23, v_lanes),
                );
                _mm_storeu_si128(entry_ptr, rows01);
                _mm_storeu_si128(entry_ptr.add(1), rows23);
            }
            return;
        }
        // An entry is in [0, p²) and c·v ≤ (p − 1)², so the difference is
        // above −p², and p² masked in by the lane's sign brings it back.
        let p2 = _mm_set1_epi64x(p2);
        // Each 64-bit lane all ones where it is negative: the sign of its
        // high 32 bits, spread over both halves.
        let negative =
            |lanes: __m128i| _mm_shuffle_epi32::<0b11_11_01_01>(_mm_srai_epi32::<31>(lanes));
        for (&col, &v) in cols.iter().zip(coefs) {
            let v_lanes = spread(v);
            let entry_ptr = entry(band, col).as_mut_ptr().cast::<__m128i>();
            let rows01 = _mm_sub_epi64(_mm_loadu_si128(entry_ptr), _mm_mul_epu32(m01, v_lanes));
            let rows23 = _mm_sub_epi64(
                _mm_loadu_si128(entry_ptr.add(1)),
                _mm_mul_epu32(m23, v_lanes),
            );
            let rows01 = _mm_add_epi64(rows01, _mm_and_si128(negative(rows01), p2));
            let rows23 = _mm_add_epi64(rows23, _mm_and_si128(negative(rows23), p2));
            _mm_storeu_si128(entry_ptr, rows01);
            _mm_storeu_si128(entry_ptr.add(1), rows23);
        }
    }
}

/// `Gf31::sub_mul_band` elsewhere than x86-64: the same arithmetic, one
/// band row at a time.
///
/// # Safety
///
/// As `Field::sub_mul_band`.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sub_mul_band_scalar(
    band: &mut [[i64; BAND]],
    cols: &[MonoId],
    coefs: &[u32],
    c: &[u32; BAND],
    p: u32,
    p2: i64,
) {
    for (&col, &v) in cols.iter().zip(coefs) {
        // SAFETY: the caller's guarantee, col < band.len().
        let e = unsafe { entry(band, col) };
        for (e, &c) in e.iter_mut().zip(c) {
            *e -= i64::from(c) * i64::from(v);
            if !delays(p) && *e < 0 {
                *e += p2;
            }
        }
    }
}

// gfp-large-primes.1 design -- the primes from 2^31 to 2^64 take one field of u64 coefficients, not a field of u32 coefficients below 2^32 beside one of u64 coefficients from 2^32 (log 0015-1)
/// GF(p) for a prime p with 2^31 ≤ p < 2^64, coefficients in [0, p) as
/// `u64`.
#[derive(Clone, Copy, Debug)]
pub struct Gf64 {
    p: u64,
}

impl Gf64 {
    /// GF(p), or None unless p is a prime with 2^31 ≤ p < 2^64.
    #[must_use]
    pub fn new(p: u64) -> Option<Gf64> {
        (p >= 1 << 31 && is_prime_64(p)).then_some(Gf64 { p })
    }

    // q-modulus-time.8 design -- a replay's field is built for the prime the lift's draw tested, without testing it again, not Gf64::new's strong probable-prime test at each replay (log 0036-1)
    /// GF(p) for a prime p with 2^31 ≤ p < 2^64 the caller has tested; p
    /// is not tested again.
    pub(crate) fn prime(p: u64) -> Gf64 {
        debug_assert!(p >= 1 << 31 && is_prime_64(p), "{p} is a prime past 2^31");
        Gf64 { p }
    }
}

/// a·b mod p, p > 0.
#[inline(always)]
pub(crate) fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    debug_assert!(p > 0, "a product modulo 0");
    (u128::from(a) * u128::from(b) % u128::from(p)).narrow()
}

// gfp-large-primes.9 cited: dumas-2008 §2.1.1 -- multiplication is followed by machine remaindering while division is performed via the extended gcd algorithm
// q-lift.41 cited: dumas-2008 §2.1.1 -- multiplication is followed by machine remaindering while division is performed via the extended gcd algorithm
/// a⁻¹ mod p for a prime p and a in [1, p).
pub(crate) fn inverse_mod(a: u64, p: u64) -> u64 {
    debug_assert!(a != 0 && a < p, "{a} is no unit below {p}");
    // The extended Euclidean algorithm on p and a in i128, which holds every
    // p < 2^64; every |t| stays at most p.
    let (mut r0, mut r1) = (i128::from(p), i128::from(a));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    debug_assert_eq!(r0, 1, "p is prime");
    t0.rem_euclid(i128::from(p)).narrow()
}

/// The twelve prime bases from 2 to 37.
const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

// gfp-large-primes.10 cited: miller-1976 _Tests for primality_, via rabin-1980 §1 -- a base b with ((b^((n−1)/2^k) mod n) − 1, n) ≠ 1, n shows n composite; rabin-1980 §1, Theorem 1 -- at least 3(n−1)/4 of the bases are witnesses of a composite n; sorenson-2017 §1, Theorem 1.1 -- ψ₁₂, the least strong pseudoprime to the first twelve prime bases, is 318665857834031151167461, past 2^64
/// Whether n is prime, for every n < 2^64 (gfp-large-primes.10).
#[allow(
    clippy::many_single_char_names,
    reason = "n, s, d, a, x are the Miller-Rabin notation of miller-1976 the comments write"
)]
pub(crate) fn is_prime_64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for prime_base in BASES {
        if n.is_multiple_of(prime_base) {
            return n == prime_base;
        }
    }
    // n is odd and past 37: n − 1 = 2^s·d with d odd.
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'bases: for a in BASES {
        // x = a^d mod n by square and multiply.
        let (mut x, mut base, mut exponent) = (1, a, d);
        while exponent > 0 {
            if exponent & 1 == 1 {
                x = mul_mod(x, base, n);
            }
            base = mul_mod(base, base, n);
            exponent >>= 1;
        }
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

impl Field for Gf64 {
    type Coef = u64;
    // gfp-large-primes.4 cited: demin-2024 §4.2 -- the generic arithmetic takes remainder modulo a prime after each multiplication; if the prime fits in 27 bits, we opt to use delayed arithmetic; otherwise, we use the generic arithmetic; dumas-2008 §2.1.2 -- λ accumulations without any division where λ(p−1)² < 2^γ
    /// A dense entry lies in [0, 2p) between two operations where p < 2^62,
    /// and in [0, p) otherwise.
    type Acc = u64;

    fn characteristic(&self) -> u64 {
        self.p
    }

    fn coef(&self, v: u64) -> u64 {
        v % self.p
    }

    fn value(&self, c: u64) -> u64 {
        c
    }

    fn add(&self, a: u64, b: u64) -> u64 {
        // a − (p − b) where that does not borrow, a + b < p where it would.
        let t = self.p - b;
        if a >= t { a - t } else { a + b }
    }

    fn sub(&self, a: u64, b: u64) -> u64 {
        let (r, borrow) = a.overflowing_sub(b);
        if borrow { r.wrapping_add(self.p) } else { r }
    }

    fn neg(&self, a: u64) -> u64 {
        if a == 0 { 0 } else { self.p - a }
    }

    // gfp-large-primes.8 design -- every product outside a row operation takes the u128 remainder, not Barrett's product with a pre-inverse of p (log 0015-1)
    fn mul(&self, a: u64, b: u64) -> u64 {
        mul_mod(a, b, self.p)
    }

    fn inv(&self, a: u64) -> Option<u64> {
        (a != 0).then(|| inverse_mod(a, self.p))
    }

    unsafe fn load(&self, dense: &mut [u64], cols: &[MonoId], coefs: &[u64]) {
        debug_assert_eq!(cols.len(), coefs.len());
        for (&c, &v) in cols.iter().zip(coefs) {
            // SAFETY: `load`'s condition, c < dense.len().
            unsafe { *entry(dense, c) = v };
        }
    }

    #[allow(
        clippy::many_single_char_names,
        reason = "c, d, e, p, q are the notation of vanderhoeven-2016 and harvey-2014 the comments write"
    )]
    unsafe fn sub_mul(&self, dense: &mut [u64], cols: &[MonoId], coefs: &[u64], c: u64) {
        debug_assert_eq!(cols.len(), coefs.len());
        let p = self.p;
        // gfp-large-primes.7 cited: vanderhoeven-2016 §2.2.3 -- in order to remove the branching involved in line 4 of Function 6, we replace it by d = min(d, d − p); harvey-2014 §2 -- the modular subtraction works for any p < β by using the borrow generated by the subtraction T = X − Y, from Niels Möller by personal communication
        // e − d, plus p where it borrowed: in [0, p) for e and d in [0, p).
        let subtract = |e: &mut u64, d: u64| {
            let (difference, borrow) = e.overflowing_sub(d);
            *e = difference.wrapping_add(p & 0u64.wrapping_sub(u64::from(borrow)));
        };
        // q-modulus-time.3 cited: harvey-2014 §3 -- Shoup's product works for any 0 ≤ T < β, and a redundant representation in [0, 2p) removes the adjustment of its result, for p < β/4; gokavarapu-2026 §8.2, independently -- redundant representatives in [0, kp), the invariant a ≡ a_true mod p
        // q-modulus-time.4 cited: harvey-2014 §3 -- the redundant representation requires the stronger condition p < β/4
        if p < 1 << 62 {
            let psi = shoup_factor(c, p);
            let twice = 2 * p;
            // q-modulus-time.5 tuned -- the redundant row operation takes one entry per turn, not four entries per turn as Gf31::sub_mul (log 0036-1)
            for (&col, &v) in cols.iter().zip(coefs) {
                // v·c − q·p lies in [0, 2p), and 2p < 2^63.
                let d = shoup_redundant(v, c, psi, p);
                // SAFETY: `sub_mul`'s condition, col < dense.len().
                let e = unsafe { entry(dense, col) };
                // e and d lie in [0, 2p), so e − d lies in (−2p, 2p), and
                // 2p added where it borrowed brings it into [0, 2p).
                let (difference, borrow) = e.overflowing_sub(d);
                *e = difference.wrapping_add(twice & 0u64.wrapping_sub(u64::from(borrow)));
            }
            return;
        }
        // gfp-large-primes.5 cited: vanderhoeven-2016 §2.2.2, Function 8 and Proposition 4 -- for a fixed multiplicand y precompute ψ := ⌊2^n y/p⌋, c = (xψ) >> n, d = xy − cp, one correction; if m ≤ n − 1 line 2 is computed in one word; §2.3.1, Remark 8 -- Barrett's approach is expected to be always faster for this task; farnqvist-2005 §4.2, independently, via harvey-2014 §2, attributes it to Victor Shoup by e-mail [13] -- y_pinv = 2⌊2^(W−1)·y/p⌋ precomputed, q = ⌊x·y_pinv/2^W⌋, and r = xy − qp is the answer or p too big
        if p < 1 << 63 {
            let psi = shoup_factor(c, p);
            for (&col, &v) in cols.iter().zip(coefs) {
                let d = shoup_mul(v, c, psi, p);
                // SAFETY: `sub_mul`'s condition, col < dense.len().
                subtract(unsafe { entry(dense, col) }, d);
            }
            return;
        }
        // gfp-large-primes.6 design -- where p ≥ 2^63 each product is the u128 remainder, not the pre-inverse with v·c − q·p in two words (log 0015-1)
        for (&col, &v) in cols.iter().zip(coefs) {
            let d = mul_mod(v, c, p);
            // SAFETY: as above.
            subtract(unsafe { entry(dense, col) }, d);
        }
    }

    fn take(&self, entry: &mut u64) -> u64 {
        // e − p wraps past e unless e ≥ p: e in [0, 2p) comes back in
        // [0, p), and e in [0, p) as it is.
        let e = std::mem::take(entry);
        e.min(e.wrapping_sub(self.p))
    }
}

// q-learn-time.2 design -- the pair learn's field is Gf31Pair, two primes of [2^30, 2^31) with u32 residues and an i64 dense entry per lane, not Gf64Pair at two primes of [2^61, 2^62) whose first lane's image the lift could combine as a replay's (log 0036-1, 0052-1)
// q-learn-time.3 design -- each lane's row operation adds p_k² where the entry goes below 0, whatever its prime, and take reduces the entry by rem_euclid, not Gf31's two loops with the delayed form below 2^16 (not measured)
/// GF(p) × GF(q) for two distinct primes below 2^31: a coefficient is a
/// pair of residues, one per lane, and every operation runs lane by lane
/// as `Gf31` runs it for a prime past 2^16.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gf31Pair {
    lanes: [Gf31; 2],
}

impl Gf31Pair {
    // q-lift.8 design -- the lift tests a candidate prime by gfp-large-primes.10's strong probable-prime test and builds its field without testing it again, not Gf31::new's trial division at each candidate and each run (log 0016-2)
    /// For two distinct primes below 2^31 the caller has tested.
    pub(crate) fn primes(p: u64, q: u64) -> Gf31Pair {
        debug_assert!(p != q, "two distinct primes");
        Gf31Pair {
            lanes: [Gf31::prime(p), Gf31::prime(q)],
        }
    }
}

impl Field for Gf31Pair {
    type Coef = [u32; 2];
    /// Each lane's entry in [0, `p_k²`) between two operations.
    type Acc = [i64; 2];

    const LANES: usize = 2;

    /// The first lane's prime.
    fn characteristic(&self) -> u64 {
        u64::from(self.lanes[0].p)
    }

    fn coef(&self, v: u64) -> [u32; 2] {
        [self.lanes[0].coef(v), self.lanes[1].coef(v)]
    }

    /// The first lane's residue.
    fn value(&self, c: [u32; 2]) -> u64 {
        u64::from(c[0])
    }

    fn add(&self, a: [u32; 2], b: [u32; 2]) -> [u32; 2] {
        [self.lanes[0].add(a[0], b[0]), self.lanes[1].add(a[1], b[1])]
    }

    fn sub(&self, a: [u32; 2], b: [u32; 2]) -> [u32; 2] {
        [self.lanes[0].sub(a[0], b[0]), self.lanes[1].sub(a[1], b[1])]
    }

    fn neg(&self, a: [u32; 2]) -> [u32; 2] {
        [self.lanes[0].neg(a[0]), self.lanes[1].neg(a[1])]
    }

    fn mul(&self, a: [u32; 2], b: [u32; 2]) -> [u32; 2] {
        [self.lanes[0].mul(a[0], b[0]), self.lanes[1].mul(a[1], b[1])]
    }

    /// None where either lane is 0.
    fn inv(&self, a: [u32; 2]) -> Option<[u32; 2]> {
        Some([self.lanes[0].inv(a[0])?, self.lanes[1].inv(a[1])?])
    }

    unsafe fn load(&self, dense: &mut [[i64; 2]], cols: &[MonoId], coefs: &[[u32; 2]]) {
        debug_assert_eq!(cols.len(), coefs.len());
        for (&c, v) in cols.iter().zip(coefs) {
            // SAFETY: `load`'s condition, c < dense.len().
            unsafe { *entry(dense, c) = [i64::from(v[0]), i64::from(v[1])] };
        }
    }

    unsafe fn sub_mul(
        &self,
        dense: &mut [[i64; 2]],
        cols: &[MonoId],
        coefs: &[[u32; 2]],
        c: [u32; 2],
    ) {
        debug_assert_eq!(cols.len(), coefs.len());
        // f4-gfp.8 cited: monagan-2015 §2.2 -- 64-bit buffer, rows of 32-bit words, a 31-bit prime, and p² added when the value becomes negative
        // At each lane the entry is in [0, p_k²) and c_k·v_k ≤ (p_k − 1)²,
        // so the difference is above −p_k², and one addition of p_k² brings
        // it back.
        let p2 = [self.lanes[0].p2, self.lanes[1].p2];
        let c = [i64::from(c[0]), i64::from(c[1])];
        // q-learn-time.15 design -- each lane's correction is Gf31's branch, p² added where the entry went below 0, not monagan-2015 §2.2's branch-free form, p² & (e >> 63) added (log 0052-2)
        let step = |e: &mut [i64; 2], v: &[u32; 2]| {
            for k in 0..2 {
                e[k] -= c[k] * i64::from(v[k]);
                if e[k] < 0 {
                    e[k] += p2[k];
                }
            }
        };
        // q-learn-time.14 cited: monagan-2015 §2.2 -- unrolling the innermost loop below saved up to 34%; vanderhoeven-2016 §4.2, independently -- four by four loop unrolling as a variant of a vector operation
        // Whole chunks of four entries, then a tail of up to three; `cols`
        // and `coefs` split alike, as they have one length.
        let (cc, ct) = cols.as_chunks::<4>();
        let (vc, vt) = coefs.as_chunks::<4>();
        // SAFETY: `sub_mul`'s condition, every column below dense.len().
        unsafe {
            for (k, w) in cc.iter().zip(vc) {
                step(entry(dense, k[0]), &w[0]);
                step(entry(dense, k[1]), &w[1]);
                step(entry(dense, k[2]), &w[2]);
                step(entry(dense, k[3]), &w[3]);
            }
            for (&col, v) in ct.iter().zip(vt) {
                step(entry(dense, col), v);
            }
        }
    }

    fn take(&self, entry: &mut [i64; 2]) -> [u32; 2] {
        // Most entries of a dense row are zero, and they cost no division.
        if *entry == [0, 0] {
            return [0, 0];
        }
        let e = std::mem::take(entry);
        [
            e[0].rem_euclid(i64::from(self.lanes[0].p)).narrow(),
            e[1].rem_euclid(i64::from(self.lanes[1].p)).narrow(),
        ]
    }

    fn lane(&self, c: [u32; 2], k: usize) -> u64 {
        u64::from(c[k])
    }

    fn is_unit(&self, c: [u32; 2]) -> bool {
        c[0] != 0 && c[1] != 0
    }

    fn decimal(&self, digits: &str) -> [u32; 2] {
        [
            decimal_residue(digits, u64::from(self.lanes[0].p)).narrow(),
            decimal_residue(digits, u64::from(self.lanes[1].p)).narrow(),
        ]
    }
}

// q-later-runs-time.1 cited: graebe-1993 §4, via demin-2026 §4.2 -- a multimodular coefficient arithmetic over the base ring of the product of Z/p_i Z, the computations at the primes carried out in common; dekleine-2001 §1 and §3, via demin-2026 §4.2 -- Buchberger's algorithm over Z_p1 × … × Z_pb, the polynomial operations done simultaneously on b primes to amortize the monomial cost; demin-2024 §4.4, independently -- the apply stage glues the polynomial coefficients together and computes over N-tuples of primes
// q-later-runs-time.2 tuned -- two primes per replay, not four as demin-2024 §4.4's multi-modular computation takes, nor more (log 0039-1)
// q-later-runs-time.3 design -- the pair's primes both lie in [2^61, 2^62), each lane computing as Gf64 below 2^62, not pairs of any two primes with each lane in the row operation of its range or in Gf31 (not measured)
/// GF(p) × GF(q) for two distinct primes of [2^61, 2^62): a coefficient
/// is a pair of residues, one per lane, and every operation runs lane by
/// lane as `Gf64` runs it below 2^62.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gf64Pair {
    lanes: [Gf64; 2],
}

impl Gf64Pair {
    // q-modulus-time.8 design -- a replay's field is built for the prime the lift's draw tested, without testing it again, not Gf64::new's strong probable-prime test at each replay (log 0036-1)
    /// For two distinct primes of [2^61, 2^62) the caller has tested.
    pub(crate) fn primes(p: u64, q: u64) -> Gf64Pair {
        debug_assert!(p != q, "two distinct primes");
        debug_assert!(
            (1 << 61..1 << 62).contains(&p) && (1 << 61..1 << 62).contains(&q),
            "{p} and {q} lie in [2^61, 2^62)"
        );
        Gf64Pair {
            lanes: [Gf64::prime(p), Gf64::prime(q)],
        }
    }
}

impl Field for Gf64Pair {
    type Coef = [u64; 2];
    /// Each lane's entry in [0, `2p_k`) between two operations, as `Gf64`'s
    /// below 2^62.
    type Acc = [u64; 2];

    const LANES: usize = 2;

    /// The first lane's prime. A replay reads it only for its line in the
    /// call report (`entry::Run::report_run`).
    fn characteristic(&self) -> u64 {
        self.lanes[0].p
    }

    fn coef(&self, v: u64) -> [u64; 2] {
        [self.lanes[0].coef(v), self.lanes[1].coef(v)]
    }

    /// The first lane's residue.
    fn value(&self, c: [u64; 2]) -> u64 {
        c[0]
    }

    fn add(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [self.lanes[0].add(a[0], b[0]), self.lanes[1].add(a[1], b[1])]
    }

    fn sub(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [self.lanes[0].sub(a[0], b[0]), self.lanes[1].sub(a[1], b[1])]
    }

    fn neg(&self, a: [u64; 2]) -> [u64; 2] {
        [self.lanes[0].neg(a[0]), self.lanes[1].neg(a[1])]
    }

    fn mul(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [self.lanes[0].mul(a[0], b[0]), self.lanes[1].mul(a[1], b[1])]
    }

    /// None where either lane is 0.
    fn inv(&self, a: [u64; 2]) -> Option<[u64; 2]> {
        Some([self.lanes[0].inv(a[0])?, self.lanes[1].inv(a[1])?])
    }

    unsafe fn load(&self, dense: &mut [[u64; 2]], cols: &[MonoId], coefs: &[[u64; 2]]) {
        debug_assert_eq!(cols.len(), coefs.len());
        for (&c, &v) in cols.iter().zip(coefs) {
            // SAFETY: `load`'s condition, c < dense.len().
            unsafe { *entry(dense, c) = v };
        }
    }

    #[allow(
        clippy::many_single_char_names,
        reason = "c, d, e, p, q are the notation of harvey-2014 the comments write, k their lane"
    )]
    unsafe fn sub_mul(
        &self,
        dense: &mut [[u64; 2]],
        cols: &[MonoId],
        coefs: &[[u64; 2]],
        c: [u64; 2],
    ) {
        debug_assert_eq!(cols.len(), coefs.len());
        let p = [self.lanes[0].p, self.lanes[1].p];
        // q-modulus-time.3 cited: harvey-2014 §3 -- Shoup's product works for any 0 ≤ T < β, and a redundant representation in [0, 2p) removes the adjustment of its result, for p < β/4; gokavarapu-2026 §8.2, independently -- redundant representatives in [0, kp), the invariant a ≡ a_true mod p
        let psi = [shoup_factor(c[0], p[0]), shoup_factor(c[1], p[1])];
        let twice = [2 * p[0], 2 * p[1]];
        for (&col, v) in cols.iter().zip(coefs) {
            // SAFETY: `sub_mul`'s condition, col < dense.len().
            let e = unsafe { entry(dense, col) };
            for k in 0..2 {
                // v·c − q·p lies in [0, 2p), and 2p < 2^63.
                let d = shoup_redundant(v[k], c[k], psi[k], p[k]);
                // e and d lie in [0, 2p), so e − d lies in (−2p, 2p), and 2p
                // added where it borrowed brings it into [0, 2p).
                let (difference, borrow) = e[k].overflowing_sub(d);
                e[k] = difference.wrapping_add(twice[k] & 0u64.wrapping_sub(u64::from(borrow)));
            }
        }
    }

    fn take(&self, entry: &mut [u64; 2]) -> [u64; 2] {
        let e = std::mem::take(entry);
        [
            self.lanes[0].take(&mut { e[0] }),
            self.lanes[1].take(&mut { e[1] }),
        ]
    }

    fn lane(&self, c: [u64; 2], k: usize) -> u64 {
        c[k]
    }

    fn is_unit(&self, c: [u64; 2]) -> bool {
        c[0] != 0 && c[1] != 0
    }

    fn decimal(&self, digits: &str) -> [u64; 2] {
        [
            decimal_residue(digits, self.lanes[0].p),
            decimal_residue(digits, self.lanes[1].p),
        ]
    }
}
