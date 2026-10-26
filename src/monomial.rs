//! The monomial contract: a format stores a monomial's exponents in a fixed
//! number of words, and every operation that can leave the format says so.
//! The formats are dense exponent vectors in lanes of 8, 16, and 32 bits,
//! packed into 64-bit words, `Dense8`, `Dense16`, and `Dense32`. A call
//! computes in the narrowest that holds its input, a replay from a kept
//! shape in the shape's, and starts over in the next wider where a degree
//! passes the one it runs in.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::narrow::Narrow;

/// A degree, or an exponent, past what the format holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overflow;

/// The bits of a grevlex key, `Format::grevlex_key`, that hold the degree
/// and the exponents, at the top of a u128 (matrix-sort-time.5).
pub(crate) const KEY_BITS: u32 = 96;

/// The low bits of a grevlex key below `KEY_BITS`, where a keyed sort puts
/// its payload.
pub(crate) const PAYLOAD_BITS: u32 = u128::BITS - KEY_BITS;

/// A monomial format over a fixed number of variables. A monomial is a slice
/// of `words()` words; its total degree travels beside it.
pub trait Format: Copy + Debug + Send + Sync + 'static {
    type Word: Copy + Eq + Default + Debug + Send + Sync + 'static;
    /// The largest total degree the format holds.
    const MAX_DEGREE: u32;

    fn new(variables: usize) -> Self;
    fn variables(&self) -> usize;
    /// Words per monomial.
    fn words(&self) -> usize;
    /// Writes the monomial with these exponents, one per variable, to `out`:
    /// the degree, or Overflow where it passes `MAX_DEGREE`.
    ///
    /// # Errors
    ///
    /// `Overflow` where the degree passes `MAX_DEGREE`.
    fn encode(&self, exponents: &[u32], out: &mut [Self::Word]) -> Result<u32, Overflow>;
    fn exponent(&self, m: &[Self::Word], i: usize) -> u32;
    // result-time.2 design -- `unpack` is a required method of the trait, not a default that calls `exponent` per variable (not measured)
    /// Every exponent of m, one per variable, to `out`, whose length is the
    /// number of variables.
    fn unpack(&self, m: &[Self::Word], out: &mut [u32]);
    /// a·b to `out`, from a and b with their degrees: the degree of the
    /// product, or Overflow.
    ///
    /// # Errors
    ///
    /// `Overflow` where the degree passes `MAX_DEGREE`.
    fn mul(
        &self,
        a: &[Self::Word],
        da: u32,
        b: &[Self::Word],
        db: u32,
        out: &mut [Self::Word],
    ) -> Result<u32, Overflow>;
    /// Whether a·b = c.
    fn is_product(&self, a: &[Self::Word], b: &[Self::Word], c: &[Self::Word]) -> bool;
    /// The word of a·b at a place where a holds `a` and b holds `b`, for a
    /// product whose degree the format holds.
    fn mul_word(a: Self::Word, b: Self::Word) -> Self::Word;
    /// Whether a divides b.
    fn divides(&self, a: &[Self::Word], b: &[Self::Word]) -> bool;
    /// a / b to `out`, where b divides a: the degree of the quotient.
    fn div(&self, a: &[Self::Word], b: &[Self::Word], out: &mut [Self::Word]) -> u32;
    /// lcm(a, b) to `out`: its degree, or Overflow.
    ///
    /// # Errors
    ///
    /// `Overflow` where the degree passes `MAX_DEGREE`.
    fn lcm(
        &self,
        a: &[Self::Word],
        b: &[Self::Word],
        out: &mut [Self::Word],
    ) -> Result<u32, Overflow>;
    /// s·lcm(x, y)/x to `out`, exponent by exponent s + max(0, y − x): its
    /// degree, or Overflow where it passes `MAX_DEGREE`.
    ///
    /// # Errors
    ///
    /// `Overflow` where the degree passes `MAX_DEGREE`.
    fn mul_quotient(
        &self,
        s: &[Self::Word],
        x: &[Self::Word],
        y: &[Self::Word],
        out: &mut [Self::Word],
    ) -> Result<u32, Overflow>;
    /// The degree of lcm(a, b), which may pass `MAX_DEGREE`.
    fn lcm_degree(&self, a: &[Self::Word], b: &[Self::Word]) -> u32;
    /// Whether a and b share no variable.
    fn coprime(&self, a: &[Self::Word], b: &[Self::Word]) -> bool;
    /// a against b in the reverse lexicographic order: a > b where the last
    /// variable whose exponents differ has the smaller exponent in a.
    fn cmp_revlex(&self, a: &[Self::Word], b: &[Self::Word]) -> Ordering;
    /// a against b in the lexicographic order: a > b where the first variable
    /// whose exponents differ has the larger exponent in a.
    fn cmp_lex(&self, a: &[Self::Word], b: &[Self::Word]) -> Ordering;
    // matrix-sort-time.3 cited: monagan-2007 §2.1, Figure 1 -- grevlex is the degree, then the exponents from the last variable to the first with their comparisons inverted; after Figure 4 -- the first n exponents decide a grevlex comparison on n variables
    /// The first 96 bits of the grevlex key of m, of degree `degree`, in the
    /// top 96 bits, the low 32 bits 0. The key is the degree, then the
    /// exponents from the last variable down to the first, each complemented
    /// in its lane's bits, packed from the most significant bit. Of two
    /// monomials, the one with the smaller key is the smaller in grevlex. Two
    /// equal keys are one monomial where the key fits 96 bits but for the
    /// first variable's exponent.
    fn grevlex_key(&self, m: &[Self::Word], degree: u32) -> u128;
    /// Writes these exponents, one per variable, each at most `MAX_DEGREE`, to
    /// `out`, whatever their sum: a bound for `divides`, or exponents of no
    /// table that an order compares; not a monomial of the table.
    fn pack(&self, exponents: &[u32], out: &mut [Self::Word]);
    // many-variables-time.3 cited: roune-2012 §C.3.2 -- 64-bit divmasks have been slower in their experiments, the increased hit rate not making up for the extra overhead of dealing with more bits
    /// The divisor mask: a | b implies mask(a) & !mask(b) == 0. A variable
    /// of a sets a bit of mask(a), so masks that share no bit belong to
    /// monomials that share no variable. Each bit is set where an exponent
    /// reaches a threshold, and an lcm's exponent is the larger of its
    /// factors', so mask(lcm(a, b)) = mask(a) | mask(b).
    fn mask(&self, m: &[Self::Word]) -> u32;
}

/// The lane of a dense format: `u8`, `u16`, or `u32`, BITS bits wide.
pub trait Exponent: Copy + Eq + Ord + Default + Debug + Send + Sync + 'static {
    /// The bits of a lane.
    const BITS: u32;
    /// The largest total degree a dense format of these lanes holds,
    /// 2^(BITS−1) − 1: 127, 32767, and 2^31 − 1.
    const MAX: u32;
}

// monomial-time.4 cited: bachmann-1998 §3, "Firstly" -- the exponent values need to be less than 2^(s_e−1) instead of 2^(s_e), so that Lemma 1 (4) tests divisibility word by word
impl Exponent for u8 {
    const BITS: u32 = 8;
    const MAX: u32 = (1 << 7) - 1;
}

impl Exponent for u16 {
    const BITS: u32 = 16;
    const MAX: u32 = (1 << 15) - 1;
}

// monomial-formats.3 design -- the widest format's lanes are 32 bits, the width of a degree throughout the engine, not 64-bit lanes (not measured)
impl Exponent for u32 {
    const BITS: u32 = 32;
    // monomial-formats.4 design -- the widest format's degree bound is 2^31 − 1, so that a sum of two degrees or two exponents fits the u32 the engine computes it in, not u32::MAX with the engine's degree sums taken in u64 (not measured)
    const MAX: u32 = i32::MAX as u32;
}

// f4-gfp.11 cited: bachmann-1998 §2.2 -- a dense distributive representation is the one real choice, with fixed-length exponents no larger than a machine word and an additional degree field
// monomial-time.1 cited: bachmann-1998 §3, attributes parts of it to PoSSo 1995 [14] -- monomial operations are performed on machine words instead of directly on exponents, which sped up the computations by 1.1 to 1.7 with exponents of 16 bits and by 1.2 to 2.0 with exponents of 8 bits (Table 1)
/// One lane of type `W` per variable, packed into 64-bit words, the total
/// degree beside it. The exponent of variable i is lane i mod L of word
/// ⌊i/L⌋, L = 64 / `W::BITS` lanes, the low lanes first.
#[derive(Clone, Copy, Debug)]
pub struct Dense<W> {
    n: usize,
    /// Words per monomial, ⌈n / L⌉.
    words: usize,
    /// Mask bits per variable: ⌊32/n⌋ up to 32 variables, 1 past them,
    /// 0 for none.
    bits: u32,
    word: PhantomData<W>,
}

// monomial-formats.1 cited: bachmann-1998 §3 -- ideally switch dynamically from one exponent size to the next larger one whenever it becomes necessary; the more exponents per machine word, the faster, `char` exponents at most 127; monagan-2007 §4.2, Table 3 -- 1, 2, 4, and 8 exponents per 64-bit word, 16-bit exponents place only modest restrictions on the total degree
/// Lanes of 8 bits, total degree at most 127.
pub type Dense8 = Dense<u8>;
/// Lanes of 16 bits, total degree at most 32767.
pub type Dense16 = Dense<u16>;
/// Lanes of 32 bits, total degree at most 2^31 − 1.
pub type Dense32 = Dense<u32>;

/// The word operations, every lane at once. Each holds for lanes below
/// 2^(B−1), B = `W::BITS`, as every lane of a monomial of the format is.
impl<W: Exponent> Dense<W> {
    /// Lanes per word.
    const LANES: usize = 64 / W::BITS as usize;
    /// One lane's bits.
    const LANE: u64 = u64::MAX >> (64 - W::BITS);
    /// The low bit of every lane.
    const ONES: u64 = u64::MAX / Self::LANE;
    /// The top bit of every lane.
    const HIGH: u64 = Self::ONES << (W::BITS - 1);

    /// The top bit set in each lane where `a_i` ≥ `b_i`. Each lane computes
    /// `a_i` + 2^(B−1) − `b_i` ≥ 1, so no lane borrows from the next.
    #[inline(always)]
    fn ge(a: u64, b: u64) -> u64 {
        ((a | Self::HIGH) - b) & Self::HIGH
    }

    /// Each lane whose top bit is set in h filled with ones, the others 0.
    #[inline(always)]
    fn widen(h: u64) -> u64 {
        (h >> (W::BITS - 1)).wrapping_mul(Self::LANE)
    }

    // monomial-time.8 practice: SWAR -- lcm and the S-pair's signature take each lane's maximum or minimum from the top bits of (a | HIGH) − b, word by word, not exponent by exponent
    #[inline(always)]
    fn max(a: u64, b: u64) -> u64 {
        let m = Self::widen(Self::ge(a, b));
        (a & m) | (b & !m)
    }

    #[inline(always)]
    fn min(a: u64, b: u64) -> u64 {
        let m = Self::widen(Self::ge(a, b));
        (b & m) | (a & !m)
    }

    // monomial-time.10 practice: SWAR -- whether two monomials share a variable is read from the top bits of a + (HIGH − ONES), which mark each nonzero lane, not exponent by exponent
    /// The top bit set in each nonzero lane. A lane computes
    /// `a_i` + 2^(B−1) − 1 < 2^B, with no carry.
    #[inline(always)]
    fn nonzero(a: u64) -> u64 {
        (a + (Self::HIGH - Self::ONES)) & Self::HIGH
    }

    // monomial-time.9 practice: SWAR -- the degree of a monomial built on words is the sum of its lanes, folded in pairs to 32 bits, not lane by lane through `exponent`
    /// The sum of the lanes, for lanes below 2^B. Each fold halves the
    /// lanes and holds their sums.
    #[inline(always)]
    fn lane_sum(w: u64) -> u64 {
        let mut w = w;
        if W::BITS == 8 {
            w = (w & 0x00FF_00FF_00FF_00FF) + ((w >> 8) & 0x00FF_00FF_00FF_00FF);
        }
        if W::BITS <= 16 {
            w = (w & 0x0000_FFFF_0000_FFFF) + ((w >> 16) & 0x0000_FFFF_0000_FFFF);
        }
        (w & 0xFFFF_FFFF) + (w >> 32)
    }

    // divisibility-time.5 design -- a divisor mask of one or two bits a variable is formed a word at a time, bit 0 of each variable from the lanes that are not 0 and bit 1 from the lanes of 2 or more, read from the lanes' top bits and gathered into the low bits by one multiplication, or for 32-bit lanes by two shifts, not the mask per variable through `exponent` nor the lanes in turn (log 0043-1)
    /// The top bit of each lane of h, lane j's to bit j; h holds no other
    /// bit. For 8-bit lanes, the multiplier's bit 7k moves lane 8 − k's bit
    /// to bit 56 + (8 − k), and every other product lands on a bit of its
    /// own below 56 or past 63, so none carries. For 16-bit lanes the bits
    /// 48, 33, 18, and 3 move lane j's to bit 48 + j in the same way.
    #[inline(always)]
    fn gather(h: u64) -> u32 {
        match W::BITS {
            8 => ((h >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u32,
            16 => {
                (((h >> 15).wrapping_mul((1 << 48) | (1 << 33) | (1 << 18) | (1 << 3)) >> 48) & 0xF)
                    as u32
            }
            _ => (((h >> 31) & 1) | ((h >> 62) & 2)).narrow(),
        }
    }

    /// Bit j of x to bit 2j, for j below 8.
    #[inline(always)]
    fn spread(x: u32) -> u32 {
        let x = (x | x << 4) & 0x0F0F;
        let x = (x | x << 2) & 0x3333;
        (x | x << 1) & 0x5555
    }

    /// The degree of the monomial in these words.
    #[inline(always)]
    fn sum(m: &[u64]) -> u64 {
        m.iter().map(|&w| Self::lane_sum(w)).sum()
    }

    // rule-breaches.9 design -- the word operations on slices check on entry that each slice holds a monomial's words, in the test profile alone, not the loops that call them checking at their boundary (not measured)
    /// Whether each slice holds the words of a monomial of the format.
    fn whole(&self, slices: &[&[u64]]) -> bool {
        slices.iter().all(|m| m.len() == self.words)
    }

    /// These exponents, each below 2^(B−1), into whole words.
    fn put(exponents: &[u32], out: &mut [u64]) {
        // monomial-time.11 cited: bachmann-1998 §3, "Secondly" -- the exponent vector's length is a multiple of the word size, with (−n) mod m unnecessary exponents set to and always kept at 0
        for (o, lanes) in out.iter_mut().zip(exponents.chunks(Self::LANES)) {
            *o = lanes
                .iter()
                .enumerate()
                .fold(0, |w, (j, &e)| w | u64::from(e) << (W::BITS as usize * j));
        }
    }
}

impl<W: Exponent> Format for Dense<W> {
    type Word = u64;
    // f4-gfp.12 cited: bachmann-1998 §2.2 -- bounds on the exponents can be checked by degree bound checks, which makes exponent checks in the monomial operations unnecessary
    /// No exponent exceeds the total degree, so every exponent fits a lane,
    /// its top bit clear, where the degree does.
    const MAX_DEGREE: u32 = W::MAX;

    fn new(variables: usize) -> Dense<W> {
        let bits = if variables == 0 {
            0
        } else {
            u32::BITS / Narrow::<u32>::narrow(variables.min(u32::BITS as usize))
        };
        Dense {
            n: variables,
            words: variables.div_ceil(Self::LANES),
            bits,
            word: PhantomData,
        }
    }

    fn variables(&self) -> usize {
        self.n
    }

    fn words(&self) -> usize {
        self.words
    }

    fn encode(&self, exponents: &[u32], out: &mut [u64]) -> Result<u32, Overflow> {
        debug_assert_eq!(exponents.len(), self.n);
        debug_assert_eq!(out.len(), self.words);
        let degree: u64 = exponents.iter().map(|&e| u64::from(e)).sum();
        if degree > u64::from(Self::MAX_DEGREE) {
            return Err(Overflow);
        }
        Self::put(exponents, out);
        Ok(degree.narrow())
    }

    fn exponent(&self, m: &[u64], i: usize) -> u32 {
        debug_assert!(i < self.n, "variable {i} of {n}", n = self.n);
        debug_assert!(self.whole(&[m]), "the words of a monomial");
        let shift = W::BITS as usize * (i % Self::LANES);
        ((m[i / Self::LANES] >> shift) & Self::LANE).narrow()
    }

    // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
    /// Each whole word's L lanes from the lowest, then the last word's lanes
    /// up to the last variable. L is a constant of the format, so the loop
    /// over a word's lanes unrolls.
    #[allow(
        clippy::chunks_exact_to_as_chunks,
        reason = "`as_chunks_mut` takes no constant of the type parameter W"
    )]
    fn unpack(&self, m: &[u64], out: &mut [u32]) {
        debug_assert_eq!(m.len(), self.words);
        debug_assert_eq!(out.len(), self.n);
        let mut whole = out.chunks_exact_mut(Self::LANES);
        for (lanes, &w) in (&mut whole).zip(m) {
            for (j, e) in lanes.iter_mut().enumerate() {
                *e = ((w >> (W::BITS as usize * j)) & Self::LANE).narrow();
            }
        }
        let rest = whole.into_remainder();
        if !rest.is_empty() {
            let w = m[self.n / Self::LANES];
            for (j, e) in rest.iter_mut().enumerate() {
                *e = ((w >> (W::BITS as usize * j)) & Self::LANE).narrow();
            }
        }
    }

    // f4-gfp.13 design -- the degree check sits in every product and lcm, not in the outer loop at pair selection (not measured)
    /// Each lane of the product is at most its degree, so no lane carries.
    fn mul(
        &self,
        a: &[u64],
        da: u32,
        b: &[u64],
        db: u32,
        out: &mut [u64],
    ) -> Result<u32, Overflow> {
        debug_assert!(self.whole(&[a, b, out]), "the words of a monomial");
        let d = da + db;
        if d > Self::MAX_DEGREE {
            return Err(Overflow);
        }
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
        Ok(d)
    }

    // monomial-time.2 cited: bachmann-1998 §3, attributes it to PoSSo 1995 [14] -- the PoSSo library uses the idea to speed up monomial additions, assignments, and tests for equalities
    /// The lanes of a + b lie below 2^B, so the word sum is the lane sums
    /// and no carry makes a false equality.
    fn is_product(&self, a: &[u64], b: &[u64], c: &[u64]) -> bool {
        debug_assert!(self.whole(&[a, b, c]), "the words of a monomial");
        a.iter()
            .zip(b)
            .zip(c)
            .fold(0, |d, ((&x, &y), &z)| d | ((x + y) ^ z))
            == 0
    }

    /// Each lane of the product is at most its degree, so no lane carries
    /// into the next (bachmann-1998 §3, Lemma 1 (3)).
    #[inline(always)]
    fn mul_word(a: u64, b: u64) -> u64 {
        a + b
    }

    // monomial-time.3 cited: bachmann-1998 §3, Lemma 1 (4) and `MonDivBy` -- α divides β iff every exponent of b − a lies below 2^(s_e−1), tested on whole words with a mask of the top bits
    /// Where `b_i` ≥ `a_i` the lane of b − a keeps its top bit clear. The lowest
    /// lane with `b_i` < `a_i` borrows and sets it, whatever the borrow does to
    /// the lanes above.
    fn divides(&self, a: &[u64], b: &[u64]) -> bool {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        a.iter()
            .zip(b)
            .all(|(&x, &y)| y.wrapping_sub(x) & Self::HIGH == 0)
    }

    fn div(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> u32 {
        debug_assert!(self.whole(&[a, b, out]), "the words of a monomial");
        debug_assert!(self.divides(b, a), "the divisor divides");
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x - y;
        }
        Self::sum(out).narrow()
    }

    fn lcm(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> Result<u32, Overflow> {
        debug_assert!(self.whole(&[a, b, out]), "the words of a monomial");
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = Self::max(x, y);
        }
        let d = Self::sum(out);
        if d > u64::from(Self::MAX_DEGREE) {
            return Err(Overflow);
        }
        Ok(d.narrow())
    }

    /// At most deg a + deg b, which fits a u32.
    fn lcm_degree(&self, a: &[u64], b: &[u64]) -> u32 {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        a.iter()
            .zip(b)
            .map(|(&x, &y)| Self::lane_sum(Self::max(x, y)))
            .sum::<u64>()
            .narrow()
    }

    fn coprime(&self, a: &[u64], b: &[u64]) -> bool {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        a.iter()
            .zip(b)
            .all(|(&x, &y)| Self::nonzero(x) & Self::nonzero(y) == 0)
    }

    // signature-pair-time.1 design -- an S-pair's signature is formed in one pass over the words, s + max(0, y − x) in every lane, and its degree summed from the lanes (log 0026-1)
    /// Word by word s + y − min(x, y): the sum's lanes lie below 2^B, and
    /// the subtrahend's lane is at most y's, so neither step carries or
    /// borrows. A lane past `MAX_DEGREE` implies a degree past it, and the
    /// call returns Overflow.
    fn mul_quotient(
        &self,
        s: &[u64],
        x: &[u64],
        y: &[u64],
        out: &mut [u64],
    ) -> Result<u32, Overflow> {
        debug_assert!(self.whole(&[s, x, y, out]), "the words of a monomial");
        for (((o, &s), &x), &y) in out.iter_mut().zip(s).zip(x).zip(y) {
            *o = s + y - Self::min(x, y);
        }
        let d = Self::sum(out);
        if d > u64::from(Self::MAX_DEGREE) {
            return Err(Overflow);
        }
        Ok(d.narrow())
    }

    // monomial-time.5 cited: bachmann-1998 §3, Lemma 1 (2) and `MonComp` -- the degree compared first, then the words from the last on a little-endian machine, the exponents in their order for reverse lexicographical orders
    /// With the low lanes first, a word's most significant differing lane
    /// is its last differing variable, and the smaller word holds the
    /// smaller exponent there.
    fn cmp_revlex(&self, a: &[u64], b: &[u64]) -> Ordering {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        for (x, y) in a.iter().zip(b).rev() {
            if x != y {
                return y.cmp(x);
            }
        }
        Ordering::Equal
    }

    // monomial-time.6 design -- lex compares the first differing word at its lowest differing lane, found from the trailing zeros of a ⊕ b, not on the exponents' order reversed in the words, as one table serves grevlex and lex in one call (not measured)
    fn cmp_lex(&self, a: &[u64], b: &[u64]) -> Ordering {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        for (&x, &y) in a.iter().zip(b) {
            let d = x ^ y;
            if d != 0 {
                let shift = d.trailing_zeros() / W::BITS * W::BITS;
                return ((x >> shift) & Self::LANE).cmp(&((y >> shift) & Self::LANE));
            }
        }
        Ordering::Equal
    }

    // matrix-sort-time.4 design -- the key is built from whole exponent words, the degree and then each word complemented from the last word down, not lane by lane through `exponent` (log 0032-1)
    /// With the low lanes first, a word read as an integer orders its lanes
    /// from the last variable down, and its complement holds each lane's
    /// complement in place. The degree takes a lane's bits, as it lies
    /// below 2^(B−1).
    fn grevlex_key(&self, m: &[u64], degree: u32) -> u128 {
        debug_assert_eq!(m.len(), self.words);
        let mut key = u128::from(degree);
        let mut bits = W::BITS;
        if let Some(last) = self.words.checked_sub(1) {
            // The last word's bits that hold lanes.
            let used = Narrow::<u32>::narrow(self.n - last * Self::LANES) * W::BITS;
            for j in (0..=last).rev() {
                let width = if j == last { used } else { 64 };
                let take = width.min(KEY_BITS - bits);
                if take == 0 {
                    break;
                }
                let lanes = !m[j] & (u64::MAX >> (64 - width));
                key = (key << take) | u128::from(lanes >> (width - take));
                bits += take;
            }
        }
        key << (u128::BITS - bits)
    }

    fn pack(&self, exponents: &[u32], out: &mut [u64]) {
        debug_assert_eq!(exponents.len(), self.n);
        debug_assert!(exponents.iter().all(|&e| e <= Self::MAX_DEGREE));
        debug_assert_eq!(out.len(), self.words);
        Self::put(exponents, out);
    }

    // f4-gfp.18 cited: demin-2024 §4.1, the figure on division masks -- the mask writes each exponent in a fixed number of bits, x²yz³ as 00011 00001 00111
    // many-variables-time.1 cited: demin-2024 §4.1 -- when the number of variables is large, the division mask is compressed, so that a single bit in the mask may correspond to several variables
    /// Up to 32 variables each has `bits` bits, and bit k of variable i is
    /// set where its exponent is at least k + 1. Past 32 variable i sets
    /// bit i mod 32 where its exponent is not 0. The lanes past the last
    /// variable hold 0 (monomial-time.11), and their bits stay 0.
    fn mask(&self, m: &[u64]) -> u32 {
        debug_assert_eq!(m.len(), self.words);
        let mut mask = 0u32;
        let mut pos = 0u32;
        if self.bits <= 2 {
            // divisibility-time.5: a word at a time.
            let step = self.bits * Narrow::<u32>::narrow(Self::LANES);
            for &x in m {
                let ones = Self::gather(Self::nonzero(x));
                let bits = if self.bits == 1 {
                    ones
                } else {
                    let twos = Self::gather(Self::ge(x, 2 * Self::ONES));
                    Self::spread(ones) | Self::spread(twos) << 1
                };
                // many-variables-time.2 design -- past 32 variables variable i takes bit i mod 32, so each word's lanes land unsplit on consecutive bits, not runs of ⌈n/32⌉ consecutive variables to a bit (not measured)
                mask |= bits << pos;
                pos = (pos + step) % u32::BITS;
            }
            return mask;
        }
        let mut left = self.n;
        // divisibility-time.6 design -- a divisor mask of three bits a variable or more is formed lane by lane from each word, not per variable through `exponent` (log 0043-1)
        for &x in m {
            if left == 0 {
                break;
            }
            let take = left.min(Self::LANES);
            for j in 0..take {
                let e: u32 = ((x >> (W::BITS as usize * j)) & Self::LANE).narrow();
                let set = e.min(self.bits);
                mask |= Narrow::<u32>::narrow((1u64 << set) - 1) << pos;
                pos += self.bits;
            }
            left -= take;
        }
        mask
    }
}
