//! The monomial contract: a format stores a monomial's exponents in a fixed
//! number of words, and every operation that can leave the format says so.
//! The formats are dense exponent vectors in lanes of 4, 8, 16, and 32
//! bits, packed into 64-bit words, `Dense4`, `Dense8`, `Dense16`, and
//! `Dense32`. A call computes in the narrowest that holds its input from
//! its field's narrowest on, `Dense4` over GF(2) and `Dense8` over every
//! other field, a replay from a kept shape in the shape's, and starts over
//! in the next wider where a degree passes the one it runs in.

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

// gf2-update-rest-time.2 design-checked -- a support holds 128 bits, variable v at bit v mod 128, exact to 128 variables and folded past them, not 64 bits, which fold sat's 75 variables, nor a bit per variable at every count, whose index grows by a set per variable past them (log 0137-1, 0137-2)
/// The bits of a support, `Format::support`.
pub(crate) const SUPPORT_BITS: usize = u128::BITS as usize;

/// A monomial format over a fixed number of variables. A monomial is a slice
/// of `words()` words; its total degree, the sum of its exponents, travels
/// beside it. A word is `Word::default()` exactly where every exponent it
/// holds is 0, and a word of a·b is `mul_word` of a's and b's words there,
/// which the table's support check reads (wide-format-products-time.3).
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
    /// As `encode`, for exponents that are 0 but at `touched`, which names
    /// each variable once: the words cleared, and the exponents of `touched`
    /// written to them.
    ///
    /// # Errors
    ///
    /// `Overflow` where the degree passes `MAX_DEGREE`.
    fn encode_touched(
        &self,
        exponents: &[u32],
        touched: &[usize],
        out: &mut [Self::Word],
    ) -> Result<u32, Overflow>;
    fn exponent(&self, m: &[Self::Word], i: usize) -> u32;
    /// Sets the exponent of variable i in m to 0: the exponent it held. The
    /// degree that travels beside m falls by it.
    fn clear(&self, m: &mut [Self::Word], i: usize) -> u32;
    // result-time.2 design-checked -- `unpack` is a required method of the trait, not a default that calls `exponent` per variable (not measured)
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
    /// The word of a·b at a place where a holds `a` and b holds `b`, for two
    /// monomials of the format, whose product's degree may pass
    /// `MAX_DEGREE` and is at most twice it.
    fn mul_word(a: Self::Word, b: Self::Word) -> Self::Word;
    /// Whether a divides b.
    fn divides(&self, a: &[Self::Word], b: &[Self::Word]) -> bool {
        self.divides_n::<ANY>(a, b)
    }
    /// `divides` for a format of N words (`ANY`).
    fn divides_n<const N: usize>(&self, a: &[Self::Word], b: &[Self::Word]) -> bool;
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
    ) -> Result<u32, Overflow> {
        self.lcm_n::<ANY>(a, b, out)
    }
    /// `lcm` for a format of N words (`ANY`).
    ///
    /// # Errors
    ///
    /// `Overflow` where the degree passes `MAX_DEGREE`.
    fn lcm_n<const N: usize>(
        &self,
        a: &[Self::Word],
        b: &[Self::Word],
        out: &mut [Self::Word],
    ) -> Result<u32, Overflow> {
        let degree = self.lcm_words_n::<N>(a, b, out);
        if degree > Self::MAX_DEGREE {
            return Err(Overflow);
        }
        Ok(degree)
    }
    /// lcm(a, b) to `out`, for a format of N words (`ANY`): its degree,
    /// which may pass `MAX_DEGREE` and is at most twice it. Each exponent
    /// of the lcm is one of a's or b's, so the format's words hold the lcm
    /// whatever its degree; no table interns one past `MAX_DEGREE`, and the
    /// plain run's selection overflows at a pair whose lcm passes it
    /// (gf2-monomial-time.3).
    fn lcm_words_n<const N: usize>(
        &self,
        a: &[Self::Word],
        b: &[Self::Word],
        out: &mut [Self::Word],
    ) -> u32;
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
    fn lcm_degree(&self, a: &[Self::Word], b: &[Self::Word]) -> u32 {
        self.lcm_degree_n::<ANY>(a, b)
    }
    /// `lcm_degree` for a format of N words (`ANY`).
    fn lcm_degree_n<const N: usize>(&self, a: &[Self::Word], b: &[Self::Word]) -> u32;
    /// Whether a and b share no variable.
    fn coprime(&self, a: &[Self::Word], b: &[Self::Word]) -> bool {
        self.coprime_n::<ANY>(a, b)
    }
    /// `coprime` for a format of N words (`ANY`).
    fn coprime_n<const N: usize>(&self, a: &[Self::Word], b: &[Self::Word]) -> bool;
    /// a against b in the reverse lexicographic order: a > b where the last
    /// variable whose exponents differ has the smaller exponent in a.
    fn cmp_revlex(&self, a: &[Self::Word], b: &[Self::Word]) -> Ordering;
    /// a against b in the lexicographic order: a > b where the first variable
    /// whose exponents differ has the larger exponent in a.
    fn cmp_lex(&self, a: &[Self::Word], b: &[Self::Word]) -> Ordering;
    // matrix-sort-time.3 cited-checked: monagan-2007 §2.1, Figure 1 -- grevlex is the degree, then the exponents from the last variable to the first with their comparisons inverted; after Figure 4 -- the first n exponents decide a grevlex comparison on n variables; bachmann-1998 §4 (ii) -- a rank held in a machine integer limits the largest representable monomial, and a 32-bit rank restricts its degree
    /// The first 96 bits of the grevlex key of m, of degree `degree`, in the
    /// top 96 bits, the low 32 bits 0. The key is the degree, then the
    /// exponents from the last variable down to the first, each complemented
    /// in its lane's bits, packed from the most significant bit. Of two
    /// monomials, the one with the smaller key is the smaller in grevlex. Two
    /// equal keys are one monomial where the key fits 96 bits but for the
    /// first variable's exponent.
    fn grevlex_key(&self, m: &[Self::Word], degree: u32) -> u128;
    /// The tie key of m over its first `below` variables, or over the
    /// variables the grevlex key does not hold, the first ones, where
    /// `below` is None: it orders the monomials of one degree that agree in
    /// every other variable, in the top 96 bits, the low 32 bits 0. Of two
    /// such monomials, the one with the smaller tie key is the smaller in
    /// grevlex, or the two tie keys agree. The tie key reads the last word
    /// with a nonzero lane among the variables: the complement of its index
    /// plus one in 16 bits, all ones where there is none, then the
    /// complement of its lanes among them, and then the top 16 bits of the
    /// complement of the word below it, ones where there is none
    /// (sort-share-time.8).
    fn grevlex_tie_key(&self, m: &[Self::Word], below: Option<usize>) -> u128;
    /// Where the tie keys `key` of two monomials agree, the variables a
    /// further tie key orders them by: those of the words below the one
    /// `key` read. None where `key` read no word, or the first.
    fn tie_key_next(&self, key: u128) -> Option<usize>;
    /// Writes these exponents, one per variable, each at most `MAX_DEGREE`, to
    /// `out`, whatever their sum: a bound for `divides`, or exponents of no
    /// table that an order compares; not a monomial of the table.
    fn pack(&self, exponents: &[u32], out: &mut [Self::Word]);
    // many-variables-time.3 cited-checked:roune-2012 §C.3.2 -- 64-bit divmasks have been slower in their experiments, the increased hit rate not making up for the extra overhead of dealing with more bits
    /// The divisor mask: a | b implies mask(a) & !mask(b) == 0. A variable
    /// of a sets a bit of mask(a), so masks that share no bit belong to
    /// monomials that share no variable. Each bit is set where an exponent
    /// reaches a threshold, and an lcm's exponent is the larger of its
    /// factors', so mask(lcm(a, b)) = mask(a) | mask(b). Every threshold is
    /// at least 1, so the constant's mask alone is 0. Up to 32 variables
    /// each bit stands for one variable's exponent reaching a threshold of
    /// its own, so the bits of mask(a) outside mask(b) that stand for a
    /// variable number at most its exponent in lcm(a, b)/b, which the
    /// criteria M and F read the colons of a new element's pairs by
    /// (gf2-chain-criteria-time.5).
    fn mask(&self, m: &[Self::Word]) -> u32;
    /// The support of m: bit v mod `SUPPORT_BITS` set for each variable v
    /// whose exponent is not 0. a | b implies support(a) & !support(b) ==
    /// 0, and support(lcm(a, b)) = support(a) | support(b). To
    /// `SUPPORT_BITS` variables a bit is one variable, and two monomials
    /// share a variable exactly where their supports share a bit.
    fn support(&self, m: &[Self::Word]) -> u128;
    /// The variables of m whose exponent is not 0, in increasing order,
    /// written to `out`, which the call clears first.
    fn held(&self, m: &[Self::Word], out: &mut Vec<u32>);
    /// The total degree of m, the sum of its exponents.
    fn degree(&self, m: &[Self::Word]) -> u32;
    /// The index of the word that holds variable v's exponent, and that
    /// word of the monomial v: its exponent of v 1, every other 0.
    fn unit(&self, v: usize) -> (usize, Self::Word);
    /// The words of `out` with every bit of these variables' lanes set and
    /// no other bit: a mask that `and_word` reads them by.
    fn lanes(&self, variables: &[usize], out: &mut [Self::Word]);
    /// The bits both words hold.
    fn and_word(a: Self::Word, b: Self::Word) -> Self::Word;
    /// For `word`, the word at index j of a monomial or of a mask by
    /// `lanes`, each variable whose lane there is not 0, with that lane's
    /// value, by decreasing variable.
    fn each_variable(&self, j: usize, word: Self::Word, f: impl FnMut(usize, u32));
}

/// The lane of a dense format: `U4`, `u8`, `u16`, or `u32`, BITS bits wide.
pub trait Exponent: Copy + Eq + Ord + Default + Debug + Send + Sync + 'static {
    /// The bits of a lane.
    const BITS: u32;
    /// The largest total degree a dense format of these lanes holds,
    /// 2^(BITS−1) − 1: 7, 127, 32767, and 2^31 − 1.
    const MAX: u32;
}

/// The lane of 4 bits, which no integer type of Rust is: it names the
/// width of `Dense4`'s lanes, which its words hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct U4;

impl Exponent for U4 {
    const BITS: u32 = 4;
    const MAX: u32 = (1 << 3) - 1;
}

// monomial-time.4 practice-checked: guard bit -- `Dense8`'s degree bound is 127 and `Dense16`'s 32767, so every exponent leaves its lane's top bit clear, not 255 and 65535 with a divisibility test that combines the borrows of each lane's low bits with its top bits
impl Exponent for u8 {
    const BITS: u32 = 8;
    const MAX: u32 = (1 << 7) - 1;
}

impl Exponent for u16 {
    const BITS: u32 = 16;
    const MAX: u32 = (1 << 15) - 1;
}

// monomial-formats.3 design-checked -- the widest format's lanes are 32 bits, the width of a degree throughout the engine, not 64-bit lanes (not measured)
impl Exponent for u32 {
    const BITS: u32 = 32;
    // monomial-formats.4 design-checked -- the widest format's degree bound is 2^31 − 1, so that a sum of two degrees or two exponents fits the u32 the engine computes it in, not u32::MAX with the engine's degree sums taken in u64 (not measured)
    const MAX: u32 = i32::MAX as u32;
}

// f4-gfp.11 design-checked -- a monomial is a dense vector of fixed-width exponents, one for every variable, with its total degree beside it, not Groebner.jl's packed exponents, which hold at most 31 variables and degree 127 (log 0013-1)
// monomial-time.1 cited-checked:bachmann-1998 §3, attributes parts of it to PoSSo 1995 [14] -- monomial operations are performed on machine words instead of directly on exponents, which sped up the computations by 1.1 to 1.7 with exponents of 16 bits and by 1.2 to 2.0 with exponents of 8 bits (Table 1)
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

// monomial-formats.1 cited-checked:bachmann-1998 §3 -- ideally switch dynamically from one exponent size to the next larger one whenever it becomes necessary; the more exponents per machine word, the faster, `char` exponents at most 127; monagan-2007 §4.2, Table 3 -- 1, 2, 4, and 8 exponents per 64-bit word, 16-bit exponents place only modest restrictions on the total degree
// gf2-monomial-time.1 cited-checked: bachmann-1998 §3 -- the more exponents are encoded into one machine word, the faster the computation, as one operation handles more exponents and less memory is used, and the smaller the bound on a single exponent; monagan-2014 §2 -- b = ⌊64/(n+1)⌋ bits for each exponent and for the total degree, as few as 2, and a product computed with no overflow detection where its degree lies below 2^b
/// Lanes of 4 bits, total degree at most 7.
pub type Dense4 = Dense<U4>;
/// Lanes of 8 bits, total degree at most 127.
pub type Dense8 = Dense<u8>;
/// Lanes of 16 bits, total degree at most 32767.
pub type Dense16 = Dense<u16>;
/// Lanes of 32 bits, total degree at most 2^31 − 1.
pub type Dense32 = Dense<u32>;

/// A format of the family by the width of its lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Width {
    W4,
    W8,
    W16,
    W32,
}

/// The widths a call takes, from the narrowest to the widest: each run
/// starts in the narrowest its field takes (`entry::narrowest`).
pub(crate) const WIDTHS: (Width, Width) = (Width::W4, Width::W32);

impl Width {
    /// The width of a lane in bits.
    pub(crate) fn bits(self) -> u32 {
        match self {
            Width::W4 => 4,
            Width::W8 => 8,
            Width::W16 => 16,
            Width::W32 => 32,
        }
    }

    /// The next wider width, None for W32.
    pub(crate) fn wider(self) -> Option<Width> {
        match self {
            Width::W4 => Some(Width::W8),
            Width::W8 => Some(Width::W16),
            Width::W16 => Some(Width::W32),
            Width::W32 => None,
        }
    }
}

/// The N of a word operation's copy for any word count: its loop runs
/// over the slices' length. A copy for N from 1 to `UNROLLED` serves a
/// format of N words alone, and its loop has that constant trip count
/// (`pairs::per_words`, pair-update-rest-time.3).
pub(crate) const ANY: usize = 0;

/// The largest word count a word operation has a copy of its own for.
pub(crate) const UNROLLED: usize = 8;

/// The word operations, every lane at once. Each holds for lanes below
/// 2^(B−1), B = `W::BITS`, as every lane of a monomial of the format is.
impl<W: Exponent> Dense<W> {
    /// Lanes per word.
    const LANES: usize = 64 / W::BITS as usize;
    /// The variables `grevlex_key` holds after the degree's lane: the last
    /// ones.
    const KEY_LANES: usize = (KEY_BITS / W::BITS) as usize - 1;
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

    // monomial-time.8 practice-checked:SWAR -- lcm and the S-pair's signature take each lane's maximum or minimum from the top bits of (a | HIGH) − b, word by word, not exponent by exponent
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

    // monomial-time.10 practice-checked:SWAR -- whether two monomials share a variable is read from the top bits of a + (HIGH − ONES), which mark each nonzero lane, not exponent by exponent
    /// The top bit set in each nonzero lane. A lane computes
    /// `a_i` + 2^(B−1) − 1 < 2^B, with no carry.
    #[inline(always)]
    fn nonzero(a: u64) -> u64 {
        (a + (Self::HIGH - Self::ONES)) & Self::HIGH
    }

    // monomial-time.9 practice-checked:SWAR -- the degree of a monomial built on words is the sum of its lanes, taken a word at a time, not lane by lane through `exponent`
    // pair-update-rest-time.2 practice-checked: SWAR lane sum by one multiplication (popcount's last step) -- the lanes of 8 and 16 bits are summed by one multiplication by ONES, the sum read off the word's top lane, not folded in pairs; 32-bit lanes keep the one fold
    // gf2-monomial-time.4 design-checked -- the lanes of 4 bits are summed by the same multiplication, a word's sum at most 14 below 2^4, not folded into bytes before it (log 0136-1, 0136-2)
    /// The sum of the lanes, which lies below 2^B: a monomial of the format
    /// has a degree at most `MAX_DEGREE`, 2^(B−1) − 1, and an lcm, a
    /// quotient, and `mul_quotient`'s result at most twice it. The word
    /// times `ONES` holds in its top lane the sum of every lane, with no
    /// carry between the lanes while every partial sum lies below 2^B. The
    /// test profile checks the sum against the fold. Two 32-bit lanes are
    /// summed by one fold, since their sum may pass 2^32.
    #[inline(always)]
    fn lane_sum(w: u64) -> u64 {
        if W::BITS == u32::BITS {
            return (w & 0xFFFF_FFFF) + (w >> 32);
        }
        let sum = w.wrapping_mul(Self::ONES) >> (u64::BITS - W::BITS);
        debug_assert_eq!(sum, Self::lane_fold(w), "the lanes' sum lies below 2^B");
        sum
    }

    /// The sum of the lanes by folds, each of which halves the lanes and
    /// holds their sums: the test profile's check of `lane_sum`.
    fn lane_fold(w: u64) -> u64 {
        let mut w = w;
        if W::BITS == 4 {
            w = (w & 0x0F0F_0F0F_0F0F_0F0F) + ((w >> 4) & 0x0F0F_0F0F_0F0F_0F0F);
        }
        if W::BITS <= 8 {
            w = (w & 0x00FF_00FF_00FF_00FF) + ((w >> 8) & 0x00FF_00FF_00FF_00FF);
        }
        if W::BITS <= 16 {
            w = (w & 0x0000_FFFF_0000_FFFF) + ((w >> 16) & 0x0000_FFFF_0000_FFFF);
        }
        (w & 0xFFFF_FFFF) + (w >> 32)
    }

    // divisibility-time.5 practice-checked: SWAR -- a divisor mask of one or two bits a variable is formed a word at a time, bit 0 of each variable from the lanes that are not 0 and bit 1 from the lanes of 2 or more, read from the lanes' top bits and gathered into the low bits by one multiplication, or for 32-bit lanes by two shifts, not the mask per variable through `exponent` nor the lanes in turn (log 0043-1)
    // gf2-monomial-time.5 design-checked -- the top bits of 4-bit lanes are gathered by four folds, each a shift, an OR, and a mask that halve the distance between the bits, not by one multiplication, under which the 16 bits 4 apart, moved to consecutive bits in their order, have products that land on another lane's place, lane 3's shifted by lane 4's on lane 0's (log 0136-2)
    /// The top bit of each lane of h, lane j's to bit j; h holds no other
    /// bit. For 8-bit lanes, the multiplier's bit 7k moves lane 8 − k's bit
    /// to bit 56 + (8 − k), and every other product lands on a bit of its
    /// own below 56 or past 63, so none carries. For 16-bit lanes the bits
    /// 48, 33, 18, and 3 move lane j's to bit 48 + j in the same way. For
    /// 4-bit lanes each fold moves every other bit's group down by the
    /// distance between groups less their width: the bits 4 apart become
    /// pairs 8 apart, then fours 16 apart, eights 32 apart, and the 16.
    #[inline(always)]
    fn gather(h: u64) -> u32 {
        match W::BITS {
            4 => {
                let x = (h >> 3) & 0x1111_1111_1111_1111;
                let x = (x | x >> 3) & 0x0303_0303_0303_0303;
                let x = (x | x >> 6) & 0x000F_000F_000F_000F;
                let x = (x | x >> 12) & 0x0000_00FF_0000_00FF;
                ((x | x >> 24) & 0xFFFF).narrow()
            }
            8 => ((h >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u32,
            16 => {
                (((h >> 15).wrapping_mul((1 << 48) | (1 << 33) | (1 << 18) | (1 << 3)) >> 48) & 0xF)
                    as u32
            }
            _ => (((h >> 31) & 1) | ((h >> 62) & 2)).narrow(),
        }
    }

    /// Bit j of x to bit 2j, for j below the lanes of a word: 16 for 4-bit
    /// lanes, else at most 8.
    #[inline(always)]
    fn spread(x: u32) -> u32 {
        let x = if W::BITS == 4 {
            (x | x << 8) & 0x00FF_00FF
        } else {
            x
        };
        let x = (x | x << 4) & 0x0F0F_0F0F;
        let x = (x | x << 2) & 0x3333_3333;
        (x | x << 1) & 0x5555_5555
    }

    /// The degree of the monomial in these words.
    #[inline(always)]
    fn sum(m: &[u64]) -> u64 {
        m.iter().map(|&w| Self::lane_sum(w)).sum()
    }

    // rule-breaches.9 design-checked -- the word operations on slices check on entry that each slice holds a monomial's words, in the test profile alone, not the loops that call them checking at their boundary (not measured)
    /// Whether each slice holds the words of a monomial of the format.
    fn whole(&self, slices: &[&[u64]]) -> bool {
        slices.iter().all(|m| m.len() == self.words)
    }

    /// These exponents, each below 2^(B−1), into whole words.
    fn put(exponents: &[u32], out: &mut [u64]) {
        // monomial-time.11 design-checked -- the lanes past the last variable hold 0, so every monomial fills whole words, not 8-byte reads over the unpadded exponents with the bytes past the monomial masked off (log 0029-3)
        for (o, lanes) in out.iter_mut().zip(exponents.chunks(Self::LANES)) {
            *o = lanes
                .iter()
                .enumerate()
                .fold(0, |w, (j, &e)| w | u64::from(e) << (W::BITS as usize * j));
        }
    }

    /// The first N words of `m`, a slice whose length the compiler knows;
    /// every word where N is `ANY`. A format of N words holds N words in
    /// every monomial (`whole`).
    #[inline(always)]
    fn first<const N: usize>(m: &[u64]) -> &[u64] {
        if N == ANY { m } else { &m[..N] }
    }

    /// As `first`, for the words written.
    #[inline(always)]
    fn first_mut<const N: usize>(m: &mut [u64]) -> &mut [u64] {
        if N == ANY { m } else { &mut m[..N] }
    }
}

impl<W: Exponent> Format for Dense<W> {
    type Word = u64;
    // f4-gfp.12 cited-checked:bachmann-1998 §2.2 -- bounds on the exponents can be checked by degree bound checks, which makes exponent checks in the monomial operations unnecessary
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

    // instance-read-peak.4 design-checked -- a monomial whose exponents are 0 but at the variables a term names is encoded into words cleared first, each named exponent written into its lane, not every variable's exponent packed into the words, which took most of an interning on a text of hundreds of variables (log 0175-1, 0175-2)
    fn encode_touched(
        &self,
        exponents: &[u32],
        touched: &[usize],
        out: &mut [u64],
    ) -> Result<u32, Overflow> {
        debug_assert_eq!(exponents.len(), self.n);
        debug_assert_eq!(out.len(), self.words);
        let degree: u64 = touched.iter().map(|&i| u64::from(exponents[i])).sum();
        debug_assert_eq!(
            exponents.iter().map(|&e| u64::from(e)).sum::<u64>(),
            degree,
            "every nonzero exponent at a variable of touched, each once"
        );
        if degree > u64::from(Self::MAX_DEGREE) {
            return Err(Overflow);
        }
        out.fill(0);
        for &i in touched {
            // No exponent exceeds the degree, so each fits its lane.
            let (k, unit) = self.unit(i);
            out[k] |= unit * u64::from(exponents[i]);
        }
        Ok(degree.narrow())
    }

    fn exponent(&self, m: &[u64], i: usize) -> u32 {
        debug_assert!(i < self.n, "variable {i} of {n}", n = self.n);
        debug_assert!(self.whole(&[m]), "the words of a monomial");
        let shift = W::BITS as usize * (i % Self::LANES);
        ((m[i / Self::LANES] >> shift) & Self::LANE).narrow()
    }

    fn clear(&self, m: &mut [u64], i: usize) -> u32 {
        debug_assert!(i < self.n, "variable {i} of {n}", n = self.n);
        debug_assert!(self.whole(&[&*m]), "the words of a monomial");
        let shift = W::BITS as usize * (i % Self::LANES);
        let word = &mut m[i / Self::LANES];
        let e = ((*word >> shift) & Self::LANE).narrow();
        *word &= !(Self::LANE << shift);
        e
    }

    // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
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

    // f4-gfp.13 design-checked -- the degree check sits in every product and lcm, not in the outer loop at pair selection (not measured)
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

    // monomial-time.2 cited-checked:bachmann-1998 §3, attributes it to PoSSo 1995 [14] -- the PoSSo library uses the idea to speed up monomial additions, assignments, and tests for equalities
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

    /// Each exponent of a and b lies below 2^(B−1), so each lane of the
    /// product is their sum, below 2^B, and no lane carries into the next
    /// (bachmann-1998 §3, Lemma 1 (3)).
    #[inline(always)]
    fn mul_word(a: u64, b: u64) -> u64 {
        a + b
    }

    // monomial-time.3 cited-checked:bachmann-1998 §3, Lemma 1 (4) and `MonDivBy` -- α divides β iff every exponent of b − a lies below 2^(s_e−1), tested on whole words with a mask of the top bits
    // pair-update-rest-time.6 practice-checked: branch-free code -- the copies of `divides` and `coprime` for one to eight words OR their words' tests into one word and test it once, not a test per word with an early exit, which the copy for any count keeps
    /// Where `b_i` ≥ `a_i` the lane of b − a keeps its top bit clear. The
    /// lowest lane with `b_i` < `a_i` borrows and sets it, whatever the
    /// borrow does to the lanes above. The copy for N words joins the
    /// words' borrows by OR and tests the join once; the copy for any count
    /// leaves at the first word that fails.
    fn divides_n<const N: usize>(&self, a: &[u64], b: &[u64]) -> bool {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        let (a, b) = (Self::first::<N>(a), Self::first::<N>(b));
        let borrowed = |(&x, &y): (&u64, &u64)| y.wrapping_sub(x) & Self::HIGH;
        if N == ANY {
            return a.iter().zip(b).all(|w| borrowed(w) == 0);
        }
        a.iter().zip(b).fold(0, |join, w| join | borrowed(w)) == 0
    }

    fn div(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> u32 {
        debug_assert!(self.whole(&[a, b, out]), "the words of a monomial");
        debug_assert!(self.divides(b, a), "the divisor divides");
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x - y;
        }
        Self::sum(out).narrow()
    }

    // gf2-monomial-time.7 cited-checked: martani-2012 §Inligning; bachmann-1998 §3, "Fourthly", independently -- the inline keyword is only a hint the compiler may decline, an attribute makes the inlining sure, and an operation inlined in place of a function call each gave a considerable efficiency gain
    /// The lcm's lanes are summed in the loop that forms them. Its degree,
    /// at most twice `MAX_DEGREE`, lies below 2^32.
    #[inline(always)]
    fn lcm_words_n<const N: usize>(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> u32 {
        debug_assert!(self.whole(&[a, b, out]), "the words of a monomial");
        let (a, b, out) = (
            Self::first::<N>(a),
            Self::first::<N>(b),
            Self::first_mut::<N>(out),
        );
        let mut d = 0;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = Self::max(x, y);
            d += Self::lane_sum(*o);
        }
        d.narrow()
    }

    /// At most deg a + deg b, which fits a u32.
    fn lcm_degree_n<const N: usize>(&self, a: &[u64], b: &[u64]) -> u32 {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        let (a, b) = (Self::first::<N>(a), Self::first::<N>(b));
        a.iter()
            .zip(b)
            .map(|(&x, &y)| Self::lane_sum(Self::max(x, y)))
            .sum::<u64>()
            .narrow()
    }

    /// The copy for N words joins the words' shared lanes by OR and tests
    /// the join once (pair-update-rest-time.6); the copy for any count
    /// leaves at the first word that shares a lane.
    fn coprime_n<const N: usize>(&self, a: &[u64], b: &[u64]) -> bool {
        debug_assert!(self.whole(&[a, b]), "the words of a monomial");
        let (a, b) = (Self::first::<N>(a), Self::first::<N>(b));
        let shared = |(&x, &y): (&u64, &u64)| Self::nonzero(x) & Self::nonzero(y);
        if N == ANY {
            return a.iter().zip(b).all(|w| shared(w) == 0);
        }
        a.iter().zip(b).fold(0, |join, w| join | shared(w)) == 0
    }

    // signature-pair-time.1 design-checked -- an S-pair's signature is formed in one pass over the words, s + max(0, y − x) in every lane, and its degree summed from the lanes (log 0026-1)
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

    // monomial-time.5 cited-checked:bachmann-1998 §3, Lemma 1 (2) and `MonComp` -- the degree compared first, then the words from the last on a little-endian machine, the exponents in their order for reverse lexicographical orders
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

    // monomial-time.6 design-checked -- lex compares the first differing word at its lowest differing lane, found from the trailing zeros of a ⊕ b, not on the exponents' order reversed in the words, as one table serves grevlex and lex in one call (not measured)
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

    // matrix-sort-time.4 practice-checked: SWAR -- the key is built from whole exponent words, the degree and then each word complemented from the last word down, not lane by lane through `exponent` (log 0032-1)
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

    /// Within a run of one degree whose grevlex keys agree, the order is
    /// the reverse lexicographic one on the first `n − KEY_LANES` variables,
    /// and within a run whose tie keys agree too, on the words below the
    /// one the tie key read. Of two monomials whose last nonzero words
    /// among them differ, the one with the higher word has the larger
    /// exponent at the last variable where they differ, and is the smaller:
    /// the complement of its index is the smaller. At one word, the word
    /// read as an integer orders its lanes from the last variable down, and
    /// its complement orders the monomials; so does the next word's top,
    /// where the words agree. Past `u16::MAX` words the index would not
    /// fit, and every tie key agrees.
    fn grevlex_tie_key(&self, m: &[u64], below: Option<usize>) -> u128 {
        const NONE: u128 = u128::MAX << PAYLOAD_BITS;
        debug_assert_eq!(m.len(), self.words);
        debug_assert!(below.is_none_or(|b| b <= self.n), "variables of the format");
        if self.words == 0 || self.words > usize::from(u16::MAX) {
            return NONE;
        }
        // The variables the key reads, the word that holds the boundary,
        // and that word's bits below it: the last word, whole, where the
        // boundary lies past it.
        let below = below.unwrap_or_else(|| self.n.saturating_sub(Self::KEY_LANES));
        let (top, mask_top) = if below / Self::LANES < self.words {
            let partial = Narrow::<u32>::narrow(below % Self::LANES) * W::BITS;
            (below / Self::LANES, (1u64 << partial).wrapping_sub(1))
        } else {
            (self.words - 1, u64::MAX)
        };
        let mut j = top;
        loop {
            let lanes = m[j] & if j == top { mask_top } else { u64::MAX };
            if lanes != 0 {
                let index = !Narrow::<u16>::narrow(j + 1);
                let next = if j > 0 {
                    !m[j - 1] >> 48
                } else {
                    u64::MAX >> 48
                };
                return (u128::from(index) << (u128::BITS - 16))
                    | (u128::from(!lanes) << (PAYLOAD_BITS + 16))
                    | (u128::from(next) << PAYLOAD_BITS);
            }
            if j == 0 {
                return NONE;
            }
            j -= 1;
        }
    }

    /// The word the key read, from the complement of its index plus one in
    /// the key's top 16 bits: the variables of the words below it.
    fn tie_key_next(&self, key: u128) -> Option<usize> {
        let read = !Narrow::<u16>::narrow(key >> (u128::BITS - 16));
        match read {
            0 | 1 => None,
            _ => Some((usize::from(read) - 1) * Self::LANES),
        }
    }

    fn pack(&self, exponents: &[u32], out: &mut [u64]) {
        debug_assert_eq!(exponents.len(), self.n);
        debug_assert!(exponents.iter().all(|&e| e <= Self::MAX_DEGREE));
        debug_assert_eq!(out.len(), self.words);
        Self::put(exponents, out);
    }

    // f4-gfp.18 design-checked -- up to 32 variables the mask gives each variable ⌊32/n⌋ bits, bit k of variable i set where its exponent is at least k + 1, fixed thresholds that keep a stored mask valid for the whole call, not thresholds chosen from the monomials stored and recalculated from time to time (not measured)
    // many-variables-time.1 design-checked -- past 32 variables every variable reaches the mask, a bit standing for several variables, not the first 32 variables alone, as msolve and roune-2012 form it (log 0070-2)
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
            // bivium-signature-gap.2 design-checked -- the gathers of a monomial's lanes, its mask's, its support's, and its variables', pass over a word that holds no variable, not every word gathered, so that a monomial of a few of hundreds of variables costs its words that hold one (log 0183-1)
            // bivium-signature-gap.4 design-checked -- the passes over a word of no variable hold in every format and at every count of variables, not past 32 variables or past COPIED_WORDS words alone: on few variables a word holds one as a rule and the pass costs a test a word (log 0183-1)
            for &x in m {
                if x == 0 {
                    pos = (pos + step) % u32::BITS;
                    continue;
                }
                let ones = Self::gather(Self::nonzero(x));
                let bits = if self.bits == 1 {
                    ones
                } else {
                    let twos = Self::gather(Self::ge(x, 2 * Self::ONES));
                    Self::spread(ones) | Self::spread(twos) << 1
                };
                // many-variables-time.2 design-checked -- past 32 variables variable i takes bit i mod 32, so each word's lanes land unsplit on consecutive bits, not runs of ⌈n/32⌉ consecutive variables to a bit (not measured)
                mask |= bits << pos;
                pos = (pos + step) % u32::BITS;
            }
            return mask;
        }
        let mut left = self.n;
        // divisibility-time.6 design-checked -- a divisor mask of three bits a variable or more is formed lane by lane from each word, not per variable through `exponent` (log 0043-1)
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

    /// A word's lanes land on consecutive bits, from the word's first
    /// variable's bit on: the lanes per word divide `SUPPORT_BITS`, so no
    /// word's lanes are split. The lanes past the last variable hold 0, and
    /// their bits stay 0.
    fn support(&self, m: &[u64]) -> u128 {
        debug_assert_eq!(m.len(), self.words);
        let mut support = 0u128;
        let mut pos = 0;
        for &x in m {
            // bivium-signature-gap.2: a word of no variable sets no bit.
            if x != 0 {
                support |= u128::from(Self::gather(Self::nonzero(x))) << pos;
            }
            pos = (pos + Self::LANES) % SUPPORT_BITS;
        }
        support
    }

    /// A word's lanes that are not 0, from the word's first variable on: the
    /// lanes past the last variable hold 0 and name none.
    fn held(&self, m: &[u64], out: &mut Vec<u32>) {
        debug_assert_eq!(m.len(), self.words);
        out.clear();
        for (k, &x) in m.iter().enumerate() {
            // bivium-signature-gap.2: a word of no variable holds none.
            if x == 0 {
                continue;
            }
            let first: u32 = (k * Self::LANES).narrow();
            let mut lanes = Self::gather(Self::nonzero(x));
            while lanes != 0 {
                out.push(first + lanes.trailing_zeros());
                lanes &= lanes - 1;
            }
        }
    }

    fn degree(&self, m: &[u64]) -> u32 {
        debug_assert!(self.whole(&[m]), "the words of a monomial");
        Self::sum(m).narrow()
    }

    fn unit(&self, v: usize) -> (usize, u64) {
        debug_assert!(v < self.n, "variable {v} of {n}", n = self.n);
        (v / Self::LANES, 1 << (W::BITS as usize * (v % Self::LANES)))
    }

    fn lanes(&self, variables: &[usize], out: &mut [u64]) {
        debug_assert_eq!(out.len(), self.words);
        out.fill(0);
        for &v in variables {
            debug_assert!(v < self.n, "variable {v} of {n}", n = self.n);
            out[v / Self::LANES] |= Self::LANE << (W::BITS as usize * (v % Self::LANES));
        }
    }

    #[inline(always)]
    fn and_word(a: u64, b: u64) -> u64 {
        a & b
    }

    /// The word's highest set bit names its highest nonzero lane, which is
    /// cleared before the next.
    #[inline(always)]
    fn each_variable(&self, j: usize, word: u64, mut f: impl FnMut(usize, u32)) {
        debug_assert!(j < self.words, "word {j} of {words}", words = self.words);
        let mut x = word;
        while x != 0 {
            let lane = (u64::BITS - 1 - x.leading_zeros()) / W::BITS;
            let shift = lane * W::BITS;
            f(
                j * Self::LANES + lane as usize,
                ((x >> shift) & Self::LANE).narrow(),
            );
            x &= !(Self::LANE << shift);
        }
    }
}
