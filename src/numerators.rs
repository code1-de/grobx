//! The numerators of a part of the lift's rationals over ℚ on one vector of
//! limbs, each with its chain denominator's index, and the view through
//! which every reader takes a numerator.

use num_bigint::{BigInt, Sign};

use crate::limbs::{big, bits};
use crate::rational::{Radix, Residue};

/// A numerator: its sign and its magnitude on limbs, least significant
/// first, with no zero limb at the top; 0 has none.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Numerator<'a> {
    negative: bool,
    magnitude: &'a [u64],
}

impl Numerator<'_> {
    /// L(|n|), as `BigInt::bits`.
    pub(crate) fn bits(self) -> u64 {
        bits(self.magnitude)
    }

    /// The limbs of |n|, as `text::limbs` of its magnitude.
    pub(crate) fn limbs(self) -> usize {
        self.magnitude.len()
    }

    // q-teardown-stop-latency.8 design-checked -- a text and the basis's integers make a BigInt of a numerator from its limbs as they read it, which the text frees as it goes, not their products and gcds on the lift's own limbs, which lost against num-bigint's where the numbers have many limbs (not measured)
    /// n as num-bigint's.
    pub(crate) fn to_bigint(self) -> BigInt {
        let sign = if self.negative {
            Sign::Minus
        } else {
            Sign::Plus
        };
        BigInt::from_biguint(sign, big(self.magnitude))
    }
}

/// What the crate's unit tests read of a numerator.
#[cfg(test)]
impl<'a> Numerator<'a> {
    /// Whether n < 0.
    pub(crate) fn is_negative(self) -> bool {
        self.negative
    }

    /// |n| on limbs.
    pub(crate) fn magnitude(self) -> &'a [u64] {
        self.magnitude
    }
}

impl Residue for Numerator<'_> {
    fn residue_at(&self, radix: &Radix) -> u64 {
        radix.signed(self.negative, radix.residue(self.magnitude))
    }
}

/// A numerator's record: the end of its limbs in `Numerators::limbs`, its
/// chain denominator's index, and its sign. It owns no memory.
#[derive(Clone, Copy)]
struct Term {
    // q-teardown-stop-latency.3 design-checked -- the end of a numerator's limbs is a usize, a record of 16 bytes, not a u32 and a record of 12, whose 2^32 limbs a part would be a limit a run can reach (not measured)
    end: usize,
    den: u32,
    negative: bool,
}

// q-teardown-stop-latency.1 practice-checked: arena (region) allocation -- a part's numerators lie in one Numerators, their magnitudes' limbs one after the other in one vector and a record per numerator that owns no memory in a second, not a (BigInt, u32) per numerator, which num-bigint gives a block of its own past one limb (log 0140-1)
// q-teardown-stop-latency.2 design-checked -- every numerator lies on the vector, whatever its size, not one below 2^62 in place and a larger one by a pointer to an integer of its own as monagan-2011 §5.1 holds a polynomial's coefficients, as most of a lift's numerators pass 62 bits and would stay a free each (log 0140-1)
/// Numerators, each with its chain denominator's index, in order: their
/// limbs one after the other in one vector, and a record per numerator in
/// a second.
#[derive(Clone, Default)]
pub(crate) struct Numerators {
    limbs: Vec<u64>,
    terms: Vec<Term>,
}

impl Numerators {
    /// No numerator.
    pub(crate) fn new() -> Numerators {
        Numerators::default()
    }

    // q-teardown-stop-latency.9 design-checked -- a numerator the reconstruction makes is pushed as its 64-bit digits copied onto the vector, its BigInt freed by the caller at once, not written onto the vector by the reconstruction, whose arithmetic is num-bigint's (not measured)
    /// Appends n over the chain denominator of index `den`, which is no
    /// less than the last numerator's: its 64-bit digits and its record.
    pub(crate) fn push(&mut self, n: &BigInt, den: u32) {
        debug_assert!(
            self.terms.last().is_none_or(|t| t.den <= den),
            "the index {den} falls"
        );
        self.limbs.extend(n.magnitude().iter_u64_digits());
        let end = self.limbs.len();
        debug_assert!(
            self.terms.last().map_or(0, |t| t.end) == end || self.limbs[end - 1] != 0,
            "no zero limb at the top"
        );
        let negative = match n.sign() {
            Sign::Minus => true,
            Sign::NoSign | Sign::Plus => false,
        };
        self.terms.push(Term { end, den, negative });
    }

    pub(crate) fn len(&self) -> usize {
        self.terms.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// No numerator; both vectors keep their capacity.
    pub(crate) fn clear(&mut self) {
        self.limbs.clear();
        self.terms.clear();
    }

    /// The last numerator and its chain denominator's index.
    pub(crate) fn last(&self) -> Option<(Numerator<'_>, u32)> {
        let (t, before) = self.terms.split_last()?;
        let start = before.last().map_or(0, |b| b.end);
        Some((view(&self.limbs, start, *t), t.den))
    }

    /// Every numerator, as a run.
    pub(crate) fn run(&self) -> Run<'_> {
        Run {
            limbs: &self.limbs,
            start: 0,
            terms: &self.terms,
        }
    }
}

/// The numerator whose limbs start at `start` in `limbs` and end at the
/// record's end.
fn view(limbs: &[u64], start: usize, t: Term) -> Numerator<'_> {
    Numerator {
        negative: t.negative,
        magnitude: &limbs[start..t.end],
    }
}

/// Consecutive numerators of a `Numerators`, each with its chain
/// denominator's index.
#[derive(Clone, Copy)]
pub(crate) struct Run<'a> {
    limbs: &'a [u64],
    /// Where the run's first numerator's limbs start in `limbs`.
    start: usize,
    terms: &'a [Term],
}

impl<'a> Run<'a> {
    pub(crate) fn len(self) -> usize {
        self.terms.len()
    }

    /// Each numerator in order, with its chain denominator's index.
    pub(crate) fn iter(self) -> impl Iterator<Item = (Numerator<'a>, u32)> + 'a {
        let mut start = self.start;
        self.terms.iter().map(move |&t| {
            let n = view(self.limbs, start, t);
            start = t.end;
            (n, t.den)
        })
    }

    /// The run in runs of `size`, at least 1, the last shorter where `size`
    /// does not divide its length.
    pub(crate) fn chunks(self, size: usize) -> impl Iterator<Item = Run<'a>> + 'a {
        debug_assert!(size > 0, "chunks of 0");
        let mut start = self.start;
        self.terms.chunks(size).map(move |terms| {
            let chunk = Run {
                limbs: self.limbs,
                start,
                terms,
            };
            start = terms.last().map_or(start, |t| t.end);
            chunk
        })
    }
}
