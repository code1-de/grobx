//! A run's generators of degree 2 or more with the variables its linear
//! generators lead substituted (ccode-newton-gap.1 to .8, .12 to .16, .19;
//! gf2-input-time.6 to .12; linear-substitution-time.1 to .9).

#[cfg(test)]
use std::cell::Cell;
use std::fmt::Debug;

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::field::Field;
use crate::halt::Halt;
use crate::id::MonoId;
use crate::module::Module;
use crate::monomial::{ANY, Format};
use crate::narrow::{self, Narrow};
use crate::order::Order;
use crate::pairs::per_words;
use crate::poly::Poly;
use crate::sort::{self, RUN, sort_runs};
use crate::table::{LOAD, Limit, Table};

#[cfg(test)]
thread_local! {
    /// The test profile's switch of the substitution on this thread: false
    /// turns it off (`substitutes`).
    pub(crate) static SUBSTITUTE: Cell<bool> = const { Cell::new(true) };
    /// The test profile's switch of the keys on this thread: false takes
    /// the run's words where the keys would apply (`indices`).
    pub(crate) static INDICES: Cell<bool> = const { Cell::new(true) };
}

// ccode-newton-gap.13 design-checked -- in the test profile a thread-local, SUBSTITUTE, turns the substitution off, not a public switch (not measured)
/// Whether a run that takes its echelon substitutes: always, or in the
/// test profile what `SUBSTITUTE` holds.
pub(crate) fn substitutes() -> bool {
    #[cfg(test)]
    if !SUBSTITUTE.get() {
        return false;
    }
    true
}

// linear-substitution-time.9 design-checked -- in the test profile a thread-local, INDICES, turns the keys off, not a parameter of the call (not measured)
/// Whether a substitution takes the keys where they apply (`keyed`):
/// always, or in the test profile what `INDICES` holds.
fn indices() -> bool {
    #[cfg(test)]
    if !INDICES.get() {
        return false;
    }
    true
}

/// A run's generators after `substitute`, with its counts.
pub(crate) struct Substituted<F: Field, D: Module> {
    /// The generators in their positions, each that took its substitute
    /// replaced by it.
    pub(crate) generators: Vec<Poly<F, D>>,
    /// The generators of degree 1.
    pub(crate) linear: usize,
    /// The variables substituted.
    pub(crate) pivots: usize,
    /// The generators that held a pivot and took their substitutes.
    pub(crate) changed: usize,
    /// The generators that held a pivot and stayed as read: a sum on the
    /// way to their substitutes, or the substitutes themselves, no shorter
    /// than they.
    pub(crate) stayed: usize,
    /// The terms of the changed generators before.
    pub(crate) terms: usize,
    /// Their terms after.
    pub(crate) after: usize,
}

/// The bits of a key's lane (`Indices`).
const KEY_LANE_BITS: u32 = 16;

/// A lane's bits at the bottom of a word.
const KEY_LANE: u64 = (1 << KEY_LANE_BITS) - 1;

/// The shift of a key's top lane.
const KEY_TOP: u32 = u64::BITS - KEY_LANE_BITS;

// linear-substitution-time.3 design-checked -- a key holds KEY_LANES = 4 lanes of KEY_LANE_BITS = 16 bits, not lanes of the bits the variable count needs, 7 on sr and 9 on bivium, nor keys of two words (log 0169-1, 0169-2)
/// The lanes of a key: the most variables, each with its multiplicity, of
/// a key's monomial, its degree's bound.
pub(crate) const KEY_LANES: u32 = u64::BITS / KEY_LANE_BITS;

/// The variable count the keys stay below: a lane holds a variable's index
/// plus 1, and 0 where it holds none.
pub(crate) const KEY_LIMIT: usize = (1 << KEY_LANE_BITS) - 1;

/// Whether a substitution's terms are keys (`Indices`): where the run has
/// fewer than `KEY_LIMIT` variables and `degree`, the largest degree of a
/// term of its generators of degree 2 or more, is `KEY_LANES` or less.
pub(crate) fn keyed(variables: usize, degree: u32) -> bool {
    variables < KEY_LIMIT && degree <= KEY_LANES
}

/// How Horner's polynomials hold their terms' monomials, `width` words a
/// monomial: as the run's exponent words (`Exponents`) or as keys
/// (`Indices`).
trait Encoding<M: Format>: Copy {
    type Word: Copy + Eq + Default + Debug;

    /// The words of a monomial.
    fn width(&self) -> usize;

    /// For a monomial of N words (`pairs::per_words`): the monomial a times
    /// b, a value's monomial of degree 0 or 1, to `out`. Its degree is at
    /// most its generator's (`horner`), which the run's format holds.
    fn times<const N: usize>(&self, a: &[Self::Word], b: &[Self::Word], out: &mut [Self::Word]);

    /// The table's monomial m, of degree 0 or 1, to `out`.
    fn value(&self, table: &Table<M>, m: MonoId, out: &mut [Self::Word]);

    /// The terms of g, which holds the pivots `held` of `pivots`, split
    /// into `split`, which is empty but for its levels (`Split`): per term
    /// its exponents of the held pivots, its rest, the rest's hash, and its
    /// coefficient; and its rank where the encoding ranks the terms. Each
    /// term's words read, its slot of the table, and its exponents of the
    /// held pivots: `PROBE` units, a unit a word, and a unit a held pivot,
    /// spent `sort::RUN` terms at a time at `Site::Substitute`. A stop
    /// where the budget's check stops the call.
    fn split<F: Field, D: Module>(
        &self,
        table: &Table<M>,
        g: &Poly<F, D>,
        pivots: &Pivots<F, M, Self::Word>,
        held: &[usize],
        split: &mut Split<F, Self::Word>,
        budget: &mut Budget,
    ) -> Result<(), Stop>;

    /// The monomial of `word` interned in the table under its hash `hash`,
    /// `out` a buffer of the run's words. What `Table::intern_hashed`
    /// returns.
    fn intern(
        &self,
        table: &mut Table<M>,
        word: &[Self::Word],
        hash: u64,
        out: &mut [M::Word],
        budget: &mut Budget,
    ) -> Result<MonoId, Limit>;
}

/// A monomial as the run's exponent words.
#[derive(Clone, Copy)]
struct Exponents<M: Format>(M);

impl<M: Format> Encoding<M> for Exponents<M> {
    type Word = M::Word;

    fn width(&self) -> usize {
        self.0.words()
    }

    /// A word of a times b is `mul_word` of theirs, the word of the
    /// monomial 1 leaving a's as it is.
    #[inline(always)]
    fn times<const N: usize>(&self, a: &[M::Word], b: &[M::Word], out: &mut [M::Word]) {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = M::mul_word(x, y);
        }
    }

    fn value(&self, table: &Table<M>, m: MonoId, out: &mut [M::Word]) {
        out.copy_from_slice(table.monomial(m));
    }

    // gf2-input-time.7 design-checked -- with the run's words a term is split at the pivots by clearing their lanes in a copy of its words, its rest's hash its own less the exponents cleared, not unpacked, its exponents of the pivots taken, and encoded again (log 0158-1, 0158-3)
    fn split<F: Field, D: Module>(
        &self,
        table: &Table<M>,
        g: &Poly<F, D>,
        pivots: &Pivots<F, M, M::Word>,
        held: &[usize],
        split: &mut Split<F, M::Word>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let format = &self.0;
        let (w, k) = (format.words(), held.len());
        for (ms, cs) in g.monomials.chunks(RUN).zip(g.coefs.chunks(RUN)) {
            for (&m, &c) in ms.iter().zip(cs) {
                let at = split.rest.len();
                split.rest.extend_from_slice(table.monomial(m));
                let mut hash = table.hash(m);
                for &p in held {
                    let var = pivots.list[p].var;
                    let e = format.clear(&mut split.rest[at..], var);
                    hash = hash.wrapping_sub(table.multiplier(var).wrapping_mul(u64::from(e)));
                    split.alpha.push(e);
                }
                split.hashes.push(hash);
                split.coefs.push(c);
            }
            budget.spend(ms.len() * (PROBE + w + k), Site::Substitute)?;
        }
        Ok(())
    }

    fn intern(
        &self,
        table: &mut Table<M>,
        word: &[M::Word],
        hash: u64,
        _: &mut [M::Word],
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        let degree = self.0.degree(word);
        table.intern_hashed(word, degree, hash, budget)
    }
}

// linear-substitution-time.2 practice-checked: sparse monomial representation -- where the run has fewer than KEY_LIMIT variables and no generator of degree 2 or more has a term past KEY_LANES, Horner's polynomials hold each monomial as one word of its variables' indices plus 1, by decreasing index from the top lane, a product by a value's variable an insertion, not the run's exponent words for every term (log 0169-1, 0169-2)
/// A monomial as a key of one word: its variables, each as often as its
/// exponent, as the variable's index plus 1 in a lane of `KEY_LANE_BITS`
/// bits, by decreasing index from the top lane, the lanes below them 0.
/// The monomial 1 is the key 0. A key's monomial has degree `KEY_LANES` or
/// less, and its variables' indices lie below `KEY_LIMIT` (`keyed`).
#[derive(Clone, Copy)]
struct Indices<M: Format>(M);

/// The key `key` times the variable whose index plus 1 is `x`: x's lane
/// placed below the lanes that hold more than x, and the lanes below it
/// moved one lane down. The key holds fewer than `KEY_LANES` variables.
#[inline(always)]
fn insert(key: u64, x: u64) -> u64 {
    debug_assert_eq!(key & KEY_LANE, 0, "a key of fewer than KEY_LANES variables");
    debug_assert!(x != 0 && x <= KEY_LANE, "a variable's lane");
    let mut above = 0;
    for l in 0..KEY_LANES {
        above += u32::from(((key >> (KEY_TOP - KEY_LANE_BITS * l)) & KEY_LANE) > x);
    }
    let high = !(u64::MAX >> (KEY_LANE_BITS * above));
    (key & high) | (x << (KEY_TOP - KEY_LANE_BITS * above)) | ((key & !high) >> KEY_LANE_BITS)
}

/// The rank's bits of a held pivot's exponent (`rank`).
const RANK_BITS: u32 = u8::BITS;

/// The most held pivots a rank holds the exponents of.
const RANKED: usize = (u64::BITS / RANK_BITS) as usize;

/// A term's rank by its exponents of the held pivots, `RANKED` or fewer,
/// each below 2^`RANK_BITS`: the first one's in the top bits. Two ranks
/// compare as their exponents do, lexicographically.
#[inline(always)]
fn rank(exponents: &[u32]) -> u64 {
    debug_assert!(
        exponents.len() <= RANKED,
        "{n} exponents",
        n = exponents.len()
    );
    let mut rank = 0;
    let mut shift = u64::BITS;
    for &e in exponents {
        debug_assert!(e < 1 << RANK_BITS, "an exponent of {e}");
        shift -= RANK_BITS;
        rank |= u64::from(e) << shift;
    }
    rank
}

impl<M: Format> Encoding<M> for Indices<M> {
    type Word = u64;

    fn width(&self) -> usize {
        1
    }

    #[inline(always)]
    fn times<const N: usize>(&self, a: &[u64], b: &[u64], out: &mut [u64]) {
        out[0] = if b[0] == 0 {
            a[0]
        } else {
            insert(a[0], b[0] >> KEY_TOP)
        };
    }

    fn value(&self, table: &Table<M>, m: MonoId, out: &mut [u64]) {
        let mut key = 0;
        for (j, &word) in table.monomial(m).iter().enumerate() {
            self.0.each_variable(j, word, |v, _| {
                key = u64::from(Narrow::<u16>::narrow(v + 1)) << KEY_TOP;
            });
        }
        out[0] = key;
    }

    // linear-substitution-time.6 design-checked -- with the keys the split reads each term's words once, each variable of a nonzero exponent to its held pivot's level or into the key, not the pivots' exponents read one by one and the rest's words formed by clearing their lanes in a copy (gf2-input-time.7), then encoded (log 0169-1, 0169-2)
    fn split<F: Field, D: Module>(
        &self,
        table: &Table<M>,
        g: &Poly<F, D>,
        pivots: &Pivots<F, M, u64>,
        held: &[usize],
        split: &mut Split<F, u64>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let format = &self.0;
        let (w, k) = (format.words(), held.len());
        let Split {
            alpha,
            rest,
            hashes,
            coefs,
            ranks,
            levels,
            ..
        } = split;
        for (l, &p) in held.iter().enumerate() {
            levels[p] = l;
        }
        for (ms, cs) in g.monomials.chunks(RUN).zip(g.coefs.chunks(RUN)) {
            for (&m, &c) in ms.iter().zip(cs) {
                let at = alpha.len();
                alpha.resize(at + k, 0);
                let (mut key, mut lanes) = (0, 0);
                let mut hash = table.hash(m);
                // From the last word, so that the lanes stand by decreasing index.
                for (j, &word) in table.monomial(m).iter().enumerate().rev() {
                    format.each_variable(j, word, |v, e| {
                        let pivot = pivots.of[v];
                        if pivot == u32::MAX {
                            debug_assert!(lanes + e <= KEY_LANES, "a key's degree");
                            let lane = u64::from(Narrow::<u16>::narrow(v + 1));
                            for _ in 0..e {
                                key |= lane << (KEY_TOP - KEY_LANE_BITS * lanes);
                                lanes += 1;
                            }
                        } else {
                            alpha[at + levels[pivot as usize]] = e;
                            hash =
                                hash.wrapping_sub(table.multiplier(v).wrapping_mul(u64::from(e)));
                        }
                    });
                }
                // linear-substitution-time.7 cited-checked: gokavarapu-2026 §Graded orders, §7.3 -- where degrees are bounded each exponent fits a fixed-width lane and the vector packs into one or two machine words, a key that admits fast comparison for sorting
                if k <= RANKED {
                    ranks.push(rank(&alpha[at..]));
                }
                rest.push(key);
                hashes.push(hash);
                coefs.push(c);
            }
            budget.spend(ms.len() * (PROBE + w + k), Site::Substitute)?;
        }
        Ok(())
    }

    fn intern(
        &self,
        table: &mut Table<M>,
        word: &[u64],
        hash: u64,
        out: &mut [M::Word],
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        out.fill(M::Word::default());
        let mut degree = 0;
        for l in 0..KEY_LANES {
            let x = (word[0] >> (KEY_TOP - KEY_LANE_BITS * l)) & KEY_LANE;
            if x == 0 {
                break;
            }
            let (j, unit) = self.0.unit(usize::from(Narrow::<u16>::narrow(x)) - 1);
            out[j] = M::mul_word(out[j], unit);
            degree += 1;
        }
        table.intern_hashed(out, degree, hash, budget)
    }
}

// gf2-input-time.8 cited-checked: monagan-2015 §2.1, attributes it to Faugère and Magma [16], a private communication -- when multiplying two monomials, their hashes are added and a lookup is performed to find the product, which made monomial multiplications 20% faster
/// A polynomial in an encoding's words: its terms' words back to back,
/// the encoding's width per term, with their hashes and their
/// coefficients, in no order, no two terms of the same words and no
/// coefficient 0. A term's hash is the hash of its monomial in the run's
/// table (`Table::hash`), so a product's hash is the sum of its factors'.
struct Sparse<F: Field, W> {
    words: Vec<W>,
    hashes: Vec<u64>,
    coefs: Vec<F::Coef>,
}

impl<F: Field, W: Copy> Sparse<F, W> {
    fn new() -> Self {
        Sparse {
            words: Vec::new(),
            hashes: Vec::new(),
            coefs: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.coefs.len()
    }

    fn clear(&mut self) {
        self.words.clear();
        self.hashes.clear();
        self.coefs.clear();
    }

    #[inline]
    fn push(&mut self, word: &[W], hash: u64, coef: F::Coef) {
        self.words.extend_from_slice(word);
        self.hashes.push(hash);
        self.coefs.push(coef);
    }

    /// The words of term `k`, of `w` words per term (`term`).
    #[inline]
    fn word<const N: usize>(&self, k: usize, w: usize) -> &[W] {
        term::<N, _>(&self.words, k, w)
    }
}

/// The words of term `k` of `words`, of `w` words per term: of N words
/// where N is not `ANY`, a slice whose length the compiler knows
/// (`pairs::per_words`).
#[inline(always)]
fn term<const N: usize, T>(words: &[T], k: usize, w: usize) -> &[T] {
    let w = if N == ANY { w } else { N };
    &words[k * w..][..w]
}

/// The units of work of a term summed (`Sum::add`): its words formed, a
/// product's multiplied or a term's copied, a unit a word; its slot probed,
/// `PROBE`; and its words compared or stored, a unit a word.
fn summed_units(w: usize) -> usize {
    PROBE + 2 * w
}

/// The slots of an empty sum.
const SUM_SLOTS: usize = 1 << 6;

// gf2-input-time.9 cited-checked: yan-1998 §B Hashing -- hash tables support roughly constant time access, so addition can be performed in O(N) time simply by inserting terms, and cancelation is easily performed during insertion
// gf2-input-time.11 practice-checked: generation stamps -- a slot holds a stamp beside its term's index, and a new stamp empties every slot at once, not the slots of the drained terms emptied one by one nor the slot array refilled
// linear-substitution-time.4 design-checked -- Horner's polynomials and the sums carry no degree per term, a substitute's term taking its degree as it enters the run's table, not the degree carried per term and summed in each product (log 0169-1, 0169-2)
/// A polynomial summed term by term: its terms in the order they first
/// came, each with its words, hash, and coefficient, a coefficient 0 kept
/// where terms cancel; and the slots, per slot `stamp << 32 | index` of a
/// term, probed linearly from the hash's top bits. A slot is empty where
/// its stamp is not the sum's, so a new stamp empties every slot. The
/// slots are at least `LOAD` times the terms.
struct Sum<F: Field, W> {
    words: Vec<W>,
    hashes: Vec<u64>,
    coefs: Vec<F::Coef>,
    slots: Vec<u64>,
    /// 64 minus log2 of the slot count: the hash shifted right by it is a
    /// slot.
    shift: u32,
    /// The stamp of the slots of the sum's terms, never 0, the stamp of the
    /// slots of a new slot array.
    stamp: u32,
}

impl<F: Field, W: Copy + Eq> Sum<F, W> {
    fn new() -> Self {
        Sum {
            words: Vec::new(),
            hashes: Vec::new(),
            coefs: Vec::new(),
            slots: vec![0; SUM_SLOTS],
            shift: 64 - SUM_SLOTS.trailing_zeros(),
            stamp: 1,
        }
    }

    /// For N words a term: adds the term of these words, hash, and
    /// coefficient to the coefficient of the sum's term of the same words,
    /// or as a new term. The slots double where a new term takes them below
    /// `LOAD` times the terms (`grow`). `Halt::Size` where the
    /// terms are more than a u32 index numbers, and the stop where the
    /// budget's check stops the call in a growth.
    #[inline(always)]
    fn add<const N: usize>(
        &mut self,
        field: &F,
        word: &[W],
        hash: u64,
        coef: F::Coef,
        budget: &mut Budget,
    ) -> Result<(), Halt> {
        let w = word.len();
        let last = self.slots.len() - 1;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the shift keeps the slot below the sum's slot count"
        )]
        let mut slot = (hash >> self.shift) as usize;
        loop {
            let s = self.slots[slot];
            if (s >> u32::BITS) as u32 != self.stamp {
                let index = narrow::id(self.coefs.len())?;
                self.slots[slot] = (u64::from(self.stamp) << u32::BITS) | u64::from(index);
                self.words.extend_from_slice(word);
                self.hashes.push(hash);
                self.coefs.push(coef);
                if LOAD * self.coefs.len() > self.slots.len() {
                    self.grow(budget)?;
                }
                return Ok(());
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the low 32 bits hold a term's index"
            )]
            let i = (s & u64::from(u32::MAX)) as usize;
            if self.hashes[i] == hash && term::<N, _>(&self.words, i, w) == word {
                self.coefs[i] = field.add(self.coefs[i], coef);
                return Ok(());
            }
            slot = (slot + 1) & last;
        }
    }

    /// Doubles the slots and places every term again: `PROBE` units a term,
    /// at `Site::Substitute`.
    #[cold]
    #[inline(never)]
    fn grow(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let size = 2 * self.slots.len();
        self.slots.clear();
        self.slots.resize(size, 0);
        self.shift -= 1;
        self.stamp = 1;
        let last = size - 1;
        for (index, &hash) in (0u64..).zip(&self.hashes) {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the shift keeps the slot below the sum's slot count"
            )]
            let mut slot = (hash >> self.shift) as usize;
            while self.slots[slot] != 0 {
                slot = (slot + 1) & last;
            }
            self.slots[slot] = (u64::from(self.stamp) << u32::BITS) | index;
        }
        budget.spend(self.hashes.len() * PROBE, Site::Substitute)
    }

    /// The sum's terms whose coefficients are not 0, in the order they
    /// first came, as `out`, which is empty and whose buffers the sum takes,
    /// and the sum emptied, under a new stamp: a unit per term and word, at
    /// `Site::Substitute`.
    fn drain(&mut self, out: &mut Sparse<F, W>, w: usize, budget: &mut Budget) -> Result<(), Stop> {
        debug_assert_eq!(out.len(), 0, "an empty polynomial");
        let zero = F::Coef::default();
        let mut kept = 0;
        for i in 0..self.coefs.len() {
            if self.coefs[i] == zero {
                continue;
            }
            if kept < i {
                self.words.copy_within(i * w..(i + 1) * w, kept * w);
                self.hashes[kept] = self.hashes[i];
                self.coefs[kept] = self.coefs[i];
            }
            kept += 1;
        }
        budget.spend(self.coefs.len() * (1 + w), Site::Substitute)?;
        self.words.truncate(kept * w);
        self.hashes.truncate(kept);
        self.coefs.truncate(kept);
        std::mem::swap(&mut self.words, &mut out.words);
        std::mem::swap(&mut self.hashes, &mut out.hashes);
        std::mem::swap(&mut self.coefs, &mut out.coefs);
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.slots.fill(0);
            self.stamp = 1;
        }
        Ok(())
    }
}

/// For the test code: a round of `sums`, the stamp the sum takes first
/// where it names one, and the round's terms, each its words, hash, and
/// coefficient's value.
#[cfg(test)]
pub(crate) type Round<W> = (Option<u32>, Vec<(Vec<W>, u64, u64)>);

/// For the test code: each round's terms added to one sum of `w` words a
/// term, after the round's stamp, and drained after the round: per round
/// the drained terms' words and coefficients' values, in the order they
/// first came.
#[cfg(test)]
pub(crate) fn sums<F: Field, M: Format>(
    field: &F,
    w: usize,
    rounds: &[Round<M::Word>],
) -> Vec<Vec<(Vec<M::Word>, u64)>> {
    let mut sum = Sum::<F, M::Word>::new();
    let mut budget = Budget::unbounded();
    rounds
        .iter()
        .map(|(stamp, terms)| {
            if let Some(stamp) = stamp {
                sum.stamp = *stamp;
            }
            for (word, hash, c) in terms {
                sum.add::<ANY>(field, word, *hash, field.coef(*c), &mut budget)
                    .expect("a sum under a budget without bounds");
            }
            let mut out = Sparse::new();
            sum.drain(&mut out, w, &mut budget)
                .expect("a drain under a budget without bounds");
            (0..out.len())
                .map(|k| (out.word::<ANY>(k, w).to_vec(), field.value(out.coefs[k])))
                .collect()
        })
        .collect()
}

/// A pivot: the variable that leads a linear row, and its value, minus the
/// row's other terms, each a monomial of degree 1 or the monomial 1.
struct Pivot<F: Field, W> {
    var: usize,
    value: Sparse<F, W>,
}

/// A substitution's pivots, by increasing variable, with their values in an
/// encoding's words; per word of the run's format the mask of the pivots'
/// lanes (`Format::lanes`); and per variable the index of its pivot in
/// `list`, `u32::MAX` where it is none.
struct Pivots<F: Field, M: Format, W> {
    list: Vec<Pivot<F, W>>,
    masks: Vec<M::Word>,
    of: Vec<u32>,
}

/// A generator's terms as `horner` takes them: per term its exponents of
/// the pivots the generator holds, `alpha`, one per such pivot; its
/// monomial with those exponents 0, `rest`, in the encoding's words, with
/// its hash; and its coefficient. `ranks` holds a rank per term
/// (`rank`) where the encoding ranks them, and is empty elsewhere. `sorted`
/// orders the terms by `alpha` decreasing, lexicographically. `levels`
/// holds per pivot the level of the generator's Horner's rule it stands
/// at, where the generator holds it.
struct Split<F: Field, W> {
    alpha: Vec<u32>,
    rest: Vec<W>,
    hashes: Vec<u64>,
    coefs: Vec<F::Coef>,
    sorted: Vec<usize>,
    ranks: Vec<u64>,
    levels: Vec<usize>,
}

impl<F: Field, W> Split<F, W> {
    /// Empty but for its levels.
    fn clear(&mut self) {
        self.alpha.clear();
        self.rest.clear();
        self.hashes.clear();
        self.coefs.clear();
        self.sorted.clear();
        self.ranks.clear();
    }
}

/// The buffers the substitution keeps from product to product: the
/// polynomials given back, the words of the term a product forms, and the
/// sum Horner's rule sums in. The check copy's profile spent most of a
/// product's instructions on vectors allocated per call (log 0118-1).
struct Scratch<F: Field, W> {
    free: Vec<Sparse<F, W>>,
    word: Vec<W>,
    sum: Sum<F, W>,
}

impl<F: Field, W: Copy> Scratch<F, W> {
    /// An empty polynomial, one given back where there is one.
    fn take(&mut self) -> Sparse<F, W> {
        self.free.pop().unwrap_or_else(Sparse::new)
    }

    /// `p` given back, emptied.
    fn give(&mut self, mut p: Sparse<F, W>) {
        p.clear();
        self.free.push(p);
    }
}

// ccode-newton-gap.1 cited-checked: buchberger-1985 §6.4 (2) -- the reduction of the polynomials modulo the other polynomials in the basis should also be performed at the beginning of the algorithm
// ccode-newton-gap.2 design-checked -- only the linear generators reduce the others, not normal forms modulo every other generator's multiples, buchberger-1985's ReduceAll, which is the run's own work (log 0118-1)
// ccode-newton-gap.3 design-checked -- every variable stays, the generators alone change, not the run in the variables that lead no row with its basis assembled after it (log 0118-1)
// ccode-newton-gap.14 design-checked -- a linear generator is one whose every term has degree at most 1 and one degree 1, in every order, not every generator whose leading monomial is a variable (not measured)
/// The generators with every variable that leads a row of the linear
/// generators' reduced row echelon form replaced, in each generator of
/// degree 2 or more, by minus the row's other terms, where that makes the
/// generator shorter. The result generates the same ideal. The linear
/// generators, and every generator that holds no pivot, stay as they are.
/// Spends as it goes, at `Site::Substitute`, the units its doc comment
/// names. A stop where the budget's check stops the call, `Halt::Size`
/// where the table is full, and what the linear generators' echelon halts
/// with.
pub(crate) fn substitute<F: Field, M: Format, O: Order<M>, D: Module>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    budget: &mut Budget,
) -> Result<Substituted<F, D>, Halt> {
    let mut linear = Vec::with_capacity(generators.len());
    for g in &generators {
        linear.push(is_linear(table, g, budget)?);
    }
    let count = linear.iter().filter(|&&l| l).count();
    let mut out = Substituted {
        generators,
        linear: count,
        pivots: 0,
        changed: 0,
        stayed: 0,
        terms: 0,
        after: 0,
    };
    if count == 0 {
        return Ok(out);
    }
    // call-stop-latency.25 design-checked -- the substitution spends the terms its scans, levels, splits, sums, and internings walk, as they walk them, not a unit per term read and per product formed alone (ccode-newton-gap.12, .18), which left the merges of the time walking up to |acc|·|value|²/2 terms unspent (log 0138-6)
    let mut copies = Vec::with_capacity(count);
    for (g, _) in out.generators.iter().zip(&linear).filter(|&(_, &l)| l) {
        copies.push(g.clone());
        budget.spend(g.len(), Site::Substitute)?;
    }
    // ccode-newton-gap.4 design-checked -- the linear generators' rows come from echelon::echelon over them alone, before the run's echelon, not the run's echelon first, its rows of degree 1 substituted, and the echelon again (not measured)
    let rows = crate::echelon::echelon(field, table, order, copies, budget)?.rows;
    let format = *table.format();
    let w = format.words();
    let n = format.variables();
    let mut exponents = vec![0u32; n];
    // ccode-newton-gap.15 design-checked -- a row of the linear echelon that leads with 1 is no pivot, and the substitution goes on by the others, not a return of the unit ideal (not measured)
    let mut leads: Vec<(usize, &Poly<F, D>)> = Vec::new();
    for row in rows.iter().filter(|r| !r.is_zero()) {
        // Its lead unpacked, and each term's words and degree read.
        budget.spend(n + row.len() * (PROBE + w), Site::Substitute)?;
        let lead = row.lead();
        if table.degree(lead) == 0 {
            continue;
        }
        table.unpack(lead, &mut exponents);
        #[allow(
            clippy::expect_used,
            reason = "a monomial of degree 1 has one exponent 1"
        )]
        let var = exponents
            .iter()
            .position(|&e| e == 1)
            .expect("a monomial of degree 1 has one exponent 1");
        leads.push((var, row));
    }
    leads.sort_unstable_by_key(|&(var, _)| var);
    out.pivots = leads.len();
    if leads.is_empty() {
        return Ok(out);
    }
    let keys = indices() && keyed(n, top_degree(table, &out.generators, &linear, budget)?);
    // ccode-newton-gap.19 cited-checked:demin-2024 §4.1 -- the specialized functions that implement monomial arithmetic are compiled and inlined, which relocates the work from running time to compile time; bachmann-1998 §3 -- loop unrolling and pointer arithmetic
    if keys {
        each::<1, _, _, _, _, _>(
            field,
            table,
            order,
            Indices(format),
            &leads,
            &linear,
            &mut out,
            budget,
        )?;
    } else {
        per_words!(
            w,
            each::<_, _, _, _, _>(
                field,
                table,
                order,
                Exponents(format),
                &leads,
                &linear,
                &mut out,
                budget,
            )
        )?;
    }
    Ok(out)
}

/// For a monomial of N words of the encoding (`pairs::per_words`): in
/// `out`, each generator of degree 2 or more, by `linear`, that holds a
/// pivot replaced by its substitute where that is shorter, and the counts
/// of `out`. `leads` holds per pivot its variable and its row, by
/// increasing variable.
#[allow(
    clippy::too_many_arguments,
    reason = "the substitution's state: the field, the table, the order, the encoding, the pivots' rows, the linear generators, the result, and the budget"
)]
fn each<const N: usize, F: Field, M: Format, O: Order<M>, D: Module, E: Encoding<M>>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    encoding: E,
    leads: &[(usize, &Poly<F, D>)],
    linear: &[bool],
    out: &mut Substituted<F, D>,
    budget: &mut Budget,
) -> Result<(), Halt> {
    let format = *table.format();
    let (w, ew, n) = (format.words(), encoding.width(), format.variables());
    let mut pivots = Pivots {
        list: Vec::with_capacity(leads.len()),
        masks: vec![M::Word::default(); w],
        of: vec![u32::MAX; n],
    };
    let mut word = vec![E::Word::default(); ew];
    let mut vars = Vec::with_capacity(leads.len());
    for (p, &(var, row)) in leads.iter().enumerate() {
        let mut value = Sparse::new();
        for (&m, &c) in row.monomials.iter().zip(&row.coefs).skip(1) {
            encoding.value(table, m, &mut word);
            value.push(&word, table.hash(m), field.neg(c));
        }
        pivots.list.push(Pivot { var, value });
        pivots.of[var] = narrow::id(p)?;
        vars.push(var);
    }
    format.lanes(&vars, &mut pivots.masks);
    // The variables' pivots and the masks filled.
    budget.spend(n + w, Site::Substitute)?;
    let mut split = Split {
        alpha: Vec::new(),
        rest: Vec::new(),
        hashes: Vec::new(),
        coefs: Vec::new(),
        sorted: Vec::new(),
        ranks: Vec::new(),
        levels: vec![0; leads.len()],
    };
    let mut scratch = Scratch {
        free: Vec::new(),
        word,
        sum: Sum::new(),
    };
    let mut found = vec![false; leads.len()];
    let mut held: Vec<usize> = Vec::new();
    let mut made: Vec<(MonoId, F::Coef)> = Vec::new();
    let mut words = vec![M::Word::default(); w];
    // ccode-newton-gap.5 design-checked -- the linear generators stay as read, at their positions, their rows serving the substitution alone, not each replaced by its row (not measured)
    // ccode-newton-gap.6 design-checked -- the substitution runs once, from the generators of degree 1 as read, not again until no generator of degree 1 is left (not measured)
    for (g, &l) in out.generators.iter_mut().zip(linear) {
        if l || g.is_zero() {
            continue;
        }
        held_pivots(table, g, &pivots, &mut found, &mut held, budget)?;
        if held.is_empty() {
            continue;
        }
        let k = held.len();
        split.clear();
        encoding.split(table, g, &pivots, &held, &mut split, budget)?;
        let Split {
            alpha,
            ranks,
            sorted,
            ..
        } = &mut split;
        sorted.extend(0..g.len());
        // gf2-input-time.12 design-checked -- the split terms are sorted by their exponents of the pivots alone, which Horner's levels group them by, not by those and then their rests, which no sum reads in order, the exponents read from the ranks where the keys rank them (linear-substitution-time.7) (log 0158-3)
        if ranks.len() == g.len() {
            // linear-substitution-time.7: the ranks order the terms as their exponents of the pivots do.
            sort_runs(
                sorted,
                |&a, &b| ranks[b].cmp(&ranks[a]),
                budget,
                Site::Substitute,
            )?;
        } else {
            sort_runs(
                sorted,
                |&a, &b| alpha[b * k..(b + 1) * k].cmp(&alpha[a * k..(a + 1) * k]),
                budget,
                Site::Substitute,
            )?;
        }
        let result = horner::<N, F, M, E>(
            field,
            &encoding,
            &pivots.list,
            &held,
            &split,
            &split.sorted,
            0,
            g.len(),
            &mut scratch,
            budget,
        )?;
        // ccode-newton-gap.16 tuned-checked -- a generator's substitute replaces it only where it has fewer terms than the generator, not every substitute, which made the denser generators' rounds larger, nor every substitute of as many terms or fewer, whose equal ones changed phc-rps10's path (log 0118-1)
        let result = match result {
            Some(result) if result.len() < g.len() => result,
            Some(result) => {
                out.stayed += 1;
                scratch.give(result);
                continue;
            }
            None => {
                out.stayed += 1;
                continue;
            }
        };
        // gf2-input-time.10 design-checked -- Horner's intermediate polynomials hold their terms' words, the run's or the keys (linear-substitution-time.2), and only the result's monomials enter the run's table, the part of ccode-newton-gap.9 that stands, not every product interned in the run's table and summed by its id (log 0158-3)
        made.clear();
        for j in 0..result.len() {
            made.push((
                encoding.intern(
                    table,
                    result.word::<N>(j, ew),
                    result.hashes[j],
                    &mut words,
                    budget,
                )?,
                result.coefs[j],
            ));
            if made.len().is_multiple_of(RUN) {
                budget.spend(RUN * (PROBE + w), Site::Substitute)?;
            }
        }
        budget.spend((made.len() % RUN) * (PROBE + w), Site::Substitute)?;
        scratch.give(result);
        let substitute = sort::poly(
            &mut made,
            field,
            table,
            order,
            g.position,
            budget,
            Site::Substitute,
        )?;
        out.changed += 1;
        out.terms += g.len();
        out.after += substitute.len();
        *g = substitute;
    }
    Ok(())
}

/// Whether `g` is linear: nonzero, every term of degree at most 1 and one
/// of degree 1. The degrees are read `PROBE` units a term, spent
/// `sort::RUN` terms at a time at `Site::Substitute`. A stop where the
/// budget's check stops the call.
fn is_linear<F: Field, M: Format, D: Module>(
    table: &Table<M>,
    g: &Poly<F, D>,
    budget: &mut Budget,
) -> Result<bool, Stop> {
    let mut one = false;
    for chunk in g.monomials.chunks(RUN) {
        budget.spend(chunk.len() * PROBE, Site::Substitute)?;
        for &m in chunk {
            let degree = table.degree(m);
            if degree > 1 {
                return Ok(false);
            }
            one |= degree == 1;
        }
    }
    Ok(one)
}

// linear-substitution-time.10 design-checked -- the largest degree of the generators that are not linear is read where the linear echelon gave a pivot, in a walk of its own that ends past KEY_LANES, not in is_linear, which would read every term of every call's generators (log 0169-2)
/// The largest degree of a term of the generators that are not linear, by
/// `linear`, read until one passes `KEY_LANES`, which `keyed` then fails
/// at: `PROBE` units a term, spent `sort::RUN` terms at a time at
/// `Site::Substitute`. A stop where the budget's check stops the call.
fn top_degree<F: Field, M: Format, D: Module>(
    table: &Table<M>,
    generators: &[Poly<F, D>],
    linear: &[bool],
    budget: &mut Budget,
) -> Result<u32, Stop> {
    let mut top = 0;
    for (g, _) in generators.iter().zip(linear).filter(|&(_, &l)| !l) {
        for chunk in g.monomials.chunks(RUN) {
            budget.spend(chunk.len() * PROBE, Site::Substitute)?;
            for &m in chunk {
                top = top.max(table.degree(m));
            }
            if top > KEY_LANES {
                return Ok(top);
            }
        }
    }
    Ok(top)
}

// gf2-input-time.6 design-checked -- Horner's rule runs over the pivots the generator holds alone, not over every pivot, each of whose levels walked every term of the generator (log 0158-1, 0158-3)
// linear-substitution-time.5 cited-checked: monagan-2014 §2 (4) -- groups of exponents are isolated by masks, which eliminates branching and loops at the level of the exponents
/// The pivots `g` holds, their indices into the pivots' list in increasing
/// order, to `held`: those whose variable a term of g holds. Each word of a
/// term is read against the mask of the pivots' lanes, and each nonzero
/// lane the two share names a pivot: `PROBE` units and a unit a word a
/// term, spent `sort::RUN` terms at a time at `Site::Substitute`. The walk
/// ends where every pivot is found. `found` holds one false per pivot, and
/// holds them again on return. A stop where the budget's check stops the
/// call.
fn held_pivots<F: Field, M: Format, D: Module, W>(
    table: &Table<M>,
    g: &Poly<F, D>,
    pivots: &Pivots<F, M, W>,
    found: &mut [bool],
    held: &mut Vec<usize>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    debug_assert_eq!(found.len(), pivots.list.len(), "a flag per pivot");
    let format = table.format();
    held.clear();
    'terms: for chunk in g.monomials.chunks(RUN) {
        budget.spend(chunk.len() * (PROBE + format.words()), Site::Substitute)?;
        for &m in chunk {
            let words = table.monomial(m).iter().zip(&pivots.masks);
            for (j, (&word, &mask)) in words.enumerate() {
                format.each_variable(j, M::and_word(word, mask), |v, _| {
                    let p = pivots.of[v] as usize;
                    debug_assert!(p < found.len(), "variable {v} is a pivot");
                    if !found[p] {
                        found[p] = true;
                        held.push(p);
                    }
                });
            }
            if held.len() == found.len() {
                break 'terms;
            }
        }
    }
    held.sort_unstable();
    for &p in held.iter() {
        found[p] = false;
    }
    Ok(())
}

// ccode-newton-gap.7 practice-checked: Horner's rule -- each level evaluates its polynomial in a pivot by Horner's rule in the pivot's value, not each term's powers of the values multiplied out
// ccode-newton-gap.8 practice-checked: recursive (multivariate) Horner scheme -- Horner's rule nested over the pivots, every intermediate polynomial in the variables that lead no row, not Horner's rule pivot by pivot, whose intermediates hold the later pivots (log 0118-1)
// linear-substitution-time.1 design-checked -- a generator's substitution ends at the first sum Horner's rule drains that holds as many terms as the generator or more, and the generator stays as read, not every substitute formed whole and compared with its generator, ccode-newton-gap.16's rule alone (log 0169-1, 0169-2)
/// For a monomial of N words of the encoding (`pairs::per_words`): the
/// terms `terms` of `split`, which share their exponents of the held
/// pivots before `level` and stand in `split.sorted`'s order, with the
/// held pivots from `level` on replaced by their values: by Horner's rule
/// in the value of the held pivot `level`, whose coefficients are this
/// function at the next level. `held` names the pivots of `pivots` that
/// `split.alpha` holds the exponents of. At the last level the terms'
/// rests are distinct. None where a sum drained on the way holds `limit`
/// terms or more, each polynomial it held given back to `scratch`.
#[allow(
    clippy::too_many_arguments,
    reason = "the recursion's state: the field, the encoding, the pivots, the held ones, the terms, the level, the limit, the buffers, and the budget"
)]
fn horner<const N: usize, F: Field, M: Format, E: Encoding<M>>(
    field: &F,
    encoding: &E,
    pivots: &[Pivot<F, E::Word>],
    held: &[usize],
    split: &Split<F, E::Word>,
    terms: &[usize],
    level: usize,
    limit: usize,
    scratch: &mut Scratch<F, E::Word>,
    budget: &mut Budget,
) -> Result<Option<Sparse<F, E::Word>>, Halt> {
    let k = held.len();
    debug_assert!(level <= k, "level {level}");
    debug_assert!(
        !terms.is_empty(),
        "a generator that holds a pivot has a term"
    );
    let w = encoding.width();
    // The level's walk of its terms, and at the last their words copied.
    let copies = if level == k { w } else { 0 };
    budget.spend(terms.len() * (1 + copies), Site::Substitute)?;
    if level == k {
        let mut out = scratch.take();
        for &i in terms {
            out.push(
                term::<N, _>(&split.rest, i, w),
                split.hashes[i],
                split.coefs[i],
            );
        }
        return Ok(Some(out));
    }
    let value = &pivots[held[level]].value;
    let alpha = |i: usize| split.alpha[i * k + level];
    let mut acc: Option<Sparse<F, E::Word>> = None;
    let mut prev = 0;
    // linear-substitution-time.8 design-checked -- the terms that hold no pivot enter Horner's rule at its innermost level, in the runs of exponent 0, as every other term does, not apart, added once to the top level's last sum, whose inner sums grew past their generators (log 0169-1, 0169-2)
    for run in terms.chunk_by(|&a, &b| alpha(a) == alpha(b)) {
        let a = alpha(run[0]);
        let Some(inner) = horner::<N, F, M, E>(
            field,
            encoding,
            pivots,
            held,
            split,
            run,
            level + 1,
            limit,
            scratch,
            budget,
        )?
        else {
            if let Some(acc) = acc {
                scratch.give(acc);
            }
            return Ok(None);
        };
        acc = Some(match acc {
            None => inner,
            Some(mut sum) => {
                for _ in a + 1..prev {
                    let Some(next) = times::<N, F, M, E>(
                        field, encoding, sum, value, None, limit, scratch, budget,
                    )?
                    else {
                        scratch.give(inner);
                        return Ok(None);
                    };
                    sum = next;
                }
                let Some(next) = times::<N, F, M, E>(
                    field,
                    encoding,
                    sum,
                    value,
                    Some(inner),
                    limit,
                    scratch,
                    budget,
                )?
                else {
                    return Ok(None);
                };
                next
            }
        });
        prev = a;
    }
    #[allow(clippy::expect_used, reason = "`terms` is not empty, so it has a run")]
    let mut acc = acc.expect("`terms` is not empty, so it has a run");
    for _ in 0..prev {
        let Some(next) =
            times::<N, F, M, E>(field, encoding, acc, value, None, limit, scratch, budget)?
        else {
            return Ok(None);
        };
        acc = next;
    }
    Ok(Some(acc))
}

/// For a monomial of N words of the encoding: `acc` times `value`, plus
/// `plus` where it is given, summed in `scratch`'s sum: a product per pair
/// of their terms, its words by `Encoding::times` and its hash the sum of
/// its factors'; empty where the value is 0, a row x + c beside the row 1.
/// None where the sum drained holds `limit` terms or more, and the sum then
/// given back to `scratch` (linear-substitution-time.1). A tick, and per
/// term of `acc` and of `plus` what its terms summed spend
/// (`summed_units`), at `Site::Substitute`, and the drain's. `acc` and
/// `plus` are given back to `scratch`.
#[allow(
    clippy::too_many_arguments,
    reason = "the product's state: the field, the encoding, the factors, the summand, the limit, the buffers, and the budget"
)]
fn times<const N: usize, F: Field, M: Format, E: Encoding<M>>(
    field: &F,
    encoding: &E,
    acc: Sparse<F, E::Word>,
    value: &Sparse<F, E::Word>,
    plus: Option<Sparse<F, E::Word>>,
    limit: usize,
    scratch: &mut Scratch<F, E::Word>,
    budget: &mut Budget,
) -> Result<Option<Sparse<F, E::Word>>, Halt> {
    budget.tick(Site::Substitute)?;
    let w = encoding.width();
    let Scratch { sum, word, .. } = &mut *scratch;
    for k in 0..acc.len() {
        let (at, h, c) = (acc.word::<N>(k, w), acc.hashes[k], acc.coefs[k]);
        for j in 0..value.len() {
            encoding.times::<N>(at, value.word::<N>(j, w), word);
            let coef = field.mul(c, value.coefs[j]);
            sum.add::<N>(field, word, h.wrapping_add(value.hashes[j]), coef, budget)?;
        }
        budget.spend(value.len() * summed_units(w), Site::Substitute)?;
    }
    if let Some(plus) = &plus {
        for k in 0..plus.len() {
            sum.add::<N>(
                field,
                plus.word::<N>(k, w),
                plus.hashes[k],
                plus.coefs[k],
                budget,
            )?;
            if (k + 1).is_multiple_of(RUN) {
                budget.spend(RUN * summed_units(w), Site::Substitute)?;
            }
        }
        budget.spend((plus.len() % RUN) * summed_units(w), Site::Substitute)?;
    }
    let mut out = scratch.take();
    scratch.sum.drain(&mut out, w, budget)?;
    scratch.give(acc);
    if let Some(plus) = plus {
        scratch.give(plus);
    }
    // linear-substitution-time.1: a sum this long ends the generator's substitution.
    if out.len() >= limit {
        scratch.give(out);
        return Ok(None);
    }
    Ok(Some(out))
}
