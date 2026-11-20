//! The monomial table: every monomial of a run's polynomials, or of its
//! cofactors, stored once and named by a `MonoId`, which names it until a
//! compaction renumbers the table (`Table::compact`).

#[cfg(test)]
use std::cell::Cell;
use std::cmp::Ordering;
use std::hint::select_unpredictable;

use rand_core::Rng;

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::columns::Columns;
use crate::id::{Column, MonoId};
use crate::monomial::{ANY, Format, Overflow};
use crate::narrow::{self, Narrow, Size};
use crate::order::Order;

/// The slot of no monomial. No stored slot equals it, as no id reaches
/// `u32::MAX` (`insert`), and no tag reaches its top 32 bits (`TAG`).
pub(crate) const EMPTY_SLOT: u64 = u64::MAX;

// gf2-symbolic-speculation-time.3 design-checked -- a slot's tag is the hash's low 31 bits, so that no tag equals the top 32 bits of the empty slot and the window's selection reads no test for the empty slot, not the low 32 bits beside an empty test per slot of the window, whose masks took prototype W's lookups more instructions than the branch's (log 0178-1)
/// The bits of a hash its slot keeps as the tag: the low 31, so that no
/// tag equals the top 32 bits of `EMPTY_SLOT`.
const TAG: u64 = (1 << 31) - 1;

// gf2-symbolic-speculation-time.2 tuned-checked -- a probe selects among 3 slots from the home slot on before it tests one, not 2, whose selection the build compiled to a branch that mispredicts as the probe loop's does, nor 4, which took more instructions a lookup for fewer mispredictions spared (log 0178-1)
/// The slots from the home slot on among which a probe selects its first
/// candidate (`probe`).
pub(crate) const WINDOW: usize = 3;

#[cfg(test)]
thread_local! {
    // gf2-symbolic-speculation-time.7 design-checked -- in the test profile a thread-local sends every probe of the table to the probe loop alone, so that test code compares a call with the window and without it, not a parameter of the table nor a second probe in the test code (not measured)
    /// The test profile's choice on this thread: whether every probe runs
    /// the probe loop alone, without the window (`probe`).
    pub(crate) static WINDOW_OFF: Cell<bool> = const { Cell::new(false) };
}
// threshold-recheck.14 design-checked -- the slot array starts at 2^10 slots, not 2^9 or 2^11; neither moved the time past the margin on the tuning set (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0074-2)
/// The slot array's capacity at the start, a power of two.
const SLOTS: usize = 1 << 10;

/// The slots per monomial below which the slot array doubles
/// (threshold-recheck.15), and the slots per entry below which the
/// substitution's sums and the names' index double theirs.
pub(crate) const LOAD: usize = 2;

// call-stop-latency.7 design-checked -- a growth of the table places its ids PLACE, 2^14, at a time and spends PROBE units an id after each chunk, not the ids placed at once (log 0138-1)
/// Ids placed, and ids a compaction walks, between two spends of a growth
/// or a compaction.
pub(crate) const PLACE: usize = 1 << 14;

// call-stop-latency.44 tuned-checked -- the slots are appended and emptied FILL, 2^22, at a time, not PLACE, 2^14, at a time, under which the compactions of Geometry.Chou.337_1-p65521 ran slower, as a chunk below glibc's threshold for non-temporal stores goes through the caches (log 0138-2)
/// Slots a growth appends, or a growth or a compaction empties, between
/// two spends.
const FILL: usize = 1 << 22;

// wide-format-products-time.4 tuned-checked -- the supports and their check begin past eight words, where the copies end, not past four or sixteen (log 0132-1, 0132-2)
/// The most words a copy of `row_products_in` is compiled for. A table
/// whose format has more keeps each monomial's support (`Table::support`),
/// and its products are looked up per term by the support check.
pub(crate) const COPIED_WORDS: usize = 8;

// `Table::row_products_as` matches each word count from 1 to `COPIED_WORDS`
// to its copy of `row_products_in`, so a change of the constant changes the
// match.
const _: () = assert!(
    COPIED_WORDS == 8,
    "row_products_as holds a copy for each word count from 1 to COPIED_WORDS"
);

// call-stop-latency.6 design-checked -- the table's limit gains Stop, and every insertion returns the limit, so that a growth that stops returns at once, not Halt returned by the table nor a stop the table keeps for its caller's next check (not measured)
/// Why the table names no monomial for a product or for exponents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Limit {
    /// Its degree passes the format, or a pair's sugar passes `u32::MAX`
    /// (`pairs::update`).
    Overflow,
    /// The table holds as many monomials as a u32 id numbers.
    Size,
    /// A bound stopped the call while the insertion grew the table, which
    /// then holds the monomial, its slot array part placed: its lookups
    /// may miss what it holds, and the stopped call drops it.
    Stop(Stop),
}

impl From<Overflow> for Limit {
    fn from(_: Overflow) -> Limit {
        Limit::Overflow
    }
}

impl From<Size> for Limit {
    fn from(_: Size) -> Limit {
        Limit::Size
    }
}

impl From<Stop> for Limit {
    fn from(stop: Stop) -> Limit {
        Limit::Stop(stop)
    }
}

// f4-gfp.14 cited-checked: monagan-2015 §2.1 -- once created, a monomial persists over the whole computation; garbage collecting them did not improve the timings; the idea attributed to Magma [16], a private communication
/// The monomials of one run, or of its cofactors, in parallel arrays
/// indexed by the id. An id names its monomial until a compaction
/// renumbers the table (`compact`), which only the plain run's rounds call.
pub(crate) struct Table<M: Format> {
    format: M,
    /// The hash multipliers, one per variable.
    multipliers: Vec<u64>,
    /// The exponent words, `format.words()` per monomial, back to back.
    words: Vec<M::Word>,
    degree: Vec<u32>,
    hash: Vec<u64>,
    mask: Vec<u32>,
    // wide-format-products-time.2 cited-checked: bigatti-1997 §7.2.1 -- when the indeterminates are many, power-products hold several zero exponents, and a bit-support of each, compared by bitwise operations, often settles a test without the exponents
    // wide-format-products-time.6 design-checked -- a table of eight words or fewer keeps no supports, 8 bytes a monomial no lookup of it reads (log 0132-2)
    // wide-format-products-time.11 cited-checked: demin-2024 §4.1 -- sparse monomial representations were not effective for problems with close to a hundred variables due to costly sparse monomial arithmetic; when the number of variables is large, exponent vectors are used
    /// Per monomial its support: bit k set where group k of its words holds
    /// a nonzero word, `group` words to a group. Empty where the format has
    /// `COPIED_WORDS` words or fewer.
    support: Vec<u64>,
    // wide-format-products-time.5 design-checked -- a support bit stands for a group of ⌈words/64⌉ consecutive words, so that one u64 covers any format, not a bit per word with formats past 64 words left to the full compare, nor a list of nonzero word indices per monomial (log 0132-2)
    /// The words of a support's group, ⌈words/64⌉, at least 1; the last
    /// group may be shorter.
    group: usize,
    // f4-gfp.19 design-checked -- a monomial's column in the current matrix is a field of its table entry, not a map from monomial to column built per matrix (not measured)
    /// The round and the column slot of each monomial (`Seen`).
    seen: Vec<Seen>,
    /// The round of the last matrix a build opened in the table, 0 for
    /// none (`open`).
    round: u32,
    /// The ids by slot, each below its tag, the hash's low 31 bits (`TAG`):
    /// `tag << 32 | id`, `EMPTY_SLOT` where none.
    slots: Vec<u64>,
    /// 64 minus log2 of the slot count: the hash shifted right by it is a
    /// slot.
    shift: u32,
    /// Exponent words of a monomial being looked up.
    scratch: Vec<M::Word>,
    /// Exponent words of a quotient a test reads (`form_divided`).
    divided: Vec<M::Word>,
}

impl<M: Format> Table<M> {
    // f4-gfp.15 cited-checked: monagan-2015 §2.1 -- h = Σ r_i e_i with random integers chosen at the outset; adding hashes made monomial multiplication 20% faster; the idea attributed to Faugère and Magma [16], a private communication
    /// An empty table whose hash multipliers are drawn from `rng`.
    pub(crate) fn new(format: M, rng: &mut impl Rng) -> Table<M> {
        let multipliers = multipliers(format.variables(), rng);
        Table {
            format,
            multipliers,
            words: Vec::new(),
            degree: Vec::new(),
            hash: Vec::new(),
            mask: Vec::new(),
            support: Vec::new(),
            group: format.words().div_ceil(u64::BITS as usize).max(1),
            seen: Vec::new(),
            round: 0,
            slots: vec![EMPTY_SLOT; SLOTS],
            shift: 64 - SLOTS.trailing_zeros(),
            scratch: vec![M::Word::default(); format.words()],
            divided: vec![M::Word::default(); format.words()],
        }
    }

    pub(crate) fn format(&self) -> &M {
        &self.format
    }

    /// The number of monomials.
    pub(crate) fn len(&self) -> usize {
        self.degree.len()
    }

    /// Whether the table keeps its monomials' supports: where its format has
    /// more than `COPIED_WORDS` words.
    fn supports(&self) -> bool {
        self.format.words() > COPIED_WORDS
    }

    // table-read-time.1 design-checked -- the table's read accessors, and the writes of a column and a stamp, index their arrays with no bounds check outside the test profile, an id the table interned as their guarantee, not a bounds check per read (log 0045-1)
    // table-read-time.2 practice-checked: safe abstraction over unsafe code (Rust) -- the accessors stay safe functions whose bodies read unchecked, the table's invariant holding the id in range, not unsafe functions whose callers each guarantee the id (not measured)
    // table-read-time.3 practice-checked: debug assertions (debug_assert!) -- the test profile checks each unchecked read's index by a debug assertion, not one assert per call in every profile (log 0031-1)
    /// The words of `id`. Each accessor of an id reads with no bounds check
    /// outside the test profile, whose debug assertions check it: an id
    /// names a monomial the table holds, and the run that compacts the
    /// table gives every id it holds its new id (`compact`).
    pub(crate) fn monomial(&self, id: MonoId) -> &[M::Word] {
        let w = self.format.words();
        // SAFETY: an id of the table lies below len(), and `words` holds w
        // words per monomial.
        unsafe { at(&self.words, id.index() * w, w) }
    }

    pub(crate) fn exponent(&self, id: MonoId, i: usize) -> u32 {
        debug_assert!(
            i < self.format.variables(),
            "variable {i} of {n}",
            n = self.format.variables()
        );
        self.format.exponent(self.monomial(id), i)
    }

    /// The exponents of `id`, one per variable, to `out`, whose length is the
    /// number of variables.
    pub(crate) fn unpack(&self, id: MonoId, out: &mut [u32]) {
        self.format.unpack(self.monomial(id), out);
    }

    /// Appends the exponents of `id`, one per variable, to `out`.
    pub(crate) fn exponents(&self, id: MonoId, out: &mut Vec<u32>) {
        let at = out.len();
        out.resize(at + self.format.variables(), 0);
        self.unpack(id, &mut out[at..]);
    }

    pub(crate) fn degree(&self, id: MonoId) -> u32 {
        // SAFETY: an id of the table lies below len(), the length of
        // `degree`.
        unsafe { *entry(&self.degree, id, "degree") }
    }

    /// The hash multiplier of variable i: a monomial's hash is the sum of
    /// its exponents times their variables' multipliers (f4-gfp.15).
    pub(crate) fn multiplier(&self, i: usize) -> u64 {
        self.multipliers[i]
    }

    pub(crate) fn hash(&self, id: MonoId) -> u64 {
        // SAFETY: an id of the table lies below len(), the length of
        // `hash`.
        unsafe { *entry(&self.hash, id, "hash") }
    }

    pub(crate) fn mask(&self, id: MonoId) -> u32 {
        // SAFETY: an id of the table lies below len(), the length of
        // `mask`.
        unsafe { *entry(&self.mask, id, "mask") }
    }

    pub(crate) fn column(&self, id: MonoId) -> MonoId {
        // SAFETY: an id of the table lies below len(), the length of
        // `seen`.
        unsafe { entry(&self.seen, id, "column").column }
    }

    pub(crate) fn set_column(&mut self, id: MonoId, column: MonoId) {
        // SAFETY: an id of the table lies below len(), the length of
        // `seen`.
        unsafe { entry_mut(&mut self.seen, id, "column").column = column };
    }

    #[cfg(test)]
    pub(crate) fn stamp(&self, id: MonoId) -> u32 {
        // SAFETY: an id of the table lies below len(), the length of
        // `seen`.
        unsafe { entry(&self.seen, id, "stamp").stamp }
    }

    #[cfg(test)]
    pub(crate) fn set_stamp(&mut self, id: MonoId, round: u32) {
        // SAFETY: an id of the table lies below len(), the length of
        // `seen`.
        unsafe { entry_mut(&mut self.seen, id, "stamp").stamp = round };
    }

    // column-index-peak.19 design-checked -- the table numbers the rounds of its matrices itself, each build opening the one past the last, not the caller's count of its rounds, whose distinctness an assertion at each build would have to guard (not measured)
    /// Opens the next round of the table, the round of the matrix a build
    /// makes: one past the last a build opened, whatever the run counts
    /// its rounds by. No monomial bears its stamp before the build enters
    /// it (`enter_seen`). Size where the rounds would pass what a u32
    /// numbers.
    pub(crate) fn open(&mut self) -> Result<u32, Size> {
        self.round = narrow::id(self.round as usize + 1)?;
        Ok(self.round)
    }

    /// Enters `id` into the matrix of round `round`, with `fresh` and
    /// `previous` (`enter_seen`): its local column.
    pub(crate) fn enter(
        &mut self,
        id: MonoId,
        round: u32,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
    ) -> MonoId {
        // SAFETY: an id of the table lies below len(), the length of
        // `seen`.
        let seen = unsafe { entry_mut(&mut self.seen, id, "record") };
        enter_seen(seen, id, round, fresh, previous)
    }

    /// The entries the growable arrays hold, which `shrink_to_fit` may copy.
    pub(crate) fn held(&self) -> usize {
        self.words.len()
            + self.degree.len()
            + self.hash.len()
            + self.mask.len()
            + self.support.len()
            + self.seen.len()
    }

    /// Gives back the slack of the growable arrays. The slot array stays.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.words.shrink_to_fit();
        self.degree.shrink_to_fit();
        self.hash.shrink_to_fit();
        self.mask.shrink_to_fit();
        self.support.shrink_to_fit();
        self.seen.shrink_to_fit();
    }

    // call-stop-latency.4 design-checked -- every function that does work that grows with the input takes the call's budget, the table's insertions and compaction among them, not a second handle on the bounds held by the table nor the growth taken out of the insertions to points the run calls (not measured)
    /// The id of the monomial with these exponents, one per variable. An
    /// insertion that grows the table spends the growth on `budget`
    /// (`grow`), as every insertion below does.
    pub(crate) fn intern(
        &mut self,
        exponents: &[u32],
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        let degree = self.format.encode(exponents, &mut self.scratch)?;
        self.intern_scratch(degree, budget)
    }

    // gf2-input-time.1 design-checked -- a monomial given by the exponents of the variables a parsed term names is hashed and compared over those variables alone, and encoded only where it is new, not encoded, hashed, and compared over every variable of the format (log 0158-1, 0158-3)
    /// The id of the monomial whose exponents `exponents` holds, one per
    /// variable, every nonzero one at a variable of `touched`, which names
    /// each once: its degree and hash summed over `touched`, a candidate
    /// taken where its degree and its exponents there agree, which leaves
    /// its other exponents 0, and the exponents of `touched` encoded only
    /// where the monomial is new (`Format::encode_touched`). Overflow where
    /// the degree passes the format.
    pub(crate) fn intern_touched(
        &mut self,
        exponents: &[u32],
        touched: &[usize],
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        debug_assert_eq!(
            exponents.len(),
            self.format.variables(),
            "an exponent per variable"
        );
        let degree: u64 = touched.iter().map(|&i| u64::from(exponents[i])).sum();
        if degree > u64::from(M::MAX_DEGREE) {
            return Err(Limit::Overflow);
        }
        let degree: u32 = degree.narrow();
        let hash = sum_hash(
            &self.multipliers,
            touched.iter().map(|&i| (i, exponents[i])),
        );
        let w = self.format.words();
        let format = self.format;
        let degrees = &self.degree;
        let found = self.find(hash, |words, c| {
            let at = c.index() * w;
            degrees[c.index()] == degree
                && touched
                    .iter()
                    .all(|&i| format.exponent(&words[at..at + w], i) == exponents[i])
        });
        match found {
            Ok(id) => Ok(id),
            Err(slot) => {
                self.format
                    .encode_touched(exponents, touched, &mut self.scratch)?;
                self.insert(slot, hash, degree, budget)
            }
        }
    }

    // product-lookup-time.4 design-checked -- every caller but `matrix::multiply` takes this safe door, one assert on both ids and then the unchecked lookup, not `mul` itself an unsafe fn whose callers each guarantee their ids (not measured)
    /// The id of a·b, or the limit it reaches. Panics where a or b is no id
    /// of the table.
    pub(crate) fn mul(
        &mut self,
        a: MonoId,
        b: MonoId,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        // It holds: a caller's ids come from this table, and the cofactor
        // tracking interns a run's multiplier into its own table first.
        assert!(
            a.max(b).index() < self.len(),
            "ids {a} and {b} past the table's {len} monomials",
            len = self.len()
        );
        // SAFETY: both ids lie below len(), asserted above.
        unsafe { self.mul_unchecked(a, b, budget) }
    }

    // product-lookup-time.1 design-checked -- the lookup reads the factors' degrees, hashes, and words with no bounds check per read, the caller's guarantee in its place (log 0029-3, 0031-1)
    // product-lookup-time.7 cited-checked: drepper-2007 §6.2.2 -- inlining lets the compiler optimize larger chunks of code at a time, and code that a lopsided condition rarely runs is better moved out of the main code path, where it neither pollutes the instruction cache nor disturbs prefetching
    /// The id of a·b, or the limit it reaches. Its reads have no bounds
    /// check outside the test profile, whose debug assertions check them.
    ///
    /// # Safety
    ///
    /// a and b are ids of this table: both lie below `len()`.
    #[inline]
    pub(crate) unsafe fn mul_unchecked(
        &mut self,
        a: MonoId,
        b: MonoId,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        debug_assert!(
            a.max(b).index() < self.len(),
            "ids {a} and {b} past the table's {len} monomials",
            len = self.len()
        );
        let (a, b) = (a.index(), b.index());
        // SAFETY: a and b lie below len(), the length of `degree` and of
        // `hash`, by the caller's guarantee.
        let (da, db, ha, hb) = unsafe {
            (
                *self.degree.get_unchecked(a),
                *self.degree.get_unchecked(b),
                *self.hash.get_unchecked(a),
                *self.hash.get_unchecked(b),
            )
        };
        let degree = da + db;
        if degree > M::MAX_DEGREE {
            return Err(Limit::Overflow);
        }
        let hash = ha.wrapping_add(hb);
        let w = self.format.words();
        let (sa, sb) = (a * w, b * w);
        // wide-format-products-time.10 design-checked -- the check is chosen per table by its format's words, not per lookup by the joint support's bits against the word count, which the baseline x86-64 build counts without `popcnt` (log 0132-1, 0132-2)
        let found = if self.supports() {
            debug_assert_eq!(self.support.len(), self.len(), "a support per monomial");
            // SAFETY: a and b lie below len(), the length of `support` in a
            // table that keeps supports, by the caller's guarantee.
            let joint = unsafe { *self.support.get_unchecked(a) | *self.support.get_unchecked(b) };
            self.find(hash, |words, c| {
                // SAFETY: a and b lie below len() by the caller's guarantee,
                // and c is an id from a slot, below len() (`find`).
                unsafe { self.is_product_on(words, (sa, sb), c, joint, degree) }
            })
        } else {
            self.find(hash, |words, c| {
                // SAFETY: a and b lie below len() by the caller's guarantee,
                // and c is an id from a slot, below len() (`find`).
                let (a, b, c) = unsafe {
                    (
                        at(words, sa, w),
                        at(words, sb, w),
                        at(words, c.index() * w, w),
                    )
                };
                self.format.is_product(a, b, c)
            })
        };
        match found {
            Ok(id) => Ok(id),
            Err(slot) => self.mul_miss(slot, hash, (sa, da), (sb, db), budget),
        }
    }

    // wide-format-products-time.3 invented-checked -- a candidate c for a·b is accepted where its degree is the factors' sum and its words agree with a·b's on the groups the joint support marks, the words outside zero by the degrees, not by `is_product` over every word (log 0132-1, 0132-2)
    /// Whether c, the monomial of id `c`, is the product of the monomials
    /// whose words start at the offsets `sa` and `sb` in `words`, whose
    /// degrees sum to `degree` and whose supports join to `joint`. Outside
    /// the groups `joint` marks both factors hold zero words. Where c's
    /// degree is `degree` and its words on those groups are the product's,
    /// its lanes there sum to `degree`, so its lanes outside sum to 0 and,
    /// none being negative, are 0: c is the product (`Format`).
    ///
    /// # Safety
    ///
    /// `sa` and `sb` start the words of ids of this table, and c is an id
    /// of this table, which keeps supports.
    #[inline(always)]
    unsafe fn is_product_on(
        &self,
        words: &[M::Word],
        (sa, sb): (usize, usize),
        c: MonoId,
        joint: u64,
        degree: u32,
    ) -> bool {
        let (w, group) = (self.format.words(), self.group);
        let sc = c.index() * w;
        debug_assert!(
            self.supports() && sa.max(sb).max(sc) + w <= words.len(),
            "words {sa}, {sb}, and {sc} of {w} in a table of {len} words",
            len = words.len()
        );
        // SAFETY: sa, sb, and sc start the w words of ids of this table, by
        // the caller's guarantee.
        let full = || unsafe {
            self.format
                .is_product(at(words, sa, w), at(words, sb, w), at(words, sc, w))
        };
        // SAFETY: c lies below len(), the length of `degree`, by the
        // caller's guarantee.
        if unsafe { *entry(&self.degree, c, "degree") } != degree {
            debug_assert!(!full(), "the degree refused the product, id {c}");
            return false;
        }
        let mut groups = joint;
        while groups != 0 {
            let k = groups.trailing_zeros() as usize;
            groups &= groups - 1;
            for i in k * group..((k + 1) * group).min(w) {
                // SAFETY: a group of a support lies below the w words, and
                // sa, sb, and sc start the w words of ids of this table.
                let (of_a, of_b, of_c) = unsafe {
                    (
                        *words.get_unchecked(sa + i),
                        *words.get_unchecked(sb + i),
                        *words.get_unchecked(sc + i),
                    )
                };
                if M::mul_word(of_a, of_b) != of_c {
                    debug_assert!(!full(), "the support check refused the product, id {c}");
                    return false;
                }
            }
        }
        debug_assert!(full(), "the support check accepted id {c}, no product");
        true
    }

    // wide-format-products-time.8 design-checked -- a miss forms the product over every word, `Format::mul`, not the scratch zeroed and the joint support's words summed, as a miss comes once per new monomial (log 0132-1, 0132-2)
    /// The miss of `mul_unchecked` and of `row_products_in`: forms the
    /// product of the factors whose words start at the offsets given, of
    /// the degrees given, and stores it at the empty slot `slot`. In
    /// `mul_unchecked` only a miss computes the exponents.
    #[inline(never)]
    fn mul_miss(
        &mut self,
        slot: usize,
        hash: u64,
        (sa, da): (usize, u32),
        (sb, db): (usize, u32),
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        let w = self.format.words();
        self.format.mul(
            &self.words[sa..sa + w],
            da,
            &self.words[sb..sb + w],
            db,
            &mut self.scratch,
        )?;
        self.insert(slot, hash, da + db, budget)
    }

    // matrix-build-time.1 practice-checked: loop-invariant code motion (hoisting) -- a row's products are looked up by one method of the table, which reads the multiplier's degree, hash, and words once per row, not `mul_unchecked` per term, which reads them at every term (log 0037-1)
    /// The products of one row: t·u for each u of `monomials`, in order,
    /// each entered into the matrix of round `round` (`enter_seen`, with
    /// `fresh` and `previous`), and its local column appended to `cols`
    /// at the vector's width. Where `cols` is narrow, the caller keeps the
    /// round's local columns below 2^16, which the test profile asserts at
    /// each entry (`Column::of`). The limit where a product reaches one:
    /// `cols`, `fresh`, and `previous` then hold what the products before
    /// it gave, and the table stores nothing for it. The ids are those
    /// `mul` gives. Where t is the unit monomial, each product is u itself,
    /// entered with no lookup (`unit_row`).
    ///
    /// # Safety
    ///
    /// t and every u of `monomials` are ids of this table.
    #[allow(
        clippy::too_many_arguments,
        reason = "the row's multiplier and monomials, its round, the three vectors it fills, and the budget its growth spends on, none of which belongs with another"
    )]
    pub(crate) unsafe fn row_products(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Columns,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Limit> {
        // SAFETY: the caller's guarantee, and each copy writes at the
        // vector's width.
        unsafe {
            if cols.is_narrow() {
                self.row_products_as::<u16>(t, monomials, round, cols, fresh, previous, budget)
            } else {
                self.row_products_as::<MonoId>(t, monomials, round, cols, fresh, previous, budget)
            }
        }
    }

    /// `row_products` for columns of width `I`, the width of `cols`.
    ///
    /// # Safety
    ///
    /// As `row_products`, and `I` is the width of `cols`.
    #[inline(always)]
    #[allow(
        clippy::too_many_arguments,
        reason = "the row's multiplier and monomials, its round, the three vectors it fills, and the budget its growth spends on, none of which belongs with another"
    )]
    unsafe fn row_products_as<I: Column>(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Columns,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Limit> {
        // gf2-build-rest-time.1 design-checked -- a row whose multiplier is the unit monomial takes each of its monomials as its product, entered into the round with no lookup, not the lookup per term, which hashes, forms, probes, and compares a product equal to its factor (log 0161-1, 0161-2)
        if self.degree(t) == 0 {
            // SAFETY: the caller's guarantee, and `I` is the width of
            // `cols`.
            unsafe { self.unit_row::<I>(monomials, round, cols, fresh, previous) };
            return Ok(());
        }
        // SAFETY: the caller's guarantee.
        unsafe {
            self.row_products_by_lookup::<I>(t, monomials, round, cols, fresh, previous, budget)
        }
    }

    /// `row_products_as` where t is not the unit monomial: each product
    /// looked up, by the copy for the format's word count.
    ///
    /// # Safety
    ///
    /// As `row_products_as`.
    // gf2-build-rest-time.3 design-checked -- the lookups' dispatch by the format's word count stands in a function of its own under #[inline(never)], not inlined into row_products_as beside its test of the unit monomial, where the lookup loop ran more instructions a term on every row (log 0161-2)
    #[inline(never)]
    #[allow(
        clippy::too_many_arguments,
        reason = "the row's multiplier and monomials, its round, the three vectors it fills, and the budget its growth spends on, none of which belongs with another"
    )]
    unsafe fn row_products_by_lookup<I: Column>(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Columns,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Limit> {
        // matrix-build-time.2 cited-checked: demin-2024 §4.1 -- the monomial representation is selected at the start of the computation, and the specialized functions that implement monomial arithmetic are then just-in-time compiled and inlined, which relocates the work from running time to compile time
        // SAFETY: the caller's guarantee, and each copy runs on the format
        // of its word count.
        unsafe {
            match self.format.words() {
                1 => {
                    self.row_products_in::<1, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                2 => {
                    self.row_products_in::<2, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                3 => {
                    self.row_products_in::<3, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                4 => {
                    self.row_products_in::<4, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                // wide-format-products-time.1 tuned-checked -- a copy for every word count from one to eight, not one to four with the lookup per term past four (log 0132-1, 0132-2)
                5 => {
                    self.row_products_in::<5, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                6 => {
                    self.row_products_in::<6, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                7 => {
                    self.row_products_in::<7, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                8 => {
                    self.row_products_in::<8, I>(t, monomials, round, cols, fresh, previous, budget)
                }
                _ => self
                    .row_products_per_term::<I>(t, monomials, round, cols, fresh, previous, budget),
            }
        }
    }

    /// `row_products` where the multiplier is the unit monomial, into
    /// columns of width `I`: each u of `monomials` entered into the round
    /// as it stands (`enter`), its local column appended to `cols`.
    ///
    /// # Safety
    ///
    /// Every u of `monomials` is an id of this table, and `I` is the width
    /// of `cols`.
    unsafe fn unit_row<I: Column>(
        &mut self,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Columns,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
    ) {
        cols.reserve(monomials.len());
        let out = cols.as_mut_ptr::<I>();
        let mut len = cols.len();
        for &u in monomials {
            let column = self.enter(u, round, fresh, previous);
            // SAFETY: `len` lies below the length before the row plus the
            // row's terms, which `reserve` made room for, and `I` is the
            // vector's width.
            unsafe { out.add(len).write(I::of(column.index())) };
            len += 1;
        }
        // SAFETY: the entries before `len` are written, and `len` lies
        // within the capacity reserved.
        unsafe { cols.set_len(len) };
    }

    // column-index-peak.20 design-checked -- the lookup per term writes its row through the column vector's writer, the row reserved once, not a push per term at the vector's width, which would be a second writer of the entries beside the copies' (not measured)
    /// `row_products` by `mul_unchecked` per term.
    ///
    /// # Safety
    ///
    /// As `row_products_as`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the row's multiplier and monomials, its round, the three vectors it fills, and the budget its growth spends on, none of which belongs with another"
    )]
    unsafe fn row_products_per_term<I: Column>(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Columns,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Limit> {
        cols.reserve(monomials.len());
        let out = cols.as_mut_ptr::<I>();
        let mut len = cols.len();
        for &u in monomials {
            // SAFETY: the caller's guarantee.
            let id = match unsafe { self.mul_unchecked(t, u, budget) } {
                Ok(id) => id,
                Err(limit) => {
                    // SAFETY: the entries before `len` are written, and
                    // `len` lies within the capacity reserved.
                    unsafe { cols.set_len(len) };
                    return Err(limit);
                }
            };
            let column = self.enter(id, round, fresh, previous);
            // SAFETY: `len` lies below the length before the row plus the
            // row's terms, which `reserve` made room for, and `I` is the
            // vector's width.
            unsafe { out.add(len).write(I::of(column.index())) };
            len += 1;
        }
        // SAFETY: as at the limit above.
        unsafe { cols.set_len(len) };
        Ok(())
    }

    /// `row_products` for a format of W words and columns of width `I`.
    ///
    /// # Safety
    ///
    /// As `row_products_as`, and the format has W words.
    #[inline(always)]
    #[allow(
        clippy::too_many_arguments,
        reason = "the row's multiplier and monomials, its round, the three vectors it fills, and the budget its growth spends on, none of which belongs with another"
    )]
    unsafe fn row_products_in<const W: usize, I: Column>(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Columns,
        fresh: &mut Vec<MonoId>,
        previous: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Limit> {
        debug_assert_eq!(self.format.words(), W);
        // matrix-build-time.5 practice-checked: loop-invariant code motion -- the table's arrays are read through pointers taken before the row's first term and taken again after each miss, not each read through the table's vectors by get_unchecked
        let mut a = self.arrays();
        let t = t.index();
        debug_assert!(
            t < a.len,
            "id {t} past the table's {len} monomials",
            len = a.len
        );
        // SAFETY: t lies below len() by the caller's guarantee, and its W
        // words below the words' length, W·len().
        let (dt, ht, tw) = unsafe {
            (
                *a.degree.add(t),
                *a.hash.add(t),
                words_of::<_, W>(a.words, t),
            )
        };
        // matrix-build-time.6 practice-checked: preallocation (reserve capacity) -- the row's columns are reserved once, before its first term, and each entry written without a capacity check, not pushed per term (log 0037-1)
        cols.reserve(monomials.len());
        let out = cols.as_mut_ptr::<I>();
        let mut len = cols.len();
        for &u in monomials {
            let u = u.index();
            debug_assert!(
                u < a.len,
                "id {u} past the table's {len} monomials",
                len = a.len
            );
            // SAFETY: u lies below len() by the caller's guarantee.
            let (du, hu) = unsafe { (*a.degree.add(u), *a.hash.add(u)) };
            if dt + du > M::MAX_DEGREE {
                // SAFETY: the ids before `len` are written, and `len` lies
                // within the capacity reserved.
                unsafe { cols.set_len(len) };
                return Err(Limit::Overflow);
            }
            let hash = ht.wrapping_add(hu);
            // matrix-build-time.4 practice-checked: loop-invariant code motion (hoisting) -- the product's words are formed once per term, each the sum of the factors' words, and each candidate's words compared with them, not `is_product` per candidate, which sums the factors' words again; bachmann-1998 §3, Lemma 1 (3) and `MonAdd`, forms a product's words as the word-wise sum and is silent on where a lookup forms the product (log 0037-1)
            // SAFETY: u lies below len(), so its W words below W·len().
            let uw = unsafe { words_of::<_, W>(a.words, u) };
            let product: [M::Word; W] = std::array::from_fn(|k| M::mul_word(tw[k], uw[k]));
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the shift keeps the slot below the table's size"
            )]
            let home = (hash >> a.shift) as usize;
            let same = |c: MonoId| {
                debug_assert!(
                    c.index() < a.len,
                    "a slot holds id {c} past the table's {len} monomials",
                    len = a.len
                );
                // SAFETY: a slot that is not EMPTY_SLOT holds an id below
                // len(), whose W words lie below W·len().
                let words = unsafe { words_of::<_, W>(a.words, c.index()) };
                words == product
            };
            // SAFETY: the slot array holds `last + 1` slots, a power of two,
            // the hash shifted right by `shift` lies below the count, and
            // every slot holds EMPTY_SLOT or an id below len() under its tag.
            let found = unsafe { probe(a.slots, a.last, home, tag_of(hash), same) };
            let id = match found {
                Ok(id) => id,
                Err(slot) => {
                    let id = match self.mul_miss(slot, hash, (t * W, dt), (u * W, du), budget) {
                        Ok(id) => id,
                        Err(limit) => {
                            // SAFETY: as at the degree test above.
                            unsafe { cols.set_len(len) };
                            return Err(limit);
                        }
                    };
                    // The miss may have grown every array.
                    a = self.arrays();
                    id
                }
            };
            // SAFETY: id lies below len(), the records' length, and no other
            // reference to the records is alive.
            let seen = unsafe { &mut *a.seen.add(id.index()) };
            let column = enter_seen(seen, id, round, fresh, previous);
            // SAFETY: `len` lies below the length before the row plus the
            // row's terms, which `reserve` made room for, and `I` is the
            // vector's width.
            unsafe { out.add(len).write(I::of(column.index())) };
            len += 1;
        }
        // SAFETY: as at the degree test above.
        unsafe { cols.set_len(len) };
        Ok(())
    }

    /// Pointers to the arrays `row_products_in` reads, valid until the
    /// table next stores a monomial.
    #[inline(always)]
    fn arrays(&mut self) -> Arrays<M::Word> {
        Arrays {
            degree: self.degree.as_ptr(),
            hash: self.hash.as_ptr(),
            words: self.words.as_ptr(),
            slots: self.slots.as_ptr(),
            seen: self.seen.as_mut_ptr(),
            last: self.slots.len() - 1,
            shift: self.shift,
            len: self.degree.len(),
        }
    }

    /// The id of a / b, where b divides a.
    pub(crate) fn div(
        &mut self,
        a: MonoId,
        b: MonoId,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        debug_assert!(self.divides(b, a));
        let degree = self.degree(a) - self.degree(b);
        let hash = self.hash(a).wrapping_sub(self.hash(b));
        let w = self.format.words();
        let (sa, sb) = (a.index() * w, b.index() * w);
        // c = a / b exactly where c·b = a.
        let found = self.find(hash, |words, c| {
            let c = c.index() * w;
            let (a, b, c) = (&words[sa..sa + w], &words[sb..sb + w], &words[c..c + w]);
            self.format.is_product(c, b, a)
        });
        match found {
            Ok(id) => Ok(id),
            Err(slot) => {
                self.format.div(
                    &self.words[sa..sa + w],
                    &self.words[sb..sb + w],
                    &mut self.scratch,
                );
                self.insert(slot, hash, degree, budget)
            }
        }
    }

    /// The id of lcm(a, b), or the limit it reaches.
    #[cfg(test)]
    pub(crate) fn lcm(
        &mut self,
        a: MonoId,
        b: MonoId,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        let w = self.format.words();
        let (sa, sb) = (a.index() * w, b.index() * w);
        let degree = self.format.lcm(
            &self.words[sa..sa + w],
            &self.words[sb..sb + w],
            &mut self.scratch,
        )?;
        self.intern_scratch(degree, budget)
    }

    /// The id of the monomial of these words, of degree `degree`, interned.
    pub(crate) fn intern_words(
        &mut self,
        words: &[M::Word],
        degree: u32,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        self.scratch.copy_from_slice(words);
        self.intern_scratch(degree, budget)
    }

    /// The id of the monomial of these words, of degree `degree` and of
    /// `hash`, its hash in this table, interned.
    pub(crate) fn intern_hashed(
        &mut self,
        words: &[M::Word],
        degree: u32,
        hash: u64,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        self.scratch.copy_from_slice(words);
        debug_assert_eq!(hash, self.scratch_hash(), "the hash of the words");
        self.intern_scratch_hashed(degree, hash, budget)
    }

    // table-read-time.9 design-checked -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    /// Forms s·lcm(x, y)/x, of the monomials of these ids, in the scratch
    /// words: its degree, or Overflow where it passes the format.
    pub(crate) fn form_quotient(
        &mut self,
        s: MonoId,
        x: MonoId,
        y: MonoId,
    ) -> Result<u32, Overflow> {
        let w = self.format.words();
        let (ss, sx, sy) = (s.index() * w, x.index() * w, y.index() * w);
        self.format.mul_quotient(
            &self.words[ss..ss + w],
            &self.words[sx..sx + w],
            &self.words[sy..sy + w],
            &mut self.scratch,
        )
    }

    /// The monomial formed last, in the scratch words. Any other operation
    /// of the table may overwrite it.
    pub(crate) fn formed(&self) -> &[M::Word] {
        &self.scratch
    }

    // bivium-gap.3 design-checked -- the quotient by an input's leading monomial is formed in words of its own, which `divided` reads and only this writes, not interned nor formed in the scratch words, which hold the formed signature it is taken of (log 0163-1, 0163-2)
    /// Forms m/d in the quotient's words, `divided`: m the monomial of that
    /// id, or the monomial formed last (`formed`) where it is None, and d a
    /// divisor of it.
    pub(crate) fn form_divided(&mut self, m: Option<MonoId>, d: MonoId) {
        let w = self.format.words();
        let sd = d.index() * w;
        let by = &self.words[sd..sd + w];
        let of = match m {
            Some(m) => {
                let sm = m.index() * w;
                &self.words[sm..sm + w]
            }
            None => &self.scratch[..],
        };
        debug_assert!(
            self.format.divides(by, of),
            "the divisor divides the monomial"
        );
        self.format.div(of, by, &mut self.divided);
    }

    /// The quotient `form_divided` formed last.
    pub(crate) fn divided(&self) -> &[M::Word] {
        &self.divided
    }

    /// The id of the monomial formed last, of degree `degree`, interned.
    pub(crate) fn intern_formed(
        &mut self,
        degree: u32,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        self.intern_scratch(degree, budget)
    }

    // f4-gfp.17 cited-checked: roune-2012 §4.2 -- the divmasks eliminate around 98% of all divisibility checks when using the monomial list; berthomieu-2021 §4.1 (2) -- a 32-bit divisor mask
    /// Whether a divides b, the masks compared before the exponents.
    pub(crate) fn divides(&self, a: MonoId, b: MonoId) -> bool {
        self.mask(a) & !self.mask(b) == 0 && self.format.divides(self.monomial(a), self.monomial(b))
    }

    /// The degree of lcm(a, b), which may pass the format.
    #[cfg(test)]
    pub(crate) fn lcm_degree(&self, a: MonoId, b: MonoId) -> u32 {
        self.format.lcm_degree(self.monomial(a), self.monomial(b))
    }

    /// Whether a and b share no variable (`coprime_masked`).
    pub(crate) fn coprime(&self, a: MonoId, b: MonoId) -> bool {
        coprime_masked::<ANY, M>(
            &self.format,
            self.mask(a),
            self.monomial(a),
            self.mask(b),
            self.monomial(b),
        )
    }

    pub(crate) fn cmp<O: Order<M>>(&self, order: &O, a: MonoId, b: MonoId) -> Ordering {
        order.cmp(
            &self.format,
            self.monomial(a),
            self.degree(a),
            self.monomial(b),
            self.degree(b),
        )
    }

    /// `order`'s key of the monomial `id` (`Order::key`).
    pub(crate) fn key<O: Order<M>>(&self, order: &O, id: MonoId) -> u128 {
        order.key(&self.format, self.monomial(id), self.degree(id))
    }

    /// `order`'s tie key of the monomial `id` over its first `below`
    /// variables, or over the variables the key does not hold where `below`
    /// is None (`Order::tie_key`).
    pub(crate) fn tie_key<O: Order<M>>(&self, order: &O, id: MonoId, below: Option<usize>) -> u128 {
        order.tie_key(&self.format, self.monomial(id), below)
    }

    // bivium-signature-gap.3 design-checked -- the hash of the scratch words sums the multipliers times the exponents of the variables of the words that hold one, each word's lanes by `Format::each_variable`, not every variable's exponent read through `Format::exponent`, which finds its word and lane again (log 0183-1)
    /// The hash of the exponents in `scratch`, as `sum_hash` sums it.
    fn scratch_hash(&self) -> u64 {
        let mut hash = 0u64;
        for (j, &word) in self.scratch.iter().enumerate() {
            if word == M::Word::default() {
                continue;
            }
            self.format.each_variable(j, word, |i, e| {
                hash = hash_term(hash, &self.multipliers, i, e);
            });
        }
        hash
    }

    /// The id of the monomial in `scratch`, of degree `degree`.
    fn intern_scratch(&mut self, degree: u32, budget: &mut Budget) -> Result<MonoId, Limit> {
        let hash = self.scratch_hash();
        self.intern_scratch_hashed(degree, hash, budget)
    }

    /// The id of the monomial in `scratch`, of degree `degree` and of
    /// `hash`, its hash in this table.
    #[inline]
    fn intern_scratch_hashed(
        &mut self,
        degree: u32,
        hash: u64,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        let w = self.format.words();
        let found = self.find(hash, |words, c| {
            let c = c.index() * w;
            words[c..c + w] == self.scratch[..]
        });
        match found {
            Ok(id) => Ok(id),
            Err(slot) => self.insert(slot, hash, degree, budget),
        }
    }

    // f4-gfp.16 practice-checked: linear probing -- the table probes linearly in a power-of-two slot array, doubled at half load, not by quadratic probing
    // matrix-sort-time.8 cited-checked: bachmann-1998 §3, "Fourthly" -- monomial comparisons reduced to one inlined routine, in place of a function call each, gave a considerable efficiency gain
    // wide-format-products-time.9 design-checked -- the test takes the candidate's id, from which it reads the degree and computes the words' offset by a product, not the words' offset, from which the id is a division by the word count (log 0132-1, 0132-2)
    /// The id whose hash is `hash` and which `same`, given the words and the
    /// id, accepts; or the empty slot where it would stand. The slots' tags
    /// are compared before the words.
    #[inline(always)]
    fn find(&self, hash: u64, same: impl Fn(&[M::Word], MonoId) -> bool) -> Result<MonoId, usize> {
        // f4-gfp.44 practice-checked: multiplicative hashing -- the slot is the hash's top bits, not its low bits
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the shift keeps the slot below the table's size"
        )]
        let home = (hash >> self.shift) as usize;
        // SAFETY: the slot array holds a power of two of slots (`SLOTS`,
        // `grow`), the hash shifted right by `shift` lies below the count,
        // 2^(64 − shift), and every slot holds EMPTY_SLOT or an id below
        // len() under its tag.
        unsafe {
            probe(
                self.slots.as_ptr(),
                self.slots.len() - 1,
                home,
                tag_of(hash),
                |id| same(&self.words, id),
            )
        }
    }

    // matrix-slack.8 practice-checked: geometric growth -- the table's arrays grow by doubling, not each reserved at 2^23 entries at the table's 2^16th monomial
    /// Stores the monomial in `scratch` at the empty slot `slot`: its id.
    /// Where the monomial takes the table past its load, the slot array
    /// grows (`grow`), and a stop there is `Limit::Stop`.
    fn insert(
        &mut self,
        slot: usize,
        hash: u64,
        degree: u32,
        budget: &mut Budget,
    ) -> Result<MonoId, Limit> {
        let id = MonoId::new(self.degree.len())?;
        self.words.extend_from_slice(&self.scratch);
        self.degree.push(degree);
        self.hash.push(hash);
        self.mask.push(self.format.mask(&self.scratch));
        // wide-format-products-time.7 design-checked -- the support is computed here from the words stored, the one place a monomial enters the table, not handed in by each caller that formed the monomial (log 0132-2)
        if self.supports() {
            self.support.push(support_of(&self.scratch, self.group));
        }
        self.seen.push(Seen {
            stamp: 0,
            column: MonoId::NONE,
        });
        self.slots[slot] = slot_of(hash, id);
        // threshold-recheck.15 design-checked -- the slot array doubles once its monomials pass half of its slots, not 4 or 6 in 10 of them; neither moved the time or the memory past the margin on the tuning set (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0074-2)
        if LOAD * self.len() > self.slots.len() {
            self.grow(budget)?;
        }
        Ok(id)
    }

    // call-stop-latency.7 design-checked -- a growth of the table places its ids PLACE, 2^14, at a time and spends PROBE units an id after each chunk, not the ids placed at once (log 0138-1)
    // call-stop-latency.36 design-checked -- a growth appends its new slots and empties the old ones a chunk at a time, a unit a slot spent after each chunk, not the slot array resized and emptied in one step before its placement (not measured)
    // call-stop-latency.37 design-checked -- a stop inside a growth or a compaction leaves no slot and no id past an array's end, the new slots dropped where the stop comes before they are all appended, a compaction's slots emptied before its moves and its degrees moved first, not the table left as the stop found it (not measured)
    /// Doubles the slot array and places every id again, spent at
    /// `Site::Place` as it goes: the new slots appended `FILL` at a time,
    /// then the old ones emptied (`empty`) and every id placed (`place`). A
    /// stop while the new slots are appended leaves the slot array as it
    /// was; a stop after leaves it doubled and part placed. Either way each
    /// slot holds `EMPTY_SLOT` or an id of the table under its tag, at a
    /// slot count that is a power of two.
    fn grow(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let old = self.slots.len();
        self.slots.reserve_exact(old);
        while self.slots.len() < 2 * old {
            let n = FILL.min(2 * old - self.slots.len());
            self.slots.resize(self.slots.len() + n, EMPTY_SLOT);
            if let Err(stop) = budget.spend(n, Site::Place) {
                self.slots.truncate(old);
                return Err(stop);
            }
        }
        self.shift -= 1;
        self.empty(old, Site::Place, budget)?;
        self.place(Site::Place, budget)
    }

    /// Empties the first `len` slots, `FILL` at a time, a unit a slot spent
    /// at `site` after each chunk.
    fn empty(&mut self, len: usize, site: Site, budget: &mut Budget) -> Result<(), Stop> {
        for chunk in self.slots[..len].chunks_mut(FILL) {
            chunk.fill(EMPTY_SLOT);
            budget.spend(chunk.len(), site)?;
        }
        Ok(())
    }

    /// Places every id by its hash, with its tag, probing linearly from the
    /// hash's top bits, into a slot array that holds none, `PLACE` ids at a
    /// time, `PROBE` units an id spent at `site` after each chunk.
    fn place(&mut self, site: Site, budget: &mut Budget) -> Result<(), Stop> {
        let last = self.slots.len() - 1;
        for (k, chunk) in self.hash.chunks(PLACE).enumerate() {
            for (i, &hash) in chunk.iter().enumerate() {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "the shift keeps the slot below the table's size"
                )]
                let mut slot = (hash >> self.shift) as usize;
                while self.slots[slot] != EMPTY_SLOT {
                    slot = (slot + 1) & last;
                }
                self.slots[slot] = slot_of(hash, MonoId::at(k * PLACE + i));
            }
            budget.spend(chunk.len() * PROBE, site)?;
        }
        Ok(())
    }

    // dead-monomial-peak.2 practice-checked: sliding compaction -- the kept monomials move down in the table's arrays in the order of their ids, in place, not interned into a fresh table as the carry's (signature-run-losses.5), which stands beside the old one while it fills
    // dead-monomial-peak.5 design-checked -- the arrays keep their capacity and the slot array its size, emptied and every kept id placed again, not the arrays shrunk to the kept monomials and the slot array to the least that holds them, which the next rounds copy again as they grow them (log 0117-1)
    // call-stop-latency.8 design-checked -- a compaction spends its moves a unit a word, its slots emptied a unit each, and its ids placed PROBE units each, PLACE at a time, and the run's walks of the basis's and the buffers' ids a unit an id, not no spend (dead-monomial-peak.8) (log 0138-1)
    /// Keeps the monomials `live` marks and drops the rest: each kept one
    /// moves down to the id `live.id` names, in the order of the ids, with
    /// its words, degree, hash, mask, support, and record, and every kept id
    /// is placed again (`place`). The arrays keep their capacity and the slot
    /// array its size. The hash multipliers and the round stay, so a kept
    /// monomial keeps its hash, its stamp, and its column slot. The count
    /// kept, or the stop where the budget's check stops the call: the work
    /// is spent at `Site::Compact` as it goes, the marks' words, the slots
    /// emptied, and the words moved a unit each and the ids placed `PROBE`
    /// units each. The slot array is emptied first and the degrees, whose
    /// length numbers the monomials, move before the other arrays, so that
    /// after a stop no slot and no id lies past an array's end.
    pub(crate) fn compact(&mut self, live: &mut Live, budget: &mut Budget) -> Result<usize, Stop> {
        debug_assert_eq!(
            live.len,
            self.len(),
            "the marks of this table's {len} monomials",
            len = self.len()
        );
        let kept = live.seal();
        budget.spend(live.bits.len(), Site::Compact)?;
        self.empty(self.slots.len(), Site::Compact, budget)?;
        keep_marked(&mut self.degree, 1, live, kept, budget)?;
        keep_marked(&mut self.words, self.format.words(), live, kept, budget)?;
        keep_marked(&mut self.hash, 1, live, kept, budget)?;
        keep_marked(&mut self.mask, 1, live, kept, budget)?;
        if self.supports() {
            keep_marked(&mut self.support, 1, live, kept, budget)?;
        }
        keep_marked(&mut self.seen, 1, live, kept, budget)?;
        self.place(Site::Compact, budget)?;
        Ok(kept)
    }

    /// The slot array's length.
    #[cfg(test)]
    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

// dead-monomial-peak.3 practice-checked: rank on a bit vector -- a kept monomial's new id is the count of the kept ids before its old one, from a bit per id and a count per 64 ids, not a map of a 4-byte id per old id, which stands beside the table at its largest (not measured)
/// The monomials of a table that a compaction keeps (`Table::compact`): a
/// bit per id, bit `i % 64` of word `i / 64`, and once sealed, per word,
/// the kept ids before it. A kept id's new id is that count plus the kept
/// ids below it in its word.
pub(crate) struct Live {
    bits: Vec<u64>,
    /// Per word of `bits`, the kept ids before it; empty until `seal`.
    before: Vec<u32>,
    /// The monomials of the table the marks are for.
    len: usize,
}

impl Live {
    /// No monomial of a table of `len` monomials marked.
    pub(crate) fn new(len: usize) -> Live {
        Live {
            bits: vec![0; len.div_ceil(64)],
            before: Vec::new(),
            len,
        }
    }

    /// Marks the monomial `id` kept.
    #[inline]
    pub(crate) fn mark(&mut self, id: MonoId) {
        let i = id.index();
        debug_assert!(
            id != MonoId::NONE && i < self.len,
            "id {id} past the table's {len} monomials",
            len = self.len
        );
        self.bits[i / 64] |= 1 << (i % 64);
    }

    /// Whether the id of index `i` is marked.
    fn is_marked(&self, i: usize) -> bool {
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }

    /// Counts the kept ids before each word into `before`: the count kept.
    fn seal(&mut self) -> usize {
        let mut count = 0;
        self.before = self
            .bits
            .iter()
            .map(|w| {
                let before = count;
                count += w.count_ones();
                before
            })
            .collect();
        count as usize
    }

    /// The new id of the kept monomial `id`, once sealed.
    #[inline]
    pub(crate) fn id(&self, id: MonoId) -> MonoId {
        let i = id.index();
        debug_assert_eq!(self.before.len(), self.bits.len(), "the marks sealed");
        debug_assert!(
            i < self.len && self.is_marked(i),
            "id {id} not kept of the table's {len} monomials",
            len = self.len
        );
        let below = self.bits[i / 64] & ((1 << (i % 64)) - 1);
        MonoId::at(self.before[i / 64] as usize + below.count_ones() as usize)
    }
}

/// The runs of `width` entries of `v` whose index `live` marks, moved down
/// in order, each only where its new place differs from its old, and `v`
/// truncated to `kept` runs at its capacity. The marks are read `PLACE` ids
/// at a time, each chunk spending a unit per word of the marks and per entry
/// of `v` kept at `Site::Compact`; a stop where the budget's check stops the
/// call, `v` then at its length, part moved.
fn keep_marked<T: Copy>(
    v: &mut Vec<T>,
    width: usize,
    live: &Live,
    kept: usize,
    budget: &mut Budget,
) -> Result<(), Stop> {
    debug_assert!(
        width > 0 && v.len() == live.len * width,
        "{len} entries for {n} monomials of {width}",
        len = v.len(),
        n = live.len
    );
    let per = PLACE / u64::BITS as usize;
    let mut to = 0;
    for (c, chunk) in live.bits.chunks(per).enumerate() {
        let from = to;
        for (j, &word) in chunk.iter().enumerate() {
            let mut marked = word;
            while marked != 0 {
                let i = (c * per + j) * 64 + marked.trailing_zeros() as usize;
                marked &= marked - 1;
                if to != i {
                    if width == 1 {
                        v[to] = v[i];
                    } else {
                        v.copy_within(i * width..(i + 1) * width, to * width);
                    }
                }
                to += 1;
            }
        }
        budget.spend(chunk.len() + (to - from) * width, Site::Compact)?;
    }
    debug_assert_eq!(to, kept, "the kept ids counted");
    v.truncate(kept * width);
    Ok(())
}

// many-variables-time.4 design-checked -- masks that share no bit mean no shared variable at any n, and masks that share a bit a shared variable up to 32 variables, past which the exponents decide, not a shared bit read as a shared variable at every n nor the exponents read at every test (not measured)
/// Whether the monomials a, of mask `mask_a` and words `a`, and b, of
/// `mask_b` and `b`, share no variable, for a format of N words
/// (`monomial::ANY`). Up to 32 variables a bit of the mask stands for one
/// variable, past 32 for several (`Format::mask`).
#[inline(always)]
pub(crate) fn coprime_masked<const N: usize, M: Format>(
    format: &M,
    mask_a: u32,
    a: &[M::Word],
    mask_b: u32,
    b: &[M::Word],
) -> bool {
    if mask_a & mask_b == 0 {
        return true;
    }
    format.variables() > u32::BITS as usize && format.coprime_n::<N>(a, b)
}

/// Masks tested at a time.
const BLOCK: usize = 8;

// call-stop-latency.71 design-checked -- a search by find_passing spends PROBE units per candidate its test ran on beside its blocks of masks, the candidates counted in the test's closure, not the blocks alone (.14), as on few variables most masks pass and each candidate's test reads its words at a random place (log 0138-8)
/// The units a search by `find_passing` over `masks` masks spends where
/// its test ran on `tested` candidates: a unit per block of `BLOCK` masks,
/// the blocks it tests where it finds none (call-stop-latency.14), and
/// `PROBE` units a candidate, whose test reads the candidate's words at a
/// random place (.71).
pub(crate) fn search_units(masks: usize, tested: usize) -> usize {
    masks.div_ceil(BLOCK) + tested * PROBE
}

// call-stop-latency.60 design-checked -- a monomial interned from its exponents spends MONOMIAL_PASSES, 3, units a variable and PROBE, its exponents written, encoded, and hashed and its slot probed, not a unit a term (log 0138-6)
/// Passes over a monomial's exponents that its interning from exponents
/// takes: the exponents written, encoded, and hashed.
pub(crate) const MONOMIAL_PASSES: usize = 3;

/// The units of work of interning a monomial of `n` variables from its
/// exponents: `MONOMIAL_PASSES` a variable and `PROBE`, its slot probed.
pub(crate) const fn monomial_units(n: usize) -> usize {
    MONOMIAL_PASSES * n + PROBE
}

// instance-read-peak.6 design-checked -- a monomial interned from the exponents of its touched variables spends, where it is new, a unit a word of the format, its words cleared and stored, not a unit a variable, which it no longer encodes (log 0175-1, 0175-2)
/// The units of work of interning a monomial from the exponents of its
/// `touched` variables, in a format of `words` words a monomial
/// (`Table::intern_touched`): `MONOMIAL_PASSES` a touched variable, its
/// exponent written, hashed, and compared, and `PROBE`, its slot probed;
/// and where it is new `words`, its words cleared and stored.
pub(crate) const fn touched_units(touched: usize, words: usize, new: bool) -> usize {
    MONOMIAL_PASSES * touched + PROBE + if new { words } else { 0 }
}

/// A hash multiplier per variable of `variables`, drawn from `rng`: a
/// table's (f4-gfp.15), and the parse's for the text's distinct monomials
/// (instance-read-peak.3).
pub(crate) fn multipliers(variables: usize, rng: &mut impl Rng) -> Vec<u64> {
    (0..variables).map(|_| rng.next_u64()).collect()
}

/// The hash of the monomial whose nonzero exponents `exponents` gives, each
/// variable once with its exponent: the sum of `r·e` over them, `r` the
/// variable's multiplier in `multipliers` (f4-gfp.15).
#[inline(always)]
pub(crate) fn sum_hash(multipliers: &[u64], exponents: impl Iterator<Item = (usize, u32)>) -> u64 {
    exponents.fold(0u64, |h, (i, e)| hash_term(h, multipliers, i, e))
}

/// The hash `h` with variable i's term at exponent e added: its multiplier
/// in `multipliers` times e, wrapping.
#[inline(always)]
fn hash_term(h: u64, multipliers: &[u64], i: usize, e: u32) -> u64 {
    debug_assert!(
        i < multipliers.len(),
        "variable {i} of {n}",
        n = multipliers.len()
    );
    h.wrapping_add(multipliers[i].wrapping_mul(u64::from(e)))
}

/// The first k, in increasing order, whose mask `masks[k]` passes `mask` —
/// a monomial of mask `masks[k]` may divide one of mask `mask` — and for
/// which `test` holds; None where none. The masks are tested eight at a
/// time, into a bit set of the candidates, before `test` runs on them.
/// The masks after the last whole block of eight are tested one at a time.
/// With both sides complemented the test reads the reverse inclusion, as
/// the plain run's B criterion asks it (plain-update-time.2).
#[inline(always)]
pub(crate) fn find_passing(
    masks: &[u32],
    mask: u32,
    mut test: impl FnMut(usize) -> bool,
) -> Option<usize> {
    let not_m = !mask;
    let (blocks, rest) = masks.as_chunks::<BLOCK>();
    let mut from = 0;
    while let Some((b, mut candidates)) = next_block(blocks, from, not_m) {
        while candidates != 0 {
            let k = b * BLOCK + candidates.trailing_zeros() as usize;
            candidates &= candidates - 1;
            if test(k) {
                return Some(k);
            }
        }
        from = b + 1;
    }
    let start = blocks.len() * BLOCK;
    (start..)
        .zip(rest)
        .find(|&(k, &mask)| mask & not_m == 0 && test(k))
        .map(|(k, _)| k)
}

// divisibility-time.7 design-checked -- the scan for the next block of eight masks with a candidate stands in a function of its own that the compiler does not inline, and the caller's rule runs on that block's candidates, not the scan inlined with the rule into each search (log 0043-1)
// mask-scan-time.5 tuned-checked -- a step tests eight masks, not 32 masks by one OR of their four blocks' tests before the block's own (log 0073-1)
/// The first block from `from` on with a candidate, a mask that passes the
/// query's mask `!not_m`: a monomial of that mask may divide one of the
/// query's. The block and its candidates, bit k for its k-th mask; None
/// where no block from `from` on has one.
#[inline(never)]
fn next_block(blocks: &[[u32; BLOCK]], from: usize, not_m: u32) -> Option<(usize, u32)> {
    for (b, block) in blocks.iter().enumerate().skip(from) {
        let mut candidates = 0u32;
        for (k, &mask) in block.iter().enumerate() {
            candidates |= u32::from(mask & not_m == 0) << k;
        }
        if candidates != 0 {
            return Some((b, candidates));
        }
    }
    None
}

// column-index-peak.4 practice-checked: array of structures -- a monomial's stamp and its column slot stand in one record, which every product of a build reads, not in two arrays of their own
/// A monomial's round and column slot: the round of the last matrix it
/// entered, 0 for none (`Table::open`), and its column there,
/// `MonoId::NONE` for none; while that matrix builds, its local column
/// (`enter_seen`).
#[derive(Clone, Copy)]
#[repr(C)]
struct Seen {
    stamp: u32,
    column: MonoId,
}

// column-index-peak.3 design-checked -- the build gives each monomial new to the round its local column, the count of the round's monomials before it in the order met, and writes that for every product, which the numbering maps to the sorted columns through a permutation, not the monomial id written and mapped through the table's column slot (monagan-2015 §2.1), the columns narrowed after the numbering (session 0113-1), nor the rows encoded after symbolic preprocessing by a second multiplication (pearce-2026 §2.1) (not measured)
/// Enters the monomial `id`, whose record is `seen`, into the matrix of
/// round `round`: where it is new to the round, it takes the round's
/// stamp, its id is appended to `fresh` and its column slot's value to
/// `previous`, and the slot takes its local column, the count of the
/// round's monomials entered before it, `previous`' length before. Its
/// local column.
#[inline(always)]
fn enter_seen(
    seen: &mut Seen,
    id: MonoId,
    round: u32,
    fresh: &mut Vec<MonoId>,
    previous: &mut Vec<MonoId>,
) -> MonoId {
    if seen.stamp != round {
        seen.stamp = round;
        fresh.push(id);
        let local = MonoId::at(previous.len());
        previous.push(seen.column);
        seen.column = local;
    }
    seen.column
}

/// Pointers to a table's arrays and the slot array's shape, as `arrays`
/// took them.
struct Arrays<W> {
    degree: *const u32,
    hash: *const u64,
    words: *const W,
    slots: *const u64,
    seen: *mut Seen,
    /// The slot count less one.
    last: usize,
    shift: u32,
    /// The number of monomials, for the test profile's checks.
    len: usize,
}

/// The W words of the monomial `id` in the words `words` points at, with
/// no bounds check.
///
/// # Safety
///
/// The words from `id`·W to (`id` + 1)·W lie in the array `words` points
/// into.
#[inline(always)]
unsafe fn words_of<T: Copy, const W: usize>(words: *const T, id: usize) -> [T; W] {
    // SAFETY: the caller's guarantee; an array of T has T's alignment.
    unsafe { words.add(id * W).cast::<[T; W]>().read() }
}

/// The support of a monomial's words: bit k set where words `k·group` to
/// `(k + 1)·group` hold a word other than the default, the word of zero
/// exponents.
fn support_of<W: Copy + Eq + Default>(words: &[W], group: usize) -> u64 {
    debug_assert!(
        group > 0 && words.len() <= group * u64::BITS as usize,
        "{len} words in groups of {group}",
        len = words.len()
    );
    words
        .chunks(group)
        .enumerate()
        .filter(|(_, g)| g.iter().any(|&x| x != W::default()))
        .fold(0, |s, (k, _)| s | 1 << k)
}

/// The slot of the id `id` whose hash is `hash`: the tag, the hash's low
/// 31 bits (`TAG`), above the id.
pub(crate) fn slot_of(hash: u64, id: MonoId) -> u64 {
    ((hash & TAG) << u32::BITS) | u64::from(id.raw())
}

/// The tag of the hash `hash` (`TAG`).
pub(crate) fn tag_of(hash: u64) -> u32 {
    (hash & TAG).narrow()
}

/// The tag a slot holds above its id. The empty slot's is no hash's.
fn tag_in(slot: u64) -> u32 {
    (slot >> u32::BITS).narrow()
}

/// The id a slot that is not empty holds.
fn id_in(slot: u64) -> MonoId {
    MonoId::from_raw((slot & u64::from(u32::MAX)).narrow())
}

// product-lookup-time.5 design-checked -- the probe reads the slot array with no bounds check, by the table's invariants: the first slot is the hash's top bits, every next one is masked, and every slot holds EMPTY_SLOT or an id below len() (log 0031-1)
// product-lookup-time.6 cited-checked: maier-2018 §5.7 -- a signature of the key, hash bits not used to find its position in the table, is stored beside the reference to the key, and a search compares the signatures before a full key comparison that involves a costly pointer dereference
// gf2-symbolic-speculation-time.1 cited-checked: harvey-2014 §2 -- a conditional addition or subtraction whose condition holds with probability 50 % or about 25 % is done by a conditional move instruction rather than by a branch, which it is still reasonable to prefer
// gf2-symbolic-speculation-time.8 design-checked -- the window keeps its candidate by comparing each slot's tag and moving the whole slot, not each slot tested for its tag and for the empty slot into a mask and the first match chosen by the masks, which took more instructions a lookup than the mispredictions they spared (log 0178-1)
// gf2-symbolic-speculation-time.9 design-checked -- the window reads the slots of the slot array itself, not a control byte a slot beside the slot array, eight of them compared as one word, which spared the mispredictions at more instructions a lookup and lost where the lookups probe few slots past the home slot (log 0178-1)
// gf2-symbolic-speculation-time.4 design-checked -- where the window passes the slot array's end, the probe loop runs alone, not the slot array padded with WINDOW − 1 slots that mirror its first ones (log 0178-1)
// gf2-symbolic-speculation-time.5 design-checked -- where the window's candidate is not the hash's monomial, the probe loop runs from the home slot, not from the slot past the window, which would need the window's test for the empty slot (log 0178-1)
// gf2-symbolic-speculation-time.6 design-checked -- every lookup of the table probes by this one function, the copies' products, the lookups per term, the interns, and the quotients alike, not the copies' products alone (log 0178-1)
/// The probe of the slot array at `slots`, of `last + 1` slots, for the
/// tag `tag` from the home slot `home` on: the first id under that tag
/// that `same` accepts, or the first empty slot, where the id would stand.
/// The `WINDOW` slots from the home slot on are read first, and the first
/// of them whose tag is `tag` is kept without a branch a slot. Where `same`
/// accepts it, the probe ends there; otherwise the probe loop tests each
/// slot from the home slot on, as it does alone where the window passes
/// the array's end and, in the test profile, under `WINDOW_OFF`.
///
/// # Safety
///
/// `slots` points at `last + 1` slots, a power of two, `home` lies at or
/// below `last`, every slot holds `EMPTY_SLOT` or an id under its tag that
/// `same` may be asked of, and an id `same` accepts stands at or before the
/// first empty slot from the home slot on, where linear probing places it.
#[inline(always)]
pub(crate) unsafe fn probe(
    slots: *const u64,
    last: usize,
    home: usize,
    tag: u32,
    mut same: impl FnMut(MonoId) -> bool,
) -> Result<MonoId, usize> {
    debug_assert!(!slots.is_null(), "a slot array");
    debug_assert!(last.wrapping_add(1).is_power_of_two(), "{last} + 1 slots");
    debug_assert!(home <= last, "slot {home} past the {last} + 1 slots");
    debug_assert!(u64::from(tag) <= TAG, "tag {tag} past the hash's bits");
    let windowed = home + WINDOW <= last + 1;
    #[cfg(test)]
    let windowed = windowed && !WINDOW_OFF.get();
    if windowed {
        // SAFETY: the WINDOW slots from the home slot on lie at or below
        // `last`.
        let mut candidate = unsafe { *slots.add(home + WINDOW - 1) };
        for k in (0..WINDOW - 1).rev() {
            // SAFETY: as above.
            let s = unsafe { *slots.add(home + k) };
            candidate = select_unpredictable(tag_in(s) == tag, s, candidate);
        }
        if tag_in(candidate) == tag {
            let id = id_in(candidate);
            if same(id) {
                return Ok(id);
            }
        }
    }
    let mut slot = home;
    loop {
        // SAFETY: the home slot lies at or below `last`, by the caller's
        // guarantee, and every next slot is masked by `last`.
        let s = unsafe { *slots.add(slot) };
        if s == EMPTY_SLOT {
            return Err(slot);
        }
        if tag_in(s) == tag && same(id_in(s)) {
            return Ok(id_in(s));
        }
        slot = (slot + 1) & last;
    }
}

/// The entry of monomial `id` in the per-monomial array `array`, named
/// `what`, with no bounds check outside the test profile, whose debug
/// assertions check it.
///
/// # Safety
///
/// `id` lies below `array.len()`.
#[inline(always)]
unsafe fn entry<'a, T>(array: &'a [T], id: MonoId, what: &str) -> &'a T {
    debug_assert!(
        id.index() < array.len(),
        "{what} of id {id} past the table's {len} monomials",
        len = array.len()
    );
    // SAFETY: the caller's guarantee.
    unsafe { array.get_unchecked(id.index()) }
}

/// The entry of monomial `id` in the per-monomial array `array`, named
/// `what`, to write, with no bounds check outside the test profile, whose
/// debug assertions check it.
///
/// # Safety
///
/// `id` lies below `array.len()`.
#[inline(always)]
unsafe fn entry_mut<'a, T>(array: &'a mut [T], id: MonoId, what: &str) -> &'a mut T {
    debug_assert!(
        id.index() < array.len(),
        "{what} of id {id} past the table's {len} monomials",
        len = array.len()
    );
    // SAFETY: the caller's guarantee.
    unsafe { array.get_unchecked_mut(id.index()) }
}

/// The `w` words from `start`, with no bounds check outside the test
/// profile, whose debug assertions check it.
///
/// # Safety
///
/// `start + w` lies at or below `words.len()`.
#[inline(always)]
unsafe fn at<W>(words: &[W], start: usize, w: usize) -> &[W] {
    debug_assert!(
        start + w <= words.len(),
        "words {start} to {end} past the table's {len}",
        end = start + w,
        len = words.len()
    );
    // SAFETY: the caller's guarantee.
    unsafe { words.get_unchecked(start..start + w) }
}

#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::*;
    use crate::monomial::Dense16;
    use crate::order::Grevlex;

    fn table(n: usize) -> Table<Dense16> {
        Table::new(Dense16::new(n), &mut Xoshiro256PlusPlus::seed_from_u64(7))
    }

    fn exponents(t: &Table<Dense16>, id: MonoId) -> Vec<u32> {
        (0..t.format().variables())
            .map(|i| t.exponent(id, i))
            .collect()
    }

    /// Products, quotients, and lcms of pairs of 1728 monomials, every
    /// seventh against every eleventh, find the id the exponents intern to,
    /// across many doublings of the slot array, and ids never move.
    #[test]
    fn operations_agree_with_interning() {
        let mut t = table(3);
        let mut ids = Vec::new();
        for a in 0..12 {
            for b in 0..12 {
                for c in 0..12 {
                    ids.push((
                        (a, b, c),
                        t.intern(&[a, b, c], &mut Budget::unbounded()).unwrap(),
                    ));
                }
            }
        }
        assert_eq!(t.len(), 12 * 12 * 12);
        for &((a, b, c), id) in &ids {
            assert_eq!(t.intern(&[a, b, c], &mut Budget::unbounded()), Ok(id));
            assert_eq!(exponents(&t, id), [a, b, c]);
            assert_eq!(t.degree(id), a + b + c);
        }
        for &((a, b, c), x) in ids.iter().step_by(7) {
            for &((d, e, f), y) in ids.iter().step_by(11) {
                let p = t.mul(x, y, &mut Budget::unbounded()).unwrap();
                assert_eq!(exponents(&t, p), [a + d, b + e, c + f]);
                assert_eq!(
                    t.intern(&[a + d, b + e, c + f], &mut Budget::unbounded()),
                    Ok(p)
                );
                assert_eq!(t.div(p, y, &mut Budget::unbounded()), Ok(x));
                assert_eq!(t.div(p, x, &mut Budget::unbounded()), Ok(y));
                let l = t.lcm(x, y, &mut Budget::unbounded()).unwrap();
                assert_eq!(exponents(&t, l), [a.max(d), b.max(e), c.max(f)]);
                assert_eq!(
                    t.intern(&[a.max(d), b.max(e), c.max(f)], &mut Budget::unbounded()),
                    Ok(l)
                );
                assert!(t.divides(x, l) && t.divides(y, l));
                assert_eq!(t.divides(x, y), a <= d && b <= e && c <= f);
            }
        }
        // ids interned first still name their monomials
        for &((a, b, c), id) in &ids {
            assert_eq!(exponents(&t, id), [a, b, c]);
        }
    }

    #[test]
    fn products_past_the_format_overflow() {
        let mut t = table(2);
        let x = t.intern(&[20000, 0], &mut Budget::unbounded()).unwrap();
        let y = t.intern(&[0, 15000], &mut Budget::unbounded()).unwrap();
        let len = t.len();
        assert_eq!(t.mul(x, y, &mut Budget::unbounded()), Err(Limit::Overflow));
        assert_eq!(t.lcm(x, y, &mut Budget::unbounded()), Err(Limit::Overflow));
        assert_eq!(t.len(), len, "an overflow stores nothing");
        let z = t.intern(&[12767, 0], &mut Budget::unbounded()).unwrap();
        let p = t.mul(x, z, &mut Budget::unbounded()).unwrap();
        assert_eq!(exponents(&t, p), [32767, 0]);
    }

    #[test]
    fn columns_stamps_and_order() {
        let mut t = table(2);
        let x = t.intern(&[1, 0], &mut Budget::unbounded()).unwrap();
        let y = t.intern(&[0, 1], &mut Budget::unbounded()).unwrap();
        assert_eq!((t.column(x), t.stamp(x)), (MonoId::NONE, 0));
        t.set_column(x, MonoId::at(3));
        t.set_stamp(y, 5);
        assert_eq!((t.column(x), t.stamp(y)), (MonoId::at(3), 5));
        assert_eq!(t.cmp(&Grevlex, x, y), Ordering::Greater);
        assert_eq!(t.cmp(&Grevlex, y, y), Ordering::Equal);
    }
}
