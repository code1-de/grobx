//! The monomial table: every monomial of a run's polynomials, or of its
//! cofactors, stored once and named by a `MonoId` that never changes.

use std::cmp::Ordering;

use rand_core::Rng;

use crate::id::MonoId;
use crate::monomial::{Format, Overflow};
use crate::narrow::{Narrow, Size};
use crate::order::Order;

/// The slot of no monomial. No stored slot equals it, as no id reaches
/// `u32::MAX` (`insert`).
const EMPTY_SLOT: u64 = u64::MAX;
// threshold-recheck.14 design -- the slot array starts at 2^10 slots, not 2^9 or 2^11; neither moved the time past the margin on the tuning set (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0074-2)
/// The slot array's capacity at the start, a power of two.
const SLOTS: usize = 1 << 10;

/// The slots per monomial below which the slot array doubles
/// (threshold-recheck.15).
const LOAD: usize = 2;

/// Why the table names no monomial for a product or for exponents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Limit {
    /// Its degree passes the format.
    Overflow,
    /// The table holds as many monomials as a u32 id numbers.
    Size,
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

// f4-gfp.14 cited: monagan-2015 §2.1 -- once created, a monomial persists over the whole computation; garbage collecting them did not improve the timings; the idea attributed to Magma [16], a private communication
/// The monomials of one run, or of its cofactors, in parallel arrays
/// indexed by the id.
pub(crate) struct Table<M: Format> {
    format: M,
    /// The hash multipliers, one per variable.
    multipliers: Vec<u64>,
    /// The exponent words, `format.words()` per monomial, back to back.
    words: Vec<M::Word>,
    degree: Vec<u32>,
    hash: Vec<u64>,
    mask: Vec<u32>,
    // f4-gfp.19 cited: monagan-2015 §2.1 -- the monomials contain a field for a column index, which maps polynomials to matrix rows
    /// The column in the last matrix the monomial entered, `MonoId::NONE`
    /// for none.
    column: Vec<MonoId>,
    /// The round of the last matrix the monomial entered, 0 for none.
    stamp: Vec<u32>,
    /// The ids by slot, each below its tag, the hash's low 32 bits:
    /// `tag << 32 | id`, `EMPTY_SLOT` where none.
    slots: Vec<u64>,
    /// 64 minus log2 of the slot count: the hash shifted right by it is a
    /// slot.
    shift: u32,
    /// Exponent words of a monomial being looked up.
    scratch: Vec<M::Word>,
}

impl<M: Format> Table<M> {
    // f4-gfp.15 cited: monagan-2015 §2.1 -- h = Σ r_i e_i with random integers chosen at the outset; adding hashes made monomial multiplication 20% faster; the idea attributed to Faugère and Magma [16], a private communication
    /// An empty table whose hash multipliers are drawn from `rng`.
    pub(crate) fn new(format: M, rng: &mut impl Rng) -> Table<M> {
        let multipliers = (0..format.variables()).map(|_| rng.next_u64()).collect();
        Table {
            format,
            multipliers,
            words: Vec::new(),
            degree: Vec::new(),
            hash: Vec::new(),
            mask: Vec::new(),
            column: Vec::new(),
            stamp: Vec::new(),
            slots: vec![EMPTY_SLOT; SLOTS],
            shift: 64 - SLOTS.trailing_zeros(),
            scratch: vec![M::Word::default(); format.words()],
        }
    }

    pub(crate) fn format(&self) -> &M {
        &self.format
    }

    /// The number of monomials.
    pub(crate) fn len(&self) -> usize {
        self.degree.len()
    }

    // table-read-time.1 design -- the table's read accessors, and the writes of a column and a stamp, index their arrays with no bounds check outside the test profile, an id the table interned as their guarantee, not a bounds check per read (log 0045-1)
    // table-read-time.2 design -- the accessors stay safe functions whose bodies read unchecked, the table's invariant holding the id in range, not unsafe functions whose callers each guarantee the id (not measured)
    // table-read-time.3 design -- the test profile checks each unchecked read's index by a debug assertion, not one assert per call in every profile (log 0031-1)
    /// The words of `id`. Each accessor of an id reads with no bounds check
    /// outside the test profile, whose debug assertions check it: an id
    /// names a monomial the table interned, and the table removes none.
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
        // `column`.
        unsafe { *entry(&self.column, id, "column") }
    }

    pub(crate) fn set_column(&mut self, id: MonoId, column: MonoId) {
        // SAFETY: an id of the table lies below len(), the length of
        // `column`.
        unsafe { *entry_mut(&mut self.column, id, "column") = column };
    }

    pub(crate) fn stamp(&self, id: MonoId) -> u32 {
        // SAFETY: an id of the table lies below len(), the length of
        // `stamp`.
        unsafe { *entry(&self.stamp, id, "stamp") }
    }

    pub(crate) fn set_stamp(&mut self, id: MonoId, round: u32) {
        // SAFETY: an id of the table lies below len(), the length of
        // `stamp`.
        unsafe { *entry_mut(&mut self.stamp, id, "stamp") = round };
    }

    /// Gives back the slack of the growable arrays. The slot array stays.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.words.shrink_to_fit();
        self.degree.shrink_to_fit();
        self.hash.shrink_to_fit();
        self.mask.shrink_to_fit();
        self.column.shrink_to_fit();
        self.stamp.shrink_to_fit();
    }

    /// The id of the monomial with these exponents, one per variable.
    pub(crate) fn intern(&mut self, exponents: &[u32]) -> Result<MonoId, Limit> {
        let degree = self.format.encode(exponents, &mut self.scratch)?;
        Ok(self.intern_scratch(degree)?)
    }

    // product-lookup-time.4 design -- every caller but `matrix::multiply` takes this safe door, one assert on both ids and then the unchecked lookup, not `mul` itself an unsafe fn whose callers each guarantee their ids (not measured)
    /// The id of a·b, or the limit it reaches. Panics where a or b is no id
    /// of the table.
    pub(crate) fn mul(&mut self, a: MonoId, b: MonoId) -> Result<MonoId, Limit> {
        // It holds: a caller's ids come from this table, and the cofactor
        // tracking interns a run's multiplier into its own table first.
        assert!(
            a.max(b).index() < self.len(),
            "ids {a} and {b} past the table's {len} monomials",
            len = self.len()
        );
        // SAFETY: both ids lie below len(), asserted above.
        unsafe { self.mul_unchecked(a, b) }
    }

    // product-lookup-time.1 design -- the lookup reads the factors' degrees, hashes, and words with no bounds check per read, the caller's guarantee in its place (log 0029-3, 0031-1)
    // product-lookup-time.7 cited: bachmann-1998 §3, "Fourthly" -- monomial comparisons reduced to one inlined routine, in place of a function call each, gave a considerable efficiency gain
    /// The id of a·b, or the limit it reaches. Its reads have no bounds
    /// check outside the test profile, whose debug assertions check them.
    ///
    /// # Safety
    ///
    /// a and b are ids of this table: both lie below `len()`.
    #[inline]
    pub(crate) unsafe fn mul_unchecked(&mut self, a: MonoId, b: MonoId) -> Result<MonoId, Limit> {
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
        let found = self.find(hash, |words, c| {
            // SAFETY: a and b lie below len() by the caller's guarantee, and
            // c starts the words of an id from a slot, below len() (`find`).
            let (a, b, c) = unsafe { (at(words, sa, w), at(words, sb, w), at(words, c, w)) };
            self.format.is_product(a, b, c)
        });
        match found {
            Ok(id) => Ok(id),
            Err(slot) => self.mul_miss(slot, hash, (sa, da), (sb, db)),
        }
    }

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
    ) -> Result<MonoId, Limit> {
        let w = self.format.words();
        self.format.mul(
            &self.words[sa..sa + w],
            da,
            &self.words[sb..sb + w],
            db,
            &mut self.scratch,
        )?;
        Ok(self.insert(slot, hash, da + db)?)
    }

    // matrix-build-time.1 design -- a row's products are looked up by one method of the table, which reads the multiplier's degree, hash, and words once per row, not `mul_unchecked` per term, which reads them at every term (log 0037-1)
    /// The products of one row: t·u for each u of `monomials`, in order,
    /// their ids appended to `cols`. A product whose stamp is not `round`
    /// takes it, and its id is appended to `fresh` as well, in the order
    /// met. The limit where a product reaches one: `cols` and `fresh` then
    /// hold what the products before it gave, and the table stores nothing
    /// for it. The ids are those `mul` gives.
    ///
    /// # Safety
    ///
    /// t and every u of `monomials` are ids of this table.
    pub(crate) unsafe fn row_products(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Vec<MonoId>,
        fresh: &mut Vec<MonoId>,
    ) -> Result<(), Limit> {
        // matrix-build-time.2 cited: demin-2024 §4.1 -- the specialized functions that implement monomial arithmetic are compiled and inlined, which relocates the work from running time to compile time; bachmann-1998 §3, independently -- the implementation contains optimizations like loop unrolling and pointer arithmetic
        // SAFETY: the caller's guarantee, and each copy runs on the format
        // of its word count.
        unsafe {
            match self.format.words() {
                1 => self.row_products_in::<1>(t, monomials, round, cols, fresh),
                2 => self.row_products_in::<2>(t, monomials, round, cols, fresh),
                3 => self.row_products_in::<3>(t, monomials, round, cols, fresh),
                4 => self.row_products_in::<4>(t, monomials, round, cols, fresh),
                // matrix-build-time.3 design -- a format of more than four words takes the lookup per term, not a copy for every word count up to eight (not measured)
                _ => self.row_products_per_term(t, monomials, round, cols, fresh),
            }
        }
    }

    /// `row_products` by `mul_unchecked` per term.
    ///
    /// # Safety
    ///
    /// As `row_products`.
    unsafe fn row_products_per_term(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Vec<MonoId>,
        fresh: &mut Vec<MonoId>,
    ) -> Result<(), Limit> {
        for &u in monomials {
            // SAFETY: the caller's guarantee.
            let id = unsafe { self.mul_unchecked(t, u) }?;
            cols.push(id);
            if self.stamp[id.index()] != round {
                self.stamp[id.index()] = round;
                fresh.push(id);
            }
        }
        Ok(())
    }

    /// `row_products` for a format of W words.
    ///
    /// # Safety
    ///
    /// As `row_products`, and the format has W words.
    #[inline(always)]
    unsafe fn row_products_in<const W: usize>(
        &mut self,
        t: MonoId,
        monomials: &[MonoId],
        round: u32,
        cols: &mut Vec<MonoId>,
        fresh: &mut Vec<MonoId>,
    ) -> Result<(), Limit> {
        debug_assert_eq!(self.format.words(), W);
        // matrix-build-time.5 cited: bachmann-1998 §3 -- the implementation contains optimizations like loop unrolling and the use of pointer arithmetic
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
        // matrix-build-time.6 design -- the row's columns are reserved once, before its first term, and each id written without a capacity check, not pushed per term (log 0037-1)
        cols.reserve(monomials.len());
        let out = cols.as_mut_ptr();
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
            let tag: u32 = (hash & u64::from(u32::MAX)).narrow();
            // matrix-build-time.4 design -- the product's words are formed once per term, each the sum of the factors' words, and each candidate's words compared with them, not `is_product` per candidate, which sums the factors' words again; bachmann-1998 §3, Lemma 1 (3) and `MonAdd`, forms a product's words as the word-wise sum and is silent on where a lookup forms the product (log 0037-1)
            // SAFETY: u lies below len(), so its W words below W·len().
            let uw = unsafe { words_of::<_, W>(a.words, u) };
            let product: [M::Word; W] = std::array::from_fn(|k| M::mul_word(tw[k], uw[k]));
            // product-lookup-time.5 design -- the probe reads the slot array with no bounds check, by the table's invariants: the first slot is the hash's top bits, every next one is masked, and every slot holds EMPTY_SLOT or an id below len() (log 0031-1)
            // product-lookup-time.6 design -- each slot holds the hash's low 32 bits beside the id, compared before the words, not the stored hash read from the hash array at each probed slot (log 0031-1)
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the shift keeps the slot below the table's size"
            )]
            let mut slot = (hash >> a.shift) as usize;
            let id = loop {
                debug_assert!(
                    slot <= a.last,
                    "slot {slot} past the table's {slots} slots",
                    slots = a.last + 1
                );
                // SAFETY: the hash shifted right by `shift` lies below the
                // slot count, and every next slot is masked by the count
                // less one.
                let s = unsafe { *a.slots.add(slot) };
                if s == EMPTY_SLOT {
                    let id = match self.mul_miss(slot, hash, (t * W, dt), (u * W, du)) {
                        Ok(id) => id,
                        Err(limit) => {
                            // SAFETY: as at the degree test above.
                            unsafe { cols.set_len(len) };
                            return Err(limit);
                        }
                    };
                    // The miss may have grown every array.
                    a = self.arrays();
                    break id;
                }
                let id = MonoId::from_raw((s & u64::from(u32::MAX)).narrow());
                debug_assert!(
                    id.index() < a.len,
                    "slot {slot} holds id {id} past the table's {len} monomials",
                    len = a.len
                );
                if (s >> u32::BITS) as u32 == tag
                    // SAFETY: a slot that is not EMPTY_SLOT holds an id below
                    // len().
                    && unsafe { words_of::<_, W>(a.words, id.index()) } == product
                {
                    break id;
                }
                slot = (slot + 1) & a.last;
            };
            // SAFETY: `len` lies below the length before the row plus the
            // row's terms, which `reserve` made room for.
            unsafe { out.add(len).write(id) };
            len += 1;
            // SAFETY: id lies below len(), the stamps' length.
            unsafe {
                let stamp = a.stamp.add(id.index());
                if *stamp != round {
                    *stamp = round;
                    fresh.push(id);
                }
            }
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
            stamp: self.stamp.as_mut_ptr(),
            last: self.slots.len() - 1,
            shift: self.shift,
            len: self.degree.len(),
        }
    }

    /// The id of a / b, where b divides a.
    pub(crate) fn div(&mut self, a: MonoId, b: MonoId) -> Result<MonoId, Size> {
        debug_assert!(self.divides(b, a));
        let degree = self.degree(a) - self.degree(b);
        let hash = self.hash(a).wrapping_sub(self.hash(b));
        let w = self.format.words();
        let (sa, sb) = (a.index() * w, b.index() * w);
        // c = a / b exactly where c·b = a.
        let found = self.find(hash, |words, c| {
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
                self.insert(slot, hash, degree)
            }
        }
    }

    /// The id of lcm(a, b), or the limit it reaches.
    #[cfg(test)]
    pub(crate) fn lcm(&mut self, a: MonoId, b: MonoId) -> Result<MonoId, Limit> {
        let w = self.format.words();
        let (sa, sb) = (a.index() * w, b.index() * w);
        let degree = self.format.lcm(
            &self.words[sa..sa + w],
            &self.words[sb..sb + w],
            &mut self.scratch,
        )?;
        Ok(self.intern_scratch(degree)?)
    }

    /// lcm(a, b) to `out`, not interned: its degree, or Overflow where it
    /// passes the format.
    pub(crate) fn lcm_to(
        &self,
        a: MonoId,
        b: MonoId,
        out: &mut [M::Word],
    ) -> Result<u32, Overflow> {
        self.format.lcm(self.monomial(a), self.monomial(b), out)
    }

    /// The id of the monomial of these words, of degree `degree`, interned.
    pub(crate) fn intern_words(&mut self, words: &[M::Word], degree: u32) -> Result<MonoId, Size> {
        self.scratch.copy_from_slice(words);
        self.intern_scratch(degree)
    }

    // table-read-time.9 design -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
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

    /// The id of the monomial formed last, of degree `degree`, interned.
    pub(crate) fn intern_formed(&mut self, degree: u32) -> Result<MonoId, Size> {
        self.intern_scratch(degree)
    }

    // f4-gfp.17 cited: roune-2012 §4.2 -- the divmasks eliminate around 98% of all divisibility checks when using the monomial list; berthomieu-2021 §4.1 (2) -- a 32-bit divisor mask
    /// Whether a divides b, the masks compared before the exponents.
    pub(crate) fn divides(&self, a: MonoId, b: MonoId) -> bool {
        self.mask(a) & !self.mask(b) == 0 && self.format.divides(self.monomial(a), self.monomial(b))
    }

    /// The degree of lcm(a, b), which may pass the format.
    pub(crate) fn lcm_degree(&self, a: MonoId, b: MonoId) -> u32 {
        self.format.lcm_degree(self.monomial(a), self.monomial(b))
    }

    // many-variables-time.4 design -- masks that share no bit mean no shared variable at any n, and masks that share a bit a shared variable up to 32 variables, past which the exponents decide, not a shared bit read as a shared variable at every n nor the exponents read at every test (not measured)
    /// Whether a and b share no variable. Up to 32 variables a bit of the
    /// mask stands for one variable, past 32 for several (`Format::mask`).
    pub(crate) fn coprime(&self, a: MonoId, b: MonoId) -> bool {
        if self.mask(a) & self.mask(b) == 0 {
            return true;
        }
        self.format.variables() > u32::BITS as usize
            && self.format.coprime(self.monomial(a), self.monomial(b))
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

    /// The hash of the exponents in `scratch`.
    fn scratch_hash(&self) -> u64 {
        (0..self.format.variables()).fold(0u64, |h, i| {
            h.wrapping_add(
                self.multipliers[i].wrapping_mul(u64::from(self.format.exponent(&self.scratch, i))),
            )
        })
    }

    /// The id of the monomial in `scratch`, of degree `degree`.
    fn intern_scratch(&mut self, degree: u32) -> Result<MonoId, Size> {
        let hash = self.scratch_hash();
        let w = self.format.words();
        let found = self.find(hash, |words, c| words[c..c + w] == self.scratch[..]);
        match found {
            Ok(id) => Ok(id),
            Err(slot) => self.insert(slot, hash, degree),
        }
    }

    // f4-gfp.16 cited: berthomieu-2021 §4.1 (1) -- hashing tables with linear probing store the exponent vectors
    // product-lookup-time.5 design -- the probe reads the slot array with no bounds check, by the table's invariants: the first slot is the hash's top bits, every next one is masked, and every slot holds EMPTY_SLOT or an id below len() (log 0031-1)
    // product-lookup-time.6 design -- each slot holds the hash's low 32 bits beside the id, compared before the words, not the stored hash read from the hash array at each probed slot (log 0031-1)
    // matrix-sort-time.8 cited: bachmann-1998 §3, "Fourthly" -- monomial comparisons reduced to one inlined routine, in place of a function call each, gave a considerable efficiency gain
    /// The id whose hash is `hash` and whose words, which start at the
    /// offset given, `same` accepts; or the empty slot where it would stand.
    /// The slots' tags are compared before the words.
    #[inline(always)]
    fn find(&self, hash: u64, same: impl Fn(&[M::Word], usize) -> bool) -> Result<MonoId, usize> {
        let w = self.format.words();
        let last = self.slots.len() - 1;
        let tag: u32 = (hash & u64::from(u32::MAX)).narrow();
        // f4-gfp.44 practice: multiplicative hashing -- the slot is the hash's top bits, not its low bits
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the shift keeps the slot below the table's size"
        )]
        let mut slot = (hash >> self.shift) as usize;
        loop {
            debug_assert!(
                slot < self.slots.len(),
                "slot {slot} past the table's {len} slots",
                len = self.slots.len()
            );
            // SAFETY: the hash shifted right by `shift` lies below the slot
            // count, 2^(64 − shift), and every next slot is masked by the
            // count less one.
            let s = unsafe { *self.slots.get_unchecked(slot) };
            if s == EMPTY_SLOT {
                return Err(slot);
            }
            let id = MonoId::from_raw((s & u64::from(u32::MAX)).narrow());
            if (s >> u32::BITS) as u32 == tag && same(&self.words, id.index() * w) {
                return Ok(id);
            }
            slot = (slot + 1) & last;
        }
    }

    // matrix-slack.8 practice: geometric growth -- the table's arrays grow by doubling, not each reserved at 2^23 entries at the table's 2^16th monomial
    /// Stores the monomial in `scratch` at the empty slot `slot`: its id.
    fn insert(&mut self, slot: usize, hash: u64, degree: u32) -> Result<MonoId, Size> {
        let id = MonoId::new(self.degree.len())?;
        self.words.extend_from_slice(&self.scratch);
        self.degree.push(degree);
        self.hash.push(hash);
        self.mask.push(self.format.mask(&self.scratch));
        self.column.push(MonoId::NONE);
        self.stamp.push(0);
        self.slots[slot] = slot_of(hash, id);
        // threshold-recheck.15 design -- the slot array doubles once its monomials pass half of its slots, not 4 or 6 in 10 of them; neither moved the time or the memory past the margin on the tuning set (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0074-2)
        if LOAD * self.len() > self.slots.len() {
            self.grow();
        }
        Ok(id)
    }

    /// Doubles the slot array and places every id again by its hash, with
    /// its tag.
    fn grow(&mut self) {
        let size = 2 * self.slots.len();
        self.slots.clear();
        self.slots.resize(size, EMPTY_SLOT);
        self.shift -= 1;
        let last = size - 1;
        for (id, &hash) in self.hash.iter().enumerate() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the shift keeps the slot below the table's size"
            )]
            let mut slot = (hash >> self.shift) as usize;
            while self.slots[slot] != EMPTY_SLOT {
                slot = (slot + 1) & last;
            }
            self.slots[slot] = slot_of(hash, MonoId::at(id));
        }
    }
}

/// Masks tested at a time.
const BLOCK: usize = 8;

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

// divisibility-time.7 design -- the scan for the next block of eight masks with a candidate stands in a function of its own that the compiler does not inline, and the caller's rule runs on that block's candidates, not the scan inlined with the rule into each search (log 0043-1)
// mask-scan-time.5 design -- a step tests eight masks, not 32 masks by one OR of their four blocks' tests before the block's own (log 0073-1)
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

/// Pointers to a table's arrays and the slot array's shape, as `arrays`
/// took them.
struct Arrays<W> {
    degree: *const u32,
    hash: *const u64,
    words: *const W,
    slots: *const u64,
    stamp: *mut u32,
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

/// The slot of the id `id` whose hash is `hash`: the tag, the hash's low
/// 32 bits, above the id.
fn slot_of(hash: u64, id: MonoId) -> u64 {
    (hash << u32::BITS) | u64::from(id.raw())
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
                    ids.push(((a, b, c), t.intern(&[a, b, c]).unwrap()));
                }
            }
        }
        assert_eq!(t.len(), 12 * 12 * 12);
        for &((a, b, c), id) in &ids {
            assert_eq!(t.intern(&[a, b, c]), Ok(id));
            assert_eq!(exponents(&t, id), [a, b, c]);
            assert_eq!(t.degree(id), a + b + c);
        }
        for &((a, b, c), x) in ids.iter().step_by(7) {
            for &((d, e, f), y) in ids.iter().step_by(11) {
                let p = t.mul(x, y).unwrap();
                assert_eq!(exponents(&t, p), [a + d, b + e, c + f]);
                assert_eq!(t.intern(&[a + d, b + e, c + f]), Ok(p));
                assert_eq!(t.div(p, y), Ok(x));
                assert_eq!(t.div(p, x), Ok(y));
                let l = t.lcm(x, y).unwrap();
                assert_eq!(exponents(&t, l), [a.max(d), b.max(e), c.max(f)]);
                assert_eq!(t.intern(&[a.max(d), b.max(e), c.max(f)]), Ok(l));
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
        let x = t.intern(&[20000, 0]).unwrap();
        let y = t.intern(&[0, 15000]).unwrap();
        let len = t.len();
        assert_eq!(t.mul(x, y), Err(Limit::Overflow));
        assert_eq!(t.lcm(x, y), Err(Limit::Overflow));
        assert_eq!(t.len(), len, "an overflow stores nothing");
        let z = t.intern(&[12767, 0]).unwrap();
        let p = t.mul(x, z).unwrap();
        assert_eq!(exponents(&t, p), [32767, 0]);
    }

    #[test]
    fn columns_stamps_and_order() {
        let mut t = table(2);
        let x = t.intern(&[1, 0]).unwrap();
        let y = t.intern(&[0, 1]).unwrap();
        assert_eq!((t.column(x), t.stamp(x)), (MonoId::NONE, 0));
        t.set_column(x, MonoId::at(3));
        t.set_stamp(y, 5);
        assert_eq!((t.column(x), t.stamp(y)), (MonoId::at(3), 5));
        assert_eq!(t.cmp(&Grevlex, x, y), Ordering::Greater);
        assert_eq!(t.cmp(&Grevlex, y, y), Ordering::Equal);
    }
}
