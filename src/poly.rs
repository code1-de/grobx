//! Polynomials over the monomial table, and the basis an F4 run builds.

#[cfg(test)]
use std::cell::Cell;
use std::ops::Range;

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::field::Field;
use crate::halt::Halt;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::Format;
use crate::narrow::{Narrow, Size};
#[cfg(test)]
use crate::order::Order;
use crate::table::{PLACE, Table};

/// A polynomial: monomial ids of one table in decreasing order, with their
/// coefficients, none zero, in the same order.
pub(crate) struct Poly<F: Field, D: Module> {
    pub(crate) monomials: Vec<MonoId>,
    pub(crate) coefs: Vec<F::Coef>,
    // f4-gfp.40 design-checked -- a row carries its module position as the module's type Position, () for rank 1, not a u32 that is always 0 (not measured)
    pub(crate) position: D::Position,
}

impl<F: Field, D: Module> Clone for Poly<F, D> {
    fn clone(&self) -> Self {
        Poly {
            monomials: self.monomials.clone(),
            coefs: self.coefs.clone(),
            position: self.position,
        }
    }
}

impl<F: Field, D: Module> std::fmt::Debug for Poly<F, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Poly")
            .field("monomials", &self.monomials)
            .field("coefs", &self.coefs)
            .field("position", &self.position)
            .finish()
    }
}

impl<F: Field, D: Module> Poly<F, D> {
    /// The polynomial of these terms, in any order: like terms combined,
    /// zero terms dropped. `terms` is left empty. For the test code, which
    /// builds its polynomials outside a call; a call's sort spends
    /// (`sort::poly`).
    #[cfg(test)]
    pub(crate) fn from_terms<M: Format, O: Order<M>>(
        terms: &mut Vec<(MonoId, F::Coef)>,
        field: &F,
        table: &Table<M>,
        order: &O,
        position: D::Position,
    ) -> Poly<F, D> {
        terms.sort_unstable_by(|a, b| table.cmp(order, b.0, a.0));
        #[allow(clippy::expect_used, reason = "a budget without bounds never stops")]
        Poly::from_sorted(terms, field, position, &mut Budget::unbounded(), Site::Map)
            .expect("a budget without bounds never stops")
    }

    /// The polynomial of these terms, in decreasing order of their
    /// monomials, like terms next to each other: like terms combined, zero
    /// terms dropped, a unit a term spent at `site`, `PLACE` terms at a
    /// time. `terms` is left empty. A stop where the budget's check stops
    /// the call.
    pub(crate) fn from_sorted(
        terms: &mut Vec<(MonoId, F::Coef)>,
        field: &F,
        position: D::Position,
        budget: &mut Budget,
        site: Site,
    ) -> Result<Poly<F, D>, Stop> {
        let mut monomials = Vec::with_capacity(terms.len());
        let mut coefs: Vec<F::Coef> = Vec::with_capacity(terms.len());
        for chunk in terms.chunks(PLACE) {
            for &(m, c) in chunk {
                if monomials.last() == Some(&m)
                    && let Some(last) = coefs.last_mut()
                {
                    *last = field.add(*last, c);
                } else {
                    // The previous monomial is complete: drop it where its
                    // coefficient summed to zero.
                    if coefs.last() == Some(&F::Coef::default()) {
                        monomials.pop();
                        coefs.pop();
                    }
                    monomials.push(m);
                    coefs.push(c);
                }
            }
            budget.spend(chunk.len(), site)?;
        }
        terms.clear();
        if coefs.last() == Some(&F::Coef::default()) {
            monomials.pop();
            coefs.pop();
        }
        Ok(Poly {
            monomials,
            coefs,
            position,
        })
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }

    /// The number of terms.
    pub(crate) fn len(&self) -> usize {
        self.monomials.len()
    }

    // table-read-time.4 design-checked -- the basis's element, leading monomial, and length, and a nonzero polynomial's leading monomial, read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// The leading monomial of a nonzero polynomial.
    pub(crate) fn lead(&self) -> MonoId {
        debug_assert!(
            !self.monomials.is_empty(),
            "the lead of the zero polynomial"
        );
        // SAFETY: a nonzero polynomial has a first monomial.
        unsafe { *self.monomials.get_unchecked(0) }
    }

    /// Scales a nonzero polynomial to leading coefficient 1: the factor.
    #[allow(
        clippy::expect_used,
        reason = "every caller passes a nonzero polynomial over a field of one lane"
    )]
    pub(crate) fn make_monic(&mut self, field: &F) -> F::Coef {
        self.try_monic(field)
            .expect("a nonzero polynomial's leading coefficient is a unit")
    }

    /// As `make_monic`, or None, the polynomial left as it is, where its
    /// leading coefficient has no inverse: over several lanes, 0 at one
    /// lane.
    pub(crate) fn try_monic(&mut self, field: &F) -> Option<F::Coef> {
        let s = field.inv(self.coefs[0])?;
        for c in &mut self.coefs {
            *c = field.mul(*c, s);
        }
        Some(s)
    }
}

/// The run's elements on their way into the basis, each with its
/// cofactors.
pub(crate) type Elements<F, D, E> = Vec<(Poly<F, D>, E)>;

/// A polynomial as the lift reads it: its terms' exponent vectors in an
/// `Exponents`, in decreasing order of the call's order, and their
/// coefficients in [0, p), 0 only where `lift::merge` gave an image a term
/// it lacked or where a pair image's term vanishes at its first prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sparse {
    pub(crate) exponents: Exponents,
    pub(crate) coefs: Vec<u64>,
}

#[cfg(test)]
thread_local! {
    // q-image-exponents-peak.10 design-checked -- in the test profile a thread-local makes every exponent vector start in `u32` on its thread, so that the test code compares a call with bytes against the same call with `u32`, not a parameter of the call (not measured)
    /// Whether the exponent vectors made on this thread hold `u32` from
    /// their making (`Exponents::with_capacity`).
    pub(crate) static EXPONENTS_IN_WORDS: Cell<bool> = const { Cell::new(false) };
}

// q-image-exponents-peak.1 cited-checked: bachmann-1998 §3, "Thirdly" -- the more exponents are encoded into one machine word, the less memory is used, and `char` exponents bound an exponent by 127
// q-image-exponents-peak.2 cited-checked: bachmann-1998 §3, "Thirdly" -- ideally the representation switches dynamically from one exponent size to the next larger one whenever it becomes necessary
// q-image-exponents-peak.3 design-checked -- two widths, a byte and a `u32`, not three with a `u16` between, which serves exponents from 256 to 65535 alone (not measured)
// q-image-exponents-peak.4 design-checked -- the exponent vectors are one type with the two widths behind its methods, not the image, the part, and the lift generic over the exponent's type, which compiles twice and fixes a width per call (not measured)
/// Exponent vectors, a term's exponents back to back, n per term for the
/// n of their caller. A vector holds each exponent in a byte while every
/// exponent it holds fits one, and each in a `u32` from the first that
/// does not, which widens the whole vector. An image's terms and a part of
/// the lift hold them.
#[derive(Clone, Debug)]
pub(crate) enum Exponents {
    Bytes(Vec<u8>),
    Words(Vec<u32>),
}

impl Exponents {
    /// No exponent, with room for `capacity`: in bytes, or in `u32` in the
    /// test profile where `EXPONENTS_IN_WORDS` is set on the thread.
    pub(crate) fn with_capacity(capacity: usize) -> Exponents {
        #[cfg(test)]
        if EXPONENTS_IN_WORDS.get() {
            return Exponents::Words(Vec::with_capacity(capacity));
        }
        Exponents::Bytes(Vec::with_capacity(capacity))
    }

    /// No exponent.
    pub(crate) fn new() -> Exponents {
        Exponents::with_capacity(0)
    }

    /// The exponents held.
    pub(crate) fn len(&self) -> usize {
        match self {
            Exponents::Bytes(v) => v.len(),
            Exponents::Words(v) => v.len(),
        }
    }

    /// Appends `exponents`. A vector in bytes where one of them does not
    /// fit a byte widens first, every value and its room kept.
    pub(crate) fn extend_from_slice(&mut self, exponents: &[u32]) {
        match self {
            Exponents::Bytes(v) if exponents.iter().all(|&e| u8::try_from(e).is_ok()) => {
                v.extend(exponents.iter().map(|&e| Narrow::<u8>::narrow(e)));
            }
            Exponents::Bytes(v) => {
                // q-image-exponents-peak.11 design-checked -- a vector that widens keeps the room it holds, not the values alone grown as they are appended to, so that a merge whose vector widens part way stays within the merged length it reserved (not measured)
                let mut words = Vec::with_capacity(v.capacity().max(v.len() + exponents.len()));
                words.extend(v.iter().map(|&e| u32::from(e)));
                words.extend_from_slice(exponents);
                *self = Exponents::Words(words);
            }
            Exponents::Words(v) => v.extend_from_slice(exponents),
        }
    }

    /// Appends `other`'s exponents at `range` as `extend_from_slice`
    /// appends their values.
    pub(crate) fn extend_from(&mut self, other: &Exponents, range: Range<usize>) {
        debug_assert!(
            range.end <= other.len(),
            "{range:?} of {len}",
            len = other.len()
        );
        match (&mut *self, other) {
            (Exponents::Bytes(v), Exponents::Bytes(o)) => v.extend_from_slice(&o[range]),
            (Exponents::Words(v), Exponents::Words(o)) => v.extend_from_slice(&o[range]),
            (Exponents::Words(v), Exponents::Bytes(o)) => {
                v.extend(o[range].iter().map(|&e| u32::from(e)));
            }
            (Exponents::Bytes(_), Exponents::Words(o)) => self.extend_from_slice(&o[range]),
        }
    }

    /// The exponents at `range` to `out`, as `u32`.
    pub(crate) fn widen_into(&self, range: Range<usize>, out: &mut [u32]) {
        debug_assert!(
            range.end <= self.len() && range.len() == out.len(),
            "{range:?} of {len} into {out}",
            len = self.len(),
            out = out.len()
        );
        match self {
            Exponents::Bytes(v) => {
                for (o, &e) in out.iter_mut().zip(&v[range]) {
                    *o = u32::from(e);
                }
            }
            Exponents::Words(v) => out.copy_from_slice(&v[range]),
        }
    }

    // q-image-exponents-peak.6 design-checked -- two vectors, and two terms, are equal where their values are, whatever the width each is held in, not equal only at one width: a part widened by one image may hold the terms of the next image in bytes (not measured)
    /// Whether the exponents at `range` equal `other`'s at `at`, value by
    /// value.
    pub(crate) fn same(&self, range: Range<usize>, other: &Exponents, at: Range<usize>) -> bool {
        debug_assert!(
            range.end <= self.len() && at.end <= other.len(),
            "{range:?} of {len} against {at:?} of {other_len}",
            len = self.len(),
            other_len = other.len()
        );
        if range.len() != at.len() {
            return false;
        }
        match (self, other) {
            (Exponents::Bytes(a), Exponents::Bytes(b)) => a[range] == b[at],
            (Exponents::Words(a), Exponents::Words(b)) => a[range] == b[at],
            (Exponents::Bytes(a), Exponents::Words(b)) => a[range]
                .iter()
                .zip(&b[at])
                .all(|(&x, &y)| u32::from(x) == y),
            (Exponents::Words(a), Exponents::Bytes(b)) => a[range]
                .iter()
                .zip(&b[at])
                .all(|(&x, &y)| x == u32::from(y)),
        }
    }

    /// Every exponent as `u32`, in order.
    pub(crate) fn values(&self) -> impl Iterator<Item = u32> + '_ {
        // One of the two is empty.
        let (bytes, words): (&[u8], &[u32]) = match self {
            Exponents::Bytes(v) => (v, &[]),
            Exponents::Words(v) => (&[], v),
        };
        bytes
            .iter()
            .map(|&e| u32::from(e))
            .chain(words.iter().copied())
    }

    /// The bytes its exponents take.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Exponents::Bytes(v) => v.len(),
            Exponents::Words(v) => 4 * v.len(),
        }
    }
}

// q-image-exponents-peak.6 design-checked -- two vectors, and two terms, are equal where their values are, whatever the width each is held in, not equal only at one width: a part widened by one image may hold the terms of the next image in bytes (not measured)
impl PartialEq for Exponents {
    fn eq(&self, other: &Exponents) -> bool {
        self.same(0..self.len(), other, 0..other.len())
    }
}

impl Eq for Exponents {}

#[cfg(test)]
impl Extend<u32> for Exponents {
    fn extend<I: IntoIterator<Item = u32>>(&mut self, iter: I) {
        let values: Vec<u32> = iter.into_iter().collect();
        self.extend_from_slice(&values);
    }
}

#[cfg(test)]
impl FromIterator<u32> for Exponents {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Exponents {
        let mut exponents = Exponents::new();
        exponents.extend(iter);
        exponents
    }
}

/// What `Basis::push_min` did with an element: its id, whether it joined
/// `G_min`, and how many members of `G_min` left it, none unless it joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entered {
    pub(crate) id: ElementId,
    pub(crate) joined: bool,
    pub(crate) left: usize,
}

/// The basis of a run: its elements in the order they were made, and `G_min`,
/// those no other element makes redundant.
pub(crate) struct Basis<F: Field, D: Module> {
    elements: Vec<Poly<F, D>>,
    lead: Vec<MonoId>,
    length: Vec<u32>,
    redundant: Vec<bool>,
    /// The elements not redundant, in the order made.
    min: Vec<ElementId>,
}

impl<F: Field, D: Module> Basis<F, D> {
    pub(crate) fn new() -> Basis<F, D> {
        Basis {
            elements: Vec::new(),
            lead: Vec::new(),
            length: Vec::new(),
            redundant: Vec::new(),
            min: Vec::new(),
        }
    }

    /// Adds a nonzero monic element: its index, or `Size` where the
    /// element's id passes what a u32 id numbers, nothing added. A redundant
    /// element joins the basis but not `G_min`.
    pub(crate) fn push(&mut self, poly: Poly<F, D>, redundant: bool) -> Result<ElementId, Size> {
        debug_assert!(!poly.is_zero());
        let i = ElementId::new(self.elements.len())?;
        self.lead.push(poly.lead());
        self.length.push(poly.len().narrow());
        self.redundant.push(redundant);
        if !redundant {
            self.min.push(i);
        }
        self.elements.push(poly);
        Ok(i)
    }

    /// Marks these elements redundant, and takes them out of `G_min`, a
    /// unit a member spent at `Site::Enter` after each `PLACE` members;
    /// nothing where `elements` is empty. A stop where the budget's check
    /// stops the call, `G_min` then part compacted, which the stopped run
    /// drops.
    fn mark_redundant(&mut self, elements: &[ElementId], budget: &mut Budget) -> Result<(), Stop> {
        if elements.is_empty() {
            return Ok(());
        }
        for &i in elements {
            self.redundant[i.index()] = true;
        }
        let mut kept = 0;
        for start in (0..self.min.len()).step_by(PLACE) {
            let end = (start + PLACE).min(self.min.len());
            for k in start..end {
                let i = self.min[k];
                if !self.redundant[i.index()] {
                    self.min[kept] = i;
                    kept += 1;
                }
            }
            budget.spend(end - start, Site::Enter)?;
        }
        self.min.truncate(kept);
        Ok(())
    }

    // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
    /// Adds a nonzero monic element and keeps `G_min`: what became of it,
    /// or `Halt::Size` as `push` returns it. The element joins `G_min` where
    /// no leading monomial of `G_min` divides its own, and the elements whose
    /// leading monomial its own divides leave it. `G_min`'s scans spend
    /// `PROBE` units a member, two leading monomials read from the table,
    /// at `Site::Enter` after each `PLACE` members, and the members' removal
    /// as `mark_redundant` spends; a stop where the budget's check stops
    /// the call, before the element is added.
    pub(crate) fn push_min<M: Format>(
        &mut self,
        table: &Table<M>,
        poly: Poly<F, D>,
        budget: &mut Budget,
    ) -> Result<Entered, Halt> {
        let lead = poly.lead();
        let mut divided = false;
        for chunk in self.min.chunks(PLACE) {
            divided = chunk.iter().any(|&i| table.divides(self.lead(i), lead));
            budget.spend(chunk.len() * PROBE, Site::Enter)?;
            if divided {
                break;
            }
        }
        self.push_min_known(table, poly, divided, budget)
    }

    // gf2-chain-criteria-time.3 design-checked -- the plain run's update hands the push whether its search found a member of G_min that divides the element's leading monomial, and the push runs no scan of its own for one, not push_min's scan run again, which reads each member's leading monomial from the table where none divides (log 0170-1, 0170-2)
    /// `push_min` where the caller knows whether a leading monomial of
    /// `G_min` divides the element's, `divided`: the plain run's update,
    /// whose search for a divided element's pair found it
    /// (gf2-chain-criteria-time.1). The test profile checks `divided`
    /// against `G_min`. Spent as `push_min` spends after its scan for a
    /// divisor.
    pub(crate) fn push_min_known<M: Format>(
        &mut self,
        table: &Table<M>,
        poly: Poly<F, D>,
        divided: bool,
        budget: &mut Budget,
    ) -> Result<Entered, Halt> {
        let lead = poly.lead();
        debug_assert_eq!(
            divided,
            self.min.iter().any(|&i| table.divides(self.lead(i), lead)),
            "whether a member of G_min divides the element's leading monomial"
        );
        if divided {
            let id = self.push(poly, true)?;
            return Ok(Entered {
                id,
                joined: false,
                left: 0,
            });
        }
        let mut gone: Vec<ElementId> = Vec::new();
        for chunk in self.min.chunks(PLACE) {
            gone.extend(
                chunk
                    .iter()
                    .copied()
                    .filter(|&i| table.divides(lead, self.lead(i))),
            );
            budget.spend(chunk.len() * PROBE, Site::Enter)?;
        }
        self.join(poly, &gone, budget)
    }

    // pair-index-upkeep-time.6 practice-checked: bitmap index -- where the plain run's pair set indexes, its update finds the members an element makes redundant by G_min's index and hands them to the push, which runs no scan of its own, not push_min_known's scan of every member through the table (log 0171-1, 0171-2)
    /// `push_min_known` where no leading monomial of `G_min` divides the
    /// element's and the caller found the members whose leading monomials
    /// the element's divides, `gone`, in `G_min`'s order: the plain run's
    /// update where its pair set indexes (`pairs::leaving`). The test
    /// profile checks both against `G_min`. Spent as `push_min` spends
    /// after its scans.
    pub(crate) fn push_min_leaving<M: Format>(
        &mut self,
        table: &Table<M>,
        poly: Poly<F, D>,
        gone: &[ElementId],
        budget: &mut Budget,
    ) -> Result<Entered, Halt> {
        let lead = poly.lead();
        debug_assert!(
            !self.min.iter().any(|&i| table.divides(self.lead(i), lead)),
            "no member of G_min divides the element's leading monomial"
        );
        debug_assert!(
            self.min
                .iter()
                .copied()
                .filter(|&i| table.divides(lead, self.lead(i)))
                .eq(gone.iter().copied()),
            "the members the element's leading monomial divides, in G_min's order"
        );
        self.join(poly, gone, budget)
    }

    /// Adds `poly` to `G_min`, which the members `gone` leave: they are
    /// marked redundant (`mark_redundant`) and the element is pushed.
    fn join(
        &mut self,
        poly: Poly<F, D>,
        gone: &[ElementId],
        budget: &mut Budget,
    ) -> Result<Entered, Halt> {
        self.mark_redundant(gone, budget)?;
        let id = self.push(poly, false)?;
        Ok(Entered {
            id,
            joined: true,
            left: gone.len(),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.elements.len()
    }

    // call-stop-latency.41 design-checked -- a walk of the ids spends a unit an id after each chunk of PLACE ids of a vector, not a count per id in the caller's closure, which made the compaction's walks of ccode-cooper45t4-p65521 cost more instructions (log 0138-2)
    /// Every monomial id of the basis to `f`: each element's monomials,
    /// then the leading monomials (`f4::compact`). A unit an id is spent at
    /// `site` after each chunk of `PLACE` ids of a vector, and a stop where
    /// the budget's check stops the call ends the walk.
    pub(crate) fn ids_mut(
        &mut self,
        mut f: impl FnMut(&mut MonoId),
        budget: &mut Budget,
        site: Site,
    ) -> Result<(), Stop> {
        for g in &mut self.elements {
            for chunk in g.monomials.chunks_mut(PLACE) {
                chunk.iter_mut().for_each(&mut f);
                budget.spend(chunk.len(), site)?;
            }
        }
        for chunk in self.lead.chunks_mut(PLACE) {
            chunk.iter_mut().for_each(&mut f);
            budget.spend(chunk.len(), site)?;
        }
        Ok(())
    }

    /// Gives back the slack of the growable arrays.
    pub(crate) fn shrink_to_fit(&mut self) {
        for g in &mut self.elements {
            g.monomials.shrink_to_fit();
            g.coefs.shrink_to_fit();
        }
        self.elements.shrink_to_fit();
        self.lead.shrink_to_fit();
        self.length.shrink_to_fit();
        self.redundant.shrink_to_fit();
        self.min.shrink_to_fit();
    }

    // table-read-time.4 design-checked -- the basis's element, leading monomial, and length, and a nonzero polynomial's leading monomial, read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// Element i. This read, `lead`, and `length` have no bounds check
    /// outside the test profile, whose debug assertions check them: an
    /// index names an element the basis pushed, and `push` grows
    /// `elements`, `lead`, and `length` together and the basis removes none.
    pub(crate) fn element(&self, i: ElementId) -> &Poly<F, D> {
        debug_assert!(
            i.index() < self.elements.len(),
            "element {i} past the basis's {len}",
            len = self.elements.len()
        );
        // SAFETY: an element index lies below the elements pushed.
        unsafe { self.elements.get_unchecked(i.index()) }
    }

    /// Element i's leading monomial.
    pub(crate) fn lead(&self, i: ElementId) -> MonoId {
        debug_assert!(
            i.index() < self.lead.len(),
            "the lead of element {i} past the basis's {len}",
            len = self.lead.len()
        );
        // SAFETY: as in `element`.
        unsafe { *self.lead.get_unchecked(i.index()) }
    }

    /// The element's number of terms.
    pub(crate) fn length(&self, i: ElementId) -> usize {
        debug_assert!(
            i.index() < self.length.len(),
            "the length of element {i} past the basis's {len}",
            len = self.length.len()
        );
        // SAFETY: as in `element`.
        unsafe { *self.length.get_unchecked(i.index()) as usize }
    }

    // linear-tail-time.6 cited-checked: buchberger-1985 §6.4, Subalgorithm New Basis -- each polynomial of the basis is replaced by its normal form modulo the others, G := K, the pairs B formed before it standing
    /// Element i takes `poly` in place of its polynomial, of the same
    /// leading monomial (`f4::tails`): its length follows, and its lead,
    /// its place in `G_min`, and its redundant mark stay.
    pub(crate) fn replace(&mut self, i: ElementId, poly: Poly<F, D>) {
        debug_assert!(
            i.index() < self.elements.len(),
            "element {i} past the basis's {len}",
            len = self.elements.len()
        );
        debug_assert_eq!(
            poly.lead(),
            self.lead[i.index()],
            "element {i} replaced by a polynomial of another leading monomial"
        );
        self.length[i.index()] = poly.len().narrow();
        self.elements[i.index()] = poly;
    }

    /// `G_min`, in the order made.
    pub(crate) fn min(&self) -> &[ElementId] {
        &self.min
    }

    /// The elements in the order made, `G_min`'s and the redundant ones.
    pub(crate) fn into_elements(self) -> Vec<Poly<F, D>> {
        self.elements
    }
}
