//! Polynomials over the monomial table, and the basis an F4 run builds.

use crate::field::Field;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::Format;
use crate::narrow::{Narrow, Size};
use crate::order::Order;
use crate::table::Table;

/// A polynomial: monomial ids of one table in decreasing order, with their
/// coefficients, none zero, in the same order.
pub(crate) struct Poly<F: Field, D: Module> {
    pub(crate) monomials: Vec<MonoId>,
    pub(crate) coefs: Vec<F::Coef>,
    // f4-gfp.40 design -- a row carries its module position as the module's type Position, () for rank 1, not a u32 that is always 0 (not measured)
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
    /// zero terms dropped. `terms` is left empty.
    pub(crate) fn from_terms<M: Format, O: Order<M>>(
        terms: &mut Vec<(MonoId, F::Coef)>,
        field: &F,
        table: &Table<M>,
        order: &O,
        position: D::Position,
    ) -> Poly<F, D> {
        terms.sort_unstable_by(|a, b| table.cmp(order, b.0, a.0));
        let mut monomials = Vec::with_capacity(terms.len());
        let mut coefs: Vec<F::Coef> = Vec::with_capacity(terms.len());
        for (m, c) in terms.drain(..) {
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
        if coefs.last() == Some(&F::Coef::default()) {
            monomials.pop();
            coefs.pop();
        }
        Poly {
            monomials,
            coefs,
            position,
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }

    /// The number of terms.
    pub(crate) fn len(&self) -> usize {
        self.monomials.len()
    }

    // table-read-time.4 design -- the basis's element, leading monomial, and length, and a nonzero polynomial's leading monomial, read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
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

    /// Marks these elements redundant, and takes them out of `G_min`.
    fn mark_redundant(&mut self, elements: &[ElementId]) {
        for &i in elements {
            self.redundant[i.index()] = true;
        }
        let redundant = &self.redundant;
        self.min.retain(|&i| !redundant[i.index()]);
    }

    /// Adds a nonzero monic element and keeps `G_min`: its index, or `Size`
    /// as `push` returns it. The element
    /// joins `G_min` where no leading monomial of `G_min` divides its own, and
    /// the elements whose leading monomial its own divides leave it.
    pub(crate) fn push_min<M: Format>(
        &mut self,
        table: &Table<M>,
        poly: Poly<F, D>,
    ) -> Result<ElementId, Size> {
        let lead = poly.lead();
        let divided = self.min.iter().any(|&i| table.divides(self.lead(i), lead));
        if divided {
            return self.push(poly, true);
        }
        let gone: Vec<ElementId> = self
            .min
            .iter()
            .copied()
            .filter(|&i| table.divides(lead, self.lead(i)))
            .collect();
        self.mark_redundant(&gone);
        self.push(poly, false)
    }

    pub(crate) fn len(&self) -> usize {
        self.elements.len()
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

    // table-read-time.4 design -- the basis's element, leading monomial, and length, and a nonzero polynomial's leading monomial, read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
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

    /// `G_min`, in the order made.
    pub(crate) fn min(&self) -> &[ElementId] {
        &self.min
    }
}
