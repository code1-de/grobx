//! Cofactor tracking: for every element g of the basis the cofactors `c_j`
//! with g = Σ `c_j·f_j` over the instance's generators `f_j` (`OBJECTIVES.md`,
//! _Call parameters_). `Tracked` tracks them, in a build with the feature
//! `cofactors` alone. `NoCofactors` tracks nothing, and with it the hooks
//! compile to nothing.
//!
//! The trait is the crate's own: its hooks name the engine's monomial table
//! and its budget.

#[cfg(feature = "cofactors")]
use rand_core::Rng;

use crate::Stop;
use crate::budget::Budget;
#[cfg(feature = "cofactors")]
use crate::budget::Site;
use crate::field::Field;
use crate::halt::Halt;
use crate::id::{KeepId, MonoId, RowId, RowSource};
#[cfg(feature = "cofactors")]
use crate::module::Rank1;
use crate::monomial::Format;
#[cfg(feature = "cofactors")]
use crate::narrow::{self, Narrow};
use crate::order::Order;
use crate::poly::Sparse;
#[cfg(feature = "cofactors")]
use crate::poly::{Exponents, Poly};
#[cfg(feature = "cofactors")]
use crate::sort::{KEY, RUN, payload, same_key, sort_runs};
use crate::table::Table;
#[cfg(feature = "cofactors")]
use crate::table::monomial_units;
#[cfg(feature = "cofactors")]
use crate::text::write_poly;

// text-stop-latency.5 design-checked -- each entry of a cofactor text, over GF(p) and over Q, spends ENTRY, 16, units beside its terms, not a tick per entry nor nothing (not measured)
/// Units an entry spends beside its terms: an entry of a cofactor text or
/// of a replay's cofactor image, and an entry a composition or a lift
/// reads.
pub(crate) const ENTRY: usize = 16;

// cofactor-stop-latency.3 tuned-checked -- a composition and a lift spend TERM, 64, units a term after each chunk of TERMS, 2^12, terms of an entry, not a spend per entry of its whole length nor 16 units a term (log 0058-1)
/// Units of work per term of a composition or a lift: one product in the
/// cofactor table, summed.
#[cfg(feature = "cofactors")]
pub(crate) const TERM: usize = 64;

/// The terms of an entry a composition or a lift sums between two spends.
#[cfg(feature = "cofactors")]
pub(crate) const TERMS: usize = 1 << 12;

/// A quotient of a division by a basis: the index of the basis element,
/// and the quotient's terms, (monomial of the run's table, coefficient).
pub(crate) type Quotient<F> = (u32, Vec<(MonoId, <F as Field>::Coef)>);

/// What a run tells its cofactor tracking. A matrix's rows are t·g for
/// elements g of the basis. The reduction of a row subtracts multiples of
/// pivots from it, each a row of the matrix or a row kept before, and a row
/// that is kept becomes a new element.
pub(crate) trait Cofactors<F: Field, M: Format> {
    /// The cofactors of an element on its way into the basis.
    type Element;

    // fglm-lex.14 design-checked -- the division and the composition run where the cofactor tracking keeps cofactors, Cofactors::TRACKS, a constant of the trait, not a runtime flag or a division in every lex call (not measured)
    /// Whether the tracking keeps cofactors: false for `NoCofactors`.
    const TRACKS: bool;

    /// The element made from generator j, made monic by the factor s. Its
    /// entries spend a unit of `budget` a generator. A stop where the
    /// budget's check stops the call.
    fn input(&mut self, j: usize, s: F::Coef, budget: &mut Budget) -> Result<Self::Element, Stop>;
    /// What makes the cofactors of an element whose cofactors are not
    /// tracked: Some for `NoCofactors`, None for a tracking that keeps
    /// them, which has no such element. A carry to the plain run takes one
    /// for each element it carries (signature-run-losses.1).
    fn untracked() -> Option<fn() -> Self::Element>;
    /// The next element of the basis, in the order made, has these
    /// cofactors.
    fn push(&mut self, element: Self::Element);
    /// A matrix of `rows` rows is reduced next, and the rows kept so far are
    /// forgotten.
    fn matrix(&mut self, rows: usize);
    /// The reduction of row r begins. A row begun before and not kept
    /// reduced to zero.
    fn start(&mut self, r: RowId);
    // call-stop-latency.66 design-checked -- the hooks input, start_kept, sub_kept, and keep take the call's budget and return its stop, which NoCofactors' Ok(()) compiles to nothing, not each hook returning the units it walked for its caller to spend (log 0138-7)
    /// The row being reduced begins as kept row k. Kept row k is read no
    /// more: the row replaces it once kept. Spends a unit of `budget` per
    /// row of k's combination; a stop where the budget's check stops the
    /// call.
    fn start_kept(&mut self, k: KeepId, budget: &mut Budget) -> Result<(), Stop>;
    /// The row being reduced loses v times row q of the matrix.
    fn sub_row(&mut self, q: RowId, v: F::Coef);
    /// The row being reduced loses v times kept row k. Spends a unit of
    /// `budget` per row of k's combination; a stop where the budget's check
    /// stops the call.
    fn sub_kept(&mut self, k: KeepId, v: F::Coef, budget: &mut Budget) -> Result<(), Stop>;
    /// The row being reduced is multiplied by s.
    fn scale(&mut self, s: F::Coef);
    /// The row being reduced is kept, as the next kept row. Spends a unit
    /// of `budget` per row its combination touched; a stop where the
    /// budget's check stops the call.
    fn keep(&mut self, budget: &mut Budget) -> Result<(), Stop>;
    /// The cofactors of kept row k, from those of the elements its matrix's
    /// rows `rows` multiply, over the run's table `table`. Each row of the
    /// combination spends what its multiplier's interning spends,
    /// `table::monomial_units`, and `ENTRY` units of `budget` per
    /// generator, and `TERM` units a term of the entry it reads, after each
    /// `TERMS` of them. A stop where the budget's check stops the call.
    fn compose(
        &mut self,
        k: KeepId,
        rows: &[RowSource],
        table: &Table<M>,
        budget: &mut Budget,
    ) -> Result<Self::Element, Halt>;
    /// The cofactors of `Σ_u` `q_u·g_u`: `quotients` holds per element u, by
    /// index into `elements`, the terms of `q_u`, monomials of the run's table
    /// `table`; `elements[u]` holds `g_u`'s cofactors. Each term of a
    /// quotient spends what its monomial's interning spends,
    /// `table::monomial_units`, and `ENTRY` units of `budget` per
    /// generator, and `TERM` units a term of the entry it reads, after each
    /// `TERMS` of them. A stop where the budget's check stops the call.
    fn lift(
        &mut self,
        quotients: &[Quotient<F>],
        elements: &[Self::Element],
        table: &Table<M>,
        budget: &mut Budget,
    ) -> Result<Self::Element, Halt>;
    // text-stop-latency.1 design-checked -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
    /// Writes the cofactor text of these elements, in this order, with the
    /// instance's variable names. Each entry spends `ENTRY` units of
    /// `budget`, its sort what `sort_ids` spends and `KEY` units a term
    /// gathered, and its text what `write_poly` spends. A stop where the
    /// budget's check stops the call, the text then cut short.
    fn write<O: Order<M>>(
        &self,
        out: &mut String,
        names: &[&str],
        order: &O,
        elements: &[Self::Element],
        budget: &mut Budget,
    ) -> Result<(), Stop>;
    /// Per element, per generator, its cofactor as the lift reads it, the
    /// terms in decreasing order of `order`; empty for `NoCofactors`. Each
    /// entry spends `ENTRY` units of `budget`, its sort what `sort_ids`
    /// spends, and `KEY` units and its variables a term gathered. A stop
    /// where the budget's check stops the call.
    fn image<O: Order<M>>(
        &self,
        order: &O,
        elements: &[Self::Element],
        budget: &mut Budget,
    ) -> Result<Vec<Vec<Sparse>>, Stop>;
}

// f4-gfp.25 practice-checked:monomorphization -- the cofactor tracking is a type parameter of run, and its empty implementation a zero-sized type with empty methods, not a runtime flag tested at each hook
/// No cofactors: every hook is empty.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCofactors;

impl<F: Field, M: Format> Cofactors<F, M> for NoCofactors {
    type Element = ();

    const TRACKS: bool = false;

    #[inline(always)]
    fn input(&mut self, _: usize, _: F::Coef, _: &mut Budget) -> Result<(), Stop> {
        Ok(())
    }

    fn untracked() -> Option<fn()> {
        Some(|| ())
    }

    #[inline(always)]
    fn push(&mut self, (): ()) {}

    #[inline(always)]
    fn matrix(&mut self, _: usize) {}

    #[inline(always)]
    fn start(&mut self, _: RowId) {}

    #[inline(always)]
    fn start_kept(&mut self, _: KeepId, _: &mut Budget) -> Result<(), Stop> {
        Ok(())
    }

    #[inline(always)]
    fn sub_row(&mut self, _: RowId, _: F::Coef) {}

    #[inline(always)]
    fn sub_kept(&mut self, _: KeepId, _: F::Coef, _: &mut Budget) -> Result<(), Stop> {
        Ok(())
    }

    #[inline(always)]
    fn scale(&mut self, _: F::Coef) {}

    #[inline(always)]
    fn keep(&mut self, _: &mut Budget) -> Result<(), Stop> {
        Ok(())
    }

    #[inline(always)]
    fn compose(
        &mut self,
        _: KeepId,
        _: &[RowSource],
        _: &Table<M>,
        _: &mut Budget,
    ) -> Result<(), Halt> {
        Ok(())
    }

    #[inline(always)]
    fn lift(
        &mut self,
        _: &[Quotient<F>],
        _: &[()],
        _: &Table<M>,
        _: &mut Budget,
    ) -> Result<(), Halt> {
        Ok(())
    }

    #[inline(always)]
    fn write<O: Order<M>>(
        &self,
        _: &mut String,
        _: &[&str],
        _: &O,
        _: &[()],
        _: &mut Budget,
    ) -> Result<(), Stop> {
        Ok(())
    }

    fn image<O: Order<M>>(
        &self,
        _: &O,
        _: &[()],
        _: &mut Budget,
    ) -> Result<Vec<Vec<Sparse>>, Stop> {
        Ok(Vec::new())
    }
}

/// The cofactors of one element: per generator j its cofactor, as terms
/// (monomial of the cofactor table, nonzero coefficient) in no order.
#[cfg(feature = "cofactors")]
pub(crate) struct CofactorRow<F: Field> {
    terms: Vec<(MonoId, F::Coef)>,
    /// Generator j's terms end at `ends[j]`.
    ends: Vec<u32>,
}

#[cfg(feature = "cofactors")]
impl<F: Field> CofactorRow<F> {
    fn entry(&self, j: usize) -> &[(MonoId, F::Coef)] {
        let start = if j == 0 { 0 } else { self.ends[j - 1] };
        &self.terms[start as usize..self.ends[j] as usize]
    }
}

// f4-gfp.23 design-checked -- cofactors are tracked by combinations, a row being reduced keeping the sparse combination of the matrix's rows it is made of and a new element's cofactors composed once from it, not every row operation applied to cofactor polynomials as it happens (log 0004-4)
/// Cofactors tracked by combinations: a row being reduced keeps the
/// combination of its matrix's rows it is made of, dense over them, a kept
/// row keeps it sparse, and a new element's
/// cofactors are composed once, from its combination and the cofactors of
/// the elements those rows multiply.
#[cfg(feature = "cofactors")]
pub struct Tracked<F: Field, M: Format> {
    field: F,
    /// The cofactors' monomials, and the sum of a cofactor composed.
    sums: Sums<F, M>,
    /// The monomial 1 of the cofactor table.
    one: MonoId,
    /// How many generators the instance has, zero ones included.
    generators: usize,
    /// The cofactors of every element of the basis, in the order made.
    elements: Vec<CofactorRow<F>>,
    // f4-gfp.46 cited-checked:gilbert-1988 §2 -- u_j is computed as a dense n-vector, so that a multiple of a column is subtracted from it in constant time per nonzero in that column
    /// The combination of the row being reduced, dense over the matrix's
    /// rows. `touched` holds every row where it may be nonzero, some twice.
    comb: Vec<F::Coef>,
    touched: Vec<RowId>,
    /// What the row being reduced is multiplied by, the combination aside.
    factor: F::Coef,
    /// The combination of each row kept from the current matrix: (row of the
    /// matrix, nonzero coefficient).
    kept: Vec<Vec<(RowId, F::Coef)>>,
    /// The multipliers of a combination's rows, or a quotient's monomials,
    /// in the cofactor table.
    multipliers: Vec<MonoId>,
}

/// The cofactor table, and the cofactor of one generator while `compose`
/// or `lift` sums it.
#[cfg(feature = "cofactors")]
struct Sums<F: Field, M: Format> {
    // f4-gfp.24 design-checked -- cofactor monomials have a table of their own, of the same format, not the engine's (not measured)
    /// The cofactors' monomials.
    table: Table<M>,
    // f4-gfp.47 practice-checked:sparse accumulator -- a cofactor is summed into an array over the cofactor table's monomial ids with a list of the ids touched, not into a map keyed by monomial
    /// One generator's cofactor while it is summed, dense over `table`,
    /// and every monomial where it may be nonzero, some twice.
    sum: Vec<F::Coef>,
    summed: Vec<MonoId>,
    /// The exponents of a monomial on its way into `table`.
    exponents: Vec<u32>,
}

#[cfg(feature = "cofactors")]
impl<F: Field, M: Format> Sums<F, M> {
    /// The monomial `t` of the run's table `from` in the cofactor table:
    /// `table::monomial_units` units of `budget`, and a growth of the table
    /// spent there.
    #[inline(always)]
    fn intern(&mut self, from: &Table<M>, t: MonoId, budget: &mut Budget) -> Result<MonoId, Halt> {
        for (i, e) in self.exponents.iter_mut().enumerate() {
            *e = from.exponent(t, i);
        }
        // call-stop-latency.65 design-checked -- a multiplier of a composition or a quotient's monomial of a lift, interned into the cofactor table, spends table::monomial_units, as the parse's monomials, not unspent beside the ENTRY units per generator its row spends (log 0138-7)
        budget.spend(monomial_units(self.exponents.len()), Site::Cofactors)?;
        Ok(self.table.intern(&self.exponents, budget)?)
    }

    /// Adds the terms `entry` times c·s to the sum, s a monomial of the
    /// cofactor table: `ENTRY` units of `budget` for the entry, and `TERM`
    /// units a term after each `TERMS` of them.
    #[inline(always)]
    fn add(
        &mut self,
        field: F,
        s: MonoId,
        c: F::Coef,
        entry: &[(MonoId, F::Coef)],
        budget: &mut Budget,
    ) -> Result<(), Halt> {
        let zero = F::Coef::default();
        // cofactor-stop-latency.2 design-checked -- a composition and a lift spend ENTRY, 16, units per entry they read, in place of the tick per row and generator, and per term of a quotient and generator, not the tick kept beside the spends (not measured)
        budget.spend(ENTRY, Site::Cofactors)?;
        for chunk in entry.chunks(TERMS) {
            for &(t, a) in chunk {
                // The product's degree is checked against the format.
                let st = self.table.mul(s, t, budget)?;
                if st.index() >= self.sum.len() {
                    self.sum.resize(self.table.len(), zero);
                }
                let x = &mut self.sum[st.index()];
                if *x == zero {
                    self.summed.push(st);
                }
                *x = field.add(*x, field.mul(c, a));
            }
            // cofactor-stop-latency.3 tuned-checked -- a composition and a lift spend TERM, 64, units a term after each chunk of TERMS, 2^12, terms of an entry, not a spend per entry of its whole length nor 16 units a term (log 0058-1)
            budget.spend(TERM * chunk.len(), Site::Cofactors)?;
        }
        Ok(())
    }

    /// Moves the sum's nonzero terms to `out` as its next generator's
    /// cofactor, and leaves the sum 0.
    #[inline(always)]
    fn drain(&mut self, out: &mut CofactorRow<F>) -> Result<(), Halt> {
        let zero = F::Coef::default();
        for &u in &self.summed {
            // A monomial summed twice is read once: the first read clears
            // it.
            let x = std::mem::take(&mut self.sum[u.index()]);
            if x != zero {
                out.terms.push((u, x));
            }
        }
        self.summed.clear();
        out.ends.push(narrow::id(out.terms.len())?);
        Ok(())
    }
}

#[cfg(feature = "cofactors")]
impl<F: Field, M: Format> Tracked<F, M> {
    /// Tracking for an instance of `generators` generators, whose cofactor
    /// table draws its hash multipliers from `rng`.
    pub(crate) fn new(field: F, format: M, generators: usize, rng: &mut impl Rng) -> Tracked<F, M> {
        let mut table = Table::new(format, rng);
        #[allow(clippy::expect_used, reason = "the table is new, and 1 has degree 0")]
        let one = table
            .intern(&vec![0; format.variables()], &mut Budget::unbounded())
            .expect("the monomial 1 fits every format and the empty table, which it does not grow");
        Tracked {
            field,
            sums: Sums {
                table,
                sum: Vec::new(),
                summed: Vec::new(),
                exponents: vec![0; format.variables()],
            },
            one,
            generators,
            elements: Vec::new(),
            comb: Vec::new(),
            touched: Vec::new(),
            factor: field.coef(1),
            kept: Vec::new(),
            multipliers: Vec::new(),
        }
    }

    /// Adds c to the combination at row q.
    #[inline]
    fn add(&mut self, q: RowId, c: F::Coef) {
        let zero = F::Coef::default();
        let x = &mut self.comb[q.index()];
        if *x == zero {
            self.touched.push(q);
        }
        *x = self.field.add(*x, c);
    }

    /// Clears the combination.
    fn clear(&mut self) {
        for &q in &self.touched {
            self.comb[q.index()] = F::Coef::default();
        }
        self.touched.clear();
    }
}

#[cfg(feature = "cofactors")]
impl<F: Field, M: Format> Cofactors<F, M> for Tracked<F, M> {
    type Element = CofactorRow<F>;

    const TRACKS: bool = true;

    fn input(&mut self, j: usize, s: F::Coef, budget: &mut Budget) -> Result<CofactorRow<F>, Stop> {
        let ends = (0..self.generators).map(|i| u32::from(i >= j)).collect();
        // call-stop-latency.33 design-checked -- the cofactor tracking spends the combinations its kept rows' walks pass, a unit a row, a unit a generator of an input's entries, and a term an image gathers its variables beside KEY, not the reducer's spends of the row's entries alone (log 0138-7)
        budget.spend(self.generators, Site::Cofactors)?;
        Ok(CofactorRow {
            terms: vec![(self.one, s)],
            ends,
        })
    }

    fn untracked() -> Option<fn() -> CofactorRow<F>> {
        None
    }

    fn push(&mut self, element: CofactorRow<F>) {
        self.elements.push(element);
    }

    fn matrix(&mut self, rows: usize) {
        self.clear();
        self.comb.clear();
        self.comb.resize(rows, F::Coef::default());
        self.kept.clear();
    }

    fn start(&mut self, r: RowId) {
        self.clear();
        self.factor = self.field.coef(1);
        self.add(r, self.field.coef(1));
    }

    // order-family.21 design-checked -- start_kept moves kept row k's combination into the row being reduced, not a copy that leaves it in place (not measured)
    fn start_kept(&mut self, k: KeepId, budget: &mut Budget) -> Result<(), Stop> {
        self.clear();
        self.factor = self.field.coef(1);
        // Nothing reads kept row k again.
        let kept = std::mem::take(&mut self.kept[k.index()]);
        for &(q, c) in &kept {
            self.add(q, c);
        }
        // call-stop-latency.33 design-checked -- the cofactor tracking spends the combinations its kept rows' walks pass, a unit a row, a unit a generator of an input's entries, and a term an image gathers its variables beside KEY, not the reducer's spends of the row's entries alone (log 0138-7)
        budget.spend(kept.len(), Site::Cofactors)
    }

    fn sub_row(&mut self, q: RowId, v: F::Coef) {
        self.add(q, self.field.neg(v));
    }

    fn sub_kept(&mut self, k: KeepId, v: F::Coef, budget: &mut Budget) -> Result<(), Stop> {
        let f = self.field;
        let minus = f.neg(v);
        // The kept rows stay as they are while the row is reduced.
        let kept = std::mem::take(&mut self.kept[k.index()]);
        for &(q, c) in &kept {
            self.add(q, f.mul(minus, c));
        }
        let walked = kept.len();
        self.kept[k.index()] = kept;
        // call-stop-latency.33 design-checked -- the cofactor tracking spends the combinations its kept rows' walks pass, a unit a row, a unit a generator of an input's entries, and a term an image gathers its variables beside KEY, not the reducer's spends of the row's entries alone (log 0138-7)
        budget.spend(walked, Site::Cofactors)
    }

    fn scale(&mut self, s: F::Coef) {
        self.factor = self.field.mul(self.factor, s);
    }

    fn keep(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let f = self.field;
        let zero = F::Coef::default();
        let mut combination = Vec::new();
        for &q in &self.touched {
            // A row touched twice is read once: the first read clears it.
            let x = std::mem::take(&mut self.comb[q.index()]);
            if x != zero {
                combination.push((q, f.mul(x, self.factor)));
            }
        }
        let walked = self.touched.len();
        self.touched.clear();
        self.kept.push(combination);
        // call-stop-latency.33 design-checked -- the cofactor tracking spends the combinations its kept rows' walks pass, a unit a row, a unit a generator of an input's entries, and a term an image gathers its variables beside KEY, not the reducer's spends of the row's entries alone (log 0138-7)
        budget.spend(walked, Site::Cofactors)
    }

    fn compose(
        &mut self,
        k: KeepId,
        rows: &[RowSource],
        table: &Table<M>,
        budget: &mut Budget,
    ) -> Result<CofactorRow<F>, Halt> {
        let combination = &self.kept[k.index()];
        // Each row's multiplier, a monomial of the run's table, in the
        // cofactor table.
        self.multipliers.clear();
        for &(q, _) in combination {
            let t = rows[q.index()].multiplier;
            self.multipliers.push(self.sums.intern(table, t, budget)?);
        }
        let mut out = CofactorRow {
            terms: Vec::new(),
            ends: Vec::with_capacity(self.generators),
        };
        for j in 0..self.generators {
            for (&(q, c), &t) in combination.iter().zip(&self.multipliers) {
                let element = &self.elements[rows[q.index()].element.index()];
                self.sums.add(self.field, t, c, element.entry(j), budget)?;
            }
            self.sums.drain(&mut out)?;
        }
        Ok(out)
    }

    fn lift(
        &mut self,
        quotients: &[Quotient<F>],
        elements: &[CofactorRow<F>],
        table: &Table<M>,
        budget: &mut Budget,
    ) -> Result<CofactorRow<F>, Halt> {
        // Each quotient's monomials, of the run's table, in the cofactor
        // table, one after the other.
        self.multipliers.clear();
        for (_, q) in quotients {
            for &(s, _) in q {
                self.multipliers.push(self.sums.intern(table, s, budget)?);
            }
        }
        let mut out = CofactorRow {
            terms: Vec::new(),
            ends: Vec::with_capacity(self.generators),
        };
        for j in 0..self.generators {
            let mut multipliers = self.multipliers.iter();
            for (u, q) in quotients {
                let entry = elements[*u as usize].entry(j);
                for (&(_, c), &s) in q.iter().zip(&mut multipliers) {
                    self.sums.add(self.field, s, c, entry, budget)?;
                }
            }
            self.sums.drain(&mut out)?;
        }
        Ok(out)
    }

    fn write<O: Order<M>>(
        &self,
        out: &mut String,
        names: &[&str],
        order: &O,
        elements: &[CofactorRow<F>],
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        out.push('[');
        let mut entry: Poly<F, Rank1> = Poly {
            monomials: Vec::new(),
            coefs: Vec::new(),
            position: (),
        };
        let mut terms: Vec<(MonoId, F::Coef)> = Vec::new();
        for (i, element) in elements.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push('[');
            for j in 0..self.generators {
                if j > 0 {
                    out.push(',');
                }
                // text-stop-latency.5 design-checked -- each entry of a cofactor text, over GF(p) and over Q, spends ENTRY, 16, units beside its terms, not a tick per entry nor nothing (not measured)
                budget.spend(ENTRY, Site::Cofactors)?;
                terms.clear();
                terms.extend_from_slice(element.entry(j));
                // f4-gfp.48 design-checked -- an element's cofactors stay in the order summed and are sorted when the cofactor text is written, not after each composition (not measured)
                sort_terms(&self.sums.table, order, &mut terms, budget)?;
                entry.monomials.clear();
                entry.coefs.clear();
                for &(u, c) in &terms {
                    entry.monomials.push(u);
                    entry.coefs.push(c);
                }
                // text-stop-latency.1 design-checked -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
                write_poly(out, names, &self.sums.table, &self.field, &entry, budget)?;
            }
            out.push(']');
        }
        out.push_str("]:\n");
        Ok(())
    }

    fn image<O: Order<M>>(
        &self,
        order: &O,
        elements: &[CofactorRow<F>],
        budget: &mut Budget,
    ) -> Result<Vec<Vec<Sparse>>, Stop> {
        let n = self.sums.table.format().variables();
        let mut image = Vec::with_capacity(elements.len());
        let mut ids: Vec<MonoId> = Vec::new();
        let mut term = vec![0; n];
        for element in elements {
            let mut row = Vec::with_capacity(self.generators);
            for j in 0..self.generators {
                // cofactor-stop-latency.4 design-checked -- a replay's cofactor image sorts each entry by sort_ids and gathers its terms RUN at a time, spending ENTRY an entry and KEY a term, not one comparison sort per entry and one pass, unspent (log 0055-2, 0058-1)
                budget.spend(ENTRY, Site::Cofactors)?;
                let entry = element.entry(j);
                ids.clear();
                ids.extend(entry.iter().map(|t| t.0));
                // As `write` sorts them (f4-gfp.48).
                let at = sort_ids(&self.sums.table, order, &ids, budget)?;
                let mut sparse = Sparse {
                    exponents: Exponents::with_capacity(entry.len() * n),
                    coefs: Vec::with_capacity(entry.len()),
                };
                for chunk in at.chunks(RUN) {
                    for &i in chunk {
                        let (u, c) = entry[i as usize];
                        // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
                        self.sums.table.unpack(u, &mut term);
                        sparse.exponents.extend_from_slice(&term);
                        sparse.coefs.push(self.field.value(c));
                    }
                    // call-stop-latency.33 design-checked -- the cofactor tracking spends the combinations its kept rows' walks pass, a unit a row, a unit a generator of an input's entries, and a term an image gathers its variables beside KEY, not the reducer's spends of the row's entries alone (log 0138-7)
                    budget.spend(chunk.len() * (KEY + n), Site::Cofactors)?;
                }
                row.push(sparse);
            }
            image.push(row);
        }
        Ok(image)
    }
}

// text-stop-latency.9 design-checked -- the sort orders the entry's monomial ids and the terms follow by a gather, the keys' decreasing sort one function for every format and order, not the terms sorted with their coefficients (log 0057-1)
/// Sorts the terms by decreasing monomial: their monomial ids by
/// `sort_ids`, then the terms gathered in that order, `RUN` at a time, each
/// spending `KEY` units a term. A stop where the budget's check stops the
/// call, the terms then as they were.
#[cfg(feature = "cofactors")]
fn sort_terms<M: Format, O: Order<M>, C: Copy>(
    table: &Table<M>,
    order: &O,
    terms: &mut Vec<(MonoId, C)>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let ids: Vec<MonoId> = terms.iter().map(|t| t.0).collect();
    let at = sort_ids(table, order, &ids, budget)?;
    let mut sorted: Vec<(MonoId, C)> = Vec::with_capacity(terms.len());
    // text-stop-latency.8 design-checked -- the runs of equal keys are sorted through sort_runs, and the keys built and the terms gathered RUN at a time, each spent, not the ties sorted as matrix::ties sorts them nor the keys and the gather in one pass each, unspent (log 0057-1)
    for chunk in at.chunks(RUN) {
        sorted.extend(chunk.iter().map(|&i| terms[i as usize]));
        budget.spend(chunk.len() * KEY, Site::Cofactors)?;
    }
    *terms = sorted;
    Ok(())
}

/// The indices of `ids`, distinct monomials, in decreasing order of their
/// monomials, the order `sort_unstable_by` returns on them. In a keyed
/// order they are sorted by their keys, built `RUN` at a time and spending
/// `KEY` units a key, and each run of equal keys by the order's comparison;
/// in any other order by the comparison. Each sort is `sort_runs`'s. A stop
/// where the budget's check stops the call.
#[cfg(feature = "cofactors")]
pub(crate) fn sort_ids<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    ids: &[MonoId],
    budget: &mut Budget,
) -> Result<Vec<u32>, Stop> {
    // text-stop-latency.6 design-checked -- a cofactor entry over GF(p) is sorted in runs past RUN elements by sort_runs, not one sort spent after it (log 0057-1)
    if !O::KEYED {
        let mut at: Vec<u32> = (0..ids.len().narrow()).collect();
        let cmp = |&a: &u32, &b: &u32| table.cmp(order, ids[b as usize], ids[a as usize]);
        sort_runs(&mut at, cmp, budget, Site::Cofactors)?;
        return Ok(at);
    }
    // text-stop-latency.7 cited-checked:bachmann-1998 §4, attributes it to Macaulay [4] -- a monomial is compared by its rank, one integer whose order is the monomial order
    // matrix-sort-time.5 design-checked -- an element of a keyed sort is one u128, the key's first 96 bits above a 32-bit payload, the id or the row, not the key's first 128 bits beside the payload in 24 bytes (log 0032-1)
    // Here the payload is the term's index in `ids`.
    let mut keyed: Vec<u128> = Vec::with_capacity(ids.len());
    // text-stop-latency.8 design-checked -- the runs of equal keys are sorted through sort_runs, and the keys built and the terms gathered RUN at a time, each spent, not the ties sorted as matrix::ties sorts them nor the keys and the gather in one pass each, unspent (log 0057-1)
    for (k, chunk) in ids.chunks(RUN).enumerate() {
        let start = k * RUN;
        let keys = chunk.iter().enumerate();
        keyed.extend(keys.map(|(i, &u)| table.key(order, u) | (start + i) as u128));
        budget.spend(chunk.len() * KEY, Site::Cofactors)?;
    }
    decreasing(&mut keyed, budget)?;
    // matrix-sort-time.6 design-checked -- elements whose first 96 bits agree are sorted by the order's comparison, not by the key's next 96 bits built for their run (log 0032-1)
    for run in keyed.chunk_by_mut(|&a, &b| same_key(a, b)) {
        if run.len() > 1 {
            let cmp = |&a: &u128, &b: &u128| {
                table.cmp(order, ids[payload(b) as usize], ids[payload(a) as usize])
            };
            sort_runs(run, cmp, budget, Site::Cofactors)?;
        }
    }
    Ok(keyed.iter().map(|&k| payload(k)).collect())
}

// text-stop-latency.9 design-checked -- the sort orders the entry's monomial ids and the terms follow by a gather, the keys' decreasing sort one function for every format and order, not the terms sorted with their coefficients (log 0057-1)
/// Sorts the keyed elements into decreasing order by `sort_runs`. A stop
/// where the budget's check stops the call.
#[cfg(feature = "cofactors")]
#[inline(never)]
fn decreasing(keyed: &mut [u128], budget: &mut Budget) -> Result<(), Stop> {
    sort_runs(keyed, |a, b| b.cmp(a), budget, Site::Cofactors)
}
