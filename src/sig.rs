//! Signatures (lairez-2022 §3, eder-2017 §4): each basis element's, the
//! known syzygy signatures, and the pending ones.

use std::cmp::Ordering;
use std::collections::HashSet;

use crate::field::Field;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::{Format, Overflow};
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::Basis;
use crate::table::{Limit, Table, find_passing};

/// A signature `m⊗e_index` in Schreyer's shifted form: input i has
/// `lm(f_i)⊗e_i` (signature-pruning.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Sig {
    /// A monomial of the run's table.
    pub(crate) monomial: MonoId,
    /// The input's position among the nonzero generators.
    pub(crate) index: u32,
}

impl Sig {
    // signature-pruning.2 cited: gao-2016 §3 -- the g-weighted order followed by POT (g2) is the clear winner of four signature orders; lairez-2022 Remark 25 and Example 26 -- sig g_i = lm g_i ⊗ e_i with TOP, which eder-2017 calls lt-pot
    /// Schreyer's order: the monomials by `order`, then the indices.
    pub(crate) fn cmp<M: Format, O: Order<M>>(
        self,
        other: Sig,
        table: &Table<M>,
        order: &O,
    ) -> Ordering {
        table
            .cmp(order, self.monomial, other.monomial)
            .then(self.index.cmp(&other.index))
    }

    /// self divides other: the same index, and the monomial divides.
    #[cfg(test)]
    pub(crate) fn divides<M: Format>(self, other: Sig, table: &Table<M>) -> bool {
        self.index == other.index && table.divides(self.monomial, other.monomial)
    }

    /// t·self.
    pub(crate) fn mul<M: Format>(self, t: MonoId, table: &mut Table<M>) -> Result<Sig, Limit> {
        Ok(Sig {
            monomial: table.mul(t, self.monomial)?,
            index: self.index,
        })
    }

    /// The degree of the monomial, the sugar degree (signature-pruning.3).
    pub(crate) fn degree<M: Format>(self, table: &Table<M>) -> u32 {
        table.degree(self.monomial)
    }
}

// signature-pruning.21 cited: roune-2012 §3.4 -- store the sig-lead ratio and compare ratios instead of signatures
/// Compares the sig-lead ratios `s_a/l_a` and `s_b/l_b`, monomials of the
/// table, under `order`, as `s_a·l_b` against `s_b·l_a` with their gcd divided
/// out. The two are formed in scratch words and not interned. Overflow
/// where one passes the format.
pub(crate) fn cmp_ratio<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    (s_a, l_a): (MonoId, MonoId),
    (s_b, l_b): (MonoId, MonoId),
) -> Result<Ordering, Overflow> {
    let format = table.format();
    let variables = format.variables();
    // signature-pruning.27 cited: roune-2012 §3.4 -- bA < aB exactly where A/a < B/b, negative exponents allowed
    // A monomial order keeps a comparison when a common factor is divided
    // out, so the two products with their gcd divided out compare as the
    // ratios do.
    let mut exponents = vec![0u32; 2 * variables];
    let (left, right) = exponents.split_at_mut(variables);
    for variable in 0..variables {
        let left_exponent = table.exponent(s_a, variable) + table.exponent(l_b, variable);
        let right_exponent = table.exponent(s_b, variable) + table.exponent(l_a, variable);
        let common = left_exponent.min(right_exponent);
        (left[variable], right[variable]) = (left_exponent - common, right_exponent - common);
    }
    let word_count = format.words();
    let mut words = vec![M::Word::default(); 2 * word_count];
    let (left_words, right_words) = words.split_at_mut(word_count);
    let left_degree = format.encode(left, left_words)?;
    let right_degree = format.encode(right, right_words)?;
    Ok(order.cmp(format, left_words, left_degree, right_words, right_degree))
}

// signature-pruning.20 cited: roune-2012 §4.2 -- the divmask monomial list eliminates around 98% of divisibility checks
/// Monomials of the table, none of which divides another, each with its
/// divisor mask, scanned in a list.
#[derive(Clone, Debug, Default)]
struct Antichain {
    ids: Vec<MonoId>,
    // signature-pair-time.9 design -- the masks stand in an array of their own beside the ids, and a query tests eight of them at a time before it compares exponents (log 0026-1)
    /// The members' masks, in step with `ids`.
    masks: Vec<u32>,
    // signature-pair-time.10 design -- the member that answered the last yes is tried first (log 0026-1)
    /// The position of the member that answered the last yes. A stale
    /// position names another member, or none.
    last: usize,
}

impl Antichain {
    // table-read-time.9 design -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    /// Whether a member divides the monomial of words `m` and mask `mask`.
    fn divides<M: Format>(&mut self, table: &Table<M>, m: &[M::Word], mask: u32) -> bool {
        let not_m = !mask;
        let format = table.format();
        let (ids, masks) = (&self.ids, &self.masks);
        let divides = |k: usize| format.divides(table.monomial(ids[k]), m);
        let last = self.last;
        if last < masks.len() && masks[last] & not_m == 0 && divides(last) {
            return true;
        }
        match find_passing(masks, mask, divides) {
            Some(k) => {
                self.last = k;
                true
            }
            None => false,
        }
    }

    /// Removes the members m divides. The rest keep their order.
    fn drop_multiples<M: Format>(&mut self, table: &Table<M>, m: MonoId) {
        let mask = table.mask(m);
        let format = table.format();
        let mut kept = 0;
        for k in 0..self.ids.len() {
            let (b, mask_b) = (self.ids[k], self.masks[k]);
            if mask & !mask_b != 0 || !format.divides(table.monomial(m), table.monomial(b)) {
                self.ids[kept] = b;
                self.masks[kept] = mask_b;
                kept += 1;
            }
        }
        self.ids.truncate(kept);
        self.masks.truncate(kept);
    }

    /// Adds m, which no member divides and which divides no member.
    fn push<M: Format>(&mut self, table: &Table<M>, m: MonoId) {
        self.ids.push(m);
        self.masks.push(table.mask(m));
    }

    fn shrink_to_fit(&mut self) {
        self.ids.shrink_to_fit();
        self.masks.shrink_to_fit();
    }
}

/// A basis element in the reducer order, with its leading monomial and its
/// number of terms.
#[derive(Clone, Copy, Debug)]
struct Entry {
    element: ElementId,
    lead: MonoId,
    terms: u32,
}

/// The signatures of a run: every basis element's, and the known syzygy
/// signatures.
pub(crate) struct Signatures {
    /// Per element its signature.
    sig: Vec<Sig>,
    // signature-pruning.23 cited: roune-2012 §3.4 -- an integer per basis element orders the sig-lead ratios, so a ratio comparison is an integer comparison; the integers are spaced far apart and rebuilt where a new element would need one between k and k+1
    /// Per element the integer of its (sig-lead ratio, index): the
    /// integers order them, and equal ones share one. No element has 0 or
    /// `u64::MAX`, the bounds where a neighbour is missing.
    tau: Vec<u64>,
    /// The elements by (sig-lead ratio, index, terms, element).
    sorted: Vec<Entry>,
    // matrix-build-time.7 design -- the reducer order's masks stand in an array of their own, in step with the order, and the reducer search tests eight at a time before it tests the candidates' divisibility in order, not a mask in each entry tested one by one; roune-2012 §4.2 eliminates around 98% of divisibility checks by the divmask, one mask per test (signature-pruning.20) (log 0035-1, 0037-1)
    /// The divisor mask of each entry's leading monomial, in step with
    /// `sorted`.
    masks: Vec<u32>,
    /// Per index its elements, in the order made.
    by_index: Vec<Vec<ElementId>>,
    /// Per index the monomials of the known syzygy signatures, minimal.
    syzygies: Vec<Antichain>,
}

impl Signatures {
    /// For a run with `inputs` nonzero inputs.
    pub(crate) fn new(inputs: usize) -> Signatures {
        Signatures {
            sig: Vec::new(),
            tau: Vec::new(),
            sorted: Vec::new(),
            masks: Vec::new(),
            by_index: vec![Vec::new(); inputs],
            syzygies: vec![Antichain::default(); inputs],
        }
    }

    // signature-pruning.10 cited: eder-2017 §13, Algorithm (Symbolic Preprocessing respecting signatures) and Property (1) -- the single reducer of minimal signature, the smaller the signature the bigger the probability it reduces another row; roune-2012 §3.4 -- compare sig-lead ratios instead of signatures
    // rule-breaches.7 design -- the element recorded is the next past those recorded, which `basis` pushed last, not an id the caller passes, with an error or a debug assertion where it is not the next (not measured)
    /// Records the signature of the element `basis` pushed last, nonzero,
    /// the next past the elements recorded. It takes its place in the
    /// reducer order by binary search. Overflow where a ratio comparison
    /// passes the format.
    pub(crate) fn push<F: Field, M: Format, O: Order<M>, D: Module>(
        &mut self,
        table: &Table<M>,
        order: &O,
        basis: &Basis<F, D>,
        sig: Sig,
    ) -> Result<(), Overflow> {
        debug_assert_eq!(
            basis.len(),
            self.sig.len() + 1,
            "the basis holds one element past those recorded"
        );
        let element = ElementId::at(self.sig.len());
        let lead = basis.lead(element);
        let new = Entry {
            element,
            lead,
            terms: basis.length(element).narrow(),
        };
        // The class of an entry: its (ratio, index).
        let class = |e: &Entry| -> Result<Ordering, Overflow> {
            let s = self.sig[e.element.index()];
            let ratio = cmp_ratio(table, order, (s.monomial, e.lead), (sig.monomial, lead))?;
            Ok(ratio.then(s.index.cmp(&sig.index)))
        };
        // The first entry after the new one. An entry of the same class and
        // terms comes before it, as its element was made first.
        let (mut lo, mut hi) = (0, self.sorted.len());
        while lo < hi {
            let mid = usize::midpoint(lo, hi);
            let e = &self.sorted[mid];
            let before = class(e)?.then(e.terms.cmp(&new.terms)).is_le();
            if before {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let p = lo;
        let left = p.checked_sub(1).map(|k| self.sorted[k]);
        let right = self.sorted.get(p).copied();
        let mut tau = None;
        for e in left.iter().chain(&right) {
            if class(e)?.is_eq() {
                tau = Some(self.tau[e.element.index()]);
                break;
            }
        }
        self.sig.push(sig);
        self.sorted.insert(p, new);
        self.masks.insert(p, table.mask(lead));
        self.by_index[sig.index as usize].push(element);
        if let Some(t) = tau {
            self.tau.push(t);
            return Ok(());
        }
        let below = left.map_or(0, |e| self.tau[e.element.index()]);
        let above = right.map_or(u64::MAX, |e| self.tau[e.element.index()]);
        self.tau.push(below + (above - below) / 2);
        if above - below < 2 {
            self.respace(p);
        }
        Ok(())
    }

    /// Gives the classes along `sorted` integers spaced evenly over u64.
    /// `sorted[p]` is the new element, of a class of its own, and its
    /// integer is a placeholder.
    fn respace(&mut self, p: usize) {
        let sorted = &self.sorted;
        let tau = &mut self.tau;
        let starts =
            |k: usize, previous: u64, old: u64| k == 0 || k == p || k == p + 1 || old != previous;
        let (mut classes, mut previous) = (0u64, 0u64);
        for (k, e) in sorted.iter().enumerate() {
            let old = tau[e.element.index()];
            classes += u64::from(starts(k, previous, old));
            previous = old;
        }
        let spacing = u64::MAX / (classes + 1);
        let (mut class, mut previous) = (0u64, 0u64);
        for (k, e) in sorted.iter().enumerate() {
            let old = tau[e.element.index()];
            class += u64::from(starts(k, previous, old));
            previous = old;
            tau[e.element.index()] = class * spacing;
        }
    }

    // table-read-time.5 design -- an element's signature, the order of two elements' ratios, and the triangle's bit read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// The signature of element `element`, entered by `push`. The read has
    /// no bounds check outside the test profile, whose debug assertions
    /// check it.
    pub(crate) fn sig(&self, element: ElementId) -> Sig {
        debug_assert!(
            element.index() < self.sig.len(),
            "the signature of element {element} past the {len} entered",
            len = self.sig.len()
        );
        // SAFETY: `push` gives each element entered its signature.
        unsafe { *self.sig.get_unchecked(element.index()) }
    }

    /// Gives back the slack of the growable arrays.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.sig.shrink_to_fit();
        self.tau.shrink_to_fit();
        self.sorted.shrink_to_fit();
        self.masks.shrink_to_fit();
        for list in &mut self.by_index {
            list.shrink_to_fit();
        }
        for known in &mut self.syzygies {
            known.shrink_to_fit();
        }
    }

    // signature-pruning.16 cited: roune-2012 §3.1 -- the signature criterion eliminates all syzygy signatures divisible by some other syzygy signature
    /// Records a known syzygy signature: false where a known one divides it
    /// and nothing changes. The known ones it divides are forgotten.
    pub(crate) fn push_syzygy<M: Format>(&mut self, table: &Table<M>, sig: Sig) -> bool {
        if self.is_syzygy(table, sig) {
            return false;
        }
        let known = &mut self.syzygies[sig.index as usize];
        known.drop_multiples(table, sig.monomial);
        known.push(table, sig.monomial);
        true
    }

    /// Whether a known syzygy signature divides `sig`.
    pub(crate) fn is_syzygy<M: Format>(&mut self, table: &Table<M>, sig: Sig) -> bool {
        let m = sig.monomial;
        self.syzygies[sig.index as usize].divides(table, table.monomial(m), table.mask(m))
    }

    /// Whether a known syzygy signature of index `formed.index` divides the
    /// monomial formed last in the table.
    pub(crate) fn is_syzygy_formed<M: Format>(
        &mut self,
        table: &Table<M>,
        formed: &Formed,
    ) -> bool {
        self.syzygies[formed.index as usize].divides(table, table.formed(), formed.mask)
    }

    // signature-pruning.4 cited: arri-2011 §7.2 -- start with a polynomial with the smallest leading term known for a signature, where F5 keeps the first generated; via eder-2013 §3.3, Theorem 13; gao-2016 §3, independently -- one J-pair for each J-signature, the one whose v-part is minimal; the tie as eder-2017 §7.3, Definition of ⊴_rat
    /// The element of the reductant at `sig`: of the elements whose
    /// signature divides it, the one of largest sig-lead ratio, ties to the
    /// larger signature, then to the element made first. None where none
    /// divides it.
    pub(crate) fn reductant<M: Format, O: Order<M>>(
        &self,
        table: &Table<M>,
        order: &O,
        sig: Sig,
    ) -> Option<ElementId> {
        let mut best: Option<ElementId> = None;
        for &e in &self.by_index[sig.index as usize] {
            let s = self.sig[e.index()].monomial;
            if !table.divides(s, sig.monomial) {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => match self.tau[e.index()].cmp(&self.tau[b.index()]) {
                    Ordering::Greater => true,
                    Ordering::Less => false,
                    Ordering::Equal => table.cmp(order, s, self.sig[b.index()].monomial).is_gt(),
                },
            };
            if better {
                best = Some(e);
            }
        }
        best
    }

    // table-read-time.9 design -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    /// The reducer of monomial `m`: of the elements whose leading monomial
    /// divides m, the one of least (sig-lead ratio, index), ties to fewer
    /// terms, then to the element made first (signature-pruning.10). None
    /// where none divides m.
    pub(crate) fn reducer<M: Format>(&self, table: &Table<M>, m: MonoId) -> Option<ElementId> {
        let (format, words) = (table.format(), table.monomial(m));
        let sorted = &self.sorted;
        find_passing(&self.masks, table.mask(m), |k| {
            format.divides(table.monomial(sorted[k].lead), words)
        })
        .map(|k| sorted[k].element)
    }

    // table-read-time.5 design -- an element's signature, the order of two elements' ratios, and the triangle's bit read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// Whether element e's (sig-lead ratio, index) is less than element
    /// f's: at a monomial both leading monomials divide, e's multiple has
    /// the smaller signature. The reads have no bounds check outside the
    /// test profile, whose debug assertions check them.
    pub(crate) fn before(&self, e: ElementId, f: ElementId) -> bool {
        debug_assert!(
            e.max(f).index() < self.tau.len(),
            "the ratios of elements {e} and {f} past the {len} entered",
            len = self.tau.len()
        );
        // SAFETY: `push` gives each element entered its ratio's integer.
        unsafe { self.tau.get_unchecked(e.index()) < self.tau.get_unchecked(f.index()) }
    }
}

/// The signature of an S-pair, formed in the table's scratch words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Formed {
    pub(crate) index: u32,
    pub(crate) degree: u32,
    pub(crate) mask: u32,
}

/// The S-pair of elements g and h: None where their (sig-lead ratio, index)
/// is equal. Else its signature, the larger of lcm/lm(g)·sig(g) and
/// lcm/lm(h)·sig(h), is formed in the table's scratch words, and its index,
/// degree, and mask are returned. The larger is read from the (sig-lead
/// ratio, index) of g and h, and only it is formed. Overflow where it passes
/// the format.
pub(crate) fn pair<F: Field, M: Format, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    sigs: &Signatures,
    g: ElementId,
    h: ElementId,
) -> Result<Option<Formed>, Overflow> {
    let (larger, smaller) = match sigs.tau[g.index()].cmp(&sigs.tau[h.index()]) {
        Ordering::Equal => return Ok(None),
        Ordering::Greater => (g, h),
        Ordering::Less => (h, g),
    };
    let larger_sig = sigs.sig(larger);
    let degree =
        table.form_quotient(larger_sig.monomial, basis.lead(larger), basis.lead(smaller))?;
    let mask = table.format().mask(table.formed());
    Ok(Some(Formed {
        index: larger_sig.index,
        degree,
        mask,
    }))
}

// signature-pair-time.5 cited: roune-2012 §3.5 -- we use a triangle of bits, one bit for each S-pair, so we can see in constant time if we know S(α, γ) to have a syzygy signature
/// One bit per pair of elements: set where its S-pair is known to have a
/// syzygy signature.
pub(crate) struct Triangle {
    /// Row j holds the bits of the pairs (i, j), i < j.
    rows: Vec<Vec<u64>>,
}

impl Triangle {
    pub(crate) fn new() -> Triangle {
        Triangle { rows: Vec::new() }
    }

    /// Rows for the elements up to `elements`.
    pub(crate) fn grow(&mut self, elements: usize) {
        while self.rows.len() < elements {
            let j = self.rows.len();
            self.rows.push(vec![0; j.div_ceil(64)]);
        }
    }

    // table-read-time.5 design -- an element's signature, the order of two elements' ratios, and the triangle's bit read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// The bit of the pair (i, j). The read has no bounds check outside the
    /// test profile, whose debug assertions check it.
    pub(crate) fn get(&self, i: ElementId, j: ElementId) -> bool {
        let (i, j) = (i.min(j).index(), i.max(j).index());
        debug_assert!(
            j < self.rows.len() && i / 64 < self.rows[j].len(),
            "the pair ({i}, {j}) past the triangle's {len} rows",
            len = self.rows.len()
        );
        // SAFETY: `grow` gives row j of every element j entered j.div_ceil(64)
        // words, and i < j.
        unsafe { self.rows.get_unchecked(j).get_unchecked(i / 64) >> (i % 64) & 1 != 0 }
    }

    pub(crate) fn set(&mut self, i: ElementId, j: ElementId) {
        let (i, j) = (i.min(j).index(), i.max(j).index());
        self.rows[j][i / 64] |= 1 << (i % 64);
    }
}

// signature-pair-time.4 cited: roune-2012 §3.5 -- the base divisor that will eliminate the most S-pairs is the one with the highest sig-lead ratio, so we use that one; the numbers are based on two base divisors per β, as that minimized the total runtime
/// The base divisors of an element t (roune-2012 §3.5), in a format of
/// words `W`.
pub(crate) struct BaseDivisors<W> {
    /// Of largest τ among the earlier elements whose leading monomial
    /// divides lm(t).
    pub(crate) high: Option<ElementId>,
    // monomial-time.12 design -- the low ratio base divisor's x^v is packed words of the format, an infinite exponent held as MAX_DEGREE, and in(γ) | x^v is the format's divisibility test, not a vector of u32 with u32::MAX for an infinite exponent compared exponent by exponent (log 0029-1)
    /// Of largest τ among the earlier elements whose signature divides
    /// 𝔰(t), with x^v as packed words of the format, an infinite exponent
    /// held as `MAX_DEGREE`.
    pub(crate) low: Option<(ElementId, Vec<W>)>,
}

/// Which base divisor eliminated an S-pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Base {
    High,
    Low,
}

impl<W: Copy + Default> BaseDivisors<W> {
    // signature-pair-time.7 design -- the base divisors are found by a scan of the earlier elements, by leading monomial and by signature within the index (log 0026-1)
    /// The base divisors of element t among the elements before it.
    pub(crate) fn of<F: Field, M: Format<Word = W>, D: Module>(
        table: &Table<M>,
        basis: &Basis<F, D>,
        sigs: &Signatures,
        t: ElementId,
    ) -> BaseDivisors<W> {
        // signature-pair-time.13 design -- among candidates of equal sig-lead ratio and index, the base divisor is the one made first (not measured)
        let larger = |e: ElementId, best: Option<ElementId>| {
            best.is_none_or(|b| sigs.tau[e.index()] > sigs.tau[b.index()])
        };
        let b = basis.lead(t);
        let mut high = None;
        for e in (0..t.index()).map(ElementId::at) {
            if larger(e, high) && table.divides(basis.lead(e), b) {
                high = Some(e);
            }
        }
        let s = sigs.sig(t);
        let mut alpha = None;
        // An index's elements stand in the order made.
        for &e in &sigs.by_index[s.index as usize] {
            if e >= t {
                break;
            }
            if larger(e, alpha) && table.divides(sigs.sig(e).monomial, s.monomial) {
                alpha = Some(e);
            }
        }
        // roune-2012 Theorem 8: x^p = lm(α′)·𝔰(t)/𝔰(α′), and v_i = ∞ where
        // b_i ≤ p_i, else max(p_i, a_i), which is p_i since 𝔰(α′) | 𝔰(t).
        // A finite v_i lies below b_i, so every v_i is at most MAX_DEGREE.
        let low = alpha.map(|a| {
            let (lead, sa) = (basis.lead(a), sigs.sig(a).monomial);
            let format = table.format();
            let v: Vec<u32> = (0..format.variables())
                .map(|i| {
                    let p = table.exponent(lead, i) + table.exponent(s.monomial, i)
                        - table.exponent(sa, i);
                    if table.exponent(b, i) <= p {
                        M::MAX_DEGREE
                    } else {
                        p
                    }
                })
                .collect();
            let mut words = vec![W::default(); format.words()];
            format.pack(&v, &mut words);
            (a, words)
        });
        BaseDivisors { high, low }
    }

    // signature-pair-time.3 cited: roune-2012 §3.5 and Appendix B.2, Theorems 18 and 19 -- the base divisor criterion is strictly weaker than the signature criterion, but it is faster to check; for yang1 the base divisors are a 35% performance improvement, and they eliminate 71% of the S-pairs
    /// Which base divisor eliminates S(t, g), g < t, where one does: its
    /// S-pair with g is known to have a syzygy signature, and that
    /// signature divides 𝔰(S(t, g)) by roune-2012 Theorem 7 or 8. τ orders
    /// the (sig-lead ratio, index) classes as Schreyer's order orders the
    /// ratios of signatures (signature-pruning.23), and the comparisons are
    /// strict as the theorems state them.
    pub(crate) fn eliminates<F: Field, M: Format<Word = W>, D: Module>(
        &self,
        table: &Table<M>,
        basis: &Basis<F, D>,
        sigs: &Signatures,
        triangle: &Triangle,
        t: ElementId,
        g: ElementId,
    ) -> Option<Base> {
        let tau = |e: ElementId| sigs.tau[e.index()];
        let (tg, tt) = (tau(g), tau(t));
        if let Some(a) = self.high
            && a != g
            && tg > tau(a)
            && tg > tt
            && triangle.get(a, g)
        {
            return Some(Base::High);
        }
        if let Some((a, v)) = &self.low
            && *a != g
            && tg < tau(*a)
            && tg < tt
            && triangle.get(*a, g)
        {
            // No exponent of a monomial of the format passes MAX_DEGREE, so
            // it stands for ∞.
            if table.format().divides(table.monomial(basis.lead(g)), v) {
                return Some(Base::Low);
            }
        }
        None
    }
}

// signature-pair-time.12 design -- run_sig takes a Pairs: whether to use base divisors, and the count of every kind of S-pair (not measured)
/// The S-pairs of a signature run: whether the run starts out using base
/// divisors, and what became of every S-pair it considered.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Pairs {
    pub(crate) bases: bool,
    pub(crate) considered: u64,
    /// Of equal (sig-lead ratio, index): no signature.
    pub(crate) equal: u64,
    /// Eliminated by the high ratio base divisor.
    pub(crate) high: u64,
    /// Eliminated by the low ratio base divisor.
    pub(crate) low: u64,
    /// Formed, and a known syzygy signature divides it.
    pub(crate) syzygy: u64,
    /// Formed, coprime leading monomials, entered as a syzygy signature.
    pub(crate) koszul: u64,
    /// Formed and queued, those already pending among them.
    pub(crate) queued: u64,
    /// Coprime leading monomials, past the format: dropped.
    pub(crate) overflow: u64,
}

impl Pairs {
    pub(crate) fn new(bases: bool) -> Pairs {
        Pairs {
            bases,
            ..Pairs::default()
        }
    }
}

/// The pending signatures.
pub(crate) struct Queue {
    /// Per index the monomials of the pending signatures.
    pending: Vec<Vec<MonoId>>,
    // signature-pruning.26 design -- a hash set of the pending signatures stands beside the lists and finds a signature already pending, not a scan of its index's list (not measured)
    /// The pending signatures.
    set: HashSet<Sig>,
}

impl Queue {
    pub(crate) fn new(inputs: usize) -> Queue {
        Queue {
            pending: vec![Vec::new(); inputs],
            set: HashSet::new(),
        }
    }

    // signature-pruning.25 cited: lairez-2022 §5.7, Theorem 62 -- Algorithm 8 is correct with Invariant (1), every critical signature pending or the basis a rewrite basis there; eder-2017 §6.2, Lemma (singular criterion) -- for any signature exactly one element needs to be handled
    /// Adds `sig` unless it is pending. A pending signature that divides
    /// another stays beside it, and the run drops a multiple of a known
    /// syzygy signature when a round takes it (gfp-unsolved.14).
    pub(crate) fn push(&mut self, sig: Sig) {
        if self.set.insert(sig) {
            self.pending[sig.index as usize].push(sig.monomial);
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Gives back the slack of the growable arrays and of the set.
    pub(crate) fn shrink_to_fit(&mut self) {
        for list in &mut self.pending {
            list.shrink_to_fit();
        }
        self.set.shrink_to_fit();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.set.len()
    }

    // signature-pruning.3 cited: eder-2013a §3 -- signature-based algorithms should always be implemented without a purely polynomial degree preselection; §4, Theorem 12 -- sig-deg(f) equals the sugar degree of poly(f)
    /// Takes out every pending signature of the least degree, in increasing
    /// order of signature.
    pub(crate) fn pop_least<M: Format, O: Order<M>>(
        &mut self,
        table: &Table<M>,
        order: &O,
    ) -> Vec<Sig> {
        let least = self
            .pending
            .iter()
            .flat_map(|p| p.iter().map(|&m| table.degree(m)))
            .min();
        let Some(d) = least else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (index, pending) in self.pending.iter_mut().enumerate() {
            pending.retain(|&monomial| {
                if table.degree(monomial) != d {
                    return true;
                }
                out.push(Sig {
                    monomial,
                    index: index.narrow(),
                });
                false
            });
        }
        for s in &out {
            self.set.remove(s);
        }
        out.sort_by(|a, b| a.cmp(*b, table, order));
        out
    }
}

#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::*;
    use crate::field::Gf31;
    use crate::module::Rank1;
    use crate::monomial::Dense16;
    use crate::order::Grevlex;
    use crate::poly::Poly;

    /// Enters elements of one index whose signatures are all x^count and
    /// whose leads are x^l for each l of `leads`, so the ratio of x^l is
    /// x^(count - l). `before` reads the ratios right after every element.
    /// After the last, `pair` reads them right, and every integer lies
    /// strictly between 0 and `u64::MAX`.
    fn enter(count: u32, leads: &[u32]) {
        let mut table: Table<Dense16> =
            Table::new(Dense16::new(2), &mut Xoshiro256PlusPlus::seed_from_u64(5));
        let field = Gf31::new(65521).unwrap();
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        let mut sigs = Signatures::new(1);
        let s = table.intern(&[count, 0]).unwrap();
        for (k, &l) in leads.iter().enumerate() {
            let lead = table.intern(&[l, 0]).unwrap();
            let poly = Poly {
                monomials: vec![lead],
                coefs: vec![field.coef(1)],
                position: (),
            };
            basis.push(poly, false).unwrap();
            let sig = Sig {
                monomial: s,
                index: 0,
            };
            sigs.push(&table, &Grevlex, &basis, sig).unwrap();
            // The larger lead, the smaller ratio.
            for a in 0..=k {
                for b in 0..=k {
                    assert_eq!(
                        sigs.before(ElementId::at(a), ElementId::at(b)),
                        leads[a] > leads[b],
                        "{a} before {b}, {k} in"
                    );
                }
            }
        }
        assert!(sigs.tau.iter().all(|&t| 0 < t && t < u64::MAX));
        for g in 0..leads.len() {
            for h in 0..leads.len() {
                let (g, h) = (ElementId::at(g), ElementId::at(h));
                let got = pair(&mut table, &basis, &sigs, g, h).unwrap();
                let (g, h) = (g.index(), h.index());
                let (a, b) = (leads[g], leads[h]);
                if a == b {
                    assert_eq!(got, None);
                } else {
                    // The larger ratio is the lesser lead's, and the pair's
                    // signature x^(max - min + count).
                    let formed = got.unwrap();
                    let exponent = table.format().exponent(table.formed(), 0);
                    assert_eq!(exponent, a.abs_diff(b) + count);
                    assert_eq!(formed.index, 0);
                    let (lg, lh) = (basis.lead(ElementId::at(g)), basis.lead(ElementId::at(h)));
                    let coprime = table.coprime(lg, lh);
                    assert_eq!(coprime, a == 0 || b == 0);
                }
            }
        }
    }

    /// Ratios that fall, each new one below all before it, use up the
    /// integers below the first and make them respaced. Every other element
    /// repeats its predecessor's ratio and shares its integer.
    #[test]
    fn integers_are_respaced_below_the_least() {
        let leads: Vec<u32> = (0..150).map(|k| k / 2).collect();
        enter(150, &leads);
    }

    /// As above, with an element of ratio 1 entered first, so the new ratios
    /// fall into the gap above it.
    #[test]
    fn integers_are_respaced_between_two() {
        let mut leads = vec![150];
        leads.extend((0..150).map(|k| k / 2));
        enter(150, &leads);
    }
}
