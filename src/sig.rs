//! Signatures (lairez-2022 §3, eder-2017 §4): each basis element's and the
//! known syzygy signatures. The pending ones stand in `queue::Queue`.

use std::cmp::Ordering;

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::field::Field;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::{Format, Overflow};
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::Basis;
use crate::table::{Limit, PLACE, Table, find_passing, search_units};

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
    // signature-pruning.2 cited-checked: gao-2016 §3 -- the g-weighted order followed by POT (g2) is the clear winner of four signature orders; lairez-2022 Remark 25 and Example 26 -- sig g_i = lm g_i ⊗ e_i with TOP, which eder-2017 calls lt-pot
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

    /// t·self, a growth of the table spent on `budget`.
    pub(crate) fn mul<M: Format>(
        self,
        t: MonoId,
        table: &mut Table<M>,
        budget: &mut Budget,
    ) -> Result<Sig, Limit> {
        Ok(Sig {
            monomial: table.mul(t, self.monomial, budget)?,
            index: self.index,
        })
    }

    /// The degree of the monomial, the sugar degree (signature-pruning.3).
    pub(crate) fn degree<M: Format>(self, table: &Table<M>) -> u32 {
        table.degree(self.monomial)
    }
}

// signature-pruning.21 practice-checked: cross-multiplication -- sig-lead ratios are compared as s_a·l_b against s_b·l_a, formed in scratch words and never interned, not each ratio stored as a Laurent monomial
// bivium-signature-gap.1 cited-checked: roune-2012 §3.4; bachmann-1998 §3, "Firstly", independently -- computing E and F with the gcd divided out can take a lot of time, and E < F exactly where bA < aB; exponents added a machine word at a time stay below 2^(s_e−1), so no lane overflows
/// Compares the sig-lead ratios `s_a/l_a` and `s_b/l_b`, monomials of the
/// table, under `order`, as `s_a·l_b` against `s_b·l_a`. Each product is
/// the word sum of its factors, formed in a scratch array and not
/// interned: a lane of a monomial of the format holds at most
/// `MAX_DEGREE`, 2^(B−1) − 1, so a lane of the sum holds at most
/// 2^B − 2 and carries into none beside it, and its degree, the sum of
/// the factors', fits a u32.
pub(crate) fn cmp_ratio<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    (s_a, l_a): (MonoId, MonoId),
    (s_b, l_b): (MonoId, MonoId),
) -> Ordering {
    let format = table.format();
    let w = format.words();
    let mut words = vec![M::Word::default(); 2 * w];
    let (left, right) = words.split_at_mut(w);
    let factors = table
        .monomial(s_a)
        .iter()
        .zip(table.monomial(l_b))
        .zip(table.monomial(s_b).iter().zip(table.monomial(l_a)));
    for ((l, r), ((&a, &b), (&c, &d))) in left.iter_mut().zip(right.iter_mut()).zip(factors) {
        *l = M::mul_word(a, b);
        *r = M::mul_word(c, d);
    }
    let left_degree = table.degree(s_a) + table.degree(l_b);
    let right_degree = table.degree(s_b) + table.degree(l_a);
    order.cmp(format, left, left_degree, right, right_degree)
}

/// The units of a comparison of two sig-lead ratios (`cmp_ratio`) in a
/// format of `words` words: four monomials read at random places, `PROBE`
/// units each, the two products' words a unit each, and the scratch array
/// allocated, `PROBE` units.
fn ratio_units(words: usize) -> usize {
    5 * PROBE + 2 * words
}

// signature-pruning.20 practice-checked: divisor masks (divmasks) on a monomial list -- the known syzygy signatures are a list of monomials per index, scanned with divisor masks, not kd-trees (the pending signatures: call-stop-latency.57)
/// Monomials of the table, each with its divisor mask, scanned in a list.
/// None divides another, but that a coprime S-pair's Koszul signature
/// joins with the members it divides left in (`Signatures::push_coprime`,
/// bivium-rest-gap.6): each such member has its divisor among the members,
/// so a search answers as over the minimal ones.
#[derive(Clone, Debug, Default)]
struct Antichain {
    ids: Vec<MonoId>,
    // signature-pair-time.9 practice-checked: structure of arrays -- the masks stand in an array of their own beside the ids, and a query tests eight of them at a time before it compares exponents (log 0026-1)
    /// The members' masks, in step with `ids`.
    masks: Vec<u32>,
    // signature-pair-time.10 design-checked -- the member that answered the last yes is tried first (log 0026-1)
    /// The position of the member that answered the last yes. A stale
    /// position names another member, or none.
    last: usize,
}

impl Antichain {
    // table-read-time.9 design-checked -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    // call-stop-latency.20 design-checked -- the signature run's searches, the antichain's, the reducer's, the reductant's, the base divisors', and the signatures' pushes, take the budget and spend a unit per member visited, not unspent (log 0138-5)
    /// Whether a member divides the monomial of words `m` and mask `mask`.
    /// The search spends a unit per block of the members' masks it may test
    /// and `PROBE` units per member it tests (`table::search_units`) at
    /// `Site::Antichain`; a stop where the budget's check stops the call.
    fn divides<M: Format>(
        &mut self,
        table: &Table<M>,
        m: &[M::Word],
        mask: u32,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        let not_m = !mask;
        let format = table.format();
        let (ids, masks) = (&self.ids, &self.masks);
        // call-stop-latency.71 design-checked -- a search by find_passing spends PROBE units per candidate its test ran on beside its blocks of masks, the candidates counted in the test's closure, not the blocks alone (.14), as on few variables most masks pass and each candidate's test reads its words at a random place (log 0138-8)
        let mut tested = 0;
        let mut divides = |k: usize| {
            tested += 1;
            format.divides(table.monomial(ids[k]), m)
        };
        let last = self.last;
        let found = if last < masks.len() && masks[last] & not_m == 0 && divides(last) {
            Some(last)
        } else {
            find_passing(masks, mask, &mut divides)
        };
        budget.spend(search_units(masks.len(), tested), Site::Antichain)?;
        match found {
            Some(k) => {
                self.last = k;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Removes the members m divides. The rest keep their order. The scan
    /// spends a unit a member at `Site::Antichain` after each `PLACE`; a
    /// stop where the budget's check stops the call, the members then part
    /// moved, which the stopped run drops.
    fn drop_multiples<M: Format>(
        &mut self,
        table: &Table<M>,
        m: MonoId,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let mask = table.mask(m);
        let format = table.format();
        let len = self.ids.len();
        let mut kept = 0;
        for start in (0..len).step_by(PLACE) {
            let end = (start + PLACE).min(len);
            for k in start..end {
                let (b, mask_b) = (self.ids[k], self.masks[k]);
                if mask & !mask_b != 0 || !format.divides(table.monomial(m), table.monomial(b)) {
                    self.ids[kept] = b;
                    self.masks[kept] = mask_b;
                    kept += 1;
                }
            }
            budget.spend(end - start, Site::Antichain)?;
        }
        self.ids.truncate(kept);
        self.masks.truncate(kept);
        Ok(())
    }

    /// Adds m, which no member divides.
    fn push<M: Format>(&mut self, table: &Table<M>, m: MonoId) {
        self.ids.push(m);
        self.masks.push(table.mask(m));
    }

    fn shrink_to_fit(&mut self) {
        self.ids.shrink_to_fit();
        self.masks.shrink_to_fit();
    }
}

// bivium-gap.5 design-checked -- past KOSZUL_INDEX_VARIABLES variables the inputs' leading monomials are held by their first variable as well, and a test reads the lists of the quotient's variables, not the scan of their masks alone, whose bits each stand for several variables there (log 0163-1, 0163-2)
/// The variables past which the inputs' leading monomials are held by their
/// first variable (`Inputs::by_first`).
const KOSZUL_INDEX_VARIABLES: usize = 128;

/// The leading monomials of the signature run's inputs, by index, which hold
/// the inputs' Koszul signatures `lm(f_i)·lm(f_j)·e_j`, i < j
/// (`Signatures::koszul`).
#[derive(Debug, Default)]
struct Inputs {
    leads: Vec<MonoId>,
    /// The leads' masks, in step with `leads`.
    masks: Vec<u32>,
    /// Past `KOSZUL_INDEX_VARIABLES` variables, per variable v the indices
    /// of the leads whose first variable is v, in increasing order: those of
    /// `of[starts[v]..starts[v + 1]]`.
    by_first: Option<ByFirst>,
    /// The variables of the quotient a test reads.
    held: Vec<u32>,
}

/// The inputs' indices by their leads' first variable.
#[derive(Debug)]
struct ByFirst {
    starts: Vec<u32>,
    of: Vec<u32>,
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
    // signature-pruning.23 cited-checked: roune-2012 §3.4 -- an integer per basis element orders the sig-lead ratios, so a ratio comparison is an integer comparison; the integers are spaced far apart and rebuilt where a new element would need one between k and k+1
    /// Per element the integer of its (sig-lead ratio, index): the
    /// integers order them, and equal ones share one. No element has 0 or
    /// `u64::MAX`, the bounds where a neighbour is missing.
    tau: Vec<u64>,
    /// The elements by (sig-lead ratio, index, terms, element).
    sorted: Vec<Entry>,
    // matrix-build-time.7 practice-checked: structure of arrays -- the reducer order's masks stand in an array of their own, in step with the order, and the reducer search tests eight at a time before it tests the candidates' divisibility in order, not a mask in each entry tested one by one; roune-2012 §4.2 eliminates around 98% of divisibility checks by the divmask, one mask per test (signature-pruning.20) (log 0035-1, 0037-1)
    /// The divisor mask of each entry's leading monomial, in step with
    /// `sorted`.
    masks: Vec<u32>,
    /// Per index its elements, in the order made.
    by_index: Vec<Vec<ElementId>>,
    /// Per index the monomials of the known syzygy signatures, minimal but
    /// for the members a coprime S-pair's Koszul signature divides
    /// (`push_coprime`).
    syzygies: Vec<Antichain>,
    /// The inputs' leading monomials, which hold their Koszul signatures.
    inputs: Inputs,
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
            inputs: Inputs::default(),
        }
    }

    /// Takes the inputs' leading monomials, `leads[i]` that of the input of
    /// index i, each a monomial of a variable or more: their Koszul
    /// signatures are known from here on.
    pub(crate) fn set_inputs<M: Format>(&mut self, table: &Table<M>, leads: Vec<MonoId>) {
        debug_assert_eq!(
            leads.len(),
            self.by_index.len(),
            "a leading monomial per input"
        );
        let format = table.format();
        let n = format.variables();
        let inputs = &mut self.inputs;
        inputs.masks = leads.iter().map(|&l| table.mask(l)).collect();
        inputs.by_first = (n > KOSZUL_INDEX_VARIABLES).then(|| {
            let mut firsts = Vec::with_capacity(leads.len());
            for &l in &leads {
                format.held(table.monomial(l), &mut inputs.held);
                #[allow(
                    clippy::expect_used,
                    reason = "the run ends on a constant input before it takes the inputs' leads (`f4::constant_input`)"
                )]
                let first = *inputs.held.first().expect("a lead of a variable or more");
                firsts.push(first as usize);
            }
            let mut starts = vec![0u32; n + 1];
            for &v in &firsts {
                starts[v + 1] += 1;
            }
            for v in 0..n {
                starts[v + 1] += starts[v];
            }
            let mut next = starts.clone();
            let mut of = vec![0u32; leads.len()];
            for (i, &v) in firsts.iter().enumerate() {
                of[next[v] as usize] = i.narrow();
                next[v] += 1;
            }
            ByFirst { starts, of }
        });
        inputs.leads = leads;
    }

    // bivium-gap.1 cited-checked:faugere-2002 §5, Definition 2, and §7.2, CritPair, via albrecht-2010 §2, Theorem 2.4 -- an S-polynomial of signature t·e_i is redundant where the leading monomial of an element of lower index divides t; roune-2012 §3.3 -- inserting all Koszul syzygy signatures into the known ones costs significant overhead in space and in time for checking them
    // bivium-gap.2 design-checked -- the elements of lower index are the inputs alone, whose products with lm(f_j) are the Koszul signatures the run entered among the known ones before, not every element of lower index, so that every call takes its path in Schreyer's order (log 0163-1, 0163-2)
    /// Whether a Koszul signature of the inputs, `lm(f_i)·lm(f_j)·e_j` for an
    /// input i below j, divides the signature of index j whose monomial is
    /// `m`, or the monomial formed last where `m` is None, of mask `mask`:
    /// whether `lm(f_i)` divides its quotient by `lm(f_j)`, which divides it.
    /// No Koszul signature is known before `set_inputs`. The quotient spends
    /// the format's words, and the search a unit per block of the leads'
    /// masks or per lead it visits and `PROBE` units per lead it tests, at
    /// `Site::Koszul`; a stop where the budget's check stops the call.
    fn koszul<M: Format>(
        &mut self,
        table: &mut Table<M>,
        m: Option<MonoId>,
        j: u32,
        mask: u32,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        let j = j as usize;
        debug_assert!(j < self.by_index.len(), "the index {j} of an input");
        let inputs = &mut self.inputs;
        if j == 0 || inputs.leads.is_empty() {
            return Ok(false);
        }
        table.form_divided(m, inputs.leads[j]);
        let table = &*table;
        let format = table.format();
        let quotient = table.divided();
        let mut tested = 0;
        // bivium-gap.4 design-checked -- a lead's mask is tested against the signature's mask, which passes every divisor of the quotient, not against the quotient's own, a pass over its words more (log 0163-1, 0163-2)
        let (found, units) = match &inputs.by_first {
            None => {
                let leads = &inputs.leads;
                let found = find_passing(&inputs.masks[..j], mask, |input| {
                    tested += 1;
                    format.divides(table.monomial(leads[input]), quotient)
                })
                .is_some();
                (found, search_units(j, tested))
            }
            Some(ByFirst { starts, of }) => {
                format.held(quotient, &mut inputs.held);
                let mut visited = 0;
                let mut found = false;
                'search: for &variable in &inputs.held {
                    let variable = variable as usize;
                    let list = &of[starts[variable] as usize..starts[variable + 1] as usize];
                    for &input in list {
                        let input = input as usize;
                        if input >= j {
                            break;
                        }
                        visited += 1;
                        if inputs.masks[input] & !mask == 0 {
                            tested += 1;
                            if format.divides(table.monomial(inputs.leads[input]), quotient) {
                                found = true;
                                break 'search;
                            }
                        }
                    }
                }
                (found, visited + tested * PROBE)
            }
        };
        // bivium-gap.7 design-checked -- the test spends at Site::Koszul the quotient's words, a unit per block of the leads' masks or per lead visited, and PROBE units per lead tested, not unspent nor spent at Site::Antichain (log 0163-2)
        budget.spend(format.words() + units, Site::Koszul)?;
        Ok(found)
    }

    // signature-pruning.10 cited-checked: eder-2017 §13, Algorithm (Symbolic Preprocessing respecting signatures) and Property (1) -- the single reducer of minimal signature, the smaller the signature the bigger the probability it reduces another row; roune-2012 §3.4 -- compare sig-lead ratios instead of signatures
    // rule-breaches.7 design-checked -- the element recorded is the next past those recorded, which `basis` pushed last, not an id the caller passes, with an error or a debug assertion where it is not the next (not measured)
    // call-stop-latency.20 design-checked -- the signature run's searches, the antichain's, the reducer's, the reductant's, the base divisors', and the signatures' pushes, take the budget and spend a unit per member visited, not unspent (log 0138-5)
    /// Records the signature of the element `basis` pushed last, nonzero,
    /// the next past the elements recorded. It takes its place in the
    /// reducer order by binary search. The push spends at `Site::SigEnter`
    /// its comparisons, `ratio_units` each, and the entries its insertion
    /// moves, a unit each, once it is in, and the respacing its own
    /// (`respace`); a stop where the budget's check stops the call.
    pub(crate) fn push<F: Field, M: Format, O: Order<M>, D: Module>(
        &mut self,
        table: &Table<M>,
        order: &O,
        basis: &Basis<F, D>,
        sig: Sig,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
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
        let mut compared = 0;
        let mut class = |e: &Entry| -> Ordering {
            compared += 1;
            let s = self.sig[e.element.index()];
            cmp_ratio(table, order, (s.monomial, e.lead), (sig.monomial, lead))
                .then(s.index.cmp(&sig.index))
        };
        // The first entry after the new one. An entry of the same class and
        // terms comes before it, as its element was made first.
        let (mut lo, mut hi) = (0, self.sorted.len());
        while lo < hi {
            let mid = usize::midpoint(lo, hi);
            let e = &self.sorted[mid];
            let before = class(e).then(e.terms.cmp(&new.terms)).is_le();
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
            if class(e).is_eq() {
                tau = Some(self.tau[e.element.index()]);
                break;
            }
        }
        let moved = self.sorted.len() - p;
        self.sig.push(sig);
        self.sorted.insert(p, new);
        self.masks.insert(p, table.mask(lead));
        self.by_index[sig.index as usize].push(element);
        budget.spend(
            1 + compared * ratio_units(table.format().words()) + moved,
            Site::SigEnter,
        )?;
        if let Some(t) = tau {
            self.tau.push(t);
            return Ok(());
        }
        let below = left.map_or(0, |e| self.tau[e.element.index()]);
        let above = right.map_or(u64::MAX, |e| self.tau[e.element.index()]);
        self.tau.push(below + (above - below) / 2);
        if above - below < 2 {
            self.respace(p, budget)?;
        }
        Ok(())
    }

    /// Gives the classes along `sorted` integers spaced evenly over u64.
    /// `sorted[p]` is the new element, of a class of its own, and its
    /// integer is a placeholder. Each of its two passes spends `PROBE` units
    /// an entry, its integer read at a random place, at `Site::SigEnter`
    /// after each `PLACE`; a stop where the budget's check stops the call,
    /// the integers then part respaced, which the stopped run drops.
    fn respace(&mut self, p: usize, budget: &mut Budget) -> Result<(), Stop> {
        let sorted = &self.sorted;
        let tau = &mut self.tau;
        let starts =
            |k: usize, previous: u64, old: u64| k == 0 || k == p || k == p + 1 || old != previous;
        let (mut classes, mut previous) = (0u64, 0u64);
        for (c, chunk) in sorted.chunks(PLACE).enumerate() {
            for (i, e) in chunk.iter().enumerate() {
                let old = tau[e.element.index()];
                classes += u64::from(starts(c * PLACE + i, previous, old));
                previous = old;
            }
            budget.spend(chunk.len() * PROBE, Site::SigEnter)?;
        }
        let spacing = u64::MAX / (classes + 1);
        let (mut class, mut previous) = (0u64, 0u64);
        for (c, chunk) in sorted.chunks(PLACE).enumerate() {
            for (i, e) in chunk.iter().enumerate() {
                let old = tau[e.element.index()];
                class += u64::from(starts(c * PLACE + i, previous, old));
                previous = old;
                tau[e.element.index()] = class * spacing;
            }
            budget.spend(chunk.len() * PROBE, Site::SigEnter)?;
        }
        Ok(())
    }

    // table-read-time.5 design-checked -- an element's signature, the order of two elements' ratios, and the triangle's bit read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
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

    // signature-pruning.16 cited-checked: roune-2012 §3.3 -- the signature criterion eliminates all syzygy signatures that are divisible by some other syzygy signature
    /// Records a known syzygy signature: false where a known one or a
    /// Koszul signature of the inputs divides it and nothing changes. The
    /// known ones it divides are forgotten. The search spends as
    /// `is_syzygy` spends, and the scan at `Site::Antichain`; a stop where
    /// the budget's check stops the call.
    pub(crate) fn push_syzygy<M: Format>(
        &mut self,
        table: &mut Table<M>,
        sig: Sig,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        if self.is_syzygy(table, sig, budget)? {
            return Ok(false);
        }
        let known = &mut self.syzygies[sig.index as usize];
        known.drop_multiples(table, sig.monomial, budget)?;
        known.push(table, sig.monomial);
        Ok(true)
    }

    // bivium-rest-gap.5 design-checked -- the Koszul signature of an S-pair of coprime leading monomials is recorded with no search, which the search of its formed signature answered right before with the known ones as they stand, not that search run again by `push_syzygy` (log 0181-1)
    // bivium-rest-gap.6 design-checked -- the record leaves the known syzygy signatures of the index that the Koszul signature divides among them, not dropped by a scan of every known one of the index (signature-pruning.16), whose scans took bivium-115's entering more instructions than its searches spend on the members left; every search answers as before (log 0181-1)
    /// Records the Koszul signature `sig` of an S-pair of coprime leading
    /// monomials, which no known syzygy signature and no Koszul signature of
    /// the inputs divides: the caller's search of its formed signature found
    /// none (`is_syzygy_formed`). The known ones it divides stay. It spends
    /// nothing beside that search.
    pub(crate) fn push_coprime<M: Format>(&mut self, table: &Table<M>, sig: Sig) {
        debug_assert!(
            (sig.index as usize) < self.syzygies.len(),
            "the index {index} of an input",
            index = sig.index
        );
        self.syzygies[sig.index as usize].push(table, sig.monomial);
    }

    // bivium-gap.6 design-checked -- the inputs' Koszul signatures are tested after the search of the known syzygy signatures finds none, not before it (log 0163-1, 0163-2)
    /// Whether a known syzygy signature or a Koszul signature of the inputs
    /// divides `sig`, the search of the known ones spent at
    /// `Site::Antichain` and the Koszul test at `Site::Koszul` (`koszul`); a
    /// stop where the budget's check stops the call.
    pub(crate) fn is_syzygy<M: Format>(
        &mut self,
        table: &mut Table<M>,
        sig: Sig,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        let m = sig.monomial;
        let mask = table.mask(m);
        if self.syzygies[sig.index as usize].divides(table, table.monomial(m), mask, budget)? {
            return Ok(true);
        }
        self.koszul(table, Some(m), sig.index, mask, budget)
    }

    /// Whether a known syzygy signature or a Koszul signature of the inputs
    /// of index `formed.index` divides the monomial formed last in the
    /// table, spent as `is_syzygy` spends; a stop where the budget's check
    /// stops the call.
    pub(crate) fn is_syzygy_formed<M: Format>(
        &mut self,
        table: &mut Table<M>,
        formed: &Formed,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        if self.syzygies[formed.index as usize].divides(
            table,
            table.formed(),
            formed.mask,
            budget,
        )? {
            return Ok(true);
        }
        self.koszul(table, None, formed.index, formed.mask, budget)
    }

    // signature-pruning.4 cited-checked: arri-2011 §7.2 -- start with a polynomial with the smallest leading term known for a signature, where F5 keeps the first generated; via eder-2013 §3.3, Theorem 13; gao-2016 §3, independently -- one J-pair for each J-signature, the one whose v-part is minimal; the tie as eder-2017 §7.3, Definition of ⊴_rat
    // call-stop-latency.20 design-checked -- the signature run's searches, the antichain's, the reducer's, the reductant's, the base divisors', and the signatures' pushes, take the budget and spend a unit per member visited, not unspent (log 0138-5)
    /// The element of the reductant at `sig`: of the elements whose
    /// signature divides it, the one of largest sig-lead ratio, ties to the
    /// larger signature, then to the element made first. None where none
    /// divides it. The scan of the elements of `sig`'s index spends `PROBE`
    /// units an element, its signature read at a random place, at
    /// `Site::Reducer` after each `PLACE`; a stop where the budget's check
    /// stops the call.
    pub(crate) fn reductant<M: Format, O: Order<M>>(
        &self,
        table: &Table<M>,
        order: &O,
        sig: Sig,
        budget: &mut Budget,
    ) -> Result<Option<ElementId>, Stop> {
        let mut best: Option<ElementId> = None;
        for chunk in self.by_index[sig.index as usize].chunks(PLACE) {
            for &e in chunk {
                let s = self.sig[e.index()].monomial;
                if !table.divides(s, sig.monomial) {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some(b) => match self.tau[e.index()].cmp(&self.tau[b.index()]) {
                        Ordering::Greater => true,
                        Ordering::Less => false,
                        Ordering::Equal => {
                            table.cmp(order, s, self.sig[b.index()].monomial).is_gt()
                        }
                    },
                };
                if better {
                    best = Some(e);
                }
            }
            budget.spend(chunk.len() * PROBE, Site::Reducer)?;
        }
        Ok(best)
    }

    /// The elements whose signatures it holds, the masks a reducer search
    /// may test, which its caller spends with the candidates the search
    /// tested (`table::search_units`).
    pub(crate) fn len(&self) -> usize {
        self.masks.len()
    }

    // table-read-time.9 design-checked -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    /// The reducer of monomial `m`: of the elements whose leading monomial
    /// divides m, the one of least (sig-lead ratio, index), ties to fewer
    /// terms, then to the element made first (signature-pruning.10), None
    /// where none divides m; and the candidates whose divisibility the
    /// search tested.
    pub(crate) fn reducer<M: Format>(
        &self,
        table: &Table<M>,
        m: MonoId,
    ) -> (Option<ElementId>, usize) {
        let (format, words) = (table.format(), table.monomial(m));
        let sorted = &self.sorted;
        let mut tested = 0;
        let found = find_passing(&self.masks, table.mask(m), |k| {
            tested += 1;
            format.divides(table.monomial(sorted[k].lead), words)
        })
        .map(|k| sorted[k].element);
        (found, tested)
    }

    // table-read-time.5 design-checked -- an element's signature, the order of two elements' ratios, and the triangle's bit read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
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

// signature-pair-time.5 cited-checked: roune-2012 §3.5 -- we use a triangle of bits, one bit for each S-pair, so we can see in constant time if we know S(α, γ) to have a syzygy signature
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

    /// Rows for the elements up to `elements`, each spent a unit and a unit
    /// a word at `Site::SigEnter`; a stop where the budget's check stops the
    /// call.
    pub(crate) fn grow(&mut self, elements: usize, budget: &mut Budget) -> Result<(), Stop> {
        while self.rows.len() < elements {
            let j = self.rows.len();
            self.rows.push(vec![0; j.div_ceil(64)]);
            budget.spend(1 + j.div_ceil(64), Site::SigEnter)?;
        }
        Ok(())
    }

    // table-read-time.5 design-checked -- an element's signature, the order of two elements' ratios, and the triangle's bit read with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
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

// signature-pair-time.4 cited-checked: roune-2012 §3.5 -- the base divisor that will eliminate the most S-pairs is the one with the highest sig-lead ratio, so we use that one; the numbers are based on two base divisors per β, as that minimized the total runtime
/// The base divisors of an element t (roune-2012 §3.5), in a format of
/// words `W`.
pub(crate) struct BaseDivisors<W> {
    /// Of largest τ among the earlier elements whose leading monomial
    /// divides lm(t).
    pub(crate) high: Option<ElementId>,
    // monomial-time.12 design-checked -- the low ratio base divisor's x^v is packed words of the format, an infinite exponent held as MAX_DEGREE, and in(γ) | x^v is the format's divisibility test, not a vector of u32 with u32::MAX for an infinite exponent compared exponent by exponent (log 0029-1)
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
    // signature-pair-time.7 design-checked -- the base divisors are found by a scan of the earlier elements, by leading monomial and by signature within the index (log 0026-1)
    // call-stop-latency.20 design-checked -- the signature run's searches, the antichain's, the reducer's, the reductant's, the base divisors', and the signatures' pushes, take the budget and spend a unit per member visited, not unspent (log 0138-5)
    /// The base divisors of element t among the elements before it. Its two
    /// scans spend `PROBE` units an element, a monomial read at a random
    /// place, after each `PLACE`, and the low ratio divisor's x^v a unit a
    /// variable, at `Site::SigEnter`; a stop where the budget's check stops
    /// the call.
    pub(crate) fn of<F: Field, M: Format<Word = W>, D: Module>(
        table: &Table<M>,
        basis: &Basis<F, D>,
        sigs: &Signatures,
        t: ElementId,
        budget: &mut Budget,
    ) -> Result<BaseDivisors<W>, Stop> {
        // signature-pair-time.13 design-checked -- among candidates of equal sig-lead ratio and index, the base divisor is the one made first (not measured)
        let larger = |e: ElementId, best: Option<ElementId>| {
            best.is_none_or(|b| sigs.tau[e.index()] > sigs.tau[b.index()])
        };
        let b = basis.lead(t);
        let mut high = None;
        for start in (0..t.index()).step_by(PLACE) {
            let end = (start + PLACE).min(t.index());
            for e in (start..end).map(ElementId::at) {
                if larger(e, high) && table.divides(basis.lead(e), b) {
                    high = Some(e);
                }
            }
            budget.spend((end - start) * PROBE, Site::SigEnter)?;
        }
        let s = sigs.sig(t);
        let mut alpha = None;
        // An index's elements stand in the order made.
        let earlier = sigs.by_index[s.index as usize].partition_point(|&e| e < t);
        for chunk in sigs.by_index[s.index as usize][..earlier].chunks(PLACE) {
            for &e in chunk {
                if larger(e, alpha) && table.divides(sigs.sig(e).monomial, s.monomial) {
                    alpha = Some(e);
                }
            }
            budget.spend(chunk.len() * PROBE, Site::SigEnter)?;
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
        if low.is_some() {
            budget.spend(table.format().variables(), Site::SigEnter)?;
        }
        Ok(BaseDivisors { high, low })
    }

    // signature-pair-time.3 cited-checked: roune-2012 §3.5 and Appendix B.2, Theorems 18 and 19 -- the base divisor criterion is strictly weaker than the signature criterion, but it is faster to check; for yang1 the base divisors are a 35% performance improvement, and they eliminate 71% of the S-pairs
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

// signature-pair-time.12 design-checked -- run_sig takes a Pairs: whether to use base divisors, and the count of every kind of S-pair (not measured)
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
    /// Formed, and a known syzygy signature or a Koszul signature of the
    /// inputs divides it.
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
        let s = table.intern(&[count, 0], &mut Budget::unbounded()).unwrap();
        for (k, &l) in leads.iter().enumerate() {
            let lead = table.intern(&[l, 0], &mut Budget::unbounded()).unwrap();
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
            sigs.push(&table, &Grevlex, &basis, sig, &mut Budget::unbounded())
                .unwrap();
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
