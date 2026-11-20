//! The change of order of a zero-dimensional ideal (faugere-1993): from its
//! reduced Gröbner basis in a source order, the staircase and the normal
//! forms of its border, then the reduced basis in a target order by linear
//! algebra over the quotient ring; and the division by a reduced basis,
//! which expresses an element of the ideal in that basis.
//!
//! Every monomial of the change has degree at most D, the staircase's size:
//! a staircase is closed under division, so a monomial of degree d in it has
//! at least d + 1 divisors in it.

use std::collections::HashMap;

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::cofactors::{Cofactors, Quotient};
use crate::field::Field;
use crate::halt::Halt;
use crate::id::{BorderId, MonoId};
use crate::module::Rank1;
use crate::monomial::Format;
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::{Elements, Poly};
use crate::sort;
use crate::table::Table;

/// Where `x_j·b` lies, for b of a staircase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// In the staircase, at this index.
    Inside(MonoId),
    /// On the border, at this index.
    Border(BorderId),
}

/// The staircase of a reduced Gröbner basis of a zero-dimensional ideal in
/// the source order: its monomials, its border, and where `x_j·b` lies.
pub(crate) struct Staircase {
    /// B, by increasing source order. A monomial's index is its coordinate.
    monomials: Vec<MonoId>,
    /// `x_j·b` for b in B outside B, by increasing source order
    /// (faugere-1993 Definition 2.2).
    border: Vec<MonoId>,
    /// Entry b·n + j is where `x_j·B[b]` lies.
    next: Vec<Place>,
    /// The ids of `x_1`, …, `x_n`.
    variables: Vec<MonoId>,
    /// The place of every monomial of B and of the border, by id.
    place: HashMap<MonoId, Place>,
}

impl Staircase {
    /// D, the number of the staircase's monomials.
    pub(crate) fn len(&self) -> usize {
        self.monomials.len()
    }

    /// The staircase's monomials, by increasing source order.
    #[cfg(test)]
    pub(crate) fn monomials(&self) -> &[MonoId] {
        &self.monomials
    }

    /// The border, by increasing source order.
    #[cfg(test)]
    pub(crate) fn border(&self) -> &[MonoId] {
        &self.border
    }

    /// Where `x_j·B[b]` lies.
    fn next(&self, b: MonoId, j: usize) -> Place {
        self.next[b.index() * self.variables.len() + j]
    }

    /// Sorts B and the border by increasing `order`, and the places follow.
    /// The sorts are `sort_runs`'s, and the places' renaming and the lists'
    /// gathers spend a unit an entry on `budget`, all at `Site::Fglm`. A
    /// stop where the budget's check stops the call, the staircase then
    /// part sorted, for the call to drop.
    fn sort<M: Format, O: Order<M>>(
        &mut self,
        table: &Table<M>,
        order: &O,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let n = self.variables.len();
        let (inside_old, inside_new) = sorted(table, order, &self.monomials, budget)?;
        let (border_old, border_new) = sorted(table, order, &self.border, budget)?;
        let renamed = |p: Place| match p {
            Place::Inside(i) => Place::Inside(MonoId::at(inside_new[i.index()] as usize)),
            Place::Border(k) => Place::Border(BorderId::at(border_new[k.index()] as usize)),
        };
        let mut next = Vec::with_capacity(self.next.len());
        for &o in &inside_old {
            let o = o as usize;
            next.extend(self.next[o * n..(o + 1) * n].iter().map(|&p| renamed(p)));
            budget.spend(n, Site::Fglm)?;
        }
        self.next = next;
        self.monomials = inside_old
            .iter()
            .map(|&o| self.monomials[o as usize])
            .collect();
        self.border = border_old
            .iter()
            .map(|&o| self.border[o as usize])
            .collect();
        // call-stop-latency.69 design-checked -- a loop of FGLM bounded by MAX_STAIRCASE entries or rows spends once per loop, not RUN entries at a time, as a loop of 65,535 entries is under a millisecond (log 0138-7)
        budget.spend(self.monomials.len() + self.border.len(), Site::Fglm)?;
        for (k, p) in self.place.values_mut().enumerate() {
            *p = renamed(*p);
            if (k + 1) % sort::RUN == 0 {
                budget.spend(sort::RUN, Site::Fglm)?;
            }
        }
        budget.spend(self.place.len() % sort::RUN, Site::Fglm)
    }
}

/// The distinct monomials of `list` by increasing `order`: the old index at
/// each new one, and the new index at each old one. The sort is
/// `sort_runs`'s, and the new indices spend a unit each, at `Site::Fglm`. A
/// stop where the budget's check stops the call.
fn sorted<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    list: &[MonoId],
    budget: &mut Budget,
) -> Result<(Vec<u32>, Vec<u32>), Stop> {
    let mut old: Vec<u32> = (0..list.len().narrow()).collect();
    // The monomials are distinct, so every sort gives one order.
    let cmp = |a: &u32, b: &u32| table.cmp(order, list[*a as usize], list[*b as usize]);
    // call-stop-latency.68 design-checked -- the staircase and its border are sorted by sort_runs, not by sort::keys (.11), as they hold at most MAX_STAIRCASE·(n + 1) distinct monomials sorted once a change (log 0138-7)
    sort::sort_runs(&mut old, cmp, budget, Site::Fglm)?;
    let mut new = vec![0u32; old.len()];
    for (k, &o) in old.iter().enumerate() {
        new[o as usize] = k.narrow();
    }
    budget.spend(old.len(), Site::Fglm)?;
    Ok((old, new))
}

/// Whether every variable has a pure power, or 1, among `leads`: whether
/// the ideal whose reduced basis has these leading monomials is
/// zero-dimensional (buchberger-1985 §6.6, Method 6.9). Each variable's
/// scan spends a unit of `budget` per exponent it may read, at
/// `Site::Fglm`; a stop where the budget's check stops the call.
pub(crate) fn zero_dimensional<M: Format>(
    table: &Table<M>,
    leads: &[MonoId],
    budget: &mut Budget,
) -> Result<bool, Stop> {
    let n = table.format().variables();
    for i in 0..n {
        // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
        budget.spend(leads.len() * n, Site::Fglm)?;
        let pure = leads
            .iter()
            .any(|&m| (0..n).all(|k| k == i || table.exponent(m, k) == 0));
        if !pure {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The largest staircase the change takes (monomial-formats.11).
pub(crate) const MAX_STAIRCASE: usize = 65535;

/// The staircase of the reduced basis with leading monomials `leads` in
/// `order`; None where the ideal is not zero-dimensional. Where the
/// staircase passes a bound, the first it passes decides: `Halt::Overflow`
/// past `M::MAX_DEGREE` monomials (monomial-formats.10), None past
/// `MAX_STAIRCASE`. Ticks the budget once per staircase monomial, and
/// spends at `Site::Fglm` what `zero_dimensional` and `Staircase::sort`
/// spend, and per staircase monomial two `PROBE` units a product, its
/// lookups in the table and among the places, and a unit per leading
/// monomial a new product tests.
pub(crate) fn staircase<M: Format, O: Order<M>>(
    table: &mut Table<M>,
    order: &O,
    leads: &[MonoId],
    budget: &mut Budget,
) -> Result<Option<Staircase>, Halt> {
    if !zero_dimensional(table, leads, budget)? {
        return Ok(None);
    }
    let variable_count = table.format().variables();
    let mut exponents = vec![0; variable_count];
    let mut variables = Vec::with_capacity(variable_count);
    for variable in 0..variable_count {
        exponents[variable] = 1;
        variables.push(table.intern(&exponents, budget)?);
        exponents[variable] = 0;
    }
    let mut st = Staircase {
        monomials: Vec::new(),
        border: Vec::new(),
        next: Vec::new(),
        variables,
        place: HashMap::new(),
    };
    // G = {1}: the staircase and its border are empty.
    if leads.iter().any(|&m| table.degree(m) == 0) {
        return Ok(Some(st));
    }
    // A walk from 1 by the variables: each monomial of B in turn, times each
    // variable. B grows degree by degree, so a monomial of degree d joins
    // after its d divisors on a chain from 1, and a product of a variable
    // and a monomial of B has degree at most |B|.
    let one = table.intern(&exponents, budget)?;
    st.monomials.push(one);
    st.place.insert(one, Place::Inside(MonoId::at(0)));
    let mut position = 0;
    while position < st.monomials.len() {
        // fglm-lex.17 design-checked -- the staircase ticks the budget once per staircase monomial, not a full check once per tested monomial (not measured)
        budget.tick(Site::Fglm)?;
        let monomial = st.monomials[position];
        // A product's lookup in the table and among the places.
        let mut units = variable_count * 2 * PROBE;
        for variable in 0..variable_count {
            let product = table.mul(st.variables[variable], monomial, budget)?;
            let place = if let Some(&known) = st.place.get(&product) {
                known
            } else {
                units += leads.len();
                let new_place = if leads.iter().any(|&lead| table.divides(lead, product)) {
                    st.border.push(product);
                    Place::Border(BorderId::at(st.border.len() - 1))
                } else {
                    // monomial-formats.11 design-checked -- the change takes a staircase of at most 65535 monomials in every format, and one larger takes F4 in lex, not FGLM for every D the format holds (log 0013-1)
                    if st.monomials.len() >= MAX_STAIRCASE {
                        return Ok(None);
                    }
                    // monomial-formats.10 design-checked -- a staircase past the format's degree bound is an overflow, and the call starts over in a wider format, not F4 in lex (log 0013-1)
                    if st.monomials.len() as u64 >= u64::from(M::MAX_DEGREE) {
                        return Err(Halt::Overflow);
                    }
                    st.monomials.push(product);
                    Place::Inside(MonoId::at(st.monomials.len() - 1))
                };
                st.place.insert(product, new_place);
                new_place
            };
            st.next.push(place);
        }
        // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
        budget.spend(units, Site::Fglm)?;
        position += 1;
    }
    st.sort(table, order, budget)?;
    Ok(Some(st))
}

/// A sparse row of the field contract: column indices increasing, and
/// their nonzero coefficients.
pub(crate) struct Row<F: Field> {
    pub(crate) cols: Vec<MonoId>,
    pub(crate) coefs: Vec<F::Coef>,
}

impl<F: Field> Row<F> {
    fn new() -> Row<F> {
        Row {
            cols: Vec::new(),
            coefs: Vec::new(),
        }
    }

    /// The row `c·e_k`.
    fn unit(k: MonoId, c: F::Coef) -> Row<F> {
        Row {
            cols: vec![k],
            coefs: vec![c],
        }
    }
}

/// The dense row read back as a sparse row, and left zero.
fn read<F: Field>(field: &F, dense: &mut [F::Acc]) -> Row<F> {
    let zero = F::Coef::default();
    let mut row = Row::new();
    for (k, entry) in dense.iter_mut().enumerate() {
        let c = field.take(entry);
        if c != zero {
            row.cols.push(MonoId::at(k));
            row.coefs.push(c);
        }
    }
    row
}

/// The normal form of `x_j·p`, where `v` is the normal form of p over the
/// staircase and `forms` holds the normal form of every border monomial
/// `x_j·b` for b where v is nonzero, by the border's index. `dense` holds D
/// zero entries and is left so. Each border row added ticks the budget and
/// spends a unit an entry, and each of the two passes over v `PROBE` units
/// an entry, the load and the read back a unit an entry, at `Site::Fglm`.
fn product<F: Field>(
    field: &F,
    st: &Staircase,
    forms: &[Row<F>],
    j: usize,
    v: &Row<F>,
    dense: &mut [F::Acc],
    budget: &mut Budget,
) -> Result<Row<F>, Halt> {
    // x_j·b increases with b, so the entries inside the staircase stand by
    // increasing column, each at its own.
    let mut inside: Row<F> = Row::new();
    for (&b, &c) in v.cols.iter().zip(&v.coefs) {
        match st.next(b, j) {
            Place::Inside(i) => {
                inside.cols.push(i);
                inside.coefs.push(c);
            }
            Place::Border(_) => {}
        }
    }
    // SAFETY: an inside column indexes the staircase, below D, and `dense`
    // holds D entries, as both callers allocate it; a coefficient each.
    unsafe { field.load(dense, &inside.cols, &inside.coefs) };
    // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
    // call-stop-latency.69 design-checked -- a loop of FGLM bounded by MAX_STAIRCASE entries or rows spends once per loop, not RUN entries at a time, as a loop of 65,535 entries is under a millisecond (log 0138-7)
    budget.spend(v.cols.len() * PROBE + inside.cols.len(), Site::Fglm)?;
    for (&b, &c) in v.cols.iter().zip(&v.coefs) {
        match st.next(b, j) {
            Place::Inside(_) => {}
            Place::Border(k) => {
                budget.tick(Site::Fglm)?;
                let row = &forms[k.index()];
                // SAFETY: a normal form's columns index the staircase, a
                // coefficient each.
                unsafe { field.sub_mul(dense, &row.cols, &row.coefs, field.neg(c)) };
                budget.spend(row.cols.len(), Site::Fglm)?;
            }
        }
    }
    budget.spend(v.cols.len() * PROBE + dense.len(), Site::Fglm)?;
    Ok(read(field, dense))
}

// fglm-lex.7 practice-checked: multiplication matrices (matricial FGLM) -- the normal forms of G₁'s border monomials are computed once, before the change, in increasing grevlex, by Matphi's recurrence, not the normal form of x_i·p computed by reduction modulo G₁ when the change needs it
/// The multiplication table: the normal form of every border monomial of
/// `staircase` modulo `basis`, the reduced basis it is the staircase of, as
/// a row over the staircase, by the border's index. Each border row added
/// ticks the budget, and the lookups of the leading monomials, of a tail's
/// terms, and of a monomial's quotients spend `PROBE` units each, at
/// `Site::Fglm`, beside what `product` spends. `Halt::Misfit(0)` where a
/// tail monomial of `basis` lies outside the staircase: `basis` is not
/// reduced.
pub(crate) fn border_forms<F: Field, M: Format>(
    field: &F,
    table: &mut Table<M>,
    staircase: &Staircase,
    basis: &[Poly<F, Rank1>],
    budget: &mut Budget,
) -> Result<Vec<Row<F>>, Halt> {
    let st = staircase;
    let n = st.variables.len();
    let leads: HashMap<MonoId, usize> = basis
        .iter()
        .enumerate()
        .map(|(u, g)| (g.lead(), u))
        .collect();
    // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
    budget.spend(basis.len() * PROBE, Site::Fglm)?;
    let mut dense = vec![F::Acc::default(); st.len()];
    let mut forms: Vec<Row<F>> = Vec::with_capacity(st.border.len());
    for &m in &st.border {
        // faugere-1993 Proposition 2.1 (i): m leads an element g of the
        // basis, and its normal form is m − g, minus g's tail, whose
        // monomials lie in the staircase, decreasing.
        if let Some(&u) = leads.get(&m) {
            let g = &basis[u];
            debug_assert_eq!(field.value(g.coefs[0]), 1, "the basis is monic");
            let mut row = Row::new();
            for (&t, &c) in g.monomials[1..].iter().zip(&g.coefs[1..]).rev() {
                let i = match st.place.get(&t) {
                    Some(&Place::Inside(i)) => i,
                    Some(&Place::Border(_)) | None => return Err(Halt::Misfit(0)),
                };
                row.cols.push(i);
                row.coefs.push(field.neg(c));
            }
            budget.spend(g.len() * PROBE, Site::Fglm)?;
            forms.push(row);
            continue;
        }
        // Proposition 2.1 (ii): m = x_l·m' with m' on the border, and m' < m
        // stands before m.
        let mut before = None;
        budget.spend(n * PROBE, Site::Fglm)?;
        for l in 0..n {
            if table.exponent(m, l) == 0 {
                continue;
            }
            let q = table.div(m, st.variables[l], budget)?;
            match st.place.get(&q) {
                Some(&Place::Border(k)) => {
                    before = Some((l, k));
                    break;
                }
                Some(&Place::Inside(_)) | None => {}
            }
        }
        #[allow(
            clippy::expect_used,
            reason = "faugere-1993 Proposition 2.1 (ii), as the comment above writes it"
        )]
        let (l, k) = before.expect("a border monomial that leads no element has one before it");
        let row = product(field, st, &forms, l, &forms[k.index()], &mut dense, budget)?;
        forms.push(row);
    }
    Ok(forms)
}

/// Pushes `id` onto the binary heap `heap`, whose top `heap[0]` is the id
/// that `first` puts before every other.
fn push(heap: &mut Vec<MonoId>, id: MonoId, first: impl Fn(MonoId, MonoId) -> bool) {
    heap.push(id);
    let mut i = heap.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if !first(heap[i], heap[parent]) {
            break;
        }
        heap.swap(i, parent);
        i = parent;
    }
}

/// Takes the top off the binary heap `heap`, ordered by `first` as `push`
/// orders it.
fn pop(heap: &mut Vec<MonoId>, first: impl Fn(MonoId, MonoId) -> bool) -> Option<MonoId> {
    let tail = heap.pop()?;
    if heap.is_empty() {
        return Some(tail);
    }
    let top = std::mem::replace(&mut heap[0], tail);
    let mut i = 0;
    loop {
        let (l, r) = (2 * i + 1, 2 * i + 2);
        let mut least = i;
        if l < heap.len() && first(heap[l], heap[least]) {
            least = l;
        }
        if r < heap.len() && first(heap[r], heap[least]) {
            least = r;
        }
        if least == i {
            return Some(top);
        }
        heap.swap(i, least);
        i = least;
    }
}

/// A monomial `x_j·b` waiting to be tested, b of the new staircase.
struct Candidate {
    /// How often it was inserted: once per variable whose quotient lies in
    /// the new staircase.
    count: u32,
    /// The variable and the new staircase's index of its first insertion.
    var: u32,
    from: MonoId,
}

// fglm-lex.8 cited-checked: faugere-1993 Theorem 5.1 -- the monomial work takes O(n²D²) with an elementary implementation of ListOfNexts, O(n²D log(nD)) where the list is an efficient priority queue
/// The monomials waiting to be tested: a binary heap of their ids under the
/// target order, the least on top, and each one's candidate.
struct Candidates {
    heap: Vec<MonoId>,
    map: HashMap<MonoId, Candidate>,
}

/// The levels of a binary heap of `len` ids: the comparisons a push or a
/// pop makes at most.
fn levels(len: usize) -> usize {
    len.bit_width() as usize
}

impl Candidates {
    /// Inserts `x_j·m` for every variable `x_j`, m the new staircase's monomial
    /// at index `from`. Spends at `Site::Fglm` per variable `PROBE` units a
    /// product and a lookup, and for the monomial's push and its later pop
    /// (`pop`) two `PROBE` for the map and two units per level of the heap.
    fn insert<M: Format, O: Order<M>>(
        &mut self,
        table: &mut Table<M>,
        target: &O,
        variables: &[MonoId],
        m: MonoId,
        from: MonoId,
        budget: &mut Budget,
    ) -> Result<(), Halt> {
        for (j, &x) in variables.iter().enumerate() {
            let t = table.mul(x, m, budget)?;
            if let Some(c) = self.map.get_mut(&t) {
                c.count += 1;
                continue;
            }
            let c = Candidate {
                count: 1,
                var: j.narrow(),
                from,
            };
            self.map.insert(t, c);
            push(&mut self.heap, t, |a, b| table.cmp(target, a, b).is_lt());
        }
        // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
        // call-stop-latency.70 design-checked -- a monomial pushed onto a heap of FGLM spends its pop beside its push, two units per level of the heap and PROBE for its map's entry, not the pop spending its own levels (log 0138-7)
        let units = 4 * PROBE + 2 * levels(self.heap.len());
        budget.spend(variables.len() * units, Site::Fglm)?;
        Ok(())
    }

    /// The least monomial waiting, and its candidate. A monomial taken does
    /// not come back: every monomial inserted after it is larger. Its
    /// insertion spent its pop.
    fn pop<M: Format, O: Order<M>>(
        &mut self,
        table: &Table<M>,
        target: &O,
    ) -> Option<(MonoId, Candidate)> {
        let m = pop(&mut self.heap, |a, b| table.cmp(target, a, b).is_lt())?;
        #[allow(
            clippy::expect_used,
            reason = "the heap and the map gain and lose a monomial together"
        )]
        let c = self.map.remove(&m).expect("a monomial in the heap waits");
        Some((m, c))
    }
}

/// A row of the semi-echelon form of the new staircase's normal forms. The
/// row is zero at the pivots of the rows before it.
struct Echelon<F: Field> {
    /// Its first column, where it holds 1.
    pivot: MonoId,
    /// The rest of the row.
    row: Row<F>,
    /// The row is Σ `comb_k·NF(staircase_k)`, over the new staircase.
    comb: Row<F>,
}

/// The reduced Gröbner basis in `target` of the ideal whose reduced basis
/// in `source` is `elements`, by increasing leading monomial in `target`,
/// each element with its cofactors composed from `elements`' where `C`
/// tracks them. `staircase` is that of `elements`. The budget is checked at
/// the start and ticked once per row operation and per division step; at
/// `Site::Fglm` each candidate spends a unit a variable for its test, a
/// unit per row of the echelon form its loop visits, and the dense rows
/// loaded and read back and the rows it makes a unit an entry, beside what
/// `border_forms`, `product`, `Candidates::insert`, `sort::poly`, and
/// `divide` spend; and the cofactors' lift spends what `Cofactors::lift`
/// says. `Halt::Misfit(0)` where a vector that joins the echelon form has
/// no inverse at its first entry: over several lanes, 0 at one lane.
/// `Halt::Misfit(0)` too where `elements` is not reduced (`border_forms`),
/// and where `C` tracks cofactors and an element leaves a remainder in its
/// division by `elements` (`divide`).
#[allow(
    clippy::too_many_arguments,
    reason = "the field, the table, the two orders, the staircase, the elements and their cofactors, and the budget, none of which belongs with another"
)]
pub(crate) fn change<F, M, O1, O2, C>(
    field: &F,
    table: &mut Table<M>,
    source: &O1,
    target: &O2,
    staircase: &Staircase,
    elements: Elements<F, Rank1, C::Element>,
    cof: &mut C,
    budget: &mut Budget,
) -> Result<Elements<F, Rank1, C::Element>, Halt>
where
    F: Field,
    M: Format,
    O1: Order<M>,
    O2: Order<M>,
    C: Cofactors<F, M>,
{
    // fglm-lex.17 design-checked -- the change ticks the budget once per row operation and per division step, and checks it fully at its start, not a full check once per tested monomial (not measured)
    budget.check(|| {}, Site::Fglm)?;
    let st = staircase;
    let d = st.len();
    if d == 0 {
        // G = {1}, the unit ideal's reduced basis in every order.
        return Ok(elements);
    }
    let (basis, rows): (Vec<_>, Vec<_>) = elements.into_iter().unzip();
    let forms = border_forms(field, table, st, &basis, budget)?;
    let variable_count = st.variables.len();
    let (zero, one) = (F::Coef::default(), field.coef(1));

    // The new staircase, its monomials in the order found with their normal
    // forms over B, starts with 1, B[0].
    let first = MonoId::at(0);
    let mut stair: Vec<MonoId> = vec![st.monomials[0]];
    let mut normal: Vec<Row<F>> = vec![Row::unit(first, one)];
    let mut echelon: Vec<Echelon<F>> = vec![Echelon {
        pivot: first,
        row: Row::new(),
        comb: Row::unit(first, one),
    }];
    let mut candidates = Candidates {
        heap: Vec::new(),
        map: HashMap::new(),
    };
    candidates.insert(table, target, &st.variables, stair[0], first, budget)?;

    let mut dense = vec![F::Acc::default(); d];
    let mut combination = vec![F::Acc::default(); d];
    let mut made: Vec<Poly<F, Rank1>> = Vec::new();
    while let Some((m, candidate)) = candidates.pop(table, target) {
        // fglm-lex.9 cited-checked: faugere-1993 §4, Management of NextMonom -- store the number of insertions; the quotient of an element of staircase or MBasis by any variable is in MBasis, so the test is replaced by a count, which is much faster
        // A monomial is a multiple of a leading monomial found before where
        // fewer of its quotients by a variable lie in the new staircase than
        // it has variables.
        let holds = (0..variable_count)
            .filter(|&variable| table.exponent(m, variable) > 0)
            .count();
        // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
        budget.spend(variable_count, Site::Fglm)?;
        if (candidate.count as usize) < holds {
            continue;
        }
        // fglm-lex.10 cited-checked: faugere-1993 §4, Proposition 4.1's proof -- the coordinates of x_j·b_i come from the table C and T(G₁); Management of NextMonom -- store with x_i·m the variable x_i and a pointer to the normal form of m; bigatti-2025 §2.6 -- v := M^(k)·V_l
        let v = product(
            field,
            st,
            &forms,
            candidate.var as usize,
            &normal[candidate.from.index()],
            &mut dense,
            budget,
        )?;
        // fglm-lex.11 design-checked -- linear dependence is tested against a semi-echelon form of the new staircase's normal forms, each row with its combination of them, not bigatti-2025 §2.5's transition matrix P, D × D, with λ = P·v and Gauss–Jordan updates (log 0012-1, 0012-2)
        // SAFETY: `product` reads its row from a dense row of D entries,
        // and `dense` holds D.
        unsafe { field.load(&mut dense, &v.cols, &v.coefs) };
        // call-stop-latency.69 design-checked -- a loop of FGLM bounded by MAX_STAIRCASE entries or rows spends once per loop, not RUN entries at a time, as a loop of 65,535 entries is under a millisecond (log 0138-7)
        budget.spend(v.cols.len() + echelon.len(), Site::Fglm)?;
        for echelon_row in &echelon {
            let a = field.take(&mut dense[echelon_row.pivot.index()]);
            if a == zero {
                continue;
            }
            budget.tick(Site::Fglm)?;
            // SAFETY: an echelon row is read from `dense`. A combination's
            // columns index the new staircase, which holds at most D
            // monomials: the quotient's dimension is the same in every
            // order, and once D are in, every vector reduces to zero.
            unsafe {
                field.sub_mul(&mut dense, &echelon_row.row.cols, &echelon_row.row.coefs, a);
                field.sub_mul(
                    &mut combination,
                    &echelon_row.comb.cols,
                    &echelon_row.comb.coefs,
                    field.neg(a),
                );
            }
            let (row, comb) = (&echelon_row.row, &echelon_row.comb);
            budget.spend(row.cols.len() + comb.cols.len(), Site::Fglm)?;
        }
        // v reduced is v − Σ_k a_k·NF(staircase_k).
        budget.spend(d + stair.len(), Site::Fglm)?;
        let rest = read(field, &mut dense);
        let a = read(field, &mut combination[..stair.len()]);
        if rest.cols.is_empty() {
            // NF(m) = Σ_k a_k·NF(staircase_k): m − Σ_k a_k·staircase_k lies
            // in the ideal, and leads with m, which comes after every
            // monomial of the new staircase.
            let mut terms = vec![(m, one)];
            for (&k, &coef) in a.cols.iter().zip(&a.coefs) {
                terms.push((stair[k.index()], field.neg(coef)));
            }
            made.push(sort::poly(
                &mut terms,
                field,
                table,
                target,
                (),
                budget,
                Site::Fglm,
            )?);
            continue;
        }
        // m joins the new staircase, and v reduced, scaled to 1 at its first
        // column, joins the semi-echelon form.
        // q-learn-time.4 cited-checked: dekleine-2001 §3 -- on each division with a nonzero remainder the list of leading coefficients is checked, and a c_i = 0 means the leading term of r is lost under p_i, with very high probability a bad prime; graebe-1993 §4, independently -- a leading coefficient detected as nonzero that is zero mod p_i shows p_i unlucky
        let Some(scale) = field.inv(rest.coefs[0]) else {
            return Err(Halt::Misfit(0));
        };
        budget.spend(a.cols.len() + rest.cols.len(), Site::Fglm)?;
        let new = MonoId::at(stair.len());
        let mut comb = Row {
            cols: a.cols,
            coefs: a
                .coefs
                .iter()
                .map(|&coef| field.mul(field.neg(coef), scale))
                .collect(),
        };
        comb.cols.push(new);
        comb.coefs.push(scale);
        echelon.push(Echelon {
            pivot: rest.cols[0],
            row: Row {
                cols: rest.cols[1..].to_vec(),
                coefs: rest.coefs[1..]
                    .iter()
                    .map(|&coef| field.mul(coef, scale))
                    .collect(),
            },
            comb,
        });
        stair.push(m);
        normal.push(v);
        candidates.insert(table, target, &st.variables, m, new, budget)?;
    }

    let mut out = Vec::with_capacity(made.len());
    for g in made {
        // fglm-lex.14 design-checked -- the division and the composition run where the cofactor tracking keeps cofactors, Cofactors::TRACKS, a constant of the trait, not a runtime flag or a division in every lex call (not measured)
        // fglm-lex.12 design-checked -- the cofactors of an element g of G₂ are Σ_u q_u·c_u, q_u the quotients of g's division by G₁ and c_u the cofactors of G₁'s elements, not F4 in lex with the cofactor tracking whenever a call asks for cofactors (log 0012-1, 0012-2)
        let element = if C::TRACKS {
            let quotients = divide(field, table, source, &g, &basis, budget)?;
            cof.lift(&quotients, &rows, table, budget)?
        } else {
            cof.lift(&[], &[], table, budget)?
        };
        out.push((g, element));
    }
    Ok(out)
}

/// The quotients of g by the reduced basis `basis` in `order`: per element
/// u, by index, the terms of `q_u` where `q_u` is not zero, with g =
/// Σ `q_u·basis[u]`. The largest term of the remainder is divided first, by
/// the first element whose leading monomial divides it. `Halt::Misfit(0)`
/// where g leaves a remainder, which a replayed basis that is no Gröbner
/// basis at its prime allows. Ticks the budget once per step, and spends at
/// `Site::Fglm` per step a unit per element its search tests and `PROBE`
/// units its quotient, and per term of the divisor's tail `PROBE` units
/// its product, two `PROBE` for the remainder's map, and two units per
/// level of the heap for a push and its pop, whether the product is pushed
/// or met in the remainder.
pub(crate) fn divide<F: Field, M: Format, O: Order<M>>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    g: &Poly<F, Rank1>,
    basis: &[Poly<F, Rank1>],
    budget: &mut Budget,
) -> Result<Vec<Quotient<F>>, Halt> {
    let zero = F::Coef::default();
    let first =
        |table: &Table<M>, left: MonoId, right: MonoId| table.cmp(order, left, right).is_gt();
    // fglm-lex.13 design-checked -- the division keeps its remainder in a hash map from monomial id to coefficient, with a max-heap of the ids under the order, not a geobucket or a heap of the terms of the products q_u·g_u (not measured)
    // A monomial once taken off the heap does not come back: every monomial
    // pushed after it is smaller. Its entry in `rest` goes with it.
    let mut rest: HashMap<MonoId, F::Coef> = HashMap::with_capacity(g.len());
    let mut heap: Vec<MonoId> = Vec::with_capacity(g.len());
    for (&monomial, &coef) in g.monomials.iter().zip(&g.coefs) {
        rest.insert(monomial, coef);
        push(&mut heap, monomial, |left, right| first(table, left, right));
    }
    // call-stop-latency.32 design-checked -- FGLM's steps spend the elements their scans, sorts, reads, and searches visit beside their ticks, PROBE units a random read of the table, a place, or the multiplication table and a unit per level of a heap, not a tick per step alone (fglm-lex.17) (log 0138-7)
    // call-stop-latency.70 design-checked -- a monomial pushed onto a heap of FGLM spends its pop beside its push, two units per level of the heap and PROBE for its map's entry, not the pop spending its own levels (log 0138-7)
    budget.spend(g.len() * (3 * PROBE + 2 * levels(heap.len())), Site::Fglm)?;
    let mut quotients: Vec<Vec<(MonoId, F::Coef)>> = vec![Vec::new(); basis.len()];
    while let Some(monomial) = pop(&mut heap, |left, right| first(table, left, right)) {
        #[allow(
            clippy::expect_used,
            reason = "the heap and rest gain and lose a monomial together"
        )]
        let coef = rest
            .remove(&monomial)
            .expect("a monomial in the heap is in the remainder");
        if coef == zero {
            continue;
        }
        budget.tick(Site::Fglm)?;
        let found = basis
            .iter()
            .position(|element| table.divides(element.lead(), monomial));
        budget.spend(found.map_or(basis.len(), |u| u + 1) + PROBE, Site::Fglm)?;
        let Some(u) = found else {
            return Err(Halt::Misfit(0));
        };
        let divisor = &basis[u];
        debug_assert_eq!(field.value(divisor.coefs[0]), 1, "the basis is monic");
        let multiplier = table.div(monomial, divisor.lead(), budget)?;
        quotients[u].push((multiplier, coef));
        for (&term, &term_coef) in divisor.monomials[1..].iter().zip(&divisor.coefs[1..]) {
            let product = table.mul(multiplier, term, budget)?;
            let entry = rest.entry(product).or_insert_with(|| {
                push(&mut heap, product, |left, right| first(table, left, right));
                zero
            });
            *entry = field.sub(*entry, field.mul(coef, term_coef));
        }
        let tail = divisor.len() - 1;
        budget.spend(tail * (3 * PROBE + 2 * levels(heap.len())), Site::Fglm)?;
    }
    Ok(quotients
        .into_iter()
        .enumerate()
        .filter(|(_, q)| !q.is_empty())
        .map(|(u, q)| (u.narrow(), q))
        .collect())
}
