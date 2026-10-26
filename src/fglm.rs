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

use crate::budget::Budget;
use crate::cofactors::{Cofactors, Quotient};
use crate::f4::{Elements, Halt};
use crate::field::Field;
use crate::id::{BorderId, MonoId};
use crate::module::Rank1;
use crate::monomial::Format;
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::Poly;
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
    /// Entry b·n + j is where `x_j·B`[b] lies.
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

    /// Where `x_j·B`[b] lies.
    fn next(&self, b: MonoId, j: usize) -> Place {
        self.next[b.index() * self.variables.len() + j]
    }

    /// Sorts B and the border by increasing `order`, and the places follow.
    fn sort<M: Format, O: Order<M>>(&mut self, table: &Table<M>, order: &O) {
        let n = self.variables.len();
        // The old index at each new one, and the new index at each old one.
        let sorted = |list: &[MonoId]| {
            let mut old: Vec<u32> = (0..list.len().narrow()).collect();
            old.sort_unstable_by(|&a, &b| table.cmp(order, list[a as usize], list[b as usize]));
            let mut new = vec![0u32; old.len()];
            for (k, &o) in old.iter().enumerate() {
                new[o as usize] = k.narrow();
            }
            (old, new)
        };
        let (inside_old, inside_new) = sorted(&self.monomials);
        let (border_old, border_new) = sorted(&self.border);
        let renamed = |p: Place| match p {
            Place::Inside(i) => Place::Inside(MonoId::at(inside_new[i.index()] as usize)),
            Place::Border(k) => Place::Border(BorderId::at(border_new[k.index()] as usize)),
        };
        let mut next = Vec::with_capacity(self.next.len());
        for &o in &inside_old {
            let o = o as usize;
            next.extend(self.next[o * n..(o + 1) * n].iter().map(|&p| renamed(p)));
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
        for p in self.place.values_mut() {
            *p = renamed(*p);
        }
    }
}

/// Whether every variable has a pure power, or 1, among `leads`: whether
/// the ideal whose reduced basis has these leading monomials is
/// zero-dimensional (buchberger-1985 §6.6, Method 6.9).
pub(crate) fn zero_dimensional<M: Format>(table: &Table<M>, leads: &[MonoId]) -> bool {
    let n = table.format().variables();
    (0..n).all(|i| {
        leads
            .iter()
            .any(|&m| (0..n).all(|k| k == i || table.exponent(m, k) == 0))
    })
}

/// The largest staircase the change takes (monomial-formats.11).
pub(crate) const MAX_STAIRCASE: usize = 65535;

/// The staircase of the reduced basis with leading monomials `leads` in
/// `order`; None where the ideal is not zero-dimensional. Where the
/// staircase passes a bound, the first it passes decides: `Halt::Overflow`
/// past `M::MAX_DEGREE` monomials (monomial-formats.10), None past
/// `MAX_STAIRCASE`. Ticks the budget once per staircase monomial.
pub(crate) fn staircase<M: Format, O: Order<M>>(
    table: &mut Table<M>,
    order: &O,
    leads: &[MonoId],
    budget: &mut Budget,
) -> Result<Option<Staircase>, Halt> {
    if !zero_dimensional(table, leads) {
        return Ok(None);
    }
    let variable_count = table.format().variables();
    let mut exponents = vec![0; variable_count];
    let mut variables = Vec::with_capacity(variable_count);
    for variable in 0..variable_count {
        exponents[variable] = 1;
        variables.push(table.intern(&exponents)?);
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
    let one = table.intern(&exponents)?;
    st.monomials.push(one);
    st.place.insert(one, Place::Inside(MonoId::at(0)));
    let mut position = 0;
    while position < st.monomials.len() {
        // fglm-lex.17 design -- the staircase ticks the budget once per staircase monomial, not a full check once per tested monomial (not measured)
        budget.tick()?;
        let monomial = st.monomials[position];
        for variable in 0..variable_count {
            let product = table.mul(st.variables[variable], monomial)?;
            let place = if let Some(&known) = st.place.get(&product) {
                known
            } else {
                let new_place = if leads.iter().any(|&lead| table.divides(lead, product)) {
                    st.border.push(product);
                    Place::Border(BorderId::at(st.border.len() - 1))
                } else {
                    // monomial-formats.11 design -- the change takes a staircase of at most 65535 monomials in every format, and one larger takes F4 in lex, not FGLM for every D the format holds (log 0013-1)
                    if st.monomials.len() >= MAX_STAIRCASE {
                        return Ok(None);
                    }
                    // monomial-formats.10 design -- a staircase past the format's degree bound is an overflow, and the call starts over in a wider format, not F4 in lex (log 0013-1)
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
        position += 1;
    }
    st.sort(table, order);
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
/// zero entries and is left so. Each border row added ticks the budget.
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
    for (&b, &c) in v.cols.iter().zip(&v.coefs) {
        match st.next(b, j) {
            Place::Inside(_) => {}
            Place::Border(k) => {
                budget.tick()?;
                let row = &forms[k.index()];
                // SAFETY: a normal form's columns index the staircase, a
                // coefficient each.
                unsafe { field.sub_mul(dense, &row.cols, &row.coefs, field.neg(c)) };
            }
        }
    }
    Ok(read(field, dense))
}

// fglm-lex.7 cited: faugere-1993 §3, Proposition 3.1 and Procedure 3.1 -- the columns are built in the order the x_i·b_k appear in B ∪ M, each a leading term's tail or x_l·m' with m' computed before, O(nD(I)³) in all; faugere-2017 §6.1 -- a column is a unit vector, read from an element of G, or a normal form to compute
/// The multiplication table: the normal form of every border monomial of
/// `staircase` modulo `basis`, the reduced basis it is the staircase of, as
/// a row over the staircase, by the border's index. Each border row added
/// ticks the budget. `Halt::Misfit(0)` where a tail monomial of `basis`
/// lies outside the staircase: `basis` is not reduced.
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
            forms.push(row);
            continue;
        }
        // Proposition 2.1 (ii): m = x_l·m' with m' on the border, and m' < m
        // stands before m.
        let mut before = None;
        for l in 0..n {
            if table.exponent(m, l) == 0 {
                continue;
            }
            let q = table.div(m, st.variables[l])?;
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

// fglm-lex.8 cited: faugere-1993 Theorem 5.1 -- the monomial work takes O(n²D²) with an elementary implementation of ListOfNexts, O(n²D log(nD)) where the list is an efficient priority queue
/// The monomials waiting to be tested: a binary heap of their ids under the
/// target order, the least on top, and each one's candidate.
struct Candidates {
    heap: Vec<MonoId>,
    map: HashMap<MonoId, Candidate>,
}

impl Candidates {
    /// Inserts `x_j·m` for every variable `x_j`, m the new staircase's monomial
    /// at index `from`.
    fn insert<M: Format, O: Order<M>>(
        &mut self,
        table: &mut Table<M>,
        target: &O,
        variables: &[MonoId],
        m: MonoId,
        from: MonoId,
    ) -> Result<(), Halt> {
        for (j, &x) in variables.iter().enumerate() {
            let t = table.mul(x, m)?;
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
        Ok(())
    }

    /// The least monomial waiting, and its candidate. A monomial taken does
    /// not come back: every monomial inserted after it is larger.
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
/// the start and ticked once per row operation and per division step, and
/// the cofactors' lift spends what `Cofactors::lift` says. `Halt::Misfit(0)`
/// where a vector that joins the echelon form has no inverse at its first
/// entry: over several lanes, 0 at one lane. `Halt::Misfit(0)` too where
/// `elements` is not reduced (`border_forms`), and where `C` tracks
/// cofactors and an element leaves a remainder in its division by
/// `elements` (`divide`).
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
    // fglm-lex.17 design -- the change ticks the budget once per row operation and per division step, and checks it fully at its start, not a full check once per tested monomial (not measured)
    budget.check(|| {})?;
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
    candidates.insert(table, target, &st.variables, stair[0], first)?;

    let mut dense = vec![F::Acc::default(); d];
    let mut combination = vec![F::Acc::default(); d];
    let mut made: Vec<Poly<F, Rank1>> = Vec::new();
    while let Some((m, candidate)) = candidates.pop(table, target) {
        // fglm-lex.9 cited: faugere-1993 §4, Management of NextMonom -- store the number of insertions; the quotient of an element of staircase or MBasis by any variable is in MBasis, so the test is replaced by a count, which is much faster
        // A monomial is a multiple of a leading monomial found before where
        // fewer of its quotients by a variable lie in the new staircase than
        // it has variables.
        let holds = (0..variable_count)
            .filter(|&variable| table.exponent(m, variable) > 0)
            .count();
        if (candidate.count as usize) < holds {
            continue;
        }
        // fglm-lex.10 cited: faugere-1993 §4, Proposition 4.1's proof -- the coordinates of x_j·b_i come from the table C and T(G₁); Management of NextMonom -- store with x_i·m the variable x_i and a pointer to the normal form of m; bigatti-2025 §2.6 -- v := M^(k)·V_l
        let v = product(
            field,
            st,
            &forms,
            candidate.var as usize,
            &normal[candidate.from.index()],
            &mut dense,
            budget,
        )?;
        // fglm-lex.11 cited: faugere-1993 §4, Proposition 4.1's proof -- if an echelon form of the matrix C is maintained, for testing linear independency and incrementing C, only O(D(I)²) are needed
        // SAFETY: `product` reads its row from a dense row of D entries,
        // and `dense` holds D.
        unsafe { field.load(&mut dense, &v.cols, &v.coefs) };
        for echelon_row in &echelon {
            let a = field.take(&mut dense[echelon_row.pivot.index()]);
            if a == zero {
                continue;
            }
            budget.tick()?;
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
        }
        // v reduced is v − Σ_k a_k·NF(staircase_k).
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
            made.push(Poly::from_terms(&mut terms, field, table, target, ()));
            continue;
        }
        // m joins the new staircase, and v reduced, scaled to 1 at its first
        // column, joins the semi-echelon form.
        // q-learn-time.4 cited: dekleine-2001 §3 -- on each division with a nonzero remainder the list of leading coefficients is checked, and a c_i = 0 means the leading term of r is lost under p_i, with very high probability a bad prime; graebe-1993 §4, independently -- a leading coefficient detected as nonzero that is zero mod p_i shows p_i unlucky
        let Some(scale) = field.inv(rest.coefs[0]) else {
            return Err(Halt::Misfit(0));
        };
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
        candidates.insert(table, target, &st.variables, m, new)?;
    }

    let mut out = Vec::with_capacity(made.len());
    for g in made {
        // fglm-lex.14 design -- the division and the composition run where the cofactor tracking keeps cofactors, Cofactors::TRACKS, a constant of the trait, not a runtime flag or a division in every lex call (not measured)
        // fglm-lex.12 cited: buchberger-1985 §6.7, Method 6.13 -- reduce f to zero modulo G and collect the multiples of the g_i used in the reduction; Method 6.17 -- the representations composed with the transformation matrix X of G in F
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
/// Σ `q_u·basis`[u]. The largest term of the remainder is divided first, by
/// the first element whose leading monomial divides it. `Halt::Misfit(0)`
/// where g leaves a remainder, which a replayed basis that is no Gröbner
/// basis at its prime allows. Ticks the budget once per step.
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
    // fglm-lex.13 design -- the division keeps its remainder in a hash map from monomial id to coefficient, with a max-heap of the ids under the order, not a geobucket or a heap of the terms of the products q_u·g_u (not measured)
    // A monomial once taken off the heap does not come back: every monomial
    // pushed after it is smaller. Its entry in `rest` goes with it.
    let mut rest: HashMap<MonoId, F::Coef> = HashMap::with_capacity(g.len());
    let mut heap: Vec<MonoId> = Vec::with_capacity(g.len());
    for (&monomial, &coef) in g.monomials.iter().zip(&g.coefs) {
        rest.insert(monomial, coef);
        push(&mut heap, monomial, |left, right| first(table, left, right));
    }
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
        budget.tick()?;
        let Some(u) = basis
            .iter()
            .position(|element| table.divides(element.lead(), monomial))
        else {
            return Err(Halt::Misfit(0));
        };
        let divisor = &basis[u];
        debug_assert_eq!(field.value(divisor.coefs[0]), 1, "the basis is monic");
        let multiplier = table.div(monomial, divisor.lead())?;
        quotients[u].push((multiplier, coef));
        for (&term, &term_coef) in divisor.monomials[1..].iter().zip(&divisor.coefs[1..]) {
            let product = table.mul(multiplier, term)?;
            let entry = rest.entry(product).or_insert_with(|| {
                push(&mut heap, product, |left, right| first(table, left, right));
                zero
            });
            *entry = field.sub(*entry, field.mul(coef, term_coef));
        }
    }
    Ok(quotients
        .into_iter()
        .enumerate()
        .filter(|(_, q)| !q.is_empty())
        .map(|(u, q)| (u.narrow(), q))
        .collect())
}
