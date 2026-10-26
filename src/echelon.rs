//! The generators of a run in reduced row echelon form, as vectors over
//! their monomials (input-redundancy-time.1 to .6), at a learn or as a
//! replay's record has them (run-generators-time.5 to .8, .10 to .12).

use crate::budget::Budget;
use crate::f4::Halt;
use crate::field::Field;
use crate::id::{MonoId, RowId};
use crate::matrix;
use crate::module::Module;
use crate::monomial::Format;
use crate::order::Order;
use crate::poly::Poly;
use crate::table::{Limit, Table};

/// The generators of a run in reduced row echelon form (`echelon`).
pub(crate) struct Echelon<F: Field, D: Module> {
    /// One row per generator, in its position: the pivot row of the
    /// column it won, monic, or zero.
    pub(crate) rows: Vec<Poly<F, D>>,
    /// The matrix's columns: the generators' distinct monomials.
    pub(crate) columns: usize,
}

/// A sparse row of the echelon's matrix: its columns, increasing, and its
/// coefficients, none zero, in the same order.
struct Row<F: Field> {
    cols: Vec<MonoId>,
    coefs: Vec<F::Coef>,
}

impl<F: Field> Default for Row<F> {
    fn default() -> Self {
        Row {
            cols: Vec::new(),
            coefs: Vec::new(),
        }
    }
}

impl<F: Field> Row<F> {
    fn with_capacity(n: usize) -> Self {
        Row {
            cols: Vec::with_capacity(n),
            coefs: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, col: MonoId, coef: F::Coef) {
        self.cols.push(col);
        self.coefs.push(coef);
    }
}

/// The matrix of the generators' coefficients that `echelon` and `replay`
/// reduce (`columns_and_rows`).
struct Matrix<F: Field, D: Module> {
    /// The columns' monomials: the generators' distinct monomials in
    /// decreasing order.
    monomials: Vec<MonoId>,
    /// One row per generator, in its position.
    rows: Vec<Row<F>>,
    /// Each generator's module position.
    positions: Vec<D::Position>,
}

impl<F: Field, D: Module> Matrix<F, D> {
    /// The rows as polynomials over the columns' monomials, each in its
    /// generator's position.
    fn into_echelon(self) -> Echelon<F, D> {
        let Matrix {
            monomials,
            rows,
            positions,
        } = self;
        let rows = rows
            .into_iter()
            .zip(positions)
            .map(|(row, position)| Poly {
                monomials: row.cols.iter().map(|c| monomials[c.index()]).collect(),
                coefs: row.coefs,
                position,
            })
            .collect();
        Echelon {
            rows,
            columns: monomials.len(),
        }
    }
}

/// The generators' matrix: a column per distinct monomial in decreasing
/// `order`, each monomial's column set in `table`, and a row per generator.
/// The columns' sort spends what `matrix::sort_decreasing` spends, and
/// their numbering a unit a term. A stop where the budget's check stops the
/// call, and `Halt::Size` where the generators are more than a u32 id
/// numbers.
fn columns_and_rows<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    budget: &mut Budget,
) -> Result<Matrix<F, D>, Halt> {
    RowId::new(generators.len())?;
    let mut monomials: Vec<MonoId> = generators
        .iter()
        .flat_map(|g| g.monomials.iter().copied())
        .collect();
    let terms = monomials.len();
    matrix::sort_decreasing(table, order, &mut monomials, budget)?;
    // A sort brings equal ids together: the order is total on monomials.
    monomials.dedup();
    for (c, &u) in monomials.iter().enumerate() {
        table.set_column(u, MonoId::at(c));
    }
    let mut positions = Vec::with_capacity(generators.len());
    let rows = generators
        .into_iter()
        .map(|g| {
            positions.push(g.position);
            Row {
                cols: g.monomials.iter().map(|&u| table.column(u)).collect(),
                coefs: g.coefs,
            }
        })
        .collect();
    budget.spend(terms)?;
    Ok(Matrix {
        monomials,
        rows,
        positions,
    })
}

// input-redundancy-time.1 cited: buchberger-1985 §6.4 (2) -- the reduction of the polynomials modulo the other polynomials in the basis should also be performed at the beginning of the algorithm
// input-redundancy-time.2 design -- the reduction is linear, Gaussian elimination on the matrix of the generators' coefficients, not normal forms modulo the others' multiples (log 0068-1)
// input-redundancy-time.3 design -- each row stands at the position of the generator whose row won its column, and a generator whose row became zero stands as zero, not the rows sorted by their leading monomials (log 0068-1)
/// The generators in reduced row echelon form: the matrix of their
/// coefficients, a column per monomial in decreasing `order`, a row per
/// generator. Each row stands at its generator's position, and a generator
/// that depends on those whose rows won columns before it is zero. The
/// rows are monic, their leading monomials distinct, and no row holds
/// another's leading monomial. The matrix spends what `columns_and_rows`
/// spends, and each row operation the entries it subtracts
/// (`Budget::spend_subtracted`).
/// `Halt::Misfit(0)` where a pivot's leading coefficient has no inverse:
/// over several lanes, 0 at one lane. A stop where the budget's check stops
/// the call, and `Halt::Size` where the generators are more than a u32 id
/// numbers.
pub(crate) fn echelon<F: Field, M: Format, O: Order<M>, D: Module>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    budget: &mut Budget,
) -> Result<Echelon<F, D>, Halt> {
    let mut matrix = columns_and_rows(table, order, generators, budget)?;
    let mut pivot = vec![RowId::NONE; matrix.monomials.len()];
    forward(field, &mut matrix.rows, &mut pivot, budget)?;
    back(field, &mut matrix.rows, &pivot, budget)?;
    Ok(matrix.into_echelon())
}

/// The generators in reduced row echelon form, as `echelon` makes them,
/// from the record of an `echelon` at another prime: `leads` holds per
/// generator the exponents of its row's leading monomial there, None where
/// its row was zero. Each generator with a recorded leading monomial is
/// the pivot row of its column, and every other reduces to zero
/// (`follow`); the back pass is `echelon`'s. Where `echelon` here gives
/// the recorded leading monomials, the replay gives its rows and spends
/// what it spends. `Halt::Misfit(0)` where the record does not fit:
/// another count of generators or of variables, a recorded leading
/// monomial that no generator holds or that two generators share, and
/// where `follow` misfits. A stop where the budget's check stops the call,
/// and `Halt::Size` where the generators are more than a u32 id numbers.
pub(crate) fn replay<F: Field, M: Format, O: Order<M>, D: Module>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    leads: &[Option<Vec<u32>>],
    budget: &mut Budget,
) -> Result<Echelon<F, D>, Halt> {
    if leads.len() != generators.len() {
        return Err(Halt::Misfit(0));
    }
    let mut matrix = columns_and_rows(table, order, generators, budget)?;
    let pivot = recorded(table, &matrix.monomials, leads)?;
    follow(field, &mut matrix.rows, &pivot, budget)?;
    back(field, &mut matrix.rows, &pivot, budget)?;
    Ok(matrix.into_echelon())
}

// run-generators-time.8 design -- a recorded leading monomial's column is found by interning its exponents and reading the table's column, which must map back to it, not a lookup in the table that inserts nothing (not measured)
/// The pivot row of each column of `monomials` by the record `leads`: the
/// generator whose recorded leading monomial the column's is,
/// `RowId::NONE` for none. `Halt::Misfit(0)` where a recorded leading
/// monomial has another count of variables, is no column's, or is two
/// generators'.
fn recorded<M: Format>(
    table: &mut Table<M>,
    monomials: &[MonoId],
    leads: &[Option<Vec<u32>>],
) -> Result<Vec<RowId>, Halt> {
    let n = table.format().variables();
    let mut pivot = vec![RowId::NONE; monomials.len()];
    for (j, lead) in leads.iter().enumerate() {
        let Some(e) = lead else {
            continue;
        };
        if e.len() != n {
            return Err(Halt::Misfit(0));
        }
        // The table holds every generator's monomial, so one it cannot
        // hold, past the format or past its ids, is no generator's.
        let u = match table.intern(e) {
            Ok(u) => u,
            Err(Limit::Overflow | Limit::Size) => return Err(Halt::Misfit(0)),
        };
        // The column of a monomial no generator holds is none, past the
        // columns, or an earlier matrix's, whose monomial here is another.
        let c = table.column(u).index();
        if monomials.get(c) != Some(&u) || pivot[c] != RowId::NONE {
            return Err(Halt::Misfit(0));
        }
        pivot[c] = RowId::at(j);
    }
    Ok(pivot)
}

// input-redundancy-time.4 cited: bouillaguet-2016 §2.3 -- when several rows have the same leftmost coefficient, the sparsest row is selected
/// The forward pass, over the columns from the first: of the rows that
/// lead at a column, the sparsest, ties to the least position, is its pivot
/// (`pivot`), made monic, and every other is reduced by it and leads at a
/// later column, or is zero.
fn forward<F: Field>(
    field: &F,
    rows: &mut [Row<F>],
    pivot: &mut [RowId],
    budget: &mut Budget,
) -> Result<(), Halt> {
    // Per column, the rows that lead there.
    let mut bucket: Vec<Vec<RowId>> = vec![Vec::new(); pivot.len()];
    for (j, row) in rows.iter().enumerate() {
        if let Some(&lead) = row.cols.first() {
            bucket[lead.index()].push(RowId::at(j));
        }
    }
    for c in 0..pivot.len() {
        let leading = std::mem::take(&mut bucket[c]);
        let Some(&w) = leading
            .iter()
            .min_by_key(|&&j| (rows[j.index()].cols.len(), j))
        else {
            continue;
        };
        let winner = &mut rows[w.index()];
        let Some(s) = field.inv(winner.coefs[0]) else {
            return Err(Halt::Misfit(0));
        };
        for v in &mut winner.coefs {
            *v = field.mul(*v, s);
        }
        pivot[c] = w;
        for &j in leading.iter().filter(|&&j| j != w) {
            let row = std::mem::take(&mut rows[j.index()]);
            let by = &rows[w.index()];
            let reduced = subtract(field, &row, by);
            budget.spend_subtracted(by.cols.len())?;
            if let Some(&lead) = reduced.cols.first() {
                bucket[lead.index()].push(j);
            }
            rows[j.index()] = reduced;
        }
    }
    Ok(())
}

/// row − a·by, a the leading coefficient of `row`, for two sparse rows that
/// lead at one column, `by` monic: the entries after the leading one,
/// merged, those that cancel left out.
fn subtract<F: Field>(field: &F, row: &Row<F>, by: &Row<F>) -> Row<F> {
    debug_assert_eq!(row.cols.first(), by.cols.first(), "one leading column");
    let zero = F::Coef::default();
    let a = row.coefs[0];
    let mut out = Row::with_capacity(row.cols.len() + by.cols.len());
    let (mut x, mut y) = (1, 1);
    while x < row.cols.len() || y < by.cols.len() {
        if y == by.cols.len() || (x < row.cols.len() && row.cols[x] < by.cols[y]) {
            out.push(row.cols[x], row.coefs[x]);
            x += 1;
        } else if x == row.cols.len() || by.cols[y] < row.cols[x] {
            out.push(by.cols[y], field.neg(field.mul(a, by.coefs[y])));
            y += 1;
        } else {
            let v = field.sub(row.coefs[x], field.mul(a, by.coefs[y]));
            if v != zero {
                out.push(row.cols[x], v);
            }
            x += 1;
            y += 1;
        }
    }
    out
}

// input-redundancy-time.5 cited: heisinger-2025 §6.1 -- the reduced row echelon form in two steps, the row echelon form, then backward elimination
// input-redundancy-time.6 cited: martani-2012 §1.2 -- while reducing a row, a dense array represents it, and the penalty of copying the sparse row to it and back is negligible against the axpy operations
/// The back pass, over the pivot columns from the last: a pivot row with an
/// entry at a pivot column right of its lead is reduced in a dense row of
/// the field by the pivot rows of those columns, final already. Such a row
/// holds entries at its own pivot column and at columns of no pivot alone,
/// so the entry loaded at each pivot column is its subtraction's factor.
fn back<F: Field>(
    field: &F,
    rows: &mut [Row<F>],
    pivot: &[RowId],
    budget: &mut Budget,
) -> Result<(), Halt> {
    let zero = F::Coef::default();
    let mut dense: Vec<F::Acc> = vec![F::Acc::default(); pivot.len()];
    let mut touched = vec![false; pivot.len()];
    // The columns the dense row holds, as touched.
    let mut list: Vec<MonoId> = Vec::new();
    for &p in pivot.iter().rev() {
        if p == RowId::NONE {
            continue;
        }
        let pivoted = |u: &MonoId| pivot[u.index()] != RowId::NONE;
        if !rows[p.index()].cols[1..].iter().any(pivoted) {
            continue;
        }
        let row = std::mem::take(&mut rows[p.index()]);
        // SAFETY: every column of a row was numbered below the columns'
        // count, the dense row's length, and a row's columns and
        // coefficients have one length.
        unsafe { field.load(&mut dense, &row.cols, &row.coefs) };
        list.clear();
        for &u in &row.cols {
            touched[u.index()] = true;
            list.push(u);
        }
        for (&u, &a) in row.cols.iter().zip(&row.coefs).skip(1) {
            let q = pivot[u.index()];
            if q == RowId::NONE {
                continue;
            }
            let by = &rows[q.index()];
            // SAFETY: as above.
            unsafe { field.sub_mul(&mut dense, &by.cols, &by.coefs, a) };
            budget.spend_subtracted(by.cols.len())?;
            for &d in &by.cols {
                if !touched[d.index()] {
                    touched[d.index()] = true;
                    list.push(d);
                }
            }
        }
        list.sort_unstable();
        let mut out = Row::with_capacity(list.len());
        for &u in &list {
            touched[u.index()] = false;
            let v = field.take(&mut dense[u.index()]);
            if v != zero {
                out.push(u, v);
            }
        }
        rows[p.index()] = out;
    }
    Ok(())
}

// run-generators-time.6 cited: martani-2012 §1.2 -- reduce a row by all the pivot rows discovered so far, in a dense array; the penalty of copying the sparse row to it and back is negligible against the axpy operations; monagan-2015 §2.2, independently -- a row is moved into a dense buffer and reduced by the pivots, looked up per column
// run-generators-time.7 design -- each row the record names zero is reduced in full by the pivot rows, and an entry left is a misfit, not a random linear combination of those rows reduced once (not measured)
/// The replay's forward pass: the rows of `forward` where its pivots are
/// `pivot`'s. The pivot rows by increasing column, each reduced once in a
/// dense row of the field by the pivot rows of the columns before its own
/// and made monic; then every other row reduced in full by the pivot rows.
/// The rows are `forward`'s: a pivot row's is the one row of the span of
/// the pivot rows up to its column that is zero before the column and 1
/// at it. Each subtraction spends what `forward`'s spends.
/// `Halt::Misfit(0)` where an entry is left at a column of no pivot, before
/// a pivot row's column or anywhere in another row, and where a pivot
/// row's entry at its column has no inverse: 0, or over several lanes 0 at
/// one lane. A stop where the budget's check stops the call.
fn follow<F: Field>(
    field: &F,
    rows: &mut [Row<F>],
    pivot: &[RowId],
    budget: &mut Budget,
) -> Result<(), Halt> {
    let zero = F::Coef::default();
    let one = field.coef(1);
    let columns = pivot.len();
    let mut dense: Vec<F::Acc> = vec![F::Acc::default(); columns];
    let mut held = Held::new(columns, rows.len());
    let mut pivoted = vec![false; rows.len()];
    for (c, &p) in pivot.iter().enumerate() {
        if p == RowId::NONE {
            continue;
        }
        pivoted[p.index()] = true;
        let row = &mut rows[p.index()];
        // run-generators-time.11 cited: monagan-2015 §2.2 -- rows whose leading term is a new pivot are inserted directly into the pivots array, bypassing the buffer; heisinger-2025 §6.1, independently -- a row whose first column has no pivot yet is not reduced, only normalized so that its pivot is 1
        if row.cols.first() == Some(&MonoId::at(c)) {
            let Some(s) = field.inv(row.coefs[0]) else {
                return Err(Halt::Misfit(0));
            };
            for v in &mut row.coefs {
                *v = field.mul(*v, s);
            }
            held.keep(p.index(), &row.cols[1..]);
            continue;
        }
        let row = std::mem::take(row);
        // SAFETY: every column of a row was numbered below the columns'
        // count, the dense row's length, and a row's columns and
        // coefficients have one length.
        unsafe { field.load(&mut dense, &row.cols, &row.coefs) };
        held.start(&row.cols);
        while let Some(u) = held.take_below(c) {
            cancel(field, &mut dense, &mut held, rows, pivot, u, budget)?;
        }
        held.release(c);
        let Some(s) = field.inv(field.take(&mut dense[c])) else {
            return Err(Halt::Misfit(0));
        };
        let mut out = Row::with_capacity(row.cols.len());
        out.push(MonoId::at(c), one);
        while let Some(u) = held.take_below(columns) {
            let v = field.take(&mut dense[u]);
            if v != zero {
                out.push(MonoId::at(u), field.mul(v, s));
            }
        }
        held.keep(p.index(), &out.cols[1..]);
        rows[p.index()] = out;
    }
    for (j, &is_pivot) in pivoted.iter().enumerate() {
        if is_pivot || rows[j].cols.is_empty() {
            continue;
        }
        let row = std::mem::take(&mut rows[j]);
        // SAFETY: as above.
        unsafe { field.load(&mut dense, &row.cols, &row.coefs) };
        held.start(&row.cols);
        while let Some(u) = held.take_below(columns) {
            cancel(field, &mut dense, &mut held, rows, pivot, u, budget)?;
        }
    }
    Ok(())
}

/// Takes the dense row's entry at column u, the least it holds, and
/// cancels it by u's pivot row, final already, whose further columns it
/// holds then. `Halt::Misfit(0)` where the entry is not 0 and u has no
/// pivot row.
fn cancel<F: Field>(
    field: &F,
    dense: &mut [F::Acc],
    held: &mut Held,
    rows: &[Row<F>],
    pivot: &[RowId],
    u: usize,
    budget: &mut Budget,
) -> Result<(), Halt> {
    let v = field.take(&mut dense[u]);
    if v == F::Coef::default() {
        return Ok(());
    }
    let q = pivot[u];
    if q == RowId::NONE {
        return Err(Halt::Misfit(0));
    }
    let by = &rows[q.index()];
    debug_assert_eq!(by.cols.first(), Some(&MonoId::at(u)), "a final pivot row");
    // The pivot row is monic, so its entry at u, 1, cancels v, which the
    // dense row no longer holds: the subtraction takes the entries after.
    // SAFETY: as in `follow`.
    unsafe { field.sub_mul(dense, &by.cols[1..], &by.coefs[1..], v) };
    budget.spend_subtracted(by.cols.len())?;
    held.mark_row(q.index(), &by.cols[1..]);
    Ok(())
}

/// The bits of a u64 word of `Held`.
const WORD: usize = u64::BITS as usize;

// run-generators-time.10 design -- the replay's forward pass takes the dense row's columns in increasing order from a bit set of the columns it holds, read a word at a time from the row's first column, not a list of the columns sorted again where a subtraction adds some, nor the dense row's entries read one by one (log 0087-3)
/// The columns at which a dense row holds entries, a bit each, taken from
/// the least; and the bits of the pivot rows that keep theirs (`keep`).
struct Held {
    words: Vec<u64>,
    /// The word of the least column held, or a word below it.
    next: usize,
    /// The word of the greatest column held, or a word above it.
    last: usize,
    /// Per row, the word of its second column and its bits from there,
    /// where it keeps them; else no words.
    kept: Vec<(usize, Vec<u64>)>,
}

impl Held {
    /// For a dense row of `columns` entries, none held, and `rows` rows
    /// that keep no bits.
    fn new(columns: usize, rows: usize) -> Held {
        Held {
            words: vec![0; columns / WORD + 1],
            next: 0,
            last: 0,
            kept: (0..rows).map(|_| (0, Vec::new())).collect(),
        }
    }

    /// Holds the columns of a row, increasing, where none is held.
    fn start(&mut self, cols: &[MonoId]) {
        debug_assert!(self.words.iter().all(|&w| w == 0), "a column held");
        let first = cols.first().map_or(0, |u| u.index() / WORD);
        self.next = first;
        self.last = first;
        self.mark(cols);
    }

    /// Holds the columns `cols`, increasing, each past the last column
    /// taken.
    fn mark(&mut self, cols: &[MonoId]) {
        for &u in cols {
            self.words[u.index() / WORD] |= 1u64 << (u.index() % WORD);
        }
        if let Some(u) = cols.last() {
            self.last = self.last.max(u.index() / WORD);
        }
    }

    // run-generators-time.12 design -- a pivot row whose columns after its first span fewer words than they number keeps their bits, and a subtraction of it ORs those words into the set, not a bit set per column at every subtraction (log 0087-3)
    /// Keeps the bits of row j's columns after its first, `cols`, where
    /// they span fewer words than they number.
    fn keep(&mut self, j: usize, cols: &[MonoId]) {
        let (Some(first), Some(last)) = (cols.first(), cols.last()) else {
            return;
        };
        let (w, end) = (first.index() / WORD, last.index() / WORD + 1);
        if end - w >= cols.len() {
            return;
        }
        let mut bits = vec![0; end - w];
        for &u in cols {
            bits[u.index() / WORD - w] |= 1u64 << (u.index() % WORD);
        }
        self.kept[j] = (w, bits);
    }

    /// Holds row j's columns after its first, `cols`, each past the last
    /// column taken: by its kept bits where it keeps them.
    fn mark_row(&mut self, j: usize, cols: &[MonoId]) {
        if self.kept[j].1.is_empty() {
            self.mark(cols);
            return;
        }
        let (w, bits) = &self.kept[j];
        for (word, &b) in self.words[*w..*w + bits.len()].iter_mut().zip(bits) {
            *word |= b;
        }
        self.last = self.last.max(*w + bits.len() - 1);
    }

    /// The least column held, where it lies below `end`, no longer held.
    fn take_below(&mut self, end: usize) -> Option<usize> {
        while self.next <= self.last {
            let bits = self.words[self.next];
            if bits != 0 {
                let u = self.next * WORD + bits.trailing_zeros() as usize;
                if u >= end {
                    return None;
                }
                self.words[self.next] = bits & (bits - 1);
                return Some(u);
            }
            self.next += 1;
        }
        None
    }

    /// Column u no longer held.
    fn release(&mut self, u: usize) {
        self.words[u / WORD] &= !(1u64 << (u % WORD));
    }
}
