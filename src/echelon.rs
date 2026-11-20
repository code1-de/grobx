//! The generators of a run in reduced row echelon form, as vectors over
//! their monomials (input-redundancy-time.1 to .6), at a learn or as a
//! replay's record has them (run-generators-time.5 to .8, .10 to .12); and
//! over GF(2) the rows that the linear rows and the field equations put in
//! the ideal of the others left out (bivium-rest-gap.1 to .3).

#[cfg(test)]
use std::cell::Cell;

use crate::Stop;
use crate::budget::{Budget, INVERSE, PROBE, Site};
use crate::field::Field;
use crate::halt::Halt;
use crate::id::{MonoId, RowId};
use crate::module::Module;
use crate::monomial::Format;
use crate::order::Order;
use crate::poly::Poly;
use crate::sort::{self, RUN, sort_runs};
use crate::table::{Limit, Table, monomial_units};

#[cfg(test)]
thread_local! {
    // bivium-rest-gap.7 design-checked -- in the test profile a thread-local, true by default, turns the squares' rule off, so that test code compares a call with and without it, not a parameter of the call, which a caller has no use for (not measured)
    /// The test profile's switch of the squares' rule on this thread
    /// (`squares_out`): false leaves every row of the echelon as it is.
    pub(crate) static SQUARES_OUT: Cell<bool> = const { Cell::new(true) };
}

/// Whether a run over GF(2) leaves the squares of its linear rows out of
/// its echelon's rows (`squares_out`): always, or in the test profile what
/// `SQUARES_OUT` holds.
pub(crate) fn squares_out_on() -> bool {
    #[cfg(test)]
    if !SQUARES_OUT.get() {
        return false;
    }
    true
}

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
    /// generator's position, a unit an entry spent at `Site::Echelon`. A
    /// stop where the budget's check stops the call.
    fn into_echelon(self, budget: &mut Budget) -> Result<Echelon<F, D>, Stop> {
        let Matrix {
            monomials,
            rows,
            positions,
        } = self;
        let mut polys = Vec::with_capacity(rows.len());
        for (row, position) in rows.into_iter().zip(positions) {
            polys.push(Poly {
                monomials: row.cols.iter().map(|c| monomials[c.index()]).collect(),
                coefs: row.coefs,
                position,
            });
            budget.spend(row.cols.len(), Site::Echelon)?;
        }
        Ok(Echelon {
            rows: polys,
            columns: monomials.len(),
        })
    }
}

// call-stop-latency.26 design-checked -- the echelon spends its gathering, numbering, loops over the columns, copies, scalings, loads, scans, marks, and sorts as it goes, a unit an entry or word and PROBE a random access, not its sort and its subtractions alone with one spend of the terms after the matrix is made (log 0138-6)
/// The generators' matrix: a column per distinct monomial in decreasing
/// `order`, each monomial's column set in `table`, and a row per generator.
/// It spends at `Site::Echelon`: the monomials gathered a unit a term per
/// generator, their sort what `sort::sort_decreasing` spends, the repeats'
/// removal a unit a term and each column numbered `PROBE` units, both
/// `sort::RUN` at a time, and each row's columns read `PROBE` units a term.
/// A stop where the budget's check stops the call, and `Halt::Size` where
/// the generators are more than a u32 id numbers.
fn columns_and_rows<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    budget: &mut Budget,
) -> Result<Matrix<F, D>, Halt> {
    RowId::new(generators.len())?;
    let terms = generators.iter().map(Poly::len).sum();
    let mut monomials: Vec<MonoId> = Vec::with_capacity(terms);
    for g in &generators {
        monomials.extend_from_slice(&g.monomials);
        budget.spend(g.len(), Site::Echelon)?;
    }
    sort::sort_decreasing(table, order, &mut monomials, budget, Site::Echelon)?;
    // A sort brings equal ids together: the order is total on monomials.
    let mut kept = 0;
    for start in (0..monomials.len()).step_by(RUN) {
        let end = (start + RUN).min(monomials.len());
        for i in start..end {
            if kept == 0 || monomials[kept - 1] != monomials[i] {
                monomials[kept] = monomials[i];
                kept += 1;
            }
        }
        budget.spend(end - start, Site::Echelon)?;
    }
    monomials.truncate(kept);
    for (k, chunk) in monomials.chunks(RUN).enumerate() {
        for (c, &u) in chunk.iter().enumerate() {
            table.set_column(u, MonoId::at(k * RUN + c));
        }
        budget.spend(chunk.len() * PROBE, Site::Echelon)?;
    }
    let mut positions = Vec::with_capacity(generators.len());
    let mut rows = Vec::with_capacity(generators.len());
    for g in generators {
        positions.push(g.position);
        budget.spend(g.len() * PROBE, Site::Echelon)?;
        rows.push(Row {
            cols: g.monomials.iter().map(|&u| table.column(u)).collect(),
            coefs: g.coefs,
        });
    }
    Ok(Matrix {
        monomials,
        rows,
        positions,
    })
}

// input-redundancy-time.1 cited-checked: buchberger-1985 §6.4 (2) -- the reduction of the polynomials modulo the other polynomials in the basis should also be performed at the beginning of the algorithm
// input-redundancy-time.2 design-checked -- the reduction is linear, Gaussian elimination on the matrix of the generators' coefficients, not normal forms modulo the others' multiples (log 0068-1)
// input-redundancy-time.3 design-checked -- each row stands at the position of the generator whose row won its column, and a generator whose row became zero stands as zero, not the rows sorted by their leading monomials (log 0068-1)
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
    budget.spend(pivot.len(), Site::Echelon)?;
    forward(field, &mut matrix.rows, &mut pivot, budget)?;
    back(field, &mut matrix.rows, &pivot, budget)?;
    Ok(matrix.into_echelon(budget)?)
}

// bivium-rest-gap.1 cited-checked: bardet-2004 §1.5.2, Proposition 1.5.2 -- where the field equations lie in the ideal, the reductions to zero that come from the relations f^2 = f are removed, their rows linear combinations of the rows before them; design: over GF(2) a generator that is the square of a linear generator less the field equations of its tail's variables, x_v^2 + L = (x_v + L)^2 + Σ (x_j^2 + x_j), found among the echelon's rows and left out before the run, not entered and reduced to zero at its own signature in the signature run's first round, its S-pairs with every other input formed and eliminated first (log 0181-1)
// bivium-rest-gap.2 design-checked -- the rule reads the echelon's rows, in which a linear row's tail holds no variable that a linear row leads and the field equation of every other variable stands as read, not the generators as read, whose field equation of x_v holds x_v and whose linear generators' tails may share variables (not measured)
// bivium-rest-gap.3 design-checked -- a row left out stands as zero at its generator's position, as a generator the echelon finds dependent (input-redundancy-time.3), not taken out of the rows (not measured)
/// Over GF(2), leaves out as zero each row of `rows`, the generators in
/// reduced row echelon form (`echelon`), that is `x_v^2 + L` for a linear
/// row `x_v + L`, where every variable of L has its field equation
/// `x_j^2 + x_j` among the rows: `x_v^2 + L = (x_v + L)^2 + Σ_j (x_j^2 + x_j)`
/// over GF(2), so the row lies in the ideal of the rows that stay. A linear
/// row has a leading monomial of degree 1 and no term of a degree past 1;
/// the field equation of `x_j` is the row `x_j^2 + x_j`. The rule reads the linear rows
/// and the field equations alone and leaves out neither, so the rows'
/// ideal stays. The number of rows left out. Each row's terms are read in
/// a first pass and again in a second, a unit a row and a unit a term
/// spent at `Site::Echelon` after each `RUN` rows; a stop where the budget's check
/// stops the call, the rows then part left out, which the stopped run
/// drops.
pub(crate) fn squares_out<F: Field, M: Format, D: Module>(
    table: &Table<M>,
    rows: &mut [Poly<F, D>],
    budget: &mut Budget,
) -> Result<usize, Stop> {
    let format = table.format();
    let n = format.variables();
    // Per variable, the linear row it leads, and whether its field equation
    // stands among the rows.
    let mut linear: Vec<Option<usize>> = vec![None; n];
    let mut field = vec![false; n];
    let mut held = Vec::new();
    // The variable and the degree of a monomial of one variable; None for
    // the constant and for a monomial of several.
    let mut single = |m: MonoId| {
        format.held(table.monomial(m), &mut held);
        (held.len() == 1).then(|| (held[0] as usize, table.degree(m)))
    };
    for start in (0..rows.len()).step_by(RUN) {
        let end = (start + RUN).min(rows.len());
        let mut terms = 0;
        for (k, g) in rows.iter().enumerate().take(end).skip(start) {
            terms += g.len();
            if g.is_zero() {
                continue;
            }
            let linear_tail = g.monomials[1..].iter().all(|&m| table.degree(m) <= 1);
            match single(g.lead()) {
                Some((v, 1)) if linear_tail => linear[v] = Some(k),
                Some((v, 2)) if g.len() == 2 && single(g.monomials[1]) == Some((v, 1)) => {
                    field[v] = true;
                }
                Some(_) | None => {}
            }
        }
        budget.spend(end - start + terms, Site::Echelon)?;
    }
    let mut out = 0;
    for start in (0..rows.len()).step_by(RUN) {
        let end = (start + RUN).min(rows.len());
        let mut terms = 0;
        for k in start..end {
            terms += rows[k].len();
            if rows[k].is_zero() {
                continue;
            }
            let Some((v, 2)) = single(rows[k].lead()) else {
                continue;
            };
            let Some(l) = linear[v] else {
                continue;
            };
            let (g, l) = (&rows[k], &rows[l]);
            if g.monomials[1..] != l.monomials[1..] || g.coefs[1..] != l.coefs[1..] {
                continue;
            }
            let fields = l.monomials[1..]
                .iter()
                .all(|&m| table.degree(m) == 0 || single(m).is_some_and(|(j, _)| field[j]));
            if fields {
                rows[k].monomials.clear();
                rows[k].coefs.clear();
                out += 1;
            }
        }
        budget.spend(end - start + terms, Site::Echelon)?;
    }
    Ok(out)
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
    let pivot = recorded(table, &matrix.monomials, leads, budget)?;
    follow(field, &mut matrix.rows, &pivot, budget)?;
    back(field, &mut matrix.rows, &pivot, budget)?;
    Ok(matrix.into_echelon(budget)?)
}

// run-generators-time.8 design-checked -- a recorded leading monomial's column is found by interning its exponents and reading the table's column, which must map back to it, not a lookup in the table that inserts nothing (not measured)
/// The pivot row of each column of `monomials` by the record `leads`: the
/// generator whose recorded leading monomial the column's is,
/// `RowId::NONE` for none. `Halt::Misfit(0)` where a recorded leading
/// monomial has another count of variables, is no column's, or is two
/// generators'. Each recorded leading monomial spends its interning
/// (`monomial_units`) and its column's read, `PROBE`, and the pivots' array
/// a unit a column once made, at `Site::Echelon`. A stop where the
/// budget's check stops the call.
fn recorded<M: Format>(
    table: &mut Table<M>,
    monomials: &[MonoId],
    leads: &[Option<Vec<u32>>],
    budget: &mut Budget,
) -> Result<Vec<RowId>, Halt> {
    let n = table.format().variables();
    let mut pivot = vec![RowId::NONE; monomials.len()];
    budget.spend(pivot.len(), Site::Echelon)?;
    for (j, lead) in leads.iter().enumerate() {
        budget.spend(1, Site::Echelon)?;
        let Some(e) = lead else {
            continue;
        };
        budget.spend(monomial_units(n) + PROBE, Site::Echelon)?;
        if e.len() != n {
            return Err(Halt::Misfit(0));
        }
        // The table holds every generator's monomial, so one it cannot
        // hold, past the format or past its ids, is no generator's.
        let u = match table.intern(e, budget) {
            Ok(u) => u,
            Err(Limit::Overflow | Limit::Size) => return Err(Halt::Misfit(0)),
            Err(Limit::Stop(stop)) => return Err(Halt::Stop(stop)),
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

// input-redundancy-time.4 cited-checked: bouillaguet-2016 §2.3 -- when several rows have the same leftmost coefficient, the sparsest row is selected
/// The forward pass, over the columns from the first: of the rows that
/// lead at a column, the sparsest, ties to the least position, is its pivot
/// (`pivot`), made monic, and every other is reduced by it and leads at a
/// later column, or is zero. It spends at `Site::Echelon`: the buckets a
/// unit a column and a unit a row, the loop over the columns a unit a
/// column, `sort::RUN` at a time, and at a column with a pivot the rows
/// that lead there a unit each, the winner's inversion `INVERSE` and its
/// scaling a unit an entry, and each reduced row's copy a unit an entry
/// beside the entries its subtraction subtracts. A stop where the budget's
/// check stops the call.
fn forward<F: Field>(
    field: &F,
    rows: &mut [Row<F>],
    pivot: &mut [RowId],
    budget: &mut Budget,
) -> Result<(), Halt> {
    // Per column, the rows that lead there.
    let mut bucket: Vec<Vec<RowId>> = Vec::with_capacity(pivot.len());
    while bucket.len() < pivot.len() {
        let more = RUN.min(pivot.len() - bucket.len());
        bucket.resize_with(bucket.len() + more, Vec::new);
        budget.spend(more, Site::Echelon)?;
    }
    for (j, row) in rows.iter().enumerate() {
        if let Some(&lead) = row.cols.first() {
            bucket[lead.index()].push(RowId::at(j));
        }
    }
    budget.spend(rows.len(), Site::Echelon)?;
    for start in (0..pivot.len()).step_by(RUN) {
        let end = (start + RUN).min(pivot.len());
        for c in start..end {
            let leading = std::mem::take(&mut bucket[c]);
            let Some(&w) = leading
                .iter()
                .min_by_key(|&&j| (rows[j.index()].cols.len(), j))
            else {
                continue;
            };
            let winner = &mut rows[w.index()];
            budget.spend(leading.len() + INVERSE + winner.cols.len(), Site::Echelon)?;
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
                budget.spend_subtracted(by.cols.len(), Site::Echelon)?;
                budget.spend(row.cols.len(), Site::Echelon)?;
                if let Some(&lead) = reduced.cols.first() {
                    bucket[lead.index()].push(j);
                }
                rows[j.index()] = reduced;
            }
        }
        budget.spend(end - start, Site::Echelon)?;
    }
    Ok(())
}

/// row − a·by, a the leading coefficient of `row`, for two sparse rows that
/// lead at one column, `by` monic: the entries after the leading one,
/// merged, those that cancel left out, and an entry of `by` alone that is
/// zero at every lane with them.
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
            // echelon-lane-zero.1 design-checked -- an entry of the pivot row that `row` lacks is tested against zero as a cancelled one is, and dropped where it is zero at every lane, a zero at one lane and b at the other over a pair, not pushed as it is and left to the pair's fit test where it comes to lead
            let v = field.neg(field.mul(a, by.coefs[y]));
            if v != zero {
                out.push(by.cols[y], v);
            }
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

// input-redundancy-time.5 practice-checked: Gaussian elimination with back substitution -- the reduced echelon form takes two passes, the forward pass by columns, then the back pass over the pivot columns from the last, each pivot row reduced by the final pivot rows right of its lead, not one pass that subtracts each new pivot row from the pivot rows before it as it is found
// input-redundancy-time.6 cited-checked: heisinger-2025 §6.1, Remark 1 -- reductions performed directly within the sparse rows, with no dense buffer, proved several orders of magnitude slower, and hybrids that tracked the buffer's nonzero positions were outperformed by the fully dense approach
/// The back pass, over the pivot columns from the last: a pivot row with an
/// entry at a pivot column right of its lead is reduced in a dense row of
/// the field by the pivot rows of those columns, final already. Such a row
/// holds entries at its own pivot column and at columns of no pivot alone,
/// so the entry loaded at each pivot column is its subtraction's factor.
/// It spends at `Site::Echelon`: the dense row and its marks a unit a
/// column once made, the loop over the columns a unit a column,
/// `sort::RUN` at a time, each pivot row's test a unit an entry, and of a
/// row it reduces the load and the marks a unit an entry, each
/// subtraction's marks a unit an entry beside the entries it subtracts,
/// the sort of the columns the dense row holds, and their take a unit
/// each. A stop where the budget's check stops the call.
fn back<F: Field>(
    field: &F,
    rows: &mut [Row<F>],
    pivot: &[RowId],
    budget: &mut Budget,
) -> Result<(), Halt> {
    let zero = F::Coef::default();
    let mut dense: Vec<F::Acc> = vec![F::Acc::default(); pivot.len()];
    let mut touched = vec![false; pivot.len()];
    budget.spend(2 * pivot.len(), Site::Echelon)?;
    // The columns the dense row holds, as touched.
    let mut list: Vec<MonoId> = Vec::new();
    for chunk in pivot.rchunks(RUN) {
        for &p in chunk.iter().rev() {
            if p == RowId::NONE {
                continue;
            }
            let pivoted = |u: &MonoId| pivot[u.index()] != RowId::NONE;
            budget.spend(rows[p.index()].cols.len(), Site::Echelon)?;
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
            budget.spend(2 * row.cols.len(), Site::Echelon)?;
            for (&u, &a) in row.cols.iter().zip(&row.coefs).skip(1) {
                let q = pivot[u.index()];
                if q == RowId::NONE {
                    continue;
                }
                let by = &rows[q.index()];
                // SAFETY: as above.
                unsafe { field.sub_mul(&mut dense, &by.cols, &by.coefs, a) };
                budget.spend_subtracted(by.cols.len(), Site::Echelon)?;
                for &d in &by.cols {
                    if !touched[d.index()] {
                        touched[d.index()] = true;
                        list.push(d);
                    }
                }
                budget.spend(by.cols.len(), Site::Echelon)?;
            }
            sort_runs(&mut list, MonoId::cmp, budget, Site::Echelon)?;
            let mut out = Row::with_capacity(list.len());
            for &u in &list {
                touched[u.index()] = false;
                let v = field.take(&mut dense[u.index()]);
                if v != zero {
                    out.push(u, v);
                }
            }
            budget.spend(list.len(), Site::Echelon)?;
            rows[p.index()] = out;
        }
        budget.spend(chunk.len(), Site::Echelon)?;
    }
    Ok(())
}

// run-generators-time.6 cited-checked: martani-2012 §1.2 -- reduce a row by all the pivot rows discovered so far, in a dense array; the penalty of copying the sparse row to it and back is negligible against the axpy operations; monagan-2015 §2.2, independently -- a row is moved into a dense buffer and reduced by the pivots, looked up per column
// run-generators-time.7 design-checked -- each row the record names zero is reduced in full by the pivot rows, and an entry left is a misfit, not a random linear combination of those rows reduced once (not measured)
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
/// one lane. It spends at `Site::Echelon`: the dense row, its marks, and
/// the rows' flags a unit each once made, the loop over the columns a unit
/// a column, `sort::RUN` at a time, each pivot row's inversion `INVERSE`
/// and its scaling, load, marks, and kept bits a unit an entry, the words
/// of the marks its takes pass a unit each, and each cancel's marks a unit
/// an entry beside the entries it subtracts. A stop where the budget's
/// check stops the call.
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
    budget.spend(columns + held.words.len() + 2 * rows.len(), Site::Echelon)?;
    for (k, chunk) in pivot.chunks(RUN).enumerate() {
        for (i, &p) in chunk.iter().enumerate() {
            let c = k * RUN + i;
            if p == RowId::NONE {
                continue;
            }
            pivoted[p.index()] = true;
            let row = &mut rows[p.index()];
            // The inversion, and the row scaled and kept or loaded and marked.
            budget.spend(INVERSE + 2 * row.cols.len(), Site::Echelon)?;
            // run-generators-time.11 cited-checked: monagan-2015 §2.2 -- rows whose leading term is a new pivot are inserted directly into the pivots array, bypassing the buffer; heisinger-2025 §6.1, independently -- a row whose first column has no pivot yet is not reduced, only normalized so that its pivot is 1
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
            budget.spend(2 * out.cols.len() + held.passed(), Site::Echelon)?;
            rows[p.index()] = out;
        }
        budget.spend(chunk.len(), Site::Echelon)?;
    }
    for (j, &is_pivot) in pivoted.iter().enumerate() {
        budget.spend(1, Site::Echelon)?;
        if is_pivot || rows[j].cols.is_empty() {
            continue;
        }
        let row = std::mem::take(&mut rows[j]);
        budget.spend(2 * row.cols.len(), Site::Echelon)?;
        // SAFETY: as above.
        unsafe { field.load(&mut dense, &row.cols, &row.coefs) };
        held.start(&row.cols);
        while let Some(u) = held.take_below(columns) {
            cancel(field, &mut dense, &mut held, rows, pivot, u, budget)?;
        }
        budget.spend(held.passed(), Site::Echelon)?;
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
    budget.spend_subtracted(by.cols.len(), Site::Echelon)?;
    held.mark_row(q.index(), &by.cols[1..]);
    budget.spend(by.cols.len(), Site::Echelon)?;
    Ok(())
}

/// The bits of a u64 word of `Held`.
const WORD: usize = u64::BITS as usize;

// run-generators-time.10 design-checked -- the replay's forward pass takes the dense row's columns in increasing order from a bit set of the columns it holds, read a word at a time from the row's first column, not a list of the columns sorted again where a subtraction adds some, nor the dense row's entries read one by one (log 0087-3)
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
    /// The words `take_below` stepped past since the last `passed`.
    stepped: usize,
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
            stepped: 0,
        }
    }

    /// The words `take_below` stepped past since the last call, which the
    /// caller spends.
    fn passed(&mut self) -> usize {
        std::mem::take(&mut self.stepped)
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

    // run-generators-time.12 design-checked -- a pivot row whose columns after its first span fewer words than they number keeps their bits, and a subtraction of it ORs those words into the set, not a bit set per column at every subtraction (log 0087-3)
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
            self.stepped += 1;
        }
        None
    }

    /// Column u no longer held.
    fn release(&mut self, u: usize) {
        self.words[u / WORD] &= !(1u64 << (u % WORD));
    }
}
