//! The matrix of a round: its rows t·g, symbolic preprocessing, the columns,
//! the pivots, and the reduction of the rows that are no pivot.

use std::cmp::Ordering;
use std::mem;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stop;
use crate::budget::Budget;
use crate::cofactors::{Cofactors, NoCofactors};
use crate::f4::Halt;
use crate::field::{BAND, Field};
use crate::id::{ElementId, KeepId, MadeId, MonoId, RowId};
use crate::module::Module;
use crate::monomial::{Format, PAYLOAD_BITS};
use crate::narrow::{self, Narrow};
use crate::order::Order;
use crate::poly::{Basis, Poly};
use crate::report::Stage;
use crate::sig::{Sig, Signatures};
use crate::table::{Limit, Table, find_passing};
use crate::trace::RowSource;

// f4-gfp.20 cited: faugere-1999 §3.4 -- coefficients are not duplicated, since some rows are just term multiples of others; demin-2024 §4.1 -- the arrays of coefficients are used both as polynomials and as matrix rows
/// A matrix: its rows t·g, each with its columns, and the monomial of each
/// column. A row's coefficients are its element's. The columns fill the
/// vector the build was handed: in a run's rounds the one the run's buffers
/// kept (`Buffers`), in the inter-reduction a new one.
pub(crate) struct Matrix {
    // round-memory-time.10 design -- a matrix's rows, offsets, and monomials grow by doubling, and no matrix passes them to the next, not each kept from round to round as the column vector is, since they hold 8 bytes a row or 4 a column where the columns hold 4 bytes an entry (not measured)
    /// The rows given, then the reducers symbolic preprocessing added.
    pub(crate) rows: Vec<RowSource>,
    /// How many rows were given.
    pub(crate) given: usize,
    // matrix-slack.5 cited: martani-2012 §3.9 -- where the rows of a matrix A are interleaved in memory with those of B, freeing A does not necessarily free its memory, unless each matrix's memory is allocated on its own
    /// Every row's columns, increasing, back to back, in the column vector
    /// the build was handed, which a run's buffers keep for its next matrix
    /// (`Buffers::keep_cols`), save where a replay keeps the matrix in its
    /// shape.
    cols: Vec<MonoId>,
    /// Row r's columns are `cols[offsets[r]..offsets[r + 1]]`.
    offsets: Vec<usize>,
    /// The monomial of each column, decreasing.
    pub(crate) monomials: Vec<MonoId>,
}

impl Matrix {
    /// A matrix of the rows `given`, all of them given, with no columns yet:
    /// its rows' columns are to fill `cols`, cleared here.
    fn new(given: Vec<RowSource>, mut cols: Vec<MonoId>) -> Matrix {
        cols.clear();
        Matrix {
            given: given.len(),
            rows: given,
            cols,
            offsets: vec![0],
            monomials: Vec::new(),
        }
    }

    pub(crate) fn columns(&self) -> usize {
        self.monomials.len()
    }

    // table-read-time.6 design -- a matrix's row reads its offsets and slices its columns, and the reduction reads a row's source, with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// Row r's columns. The reads have no bounds check outside the test
    /// profile, whose debug assertions check them.
    pub(crate) fn row(&self, r: RowId) -> &[MonoId] {
        debug_assert!(
            r.index() + 1 < self.offsets.len(),
            "row {r} past the matrix's {rows}",
            rows = self.offsets.len() - 1
        );
        // SAFETY: a row index lies below the rows, and `offsets` holds an
        // entry per row and one more, increasing to `cols.len()`.
        unsafe {
            let (a, b) = (
                *self.offsets.get_unchecked(r.index()),
                *self.offsets.get_unchecked(r.index() + 1),
            );
            debug_assert!(
                a <= b && b <= self.cols.len(),
                "row {r}'s offsets {a} and {b}"
            );
            self.cols.get_unchecked(a..b)
        }
    }

    /// Gives back the slack of the growable arrays.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.rows.shrink_to_fit();
        self.cols.shrink_to_fit();
        self.offsets.shrink_to_fit();
        self.monomials.shrink_to_fit();
    }

    /// The column vector, the rest of the matrix dropped.
    pub(crate) fn into_cols(self) -> Vec<MonoId> {
        self.cols
    }

    /// Row r's leading column.
    fn lead(&self, r: RowId) -> MonoId {
        self.cols[self.offsets[r.index()]]
    }

    /// The columns' capacity, in entries.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.cols.capacity()
    }

    /// The entries of every row's columns.
    pub(crate) fn entries(&self) -> usize {
        self.cols.len()
    }
}

// matrix-slack.2 tuned -- a matrix's columns take a reservation once they would pass 2^18 entries, not before every matrix's first row, nor past 2^16 or 2^20 entries (log 0041-1)
/// The entries past which a matrix's columns take a reservation: 2^18,
/// 1 MiB of `u32`.
pub(crate) const RESERVE_PAST: usize = 1 << 18;

// matrix-slack.3 design -- a reservation holds 2^23 entries, 32 MiB, which glibc's malloc maps outside its heap where its free memory cannot serve it, whatever its dynamic threshold, not a smaller one it may serve from its heap (not measured)
/// The entries a reservation holds: 2^23, 32 MiB of `u32`.
pub(crate) const RESERVATION: usize = 1 << 23;

/// Makes room for `need` more columns: where `cols` would pass
/// `RESERVE_PAST` entries and holds less than `RESERVATION` of capacity,
/// reserves `RESERVATION` entries, or `cols.len() + need` where that is
/// more. Else it leaves `cols` as it is, and `Table::row_products`, which
/// reserves room for each row in a format of up to four words and pushes
/// term by term past that, grows it by doubling.
pub(crate) fn make_room(cols: &mut Vec<MonoId>, need: usize) {
    let len = cols.len();
    if cols.capacity() < RESERVATION && len + need > RESERVE_PAST {
        cols.reserve_exact(RESERVATION.max(len + need) - len);
    }
}

// matrix-stop-latency.5 design -- symbolic preprocessing spends 16 units a product, every row it multiplies, the given rows included, not ticks per reducer row alone nor 8 units a product (log 0055-2, 0056-1)
/// Units of work per product of symbolic preprocessing.
pub(crate) const PRODUCT: usize = 16;

// matrix-stop-latency.7 design -- a keyed sort stays one sort, spent after its keys and after its sort, not sorted in runs as well (log 0056-1)
// threshold-recheck.13 design -- a keyed sort spends KEY, 16, units a key once its keys are built, not none, as building a key reads every exponent word of its element; at these spends no call of run O, the tuning set's instances past 0.1 s but cyclic-9-p65521, returned past 0.1 s after its deadline (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0056-1, 0074-2)
/// Units of work per key a keyed sort builds, and per term a cofactor
/// entry gathers into its sorted order.
pub(crate) const KEY: usize = 16;

// threshold-recheck.12 design -- an element of a keyed sort, or of a sort of at most RUN elements, spends SORTED, 64, units, not one unit as an entry of a row operation spends, as a sort makes about log2 n comparisons an element; at these spends no call of run O, the tuning set's instances past 0.1 s but cyclic-9-p65521, returned past 0.1 s after its deadline (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0056-1, 0074-2)
/// Units of work per element of a keyed sort, and of a sort of at most
/// `RUN` elements by `sort_runs`.
pub(crate) const SORTED: usize = 64;

// matrix-stop-latency.6 design -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
/// Elements per run of `sort_runs`, and per chunk of the work spent `RUN`
/// elements at a time.
pub(crate) const RUN: usize = 1 << 15;

// round-memory-time.4 cited: martani-2012 §3.9, attributes it to Lachartre 2008 [1] -- the original implementation reserves a big memory chunk at the start of the program and performs all the computations inside it, which eliminates the dynamic allocations and deallocations; monagan-2007 §2.2, independently -- the merge buffers are reused in the next call rather than freed each time
// round-memory-time.6 cited: martani-2012 §3.9, attributes it to Lachartre 2008 [1] -- the original implementation reserves a big memory chunk at the start of the program and performs all the computations inside it, which eliminates the dynamic allocations and deallocations
// round-memory-time.8 design -- the buffers are an object the run owns and passes to its builds and reductions, not pools of the thread (log 0051-1)
// round-memory-time.12 design -- the work vectors a reduction builds its rows in are the run's buffers', kept from matrix to matrix, not the reducer's own for one matrix (not measured)
/// The buffers the matrices of one run take in turn: the column vector
/// the next matrix's rows fill, the dense row of the reductions, the two
/// work vectors a reduction builds a row in, and the band of the
/// reductions that take bands with its rows' work vectors. Each holds what
/// the matrix before left, or nothing.
pub(crate) struct Buffers<F: Field> {
    /// The column vector of the run's last matrix, its entries stale.
    cols: Vec<MonoId>,
    /// The dense row of the last reduction.
    dense: Vec<F::Acc>,
    /// A row's columns and coefficients while it is built.
    row_cols: Vec<MonoId>,
    row_coefs: Vec<F::Coef>,
    // row-traffic-time.10 cited: monagan-2007 §2.2 -- to further amortize the cost of memory allocation the merge buffers are reused in the next call rather than freed each time
    /// The band's dense rows of the last reduction that took a band.
    band: Vec<[F::Acc; BAND]>,
    // row-traffic-time.11 cited: monagan-2015 §2.2 -- a row to be reduced is moved into the buffer, and if the result is not zero, the buffer contents are copied to a new sparse row
    /// The band rows' columns and coefficients while they are built, a
    /// pair per band row.
    band_cols: [Vec<MonoId>; BAND],
    band_coefs: [Vec<F::Coef>; BAND],
    /// The scan of the reducers made from the buffers (`Reducer::new`).
    #[cfg(test)]
    pub(crate) scan: Scan,
}

impl<F: Field> Buffers<F> {
    /// Buffers that hold nothing.
    pub(crate) fn new() -> Buffers<F> {
        Buffers {
            cols: Vec::new(),
            dense: Vec::new(),
            row_cols: Vec::new(),
            row_coefs: Vec::new(),
            band: Vec::new(),
            band_cols: Default::default(),
            band_coefs: Default::default(),
            #[cfg(test)]
            scan: Scan::Chosen,
        }
    }

    /// The column vector for the next matrix: empty, its capacity the one
    /// of the vector kept last, or none.
    pub(crate) fn take_cols(&mut self) -> Vec<MonoId> {
        let mut cols = std::mem::take(&mut self.cols);
        cols.clear();
        cols
    }

    /// Keeps the column vector of `matrix` for the next matrix, in place of
    /// the one held; the rest of the matrix is dropped.
    pub(crate) fn keep_cols(&mut self, matrix: Matrix) {
        self.cols = matrix.into_cols();
    }

    /// Gives back every buffer: each holds nothing and no capacity.
    pub(crate) fn release(&mut self) {
        *self = Buffers::new();
    }

    /// The capacities of the column vector, the dense row, and the two
    /// work vectors, in that order.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> [usize; 4] {
        [
            self.cols.capacity(),
            self.dense.capacity(),
            self.row_cols.capacity(),
            self.row_coefs.capacity(),
        ]
    }

    /// Fills the dense row with `len` copies of `value`: a dense row no
    /// reduction leaves, which the next must not trust to be zero
    /// (round-memory-time.5).
    #[cfg(test)]
    pub(crate) fn soil(&mut self, len: usize, value: F::Acc) {
        self.dense.clear();
        self.dense.resize(len, value);
    }

    /// The capacities of the band, in elements, and of the band rows' work
    /// vectors, the columns' summed and the coefficients' summed.
    #[cfg(test)]
    pub(crate) fn band_capacities(&self) -> [usize; 3] {
        [
            self.band.capacity(),
            self.band_cols.iter().map(Vec::capacity).sum(),
            self.band_coefs.iter().map(Vec::capacity).sum(),
        ]
    }

    /// Fills the band with `len` elements of `value` in every band row: a
    /// band no reduction leaves, which the next must not trust to be zero
    /// (row-traffic-time.10).
    #[cfg(test)]
    pub(crate) fn soil_band(&mut self, len: usize, value: F::Acc) {
        self.band.clear();
        self.band.resize(len, [value; BAND]);
    }
}

// order-family.12 cited: giovini-1991 §2 -- simplifying only where it does not increase the sugar was found often equivalent to the original strategy, and sometimes worse; bigatti-2025 §3.2, Remark 58 -- symPre ends, since every term it adds is strictly lower than the one it reduces, in any term order
/// Builds the matrix of the rows `given` in round `round`: symbolic
/// preprocessing adds a reducer row for every monomial of the matrix that a
/// leading monomial of `G_min` divides, save the given rows' leading
/// monomials, whatever the sugar of the reducer. Each reducer row ticks the budget. Overflow where a product
/// passes the format. Every row spends its products, and the columns their
/// sort and mapping (`multiply`, `columns`); a stop where the budget's
/// check stops the call. The rows' columns fill `cols`, cleared first.
pub(crate) fn build<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    round: u32,
    given: Vec<RowSource>,
    cols: Vec<MonoId>,
    budget: &mut Budget,
) -> Result<Matrix, Halt> {
    let mut m = Matrix::new(given, cols);
    // The given rows' leading monomials are done from the start.
    for r in 0..m.given {
        let row = m.rows[r];
        let lead = table.mul(row.multiplier, basis.lead(row.element))?;
        if table.stamp(lead) != round {
            table.set_stamp(lead, round);
            m.monomials.push(lead);
        }
    }
    // Monomials new to the round wait in `work` for their divisor search.
    let mut work: Vec<MonoId> = Vec::new();
    for r in 0..m.given {
        let row = m.rows[r];
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
    }
    let divisors = Divisors::new(table, basis);
    while let Some(u) = work.pop() {
        // matrix-stop-latency.9 design -- the divisor search of symbolic preprocessing and the signature run's reducer search are not spent, not a spend per monomial searched (log 0056-1)
        if let Some(g) = divisors.find(table, u) {
            budget.tick()?;
            let row = RowSource {
                element: g,
                multiplier: table.div(u, basis.lead(g))?,
            };
            m.rows.push(row);
            multiply(table, basis, row, round, &mut m, &mut work, budget)?;
        }
    }

    columns(table, order, &mut m, budget)?;
    Ok(m)
}

// signature-pruning.8 cited: albrecht-2010 §3, S-Polynomials F5 -- discards the component (v, l) and relies on Symbolic Preprocessing to find a reductor; §5 -- this allows a reductor that had more reductions applied to it
/// Builds the matrix of round `round` from its reductants `given`, each
/// with its signature: symbolic preprocessing adds, for every monomial of
/// the matrix, the reductants' leading monomials among them, the reducer
/// of least signature where that signature lies below the largest
/// reductant signature. Each reducer row ticks the budget, every row spends
/// its products, and the columns their sort and mapping (`multiply`,
/// `columns`). The matrix, and each row's signature. Overflow where a
/// product passes the format, a stop where the budget's check stops the
/// call. The rows' columns fill `cols`, cleared first.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, basis, signatures, and order the rows come from, the round, the rows and the columns' buffer, and the budget, none of which belongs with another"
)]
pub(crate) fn build_sig<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    sigs: &Signatures,
    order: &O,
    round: u32,
    given: Vec<(RowSource, Sig)>,
    cols: Vec<MonoId>,
    budget: &mut Budget,
) -> Result<(Matrix, Vec<Sig>), Halt> {
    #[allow(clippy::expect_used, reason = "the only caller skips an empty given")]
    let largest = given
        .iter()
        .map(|&(_, s)| s)
        .max_by(|a, b| a.cmp(*b, table, order))
        .expect("a matrix has a reductant");
    let (rows, mut row_sigs): (Vec<RowSource>, Vec<Sig>) = given.into_iter().unzip();
    let mut m = Matrix::new(rows, cols);
    // Every monomial new to the round waits in `work` for its reducer, the
    // reductants' leading monomials among them.
    let mut work: Vec<MonoId> = Vec::new();
    for r in 0..m.given {
        let row = m.rows[r];
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
    }
    while let Some(u) = work.pop() {
        // matrix-stop-latency.9 design -- the divisor search of symbolic preprocessing and the signature run's reducer search are not spent, not a spend per monomial searched (log 0056-1)
        let Some(g) = sigs.reducer(table, u) else {
            continue;
        };
        let t = table.div(u, basis.lead(g))?;
        // signature-pruning.11 design -- symbolic preprocessing adds no reducer whose signature reaches the largest reductant signature of the matrix (not measured)
        // A signature the format cannot hold lies above the largest, which
        // it holds.
        let s = match sigs.sig(g).mul(t, table) {
            Ok(s) => s,
            Err(Limit::Overflow) => continue,
            Err(Limit::Size) => return Err(Halt::Size),
        };
        if s.cmp(largest, table, order).is_ge() {
            continue;
        }
        budget.tick()?;
        let row = RowSource {
            element: g,
            multiplier: t,
        };
        m.rows.push(row);
        row_sigs.push(s);
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
    }
    columns(table, order, &mut m, budget)?;
    Ok((m, row_sigs))
}

// trace-replay.1 cited: berthomieu-2021 §4.3, via berthomieu-2022a §3.4 (3) -- the matrices are generated from the trace's polynomials, with no need of handling critical pairs or symbolic preprocessing
/// Builds the matrix of round `round` from the rows `rows` alone: their
/// monomials as columns, and no symbolic preprocessing. Each row ticks the
/// budget and spends its products, and the columns spend their sort and
/// mapping (`multiply`, `columns`). Overflow where a product passes the
/// format, a stop where the budget's check stops the call. The rows'
/// columns fill `cols`, cleared first.
pub(crate) fn build_rows<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    round: u32,
    rows: Vec<RowSource>,
    cols: Vec<MonoId>,
    budget: &mut Budget,
) -> Result<Matrix, Halt> {
    let mut m = Matrix::new(rows, cols);
    // No divisor search takes the monomials new to the round.
    let mut work: Vec<MonoId> = Vec::new();
    for r in 0..m.given {
        budget.tick()?;
        let row = m.rows[r];
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
        work.clear();
    }
    columns(table, order, &mut m, budget)?;
    Ok(m)
}

/// Numbers the columns: the round's monomials in decreasing order. The
/// rows' monomial ids become their columns, each below the matrix's width,
/// or the call panics. The sort spends what `sort_decreasing` spends, and
/// the mapping a unit an entry after every `RUN` entries; a stop where the
/// budget's check stops the call. Size where the rows are more than a u32
/// id numbers: every row of the matrix is in, and a row's index is narrowed
/// from here on. The call report's step ends its stage `Symbolic` as the
/// numbering starts and `Encode` as it ends.
fn columns<M: Format, O: Order<M>>(
    table: &mut Table<M>,
    order: &O,
    m: &mut Matrix,
    budget: &mut Budget,
) -> Result<(), Halt> {
    narrow::id(m.rows.len())?;
    budget.mark(Stage::Symbolic);
    sort_decreasing(table, order, &mut m.monomials, budget)?;
    for (c, &u) in m.monomials.iter().enumerate() {
        table.set_column(u, MonoId::at(c));
    }
    let mut most = MonoId::at(0);
    for chunk in m.cols.chunks_mut(RUN) {
        for c in chunk.iter_mut() {
            *c = table.column(*c);
            most = most.max(*c);
        }
        budget.spend(chunk.len())?;
    }
    // reduction-time.4 design -- every column is asserted below the matrix's width once per matrix, where it is built, not trusted by construction alone; the row operations write the dense row unchecked (not measured)
    // It holds: an id joined `m.monomials` where it was new to the round, and
    // each matrix of a table is built in a round of its own, from 1 on. A run
    // starts in a new table, and a kept shape's table builds no matrix.
    assert!(
        m.cols.is_empty() || most.index() < m.monomials.len(),
        "column {most} past the matrix's {len} columns",
        len = m.monomials.len()
    );
    budget.mark(Stage::Encode);
    Ok(())
}

// matrix-sort-time.1 cited: bachmann-1998 §4, attributes it to Macaulay [4] -- a monomial is compared by its rank, one integer whose order is the monomial order
// matrix-sort-time.2 practice: decorate-sort-undecorate -- each element of a sort is keyed as the sort starts, and the keys are dropped when it returns, not a key stored in the table per monomial
/// Sorts the monomials `ids` of the table into decreasing order. In an
/// order without keys the sort is `sort_runs`'s. In a keyed order the
/// sort spends `KEY` units an element once the keys are built and `SORTED`
/// once they are sorted. A stop where the budget's check stops the call.
pub(crate) fn sort_decreasing<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    ids: &mut [MonoId],
    budget: &mut Budget,
) -> Result<(), Stop> {
    // matrix-stop-latency.6 design -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
    if !O::KEYED {
        return sort_runs(ids, |&a, &b| table.cmp(order, b, a), budget);
    }
    // matrix-stop-latency.7 design -- a keyed sort stays one sort, spent after its keys and after its sort, not sorted in runs as well (log 0056-1)
    let mut keyed: Vec<u128> = ids
        .iter()
        .map(|&u| table.key(order, u) | u128::from(u.raw()))
        .collect();
    budget.spend(ids.len() * KEY)?;
    keyed.sort_unstable_by(|a, b| b.cmp(a));
    ties(&mut keyed, |a, b| {
        table.cmp(order, MonoId::from_raw(b), MonoId::from_raw(a))
    });
    budget.spend(ids.len() * SORTED)?;
    for (u, &k) in ids.iter_mut().zip(&keyed) {
        *u = MonoId::from_raw(payload(k));
    }
    Ok(())
}

/// The rows of a matrix whose signatures are `sigs`, in increasing order
/// of signature, the rows of one signature in increasing order. The sort
/// spends as `sort_decreasing`'s does, and a stop where the budget's check
/// stops the call.
pub(crate) fn by_signature<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    sigs: &[Sig],
    budget: &mut Budget,
) -> Result<Vec<RowId>, Stop> {
    let cmp = |a: RowId, b: RowId| {
        sigs[a.index()]
            .cmp(sigs[b.index()], table, order)
            .then(a.cmp(&b))
    };
    // matrix-stop-latency.6 design -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
    if !O::KEYED {
        let mut rows: Vec<RowId> = (0..sigs.len()).map(RowId::at).collect();
        sort_runs(&mut rows, |&a, &b| cmp(a, b), budget)?;
        return Ok(rows);
    }
    // matrix-stop-latency.7 design -- a keyed sort stays one sort, spent after its keys and after its sort, not sorted in runs as well (log 0056-1)
    let mut keyed: Vec<u128> = sigs
        .iter()
        .enumerate()
        .map(|(r, s)| table.key(order, s.monomial) | r as u128)
        .collect();
    budget.spend(sigs.len() * KEY)?;
    keyed.sort_unstable();
    let row = |k: u32| RowId::at(k as usize);
    ties(&mut keyed, |a, b| cmp(row(a), row(b)));
    budget.spend(sigs.len() * SORTED)?;
    Ok(keyed.iter().map(|&k| row(payload(k))).collect())
}

// matrix-stop-latency.6 design -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
/// Sorts `v` by `cmp`, whose elements are distinct under it, into the
/// order `sort_unstable_by` returns. At most `RUN` elements are sorted at
/// once and spend `SORTED` units an element. More are sorted in runs of
/// `RUN` elements, a full check after each, and the runs are merged
/// pairwise into a buffer of `v`'s length, a full check each time an
/// element taken by comparison brings the buffer to a multiple of `RUN`,
/// the rest of a run copied without one. A stop where the budget's check
/// stops the call, `v` then in no order.
pub(crate) fn sort_runs<T: Copy>(
    v: &mut [T],
    mut cmp: impl FnMut(&T, &T) -> Ordering,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let n = v.len();
    if n <= RUN {
        v.sort_unstable_by(&mut cmp);
        return budget.spend(n * SORTED);
    }
    for run in v.chunks_mut(RUN) {
        run.sort_unstable_by(&mut cmp);
        budget.check(|| {})?;
    }
    let mut buf: Vec<T> = Vec::with_capacity(n);
    let mut width = RUN;
    while width < n {
        buf.clear();
        let mut start = 0;
        while start < n {
            let mid = (start + width).min(n);
            let end = (start + 2 * width).min(n);
            let (mut i, mut j) = (start, mid);
            while i < mid && j < end {
                // The left run's element first on a tie, as no two compare
                // equal.
                if cmp(&v[j], &v[i]).is_lt() {
                    buf.push(v[j]);
                    j += 1;
                } else {
                    buf.push(v[i]);
                    i += 1;
                }
                if buf.len().is_multiple_of(RUN) {
                    budget.check(|| {})?;
                }
            }
            buf.extend_from_slice(&v[i..mid]);
            buf.extend_from_slice(&v[j..end]);
            start = end;
        }
        v.copy_from_slice(&buf);
        width *= 2;
    }
    Ok(())
}

// matrix-sort-time.5 design -- an element of a keyed sort is one u128, the key's first 96 bits above a 32-bit payload, the id or the row, not the key's first 128 bits beside the payload in 24 bytes (log 0032-1)
// matrix-sort-time.6 design -- elements whose first 96 bits agree are sorted by the order's comparison, not by the key's next 96 bits built for their run (log 0032-1)
/// Sorts each run of `keyed` whose elements agree in their top 96 bits by
/// `cmp` on their payloads, the low 32 bits.
fn ties(keyed: &mut [u128], mut cmp: impl FnMut(u32, u32) -> Ordering) {
    for run in keyed.chunk_by_mut(|&a, &b| same_key(a, b)) {
        if run.len() > 1 {
            run.sort_unstable_by(|&a, &b| cmp(payload(a), payload(b)));
        }
    }
}

/// The payload of an element of a keyed sort, its low 32 bits.
pub(crate) fn payload(k: u128) -> u32 {
    (k & u128::from(u32::MAX)).narrow()
}

const _: () = assert!(PAYLOAD_BITS == u32::BITS, "a payload is a u32");

/// Whether two elements of a keyed sort agree in their keys above their
/// payloads.
#[inline]
pub(crate) fn same_key(a: u128, b: u128) -> bool {
    a >> PAYLOAD_BITS == b >> PAYLOAD_BITS
}

/// Appends the monomials of t·g to the matrix's columns, as monomial ids
/// until the columns are known, and puts those new to the round on `work`.
/// Then spends `PRODUCT` units a product. Overflow where a product passes
/// the format, a stop where the budget's check stops the call.
fn multiply<F: Field, M: Format, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    row: RowSource,
    round: u32,
    m: &mut Matrix,
    work: &mut Vec<MonoId>,
    budget: &mut Budget,
) -> Result<(), Halt> {
    let start = work.len();
    let monomials = &basis.element(row.element).monomials;
    // matrix-slack.1 cited: heisinger-2025 §5 -- malloc returns uninitialized memory, so 4 GB of sequential heap is reserved in the process space when a store starts, and wired into the process page by page as entries are appended at its tip
    // matrix-slack.4 design -- the reservation is the capacity of the columns' Vec, reserved exactly and mapped by the allocator, not a buffer of the crate's own over mmap, mremap, and munmap (not measured)
    // matrix-slack.7 design -- the reservation is made before the row whose products would pass RESERVE_PAST, not before a matrix's first row where the run's matrix before it passed RESERVE_PAST (log 0041-1)
    make_room(&mut m.cols, monomials.len());
    // product-lookup-time.2 design -- the lookup runs with no check of its ids, the construction of the row's multiplier and of the element's monomials as its guarantee, not one assert on both ids per lookup (log 0031-1)
    // product-lookup-time.3 design -- nor are a row's ids checked once as a batch, by their maximum, before its lookups (log 0031-1)
    // SAFETY: the run's table returned the multiplier, by `intern` (the
    // inter-reduction's unit monomial, a replayed row's multiplier from its
    // exponents) or by `div` (a pair's, a reducer's, a reductant's), and the
    // element's monomials, by interning the input's terms or as the
    // monomials of the matrix that made the element. A run holds one table
    // and one basis.
    unsafe { table.row_products(row.multiplier, monomials, round, &mut m.cols, work) }?;
    m.monomials.extend_from_slice(&work[start..]);
    m.offsets.push(m.cols.len());
    // matrix-stop-latency.5 design -- symbolic preprocessing spends 16 units a product, every row it multiplies, the given rows included, not ticks per reducer row alone nor 8 units a product (log 0055-2, 0056-1)
    budget.spend(monomials.len() * PRODUCT)?;
    Ok(())
}

// mask-scan-time.1 design -- G_min stands in increasing order of its elements' terms, ties in the order made, so that the first divisor the search finds is the rule's and the search stops there, not every candidate tested in the order made and the one of fewest terms kept (log 0073-1)
// mask-scan-time.2 design -- the order is made at each round by a sort of G_min, not kept in the basis as elements join and leave G_min (not measured)
/// `G_min`'s leading monomials with their masks, in increasing order of
/// their elements' terms, ties in the order made, for the divisor search
/// of symbolic preprocessing.
pub(crate) struct Divisors {
    element: Vec<ElementId>,
    lead: Vec<MonoId>,
    mask: Vec<u32>,
}

impl Divisors {
    // mask-scan-time.6 design -- the sort spends nothing from the budget, not a spend per element sorted (not measured)
    pub(crate) fn new<F: Field, M: Format, D: Module>(
        table: &Table<M>,
        basis: &Basis<F, D>,
    ) -> Divisors {
        let mut element = basis.min().to_vec();
        // The index is the order made, so the unstable sort leaves ties in
        // that order and allocates nothing.
        element.sort_unstable_by_key(|&i| (basis.length(i), i));
        Divisors {
            lead: element.iter().map(|&i| basis.lead(i)).collect(),
            mask: element.iter().map(|&i| table.mask(basis.lead(i))).collect(),
            element,
        }
    }

    // f4-gfp.5 cited: berthomieu-2021 §4.1 (4) and §4.2 -- the sparsest possible rows are the pivot rows, and the sparsest row for each pivot goes into the upper pivot part
    // matrix-build-time.8 design -- G_min's masks are tested eight at a time before the divisibility test runs on the candidates in order, not one mask test per element (log 0037-1)
    // table-read-time.9 design -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    // mask-scan-time.3 design -- a search that finds no divisor leaves no record, not per monomial the basis's length at that search, so that its next search scans only the elements made since (log 0043-1, 0073-1)
    // mask-scan-time.4 design -- the search tests G_min's masks, then the candidates' divisibility, not per variable and exponent a bit set of the elements whose exponent is at most it, ANDed over the variables where the monomial's exponent lies below the largest (log 0073-1)
    /// The element of `G_min` with the fewest terms whose leading monomial
    /// divides u, ties to the element made first; None where none does.
    pub(crate) fn find<M: Format>(&self, table: &Table<M>, u: MonoId) -> Option<ElementId> {
        let (format, words) = (table.format(), table.monomial(u));
        find_passing(&self.mask, table.mask(u), |k| {
            format.divides(table.monomial(self.lead[k]), words)
        })
        .map(|k| self.element[k])
    }
}

/// A row the reduction made: the row of the matrix it was made from, its
/// columns, increasing, its coefficients, and the index of its last keeping
/// by the cofactor tracking.
pub(crate) struct Reduced<F: Field> {
    pub(crate) row: RowId,
    pub(crate) cols: Vec<MonoId>,
    pub(crate) coefs: Vec<F::Coef>,
    pub(crate) kept: KeepId,
}

impl<F: Field> Reduced<F> {
    // round-memory-time.3 design -- the polynomial made from a row takes the row's column vector, each column replaced by its monomial in place, not a new vector of the monomials collected from the columns (log 0051-1)
    /// The polynomial of the row, its columns mapped back to monomials in
    /// the row's column vector, which the polynomial takes.
    pub(crate) fn poly<D: Module>(self, matrix: &Matrix) -> Poly<F, D> {
        let mut monomials = self.cols;
        for c in &mut monomials {
            *c = matrix.monomials[c.index()];
        }
        Poly {
            monomials,
            coefs: self.coefs,
            position: D::Position::default(),
        }
    }

    /// As `poly`, the row's vectors taken and left empty.
    pub(crate) fn take_poly<D: Module>(&mut self, matrix: &Matrix) -> Poly<F, D> {
        Reduced {
            cols: mem::take(&mut self.cols),
            coefs: mem::take(&mut self.coefs),
            ..*self
        }
        .poly(matrix)
    }
}

/// What the reduction of a round's matrix made of the rows it ran on.
pub(crate) struct Reduction<F: Field> {
    /// The new pivots, monic, in the order made, each kept by the cofactor
    /// tracking in that order, and kept again where it was back-substituted.
    /// A row made from a combination names its block's first row.
    pub(crate) made: Vec<Reduced<F>>,
    /// The rows that reduced to zero, in the order reduced, and those a test
    /// ruled zero: in a signature round in signature order, in a learn's
    /// plain round in the order of the rows to reduce. Empty where the plain
    /// run combined rows.
    pub(crate) zero: Vec<RowId>,
    // trace-replay.11 design -- the reduction hands on the pivot of every column at its end, not a learner that works the pivots out again from the matrix (not measured)
    /// The pivot of every column when the reduction ended.
    pub(crate) pivot: Vec<Pivot>,
    // trace-replay.17 design -- the signature reduction hands on the order in which it took the rows, not the learner sorting the rows by signature again (not measured)
    /// The rows in the order the reduction took them, empty where it set
    /// every pivot row before it reduced a row.
    pub(crate) taken: Vec<RowId>,
    // gfp-unsolved.9 design -- Stats counts the reductions run, rows or combinations, not the rows of the blocks combined (log 0014-1)
    /// The reductions run: rows, or combinations of rows.
    pub(crate) reductions: u64,
    /// Those of them that reduced to zero.
    pub(crate) zeros: u64,
}

/// What `Reducer::test_groups` did: how many rows from the start of the
/// rows it tested its groups ruled zero, whether a test has failed, and the
/// combinations it reduced and those of them that reduced to zero.
struct Tested {
    ruled: usize,
    failed: bool,
    reductions: u64,
    zeros: u64,
}

// gfp-unsolved.7 design -- a combination's coefficients are uniform in [1, p), from a Xoshiro256PlusPlus seeded by the call's generator after its other draws, not a generator of its own seeded by a constant or coefficients in [0, p) (not measured)
/// The random combinations of a reduction (gfp-unsolved.2): the
/// coefficients' generator.
pub(crate) struct Combine {
    rng: Xoshiro256PlusPlus,
}

impl Combine {
    pub(crate) fn new(seed: u64) -> Combine {
        Combine {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    // gfp-unsolved.18 practice: Lemire's multiply-shift range reduction -- a coefficient in [1, p) is the high word of a 64-bit draw times p − 1, plus 1, not the draw reduced mod p − 1 nor a draw rejected and drawn again
    /// A coefficient of [1, p), for 2 ≤ p < 2^64: each value has a
    /// probability within 2^−64 of 1/(p − 1).
    fn coef(&mut self, p: u64) -> u64 {
        debug_assert!(p >= 2, "p ≥ 2");
        1 + ((u128::from(self.rng.next_u64()) * u128::from(p - 1)) >> 64) as u64
    }
}

// gfp-unsolved.3 tuned -- a block of BLOCK rows is 64 consecutive rows of those to reduce, by leading column, not 16, 32, 128, or 256 rows, nor the rows in another order (log 0014-1)
/// The rows of a matrix's first block of combinations (gfp-unsolved.2),
/// and of every block after one that made a pivot.
pub(crate) const BLOCK: usize = 64;

// plain-zero-time.6 tuned -- a grown block that has made more than 16 pivots stops before its next band or combination, its rows taken again from its first, in a block of BLOCK rows, not a grown block reduced to its end, nor a stop past 4, 8, or 32 pivots (log 0077-1)
/// The pivots past which a block of more than `BLOCK` rows stops before
/// its next band or combination (`reduce`).
pub(crate) const GROWN_PIVOTS: usize = 16;

// gfp-unsolved.4 cited: berthomieu-2021 §4.2 -- one can increase the probability of correctness by doing more than one reduction to zero before the block is finished
/// Combinations that reduce to zero before a block is done.
pub(crate) const ZEROS: usize = 2;

// signature-zero-time.3 tuned -- a test starts once two rows of a run reduced on their own have reduced to zero one after another, not after each row that reduces to zero (log 0049-1)
/// Rows of a signature round's run reduced to zero one after another before
/// a test of the run's rest (`reduce_sig`), and of a learn's plain round
/// before a test of the rows after them (`reduce`).
pub(crate) const STREAK: usize = 2;

// signature-zero-time.5 design -- after a failed test in the run, a test takes a block of 8 rows, then twice the last block's rows after each block that passes, not the whole rest at every test (log 0049-1)
/// The rows of a signature round's first block after a failed test in its
/// run (`reduce_sig`), and of a learn's plain round's (`reduce`).
pub(crate) const FIRST: usize = 8;

/// The rows per combination of a test below which a group takes no test
/// (signature-zero-time.9).
pub(crate) const GROUP_ROWS: usize = 2;

/// The bound on a test's false zeros: s combinations reduced to zero
/// rule a group zero where (p − 1)^s reaches it (signature-zero-time.8).
pub(crate) const ZERO_BOUND: u128 = 1 << 31;

// signature-zero-time.8 cited: demin-2026b §2.5, Proposition 3 -- the number s of reductions to zero that ends a block is computed from the bound on the probability of an incorrect result and the field's size; monagan-2017 §2 -- if zero is obtained k times in a row the rows are discarded with a probability of error of 1/p^k
/// The combinations of a test, in a signature round or a learn's plain
/// round, s, the least integer with
/// (p − 1)^s ≥ 2^31; None for p = 2 and p = 3, which take no test.
pub(crate) fn zero_combinations(p: u64) -> Option<usize> {
    if p <= 3 {
        return None;
    }
    let q = u128::from(p - 1);
    let (mut s, mut power) = (1, q);
    // power < 2^31 and q < 2^64 keep the product below 2^95.
    while power < ZERO_BOUND {
        s += 1;
        power *= q;
    }
    Some(s)
}

/// Where the pivot of a column is: a row of the matrix, or a row the
/// reduction made.
#[derive(Clone, Copy)]
pub(crate) enum Pivot {
    None,
    Row(RowId),
    New(MadeId),
}

impl Pivot {
    /// Whether the column has no pivot.
    #[inline]
    fn is_none(self) -> bool {
        match self {
            Pivot::None => true,
            Pivot::Row(_) | Pivot::New(_) => false,
        }
    }

    /// Whether the pivot is a row the reduction made.
    #[inline]
    fn is_new(self) -> bool {
        match self {
            Pivot::New(_) => true,
            Pivot::None | Pivot::Row(_) => false,
        }
    }
}

// row-traffic-time.7 tuned -- a signature round takes bands only where its matrix's rows hold 64 entries or more in the mean, not in every matrix, nor from 32 or 128 entries a row (log 0044-1)
/// The fewest entries a matrix's rows hold in the mean for its signature
/// round to reduce its reductants in bands, and its plain run its
/// combinations: 64.
pub(crate) const BAND_ENTRIES: usize = 64;

/// Whether `rows` rows of `entries` entries in all hold `BAND_ENTRIES`
/// entries or more in the mean.
pub(crate) fn long_rows(entries: usize, rows: usize) -> bool {
    entries >= BAND_ENTRIES * rows
}

// gfp-unsolved.5 design -- the rows are combined only where p² ≥ 2^31, and a smaller p reduces each row on its own, not combinations for every p with more zeros for a smaller p (not measured)
// call-report.11 design -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
/// Whether the plain run's reduction combines rows at p, where its hooks
/// allow it (`f4::combines`): p² ≥ 2^31, p ≥ 46341, which computes no
/// square past 2^64.
pub(crate) fn combines_rows(p: u64) -> bool {
    p >= COMBINES_FROM
}

/// The least p with p² ≥ `ZERO_BOUND` (gfp-unsolved.5): 46340² lies below
/// 2^31, 46341² above.
const COMBINES_FROM: u64 = 46341;

const _: () = assert!(
    (COMBINES_FROM as u128 - 1) * (COMBINES_FROM as u128 - 1) < ZERO_BOUND
        && COMBINES_FROM as u128 * COMBINES_FROM as u128 >= ZERO_BOUND,
    "COMBINES_FROM is the least p with p² ≥ ZERO_BOUND"
);

// signature-zero-time.10 design -- the tests run where no cofactors are tracked, neither the trace nor the learner listens, and the field has one lane, not in a learn run as well (log 0049-1)
// call-report.11 design -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
/// Whether a signature round rules rows zero by tests over `F` at p, where
/// its hooks allow it (`f4::combines`): the field has one lane, and p
/// takes a test (`zero_combinations`).
pub(crate) fn tests_rows<F: Field>(p: u64) -> bool {
    F::LANES == 1 && zero_combinations(p).is_some()
}

// row-traffic-time.7 tuned -- a signature round takes bands only where its matrix's rows hold 64 entries or more in the mean, not in every matrix, nor from 32 or 128 entries a row (log 0044-1)
/// Whether a signature round reduces the reductants of `matrix` in bands,
/// and a plain run that combines rows its combinations (`reduce`): the
/// field has them, `C` tracks no cofactors, and the matrix's rows are
/// long (`long_rows`).
pub(crate) fn takes_bands<F: Field, M: Format, C: Cofactors<F, M>>(matrix: &Matrix) -> bool {
    // row-traffic-time.9 design -- a round that tracks cofactors reduces each reductant alone, not bands with their cofactors tracked by a start, a subtraction, and a keep per band row (not measured)
    F::BANDS && !C::TRACKS && long_rows(matrix.entries(), matrix.rows.len())
}

// row-traffic-time.1 cited: martani-2012 §3.5.1, via boyer-2016 §2 and §4.2 -- the multiline row reduction reduces two rows at the same time, each entry of the reducer read once for both
/// The end of the band that starts at rows[k], a reductant, below
/// `given`: past the run of reductants from k, at most BAND rows. k + 1
/// where the run holds one row.
pub(crate) fn band_end(rows: &[RowId], given: usize, k: usize) -> usize {
    debug_assert!(
        k < rows.len() && rows[k].index() < given,
        "rows[k] is a reductant below given"
    );
    let mut j = k + 1;
    while j < rows.len() && j - k < BAND && rows[j].index() < given {
        j += 1;
    }
    j
}

// row-scan-time.10 design -- the scan is a parameter of the test profile carried by the buffers, Dense or Marked for the test code and the counts' choice otherwise, not the test code timing the call or counting the rows each scan took (not measured)
/// How the reducers of a test's buffers scan their dense row: as the
/// counts choose, row by row (`Reducer::count`), or always one way.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scan {
    /// The scan the counts choose, as in every call.
    Chosen,
    /// Every column from the row's start.
    Dense,
    /// The marked columns alone.
    Marked,
}

#[cfg(test)]
impl Scan {
    /// Whether a row takes the marked scan where the counts choose
    /// `chosen`.
    fn marks(self, chosen: bool) -> bool {
        match self {
            Scan::Chosen => chosen,
            Scan::Dense => false,
            Scan::Marked => true,
        }
    }
}

// row-scan-time.2 tuned -- a row takes the marked scan where the columns the dense scans of the rows before it would pass exceed twice the entries those rows wrote, once four rows are counted, not the marked scan for the rest of the matrix once the counts first say so, a ratio of 8 or 1, nor the scan chosen once per matrix from its shape (log 0072-1)
/// A row takes the marked scan where the columns the rows' dense scans
/// would pass exceed `MARKED_RATIO` times the entries the rows wrote: 2.
pub(crate) const MARKED_RATIO: u64 = 2;

/// The rows a reducer counts before a row takes the marked scan: 4.
pub(crate) const MARKED_ROWS: u64 = 4;

// row-scan-time.3 design -- both counts are halved every 64 rows, so that they weigh the recent rows, not counts over every row of the matrix (log 0072-1)
/// Both counts are halved after every `MARKED_DECAY` rows: 64.
pub(crate) const MARKED_DECAY: u64 = 64;

// row-scan-time.6 design -- the marks are one level of words, read word by word from the word of the row's start, not a second level of a bit per word of marks (log 0072-1)
/// The columns a word of the marks holds.
const MARK_BITS: usize = u64::BITS as usize;

/// Sets the mark of every column of `cols`: bit c mod `MARK_BITS` of word
/// c / `MARK_BITS`. The writes have no bounds check outside the test
/// profile, whose debug assertions check them.
#[inline(always)]
fn mark(marks: &mut [u64], cols: &[MonoId]) {
    for &c in cols {
        let c = c.index();
        debug_assert!(
            c / MARK_BITS < marks.len(),
            "column {c} past the marks' {len} words",
            len = marks.len()
        );
        // SAFETY: a matrix's columns lie below its width, and the marks
        // hold a word per `MARK_BITS` columns of it (`Reducer::take_marked`).
        unsafe { *marks.get_unchecked_mut(c / MARK_BITS) |= 1 << (c % MARK_BITS) };
    }
}

// row-scan-time.1 design -- the marked scan visits the columns whose mark a load, an addition, or a row operation of the row set, in increasing column order, each mark cleared as read, not every column from the row's start, the row's reach in the graph of the pivot rows found before the numerical work, a heap of the columns written, nor the scan ended at the last column a row operation reached (log 0072-1)
/// The lowest marked column from word `*w` on, its mark cleared and `*w`
/// its word; None, `*w` the marks' length, where no mark is set from word
/// `*w` on. The word is read again at each call, so that a mark set past
/// the column returned in the same word is found.
#[inline(always)]
fn next_marked(marks: &mut [u64], w: &mut usize) -> Option<usize> {
    while let Some(word) = marks.get_mut(*w) {
        if *word != 0 {
            let bit = word.trailing_zeros() as usize;
            *word &= *word - 1;
            return Some(*w * MARK_BITS + bit);
        }
        *w += 1;
    }
    None
}

/// The reduction of a matrix's rows by its pivots. The dense row, the
/// work vectors, the band, and the band rows' work vectors come from the
/// buffers the caller passes, the run's or the inter-reduction's own, and
/// go back to them where the reduction returns its result.
pub(crate) struct Reducer<'a, F: Field, D: Module> {
    field: &'a F,
    basis: &'a Basis<F, D>,
    matrix: &'a Matrix,
    // round-memory-time.9 design -- the pivot array stays one per matrix, handed on with the reduction, not kept from round to round as the dense row is (log 0051-1)
    pub(crate) pivot: Vec<Pivot>,
    /// The rows the reduction made, by their index in `Pivot::New`.
    pub(crate) made: Vec<Reduced<F>>,
    /// How many rows the cofactor tracking has kept, by `keep` and the
    /// back-substitution; `inter_reduce` numbers its rows itself.
    kept: usize,
    /// The dense row, an entry per column of the matrix, zero between two
    /// rows.
    dense: Vec<F::Acc>,
    /// The work vectors a row is built in: `begin` hands them to the row
    /// and `end` takes them back, holding the row's entries, stale until
    /// the next `begin` clears them.
    row_cols: Vec<MonoId>,
    row_coefs: Vec<F::Coef>,
    // row-traffic-time.2 design -- a band's dense rows are one array whose element holds the band's entries of one column side by side, not a dense row per band row; the nearest, martani-2012 §3.5's multiline, keeps the entries of all its rows at a column next to each other in the reducer rows, and reduces the dense rows `temp1` and `temp2` apart (log 0034-1, 0044-1)
    /// The band's dense rows, interleaved by column: empty from `new`
    /// until the reduction's first band (`widen_band`), then an element per
    /// column of the matrix, BAND rows each, zero between two bands.
    pub(crate) band: Vec<[F::Acc; BAND]>,
    /// The work vectors band row k is built in: `begin_band` hands them
    /// to band row k's result and `end_band` takes them back, holding the
    /// row's entries, stale until the next band clears them.
    band_cols: [Vec<MonoId>; BAND],
    band_coefs: [Vec<F::Coef>; BAND],
    // row-scan-time.8 design -- the marks are the reducer's own, allocated at its first marked row, not marks taken from the run's buffers and kept from round to round, which a stopped reduction leaves set (not measured)
    /// A mark per column, set where a write of the marked scan may have
    /// left an entry: empty until the reducer's first marked row, then a
    /// word per `MARK_BITS` columns of the matrix, every mark clear between
    /// two rows.
    marks: Vec<u64>,
    /// Whether the next row takes the marked scan.
    marked: bool,
    /// The columns the counted rows' dense scans would pass, from each
    /// row's start to the matrix's width, and the entries their loads wrote
    /// and the budget counted as subtracted and added, a band's with the
    /// next row counted, both halved every `MARKED_DECAY` rows (`count`).
    passed: u64,
    written: u64,
    /// The rows counted.
    rows: u64,
    /// The budget's entries subtracted and added at the last count.
    counted: u64,
    /// The scan the buffers asked for.
    #[cfg(test)]
    scan: Scan,
}

impl<'a, F: Field, D: Module> Reducer<'a, F, D> {
    /// Takes the dense row, the work vectors, the band, and the band rows'
    /// work vectors out of `bufs`: the dense row cleared and resized to the
    /// matrix's width, every entry zero, and the others cleared, their
    /// capacity kept. The counts start at none, the budget's entries
    /// subtracted and added as `budget` holds them: the reducer's rows run
    /// under it. The first row takes the dense scan, in the test profile the
    /// marked scan where `bufs` asks for it.
    pub(crate) fn new(
        field: &'a F,
        basis: &'a Basis<F, D>,
        matrix: &'a Matrix,
        bufs: &mut Buffers<F>,
        budget: &Budget,
    ) -> Reducer<'a, F, D> {
        // round-memory-time.5 design -- a reduction takes the dense row zeroed whole, every entry up to the matrix's width written zero, not the dense row the reduction before left trusted to be zero and only the entries past its length written (log 0051-1)
        let mut dense = std::mem::take(&mut bufs.dense);
        dense.clear();
        dense.resize(matrix.columns(), F::Acc::default());
        let mut row_cols = std::mem::take(&mut bufs.row_cols);
        let mut row_coefs = std::mem::take(&mut bufs.row_coefs);
        row_cols.clear();
        row_coefs.clear();
        let mut band = std::mem::take(&mut bufs.band);
        band.clear();
        let mut band_cols = std::mem::take(&mut bufs.band_cols);
        let mut band_coefs = std::mem::take(&mut bufs.band_coefs);
        band_cols.iter_mut().for_each(Vec::clear);
        band_coefs.iter_mut().for_each(Vec::clear);
        let (subtracted, added) = budget.counts();
        let red = Reducer {
            field,
            basis,
            matrix,
            pivot: vec![Pivot::None; matrix.columns()],
            made: Vec::new(),
            kept: 0,
            dense,
            row_cols,
            row_coefs,
            band,
            band_cols,
            band_coefs,
            marks: Vec::new(),
            marked: false,
            passed: 0,
            written: 0,
            rows: 0,
            counted: subtracted + added,
            #[cfg(test)]
            scan: bufs.scan,
        };
        #[cfg(test)]
        let red = {
            let mut red = red;
            red.take_marked(red.scan.marks(false));
            red
        };
        red
    }

    // row-scan-time.8 design -- the marks are the reducer's own, allocated at its first marked row, not marks taken from the run's buffers and kept from round to round, which a stopped reduction leaves set (not measured)
    /// Sets whether the next row takes the marked scan, and where it does
    /// and the marks are empty, makes them: a word per `MARK_BITS` columns of
    /// the matrix, every mark clear.
    fn take_marked(&mut self, marked: bool) {
        if marked && self.marks.is_empty() {
            self.marks = vec![0; self.dense.len().div_ceil(MARK_BITS)];
        }
        self.marked = marked;
    }

    // row-scan-time.2 tuned -- a row takes the marked scan where the columns the dense scans of the rows before it would pass exceed twice the entries those rows wrote, once four rows are counted, not the marked scan for the rest of the matrix once the counts first say so, a ratio of 8 or 1, nor the scan chosen once per matrix from its shape (log 0072-1)
    // row-scan-time.3 design -- both counts are halved every 64 rows, so that they weigh the recent rows, not counts over every row of the matrix (log 0072-1)
    // row-scan-time.4 design -- the entries written are the budget's counts of entries subtracted and added, read once a row, beside the load's length, not a count kept in the reducer at each row operation of the scan (log 0072-1)
    /// Counts the row just reduced from `from`, in either scan: the columns
    /// its dense scan would pass, and the entries the budget counted as
    /// subtracted and added since the last count, the row's own and those of
    /// the bands added or swept since, beside the load's length `reduce`
    /// counted. Every `MARKED_DECAY` rows both
    /// counts are halved. Then chooses the next row's scan: marked where
    /// `MARKED_ROWS` rows or more are counted and the columns passed exceed
    /// `MARKED_RATIO` times the entries written. In the test profile the
    /// buffers' scan may fix the choice (`Scan::marks`).
    fn count(&mut self, from: MonoId, budget: &Budget) {
        debug_assert!(
            from.index() <= self.dense.len(),
            "a start of {from} past the matrix's {len} columns",
            len = self.dense.len()
        );
        self.passed += (self.dense.len() - from.index()) as u64;
        let (subtracted, added) = budget.counts();
        let now = subtracted + added;
        debug_assert!(
            now >= self.counted,
            "the budget's {now} entries below the {counted} counted: the reducer runs under \
             the budget it was made with",
            counted = self.counted
        );
        self.written += now - self.counted;
        self.counted = now;
        self.rows += 1;
        if self.rows.is_multiple_of(MARKED_DECAY) {
            self.passed /= 2;
            self.written /= 2;
        }
        let chosen =
            self.rows >= MARKED_ROWS && self.passed > MARKED_RATIO.saturating_mul(self.written);
        #[cfg(test)]
        let chosen = self.scan.marks(chosen);
        self.take_marked(chosen);
    }

    /// Gives the dense row, the work vectors, the band, and the band rows'
    /// work vectors back to `bufs`.
    pub(crate) fn give_back(&mut self, bufs: &mut Buffers<F>) {
        bufs.dense = std::mem::take(&mut self.dense);
        bufs.row_cols = std::mem::take(&mut self.row_cols);
        bufs.row_coefs = std::mem::take(&mut self.row_coefs);
        bufs.band = std::mem::take(&mut self.band);
        bufs.band_cols = std::mem::take(&mut self.band_cols);
        bufs.band_coefs = std::mem::take(&mut self.band_coefs);
    }

    // round-memory-time.1 cited: monagan-2007 §2.2 -- the merge buffers grow dynamically, and to further amortize the cost of memory allocation they are reused in the next call rather than freed each time
    /// A row made from row `r` of the matrix, its vectors the work vectors,
    /// empty.
    fn begin(&mut self, r: RowId) -> Reduced<F> {
        let mut cols = std::mem::take(&mut self.row_cols);
        let mut coefs = std::mem::take(&mut self.row_coefs);
        cols.clear();
        coefs.clear();
        Reduced {
            row: r,
            cols,
            coefs,
            kept: KeepId::NONE,
        }
    }

    // round-memory-time.2 cited: monagan-2015 §2.2 -- if the result is not zero, the buffer contents are copied to a new sparse row
    /// The row `out` with its entries copied into vectors of its length, and
    /// the work vectors back in the reducer. A row of no entry allocates
    /// nothing.
    fn end(&mut self, mut out: Reduced<F>) -> Reduced<F> {
        let cols = out.cols.as_slice().to_vec();
        let coefs = out.coefs.as_slice().to_vec();
        self.row_cols = std::mem::replace(&mut out.cols, cols);
        self.row_coefs = std::mem::replace(&mut out.coefs, coefs);
        out
    }

    // table-read-time.6 design -- a matrix's row reads its offsets and slices its columns, and the reduction reads a row's source, with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// Row r's coefficients, its element's. The read of row r's source has
    /// no bounds check outside the test profile, whose debug assertions
    /// check it.
    fn coefs(&self, r: RowId) -> &'a [F::Coef] {
        let rows = &self.matrix.rows;
        debug_assert!(
            r.index() < rows.len(),
            "row {r} past the matrix's {len}",
            len = rows.len()
        );
        // SAFETY: a row index lies below the rows.
        &self
            .basis
            .element(unsafe { rows.get_unchecked(r.index()) }.element)
            .coefs
    }

    // f4-gfp.7 cited: monagan-2015 §2.2 -- a row is cancelled wherever a pivot is known, a nonzero result made monic and inserted as a pivot; it does not pay to perform only top reductions
    /// Reduces row r by the pivots in every column from `from` on: the
    /// entries left, in increasing column order, not made monic. The columns
    /// before `from` must hold no entry of the row but its leading one,
    /// which is kept as it is. The cofactor tracking hears of every
    /// subtraction, and the budget of every row operation (`cancel`). The
    /// row is built in the work vectors and copied at its length (`end`). A
    /// stop where the budget's check stops the call.
    pub(crate) fn reduce<M: Format, C: Cofactors<F, M>>(
        &mut self,
        r: RowId,
        from: MonoId,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<Reduced<F>, Stop> {
        let f = self.field;
        let cols = self.matrix.row(r);
        let coefs = self.coefs(r);
        // SAFETY: a matrix's columns lie below its width, the dense row's
        // length (`columns`), and row r has a coefficient per column.
        unsafe { f.load(&mut self.dense, cols, coefs) };
        // row-scan-time.4 design -- the entries written are the budget's counts of entries subtracted and added, read once a row, beside the load's length, not a count kept in the reducer at each row operation of the scan (log 0072-1)
        self.written += cols.len() as u64;
        let lead = cols[0];
        if self.marked {
            // The lead before `from` is taken outside the scan.
            mark(&mut self.marks, if lead < from { &cols[1..] } else { cols });
        }
        cof.start(r);
        let mut out = self.begin(r);
        if lead < from {
            out.cols.push(lead);
            out.coefs.push(f.take(&mut self.dense[lead.index()]));
        }
        self.cancel(from, &mut out, cof, budget)?;
        Ok(self.end(out))
    }

    // table-read-time.7 design -- the scan reads its dense row, its pivot array, and its made rows with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    // table-read-time.8 design -- the scan reads the dense row at its column unchecked, not the dense row padded to a power of two and indexed by the column masked with its length less one (log 0030-1)
    /// The entry at column c of the dense row, set to zero and cancelled by
    /// the pivot of c: v times the pivot row's tail subtracted, which clears
    /// the column as every pivot is monic, and where `MARKED` holds the
    /// tail's columns marked. The entry's value where it is nonzero and c has
    /// no pivot, None else. The cofactor tracking hears of the subtraction,
    /// and the budget is spent the subtracted row's length, counted as
    /// subtracted (`Budget::spend_subtracted`). The reads have no bounds
    /// check outside the test profile, whose debug assertions check c
    /// against the dense row's length, the pivot array's against it, and
    /// each made row's index. A stop where the budget's check stops the
    /// call.
    #[inline(always)]
    fn cancel_entry<const MARKED: bool, M: Format, C: Cofactors<F, M>>(
        &mut self,
        c: usize,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<Option<F::Coef>, Stop> {
        let field = self.field;
        debug_assert!(
            c < self.dense.len(),
            "column {c} past the matrix's {len}",
            len = self.dense.len()
        );
        // SAFETY: c lies below the dense row's length: the dense scan's
        // bound, or a mark's column (`mark`).
        let v = field.take(unsafe { self.dense.get_unchecked_mut(c) });
        if v == F::Coef::default() {
            return Ok(None);
        }
        // SAFETY: the pivot array holds an entry per column, as the dense
        // row does (`new`).
        match unsafe { *self.pivot.get_unchecked(c) } {
            Pivot::None => return Ok(Some(v)),
            Pivot::Row(q) => {
                let (pc, pv) = (self.matrix.row(q), self.coefs(q));
                // SAFETY: as in `reduce`.
                unsafe { field.sub_mul(&mut self.dense, &pc[1..], &pv[1..], v) };
                // row-scan-time.5 design -- the marked scan subtracts by the field's sub_mul and then marks the subtracted row's columns in a pass of its own, not a row operation of each field that marks as it writes (not measured)
                if MARKED {
                    mark(&mut self.marks, &pc[1..]);
                }
                cof.sub_row(q, v);
                // matrix-stop-latency.3 design -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
                budget.spend_subtracted(pc.len())?;
            }
            Pivot::New(k) => {
                debug_assert!(
                    k.index() < self.made.len(),
                    "made row {k} past the {len} made",
                    len = self.made.len()
                );
                // SAFETY: `keep` names a made row a pivot as it pushes the
                // row, and no made row goes.
                let row = unsafe { self.made.get_unchecked(k.index()) };
                // SAFETY: a made row's columns are a matrix row's leading
                // column and columns of the dense row's scan, a coefficient
                // each.
                unsafe { field.sub_mul(&mut self.dense, &row.cols[1..], &row.coefs[1..], v) };
                // row-scan-time.5 design -- the marked scan subtracts by the field's sub_mul and then marks the subtracted row's columns in a pass of its own, not a row operation of each field that marks as it writes (not measured)
                if MARKED {
                    mark(&mut self.marks, &row.cols[1..]);
                }
                cof.sub_kept(row.kept, v);
                // matrix-stop-latency.3 design -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
                budget.spend_subtracted(row.cols.len())?;
            }
        }
        Ok(None)
    }

    /// Reduces the dense row by the pivots in every column from `from` on:
    /// an entry at a column with a pivot is cancelled, and one at a column
    /// without joins `out` (`cancel_entry`). The cofactor tracking hears of
    /// every subtraction, and each subtraction spends the subtracted row's
    /// length. The row takes the scan the counts chose before it: every
    /// column from `from` on, or the marked columns alone, which hold every
    /// entry the row's writes may have left. Then the row is counted
    /// (`count`). A stop where the budget's check stops the call, the dense
    /// row and the marks then not zero.
    fn cancel<M: Format, C: Cofactors<F, M>>(
        &mut self,
        from: MonoId,
        out: &mut Reduced<F>,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert_eq!(self.pivot.len(), self.dense.len(), "a pivot a column");
        // matrix-stop-latency.4 design -- the scan of the dense row is not spent, the rows' ticks bound it, not a spend per row of the columns the scan passed (log 0056-1)
        if self.marked {
            // row-scan-time.1 design -- the marked scan visits the columns whose mark a load, an addition, or a row operation of the row set, in increasing column order, each mark cleared as read, not every column from the row's start, the row's reach in the graph of the pivot rows found before the numerical work, a heap of the columns written, nor the scan ended at the last column a row operation reached (log 0072-1)
            let mut w = self.first_word(from);
            while let Some(c) = next_marked(&mut self.marks, &mut w) {
                if let Some(v) = self.cancel_entry::<true, M, C>(c, cof, budget)? {
                    out.cols.push(MonoId::at(c));
                    out.coefs.push(v);
                }
            }
        } else {
            for c in from.index()..self.dense.len() {
                if let Some(v) = self.cancel_entry::<false, M, C>(c, cof, budget)? {
                    out.cols.push(MonoId::at(c));
                    out.coefs.push(v);
                }
            }
        }
        self.count(from, budget);
        Ok(())
    }

    /// The word of the marks the marked scan from `from` starts at. The
    /// test profile's debug assertions check that the marks hold a word per
    /// `MARK_BITS` columns, and none before `from`.
    fn first_word(&self, from: MonoId) -> usize {
        let w = from.index() / MARK_BITS;
        debug_assert_eq!(
            self.marks.len(),
            self.dense.len().div_ceil(MARK_BITS),
            "a mark a column"
        );
        debug_assert!(
            self.marks
                .get(w)
                .is_none_or(|&word| word & ((1 << (from.index() % MARK_BITS)) - 1) == 0),
            "a mark before the start {from}"
        );
        w
    }

    /// The dense row gains k times row r, its columns marked where the row
    /// takes the marked scan, and the budget is spent the row's length,
    /// counted as added (`Budget::spend_added`). A stop where the budget's
    /// check stops the call.
    fn add(&mut self, r: RowId, k: F::Coef, budget: &mut Budget) -> Result<(), Stop> {
        let (f, matrix) = (self.field, self.matrix);
        let (cols, coefs) = (matrix.row(r), self.coefs(r));
        // SAFETY: as in `reduce`.
        unsafe { f.sub_mul(&mut self.dense, cols, coefs, f.neg(k)) };
        if self.marked {
            mark(&mut self.marks, cols);
        }
        // matrix-stop-latency.3 design -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
        budget.spend_added(cols.len())
    }

    /// Reduces the dense row as loaded, a combination of rows, by the pivots
    /// in every column from `from` on, as `reduce` does. The columns before
    /// `from` must hold no entry. The entries left, in increasing column
    /// order, not made monic, as made from row `r`, built as `reduce` builds
    /// it, and spent as `reduce` spends. `C` tracks no cofactors.
    fn reduce_loaded<M: Format, C: Cofactors<F, M>>(
        &mut self,
        r: RowId,
        from: MonoId,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<Reduced<F>, Stop> {
        let mut out = self.begin(r);
        self.cancel(from, &mut out, cof, budget)?;
        Ok(self.end(out))
    }

    // signature-zero-time.7 design -- a combination stops at its first nonzero entry in a column without a pivot, which shows it nonzero, not reduced to its end (log 0049-1)
    /// Whether the dense row as loaded, a combination of rows, reduces to
    /// zero by the pivots in every column from `from` on, as `cancel`
    /// reduces it, in the scan `cancel` would take, but making no row: false
    /// at the first nonzero entry in a column without a pivot, after the
    /// dense row is set to zero from that column on. The columns before
    /// `from` must hold no entry. The dense row and the marks are zero after
    /// either, and the combination is counted (`count`). Each row operation
    /// spends its row's length right after it, as in `cancel`. No cofactor
    /// tracking hears of it. A stop where the budget's check stops the call,
    /// the dense row and the marks then not zero.
    fn reduces_to_zero<M: Format>(
        &mut self,
        from: MonoId,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        debug_assert_eq!(self.pivot.len(), self.dense.len(), "a pivot a column");
        // matrix-stop-latency.4 design -- the scan of the dense row is not spent, the rows' ticks bound it, not a spend per row of the columns the scan passed (log 0056-1)
        let zero = 'scan: {
            if self.marked {
                // row-scan-time.1 design -- the marked scan visits the columns whose mark a load, an addition, or a row operation of the row set, in increasing column order, each mark cleared as read, not every column from the row's start, the row's reach in the graph of the pivot rows found before the numerical work, a heap of the columns written, nor the scan ended at the last column a row operation reached (log 0072-1)
                let mut w = self.first_word(from);
                while let Some(c) = next_marked(&mut self.marks, &mut w) {
                    let left =
                        self.cancel_entry::<true, M, NoCofactors>(c, &mut NoCofactors, budget)?;
                    if left.is_some() {
                        // row-scan-time.9 design -- the marked exit at a column without a pivot zeroes the entries at the marks from that column on and clears the marks, not the dense row filled with zeros to its end and every word cleared (not measured)
                        while let Some(c) = next_marked(&mut self.marks, &mut w) {
                            debug_assert!(
                                c < self.dense.len(),
                                "column {c} past the matrix's {len}",
                                len = self.dense.len()
                            );
                            // SAFETY: a mark's column lies below the dense
                            // row's length (`mark`).
                            unsafe { *self.dense.get_unchecked_mut(c) = F::Acc::default() };
                        }
                        break 'scan false;
                    }
                }
            } else {
                for c in from.index()..self.dense.len() {
                    let left =
                        self.cancel_entry::<false, M, NoCofactors>(c, &mut NoCofactors, budget)?;
                    if left.is_some() {
                        self.dense[c..].fill(F::Acc::default());
                        break 'scan false;
                    }
                }
            }
            true
        };
        self.count(from, budget);
        Ok(zero)
    }

    /// The tests of `rest`, rows that would each be reduced on its own by
    /// the pivots set before the first of them, `row` naming each
    /// (`reduce_sig`, `reduce`). The first group is the whole of `rest`,
    /// or a block of `FIRST` rows where a test has `failed` before. A group
    /// passes where s random combinations of its rows, drawn by `combine`,
    /// each reduce to zero (`reduces_to_zero`) from the leftmost
    /// leading column among its rows, and the next group then holds twice
    /// its rows, at most `BLOCK`. The tests end at a group that fails or at
    /// one of fewer than 2s rows. Each combination is a tick of the budget
    /// and each row added to it spends its length. The rows the passing
    /// groups rule zero join `zero`. A stop where the budget's check stops
    /// the call.
    fn test_groups<M: Format, T: Copy>(
        &mut self,
        rest: &[T],
        row: impl Fn(T) -> RowId,
        failed: bool,
        (combine, s): (&mut Combine, usize),
        zero: &mut Vec<RowId>,
        budget: &mut Budget,
    ) -> Result<Tested, Stop> {
        let p = self.field.characteristic();
        let mut tested = Tested {
            ruled: 0,
            failed,
            reductions: 0,
            zeros: 0,
        };
        // signature-zero-time.4 design -- a run's first test takes the whole rest of the run, not a block (log 0049-1)
        // signature-zero-time.6 tuned -- a block holds at most BLOCK rows, the plain run's block, not doubling without a cap (log 0049-1)
        let mut block = if failed { FIRST } else { usize::MAX };
        loop {
            let group_len = block.min(rest.len() - tested.ruled);
            // signature-zero-time.9 design -- a group of fewer than 2s rows takes no test, not a test of any group of two rows or more (not measured)
            if group_len < GROUP_ROWS * s {
                break;
            }
            let group = &rest[tested.ruled..tested.ruled + group_len];
            let Some(from) = group.iter().map(|&k| self.matrix.lead(row(k))).min() else {
                break;
            };
            // signature-zero-time.1 cited: monagan-2017 §2, attributes it to A. Steel's private communication and found it independently -- random linear combinations of a set's rows are reduced, and if zero is obtained k times in a row the set is discarded with a probability of error of 1/p^k
            // signature-zero-time.14 design -- the s combinations of a test are reduced one after another in the dense row, not loaded into a band and reduced in one sweep (not measured)
            let mut passed = true;
            for _ in 0..s {
                budget.tick()?;
                for &k in group {
                    self.add(row(k), self.field.coef(combine.coef(p)), budget)?;
                }
                // signature-zero-time.12 design -- each combination counts as a reduction, and one that reduces to zero as a zero, not the rows the test rules zero (not measured)
                tested.reductions += 1;
                if !self.reduces_to_zero::<M>(from, budget)? {
                    passed = false;
                    break;
                }
                tested.zeros += 1;
            }
            if !passed {
                tested.failed = true;
                break;
            }
            zero.extend(group.iter().map(|&k| row(k)));
            tested.ruled += group_len;
            block = (2 * group_len).min(BLOCK);
        }
        Ok(tested)
    }

    /// Keeps a nonzero reduced row: made monic, the pivot of its leading
    /// column, and kept by the cofactor tracking. False, and nothing kept,
    /// where its leading entry has no inverse: over several lanes, 0 at one
    /// lane.
    pub(crate) fn keep<M: Format, C: Cofactors<F, M>>(
        &mut self,
        mut row: Reduced<F>,
        cof: &mut C,
    ) -> bool {
        let f = self.field;
        // q-learn-time.4 cited: dekleine-2001 §3 -- on each division with a nonzero remainder the list of leading coefficients is checked, and a c_i = 0 means the leading term of r is lost under p_i, with very high probability a bad prime; graebe-1993 §4, independently -- a leading coefficient detected as nonzero that is zero mod p_i shows p_i unlucky
        // q-learn-time.5 design -- a disagreement of the lanes halts the run with Halt::Misfit(0), and the call returns Error::Misfit { step: 0 }, not the lane at 0 switched out and the run continued at the other prime, nor a halt of a variant of its own (not measured)
        let Some(s) = f.inv(row.coefs[0]) else {
            return false;
        };
        for c in &mut row.coefs {
            *c = f.mul(*c, s);
        }
        cof.scale(s);
        cof.keep();
        row.kept = KeepId::at(self.kept);
        self.kept += 1;
        self.pivot[row.cols[0].index()] = Pivot::New(MadeId::at(self.made.len()));
        self.made.push(row);
        true
    }

    /// Band row k's result, made from row `r` of the matrix: its vectors
    /// band row k's work vectors, empty.
    fn begin_band(&mut self, k: usize, r: RowId) -> Reduced<F> {
        let mut cols = std::mem::take(&mut self.band_cols[k]);
        let mut coefs = std::mem::take(&mut self.band_coefs[k]);
        cols.clear();
        coefs.clear();
        Reduced {
            row: r,
            cols,
            coefs,
            kept: KeepId::NONE,
        }
    }

    // row-traffic-time.11 cited: monagan-2015 §2.2 -- a row to be reduced is moved into the buffer, and if the result is not zero, the buffer contents are copied to a new sparse row
    /// Band row k's result `out` with its entries copied into vectors of
    /// its length, and the work vectors back in the reducer as band row
    /// k's. A row of no entry allocates nothing.
    fn end_band(&mut self, k: usize, mut out: Reduced<F>) -> Reduced<F> {
        let cols = out.cols.as_slice().to_vec();
        let coefs = out.coefs.as_slice().to_vec();
        self.band_cols[k] = std::mem::replace(&mut out.cols, cols);
        self.band_coefs[k] = std::mem::replace(&mut out.coefs, coefs);
        out
    }

    /// The band, where it is empty, an element per column of the matrix,
    /// every entry zero.
    fn widen_band(&mut self) {
        // row-traffic-time.10 cited: monagan-2007 §2.2 -- to further amortize the cost of memory allocation the merge buffers are reused in the next call rather than freed each time
        // The band from the buffers, cleared by `new`, is zeroed whole at
        // the reduction's first band, up to the matrix's width: no band the
        // buffers held is trusted to be zero.
        if self.band.is_empty() {
            self.band
                .resize(self.matrix.columns(), [F::Acc::default(); BAND]);
        }
    }

    /// Reduces the rows `rows`, 2 to BAND of them, in one sweep, each as
    /// `reduce` reduces it from its leading column, in turn: by the pivots
    /// set before the band and by the rows of the band before it, made
    /// monic. Each row's leading column holds a pivot. `F::BANDS` holds,
    /// and no cofactors are tracked. The rows are loaded into the band and
    /// swept (`sweep_band`): each row's result in the rows' order. A stop
    /// where the budget's check stops the call, the band then not zero.
    pub(crate) fn reduce_band(
        &mut self,
        rows: &[RowId],
        budget: &mut Budget,
    ) -> Result<Vec<Reduced<F>>, Stop> {
        let n = rows.len();
        debug_assert!((2..=BAND).contains(&n), "a band of {n} rows");
        debug_assert!(F::BANDS, "a field with bands");
        let (f, matrix) = (self.field, self.matrix);
        self.widen_band();
        let mut out: Vec<Reduced<F>> = Vec::with_capacity(n);
        let mut from = MonoId::NONE;
        for (k, &r) in rows.iter().enumerate() {
            let (cols, coefs) = (matrix.row(r), self.coefs(r));
            debug_assert!(
                !matches!(self.pivot[cols[0].index()], Pivot::None),
                "a band row's leading column has a pivot"
            );
            from = from.min(cols[0]);
            // SAFETY: as in `reduce`, and k < n ≤ BAND.
            unsafe { f.load_band(&mut self.band, k, cols, coefs) };
            out.push(self.begin_band(k, r));
        }
        self.sweep_band(n, from, out, budget)
    }

    // plain-reduction-time.2 design -- a band's combinations are formed in one pass over each row of the block, by `sub_mul_band` with the negated coefficients, not a pass per band row nor each combination formed in the dense row and moved to the band (log 0062-1)
    /// Band row j gains k[j] times row r, for every j < BAND, in one pass
    /// over the row. A coefficient 0 leaves its band row as it is. The band
    /// is first widened where it is empty (`widen_band`). `F::BANDS` holds.
    /// The pass spends `BAND` times the row's length, counted as added, and
    /// a stop where the budget's check stops the call.
    pub(crate) fn add_band(
        &mut self,
        r: RowId,
        k: &[F::Coef; BAND],
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert!(F::BANDS, "a field with bands");
        let (f, matrix) = (self.field, self.matrix);
        self.widen_band();
        let (cols, coefs) = (matrix.row(r), self.coefs(r));
        let neg = k.map(|c| f.neg(c));
        // SAFETY: as in `reduce`: the band holds an element per column of
        // the matrix (`widen_band`).
        unsafe { f.sub_mul_band(&mut self.band, cols, coefs, &neg) };
        // plain-reduction-time.6 design -- each row added to a band spends its length once per band row, BAND times, right after its pass, not its length once per pass (not measured)
        budget.spend_added(BAND * cols.len())
    }

    /// Reduces band rows 0 to n − 1 as loaded, 2 ≤ n ≤ BAND, combinations
    /// of rows made from row `r` of the matrix, by the pivots in every
    /// column from `from` on (`sweep_band`). The columns before `from` must
    /// hold no entry of the band. Each row's result in band order, as made
    /// from row `r`. A stop where the budget's check stops the call, the
    /// band then not zero.
    fn reduce_band_loaded(
        &mut self,
        n: usize,
        r: RowId,
        from: MonoId,
        budget: &mut Budget,
    ) -> Result<Vec<Reduced<F>>, Stop> {
        debug_assert!((2..=BAND).contains(&n), "a band of {n} rows");
        let out: Vec<Reduced<F>> = (0..n).map(|k| self.begin_band(k, r)).collect();
        self.sweep_band(n, from, out, budget)
    }

    /// Reduces band rows 0 to n − 1 in one sweep over the columns from
    /// `from` on, each in turn as `cancel` reduces the dense row: by the
    /// pivots set before the band and by the band rows before it, made
    /// monic. The band rows from n on hold zeros. `F::BANDS` holds, and no
    /// cofactors are tracked. Each pivot row subtracted spends its length
    /// once per band row it is subtracted from, right after it, counted as
    /// subtracted. `out` holds
    /// the n rows' results as `begin_band` made them. Each row's result in
    /// band order, not monic, empty where the row reduced to zero, built in
    /// the band rows' work vectors and copied at its length (`end_band`).
    /// The band is zero after, where every entry lay from `from` on. A stop
    /// where the budget's check stops the call, the band then not zero.
    #[allow(
        clippy::many_single_char_names,
        reason = "n, k, i, a, v are the notation of the comments' formulas, after faugere-2010 §4.4"
    )]
    fn sweep_band(
        &mut self,
        n: usize,
        from: MonoId,
        mut out: Vec<Reduced<F>>,
        budget: &mut Budget,
    ) -> Result<Vec<Reduced<F>>, Stop> {
        debug_assert!(F::BANDS, "a field with bands");
        debug_assert_eq!(out.len(), n, "a result a band row");
        let (field, matrix) = (self.field, self.matrix);
        let zero = F::Coef::default();
        // Whether band row k has taken its leading column, and the inverse
        // of its entry there.
        let mut led = [false; BAND];
        let mut inv = [zero; BAND];
        // row-traffic-time.4 cited: faugere-2010 §4.4 -- the row operations done on the blocks are kept in a matrix P, and each later block is updated by a left product by P before it is reduced
        // Band row k as the one-row reduction holds it is its accumulator
        // less Σ_{i<k} rel[k][i] times band row i as it holds it: rel[k][i]
        // is band row k's multiplier of band row i as it holds it, the
        // multiples of band row i made monic it subtracted times the inverse
        // of band row i's entry at the column it took as its lead.
        let mut rel = [[zero; BAND]; BAND];
        let mut has_rel = [false; BAND];
        // row-scan-time.7 design -- the back-substitution and the band sweeps keep their dense scans, not the marked scan (log 0072-1)
        for column in from.index()..self.band.len() {
            // The accumulator values, zero past n.
            let mut a = [zero; BAND];
            let mut nonzero = 0;
            for (a, entry) in a.iter_mut().zip(&mut self.band[column]) {
                *a = field.take(entry);
                nonzero += usize::from(*a != zero);
            }
            if nonzero == 0 {
                continue;
            }
            // Every pivot is monic, so subtracting a[k] times its tail from
            // each accumulator clears the column for every band row.
            let pivot: Option<(&[MonoId], &[F::Coef])> = match self.pivot[column] {
                Pivot::None => None,
                Pivot::Row(row) => Some((matrix.row(row), self.coefs(row))),
                Pivot::New(made_index) => {
                    let made_row = &self.made[made_index.index()];
                    Some((&made_row.cols, &made_row.coefs))
                }
            };
            if let Some((pc, pv)) = pivot {
                if nonzero == 1
                    && let Some(k) = a.iter().position(|&v| v != zero)
                {
                    // row-traffic-time.5 cited: boyer-2016 §4.2 -- where only one of a multiline's rows has an entry, a specialized AXPY reduces one row, and lightweight tests before the loop run it only on nonzero multipliers
                    // SAFETY: as in `cancel`, a pivot row's columns lie
                    // below the matrix's width, the band's length, and
                    // k < n ≤ BAND.
                    unsafe { field.sub_mul_band_row(&mut self.band, k, &pc[1..], &pv[1..], a[k]) };
                    // row-traffic-time.12 design -- each pivot row a band subtracts spends its length once per band row it is subtracted from, right after the subtraction, not once per band nor no spend in the sweep (log 0044-1)
                    budget.spend_subtracted(pc.len())?;
                } else {
                    // SAFETY: as above.
                    unsafe { field.sub_mul_band(&mut self.band, &pc[1..], &pv[1..], &a) };
                    // row-traffic-time.12 design -- each pivot row a band subtracts spends its length once per band row it is subtracted from, right after the subtraction, not once per band nor no spend in the sweep (log 0044-1)
                    budget.spend_subtracted(pc.len() * nonzero)?;
                }
                continue;
            }
            // No pivot: each band row's value is its accumulator value less
            // its relations' multiples of the values of the rows before it.
            let mut values = [zero; BAND];
            // The band row that took this column as its leading column.
            let mut claimed: Option<usize> = None;
            for k in 0..n {
                let mut v = a[k];
                if has_rel[k] {
                    for i in 0..k {
                        if rel[k][i] != zero && values[i] != zero {
                            v = field.sub(v, field.mul(rel[k][i], values[i]));
                        }
                    }
                }
                if v == zero {
                    continue;
                }
                // The one-row reduction subtracts v times band row i made
                // monic, and the value here becomes zero.
                if let Some(i) = claimed {
                    rel[k][i] = field.add(rel[k][i], field.mul(v, inv[i]));
                    has_rel[k] = true;
                    continue;
                }
                values[k] = v;
                if !led[k] {
                    led[k] = true;
                    #[allow(
                        clippy::expect_used,
                        reason = "v is nonzero, and bands run only over Gf31, a field of one lane"
                    )]
                    let inverse = field.inv(v).expect("a value is nonzero");
                    inv[k] = inverse;
                    claimed = Some(k);
                }
                out[k].cols.push(MonoId::at(column));
                out[k].coefs.push(v);
            }
        }
        Ok(out
            .into_iter()
            .enumerate()
            .map(|(k, row)| self.end_band(k, row))
            .collect())
    }

    // order-family.18 cited: faugere-1999 §2.2, Definition 2.3 -- the row echelon form of M is unique, the reduced Gröbner basis of its rows' equations, each pivot column zero in every other row; demin-2024 §4.2 -- the rows of D' are inter-reduced to obtain the reduced row echelon form
    /// Inter-reduces the made rows to the reduced row echelon form: by
    /// decreasing leading column, each row's entries that are the leading
    /// column of another made row are cancelled with that row, already
    /// back-substituted. A row with such an entry is kept again by the
    /// cofactor tracking and replaces itself in `made`, built in the work
    /// vectors and copied at its length. Each row ticks the budget, and each
    /// subtraction spends the subtracted row's length, counted as
    /// subtracted.
    /// No row changes its leading column, and none is made monic again:
    /// each subtracted row is monic, and the leading entry is untouched.
    fn back_substitute<M: Format, C: Cofactors<F, M>>(
        &mut self,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let f = self.field;
        let zero = F::Coef::default();
        let mut by_lead: Vec<MadeId> = (0..self.made.len()).map(MadeId::at).collect();
        by_lead.sort_unstable_by_key(|&k| std::cmp::Reverse(self.made[k.index()].cols[0]));
        for k in by_lead {
            budget.tick()?;
            let row = &self.made[k.index()];
            // order-family.20 design -- a made row whose tail holds no leading column of another made row is left as it is, not begun and kept again by the cofactor tracking (not measured)
            let new = |&c: &MonoId| self.pivot[c.index()].is_new();
            if !row.cols[1..].iter().any(new) {
                continue;
            }
            // SAFETY: as in `cancel`, for a made row.
            unsafe { f.load(&mut self.dense, &row.cols, &row.coefs) };
            cof.start_kept(row.kept);
            let (source, lead) = (row.row, row.cols[0]);
            let mut out = self.begin(source);
            out.cols.push(lead);
            out.coefs.push(f.take(&mut self.dense[lead.index()]));
            // A subtracted row changes only columns right of its leading one,
            // and holds no leading column of a made row and no reducer's.
            // row-scan-time.7 design -- the back-substitution and the band sweeps keep their dense scans, not the marked scan (log 0072-1)
            for c in lead.index() + 1..self.dense.len() {
                let v = f.take(&mut self.dense[c]);
                if v == zero {
                    continue;
                }
                match self.pivot[c] {
                    Pivot::New(j) => {
                        let p = &self.made[j.index()];
                        // SAFETY: as in `cancel`, for a made row.
                        unsafe { f.sub_mul(&mut self.dense, &p.cols[1..], &p.coefs[1..], v) };
                        cof.sub_kept(p.kept, v);
                        // matrix-stop-latency.3 design -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
                        budget.spend_subtracted(p.cols.len())?;
                    }
                    Pivot::None => {
                        out.cols.push(MonoId::at(c));
                        out.coefs.push(v);
                    }
                    #[allow(
                        clippy::unreachable,
                        reason = "`reduce` sets every reducer's pivot before its first made row, and `replay` does where it back-substitutes, as its debug_assert checks"
                    )]
                    Pivot::Row(q) => unreachable!(
                        "made row {k} holds column {c}, the pivot of reducer row {q}, which the forward pass cancels"
                    ),
                }
            }
            cof.keep();
            out.kept = KeepId::at(self.kept);
            self.kept += 1;
            self.made[k.index()] = self.end(out);
        }
        Ok(())
    }
}

// f4-gfp.6 cited: boyer-2016 §4.3 -- when only the reduction of D is needed, the TRSM step can be omitted, and it is enough to reduce the elements below the pivots
/// Reduces the matrix of a round. Each reducer row is the pivot of its
/// leading column, and of the given rows that share a leading column the
/// one with the fewest terms is. The other given rows are reduced by
/// decreasing leading column, those of one leading column by decreasing
/// row index, by the pivots and the new pivots found
/// before them, each row a tick of the budget. Pivot rows are not reduced.
/// Where `combine` is Some and p² ≥ 2^31, p ≥ 46341, those rows are
/// reduced in random
/// combinations instead, in blocks of consecutive rows of that
/// order, the first the rows that lead farthest right, each
/// combination a tick of the budget, until `ZEROS` of a block's
/// combinations have reduced to zero or they have made as many pivots as
/// the block has rows. A block holds `BLOCK` rows, or twice the rows of
/// the block before it where that block made no pivot, or the rows left
/// where they are fewer. A block of more than `BLOCK` rows that has made
/// more than `GROWN_PIVOTS` pivots stops before its next band or
/// combination: its pivots stay, and the blocks after it take its rows
/// again from its first. `C` must track no cofactors then. Where
/// `takes_bands` holds, a block of n rows forms its combinations
/// min(`BAND`, n − pivots) at a time in the band and reduces them in one
/// sweep (`Reducer::add_band`, `Reducer::reduce_band_loaded`), while none
/// has reduced to zero and n − pivots ≥ 2, pivots the rows the block has
/// made; then one at a time. Where `full` is true the
/// new rows are then inter-reduced (`Reducer::back_substitute`), and the
/// call report's step ends its stage `Reduce` before and `Backsub` after.
/// Once the pivots are set the setup spends the matrix's rows and columns,
/// each row added to a combination spends its length, to a band `BAND`
/// times its length, and each row operation its row's
/// (`Reducer::cancel`), in a band once per band row it is subtracted from.
/// `Halt::Misfit(0)` where a row to keep has no inverse at its leading
/// entry, over several lanes 0 at one lane (`Reducer::keep`), and a stop
/// where the budget's check stops the call. The dense row
/// and the work vectors come from `bufs`, and go back to it where the
/// reduction returns its result or a misfit.
///
/// Where the rows are reduced each on its own, `tests` is Some, `C`
/// tracks no cofactors, and p > 3, the rows are tested as `reduce_sig`
/// tests a run's: once `STREAK` rows have reduced to zero one after
/// another, a test rules a group of the rows after them zero by
/// s = `zero_combinations(p)` random combinations of the group's rows that
/// each reduce to zero (`Reducer::reduces_to_zero`), each combination a
/// reduction and a tick of the budget, and each row added to it spending
/// its length. The group is the rows to the matrix's end, or after a
/// failed test a block of `FIRST` rows, doubling after each block that
/// passes up to `BLOCK`. A group of fewer than 2s rows takes no test. A
/// failed test leaves its rows to be reduced on their own. The rows a test
/// rules zero join `Reduction::zero` and make no pivot. Over several lanes
/// p is the first lane's prime.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, basis, matrix, and cofactors, the budget, whether the reduction is full, the combinations, the buffers, and the tests, none of which belongs with another"
)]
pub(crate) fn reduce<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    basis: &Basis<F, D>,
    matrix: &Matrix,
    cof: &mut C,
    budget: &mut Budget,
    full: bool,
    combine: Option<&mut Combine>,
    bufs: &mut Buffers<F>,
    tests: Option<&mut Combine>,
) -> Result<Reduction<F>, Halt> {
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    for r in (matrix.given..matrix.rows.len()).map(RowId::at) {
        let c = matrix.lead(r).index();
        debug_assert!(matches!(red.pivot[c], Pivot::None), "one reducer a column");
        red.pivot[c] = Pivot::Row(r);
    }
    // f4-gfp.5 cited: berthomieu-2021 §4.1 (4) and §4.2 -- the sparsest possible rows are the pivot rows, and the sparsest row for each pivot goes into the upper pivot part
    let mut rest: Vec<RowId> = Vec::new();
    for r in (0..matrix.given).map(RowId::at) {
        let c = matrix.lead(r).index();
        match red.pivot[c] {
            Pivot::Row(q) if matrix.row(q).len() > matrix.row(r).len() => {
                rest.push(q);
                red.pivot[c] = Pivot::Row(r);
            }
            Pivot::Row(_) => rest.push(r),
            Pivot::None | Pivot::New(_) => red.pivot[c] = Pivot::Row(r),
        }
    }
    // cyclic-gap.1 design -- the rows to reduce go by decreasing leading column, those of one leading column by decreasing row index, not by increasing leading column nor the sparsest first (log 0066-1)
    rest.sort_unstable_by_key(|&r| std::cmp::Reverse((matrix.lead(r), r)));
    // matrix-stop-latency.8 design -- a reduction's setup spends the matrix's rows and columns once, after its reducer is made and any pivots it sets before its first row, not no spend for the setup (log 0056-1)
    budget.spend(matrix.rows.len() + matrix.columns())?;
    let mut zero = Vec::new();
    let (mut reductions, mut zeros) = (0, 0);
    let p = field.characteristic();
    match combine {
        Some(combine) if combines_rows(p) => {
            // gfp-unsolved.8 design -- a run that tracks cofactors reduces each row on its own, not combinations with their cofactors tracked through a hook that begins an empty combination and adds its rows (log 0014-1)
            debug_assert!(!C::TRACKS, "no cofactors of a combination");
            // gfp-unsolved.2 cited: monagan-2017 §2, via berthomieu-2021 §4.2 and demin-2024 §4.2, attributes it to A. Steel's private communication and found it independently -- the rows are partitioned into sets, random linear combinations of a set's rows are reduced, and zero comes by chance with probability about 1/p; berthomieu-2021 §4.2 -- blocks of ℓ rows, a random linear combination reduced until the first reduction to zero or ℓ of them
            // plain-reduction-time.5 design -- a plain run's matrix takes bands for its combinations where its rows hold BAND_ENTRIES entries or more in the mean, as a signature round's, not in every matrix (log 0062-1)
            let bands = takes_bands::<F, M, C>(matrix);
            // cyclic-gap.2 design -- the blocks are cut from the start of that order, the first the BLOCK rows that lead farthest right, not cut by increasing leading column and taken last to first (log 0066-1)
            // plain-zero-time.8 design -- blocks grow in every matrix that combines rows, not only where the matrix takes bands (log 0077-1)
            let (mut start, mut size) = (0, BLOCK);
            while start < rest.len() {
                let first = start;
                let block = &rest[start..rest.len().min(start + size)];
                start += block.len();
                let grown = block.len() > BLOCK;
                // Every row of the block leads at or right of its last.
                let from = matrix.lead(block[block.len() - 1]);
                let (mut pivots, mut done, mut stopped) = (0, 0, false);
                // plain-reduction-time.1 invented -- a block's random combinations are formed up to BAND at a time in the band and reduced in one sweep, `Reducer::sweep_band`'s, not one combination at a time in the dense row (log 0062-1)
                // plain-reduction-time.4 design -- a block takes bands of min(BAND, n − pivots) combinations while none has reduced to zero and two pivots or more are left to make, then one at a time, not bands to the block's end, bands of BAND always, nor one band and then one at a time (not measured)
                // plain-zero-time.9 design -- after a band whose combinations all made pivots the block takes another band, not one combination alone that tests whether the block is spent (log 0077-1)
                while bands && done == 0 && block.len() - pivots >= 2 {
                    if grown && pivots > GROWN_PIVOTS {
                        stopped = true;
                        break;
                    }
                    let lanes = BAND.min(block.len() - pivots);
                    // A bound that stops the call at a band combination's
                    // tick stops it as at that combination's own: the
                    // round keeps nothing.
                    for _ in 0..lanes {
                        budget.tick()?;
                    }
                    for &r in block {
                        // plain-reduction-time.3 design -- a block row's coefficients are drawn together, one for each band row in band-row order, not combination by combination (not measured)
                        let k: [F::Coef; BAND] = std::array::from_fn(|j| {
                            if j < lanes {
                                field.coef(combine.coef(p))
                            } else {
                                F::Coef::default()
                            }
                        });
                        red.add_band(r, &k, budget)?;
                    }
                    // Kept in band order, the band rows set the pivots the
                    // same combinations reduced one after another set: the
                    // sweep reduced each by the band rows before it.
                    for row in red.reduce_band_loaded(lanes, block[0], from, budget)? {
                        reductions += 1;
                        if row.cols.is_empty() {
                            done += 1;
                            zeros += 1;
                            continue;
                        }
                        pivots += 1;
                        if !red.keep(row, cof) {
                            red.give_back(bufs);
                            return Err(Halt::Misfit(0));
                        }
                    }
                }
                // gfp-unsolved.19 design -- a block is also done once its combinations have made as many pivots as it has rows, which span it, and not after as many combinations as it has rows, where a combination of a full block that reduces to zero by chance leaves it a pivot short (log 0014-2)
                // plain-zero-time.3 design -- each block ends after ZEROS of its combinations reduce to zero, not at its first zero with two combinations of every row to reduce at the round's end (log 0077-1)
                while !stopped && pivots < block.len() && done < ZEROS {
                    if grown && pivots > GROWN_PIVOTS {
                        stopped = true;
                        break;
                    }
                    budget.tick()?;
                    for &r in block {
                        let k = field.coef(combine.coef(p));
                        red.add(r, k, budget)?;
                    }
                    let row = red.reduce_loaded(block[0], from, cof, budget)?;
                    reductions += 1;
                    if row.cols.is_empty() {
                        done += 1;
                        zeros += 1;
                        continue;
                    }
                    pivots += 1;
                    if !red.keep(row, cof) {
                        red.give_back(bufs);
                        return Err(Halt::Misfit(0));
                    }
                }
                if stopped {
                    start = first;
                    size = BLOCK;
                } else if pivots == 0 {
                    // plain-zero-time.1 invented -- a block after a block that made no pivot holds twice that block's rows, not every block BLOCK rows (log 0077-1)
                    // plain-zero-time.2 design -- the grown block is a block, its combinations making its pivots until ZEROS reduce to zero, not a test group of two combinations that rules it zero or hands its rows to blocks of BLOCK (log 0077-1)
                    // plain-zero-time.4 tuned -- twice the rows, not four times (log 0077-1)
                    // plain-zero-time.5 tuned -- a block holds at most the rows left, not at most 1024 rows (log 0077-1)
                    size = 2 * block.len();
                } else {
                    // plain-zero-time.7 tuned -- a block after a block that made a pivot holds BLOCK rows, not twice the rows where the block made one or two pivots (log 0077-1)
                    size = BLOCK;
                }
            }
        }
        Some(_) | None => {
            // learn-zero-time.1 invented -- a learn run's plain run, where no cofactors are tracked, rules its rows zero by signature-zero-time.1's test of random combinations, the rows a test rules zero recorded as rows that reduced to zero, not each row reduced on its own (log 0063-1)
            // learn-zero-time.6 design -- over two lanes a test takes the s of the first lane's prime, the field's characteristic, not the s of the product of the lanes' primes (not measured)
            let mut test = tests.filter(|_| !C::TRACKS).zip(zero_combinations(p));
            // The rows reduced to zero one after another, and whether a
            // test has failed.
            let (mut streak, mut failed) = (0, false);
            let mut next = 0;
            while next < rest.len() {
                let r = rest[next];
                next += 1;
                budget.tick()?;
                let lead = matrix.lead(r);
                let row = red.reduce(r, lead, cof, budget)?;
                reductions += 1;
                if !row.cols.is_empty() {
                    streak = 0;
                    if !red.keep(row, cof) {
                        red.give_back(bufs);
                        return Err(Halt::Misfit(0));
                    }
                    continue;
                }
                zero.push(r);
                zeros += 1;
                streak += 1;
                let Some((combine, s)) = test.as_mut() else {
                    continue;
                };
                // zero-rows-time.4 tuned -- a test starts after STREAK rows reduced to zero one after another, 2, not after each row that reduces to zero (log 0078-1)
                if streak < STREAK {
                    continue;
                }
                streak = 0;
                // learn-zero-time.2 design -- the plain run's test takes the signature run's groups, the rows to the matrix's end first and after a failed test blocks of FIRST rows doubling up to BLOCK, not blocks of BLOCK rows from the first test (log 0063-1)
                // zero-rows-time.3 design -- a failed test hands its rows to the reductions each on its own, not its first half tested again while the half holds 2s rows or more (log 0078-1)
                let tested = red.test_groups::<M, RowId>(
                    &rest[next..],
                    |r| r,
                    failed,
                    (combine, *s),
                    &mut zero,
                    budget,
                )?;
                failed = tested.failed;
                reductions += tested.reductions;
                zeros += tested.zeros;
                // zero-rows-time.2 design -- after a failed test the rows go on each on its own until the next streak, not a test of the next block right after the next row that makes a pivot (log 0078-1)
                next += tested.ruled;
            }
        }
    }
    if full {
        budget.mark(Stage::Reduce);
        red.back_substitute(cof, budget)?;
        budget.mark(Stage::Backsub);
    }
    red.give_back(bufs);
    Ok(Reduction {
        made: red.made,
        zero,
        pivot: red.pivot,
        taken: Vec::new(),
        reductions,
        zeros,
    })
}

// signature-pruning.12 cited: eder-2017 §13, Algorithm F4-RB -- the rows sorted by signature, the row echelon form computed without row swapping; lairez-2022 §5.7 -- the reductant reduced modulo AG^{<σ} ∪ N, multiples of N not used
/// Reduces the matrix of a signature round, whose rows have the signatures
/// `sigs`, in increasing order of signature. A reducer row becomes the
/// pivot of its leading column where that column has none. Where it has
/// one, the reducer row is dropped where the row before it has its
/// signature, and is reduced as a reductant otherwise. A given row, a
/// reductant, is reduced by the pivots set before it. Where `takes_bands`
/// holds, a run of consecutive reductants is reduced in bands of up to
/// `BAND` rows (`band_end`, `Reducer::reduce_band`), each row as it is
/// reduced alone. Each reduction is a tick of the budget. The sort by
/// signature spends as `by_signature` does, the setup the matrix's rows
/// and columns once the reducer is made, and each row operation its row's
/// length (`Reducer::cancel`), in a band once per band row it is
/// subtracted from (`Reducer::reduce_band`). The new pivots each make a
/// new element with their
/// row's signature, and the rows that reduced to zero are syzygies.
/// `Halt::Misfit(0)` where a row to keep has no inverse at its leading
/// entry, as in `reduce`, and a stop where the budget's check stops the
/// call. The dense row, the band, and the work vectors come from
/// `bufs`, and go back to it where the reduction returns its result or a
/// misfit.
///
/// A run is a stretch of the order that no reducer row setting a pivot
/// interrupts. Where `combine` is Some, the field has one lane, `C` tracks
/// no cofactors, and p > 3, a run's rows are tested: once `STREAK` rows of
/// the run have reduced to zero one after another, alone or in a band, a
/// test rules a group of the rows after them zero by
/// s = `zero_combinations(p)` random combinations of the group's rows that
/// each reduce to zero (`Reducer::reduces_to_zero`), each combination a
/// reduction and a tick of the budget, and each row added to it spending
/// its length. After a band the group starts past the band's last row. The
/// group is the run's rest, or after a failed test in the run a block of
/// `FIRST` rows, doubling after each block that passes up to `BLOCK`. A
/// group of fewer than 2s rows takes no test. A failed test leaves its
/// rows to be reduced as the run's other rows are.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, basis, matrix, and signatures, the cofactors, the budget, the buffers, and the combinations, none of which belongs with another"
)]
pub(crate) fn reduce_sig<F: Field, M: Format, O: Order<M>, D: Module, C: Cofactors<F, M>>(
    field: &F,
    table: &Table<M>,
    order: &O,
    basis: &Basis<F, D>,
    matrix: &Matrix,
    sigs: &[Sig],
    cof: &mut C,
    budget: &mut Budget,
    bufs: &mut Buffers<F>,
    combine: Option<&mut Combine>,
) -> Result<Reduction<F>, Halt> {
    // signature-pruning.13 design -- a reductant comes before every reducer of its own signature (not measured)
    // The reductants are the given rows, the first: at equal signatures the
    // row decides, and a reductant comes first.
    debug_assert_eq!(sigs.len(), matrix.rows.len());
    let rows = by_signature(table, order, sigs, budget)?;
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    // matrix-stop-latency.8 design -- a reduction's setup spends the matrix's rows and columns once, after its reducer is made and any pivots it sets before its first row, not no spend for the setup (log 0056-1)
    budget.spend(matrix.rows.len() + matrix.columns())?;
    let mut zero = Vec::new();
    let (mut reductions, mut zeros) = (0, 0);
    let bands = takes_bands::<F, M, C>(matrix);
    let p = field.characteristic();
    // signature-zero-time.10 design -- the tests run where no cofactors are tracked, neither the trace nor the learner listens, and the field has one lane, not in a learn run as well (log 0049-1)
    let mut test = combine
        .filter(|_| !C::TRACKS && tests_rows::<F>(p))
        .zip(zero_combinations(p));
    // The run's rows reduced to zero one after another, alone or in a
    // band, and whether a test of the run has failed.
    let (mut streak, mut failed) = (0, false);
    // The positions of the run's rest in the order.
    let mut rest: Vec<usize> = Vec::new();
    let mut next = 0;
    while next < rows.len() {
        let start = next;
        let row_id = rows[start];
        // row-traffic-time.1 cited: martani-2012 §3.5.1, via boyer-2016 §2 and §4.2 -- the multiline row reduction reduces two rows at the same time, each entry of the reducer read once for both
        let end = if bands && row_id.index() < matrix.given {
            band_end(&rows, matrix.given, start)
        } else {
            start + 1
        };
        next = end;
        if end - start >= 2 {
            // A bound that stops the call at a band row's tick stops
            // it as at that row's own: the round keeps nothing.
            for _ in start..end {
                budget.tick()?;
            }
            for row in red.reduce_band(&rows[start..end], budget)? {
                reductions += 1;
                if row.cols.is_empty() {
                    zero.push(row.row);
                    zeros += 1;
                    streak += 1;
                    continue;
                }
                streak = 0;
                if !red.keep(row, cof) {
                    red.give_back(bufs);
                    return Err(Halt::Misfit(0));
                }
            }
        } else {
            let lead = matrix.lead(row_id);
            if row_id.index() >= matrix.given {
                if red.pivot[lead.index()].is_none() {
                    red.pivot[lead.index()] = Pivot::Row(row_id);
                    // A run starts after the row.
                    (streak, failed) = (0, false);
                    continue;
                }
                // gfp-unsolved.13 cited: eder-2017 §6.2, Lemma (singular criterion) -- for any signature T exactly one element needs to be handled
                if start > 0 && sigs[rows[start - 1].index()] == sigs[row_id.index()] {
                    continue;
                }
                // gfp-unsolved.12 cited: eder-2017 §13, Algorithm F4-RB -- the row echelon form of M_d computed without row swapping, and every row whose leading term changed joins the basis, or the syzygies where it is zero
                // Symbolic preprocessing adds one reducer a column, and a
                // reductant sets no `Row` pivot: the pivot is a row made before.
                debug_assert!(
                    matches!(red.pivot[lead.index()], Pivot::New(_)),
                    "a reducer row is the only one of its column"
                );
            } else {
                // The reducer of least signature for the reductant's leading
                // monomial lies below it: the round took the reductant for that.
                debug_assert!(
                    !matches!(red.pivot[lead.index()], Pivot::None),
                    "a reductant's leading column has a pivot"
                );
            }
            budget.tick()?;
            // signature-pruning.14 cited: eder-2017 §14.3 (2) -- full s-reductions mostly behave better than only top s-reductions
            let row = red.reduce(row_id, lead, cof, budget)?;
            reductions += 1;
            if !row.cols.is_empty() {
                streak = 0;
                if !red.keep(row, cof) {
                    red.give_back(bufs);
                    return Err(Halt::Misfit(0));
                }
                continue;
            }
            zero.push(row_id);
            zeros += 1;
            streak += 1;
        }
        // signature-zero-time.13 design -- a band's rows count in the run's streak, and where it stands at STREAK or more after the band's last row a test of the rest after the band starts, not the streak counting the rows reduced alone, a band's rows leaving it at 0 (log 0061-1)
        let Some((combine, s)) = test.as_mut() else {
            continue;
        };
        if streak < STREAK {
            continue;
        }
        streak = 0;
        let s = *s;
        // signature-zero-time.2 design -- a test combines rows of one run, which have the same pivots before them, not rows across a reducer row that sets a pivot (not measured)
        // The rest: the rows from `next` up to the first reducer row that
        // sets a pivot, or the round's end, the rows dropped there left out.
        rest.clear();
        let mut run_end = rows.len();
        for position in next..rows.len() {
            let row = rows[position];
            if row.index() >= matrix.given {
                if red.pivot[matrix.lead(row).index()].is_none() {
                    run_end = position;
                    break;
                }
                if sigs[rows[position - 1].index()] == sigs[row.index()] {
                    continue;
                }
            }
            rest.push(position);
        }
        let tested = red.test_groups::<M, usize>(
            &rest,
            |position| rows[position],
            failed,
            (combine, s),
            &mut zero,
            budget,
        )?;
        failed = tested.failed;
        reductions += tested.reductions;
        zeros += tested.zeros;
        let tested = tested.ruled;
        next = rest.get(tested).copied().unwrap_or(run_end);
    }
    red.give_back(bufs);
    Ok(Reduction {
        made: red.made,
        zero,
        pivot: red.pivot,
        taken: rows,
        reductions,
        zeros,
    })
}

// trace-replay.16 cited: demin-2024 §4.4 -- the apply stage mirrors the F4 computation without the redundant rows, and caches the order of rows and columns in the matrices; parisse-2013 §4, step 8, independently -- care must be taken to keep row ordering when learning is active
/// Reduces a replayed matrix, its rows in order: a row not in `reductees`
/// becomes the pivot of its leading column, and a row in it is reduced by
/// the pivots set before it, a tick of the budget each. `reductees` is
/// increasing, and `leads` holds the monomial each must reduce to. Where
/// `full` is true the rows made are then inter-reduced, and the report's
/// stages marked, as `reduce` does.
/// The setup spends the matrix's rows and columns once the reducer is
/// made, and each row operation its row's length (`Reducer::cancel`). A
/// stop where the budget's check stops the call.
/// The rows made, or None where one reduced to zero or to another leading
/// monomial, where a counted row's leading entry is not a unit, or where a
/// pivot row's leading column has a pivot. The dense row and the work
/// vectors come from `bufs`, and go back to it where the reduction returns
/// its result, None among them.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, basis, and matrix, the rows and leads the record names, the cofactors, the budget, whether the reduction is full, and the buffers, none of which belongs with another"
)]
pub(crate) fn replay<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    basis: &Basis<F, D>,
    matrix: &Matrix,
    reductees: &[RowId],
    leads: &[MonoId],
    cof: &mut C,
    budget: &mut Budget,
    full: bool,
    bufs: &mut Buffers<F>,
) -> Result<Option<Vec<Reduced<F>>>, Stop> {
    debug_assert!(reductees.is_sorted(), "the counted rows in row order");
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    // matrix-stop-latency.8 design -- a reduction's setup spends the matrix's rows and columns once, after its reducer is made and any pivots it sets before its first row, not no spend for the setup (log 0056-1)
    budget.spend(matrix.rows.len() + matrix.columns())?;
    let mut next = reductees.iter().zip(leads).peekable();
    for r in (0..matrix.rows.len()).map(RowId::at) {
        let c = matrix.lead(r);
        let Some((_, &lead)) = next.next_if(|&(&q, _)| q == r) else {
            if !red.pivot[c.index()].is_none() {
                red.give_back(bufs);
                return Ok(None);
            }
            // `back_substitute` meets no reducer's column in a made row.
            debug_assert!(
                !full || red.made.is_empty(),
                "pivot row {r} after {len} made rows of an inter-reduced record",
                len = red.made.len()
            );
            red.pivot[c.index()] = Pivot::Row(r);
            continue;
        };
        budget.tick()?;
        let row = red.reduce(r, c, cof, budget)?;
        // trace-replay.6 design -- a replay checks each counted row's result against the recorded leading monomial and stops at the first that differs, not the leading monomials of the output compared with the learn run's after the replay (not measured)
        if row.cols.first().map(|&c| matrix.monomials[c.index()]) != Some(lead) {
            red.give_back(bufs);
            return Ok(None);
        }
        // q-later-runs-time.4 design -- a pair fits only where every counted row's leading entry is a unit at both primes, and where it does not each prime replays alone, not the pair a misfit at both primes nor the lane that fits taken from the pair's run (not measured)
        if !field.is_unit(row.coefs[0]) {
            red.give_back(bufs);
            return Ok(None);
        }
        let kept = red.keep(row, cof);
        debug_assert!(kept, "a unit leads the row");
    }
    if full {
        budget.mark(Stage::Reduce);
        red.back_substitute(cof, budget)?;
        budget.mark(Stage::Backsub);
    }
    red.give_back(bufs);
    Ok(Some(red.made))
}

// f4-gfp.21 design -- the reduced basis comes from one more matrix, each element's tail reduced as in a round, not from division by the others one element at a time (not measured)
/// Reduces the tails of the given rows of a matrix built from `G_min`, each
/// row the pivot of its leading column and a tick of the budget: the given
/// rows reduced, in order, monic as they were, each kept by the cofactor
/// tracking in that order. Once the pivots are set the setup spends the
/// matrix's rows and columns, and each row operation spends its row's
/// length (`Reducer::cancel`); a stop where the budget's check stops the
/// call. The dense row and the work vectors come from
/// `bufs`, and go back to it where the reduction returns its result.
pub(crate) fn inter_reduce<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    basis: &Basis<F, D>,
    matrix: &Matrix,
    cof: &mut C,
    budget: &mut Budget,
    bufs: &mut Buffers<F>,
) -> Result<Vec<Reduced<F>>, Stop> {
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    for r in (0..matrix.rows.len()).map(RowId::at) {
        let c = matrix.lead(r).index();
        debug_assert!(matches!(red.pivot[c], Pivot::None), "one pivot a column");
        red.pivot[c] = Pivot::Row(r);
    }
    // matrix-stop-latency.8 design -- a reduction's setup spends the matrix's rows and columns once, after its reducer is made and any pivots it sets before its first row, not no spend for the setup (log 0056-1)
    budget.spend(matrix.rows.len() + matrix.columns())?;
    let mut out = Vec::with_capacity(matrix.given);
    for r in (0..matrix.given).map(RowId::at) {
        budget.tick()?;
        let after_lead = MonoId::at(matrix.lead(r).index() + 1);
        let mut row = red.reduce(r, after_lead, cof, budget)?;
        cof.keep();
        row.kept = KeepId::at(r.index());
        out.push(row);
    }
    red.give_back(bufs);
    Ok(out)
}
