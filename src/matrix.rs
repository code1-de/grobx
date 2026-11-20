//! The matrix of a round: its rows t·g, symbolic preprocessing, the columns,
//! the pivots, and the reduction of the rows that are no pivot.

#[cfg(test)]
use std::cell::Cell;
use std::cmp::Ordering;
use std::mem;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::cofactors::{Cofactors, NoCofactors};
use crate::columns::{Columns, NARROW_COLUMNS, Row, by_width};
use crate::divisors::Divisors;
use crate::field::{BAND, Field};
use crate::halt::Halt;
use crate::id::{Column, KeepId, MadeId, MonoId, RowId, RowSource};
use crate::module::Module;
use crate::monomial::{Format, PAYLOAD_BITS};
use crate::narrow;
use crate::order::Order;
use crate::packed::{self, Rows};
use crate::poly::{Basis, Poly};
use crate::report::{RoundMode, Stage};
use crate::sig::{Sig, Signatures};
use crate::sketch::{self, Sketch};
use crate::sort::{
    self, KEY, RUN, by_signature, keyed_decreasing, no_piece, payload, sort_decreasing,
};
use crate::table::{Limit, PLACE, Table, search_units};

// f4-gfp.20 cited-checked: faugere-1999 §3.4 -- coefficients are not duplicated, since some rows are just term multiples of others; demin-2024 §4.1 -- the arrays of coefficients are used both as polynomials and as matrix rows
/// A matrix: its rows t·g, each with its columns, and the monomial of each
/// column. A row's coefficients are its element's. The columns fill the
/// vector the build was handed: in a run's rounds the one the run's buffers
/// kept (`Buffers`), behind the rows of the matrix before where the build
/// holds them, in the inter-reduction a new one.
pub(crate) struct Matrix {
    /// The rows given, then the reducers symbolic preprocessing added.
    pub(crate) rows: Vec<RowSource>,
    /// How many rows were given.
    pub(crate) given: usize,
    // matrix-slack.5 design-checked -- a matrix's columns stay one array, not blocks of 4096 to 2^20 entries with each row in one block (log 0041-1)
    /// Every row's columns, increasing, in the column vector the build was
    /// handed, which a run's buffers keep for its next matrix
    /// (`Buffers::keep`), save where a replay keeps the matrix in its shape.
    cols: Columns,
    // rebuilt-reducer-time.4 design-checked -- the kept rows stay in the one column vector, the next matrix's rows appended behind them, each row addressed by a span (start, end), not the kept rows copied into a second column vector, nor the matrix before kept whole beside the next
    /// Row r's columns are `cols[spans[r].0..spans[r].1]`. While a build
    /// holds rows of the matrix before, the vector holds them at its front,
    /// taken or dead, and the rows built behind them. The compaction
    /// (`drop_kept`) leaves the vector tight: the rows taken at its front,
    /// in the order they landed (`front`), the rest behind in row order.
    /// Until `columns` numbers the round's columns, a taken row's entries
    /// are the matrix before's columns (sort-share-time.1) and every other
    /// row's are local columns (`Table::row_products`).
    spans: Vec<(usize, usize)>,
    /// The monomial of each column, decreasing; until `columns` numbers
    /// them, the round's monomials in the order they entered, a monomial's
    /// local column its index.
    pub(crate) monomials: Vec<MonoId>,
    // column-index-peak.5 design-checked -- the column slot's value before the round is kept per local column in `previous`, a vector of the matrix, not in a third field per monomial in the table nor the local column in a field of its own beside the slot (not measured)
    /// Per local column, the column slot its monomial held before the
    /// round, which `take_kept` and the merge read: the monomial's column in
    /// the last matrix it entered. Empty once `columns` has numbered the
    /// round's monomials.
    previous: Vec<MonoId>,
    /// The reducer rows taken from the matrix before.
    kept: usize,
    // rebuilt-reducer-time.14 design-checked -- the compaction records the taken rows in the order they landed, and the next build derives the kept matrix's rows in the order of their columns from that in one pass, not the kept rows sorted by their spans at each build, nor the order of every row stored per matrix
    /// The rows taken from the matrix before, in the order of their
    /// columns, which lie before every other row's; empty where none was
    /// taken.
    front: Vec<RowId>,
    /// The matrix before, while the build holds its rows.
    before: Option<Kept>,
    /// The monomial of each column of the matrix before, from the
    /// compaction (`drop_kept`), which drops the rest of that matrix, to
    /// the numbering (`columns`), which maps the taken rows' entries
    /// through them; empty where the build holds no matrix before.
    before_columns: Vec<MonoId>,
}

/// What a build holds of the matrix before, beside the column vector,
/// whose first `len` entries are that matrix's rows at their spans
/// (`Buffers::take_last`).
struct Kept {
    rows: Vec<RowSource>,
    spans: Vec<(usize, usize)>,
    monomials: Vec<MonoId>,
    /// The entries of the kept matrix: its rows lie in `cols[..len]`.
    len: usize,
    /// The kept matrix's rows in the order of their columns.
    order: Vec<RowId>,
    /// Per column of the kept matrix, the reducer row that leads at it, or
    /// `RowId::NONE`.
    by_col: Vec<RowId>,
    /// Per kept row, the row of the new matrix that took it, or
    /// `RowId::NONE`.
    taken: Vec<RowId>,
    /// The words past which the build drops the kept rows (`Matrix::room`).
    bound: usize,
}

impl Matrix {
    /// A matrix of the rows `given`, all of them given, with no columns yet:
    /// its rows' columns are to fill `cols`, cleared and made narrow, or,
    /// where the build holds the matrix before, `before`, holding that
    /// matrix's rows at its front at that matrix's width. A narrow build
    /// widens before a row that could take it past what its entries hold
    /// (`fit`).
    fn new(given: Vec<RowSource>, mut cols: Columns, before: Option<Kept>) -> Matrix {
        debug_assert!(
            before
                .as_ref()
                .map_or(cols.is_empty(), |k| k.len == cols.len()),
            "the column vector holds the kept matrix's entries, or none"
        );
        // column-index-peak.8 design-checked -- a build that holds the rows of the matrix before runs at that matrix's width, wide where it was wide, not its kept region narrowed at the build's start where it fits: the high-water mark a wide matrix set stays, and a narrow matrix after it lowers no peak (not measured)
        // column-index-peak.16 design-checked -- every build starts narrow where it holds no matrix before, the plain run's, the signature run's, the replays', and the inter-reduction's, not the run's rounds alone (not measured)
        // A bound of 0, the test profile's, admits no narrow column.
        if before.is_none() {
            cols.set_narrow(narrow_bound() > 0);
        }
        Matrix {
            given: given.len(),
            rows: given,
            cols,
            spans: Vec::new(),
            monomials: Vec::new(),
            previous: Vec::new(),
            kept: 0,
            front: Vec::new(),
            before,
            before_columns: Vec::new(),
        }
    }

    pub(crate) fn columns(&self) -> usize {
        self.monomials.len()
    }

    // table-read-time.6 design-checked -- a matrix's row reads its span and slices its columns, and the reduction reads a row's source, with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    /// Row r's columns, at the column vector's width. The reads have no
    /// bounds check outside the test profile, whose debug assertions check
    /// them.
    pub(crate) fn row(&self, r: RowId) -> Row<'_> {
        debug_assert!(
            r.index() < self.spans.len(),
            "row {r} past the matrix's {rows}",
            rows = self.spans.len()
        );
        // SAFETY: a row index lies below the rows, `spans` holds an entry per
        // row, and each span lies within the vector.
        unsafe {
            let (a, b) = *self.spans.get_unchecked(r.index());
            self.cols.row_unchecked(a, b)
        }
    }

    /// Gives back the slack of the growable arrays.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.rows.shrink_to_fit();
        self.cols.shrink_to_fit();
        self.spans.shrink_to_fit();
        self.monomials.shrink_to_fit();
        self.front.shrink_to_fit();
    }

    // call-stop-latency.41 design-checked -- a walk of the ids spends a unit an id after each chunk of PLACE ids of a vector, not a count per id in the caller's closure, which made the compaction's walks of ccode-cooper45t4-p65521 cost more instructions (log 0138-2)
    /// Every monomial id the built matrix holds to `f`: each row's
    /// multiplier, then the columns' monomials. `cols`, `spans`, and
    /// `front` hold columns and rows, and no monomial. A unit an id is
    /// spent at `site` after each chunk of `PLACE` rows or monomials, and
    /// a stop where the budget's check stops the call ends the walk.
    fn ids_mut(
        &mut self,
        mut f: impl FnMut(&mut MonoId),
        budget: &mut Budget,
        site: Site,
    ) -> Result<(), Stop> {
        debug_assert!(
            self.before.is_none() && self.previous.is_empty() && self.before_columns.is_empty(),
            "a built matrix"
        );
        for chunk in self.rows.chunks_mut(PLACE) {
            for r in chunk.iter_mut() {
                f(&mut r.multiplier);
            }
            budget.spend(chunk.len(), site)?;
        }
        for chunk in self.monomials.chunks_mut(PLACE) {
            chunk.iter_mut().for_each(&mut f);
            budget.spend(chunk.len(), site)?;
        }
        Ok(())
    }

    /// The column vector, the rest of the matrix dropped.
    pub(crate) fn into_cols(self) -> Columns {
        self.cols
    }

    /// Row r's leading column.
    fn lead(&self, r: RowId) -> MonoId {
        MonoId::at(self.cols.entry(self.spans[r.index()].0))
    }

    /// The column vector's capacity, in words.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.cols.capacity()
    }

    /// The entries of every row's columns, once the build has compacted
    /// the vector.
    pub(crate) fn entries(&self) -> usize {
        self.cols.len()
    }

    /// The words the column vector's entries fill.
    pub(crate) fn words(&self) -> usize {
        self.cols.words().len()
    }

    /// The bytes of an entry of the column vector, 2 or 4.
    pub(crate) fn index(&self) -> usize {
        self.cols.entry_bytes()
    }

    /// The matrix with its column vector narrow; None where a column passes
    /// `u16`. The matrix is built: it holds no matrix before.
    #[cfg(test)]
    pub(crate) fn narrowed(&self) -> Option<Matrix> {
        debug_assert!(self.before.is_none(), "a built matrix");
        Some(Matrix {
            rows: self.rows.clone(),
            given: self.given,
            cols: self.cols.narrowed()?,
            spans: self.spans.clone(),
            monomials: self.monomials.clone(),
            previous: self.previous.clone(),
            kept: self.kept,
            front: self.front.clone(),
            before: None,
            before_columns: self.before_columns.clone(),
        })
    }

    /// The reducer rows taken from the matrix before.
    pub(crate) fn kept(&self) -> usize {
        self.kept
    }

    // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
    /// The column vector, and what the next build holds of the matrix
    /// (`Kept`), its rows at their spans at the vector's front, where the
    /// matrix has a reducer row; else the vector cleared and nothing.
    /// `bound` is the words past which that build drops the rows. Spent at
    /// `Site::Kept` as it goes, `RUN` at a time: the maps a unit an entry,
    /// a reducer row's lead `budget::PROBE` units, read and mapped, and the order a
    /// unit a row. A stop where the budget's check stops the call, the
    /// matrix dropped.
    fn into_kept(self, bound: usize, budget: &mut Budget) -> Result<(Columns, Option<Kept>), Stop> {
        let Matrix {
            rows,
            given,
            mut cols,
            spans,
            monomials,
            front,
            ..
        } = self;
        if rows.len() == given {
            cols.clear();
            return Ok((cols, None));
        }
        // rebuilt-reducer-time.12 design-checked -- the kept matrix's reducer rows alone are taken, its given rows not, so the map of its columns names no given row
        let mut by_col = filled(monomials.len(), RowId::NONE, budget, Site::Kept)?;
        for (k, chunk) in spans[given..].chunks(RUN).enumerate() {
            for (j, &(start, _)) in chunk.iter().enumerate() {
                let c = MonoId::at(cols.entry(start));
                debug_assert!(
                    by_col[c.index()] == RowId::NONE,
                    "two reducer rows lead at column {c}"
                );
                by_col[c.index()] = RowId::at(given + k * RUN + j);
            }
            budget.spend(chunk.len() * crate::budget::PROBE, Site::Kept)?;
        }
        // The rows taken lie at the front, the rest follow in row order.
        let taken_end = front.last().map_or(0, |r| spans[r.index()].1);
        let mut order = front;
        for start in (0..rows.len()).step_by(RUN) {
            let end = (start + RUN).min(rows.len());
            order.extend(
                (start..end)
                    .map(RowId::at)
                    .filter(|r| spans[r.index()].0 >= taken_end),
            );
            budget.spend(end - start, Site::Kept)?;
        }
        debug_assert_eq!(
            order.len(),
            rows.len(),
            "every row in the order of its columns"
        );
        let kept = Kept {
            len: cols.len(),
            taken: filled(rows.len(), RowId::NONE, budget, Site::Kept)?,
            rows,
            spans,
            monomials,
            order,
            by_col,
            bound,
        };
        Ok((cols, Some(kept)))
    }

    // rebuilt-reducer-time.1 invented-checked -- a reducer row the matrix before holds under the same (element, multiplier) is taken from it as built, its columns that matrix's, read through its monomials, in place of its products, not every row built from the basis alone (f4-gfp.4), nor Simplify (faugere-1999 §2.4), which replaces a product by an equivalent reduced row and gave monagan-2015 §2 no speedup, nor the kept matrix decoded whole after its reduction
    // rebuilt-reducer-time.2 design-checked -- a kept row is taken only where the search names its (element, multiplier), not for its lead alone with the search skipped, so that every matrix is row for row the one the build without kept rows makes
    // rebuilt-reducer-time.3 design-checked -- the kept row is found through the table's column slot of the lead, which the kept matrix's `columns` wrote, checked against the kept matrix's monomial at that column and the reducer row leading there, not a hash map from (element, multiplier) to the row, nor a search over the kept rows
    // sort-share-time.1 design-checked -- a kept row's entries stay the matrix before's columns through the build, and `columns` maps them to the round's columns through a table over the matrix before's columns, read at indices that increase within a row, not decoded to monomial ids here and mapped through the table's column slots as every other entry, which wrote every kept entry once and gathered it again at random over the table's size
    /// Takes `row`, the row last pushed, whose lead is `u`, a monomial of
    /// the round, from the matrix before where the build holds it under the
    /// same (element, multiplier): the lead's column there is the slot it
    /// held before the round (`previous`), its columns, that matrix's,
    /// become the row's as they stand, to be mapped to the round's columns
    /// at the numbering (`columns`), and each monomial among them enters
    /// the round and is put on `work` where it was new to it, as the
    /// products would have put it (`Table::enter`); a narrow vector widens
    /// first where they could pass what its entries hold (`fit`). Then
    /// spends `PRODUCT` units an entry. Whether the row was taken; a stop
    /// where the budget's check stops the call.
    fn take_kept<M: Format>(
        &mut self,
        table: &mut Table<M>,
        u: MonoId,
        row: RowSource,
        round: u32,
        work: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        let Some(kept) = &mut self.before else {
            return Ok(false);
        };
        let column = self.previous[table.column(u).index()];
        if column.index() >= kept.monomials.len() || kept.monomials[column.index()] != u {
            return Ok(false);
        }
        let old = kept.by_col[column.index()];
        if old == RowId::NONE || kept.rows[old.index()] != row {
            return Ok(false);
        }
        let (from, to) = kept.spans[old.index()];
        debug_assert!(
            to <= kept.len,
            "the kept row's span {from} to {to} past the kept region's {len}",
            len = kept.len
        );
        fit(&mut self.cols, self.monomials.len(), to - from, budget)?;
        let start = work.len();
        by_width!(self.cols.row(from, to), |cols| for &col in cols {
            let col = col.column();
            debug_assert!(
                col < kept.monomials.len(),
                "column {col} past the kept matrix's {columns}",
                columns = kept.monomials.len()
            );
            // SAFETY: an entry of the kept region is a column of the kept
            // matrix, below its monomials' length, which its `columns`
            // asserted.
            let id = unsafe { *kept.monomials.get_unchecked(col) };
            table.enter(id, round, work, &mut self.previous);
        });
        self.monomials.extend_from_slice(&work[start..]);
        self.spans.push((from, to));
        kept.taken[old.index()] = RowId::at(self.rows.len() - 1);
        self.kept += 1;
        // rebuilt-reducer-time.7 design-checked -- a kept row ticks and spends what a built row ticks and spends, not a smaller spend for the copy, so that the budget's checks fall where they fall without kept rows
        budget.spend((to - from) * PRODUCT, Site::Products)?;
        Ok(true)
    }

    // rebuilt-reducer-time.5 design-checked -- where a row would take the vector past the most words a matrix of the run has held plus KEPT_SLACK, 2^20 words, past RESERVE_PAST while it holds no reservation, or past its capacity while it holds one, the untaken kept rows are dropped and the vector compacted before the growth, which is then decided on the compacted length, not the kept rows held whatever the length, a slack of 2^18 or 2^22 words, the growth decided on the length with the kept rows, nor the rows dropped before a doubling below the reservation, which stays below 2 MiB and turns no huge page on, as the build without kept rows grows (log 0102-1, _Step 0_, _The memory a design may hold_; log 0102-2, _Step 6_)
    /// Drops the rows of the matrix before, where the build holds them,
    /// before a row of `need` entries that would take the vector's words
    /// past its bound, past `RESERVE_PAST` while it holds no reservation,
    /// or past its capacity while it holds one, spent as `drop_kept`
    /// spends; a stop where the budget's check stops the call.
    fn room(&mut self, need: usize, budget: &mut Budget) -> Result<(), Stop> {
        let Some(kept) = &self.before else {
            return Ok(());
        };
        let (len, capacity) = (
            self.cols.words_for(self.cols.len() + need),
            self.cols.capacity(),
        );
        if len > kept.bound
            || (capacity < RESERVATION && len > RESERVE_PAST)
            || (capacity >= RESERVATION && len > capacity)
        {
            self.drop_kept(budget)?;
        }
        Ok(())
    }

    // rebuilt-reducer-time.6 design-checked -- the untaken kept rows are dropped and the vector compacted at the end of every build, the taken rows slid down in the order of their columns and the new rows behind them, not the dead rows left as holes from round to round, nor the rows compacted into row order, which needs a permutation of segments of unequal length
    /// Drops the rows of the matrix before and compacts the vector: the
    /// taken rows slide down in the order of their columns, a run of
    /// adjacent ones as one copy, the rows built behind the kept region
    /// follow, and the dead entries go. The matrix before's monomials stay
    /// as `before_columns`, for the numbering. Nothing where the build
    /// holds none. Spent at `Site::Kept` as it goes: a unit a kept row and
    /// a span, `RUN` at a time, and the entries slid as `slide` spends. A
    /// stop where the budget's check stops the call, the matrix then part
    /// compacted, which the stopped build drops.
    fn drop_kept(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let Some(kept) = self.before.take() else {
            return Ok(());
        };
        debug_assert!(self.front.is_empty(), "the compaction runs once a build");
        let mut write = 0;
        // The adjacent taken rows not yet slid.
        let mut run: Option<(usize, usize)> = None;
        for chunk in kept.order.chunks(RUN) {
            for &r in chunk {
                let new = kept.taken[r.index()];
                if new == RowId::NONE {
                    continue;
                }
                let (a, b) = kept.spans[r.index()];
                debug_assert!(
                    run.is_none_or(|(_, e)| e <= a),
                    "kept row {r} out of the order of its columns"
                );
                let start = match run {
                    Some((s, e)) if e == a => s,
                    Some((s, e)) => {
                        write = self.slide(s, e, write, budget)?;
                        a
                    }
                    None => a,
                };
                run = Some((start, b));
                self.spans[new.index()] = (write + a - start, write + b - start);
                self.front.push(new);
            }
            budget.spend(chunk.len(), Site::Kept)?;
        }
        if let Some((s, e)) = run {
            write = self.slide(s, e, write, budget)?;
        }
        let dead = kept.len - write;
        if dead > 0 {
            self.slide(kept.len, self.cols.len(), write, budget)?;
            for chunk in self.spans.chunks_mut(RUN) {
                for span in chunk.iter_mut() {
                    if span.0 >= kept.len {
                        span.0 -= dead;
                        span.1 -= dead;
                    }
                }
                budget.spend(chunk.len(), Site::Kept)?;
            }
            self.cols.truncate(self.cols.len() - dead);
        }
        debug_assert!(
            self.spans
                .iter()
                .all(|&(a, b)| a <= b && b <= self.cols.len()),
            "a span past the vector"
        );
        debug_assert_eq!(
            self.spans.iter().map(|&(a, b)| b - a).sum::<usize>(),
            self.cols.len(),
            "the spans' lengths and the vector's"
        );
        self.before_columns = kept.monomials;
        Ok(())
    }

    /// Slides `cols[s..e]` down to `write`, at or below s: the end of the
    /// slid run. The entries move `RUN` at a time from the first on, each
    /// piece's place below its source and below every later piece's, a
    /// unit an entry spent at `Site::Kept` after each piece; a stop where
    /// the budget's check stops the call, the run then part slid.
    fn slide(
        &mut self,
        s: usize,
        e: usize,
        write: usize,
        budget: &mut Budget,
    ) -> Result<usize, Stop> {
        debug_assert!(write <= s, "a slide up, from {s} to {write}");
        if write < s {
            for from in (s..e).step_by(RUN) {
                let to = (from + RUN).min(e);
                self.cols.copy_within(from..to, write + (from - s));
                budget.spend(to - from, Site::Kept)?;
            }
        }
        Ok(write + (e - s))
    }
}

// column-index-peak.10 design-checked -- the bounds RESERVE_PAST, RESERVATION, and KEPT_SLACK, and the most a matrix of the run has held, count the column vector's words of 4 bytes, not its entries of either width, so that a narrow vector reserves and drops its kept rows at the bytes a wide one does (not measured)
// matrix-slack.2 tuned-checked -- a matrix's columns take a reservation once they would pass 2^18 words, not before every matrix's first row, nor past 2^16 or 2^20 words (log 0041-1)
/// The words past which a matrix's column vector takes a reservation:
/// 2^18, 1 MiB.
pub(crate) const RESERVE_PAST: usize = 1 << 18;

// matrix-slack.3 design-checked -- a reservation holds 2^23 words, 32 MiB, which glibc's malloc maps outside its heap where its free memory cannot serve it, whatever its dynamic threshold, not a smaller one it may serve from its heap (not measured)
/// The words a reservation holds: 2^23, 32 MiB.
pub(crate) const RESERVATION: usize = 1 << 23;

/// The words the column vector may hold past the most a matrix of the run
/// has held while it holds rows of the matrix before: 2^20, 4 MiB
/// (rebuilt-reducer-time.5, `Matrix::room`).
pub(crate) const KEPT_SLACK: usize = 1 << 20;

#[cfg(test)]
thread_local! {
    // rebuilt-reducer-time.13 design-checked -- in the test profile a thread-local bound stands in for KEPT_SLACK and another turns the keeping off, so that the test code compares a call with kept rows against the same call without, not a parameter of the call nor the slack alone lowered
    /// The test profile's slack on this thread, where Some; None leaves
    /// `KEPT_SLACK`.
    pub(crate) static KEPT_BOUND: Cell<Option<usize>> = const { Cell::new(None) };
    /// Whether the builds on this thread take no row of the matrix before:
    /// `Buffers::take_last` then gives the column vector alone.
    pub(crate) static KEPT_OFF: Cell<bool> = const { Cell::new(false) };
    // short-row-reduce-time.7 design-checked -- in the test profile a thread-local forced mode stands in for the probe's decision in every no-band round of BLOCK + PROBE rows to reduce or more, not a parameter of the call nor the probe's constants lowered
    /// The test profile's mode on this thread, where Some: what every
    /// round of `BLOCK + PROBE` rows to reduce or more whose matrix takes
    /// no bands reduces the rows after its first block in, in place of the
    /// run's decision (`round_mode`).
    pub(crate) static MODE_FORCED: Cell<Option<RoundMode>> = const { Cell::new(None) };
    // column-index-peak.15 design-checked -- in the test profile a thread-local bound stands in for NARROW_COLUMNS, Some(0) building every matrix wide, so that the test code compares a call with narrow matrices against the same call wide and widens builds at small rounds, not a parameter of the call nor the constant lowered (not measured)
    /// The test profile's bound on this thread, where Some: the round's
    /// monomials a narrow build may hold, in place of `NARROW_COLUMNS`.
    pub(crate) static NARROW_BOUND: Cell<Option<usize>> = const { Cell::new(None) };
    // gf2-forced-zeros-time.15 design-checked -- in the test profile a thread-local stands in for the rule on the rows' entries, and another for the echelon's cap, so that the test code takes the sketches on small systems and fills the echelon, not parameters of the call nor the constants lowered (not measured)
    /// The test profile's choice on this thread, where Some: whether a
    /// plain round over a field that packs, whose run may combine rows,
    /// takes the sketches, in place of `SKETCH_ROW_ENTRIES` (`sketches`).
    pub(crate) static SKETCH_FORCED: Cell<Option<bool>> = const { Cell::new(None) };
    /// The test profile's cap on this thread, where Some: the residues the
    /// echelon holds before the sketch is drawn again, in place of
    /// `SKETCH_ECHELON`.
    pub(crate) static ECHELON_FORCED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The slack past the most words a matrix of the run has held:
/// `KEPT_SLACK`, or in the test profile what `KEPT_BOUND` holds.
fn kept_slack() -> usize {
    #[cfg(test)]
    if let Some(bound) = KEPT_BOUND.get() {
        return bound;
    }
    KEPT_SLACK
}

/// The round's monomials a narrow build may hold: `NARROW_COLUMNS`, or in
/// the test profile what `NARROW_BOUND` holds.
fn narrow_bound() -> usize {
    #[cfg(test)]
    if let Some(bound) = NARROW_BOUND.get() {
        return bound;
    }
    NARROW_COLUMNS
}

// column-index-peak.7 design-checked -- the bound is checked before each row, the round's monomials plus the row's terms against it, not at each product new to the round, which would stand in the products' loop and stop and resume it mid-row (not measured)
/// Widens `cols`, a build's column vector, where it is narrow and the
/// round's `columns` monomials and `terms` more, the most a row of `terms`
/// entries enters, would pass the monomials a narrow build may hold
/// (`narrow_bound`): every column the numbering gives a narrow matrix then
/// lies below them. The widening spends a unit an entry at
/// `Site::Products` after each `RUN` entries; a stop where the budget's
/// check stops the call.
fn fit(cols: &mut Columns, columns: usize, terms: usize, budget: &mut Budget) -> Result<(), Stop> {
    if cols.is_narrow() && columns + terms > narrow_bound() {
        cols.widen(RUN, |entries| budget.spend(entries, Site::Products))?;
    }
    Ok(())
}

/// `n` copies of `value`, filled `RUN` at a time, a unit each spent at
/// `site` after each chunk; a stop where the budget's check stops the call.
pub(crate) fn filled<T: Copy>(
    n: usize,
    value: T,
    budget: &mut Budget,
    site: Site,
) -> Result<Vec<T>, Stop> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let filled = RUN.min(n - v.len());
        v.resize(v.len() + filled, value);
        budget.spend(filled, site)?;
    }
    Ok(v)
}

/// Makes room for `need` more columns: where `cols` would pass
/// `RESERVE_PAST` words and holds less than `RESERVATION` words of
/// capacity, reserves `RESERVATION` words, or the words of `cols.len() +
/// need` entries where those are more. Else it leaves `cols` as it is, and
/// `Table::row_products`, which reserves room for each row, grows it by
/// doubling.
pub(crate) fn make_room(cols: &mut Columns, need: usize) {
    let words = cols.words_for(cols.len() + need);
    if cols.capacity() < RESERVATION && words > RESERVE_PAST {
        cols.reserve_exact(RESERVATION.max(words) - cols.words().len());
    }
}

// matrix-stop-latency.5 design-checked -- symbolic preprocessing spends 16 units a product, every row it multiplies, the given rows included, not ticks per reducer row alone nor 8 units a product (log 0055-2, 0056-1)
/// Units of work per product of symbolic preprocessing.
pub(crate) const PRODUCT: usize = 16;

// call-stop-latency.72 design-checked -- a reducer the search finds spends FORMED, 3 · PROBE, units for its multiplier's quotient, its signature's product, and their comparison with the largest, before the tests that may reject it, not the tick of the row it adds alone, which a reducer rejected never reaches (log 0138-8)
/// Units of work of a reducer the signature run's build finds: its
/// multiplier's quotient, its signature's product, and their comparison
/// with the largest reductant signature, each a random access into the
/// table.
const FORMED: usize = 3 * crate::budget::PROBE;

// round-memory-time.4 cited-checked: martani-2012 §3.9, attributes it to Lachartre 2008 [1] -- the original implementation reserves a big memory chunk at the start of the program and performs all the computations inside it, which eliminates the dynamic allocations and deallocations; monagan-2007 §2.2, independently -- the merge buffers are reused in the next call rather than freed each time
// round-memory-time.6 cited-checked: monagan-2007 §2.2 -- to further amortize the cost of memory allocation, the merge buffers p and p' are reused in the next call to an algorithm rather than freed each time
// round-memory-time.8 design-checked -- the buffers are an object the run owns and passes to its builds and reductions, not pools of the thread (log 0051-1)
// round-memory-time.12 cited-checked: monagan-2007 §2.2 -- the merge buffers p and p' grow dynamically and are reused in the next call to an algorithm rather than freed each time
/// The buffers the matrices of one run take in turn: the run's last
/// matrix, whose column vector the next matrix's rows fill and whose
/// reducer rows the next build may take, the dense row of the reductions,
/// the two work vectors a reduction builds a row in, and the band of the
/// reductions that take bands with its rows' work vectors. Each holds what
/// the matrix before left, or nothing.
pub(crate) struct Buffers<F: Field> {
    // rebuilt-reducer-time.8 design-checked -- the buffers keep the last matrix whole, its rows, spans, and monomials beside its column vector, for the next build, dropped when that build ends, not the column vector alone (round-memory-time.10, which gives way)
    /// The run's last matrix, whole, for the next build (`take_last`);
    /// None where the run has no matrix yet, or a build took it.
    last: Option<Matrix>,
    /// The most words a matrix of the run has held.
    high: usize,
    /// The dense row of the last reduction.
    dense: Vec<F::Acc>,
    /// A row's columns and coefficients while it is built.
    row_cols: Vec<MonoId>,
    row_coefs: Vec<F::Coef>,
    // row-traffic-time.10 cited-checked: monagan-2007 §2.2 -- to further amortize the cost of memory allocation the merge buffers are reused in the next call rather than freed each time
    /// The band's dense rows of the last reduction that took a band.
    band: Vec<[F::Acc; BAND]>,
    // row-traffic-time.11 design-checked -- a band row's result is built in a pair of work vectors of its own, BAND pairs from the buffers, and copied at its length, not a band row's vectors made empty and grown by push (log 0060-2)
    /// The band rows' columns and coefficients while they are built, a
    /// pair per band row.
    band_cols: [Vec<MonoId>; BAND],
    band_coefs: [Vec<F::Coef>; BAND],
    /// The scan of the reducers made from the buffers (`Reducer::new`).
    #[cfg(test)]
    pub(crate) scan: Scan,
    /// Whether the back-substitution and the inter-reduction of the
    /// reducers made from the buffers may take bands
    /// (`Reducer::bands_in_echelon`).
    #[cfg(test)]
    pub(crate) bands: bool,
}

impl<F: Field> Buffers<F> {
    /// Buffers that hold nothing.
    pub(crate) fn new() -> Buffers<F> {
        Buffers {
            last: None,
            high: 0,
            dense: Vec::new(),
            row_cols: Vec::new(),
            row_coefs: Vec::new(),
            band: Vec::new(),
            band_cols: Default::default(),
            band_coefs: Default::default(),
            #[cfg(test)]
            scan: Scan::Chosen,
            #[cfg(test)]
            bands: true,
        }
    }

    /// The column vector for the next matrix: empty, its capacity the one
    /// of the vector kept last, or none. The rest of the last matrix drops.
    pub(crate) fn take_cols(&mut self) -> Columns {
        let mut cols = self
            .last
            .take()
            .map_or_else(Columns::new, Matrix::into_cols);
        cols.clear();
        cols
    }

    /// The column vector for the next matrix with the last matrix's rows at
    /// its front, and what the build holds of that matrix (`Kept`); the
    /// vector cleared and nothing where no matrix is kept, where the kept
    /// matrix has no reducer row, or, in the test profile, under
    /// `KEPT_OFF`. Spent as `Matrix::into_kept` spends; a stop where the
    /// budget's check stops the call.
    fn take_last(&mut self, budget: &mut Budget) -> Result<(Columns, Option<Kept>), Stop> {
        #[cfg(test)]
        if KEPT_OFF.get() {
            return Ok((self.take_cols(), None));
        }
        let bound = self.high + kept_slack();
        match self.last.take() {
            Some(matrix) => matrix.into_kept(bound, budget),
            None => Ok((Columns::new(), None)),
        }
    }

    /// Every monomial id the kept matrix holds to `f` (`Matrix::ids_mut`),
    /// nothing where none is kept, spent at `site` as `Matrix::ids_mut`
    /// spends. The other buffers are scratch of a build or a reduction and
    /// hold no monomial between two.
    pub(crate) fn ids_mut(
        &mut self,
        f: impl FnMut(&mut MonoId),
        budget: &mut Budget,
        site: Site,
    ) -> Result<(), Stop> {
        match &mut self.last {
            Some(m) => m.ids_mut(f, budget, site),
            None => Ok(()),
        }
    }

    /// Keeps `matrix`, the run's last, for the next matrix, in place of the
    /// one held.
    pub(crate) fn keep(&mut self, matrix: Matrix) {
        self.high = self.high.max(matrix.words());
        self.last = Some(matrix);
    }

    // linear-tail-time.7 design-checked -- the kept matrix is dropped where the plain run inter-reduces G_min between two rounds, not kept, as the inter-reduction's build writes the table's column slots, through which the next build finds a kept row (rebuilt-reducer-time.3), and a changed member's kept rows no longer match it; kept, the first probe and the plan's first code panicked at the columns' check (log 0166-1, 0166-2)
    /// Drops the kept matrix, whose column slots a build between two rounds
    /// overwrites (`f4::tails`): the next build takes no row of a matrix
    /// before.
    pub(crate) fn forget(&mut self) {
        self.last = None;
    }

    /// Buffers that hold the column vector `cols` alone, cleared, as a run's
    /// hold after a matrix with no reducer row. The next build sets its
    /// width (`Matrix::new`).
    #[cfg(test)]
    pub(crate) fn of_cols(mut cols: Columns) -> Buffers<F> {
        cols.clear();
        let mut bufs = Buffers::new();
        bufs.last = Some(Matrix::new(Vec::new(), cols, None));
        bufs
    }

    /// Gives back every buffer: each holds nothing and no capacity.
    pub(crate) fn release(&mut self) {
        *self = Buffers::new();
    }

    /// The capacities of the column vector, in words, the dense row, and
    /// the two work vectors, in that order.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> [usize; 4] {
        [
            self.last.as_ref().map_or(0, Matrix::capacity),
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

// order-family.12 cited-checked: giovini-1991 §2 -- simplifying only where it does not increase the sugar was found often equivalent to the original strategy, and sometimes worse; bigatti-2025 §3.2, Remark 58 -- symPre ends, since every term it adds is strictly lower than the one it reduces, in any term order
/// Builds the matrix of the rows `given` in the table's next round
/// (`Table::open`): symbolic
/// preprocessing adds a reducer row for every monomial of the matrix that a
/// leading monomial of `G_min` divides, save the given rows' leading
/// monomials, whatever the sugar of the reducer. Each reducer row ticks the budget. Overflow where a product
/// passes the format. Every row spends its products, a given row's lead
/// `PRODUCT` units, each divisor search its masks' blocks and the
/// candidates it tests (`Divisors::find`), and the columns their
/// sort and mapping (`multiply`, `columns`); a stop where the budget's
/// check stops the call. The rows' columns fill the column vector of
/// `bufs`, behind the rows of the run's last matrix where `bufs` keeps it
/// (`Buffers::take_last`): a reducer row that matrix holds is taken from it
/// as built (`Matrix::take_kept`), the rest of it is dropped before the
/// columns are numbered (`Matrix::drop_kept`), and the numbering merges
/// its columns with the new ones and maps the taken rows through it
/// (`columns`).
pub(crate) fn build<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    given: Vec<RowSource>,
    bufs: &mut Buffers<F>,
    budget: &mut Budget,
) -> Result<Matrix, Halt> {
    build_searched(table, basis, order, given, bufs, budget, |_| true)
}

/// As `build`, where symbolic preprocessing searches a divisor for the
/// monomials `searched` holds alone: the caller knows that no leading
/// monomial of `G_min` divides a monomial `searched` refuses
/// (gf2-tails-time.5).
pub(crate) fn build_searched<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    given: Vec<RowSource>,
    bufs: &mut Buffers<F>,
    budget: &mut Budget,
    searched: impl Fn(MonoId) -> bool,
) -> Result<Matrix, Halt> {
    let round = table.open()?;
    let (cols, before) = bufs.take_last(budget)?;
    let mut m = Matrix::new(given, cols, before);
    // The given rows' leading monomials are done from the start: they
    // enter the round, and no divisor search takes them.
    for start in (0..m.given).step_by(RUN) {
        let end = (start + RUN).min(m.given);
        for r in start..end {
            let row = m.rows[r];
            let lead = table.mul(row.multiplier, basis.lead(row.element), budget)?;
            table.enter(lead, round, &mut m.monomials, &mut m.previous);
        }
        budget.spend((end - start) * PRODUCT, Site::Products)?;
    }
    // Monomials new to the round wait in `work` for their divisor search.
    let mut work: Vec<MonoId> = Vec::new();
    for r in 0..m.given {
        let row = m.rows[r];
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
    }
    let divisors = Divisors::new(table, basis, budget)?;
    while let Some(u) = work.pop() {
        if !searched(u) {
            continue;
        }
        // call-stop-latency.14 design-checked -- the divisor search of symbolic preprocessing spends a unit per block of G_min's masks a search may test, and the signature run's reducer search per block of the basis's, a search that finds none as one that finds one, not unspent (matrix-stop-latency.9), where stretches of searches that found none went 0.38 s between two checks on PHP(8,7) (log 0138-1)
        // call-stop-latency.71 design-checked -- a search by find_passing spends PROBE units per candidate its test ran on beside its blocks of masks, the candidates counted in the test's closure, not the blocks alone (.14), as on few variables most masks pass and each candidate's test reads its words at a random place (log 0138-8)
        let (found, units) = divisors.find(table, u);
        budget.spend(units, Site::Divisors)?;
        if let Some(g) = found {
            budget.tick(Site::Products)?;
            let row = RowSource {
                element: g,
                multiplier: table.div(u, basis.lead(g), budget)?,
            };
            m.rows.push(row);
            if !m.take_kept(table, u, row, round, &mut work, budget)? {
                multiply(table, basis, row, round, &mut m, &mut work, budget)?;
            }
        }
    }
    m.drop_kept(budget)?;
    columns(table, order, &mut m, budget)?;
    Ok(m)
}

// signature-pruning.8 cited-checked: albrecht-2010 §5 -- F5/Ars adds both components of an S-polynomial to the polynomials scheduled for reduction, while F4/5 adds only the component with the bigger signature and relies on Symbolic Preprocessing to find a reductor, which may have had more reductions applied to it already
/// Builds the matrix of the table's next round (`Table::open`) from its
/// reductants `given`, each with its signature: symbolic preprocessing adds, for every monomial of
/// the matrix, the reductants' leading monomials among them, the reducer
/// of least signature where that signature lies below the largest
/// reductant signature. Each reducer row ticks the budget, every row spends
/// its products, each reducer search the blocks of the basis's masks and
/// the candidates it tests (`table::search_units`), each reducer found
/// `FORMED` units, and the columns their sort and mapping
/// (`multiply`, `columns`). The matrix, and each row's signature. Overflow where a
/// product passes the format, a stop where the budget's check stops the
/// call. The rows' columns fill the column vector of `bufs`, behind the
/// rows of the run's last matrix where `bufs` keeps it, as in `build`: a
/// reducer row that matrix holds is taken from it as built
/// (`Matrix::take_kept`), and the rest of it is dropped before the
/// columns are numbered (`Matrix::drop_kept`).
// rebuilt-reducer-time.10 design-checked -- the signature run's build takes a reducer row from the matrix before by the plain run's rule, the row's signature formed and tested before the row is pushed as without kept rows, not the plain run's build alone (log 0102-1, _Step 0_, _The signature run's repeated rows_)
#[allow(
    clippy::too_many_arguments,
    reason = "the table, basis, signatures, and order the rows come from, the rows and the run's buffers, and the budget, none of which belongs with another"
)]
pub(crate) fn build_sig<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    sigs: &Signatures,
    order: &O,
    given: Vec<(RowSource, Sig)>,
    bufs: &mut Buffers<F>,
    budget: &mut Budget,
) -> Result<(Matrix, Vec<Sig>), Halt> {
    #[allow(clippy::expect_used, reason = "the only caller skips an empty given")]
    let largest = given
        .iter()
        .map(|&(_, s)| s)
        .max_by(|a, b| a.cmp(*b, table, order))
        .expect("a matrix has a reductant");
    let (rows, mut row_sigs): (Vec<RowSource>, Vec<Sig>) = given.into_iter().unzip();
    let round = table.open()?;
    let (cols, before) = bufs.take_last(budget)?;
    let mut m = Matrix::new(rows, cols, before);
    // Every monomial new to the round waits in `work` for its reducer, the
    // reductants' leading monomials among them.
    let mut work: Vec<MonoId> = Vec::new();
    for r in 0..m.given {
        let row = m.rows[r];
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
    }
    while let Some(u) = work.pop() {
        // call-stop-latency.14 design-checked -- the divisor search of symbolic preprocessing spends a unit per block of G_min's masks a search may test, and the signature run's reducer search per block of the basis's, a search that finds none as one that finds one, not unspent (matrix-stop-latency.9), where stretches of searches that found none went 0.38 s between two checks on PHP(8,7) (log 0138-1)
        // call-stop-latency.71 design-checked -- a search by find_passing spends PROBE units per candidate its test ran on beside its blocks of masks, the candidates counted in the test's closure, not the blocks alone (.14), as on few variables most masks pass and each candidate's test reads its words at a random place (log 0138-8)
        let (found, tested) = sigs.reducer(table, u);
        budget.spend(search_units(sigs.len(), tested), Site::Divisors)?;
        let Some(g) = found else {
            continue;
        };
        // call-stop-latency.72 design-checked -- a reducer the search finds spends FORMED, 3 · PROBE, units for its multiplier's quotient, its signature's product, and their comparison with the largest, before the tests that may reject it, not the tick of the row it adds alone, which a reducer rejected never reaches (log 0138-8)
        budget.spend(FORMED, Site::Divisors)?;
        let t = table.div(u, basis.lead(g), budget)?;
        // signature-pruning.11 design-checked -- symbolic preprocessing adds no reducer whose signature reaches the largest reductant signature of the matrix (not measured)
        // A signature the format cannot hold lies above the largest, which
        // it holds.
        let s = match sigs.sig(g).mul(t, table, budget) {
            Ok(s) => s,
            Err(Limit::Overflow) => continue,
            Err(Limit::Size) => return Err(Halt::Size),
            Err(Limit::Stop(stop)) => return Err(Halt::Stop(stop)),
        };
        if s.cmp(largest, table, order).is_ge() {
            continue;
        }
        budget.tick(Site::Products)?;
        let row = RowSource {
            element: g,
            multiplier: t,
        };
        m.rows.push(row);
        row_sigs.push(s);
        if !m.take_kept(table, u, row, round, &mut work, budget)? {
            multiply(table, basis, row, round, &mut m, &mut work, budget)?;
        }
    }
    m.drop_kept(budget)?;
    columns(table, order, &mut m, budget)?;
    Ok((m, row_sigs))
}

// trace-replay.1 cited-checked: berthomieu-2021 §4.3, via berthomieu-2022a §3.4 (3) -- the matrices are generated from the trace's polynomials, with no need of handling critical pairs or symbolic preprocessing
/// Builds the matrix of the table's next round (`Table::open`) from the
/// rows `rows` alone: their
/// monomials as columns, and no symbolic preprocessing. Each row ticks the
/// budget and spends its products, and the columns spend their sort and
/// mapping (`multiply`, `columns`). Overflow where a product passes the
/// format, a stop where the budget's check stops the call. The rows'
/// columns fill the column vector of `bufs`, cleared first
/// (`Buffers::take_cols`).
// rebuilt-reducer-time.11 design-checked -- the replays' builds and the inter-reduction's take no row of the matrix before, not the replays' rows kept too
pub(crate) fn build_rows<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    rows: Vec<RowSource>,
    bufs: &mut Buffers<F>,
    budget: &mut Budget,
) -> Result<Matrix, Halt> {
    let round = table.open()?;
    let mut m = Matrix::new(rows, bufs.take_cols(), None);
    // No divisor search takes the monomials new to the round.
    let mut work: Vec<MonoId> = Vec::new();
    for r in 0..m.given {
        budget.tick(Site::Products)?;
        let row = m.rows[r];
        multiply(table, basis, row, round, &mut m, &mut work, budget)?;
        work.clear();
    }
    columns(table, order, &mut m, budget)?;
    Ok(m)
}

/// Numbers the columns: the round's monomials in decreasing order. The
/// rows' local columns become their columns, each below the matrix's
/// width, or the call panics. Where the build held the matrix before
/// (`before_columns`), in a keyed order its columns among the round's
/// monomials come in its order and the others sorted, merged
/// (`merge_columns`), and the taken rows' entries, that matrix's columns,
/// map through the merge's table; in an order without keys every monomial
/// is sorted and a taken row's entry maps through the matrix before's
/// monomial at it. The sort spends what `sort_decreasing` spends, the
/// merge what `merge_columns` spends, the permutation `2 · budget::PROBE` units a
/// monomial, its slot read and written and its local column's written,
/// and the mapping a unit an entry, each after every `RUN`; a stop where
/// the budget's check stops the call.
/// Size where the rows are more than a u32 id numbers: every row of the
/// matrix is in, and a row's index is narrowed from here on. The call
/// report's step ends its stage `Symbolic` as the numbering starts and
/// `Encode` as it ends.
fn columns<M: Format, O: Order<M>>(
    table: &mut Table<M>,
    order: &O,
    m: &mut Matrix,
    budget: &mut Budget,
) -> Result<(), Halt> {
    narrow::id(m.rows.len())?;
    budget.mark(Stage::Symbolic);
    let before = mem::take(&mut m.before_columns);
    let previous = mem::take(&mut m.previous);
    debug_assert_eq!(
        previous.len(),
        m.monomials.len(),
        "a column slot before per round monomial"
    );
    // The taken rows' entries, at the vector's front (sort-share-time.1).
    let taken_end = m.front.last().map_or(0, |r| m.spans[r.index()].1);
    debug_assert!(
        taken_end <= m.cols.len() && (!before.is_empty() || taken_end == 0),
        "taken rows to {taken_end} of {len} entries, the matrix before's {width} columns held",
        len = m.cols.len(),
        width = before.len()
    );
    // sort-share-time.7 design-checked -- the merge runs in keyed orders, and in an order without keys the columns are sorted as without the matrix before, not a merge by the order's comparison in every order, which would take up the unkeyed sort's runs and the budget's checks in them (matrix-stop-latency.6)
    let old_to_new = if O::KEYED && !before.is_empty() {
        Some(merge_columns(
            table,
            order,
            &mut m.monomials,
            &previous,
            &before,
            budget,
        )?)
    } else {
        sort_decreasing(table, order, &mut m.monomials, budget, Site::Columns)?;
        None
    };
    // column-index-peak.3 design-checked -- the build gives each monomial new to the round its local column, the count of the round's monomials before it in the order met, and writes that for every product, which the numbering maps to the sorted columns through a permutation, not the monomial id written and mapped through the table's column slot (monagan-2015 §2.1), the columns narrowed after the numbering (session 0113-1), nor the rows encoded after symbolic preprocessing by a second multiplication (pearce-2026 §2.1) (not measured)
    // column-index-peak.17 design-checked -- the permutation from local to sorted columns is written into the vector of the column slots before, which the merge has read, not into a vector of its own, 4 bytes a column more at the numbering (not measured)
    // Per local column, read from the monomial's slot, its column.
    // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
    let mut perm = previous;
    for (k, chunk) in m.monomials.chunks(RUN).enumerate() {
        for (j, &u) in chunk.iter().enumerate() {
            let c = MonoId::at(k * RUN + j);
            perm[table.column(u).index()] = c;
            table.set_column(u, c);
        }
        budget.spend(chunk.len() * 2 * crate::budget::PROBE, Site::Columns)?;
    }
    let table: &Table<M> = table;
    let most = by_width!(mut m.cols.entries_mut(), |cols| {
        map_rows(cols, taken_end, old_to_new.as_deref(), &before, &perm, table, budget)
    })?;
    // reduction-time.4 design-checked -- every column is asserted below the matrix's width once per matrix, where it is built, not trusted by construction alone; the row operations write the dense row unchecked (not measured)
    // It holds: an id joined `m.monomials` where it was new to the round, and
    // the table opens each matrix's round (`Table::open`). A built row's
    // entry is a local column, which `perm` maps below the width. A taken
    // row's entry is a column of the matrix before that the
    // round's monomials hold, which the merge numbered; one the merge did
    // not maps to `MonoId::NONE`, past every width. A narrow build widens
    // before a row that could take the round's monomials past what its
    // entries hold (`fit`).
    assert!(
        m.cols.is_empty() || (most < m.monomials.len() && m.cols.holds(most)),
        "column {most} past the matrix's {len} columns or its {bytes}-byte entries",
        len = m.monomials.len(),
        bytes = m.cols.entry_bytes()
    );
    budget.mark(Stage::Encode);
    Ok(())
}

/// Maps the entries of `cols`, a matrix's column vector at its width, to
/// the round's columns (`columns`): the taken rows' entries, the first
/// `taken_end`, columns of the matrix before, through `old_to_new` where
/// the merge numbered them and else through the column of the matrix
/// before's monomial at each, `before` (sort-share-time.1); the built
/// rows' entries, local columns, through `perm`. The largest column
/// written, as `map_columns` gives it; a stop where the budget's check
/// stops the call.
fn map_rows<I: Column, M: Format>(
    cols: &mut [I],
    taken_end: usize,
    old_to_new: Option<&[MonoId]>,
    before: &[MonoId],
    perm: &[MonoId],
    table: &Table<M>,
    budget: &mut Budget,
) -> Result<usize, Stop> {
    let (taken, built) = cols.split_at_mut(taken_end);
    let most = match old_to_new {
        Some(old_to_new) => map_columns(
            taken,
            |c| {
                debug_assert!(
                    c < old_to_new.len(),
                    "column {c} past the matrix before's {width}",
                    width = old_to_new.len()
                );
                // SAFETY: a taken row's entry is a column of the matrix
                // before, below its width, which that matrix's `columns`
                // asserted, and `old_to_new` holds an entry per column of it.
                let new = unsafe { *old_to_new.get_unchecked(c) };
                debug_assert!(
                    new != MonoId::NONE,
                    "column {c} of the matrix before, an entry of a taken row, is no column of the round"
                );
                new.index()
            },
            budget,
        )?,
        None => map_columns(taken, |c| table.column(before[c]).index(), budget)?,
    };
    // column-index-peak.18 design-checked -- a built row's local column is read from the permutation unchecked, not with a bounds check per entry, which slowed the numbering markedly, nor read by `get` with a column past it mapped to `MonoId::NONE`, no faster (log 0115-3)
    Ok(most.max(map_columns(
        built,
        |c| {
            debug_assert!(
                c < perm.len(),
                "local column {c} past the round's {len} monomials",
                len = perm.len()
            );
            // SAFETY: a built row's entry is the column slot of a monomial
            // that bears the round's stamp, which only its entry into this
            // build gave it, as the table opened the round for this build
            // (`Table::open`): its local column, below the round's
            // monomials, the length of `perm`.
            unsafe { perm.get_unchecked(c) }.index()
        },
        budget,
    )?))
}

// column-index-peak.21 design-checked -- the numbering's map keeps the largest column it gives before the column is narrowed to the entry's width, not the largest entry written, which a narrow entry would cut below the width's check (not measured)
/// Maps every entry of `cols` through `column`, which takes and gives a
/// column as an index, and spends a unit an entry after every `RUN`
/// entries: the largest column `column` gave, 0 where `cols` is empty. A
/// stop where the budget's check stops the call. Not inlined, so that the
/// pass compiles the same whatever its caller holds: inlined into
/// `columns`, one build of it took a fifth more instructions an entry than
/// another of the same source (log 0105-3, _Step 6_).
#[inline(never)]
fn map_columns<I: Column>(
    cols: &mut [I],
    column: impl Fn(usize) -> usize,
    budget: &mut Budget,
) -> Result<usize, Stop> {
    let mut most = 0;
    for chunk in cols.chunks_mut(RUN) {
        for c in chunk.iter_mut() {
            let new = column(c.column());
            most = most.max(new);
            *c = I::of(new);
        }
        budget.spend(chunk.len(), Site::Columns)?;
    }
    Ok(most)
}

// sort-share-time.2 practice-checked: merge of sorted sequences -- the round's columns that were the matrix before's are taken in its order, which is decreasing, and merged by key with the new ones sorted, not every column of the round sorted (matrix-sort-time.2): a merge is linear where the sort is n log n, and the old columns are most of a round's on the instances of many same-degree rounds
// sort-share-time.3 practice-checked: bitmap sort -- the old columns present are found through the table's column slot of each round monomial, checked against the matrix before's monomial at it, marked in a bit set over the matrix before's columns, and read from the bit set in order, not a stamp per column of the matrix before nor the old columns' indices collected and sorted: the slot check is `take_kept`'s and costs one gather a round monomial, and the bit set's scan is sequential, a word per 64 columns
// sort-share-time.4 design-checked -- the old columns' keys for the merge are built again from their monomials, in their order, not kept from the matrix before's sort, 16 bytes a column beside the matrix before, nor the merge by the order's comparison, which reads whole monomials at every step
/// Numbers the round's monomials `monomials` in decreasing order where the
/// build held the matrix before, whose columns' monomials are `before`,
/// decreasing: the monomials that were its columns, found through the
/// column slot each held before the round, `previous` at its local column,
/// checked against `before` at it, come in its order, the others are
/// sorted as `sort_decreasing` sorts them, and the two lists merge by key,
/// the order's comparison deciding where two keys agree. Per column of the
/// matrix before, the round's column it became, or `MonoId::NONE` where it
/// is no column of the round. The sort spends what `sort_decreasing`
/// spends over the new monomials; the lookup of the old columns present
/// `budget::PROBE` units a round monomial, and the merge `KEY` units an old column
/// and a unit a new monomial it places, each after every `RUN`, at
/// `Site::Columns`; a stop where the budget's check stops the call. In a
/// keyed order alone.
fn merge_columns<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    monomials: &mut Vec<MonoId>,
    previous: &[MonoId],
    before: &[MonoId],
    budget: &mut Budget,
) -> Result<Vec<MonoId>, Stop> {
    debug_assert!(O::KEYED, "a merge by keys");
    debug_assert!(!before.is_empty(), "a matrix before with columns");
    // sort-share-time.11 design-checked -- the bit set and the table over the matrix before's columns are allocated per build and dropped with the numbering, not kept in the run's buffers: they hold a bit and 4 bytes a column of the matrix before, and the instances whose rounds share few columns, where the allocation has nothing to pay for, ran within +0.002 in ln of their instructions against the base (log 0105-3, _Step 6_)
    let mut present = vec![0u64; before.len().div_ceil(MARK_BITS)];
    let mut old_to_new = filled(before.len(), MonoId::NONE, budget, Site::Columns)?;
    // The monomials that were no column of the matrix before, compacted to
    // the vector's front; the rest are marked. Each round monomial is in
    // the vector once (the stamps), so the marks are the old columns
    // present.
    let total = monomials.len();
    let mut new = 0;
    for (k, chunk) in previous.chunks(RUN).enumerate() {
        for (j, &c) in chunk.iter().enumerate() {
            let u = monomials[k * RUN + j];
            if c.index() < before.len() && before[c.index()] == u {
                mark(&mut present, &[c]);
            } else {
                monomials[new] = u;
                new += 1;
            }
        }
        budget.spend(chunk.len() * crate::budget::PROBE, Site::Columns)?;
    }
    let keyed = keyed_decreasing(table, order, &monomials[..new], budget, Site::Columns)?;
    monomials.clear();
    // The marks read in increasing column order are the old columns in
    // decreasing order of their monomials. Before each old column come the
    // new monomials above it, and after the last the rest. The old columns
    // merged since the last spend of theirs.
    let (mut word, mut at) = (0, 0);
    let mut olds = 0;
    while let Some(column) = next_marked(&mut present, &mut word) {
        let old = before[column];
        let key = table.key(order, old);
        // sort-share-time.10 practice-checked: galloping (exponential) search -- the new elements above an old column are found by a galloping search from the merge's position and copied in bulk, not compared and pushed one by one: a round whose columns are mostly new then pays the sort's write-back per new element and a compare or two per old column, and a round whose columns are mostly old a compare per old column (log 0105-3, _Step 6_)
        let goes_before = |&element: &u128| {
            // Distinct monomials never compare equal, and an old column is
            // no new monomial.
            let cmp = key
                .cmp(&((element >> PAYLOAD_BITS) << PAYLOAD_BITS))
                .then_with(|| table.cmp(order, old, MonoId::from_raw(payload(element))));
            debug_assert!(
                matches!(cmp, Ordering::Greater | Ordering::Less),
                "an old column {old} among the new"
            );
            match cmp {
                Ordering::Less => true,
                Ordering::Greater | Ordering::Equal => false,
            }
        };
        // Where the columns are mostly old, no new element lies above most
        // old columns, and the first probe alone decides.
        if keyed.get(at).is_some_and(goes_before) {
            let above = 1 + prefix_len(&keyed[at + 1..], goes_before);
            place_keyed(monomials, &keyed[at..at + above], budget)?;
            at += above;
        }
        old_to_new[column] = MonoId::at(monomials.len());
        monomials.push(old);
        olds += 1;
        if olds == RUN {
            budget.spend(olds * KEY, Site::Columns)?;
            olds = 0;
        }
    }
    budget.spend(olds * KEY, Site::Columns)?;
    place_keyed(monomials, &keyed[at..], budget)?;
    debug_assert_eq!(monomials.len(), total, "every round monomial numbered once");
    Ok(old_to_new)
}

/// Appends the ids of the keyed elements `keyed` to `monomials`, a unit an
/// element spent at `Site::Columns` after each `RUN`; a stop where the
/// budget's check stops the call.
fn place_keyed(
    monomials: &mut Vec<MonoId>,
    keyed: &[u128],
    budget: &mut Budget,
) -> Result<(), Stop> {
    for chunk in keyed.chunks(RUN) {
        monomials.extend(
            chunk
                .iter()
                .map(|&element| MonoId::from_raw(payload(element))),
        );
        budget.spend(chunk.len(), Site::Columns)?;
    }
    Ok(())
}

/// The length of the prefix of `v` on which `holds` holds, where it holds
/// on a prefix and fails past it: the front is probed at doubling steps
/// until one fails or the slice ends, and the last step is bisected.
fn prefix_len<T>(v: &[T], holds: impl Fn(&T) -> bool) -> usize {
    // `v[..done]` holds, and `v[done + step - 1]` is the next probe.
    let (mut done, mut step) = (0, 1);
    while done + step <= v.len() && holds(&v[done + step - 1]) {
        done += step;
        step *= 2;
    }
    let end = (done + step).min(v.len());
    done + v[done..end].partition_point(|e| holds(e))
}

/// Appends the monomials of t·g to the matrix's columns, as local columns
/// until the columns are numbered, and puts those new to the round on
/// `work` (`Table::row_products`), `RUN` terms of g at a time, each chunk
/// spending `PRODUCT` units a product after it. Overflow where a product passes
/// the format, a stop where the budget's check stops the call. Where the
/// vector is narrow and the row could take the round's monomials past what
/// its entries hold, the vector widens first (`fit`). Where the
/// row would take the vector past what it may hold with the rows of the
/// matrix before, those are dropped first (`Matrix::room`).
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
    fit(&mut m.cols, m.monomials.len(), monomials.len(), budget)?;
    // matrix-slack.1 practice-checked: address-space reservation -- a matrix's columns that would pass RESERVE_PAST entries take a reservation of RESERVATION entries, whose pages the rows write as they are appended, not columns growing by doubling throughout
    // matrix-slack.4 design-checked -- the reservation is the capacity of the columns' Vec, reserved exactly and mapped by the allocator, not a buffer of the crate's own over mmap, mremap, and munmap (not measured)
    // matrix-slack.7 design-checked -- the reservation is made before the row whose products would pass RESERVE_PAST, not before a matrix's first row where the run's matrix before it passed RESERVE_PAST (log 0041-1)
    m.room(monomials.len(), budget)?;
    make_room(&mut m.cols, monomials.len());
    let start_col = m.cols.len();
    // product-lookup-time.2 design-checked -- the lookup runs with no check of its ids, the construction of the row's multiplier and of the element's monomials as its guarantee, not one assert on both ids per lookup (log 0031-1)
    // product-lookup-time.3 design-checked -- nor are a row's ids checked once as a batch, by their maximum, before its lookups (log 0031-1)
    // call-stop-latency.2 practice-checked: cooperative cancellation -- every loop of a call whose work grows with the input spends its work on the budget inside the loop, per iteration or per chunk of at most RUN iterations where an iteration's work is a few nanoseconds, not a spend once before or after the whole loop, a tick per iteration whose work grows, nor no spend (log 0138-1)
    // A row of an element of millions of terms spent once after its
    // products was one stretch without a check: its products are formed
    // and spent `RUN` terms at a time.
    for chunk in monomials.chunks(RUN) {
        // SAFETY: the run's table returned the multiplier, by `intern` (the
        // inter-reduction's unit monomial, a replayed row's multiplier from
        // its exponents) or by `div` (a pair's, a reducer's, a reductant's),
        // and the element's monomials, by interning the input's terms or as
        // the monomials of the matrix that made the element. A run holds one
        // table and one basis.
        unsafe {
            table.row_products(
                row.multiplier,
                chunk,
                round,
                &mut m.cols,
                work,
                &mut m.previous,
                budget,
            )
        }?;
        // matrix-stop-latency.5 design-checked -- symbolic preprocessing spends 16 units a product, every row it multiplies, the given rows included, not ticks per reducer row alone nor 8 units a product (log 0055-2, 0056-1)
        // gf2-build-rest-time.2 design-checked -- a row by the unit monomial spends PRODUCT units a term as every row does, not fewer for its terms' entry with no lookup, so that the run's work, which its handovers and bounded runs read, and with it every call's path stay as they are (log 0161-1, 0161-2)
        budget.spend(chunk.len() * PRODUCT, Site::Products)?;
    }
    m.monomials.extend_from_slice(&work[start..]);
    m.spans.push((start_col, m.cols.len()));
    Ok(())
}

// column-index-peak.13 design-checked -- the rows a reduction makes keep 4-byte columns, whatever their matrix's width, a row taken as it stands copied wide, not made rows in their matrix's width: a made row's column vector becomes its polynomial's monomial vector in place (round-memory-time.3) (not measured)
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
    // round-memory-time.3 design-checked -- the polynomial made from a row takes the row's column vector, each column replaced by its monomial in place, not a new vector of the monomials collected from the columns (log 0051-1)
    // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
    /// The polynomial of the row, its columns mapped back to monomials in
    /// the row's column vector, which the polynomial takes, a unit an entry
    /// spent at `Site::Decode` after each `RUN`; a stop where the budget's
    /// check stops the call.
    pub(crate) fn poly<D: Module>(
        self,
        matrix: &Matrix,
        budget: &mut Budget,
    ) -> Result<Poly<F, D>, Stop> {
        let mut monomials = self.cols;
        for chunk in monomials.chunks_mut(RUN) {
            for c in chunk.iter_mut() {
                *c = matrix.monomials[c.index()];
            }
            budget.spend(chunk.len(), Site::Decode)?;
        }
        Ok(Poly {
            monomials,
            coefs: self.coefs,
            position: D::Position::default(),
        })
    }

    /// As `poly`, the row's vectors taken and left empty.
    pub(crate) fn take_poly<D: Module>(
        &mut self,
        matrix: &Matrix,
        budget: &mut Budget,
    ) -> Result<Poly<F, D>, Stop> {
        Reduced {
            cols: mem::take(&mut self.cols),
            coefs: mem::take(&mut self.coefs),
            ..*self
        }
        .poly(matrix, budget)
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
    /// plain round in the order of the rows to reduce, and in a plain round
    /// that combined rows the rows reduced on their own, the probe's and,
    /// where the round took the rows' mode, those after its first block.
    /// Empty where every row to reduce went into a combination.
    pub(crate) zero: Vec<RowId>,
    // trace-replay.11 design-checked -- the reduction hands on the pivot of every column at its end, not a learner that works the pivots out again from the matrix (not measured)
    /// The pivot of every column when the reduction ended.
    pub(crate) pivot: Vec<Pivot>,
    // trace-replay.17 design-checked -- the signature reduction hands on the order in which it took the rows, not the learner sorting the rows by signature again (not measured)
    /// The rows in the order the reduction took them, empty where it set
    /// every pivot row before it reduced a row.
    pub(crate) taken: Vec<RowId>,
    // gfp-unsolved.9 design-checked -- Stats counts the reductions run, rows or combinations, not the rows of the blocks combined (log 0014-1)
    /// The reductions run: rows, or combinations of rows.
    pub(crate) reductions: u64,
    /// Those of them that reduced to zero.
    pub(crate) zeros: u64,
    /// The mode the rows to reduce after the first block took where the
    /// reduction combined rows (`reduce`); None where it reduced each row on
    /// its own throughout, in a signature round among them.
    pub(crate) mode: Option<RoundMode>,
    /// Whether the reduction's matrix took bands (`takes_bands`): a plain
    /// round's combinations, or a signature round's reductants that it
    /// reduced in no packed rows.
    pub(crate) bands: bool,
    /// The K columns' words where the reduction reduced its rows in packed
    /// rows (`packed::Rows`), None where it did not.
    pub(crate) packed: Option<u32>,
    /// The sketches' draws where the reduction took the sketches
    /// (`Zeros::Sketches`), None where it did not.
    pub(crate) sketches: Option<u32>,
}

/// What a reduction's rows have made beside the pivots, for `Reduction`:
/// the rows that reduced to zero or a test ruled zero, the reductions run,
/// and those of them that reduced to zero.
struct Tally {
    zero: Vec<RowId>,
    reductions: u64,
    zeros: u64,
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

// gfp-unsolved.7 design-checked -- a combination's coefficients are uniform in [1, p), from a Xoshiro256PlusPlus seeded by the call's generator after its other draws, not a generator of its own seeded by a constant or coefficients in [0, p) (not measured)
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

    // gfp-unsolved.18 practice-checked: Lemire's multiply-shift range reduction -- a coefficient in [1, p) is the high word of a 64-bit draw times p − 1, plus 1, not the draw reduced mod p − 1 nor a draw rejected and drawn again
    /// A coefficient of [1, p), for 2 ≤ p < 2^64: each value has a
    /// probability within 2^−64 of 1/(p − 1).
    fn coef(&mut self, p: u64) -> u64 {
        debug_assert!(p >= 2, "p ≥ 2");
        1 + ((u128::from(self.rng.next_u64()) * u128::from(p - 1)) >> 64) as u64
    }

    // gf2-forced-zeros-time.10 design-checked -- a sketch's words come from the run's Combine, the draw of its combinations, seeded by the run's seed, not a generator of the sketch's own seeded by a constant (log 0134-2)
    /// A word of a sketch, two draws.
    fn word(&mut self) -> u128 {
        (u128::from(self.rng.next_u64()) << 64) | u128::from(self.rng.next_u64())
    }
}

// gfp-unsolved.3 tuned-checked -- a block of BLOCK rows is 64 consecutive rows of those to reduce, by leading column, not 16, 32, 128, or 256 rows, nor the rows in another order (log 0014-1)
/// The rows of a matrix's first block of combinations (gfp-unsolved.2),
/// and of every block after one that made a pivot.
pub(crate) const BLOCK: usize = 64;

// gf2-bound-losses.3 design-checked -- the projection holds once the reduction has begun PROJECTED_ROWS of its rows to reduce, a block's, not from the first row, one row's work projected over the round (log 0131-2)
/// The rows to reduce a round's reduction has begun before its row work
/// is projected over the round (`Reducer::within_bound`): a block's.
pub(crate) const PROJECTED_ROWS: usize = BLOCK;

/// The rows of a probing round's second block that the probe reduces on
/// their own (short-row-reduce-time.2), and with `BLOCK` the fewest rows
/// to reduce of a round that takes a mode after its first block (`reduce`).
pub(crate) const PROBE: usize = 8;

// short-row-reduce-time.4 design-checked -- a run probes at its first round that reduces BLOCK + PROBE rows or more, and again at every round whose rows to reduce number twice the last probed round's or more, or half or fewer, or after REPROBE rounds without a probe, not a probe at every round nor one probe per run
/// The rounds without a probe after which a round of `BLOCK + PROBE` rows
/// to reduce or more probes again (`Modes::probes`).
pub(crate) const REPROBE: u32 = 4;

/// A plain run's choice between the two modes (`RoundMode`), one per run that
/// combines rows (`f4::run`): the last decision, which a round of
/// `BLOCK + PROBE` rows to reduce or more whose matrix takes no bands
/// reduces the rows after its first block in (`reduce`), and what a probe
/// reads beside the round's own figures: the rows to reduce the run's
/// rounds have reduced and the pivots they made, the rows to reduce of the
/// last probed round, and the rounds since it.
pub(crate) struct Modes {
    /// The last decision; None before the first.
    pub(crate) decision: Option<RoundMode>,
    /// The rows to reduce of the last probed round; 0 before any.
    probed_rows: usize,
    /// The rounds since the last probe, none of which probed.
    since: u32,
    /// The rows to reduce the run's rounds have reduced, in either mode.
    rows: u64,
    /// The pivots those rows made.
    pivots: u64,
}

/// What the probe of a round read (`Modes::decide`): of the round's first
/// block, its reductions, the entries they subtracted, and the pivots it
/// made; and the entries the `PROBE` rows reduced on their own subtracted.
struct Probe {
    reductions: u64,
    subtracted: u64,
    pivots: usize,
    alone: u64,
}

impl Modes {
    pub(crate) fn new() -> Modes {
        Modes {
            decision: None,
            probed_rows: 0,
            since: 0,
            rows: 0,
            pivots: 0,
        }
    }

    /// Whether a round of `rest` rows to reduce whose matrix takes no bands
    /// probes: it holds `BLOCK + PROBE` rows or more, and no round has
    /// probed, or its rows number twice the last probed round's or more, or
    /// half or fewer, or `REPROBE` rounds have passed without a probe.
    fn probes(&self, rest: usize) -> bool {
        rest >= BLOCK + PROBE
            && (self.decision.is_none()
                || rest >= 2 * self.probed_rows
                || 2 * rest <= self.probed_rows
                || self.since >= REPROBE)
    }

    /// The decision of a round of `rest` rows to reduce from its probe,
    /// which becomes the run's last: the rows' mode where
    /// `BLOCK`·c₁ < (`BLOCK`·ρ̂ + `ZEROS`)·c₆₄, c₁ the entries the probe
    /// rows subtracted a row, c₆₄ those the first block subtracted a
    /// reduction, and ρ̂ the run's pivots plus one over its rows to reduce,
    /// the first block among them; the blocks' mode otherwise. The round
    /// counts as probed.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the decision weighs counts against each other, and f64 holds every count where a product of four may pass u128"
    )]
    fn decide(&mut self, rest: usize, probe: &Probe) -> RoundMode {
        debug_assert!(
            rest >= BLOCK + PROBE,
            "a probed round holds BLOCK + PROBE rows or more"
        );
        debug_assert!(probe.reductions >= 1, "the first block ran a combination");
        // short-row-reduce-time.3 design-checked -- the rows' mode is taken where BLOCK·c₁ < (BLOCK·ρ̂ + ZEROS)·c₆₄, ρ̂ the run's pivots made from rows to reduce plus 1 over its rows to reduce so far, the probed block among them, not ρ̂ from the probed block alone, ρ̂ = 1, nor the decision by time
        // Per block of BLOCK rows the rows' mode costs about BLOCK rows of
        // c₁, the blocks' mode about BLOCK·ρ̂ + ZEROS combinations of c₆₄.
        let rows = (self.rows + BLOCK as u64) as f64;
        let pivots = (self.pivots + probe.pivots as u64 + 1) as f64;
        let c1 = probe.alone as f64 / PROBE as f64;
        let c64 = probe.subtracted as f64 / probe.reductions as f64;
        let decision = if BLOCK as f64 * c1 < (BLOCK as f64 * pivots / rows + ZEROS as f64) * c64 {
            RoundMode::Rows
        } else {
            RoundMode::Blocks
        };
        self.decision = Some(decision);
        self.probed_rows = rest;
        self.since = 0;
        decision
    }

    // short-row-reduce-time.9 design-checked -- ρ̂'s rows and pivots count every row to reduce the run's rounds have reduced, in either mode and in rounds with bands, and the pivots those rows made, not the rows reduced in the blocks' mode alone
    /// A round of `rows` rows to reduce that made `pivots` pivots has
    /// ended, one more round since the last probe where it did not probe.
    fn ended(&mut self, rows: usize, pivots: usize, probed: bool) {
        self.rows += rows as u64;
        self.pivots += pivots as u64;
        if !probed {
            self.since = self.since.saturating_add(1);
        }
    }
}

/// The test profile's mode where `MODE_FORCED` holds one, which stands in
/// for the probe and the run's decision; None outside the test profile.
fn forced_mode() -> Option<RoundMode> {
    #[cfg(test)]
    if let Some(forced) = MODE_FORCED.get() {
        return Some(forced);
    }
    None
}

// plain-zero-time.6 tuned-checked -- a grown block that has made more than 16 pivots stops before its next band or combination, its rows taken again from its first, in a block of BLOCK rows, not a grown block reduced to its end, nor a stop past 4, 8, or 32 pivots (log 0077-1)
/// The pivots past which a block of more than `BLOCK` rows stops before
/// its next band or combination (`reduce`).
pub(crate) const GROWN_PIVOTS: usize = 16;

// gfp-unsolved.4 cited-checked: monagan-2017 §2, via berthomieu-2021 §4.2 -- zero comes by chance with probability about 1/p, so a set of rows is discarded once zero is obtained k times in a row, with a probability of error of 1/p^k
/// Combinations that reduce to zero before a block is done.
pub(crate) const ZEROS: usize = 2;

// signature-zero-time.3 tuned-checked -- a test starts once two rows of a run reduced on their own have reduced to zero one after another, not after each row that reduces to zero (log 0049-1)
/// Rows of a signature round's run reduced to zero one after another before
/// a test of the run's rest (`reduce_sig`), and of a learn's plain round
/// before a test of the rows after them (`reduce`).
pub(crate) const STREAK: usize = 2;

// signature-zero-time.5 design-checked -- after a failed test in the run, a test takes a block of 8 rows, then twice the last block's rows after each block that passes, not the whole rest at every test (log 0049-1)
/// The rows of a signature round's first block after a failed test in its
/// run (`reduce_sig`), and of a learn's plain round's (`reduce`).
pub(crate) const FIRST: usize = 8;

/// The rows per combination of a test below which a group takes no test
/// (signature-zero-time.9).
pub(crate) const GROUP_ROWS: usize = 2;

/// The bound on a test's false zeros: s combinations reduced to zero
/// rule a group zero where (p − 1)^s reaches it (signature-zero-time.8).
pub(crate) const ZERO_BOUND: u128 = 1 << 31;

// signature-zero-time.8 cited-checked: demin-2026b §2.5, Proposition 3 -- the number s of reductions to zero that ends a block is computed from the bound on the probability of an incorrect result and the field's size; monagan-2017 §2 -- if zero is obtained k times in a row the rows are discarded with a probability of error of 1/p^k
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

// gf2-forced-zeros-time.4 cited-checked: demin-2026b §2.2, Corollary 1's proof -- if the first i − 1 rows span a space of q^(i−1) elements, the i-th row, picked uniformly from k^N, is outside this span with probability 1 − 1/q^(N−i+1)
/// The residues a sketch's echelon holds before the sketch is drawn again
/// (`Reducer::reduce_rows_alone`): 97, `sketch::BITS` less the 31 bits of
/// `ZERO_BOUND`. A row that does not reduce to zero has a residue of 0 with
/// probability at most 2^(e − 128) for e residues in the echelon, 2^−31 at
/// most.
pub(crate) const SKETCH_ECHELON: usize = (sketch::BITS - ZERO_BOUND.ilog2()) as usize;

const _: () = assert!(
    ZERO_BOUND.is_power_of_two() && ZERO_BOUND.ilog2() < sketch::BITS,
    "ZERO_BOUND is a power of two below 2^BITS"
);

// gf2-forced-zeros-time.7 tuned-checked -- a plain round over GF(2) takes the sketches only where its matrix's rows hold 16 entries or more in the mean, not in every round (log 0134-1, 0134-2)
/// The fewest entries a matrix's rows hold in the mean for its plain round
/// over GF(2) to take the sketches: 16. Taken in every round, the sketches
/// made the rounds of short rows, each cheap to reduce, take longer. With
/// this rule those rounds reduce as fast as without the sketches (log
/// 0134-1, _Step 1_).
pub(crate) const SKETCH_ROW_ENTRIES: usize = 16;

/// Whether the plain round of `matrix` over a field that packs, whose run
/// may combine rows, takes the sketches: its rows hold `SKETCH_ROW_ENTRIES`
/// entries or more in the mean. In the test profile `SKETCH_FORCED`
/// decides where it holds a choice.
fn sketches(matrix: &Matrix) -> bool {
    let takes = matrix
        .rows
        .len()
        .checked_mul(SKETCH_ROW_ENTRIES)
        .is_some_and(|least| matrix.entries() >= least);
    #[cfg(test)]
    let takes = SKETCH_FORCED.get().unwrap_or(takes);
    takes
}

/// The echelon's cap: `SKETCH_ECHELON`, in the test profile
/// `ECHELON_FORCED` where it holds one.
fn echelon_cap() -> usize {
    #[cfg(test)]
    if let Some(cap) = ECHELON_FORCED.get() {
        return cap;
    }
    SKETCH_ECHELON
}

/// What a plain round's rows, each reduced on its own, take to rule rows
/// zero (`Reducer::reduce_rows_alone`).
pub(crate) enum Zeros<'c> {
    /// No test: every row is reduced.
    None,
    /// The tests of random combinations: their draw, and s
    /// (`zero_combinations`).
    Tests(&'c mut Combine, usize),
    /// The sketches over GF(2), their words from the run's draw.
    Sketches(&'c mut Combine),
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

/// A row that a band takes with its leading entry before the sweep
/// (`Reducer::sweep_led`): a made row, or a given row of the matrix.
#[derive(Clone, Copy)]
enum Led {
    Made(MadeId),
    Given(RowId),
}

// row-traffic-time.7 tuned-checked -- a signature round takes bands only where its matrix's rows hold 64 entries or more in the mean, not in every matrix, nor from 32 or 128 entries a row (log 0044-1)
/// The fewest entries a matrix's rows hold in the mean for its signature
/// round to reduce its reductants in bands, and its plain run its
/// combinations: 64.
pub(crate) const BAND_ENTRIES: usize = 64;

/// Whether `rows` rows of `entries` entries in all hold `BAND_ENTRIES`
/// entries or more in the mean.
pub(crate) fn long_rows(entries: usize, rows: usize) -> bool {
    entries >= BAND_ENTRIES * rows
}

// gfp-unsolved.5 design-checked -- the rows are combined only where p² ≥ 2^31, and a smaller p reduces each row on its own, not combinations for every p with more zeros for a smaller p (not measured)
// call-report.11 design-checked -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
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

// signature-zero-time.10 design-checked -- the tests run where no cofactors are tracked, neither the trace nor the learner listens, and the field has one lane, not in a learn run as well (log 0049-1)
// call-report.11 design-checked -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
/// Whether a signature round rules rows zero by tests over `F` at p, where
/// its hooks allow it (`f4::combines`): the field has one lane, and p
/// takes a test (`zero_combinations`).
pub(crate) fn tests_rows<F: Field>(p: u64) -> bool {
    F::LANES == 1 && zero_combinations(p).is_some()
}

// row-traffic-time.7 tuned-checked -- a signature round takes bands only where its matrix's rows hold 64 entries or more in the mean, not in every matrix, nor from 32 or 128 entries a row (log 0044-1)
/// Whether a signature round reduces the reductants of `matrix` in bands,
/// and a plain run that combines rows its combinations (`reduce`): a
/// reduction may take bands (`may_band`), and the matrix's rows are long
/// (`long_rows`).
pub(crate) fn takes_bands<F: Field, M: Format, C: Cofactors<F, M>>(matrix: &Matrix) -> bool {
    may_band::<F, M, C>() && long_rows(matrix.entries(), matrix.rows.len())
}

/// Whether a reduction over `F` that `C` tracks may take bands: the field
/// has them, and `C` tracks no cofactors.
fn may_band<F: Field, M: Format, C: Cofactors<F, M>>() -> bool {
    // row-traffic-time.9 design-checked -- a round that tracks cofactors reduces each reductant alone, not bands with their cofactors tracked by a start, a subtraction, and a keep per band row (not measured)
    F::BANDS && !C::TRACKS
}

// row-traffic-time.1 cited-checked: martani-2012 §3.5.1, via boyer-2016 §2 and §4.2 -- the multiline row reduction reduces two rows at the same time, each entry of the reducer read once for both
/// The end of the band that starts at `rows[k]`, a reductant, below
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

/// The key of a row of leading column `lead` and id `id` for `sort::keys`:
/// the column above the id, so that the keys go by leading column, and by
/// id within one.
fn lead_key(lead: MonoId, id: u32) -> u128 {
    (u128::from(lead.raw()) << PAYLOAD_BITS) | u128::from(id)
}

/// The rows the payloads of `keys` name, in the keys' order, a unit a row
/// spent at `site` after each `RUN`. A stop where the budget's check stops
/// the call.
fn keyed_rows(keys: &[u128], budget: &mut Budget, site: Site) -> Result<Vec<RowId>, Stop> {
    let mut rows = Vec::with_capacity(keys.len());
    for chunk in keys.chunks(RUN) {
        rows.extend(chunk.iter().map(|&k| RowId::from_raw(payload(k))));
        budget.spend(chunk.len(), site)?;
    }
    Ok(rows)
}

// call-stop-latency.56 design-checked -- a row a reduction's loop passes without reducing it is counted in a local count, which spends RUN rows at a time, not a spend per row, a read and a write of the budget at each row of a loop that does little else; a build that spent per row ran no slower once rebuilt (log 0138-4)
/// Counts a row a reduction's loop passes without reducing it in
/// `passed`, and once `RUN` are counted spends them, `budget::PROBE` units
/// each, at `Site::Setup`. A stop where the budget's check stops the call.
fn pass(passed: &mut usize, budget: &mut Budget) -> Result<(), Stop> {
    debug_assert!(*passed < RUN, "{passed} rows counted, past RUN");
    *passed += 1;
    if *passed < RUN {
        return Ok(());
    }
    *passed = 0;
    budget.spend(RUN * crate::budget::PROBE, Site::Setup)
}

// row-scan-time.10 design-checked -- the scan is a parameter of the test profile carried by the buffers, Dense or Marked for the test code and the counts' choice otherwise, not the test code timing the call or counting the rows each scan took (not measured)
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

// row-scan-time.2 tuned-checked -- a row takes the marked scan where the columns the dense scans of the rows before it would pass exceed twice the entries those rows wrote, once four rows are counted, not the marked scan for the rest of the matrix once the counts first say so, a ratio of 8 or 1, nor the scan chosen once per matrix from its shape (log 0072-1)
/// A row takes the marked scan where the columns the rows' dense scans
/// would pass exceed `MARKED_RATIO` times the entries the rows wrote: 2.
pub(crate) const MARKED_RATIO: u64 = 2;

/// The rows a reducer counts before a row takes the marked scan: 4.
pub(crate) const MARKED_ROWS: u64 = 4;

// row-scan-time.3 transferred-checked: operating systems, aging (use counters halved periodically) -- carried into the counts that choose the scan: both counts are halved every 64 rows, so that they weigh the recent rows, not counts over every row of the matrix (log 0072-1)
/// Both counts are halved after every `MARKED_DECAY` rows: 64.
pub(crate) const MARKED_DECAY: u64 = 64;

// row-scan-time.6 design-checked -- the marks are one level of words, read word by word from the word of the row's start, not a second level of a bit per word of marks (log 0072-1)
/// The columns a word of the marks holds.
const MARK_BITS: usize = u64::BITS as usize;

/// Sets the mark of every column of `cols`: bit c mod `MARK_BITS` of word
/// c / `MARK_BITS`. The writes have no bounds check outside the test
/// profile, whose debug assertions check them.
#[inline(always)]
pub(crate) fn mark<I: Column>(marks: &mut [u64], cols: &[I]) {
    for &c in cols {
        let c = c.column();
        debug_assert!(
            c / MARK_BITS < marks.len(),
            "column {c} past the marks' {len} words",
            len = marks.len()
        );
        // SAFETY: a matrix's columns lie below its width, and the marks
        // hold a word per `MARK_BITS` columns of it (`Reducer::take_marked`,
        // `merge_columns`).
        unsafe { *marks.get_unchecked_mut(c / MARK_BITS) |= 1 << (c % MARK_BITS) };
    }
}

// row-scan-time.1 practice-checked: sparse accumulator -- the marked scan visits the columns whose mark a load, an addition, or a row operation of the row set, in increasing column order, each mark cleared as read, not every column from the row's start, the row's reach in the graph of the pivot rows found before the numerical work, a heap of the columns written, nor the scan ended at the last column a row operation reached (log 0072-1)
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
    // round-memory-time.9 design-checked -- the pivot array stays one per matrix, handed on with the reduction, not kept from round to round as the dense row is (log 0051-1)
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
    // row-traffic-time.2 practice-checked: array of structures -- a band's dense rows are one array whose element holds the band's entries of one column side by side, not a dense row per band row; the nearest, martani-2012 §3.5's multiline, keeps the entries of all its rows at a column next to each other in the reducer rows, and reduces the dense rows `temp1` and `temp2` apart (log 0034-1, 0044-1)
    /// The band's dense rows, interleaved by column: empty from `new`
    /// until the reduction's first band (`widen_band`), then an element per
    /// column of the matrix, BAND rows each, zero between two bands.
    pub(crate) band: Vec<[F::Acc; BAND]>,
    /// The work vectors band row k is built in: `begin_band` hands them
    /// to band row k's result and `end_band` takes them back, holding the
    /// row's entries, stale until the next band clears them.
    band_cols: [Vec<MonoId>; BAND],
    band_coefs: [Vec<F::Coef>; BAND],
    // row-scan-time.8 design-checked -- the marks are the reducer's own, allocated at its first marked row, not marks taken from the run's buffers and kept from round to round, which a stopped reduction leaves set (not measured)
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
    // gf2-reduction-time.16 design-checked -- the packed rows live in the round's reducer, made at its first reduced row and dropped at its end, not kept in the run's buffers from round to round as the dense row is (log 0125-3)
    /// The round's packed rows: made at the reducer's first reduced row
    /// where it packs (`pack`), None before and where it does not.
    packed: Option<Rows>,
    /// The K columns' words where the reducer reduces in packed rows from
    /// its first reduced row on (`pack`), None where it does not.
    packs: Option<u32>,
    /// The budget's row work when the round's rows to reduce began, the
    /// round's rows to reduce, and the rows of them begun: the projection
    /// of the row work over the round (`within_bound`). 0 each but where
    /// `reduce` set them.
    start: u64,
    to_reduce: usize,
    reached: usize,
    /// The scan the buffers asked for.
    #[cfg(test)]
    scan: Scan,
    /// Whether the buffers allow the back-substitution and the
    /// inter-reduction bands.
    #[cfg(test)]
    bands: bool,
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
        // round-memory-time.5 design-checked -- a reduction takes the dense row zeroed whole, every entry up to the matrix's width written zero, not the dense row the reduction before left trusted to be zero and only the entries past its length written (log 0051-1)
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
            packed: None,
            packs: None,
            start: 0,
            to_reduce: 0,
            reached: 0,
            #[cfg(test)]
            scan: bufs.scan,
            #[cfg(test)]
            bands: bufs.bands,
        };
        #[cfg(test)]
        let red = {
            let mut red = red;
            red.take_marked(red.scan.marks(false));
            red
        };
        red
    }

    // row-scan-time.8 design-checked -- the marks are the reducer's own, allocated at its first marked row, not marks taken from the run's buffers and kept from round to round, which a stopped reduction leaves set (not measured)
    /// Sets whether the next row takes the marked scan, and where it does
    /// and the marks are empty, makes them: a word per `MARK_BITS` columns of
    /// the matrix, every mark clear.
    fn take_marked(&mut self, marked: bool) {
        if marked && self.marks.is_empty() {
            self.marks = vec![0; self.dense.len().div_ceil(MARK_BITS)];
        }
        self.marked = marked;
    }

    // row-scan-time.2 tuned-checked -- a row takes the marked scan where the columns the dense scans of the rows before it would pass exceed twice the entries those rows wrote, once four rows are counted, not the marked scan for the rest of the matrix once the counts first say so, a ratio of 8 or 1, nor the scan chosen once per matrix from its shape (log 0072-1)
    // row-scan-time.3 transferred-checked: operating systems, aging (use counters halved periodically) -- carried into the counts that choose the scan: both counts are halved every 64 rows, so that they weigh the recent rows, not counts over every row of the matrix (log 0072-1)
    // row-scan-time.4 design-checked -- the entries written are the budget's counts of entries subtracted and added, read once a row, beside the load's length, not a count kept in the reducer at each row operation of the scan (log 0072-1)
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

    // gf2-signature-losses.5 design-checked -- the bound on the row work is read once per row, or per combination or band of combinations, before it is reduced, in the plain round's row loops, not at the budget's full checks, which return the caller's Stop, nor at a round's end (not measured)
    // gf2-bound-losses.2 design-checked -- the projection holds where the round reduces in sparse rows, not in a round in packed rows, whose first block reduces the early pivots to their reduced rows, which projected over mqbool-21's round 4 halted it (log 0131-1, 0131-2)
    /// `Halt::Outworked`, the buffers given back to `bufs`, where the row
    /// work lies past its bound (`Budget::outworked`), or where the round
    /// reduces in sparse rows, has begun `PROJECTED_ROWS` of its rows to
    /// reduce or more, and its row work projected over them and over its
    /// sugar passes the bound (`Budget::projected_past`, gf2-bound-losses.1,
    /// gf2-build-time.3).
    #[inline]
    fn within_bound(&mut self, budget: &Budget, bufs: &mut Buffers<F>) -> Result<(), Halt> {
        let projected = self.packs.is_none()
            && self.reached >= PROJECTED_ROWS
            && budget.projected_past(self.start, self.reached, self.to_reduce);
        if budget.outworked() || projected {
            self.give_back(bufs);
            return Err(Halt::Outworked);
        }
        Ok(())
    }

    /// Whether the back-substitution and the inter-reduction take bands
    /// where `C` tracks the cofactors: where a reduction may (`may_band`),
    /// and in the test profile where the buffers allow it.
    #[allow(
        clippy::unused_self,
        reason = "the test profile reads the buffers' switch from the reducer"
    )]
    fn bands_in_echelon<M: Format, C: Cofactors<F, M>>(&self) -> bool {
        #[cfg(test)]
        if !self.bands {
            return false;
        }
        may_band::<F, M, C>()
    }

    // round-memory-time.1 cited-checked: monagan-2007 §2.2 -- the merge buffers grow dynamically, and to further amortize the cost of memory allocation they are reused in the next call rather than freed each time
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

    // round-memory-time.2 design-checked -- a nonzero row is copied into vectors of its length when its reduction ends, and a row that reduced to zero allocates nothing, not the row handed the work vectors themselves with new ones taken for the next row, nor the work vectors shrunk in place (log 0051-1)
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

    // table-read-time.6 design-checked -- a matrix's row reads its span and slices its columns, and the reduction reads a row's source, with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
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

    // f4-gfp.7 cited-checked: monagan-2015 §2.2 -- a row is cancelled wherever a pivot is known, a nonzero result made monic and inserted as a pivot; it does not pay to perform only top reductions
    /// Reduces row r by the pivots in every column from `from` on: the
    /// entries left, in increasing column order, not made monic. The columns
    /// before `from` must hold no entry of the row but its leading one,
    /// which is kept as it is. The cofactor tracking hears of every
    /// subtraction, and the budget of every row operation (`cancel`). The
    /// row is built in the work vectors and copied at its length (`end`).
    /// Where the reducer packs (`pack`), the row is reduced in the round's
    /// packed rows instead (`reduce_packed`). A stop where the budget's
    /// check stops the call.
    pub(crate) fn reduce<M: Format, C: Cofactors<F, M>>(
        &mut self,
        r: RowId,
        from: MonoId,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<Reduced<F>, Stop> {
        if self.packs.is_some() {
            debug_assert!(!C::TRACKS, "no cofactors of a packed reduction");
            return self.reduce_packed(r, from, budget);
        }
        let f = self.field;
        let cols = self.matrix.row(r);
        let coefs = self.coefs(r);
        // SAFETY: a matrix's columns lie below its width, the dense row's
        // length (`columns`), and row r has a coefficient per column.
        by_width!(cols, |c| unsafe { f.load(&mut self.dense, c, coefs) });
        // row-scan-time.4 design-checked -- the entries written are the budget's counts of entries subtracted and added, read once a row, beside the load's length, not a count kept in the reducer at each row operation of the scan (log 0072-1)
        self.written += cols.len() as u64;
        let lead = cols.lead();
        if self.marked {
            // The lead before `from` is taken outside the scan.
            let marked = if lead < from { cols.tail() } else { cols };
            by_width!(marked, |c| mark(&mut self.marks, c));
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

    /// Makes the reducer reduce in packed rows from its first reduced row
    /// on (`reduce_packed`), its K columns taking `words` words
    /// (`packed::packs`). The field packs, and no row has been reduced.
    fn pack(&mut self, words: u32) {
        debug_assert!(F::PACKS && F::LANES == 1, "a field that packs, of one lane");
        debug_assert!(
            self.packed.is_none() && self.made.is_empty() && self.rows == 0,
            "no row reduced before the reducer packs"
        );
        self.packs = Some(words);
    }

    // gf2-reduction-time.5 cited-checked: faugere-2010 §3.3, attributes it to Lachartre 2008 [11] -- the pivot rows are reduced by themselves, B ← A⁻¹B, A only read and kept sparse, B written; makarim-2017 §4 -- instead of adding a reducible term to the output to later cancel it again, a tail-reduced basis multiple of the term's leading monomial is obtained and its full-reduced tail subtracted
    /// Reduces row r by the pivots in every column from `from` on, as
    /// `reduce` does, in the round's packed rows (`packed::Rows::reduce`),
    /// which the reducer's first such row makes from the pivots set then:
    /// the entries left, in increasing column order, each coefficient 1.
    /// The columns before `from` must hold no entry of the row but its
    /// leading one, which is kept as it is. The row is built in the work
    /// vectors and copied at its length (`end`). No cofactors are tracked.
    /// A stop where the budget's check stops the call.
    fn reduce_packed(
        &mut self,
        r: RowId,
        from: MonoId,
        budget: &mut Budget,
    ) -> Result<Reduced<F>, Stop> {
        let (f, matrix) = (self.field, self.matrix);
        let cols = matrix.row(r);
        let lead = cols.lead();
        let mut out = self.begin(r);
        let rest = if lead < from {
            out.cols.push(lead);
            out.coefs.push(self.coefs(r)[0]);
            cols.tail()
        } else {
            cols
        };
        let pivot = &self.pivot;
        let packed = self.packed.get_or_insert_with(|| {
            Rows::new(pivot.iter().map(|p| match *p {
                Pivot::Row(q) => Some(q),
                Pivot::None | Pivot::New(_) => None,
            }))
        });
        debug_assert_eq!(
            self.packs.map(|words| words as usize),
            Some(packed.words()),
            "the K columns' words where the reducer packs"
        );
        packed.reduce(rest, |q| matrix.row(q), &mut out.cols, budget)?;
        out.coefs.resize(out.cols.len(), f.coef(1));
        Ok(self.end(out))
    }

    /// Sets row r as the pivot of column c, its leading column, which has
    /// none. Where the round has its packed rows, after its first reduced
    /// row, the row's packed row is stored first (`packed::Rows::set_later`).
    /// A stop where the budget's check stops the call.
    fn set_row(&mut self, c: usize, r: RowId, budget: &mut Budget) -> Result<(), Stop> {
        debug_assert!(
            self.pivot.get(c).is_some_and(|p| p.is_none()) && self.matrix.lead(r).index() == c,
            "row {r} leads column {c}, which has no pivot"
        );
        if let Some(packed) = self.packed.as_mut() {
            let matrix = self.matrix;
            packed.set_later(matrix.row(r), |q| matrix.row(q), budget)?;
        }
        self.pivot[c] = Pivot::Row(r);
        Ok(())
    }

    /// Given row r of the inter-reduction (`inter_reduce`) reduced on its
    /// own from the column after its lead (`reduce`), and kept by the
    /// cofactor tracking as its kept row `kept`. A stop where the budget's
    /// check stops the call.
    fn reduce_given<M: Format, C: Cofactors<F, M>>(
        &mut self,
        r: RowId,
        kept: usize,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<Reduced<F>, Stop> {
        let after_lead = MonoId::at(self.matrix.lead(r).index() + 1);
        let mut row = self.reduce(r, after_lead, cof, budget)?;
        cof.keep(budget)?;
        row.kept = KeepId::at(kept);
        Ok(row)
    }

    // table-read-time.7 design-checked -- the scan reads its dense row, its pivot array, and its made rows with no bounds check outside the test profile, not a bounds check per read (log 0045-1)
    // table-read-time.8 design-checked -- the scan reads the dense row at its column unchecked, not the dense row padded to a power of two and indexed by the column masked with its length less one (log 0030-1)
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
                by_width!(pc.tail(), |tail| {
                    // SAFETY: as in `reduce`.
                    unsafe { field.sub_mul(&mut self.dense, tail, &pv[1..], v) };
                    // row-scan-time.5 design-checked -- the marked scan subtracts by the field's sub_mul and then marks the subtracted row's columns in a pass of its own, not a row operation of each field that marks as it writes (not measured)
                    if MARKED {
                        mark(&mut self.marks, tail);
                    }
                });
                cof.sub_row(q, v);
                // matrix-stop-latency.3 design-checked -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
                budget.spend_subtracted(pc.len(), Site::Row)?;
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
                // row-scan-time.5 design-checked -- the marked scan subtracts by the field's sub_mul and then marks the subtracted row's columns in a pass of its own, not a row operation of each field that marks as it writes (not measured)
                if MARKED {
                    mark(&mut self.marks, &row.cols[1..]);
                }
                cof.sub_kept(row.kept, v, budget)?;
                // matrix-stop-latency.3 design-checked -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
                budget.spend_subtracted(row.cols.len(), Site::Row)?;
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
    /// (`count`), and its scan spent at `Site::Scan` (`scanned`). A stop
    /// where the budget's check stops the call, the dense row and the marks
    /// then not zero.
    fn cancel<M: Format, C: Cofactors<F, M>>(
        &mut self,
        from: MonoId,
        out: &mut Reduced<F>,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert_eq!(self.pivot.len(), self.dense.len(), "a pivot a column");
        if self.marked {
            // row-scan-time.1 practice-checked: sparse accumulator -- the marked scan visits the columns whose mark a load, an addition, or a row operation of the row set, in increasing column order, each mark cleared as read, not every column from the row's start, the row's reach in the graph of the pivot rows found before the numerical work, a heap of the columns written, nor the scan ended at the last column a row operation reached (log 0072-1)
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
        // The scan the row took: `count` chooses the next row's.
        let scanned = self.scanned(from);
        self.count(from, budget);
        // call-stop-latency.16 design-checked -- a scan of a row spends what it reads, the columns it passes, the words of the marks, or the entries of a made row's tail, a unit each, once per row, not the scan unspent and bounded by the rows' ticks (matrix-stop-latency.4) (not measured)
        budget.spend(scanned, Site::Scan)
    }

    /// The units of the scan of a row from `from`, in the scan the counts
    /// chose for it: the columns from `from` to the matrix's width,
    /// or where it takes the marked scan the words of the marks from the
    /// one that holds `from`, which the scan reads to the marks' end.
    fn scanned(&self, from: MonoId) -> usize {
        debug_assert!(
            from.index() <= self.dense.len(),
            "a start of {from} past the matrix's {len} columns",
            len = self.dense.len()
        );
        if self.marked {
            self.marks.len() - from.index() / MARK_BITS
        } else {
            self.dense.len() - from.index()
        }
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
        let k = f.neg(k);
        by_width!(cols, |c| {
            // SAFETY: as in `reduce`.
            unsafe { f.sub_mul(&mut self.dense, c, coefs, k) };
            if self.marked {
                mark(&mut self.marks, c);
            }
        });
        // matrix-stop-latency.3 design-checked -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
        budget.spend_added(cols.len(), Site::Row)
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

    // signature-zero-time.7 design-checked -- a combination stops at its first nonzero entry in a column without a pivot, which shows it nonzero, not reduced to its end (log 0049-1)
    /// Whether the dense row as loaded, a combination of rows, reduces to
    /// zero by the pivots in every column from `from` on, as `cancel`
    /// reduces it, in the scan `cancel` would take, but making no row: false
    /// at the first nonzero entry in a column without a pivot, after the
    /// dense row is set to zero from that column on. The columns before
    /// `from` must hold no entry. The dense row and the marks are zero after
    /// either, the combination is counted (`count`), and its scan spent as
    /// in `cancel`, the dense row's fill from the exit's column counted as
    /// the columns it passes. Each row operation spends its row's length
    /// right after it, as in `cancel`. No cofactor tracking hears of it. A
    /// stop where the budget's check stops the call, the dense row and the
    /// marks then not zero.
    fn reduces_to_zero<M: Format>(
        &mut self,
        from: MonoId,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        debug_assert_eq!(self.pivot.len(), self.dense.len(), "a pivot a column");
        let zero = 'scan: {
            if self.marked {
                // row-scan-time.1 practice-checked: sparse accumulator -- the marked scan visits the columns whose mark a load, an addition, or a row operation of the row set, in increasing column order, each mark cleared as read, not every column from the row's start, the row's reach in the graph of the pivot rows found before the numerical work, a heap of the columns written, nor the scan ended at the last column a row operation reached (log 0072-1)
                let mut w = self.first_word(from);
                while let Some(c) = next_marked(&mut self.marks, &mut w) {
                    let left =
                        self.cancel_entry::<true, M, NoCofactors>(c, &mut NoCofactors, budget)?;
                    if left.is_some() {
                        // row-scan-time.9 design-checked -- the marked exit at a column without a pivot zeroes the entries at the marks from that column on and clears the marks, not the dense row filled with zeros to its end and every word cleared (not measured)
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
        // The scan the row took: `count` chooses the next row's.
        let scanned = self.scanned(from);
        self.count(from, budget);
        // call-stop-latency.16 design-checked -- a scan of a row spends what it reads, the columns it passes, the words of the marks, or the entries of a made row's tail, a unit each, once per row, not the scan unspent and bounded by the rows' ticks (matrix-stop-latency.4) (not measured)
        budget.spend(scanned, Site::Scan)?;
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
    /// groups rule zero join `zero`. A group's least leading column spends
    /// `budget::PROBE` units a row, and its rows joining `zero` a unit each,
    /// `RUN` at a time, at `Site::Row`. A stop where the budget's check
    /// stops the call.
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
        // signature-zero-time.4 design-checked -- a run's first test takes the whole rest of the run, not a block (log 0049-1)
        // signature-zero-time.6 tuned-checked -- a block holds at most BLOCK rows, the plain run's block, not doubling without a cap (log 0049-1)
        let mut block = if failed { FIRST } else { usize::MAX };
        loop {
            let group_len = block.min(rest.len() - tested.ruled);
            // signature-zero-time.9 design-checked -- a group of fewer than 2s rows takes no test, not a test of any group of two rows or more (not measured)
            if group_len < GROUP_ROWS * s {
                break;
            }
            let group = &rest[tested.ruled..tested.ruled + group_len];
            // call-stop-latency.52 design-checked -- a test group's least leading column is sought RUN rows at a time, PROBE units a row, and the rows it rules zero join them a unit each, RUN at a time, not unspent before the group's first combination (not measured)
            let mut from = MonoId::NONE;
            for chunk in group.chunks(RUN) {
                for &k in chunk {
                    from = from.min(self.matrix.lead(row(k)));
                }
                budget.spend(chunk.len() * crate::budget::PROBE, Site::Row)?;
            }
            if from == MonoId::NONE {
                break;
            }
            // signature-zero-time.1 cited-checked: monagan-2017 §2, attributes it to A. Steel's private communication and found it independently -- random linear combinations of a set's rows are reduced, and if zero is obtained k times in a row the set is discarded with a probability of error of 1/p^k
            // signature-zero-time.14 design-checked -- the s combinations of a test are reduced one after another in the dense row, not loaded into a band and reduced in one sweep (not measured)
            let mut passed = true;
            for _ in 0..s {
                budget.tick(Site::Row)?;
                for &k in group {
                    self.add(row(k), self.field.coef(combine.coef(p)), budget)?;
                }
                // signature-zero-time.12 design-checked -- each combination counts as a reduction, and one that reduces to zero as a zero, not the rows the test rules zero (not measured)
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
            for chunk in group.chunks(RUN) {
                zero.extend(chunk.iter().map(|&k| row(k)));
                budget.spend(chunk.len(), Site::Row)?;
            }
            tested.ruled += group_len;
            block = (2 * group_len).min(BLOCK);
        }
        Ok(tested)
    }

    /// Reduces `rows`, rows to reduce by decreasing leading column, in
    /// random combinations of blocks of consecutive rows, the first of
    /// `size` rows, as `reduce` states: each combination a tick of the
    /// budget, in bands where `bands`, a block done after `ZEROS` zeros or
    /// as many pivots as it has rows, a block after one that made no pivot
    /// twice its rows, and a grown block stopped past `GROWN_PIVOTS`
    /// pivots. The size of the block after the last, for a caller that
    /// goes on in blocks. `Halt::Misfit(0)` where a row to keep has no
    /// inverse at its leading entry, `Halt::Outworked` where the row work,
    /// done or projected over the round, lies past its bound before a band or
    /// a combination (`within_bound`), the buffers given back at either, and
    /// a stop where the budget's check stops the call.
    #[allow(
        clippy::too_many_arguments,
        reason = "the rows, the first block's size, whether bands are taken, the combinations, the cofactors, the budget, the buffers, and the tally, none of which belongs with another"
    )]
    fn reduce_blocks<M: Format, C: Cofactors<F, M>>(
        &mut self,
        rows: &[RowId],
        size: usize,
        bands: bool,
        combine: &mut Combine,
        cof: &mut C,
        budget: &mut Budget,
        bufs: &mut Buffers<F>,
        tally: &mut Tally,
    ) -> Result<usize, Halt> {
        let f = self.field;
        let p = f.characteristic();
        // cyclic-gap.2 design-checked -- the blocks are cut from the start of that order, the first the BLOCK rows that lead farthest right, not cut by increasing leading column and taken last to first (log 0066-1)
        // plain-zero-time.8 design-checked -- blocks grow in every matrix that combines rows, not only where the matrix takes bands (log 0077-1)
        let (mut start, mut size) = (0, size);
        // The rows of `rows` some block has begun: a stopped block begun
        // again from its first row counts none of them twice.
        let mut begun = 0;
        while start < rows.len() {
            let first = start;
            let block = &rows[start..rows.len().min(start + size)];
            start += block.len();
            // gf2-bound-losses.4 design-checked -- the rows begun are the rows to reduce the reduction has taken up, each block's rows that no block before took, each row reduced on its own as it begins, and the rows a zero test rules zero, not the reductions done, which count a block's combinations (log 0131-2)
            if start > begun {
                self.reached += start - begun;
                begun = start;
            }
            let grown = block.len() > BLOCK;
            // Every row of the block leads at or right of its last.
            let from = self.matrix.lead(block[block.len() - 1]);
            let (mut pivots, mut done, mut stopped) = (0, 0, false);
            // plain-reduction-time.1 cited-checked: martani-2012 §3.5.1.1 -- the axpy reduces two rows at the same time, the dense arrays temp1 and temp2, in one pass over a multiline row, each of its positions and values fetched once from memory and used twice
            // plain-reduction-time.4 design-checked -- a block takes bands of min(BAND, n − pivots) combinations while none has reduced to zero and two pivots or more are left to make, then one at a time, not bands to the block's end, bands of BAND always, nor one band and then one at a time (not measured)
            // plain-zero-time.9 design-checked -- after a band whose combinations all made pivots the block takes another band, not one combination alone that tests whether the block is spent (log 0077-1)
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
                    budget.tick(Site::Row)?;
                }
                self.within_bound(budget, bufs)?;
                for &r in block {
                    // plain-reduction-time.3 design-checked -- a block row's coefficients are drawn together, one for each band row in band-row order, not combination by combination (not measured)
                    let k: [F::Coef; BAND] = std::array::from_fn(|j| {
                        if j < lanes {
                            f.coef(combine.coef(p))
                        } else {
                            F::Coef::default()
                        }
                    });
                    self.add_band(r, &k, budget)?;
                }
                // Kept in band order, the band rows set the pivots the
                // same combinations reduced one after another set: the
                // sweep reduced each by the band rows before it.
                for row in self.reduce_band_loaded(lanes, block[0], from, budget)? {
                    tally.reductions += 1;
                    if row.cols.is_empty() {
                        done += 1;
                        tally.zeros += 1;
                        continue;
                    }
                    pivots += 1;
                    if !self.keep(row, cof, budget)? {
                        self.give_back(bufs);
                        return Err(Halt::Misfit(0));
                    }
                }
            }
            // gfp-unsolved.19 design-checked -- a block is also done once its combinations have made as many pivots as it has rows, which span it, and not after as many combinations as it has rows, where a combination of a full block that reduces to zero by chance leaves it a pivot short (log 0014-2)
            // plain-zero-time.3 design-checked -- each block ends after ZEROS of its combinations reduce to zero, not at its first zero with two combinations of every row to reduce at the round's end (log 0077-1)
            while !stopped && pivots < block.len() && done < ZEROS {
                if grown && pivots > GROWN_PIVOTS {
                    stopped = true;
                    break;
                }
                budget.tick(Site::Row)?;
                self.within_bound(budget, bufs)?;
                for &r in block {
                    let k = f.coef(combine.coef(p));
                    self.add(r, k, budget)?;
                }
                let row = self.reduce_loaded(block[0], from, cof, budget)?;
                tally.reductions += 1;
                if row.cols.is_empty() {
                    done += 1;
                    tally.zeros += 1;
                    continue;
                }
                pivots += 1;
                if !self.keep(row, cof, budget)? {
                    self.give_back(bufs);
                    return Err(Halt::Misfit(0));
                }
            }
            if stopped {
                start = first;
                size = BLOCK;
            } else if pivots == 0 {
                // plain-zero-time.1 transferred-checked: search algorithms, exponential search -- carried into the sizes of the blocks: a block after a block that made no pivot holds twice that block's rows, not every block BLOCK rows (log 0077-1)
                // plain-zero-time.2 design-checked -- the grown block is a block, its combinations making its pivots until ZEROS reduce to zero, not a test group of two combinations that rules it zero or hands its rows to blocks of BLOCK (log 0077-1)
                // plain-zero-time.4 tuned-checked -- twice the rows, not four times (log 0077-1)
                // plain-zero-time.5 design-checked -- a block holds at most the rows left, not at most 1024 rows (log 0077-1)
                size = 2 * block.len();
            } else {
                // plain-zero-time.7 tuned-checked -- a block after a block that made a pivot holds BLOCK rows, not twice the rows where the block made one or two pivots (log 0077-1)
                size = BLOCK;
            }
        }
        Ok(size)
    }

    /// Reduces `rows`, rows to reduce by decreasing leading column, each on
    /// its own from its leading column, a tick of the budget each, and rules
    /// rows zero as `zeros` says. With `Zeros::Tests`, the combinations' draw
    /// and s, after `STREAK` rows reduced to zero one after another the rows
    /// after them are tested in groups (`test_groups`), as `reduce` states.
    /// With `Zeros::Sketches`, after `STREAK` rows reduced to zero one after
    /// another a sketch is drawn where rows remain (`draw_sketch`), and each
    /// row after it is tested by its residue (`Sketch::residue`): a row of
    /// residue 0 is ruled zero, a tick and its entries spent, and every other
    /// row is reduced and makes a pivot, its residue kept in the echelon. Once
    /// the echelon holds its cap the sketch is drawn again. The rows a test or
    /// a sketch rules zero join the tally's zero rows. The draws, where
    /// `zeros` is `Zeros::Sketches`. `Halt::Misfit(0)` where a row to keep
    /// has no inverse at its leading entry, `Halt::Outworked` where the row
    /// work, done or projected over the round, lies past its bound before a
    /// row reduced (`within_bound`), the buffers given back at either, and a
    /// stop where the budget's check stops the call.
    fn reduce_rows_alone<M: Format, C: Cofactors<F, M>>(
        &mut self,
        rows: &[RowId],
        zeros: Zeros<'_>,
        cof: &mut C,
        budget: &mut Budget,
        bufs: &mut Buffers<F>,
        tally: &mut Tally,
    ) -> Result<Option<u32>, Halt> {
        // The tests' draw and s, and the sketches' draw.
        let (mut tests, mut sketches) = match zeros {
            Zeros::None => (None, None),
            Zeros::Tests(combine, s) => (Some((combine, s)), None),
            Zeros::Sketches(combine) => (None, Some(combine)),
        };
        // The rows reduced to zero one after another, and whether a
        // test has failed.
        let (mut streak, mut failed) = (0, false);
        // gf2-forced-zeros-time.6 design-checked -- a sketch is drawn once STREAK rows have reduced to zero one after another and again once its echelon is full, not where the rows left times the zero rows' recent or mean row work exceed the draw's entries (log 0134-1, 0134-2)
        let mut sketch: Option<Sketch> = None;
        let mut next = 0;
        while next < rows.len() {
            let r = rows[next];
            next += 1;
            budget.tick(Site::Row)?;
            // gf2-forced-zeros-time.1 invented-checked -- a row is ruled zero where its residue by the round's sketch is 0, the exclusive or of its columns' words, each bit a vector every pivot annihilates, not each row reduced on its own, nor random combinations of the rows, which over GF(2) pass a nonzero group with probability 1/2 each (signature-zero-time.8), nor a vector of the matrix's pivot rows alone, blind to the rows the reduction makes (log 0134-1, 0134-2)
            let residue = match sketch.as_ref() {
                Some(sk) => {
                    let cols = self.matrix.row(r);
                    budget.spend(cols.len(), Site::Row)?;
                    let x = sk.residue(cols);
                    if x == 0 {
                        // gf2-forced-zeros-time.9 design-checked -- a row a sketch rules zero joins the zero rows and counts as no reduction and no zero, which the round's reductions run, not a reduction to zero each (not measured)
                        tally.zero.push(r);
                        continue;
                    }
                    Some(x)
                }
                None => None,
            };
            // gf2-forced-zeros-time.8 design-checked -- the rows a sketch rules zero count as no row begun, so that the projection of the row work over the round reads the rows reduced, not the rows ruled counted as begun at no row work, nor charged the mean zero row's work (log 0134-1, 0134-2)
            self.reached += 1;
            self.within_bound(budget, bufs)?;
            let lead = self.matrix.lead(r);
            let row = self.reduce(r, lead, cof, budget)?;
            tally.reductions += 1;
            if !row.cols.is_empty() {
                streak = 0;
                if !self.keep(row, cof, budget)? {
                    self.give_back(bufs);
                    return Err(Halt::Misfit(0));
                }
                // gf2-forced-zeros-time.3 design-checked -- the rows made after the draw join the test by their residues, kept in an echelon by their top bit, so that a row of residue other than 0 makes a pivot for certain, not the sketch left stale and drawn again after rows that passed and reduced to zero, nor drawn again after every made row (log 0134-1, 0134-2)
                if let (Some(x), Some(sk), Some(combine)) =
                    (residue, sketch.as_mut(), sketches.as_deref_mut())
                    && sk.keep(x, echelon_cap())
                {
                    self.draw_sketch(sk, combine, budget)?;
                }
                continue;
            }
            debug_assert!(
                residue.is_none(),
                "a row of residue other than 0 lies outside the span of the pivots"
            );
            tally.zero.push(r);
            tally.zeros += 1;
            streak += 1;
            if let Some(combine) = sketches.as_deref_mut() {
                if sketch.is_none() && streak >= STREAK && next < rows.len() {
                    let mut sk = Sketch::new();
                    self.draw_sketch(&mut sk, combine, budget)?;
                    sketch = Some(sk);
                }
                continue;
            }
            let Some((combine, s)) = tests.as_mut() else {
                continue;
            };
            // zero-rows-time.4 tuned-checked -- a test starts after STREAK rows reduced to zero one after another, 2, not after each row that reduces to zero (log 0078-1)
            if streak < STREAK {
                continue;
            }
            streak = 0;
            // learn-zero-time.2 design-checked -- the plain run's test takes the signature run's groups, the rows to the matrix's end first and after a failed test blocks of FIRST rows doubling up to BLOCK, not blocks of BLOCK rows from the first test (log 0063-1)
            // zero-rows-time.3 design-checked -- a failed test hands its rows to the reductions each on its own, not its first half tested again while the half holds 2s rows or more (log 0078-1)
            let tested = self.test_groups::<M, RowId>(
                &rows[next..],
                |r| r,
                failed,
                (combine, *s),
                &mut tally.zero,
                budget,
            )?;
            failed = tested.failed;
            tally.reductions += tested.reductions;
            tally.zeros += tested.zeros;
            // zero-rows-time.2 design-checked -- after a failed test the rows go on each on its own until the next streak, not a test of the next block right after the next row that makes a pivot (log 0078-1)
            next += tested.ruled;
            self.reached += tested.ruled;
        }
        Ok(sketches.map(|_| sketch.as_ref().map_or(0, Sketch::draws)))
    }

    // gf2-forced-zeros-time.2 practice-checked: back substitution -- a sketch's words are drawn by one pass from the matrix's last column to its first, a random word at a column without a pivot and the exclusive or of the words of the pivot row's tail at a column with one, not from the packed rows' reduced rows over the K columns, which a round in sparse rows lacks (log 0134-1, 0134-2)
    /// Draws `sketch` over the pivots as set now (`Sketch::draw`): a
    /// matrix row's tail or a made row's at each column that has a pivot, the
    /// words from `combine`. A stop where the budget's check stops the call.
    fn draw_sketch(
        &self,
        sketch: &mut Sketch,
        combine: &mut Combine,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let (matrix, made, pivot) = (self.matrix, &self.made, &self.pivot);
        sketch.draw(
            matrix.columns(),
            |c| match pivot[c] {
                Pivot::None => None,
                Pivot::Row(q) => Some(matrix.row(q).tail()),
                Pivot::New(m) => Some(Row::Wide(&made[m.index()].cols[1..])),
            },
            || combine.word(),
            budget,
        )
    }

    // call-stop-latency.66 design-checked -- the hooks input, start_kept, sub_kept, and keep take the call's budget and return its stop, which NoCofactors' Ok(()) compiles to nothing, not each hook returning the units it walked for its caller to spend (log 0138-7)
    /// Keeps a nonzero reduced row: made monic, the pivot of its leading
    /// column, and kept by the cofactor tracking, which spends on `budget`,
    /// and where the round has its packed rows, its packed row stored
    /// (`packed::Rows::keep`). False, and nothing kept, where its leading
    /// entry has no inverse: over several lanes, 0 at one lane. A stop where
    /// the budget's check stops the call.
    pub(crate) fn keep<M: Format, C: Cofactors<F, M>>(
        &mut self,
        mut row: Reduced<F>,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<bool, Stop> {
        let f = self.field;
        // q-learn-time.4 cited-checked: dekleine-2001 §3 -- on each division with a nonzero remainder the list of leading coefficients is checked, and a c_i = 0 means the leading term of r is lost under p_i, with very high probability a bad prime; graebe-1993 §4, independently -- a leading coefficient detected as nonzero that is zero mod p_i shows p_i unlucky
        // q-learn-time.5 design-checked -- a disagreement of the lanes halts the run with Halt::Misfit(0), and the call returns Error::Misfit { step: 0 }, not the lane at 0 switched out and the run continued at the other prime, nor a halt of a variant of its own (not measured)
        let Some(s) = f.inv(row.coefs[0]) else {
            return Ok(false);
        };
        for c in &mut row.coefs {
            *c = f.mul(*c, s);
        }
        cof.scale(s);
        cof.keep(budget)?;
        row.kept = KeepId::at(self.kept);
        self.kept += 1;
        self.pivot[row.cols[0].index()] = Pivot::New(MadeId::at(self.made.len()));
        if let Some(packed) = self.packed.as_mut() {
            packed.keep(&row.cols);
        }
        self.made.push(row);
        Ok(true)
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

    // row-traffic-time.11 design-checked -- a band row's result is built in a pair of work vectors of its own, BAND pairs from the buffers, and copied at its length, not a band row's vectors made empty and grown by push (log 0060-2)
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
    /// every entry zero, its fill spent `BAND` units a column at
    /// `Site::Setup`. A stop where the budget's check stops the call.
    fn widen_band(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        // row-traffic-time.10 cited-checked: monagan-2007 §2.2 -- to further amortize the cost of memory allocation the merge buffers are reused in the next call rather than freed each time
        // The band from the buffers, cleared by `new`, is zeroed whole at
        // the reduction's first band, up to the matrix's width: no band the
        // buffers held is trusted to be zero.
        if self.band.is_empty() {
            let columns = self.matrix.columns();
            self.band.resize(columns, [F::Acc::default(); BAND]);
            // call-stop-latency.53 design-checked -- the reducer's arrays spend once filled, the dense row and the pivot array the matrix's columns and the band BAND units a column, as a row's scan spends its columns once, not unspent (not measured)
            budget.spend(BAND * columns, Site::Setup)?;
        }
        Ok(())
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
        self.widen_band(budget)?;
        let mut out: Vec<Reduced<F>> = Vec::with_capacity(n);
        let mut from = MonoId::NONE;
        for (k, &r) in rows.iter().enumerate() {
            let (cols, coefs) = (matrix.row(r), self.coefs(r));
            debug_assert!(
                !matches!(self.pivot[cols.lead().index()], Pivot::None),
                "a band row's leading column has a pivot"
            );
            from = from.min(cols.lead());
            // SAFETY: as in `reduce`, and k < n ≤ BAND.
            by_width!(cols, |c| unsafe {
                f.load_band(&mut self.band, k, c, coefs);
            });
            out.push(self.begin_band(k, r));
        }
        let out = self.sweep_band::<false>(n, from, out, budget, Site::Row)?;
        Ok(self.end_bands(out))
    }

    // plain-reduction-time.2 cited-checked: martani-2012 §3.5.1.1 -- one axpy adds a row to the two dense arrays temp1 and temp2 at once, by the scalars a1 and a2, each of the row's positions and values fetched once from memory and used twice
    /// Band row j gains `k[j]` times row r, for every j < BAND, in one pass
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
        self.widen_band(budget)?;
        let (cols, coefs) = (matrix.row(r), self.coefs(r));
        let neg = k.map(|c| f.neg(c));
        // SAFETY: as in `reduce`: the band holds an element per column of
        // the matrix (`widen_band`).
        by_width!(cols, |c| unsafe {
            f.sub_mul_band(&mut self.band, c, coefs, &neg);
        });
        // plain-reduction-time.6 design-checked -- each row added to a band spends its length once per band row, BAND times, right after its pass, not its length once per pass (not measured)
        budget.spend_added(BAND * cols.len(), Site::Row)
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
        let out = self.sweep_band::<false>(n, from, out, budget, Site::Row)?;
        Ok(self.end_bands(out))
    }

    /// The band rows' results `out`, in band order, each copied at its
    /// length (`end_band`).
    fn end_bands(&mut self, out: Vec<Reduced<F>>) -> Vec<Reduced<F>> {
        out.into_iter()
            .enumerate()
            .map(|(k, row)| self.end_band(k, row))
            .collect()
    }

    /// Reduces the rows `rows`, 2 to BAND of them, in one sweep, each by the
    /// pivots in every column after its leading one: each row's leading
    /// entry begins its result, and its tail is loaded into the band, before
    /// the sweep (`sweep_band` with `LED`). Each row's leading column holds
    /// a pivot, and the sweep starts after the least of them. `F::BANDS`
    /// holds, and no cofactors are tracked. Each row's result in the rows'
    /// order, not copied: built in the band rows' work vectors, which
    /// `end_band` copies out. The sweep is spent at `site`. A stop where the
    /// budget's check stops the call, the band then not zero.
    fn sweep_led(
        &mut self,
        rows: &[Led],
        budget: &mut Budget,
        site: Site,
    ) -> Result<Vec<Reduced<F>>, Stop> {
        let n = rows.len();
        debug_assert!((2..=BAND).contains(&n), "a band of {n} rows");
        let f = self.field;
        self.widen_band(budget)?;
        let mut out: Vec<Reduced<F>> = Vec::with_capacity(n);
        let mut from = MonoId::NONE;
        for (k, &led) in rows.iter().enumerate() {
            let (source, cols, coefs) = match led {
                Led::Made(j) => {
                    let row = &self.made[j.index()];
                    (row.row, Row::Wide(&row.cols), row.coefs.as_slice())
                }
                Led::Given(r) => (r, self.matrix.row(r), self.coefs(r)),
            };
            debug_assert!(
                !self.pivot[cols.lead().index()].is_none(),
                "a band row's leading column has a pivot"
            );
            // back-substitution-time.4 design-checked -- each band row's lead goes to its result and its tail into the band before the sweep, which starts after the band's least leading column, not each row loaded whole and its lead taken by the sweep as a column without a pivot
            // SAFETY: as in `cancel`, for a made row or a matrix row, and
            // k < n ≤ BAND.
            by_width!(cols.tail(), |tail| unsafe {
                f.load_band(&mut self.band, k, tail, &coefs[1..]);
            });
            let (lead, lc) = (cols.lead(), coefs[0]);
            from = from.min(MonoId::at(lead.index() + 1));
            let mut row = self.begin_band(k, source);
            row.cols.push(lead);
            row.coefs.push(lc);
            out.push(row);
        }
        self.sweep_band::<true>(n, from, out, budget, site)
    }

    /// Reduces band rows 0 to n − 1 in one sweep over the columns from
    /// `from` on, each in turn as `cancel` reduces the dense row: by the
    /// pivots set before the band and by the band rows before it, made
    /// monic. Where `LED` holds, every band row's leading entry is in its
    /// result already, and each is reduced by the pivots alone: no column
    /// becomes a band row's lead, and every entry at a column without a
    /// pivot joins its row's result. The band rows from n on hold zeros.
    /// `F::BANDS` holds, and no cofactors are tracked. Each pivot row
    /// subtracted spends its length once per band row it is subtracted
    /// from, right after it, counted as subtracted, and the sweep `BAND`
    /// units a column it passed once it ends, each at `site`. `out` holds
    /// the n rows' results as `begin_band` made them, and where `LED` holds
    /// with their leading entries. Each row's result in band order, not
    /// monic, empty where the row reduced to zero, in the band rows' work
    /// vectors, not copied. The band is zero after, where every entry lay
    /// from `from` on. A stop where the budget's check stops the call, the
    /// band then not zero.
    #[allow(
        clippy::many_single_char_names,
        reason = "n, k, i, a, v are the notation of the comments' formulas, after faugere-2010 §4.4"
    )]
    fn sweep_band<const LED: bool>(
        &mut self,
        n: usize,
        from: MonoId,
        mut out: Vec<Reduced<F>>,
        budget: &mut Budget,
        site: Site,
    ) -> Result<Vec<Reduced<F>>, Stop> {
        debug_assert!(F::BANDS, "a field with bands");
        debug_assert_eq!(out.len(), n, "a result a band row");
        debug_assert!(
            from.index() <= self.band.len(),
            "a sweep from {from} past the band's {len} columns",
            len = self.band.len()
        );
        let (field, matrix) = (self.field, self.matrix);
        let zero = F::Coef::default();
        // Whether band row k has taken its leading column, and the inverse
        // of its entry there.
        // back-substitution-time.4 design-checked -- each band row's lead goes to its result and its tail into the band before the sweep, which starts after the band's least leading column, not each row loaded whole and its lead taken by the sweep as a column without a pivot
        // A made row's leading column is its own pivot, which a whole row
        // would meet in the sweep.
        let mut led = [LED; BAND];
        let mut inv = [zero; BAND];
        // row-traffic-time.4 cited-checked: faugere-2010 §4.4 -- the row operations done on the blocks are kept in a matrix P, and each later block is updated by a left product by P before it is reduced
        // Band row k as the one-row reduction holds it is its accumulator
        // less Σ_{i<k} rel[k][i] times band row i as it holds it: rel[k][i]
        // is band row k's multiplier of band row i as it holds it, the
        // multiples of band row i made monic it subtracted times the inverse
        // of band row i's entry at the column it took as its lead.
        let mut rel = [[zero; BAND]; BAND];
        let mut has_rel = [false; BAND];
        // row-scan-time.7 design-checked -- the back-substitution and the band sweeps keep their dense scans, not the marked scan (log 0072-1)
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
            let pivot: Option<(Row<'_>, &[F::Coef])> = match self.pivot[column] {
                Pivot::None => None,
                Pivot::Row(row) => Some((matrix.row(row), self.coefs(row))),
                Pivot::New(made_index) => {
                    let made_row = &self.made[made_index.index()];
                    Some((Row::Wide(&made_row.cols), &made_row.coefs))
                }
            };
            if let Some((pc, pv)) = pivot {
                if nonzero == 1
                    && let Some(k) = a.iter().position(|&v| v != zero)
                {
                    // row-traffic-time.5 cited-checked: boyer-2016 §4.2 -- where only one of a multiline's rows has an entry, a specialized AXPY reduces one row, and lightweight tests before the loop run it only on nonzero multipliers
                    // SAFETY: as in `cancel`, a pivot row's columns lie
                    // below the matrix's width, the band's length, and
                    // k < n ≤ BAND.
                    by_width!(pc.tail(), |tail| unsafe {
                        field.sub_mul_band_row(&mut self.band, k, tail, &pv[1..], a[k]);
                    });
                    // row-traffic-time.12 design-checked -- each pivot row a band subtracts spends its length once per band row it is subtracted from, right after the subtraction, not once per band nor no spend in the sweep (log 0044-1)
                    budget.spend_subtracted(pc.len(), site)?;
                } else {
                    // SAFETY: as above.
                    by_width!(pc.tail(), |tail| unsafe {
                        field.sub_mul_band(&mut self.band, tail, &pv[1..], &a);
                    });
                    // row-traffic-time.12 design-checked -- each pivot row a band subtracts spends its length once per band row it is subtracted from, right after the subtraction, not once per band nor no spend in the sweep (log 0044-1)
                    budget.spend_subtracted(pc.len() * nonzero, site)?;
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
                        reason = "v is nonzero, and bands run only over fields of one lane, Gf31 and Gf2"
                    )]
                    let inverse = field.inv(v).expect("a value is nonzero");
                    inv[k] = inverse;
                    claimed = Some(k);
                }
                out[k].cols.push(MonoId::at(column));
                out[k].coefs.push(v);
            }
        }
        // call-stop-latency.51 design-checked -- a band's sweep spends BAND units a column it passes, once it ends, as the band holds BAND entries a column, not a unit a column as a row's scan spends (call-stop-latency.16) (not measured)
        budget.spend(BAND * (self.band.len() - from.index()), site)?;
        Ok(out)
    }

    // order-family.18 design-checked -- where the order is not graded, a round's new rows are inter-reduced to the reduced row echelon form by a back-substitution after the forward pass, not left in the echelon form of grevlex, each new row reduced by the pivots found before it alone (log 0011-1)
    /// Inter-reduces the made rows to the reduced row echelon form: by
    /// decreasing leading column, each row's entries that are the leading
    /// column of another made row are cancelled with that row, already
    /// back-substituted. A row with such an entry is kept again by the
    /// cofactor tracking and replaces itself in `made`, copied at its
    /// length. Each row ticks the budget, and each subtraction spends the
    /// subtracted row's length, counted as subtracted.
    /// No row changes its leading column, and none is made monic again:
    /// each subtracted row is monic, and the leading entry is untouched.
    ///
    /// Where the reduction takes bands (`bands_in_echelon`), every row ticks
    /// the budget first, and the rows with such an entry are taken `BAND`
    /// at a time in that order, the last group the rows left. A group of
    /// two rows or more is reduced in one band sweep (`sweep_led`), where a
    /// band row's entry at another band row's leading column is cancelled
    /// with that row before its back-substitution; a group of one row is
    /// reduced on its own. The reduced row echelon form is unique, so each
    /// row comes out as the reduction of one row at a time makes it. Each
    /// group ticks the budget at `Site::BackBands`, and a band's sweep is
    /// spent there.
    ///
    /// The made rows are sorted by `made_by_lead`, and each row's test for
    /// the leading column of another made row spends the entries it reads
    /// (`needs`).
    fn back_substitute<M: Format, C: Cofactors<F, M>>(
        &mut self,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let by_lead = self.made_by_lead(budget)?;
        // back-substitution-time.1 cited-checked: martani-2012 §3.5.1.1 -- in the Trsm step a multiline of two rows, the dense arrays temp1 and temp2, is reduced by one pivot vector, reducing 2 rows at the same time, its index and value fetched once and used twice; boyer-2016 §4.2, independently -- the multiline optimizes the AXPY and TRSM computations of the reduction's second step
        if !self.bands_in_echelon::<M, C>() {
            for k in by_lead {
                budget.tick(Site::Row)?;
                if self.needs(k, budget)? {
                    self.back_substitute_row(k, cof, budget)?;
                }
            }
            return Ok(());
        }
        let mut need: Vec<MadeId> = Vec::new();
        for k in by_lead {
            budget.tick(Site::Row)?;
            if self.needs(k, budget)? {
                need.push(k);
            }
        }
        // back-substitution-time.2 design-checked -- a band takes BAND consecutive made rows of those that need back-substitution, by decreasing leading column, not bands of rows chosen so that none holds another band row's leading column
        // Which rows meet another's lead shows only as the sweep runs, as a
        // subtraction fills them; consecutive rows share their pivots.
        for group in need.chunks(BAND) {
            // call-stop-latency.17 design-checked -- the back-substitution's band path ticks each band and each row it takes alone, and the inter-reduction's reductions each row and band, not their ticks spent in a first loop over the rows (back-substitution-time.1) (not measured)
            budget.tick(Site::BackBands)?;
            let [k] = group else {
                self.back_substitute_band(group, cof, budget)?;
                continue;
            };
            self.back_substitute_row(*k, cof, budget)?;
        }
        Ok(())
    }

    /// The made rows by decreasing leading column, sorted by `sort::keys`
    /// at `Site::Setup`, a key `budget::PROBE` units, `RUN` at a time; a
    /// stop where the budget's check stops the call.
    fn made_by_lead(&self, budget: &mut Budget) -> Result<Vec<MadeId>, Stop> {
        // call-stop-latency.18 design-checked -- the rows to reduce, the made rows the back-substitution takes, and the inter-reduction's rows are sorted by sort::keys on their leading columns above their ids, and the signature round's scan of a run spends PROBE units a position, RUN positions at a time, not sort_unstable before one spend (matrix-stop-latency.8) nor the scan unspent (not measured)
        let mut by_lead = Vec::with_capacity(self.made.len());
        for start in (0..self.made.len()).step_by(RUN) {
            let end = (start + RUN).min(self.made.len());
            by_lead
                .extend((start..end).map(|k| lead_key(self.made[k].cols[0], MadeId::at(k).raw())));
            budget.spend((end - start) * crate::budget::PROBE, Site::Setup)?;
        }
        sort::keys(&mut by_lead, true, budget, Site::Setup, no_piece)?;
        Ok(by_lead
            .iter()
            .map(|&key| MadeId::from_raw(payload(key)))
            .collect())
    }

    /// The back-substitution of a round that reduces in packed rows: the
    /// made rows' reduced row echelon form, computed in the packed rows the
    /// round keeps of them (`packed::Rows::back_substitute`), the rows by
    /// decreasing leading column (`made_by_lead`). A made row whose result
    /// differs takes it in place of its row, each coefficient 1, and is
    /// kept again, by decreasing leading column, as `back_substitute_row`
    /// keeps it; every other made row stays as it is. No cofactors are
    /// tracked. A stop where the budget's check stops the call.
    fn back_substitute_packed(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let by_lead = self.made_by_lead(budget)?;
        let leads: Vec<MonoId> = by_lead
            .iter()
            .map(|&k| self.made[k.index()].cols[0])
            .collect();
        let one = self.field.coef(1);
        let (made, kept) = (&mut self.made, &mut self.kept);
        #[allow(
            clippy::expect_used,
            reason = "`reduce` back-substitutes in packed rows only where the round's packed rows stand"
        )]
        let packed = self.packed.as_ref().expect("the round's packed rows");
        packed.back_substitute(
            &leads,
            |i, tail| {
                let k = by_lead[i].index();
                let mut cols = Vec::with_capacity(tail.len() + 1);
                cols.push(leads[i]);
                cols.extend_from_slice(tail);
                made[k] = Reduced {
                    row: made[k].row,
                    coefs: vec![one; cols.len()],
                    cols,
                    kept: KeepId::at(*kept),
                };
                *kept += 1;
            },
            budget,
        )
    }

    // order-family.20 design-checked -- a made row whose tail holds no leading column of another made row is left as it is, not begun and kept again by the cofactor tracking (not measured)
    /// Whether made row k's tail holds the leading column of another made
    /// row, which its back-substitution cancels. The entries the test reads
    /// are spent a unit each at `Site::Scan`, and a stop where the budget's
    /// check stops the call.
    fn needs(&self, k: MadeId, budget: &mut Budget) -> Result<bool, Stop> {
        debug_assert!(
            k.index() < self.made.len(),
            "made row {k} past the {len} made",
            len = self.made.len()
        );
        let tail = &self.made[k.index()].cols[1..];
        let at = tail.iter().position(|c| self.pivot[c.index()].is_new());
        // call-stop-latency.16 design-checked -- a scan of a row spends what it reads, the columns it passes, the words of the marks, or the entries of a made row's tail, a unit each, once per row, not the scan unspent and bounded by the rows' ticks (matrix-stop-latency.4) (not measured)
        budget.spend(at.map_or(tail.len(), |i| i + 1), Site::Scan)?;
        Ok(at.is_some())
    }

    /// The back-substitution of made row k on its own (`back_substitute`):
    /// its entries at the leading columns of other made rows cancelled with
    /// those rows, in one dense scan from the column after its lead, the
    /// result built in the work vectors, kept again by the cofactor
    /// tracking, and in place of row k. Each subtraction spends the
    /// subtracted row's length, counted as subtracted, and the scan the
    /// columns it passes, at `Site::Scan`; a stop where the budget's check
    /// stops the call.
    fn back_substitute_row<M: Format, C: Cofactors<F, M>>(
        &mut self,
        k: MadeId,
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let field = self.field;
        let zero = F::Coef::default();
        let row = &self.made[k.index()];
        // SAFETY: as in `cancel`, for a made row.
        unsafe { field.load(&mut self.dense, &row.cols, &row.coefs) };
        cof.start_kept(row.kept, budget)?;
        let (source, lead) = (row.row, row.cols[0]);
        let mut out = self.begin(source);
        out.cols.push(lead);
        out.coefs.push(field.take(&mut self.dense[lead.index()]));
        // A subtracted row changes only columns right of its leading one,
        // and holds no leading column of a made row and no reducer's.
        // row-scan-time.7 design-checked -- the back-substitution and the band sweeps keep their dense scans, not the marked scan (log 0072-1)
        for c in lead.index() + 1..self.dense.len() {
            let v = field.take(&mut self.dense[c]);
            if v == zero {
                continue;
            }
            match self.pivot[c] {
                Pivot::New(j) => {
                    let made_row = &self.made[j.index()];
                    // SAFETY: as in `cancel`, for a made row.
                    unsafe {
                        field.sub_mul(
                            &mut self.dense,
                            &made_row.cols[1..],
                            &made_row.coefs[1..],
                            v,
                        );
                    };
                    cof.sub_kept(made_row.kept, v, budget)?;
                    // matrix-stop-latency.3 design-checked -- each row operation spends its row's length right after it, not a spend per reduced row of what it subtracted nor a count kept here and spent once it passes 2^12 (log 0056-1)
                    budget.spend_subtracted(made_row.cols.len(), Site::Row)?;
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
        // call-stop-latency.16 design-checked -- a scan of a row spends what it reads, the columns it passes, the words of the marks, or the entries of a made row's tail, a unit each, once per row, not the scan unspent and bounded by the rows' ticks (matrix-stop-latency.4) (not measured)
        budget.spend(self.dense.len() - lead.index() - 1, Site::Scan)?;
        cof.keep(budget)?;
        out.kept = KeepId::at(self.kept);
        self.kept += 1;
        self.made[k.index()] = self.end(out);
        Ok(())
    }

    /// The back-substitution of the made rows `group`, 2 to BAND of them,
    /// in one band sweep (`sweep_led`): each row's entries at the leading
    /// columns of other made rows cancelled with those rows as `made` holds
    /// them. Each result is kept again by the cofactor tracking, which
    /// tracks none, and in place of its made row, in the group's order. The
    /// sweep is spent at `Site::BackBands`. A stop where the budget's check
    /// stops the call, the band then not zero.
    fn back_substitute_band<M: Format, C: Cofactors<F, M>>(
        &mut self,
        group: &[MadeId],
        cof: &mut C,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert!(!C::TRACKS, "no cofactors of a band");
        let rows: Vec<Led> = group.iter().map(|&k| Led::Made(k)).collect();
        // back-substitution-time.3 design-checked -- a band row's leading column met in another band row is cancelled by that band row as `made` holds it, before its back-substitution, not by relations between the band rows (faugere-2010 §4.4) nor by the band rows back-substituted among themselves after the sweep
        // The sweep cancels every pivot's column after it as well, and the
        // reduced row echelon form is unique (faugere-1999 §2.2): each row
        // comes out as the back-substitution of one row at a time makes it.
        let out = self.sweep_led(&rows, budget, Site::BackBands)?;
        // back-substitution-time.9 design-checked -- a band's results replace their made rows one at a time, each copied out right before the made row it replaces is dropped, not all of a band's results copied out before any replaces its row
        // Where the copies came first, a later round's forward reductions
        // took longer at equal instructions (log 0091-1, _Step 3_).
        for (b, (&k, row)) in group.iter().zip(out).enumerate() {
            cof.keep(budget)?;
            let mut row = self.end_band(b, row);
            row.kept = KeepId::at(self.kept);
            self.kept += 1;
            self.made[k.index()] = row;
        }
        Ok(())
    }
}

// f4-gfp.6 cited-checked: boyer-2016 §4.3 -- when only the reduction of D is needed, the TRSM step can be omitted, and it is enough to reduce the elements below the pivots
/// Reduces the matrix of a round. Each reducer row is the pivot of its
/// leading column, and of the given rows that share a leading column the
/// one with the fewest terms is. The other given rows are reduced by
/// decreasing leading column, those of one leading column by decreasing
/// row index, by the pivots and the new pivots found
/// before them, each row a tick of the budget. Pivot rows are not reduced.
/// Where `combine` is Some and p² ≥ 2^31, p ≥ 46341, those rows are
/// reduced in random
/// combinations instead (`Reducer::reduce_blocks`), in blocks of
/// consecutive rows of that
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
/// made; then one at a time. Where the matrix takes no bands and holds
/// `BLOCK` + `PROBE` rows to reduce or more, the rows after the first
/// block take the round's mode (`RoundMode`): the blocks' combinations on,
/// or each row on its own with the tests below, their combinations drawn
/// by `combine` (`Reducer::reduce_rows_alone`). The mode is the last
/// decision of `modes`, or where the round probes (`Modes::probes`) the
/// probe's: the `PROBE` rows of the second block that lead farthest left
/// are reduced each on its own, counted as the rows' mode counts them, and
/// `Modes::decide` weighs the entries they subtracted a row against the
/// first block's a reduction. The rest of the round is then reduced in the
/// decided mode, those rows left out. Without `modes`, or before any
/// decision, the mode is the blocks'. In the test profile `MODE_FORCED`
/// stands in for the probe and the decision where it holds a mode.
/// `Reduction::mode` names the mode. Where the rows are reduced each on
/// its own, the field packs (`Field::PACKS`), `C` tracks no cofactors, and
/// the round's rows, entries, columns, and pivots fit (`packed::packs`),
/// the rows are reduced in packed rows (`Reducer::reduce_packed`), each by the
/// reduced rows of the pivots set before them and the packed rows of the
/// rows made since. Where `full` is true the
/// new rows are then inter-reduced (`Reducer::back_substitute`), in the
/// round's packed rows where it reduced in them
/// (`Reducer::back_substitute_packed`), and the
/// call report's step ends its stage `Reduce` before and `Backsub` after.
/// The setup spends the matrix's columns once the reducer is made, its
/// loops over the rows `budget::PROBE` units a row, and the sort of the
/// rows to reduce as `sort::keys` spends, each at `Site::Setup`. Each row
/// added to a combination spends its length, to a band `BAND` times its
/// length, each row operation its row's (`Reducer::cancel`), in a band
/// once per band row it is subtracted from, each scan of a row the columns
/// it passes (`Reducer::scanned`), and each packed row added its entries
/// (`packed::Rows`).
/// `Halt::Misfit(0)` where a row to keep has no inverse at its leading
/// entry, over several lanes 0 at one lane (`Reducer::keep`),
/// `Halt::Outworked` where the budget's row work lies past its bound
/// before a row to reduce or a combination (`Budget::outworked`), or,
/// where the round reduces in sparse rows, projected over its rows to
/// reduce passes the bound (`Reducer::within_bound`), and a stop where the
/// budget's check stops the call. The back-substitution
/// reads no bound. The dense row and the work vectors come from `bufs`,
/// and go back to it where the reduction returns its result, a misfit, or
/// `Halt::Outworked`.
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
    reason = "the field, basis, matrix, and cofactors, the budget, whether the reduction is full, the combinations, the buffers, the tests, and the modes, none of which belongs with another"
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
    mut modes: Option<&mut Modes>,
) -> Result<Reduction<F>, Halt> {
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    // call-stop-latency.53 design-checked -- the reducer's arrays spend once filled, the dense row and the pivot array the matrix's columns and the band BAND units a column, as a row's scan spends its columns once, not unspent (not measured)
    budget.spend(matrix.columns(), Site::Setup)?;
    // call-stop-latency.54 design-checked -- a reduction's loops over its rows spend as they go, RUN rows at a time: PROBE units a row whose pivot they set or read, a unit a row of the order searched for its first reductant, and PROBE units a row the order passes without reducing it, not the matrix's rows and columns spent once after the setup (matrix-stop-latency.8) (not measured)
    for start in (matrix.given..matrix.rows.len()).step_by(RUN) {
        let end = (start + RUN).min(matrix.rows.len());
        for r in (start..end).map(RowId::at) {
            let c = matrix.lead(r).index();
            debug_assert!(matches!(red.pivot[c], Pivot::None), "one reducer a column");
            red.pivot[c] = Pivot::Row(r);
        }
        budget.spend((end - start) * crate::budget::PROBE, Site::Setup)?;
    }
    // f4-gfp.5 design-checked -- where several rows can be a column's pivot, the one with the fewest terms is, ties going to the element made first and to the first pair row, not the first divisor in the order of G and the first pair row (not measured)
    let mut keys: Vec<u128> = Vec::new();
    for start in (0..matrix.given).step_by(RUN) {
        let end = (start + RUN).min(matrix.given);
        for r in (start..end).map(RowId::at) {
            let lead = matrix.lead(r);
            let c = lead.index();
            match red.pivot[c] {
                Pivot::Row(q) if matrix.row(q).len() > matrix.row(r).len() => {
                    keys.push(lead_key(lead, q.raw()));
                    red.pivot[c] = Pivot::Row(r);
                }
                Pivot::Row(_) => keys.push(lead_key(lead, r.raw())),
                Pivot::None | Pivot::New(_) => red.pivot[c] = Pivot::Row(r),
            }
        }
        budget.spend((end - start) * crate::budget::PROBE, Site::Setup)?;
    }
    // cyclic-gap.1 design-checked -- the rows to reduce go by decreasing leading column, those of one leading column by decreasing row index, not by increasing leading column nor the sparsest first (log 0066-1)
    // call-stop-latency.18 design-checked -- the rows to reduce, the made rows the back-substitution takes, and the inter-reduction's rows are sorted by sort::keys on their leading columns above their ids, and the signature round's scan of a run spends PROBE units a position, RUN positions at a time, not sort_unstable before one spend (matrix-stop-latency.8) nor the scan unspent (not measured)
    sort::keys(&mut keys, true, budget, Site::Setup, no_piece)?;
    let mut rest = keyed_rows(&keys, budget, Site::Setup)?;
    drop(keys);
    // gf2-reduction-time.12 design-checked -- the packed reduction runs where the field packs and no cofactors are tracked, and a round that tracks cofactors reduces as it is, not the cofactors tracked through the packed rows (not measured)
    // Every row that is no row to reduce has set its column's pivot.
    if F::PACKS
        && !C::TRACKS
        && let Some(words) = packed::packs(
            matrix.columns(),
            matrix.rows.len(),
            matrix.entries(),
            matrix.rows.len() - rest.len(),
        )
    {
        red.pack(words);
    }
    let mut tally = Tally {
        zero: Vec::new(),
        reductions: 0,
        zeros: 0,
    };
    let p = field.characteristic();
    // The rows to reduce, and whether the round probed (`Modes::decide`).
    let to_reduce = rest.len();
    red.start = budget.row_work();
    red.to_reduce = to_reduce;
    let mut probed = false;
    let mut bands = false;
    let mut sketched = None;
    let mode = match combine {
        Some(combine) if combines_rows(p) => {
            // gfp-unsolved.8 design-checked -- a run that tracks cofactors reduces each row on its own, not combinations with their cofactors tracked through a hook that begins an empty combination and adds its rows (log 0014-1)
            debug_assert!(!C::TRACKS, "no cofactors of a combination");
            // gfp-unsolved.2 cited-checked: monagan-2017 §2, via berthomieu-2021 §4.2 and demin-2024 §4.2, attributes it to A. Steel's private communication and found it independently -- the rows are partitioned into sets, random linear combinations of a set's rows are reduced, and zero comes by chance with probability about 1/p; berthomieu-2021 §4.2 -- blocks of ℓ rows, a random linear combination reduced until the first reduction to zero or ℓ of them
            // plain-reduction-time.5 design-checked -- a plain run's matrix takes bands for its combinations where its rows hold BAND_ENTRIES entries or more in the mean, as a signature round's, not in every matrix (log 0062-1)
            bands = takes_bands::<F, M, C>(matrix);
            // short-row-reduce-time.1 invented-checked -- a plain round whose matrix takes no bands, in a run that combines rows, reduces the rows after its first block either as random combinations of blocks or each on its own with the zero tests, chosen per round, not the blocks' combinations in every such round (berthomieu-2021 §4.2, monagan-2017 §2) nor each row on its own in every such round
            if bands || rest.len() < BLOCK + PROBE {
                red.reduce_blocks(&rest, BLOCK, bands, combine, cof, budget, bufs, &mut tally)?;
                Some(RoundMode::Blocks)
            } else {
                // The reductions, subtracted entries, and pivots before the
                // first block.
                let before = (tally.reductions, budget.counts().0, red.made.len());
                let size = red.reduce_blocks(
                    &rest[..BLOCK],
                    BLOCK,
                    false,
                    combine,
                    cof,
                    budget,
                    bufs,
                    &mut tally,
                )?;
                let mode = if let Some(forced) = forced_mode() {
                    forced
                } else if let Some(modes) = modes.as_deref_mut().filter(|m| m.probes(to_reduce)) {
                    // short-row-reduce-time.2 design-checked -- the round's first block is reduced as combinations and gives c₆₄, the entries subtracted over its reductions, and its pivots, then PROBE rows of the next block, those leading farthest left, are reduced on their own and give c₁, not the probe's rows from the block's right end, a whole block on its own, nor no probe with the previous round's figures alone
                    let (first_reductions, first_subtracted, first_pivots) = (
                        tally.reductions - before.0,
                        budget.counts().0 - before.1,
                        red.made.len() - before.2,
                    );
                    let end = rest.len().min(2 * BLOCK);
                    let probe_rows: Vec<RowId> = rest.drain(end - PROBE..end).collect();
                    let subtracted = budget.counts().0;
                    // short-row-reduce-time.8 design-checked -- the probe's rows on their own are counted and reported as the rows' mode counts them, each a reduction, a zero where it reduces to zero, and no test among them, not left out of the counts
                    red.reduce_rows_alone(&probe_rows, Zeros::None, cof, budget, bufs, &mut tally)?;
                    probed = true;
                    modes.decide(
                        to_reduce,
                        &Probe {
                            reductions: first_reductions,
                            subtracted: first_subtracted,
                            pivots: first_pivots,
                            alone: budget.counts().0 - subtracted,
                        },
                    )
                } else {
                    modes
                        .as_deref()
                        .and_then(|m| m.decision)
                        .unwrap_or(RoundMode::Blocks)
                };
                match mode {
                    RoundMode::Blocks => {
                        red.reduce_blocks(
                            &rest[BLOCK..],
                            size,
                            false,
                            combine,
                            cof,
                            budget,
                            bufs,
                            &mut tally,
                        )?;
                    }
                    RoundMode::Rows => {
                        // short-row-reduce-time.5 design-checked -- in the rows' mode of a call run the tests' combinations are drawn from the run's Combine, the draw of its blocks' combinations, not a Combine of its own nor no tests
                        let zeros = zero_combinations(p)
                            .map_or(Zeros::None, |s| Zeros::Tests(&mut *combine, s));
                        red.reduce_rows_alone(
                            &rest[BLOCK..],
                            zeros,
                            cof,
                            budget,
                            bufs,
                            &mut tally,
                        )?;
                    }
                }
                Some(mode)
            }
        }
        // gf2-forced-zeros-time.11 design-checked -- the sketches run in a plain round whose run may combine rows, no trace or learner listening and no cofactors tracked, over a field that packs, GF(2), not over GF(p) with sketches in GF(p), where the combinations and the tests run, nor in the signature run, whose zero rows are its syzygies (log 0134-2)
        Some(combine) if F::PACKS && sketches(matrix) => {
            sketched = red.reduce_rows_alone(
                &rest,
                Zeros::Sketches(combine),
                cof,
                budget,
                bufs,
                &mut tally,
            )?;
            None
        }
        Some(_) | None => {
            // learn-zero-time.1 invented-checked -- a learn run's plain run, where no cofactors are tracked, rules its rows zero by signature-zero-time.1's test of random combinations, the rows a test rules zero recorded as rows that reduced to zero, not each row reduced on its own (log 0063-1)
            // learn-zero-time.6 design-checked -- over two lanes a test takes the s of the first lane's prime, the field's characteristic, not the s of the product of the lanes' primes (not measured)
            let zeros = match tests.filter(|_| !C::TRACKS).zip(zero_combinations(p)) {
                Some((combine, s)) => Zeros::Tests(combine, s),
                None => Zeros::None,
            };
            red.reduce_rows_alone(&rest, zeros, cof, budget, bufs, &mut tally)?;
            None
        }
    };
    if let Some(modes) = modes {
        modes.ended(to_reduce, red.made.len(), probed);
    }
    if full {
        budget.mark(Stage::Reduce);
        // gf2-packed-backsub-time.2 design-checked -- a round that reduces in packed rows back-substitutes in the packed rows it keeps of its made rows, and a round in sparse rows in bands, not every round over GF(2) in packed rows of its own over its made rows' columns, which cost more than the round's own on hfe and mqbool and gained sr's wide rounds little (log 0135-1, 0152-1, 0152-2)
        match red.packed {
            Some(_) => red.back_substitute_packed(budget)?,
            None => red.back_substitute(cof, budget)?,
        }
        budget.mark(Stage::Backsub);
    }
    red.give_back(bufs);
    let packed = red.packs.filter(|_| red.packed.is_some());
    Ok(Reduction {
        made: red.made,
        zero: tally.zero,
        pivot: red.pivot,
        taken: Vec::new(),
        reductions: tally.reductions,
        zeros: tally.zeros,
        mode,
        bands,
        packed,
        sketches: sketched,
    })
}

// signature-pruning.12 cited-checked: lairez-2022 §5.7 -- the reductants of a batch of signatures are reduced in increasing order of signature, each modulo AG^{<σ} ∪ N, the batch's new elements N used but not their multiples, which join G only after the batch
/// Reduces the matrix of a signature round, whose rows have the signatures
/// `sigs`, in increasing order of signature. A reducer row becomes the
/// pivot of its leading column where that column has none. Where it has
/// one, the reducer row is dropped where the row before it has its
/// signature, and is reduced as a reductant otherwise. A given row, a
/// reductant, is reduced by the pivots set before it. Where the field
/// packs (`Field::PACKS`), `C` tracks no cofactors, and the round's rows,
/// entries, columns, and reducer rows before its first reductant fit
/// (`packed::packs`), every row is reduced in packed rows
/// (`Reducer::reduce_packed`), and a reducer row that sets its pivot after
/// the first reductant takes its packed row then (`Reducer::set_row`).
/// Where the round takes no packed
/// rows and `takes_bands` holds, a run of consecutive reductants is reduced
/// in bands of up to
/// `BAND` rows (`band_end`, `Reducer::reduce_band`), each row as it is
/// reduced alone. Each reduction is a tick of the budget. The sort by
/// signature spends as `by_signature` does, and at `Site::Setup` the
/// matrix's columns once the reducer is made, the search for the first
/// reductant a unit a row, and each row the order passes without reducing
/// it `budget::PROBE` units. Each row operation spends its row's length
/// (`Reducer::cancel`), in a band once per band row it is subtracted from
/// (`Reducer::reduce_band`), each scan of a row the columns it passes,
/// and each packed row added its entries (`packed::Rows`). The new pivots
/// each make a
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
/// rows to be reduced as the run's other rows are. The scan of the run
/// for its rest spends `budget::PROBE` units a position, `RUN` positions
/// at a time, at `Site::SigScan`.
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
    // signature-pruning.13 design-checked -- a reductant comes before every reducer of its own signature (not measured)
    // The reductants are the given rows, the first: at equal signatures the
    // row decides, and a reductant comes first.
    debug_assert_eq!(sigs.len(), matrix.rows.len());
    let rows = by_signature(table, order, sigs, budget, Site::SigSort)?;
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    // call-stop-latency.53 design-checked -- the reducer's arrays spend once filled, the dense row and the pivot array the matrix's columns and the band BAND units a column, as a row's scan spends its columns once, not unspent (not measured)
    budget.spend(matrix.columns(), Site::Setup)?;
    // The reducer rows before the first reductant set their pivots before
    // any row is reduced: no column has a pivot yet, and each leads a
    // column of its own.
    // call-stop-latency.54 design-checked -- a reduction's loops over its rows spend as they go, RUN rows at a time: PROBE units a row whose pivot they set or read, a unit a row of the order searched for its first reductant, and PROBE units a row the order passes without reducing it, not the matrix's rows and columns spent once after the setup (matrix-stop-latency.8) (not measured)
    let mut early = rows.len();
    for (k, chunk) in rows.chunks(RUN).enumerate() {
        budget.spend(chunk.len(), Site::Setup)?;
        if let Some(at) = chunk.iter().position(|r| r.index() < matrix.given) {
            early = k * RUN + at;
            break;
        }
    }
    // gf2-reduction-time.12 design-checked -- the packed reduction runs where the field packs and no cofactors are tracked, and a round that tracks cofactors reduces as it is, not the cofactors tracked through the packed rows (not measured)
    let packs = if F::PACKS && !C::TRACKS {
        packed::packs(matrix.columns(), matrix.rows.len(), matrix.entries(), early)
    } else {
        None
    };
    if let Some(words) = packs {
        red.pack(words);
    }
    let mut zero = Vec::new();
    let (mut reductions, mut zeros) = (0, 0);
    // gf2-reduction-time.15 design-checked -- a signature round that reduces in packed rows takes no band, not bands of packed rows (log 0125-3)
    // A band reads a sparse pivot row once for four rows, where a packed
    // addition is a pass over words that four rows' accumulators would read
    // four times.
    let bands = packs.is_none() && takes_bands::<F, M, C>(matrix);
    let p = field.characteristic();
    // signature-zero-time.10 design-checked -- the tests run where no cofactors are tracked, neither the trace nor the learner listens, and the field has one lane, not in a learn run as well (log 0049-1)
    let mut test = combine
        .filter(|_| !C::TRACKS && tests_rows::<F>(p))
        .zip(zero_combinations(p));
    // The run's rows reduced to zero one after another, alone or in a
    // band, and whether a test of the run has failed.
    let (mut streak, mut failed) = (0, false);
    // The positions of the run's rest in the order.
    let mut rest: Vec<usize> = Vec::new();
    // The rows passed without a reduction and not yet spent (`pass`).
    let mut passed = 0;
    let mut next = 0;
    while next < rows.len() {
        let start = next;
        let row_id = rows[start];
        // row-traffic-time.1 cited-checked: martani-2012 §3.5.1, via boyer-2016 §2 and §4.2 -- the multiline row reduction reduces two rows at the same time, each entry of the reducer read once for both
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
                budget.tick(Site::Row)?;
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
                if !red.keep(row, cof, budget)? {
                    red.give_back(bufs);
                    return Err(Halt::Misfit(0));
                }
            }
        } else {
            let lead = matrix.lead(row_id);
            if row_id.index() >= matrix.given {
                // call-stop-latency.54 design-checked -- a reduction's loops over its rows spend as they go, RUN rows at a time: PROBE units a row whose pivot they set or read, a unit a row of the order searched for its first reductant, and PROBE units a row the order passes without reducing it, not the matrix's rows and columns spent once after the setup (matrix-stop-latency.8) (not measured)
                if red.pivot[lead.index()].is_none() {
                    red.set_row(lead.index(), row_id, budget)?;
                    pass(&mut passed, budget)?;
                    // A run starts after the row.
                    (streak, failed) = (0, false);
                    continue;
                }
                // gfp-unsolved.13 cited-checked: eder-2017 §6.2, Lemma (singular criterion) -- for any signature T exactly one element needs to be handled
                if start > 0 && sigs[rows[start - 1].index()] == sigs[row_id.index()] {
                    pass(&mut passed, budget)?;
                    continue;
                }
                // gfp-unsolved.12 cited-checked: eder-2017 §13, Algorithm F4-RB -- the row echelon form of M_d computed without row swapping, and every row whose leading term changed joins the basis, or the syzygies where it is zero
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
            budget.tick(Site::Row)?;
            // signature-pruning.14 cited-checked: eder-2017 §14.3 (2) -- full s-reductions mostly behave better than only top s-reductions
            let row = red.reduce(row_id, lead, cof, budget)?;
            reductions += 1;
            if !row.cols.is_empty() {
                streak = 0;
                if !red.keep(row, cof, budget)? {
                    red.give_back(bufs);
                    return Err(Halt::Misfit(0));
                }
                continue;
            }
            zero.push(row_id);
            zeros += 1;
            streak += 1;
        }
        // signature-zero-time.13 design-checked -- a band's rows count in the run's streak, and where it stands at STREAK or more after the band's last row a test of the rest after the band starts, not the streak counting the rows reduced alone, a band's rows leaving it at 0 (log 0061-1)
        let Some((combine, s)) = test.as_mut() else {
            continue;
        };
        if streak < STREAK {
            continue;
        }
        streak = 0;
        let s = *s;
        // signature-zero-time.2 design-checked -- a test combines rows of one run, which have the same pivots before them, not rows across a reducer row that sets a pivot (not measured)
        // The rest: the rows from `next` up to the first reducer row that
        // sets a pivot, or the round's end, the rows dropped there left out.
        rest.clear();
        let mut run_end = rows.len();
        // call-stop-latency.18 design-checked -- the rows to reduce, the made rows the back-substitution takes, and the inter-reduction's rows are sorted by sort::keys on their leading columns above their ids, and the signature round's scan of a run spends PROBE units a position, RUN positions at a time, not sort_unstable before one spend (matrix-stop-latency.8) nor the scan unspent (not measured)
        let mut from = next;
        while from < run_end {
            let end = (from + RUN).min(run_end);
            for position in from..end {
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
            budget.spend((end - from) * crate::budget::PROBE, Site::SigScan)?;
            from = end;
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
    budget.spend(passed * crate::budget::PROBE, Site::Setup)?;
    red.give_back(bufs);
    let packed = red.packs.filter(|_| red.packed.is_some());
    Ok(Reduction {
        made: red.made,
        zero,
        pivot: red.pivot,
        taken: rows,
        reductions,
        zeros,
        mode: None,
        bands,
        packed,
        sketches: None,
    })
}

// trace-replay.16 cited-checked: demin-2024 §4.4 -- the apply stage mirrors the F4 computation without the redundant rows, and caches the order of rows and columns in the matrices; parisse-2013 §4, step 8, independently -- care must be taken to keep row ordering when learning is active
/// Reduces a replayed matrix, its rows in order: a row not in `reductees`
/// becomes the pivot of its leading column, and a row in it is reduced by
/// the pivots set before it, a tick of the budget each. `reductees` is
/// increasing, and `leads` begins with the monomial each must reduce to. Where
/// `full` is true the rows made are then inter-reduced, and the report's
/// stages marked, as `reduce` does.
/// The setup spends the matrix's columns once the reducer is made, and
/// each pivot row `budget::PROBE` units, at `Site::Setup`; each row
/// operation spends its row's length (`Reducer::cancel`). A stop where the
/// budget's check stops the call.
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
    // call-stop-latency.53 design-checked -- the reducer's arrays spend once filled, the dense row and the pivot array the matrix's columns and the band BAND units a column, as a row's scan spends its columns once, not unspent (not measured)
    budget.spend(matrix.columns(), Site::Setup)?;
    let mut next = reductees.iter().zip(leads).peekable();
    // The pivot rows set and not yet spent (`pass`).
    let mut passed = 0;
    for r in (0..matrix.rows.len()).map(RowId::at) {
        let c = matrix.lead(r);
        let Some((_, &lead)) = next.next_if(|&(&q, _)| q == r) else {
            // call-stop-latency.54 design-checked -- a reduction's loops over its rows spend as they go, RUN rows at a time: PROBE units a row whose pivot they set or read, a unit a row of the order searched for its first reductant, and PROBE units a row the order passes without reducing it, not the matrix's rows and columns spent once after the setup (matrix-stop-latency.8) (not measured)
            pass(&mut passed, budget)?;
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
        budget.tick(Site::Row)?;
        let row = red.reduce(r, c, cof, budget)?;
        // trace-replay.6 design-checked -- a replay checks each counted row's result against the recorded leading monomial and stops at the first that differs, not the leading monomials of the output compared with the learn run's after the replay (not measured)
        if row.cols.first().map(|&c| matrix.monomials[c.index()]) != Some(lead) {
            red.give_back(bufs);
            return Ok(None);
        }
        // q-later-runs-time.4 design-checked -- a pair fits only where every counted row's leading entry is a unit at both primes, and where it does not each prime replays alone, not the pair a misfit at both primes nor the lane that fits taken from the pair's run (not measured)
        if !field.is_unit(row.coefs[0]) {
            red.give_back(bufs);
            return Ok(None);
        }
        let kept = red.keep(row, cof, budget)?;
        debug_assert!(kept, "a unit leads the row");
    }
    budget.spend(passed * crate::budget::PROBE, Site::Setup)?;
    if full {
        budget.mark(Stage::Reduce);
        red.back_substitute(cof, budget)?;
        budget.mark(Stage::Backsub);
    }
    red.give_back(bufs);
    Ok(Some(red.made))
}

// f4-gfp.21 design-checked -- the reduced basis comes from one more matrix, each element's tail reduced as in a round, not from division by the others one element at a time (not measured)
/// Reduces the tails of the given rows of a matrix built from `G_min`, each
/// row the pivot of its leading column. Each given row, in order, ticks the
/// budget, and one whose tail holds no column with a pivot is taken as it
/// stands. The others are reduced from the column after their leads
/// (`Reducer::reduce`), in order. Where the reduction takes bands
/// (`Reducer::bands_in_echelon`), they are taken by increasing leading
/// column instead, `BAND` at a time in one band sweep
/// (`Reducer::sweep_led`) where the reducer's counts choose the dense scan
/// for the next row and two rows or more are left, else the next row on
/// its own; each band row is counted after the sweep, in band order, as a
/// row of the dense scan from the column after its lead (`Reducer::count`).
/// The result holds every given row once, monic as
/// it was, in the order taken, each kept by the cofactor tracking in that
/// order. The setup spends the matrix's columns once the reducer is made
/// and the pivots `budget::PROBE` units a row, and the sort of the rows by
/// increasing leading column as `sort::keys` spends, at `Site::Setup`.
/// Each given row's test and copy spend its length at `Site::Inter`, where
/// each reduction, band or row, ticks the budget, and each row operation
/// spends its row's length (`Reducer::cancel`), in a band once per band
/// row it is subtracted from, and each scan of a row the columns it passes;
/// a stop where the budget's check stops the call. The dense row, the
/// band, and the work vectors come from `bufs`, and go back to it where the
/// reduction returns its result.
pub(crate) fn inter_reduce<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    basis: &Basis<F, D>,
    matrix: &Matrix,
    cof: &mut C,
    budget: &mut Budget,
    bufs: &mut Buffers<F>,
) -> Result<Vec<Reduced<F>>, Stop> {
    let mut red = Reducer::new(field, basis, matrix, bufs, budget);
    // call-stop-latency.53 design-checked -- the reducer's arrays spend once filled, the dense row and the pivot array the matrix's columns and the band BAND units a column, as a row's scan spends its columns once, not unspent (not measured)
    budget.spend(matrix.columns(), Site::Setup)?;
    // call-stop-latency.54 design-checked -- a reduction's loops over its rows spend as they go, RUN rows at a time: PROBE units a row whose pivot they set or read, a unit a row of the order searched for its first reductant, and PROBE units a row the order passes without reducing it, not the matrix's rows and columns spent once after the setup (matrix-stop-latency.8) (not measured)
    for start in (0..matrix.rows.len()).step_by(RUN) {
        let end = (start + RUN).min(matrix.rows.len());
        for r in (start..end).map(RowId::at) {
            let c = matrix.lead(r).index();
            debug_assert!(matches!(red.pivot[c], Pivot::None), "one pivot a column");
            red.pivot[c] = Pivot::Row(r);
        }
        budget.spend((end - start) * crate::budget::PROBE, Site::Setup)?;
    }
    let mut out = Vec::with_capacity(matrix.given);
    // The keys of the rows that need reduction (`lead_key`), in row order.
    let mut work: Vec<u128> = Vec::new();
    for r in (0..matrix.given).map(RowId::at) {
        budget.tick(Site::Inter)?;
        // back-substitution-time.8 design-checked -- a given row whose tail holds no column with a pivot is taken as it stands, in every field and with cofactors too, not loaded and scanned
        // Such a row is reduced already, and its load and scan would cost
        // the matrix's width.
        let row = matrix.row(r);
        let reduced = by_width!(row.tail(), |tail| tail
            .iter()
            .all(|c| red.pivot[c.column()].is_none()));
        // call-stop-latency.55 design-checked -- the inter-reduction's test of a given row's tail for a pivot and the copy of a row taken as it stands spend a unit an entry beside the row's tick, the test the row's length, not the tick alone (not measured)
        let copied = if reduced { row.len() } else { 0 };
        budget.spend(row.len() + copied, Site::Inter)?;
        if reduced {
            cof.start(r);
            cof.keep(budget)?;
            out.push(Reduced {
                row: r,
                cols: row.to_wide(),
                coefs: red.coefs(r).to_vec(),
                kept: KeepId::at(out.len()),
            });
        } else {
            work.push(lead_key(row.lead(), r.raw()));
        }
    }
    // back-substitution-time.5 cited-checked: martani-2012 §3.5.1.1 -- in the Trsm step a multiline of two rows is reduced by one pivot vector, its index and value fetched once and used twice; boyer-2016 §4.2, independently -- the multiline optimizes the AXPY and TRSM computations of the reduction's second step
    if !red.bands_in_echelon::<M, C>() {
        for &key in &work {
            // call-stop-latency.17 design-checked -- the back-substitution's band path ticks each band and each row it takes alone, and the inter-reduction's reductions each row and band, not their ticks spent in a first loop over the rows (back-substitution-time.1) (not measured)
            budget.tick(Site::Inter)?;
            let row = red.reduce_given(RowId::from_raw(payload(key)), out.len(), cof, budget)?;
            out.push(row);
        }
        red.give_back(bufs);
        return Ok(out);
    }
    // back-substitution-time.7 design-checked -- the given rows that need reduction go by increasing leading column, not in the order made
    // Four rows of near leads sweep few columns before the band's later
    // leads.
    // call-stop-latency.18 design-checked -- the rows to reduce, the made rows the back-substitution takes, and the inter-reduction's rows are sorted by sort::keys on their leading columns above their ids, and the signature round's scan of a run spends PROBE units a position, RUN positions at a time, not sort_unstable before one spend (matrix-stop-latency.8) nor the scan unspent (not measured)
    sort::keys(&mut work, false, budget, Site::Setup, no_piece)?;
    let work = keyed_rows(&work, budget, Site::Setup)?;
    let mut next = 0;
    while next < work.len() {
        let n = BAND.min(work.len() - next);
        // call-stop-latency.17 design-checked -- the back-substitution's band path ticks each band and each row it takes alone, and the inter-reduction's reductions each row and band, not their ticks spent in a first loop over the rows (back-substitution-time.1) (not measured)
        budget.tick(Site::Inter)?;
        // back-substitution-time.6 design-checked -- a band where the reducer's counts choose the dense scan for the next row, and the next row alone with the scan they choose where they choose the marked scan, each band row counted as a row of the dense scan, not a band for every four rows, nor bands only in a matrix of BAND_ENTRIES entries a row and a mean row of a 32nd of its width or more
        // A band's sweep reads every column from its least lead on, where
        // the marked scan of a sparse row in a wide matrix reads its marked
        // columns alone; martani-2012 §3.5.2 found the multiline no faster
        // on a matrix of 5 % density.
        if red.marked || n == 1 {
            let row = red.reduce_given(work[next], out.len(), cof, budget)?;
            out.push(row);
            next += 1;
            continue;
        }
        let band = &work[next..next + n];
        let rows: Vec<Led> = band.iter().map(|&r| Led::Given(r)).collect();
        let swept = red.sweep_led(&rows, budget, Site::Inter)?;
        for (row, &r) in red.end_bands(swept).into_iter().zip(band) {
            cof.keep(budget)?;
            out.push(Reduced {
                kept: KeepId::at(out.len()),
                ..row
            });
            // back-substitution-time.10 design-checked -- each band row is counted after the sweep, in band order, by the columns from the one after its lead and the entries the budget counted since the last count, not its load's length as well, as `reduce` counts a row's load
            red.count(MonoId::at(matrix.lead(r).index() + 1), budget);
        }
        next += n;
    }
    red.give_back(bufs);
    Ok(out)
}
