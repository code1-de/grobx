//! The packed rows of a round over GF(2) (gf2-reduction-time.5 to .16). The
//! pivots a round has set when it reduces its first row are its early
//! pivots, and their leading columns its A columns; every other column is a
//! K column. Each early pivot is reduced by the others once, to its reduced
//! row over the K columns alone, and each pivot set later takes its packed
//! row when it is set. A row is then reduced by adding the reduced row of
//! each early pivot it meets once, and the packed row of each later pivot
//! its scan of the K columns meets. The made rows are back-substituted in
//! the packed rows they take when they are made, by tables of the sums of
//! a few rows each (gf2-packed-backsub-time.1 to .9). Every coefficient is
//! 1: the caller's field packs (`Field::PACKS`). The module reads its rows
//! through the caller, and knows no matrix.

#[cfg(test)]
use std::cell::Cell;

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::columns::{Row, by_width};
use crate::id::{Column, MonoId, RowId};
use crate::narrow::Narrow;

/// The K columns a word of a packed row holds: K column 64w + b is bit b
/// of word w.
const WORD: usize = u64::BITS as usize;

// gf2-reduction-time.13 tuned-checked -- a round reduces in packed rows where its K columns take at most 256 words, not 64, 128, or 512 words (log 0125-1, 0125-4)
/// The most words a round's K columns take where the round reduces in
/// packed rows: 256. In session 0125-4 a bound of 64 or 128 words left out
/// rounds of bivium, hfe, and mqbool that took less time packed, and one
/// of 512 words packed no round of the tuning set that 256 left out. In the
/// call reports of session 0125-1 the widest rounds of sr subtracted fewer
/// entries than a packed reduction would XOR words.
pub(crate) const PACKED_WORDS: usize = 256;

// gf2-reduction-time.21 tuned-checked -- a round reduces in packed rows only where its matrix holds 100,000 entries or more, not at every size, nor from 1,000,000 entries (log 0125-3)
/// The fewest entries a round's matrix holds where the round reduces in
/// packed rows: 100,000. In session 0125-3 the smaller rounds over GF(2)
/// took as long packed as sparse, and where they packed, the entries their
/// reductions subtracted moved the pair-work handover of sr-12 and sr-15
/// (pair-work-ratio.1), whose rounds there lie that small. The rounds of
/// 100,000 to 1,000,000 entries took less time packed.
pub(crate) const PACKED_ENTRIES: usize = 100_000;

// gf2-reduction-time.22 tuned-checked -- a round reduces in packed rows only where its rows hold 16 entries or more in the mean, not at any mean, nor from 32 entries (log 0125-3)
/// The fewest entries a round's rows hold in the mean where the round
/// reduces in packed rows: 16. In session 0125-3 the rounds of shorter
/// rows over GF(2), all of sat, took longer packed and held their packed
/// rows' memory, and bivium's rounds of degree 3, of fewer than 32 entries
/// a row, gained the most.
pub(crate) const PACKED_ROW_ENTRIES: usize = 16;

// gf2-reduction-time.14 design-checked -- a round reduces in packed rows only where its early pivots times its K columns' words stay at most 2^25 words, 256 MiB, not with no bound on the early pivots' reduced rows (log 0125-3)
/// The most words the reduced rows of a round's early pivots may take at
/// the K columns' whole width where the round reduces in packed rows:
/// 2^25, 256 MiB.
pub(crate) const PACKED_TABLE: usize = 1 << 25;

// gf2-packed-backsub-time.4 tuned-checked -- a table of the back-substitution holds the sums of 8 made rows, not of 4, 6, or 10 (log 0135-1, 0152-1, 0152-2)
/// The made rows a table of the back-substitution sums
/// (`Rows::back_substitute`): 8, a table of 2^8 sums. In session 0135-1
/// tables of 4, 6, 8, and 10 rows took hfe and mqbool within the spread of
/// a call of each other, 6 and 8 the least.
pub(crate) const TABLE_ROWS: usize = 8;

#[cfg(test)]
thread_local! {
    // gf2-reduction-time.19 design-checked -- in the test profile a thread-local fixes whether a round reduces in packed rows, so that test code compares the two reductions on one round, not a parameter of the call nor the bounds lowered (not measured)
    /// The test profile's choice on this thread, where Some: whether every
    /// round over a field that packs and tracks no cofactors reduces in
    /// packed rows, in place of the bounds (`packs`).
    pub(crate) static PACK_FORCED: Cell<Option<bool>> = const { Cell::new(None) };
}

/// The words of the K columns where a round of `columns` columns, `rows`
/// rows, `entries` entries, and `early` early pivots reduces in packed
/// rows: where its entries number `PACKED_ENTRIES` or more and
/// `PACKED_ROW_ENTRIES` times its rows or more, its K columns' words at
/// most `PACKED_WORDS`, and the early pivots times those words at most
/// `PACKED_TABLE`; None where it does not. In the test profile
/// `PACK_FORCED` decides where it holds a choice.
pub(crate) fn packs(columns: usize, rows: usize, entries: usize, early: usize) -> Option<u32> {
    debug_assert!(
        early <= columns && early <= rows,
        "{early} early pivots past the {columns} columns or the {rows} rows"
    );
    let words = (columns - early).div_ceil(WORD);
    let fits = entries >= PACKED_ENTRIES
        && rows
            .checked_mul(PACKED_ROW_ENTRIES)
            .is_some_and(|least| entries >= least)
        && words <= PACKED_WORDS
        && early
            .checked_mul(words)
            .is_some_and(|table| table <= PACKED_TABLE);
    #[cfg(test)]
    let fits = PACK_FORCED.get().unwrap_or(fits);
    // A matrix's columns number below 2^32 (`MonoId`).
    fits.then(|| words.narrow())
}

// gf2-reduction-time.6 practice-checked: bit-packed rows over GF(2) -- a packed row holds 64 K columns to a u64 word, column 64w + b at bit b of word w, and an addition XORs its words into the accumulator's
// gf2-reduction-time.10 cited-checked: makarim-2017 §4.1, §5.2 -- operations on tails are more efficient where the tails are stored as truncated coefficient vectors relative to a global list of non-reducible monomials, truncated by removing trailing zeros
/// A packed row: its words `Rows::words[at..at + len]`, the first of them
/// word `word` of the K columns, and the entries they hold.
#[derive(Clone, Copy, Debug)]
struct Packed {
    at: usize,
    word: u32,
    len: u32,
    entries: u32,
}

/// What a column of the matrix is to the packed rows: an A column, which
/// the early pivot row it holds leads, or a K column, with its index among
/// the K columns.
#[derive(Clone, Copy, Debug)]
enum Slot {
    A(RowId),
    K(u32),
}

/// A round's packed rows (gf2-reduction-time.5).
pub(crate) struct Rows {
    /// Per column of the matrix, what it is to the packed rows.
    slot: Vec<Slot>,
    /// Per K column, its column of the matrix.
    column: Vec<MonoId>,
    /// Per column of the matrix, the reduced row of the early pivot that
    /// leads it, once computed: the pivot row's tail over the K columns
    /// alone, with no entry at an A column. None at a K column, and at an A
    /// column until computed.
    early: Vec<Option<Packed>>,
    /// Per K column, the packed row of the pivot set there after the
    /// round's first reduced row, a later reducer row's or a made row's,
    /// with its leading entry; None where no pivot was set there since.
    later: Vec<Option<Packed>>,
    // gf2-reduction-time.20 design-checked -- the scan finds the bits of the later pivots by a word of a bit per K column ANDed with the accumulator's word, and passes a word's other bits at once, not each set bit's later pivot read from `later` (log 0125-3)
    /// A bit per K column, at its place in the accumulator's words, set
    /// where `later` holds a packed row.
    led: Vec<u64>,
    /// Every packed row's words, appended.
    words: Vec<u64>,
    /// The accumulator: a word per `WORD` K columns, zero between two rows.
    acc: Vec<u64>,
    /// The accumulator's words a row has touched, [lo, hi), none where
    /// lo ≥ hi.
    lo: usize,
    hi: usize,
    /// The walk's stack: the early pivot rows whose reduced row waits on
    /// others, each with the position in its tail its scan goes on from
    /// (`walk`).
    stack: Vec<(RowId, usize)>,
}

impl Rows {
    // gf2-reduction-time.7 design-checked -- the A columns are those whose pivot is set when the round reduces its first row, and every other column is a K column, the columns of a signature round's reducer rows set after its first reductant among them, not the columns that any reducer row of the matrix leads (log 0125-1, 0125-3)
    /// The packed rows of a round whose pivots, by column, are `pivots` when
    /// it reduces its first row: Some, the pivot row, at an A column, and
    /// None at a K column. The K columns take their indices in increasing
    /// column order. No reduced row is computed yet, and the accumulator is
    /// zero.
    pub(crate) fn new(pivots: impl ExactSizeIterator<Item = Option<RowId>>) -> Rows {
        let mut slot = Vec::with_capacity(pivots.len());
        let mut column = Vec::new();
        for (c, pivot) in pivots.enumerate() {
            if let Some(q) = pivot {
                slot.push(Slot::A(q));
                continue;
            }
            // A matrix's columns number below 2^32 (`MonoId`).
            slot.push(Slot::K(column.len().narrow()));
            column.push(MonoId::at(c));
        }
        let words = column.len().div_ceil(WORD);
        Rows {
            early: vec![None; slot.len()],
            later: vec![None; column.len()],
            led: vec![0; words],
            words: Vec::new(),
            acc: vec![0; words],
            lo: usize::MAX,
            hi: 0,
            stack: Vec::new(),
            slot,
            column,
        }
    }

    /// The words of the K columns.
    pub(crate) fn words(&self) -> usize {
        self.acc.len()
    }

    /// Reduces the row of the columns `cols` by the round's pivots as set
    /// now, `rows` reading a row of the matrix: the columns of the entries
    /// left join `out`, increasing. First the reduced row of each early
    /// pivot at an A column of the row that has none is computed (`reach`),
    /// while the accumulator is zero. Then each entry at an A column adds
    /// that column's reduced row, and each at a K column flips its bit
    /// (`load`). Then the scan visits the accumulator's bits in increasing
    /// order: a bit whose K column has a later pivot adds that pivot's
    /// packed row, which clears it, and every other bit is an entry of the
    /// result (`scan`). Each packed row added spends its entries (`add`).
    /// The accumulator is zero after it, and after a stop, which the
    /// budget's check returns.
    pub(crate) fn reduce<'m>(
        &mut self,
        cols: Row<'_>,
        rows: impl Fn(RowId) -> Row<'m>,
        out: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert!(
            self.lo >= self.hi && self.stack.is_empty(),
            "the accumulator is zero between two rows"
        );
        let done = by_width!(cols, |c| self.reduce_cols(c, &rows, out, budget));
        if done.is_err() {
            self.clear();
        }
        done
    }

    /// `reduce` over the columns `cols`, the accumulator not cleared after
    /// a stop.
    fn reduce_cols<'m, I: Column>(
        &mut self,
        cols: &[I],
        rows: &impl Fn(RowId) -> Row<'m>,
        out: &mut Vec<MonoId>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        self.reach(cols, rows, budget)?;
        self.load(cols, budget)?;
        self.scan(Some(out), budget)?;
        // The scan with a result clears every bit it visits.
        self.lo = usize::MAX;
        self.hi = 0;
        Ok(())
    }

    // gf2-reduction-time.9 design-checked -- a pivot set after the round's first row, a later reducer row or a made row, takes its packed row when its pivot is set, the reducer row's tail reduced by the pivots set before it and the made row from its entries, not the later reducer row's packed row computed where a row first meets it, nor the later pivots' rows reduced by their tails in the scan (log 0125-3)
    /// Stores the packed row of the reducer row of the columns `cols`, set
    /// as the pivot of its leading column after the round's first reduced
    /// row, `rows` reading a row of the matrix: its tail reduced as `reduce`
    /// reduces a row, the bits that no later pivot clears kept, and its
    /// leading entry. Every pivot it needs has its packed row then. The
    /// leading column is a K column: it had no pivot when the round reduced
    /// its first row. Spends as `reduce` spends, and a stop where the
    /// budget's check stops the call, the accumulator then zero.
    pub(crate) fn set_later<'m>(
        &mut self,
        cols: Row<'_>,
        rows: impl Fn(RowId) -> Row<'m>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert!(
            self.lo >= self.hi && self.stack.is_empty(),
            "the accumulator is zero between two rows"
        );
        let done = self.later_row(cols, &rows, budget);
        if done.is_err() {
            self.clear();
        }
        done
    }

    /// `set_later`, the accumulator not cleared after a stop.
    fn later_row<'m>(
        &mut self,
        cols: Row<'_>,
        rows: &impl Fn(RowId) -> Row<'m>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let lead = self.k(cols.lead().index());
        by_width!(cols.tail(), |tail| {
            self.reach(tail, rows, budget)?;
            self.load(tail, budget)?;
        });
        self.scan(None, budget)?;
        self.flip(lead);
        self.store_later(lead);
        Ok(())
    }

    // gf2-reduction-time.9 design-checked -- a pivot set after the round's first row, a later reducer row or a made row, takes its packed row when its pivot is set, the reducer row's tail reduced by the pivots set before it and the made row from its entries, not the later reducer row's packed row computed where a row first meets it, nor the later pivots' rows reduced by their tails in the scan (log 0125-3)
    /// Stores the packed row of a made row of the columns `cols`, the
    /// pivot of its leading column, from its entries. Each is a K column:
    /// the scan left it, or no pivot led it when the round reduced its first
    /// row.
    pub(crate) fn keep(&mut self, cols: &[MonoId]) {
        debug_assert!(!cols.is_empty(), "a made row holds its leading entry");
        debug_assert!(
            self.lo >= self.hi && self.stack.is_empty(),
            "the accumulator is zero between two rows"
        );
        for &c in cols {
            let j = self.k(c.index());
            self.flip(j);
        }
        let lead = self.k(cols[0].index());
        self.store_later(lead);
    }

    /// The index among the K columns of column c of the matrix, a K column.
    fn k(&self, c: usize) -> usize {
        match self.slot[c] {
            Slot::K(j) => j as usize,
            #[allow(
                clippy::unreachable,
                reason = "a later pivot leads a column that had no pivot when the round reduced its first row, and a made row's entries lie at the columns its scan left, K columns all"
            )]
            Slot::A(q) => unreachable!(
                "column {c} is an A column, early pivot row {q}'s, which no later pivot or made row holds"
            ),
        }
    }

    // gf2-reduction-time.8 cited-checked: makarim-2017 §4, Algorithm 5, and §4.1 -- GETREDUCTOR computes a reductor where a term first needs it, by MULFULLREDUCE of the reducer's tail, and keeps it in M; tail-reducing all of M proactively is much less efficient than doing it lazily, as an element is retrieved
    /// Computes the reduced row of the early pivot at each A column of
    /// `cols` that has none (`walk`), `rows` reading a row of the matrix.
    /// The accumulator is zero before and after.
    fn reach<'m, I: Column>(
        &mut self,
        cols: &[I],
        rows: &impl Fn(RowId) -> Row<'m>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        for &c in cols {
            let c = c.column();
            match self.slot[c] {
                Slot::A(q) if self.early[c].is_none() => self.walk(q, rows, budget)?,
                Slot::A(_) | Slot::K(_) => {}
            }
        }
        Ok(())
    }

    /// Computes the reduced row of early pivot row q, and first those of
    /// the early pivots at the A columns of its tail that have none, each
    /// once. The stack takes q. The row on top scans its tail from where it
    /// stopped, and at the first A column whose reduced row is missing it
    /// waits on that column's row, which goes on top. A row whose tail
    /// misses none adds its tail into the accumulator (`load`), stores the
    /// result as its leading column's reduced row (`store`), and leaves the
    /// stack. A row's tail lies right of its lead, so no row waits on
    /// itself. The accumulator is zero before and after.
    fn walk<'m>(
        &mut self,
        q: RowId,
        rows: &impl Fn(RowId) -> Row<'m>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        self.stack.push((q, 0));
        while let Some((q, from)) = self.stack.pop() {
            let row = rows(q);
            if let Some((i, a)) = by_width!(row.tail(), |tail| self.missing(tail, from)) {
                self.stack.push((q, i + 1));
                self.stack.push((a, 0));
                continue;
            }
            by_width!(row.tail(), |tail| self.load(tail, budget))?;
            let reduced = self.store();
            self.early[row.lead().index()] = Some(reduced);
        }
        Ok(())
    }

    /// The first position from `from` on in `tail` of an A column whose
    /// reduced row is missing, with that column's early pivot row; None
    /// where every A column of the tail from there has its reduced row.
    fn missing<I: Column>(&self, tail: &[I], from: usize) -> Option<(usize, RowId)> {
        tail.iter().enumerate().skip(from).find_map(|(i, &c)| {
            let c = c.column();
            match self.slot[c] {
                Slot::A(a) if self.early[c].is_none() => Some((i, a)),
                Slot::A(_) | Slot::K(_) => None,
            }
        })
    }

    /// The accumulator gains the row of the columns `cols`: the reduced row
    /// of the early pivot at each A column, which `reach` or `walk`
    /// computed before, and the bit of each K column. Each reduced row added
    /// spends its entries (`add`), and a stop where the budget's check
    /// stops the call.
    fn load<I: Column>(&mut self, cols: &[I], budget: &mut Budget) -> Result<(), Stop> {
        for &c in cols {
            let c = c.column();
            match self.slot[c] {
                Slot::A(_) => {
                    #[allow(
                        clippy::expect_used,
                        reason = "`reach` computes the reduced row of every A column of a row before its load, and `walk` of every A column of a tail before its own"
                    )]
                    let reduced = self.early[c].expect("an A column's reduced row before its load");
                    self.add(reduced, budget)?;
                }
                Slot::K(j) => self.flip(j as usize),
            }
        }
        Ok(())
    }

    /// Flips the bit of K column j, whose word joins those touched.
    #[inline]
    fn flip(&mut self, j: usize) {
        let w = j / WORD;
        self.acc[w] ^= 1 << (j % WORD);
        self.lo = self.lo.min(w);
        self.hi = self.hi.max(w + 1);
    }

    // gf2-reduction-time.11 design-checked -- each packed row added spends its entries, counted once when the row is stored, as entries subtracted, not the words it XORs nor nothing (log 0125-3)
    /// The accumulator gains the packed row `p`: the exclusive or of its
    /// words into the accumulator's from its first word on, which join those
    /// touched. The
    /// row spends its entries, counted as subtracted, which keep the unit
    /// and about the size of a sparse row's that the run's handovers read,
    /// and adds its words to the row work (`Budget::spend_packed`). A stop
    /// where the budget's check stops the call.
    // gf2-packed-forward-time.1 cited-checked: martani-2012 §Inligning; bachmann-1998 §3, "Fourthly", independently -- the inline keyword is only a hint the compiler may decline, an attribute makes the inlining sure, and an operation inlined in place of a function call each gave a considerable efficiency gain
    #[inline(always)]
    fn add(&mut self, p: Packed, budget: &mut Budget) -> Result<(), Stop> {
        let (w, len) = (p.word as usize, p.len as usize);
        if len == 0 {
            return Ok(());
        }
        let row = &self.words[p.at..p.at + len];
        for (a, &x) in self.acc[w..w + len].iter_mut().zip(row) {
            *a ^= x;
        }
        self.lo = self.lo.min(w);
        self.hi = self.hi.max(w + len);
        budget.spend_packed(p.entries as usize, len)
    }

    /// Visits the accumulator's bits from word `lo` to `hi` in increasing
    /// order, each word read again after an addition: a bit whose K column
    /// has a later pivot adds that pivot's packed row (`add`), which clears
    /// it and changes only the bits after it. Where `out` is Some, every
    /// other bit is an entry of the result, its column pushed to `out` and
    /// the bit cleared; where it is None, those bits stay. A stop where the
    /// budget's check stops the call.
    fn scan(&mut self, mut out: Option<&mut Vec<MonoId>>, budget: &mut Budget) -> Result<(), Stop> {
        let mut w = self.lo;
        while w < self.hi {
            let word = self.acc[w];
            let hits = word & self.led[w];
            // The bits before the word's first later pivot, every bit where
            // it holds none.
            let free = if hits == 0 {
                word
            } else {
                word & (hits.isolate_lowest_one() - 1)
            };
            if let Some(out) = out.as_deref_mut() {
                let mut bits = free;
                while bits != 0 {
                    out.push(self.column[w * WORD + bits.trailing_zeros() as usize]);
                    bits &= bits - 1;
                }
                self.acc[w] ^= free;
            }
            if hits == 0 {
                w += 1;
                continue;
            }
            let j = w * WORD + hits.trailing_zeros() as usize;
            #[allow(
                clippy::expect_used,
                reason = "`store_later` sets a K column's bit of `led` as it stores its packed row"
            )]
            let row = self.later[j].expect("a later pivot's packed row beside its bit");
            self.add(row, budget)?;
        }
        Ok(())
    }

    // gf2-packed-backsub-time.1 practice-checked: bit-packed rows over GF(2) -- the back-substitution of a round in packed rows adds packed rows, 64 K columns to a word, not the made rows' sparse rows in bands of a byte an entry (back-substitution-time.1)
    // gf2-packed-backsub-time.3 cited-checked: bard-2006 §3 and §3.1, via albrecht-2011 §3.2 -- a table of all 2^k linear combinations of k pivot rows clears a row's k columns by one addition, as if Gaussian elimination had been performed, and run on the rows above the pivots as well it gives the reduced row echelon form
    // gf2-packed-backsub-time.5 design-checked -- each block's table is applied to every made row after the block and then dropped, so that one table stands at a time, not the tables of every block kept and each row back-substituted by them in turn (log 0152-1, 0152-2)
    // gf2-packed-backsub-time.8 design-checked -- a row added to a block row and a sum added to a row spend the entries of the rows they sum as subtracted and their words as row work, and a table's build the entries of the row each sum adds, not the words alone nor no spend, so that the count is the one-row back-substitution's and the tables' (log 0152-1, 0152-2)
    // gf2-packed-backsub-time.9 design-checked -- the back-substitution stands in this module beside the packed rows it reads, and the reducer writes the results back through a closure, not a module of its own nor the reducer's code (log 0152-1, 0152-2)
    /// Back-substitutes the made rows that lead the matrix columns `leads`,
    /// in decreasing order, each of which holds its packed row (`keep`):
    /// their reduced row echelon form, each row's bits at the other rows'
    /// leading columns cleared. The rows are taken `TABLE_ROWS` at a time,
    /// a block. Each block row adds the block rows before it that its bits
    /// at their leading columns name. Then a table holds the 2^k sums of
    /// the block's k rows, and each row after the block adds the sum its
    /// bits at the block's leading columns name. A back-substituted row holds
    /// no entry at another row's leading column, so a row's bits there stay
    /// as they were through the sums of other blocks, and name the rows the
    /// back-substitution of one row at a time adds. Each row whose result
    /// differs from its packed row ticks the budget at `Site::Row`, and
    /// `changed(i, tail)` hears its index in `leads` and the matrix columns
    /// of its result's tail, increasing, by increasing index. Each row added
    /// to a block row and each sum added to a row spends the entries of the
    /// rows it sums, counted as subtracted, and its words as row work
    /// (`Budget::spend_packed`): a table's sum is built from the sum without
    /// its lowest row and that row, and spends that row's entries. The
    /// packed rows stay as they were. A stop where the budget's check stops
    /// the call.
    pub(crate) fn back_substitute(
        &self,
        leads: &[MonoId],
        mut changed: impl FnMut(usize, &[MonoId]),
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        debug_assert!(
            leads.windows(2).all(|w| w[0] > w[1]),
            "the leading columns by decreasing column"
        );
        let mut rows = Echelon::new(self, leads);
        let n = leads.len();
        let words = self.acc.len();
        // A table's sums, `width` words each from the block's least
        // leading column's word, and the entries of the rows each sums.
        let mut table: Vec<u64> = Vec::new();
        let mut sums: Vec<usize> = Vec::new();
        for b in (0..n).step_by(TABLE_ROWS) {
            let e = n.min(b + TABLE_ROWS);
            for t in b..e {
                for u in b..t {
                    if rows.bit(t, rows.js[u]) {
                        let spent = rows.add_row(t, u);
                        budget.spend_packed(rows.entries[u], spent)?;
                    }
                }
                rows.count(t);
            }
            if e == n {
                break;
            }
            // The block's rows lead by decreasing column, so its last row's
            // leading column lies in the least word.
            let from = rows.first[e - 1];
            let width = words - from;
            let k = e - b;
            table.clear();
            table.resize(width << k, 0);
            sums.clear();
            sums.resize(1 << k, 0);
            // gf2-packed-backsub-time.7 design-checked -- a table's sum is built from the sum without its lowest row and that row, one row added a sum, not in the order of a Gray code, where each sum adds or takes away one row of the sum before it (log 0152-1, 0152-2)
            let mut built = 0;
            for g in 1..1usize << k {
                let t = g.trailing_zeros() as usize;
                let prev = g & (g - 1);
                let (done, next) = table.split_at_mut(g * width);
                let next = &mut next[..width];
                next.copy_from_slice(&done[prev * width..(prev + 1) * width]);
                let row = rows.words(b + t);
                for (a, &x) in next[width - row.len()..].iter_mut().zip(row) {
                    *a ^= x;
                }
                sums[g] = sums[prev] + rows.entries[b + t];
                built += rows.entries[b + t];
            }
            budget.spend_packed(built, (sums.len() - 1) * width)?;
            for r in e..n {
                let index = (0..k).fold(0, |index, t| {
                    index | usize::from(rows.bit(r, rows.js[b + t])) << t
                });
                if index != 0 {
                    rows.add_sum(r, from, &table[index * width..(index + 1) * width]);
                    budget.spend_packed(sums[index], width)?;
                }
            }
        }
        let mut tail = Vec::new();
        for i in (0..n).filter(|&i| rows.changed[i]) {
            budget.tick(Site::Row)?;
            tail.clear();
            for (w, &word) in rows.words(i).iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let j = (rows.first[i] + w) * WORD + bits.trailing_zeros() as usize;
                    if j != rows.js[i] {
                        tail.push(self.column[j]);
                    }
                    bits &= bits - 1;
                }
            }
            changed(i, &tail);
        }
        Ok(())
    }

    /// Stores the accumulator as the packed row of the later pivot at K
    /// column j (`store`), and sets its bit of `led`.
    fn store_later(&mut self, j: usize) {
        debug_assert!(j < self.later.len(), "K column {j} past the K columns");
        let row = self.store();
        self.later[j] = Some(row);
        self.led[j / WORD] |= 1 << (j % WORD);
    }

    /// The accumulator as a packed row, its words from the first nonzero
    /// one to the last appended to `words`, and the accumulator zero after.
    fn store(&mut self) -> Packed {
        let at = self.words.len();
        let mut row = Packed {
            at,
            word: 0,
            len: 0,
            entries: 0,
        };
        if self.lo < self.hi {
            let span = &mut self.acc[self.lo..self.hi];
            let first = span.iter().take_while(|&&x| x == 0).count();
            let end = span.len() - span.iter().rev().take_while(|&&x| x == 0).count();
            if first < end {
                let kept = &span[first..end];
                self.words.extend_from_slice(kept);
                // The K columns number below 2^32, as the matrix's.
                row = Packed {
                    at,
                    word: (self.lo + first).narrow(),
                    len: kept.len().narrow(),
                    entries: kept.iter().map(|x| x.count_ones()).sum(),
                };
            }
            span.fill(0);
        }
        self.lo = usize::MAX;
        self.hi = 0;
        row
    }

    /// Clears the accumulator and the walk's stack after a stop.
    fn clear(&mut self) {
        if self.lo < self.hi {
            self.acc[self.lo..self.hi].fill(0);
        }
        self.lo = usize::MAX;
        self.hi = 0;
        self.stack.clear();
    }
}

/// The made rows of a back-substitution (`Rows::back_substitute`), each
/// in full words from the word of its leading column to the K columns'
/// end.
struct Echelon {
    /// Per row, the K column it leads.
    js: Vec<usize>,
    /// Per row, the word of its leading column, where its words begin.
    first: Vec<usize>,
    /// Per row, where its words begin in `words`.
    at: Vec<usize>,
    /// Every row's words, by row.
    words: Vec<u64>,
    /// Per row, the entries of its packed row, and of its result once
    /// counted again in its block (`count`): the entries a sum of it
    /// spends.
    entries: Vec<usize>,
    /// Per row, whether a sum was added to it.
    changed: Vec<bool>,
}

// gf2-packed-backsub-time.6 design-checked -- each row is held in full words from the word of its leading column to the K columns' end, not at the K columns' whole width, which holds about twice the words, nor as its packed row from its first nonzero word to its last, which a sum widens (log 0135-1, 0152-1, 0152-2)
impl Echelon {
    /// The rows of `rows` that lead the matrix columns `leads`, each
    /// copied from its packed row.
    fn new(rows: &Rows, leads: &[MonoId]) -> Echelon {
        let words = rows.acc.len();
        let js: Vec<usize> = leads.iter().map(|c| rows.k(c.index())).collect();
        let first: Vec<usize> = js.iter().map(|&j| j / WORD).collect();
        let mut at = Vec::with_capacity(js.len());
        let mut len = 0;
        for &f in &first {
            at.push(len);
            len += words - f;
        }
        let mut held = vec![0; len];
        let mut entries = Vec::with_capacity(js.len());
        for (i, &j) in js.iter().enumerate() {
            #[allow(
                clippy::expect_used,
                reason = "`keep` stores the packed row of every made row, at the K column it leads"
            )]
            let p = rows.later[j].expect("a made row's packed row");
            debug_assert_eq!(
                p.word as usize, first[i],
                "a packed row begins at its leading column's word"
            );
            let row = &rows.words[p.at..p.at + p.len as usize];
            held[at[i]..at[i] + row.len()].copy_from_slice(row);
            entries.push(p.entries as usize);
        }
        Echelon {
            changed: vec![false; js.len()],
            js,
            first,
            at,
            words: held,
            entries,
        }
    }

    /// Row i's words.
    fn words(&self, i: usize) -> &[u64] {
        debug_assert!(
            i < self.at.len(),
            "row {i} past the {n} rows",
            n = self.at.len()
        );
        let end = self.at.get(i + 1).copied().unwrap_or(self.words.len());
        &self.words[self.at[i]..end]
    }

    /// Whether row i holds K column j, which lies in its words.
    fn bit(&self, i: usize, j: usize) -> bool {
        debug_assert!(
            j / WORD >= self.first[i],
            "K column {j} before row {i}'s words"
        );
        self.words[self.at[i] + j / WORD - self.first[i]] >> (j % WORD) & 1 == 1
    }

    /// Row t gains row u, an earlier row, whose leading column lies right
    /// of row t's. The words added.
    fn add_row(&mut self, t: usize, u: usize) -> usize {
        debug_assert!(u < t, "row {u} after row {t}");
        let (from, len) = (self.at[u], self.words(u).len());
        // Row u's words end where the next row's begin, at row t's or
        // before, and its first word is row t's or after.
        let to = self.at[t] + self.first[u] - self.first[t];
        let (held, rest) = self.words.split_at_mut(to);
        for (a, &x) in rest[..len].iter_mut().zip(&held[from..from + len]) {
            *a ^= x;
        }
        self.changed[t] = true;
        len
    }

    /// Counts row i's entries again where a sum was added to it.
    fn count(&mut self, i: usize) {
        debug_assert!(
            i < self.at.len(),
            "row {i} past the {n} rows",
            n = self.at.len()
        );
        if self.changed[i] {
            self.entries[i] = self.words(i).iter().map(|x| x.count_ones() as usize).sum();
        }
    }

    /// Row r gains `sum`, the words of a table's sum from word `from`, at
    /// or after the word of row r's leading column.
    fn add_sum(&mut self, r: usize, from: usize, sum: &[u64]) {
        debug_assert!(from >= self.first[r], "a sum before row {r}'s words");
        let to = self.at[r] + from - self.first[r];
        for (a, &x) in self.words[to..to + sum.len()].iter_mut().zip(sum) {
            *a ^= x;
        }
        self.changed[r] = true;
    }
}
