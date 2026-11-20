//! A matrix's column vector: its rows' columns, 4 bytes an entry, or 2
//! bytes an entry, two to a word, where the vector is narrow; and the view
//! of one row's columns at the vector's width.

use std::ops::Range;
use std::slice;

use crate::id::{Column, MonoId};

/// `body` with `cols` bound to the columns of `row`, a `Row`, as `&[u16]`
/// or as `&[MonoId]`: a `match` with `body` in both arms, so that a call
/// site names a row operation once and the compiler takes its copy for
/// each width (column-index-peak.11). `by_width!(mut row, |cols| body)`
/// binds the entries of a `RowMut`.
macro_rules! by_width {
    (mut $row:expr, |$cols:ident| $body:expr) => {
        match $row {
            $crate::columns::RowMut::Narrow($cols) => $body,
            $crate::columns::RowMut::Wide($cols) => $body,
        }
    };
    ($row:expr, |$cols:ident| $body:expr) => {
        match $row {
            $crate::columns::Row::Narrow($cols) => $body,
            $crate::columns::Row::Wide($cols) => $body,
        }
    };
}

pub(crate) use by_width;

/// The columns a narrow vector's entries number: 2^16, those a `u16`
/// holds.
pub(crate) const NARROW_COLUMNS: usize = 1 << u16::BITS;

// column-index-peak.1 cited-checked: faugere-1999 §3.4 -- when the matrices are big their storage is compressed; the coefficients are not duplicated, so only the positions of the nonzero elements are compressed, and no compression is inefficient both for CPU time and memory
// column-index-peak.2 design-checked -- a column takes a fixed width, 2 bytes where the round's monomials number at most 2^16 and 4 bytes past that, not the gaps between a row's columns in a byte each with an escape past 255 (faugere-1999 §3.4, pearce-2026 §2.1), nor 2 bytes always with the columns cut in blocks of 2^16 (not measured)
// column-index-peak.9 design-checked -- the column vector stays one vector of 4-byte words, a narrow vector's entries two to a word, not a vector of `u16` beside one of `MonoId` (not measured)
/// The columns of a matrix's rows, in one vector of words that a run's
/// buffers keep from matrix to matrix (`matrix::Buffers`). Wide, entry i is
/// word i, a `MonoId`. Narrow, entry i is the half-word i of the words read
/// as `u16`: the words hold ⌈len/2⌉, and where the entries are odd in
/// number the half-word past the last is 0, so that every word is
/// initialized.
pub(crate) struct Columns {
    words: Vec<MonoId>,
    narrow: bool,
    /// The entries.
    len: usize,
}

impl Columns {
    /// A wide vector of no entry and no capacity.
    pub(crate) fn new() -> Columns {
        Columns {
            words: Vec::new(),
            narrow: false,
            len: 0,
        }
    }

    /// The entries.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether an entry takes 2 bytes.
    pub(crate) fn is_narrow(&self) -> bool {
        self.narrow
    }

    /// The bytes of an entry: 2 where the vector is narrow, 4 where it is
    /// wide.
    pub(crate) fn entry_bytes(&self) -> usize {
        if self.narrow {
            size_of::<u16>()
        } else {
            size_of::<MonoId>()
        }
    }

    /// Whether an entry of the vector's width holds column `column`.
    pub(crate) fn holds(&self, column: usize) -> bool {
        if self.narrow {
            u16::try_from(column).is_ok()
        } else {
            u32::try_from(column).is_ok()
        }
    }

    /// The words the entries fill.
    pub(crate) fn words(&self) -> &[MonoId] {
        &self.words
    }

    /// The words `entries` entries fill at the vector's width.
    pub(crate) fn words_for(&self, entries: usize) -> usize {
        if self.narrow {
            entries.div_ceil(2)
        } else {
            entries
        }
    }

    /// The capacity, in words.
    pub(crate) fn capacity(&self) -> usize {
        self.words.capacity()
    }

    /// Drops every entry. The capacity and the width stay.
    pub(crate) fn clear(&mut self) {
        self.words.clear();
        self.len = 0;
    }

    /// Sets the width of an empty vector.
    pub(crate) fn set_narrow(&mut self, narrow: bool) {
        debug_assert!(self.is_empty(), "the width of a vector of entries");
        self.narrow = narrow;
    }

    /// Room for `words` more words, exactly.
    pub(crate) fn reserve_exact(&mut self, words: usize) {
        self.words.reserve_exact(words);
    }

    /// Room for `entries` more entries, by the words' growth.
    pub(crate) fn reserve(&mut self, entries: usize) {
        let words = self.words_for(self.len + entries) - self.words.len();
        self.words.reserve(words);
    }

    /// Gives back the slack of the words.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.words.shrink_to_fit();
    }

    /// Where entry 0 lies, read at width `I`, `u16` where the vector is
    /// narrow and `MonoId` where it is wide: a writer of entries past the
    /// length up to the capacity, which `set_len` then counts.
    pub(crate) fn as_mut_ptr<I: Column>(&mut self) -> *mut I {
        debug_assert_eq!(
            size_of::<I>(),
            self.entry_bytes(),
            "a writer at the vector's width"
        );
        self.words.as_mut_ptr().cast::<I>()
    }

    /// Sets the entries to `len`, and where the vector is narrow and `len`
    /// is odd, writes the half-word past the last 0.
    ///
    /// # Safety
    ///
    /// The entries below `len` are written, and `len` entries fit the
    /// capacity: `words_for(len)` words.
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        let words = self.words_for(len);
        debug_assert!(
            words <= self.words.capacity(),
            "{len} entries past the capacity of {capacity} words",
            capacity = self.words.capacity()
        );
        if self.narrow && len % 2 == 1 {
            // SAFETY: half-word `len` lies in word ⌈len/2⌉ − 1, within the
            // capacity by the caller's guarantee.
            unsafe { self.words.as_mut_ptr().cast::<u16>().add(len).write(0) };
        }
        // SAFETY: every word below `words` holds written half-words or
        // entries, by the caller's guarantee and the write above.
        unsafe { self.words.set_len(words) };
        self.len = len;
    }

    /// The entries a to b, at the vector's width.
    pub(crate) fn row(&self, a: usize, b: usize) -> Row<'_> {
        match self.entries() {
            Row::Narrow(e) => Row::Narrow(&e[a..b]),
            Row::Wide(e) => Row::Wide(&e[a..b]),
        }
    }

    /// As `row`, with no bounds check outside the test profile, whose
    /// debug assertions check them.
    ///
    /// # Safety
    ///
    /// a ≤ b ≤ `len()`.
    pub(crate) unsafe fn row_unchecked(&self, a: usize, b: usize) -> Row<'_> {
        debug_assert!(
            a <= b && b <= self.len,
            "entries {a} to {b} of {len}",
            len = self.len
        );
        let words = self.words.as_ptr();
        // SAFETY: the entries a to b lie below `len`, by the caller's
        // guarantee, and so within the words: a narrow entry i is half-word
        // i of the words, of a `MonoId` that is a u32, two u16 at its
        // alignment.
        unsafe {
            if self.narrow {
                Row::Narrow(slice::from_raw_parts(words.cast::<u16>().add(a), b - a))
            } else {
                Row::Wide(slice::from_raw_parts(words.add(a), b - a))
            }
        }
    }

    /// Every entry, at the vector's width.
    pub(crate) fn entries(&self) -> Row<'_> {
        // SAFETY: 0 ≤ len.
        unsafe { self.row_unchecked(0, self.len) }
    }

    /// Every entry, at the vector's width, to write.
    pub(crate) fn entries_mut(&mut self) -> RowMut<'_> {
        let (len, words) = (self.len, self.words.as_mut_ptr());
        // SAFETY: as in `row_unchecked`, and the borrow of the vector is
        // mutable.
        unsafe {
            if self.narrow {
                RowMut::Narrow(slice::from_raw_parts_mut(words.cast::<u16>(), len))
            } else {
                RowMut::Wide(slice::from_raw_parts_mut(words, len))
            }
        }
    }

    /// Entry i as an index.
    pub(crate) fn entry(&self, i: usize) -> usize {
        self.entries().column(i)
    }

    /// Copies the entries `src` to the entries from `dst` on, as
    /// `slice::copy_within` does.
    pub(crate) fn copy_within(&mut self, src: Range<usize>, dst: usize) {
        by_width!(mut self.entries_mut(), |e| e.copy_within(src, dst));
    }

    /// Keeps the first `len` entries, where the vector holds more.
    pub(crate) fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        self.words.truncate(self.words_for(len));
        // SAFETY: the entries below `len` are written, and their words
        // are held.
        unsafe { self.set_len(len) };
    }

    // column-index-peak.6 design-checked -- a narrow vector widens in place, from the last entry to the first, not a build started again wide, nor a wide copy beside the narrow vector (not measured)
    /// Makes the vector wide in place: the words grow to the entries, and
    /// from the last entry down to the first each is read as a half-word
    /// and written as a word. Word i covers half-words 2i and 2i + 1, at
    /// or past i, which are read before it is written. `spend` hears the
    /// entries the growth wrote and those widened, `chunk` or fewer at a
    /// time, after each chunk: an Err it returns ends the widening there,
    /// the vector then narrow, its words grown and the entries from the
    /// stop on garbled, which no caller reads after a stop.
    pub(crate) fn widen<E>(
        &mut self,
        chunk: usize,
        mut spend: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        debug_assert!(chunk > 0, "a chunk holds an entry");
        if !self.narrow {
            return Ok(());
        }
        let len = self.len;
        while self.words.len() < len {
            let grown = chunk.min(len - self.words.len());
            self.words
                .resize(self.words.len() + grown, MonoId::from_raw(0));
            spend(grown)?;
        }
        let words = self.words.as_mut_ptr();
        let mut end = len;
        while end > 0 {
            let start = end.saturating_sub(chunk);
            for i in (start..end).rev() {
                // SAFETY: i < len, the words' length, and half-word i lies
                // in word i/2; the entries above i are read already.
                unsafe {
                    let entry = words.cast::<u16>().add(i).read();
                    words.add(i).write(MonoId::from_raw(u32::from(entry)));
                }
            }
            spend(end - start)?;
            end = start;
        }
        self.narrow = false;
        Ok(())
    }

    /// A vector of the width `narrow` names holding `entries`, each of which
    /// that width holds.
    #[cfg(test)]
    pub(crate) fn of(narrow: bool, entries: &[usize]) -> Columns {
        let mut c = Columns::new();
        c.set_narrow(narrow);
        c.extend(entries);
        c
    }

    /// Appends `entries`, each of which the vector's width holds, as
    /// `Table::row_products` appends a row: reserved, then written.
    #[cfg(test)]
    pub(crate) fn extend(&mut self, entries: &[usize]) {
        /// Writes `entries` at width `I`, past the entries of `c`.
        fn write<I: Column>(c: &mut Columns, entries: &[usize]) {
            c.reserve(entries.len());
            let (out, len) = (c.as_mut_ptr::<I>(), c.len());
            for (k, &e) in entries.iter().enumerate() {
                // SAFETY: len + k lies below the entries reserved.
                unsafe { out.add(len + k).write(I::of(e)) };
            }
            // SAFETY: every entry is written, within the reservation.
            unsafe { c.set_len(len + entries.len()) };
        }
        if self.narrow {
            write::<u16>(self, entries);
        } else {
            write::<MonoId>(self, entries);
        }
    }

    /// The vector narrow with the same entries; None where an entry passes
    /// `u16`.
    #[cfg(test)]
    pub(crate) fn narrowed(&self) -> Option<Columns> {
        let entries: Vec<usize> = (0..self.len).map(|i| self.entry(i)).collect();
        entries
            .iter()
            .all(|&e| u16::try_from(e).is_ok())
            .then(|| Columns::of(true, &entries))
    }
}

// column-index-peak.12 design-checked -- a matrix row is read through a view of two kinds, on which each row operation dispatches once, not the reducer compiled per column width, nor the row's columns widened into a buffer before each row operation (not measured)
/// One row's columns at the width of its vector (`Columns`).
#[derive(Clone, Copy)]
pub(crate) enum Row<'a> {
    Narrow(&'a [u16]),
    Wide(&'a [MonoId]),
}

impl<'a> Row<'a> {
    pub(crate) fn len(self) -> usize {
        by_width!(self, |c| c.len())
    }

    /// Column i as an index.
    pub(crate) fn column(self, i: usize) -> usize {
        by_width!(self, |c| c[i].column())
    }

    /// The leading column, the first.
    pub(crate) fn lead(self) -> MonoId {
        MonoId::at(self.column(0))
    }

    /// The columns after the lead.
    pub(crate) fn tail(self) -> Row<'a> {
        match self {
            Row::Narrow(c) => Row::Narrow(&c[1..]),
            Row::Wide(c) => Row::Wide(&c[1..]),
        }
    }

    /// The columns as `MonoId`s, in a vector of their own.
    pub(crate) fn to_wide(self) -> Vec<MonoId> {
        match self {
            Row::Narrow(c) => c.iter().map(|&e| MonoId::at(e.column())).collect(),
            Row::Wide(c) => c.to_vec(),
        }
    }
}

/// Entries of a `Columns` at its width, to write.
pub(crate) enum RowMut<'a> {
    Narrow(&'a mut [u16]),
    Wide(&'a mut [MonoId]),
}
