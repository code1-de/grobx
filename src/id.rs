//! The ids a run numbers its items by, each a type of its own over u32, so
//! that an id of one kind cannot stand where another kind belongs.
//! `u32::MAX` marks no item. The crate numbers its items; a trace reads
//! their ids, and a caller of the field contract numbers its dense rows'
//! columns by `MonoId::from_raw`.

use std::fmt;

use crate::narrow::{self, Narrow, Size};

macro_rules! ids {
    ($($(#[$attr:meta])* $vis:vis $name:ident;)*) => {$(
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        $vis struct $name(u32);

        #[allow(dead_code, reason = "every id has the same methods, and not every id calls each")]
        impl $name {
            /// The mark of no item.
            pub(crate) const NONE: $name = $name(u32::MAX);

            /// The id of the item a collection gains when it holds `n`
            /// (`narrow::id`).
            pub(crate) fn new(n: usize) -> Result<$name, Size> {
                narrow::id(n).map($name)
            }

            /// The id of index `i`, below a count that `new` numbered.
            #[inline]
            pub(crate) fn at(i: usize) -> $name {
                $name(i.narrow())
            }

            /// The id as an index.
            #[inline]
            #[must_use]
            $vis fn index(self) -> usize {
                self.0 as usize
            }

            /// The u32 that holds the id, where it is packed with other bits.
            #[inline]
            #[must_use]
            $vis fn raw(self) -> u32 {
                self.0
            }

            /// The id the u32 `raw` holds.
            #[inline]
            #[must_use]
            $vis fn from_raw(raw: u32) -> $name {
                $name(raw)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    )*};
}

ids! {
    /// A monomial: its id in a monomial table, its column in a matrix, or
    /// its coordinate over a staircase.
    pub MonoId;
    /// A basis element, numbered in the order the basis made it.
    pub ElementId;
    /// A row of a matrix, numbered in the order of its rows.
    pub RowId;
    /// A row a reduction made, numbered in the order made.
    pub(crate) MadeId;
    /// A row the cofactor tracking kept, numbered in the order kept.
    pub(crate) KeepId;
    /// A monomial of a staircase's border, numbered by increasing source
    /// order.
    pub(crate) BorderId;
}
