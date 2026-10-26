//! The module contract: rows are elements of a free module, and a row
//! carries its position. Rank 1, the ideal case, is its one implementation
//! here.

use std::fmt::Debug;

/// A free module of finite rank over the polynomial ring.
pub trait Module: Copy + Debug + Send + Sync + 'static {
    /// Where in the module a row stands: its basis vector.
    type Position: Copy + Eq + Ord + Default + Debug + Send + Sync + 'static;
    fn rank(&self) -> usize;
}

/// The module of rank 1, the polynomial ring itself. Its one position is
/// `()`, which takes no space.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rank1;

impl Module for Rank1 {
    type Position = ();

    fn rank(&self) -> usize {
        1
    }
}
