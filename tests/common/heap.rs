//! page-fault-time's test code reads where glibc's malloc took a block
//! from the bytes `malloc_usable_size` gives past the request
//! (`iterations/0090-page-fault-time/plan.md`, _Test code_). The crate's
//! suite and the C interface's take these bounds.

/// A request's bytes past it in a block from the heap, at most: a chunk's
/// header and its rounding to 16 bytes.
pub const HEAP_EXCESS: u64 = 64;

/// A request's bytes past it in a mapping of its own, at least, for a
/// request that is a multiple of a page: the page its header takes, less
/// 32 bytes, which the header's 16 lie within.
pub const MAPPED_EXCESS: u64 = 4096 - 32;
