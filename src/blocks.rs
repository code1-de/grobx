//! Where the blocks of memory come from: `Allocator`, the global allocator
//! the Python module, the C library, the CLI, and the crate's tests set,
//! serves a request of at most `SMALL`
//! bytes from mimalloc and a larger one from the system's malloc, on Linux;
//! elsewhere the system's allocator serves both. A program
//! that sets no global allocator runs the crate under its own, and
//! `collect` then finds nothing of mimalloc's to give back. `pad_heap`
//! sets, at a call's entry, how glibc's malloc grows its heap and which
//! blocks take a mapping of their own.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;
use std::sync::OnceLock;

use libc::c_int;
#[cfg(target_os = "linux")]
use mimalloc::MiMalloc;

use crate::Error;
#[cfg(not(target_os = "linux"))]
use crate::os::{Resident, peak};

// small-block-time.2 tuned -- the largest request mimalloc serves is 1024 bytes, not 256, 2048, 4096, 16384, or 65536 (log 0053-1)
/// The largest request, in bytes, that mimalloc serves.
pub const SMALL: usize = 1024;

// port.26 design -- mimalloc serves the small blocks on Linux alone, where small-block-time was measured; elsewhere every request is the system's, not mimalloc with its arena reservation turned off, nor a hook before main that sets it (not measured)
// mimalloc 3.3.2 failed with an access violation as it reserved its first
// arena, before main, in a Windows 11 guest under KVM; on Windows 10 on the
// machine itself it ran.
/// The allocator of the requests of at most `SMALL` bytes: mimalloc on
/// Linux, the system's allocator elsewhere.
#[cfg(target_os = "linux")]
const SMALL_SIDE: MiMalloc = MiMalloc;
#[cfg(not(target_os = "linux"))]
const SMALL_SIDE: System = System;

// page-fault-time.3 tuned -- the heap's pad is 16 MiB, not 4, 32, or 64 MiB (log 0090-1)
// A larger pad took no more time off and put more peaks past their
// allowance, a smaller one leaves fewer extents a huge page backs (log
// 0090-1, _Step 1_).
/// The bytes glibc's malloc adds to each growth of its main heap, and
/// keeps free at the heap's top when it trims it (`M_TOP_PAD`), from the
/// first call of a process on.
pub const TOP_PAD: c_int = 16 << 20;

// page-fault-time.4 design -- the crate sets M_MMAP_THRESHOLD beside the pad, not the threshold left as setting the pad leaves it (log 0090-1)
// page-fault-time.5 tuned -- the threshold is 2 MiB, a huge page, not 8 or 32 MiB (log 0090-1)
// Setting M_TOP_PAD disables glibc's dynamic threshold, which stays at
// 128 KiB (mallopt-3 §M_MMAP_THRESHOLD): every block of 128 KiB or more
// that the heap cannot hold would take a mapping, whose pages the kernel
// zeroes and faults afresh. The larger thresholds held more of the heap
// past their blocks' frees (log 0090-1, _Step 1_).
/// The size in bytes from which a block that the heap's free space cannot
/// hold takes a mapping of its own (`M_MMAP_THRESHOLD`), from the first
/// call of a process on: a huge page's.
pub const MMAP_THRESHOLD: c_int = 2 << 20;

// small-block-time.1 design -- a request of at most SMALL bytes is served by mimalloc and a larger one by the system's malloc, not every request by one of the two, nor the small ones by free lists of the crate's own (log 0053-1)
// small-block-time.4 design -- the crate exports the allocator and the Python module, the C library, the CLI, and the crate's tests set it as their global allocator, not the crate setting it for every program that links it (not measured)
/// A global allocator: a request of at most `SMALL` bytes is served by
/// mimalloc, and a larger one by the system's allocator, `System`; off
/// Linux `System` serves both (`SMALL_SIDE`). A program sets it by
///
/// ```
/// #[global_allocator]
/// static GLOBAL: grobx::Allocator = grobx::Allocator;
/// # fn main() {}
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Allocator;

// small-block-time.3 design -- a block is freed and resized by the allocator its layout's size names, and a resize across SMALL moves it to the other, not a lookup of the pointer in mimalloc's heap region (not measured)
// SAFETY: every block is served, freed, and resized by the allocator its
// layout's size names, `SMALL_SIDE` or `System`, each of which keeps
// `GlobalAlloc`'s contract, and a resize across `SMALL` moves the block.
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, handed on.
        unsafe {
            if layout.size() <= SMALL {
                SMALL_SIDE.alloc(layout)
            } else {
                System.alloc(layout)
            }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, handed on.
        unsafe {
            if layout.size() <= SMALL {
                SMALL_SIDE.alloc_zeroed(layout)
            } else {
                System.alloc_zeroed(layout)
            }
        }
    }

    /// The block goes back to the allocator its size names: the layout
    /// `GlobalAlloc` hands over is the one the block was served or last
    /// resized at.
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the allocator the size names served the block at this
        // layout.
        unsafe {
            if layout.size() <= SMALL {
                SMALL_SIDE.dealloc(ptr, layout);
            } else {
                System.dealloc(ptr, layout);
            }
        }
    }

    /// A resize within one side of `SMALL` is that side's. A resize across
    /// it moves the block: a new block of `new_size` at the old alignment,
    /// a copy of the shorter of the two lengths, and the old block freed.
    /// Where the new block cannot be had, the result is null and the old
    /// block stays as it was.
    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let was = layout.size() <= SMALL;
        let will = new_size <= SMALL;
        if was && will {
            // SAFETY: SMALL_SIDE served the block at this layout.
            return unsafe { SMALL_SIDE.realloc(ptr, layout, new_size) };
        }
        if !was && !will {
            // SAFETY: the system's allocator served the block at this
            // layout.
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        // SAFETY: the caller guarantees that new_size is above zero and,
        // rounded up to the alignment, does not overflow isize.
        let new = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        // SAFETY: the new layout's size is above zero.
        let moved = unsafe { self.alloc(new) };
        if !moved.is_null() {
            // SAFETY: both blocks hold the shorter length, they are apart,
            // and the old block was served at its layout.
            unsafe {
                ptr::copy_nonoverlapping(ptr, moved, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        moved
    }
}

// small-block-time.7 design -- mi_collect, mi_process_info, and mi_is_in_heap_region are declared here, not through libmimalloc-sys's feature extended, which depends on the crate cty that the cargo cache lacks (not measured)
#[cfg(target_os = "linux")]
unsafe extern "C" {
    /// mimalloc's `mi_collect`: gives free memory back to the system,
    /// all of it where `force` is set.
    fn mi_collect(force: bool);
    /// mimalloc's `mi_process_info`: the process's times, resident set,
    /// commit, and page faults, each written where its pointer is not null.
    #[allow(clippy::too_many_arguments, reason = "mimalloc's signature")]
    fn mi_process_info(
        elapsed_msecs: *mut usize,
        user_msecs: *mut usize,
        system_msecs: *mut usize,
        current_rss: *mut usize,
        peak_rss: *mut usize,
        current_commit: *mut usize,
        peak_commit: *mut usize,
        page_faults: *mut usize,
    );
    /// mimalloc's `mi_is_in_heap_region`: whether `p` lies in mimalloc's
    /// heap region.
    #[cfg(test)]
    fn mi_is_in_heap_region(p: *const std::ffi::c_void) -> bool;
}

// small-block-time.6 design -- before a memory stop mimalloc gives its free memory back by mi_collect(true), not left to its own purge beside glibc's malloc_trim(0) (log 0053-1)
/// Gives mimalloc's free memory back to the system.
#[cfg(target_os = "linux")]
pub(crate) fn collect() {
    // SAFETY: mi_collect frees no live block and moves none.
    unsafe { mi_collect(true) }
}

// page-fault-time.1 design -- the heap's growth is padded by glibc's M_TOP_PAD, which the kernel backs with huge pages, not the blocks of 4 MiB or more in 2 MiB aligned mappings of the crate's own with MADV_HUGEPAGE (log 0090-1)
// page-fault-time.2 cited: mallopt-3 §Environment variables -- the environment's variables take effect only where they are defined before the first allocation, and mallopt's settings take precedence over them
// page-fault-time.6 design -- the parameters are set once a process, at the first call's entry, kept by a OnceLock, not at every call nor at the load of the Python module, the C library, or the CLI (not measured)
// page-fault-time.7 design -- a mallopt that returns other than 1 is Error::Malloc, which that call and every later call return, not a refusal ignored nor a panic (not measured)
// page-fault-time.8 design -- no parameter, keyword, switch, or field turns the setting off (not measured)
// The setting took the time down past the margin, with the system's share
// of the CPU time and the user time both lower, and the peaks within the
// mild form (log 0090-2, _Step 7_).
/// Sets glibc's `M_TOP_PAD` to `TOP_PAD` and `M_MMAP_THRESHOLD` to
/// `MMAP_THRESHOLD` for the process, at its first call, and returns
/// `Error::Malloc` naming the parameter `mallopt` refused, at that call and
/// every later one. A caller's own setting after the first call stays.
pub(crate) fn pad_heap() -> Result<(), Error> {
    static REFUSED: OnceLock<Option<&'static str>> = OnceLock::new();
    match *REFUSED.get_or_init(set_parameters) {
        None => Ok(()),
        Some(param) => Err(Error::Malloc { param }),
    }
}

// page-fault-time.9 design -- the parameters are set where the C library is glibc, as budget::trim calls malloc_trim, and elsewhere the call sets none, not a crate that builds on glibc alone (not measured)
/// `mallopt` on `M_TOP_PAD`, then on `M_MMAP_THRESHOLD`: the name of the
/// first it refused, None where it took both.
fn set_parameters() -> Option<&'static str> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    for (param, name, value) in [
        (libc::M_TOP_PAD, "M_TOP_PAD", TOP_PAD),
        (libc::M_MMAP_THRESHOLD, "M_MMAP_THRESHOLD", MMAP_THRESHOLD),
    ] {
        // SAFETY: mallopt reads its two integers and changes glibc's
        // parameters under glibc's lock; it touches no memory of ours.
        // It returns 1 on success and 0 on error (mallopt-3, RETURN VALUE).
        if unsafe { libc::mallopt(param, value) } != 1 {
            return Some(name);
        }
    }
    None
}

/// The process's resident set and its peak in bytes, as mimalloc reads
/// them from the system.
#[cfg(target_os = "linux")]
pub(crate) fn rss() -> (usize, usize) {
    let (mut current, mut peak) = (0, 0);
    let null = std::ptr::null_mut();
    // SAFETY: mi_process_info writes through the two valid pointers and
    // skips the null ones.
    unsafe {
        mi_process_info(
            null,
            null,
            null,
            &raw mut current,
            &raw mut peak,
            null,
            null,
            null,
        );
    }
    (current, peak)
}

/// The process's resident set and its peak in bytes, as the system reads
/// them (`os`): each 0 where it cannot be read, as `mi_process_info`
/// leaves it on Linux.
#[cfg(not(target_os = "linux"))]
pub(crate) fn rss() -> (usize, usize) {
    let bytes = |b: Option<u64>| b.and_then(|b| usize::try_from(b).ok()).unwrap_or(0);
    (bytes(Resident::open().and_then(|r| r.bytes())), bytes(peak()))
}

/// Whether mimalloc's heap region holds `p`.
#[cfg(all(test, target_os = "linux"))]
pub(crate) fn mimalloc_holds<T>(p: *const T) -> bool {
    // SAFETY: mi_is_in_heap_region reads mimalloc's own bookkeeping alone.
    unsafe { mi_is_in_heap_region(p.cast()) }
}
