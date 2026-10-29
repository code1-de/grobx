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
use std::cell::{Cell, RefCell};
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::ffi::{CStr, c_void};
use std::fmt::{self, Write as _};
use std::ptr;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::sync::Once;
use std::sync::OnceLock;
use std::time::Instant;

use libc::c_int;
#[cfg(target_os = "linux")]
use mimalloc::MiMalloc;

#[cfg(not(target_os = "linux"))]
use crate::os::{Resident, peak};
use crate::report::{Opt, Secs};
use crate::{Error, ReportSink};

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
        let p = unsafe {
            if layout.size() <= SMALL {
                SMALL_SIDE.alloc(layout)
            } else {
                System.alloc(layout)
            }
        };
        hear("alloc", layout.size(), None, p)
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, handed on.
        let p = unsafe {
            if layout.size() <= SMALL {
                SMALL_SIDE.alloc_zeroed(layout)
            } else {
                System.alloc_zeroed(layout)
            }
        };
        hear("zeroed", layout.size(), None, p)
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
            let p = unsafe { SMALL_SIDE.realloc(ptr, layout, new_size) };
            return hear("realloc", new_size, Some(layout.size()), p);
        }
        if !was && !will {
            // SAFETY: the system's allocator served the block at this
            // layout.
            let p = unsafe { System.realloc(ptr, layout, new_size) };
            return hear("realloc", new_size, Some(layout.size()), p);
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

// error-report.4 design -- a request of LARGE bytes or more, and every request the allocator refuses, is a line of the call report on its sink, written by the allocator, not try_reserve at each site that may make one, nor a line on standard error, which an aborted trial's harness does not keep (not measured)
// A refused request ends in handle_alloc_error, which aborts; the line
// before it is the only trace of its size and its site.
/// The bytes from which a served request is a line of the call report:
/// 1 GiB.
pub const LARGE: usize = 1 << 30;

/// The bytes of an `alloc` line: its fields and `FRAMES` frames of at most
/// `NAME` bytes and an offset each fit.
const LINE: usize = 4096;

/// The return addresses an `alloc` line gives.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
const FRAMES: usize = 32;

/// The bytes of an object's file name a frame gives at most.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
const NAME: usize = 64;

thread_local! {
    /// The call report's sink of the call this thread runs, and the call's
    /// start (`watch`).
    static WATCHED: RefCell<Option<(Instant, ReportSink)>> = const { RefCell::new(None) };
    /// Whether this thread writes a request's line, so that a request the
    /// writing makes is not heard in turn.
    static HEARING: Cell<bool> = const { Cell::new(false) };
}

/// While it lives, the requests `hear` reports on this thread are lines of
/// the call report on the sink `watch` named; dropped, the watch before it
/// stands again.
pub(crate) struct Watch(Option<(Instant, ReportSink)>);

/// Watches this thread's requests for the call that started at `start`,
/// whose report goes to `sink`.
pub(crate) fn watch(start: Instant, sink: ReportSink) -> Watch {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    warm_backtrace();
    Watch(WATCHED.with(|w| w.replace(Some((start, sink)))))
}

impl Drop for Watch {
    fn drop(&mut self) {
        let before = self.0.take();
        WATCHED.with(|w| w.replace(before));
    }
}

/// `p`, the block served for a request of `bytes`, null where the request
/// was refused; `from` is the bytes of the block a resize moves. A request
/// of `LARGE` bytes or more, or a refused one, is heard (`heard`).
#[inline]
fn hear(kind: &str, bytes: usize, from: Option<usize>, p: *mut u8) -> *mut u8 {
    if bytes >= LARGE || p.is_null() {
        heard(kind, bytes, from, !p.is_null());
    }
    p
}

/// Writes the request's `alloc` line where this thread's call watches
/// (`watch`), and does nothing else. A request while the thread's locals
/// are torn down comes after its last call, so no report hears it. A
/// writer that is busy or fails takes no line here: the report's next line
/// meets the failure and ends the writing (`Params::report_sink`).
#[cold]
#[inline(never)]
fn heard(kind: &str, bytes: usize, from: Option<usize>, served: bool) {
    let Ok(again) = HEARING.try_with(|h| h.replace(true)) else {
        return;
    };
    if again {
        return;
    }
    let Ok(()) = WATCHED.try_with(|w| {
        if let Ok(w) = w.try_borrow()
            && let Some((start, sink)) = &*w
        {
            let mut line = Line::new();
            if write_heard(&mut line, *start, kind, bytes, from, served).is_ok() {
                match sink.try_put(line.as_str()) {
                    None | Some(Ok(()) | Err(_)) => {}
                }
            } else {
                debug_assert!(false, "an alloc line past {LINE} bytes");
            }
        }
    }) else {
        return;
    };
    HEARING.with(|h| h.set(false));
}

/// The `alloc` line of a request.
fn write_heard(
    out: &mut Line,
    start: Instant,
    kind: &str,
    bytes: usize,
    from: Option<usize>,
    served: bool,
) -> fmt::Result {
    write!(
        out,
        "{time} alloc kind={kind} bytes={bytes}{from} result={result}",
        time = Secs(start.elapsed()),
        from = Opt("from", from),
        result = if served { "served" } else { "refused" }
    )?;
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    write_stack(out)?;
    out.write_char('\n')
}

/// A line written on the stack, since the allocator that writes it may not
/// ask itself for memory.
struct Line {
    bytes: [u8; LINE],
    len: usize,
}

impl Line {
    fn new() -> Line {
        Line {
            bytes: [0; LINE],
            len: 0,
        }
    }

    #[allow(
        clippy::expect_used,
        reason = "write_str appends whole strs alone, so the bytes are UTF-8"
    )]
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("a Line holds whole strs alone")
    }
}

impl fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        let to = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        to.copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

// error-report.6 design -- an alloc line gives its request's return addresses as an object's file name and the offset into it, which addr2line maps to a function, not the symbols backtrace_symbols names, which it allocates for, nor no addresses (not measured)
/// ` stack=` and the return addresses of the request, each its object's
/// file name, at most `NAME` bytes of it, and `+0x` its offset into the
/// object, or the address alone where no object holds it.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn write_stack(out: &mut Line) -> fmt::Result {
    let mut frames = [ptr::null_mut::<c_void>(); FRAMES];
    let size = c_int::try_from(FRAMES).map_err(|_| fmt::Error)?;
    // SAFETY: backtrace writes at most `size` addresses into `frames`.
    let n = unsafe { libc::backtrace(frames.as_mut_ptr(), size) };
    out.write_str(" stack=")?;
    for (i, &frame) in frames
        .iter()
        .take(usize::try_from(n).unwrap_or(0))
        .enumerate()
    {
        if i > 0 {
            out.write_char(',')?;
        }
        // SAFETY: Dl_info is plain data, for which zeroes are valid.
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        // SAFETY: dladdr writes `info`, whose name points into the loader's
        // tables, which stay while the object is loaded.
        let found = unsafe { libc::dladdr(frame, &raw mut info) } != 0;
        if found && !info.dli_fname.is_null() {
            // SAFETY: dli_fname is a NUL-terminated string of the loader's.
            let path = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
            let name = path.rsplit(|&b| b == b'/').next().unwrap_or(path);
            for &b in name.iter().take(NAME) {
                let c = if b.is_ascii_graphic() && b != b'=' && b != b',' {
                    char::from(b)
                } else {
                    '_'
                };
                out.write_char(c)?;
            }
            let offset = (frame as usize).wrapping_sub(info.dli_fbase as usize);
            write!(out, "+0x{offset:x}")?;
        } else {
            write!(out, "0x{address:x}", address = frame as usize)?;
        }
    }
    Ok(())
}

// error-report.7 design -- libgcc is loaded by a first backtrace as a call starts to watch, not left to the first request heard (not measured)
// backtrace(3), NOTES: backtrace does not call malloc itself, but libgcc,
// which it loads at its first use, does as it loads.
/// Calls `backtrace` once a process, so that libgcc is loaded before a
/// request that the system refuses has to write its stack.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn warm_backtrace() {
    static WARM: Once = Once::new();
    WARM.call_once(|| {
        let mut frame = [ptr::null_mut::<c_void>(); 1];
        // SAFETY: backtrace writes at most one address into `frame`.
        unsafe { libc::backtrace(frame.as_mut_ptr(), 1) };
    });
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
    (
        bytes(Resident::open().and_then(|r| r.bytes())),
        bytes(peak()),
    )
}

/// Whether mimalloc's heap region holds `p`.
#[cfg(all(test, target_os = "linux"))]
pub(crate) fn mimalloc_holds<T>(p: *const T) -> bool {
    // SAFETY: mi_is_in_heap_region reads mimalloc's own bookkeeping alone.
    unsafe { mi_is_in_heap_region(p.cast()) }
}
