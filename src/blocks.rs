//! Where the blocks of memory come from: `Allocator`, the global allocator
//! the Python module, the C library, the CLI, and the crate's tests set,
//! serves every request from the system's allocator and makes the large
//! and the refused ones lines of the call report. A program that sets no
//! global allocator runs the crate under its own, and its call reports
//! then have no `alloc` line.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::ffi::{CStr, c_void};
use std::fmt::{self, Write as _};
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::ptr;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::sync::Once;
use std::time::Instant;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
use libc::c_int;

use crate::ReportSink;
use crate::report::{Opt, Secs};

// small-block-time.4 design-checked -- the crate exports the allocator and the Python module, the C library, the CLI, and the crate's tests set it as their global allocator, not the crate setting it for every program that links it (not measured)
/// A global allocator: the system's allocator, `System`, serves every
/// request, and a request of `REPORTED` bytes or more, or a refused one, is
/// a line of the call report (`hear`). A program sets it by
///
/// ```
/// #[global_allocator]
/// static GLOBAL: grobx::Allocator = grobx::Allocator;
/// # fn main() {}
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Allocator;

// SAFETY: `System` serves, frees, and resizes every block, and it keeps
// `GlobalAlloc`'s contract.
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, handed on.
        let p = unsafe { System.alloc(layout) };
        hear("alloc", layout.size(), None, p)
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout, handed on.
        let p = unsafe { System.alloc_zeroed(layout) };
        hear("zeroed", layout.size(), None, p)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `System` served the block at this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `System` served the block at this layout; the caller's
        // new size, handed on.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        hear("realloc", new_size, Some(layout.size()), p)
    }
}

// error-report.4 design-checked -- a request of REPORTED bytes or more, and every request the allocator refuses, is a line of the call report on its sink, written by the allocator, not try_reserve at each site that may make one, nor a line on standard error, which an aborted trial's harness does not keep (not measured)
// A refused request ends in handle_alloc_error, which aborts; the line
// before it is the only trace of its size and its site.
/// The bytes from which a served request is a line of the call report:
/// 1 GiB.
pub const REPORTED: usize = 1 << 30;

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
/// of `REPORTED` bytes or more, or a refused one, is heard (`heard`).
#[inline]
fn hear(kind: &str, bytes: usize, from: Option<usize>, p: *mut u8) -> *mut u8 {
    if bytes >= REPORTED || p.is_null() {
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

// error-report.6 design-checked -- an alloc line gives its request's return addresses as an object's file name and the offset into it, which addr2line maps to a function, not the symbols backtrace_symbols names, which it allocates for, nor no addresses (not measured)
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

// error-report.7 practice-checked: libgcc preloaded by a first backtrace(), as backtrace(3) advises -- libgcc is loaded by a first backtrace as a call starts to watch, not left to the first request heard (not measured)
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
