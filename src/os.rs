//! What the engine asks of the operating system: its page size, the
//! process's resident set and its peak, whole pages of memory given back,
//! and whether it has a clock. Linux, macOS, and Windows each answer by
//! calls of their own. wasm32 in a browser answers none: it reads no page
//! size, no resident set, and no peak, gives no pages back, and has no
//! clock, since std's `Instant::now` panics there.

#[cfg(target_os = "linux")]
use std::fs::{self, File};
use std::io;
#[cfg(target_os = "macos")]
use std::mem;
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;

#[cfg(windows)]
use windows_sys::Win32::System::Memory::DiscardVirtualMemory;
#[cfg(windows)]
use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::GetCurrentProcess;

// port.2 design-checked -- a system other than Linux, macOS, Windows, and wasm32 in a browser fails the build, not a build that finds the page size or the resident set missing when a call reads it (not measured)
#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    windows,
    all(target_family = "wasm", target_os = "unknown")
)))]
compile_error!("grobx builds on Linux, macOS, Windows, and wasm32-unknown-unknown");

/// Whether `Instant::now` reads a clock, rather than panicking.
pub(crate) const CLOCK: bool = !cfg!(all(target_family = "wasm", target_os = "unknown"));

/// The system's page size in bytes; None where the system gives none.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub(crate) fn page_size() -> Option<u64> {
    None
}

/// The system's page size in bytes; None where the system gives none.
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub(crate) fn page_size() -> Option<u64> {
    #[cfg(unix)]
    // SAFETY: sysconf reads a constant of the system and touches no memory
    // of ours.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    #[cfg(windows)]
    let page = {
        let mut info = SYSTEM_INFO::default();
        // SAFETY: GetSystemInfo writes the SYSTEM_INFO it is given and no
        // other memory of ours.
        unsafe { GetSystemInfo(&raw mut info) };
        i64::from(info.dwPageSize)
    };
    u64::try_from(page).ok().filter(|&p| p > 0)
}

// f4-gfp.28 design-checked -- the resident set is read from /proc/self/statm, its second field times the page size, not getrusage's ru_maxrss, a high-water mark (not measured)
// port.3 design-checked -- on macOS the resident set is proc_pidinfo's pti_resident_size, not task_info's resident_size, whose mach_task_self libc deprecates (not measured)
// port.4 design-checked -- on Windows the resident set is the working set GetProcessMemoryInfo reads, not mimalloc's mi_process_info, which reads it by the same call (not measured)
/// The process's resident set: on Linux read from `/proc/self/statm`, on
/// macOS by `proc_pidinfo`, and on Windows the working set. The one reader
/// of it, for the memory ceiling and for the call report's start figure
/// (report-peak-inherited.2).
pub(crate) struct Resident {
    #[cfg(target_os = "linux")]
    file: File,
    /// The page size in bytes.
    #[cfg(target_os = "linux")]
    page: u64,
}

impl Resident {
    /// None where the resident set cannot be read.
    pub(crate) fn open() -> Option<Resident> {
        #[cfg(target_os = "linux")]
        let resident = Resident {
            file: File::open("/proc/self/statm").ok()?,
            page: page_size()?,
        };
        #[cfg(not(target_os = "linux"))]
        let resident = Resident {};
        resident.bytes()?;
        Some(resident)
    }

    /// The resident set in bytes.
    #[cfg(target_os = "linux")]
    pub(crate) fn bytes(&self) -> Option<u64> {
        // The fields are seven counts of pages; 128 bytes hold them.
        let mut buf = [0u8; 128];
        let n = self.file.read_at(&mut buf, 0).ok()?;
        let text = std::str::from_utf8(&buf[..n]).ok()?;
        let pages: u64 = text.split_ascii_whitespace().nth(1)?.parse().ok()?;
        pages.checked_mul(self.page)
    }

    /// The resident set in bytes.
    #[cfg(target_os = "macos")]
    #[allow(
        clippy::unused_self,
        reason = "proc_pidinfo reads the process by its id, which needs no state"
    )]
    pub(crate) fn bytes(&self) -> Option<u64> {
        let size = libc::c_int::try_from(size_of::<libc::proc_taskinfo>()).ok()?;
        // SAFETY: proc_taskinfo holds integers alone, for which zero bytes
        // are a value.
        let mut info: libc::proc_taskinfo = unsafe { mem::zeroed() };
        // SAFETY: proc_pidinfo writes at most `size` bytes, the size of
        // `info`, to `info`, and no other memory of ours.
        let n = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                (&raw mut info).cast(),
                size,
            )
        };
        (n == size).then_some(info.pti_resident_size)
    }

    /// The resident set in bytes.
    #[cfg(windows)]
    #[allow(
        clippy::unused_self,
        reason = "GetProcessMemoryInfo reads the calling process, which needs no state"
    )]
    pub(crate) fn bytes(&self) -> Option<u64> {
        u64::try_from(counters()?.WorkingSetSize).ok()
    }

    /// The resident set in bytes, which a browser does not give.
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    #[allow(
        clippy::unused_self,
        reason = "a browser gives no resident set, which needs no state"
    )]
    pub(crate) fn bytes(&self) -> Option<u64> {
        None
    }
}

/// The process's memory counters, the working set and its peak among them.
#[cfg(windows)]
fn counters() -> Option<PROCESS_MEMORY_COUNTERS> {
    let mut counters = PROCESS_MEMORY_COUNTERS::default();
    let size = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS>()).ok()?;
    // SAFETY: GetCurrentProcess returns a pseudo handle, which needs no
    // closing, and GetProcessMemoryInfo writes at most `size` bytes, the
    // size of `counters`, to `counters`.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &raw mut counters, size) };
    (ok != 0).then_some(counters)
}

/// The bytes of the kB `/proc/self/status` counts in.
#[cfg(target_os = "linux")]
const KIB: u64 = 1024;

// report-peak-inherited.1 design-checked -- the process's peak resident set is VmHWM of /proc/self/status, the kernel's high-water mark of the process's own image, not mimalloc's mi_process_info peak, getrusage's ru_maxrss, into which exec carries the high-water mark of the image the process was started from
/// The process's peak resident set in bytes, `VmHWM` of
/// `/proc/self/status`: the high-water mark since the process began, or
/// since its owner last reset it through `/proc/self/clear_refs`. None
/// where the file cannot be read or holds no `VmHWM` line of a count
/// and `kB`.
#[cfg(target_os = "linux")]
pub(crate) fn peak() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find_map(|l| l.strip_prefix("VmHWM:"))?;
    let mut words = line.split_ascii_whitespace();
    let kib: u64 = words.next()?.parse().ok()?;
    if words.next() != Some("kB") || words.next().is_some() {
        return None;
    }
    kib.checked_mul(KIB)
}

// port.10 design-checked -- on macOS the peak resident set is getrusage's ru_maxrss, in bytes there, not proc_pid_rusage's ri_lifetime_max_phys_footprint, a footprint and not a resident set; whether exec carries the high-water mark of the image the process was started from into it, as Linux does into its ru_maxrss, is not known (not measured)
/// The process's peak resident set in bytes, getrusage's `ru_maxrss`. None
/// where getrusage fails.
#[cfg(target_os = "macos")]
pub(crate) fn peak() -> Option<u64> {
    // SAFETY: rusage holds integers alone, for which zero bytes are a
    // value.
    let mut usage: libc::rusage = unsafe { mem::zeroed() };
    // SAFETY: getrusage writes the rusage it is given and no other memory
    // of ours.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) } != 0 {
        return None;
    }
    u64::try_from(usage.ru_maxrss).ok()
}

// port.11 design-checked -- on Windows the peak resident set is the peak working set, PeakWorkingSetSize, of the GetProcessMemoryInfo call that reads the working set, not a reader of its own (not measured)
/// The process's peak resident set in bytes, the peak working set. None
/// where `GetProcessMemoryInfo` fails.
#[cfg(windows)]
pub(crate) fn peak() -> Option<u64> {
    u64::try_from(counters()?.PeakWorkingSetSize).ok()
}

/// The process's peak resident set in bytes, which a browser does not give.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub(crate) fn peak() -> Option<u64> {
    None
}

/// The resident set in bytes, for the test code.
#[cfg(test)]
pub(crate) fn resident() -> u64 {
    Resident::open()
        .and_then(|r| r.bytes())
        .expect("the resident set is readable here")
}

// port.5 design-checked -- on macOS the pages are given back by madvise(MADV_DONTNEED), as on Linux, not by MADV_FREE_REUSABLE (not measured)
// port.6 design-checked -- on Windows the pages are given back by DiscardVirtualMemory, not by VirtualAlloc with MEM_RESET (not measured)
/// Gives `pages` back to the system: whole pages of `page` bytes. Their
/// bytes afterwards may be anything: zero on Linux, kept or zero on macOS
/// and Windows.
///
/// # Errors
///
/// `InvalidInput` where `pages` does not start and end at a page boundary,
/// and the system's error where it refuses.
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub(crate) fn discard(pages: &mut [u8], page: usize) -> io::Result<()> {
    if pages.is_empty() {
        return Ok(());
    }
    // madvise rounds the length up to a whole page and would give back
    // bytes past `pages`.
    if !pages.as_ptr().addr().is_multiple_of(page) || !pages.len().is_multiple_of(page) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{len} bytes at {at:p} are not whole pages of {page} bytes",
                len = pages.len(),
                at = pages.as_ptr()
            ),
        ));
    }
    #[cfg(unix)]
    {
        // SAFETY: `pages` is borrowed mutably and is whole pages. madvise
        // changes no memory but their bytes, which a slice of bytes may
        // hold whatever they are (madvise-2, MADV_DONTNEED).
        let r =
            unsafe { libc::madvise(pages.as_mut_ptr().cast(), pages.len(), libc::MADV_DONTNEED) };
        if r != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(windows)]
    {
        // SAFETY: `pages` is borrowed mutably and is whole pages.
        // DiscardVirtualMemory changes no memory but their bytes, which a
        // slice of bytes may hold whatever they are.
        let r = unsafe { DiscardVirtualMemory(pages.as_mut_ptr().cast(), pages.len()) };
        if r != 0 {
            return Err(io::Error::from_raw_os_error(r.cast_signed()));
        }
    }
    Ok(())
}

/// `Unsupported`: a wasm32 memory grows and never shrinks.
///
/// # Errors
///
/// Always.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub(crate) fn discard(_pages: &mut [u8], _page: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "wasm32 in a browser gives no pages back",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_tests::measures_here;

    /// In a child process of this test binary, which runs this test alone, a
    /// read reads the resident set anew: 64 MiB written raise it by about
    /// that much. The parent runs the child.
    #[test]
    fn the_resident_set_follows_allocations() {
        if !measures_here("os::tests::the_resident_set_follows_allocations") {
            return;
        }
        let resident = Resident::open().expect("the resident set is readable here");
        let before = resident.bytes().unwrap();
        let block = std::hint::black_box(vec![1u8; 64 << 20]);
        let after = resident.bytes().unwrap();
        assert!(after >= before + (60 << 20), "{before} B, then {after} B");
        drop(block);
    }

    /// Whole pages are given back, an empty range is nothing to give back,
    /// and a range off a page boundary at either end is refused.
    #[test]
    fn discard_takes_whole_pages_alone() {
        let page = usize::try_from(page_size().unwrap()).unwrap();
        let mut bytes = vec![1u8; 4 * page];
        let base = bytes.as_ptr().addr();
        let start = base.next_multiple_of(page) - base;
        let pages = &mut bytes[start..start + 2 * page];
        discard(pages, page).unwrap();
        discard(&mut pages[..0], page).unwrap();
        for ragged in [1..page, 0..page - 1] {
            let e = discard(&mut pages[ragged.clone()], page).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput, "{ragged:?}: {e}");
        }
    }
}
