//! SIGINT raises the call's cancel flag, and a second SIGINT ends the
//! process.

use std::io;
use std::mem;
use std::ptr;
use std::thread;

use grobx_cli::ERROR;

// cli.18 practice-checked: dedicated signal thread -- SIGINT is blocked in the main thread before any other thread starts and a thread of its own takes it by sigwait, not a handler that sets a static flag a thread copies to the cancel flag
/// Has SIGINT raise `cancel`, and a second SIGINT end the process, where
/// SIGINT is not ignored at the start. Called from the main thread before
/// any other thread of the process exists, so that every thread inherits
/// SIGINT blocked. A failure of the thread that takes SIGINT is said on
/// standard error and ends the process with exit status `ERROR`.
pub fn watch(cancel: &grobx::Cancel) -> io::Result<()> {
    // SAFETY: sigaction with a null new action reads SIGINT's disposition
    // into `old`, a zeroed sigaction, and changes nothing.
    let old = unsafe {
        let mut old: libc::sigaction = mem::zeroed();
        if libc::sigaction(libc::SIGINT, ptr::null(), &raw mut old) != 0 {
            return Err(io::Error::last_os_error());
        }
        old
    };
    // cli.20 practice-checked: inherited SIG_IGN kept -- where SIGINT is ignored at the start it stays ignored and nothing watches it, not a watch whatever its disposition
    if old.sa_sigaction == libc::SIG_IGN {
        return Ok(());
    }
    let set = sigint()?;
    // SAFETY: pthread_sigmask adds SIGINT to the calling thread's mask and
    // writes no old mask.
    let r = unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &raw const set, ptr::null_mut()) };
    if r != 0 {
        return Err(io::Error::from_raw_os_error(r));
    }
    let cancel = cancel.clone();
    thread::Builder::new()
        .name("sigint".to_string())
        .spawn(move || {
            let failed = match wait(&set).and_then(|()| {
                cancel.set();
                wait(&set)
            }) {
                Ok(()) => end(&set),
                Err(e) => e,
            };
            crate::say(&format!("SIGINT: {failed}"));
            std::process::exit(i32::from(ERROR));
        })?;
    Ok(())
}

// silent-fallbacks.1 design-checked -- a failure of sigemptyset or sigaddset is the watch's error, not dropped, nor an expect on SIGINT being a valid signal number
/// The signal set that holds SIGINT alone.
fn sigint() -> io::Result<libc::sigset_t> {
    // SAFETY: sigemptyset initialises the zeroed set, and sigaddset adds a
    // valid signal number to it.
    unsafe {
        let mut set: libc::sigset_t = mem::zeroed();
        if libc::sigemptyset(&raw mut set) != 0 || libc::sigaddset(&raw mut set, libc::SIGINT) != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(set)
    }
}

// silent-fallbacks.2 design-checked -- a failed sigwait is an error that ends the process, not a sigwait called again, which a failure that recurs would hang on, nor the thread ended, which would leave SIGINT blocked in every thread and taken by none
/// Waits until SIGINT, blocked in the calling thread, is taken.
fn wait(set: &libc::sigset_t) -> io::Result<()> {
    let mut sig = 0;
    // SAFETY: `set` is an initialised signal set, and sigwait writes the
    // signal it takes to `sig`.
    let r = unsafe { libc::sigwait(set, &raw mut sig) };
    if r != 0 {
        return Err(io::Error::from_raw_os_error(r));
    }
    debug_assert_eq!(sig, libc::SIGINT, "the set holds SIGINT alone");
    Ok(())
}

// cli.19 practice-checked: re-raising a signal by its default action (Cracauer, Proper handling of SIGINT/SIGQUIT) -- a second SIGINT ends the process by SIGINT's default action, not by exit(130) or not at all (not measured)
// silent-fallbacks.3 design-checked -- where a step fails, or the process outlives raise, the thread says why and exits with ERROR, not exit(130) with or without a word
/// Ends the process by SIGINT's default action: restores it, unblocks
/// SIGINT in the calling thread, and raises SIGINT there, whose default
/// action ends the process before `raise` returns. Returns only where a
/// step failed, with its error.
fn end(set: &libc::sigset_t) -> io::Error {
    // SAFETY: signal restores SIGINT's default action.
    if unsafe { libc::signal(libc::SIGINT, libc::SIG_DFL) } == libc::SIG_ERR {
        return io::Error::last_os_error();
    }
    // SAFETY: pthread_sigmask removes the initialised set `set` from the
    // calling thread's mask and writes no old mask.
    let r = unsafe { libc::pthread_sigmask(libc::SIG_UNBLOCK, set, ptr::null_mut()) };
    if r != 0 {
        return io::Error::from_raw_os_error(r);
    }
    // SAFETY: raise sends SIGINT to the calling thread.
    if unsafe { libc::raise(libc::SIGINT) } != 0 {
        return io::Error::last_os_error();
    }
    io::Error::other("the process outlived SIGINT's default action")
}
