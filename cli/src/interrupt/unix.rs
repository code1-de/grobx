//! SIGINT raises the call's cancel flag, and a second SIGINT ends the
//! process.

use std::io;
use std::mem;
use std::ptr;
use std::thread;

// cli.18 practice: dedicated signal thread -- SIGINT is blocked in the main thread before any other thread starts and a thread of its own takes it by sigwait, not a handler that sets a static flag a thread copies to the cancel flag
/// Has SIGINT raise `cancel`, and a second SIGINT end the process, where
/// SIGINT is not ignored at the start. Called from the main thread before
/// any other thread of the process exists, so that every thread inherits
/// SIGINT blocked.
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
    // cli.20 practice: inherited SIG_IGN kept -- where SIGINT is ignored at the start it stays ignored and nothing watches it, not a watch whatever its disposition
    if old.sa_sigaction == libc::SIG_IGN {
        return Ok(());
    }
    let set = sigint();
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
            wait(&set);
            cancel.set();
            wait(&set);
            // cli.19 design -- a second SIGINT ends the process by SIGINT's default action, not by exit(130) or not at all (not measured)
            // SAFETY: signal restores SIGINT's default action, the thread
            // unblocks SIGINT in its own mask, and raise sends SIGINT to the
            // thread, whose default action ends the process.
            unsafe {
                libc::signal(libc::SIGINT, libc::SIG_DFL);
                libc::pthread_sigmask(libc::SIG_UNBLOCK, &raw const set, ptr::null_mut());
                libc::raise(libc::SIGINT);
            }
            std::process::exit(130);
        })?;
    Ok(())
}

/// The signal set that holds SIGINT alone.
fn sigint() -> libc::sigset_t {
    // SAFETY: sigemptyset initialises the zeroed set, and sigaddset adds a
    // valid signal number to it.
    unsafe {
        let mut set: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&raw mut set);
        libc::sigaddset(&raw mut set, libc::SIGINT);
        set
    }
}

/// Waits until SIGINT, blocked in the calling thread, is taken.
fn wait(set: &libc::sigset_t) {
    loop {
        let mut sig = 0;
        // SAFETY: `set` is an initialised signal set, and sigwait writes the
        // signal it takes to `sig`.
        let r = unsafe { libc::sigwait(set, &raw mut sig) };
        if r == 0 && sig == libc::SIGINT {
            return;
        }
    }
}
