//! Ctrl-C raises the call's cancel flag, and a second Ctrl-C ends the
//! process.

use std::io;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use windows_sys::Win32::Foundation::{FALSE, TRUE};
use windows_sys::Win32::System::Console::{CTRL_C_EVENT, SetConsoleCtrlHandler};
use windows_sys::core::BOOL;

/// The flag the first Ctrl-C raises.
static CANCEL: OnceLock<grobx::Cancel> = OnceLock::new();
/// Whether a Ctrl-C has come since the watch started.
static TAKEN: AtomicBool = AtomicBool::new(false);

// port.7 design -- Ctrl-C is taken by a console control handler, which the system runs in a thread of its own, not by the C runtime's signal(SIGINT) (not measured)
/// Has Ctrl-C raise `cancel`, and a second Ctrl-C end the process. Where a
/// parent started the process with Ctrl-C ignored, the system calls no
/// handler for it, and it stays ignored (cli.20).
pub fn watch(cancel: &grobx::Cancel) -> io::Result<()> {
    if CANCEL.set(cancel.clone()).is_err() {
        return Err(io::Error::other("the watch on Ctrl-C started twice"));
    }
    // SAFETY: `handler` lives as long as the program and reads statics
    // alone.
    if unsafe { SetConsoleCtrlHandler(Some(handler), TRUE) } == FALSE {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

// port.8 design -- a second Ctrl-C goes on to the system's handler, which ends the process by Ctrl-C as cli.19 has a second SIGINT do, not exit(130) (not measured)
/// The first Ctrl-C raises the flag. A second one, and every other event,
/// goes on to the system's handler, which ends the process.
extern "system" fn handler(event: u32) -> BOOL {
    if event != CTRL_C_EVENT || TAKEN.swap(true, Ordering::Relaxed) {
        return FALSE;
    }
    let Some(cancel) = CANCEL.get() else {
        return FALSE;
    };
    cancel.set();
    TRUE
}
