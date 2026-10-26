//! Ctrl-C raises the call's cancel flag, and a second Ctrl-C ends the
//! process. Unix delivers Ctrl-C as SIGINT, and Windows to a console
//! control handler.

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub use unix::watch;
#[cfg(windows)]
pub use windows::watch;
