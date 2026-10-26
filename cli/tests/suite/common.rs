//! Test code shared by the CLI's tests: the binary, an instance file and a
//! FIFO in a scratch directory, a run of the binary, the crate's call for
//! the same arguments, and SIGINT's bit in `/proc/<pid>/status`. The FIFO
//! and SIGINT are Unix's alone.

#[cfg(unix)]
use std::ffi::CString;
use std::fs;
#[cfg(unix)]
use std::io;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use grobx::{Outcome, Params};

/// The binary under test, built by cargo in this test's profile.
pub const BIN: &str = env!("CARGO_BIN_EXE_grobx");

/// A directory of this test's own, empty, under cargo's scratch directory.
pub fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("cli")
        .join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The file `<name>.ms` in `dir`, holding `text`.
pub fn write_instance(dir: &Path, name: &str, text: &str) -> PathBuf {
    let file = dir.join(format!("{name}.ms"));
    fs::write(&file, text).unwrap();
    file
}

/// A FIFO `name` in `dir`. A child that opens it to write waits until the
/// test opens it to read.
#[cfg(unix)]
pub fn fifo(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let c = CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: mkfifo reads the NUL-terminated path `c` holds.
    let r = unsafe { libc::mkfifo(c.as_ptr(), 0o600) };
    let error = io::Error::last_os_error();
    assert_eq!(r, 0, "mkfifo {path}: {error}", path = path.display());
    path
}

/// The binary with `args` and pipes for its three streams. On Unix SIGINT
/// starts at its default action, unblocked, whatever this process
/// inherited.
pub fn command<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Command {
    let mut cmd = Command::new(BIN);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    sigint_as(&mut cmd, libc::SIG_DFL);
    cmd
}

/// As `command`, with SIGINT ignored.
#[cfg(unix)]
pub fn ignoring_sigint<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Command {
    let mut cmd = command(args);
    // The child runs pre_exec's closures in the order they were added, so
    // this one's SIG_IGN follows command's SIG_DFL.
    sigint_as(&mut cmd, libc::SIG_IGN);
    cmd
}

/// Has the child of `cmd` start with SIGINT unblocked and at `action`.
#[cfg(unix)]
fn sigint_as(cmd: &mut Command, action: libc::sighandler_t) {
    // SAFETY: the closure runs in the child between fork and exec and calls
    // sigemptyset, sigaddset, sigprocmask, and signal alone, each
    // async-signal-safe.
    unsafe {
        cmd.pre_exec(move || {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&raw mut set);
            libc::sigaddset(&raw mut set, libc::SIGINT);
            if libc::sigprocmask(libc::SIG_UNBLOCK, &raw const set, std::ptr::null_mut()) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::signal(libc::SIGINT, action) == libc::SIG_ERR {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// What a run of the binary gave.
#[derive(Debug)]
pub struct Ran {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

impl Ran {
    /// The exit status, which a run that a signal ended has not.
    pub fn code(&self) -> i32 {
        self.status
            .code()
            .unwrap_or_else(|| panic!("no exit status: {self:?}"))
    }
}

/// Waits for `child` to end, at most `limit`; kills it past that and
/// fails. Its standard output and standard error are read while it runs,
/// so that a child that writes past a pipe's buffer does not wait on it.
pub fn finish(mut child: Child, limit: Duration) -> Ran {
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                pipe.read_to_end(&mut bytes).unwrap();
            }
            bytes
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as _));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as _));
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the binary ran past {limit:?}");
        }
        thread::sleep(Duration::from_millis(5));
    };
    let stdout = stdout.join().unwrap();
    let stderr = stderr.join().unwrap();
    Ran {
        status,
        stdout: String::from_utf8(stdout).expect("standard output is UTF-8"),
        stderr: String::from_utf8(stderr).expect("standard error is UTF-8"),
    }
}

/// Runs the binary with `args` and `stdin` as its standard input, SIGINT at
/// its default action, and waits at most 120 s.
pub fn grobx<S: AsRef<std::ffi::OsStr>>(args: &[S], stdin: &str) -> Ran {
    grobx_in(args, stdin, None)
}

/// As `grobx`, in the working directory `dir` where it is given.
pub fn grobx_in<S: AsRef<std::ffi::OsStr>>(args: &[S], stdin: &str, dir: Option<&Path>) -> Ran {
    let mut cmd = command(args);
    if let Some(dir) = dir {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn().expect("the binary starts");
    let mut input = child.stdin.take().unwrap();
    let text = stdin.to_string();
    // The binary may exit before it reads, so a broken pipe is no failure.
    let writer = thread::spawn(move || {
        let _ = input.write_all(text.as_bytes());
    });
    let ran = finish(child, Duration::from_secs(120));
    writer.join().unwrap();
    ran
}

/// What the crate's entry point returns for `text` under `params`.
pub fn crate_call(text: &str, params: &Params) -> Outcome {
    grobx::groebner(text, params).unwrap_or_else(|e| panic!("the crate refuses the text: {e}"))
}

/// The line the CLI writes to standard error for the checker's verdict, as
/// the test reads `outcome`.
pub fn verdict(outcome: &Outcome) -> String {
    match outcome.certified {
        None => String::new(),
        Some(true) => "grobx: checker: certified\n".to_string(),
        Some(false) => match &outcome.flaw {
            Some(flaw) => format!("grobx: checker: rejected: {flaw}\n"),
            None => "grobx: checker: rejected\n".to_string(),
        },
    }
}

/// Whether SIGINT's bit is set in the mask `field` of `/proc/<pid>/status`:
/// `SigBlk`, the main thread's blocked signals, or `ShdPnd`, the signals
/// pending for the process.
#[cfg(unix)]
pub fn sigint_in(pid: u32, field: &str) -> bool {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
    let line = status
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{field}:")))
        .unwrap_or_else(|| panic!("/proc/{pid}/status has no {field}"));
    let mask = u64::from_str_radix(line.trim(), 16).unwrap();
    mask & (1 << (libc::SIGINT - 1)) != 0
}

/// Waits until `cond` holds, at most 10 s.
#[cfg(unix)]
pub fn until(what: &str, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while !cond() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "{what} within 10 s"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// Sends SIGINT to `child`.
#[cfg(unix)]
pub fn sigint(child: &Child) {
    // SAFETY: kill sends a signal to the child's process id, which the child
    // holds until it is waited for.
    let r = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) };
    assert_eq!(r, 0, "kill: {error}", error = io::Error::last_os_error());
}
