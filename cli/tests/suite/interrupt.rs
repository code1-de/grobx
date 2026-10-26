//! The item's test code, cli: SIGINT ends a process that reads its
//! instance, raises the cancel flag once the instance is read, a second
//! one ends the process, and an ignored one, sent while the process reads
//! its instance, stays ignored (cli.18 to .21).
//! A call that SIGINT stops runs on `dense(8, 3, 65521, 1)` or
//! `dense(5, 3, 0, 1)`, which run for seconds without it. A process that
//! a SIGINT before the call leaves running waits to open `--output`, a
//! FIFO the test has not opened yet.

use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::thread;
use std::time::Duration;

use grobx::{MonomialOrder, Params, Threads};

use crate::common::{
    command, crate_call, fifo, finish, ignoring_sigint, scratch, sigint, sigint_in, until,
    write_instance,
};
use crate::systems::dense;

#[test]
fn sigint_stops_a_running_call() {
    let long = write_instance(&scratch("sigint-running"), "long", &dense(8, 3, 65521, 1));
    let child = command(&[OsStr::new("--seed"), "1".as_ref(), long.as_os_str()])
        .spawn()
        .expect("the binary starts");
    until("SIGINT blocked", || sigint_in(child.id(), "SigBlk"));
    thread::sleep(Duration::from_millis(300));
    sigint(&child);
    let ran = finish(child, Duration::from_secs(60));
    assert_eq!(ran.code(), 1, "{ran:?}");
    assert_eq!(ran.stdout, "");
    assert_eq!(ran.stderr, "grobx: stopped: cancel\n");
}

#[test]
fn sigint_during_the_read_ends_the_process() {
    let child = command(&["--seed", "1"])
        .spawn()
        .expect("the binary starts");
    thread::sleep(Duration::from_millis(300));
    assert!(
        !sigint_in(child.id(), "SigBlk"),
        "the watch started before the instance was read"
    );
    sigint(&child);
    let ran = finish(child, Duration::from_secs(10));
    assert_eq!(ran.status.signal(), Some(libc::SIGINT), "{ran:?}");
    assert_eq!(ran.stdout, "");
    assert_eq!(ran.stderr, "");
}

#[test]
fn sigint_before_the_call_stops_it() {
    let dir = scratch("sigint-before");
    for (stem, text) in [("p65521", dense(8, 3, 65521, 1)), ("q", dense(5, 3, 0, 1))] {
        let instance = write_instance(&dir, stem, &text);
        let output = fifo(&dir, &format!("{stem}.out"));
        let child = command(&[
            OsStr::new("--seed"),
            "1".as_ref(),
            "--output".as_ref(),
            output.as_os_str(),
            instance.as_os_str(),
        ])
        .spawn()
        .expect("the binary starts");
        let pid = child.id();
        until("SIGINT blocked", || sigint_in(pid, "SigBlk"));
        sigint(&child);
        until("SIGINT taken", || !sigint_in(pid, "ShdPnd"));
        thread::sleep(Duration::from_millis(100));
        let mut basis = File::open(&output).unwrap();
        let ran = finish(child, Duration::from_secs(60));
        let mut written = String::new();
        basis.read_to_string(&mut written).unwrap();
        assert_eq!(ran.code(), 1, "{stem}: {ran:?}");
        assert_eq!(written, "", "{stem}");
        assert_eq!(ran.stdout, "", "{stem}");
        assert_eq!(ran.stderr, "grobx: stopped: cancel\n", "{stem}");
    }
}

#[test]
fn a_second_sigint_ends_the_process() {
    let dir = scratch("sigint-second");
    let instance = write_instance(&dir, "short", &dense(3, 3, 65521, 1));
    let output = fifo(&dir, "short.out");
    let mut child = command(&[
        OsStr::new("--seed"),
        "1".as_ref(),
        "--output".as_ref(),
        output.as_os_str(),
        instance.as_os_str(),
    ])
    .spawn()
    .expect("the binary starts");
    let pid = child.id();
    until("SIGINT blocked", || sigint_in(pid, "SigBlk"));
    sigint(&child);
    until("SIGINT taken", || !sigint_in(pid, "ShdPnd"));
    thread::sleep(Duration::from_millis(100));
    assert!(
        child.try_wait().unwrap().is_none(),
        "the first SIGINT ended the process"
    );
    sigint(&child);
    let ran = finish(child, Duration::from_secs(10));
    assert_eq!(ran.status.signal(), Some(libc::SIGINT), "{ran:?}");
}

#[test]
fn an_ignored_sigint_stays_ignored() {
    let text = dense(3, 3, 65521, 1);
    let mut child = ignoring_sigint(&["--seed", "1"])
        .spawn()
        .expect("the binary starts");
    thread::sleep(Duration::from_millis(300));
    sigint(&child);
    thread::sleep(Duration::from_millis(100));
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(text.as_bytes()).unwrap();
    drop(stdin);
    let ran = finish(child, Duration::from_secs(60));
    let params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(std::num::NonZeroUsize::MIN),
        1,
    );
    let expected = crate_call(&text, &params);
    assert_eq!(ran.code(), 0, "{ran:?}");
    assert_eq!(Some(ran.stdout), expected.basis);
    assert_eq!(ran.stderr, "");
}
