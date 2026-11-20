//! A test that reads the process's resident set reads it in a child
//! process of the test binary that runs the one test alone: what the other
//! tests take and free on the harness's threads then moves no figure the
//! test reads. The parent runs the child and asserts that it passed
//! (`BACKLOG.md`, `resident-set-tests-race`; log 0102-3, _Step 6_).

use std::process::Command;

/// The variable that tells the child process to measure.
const CHILD: &str = "GROBX_TEST_CHILD";

/// Whether this process is the child that measures. In the parent, runs
/// `test`, the test's path in the crate, alone in a child process of this
/// test binary, asserts that the child passed it, and answers false.
pub(crate) fn measures_here(test: &str) -> bool {
    if std::env::var_os(CHILD).is_some() {
        return true;
    }
    let exe = std::env::current_exe().expect("the test binary knows its path");
    let out = Command::new(&exe)
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap_or_else(|e| panic!("{exe}: {e}", exe = exe.display()));
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the child failed: {out:?}");
    assert!(said.contains("1 passed"), "the child ran no test: {said}");
    false
}
