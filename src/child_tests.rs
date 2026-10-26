//! Test code that runs one test of the test binary alone in a child
//! process. A test that measures the process as a whole, such as its
//! resident set, measures nothing where the binary's other tests allocate
//! and free in the same process meanwhile.

use std::process::Command;

/// The variable that tells a child process it is the child.
const CHILD: &str = "GROBX_TEST_CHILD";

/// Whether this process is a child that `run` started.
pub(crate) fn running() -> bool {
    std::env::var_os(CHILD).is_some()
}

/// Runs `test`, a test's path in this binary, alone in a child process of
/// the test binary, where `running` holds. Fails where the child fails or
/// runs no test.
pub(crate) fn run(test: &str) {
    let exe = std::env::current_exe().expect("the test binary knows its path");
    let out = Command::new(&exe)
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap_or_else(|e| panic!("{exe}: {e}", exe = exe.display()));
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the child failed: {out:?}");
    assert!(said.contains("1 passed"), "the child ran no test: {said}");
}
