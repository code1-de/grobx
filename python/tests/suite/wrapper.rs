//! The item's test code: the Python module end to end. The test builds the
//! extension in this test's profile, into the target directory this test
//! was built in, copies the library as `MODULE` into a scratch directory,
//! and runs `PYTHON` on scripts that import it. Where `PYTHON` does not
//! run, the test fails.

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use grobx_core::VERSION;

use crate::systems::dense;

/// The interpreter the scripts run under.
const PYTHON: &str = if cfg!(windows) { "python" } else { "python3" };
/// The file the module is imported from. Python on Windows imports an
/// extension module from a `.pyd` file alone.
const MODULE: &str = if cfg!(windows) {
    "grobx.pyd"
} else {
    "grobx.so"
};

/// The profile directory this test was built in, the one above `deps/`,
/// where the test binary lies.
fn profile_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    exe.parent().unwrap().parent().unwrap().to_path_buf()
}

/// The directory `MODULE` is imported from.
fn module_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let profile_dir = profile_dir();
        let target = profile_dir.parent().unwrap();
        let name = profile_dir
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        // Cargo builds the profile dev into debug/.
        let profile = if name == "debug" {
            "dev".to_string()
        } else {
            name
        };
        let status = Command::new(env!("CARGO"))
            .args([
                "build",
                "--offline",
                "--profile",
                &profile,
                "-p",
                "grobx-python",
                "--manifest-path",
            ])
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.toml"))
            .arg("--target-dir")
            .arg(target)
            .status()
            .expect("cargo runs");
        assert!(status.success(), "the extension does not build: {status}");
        let built = profile_dir.join(format!("{DLL_PREFIX}grobx{DLL_SUFFIX}"));
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("python-module");
        fs::create_dir_all(&dir).unwrap();
        fs::copy(&built, dir.join(MODULE))
            .unwrap_or_else(|e| panic!("{built}: {e}", built = built.display()));
        dir
    })
}

/// Runs `script` under `PYTHON` with the module importable; fails with
/// its output where it fails.
pub(crate) fn python(script: &str) {
    let out = Command::new(PYTHON)
        .arg("-c")
        .arg(script)
        .env("PYTHONPATH", module_dir())
        .output()
        .unwrap_or_else(|e| panic!("{PYTHON} does not run: {e}"));
    assert!(
        out.status.success(),
        "the script failed: {status}\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
        status = out.status,
        stdout = String::from_utf8_lossy(&out.stdout),
        stderr = String::from_utf8_lossy(&out.stderr)
    );
}

const PRELUDE: &str = r#"
import hashlib, os, threading, time
import grobx
TINY = "x,y\n65521\nx^2+y,\nx*y+1\n"
def raises(kind, f):
    try:
        f()
    except kind:
        return
    raise AssertionError(f"no {kind.__name__}")
"#;

pub(crate) fn run(body: &str) {
    run_with(&[], body);
}

/// As `run`, each `(name, text)` of `texts` bound to the Python variable
/// `name` before `body`. A text is ASCII with no control character but a
/// tab, a line feed, and a carriage return, whose escapes in Rust's `{:?}`
/// are Python's; the function asserts the ASCII alone.
pub(crate) fn run_with(texts: &[(&str, &str)], body: &str) {
    let mut bound = String::new();
    for (name, text) in texts {
        assert!(text.is_ascii(), "{name}: not ASCII");
        bound.push_str(&format!("{name} = {text:?}\n"));
    }
    python(&format!("{PRELUDE}\n{bound}{body}"));
}

/// The call `bin/measure.py` makes, its result, and the digest it takes.
#[test]
fn the_measured_call() {
    run(r#"
r = grobx.groebner(TINY, order="grevlex", threads=1, seed=0)
assert r.basis == "[y^2+65520*x,\nx*y+1,\nx^2+y]:\n", repr(r.basis)
assert r.stopped is None and r.cofactors is None and r.certified is None and r.flaw is None
assert r.digest == hashlib.sha256(r.basis.encode()).hexdigest()
r = grobx.groebner(TINY, order="grevlex", threads=0, seed=2**64 - 1)
assert r.basis.startswith("[y^2")
"#);
}

#[test]
fn keywords_and_their_defaults() {
    run(r#"
r = grobx.groebner(TINY)
assert r.basis == grobx.groebner(TINY, order="grevlex", threads=0, seed=r.seed).basis, repr(r.basis)
assert grobx.groebner(TINY).seed != r.seed, r.seed
assert grobx.groebner(TINY, seed=7).seed == 7
assert grobx.groebner(TINY, seed=2**64 - 1).seed == 2**64 - 1
raises(TypeError, lambda: grobx.groebner(TINY, "grevlex", 1, 1))
raises(Exception, lambda: grobx.groebner(TINY, order="grevlex", threads=1, seed=-1))
raises(Exception, lambda: grobx.groebner(TINY, order="grevlex", threads=1, seed=2**64))
"#);
}

#[test]
fn errors_are_grobx_error() {
    run(r#"
assert issubclass(grobx.Error, ValueError)
raises(grobx.Error, lambda: grobx.groebner("x\n7\nx+y\n", order="grevlex", threads=1, seed=1))
raises(grobx.Error, lambda: grobx.groebner(TINY, order="deglex", threads=1, seed=1))
raises(grobx.Error, lambda: grobx.groebner("x\n4\nx\n", order="grevlex", threads=1, seed=1))
raises(Exception, lambda: grobx.groebner(TINY, order="grevlex", threads=1, seed=1, checker=0))
raises(Exception, lambda: grobx.groebner(TINY, order="grevlex", threads=1, seed=1, checker="sometimes"))
"#);
}

#[test]
fn cofactors_and_the_checker() {
    run_with(
        &[("DENSE", &dense(3, 3, 65521, 1))],
        r#"
r = grobx.groebner(TINY, order="grevlex", threads=1, seed=1, cofactors=True)
assert r.cofactors.startswith("[[") and r.cofactors.endswith("]:\n"), repr(r.cofactors)
assert r.certified is None
for checker in ["exact", 1, 6]:
    r = grobx.groebner(DENSE, order="grevlex", threads=1, seed=1, checker=checker)
    assert r.certified is True and r.flaw is None, (checker, r.certified, r.flaw)
    assert r.cofactors is None
"#,
    );
}

/// The bounds stop calls on `LONG`, `dense(8, 3, 65521, 1)`, which runs
/// for seconds without them. Bounds that are not reached leave the basis of
/// `MID` as the call without them gives it.
#[test]
fn every_bound_stops_the_run() {
    run_with(
        &[
            ("MID", &dense(4, 3, 65521, 1)),
            ("LONG", &dense(8, 3, 65521, 1)),
        ],
        r#"
r = grobx.groebner(LONG, order="grevlex", threads=1, seed=1, deadline=0)
assert r.basis is None and r.stopped == "deadline" and r.digest is None, (r.stopped, r.basis)
r = grobx.groebner(LONG, order="grevlex", threads=1, seed=1, memory=1)
assert r.basis is None and r.stopped == "memory", r.stopped
cancel = grobx.Cancel()
assert not cancel.is_set()
cancel.set()
assert cancel.is_set()
r = grobx.groebner(LONG, order="grevlex", threads=1, seed=1, cancel=cancel)
assert r.basis is None and r.stopped == "cancel", r.stopped
cancel = grobx.Cancel()
timer = threading.Timer(0.1, cancel.set)
start = time.monotonic()
timer.start()
r = grobx.groebner(LONG, order="grevlex", threads=1, seed=1, cancel=cancel)
timer.join()
assert r.basis is None and r.stopped == "cancel", r.stopped
assert time.monotonic() - start < 10, time.monotonic() - start
r = grobx.groebner(MID, order="grevlex", threads=1, seed=1, deadline=3_600_000, memory=2**40, cancel=grobx.Cancel())
assert r.stopped is None and r.basis == grobx.groebner(MID, order="grevlex", threads=1, seed=1).basis
"#,
    );
}

#[test]
fn the_version_is_the_crates() {
    run(&format!(
        "assert grobx.__version__ == {VERSION:?}, grobx.__version__"
    ));
}
