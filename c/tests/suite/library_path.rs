//! c-suite-library-path, the item's test code (plan.md, _Test code_): the
//! suite's programs run the `libgrobxc.so` that the suite's own build made,
//! whatever other copy lies first on the library path.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::iter;
use std::path::{Path, PathBuf};
use std::process::Command;

use grobx::MonomialOrder;

use crate::common::{
    Args, CC, CXX, HEADER_DIR, PROGRAMS, WARNINGS, lib_file, program, run_command, scratch,
    write_instance,
};
use crate::systems::dense;

/// The version `stale.c` reports.
const STALE: &str = "stale";

// c-suite-library-path.4 practice-checked: test stub -- the library other than the build's is a stand-in built from stale.c, whose version is no crate's, not a library of another version of the crate (log 0119-1)
/// A directory of its own under `dir` that holds `stale.c` built as a
/// shared library of the interface's name.
fn stale_dir(dir: &Path) -> PathBuf {
    let stale = dir.join("stale");
    fs::create_dir_all(&stale).unwrap();
    run_command(
        Command::new(CC)
            .args(["-std=c11", "-shared", "-fPIC"])
            .args(WARNINGS)
            .args(["-I", HEADER_DIR])
            .arg(Path::new(PROGRAMS).join("stale.c"))
            .arg("-o")
            .arg(stale.join(lib_file())),
    );
    stale
}

/// `LD_LIBRARY_PATH` with `dir` first and this process's entries after it.
/// An empty entry, which the loader reads as the working directory, is
/// left out.
fn path_with(dir: &Path) -> OsString {
    let inherited = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    let dirs = iter::once(dir.to_path_buf())
        .chain(env::split_paths(&inherited).filter(|d| !d.as_os_str().is_empty()));
    env::join_paths(dirs).unwrap()
}

// c-suite-library-path.6 design-checked -- test 1 runs linkage.cpp and the C driver, the C++ compiler's program and the C compiler's, not every program of the suite, which shares their link (not measured)
/// Test 1: with a directory first on `LD_LIBRARY_PATH` that holds a
/// `libgrobxc.so` other than the build's, `linkage.cpp` prints the crate
/// version and the C driver's version line names it.
#[test]
fn a_stale_library_first_on_the_path() {
    let dir = scratch("library_path");
    let path = path_with(&stale_dir(&dir));

    let out = run_command(Command::new(program("linkage.cpp")).env("LD_LIBRARY_PATH", &path));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("{version}\n", version = grobx::VERSION),
        "linkage.cpp"
    );

    let basis = dir.join("out");
    fs::create_dir_all(&basis).unwrap();
    let instance = write_instance(&dir, &dense(3, 3, 65521, 1));
    let args = Args::new(MonomialOrder::Grevlex, 1);
    let out = run_command(
        Command::new(program("driver.c"))
            .args(args.argv(&instance, &basis))
            .env("LD_LIBRARY_PATH", &path),
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let want = format!("version {version}", version = grobx::VERSION);
    assert!(stdout.lines().any(|l| l == want), "driver.c: {stdout}");
}

// c-suite-library-path.5 practice-checked: positive control -- a control shows that the loader takes the stand-in where a program finds the library by the path, not test 1 alone, which a stand-in the loader passes over would let pass (log 0119-1)
/// Test 2: `stale.c`'s library stands in for the build's where the loader
/// finds the library by the path: `linkage.cpp` linked against it without a
/// path of its own, run with its directory first on `LD_LIBRARY_PATH`,
/// prints its version. So test 1 passes only where the suite's programs
/// take the build's library.
#[test]
fn the_stale_library_is_taken_by_the_path() {
    let dir = scratch("library_path-control");
    let stale = stale_dir(&dir);
    let exe = dir.join("linkage");
    run_command(
        Command::new(CXX)
            .arg("-std=c++17")
            .args(WARNINGS)
            .args(["-I", HEADER_DIR])
            .arg(Path::new(PROGRAMS).join("linkage.cpp"))
            .arg("-L")
            .arg(&stale)
            .arg("-lgrobxc")
            .arg("-o")
            .arg(&exe),
    );
    let out = run_command(Command::new(&exe).env("LD_LIBRARY_PATH", path_with(&stale)));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), format!("{STALE}\n"));
}
