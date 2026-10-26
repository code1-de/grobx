//! test-build-time, the item's test code for grobx: the suite lists every
//! test of the inventory, the tests the 32 integration test binaries ran
//! but those removed on purpose since (test-build-time.7), every test
//! file of the workspace is a module of its package's suite
//! (test-build-time.6), and grobx-python's lib holds no unit test, since
//! cargo compiles it as no test binary (test-build-time.5).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The tests this binary lists, by `<this binary> --list`, each as its
/// path within the binary.
fn listed() -> BTreeSet<String> {
    let exe = std::env::current_exe().expect("the test binary knows its path");
    let out = Command::new(&exe)
        .arg("--list")
        .output()
        .unwrap_or_else(|e| panic!("{exe}: {e}", exe = exe.display()));
    assert!(
        out.status.success(),
        "{exe} --list: {out:?}",
        exe = exe.display()
    );
    String::from_utf8(out.stdout)
        .expect("--list prints UTF-8")
        .lines()
        .filter_map(|l| l.strip_suffix(": test"))
        .map(str::to_string)
        .collect()
}

/// The inventory's tests, each as the suite names it: `<binary>::<path>`.
fn inventory() -> Vec<String> {
    include_str!("inventory.txt")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (binary, path) = l
                .split_once('\t')
                .unwrap_or_else(|| panic!("an inventory line without a tab: {l:?}"));
            format!("{binary}::{path}")
        })
        .collect()
}

// test-build-time.7 design -- an inventory of the tests the integration test binaries listed before the move, checked against the suite's own --list, not the lists compared once by hand (not measured)
#[test]
fn every_test_of_the_inventory_is_listed() {
    let listed = listed();
    let missing: Vec<String> = inventory()
        .into_iter()
        .filter(|t| !listed.contains(t))
        .collect();
    assert!(
        missing.is_empty(),
        "{count} tests of inventory.txt are not in the suite: {missing:#?}",
        count = missing.len()
    );
}

/// The `tests/` directory of each package of the workspace.
fn test_dirs() -> [PathBuf; 5] {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    [
        root.join("tests"),
        root.join("checker").join("tests"),
        root.join("python").join("tests"),
        root.join("cli").join("tests"),
        root.join("c").join("tests"),
    ]
}

/// The entries of `dir`, sorted.
fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}", dir = dir.display()))
        .map(|e| e.expect("a directory entry").path())
        .collect();
    found.sort();
    found
}

fn is_rust(path: &Path) -> bool {
    path.is_file() && path.extension().is_some_and(|x| x == "rs")
}

/// The modules `main` declares by a line `mod <name>;`.
fn declared(main: &str) -> BTreeSet<&str> {
    main.lines()
        .filter_map(|l| l.trim().strip_prefix("mod ")?.strip_suffix(';'))
        .collect()
}

// test-build-time.6 design -- a test fails where a test file lies outside its package's suite or a file of the suite is not declared, not a comment in main.rs asking for it (log 0019-1)
#[test]
fn every_test_file_is_a_module_of_the_suite() {
    for tests in test_dirs() {
        let loose: Vec<PathBuf> = entries(&tests).into_iter().filter(|p| is_rust(p)).collect();
        assert!(
            loose.is_empty(),
            "{tests}: test files outside tests/suite/, each compiled as a test binary of its own: \
             {loose:#?}. Move each into tests/suite/ and declare it in tests/suite/main.rs",
            tests = tests.display()
        );
        let suite = tests.join("suite");
        for dir in entries(&tests).into_iter().filter(|p| p.is_dir()) {
            assert!(
                dir == suite || !dir.join("main.rs").exists(),
                "{dir}: a test binary beside tests/suite/",
                dir = dir.display()
            );
        }
        let main = fs::read_to_string(suite.join("main.rs"))
            .unwrap_or_else(|e| panic!("{suite}/main.rs: {e}", suite = suite.display()));
        let declared = declared(&main);
        let undeclared: Vec<PathBuf> = entries(&suite)
            .into_iter()
            .filter(|p| is_rust(p) && p.file_name().is_some_and(|n| n != "main.rs"))
            .filter(|p| {
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                !declared.contains(stem)
            })
            .collect();
        assert!(
            undeclared.is_empty(),
            "{suite}: files no `mod` line of main.rs declares, so nothing compiles \
             them: {undeclared:#?}",
            suite = suite.display()
        );
    }
}

/// The `.rs` files under `dir`, in its subdirectories too.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for path in entries(dir) {
        if path.is_dir() {
            found.extend(rust_files(&path));
        } else if is_rust(&path) {
            found.push(path);
        }
    }
    found
}

/// The lines of the TOML table whose header line is `name`, such as
/// `[lib]`, in `manifest`, up to the next table.
fn table<'a>(manifest: &'a str, name: &str) -> Vec<&'a str> {
    manifest
        .lines()
        .skip_while(|l| l.trim() != name)
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .collect()
}

// test-build-time.5 design -- grobx-python's lib, which holds no unit test, is compiled as no test binary, and a test fails where its source gains one (log 0019-1)
#[test]
fn the_python_lib_is_no_test_binary() {
    let python = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let manifest = fs::read_to_string(python.join("Cargo.toml")).expect("python/Cargo.toml");
    assert!(
        table(&manifest, "[lib]")
            .iter()
            .any(|l| l.split_whitespace().collect::<String>() == "test=false"),
        "the [lib] table of python/Cargo.toml has no `test = false`"
    );
    for file in rust_files(&python.join("src")) {
        let text = fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("{file}: {e}", file = file.display()));
        assert!(
            !text.contains("#[test]") && !text.contains("#[cfg(test)]"),
            "{file}: a unit test in grobx-python's lib, which cargo compiles as no test binary \
             (`test = false` in python/Cargo.toml); its tests belong in python/tests/suite/",
            file = file.display()
        );
    }
}
