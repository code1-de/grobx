//! test-build-time, the item's test code for grobx-checker: the suite
//! lists every test of the inventory, the tests its 21 integration test
//! binaries ran but those removed on purpose since (test-build-time.7).

use std::collections::BTreeSet;
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
