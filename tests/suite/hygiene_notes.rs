//! hygiene, sub-item 1's test code: every file that says in its module doc
//! comment that it is skipped holds the skip mark, and the checker names
//! the four literals of `checker-literals`
//! (`iterations/0147-hygiene/plan.md`, _Test code_, 1 and 3).

use crate::common::sources::{literals, stale_skip_notes, workspace_sources};

// hygiene.5 design-checked-- no test runs rustfmt: the item's last sub-item runs `cargo fmt --all -- --check`, and its iteration log holds the output, not a test of the suite that runs the check (not measured)

/// Each file of the checker with the literals `checker-literals` names, as
/// its code wrote them.
const LITERALS: [(&str, &[&str]); 3] = [
    (
        "checker/src/divisors.rs",
        &["2 * self.values.len() > self.slots.len()", "h >> 32"],
    ),
    ("checker/src/cheap.rs", &["vec![false; 1 << 16]"]),
    ("checker/src/parse.rs", &["(1 << 31) - 1"]),
];

// hygiene.4 design-checked -- the skip-note scan and the copy scan run as tests of grobx's suite beside the rule scans, over `workspace_sources`, not as a script outside the test command (not measured)
/// Test 1: no file of the workspace says it is skipped without the mark.
#[test]
fn every_file_that_says_it_is_skipped_holds_the_mark() {
    let stale = stale_skip_notes(&workspace_sources());
    assert!(stale.is_empty(), "{stale:#?}");
}

/// Test 3: none of the checker's four literals stands in its code.
#[test]
fn the_checker_names_its_literals() {
    let found: Vec<String> = LITERALS
        .iter()
        .flat_map(|&(path, needles)| literals(path, needles))
        .collect();
    assert!(found.is_empty(), "{found:#?}");
}
