//! hygiene, sub-item 4's test code: the root package's unit tests hold no
//! copies of a function (`test-helper-copies`;
//! `iterations/0147-hygiene/plan.md`, _Test code_, 11).

use crate::common::sources::{HelperKind, copies, workspace_sources};

/// Test 11: no two unit test files of the root package define copies of
/// one function (hygiene.7).
#[test]
fn the_unit_tests_share_their_functions() {
    let unit_tests: Vec<_> = workspace_sources()
        .into_iter()
        .filter(|source| source.is_test() && source.krate().is_empty())
        .collect();
    let found = copies(&unit_tests, &[HelperKind::Fn]);
    assert!(found.is_empty(), "{n} groups: {found:#?}", n = found.len());
}
