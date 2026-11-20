//! hygiene, sub-item 3's test code: the root package's unit tests hold no
//! copies of a constant, a static, or a type (`test-helper-copies`;
//! `iterations/0147-hygiene/plan.md`, _Test code_, 10).

use crate::common::sources::{HelperKind, copies, workspace_sources};

/// Test 10: no two unit test files of the root package define copies of
/// one constant, static, or type (hygiene.7).
#[test]
fn the_unit_tests_share_their_constants_and_types() {
    let unit_tests: Vec<_> = workspace_sources()
        .into_iter()
        .filter(|source| source.is_test() && source.krate().is_empty())
        .collect();
    let found = copies(
        &unit_tests,
        &[
            HelperKind::Const,
            HelperKind::Static,
            HelperKind::Struct,
            HelperKind::Enum,
            HelperKind::Type,
        ],
    );
    assert!(found.is_empty(), "{n} groups: {found:#?}", n = found.len());
}
