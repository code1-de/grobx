//! hygiene, the item's test code: no test crate of the workspace holds
//! copies of a helper (`test-helper-copies`; `iterations/0147-hygiene/plan.md`,
//! _Test code_, 12).

use crate::common::sources::{HelperKind, copies, workspace_sources};

// hygiene.4 design-checked -- the skip-note scan and the copy scan run as tests of grobx's suite beside the rule scans, over `workspace_sources`, not as a script outside the test command (not measured)
/// Test 12: in no test crate do two files define copies of one helper, of
/// any kind (hygiene.7).
#[test]
fn no_test_crate_holds_copies() {
    let found = copies(&workspace_sources(), &HelperKind::ALL);
    assert!(found.is_empty(), "{n} groups: {found:#?}", n = found.len());
}
