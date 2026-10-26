//! rule-breaches, sub-item 4: the crate's integration tests under `tests/`
//! name their format arguments, order their `use`s, and name the variants
//! they choose between (`iterations/0089-rule-breaches/plan.md`, _Test
//! code_, test 11).

use crate::common::sources::{
    Enums, misplaced_uses, positional_formats, rust_sources, variant_choices, wildcard_arms,
    workspace_sources,
};

/// Test 11: the four scans over `tests/` find nothing.
#[test]
fn the_integration_tests_follow_the_rules() {
    let sources = rust_sources(&["tests"]);
    let formats = positional_formats(&sources);
    assert!(formats.is_empty(), "{formats:#?}");
    let uses = misplaced_uses(&sources);
    assert!(uses.is_empty(), "{uses:#?}");
    let enums = Enums::of(&workspace_sources());
    let choices = variant_choices(&sources, &enums);
    assert!(choices.is_empty(), "{choices:#?}");
    let arms = wildcard_arms(&sources, &enums);
    assert!(arms.is_empty(), "{arms:#?}");
}
