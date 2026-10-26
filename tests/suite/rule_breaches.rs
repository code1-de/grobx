//! rule-breaches, the item: the whole workspace names its format
//! arguments, orders its `use`s, and names the variants it chooses
//! between, and the cases of `q_tuning.rs` have names of their own
//! (`iterations/0089-rule-breaches/plan.md`, _Test code_, tests 12 and 13).

use std::collections::BTreeSet;

use crate::common::sources::{
    Enums, misplaced_uses, positional_formats, variant_choices, wildcard_arms, workspace_sources,
};
use crate::q_tuning;

/// Test 12: the four scans over every package's code, test code among it,
/// find nothing.
#[test]
fn the_workspace_follows_the_rules() {
    let sources = workspace_sources();
    let formats = positional_formats(&sources);
    assert!(formats.is_empty(), "{formats:#?}");
    let uses = misplaced_uses(&sources);
    assert!(uses.is_empty(), "{uses:#?}");
    let enums = Enums::of(&sources);
    let choices = variant_choices(&sources, &enums);
    assert!(choices.is_empty(), "{choices:#?}");
    let arms = wildcard_arms(&sources, &enums);
    assert!(arms.is_empty(), "{arms:#?}");
}

/// Test 13: each case of `q_tuning.rs` has a name no other case has, so
/// that a failure names its case.
#[test]
fn the_q_tuning_cases_have_names_of_their_own() {
    let names: Vec<String> = q_tuning::systems()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let distinct: BTreeSet<&String> = names.iter().collect();
    assert_eq!(distinct.len(), names.len(), "{names:?}");
}
