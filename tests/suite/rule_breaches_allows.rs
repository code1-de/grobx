//! rule-breaches, sub-item 3: every suppression gives its reason, the
//! module-wide suppressions of the audit's places are gone, the C
//! interface names no unused dev-dependency, and the production code and
//! the crate's unit tests name their format arguments, order their `use`s,
//! and name the variants they choose between
//! (`iterations/0089-rule-breaches/plan.md`, _Test code_, tests 8 to 10).

use crate::common::sources::{
    Enums, allows, manifest, misplaced_uses, positional_formats, rust_sources, table_keys,
    variant_choices, wildcard_arms, workspace_sources,
};

/// The files whose module-wide suppressions rule-breaches.13 takes away:
/// each suppresses nothing, or what one function holds.
const NO_MODULE_ALLOW: [&str; 4] = [
    "src/cofactor_stop_tests.rs",
    "src/rational_tests.rs",
    "src/read_tests.rs",
    "src/slack_tests.rs",
];

/// Test 8: every suppression of the workspace gives its reason
/// (rule-breaches.14). No module-wide one suppresses `dead_code`, and none
/// stands in the files of `NO_MODULE_ALLOW` (rule-breaches.13).
#[test]
fn every_suppression_gives_its_reason() {
    let found = allows(&workspace_sources());
    let without: Vec<&str> = found
        .iter()
        .filter(|allow| !allow.reason)
        .map(|allow| allow.place.as_str())
        .collect();
    assert!(without.is_empty(), "{without:#?}");
    let dead: Vec<&str> = found
        .iter()
        .filter(|allow| allow.inner && allow.lints.iter().any(|lint| lint == "dead_code"))
        .map(|allow| allow.place.as_str())
        .collect();
    assert!(dead.is_empty(), "{dead:#?}");
    let module: Vec<&str> = found
        .iter()
        .filter(|allow| {
            allow.inner
                && NO_MODULE_ALLOW
                    .iter()
                    .any(|file| allow.place.split(':').next() == Some(*file))
        })
        .map(|allow| allow.place.as_str())
        .collect();
    assert!(module.is_empty(), "{module:#?}");
}

/// Test 9: the C interface's manifest names no `serde_json`, which nothing
/// under `c/` uses.
#[test]
fn the_c_interface_names_no_unused_dev_dependency() {
    let keys = table_keys(&manifest("c"), "dev-dependencies");
    assert!(!keys.iter().any(|key| key == "serde_json"), "{keys:?}");
}

/// Test 10: production code and the crate's unit tests hold no positional
/// format argument (rule-breaches.1), no `use` out of order
/// (rule-breaches.2), and no choice that leaves a variant unnamed
/// (rule-breaches.3 to .5).
#[test]
fn production_code_and_the_unit_tests_follow_the_rules() {
    let sources = rust_sources(&["src", "cli/src", "c/src", "python/src", "checker/src"]);
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
