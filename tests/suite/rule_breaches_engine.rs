//! rule-breaches, sub-item 1: production code asserts nothing but the two
//! guards of rule-breaches.8, chooses between an enum's variants by no
//! `==`, `!=`, `matches!`, or `let`, and takes no catch-all arm on an enum
//! (`iterations/0089-rule-breaches/plan.md`, _Test code_, tests 2 and 3).

use crate::common::sources::{
    Enums, Source, production_asserts, variant_choices, wildcard_arms, workspace_sources,
};

/// Every package's production code: its files that are no test code.
fn production_files() -> Vec<Source> {
    workspace_sources()
        .into_iter()
        .filter(|source| !source.is_test())
        .collect()
}

/// Test 2: the only assertions of production code are the guards of
/// `Table::mul` and of a matrix's columns (rule-breaches.6 to .8).
#[test]
fn production_code_asserts_its_two_guards_alone() {
    let found = production_asserts(&production_files());
    let files: Vec<&str> = found
        .iter()
        .map(|place| place.split(':').next().unwrap_or(""))
        .collect();
    assert_eq!(files, ["src/matrix.rs", "src/table.rs"], "{found:#?}");
    assert!(
        found.iter().all(|place| place.ends_with(" assert!")),
        "{found:#?}"
    );
}

/// Test 3: production code, its `#[cfg(test)]` items among it, names every
/// variant it chooses between (rule-breaches.3 to .5).
#[test]
fn production_code_names_every_variant() {
    let enums = Enums::of(&workspace_sources());
    let files = production_files();
    let choices = variant_choices(&files, &enums);
    assert!(choices.is_empty(), "{choices:#?}");
    let arms = wildcard_arms(&files, &enums);
    assert!(arms.is_empty(), "{arms:#?}");
}
