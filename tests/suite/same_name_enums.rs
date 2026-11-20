//! hygiene, sub-item 2's test code: where two packages define enums of
//! one name, the scan of catch-all arms tells which one a `match` chooses
//! between, by the file's `use` paths, the pattern's crate, and the
//! variants its arms name, and says so where it cannot tell
//! (`same-name-enum-scan`; `iterations/0147-hygiene/plan.md`, _Test code_,
//! 4 to 8).

use crate::common::sources::{Enums, Source, wildcard_arms_in};

/// The checker's non-exhaustive `Mode`, as `checker/src/lib.rs` would
/// define it.
const CHECKER_MODE: &str = "#[non_exhaustive]\npub enum Mode {\n    Exact,\n    Cheap,\n}\n";

/// An exhaustive `Mode` of the core, as `src/matrix.rs` would define it.
const CORE_MODE: &str = "pub(crate) enum Mode {\n    Dense,\n    Sparse,\n}\n";

/// What `wildcard_arms_in` finds in the last of `sources`, the enums of all
/// of them read, each finding's words alone.
fn found(sources: &[Source]) -> Vec<String> {
    let enums = Enums::of(sources);
    let last = sources.last().expect("a source to scan");
    wildcard_arms_in(last, &enums)
        .into_iter()
        .map(|(_, what)| what)
        .collect()
}

/// The two enums `Mode` and `user` at `path`.
fn with_modes(path: &str, user: &str) -> Vec<String> {
    found(&[
        Source::new("checker/src/lib.rs", CHECKER_MODE),
        Source::new("src/matrix.rs", CORE_MODE),
        Source::new(path, user),
    ])
}

/// Test 4: a `_` arm on the checker's non-exhaustive `Mode`, which the CLI
/// takes by its `use`, passes, though the core defines an exhaustive enum
/// of that name.
#[test]
fn a_catch_all_on_another_packages_non_exhaustive_enum_passes() {
    let user = "use grobx_checker::Mode;\n\nfn f(m: Mode) -> u8 {\n    match m {\n        \
                Mode::Exact => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("cli/src/main.rs", user);
    assert!(found.is_empty(), "{found:#?}");
}

/// Test 5: a `_` arm on the core's exhaustive `Mode`, which a file of the
/// core takes by `use crate::`, is a finding, though the checker defines a
/// non-exhaustive enum of that name.
#[test]
fn a_catch_all_on_the_exhaustive_enum_of_that_name_fails() {
    let user = "use crate::matrix::Mode;\n\nfn g(m: Mode) -> u8 {\n    match m {\n        \
                Mode::Dense => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("src/f4.rs", user);
    assert_eq!(found, ["a catch-all arm on Mode"], "{found:#?}");
}

/// Test 6: with glob `use`s alone, the variants the arms name tell the two
/// `Mode`s apart.
#[test]
fn the_arms_variants_tell_two_enums_of_one_name_apart() {
    let open = "use grobx_checker::*;\n\nfn f(m: Mode) -> u8 {\n    match m {\n        \
                Mode::Exact => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("cli/src/main.rs", open);
    assert!(found.is_empty(), "{found:#?}");
    let closed = "use crate::matrix::*;\n\nfn g(m: Mode) -> u8 {\n    match m {\n        \
                  Mode::Dense => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("src/f4.rs", closed);
    assert_eq!(found, ["a catch-all arm on Mode"], "{found:#?}");
}

/// Test 7: two enums `Kind` of the same variants, the checker's
/// non-exhaustive and the core's exhaustive, and a match in the CLI that
/// nothing ties to either: the finding says the scan cannot tell.
#[test]
fn the_scan_says_where_it_cannot_tell() {
    let user = "use grobx::*;\nuse grobx_checker::*;\n\nfn f(k: Kind) -> u8 {\n    \
                match k {\n        Kind::A => 1,\n        _ => 2,\n    }\n}\n";
    let found = found(&[
        Source::new(
            "checker/src/lib.rs",
            "#[non_exhaustive]\npub enum Kind {\n    A,\n    B,\n}\n",
        ),
        Source::new("src/kind.rs", "pub enum Kind {\n    A,\n    B,\n}\n"),
        Source::new("cli/src/main.rs", user),
    ]);
    assert_eq!(found.len(), 1, "{found:#?}");
    assert!(found[0].contains("cannot tell"), "{found:#?}");
}

/// Test 8: a list of `use`s and a pattern's path through the crate name
/// the enum as a single `use` does.
#[test]
fn a_use_list_and_a_crate_path_name_the_enum() {
    let listed = "use grobx_checker::{Mode, Verdict};\n\nfn f(m: Mode) -> u8 {\n    \
                  match m {\n        Mode::Exact => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("cli/src/main.rs", listed);
    assert!(found.is_empty(), "{found:#?}");
    let pathed = "fn f(m: grobx_checker::Mode) -> u8 {\n    match m {\n        \
                  grobx_checker::Mode::Exact => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("cli/src/lib.rs", pathed);
    assert!(found.is_empty(), "{found:#?}");
    let local = "use crate::{matrix::Mode, order};\n\nfn g(m: Mode) -> u8 {\n    match m {\n        \
                 Mode::Dense => 1,\n        _ => 2,\n    }\n}\n";
    let found = with_modes("src/f4.rs", local);
    assert_eq!(found, ["a catch-all arm on Mode"], "{found:#?}");
}
