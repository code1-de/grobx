//! hygiene, the scans of `tests/common/sources.rs` that the item adds, on
//! sources written out here: the skip-note scan reads a file's module doc
//! comment, and the copy scan reads a helper by its tokens
//! (`iterations/0147-hygiene/plan.md`, _Test code_, 2 and 9). These tests
//! check the test code, not the subject program, and run from step 5 on.

use crate::common::sources::{HelperKind, Source, copies, stale_skip_notes};

/// Test 2: the skip-note scan reads the module doc comment, across its line
/// breaks and whatever its case, and a file that holds the mark passes.
#[test]
fn the_skip_note_scan_reads_the_module_doc_comment() {
    let sources = [
        Source::new(
            "src/a_tests.rs",
            "//! a, its test code. Skipped until sub-item 2.\n\nfn f() {}\n",
        ),
        Source::new(
            "src/b_tests.rs",
            "//! b, its test code. The file is skipped\n//! until the item's sub-item starts.\n",
        ),
        Source::new(
            "src/c_tests.rs",
            "//! c, its test code. SKIPPED UNTIL sub-item 3.\n",
        ),
        Source::new(
            "src/d_tests.rs",
            "#![cfg(any())]\n//! d, its test code. The file is skipped until sub-item 1 starts.\n",
        ),
        Source::new(
            "src/e_tests.rs",
            "//! e, its test code.\n\n/// A prime is skipped until the next.\nfn g() {}\n",
        ),
    ];
    let stale = stale_skip_notes(&sources);
    assert_eq!(
        stale,
        ["src/a_tests.rs:1", "src/b_tests.rs:1", "src/c_tests.rs:1"],
        "{stale:#?}"
    );
}

/// The groups of copies of every kind in `a` at `src/a_tests.rs` and `b` at
/// `src/b_tests.rs`.
fn copies_of(a: &str, b: &str) -> Vec<String> {
    copies(
        &[
            Source::new("src/a_tests.rs", a),
            Source::new("src/b_tests.rs", b),
        ],
        &HelperKind::ALL,
    )
}

/// Test 9: the copy scan finds copies by their tokens, whatever their
/// comments, attributes, visibility, or a `fn`'s number literals, and
/// nothing else.
#[test]
fn the_copy_scan_reads_tokens() {
    let found = copies_of(
        "const P: u64 = 7;\n",
        "/// A prime.\n#[allow(dead_code, reason = \"some tests\")]\npub(crate) const P: u64 = 7; // a prime\n",
    );
    assert_eq!(
        found,
        ["const P: src/a_tests.rs:1, src/b_tests.rs:3"],
        "{found:#?}"
    );
    let found = copies_of(
        "fn f() -> u64 {\n    7\n}\n",
        "fn f() -> u64 {\n    11\n}\n",
    );
    assert_eq!(
        found,
        ["fn f: src/a_tests.rs:1, src/b_tests.rs:1"],
        "{found:#?}"
    );
    for (a, b) in [
        ("const P: u64 = 7;\n", "const P: u64 = 11;\n"),
        ("fn n() -> u64 {\n    7\n}\n", "const n: u64 = 7;\n"),
        ("fn f() -> u64 {\n    7\n}\n", "fn f() -> u32 {\n    7\n}\n"),
        ("const S: &str = \"x\";\n", "const S: &str = \"y\";\n"),
        ("const C: char = 'x';\n", "const C: char = 'y';\n"),
        (
            "#[test]\nfn t() {\n    f();\n}\n",
            "#[test]\nfn t() {\n    f();\n}\n",
        ),
    ] {
        let found = copies_of(a, b);
        assert!(found.is_empty(), "{a:?} against {b:?}: {found:#?}");
    }
    let apart: Vec<Source> = [
        ("src/a_tests.rs", "const P: u64 = 7;\n"),
        ("src/a.rs", "const P: u64 = 7;\n"),
        ("tests/suite/x.rs", "const P: u64 = 7;\n"),
    ]
    .iter()
    .map(|&(path, text)| Source::new(path, text))
    .collect();
    let found = copies(&apart, &[HelperKind::Const]);
    assert!(found.is_empty(), "{found:#?}");
}
