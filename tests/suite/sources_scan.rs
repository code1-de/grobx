//! rule-breaches, the scans of `tests/common/sources.rs` on sources
//! written out here: each finds what its rule names and nothing else, on
//! the edge cases of `iterations/0089-rule-breaches/plan.md`, _Edge cases_.
//! These tests check the test code, not the subject program, and run from
//! step 5 on (plan.md, _Test code_, tests 1 and 7).

use std::collections::BTreeSet;

use crate::common::sources::{
    Enums, Source, allows_in, lex, misplaced_uses_in, positional, positional_formats_in,
    production, production_asserts_in, variant_choices_in, wildcard_arms_in,
};

/// The lines of a scan's finds.
fn lines(found: &[(usize, String)]) -> Vec<usize> {
    found.iter().map(|&(line, _)| line).collect()
}

/// The external crates of the written-out sources.
fn external() -> BTreeSet<String> {
    ["grobx", "num_bigint", "sha2"]
        .iter()
        .map(|c| (*c).to_string())
        .collect()
}

/// The enums of a written-out source, with std's and num-bigint's.
fn enums(source: &Source) -> Enums {
    Enums::of(std::slice::from_ref(source))
}

/// Test 1a: comments and the insides of literals are blanked in the code,
/// their line breaks kept, and a lifetime is code.
#[test]
fn the_lexer_blanks_comments_and_literals() {
    let text = concat!(
        "let a = \"x{}\\\"y\"; // c{}\n",
        "/* outer /* inner */ still */ let b = r#\"q\"{}\"#;\n",
        "let c = b\"z\"; let d = 'ℚ'; let e = '\\''; fn f<'a>(x: &'a u8) {}\n",
    );
    let lexed = lex(text);
    assert_eq!(lexed.code.len(), text.len());
    assert_eq!(lexed.code.lines().count(), 3);
    let first = lexed.code.lines().next().expect("a first line");
    assert_eq!(first.trim_end(), "let a = \"      \";");
    let second = lexed.code.lines().nth(1).expect("a second line");
    assert!(second.trim_start().starts_with("let b = r#\""), "{second}");
    assert!(
        !second.contains("inner") && !second.contains("{}"),
        "{second}"
    );
    let third = lexed.code.lines().nth(2).expect("a third line");
    assert!(third.contains("fn f<'a>(x: &'a u8)"), "{third}");
    assert!(!third.contains('ℚ') && !third.contains('z'), "{third}");
    let texts: Vec<&str> = lexed.literals.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(texts, ["x{}\\\"y", "q\"{}", "z"]);
    assert!(lexed.uncommented.contains("x{}") && !lexed.uncommented.contains("c{}"));
}

/// Test 1b: production code leaves out `#[cfg(test)]` items: a module, a
/// function, an enum's variant, and a match arm; `#[cfg(not(test))]` stays.
#[test]
fn production_code_leaves_out_test_items() {
    let source = Source::new(
        "src/x.rs",
        concat!(
            "enum E {\n",
            "    A,\n",
            "    #[cfg(test)]\n",
            "    Alone,\n",
            "}\n",
            "fn g(e: E) -> u8 {\n",
            "    match e {\n",
            "        E::A => 1,\n",
            "        #[cfg(test)]\n",
            "        E::Alone => tested(),\n",
            "    }\n",
            "}\n",
            "#[cfg(test)]\n",
            "fn tested() -> u8 { 2 }\n",
            "#[cfg(not(test))]\n",
            "fn kept() {}\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    fn inner() {}\n",
            "}\n",
        ),
    );
    let code = production(&source).code;
    assert!(code.contains("A,") && code.contains("E::A => 1,"), "{code}");
    assert!(
        !code.contains("Alone") && !code.contains("tested"),
        "{code}"
    );
    assert!(
        code.contains("fn kept()") && !code.contains("inner"),
        "{code}"
    );
    assert_eq!(code.lines().count(), 20);
}

/// Test 1c: the choices between variants by `==`, `!=`, `matches!` outside
/// an assertion, and a `let` that chooses; a value compared with
/// `Some(&0)`, a constant, the assertions, and a `matches!` on bytes stay
/// (rule-breaches.3 to .5).
#[test]
fn choices_between_variants() {
    let source = Source::new(
        "src/x.rs",
        concat!(
            "pub enum Engine { Replay, Signature, Plain }\n",
            "fn f(e: Engine, o: Option<bool>, v: &[u8], c: Ordering) {\n",
            "    let a = e == Engine::Signature && v.len() > 1;\n",
            "    if Engine::Plain != e {}\n",
            "    let b = o == Some(true);\n",
            "    if c != Ordering::Greater {}\n",
            "    let d = v.last() == Some(&0) || v.len() == usize::MAX;\n",
            "    debug_assert!(matches!(e, Engine::Replay), \"{e:?}\");\n",
            "    let g = matches!(e, Engine::Replay);\n",
            "    if let Engine::Plain = e {}\n",
            "    let Engine::Plain = e else { return };\n",
            "    if let Some(x) = o {}\n",
            "    if a && let Engine::Replay = e {}\n",
            "    let h = o.is_none() || None == o;\n",
            "    let k = matches!(v[0], b'+' | b'-');\n",
            "}\n",
        ),
    );
    let found = variant_choices_in(&source, &enums(&source));
    assert_eq!(lines(&found), [3, 4, 5, 6, 9, 10, 11, 13, 14], "{found:?}");
}

/// Test 1d: a `_` arm or a bare binding on an enum's variants, and a `_`
/// in a tuple opposite a variant; matches on bytes, numbers, and another
/// crate's `#[non_exhaustive]` enum stay, and a guarded `_` before arms that
/// name every variant.
#[test]
fn catch_all_arms() {
    let other = Source::new(
        "src/lib.rs",
        "#[non_exhaustive]\npub enum Error { Syntax, Size }\npub enum Stop { Deadline, Cancel }\n",
    );
    let source = Source::new(
        "tests/suite/x.rs",
        concat!(
            "fn f(x: Option<u8>, s: Stop, e: grobx::Error, n: u32, p: (Stop, Option<u8>)) {\n",
            "    match x { Some(b',') => 1, Some(b'\\n') => 2, _ => 3 };\n",
            "    match x { Some(v) if v > 2 => v, _ => 0 };\n",
            "    match s { Stop::Deadline => 1, other => 2 };\n",
            "    match n { 0 => 1, m => m };\n",
            "    match e { Error::Syntax => 1, _ => 2 };\n",
            "    match p { (Stop::Deadline, _) => 1, (Stop::Cancel, Some(_)) => 2, (Stop::Cancel, None) => 3 };\n",
            "    match x { Some(v) => v, None => 0 };\n",
            "    match x {\n",
            "        Some(v) => {\n",
            "            match v { 1 => 0, _ => 1 }\n",
            "        }\n",
            "        got => panic!(\"{got:?}\"),\n",
            "    };\n",
            "    match s { _ if n > 0 => 0, Stop::Deadline => 1, Stop::Cancel => 2 };\n",
            "}\n",
        ),
    );
    let sources = [other, source];
    let enums = Enums::of(&sources);
    let found = wildcard_arms_in(&sources[1], &enums);
    assert_eq!(lines(&found), [2, 3, 4, 7, 13], "{found:?}");
    let own = wildcard_arms_in(
        &Source::new(
            "src/y.rs",
            "fn g(e: Error) { match e { Error::Syntax => 1, _ => 2 }; }\n",
        ),
        &enums,
    );
    assert_eq!(
        lines(&own),
        [1],
        "the crate's own non_exhaustive enum: {own:?}"
    );
}

/// Test 1e: the panicking macros of production code; `debug_assert!`, a
/// `const` item, a `#[cfg(test)]` item, a comment, and test code stay.
#[test]
fn panics_in_production_code() {
    let text = concat!(
        "const _: () = assert!(1 < 2);\n",
        "const V: u8 = match 1 { 1 => 1, _ => panic!(\"no\") };\n",
        "pub fn f(x: u8) {\n",
        "    debug_assert!(x > 0);\n",
        "    assert!(x < 9, \"{x}\");\n",
        "    // assert!(x == 3);\n",
        "    assert_eq!(x, 4);\n",
        "    if x == 5 { panic!(\"five\") }\n",
        "}\n",
        "#[cfg(test)]\n",
        "fn t() { assert!(true); }\n",
        "const fn g() {}\n",
    );
    let found = production_asserts_in(&Source::new("src/x.rs", text));
    assert_eq!(
        found,
        [
            (5, "assert!".to_string()),
            (7, "assert_eq!".to_string()),
            (8, "panic!".to_string())
        ]
    );
    for path in ["src/x_tests.rs", "tests/suite/x.rs"] {
        let found = production_asserts_in(&Source::new(path, text));
        assert!(found.is_empty(), "{path}: {found:?}");
    }
}

/// Test 7a: positional placeholders, and the named and escaped ones.
#[test]
fn positional_placeholders() {
    for text in [
        "{}",
        "a {} b",
        "{:?}",
        "{:>5}",
        "{0}",
        "{x} and {}",
        "{1:?}",
    ] {
        assert!(positional(text), "{text}");
    }
    for text in ["", "{{}}", "{x}", "{x:?}", "{x:>5}", "{{x}} {y}", "}} {{"] {
        assert!(!positional(text), "{text}");
    }
}

/// Test 7b: the format strings with a positional placeholder, each at its
/// macro's place for it; a compared value, a string outside a formatting
/// macro, a comment, and a nested string stay.
#[test]
fn format_strings() {
    let text = concat!(
        "fn f(x: u8, v: &[u8], out: &mut String) {\n",
        "    let a = format!(\"{}\", x);\n",
        "    write!(out, \"{x} {:?}\", v).ok();\n",
        "    assert!(x > 0, \"{} past\", v.len());\n",
        "    assert_eq!(\n",
        "        x,\n",
        "        3,\n",
        "        \"{}\",\n",
        "        v.len()\n",
        "    );\n",
        "    assert_eq!(a, \"{}\");\n",
        "    let b = \"{}\";\n",
        "    // format!(\"{}\", x);\n",
        "    let c = format!(\"{x} {{}} {len}\", len = v.len());\n",
        "    let d = format!(\"{}\", format!(\"{x}\"));\n",
        "    let e = format!(r#\"{} \"q\"\"#, x);\n",
        "}\n",
    );
    let found = positional_formats_in(&Source::new("src/x.rs", text));
    assert_eq!(lines(&found), [2, 3, 4, 8, 15, 16], "{found:?}");
}

/// Test 7c: `use` declarations out of the order of rule-breaches.2, and a
/// file in that order, with a test module, a `#[cfg]` line, a `pub use`,
/// a `use` over several lines, and a macro's re-export after the macro.
#[test]
fn use_order() {
    let good = concat!(
        "//! A module.\n",
        "#![allow(dead_code, reason = \"r\")]\n",
        "\n",
        "mod a;\n",
        "\n",
        "use std::fmt;\n",
        "#[cfg(target_arch = \"x86_64\")]\n",
        "use std::arch::x86_64::{\n",
        "    __m128i, _mm_add_epi64,\n",
        "};\n",
        "\n",
        "use grobx::Params;\n",
        "use sha2::Digest;\n",
        "\n",
        "use crate::common::{self, run};\n",
        "pub use a::B;\n",
        "\n",
        "macro_rules! by_order { () => {} }\n",
        "pub(crate) use by_order;\n",
        "\n",
        "fn f() {}\n",
        "\n",
        "#[cfg(test)]\n",
        "mod tests {\n",
        "    use std::fmt;\n",
        "\n",
        "    use super::f;\n",
        "}\n",
    );
    let found = misplaced_uses_in(&Source::new("src/x.rs", good), &external());
    assert!(found.is_empty(), "{found:?}");
    let bad = concat!(
        "use crate::common;\n",
        "\n",
        "use std::fmt;\n",
        "use grobx::Params;\n",
        "use sha2::Digest;\n",
        "\n",
        "use num_bigint::BigInt;\n",
        "\n",
        "use std::io;\n",
        "\n",
        "fn f() {\n",
        "    use std::mem;\n",
        "}\n",
        "use std::ptr;\n",
        "#[cfg(target_arch = \"x86_64\")]\n",
        "pub fn g() {\n",
        "    #[cfg(target_arch = \"x86_64\")]\n",
        "    {\n",
        "        use std::arch::x86_64::__cpuid;\n",
        "    }\n",
        "}\n",
    );
    let found = misplaced_uses_in(&Source::new("tests/suite/x.rs", bad), &external());
    let whats: Vec<&str> = found.iter().map(|(_, what)| what.as_str()).collect();
    assert_eq!(lines(&found), [3, 4, 7, 9, 12, 14, 19], "{found:?}");
    assert_eq!(
        whats,
        [
            "group 0 after group 2",
            "no blank line before group 1",
            "a blank line inside group 1",
            "group 0 after group 1",
            "a use inside a body",
            "a use after an item",
            "a use inside a body"
        ]
    );
}

/// Test 7d: the lint suppressions, inner and outer, with and without a
/// reason, over one line or several; one inside `cfg_attr` and one in a
/// comment stay out.
#[test]
fn suppressions() {
    let text = concat!(
        "#![allow(dead_code)]\n",
        "#[allow(clippy::too_many_arguments)]\n",
        "fn f() {}\n",
        "#[allow(\n",
        "    clippy::cast_possible_truncation,\n",
        "    reason = \"it fits\"\n",
        ")]\n",
        "fn g() {}\n",
        "#[cfg_attr(test, allow(unused))]\n",
        "// #[allow(unused)]\n",
        "fn h() {}\n",
    );
    let found = allows_in(&Source::new("src/x.rs", text));
    let seen: Vec<(String, bool, Vec<String>, bool)> = found
        .into_iter()
        .map(|a| (a.place, a.inner, a.lints, a.reason))
        .collect();
    assert_eq!(
        seen,
        [
            (
                "src/x.rs:1".to_string(),
                true,
                vec!["dead_code".to_string()],
                false
            ),
            (
                "src/x.rs:2".to_string(),
                false,
                vec!["clippy::too_many_arguments".to_string()],
                false
            ),
            (
                "src/x.rs:4".to_string(),
                false,
                vec!["clippy::cast_possible_truncation".to_string()],
                true
            ),
        ]
    );
}
