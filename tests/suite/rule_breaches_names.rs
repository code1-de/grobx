//! rule-breaches, sub-item 2: the magic numbers of the item's inventory
//! stand nowhere as literals, each piece of logic it names has one
//! definition, and a `Stop` names itself (`iterations/0089-rule-breaches/plan.md`,
//! _Test code_, tests 4 to 6).

use grobx::Stop;

use crate::common::sources::{Source, lines_where, literals, rust_sources};

/// Each file of `instrument/inventory-names.md`, _1_, with the literals it
/// held as its code wrote them, which constants replace (rule-breaches.10).
const LITERALS: [(&str, &[&str]); 17] = [
    ("src/field.rs", &["< 1 << 16", ">= 1 << 16"]),
    (
        "src/text.rs",
        &[
            "[0u8; 19]",
            "[0u8; 20]",
            "(0..9)",
            "spare(out, 20",
            "= 10_000_000_000_000_000_000",
            "22 + monomial_longest",
            "name.len() + 12",
            "bits * 30103",
        ],
    ),
    ("src/rational.rs", &["2 + digits_bound"]),
    ("src/monomial.rs", &["96 - bits", "128 - bits"]),
    (
        "src/matrix.rs",
        &[">> 32 ==", "p >= 46341", "< 1 << 31", "< 2 * s"],
    ),
    ("src/cofactors.rs", &[">> 32 =="]),
    ("src/table.rs", &["2 * self.len()", ">> 32", "<< 32"]),
    ("src/f4.rs", &["4 * zeros", "2 * zeros", "10 * read()"]),
    ("src/lift.rs", &["misfits == 2"]),
    (
        "src/outgrown_round_tests.rs",
        &["div_ceil(4)", "6 * least / 10", "(8 * made).div_ceil(10)"],
    ),
    ("src/cyclic_gap_tests.rs", &["46341"]),
    ("src/zero_rows_tests.rs", &["2 * s"]),
    ("checker/src/field.rs", &["chunks(19)"]),
    ("checker/src/rational.rs", &["div_ceil(19)"]),
    (
        "checker/src/cheap.rs",
        &["div_ceil(19)", "(1 << 31) +", "=> 1 << 31", "k / 4"],
    ),
    ("checker/src/parse.rs", &["min(1 << 33)", "min(1 << 65)"]),
    (
        "checker/src/divisors.rs",
        &["vec![0; 1024]", "wrapping_mul(0x9e37"],
    ),
];

/// Test 4: no literal of `LITERALS` stands in its file, comments left out.
#[test]
fn the_magic_numbers_are_named() {
    let found: Vec<String> = LITERALS
        .iter()
        .flat_map(|&(file, needles)| literals(file, needles))
        .collect();
    assert!(found.is_empty(), "{found:#?}");
}

/// The production code of the directories `dirs`.
fn production_files(dirs: &[&str]) -> Vec<Source> {
    rust_sources(dirs)
        .into_iter()
        .filter(|source| !source.is_test())
        .collect()
}

/// Whether `line` takes a word product modulo a prime by its u128
/// remainder: `u128::from(a) * u128::from(b) % u128::from(p)`.
fn word_product(line: &str) -> bool {
    line.split_once(") * u128::from(")
        .is_some_and(|(before, after)| {
            before.contains("u128::from(")
                && after.split_once(')').is_some_and(|(factor, rest)| {
                    !factor.contains('(') && rest.starts_with(" % u128::from(")
                })
        })
}

/// Test 5: each piece of logic of `instrument/inventory-names.md`, _2_,
/// stands once in production code, as a scan can tell it: the word product
/// and the inverse modulo a prime, an integer's limbs, the constant exits
/// of the runs, the cofactors' sum, a stop's names, and the checker's
/// power (rule-breaches.11).
#[test]
fn each_piece_of_logic_stands_once() {
    let engine = production_files(&["src", "cli/src", "c/src", "python/src"]);
    let found = lines_where(&engine, word_product);
    assert_eq!(found.len(), 1, "the word product: {found:#?}");
    for needle in [
        "(0i128, 1i128)",
        "bits().div_ceil(64)",
        "inputs.swap_remove(k)",
        "m.monomials[r.cols[0].index()]) == 0",
        "summed.push(",
        "\"deadline\"",
        "\"cancel\"",
        "\"memory\"",
    ] {
        let found = lines_where(&engine, |line| line.contains(needle));
        assert_eq!(found.len(), 1, "{needle}: {found:#?}");
    }
    let checker = production_files(&["checker/src"]);
    let found = lines_where(&checker, |line| line.contains("fn pow("));
    assert_eq!(found.len(), 1, "{found:#?}");
}

/// Test 6: a `Stop` names itself as the CLI, the Python module, and the
/// call report name it.
#[test]
fn a_stop_names_itself() {
    assert_eq!(Stop::Deadline.name(), "deadline");
    assert_eq!(Stop::Cancel.name(), "cancel");
    assert_eq!(Stop::Memory.name(), "memory");
}
