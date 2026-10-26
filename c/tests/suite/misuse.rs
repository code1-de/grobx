//! c-interface, the item's test code for what the C interface refuses and
//! for its small functions: `misuse.c`, and the driver's version.

use std::collections::HashMap;

use grobx::MonomialOrder;
use grobxc::*;

use crate::common::{Args, drive, program, run, scratch};
use crate::systems::dense;

/// `misuse.c`'s lines, each by its case's name: the rest of the line.
fn lines() -> HashMap<String, String> {
    let out = run(&program("misuse.c"), &[]);
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| {
            let (name, rest) = l.split_once(' ').unwrap_or((l, ""));
            (name.to_string(), rest.to_string())
        })
        .collect()
}

/// A case's status and the rest of its line.
fn status(lines: &HashMap<String, String>, case: &str) -> (i32, String) {
    let line = lines
        .get(case)
        .unwrap_or_else(|| panic!("no case {case}: {lines:?}"));
    let (status, rest) = line.split_once(' ').unwrap();
    (status.parse().unwrap(), rest.to_string())
}

/// NULL for `text` or `params`, an order out of range, a block order whose
/// `blocks` is NULL while `nblocks` is not 0, a text that is not UTF-8, a
/// checker mode out of range, and the cheap mode with 0 points are
/// `GROBX_ARGUMENT` with a message; a NULL result is `GROBX_ARGUMENT`
/// alone. The empty text, and a block order with no sizes or a size 0,
/// reach the crate, which refuses them.
#[test]
fn what_the_interface_refuses() {
    let lines = lines();
    for case in [
        "null-text",
        "null-params",
        "order-3",
        "null-blocks",
        "text-not-utf8",
        "checker-3",
        "cheap-0",
    ] {
        let (code, message) = status(&lines, case);
        assert_eq!(code, GROBX_ARGUMENT, "{case}: {message}");
        assert_ne!(message, "-", "{case}: a message");
    }
    assert_eq!(
        status(&lines, "null-result"),
        (GROBX_ARGUMENT, "-".to_string())
    );
    let e = grobx::groebner("", &Args::new(MonomialOrder::Grevlex, 1).params()).unwrap_err();
    assert_eq!(status(&lines, "empty-text"), (GROBX_SYNTAX, e.to_string()));
    for case in ["block-none", "block-0"] {
        let (code, message) = status(&lines, case);
        assert_eq!(code, GROBX_ORDER, "{case}: {message}");
    }
}

/// `grobx_digest` refuses NULL and a text that is not UTF-8, and writes
/// SHA-256 of `abc`.
#[test]
fn the_digest() {
    let lines = lines();
    for case in ["digest-null-text", "digest-null-out", "digest-not-utf8"] {
        assert_eq!(status(&lines, case).0, GROBX_ARGUMENT, "{case}");
    }
    assert_eq!(
        status(&lines, "digest-abc"),
        (
            GROBX_OK,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string()
        )
    );
}

/// NULL is never raised, and a new flag is raised once set.
#[test]
fn the_cancel_flag() {
    let lines = lines();
    assert_eq!(lines["cancel-null"], "0");
    assert_eq!(lines["cancel"], "0 1");
}

/// `grobx_params_new(GROBX_LEX, 7, 11)` keeps the order, the threads, and
/// the seed, and sets no blocks, every bound absent, no checker, and no
/// cofactors.
#[test]
fn params_new_fills_the_defaults() {
    let lines = lines();
    assert_eq!(
        lines["params"],
        format!("{GROBX_LEX} 1 0 7 11 {GROBX_NONE} 1 {GROBX_NONE} {GROBX_CHECKER_OFF} 0 0")
    );
}

/// `grobx_params_default()` is grevlex on all threads with no blocks, every
/// bound absent, no checker, and no cofactors, and two calls draw two seeds.
/// A collision of two 64-bit draws would fail this once in 2^64 runs.
#[test]
fn params_default_draws_the_seed() {
    let lines = lines();
    assert_eq!(
        lines["default"],
        format!("{GROBX_GREVLEX} 1 0 0 1 {GROBX_NONE} 1 {GROBX_NONE} {GROBX_CHECKER_OFF} 0 0")
    );
}

/// `grobx_version()` is the crate version.
#[test]
fn the_version_is_the_crates() {
    let (_, _, version) = drive(
        &scratch("version"),
        &dense(2, 3, 65521, 1),
        &Args::new(MonomialOrder::Grevlex, 1),
    );
    assert_eq!(version, grobx::VERSION);
}
