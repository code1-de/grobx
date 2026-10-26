//! c-interface, the item's test code for the header against the library:
//! the sizes, offsets, and constants `abi.c` prints from `grobx.h` are the
//! Rust types' and constants' of the same names (c-interface.14).

use std::collections::HashMap;
use std::mem::{offset_of, size_of};

use grobxc::*;

use crate::common::{program, run};

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "a table, one row per field and constant of the header"
)]
fn the_header_matches_the_library() {
    let out = run(&program("abi.c"), &[]);
    let header: HashMap<String, String> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| {
            let (name, value) = l.rsplit_once(' ').unwrap();
            (name.to_string(), value.to_string())
        })
        .collect();
    let library: Vec<(&str, String)> = vec![
        ("sizeof grobx_params", size_of::<grobx_params>().to_string()),
        (
            "grobx_params.order",
            offset_of!(grobx_params, order).to_string(),
        ),
        (
            "grobx_params.blocks",
            offset_of!(grobx_params, blocks).to_string(),
        ),
        (
            "grobx_params.nblocks",
            offset_of!(grobx_params, nblocks).to_string(),
        ),
        (
            "grobx_params.threads",
            offset_of!(grobx_params, threads).to_string(),
        ),
        (
            "grobx_params.seed",
            offset_of!(grobx_params, seed).to_string(),
        ),
        (
            "grobx_params.deadline",
            offset_of!(grobx_params, deadline).to_string(),
        ),
        (
            "grobx_params.cancel",
            offset_of!(grobx_params, cancel).to_string(),
        ),
        (
            "grobx_params.memory",
            offset_of!(grobx_params, memory).to_string(),
        ),
        (
            "grobx_params.checker",
            offset_of!(grobx_params, checker).to_string(),
        ),
        (
            "grobx_params.points",
            offset_of!(grobx_params, points).to_string(),
        ),
        (
            "grobx_params.cofactors",
            offset_of!(grobx_params, cofactors).to_string(),
        ),
        ("sizeof grobx_result", size_of::<grobx_result>().to_string()),
        (
            "grobx_result.basis",
            offset_of!(grobx_result, basis).to_string(),
        ),
        (
            "grobx_result.cofactors",
            offset_of!(grobx_result, cofactors).to_string(),
        ),
        (
            "grobx_result.stopped",
            offset_of!(grobx_result, stopped).to_string(),
        ),
        (
            "grobx_result.certified",
            offset_of!(grobx_result, certified).to_string(),
        ),
        (
            "grobx_result.flaw",
            offset_of!(grobx_result, flaw).to_string(),
        ),
        (
            "grobx_result.error",
            offset_of!(grobx_result, error).to_string(),
        ),
        ("GROBX_OK", GROBX_OK.to_string()),
        ("GROBX_SYNTAX", GROBX_SYNTAX.to_string()),
        ("GROBX_UNKNOWN_VARIABLE", GROBX_UNKNOWN_VARIABLE.to_string()),
        (
            "GROBX_REPEATED_VARIABLE",
            GROBX_REPEATED_VARIABLE.to_string(),
        ),
        ("GROBX_CHARACTERISTIC", GROBX_CHARACTERISTIC.to_string()),
        ("GROBX_OVERFLOW", GROBX_OVERFLOW.to_string()),
        ("GROBX_NO_INVERSE", GROBX_NO_INVERSE.to_string()),
        ("GROBX_ORDER", GROBX_ORDER.to_string()),
        ("GROBX_MEMORY", GROBX_MEMORY.to_string()),
        ("GROBX_ERROR", GROBX_ERROR.to_string()),
        ("GROBX_ARGUMENT", GROBX_ARGUMENT.to_string()),
        ("GROBX_PANIC", GROBX_PANIC.to_string()),
        ("GROBX_SIZE", GROBX_SIZE.to_string()),
        ("GROBX_DROPS", GROBX_DROPS.to_string()),
        ("GROBX_NONE", GROBX_NONE.to_string()),
        ("GROBX_GREVLEX", GROBX_GREVLEX.to_string()),
        ("GROBX_LEX", GROBX_LEX.to_string()),
        ("GROBX_BLOCK", GROBX_BLOCK.to_string()),
        ("GROBX_CHECKER_OFF", GROBX_CHECKER_OFF.to_string()),
        ("GROBX_CHECKER_EXACT", GROBX_CHECKER_EXACT.to_string()),
        ("GROBX_CHECKER_CHEAP", GROBX_CHECKER_CHEAP.to_string()),
        ("GROBX_STOP_NONE", GROBX_STOP_NONE.to_string()),
        ("GROBX_STOP_DEADLINE", GROBX_STOP_DEADLINE.to_string()),
        ("GROBX_STOP_CANCEL", GROBX_STOP_CANCEL.to_string()),
        ("GROBX_STOP_MEMORY", GROBX_STOP_MEMORY.to_string()),
        ("GROBX_UNCHECKED", GROBX_UNCHECKED.to_string()),
        ("GROBX_REJECTED", GROBX_REJECTED.to_string()),
        ("GROBX_CERTIFIED", GROBX_CERTIFIED.to_string()),
    ];
    assert_eq!(header.len(), library.len(), "{header:?}");
    for (name, value) in library {
        assert_eq!(header.get(name), Some(&value), "{name}");
    }
}

/// `linkage.cpp` includes `grobx.h` in C++, links against the library
/// by the functions' C names, and prints the crate version.
#[test]
fn a_cpp_program_links() {
    let out = run(&program("linkage.cpp"), &[]);
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("{version}\n", version = grobx::VERSION)
    );
}
