//! signature-pair-time, the item's test code: the edge cases keep their
//! bases, and a call with cofactors is certified (plan.md, _Test code_).

use std::time::Duration;

use grobx::{CheckerMode, Params};

use crate::common::{P, dense, params, run};

const SEED: u64 = 0x5197;

/// A measured call's parameters with a deadline of 120 s.
fn bounded(seed: u64) -> Params {
    let mut p = params(seed);
    p.deadline = Some(Duration::from_secs(120));
    p
}

/// `signatures.rs`'s edge cases whose basis text it writes.
const EDGE: &[(&str, &str, &str)] = &[
    ("zero", "x,y\n65521\n0\n", "[]:\n"),
    ("zeros", "x,y\n7\n0,\n0\n", "[]:\n"),
    ("none", "x,y\n65521\n", "[]:\n"),
    ("x-plus-x", "x\n2\nx+x\n", "[]:\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n", "[1]:\n"),
    ("unit-p2", "x\n2\nx,\nx+1\n", "[1]:\n"),
    ("unit-constant", "x,y\n65521\n5\n", "[1]:\n"),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
        "[1]:\n",
    ),
    ("gb", "x,y\n65521\nx+y,\ny^2+1\n", "[x+y,\ny^2+1]:\n"),
    (
        "gb-swapped",
        "x,y\n65521\ny^2+1,\nx+y\n",
        "[x+y,\ny^2+1]:\n",
    ),
    (
        "tiny",
        "x,y\n65521\nx^2+y,\nx*y+1\n",
        "[y^2+65520*x,\nx*y+1,\nx^2+y]:\n",
    ),
    (
        "duplicates",
        "x,y\n65521\nx^2+y,\nx^2+y,\nx*y+1\n",
        "[y^2+65520*x,\nx*y+1,\nx^2+y]:\n",
    ),
    (
        "multiples",
        "x,y\n65521\n2*x^2+2*y,\nx*y+1,\n3*x*y+3\n",
        "[y^2+65520*x,\nx*y+1,\nx^2+y]:\n",
    ),
    ("gcd", "x\n65521\nx^3-1,\nx^2-1\n", "[x+65520]:\n"),
    ("gcd-p7", "x\n7\nx^4-1,\nx^6-1\n", "[x^2+6]:\n"),
    ("monic-power", "x\n65521\nx^5+x+1\n", "[x^5+x+1]:\n"),
];

/// `signatures.rs`'s edge cases whose basis the run without signatures
/// gives: the zero polynomial among the generators, and eliminable
/// variables.
const SAME: &[(&str, &str)] = &[
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    (
        "eliminable-first",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
    ),
    ("eliminable-last", "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
    (
        "eliminable-linear",
        "x,y,z\n65521\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n",
    ),
];

#[test]
fn the_edge_cases_keep_their_bases() {
    for (name, text, want) in EDGE {
        let basis = run(text, &bounded(SEED)).basis.unwrap();
        assert_eq!(basis, *want, "{name}");
    }
    for (name, text) in SAME {
        let on = run(text, &bounded(SEED)).basis.unwrap();
        let mut plain = bounded(SEED);
        plain.signatures = false;
        let off = run(text, &plain).basis.unwrap();
        assert_eq!(on, off, "{name}");
    }
}

#[test]
fn a_call_with_cofactors_is_certified() {
    let mut p = bounded(SEED);
    p.cofactors = true;
    p.checker = Some(CheckerMode::Exact);
    let outcome = run(&dense(3, 3, P, 1), &p);
    assert_eq!(
        outcome.certified,
        Some(true),
        "cubic-3: {flaw:?}",
        flaw = outcome.flaw
    );
}
