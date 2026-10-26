//! signature-zero-time, the item's test code: the edge cases of
//! `OBJECTIVES.md`, other primes, cofactors, and a deadline
//! (`iterations/0049-signature-zero-time/plan.md`, _Test code_, 4, 6, 8, and 9). A
//! call that rules rows zero by random combinations must return the basis
//! of the same call whose trace listens, which takes no test.

use std::time::Duration;

use grobx::id::RowId;
use grobx::trace::{RowSource, Trace};
use grobx::{CheckerMode, Outcome, Params, Stop};

use crate::common::{P, dense, params, random_square, random_system, run, with_cofactors};

/// A trace that listens and keeps nothing: the run takes no test
/// (signature-zero-time.10).
struct Listening;

impl Trace for Listening {
    fn matrix(&mut self, _: &[RowSource]) {}
    fn counted(&mut self, _: &[RowId]) {}
}

/// The outcome of `text` under `params` with a trace that listens.
fn listened(text: &str, params: &Params) -> Outcome {
    let outcome = grobx::groebner_with(text, params, &mut Listening)
        .unwrap_or_else(|e| panic!("groebner_with failed: {e:?}\non\n{text}"));
    assert!(outcome.basis.is_some(), "the call stopped: {outcome:?}");
    outcome
}

/// The primes of the unit tests, 2, and 3: s from 1 to 16 where a test
/// runs.
const PRIMES: [u64; 13] = [
    2,
    3,
    5,
    7,
    251,
    1297,
    46337,
    46349,
    65521,
    2_147_483_647,
    2_147_483_659,
    2_305_843_009_213_693_951,
    18_446_744_073_709_551_557,
];

/// The basis of `text` with signatures, which must be that of the same
/// call with a trace that listens and that of the plain run.
fn same_basis(name: &str, text: &str, seed: u64) -> String {
    let p = params(seed);
    let on = run(text, &p).basis.unwrap();
    let heard = listened(text, &p).basis.unwrap();
    assert_eq!(on, heard, "{name}: with a trace that listens");
    let mut plain = params(seed);
    plain.signatures = false;
    let off = run(text, &plain).basis.unwrap();
    assert_eq!(on, off, "{name}: the plain run");
    on
}

/// The edge cases of `OBJECTIVES.md` on x, y, z, as text-time's test 4
/// takes them: name, generators, and the basis over GF(65521) in grevlex.
const EDGE: [(&str, &str, &str); 7] = [
    ("the zero ideal", "0", "[]:\n"),
    ("F = (0, 0)", "0,\n0", "[]:\n"),
    ("no generator", "", "[]:\n"),
    ("the unit ideal", "x*z-1,\nx", "[1]:\n"),
    ("already a Gröbner basis", "x+z,\nz^2+1", "[x+z,\nz^2+1]:\n"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
        "[y^2+65520*x,\nx*y+65520*z,\nx^2+65520*y*z]:\n",
    ),
    (
        "an eliminable variable",
        "x-y^2-z,\ny*z-1,\nz^2-y",
        "[x+65519*z,\nz^2+65520*y,\ny*z+65520,\ny^2+65520*z]:\n",
    ),
];

/// Test 4: the edge cases over every prime of `PRIMES`, signatures on,
/// and two texts that `Dense16` and `Dense32` read: each basis as with a
/// trace that listens and as the plain run's, and over 65521 as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in PRIMES {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            let basis = same_basis(&format!("{case} over {p}"), &text, 0x5e70);
            if p == P {
                assert_eq!(basis, want, "{case} over {p}");
            }
        }
    }
    for (format, generators) in [
        ("Dense16", "x^200-y*z,\ny^2-z,\nx*z-y"),
        ("Dense32", "x^40000-y,\ny^2-z"),
    ] {
        for p in PRIMES {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            same_basis(&format!("{format} over {p}"), &text, 0x5e76);
        }
    }
}

/// `text`, an instance's text, with its characteristic replaced by `p`.
fn at(text: &str, p: u64) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    let p = p.to_string();
    lines[1] = &p;
    lines.join("\n") + "\n"
}

/// Test 6: a dense cubic system's and a random square system's generators, and
/// random systems, read at every prime of `PRIMES`: each basis as with a
/// trace that listens and as the plain run's.
#[test]
fn other_primes() {
    for (name, text) in [
        ("cubic-3", dense(3, 3, P, 1)),
        ("square", random_square(P, 1)),
    ] {
        for p in PRIMES {
            same_basis(&format!("{name} at {p}"), &at(&text, p), 0x5e72);
        }
    }
    for p in PRIMES {
        for seed in 0..4 {
            let text = random_system(p, seed);
            same_basis(&format!("random system {seed} over {p}"), &text, seed);
        }
    }
}

/// Test 8: a dense cubic system over GF(65521) with cofactors and the exact
/// checker: certified, and the cofactor text of the same call with a trace
/// that listens.
#[test]
fn cofactors() {
    let stem = "cubic-3-p65521";
    let text = dense(3, 3, P, 1);
    let mut p = with_cofactors(0x5e74);
    p.checker = Some(CheckerMode::Exact);
    let outcome = run(&text, &p);
    assert_eq!(
        outcome.certified,
        Some(true),
        "{stem}: {flaw:?}",
        flaw = outcome.flaw
    );
    let heard = listened(&text, &p);
    assert_eq!(outcome.cofactors, heard.cofactors, "{stem}");
    assert_eq!(heard.certified, Some(true), "{stem}");
}

/// Test 9: a long dense cubic system over GF(65521) with a deadline of 30 ms
/// stops at a check of the budget.
#[test]
fn a_deadline() {
    let mut p = params(0x5e75);
    p.deadline = Some(Duration::from_millis(30));
    let outcome = grobx::groebner(&dense(8, 3, P, 1), &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Deadline));
    assert!(outcome.basis.is_none());
}
