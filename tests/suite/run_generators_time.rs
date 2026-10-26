//! run-generators-time, the item's integration tests: calls whose runs
//! read the generators once and whose replays take their echelon from the
//! record return what calls with cofactors return, which take the
//! generators as read and which the exact checker accepts; and the errors
//! of a text at their lines
//! (`iterations/0087-run-generators-time/plan.md`, _Test code_, 7 to 12).

use grobx::Error;
use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, params, run, with_cofactors};
use crate::input_redundancy::{redundant, text};
use crate::q::accepted_q;

/// The edge cases of `OBJECTIVES.md` on x, y, z: name and generators.
pub(crate) const EDGE: [(&str, &str); 7] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// Over GF(p): the call without cofactors returns the basis text of the
/// call with them, which the exact checker accepts.
fn same_over_gfp(case: &str, instance: &str) {
    let with = run(instance, &with_cofactors(1));
    assert_accepted(case, instance, &with, Mode::Exact);
    let without = run(instance, &params(1));
    assert_eq!(without.basis, with.basis, "{case}\n{instance}");
}

/// Over ℚ: the call without cofactors returns the basis text of the call
/// with them, which `accepted_q` checks.
fn same_over_q(case: &str, instance: &str) {
    let with = accepted_q(case, instance, "grevlex");
    let without = run(instance, &params(1));
    assert_eq!(without.basis, with.basis, "{case}\n{instance}");
}

/// Test 7: the edge cases over GF(65521) and over ℚ.
#[test]
fn the_edge_cases() {
    for (case, gens) in EDGE {
        same_over_gfp(case, &format!("x,y,z\n{P}\n{gens}\n"));
        same_over_q(case, &format!("x,y,z\n0\n{gens}\n"));
    }
}

/// Test 8, run-generators-time.1, .5, and .6: systems with linear
/// redundancy over ℚ, whose lifts replay their records at several pairs of
/// primes and whose records leave zero rows.
#[test]
fn redundant_systems_over_q() {
    for seed in 1..=6 {
        let (names, gens) = redundant(0, seed);
        same_over_q(&format!("seed {seed}"), &text(0, &names, &gens));
    }
}

/// Test 9, run-generators-time.1 and .2: coefficients written as fractions,
/// products of literals, and integers of 20 to 100 digits, over ℚ, two of
/// the generators dependent.
#[test]
fn literals_and_fractions_over_q() {
    let big = "1234567890".repeat(10);
    let instance = format!(
        "x,y,z\n0\n\
         2/3*x+5*y*7-123456789012345678901234567890*z,\n\
         x*y-11/13*z^2+1,\n\
         y^2-z*99999999999999999999+x*{big},\n\
         x+18446744073709551616*y,\n\
         2*x+36893488147419103232*y,\n\
         3 * x^2 * 4/5 - y\n"
    );
    same_over_q("literals and fractions", &instance);
}

/// The error of a call that must fail.
fn error(instance: &str) -> Error {
    match grobx::groebner(instance, &params(1)) {
        Ok(outcome) => panic!("no error but {outcome:?}\n{instance}"),
        Err(e) => e,
    }
}

/// Test 10, run-generators-time.3: errors at their lines, the first in
/// the text's order.
#[test]
fn errors_at_their_lines() {
    // Over GF(65521) a denominator 65521 at line 5 before a syntax error
    // at line 7.
    let instance = format!("x,y\n{P}\nx+y,\nx-y,\n1/65521*x,\ny,\nx+*y\n");
    assert_eq!(error(&instance), Error::NoInverse { line: 5 });
    // The syntax error at line 4 before the denominator at line 6.
    let instance = format!("x,y\n{P}\nx+y,\nx+*y,\ny,\n1/65521*x\n");
    assert!(matches!(error(&instance), Error::Syntax { line: 4, .. }));
    // An unknown variable before a denominator 65521 in one term.
    let instance = format!("x,y\n{P}\nx+y,\nq*1/65521*x\n");
    assert!(matches!(
        error(&instance),
        Error::UnknownVariable { line: 4, .. }
    ));
    // Over ℚ a denominator 0 at line 4 before a syntax error at line 5.
    let instance = "x,y\n0\nx+y,\n1/0*x,\nx+*y\n";
    assert_eq!(error(instance), Error::NoInverse { line: 4 });
    // Over ℚ the syntax error at line 4 before the denominator 0 at line 5.
    let instance = "x,y\n0\nx+y,\nx+*y,\n1/0*x\n";
    assert!(matches!(error(instance), Error::Syntax { line: 4, .. }));
}

/// Test 11, run-generators-time.1: systems with linear redundancy over
/// GF(65521) and GF(2^31 − 1), whose signature runs hand over to the plain
/// run, which reads the generators again, on some of them.
#[test]
fn dependent_generators_over_gfp() {
    for p in [P, P31] {
        for seed in 1..=6 {
            let (names, gens) = redundant(p, seed);
            same_over_gfp(&format!("p {p}, seed {seed}"), &text(p, &names, &gens));
        }
    }
}

/// Test 12: test 8's systems under seeds 1 to 4 give one basis text each.
#[test]
fn the_seeds_agree() {
    for seed in 1..=3 {
        let (names, gens) = redundant(0, seed);
        let instance = text(0, &names, &gens);
        let first = run(&instance, &params(1)).basis;
        for s in 2..=4 {
            assert_eq!(
                run(&instance, &params(s)).basis,
                first,
                "system {seed} under seed {s}"
            );
        }
    }
}
