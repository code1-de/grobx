//! divisibility-time, the item's test code: the edge cases of
//! `OBJECTIVES.md`, and the checker on dense systems in the plain run and
//! in the signature run (plan.md, _Test code_, 7 and 10).

use std::num::NonZeroU32;

use grobx::CheckerMode;
use grobx_checker::Mode;

use crate::common::{P, assert_accepted, dense, params, run, with_cofactors};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// The edge cases of `OBJECTIVES.md` on x, y, z: name, generators, and the
/// basis over GF(65521) in grevlex.
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

/// Test 7: the edge cases over 65521 in grevlex, signatures on and off,
/// over 65521 in lex, and over ℚ in grevlex: each basis canonical and
/// accepted in the exact mode, and over 65521 in grevlex as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut params = with_cofactors(0xd1f2);
            params.signatures = signatures;
            let outcome = run(&text, &params);
            let basis = outcome.basis.as_deref().unwrap();
            let name = format!("{case} over {P}, signatures {signatures}");
            assert_canonical(&name, &text, basis, "grevlex");
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(basis, want, "{name}");
        }
        accepted_in(&format!("{case} over {P}"), &text, "lex", Mode::Exact);
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}

/// Test 10: the oracle. A dense cubic system over GF(65521) in the plain run,
/// signatures off, and another in the signature run, with the checker in
/// the cheap mode at two points: certified.
#[test]
fn the_checker_certifies_them() {
    for (seed, signatures) in [(1, false), (2, true)] {
        let mut p = params(0xd1f5);
        p.signatures = signatures;
        p.checker = Some(CheckerMode::Cheap {
            points: NonZeroU32::new(2).unwrap(),
        });
        let outcome = run(&dense(4, 3, P, seed), &p);
        assert_eq!(
            outcome.certified,
            Some(true),
            "seed {seed}: {flaw:?}",
            flaw = outcome.flaw
        );
    }
}
