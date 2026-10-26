//! table-read-time, the item's test code: the edge cases of
//! `OBJECTIVES.md`, and the checker on dense systems in the signature run
//! and in the plain run (plan.md, _Test code_, 3 and 6). The test profile
//! checks every read the item takes unchecked, so each test runs them
//! checked.

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

/// Test 3: the edge cases over 65521 in grevlex, signatures on and off,
/// with cofactors, each basis canonical, accepted in the exact mode, and as
/// written; over 65521 in lex and `block:1`, accepted; and over ℚ in
/// grevlex, accepted.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut params = with_cofactors(0x7a1c);
            params.signatures = signatures;
            let outcome = run(&text, &params);
            let basis = outcome.basis.as_deref().unwrap();
            let name = format!("{case} over {P}, signatures {signatures}");
            assert_canonical(&name, &text, basis, "grevlex");
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(basis, want, "{name}");
        }
        for order in ["lex", "block:1"] {
            accepted_in(&format!("{case} over {P}"), &text, order, Mode::Exact);
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}

/// Test 6: the oracle. A dense cubic system over GF(65521) in the signature run
/// and another in the plain run, signatures off, with the checker in the
/// cheap mode at two points: certified.
#[test]
fn the_checker_certifies_them() {
    for (seed, signatures) in [(3, true), (4, false)] {
        let mut p = params(0x7a1f);
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
