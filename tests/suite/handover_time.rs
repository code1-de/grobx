//! handover-time, the item's test code: the edge cases of `OBJECTIVES.md`
//! over GF(p) and ℚ, signatures on and off, with
//! cofactors and the checker (plan.md, _Test code_, 4).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, params, run, with_cofactors};
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

/// Test 4: the edge cases over 65521 and 2^31 − 1 in grevlex, signatures
/// on and off, with cofactors and the checker in the exact mode, in lex
/// and `block:1`, and over ℚ in grevlex: each basis canonical and
/// accepted, and over 65521 in grevlex as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in [P, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut plain = params(0x0b30);
                plain.signatures = signatures;
                let basis = run(&text, &plain).basis.unwrap();
                let mut params = with_cofactors(0x0b30);
                params.signatures = signatures;
                let outcome = run(&text, &params);
                let name = format!("{case} over {p}, signatures {signatures}");
                assert_eq!(outcome.basis.as_deref(), Some(basis.as_str()), "{name}");
                assert_canonical(&name, &text, &basis, "grevlex");
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                if p == P {
                    assert_eq!(basis, want, "{name}");
                }
            }
            for order in ["lex", "block:1"] {
                accepted_in(&format!("{case} over {p}"), &text, order, Mode::Exact);
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}
