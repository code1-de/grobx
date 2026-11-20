//! handover-time, the item's test code: the edge cases of `OBJECTIVES.md`
//! over GF(p) and ℚ, signatures on and off, with
//! cofactors and the checker (plan.md, _Test code_, 4).

use grobx_checker::Mode;

use crate::common::{EDGE_BASES, P, P31, assert_accepted, params, run, with_cofactors};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// Test 4: the edge cases over 65521 and 2^31 − 1 in grevlex, signatures
/// on and off, with cofactors and the checker in the exact mode, in lex
/// and `block:1`, and over ℚ in grevlex: each basis canonical and
/// accepted, and over 65521 in grevlex as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE_BASES {
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
