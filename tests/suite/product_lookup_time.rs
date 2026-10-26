//! product-lookup-time, the item's test code: calls whose rows the product
//! lookup builds without bounds checks and with the hash's low bits in each
//! slot (product-lookup-time.1 to .7), over GF(p) and ℚ, in grevlex and
//! lex, with cofactors and through FGLM, and past the degree bounds of
//! `Dense8` and `Dense16` (plan.md, _Test code_, 4, 6, and 7).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, dense, params, run, with_cofactors};
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

/// The edge cases over 65521 and 2^31 − 1 in grevlex, signatures on and
/// off, and in lex, and over ℚ in grevlex and lex: each basis canonical
/// and accepted by the checker in the exact mode with its cofactors, and
/// over 65521 in grevlex as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in [P, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut params = with_cofactors(0x100c);
                params.signatures = signatures;
                let outcome = run(&text, &params);
                let basis = outcome.basis.as_deref().unwrap();
                let name = format!("{case} over {p}, signatures {signatures}");
                assert_canonical(&name, &text, basis, "grevlex");
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                if p == P {
                    assert_eq!(basis, want, "{name}");
                }
            }
            accepted_in(&format!("{case} over {p}"), &text, "lex", Mode::Exact);
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        for order in ["grevlex", "lex"] {
            accepted_q(&format!("{case} over ℚ"), &text, order);
        }
    }
}

/// The pair's lcm has degree 200, past `Dense8`'s 127, and 40000, past
/// `Dense16`'s 32767: an lcm or a product past the format returns
/// Overflow, and the call goes on in the next wider format, over 65521 and
/// 2^31 − 1, signatures on and off; each basis as written and accepted
/// with its cofactors in the exact mode.
#[test]
fn past_dense8_and_dense16() {
    for p in [P, P31] {
        let minus = p - 1;
        for e in [100u64, 20000] {
            let text = format!("x,y\n{p}\nx^{e}*y-1,\nx*y^{e}-1\n");
            let want = format!(
                "[x^{less1}+{minus}*y^{less1},\nx*y^{e}+{minus},\n\
                 y^{twice_less1}+{minus}*x^{less2}]:\n",
                less1 = e - 1,
                twice_less1 = 2 * e - 1,
                less2 = e - 2
            );
            for signatures in [true, false] {
                let mut params = params(0x100c);
                params.signatures = signatures;
                let basis = run(&text, &params).basis;
                assert_eq!(
                    basis.as_deref(),
                    Some(want.as_str()),
                    "{text:?}, signatures {signatures}"
                );
            }
            let outcome = accepted_in("past", &text, "grevlex", Mode::Exact);
            assert_eq!(outcome.basis.as_deref(), Some(want.as_str()), "{text:?}");
        }
    }
}

/// The cofactor table's lookups and FGLM's: dense cubic systems with
/// cofactors on three and four variables over 65521 and on two and three
/// over ℚ, and on three variables in lex over 65521 and 2^31 − 1, each
/// accepted by the checker in the exact mode.
#[test]
fn cofactors_and_fglm() {
    for m in 3..=4 {
        let text = dense(m, 3, P, m as u64);
        accepted_in(&format!("cubic-{m} over {P}"), &text, "grevlex", Mode::Exact);
    }
    for m in 2..=3 {
        accepted_q(&format!("cubic-{m} over ℚ"), &dense(m, 3, 0, m as u64), "grevlex");
    }
    for p in [P, P31] {
        let text = dense(3, 3, p, 3);
        accepted_in(&format!("cubic-3 over {p}"), &text, "lex", Mode::Exact);
    }
}
