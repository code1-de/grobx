//! matrix-sort-time, the item's test code: calls whose matrices sort their
//! columns and their rows on grevlex keys (matrix-sort-time.1 to .8), over
//! GF(p) and ℚ, in grevlex, lex, and block orders, with cofactors, and in
//! the wider formats, where two monomials' keys can agree (plan.md, _Test
//! code_, 4, 6, and 7).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, dense, params, random_square, run, with_cofactors};
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
/// off, in lex, and in `block:1`, and over ℚ in the three: each basis
/// canonical and accepted by the checker in the exact mode with its
/// cofactors, and over 65521 in grevlex as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in [P, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut params = with_cofactors(0x5047);
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
            for order in ["lex", "block:1"] {
                accepted_in(&format!("{case} over {p}"), &text, order, Mode::Exact);
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        for order in ["grevlex", "lex", "block:1"] {
            accepted_q(&format!("{case} over ℚ"), &text, order);
        }
    }
}

/// `text` with a last variable t and the generator t^e − 1.
fn with_t(text: &str, e: u64) -> String {
    let (names, rest) = text.split_once('\n').unwrap();
    format!(
        "{names},t\n{generators},\nt^{e}-1\n",
        generators = rest.trim_end()
    )
}

/// A dense cubic system over 65521 on six variables, with a seventh, t, and
/// t^130 − 1, which the call computes in `Dense16`, or t^40000 − 1, in
/// `Dense32`: the key holds the degree and five or two exponents, and
/// distinct monomials of the dense system share their keys. Each basis is
/// the dense system's with t^e − 1 last, signatures on and off. A dense
/// cubic system over 65521 on four variables with t^130 − 1 is accepted
/// with its cofactors by the checker in the exact mode.
#[test]
fn wider_formats_and_long_keys() {
    // test-mq-systems.9 design -- the dense cubic system on six variables, so that beside t its distinct monomials share their keys, where the map's on five variables gives no two monomials one key (not measured)
    let text = dense(6, 3, P, 1);
    let alone = run(&text, &params(0x5047)).basis.unwrap();
    for e in [130u64, 40000] {
        let with = with_t(&text, e);
        let want = alone.replace("]:\n", &format!(",\nt^{e}+65520]:\n"));
        for signatures in [true, false] {
            let mut p = params(0x5047);
            p.signatures = signatures;
            let basis = run(&with, &p).basis.unwrap();
            assert_eq!(
                basis, want,
                "cubic-6 with t^{e} − 1, signatures {signatures}"
            );
        }
    }
    let text = with_t(&dense(4, 3, P, 2), 130);
    accepted_in("cubic-4 with t^130 − 1", &text, "grevlex", Mode::Exact);
}

/// The orders that keep the comparison sort: x*y − z^2, y*z − x over
/// 65521, a positive-dimensional ideal that lex runs F4 on, in lex,
/// `block:1`, and `block:2`; a random square system over 65521 on three
/// variables in `block:2`. Each is accepted with its cofactors by the
/// checker in the exact mode.
#[test]
fn orders_without_a_key() {
    let text = format!("x,y,z\n{P}\nx*y-z^2,\ny*z-x\n");
    for order in ["lex", "block:1", "block:2"] {
        accepted_in("x*y − z^2, y*z − x", &text, order, Mode::Exact);
    }
    // test-mq-systems.12 design -- a call that F4 runs in block:2 takes the random square system of degrees 2, 3, and 2, where the map's dense cubic one on three variables runs for seconds (log 0084-3)
    accepted_in("square-27", &random_square(P, 27), "block:2", Mode::Exact);
}
