//! matrix-build-time, the item's test code: calls whose matrices are built
//! by `Table::row_products` and by the reducer searches that test their
//! masks eight at a time (matrix-build-time.1 to .8), over GF(p) and ℚ, in
//! grevlex, lex, and a block order, with cofactors, in formats of one to
//! five words and past the degree bounds of `Dense8` and `Dense16`
//! (plan.md, _Test code_, 5 and 6).

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

/// Test 5: the edge cases over 65521 and 2^31 − 1 in grevlex, signatures
/// on and off, with cofactors and the checker in the exact mode, in lex and
/// `block:1`, and over ℚ in grevlex and lex: each basis canonical and
/// accepted, and over 65521 in grevlex as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in [P, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut params = with_cofactors(0x0b20);
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
        for order in ["grevlex", "lex"] {
            accepted_q(&format!("{case} over ℚ"), &text, order);
        }
    }
}

/// The variables x1 to xn.
fn variables(n: usize) -> String {
    (1..=n)
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// A system on n ≥ 3 variables whose matrices hold monomials of every word
/// of the format: the chain `x_i − x_{i+1} + i` for i < n − 2, and on the last
/// three variables a, b, c the eliminable case a − b^2 − c, b·c − 1,
/// c^2 − b; on two variables the eliminable case's last two, and on one
/// x1^3 − x1 − 1.
fn chain(n: usize, p: u64) -> String {
    let mut generators: Vec<String> = Vec::new();
    if n >= 3 {
        for i in 1..n - 2 {
            generators.push(format!("x{i}-x{next}+{i}", next = i + 1));
        }
        let (a, b, c) = (n - 2, n - 1, n);
        generators.push(format!("x{a}-x{b}^2-x{c}"));
        generators.push(format!("x{b}*x{c}-1"));
        generators.push(format!("x{c}^2-x{b}"));
    } else if n == 2 {
        generators.push("x1*x2-1".to_string());
        generators.push("x2^2-x1".to_string());
    } else {
        generators.push("x1^3-x1-1".to_string());
    }
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = variables(n),
        generators = generators.join(",\n")
    )
}

/// The chain `x_i − x_{i+1} + i` for i < n − 1, and on the last two
/// variables the pair `x_{n−1}^e·x_n − 1` and `x_{n−1}·x_n^e − 1`, of degree
/// e + 1, whose lcm has degree 2e,
/// past `Dense8`'s 127 at e = 100 and past `Dense16`'s 32767 at e = 20000.
fn past(n: usize, p: u64, e: u64) -> String {
    let mut generators: Vec<String> = Vec::new();
    for i in 1..n - 1 {
        generators.push(format!("x{i}-x{next}+{i}", next = i + 1));
    }
    let (a, b) = (n - 1, n);
    generators.push(format!("x{a}^{e}*x{b}-1"));
    generators.push(format!("x{a}*x{b}^{e}-1"));
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = variables(n),
        generators = generators.join(",\n")
    )
}

/// Test 6: systems on 1 to 40 variables, one to five words of `Dense8`,
/// over 65521 and 2^31 − 1, signatures on and off, the same basis with
/// cofactors and accepted in the exact mode, and over ℚ accepted; and on 2,
/// 5, and 9 variables past the degree bounds of `Dense8` and `Dense16`,
/// the runs that go on in `Dense16`'s one to three words and `Dense32`'s
/// one, three, and five.
#[test]
fn every_word_count() {
    for n in [1, 2, 3, 8, 9, 17, 25, 33, 40] {
        for p in [P, P31] {
            let text = chain(n, p);
            let mut bases = Vec::new();
            for signatures in [true, false] {
                let mut params = with_cofactors(0x0b21);
                params.signatures = signatures;
                let outcome = run(&text, &params);
                let name = format!("{n} variables over {p}, signatures {signatures}");
                assert_canonical(&name, &text, outcome.basis.as_deref().unwrap(), "grevlex");
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                bases.push(outcome.basis.unwrap());
                let mut plain = params.clone();
                plain.cofactors = false;
                bases.push(run(&text, &plain).basis.unwrap());
            }
            assert!(
                bases.windows(2).all(|w| w[0] == w[1]),
                "{n} variables over {p}: the runs differ"
            );
        }
        accepted_q(&format!("{n} variables over ℚ"), &chain(n, 0), "grevlex");
    }
    for n in [2, 5, 9] {
        for p in [P, P31] {
            for e in [100, 20000] {
                let text = past(n, p, e);
                let mut bases = Vec::new();
                for signatures in [true, false] {
                    let mut params = params(0x0b22);
                    params.signatures = signatures;
                    bases.push(run(&text, &params).basis.unwrap());
                }
                let outcome = accepted_in(
                    &format!("{n} variables, e = {e}"),
                    &text,
                    "grevlex",
                    Mode::Exact,
                );
                bases.push(outcome.basis.unwrap());
                assert!(
                    bases.windows(2).all(|w| w[0] == w[1]),
                    "{n} variables over {p}, e = {e}: the runs differ"
                );
            }
        }
    }
}
