//! monomial-time, the item's test code: calls through the public API whose
//! monomials the formats hold on packed words keep their results: the
//! checker's verdict at the lane
//! borders, the calls past the bounds 127 and 32767, the edge cases at a
//! lane border, and lex bases of generated systems (plan.md, _Test code_).

use std::fmt::Write;

use grobx_checker::Mode;

use crate::common::{P, Rng, dense, params_in, random_square, run};
use crate::orders::accepted_in;

/// A random instance over GF(65521) on the variables x1, …, xn: two or three
/// generators of two to four terms, each of degree one to three in
/// variables drawn from all n, a coefficient in [1, p).
fn spread(n: usize, seed: u64) -> String {
    let p = 65521u64;
    let mut rng = Rng::new(seed ^ (n as u64).wrapping_mul(0x9e37));
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let mut generators = Vec::new();
    for _ in 0..2 + rng.below(2) {
        let mut terms = Vec::new();
        for _ in 0..2 + rng.below(3) {
            let mut term = (1 + rng.below(p - 1)).to_string();
            let mut e = vec![0u32; n];
            for _ in 0..=rng.below(3) {
                e[rng.below(n as u64) as usize] += 1;
            }
            for (name, &k) in names.iter().zip(&e) {
                if k > 0 {
                    term.push('*');
                    term.push_str(name);
                    if k > 1 {
                        write!(term, "^{k}").unwrap();
                    }
                }
            }
            terms.push(term);
        }
        generators.push(terms.join("+"));
    }
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// Random instances on 7, 8, 9, 15, 16, and 17 variables, the lane borders
/// of 8-bit lanes, in grevlex, lex, and `block:2`, with cofactors: the
/// checker accepts each in the exact mode.
#[test]
fn the_checker_certifies_at_the_lane_borders() {
    for n in [7, 8, 9, 15, 16, 17] {
        for seed in 0..10 {
            let text = spread(n, seed);
            for order in ["grevlex", "lex", "block:2"] {
                accepted_in(
                    &format!("{n} variables, seed {seed}"),
                    &text,
                    order,
                    Mode::Exact,
                );
            }
        }
    }
}

/// The calls past `Dense8`'s bound 127 and `Dense16`'s 32767: from the
/// input, from the run, and from a lex staircase, and a lex staircase at
/// `Dense8`'s bound; each basis as written here, and accepted by the
/// checker.
#[test]
fn degrees_past_127_and_32767() {
    let cases = [
        (
            "x,y\n65521\nx^128-y,\ny^2-1\n",
            "grevlex",
            "[y^2+65520,\nx^128+65520*y]:\n",
        ),
        (
            "x,y\n65521\nx^100*y-1,\nx*y^100-1\n",
            "grevlex",
            "[x^99+65520*y^99,\nx*y^100+65520,\ny^199+65520*x^98]:\n",
        ),
        (
            "x,y\n65521\nx^8-1,\ny^16-1\n",
            "lex",
            "[y^16+65520,\nx^8+65520]:\n",
        ),
        (
            "x,y\n65521\nx-1,\ny^127-1\n",
            "lex",
            "[y^127+65520,\nx+65520]:\n",
        ),
        (
            "x,y\n65521\nx^20000*y-1,\nx*y^20000-1\n",
            "grevlex",
            "[x^19999+65520*y^19999,\nx*y^20000+65520,\ny^39999+65520*x^19998]:\n",
        ),
    ];
    for (text, order, want) in cases {
        let got = accepted_in("past a bound", text, order, Mode::Exact);
        assert_eq!(got.basis.as_deref(), Some(want), "{text:?} in {order}");
    }
}

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, on 8 and 9
/// variables, x_{n−1} and `x_n` the last two: each basis as written here,
/// and accepted by the checker.
#[test]
fn the_edge_cases_at_a_lane_border() {
    for n in [8usize, 9] {
        let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
        let (v, w) = (&names[n - 2], &names[n - 1]);
        let cases = [
            ("the zero ideal", "0".to_string(), "[]:\n".to_string()),
            ("F = (0, 0)", "0,\n0".to_string(), "[]:\n".to_string()),
            ("no generator", String::new(), "[]:\n".to_string()),
            (
                "the unit ideal",
                format!("x1*{w}-1,\nx1"),
                "[1]:\n".to_string(),
            ),
            (
                "already a Gröbner basis",
                format!("x1+{w},\n{w}^2+1"),
                format!("[x1+{w},\n{w}^2+1]:\n"),
            ),
            (
                "the zero polynomial among the generators",
                format!("x1*{v}-{w},\n0,\n{v}^2-x1"),
                format!("[{v}^2+65520*x1,\nx1*{v}+65520*{w},\nx1^2+65520*{v}*{w}]:\n"),
            ),
            (
                "an eliminable variable",
                format!("x1-{v}^2-{w},\n{v}*{w}-1,\n{w}^2-{v}"),
                format!("[x1+65519*{w},\n{w}^2+65520*{v},\n{v}*{w}+65520,\n{v}^2+65520*{w}]:\n"),
            ),
        ];
        for (case, generators, want) in cases {
            let text = format!(
                "{variables}\n65521\n{generators}\n",
                variables = names.join(",")
            );
            let got = accepted_in(case, &text, "grevlex", Mode::Exact);
            assert_eq!(
                got.basis.as_deref(),
                Some(want.as_str()),
                "{case}, {n} variables"
            );
        }
    }
}

/// Dense and random square instances in lex, whose change of order sorts
/// its staircase in lex: the call without cofactors returns the basis of
/// the call with cofactors, which is canonical in lex and accepted by the
/// checker in it.
#[test]
fn lex_on_generated_systems() {
    let mut texts: Vec<String> = (2..=3).map(|m| dense(m, 3, P, 1)).collect();
    texts.extend((0..2).map(|seed| random_square(P, seed)));
    for text in texts {
        let ours = run(&text, &params_in("lex", 0x7a11)).basis;
        let checked = accepted_in("lex", &text, "lex", Mode::Exact);
        assert_eq!(ours, checked.basis, "{text}");
    }
}
