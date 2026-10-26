//! many-variables-time, the item's test code: calls on more than 32
//! variables whose generators' variables share the divisor masks' bits,
//! against the same systems on their own variables, and the edge cases of
//! `OBJECTIVES.md` on 40 variables
//! (`iterations/0070-many-variables-time/plan.md`, _Test code_, 5 and 6).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, params_in, random_square, random_system, run};
use crate::large::LARGE;

/// A prime of [2^61, 2^62): 2^62 − 57.
pub(crate) const P62: u64 = LARGE[3];

/// The numbers of variables, each with the indices, from 0, its systems'
/// variables take in order: indices 32 apart share a bit of the mask.
const SPREADS: [(usize, [usize; 4]); 5] = [
    (33, [0, 1, 31, 32]),
    (40, [0, 7, 32, 39]),
    (64, [0, 31, 32, 63]),
    (65, [0, 32, 33, 64]),
    (100, [3, 35, 67, 99]),
];

/// The instance `text`, whose variables are single letters, on x1, …, xn:
/// its k-th variable renamed x(at[k] + 1), every xi declared.
fn spread(text: &str, n: usize, at: &[usize]) -> String {
    let mut lines = text.splitn(2, '\n');
    let names: Vec<char> = lines
        .next()
        .expect("a variable line")
        .split(',')
        .map(|v| {
            let mut c = v.chars();
            let first = c.next().expect("a name");
            assert!(c.next().is_none(), "a name of one letter: {v:?}");
            first
        })
        .collect();
    assert!(names.len() <= at.len(), "{names:?}");
    let rest = lines.next().unwrap_or("");
    let declared: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let mut out = declared.join(",");
    out.push('\n');
    for c in rest.chars() {
        match names.iter().position(|&v| v == c) {
            Some(k) => {
                out.push('x');
                out.push_str(&(at[k] + 1).to_string());
            }
            None => out.push(c),
        }
    }
    out
}

/// The basis text `text` on x1, …, xn with each x(at[k] + 1) renamed the
/// k-th of `names`; every variable of the text is one of them.
fn gather(text: &str, at: &[usize], names: &[char]) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != 'x' {
            out.push(c);
            continue;
        }
        let mut digits = String::new();
        while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
            digits.push(d);
            chars.next();
        }
        let index: usize = digits.parse::<usize>().expect("a variable x<i>") - 1;
        let k = at.iter().position(|&a| a == index).unwrap_or_else(|| {
            panic!(
                "x{number} is none of the system's variables\n{text}",
                number = index + 1
            )
        });
        out.push(names[k]);
    }
    out
}

/// The single-letter variables of `text`'s first line.
fn letters(text: &str) -> Vec<char> {
    text.lines()
        .next()
        .expect("a variable line")
        .split(',')
        .map(|v| v.chars().next().expect("a name"))
        .collect()
}

/// The call on `text` spread onto n variables at `at`, and on `text`
/// itself, under `params`: the first's basis text, its names gathered, is
/// the second's. Returns the spread instance.
fn agree(case: &str, text: &str, n: usize, at: &[usize], params: &grobx::Params) -> String {
    let wide = spread(text, n, at);
    let got = run(&wide, params).basis.expect("a basis");
    let want = run(text, params).basis.expect("a basis");
    assert_eq!(
        gather(&got, at, &letters(text)),
        want,
        "{case} on {n} variables\n{wide}"
    );
    wide
}

/// Test 5, many-variables-time.1 to .4: random and random square systems
/// of seeds 0 to 2 spread onto 33 to 100 variables, over 65521, 2^31 − 1, 2^62 − 57, and
/// ℚ, in grevlex, signatures on and off over GF(p): each basis is the
/// system's on its own variables. With cofactors over 65521 and ℚ the
/// checker accepts the spread call's result in the exact mode. Lex is left
/// out: a spread system's ideal is positive-dimensional in its unused
/// variables, so a lex call runs F4 in lex where the system on its own
/// variables changes its basis by FGLM.
#[test]
fn systems_on_many_variables() {
    for p in [P, P31, P62, 0] {
        for seed in 0..3 {
            let systems = [
                (format!("random system {seed}"), random_system(p, seed)),
                (format!("random square {seed}"), random_square(p, seed)),
            ];
            for (case, text) in &systems {
                for &(n, at) in &SPREADS {
                    let signatures: &[bool] = if p == 0 { &[true] } else { &[true, false] };
                    for &on in signatures {
                        let mut params = params_in("grevlex", seed);
                        params.signatures = on;
                        let case = format!("{case} over {p}, signatures {on}");
                        agree(&case, text, n, &at, &params);
                    }
                    if p == P || p == 0 {
                        let mut params = params_in("grevlex", seed);
                        params.cofactors = true;
                        let wide = spread(text, n, &at);
                        let outcome = run(&wide, &params);
                        assert_accepted(case, &wide, &outcome, Mode::Exact);
                    }
                }
            }
        }
    }
}

/// The edge cases of `OBJECTIVES.md` on the variables a and b: name,
/// generators, and the basis where it is known before the call.
pub(crate) const EDGE: [(&str, &str, Option<&str>); 6] = [
    ("the zero ideal", "0", Some("[]:\n")),
    ("F = (0, 0)", "0,\n0", Some("[]:\n")),
    ("the unit ideal", "a*b-1,\na", Some("[1]:\n")),
    ("already a Gröbner basis", "a+b,\nb^2+1", None),
    (
        "the zero polynomial among the generators",
        "a*b-b,\n0,\nb^2-a",
        None,
    ),
    ("an eliminable variable", "a-b^2,\nb^3-2", None),
];

/// Test 6: the edge cases on 40 variables, a and b at x1 and x33, whose
/// masks share their bit, over 65521 and ℚ: each basis the system's on a
/// and b, and the bases known before the call as written.
#[test]
fn the_edge_cases_on_40_variables() {
    let at = [0, 32];
    for p in [P, 0] {
        for (case, generators, known) in EDGE {
            let text = format!("a,b\n{p}\n{generators}\n");
            let params = params_in("grevlex", 1);
            let wide = agree(case, &text, 40, &at, &params);
            if let Some(known) = known {
                assert_eq!(
                    run(&wide, &params).basis.as_deref(),
                    Some(known),
                    "{case} over {p}"
                );
            }
        }
    }
}
