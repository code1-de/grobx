//! result-time, the item's test code: the basis text and the cofactor text
//! that a call writes over GF(p) and over ℚ, byte for byte, through the
//! crate's public API (result-time.1 to .5 and .7 to .16, .6 withdrawn
//! by text-time; plan.md, _Test code_, 12 to 14).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, dense, params, run, with_cofactors};
use crate::large::LARGEST;

/// The edge cases of `OBJECTIVES.md` on x, y, z: name and generators.
const CASES: [(&str, &str); 7] = [
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

/// The basis text and the cofactor text of case k over a characteristic
/// whose texts of −1 and −2 are `m` and `m2`, `-1` and `-2` over ℚ.
fn want(k: usize, m: &str, m2: &str) -> (String, String) {
    let neg = |s: &str| {
        if m == "-1" {
            format!("-{s}")
        } else {
            format!("{m}*{s}")
        }
    };
    // The texts over GF(p) write −1 as p − 1 and −2 as p − 2 before a
    // monomial, and over ℚ as `-` and `-2*`.
    let plus = |s: &str| {
        if m == "-1" {
            format!("-{s}")
        } else {
            format!("+{m}*{s}")
        }
    };
    let plus2 = |s: &str| {
        if m == "-1" {
            format!("-2*{s}")
        } else {
            format!("+{m2}*{s}")
        }
    };
    let constant = if m == "-1" {
        "-1".to_string()
    } else {
        format!("+{m}")
    };
    match k {
        0..=2 => ("[]:\n".to_string(), "[]:\n".to_string()),
        3 => ("[1]:\n".to_string(), format!("[[{m},z]]:\n")),
        4 => (
            "[x+z,\nz^2+1]:\n".to_string(),
            "[[1,0],\n[0,1]]:\n".to_string(),
        ),
        5 => (
            format!(
                "[y^2{plus_x},\nx*y{plus_z},\nx^2{plus_yz}]:\n",
                plus_x = plus("x"),
                plus_z = plus("z"),
                plus_yz = plus("y*z")
            ),
            format!("[[0,0,1],\n[1,0,0],\n[y,0,{neg_x}]]:\n", neg_x = neg("x")),
        ),
        6 => (
            format!(
                "[x{plus2_z},\nz^2{plus_y},\ny*z{constant},\ny^2{plus_z}]:\n",
                plus2_z = plus2("z"),
                plus_y = plus("y"),
                plus_z = plus("z")
            ),
            format!(
                "[[1,z,{neg_y}],\n[0,0,1],\n[0,1,0],\n[0,z,{neg_y}]]:\n",
                neg_y = neg("y")
            ),
        ),
        _ => unreachable!(),
    }
}

/// The edge cases over 65521, 2^31 − 1, and the largest prime below 2^64,
/// and over ℚ, with cofactors: the basis text and the cofactor text as
/// written here, and accepted by the checker in the exact mode.
#[test]
fn the_edge_cases_as_written() {
    for p in [P, P31, LARGEST, 0] {
        let (m, m2) = if p == 0 {
            ("-1".to_string(), "-2".to_string())
        } else {
            ((p - 1).to_string(), (p - 2).to_string())
        };
        for (k, (case, generators)) in CASES.iter().enumerate() {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            let outcome = run(&text, &with_cofactors(0x0071));
            let (basis, cofactors) = want(k, &m, &m2);
            let name = format!("{case} over {p}");
            assert_eq!(outcome.basis.as_deref(), Some(basis.as_str()), "{name}");
            assert_eq!(
                outcome.cofactors.as_deref(),
                Some(cofactors.as_str()),
                "{name}"
            );
            assert_accepted(&name, &text, &outcome, Mode::Exact);
        }
    }
}

/// Over ℚ, numbers at the edge of 64 bits: x − c·y for c = 2^64 − 1,
/// 2^64, and 2^64 + 1, whose basis is the generator and whose cofactor 1;
/// and c·x − 3c·y, whose basis is x − 3·y and whose cofactor 1/c.
#[test]
fn numbers_at_two_to_the_64() {
    for c in [u128::from(u64::MAX), 1 << 64, (1 << 64) + 1] {
        let text = format!("x,y\n0\nx-{c}*y\n");
        let outcome = run(&text, &with_cofactors(0x0071));
        assert_eq!(
            outcome.basis.as_deref(),
            Some(format!("[x-{c}*y]:\n").as_str()),
            "x − {c}·y"
        );
        assert_eq!(outcome.cofactors.as_deref(), Some("[[1]]:\n"), "x − {c}·y");
        let three = 3 * c;
        let text = format!("x,y\n0\n{c}*x-{three}*y\n");
        let outcome = run(&text, &with_cofactors(0x0071));
        assert_eq!(
            outcome.basis.as_deref(),
            Some("[x-3*y]:\n"),
            "{c}·(x − 3·y)"
        );
        assert_eq!(
            outcome.cofactors.as_deref(),
            Some(format!("[[1/{c}]]:\n").as_str()),
            "{c}·(x − 3·y)"
        );
        assert_accepted(&format!("{c}·(x − 3·y)"), &text, &outcome, Mode::Exact);
    }
}

/// The variables of `text`, an instance or a basis text, renamed: each
/// name of `from` becomes the name of `to` at its place. A name is a run of
/// ASCII letters, digits, and underscores that starts with a letter or an
/// underscore.
fn renamed(text: &str, from: &[&str], to: &[String]) -> String {
    let mut out = String::with_capacity(2 * text.len());
    let bytes = text.as_bytes();
    let mut k = 0;
    while k < bytes.len() {
        let b = bytes[k];
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = k;
            while k < bytes.len() && (bytes[k].is_ascii_alphanumeric() || bytes[k] == b'_') {
                k += 1;
            }
            let name = &text[start..k];
            let i = from.iter().position(|&f| f == name).expect("a variable");
            out.push_str(&to[i]);
        } else {
            out.push(b as char);
            k += 1;
        }
    }
    out
}

/// Exponents at the degree bounds of the three formats, x^127 − 1,
/// x^32767 − y, and x^2147483647 − 1, over 65521 and over ℚ; and a dense
/// cubic system on three variables over 65521 with its variables renamed by names
/// of 30 characters and more, its basis text the original's with the names
/// replaced.
#[test]
fn wide_exponents_and_long_names() {
    for (generator, over_p, over_q) in [
        ("x^127-1", "[x^127+65520]:\n", "[x^127-1]:\n"),
        ("x^32767-y", "[x^32767+65520*y]:\n", "[x^32767-y]:\n"),
        (
            "x^2147483647-1",
            "[x^2147483647+65520]:\n",
            "[x^2147483647-1]:\n",
        ),
    ] {
        for (p, want) in [(P, over_p), (0, over_q)] {
            let text = format!("x,y\n{p}\n{generator}\n");
            assert_eq!(
                run(&text, &params(0x0071)).basis.as_deref(),
                Some(want),
                "{generator} over {p}"
            );
        }
    }
    let stem = "cubic-3-p65521";
    let text = dense(3, 3, P, 1);
    let original = run(&text, &params(0x0071)).basis.unwrap();
    let from: Vec<&str> = text.lines().next().unwrap().split(',').collect();
    let to: Vec<String> = from
        .iter()
        .enumerate()
        .map(|(i, f)| {
            format!(
                "a_variable_named_at_length_{pad}_{f}",
                pad = "x".repeat(3 * i)
            )
        })
        .collect();
    let long = renamed(&text, &from, &to);
    let basis = run(&long, &params(0x0071)).basis.unwrap();
    assert_eq!(basis, renamed(&original, &from, &to), "{stem}, renamed");
}
