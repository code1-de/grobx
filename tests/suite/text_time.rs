//! text-time, the item's test code: the edge cases of `OBJECTIVES.md`;
//! long names and monomials that elements share; and the integers of a
//! basis over ℚ on both
//! sides of 64 limbs (plan.md, _Test code_, 4, 5, and 7). The tests check the
//! texts over 65521 in grevlex, and the zero and the unit ideal over ℚ,
//! byte for byte, and the other edge cases as the exact checker accepts
//! them.

use grobx_checker::Mode;
use num_bigint::BigUint;

use crate::common::{P, assert_accepted, params, run, with_cofactors};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// The edge cases of `OBJECTIVES.md` on x, y, z, as table-read-time's test
/// 3 takes them: name, generators, and the basis over GF(65521) in
/// grevlex.
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

/// Test 4: the edge cases over 65521 in grevlex, signatures on and off,
/// with cofactors, each basis canonical, accepted in the exact mode, and as
/// written; in lex and `block:1`, accepted; over ℚ in grevlex, accepted,
/// and the zero ideal and the unit ideal as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut params = with_cofactors(0x7e70);
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
        let outcome = accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
        if want == "[]:\n" || want == "[1]:\n" {
            assert_eq!(outcome.basis.as_deref(), Some(want), "{case} over ℚ");
        }
    }
}

/// Twelve names: eight of 60 characters, at 0 to 3 and 8 to 11, x at 4,
/// and three of 29, 30, and 31 characters at 5, 6, and 7.
fn long_names() -> Vec<String> {
    (0..12)
        .map(|i| {
            if i == 4 {
                return "x".to_string();
            }
            let len = match i {
                5 => 29,
                6 => 30,
                7 => 31,
                _ => 60,
            };
            let mut s = format!("variable_{i}_");
            while s.len() < len {
                s.push(char::from(b'a' + (s.len() % 26) as u8));
            }
            s
        })
        .collect()
}

/// The canonical text of a monomial of these exponents on these names.
fn monomial(names: &[String], exponents: &[u32]) -> String {
    names
        .iter()
        .zip(exponents)
        .filter(|&(_, &e)| e > 0)
        .map(|(name, &e)| match e {
            1 => name.clone(),
            _ => format!("{name}^{e}"),
        })
        .collect::<Vec<_>>()
        .join("*")
}

/// The tail monomials s of test 5, of total degree at most 5, each with
/// the length of its text: the name of 31 characters alone; the one of 30
/// squared; the one of 29 squared and x; and four of 60 and the one of 29.
const TAILS: [([u32; 12], usize); 4] = [
    ([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 31),
    ([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], 32),
    ([0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0], 33),
    ([1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], 273),
];

/// Test 5: on the twelve names, the two generators L1 + c·s and L2 + c·s,
/// L1 the product of the first six variables and L2 of the last six. Their
/// leading monomials are coprime and s, of degree at most 5, is divisible
/// by neither, so the two are the reduced Gröbner basis, L2's element first
/// in grevlex, and both hold s. Over 65521 with c = 1 and c = p − 1 and over
/// ℚ with c = 2^70 + 1: each basis as written, and accepted in the exact
/// mode with cofactors. And one generator alone whose leading monomial's
/// text passes 255 bytes: its basis as written.
#[test]
fn long_names_and_shared_monomials() {
    let names = long_names();
    let header = names.join(",");
    let l1 = monomial(&names, &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
    let l2 = monomial(&names, &[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    assert!(l1.len() > 32 && l2.len() > 32, "{l1} {l2}");
    let big = "1180591620717411303425"; // 2^70 + 1
    for (s, len) in TAILS {
        let s = monomial(&names, &s);
        assert_eq!(s.len(), len, "{s}");
        for (characteristic, c) in [(P, "1"), (P, "65520"), (0, big)] {
            let term = if c == "1" {
                s.clone()
            } else {
                format!("{c}*{s}")
            };
            let text = format!("{header}\n{characteristic}\n{l1}+{term},\n{l2}+{term}\n");
            let want = format!("[{l2}+{term},\n{l1}+{term}]:\n");
            let case = format!("a tail of {len} bytes, c = {c}, over {characteristic}");
            if characteristic == 0 {
                let outcome = accepted_q(&case, &text, "grevlex");
                assert_eq!(outcome.basis.as_deref(), Some(want.as_str()), "{case}");
            } else {
                let outcome = run(&text, &with_cofactors(0x7e71));
                assert_accepted(&case, &text, &outcome, Mode::Exact);
                assert_eq!(outcome.basis.as_deref(), Some(want.as_str()), "{case}");
            }
        }
    }
    // One generator alone: its leading monomial of degree 10 on five of the
    // long names, past 255 bytes, and the tail 3·x.
    let lead = monomial(&names, &[2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0]);
    assert!(lead.len() > 255, "{lead}");
    for characteristic in [P, 0] {
        let text = format!("{header}\n{characteristic}\n{lead}+3*x\n");
        let outcome = run(&text, &params(0x7e72));
        assert_eq!(
            outcome.basis.as_deref(),
            Some(format!("[{lead}+3*x]:\n").as_str()),
            "one generator over {characteristic}"
        );
    }
}

/// Test 7: over ℚ, the generators x − a, y − b·x, and z − 7, with
/// a = 2^3000 + 1 and b = 2^2000 + 3, have the basis z − 7, y − a·b, and
/// x − a. a·b passes 64 limbs, a has 65 to 4096 bits, and 7 fits a word.
/// The basis text holds each integer as num-bigint writes it.
#[test]
fn integers_on_both_sides_of_64_limbs() {
    let one = BigUint::from(1u32);
    let a = (&one << 3000usize) + 1u32;
    let b = (&one << 2000usize) + 3u32;
    let ab = &a * &b;
    assert!(ab.bits() > 4096 && a.bits() > 64 && a.bits() <= 4096);
    let text = format!("x,y,z\n0\nx-{a},\ny-{b}*x,\nz-7\n");
    let outcome = run(&text, &params(0x7e74));
    let want = format!("[z-7,\ny-{ab},\nx-{a}]:\n");
    assert_eq!(outcome.basis.as_deref(), Some(want.as_str()));
}
