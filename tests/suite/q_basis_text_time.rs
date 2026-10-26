//! q-basis-text-time, the item's integration tests: the edge cases of
//! `OBJECTIVES.md` over ℚ with integers past the passes' bound; the
//! integers of a basis over ℚ around the bounds of the splits; and a
//! cofactor text over ℚ with integers past the bound (plan.md, _Test
//! code_, 4 to 6). Every integer of a text named is num-bigint's
//! `Display`, and every basis the exact checker accepts with its
//! cofactors.

use num_bigint::BigUint;

use crate::common::{params, run};
use crate::q::accepted_q;

/// a of 20 limbs: 2^1279 − 1.
fn a20() -> BigUint {
    (BigUint::from(1u32) << 1279usize) - 1u32
}

/// b of 70 limbs: 3^2800.
fn b70() -> BigUint {
    num_traits::pow(BigUint::from(3u32), 2800)
}

/// 10^n.
fn ten(n: usize) -> BigUint {
    num_traits::pow(BigUint::from(10u32), n)
}

/// Test 4: the edge cases on x, y, z over ℚ, a of 20 limbs and b of 70 in
/// the generators where the case holds integers, each canonical and
/// accepted by the exact checker with its cofactors, and its text, where
/// the case gives it, byte for byte.
#[test]
fn the_edge_cases_over_q() {
    let (a, b) = (a20(), b70());
    assert_eq!(a.bits().div_ceil(64), 20);
    assert_eq!(b.bits().div_ceil(64), 70);
    let ab = &a * &b;
    let cases: [(&str, String, Option<String>); 7] = [
        ("the zero ideal", "0".into(), Some("[]:\n".into())),
        ("F = (0, 0)", "0,\n0".into(), Some("[]:\n".into())),
        ("no generator", String::new(), Some("[]:\n".into())),
        (
            "the unit ideal",
            format!("{a}*x*z-1,\nx"),
            Some("[1]:\n".into()),
        ),
        (
            "already a Gröbner basis",
            format!("x+{a}*z,\nz^2+{b}"),
            Some(format!("[x+{a}*z,\nz^2+{b}]:\n")),
        ),
        (
            "the zero polynomial among the generators",
            format!("x-{a},\n0,\ny-{b}*x"),
            Some(format!("[y-{ab},\nx-{a}]:\n")),
        ),
        (
            "an eliminable variable",
            format!("x-{a}*y^2-z,\ny*z-1,\nz^2-y"),
            None,
        ),
    ];
    for (case, generators, want) in cases {
        let text = format!("x,y,z\n0\n{generators}\n");
        let outcome = accepted_q(case, &text, "grevlex");
        if let Some(want) = want {
            assert_eq!(outcome.basis.as_deref(), Some(want.as_str()), "{case}");
        }
    }
}

/// Test 5: over ℚ the generators x − a, y − b·x, and z − c have the basis
/// z − c, y − a·b, and x − a. The integers lie at the passes' bound, 12
/// and 13 limbs; at parts of 8 and 9 chunks; at a quotient of one chunk
/// over a remainder with zero chunks; and at a part below its split's
/// power. The text holds each as num-bigint's `Display` writes it.
#[test]
fn integers_around_the_splits() {
    let one = BigUint::from(1u32);
    let cases = [
        (
            (&one << 768usize) - 1u32,
            BigUint::from(3u32),
            BigUint::from(7u32),
        ),
        ((&one << 768usize) + 1u32, ten(456) - 1u32, ten(152)),
        (ten(171), ten(304) + ten(19), ten(152) - 1u32),
        (ten(1216) + ten(19) + 1u32, ten(608) - 1u32, &one << 64usize),
    ];
    for (a, b, c) in cases {
        let text = format!("x,y,z\n0\nx-{a},\ny-{b}*x,\nz-{c}\n");
        let outcome = run(&text, &params(0x0b5d));
        let ab = &a * &b;
        let want = format!("[z-{c},\ny-{ab},\nx-{a}]:\n");
        assert_eq!(
            outcome.basis.as_deref(),
            Some(want.as_str()),
            "a of {a_bits} bits, b of {b_bits}",
            a_bits = a.bits(),
            b_bits = b.bits()
        );
    }
}

/// Test 6: x·y − a and b·x − 1 over ℚ, a of 15 limbs and b of 70, have the
/// basis y − a·b and b·x − 1; the exact checker reads the cofactors'
/// integers from the cofactor text and accepts.
#[test]
fn cofactors_with_large_integers() {
    let a = (BigUint::from(1u32) << 959usize) + 1u32;
    let b = b70();
    assert_eq!(a.bits().div_ceil(64), 15);
    let text = format!("x,y\n0\nx*y-{a},\n{b}*x-1\n");
    let outcome = accepted_q("x·y − a and b·x − 1", &text, "grevlex");
    let ab = &a * &b;
    let want = format!("[y-{ab},\n{b}*x-1]:\n");
    assert_eq!(outcome.basis.as_deref(), Some(want.as_str()));
}
