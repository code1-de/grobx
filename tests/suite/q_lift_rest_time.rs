//! q-lift-rest-time, the item's integration tests: calls over ℚ whose
//! replays run in pairs, which merge a pair's terms once and attempt once
//! a pair, return the basis text of the call with cofactors, whose replays
//! run alone and which the exact checker accepts: the edge cases of
//! `OBJECTIVES.md`, elements whose denominators do not nest, and lex and
//! block orders (`iterations/0088-q-lift-rest-time/plan.md`, _Test code_,
//! 6 to 8).

use std::fmt::Write;

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::One;

use crate::common::{params_in, run};
use crate::q::accepted_q;
use crate::run_generators_time::EDGE;

/// The edge cases again, with fractions among the coefficients.
const FRACTIONS: [(&str, &str); 3] = [
    ("already a Gröbner basis, fractions", "x+1/3*z,\nz^2+2/5"),
    ("the unit ideal, fractions", "2/3*x*z-1,\n5/7*x"),
    (
        "an eliminable variable, fractions",
        "x-1/2*y^2-z,\n3*y*z-1,\nz^2-1/7*y",
    ),
];

/// In the order named `order`, the calls without cofactors under seeds 1
/// to 4 return the basis text of the call with them, which `accepted_q`
/// checks.
fn agrees(case: &str, instance: &str, order: &str) {
    let with = accepted_q(case, instance, order);
    for seed in 1..=4 {
        let without = run(instance, &params_in(order, seed));
        assert_eq!(
            without.basis, with.basis,
            "{case} in {order}, seed {seed}\n{instance}"
        );
    }
}

/// Test 6: the edge cases on x, y, z over ℚ, and with fractions.
#[test]
fn the_edge_cases() {
    for (case, gens) in EDGE.into_iter().chain(FRACTIONS) {
        agrees(case, &format!("x,y,z\n0\n{gens}\n"), "grevlex");
    }
}

/// The least integer from `from` on that is prime to `to`.
fn prime_to(mut from: BigUint, to: &BigUint) -> BigUint {
    while !from.gcd(to).is_one() {
        from += 1u32;
    }
    from
}

/// D·v₀ + a₁·v₁ + … + a₆·v₆, D = d₁⋯d₆ for the pairwise coprime dᵢ =
/// pᵢ^eᵢ of about 40 bits, aᵢ = nᵢ·D/dᵢ: the monic element's coefficients
/// nᵢ/dᵢ have denominators that do not nest. n₁ of about 900 bits, n₂ to
/// n₆ of about 150, nᵢ prime to dᵢ, each from `salt`.
fn unnested(vars: &[String], salt: u32) -> String {
    let d: Vec<BigUint> = [(3u32, 25u32), (5, 17), (7, 14), (11, 11), (13, 10), (17, 9)]
        .iter()
        .map(|&(p, e)| num_traits::pow(BigUint::from(p), e as usize))
        .collect();
    let big_d = d.iter().fold(BigUint::one(), |acc, x| acc * x);
    let mut text = format!("{big_d}*{first}", first = vars[0]);
    for (i, di) in d.iter().enumerate() {
        let bits = if i == 0 { 900 } else { 150 };
        let seed = (BigUint::one() << bits) / (BigUint::from(salt) + 2u32 + i as u32);
        let n = prime_to(seed, di);
        let a = n * (&big_d / di);
        write!(text, "+{a}*{var}", var = vars[i + 1]).unwrap();
    }
    text
}

/// Test 7: one generator `unnested` on seven variables, and three such
/// generators on disjoint variables, over ℚ.
#[test]
fn unnested_denominators() {
    let vars: Vec<String> = (0..21).map(|i| format!("v{i}")).collect();
    let one = unnested(&vars[..7], 1);
    let names = vars[..7].join(",");
    agrees("one generator", &format!("{names}\n0\n{one}\n"), "grevlex");
    let three: Vec<String> = (0..3)
        .map(|g| unnested(&vars[7 * g..7 * g + 7], g as u32 + 1))
        .collect();
    let names = vars.join(",");
    agrees(
        "three generators",
        &format!("{names}\n0\n{generators}\n", generators = three.join(",\n")),
        "grevlex",
    );
}

/// Test 8: a zero-dimensional system over ℚ with fractions, in lex, whose
/// record FGLM changes, and in `block:1`.
#[test]
fn lex_and_block() {
    let instance = "x,y,z\n0\n\
                    x^2+2/3*y*z-1,\n\
                    y^2-5/7*x+z,\n\
                    z^2+1/11*x*y-3\n";
    for order in ["lex", "block:1"] {
        agrees("three quadrics", instance, order);
    }
}
