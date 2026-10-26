//! Agree over the small random systems of `random_gfp.rs` and `random_q.rs`
//! (`OBJECTIVES.md`, _Checker_): on Groebner.jl's result and on every mutant
//! Reject draws for it, the cheap mode reaches the exact mode's verdict, of
//! the whole check and of the rows stage alone.

use std::ops::RangeInclusive;

use crate::agree::agree_full;
use crate::common::{Case, random_system, random_system_q};

fn random(p: u64) {
    for seed in 1..=10 {
        let instance = random_system(p, seed);
        let case = Case::solved(&format!("random-p{p}-{seed}"), &instance);
        agree_full(&case, seed);
    }
}

fn random_q(digits: RangeInclusive<usize>, tag: &str) {
    for seed in 1..=10 {
        let instance = random_system_q(digits.clone(), seed);
        let case = Case::solved(&format!("random-q-{tag}-{seed}"), &instance);
        agree_full(&case, seed);
    }
}

/// GF(2): every row of degree 2 or more is checked exactly (checker-cheap.3).
#[test]
#[ignore = "calls groebner-jl"]
fn p2() {
    random(2);
}

#[test]
#[ignore = "calls groebner-jl"]
fn p3() {
    random(3);
}

#[test]
#[ignore = "calls groebner-jl"]
fn p7() {
    random(7);
}

#[test]
#[ignore = "calls groebner-jl"]
fn p65521() {
    random(65521);
}

/// 2^32 − 5.
#[test]
#[ignore = "calls groebner-jl"]
fn p4294967291() {
    random(4_294_967_291);
}

/// 2^63 − 25.
#[test]
#[ignore = "calls groebner-jl"]
fn p9223372036854775783() {
    random(9_223_372_036_854_775_783);
}

#[test]
#[ignore = "calls groebner-jl"]
fn q_small_coefficients() {
    random_q(1..=3, "small");
}

#[test]
#[ignore = "calls groebner-jl"]
fn q_coefficients_past_2_64() {
    random_q(20..=25, "big");
}
