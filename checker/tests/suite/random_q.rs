//! Accept and Reject over small random systems over Q (`OBJECTIVES.md`,
//! _Checker_): ten systems with coefficients of 1 to 3 digits, and ten with
//! coefficients of 20 to 25 digits, nearly all past 2^64, each solved by
//! Groebner.jl.

use std::ops::RangeInclusive;

use crate::common::{Case, accept_and_reject, random_system_q};

fn random(digits: RangeInclusive<usize>, tag: &str) {
    for seed in 1..=10 {
        let instance = random_system_q(digits.clone(), seed);
        let case = Case::solved(&format!("random-q-{tag}-{seed}"), &instance);
        accept_and_reject(&case, seed);
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn small_coefficients() {
    random(1..=3, "small");
}

#[test]
#[ignore = "calls groebner-jl"]
fn coefficients_past_2_64() {
    random(20..=25, "big");
}
