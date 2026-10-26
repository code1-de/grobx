//! Accept and Reject over small random systems over GF(p)
//! (`OBJECTIVES.md`, _Checker_): ten systems per prime, each solved by
//! Groebner.jl, from 2 up to the largest prime below 2^63, the largest
//! Groebner.jl takes.

use crate::common::{Case, accept_and_reject, random_system};

fn random(p: u64) {
    for seed in 1..=10 {
        let instance = random_system(p, seed);
        let case = Case::solved(&format!("random-p{p}-{seed}"), &instance);
        accept_and_reject(&case, seed);
    }
}

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
