//! Accept, Reject, and Agree over the small random systems of
//! `random_gfp.rs` and `random_q.rs`, ten per prime and per range of digits,
//! each in three orders besides grevlex: lex; `block:k` with k drawn from 1
//! to n − 1 (checker-orders.14); and an order whose matrix is grevlex's,
//! lex's, or `block:k`'s with its columns permuted (checker-orders.16).

use std::ops::RangeInclusive;

use crate::common::{Rng, random_system, random_system_q};
use crate::orders::{
    Jl, accept_and_reject_in, agree_full_in, permutation, solved_in, solved_permuted, variables,
};

/// Accept, Reject, and Agree on `instance` in lex, in a block order, and in
/// a permuted order, each drawn from `seed`.
fn in_three_orders(name: &str, instance: &str, seed: u64) {
    let n = variables(instance);
    let mut rng = Rng::new(seed ^ 0x0de5);
    // checker-orders.14 design -- the Accept cases take lex and block:k: k drawn from 1 to n − 1 on random systems, k = 1 on the edge cases (not measured)
    let k = 1 + rng.index(n - 1);
    for jl in [Jl::Lex, Jl::Block(k)] {
        let (case, order) = solved_in(name, instance, jl);
        accept_and_reject_in(&case, &order, seed);
        agree_full_in(&case, &order, seed);
    }
    let jl = [Jl::Grevlex, Jl::Lex, Jl::Block(k)][rng.index(3)];
    // checker-orders.16 design -- matrix orders beyond grevlex, lex, and block:k are tested on Groebner.jl's results for the instance with its variable line permuted (not measured)
    let perm = permutation(n, &mut rng);
    let (case, order) = solved_permuted(&format!("{name}-permuted"), instance, jl, &perm);
    accept_and_reject_in(&case, &order, seed);
    agree_full_in(&case, &order, seed);
}

fn random(p: u64) {
    for seed in 1..=10 {
        let instance = random_system(p, seed);
        in_three_orders(&format!("random-p{p}-{seed}"), &instance, seed);
    }
}

fn random_q(digits: RangeInclusive<usize>, tag: &str) {
    for seed in 1..=10 {
        let instance = random_system_q(digits.clone(), seed);
        in_three_orders(&format!("random-q-{tag}-{seed}"), &instance, seed);
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
