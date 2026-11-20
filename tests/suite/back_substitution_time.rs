//! back-substitution-time, the item's integration tests: calls whose
//! back-substitutions and final inter-reductions take bands, and calls
//! that take none, return the bases of the calls with cofactors, which
//! reduce one row at a time; and a deadline stops a long call
//! (`iterations/0091-back-substitution-time/plan.md`, _Test code_, 5 to 8).

use std::time::{Duration, Instant};

use grobx::Stop;
use grobx_checker::Mode;

use crate::common::{
    EDGE_GENERATORS, P, P31, assert_accepted_in, dense, params, params_in, random_system, run,
    with_cofactors_in,
};
use crate::large::LEAST;
use crate::orders::{Family, assert_canonical, names};
use crate::q::accepted_q;

/// Over GF(p): the call in the order `order` without cofactors, signatures
/// as `signatures` says, returns the basis text of the call with them,
/// which is canonical in the order and which the exact checker accepts in
/// it.
fn same_in(case: &str, instance: &str, order: &str, signatures: bool) {
    let name = format!("{case} in {order}, signatures {signatures}");
    let mut tracked = with_cofactors_in(order, 0x0b5b);
    tracked.signatures = signatures;
    let with = run(instance, &tracked);
    let basis = with.basis.as_deref().unwrap();
    assert_canonical(&name, instance, basis, order);
    let checker_order = Family::named(order).checker(names(instance).len());
    assert_accepted_in(&name, instance, &with, &checker_order, Mode::Exact);
    let mut plain = params_in(order, 0x0b5b);
    plain.signatures = signatures;
    let without = run(instance, &plain);
    assert_eq!(without.basis, with.basis, "{name}\n{instance}");
}

/// Test 5: the edge cases over 65521, 2^31 − 1, and ℚ in grevlex,
/// signatures off and on.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE_GENERATORS {
        for p in [P, P31] {
            for signatures in [false, true] {
                same_in(
                    &format!("{case} over {p}"),
                    &format!("x,y,z\n{p}\n{generators}\n"),
                    "grevlex",
                    signatures,
                );
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        let accepted = accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
        for signatures in [false, true] {
            let mut plain = params(0x0b5c);
            plain.signatures = signatures;
            assert_eq!(
                run(&text, &plain).basis,
                accepted.basis,
                "{case} over ℚ, signatures {signatures}"
            );
        }
    }
}

/// Test 6: back-substitution-time.1 to .9. Dense cubic systems on four
/// variables over 65521 and 2^31 − 1 in grevlex, lex, and `block:1`,
/// signatures off and on. Their rounds back-substitute their new rows in
/// bands, and their inter-reductions take bands and rows as they stand
/// (log 0091-1, _Step 5_). `block:2` takes such a system past a minute.
#[test]
fn dense_systems_in_three_orders() {
    for p in [P, P31] {
        for seed in 1..=3 {
            let text = dense(4, 3, p, seed);
            for order in ["grevlex", "lex", "block:1"] {
                for signatures in [false, true] {
                    same_in(
                        &format!("dense 4, seed {seed}, p {p}"),
                        &text,
                        order,
                        signatures,
                    );
                }
            }
        }
    }
}

/// Test 7: back-substitution-time.8 over ℚ, and a field without bands.
/// Random systems over ℚ, whose learns and replays take no band and whose
/// inter-reductions take their reduced rows as they stand, return the
/// basis `accepted_q` accepts; and over 2^31 + 11 the call without
/// cofactors returns the basis of the call with them.
#[test]
fn over_q_and_past_2_31() {
    for seed in 1..=12 {
        let text = random_system(0, seed);
        let accepted = accepted_q(&format!("seed {seed} over ℚ"), &text, "grevlex");
        assert_eq!(
            run(&text, &params(0x0b5d)).basis,
            accepted.basis,
            "seed {seed} over ℚ"
        );
        same_in(
            &format!("seed {seed} over {LEAST}"),
            &random_system(LEAST, seed),
            "grevlex",
            false,
        );
    }
    for seed in 1..=2 {
        let text = dense(4, 3, LEAST, seed);
        same_in(
            &format!("dense 4, seed {seed}, p {LEAST}"),
            &text,
            "grevlex",
            false,
        );
    }
}

/// Test 8: a dense cubic system on eight variables over 65521, whose plain
/// run back-substitutes in bands, under a deadline of 50 ms: the call stops
/// with `Deadline`, no basis, and returns within 0.5 s of its start.
#[test]
fn a_deadline_stops_a_long_call() {
    const DEADLINE: Duration = Duration::from_millis(50);
    let text = dense(8, 3, P, 2);
    let mut p = params(2);
    p.signatures = false;
    p.deadline = Some(DEADLINE);
    let start = Instant::now();
    let outcome = grobx::groebner(&text, &p).expect("the call runs");
    let took = start.elapsed();
    assert_eq!(outcome.stopped, Some(Stop::Deadline), "cubic-8");
    assert_eq!(outcome.basis, None, "cubic-8");
    assert!(
        took < Duration::from_millis(500),
        "cubic-8: returned after {took:?}"
    );
}
