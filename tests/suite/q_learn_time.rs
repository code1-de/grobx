//! q-learn-time, the item's test code: the edge cases over Q in three
//! orders; orders and
//! the checker over Q, texts whose pair learn may disagree among them; and
//! a bound that stops a call over Q (plan.md, _Test code_, 8 to 10).

use std::time::Duration;

use grobx::{CheckerMode, Stop};

use crate::common::{
    EDGE_GENERATORS, FAR_DEADLINE, bounded_in, dense, params_in, random_square, run,
    with_cofactors_in,
};
use crate::q::{accepted_q, assert_canonical_q};

/// Test 8, q-learn-time.10: the edge cases over Q in grevlex, lex, and
/// `block:1`: each basis canonical and accepted by the checker in the
/// exact mode, with cofactors, and the call without cofactors returns the
/// same basis.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE_GENERATORS {
        let text = format!("x,y,z\n0\n{generators}\n");
        for order in ["grevlex", "lex", "block:1"] {
            let checked = accepted_q(case, &text, order);
            let plain = run(&text, &bounded_in(order, 1));
            assert_eq!(plain.basis, checked.basis, "{case} in {order}");
        }
    }
}

/// Test 9, q-learn-time.10: generated systems over Q in lex and `block:2`,
/// which the checker certifies in the exact mode inside the call; and, in
/// three orders, the texts whose coefficients the primes
/// `learn_pair_tests.rs` forces divide (q-learn-time.4, .6), which a drawn
/// pair meets only by chance, accepted by the checker in the exact mode
/// outside the call.
#[test]
fn orders_and_the_checker() {
    for (name, text, order) in [
        ("cubic-2", dense(2, 3, 0, 1), "lex"),
        ("square-3", random_square(0, 3), "lex"),
        ("square-3", random_square(0, 3), "block:2"),
        ("square-2", random_square(0, 2), "block:2"),
    ] {
        let mut checked = with_cofactors_in(order, 3);
        checked.checker = Some(CheckerMode::Exact);
        checked.deadline = Some(FAR_DEADLINE);
        let outcome = run(&text, &checked);
        let basis = outcome.basis.as_deref().unwrap();
        assert_canonical_q(name, &text, basis, order);
        assert_eq!(
            outcome.certified,
            Some(true),
            "{name} in {order}: {flaw:?}",
            flaw = outcome.flaw
        );
    }
    let (l1, l2) = (2_147_483_647_u64, 2_147_483_629_u64);
    for (case, text) in [
        (
            "a leading coefficient",
            format!("x,y,z\n0\n{l2}*x^2+y,\nx*y-1\n"),
        ),
        (
            "a row zero at one prime",
            format!("x,y,z\n0\nx*y-1,\nx*y+{l1}*z-1\n"),
        ),
        ("an FGLM pivot", format!("x,y\n0\ny^2-{l2}*x,\nx^2-1\n")),
        ("a denominator", format!("x,y\n0\nx-1/{l2}*y,\ny^2-x\n")),
    ] {
        for order in ["grevlex", "lex", "block:1"] {
            accepted_q(&format!("{case} in {order}"), &text, order);
        }
    }
}

/// Test 10: a deadline of 1 ms stops a call over Q, with no basis.
#[test]
fn a_bound_stops_a_call_over_q() {
    let mut p = params_in("grevlex", 1);
    p.deadline = Some(Duration::from_millis(1));
    let outcome = grobx::groebner(&dense(5, 3, 0, 1), &p).unwrap();
    assert!(outcome.basis.is_none(), "{outcome:?}");
    assert_eq!(outcome.stopped, Some(Stop::Deadline));
}
