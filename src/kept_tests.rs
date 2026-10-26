//! q-replay-time, sub-item 2: the lift's basis where a term of an element
//! vanishes at a forced prime, and the lift's bases of the edge cases
//! (plan.md, _Test code_).

use std::num::NonZeroUsize;

use crate::lift::{Report, lift_text};
use crate::{MonomialOrder, Outcome, Params, Threads};

/// A prime near 2^30.5 that the case below makes a coefficient vanish at,
/// as in `lift_tests.rs`.
const U1: u64 = 1_518_500_279;
/// Primes near 2^31, lucky for every case here.
const L1: u64 = 2_147_483_647;
const L2: u64 = 2_147_483_629;
const L3: u64 = 2_147_483_587;

fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

/// The lift of `text` with the primes `forced` first: its outcome, which
/// must hold a basis.
fn lift(text: &str, params: &Params, forced: &[u64]) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text(text, params, forced.to_vec(), &mut report)
        .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    outcome
}

fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// q-replay-time.4 to .6's case: the first step makes y − U1·z over Q and
/// y modulo U1. The lift with U1 forced between lucky primes returns the
/// basis over Q. No result shows that the lift checked the shape and refit
/// it.
#[test]
fn the_lift_where_a_term_vanishes() {
    let a = U1 + 1;
    let text = format!("x,y,z\n0\nx^2+{a}*z,\nx^2+y+z\n");
    let outcome = lift(&text, &params(1), &[L1, L2, U1, L3]);
    assert_eq!(basis(&outcome), format!("[y-{U1}*z,\nx^2+{a}*z]:\n"));
}

/// `OBJECTIVES.md`, _Domain specifics the workflow points at_, through the lift: the bases of
/// `tests/suite/q_edge.rs` in grevlex and of a constant input.
#[test]
fn the_edge_cases_through_the_lift() {
    let cases = [
        ("the zero ideal", "x,y\n0\n0\n", "[]:\n"),
        ("F = (0), two zeros", "x,y\n0\n0,\n0\n", "[]:\n"),
        ("no generator", "x,y\n0\n", "[]:\n"),
        ("the unit ideal", "x,y\n0\nx*y-1/2,\nx,\ny+3\n", "[1]:\n"),
        ("a constant input", "x,y\n0\nx+y,\n7\n", "[1]:\n"),
        (
            "a Gröbner basis already",
            "x,y\n0\n3*x+1,\n2*y^2-5\n",
            "[3*x+1,\n2*y^2-5]:\n",
        ),
        (
            "a zero generator",
            "x,y\n0\nx^2+y,\n0,\nx*y+1\n",
            "[y^2-x,\nx*y+1,\nx^2+y]:\n",
        ),
        (
            "an eliminable variable",
            "x,y\n0\nx-1/3*y^2,\ny^3-2\n",
            "[y^2-3*x,\n3*x*y-2,\n9*x^2-2*y]:\n",
        ),
    ];
    for (case, text, expected) in cases {
        let outcome = lift(text, &params(1), &[]);
        assert_eq!(basis(&outcome), expected, "{case}");
    }
}
