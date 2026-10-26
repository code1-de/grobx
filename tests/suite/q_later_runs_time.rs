//! q-later-runs-time, the item's test code: the edge cases over Q through
//! the call, whose replays run in pairs; and random systems over Q in lex
//! and `block:1`, each call without cofactors against the call with
//! cofactors that the checker accepts (plan.md, _Test code_, 8 and 10).

use std::time::Duration;

use grobx::{CheckerMode, Stop};

use crate::common::{params_in, run, with_cofactors_in};
use crate::q::{assert_canonical_q, finished_q, random_q_system};

const ORDERS: [&str; 3] = ["grevlex", "lex", "block:1"];

/// The deadline of every call of test 8, far past what each takes: a fault
/// that keeps the lift from completing stops the call there and fails the
/// test. A fault that makes the replays misfit ends it in `Error::Drops`
/// before.
const DEADLINE: Duration = Duration::from_secs(60);

/// `params_in` with the deadline.
fn bounded_in(order: &str, seed: u64) -> grobx::Params {
    let mut p = params_in(order, seed);
    p.deadline = Some(DEADLINE);
    p
}

/// `OBJECTIVES.md`, _Domain specifics the workflow points at_, over Q: each case, its text, and
/// its basis in grevlex, as `q_edge.rs` and `kept_tests.rs` write them.
fn edge_cases() -> [(&'static str, &'static str, &'static str); 8] {
    [
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
    ]
}

/// Test 8, q-later-runs-time.1, .6: in every order the call without
/// cofactors, whose replays run in pairs, and alone in lex where FGLM
/// changes the record, returns a canonical basis, in
/// grevlex the one written; the call with cofactors, whose replays run
/// alone, returns the same, and the checker accepts it in the exact mode,
/// outside the call and inside it.
#[test]
fn the_edge_cases_in_pairs() {
    for (case, text, grevlex) in edge_cases() {
        for name in ORDERS {
            let paired = run(text, &bounded_in(name, 1));
            let basis = paired.basis.as_deref().unwrap();
            assert_canonical_q(case, text, basis, name);
            if name == "grevlex" {
                assert_eq!(basis, grevlex, "{case}");
            }
            let alone = finished_q(case, text, name, DEADLINE)
                .unwrap_or_else(|| panic!("{case} in {name}: stopped by the deadline"));
            assert_eq!(alone.basis.as_deref(), Some(basis), "{case} in {name}");
            let mut checked = with_cofactors_in(name, 2);
            checked.checker = Some(CheckerMode::Exact);
            checked.deadline = Some(DEADLINE);
            let outcome = run(text, &checked);
            assert_eq!(outcome.basis.as_deref(), Some(basis), "{case} in {name}");
            assert_eq!(
                outcome.certified,
                Some(true),
                "{case} in {name}: {flaw:?}",
                flaw = outcome.flaw
            );
        }
    }
}

/// Test 10, q-later-runs-time.6: `q_random.rs`'s random systems over Q
/// in lex, whose records FGLM changes where the ideal is zero-dimensional
/// and whose replays then run alone, and in `block:1`, whose replays run in
/// pairs: the call without cofactors returns the basis of the call with
/// cofactors, which the checker accepts in the exact mode, 9 in 10 within
/// 10 s.
#[test]
fn random_systems_in_lex_and_a_block_order() {
    const RANDOM_DEADLINE: Duration = Duration::from_secs(10);
    let (mut finished, mut run) = (0, 0);
    for seed in 0..5 {
        let text = random_q_system(seed);
        for order in ["lex", "block:1"] {
            run += 1;
            let case = format!("q-random-{seed}");
            let mut p = params_in(order, 0x0de7);
            p.deadline = Some(RANDOM_DEADLINE);
            let outcome = grobx::groebner(&text, &p)
                .unwrap_or_else(|e| panic!("{case} in {order}: {e:?}\n{text}"));
            let Some(basis) = outcome.basis.as_deref() else {
                assert_eq!(outcome.stopped, Some(Stop::Deadline), "{case} in {order}");
                continue;
            };
            let Some(checked) = finished_q(&case, &text, order, RANDOM_DEADLINE) else {
                continue;
            };
            finished += 1;
            assert_eq!(checked.basis.as_deref(), Some(basis), "{case} in {order}");
        }
    }
    assert!(10 * finished >= 9 * run, "{finished} of {run} finished");
}
