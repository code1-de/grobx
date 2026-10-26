//! q-lift, the item's test code: the edge cases over Q (plan.md, _Edge
//! cases_), in grevlex, lex, and `block:1`, with cofactors that the
//! checker accepts in the exact mode, the bases written by hand, msolve's
//! in grevlex, and Groebner.jl's in lex and `block:1` (log 0016-1, _Step
//! 5_). `like-terms` in grevlex was checked against Groebner.jl alone.

use std::time::Duration;

use grobx::Stop;

use crate::common::{self, params_in};
use crate::q::{accepted_q, finished_q};

/// The product of two primes near 2^30.5 (pauer-1992 §4, example 4).
const N: &str = "2305843118582081747";

/// A case: its name, its text over Q, and its basis in grevlex, lex, and
/// `block:1`; None for an order the case is not run in.
struct Case {
    name: &'static str,
    text: String,
    bases: [Option<&'static str>; 3],
}

const ORDERS: [&str; 3] = ["grevlex", "lex", "block:1"];

fn case(name: &'static str, text: &str, bases: [Option<&'static str>; 3]) -> Case {
    Case {
        name,
        text: text.to_string(),
        bases,
    }
}

fn cases() -> Vec<Case> {
    let empty = Some("[]:\n");
    let one = Some("[1]:\n");
    let pauer: &'static str = Box::leak(format!("[y,\nx+{N}]:\n").into_boxed_str());
    vec![
        // OBJECTIVES.md, Domain specifics.
        case("zero", "x,y\n0\n0\n", [empty; 3]),
        case("zeros", "x,y\n0\n0,\n0\n", [empty; 3]),
        case("none", "x,y\n0\n", [empty; 3]),
        case("unit", "x,y\n0\nx*y-1/2,\nx,\ny+3\n", [one; 3]),
        // arnold-2003, Example 8.3: four generators of the unit ideal.
        case(
            "arnold-8.3",
            "x,y,z\n0\n5*x^3*y^2*z+3*y^3*x^2*z+7*x*y^2*z^2,\n3*x*y^2*z^2+x^5+11*y^2*z^2,\n4*x*y*z+7*x^3+12*y^3+1,\n3*x^3-4*y^3+y*z^2\n",
            [one, None, None],
        ),
        case(
            "basis-already",
            "x,y\n0\n3*x+1,\n2*y^2-5\n",
            [
                Some("[3*x+1,\n2*y^2-5]:\n"),
                Some("[2*y^2-5,\n3*x+1]:\n"),
                Some("[2*y^2-5,\n3*x+1]:\n"),
            ],
        ),
        // arnold-2003, Examples 8.1 and 8.2; the bases msolve's.
        case(
            "arnold-8.1",
            "x,y,z\n0\n8*x^2*y^2+5*x*y^3+3*x^3*z+x^2*y*z,\nx^5+2*y^3*z^2+13*y^2*z^3+5*y*z^4,\n8*x^3+12*y^3+x*z^2+3,\n7*x^2*y^4+18*x*y^3*z^2+y^3*z^3\n",
            [Some("[x,\nz^2,\n4*y^3+1]:\n"), None, None],
        ),
        case(
            "arnold-8.2",
            "x,y,z\n0\n2*x*y^4*z^2+x^3*y^2*z-x^2*y^3*z+2*x*y*z^2+7*y^3+7,\n2*x^2*y^4*z+x^2*y*z^2-x*y^2*z^2+2*x^2*y*z-12*x+12*y,\n2*y^5*z+x^2*y^2*z-x*y^3*z-x*y^3+y^4+2*y^2*z,\n3*x*y^4*z^3+x^2*y^2*z-x*y^3*z+4*y^3*z^2+3*x*y*z^3+4*z^2-x+y\n",
            [Some("[x-y,\ny^3+1]:\n"), None, None],
        ),
        case(
            "zero-generator",
            "x,y\n0\nx^2+y,\n0,\nx*y+1\n",
            [
                Some("[y^2-x,\nx*y+1,\nx^2+y]:\n"),
                Some("[y^3+1,\nx-y^2]:\n"),
                Some("[y^3+1,\nx-y^2]:\n"),
            ],
        ),
        case(
            "eliminable",
            "x,y\n0\nx-1/3*y^2,\ny^3-2\n",
            [
                Some("[y^2-3*x,\n3*x*y-2,\n9*x^2-2*y]:\n"),
                Some("[y^3-2,\n3*x-y^2]:\n"),
                Some("[y^3-2,\n3*x-y^2]:\n"),
            ],
        ),
        // The item's own.
        case(
            "fractions",
            "x,y\n0\n2*x^2+3*y-1/2,\nx*y+5/3\n",
            [
                Some("[18*y^2-20*x-3*y,\n3*x*y+5,\n4*x^2+6*y-1]:\n"),
                Some("[54*y^3-9*y^2+100,\n20*x-18*y^2+3*y]:\n"),
                Some("[54*y^3-9*y^2+100,\n20*x-18*y^2+3*y]:\n"),
            ],
        ),
        case(
            "like-terms",
            "x,y\n0\n1/2*x+1/3*x-y,\ny^2-1\n",
            [
                Some("[5*x-6*y,\ny^2-1]:\n"),
                Some("[y^2-1,\n5*x-6*y]:\n"),
                Some("[y^2-1,\n5*x-6*y]:\n"),
            ],
        ),
        case(
            "lowest-terms",
            "x\n0\n2/4*x-3/6\n",
            [Some("[x-1]:\n"), Some("[x-1]:\n"), None],
        ),
        case(
            "negative-lead",
            "x,y\n0\n-2*x+y,\n-y^2+1\n",
            [
                Some("[2*x-y,\ny^2-1]:\n"),
                Some("[y^2-1,\n2*x-y]:\n"),
                Some("[y^2-1,\n2*x-y]:\n"),
            ],
        ),
        case(
            "huge",
            "x,y\n0\nx-123456789012345678901234567890*y,\ny^2-98765432109876543210\n",
            [
                Some("[x-123456789012345678901234567890*y,\ny^2-98765432109876543210]:\n"),
                Some("[y^2-98765432109876543210,\nx-123456789012345678901234567890*y]:\n"),
                Some("[y^2-98765432109876543210,\nx-123456789012345678901234567890*y]:\n"),
            ],
        ),
        case(
            "leading-zeros",
            "x\n000\nx-1/2\n",
            [Some("[2*x-1]:\n"), Some("[2*x-1]:\n"), None],
        ),
        case(
            "pauer-4",
            &format!("x,y\n0\nx+{N},\nx*y\n"),
            [Some(pauer), Some(pauer), Some(pauer)],
        ),
    ]
}

// q-lift.43 design -- the edge cases in lex and block:1 run with a deadline of 10 s, and, as order-family.19's cases in an order that is not graded, a call the deadline stops passes where 9 in 10 finish, not every case in every order asserted to finish (log 0016-4)
/// Every case in every order it takes: with cofactors, canonical, and
/// accepted by the checker in the exact mode, the basis as written; and
/// the call with signatures off and no cofactors gives the same basis. In
/// grevlex every call finishes. The arnold cases run in grevlex alone. In
/// lex and `block:1` each call has a deadline of 10 s; a call it stops
/// passes, and 9 in 10 calls finish.
#[test]
fn edge_cases_over_q() {
    let deadline = Duration::from_secs(10);
    let (mut calls, mut finished) = (0, 0);
    for c in cases() {
        for (order, want) in ORDERS.iter().zip(c.bases) {
            let Some(want) = want else { continue };
            let mut plain = params_in(order, 0x0de7);
            plain.signatures = false;
            let outcomes = if *order == "grevlex" {
                vec![
                    Some(accepted_q(c.name, &c.text, order)),
                    Some(common::run(&c.text, &plain)),
                ]
            } else {
                plain.deadline = Some(deadline);
                let other = grobx::groebner(&c.text, &plain)
                    .unwrap_or_else(|e| panic!("{name} in {order}: {e:?}", name = c.name));
                if other.basis.is_none() {
                    assert_eq!(
                        other.stopped,
                        Some(Stop::Deadline),
                        "{name} in {order}",
                        name = c.name
                    );
                    eprintln!(
                        "{name} in {order}, signatures off: stopped by the deadline",
                        name = c.name
                    );
                }
                vec![
                    finished_q(c.name, &c.text, order, deadline),
                    other.basis.is_some().then_some(other),
                ]
            };
            for (k, outcome) in outcomes.into_iter().enumerate() {
                calls += 1;
                let Some(outcome) = outcome else { continue };
                finished += 1;
                let plain = if k == 1 { ", signatures off" } else { "" };
                assert_eq!(
                    outcome.basis.as_deref(),
                    Some(want),
                    "{name} in {order}{plain}",
                    name = c.name
                );
                if k == 1 {
                    assert_eq!(
                        outcome.cofactors,
                        None,
                        "{name} in {order}{plain}",
                        name = c.name
                    );
                }
            }
        }
    }
    assert!(10 * finished >= 9 * calls, "{finished} of {calls} finished");
}
