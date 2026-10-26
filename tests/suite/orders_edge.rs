//! order-family, the item's test code: the edge cases in lex and in block
//! orders, hand-computed bases, one variable, the signatures switch, the
//! checker parameter, and the bounds (plan.md, _Edge cases_). The overflow
//! in lex is tested in `formats.rs` (monomial-formats.7).
//! Every basis computed with cofactors is canonical in its order and
//! accepted by the checker in it.

use std::num::NonZeroU32;
use std::time::Duration;

use grobx::{Cancel, CheckerMode, Stop};
use grobx_checker::Mode;

use crate::common::{dense, params_in, random_square, random_system, run};
use crate::orders::{accepted_in, lex_as_blocks, names};

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, on two or more
/// variables: name, instance, and the basis text in lex and in `block:1`
/// where it is written here.
const EDGE: &[(&str, &str, Option<&str>)] = &[
    ("zero", "x,y\n65521\n0\n", Some("[]:\n")),
    ("zeros", "x,y\n7\n0,\n0\n", Some("[]:\n")),
    ("none", "x,y\n65521\n", Some("[]:\n")),
    ("x-plus-x", "x,y\n2\nx+x\n", Some("[]:\n")),
    ("unit", "x,y\n65521\nx*y-1,\nx\n", Some("[1]:\n")),
    ("unit-p2", "x,y\n2\nx,\nx+1\n", Some("[1]:\n")),
    ("unit-constant", "x,y\n65521\n5\n", Some("[1]:\n")),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
        Some("[1]:\n"),
    ),
    ("gb", "x,y\n65521\nx+y,\ny^2+1\n", Some("[y^2+1,\nx+y]:\n")),
    (
        "gb-swapped",
        "x,y\n65521\ny^2+1,\nx+y\n",
        Some("[y^2+1,\nx+y]:\n"),
    ),
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n", None),
    (
        "eliminable-first",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        None,
    ),
    (
        "eliminable-last",
        "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        None,
    ),
    (
        "eliminable-linear",
        "x,y,z\n65521\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n",
        None,
    ),
];

/// Every edge case in lex, in `block:1`, and on three variables in
/// `block:1,1`, accepted in the exact mode. A basis written above is the
/// text. Lex is `block:1,…,1`.
#[test]
fn the_edge_cases() {
    for &(name, text, want) in EDGE {
        let n = names(text).len();
        let mut orders = vec!["lex", "block:1"];
        if n == 3 {
            orders.push("block:1,1");
        }
        for order in orders {
            let outcome = accepted_in(name, text, order, Mode::Exact);
            let basis = outcome.basis.unwrap();
            if let Some(want) = want {
                assert_eq!(basis, want, "{name} in {order}");
            }
            if order == "lex" {
                let blocks = run(text, &params_in(&lex_as_blocks(n), 1)).basis.unwrap();
                assert_eq!(
                    basis,
                    blocks,
                    "{name}: lex and {as_blocks}",
                    as_blocks = lex_as_blocks(n)
                );
            }
        }
    }
}

/// The zero generator changes nothing, in lex and in `block:1`.
#[test]
fn a_zero_generator_changes_nothing() {
    for order in ["lex", "block:1"] {
        let with = run("x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n", &params_in(order, 1));
        let without = run("x,y,z\n65521\nx*y-z,\ny^2-x\n", &params_in(order, 1));
        assert_eq!(with.basis, without.basis, "{order}");
    }
}

/// A reduced lex basis given as an instance is its own reduced lex basis.
#[test]
fn a_lex_basis_is_its_own() {
    let text = dense(2, 3, 65521, 1);
    let lex = accepted_in("cubic-2", &text, "lex", Mode::Exact)
        .basis
        .unwrap();
    let elements = lex
        .trim_end()
        .trim_start_matches('[')
        .trim_end_matches("]:");
    let line = text.lines().next().unwrap();
    let again = format!("{line}\n65521\n{elements}\n");
    assert_eq!(run(&again, &params_in("lex", 1)).basis.unwrap(), lex);
}

/// Bases computed by hand.
#[test]
fn hand_computed_bases() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "x,y\n65521\nx-y^2,\ny^3-1\n",
            "lex",
            "[y^3+65520,\nx+65520*y^2]:\n",
        ),
        (
            "x,y\n65521\nx-y^2,\ny^3-1\n",
            "block:1",
            "[y^3+65520,\nx+65520*y^2]:\n",
        ),
        (
            "x,y\n65521\nx-y^2,\ny^3-1\n",
            "grevlex",
            "[y^2+65520*x,\nx*y+65520,\nx^2+65520*y]:\n",
        ),
        (
            "x,y\n65521\nx^2+y,\nx*y+1\n",
            "lex",
            "[y^3+1,\nx+65520*y^2]:\n",
        ),
        (
            "x,y\n65521\nx^2+y,\nx*y+1\n",
            "block:1",
            "[y^3+1,\nx+65520*y^2]:\n",
        ),
        (
            "x,y,z\n65521\nx-1,\ny-z^2\n",
            "lex",
            "[y+65520*z^2,\nx+65520]:\n",
        ),
        (
            "x,y,z\n65521\nx-1,\ny-z^2\n",
            "block:1,1",
            "[y+65520*z^2,\nx+65520]:\n",
        ),
        (
            "x,y,z\n65521\nx-1,\ny-z^2\n",
            "block:1",
            "[z^2+65520*y,\nx+65520]:\n",
        ),
        (
            "x,y,z\n65521\nx-1,\ny-z^2\n",
            "block:2",
            "[y+65520*z^2,\nx+65520]:\n",
        ),
        (
            "x,y,z\n65521\nx-1,\ny-z^2\n",
            "grevlex",
            "[x+65520,\nz^2+65520*y]:\n",
        ),
    ];
    for &(text, order, want) in cases {
        let outcome = accepted_in(order, text, order, Mode::Exact);
        assert_eq!(outcome.basis.unwrap(), want, "{order} on {text:?}");
    }
}

/// On one variable lex is grevlex.
#[test]
fn one_variable() {
    for text in ["x\n65521\nx^3-1,\nx^2-1\n", "x\n7\nx^4-1,\nx^6-1\n"] {
        let lex = accepted_in("one", text, "lex", Mode::Exact);
        assert_eq!(
            lex.basis,
            run(text, &params_in("grevlex", 1)).basis,
            "{text:?}"
        );
    }
}

/// The instances and orders the switches run on: name, instance, and
/// order. The system on x, y, z, w is not zero-dimensional.
fn small() -> Vec<(&'static str, String, &'static str)> {
    let positive = "x,y,z,w\n65521\nx*y-z,\ny*z-w,\nx^2-w\n";
    vec![
        ("cubic-3", dense(3, 3, 65521, 1), "lex"),
        // test-mq-systems.12 design -- a call that F4 runs in block:2 takes the random square system of degrees 2, 3, and 2, where the map's dense cubic one on three variables runs for seconds (log 0084-3)
        ("square-27", random_square(65521, 27), "block:2"),
        ("positive", positive.to_string(), "lex"),
        ("positive", positive.to_string(), "block:2"),
        ("square", random_square(65521, 1), "block:1"),
    ]
}

/// Outside grevlex a call with signatures returns the basis of the call
/// without (order-family.14, fglm-lex.2).
#[test]
fn signatures_do_not_matter_outside_grevlex() {
    let mut cases = small();
    for seed in 0..5 {
        let text = random_system(65521, seed);
        cases.push(("random", text.clone(), "lex"));
        cases.push(("random", text, "block:1"));
    }
    for (name, text, order) in cases {
        let mut on = params_in(order, 3);
        on.signatures = true;
        let mut off = params_in(order, 3);
        off.signatures = false;
        let (on, off) = (run(&text, &on), run(&text, &off));
        assert_eq!(on.basis, off.basis, "{name} in {order}\n{text}");
    }
}

/// The checker parameter rules in the call's order.
#[test]
fn the_checker_parameter() {
    for (stem, text, order) in small() {
        for mode in [
            CheckerMode::Exact,
            CheckerMode::Cheap {
                points: NonZeroU32::new(6).unwrap(),
            },
        ] {
            let mut p = params_in(order, 5);
            p.checker = Some(mode);
            let outcome = run(&text, &p);
            assert_eq!(
                outcome.certified,
                Some(true),
                "{stem} in {order}, {mode:?}: {flaw:?}",
                flaw = outcome.flaw
            );
            assert_eq!(outcome.flaw, None);
            assert_eq!(outcome.cofactors, None);
        }
    }
}

/// The cofactors the call returns in an order are the ones the checker
/// accepts in it, and a call with them returns the basis of one without.
#[test]
fn cofactors_in_an_order() {
    for (stem, text, order) in small() {
        let with = accepted_in(stem, &text, order, Mode::Exact);
        assert!(with.cofactors.is_some(), "{stem} in {order}");
        assert_eq!(
            with.basis,
            run(&text, &params_in(order, 9)).basis,
            "{stem} in {order}"
        );
    }
}

/// Every bound stops a lex run and a block run.
#[test]
fn every_bound_stops_the_run() {
    let text = dense(8, 3, 65521, 1);
    for order in ["lex", "block:4"] {
        let mut p = params_in(order, 1);
        p.deadline = Some(Duration::ZERO);
        let outcome = grobx::groebner(&text, &p).unwrap();
        assert_eq!(outcome.stopped, Some(Stop::Deadline), "{order}");
        assert_eq!(outcome.basis, None);

        let mut p = params_in(order, 1);
        let cancel = Cancel::new();
        cancel.set();
        p.cancel = Some(cancel);
        let outcome = grobx::groebner(&text, &p).unwrap();
        assert_eq!(outcome.stopped, Some(Stop::Cancel), "{order}");
        assert_eq!(outcome.basis, None);

        let mut p = params_in(order, 1);
        p.memory = Some(1);
        let outcome = grobx::groebner(&text, &p).unwrap();
        assert_eq!(outcome.stopped, Some(Stop::Memory), "{order}");
        assert_eq!(outcome.basis, None);
    }
}
