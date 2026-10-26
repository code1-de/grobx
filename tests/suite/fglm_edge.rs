//! fglm-lex, the item's test code: the edge cases in lex, hand-computed
//! bases, signatures, the checker parameter, cofactors, and the bounds
//! inside the change (plan.md, _Edge cases_). Every basis computed with
//! cofactors is canonical in lex and accepted by the checker in it.

use std::fmt::Write;
use std::num::NonZeroU32;
use std::thread;
use std::time::{Duration, Instant};

use grobx::{Cancel, CheckerMode, Params, Stop};
use grobx_checker::Mode;

use crate::common::{Rng, dense, params_in, random_square, run};
use crate::orders::{accepted_in, lex_as_blocks, names};

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points
/// at_, and the item's positive-dimensional `x*y-1`: name, instance, and
/// the basis text in lex where it is written here.
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
    ("positive-dimensional", "x,y\n65521\nx*y-1\n", None),
];

/// Every edge case in lex, accepted in the exact mode, the text written
/// above where there is one, the `block:1,…,1` basis (fglm-lex.1, .3).
#[test]
fn the_edge_cases() {
    for &(name, text, want) in EDGE {
        let outcome = accepted_in(name, text, "lex", Mode::Exact);
        let basis = outcome.basis.unwrap();
        if let Some(want) = want {
            assert_eq!(basis, want, "{name}");
        }
        let blocks = lex_as_blocks(names(text).len());
        let f4 = run(text, &params_in(&blocks, 1));
        assert_eq!(Some(basis), f4.basis, "{name}: lex and {blocks}");
    }
}

/// Bases computed by hand, each with cofactors, accepted in the exact mode.
#[test]
fn hand_computed_bases() {
    let cases: &[(&str, &str)] = &[
        ("x,y\n65521\nx-3,\ny-5\n", "[y+65516,\nx+65518]:\n"),
        ("x,y\n65521\nx^2,\ny^2\n", "[y^2,\nx^2]:\n"),
        ("x,y\n65521\nx^2,\nx*y,\ny^2\n", "[y^2,\nx*y,\nx^2]:\n"),
        ("x,y\n65521\nx^2+y,\nx*y+1\n", "[y^3+1,\nx+65520*y^2]:\n"),
        (
            "x,y\n65521\nx-y^2,\ny^3-1\n",
            "[y^3+65520,\nx+65520*y^2]:\n",
        ),
        ("x\n65521\nx^3-1,\nx^2-1\n", "[x+65520]:\n"),
        ("x\n7\nx^4-1,\nx^6-1\n", "[x^2+6]:\n"),
        (
            "x,y,z\n65521\nx^20-y,\ny^20-z,\nz^20-x\n",
            "[z^8000+65520*z,\ny+65520*z^400,\nx+65520*z^20]:\n",
        ),
    ];
    for &(text, want) in cases {
        let outcome = accepted_in("hand", text, "lex", Mode::Exact);
        assert_eq!(outcome.basis.unwrap(), want, "{text:?}");
    }
}

/// On one variable lex is grevlex.
#[test]
fn one_variable() {
    for text in ["x\n65521\nx^3-1,\nx^2-1\n", "x\n7\nx^4-1,\nx^6-1\n"] {
        assert_eq!(
            run(text, &params_in("lex", 1)).basis,
            run(text, &params_in("grevlex", 1)).basis,
            "{text:?}"
        );
    }
}

/// The zero-dimensional instances the signature, checker, and cofactor
/// cases run on.
fn zero_dimensional() -> Vec<(String, String)> {
    let mut list = vec![
        (
            "tiny".to_string(),
            "x,y\n65521\nx^2+y,\nx*y+1\n".to_string(),
        ),
        ("cubic-2".to_string(), dense(2, 3, 65521, 1)),
        ("cubic-3".to_string(), dense(3, 3, 65521, 2)),
    ];
    for seed in 0..5 {
        list.push((format!("square-{seed}"), random_square(65521, seed)));
    }
    list
}

/// Signatures do not change a zero-dimensional lex call's basis
/// (fglm-lex.2).
#[test]
fn signatures_do_not_change_the_basis() {
    for (name, text) in zero_dimensional() {
        let mut bases = Vec::new();
        for signatures in [true, false] {
            let mut p = params_in("lex", 3);
            p.signatures = signatures;
            bases.push(run(&text, &p).basis.unwrap());
        }
        assert_eq!(bases[0], bases[1], "{name}: signatures on and off");
    }
}

/// The checker parameter rules on the changed basis in lex.
#[test]
fn the_checker_parameter() {
    for (name, text) in zero_dimensional().into_iter().take(3) {
        for mode in [
            CheckerMode::Exact,
            CheckerMode::Cheap {
                points: NonZeroU32::new(6).unwrap(),
            },
        ] {
            let mut p = params_in("lex", 5);
            p.checker = Some(mode);
            let outcome = run(&text, &p);
            assert_eq!(
                outcome.certified,
                Some(true),
                "{name}, {mode:?}: {flaw:?}",
                flaw = outcome.flaw
            );
            assert_eq!(outcome.flaw, None);
            assert_eq!(outcome.cofactors, None);
        }
    }
}

/// A call with cofactors returns the basis of the call without, and the
/// checker accepts the cofactors it lifted (fglm-lex.12).
#[test]
fn cofactors_return_the_same_basis() {
    for (name, text) in zero_dimensional() {
        let with = accepted_in(&name, &text, "lex", Mode::Exact);
        assert!(with.cofactors.is_some(), "{name}");
        assert_eq!(with.basis, run(&text, &params_in("lex", 9)).basis, "{name}");
    }
}

/// On x, y, z over GF(65521), each generator its variable to the 16th plus
/// twenty random terms of lower degree. The generators are a grevlex basis,
/// and the grevlex call finishes within 100 ms. In lex the change of
/// D = 4096 runs past every bound below.
fn powers() -> String {
    let mut rng = Rng::new(0xf91);
    let generators: Vec<String> = ["x", "y", "z"]
        .iter()
        .map(|v| {
            let mut poly = format!("{v}^16");
            for _ in 0..20 {
                let mut degree = rng.below(16);
                write!(poly, "+{coef}", coef = 1 + rng.below(65520)).unwrap();
                for name in ["x", "y", "z"] {
                    let e = rng.below(degree + 1);
                    degree -= e;
                    if e > 0 {
                        write!(poly, "*{name}^{e}").unwrap();
                    }
                }
            }
            poly
        })
        .collect();
    let text = format!(
        "x,y,z\n65521\n{generators}\n",
        generators = generators.join(",\n")
    );
    let mut p = params_in("grevlex", 1);
    p.deadline = Some(Duration::from_millis(100));
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert!(
        outcome.basis.is_some(),
        "the grevlex call takes 100 ms\n{text}"
    );
    text
}

/// A deadline stops a call inside the change, within 2 s of it.
#[test]
fn a_deadline_stops_the_change() {
    let text = powers();
    let mut p = params_in("lex", 1);
    let deadline = Duration::from_millis(300);
    p.deadline = Some(deadline);
    let start = Instant::now();
    let outcome = grobx::groebner(&text, &p).unwrap();
    let took = start.elapsed();
    assert_eq!(outcome.stopped, Some(Stop::Deadline), "after {took:?}");
    assert_eq!(outcome.basis, None);
    assert!(took < deadline + Duration::from_secs(2), "{took:?}");
}

/// A cancel raised from another thread stops a call inside the change,
/// within 2 s of it.
#[test]
fn a_cancel_stops_the_change() {
    let text = powers();
    let cancel = Cancel::new();
    let mut p: Params = params_in("lex", 1);
    p.cancel = Some(cancel.clone());
    let delay = Duration::from_millis(300);
    let start = Instant::now();
    let stopper = thread::spawn(move || {
        thread::sleep(delay);
        cancel.set();
    });
    let outcome = grobx::groebner(&text, &p).unwrap();
    let took = start.elapsed();
    stopper.join().unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel), "after {took:?}");
    assert_eq!(outcome.basis, None);
    assert!(took >= delay, "{took:?}");
    assert!(took < delay + Duration::from_secs(2), "{took:?}");
}
