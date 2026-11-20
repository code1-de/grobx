//! deglex-order, the item's test code: calls in the degree lexicographic
//! order through the crate, over GF(p), GF(2), and ℚ, on the edge cases,
//! bases computed by hand, and generated systems. Each result is canonical
//! in deglex and accepted by the checker in it, in its matrix order of a
//! row of ones and the unit rows of the first n − 1 variables
//! (`iterations/0157-deglex-order/plan.md`, _Test code_, 3 to 10).

use std::time::Duration;

use grobx::trace::NoTrace;
use grobx::{Cancel, CheckerMode, Error, MonomialOrder, Stop};
use grobx_checker::Mode;

use crate::common::systems::boolean;
use crate::common::{
    EDGE_GENERATORS, P31, PRIMES, TINY, assert_accepted_in, dense, params_in, random_square,
    random_system, reported, run, signatures_in, with_bases, with_cofactors_in,
};
use crate::large::over;
use crate::orders::{Family, accepted_in, assert_ordered, learn, names};
use crate::report::report;

/// `EDGE_GENERATORS` with their bases over GF(65521) in deglex, each found
/// by hand: on the three cases with a basis of several elements the
/// leading monomials are those of grevlex, and so are the texts.
const EDGE_DEGLEX: [(&str, &str, &str); 7] = with_bases([
    "[]:\n",
    "[]:\n",
    "[]:\n",
    "[1]:\n",
    "[x+z,\nz^2+1]:\n",
    "[y^2+65520*x,\nx*y+65520*z,\nx^2+65520*y*z]:\n",
    "[x+65519*z,\nz^2+65520*y,\ny*z+65520,\ny^2+65520*z]:\n",
]);

/// The instance on x, y, z over the characteristic p.
fn on_xyz(p: u64, generators: &str) -> String {
    format!("x,y,z\n{p}\n{generators}\n")
}

/// The outcome of `text` in deglex with cofactors, ordered in deglex and
/// accepted by the checker in its exact mode; over GF(p) its leading
/// coefficients 1 as well, over ℚ its scaling the checker's verdict on a
/// reduced basis.
fn in_deglex(case: &str, text: &str) -> grobx::Outcome {
    if text.lines().nth(1) == Some("0") {
        let outcome = run(text, &with_cofactors_in("deglex", 0x0de7));
        let basis = outcome.basis.as_deref().unwrap();
        assert_ordered(case, text, basis, "deglex");
        let order = Family::Deglex.checker(names(text).len());
        assert_accepted_in(case, text, &outcome, &order, Mode::Exact);
        outcome
    } else {
        accepted_in(case, text, "deglex", Mode::Exact)
    }
}

/// Test 3: `deglex` names the order, and a call takes it.
#[test]
fn the_name_of_deglex() {
    let order: MonomialOrder = "deglex".parse().unwrap();
    assert_eq!(order, MonomialOrder::Deglex);
    assert_eq!(order.to_string(), "deglex");
    let three = "x,y,z\n65521\nx*y-z,\ny^2-x*z,\nz^3-x\n";
    assert!(run(three, &params_in("deglex", 1)).basis.is_some());
}

/// Test 4: the edge cases over GF(65521), each with its basis written above,
/// and over GF(2) and ℚ, each accepted by the checker in deglex.
#[test]
fn the_edge_cases_in_deglex() {
    for (name, generators, want) in EDGE_DEGLEX {
        let text = on_xyz(65521, generators);
        let basis = in_deglex(name, &text).basis.unwrap();
        assert_eq!(basis, want, "{name}");
    }
    for p in [2, 0] {
        for (name, generators) in EDGE_GENERATORS {
            in_deglex(&format!("{name} over {p}"), &on_xyz(p, generators));
        }
    }
}

/// Test 5: bases computed by hand. y² − xz leads with xz in deglex and with
/// y² in grevlex. xz − y² and y³ − 1 have coprime leading monomials in
/// deglex, so they are the reduced basis; in grevlex y² − xz leads with y²,
/// y³ − 1 reduces to xyz − 1, and their S-polynomial to x²z² − y.
#[test]
fn bases_computed_by_hand() {
    let cases: &[(&str, &str, &str)] = &[
        ("y^2-x*z", "65521", "[x*z+65520*y^2]:\n"),
        ("y^2+x*z", "2", "[x*z+y^2]:\n"),
        ("y^2-x*z", "0", "[x*z-y^2]:\n"),
        ("x*z-y^2,\ny^3-1", "65521", "[x*z+65520*y^2,\ny^3+65520]:\n"),
        ("x-1,\ny-z^2", "65521", "[x+65520,\nz^2+65520*y]:\n"),
        ("x*z-y^2,\ny^3-1", "0", "[x*z-y^2,\ny^3-1]:\n"),
    ];
    for &(generators, p, want) in cases {
        let text = format!("x,y,z\n{p}\n{generators}\n");
        let basis = in_deglex(generators, &text).basis.unwrap();
        assert_eq!(basis, want, "{generators} over {p}");
    }
    let text = on_xyz(65521, "x*z-y^2,\ny^3-1");
    let grevlex = run(&text, &params_in("grevlex", 1)).basis.unwrap();
    assert_eq!(
        grevlex,
        "[y^2+65520*x*z,\nx*y*z+65520,\nx^2*z^2+65520*y]:\n"
    );
}

/// Test 6: on two variables deglex is grevlex, and on one it is grevlex
/// and lex.
#[test]
fn deglex_on_one_and_two_variables() {
    let mut cases = vec![TINY.to_string()];
    for p in PRIMES {
        for seed in 1..=3 {
            cases.push(dense(2, 3, p, seed));
        }
    }
    for text in &cases {
        let deglex = in_deglex("two variables", text).basis;
        assert_eq!(deglex, run(text, &params_in("grevlex", 1)).basis, "{text}");
    }
    for text in [
        "x\n65521\nx^3-1,\nx^2-1\n",
        "x\n7\nx^4-1,\nx^6-1\n",
        "x\n0\nx^4-1,\nx^6-1\n",
    ] {
        let deglex = in_deglex("one variable", text).basis;
        assert_eq!(deglex, run(text, &params_in("grevlex", 1)).basis, "{text}");
        assert_eq!(deglex, run(text, &params_in("lex", 1)).basis, "{text}");
    }
}

/// The generated systems the tests below run on over every prime of
/// `PRIMES` and over ℚ: name and instance.
fn generated() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for p in PRIMES.into_iter().chain([0]) {
        for seed in 0..6 {
            out.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
        for seed in [1, 2, 3, 27] {
            out.push((format!("square-p{p}-{seed}"), random_square(p, seed)));
        }
        out.push((format!("cubic-p{p}"), dense(3, 3, p, 1)));
    }
    for seed in 0..4 {
        out.push((format!("boolean-{seed}"), boolean(5, 6, seed)));
    }
    out
}

/// Test 7: every generated system in deglex, accepted by the checker in its
/// exact mode, the same basis without cofactors, and the same with
/// signatures off as with them on.
#[test]
fn generated_systems_in_deglex() {
    for (name, text) in generated() {
        let cofactors = in_deglex(&name, &text).basis;
        let on = run(&text, &signatures_in("deglex", true, 3)).basis;
        assert_eq!(cofactors, on, "{name}: with cofactors and without\n{text}");
        let off = run(&text, &signatures_in("deglex", false, 5)).basis;
        assert_eq!(on, off, "{name}: signatures on and off\n{text}");
    }
}

/// Test 8: the checker parameter certifies a deglex call in both modes,
/// and the call returns no cofactors it was not asked for.
#[test]
fn the_checker_parameter_in_deglex() {
    for (name, text) in generated().into_iter().step_by(3) {
        for mode in [
            CheckerMode::Exact,
            CheckerMode::Cheap {
                points: std::num::NonZeroU32::new(6).unwrap(),
            },
        ] {
            let mut p = params_in("deglex", 5);
            p.checker = Some(mode);
            let outcome = run(&text, &p);
            assert_eq!(
                outcome.certified,
                Some(true),
                "{name}, {mode:?}: {flaw:?}",
                flaw = outcome.flaw
            );
            assert_eq!(outcome.flaw, None, "{name}");
            assert_eq!(outcome.cofactors, None, "{name}");
        }
    }
}

/// Test 9: every bound stops a deglex run.
#[test]
fn every_bound_stops_a_deglex_run() {
    let text = dense(8, 3, 65521, 1);
    let mut p = params_in("deglex", 1);
    p.deadline = Some(Duration::ZERO);
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Deadline));
    assert_eq!(outcome.basis, None);

    let mut p = params_in("deglex", 1);
    let cancel = Cancel::new();
    cancel.set();
    p.cancel = Some(cancel);
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel));
    assert_eq!(outcome.basis, None);

    let mut p = params_in("deglex", 1);
    p.memory = Some(1);
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Memory));
    assert_eq!(outcome.basis, None);
}

/// Test 10: the call report's `call` line and every `run` line name
/// deglex, over GF(65521) and over ℚ.
#[test]
fn the_report_names_deglex() {
    for p in [65521, 0] {
        let text = dense(3, 3, p, 1);
        let outcome = run(&text, &reported(params_in("deglex", 1)));
        let lines = report(&outcome);
        assert!(
            lines[0].is("call", &[("order", "deglex")]),
            "{call:?}",
            call = lines[0]
        );
        let runs: Vec<_> = lines.iter().filter(|l| l.kind == "run").collect();
        assert!(!runs.is_empty(), "over {p}: no run line");
        for line in runs {
            assert_eq!(line.value("order"), "deglex", "over {p}: {line:?}");
        }
    }
}

/// Test 11: a record learned in deglex over one prime replays in deglex over
/// another to the deglex run's basis there, and a record of another order
/// does not fit a deglex replay, nor a deglex record another order's.
#[test]
fn a_deglex_record_replays_in_deglex() {
    let mut cases = vec![
        ("tiny".to_string(), TINY.to_string()),
        ("cubic-3".to_string(), dense(3, 3, 65521, 1)),
    ];
    for seed in 0..5 {
        cases.push((format!("random-{seed}"), random_system(65521, seed)));
    }
    for (name, text) in cases {
        let record = learn(&name, &text, "deglex");
        let there = over(&text, P31);
        let replayed = grobx::replay(&there, &params_in("deglex", 1), &record, &mut NoTrace)
            .unwrap_or_else(|e| panic!("{name}: the replay failed: {e:?}"));
        assert_eq!(
            replayed.basis,
            run(&there, &params_in("deglex", 1)).basis,
            "{name}"
        );
    }
    for (learned, replayed) in [
        ("deglex", "grevlex"),
        ("grevlex", "deglex"),
        ("deglex", "lex"),
        ("lex", "deglex"),
    ] {
        let record = learn("tiny", TINY, learned);
        let got = grobx::replay(TINY, &params_in(replayed, 1), &record, &mut NoTrace);
        assert!(
            matches!(got, Err(Error::Misfit { step: 0 })),
            "learned in {learned}, replayed in {replayed}: {got:?}"
        );
    }
}
