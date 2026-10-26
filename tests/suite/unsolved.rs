//! gfp-unsolved, the item's test code: calls with cofactors and without,
//! signatures on and off, on dense systems, the edge cases, and random
//! systems, each returning the basis of signatures off and accepted by the
//! checker; lex calls; and a learned record replayed over its prime and
//! over another (plan.md, _Test code_). How long a call takes in the
//! measuring condition is step 7's to rule.

use std::num::NonZeroU32;

use grobx::trace::NoTrace;
use grobx_checker::{Mode, Order};

use crate::common::{
    P, P31, assert_accepted, assert_accepted_in, degree_bound, dense, params, params_in, points,
    random_system, run, with_cofactors, with_cofactors_in,
};
use crate::large::over;

/// With cofactors, signatures on and off: dense cubic systems over
/// GF(65521) give one basis, and the checker accepts it in the cheap mode.
#[test]
fn accepted_with_cofactors() {
    for seed in 1..=3 {
        let text = dense(4, 3, P, seed);
        let mut bases = Vec::new();
        for signatures in [true, false] {
            let mut p = with_cofactors(0xc0f);
            p.signatures = signatures;
            let outcome = run(&text, &p);
            let k = points(P, degree_bound(&text, &outcome));
            let mode = Mode::Cheap {
                points: NonZeroU32::new(k).unwrap(),
                seed: 0xc0f ^ u64::from(signatures),
            };
            let case = format!("dense, seed {seed}, signatures {signatures}");
            assert_accepted(&case, &text, &outcome, mode);
            bases.push(outcome.basis);
        }
        assert_eq!(bases[0], bases[1], "seed {seed}: signatures on and off");
    }
}

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, over 65521.
const EDGE: &[(&str, &str)] = &[
    ("zero-ideal", "x,y\n65521\n0,\n0\n"),
    ("f-is-zero", "x,y\n65521\n0\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n"),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
    ),
    ("already-a-basis", "x,y\n65521\nx+y,\ny^2+1\n"),
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    ("eliminable", "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
];

#[test]
fn the_edge_cases_with_signatures() {
    for &(name, text) in EDGE {
        for p in [7, P, P31] {
            let text = over(text, p);
            let outcome = run(&text, &with_cofactors(3));
            let mut off = with_cofactors(3);
            off.signatures = false;
            let plain = run(&text, &off).basis;
            assert_eq!(
                outcome.basis, plain,
                "{name} over {p}: the plain run's basis"
            );
            assert_eq!(
                run(&text, &params(3)).basis,
                plain,
                "{name} over {p}, no cofactors: the plain run's basis"
            );
            assert_accepted(&format!("{name}-p{p}"), &text, &outcome, Mode::Exact);
        }
    }
}

#[test]
fn random_systems_with_signatures() {
    for p in [P, P31] {
        for seed in 0..30 {
            let text = random_system(p, seed);
            let on = run(&text, &with_cofactors(seed));
            let mut off = params(seed);
            off.signatures = false;
            let plain = run(&text, &off).basis;
            assert_eq!(on.basis, plain, "random-p{p}-{seed}\n{text}");
            assert_eq!(
                run(&text, &params(seed)).basis,
                plain,
                "random-p{p}-{seed}, no cofactors\n{text}"
            );
            assert_accepted(&format!("random-p{p}-{seed}"), &text, &on, Mode::Exact);
        }
    }
}

/// A lex call with signatures on and one with cofactors return the lex
/// basis of the call with signatures off, and the checker accepts the
/// basis with cofactors.
#[test]
fn a_lex_call_with_signatures_returns_the_plain_basis() {
    let text = dense(3, 3, P, 1);
    let case = "cubic-3";
    let mut off = params_in("lex", 4);
    off.signatures = false;
    let plain = run(&text, &off).basis;
    assert_eq!(run(&text, &params_in("lex", 4)).basis, plain, "{case}");
    let with = run(&text, &with_cofactors_in("lex", 4));
    assert_eq!(with.basis, plain, "{case}, cofactors");
    assert_accepted_in(case, &text, &with, &Order::Lex, Mode::Exact);
}

/// A learned record replays over its prime to the learned basis, and over
/// another prime to the basis of a call there.
#[test]
fn a_record_replays() {
    for m in [3, 4] {
        let text = dense(m, 3, P, 2);
        let case = format!("cubic-{m}");
        let (outcome, record) = grobx::learn(&text, &params(1), &mut NoTrace).unwrap();
        let record = record.expect("a record");
        let again = grobx::replay(&text, &params(2), &record, &mut NoTrace).unwrap();
        assert_eq!(again.basis, outcome.basis, "{case}");
        let other = over(&text, P31);
        let replayed = grobx::replay(&other, &params(2), &record, &mut NoTrace).unwrap();
        assert_eq!(
            replayed.basis,
            run(&other, &params(1)).basis,
            "{case} over {P31}"
        );
    }
}
