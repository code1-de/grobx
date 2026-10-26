//! row-traffic-time, the item's test code: the edge cases of
//! `OBJECTIVES.md`; the checker on generated systems; a deadline; and one
//! basis over 2^31 − 1 with signatures on and off and with cofactors
//! (`iterations/0044-row-traffic-time/plan.md`, _Test code_, 5 and 8, and
//! `iterations/0060-row-traffic-time/plan.md`, _Test code_, 11 and 12).

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use grobx::{CheckerMode, Stop};
use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, dense, params, random_square, run, with_cofactors};
use crate::orders::assert_canonical;
use crate::q::accepted_q;

/// The edge cases of `OBJECTIVES.md` on x, y, z: name, generators, and the
/// basis over GF(65521) in grevlex.
const EDGE: [(&str, &str, &str); 7] = [
    ("the zero ideal", "0", "[]:\n"),
    ("F = (0, 0)", "0,\n0", "[]:\n"),
    ("no generator", "", "[]:\n"),
    ("the unit ideal", "x*z-1,\nx", "[1]:\n"),
    ("already a Gröbner basis", "x+z,\nz^2+1", "[x+z,\nz^2+1]:\n"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
        "[y^2+65520*x,\nx*y+65520*z,\nx^2+65520*y*z]:\n",
    ),
    (
        "an eliminable variable",
        "x-y^2-z,\ny*z-1,\nz^2-y",
        "[x+65519*z,\nz^2+65520*y,\ny*z+65520,\ny^2+65520*z]:\n",
    ),
];

/// Test 5: the edge cases over 65521 and ℚ in grevlex, signatures on and
/// off: each basis canonical and accepted in the exact mode, and over
/// 65521 as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut params = with_cofactors(0x7a1e);
            params.signatures = signatures;
            let outcome = run(&text, &params);
            let basis = outcome.basis.as_deref().unwrap();
            let name = format!("{case} over {P}, signatures {signatures}");
            assert_canonical(&name, &text, basis, "grevlex");
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(basis, want, "{name}");
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}

/// Generated systems over 65521: name and text.
fn systems() -> [(&'static str, String); 2] {
    [
        ("cubic-3", dense(3, 3, P, 1)),
        ("square", random_square(P, 2)),
    ]
}

/// Test 8: the oracle. Generated systems over GF(65521) with the checker
/// in the cheap mode at two points: certified.
#[test]
fn the_checker_certifies_them() {
    for (stem, text) in systems() {
        let mut p = params(0x7a22);
        p.checker = Some(CheckerMode::Cheap {
            points: NonZeroU32::new(2).unwrap(),
        });
        let outcome = run(&text, &p);
        assert_eq!(
            outcome.certified,
            Some(true),
            "{stem}: {flaw:?}",
            flaw = outcome.flaw
        );
    }
}

/// Test 11: row-traffic-time.12. A dense cubic system on eight variables over
/// GF(65521), which runs for seconds, under seed 1 with a deadline of
/// 300 ms: the call stops with `Deadline`, no basis, and returns within 1 s
/// past the deadline.
#[test]
fn a_deadline_stops_a_long_call() {
    const DEADLINE: Duration = Duration::from_millis(300);
    let stem = "cubic-8-p65521";
    let text = dense(8, 3, P, 1);
    let mut p = params(1);
    p.deadline = Some(DEADLINE);
    let start = Instant::now();
    let outcome = grobx::groebner(&text, &p).expect("the call runs");
    let took = start.elapsed();
    assert_eq!(outcome.stopped, Some(Stop::Deadline), "{stem}");
    assert_eq!(outcome.basis, None, "{stem}");
    assert!(
        took < DEADLINE + Duration::from_secs(1),
        "{stem}: returned {late:?} past the deadline",
        late = took.saturating_sub(DEADLINE)
    );
}

/// `text` with its characteristic, its second line, set to `p`.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let variables = lines.next().unwrap();
    lines.next().unwrap();
    format!(
        "{variables}\n{p}\n{generators}",
        generators = lines.next().unwrap_or("")
    )
}

/// Test 12: generated systems with their
/// characteristic set to 2^31 − 1, in grevlex under seeds 1 and 2: the
/// basis with signatures on and no cofactors is the basis with signatures
/// on and cofactors, and the basis with signatures off. The checker in the
/// exact mode accepts the second with its cofactors.
#[test]
fn cofactors_and_signatures_keep_the_basis_over_2_31() {
    for (stem, text) in systems() {
        let text = over(&text, P31);
        for seed in [1, 2] {
            let name = format!("{stem} over {P31}, seed {seed}");
            let banded = run(&text, &params(seed)).basis.unwrap();
            let tracked = run(&text, &with_cofactors(seed));
            assert_accepted(&name, &text, &tracked, Mode::Exact);
            let mut p = params(seed);
            p.signatures = false;
            let plain = run(&text, &p).basis.unwrap();
            assert_eq!(
                Some(&banded),
                tracked.basis.as_ref(),
                "{name}: with cofactors"
            );
            assert_eq!(banded, plain, "{name}: signatures off");
        }
    }
}
