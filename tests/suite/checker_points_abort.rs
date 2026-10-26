//! checker-points-abort, the item's test code through the crate (plan.md,
//! _Test code_, test 5): a call whose checker runs in the cheap mode at
//! 2^32 − 1 points stops at its deadline, with no basis and no verdict, on
//! a dense system over GF(65521) and one over ℚ and on the edge cases of
//! `OBJECTIVES.md`, _Domain specifics the workflow points at_.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use grobx::{CheckerMode, Stop};

use crate::common::{P, dense, params};

/// The deadline of every call.
const DEADLINE: Duration = Duration::from_millis(300);
/// How long a call may take to stop. The code before the item aborted the
/// process at once.
const PATIENCE: Duration = Duration::from_secs(10);

/// A dense cubic system over each field and the edge cases, each with its
/// name.
fn cases() -> Vec<(String, String)> {
    let mut out = vec![
        ("cubic-3".to_string(), dense(3, 3, P, 1)),
        ("cubic-3 over Q".to_string(), dense(3, 3, 0, 1)),
    ];
    for (name, text) in [
        ("zero ideal", "x,y\n65521\n0\n"),
        ("zero ideal over Q", "x,y\n0\n0\n"),
        ("F = (0, 0)", "x,y\n65521\n0,\n0\n"),
        ("unit ideal", "x,y\n65521\nx*y-1,\nx\n"),
        ("unit ideal over Q", "x,y\n0\nx*y-1,\nx\n"),
        ("already a Gröbner basis", "x,y\n65521\nx+y,\ny^2+1\n"),
        ("a zero generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
        ("a zero generator over Q", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n"),
        (
            "an eliminable variable",
            "x,y,z\n65521\nx-y^2-3*z,\ny^3-z,\nz^2-1\n",
        ),
    ] {
        out.push((name.to_string(), text.to_string()));
    }
    out
}

#[test]
fn many_points_stop_at_the_deadline() {
    for (name, text) in cases() {
        let mut p = params(0x9017);
        p.checker = Some(CheckerMode::Cheap {
            points: NonZeroU32::MAX,
        });
        p.deadline = Some(DEADLINE);
        let start = Instant::now();
        let outcome = grobx::groebner(&text, &p).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let took = start.elapsed();
        assert_eq!(outcome.stopped, Some(Stop::Deadline), "{name}");
        assert_eq!(
            (
                &outcome.basis,
                &outcome.cofactors,
                &outcome.certified,
                &outcome.flaw
            ),
            (&None, &None, &None, &None),
            "{name}"
        );
        assert!(took < PATIENCE, "{name}: stopped after {took:?}");
    }
}
