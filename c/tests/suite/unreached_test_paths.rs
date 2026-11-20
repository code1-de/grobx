//! unreached-test-paths, the item's test code through the C interface
//! (`iterations/0151-unreached-test-paths/plan.md`, _Test code_, test 5): the checker's points
//! stop at a deadline that falls in the check.

use std::num::NonZeroU32;

use grobx::{CheckerMode, MonomialOrder};
use grobxc::{GROBX_CERTIFIED, GROBX_OK, GROBX_STOP_DEADLINE, GROBX_STOP_NONE, GROBX_UNCHECKED};

use crate::common::{Args, assert_matches};
use crate::systems::dense;

// unreached-test-paths.1 design-checked -- the checker's points meet their deadline on a system whose run ends in milliseconds, beside the same call at 2 points, certified, not on the system of `checker_points.rs`, whose run the deadline stops before the check starts, nor with a later deadline, which a run on a loaded machine can outlast (log 0151-1)
/// Test 5: on a dense cubic system on three variables over GF(65521) and over ℚ, whose run ends
/// in milliseconds, the cheap checker at 2^32 − 1 points with a deadline of 300 ms returns
/// `GROBX_OK` stopped at the deadline, with no basis and no verdict, as the crate does; at 2
/// points with the same deadline it returns the crate's basis, certified.
#[test]
fn the_checkers_points_stop_at_the_deadline() {
    for (name, text) in [
        ("cubic-3-p65521", dense(3, 3, 65521, 1)),
        ("cubic-3-p0", dense(3, 3, 0, 1)),
    ] {
        let mut args = Args::new(MonomialOrder::Grevlex, 7);
        args.deadline = Some(300);
        args.checker = Some(CheckerMode::Cheap {
            points: NonZeroU32::MAX,
        });
        let got = assert_matches(&format!("points-many-{name}"), &text, &args);
        assert_eq!(
            (got.status, got.stopped, got.certified),
            (GROBX_OK, GROBX_STOP_DEADLINE, GROBX_UNCHECKED),
            "{name}: {got:?}"
        );
        assert_eq!((&got.basis, &got.flaw), (&None, &None), "{name}: {got:?}");
        args.checker = Some(CheckerMode::Cheap {
            points: NonZeroU32::new(2).expect("2 is not 0"),
        });
        let got = assert_matches(&format!("points-two-{name}"), &text, &args);
        assert_eq!(
            (got.status, got.stopped, got.certified),
            (GROBX_OK, GROBX_STOP_NONE, GROBX_CERTIFIED),
            "{name}: {got:?}"
        );
        assert!(got.basis.is_some(), "{name}: no basis");
    }
}
