//! checker-points-abort, the item's test code through the C interface
//! (plan.md, _Test code_, test 7): the cheap mode at 0xFFFFFFFF points with
//! a deadline returns `GROBX_OK` stopped at the deadline, with no basis and
//! no verdict.

use std::num::NonZeroU32;

use grobx::{CheckerMode, MonomialOrder};
use grobxc::{GROBX_OK, GROBX_STOP_DEADLINE, GROBX_UNCHECKED};

use crate::common::{Args, drive, scratch};
use crate::systems::dense;

/// The call runs on `dense(8, 3, 65521, 1)`, which runs for seconds
/// without the deadline: the deadline stops its run before the checker
/// starts. The crate's `checker_points_abort.rs` reaches the checker.
#[test]
fn many_points_stop_at_the_deadline() {
    let dir = scratch("checker-points");
    let mut args = Args::new(MonomialOrder::Grevlex, 7);
    args.checker = Some(CheckerMode::Cheap {
        points: NonZeroU32::MAX,
    });
    args.deadline = Some(300);
    let (got, digest, _) = drive(&dir, &dense(8, 3, 65521, 1), &args);
    assert_eq!(
        (got.status, got.stopped, got.certified),
        (GROBX_OK, GROBX_STOP_DEADLINE, GROBX_UNCHECKED),
        "{got:?}"
    );
    assert_eq!(
        (&got.basis, &digest, &got.flaw),
        (&None, &None, &None),
        "{got:?}"
    );
}
