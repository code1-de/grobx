//! checker-points-abort, the item's test code through the CLI (plan.md,
//! _Test code_, test 6): `--checker 4294967295` with a deadline stops at
//! the deadline, exits 1, and writes no basis.

use crate::common::grobx;
use crate::systems::dense;

/// The call runs on `dense(8, 3, 65521, 1)`, which runs for seconds
/// without the deadline: the deadline stops its run before the checker
/// starts. The crate's `checker_points_abort.rs` reaches the checker.
#[test]
fn many_points_stop_at_the_deadline() {
    let text = dense(8, 3, 65521, 1);
    let ran = grobx(
        &[
            "--seed",
            "7",
            "--checker",
            "4294967295",
            "--deadline",
            "300",
        ],
        &text,
    );
    assert_eq!(ran.code(), 1, "{ran:?}");
    assert_eq!(ran.stderr, "grobx: stopped: deadline\n", "{ran:?}");
    assert!(ran.stdout.is_empty(), "{ran:?}");
}
