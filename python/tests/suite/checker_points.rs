//! checker-points-abort, the item's test code through the Python module
//! (plan.md, _Test code_, test 8): `checker=4294967295` with a deadline
//! stops at the deadline, with no basis and no verdict.

use crate::systems::dense;
use crate::wrapper::run_with;

/// The script ends itself after 20 s. The call runs on `dense(8, 3,
/// 65521, 1)`, which runs for seconds without the deadline: the deadline
/// stops its run before the checker starts. The crate's
/// `checker_points_abort.rs` reaches the checker.
#[test]
fn many_points_stop_at_the_deadline() {
    run_with(
        &[("LONG", &dense(8, 3, 65521, 1))],
        r#"
import faulthandler
faulthandler.dump_traceback_later(20, exit=True)
r = grobx.groebner(LONG, order="grevlex", threads=1, seed=7, checker=4294967295, deadline=300)
assert r.stopped == "deadline", r.stopped
assert r.basis is None and r.certified is None and r.flaw is None, (r.basis, r.certified, r.flaw)
"#,
    );
}
