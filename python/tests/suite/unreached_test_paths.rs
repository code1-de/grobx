//! unreached-test-paths, the item's test code through the Python module
//! (`iterations/0151-unreached-test-paths/plan.md`, _Test code_, test 6): the checker's points
//! stop at a deadline that falls in the check.

use crate::systems::dense;
use crate::wrapper::run_with;

// unreached-test-paths.1 design-checked -- the checker's points meet their deadline on a system whose run ends in milliseconds, beside the same call at 2 points, certified, not on the system of `checker_points.rs`, whose run the deadline stops before the check starts, nor with a later deadline, which a run on a loaded machine can outlast (log 0151-1)
/// Test 6: on a dense cubic system on three variables over GF(65521) and over ℚ, whose run ends
/// in milliseconds, `checker=4294967295` with `deadline=300` stops at the deadline with no basis
/// and no verdict; `checker=2` with the same deadline returns the basis of the call without the
/// checker, certified. The script ends itself after 20 s.
#[test]
fn the_checkers_points_stop_at_the_deadline() {
    run_with(
        &[("P", &dense(3, 3, 65521, 1)), ("Q", &dense(3, 3, 0, 1))],
        r#"
import faulthandler
faulthandler.dump_traceback_later(20, exit=True)
for name, text in [("cubic-3-p65521", P), ("cubic-3-p0", Q)]:
    r = grobx.groebner(text, order="grevlex", threads=1, seed=7, checker=4294967295, deadline=300)
    got = (r.stopped, r.basis, r.certified, r.flaw)
    assert got == ("deadline", None, None, None), (name, got)
    r = grobx.groebner(text, order="grevlex", threads=1, seed=7, checker=2, deadline=300)
    got = (r.stopped, r.certified, r.flaw)
    assert got == (None, True, None), (name, got)
    assert r.basis == grobx.groebner(text, order="grevlex", threads=1, seed=7).basis, name
"#,
    );
}
