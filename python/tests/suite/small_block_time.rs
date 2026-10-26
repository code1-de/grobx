//! small-block-time, the item's test code for the Python module (plan.md,
//! _Test code_, test 12): a process that imports the module runs it under
//! `grobx::Allocator`, mimalloc serving its small blocks.

use crate::systems::dense;
use crate::wrapper::run_with;

/// Test 12, small-block-time.4: a child interpreter with
/// `MIMALLOC_VERBOSE=1` imports the module and calls it over GF(65521) and
/// over ℚ: mimalloc reports on standard error the memory it reserves to
/// serve a block, and each basis digest is the parent's for the same call.
#[test]
fn the_module_runs_under_the_allocator() {
    run_with(
        &[("P", &dense(4, 3, 65521, 1)), ("Q", &dense(3, 3, 0, 1))],
        r#"
import subprocess, sys
child = """
import sys, grobx
for text in sys.argv[1:]:
    print(grobx.groebner(text, order="grevlex", threads=1, seed=3).digest)
"""
done = subprocess.run([sys.executable, "-c", child, P, Q], env={**os.environ, "MIMALLOC_VERBOSE": "1"}, capture_output=True, text=True)
assert done.returncode == 0, done.stderr
assert any(l.startswith("mimalloc: reserved") for l in done.stderr.splitlines()), done.stderr
want = [grobx.groebner(text, order="grevlex", threads=1, seed=3).digest for text in [P, Q]]
assert done.stdout.split() == want, (done.stdout, want)
"#,
    );
}
