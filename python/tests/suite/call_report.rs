//! call-report, sub-item 3: the report through the Python module
//! (`iterations/0064-call-report/plan.md`, _Test code_, 20, and
//! call-report.16).

use crate::systems::dense;
use crate::wrapper::run_with;

/// `report=True` returns the report as a `str`, whose first line is the
/// call's and last line its end, beside the basis the call without it
/// returns, over GF(65521) and over ℚ; `report=False` and no keyword
/// return None; a call a deadline stops returns a report that ends with
/// its stop; an error raises grobx.Error as without the report.
/// `report_file`, a str or a path, holds the report `report` returns; an
/// error leaves it ending with the `error` line; a file that cannot be
/// created raises `OSError`, and a value that is no path `TypeError`. The
/// stopped call runs on `dense(8, 3, 65521, 1)`, which runs for seconds
/// without the deadline.
#[test]
fn the_report_through_python() {
    run_with(
        &[
            ("P", &dense(4, 3, 65521, 1)),
            ("Q", &dense(3, 3, 0, 1)),
            ("LONG", &dense(8, 3, 65521, 1)),
        ],
        r#"
def kinds(report):
    lines = report.split("\n")
    assert lines[-1] == "", repr(report[-40:])
    lines = lines[:-1]
    for line in lines:
        words = line.split(" ")
        assert len(words) >= 2, line
        seconds, fraction = words[0].split(".")
        assert seconds.isdigit() and fraction.isdigit() and len(fraction) == 6, line
        for word in words[2:]:
            key, value = word.split("=")
            assert key and value, line
    return [line.split(" ")[1] for line in lines]

for stem, text in [("p65521", P), ("q", Q)]:
    plain = grobx.groebner(text, order="grevlex", threads=1, seed=5)
    assert plain.report is None, stem
    off = grobx.groebner(text, order="grevlex", threads=1, seed=5, report=False)
    assert off.report is None, stem
    on = grobx.groebner(text, order="grevlex", threads=1, seed=5, report=True)
    assert isinstance(on.report, str), type(on.report)
    assert on.basis == plain.basis, stem
    assert on.report is on.report, "every read returns the same object"
    k = kinds(on.report)
    assert k[0] == "call" and k[1] == "read" and k[-1] == "end", (stem, k)
    assert "run" in k and "ran" in k, (stem, k)
    assert on.report.split("\n")[-2].split(" ")[2] == "result=basis", stem

r = grobx.groebner(LONG, order="grevlex", threads=1, seed=5, deadline=50, report=True)
assert r.stopped == "deadline", r.stopped
k = kinds(r.report)
assert k[-2:] == ["stop", "end"], k
assert r.report.split("\n")[-2].split(" ")[2] == "result=stopped"

raises(grobx.Error, lambda: grobx.groebner("x\n7\nx^2147483648\n", order="grevlex",
                                           threads=1, seed=5, report=True))

import os, pathlib, tempfile
d = tempfile.mkdtemp()
for path in [os.path.join(d, "ok.report"), pathlib.Path(d) / "ok-path.report"]:
    r = grobx.groebner(P, order="grevlex", threads=1, seed=5, report_file=path)
    with open(path) as f:
        written = f.read()
    assert written == r.report, (written, r.report)
    assert kinds(written)[-1] == "end", written

failed = os.path.join(d, "error.report")
raises(grobx.Error, lambda: grobx.groebner("x\n7\nx^2147483648\n", order="grevlex",
                                           threads=1, seed=5, report_file=failed))
with open(failed) as f:
    written = f.read()
k = kinds(written)
assert k[0] == "call" and k[-1] == "error", k
assert written.split("\n")[-2].split(" ")[2:] == [
    "error=Overflow", "message=line_3:_a_degree_past_the_widest_monomial_format"], written

raises(OSError, lambda: grobx.groebner(P, report_file=os.path.join(d, "missing", "x.report")))
raises(TypeError, lambda: grobx.groebner(P, report_file=5))
"#,
    );
}
