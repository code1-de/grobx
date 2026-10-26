//! q-zero-denominator, the item's test code through the Python module: a
//! denominator 0 over ℚ raises `grobx.Error` naming `NoInverse` and its
//! line, rather than run on, with and without a deadline (plan.md, _Test
//! code_).

use crate::wrapper::run;

/// The script ends itself after 20 s: the code before the item answered
/// none without a bound.
#[test]
fn a_denominator_0_raises() {
    run(r#"
import signal
signal.alarm(20)
for text, line in [("x\n0\nx+1/0\n", 3), ("x,y\n0\nx+1/2,\ny-1/0\n", 4)]:
    for deadline in [None, 60000]:
        try:
            grobx.groebner(text, order="grevlex", threads=1, seed=1, deadline=deadline)
        except grobx.Error as e:
            assert str(e).startswith(f"NoInverse: line {line}: "), (str(e), text, deadline)
        else:
            raise AssertionError(f"no grobx.Error: {text!r}, deadline {deadline}")
"#);
}
