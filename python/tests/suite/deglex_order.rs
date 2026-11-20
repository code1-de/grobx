//! deglex-order, the item's test code through the Python module:
//! `order="deglex"` returns the basis computed by hand, which the exact
//! checker certifies, over GF(65521), GF(2), and ℚ
//! (`iterations/0157-deglex-order/plan.md`, _Test code_, 14).

use crate::wrapper::python;

/// Test 14: the bases of test 5 through `grobx.groebner`.
#[test]
fn the_python_module_in_deglex() {
    python(
        r#"
import grobx
cases = [
    ("x,y,z\n65521\ny^2-x*z\n", "[x*z+65520*y^2]:\n"),
    ("x,y,z\n2\ny^2+x*z\n", "[x*z+y^2]:\n"),
    ("x,y,z\n0\ny^2-x*z\n", "[x*z-y^2]:\n"),
    ("x,y,z\n65521\nx*z-y^2,\ny^3-1\n", "[x*z+65520*y^2,\ny^3+65520]:\n"),
    ("x,y,z\n0\nx*z-y^2,\ny^3-1\n", "[x*z-y^2,\ny^3-1]:\n"),
]
for text, want in cases:
    r = grobx.groebner(text, order="deglex", threads=1, seed=1)
    assert r.basis == want, (text, repr(r.basis))
    r = grobx.groebner(text, order="deglex", threads=1, seed=1, checker="exact")
    assert r.basis == want and r.certified is True and r.flaw is None, (text, r.certified, r.flaw)
r = grobx.groebner(cases[3][0], order="grevlex", threads=1, seed=1)
assert r.basis != cases[3][1], repr(r.basis)
"#,
    );
}
