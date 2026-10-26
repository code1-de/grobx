//! order-family, the item's test code: the order family through the Python
//! module, on `wrapper.rs`'s module.

use crate::wrapper::python;

#[test]
fn the_orders_through_python() {
    python(
        r#"
import grobx
TINY = "x,y\n65521\nx^2+y,\nx*y+1\n"
def raises(kind, f):
    try:
        f()
    except kind:
        return
    raise AssertionError(f"no {kind.__name__}")
for order in ["lex", "block:1"]:
    r = grobx.groebner(TINY, order=order, threads=1, seed=1)
    assert r.basis == "[y^3+1,\nx+65520*y^2]:\n", (order, repr(r.basis))
    r = grobx.groebner(TINY, order=order, threads=1, seed=1, checker="exact")
    assert r.certified is True and r.flaw is None, (order, r.certified, r.flaw)
r = grobx.groebner("x,y,z\n65521\nx-1,\ny-z^2\n", order="block:1", threads=1, seed=1)
assert r.basis == "[z^2+65520*y,\nx+65520]:\n", repr(r.basis)
raises(grobx.Error, lambda: grobx.groebner(TINY, order="deglex", threads=1, seed=1))
raises(grobx.Error, lambda: grobx.groebner(TINY, order="block:2", threads=1, seed=1))
"#,
    );
}
