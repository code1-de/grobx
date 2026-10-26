//! q-lift, the item's test code: the call over Q through the Python
//! module, on `wrapper.rs`'s module.

use crate::wrapper::python;

/// The measured call over Q returns msolve's basis of an instance with
/// fractions (log 0016-1, _Step 2_) and its digest; with cofactors and the
/// exact checker the same basis, certified, and the cofactor matrix.
#[test]
fn q_through_python() {
    python(
        r#"
import hashlib
import grobx
text = "x,y\n0\n2*x^2+3*y-1/2,\nx*y+5/3\n"
r = grobx.groebner(text, order="grevlex", threads=1, seed=1)
assert r.stopped is None
want = "[18*y^2-20*x-3*y,\n3*x*y+5,\n4*x^2+6*y-1]:\n"
assert r.basis == want, repr(r.basis)
assert r.digest == hashlib.sha256(want.encode()).hexdigest()
c = grobx.groebner(text, order="grevlex", threads=1, seed=1, cofactors=True, checker="exact")
assert c.basis == want and c.certified is True, repr(c.flaw)
assert c.cofactors.startswith("[[") and c.cofactors.endswith("]]:\n"), repr(c.cofactors)
"#,
    );
}
