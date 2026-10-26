//! gfp-large-primes, the item's test code: the primes from 2^31 to 2^64
//! through the Python module, on `wrapper.rs`'s module.

use crate::wrapper::python;

/// The measured call over the least prime past 2^31 and the largest below
/// 2^64 returns the basis written here, valid over every prime: y·(x² + y)
/// − x·(x·y + 1) = y² − x. A composite characteristic raises `grobx.Error`.
#[test]
fn large_primes_through_python() {
    python(
        r#"
import grobx
def raises(kind, f):
    try:
        f()
    except kind:
        return
    raise AssertionError(f"no {kind.__name__}")
for p in [2147483659, 18446744073709551557]:
    text = f"x,y\n{p}\nx^2+y,\nx*y+1\n"
    r = grobx.groebner(text, order="grevlex", threads=1, seed=1)
    assert r.stopped is None
    assert r.basis == f"[y^2+{p - 1}*x,\nx*y+1,\nx^2+y]:\n", repr(r.basis)
    c = grobx.groebner(text, order="grevlex", threads=1, seed=1, cofactors=True, checker="exact")
    assert c.basis == r.basis and c.certified is True, repr(c.flaw)
for c in [4294967297, 3825123056546413051, 2**64 - 1, 2**64]:
    raises(grobx.Error, lambda: grobx.groebner(f"x,y\n{c}\nx^2+y,\nx*y+1\n", order="grevlex", threads=1, seed=1))
"#,
    );
}
