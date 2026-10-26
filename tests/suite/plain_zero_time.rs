//! plain-zero-time, the item's test code: calls whose plain run reduces
//! rounds of rows that mostly lie in the span of their pivots, against the
//! call that reduces each row on its own
//! (`iterations/0077-plain-zero-time/plan.md`, _Test code_, 4).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, params, run, systems::random_sparse, with_cofactors};
use crate::orders::assert_canonical;

/// The primes and seeds of `random_sparse` on six variables whose plain
/// run, signatures off, reduces a block of more than 64 rows, `BLOCK`: a
/// probe in a scratch copy found them in session 0077-1.
const CASES: [(u64, u64); 7] = [
    (P, 2),
    (P, 3),
    (P, 7),
    (P31, 0),
    (P31, 1),
    (P31, 9),
    (P31, 10),
];

/// Test 4: random sparse systems on 6 variables over 65521 and 2^31 − 1,
/// signatures off: the basis is the one the call with cofactors returns,
/// which reduces each row on its own and which the checker accepts in the
/// exact mode, and it is canonical.
#[test]
fn calls_over_rows_that_reduce_to_zero() {
    for (p, seed) in CASES {
        let text = random_sparse(6, p, seed);
        let name = format!("sparse-6 over {p}, seed {seed}");
        let mut plain = params(0x9e10 ^ seed);
        plain.signatures = false;
        let combined = run(&text, &plain).basis.unwrap();
        let mut tracked = with_cofactors(0x9e10 ^ seed);
        tracked.signatures = false;
        let outcome = run(&text, &tracked);
        assert_accepted(&name, &text, &outcome, Mode::Exact);
        assert_canonical(&name, &text, &combined, "grevlex");
        assert_eq!(Some(&combined), outcome.basis.as_ref(), "{name}");
    }
}
