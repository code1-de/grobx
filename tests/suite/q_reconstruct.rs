//! q-reconstruct-time, the item's test code: two calls over Q with
//! cofactors are accepted by the checker in the exact mode (plan.md,
//! _Test code_).

use grobx_checker::Mode;

use crate::common::{assert_accepted, dense, random_square, run, with_cofactors};

/// Two calls over Q with cofactors (q-lift.17): the checker accepts both
/// results.
#[test]
fn accepted_with_cofactors() {
    for (name, text) in [
        ("cubic-3", dense(3, 3, 0, 2)),
        ("square-1", random_square(0, 1)),
    ] {
        let outcome = run(&text, &with_cofactors(0x5eed));
        assert_accepted(name, &text, &outcome, Mode::Exact);
    }
}
