//! q-crt-time, the item's test code: a call over Q with cofactors is
//! accepted by the checker in the exact mode (plan.md, _Test code_).

use grobx_checker::Mode;

use crate::common::{assert_accepted, dense, run, with_cofactors};

/// The seed of every call here.
const SEED: u64 = 0xc47;

/// q-crt-time.1: a call over Q with cofactors. The checker accepts the
/// result.
#[test]
fn accepted_with_cofactors() {
    let text = dense(3, 3, 0, 3);
    let outcome = run(&text, &with_cofactors(SEED));
    assert_accepted("cubic-3", &text, &outcome, Mode::Exact);
}
