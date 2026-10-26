//! q-replay-time, the item's test code: a call over Q with cofactors is
//! accepted by the checker in the exact mode (plan.md, _Test code_).

use grobx_checker::Mode;

use crate::common::{assert_accepted, dense, run, with_cofactors};

/// A call over Q with cofactors (q-replay-time.7): the checker accepts its
/// result.
#[test]
fn accepted_with_cofactors() {
    let text = dense(3, 3, 0, 1);
    let outcome = run(&text, &with_cofactors(0x5eed));
    assert_accepted("cubic-3", &text, &outcome, Mode::Exact);
}
