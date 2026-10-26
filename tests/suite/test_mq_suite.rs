//! test-mq-systems, sub-item 2: no file of the crate's suite calls `dense`
//! of degree 2 (`iterations/0084-test-mq-systems/plan.md`, _Test code_, 7).

use crate::common::dense_quadric_calls;

/// Test 7: no file under `tests/` calls `dense` of degree 2.
#[test]
fn the_suite_writes_no_dense_quadrics() {
    let calls = dense_quadric_calls(&["tests"]);
    assert!(calls.is_empty(), "dense quadrics at {calls:#?}");
}
