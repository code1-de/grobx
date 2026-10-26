//! test-mq-systems, the item: `dense` refuses degree 2, whose systems are
//! the instance corpus's `mq` family, and no file of the workspace's test
//! code calls it so (`iterations/0084-test-mq-systems/plan.md`, _Test
//! code_, 1 and 2).

use crate::common::{P, dense, dense_quadric_calls};

/// Test 1: `dense` of degree 2 panics, its message naming the family
/// (test-mq-systems.3). The degree is a variable, so that test 2's scan
/// reads no literal 2 here.
#[test]
#[should_panic(expected = "mq family")]
fn dense_refuses_quadrics() {
    let quadrics = 2;
    dense(3, quadrics, P, 1);
}

/// Test 2: no file of the test code of the crate, the C interface, the
/// CLI, or the Python extension calls `dense` of degree 2.
#[test]
fn no_test_writes_dense_quadrics() {
    let calls = dense_quadric_calls(&["src", "tests", "c/tests", "cli/tests", "python/tests"]);
    assert!(calls.is_empty(), "dense quadrics at {calls:#?}");
}
