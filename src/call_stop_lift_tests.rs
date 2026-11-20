//! call-stop-latency, sub-item 6's test code: the lift's verification, its result's finish, its
//! merge, its reconstruction, FGLM's steps, and the cofactor tracking's kept rows each stop a call
//! from inside (`iterations/0138-call-stop-latency/plan.md`, _Test code_,
//! `call_stop_lift_tests.rs`).

use crate::budget::Site;
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::systems::{dense, random_system};

/// The lift's, FGLM's, and the cofactors' sites, each with a call that reaches it: random systems
/// over ℚ; a dense zero-dimensional system in lex over GF(65521), which FGLM changes; a random
/// system with cofactors.
pub(crate) fn cases() -> Vec<(Site, Case)> {
    let q = || Case::new(random_system(0, 11), "grevlex", false);
    vec![
        (Site::Verify, q()),
        (Site::Finish, q()),
        (Site::Merge, q()),
        (Site::Reconstruct, q()),
        (Site::Fglm, Case::new(dense(3, 3, 65521, 2), "lex", false)),
        (
            Site::Cofactors,
            Case::new(random_system(65521, 12), "grevlex", true),
        ),
    ]
}

/// Test 1: every site of the lift, FGLM, and the cofactors stops a call from inside it, the
/// call's result unchanged with every spend checking (call-stop-latency.30 to .33).
#[test]
fn every_site_of_the_lift_stops() {
    let mut failures = Vec::new();
    for (site, case) in cases() {
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}
