//! The release closures of the full checks, which give back what a run can rebuild where the
//! resident set is above the ceiling, and after which the run goes on. No call of the test code
//! passes its ceiling, so `RELEASE_FORCED` runs them at every full check.

use crate::budget::{RELEASE_FORCED, RELEASES};
use crate::call::built;
use crate::call_stop_sites_tests::{Case, returned};
use crate::{
    call_stop_lift_tests, call_stop_plain_tests, call_stop_read_tests, call_stop_reduce_tests,
    call_stop_sig_tests, groebner,
};

/// Every call of the call-stop tests, which reach every engine and every site of the budget,
/// returns with a release at every full check what it returns without one.
#[test]
fn a_release_at_every_check_changes_no_result() {
    let cases: Vec<Case> = [
        call_stop_plain_tests::cases(),
        call_stop_sig_tests::cases(),
        call_stop_reduce_tests::cases(),
        call_stop_read_tests::cases(),
        call_stop_lift_tests::cases(),
    ]
    .into_iter()
    .flatten()
    .map(|(_, case)| case)
    .collect();
    let mut failures = Vec::new();
    let mut ran = 0;
    for (i, case) in cases.iter().enumerate() {
        if built(&case.params).is_err() {
            continue;
        }
        let plain = groebner(&case.text, &case.params);
        RELEASE_FORCED.set(true);
        let released = groebner(&case.text, &case.params);
        RELEASE_FORCED.set(false);
        match (plain, released) {
            (Ok(p), Ok(r)) => {
                ran += 1;
                if returned(&p) != returned(&r) {
                    failures.push(format!("case {i}: the release changed the result"));
                }
            }
            (p, r) => {
                let (without, with) = (p.err(), r.err());
                failures.push(format!("case {i}: without {without:?}, with {with:?}"));
            }
        }
    }
    assert!(ran > 0, "no case was built");
    assert!(RELEASES.get() > 0, "no full check released");
    assert!(failures.is_empty(), "{failures:#?}");
}
