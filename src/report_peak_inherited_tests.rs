//! report-peak-inherited, the crate's test code: the process's peak
//! against its resident set
//! (`iterations/0099-cli-report-device/plan.md`, _Test code_, 7).

use crate::child_tests::measures_here;
use crate::os::{Resident, peak};

/// The bytes the test holds, every page touched.
const HELD: usize = 64 << 20;

/// The least `HELD` raises the resident set by; the rest may be resident
/// already.
const RAISED: u64 = 60 << 20;

/// Test 7: report-peak-inherited.1, in a child process of this test
/// binary, which runs this test alone. `peak` is Some; `HELD` touched bytes
/// raise the resident set by at least `RAISED`, and the peak read after
/// is at least that resident set and at least the peak read before. The
/// parent runs the child.
#[test]
fn the_peak_is_at_least_the_resident_set() {
    if !measures_here("report_peak_inherited_tests::the_peak_is_at_least_the_resident_set") {
        return;
    }
    let resident = Resident::open().expect("the resident set is readable here");
    let before = peak().expect("the peak is readable here");
    let rss_before = resident.bytes().expect("the resident set is readable");
    let mut held = vec![0u8; HELD];
    for i in (0..HELD).step_by(4096) {
        held[i] = 1;
    }
    let rss_after = resident.bytes().expect("the resident set is readable");
    assert!(
        rss_after >= rss_before + RAISED,
        "{rss_before} B, then {rss_after} B"
    );
    let after = peak().expect("the peak is readable here");
    assert!(after >= rss_after, "peak {after} B, resident {rss_after} B");
    assert!(after >= before, "peak {before} B, then {after} B");
    assert!(held.iter().step_by(4096).all(|&b| b == 1));
}
