//! report-peak-inherited, the CLI's test code: the report's peak is the
//! binary's own, whatever resident set the process that started it held
//! (`iterations/0099-cli-report-device/plan.md`, _Test code_, 6).

use std::fs;

use crate::common::{grobx, scratch, write_instance};
use crate::systems::dense;

/// The bytes this process holds, every page touched, while it starts the
/// binary.
const HELD: usize = 256 << 20;

/// The field `key` of the report's `end` line, in bytes.
fn field(end: &str, key: &str) -> u64 {
    let value = end
        .split(' ')
        .find_map(|f| f.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} in {end}"));
    value
        .parse()
        .unwrap_or_else(|e| panic!("{key}={value}: {e}"))
}

/// Test 6: report-peak-inherited.1. The binary started from this process
/// while it holds `HELD` touched bytes reports a peak below half of them,
/// at or above its start, and a growth that is their difference.
#[test]
fn the_peak_is_the_process_s_own() {
    let mut held = vec![0u8; HELD];
    for i in (0..HELD).step_by(4096) {
        held[i] = 1;
    }
    let dir = scratch("report-peak-inherited");
    let file = dir.join("dense.report");
    let report = format!("--report={report}", report = file.display());
    let input = write_instance(&dir, "dense", &dense(4, 3, 65521, 1))
        .display()
        .to_string();
    let ran = grobx(&["--seed", "3", report.as_str(), input.as_str()], "");
    assert_eq!(ran.code(), 0, "{ran:?}");
    // `held` stays resident across the start of the binary.
    assert!(held.iter().step_by(4096).all(|&b| b == 1));
    let text = fs::read_to_string(&file).unwrap();
    let end = text.lines().last().unwrap();
    assert!(end.contains(" end "), "{end}");
    let start = field(end, "process-rss-start");
    let peak = field(end, "process-rss-peak");
    let grown = field(end, "process-rss-grown");
    assert!(peak < HELD as u64 / 2, "{end}");
    assert!(peak >= start, "{end}");
    assert_eq!(grown, peak - start, "{end}");
}
