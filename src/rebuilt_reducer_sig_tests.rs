//! rebuilt-reducer-time, sub-item 2, the crate's test code: the signature
//! run's matrices stay as the build without kept rows makes them, and some
//! signature round takes rows from the matrix before
//! (`iterations/0102-rebuilt-reducer-time/plan.md`, _Test code_, 9).

use crate::MonomialOrder;
use crate::rebuilt_reducer_tests::{agree, pair, reported, systems};
use crate::report_tests::Line;

/// The `kept` of the signature runs' round lines, summed.
fn signature_kept(lines: &[Line]) -> usize {
    let mut signature = false;
    let mut total = 0;
    for line in lines {
        match line.kind.as_str() {
            "run" => signature = line.get("engine") == Some("signature"),
            "round" if signature => {
                total += line.get("kept").unwrap().parse::<usize>().unwrap();
            }
            _ => {}
        }
    }
    total
}

#[test]
fn the_signature_run_takes_rows() {
    let mut params = reported(MonomialOrder::Grevlex);
    params.signatures = true;
    let mut total = 0;
    for p in [65521, 7] {
        for (name, text) in systems(p) {
            let (off, on) = pair(&text, &params);
            agree(&name, &off, &on);
            total += signature_kept(&on.1);
        }
    }
    assert!(
        total > 0,
        "no signature round took a row from the matrix before"
    );
}
