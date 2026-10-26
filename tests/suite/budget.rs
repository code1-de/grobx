//! Sub-item 3: the budget's bounds (f4-gfp.27; the checker's
//! `tests/suite/poll.rs` checks its poll, f4-gfp.26). Each bound stops a
//! run it is set to stop, with no basis and the reason; bounds that are not
//! reached change nothing.
//!
//! A memory ceiling where the resident set cannot be read refuses to start
//! with `Error::Memory`. That machine is not this one, so no test here
//! reaches it.

use std::thread;
use std::time::{Duration, Instant};

use grobx::{Cancel, CheckerMode, Stop};

use crate::common::{P, dense, params, run};

/// An instance done in milliseconds.
fn mid() -> String {
    dense(4, 3, P, 1)
}

/// An instance that takes far longer than the bounds below.
fn long() -> String {
    dense(8, 3, P, 1)
}

fn stopped(text: &str, params: &grobx::Params) -> Stop {
    let outcome = grobx::groebner(text, params).expect("the call runs");
    assert_eq!(outcome.basis, None, "a stopped run has no basis");
    assert_eq!(outcome.cofactors, None);
    assert_eq!(outcome.certified, None);
    outcome.stopped.expect("the run stopped")
}

#[test]
fn a_deadline_that_has_passed_stops_the_run() {
    let mut p = params(1);
    p.deadline = Some(Duration::ZERO);
    assert_eq!(stopped(&mid(), &p), Stop::Deadline);
}

#[test]
fn a_short_deadline_stops_a_long_run_soon() {
    let mut p = params(1);
    p.deadline = Some(Duration::from_millis(200));
    let start = Instant::now();
    assert_eq!(stopped(&long(), &p), Stop::Deadline);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{elapsed:?}",
        elapsed = start.elapsed()
    );
}

#[test]
fn a_cancel_set_before_the_call_stops_it() {
    let cancel = Cancel::new();
    cancel.set();
    let mut p = params(1);
    p.cancel = Some(cancel);
    assert_eq!(stopped(&mid(), &p), Stop::Cancel);
}

#[test]
fn a_cancel_raised_from_another_thread_stops_the_run() {
    let cancel = Cancel::new();
    let mut p = params(1);
    p.cancel = Some(cancel.clone());
    let stopper = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        cancel.set();
        Instant::now()
    });
    let outcome = grobx::groebner(&long(), &p).expect("the call runs");
    let ended = Instant::now();
    let raised = stopper.join().unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel), "{outcome:?}");
    assert_eq!(outcome.basis, None);
    assert!(
        ended.duration_since(raised) < Duration::from_secs(5),
        "{took:?}",
        took = ended.duration_since(raised)
    );
}

#[test]
fn a_ceiling_below_the_resident_set_stops_the_run() {
    let mut p = params(1);
    p.memory = Some(1);
    assert_eq!(stopped(&mid(), &p), Stop::Memory);
}

#[test]
fn a_cancel_set_before_a_call_with_the_checker_stops_it() {
    let cancel = Cancel::new();
    cancel.set();
    let mut p = params(1);
    p.cancel = Some(cancel);
    p.checker = Some(CheckerMode::Exact);
    assert_eq!(stopped(&mid(), &p), Stop::Cancel);
}

#[test]
fn generous_bounds_change_nothing() {
    let free = run(&mid(), &params(1));
    let mut p = params(1);
    p.deadline = Some(Duration::from_secs(3600));
    p.cancel = Some(Cancel::new());
    p.memory = Some(1 << 40);
    assert_eq!(run(&mid(), &p), free);
}
