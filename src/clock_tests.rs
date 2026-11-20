//! A call reads the clock only for a deadline or the call report, and one
//! that asks for either where there is no clock is `Error::Clock`: wasm32
//! in a browser, whose `Instant::now` panics.

use std::time::{Duration, Instant};

use crate::budget::{self, Budget};
use crate::{Error, MonomialOrder, Params, Threads};

fn params(deadline: Option<Duration>, report: bool) -> Params {
    let mut params = Params::new(MonomialOrder::Grevlex, Threads::All, 1);
    params.deadline = deadline;
    params.report = report;
    params
}

/// The start is read for a deadline, for the report, and for both, here,
/// where there is a clock, and not for neither.
#[test]
fn start_reads_the_clock_for_a_deadline_or_the_report() {
    let mut failures = Vec::new();
    let second = Some(Duration::from_secs(1));
    for (deadline, report, read) in [
        (None, false, false),
        (second, false, true),
        (None, true, true),
        (second, true, true),
    ] {
        let before = Instant::now();
        let start = budget::start(&params(deadline, report));
        match start {
            Some(start) if read && start >= before => {}
            None if !read => {}
            Some(_) | None => failures.push(format!("{deadline:?} {report}: {start:?}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// With no start, a deadline and the report are `Error::Clock`, and a
/// call with neither has its budget; with a start, every one has it.
#[test]
fn new_refuses_a_deadline_or_the_report_without_a_start() {
    let mut failures = Vec::new();
    let second = Some(Duration::from_secs(1));
    for (deadline, report) in [(None, false), (second, false), (None, true), (second, true)] {
        let p = params(deadline, report);
        let timed = deadline.is_some() || report;
        match Budget::new(None, &p) {
            Err(Error::Clock) if timed => {}
            Ok(_) if !timed => {}
            Ok(_) => failures.push(format!("{deadline:?} {report}: a budget without a start")),
            Err(e) => failures.push(format!("{deadline:?} {report}: {e:?} without a start")),
        }
        if let Err(e) = Budget::new(Instant::now(), &p) {
            failures.push(format!("{deadline:?} {report}: {e:?} with a start"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
