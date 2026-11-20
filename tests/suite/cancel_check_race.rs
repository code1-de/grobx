//! cancel-check-race, the item's test code: a cancel raised by the call
//! report's writer as the call writes its cofactor text, after its texts
//! and before its check, is read inside the check, whatever the load
//! (`iterations/0114-cancel-check-race/plan.md`, _Test code_, 1 and 2).

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use grobx::{Cancel, CheckerMode, Outcome, Params, ReportSink, Stop};

use crate::checker_stop_latency::{LATENCY, cheap, checked, long};
use crate::common::{EDGE_GENERATORS, P, run};

/// What the call report's line of the cofactor text holds: the call's
/// last line before its check, over GF(p) and over ℚ.
const COFACTORS_LINE: &str = " text kind=cofactors ";

/// When a `RaiseAtCheck` raised its flag, None until it has.
type Raised = Arc<Mutex<Option<Instant>>>;

// cancel-check-race.1 design-checked -- the cancel is raised by the call report's writer, on the call's own thread, as it takes a line the call writes before its check, not from another thread at a share of an unbounded call's time with the assertion allowing the interval between full checks (log 0114-1)
// cancel-check-race.2 design-checked -- the line is the cofactor text's, the last before the check over GF(p) and over ℚ, not the basis text's, after which the cofactor text's writer reads the budget over GF(p) (not measured)
/// A writer of the call report that raises `cancel` as it takes the line
/// of the cofactor text, and keeps in `raised` the time just before the
/// raise.
struct RaiseAtCheck {
    cancel: Cancel,
    raised: Raised,
    /// The bytes taken since the last line break.
    line: Vec<u8>,
}

impl Write for RaiseAtCheck {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.line.extend_from_slice(buf);
        while let Some(end) = self.line.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.line.drain(..=end).collect();
            let line = String::from_utf8(line).expect("the report is UTF-8");
            if line.contains(COFACTORS_LINE) {
                *self.raised.lock().unwrap() = Some(Instant::now());
                self.cancel.set();
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `checked(mode, seed)` with a cancel flag, the report on, and a
/// `RaiseAtCheck` as its writer; and when the writer raised the flag.
fn raising(mode: CheckerMode, seed: u64) -> (Params, Raised) {
    let cancel = Cancel::new();
    let raised = Arc::new(Mutex::new(None));
    let mut p = checked(mode, seed);
    p.cancel = Some(cancel.clone());
    p.report = true;
    p.report_sink = Some(ReportSink::new(RaiseAtCheck {
        cancel,
        raised: Arc::clone(&raised),
        line: Vec::new(),
    }));
    (p, raised)
}

/// Asserts that `outcome` is a stop by the cancel with no part of a
/// result: no basis, no cofactors, and no verdict.
fn assert_cancelled(case: &str, outcome: &Outcome) {
    assert_eq!(outcome.stopped, Some(Stop::Cancel), "{case}");
    assert_eq!(outcome.basis, None, "{case}");
    assert_eq!(outcome.cofactors, None, "{case}");
    assert_eq!(outcome.certified, None, "{case}");
    assert_eq!(outcome.flaw, None, "{case}");
}

// cancel-check-race.3 design-checked -- the test asserts the stop by the cancel and no part of a result, with no branch that accepts a returned basis, which only a check that reads no flag returns here (log 0114-1)
// cancel-check-race.4 design-checked -- the test runs test 5's three calls, the exact check over ℚ and over GF(65521) and the cheap one over ℚ, not the first alone (log 0114-1)
/// Test 1, the item's acceptance criterion: a cancel raised as each of
/// `checker_stop_latency.rs`' test 5's calls writes its cofactor text
/// stops the call with `Cancel`, with no basis, no cofactors, and no
/// verdict, within 0.5 s of the raise.
#[test]
fn a_cancel_raised_before_the_check_stops_it() {
    for (name, text, mode) in long() {
        let case = format!("{name}, {mode:?}");
        let (p, raised) = raising(mode, 1);
        let outcome = grobx::groebner(&text, &p).expect("the call runs");
        let ended = Instant::now();
        let raised = raised
            .lock()
            .unwrap()
            .expect("the call wrote its cofactor text");
        assert_cancelled(&case, &outcome);
        let late = ended.saturating_duration_since(raised);
        assert!(late < LATENCY, "{case}: returned {late:?} after the raise");
    }
}

// cancel-check-race.5 design-checked -- on the edge cases either the stop or the whole result is accepted, not the outcome each mode gave in the prototype, which counts the check's polls (log 0114-1)
/// Test 2: on the edge cases over GF(65521) and ℚ, in the exact mode and
/// at 100 points, a cancel raised as the call writes its cofactor text
/// either stops the call with `Cancel` and no part of a result, or leaves
/// the basis and the cofactors of the call without it, certified, where
/// the check ended before a full check of the budget. Where the writer
/// raised nothing, the call returns the second.
#[test]
fn the_edge_cases_stop_or_return_whole() {
    for p in [P, 0] {
        for (name, generators) in EDGE_GENERATORS {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for mode in [CheckerMode::Exact, cheap(100)] {
                let case = format!("{name} over {p}, {mode:?}");
                let free = run(&text, &checked(mode, 1));
                assert_eq!(
                    free.certified,
                    Some(true),
                    "{case}: {flaw:?}",
                    flaw = free.flaw
                );
                let (params, raised) = raising(mode, 1);
                let outcome = grobx::groebner(&text, &params).expect("the call runs");
                let raised = raised.lock().unwrap().is_some();
                if raised && outcome.stopped.is_some() {
                    assert_cancelled(&case, &outcome);
                } else {
                    assert_eq!(outcome.stopped, None, "{case}");
                    assert_eq!(outcome.basis, free.basis, "{case}");
                    assert_eq!(outcome.cofactors, free.cofactors, "{case}");
                    assert_eq!(outcome.certified, Some(true), "{case}");
                    assert_eq!(outcome.flaw, None, "{case}");
                }
            }
        }
    }
}
