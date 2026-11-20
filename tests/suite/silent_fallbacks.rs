//! silent-fallbacks, the item's integration tests: a report sink that
//! refuses a write leaves its error in the outcome, the call otherwise
//! unchanged, and over ℚ the lifted cofactors join their elements
//! (`iterations/0094-silent-fallbacks/plan.md`, _Test code_, 1 to 3).

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use grobx::{Outcome, Params, ReportSink};

use crate::common::{EDGE_GENERATORS, P, dense, params, run};
use crate::q::accepted_q;
use crate::report::{report as lines_of, untimed};

/// The message of the error a `Refusing` writer returns.
const REFUSED: &str = "the sink refuses the write";

/// The seed of every call of the tests.
const SEED: u64 = 0x5f5f;

/// A writer that keeps what it takes in `took`: every write where `left`
/// is None, else `left` writes, refusing every write after them.
struct Refusing {
    took: Arc<Mutex<Vec<u8>>>,
    left: Option<usize>,
}

impl Write for Refusing {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.left {
            Some(0) => return Err(io::Error::other(REFUSED)),
            Some(left) => *left -= 1,
            None => {}
        }
        self.took.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `params` with a `Refusing` sink of `left` writes, and what the sink
/// takes.
fn with_sink(params: &Params, left: Option<usize>) -> (Params, Arc<Mutex<Vec<u8>>>) {
    let took = Arc::new(Mutex::new(Vec::new()));
    let mut p = params.clone();
    p.report_sink = Some(ReportSink::new(Refusing {
        took: Arc::clone(&took),
        left,
    }));
    (p, took)
}

/// What the sink took, as text.
fn taken(took: &Arc<Mutex<Vec<u8>>>) -> String {
    String::from_utf8(took.lock().unwrap().clone()).expect("the report is UTF-8")
}

/// The cases of tests 1 and 2: the edge cases of `OBJECTIVES.md` and a
/// dense cubic system on three variables, over GF(65521) and over ℚ.
fn cases() -> Vec<(String, String)> {
    let mut list = Vec::new();
    for p in [P, 0] {
        for (case, generators) in EDGE_GENERATORS {
            list.push((
                format!("{case} over {p}"),
                format!("x,y,z\n{p}\n{generators}\n"),
            ));
        }
        list.push((format!("dense(3, 3, {p}, 1)"), dense(3, 3, p, 1)));
    }
    list
}

/// The outcome of `text` under `params`, a call that returns one.
fn call(case: &str, text: &str, params: &Params) -> Outcome {
    grobx::groebner(text, params).unwrap_or_else(|e| panic!("{case}: {e:?}\n{text}"))
}

/// `got` and `want` agree in all but the sink's error and the report's
/// times.
fn assert_same(case: &str, got: &Outcome, want: &Outcome) {
    assert_eq!(got.basis, want.basis, "{case}: the basis");
    assert_eq!(got.stopped, want.stopped, "{case}: the stop");
    assert_eq!(got.cofactors, want.cofactors, "{case}: the cofactors");
    assert_eq!(got.certified, want.certified, "{case}: the verdict");
    assert_eq!(got.flaw, want.flaw, "{case}: the flaw");
    assert_eq!(got.stats, want.stats, "{case}: the stats");
    assert_eq!(
        untimed(&lines_of(got)),
        untimed(&lines_of(want)),
        "{case}: the report"
    );
}

/// Test 1: silent-fallbacks.5. A sink that refuses its first write, the
/// `call` line, or its third: the outcome's `sink_error` holds the
/// writer's message, the sink took the lines before the refused one and
/// no more, and the call is the call without a sink.
#[test]
fn a_refused_write_is_the_outcomes() {
    for (case, text) in cases() {
        for cofactors in [false, true] {
            let mut base = params(SEED);
            base.report = true;
            base.cofactors = cofactors;
            let without = call(&case, &text, &base);
            for k in [1, 3] {
                let name = format!("{case}, cofactors {cofactors}, write {k} refused");
                let (p, took) = with_sink(&base, Some(k - 1));
                let got = call(&name, &text, &p);
                assert_eq!(got.sink_error.as_deref(), Some(REFUSED), "{name}");
                let report = got.report.as_deref().expect("the report");
                let before: String = report.split_inclusive('\n').take(k - 1).collect();
                assert_eq!(taken(&took), before, "{name}: the lines the sink took");
                assert_same(&name, &got, &without);
            }
        }
    }
}

/// Test 2: silent-fallbacks.5. A sink that takes every write holds the
/// outcome's report, and the outcome no sink error; a call without a sink
/// has none; a sink with the report off takes nothing and the outcome has
/// none.
#[test]
fn a_sink_that_takes_every_line() {
    for (case, text) in cases() {
        let mut on = params(SEED);
        on.report = true;
        let without = call(&case, &text, &on);
        assert_eq!(without.sink_error, None, "{case}: no sink");
        let (p, took) = with_sink(&on, None);
        let got = call(&case, &text, &p);
        assert_eq!(got.sink_error, None, "{case}: a sink that takes every line");
        assert_eq!(Some(taken(&took)), got.report, "{case}: the sink's lines");
        assert_same(&case, &got, &without);
        let (p, took) = with_sink(&params(SEED), Some(0));
        let off = call(&case, &text, &p);
        assert_eq!(off.sink_error, None, "{case}: the report off");
        assert_eq!(off.report, None, "{case}: the report off");
        assert_eq!(taken(&took), "", "{case}: the report off");
    }
}

/// Test 3: silent-fallbacks.7. Over ℚ, the edge cases of `OBJECTIVES.md`
/// and a dense cubic system with cofactors: the exact checker accepts the
/// lifted basis and cofactors, and the basis is the call's without
/// cofactors.
#[test]
fn cofactors_over_q_join_their_elements() {
    let mut list: Vec<(String, String)> = EDGE_GENERATORS
        .iter()
        .map(|(case, generators)| {
            (
                format!("{case} over ℚ"),
                format!("x,y,z\n0\n{generators}\n"),
            )
        })
        .collect();
    list.push(("dense(3, 3, 0, 1)".to_string(), dense(3, 3, 0, 1)));
    for (case, text) in list {
        let lifted = accepted_q(&case, &text, "grevlex");
        let plain = run(&text, &params(SEED));
        assert_eq!(lifted.basis, plain.basis, "{case}: without cofactors");
    }
}
