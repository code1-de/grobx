//! call-report, sub-item 1: the report of a call over GF(p) through the
//! crate's four entry points: its lines, its runs and rounds against the
//! outcome's stats, and an outcome that the report does not change
//! (`iterations/0064-call-report/plan.md`, _Test code_, 14 and 15; 5
//! without the zero ideal's `rounds=0`; 6 and 12 on generated systems with
//! no reference digest; 8, 9, 11, and 13 cut to the stats, the `call` line,
//! and the texts; and a test 10 of its own, on the `call` line's entry and
//! trace).

use std::time::Duration;

use grobx::id::RowId;
use grobx::trace::{NoTrace, RowSource, Trace};
use grobx::{Cancel, CheckerMode, Error, Outcome, Params, Stop};

use crate::common::{P, P31, dense, params, params_in, random_square};
use crate::report::{
    Line, assert_stage_times, assert_stats, of, report as lines_of, untimed, without_report,
};

/// `params` with the report on.
fn reported(seed: u64) -> Params {
    let mut p = params(seed);
    p.report = true;
    p
}

/// The outcome of `text` under `params`, which ask for the report, and
/// the same call's outcome without it: the two agree but for the report,
/// which the first holds and the second does not.
fn both(text: &str, params: &Params) -> (Outcome, Vec<Line>) {
    assert!(params.report);
    let on = grobx::groebner(text, params).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    let mut off_params = params.clone();
    off_params.report = false;
    let off = grobx::groebner(text, &off_params).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    assert_eq!(off.report, None, "a call that did not ask holds a report");
    assert_eq!(
        without_report(&on),
        off,
        "the report changed the outcome:\n{text}"
    );
    let lines = lines_of(&on);
    (on, lines)
}

/// The fields of the `call` line but `threads`, `echelon`, and `cpu`, read
/// back against `params`.
fn assert_call_line(line: &Line, params: &Params, entry: &str, trace: &str) {
    let on = |b: bool| if b { "on" } else { "off" };
    assert_eq!(line.kind, "call");
    assert_eq!(line.value("entry"), entry);
    assert_eq!(line.value("order"), params.order.to_string());
    assert_eq!(line.num("seed"), params.seed);
    assert_eq!(line.value("signatures"), on(params.signatures));
    assert_eq!(line.value("cofactors"), on(params.cofactors));
    assert_eq!(line.value("cancel"), on(params.cancel.is_some()));
    assert_eq!(line.value("trace"), trace);
    match params.deadline {
        None => assert_eq!(line.value("deadline"), "none"),
        Some(d) => assert_eq!(line.num("deadline"), d.as_millis() as u64),
    }
    match params.memory {
        None => assert_eq!(line.value("memory"), "none"),
        Some(m) => assert_eq!(line.num("memory"), m),
    }
    match params.checker {
        None => assert_eq!(line.value("checker"), "off"),
        Some(CheckerMode::Exact) => assert_eq!(line.value("checker"), "exact"),
        Some(CheckerMode::Cheap { points }) => {
            assert_eq!(line.value("checker"), "cheap");
            assert_eq!(line.num("points"), u64::from(points.get()));
        }
    }
}

/// The edge cases of `OBJECTIVES.md` on x, y, z: name and generators.
const EDGE: [(&str, &str); 7] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// Test 5: the edge cases over GF(65521), signatures on and off: each
/// report parses and agrees with the stats; the unit ideal writes
/// `constant`, and an ideal of no nonzero generator a run that reads none
/// and makes no element.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut p = reported(0xca11);
            p.signatures = signatures;
            let (outcome, lines) = both(&text, &p);
            let name = format!("{case}, signatures {signatures}");
            assert_call_line(&lines[0], &p, "groebner", "off");
            assert!(
                lines[1].is("read", &[("variables", "3"), ("characteristic", "65521")]),
                "{name}: {read:?}",
                read = lines[1]
            );
            assert_stats(&lines, &outcome.stats);
            assert_stage_times(&lines);
            let basis = outcome.basis.as_deref().unwrap();
            if basis == "[1]:\n" {
                assert_eq!(of(&lines, "constant").len(), 1, "{name}");
            }
            if basis == "[]:\n" {
                let inputs = of(&lines, "inputs");
                assert!(inputs.iter().all(|l| l.num("nonzero") == 0), "{name}");
                let ran = of(&lines, "ran");
                let last = ran.last().unwrap();
                assert!(
                    last.is("ran", &[("outcome", "basis"), ("elements", "0")]),
                    "{name}: {last:?}"
                );
            }
        }
    }
}

/// The instances of tests 6 and 14 over GF(65521): dense cubic systems on
/// three and four variables and random square systems.
fn instances() -> Vec<(String, String)> {
    let mut list: Vec<(String, String)> = (3..=4)
        .map(|m| (format!("cubic-{m}"), dense(m, 3, P, m as u64)))
        .collect();
    for seed in 1..=2 {
        list.push((format!("square-{seed}"), random_square(P, seed)));
    }
    list
}

/// Test 6: generated instances over GF(65521), signatures on and off: each
/// report parses, holds the outcome's stats, and gives every round eight
/// stage times that fit between its line and the line before it. The
/// first run starts the call over 65521 in grevlex, each `round` line names
/// its run's matrix sizes, and the last `ran` line counts the basis's
/// elements.
#[test]
fn generated_instances() {
    for (name, text) in instances() {
        for signatures in [true, false] {
            let mut p = reported(0x7e57);
            p.signatures = signatures;
            let (outcome, lines) = both(&text, &p);
            assert_stats(&lines, &outcome.stats);
            assert_stage_times(&lines);
            let first = &of(&lines, "run")[0];
            assert!(
                first.is(
                    "run",
                    &[("role", "call"), ("why", "start"), ("order", "grevlex")]
                ),
                "{name}: {first:?}"
            );
            assert_eq!(first.value("primes"), "65521", "{name}");
            let mut engine = "";
            for round in lines
                .iter()
                .filter(|l| l.kind == "run" || l.kind == "round")
            {
                if round.kind == "run" {
                    engine = round.value("engine");
                    continue;
                }
                assert!(round.num("rows") > 0, "{name}: {round:?}");
                assert!(round.num("columns") > 0, "{name}: {round:?}");
                assert!(
                    round.num("entries") >= round.num("rows"),
                    "{name}: {round:?}"
                );
                round.num("subtracted");
                round.num("added");
                round.num("degree");
                round.num("pairs");
                assert!(
                    matches!(round.value("bands"), "yes" | "no"),
                    "{name}: {round:?}"
                );
                if engine == "signature" {
                    assert!(round.num("dropped") <= round.num("pending"), "{round:?}");
                    assert!(round.get("left").is_none(), "{round:?}");
                } else {
                    round.num("left");
                    assert!(round.get("pending").is_none(), "{round:?}");
                }
            }
            let ran = of(&lines, "ran");
            let last = ran.last().unwrap();
            let elements = outcome.basis.as_deref().unwrap().matches(",\n").count() as u64 + 1;
            assert_eq!(last.num("elements"), elements, "{name}: {last:?}");
            if signatures && last.value("outcome") == "basis" && ran.len() == 1 {
                for key in [
                    "considered",
                    "equal",
                    "high",
                    "low",
                    "syzygy",
                    "koszul",
                    "queued",
                    "past-format",
                ] {
                    last.num(key);
                }
            }
        }
    }
}

/// Test 8: the widths over GF(65521). A term past `Dense8`'s 127 at the
/// read, a pair's lcm past it in the run, a lex staircase past 127
/// monomials, and an update past it after many rounds each give a report
/// that holds the outcome's stats. A degree past 2^31 − 1 is an error and
/// returns no report.
#[test]
fn the_widths() {
    // The last text was found by a search over random binomials: the
    // 8-bit run's update overflows after round 20.
    for text in [
        "x,y\n65521\nx^200-y,\ny^2-1\n",
        "x,y\n65521\nx^100*y-1,\nx*y^100-1\n",
        "x,y,z\n65521\nx^39*y^16*z^22-x^44*y^41*z^33,\n\
         x^29*y^15*z^41-6*x^3*y^10*z^7,\nx^30*y^15*z^24-4*x^34*y^6*z^36\n",
    ] {
        for signatures in [true, false] {
            let mut p = reported(1);
            p.signatures = signatures;
            let (outcome, lines) = both(text, &p);
            assert_stats(&lines, &outcome.stats);
        }
    }
    let (outcome, lines) = both("x,y\n65521\nx^8-1,\ny^16-1\n", &{
        let mut p = params_in("lex", 1);
        p.report = true;
        p
    });
    assert_eq!(
        outcome.basis.as_deref(),
        Some("[y^16+65520,\nx^8+65520]:\n")
    );
    assert_stats(&lines, &outcome.stats);

    let got = grobx::groebner("x\n65521\nx^2147483648-1\n", &reported(1));
    assert!(
        matches!(got, Err(Error::Overflow { line: Some(3) })),
        "{got:?}"
    );
}

/// Test 9: lex and a block order over GF(65521), on a zero-dimensional
/// ideal and on a positive-dimensional one in lex: each report holds the
/// outcome's stats, and the `call` line names the order.
#[test]
fn lex_and_block() {
    let lex = |seed| {
        let mut p = params_in("lex", seed);
        p.report = true;
        p
    };
    for text in [dense(3, 3, P, 1), "x,y,z\n65521\nx*y,\ny*z\n".to_string()] {
        let (outcome, lines) = both(&text, &lex(3));
        assert_stats(&lines, &outcome.stats);
        assert!(
            lines[0].is("call", &[("order", "lex")]),
            "{call:?}",
            call = lines[0]
        );
    }

    let mut p = params_in("block:1", 5);
    p.report = true;
    let (outcome, lines) = both(&dense(3, 3, P, 2), &p);
    assert_stats(&lines, &outcome.stats);
    assert!(
        lines[0].is("call", &[("order", "block:1")]),
        "{call:?}",
        call = lines[0]
    );
}

/// A trace that listens and keeps nothing.
struct Listening;

impl Trace for Listening {
    fn matrix(&mut self, _: &[RowSource]) {}
    fn counted(&mut self, _: &[RowId]) {}
}

/// Test 10: the `call` line names a listening trace through
/// `groebner_with`, and the entry `learn`, whose first run's role is
/// `learn`.
#[test]
fn the_call_line_names_the_entry_and_the_trace() {
    let text = dense(4, 3, P, 1);
    let mut params = reported(0x30de);
    params.signatures = false;
    let heard = grobx::groebner_with(&text, &params, &mut Listening).unwrap();
    let lines = lines_of(&heard);
    assert_call_line(&lines[0], &params, "groebner", "listens");
    assert_stats(&lines, &heard.stats);

    let (learned, _) = grobx::learn(&text, &params, &mut NoTrace).unwrap();
    let lines = lines_of(&learned);
    assert_call_line(&lines[0], &params, "learn", "off");
    assert!(
        of(&lines, "run")[0].is("run", &[("role", "learn")]),
        "{run:?}",
        run = of(&lines, "run")[0]
    );
    assert_stats(&lines, &learned.stats);
}

/// Test 11: `learn` and `replay` of a dense cubic system over 65521 and
/// 2^31 − 1: the learn's report names its role, and the replay's names
/// its entry and one run of the role `replay`. The replay returns the
/// learned basis, and the outcome the replay returns without the report.
#[test]
fn learn_and_replay() {
    for p in [P, P31] {
        let text = dense(4, 3, p, 1);
        for signatures in [true, false] {
            let mut params = reported(0x1ea2);
            params.signatures = signatures;
            let (learned, record) = grobx::learn(&text, &params, &mut NoTrace).unwrap();
            let record = record.expect("a record");
            let lines = lines_of(&learned);
            assert_stats(&lines, &learned.stats);
            assert!(of(&lines, "run")[0].is("run", &[("role", "learn")]));
            let replayed = grobx::replay(&text, &params, &record, &mut NoTrace).unwrap();
            assert_eq!(replayed.basis, learned.basis, "over {p}");
            let lines = lines_of(&replayed);
            assert_call_line(&lines[0], &params, "replay", "off");
            assert_stats(&lines, &replayed.stats);
            let runs = of(&lines, "run");
            assert_eq!(runs.len(), 1, "{runs:?}");
            assert!(runs[0].is("run", &[("role", "replay")]), "{runs:?}");
            let mut off = params.clone();
            off.report = false;
            let without = grobx::replay(&text, &off, &record, &mut NoTrace).unwrap();
            assert_eq!(without_report(&replayed), without, "over {p}");
        }
    }
}

/// Test 12: bounds. A deadline of 50 ms stops a long call, and a cancel
/// raised before the call stops it before its first round: each report
/// ends with `stop` and `end result=stopped`, and the last `ran` holds the
/// stopped run's counts.
#[test]
fn bounds_stop_the_call() {
    let text = dense(8, 3, P, 1);
    let mut p = reported(0xb0d);
    p.deadline = Some(Duration::from_millis(50));
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(
        outcome.stopped,
        Some(Stop::Deadline),
        "{stats:?}",
        stats = outcome.stats
    );
    let lines = lines_of(&outcome);
    assert_call_line(&lines[0], &p, "groebner", "off");
    assert_stats(&lines, &outcome.stats);
    let last = *of(&lines, "ran").last().unwrap();
    assert!(last.is("ran", &[("outcome", "stop")]), "{last:?}");

    let cancel = Cancel::new();
    cancel.set();
    let mut p = reported(0xb0d);
    p.cancel = Some(cancel);
    let outcome = grobx::groebner(&text, &p).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel));
    let lines = lines_of(&outcome);
    assert_call_line(&lines[0], &p, "groebner", "off");
    assert_stats(&lines, &outcome.stats);
    assert!(
        lines.iter().all(|l| l.kind != "round"),
        "a round before the first check"
    );
}

/// Test 13: cofactors and the exact checker on a dense cubic system over 65521:
/// the run names its cofactors, and the basis text, the cofactor text, and
/// the checker each write their line after the last run's `ran`, with the
/// texts' lengths.
#[test]
fn cofactors_and_the_checker() {
    let mut p = reported(0xc0f);
    p.cofactors = true;
    p.checker = Some(CheckerMode::Exact);
    let (outcome, lines) = both(&dense(4, 3, P, 1), &p);
    assert_eq!(outcome.certified, Some(true));
    assert_call_line(&lines[0], &p, "groebner", "off");
    assert_stats(&lines, &outcome.stats);
    let run = &of(&lines, "run")[0];
    assert!(run.is("run", &[("cofactors", "on")]), "{run:?}");
    let n = lines.len();
    let tail: Vec<&str> = lines[n - 5..].iter().map(|l| l.kind.as_str()).collect();
    assert_eq!(tail, ["ran", "text", "text", "check", "end"], "{lines:?}");
    let basis = &lines[n - 4];
    assert!(basis.is("text", &[("kind", "basis")]), "{basis:?}");
    assert_eq!(
        basis.num("bytes"),
        outcome.basis.as_deref().unwrap().len() as u64
    );
    let cofactors = &lines[n - 3];
    assert!(
        cofactors.is("text", &[("kind", "cofactors")]),
        "{cofactors:?}"
    );
    assert_eq!(
        cofactors.num("bytes"),
        outcome.cofactors.as_deref().unwrap().len() as u64
    );
    let check = &lines[n - 2];
    assert!(
        check.is("check", &[("mode", "exact"), ("verdict", "certified")]),
        "{check:?}"
    );
    check.secs("time");
    basis.secs("time");
}

/// Test 14: the report off. Each of the four entry points returns no
/// report where the call does not ask for one, and the instances of test 6
/// under two seeds return the same outcome with it as without it.
#[test]
fn the_report_off() {
    let text = dense(4, 3, P, 1);
    let p = params(9);
    assert!(!p.report, "Params::new leaves the report off");
    assert_eq!(grobx::groebner(&text, &p).unwrap().report, None);
    assert_eq!(
        grobx::groebner_with(&text, &p, &mut Listening)
            .unwrap()
            .report,
        None
    );
    let (learned, record) = grobx::learn(&text, &p, &mut NoTrace).unwrap();
    assert_eq!(learned.report, None);
    let replayed = grobx::replay(&text, &p, &record.unwrap(), &mut NoTrace).unwrap();
    assert_eq!(replayed.report, None);
    for (_, text) in instances() {
        for seed in [2, u64::MAX] {
            both(&text, &reported(seed));
        }
    }
}

/// Test 15: the same call twice with the report on writes the same report
/// once every time is taken out.
#[test]
fn the_report_is_the_calls() {
    for (name, text) in [
        ("cubic-4", dense(4, 3, P, 1)),
        ("square-1", random_square(P, 1)),
    ] {
        let a = grobx::groebner(&text, &reported(4)).unwrap();
        let b = grobx::groebner(&text, &reported(4)).unwrap();
        assert_eq!(untimed(&lines_of(&a)), untimed(&lines_of(&b)), "{name}");
    }
}
