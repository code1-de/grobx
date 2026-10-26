//! trace-replay, sub-item 1: `learn` against a run with the mode off, and a
//! stopped `learn` (plan.md, _Test code_).

use std::time::Duration;

use grobx::trace::{NoTrace, Record};
use grobx::{Cancel, Outcome, Params, Stop};

use crate::common::{PRIMES, dense, params, random_system};

/// The parameters of a measured call, with or without signatures and
/// cofactors.
fn with(seed: u64, signatures: bool, cofactors: bool) -> Params {
    let mut p = params(seed);
    p.signatures = signatures;
    p.cofactors = cofactors;
    p
}

/// The instances, name and text: the tiny one, two dense cubic systems, the edge
/// cases, and random systems.
fn instances() -> Vec<(String, String)> {
    let mut list: Vec<(String, String)> = vec![(
        "tiny".to_string(),
        "x,y\n65521\nx^2+y,\nx*y+1\n".to_string(),
    )];
    for (m, seed) in [(3, 1), (4, 2)] {
        list.push((format!("cubic-{m}"), dense(m, 3, 65521, seed)));
    }
    for (name, text) in EDGE {
        list.push((name.to_string(), text.to_string()));
    }
    for p in PRIMES {
        for seed in 0..2 {
            list.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
    }
    list
}

/// The edge cases of plan.md.
const EDGE: &[(&str, &str)] = &[
    ("zero", "x,y\n65521\n0\n"),
    ("no generator", "x,y\n65521\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n"),
    ("constant", "x,y\n65521\n1\n"),
    ("constant among others", "x,y\n65521\nx^2+y,\nx*y+1,\n7\n"),
    ("already a basis", "x,y\n65521\nx+y,\ny^2+1\n"),
    ("zero generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    ("eliminable", "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
];

fn learn(text: &str, params: &Params) -> (Outcome, Option<Record>) {
    grobx::learn(text, params, &mut NoTrace).unwrap_or_else(|e| panic!("{e:?}\n{text}"))
}

/// `learn` returns what a run returns but for the counts, and a record.
#[test]
fn learn_returns_what_a_run_returns() {
    for (name, text) in instances() {
        for signatures in [false, true] {
            for cof in [false, true] {
                let params = with(1, signatures, cof);
                let (outcome, record) = learn(&text, &params);
                let run = grobx::groebner(&text, &params).unwrap();
                assert_eq!(
                    Outcome {
                        stats: run.stats,
                        ..outcome.clone()
                    },
                    run,
                    "{name}, signatures {signatures}, cofactors {cof}: all but the counts"
                );
                assert!(record.is_some(), "{name}: a run with a basis has a record");
            }
        }
    }
}

#[test]
fn a_stopped_learn_has_no_record() {
    let text = dense(8, 3, 65521, 1);
    for signatures in [false, true] {
        let mut cancelled = with(1, signatures, false);
        let cancel = Cancel::new();
        cancel.set();
        cancelled.cancel = Some(cancel);
        let mut late = with(1, signatures, false);
        late.deadline = Some(Duration::ZERO);
        for (params, why) in [(cancelled, Stop::Cancel), (late, Stop::Deadline)] {
            let (outcome, record) = learn(&text, &params);
            assert_eq!(outcome.stopped, Some(why), "signatures {signatures}");
            assert_eq!(outcome.basis, None);
            assert!(record.is_none(), "a stopped learn run has no record");
        }
    }
}
