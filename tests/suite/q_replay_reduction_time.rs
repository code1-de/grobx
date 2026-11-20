//! q-replay-reduction-time, the item's test code through the public API:
//! the record `learn` returns is whole, and the calls over ℚ, whose lift
//! cuts its records, keep their basis
//! (`iterations/0098-q-replay-reduction-time/plan.md`, _Test code_, 8 and
//! 9).

use grobx::id::RowId;
use grobx::trace::{RowSource, Trace};
use grobx::{CheckerMode, Params};

use crate::common::systems::random_sparse;
use crate::common::{P, P31, dense, params_in, random_square, random_system};
use crate::large::over;
use crate::report::{of, report as lines_of};

/// The edge cases of plan.md: name and generators on x, y, z.
const EDGE: &[(&str, &str)] = &[
    ("zero", "0"),
    ("no generator", ""),
    ("unit", "x*y-1,\nx"),
    ("constant among others", "x^2+y,\nx*y+1,\n7"),
    ("already a basis", "x+y,\ny^2+1"),
    ("zero generator", "x*y-z,\n0,\ny^2-x"),
    ("eliminable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// Counts, per matrix the trace hears, the rows that made a new element.
#[derive(Default)]
struct Counts(Vec<usize>);

impl Trace for Counts {
    fn matrix(&mut self, _: &[RowSource]) {}

    fn counted(&mut self, rows: &[RowId]) {
        self.0.push(rows.len());
    }
}

/// The generated systems over GF(p), or ℚ where p is 0, and the edge
/// cases: name and text. `small` leaves out the dense cubic and the sparse
/// systems, which over ℚ with cofactors, or in lex and `block:1`, take
/// seconds.
fn systems(p: u64, small: bool) -> Vec<(String, String)> {
    let mut list = Vec::new();
    for seed in 0..6 {
        list.push((format!("random-{seed}"), random_system(p, seed)));
        list.push((format!("square-{seed}"), random_square(p, seed)));
    }
    if !small {
        list.push(("cubic-3".to_string(), dense(3, 3, p, 1)));
        for seed in 0..6 {
            list.push((format!("sparse4-{seed}"), random_sparse(4, p, seed)));
            list.push((format!("sparse5-{seed}"), random_sparse(5, p, seed)));
        }
    }
    for (name, generators) in EDGE {
        list.push((name.to_string(), format!("x,y,z\n{p}\n{generators}\n")));
    }
    list
}

fn plain(order: &str, signatures: bool) -> Params {
    let mut p = params_in(order, 1);
    p.signatures = signatures;
    p
}

#[test]
fn a_public_record_is_whole() {
    for (name, text) in systems(P, false) {
        let params = plain("grevlex", false);
        let mut learned = Counts::default();
        let (outcome, record) = grobx::learn(&text, &params, &mut learned)
            .unwrap_or_else(|e| panic!("{name}: {e:?}\n{text}"));
        let record = record.unwrap_or_else(|| panic!("{name}: no record"));
        let mut replayed = Counts::default();
        let there = over(&text, P31);
        let replay = grobx::replay(&there, &params, &record, &mut replayed)
            .unwrap_or_else(|e| panic!("{name}: the replay failed: {e:?}\n{text}"));
        assert!(replay.basis.is_some(), "{name}: the replay stopped");
        let unit = outcome.basis.as_deref() == Some("[1]:\n");
        let rounds = |c: &Counts| {
            let n = if unit {
                c.0.len()
            } else {
                c.0.len().saturating_sub(1)
            };
            c.0[..n].iter().sum::<usize>()
        };
        assert_eq!(
            rounds(&replayed),
            rounds(&learned),
            "{name}: the replay of a public record reduces every counted row"
        );
    }
}

#[test]
fn calls_over_q_skip_and_keep_their_basis() {
    let mut skipped = 0;
    for order in ["grevlex", "lex", "block:1"] {
        for (name, text) in systems(0, true) {
            for signatures in [false, true] {
                let name = format!("{name}, {order}, signatures {signatures}");
                let mut checked = plain(order, signatures);
                checked.cofactors = true;
                checked.checker = Some(CheckerMode::Exact);
                checked.report = true;
                let with = grobx::groebner(&text, &checked)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}\n{text}"));
                assert_eq!(with.certified, Some(true), "{name}: {with:?}");
                let mut reported = plain(order, signatures);
                reported.report = true;
                let without = grobx::groebner(&text, &reported)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}\n{text}"));
                assert_eq!(with.basis, without.basis, "{name}");
                for outcome in [&with, &without] {
                    let lines = lines_of(outcome);
                    for learn in of(&lines, "learn") {
                        if learn.get("outcome") == Some("recorded") {
                            skipped += learn.num("skipped");
                        } else {
                            assert_eq!(learn.get("skipped"), None, "{name}: {learn:?}");
                        }
                    }
                }
            }
        }
    }
    assert!(skipped > 0, "no call over ℚ skipped a row");
}
