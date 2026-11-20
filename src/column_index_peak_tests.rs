//! column-index-peak, the item's test code in the crate: every call with
//! its matrices' columns in 2 bytes where the round's monomials allow
//! returns what the same call returns with every matrix in 4 bytes, over
//! GF(p) and ℚ, the signatures off and on, in every order, with
//! cofactors, on the edge cases, under the test profile's bounds that
//! widen a build midway, and under a deadline
//! (`iterations/0115-column-index-peak/plan.md`, _Test code_, 9 to 13).
//!
//! A pair of calls is the same call with `NARROW_BOUND` at Some(0), every
//! matrix wide, and unset; the two agree where their bases are equal,
//! their cofactor texts where tracked, their reports' lines are of the
//! same kinds in the same order, and their round and inter lines agree
//! field for field but `index` and the times.

use std::time::Duration;

use crate::call::built;
use crate::matrix::NARROW_BOUND;
use crate::rebuilt_reducer_tests::{TIMES, reported};
use crate::report_tests::{Line, lines};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_sparse};
use crate::{CheckerMode, MonomialOrder, Params, Stop};

/// What a call returns: the basis text, the cofactor text, and the
/// report's lines.
type Call = (Option<String>, Option<String>, Vec<Line>);

/// The call of `text` under `bound`, which must not stop.
fn call(text: &str, params: &Params, bound: Option<usize>) -> Call {
    NARROW_BOUND.set(bound);
    let outcome = crate::groebner(text, params).unwrap_or_else(|e| panic!("{e:?} on\n{text}"));
    NARROW_BOUND.set(None);
    assert!(outcome.stopped.is_none(), "the call stopped: {outcome:?}");
    (
        outcome.basis,
        outcome.cofactors,
        lines(outcome.report.as_deref().unwrap()),
    )
}

/// The round and inter lines, each without its `index` field and its
/// times.
fn matrices(lines: &[Line]) -> Vec<Vec<(String, String)>> {
    lines
        .iter()
        .filter(|l| l.kind == "round" || l.kind == "inter")
        .map(|l| {
            l.fields
                .iter()
                .filter(|(k, _)| k != "index" && !TIMES.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .collect()
}

/// Asserts the two calls agree.
fn agree(name: &str, wide: &Call, narrow: &Call) {
    assert_eq!(wide.0, narrow.0, "{name}: the bases");
    assert_eq!(wide.1, narrow.1, "{name}: the cofactor texts");
    let kinds = |l: &[Line]| l.iter().map(|l| l.kind.clone()).collect::<Vec<String>>();
    assert_eq!(kinds(&wide.2), kinds(&narrow.2), "{name}: the lines");
    assert_eq!(
        matrices(&wide.2),
        matrices(&narrow.2),
        "{name}: the round and inter lines"
    );
}

/// The `index` and `columns` of every round and inter line, in order.
fn widths(lines: &[Line]) -> Vec<(usize, usize)> {
    lines
        .iter()
        .filter(|l| l.kind == "round" || l.kind == "inter")
        .map(|l| {
            let field = |k: &str| {
                l.get(k)
                    .unwrap_or_else(|| panic!("a {kind} line names {k}", kind = l.kind))
                    .parse::<usize>()
                    .unwrap()
            };
            (field("index"), field("columns"))
        })
        .collect()
}

/// The pair of calls of `text`: every matrix wide, and unbounded.
fn pair(name: &str, text: &str, params: &Params) -> (Call, Call) {
    let wide = call(text, params, Some(0));
    let narrow = call(text, params, None);
    agree(name, &wide, &narrow);
    assert!(
        widths(&wide.2).iter().all(|&(index, _)| index == 4),
        "{name}: every matrix 4 bytes under the bound 0"
    );
    (wide, narrow)
}

#[test]
fn the_calls_stay() {
    let mut narrow_rounds = 0;
    for p in [65521, 7, 0] {
        for s in 1..=4 {
            for (system, text) in [
                ("sparse4", random_sparse(4, p, s)),
                ("sparse5", random_sparse(5, p, s)),
                ("dense33", dense(3, 3, p, s)),
            ] {
                for signatures in [false, true] {
                    let mut params = reported(MonomialOrder::Grevlex);
                    params.signatures = signatures;
                    let name = format!("{system}-p{p}-{s}, signatures {signatures}");
                    let (_, narrow) = pair(&name, &text, &params);
                    let w = widths(&narrow.2);
                    assert!(
                        w.iter().all(|&(index, _)| index == 2 || index == 4),
                        "{name}: an index of 2 or 4 bytes"
                    );
                    narrow_rounds += w.iter().filter(|&&(index, _)| index == 2).count();
                }
            }
        }
    }
    assert!(narrow_rounds > 0, "some matrix of the set 2 bytes");
}

#[test]
fn the_bound_widens_mid_build() {
    let params = reported(MonomialOrder::Grevlex);
    for p in [65521, 0] {
        for s in [1, 2] {
            let text = random_sparse(5, p, s);
            let wide = call(&text, &params, Some(0));
            // Beside the plan's bounds, each matrix's columns c of the wide
            // call and one fewer: a round of exactly the bound's monomials,
            // and one more.
            let mut bounds = vec![1, 8, 64, 300];
            for (_, columns) in widths(&wide.2) {
                bounds.extend([columns.saturating_sub(1).max(1), columns.max(1)]);
            }
            bounds.sort_unstable();
            bounds.dedup();
            for bound in bounds {
                let name = format!("sparse5-p{p}-{s} at {bound}");
                let bounded = call(&text, &params, Some(bound));
                agree(&name, &wide, &bounded);
                for (index, columns) in widths(&bounded.2) {
                    assert!(
                        index == 4 || columns <= bound,
                        "{name}: a matrix of {columns} columns in 2 bytes"
                    );
                }
            }
        }
    }
}

#[test]
fn other_orders_and_cofactors() {
    let eliminable = "x,y,z\n7\nx+z,\nz^2+1\n";
    for order in ["lex", "block:1"] {
        let params = reported(order.parse().unwrap());
        for (name, text) in [
            ("dense33-p7-4", dense(3, 3, 7, 4)),
            ("eliminable", eliminable.to_string()),
        ] {
            pair(&format!("{name} in {order}"), &text, &params);
        }
    }
    for p in [65521, 0] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        if built(&params).is_err() {
            continue;
        }
        let name = format!("sparse4-p{p}-1 with cofactors");
        let (wide, _) = pair(&name, &random_sparse(4, p, 1), &params);
        assert!(wide.1.is_some(), "{name}: the cofactor text");
    }
}

#[test]
fn the_edge_cases() {
    let params = reported(MonomialOrder::Grevlex);
    for p in [65521, 0] {
        for (k, gens) in EDGE.iter().enumerate() {
            let text = format!("x,y,z\n{p}\n{gens}\n");
            pair(&format!("edge {k} over {p}"), &text, &params);
        }
    }
}

#[test]
fn a_stop_inside_a_narrow_build() {
    let text = random_sparse(5, 65521, 1);
    let params = reported(MonomialOrder::Grevlex);
    let full = call(&text, &params, None).0.unwrap();
    for bound in [None, Some(8)] {
        for ms in [0, 1, 2, 4, 8, 16, 32] {
            let mut params = reported(MonomialOrder::Grevlex);
            params.deadline = Some(Duration::from_millis(ms));
            NARROW_BOUND.set(bound);
            let outcome = crate::groebner(&text, &params).unwrap();
            NARROW_BOUND.set(None);
            let name = format!("{ms} ms at {bound:?}");
            match outcome.stopped {
                Some(Stop::Deadline) => assert!(outcome.basis.is_none(), "{name}: {outcome:?}"),
                None => assert_eq!(outcome.basis.unwrap(), full, "{name}"),
                Some(Stop::Cancel | Stop::Memory) => panic!("{name}: {outcome:?}"),
            }
        }
    }
}
