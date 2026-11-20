//! short-row-reduce-time, sub-item 1, the crate's test code: a plain round
//! whose matrix takes no bands gives the same basis whether its rows are
//! reduced as random combinations of blocks or each on its own with the
//! zero tests, over GF(p) and ℚ and in every order, on the edge cases,
//! and under a deadline; the round line's `mode` and `zero-rows`; and the
//! rounds of long rows keep the blocks' mode
//! (`iterations/0103-short-row-reduce-time/plan.md`, _Test code_, 1 to 6).

use std::time::Duration;

use crate::matrix::{BLOCK, MODE_FORCED, PROBE};
use crate::rebuilt_reducer_tests::{Call, call, reported};
use crate::report::RoundMode;
use crate::report_tests::Line;
use crate::shared_tests::plain_rounds;
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_binomials, random_sparse};
use crate::{MonomialOrder, Params};

/// The stage times of a round line, and the fields the two modes differ
/// in.
const DIFFER: [&str; 15] = [
    "select",
    "symbolic",
    "encode",
    "reduce",
    "backsub",
    "record",
    "decode",
    "update",
    "mode",
    "zero-rows",
    "zero",
    "reductions",
    "subtracted",
    "added",
    "kept",
];

/// The same call under the three settings of the forced mode: None,
/// Some(Blocks), Some(Rows).
fn triple(text: &str, params: &Params) -> [Call; 3] {
    let settings = [None, Some(RoundMode::Blocks), Some(RoundMode::Rows)];
    settings.map(|s| {
        MODE_FORCED.set(s);
        let c = call(text, params);
        MODE_FORCED.set(None);
        c
    })
}

/// The `round` lines of the plain run, each without the fields the modes
/// differ in.
fn rounds(lines: &[Line]) -> Vec<Vec<(String, String)>> {
    plain_rounds(lines)
        .iter()
        .map(|l| {
            l.fields
                .iter()
                .filter(|(k, _)| !DIFFER.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .collect()
}

/// Asserts the three calls agree: the bases are equal, the lines' kinds
/// are the same in the same order, and the plain round lines agree field
/// for field but those the modes differ in.
fn agree(name: &str, calls: &[Call; 3]) {
    let kinds = |l: &[Line]| l.iter().map(|l| l.kind.clone()).collect::<Vec<String>>();
    for (i, c) in calls.iter().enumerate().skip(1) {
        assert_eq!(calls[0].0, c.0, "{name}: the bases, setting {i}");
        assert_eq!(
            kinds(&calls[0].1),
            kinds(&c.1),
            "{name}: the lines, setting {i}"
        );
        assert_eq!(
            rounds(&calls[0].1),
            rounds(&c.1),
            "{name}: the round lines, setting {i}"
        );
    }
}

/// The systems of tests 1 and 2 over `p`: random binomial systems and
/// random sparse systems on five variables, seeds 1 to 4.
fn systems(p: u64) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for s in 1..=4 {
        out.push((format!("binomials-p{p}-{s}"), random_binomials(p, s)));
        out.push((format!("sparse5-p{p}-{s}"), random_sparse(5, p, s)));
    }
    out
}

/// The given rows of a round line. They bound the rows to reduce, the
/// given rows that are no pivot, from above, and from below by half: every
/// leading monomial of the given rows is a selected pair's lcm, at which
/// both rows of the pair lead, so at most half the given rows are pivots.
fn given(l: &Line) -> usize {
    l.get("given").unwrap().parse().unwrap()
}

#[test]
fn the_bases_agree_in_every_mode() {
    for p in [65521, 7] {
        for (name, text) in systems(p) {
            agree(&name, &triple(&text, &reported(MonomialOrder::Grevlex)));
        }
    }
}

/// Test 2: under the rows forced, a round of `BLOCK + PROBE` rows to
/// reduce or more names `mode=rows` with `zero-rows`, and a round of fewer
/// `mode=blocks` without; under the blocks forced every round
/// `mode=blocks`. The line shows the given rows, which bound the rows to
/// reduce (`given`): `rows` stands only where the given rows reach
/// `BLOCK + PROBE`, and `blocks` only where they stay below twice that.
#[test]
fn the_round_line_names_the_mode() {
    let params = reported(MonomialOrder::Grevlex);
    let least = BLOCK + PROBE;
    for (name, text) in systems(65521) {
        let [_, blocks, rows] = triple(&text, &params);
        let mut forced = 0;
        for l in plain_rounds(&rows.1) {
            let mode = l
                .get("mode")
                .unwrap_or_else(|| panic!("{name}: a plain round line names its mode"));
            let given = given(l);
            match mode {
                "rows" => {
                    assert!(
                        given >= least,
                        "{name}: a round of fewer than {least} rows takes the blocks' mode, given {given}"
                    );
                    assert!(
                        l.get("zero-rows").is_some(),
                        "{name}: zero-rows in the rows' mode"
                    );
                    forced += 1;
                }
                "blocks" => {
                    assert!(
                        given < 2 * least,
                        "{name}: forced rows on a round of {given} given rows"
                    );
                    assert!(
                        l.get("zero-rows").is_none(),
                        "{name}: no zero-rows in the blocks' mode"
                    );
                }
                other => panic!("{name}: the mode {other}"),
            }
        }
        // The binomial systems' large rounds take the rows' mode; the sparse
        // systems' rounds hold fewer than `least` rows to reduce.
        assert!(
            forced > 0 || !name.starts_with("binomials"),
            "{name}: some round holds enough rows to take the rows' mode"
        );
        for l in plain_rounds(&blocks.1) {
            assert_eq!(l.get("mode"), Some("blocks"), "{name}: forced blocks");
            assert!(
                l.get("zero-rows").is_none(),
                "{name}: no zero-rows in the blocks' mode"
            );
        }
    }
}

#[test]
fn over_q_and_in_other_orders() {
    for (name, text) in systems(0) {
        agree(&name, &triple(&text, &reported(MonomialOrder::Grevlex)));
    }
    let elim = "x,z\n7\nx+z,\nz^2+1\n";
    for order in [MonomialOrder::Lex, MonomialOrder::Block(vec![1])] {
        for (name, text) in [
            ("sparse5-p7-4".to_string(), random_sparse(5, 7, 4)),
            ("dense33-p7-4".to_string(), dense(3, 3, 7, 4)),
            ("elim".to_string(), elim.to_string()),
        ] {
            agree(
                &format!("{name} {order:?}"),
                &triple(&text, &reported(order.clone())),
            );
        }
    }
}

#[test]
fn the_edge_cases() {
    for p in [65521, 0] {
        for (i, system) in EDGE.iter().enumerate() {
            let text = format!("x,y,z\n{p}\n{system}\n");
            agree(
                &format!("edge {i} p{p}"),
                &triple(&text, &reported(MonomialOrder::Grevlex)),
            );
        }
    }
}

#[test]
fn long_rows_keep_the_blocks_mode() {
    let params = reported(MonomialOrder::Grevlex);
    for (name, text) in [
        ("dense33", dense(3, 3, 65521, 1)),
        ("dense43", dense(4, 3, 65521, 1)),
    ] {
        let [none, _, rows] = triple(&text, &params);
        for l in plain_rounds(&none.1) {
            assert_eq!(
                l.get("mode"),
                Some("blocks"),
                "{name}: no probe decides rows on long rows"
            );
        }
        for l in plain_rounds(&rows.1) {
            if l.get("bands") == Some("yes") {
                assert_eq!(
                    l.get("mode"),
                    Some("blocks"),
                    "{name}: a round with bands keeps the blocks' mode"
                );
            }
        }
    }
}

#[test]
fn a_deadline_inside_the_rows_mode() {
    let text = random_binomials(65521, 1);
    let mut params = reported(MonomialOrder::Grevlex);
    params.signatures = false;
    MODE_FORCED.set(Some(RoundMode::Rows));
    let unbounded = crate::groebner(&text, &params).unwrap();
    assert!(unbounded.stopped.is_none());
    for ms in [0, 1, 2, 4, 8, 16, 32] {
        let mut bounded = params.clone();
        bounded.deadline = Some(Duration::from_millis(ms));
        let outcome = crate::groebner(&text, &bounded).unwrap();
        match outcome.stopped {
            Some(_) => assert!(
                outcome.basis.is_none(),
                "{ms} ms: no basis where the deadline stopped the call"
            ),
            None => assert_eq!(
                outcome.basis, unbounded.basis,
                "{ms} ms: the unbounded call's basis"
            ),
        }
    }
    MODE_FORCED.set(None);
}
