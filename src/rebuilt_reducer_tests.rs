//! rebuilt-reducer-time, sub-item 1, the crate's test code: the matrices
//! stay as the build without kept rows makes them, over GF(p) and ℚ and in
//! every order, under the test profile's bound, on the edge cases, and
//! under a deadline; the round line's `kept`; and a kept row taken and the
//! rest compacted at the matrix level
//! (`iterations/0102-rebuilt-reducer-time/plan.md`, _Test code_, 1 to 7).

use std::num::NonZeroUsize;
use std::time::Duration;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::field::{Field, Gf31};
use crate::id::{MonoId, RowId, RowSource};
use crate::matrix::{self, Buffers, KEPT_BOUND, KEPT_OFF, Matrix};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::report_tests::{Line, lines};
use crate::shared_tests::reported_budget;
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_sparse};
use crate::table::Table;
use crate::{MonomialOrder, Params, Stop, Threads};

/// The stage times of a round line, which the two calls of a pair differ
/// in.
pub(crate) const TIMES: [&str; 8] = [
    "select", "symbolic", "encode", "reduce", "backsub", "record", "decode", "update",
];

/// The parameters of a call in `order` on one thread under seed 1, the
/// report on, the signatures off.
pub(crate) fn reported(order: MonomialOrder) -> Params {
    let mut params = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), 1);
    params.report = true;
    params.signatures = false;
    params
}

/// What a call returns: the basis text and the report's lines.
pub(crate) type Call = (Option<String>, Vec<Line>);

/// The basis text and the report's lines of a call of `text` that must
/// not stop.
pub(crate) fn call(text: &str, params: &Params) -> Call {
    let outcome = crate::groebner(text, params).unwrap_or_else(|e| panic!("{e:?} on\n{text}"));
    assert!(outcome.stopped.is_none(), "the call stopped: {outcome:?}");
    (outcome.basis, lines(outcome.report.as_deref().unwrap()))
}

/// The same call with the keeping off and then on.
pub(crate) fn pair(text: &str, params: &Params) -> (Call, Call) {
    KEPT_OFF.set(true);
    let off = call(text, params);
    KEPT_OFF.set(false);
    let on = call(text, params);
    (off, on)
}

/// The `round` lines, each without its `kept` field and its times.
fn rounds(lines: &[Line]) -> Vec<Vec<(String, String)>> {
    lines
        .iter()
        .filter(|l| l.kind == "round")
        .map(|l| {
            l.fields
                .iter()
                .filter(|(k, _)| k != "kept" && !TIMES.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .collect()
}

/// Asserts the two calls agree: the bases are equal, the lines' kinds are
/// the same in the same order, and the round lines agree field for field
/// but `kept` and the times.
pub(crate) fn agree(name: &str, off: &Call, on: &Call) {
    assert_eq!(off.0, on.0, "{name}: the bases");
    let kinds = |l: &[Line]| l.iter().map(|l| l.kind.clone()).collect::<Vec<String>>();
    assert_eq!(kinds(&off.1), kinds(&on.1), "{name}: the lines");
    assert_eq!(rounds(&off.1), rounds(&on.1), "{name}: the round lines");
}

/// The `kept` of every round line, in order.
pub(crate) fn kept(lines: &[Line]) -> Vec<usize> {
    lines
        .iter()
        .filter(|l| l.kind == "round")
        .map(|l| {
            l.get("kept")
                .expect("a round line names the rows kept")
                .parse()
                .unwrap()
        })
        .collect()
}

/// The systems of tests 1 and 9 over `p`: random sparse systems on four and
/// five variables and dense cubics on three, seeds 1 to 4.
pub(crate) fn systems(p: u64) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for s in 1..=4 {
        out.push((format!("sparse4-p{p}-{s}"), random_sparse(4, p, s)));
        out.push((format!("sparse5-p{p}-{s}"), random_sparse(5, p, s)));
        out.push((format!("dense33-p{p}-{s}"), dense(3, 3, p, s)));
    }
    out
}

#[test]
fn the_matrices_stay() {
    let params = reported(MonomialOrder::Grevlex);
    let mut total = 0;
    for p in [65521, 7] {
        for (name, text) in systems(p) {
            let (off, on) = pair(&text, &params);
            agree(&name, &off, &on);
            total += kept(&on.1).iter().sum::<usize>();
        }
    }
    assert!(total > 0, "no round took a row from the matrix before");
}

#[test]
fn over_q_and_in_other_orders() {
    let params = reported(MonomialOrder::Grevlex);
    for s in 1..=4 {
        for n in [4, 5] {
            let text = random_sparse(n, 0, s);
            let (off, on) = pair(&text, &params);
            agree(&format!("sparse{n}-q-{s}"), &off, &on);
        }
    }
    let eliminable = "x,y,z\n7\nx+z,\nz^2+1\n";
    for order in ["lex", "block:1"] {
        let params = reported(order.parse().unwrap());
        for (name, text) in [
            ("dense33-p7-4", dense(3, 3, 7, 4)),
            ("eliminable", eliminable.to_string()),
        ] {
            let (off, on) = pair(&text, &params);
            agree(&format!("{name} in {order}"), &off, &on);
        }
    }
}

#[test]
fn the_bound_drops_the_kept_rows() {
    let params = reported(MonomialOrder::Grevlex);
    for p in [65521, 7] {
        for s in [1, 2] {
            let name = format!("sparse5-p{p}-{s}");
            let text = random_sparse(5, p, s);
            let unbounded = call(&text, &params);
            let mut before: Option<Vec<usize>> = None;
            for bound in [0, 1, 64, 4096] {
                KEPT_BOUND.set(Some(bound));
                let bounded = call(&text, &params);
                KEPT_BOUND.set(None);
                agree(&format!("{name} at {bound}"), &unbounded, &bounded);
                let now = kept(&bounded.1);
                if let Some(before) = &before {
                    assert_eq!(before.len(), now.len(), "{name}: the rounds");
                    for (r, (b, n)) in before.iter().zip(&now).enumerate() {
                        assert!(b <= n, "{name}: round {r} kept {n} at {bound} after {b}");
                    }
                }
                before = Some(now);
            }
            let last = before.unwrap();
            for (r, (b, n)) in last.iter().zip(kept(&unbounded.1)).enumerate() {
                assert!(*b <= n, "{name}: round {r} kept {n} unbounded after {b}");
            }
        }
    }
}

#[test]
fn the_edge_cases() {
    let params = reported(MonomialOrder::Grevlex);
    for p in [65521, 0] {
        for (k, gens) in EDGE.iter().enumerate() {
            let text = format!("x,y,z\n{p}\n{gens}\n");
            let (off, on) = pair(&text, &params);
            agree(&format!("edge {k} over {p}"), &off, &on);
        }
    }
}

#[test]
fn kept_rows_are_counted() {
    let params = reported(MonomialOrder::Grevlex);
    for (p, s) in [(65521, 1), (7, 2)] {
        let name = format!("sparse5-p{p}-{s}");
        let (_, lines) = call(&random_sparse(5, p, s), &params);
        let (mut first, mut total) = (false, 0);
        for line in &lines {
            match line.kind.as_str() {
                "run" => first = true,
                "round" => {
                    let field = |k: &str| line.get(k).unwrap().parse::<usize>().unwrap();
                    let (kept, rows, given) = (field("kept"), field("rows"), field("given"));
                    assert!(kept + given <= rows, "{name}: {line:?}");
                    if first {
                        assert_eq!(kept, 0, "{name}: a run's first round: {line:?}");
                        first = false;
                    }
                    total += kept;
                }
                "inter" => assert!(line.get("kept").is_none(), "{name}: {line:?}"),
                _ => {}
            }
        }
        assert!(
            total > 0,
            "{name}: no round took a row from the matrix before"
        );
    }
}

#[test]
fn a_stop_inside_a_build_that_holds_kept_rows() {
    let text = random_sparse(5, 65521, 1);
    let params = reported(MonomialOrder::Grevlex);
    let full = call(&text, &params).0.unwrap();
    for ms in [0, 1, 2, 4, 8, 16, 32] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.deadline = Some(Duration::from_millis(ms));
        let outcome = crate::groebner(&text, &params).unwrap();
        match outcome.stopped {
            Some(Stop::Deadline) => assert!(outcome.basis.is_none(), "{ms} ms: {outcome:?}"),
            None => assert_eq!(outcome.basis.unwrap(), full, "{ms} ms"),
            Some(Stop::Cancel | Stop::Memory) => panic!("{ms} ms: {outcome:?}"),
        }
    }
}

/// A monic element of the basis from its terms on x, y, z, w.
pub(crate) fn element(
    table: &mut Table<Dense16>,
    field: &Gf31,
    terms: &[([u32; 4], u64)],
) -> Poly<Gf31, Rank1> {
    let mut terms: Vec<(MonoId, <Gf31 as Field>::Coef)> = terms
        .iter()
        .map(|&(e, c)| {
            (
                table.intern(&e, &mut Budget::unbounded()).unwrap(),
                field.coef(c),
            )
        })
        .collect();
    let mut g = Poly::from_terms(&mut terms, field, table, &Grevlex, ());
    g.make_monic(field);
    g
}

/// The two rows of the S-pair of elements `i` and `j`.
pub(crate) fn halves(
    table: &mut Table<Dense16>,
    basis: &Basis<Gf31, Rank1>,
    i: usize,
    j: usize,
) -> [RowSource; 2] {
    let (i, j) = (basis.min()[i], basis.min()[j]);
    let lcm = table
        .lcm(basis.lead(i), basis.lead(j), &mut Budget::unbounded())
        .unwrap();
    [i, j].map(|g| RowSource {
        element: g,
        multiplier: table
            .div(lcm, basis.lead(g), &mut Budget::unbounded())
            .unwrap(),
    })
}

/// Asserts the two matrices are the same: rows, given, every row's
/// columns, monomials, and entries.
pub(crate) fn same(name: &str, a: &Matrix, b: &Matrix) {
    assert_eq!(a.rows, b.rows, "{name}: the rows");
    assert_eq!(a.given, b.given, "{name}: the given rows");
    assert_eq!(a.monomials, b.monomials, "{name}: the monomials");
    assert_eq!(a.entries(), b.entries(), "{name}: the entries");
    for r in 0..a.rows.len() {
        assert_eq!(
            a.row(RowId::at(r)).to_wide(),
            b.row(RowId::at(r)).to_wide(),
            "{name}: row {r}"
        );
    }
}

#[test]
fn a_kept_row_is_taken_and_the_rest_compacted() {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(4), &mut Xoshiro256PlusPlus::seed_from_u64(7));
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    // Leads x^2, y^2, z^2, xy, and w^2 in grevlex: G_min is every element.
    for terms in [
        [([2, 0, 0, 0], 1), ([0, 1, 1, 0], 3)],
        [([0, 2, 0, 0], 1), ([1, 0, 0, 1], 5)],
        [([0, 0, 2, 0], 1), ([0, 0, 0, 1], 7)],
        [([1, 1, 0, 0], 1), ([0, 0, 1, 1], 11)],
        [([0, 0, 0, 2], 1), ([1, 0, 0, 0], 13)],
    ] {
        let g = element(&mut table, &field, &terms);
        basis.push_min(&table, g, &mut Budget::unbounded()).unwrap();
    }
    assert_eq!(basis.min().len(), 5);
    // Round 1: the pairs (g1, g4) and (g2, g4), whose tails take the
    // reducers (g2, z) and (g1, w). Round 2: the same pairs and (g3, g5),
    // whose tails take (g5, w) and (g3, x) beside the two kept rows.
    let first: Vec<RowSource> = [
        halves(&mut table, &basis, 0, 3),
        halves(&mut table, &basis, 1, 3),
    ]
    .concat();
    let mut second = first.clone();
    second.extend(halves(&mut table, &basis, 2, 4));
    let mut bufs: Buffers<Gf31> = Buffers::new();
    let m1 = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        first.clone(),
        &mut bufs,
        &mut reported_budget(),
    )
    .unwrap();
    let reducers: Vec<RowSource> = m1.rows[m1.given..].to_vec();
    assert_eq!(reducers.len(), 2, "round 1's reducers");
    bufs.keep(m1);
    let m2 = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        second.clone(),
        &mut bufs,
        &mut reported_budget(),
    )
    .unwrap();
    let mut fresh: Buffers<Gf31> = Buffers::new();
    let m2_fresh = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        second.clone(),
        &mut fresh,
        &mut reported_budget(),
    )
    .unwrap();
    same("round 2", &m2_fresh, &m2);
    let repeated = m2.rows[m2.given..]
        .iter()
        .filter(|r| reducers.contains(r))
        .count();
    assert_eq!(repeated, 2, "round 2's rows the first matrix held");
    assert_eq!(m2.kept(), repeated, "the rows taken");
    // Under a bound of 0 the kept rows go at the first row, and the matrix
    // is the same still.
    let mut again: Buffers<Gf31> = Buffers::new();
    let m1_again = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        first,
        &mut again,
        &mut reported_budget(),
    )
    .unwrap();
    again.keep(m1_again);
    KEPT_BOUND.set(Some(0));
    let m2_bounded = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        second,
        &mut again,
        &mut reported_budget(),
    )
    .unwrap();
    KEPT_BOUND.set(None);
    same("round 2 under the bound", &m2_fresh, &m2_bounded);
    assert_eq!(m2_bounded.kept(), 0, "no row taken under the bound");
}
