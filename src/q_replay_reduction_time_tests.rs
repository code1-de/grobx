//! q-replay-reduction-time, the crate's test code: the cut of a record to
//! the counted rows its learn run's output depends on, on records written
//! out, and the replays of the lift's cut records over GF(p)
//! (`iterations/0098-q-replay-reduction-time/plan.md`, _Test code_, 1 to
//! 7).

use std::num::NonZeroUsize;

use crate::budget::Budget;
use crate::id::{ElementId, RowId, RowSource};
use crate::systems::{dense, random_sparse, random_square, random_system};
use crate::trace::{Output, Reach, Record, Step, Trace, cut};
use crate::{CheckerMode, MonomialOrder, Outcome, Params, Threads};

/// The variables of the records written out.
const N: usize = 2;

/// The primes the generated systems are learned over.
const LEARN: [u64; 3] = [7, 65521, 2_147_483_647];

/// The primes above 2^30 a record learned over 65521 or 2^31 − 1 is
/// replayed over besides its own.
const OTHER: [u64; 2] = [2_147_483_647, 2_147_483_629];

/// The edge cases of plan.md over GF(65521).
const EDGE: &[(&str, &str)] = &[
    ("zero", "x,y\n65521\n0\n"),
    ("no generator", "x,y\n65521\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n"),
    ("constant among others", "x,y\n65521\nx^2+y,\nx*y+1,\n7\n"),
    ("already a basis", "x,y\n65521\nx+y,\ny^2+1\n"),
    ("zero generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    ("eliminable", "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
];

/// A row written out: its element and its multiplier's exponents.
type Row = (u32, [u32; N]);

/// A step of a record written out: its rows in the order taken, the
/// positions of its counted rows, each counted row's leading exponents in
/// that order, and the positions in the order the new elements entered.
fn step(rows: &[Row], reductees: &[usize], leads: &[[u32; N]], entered: &[u32]) -> Step {
    Step {
        elements: rows
            .iter()
            .map(|&(e, _)| ElementId::at(e as usize))
            .collect(),
        multipliers: rows.iter().flat_map(|&(_, m)| m).collect(),
        reductees: reductees.iter().map(|&r| RowId::at(r)).collect(),
        leads: leads.iter().flatten().copied().collect(),
        entered: entered.to_vec(),
    }
}

/// A step's reach written out: per row, the rows of its pivots.
fn reach(rows: &[&[u32]]) -> Reach {
    let mut start = vec![0];
    let mut all = Vec::new();
    for r in rows {
        all.extend_from_slice(r);
        start.push(u32::try_from(all.len()).unwrap());
    }
    Reach { start, rows: all }
}

/// A record of two inputs, x^2 and y^2, ids 0 and 1, and these steps.
fn record(steps: Vec<Step>) -> Record {
    Record {
        order: MonomialOrder::Grevlex,
        variables: N,
        inputs: vec![Some(vec![2, 0]), Some(vec![0, 2])],
        entered: vec![0, 1],
        steps,
        changed: false,
        inter_reduced: true,
        echelon: false,
    }
}

fn ids(ids: &[usize]) -> Vec<ElementId> {
    ids.iter().map(|&i| ElementId::at(i)).collect()
}

/// Two steps. Step 0: pivot rows of elements 0 and 1, counted rows of
/// elements 0 and 1 that enter in reverse, ids 3 and 2. Step 1: a pivot row
/// of element 2, counted rows of elements 0 and 1, ids 4 and 5.
fn two_steps() -> (Record, Vec<Reach>) {
    let s0 = step(
        &[(0, [0, 1]), (1, [1, 0]), (0, [0, 2]), (1, [2, 0])],
        &[2, 3],
        &[[1, 1], [0, 3]],
        &[1, 0],
    );
    let s1 = step(
        &[(2, [1, 0]), (0, [1, 1]), (1, [1, 1])],
        &[1, 2],
        &[[1, 2], [3, 0]],
        &[0, 1],
    );
    let reaches = vec![reach(&[&[], &[], &[0], &[1]]), reach(&[&[], &[0], &[]])];
    (record(vec![s0, s1]), reaches)
}

#[test]
fn cut_keeps_what_g_min_reaches() {
    let (mut r, reaches) = two_steps();
    cut(
        &mut r,
        &reaches,
        Output::Basis(&ids(&[1, 4])),
        &mut Budget::unbounded(),
    )
    .unwrap();
    // Step 1: id 4, its counted row 1, and the pivot row it reaches stay;
    // id 5 is skipped, its lead after the kept one's.
    let want1 = step(
        &[(2, [1, 0]), (0, [1, 1])],
        &[1],
        &[[1, 2], [3, 0]],
        &[0, 1],
    );
    assert_eq!(r.steps[1], want1, "step 1");
    // Step 0: step 1 multiplies element 2, the counted row at position 1,
    // and it reaches the pivot row of element 1; the counted row of id 3
    // is skipped.
    let want0 = step(
        &[(1, [1, 0]), (1, [2, 0])],
        &[1],
        &[[0, 3], [1, 1]],
        &[0, 1],
    );
    assert_eq!(r.steps[0], want0, "step 0");
    assert_eq!(r.skipped(), 2);
    assert_eq!(r.reduced(), 2);
    assert_eq!(r.rows(), 4);
    assert_eq!((r.entered.clone(), r.inputs.len()), (vec![0, 1], 2));
}

#[test]
fn a_new_pivot_keeps_its_row() {
    // Pivot row 1's column holds the new pivot of counted row 2, and
    // counted row 3's column holds pivot row 1's.
    let s0 = step(
        &[(0, [0, 1]), (1, [1, 0]), (0, [0, 2]), (1, [2, 0])],
        &[2, 3],
        &[[1, 1], [0, 3]],
        &[0, 1],
    );
    let reaches = vec![reach(&[&[], &[2], &[0], &[1]])];
    let whole = record(vec![s0.clone()]);
    let mut r = whole.clone();
    cut(
        &mut r,
        &reaches,
        Output::Basis(&ids(&[0, 1, 3])),
        &mut Budget::unbounded(),
    )
    .unwrap();
    assert_eq!(r, whole, "id 2 is not needed, and its row is kept");
    assert_eq!(r.skipped(), 0);
    let mut r = whole.clone();
    cut(
        &mut r,
        &reaches,
        Output::Basis(&ids(&[0, 1, 2])),
        &mut Budget::unbounded(),
    )
    .unwrap();
    let want = step(
        &[(0, [0, 1]), (0, [0, 2])],
        &[1],
        &[[1, 1], [0, 3]],
        &[0, 1],
    );
    assert_eq!(r.steps[0], want, "id 3 skipped, pivot row 1 dropped");
    assert_eq!(r.skipped(), 1);
}

#[test]
fn a_step_may_skip_all() {
    let steps = vec![
        step(&[(0, [0, 1]), (1, [1, 0])], &[1], &[[1, 1]], &[0]),
        step(&[(1, [0, 1]), (0, [0, 2])], &[1], &[[0, 3]], &[0]),
        step(&[(3, [1, 0]), (2, [0, 1])], &[1], &[[2, 1]], &[0]),
    ];
    let reaches = vec![
        reach(&[&[], &[0]]),
        reach(&[&[], &[0]]),
        reach(&[&[], &[0]]),
    ];
    let mut r = record(steps.clone());
    // Ids: step 0 makes 2, step 1 makes 3, step 2 makes 4.
    cut(
        &mut r,
        &reaches,
        Output::Basis(&ids(&[0, 1, 3])),
        &mut Budget::unbounded(),
    )
    .unwrap();
    assert_eq!(r.steps[2], step(&[], &[], &[[2, 1]], &[0]), "the last step");
    assert_eq!(r.steps[1], steps[1], "the middle step");
    assert_eq!(
        r.steps[0],
        step(&[], &[], &[[1, 1]], &[0]),
        "the first step"
    );
    assert_eq!(r.skipped(), 2);
    assert_eq!(r.reduced(), 1);
}

#[test]
fn the_constant_keeps_its_row() {
    let s0 = step(&[(0, [0, 1]), (1, [1, 0])], &[1], &[[1, 1]], &[0]);
    let s1 = step(
        &[(2, [1, 0]), (0, [0, 1]), (1, [1, 1]), (0, [0, 2])],
        &[2, 3],
        &[[0, 0], [1, 0]],
        &[0, 1],
    );
    let reaches = vec![reach(&[&[], &[0]]), reach(&[&[], &[], &[0], &[1]])];
    let mut r = record(vec![s0.clone(), s1]);
    cut(&mut r, &reaches, Output::Constant, &mut Budget::unbounded()).unwrap();
    let want1 = step(
        &[(2, [1, 0]), (1, [1, 1])],
        &[1],
        &[[0, 0], [1, 0]],
        &[0, 1],
    );
    assert_eq!(r.steps[1], want1, "the constant's row and its closure");
    assert_eq!(
        r.steps[0], s0,
        "the element the constant's closure multiplies"
    );
    assert_eq!(r.skipped(), 1);
}

#[test]
fn nothing_skipped_leaves_the_record() {
    let (whole, reaches) = two_steps();
    let mut r = whole.clone();
    cut(
        &mut r,
        &reaches,
        Output::Basis(&ids(&[0, 1, 2, 3, 4, 5])),
        &mut Budget::unbounded(),
    )
    .unwrap();
    assert_eq!(r, whole);
    assert_eq!(r.skipped(), 0);
}

/// Counts, per matrix the trace hears, the rows that made a new element.
#[derive(Default)]
struct Counts(Vec<usize>);

impl Trace for Counts {
    fn matrix(&mut self, _: &[RowSource]) {}

    fn counted(&mut self, rows: &[RowId]) {
        self.0.push(rows.len());
    }
}

impl Counts {
    /// The counted rows of the rounds: every matrix but the inter-reduction,
    /// the last, which a run that ends with a constant does not reach.
    fn rounds(&self, basis: &str) -> usize {
        let n = if basis == "[1]:\n" {
            self.0.len()
        } else {
            self.0.len().saturating_sub(1)
        };
        self.0[..n].iter().sum()
    }
}

fn params(order: &str, signatures: bool) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    p.order = order.parse().unwrap();
    p.signatures = signatures;
    p
}

/// The instance with its characteristic, the second line, replaced by p.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let variables = lines.next().unwrap();
    lines.next();
    let rest = lines.next().unwrap_or("");
    format!("{variables}\n{p}\n{rest}")
}

/// The generated systems over GF(p), with `small` those that compute in
/// lex too: name and text.
fn systems(p: u64, small: bool) -> Vec<(String, String)> {
    let mut list = Vec::new();
    for seed in 0..6 {
        list.push((format!("random-{seed}"), random_system(p, seed)));
        list.push((format!("square-{seed}"), random_square(p, seed)));
    }
    list.push(("cubic-3".to_string(), dense(3, 3, p, 1)));
    if !small {
        for seed in 0..6 {
            list.push((format!("sparse4-{seed}"), random_sparse(4, p, seed)));
            list.push((format!("sparse5-{seed}"), random_sparse(5, p, seed)));
        }
    }
    for (name, text) in EDGE {
        list.push((name.to_string(), over(text, p)));
    }
    list
}

/// The outcome, the cut record, and the learn trace's counts of the
/// lift's learn of `text`.
fn learn_cut(name: &str, text: &str, params: &Params) -> (Outcome, Record, Counts) {
    let mut counts = Counts::default();
    let mut recording = crate::trace::Recording::cutting();
    let outcome = crate::call::groebner(text, params, &mut counts, &mut recording)
        .unwrap_or_else(|e| panic!("{name}: the learn failed: {e:?}\n{text}"));
    assert!(outcome.basis.is_some(), "{name}: the learn stopped");
    let record = recording.finish(params.order.clone());
    (outcome, record, counts)
}

/// The outcome and the trace's counts of a replay that must fit.
fn replay(name: &str, text: &str, params: &Params, record: &Record) -> (Outcome, Counts) {
    let mut counts = Counts::default();
    let outcome = crate::replay(text, params, record, &mut counts)
        .unwrap_or_else(|e| panic!("{name}: the replay failed: {e:?}\n{text}"));
    assert!(
        outcome.stopped.is_none() && outcome.basis.is_some(),
        "{name}: the replay stopped: {outcome:?}"
    );
    (outcome, counts)
}

fn basis(text: &str, params: &Params) -> String {
    crate::groebner(text, params)
        .unwrap_or_else(|e| panic!("{e:?}\n{text}"))
        .basis
        .expect("a basis")
}

#[test]
fn cut_records_replay_the_basis() {
    let mut skipped = 0;
    for p in LEARN {
        for order in ["grevlex", "lex", "block:1"] {
            for (name, text) in systems(p, order != "grevlex") {
                for signatures in [false, true] {
                    let name = format!("{name} over {p}, {order}, signatures {signatures}");
                    let params = params(order, signatures);
                    let (learned, record, learn_counts) = learn_cut(&name, &text, &params);
                    let basis_text = learned.basis.as_deref().unwrap();
                    skipped += record.skipped();
                    let (own, counts) = replay(&name, &text, &params, &record);
                    assert_eq!(own.basis, learned.basis, "{name}: over its own prime");
                    if order == "grevlex" && !signatures {
                        assert_eq!(
                            counts.rounds(basis_text) + record.skipped(),
                            learn_counts.rounds(basis_text),
                            "{name}: the counted rows replayed and skipped"
                        );
                    }
                    if p == 7 {
                        continue;
                    }
                    for q in OTHER {
                        let there = over(&text, q);
                        let (other, _) = replay(&name, &there, &params, &record);
                        assert_eq!(
                            other.basis.as_deref(),
                            Some(basis(&there, &params).as_str()),
                            "{name}: over {q}"
                        );
                    }
                }
            }
        }
    }
    assert!(skipped > 0, "no record skipped a row");
}

#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn cut_replays_track_cofactors() {
    for (name, text) in systems(65521, false) {
        for signatures in [false, true] {
            let name = format!("{name}, signatures {signatures}");
            // A record a replay with cofactors fits: the echelon off
            // (input-redundancy-time.8).
            let mut learn = params("grevlex", signatures);
            learn.echelon = false;
            let (_, record, _) = learn_cut(&name, &text, &learn);
            let there = over(&text, OTHER[0]);
            let mut checked = params("grevlex", signatures);
            checked.cofactors = true;
            checked.checker = Some(CheckerMode::Exact);
            let (with, _) = replay(&name, &there, &checked, &record);
            assert_eq!(with.certified, Some(true), "{name}: {with:?}");
            let (without, _) = replay(&name, &there, &params("grevlex", signatures), &record);
            assert_eq!(with.basis, without.basis, "{name}");
        }
    }
}
