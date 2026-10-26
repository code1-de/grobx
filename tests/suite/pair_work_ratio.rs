//! pair-work-ratio, the item's test code (pair-work-ratio.1 and .2):
//! systems of random binomials whose signature runs hand over by the
//! pair-work test at a round whose S-pairs reach the entries subtracted
//! and fall short of 4 times them, over GF(p) and ℚ, and the edge cases of
//! `OBJECTIVES.md`, at which the test reads no round
//! (`iterations/0080-pair-work-ratio/plan.md`, _Test code_, 3 to 5).

use grobx::Outcome;
use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted, params, run, systems::random_binomials, with_cofactors,
};
use crate::orders::assert_canonical;
use crate::pair_work_handover::{EDGE, binomials, handed_by_pair_work, reported, unsigned};
use crate::q::{accepted_q, assert_canonical_q};
use crate::report::{report as lines_of, run_lines};

/// The multiple pair-work-handover.3 named, withdrawn: the rounds these
/// tests count read less than it.
const WITHDRAWN: u64 = 4;

/// The systems of tests 3 and 4 at p, or ℚ where p is 0: binomials of
/// degree 2 on 10, 12, and 14 variables, 6 seeds each, and over GF(p)
/// `random_binomials`, 12 seeds.
fn systems(p: u64) -> Vec<(String, String, u64)> {
    let mut systems = Vec::new();
    for n in [10, 12, 14] {
        for seed in 0..6 {
            let case = format!("binomials({n}, 2) over {p}, seed {seed}");
            systems.push((case, binomials(n, 2, p, seed), seed));
        }
    }
    if p != 0 {
        for seed in 0..12 {
            let case = format!("random_binomials over {p}, seed {seed}");
            systems.push((case, random_binomials(p, seed), seed));
        }
    }
    systems
}

/// Where the call's first run handed over by the pair-work test: the
/// S-pairs its `handover` line read and the entries its round subtracted.
fn pair_work_handover(outcome: &Outcome) -> Option<(u64, u64)> {
    let lines = lines_of(outcome);
    let first = run_lines(&lines, 1);
    let line = first
        .iter()
        .find(|l| l.kind == "handover" && l.get("pair-work") == Some("hand-over"))?;
    let round = line.num("round");
    let counts = first
        .iter()
        .find(|l| l.kind == "round" && l.num("number") == round)
        .unwrap_or_else(|| panic!("no round {round}: {line:?}"));
    Some((line.num("pairs"), counts.num("subtracted")))
}

/// Whether the call handed over by the pair-work test at a round whose
/// S-pairs fall short of `WITHDRAWN` times the entries subtracted, each
/// line of its first run checked by `handed_by_pair_work`.
fn handed_below_withdrawn(case: &str, outcome: &Outcome) -> bool {
    handed_by_pair_work(case, outcome);
    pair_work_handover(outcome).is_some_and(|(pairs, subtracted)| {
        assert!(pairs >= subtracted, "{case}: {pairs} against {subtracted}");
        pairs < WITHDRAWN * subtracted
    })
}

/// Test 3, pair-work-ratio.1 and .2: the systems over 65521 and
/// 2^31 − 1 in grevlex. Each basis with signatures is the one without,
/// canonical, and the one with cofactors, which the checker accepts in
/// the exact mode. At least 5 calls at each prime hand over by the
/// pair-work test at a round whose S-pairs reach the entries subtracted
/// and fall short of 4 times them.
#[test]
fn binomials_hand_over_below_four_times() {
    for p in [P, P31] {
        let mut handed = 0;
        for (case, text, seed) in systems(p) {
            let outcome = run(&text, &reported(params(seed)));
            handed += usize::from(handed_below_withdrawn(&case, &outcome));
            let basis = outcome.basis.unwrap();
            let off = run(&text, &unsigned(params(seed)));
            assert_eq!(off.basis.as_deref(), Some(basis.as_str()), "{case}");
            assert_canonical(&case, &text, &basis, "grevlex");
            let with = run(&text, &with_cofactors(seed));
            assert_eq!(with.basis.as_deref(), Some(basis.as_str()), "{case}");
            assert_accepted(&case, &text, &with, Mode::Exact);
        }
        assert!(
            handed >= 5,
            "over {p}: {handed} handed over below 4 times the entries"
        );
    }
}

/// Test 4, pair-work-ratio.1: the binomial systems over ℚ. Each basis
/// with signatures is the one without, canonical, and the one with
/// cofactors, which the checker accepts. At least 5 learns hand over by
/// the pair-work test below 4 times the entries subtracted.
#[test]
fn binomials_over_q_hand_over_below_four_times() {
    let mut handed = 0;
    for (case, text, seed) in systems(0) {
        let outcome = run(&text, &reported(params(seed)));
        handed += usize::from(handed_below_withdrawn(&case, &outcome));
        let basis = outcome.basis.unwrap();
        let off = run(&text, &unsigned(params(seed)));
        assert_eq!(off.basis.as_deref(), Some(basis.as_str()), "{case}");
        assert_canonical_q(&case, &text, &basis, "grevlex");
        let with = accepted_q(&case, &text, "grevlex");
        assert_eq!(with.basis.as_deref(), Some(basis.as_str()), "{case}");
    }
    assert!(
        handed >= 5,
        "over ℚ: {handed} learns handed over below 4 times the entries"
    );
}

/// Test 5: the edge cases over 65521, 7, 2^31 − 1, and ℚ, signatures on
/// and off. The pair-work test reads no round of theirs, and each basis
/// with signatures is the one without.
#[test]
fn the_edge_cases_read_no_pair_work() {
    for (case, generators, _) in EDGE {
        for p in [P, 7, P31, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            let name = format!("{case} over {p}");
            let on = run(&text, &reported(params(0x5a7e)));
            let lines = lines_of(&on);
            assert!(
                lines.iter().all(|l| l.get("pair-work").is_none()),
                "{name}: the pair-work test read a round"
            );
            let off = run(&text, &unsigned(params(0x5a7e)));
            assert_eq!(on.basis, off.basis, "{name}");
        }
    }
}
