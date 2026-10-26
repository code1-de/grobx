//! call-report, the item's test code: over dense and random square systems
//! over GF(65521) and over ℚ, a call without the report returns none, and
//! the same call with it returns the same outcome and a report that agrees
//! with its stats (`iterations/0064-call-report/plan.md`, _Test code_, 23).

use crate::common::{P, dense, params, random_square};
use crate::report::{
    Line, assert_run_counts, assert_stage_times, assert_stats, of, report as lines_of, run_lines,
    without_report,
};

/// The systems over the characteristic p, each with its name: dense cubic
/// ones on `variables`, and random square ones under three seeds.
fn systems(p: u64, variables: std::ops::RangeInclusive<usize>) -> Vec<(String, String)> {
    let dense_systems = variables.map(|m| (format!("cubic-{m}-p{p}"), dense(m, 3, p, 1)));
    let squares = (0..3).map(|seed| (format!("square-{seed}-p{p}"), random_square(p, seed)));
    dense_systems.chain(squares).collect()
}

/// The call as measured, without the report and with it: the first holds
/// none and a basis, the second the same outcome and a report.
fn both(case: &str, text: &str) -> (grobx::Outcome, Vec<Line>) {
    let p = params(0x2e9);
    let off = grobx::groebner(text, &p).unwrap_or_else(|e| panic!("{case}: {e:?}"));
    assert_eq!(off.report, None, "{case}");
    assert!(off.basis.is_some(), "{case}: stopped");
    let mut on = p.clone();
    on.report = true;
    let on = grobx::groebner(text, &on).unwrap_or_else(|e| panic!("{case}: {e:?}"));
    assert_eq!(
        without_report(&on),
        off,
        "{case}: the report changed the outcome"
    );
    let lines = lines_of(&on);
    (on, lines)
}

/// Over GF(65521): every run's counts, the outcome's stats, and the stage
/// times, as call-report's test 6 checks them.
#[test]
fn over_gfp() {
    for (case, text) in systems(P, 2..=4) {
        let (outcome, lines) = both(&case, &text);
        assert_stats(&lines, &outcome.stats);
        assert_stage_times(&lines);
    }
}

/// Over ℚ: every run's counts, a verification that passed, and the
/// outcome's stats in the `ran` line of the learn run whose `learn` line
/// recorded last, as call-report's test 19 checks them.
#[test]
fn over_q() {
    for (case, text) in systems(0, 2..=3) {
        let (outcome, lines) = both(&case, &text);
        assert!(lines[2].is("lift", &[]), "{case}");
        for ran in of(&lines, "ran") {
            assert_run_counts(run_lines(&lines, ran.num("number")));
        }
        assert_stage_times(&lines);
        let recorded = lines
            .iter()
            .rposition(|l| l.is("learn", &[("outcome", "recorded")]))
            .unwrap_or_else(|| panic!("{case}: no record"));
        let ran = lines[..recorded]
            .iter()
            .rfind(|l| l.kind == "ran")
            .expect("the learn's run");
        let stats = &outcome.stats;
        assert_eq!(
            (ran.num("rounds"), ran.num("reduced"), ran.num("zero")),
            (u64::from(stats.rounds), stats.reduced, stats.zero),
            "{case}: {ran:?}"
        );
        assert!(
            lines[recorded..]
                .iter()
                .any(|l| l.is("verify", &[("outcome", "passed")])),
            "{case}"
        );
    }
}
