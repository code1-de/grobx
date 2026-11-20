//! unreached-test-paths, the item's test code in the core: a pair learn whose round keeps a row
//! whose lead is 0 at one prime alone halts with a misfit, and a stop between a round's count and
//! its end keeps the round's line in the report
//! (`iterations/0151-unreached-test-paths/plan.md`, _Test code_, tests 1 and 2).
// unreached-test-paths.8 design-checked -- the item's new tests stand in files of their own named after the item, in the core's unit tests and in the suites of the CLI, the C interface, and the Python module, and its changes to existing checks in their files, not each new test in the file of the item whose path it reaches (not measured)

use crate::budget::Site;
use crate::call_stop_sites_tests::forced;
use crate::learn_pair_tests::{disagreements, learn_alone, pair_learn, params_in, run_at};
use crate::report_tests::lines;
use crate::shared_tests::{L1, L2};
use crate::{Error, Stop, call_stop_item_tests};

// unreached-test-paths.7 design-checked -- the pair learn's halt in a round's reduction is reached by test 3's cases of a row zero at one prime alone with the generators' echelon form off, not by a system whose echelon form leaves its generators as they are, whose halt a later echelon may yet take, nor by the halt's place read from the call report (log 0151-1)
/// Test 1, q-learn-time.4, .5 in a round's reduction: the two cases of test 3 of
/// `learn_pair_tests.rs` whose row is zero at one prime alone, with the generators' echelon form
/// off, so that the row's lead is first inverted where a round keeps it (`Reducer::keep`), with
/// signatures on and off and the two primes either way round. The pair learn halts with
/// `Misfit { step: 0 }`, and the learn run at L1 alone and the run at L2 alone each return an
/// image.
#[test]
fn a_round_halts_the_pair_learn() {
    let rows: Vec<_> = disagreements()
        .into_iter()
        .filter(|(case, ..)| case.starts_with("a row zero"))
        .collect();
    assert_eq!(rows.len(), 2, "the cases of a row zero at one prime alone");
    for (case, text, order, _) in rows {
        for signatures in [true, false] {
            let mut p = params_in(order, 1);
            p.echelon = false;
            p.signatures = signatures;
            for primes in [(L1, L2), (L2, L1)] {
                let got = pair_learn(&text, &p, primes);
                assert!(
                    matches!(got, Err(Error::Misfit { step: 0 })),
                    "{case}, signatures {signatures}, primes {primes:?}: {got:?}",
                    got = got.map(|_| "two lanes")
                );
            }
            learn_alone(&text, &p, L1);
            run_at(&text, &p, L2);
        }
    }
}

// unreached-test-paths.4 design-checked -- the stop between a round's count and its end is forced at the first full check of Site::Decode, where the run lists the rows that counted, not set by a deadline, which lands there by its timing alone, nor forced at Site::Update, whose first check comes as the inputs enter, before any round (log 0151-1)
/// Test 2, call-report.19: the case of `Site::Decode` (`call_stop_item_tests::case`), with its
/// report, signatures on and off, and a stop forced at the first full check of `Site::Decode`,
/// which comes after the first round's count and before the round's end. The call stops with
/// `Deadline` and no basis, and the stopped run's `ran` line counts at least one round and as
/// many rounds as the run has `round` lines.
#[test]
fn a_round_a_stop_ends_keeps_its_line() {
    let mut case = call_stop_item_tests::case(Site::Decode).expect("a case of Site::Decode");
    case.params.report = true;
    for signatures in [true, false] {
        case.params.signatures = signatures;
        let (got, _) = forced(&case, Some(Site::Decode));
        let got = got.unwrap_or_else(|e| panic!("signatures {signatures}: {e}"));
        assert_eq!(
            (got.stopped, got.basis.is_some()),
            (Some(Stop::Deadline), false),
            "signatures {signatures}"
        );
        let lines = lines(got.report.as_deref().expect("the report is on"));
        let ran = lines
            .iter()
            .rposition(|l| l.kind == "ran")
            .expect("a ran line");
        let run = lines[..ran]
            .iter()
            .rposition(|l| l.kind == "run")
            .expect("a run line before it");
        let stopped = &lines[ran];
        assert_eq!(
            stopped.get("outcome"),
            Some("stop"),
            "signatures {signatures}: {stopped:?}"
        );
        let counted: usize = stopped
            .get("rounds")
            .and_then(|r| r.parse().ok())
            .expect("the ran line's rounds");
        let written = lines[run..ran].iter().filter(|l| l.kind == "round").count();
        assert!(counted >= 1, "signatures {signatures}: no round counted");
        assert_eq!(
            counted, written,
            "signatures {signatures}: the rounds counted and the round lines"
        );
    }
}
