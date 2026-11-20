//! signature-run-losses, the item's integration tests: calls whose
//! signature run hands over after `CARRY_WORK` entries or more, which the
//! plain run carries on from, return the basis of the call with the
//! signatures off, canonical; over 65521 the call with cofactors, which
//! never hands over, returns it, and the exact checker accepts it
//! (`iterations/0112-signature-run-losses/plan.md`, _Test code_, 8). The
//! report of a call that carries names both runs, each with its own
//! counts, and the outcome counts both (signature-run-losses.8, .9).

use grobx_checker::Mode;

use crate::common::{P, P31, assert_accepted, params, run, systems::random_sparse, with_cofactors};
use crate::large::LEAST;
use crate::orders::assert_canonical;
use crate::report::{assert_stats, report};

/// The variables of the random sparse systems. Over 65521, 2^31 − 1, and
/// 2,147,483,659, seeds 0 to 7, 31 of their 48 calls at seed 1 hand over
/// after `CARRY_WORK` entries or more, 6 of them at the bound of the
/// round's degree and 25 one below it, and 13 hand over below it, in a
/// copy with the item's code (log 0112-1, _Step 5_).
const VARIABLES: [usize; 2] = [6, 7];

/// The seeds of the systems.
const SYSTEMS: u64 = 8;

/// The seeds of each call.
const SEEDS: [u64; 3] = [1, 2, 3];

/// Test 8: signature-run-losses.1 to .7. Random sparse systems on 6 and 7
/// variables over 65521, 2^31 − 1, and the least prime past 2^31, three
/// seeds each: the call returns the basis of the call with the signatures
/// off, which is canonical; over 65521 on 6 variables the call with
/// cofactors returns it, and the exact checker accepts it. On 7 variables
/// the calls with cofactors took 0.8 to 38 s in release (log 0112-1,
/// _Step 5_).
#[test]
fn carried_calls_give_the_plain_basis() {
    for n in VARIABLES {
        for p in [P, P31, LEAST] {
            for system in 0..SYSTEMS {
                let text = random_sparse(n, p, system);
                let name = format!("sparse-{n} over {p}, system {system}");
                for seed in SEEDS {
                    let mut off = params(seed);
                    off.signatures = false;
                    let want = run(&text, &off).basis.unwrap();
                    assert_eq!(
                        run(&text, &params(seed)).basis.as_deref(),
                        Some(want.as_str()),
                        "{name}, seed {seed}\n{text}"
                    );
                    if seed == SEEDS[0] {
                        assert_canonical(&name, &text, &want, "grevlex");
                        if p == P && n == VARIABLES[0] {
                            let outcome = run(&text, &with_cofactors(seed));
                            assert_eq!(outcome.basis.as_deref(), Some(want.as_str()), "{name}");
                            assert_accepted(&name, &text, &outcome, Mode::Exact);
                        }
                    }
                }
            }
        }
    }
}

/// signature-run-losses.8 and .9. The calls of test 8's systems on 6
/// variables over 65521 at its first seed, with the report on: every
/// report's runs hold their own counts and the outcome's stats their sum
/// (`report::assert_stats`). Where the signature run carries, its `ran`
/// line with the outcome `carry` and its S-pairs' fates comes right before
/// the plain run's `run` line with `why=carry`, and that one right before
/// the `carry` line; the plain run's `ran` line names no fates. At least
/// one of the calls carries.
#[test]
fn a_carried_call_reports_both_runs() {
    let mut carried = 0;
    for system in 0..SYSTEMS {
        let text = random_sparse(VARIABLES[0], P, system);
        let mut p = params(SEEDS[0]);
        p.report = true;
        let outcome = run(&text, &p);
        let lines = report(&outcome);
        assert_stats(&lines, &outcome.stats);
        let Some(k) = lines
            .iter()
            .position(|l| l.is("ran", &[("outcome", "carry")]))
        else {
            continue;
        };
        carried += 1;
        let (ran, plain, carry) = (&lines[k], &lines[k + 1], &lines[k + 2]);
        ran.num("considered");
        assert!(
            plain.is("run", &[("engine", "plain"), ("why", "carry")]),
            "system {system}: {plain:?}"
        );
        assert_eq!(carry.kind, "carry", "system {system}: {carry:?}");
        assert!(
            carry.num("made") <= carry.num("elements") && carry.num("work") > 0,
            "system {system}: {carry:?}"
        );
        let last = lines.iter().rev().find(|l| l.kind == "ran").unwrap();
        assert!(
            last.is("ran", &[("outcome", "basis")]) && last.get("considered").is_none(),
            "system {system}: {last:?}"
        );
    }
    assert!(carried > 0, "no call carried");
}
