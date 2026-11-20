//! call-stop-latency, the crate's test code: the plain run's update
//! spends at least a unit per place of the pair set and per member of
//! `G_min`, and a bound stops the call at the update whose spends reach the
//! units left (call-stop-latency.1, an intervention of 2026-10-30; the
//! spends as the update goes, call-stop-latency.13).

use std::time::Instant;

use crate::Stop;
use crate::budget::Budget;
use crate::divisibility_tests::Engine;
use crate::halt::Halt;
use crate::monomial::Dense16;
use crate::order::Grevlex;
use crate::shared_tests::{fill, params, passed_after};

/// The variables of every engine here.
const N: usize = 6;

/// The degree of every leading monomial here: no two distinct monomials of
/// one degree divide each other, so every element joins `G_min`.
const DEGREE: u32 = 3;

/// Every monomial of degree `DEGREE` in `N` variables.
fn leads() -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    fill(&mut [0u32; N], 0, DEGREE, &mut out);
    out
}

#[test]
fn update() {
    let mut failures = Vec::new();
    let leads = leads();

    // The spends under a budget that never stops.
    let mut engine: Engine<Dense16> = Engine::new(N, 1);
    let mut budget = Budget::new(Instant::now(), &params(1)).unwrap();
    let mut spends = Vec::with_capacity(leads.len());
    for (k, lead) in leads.iter().enumerate() {
        let before = budget.spent();
        let scanned = (engine.pairs.held() + engine.min_len()) as u64;
        let (got, _) = engine.enter_under(&Grevlex, lead, DEGREE, 0, &mut budget);
        if got.is_err() {
            failures.push(format!("update {k} without a bound: {got:?}"));
        }
        let spent = budget.spent() - before;
        if spent < scanned {
            failures.push(format!(
                "update {k}: spent {spent}, fewer than the {scanned} places and members it scanned"
            ));
        }
        spends.push(spent);
    }
    let total: u64 = spends.iter().sum();
    let (held, elements) = (engine.pairs.held(), leads.len());
    if held <= elements {
        failures.push(format!(
            "the set holds {held} places for {elements} elements, too few to tell the places from G_min's members"
        ));
    }

    // A passed deadline stops the call at the first update whose spends
    // reach the units left, and not before: an update spends as it does
    // under a budget that never stops.
    for units in [1, total / 3, total / 2, total, total + 1] {
        let mut engine: Engine<Dense16> = Engine::new(N, 1);
        let mut budget = passed_after(units);
        let mut sum = 0u64;
        let stop_at = spends.iter().position(|&s| {
            sum += s;
            sum >= units
        });
        for (k, lead) in leads.iter().enumerate() {
            let (got, _) = engine.enter_under(&Grevlex, lead, DEGREE, 0, &mut budget);
            let want = if Some(k) == stop_at {
                Err(Halt::Stop(Stop::Deadline))
            } else {
                Ok(())
            };
            if got != want {
                failures.push(format!("{units} units, update {k}: {got:?}, not {want:?}"));
            }
            if got.is_err() {
                break;
            }
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}
