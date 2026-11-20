//! call-stop-latency, the item's test code: every site of the budget stops a call from inside its
//! loop, and with every spend checking every call returns what it returns without the forcing
//! (`iterations/0138-call-stop-latency/plan.md`, _Test code_, `call_stop_item_tests.rs`, 1 and 2).

use std::num::NonZeroU32;

use crate::budget::Site;
use crate::call_stop_sites_tests::{Case, forced, stops_at};
use crate::f4::{CARRY_LEAST, COMPACT_BOUND, LARGE_LEAST};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{boolean, random_square, random_system, term};
use crate::{
    CheckerMode, call_stop_lift_tests, call_stop_plain_tests, call_stop_read_tests,
    call_stop_reduce_tests, call_stop_sig_tests, groebner,
};

/// Every site, by a `match` that names each variant, so that a new site needs its line here.
fn every_site() -> Vec<Site> {
    let all = Site::ALL.to_vec();
    for &site in &all {
        match site {
            Site::Run
            | Site::Round
            | Site::Prime
            | Site::Products
            | Site::Checker
            | Site::Append
            | Site::Row
            | Site::Place
            | Site::Compact
            | Site::Columns
            | Site::SigSort
            | Site::Select
            | Site::Update
            | Site::Divisors
            | Site::PairRows
            | Site::Enter
            | Site::Kept
            | Site::Decode
            | Site::Monic
            | Site::Sugar
            | Site::Scan
            | Site::BackBands
            | Site::Inter
            | Site::Setup
            | Site::SigScan
            | Site::SigEnter
            | Site::Antichain
            | Site::Reducer
            | Site::Queue
            | Site::Koszul
            | Site::Carry
            | Site::Parse
            | Site::Map
            | Site::Names
            | Site::Substitute
            | Site::Echelon
            | Site::Trace
            | Site::Images
            | Site::Text
            | Site::Verify
            | Site::Finish
            | Site::Merge
            | Site::Reconstruct
            | Site::Fglm
            | Site::Cofactors => {}
        }
    }
    all
}

/// One generator over GF(65521) of the 969 monomials of degree at most 16 in x, y, and z, each with
/// a coefficient of its own.
fn wide() -> String {
    let names = ["x", "y", "z"];
    let mut terms = Vec::new();
    for a in 0..=16u32 {
        for b in 0..=16 - a {
            for c in 0..=16 - a - b {
                let k = u64::try_from(terms.len()).expect("969 terms");
                terms.push(term(1 + k, &names, &[a, b, c]));
            }
        }
    }
    let poly = terms.join("+");
    format!("x,y,z\n65521\n{poly}\n")
}

/// The calls that reach the sites no file of sub-items 2 to 6 names: the runs', the rounds', the
/// products', and the rows' a random system over GF(65521) that runs rounds; the table's growth a
/// generator of every monomial of degree at most 16 in three variables, more than the table's first
/// slots hold (`wide`);
/// the signature sort's the system with cofactors, which runs the signature run to its end; the
/// primes', the appends', and the trace's a random system over ℚ, whose lift learns a record and
/// appends each prime's images; the checker's the system with cofactors and the cheap checker.
fn rest() -> Vec<(Site, Case)> {
    let gfp = || Case::new(random_system(65521, 5), "grevlex", false);
    let q = || Case::new(random_system(0, 11), "grevlex", false);
    let mut checked = Case::new(random_system(65521, 5), "grevlex", true);
    checked.params.checker = Some(CheckerMode::Cheap {
        points: NonZeroU32::new(2).expect("2 is not 0"),
    });
    vec![
        (Site::Run, gfp()),
        (Site::Round, gfp()),
        (Site::Products, gfp()),
        (Site::Row, gfp()),
        (Site::Place, Case::new(wide(), "grevlex", false)),
        (
            Site::SigSort,
            Case::new(random_system(65521, 5), "grevlex", true),
        ),
        (Site::Prime, q()),
        (Site::Append, q()),
        (Site::Trace, q()),
        (Site::Checker, checked),
    ]
}

/// The case of `site`: the first of sub-items 2 to 6 and of `rest` that names it, or None.
pub(crate) fn case(site: Site) -> Option<Case> {
    [
        call_stop_plain_tests::cases(),
        call_stop_reduce_tests::cases(),
        call_stop_sig_tests::cases(),
        call_stop_read_tests::cases(),
        call_stop_lift_tests::cases(),
        rest(),
    ]
    .into_iter()
    .flatten()
    .find(|(s, _)| *s == site)
    .map(|(_, case)| case)
}

/// Test 1, call-stop-latency.2, .5: every site of the budget, by its case, stops a call from
/// inside its loop, with the compaction at every round's few monomials, and the handover at every
/// round and the carry at any work.
#[test]
fn every_site_stops() {
    let mut failures = Vec::new();
    COMPACT_BOUND.set(Some(64));
    LARGE_LEAST.set(Some(1));
    CARRY_LEAST.set(Some(1));
    for site in every_site() {
        let Some(case) = case(site) else {
            failures.push(format!("{site:?}: no case reaches it"));
            continue;
        };
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    COMPACT_BOUND.set(None);
    LARGE_LEAST.set(None);
    CARRY_LEAST.set(None);
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 2, the item's product: the edge cases and random systems over GF(2), GF(65521), and ℚ, in
/// grevlex, lex, and `block:1`, with and without cofactors and the checker, return with every
/// spend checking what they return without the forcing.
#[test]
fn results_unchanged_with_every_spend_checking() {
    let mut failures = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for p in [2u64, 65521, 0] {
        texts.extend(EDGE.iter().map(|gens| format!("x,y,z\n{p}\n{gens}\n")));
        if p != 2 {
            texts.extend((0..4).map(|seed| random_system(p, seed)));
            texts.extend((0..2).map(|seed| random_square(p, seed)));
        }
    }
    texts.extend((0..3).map(|seed| boolean(8, 10, seed)));
    for text in &texts {
        for order in ["grevlex", "lex", "block:1"] {
            for cofactors in [false, true] {
                let mut case = Case::new(text.clone(), order, cofactors);
                if cofactors {
                    case.params.checker = Some(CheckerMode::Exact);
                }
                let plain = groebner(&case.text, &case.params);
                let (got, _) = forced(&case, None);
                let same = match (&plain, &got) {
                    (Ok(a), Ok(b)) => {
                        (&a.basis, &a.stopped, &a.cofactors, &a.certified)
                            == (&b.basis, &b.stopped, &b.cofactors, &b.certified)
                    }
                    (Err(a), Err(b)) => a == b,
                    (Ok(_), Err(_)) | (Err(_), Ok(_)) => false,
                };
                if !same {
                    failures.push(format!("{order}, cofactors {cofactors}:\n{text}"));
                }
            }
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}
