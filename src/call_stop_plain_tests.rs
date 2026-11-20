//! call-stop-latency, sub-item 2's test code: the plain run's selection, update, divisor search,
//! pair rows, columns, kept matrix, decoding, entering, sugars, monic inputs, and compaction each
//! stop a call from inside, and the selection, the update, and the divisor search spend as they go
//! (`iterations/0138-call-stop-latency/plan.md`, _Test code_, `call_stop_plain_tests.rs`).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, spends_as_it_goes, stops_at};
use crate::divisibility_tests::Engine;
use crate::f4::COMPACT_BOUND;
use crate::field::{Field, Gf31};
use crate::halt::Halt;
use crate::id::RowSource;
use crate::matrix::{self, Buffers};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::pairs::{self, HOLES};
use crate::poly::{Basis, Poly};
use crate::shared_tests::{below, rng};
use crate::sort::RUN;
use crate::systems::{boolean, random_system};
use crate::table::{Limit, Table};

/// The plain run's sites, each with a call that reaches it: boolean systems over GF(2), random
/// systems over GF(65521) in grevlex and in `block:1`, which keeps the plain run in an order without
/// a grading, where a lex call runs it in grevlex and changes the order by FGLM; the compaction
/// over GF(2), whose rounds intern 64 monomials and more, where `COMPACT_BOUND` holds 64. Seed 5's
/// system runs rounds over GF(65521), where seed 2's goes without one.
pub(crate) fn cases() -> Vec<(Site, Case)> {
    let gf2 = || Case::new(boolean(12, 16, 1), "grevlex", false);
    let gfp = || Case::new(random_system(65521, 5), "grevlex", false);
    vec![
        (Site::Select, gf2()),
        (Site::Update, gf2()),
        (Site::Divisors, gf2()),
        (Site::PairRows, gf2()),
        (Site::Enter, gf2()),
        (Site::Columns, gfp()),
        (Site::Kept, gfp()),
        (Site::Decode, gfp()),
        (Site::Monic, gfp()),
        (Site::Compact, gf2()),
        (
            Site::Sugar,
            Case::new(random_system(65521, 5), "block:1", false),
        ),
    ]
}

/// The variables of the engines of tests 2 and 3.
const N: usize = 9;

/// The degree of every lead of the engines of tests 2 and 3: no two distinct monomials of one
/// degree divide each other, so every lead joins `G_min`.
const DEGREE: u32 = 6;

/// The leads of the engines of tests 2 and 3: every monomial of degree `DEGREE` on `N`
/// variables, 3,003, in a drawn order, whose updates leave a set of tens of thousands of places,
/// and of more than a chunk of `RUN` where they fill no hole (`unfilled`), so that each pass of a
/// selection spends at least twice.
fn leads() -> Vec<(Vec<u32>, u32)> {
    let mut all = Vec::new();
    every(&mut [0u32; N], 0, DEGREE, &mut all);
    let mut r = rng(31);
    for i in (1..all.len()).rev() {
        all.swap(i, below(&mut r, i as u64 + 1) as usize);
    }
    all.into_iter().map(|e| (e, DEGREE)).collect()
}

/// The exponent vectors that agree with `e` before `at` and spend `left` from `at` on, appended
/// to `out`.
fn every(e: &mut [u32], at: usize, left: u32, out: &mut Vec<Vec<u32>>) {
    if at == e.len() - 1 {
        e[at] = left;
        out.push(e.to_vec());
        return;
    }
    for k in 0..=left {
        e[at] = k;
        every(e, at + 1, left - k, out);
    }
}

/// An engine with every lead of `leads` entered under a budget that never stops.
fn engine(leads: &[(Vec<u32>, u32)]) -> Engine<Dense16> {
    let mut engine: Engine<Dense16> = Engine::new(N, 3);
    for (e, degree) in leads {
        let (got, _) = engine.enter(&Grevlex, e, *degree);
        got.expect("an update under an unbounded budget");
    }
    engine
}

/// Test 1: every site of the plain run stops a call from inside it, the call's result unchanged
/// with every spend checking (call-stop-latency.12 to .15).
#[test]
fn every_site_of_the_plain_run_stops() {
    let mut failures = Vec::new();
    COMPACT_BOUND.set(Some(64));
    for (site, case) in cases() {
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    COMPACT_BOUND.set(None);
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// The engine of `engine` built with no hole filled (`HOLES`), whose set holds every place the B
/// criterion vacated beside its live pairs: more than a chunk of `RUN` places, where the pushes
/// that fill its holes leave fewer (pair-index-upkeep-time.1).
fn unfilled(leads: &[(Vec<u32>, u32)]) -> Engine<Dense16> {
    HOLES.set(false);
    let e = engine(leads);
    HOLES.set(true);
    e
}

/// Test 2, call-stop-latency.12: a selection over a set of tens of thousands of places spends as
/// it passes over them, and stops inside under a passed deadline.
#[test]
fn a_selection_spends_as_it_passes() {
    let leads = leads();
    let held = unfilled(&leads).pairs.held();
    assert!(held > RUN, "{held} places, too few");
    let select = |b: &mut Budget| -> Result<usize, Stop> {
        let mut e = unfilled(&leads);
        match pairs::select(&mut e.pairs, &mut e.table, &Grevlex, b) {
            Ok(taken) => Ok(taken.len()),
            Err(Limit::Stop(s)) => Err(s),
            Err(limit @ (Limit::Overflow | Limit::Size)) => panic!("the selection: {limit:?}"),
        }
    };
    if let Err(e) = spends_as_it_goes(Site::Select, 1 << 12, 4, select) {
        panic!("{e}");
    }
}

/// Test 3, call-stop-latency.13: one update over a set of tens of thousands of places spends as
/// it scans, and stops inside under a passed deadline.
#[test]
fn an_update_spends_as_it_scans() {
    let leads = leads();
    let (last, rest) = leads.split_last().expect("leads");
    let update = |b: &mut Budget| -> Result<(), Stop> {
        let mut e = engine(rest);
        match e.enter_under(&Grevlex, &last.0, last.1, 0, b).0 {
            Ok(()) => Ok(()),
            Err(Halt::Stop(s)) => Err(s),
            Err(
                halt @ (Halt::Overflow
                | Halt::Size
                | Halt::Misfit(_)
                | Halt::Handover
                | Halt::Outworked),
            ) => panic!("the update: {halt:?}"),
        }
    };
    if let Err(e) = spends_as_it_goes(Site::Update, 1 << 10, 4, update) {
        panic!("{e}");
    }
}

/// The members of test 4's basis.
pub(crate) const MEMBERS: usize = 2_000;

/// The multipliers of each member in test 4's given rows: `x_{N-1}` to the powers below this.
const POWERS: u32 = 3;

/// Test 4's build under `budget`: a basis of `MEMBERS` elements on `N` variables, each a lead of
/// degree `DEGREE + 1` in the first `N - 1` variables and three drawn tail terms of degree
/// `DEGREE` or less, and given rows of every member times each power of `x_{N-1}` below
/// `POWERS`. A lead's degree in the first `N - 1` variables passes every tail product's, so every
/// divisor search of the build, over `G_min`'s `MEMBERS` members, finds none. The rows the build
/// made, or its halt.
fn divisor_free(budget: &mut Budget) -> Result<usize, Halt> {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(N), &mut rng(41));
    let mut u = Budget::unbounded();
    let mut leads = Vec::new();
    every(&mut [0u32; N - 1], 0, DEGREE + 1, &mut leads);
    let mut r = rng(43);
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    for lead in leads.iter().take(MEMBERS) {
        let mut e = lead.clone();
        e.push(0);
        let mut terms = vec![(table.intern(&e, &mut u).unwrap(), field.coef(1))];
        while terms.len() < 4 {
            let mut tail = vec![0u32; N];
            for _ in 0..below(&mut r, u64::from(DEGREE) + 1) {
                tail[below(&mut r, N as u64) as usize] += 1;
            }
            let m = table.intern(&tail, &mut u).unwrap();
            if terms.iter().all(|&(t, _)| t != m) {
                terms.push((m, field.coef(1 + below(&mut r, 65520))));
            }
        }
        let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
        basis.push_min(&table, g, &mut u).unwrap();
    }
    let mut rows = Vec::new();
    for j in 0..POWERS {
        let mut power = vec![0u32; N];
        power[N - 1] = j;
        let multiplier = table.intern(&power, &mut u).unwrap();
        rows.extend(basis.min().iter().map(|&element| RowSource {
            element,
            multiplier,
        }));
    }
    let m = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        rows,
        &mut Buffers::new(),
        budget,
    )?;
    Ok(m.rows.len())
}

/// Test 4, call-stop-latency.14: a build whose divisor searches find none, over a basis of
/// `MEMBERS` members, spends its searches as it goes, and stops inside under a passed deadline.
#[test]
fn a_divisor_search_spends() {
    let given = MEMBERS * POWERS as usize;
    let rows = divisor_free(&mut Budget::unbounded()).unwrap();
    assert_eq!(rows, given, "a search found a divisor");
    let build = |b: &mut Budget| -> Result<usize, Stop> {
        match divisor_free(b) {
            Ok(rows) => Ok(rows),
            Err(Halt::Stop(s)) => Err(s),
            Err(
                halt @ (Halt::Overflow
                | Halt::Size
                | Halt::Misfit(_)
                | Halt::Handover
                | Halt::Outworked),
            ) => panic!("the build: {halt:?}"),
        }
    };
    if let Err(e) = spends_as_it_goes(Site::Divisors, 1 << 12, 4, build) {
        panic!("{e}");
    }
}
