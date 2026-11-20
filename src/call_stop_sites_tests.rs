//! call-stop-latency, sub-item 1's test code: a stop forced at a site of the budget, the table's
//! growth and compaction in chunks, and the keyed sorts by pieces; and the harness the item's other
//! test files take (`iterations/0138-call-stop-latency/plan.md`, _Test code_,
//! `call_stop_sites_tests.rs`, 1 to 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::Rng;

use crate::budget::{Budget, CHECKED, CHECKS_FORCED, PROBE, STOP_FORCED, Site};
use crate::call::built;
use crate::id::MonoId;
use crate::monomial::{Dense8, Dense16, Format};
use crate::order::Grevlex;
use crate::shared_tests::{below, fill, params, passed_after, rng};
use crate::sig::Sig;
use crate::sort::{self, PIECE_FORCED, by_signature, sort_decreasing};
use crate::table::{Limit, Live, PLACE, Table};
use crate::{Error, MonomialOrder, Outcome, Params, Stop, Threads, groebner};

/// A call of the test code: its instance text and its parameters.
pub(crate) struct Case {
    pub(crate) text: String,
    pub(crate) params: Params,
}

impl Case {
    /// The call of `text` in the order named, with cofactors where `cofactors`, seed 1.
    pub(crate) fn new(text: impl Into<String>, order: &str, cofactors: bool) -> Case {
        let order: MonomialOrder = order.parse().expect("a name of the family");
        let mut params = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), 1);
        params.cofactors = cofactors;
        Case {
            text: text.into(),
            params,
        }
    }
}

/// The call of `case` with every spend and tick checking and a stop forced at the first check
/// of `stop` where Some, and the sites of the full checks it made, in order.
pub(crate) fn forced(case: &Case, stop: Option<Site>) -> (Result<Outcome, Error>, Vec<Site>) {
    CHECKED.with_borrow_mut(Vec::clear);
    CHECKS_FORCED.set(true);
    STOP_FORCED.set(stop);
    let got = groebner(&case.text, &case.params);
    CHECKS_FORCED.set(false);
    STOP_FORCED.set(None);
    (got, CHECKED.with_borrow_mut(std::mem::take))
}

/// What a call returned, the parts the forcing must leave alone.
pub(crate) fn returned(
    o: &Outcome,
) -> (
    &Option<String>,
    &Option<Stop>,
    &Option<String>,
    &Option<bool>,
) {
    (&o.basis, &o.stopped, &o.cofactors, &o.certified)
}

/// Whether `case`'s call, with every spend checking, makes a full check at `site` and returns
/// what it returns without the forcing, and, with a stop forced at `site`'s first check, returns
/// stopped by the deadline at that check: Err with what failed. Ok where the build refuses the
/// call (`call::built`).
pub(crate) fn stops_at(site: Site, case: &Case) -> Result<(), String> {
    if built(&case.params).is_err() {
        return Ok(());
    }
    let plain = groebner(&case.text, &case.params)
        .map_err(|e| format!("{site:?}: the call without the forcing failed: {e}"))?;
    if plain.stopped.is_some() {
        return Err(format!("{site:?}: the call without the forcing stopped"));
    }
    let (got, sites) = forced(case, None);
    let got =
        got.map_err(|e| format!("{site:?}: the call with every spend checking failed: {e}"))?;
    if returned(&got) != returned(&plain) {
        let (got, plain) = (returned(&got), returned(&plain));
        return Err(format!(
            "{site:?}: with every spend checking the call returned {got:?}, not {plain:?}"
        ));
    }
    if !sites.contains(&site) {
        let checks = sites.len();
        let mut at: Vec<Site> = sites.clone();
        at.sort_by_key(|&s| s as usize);
        at.dedup();
        return Err(format!(
            "{site:?}: no full check there among {checks} checks, at {at:?}"
        ));
    }
    let (stopped, sites) = forced(case, Some(site));
    let stopped = stopped.map_err(|e| format!("{site:?}: the call stopped there failed: {e}"))?;
    match (stopped.stopped, stopped.basis.is_some()) {
        (Some(Stop::Deadline), false) => {}
        (stop @ (Some(Stop::Deadline | Stop::Cancel | Stop::Memory) | None), basis) => {
            return Err(format!(
                "{site:?}: a stop forced there returned {stop:?} and a basis {basis}"
            ));
        }
    }
    if sites.last() != Some(&site) {
        let last = sites.last();
        return Err(format!("{site:?}: the call's last check was at {last:?}"));
    }
    Ok(())
}

/// Whether `work`, run under a budget that never stops and checks every `per` units, makes at
/// least `at_least` full checks at `site`, and, run afresh under a passed deadline with `per`
/// units between two checks, stops with `Stop::Deadline` after spending less than half of what it
/// spends unstopped: Err with what failed. `work` builds its state afresh at each run.
pub(crate) fn spends_as_it_goes<T>(
    site: Site,
    per: u64,
    at_least: u64,
    mut work: impl FnMut(&mut Budget) -> Result<T, Stop>,
) -> Result<(), String> {
    let mut b = Budget::unbounded();
    b.set_work(per);
    if let Err(s) = work(&mut b) {
        return Err(format!("{site:?}: stopped with no bound: {s:?}"));
    }
    let (whole, checks) = (b.spent(), b.checked_at(site));
    if checks < at_least {
        return Err(format!(
            "{site:?}: {checks} checks there for {whole} units at {per} a check, not {at_least}"
        ));
    }
    let mut p = passed_after(per);
    match work(&mut p) {
        Err(Stop::Deadline) => {}
        Err(s @ (Stop::Cancel | Stop::Memory)) => {
            return Err(format!(
                "{site:?}: stopped by {s:?} under a passed deadline"
            ));
        }
        Ok(_) => return Err(format!("{site:?}: ran to its end under a passed deadline")),
    }
    if 2 * p.spent() >= whole {
        let spent = p.spent();
        return Err(format!(
            "{site:?}: stopped after {spent} of its {whole} units"
        ));
    }
    Ok(())
}

/// The exponents of `count` distinct monomials on `n` variables, by increasing degree, each degree
/// in a drawn order.
fn monomials(n: usize, count: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(count);
    let mut degree = 0u32;
    while out.len() < count {
        let mut level = Vec::new();
        fill(&mut vec![0; n], 0, degree, &mut level);
        let mut r = rng(seed ^ u64::from(degree));
        for i in (1..level.len()).rev() {
            level.swap(i, below(&mut r, i as u64 + 1) as usize);
        }
        out.extend(level.into_iter().take(count - out.len()));
        degree += 1;
    }
    out
}

/// Test 1, call-stop-latency.5: with every spend checking, the checks record their sites in
/// order, and a stop forced at a site comes at its first check and at no other site's; the
/// harness tells a loop that spends as it goes from one that spends once after its work.
#[test]
fn a_stop_forced_at_a_site() {
    let mut failures = Vec::new();
    CHECKED.with_borrow_mut(Vec::clear);
    CHECKS_FORCED.set(true);
    STOP_FORCED.set(Some(Site::Compact));
    let mut b = Budget::new(Instant::now(), &params(1)).unwrap();
    let got = [
        b.spend(1, Site::Place),
        b.spend(1, Site::Place),
        b.tick(Site::Row),
        b.spend(1, Site::Compact),
    ];
    CHECKS_FORCED.set(false);
    STOP_FORCED.set(None);
    let want = [Ok(()), Ok(()), Ok(()), Err(Stop::Deadline)];
    if got != want {
        failures.push(format!("the spends returned {got:?}, not {want:?}"));
    }
    let sites = CHECKED.with_borrow_mut(std::mem::take);
    let order = [Site::Place, Site::Place, Site::Row, Site::Compact];
    if sites != order {
        failures.push(format!("the checks recorded {sites:?}, not {order:?}"));
    }
    if b.checked_at(Site::Place) != 2 || b.checked_at(Site::Compact) != 1 {
        failures.push("the budget's counts per site".to_string());
    }

    // A loop over 100,000 elements, spent as it goes and spent once after.
    let as_it_goes = |b: &mut Budget| -> Result<u64, Stop> {
        let mut sum = 0u64;
        for chunk in (0..100_000u64).collect::<Vec<_>>().chunks(1024) {
            sum += chunk.iter().sum::<u64>();
            b.spend(chunk.len(), Site::Place)?;
        }
        Ok(sum)
    };
    let after = |b: &mut Budget| -> Result<u64, Stop> {
        let sum = (0..100_000u64).sum();
        b.spend(100_000, Site::Place)?;
        Ok(sum)
    };
    if let Err(e) = spends_as_it_goes(Site::Place, 4096, 20, as_it_goes) {
        failures.push(format!("a loop spent as it goes: {e}"));
    }
    if spends_as_it_goes(Site::Place, 4096, 20, after).is_ok() {
        failures.push("a loop spent once after its work passed".to_string());
    }

    // A call checks at the row operations' site and stops there: seed 5's system reduces rows,
    // where seed 1's goes without a round.
    let case = Case::new(crate::systems::random_system(65521, 5), "grevlex", false);
    if let Err(e) = stops_at(Site::Row, &case) {
        failures.push(e);
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 2, call-stop-latency.4, .6, .7: a growth of the table places its ids `PLACE` at a time,
/// a check at `Site::Place` per chunk, and a passed deadline stops it inside, after one chunk's
/// spend; the ids are the table's without a budget.
#[test]
fn a_growth_places_in_chunks() {
    let mut failures = Vec::new();
    let monos = monomials(6, 1 << 17, 1);
    let per = (PLACE * PROBE) as u64;

    // The ids under a budget that never stops, a check per chunk's spend, and the insertions
    // that grew the table, with the ids each growth placed.
    let mut t = Table::new(Dense16::new(6), &mut rng(7));
    let mut b = Budget::unbounded();
    b.set_work(per);
    let mut ids = Vec::with_capacity(monos.len());
    let mut growths = Vec::new();
    for (k, e) in monos.iter().enumerate() {
        let (slots, before) = (t.slot_count(), b.checked_at(Site::Place));
        ids.push(t.intern(e, &mut b).unwrap());
        if t.slot_count() > slots {
            let placed = t.len();
            growths.push((k, placed));
            let checks = b.checked_at(Site::Place) - before;
            let want = (placed / PLACE) as u64;
            if checks < want {
                failures.push(format!(
                    "a growth placing {placed} ids checked {checks} times, not {want}"
                ));
            }
        }
    }
    if !growths.iter().any(|&(_, placed)| placed > 2 * PLACE) {
        failures.push(format!("no growth past two chunks: {growths:?}"));
    }
    // Each monomial found again under its id, and the ids those of a table that grew under
    // `Budget::unbounded`'s own units.
    let mut plain = Table::new(Dense16::new(6), &mut rng(7));
    let mut u = Budget::unbounded();
    for (e, &id) in monos.iter().zip(&ids) {
        let again = t.intern(e, &mut u).unwrap();
        let other = plain.intern(e, &mut u).unwrap();
        if again != id || other != id {
            failures.push(format!(
                "{e:?}: {id}, then {again}, and {other} in a second table"
            ));
            break;
        }
    }
    if t.len() != monos.len() {
        let len = t.len();
        failures.push(format!("{len} monomials after the lookups"));
    }

    // The table filled up to the insertion that grows it past two chunks, then that insertion
    // under a passed deadline with a chunk's units between two checks: it stops after one
    // chunk's spend.
    let &(at, placed) = growths
        .iter()
        .find(|&&(_, placed)| placed > 2 * PLACE)
        .expect("a growth past two chunks");
    let mut t = Table::new(Dense16::new(6), &mut rng(7));
    for e in &monos[..at] {
        t.intern(e, &mut u).unwrap();
    }
    let mut p = passed_after(per);
    let got = t.intern(&monos[at], &mut p);
    match got {
        Err(Limit::Stop(Stop::Deadline)) => {}
        Ok(_) | Err(Limit::Overflow | Limit::Size | Limit::Stop(Stop::Cancel | Stop::Memory)) => {
            failures.push(format!(
                "the growth of {placed} ids under a passed deadline: {got:?}"
            ));
        }
    }
    if p.spent() >= (placed * PROBE) as u64 {
        let (spent, whole) = (p.spent(), placed * PROBE);
        failures.push(format!(
            "the growth spent {spent} units before its stop, its whole {whole}"
        ));
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// A compaction of test 3: its name, and which of the monomials, by their order of interning, it
/// keeps.
type Keep = (&'static str, fn(usize) -> bool);

/// Test 3, call-stop-latency.8: a compaction keeping none, all, half, and every third of 2^16
/// monomials keeps them, renumbered in order, found again under their new ids, as the compaction
/// under a budget that never stops does; and it spends its moves and its placement as it goes.
#[test]
fn a_compaction_moves_in_chunks() {
    let mut failures = Vec::new();
    let monos = monomials(6, 1 << 16, 3);
    let build = || {
        let mut t = Table::new(Dense16::new(6), &mut rng(5));
        let mut u = Budget::unbounded();
        let ids: Vec<MonoId> = monos.iter().map(|e| t.intern(e, &mut u).unwrap()).collect();
        (t, ids)
    };
    let keeps: [Keep; 4] = [
        ("none", |_| false),
        ("all", |_| true),
        ("half", |k| k % 2 == 0),
        ("every third", |k| k % 3 == 0),
    ];
    for (name, keep) in keeps {
        let (mut t, ids) = build();
        let mut live = Live::new(t.len());
        for (k, &id) in ids.iter().enumerate() {
            if keep(k) {
                live.mark(id);
            }
        }
        let mut b = Budget::unbounded();
        b.set_work(1 << 12);
        let kept = t.compact(&mut live, &mut b).unwrap();
        let want = (0..ids.len()).filter(|&k| keep(k)).count();
        if kept != want || t.len() != want {
            let len = t.len();
            failures.push(format!(
                "{name}: kept {kept}, {len} in the table, not {want}"
            ));
        }
        let mut u = Budget::unbounded();
        let mut next = 0u32;
        for (k, e) in monos.iter().enumerate() {
            if !keep(k) {
                continue;
            }
            let found = t.intern(e, &mut u).unwrap();
            if found != MonoId::from_raw(next) {
                failures.push(format!("{name}: {e:?} found as {found}, not {next}"));
                break;
            }
            next += 1;
        }
        if t.len() != want {
            failures.push(format!("{name}: the lookups added monomials"));
        }
    }
    let half = |b: &mut Budget| -> Result<usize, Stop> {
        let (mut t, ids) = build();
        let mut live = Live::new(t.len());
        for &id in ids.iter().step_by(2) {
            live.mark(id);
        }
        t.compact(&mut live, b)
    };
    if let Err(e) = spends_as_it_goes(Site::Compact, 1 << 12, 8, half) {
        failures.push(e);
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 4, call-stop-latency.9, .11: with `PIECE_FORCED` at 64, the keyed sorts of 0, 1, 64,
/// 65, and 20,000 monomials and of 20,000 signatures give the comparison sort's order, and the
/// large ones spend as they go at the site their caller names.
#[test]
fn keyed_sorts_by_pieces() {
    let mut failures = Vec::new();
    PIECE_FORCED.set(Some(64));
    let monos = monomials(6, 20_000, 9);
    let mut t = Table::new(Dense16::new(6), &mut rng(11));
    let mut u = Budget::unbounded();
    let all: Vec<MonoId> = monos.iter().map(|e| t.intern(e, &mut u).unwrap()).collect();
    let mut drawn = all.clone();
    let mut r = rng(13);
    for i in (1..drawn.len()).rev() {
        drawn.swap(i, below(&mut r, i as u64 + 1) as usize);
    }
    for len in [0, 1, 64, 65, drawn.len()] {
        let mut got = drawn[..len].to_vec();
        sort_decreasing(
            &t,
            &Grevlex,
            &mut got,
            &mut Budget::unbounded(),
            Site::Columns,
        )
        .unwrap();
        let mut want = drawn[..len].to_vec();
        want.sort_by(|&a, &b| t.cmp(&Grevlex, b, a));
        if got != want {
            failures.push(format!("sort_decreasing of {len} ids"));
        }
    }
    let sort_all = |b: &mut Budget| -> Result<(), Stop> {
        let mut got = drawn.clone();
        sort_decreasing(&t, &Grevlex, &mut got, b, Site::Columns)
    };
    if let Err(e) = spends_as_it_goes(Site::Columns, 1 << 12, 8, sort_all) {
        failures.push(e);
    }

    let sigs: Vec<Sig> = (0..20_000)
        .map(|_| Sig {
            monomial: drawn[below(&mut r, 3_000) as usize],
            index: below(&mut r, 4) as u32,
        })
        .collect();
    let got = by_signature(&t, &Grevlex, &sigs, &mut Budget::unbounded(), Site::SigSort).unwrap();
    let mut want: Vec<_> = (0..sigs.len()).map(crate::id::RowId::at).collect();
    want.sort_by(|&a, &b| {
        sigs[a.index()]
            .cmp(sigs[b.index()], &t, &Grevlex)
            .then(a.cmp(&b))
    });
    if got != want {
        failures.push("by_signature of 20,000 signatures".to_string());
    }
    let sort_sigs = |b: &mut Budget| by_signature(&t, &Grevlex, &sigs, b, Site::SigSort);
    if let Err(e) = spends_as_it_goes(Site::SigSort, 1 << 12, 8, sort_sigs) {
        failures.push(e);
    }
    PIECE_FORCED.set(None);
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 5, call-stop-latency.10: in `Dense8` on 40 variables, monomials of one degree whose keys
/// tie, in one run of thousands among others, sort into the comparison sort's order, and the
/// ties spend as they go.
#[test]
fn long_ties_spend() {
    let mut failures = Vec::new();
    let n = 40;
    let mut t = Table::new(Dense8::new(n), &mut rng(17));
    let mut u = Budget::unbounded();
    let mut r = rng(19);
    let mut ids = Vec::new();
    while ids.len() < 40_000 {
        let mut e = vec![0u32; n];
        for _ in 0..6 {
            e[below(&mut r, n as u64) as usize] += 1;
        }
        let id = t.intern(&e, &mut u).unwrap();
        if id.index() == ids.len() {
            ids.push(id);
        }
    }
    let mut by_key: HashMap<u128, Vec<MonoId>> = HashMap::new();
    for &id in &ids {
        by_key.entry(t.key(&Grevlex, id)).or_default().push(id);
    }
    let longest = by_key.values().max_by_key(|v| v.len()).unwrap().clone();
    if longest.len() < 2_000 {
        let len = longest.len();
        failures.push(format!("the longest run of equal keys holds {len}"));
    }
    let mut got = ids.clone();
    sort_decreasing(
        &t,
        &Grevlex,
        &mut got,
        &mut Budget::unbounded(),
        Site::Columns,
    )
    .unwrap();
    let mut want = ids.clone();
    want.sort_by(|&a, &b| t.cmp(&Grevlex, b, a));
    if got != want {
        failures.push("the sort of 40,000 monomials with long ties".to_string());
    }
    let sort_run = |b: &mut Budget| -> Result<(), Stop> {
        let mut got = longest.clone();
        sort_decreasing(&t, &Grevlex, &mut got, b, Site::Columns)
    };
    if let Err(e) = spends_as_it_goes(Site::Columns, 1 << 12, 4, sort_run) {
        failures.push(e);
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 6, call-stop-latency.9, .34: `sort::keys` with `PIECE_FORCED` at 64 gives
/// `sort_unstable`'s order over keys random, all equal, equal above their low 40 bits, distinct
/// in their top bits alone, and with many repeats, increasing and decreasing; it hands its pieces
/// to `piece` in order, each run of equal keys inside one piece; and it spends as it goes.
#[test]
fn the_keys_sorted_by_pieces() {
    let mut failures = Vec::new();
    PIECE_FORCED.set(Some(64));
    let mut r = rng(23);
    let draw = |r: &mut rand_xoshiro::Xoshiro256PlusPlus| {
        u128::from(r.next_u64()) << 64 | u128::from(r.next_u64())
    };
    let kinds: Vec<(&str, Vec<u128>)> = vec![
        ("random", (0..10_000).map(|_| draw(&mut r)).collect()),
        (
            "all equal",
            vec![0x1234_5678_9abc_def0_1234_5678_9abc_def0; 10_000],
        ),
        (
            "equal above 40 bits",
            (0..10_000)
                .map(|_| 0xffff_0000_0000_0000_0000_0000_0000_0000 | draw(&mut r) >> 88)
                .collect(),
        ),
        (
            "top bits alone",
            (0..10_000)
                .map(|_| draw(&mut r) & !((1u128 << 112) - 1))
                .collect(),
        ),
        (
            "repeats",
            (0..10_000)
                .map(|_| u128::from(below(&mut r, 300)) << 70)
                .collect(),
        ),
        ("none", Vec::new()),
        ("one", vec![7]),
        ("a piece", (0..64).map(|_| draw(&mut r)).collect()),
        ("past a piece", (0..65).map(|_| draw(&mut r)).collect()),
    ];
    for (name, keys) in &kinds {
        for decreasing in [false, true] {
            let mut got = keys.clone();
            let mut pieces: Vec<(usize, u128, u128)> = Vec::new();
            let record = |p: &mut [u128], _: &mut Budget| -> Result<(), Stop> {
                pieces.push((p.len(), p[0], p[p.len() - 1]));
                Ok(())
            };
            sort::keys(
                &mut got,
                decreasing,
                &mut Budget::unbounded(),
                Site::Columns,
                record,
            )
            .unwrap();
            let mut want = keys.clone();
            want.sort_unstable();
            if decreasing {
                want.reverse();
            }
            if got != want {
                failures.push(format!(
                    "{name}, decreasing {decreasing}: not sort_unstable's order"
                ));
            }
            let (mut at, mut last) = (0usize, None);
            for &(len, first, end) in &pieces {
                if len == 0 || got[at] != first || got[at + len - 1] != end {
                    failures.push(format!("{name}: a piece at {at} of {len} out of order"));
                    break;
                }
                if last == Some(first) {
                    failures.push(format!("{name}: equal keys across two pieces at {at}"));
                    break;
                }
                last = Some(end);
                at += len;
            }
            if at != got.len() {
                let len = got.len();
                failures.push(format!("{name}: the pieces hold {at} of {len} keys"));
            }
        }
    }
    let random = kinds[0].1.clone();
    let sort_random = |b: &mut Budget| -> Result<(), Stop> {
        let mut got = random.clone();
        sort::keys(&mut got, true, b, Site::Columns, |_, _| Ok(()))
    };
    if let Err(e) = spends_as_it_goes(Site::Columns, 1 << 10, 8, sort_random) {
        failures.push(e);
    }
    PIECE_FORCED.set(None);
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}
