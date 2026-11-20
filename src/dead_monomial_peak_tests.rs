//! dead-monomial-peak, the item's test code: a plain run that compacts its
//! monomial table at every round's end returns what the same run that
//! never compacts returns, over GF(p) and ℚ, in every order, with
//! cofactors and the checker, on the edge cases, and under a memory
//! ceiling, its next builds taking the same kept rows; the report's
//! `compact` lines; the table's compaction itself; a learn's record; and a
//! stop after a compaction
//! (`iterations/0117-dead-monomial-peak/plan.md`, _Test code_, 1 to 5).
//!
//! A pair of calls is the same call with `COMPACT_BOUND` Some(0), a
//! compaction at every round's end, and Some(`usize::MAX`), none.

use std::time::Duration;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::call::built;
use crate::f4::COMPACT_BOUND;
use crate::id::MonoId;
use crate::monomial::{Dense16, Format};
use crate::rebuilt_reducer_tests::{TIMES, reported, systems};
use crate::report_tests::Line;
use crate::shared_tests::{FIELDS, report};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_sparse};
use crate::table::{Live, Table};
use crate::trace::NoTrace;
use crate::{CheckerMode, MonomialOrder, Outcome, Params, Stop};

/// The bound of a call that compacts at every round's end.
const EVERY: Option<usize> = Some(0);

/// The bound of a call that never compacts.
const NEVER: Option<usize> = Some(usize::MAX);

/// The call of `text` under the compaction bound `bound`.
fn call(text: &str, params: &Params, bound: Option<usize>) -> Outcome {
    COMPACT_BOUND.set(bound);
    let outcome = crate::groebner(text, params);
    COMPACT_BOUND.set(None);
    outcome.unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
}

/// The kinds of the report's lines, in order, the `compact` lines left out.
fn kinds(lines: &[Line]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.kind != "compact")
        .map(|l| l.kind.clone())
        .collect()
}

/// The `round` lines, each without its times.
fn rounds(lines: &[Line]) -> Vec<Vec<(String, String)>> {
    lines
        .iter()
        .filter(|l| l.kind == "round")
        .map(|l| {
            l.fields
                .iter()
                .filter(|(k, _)| !TIMES.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .collect()
}

/// Asserts that the call of `text` returns with a compaction at every
/// round's end what it returns with none, its report the same but the
/// `compact` lines and the times, and returns the outcome that compacts.
fn pair(name: &str, text: &str, params: &Params) -> Outcome {
    let never = call(text, params, NEVER);
    let every = call(text, params, EVERY);
    assert_eq!(every.basis, never.basis, "{name}: the bases");
    assert_eq!(
        every.cofactors, never.cofactors,
        "{name}: the cofactor texts"
    );
    assert_eq!(every.stopped, never.stopped, "{name}: the stops");
    assert_eq!(every.certified, never.certified, "{name}: the verdicts");
    assert_eq!(every.flaw, never.flaw, "{name}: the flaws");
    assert_eq!(every.stats, never.stats, "{name}: the stats");
    let (e, n) = (report(&every), report(&never));
    assert_eq!(kinds(&e), kinds(&n), "{name}: the report's lines");
    assert_eq!(rounds(&e), rounds(&n), "{name}: the round lines");
    every
}

/// Test 1, dead-monomial-peak.1, .2, .4 to .7, .10, .11: every call that
/// compacts at every round's end returns what it returns with no
/// compaction, its round lines the same but the times, `kept` among them:
/// random sparse and dense systems and the edge cases over 7, 65521,
/// 2^31 − 1, 2^61 − 1, and ℚ, the signatures off and on; a dense system
/// and an eliminable one in lex and `block:1`; a system over ten variables,
/// whose monomials take two words, and one whose run widens its format;
/// with cofactors and the exact checker; and under a memory ceiling.
#[test]
fn every_result_stays() {
    for p in FIELDS {
        let mut texts = systems(p);
        for (k, gens) in EDGE.iter().enumerate() {
            texts.push((format!("edge {k}"), format!("x,y,z\n{p}\n{gens}\n")));
        }
        for (system, text) in &texts {
            for signatures in [false, true] {
                let mut params = reported(MonomialOrder::Grevlex);
                params.signatures = signatures;
                let name = format!("{system} over {p}, signatures {signatures}");
                let outcome = pair(&name, text, &params);
                assert!(outcome.basis.is_some(), "{name}: a basis");
            }
        }
    }
    let eliminable = "x,y,z\n7\nx+z,\nz^2+1\n";
    for order in ["lex", "block:1"] {
        let params = reported(order.parse().unwrap());
        for (name, text) in [
            ("dense33-p7-4", dense(3, 3, 7, 4)),
            ("eliminable", eliminable.to_string()),
        ] {
            pair(&format!("{name} in {order}"), &text, &params);
        }
    }
    let params = reported(MonomialOrder::Grevlex);
    let sparse = random_sparse(4, 65521, 1);
    let (_, generators) = sparse.split_once('\n').unwrap();
    let wide = format!("x1,x2,x3,x4,x5,x6,x7,x8,x9,x10\n{generators}");
    let outcome = pair("sparse4-p65521-1 over ten variables", &wide, &params);
    assert!(
        report(&outcome)
            .iter()
            .any(|l| l.is("run", &[("width", "8")])),
        "ten variables in lanes of 8 bits, two words a monomial"
    );
    let widens = "x,y,z\n65521\nx^100*y+y^2+z+1,\nx*y^30+x^2*z+x,\nz^2+x*y+3\n";
    let outcome = pair("a run that widens", widens, &params);
    assert!(
        report(&outcome)
            .iter()
            .any(|l| l.is("run", &[("width", "16"), ("why", "wider")])),
        "a run that widens"
    );
    for p in [65521, 0] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        if built(&params).is_err() {
            continue;
        }
        let name = format!("sparse4-p{p}-1 with cofactors");
        let outcome = pair(&name, &random_sparse(4, p, 1), &params);
        assert!(outcome.cofactors.is_some(), "{name}: the cofactor text");
        assert_eq!(outcome.certified, Some(true), "{name}: certified");
    }
    let mut params = reported(MonomialOrder::Grevlex);
    params.memory = Some(crate::os::resident() + (1 << 30));
    pair(
        "sparse5-p65521-1 under a ceiling",
        &random_sparse(5, 65521, 1),
        &params,
    );
}

/// A field of a `compact` line as a count.
fn count(line: &Line, key: &str) -> usize {
    line.get(key)
        .unwrap_or_else(|| panic!("a compact line names `{key}`"))
        .parse()
        .unwrap()
}

/// Test 2, dead-monomial-peak.1, .4, .9, .10: compacting at every round's
/// end, the report holds a `compact` line after each round line of the
/// plain run, which names its round, keeps at most what the table held,
/// and holds at least what the compaction before kept; on one system at
/// least a line keeps fewer. With no compaction, and with the default
/// bound on these systems, the report holds no `compact` line.
#[test]
fn the_compact_lines() {
    let mut texts: Vec<(String, String)> = (1..=4)
        .map(|s| (format!("sparse5-{s}"), random_sparse(5, 65521, s)))
        .collect();
    texts.push(("dense33-1".to_string(), dense(3, 3, 65521, 1)));
    let params = reported(MonomialOrder::Grevlex);
    let mut dropped = false;
    for (name, text) in &texts {
        let lines = report(&call(text, &params, EVERY));
        let mut kept_before = 0;
        let mut compacts = 0;
        for (k, line) in lines.iter().enumerate() {
            if line.kind != "compact" {
                continue;
            }
            compacts += 1;
            let before = lines[..k]
                .iter()
                .rev()
                .find(|l| l.kind != "compact")
                .expect("a line before a compact line");
            assert_eq!(before.kind, "round", "{name}: a compact line after a round");
            assert_eq!(
                line.get("round"),
                before.get("number"),
                "{name}: the compact line's round"
            );
            let (monomials, kept) = (count(line, "monomials"), count(line, "kept"));
            assert!(kept <= monomials, "{name}: kept {kept} of {monomials}");
            assert!(
                monomials >= kept_before,
                "{name}: {monomials} monomials after {kept_before} kept"
            );
            dropped |= kept < monomials;
            kept_before = kept;
        }
        let plain_rounds = lines.iter().filter(|l| l.kind == "round").count();
        let constant = lines.iter().any(|l| l.kind == "constant");
        assert_eq!(
            compacts + usize::from(constant),
            plain_rounds,
            "{name}: a compact line after each round"
        );
        for bound in [NEVER, None] {
            let lines = report(&call(text, &params, bound));
            assert!(
                lines.iter().all(|l| l.kind != "compact"),
                "{name}: a compact line under the bound {bound:?}"
            );
        }
    }
    assert!(dropped, "no compaction kept fewer monomials than it held");
}

/// The exponents of `id` in `table`.
fn exponents(table: &Table<Dense16>, id: MonoId) -> Vec<u32> {
    let mut e = Vec::new();
    table.exponents(id, &mut e);
    e
}

/// Every monomial over three variables of degree at most 13, 560 of them,
/// interned in a fresh table, with the column slot `i + 1` and the stamp
/// `i % 5 + 1` of the i-th set: the table and the ids in order.
fn filled() -> (Table<Dense16>, Vec<MonoId>) {
    let mut table = Table::new(Dense16::new(3), &mut Xoshiro256PlusPlus::seed_from_u64(7));
    let mut ids = Vec::new();
    for d in 0..=13 {
        for a in 0..=d {
            for b in 0..=d - a {
                ids.push(
                    table
                        .intern(&[a, b, d - a - b], &mut Budget::unbounded())
                        .unwrap(),
                );
            }
        }
    }
    for (i, &id) in ids.iter().enumerate() {
        table.set_column(id, MonoId::at(i + 1));
        table.set_stamp(id, u32::try_from(i % 5 + 1).unwrap());
    }
    (table, ids)
}

/// Test 3, dead-monomial-peak.1 to .3, .5: a table of 560 monomials, its
/// slot array grown past 2^10 slots, compacted to every third id keeps
/// those in their order, each with its exponents, degree, hash, mask,
/// column slot, and stamp, at its slot array's length; a kept monomial
/// interns to its new id and a dropped one to the next. A compaction of
/// every id keeps every id, and one of none leaves an empty table.
#[test]
fn the_table_compacts() {
    let (mut table, ids) = filled();
    assert_eq!(ids.len(), 560, "the monomials of degree at most 13");
    assert_eq!(table.len(), 560, "every monomial interned once");
    let slots = table.slot_count();
    assert!(slots > 1 << 10, "the slot array grew: {slots}");
    let olds: Vec<(Vec<u32>, u32, u64, u32, MonoId, u32)> = ids
        .iter()
        .map(|&u| {
            (
                exponents(&table, u),
                table.degree(u),
                table.hash(u),
                table.mask(u),
                table.column(u),
                table.stamp(u),
            )
        })
        .collect();
    let mut live = Live::new(table.len());
    for &u in ids.iter().step_by(3) {
        live.mark(u);
    }
    let kept = table.compact(&mut live, &mut Budget::unbounded()).unwrap();
    assert_eq!(kept, 187, "every third of 560");
    assert_eq!(table.len(), kept, "the table holds the kept monomials");
    assert_eq!(table.slot_count(), slots, "the slot array's length");
    for (k, (old, &u)) in olds.iter().zip(&ids).enumerate().step_by(3) {
        let new = live.id(u);
        assert_eq!(new, MonoId::at(k / 3), "the kept id {k} in order");
        let now = (
            exponents(&table, new),
            table.degree(new),
            table.hash(new),
            table.mask(new),
            table.column(new),
            table.stamp(new),
        );
        assert_eq!(&now, old, "monomial {k} at its new id {new}");
        assert_eq!(
            table.intern(&old.0, &mut Budget::unbounded()).unwrap(),
            new,
            "monomial {k} interned"
        );
    }
    assert_eq!(table.len(), kept, "no kept monomial interned twice");
    let dropped = &olds[1].0;
    assert_eq!(
        table.intern(dropped, &mut Budget::unbounded()).unwrap(),
        MonoId::at(kept),
        "a dropped monomial interned afresh"
    );

    let (mut all, ids) = filled();
    let mut live = Live::new(all.len());
    for &u in &ids {
        live.mark(u);
    }
    assert_eq!(
        all.compact(&mut live, &mut Budget::unbounded()),
        Ok(560),
        "every id kept"
    );
    for &u in &ids {
        assert_eq!(live.id(u), u, "the id {u} stays");
    }
    for (i, &u) in ids.iter().enumerate() {
        assert_eq!(all.column(u), MonoId::at(i + 1), "the column slot of {u}");
    }

    let (mut none, _) = filled();
    let mut live = Live::new(none.len());
    assert_eq!(
        none.compact(&mut live, &mut Budget::unbounded()),
        Ok(0),
        "no id kept"
    );
    assert_eq!(none.len(), 0, "an empty table");
    assert_eq!(
        none.intern(&[1, 2, 3], &mut Budget::unbounded()).unwrap(),
        MonoId::at(0),
        "id 0 afresh"
    );
    assert_eq!(
        none.intern(&[1, 2, 3], &mut Budget::unbounded()).unwrap(),
        MonoId::at(0),
        "found again"
    );
}

/// Test 4, dead-monomial-peak.1, .7: a learn whose plain run compacts at
/// every round's end returns the record and the outcome of the learn that
/// never compacts.
#[test]
fn the_record_stays() {
    let params = reported(MonomialOrder::Grevlex);
    for (name, text) in systems(65521) {
        let learn = |bound| {
            COMPACT_BOUND.set(bound);
            let got = crate::learn(&text, &params, &mut NoTrace);
            COMPACT_BOUND.set(None);
            got.unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };
        let (never, never_record) = learn(NEVER);
        let (every, every_record) = learn(EVERY);
        assert!(never_record.is_some(), "{name}: a record");
        assert_eq!(every_record, never_record, "{name}: the records");
        assert_eq!(every.basis, never.basis, "{name}: the bases");
        assert_eq!(every.stats, never.stats, "{name}: the stats");
    }
}

/// Test 5, dead-monomial-peak.8, which call-stop-latency.8 replaced: under
/// deadlines of 0 to 32 ms, compacting at every round's end, a call returns
/// its full basis or stops at the deadline with no basis.
#[test]
fn a_stop_after_a_compaction() {
    for p in [65521, 0] {
        let text = if p == 0 {
            random_sparse(4, p, 1)
        } else {
            random_sparse(5, p, 1)
        };
        let full = call(&text, &reported(MonomialOrder::Grevlex), EVERY)
            .basis
            .unwrap();
        for ms in [0, 1, 2, 4, 8, 16, 32] {
            let mut params = reported(MonomialOrder::Grevlex);
            params.deadline = Some(Duration::from_millis(ms));
            let outcome = call(&text, &params, EVERY);
            let name = format!("{ms} ms over {p}");
            match outcome.stopped {
                Some(Stop::Deadline) => assert!(outcome.basis.is_none(), "{name}: {outcome:?}"),
                None => assert_eq!(outcome.basis.unwrap(), full, "{name}"),
                Some(Stop::Cancel | Stop::Memory) => panic!("{name}: {outcome:?}"),
            }
        }
    }
}
