//! gf2-forced-zeros-time, the item's test code: calls over GF(2) whose
//! plain rounds rule rows zero by sketches return the basis the exact
//! checker certifies, with every round in packed rows or in sparse rows, the
//! echelon filled and the sketch drawn again, and under forced bounds on the
//! row work; the run line and the round lines name the sketches where the
//! rule on the rows' entries takes them, and nowhere else; the same seed
//! draws the same sketches; and a sketch's residues rule the span of its
//! pivots and of the rows kept since, and only it
//! (`iterations/0134-gf2-forced-zeros-time/plan.md`, _Test code_, 1 to 6).

use crate::budget::Budget;
use crate::columns::Row;
use crate::gf2_bound_losses_tests::{bounds, certified, halts, systems};
use crate::gf2_start_over_tests::{call, reported};
use crate::groebner;
use crate::id::MonoId;
use crate::matrix::{ECHELON_FORCED, SKETCH_ECHELON, SKETCH_FORCED, SKETCH_ROW_ENTRIES};
use crate::packed::PACK_FORCED;
use crate::rebuilt_reducer_tests::TIMES;
use crate::report_tests::Line;
use crate::sketch::Sketch;
use crate::systems::{Rng, cubics, random_sparse};

/// The call of `text` under `seed`, signatures on as a call takes them,
/// with `SKETCH_FORCED` at `sketch`, `PACK_FORCED` at `pack`, and
/// `ECHELON_FORCED` at `cap` on this thread, each back to None after: its
/// basis and its report's lines.
fn forced(
    text: &str,
    seed: u64,
    sketch: Option<bool>,
    pack: Option<bool>,
    cap: Option<usize>,
) -> (String, Vec<Line>) {
    SKETCH_FORCED.set(sketch);
    PACK_FORCED.set(pack);
    ECHELON_FORCED.set(cap);
    let out = call(text, &reported(seed, true), None);
    SKETCH_FORCED.set(None);
    PACK_FORCED.set(None);
    ECHELON_FORCED.set(None);
    out
}

/// The dense boolean cubics of tests 1 and 4, by name: `cubics(n, m,
/// seed)` for n = 8 to 10, m = n and n + 2, two seeds each, whose rounds
/// hold rows of 16 entries or more in the mean and of fewer.
fn dense_cubics() -> Vec<(String, String)> {
    let mut texts = Vec::new();
    for n in 8..=10usize {
        for m in [n, n + 2] {
            for seed in 0..2u64 {
                texts.push((format!("cubics({n}, {m}, {seed})"), cubics(n, m, seed)));
            }
        }
    }
    texts
}

/// The round lines of `lines` whose run line names `reduce=sketches`.
fn sketched_rounds(lines: &[Line]) -> Vec<&Line> {
    let mut sketched = false;
    let mut rounds = Vec::new();
    for line in lines {
        match line.kind.as_str() {
            "run" => sketched = line.get("reduce") == Some("sketches"),
            "round" if sketched => rounds.push(line),
            _ => {}
        }
    }
    rounds
}

/// The integer of `key` on `line`, which must hold it.
fn num(line: &Line, key: &str) -> u64 {
    let v = line
        .get(key)
        .unwrap_or_else(|| panic!("no {key} on {line:?}"));
    v.parse()
        .unwrap_or_else(|e| panic!("{key}={v} on {line:?}: {e}"))
}

/// Test 1: gf2-forced-zeros-time.1, .2, .3, .8, .9, .11. Over GF(2), on
/// `gf2_bound_losses_tests`' boolean systems, the edge cases on x, y, z,
/// and the dense boolean cubics, each call with the sketches forced on, forced off, and by the rule, and
/// with every round packed, none, and by the bounds, returns the basis the
/// call with cofactors and signatures off returns, which the exact checker
/// certifies. Where the sketches are forced on, every round line of a run
/// that names `reduce=sketches` names its draws, and its zero rows are at
/// least its reductions to zero; where they are forced off, no round line
/// names draws. Among the calls with the sketches forced on, a round draws a
/// sketch and a sketch rules rows zero, in packed rows and in sparse rows.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn sketched_calls_return_the_certified_basis() {
    // Rounds that drew, and rounds whose sketch ruled rows zero, packed and
    // sparse.
    let (mut drawn, mut ruled_packed, mut ruled_sparse) = (0, 0, 0);
    let mut texts = systems(true);
    texts.extend(dense_cubics());
    for (name, text) in texts {
        let seed = 0xf0e1;
        let want = certified(&name, &text, seed);
        for sketch in [Some(true), Some(false), None] {
            for pack in [Some(true), Some(false), None] {
                let case = format!("{name}, sketches {sketch:?}, packed {pack:?}");
                let (basis, lines) = forced(&text, seed, sketch, pack, None);
                assert_eq!(basis, want, "{case}");
                match sketch {
                    Some(false) => {
                        for line in lines.iter().filter(|l| l.kind == "round") {
                            assert_eq!(line.get("sketches"), None, "{case}: {line:?}");
                        }
                        continue;
                    }
                    Some(true) => {}
                    None => continue,
                }
                for line in sketched_rounds(&lines) {
                    let sketches = num(line, "sketches");
                    let (zero_rows, zero) = (num(line, "zero-rows"), num(line, "zero"));
                    assert!(zero_rows >= zero, "{case}: {line:?}");
                    drawn += usize::from(sketches > 0);
                    if zero_rows > zero {
                        match line.get("packed") {
                            Some("no") => ruled_sparse += 1,
                            Some(_) => ruled_packed += 1,
                            None => panic!("{case}: no packed on {line:?}"),
                        }
                    }
                }
            }
        }
    }
    assert!(drawn > 0, "no round drew a sketch");
    assert!(
        ruled_packed > 0,
        "no sketch ruled a row zero in packed rows"
    );
    assert!(
        ruled_sparse > 0,
        "no sketch ruled a row zero in sparse rows"
    );
}

/// Test 2: gf2-forced-zeros-time.3, .4. Test 1's boolean systems with the
/// sketches forced on and the echelon's cap forced to 1, 2, and 3: each call
/// returns test 1's basis, and among them a round draws its sketch again
/// after its echelon filled, two draws or more on its line.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn a_full_echelon_draws_the_sketch_again() {
    assert_eq!(SKETCH_ECHELON, 97, "128 bits less the 31 of ZERO_BOUND");
    let mut again = 0;
    for (name, text) in systems(false) {
        let seed = 0xf0e2;
        let want = certified(&name, &text, seed);
        for cap in [1, 2, 3] {
            let case = format!("{name}, cap {cap}");
            let (basis, lines) = forced(&text, seed, Some(true), None, Some(cap));
            assert_eq!(basis, want, "{case}");
            again += sketched_rounds(&lines)
                .iter()
                .filter(|l| num(l, "sketches") >= 2)
                .count();
        }
    }
    assert!(again > 0, "no round drew its sketch again");
}

/// Test 3: gf2-forced-zeros-time.8, .12. Test 1's systems, the edge cases
/// among them, with the sketches forced on under `gf2_bound_losses_tests`'
/// forced bounds on the row work: each call returns test 1's basis, every
/// `outworked` line names the forced bound, and a call halts by the
/// projection, its `work` at most its `bound`, and one past the bound.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn sketched_calls_under_a_bound() {
    let (mut projected, mut past) = (0, 0);
    for (name, text) in systems(true) {
        let seed = 0xf0e3;
        let want = certified(&name, &text, seed);
        for bound in bounds() {
            let case = format!("{name}, forced {bound:?}");
            SKETCH_FORCED.set(Some(true));
            let (basis, lines) = call(&text, &reported(seed, true), bound);
            SKETCH_FORCED.set(None);
            assert_eq!(basis, want, "{case}");
            let halts = halts(&case, &lines, bound);
            projected += halts.outworked.iter().filter(|(w, b)| w <= b).count();
            past += halts.outworked.iter().filter(|(w, b)| w > b).count();
        }
    }
    assert!(projected > 0, "no halt by the projection");
    assert!(past > 0, "no halt past the bound");
}

/// Test 4: gf2-forced-zeros-time.7, .11, .13. By the rule on the rows'
/// entries: over GF(2) a call without cofactors names `reduce=sketches` on
/// its plain runs' run lines, and a round line of such a run names its draws
/// exactly where its matrix's rows hold `SKETCH_ROW_ENTRIES` entries or more
/// in the mean, on test 1's systems. Among the calls, rounds of both kinds
/// stand, and a round draws a sketch unforced. A call with cofactors names
/// `reduce=sketches` on no run line, nor do calls over GF(3) and GF(65521),
/// and none of these names draws on a round line.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn the_report_names_the_sketches_by_the_rule() {
    let (mut long, mut short, mut drawn) = (0, 0, 0);
    let mut texts: Vec<(String, String)> = systems(true);
    texts.extend(dense_cubics());
    for (name, text) in &texts {
        let seed = 0xf0e4;
        let (basis, lines) = call(text, &reported(seed, true), None);
        let want = certified(name, text, seed);
        assert_eq!(basis, want, "{name}");
        let plain = lines
            .iter()
            .filter(|l| l.kind == "run" && l.get("engine") == Some("plain"));
        for line in plain {
            assert_eq!(line.get("reduce"), Some("sketches"), "{name}: {line:?}");
        }
        for line in sketched_rounds(&lines) {
            let (rows, entries) = (num(line, "rows"), num(line, "entries"));
            let takes = entries >= SKETCH_ROW_ENTRIES as u64 * rows;
            assert_eq!(line.get("sketches").is_some(), takes, "{name}: {line:?}");
            if takes {
                long += 1;
                drawn += usize::from(num(line, "sketches") > 0);
            } else {
                short += 1;
            }
        }
        // A call with cofactors runs no sketch.
        let mut params = reported(seed, true);
        params.cofactors = true;
        let outcome = groebner(text, &params).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(
            outcome.basis.as_deref(),
            Some(want.as_str()),
            "{name}, cofactors"
        );
        let report = outcome.report.unwrap();
        assert!(
            !report.contains("reduce=sketches"),
            "{name}, cofactors: {report}"
        );
        assert!(
            !report.contains(" sketches="),
            "{name}, cofactors: {report}"
        );
    }
    assert!(
        long > 0 && short > 0,
        "{long} long and {short} short rounds"
    );
    assert!(drawn > 0, "no round drew a sketch by the rule");
    for p in [3, 65521] {
        for seed in 0..4 {
            let text = random_sparse(5, p, seed);
            let (_, lines) = call(&text, &reported(seed, true), None);
            for line in &lines {
                assert_ne!(line.get("reduce"), Some("sketches"), "p {p}: {line:?}");
                assert_eq!(line.get("sketches"), None, "p {p}: {line:?}");
            }
        }
    }
}

/// Test 5: gf2-forced-zeros-time.10. Test 1's systems with the sketches
/// forced on, each called twice under one seed: the same basis and the same
/// report lines, kinds and fields, but the times.
#[test]
fn one_seed_draws_the_same_sketches() {
    for (name, text) in systems(true) {
        let (first, a) = forced(&text, 0xf0e5, Some(true), None, None);
        let (second, b) = forced(&text, 0xf0e5, Some(true), None, None);
        assert_eq!(first, second, "{name}");
        assert_eq!(a.len(), b.len(), "{name}");
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.kind, y.kind, "{name}");
            let keep = |l: &Line| -> Vec<(String, String)> {
                l.fields
                    .iter()
                    .filter(|(k, _)| !TIMES.contains(&k.as_str()) && k != "time")
                    .cloned()
                    .collect()
            };
            if matches!(x.kind.as_str(), "round" | "run" | "ran" | "handover") {
                assert_eq!(keep(x), keep(y), "{name}: {x:?} against {y:?}");
            }
        }
    }
}

/// A row of the columns `cols` as `Sketch` reads it.
fn row(cols: &[MonoId]) -> Row<'_> {
    Row::Wide(cols)
}

/// The sum over GF(2) of the rows `rows`, its columns increasing.
fn sum(rows: &[&[MonoId]]) -> Vec<MonoId> {
    let mut all: Vec<MonoId> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    all.sort_unstable();
    let mut out: Vec<MonoId> = Vec::new();
    for c in all {
        if out.last() == Some(&c) {
            out.pop();
        } else {
            out.push(c);
        }
    }
    out
}

/// Test 6: gf2-forced-zeros-time.2, .3, .12, `Sketch` alone. On 200
/// columns, 120 pivot rows at drawn columns, each a lead and a tail of 0 to
/// 9 drawn columns right of it, under three seeds of the words: after a
/// draw every pivot row and every sum of drawn pivot rows has residue 0, a
/// row with an entry at a column without a pivot and drawn pivot rows beside
/// it does not, and the draw spends at least the tails' entries and the
/// columns. Each row of residue other than 0 kept, every sum of drawn kept
/// rows and pivot rows has residue 0, until the echelon holds its cap, which
/// `keep` then says, and a draw empties it.
#[test]
#[allow(
    clippy::cast_possible_truncation,
    reason = "test code: the draws narrow to indices below COLUMNS"
)]
fn a_sketch_rules_its_span() {
    const COLUMNS: usize = 200;
    for seed in [1u64, 2, 3] {
        let mut rng = Rng::new(0x5ce7 ^ seed);
        // The pivot rows by column: lead first, then the tail, increasing.
        let mut pivots: Vec<Option<Vec<MonoId>>> = vec![None; COLUMNS];
        for _ in 0..120 {
            let c = rng.below(COLUMNS as u64 - 1) as usize;
            let mut cols = vec![MonoId::at(c)];
            for _ in 0..rng.below(10) {
                cols.push(MonoId::at(
                    c + 1 + rng.below((COLUMNS - c - 1) as u64) as usize,
                ));
            }
            cols.sort_unstable();
            cols.dedup();
            pivots[c] = Some(cols);
        }
        let led: Vec<usize> = (0..COLUMNS).filter(|&c| pivots[c].is_some()).collect();
        let free: Vec<usize> = (0..COLUMNS).filter(|&c| pivots[c].is_none()).collect();
        let tails: usize = pivots.iter().flatten().map(|p| p.len() - 1).sum();
        let mut words = Rng::new(seed);
        let mut budget = Budget::unbounded();
        let mut sketch = Sketch::new();
        sketch
            .draw(
                COLUMNS,
                |c| pivots[c].as_ref().map(|p| row(&p[1..])),
                || (u128::from(words.next()) << 64) | u128::from(words.next()),
                &mut budget,
            )
            .unwrap();
        assert_eq!(sketch.draws(), 1);
        assert!(
            budget.spent() >= (tails + COLUMNS) as u64,
            "the draw spends"
        );
        let pick = |rng: &mut Rng, from: &[usize]| from[rng.below(from.len() as u64) as usize];
        for &c in &led {
            assert_eq!(
                sketch.residue(row(pivots[c].as_ref().unwrap())),
                0,
                "pivot {c}"
            );
        }
        for _ in 0..50 {
            let some: Vec<&[MonoId]> = (0..=rng.below(6))
                .map(|_| pivots[pick(&mut rng, &led)].as_deref().unwrap())
                .collect();
            let s = sum(&some);
            assert_eq!(sketch.residue(row(&s)), 0, "a sum of pivots {s:?}");
            let off = [MonoId::at(pick(&mut rng, &free))];
            let mut with: Vec<&[MonoId]> = some.clone();
            with.push(&off);
            let t = sum(&with);
            assert_ne!(
                sketch.residue(row(&t)),
                0,
                "{t:?} has an entry off the pivots"
            );
        }
        // Rows made since the draw: each a column without a pivot and a
        // tail, kept until the cap.
        let mut kept: Vec<Vec<MonoId>> = Vec::new();
        let cap = 6;
        loop {
            let c = pick(&mut rng, &free);
            let mut made = vec![MonoId::at(c)];
            made.extend((0..rng.below(5)).map(|_| MonoId::at(pick(&mut rng, &free))));
            let made = sum(&[&made]);
            let x = sketch.residue(row(&made));
            if x == 0 {
                // A row in the span of the pivots and the rows kept.
                continue;
            }
            let full = sketch.keep(x, cap);
            kept.push(made);
            for _ in 0..20 {
                let mut some: Vec<&[MonoId]> = (0..rng.below(4))
                    .map(|_| pivots[pick(&mut rng, &led)].as_deref().unwrap())
                    .collect();
                for k in &kept {
                    if rng.below(2) == 1 {
                        some.push(k);
                    }
                }
                let s = sum(&some);
                assert_eq!(sketch.residue(row(&s)), 0, "a sum of pivots and kept rows");
            }
            let n = kept.len();
            assert_eq!(full, n >= cap, "{n} kept");
            if full {
                break;
            }
        }
        sketch
            .draw(
                COLUMNS,
                |c| pivots[c].as_ref().map(|p| row(&p[1..])),
                || (u128::from(words.next()) << 64) | u128::from(words.next()),
                &mut budget,
            )
            .unwrap();
        assert_eq!(sketch.draws(), 2);
        // The echelon is empty: a kept row is no longer in its span.
        let x = sketch.residue(row(&kept[0]));
        assert_ne!(x, 0, "a kept row after the draw");
        assert!(!sketch.keep(x, cap), "one residue in an empty echelon");
    }
}
