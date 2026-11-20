//! gf2-packed-backsub-time, the item's test code: the back-substitution of a
//! round in packed rows, by the tables of the sums of `TABLE_ROWS` made
//! rows, makes the reduced row echelon form that the back-substitution of
//! one row at a time makes, and spends the entries that form adds and the
//! tables' sums; plain rounds over GF(2) reduce in packed rows to the
//! rows they reduce to in sparse rows; calls over GF(2) whose rounds pack
//! return the basis the exact checker certifies; and a back-substitution
//! the budget stops returns the stop
//! (`iterations/0152-gf2-packed-backsub-time/plan.md`, _Test code_, 1 to
//! 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::Stop;
use crate::gf2_bound_losses_tests::{certified, systems};
use crate::gf2_start_over_tests::{call, reported};
use crate::halt::Halt;
use crate::id::MonoId;
use crate::matrix::Buffers;
use crate::packed::{PACK_FORCED, Rows, TABLE_ROWS};
use crate::packed_reduction_tests::{Plain, forced, outcome};
use crate::report_tests::Line;
use crate::shared_tests::{budget, passed_after};
use crate::systems::{Rng, cubics};

/// A set of made rows over `width` K columns as the forward reduction
/// keeps them: `n` rows of distinct leading columns, kept in a drawn order,
/// each holding beside its leading column a drawn set of the columns right
/// of it, each with probability `density` in 8, but none that leads a row
/// kept before it. Each row's columns, increasing, in the order kept.
fn echelon(seed: u64, width: usize, n: usize, density: u64) -> Vec<Vec<usize>> {
    assert!(n <= width, "{n} rows past {width} columns");
    let mut rng = Rng::new(seed ^ 0xbac5);
    let mut leads: Vec<usize> = (0..width).collect();
    for i in 0..n {
        let j = i + rng.below((width - i) as u64) as usize;
        leads.swap(i, j);
    }
    leads.truncate(n);
    let mut kept = vec![false; width];
    let mut rows = Vec::with_capacity(n);
    for &lead in &leads {
        let mut cols = vec![lead];
        for (c, &before) in kept.iter().enumerate().skip(lead + 1) {
            if !before && rng.below(8) < density {
                cols.push(c);
            }
        }
        kept[lead] = true;
        rows.push(cols);
    }
    rows
}

/// The reduced row echelon form of `rows` by the back-substitution of one
/// row at a time, by decreasing leading column: each row plus the result
/// of every other row whose leading column it holds. Per row in the order
/// of `rows`, its result's columns, increasing; and the entries of the
/// results the one-row form adds, in all.
fn one_at_a_time(rows: &[Vec<usize>], width: usize) -> (Vec<Vec<usize>>, usize) {
    let mut by_lead: Vec<usize> = (0..rows.len()).collect();
    by_lead.sort_by_key(|&i| std::cmp::Reverse(rows[i][0]));
    let mut lead_of = vec![None; width];
    for (i, row) in rows.iter().enumerate() {
        lead_of[row[0]] = Some(i);
    }
    let mut done: Vec<Option<Vec<bool>>> = vec![None; rows.len()];
    let mut added = 0;
    for &i in &by_lead {
        let mut dense = vec![false; width];
        for &c in &rows[i] {
            dense[c] = true;
        }
        let mut sum = dense.clone();
        for (c, &set) in dense.iter().enumerate().skip(rows[i][0] + 1) {
            if let (true, Some(other)) = (set, lead_of[c]) {
                let result = done[other].as_ref().expect("a row of a larger lead first");
                added += result.iter().filter(|&&x| x).count();
                for (a, &x) in sum.iter_mut().zip(result) {
                    *a ^= x;
                }
            }
        }
        done[i] = Some(sum);
    }
    let results = done
        .into_iter()
        .map(|d| {
            let d = d.expect("every row back-substituted");
            (0..width).filter(|&c| d[c]).collect()
        })
        .collect();
    (results, added)
}

/// The entries the tables of `results`, by decreasing leading column,
/// build: every block of `TABLE_ROWS` rows but the last builds 2^k − 1
/// sums, each from the sum without its lowest row and that row, so that
/// the block's row t is added in 2^(k − 1 − t) of them.
fn table_entries(results: &[Vec<usize>]) -> usize {
    let mut by_lead: Vec<&Vec<usize>> = results.iter().collect();
    by_lead.sort_by_key(|r| std::cmp::Reverse(r[0]));
    let blocks = by_lead.len().div_ceil(TABLE_ROWS);
    let mut entries = 0;
    for block in by_lead.chunks(TABLE_ROWS).take(blocks.saturating_sub(1)) {
        let k = block.len();
        for (t, row) in block.iter().enumerate() {
            entries += row.len() << (k - 1 - t);
        }
    }
    entries
}

/// The made rows counts of test 1: none, one, a block less one, a block,
/// a block and one, two blocks and one, and many blocks with a short last.
const COUNTS: [usize; 9] = [0, 1, 7, 8, 9, 17, 40, 63, 100];

/// The K columns of test 1: a word less one, a word, a word and one, and
/// past three words.
const WIDTHS: [usize; 4] = [63, 64, 65, 200];

/// Test 1: gf2-packed-backsub-time.1, .3 to .8. `packed::Rows::back_substitute`
/// on made rows kept as the forward reduction keeps them, over each width
/// of `WIDTHS`, each count of `COUNTS` that fits, sparse and dense, three
/// seeds each: it hears of exactly the rows whose reduced row echelon form
/// differs from their row, each with the tail the back-substitution of one
/// row at a time makes, and spends as subtracted the entries that form
/// adds and the entries of the tables' sums. Rows that lead the K
/// columns' last word, and rows of their leading column alone, stand among
/// them.
#[test]
fn the_back_substitution_by_tables() {
    let mut cases = 0;
    for &width in &WIDTHS {
        for &n in COUNTS.iter().filter(|&&n| n <= width) {
            for density in [1, 4] {
                for seed in 0..3u64 {
                    let case = format!("width {width}, {n} rows, density {density}, seed {seed}");
                    let rows = echelon(
                        0xbac0 + 64 * seed + width as u64 + n as u64,
                        width,
                        n,
                        density,
                    );
                    let mut packed = Rows::new(std::iter::repeat_n(None, width));
                    for row in &rows {
                        let cols: Vec<MonoId> = row.iter().map(|&c| MonoId::at(c)).collect();
                        packed.keep(&cols);
                    }
                    let mut by_lead: Vec<usize> = (0..n).collect();
                    by_lead.sort_by_key(|&i| std::cmp::Reverse(rows[i][0]));
                    let leads: Vec<MonoId> =
                        by_lead.iter().map(|&i| MonoId::at(rows[i][0])).collect();
                    let mut heard: Vec<(usize, Vec<usize>)> = Vec::new();
                    let mut b = budget();
                    let before = b.counts().0;
                    packed
                        .back_substitute(
                            &leads,
                            |i, tail| heard.push((i, tail.iter().map(|c| c.index()).collect())),
                            &mut b,
                        )
                        .unwrap_or_else(|e| panic!("{case}: {e:?}"));
                    let (want, added) = one_at_a_time(&rows, width);
                    let mut expected: Vec<(usize, Vec<usize>)> = Vec::new();
                    for (place, &i) in by_lead.iter().enumerate() {
                        if want[i] != rows[i] {
                            expected.push((place, want[i][1..].to_vec()));
                        }
                    }
                    assert_eq!(heard, expected, "{case}");
                    assert_eq!(
                        b.counts().0 - before,
                        (added + table_entries(&want)) as u64,
                        "{case}: the entries spent"
                    );
                    cases += usize::from(!expected.is_empty());
                }
            }
        }
    }
    assert!(cases > 0, "no case changed a row");
}

/// Test 2: gf2-packed-backsub-time.1, .2. `matrix::reduce` over `Gf2` with the
/// back-substitution, `PACK_FORCED` on and off, on plain rounds of 9 to
/// 200 K columns whose given rows make up to every K column's row, sparse
/// and dense tails: one outcome, the round packed where it is forced to,
/// and a packed round makes more than two tables' rows.
#[test]
fn packed_rounds_back_substitute_as_sparse_rounds() {
    let mut most = 0;
    for (i, &k) in [9usize, 64, 65, 130, 200].iter().enumerate() {
        for seed in 0..4u64 {
            for density in [1, 4] {
                let m = Plain::new(
                    0xbad0 + 16 * i as u64 + seed,
                    30 + 5 * seed as usize,
                    k,
                    k + 8,
                    density,
                );
                let case = format!("k {k}, seed {seed}, density {density}");
                let sparse = forced(false, || m.reduce(true, &mut budget())).unwrap();
                let packed = forced(true, || m.reduce(true, &mut budget())).unwrap();
                assert!(
                    sparse.packed.is_none() && packed.packed.is_some(),
                    "{case}: the round packs where it is forced to"
                );
                assert_eq!(outcome(&packed), outcome(&sparse), "{case}");
                most = most.max(packed.made.len());
            }
        }
    }
    assert!(most > 2 * TABLE_ROWS, "the most made rows {most}");
}

/// The call of `text` under `seed`, signatures on as a call takes them,
/// with `PACK_FORCED` at `pack` on this thread, and back to None after: its
/// basis and its report's lines.
fn packed_call(text: &str, seed: u64, pack: Option<bool>) -> (String, Vec<Line>) {
    PACK_FORCED.set(pack);
    let out = call(text, &reported(seed, true), None);
    PACK_FORCED.set(None);
    out
}

/// Test 3: gf2-packed-backsub-time.1, .2. Over GF(2), on `gf2_bound_losses_tests`'
/// boolean systems, the edge cases on x, y, z, and the dense boolean cubics
/// `cubics(n, m, seed)` for n = 8 and 9, each call with `PACK_FORCED` at
/// Some(true), Some(false), and None returns the basis the call with
/// cofactors and signatures off returns, which the exact checker
/// certifies. Among the calls with every round packed, a round in packed
/// rows makes more than two tables' rows.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn packed_calls_return_the_certified_basis() {
    let mut most = 0u64;
    let mut texts = systems(true);
    for n in 8..=9usize {
        for seed in 0..2u64 {
            texts.push((format!("cubics({n}, {n}, {seed})"), cubics(n, n, seed)));
        }
    }
    for (name, text) in texts {
        let seed = 0xbae1;
        let want = certified(&name, &text, seed);
        for pack in [Some(true), Some(false), None] {
            let case = format!("{name}, packed {pack:?}");
            let (basis, lines) = packed_call(&text, seed, pack);
            assert_eq!(basis, want, "{case}");
            match pack {
                Some(true) => {}
                Some(false) | None => continue,
            }
            for line in lines.iter().filter(|l| l.kind == "round") {
                if line.get("packed").is_some_and(|p| p != "no") {
                    let pivots: u64 = line.get("pivots").unwrap().parse().unwrap();
                    most = most.max(pivots);
                }
            }
        }
    }
    assert!(
        most > 2 * TABLE_ROWS as u64,
        "the most pivots of a packed round {most}"
    );
}

/// Test 4: gf2-packed-backsub-time.3, .8. A plain round in packed rows whose forward
/// reduction spends u units, under a budget whose deadline has passed with
/// u + e units between full checks, for each e of 1 to the
/// back-substitution's units by doubling: without the back-substitution it
/// reduces with no full check, and with it the back-substitution stops
/// with `Stop::Deadline`; the same buffers then reduce the round as fresh
/// ones do.
#[test]
fn a_stopped_back_substitution() {
    let m = Plain::new(0xbaf0, 40, 40, 30, 4);
    let mut b = budget();
    let forward = forced(true, || m.reduce(false, &mut b)).unwrap();
    let u = b.spent();
    let mut b = budget();
    let fresh = forced(true, || m.reduce(true, &mut b)).unwrap();
    let (back, made, checks) = (b.spent() - u, fresh.made.len(), b.checks());
    assert!(
        made > TABLE_ROWS && back > 0 && checks == 0,
        "{made} made rows, {back} units in the back-substitution, {checks} checks"
    );
    assert!(!forward.made.is_empty(), "the round makes rows");
    let mut e = 1;
    while e <= back {
        let mut b = passed_after(u + e);
        let ahead = forced(true, || m.reduce(false, &mut b));
        assert!(
            ahead.is_ok() && b.checks() == 0,
            "{e} units: the forward reduction reached a check"
        );
        let mut bufs = Buffers::new();
        let stopped = forced(true, || {
            m.reduce_in(true, &mut passed_after(u + e), &mut bufs)
        });
        assert!(
            matches!(stopped, Err(Halt::Stop(Stop::Deadline))),
            "{e} units: {made:?}",
            made = stopped.map(|r| r.made.len())
        );
        let again = forced(true, || m.reduce_in(true, &mut budget(), &mut bufs)).unwrap();
        assert_eq!(
            outcome(&again),
            outcome(&fresh),
            "{e} units, the buffers after the stop"
        );
        e *= 2;
    }
}
