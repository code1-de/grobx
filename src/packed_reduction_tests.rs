//! gf2-reduction-time, sub-item 2's test code: the packed reduction of
//! plain rounds and of signature rounds over `Gf2` against the reduction
//! by the pivot rows' tails, a stopped packed reduction, and the entries a
//! packed round spends (`iterations/0125-gf2-reduction-time/plan.md`,
//! _Test code_, 7 to 10).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stop;
use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::field::{Field, Gf2};
use crate::halt::Halt;
use crate::id::{ElementId, MonoId, RowId, RowSource};
use crate::matrix::{self, Buffers, Matrix, Pivot, Reduction};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::packed::{
    PACK_FORCED, PACKED_ENTRIES, PACKED_ROW_ENTRIES, PACKED_TABLE, PACKED_WORDS, packs,
};
use crate::poly::{Basis, Poly};
use crate::shared_tests::{budget, passed_after};
use crate::signature_zero_tests::{Made, Round, made};
use crate::systems::Rng;
use crate::table::Table;

/// What a reduction returns, to compare: its rows made, rows that reduced
/// to zero, pivots, and order, its reductions, and its zeros.
pub(crate) type Outcome = (Made<u8>, u64, u64);

pub(crate) fn outcome(reduction: &Reduction<Gf2>) -> Outcome {
    (made(reduction), reduction.reductions, reduction.zeros)
}

/// `f` with `PACK_FORCED` at `packs` on this thread, and back to None
/// after.
pub(crate) fn forced<T>(packs: bool, f: impl FnOnce() -> T) -> T {
    PACK_FORCED.set(Some(packs));
    let out = f();
    PACK_FORCED.set(None);
    out
}

/// A plain round's matrix over GF(2) whose K columns are known: the
/// monomials of degree up to 12 on four variables in decreasing grevlex
/// order, of which the first `early + k` are its columns; `early` of them,
/// the first and others drawn, are led by a reducer row each, and the
/// other `k` by none. The `given` given rows each lead at a drawn
/// reducer's column, the first at the first column. Every row's tail is a
/// random set of the columns right of its lead, each kept with probability
/// `density` in 8, and the first given row's tail holds every K column, so
/// that each is a column of the matrix.
pub(crate) struct Plain {
    field: Gf2,
    basis: Basis<Gf2, Rank1>,
    matrix: Matrix,
}

impl Plain {
    pub(crate) fn new(seed: u64, early: usize, k: usize, given: usize, density: u64) -> Plain {
        let field = Gf2::new();
        let mut rng = Rng::new(seed ^ 0x9fa0);
        let mut table = Table::new(
            Dense16::new(4),
            &mut Xoshiro256PlusPlus::seed_from_u64(seed),
        );
        let mut monomials: Vec<MonoId> = Vec::new();
        for d in (0..=12u32).rev() {
            let mut of_d = Vec::new();
            for a in 0..=d {
                for b in 0..=d - a {
                    for c in 0..=d - a - b {
                        of_d.push(
                            table
                                .intern(&[a, b, c, d - a - b - c], &mut Budget::unbounded())
                                .unwrap(),
                        );
                    }
                }
            }
            of_d.sort_by(|&u, &v| table.cmp(&Grevlex, v, u));
            monomials.extend(of_d);
        }
        let width = early + k;
        assert!(
            width <= monomials.len(),
            "{width} columns past the monomials"
        );
        let columns = &monomials[..width];
        // Which of the columns a reducer leads: the first among them, so that
        // the first given row's tail reaches every K column.
        let mut reduced = vec![true; width];
        for _ in 0..k {
            loop {
                let c = 1 + rng.below(width as u64 - 1) as usize;
                if reduced[c] {
                    reduced[c] = false;
                    break;
                }
            }
        }
        let leads: Vec<usize> = (0..width).filter(|&c| reduced[c]).collect();
        let mut basis: Basis<Gf2, Rank1> = Basis::new();
        let one = table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
        let element = |lead: usize, all_k: bool, rng: &mut Rng, basis: &mut Basis<Gf2, Rank1>| {
            let mut terms = vec![(columns[lead], field.coef(1))];
            for c in lead + 1..width {
                if (all_k && !reduced[c]) || rng.below(8) < density {
                    terms.push((columns[c], field.coef(1)));
                }
            }
            let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
            basis.push(g, false).unwrap();
            RowSource {
                element: ElementId::at(basis.len() - 1),
                multiplier: one,
            }
        };
        let mut rows = Vec::new();
        for i in 0..given {
            let lead = if i == 0 {
                0
            } else {
                leads[rng.below(leads.len() as u64) as usize]
            };
            rows.push(element(lead, i == 0, &mut rng, &mut basis));
        }
        for &lead in &leads {
            rows.push(element(lead, false, &mut rng, &mut basis));
        }
        let mut matrix = matrix::build_rows(
            &mut table,
            &basis,
            &Grevlex,
            rows,
            &mut Buffers::new(),
            &mut budget(),
        )
        .unwrap();
        matrix.given = given;
        assert_eq!(matrix.columns(), width, "the columns of the matrix");
        Plain {
            field,
            basis,
            matrix,
        }
    }

    /// `matrix::reduce` under `budget` and fresh buffers, with the
    /// back-substitution where `full` holds.
    pub(crate) fn reduce(&self, full: bool, budget: &mut Budget) -> Result<Reduction<Gf2>, Halt> {
        self.reduce_in(full, budget, &mut Buffers::new())
    }

    /// As `reduce`, from `bufs`.
    pub(crate) fn reduce_in(
        &self,
        full: bool,
        budget: &mut Budget,
        bufs: &mut Buffers<Gf2>,
    ) -> Result<Reduction<Gf2>, Halt> {
        matrix::reduce::<Gf2, Dense16, Rank1, NoCofactors>(
            &self.field,
            &self.basis,
            &self.matrix,
            &mut NoCofactors,
            budget,
            full,
            None,
            bufs,
            None,
            None,
        )
    }
}

/// The K columns of test 7's matrices: none, one, a word less one, a word,
/// a word and one, and past two words.
pub(crate) const K: [usize; 6] = [0, 1, 63, 64, 65, 130];

/// Test 7: `matrix::reduce` over `Gf2` with `PACK_FORCED` on and off, on
/// plain rounds of each K of `K`, sparse and dense tails, with the
/// back-substitution and without: one outcome. A round of K = 0 reduces
/// every row to zero.
#[test]
fn packed_plain_rounds() {
    for (i, &k) in K.iter().enumerate() {
        for seed in 0..6u64 {
            for density in [1, 4] {
                let m = Plain::new(
                    0x9fb0 + 16 * i as u64 + seed,
                    40 + 7 * seed as usize,
                    k,
                    24,
                    density,
                );
                for full in [false, true] {
                    let case = format!("k {k}, seed {seed}, density {density}, full {full}");
                    let sparse = forced(false, || m.reduce(full, &mut budget())).unwrap();
                    let packed = forced(true, || m.reduce(full, &mut budget())).unwrap();
                    assert!(
                        sparse.packed.is_none() && packed.packed.is_some(),
                        "{case}: the round packs where it is forced to"
                    );
                    assert_eq!(outcome(&packed), outcome(&sparse), "{case}");
                    if k == 0 {
                        assert!(packed.made.is_empty(), "{case}: every row reduces to zero");
                    }
                }
            }
        }
    }
}

/// Which of the cases test 8 asserts present a signature round's
/// reduction shows: a reducer row that sets its column's pivot after the
/// first reductant in signature order, a reducer row reduced as a
/// reductant, and a reducer row dropped for its signature.
fn cases(reduction: &Reduction<Gf2>, given: usize) -> [bool; 3] {
    let at = |r: RowId| reduction.taken.iter().position(|&q| q == r).unwrap();
    let first = reduction
        .taken
        .iter()
        .position(|r| r.index() < given)
        .unwrap_or(usize::MAX);
    let late = reduction.pivot.iter().any(|p| match *p {
        Pivot::Row(q) => q.index() >= given && at(q) > first,
        Pivot::None | Pivot::New(_) => false,
    });
    let pivots: Vec<RowId> = reduction
        .pivot
        .iter()
        .filter_map(|p| match *p {
            Pivot::Row(q) => Some(q),
            Pivot::None | Pivot::New(_) => None,
        })
        .collect();
    let reduced: Vec<RowId> = reduction
        .made
        .iter()
        .map(|r| r.row)
        .chain(reduction.zero.iter().copied())
        .filter(|r| r.index() >= given)
        .collect();
    let dropped = reduction
        .taken
        .iter()
        .any(|&r| r.index() >= given && !pivots.contains(&r) && !reduced.contains(&r));
    [late, !reduced.is_empty(), dropped]
}

/// Test 8: `matrix::reduce_sig` over `Gf2` with `PACK_FORCED` on and off
/// on random signature rounds with tails: one outcome; among the rounds a
/// reducer row set after the first reductant, a reducer row reduced as a
/// reductant, and a reducer row dropped for its signature, each present.
#[test]
fn packed_signature_rounds() {
    let mut seen = [false; 3];
    for seed in 0..24u64 {
        for (elements, reducers, reductants, tail) in
            [(8, 16, 8, 4), (12, 24, 16, 6), (20, 48, 32, 12)]
        {
            for long in [false, true] {
                let round = Round::random(
                    Gf2::new(),
                    0x9fc0 + seed,
                    elements,
                    reducers,
                    reductants,
                    tail,
                    long,
                );
                let case = format!("seed {seed}, {elements} elements, long {long}");
                let sparse = forced(false, || round.reduce(None));
                let packed = forced(true, || round.reduce(None));
                assert!(
                    sparse.packed.is_none() && packed.packed.is_some(),
                    "{case}: the round packs where it is forced to"
                );
                assert_eq!(outcome(&packed), outcome(&sparse), "{case}");
                for (s, c) in seen.iter_mut().zip(cases(&sparse, round.matrix.given)) {
                    *s |= c;
                }
            }
        }
    }
    assert_eq!(
        seen, [true; 3],
        "a later reducer row, a reducer reduced, a reducer dropped"
    );
}

/// The units of work past a reduction's setup at which test 9's budgets
/// stop it.
const EXTRA: [u64; 10] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 64];

/// Test 9: a packed round under a budget whose deadline has passed, with
/// its setup's spend, the matrix's rows and columns, and 1 to 64 units more
/// between full checks: it stops with `Stop::Deadline`, at the walk of the
/// reduced rows or at a row's scan, as the units fall; the same buffers
/// then reduce the round as fresh ones do.
#[test]
fn a_stopped_packed_reduction() {
    let m = Plain::new(0x9fd0, 60, 70, 30, 4);
    let setup = (m.matrix.rows.len() + m.matrix.columns()) as u64;
    let mut b = budget();
    let before = b.counts().0;
    let fresh = forced(true, || m.reduce(true, &mut b)).unwrap();
    let spent = b.counts().0 - before;
    assert!(
        spent > EXTRA[EXTRA.len() - 1],
        "the round spends {spent} entries"
    );
    for extra in EXTRA {
        let mut bufs = Buffers::new();
        let stopped = forced(true, || {
            m.reduce_in(true, &mut passed_after(setup + extra), &mut bufs)
        });
        assert!(
            matches!(stopped, Err(Halt::Stop(Stop::Deadline))),
            "{extra} units: {made:?}",
            made = stopped.map(|r| r.made.len())
        );
        let again = forced(true, || m.reduce_in(true, &mut budget(), &mut bufs)).unwrap();
        assert_eq!(
            outcome(&again),
            outcome(&fresh),
            "{extra} units, the buffers after the stop"
        );
    }
}

/// Test 10: a hand-built round, its columns c0 > … > c5: the given rows
/// c0 + c1 and c2 + c4 + c5, the reducer rows c0 + c2 + c4, c2 + c3, and
/// c3 + c5. `matrix::reduce` takes c0 + c1, the sparser, as c0's pivot and
/// reduces c0 + c2 + c4 in its place: the early pivots lead c0, c2, and
/// c3, and the K columns are c1, c4, and c5. The reduced rows: c3's c5;
/// c2's c5, by adding c3's, 1 entry; c0's c1. Row c2 + c4 + c5, first by
/// its leading column, adds c2's, 1 entry, and ends c4, a made row. Row
/// c0 + c2 + c4 adds c0's and c2's, 2 entries, then the made row at c4, 1
/// entry, and ends c1 + c5. The round spends 5 entries and makes c4 and
/// c1 + c5.
#[test]
fn the_entries_a_packed_round_spends() {
    let field = Gf2::new();
    let mut table = Table::new(
        Dense16::new(4),
        &mut Xoshiro256PlusPlus::seed_from_u64(0x9fe0),
    );
    let mut c: Vec<MonoId> = [
        [3, 0, 0, 0],
        [2, 1, 0, 0],
        [2, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 2, 1, 0],
        [0, 0, 3, 0],
    ]
    .iter()
    .map(|e| table.intern(e, &mut Budget::unbounded()).unwrap())
    .collect();
    c.sort_by(|&u, &v| table.cmp(&Grevlex, v, u));
    let one = table.intern(&[0; 4], &mut Budget::unbounded()).unwrap();
    let mut basis: Basis<Gf2, Rank1> = Basis::new();
    let row = |cols: &[usize], basis: &mut Basis<Gf2, Rank1>| {
        let mut terms: Vec<(MonoId, u8)> = cols.iter().map(|&i| (c[i], 1)).collect();
        basis
            .push(
                Poly::from_terms(&mut terms, &field, &table, &Grevlex, ()),
                false,
            )
            .unwrap();
        RowSource {
            element: ElementId::at(basis.len() - 1),
            multiplier: one,
        }
    };
    let rows = vec![
        row(&[0, 1], &mut basis),
        row(&[2, 4, 5], &mut basis),
        row(&[0, 2, 4], &mut basis),
        row(&[2, 3], &mut basis),
        row(&[3, 5], &mut basis),
    ];
    let mut matrix = matrix::build_rows(
        &mut table,
        &basis,
        &Grevlex,
        rows,
        &mut Buffers::new(),
        &mut budget(),
    )
    .unwrap();
    matrix.given = 2;
    let mut b = budget();
    let before = b.counts().0;
    let reduction = forced(true, || {
        matrix::reduce::<Gf2, Dense16, Rank1, NoCofactors>(
            &field,
            &basis,
            &matrix,
            &mut NoCofactors,
            &mut b,
            false,
            None,
            &mut Buffers::new(),
            None,
            None,
        )
    })
    .unwrap();
    let monomials = |r: &matrix::Reduced<Gf2>| -> Vec<MonoId> {
        r.cols.iter().map(|x| matrix.monomials[x.index()]).collect()
    };
    let made: Vec<Vec<MonoId>> = reduction.made.iter().map(monomials).collect();
    assert_eq!(reduction.packed, Some(1), "three K columns, a word");
    assert_eq!(made, vec![vec![c[4]], vec![c[1], c[5]]], "the made rows");
    assert_eq!(
        b.counts().0 - before,
        5,
        "the entries the packed round subtracted"
    );
}

/// The bounds of `packs` (gf2-reduction-time.13, .14, .21, and .22), each
/// at its edge: a round packs at the least entries, the least entries a
/// row, the most words, and the most words of reduced rows, and not one
/// past any of them; a round of no K column packs in no word; and
/// `PACK_FORCED` decides either way where it holds a choice.
#[test]
fn the_bounds_of_a_packed_round() {
    // columns, rows, entries, early
    let words = |early: usize, k: usize| (early + k, 2 * early, early, k);
    let fit = |(columns, rows, early, _): (usize, usize, usize, usize), entries: usize| {
        packs(columns, rows, entries, early)
    };
    let most = PACKED_WORDS * 64;
    let enough = |rows: usize| PACKED_ENTRIES.max(PACKED_ROW_ENTRIES * rows);
    let round = words(100, most);
    assert_eq!(fit(round, enough(round.1)), Some(PACKED_WORDS as u32));
    assert_eq!(fit(round, enough(round.1) - 1), None, "an entry short");
    let wide = words(100, most + 1);
    assert_eq!(fit(wide, enough(wide.1)), None, "a word past");
    let none = words(100, 0);
    assert_eq!(fit(none, enough(none.1)), Some(0), "no K column");
    // The least entries a row past the least entries.
    let long = words(PACKED_ENTRIES, 64);
    assert_eq!(fit(long, PACKED_ROW_ENTRIES * long.1), Some(1));
    assert_eq!(
        fit(long, PACKED_ROW_ENTRIES * long.1 - 1),
        None,
        "the rows short"
    );
    // The most words of reduced rows, at one word a row and at the most.
    let table = words(PACKED_TABLE / PACKED_WORDS, most);
    assert_eq!(fit(table, enough(table.1)), Some(PACKED_WORDS as u32));
    let past = words(PACKED_TABLE / PACKED_WORDS + 1, most);
    assert_eq!(
        fit(past, enough(past.1)),
        None,
        "the reduced rows' words past"
    );
    let forced_fit = forced(true, || fit(past, 0));
    assert_eq!(forced_fit, Some(PACKED_WORDS as u32), "forced to pack");
    let forced_off = forced(false, || fit(round, enough(round.1)));
    assert_eq!(forced_off, None, "forced not to pack");
}
