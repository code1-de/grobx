//! gf2-build-time, the item's test code: the selection splits the last
//! split's sugar again after a piece one of whose elements fell, whatever
//! the piece's yield; an element's fall; the projection of a round's row
//! work over its sugar; and calls over GF(2) under lowered splits and
//! forced bounds return the basis the exact checker certifies
//! (`iterations/0159-gf2-build-time/plan.md`, _Test code_, 1 to 4).

use crate::budget::{Budget, Site};
use crate::gf2_bound_losses_tests::{bounds, certified, halts, systems, worked};
use crate::gf2_start_over_tests::{call, reported};
use crate::monomial::{Dense8, Format};
use crate::pairs::{SPLIT_BOUNDS, SPLIT_LEAST, SPLIT_TAKE, fall};
use crate::shared_tests::rng;
use crate::split_piece_time_tests::{D, World, monomials, step};
use crate::systems::cubics;
use crate::table::Table;

/// The lowered splits of test 4, the most pairs a selection takes whole
/// and the pairs a split takes, each with room for a repeat between them.
const LOWERED: [(usize, usize); 3] = [(8, 2), (16, 4), (64, 8)];

/// A set whose sugar 5 holds `3 · SPLIT_LEAST` pairs, `above` pairs of
/// sugar 6 beside them, after the selection that splits sugar 5 and a
/// note of a piece of `made` elements, `fallen` of which fell.
fn noted_world(above: usize, made: usize, fallen: usize, seed: u64) -> World {
    let lcms = monomials(D);
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let mut w = World::new(&[], seed);
    w.push_all(&lcms, 3 * l, 5);
    w.push_all(&lcms, above, 6);
    step(&mut w, "the first split of 5", 5, t, 3 * l - t);
    w.set.note_piece(t, made, fallen);
    w
}

/// Test 1: gf2-build-time.1 and .2. After a split of sugar 5, a note of a
/// piece of an element of every pair has the next selection of 5 split
/// it again where one element or more fell, and take the rest whole where
/// none did. A note of a piece of half as many elements as pairs, one of
/// which fell, splits it again too. Where the sugar holds fewer than half
/// the set's live pairs, a piece whose every element fell has the rest
/// taken whole.
#[test]
fn a_piece_whose_element_fell_splits_its_sugar_again() {
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let left = 3 * l - t;
    for (made, fallen, again) in [
        (t, 1, true),
        (t, t / 2, true),
        (t, t, true),
        (t / 2, 1, true),
        (t, 0, false),
        (t / 2, 0, false),
    ] {
        let mut w = noted_world(0, made, fallen, 0x159a ^ (made * 1031 + fallen) as u64);
        let case = format!("a piece of {made} elements of {t} pairs, {fallen} fallen");
        if again {
            step(&mut w, &case, 5, t, left - t);
        } else {
            step(&mut w, &case, 5, left, 0);
        }
    }
    let mut w = noted_world(left + 1, t, t, 0x159b);
    step(
        &mut w,
        "every element fallen, less than half the set",
        5,
        left,
        0,
    );
    let mut w = noted_world(left, t, t, 0x159c);
    step(&mut w, "every element fallen, half the set", 5, t, left - t);
}

/// Test 2: an element's fall. For every monomial of degree up to `D` on
/// the tests' variables and every sugar from its degree to `D`, `fall` is
/// the sugar less the monomial's degree: 0 where the two are equal.
#[test]
fn an_elements_fall() {
    let mut table = Table::new(Dense8::new(4), &mut rng(0x159d));
    for d in 0..=D {
        for e in monomials(d).iter().step_by(97) {
            let m = table.intern(e, &mut Budget::unbounded()).unwrap();
            for sugar in d..=D {
                assert_eq!(fall(&table, sugar, m), sugar - d, "{e:?} at sugar {sugar}");
            }
        }
    }
}

/// The rows and the rows begun of test 3: every pair up to `ROWS` rows.
const ROWS: usize = 48;

/// Whether s + r · rows · sugar / (reached · piece), rounded down, passes
/// s + u, in u128.
fn passes(s: u64, u: u64, r: u64, rows: usize, reached: usize, share: (usize, usize)) -> bool {
    let (piece, sugar) = share;
    let projected = u128::from(s)
        + u128::from(r) * rows as u128 * sugar as u128 / (reached as u128 * piece as u128);
    projected > u128::from(s) + u128::from(u)
}

/// Test 3: gf2-build-time.3 and .5. A budget bounded at u units after s
/// units of row work, and r units more: within `over_sugar(piece, sugar,
/// …)`, `projected_past(s, reached, rows)` holds exactly where s + r ·
/// rows · sugar / (reached · piece), rounded down, passes s + u, at every
/// rows and reached up to `ROWS`; before and after it, and within
/// `over_sugar(n, n, …)`, exactly where s + r · rows / reached does. A
/// nested `over_sugar` takes its own share and gives the outer one back,
/// and `over_sugar` returns what its closure returns. Within the largest
/// share a budget bounded at `u64::MAX`, which a projection past
/// `u64::MAX` does not pass, never projects past.
#[test]
fn the_projection_over_the_sugar() {
    let cases: [(u64, u64, u64); 5] = [
        (0, 1000, 1),
        (0, 1000, 999),
        (17, 1 << 20, 1 << 14),
        (1 << 30, 1 << 40, 1 << 33),
        (5, 3, 0),
    ];
    let shares: [(usize, usize); 5] = [(1, 1), (1, 2), (3, 7), (512, 18_634), (9, 9)];
    for (s, u, r) in cases {
        let mut budget = worked(s);
        assert_eq!(budget.bound(u), u, "the units of the bound");
        budget
            .spend_subtracted(usize::try_from(r).unwrap(), Site::Row)
            .expect("an unbounded budget stops no call");
        let whole = |budget: &Budget, case: &str| {
            for rows in 1..=ROWS {
                for reached in 1..=rows {
                    assert_eq!(
                        budget.projected_past(s, reached, rows),
                        passes(s, u, r, rows, reached, (1, 1)),
                        "{case}: s {s}, u {u}, r {r}, {reached} of {rows} rows"
                    );
                }
            }
        };
        whole(&budget, "before");
        for (piece, sugar) in shares {
            let got = budget.over_sugar(piece, sugar, |budget| {
                for rows in 1..=ROWS {
                    for reached in 1..=rows {
                        assert_eq!(
                            budget.projected_past(s, reached, rows),
                            passes(s, u, r, rows, reached, (piece, sugar)),
                            "s {s}, u {u}, r {r}, {reached} of {rows} rows, {piece} of {sugar}"
                        );
                    }
                }
                budget.over_sugar(1, 1, |budget| whole(budget, "nested"));
                assert_eq!(
                    budget.projected_past(s, 1, ROWS),
                    passes(s, u, r, ROWS, 1, (piece, sugar)),
                    "the outer share after a nested one, {piece} of {sugar}"
                );
                piece + sugar
            });
            assert_eq!(got, piece + sugar, "what the closure returns");
            whole(&budget, "after");
        }
    }
    let mut most = worked(3);
    most.bound(u64::MAX);
    most.spend_subtracted(1 << 62, Site::Row)
        .expect("an unbounded budget stops no call");
    most.over_sugar(1, usize::MAX, |budget| {
        assert!(
            !budget.projected_past(3, 1, usize::MAX),
            "a bound of u64::MAX"
        );
    });
}

/// Test 4: gf2-build-time.1 to .5 in calls. Over GF(2), in grevlex, the
/// boolean systems and the edge cases of `gf2_bound_losses_tests` and
/// cubic systems on 8 to 10 variables, with the signatures on, so that the
/// call starts in the bounded plain run: every call under each split of
/// `LOWERED` and each forced bound returns the basis the exact checker
/// certifies. Among the calls, one halts by the projection, its `work` at
/// most its `bound`.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn lowered_splits_and_bounds_over_gf2_leave_every_basis() {
    let mut texts = systems(true);
    texts.extend((8..=10usize).map(|n| {
        let m = 2 * n;
        (format!("cubics({n}, {m})"), cubics(n, m, 0x159e + n as u64))
    }));
    let mut projected = 0;
    for (i, (name, text)) in texts.iter().enumerate() {
        let seed = 0x159f ^ i as u64;
        let want = certified(name, text, seed);
        for split in LOWERED {
            for forced in bounds() {
                let case = format!("{name}, split {split:?}, forced {forced:?}");
                SPLIT_BOUNDS.set(Some(split));
                let (basis, lines) = call(text, &reported(seed, true), forced);
                SPLIT_BOUNDS.set(None);
                assert_eq!(basis, want, "{case}");
                let halts = halts(&case, &lines, forced);
                projected += halts.outworked.iter().filter(|(w, b)| w <= b).count();
            }
        }
    }
    assert!(projected > 0, "no halt by the projection");
}
