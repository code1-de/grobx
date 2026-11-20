//! split-piece-time, the crate's test code: the piece a split takes, the
//! pairs of least lcm in the run's order, a sugar split once, and calls
//! whose split the test profile lowers
//! (`iterations/0096-split-piece-time/plan.md`, _Test code_, 1 to 7).

use std::cmp::Ordering;
use std::num::NonZeroUsize;

use rand_core::Rng;

use crate::budget::Budget;
use crate::id::{ElementId, MonoId};
use crate::monomial::{Dense8, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::pairs::{self, Pair, PairSet, SPLIT_BOUNDS, SPLIT_LEAST, SPLIT_TAKE};
use crate::report_tests::lines;
use crate::shared_tests::{OFF, rng};
use crate::systems::{dense, random_square, random_system};
use crate::table::Table;
use crate::{CheckerMode, MonomialOrder, Outcome, Params, Threads};

/// The variables of the tests' lcms.
const N: usize = 4;

/// The degree of the split sugar's lcms: its monomials on `N` variables
/// number 2,925, more than `SPLIT_LEAST` + 4.
pub(crate) const D: u32 = 24;

/// A monomial order on exponent vectors, written out in the tests.
type Cmp = fn(&[u32], &[u32]) -> Ordering;

/// Every exponent vector of total degree `d` on `N` variables.
pub(crate) fn monomials(d: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for a in 0..=d {
        for b in 0..=d - a {
            for c in 0..=d - a - b {
                out.push(vec![a, b, c, d - a - b - c]);
            }
        }
    }
    out
}

/// `v` in an order drawn from `seed`.
fn shuffled<T>(mut v: Vec<T>, seed: u64) -> Vec<T> {
    let mut r = rng(seed);
    for k in (1..v.len()).rev() {
        let j = usize::try_from(r.next_u64() % (k as u64 + 1)).unwrap();
        v.swap(k, j);
    }
    v
}

/// Grevlex on exponent vectors, written out apart from the crate's: the
/// larger degree, then the smaller exponent at the last variable where
/// they differ.
fn grevlex(a: &[u32], b: &[u32]) -> Ordering {
    let (da, db): (u32, u32) = (a.iter().sum(), b.iter().sum());
    da.cmp(&db).then_with(|| {
        (0..a.len())
            .rev()
            .find(|&i| a[i] != b[i])
            .map_or(Ordering::Equal, |i| b[i].cmp(&a[i]))
    })
}

/// Lex on exponent vectors: the larger exponent at the first variable
/// where they differ.
fn lex(a: &[u32], b: &[u32]) -> Ordering {
    a.cmp(b)
}

/// `block:2` on `N` variables: grevlex on the first two, then on the rest.
fn block2(a: &[u32], b: &[u32]) -> Ordering {
    grevlex(&a[..2], &b[..2]).then_with(|| grevlex(&a[2..], &b[2..]))
}

/// A set of pairs over one table: each pair's lcm stands in the set's
/// words, interned by the selection that takes it
/// (pair-update-rest-time.1), and the pair's `i` is its place in the set,
/// by which the tests tell it apart.
pub(crate) struct World {
    table: Table<Dense8>,
    pub(crate) set: PairSet<u64>,
    /// Per place, the lcm's exponents.
    lcms: Vec<Vec<u32>>,
    /// Per place, the pair's sugar.
    sugars: Vec<u32>,
}

impl World {
    /// An empty set over a table where `interned` was interned in its
    /// order.
    pub(crate) fn new(interned: &[Vec<u32>], seed: u64) -> World {
        let mut table = Table::new(Dense8::new(N), &mut rng(seed));
        for e in interned {
            table.intern(e, &mut Budget::unbounded()).unwrap();
        }
        World {
            set: PairSet::new(table.format().words()),
            table,
            lcms: Vec::new(),
            sugars: Vec::new(),
        }
    }

    /// Pushes a pair of sugar `sugar` with the lcm `lcm`; its place.
    fn push(&mut self, lcm: &[u32], sugar: u32) -> usize {
        let k = self.lcms.len();
        let format = self.table.format();
        let mut words = vec![0u64; format.words()];
        format.encode(lcm, &mut words).unwrap();
        let pair = Pair {
            i: ElementId::at(k),
            j: ElementId::at(k + 1),
            lcm: MonoId::NONE,
            degree: lcm.iter().sum(),
            sugar,
        };
        self.set.push(pair, 1, &words, None);
        self.lcms.push(lcm.to_vec());
        self.sugars.push(sugar);
        k
    }

    /// Pushes `n` pairs of sugar `sugar`, their lcms the first `n` of
    /// `lcms` taken in turn; their places.
    pub(crate) fn push_all(&mut self, lcms: &[Vec<u32>], n: usize, sugar: u32) -> Vec<usize> {
        (0..n)
            .map(|k| self.push(&lcms[k % lcms.len()], sugar))
            .collect()
    }

    /// The selection in `order`; the places of the pairs it took, in its
    /// order.
    fn select<O: Order<Dense8>>(&mut self, order: &O) -> Vec<usize> {
        places(
            &pairs::select(
                &mut self.set,
                &mut self.table,
                order,
                &mut Budget::unbounded(),
            )
            .unwrap(),
        )
    }

    /// The places of the pairs the set holds, in its order.
    fn held(&self) -> Vec<usize> {
        places(self.set.iter())
    }

    /// Of `among`, the `take` places of least lcm by `cmp`, ties by place,
    /// in the set's order.
    fn least(
        &self,
        among: &[usize],
        take: usize,
        cmp: fn(&[u32], &[u32]) -> Ordering,
    ) -> Vec<usize> {
        let mut by = among.to_vec();
        by.sort_by(|&a, &b| cmp(&self.lcms[a], &self.lcms[b]).then(a.cmp(&b)));
        by.truncate(take);
        by.sort_unstable();
        by
    }
}

/// The places of `pairs`, in their order.
fn places<'a>(pairs: impl IntoIterator<Item = &'a Pair>) -> Vec<usize> {
    pairs.into_iter().map(|p| p.i.index()).collect()
}

/// `a` without the places of `b`, in its order.
fn without(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter().copied().filter(|k| !b.contains(k)).collect()
}

/// Test 1: split-piece-time.1 and .3. `SPLIT_LEAST` + 1 pairs of sugar
/// `D` whose distinct lcms stand in the set in one order and were interned
/// in another, between pairs of sugar `D` + 1: the selection takes the
/// `SPLIT_TAKE` pairs of least lcm in grevlex, in the set's order, and
/// the set keeps every other pair in its order. `SPLIT_LEAST` pairs are
/// taken whole.
#[test]
fn the_piece_is_the_least_lcms() {
    for n in [SPLIT_LEAST, SPLIT_LEAST + 1] {
        let lcms = shuffled(monomials(D), 0x5b1 ^ n as u64);
        let mut w = World::new(&shuffled(monomials(D), 0x5b2 ^ n as u64), 0x5b3);
        let higher = monomials(D + 1);
        let before = w.push_all(&higher, 3, D + 1);
        let split = w.push_all(&lcms, n, D);
        let after = w.push_all(&higher[3..], 2, D + 1);
        let got = w.select(&Grevlex);
        let held = w.held();
        if n == SPLIT_LEAST {
            assert_eq!(got, split, "{n} pairs: taken whole");
            assert_eq!(w.set.deferred(), 0, "{n} pairs: none deferred");
            assert_eq!(held, [before, after].concat(), "{n} pairs: the rest kept");
        } else {
            let want = w.least(&split, SPLIT_TAKE, grevlex);
            assert_eq!(got, want, "{n} pairs: the least lcms");
            assert_eq!(
                w.set.deferred(),
                n - SPLIT_TAKE,
                "{n} pairs: the rest deferred"
            );
            let rest = without(&split, &want);
            assert_eq!(
                held,
                [before, rest, after].concat(),
                "{n} pairs: the rest kept in order"
            );
            assert_ne!(
                got[..],
                split[..SPLIT_TAKE],
                "{n} pairs: not the first places"
            );
        }
        assert_eq!(w.set.len(), held.len(), "{n} pairs: the count");
    }
}

/// Test 2: split-piece-time.1 and .3. `SPLIT_LEAST` + 1 pairs whose lcms
/// take seven values, so that the piece's bound cuts a run of equal lcms:
/// the piece holds every pair of a smaller lcm and, of the lcm at the
/// bound, the earliest places.
#[test]
fn equal_lcms_go_by_place() {
    let values: Vec<Vec<u32>> = shuffled(monomials(D), 0x5b4).into_iter().take(7).collect();
    let mut w = World::new(&values, 0x5b5);
    let lcms: Vec<Vec<u32>> = (0..=SPLIT_LEAST)
        .map(|k| values[(k * 5 + k / 3) % 7].clone())
        .collect();
    let split = w.push_all(&lcms, SPLIT_LEAST + 1, D);
    let got = w.select(&Grevlex);
    let want = w.least(&split, SPLIT_TAKE, grevlex);
    assert_eq!(got, want, "the least lcms, ties by place");
    let bound = &w.lcms[*want
        .iter()
        .max_by(|&&a, &&b| grevlex(&w.lcms[a], &w.lcms[b]).then(a.cmp(&b)))
        .unwrap()];
    let at_bound: Vec<usize> = split
        .iter()
        .copied()
        .filter(|&k| &w.lcms[k] == bound)
        .collect();
    let taken: Vec<usize> = at_bound
        .iter()
        .copied()
        .filter(|k| got.contains(k))
        .collect();
    let (cut, run) = (taken.len(), at_bound.len());
    assert!(
        cut > 0 && cut < run,
        "the bound cuts a run of equal lcms: {cut} of {run}"
    );
    assert_eq!(
        taken,
        at_bound[..taken.len()],
        "the earliest places of the lcm at the bound"
    );
}

/// Test 3: split-piece-time.1. The lcms of test 1 in lex and in `block:2`:
/// each piece is the least lcms by that order, and grevlex's and lex's
/// pieces differ.
#[test]
fn the_order_decides() {
    let lcms = shuffled(monomials(D), 0x5b6);
    let mut pieces = Vec::new();
    for which in 0..3 {
        let mut w = World::new(&shuffled(monomials(D), 0x5b7), 0x5b8);
        let split = w.push_all(&lcms, SPLIT_LEAST + 1, D);
        let (got, cmp): (Vec<usize>, Cmp) = match which {
            0 => (w.select(&Grevlex), grevlex),
            1 => (w.select(&Lex), lex),
            _ => (w.select(&Block::new(&[2], N).unwrap()), block2),
        };
        assert_eq!(got, w.least(&split, SPLIT_TAKE, cmp), "order {which}");
        pieces.push(got);
    }
    assert_ne!(pieces[0], pieces[1], "grevlex's piece and lex's");
}

/// Test 4: split-piece-time.1. Of `SPLIT_LEAST` + 4 pairs, three whose
/// lcms are among the least are dropped: the piece is the `SPLIT_TAKE`
/// least live lcms, the set holds at most as many holes as live pairs
/// after it (pair-index-upkeep-time.3), and `deferred` counts the live
/// pairs left. `SPLIT_LEAST` + 1 pairs with one dropped
/// are taken whole.
#[test]
fn dropped_pairs_are_not_taken() {
    let lcms = shuffled(monomials(D), 0x5b9);
    let mut w = World::new(&[], 0x5ba);
    let split = w.push_all(&lcms, SPLIT_LEAST + 4, D);
    let least = w.least(&split, 3, grevlex);
    for &k in &least {
        w.set.drop_at(k);
    }
    let live = without(&split, &least);
    let got = w.select(&Grevlex);
    let want = w.least(&live, SPLIT_TAKE, grevlex);
    assert_eq!(got, want, "the least live lcms");
    assert_eq!(
        w.set.deferred(),
        SPLIT_LEAST + 1 - SPLIT_TAKE,
        "the live pairs left"
    );
    assert_eq!(w.held(), without(&live, &want), "the rest kept in order");
    assert!(
        w.set.held() - w.set.len() <= w.set.len(),
        "at most as many holes as live pairs"
    );

    let mut w = World::new(&[], 0x5bb);
    let split = w.push_all(&lcms, SPLIT_LEAST + 1, D);
    w.set.drop_at(split[7]);
    let got = w.select(&Grevlex);
    assert_eq!(
        got,
        without(&split, &[split[7]]),
        "one dropped: taken whole"
    );
    assert_eq!(w.set.deferred(), 0);
}

/// One selection of test 5: its sugar, the pairs it took, and `deferred`.
pub(crate) fn step(w: &mut World, case: &str, sugar: u32, taken: usize, deferred: usize) {
    let got = w.select(&Grevlex);
    assert_eq!(got.len(), taken, "{case}: the pairs taken");
    assert!(
        got.iter().all(|&k| w.sugars[k] == sugar),
        "{case}: the sugar {sugar}"
    );
    assert_eq!(w.set.deferred(), deferred, "{case}: deferred");
}

/// Test 5: split-piece-time.2 and redundant-round-pairs.4. A sugar splits
/// once: after a split, a lower sugar is taken whole right after it, and
/// the split sugar's rest whole past `SPLIT_LEAST`; pairs of the split
/// sugar pushed later are taken whole until a higher sugar is split, and
/// then split again. `clear`, and the selection of an empty set, forget a
/// split.
#[test]
fn a_sugar_splits_once() {
    let lcms = monomials(D);
    let (l, t) = (SPLIT_LEAST, SPLIT_TAKE);
    let mut w = World::new(&[], 0x5bc);
    w.push_all(&lcms, 3 * l, 5);
    step(&mut w, "a split of 5", 5, t, 3 * l - t);
    w.push_all(&lcms, l + 1, 4);
    step(&mut w, "4 right after the split", 4, l + 1, 0);
    step(&mut w, "the rest of 5", 5, 3 * l - t, 0);
    assert!(w.set.is_empty());
    w.push_all(&lcms, l + 1, 5);
    w.push_all(&lcms, l + 1, 6);
    step(&mut w, "5 split once", 5, l + 1, 0);
    step(&mut w, "a split of 6", 6, t, l + 1 - t);
    step(&mut w, "the rest of 6", 6, l + 1 - t, 0);
    w.push_all(&lcms, l + 1, 5);
    step(&mut w, "5 after a split of 6", 5, t, l + 1 - t);
    w.set.clear();
    w.push_all(&lcms, l + 1, 5);
    step(&mut w, "5 after clear", 5, t, l + 1 - t);
    step(&mut w, "the rest of 5 after clear", 5, l + 1 - t, 0);
    assert!(w.select(&Grevlex).is_empty(), "an empty set");
    assert_eq!(w.set.deferred(), 0);
    w.push_all(&lcms, l + 1, 5);
    step(&mut w, "5 after an empty set", 5, t, l + 1 - t);
}

/// The parameters of a call in `order` on one thread under `seed`, its
/// report on, the signatures off.
pub(crate) fn reported(order: MonomialOrder, seed: u64) -> Params {
    let mut params = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), seed);
    params.report = true;
    params.signatures = false;
    params
}

/// The outcome of the call of `text` under the split `bounds` on this
/// thread, and the bounds back to None after; None where an error ended
/// the call.
pub(crate) fn outcome(
    text: &str,
    params: &Params,
    bounds: Option<(usize, usize)>,
) -> Option<Outcome> {
    SPLIT_BOUNDS.set(bounds);
    let outcome = crate::groebner(text, params);
    SPLIT_BOUNDS.set(None);
    outcome.ok()
}

/// The basis text and the report of the call of `text` under the split
/// `bounds`; None where an error ended the call.
fn call(text: &str, params: &Params, bounds: Option<(usize, usize)>) -> Option<(String, String)> {
    let outcome = outcome(text, params, bounds)?;
    Some((outcome.basis.unwrap(), outcome.report.unwrap()))
}

/// The edge cases of `OBJECTIVES.md` on x, y, z: their generators.
pub(crate) const EDGE: [&str; 7] = [
    "0",
    "0,\n0",
    "",
    "x*z-1,\nx",
    "x+z,\nz^2+1",
    "x*y-z,\n0,\ny^2-x",
    "x-y^2-z,\ny*z-1,\nz^2-y",
];

/// The systems of tests 6 and 7 over `p`: random systems and random
/// square systems, seeds 0 to 11, dense cubics on three variables, seeds 0
/// to 3, and the edge cases.
fn systems(p: u64) -> Vec<String> {
    let mut texts: Vec<String> = (0..12)
        .flat_map(|seed| [random_system(p, seed), random_square(p, seed)])
        .collect();
    texts.extend((0..4).map(|seed| dense(3, 3, p, seed)));
    texts.extend(EDGE.iter().map(|gens| format!("x,y,z\n{p}\n{gens}\n")));
    texts
}

/// Test 6: split-piece-time.1 and .2 with the split lowered: over 7,
/// 65521, 2^31 − 1, and ℚ, in grevlex, lex, and `block:1`, the plain run
/// from the start, and with cofactors and the exact checker over 65521,
/// every call under a lowered split returns the basis of the call with
/// the split off.
#[test]
fn lowered_splits_leave_every_basis() {
    let lowered = [Some((1, 1)), Some((2, 1)), Some((3, 2))];
    for p in [7, 65521, 2_147_483_647, 0] {
        for order in [
            MonomialOrder::Grevlex,
            MonomialOrder::Lex,
            MonomialOrder::Block(vec![1]),
        ] {
            for (i, text) in systems(p).iter().enumerate() {
                let params = reported(order.clone(), 0x5bd ^ i as u64);
                let Some((want, _)) = call(text, &params, OFF) else {
                    continue;
                };
                for bounds in lowered {
                    let (basis, _) = call(text, &params, bounds).unwrap();
                    assert_eq!(
                        basis, want,
                        "p {p}, {order:?}, system {i}, {bounds:?}\n{text}"
                    );
                }
            }
        }
    }
    for (i, text) in systems(65521).iter().enumerate() {
        let mut params = reported(MonomialOrder::Grevlex, 0x5be ^ i as u64);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        let Some((want, _)) = call(text, &params, OFF) else {
            continue;
        };
        for bounds in lowered {
            let (basis, _) = call(text, &params, bounds).unwrap();
            assert_eq!(basis, want, "cofactors, system {i}, {bounds:?}\n{text}");
        }
    }
}

/// Test 7: split-piece-time.2 in a call. Over 65521 and ℚ in grevlex,
/// under Some((1, 1)): in each plain run of a call's report, no round of
/// the last split's degree holds deferred pairs before a round of a degree
/// above it, and some run's round holds deferred pairs.
#[test]
fn a_sugar_splits_once_in_a_call() {
    let mut splits = 0;
    for p in [65521, 0] {
        for (i, text) in systems(p).iter().enumerate() {
            let params = reported(MonomialOrder::Grevlex, 0x5bf ^ i as u64);
            let Some((_, report)) = call(text, &params, Some((1, 1))) else {
                continue;
            };
            let mut plain = false;
            let mut last: Option<u32> = None;
            for line in lines(&report) {
                if line.kind == "run" {
                    plain = line.get("engine") == Some("plain");
                    last = None;
                }
                if line.kind != "round" || !plain {
                    continue;
                }
                let degree: u32 = line.get("degree").unwrap().parse().unwrap();
                let deferred: usize = line.get("deferred").unwrap().parse().unwrap();
                if last.is_some_and(|s| degree > s) {
                    last = None;
                }
                if deferred > 0 {
                    assert_ne!(
                        last,
                        Some(degree),
                        "p {p}, system {i}: degree {degree} split again\n{report}"
                    );
                    last = Some(degree);
                    splits += 1;
                }
            }
        }
    }
    assert!(
        splits > 0,
        "no round of a lowered split held deferred pairs"
    );
}
