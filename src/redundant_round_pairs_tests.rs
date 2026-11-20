//! redundant-round-pairs, the crate's test code: the plain run's selection
//! of a sugar of many pairs, and calls whose split the test profile lowers
//! (`iterations/0095-redundant-round-pairs/plan.md`, _Test code_, 1 to 5).

use std::num::NonZeroUsize;

use crate::budget::Budget;
use crate::id::{ElementId, MonoId};
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::pairs::{self, Pair, PairSet, SPLIT_BOUNDS, SPLIT_LEAST, SPLIT_TAKE};
use crate::report_tests::lines;
use crate::shared_tests::{OFF, rng, systems};
use crate::table::Table;
use crate::{CheckerMode, MonomialOrder, Params, Threads};

/// The degree of the lcms of `Lcms`.
const LCM_DEGREE: u32 = 16_000;

/// The lcms of tests 1 to 3: the lcm of number k is x^k·y^(`LCM_DEGREE` −
/// k), which grows with k in grevlex, so that the first numbers in a set
/// are the least lcms (split-piece-time.1). The set holds each lcm's
/// words, one word in `Dense16` on two variables, and the selection
/// interns the lcms it takes (pair-update-rest-time.1).
struct Lcms {
    table: Table<Dense16>,
    /// Per number, its lcm's word.
    words: Vec<u64>,
}

impl Lcms {
    fn new() -> Lcms {
        let table = Table::new(Dense16::new(2), &mut rng(0x5e1b));
        let format = *table.format();
        assert_eq!(format.words(), 1, "two variables in one word");
        let words = (0..=LCM_DEGREE)
            .map(|k| {
                let mut word = [0u64; 1];
                format.encode(&[k, LCM_DEGREE - k], &mut word).unwrap();
                word[0]
            })
            .collect();
        Lcms { table, words }
    }

    /// A pair of sugar `sugar` and degree `LCM_DEGREE`, its elements and
    /// lcm numbered by `k`, so that each pair of a set is told from every
    /// other by `k`.
    fn pair(k: usize, sugar: u32) -> Pair {
        Pair {
            i: ElementId::at(k),
            j: ElementId::at(k + 1),
            lcm: MonoId::NONE,
            degree: LCM_DEGREE,
            sugar,
        }
    }

    /// Pushes `n` pairs of `sugar`, numbered from `from`, into `set`; the
    /// number after the last.
    fn push(&self, set: &mut PairSet<u64>, from: usize, n: usize, sugar: u32) -> usize {
        for k in from..from + n {
            set.push(Self::pair(k, sugar), 1, &self.words[k..=k], None);
        }
        from + n
    }

    /// The selection out of `set` in grevlex.
    fn select(&mut self, set: &mut PairSet<u64>) -> Vec<Pair> {
        pairs::select(set, &mut self.table, &Grevlex, &mut Budget::unbounded()).unwrap()
    }
}

/// The numbers of `pairs`, in their order.
fn numbers<'a>(pairs: impl IntoIterator<Item = &'a Pair>) -> Vec<u32> {
    pairs.into_iter().map(|p| p.i.raw()).collect()
}

/// The numbers from `from` below `to`.
fn span(from: usize, to: usize) -> Vec<u32> {
    (from..to).map(|k| u32::try_from(k).unwrap()).collect()
}

/// Test 1: redundant-round-pairs.1 to .3. A sugar of `SPLIT_LEAST` live
/// pairs is taken whole; one of `SPLIT_LEAST` + 1 is split: its first
/// `SPLIT_TAKE` pairs in the set's order are taken, and the set keeps the
/// rest of it and every pair of the other sugars, in their order.
/// The lcms grow with the place (`Lcms`).
#[test]
fn a_large_sugar_is_split() {
    let mut lcms = Lcms::new();
    for n in [SPLIT_LEAST, SPLIT_LEAST + 1] {
        let mut set = PairSet::new(1);
        let a = lcms.push(&mut set, 0, 3, 6);
        let b = lcms.push(&mut set, a, n, 5);
        let c = lcms.push(&mut set, b, 2, 7);
        let got = lcms.select(&mut set);
        let held = numbers(set.iter());
        if n == SPLIT_LEAST {
            assert_eq!(numbers(&got), span(a, b), "{n} pairs: taken whole");
            assert_eq!(set.deferred(), 0, "{n} pairs: none deferred");
            let mut want = span(0, a);
            want.extend(span(b, c));
            assert_eq!(held, want, "{n} pairs: the other sugars kept");
        } else {
            assert_eq!(
                numbers(&got),
                span(a, a + SPLIT_TAKE),
                "{n} pairs: the first taken"
            );
            assert_eq!(
                set.deferred(),
                n - SPLIT_TAKE,
                "{n} pairs: the rest deferred"
            );
            let mut want = span(0, a);
            want.extend(span(a + SPLIT_TAKE, b));
            want.extend(span(b, c));
            assert_eq!(held, want, "{n} pairs: the rest kept in order");
        }
        assert_eq!(set.len(), held.len(), "{n} pairs: the count");
        assert!(
            got.iter().all(|p| p.sugar == 5),
            "{n} pairs: the least sugar"
        );
    }
}

/// Test 2: redundant-round-pairs.4. The selection after a split takes its
/// least sugar whole, at the same sugar past `SPLIT_LEAST` and at a lower
/// one; the selection after a whole one splits again; an empty set
/// selects nothing.
/// The lcms grow with the place (`Lcms`).
#[test]
fn the_selection_after_a_split_takes_all() {
    let mut lcms = Lcms::new();
    let mut set = PairSet::new(1);
    let a = lcms.push(&mut set, 0, 3 * SPLIT_LEAST, 5);
    let got = lcms.select(&mut set);
    assert_eq!(numbers(&got), span(0, SPLIT_TAKE), "the split");
    assert_eq!(set.deferred(), 3 * SPLIT_LEAST - SPLIT_TAKE);
    let got = lcms.select(&mut set);
    assert_eq!(numbers(&got), span(SPLIT_TAKE, a), "the rest whole");
    assert_eq!(set.deferred(), 0);
    assert!(set.is_empty());
    let b = lcms.push(&mut set, a, SPLIT_LEAST + 1, 4);
    let got = lcms.select(&mut set);
    assert_eq!(numbers(&got), span(a, a + SPLIT_TAKE), "a split again");
    assert_eq!(set.deferred(), SPLIT_LEAST + 1 - SPLIT_TAKE);
    let c = lcms.push(&mut set, b, SPLIT_LEAST + 1, 3);
    let got = lcms.select(&mut set);
    assert_eq!(
        numbers(&got),
        span(b, c),
        "a lower sugar whole after a split"
    );
    assert_eq!(set.deferred(), 0);
    let got = lcms.select(&mut set);
    assert_eq!(
        numbers(&got),
        span(a + SPLIT_TAKE, b),
        "the rest of sugar 4"
    );
    assert!(lcms.select(&mut set).is_empty(), "an empty set");
    assert_eq!(set.deferred(), 0);
}

/// Test 3: redundant-round-pairs.1 and .5. Pairs the B criterion dropped
/// count for no split and are not taken: `SPLIT_LEAST` + 1 pairs with one
/// dropped are taken whole; `SPLIT_LEAST` + 3 with two dropped are split,
/// the first `SPLIT_TAKE` live ones taken in order; after a selection the
/// set holds at most as many holes as live pairs (pair-index-upkeep-time.3).
/// The lcms grow with the place (`Lcms`).
#[test]
fn dropped_pairs_are_neither_counted_nor_taken() {
    let mut lcms = Lcms::new();
    let mut set = PairSet::new(1);
    let a = lcms.push(&mut set, 0, SPLIT_LEAST + 1, 5);
    set.drop_at(0);
    let got = lcms.select(&mut set);
    assert_eq!(numbers(&got), span(1, a), "one dropped: taken whole");
    assert_eq!(set.deferred(), 0);
    let mut set = PairSet::new(1);
    let a = lcms.push(&mut set, 0, SPLIT_LEAST + 3, 5);
    set.drop_at(2);
    set.drop_at(4);
    let got = lcms.select(&mut set);
    let live: Vec<u32> = span(0, a)
        .into_iter()
        .filter(|&k| k != 2 && k != 4)
        .collect();
    assert_eq!(
        numbers(&got),
        live[..SPLIT_TAKE],
        "two dropped: the first live taken"
    );
    assert_eq!(set.deferred(), SPLIT_LEAST + 1 - SPLIT_TAKE);
    assert_eq!(numbers(set.iter()), live[SPLIT_TAKE..], "the rest kept");
    assert!(
        set.held() - set.len() <= set.len(),
        "at most as many holes as live pairs"
    );
}

/// The parameters of a call in `order` on one thread under `seed`, with
/// its report and the signatures as `signatures` says.
fn reported(order: MonomialOrder, seed: u64, signatures: bool) -> Params {
    let mut params = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), seed);
    params.report = true;
    params.signatures = signatures;
    params
}

/// The basis text of the call of `text` and the `deferred` counts of its
/// report's round lines, under the split `bounds`; None where an error
/// ended the call.
fn call(
    text: &str,
    params: &Params,
    bounds: Option<(usize, usize)>,
) -> Option<(String, Vec<usize>)> {
    SPLIT_BOUNDS.set(bounds);
    let outcome = crate::groebner(text, params);
    SPLIT_BOUNDS.set(None);
    let outcome = outcome.ok()?;
    let report = outcome.report.unwrap();
    let deferred = lines(&report)
        .into_iter()
        .filter(|l| l.kind == "round")
        .filter_map(|l| l.get("deferred").map(|v| v.parse().unwrap()))
        .collect();
    Some((outcome.basis.unwrap(), deferred))
}

/// Test 4: redundant-round-pairs.1, .4, and .6 with the split lowered:
/// over 7, 65521, 2^31 − 1, and ℚ, in grevlex and `block:1`, the plain run
/// from the start, and with cofactors and the exact checker over 65521,
/// every call under a lowered split returns the basis of the call with the
/// split off, and some call's round leaves pairs.
#[test]
fn lowered_splits_leave_every_basis() {
    let mut deferred = 0;
    for p in [7, 65521, 2_147_483_647, 0] {
        for order in [MonomialOrder::Grevlex, MonomialOrder::Block(vec![1])] {
            for (i, text) in systems(p, 12, 4).iter().enumerate() {
                let params = reported(order.clone(), 0x5e1c ^ i as u64, false);
                let Some((want, _)) = call(text, &params, OFF) else {
                    continue;
                };
                for bounds in [Some((1, 1)), Some((2, 1))] {
                    let (basis, counts) = call(text, &params, bounds).unwrap();
                    assert_eq!(
                        basis, want,
                        "p {p}, {order:?}, system {i}, {bounds:?}\n{text}"
                    );
                    deferred += counts.iter().filter(|&&n| n > 0).count();
                }
            }
        }
    }
    assert!(deferred > 0, "no round of a lowered split left pairs");
    for (i, text) in systems(65521, 12, 4).iter().enumerate() {
        let mut params = reported(MonomialOrder::Grevlex, 0x5e1d ^ i as u64, false);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        let Some((want, _)) = call(text, &params, OFF) else {
            continue;
        };
        let (basis, _) = call(text, &params, Some((1, 1))).unwrap();
        assert_eq!(basis, want, "cofactors, system {i}\n{text}");
    }
}

/// Test 5: redundant-round-pairs.7. Every round line of a plain run names
/// its deferred pairs right before `left`, those of a signature run and of
/// a replay none; with the split lowered some line's count lies above 0,
/// and under the production bounds every one is 0 on these systems.
#[test]
fn a_round_names_its_deferred_pairs() {
    let mut above = false;
    for p in [65521, 0] {
        for (i, text) in systems(p, 12, 4).iter().enumerate() {
            for signatures in [true, false] {
                let params = reported(MonomialOrder::Grevlex, 0x5e1e ^ i as u64, signatures);
                for bounds in [Some((1, 1)), None] {
                    SPLIT_BOUNDS.set(bounds);
                    let outcome = crate::groebner(text, &params);
                    SPLIT_BOUNDS.set(None);
                    let Ok(outcome) = outcome else {
                        continue;
                    };
                    let mut engine = String::new();
                    for line in lines(&outcome.report.unwrap()) {
                        if line.kind == "run" {
                            engine = line.get("engine").unwrap().to_string();
                        }
                        if line.kind != "round" {
                            continue;
                        }
                        let keys: Vec<&str> = line.fields.iter().map(|(k, _)| k.as_str()).collect();
                        let at = keys.iter().position(|&k| k == "deferred");
                        if engine != "plain" {
                            assert_eq!(at, None, "p {p}, system {i}, {engine}: {line:?}");
                            continue;
                        }
                        let at = at.unwrap_or_else(|| panic!("p {p}, system {i}: {line:?}"));
                        assert_eq!(keys[at + 1], "left", "{line:?}");
                        let n: usize = line.get("deferred").unwrap().parse().unwrap();
                        if bounds.is_none() {
                            assert_eq!(n, 0, "p {p}, system {i}: {line:?}");
                        }
                        above |= n > 0;
                    }
                }
            }
        }
    }
    assert!(
        above,
        "no round line of a lowered split held deferred pairs"
    );
}
