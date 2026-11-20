//! pair-update-rest-time, sub-item 2, the crate's test code: the pair
//! set's lcm words with the lcms interned at the selection that takes
//! them, and a split's piece by the words
//! (`iterations/0107-pair-update-rest-time/plan.md`, _Test code_, 3 and 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::budget::Budget;
use crate::divisibility_tests::{Engine, RefPair, Reference, degree, random_lanes, written};
use crate::id::MonoId;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::pairs::{self, SPLIT_BOUNDS};
use crate::shared_tests::{below, rng, select_reference};

/// Test 3 in one format and order: random leads entered one at a time,
/// a selection after about a quarter of the updates. After every update,
/// the set's words are the reference's lcms, every live pair's `lcm` is
/// `NONE`, and the table gained no id; after every selection, the pairs
/// taken carry their lcm's id, whose exponents are the words', the table
/// gained no id but those, and the pairs left are the reference's.
fn check_words_on<M: Format, O: Order<M>>(order: &O, seed: u64, variables: impl Fn(u64) -> usize) {
    let mut rng = rng(seed);
    for world in 0..40 {
        let n = variables(world);
        let mut engine: Engine<M> = Engine::new(n, seed ^ world);
        let mut reference = Reference::new(O::GRADED, M::MAX_DEGREE);
        let elements = 10 + below(&mut rng, 40) as usize;
        for e in 0..elements {
            let most = 1 + below(&mut rng, 5) as u32;
            let lead = random_lanes(&mut rng, n, 3, Some(most));
            let sugar = degree(&lead) + below(&mut rng, 4) as u32;
            let (got, gained) = engine.enter(order, &lead, sugar);
            got.unwrap();
            reference.update(lead.clone(), sugar).unwrap();
            let case = format!("world {world}, element {e}, lead {lead:?}");
            // The lead itself is interned by the test's `monomial`, before
            // the update; the update interns nothing.
            assert!(gained.is_empty(), "{case}: the update interned {gained:?}");
            assert_eq!(engine.written(), reference.pairs, "{case}: the set's words");
            assert!(
                engine.pairs.iter().all(|p| p.lcm == MonoId::NONE),
                "{case}: a live pair's lcm is interned"
            );
            if below(&mut rng, 4) == 0 {
                let before = engine.table.len();
                let got = pairs::select(
                    &mut engine.pairs,
                    &mut engine.table,
                    order,
                    &mut Budget::unbounded(),
                )
                .unwrap();
                let want = select_reference(&mut reference.pairs);
                assert_eq!(written(&engine.table, &got), want, "{case}: the selection");
                assert!(
                    got.iter().all(|p| p.lcm != MonoId::NONE),
                    "{case}: a selected pair's lcm is interned"
                );
                // The ids gained are among the selected lcms, each once.
                let mut ids: Vec<MonoId> = got.iter().map(|p| p.lcm).collect();
                ids.sort_unstable();
                ids.dedup();
                let gained = engine.table.len() - before;
                let count = ids.len();
                assert!(gained <= count, "{case}: {gained} ids for {count} lcms");
                assert!(
                    ids.iter().all(|&id| id.index() < engine.table.len()),
                    "{case}: an lcm's id past the table"
                );
                assert_eq!(engine.written(), reference.pairs, "{case}: the pairs left");
            }
        }
        // The B criterion may drop the last live pairs, which the set holds
        // until a selection (plain-zero-pairs-time.1).
        while !reference.pairs.is_empty() || engine.pairs.held() > 0 {
            let got = pairs::select(
                &mut engine.pairs,
                &mut engine.table,
                order,
                &mut Budget::unbounded(),
            )
            .unwrap();
            let want = select_reference(&mut reference.pairs);
            assert_eq!(
                written(&engine.table, &got),
                want,
                "world {world}: at the end"
            );
        }
        assert!(engine.pairs.is_empty(), "world {world}: the set is empty");
        assert_eq!(engine.pairs.held(), 0, "world {world}: nothing held");
    }
}

/// Test 3: pair-update-rest-time.1, in each format, grevlex and lex, on
/// 2 to 6 and on 33 to 72 variables.
#[test]
fn the_lcms_are_interned_at_the_selection() {
    let few = |world: u64| 2 + (world % 5) as usize;
    let many = |world: u64| 33 + (world % 40) as usize;
    check_words_on::<Dense8, _>(&Grevlex, 0xa107_0031, few);
    check_words_on::<Dense8, _>(&Lex, 0xa107_0032, many);
    check_words_on::<Dense16, _>(&Grevlex, 0xa107_0033, many);
    check_words_on::<Dense16, _>(&Lex, 0xa107_0034, few);
    check_words_on::<Dense32, _>(&Grevlex, 0xa107_0035, few);
    check_words_on::<Dense32, _>(&Lex, 0xa107_0036, many);
}

/// Test 4 in one order: under a split of (4, 2), leads of one degree
/// whose pairs share one sugar and number more than four, the selection
/// takes the two of least lcm in the order, ties by place, as the
/// reference's lcms compared in the order give them, and leaves the rest.
fn check_piece_on<O: Order<Dense16>>(order: &O, seed: u64) {
    let mut rng = rng(seed);
    let n = 4;
    for world in 0..40 {
        let mut engine: Engine<Dense16> = Engine::new(n, seed ^ world);
        let mut reference = Reference::new(O::GRADED, Dense16::MAX_DEGREE);
        // Leads of degree 2 on four variables: every pair's lcm has degree
        // 3 or 4 and the sugars agree within a degree.
        for _ in 0..(6 + below(&mut rng, 6) as usize) {
            let lead = random_lanes(&mut rng, n, 2, Some(2));
            let sugar = degree(&lead);
            engine.enter(order, &lead, sugar).0.unwrap();
            reference.update(lead.clone(), sugar).unwrap();
        }
        let live = reference.pairs.len();
        if live <= 4 {
            continue;
        }
        let least = reference.pairs.iter().map(|p| p.sugar).min().unwrap();
        let places: Vec<usize> = (0..live)
            .filter(|&k| reference.pairs[k].sugar == least)
            .collect();
        if places.len() <= 4 {
            continue;
        }
        // The reference's piece: the two least lcms in the order, ties by
        // place, through the order's comparison on packed words.
        let format = engine.table.format();
        let mut sorted = places.clone();
        sorted.sort_by(|&a, &b| {
            let (pa, pb) = (&reference.pairs[a], &reference.pairs[b]);
            let (mut wa, mut wb) = (vec![0u64; format.words()], vec![0u64; format.words()]);
            format.encode(&pa.lcm, &mut wa).unwrap();
            format.encode(&pb.lcm, &mut wb).unwrap();
            order
                .cmp(format, &wa, pa.degree, &wb, pb.degree)
                .then(a.cmp(&b))
        });
        let want: Vec<RefPair> = {
            let mut take = sorted[..2].to_vec();
            take.sort_unstable();
            take.iter().map(|&k| reference.pairs[k].clone()).collect()
        };
        SPLIT_BOUNDS.set(Some((4, 2)));
        let got = pairs::select(
            &mut engine.pairs,
            &mut engine.table,
            order,
            &mut Budget::unbounded(),
        )
        .unwrap();
        SPLIT_BOUNDS.set(None);
        let count = places.len();
        let case = format!("world {world}, {live} live pairs, {count} of sugar {least}");
        assert_eq!(written(&engine.table, &got), want, "{case}: the piece");
        assert_eq!(
            engine.pairs.deferred(),
            places.len() - 2,
            "{case}: the pairs deferred"
        );
        let left: Vec<RefPair> = reference
            .pairs
            .iter()
            .filter(|p| !want.contains(p))
            .cloned()
            .collect();
        assert_eq!(engine.written(), left, "{case}: the pairs left");
    }
}

/// Test 4: pair-update-rest-time.1, the piece by the words, in grevlex,
/// lex, and a block order with the first variable in the first block.
#[test]
fn a_split_takes_the_least_lcms_by_their_words() {
    check_piece_on(&Grevlex, 0xa107_0041);
    check_piece_on(&Lex, 0xa107_0042);
    check_piece_on(&Block::new(&[1], 4).unwrap(), 0xa107_0043);
}
