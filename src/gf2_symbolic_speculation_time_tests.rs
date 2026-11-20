//! gf2-symbolic-speculation-time, the item's test code: every lookup of the
//! table probes through `probe`, its window and its probe loop, and agrees
//! with interning whatever the hashes, across the slot array's doublings and
//! a compaction, on formats of one to eight words and past eight, with the
//! window and without it; a call returns what it returns without the
//! window, and the exact checker certifies it; and `probe` on slot arrays
//! written out answers as a walk of their slots
//! (`iterations/0178-gf2-symbolic-speculation-time/plan.md`, _Test code_,
//! 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::build_tests;
use crate::call::built;
use crate::gf2_update_rest_time_tests::{spread, systems_on};
use crate::id::MonoId;
use crate::instance_read_time_tests::checked;
use crate::linear_substitution_time_tests::untimed;
use crate::lookup_tests;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::{HASHES, below, multipliers, rng, table};
use crate::split_piece_time_tests::EDGE;
use crate::systems::placed;
use crate::table::{COPIED_WORDS, EMPTY_SLOT, WINDOW, WINDOW_OFF, probe, slot_of, tag_of};
use crate::wide_format_products_tests::survives_a_compaction;
use crate::{MonomialOrder, Outcome, Params};

/// The hashes test 1 takes beside `HASHES`: tags that agree in their low 31
/// bits with bit 31 apart, and home slots among the slot array's last.
const WINDOW_HASHES: [&str; 2] = ["tags that agree in 31 bits", "homes near the end"];

/// `f` with `WINDOW_OFF` at `off`, and the window back on after it.
fn windowed<T>(off: bool, f: impl FnOnce() -> T) -> T {
    WINDOW_OFF.set(off);
    let got = f();
    WINDOW_OFF.set(false);
    got
}

/// Test 1 on tables of format `M` on each count of variables under each
/// hash of `HASHES` and `WINDOW_HASHES`: the lookups through both doors and
/// by interning (`lookup_tests::agree`), a compaction
/// (`wide_format_products_tests::survives_a_compaction`), and on formats of
/// `COPIED_WORDS` words or fewer, whose rows the copies look up, a row's
/// products against the lookup per term on a twin table
/// (`build_tests::agree`).
fn lookups_agree_in<M: Format>(counts: &[usize], seed: u64) {
    for &n in counts {
        for name in HASHES.iter().chain(&WINDOW_HASHES) {
            let case = format!(
                "{words} words on {n} variables, {name}, window off {off}",
                words = M::new(n).words(),
                off = WINDOW_OFF.get()
            );
            let t = table::<M>(n, multipliers(name, n, seed));
            lookup_tests::agree::<M>(n, t, seed ^ n as u64, &case);
            survives_a_compaction::<M>(n, name, seed);
            if M::new(n).words() <= COPIED_WORDS {
                for narrow in [false, true] {
                    build_tests::agree::<M>(n, name, seed ^ n as u64, narrow, false);
                }
            }
        }
    }
}

/// Test 1, gf2-symbolic-speculation-time.1 to .6, .8, .9: the lookups of
/// the copies of one to eight words and the lookups per term past eight,
/// the interns, and a compaction's placement agree with interning, one id a
/// monomial, under drawn hashes, tags that all agree in 32 bits and in their
/// low 31 alone, one probe chain, chains that wrap, and home slots among the
/// last, across the slot array's doublings, with the window and without it.
#[test]
fn lookups_agree_with_interning_with_the_window_and_without() {
    for off in [false, true] {
        windowed(off, || {
            // Dense4: 16 lanes a word, 1, 2, and 9 words.
            lookups_agree_in::<Dense4>(&[16, 17, 129], 0x5e41);
            // Dense8: 8 lanes a word, 1, 2, 5, 8, and 9 words.
            lookups_agree_in::<Dense8>(&[1, 9, 40, 64, 65], 0x5e42);
            // Dense16 and Dense32: 4 and 2 lanes a word, 1 and 2 words.
            lookups_agree_in::<Dense16>(&[4, 5], 0x5e43);
            lookups_agree_in::<Dense32>(&[2, 3], 0x5e44);
        });
    }
}

/// The call of `text` under `params` with `WINDOW_OFF` at `off`.
fn call_windowed(text: &str, params: &Params, off: bool) -> Outcome {
    windowed(off, || {
        crate::groebner(text, params).unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
    })
}

/// Test 2, gf2-symbolic-speculation-time.1 to .7: the edge cases of
/// `OBJECTIVES.md` over GF(2), GF(7), GF(65521), and ℚ placed on 3, 20,
/// 70, and 300 variables, and gf2-update-rest-time's systems on 20 and 129
/// (`gf2_update_rest_time_tests::systems_on`), so that the copies of one to
/// eight words and the lookups per term all probe: the exact checker
/// certifies each call in grevlex, and in grevlex and lex, the signatures
/// off and on, a call returns the basis and the report, but its times, that
/// it returns without the window.
#[test]
fn calls_return_what_they_return_without_the_window() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for p in [2u64, 7, 65521, 0] {
        for n in [3, 20, 70, 300] {
            for (k, g) in EDGE.iter().enumerate() {
                let text = placed(&format!("x,y,z\n{p}\n{g}\n"), &spread(3, n), n);
                texts.push((format!("edge {k} over {p} on {n}"), text));
            }
        }
    }
    for n in [20, 129] {
        texts.extend(systems_on(n));
    }
    for (name, text) in &texts {
        assert_eq!(checked(text).certified, Some(true), "{name}: certified");
        for order in [MonomialOrder::Grevlex, MonomialOrder::Lex] {
            for signatures in [false, true] {
                let mut params = reported(order.clone());
                params.signatures = signatures;
                if built(&params).is_err() {
                    continue;
                }
                let case = format!("{name}, {order:?}, signatures {signatures}");
                let looped = call_windowed(text, &params, true);
                let probed = call_windowed(text, &params, false);
                assert_eq!(probed.basis, looped.basis, "{case}: the basis");
                assert_eq!(untimed(&probed), untimed(&looped), "{case}: the report");
            }
        }
    }
}

/// What a slot of test 3's arrays holds: no id, the id sought under the
/// lookup's tag, another id under that tag, or an id under another tag.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Empty,
    Sought,
    Rejected,
    Other,
}

/// The id test 3 seeks.
const SOUGHT: usize = 1;

/// The answer a walk of the slots `kinds` gives from `home` on: the id
/// sought at the first slot that holds it, or the first empty slot, each
/// slot after the last the first.
fn walk(kinds: &[Kind], home: usize) -> Result<MonoId, usize> {
    let mut slot = home;
    loop {
        match kinds[slot] {
            Kind::Empty => return Err(slot),
            Kind::Sought => return Ok(MonoId::at(SOUGHT)),
            Kind::Rejected | Kind::Other => slot = (slot + 1) % kinds.len(),
        }
    }
}

/// Whether linear probing could have written `kinds` for a lookup from
/// `home`: an empty slot somewhere, the id sought once at most, and never
/// past an empty slot from the home slot on.
fn placeable(kinds: &[Kind], home: usize) -> bool {
    let (mut sought, mut empty, mut past_empty) = (0, false, false);
    for d in 0..kinds.len() {
        match kinds[(home + d) % kinds.len()] {
            Kind::Empty => empty = true,
            Kind::Sought => {
                sought += 1;
                past_empty |= empty;
            }
            Kind::Rejected | Kind::Other => {}
        }
    }
    empty && sought <= 1 && !past_empty
}

/// `probe` on the slots `kinds` under the hashes `(hash, other)`, the
/// lookup's and the other ids', from every home slot linear probing could
/// have written them for, with the window and without it: the answer is the
/// walk's, and `same` is asked only of ids under the lookup's tag, never of
/// the empty slot's.
fn probes_as_the_walk(kinds: &[Kind], (hash, other): (u64, u64), case: &str) {
    let slots: Vec<u64> = kinds
        .iter()
        .enumerate()
        .map(|(k, kind)| match kind {
            Kind::Empty => EMPTY_SLOT,
            Kind::Sought => slot_of(hash, MonoId::at(SOUGHT)),
            Kind::Rejected => slot_of(hash, MonoId::at(10 + k)),
            Kind::Other => slot_of(other, MonoId::at(100 + k)),
        })
        .collect();
    for home in (0..kinds.len()).filter(|&h| placeable(kinds, h)) {
        for off in [false, true] {
            let mut asked = Vec::new();
            let got = windowed(off, || {
                // SAFETY: `slots` holds a power of two of slots, `home` lies
                // below their count, every slot holds EMPTY_SLOT or an id
                // under its tag, and the id sought stands before the first
                // empty slot from the home slot on (`placeable`).
                unsafe {
                    probe(slots.as_ptr(), slots.len() - 1, home, tag_of(hash), |id| {
                        asked.push(id);
                        id == MonoId::at(SOUGHT)
                    })
                }
            });
            assert_eq!(
                got,
                walk(kinds, home),
                "{case}, {kinds:?} from {home}, window off {off}: the answer"
            );
            for id in asked {
                assert_ne!(
                    id,
                    MonoId::NONE,
                    "{case}, {kinds:?} from {home}, window off {off}: asked of the empty slot"
                );
                assert!(
                    id.index() < 100,
                    "{case}, {kinds:?} from {home}, window off {off}: asked of {id} under \
                     another tag"
                );
            }
        }
    }
}

/// The hashes of test 3, the lookup's and the other ids': all ones, whose
/// tag is the largest; 0; bit 31 alone, whose tag is 0, beside the largest
/// tag; and a hash drawn beside its neighbour.
const PAIRS: [(u64, u64); 4] = [
    (u64::MAX, 0),
    (0, u64::MAX),
    (1 << 31, (1 << 31) - 1),
    (0x9e37_79b9_7f4a_7c15, 0x9e37_79b9_7f4a_7c14),
];

/// Test 3, gf2-symbolic-speculation-time.1 to .5, .7, .8: `probe` on every
/// array of 8 slots of the four kinds that linear probing could have written
/// and on 2,000 drawn arrays of 16, from every home slot, the window's
/// candidate at its first, second, and third slot or none, tags that agree
/// before the id sought, an empty slot inside the window, a window that
/// passes the array's end, and chains that wrap, with the window and
/// without it: the walk's answer, `same` asked of no empty slot and of no id
/// under another tag.
#[test]
fn probe_answers_as_a_walk_of_the_slots() {
    const KINDS: [Kind; 4] = [Kind::Empty, Kind::Sought, Kind::Rejected, Kind::Other];
    assert_eq!(WINDOW, 3, "the window the arrays of 8 cover");
    for pair in PAIRS {
        let case = format!("{pair:x?}");
        for code in 0..KINDS.len().pow(8) {
            let kinds: Vec<Kind> = (0..8).map(|k| KINDS[(code >> (2 * k)) & 3]).collect();
            probes_as_the_walk(&kinds, pair, &case);
        }
        let mut r = rng(0x5e45);
        for _ in 0..2_000 {
            let kinds: Vec<Kind> = (0..16)
                .map(|_| match below(&mut r, 8) {
                    0 | 1 => Kind::Empty,
                    2 => Kind::Sought,
                    3 | 4 => Kind::Rejected,
                    _ => Kind::Other,
                })
                .collect();
            probes_as_the_walk(&kinds, pair, &case);
        }
    }
}
