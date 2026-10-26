//! monomial-formats: every width gives one outcome on the cases it holds,
//! with one record, and an overflow at each width's bound starts the call
//! over in the next (plan.md, _Test code_, but its test on the tuning set:
//! on generated systems, the records compared in place of the traces, at
//! the degree bounds monomial-time set).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::entry::{self, Width};
use crate::f4::Halt;
use crate::fglm::{self, MAX_STAIRCASE};
use crate::fglm_tests::PRIMES;
use crate::id::{MonoId, RowId};
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::Grevlex;
use crate::systems::{dense, random_square, random_system};
use crate::table::{Limit, Table};
use crate::trace::{NoLearner, NoTrace, Record, Recording, RowSource, Trace};
use crate::{CheckerMode, Error, Outcome, Params, Threads};

const WIDTHS: [Width; 3] = [Width::W8, Width::W16, Width::W32];

/// The deadline of each call of `every_width` and of
/// `the_outcome_and_the_record_agree_in_every_width`.
const DEADLINE: Duration = Duration::from_millis(200);

/// One thread and a fixed seed; with cofactors, the checker in the exact
/// mode.
fn params(order: &str, cofactors: bool) -> Params {
    let order = order.parse().expect("a name of the family");
    let mut p = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), 0x5eed);
    p.cofactors = cofactors;
    if cofactors {
        p.checker = Some(CheckerMode::Exact);
    }
    p
}

/// A trace that keeps the rows of every matrix it hears of, in order.
#[derive(Default)]
struct Heard(Vec<Vec<RowSource>>);

impl Trace for Heard {
    fn matrix(&mut self, rows: &[RowSource]) {
        self.0.push(rows.to_vec());
    }

    fn counted(&mut self, _: &[RowId]) {}
}

/// The call in the widths from `first` to `last`.
fn call(text: &str, params: &Params, first: Width, last: Width) -> Result<Outcome, Error> {
    entry::groebner_in(text, params, first, last, &mut NoTrace, &mut NoLearner)
}

/// The call in the width `w` alone.
fn alone(text: &str, params: &Params, w: Width) -> Result<Outcome, Error> {
    call(text, params, w, w)
}

/// The call in the widths from `first` to `last`, with the matrices it
/// heard and its record.
fn heard(
    text: &str,
    params: &Params,
    first: Width,
    last: Width,
) -> (Result<Outcome, Error>, Heard, Record) {
    let mut trace = Heard::default();
    let mut recording = Recording::new();
    let got = entry::groebner_in(text, params, first, last, &mut trace, &mut recording);
    (got, trace, recording.finish(params.order.clone()))
}

// monomial-formats.15 design -- the test code compares the formats on whole outcomes, the basis, the cofactors, the stats, the checker's verdict, and the record, not the basis alone (not measured)
/// Each width alone returns W32's outcome or `Error::Overflow`, and the call
/// in all widths returns the narrowest's that holds the case
/// (monomial-formats.15). None where a deadline stopped a width, or the
/// call in all widths.
fn one_outcome(name: &str, text: &str, params: &Params) -> Option<Outcome> {
    let outcomes: Vec<Result<Outcome, Error>> =
        WIDTHS.iter().map(|&w| alone(text, params, w)).collect();
    let stopped = |o: &Result<Outcome, Error>| match o {
        Ok(o) => o.stopped.is_some(),
        Err(_) => false,
    };
    if outcomes.iter().any(stopped) {
        return None;
    }
    let widest = match &outcomes[2] {
        Ok(o) => o.clone(),
        Err(e) => panic!(
            "{name} in {order}: W32: {e:?}\n{text}",
            order = params.order
        ),
    };
    for (w, got) in WIDTHS.iter().zip(&outcomes) {
        match got {
            Ok(o) => assert_eq!(
                o,
                &widest,
                "{name} in {order}: {w:?} and W32\n{text}",
                order = params.order
            ),
            Err(Error::Overflow { .. }) => {}
            Err(e) => panic!(
                "{name} in {order}: {w:?}: {e:?}\n{text}",
                order = params.order
            ),
        }
    }
    let all = call(text, params, Width::W8, Width::W32);
    if stopped(&all) {
        return None;
    }
    assert_eq!(
        all.ok(),
        Some(widest.clone()),
        "{name} in {order}: all widths",
        order = params.order
    );
    Some(widest)
}

/// Every case in grevlex, lex, and `block:1`, with and without cofactors, a
/// `DEADLINE` on each call: one outcome, compared on at least nine
/// cases in ten.
fn every_width(cases: &[(String, String)]) {
    let (mut compared, mut total) = (0, 0);
    for (name, text) in cases {
        for order in ["grevlex", "lex", "block:1"] {
            for cofactors in [false, true] {
                let mut p = params(order, cofactors);
                p.deadline = Some(DEADLINE);
                total += 1;
                if one_outcome(name, text, &p).is_some() {
                    compared += 1;
                } else {
                    eprintln!("{name} in {order}, cofactors {cofactors}: stopped by the deadline");
                }
            }
        }
    }
    assert!(10 * compared >= 9 * total, "{compared} of {total} compared");
}

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, as `fglm_edge.rs`
/// writes them.
const EDGE: &[(&str, &str)] = &[
    ("zero", "x,y\n65521\n0\n"),
    ("zeros", "x,y\n7\n0,\n0\n"),
    ("none", "x,y\n65521\n"),
    ("x-plus-x", "x,y\n2\nx+x\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n"),
    ("unit-p2", "x,y\n2\nx,\nx+1\n"),
    ("unit-constant", "x,y\n65521\n5\n"),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
    ),
    ("gb", "x,y\n65521\nx+y,\ny^2+1\n"),
    ("gb-swapped", "x,y\n65521\ny^2+1,\nx+y\n"),
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    (
        "eliminable-first",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
    ),
    ("eliminable-last", "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
    (
        "eliminable-linear",
        "x,y,z\n65521\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n",
    ),
    ("positive-dimensional", "x,y\n65521\nx*y-1\n"),
];

#[test]
fn the_edge_cases_in_every_width() {
    let cases: Vec<(String, String)> = EDGE
        .iter()
        .map(|&(name, text)| (name.to_string(), text.to_string()))
        .collect();
    every_width(&cases);
}

#[test]
fn random_systems_in_every_width() {
    let mut cases = Vec::new();
    for p in PRIMES {
        for seed in 0..10 {
            cases.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
    }
    every_width(&cases);
}

#[test]
fn square_systems_in_every_width() {
    let mut cases = Vec::new();
    for p in PRIMES {
        for seed in 0..10 {
            cases.push((format!("square-p{p}-{seed}"), random_square(p, seed)));
        }
    }
    every_width(&cases);
}

#[test]
fn dense_systems_in_every_width() {
    let cases: Vec<(String, String)> = (2..=3)
        .map(|m| (format!("cubic-{m}"), dense(m, 3, 65521, m as u64)))
        .collect();
    every_width(&cases);
}

/// The same call in every width returns the same outcome and records the
/// same record, signatures on and off and in every order. A case whose
/// `W8` call `DEADLINE` stops is left out.
#[test]
fn the_outcome_and_the_record_agree_in_every_width() {
    let mut cases: Vec<String> = vec![dense(3, 3, 65521, 1), dense(3, 3, 65521, 2)];
    for seed in 0..5 {
        cases.push(random_system(65521, seed));
        cases.push(random_square(7, seed));
    }
    for text in &cases {
        for (order, signatures) in [
            ("grevlex", true),
            ("grevlex", false),
            ("lex", true),
            ("block:1", false),
        ] {
            let mut p = params(order, false);
            p.signatures = signatures;
            p.deadline = Some(DEADLINE);
            let (first, _, record) = heard(text, &p, Width::W8, Width::W8);
            let first = first.unwrap();
            if first.stopped.is_some() {
                eprintln!("{order} {signatures}: stopped by the deadline\n{text}");
                continue;
            }
            for w in [Width::W16, Width::W32] {
                let (got, _, r) = heard(text, &p, w, w);
                assert_eq!(got.unwrap(), first, "{order} {signatures} {w:?}\n{text}");
                assert_eq!(r, record, "{order} {signatures} {w:?}\n{text}");
            }
        }
    }
}

/// Cases a width passes: the text, its order, the widest width it passes,
/// and whether its input passes it.
const PAST: &[(&str, &str, Width, bool)] = &[
    ("x\n65521\nx^256-1\n", "grevlex", Width::W8, true),
    ("x\n65521\nx^65536-1\n", "grevlex", Width::W16, true),
    (
        "x,y\n65521\nx^100*y-1,\nx*y^100-1\n",
        "grevlex",
        Width::W8,
        false,
    ),
    (
        "x,y\n65521\nx^20000*y-1,\nx*y^20000-1\n",
        "grevlex",
        Width::W16,
        false,
    ),
    ("x,y\n65521\nx^16-1,\ny^17-1\n", "lex", Width::W8, false),
    ("x,y\n65521\nx^300-y,\ny^300-x\n", "lex", Width::W16, false),
    (
        "x,y\n65521\nx^300-y,\ny^300-x\n",
        "block:1",
        Width::W16,
        false,
    ),
];

/// Each case alone in the width it passes is `Error::Overflow`, with the
/// line where the input passes it; alone in the next it holds; and the call
/// in all widths returns that outcome. Cut at the width it passes, the call
/// stops with `Error::Overflow`.
#[test]
fn each_bound_starts_the_call_over() {
    for &(text, order, past, input) in PAST {
        for cofactors in [false, true] {
            for signatures in [true, false] {
                let mut p = params(order, cofactors);
                p.signatures = signatures;
                let case =
                    format!("{text:?} in {order}, cofactors {cofactors}, signatures {signatures}");
                match alone(text, &p, past) {
                    // An input that passes the width does so in its one
                    // generator, on line 3.
                    Err(Error::Overflow { line }) => {
                        assert_eq!(line, input.then_some(3), "{case}");
                    }
                    got @ (Ok(_) | Err(_)) => panic!("{case}: {past:?} alone: {got:?}"),
                }
                for w in WIDTHS.iter().filter(|&&w| w < past) {
                    let got = alone(text, &p, *w);
                    assert!(
                        matches!(got, Err(Error::Overflow { .. })),
                        "{case}: {w:?}: {got:?}"
                    );
                }
                let next = past.wider().unwrap();
                let held = alone(text, &p, next).unwrap();
                assert!(held.basis.is_some(), "{case}: {next:?}: {held:?}");
                if cofactors {
                    assert_eq!(
                        held.certified,
                        Some(true),
                        "{case}: {flaw:?}",
                        flaw = held.flaw
                    );
                }
                assert_eq!(
                    call(text, &p, Width::W8, Width::W32).unwrap(),
                    held,
                    "{case}"
                );
                let cut = call(text, &p, Width::W8, past);
                assert!(
                    matches!(cut, Err(Error::Overflow { .. })),
                    "{case}: {cut:?}"
                );
            }
        }
    }
}

/// Cases a width holds at its bound: the text, its order, and the narrowest
/// width that holds it.
const AT: &[(&str, &str, Width)] = &[
    ("x\n65521\nx^127-1\n", "grevlex", Width::W8),
    ("x\n65521\nx^32767-1\n", "grevlex", Width::W16),
    ("x,y\n65521\nx^2147483647-y\n", "grevlex", Width::W32),
    ("x,y\n65521\nx-1,\ny^127-1\n", "lex", Width::W8),
];

/// At the bound the width holds the case, every wider width gives its
/// outcome, and a narrower one is `Error::Overflow`.
#[test]
fn each_bound_holds_its_own() {
    for &(text, order, w) in AT {
        for cofactors in [false, true] {
            let p = params(order, cofactors);
            let case = format!("{text:?} in {order}, cofactors {cofactors}");
            let held = alone(text, &p, w).unwrap_or_else(|e| panic!("{case}: {w:?}: {e:?}"));
            assert!(held.basis.is_some(), "{case}");
            for v in WIDTHS {
                let got = alone(text, &p, v);
                if v < w {
                    assert!(
                        matches!(got, Err(Error::Overflow { .. })),
                        "{case}: {v:?}: {got:?}"
                    );
                } else {
                    assert_eq!(got.unwrap(), held, "{case}: {v:?}");
                }
            }
            assert_eq!(
                call(text, &p, Width::W8, Width::W32).unwrap(),
                held,
                "{case}"
            );
        }
    }
}

/// Past 2^31 − 1 every width is `Error::Overflow`: in the run, no line; in
/// the input, its line.
#[test]
fn past_the_widest_width() {
    let cases = [
        ("x,y\n65521\nx^1500000000*y-1,\nx*y^1500000000-1\n", None),
        ("x,y\n65521\nx^2147483648-y\n", Some(3)),
    ];
    for (text, line) in cases {
        for order in ["grevlex", "lex", "block:1"] {
            let p = params(order, false);
            for w in WIDTHS {
                let got = alone(text, &p, w);
                assert!(
                    matches!(got, Err(Error::Overflow { .. })),
                    "{text:?} {order} {w:?}: {got:?}"
                );
            }
            let got = call(text, &p, Width::W8, Width::W32);
            assert_eq!(got, Err(Error::Overflow { line }), "{text:?} {order}");
        }
    }
}

/// A call that starts over hears the matrices of the run the overflow
/// stopped, then the next run's, and records and counts the next run's.
#[test]
fn a_call_that_starts_over_hears_every_run() {
    let cases = [
        ("x,y\n65521\nx^100*y-1,\nx*y^100-1\n", "grevlex"),
        ("x,y\n65521\nx^16-1,\ny^17-1\n", "lex"),
    ];
    for (text, order) in cases {
        for signatures in [true, false] {
            let mut p = params(order, false);
            p.signatures = signatures;
            let (all, all_trace, all_record) = heard(text, &p, Width::W8, Width::W32);
            let (w8, trace8, _) = heard(text, &p, Width::W8, Width::W8);
            let (w16, trace16, record16) = heard(text, &p, Width::W16, Width::W16);
            assert!(
                matches!(w8, Err(Error::Overflow { line: None })),
                "{text:?}: {w8:?}"
            );
            let mut want = trace8.0;
            want.extend(trace16.0);
            assert_eq!(all_trace.0, want, "{text:?} {signatures}");
            assert_eq!(all_record, record16, "{text:?} {signatures}");
            assert_eq!(all.unwrap(), w16.unwrap(), "{text:?} {signatures}");
        }
    }
    // The lex call's W8 run is a whole grevlex run: its trace is not empty.
    let (_, trace8, _) = heard(cases[1].0, &params("lex", false), Width::W8, Width::W8);
    assert_ne!(trace8.0.len(), 0, "the lex call's W8 run heard no matrix");
}

/// A record replays in every width that holds its run, with the learn run's
/// basis and one outcome, and alone in a width its run passes it is
/// `Error::Overflow`. Every width holds the runs of the cases marked so.
#[test]
fn a_replay_in_every_width() {
    let cases = [
        (dense(3, 3, 65521, 1), "grevlex", true),
        (dense(3, 3, 65521, 2), "lex", true),
        (
            "x,y\n65521\nx^100*y-1,\nx*y^100-1\n".to_string(),
            "grevlex",
            false,
        ),
        ("x,y\n65521\nx^16-1,\ny^17-1\n".to_string(), "lex", false),
    ];
    for (text, order, every) in &cases {
        let p = params(order, false);
        let (learned, _, record) = heard(text, &p, Width::W8, Width::W32);
        let learned = learned.unwrap();
        let replayed = entry::replay_in(text, &p, &record, Width::W8, Width::W32, &mut NoTrace)
            .unwrap_or_else(|e| panic!("{order}: {e:?}\n{text}"));
        assert_eq!(replayed.basis, learned.basis, "{order}\n{text}");
        for w in WIDTHS {
            match entry::replay_in(text, &p, &record, w, w, &mut NoTrace) {
                Ok(o) => assert_eq!(o, replayed, "{order} {w:?}\n{text}"),
                Err(Error::Overflow { .. }) => {
                    assert!(!every && w < Width::W32, "{order} {w:?}\n{text}");
                }
                Err(e) => panic!("{order} {w:?}: {e:?}\n{text}"),
            }
        }
    }
    for (text, order, _) in cases.iter().filter(|(_, _, every)| !every) {
        let p = params(order, false);
        let (_, _, record) = heard(text, &p, Width::W8, Width::W32);
        let got = entry::replay_in(text, &p, &record, Width::W8, Width::W8, &mut NoTrace);
        assert!(
            matches!(got, Err(Error::Overflow { line: None })),
            "{order}: {got:?}\n{text}"
        );
    }
}

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// `intern`, `mul`, and `lcm` at the bound and one past it; an overflow
/// stores nothing; a degree sum one past the bound does not wrap.
fn table_bound<M: Format>() {
    let max = M::MAX_DEGREE;
    let half = max / 2 + 1;
    let mut t = Table::new(M::new(2), &mut rng(3));
    let top = t.intern(&[max, 0]).unwrap();
    assert_eq!(t.degree(top), max);
    assert_eq!(t.intern(&[max + 1, 0]), Err(Limit::Overflow));
    assert_eq!(t.intern(&[half, half]), Err(Limit::Overflow));
    let x = t.intern(&[half, 0]).unwrap();
    let y = t.intern(&[0, half]).unwrap();
    let len = t.len();
    assert_eq!(t.mul(x, y), Err(Limit::Overflow));
    assert_eq!(t.mul(x, x), Err(Limit::Overflow));
    assert_eq!(t.lcm(x, y), Err(Limit::Overflow));
    assert_eq!(t.len(), len, "an overflow stores nothing");
    assert_eq!(t.lcm_degree(x, y), max + 1);
    let z = t.intern(&[max - half, 0]).unwrap();
    let p = t.mul(x, z).unwrap();
    assert_eq!(p, top);
    assert_eq!(t.div(p, z), Ok(x));
    assert_eq!(t.lcm(top, x), Ok(top));
}

#[test]
fn the_table_at_each_bound() {
    table_bound::<Dense8>();
    table_bound::<Dense16>();
    table_bound::<Dense32>();
}

/// One sequence of table operations: each result's id, hash, mask, and
/// degree.
fn sequence<M: Format>() -> Vec<(MonoId, u64, u32, u32)> {
    let n = 5;
    let mut t = Table::new(M::new(n), &mut rng(9));
    let mut draw = rng(11);
    let mut ids = Vec::new();
    for _ in 0..300 {
        let mut e = vec![0u32; n];
        for _ in 0..12 {
            e[(draw.next_u64() % n as u64) as usize] += (draw.next_u64() % 4) as u32;
        }
        ids.push(t.intern(&e).unwrap());
    }
    let mut out = Vec::new();
    for k in 0..ids.len() - 1 {
        let (a, b) = (ids[k], ids[k + 1]);
        let p = t.mul(a, b).unwrap();
        let l = t.lcm(a, b).unwrap();
        let q = t.div(p, b).unwrap();
        for id in [a, p, l, q] {
            out.push((id, t.hash(id), t.mask(id), t.degree(id)));
        }
    }
    out
}

/// One sequence of operations gives the same ids, hashes, masks, and
/// degrees in every format: the reason a run does the same work in each.
#[test]
fn the_table_is_the_same_in_every_format() {
    let narrow = sequence::<Dense8>();
    assert_eq!(narrow, sequence::<Dense16>());
    assert_eq!(narrow, sequence::<Dense32>());
}

/// The staircase of the leading monomials x^a, y^b: its size, None, or
/// `Halt::Overflow`.
fn staircase<M: Format>(a: u32, b: u32) -> Result<Option<usize>, Halt> {
    let mut t = Table::new(M::new(2), &mut rng(1));
    let leads = [t.intern(&[a, 0]).unwrap(), t.intern(&[0, b]).unwrap()];
    let p = params("lex", false);
    let mut budget = Budget::new(Instant::now(), &p).unwrap();
    fglm::staircase(&mut t, &Grevlex, &leads, &mut budget).map(|s| s.map(|s| s.len()))
}

/// 127 monomials in `Dense8`, 128 an overflow; 32767 in `Dense16`, 32768
/// an overflow; 65535 in `Dense32`, 65536 None (monomial-formats.10, .11).
#[test]
fn the_staircase_at_each_bound() {
    assert_eq!(MAX_STAIRCASE, 65535);
    assert_eq!(staircase::<Dense8>(1, 127), Ok(Some(127)));
    assert_eq!(staircase::<Dense8>(8, 16), Err(Halt::Overflow));
    assert_eq!(staircase::<Dense16>(8, 16), Ok(Some(128)));
    assert_eq!(staircase::<Dense16>(151, 217), Ok(Some(32767)));
    assert_eq!(staircase::<Dense16>(128, 256), Err(Halt::Overflow));
    assert_eq!(staircase::<Dense32>(255, 257), Ok(Some(65535)));
    assert_eq!(staircase::<Dense32>(256, 256), Ok(None));
}

/// Dense cubic systems on five variables, D = 243, in lex: `W8` alone
/// passes its staircase, and `W16`, `W32`, and the call give one outcome.
#[test]
fn a_lex_staircase_past_w8() {
    for seed in 1..=2 {
        let text = dense(5, 3, 65521, seed);
        let p = params("lex", false);
        let got = alone(&text, &p, Width::W8);
        assert!(
            matches!(got, Err(Error::Overflow { line: None })),
            "seed {seed}: {got:?}"
        );
        let held = alone(&text, &p, Width::W16).unwrap();
        assert_eq!(alone(&text, &p, Width::W32).unwrap(), held, "seed {seed}");
        assert_eq!(
            call(&text, &p, Width::W8, Width::W32).unwrap(),
            held,
            "seed {seed}"
        );
    }
}
