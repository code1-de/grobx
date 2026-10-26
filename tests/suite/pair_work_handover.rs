//! pair-work-handover, the item's test code (pair-work-handover.1, .2, and
//! .4 to .6, and pair-work-ratio.1, which withdrew .3's multiple):
//! the edge cases of `OBJECTIVES.md`, and systems of random binomials whose
//! signature runs hand over at a round of the pair work, by the pair-work
//! test or by a test before it, and dense systems whose
//! runs it keeps, over GF(p) and ℚ, in grevlex and outside it
//! (`iterations/0067-pair-work-handover/plan.md`, _Test code_, 4 to 10).

use std::fmt::Write;

use grobx::{Outcome, Params};
use grobx_checker::Mode;

use crate::common::{self, P, P31, Rng, assert_accepted, params, params_in, run, with_cofactors};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::{accepted_q, assert_canonical_q};
use crate::report::{Line, report as lines_of, run_lines, untimed};

/// The edge cases of `OBJECTIVES.md` on x, y, z: name, generators, and the
/// basis over GF(65521) in grevlex.
pub(crate) const EDGE: [(&str, &str, &str); 7] = [
    ("the zero ideal", "0", "[]:\n"),
    ("F = (0, 0)", "0,\n0", "[]:\n"),
    ("no generator", "", "[]:\n"),
    ("the unit ideal", "x*z-1,\nx", "[1]:\n"),
    ("already a Gröbner basis", "x+z,\nz^2+1", "[x+z,\nz^2+1]:\n"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
        "[y^2+65520*x,\nx*y+65520*z,\nx^2+65520*y*z]:\n",
    ),
    (
        "an eliminable variable",
        "x-y^2-z,\ny*z-1,\nz^2-y",
        "[x+65519*z,\nz^2+65520*y,\ny*z+65520,\ny^2+65520*z]:\n",
    ),
];

/// The fewest S-pairs a round's update would form for the pair-work test
/// to read the round, and how many times the entries subtracted they reach
/// where it hands over: the crate's `PAIRS` and `PAIR_WORK`.
const PAIRS: u64 = 1_000;
const PAIR_WORK: u64 = 1;

/// The bound of a drawn coefficient over ℚ.
const Q_BOUND: u64 = 1 << 20;

/// A system of random binomials over GF(p), or over ℚ where p is 0: n of
/// the variables x1, x2, …, 3n generators, each a monomial of degree d and
/// another of degree 1 to d, with coefficients in [1, p), or in
/// [1, `Q_BOUND`) with a random sign over ℚ.
pub(crate) fn binomials(n: usize, d: u32, p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(17) ^ 0xb1a0);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let monomial = |rng: &mut Rng, degree: u32| -> Vec<u32> {
        let mut e = vec![0u32; n];
        for _ in 0..degree {
            e[rng.below(n as u64) as usize] += 1;
        }
        e
    };
    let mut generators = Vec::new();
    for _ in 0..3 * n {
        let lead = monomial(&mut rng, d);
        let mut other = lead.clone();
        while other == lead {
            let degree = 1 + rng.below(u64::from(d)) as u32;
            other = monomial(&mut rng, degree);
        }
        let mut terms = Vec::new();
        for e in [lead, other] {
            let c = 1 + rng.below(if p == 0 { Q_BOUND - 1 } else { p - 1 });
            let sign = if p == 0 && rng.below(2) == 1 {
                "-"
            } else {
                "+"
            };
            let mut term = format!("{sign}{c}");
            for (name, &k) in names.iter().zip(&e) {
                match k {
                    0 => {}
                    1 => write!(term, "*{name}").unwrap(),
                    k => write!(term, "*{name}^{k}").unwrap(),
                }
            }
            terms.push(term);
        }
        generators.push(terms.concat().trim_start_matches('+').to_string());
    }
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// The binomial systems of the tests over GF(p) and ℚ: on 10 to 13
/// variables of degree 3, by seed.
fn system(p: u64, seed: u64) -> String {
    binomials(10 + (seed % 4) as usize, 3, p, seed)
}

/// `params` with the report on.
pub(crate) fn reported(params: Params) -> Params {
    let mut params = params;
    params.report = true;
    params
}

/// `params` with signatures off.
pub(crate) fn unsigned(params: Params) -> Params {
    let mut params = params;
    params.signatures = false;
    params
}

/// The rounds of the run of `run`'s lines whose `handover` line names the
/// pair-work test, each checked against its `round` line and the run's
/// counts: the S-pairs are made·B + made·(made − 1)/2 with B the run's
/// inputs, the nonzero rows of its `echelon` line where it has one and its
/// nonzero generators where not (input-redundancy-time.1), and the
/// elements its rounds before made, `PAIRS` or
/// more; the test hands over where they reach `PAIR_WORK` times the
/// round's entries subtracted; the line's verdict is the test's where it
/// hands over; and a round that hands over is the run's last. The verdicts
/// in order, and, where the early, the gap, or the outgrown test handed
/// over at a round of `PAIRS` S-pairs or more, which the pair-work test
/// then does not read (pair-work-handover.6), whether its S-pairs reach
/// `PAIR_WORK` times the round's entries subtracted.
fn pair_work_rounds(case: &str, run: &[Line]) -> (Vec<bool>, Option<bool>) {
    let mut basis = run
        .iter()
        .find(|l| l.kind == "echelon")
        .or_else(|| run.iter().find(|l| l.kind == "inputs"))
        .map_or(0, |l| {
            if l.kind == "echelon" {
                l.num("rows")
            } else {
                l.num("nonzero")
            }
        });
    // Per round of `PAIRS` S-pairs or more: its number, the S-pairs, and
    // the entries subtracted.
    let mut read = Vec::new();
    let mut ahead = None;
    for line in run {
        match line.kind.as_str() {
            "round" => {
                let made = line.num("pivots");
                let pairs = made * basis + made * made.saturating_sub(1) / 2;
                basis += made;
                if pairs >= PAIRS {
                    read.push((line.num("number"), pairs, line.num("subtracted")));
                }
            }
            "handover" => {
                let round = line.num("round");
                let counts = read.iter().find(|(r, ..)| *r == round);
                let Some(verdict) = line.get("pair-work") else {
                    // pair-work-handover.6: the test reads no round that the
                    // early, the gap, or the outgrown test handed over.
                    let earlier = ["early", "gap", "outgrown"]
                        .into_iter()
                        .any(|test| line.get(test) == Some("hand-over"));
                    match counts {
                        Some(&(_, pairs, subtracted)) if earlier => {
                            ahead = Some(pairs >= PAIR_WORK * subtracted);
                        }
                        Some(_) => panic!("{case}: round {round} unread: {line:?}"),
                        None => {}
                    }
                    continue;
                };
                let &(_, pairs, subtracted) = counts.unwrap_or_else(|| {
                    panic!("{case}: round {round} reads too few pairs: {line:?}")
                });
                assert_eq!(line.num("pairs"), pairs, "{case}, round {round}: {line:?}");
                let hands = pairs >= PAIR_WORK * subtracted;
                assert_eq!(
                    verdict,
                    if hands { "hand-over" } else { "stay" },
                    "{case}: {line:?}"
                );
                if hands {
                    assert_eq!(line.value("verdict"), "hand-over", "{case}: {line:?}");
                }
            }
            _ => {}
        }
    }
    let verdicts: Vec<bool> = run
        .iter()
        .filter_map(|l| (l.kind == "handover").then(|| l.get("pair-work")).flatten())
        .map(|v| v == "hand-over")
        .collect();
    if let Some(k) = verdicts.iter().position(|&h| h) {
        assert_eq!(k + 1, verdicts.len(), "{case}: a read after the handover");
        let ran = run.last().unwrap();
        assert!(ran.is("ran", &[("outcome", "handover")]), "{case}: {ran:?}");
    }
    (verdicts, ahead)
}

/// Whether the call's first run is a signature run that hands over at a
/// round whose S-pairs the pair-work test hands over by, and that the
/// plain run then computed the basis: by the pair-work test, or by the
/// early, the gap, or the outgrown test before it at such a round
/// (pair-work-handover.6); every round's line checked by
/// `pair_work_rounds`.
pub(crate) fn handed_by_pair_work(case: &str, outcome: &Outcome) -> bool {
    let lines = lines_of(outcome);
    let first = run_lines(&lines, 1);
    if !first[0].is("run", &[("engine", "signature")]) {
        return false;
    }
    let (read, ahead) = pair_work_rounds(case, first);
    let handed = match (read.last(), ahead) {
        (Some(true), Some(_) | None) | (Some(false) | None, Some(true)) => true,
        (Some(false) | None, Some(false) | None) => false,
    };
    if handed {
        let next = run_lines(&lines, 2);
        assert!(
            next[0].is("run", &[("engine", "plain"), ("engine-by", "handover")]),
            "{case}: {run:?}",
            run = next[0]
        );
    }
    handed
}

/// Test 4: the edge cases over 65521, 7, and 2^31 − 1 in grevlex,
/// signatures on and off, with the report, and with cofactors and the
/// checker in the exact mode; in lex and `block:1`; and over ℚ in grevlex:
/// each basis canonical and accepted, and over 65521 in grevlex as
/// written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in [P, 7, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut plain = reported(params(0x9a17));
                plain.signatures = signatures;
                let outcome = run(&text, &plain);
                let name = format!("{case} over {p}, signatures {signatures}");
                assert!(!handed_by_pair_work(&name, &outcome), "{name}");
                let basis = outcome.basis.unwrap();
                let mut params = with_cofactors(0x9a17);
                params.signatures = signatures;
                let outcome = run(&text, &params);
                assert_eq!(outcome.basis.as_deref(), Some(basis.as_str()), "{name}");
                assert_canonical(&name, &text, &basis, "grevlex");
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                if p == P {
                    assert_eq!(basis, want, "{name}");
                }
            }
            for order in ["lex", "block:1"] {
                accepted_in(&format!("{case} over {p}"), &text, order, Mode::Exact);
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}

/// Test 5, pair-work-handover.1, .2, and .4 to .6, and pair-work-ratio.1:
/// binomial systems over 65521 and 2^31 − 1, 16 seeds each, in grevlex.
/// Each basis with signatures is the one without, canonical, and the one
/// with cofactors, which the checker
/// accepts in the exact mode. Every `handover` line that names the
/// pair-work test agrees with its round's counts, and at least 12 calls
/// at each prime hand over at a round whose counts it hands over by: by
/// it, or by the early, the gap, or the outgrown test before it
/// (pair-work-handover.6), the early test reading a round of
/// `EARLY_ROUND` elements or more where the plain run combines rows
/// (outgrown-round-size.2).
#[test]
fn binomial_systems_hand_over_by_the_pair_work() {
    for p in [P, P31] {
        let mut handed = 0;
        for seed in 0..16 {
            let text = system(p, seed);
            let case = format!("binomials over {p}, seed {seed}");
            let outcome = run(&text, &reported(params(seed)));
            handed += usize::from(handed_by_pair_work(&case, &outcome));
            let basis = outcome.basis.unwrap();
            let off = run(&text, &unsigned(params(seed)));
            assert_eq!(off.basis.as_deref(), Some(basis.as_str()), "{case}");
            assert_canonical(&case, &text, &basis, "grevlex");
            let with = run(&text, &with_cofactors(seed));
            assert_eq!(with.basis.as_deref(), Some(basis.as_str()), "{case}");
            assert_accepted(&case, &text, &with, Mode::Exact);
        }
        assert!(
            handed >= 12,
            "over {p}: {handed} of 16 handed over at a round of the pair work"
        );
    }
}

/// Test 6, pair-work-handover.1 and .2: dense systems over 65521 on six
/// variables of degree 3, 3 seeds, whose rounds' reductions outweigh their
/// S-pairs: the pair-work test reads rounds and keeps the run at each, by
/// the lines test 5 checks, and each basis with signatures is the one
/// without.
#[test]
fn dense_systems_stay() {
    for seed in 0..3 {
        let text = common::dense(6, 3, P, seed);
        let case = format!("dense over {P}, seed {seed}");
        let outcome = run(&text, &reported(params(seed)));
        let lines = lines_of(&outcome);
        let (verdicts, _) = pair_work_rounds(&case, run_lines(&lines, 1));
        assert!(
            !verdicts.is_empty(),
            "{case}: the pair-work test read no round"
        );
        assert!(verdicts.iter().all(|&h| !h), "{case}: {verdicts:?}");
        let off = run(&text, &unsigned(params(seed)));
        assert_eq!(off.basis, outcome.basis, "{case}");
    }
}

/// Test 7: binomial systems over 7, where the plain run reduces each row
/// on its own, 8 seeds: each basis with signatures the one without,
/// canonical, and the `handover` lines as test 5 checks them.
#[test]
fn binomial_systems_over_seven() {
    for seed in 0..8 {
        let text = system(7, seed);
        let case = format!("binomials over 7, seed {seed}");
        let outcome = run(&text, &reported(params(seed)));
        handed_by_pair_work(&case, &outcome);
        let basis = outcome.basis.unwrap();
        let off = run(&text, &unsigned(params(seed)));
        assert_eq!(off.basis.as_deref(), Some(basis.as_str()), "{case}");
        assert_canonical(&case, &text, &basis, "grevlex");
    }
}

/// Test 8: binomial systems over ℚ, 8 seeds: each basis with signatures
/// the one without and canonical, the learn run's `handover` lines as
/// test 5 checks them, at least 4 learns handed over at a round whose
/// counts the pair-work test hands over by (`handed_by_pair_work`), and
/// the basis with cofactors, which the checker accepts.
#[test]
fn binomial_systems_over_q() {
    let mut handed = 0;
    for seed in 0..8 {
        let text = system(0, seed);
        let case = format!("binomials over ℚ, seed {seed}");
        let outcome = run(&text, &reported(params(seed)));
        handed += usize::from(handed_by_pair_work(&case, &outcome));
        let basis = outcome.basis.unwrap();
        let off = run(&text, &unsigned(params(seed)));
        assert_eq!(off.basis.as_deref(), Some(basis.as_str()), "{case}");
        assert_canonical_q(&case, &text, &basis, "grevlex");
        let with = accepted_q(&case, &text, "grevlex");
        assert_eq!(with.basis.as_deref(), Some(basis.as_str()), "{case}");
    }
    assert!(
        handed >= 4,
        "over ℚ: {handed} of 8 learns handed over at a round of the pair work"
    );
}

/// Test 9, gfp-unsolved.17 kept: a call that tracks cofactors keeps its
/// signature run to its end, and its report holds no `handover` line;
/// and one seed computes one path: two calls under it write the same
/// report but for its times and resident sets.
#[test]
fn cofactors_keep_the_run_and_a_seed_its_path() {
    for seed in 0..4 {
        let text = system(P, seed);
        let case = format!("binomials over {P}, seed {seed}");
        let outcome = run(&text, &reported(with_cofactors(seed)));
        let lines = lines_of(&outcome);
        let first = run_lines(&lines, 1);
        assert!(
            first[0].is("run", &[("engine", "signature"), ("handover", "off")]),
            "{case}: {run:?}",
            run = first[0]
        );
        assert!(lines.iter().all(|l| l.kind != "handover"), "{case}");
        assert!(
            first.last().unwrap().is("ran", &[("outcome", "basis")]),
            "{case}"
        );
        let again = |params: Params| untimed(&lines_of(&run(&text, &params)));
        assert_eq!(
            again(reported(params(seed))),
            again(reported(params(seed))),
            "{case}: one seed, two paths"
        );
    }
}

/// Test 10: binomial systems over 65521 in lex and in `block:3`, 4 seeds:
/// each basis with signatures the one without, and canonical in its order.
#[test]
fn binomial_systems_outside_grevlex() {
    for order in ["lex", "block:3"] {
        for seed in 0..4 {
            let text = binomials(8, 2, P, seed);
            let case = format!("binomials over {P} in {order}, seed {seed}");
            let on = run(&text, &params_in(order, seed));
            let off = run(&text, &unsigned(params_in(order, seed)));
            assert_eq!(on.basis, off.basis, "{case}");
            assert_canonical(&case, &text, on.basis.as_deref().unwrap(), order);
        }
    }
}
