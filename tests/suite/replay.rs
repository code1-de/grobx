//! trace-replay, the item's test code: a record learned over one prime and
//! replayed over the same prime and over others, against a run without the
//! record and against the checker; the records that do not fit; the edge
//! cases and the bounds (plan.md, _Test code_).

use std::num::NonZeroU32;
use std::time::Duration;

use grobx::trace::{NoTrace, Record};
use grobx::{Cancel, CheckerMode, Error, MonomialOrder, Outcome, Params, Stop};
use grobx_checker::Mode;

use crate::common::{
    P, P31, PRIMES, assert_accepted, degree_bound, dense, params, points, random_system,
};

/// The primes above 2^30 the generated systems are replayed over.
const OTHER: [u64; 3] = [2_147_483_647, 2_147_483_629, 1_073_741_827];

/// The parameters of a measured call, with or without signatures and
/// cofactors.
fn with(seed: u64, signatures: bool, cofactors: bool) -> Params {
    let mut p = params(seed);
    p.signatures = signatures;
    p.cofactors = cofactors;
    p
}

/// The instance with its characteristic, the second line, replaced by p.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let variables = lines.next().unwrap();
    lines.next();
    let rest = lines.next().unwrap_or("");
    format!("{variables}\n{p}\n{rest}")
}

/// The record of a learn run that returns a basis, and its outcome.
fn learn(name: &str, text: &str, params: &Params) -> (Outcome, Record) {
    let (outcome, record) = grobx::learn(text, params, &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: learn failed: {e:?}\n{text}"));
    assert!(outcome.basis.is_some(), "{name}: the learn run stopped");
    (
        outcome,
        record.unwrap_or_else(|| panic!("{name}: no record")),
    )
}

/// The outcome of a replay that must fit and must not be stopped.
fn replay(name: &str, text: &str, params: &Params, record: &Record) -> Outcome {
    let outcome = grobx::replay(text, params, record, &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: the replay failed: {e:?}\n{text}"));
    assert!(
        outcome.stopped.is_none() && outcome.basis.is_some(),
        "{name}: the replay stopped: {outcome:?}"
    );
    outcome
}

/// The basis of a run without a record.
fn basis(text: &str) -> String {
    let outcome = grobx::groebner(text, &params(1)).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    outcome.basis.expect("the run returns a basis")
}

/// The dense cubic systems on two to four variables over GF(65521).
fn dense_systems() -> Vec<(String, String)> {
    (2..=4)
        .map(|m| (format!("cubic-{m}"), dense(m, 3, P, m as u64)))
        .collect()
}

/// The tiny instance, the dense systems, and random systems over every
/// prime of `PRIMES`: name and text.
fn small_instances(seeds: u64) -> Vec<(String, String)> {
    let mut list = vec![(
        "tiny".to_string(),
        "x,y\n65521\nx^2+y,\nx*y+1\n".to_string(),
    )];
    list.extend(dense_systems());
    for p in PRIMES {
        for seed in 0..seeds {
            list.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
    }
    list
}

/// Texts over GF(65521) with a coefficient that is 0 mod 65521 alone:
/// read over another prime, each is another ideal, with other leading
/// monomials.
const OTHER_IDEAL: &[(&str, &str)] = &[
    ("a vanishing leading term", "x,y\n65521\n65521*x^3+x*y-1,\ny^2-x\n"),
    ("a vanishing tail", "x,y\n65521\nx^2+65521*y,\nx*y+1\n"),
];

/// The leading monomials of a basis text, whose polynomials are monic.
fn leads(basis: &str) -> Vec<String> {
    let body = basis.trim_start_matches('[').trim_end_matches("]:\n");
    body.split(",\n")
        .filter(|g| !g.is_empty())
        .map(|g| {
            let end = g[1..].find(['+', '-']).map_or(g.len(), |k| k + 1);
            g[..end].to_string()
        })
        .collect()
}

/// The edge cases of plan.md: name and instance over GF(65521).
const EDGE: &[(&str, &str)] = &[
    ("zero", "x,y\n65521\n0\n"),
    ("no generator", "x,y\n65521\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n"),
    ("constant", "x,y\n65521\n1\n"),
    ("constant among others", "x,y\n65521\nx^2+y,\nx*y+1,\n7\n"),
    ("already a basis", "x,y\n65521\nx+y,\ny^2+1\n"),
    ("zero generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    ("eliminable", "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
];

#[test]
fn a_replay_over_the_learn_prime_returns_the_learned_basis() {
    for (name, text) in small_instances(3) {
        for signatures in [false, true] {
            let name = format!("{name}, signatures {signatures}");
            let params = with(1, signatures, false);
            let (learned, record) = learn(&name, &text, &params);
            let replayed = replay(&name, &text, &params, &record);
            assert_eq!(replayed.basis, learned.basis, "{name}");
        }
    }
}

#[test]
fn a_replay_over_other_primes_returns_the_basis_of_a_run() {
    let mut cases = dense_systems();
    for seed in 0..5 {
        cases.push((format!("random-p{P}-{seed}"), random_system(P, seed)));
    }
    for (name, text) in cases {
        for signatures in [false, true] {
            let (_, record) = learn(&name, &text, &with(1, signatures, false));
            for p in OTHER {
                let name = format!("{name}, signatures {signatures}, over {p}");
                let there = over(&text, p);
                let replayed = replay(&name, &there, &with(2, signatures, false), &record);
                assert_eq!(
                    replayed.basis.as_deref(),
                    Some(basis(&there).as_str()),
                    "{name}: the replay's basis differs from a run's"
                );
            }
        }
    }
}

#[test]
fn a_record_of_another_ideal_does_not_fit() {
    for &(case, text) in OTHER_IDEAL {
        for signatures in [false, true] {
            let (learned, record) = learn(case, text, &with(1, signatures, false));
            for p in OTHER {
                let name = format!("{case}, signatures {signatures}, over {p}");
                let there = over(text, p);
                assert_ne!(
                    leads(&basis(&there)),
                    leads(learned.basis.as_deref().unwrap()),
                    "{name}: the same leading monomials"
                );
                misfit(&name, &there, &record);
            }
        }
    }
}

#[test]
fn the_checker_accepts_a_replay() {
    let mut cases = dense_systems();
    for (name, text) in EDGE {
        cases.push((name.to_string(), text.to_string()));
    }
    for seed in 0..5 {
        cases.push((format!("random-p{P}-{seed}"), random_system(P, seed)));
    }
    for (name, text) in cases {
        for signatures in [false, true] {
            let name = format!("{name}, signatures {signatures}");
            // A record a replay with cofactors fits: the echelon off
            // (input-redundancy-time.8).
            let mut learn_params = with(1, signatures, false);
            learn_params.echelon = false;
            let (_, record) = learn(&name, &text, &learn_params);
            let there = over(&text, P31);
            let outcome = replay(&name, &there, &with(3, signatures, true), &record);
            let k = points(P31, degree_bound(&there, &outcome));
            let mode = Mode::Cheap {
                points: NonZeroU32::new(k).unwrap(),
                seed: 0x7e9a ^ name.len() as u64,
            };
            assert_accepted(&name, &there, &outcome, mode);
        }
    }
    let tiny = "x,y\n65521\nx^2+y,\nx*y+1\n";
    let mut learn_params = with(1, true, false);
    learn_params.echelon = false;
    let (_, record) = learn("tiny", tiny, &learn_params);
    let mut exact = with(1, true, false);
    exact.checker = Some(CheckerMode::Exact);
    let outcome = replay("tiny", &over(tiny, P31), &exact, &record);
    assert_eq!(outcome.certified, Some(true), "{outcome:?}");
}

/// The step of a misfit, or a panic.
fn misfit(name: &str, text: &str, record: &Record) -> u32 {
    match grobx::replay(text, &params(1), record, &mut NoTrace) {
        Err(Error::Misfit { step }) => step,
        other @ (Ok(_) | Err(_)) => {
            panic!("{name}: a record that does not fit gave {other:?}")
        }
    }
}

#[test]
fn a_record_that_does_not_fit_is_detected() {
    let tiny = "x,y\n65521\nx^2+y,\nx*y+1\n";
    for signatures in [false, true] {
        let params = with(1, signatures, false);
        for (name, text, p, at_inputs) in [
            (
                "a vanishing leading coefficient",
                "x,y\n65521\n3*x^2+y,\nx*y+1\n",
                3,
                true,
            ),
            (
                "a vanishing generator",
                "x,y\n65521\n3*x+6*y,\nx*y+1\n",
                3,
                true,
            ),
            ("a vanishing constant", "x,y\n65521\n5\n", 5, true),
            (
                "a leading term lost in a round",
                "x,y,z\n65521\nx^2+3*y,\nx*y+z\n",
                3,
                false,
            ),
            (
                "a row reduced to zero",
                "x,y,z\n65521\nx^2+3*y,\nx*y+3*z\n",
                3,
                false,
            ),
        ] {
            let name = format!("{name}, signatures {signatures}");
            let (_, record) = learn(&name, text, &params);
            let step = misfit(&name, &over(text, p), &record);
            if at_inputs {
                assert_eq!(step, 0, "{name}");
            } else {
                assert!(step >= 1, "{name}: misfit at step {step}");
            }
        }
        let (_, record) = learn("tiny", tiny, &params);
        assert_eq!(
            misfit("more variables", "x,y,z\n65521\nx^2+y,\nx*y+1\n", &record),
            0
        );
        assert_eq!(
            misfit("fewer generators", "x,y\n65521\nx^2+y\n", &record),
            0
        );
        let step = misfit("another instance", "x,y\n65521\nx^2+x,\nx*y+y\n", &record);
        assert!(step >= 1, "another instance: misfit at step {step}");
    }
}

#[test]
fn the_edge_cases() {
    for (name, text) in EDGE {
        for signatures in [false, true] {
            let name = format!("{name}, signatures {signatures}");
            let (_, record) = learn(&name, text, &with(1, signatures, false));
            for p in [P, P31] {
                let there = over(text, p);
                let replayed = replay(&name, &there, &with(1, signatures, false), &record);
                assert_eq!(
                    replayed.basis.as_deref(),
                    Some(basis(&there).as_str()),
                    "{name}, over {p}"
                );
            }
        }
    }
}

#[test]
fn the_bounds_stop_a_replay() {
    let text = dense(5, 3, P, 1);
    let (_, record) = learn("cubic-5", &text, &with(1, true, false));
    let there = over(&text, P31);
    let mut cancelled = with(1, true, false);
    let cancel = Cancel::new();
    cancel.set();
    cancelled.cancel = Some(cancel);
    let mut late = with(1, true, false);
    late.deadline = Some(Duration::ZERO);
    let mut small = with(1, true, false);
    small.memory = Some(1);
    for (params, why) in [
        (cancelled, Stop::Cancel),
        (late, Stop::Deadline),
        (small, Stop::Memory),
    ] {
        let outcome = grobx::replay(&there, &params, &record, &mut NoTrace).unwrap();
        assert_eq!(outcome.stopped, Some(why));
        assert_eq!(outcome.basis, None);
    }
}

#[test]
fn a_replay_refuses_0_and_what_a_run_refuses() {
    let tiny = "x,y\n65521\nx^2+y,\nx*y+1\n";
    let (_, record) = learn("tiny", tiny, &params(1));
    // A replay is over GF(p) alone, and a run takes 0 since q-lift; 2^32 + 1
    // and 2^64 − 1 are composite (gfp-large-primes.10).
    for p in [0, 4_294_967_297, u64::MAX] {
        assert!(
            matches!(
                grobx::replay(&over(tiny, p), &params(1), &record, &mut NoTrace),
                Err(Error::Characteristic { .. })
            ),
            "over {p}"
        );
    }
    // `block:2` on two variables leaves no last block.
    let mut unfit = params(1);
    unfit.order = MonomialOrder::Block(vec![2]);
    assert!(matches!(
        grobx::replay(tiny, &unfit, &record, &mut NoTrace),
        Err(Error::Order { .. })
    ));
}

#[test]
fn signatures_do_not_matter_to_a_replay() {
    for (name, text) in small_instances(2) {
        for learned_with in [false, true] {
            let (_, record) = learn(&name, &text, &with(1, learned_with, false));
            let there = over(&text, P31);
            let on = grobx::replay(&there, &with(1, true, false), &record, &mut NoTrace);
            let off = grobx::replay(&there, &with(1, false, false), &record, &mut NoTrace);
            assert_eq!(on, off, "{name}, learned with signatures {learned_with}");
        }
    }
}
