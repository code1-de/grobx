//! signature-pruning, the item's test code: the run with signatures against
//! the run without, the checker on the run with signatures, and the trace
//! of the run with signatures (plan.md, _Test code_).

use std::num::NonZeroU32;

use grobx::id::RowId;
use grobx::trace::{RowSource, Trace};
use grobx::{Outcome, Params};
use grobx_checker::Mode;

use crate::common::{
    P, PRIMES, assert_accepted, degree_bound, dense, params, points, random_system, run,
};

/// The parameters of a measured call, with or without signatures and
/// cofactors.
fn with(seed: u64, signatures: bool, cofactors: bool) -> Params {
    let mut p = params(seed);
    p.signatures = signatures;
    p.cofactors = cofactors;
    p
}

/// The bases with signatures and without, which must be equal.
fn same_basis(name: &str, text: &str) -> String {
    let on = run(text, &with(1, true, false)).basis.unwrap();
    let off = run(text, &with(1, false, false)).basis.unwrap();
    assert_eq!(
        on, off,
        "{name}: the bases with and without signatures\n{text}"
    );
    on
}

/// The edge cases of plan.md and the hand cases of `engine.rs`: name,
/// instance, and the basis text where it is written here.
const EDGE: &[(&str, &str, Option<&str>)] = &[
    ("zero", "x,y\n65521\n0\n", Some("[]:\n")),
    ("zeros", "x,y\n7\n0,\n0\n", Some("[]:\n")),
    ("none", "x,y\n65521\n", Some("[]:\n")),
    ("x-plus-x", "x\n2\nx+x\n", Some("[]:\n")),
    ("unit", "x,y\n65521\nx*y-1,\nx\n", Some("[1]:\n")),
    ("unit-p2", "x\n2\nx,\nx+1\n", Some("[1]:\n")),
    ("unit-constant", "x,y\n65521\n5\n", Some("[1]:\n")),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
        Some("[1]:\n"),
    ),
    ("gb", "x,y\n65521\nx+y,\ny^2+1\n", Some("[x+y,\ny^2+1]:\n")),
    (
        "gb-swapped",
        "x,y\n65521\ny^2+1,\nx+y\n",
        Some("[x+y,\ny^2+1]:\n"),
    ),
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n", None),
    (
        "eliminable-first",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        None,
    ),
    (
        "eliminable-last",
        "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        None,
    ),
    (
        "eliminable-linear",
        "x,y,z\n65521\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n",
        None,
    ),
    (
        "tiny",
        "x,y\n65521\nx^2+y,\nx*y+1\n",
        Some("[y^2+65520*x,\nx*y+1,\nx^2+y]:\n"),
    ),
    (
        "duplicates",
        "x,y\n65521\nx^2+y,\nx^2+y,\nx*y+1\n",
        Some("[y^2+65520*x,\nx*y+1,\nx^2+y]:\n"),
    ),
    (
        "multiples",
        "x,y\n65521\n2*x^2+2*y,\nx*y+1,\n3*x*y+3\n",
        Some("[y^2+65520*x,\nx*y+1,\nx^2+y]:\n"),
    ),
    // The second input's leading monomial x^2·y is a multiple of the
    // first's, x·y: the second is its own S-pair's reductant.
    (
        "input-divides-input",
        "x,y,z\n65521\nx*y-z,\nx^2*y-y^2+1,\nz^2-x\n",
        None,
    ),
    ("gcd", "x\n65521\nx^3-1,\nx^2-1\n", Some("[x+65520]:\n")),
    ("gcd-p7", "x\n7\nx^4-1,\nx^6-1\n", Some("[x^2+6]:\n")),
    ("monic-power", "x\n65521\nx^5+x+1\n", Some("[x^5+x+1]:\n")),
    ("p2", "x,y,z\n2\nx*y+z,\nx*z+y,\ny*z+x\n", None),
    ("p3", "x,y,z\n3\nx^2+y*z+1,\ny^2+x*z,\nz^2+x*y+2\n", None),
    ("p7", "x,y,z\n7\nx^3-y*z,\ny^3-x*z,\nz^3-x*y\n", None),
    (
        "p31",
        "x,y,z\n2147483647\nx^2+123456789*y*z+1,\ny^2-x*z,\nz^2+x*y+2\n",
        None,
    ),
];

#[test]
fn the_edge_cases_give_one_basis_both_ways() {
    for &(name, text, want) in EDGE {
        let basis = same_basis(name, text);
        if let Some(want) = want {
            assert_eq!(basis, want, "{name}");
        }
    }
}

#[test]
fn a_reduced_basis_is_its_own_reduced_basis_both_ways() {
    let first = same_basis("cubic-3", &dense(3, 3, P, 1));
    let elements = first
        .trim_end()
        .trim_start_matches('[')
        .trim_end_matches("]:");
    let again = format!("x1,x2,x3\n65521\n{elements}\n");
    assert_eq!(same_basis("gb-cubic-3", &again), first);
}

/// The first six seeds of `engine.rs`'s random systems, over its five
/// primes.
#[test]
fn random_systems_give_one_basis_both_ways() {
    for p in PRIMES {
        for seed in 0..6 {
            same_basis(&format!("sig-random-p{p}-{seed}"), &random_system(p, seed));
        }
    }
}

/// The dense cubic systems on two to four variables over 65521.
fn dense_systems() -> Vec<(String, String)> {
    (2..=4)
        .map(|m| (format!("cubic-{m}"), dense(m, 3, P, m as u64)))
        .collect()
}

#[test]
fn dense_systems_give_one_basis_both_ways() {
    for (name, text) in dense_systems() {
        same_basis(&name, &text);
    }
}

/// The run with signatures and cofactors, accepted by the checker in the
/// exact mode.
fn accepted_exactly(name: &str, text: &str) {
    let outcome = run(text, &with(0x5197, true, true));
    assert_accepted(name, text, &outcome, Mode::Exact);
}

#[test]
fn the_checker_accepts_the_edge_cases_with_signatures() {
    for &(name, text, _) in EDGE {
        accepted_exactly(name, text);
    }
}

#[test]
fn the_checker_accepts_random_systems_with_signatures() {
    for p in PRIMES {
        for seed in 0..8 {
            accepted_exactly(&format!("random-p{p}-{seed}"), &random_system(p, seed));
        }
    }
}

/// Dense systems computed with signatures and cofactors, accepted by the
/// checker in the cheap mode, k as checker-cheap.16 chooses it.
#[test]
fn the_checker_accepts_dense_systems_with_signatures() {
    for (name, text) in dense_systems() {
        let outcome = run(&text, &with(0x5197, true, true));
        let k = points(P, degree_bound(&text, &outcome));
        let mode = Mode::Cheap {
            points: NonZeroU32::new(k).unwrap(),
            seed: 0x5197 ^ name.len() as u64,
        };
        assert_accepted(&name, &text, &outcome, mode);
    }
}

#[test]
fn signatures_are_on_by_default() {
    assert!(params(1).signatures);
}

#[test]
fn the_same_seed_gives_the_same_outcome_with_signatures() {
    let text = dense(4, 3, P, 1);
    let p = with(9, true, true);
    let first: Outcome = run(&text, &p);
    assert_eq!(first, run(&text, &p));
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Matrix(usize),
    Counted(Vec<RowId>),
}

#[derive(Default)]
struct Recorder(Vec<Event>);

impl Trace for Recorder {
    fn matrix(&mut self, rows: &[RowSource]) {
        self.0.push(Event::Matrix(rows.len()));
    }

    fn counted(&mut self, rows: &[RowId]) {
        self.0.push(Event::Counted(rows.to_vec()));
    }
}

#[test]
fn the_trace_hears_the_signature_run() {
    for text in [
        "x,y\n65521\nx^2+y,\nx*y+1\n".to_string(),
        dense(3, 3, P, 1),
        dense(4, 3, P, 2),
        random_system(7, 3),
    ] {
        let mut recorder = Recorder::default();
        let outcome = grobx::groebner_with(&text, &with(1, true, false), &mut recorder)
            .expect("the call runs");
        assert_eq!(outcome.basis, run(&text, &with(1, true, false)).basis);
        let events = recorder.0;
        assert!(
            events.len() >= 2 && events.len().is_multiple_of(2),
            "{events:?}"
        );
        for pair in events.chunks(2) {
            let (rows, counted) = match (&pair[0], &pair[1]) {
                (Event::Matrix(rows), Event::Counted(counted)) => (rows, counted),
                (Event::Matrix(_) | Event::Counted(_), Event::Matrix(_) | Event::Counted(_)) => {
                    panic!("matrix and counted do not alternate: {pair:?}")
                }
            };
            assert!(counted.windows(2).all(|w| w[0] < w[1]), "{counted:?}");
            assert!(counted.iter().all(|&r| r.index() < *rows), "{counted:?}");
        }
    }
}
