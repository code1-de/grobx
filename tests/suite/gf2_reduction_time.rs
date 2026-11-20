//! gf2-reduction-time, the item's test code: the edge cases of
//! `OBJECTIVES.md` over GF(2), the characteristic 2 as written, generated
//! boolean systems against the checker, a deadline, and the round lines'
//! `packed` (`iterations/0125-gf2-reduction-time/plan.md`, _Test code_, 1
//! to 4 and 11).

use std::time::{Duration, Instant};

use grobx::Stop;
use grobx_checker::Mode;

use crate::common::systems::{boolean, random_sparse};
use crate::common::{P, assert_accepted, params, run, with_bases, with_cofactors};
use crate::orders::assert_canonical;
use crate::report::{of, report};

/// `EDGE_GENERATORS` with their bases over GF(2) in grevlex.
pub(crate) const EDGE: [(&str, &str, &str); 7] = with_bases([
    "[]:\n",
    "[]:\n",
    "[]:\n",
    "[1]:\n",
    "[x+z,\nz^2+1]:\n",
    "[y^2+x,\nx*y+z,\nx^2+y*z]:\n",
    "[x,\nz^2+y,\ny*z+1,\ny^2+z]:\n",
]);

/// The parameters of a call under `seed`, with cofactors where `cofactors`
/// holds, signatures as `signatures` says.
fn params_with(seed: u64, signatures: bool, cofactors: bool) -> grobx::Params {
    let mut p = if cofactors {
        with_cofactors(seed)
    } else {
        params(seed)
    };
    p.signatures = signatures;
    p
}

/// Test 1: the edge cases over GF(2), signatures on and off, with and
/// without cofactors: each basis canonical and as written, and each with
/// cofactors accepted by the checker in the exact mode.
#[test]
fn the_edge_cases_over_gf2() {
    for (case, generators, want) in EDGE {
        let text = format!("x,y,z\n2\n{generators}\n");
        for signatures in [true, false] {
            for cofactors in [true, false] {
                let outcome = run(&text, &params_with(0x9f2a, signatures, cofactors));
                let basis = outcome.basis.as_deref().unwrap();
                let name = format!("{case}, signatures {signatures}, cofactors {cofactors}");
                assert_canonical(&name, &text, basis, "grevlex");
                assert_eq!(basis, want, "{name}");
                if cofactors {
                    assert_accepted(&name, &text, &outcome, Mode::Exact);
                }
            }
        }
    }
}

/// Test 2: the characteristic written 2, 02, and 0002: one basis, and the
/// call report's read line and every run line name `field=gf2`.
#[test]
fn a_characteristic_of_two() {
    let mut first: Option<String> = None;
    for written in ["2", "02", "0002"] {
        let text = format!("x,y,z\n{written}\nx*y+z,\ny^2+x+1,\nz^2+y\n");
        let mut p = params(0x9f2b);
        p.report = true;
        let outcome = run(&text, &p);
        let lines = report(&outcome);
        let read = of(&lines, "read");
        assert_eq!(read.len(), 1, "{written}: one read line");
        assert_eq!(read[0].value("field"), "gf2", "{written}");
        let runs = of(&lines, "run");
        assert!(!runs.is_empty(), "{written}: a run line");
        for line in runs {
            assert_eq!(line.value("field"), "gf2", "{written}: {line:?}");
        }
        let basis = outcome.basis.unwrap();
        match &first {
            None => first = Some(basis),
            Some(b) => assert_eq!(&basis, b, "{written}"),
        }
    }
}

/// The boolean systems of test 3: variables, generators, and seed.
fn systems() -> Vec<(usize, usize, u64)> {
    (6..=10)
        .flat_map(|n| {
            let s = 0x9f30 + n as u64;
            [(n, 2 * n, s), (n, n + 2, s + 0x100)]
        })
        .collect()
}

/// Test 3: the boolean systems over GF(2): with cofactors, signatures on
/// and off, accepted by the checker in the exact mode; without cofactors,
/// whose rounds reduce in packed rows, signatures on and off, the basis of
/// the calls with them.
#[test]
fn boolean_systems() {
    for (n, m, seed) in systems() {
        let text = boolean(n, m, seed);
        let name = format!("boolean({n}, {m}, {seed:#x})");
        let mut bases = Vec::new();
        for signatures in [true, false] {
            for cofactors in [true, false] {
                let outcome = run(&text, &params_with(seed, signatures, cofactors));
                if cofactors {
                    let case = format!("{name}, signatures {signatures}");
                    assert_accepted(&case, &text, &outcome, Mode::Exact);
                }
                bases.push(outcome.basis.unwrap());
            }
        }
        assert!(
            bases.windows(2).all(|w| w[0] == w[1]),
            "{name}: the bases differ: {bases:?}"
        );
    }
}

/// The deadline of test 4.
const DEADLINE: Duration = Duration::from_millis(5);

/// How far past the deadline a call of test 4 may return.
const LATE: Duration = Duration::from_millis(250);

/// Test 4: a boolean system over GF(2) called with a deadline of 5 ms: it
/// stops with `Deadline`, no basis, and returns within `LATE` past the
/// deadline.
#[test]
fn a_deadline_over_gf2() {
    let text = boolean(24, 48, 0x9f50);
    let mut p = params(0x9f50);
    p.deadline = Some(DEADLINE);
    let start = Instant::now();
    let outcome = grobx::groebner(&text, &p).expect("the call runs");
    let took = start.elapsed();
    assert_eq!(outcome.stopped, Some(Stop::Deadline));
    assert_eq!(outcome.basis, None);
    assert!(
        took < DEADLINE + LATE,
        "returned {late:?} past the deadline",
        late = took.saturating_sub(DEADLINE)
    );
}

/// The most words of the K columns a round reduces in packed rows at,
/// among the values gf2-reduction-time.13 weighs.
const PACKED_WORDS_MOST: u64 = 512;

/// Test 11: a call over GF(2) names `packed=<words>` on the round line of
/// each round that reduced in packed rows, at most `PACKED_WORDS_MOST`,
/// and `packed=no` on the others, and one round line at least names its
/// words; a call over GF(65521) names `packed=no` on every round line.
#[test]
fn the_round_lines_name_packed() {
    let mut p = params(0x9f60);
    p.report = true;
    // A round of this system passes the bounds of gf2-reduction-time.21 and
    // .22, which the rounds of boolean(8, 16, 0x9f60) lie below (log 0125-4),
    // and those of boolean(17, 21, 0x9f63) since the plain run enters by
    // increasing leading monomial over GF(2) (gf2-update-time.1, log 0130-2).
    let lines = report(&run(&boolean(18, 22, 0x9f64), &p));
    let rounds = of(&lines, "round");
    assert!(!rounds.is_empty(), "a round line");
    let mut packed = 0;
    for line in &rounds {
        let value = line.value("packed");
        if value == "no" {
            continue;
        }
        let words: u64 = value
            .parse()
            .unwrap_or_else(|_| panic!("packed={value} on {line:?}"));
        assert!(words <= PACKED_WORDS_MOST, "{line:?}");
        packed += 1;
    }
    assert!(packed > 0, "no round reduced in packed rows: {rounds:?}");
    let lines = report(&run(&random_sparse(6, P, 0x9f61), &p));
    for line in of(&lines, "round") {
        assert_eq!(line.value("packed"), "no", "{line:?}");
    }
}
