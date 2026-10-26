//! zero-rows-time, the item's test code: learns whose plain run rules rows
//! zero by tests, against learns whose trace listens and so reduce each
//! row on its own; their records replayed; calls over ℚ, whose pair learn
//! takes the tests; and the edge cases
//! (`iterations/0078-zero-rows-time/plan.md`, _Test code_, 8 to 11).

use grobx::trace::{NoTrace, Record};
use grobx::{Outcome, Params};
use grobx_checker::Mode;

use crate::common::{
    Listening, P, P31, assert_accepted, params, run,
    systems::{random_binomials, random_sparse},
    with_cofactors,
};
use crate::report::{of, report as lines_of};

/// 2^61 − 1, a prime past 2^31, whose runs take `Gf64`.
const P61: u64 = (1 << 61) - 1;

/// `params` with signatures on or off.
fn with(seed: u64, signatures: bool) -> Params {
    let mut p = params(seed);
    p.signatures = signatures;
    p
}

/// `text`'s learn with the trace `T`: its outcome and record.
fn learn<T: grobx::trace::Trace>(text: &str, params: &Params, trace: &mut T) -> (Outcome, Record) {
    let (outcome, record) =
        grobx::learn(text, params, trace).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    (
        outcome,
        record.expect("a learn that returns a basis has a record"),
    )
}

/// The edge cases of `OBJECTIVES.md` on x, y, z, generators alone.
const EDGE: [(&str, &str); 7] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// Test 8: learn-zero-time.1, .4, .5. `grobx::learn` with `NoTrace`, whose
/// plain run takes the tests, against the learn with a listening trace,
/// which reduces each row on its own: random sparse systems on six
/// variables over 7, 65521, 2^31 − 1, and 2^61 − 1, signatures off and on.
/// The same basis, the same record, and the same rounds; over the cases
/// the learns with `NoTrace` reduce fewer rows.
#[test]
fn a_learn_records_what_rows_alone_record() {
    let (mut tested, mut alone) = (0, 0);
    for p in [7, P, P31, P61] {
        for seed in 0..3 {
            let text = random_sparse(6, p, seed);
            for signatures in [false, true] {
                let name = format!("sparse-6 over {p}, seed {seed}, signatures {signatures}");
                let params = with(0x7e1 ^ seed, signatures);
                let (a, ra) = learn(&text, &params, &mut NoTrace);
                let (b, rb) = learn(&text, &params, &mut Listening);
                assert_eq!(a.basis, b.basis, "{name}");
                assert!(ra == rb, "{name}: the records differ");
                assert_eq!(a.stats.rounds, b.stats.rounds, "{name}");
                tested += a.stats.reduced;
                alone += b.stats.reduced;
            }
        }
    }
    assert!(
        tested < alone,
        "{tested} reductions with the tests, {alone} alone"
    );
}

/// Test 9: a record a learn with the tests made replays. Random sparse
/// systems on six variables learned over 65521 with `NoTrace`, signatures
/// off and on, replayed with the characteristic set to 2^31 − 1: the basis
/// `groebner` returns there.
#[test]
fn a_record_learned_with_tests_replays() {
    for seed in 0..3 {
        let text = random_sparse(6, P, seed);
        let there = text.replacen(&format!("\n{P}\n"), &format!("\n{P31}\n"), 1);
        for signatures in [false, true] {
            let name = format!("sparse-6, seed {seed}, signatures {signatures}");
            let params = with(0x7e9 ^ seed, signatures);
            let (_, record) = learn(&text, &params, &mut NoTrace);
            let replayed = grobx::replay(&there, &params, &record, &mut NoTrace)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(replayed.basis, run(&there, &params).basis, "{name}");
        }
    }
}

/// Test 10: learn-zero-time.1, .6, q-zero-work-time.5. Random binomial
/// systems over ℚ and a random sparse one on five variables, signatures
/// off and on, under three seeds: the call returns the basis of the call
/// whose trace listens, whose pair learn reduces each row on its own, and
/// of the call with cofactors, which the checker accepts in the exact
/// mode. With signatures off the pair learn's first run is its plain run,
/// which the report names `reduce=tests`. Over the cases the calls whose
/// trace listens reduce more rows.
#[test]
fn calls_over_q_whose_learn_takes_tests() {
    let mut systems: Vec<(String, String)> = (0..5)
        .map(|seed| (format!("binomials-{seed}"), random_binomials(0, seed)))
        .collect();
    systems.push(("sparse-5".to_string(), random_sparse(5, 0, 3)));
    let (mut tested, mut alone) = (0, 0);
    for (system, text) in &systems {
        for signatures in [false, true] {
            for seed in 0..3u64 {
                let name = format!("{system} over ℚ, signatures {signatures}, seed {seed}");
                let mut plain = with(0x7ea ^ seed, signatures);
                plain.report = true;
                let outcome = run(text, &plain);
                let listened = grobx::groebner_with(text, &plain, &mut Listening)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert_eq!(outcome.basis, listened.basis, "{name}");
                tested += outcome.stats.reduced;
                alone += listened.stats.reduced;
                let mut tracked = with_cofactors(0x7ea ^ seed);
                tracked.signatures = signatures;
                let reference = run(text, &tracked);
                assert_accepted(&name, text, &reference, Mode::Exact);
                assert_eq!(outcome.basis, reference.basis, "{name}");
                let lines = lines_of(&outcome);
                let first = &of(&lines, "run")[0];
                if !signatures {
                    assert!(
                        first.is(
                            "run",
                            &[("role", "learn"), ("engine", "plain"), ("reduce", "tests")]
                        ),
                        "{name}: {first:?}"
                    );
                }
            }
        }
    }
    assert!(
        tested < alone,
        "{tested} reductions with the tests, {alone} alone"
    );
}

/// Test 11: the edge cases of `OBJECTIVES.md`. Over ℚ, signatures off and
/// on: the call with cofactors that the checker accepts in the exact mode,
/// and the call without them returns the same basis. Over 65521 through
/// `grobx::learn`: with `NoTrace` and with a listening trace, the same
/// basis and the same record.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        for signatures in [false, true] {
            let name = format!("{case}, signatures {signatures}");
            let q = format!("x,y,z\n0\n{generators}\n");
            let mut tracked = with_cofactors(0x7eb);
            tracked.signatures = signatures;
            let reference = run(&q, &tracked);
            assert_accepted(&name, &q, &reference, Mode::Exact);
            assert_eq!(
                run(&q, &with(0x7eb, signatures)).basis,
                reference.basis,
                "{name}"
            );
            let gfp = format!("x,y,z\n{P}\n{generators}\n");
            let params = with(0x7ec, signatures);
            let (a, ra) = learn(&gfp, &params, &mut NoTrace);
            let (b, rb) = learn(&gfp, &params, &mut Listening);
            assert_eq!(a.basis, b.basis, "{name} over {P}");
            assert!(ra == rb, "{name} over {P}: the records differ");
        }
    }
}
