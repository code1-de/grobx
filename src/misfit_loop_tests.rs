//! lift-misfit-loop, the item's test code: the records the lift drops in
//! a call, by why, up to `DROPS`, and the error at the last
//! (`iterations/0069-lift-misfit-loop/plan.md`, _Test code_, 1 to 5).
//! Unlucky primes forced first make each record drop.

use std::num::NonZeroUsize;
use std::time::Duration;

use num_bigint::BigUint;

use crate::lift::{Learns, Reconstruct, Replays, Report, lift_text, lift_text_as};
use crate::trace::NoTrace;
use crate::{Error, MonomialOrder, Outcome, Params, Threads};

/// Primes near 2^30.5 that the cases make unlucky: `lift_tests.rs`'s U1
/// and U2, and the next seven primes past U2.
const U: [u64; 9] = [
    1_518_500_279,
    1_518_500_293,
    1_518_500_309,
    1_518_500_327,
    1_518_500_419,
    1_518_500_429,
    1_518_500_443,
    1_518_500_449,
    1_518_500_471,
];
/// Primes below 2^31 from the top down, lucky for every case here.
const L: [u64; 4] = [2_147_483_647, 2_147_483_629, 2_147_483_587, 2_147_483_579];
/// Primes past 2^61 + 2^60, lucky for every case here: `later_runs_tests.rs`'s
/// W1, W2, and W3, and the next five primes past W3.
const W: [u64; 8] = [
    3_458_764_513_820_540_933,
    3_458_764_513_820_541_017,
    3_458_764_513_820_541_041,
    3_458_764_513_820_541_091,
    3_458_764_513_820_541_137,
    3_458_764_513_820_541_187,
    3_458_764_513_820_541_217,
    3_458_764_513_820_541_251,
];

/// The drops at which a call returns `Error::Drops`: `DROPS` of
/// lift-misfit-loop.4, written out here as the test code's oracle.
const BOUND: u32 = 4;
/// `BOUND` as a count of rounds.
const ROUNDS: usize = BOUND as usize;

/// Every lift's deadline, far past what each takes: a lift that learns
/// afresh without end stops there, and the test fails.
const DEADLINE: Duration = Duration::from_secs(60);

fn params(cofactors: bool) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    p.deadline = Some(DEADLINE);
    p.cofactors = cofactors;
    p
}

/// The product of `primes`.
fn product(primes: &[u64]) -> BigUint {
    primes.iter().map(|&p| BigUint::from(p)).product()
}

/// x + y and x + (c + 1)·y over Q, whose reduced basis is y, x. At a prime
/// that divides c the two span x + y alone: a learn run there records the
/// echelon x + y, 0, and a replay at a prime that divides no factor of c
/// misfits at its inputs; a verification there finds the leading monomial
/// x alone.
fn span(c: &BigUint) -> String {
    format!("x,y\n0\nx+y,\nx+{coef}*y\n", coef = c + 1u32)
}

/// c·x + y and x^2 − 1 over Q, `lift_tests.rs`'s case of an unlucky learn
/// prime: at a prime that divides c the first generator's leading monomial
/// is y, and a replay at a prime that divides no factor of c misfits at its
/// inputs, with cofactors and without.
fn lead(c: &BigUint) -> String {
    format!("x,y\n0\n{c}*x+y,\nx^2-1\n")
}

/// The lift of `text` with `forced` first, learning as `learns` says.
fn lift(
    text: &str,
    params: &Params,
    forced: &[u64],
    learns: Learns,
    report: &mut Report,
) -> Result<Outcome, Error> {
    match learns {
        Learns::Alone => lift_text(text, params, forced.to_vec(), report),
        Learns::Pairs => lift_text_as(
            text,
            params,
            forced.to_vec(),
            Reconstruct::Chain,
            Replays::Pairs,
            Learns::Pairs,
            &mut NoTrace,
            report,
        ),
    }
}

/// What a lift returns where it drops records by why.
fn dropped(misfits: u32, leads: u32, verification: u32) -> Result<Outcome, Error> {
    Err(Error::Drops {
        misfits,
        leads,
        verification,
    })
}

/// The basis of a lift that returns one, which a bound did not stop.
fn basis(case: &str, got: Result<Outcome, Error>, report: &Report) -> String {
    match got {
        Ok(Outcome {
            basis: Some(basis),
            stopped: None,
            ..
        }) => basis,
        other @ (Ok(_) | Err(_)) => panic!("{case}: no basis: {other:?}\n{report:?}"),
    }
}

/// The primes a learn run takes.
fn width(learns: Learns) -> usize {
    match learns {
        Learns::Alone => 1,
        Learns::Pairs => 2,
    }
}

/// The forced primes of `rounds` records, each learned at the primes of
/// `U` a learn run takes, and dropped after its replays misfit at two
/// primes of `W`.
fn misfit_rounds(rounds: usize, learns: Learns) -> Vec<u64> {
    let width = width(learns);
    (0..rounds)
        .flat_map(|k| {
            let learn = &U[k * width..(k + 1) * width];
            learn
                .iter()
                .chain(&W[2 * k..2 * k + 2])
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

// --- Tests 1 and 2: drops at misfits ---

/// Test 1, lift-misfit-loop.1 and .4: a record learned at an unlucky prime
/// misfits at two replays and drops. Up to `BOUND` − 1 drops, the lift
/// learns afresh at a drawn prime and returns the basis; at `BOUND` drops
/// it returns `Error::Drops` with every drop at misfits. Learn runs alone
/// and in pairs, the replays in pairs.
#[test]
fn drops_at_misfits() {
    for learns in [Learns::Alone, Learns::Pairs] {
        let width = width(learns);
        let text = span(&product(&U[..ROUNDS * width]));
        for rounds in 0..=ROUNDS {
            let case = format!("{learns:?}, {rounds} rounds");
            let forced = misfit_rounds(rounds, learns);
            let mut report = Report::default();
            let got = lift(&text, &params(false), &forced, learns, &mut report);
            let learned: Vec<u64> = (0..rounds).map(|k| U[k * width]).collect();
            assert_eq!(report.misfit, W[..2 * rounds], "{case}: {report:?}");
            if rounds < ROUNDS {
                assert_eq!(basis(&case, got, &report), "[y,\nx]:\n", "{case}");
                assert_eq!(report.learned[..rounds], learned, "{case}: {report:?}");
                assert_eq!(report.learned.len(), rounds + 1, "{case}: {report:?}");
            } else {
                assert_eq!(got, dropped(BOUND, 0, 0), "{case}: {report:?}");
                assert_eq!(report.learned, learned, "{case}: {report:?}");
                assert!(report.combined.is_empty(), "{case}: {report:?}");
            }
        }
    }
}

/// Test 2, lift-misfit-loop.1: the replays of a call with cofactors track
/// them one prime at a time, and the learn runs take no echelon. Records
/// learned at unlucky primes drop at misfits as without cofactors, and the
/// drop at `BOUND` returns `Error::Drops`.
#[test]
fn drops_at_misfits_with_cofactors() {
    let text = lead(&product(&U[..ROUNDS]));
    for rounds in [ROUNDS - 1, ROUNDS] {
        let case = format!("{rounds} rounds");
        let forced = misfit_rounds(rounds, Learns::Alone);
        let mut report = Report::default();
        let got = lift(&text, &params(true), &forced, Learns::Alone, &mut report);
        assert_eq!(report.misfit, W[..2 * rounds], "{case}: {report:?}");
        if rounds < ROUNDS {
            let got = got.unwrap_or_else(|e| panic!("{case}: {e:?}\n{report:?}"));
            assert!(
                got.basis.is_some() && got.cofactors.is_some(),
                "{case}: {got:?}"
            );
        } else {
            assert_eq!(got, dropped(BOUND, 0, 0), "{case}: {report:?}");
        }
    }
}

// --- Tests 3 and 4: drops at the verification, and of both whys ---

/// The forced primes of `rounds` records, each learned at a lucky prime of
/// `L`, complete after two replays at primes of `W`, and dropped at a
/// verification at a prime of `U` whose leading monomials differ.
fn verification_rounds(rounds: usize) -> Vec<u64> {
    (0..rounds)
        .flat_map(|k| [L[k], W[2 * k], W[2 * k + 1], U[k]])
        .collect()
}

/// Test 3, lift-misfit-loop.2: a verification whose leading monomials
/// differ drops the record, and drops at the verification count toward the
/// bound as drops at misfits do. Learn runs alone, whose verification is a
/// run at the next prime.
#[test]
fn drops_at_the_verification() {
    let text = span(&product(&U[..ROUNDS]));
    for rounds in [ROUNDS - 1, ROUNDS] {
        let case = format!("{rounds} rounds");
        let forced = verification_rounds(rounds);
        let mut report = Report::default();
        let got = lift(&text, &params(false), &forced, Learns::Alone, &mut report);
        let failed: Vec<(u64, bool)> = U[..rounds].iter().map(|&u| (u, false)).collect();
        assert_eq!(report.verified[..rounds], failed, "{case}: {report:?}");
        assert!(report.misfit.is_empty(), "{case}: {report:?}");
        if rounds < ROUNDS {
            assert_eq!(basis(&case, got, &report), "[y,\nx]:\n", "{case}");
        } else {
            assert_eq!(got, dropped(0, BOUND, 0), "{case}: {report:?}");
            assert_eq!(report.learned, L[..ROUNDS], "{case}: {report:?}");
        }
    }
}

/// Test 4, lift-misfit-loop.2 and .3: drops of both whys in one call count
/// together, a drop at misfits after a record whose replays were combined
/// among them. Two at misfits and two at the verification, alternating,
/// return `Error::Drops` with two of each.
#[test]
fn drops_of_both_whys_count_together() {
    let text = span(&product(&U[..ROUNDS]));
    let forced = [
        vec![U[0], W[0], W[1]],
        vec![L[0], W[2], W[3], U[1]],
        vec![U[2], W[4], W[5]],
        vec![L[1], W[6], W[7], U[3]],
    ]
    .concat();
    let mut report = Report::default();
    let got = lift(&text, &params(false), &forced, Learns::Alone, &mut report);
    assert_eq!(got, dropped(2, 2, 0), "{report:?}");
    assert_eq!(report.learned, [U[0], L[0], U[2], L[1]], "{report:?}");
    assert_eq!(report.misfit, [W[0], W[1], W[4], W[5]], "{report:?}");
    assert_eq!(
        report.verified,
        [(U[1], false), (U[3], false)],
        "{report:?}"
    );
}

// --- Test 5: the edge cases ---

/// Test 5: the edge cases of `OBJECTIVES.md` on x, y, z over Q, in
/// grevlex, with and without cofactors, learn runs alone and in pairs, at
/// drawn primes: each returns its basis, learns at most one record, and
/// drops none.
#[test]
fn the_edge_cases_drop_nothing() {
    let cases = [
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
    for (case, generators) in cases {
        let text = format!("x,y,z\n0\n{generators}\n");
        for cofactors in [false, true] {
            for learns in [Learns::Alone, Learns::Pairs] {
                let name = format!("{case}, cofactors {cofactors}, {learns:?}");
                let mut report = Report::default();
                let got = lift(&text, &params(cofactors), &[], learns, &mut report);
                basis(&name, got, &report);
                assert!(report.learned.len() <= 1, "{name}: {report:?}");
                assert!(report.misfit.is_empty(), "{name}: {report:?}");
                assert!(
                    report.verified.iter().all(|&(_, passed)| passed),
                    "{name}: {report:?}"
                );
            }
        }
    }
}
