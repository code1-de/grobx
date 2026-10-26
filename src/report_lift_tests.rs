//! call-report, sub-item 2, the crate's test code: the lift's lines over ℚ
//! with primes forced where a path of the lift needs an unlucky one, as
//! `lift_tests.rs` forces them, against the lift's own `lift::Report`
//! (`iterations/0064-call-report/plan.md`, _Test code_, 16 and 17).

use std::num::NonZeroUsize;

use crate::lift::{Learns, Reconstruct, Replays, Report, lift_text_as};
use crate::report_tests::{Line, lines};
use crate::trace::NoTrace;
use crate::{MonomialOrder, Outcome, Params, Threads};

/// Two primes near 2^30.5 that the test code makes unlucky.
const U1: u64 = 1_518_500_279;
const U2: u64 = 1_518_500_293;
/// Primes near 2^31, lucky for every instance here.
const L1: u64 = 2_147_483_647;
const L2: u64 = 2_147_483_629;
const L3: u64 = 2_147_483_587;

fn params(seed: u64) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    p.report = true;
    p
}

/// The lift of `text` with the primes `forced` first, learning as `learns`
/// says: its outcome, which must hold a basis, its report's lines, and the
/// lift's `Report`.
fn lift(text: &str, forced: &[u64], learns: Learns) -> (Outcome, Vec<Line>, Report) {
    let mut report = Report::default();
    let outcome = lift_text_as(
        text,
        &params(1),
        forced.to_vec(),
        Reconstruct::Chain,
        Replays::Pairs,
        learns,
        &mut NoTrace,
        &mut report,
    )
    .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(outcome.basis.is_some(), "the lift stopped: {outcome:?}");
    let text = outcome.report.as_deref().expect("the lift's report");
    let read = lines(text);
    assert_eq!(read[0].kind, "call", "{text}");
    assert!(read[1].is("read", &[("field", "q")]), "{text}");
    assert!(read[2].is("lift", &[]), "{text}");
    assert_eq!(read.last().unwrap().kind, "end", "{text}");
    agrees(&read, &report, text);
    (outcome, read, report)
}

/// The primes of a line's `primes` field.
fn primes(line: &Line) -> Vec<u64> {
    line.get("primes")
        .expect("primes")
        .split(',')
        .map(|p| p.parse().unwrap())
        .collect()
}

fn prime(line: &Line) -> u64 {
    line.get("prime").expect("prime").parse().unwrap()
}

/// The report's lines name the primes of the lift's `Report`, each list in
/// its order: the learns recorded, the pair learns unpaired, the primes
/// combined, those that did not fit, those skipped at a learn, a replay,
/// or a verification, and the verifications with their verdicts. Every
/// prime's run is a `run` line whose role says what the run was for.
fn agrees(read: &[Line], report: &Report, text: &str) {
    let mut learned = Vec::new();
    let mut unpaired = Vec::new();
    let mut combined = Vec::new();
    let mut misfit = Vec::new();
    let mut skipped = Vec::new();
    let mut verified = Vec::new();
    for line in read {
        match line.kind.as_str() {
            "learn" => match line.get("outcome") {
                Some("recorded") => learned.push(primes(line)[0]),
                Some("unpaired") => {
                    let p = primes(line);
                    assert_eq!(p.len(), 2, "{line:?}");
                    unpaired.push((p[0], p[1]));
                }
                Some("skipped") => skipped.push(primes(line)[0]),
                other @ (Some(_) | None) => panic!("a learn's outcome {other:?}: {line:?}"),
            },
            "combine" => combined.push(prime(line)),
            "misfit" => misfit.push(prime(line)),
            "skip" => skipped.push(prime(line)),
            "verify" => verified.push((prime(line), line.get("outcome") == Some("passed"))),
            "run" => {
                let role = line.get("role").unwrap();
                assert!(
                    matches!(role, "learn" | "replay" | "verify"),
                    "a run of the lift as {role}: {line:?}"
                );
            }
            _ => {}
        }
    }
    assert_eq!(learned, report.learned, "learned\n{text}");
    assert_eq!(unpaired, report.unpaired, "unpaired\n{text}");
    assert_eq!(combined, report.combined, "combined\n{text}");
    assert_eq!(misfit, report.misfit, "misfit\n{text}");
    assert_eq!(skipped, report.skipped, "skipped\n{text}");
    assert_eq!(verified, report.verified, "verified\n{text}");
}

/// The kinds of the lift's lines, in order, leaving out the runs' own.
fn lift_kinds(read: &[Line]) -> Vec<&str> {
    const RUN: [&str; 8] = [
        "run", "inputs", "echelon", "round", "handover", "constant", "inter", "ran",
    ];
    read.iter()
        .map(|l| l.kind.as_str())
        .filter(|k| !RUN.contains(k) && *k != "image")
        .collect()
}

/// Test 16, call-report.14: over U1 the first generator's leading monomial
/// is y, every replay at a lucky prime misfits at its inputs, and two
/// misfits in a row drop the record: the report names both misfits, the
/// drop, and the second learn. The lifts of `lift_tests.rs`'s other unlucky
/// cases agree with the lift's `Report` too: a verification whose leading
/// monomials differ drops the record, a replay prime whose row reduces to
/// zero misfits, and a verification prime of a denominator is skipped.
#[test]
fn misfits_and_drops() {
    let text = format!("x,y\n0\n{U1}*x+y,\nx^2-1\n");
    let (_, read, report) = lift(&text, &[U1], Learns::Alone);
    assert_eq!(report.misfit.len(), 2, "{report:?}");
    let kinds = lift_kinds(&read);
    let first = kinds.iter().position(|&k| k == "misfit").unwrap();
    let drop = kinds.iter().position(|&k| k == "drop").expect("a drop");
    assert!(first < drop, "{kinds:?}");
    assert_eq!(
        kinds[first..=drop]
            .iter()
            .filter(|&&k| k == "misfit")
            .count(),
        2,
        "{kinds:?}"
    );
    assert_eq!(
        kinds[drop - 1],
        "misfit",
        "the drop follows the second misfit: {kinds:?}"
    );
    assert!(
        read.iter()
            .filter(|l| l.kind == "drop")
            .all(|l| l.is("drop", &[("why", "misfits")])),
        "{read:?}"
    );
    assert!(
        read.iter()
            .filter(|l| l.kind == "misfit")
            .all(|l| l.is("misfit", &[("why", "record")])),
        "{read:?}"
    );
    let learns: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|&(_, &k)| k == "learn")
        .map(|(i, _)| i)
        .collect();
    assert!(learns.len() >= 2 && learns[1] > drop, "{kinds:?}");

    let n = U1 * U2;
    let (_, read, report) = lift(&format!("x,y\n0\nx+{n},\nx*y\n"), &[U1, U2], Learns::Alone);
    let differ = read
        .iter()
        .position(|l| l.is("verify", &[("outcome", "leads-differ")]))
        .unwrap_or_else(|| panic!("{report:?}"));
    assert!(
        read[differ + 1..]
            .iter()
            .find(|l| l.kind == "drop")
            .is_some_and(|l| l.is("drop", &[("why", "leads")])),
        "{read:?}"
    );

    let (_, read, _) = lift(&format!("x,y\n0\nx+{n},\nx*y\n"), &[L1, U1], Learns::Alone);
    assert!(
        read.iter()
            .any(|l| l.is("misfit", &[("prime", &U1.to_string())])),
        "{read:?}"
    );

    let (_, read, _) = lift("x\n0\n3*x-1\n", &[L1, L2, L3, 3], Learns::Alone);
    assert!(
        read.iter()
            .any(|l| l.is("skip", &[("prime", "3"), ("why", "lifted-denominator")])),
        "{read:?}"
    );
    assert!(
        read.iter()
            .any(|l| l.is("verify", &[("outcome", "passed")]))
    );
}

/// Test 17, call-report.14: a denominator that a forced prime divides. A
/// learn alone at U1 is skipped, a pair learn at (U1, L1) falls back
/// unpaired, and a replay at U1 runs alone, out of the pair's range, and
/// is skipped.
#[test]
fn denominators() {
    let text = format!("x,y\n0\nx-1/{U1},\ny^2-x\n");
    let (outcome, read, _) = lift(&text, &[U1], Learns::Alone);
    assert_eq!(
        outcome.basis.as_deref(),
        Some(format!("[{U1}*x-1,\n{U1}*y^2-1]:\n").as_str())
    );
    let u1 = U1.to_string();
    assert!(
        read.iter().any(|l| l.is(
            "learn",
            &[
                ("primes", &u1),
                ("outcome", "skipped"),
                ("why", "denominator")
            ]
        )),
        "{read:?}"
    );
    assert!(
        read.iter()
            .any(|l| l.is("ran", &[("outcome", "denominator"), ("at", "read")])),
        "the learn's run ends at its read: {read:?}"
    );

    let (_, read, report) = lift(&text, &[U1, L1], Learns::Pairs);
    assert_eq!(report.unpaired, vec![(U1, L1)], "{report:?}");
    let pair = format!("{U1},{L1}");
    assert!(
        read.iter().any(|l| l.is(
            "learn",
            &[
                ("primes", &pair),
                ("outcome", "unpaired"),
                ("why", "denominator")
            ]
        )),
        "{read:?}"
    );

    let (_, read, _) = lift(&text, &[L1, U1], Learns::Alone);
    assert!(
        read.iter()
            .any(|l| l.is("pair", &[("outcome", "alone"), ("why", "range")])
                && l.get("primes")
                    .is_some_and(|p| p.starts_with(&format!("{U1},")))),
        "{read:?}"
    );
    assert!(
        read.iter()
            .any(|l| l.is("skip", &[("prime", &u1), ("why", "denominator")])),
        "{read:?}"
    );
}
