//! gf2-echelon-losses, the item's test code: over GF(2) the bounded run and
//! the plain run take their generators as read, and the signature run takes
//! them substituted and in reduced row echelon form, every result as it was;
//! over every other field every run takes its echelon
//! (`iterations/0173-gf2-echelon-losses/plan.md`, _Test code_, 1 to 3).

use crate::call::built;
use crate::gf2_input_time_tests::{EDGE, sparse_pivots};
use crate::gf2_start_over_tests::{call, reported};
use crate::report_tests::Line;
use crate::systems::{boolean, cubics, random_system};
use crate::{CheckerMode, MonomialOrder, Params};

/// The systems over GF(2) of tests 1 and 2, each with its name: boolean
/// systems of 6 to 10 variables, cubic ones, systems whose linear
/// generators the substitution takes, and the edge cases over x, y, z.
fn systems() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for n in 6..=10usize {
        let seed = 0xec40 + n as u64;
        let m = 2 * n;
        out.push((format!("boolean({n}, {m}, {seed:#x})"), boolean(n, m, seed)));
    }
    for seed in 1..=2u64 {
        out.push((format!("cubics(7, 9, {seed})"), cubics(7, 9, seed)));
    }
    for (seed, (n, k)) in (0xec50u64..).zip([(8, 2), (10, 3), (12, 5)]) {
        let q = 2 * n;
        out.push((
            format!("sparse_pivots({n}, {k}, {q}, 2, {seed:#x})"),
            sparse_pivots(n, k, q, 2, seed),
        ));
    }
    for (k, g) in EDGE.iter().enumerate() {
        out.push((format!("edge {k}"), format!("x,y,z\n2\n{g}\n")));
    }
    out
}

/// A report's segment per run: its `run` line and the lines up to the next.
fn segments(lines: &[Line]) -> Vec<&[Line]> {
    let starts: Vec<usize> = (0..lines.len())
        .filter(|&k| lines[k].kind == "run")
        .collect();
    let ends = starts.iter().skip(1).copied().chain([lines.len()]);
    starts
        .iter()
        .zip(ends)
        .map(|(&s, e)| &lines[s..e])
        .collect()
}

/// Whether `segment` holds a line of the kind `kind`.
fn holds(segment: &[Line], kind: &str) -> bool {
    segment.iter().any(|l| l.kind == kind)
}

/// Whether `segment` reads generators of which one is not zero: its `inputs`
/// line counts a nonzero generator.
fn reads(segment: &[Line]) -> bool {
    segment
        .iter()
        .filter(|l| l.kind == "inputs")
        .any(|l| l.get("nonzero").is_some_and(|k| k != "0"))
}

/// The basis of the call of `text` under `params` with cofactors and the
/// exact checker, which certifies it; None where the build takes no
/// cofactors.
fn certified(text: &str, params: &Params, name: &str) -> Option<String> {
    let mut params = params.clone();
    params.cofactors = true;
    params.checker = Some(CheckerMode::Exact);
    built(&params).ok()?;
    let outcome = crate::groebner(text, &params).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert_eq!(outcome.certified, Some(true), "{name}: certified");
    Some(outcome.basis.unwrap_or_else(|| panic!("{name}: a basis")))
}

/// Test 1, gf2-echelon-losses.1, .3, .4: calls over GF(2) in grevlex with
/// the signatures off and on, where the bounded run does not pass its bound,
/// and in deglex, lex, and `block:2` with the signatures on: no line of the
/// report is a `substitute` or an `echelon` line, and the basis is the one
/// the call with cofactors and the exact checker returns and the checker
/// certifies.
#[test]
fn plain_runs_over_gf2_take_their_generators_as_read() {
    let calls = [
        (MonomialOrder::Grevlex, false),
        (MonomialOrder::Grevlex, true),
        (MonomialOrder::Deglex, true),
        (MonomialOrder::Lex, true),
        (MonomialOrder::Block(vec![2]), true),
    ];
    let mut checked = 0;
    for (system, text) in systems() {
        for (order, signatures) in &calls {
            let name = format!("{system} in {order}, signatures {signatures}");
            let mut params = reported(1, *signatures);
            params.order = order.clone();
            let (basis, lines) = call(&text, &params, None);
            for line in &lines {
                assert!(
                    line.kind != "substitute" && line.kind != "echelon",
                    "{name}: a {kind} line",
                    kind = line.kind
                );
            }
            if let Some(want) = certified(&text, &params, &name) {
                assert_eq!(basis, want, "{name}: the basis");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no call was checked against the checker's");
}

/// Test 2, gf2-echelon-losses.1: calls over GF(2) in grevlex with
/// `WORK_FORCED` at `Some(0)`, so that the bounded run passes its bound and
/// the call starts over in the signature run: the segments of the bounded
/// run and of a plain run that starts over after a handover hold no
/// `substitute` and no `echelon` line; the segment of each signature run
/// that reads a nonzero generator holds an `echelon` line; and the basis is
/// the call's without signatures. Among the calls a signature run's
/// segment holds a `substitute` line before its `echelon` line, and a
/// signature run follows the bounded run, asserted present.
#[test]
fn the_signature_run_takes_its_echelon() {
    let (mut signature, mut substituted) = (0, 0);
    for (system, text) in systems() {
        let (want, _) = call(&text, &reported(1, false), None);
        let (basis, lines) = call(&text, &reported(1, true), Some(0));
        assert_eq!(basis, want, "{system}: the basis");
        for segment in segments(&lines) {
            let engine = segment[0].get("engine");
            let name = format!("{system}, the run {line:?}", line = segment[0].fields);
            if engine != Some("signature") {
                assert!(!holds(segment, "substitute"), "{name}: a substitute line");
                assert!(!holds(segment, "echelon"), "{name}: an echelon line");
                continue;
            }
            if !reads(segment) {
                continue;
            }
            signature += 1;
            let at = |kind: &str| segment.iter().position(|l| l.kind == kind);
            let echelon = at("echelon").unwrap_or_else(|| panic!("{name}: no echelon line"));
            if let Some(substitute) = at("substitute") {
                assert!(
                    substitute < echelon,
                    "{name}: the substitute line after the echelon"
                );
                substituted += 1;
            }
        }
    }
    assert!(signature > 0, "no signature run read its generators");
    assert!(substituted > 0, "no signature run substituted");
}

/// Test 3, gf2-echelon-losses.3: calls over GF(7), GF(65521), and ℚ on
/// systems with linear generators and on the edge cases over x, y, z, in
/// grevlex with the signatures off and on: the segment of every run that
/// reads a nonzero generator holds an `echelon` line, and the basis is the
/// call's with `Params::echelon` off.
#[test]
fn other_fields_keep_their_echelon() {
    let mut runs = 0;
    for p in [7u64, 65521, 0] {
        let mut texts: Vec<(String, String)> = EDGE
            .iter()
            .enumerate()
            .map(|(k, g)| (format!("edge {k} over {p}"), format!("x,y,z\n{p}\n{g}\n")))
            .collect();
        if p == 0 {
            for seed in 1..=4u64 {
                texts.push((format!("random_system(0, {seed})"), random_system(0, seed)));
            }
        } else {
            for (seed, (n, k)) in (0xec60u64..).zip([(8, 2), (10, 3)]) {
                let q = 2 * n;
                texts.push((
                    format!("sparse_pivots({n}, {k}, {q}, {p}, {seed:#x})"),
                    sparse_pivots(n, k, q, p, seed),
                ));
            }
        }
        for (system, text) in texts {
            for signatures in [false, true] {
                let name = format!("{system}, signatures {signatures}");
                let params = reported(1, signatures);
                let (basis, lines) = call(&text, &params, None);
                let mut off = params.clone();
                off.echelon = false;
                let (want, _) = call(&text, &off, None);
                assert_eq!(basis, want, "{name}: the basis");
                for segment in segments(&lines) {
                    if reads(segment) {
                        runs += 1;
                        assert!(
                            holds(segment, "echelon"),
                            "{name}: the run {line:?} without an echelon line",
                            line = segment[0].fields
                        );
                    }
                }
            }
        }
    }
    assert!(runs > 0, "no run read its generators");
}
