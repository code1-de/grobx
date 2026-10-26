//! checker-stop-latency, the checker's test code: the check counts its work
//! in units, about a term's work each, and asks its poll once per 4096
//! units, so that one step whose work is a long polynomial no longer passes
//! as one (checker-stop-latency.1 to .12). On texts built so that one step
//! of a reading, an expansion, an extraction, or an evaluation at a point
//! takes N terms, the check asks its poll at least once per 4096 terms it
//! reads and writes; a poll that answers true anywhere stops the check; a
//! reading error before a stop is the error; and the edge cases of
//! `OBJECTIVES.md` (plan.md, _Test code_, 1 to 4).

use std::num::NonZeroU32;

use grobx_checker::{Error, Flaw, Mode, Order, Verdict, check, check_polled};

/// The units between two calls of the poll (`Poll`).
const STEPS: usize = 4096;

/// The terms of the long polynomials.
const N: usize = 1 << 15;

/// The characteristics the cases are read over: GF(65521), GF(2^31 − 1),
/// and ℚ.
const FIELDS: [u64; 3] = [65521, (1 << 31) - 1, 0];

/// x^(n−1) + … + x + 1, its terms in decreasing order, each times `x^shift`
/// and `y` where `y` is set.
fn powers(n: usize, shift: usize, y: bool) -> String {
    let mut out = String::new();
    for i in (0..n).rev() {
        if !out.is_empty() {
            out.push('+');
        }
        let e = i + shift;
        let x = match e {
            0 => String::new(),
            1 => "x".to_string(),
            e => format!("x^{e}"),
        };
        match (y, x.is_empty()) {
            (false, true) => out.push('1'),
            (false, false) => out.push_str(&x),
            (true, true) => out.push('y'),
            (true, false) => {
                out.push_str("y*");
                out.push_str(&x);
            }
        }
    }
    out
}

/// A case: its texts, the verdict the check gives, and the terms its
/// texts hold and the terms its expansions write, each at the least.
struct Case {
    name: &'static str,
    instance: String,
    basis: String,
    cofactors: String,
    verdict: Verdict,
    read: usize,
    written: usize,
    /// The terms of F and of G, each evaluated at every point in the cheap
    /// mode.
    values: usize,
}

const ACCEPT: Verdict = Verdict {
    groebner: Ok(()),
    reduced: Ok(()),
};

/// The cases over the characteristic p, each with N-term polynomials.
fn cases(p: u64) -> Vec<Case> {
    let c = powers(N, 0, false);
    // 1 − c: the constant cancels.
    let minus = format!(
        "-{terms}",
        terms = powers(N - 1, 1, false).replace('+', "-")
    );
    vec![
        // c·f + (1 − c)·f = g, f = g = x − 1: the row's expansion writes
        // 4N − 2 products and g's two terms, which cancel to zero.
        Case {
            name: "long entries",
            instance: format!("x\n{p}\nx-1,\nx-1\n"),
            basis: "[x-1]:\n".to_string(),
            cofactors: format!("[[{c},{minus}]]:\n"),
            verdict: ACCEPT,
            read: 2 * N,
            written: 4 * N,
            values: 6,
        },
        // A row past the last element is read and not expanded.
        Case {
            name: "a row past the last",
            instance: format!("x\n{p}\nx-1\n"),
            basis: "[x-1]:\n".to_string(),
            cofactors: format!("[[1],\n[{c}]]:\n"),
            verdict: Verdict {
                groebner: Err(Flaw::Shape),
                reduced: Ok(()),
            },
            read: N,
            written: 0,
            values: 4,
        },
        // F = G = {c}: the generator reduces by g in one step, and the row
        // 1·f − g cancels.
        Case {
            name: "a long element",
            instance: format!("x\n{p}\n{c}\n"),
            basis: format!("[{c}]:\n"),
            cofactors: "[[1]]:\n".to_string(),
            verdict: ACCEPT,
            read: 2 * N,
            written: 4 * N - 4,
            values: 2 * N,
        },
        // F = G = {x·c, y·c}: the S-polynomial y·(x·c) − x·(y·c) cancels to
        // zero, over GF(p) as the geobucket merges its second summand, over
        // ℚ in the extraction of its first step.
        Case {
            name: "an S-pair of long elements",
            instance: format!(
                "x,y\n{p}\n{xc},\n{yc}\n",
                xc = powers(N, 1, false),
                yc = powers(N, 0, true)
            ),
            basis: format!(
                "[{yc},\n{xc}]:\n",
                yc = powers(N, 0, true),
                xc = powers(N, 1, false)
            ),
            cofactors: "[[0,1],\n[1,0]]:\n".to_string(),
            verdict: ACCEPT,
            read: 4 * N,
            written: 8 * N - 4,
            values: 4 * N,
        },
    ]
}

/// The points of the cheap mode.
const POINTS: u32 = 4;

fn modes() -> [Mode; 2] {
    [
        Mode::Exact,
        Mode::Cheap {
            points: NonZeroU32::new(POINTS).unwrap(),
            seed: 0x7011,
        },
    ]
}

/// The verdict of a check whose poll never stops, and how often it asked.
fn counted(case: &Case, mode: Mode) -> (Result<Option<Verdict>, Error>, usize) {
    let mut polls = 0;
    let got = check_polled(
        &case.instance,
        &case.basis,
        &case.cofactors,
        &Order::Grevlex,
        mode,
        &mut || {
            polls += 1;
            false
        },
    );
    (got, polls)
}

/// Test 1, .1 to .12: on each case, field, and mode, a poll that never
/// stops leaves `check`'s verdict, and the check asks it at least once per
/// `STEPS` terms it reads and writes: in the exact mode the texts' terms
/// and the expansions', in the cheap mode the texts' terms and the terms
/// of F and G at every point.
#[test]
fn the_check_polls_as_it_reads_and_expands() {
    for p in FIELDS {
        for case in cases(p) {
            for mode in modes() {
                let want = check(
                    &case.instance,
                    &case.basis,
                    &case.cofactors,
                    &Order::Grevlex,
                    mode,
                );
                assert_eq!(
                    want,
                    Ok(case.verdict.clone()),
                    "{name} over {p} in {mode:?}",
                    name = case.name
                );
                let (got, polls) = counted(&case, mode);
                assert_eq!(
                    got,
                    want.map(Some),
                    "{name} over {p} in {mode:?}",
                    name = case.name
                );
                let work = match mode {
                    Mode::Exact => case.read + case.written,
                    _ => case.read + POINTS as usize * case.values,
                };
                assert!(
                    polls >= work / STEPS,
                    "{name} over {p} in {mode:?}: {polls} polls for {work} terms",
                    name = case.name
                );
            }
        }
    }
}

/// Test 2, .2 to .12: a poll that answers true at its k-th call, k the
/// first, a quarter, half, three quarters, and the last of the calls of a
/// check that never stops, stops the check with `Ok(None)`, and is not
/// called again: inside the readings,
/// the expansions, the extractions, the heap, and the values at a point.
#[test]
fn a_poll_that_answers_true_anywhere_stops_the_check() {
    for p in FIELDS {
        for case in cases(p) {
            for mode in modes() {
                let (_, total) = counted(&case, mode);
                for k in [1, total / 4, total / 2, 3 * total / 4, total] {
                    if k == 0 {
                        continue;
                    }
                    let mut calls = 0;
                    let got = check_polled(
                        &case.instance,
                        &case.basis,
                        &case.cofactors,
                        &Order::Grevlex,
                        mode,
                        &mut || {
                            calls += 1;
                            calls >= k
                        },
                    );
                    assert_eq!(
                        got,
                        Ok(None),
                        "{name} over {p} in {mode:?}, stopped at {k}",
                        name = case.name
                    );
                    assert_eq!(
                        calls,
                        k,
                        "{name} over {p} in {mode:?}: called after it answered true",
                        name = case.name
                    );
                }
            }
        }
    }
}

/// Test 3, .8: a reading error before the poll stops is the error; a
/// stop before the reading reaches an error is the stop.
#[test]
fn a_reading_error_before_the_stop_is_the_error() {
    for p in FIELDS {
        let c = powers(N, 0, false);
        let instance = format!("x\n{p}\nx-1\n");
        // The error stands after N terms: a stop at the first call comes
        // before the reading reaches it.
        let late = format!("[{c}+*]:\n");
        let stop = |basis: &str| {
            check_polled(
                &instance,
                basis,
                "[[1]]:\n",
                &Order::Grevlex,
                Mode::Exact,
                &mut || true,
            )
        };
        let never = |basis: &str| check(&instance, basis, "[[1]]:\n", &Order::Grevlex, Mode::Exact);
        assert!(
            matches!(never(&late), Err(Error::Syntax { .. })),
            "over {p}: {got:?}",
            got = never(&late)
        );
        assert_eq!(stop(&late), Ok(None), "over {p}");
        // The error stands in the first term: no poll comes before it.
        let early = "[*x-1]:\n";
        assert!(
            matches!(stop(early), Err(Error::Syntax { .. })),
            "over {p}: {got:?}",
            got = stop(early)
        );
    }
}

/// Test 4, .1: the edge cases of `OBJECTIVES.md` over x, y: a poll that
/// never stops leaves `check`'s verdict, the check asks it at least once,
/// and a poll that answers true at its first call stops the check.
#[test]
fn the_edge_cases() {
    // (name, generators, basis, cofactor rows)
    let edges = [
        ("the zero ideal", "0", "[]:\n", "[]:\n"),
        ("the unit ideal", "x,\nx-1", "[1]:\n", "[[1,-1]]:\n"),
        (
            "already a Gröbner basis",
            "x,\ny",
            "[y,\nx]:\n",
            "[[0,1],\n[1,0]]:\n",
        ),
        (
            "the zero polynomial among the generators",
            "0,\nx-1",
            "[x-1]:\n",
            "[[0,1]]:\n",
        ),
        (
            "an eliminable variable",
            "x-y,\ny^2-1",
            "[x-y,\ny^2-1]:\n",
            "[[1,0],\n[0,1]]:\n",
        ),
    ];
    for p in FIELDS {
        for (name, generators, basis, cofactors) in edges {
            let instance = format!("x,y\n{p}\n{generators}\n");
            for mode in modes() {
                let want = check(&instance, basis, cofactors, &Order::Grevlex, mode);
                assert_eq!(want, Ok(ACCEPT), "{name} over {p} in {mode:?}");
                let mut polls = 0;
                let got = check_polled(
                    &instance,
                    basis,
                    cofactors,
                    &Order::Grevlex,
                    mode,
                    &mut || {
                        polls += 1;
                        false
                    },
                );
                assert_eq!(got, want.map(Some), "{name} over {p} in {mode:?}");
                assert!(polls > 0, "{name} over {p} in {mode:?}: no poll");
                let got = check_polled(
                    &instance,
                    basis,
                    cofactors,
                    &Order::Grevlex,
                    mode,
                    &mut || true,
                );
                assert_eq!(got, Ok(None), "{name} over {p} in {mode:?}");
            }
        }
    }
}
