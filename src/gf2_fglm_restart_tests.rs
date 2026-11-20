//! gf2-fglm-restart-test, the item's test code: over GF(2) a lex call whose
//! staircase passes `Dense4`'s degree bound of 7 monomials halts its change
//! with `Halt::Overflow` in `Dense4`, starts over in `Dense8`, and returns
//! the outcome of the call in `Dense8` alone; one that passes `Dense8`'s
//! bound too starts over again in `Dense16`; and one of 7 monomials or
//! fewer stays in `Dense4` (`iterations/0148-gf2-fglm-restart-test/plan.md`,
//! _Test code_, 1 to 5).

use crate::call::built;
use crate::format_tests::{alone, call, heard, params, staircase};
use crate::halt::Halt;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format, WIDTHS, Width};
use crate::report_tests::lines;
use crate::systems::boolean;
use crate::{Error, Outcome, Params};

/// The cases of test 2, each with its lex basis: the item's example, the
/// field equations on x, y, z, D = 8; the same with the zero polynomial
/// among them; and w = x·y beside them, an eliminable variable, D = 8.
const PAST: [(&str, &str, &str); 3] = [
    (
        "the field equations on x, y, z",
        "x,y,z\n2\nx^2+x,\ny^2+y,\nz^2+z\n",
        "[z^2+z,\ny^2+y,\nx^2+x]:\n",
    ),
    (
        "the zero polynomial among the generators",
        "x,y,z\n2\nx^2+x,\n0,\ny^2+y,\nz^2+z\n",
        "[z^2+z,\ny^2+y,\nx^2+x]:\n",
    ),
    (
        "an eliminable variable",
        "w,x,y,z\n2\nw+x*y,\nw^2+w,\nx^2+x,\ny^2+y,\nz^2+z\n",
        "[z^2+z,\ny^2+y,\nx^2+x,\nw+x*y]:\n",
    ),
];

/// The cases of test 4, each with its lex basis and its staircase, None
/// where the ideal is not zero-dimensional: a staircase at `Dense4`'s
/// bound, the unit ideal's of none, and the zero ideal's edge cases.
const WITHIN: [(&str, &str, &str, Option<usize>); 5] = [
    (
        "a staircase of 7",
        "x,y,z\n2\nx^2+x,\ny^2+y,\nz^2+z,\nx*y*z\n",
        "[z^2+z,\ny^2+y,\nx*y*z,\nx^2+x]:\n",
        Some(7),
    ),
    ("the unit ideal", "x,y,z\n2\nx+1,\nx\n", "[1]:\n", Some(0)),
    ("the zero ideal", "x,y,z\n2\n0\n", "[]:\n", None),
    ("F = (0, 0)", "x,y,z\n2\n0,\n0\n", "[]:\n", None),
    ("no generator", "x,y,z\n2\n\n", "[]:\n", None),
];

/// The variables of test 3's field equations, from the least that passes
/// `Dense4`'s bound, D = 8, to one past `Dense8`'s, D = 256.
const FIELD_VARIABLES: [usize; 6] = [3, 4, 5, 6, 7, 8];

/// The (n, m) of test 5's systems `boolean(n, m, seed)`, whose staircases
/// fall on both sides of `Dense4`'s bound over `BOOLEAN_SEEDS`
/// (`instrument/bool-staircases.txt`).
const BOOLEAN: [(usize, usize); 4] = [(4, 1), (4, 2), (5, 2), (5, 3)];

/// The seeds 0 to 7 of test 5's systems.
const BOOLEAN_SEEDS: u64 = 8;

/// The least count of test 5's systems on each side of the bound: 12 of
/// its 32 pass it and 20 do not (`instrument/bool-staircases.txt`).
const EACH_SIDE: usize = 10;

/// What a call's report says of its path: the width in bits and the cause
/// of each run, in order, the verdict of each change, in order, and the
/// last change's staircase.
#[derive(Debug, PartialEq, Eq)]
struct Path {
    runs: Vec<(u32, String)>,
    verdicts: Vec<String>,
    staircase: Option<usize>,
}

// gf2-fglm-restart-test.2 design-checked -- the start over is read from the call's report, its run lines' width and why and its fglm lines' verdict, beside the matrices heard, not the matrices heard alone, which a start over after an overflow in a round gives in the same order, nor the outcome alone, the same whichever width made it (not measured)
/// The path of the call of `text` under `params` in `WIDTHS`, read from
/// its report.
fn path(case: &str, text: &str, params: &Params) -> Path {
    let mut p = params.clone();
    p.report = true;
    let outcome = crate::groebner(text, &p).unwrap_or_else(|e| panic!("{case}: {e:?}"));
    let report = outcome
        .report
        .unwrap_or_else(|| panic!("{case}: no report"));
    let lines = lines(&report);
    let runs: Vec<(u32, String)> = lines
        .iter()
        .filter(|l| l.kind == "run")
        .map(|l| {
            let width = l.get("width").and_then(|w| w.parse().ok());
            let why = l.get("why").map(str::to_string);
            width
                .zip(why)
                .unwrap_or_else(|| panic!("{case}: a run line without its width or why: {l:?}"))
        })
        .collect();
    let changes: Vec<_> = lines.iter().filter(|l| l.kind == "fglm").collect();
    let verdicts = changes
        .iter()
        .map(|l| {
            l.get("verdict")
                .unwrap_or_else(|| panic!("{case}: an fglm line without its verdict: {l:?}"))
                .to_string()
        })
        .collect();
    let staircase: Option<usize> = changes
        .last()
        .and_then(|l| l.get("staircase"))
        .map(|d| d.parse().expect("a staircase's size is a count"));
    Path {
        runs,
        verdicts,
        staircase,
    }
}

/// The degree bound of the width `w`'s format, the most monomials a
/// staircase takes in it (monomial-formats.10).
fn bound(w: Width) -> usize {
    let max = match w {
        Width::W4 => Dense4::MAX_DEGREE,
        Width::W8 => Dense8::MAX_DEGREE,
        Width::W16 => Dense16::MAX_DEGREE,
        Width::W32 => Dense32::MAX_DEGREE,
    };
    max as usize
}

/// The widths a lex call over GF(2) whose staircase holds `d` monomials
/// runs in: `W4`, and the next while the width before it holds fewer.
fn widths(d: usize) -> Vec<Width> {
    let mut widths = vec![Width::W4];
    loop {
        let w = *widths.last().expect("W4 at least");
        if d <= bound(w) {
            return widths;
        }
        widths.push(w.wider().expect("a staircase within the widest bound"));
    }
}

/// The path of a lex call over GF(2) whose staircase holds `d` monomials:
/// a run in each of `widths(d)`, the first for `start` and each later for
/// `wider`; a change in each, `overflow` in all but the last, `change` in
/// the last, at staircase `d`.
fn expected(d: usize) -> Path {
    let widths = widths(d);
    let runs = widths
        .iter()
        .enumerate()
        .map(|(i, w)| (w.bits(), if i == 0 { "start" } else { "wider" }.to_string()))
        .collect();
    let mut verdicts = vec!["overflow".to_string(); widths.len() - 1];
    verdicts.push("change".to_string());
    Path {
        runs,
        verdicts,
        staircase: Some(d),
    }
}

/// The path of a lex call over GF(2) on an ideal that is not
/// zero-dimensional: its grevlex run in `W4`, no change, and F4 in lex in
/// `W4` (fglm-lex.3).
fn positive_dimensional() -> Path {
    Path {
        runs: vec![(4, "start".to_string()), (4, "lex".to_string())],
        verdicts: vec!["positive-dimensional".to_string()],
        staircase: None,
    }
}

/// The parameters of a lex call with cofactors and without, signatures on
/// and off, each with its name; those of a build without the feature they
/// ask for left out.
fn variants() -> Vec<(String, Params)> {
    let mut variants = Vec::new();
    for cofactors in [false, true] {
        for signatures in [true, false] {
            let mut p = params("lex", cofactors);
            if built(&p).is_err() {
                continue;
            }
            p.signatures = signatures;
            variants.push((format!("cofactors {cofactors}, signatures {signatures}"), p));
        }
    }
    variants
}

/// The field equations on x1, …, xn, `boolean(n, 0, _)`, and their lex
/// basis: the same equations in increasing order of their leading
/// monomials.
fn fields(n: usize) -> (String, String) {
    let elements: Vec<String> = (1..=n).rev().map(|i| format!("x{i}^2+x{i}")).collect();
    let basis = format!("[{elements}]:\n", elements = elements.join(",\n"));
    (boolean(n, 0, 0), basis)
}

/// A basis, certified where the call tracks cofactors.
fn assert_held(case: &str, p: &Params, outcome: &Outcome) {
    assert!(outcome.basis.is_some(), "{case}: {outcome:?}");
    if p.cofactors {
        assert_eq!(
            outcome.certified,
            Some(true),
            "{case}: {flaw:?}",
            flaw = outcome.flaw
        );
    }
}

/// The checks of a case whose change passes the bound of every width of
/// `widths` but the last (plan.md, _Approach_): alone in each of those
/// widths the call is `Error::Overflow` with no line; alone in the last it
/// holds a basis, certified where it tracks cofactors; the call in
/// `WIDTHS` returns that outcome and hears the matrices of the calls in
/// `widths` alone, in their order. Returns the outcome.
fn starts_over(case: &str, text: &str, p: &Params, widths: &[Width]) -> Outcome {
    let (&last, passed) = widths.split_last().expect("a width");
    for &w in passed {
        assert_eq!(
            alone(text, p, w),
            Err(Error::Overflow { line: None }),
            "{case}: {w:?} alone"
        );
    }
    let held = alone(text, p, last).unwrap_or_else(|e| panic!("{case}: {last:?} alone: {e:?}"));
    assert_held(case, p, &held);
    let (all, all_trace, _) = heard(text, p, WIDTHS.0, WIDTHS.1);
    assert_eq!(all.as_ref(), Ok(&held), "{case}: in WIDTHS");
    let mut want = Vec::new();
    for &w in widths {
        let (_, trace, _) = heard(text, p, w, w);
        want.extend(trace.0);
    }
    assert_eq!(all_trace.0, want, "{case}: the matrices heard");
    held
}

/// The checks of a case whose change, where it runs, holds in `W4`: the
/// call alone in `W4` holds a basis, certified where it tracks cofactors,
/// and the call alone in `W8` and the call in `WIDTHS` return its outcome.
/// Returns the outcome.
fn stays(case: &str, text: &str, p: &Params) -> Outcome {
    let w4 = alone(text, p, Width::W4).unwrap_or_else(|e| panic!("{case}: W4 alone: {e:?}"));
    assert_held(case, p, &w4);
    assert_eq!(
        alone(text, p, Width::W8).as_ref(),
        Ok(&w4),
        "{case}: W8 alone"
    );
    assert_eq!(
        call(text, p, WIDTHS.0, WIDTHS.1).as_ref(),
        Ok(&w4),
        "{case}: in WIDTHS"
    );
    w4
}

/// Test 1: monomial-formats.10 in `Dense4`. A staircase of 7 monomials,
/// `Dense4`'s degree bound, holds; one of 8 is `Halt::Overflow`, which
/// `Dense8` holds.
#[test]
fn the_staircase_at_the_narrow_bound() {
    assert_eq!(staircase::<Dense4>(1, 7), Ok(Some(7)));
    assert_eq!(staircase::<Dense4>(2, 4), Err(Halt::Overflow));
    assert_eq!(staircase::<Dense8>(2, 4), Ok(Some(8)));
}

/// Test 2: monomial-formats.8 and .10 over GF(2). The cases of `PAST`, with
/// cofactors and without, signatures on and off: the call halts its change
/// in `W4` and starts over in `W8` (`starts_over`), with the case's basis;
/// its report names the run in width 4 for `start` and in width 8 for
/// `wider`, the change's `overflow` and then its `change` at staircase 8.
#[test]
fn a_staircase_past_seven_starts_the_call_over() {
    for (name, text, basis) in PAST {
        for (variant, p) in variants() {
            let case = format!("{name}, {variant}");
            let held = starts_over(&case, text, &p, &[Width::W4, Width::W8]);
            assert_eq!(held.basis.as_deref(), Some(basis), "{case}");
            assert_eq!(path(&case, text, &p), expected(8), "{case}");
        }
    }
}

/// Test 3: the field equations on n = 3 to 8 variables, D = 2^n, with
/// cofactors and without, signatures on and off: each call starts over in
/// `W8`, and for n = 7 and 8, past `Dense8`'s bound of 127, again in `W16`
/// (`starts_over`), with their basis; their reports name each run and each
/// change, at staircase 2^n.
#[test]
fn the_field_equations_start_over_by_their_degree() {
    assert_eq!(widths(1 << 6), [Width::W4, Width::W8]);
    assert_eq!(widths(1 << 7), [Width::W4, Width::W8, Width::W16]);
    for n in FIELD_VARIABLES {
        let (text, basis) = fields(n);
        let d = 1 << n;
        for (variant, p) in variants() {
            let case = format!("the field equations on {n} variables, {variant}");
            let held = starts_over(&case, &text, &p, &widths(d));
            assert_eq!(held.basis.as_deref(), Some(basis.as_str()), "{case}");
            assert_eq!(path(&case, &text, &p), expected(d), "{case}");
        }
    }
}

/// Test 4: the cases of `WITHIN`, with cofactors and without, signatures
/// on and off: the call stays in `W4` with the outcome of `W8` alone
/// (`stays`), with the case's basis; its report names one run, in width 4
/// for `start`, and the change at the case's staircase, or, where the ideal
/// is not zero-dimensional, `positive-dimensional` and F4 in lex in width
/// 4 for `lex`.
#[test]
fn a_staircase_within_seven_stays() {
    for (name, text, basis, d) in WITHIN {
        for (variant, p) in variants() {
            let case = format!("{name}, {variant}");
            let held = stays(&case, text, &p);
            assert_eq!(held.basis.as_deref(), Some(basis), "{case}");
            let want = d.map_or_else(positive_dimensional, expected);
            assert_eq!(path(&case, text, &p), want, "{case}");
        }
    }
}

// gf2-fglm-restart-test.3 design-checked -- a random case's expected path follows from the staircase its call's report names, D past 7 a start over in W8 and past 127 in W16 too, D up to 7 none, not each case marked with its path in advance (log 0148-1)
// gf2-fglm-restart-test.4 design-checked -- the systems are the field equations on 3 to 8 variables, the item's example with the zero polynomial and with an eliminable variable, a system at D = 7, 32 systems boolean(n, m, seed), and the edge cases that stay in Dense4, not the item's example alone nor random square systems over GF(2) (log 0148-1)
// gf2-fglm-restart-test.5 design-checked -- test 5 leaves out the call with cofactors and signatures on, whose signature run in grevlex reduces every S-pair its criteria leave and halts in a round of W4 on boolean(5, 2, 3) before its change, not the round's halt read from its ran line and the case checked as a start over for it (log 0148-1)
/// Test 5: the systems `boolean(n, m, seed)` of `BOOLEAN` and
/// `BOOLEAN_SEEDS`, with cofactors and without, signatures on and off, but
/// with cofactors and signatures on: each call's report names the path of
/// its staircase D (`expected`); for D past 7 the call starts over
/// (`starts_over`), else it stays (`stays`). At least `EACH_SIDE` systems
/// lie on each side of the bound, and D = 7 and D = 8 are each drawn.
#[test]
fn random_boolean_systems_by_their_staircase() {
    let mut drawn = Vec::new();
    for (n, m) in BOOLEAN {
        for seed in 0..BOOLEAN_SEEDS {
            let text = boolean(n, m, seed);
            let mut staircases = Vec::new();
            for (variant, p) in variants() {
                if p.cofactors && p.signatures {
                    continue;
                }
                let case = format!("boolean({n}, {m}, {seed}), {variant}\n{text}");
                let got = path(&case, &text, &p);
                let d = got
                    .staircase
                    .unwrap_or_else(|| panic!("{case}: no staircase: {got:?}"));
                assert_eq!(got, expected(d), "{case}");
                if d > bound(Width::W4) {
                    starts_over(&case, &text, &p, &widths(d));
                } else {
                    stays(&case, &text, &p);
                }
                staircases.push(d);
            }
            staircases.dedup();
            assert_eq!(
                staircases.len(),
                1,
                "boolean({n}, {m}, {seed}): {staircases:?}"
            );
            drawn.extend(staircases);
        }
    }
    let past = drawn.iter().filter(|&&d| d > bound(Width::W4)).count();
    let within = drawn.len() - past;
    assert!(
        past >= EACH_SIDE && within >= EACH_SIDE,
        "{past} systems past the bound and {within} within it"
    );
    assert!(drawn.contains(&7) && drawn.contains(&8), "{drawn:?}");
}
