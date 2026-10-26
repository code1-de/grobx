//! call-report, sub-item 2: the report of a call over ℚ: the lift's lines
//! beside its runs', the learn run whose record the result comes from
//! against the outcome's stats, and the texts and the checker over ℚ
//! (`iterations/0064-call-report/plan.md`, _Test code_, 18 without its
//! checks of the edge cases' lines, and 19 on generated systems with no
//! reference digest).

use grobx::{CheckerMode, Outcome, Params};

use crate::common::{dense, params, random_square};
use crate::report::{Line, assert_run_counts, of, report as lines_of, run_lines, without_report};

/// `params` with the report on.
fn reported(seed: u64) -> Params {
    let mut p = params(seed);
    p.report = true;
    p
}

/// The outcome of `text` under `params`, which ask for the report, and its
/// lines; the same call without the report returns the same outcome but
/// for the report.
fn both(text: &str, params: &Params) -> (Outcome, Vec<Line>) {
    let on = grobx::groebner(text, params).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    let mut off = params.clone();
    off.report = false;
    let off = grobx::groebner(text, &off).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    assert_eq!(
        without_report(&on),
        off,
        "the report changed the outcome:\n{text}"
    );
    let lines = lines_of(&on);
    (on, lines)
}

/// What every report of a call over ℚ that returns a basis holds: `read`
/// names the field `q`, the `lift` line follows it, every run's counts
/// agree with its rounds, every run is the lift's, a `learn` recorded
/// before the first `combine`, every `combine` after a replay's run, a
/// `verify` that passed before the texts, and the outcome's stats in the
/// `ran` line of the learn run whose record the verification checked.
fn assert_lift(name: &str, lines: &[Line], outcome: &Outcome) {
    assert!(
        lines[1].is("read", &[("field", "q"), ("characteristic", "0")]),
        "{name}"
    );
    assert_eq!(lines[2].kind, "lift", "{name}: {lift:?}", lift = lines[2]);
    for ran in of(lines, "ran") {
        assert_run_counts(run_lines(lines, ran.num("number")));
    }
    let mut role = None;
    let mut recorded = None;
    let mut learn_ran: Option<&Line> = None;
    let mut last_ran: Option<&Line> = None;
    for (k, line) in lines.iter().enumerate() {
        match line.kind.as_str() {
            "run" => {
                let r = line.value("role");
                assert!(
                    matches!(r, "learn" | "replay" | "verify"),
                    "{name}: {line:?}"
                );
                role = Some(r.to_string());
            }
            "ran" => last_ran = Some(line),
            "learn" if line.get("outcome") == Some("recorded") => {
                recorded = Some(k);
                learn_ran = last_ran;
            }
            "combine" => {
                assert!(recorded.is_some(), "{name}: a combine before a record");
                assert_eq!(role.as_deref(), Some("replay"), "{name}: {line:?}");
            }
            _ => {}
        }
    }
    let passed = lines
        .iter()
        .rposition(|l| l.is("verify", &[("outcome", "passed")]))
        .unwrap_or_else(|| panic!("{name}: no verification passed"));
    assert!(recorded.unwrap() < passed, "{name}");
    let texts = lines.iter().position(|l| l.kind == "text").expect("a text");
    assert!(passed < texts, "{name}: the texts follow the verification");
    let ran = learn_ran.expect("the learn's run");
    let stats = &outcome.stats;
    assert_eq!(
        (ran.num("rounds"), ran.num("reduced"), ran.num("zero")),
        (u64::from(stats.rounds), stats.reduced, stats.zero),
        "{name}: {ran:?}"
    );
    let basis = lines[texts..]
        .iter()
        .find(|l| l.is("text", &[("kind", "basis")]))
        .expect("the basis text");
    assert_eq!(
        basis.num("bytes"),
        outcome.basis.as_deref().unwrap().len() as u64
    );
    for combine in of(lines, "combine") {
        for key in ["merge", "reconstruct", "crt"] {
            combine.secs(key);
        }
        combine.num("open");
        combine.num("attempts");
        combine.num("primes");
    }
    let basis_text = outcome.basis.as_deref().unwrap();
    let elements = if basis_text == "[]:\n" {
        0
    } else {
        basis_text.matches(",\n").count() as u64 + 1
    };
    let last = lines[..passed]
        .iter()
        .rfind(|l| l.kind == "combine")
        .expect("a combine before the verification");
    assert_eq!(
        last.num("kept"),
        elements,
        "{name}: the last combine keeps every element"
    );
}

/// The edge cases of `OBJECTIVES.md` on x, y, z over ℚ.
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

/// Test 18: the edge cases over ℚ, signatures on and off: the lift's
/// lines of `assert_lift`, the pair learn's run over its two primes, its
/// rows by the learner in the signature run and its tests in the plain
/// run, and the outcome unchanged by the report.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        let text = format!("x,y,z\n0\n{generators}\n");
        for signatures in [true, false] {
            let mut p = reported(0x9e);
            p.signatures = signatures;
            let (outcome, lines) = both(&text, &p);
            let name = format!("{case}, signatures {signatures}");
            assert_lift(&name, &lines, &outcome);
            let lift = &lines[2];
            assert!(
                lift.is("lift", &[("learns", "pairs"), ("track", "off")]),
                "{name}"
            );
            let first = &of(&lines, "run")[0];
            assert!(
                first.is("run", &[("role", "learn"), ("field", "gf31pair")]),
                "{name}: {first:?}"
            );
            // q-zero-work-time.5: the pair learn's plain run takes tests,
            // and its signature run reduces rows for the learner.
            let (reduce, by) = if signatures {
                ("rows", Some("learner"))
            } else {
                ("tests", None)
            };
            assert_eq!(first.value("reduce"), reduce, "{name}: {first:?}");
            assert_eq!(first.get("reduce-by"), by, "{name}: {first:?}");
            assert_eq!(
                first.value("primes").split(',').count(),
                2,
                "{name}: {first:?}"
            );
        }
    }
}

/// Test 19: generated systems over ℚ: each report holds the lift's lines;
/// with cofactors and the exact checker the replays track cofactors, the
/// learn runs do not, and the two texts and the checker's line follow the
/// verification.
#[test]
fn generated_systems() {
    for (stem, text) in [
        ("cubic-2-p0", dense(2, 3, 0, 1)),
        ("cubic-3-p0", dense(3, 3, 0, 2)),
        ("square-p0", random_square(0, 1)),
    ] {
        let (outcome, lines) = both(&text, &reported(0x91));
        assert_lift(stem, &lines, &outcome);
        assert!(
            !of(&lines, "combine").is_empty(),
            "{stem}: no prime combined"
        );
        assert!(
            of(&lines, "run")
                .iter()
                .filter(|l| l.is("run", &[("role", "replay")]))
                .all(|l| l.is("run", &[("engine", "replay"), ("cofactors", "off")])),
            "{stem}"
        );

        let mut p = reported(0x92);
        p.cofactors = true;
        p.checker = Some(CheckerMode::Exact);
        let (outcome, lines) = both(&text, &p);
        assert_eq!(outcome.certified, Some(true), "{stem}");
        assert_lift(stem, &lines, &outcome);
        assert!(
            lines[2].is("lift", &[("track", "on")]),
            "{stem}: {lift:?}",
            lift = lines[2]
        );
        for run in of(&lines, "run") {
            let want = if run.value("role") == "replay" {
                "on"
            } else {
                "off"
            };
            assert_eq!(run.value("cofactors"), want, "{stem}: {run:?}");
        }
        let n = lines.len();
        let tail: Vec<&str> = lines[n - 4..].iter().map(|l| l.kind.as_str()).collect();
        assert_eq!(tail, ["text", "text", "check", "end"], "{stem}");
        assert!(lines[n - 3].is("text", &[("kind", "cofactors")]));
        assert_eq!(
            lines[n - 3].num("bytes"),
            outcome.cofactors.as_deref().unwrap().len() as u64
        );
        assert!(lines[n - 2].is("check", &[("mode", "exact"), ("verdict", "certified")]));
    }
}
