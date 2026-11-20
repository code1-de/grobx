//! ccode-newton-gap, the item's test code: a run that substitutes its
//! linear generators' pivots in its other generators returns what the same
//! run without the substitution returns, over GF(p) and ℚ, in every order,
//! with cofactors and the checker, and on the edge cases; the report's
//! `substitute` line; the substitution itself; a replay of a learn's
//! record at another prime; and a stop in the substitution
//! (`iterations/0118-ccode-newton-gap/plan.md`, _Test code_, 1 to 5).
//!
//! A pair of calls is the same call with `SUBSTITUTE` true and false.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small exponents and residues mod p"
)]

use std::fmt::Write;
use std::time::Duration;

use crate::call::built;
use crate::cyclic_gap_tests::over;
use crate::field::Gf31;
use crate::input_redundancy_tests::{read, table};
use crate::module::Rank1;
use crate::order::Grevlex;
use crate::poly::Poly;
use crate::rebuilt_reducer_tests::reported;
use crate::report_tests::Line;
use crate::shared_tests::{FIELDS, P, P31, budget, report};
use crate::substitute::{SUBSTITUTE, Substituted, substitute};
use crate::systems::{Rng, dense, random_sparse};
use crate::trace::NoTrace;
use crate::{CheckerMode, MonomialOrder, Outcome, Params, Stop};

/// Linear generators over x1 to x4, and over x1 to x5, that test 1 adds:
/// over each field of `FIELDS` their echelon leads with x1 and x2, and
/// with x1 and x3.
const LINEAR4: [&str; 2] = ["x1-3*x3+2*x4-5", "x2+x3-6*x4+1"];
const LINEAR5: [&str; 2] = ["x1+2*x2-x5+3", "x3-4*x4+x5"];

/// The edge cases over x, y, z (plan.md, _Edge cases_): the zero ideal,
/// linear generators that contradict each other, pivots that make a
/// generator a nonzero constant, an input already a Gröbner basis, the
/// zero polynomial among the generators, a generator the substitution makes
/// zero and one it makes linear, linear generators that depend on each
/// other, every variable a pivot, and fractions.
const EDGE: [&str; 13] = [
    "0",
    "0,\n0",
    "",
    "x*z-1,\nx",
    "x+1,\nx+2,\ny^2-z",
    "x-1,\ny-2,\nz-3,\nx*y*z-7",
    "x-y,\ny-z,\nz^2-1",
    "0,\nx-y,\ny^2-1",
    "x-y,\nx^2-y^2",
    "x-y-1,\nx^2-y^2,\nz^2-x",
    "x+y,\n2*x+2*y,\ny^2-z,\nz^3-x*y",
    "x-1,\ny-2,\nz-3,\nx*y*z-6",
    "x-1/3*y+2/5,\ny^2-3/2*z,\nz^2-x*y",
];

/// The system whose `x` the substitution eliminates, in lex and `block:1`.
const ELIMINABLE: &str = "x,y,z\n7\nx-y-z,\nx*y-1,\nz^2-x*z+y\n";

/// `text`'s generators with `forms` added, before them where `front`
/// holds, else after.
fn with(text: &str, forms: &[&str], front: bool) -> String {
    let mut parts = text.splitn(3, '\n');
    let variables = parts.next().unwrap();
    let p = parts.next().unwrap();
    let generators = parts.next().unwrap_or("").trim_end();
    let forms = forms.join(",\n");
    let all = if generators.is_empty() {
        forms
    } else if front {
        format!("{forms},\n{generators}")
    } else {
        format!("{generators},\n{forms}")
    };
    format!("{variables}\n{p}\n{all}\n")
}

/// `generator`, a polynomial in x3 and x4, plus `count` multiples of the
/// forms `forms` over x1 to x4, each a random coefficient below `bound`
/// times a random monomial of degree at most 3 times a form, written out
/// term by term. Its substitute by the forms' pivots is `generator`.
fn plus_multiples(
    generator: &str,
    forms: &[&str],
    count: usize,
    bound: u64,
    rng: &mut Rng,
) -> String {
    let mut out = generator.to_string();
    for _ in 0..count {
        let form = forms[rng.below(forms.len() as u64) as usize];
        let c = 1 + rng.below(bound - 1);
        let mut e = [0u64; 4];
        for _ in 0..rng.below(4) {
            e[rng.below(4) as usize] += 1;
        }
        let mut monomial = String::new();
        for (i, &k) in e.iter().enumerate().filter(|&(_, &k)| k > 0) {
            write!(monomial, "*x{v}^{k}", v = i + 1).unwrap();
        }
        let mut rest = form;
        while !rest.is_empty() {
            let sign = if rest.starts_with('-') { '-' } else { '+' };
            let body = rest.trim_start_matches(['+', '-']);
            let end = body.find(['+', '-']).unwrap_or(body.len());
            write!(out, "{sign}{c}{monomial}*{term}", term = &body[..end]).unwrap();
            rest = &body[end..];
        }
    }
    out
}

/// A system over x1 to x4 and `p` of the forms `LINEAR4` and two
/// generators, each a random sparse polynomial in x3 and x4 plus multiples
/// of the forms, whose substitutes are the sparse polynomials.
fn shortened(p: u64, seed: u64) -> String {
    let sparse = random_sparse(2, p, seed);
    let (_, rest) = sparse.split_once('\n').unwrap();
    let (_, generators) = rest.split_once('\n').unwrap();
    let bound = if p == 0 { 1 << 20 } else { p.min(1 << 20) };
    let mut rng = Rng::new(seed ^ 0x0118);
    let gens: Vec<String> = generators
        .trim_end()
        .split(",\n")
        .take(2)
        .map(|g| {
            let g = g.replace("x1", "x3").replace("x2", "x4");
            plus_multiples(&g, &LINEAR4, 3, bound, &mut rng)
        })
        .collect();
    format!(
        "x1,x2,x3,x4\n{p}\n{forms},\n{gens}\n",
        forms = LINEAR4.join(",\n"),
        gens = gens.join(",\n")
    )
}

/// The call of `text` with the substitution on or off.
pub(crate) fn call(text: &str, params: &Params, on: bool) -> Outcome {
    SUBSTITUTE.set(on);
    let outcome = crate::groebner(text, params);
    SUBSTITUTE.set(true);
    outcome.unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
}

/// The report's `substitute` lines.
fn substitutes(outcome: &Outcome) -> Vec<Line> {
    report(outcome)
        .into_iter()
        .filter(|l| l.kind == "substitute")
        .collect()
}

/// Asserts that the call of `text` returns with the substitution what it
/// returns without, and returns the outcome with it.
fn pair(name: &str, text: &str, params: &Params) -> Outcome {
    let off = call(text, params, false);
    let on = call(text, params, true);
    assert_eq!(on.basis, off.basis, "{name}: the bases");
    assert_eq!(on.cofactors, off.cofactors, "{name}: the cofactor texts");
    assert_eq!(on.stopped, off.stopped, "{name}: the stops");
    assert_eq!(on.certified, off.certified, "{name}: the verdicts");
    assert_eq!(on.flaw, off.flaw, "{name}: the flaws");
    assert!(
        substitutes(&off).is_empty(),
        "{name}: no substitution where SUBSTITUTE is false"
    );
    on
}

/// Test 1, ccode-newton-gap.1 to .6, .10, .13 to .16: every call that
/// substitutes returns what it returns without the substitution: systems
/// whose generators take shorter substitutes, random sparse systems with
/// two linear generators added and a dense one with one, whose substitutes
/// may be longer, and the edge cases over 7, 65521, 2^31 − 1, 2^61 − 1,
/// and ℚ, the
/// signatures off and on; the eliminable system in lex and `block:1`; with
/// cofactors and the exact checker, which take the generators as read; and
/// a run that widens its format.
#[test]
fn every_result_stays() {
    for p in FIELDS {
        let mut texts: Vec<(String, String, bool)> = Vec::new();
        for s in 1..=3 {
            texts.push((format!("shortened-p{p}-{s}"), shortened(p, s), true));
            texts.push((
                format!("sparse4-p{p}-{s}"),
                with(&random_sparse(4, p, s), &LINEAR4, false),
                true,
            ));
            texts.push((
                format!("sparse5-p{p}-{s}"),
                with(&random_sparse(5, p, s), &LINEAR5, false),
                true,
            ));
        }
        texts.push((
            format!("dense33-p{p}-1"),
            with(&dense(3, 3, p, 1), &["x1-x2+x3+1"], true),
            true,
        ));
        for (k, gens) in EDGE.iter().enumerate() {
            texts.push((format!("edge {k}"), format!("x,y,z\n{p}\n{gens}\n"), false));
        }
        for (system, text, linear) in &texts {
            for signatures in [false, true] {
                let mut params = reported(MonomialOrder::Grevlex);
                params.signatures = signatures;
                let name = format!("{system} over {p}, signatures {signatures}");
                let outcome = pair(&name, text, &params);
                assert!(outcome.basis.is_some(), "{name}: a basis");
                if *linear {
                    let lines = substitutes(&outcome);
                    assert!(!lines.is_empty(), "{name}: a substitute line");
                    if system.starts_with("shortened") {
                        assert!(
                            lines.iter().all(|l| count(l, "changed") == 2),
                            "{name}: both generators take their substitutes"
                        );
                    }
                }
            }
        }
    }
    for order in ["lex", "block:1"] {
        let params = reported(order.parse().unwrap());
        let outcome = pair(&format!("eliminable in {order}"), ELIMINABLE, &params);
        assert!(
            !substitutes(&outcome).is_empty(),
            "eliminable in {order}: a substitute line"
        );
    }
    for p in [65521, 0] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        if built(&params).is_err() {
            continue;
        }
        let name = format!("shortened-p{p}-1 with cofactors");
        let text = shortened(p, 1);
        let outcome = pair(&name, &text, &params);
        assert!(outcome.cofactors.is_some(), "{name}: the cofactor text");
        assert_eq!(outcome.certified, Some(true), "{name}: certified");
        assert!(
            substitutes(&outcome).is_empty(),
            "{name}: no substitution with cofactors"
        );
    }
    let params = reported(MonomialOrder::Grevlex);
    let widens = "w,x,y,z\n65521\nw-y-z,\nx^100*y+y^2+z+1,\nx*y^30+x^2*z+x,\nz^2+x*y+w\n";
    let outcome = pair("a run that widens", widens, &params);
    let lines = report(&outcome);
    assert!(
        lines
            .iter()
            .any(|l| l.is("run", &[("width", "16"), ("why", "wider")])),
        "a run that widens"
    );
    assert!(
        substitutes(&outcome).len() >= 2,
        "the run substitutes again in the wider format"
    );
}

/// A field of a `substitute` line as a count.
fn count(line: &Line, key: &str) -> usize {
    line.get(key)
        .unwrap_or_else(|| panic!("a substitute line names `{key}`"))
        .parse()
        .unwrap()
}

/// Test 2, ccode-newton-gap.11, .16: the report's `substitute` line, after
/// `inputs` and before `echelon`, with its counts and its time: of the four
/// generators that hold x, two take shorter substitutes, and one whose
/// substitute is longer and one whose substitute has as many terms stay;
/// none where the run substitutes nothing; one per run of the lift over ℚ.
#[test]
fn the_substitute_line() {
    let text = "x,y,z\n7\nx-y-1,\nx^2-y^2-2*y+z,\nx*z-y*z-1,\ny*z+x,\nx*z+y*z+y\n";
    let params = reported(MonomialOrder::Grevlex);
    let lines = report(&call(text, &params, true));
    let kinds: Vec<&str> = lines.iter().map(|l| l.kind.as_str()).collect();
    let at = kinds
        .iter()
        .position(|&k| k == "substitute")
        .expect("a substitute line");
    assert_eq!(kinds[at - 1], "inputs", "after inputs: {kinds:?}");
    assert_eq!(kinds[at + 1], "echelon", "before echelon: {kinds:?}");
    assert_eq!(kinds.iter().filter(|&&k| k == "substitute").count(), 1);
    let line = &lines[at];
    for (key, want) in [
        ("linear", 1),
        ("pivots", 1),
        ("changed", 2),
        ("stayed", 2),
        ("terms", 7),
        ("after", 4),
    ] {
        assert_eq!(count(line, key), want, "{key}");
    }
    assert!(line.get("time").is_some(), "a time");

    assert!(
        substitutes(&call(text, &params, false)).is_empty(),
        "SUBSTITUTE false"
    );
    let mut no_echelon = reported(MonomialOrder::Grevlex);
    no_echelon.echelon = false;
    assert!(
        substitutes(&call(text, &no_echelon, true)).is_empty(),
        "the echelon off"
    );
    let mut cofactors = reported(MonomialOrder::Grevlex);
    cofactors.cofactors = true;
    if built(&cofactors).is_ok() {
        assert!(
            substitutes(&call(text, &cofactors, true)).is_empty(),
            "cofactors"
        );
    }
    assert!(
        substitutes(&call(&dense(3, 3, 7, 1), &params, true)).is_empty(),
        "no linear generator"
    );

    let outcome = call(&over(text, 0), &params, true);
    let runs = report(&outcome).iter().filter(|l| l.kind == "run").count();
    assert!(runs >= 2, "the lift's runs: {runs}");
    assert_eq!(
        substitutes(&outcome).len(),
        runs,
        "a substitute line per run"
    );
}

/// Whether `a` and `b` hold the same terms.
fn same(a: &Poly<Gf31, Rank1>, b: &Poly<Gf31, Rank1>) -> bool {
    a.monomials == b.monomials && a.coefs == b.coefs
}

/// A generator of degree at most `degree` in the variables `names` of
/// `count` random terms drawn from `rng`, written out.
fn random_generator(names: &[&str], degree: u64, count: usize, rng: &mut Rng) -> String {
    let terms: Vec<String> = (0..count)
        .map(|_| {
            let mut e = vec![0u64; names.len()];
            for _ in 0..rng.below(degree + 1) {
                e[rng.below(names.len() as u64) as usize] += 1;
            }
            let mut term = (1 + rng.below(P - 1)).to_string();
            for (&k, v) in e.iter().zip(names).filter(|&(&k, _)| k > 0) {
                write!(term, "*{v}^{k}").unwrap();
            }
            term
        })
        .collect();
    terms.join("+")
}

/// Test 3, ccode-newton-gap.1, .4, .5, .7 to .9, .14 to .16:
/// `substitute::substitute` over 65521 on x1 to x4. A generator that is a
/// polynomial r in x3 and x4 plus multiples of the linear generators x1 −
/// 2x3 + 5 and x2 + x3 − x4 takes r as its substitute; a generator whose
/// substitute is longer, the linear generators, and a generator without a
/// pivot return as they were; and the counts. With linear generators that
/// contradict each other, the row that leads with 1 is no pivot.
#[test]
fn the_substitution() {
    let field = Gf31::new(P).unwrap();
    let names = "x1,x2,x3,x4";
    let forms = ["x1-2*x3+5", "x2+x3-x4"];
    let mut rng = Rng::new(0x0118);
    for round in 0..8 {
        let r = random_generator(&["x3", "x4"], 6, 6, &mut rng);
        let g = plus_multiples(&r, &forms, 2 + round, P, &mut rng);
        let h = format!("x1^3*x2^2+x1*x3+{c}", c = 1 + rng.below(P - 1));
        let generators = format!(
            "{f1},\n{g},\n{f2},\n{h},\nx3^2*x4-x4+3",
            f1 = forms[0],
            f2 = forms[1]
        );
        let mut tab = table(names);
        let gens = read(&field, &mut tab, &Grevlex, names, &generators);
        let want = read(&field, &mut tab, &Grevlex, names, &r);
        let before: Vec<Poly<Gf31, Rank1>> = gens.clone();
        let got = substitute(&field, &mut tab, &Grevlex, gens, &mut budget()).unwrap();
        let Substituted {
            generators: out, ..
        } = &got;
        assert_eq!(out.len(), 5, "round {round}");
        assert!(same(&out[1], &want[0]), "round {round}: g takes r");
        for k in [0, 2, 3, 4] {
            assert!(
                same(&out[k], &before[k]),
                "round {round}: generator {k} stays"
            );
        }
        assert_eq!(
            (
                got.linear,
                got.pivots,
                got.changed,
                got.stayed,
                got.terms,
                got.after
            ),
            (2, 2, 1, 1, before[1].len(), want[0].len()),
            "round {round}: the counts"
        );
    }

    let generators = "x1+1,\nx1+2,\nx1^2*x3+x4+x1";
    let mut tab = table(names);
    let gens = read(&field, &mut tab, &Grevlex, names, generators);
    let want = read(&field, &mut tab, &Grevlex, names, "x4");
    let got = substitute(&field, &mut tab, &Grevlex, gens, &mut budget()).unwrap();
    assert_eq!((got.linear, got.pivots, got.changed), (2, 1, 1));
    assert!(same(&got.generators[2], &want[0]), "x1 takes 0: x4");
}

/// Test 4, ccode-newton-gap.5, .10: a learn over 65521 that substitutes,
/// its record replayed over 2^31 − 1, returns the basis the call returns
/// there.
#[test]
fn the_replay_follows() {
    let mut texts: Vec<String> = (1..=3).map(|s| shortened(P, s)).collect();
    texts.push(with(&random_sparse(4, P, 1), &LINEAR4, false));
    texts.push(with(&dense(3, 3, P, 2), &["x1-x2+x3+1"], true));
    let mut replayed = 0;
    for text in &texts {
        for signatures in [false, true] {
            let mut params = reported(MonomialOrder::Grevlex);
            params.signatures = signatures;
            let (outcome, record) = crate::learn(text, &params, &mut NoTrace)
                .unwrap_or_else(|e| panic!("{e:?} on\n{text}"));
            assert!(!substitutes(&outcome).is_empty(), "the learn substitutes");
            let Some(record) = record else {
                assert!(outcome.basis.is_none(), "a basis and no record");
                continue;
            };
            let at = over(text, P31);
            let want = crate::groebner(&at, &params).unwrap().basis;
            let got = crate::replay(&at, &params, &record, &mut NoTrace)
                .unwrap_or_else(|e| panic!("{e:?} replaying\n{at}"));
            assert_eq!(got.basis, want, "signatures {signatures}\n{text}");
            assert!(!substitutes(&got).is_empty(), "the replay substitutes");
            replayed += 1;
        }
    }
    assert!(replayed >= 8, "{replayed} replays");
}

/// Test 5, ccode-newton-gap.11, .12: a call whose substitution forms many
/// products, under deadlines from 0 to 32 ms, returns its basis or stops at
/// its deadline with none; under 0 ms its run stops in the substitution.
#[test]
fn a_stop_in_the_substitution() {
    let mut rng = Rng::new(0x5ab5);
    let mut terms = Vec::new();
    for a in 0..=24u32 {
        for b in 0..=24 - a {
            for c in 0..=24 - a - b {
                let d = rng.below(25 - u64::from(a + b + c)) as u32;
                let coef = 1 + rng.below(P - 1);
                terms.push(format!("{coef}*x1^{a}*x2^{b}*x3^{c}*x4^{d}"));
            }
        }
    }
    let text = format!(
        "x1,x2,x3,x4\n{P}\nx1-x3+2,\nx2-x4-1,\n{g},\nx3^3-x4^2+x3,\nx4^3-x3*x4-1\n",
        g = terms.join("+")
    );
    let params = reported(MonomialOrder::Grevlex);
    let full = call(&text, &params, true);
    assert!(full.basis.is_some(), "the call without a deadline");
    for ms in [0, 1, 2, 4, 8, 16, 32] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.deadline = Some(Duration::from_millis(ms));
        let outcome = call(&text, &params, true);
        match outcome.stopped {
            None => assert_eq!(outcome.basis, full.basis, "{ms} ms"),
            Some(Stop::Deadline) => assert!(outcome.basis.is_none(), "{ms} ms"),
            Some(other) => panic!("{ms} ms: stopped by {other:?}"),
        }
        if ms == 0 {
            assert!(
                report(&outcome)
                    .iter()
                    .any(|l| l.is("ran", &[("outcome", "stop"), ("at", "substitute")])),
                "0 ms: the run stops in the substitution"
            );
        }
    }
}
