//! linear-tail-time, the item's test code: a plain run that inter-reduces
//! `G_min` where a member of degree 1 has joined it returns what the same
//! run without the inter-reduction returns, over GF(2), GF(p), and ℚ, in
//! grevlex, deglex, and lex, the signatures off and on, with cofactors and
//! the checker, and on the edge cases; the report's `tails` line;
//! `Basis::replace`; and `f4::tails` on a basis written out
//! (`iterations/0166-linear-tail-time/plan.md`, _Test code_, 1 to 4).
//!
//! A pair of calls is the same call with `TAILS` true and false.

use crate::budget::Budget;
use crate::call::built;
use crate::cofactors::NoCofactors;
use crate::f4::{TAILS, Tails, tails};
use crate::field::{Field, Gf31};
use crate::id::ElementId;
use crate::input_redundancy_tests::{read, table};
use crate::matrix::Buffers;
use crate::module::Rank1;
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::rebuilt_reducer_tests::reported;
use crate::report_tests::Line;
use crate::shared_tests::{FIELDS, report};
use crate::systems::{Rng, boolean, bound, cubics, term};
use crate::{CheckerMode, MonomialOrder, Outcome, Params};

/// The edge cases over x, y, z (plan.md, _Edge cases_): the zero ideal,
/// F = (0), the unit ideal, an input already a Gröbner basis, the zero
/// polynomial among the generators, an eliminable variable, a basis of
/// degree 1 alone, and members of degree 1 whose tails hold each other's
/// leading variables.
pub(crate) const EDGE: [&str; 8] = [
    "",
    "0,\n0",
    "x*z-1,\nx",
    "x-1,\ny-2,\nz^2-3",
    "0,\nx-y,\ny^2-1",
    "x-y-z,\nx*y-1,\nz^2-x*z+y",
    "x*y-z,\ny*z-x,\nx*z-y,\nx^2-1,\ny^2-1,\nz^2-1,\nx+y+z-1",
    "x^2-y-z,\ny^2-x-z,\nz^2-x-y,\nx*y*z-1",
];

/// A system over GF(p), or ℚ where p is 0, that vanishes at a point drawn
/// first: m generators on x1, …, xn, each of three to six terms of degree 1
/// or 2, every variable of a term drawn on its own, with a coefficient in
/// [1, `bound(p)`), and the constant term that makes it vanish at the
/// point, whose coordinates lie in [0, 4]. Its basis holds elements of
/// degree 1 where the point is its only zero, which m above n makes likely.
/// No family of the instance corpus is written so (`WORKFLOW.md`, step 5).
pub(crate) fn planted(n: usize, m: usize, p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(29) ^ 0x7a11);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let point: Vec<i128> = (0..n).map(|_| i128::from(rng.below(5))).collect();
    let generators: Vec<String> = (0..m)
        .map(|_| {
            let mut value: i128 = 0;
            let mut terms = Vec::new();
            for _ in 0..3 + rng.below(4) {
                let mut e = vec![0u32; n];
                for _ in 0..=rng.below(2) {
                    e[usize::try_from(rng.below(n as u64)).unwrap()] += 1;
                }
                let c = 1 + rng.below(bound(p) - 1);
                let at: i128 = e.iter().zip(&point).map(|(&k, &x)| x.pow(k)).product();
                value += i128::from(c) * at;
                terms.push(term(c, &names, &e));
            }
            let constant = if p == 0 {
                -value
            } else {
                let p = i128::from(p);
                (p - value.rem_euclid(p)) % p
            };
            let mut text = terms.join("+");
            if constant != 0 {
                if constant > 0 {
                    text.push('+');
                }
                text.push_str(&constant.to_string());
            }
            text
        })
        .collect();
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// The call of `text` with the inter-reduction between the rounds on or
/// off.
fn call(text: &str, params: &Params, on: bool) -> Outcome {
    TAILS.set(on);
    let outcome = crate::groebner(text, params);
    TAILS.set(true);
    outcome.unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
}

/// The report's `tails` lines.
pub(crate) fn tails_lines(outcome: &Outcome) -> Vec<Line> {
    report(outcome)
        .into_iter()
        .filter(|l| l.kind == "tails")
        .collect()
}

/// The count `key` of the report line `line`.
pub(crate) fn count(line: &Line, key: &str) -> usize {
    line.get(key)
        .unwrap_or_else(|| panic!("a {kind} line names `{key}`", kind = line.kind))
        .parse()
        .unwrap()
}

/// Asserts that the call of `text` returns with the inter-reduction what it
/// returns without, and that the call without writes no `tails` line;
/// returns the outcome with it.
fn pair(name: &str, text: &str, params: &Params) -> Outcome {
    let off = call(text, params, false);
    let on = call(text, params, true);
    assert_eq!(on.basis, off.basis, "{name}: the bases");
    assert_eq!(on.cofactors, off.cofactors, "{name}: the cofactor texts");
    assert_eq!(on.stopped, off.stopped, "{name}: the stops");
    assert_eq!(on.certified, off.certified, "{name}: the verdicts");
    assert_eq!(on.flaw, off.flaw, "{name}: the flaws");
    assert!(
        tails_lines(&off).is_empty(),
        "{name}: no inter-reduction where TAILS is false"
    );
    on
}

/// The systems of test 1 over the characteristic p, each with its name:
/// over GF(2) boolean and cubic systems with a planted point, over every
/// other field planted systems, and the edge cases.
fn systems(p: u64) -> Vec<(String, String)> {
    let mut texts = Vec::new();
    for s in 1..=3 {
        if p == 2 {
            texts.push((format!("boolean-8x12-{s}"), boolean(8, 12, s)));
            texts.push((format!("boolean-10x14-{s}"), boolean(10, 14, s)));
            texts.push((format!("cubics-7x9-{s}"), cubics(7, 9, s)));
        } else {
            texts.push((format!("planted-4x6-{s}"), planted(4, 6, p, s)));
            texts.push((format!("planted-6x8-{s}"), planted(6, 8, p, s)));
        }
    }
    for (k, gens) in EDGE.iter().enumerate() {
        texts.push((format!("edge {k}"), format!("x,y,z\n{p}\n{gens}\n")));
    }
    texts
}

/// Test 1, linear-tail-time.1 to .9, .12, .13: every call returns with the
/// inter-reduction what it returns without, over GF(2), 7, 65521,
/// 2^31 − 1, 2^61 − 1, and ℚ, in grevlex, the signatures off and on; the
/// planted systems in deglex and lex as well; with cofactors and the exact
/// checker, which track the cofactors and leave the inter-reduction out; and
/// over GF(2) and GF(65521) some call's inter-reduction changes a member,
/// and over ℚ none runs, as the learn records.
#[test]
fn every_result_stays() {
    for p in std::iter::once(2).chain(FIELDS) {
        let mut changed = 0;
        for (system, text) in systems(p) {
            for signatures in [false, true] {
                let mut params = reported(MonomialOrder::Grevlex);
                params.signatures = signatures;
                let name = format!("{system} over {p}, signatures {signatures}");
                let outcome = pair(&name, &text, &params);
                assert!(outcome.basis.is_some(), "{name}: a basis");
                let lines = tails_lines(&outcome);
                if p == 0 {
                    assert!(lines.is_empty(), "{name}: no inter-reduction in the learn");
                }
                changed += lines.iter().map(|l| count(l, "changed")).sum::<usize>();
            }
            if system.starts_with("planted") {
                for order in [MonomialOrder::Deglex, MonomialOrder::Lex] {
                    let name = format!("{system} over {p} in {order}");
                    let params = reported(order);
                    let outcome = pair(&name, &text, &params);
                    assert!(outcome.basis.is_some(), "{name}: a basis");
                }
            }
            let mut params = reported(MonomialOrder::Grevlex);
            params.cofactors = true;
            params.checker = Some(CheckerMode::Exact);
            if built(&params).is_err() {
                continue;
            }
            let name = format!("{system} over {p} with cofactors");
            let outcome = pair(&name, &text, &params);
            assert!(outcome.cofactors.is_some(), "{name}: the cofactor text");
            assert_eq!(outcome.certified, Some(true), "{name}: certified");
            assert!(
                tails_lines(&outcome).is_empty(),
                "{name}: no inter-reduction where the run tracks cofactors"
            );
        }
        if p == 2 || p == 65521 {
            assert!(changed > 0, "over {p} an inter-reduction changes a member");
        }
    }
}

/// Test 2, linear-tail-time.2, .3, .10: the report's `tails` line comes
/// before the line of its round, its members at least its members of degree
/// 1 and those it changed, its matrix's rows at least its given rows
/// (gf2-tails-time.4, .7), and its time a time; within a run the members of
/// degree 1 grow from line to line.
#[test]
fn tails_line_before_its_round() {
    let mut seen = 0;
    for s in 1..=4 {
        let text = boolean(10, 14, s);
        let params = reported(MonomialOrder::Grevlex);
        let outcome = call(&text, &params, true);
        let lines = report(&outcome);
        let mut last: Option<usize> = None;
        for (k, line) in lines.iter().enumerate() {
            if line.kind == "run" {
                last = None;
            }
            if line.kind != "tails" {
                continue;
            }
            seen += 1;
            let round = line.get("round").unwrap();
            let next = lines[k + 1..]
                .iter()
                .find(|l| l.kind == "round")
                .unwrap_or_else(|| panic!("boolean {s}: a round after tails round={round}"));
            assert_eq!(
                next.get("number"),
                Some(round),
                "boolean {s}: the round after the tails line"
            );
            let (members, linear, changed) = (
                count(line, "members"),
                count(line, "linear"),
                count(line, "changed"),
            );
            assert!(
                linear >= 1 && linear <= members && changed <= members,
                "boolean {s}: members={members} linear={linear} changed={changed}"
            );
            let given = count(line, "given");
            assert!(
                count(line, "rows") >= given,
                "boolean {s}: the matrix's rows"
            );
            assert!(
                given == 0 || count(line, "columns") >= 1,
                "boolean {s}: the matrix's columns"
            );
            assert!(
                count(line, "terms") >= members,
                "boolean {s}: the terms before"
            );
            assert!(
                count(line, "after") >= members,
                "boolean {s}: the terms after"
            );
            let time = line.get("time").unwrap();
            assert!(
                time.len() > 7 && time.split_once('.').is_some_and(|(_, f)| f.len() == 6),
                "boolean {s}: the time {time}"
            );
            if let Some(before) = last {
                assert!(linear > before, "boolean {s}: the members of degree 1 grow");
            }
            last = Some(linear);
        }
        let off = call(&text, &params, false);
        assert!(tails_lines(&off).is_empty(), "boolean {s}: no line off");
    }
    assert!(seen > 0, "a tails line");
}

/// The polynomials `generators` on x, y, z over GF(7) in grevlex, read
/// into `table`.
fn polys(
    field: &Gf31,
    table: &mut crate::table::Table<crate::monomial::Dense16>,
    generators: &str,
) -> Vec<Poly<Gf31, Rank1>> {
    read(field, table, &Grevlex, "x,y,z", generators)
}

/// Test 3, linear-tail-time.6: `Basis::replace` gives element i the new
/// polynomial and its length, and keeps its leading monomial, `G_min`, and
/// the redundant marks.
#[test]
fn replace_keeps_the_rest() {
    let field = Gf31::new(7).unwrap();
    let mut t = table("x,y,z");
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    for (k, g) in polys(&field, &mut t, "x+y+2*z+3,\ny+4*z+1,\nx*y+z")
        .into_iter()
        .enumerate()
    {
        basis.push(g, k == 2).unwrap();
    }
    let new = polys(&field, &mut t, "x+5*z+2").remove(0);
    let (lead, min) = (basis.lead(ElementId::at(0)), basis.min().to_vec());
    basis.replace(ElementId::at(0), new.clone());
    assert_eq!(basis.element(ElementId::at(0)).monomials, new.monomials);
    assert_eq!(basis.element(ElementId::at(0)).coefs, new.coefs);
    assert_eq!(basis.length(ElementId::at(0)), 3, "the length follows");
    assert_eq!(basis.lead(ElementId::at(0)), lead, "the lead stays");
    assert_eq!(basis.min(), min.as_slice(), "G_min stays");
    assert_eq!(basis.len(), 3);
    assert_eq!(basis.length(ElementId::at(1)), 3, "element 1 stays");
}

/// Test 4, linear-tail-time.1, .3 to .5: `tails` on `G_min` = {x + y + 2z + 3,
/// y + 4z + 1, z^2 + x + y + 1} over GF(7) in grevlex, of 2 members of
/// degree 1, every member joined since the run began: with 0 before, the
/// members are x + 5z + 2, y + 4z + 1, and z^2 + 5z + 5, worked by hand;
/// with 2 before, nothing changes; each leaves 2 members of degree 1 as the
/// last inter-reduction's (gf2-tails-time.1). A redundant element stays as
/// it is.
#[test]
fn tails_by_hand() {
    let field = Gf31::new(7).unwrap();
    for before in [0, 2] {
        let mut t = table("x,y,z");
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        let given = polys(&field, &mut t, "x+y+2*z+3,\ny+4*z+1,\nz^2+x+y+1,\nx*y+x+y");
        for (k, g) in given.iter().cloned().enumerate() {
            basis.push(g, k == 3).unwrap();
        }
        let mut bufs: Buffers<Gf31> = Buffers::new();
        let mut budget = Budget::unbounded();
        let mut last = Tails {
            linear: before,
            since: 0,
        };
        tails(
            &field,
            &mut t,
            &Grevlex,
            &mut basis,
            &mut NoCofactors,
            &mut bufs,
            &mut last,
            1,
            &mut budget,
        )
        .unwrap();
        assert_eq!(last.linear, 2, "two members of degree 1, before {before}");
        let expected = if before == 0 {
            polys(&field, &mut t, "x+5*z+2,\ny+4*z+1,\nz^2+5*z+5,\nx*y+x+y")
        } else {
            given
        };
        for (k, e) in expected.iter().enumerate() {
            let g = basis.element(ElementId::at(k));
            assert_eq!(g.monomials, e.monomials, "element {k}, before {before}");
            assert_eq!(
                g.coefs.iter().map(|&c| field.value(c)).collect::<Vec<_>>(),
                e.coefs.iter().map(|&c| field.value(c)).collect::<Vec<_>>(),
                "element {k}'s coefficients, before {before}"
            );
            assert_eq!(basis.length(ElementId::at(k)), e.len());
        }
    }
}
