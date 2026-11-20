//! gf2-tails-time, the item's test code: the plain run's inter-reductions
//! between its rounds wait, where the rounds reduce in packed rows, until its
//! members of degree 1 have grown by a quarter; they take as given rows the
//! members that hold a tail term a member joined since reduces, and search
//! no divisor for a tail term no such member's leading monomial divides;
//! every result stays as it was, over GF(2), GF(7), and GF(65521)
//! (`iterations/0174-gf2-tails-time/plan.md`, _Test code_, 1 to 4).
//!
//! A pair of calls is the same call with `TAILS_SPARE` true and false.

use crate::budget::Budget;
use crate::call::built;
use crate::cofactors::NoCofactors;
use crate::f4::{TAILS_GROWTH_PART, TAILS_SPARE, Tails, tails};
use crate::field::{Field, Gf2, Gf31};
use crate::gf2_input_time_tests::sparse_pivots;
use crate::id::ElementId;
use crate::input_redundancy_tests::{read, table};
use crate::linear_tail_time_tests::{EDGE, count, planted, tails_lines};
use crate::matrix::Buffers;
use crate::module::Rank1;
use crate::order::Grevlex;
use crate::poly::Basis;
use crate::rebuilt_reducer_tests::reported;
use crate::report_tests::Line;
use crate::shared_tests::report;
use crate::systems::{boolean, cubics};
use crate::{CheckerMode, MonomialOrder, Outcome, Params};

/// The fields of the tests: GF(2), whose rounds reduce in packed rows, and
/// two whose rounds do not.
const PRIMES: [u64; 3] = [2, 7, 65521];

/// The call of `text` with the spared inter-reductions on or off.
fn call(text: &str, params: &Params, spare: bool) -> Outcome {
    TAILS_SPARE.set(spare);
    let outcome = crate::groebner(text, params);
    TAILS_SPARE.set(true);
    outcome.unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
}

/// x1 − 2, x1^2 − x2, …, x(n−1)^2 − xn over the characteristic p: a basis of
/// degree 1 whose plain run makes one member of degree 1 a round.
fn squares(n: usize, p: u64) -> String {
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let mut generators = vec!["x1-2".to_string()];
    generators.extend((1..n).map(|k| format!("x{k}^2-x{next}", next = k + 1)));
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// The systems over the characteristic p, each with its name: over GF(2)
/// boolean and cubic systems with a planted point, over every other field
/// planted systems, over each systems whose linear generators hold their
/// pivots sparsely, chains of squares, and the edge cases over x, y, z.
fn systems(p: u64) -> Vec<(String, String)> {
    let mut texts = Vec::new();
    for s in 1..=3 {
        if p == 2 {
            for (n, m) in [(8, 12), (10, 14), (12, 16)] {
                texts.push((format!("boolean({n}, {m}, {s})"), boolean(n, m, s)));
            }
            texts.push((format!("cubics(7, 9, {s})"), cubics(7, 9, s)));
        } else {
            for (n, m) in [(4, 6), (6, 8), (8, 10)] {
                texts.push((format!("planted({n}, {m}, {s})"), planted(n, m, p, s)));
            }
        }
    }
    for (seed, (n, k)) in (0x7a10u64..).zip([(8, 2), (10, 3), (12, 5)]) {
        let q = 2 * n;
        texts.push((
            format!("sparse_pivots({n}, {k}, {q}, {p}, {seed:#x})"),
            sparse_pivots(n, k, q, p, seed),
        ));
    }
    for n in [8, 12] {
        texts.push((format!("squares({n})"), squares(n, p)));
    }
    for (k, gens) in EDGE.iter().enumerate() {
        texts.push((format!("edge {k}"), format!("x,y,z\n{p}\n{gens}\n")));
    }
    texts
}

/// The calls of test 1 to 3 on `text`: grevlex with the signatures off and
/// on, then deglex and lex, each with its name.
fn calls(system: &str, p: u64) -> Vec<(String, Params)> {
    let mut out = Vec::new();
    for signatures in [false, true] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.signatures = signatures;
        out.push((
            format!("{system} over {p}, signatures {signatures}"),
            params,
        ));
    }
    for order in [MonomialOrder::Deglex, MonomialOrder::Lex] {
        out.push((format!("{system} over {p} in {order}"), reported(order)));
    }
    out
}

/// The `tails` lines of a report, one list per run: each run's lines from
/// its `run` line to the next.
fn per_run(outcome: &Outcome) -> Vec<Vec<Line>> {
    let mut runs: Vec<Vec<Line>> = Vec::new();
    for line in report(outcome) {
        if line.kind == "run" {
            runs.push(Vec::new());
        } else if line.kind == "tails" {
            runs.last_mut()
                .expect("a tails line within a run")
                .push(line);
        }
    }
    runs
}

/// Whether the members of degree 1 grow from `before` to `after` by at
/// least a quarter of `before`, rounded up.
fn by_a_quarter(before: usize, after: usize) -> bool {
    after >= before + before.div_ceil(TAILS_GROWTH_PART)
}

/// Test 1, gf2-tails-time.1 to .6: every call returns with the spared
/// inter-reductions what it returns without, over GF(2), GF(7), and
/// GF(65521), in grevlex with the signatures off and on, in deglex, and in
/// lex, and the basis the call with cofactors and the exact checker returns,
/// which the checker certifies and which runs no inter-reduction between
/// the rounds; over each field some inter-reduction changes a member.
#[test]
fn every_result_stays() {
    for p in PRIMES {
        let mut changed = 0;
        for (system, text) in systems(p) {
            for (name, params) in calls(&system, p) {
                let on = call(&text, &params, true);
                let off = call(&text, &params, false);
                assert_eq!(on.basis, off.basis, "{name}: the bases");
                assert_eq!(on.stopped, off.stopped, "{name}: the stops");
                assert_eq!(on.certified, off.certified, "{name}: the verdicts");
                assert_eq!(on.flaw, off.flaw, "{name}: the flaws");
                assert!(on.basis.is_some(), "{name}: a basis");
                changed += tails_lines(&on)
                    .iter()
                    .map(|l| count(l, "changed"))
                    .sum::<usize>();
            }
            let mut params = reported(MonomialOrder::Grevlex);
            params.cofactors = true;
            params.checker = Some(CheckerMode::Exact);
            if built(&params).is_err() {
                continue;
            }
            let name = format!("{system} over {p} with cofactors");
            let checked = call(&text, &params, true);
            assert_eq!(checked.certified, Some(true), "{name}: certified");
            assert!(
                tails_lines(&checked).is_empty(),
                "{name}: no inter-reduction where the run tracks cofactors"
            );
            let plain = call(&text, &reported(MonomialOrder::Grevlex), true);
            assert_eq!(plain.basis, checked.basis, "{name}: the certified basis");
        }
        assert!(changed > 0, "over {p} an inter-reduction changes a member");
    }
}

/// Test 2, gf2-tails-time.1 to .3: over GF(2) each run's members of degree
/// 1 grow by a quarter or more from one `tails` line to the next, where the
/// calls without the spare hold a step that grows by less, and the spared
/// calls write fewer lines; over GF(7) and GF(65521) each run's lines name
/// the members of degree 1 the run without the spare names, and some step
/// grows by less than a quarter.
#[test]
fn the_trigger_waits_for_a_quarter_more() {
    for p in PRIMES {
        let (mut less, mut fewer) = (false, false);
        for (system, text) in systems(p) {
            for (name, params) in calls(&system, p) {
                let on = per_run(&call(&text, &params, true));
                let off = per_run(&call(&text, &params, false));
                let linear = |runs: &[Vec<Line>]| -> Vec<Vec<usize>> {
                    runs.iter()
                        .map(|r| r.iter().map(|l| count(l, "linear")).collect())
                        .collect()
                };
                let (on, off) = (linear(&on), linear(&off));
                let below = |runs: &[Vec<usize>]| {
                    runs.iter()
                        .any(|r| r.windows(2).any(|w| !by_a_quarter(w[0], w[1])))
                };
                if p == 2 {
                    assert!(!below(&on), "{name}: every step by a quarter, {on:?}");
                    less |= below(&off);
                    let lines = |runs: &[Vec<usize>]| runs.iter().map(Vec::len).sum::<usize>();
                    fewer |= lines(&on) < lines(&off);
                } else {
                    assert_eq!(on, off, "{name}: the members of degree 1 at each line");
                    less |= below(&on);
                }
            }
        }
        assert!(less, "over {p} a step by less than a quarter");
        assert!(fewer || p != 2, "over {p} the spare writes fewer lines");
    }
}

/// Test 3, gf2-tails-time.4, .5, .7: every `tails` line of a spared call
/// names at most its members as given rows and at most its given rows as
/// changed, its matrix holds its given rows, and a line of no given rows
/// names no rows and no columns; without the spare every member is a given
/// row; over each field some spared line takes fewer given rows than
/// members.
#[test]
fn the_given_rows_hold_a_reducible_term() {
    for p in PRIMES {
        let mut fewer = false;
        for (system, text) in systems(p) {
            for (name, params) in calls(&system, p) {
                for line in tails_lines(&call(&text, &params, true)) {
                    let (members, given, changed) = (
                        count(&line, "members"),
                        count(&line, "given"),
                        count(&line, "changed"),
                    );
                    let (rows, columns) = (count(&line, "rows"), count(&line, "columns"));
                    assert!(
                        given <= members && changed <= given && rows >= given,
                        "{name}: members={members} given={given} changed={changed} rows={rows}"
                    );
                    if given == 0 {
                        assert_eq!((rows, columns), (0, 0), "{name}: no matrix");
                    } else {
                        assert!(columns >= 1, "{name}: the matrix's columns");
                    }
                    fewer |= given < members;
                }
                for line in tails_lines(&call(&text, &params, false)) {
                    assert_eq!(
                        count(&line, "given"),
                        count(&line, "members"),
                        "{name}: every member a given row without the spare"
                    );
                }
            }
        }
        assert!(fewer, "over {p} a line of fewer given rows than members");
    }
}

/// The basis of the polynomials `generators` on `names` over `field` in
/// grevlex, read into `t`, every one a member of `G_min`.
fn members<F: Field>(
    field: &F,
    t: &mut crate::table::Table<crate::monomial::Dense16>,
    names: &str,
    generators: &str,
) -> Basis<F, Rank1> {
    let mut basis = Basis::new();
    for g in read(field, t, &Grevlex, names, generators) {
        basis.push(g, false).unwrap();
    }
    basis
}

/// `tails` on `basis` after `last`, the spare `spare`, in round 1.
fn tails_of<F: Field>(
    field: &F,
    t: &mut crate::table::Table<crate::monomial::Dense16>,
    basis: &mut Basis<F, Rank1>,
    last: &mut Tails,
    spare: bool,
) {
    TAILS_SPARE.set(spare);
    let done = tails(
        field,
        t,
        &Grevlex,
        basis,
        &mut NoCofactors,
        &mut Buffers::new(),
        last,
        1,
        &mut Budget::unbounded(),
    );
    TAILS_SPARE.set(true);
    done.unwrap();
}

/// Asserts that the members of `basis` are `expected`, read on `names` into
/// `t`, term by term and coefficient by coefficient.
fn assert_members<F: Field>(
    field: &F,
    t: &mut crate::table::Table<crate::monomial::Dense16>,
    basis: &Basis<F, Rank1>,
    names: &str,
    expected: &str,
    case: &str,
) {
    for (k, e) in read(field, t, &Grevlex, names, expected).iter().enumerate() {
        let g = basis.element(ElementId::at(k));
        assert_eq!(g.monomials, e.monomials, "{case}: member {k}'s monomials");
        assert_eq!(
            g.coefs.iter().map(|&c| field.value(c)).collect::<Vec<_>>(),
            e.coefs.iter().map(|&c| field.value(c)).collect::<Vec<_>>(),
            "{case}: member {k}'s coefficients"
        );
        assert_eq!(
            basis.length(ElementId::at(k)),
            e.len(),
            "{case}: member {k}'s length"
        );
    }
}

/// Test 4, gf2-tails-time.1, .4, .5: `tails` on bases written out, worked by
/// hand. Over GF(7) on x, y, z, after `Tails { linear: 1, since: 2 }`, `G_min` =
/// {x + 2y + 3, z^2 + 4z + 1, y + 5z + 6}: the first member alone holds a
/// term, y, that the member joined since divides, and becomes x + 4z + 5;
/// the others stay, and the state is `Tails { linear: 2, since: 3 }`, with
/// the spare and without. On x1, …, x8, `G_min` = {x7·x8 + x6 + 1, x1 + x8,
/// …, x5 + x8, x6 + x8}, after `Tails { linear: 5, since: 6 }`: over GF(2),
/// six members of degree 1 are fewer than five and a quarter, rounded up,
/// and nothing changes, the state among it; without the spare, and after
/// four members of degree 1, the first member becomes x7·x8 + x8 + 1; over
/// GF(7) it becomes x7·x8 + 6·x8 + 1; the state then `Tails { linear: 6,
/// since: 7 }`.
#[test]
fn tails_by_hand() {
    let xyz = "x,y,z";
    let f7 = Gf31::new(7).unwrap();
    for spare in [true, false] {
        let mut t = table(xyz);
        let mut basis = members(&f7, &mut t, xyz, "x+2*y+3,\nz^2+4*z+1,\ny+5*z+6");
        let mut last = Tails {
            linear: 1,
            since: 2,
        };
        tails_of(&f7, &mut t, &mut basis, &mut last, spare);
        let case = format!("x, y, z over 7, spare {spare}");
        assert_members(
            &f7,
            &mut t,
            &basis,
            xyz,
            "x+4*z+5,\nz^2+4*z+1,\ny+5*z+6",
            &case,
        );
        assert_eq!(
            last,
            Tails {
                linear: 2,
                since: 3
            },
            "{case}: the state"
        );
    }

    let names = "x1,x2,x3,x4,x5,x6,x7,x8";
    let given = "x7*x8+x6+1,\nx1+x8,\nx2+x8,\nx3+x8,\nx4+x8,\nx5+x8,\nx6+x8";
    let rest = "x1+x8,\nx2+x8,\nx3+x8,\nx4+x8,\nx5+x8,\nx6+x8";
    let done = Tails {
        linear: 6,
        since: 7,
    };
    let f2 = Gf2::new();
    for (before, spare, changes) in [(5, true, false), (5, false, true), (4, true, true)] {
        let mut t = table(names);
        let mut basis = members(&f2, &mut t, names, given);
        let start = Tails {
            linear: before,
            since: 6,
        };
        let mut last = start;
        tails_of(&f2, &mut t, &mut basis, &mut last, spare);
        let case = format!("x1, …, x8 over 2 after {before}, spare {spare}");
        let (expected, state) = if changes {
            (format!("x7*x8+x8+1,\n{rest}"), done)
        } else {
            (given.to_string(), start)
        };
        assert_members(&f2, &mut t, &basis, names, &expected, &case);
        assert_eq!(last, state, "{case}: the state");
    }
    let mut t = table(names);
    let mut basis = members(&f7, &mut t, names, given);
    let mut last = Tails {
        linear: 5,
        since: 6,
    };
    tails_of(&f7, &mut t, &mut basis, &mut last, true);
    let case = "x1, …, x8 over 7 after 5";
    assert_members(
        &f7,
        &mut t,
        &basis,
        names,
        &format!("x7*x8+6*x8+1,\n{rest}"),
        case,
    );
    assert_eq!(last, done, "{case}: the state");
}
