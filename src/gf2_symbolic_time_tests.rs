//! gf2-symbolic-time, the item's test code: past `FILED_PAST` variables the
//! divisor search over `G_min`'s members filed by a bit of their leads'
//! supports answers as a plain scan of `G_min` in the order made, and so it
//! does where the test profile forces the files or the scan; the constant's
//! file and an empty `G_min`; a call returns what it returns with the scan;
//! and a build whose filed searches find no divisor spends them as it goes
//! (`iterations/0176-gf2-symbolic-time/plan.md`, _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::build_tests::{SIZES, poly};
use crate::call::built;
use crate::call_stop_plain_tests::MEMBERS;
use crate::call_stop_sites_tests::spends_as_it_goes;
use crate::divisors::{Divisors, FILED_FORCED, FILED_PAST};
use crate::field::{Field, Gf31};
use crate::gf2_update_rest_time_tests::{spread, systems_on};
use crate::halt::Halt;
use crate::id::{ElementId, MonoId, RowSource};
use crate::linear_substitution_time_tests::untimed;
use crate::mask_scan_tests::rule;
use crate::matrix::{self, Buffers};
use crate::module::Rank1;
use crate::monomial::{Dense4, Dense8, Dense16, Format, SUPPORT_BITS};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::{below, rng};
use crate::split_piece_time_tests::EDGE;
use crate::systems::placed;
use crate::table::Table;
use crate::{MonomialOrder, Outcome, Params};

/// `f` with `FILED_FORCED` at `forced`, and the forcing lifted after it.
fn forced<T>(forced: Option<bool>, f: impl FnOnce() -> T) -> T {
    FILED_FORCED.set(forced);
    let got = f();
    FILED_FORCED.set(None);
    got
}

/// A lead on n variables in the shape a system over GF(2) gives its leads:
/// a product of 2 to 4 distinct variables; in a draw of eight the square of
/// one, as the field equations' leads stand; in another of eight a product
/// with one exponent of 2 or 3. Half the variables come from the first
/// `hot`, so that leads share them. Its degree is at most 6, which `Dense4`
/// holds with a variable more.
fn boolean_lead(r: &mut Xoshiro256PlusPlus, n: usize, hot: usize) -> Vec<u32> {
    let pick = |r: &mut Xoshiro256PlusPlus| {
        if below(r, 2) == 0 {
            below(r, hot as u64) as usize
        } else {
            below(r, n as u64) as usize
        }
    };
    let mut e = vec![0u32; n];
    match below(r, 8) {
        0 => e[pick(r)] = 2,
        draw => {
            let degree = 2 + below(r, 3) as usize;
            let mut set = 0;
            while set < degree {
                let v = pick(r);
                if e[v] == 0 {
                    e[v] = 1;
                    set += 1;
                }
            }
            if draw == 1 {
                let v = (0..n).find(|&v| e[v] > 0).unwrap();
                e[v] = 2 + below(r, 2) as u32;
            }
        }
    }
    e
}

/// The variable whose bit of `SUPPORT_BITS` folds onto v's, past
/// `SUPPORT_BITS` variables (`Format::support`), None where none on n.
fn folded_onto(v: usize, n: usize) -> Option<usize> {
    let w = v + SUPPORT_BITS;
    if w < n {
        Some(w)
    } else {
        v.checked_sub(SUPPORT_BITS)
    }
}

/// The bits set in x.
fn bits(x: u128) -> usize {
    (0..u128::BITS).filter(|&b| x >> b & 1 == 1).count()
}

/// What the queries of a run of test 1 met, which the test asserts are not
/// all zero (plan.md, _Test code_, 1).
#[derive(Default, Debug)]
struct Met {
    /// Queries with no divisor.
    none: u32,
    /// Queries a member's support lies in, by variable, whose lead does
    /// not divide them.
    support_alone: u32,
    /// Queries a member's support lies in only by its bits folded past
    /// `SUPPORT_BITS` variables.
    folded: u32,
    /// Queries whose answer ties in terms with a later divisor.
    tied: u32,
    /// Pushes that took an element out of `G_min`.
    left: u32,
}

/// Test 1 for one format on n variables under `filed`, `FILED_FORCED`'s
/// value: for each count of `SIZES`, 200, and 600, a basis pushed by
/// `Basis::push_min` of `boolean_lead`'s leads, every fifth a lead of
/// `G_min` with an exponent less, so that it takes the member out, each of
/// 1 to 4 terms by `build_tests::poly`; `Divisors::find` answers each query
/// as `mask_scan_tests::rule`. The queries are each lead times a variable,
/// each lead with its exponents past 1 lowered to 1, each lead with one
/// variable moved onto the one its support bit folds onto, and 200 products
/// of up to 6 distinct variables. Each search spends a unit per file it
/// visits or per block of masks it scans at least.
fn search_agrees<M: Format>(n: usize, filed: Option<bool>, seed: u64) -> Met {
    let field = Gf31::new(65521).unwrap();
    let mut met = Met::default();
    let hot = n.min(24);
    for count in SIZES.into_iter().chain([200, 600]) {
        let case = format!("{n} variables, {count} elements, files {filed:?}");
        let mut r = rng(seed ^ (count as u64) << 8);
        let mut t: Table<M> = Table::new(M::new(n), &mut r);
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        let mut elements: Vec<(Vec<u32>, usize)> = Vec::new();
        for k in 0..count {
            let min = basis.min().to_vec();
            let mut l = boolean_lead(&mut r, n, hot);
            if k % 5 == 4 && !min.is_empty() {
                let e = &elements[min[below(&mut r, min.len() as u64) as usize].index()].0;
                if e.iter().sum::<u32>() >= 3 {
                    l.clone_from(e);
                    let v = (0..n).find(|&v| l[v] > 0).unwrap();
                    l[v] -= 1;
                }
            }
            let terms = 1 + below(&mut r, 4) as usize;
            let g = poly(&mut r, &mut t, &field, &l, terms);
            let before = basis.min().len();
            let e = basis.push_min(&t, g, &mut Budget::unbounded()).unwrap().id;
            let entered = usize::from(basis.min().last() == Some(&e));
            met.left += u32::from(basis.min().len() < before + entered);
            elements.push((l, terms));
        }
        let min = basis.min().to_vec();
        let mut qs: Vec<Vec<u32>> = Vec::new();
        for (l, _) in &elements {
            let mut q = l.clone();
            q[below(&mut r, n as u64) as usize] += 1;
            qs.push(q);
            if l.iter().any(|&x| x > 1) {
                qs.push(l.iter().map(|&x| x.min(1)).collect());
            }
            if let Some((v, w)) = (0..n)
                .filter(|&v| l[v] > 0)
                .find_map(|v| folded_onto(v, n).filter(|&w| l[w] == 0).map(|w| (v, w)))
            {
                let mut q = l.clone();
                q[w] = q[v];
                q[v] = 0;
                qs.push(q);
            }
        }
        for _ in 0..200 {
            let mut q = vec![0u32; n];
            for _ in 0..below(&mut r, 7) {
                q[below(&mut r, n as u64) as usize] = 1;
            }
            qs.push(q);
        }
        qs.retain(|q| q.iter().sum::<u32>() <= M::MAX_DEGREE);
        let ids: Vec<MonoId> = qs
            .iter()
            .map(|q| t.intern(q, &mut Budget::unbounded()).unwrap())
            .collect();
        let divisors = forced(filed, || {
            Divisors::new(&t, &basis, &mut Budget::unbounded())
        })
        .unwrap();
        for (q, &u) in qs.iter().zip(&ids) {
            let want = rule(&min, &elements, q);
            let (got, units) = divisors.find(&t, u);
            assert_eq!(got, want, "{case}: {q:?}");
            // A filed search visits a file per bit of the query's support
            // and the constant's, the scan every block of eight masks.
            let least = if filed.unwrap_or(n > FILED_PAST) {
                1 + bits(t.format().support(t.monomial(u)))
            } else {
                min.len().div_ceil(8)
            };
            assert!(
                units >= least,
                "{case}: {q:?} spent {units} units, not {least}"
            );
            let holds = |l: &[u32]| l.iter().zip(q).all(|(&a, &b)| a == 0 || b > 0);
            let folds = |l: &[u32]| {
                (0..n).all(|v| l[v] == 0 || q[v] > 0 || folded_onto(v, n).is_some_and(|w| q[w] > 0))
            };
            met.none += u32::from(want.is_none());
            met.support_alone += u32::from(min.iter().any(|e| {
                let l = &elements[e.index()].0;
                holds(l) && Some(*e) != want && !l.iter().zip(q).all(|(a, b)| a <= b)
            }));
            met.folded += u32::from(min.iter().any(|e| {
                let l = &elements[e.index()].0;
                folds(l) && !holds(l)
            }));
            if let Some(w) = want {
                let terms = elements[w.index()].1;
                met.tied += u32::from(min.iter().any(|e| {
                    let (l, k) = &elements[e.index()];
                    e.index() > w.index() && *k == terms && l.iter().zip(q).all(|(a, b)| a <= b)
                }));
            }
        }
    }
    met
}

/// Test 1, gf2-symbolic-time.1 to .6: past `FILED_PAST` variables, on
/// `Dense4` and `Dense8` to 300 variables, past `SUPPORT_BITS` among them,
/// `Divisors::find` answers every query as a plain scan of `G_min` by the
/// rule; and so it does with the files forced below `FILED_PAST` and the
/// scan forced past it.
#[test]
fn filed_searches_answer_as_the_rule() {
    assert_eq!(FILED_PAST, 64, "the files start past 64 variables");
    let runs = [
        ("Dense4 on 65", search_agrees::<Dense4>(65, None, 0x5f10)),
        ("Dense4 on 100", search_agrees::<Dense4>(100, None, 0x5f11)),
        ("Dense4 on 128", search_agrees::<Dense4>(128, None, 0x5f12)),
        ("Dense4 on 129", search_agrees::<Dense4>(129, None, 0x5f13)),
        ("Dense4 on 300", search_agrees::<Dense4>(300, None, 0x5f14)),
        ("Dense8 on 65", search_agrees::<Dense8>(65, None, 0x5f15)),
        ("Dense8 on 200", search_agrees::<Dense8>(200, None, 0x5f16)),
        (
            "Dense8 on 9 filed",
            search_agrees::<Dense8>(9, Some(true), 0x5f17),
        ),
        (
            "Dense8 on 33 filed",
            search_agrees::<Dense8>(33, Some(true), 0x5f18),
        ),
        (
            "Dense4 on 64 filed",
            search_agrees::<Dense4>(64, Some(true), 0x5f19),
        ),
        (
            "Dense4 on 129 scanned",
            search_agrees::<Dense4>(129, Some(false), 0x5f1a),
        ),
    ];
    for (name, met) in runs {
        assert!(met.none > 0, "{name}: every query had a divisor: {met:?}");
        assert!(
            met.support_alone > 0,
            "{name}: no support held a query that its lead does not divide: {met:?}"
        );
        assert!(
            met.tied > 0,
            "{name}: no answer tied with a later divisor: {met:?}"
        );
        assert!(met.left > 0, "{name}: no element left G_min: {met:?}");
        if name.contains(" 129") || name.contains(" 200") || name.contains(" 300") {
            assert!(
                met.folded > 0,
                "{name}: no support held a query by folded bits alone: {met:?}"
            );
        }
    }
}

/// The element of one term, the monomial `exponents`, pushed into `basis`
/// as made, not by `push_min`.
fn push_monomial(
    t: &mut Table<Dense4>,
    basis: &mut Basis<Gf31, Rank1>,
    exponents: &[u32],
) -> ElementId {
    let field = Gf31::new(65521).unwrap();
    let m = t.intern(exponents, &mut Budget::unbounded()).unwrap();
    let g = Poly {
        monomials: vec![m],
        coefs: vec![field.coef(1)],
        position: (),
    };
    basis.push(g, false).unwrap()
}

/// Test 2, gf2-symbolic-time.1 and .3: on 70 and 300 variables, and on 9
/// with the files forced, a search among `x0·x1`, the constant, and `x2`,
/// each of one term in the order made, takes the least of them in the
/// order that divides the query, the constant for a query no other
/// divides, the unit monomial among them; the constant alone answers every
/// query; and no member answers none.
#[test]
fn the_constant_and_an_empty_g_min() {
    for (n, filed) in [(70usize, None), (300, None), (9, Some(true))] {
        let mut t: Table<Dense4> = Table::new(Dense4::new(n), &mut rng(0x5f20));
        let mut basis: Basis<Gf31, Rank1> = Basis::new();
        let mono = |vars: &[usize]| {
            let mut e = vec![0u32; n];
            for &v in vars {
                e[v] += 1;
            }
            e
        };
        let x01 = push_monomial(&mut t, &mut basis, &mono(&[0, 1]));
        let one = push_monomial(&mut t, &mut basis, &mono(&[]));
        let x2 = push_monomial(&mut t, &mut basis, &mono(&[2]));
        let queries = [
            (mono(&[0, 1, 2]), Some(x01)),
            (mono(&[0, 1]), Some(x01)),
            (mono(&[2, 5]), Some(one)),
            (mono(&[0, 2]), Some(one)),
            (mono(&[]), Some(one)),
            (mono(&[n - 1]), Some(one)),
        ];
        let all = forced(filed, || {
            Divisors::of(&t, &basis, vec![x01, one, x2], &mut Budget::unbounded())
        })
        .unwrap();
        let alone = forced(filed, || {
            Divisors::of(&t, &basis, vec![one], &mut Budget::unbounded())
        })
        .unwrap();
        let without = forced(filed, || {
            Divisors::of(&t, &basis, vec![x01, x2], &mut Budget::unbounded())
        })
        .unwrap();
        let none = forced(filed, || {
            Divisors::of(&t, &basis, Vec::new(), &mut Budget::unbounded())
        })
        .unwrap();
        for (q, want) in &queries {
            let u = t.intern(q, &mut Budget::unbounded()).unwrap();
            let case = format!("{n} variables, files {filed:?}, query {q:?}");
            assert_eq!(all.find(&t, u).0, *want, "{case}: of all three");
            assert_eq!(alone.find(&t, u).0, Some(one), "{case}: of the constant");
            let others = [x01, x2].into_iter().find(|&e| {
                let l = t.monomial(basis.lead(e)).to_vec();
                t.format().divides(&l, t.monomial(u))
            });
            assert_eq!(without.find(&t, u).0, others, "{case}: of the two others");
            assert_eq!(none.find(&t, u).0, None, "{case}: of none");
        }
    }
}

/// The call of `text` with the files forced to `filed`.
fn call_filed(text: &str, params: &Params, filed: bool) -> Outcome {
    forced(Some(filed), || {
        crate::groebner(text, params).unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
    })
}

/// Test 3, gf2-symbolic-time.1 to .6: calls over GF(2), GF(7), GF(65521),
/// and ℚ on 65 to 300 variables, on gf2-update-rest-time's systems
/// (`gf2_update_rest_time_tests::systems_on`) and the edge cases of
/// `OBJECTIVES.md` over each field placed on 70 variables, in grevlex and
/// lex, the signatures off and on: with the divisor search over the files a
/// call returns the basis and the report, but its times, that it returns
/// with the scan.
#[test]
fn calls_return_what_they_return_with_the_scan() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for p in [2u64, 7, 65521, 0] {
        for (k, g) in EDGE.iter().enumerate() {
            let text = placed(&format!("x,y,z\n{p}\n{g}\n"), &spread(3, 70), 70);
            texts.push((format!("edge {k} over {p} on 70"), text));
        }
    }
    for n in [65, 129, 300] {
        texts.extend(systems_on(n));
    }
    for (name, text) in &texts {
        for order in [MonomialOrder::Grevlex, MonomialOrder::Lex] {
            for signatures in [false, true] {
                let mut params = reported(order.clone());
                params.signatures = signatures;
                if built(&params).is_err() {
                    continue;
                }
                let case = format!("{name}, {order:?}, signatures {signatures}");
                let scanned = call_filed(text, &params, false);
                let filed = call_filed(text, &params, true);
                assert_eq!(filed.basis, scanned.basis, "{case}: the basis");
                assert_eq!(untimed(&filed), untimed(&scanned), "{case}: the report");
            }
        }
    }
}

/// The variables of test 4's build: past `FILED_PAST`.
const N: usize = 70;

/// Test 4's build under `budget`: a basis of `MEMBERS` elements on `N`
/// variables, each a product of 3 distinct variables of the first `N - 1`
/// and three tail terms of degree 2 or less in them, and given rows of
/// every member times 1 and times `x_{N-1}`. A tail product holds at most
/// two of the first `N - 1` variables, so no lead of degree 3 there divides
/// it, and every divisor search of the build finds none. The rows the
/// build made, or its halt.
fn divisor_free(budget: &mut Budget) -> Result<usize, Halt> {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(N), &mut rng(0x5f40));
    let mut u = Budget::unbounded();
    let mut r = rng(0x5f41);
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    let mut leads: Vec<Vec<u32>> = Vec::new();
    while leads.len() < MEMBERS {
        let mut e = vec![0u32; N];
        let mut set = 0;
        while set < 3 {
            let v = below(&mut r, N as u64 - 1) as usize;
            if e[v] == 0 {
                e[v] = 1;
                set += 1;
            }
        }
        if !leads.contains(&e) {
            leads.push(e);
        }
    }
    for lead in &leads {
        let mut terms = vec![(table.intern(lead, &mut u).unwrap(), field.coef(1))];
        while terms.len() < 4 {
            let mut tail = vec![0u32; N];
            for _ in 0..below(&mut r, 3) {
                tail[below(&mut r, N as u64 - 1) as usize] += 1;
            }
            let m = table.intern(&tail, &mut u).unwrap();
            if terms.iter().all(|&(t, _)| t != m) {
                terms.push((m, field.coef(1 + below(&mut r, 65520))));
            }
        }
        let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
        basis.push_min(&table, g, &mut u).unwrap();
    }
    let mut rows = Vec::new();
    for j in 0..2 {
        let mut power = vec![0u32; N];
        power[N - 1] = j;
        let multiplier = table.intern(&power, &mut u).unwrap();
        rows.extend(basis.min().iter().map(|&element| RowSource {
            element,
            multiplier,
        }));
    }
    let m = matrix::build(
        &mut table,
        &basis,
        &Grevlex,
        rows,
        &mut Buffers::new(),
        budget,
    )?;
    Ok(m.rows.len())
}

/// Test 4, gf2-symbolic-time.1 and call-stop-latency.14: a build past
/// `FILED_PAST` variables whose filed divisor searches find none, over a
/// basis of `MEMBERS` members, spends its searches as it goes, and stops
/// inside under a passed deadline.
#[test]
fn a_filed_search_spends() {
    let given = 2 * MEMBERS;
    let rows = divisor_free(&mut Budget::unbounded()).unwrap();
    assert_eq!(rows, given, "a search found a divisor");
    let build = |b: &mut Budget| -> Result<usize, Stop> {
        match divisor_free(b) {
            Ok(rows) => Ok(rows),
            Err(Halt::Stop(s)) => Err(s),
            Err(
                halt @ (Halt::Overflow
                | Halt::Size
                | Halt::Misfit(_)
                | Halt::Handover
                | Halt::Outworked),
            ) => panic!("the build: {halt:?}"),
        }
    };
    if let Err(e) = spends_as_it_goes(Site::Divisors, 1 << 10, 4, build) {
        panic!("{e}");
    }
}
