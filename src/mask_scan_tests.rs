//! mask-scan-time, the crate's test code: the divisor search of symbolic
//! preprocessing against a plain scan of `G_min` in the order made, and the
//! reducers `matrix::build` adds against the same rule
//! (`iterations/0073-mask-scan-time/plan.md`, _Test code_, 1 and 2).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::band_tests::budget;
use crate::build_tests::{SIZES, below, divides, exponents, lead, poly, queries, rng};
use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId};
use crate::matrix::{self, Divisors};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::{Grevlex, Lex, Order};
use crate::poly::{Basis, Poly};
use crate::table::Table;
use crate::trace::RowSource;

/// How a case draws its elements' terms (plan.md, _Edge cases_).
#[derive(Clone, Copy, Debug)]
enum Lengths {
    /// From 1 to n + 2 terms, the most `build_tests::poly` makes on n
    /// variables.
    Drawn,
    /// Two terms each: every divisor ties with every other.
    Two,
    /// From n + 2 terms down to 1 in the order made: the order of terms
    /// reverses the order made.
    Decreasing,
}

const LENGTHS: [Lengths; 3] = [Lengths::Drawn, Lengths::Two, Lengths::Decreasing];

/// The counts a run of test 1 met, which the test asserts are not all
/// zero (plan.md, _Test code_, 1).
#[derive(Default)]
struct Met {
    /// Queries whose answer is not the first divisor in the order made.
    reordered: u32,
    /// Queries whose answer ties in terms with a divisor made after it.
    tied: u32,
    /// Pushes that took an element out of `G_min`.
    left: u32,
    /// Queries where an element before the answer in the order of terms
    /// passes the mask and does not divide.
    passed_before: u32,
    /// Queries with no divisor where a mask passes.
    passed_alone: u32,
}

/// A case of test 1: a table, a basis in it, and per element, in the order
/// made, its leading monomial's exponents and its terms.
struct Case<M: Format> {
    table: Table<M>,
    basis: Basis<Gf31, Rank1>,
    elements: Vec<(Vec<u32>, usize)>,
    /// The pushes that took an element out of `G_min`.
    left: u32,
}

/// A basis of `count` elements on n variables over GF(65521), pushed by
/// `Basis::push_min`, each a `build_tests::poly` of the terms `lengths`
/// draws. Every fifth element's leading monomial divides that of an
/// element of `G_min` of degree 3 or more, one exponent less, so that the
/// element leaves `G_min`; every seventh is a multiple of one, one exponent
/// more, so that it never enters.
fn case<M: Format>(
    r: &mut Xoshiro256PlusPlus,
    n: usize,
    count: usize,
    lengths: Lengths,
) -> Case<M> {
    let field = Gf31::new(65521).unwrap();
    let mut t: Table<M> = Table::new(M::new(n), r);
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    let mut elements: Vec<(Vec<u32>, usize)> = Vec::new();
    let mut left = 0;
    let most = n + 2;
    for k in 0..count {
        let min = basis.min().to_vec();
        let earlier = (!min.is_empty()).then(|| {
            elements[min[below(r, min.len() as u64) as usize].index()]
                .0
                .clone()
        });
        let l = match earlier {
            Some(mut e) if k % 5 == 4 && e.iter().sum::<u32>() >= 3 => {
                let i = (0..n).find(|&i| e[i] > 0).unwrap();
                e[i] -= 1;
                e
            }
            Some(mut e) if k % 7 == 6 => {
                e[below(r, n as u64) as usize] += 1;
                e
            }
            Some(_) | None => lead(r, n),
        };
        let terms = match lengths {
            Lengths::Drawn => 1 + below(r, most as u64) as usize,
            Lengths::Two => 2,
            Lengths::Decreasing => most - (k * most / count).min(most - 1),
        };
        let g = poly(r, &mut t, &field, &l, terms);
        let before = basis.min().len();
        let e = basis.push_min(&t, g).unwrap();
        let entered = usize::from(basis.min().last() == Some(&e));
        left += u32::from(basis.min().len() < before + entered);
        elements.push((l, terms));
    }
    Case {
        table: t,
        basis,
        elements,
        left,
    }
}

/// The rule of f4-gfp.5 by a plain scan of `min` in the order made: the
/// element of fewest terms whose leading monomial divides q, ties to the
/// one made first. `elements` holds every element's leading exponents and
/// terms in the order made.
fn rule(min: &[ElementId], elements: &[(Vec<u32>, usize)], q: &[u32]) -> Option<ElementId> {
    let mut best: Option<ElementId> = None;
    for &e in min {
        let (l, terms) = &elements[e.index()];
        if divides(l, q) && best.is_none_or(|b| elements[b.index()].1 > *terms) {
            best = Some(e);
        }
    }
    best
}

/// Test 1 for one format on n variables: for each count of `SIZES` and
/// 200, and each series of lengths, `Divisors::find` answers every query
/// as `rule` does. The queries are `build_tests::queries` and each
/// leading monomial pushed times a variable.
fn search_agrees<M: Format>(n: usize, seed: u64) -> Met {
    let mut met = Met::default();
    for count in SIZES.into_iter().chain([200]) {
        for (k, lengths) in LENGTHS.into_iter().enumerate() {
            let case_name = format!(
                "{words} words on {n} variables, {count} elements, {lengths:?}",
                words = M::new(n).words()
            );
            let mut r = rng(seed ^ (count as u64) << 8 ^ k as u64);
            let Case {
                table: mut t,
                basis,
                elements,
                left,
            } = case::<M>(&mut r, n, count, lengths);
            met.left += left;
            let mut qs = queries(&mut r, n);
            for (l, _) in &elements {
                let mut q = l.clone();
                q[below(&mut r, n as u64) as usize] += 1;
                qs.push(q);
            }
            let ids: Vec<MonoId> = qs.iter().map(|q| t.intern(q).unwrap()).collect();
            let min = basis.min().to_vec();
            // The order of terms, ties in the order made, by the test's own
            // keys.
            let mut by_terms = min.clone();
            by_terms.sort_by_key(|e| (elements[e.index()].1, e.index()));
            let divisors = Divisors::new(&t, &basis);
            for (q, &u) in qs.iter().zip(&ids) {
                let want = rule(&min, &elements, q);
                let got = divisors.find(&t, u);
                assert_eq!(got, want, "{case_name}: {q:?}");
                let divides_q = |e: &ElementId| divides(&elements[e.index()].0, q);
                let first_made = min.iter().copied().find(divides_q);
                met.reordered += u32::from(first_made != want);
                let passes = |e: &ElementId| t.mask(basis.lead(*e)) & !t.mask(u) == 0;
                match want {
                    Some(w) => {
                        let terms = elements[w.index()].1;
                        met.tied += u32::from(min.iter().any(|e| {
                            e.index() > w.index() && divides_q(e) && elements[e.index()].1 == terms
                        }));
                        met.passed_before += u32::from(
                            by_terms
                                .iter()
                                .take_while(|&&e| e != w)
                                .any(|e| passes(e) && !divides_q(e)),
                        );
                    }
                    None => met.passed_alone += u32::from(min.iter().any(passes)),
                }
            }
        }
    }
    met
}

/// Test 1, mask-scan-time.1 and .2: `Divisors::find` returns the element
/// of `G_min` of fewest terms whose leading monomial divides the query,
/// ties to the one made first, as a plain scan in the order made finds
/// it, on formats of one word and of several.
#[test]
fn the_search_takes_the_fewest_terms_first() {
    let runs = [
        ("Dense16 on 3", search_agrees::<Dense16>(3, 0x3a51)),
        ("Dense8 on 9", search_agrees::<Dense8>(9, 0x3a52)),
        ("Dense16 on 9", search_agrees::<Dense16>(9, 0x3a53)),
        ("Dense32 on 9", search_agrees::<Dense32>(9, 0x3a54)),
    ];
    for (name, met) in runs {
        assert!(
            met.reordered > 0,
            "{name}: every answer the first divisor made"
        );
        assert!(met.tied > 0, "{name}: no answer tied with a later divisor");
        assert!(met.left > 0, "{name}: no element left G_min");
        // On 3 variables a mask holds 10 bits a variable and decides every
        // query here; on 9 it holds 3, and exponents past 3 pass it.
        if name.ends_with('9') {
            assert!(
                met.passed_before > 0,
                "{name}: no mask passed before the answer"
            );
            assert!(
                met.passed_alone > 0,
                "{name}: no mask passed without a divisor"
            );
        }
    }
}

/// A random basis on n variables over GF(65521) in `order`, pushed by
/// `Basis::push_min`: `count` tries of 2 to 9 terms of degree 1 to 6, each
/// made monic, a zero or constant try left out.
fn random_basis<M: Format, O: Order<M>>(
    r: &mut Xoshiro256PlusPlus,
    n: usize,
    count: usize,
    order: &O,
) -> (Table<M>, Basis<Gf31, Rank1>) {
    let field = Gf31::new(65521).unwrap();
    let mut t: Table<M> = Table::new(M::new(n), r);
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    for _ in 0..count {
        let mut terms = Vec::new();
        for _ in 0..2 + below(r, 8) {
            let mut e = vec![0u32; n];
            let mut degree = 1 + below(r, 6) as u32;
            while degree > 0 {
                e[below(r, n as u64) as usize] += 1;
                degree -= 1;
            }
            let m = t.intern(&e).unwrap();
            terms.push((m, field.coef(1 + below(r, 65520))));
        }
        let mut g: Poly<Gf31, Rank1> = Poly::from_terms(&mut terms, &field, &t, order, ());
        if g.is_zero() || t.degree(g.lead()) == 0 {
            continue;
        }
        g.make_monic(&field);
        basis.push_min(&t, g).unwrap();
    }
    (t, basis)
}

/// Test 2 for one format, order, and seed: `matrix::build` from the halves
/// of up to eight random S-pairs of `G_min`, a draw of one element twice
/// left out. Every reducer row's element is the
/// rule's for its leading monomial, and the reducer rows' leading
/// monomials are the matrix's monomials, the given rows' leading monomials
/// left out, that a leading monomial of `G_min` divides, each once. The
/// reducer rows whose element is not the first divisor in the order made.
fn reducers_agree<M: Format, O: Order<M>>(n: usize, order: &O, seed: u64) -> u32 {
    let mut r = rng(seed ^ (n as u64) << 16);
    let count = 10 + below(&mut r, 30) as usize;
    let (mut t, basis) = random_basis::<M, O>(&mut r, n, count, order);
    let min = basis.min().to_vec();
    let mut given = Vec::new();
    for _ in 0..8 {
        let i = min[below(&mut r, min.len() as u64) as usize];
        let j = min[below(&mut r, min.len() as u64) as usize];
        if i == j {
            continue;
        }
        let lcm = t.lcm(basis.lead(i), basis.lead(j)).unwrap();
        for g in [i, j] {
            given.push(RowSource {
                element: g,
                multiplier: t.div(lcm, basis.lead(g)).unwrap(),
            });
        }
    }
    let case_name = format!(
        "{n} variables, seed {seed}, {len} given rows",
        len = given.len()
    );
    let m = matrix::build(&mut t, &basis, order, 1, given, Vec::new(), &mut budget()).unwrap();
    let elements: Vec<(Vec<u32>, usize)> = (0..basis.len())
        .map(|i| {
            let e = ElementId::at(i);
            (exponents(&t, basis.lead(e)), basis.length(e))
        })
        .collect();
    let leads: Vec<MonoId> = m
        .rows
        .iter()
        .map(|row| t.mul(row.multiplier, basis.lead(row.element)).unwrap())
        .collect();
    let (given_leads, mut reducer_leads) = (leads[..m.given].to_vec(), leads[m.given..].to_vec());
    let mut reordered = 0;
    for (&row, &u) in m.rows[m.given..].iter().zip(&reducer_leads) {
        let q = exponents(&t, u);
        let want = rule(&min, &elements, &q);
        assert_eq!(Some(row.element), want, "{case_name}: the reducer of {q:?}");
        let first_made = min
            .iter()
            .copied()
            .find(|e| divides(&elements[e.index()].0, &q));
        reordered += u32::from(first_made != want);
    }
    let mut want_leads: Vec<MonoId> = m
        .monomials
        .iter()
        .copied()
        .filter(|u| !given_leads.contains(u))
        .filter(|&u| {
            let q = exponents(&t, u);
            min.iter().any(|e| divides(&elements[e.index()].0, &q))
        })
        .collect();
    reducer_leads.sort_unstable();
    want_leads.sort_unstable();
    assert_eq!(
        reducer_leads, want_leads,
        "{case_name}: the reducers' leading monomials"
    );
    reordered
}

/// Test 2, mask-scan-time.1 and .2: symbolic preprocessing adds, for every
/// monomial of the matrix but the given rows' leading monomials that a
/// leading monomial of `G_min` divides, one reducer row, of the element
/// the rule names, in grevlex and in lex.
#[test]
fn symbolic_preprocessing_adds_the_rules_reducers() {
    let mut reordered = 0;
    for seed in 0..16 {
        for n in [4, 9] {
            reordered += reducers_agree::<Dense16, _>(n, &Grevlex, seed);
            reordered += reducers_agree::<Dense16, _>(n, &Lex, seed);
        }
    }
    assert!(reordered > 0, "every reducer the first divisor made");
}
