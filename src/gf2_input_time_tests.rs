//! gf2-input-time, the item's test code in the crate: the substitution over
//! the pivots each generator holds, its terms split by their lanes and its
//! sums in a hash table of their own, against an expansion of the test's
//! own; the sums across stamps; a call that substitutes against the same
//! call without, over GF(2) and GF(p), on many pivots and on the edge
//! cases; and a stop at the sites the item's loops spend at
//! (`iterations/0158-gf2-input-time/plan.md`, _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::BTreeMap;
use std::fmt::Write;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::ccode_newton_gap_tests::call;
use crate::field::{Field, Gf2, Gf31};
use crate::gf2_start_over_tests::forcing;
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Format};
use crate::order::Grevlex;
use crate::poly::Poly;
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::{below, rng};
use crate::substitute::{Round, Substituted, substitute, sums};
use crate::table::Table;
use crate::text::{Header, Unread};
use crate::{CheckerMode, MonomialOrder};

/// A polynomial of the test's own: per exponent vector its coefficient in
/// [1, p).
pub(crate) type Terms = BTreeMap<Vec<u32>, u64>;

/// The text of `terms` on the variables x1 to xn, every term written
/// out, `0` where it has none.
pub(crate) fn written(terms: &Terms) -> String {
    if terms.is_empty() {
        return "0".to_string();
    }
    let parts: Vec<String> = terms
        .iter()
        .map(|(e, c)| {
            let mut term = c.to_string();
            for (i, &k) in e.iter().enumerate().filter(|&(_, &k)| k > 0) {
                write!(term, "*x{v}^{k}", v = i + 1).unwrap();
            }
            term
        })
        .collect();
    parts.join("+")
}

/// `a + c·b` mod p, its zero coefficients dropped.
pub(crate) fn add_into(a: &mut Terms, b: &Terms, c: u64, p: u64) {
    for (e, &v) in b {
        let s = (a.get(e).copied().unwrap_or(0) + c * v) % p;
        if s == 0 {
            a.remove(e);
        } else {
            a.insert(e.clone(), s);
        }
    }
}

/// a·b mod p.
fn times(a: &Terms, b: &Terms, p: u64) -> Terms {
    let mut out = Terms::new();
    for (ea, &ca) in a {
        let mut one = Terms::new();
        for (eb, &cb) in b {
            let e: Vec<u32> = ea.iter().zip(eb).map(|(x, y)| x + y).collect();
            one.insert(e, ca * cb % p);
        }
        add_into(&mut out, &one, 1, p);
    }
    out
}

/// `g` with each pivot variable v of `values` replaced by its value, term by
/// term, mod p: the expansion the substitution's Horner's rule must equal.
pub(crate) fn expanded(g: &Terms, values: &BTreeMap<usize, Terms>, n: usize, p: u64) -> Terms {
    let mut out = Terms::new();
    for (e, &c) in g {
        let mut rest = e.clone();
        let mut product = Terms::from([(vec![0; n], 1)]);
        for (&v, value) in values {
            for _ in 0..e[v] {
                product = times(&product, value, p);
            }
            rest[v] = 0;
        }
        let rest = Terms::from([(rest, c)]);
        add_into(&mut out, &times(&product, &rest, p), 1, p);
    }
    out
}

/// The terms of `poly` in `table`, as the test's own.
pub(crate) fn terms_of<F: Field, M: Format>(
    field: &F,
    table: &Table<M>,
    poly: &Poly<F, Rank1>,
) -> Terms {
    let n = table.format().variables();
    poly.monomials
        .iter()
        .zip(&poly.coefs)
        .map(|(&m, &c)| {
            let mut e = vec![0; n];
            table.unpack(m, &mut e);
            (e, field.value(c))
        })
        .collect()
}

/// Random terms on x1 to xn: `count` of them, each of degree at most
/// `degree` in the variables `vars`, coefficients in [1, p).
pub(crate) fn drawn(
    r: &mut Xoshiro256PlusPlus,
    vars: &[usize],
    n: usize,
    degree: u32,
    count: usize,
    p: u64,
) -> Terms {
    let mut out = Terms::new();
    for _ in 0..count {
        let mut e = vec![0u32; n];
        for _ in 0..below(r, u64::from(degree) + 1) {
            e[vars[below(r, vars.len() as u64) as usize]] += 1;
        }
        let one = Terms::from([(e, 1 + below(r, p - 1))]);
        add_into(&mut out, &one, 1, p);
    }
    out
}

/// A case of test 1 on x1 to xn over p: k linear generators, each `x_i +`
/// a drawn linear form in the variables past k, so that x1 to xk are the
/// pivots and the value of `x_i` is minus the form; and the other
/// generators, each holding the pivots of its own set: terms drawn in
/// those pivots and the variables past k, or a drawn rest plus multiples of
/// the linear generators, whose expansion is the rest.
pub(crate) struct Pivots {
    pub(crate) n: usize,
    pub(crate) p: u64,
    pub(crate) linear: Vec<Terms>,
    pub(crate) values: BTreeMap<usize, Terms>,
    pub(crate) others: Vec<Terms>,
}

/// `g` with the term `x_a·x_b` of the variables past k added where no term
/// of g has degree 2 or more, so that g is no linear generator.
fn of_degree_two(mut g: Terms, rest: &[usize], n: usize, p: u64) -> Terms {
    if g.keys().any(|e| e.iter().sum::<u32>() >= 2) {
        return g;
    }
    let mut e = vec![0; n];
    e[rest[0]] += 1;
    e[rest[rest.len() - 1]] += 1;
    add_into(&mut g, &Terms::from([(e, 1)]), 1, p);
    g
}

/// Test 1's case of seed `seed`: n variables, k pivots, values of up to
/// `width` terms, and `count` other generators of degree at most `degree`
/// in their pivots; for two pivots or more, also x1²·x2²·x_(k+1) +
/// x_(k+1)², whose Horner's rule skips a power at each level and sums the
/// products of two squares of values.
pub(crate) fn pivots(
    seed: u64,
    n: usize,
    k: usize,
    width: usize,
    count: usize,
    degree: u32,
    p: u64,
) -> Pivots {
    let mut r = rng(seed);
    let rest: Vec<usize> = (k..n).collect();
    let mut linear = Vec::new();
    let mut values = BTreeMap::new();
    for i in 0..k {
        let terms = 1 + below(&mut r, width as u64) as usize;
        let mut form = drawn(&mut r, &rest, n, 1, terms, p);
        let mut lead = vec![0; n];
        lead[i] = 1;
        let mut value = Terms::new();
        add_into(&mut value, &form, p - 1, p);
        form.insert(lead, 1);
        linear.push(form);
        values.insert(i, value);
    }
    let mut others = Vec::new();
    for j in 0..count {
        let held: Vec<usize> = match j % 4 {
            0 => vec![j % k],
            1 => (0..k).filter(|_| below(&mut r, 3) == 0).collect(),
            2 => (0..k).collect(),
            _ => Vec::new(),
        };
        let mut vars = held.clone();
        vars.extend(&rest);
        let g = if j % 3 == 0 && !held.is_empty() {
            let mut g = drawn(&mut r, &rest, n, 2, 3, p);
            for _ in 0..4 {
                let l = &linear[held[below(&mut r, held.len() as u64) as usize]];
                let m = drawn(&mut r, &vars, n, degree, 1, p);
                add_into(&mut g, &times(&m, l, p), 1, p);
            }
            g
        } else {
            let terms = 2 + below(&mut r, 12) as usize;
            drawn(&mut r, &vars, n, degree, terms, p)
        };
        others.push(of_degree_two(g, &rest, n, p));
    }
    if k >= 2 {
        let mut e = vec![0; n];
        e[0] = 2;
        e[1] = 2;
        e[k] = 1;
        let mut square = vec![0; n];
        square[k] = 2;
        others.push(Terms::from([(e, 1), (square, 1)]));
    }
    Pivots {
        n,
        p,
        linear,
        values,
        others,
    }
}

/// The generators of `text` read over `field` in grevlex into `table`.
pub(crate) fn read_into<F: Field, M: Format>(
    field: &F,
    table: &mut Table<M>,
    text: &str,
) -> Vec<Poly<F, Rank1>> {
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    header
        .generators(field, table, &Grevlex, &mut Budget::unbounded())
        .map_err(Unread::unbounded)
        .unwrap()
}

/// Whether the generator `g` holds one of the k pivots x1 to xk alone, to
/// the power 1 in each term that holds it: Horner's rule then drains one
/// sum for g, its substitute.
fn one_sum(g: &Terms, k: usize) -> bool {
    let held: Vec<usize> = (0..k).filter(|&v| g.keys().any(|e| e[v] > 0)).collect();
    held.len() == 1 && g.keys().all(|e| e[held[0]] <= 1)
}

/// Test 1 in the format M over `field`: `substitute` on each case's
/// generators, the linear ones first, leaves every linear generator and
/// every generator that holds no pivot as read; a generator that holds a
/// pivot and changes takes its expansion, which has fewer terms than it;
/// one whose expansion has no fewer terms stays as read, and so does one
/// that stays at a sum on the way, but one of one sum changes exactly
/// where its expansion has fewer terms (linear-substitution-time.1); and
/// the counts are those. The generators' terms as returned, and the
/// counts, linear, pivots, changed, and stayed.
pub(crate) fn substitutions_in<F: Field, M: Format>(
    field: &F,
    case: &Pivots,
    name: &str,
) -> (Vec<Terms>, [usize; 4]) {
    let names: Vec<String> = (1..=case.n).map(|i| format!("x{i}")).collect();
    let gens: Vec<String> = case
        .linear
        .iter()
        .chain(&case.others)
        .map(written)
        .collect();
    let text = format!(
        "{names}\n{p}\n{gens}\n",
        names = names.join(","),
        p = case.p,
        gens = gens.join(",\n")
    );
    let mut table = Table::new(
        M::new(case.n),
        &mut Xoshiro256PlusPlus::seed_from_u64(0x0158),
    );
    let before = read_into(field, &mut table, &text);
    let got = substitute(
        field,
        &mut table,
        &Grevlex,
        before.clone(),
        &mut Budget::unbounded(),
    )
    .unwrap();
    let Substituted {
        generators: out, ..
    } = &got;
    let k = case.linear.len();
    let (mut changed, mut stayed) = (0, 0);
    for (j, g) in case.others.iter().enumerate() {
        let at = k + j;
        let holds = g.keys().any(|e| e[..k].iter().any(|&x| x > 0));
        let want = expanded(g, &case.values, case.n, case.p);
        let now = terms_of(field, &table, &out[at]);
        let read = terms_of(field, &table, &before[at]);
        if holds && now != read {
            changed += 1;
            assert_eq!(now, want, "{name}: generator {j} takes its expansion");
            assert!(
                want.len() < g.len(),
                "{name}: generator {j} takes an expansion of fewer terms"
            );
        } else {
            stayed += usize::from(holds);
            assert_eq!(now, read, "{name}: generator {j} stays");
            assert!(
                !(holds && one_sum(g, k) && want.len() < g.len()),
                "{name}: generator {j}, of one sum, takes its shorter expansion"
            );
        }
    }
    for i in 0..k {
        assert_eq!(
            terms_of(field, &table, &out[i]),
            terms_of(field, &table, &before[i]),
            "{name}: linear generator {i} stays"
        );
    }
    let counts = [got.linear, got.pivots, got.changed, got.stayed];
    assert_eq!(counts, [k, k, changed, stayed], "{name}: the counts");
    (
        out.iter().map(|g| terms_of(field, &table, g)).collect(),
        counts,
    )
}

/// Test 1, gf2-input-time.6 to .10, .12: `substitute` over GF(2) and 65521,
/// in `Dense8` and `Dense16`, on cases of 1, 3, 12, and 40 pivots, values
/// of up to 1, 4, and 16 terms, and generators that hold one pivot, some,
/// all, or none, their powers up to 4 among them: each generator that
/// holds a pivot and changes takes the expansion of the test's own, which
/// has fewer terms, and every other generator stays; one of one sum
/// changes where its expansion is shorter (linear-substitution-time.1, which
/// amends ccode-newton-gap.16's rule). The product of two squares
/// of values of up to 12 terms makes sums of thousands of terms.
#[test]
fn substitutes_are_the_expansions() {
    let shapes = [
        (6, 1, 1, 8, 4),
        (10, 3, 4, 12, 4),
        (24, 12, 16, 16, 3),
        (60, 40, 4, 24, 2),
        (12, 2, 12, 4, 4),
    ];
    for (s, &(n, k, width, count, degree)) in (1u64..).zip(&shapes) {
        for p in [2u64, 65521] {
            let case = pivots(0x0158 + s, n, k, width, count, degree, p);
            let name = format!("shape {s} over {p}");
            if p == 2 {
                substitutions_in::<_, Dense8>(&Gf2::new(), &case, &format!("{name}, Dense8"));
                substitutions_in::<_, Dense16>(&Gf2::new(), &case, &format!("{name}, Dense16"));
            } else {
                let field = Gf31::new(p).unwrap();
                substitutions_in::<_, Dense8>(&field, &case, &format!("{name}, Dense8"));
                substitutions_in::<_, Dense16>(&field, &case, &format!("{name}, Dense16"));
            }
        }
    }
}

/// Test 2, gf2-input-time.9, .11: one sum over rounds that drain it:
/// each round drains its distinct words in the order they first came,
/// their coefficients summed, those that sum to 0 dropped, none of an
/// earlier round's; words under one hash stay apart; a round of more terms
/// than half the sum's slots grows them; and after the stamp of the first
/// round, 1, the sum takes `u32::MAX`, so that the next drain wraps the
/// stamp to 1 again, and a later round meets the slots the first round
/// filled.
#[test]
fn sums_across_stamps() {
    let field = Gf31::new(7).unwrap();
    let word = |a: u64, b: u64| vec![a, b];
    let wide = |c: u64| -> Vec<(Vec<u64>, u64, u64)> {
        (0..1000u64)
            .map(|i| {
                (
                    word(i, i / 3),
                    i.wrapping_mul(0x9e37_79b9_7f4a_7c15) % 97,
                    c + i % 6,
                )
            })
            .collect()
    };
    let mut twice = wide(1);
    twice.extend(wide(0).into_iter().map(|(w, h, _)| (w, h, 1)));
    let rounds: Vec<Round<u64>> = vec![
        (
            None,
            vec![
                (word(1, 0), 11, 3),
                (word(2, 0), 11, 5),
                (word(1, 0), 11, 4),
                (word(3, 0), 12, 6),
            ],
        ),
        (Some(u32::MAX), vec![(word(4, 4), 5, 6), (word(4, 4), 5, 1)]),
        (None, vec![(word(5, 5), 11, 2), (word(6, 6), 12, 3)]),
        (None, Vec::new()),
        (None, twice),
        (None, vec![(word(9, 9), 0, 1)]),
    ];
    let got = sums::<_, Dense8>(&field, 2, &rounds);
    let want: Vec<Vec<(Vec<u64>, u64)>> = vec![
        vec![(word(2, 0), 5), (word(3, 0), 6)],
        Vec::new(),
        vec![(word(5, 5), 2), (word(6, 6), 3)],
        Vec::new(),
        (0..1000u64)
            .filter(|i| (1 + i % 6 + 1) % 7 != 0)
            .map(|i| (word(i, i / 3), (1 + i % 6 + 1) % 7))
            .collect(),
        vec![(word(9, 9), 1)],
    ];
    assert_eq!(got, want);
}

/// The edge cases over GF(2) on x, y, z (plan.md, _Edge cases_): the
/// zero ideal, F = (0), the unit ideal, an input already a Gröbner basis,
/// the zero polynomial among the generators, an eliminable variable, a
/// generator the substitution makes 0, every variable a pivot, and linear
/// generators that contradict each other.
pub(crate) const EDGE: [&str; 10] = [
    "",
    "0",
    "x+y+1,\nx+y",
    "x^2+y,\ny^2+z",
    "0,\nx+y,\ny^2+z",
    "x+y+z,\nx*y+z,\nz^2+x",
    "x+y,\nx^2+y^2",
    "x+1,\ny+1,\nz,\nx*y*z+x",
    "x+y+1,\nx+y,\ny*z+1",
    "x+z,\ny+z,\nx*y+y*z+x*z+z+1",
];

/// `g` less its value at the point `a`, mod p: a polynomial that vanishes
/// at a.
pub(crate) fn through(mut g: Terms, a: &[u64], p: u64) -> Terms {
    let value = g.iter().fold(0, |sum, (e, &c)| {
        let at = e
            .iter()
            .zip(a)
            .fold(c, |v, (&k, &x)| (0..k).fold(v, |v, _| v * x % p));
        (sum + at) % p
    });
    let one = Terms::from([(vec![0; a.len()], value)]);
    add_into(&mut g, &one, p - 1, p);
    g
}

/// A system over p on n variables that vanishes at a drawn point: k
/// linear generators, each a pivot plus a sparse form in the variables
/// past k, and quadrics whose terms hold the pivots sparsely, so that each
/// holds a few of them.
pub(crate) fn sparse_pivots(n: usize, k: usize, quadrics: usize, p: u64, seed: u64) -> String {
    let mut r = rng(seed);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let rest: Vec<usize> = (k..n).collect();
    let a: Vec<u64> = (0..n).map(|_| below(&mut r, p)).collect();
    let mut gens = Vec::new();
    for i in 0..k {
        let terms = 1 + below(&mut r, 6) as usize;
        let mut form = drawn(&mut r, &rest, n, 1, terms, p);
        let mut lead = vec![0; n];
        lead[i] = 1;
        form.insert(lead, 1);
        gens.push(written(&through(form, &a, p)));
    }
    for _ in 0..quadrics {
        let mut vars = rest.clone();
        for _ in 0..2 {
            vars.push(below(&mut r, k as u64) as usize);
        }
        gens.push(written(&through(drawn(&mut r, &vars, n, 2, 6, p), &a, p)));
    }
    if p == 2 {
        for i in 1..=n {
            gens.push(format!("x{i}^2+x{i}"));
        }
    }
    format!(
        "{names}\n{p}\n{gens}\n",
        names = names.join(","),
        gens = gens.join(",\n")
    )
}

/// Test 3, gf2-input-time.1 to .12: a call returns with the substitution
/// what it returns without, a basis other than {1} where the system
/// vanishes at a point: the edge cases over GF(2), systems over GF(2)
/// of 12 to 40 variables whose quadrics hold few of their 4 to 24 pivots,
/// and systems over 7 of 12 and 16 variables, the signatures off and on;
/// and the edge cases and the smallest system with cofactors and the
/// exact checker, which certifies the call.
#[test]
fn calls_return_what_they_return_without() {
    let mut texts: Vec<(String, String)> = EDGE
        .iter()
        .enumerate()
        .map(|(k, g)| (format!("edge {k}"), format!("x,y,z\n2\n{g}\n")))
        .collect();
    let shapes = [
        (2, 12, 4, 10),
        (2, 20, 8, 16),
        (2, 40, 24, 20),
        (7, 12, 4, 16),
        (7, 16, 6, 24),
    ];
    for (s, &(p, n, k, quadrics)) in (1u64..).zip(&shapes) {
        texts.push((
            format!("sparse pivots {n}/{k} over {p}"),
            sparse_pivots(n, k, quadrics, p, 0x0158 + s),
        ));
    }
    for (c, (name, text)) in texts.iter().enumerate() {
        for signatures in [false, true] {
            let mut params = reported(MonomialOrder::Grevlex);
            params.signatures = signatures;
            let off = call(text, &params, false);
            let on = call(text, &params, true);
            assert!(
                on.basis.is_some(),
                "{name}, signatures {signatures}: a basis"
            );
            assert_eq!(on.basis, off.basis, "{name}, signatures {signatures}");
            if c >= EDGE.len() {
                assert_ne!(
                    on.basis.as_deref(),
                    Some("[1]:\n"),
                    "{name}: the system vanishes at its point"
                );
            }
        }
        if c > EDGE.len() {
            continue;
        }
        let mut params = reported(MonomialOrder::Grevlex);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        if crate::call::built(&params).is_err() {
            continue;
        }
        let on = call(text, &params, true);
        assert_eq!(on.certified, Some(true), "{name} with cofactors: certified");
    }
}

/// Test 4, gf2-input-time.2, .3, .9: a call stops at the first check of the
/// names' index, which grows past 16, 32, and 64 names, and at the first
/// check of a substitution whose sums grow, in the signature run that a
/// bound of 0 starts the call over in, and returns what it returns without
/// the forcing (`call_stop_sites_tests::stops_at`).
#[test]
fn stops_in_the_index_and_the_sums() {
    let names: Vec<String> = (1..=70).map(|i| format!("v{i}")).collect();
    let squares: Vec<String> = names.iter().map(|v| format!("{v}^2+{v}")).collect();
    let text = format!(
        "{names}\n2\n{squares}\n",
        names = names.join(","),
        squares = squares.join(",\n")
    );
    stops_at(Site::Names, &Case::new(text.clone(), "grevlex", false)).unwrap();
    stops_at(Site::Parse, &Case::new(text, "grevlex", false)).unwrap();
    let sums = sparse_pivots(30, 12, 24, 2, 0x5ab5);
    // Over GF(2) the signature run alone substitutes.
    forcing(Some(0), || {
        stops_at(Site::Substitute, &Case::new(sums, "grevlex", false))
    })
    .unwrap();
}
