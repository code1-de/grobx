//! instance-read-time, the item's test code in the crate: drawn sparse
//! texts on few and on many variables read into a fresh table and into one
//! that holds monomials already give what the test makes of their terms,
//! the fresh table numbering the monomials in the order the text first
//! writes them; one polynomial in every spelling reads alike, and the
//! errors after a factor keep their kind and line; calls on the edge cases
//! return their certified bases; a stop at the map's checks; and the read
//! in `Dense4` as in `Dense8` (`iterations/0165-instance-read-time/plan.md`,
//! _Test code_, 1 to 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;

use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::field::{Field, Gf2, Gf31, Gf64Pair};
use crate::id::MonoId;
use crate::module::Rank1;
use crate::monomial::{Dense4, Dense8, Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::parsed_tests::{Factor, Term, check, generators, made, table, text};
use crate::poly::Poly;
use crate::shared_tests::{P, REPLAY0, REPLAY1};
use crate::systems::Rng;
use crate::table::Table;
use crate::text::Header;
use crate::{CheckerMode, Error, MonomialOrder, Params, Threads};

/// A drawn sparse system on n variables: 1 to 8 generators of 1 to 12
/// terms, each term on 0 to 3 variables of exponents 1 to 3 and with 0 to
/// 2 factors below 2^20, and a like term at times: a copy of an earlier
/// term, with its sign flipped where the copy is to cancel it.
pub(crate) fn sparse(rng: &mut Rng, n: usize) -> Vec<Vec<Term>> {
    let mut gens = Vec::new();
    for _ in 0..=rng.below(8) {
        let mut terms: Vec<Term> = Vec::new();
        for _ in 0..=rng.below(12) {
            let mut exponents = vec![0; n];
            if !terms.is_empty() && rng.below(4) == 0 {
                exponents.clone_from(&terms[rng.below(terms.len() as u64) as usize].exponents);
            } else {
                for _ in 0..rng.below(4) {
                    exponents[rng.below(n as u64) as usize] = 1 + rng.below(3) as u32;
                }
            }
            let factors: Vec<Factor> = (0..rng.below(3))
                .map(|_| ((1 + rng.below(1 << 20)).to_string(), None))
                .collect();
            let mut order: Vec<usize> = (0..factors.len()).collect();
            order.extend(
                (0..n)
                    .filter(|&i| exponents[i] > 0)
                    .map(|i| factors.len() + i),
            );
            for k in (1..order.len()).rev() {
                order.swap(k, rng.below(k as u64 + 1) as usize);
            }
            terms.push(Term {
                negative: rng.below(2) == 1,
                exponents,
                factors,
                order,
            });
        }
        if rng.below(3) == 0 {
            let t = &terms[rng.below(terms.len() as u64) as usize];
            let copy = Term {
                negative: !t.negative,
                exponents: t.exponents.clone(),
                factors: t.factors.clone(),
                order: t.order.clone(),
            };
            terms.push(copy);
        }
        gens.push(terms);
    }
    gens
}

/// The exponents of the distinct monomials of `gens` in the order the
/// text writes them first.
pub(crate) fn first_written(gens: &[Vec<Term>]) -> Vec<Vec<u32>> {
    let mut seen: Vec<Vec<u32>> = Vec::new();
    for t in gens.iter().flatten() {
        if !seen.contains(&t.exponents) {
            seen.push(t.exponents.clone());
        }
    }
    seen
}

/// Test 1's check of one text into a table that holds monomials already:
/// `earlier`'s generators read into it first, as a replay reads into the
/// table a run before it built, then `gens`, which give what the test makes
/// of their terms in that table.
fn check_into_held<F: Field, M: Format, O: Order<M>>(
    case: &str,
    field: &F,
    order: &O,
    n: usize,
    (earlier, text): (&str, &str),
    gens: &[Vec<Term>],
) {
    let mut t: Table<M> = table(n);
    let header = Header::read(earlier, &mut Budget::unbounded()).unwrap();
    generators(&header, field, &mut t, order).unwrap_or_else(|e| panic!("{case}: {e:?}"));
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    let got = generators(&header, field, &mut t, order)
        .unwrap_or_else(|e| panic!("{case}: {e:?}\n{text}"));
    let want = made(field, &mut t, order, gens);
    assert_eq!(got.len(), want.len(), "{case}\n{text}");
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            (&g.monomials, &g.coefs),
            (&w.monomials, &w.coefs),
            "{case}: generator {k}\n{text}"
        );
    }
}

/// Test 1, instance-read-time.1 and .3: drawn sparse texts on 3, 40, 130,
/// and 300 variables, read over three fields, three orders, and two
/// formats into a fresh table, give what the test makes of their terms,
/// and the fresh table numbers their monomials in the order the text first
/// writes them; read into a table that holds another text's monomials,
/// they give the same.
#[test]
fn sparse_texts_as_written() {
    for seed in 0..48u64 {
        let mut rng = Rng::new(seed ^ 0x0160_5ead);
        let n = [3, 40, 130, 300][(seed % 4) as usize];
        let gens = sparse(&mut rng, n);
        let text = text(&mut rng, n, &gens);
        let other = sparse(&mut rng, n);
        let earlier = crate::parsed_tests::text(&mut rng, n, &other);
        let case = format!("seed {seed}, {n} variables");
        let block = Block::new(&[n / 2], n).unwrap();
        macro_rules! over {
            ($field:expr) => {{
                let f = $field;
                check::<_, Dense16, _>(&case, &f, &Grevlex, n, &text, &gens);
                check::<_, Dense16, _>(&case, &f, &Lex, n, &text, &gens);
                check::<_, Dense16, _>(&case, &f, &block, n, &text, &gens);
                check::<_, Dense8, _>(&case, &f, &Grevlex, n, &text, &gens);
                check_into_held::<_, Dense8, _>(&case, &f, &Grevlex, n, (&earlier, &text), &gens);
                check_into_held::<_, Dense16, _>(&case, &f, &Lex, n, (&earlier, &text), &gens);
            }};
        }
        over!(Gf31::prime(P));
        over!(Gf2::new());
        over!(Gf64Pair::primes(REPLAY0, REPLAY1));
        // A fresh table numbers the monomials as the text first writes them.
        let header = Header::read(&text, &mut Budget::unbounded()).unwrap();
        let mut t: Table<Dense16> = table(n);
        generators(&header, &Gf31::prime(P), &mut t, &Grevlex).unwrap();
        for (k, e) in first_written(&gens).iter().enumerate() {
            let id = t.intern(e, &mut Budget::unbounded()).unwrap();
            assert_eq!(id, MonoId::at(k), "{case}: monomial {k}\n{text}");
        }
    }
}

/// Generators as their terms' exponents and coefficients.
pub(crate) type Written = Vec<Vec<(Vec<u32>, u64)>>;

/// The generators of `text` over GF(65521) in grevlex into a fresh
/// `Dense16` table, each as its terms' exponents and coefficients.
pub(crate) fn read(text: &str) -> Result<Written, Error> {
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    let mut t: Table<Dense16> = table(header.names.len());
    let f = Gf31::prime(P);
    let gens: Vec<Poly<Gf31, Rank1>> = generators(&header, &f, &mut t, &Grevlex)?;
    Ok(gens
        .iter()
        .map(|g| {
            g.monomials
                .iter()
                .zip(&g.coefs)
                .map(|(&m, &c)| (exps(&t, m), f.value(c)))
                .collect()
        })
        .collect())
}

/// Test 2, instance-read-time.4 and .5: one system on names that begin one
/// another, `x`, `x1`, `x10`, `x1_`, `_y`, and `yy2`, written with and
/// without spaces and line feeds around `*`, `^`, `/`, the signs, and the
/// commas, its factors in every order, a denominator after a power and
/// after a number, and each term ended by a sign, a comma, a line feed, or
/// the text's end, reads to the same generators every way.
#[test]
fn spellings_read_alike() {
    let names = "x,x1,x10,x1_,_y,yy2";
    let spellings = [
        "x1*x10^2+x-3*_y*yy2,x1_^3/5-x*x1+1,yy2",
        "x1 * x10 ^ 2 + x - 3 * _y * yy2 , x1_ ^ 3 / 5 - x * x1 + 1 , yy2",
        "x10^2*x1+x-_y*3*yy2,\n1/5*x1_^3-x1*x+1,\nyy2\n",
        "x1*x10*x10+x-yy2*_y*3,x1_^2*x1_/5+1-x*x1,yy2",
        "x1\n*x10^\n2+x-3*_y\n*yy2,x1_^3/\n5-x*\nx1+1,\nyy2",
        "+x1*x10^2+x-3*_y*yy2,x1_^3*1/5-x*x1+1,1*yy2",
    ];
    let want = read(&format!("{names}\n0\n{s}\n", s = spellings[0])).unwrap();
    for s in spellings {
        let got = read(&format!("{names}\n0\n{s}\n")).unwrap_or_else(|e| panic!("{e:?}: {s}"));
        assert_eq!(got, want, "{s}");
    }
}

/// Test 3, instance-read-time.4 and .5: an unknown name right before a
/// `*`, a sign, a comma, a line feed, or the text's end, and a factor
/// missing after a `*` or a sign, give their errors at their lines.
#[test]
fn errors_after_a_factor() {
    let unknown = |text: &str, line: usize, name: &str| {
        let got = read(text);
        let Err(Error::UnknownVariable { line: l, name: m }) = &got else {
            panic!("{got:?}: {text}");
        };
        assert_eq!((*l, m.as_str()), (line, name), "{text}");
    };
    unknown("x,x1\n0\nx*x1,\nx2*x\n", 4, "x2");
    unknown("x,x1\n0\nx*x1,\nx+x12\n", 4, "x12");
    unknown("x,x1\n0\nx*x1,\nx1_,x\n", 4, "x1_");
    unknown("x,x1\n0\nx*x1,\nx-\ny1\n", 5, "y1");
    unknown("x,x1\n0\nx*x1,\nxx", 4, "xx");
    for (text, line) in [
        ("x,x1\n0\nx*x1,\nx*+x1\n", 4),
        ("x,x1\n0\nx*x1,\nx1*\n,x\n", 5),
        ("x,x1\n0\nx*x1,\nx1+-\nx\n", 4),
    ] {
        let got = read(text);
        let Err(Error::Syntax { line: l, .. }) = &got else {
            panic!("{got:?}: {text}");
        };
        assert_eq!(*l, line, "{text}");
    }
}

/// The call of `text` in grevlex on one thread, seed 1, with the exact
/// checker.
pub(crate) fn checked(text: &str) -> crate::Outcome {
    let mut params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    params.checker = Some(CheckerMode::Exact);
    crate::groebner(text, &params).unwrap_or_else(|e| panic!("{e:?}: {text}"))
}

/// Test 4: calls on the edge cases over GF(2), GF(65521), and ℚ, with the
/// exact checker, return their bases certified: the zero ideal, F = (0),
/// the unit ideal, a basis already, the zero polynomial among the
/// generators, and an eliminable variable, each on names that begin one
/// another and with the fast paths' spellings.
#[test]
fn edge_cases_certified() {
    for p in [2, 65521, 0] {
        for (gens, want) in [
            ("0", Some("[]:\n")),
            ("0,0", Some("[]:\n")),
            ("x1*x10+1,x1", Some("[1]:\n")),
            ("x1^2-x10,x10^2-x1_", None),
            ("x1*x10-x10*x1,x1^2+x10,x10^3", None),
            ("x1-x10*1-x1_,x1^2*x10+x1_^2", None),
        ] {
            let text = format!("x1,x10,x1_\n{p}\n{gens}\n");
            let out = checked(&text);
            assert_eq!(out.certified, Some(true), "{text}: {out:?}");
            if let Some(want) = want {
                assert_eq!(out.basis.as_deref(), Some(want), "{text}");
            }
        }
    }
}

/// Test 5: a call whose map sorts many terms makes a full check at
/// `Site::Map` and at `Site::Parse` and returns what it returns without the
/// forcing, and stops at either's first check
/// (`call_stop_sites_tests::stops_at`). Its ideal is the point 0 on 140
/// variables, each a generator, beside 60 generators of 40 terms each,
/// which the substitution takes to 0.
#[test]
fn stops_in_the_map() {
    let mut rng = Rng::new(0x0160_0005);
    let n = 140;
    let names: Vec<String> = (1..=n).map(|i| format!("v{i}")).collect();
    let mut gens = Vec::new();
    for _ in 0..60 {
        let terms: Vec<String> = (0..40)
            .map(|_| {
                let a = 1 + rng.below(n as u64);
                let b = 1 + rng.below(n as u64);
                format!("v{a}*v{b}")
            })
            .collect();
        gens.push(terms.join("+"));
    }
    let text = format!(
        "{names}\n2\n{gens},\n{points}\n",
        names = names.join(","),
        gens = gens.join(",\n"),
        points = names.join(",\n")
    );
    stops_at(Site::Map, &Case::new(text.clone(), "grevlex", false)).unwrap();
    stops_at(Site::Parse, &Case::new(text, "grevlex", false)).unwrap();
}

/// Test 6, instance-read-time.1 and .3: drawn sparse texts over GF(2) on
/// 20 and 130 variables read in `Dense4`, the format a call over GF(2)
/// starts in, where their degrees fit, give the terms they give in
/// `Dense8`.
#[test]
fn dense4_reads_as_dense8() {
    for seed in 0..12u64 {
        let mut rng = Rng::new(seed ^ 0x0160_d4d4);
        let n = [20, 130][(seed % 2) as usize];
        let gens = sparse(&mut rng, n);
        let text = text(&mut rng, n, &gens);
        let case = format!("seed {seed}, {n} variables");
        let f = Gf2::new();
        let header = Header::read(&text, &mut Budget::unbounded()).unwrap();
        let mut t4: Table<Dense4> = table(n);
        let mut t8: Table<Dense8> = table(n);
        let (Ok(g4), Ok(g8)) = (
            generators(&header, &f, &mut t4, &Grevlex),
            generators(&header, &f, &mut t8, &Grevlex),
        ) else {
            continue;
        };
        assert_eq!(g4.len(), g8.len(), "{case}");
        for (a, b) in g4.iter().zip(&g8) {
            let ea: Vec<Vec<u32>> = a.monomials.iter().map(|&m| exps(&t4, m)).collect();
            let eb: Vec<Vec<u32>> = b.monomials.iter().map(|&m| exps(&t8, m)).collect();
            assert_eq!(ea, eb, "{case}\n{text}");
        }
    }
}

/// The exponents of monomial m of `t`.
pub(crate) fn exps<M: Format>(t: &Table<M>, m: MonoId) -> Vec<u32> {
    let f = t.format();
    let mut e = vec![0; f.variables()];
    f.unpack(t.monomial(m), &mut e);
    e
}
