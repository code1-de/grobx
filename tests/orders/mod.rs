//! Test code for the order family (`order-family`, `deglex-order`): the
//! family as the test code compares it on exponent vectors, a basis text
//! read into its terms, the canonical form in an order, the checker's order
//! for a name, and a run with cofactors that the checker accepts in its
//! order. Nothing here leans on the engine's comparisons.

use std::cmp::Ordering;
use std::time::Duration;

use grobx::trace::{NoTrace, Record};
use grobx::{Outcome, Stop};
use grobx_checker::{Mode, Order};

use crate::common::{assert_accepted_in, params_in, run, with_cofactors_in};

/// An order of the family, compared on exponent vectors in the instance's
/// variable order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Family {
    Grevlex,
    /// The degree lexicographic order: the degree, then lex.
    Deglex,
    Lex,
    /// The sizes of the blocks but the last, as `block:k1,…,km` names them.
    Block(Vec<usize>),
}

impl Family {
    /// The order a name of the family names; panics on any other name.
    pub fn named(name: &str) -> Family {
        match name {
            "grevlex" => Family::Grevlex,
            "deglex" => Family::Deglex,
            "lex" => Family::Lex,
            _ => {
                let sizes = name
                    .strip_prefix("block:")
                    .unwrap_or_else(|| panic!("{name:?} is no name of the family"));
                Family::Block(sizes.split(',').map(|k| k.parse().unwrap()).collect())
            }
        }
    }

    /// The blocks on n variables, each a range of variables: grevlex is one
    /// block, and lex has none. Deglex is no block order.
    fn blocks(&self, n: usize) -> Vec<(usize, usize)> {
        match self {
            Family::Grevlex => vec![(0, n)],
            Family::Lex => Vec::new(),
            Family::Deglex => panic!("deglex is no block order"),
            Family::Block(sizes) => {
                let mut blocks = Vec::new();
                let mut start = 0;
                for &k in sizes {
                    blocks.push((start, start + k));
                    start += k;
                }
                assert!(start < n, "{self:?} leaves no last block on {n} variables");
                blocks.push((start, n));
                blocks
            }
        }
    }

    /// a against b, exponent vectors of one length.
    pub fn cmp(&self, a: &[u64], b: &[u64]) -> Ordering {
        assert_eq!(a.len(), b.len());
        let blocks = match self {
            Family::Lex => return a.cmp(b),
            Family::Deglex => {
                let (da, db): (u64, u64) = (a.iter().sum(), b.iter().sum());
                return da.cmp(&db).then_with(|| a.cmp(b));
            }
            Family::Grevlex | Family::Block(_) => self.blocks(a.len()),
        };
        for (s, e) in blocks {
            let da: u64 = a[s..e].iter().sum();
            let db: u64 = b[s..e].iter().sum();
            if da != db {
                return da.cmp(&db);
            }
            for i in (s..e).rev() {
                if a[i] != b[i] {
                    return b[i].cmp(&a[i]);
                }
            }
        }
        Ordering::Equal
    }

    /// The checker's name for the order on n variables: grevlex and lex by
    /// name, deglex by its matrix, a row of ones and then the unit rows of
    /// the first n − 1 variables, and a block order by its matrix,
    /// grevlex's rows as monagan-2007 §2.1 writes them in each block's
    /// columns.
    pub fn checker(&self, n: usize) -> Order {
        match self {
            Family::Grevlex => Order::Grevlex,
            Family::Lex => Order::Lex,
            Family::Deglex => {
                let mut rows = vec![vec![1; n]];
                for i in 0..n - 1 {
                    let mut row = vec![0; n];
                    row[i] = 1;
                    rows.push(row);
                }
                Order::Matrix(rows)
            }
            Family::Block(_) => {
                let mut rows = Vec::new();
                for (s, e) in self.blocks(n) {
                    let b = e - s;
                    let mut ones = vec![0; n];
                    ones[s..e].fill(1);
                    rows.push(ones);
                    for i in 1..b {
                        let mut row = vec![0; n];
                        row[e - i] = -1;
                        rows.push(row);
                    }
                }
                Order::Matrix(rows)
            }
        }
    }
}

/// `block:1,…,1` with n − 1 ones, which is lex on n ≥ 2 variables.
pub fn lex_as_blocks(n: usize) -> String {
    assert!(n >= 2);
    format!("block:{ones}", ones = vec!["1"; n - 1].join(","))
}

/// The instance's variable names.
pub fn names(instance: &str) -> Vec<String> {
    instance
        .lines()
        .next()
        .expect("an instance has a variable line")
        .split(',')
        .map(|v| v.trim().to_string())
        .collect()
}

/// A polynomial of a basis text: its terms as written, each its exponent
/// vector and its coefficient.
pub type Terms = Vec<(Vec<u64>, u64)>;

/// The polynomials of a basis text, each as its terms in the order
/// written: a term's exponent vector and its coefficient's digits, "1"
/// where it writes none. The terms part at each of `signs`, which a term
/// leaves out: `+` over GF(p), and `+` and `-` over ℚ, whose leading
/// coefficients are positive.
fn read_terms<'t>(
    text: &'t str,
    names: &[String],
    signs: &[char],
) -> Vec<Vec<(Vec<u64>, &'t str)>> {
    let inner = text
        .strip_prefix('[')
        .and_then(|t| t.strip_suffix("]:\n"))
        .unwrap_or_else(|| panic!("no basis text: {text:?}"));
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(",\n")
        .map(|poly| {
            poly.split(signs)
                .map(|term| {
                    let mut coef = "1";
                    let mut exponents = vec![0u64; names.len()];
                    for factor in term.split('*') {
                        if factor.starts_with(|c: char| c.is_ascii_digit()) {
                            coef = factor;
                            continue;
                        }
                        let (name, e) = factor.split_once('^').unwrap_or((factor, "1"));
                        let i = names
                            .iter()
                            .position(|n| n == name)
                            .unwrap_or_else(|| panic!("{name} in {term}"));
                        exponents[i] += e.parse::<u64>().unwrap();
                    }
                    (exponents, coef)
                })
                .collect()
        })
        .collect()
}

/// The polynomials of a canonical basis text over GF(p), each as its
/// terms in the order written.
pub fn read_basis(text: &str, names: &[String]) -> Vec<Terms> {
    read_terms(text, names, &['+'])
        .into_iter()
        .map(|poly| {
            poly.into_iter()
                .map(|(exponents, coef)| {
                    let coef = coef.parse().unwrap_or_else(|e| panic!("{coef}: {e}"));
                    (exponents, coef)
                })
                .collect()
        })
        .collect()
}

/// The monomials of a canonical basis text over any field, each
/// polynomial's in the order written.
fn read_monomials(text: &str, names: &[String]) -> Vec<Vec<Vec<u64>>> {
    read_terms(text, names, &['+', '-'])
        .into_iter()
        .map(|poly| poly.into_iter().map(|(exponents, _)| exponents).collect())
        .collect()
}

/// The basis text over GF(p) is canonical in `name`'s order: in that
/// order (`assert_ordered`), and each polynomial's leading coefficient 1
/// (`OBJECTIVES.md`, _Measurement_).
pub fn assert_canonical(case: &str, instance: &str, text: &str, name: &str) {
    assert_ordered(case, instance, text, name);
    for poly in read_basis(text, &names(instance)) {
        assert_eq!(poly[0].1, 1, "{case} in {name}: not monic\n{text}");
    }
}

/// The basis text over any field, ℚ among them, stands in `name`'s order:
/// each polynomial's terms in decreasing order, and the polynomials in
/// increasing order of their leading monomials (`OBJECTIVES.md`,
/// _Measurement_). Its scaling is the checker's reduced verdict.
pub fn assert_ordered(case: &str, instance: &str, text: &str, name: &str) {
    let family = Family::named(name);
    let polys = read_monomials(text, &names(instance));
    for poly in &polys {
        assert!(
            !poly.is_empty(),
            "{case} in {name}: an empty polynomial\n{text}"
        );
        for w in poly.windows(2) {
            assert_eq!(
                family.cmp(&w[0], &w[1]),
                Ordering::Greater,
                "{case} in {name}: terms not decreasing\n{text}"
            );
        }
    }
    for w in polys.windows(2) {
        assert_eq!(
            family.cmp(&w[0][0], &w[1][0]),
            Ordering::Less,
            "{case} in {name}: leading monomials not increasing\n{text}"
        );
    }
}

/// The outcome of a run of `instance` in the order `name` with cofactors,
/// its basis text canonical in the order and accepted by the checker in it.
pub fn accepted_in(case: &str, instance: &str, name: &str, mode: Mode) -> Outcome {
    let outcome = run(instance, &with_cofactors_in(name, 0x0de7));
    check_in(case, instance, name, mode, &outcome);
    outcome
}

// order-family.19 design-checked -- the test code's random cases in an order that is not graded run with a deadline, and a run the deadline stops passes, not every case asserted to finish (log 0011-1)
/// As `accepted_in`, with a deadline: None where the deadline stopped the
/// run (order-family.19).
pub fn finished_in(
    case: &str,
    instance: &str,
    name: &str,
    mode: Mode,
    deadline: Duration,
) -> Option<Outcome> {
    let mut params = with_cofactors_in(name, 0x0de7);
    params.deadline = Some(deadline);
    let outcome = grobx::groebner(instance, &params)
        .unwrap_or_else(|e| panic!("{case} in {name}: {e:?}\n{instance}"));
    if outcome.basis.is_none() {
        assert_eq!(outcome.stopped, Some(Stop::Deadline), "{case} in {name}");
        eprintln!("{case} in {name}: stopped by the deadline");
        return None;
    }
    check_in(case, instance, name, mode, &outcome);
    Some(outcome)
}

/// The outcome's basis text is canonical in the order `name`, and the
/// checker accepts it in that order.
fn check_in(case: &str, instance: &str, name: &str, mode: Mode, outcome: &Outcome) {
    let basis = outcome.basis.as_deref().unwrap();
    assert_canonical(case, instance, basis, name);
    let order = Family::named(name).checker(names(instance).len());
    assert_accepted_in(case, instance, outcome, &order, mode);
}

/// The record a learn of `text` in `order` returns, its run finished.
pub fn learn(name: &str, text: &str, order: &str) -> Record {
    let (outcome, record) = grobx::learn(text, &params_in(order, 1), &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: learn failed in {order}: {e:?}"));
    assert!(outcome.basis.is_some(), "{name}: the learn run stopped");
    record.unwrap_or_else(|| panic!("{name}: no record"))
}
