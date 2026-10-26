//! Test code for the order family (`order-family`): the family as the test
//! code compares it on exponent vectors, a basis text read into its terms,
//! the canonical form in an order, the checker's order for a name, and a
//! run with cofactors that the checker accepts in its order. Nothing here
//! leans on the engine's comparisons.

use std::cmp::Ordering;
use std::time::Duration;

use grobx::{Outcome, Stop};
use grobx_checker::{Mode, Order};

use crate::common::{assert_accepted_in, run, with_cofactors_in};

/// An order of the family, compared on exponent vectors in the instance's
/// variable order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Family {
    Grevlex,
    Lex,
    /// The sizes of the blocks but the last, as `block:k1,…,km` names them.
    Block(Vec<usize>),
}

impl Family {
    /// The order a name of the family names; panics on any other name.
    pub fn named(name: &str) -> Family {
        match name {
            "grevlex" => Family::Grevlex,
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
    /// block, and lex has none.
    fn blocks(&self, n: usize) -> Vec<(usize, usize)> {
        match self {
            Family::Grevlex => vec![(0, n)],
            Family::Lex => Vec::new(),
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
    /// name, a block order by its matrix, grevlex's rows as monagan-2007
    /// §2.1 writes them in each block's columns.
    pub fn checker(&self, n: usize) -> Order {
        match self {
            Family::Grevlex => Order::Grevlex,
            Family::Lex => Order::Lex,
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

/// The polynomials of a canonical basis text over GF(p), each as its
/// terms in the order written.
pub fn read_basis(text: &str, names: &[String]) -> Vec<Terms> {
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
            poly.split('+')
                .map(|term| {
                    let mut coef = 1u64;
                    let mut exponents = vec![0u64; names.len()];
                    for factor in term.split('*') {
                        if factor.starts_with(|c: char| c.is_ascii_digit()) {
                            coef = factor.parse().unwrap_or_else(|e| panic!("{term}: {e}"));
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

/// The basis text is canonical in `name`'s order: each polynomial's terms
/// in decreasing order, its leading coefficient 1, and the polynomials in
/// increasing order of their leading monomials (`OBJECTIVES.md`,
/// _Measurement_).
pub fn assert_canonical(case: &str, instance: &str, text: &str, name: &str) {
    let family = Family::named(name);
    let polys = read_basis(text, &names(instance));
    for poly in &polys {
        assert!(
            !poly.is_empty(),
            "{case} in {name}: an empty polynomial\n{text}"
        );
        assert_eq!(poly[0].1, 1, "{case} in {name}: not monic\n{text}");
        for w in poly.windows(2) {
            assert_eq!(
                family.cmp(&w[0].0, &w[1].0),
                Ordering::Greater,
                "{case} in {name}: terms not decreasing\n{text}"
            );
        }
    }
    for w in polys.windows(2) {
        assert_eq!(
            family.cmp(&w[0][0].0, &w[1][0].0),
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

// order-family.19 design -- the test code's random cases in an order that is not graded run with a deadline, and a run the deadline stops passes, not every case asserted to finish (log 0011-1)
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
