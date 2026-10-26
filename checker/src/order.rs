//! The monomial order a check rules in: the order the call names, resolved
//! against the instance's variables, and the comparison of two monomials in
//! it.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::{Error, Order};

// checker-orders.3 cited: bachmann-1998 §2.1 -- matrix-based descriptions of monomial orderings are very general, but their realization in an actual implementation is usually rather time-consuming
/// An order resolved against the instance's n variables. Grevlex and lex
/// keep comparisons of their own, and the matrix comparison serves every
/// other order.
#[derive(Clone, Debug)]
pub enum Cmp {
    Grevlex,
    Lex,
    /// n rows, the first deciding first.
    Matrix(Vec<Row>),
}

/// A row of an order's matrix.
#[derive(Clone, Debug)]
pub enum Row {
    // checker-orders.9 cited: bachmann-1998 §2.2 -- an additional degree field in the monomial pays off, as experience shows
    /// Every entry 1: the monomial's degree word.
    Degree,
    // checker-orders.8 design -- a matrix row is held as its nonzero entries, each a variable and its weight, not as n weights (not measured)
    /// The nonzero entries, each a variable and its weight.
    Sparse(Vec<(usize, i64)>),
}

impl Cmp {
    /// `order` on n variables, or `Error::Order` where it is a matrix that
    /// is not n × n, is singular, or has a column whose first nonzero entry
    /// is negative.
    pub fn new(order: &Order, n: usize) -> Result<Cmp, Error> {
        let rows = match order {
            Order::Grevlex => return Ok(Cmp::Grevlex),
            Order::Lex => return Ok(Cmp::Lex),
            Order::Matrix(rows) => rows,
        };
        // checker-orders.5 cited: robbiano-1985 Theorem 4, via bachmann-1998 §2.1 -- s ≤ n orthogonal vectors u_1, …, u_s decide every comparison: v ↦ (v·u_1, …, v·u_s) into lex is an injective ordered homomorphism
        if rows.len() != n || rows.iter().any(|row| row.len() != n) {
            return Err(Error::Order);
        }
        // checker-orders.4 cited: caboara-2004 §2, Definition 1 b) and Proposition 3 -- for W with linearly independent rows and nonzero columns, the first nonzero entry of each column positive is equivalent to Lex on the degrees of the terms being a term ordering
        for j in 0..n {
            match rows.iter().map(|row| row[j]).find(|&x| x != 0) {
                Some(x) if x > 0 => {}
                Some(_) | None => return Err(Error::Order),
            }
        }
        if !nonsingular(rows) {
            return Err(Error::Order);
        }
        let rows = rows
            .iter()
            .map(|row| {
                if row.iter().all(|&x| x == 1) {
                    Row::Degree
                } else {
                    let entries = row.iter().enumerate().filter(|&(_, &x)| x != 0);
                    Row::Sparse(entries.map(|(v, &x)| (v, x)).collect())
                }
            })
            .collect();
        Ok(Cmp::Matrix(rows))
    }
}

// checker-orders.18 practice: monomorphization -- the resolved order is matched once per loop that compares monomials, and the loop is compiled once per order, not matched at each comparison
// checker-orders.17 cited: bachmann-1998 §3, the fourth remark on Table 1 -- realizing the comparisons for the different orderings by function pointers, so that each comparison requires a function call, was considerably slower than one inlined routine
/// `$body` with `$cmp` bound to the comparison of the resolved order
/// `$order`, a `&Cmp`: one copy of the body per order, each with its
/// comparison inlined, so that a loop in the body compares without asking
/// which order it is in.
macro_rules! by_order {
    ($order:expr, $cmp:ident => $body:expr) => {
        match $order {
            $crate::order::Cmp::Grevlex => {
                let $cmp = $crate::order::grevlex;
                $body
            }
            $crate::order::Cmp::Lex => {
                let $cmp = $crate::order::lex;
                $body
            }
            $crate::order::Cmp::Matrix(rows) => {
                let $cmp = |a: &[u32], b: &[u32]| $crate::order::matrix(rows, a, b);
                $body
            }
        }
    };
}
pub(crate) use by_order;

/// Grevlex: a > b when deg a > deg b, or when the degrees are equal and the
/// last nonzero entry of a − b is negative.
#[inline]
pub fn grevlex(a: &[u32], b: &[u32]) -> Ordering {
    // checker-gfp.6 cited: bachmann-1998 §2 -- an additional degree field in the monomial pays off, as experience shows
    if a[0] != b[0] {
        return a[0].cmp(&b[0]);
    }
    for k in (1..a.len()).rev() {
        if a[k] != b[k] {
            return b[k].cmp(&a[k]);
        }
    }
    Ordering::Equal
}

/// Lex: a > b when the first nonzero entry of a − b is positive.
#[inline]
pub fn lex(a: &[u32], b: &[u32]) -> Ordering {
    for k in 1..a.len() {
        if a[k] != b[k] {
            return a[k].cmp(&b[k]);
        }
    }
    Ordering::Equal
}

// checker-orders.10 design -- a monomial keeps its degree and exponents alone, and a matrix order sums its rows at each comparison, not a vector Aα kept beside every monomial (not measured)
/// The matrix order: a > b when the first nonzero entry of M·(a − b) is
/// positive. Each entry fits `i128`: a weight is at most 2^63 in size, a
/// difference of exponents below 2^32, and n below 2^32.
#[inline]
pub fn matrix(rows: &[Row], a: &[u32], b: &[u32]) -> Ordering {
    for row in rows {
        let s = match row {
            Row::Degree => i128::from(a[0]) - i128::from(b[0]),
            Row::Sparse(entries) => entries
                .iter()
                .map(|&(v, x)| i128::from(x) * (i128::from(a[1 + v]) - i128::from(b[1 + v])))
                .sum(),
        };
        if s != 0 {
            return s.cmp(&0);
        }
    }
    Ordering::Equal
}

// checker-orders.6 design -- whether the matrix is singular is decided exactly over the integers, not by its rank modulo a prime (not measured)
// checker-orders.7 cited: bareiss-1968 §I, (3) and (8), via faugere-1993 §6, the proof of Proposition 6.2, attributes the one-step method to Jordan -- every element the one-step integer-preserving elimination makes is a determinant, an integer, and each division by the previous pivot is exact; faugere-1999 §3.3 -- the Bareiss algorithm is better than the classical Gauss algorithm
/// Whether the square matrix `rows` is nonsingular, by fraction-free
/// elimination: every entry it makes is a minor of the matrix, and each
/// division is exact.
fn nonsingular(rows: &[Vec<i64>]) -> bool {
    let n = rows.len();
    let mut a: Vec<Vec<BigInt>> = rows
        .iter()
        .map(|row| row.iter().map(|&x| BigInt::from(x)).collect())
        .collect();
    let mut previous = BigInt::one();
    for k in 0..n {
        let Some(p) = (k..n).find(|&i| !a[i][k].is_zero()) else {
            return false;
        };
        a.swap(k, p);
        let (top, below) = a.split_at_mut(k + 1);
        let pivot = &top[k];
        for row in below {
            for j in k + 1..n {
                let x = &row[j] * &pivot[k] - &row[k] * &pivot[j];
                debug_assert!((&x % &previous).is_zero(), "a Bareiss step divides exactly");
                row[j] = x / &previous;
            }
        }
        previous.clone_from(&top[k][k]);
    }
    true
}
