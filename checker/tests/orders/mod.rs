//! Test code for the orders (`checker-orders`): the orders Groebner.jl takes
//! and their matrices, the oracle's result in an order, and Accept, Reject,
//! and Agree in the order the checker is told. Accept, Reject, and Agree do
//! what `common::accept_and_reject` and `agree`'s functions do, with the
//! order passed to the checker.

use std::num::NonZeroU32;

use grobx_checker::{Flaw, Mode, Order, Verdict, check, check_cofactors};

use crate::agree::{cheap, consistent, points};
use crate::common::{Case, Expect, Rng, mutations, oracle_with};

/// An order Groebner.jl computes in: the argument of `groebner-jl --order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Jl {
    Grevlex,
    Lex,
    /// Two grevlex blocks, the first k variables in the first.
    Block(usize),
}

impl Jl {
    pub fn arg(self) -> String {
        match self {
            Jl::Grevlex => "grevlex".to_string(),
            Jl::Lex => "lex".to_string(),
            Jl::Block(k) => format!("block:{k}"),
        }
    }

    /// The order's matrix on n variables.
    pub fn matrix(self, n: usize) -> Vec<Vec<i64>> {
        match self {
            Jl::Grevlex => grevlex_matrix(n),
            Jl::Lex => lex_matrix(n),
            Jl::Block(k) => block_matrix(n, k),
        }
    }

    /// The checker's name for the order on n variables: grevlex and lex by
    /// name, a block order by its matrix (checker-orders.2).
    pub fn order(self, n: usize) -> Order {
        match self {
            Jl::Grevlex => Order::Grevlex,
            Jl::Lex => Order::Lex,
            Jl::Block(k) => Order::Matrix(block_matrix(n, k)),
        }
    }
}

/// Grevlex's matrix on n variables, as monagan-2007 §2.1 writes it: a row of
/// ones, then −1 at the last variable, at the one before it, and so on to
/// the second.
pub fn grevlex_matrix(n: usize) -> Vec<Vec<i64>> {
    let mut rows = vec![vec![1; n]];
    for i in 1..n {
        let mut row = vec![0; n];
        row[n - i] = -1;
        rows.push(row);
    }
    rows
}

/// monagan-2007 §2.1's second matrix of grevlex, with no negative entry: row
/// i has ones in its first n − i columns.
pub fn grevlex_matrix_nonnegative(n: usize) -> Vec<Vec<i64>> {
    (0..n)
        .map(|i| (0..n).map(|j| i64::from(j < n - i)).collect())
        .collect()
}

/// Lex's matrix on n variables, the identity.
pub fn lex_matrix(n: usize) -> Vec<Vec<i64>> {
    (0..n)
        .map(|i| (0..n).map(|j| i64::from(i == j)).collect())
        .collect()
}

/// Lex on n ≥ 2 variables with a weight of `i64::MAX` in its first row:
/// rows (`i64::MAX`, 1, 0, …), then the unit rows of the second variable to
/// the last. A difference of exponents is below 2^32, so the first row
/// decides by the first variable wherever its exponents differ.
pub fn lex_matrix_heavy(n: usize) -> Vec<Vec<i64>> {
    let mut rows = lex_matrix(n);
    rows[0][0] = i64::MAX;
    rows[0][1] = 1;
    rows
}

/// The matrix of `block:k` on n variables: grevlex on the first k
/// variables, then grevlex on the other n − k.
pub fn block_matrix(n: usize, k: usize) -> Vec<Vec<i64>> {
    assert!(0 < k && k < n, "block:{k} on {n} variables");
    let mut rows = Vec::new();
    for (start, len) in [(0, k), (k, n - k)] {
        for row in grevlex_matrix(len) {
            let mut full = vec![0; n];
            full[start..start + len].copy_from_slice(&row);
            rows.push(full);
        }
    }
    rows
}

/// The matrix `m` of an order on the variables as a permuted line lists
/// them, where position i holds variable `perm[i]`, as a matrix on the
/// variables as the instance lists them: column i becomes column `perm[i]`.
pub fn permuted(m: &[Vec<i64>], perm: &[usize]) -> Vec<Vec<i64>> {
    m.iter()
        .map(|row| {
            let mut out = vec![0; row.len()];
            for (i, &v) in perm.iter().enumerate() {
                out[v] = row[i];
            }
            out
        })
        .collect()
}

/// A permutation of 0..n, uniform, drawn from `rng`.
pub fn permutation(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rng.index(i + 1));
    }
    perm
}

/// How many variables the instance lists.
pub fn variables(instance: &str) -> usize {
    instance
        .lines()
        .next()
        .expect("an instance has a variable line")
        .split(',')
        .count()
}

/// Groebner.jl's result for `instance` in `jl`, as a case, and the
/// checker's order for it.
pub fn solved_in(name: &str, instance: &str, jl: Jl) -> (Case, Order) {
    let arg = jl.arg();
    let output = oracle_with(name, instance, &["--order", &arg]);
    let case = Case::from_oracle(name, instance, &output);
    (case, jl.order(variables(instance)))
}

// checker-orders.16 design -- matrix orders beyond grevlex, lex, and block:k are tested on Groebner.jl's results for the instance with its variable line permuted (not measured)
/// Groebner.jl's result in `jl` for `instance` with its variable line
/// permuted, position i holding variable `perm[i]`: a case on `instance` as
/// written, and the order whose matrix is jl's with its columns permuted
/// (checker-orders.16). Groebner.jl names every variable, so its basis and
/// cofactors fit the instance as written.
pub fn solved_permuted(name: &str, instance: &str, jl: Jl, perm: &[usize]) -> (Case, Order) {
    let (line, rest) = instance
        .split_once('\n')
        .expect("an instance has a variable line");
    let names: Vec<&str> = line.split(',').map(str::trim).collect();
    assert_eq!(perm.len(), names.len(), "{name}: one place per variable");
    let line: Vec<&str> = perm.iter().map(|&v| names[v]).collect();
    let text = format!("{line}\n{rest}", line = line.join(","));
    let arg = jl.arg();
    let output = oracle_with(name, &text, &["--order", &arg]);
    let case = Case::from_oracle(name, instance, &output);
    let matrix = permuted(&jl.matrix(names.len()), perm);
    (case, Order::Matrix(matrix))
}

/// The checker's verdict in `order`, exact mode; an error fails the test.
pub fn rule_in(case: &Case, order: &Order) -> Verdict {
    check(
        &case.instance,
        &case.basis_text(),
        &case.cofactor_text(),
        order,
        Mode::Exact,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{name}: the checker refused a well-formed result in {order:?}: {e:?}",
            name = case.name
        )
    })
}

/// Accept: the checker says the case is a reduced Gröbner basis in `order`.
/// Reject: every mutant gets its predicted verdict in `order`. A refusal of
/// the case itself fails the test at once; every other failure is gathered
/// into one message.
pub fn accept_and_reject_in(case: &Case, order: &Order, seed: u64) {
    let mut failures = Vec::new();
    let verdict = rule_in(case, order);
    if verdict.groebner.is_err() || verdict.reduced.is_err() {
        failures.push(format!("Accept: {verdict:?}"));
    }
    let mut rng = Rng::new(seed);
    for mutation in mutations(case, &mut rng) {
        let (basis, cofactors) = mutation.apply(case);
        let got = match check(&case.instance, &basis, &cofactors, order, Mode::Exact) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{mutation:?}: refused: {e:?}"));
                continue;
            }
        };
        let ok = match mutation.expect(case) {
            Expect::NotGroebner => got.groebner.is_err(),
            Expect::GroebnerNotReduced => got.groebner.is_ok() && got.reduced.is_err(),
            Expect::Unchanged => got == verdict,
        };
        if !ok {
            failures.push(format!(
                "{mutation:?}: expected {expected:?}, got {got:?}",
                expected = mutation.expect(case)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{name} in {order:?}:\n{failures}",
        name = case.name,
        failures = failures.join("\n")
    );
}

/// Calls `each` on the case's texts, then on every mutant's, with the label
/// and number of each, as `agree` does.
fn variants(case: &Case, seed: u64, mut each: impl FnMut(&str, u64, &str, &str)) {
    each("Accept", 0, &case.basis_text(), &case.cofactor_text());
    let mut rng = Rng::new(seed);
    for (n, mutation) in mutations(case, &mut rng).into_iter().enumerate() {
        let (basis, cofactors) = mutation.apply(case);
        each(&format!("{mutation:?}"), n as u64 + 1, &basis, &cofactors);
    }
}

/// The k Agree uses on `case` and its mutants: a mutation raises a degree by
/// at most 3.
fn case_points(case: &Case) -> NonZeroU32 {
    points(case.prime(), case.degree_bound() + 3)
}

/// `agree::agree_full` in `order`: on the case and every mutant, the whole
/// check and the rows stage alone give in the cheap mode what they give in
/// the exact mode, and in each mode the two are consistent.
pub fn agree_full_in(case: &Case, order: &Order, seed: u64) {
    let k = case_points(case);
    let mut failures = Vec::new();
    variants(case, seed, |label, n, basis, cofactors| {
        let mode = cheap(k, seed, n);
        let run = |mode: Mode| -> Result<(Verdict, Result<(), Flaw>), String> {
            let full = check(&case.instance, basis, cofactors, order, mode)
                .map_err(|e| format!("check refused a well-formed result: {e:?}"))?;
            let rows = check_cofactors(&case.instance, basis, cofactors, order, mode)
                .map_err(|e| format!("check_cofactors refused a well-formed result: {e:?}"))?;
            Ok((full, rows))
        };
        let (exact, got) = match (run(Mode::Exact), run(mode)) {
            (Ok(exact), Ok(got)) => (exact, got),
            (exact, got) => {
                failures.push(format!(
                    "{label}: exact {exact:?}, cheap {got:?} ({mode:?})"
                ));
                return;
            }
        };
        if n == 0 && (exact.0.groebner.is_err() || exact.0.reduced.is_err()) {
            failures.push(format!(
                "{label}: the exact mode says {verdict:?}",
                verdict = exact.0
            ));
        }
        if got != exact {
            failures.push(format!(
                "{label}: exact {exact:?}, cheap {got:?} ({mode:?})"
            ));
        }
        for (name, (full, rows)) in [("exact", &exact), ("cheap", &got)] {
            if !consistent(full, rows) {
                failures.push(format!(
                    "{label}: in the {name} mode check says {full:?} and check_cofactors {rows:?}"
                ));
            }
        }
    });
    assert!(
        failures.is_empty(),
        "{name} in {order:?} (k = {k}):\n{failures}",
        name = case.name,
        failures = failures.join("\n")
    );
}

/// Every verdict the case and each of its mutants get in `a`, they get in
/// `b`, error or not.
pub fn same_verdicts(case: &Case, a: &Order, b: &Order, seed: u64) {
    let mut failures = Vec::new();
    variants(case, seed, |label, _, basis, cofactors| {
        let x = check(&case.instance, basis, cofactors, a, Mode::Exact);
        let y = check(&case.instance, basis, cofactors, b, Mode::Exact);
        if x != y {
            failures.push(format!("{label}: {x:?} in the one, {y:?} in the other"));
        }
    });
    assert!(
        failures.is_empty(),
        "{name} in {a:?} and {b:?}:\n{failures}",
        name = case.name,
        failures = failures.join("\n")
    );
}
