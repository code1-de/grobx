//! The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, in lex and in
//! `block:1`, over GF(p) and Q (checker-orders.14): each instance solved by
//! Groebner.jl in the order, accepted, every mutant given its predicted
//! verdict, and the cheap mode reaching the exact mode's verdict.

use grobx_checker::{Mode, Order, check};

use crate::common::{Case, matrix_text, plus, quadrics};
use crate::orders::{Jl, accept_and_reject_in, agree_full_in, rule_in, solved_in, variables};

// checker-orders.14 design -- the Accept cases take lex and block:k: k drawn from 1 to n − 1 on random systems, k = 1 on the edge cases (not measured)
/// The orders of the edge cases on n variables: lex, and `block:1` where
/// there are two variables or more.
fn jls(n: usize) -> Vec<Jl> {
    if n >= 2 {
        vec![Jl::Lex, Jl::Block(1)]
    } else {
        vec![Jl::Lex]
    }
}

/// Accept, Reject, and Agree on `instance` in each order of `jls`; the
/// cases and their orders.
fn edge(name: &str, instance: &str) -> Vec<(Case, Order)> {
    let seed = 0xed9e ^ name.len() as u64;
    jls(variables(instance))
        .into_iter()
        .map(|jl| {
            let (case, order) = solved_in(name, instance, jl);
            accept_and_reject_in(&case, &order, seed);
            agree_full_in(&case, &order, seed);
            (case, order)
        })
        .collect()
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_ideal() {
    for (name, instance) in [
        ("edge-zero", "x,y\n65521\n0\n"),
        ("edge-q-zero", "x,y\n0\n0\n"),
    ] {
        for (case, order) in edge(name, instance) {
            assert!(
                case.basis.is_empty(),
                "{order:?}: {basis:?}",
                basis = case.basis
            );
            assert_eq!(case.basis_text(), "[]:\n");
        }
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_ideal_two_zeros() {
    for (name, instance) in [
        ("edge-zeros", "x,y\n7\n0,\n0\n"),
        ("edge-q-zeros", "x,y\n0\n0,\n0\n"),
    ] {
        for (case, order) in edge(name, instance) {
            assert!(
                case.basis.is_empty(),
                "{order:?}: {basis:?}",
                basis = case.basis
            );
        }
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn unit_ideal() {
    for (name, instance) in [
        ("edge-unit", "x,y\n65521\nx*y-1,\nx\n"),
        ("edge-unit-p2", "x\n2\nx,\nx+1\n"),
        ("edge-q-unit", "x,y\n0\nx*y-1,\nx\n"),
        ("edge-q-unit-fractions", "x\n0\n2*x,\n3*x+1\n"),
    ] {
        for (case, order) in edge(name, instance) {
            assert!(
                case.is_unit(),
                "{name} in {order:?}: {basis:?}",
                basis = case.basis
            );
        }
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner() {
    // x + y and y² + 1 have the leading terms x and y² in both orders.
    for (name, instance) in [
        ("edge-gb", "x,y\n65521\nx+y,\ny^2+1\n"),
        ("edge-q-gb", "x,y\n0\nx+y,\ny^2+1\n"),
    ] {
        for (case, order) in edge(name, instance) {
            let mut basis = case.basis.clone();
            basis.sort();
            assert_eq!(basis, ["x+y", "y^2+1"], "{name} in {order:?}");
        }
    }
}

/// The reduced basis of `common::quadrics` in each order, written as an
/// instance, is its own reduced basis in that order.
#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner_reduced_basis() {
    for p in [65521, 0] {
        let instance = quadrics(p);
        for jl in jls(variables(&instance)) {
            let (first, _) = solved_in(&format!("quadrics-p{p}"), &instance, jl);
            let again = format!(
                "{variables}\n{p}\n{basis}\n",
                variables = first.variables().join(","),
                basis = first.basis.join(",\n")
            );
            let name = format!(
                "edge-gb-quadrics-p{p}-{order}",
                order = jl.arg().replace(':', "")
            );
            let (case, order) = solved_in(&name, &again, jl);
            let seed = 0xed9e ^ name.len() as u64;
            accept_and_reject_in(&case, &order, seed);
            agree_full_in(&case, &order, seed);
            assert_eq!(
                case.basis, first.basis,
                "a reduced basis in {order:?} is its own reduced basis"
            );
        }
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator() {
    edge("edge-zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n");
    edge("edge-q-zero-generator", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n");
}

/// The cofactors of a zero generator may be anything: disturbing them
/// changes no verdict, in lex or in `block:1`.
#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator_cofactors_are_free() {
    for (name, instance) in [
        ("edge-zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
        ("edge-q-zero-generator", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n"),
    ] {
        for jl in jls(3) {
            let (case, order) = solved_in(name, instance, jl);
            let verdict = rule_in(&case, &order);
            for k in 0..case.rows.len() {
                let rows: Vec<String> = case
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        let mut entries: Vec<String> =
                            row.split(',').map(|c| c.trim().to_string()).collect();
                        if i == k {
                            entries[1] = plus(&entries[1], "5*x*y^2+1");
                        }
                        entries.join(",")
                    })
                    .collect();
                let cofactors = matrix_text(rows.iter().map(String::as_str));
                let got = check(
                    &case.instance,
                    &case.basis_text(),
                    &cofactors,
                    &order,
                    Mode::Exact,
                );
                assert_eq!(
                    got,
                    Ok(verdict.clone()),
                    "{name} in {order:?}, row {k}: the zero generator's cofactor changed the verdict"
                );
            }
        }
    }
}

/// x is linear in the first generator and appears in no other.
#[test]
#[ignore = "calls groebner-jl"]
fn eliminable_variable() {
    edge(
        "edge-eliminable",
        "x,y,z\n65521\nx-y^2-3*z,\ny^3-z,\nz^2-1\n",
    );
    edge("edge-q-eliminable", "x,y,z\n0\nx-y^2-3*z,\ny^3-z,\nz^2-1\n");
}
