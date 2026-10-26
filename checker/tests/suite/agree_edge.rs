//! Agree over the edge cases of `edge_gfp.rs` and `edge_q.rs`
//! (`OBJECTIVES.md`, _Domain specifics the workflow points at_): on Groebner.jl's result and on
//! every mutant Reject draws for it, the cheap mode reaches the exact mode's
//! verdict, of the whole check and of the rows stage alone.

use grobx_checker::{Mode, Order, check, check_cofactors};

use crate::agree::{agree_full, cheap, points};
use crate::common::{Case, matrix_text, plus, quadrics};

/// The seeds are those of the edge tests, so that Agree runs on Reject's
/// mutants.
fn edge(name: &str, instance: &str) -> Case {
    let case = Case::solved(name, instance);
    agree_full(&case, 0xed9e ^ name.len() as u64);
    case
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_ideal() {
    edge("edge-zero", "x,y\n65521\n0\n");
    edge("edge-q-zero", "x,y\n0\n0\n");
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_ideal_two_zeros() {
    edge("edge-zeros", "x,y\n7\n0,\n0\n");
    edge("edge-q-zeros", "x,y\n0\n0,\n0\n");
}

#[test]
#[ignore = "calls groebner-jl"]
fn unit_ideal() {
    edge("edge-unit", "x,y\n65521\nx*y-1,\nx\n");
    edge("edge-unit-p2", "x\n2\nx,\nx+1\n");
    edge("edge-q-unit", "x,y\n0\nx*y-1,\nx\n");
    edge("edge-q-unit-fractions", "x\n0\n2*x,\n3*x+1\n");
}

#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner() {
    edge("edge-gb", "x,y\n65521\nx+y,\ny^2+1\n");
    edge("edge-q-gb", "x,y\n0\nx+y,\ny^2+1\n");
}

/// The reduced basis of `common::quadrics`, written as an instance, over both
/// fields.
#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner_reduced_basis() {
    for (p, tag) in [(65521, ""), (0, "-q")] {
        let first = Case::solved(&format!("quadrics-p{p}"), &quadrics(p));
        let again = format!(
            "{variables}\n{p}\n{basis}\n",
            variables = first.variables().join(","),
            basis = first.basis.join(",\n")
        );
        edge(&format!("edge{tag}-gb-quadrics"), &again);
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator() {
    edge("edge-zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n");
    edge("edge-q-zero-generator", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n");
}

/// The cofactors of a zero generator may be anything: disturbing them
/// changes no verdict in the cheap mode either.
#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator_cofactors_are_free() {
    for (name, instance, disturbance) in [
        (
            "edge-zero-generator",
            "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n",
            "5*x*y^2+1",
        ),
        (
            "edge-q-zero-generator",
            "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n",
            "5/3*x*y^2+1/7",
        ),
    ] {
        let case = Case::solved(name, instance);
        let verdict = case.rule();
        let k = points(case.prime(), case.degree_bound() + 3);
        for row in 0..case.rows.len() {
            let rows: Vec<String> = case
                .rows
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut entries: Vec<String> =
                        r.split(',').map(|c| c.trim().to_string()).collect();
                    if i == row {
                        entries[1] = plus(&entries[1], disturbance);
                    }
                    entries.join(",")
                })
                .collect();
            let cofactors = matrix_text(rows.iter().map(String::as_str));
            let basis = case.basis_text();
            for seed in 0..4 {
                let mode = cheap(k, seed, row as u64);
                let got = check(instance, &basis, &cofactors, &Order::Grevlex, mode);
                assert_eq!(got, Ok(verdict.clone()), "{name}, row {row}, {mode:?}");
                let got = check_cofactors(instance, &basis, &cofactors, &Order::Grevlex, mode);
                assert_eq!(got, Ok(Ok(())), "{name}, row {row}, {mode:?}");
            }
            let exact = check_cofactors(instance, &basis, &cofactors, &Order::Grevlex, Mode::Exact);
            assert_eq!(exact, Ok(Ok(())), "{name}, row {row}, exact");
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

#[test]
#[ignore = "calls groebner-jl"]
fn q_common_factor() {
    edge("edge-q-common-factor", "x,y\n0\n6*x+4*y,\n9*y^2-3\n");
}

#[test]
#[ignore = "calls groebner-jl"]
fn q_coefficients_past_2_64() {
    edge(
        "edge-q-past-2-64",
        "x,y\n0\n123456789012345678901234567890*x+y,\nx*y-98765432109876543210987654321\n",
    );
}

/// Two Gröbner bases, one modulo 65521 and one modulo 2^64, that are none
/// over Q.
#[test]
#[ignore = "calls groebner-jl"]
fn q_groebner_modulo_a_prime_only() {
    edge("edge-q-modular", "x,y\n0\nx^2,\nx*y+65521*y^2\n");
    edge(
        "edge-q-modular-2-64",
        "x,y\n0\nx^2,\nx*y+18446744073709551616*y^2\n",
    );
}
