//! The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, over Q, and the
//! item's own that Groebner.jl solves: each instance solved by Groebner.jl,
//! accepted, and every mutant of it given its predicted verdict.

use crate::common::{Case, accept_and_reject, matrix_text, plus, quadrics, rule};

fn edge(name: &str, instance: &str) -> Case {
    let case = Case::solved(name, instance);
    accept_and_reject(&case, 0xed9e ^ name.len() as u64);
    case
}

/// F = (0): Groebner.jl writes `[0]`, and the case drops the zero element,
/// so G is empty.
#[test]
#[ignore = "calls groebner-jl"]
fn zero_ideal() {
    let case = edge("edge-q-zero", "x,y\n0\n0\n");
    assert!(
        case.basis.is_empty(),
        "the reduced basis of the zero ideal is empty: {basis:?}",
        basis = case.basis
    );
    assert_eq!(case.basis_text(), "[]:\n");
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_ideal_two_zeros() {
    let case = edge("edge-q-zeros", "x,y\n0\n0,\n0\n");
    assert_eq!(case.basis.len(), 0, "the zero ideal's basis");
}

#[test]
#[ignore = "calls groebner-jl"]
fn unit_ideal() {
    let case = edge("edge-q-unit", "x,y\n0\nx*y-1,\nx\n");
    assert!(case.is_unit(), "{basis:?}", basis = case.basis);
}

/// 1 = (3x + 1) − 3/2·(2x): the cofactors are fractions.
#[test]
#[ignore = "calls groebner-jl"]
fn unit_ideal_fractions() {
    let case = edge("edge-q-unit-fractions", "x\n0\n2*x,\n3*x+1\n");
    assert!(case.is_unit(), "{basis:?}", basis = case.basis);
    assert!(case.rows[0].contains('/'), "{rows:?}", rows = case.rows);
}

#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner() {
    let case = edge("edge-q-gb", "x,y\n0\nx+y,\ny^2+1\n");
    assert_eq!(case.basis, ["x+y", "y^2+1"]);
}

/// The reduced basis of `common::quadrics` over Q, written as an instance.
#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner_reduced_basis() {
    let first = Case::solved("quadrics-p0", &quadrics(0));
    let again = format!(
        "{variables}\n0\n{basis}\n",
        variables = first.variables().join(","),
        basis = first.basis.join(",\n")
    );
    let case = edge("edge-q-gb-quadrics", &again);
    assert_eq!(
        case.basis, first.basis,
        "a reduced basis is its own reduced basis"
    );
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator() {
    edge("edge-q-zero-generator", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n");
}

/// The cofactors of a zero generator may be anything: disturbing them by a
/// fraction changes no verdict.
#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator_cofactors_are_free() {
    let case = Case::solved("edge-q-zero-generator", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n");
    let verdict = case.rule();
    for k in 0..case.rows.len() {
        let rows: Vec<String> = case
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut entries: Vec<String> =
                    row.split(',').map(|c| c.trim().to_string()).collect();
                if i == k {
                    entries[1] = plus(&entries[1], "5/3*x*y^2+1/7");
                }
                entries.join(",")
            })
            .collect();
        let cofactors = matrix_text(rows.iter().map(String::as_str));
        let got = rule(&case.name, &case.instance, &case.basis_text(), &cofactors);
        assert_eq!(
            got, verdict,
            "row {k}: the zero generator's cofactor changed the verdict"
        );
    }
}

/// x is linear in the first generator and appears in no other.
#[test]
#[ignore = "calls groebner-jl"]
fn eliminable_variable() {
    edge("edge-q-eliminable", "x,y,z\n0\nx-y^2-3*z,\ny^3-z,\nz^2-1\n");
}

/// Generators with a common factor: the canonical basis divides it out.
#[test]
#[ignore = "calls groebner-jl"]
fn common_factor() {
    let case = edge("edge-q-common-factor", "x,y\n0\n6*x+4*y,\n9*y^2-3\n");
    assert_eq!(case.basis, ["3*x+2*y", "3*y^2-1"]);
}

/// Coefficients of 29 and 30 digits, past 2^64.
#[test]
#[ignore = "calls groebner-jl"]
fn coefficients_past_2_64() {
    let case = edge(
        "edge-q-past-2-64",
        "x,y\n0\n123456789012345678901234567890*x+y,\nx*y-98765432109876543210987654321\n",
    );
    assert_eq!(case.basis.len(), 2, "{basis:?}", basis = case.basis);
}

/// {x², xy + 65521y²} is a Gröbner basis over GF(65521), not over Q: over Q
/// the ideal holds y³.
#[test]
#[ignore = "calls groebner-jl"]
fn groebner_modulo_a_prime_only() {
    let case = edge("edge-q-modular", "x,y\n0\nx^2,\nx*y+65521*y^2\n");
    assert_eq!(case.basis, ["x*y+65521*y^2", "x^2", "y^3"]);
}

/// The same with 2^64, which a `u64` would wrap to 0.
#[test]
#[ignore = "calls groebner-jl"]
fn groebner_modulo_2_64_only() {
    let case = edge(
        "edge-q-modular-2-64",
        "x,y\n0\nx^2,\nx*y+18446744073709551616*y^2\n",
    );
    assert_eq!(case.basis, ["x*y+18446744073709551616*y^2", "x^2", "y^3"]);
}
