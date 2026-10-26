//! The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, over GF(p): each
//! instance solved by Groebner.jl, accepted, and every mutant of it given its
//! predicted verdict.

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
    let case = edge("edge-zero", "x,y\n65521\n0\n");
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
    let case = edge("edge-zeros", "x,y\n7\n0,\n0\n");
    assert_eq!(case.basis.len(), 0, "the zero ideal's basis");
}

#[test]
#[ignore = "calls groebner-jl"]
fn unit_ideal() {
    let case = edge("edge-unit", "x,y\n65521\nx*y-1,\nx\n");
    assert!(case.is_unit(), "{basis:?}", basis = case.basis);
}

#[test]
#[ignore = "calls groebner-jl"]
fn unit_ideal_p2() {
    let case = edge("edge-unit-p2", "x\n2\nx,\nx+1\n");
    assert!(case.is_unit(), "{basis:?}", basis = case.basis);
}

#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner() {
    let case = edge("edge-gb", "x,y\n65521\nx+y,\ny^2+1\n");
    assert_eq!(case.basis, ["x+y", "y^2+1"]);
}

/// The reduced basis of `common::quadrics` over GF(65521), written as an
/// instance.
#[test]
#[ignore = "calls groebner-jl"]
fn already_groebner_reduced_basis() {
    let first = Case::solved("quadrics-p65521", &quadrics(65521));
    let again = format!(
        "{variables}\n65521\n{basis}\n",
        variables = first.variables().join(","),
        basis = first.basis.join(",\n")
    );
    let case = edge("edge-gb-quadrics", &again);
    assert_eq!(
        case.basis, first.basis,
        "a reduced basis is its own reduced basis"
    );
}

#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator() {
    edge("edge-zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n");
}

/// The cofactors of a zero generator may be anything: disturbing them
/// changes no verdict.
#[test]
#[ignore = "calls groebner-jl"]
fn zero_generator_cofactors_are_free() {
    let case = Case::solved("edge-zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n");
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
                    entries[1] = plus(&entries[1], "5*x*y^2+1");
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
    edge(
        "edge-eliminable",
        "x,y,z\n65521\nx-y^2-3*z,\ny^3-z,\nz^2-1\n",
    );
}
