//! The input grammar (checker-gfp.3): spellings that denote the same result
//! get the same verdict, and malformed text is an error.

use grobx_checker::{Flaw, Mode, Order, Verdict, check};

const INSTANCE: &str = "x,y\n7\nx+y,\ny^2+1\n";
const BASIS: &str = "[x+y,\ny^2+1]:\n";
const COFACTORS: &str = "[[1,0],\n[0,1]]:\n";

fn holds() -> Verdict {
    Verdict {
        groebner: Ok(()),
        reduced: Ok(()),
    }
}

fn ruled(instance: &str, basis: &str, cofactors: &str) -> Result<Verdict, String> {
    check(instance, basis, cofactors, &Order::Grevlex, Mode::Exact).map_err(|e| format!("{e:?}"))
}

#[test]
fn canonical() {
    assert_eq!(ruled(INSTANCE, BASIS, COFACTORS), Ok(holds()));
}

#[test]
fn basis_spellings() {
    let spellings = [
        "[ x + y ,\n\n y^2 + 1 ] :",
        "[y+x,1+y^2]:",
        "[x+2*y-y,\ny^2+3-2]:",
        "[x^1+y^1*1,\ny*y+x^0]:",
        "[x+8*y,\ny^2-6]:",
        "[x+2/2*y,\n1/3*3*y^2+4/4]:",
        "# a comment\n#another one\n[x+y,\ny^2+1]:\n",
        "[x+y,\ny^2+1]:",
        "[x+y,y^2+1]:  \n\n",
        "\t[x+y,\r\ny^2+1]:\r\n",
    ];
    let failures: Vec<String> = spellings
        .iter()
        .filter_map(|b| match ruled(INSTANCE, b, COFACTORS) {
            Ok(v) if v == holds() => None,
            other @ (Ok(_) | Err(_)) => Some(format!("{b:?}: {other:?}")),
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

#[test]
fn cofactor_spellings() {
    let spellings = [
        "[[ 1 , 0 ],\n[0,1]]:",
        "[[8,7],[0,1]]:",
        "#comment\n[[1,0],[0,1]]:\n",
        "[[1,0],[0,3/3]]:",
        "[[1,x-x],[0,1]]:",
        "[[-6,0],[0,1+7*x*y]]:",
    ];
    let failures: Vec<String> = spellings
        .iter()
        .filter_map(|c| match ruled(INSTANCE, BASIS, c) {
            Ok(v) if v == holds() => None,
            other @ (Ok(_) | Err(_)) => Some(format!("{c:?}: {other:?}")),
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

#[test]
fn instance_spellings() {
    let spellings = [
        "x , y\n7\nx+y,\ny^2+1\n",
        "x,y\n7\nx+\ny,\ny^2\n+1\n",
        "x,y\n7\nx+y,y^2+1",
        "x,y\n7\n8*x+y,\ny^2+15\n",
        "x,y\n7\n-6*x+y,\ny^2-13\n",
        "x,y\n 7 \n\nx+y,\n\ny^2+1\n\n",
    ];
    let failures: Vec<String> = spellings
        .iter()
        .filter_map(|i| match ruled(i, BASIS, COFACTORS) {
            Ok(v) if v == holds() => None,
            other @ (Ok(_) | Err(_)) => Some(format!("{i:?}: {other:?}")),
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

/// The elements in another order, with the rows in the same order.
#[test]
fn element_order() {
    assert_eq!(
        ruled(INSTANCE, "[y^2+1,\nx+y]:\n", "[[0,1],\n[1,0]]:\n"),
        Ok(holds())
    );
}

#[test]
fn variable_names() {
    let got = ruled(
        "a_1,b2,Z\n7\na_1+b2,\nb2^2+1\n",
        "[a_1+b2,\nb2^2+1]:\n",
        COFACTORS,
    );
    assert_eq!(got, Ok(holds()));
}

#[test]
fn scaled_element() {
    let v = ruled(INSTANCE, "[2*x+2*y,\ny^2+1]:\n", "[[2,0],\n[0,1]]:\n").unwrap();
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(0)));
}

#[test]
fn malformed_basis() {
    let bad = [
        "[x+z,\ny^2+1]:",
        "[x+y,y^2+1]",
        "x+y,y^2+1]:",
        "[x+y,y^2+1: ",
        "[x+y,y^2+1]: x",
        "[x+y,,y^2+1]:",
        "[x+y,y^2+1,]:",
        "[,x+y,y^2+1]:",
        "[x^+y,y^2+1]:",
        "[x+y*,y^2+1]:",
        "[*x+y,y^2+1]:",
        "[x y,y^2+1]:",
        "[x^-1+y,y^2+1]:",
        "[x+-y,y^2+1]:",
        "[x+y,y^2+1/7]:",
        "[x+y,y^2+1/14]:",
        "[x+y,y^2+1/]:",
        "[x^4294967296+y,y^2+1]:",
        "[x^4294967295*x+y,y^2+1]:",
        "[x^4294967295*y+y,y^2+1]:",
        "[x+y,y^2+1]:[x]:",
        "",
    ];
    let accepted: Vec<String> = bad
        .iter()
        .filter_map(|b| {
            ruled(INSTANCE, b, COFACTORS)
                .ok()
                .map(|v| format!("{b:?}: {v:?}"))
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "read as results:\n{accepted}",
        accepted = accepted.join("\n")
    );
}

#[test]
fn malformed_cofactors() {
    let bad = [
        "[[1,z],[0,1]]:",
        "[[1,0],[0,1]]",
        "[[1,0][0,1]]:",
        "[1,0]:",
        "[[1,0],[0,1]]: 1",
        "[[1,0],[0,1]:",
        "[[1,,0],[0,1]]:",
        "[[1,0],[0,1/7]]:",
        "",
    ];
    let accepted: Vec<String> = bad
        .iter()
        .filter_map(|c| {
            ruled(INSTANCE, BASIS, c)
                .ok()
                .map(|v| format!("{c:?}: {v:?}"))
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "read as results:\n{accepted}",
        accepted = accepted.join("\n")
    );
}

#[test]
fn malformed_instance() {
    let bad = [
        "x,x\n7\nx+y,\ny^2+1\n",
        "x,y\n7a\nx+y,\ny^2+1\n",
        "x,y\n1\nx+y,\ny^2+1\n",
        "x,y\n18446744073709551616\nx+y,\ny^2+1\n",
        "x,y\n-7\nx+y,\ny^2+1\n",
        "\n7\nx+y,\ny^2+1\n",
        "x,,y\n7\nx+y,\ny^2+1\n",
        "1x,y\n7\nx+y,\ny^2+1\n",
        "x,y\n7\nx+z,\ny^2+1\n",
        "x,y\n7\nx+y,\n,y^2+1\n",
        "x,y\n",
        "",
    ];
    let accepted: Vec<String> = bad
        .iter()
        .filter_map(|i| {
            ruled(i, BASIS, COFACTORS)
                .ok()
                .map(|v| format!("{i:?}: {v:?}"))
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "read as results:\n{accepted}",
        accepted = accepted.join("\n")
    );
}
