//! The input grammar over Q (checker-gfp.3): spellings that denote the same
//! result get the same verdict, and malformed text is an error.

use grobx_checker::{Mode, Order, Verdict, check};

const INSTANCE: &str = "x,y\n0\nx+y,\ny^2+1\n";
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

/// The results that do not hold, each with what the checker said.
fn not_holding<'a>(
    texts: &[&'a str],
    rule: impl Fn(&'a str) -> Result<Verdict, String>,
) -> Vec<String> {
    texts
        .iter()
        .filter_map(|t| match rule(t) {
            Ok(v) if v == holds() => None,
            other @ (Ok(_) | Err(_)) => Some(format!("{t:?}: {other:?}")),
        })
        .collect()
}

/// The texts read as results, each with its verdict.
fn accepted<'a>(
    texts: &[&'a str],
    rule: impl Fn(&'a str) -> Result<Verdict, String>,
) -> Vec<String> {
    texts
        .iter()
        .filter_map(|t| rule(t).ok().map(|v| format!("{t:?}: {v:?}")))
        .collect()
}

#[test]
fn canonical() {
    assert_eq!(ruled(INSTANCE, BASIS, COFACTORS), Ok(holds()));
}

#[test]
fn basis_spellings() {
    let spellings = [
        "[2/2*x+y,\ny^2+1]:",
        "[1/2*x+1/2*x+y,\ny^2+1]:",
        "[x+3/6*y+1/2*y,\ny^2+2/2]:",
        "[x+y+100000000000000000000000-100000000000000000000000,\ny^2+1]:",
        "[x+y,\ny^2+3/2-1/2]:",
        "[4/4*x+y,\ny^2+1/1]:",
        "[x+y,\ny^2+340282366920938463463374607431768211457-340282366920938463463374607431768211456]:",
        "[x+18446744073709551617*y-18446744073709551616*y,\ny^2+1]:",
        "[x+2*y*1/2,\ny*y+1]:",
        "[x+y,y^2+1]:",
        "[y+x,\n1+y^2]:",
        "[ x + y ,\n\n y^2 + 1 ] :",
    ];
    let failures = not_holding(&spellings, |b| ruled(INSTANCE, b, COFACTORS));
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

#[test]
fn cofactor_spellings() {
    let spellings = [
        "[[1/1,0/5],\n[0,2/2]]:",
        "[[1,x-x],\n[0,1]]:",
        "[[3/3,0],\n[0,1+0*x*y]]:",
        "[[1,100000000000000000000000000-100000000000000000000000000],\n[0,1]]:",
        "[[1/2+1/2,0],\n[0,1]]:",
        "[[-1/2+3/2,0],[0,1]]:",
    ];
    let failures = not_holding(&spellings, |c| ruled(INSTANCE, BASIS, c));
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

#[test]
fn instance_spellings() {
    let spellings = [
        "x,y\n0\n2/2*x+y,\ny^2+1\n",
        "x,y\n0\nx+y,\ny^2+2-1\n",
        "x,y\n0\n18446744073709551617*x-18446744073709551616*x+y,\ny^2+1\n",
        "x,y\n 0 \n\nx+y,\n\ny^2+1\n\n",
    ];
    let failures = not_holding(&spellings, |i| ruled(i, BASIS, COFACTORS));
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

/// Generators scaled by constants: the cofactors scale back.
#[test]
fn scaled_generators() {
    assert_eq!(
        ruled(
            "x,y\n0\n2*x+2*y,\n3*y^2+3\n",
            BASIS,
            "[[1/2,0],\n[0,1/3]]:\n"
        ),
        Ok(holds())
    );
    assert_eq!(
        ruled(
            "x,y\n0\n1/2*x+1/2*y,\ny^2+1\n",
            BASIS,
            "[[2,0],\n[0,1]]:\n"
        ),
        Ok(holds())
    );
}

#[test]
fn malformed_basis() {
    let bad = [
        "[x+y,\ny^2+1/0]:",
        "[x+y,\ny^2+0/0]:",
        "[x+y,\ny^2+1/]:",
        "[x/2+y,\ny^2+1]:",
        "[x+y,\ny^2+1/-2]:",
        "[x+y,\ny^2+1.5]:",
        "[x+y,\ny^2+1/2/3]:",
        "[x+y,\ny^2+1e3]:",
        "[x+y,\ny^2+/2]:",
    ];
    let read = accepted(&bad, |b| ruled(INSTANCE, b, COFACTORS));
    assert!(
        read.is_empty(),
        "read as results:\n{read}",
        read = read.join("\n")
    );
}

#[test]
fn malformed_cofactors() {
    let bad = [
        "[[1/0,0],\n[0,1]]:",
        "[[1,0],\n[0,x/1]]:",
        "[[1,0],\n[0,1/0*x]]:",
    ];
    let read = accepted(&bad, |c| ruled(INSTANCE, BASIS, c));
    assert!(
        read.is_empty(),
        "read as results:\n{read}",
        read = read.join("\n")
    );
}

#[test]
fn malformed_instance() {
    let bad = [
        "x,y\n0\nx+1/0*y,\ny^2+1\n",
        "x,y\n-0\nx+y,\ny^2+1\n",
        "x,y\n0/1\nx+y,\ny^2+1\n",
        "x,y\n0\nx+y/2,\ny^2+1\n",
    ];
    let read = accepted(&bad, |i| ruled(i, BASIS, COFACTORS));
    assert!(
        read.is_empty(),
        "read as results:\n{read}",
        read = read.join("\n")
    );
}
