//! q-zero-denominator, the item's test code: a call over ℚ whose instance
//! holds a denominator 0 returns `Error::NoInverse` at the denominator's
//! line, at once, with and without a bound; an earlier error of the text
//! comes first; denominators that are not 0 and the instances over GF(p)
//! read as before (plan.md, _Test code_).

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use grobx::{CheckerMode, Error, Outcome, Params};

use crate::common::{params, with_cofactors};

/// How long a call may take to return its error. The code before the item
/// returned none without a bound.
const PATIENCE: Duration = Duration::from_secs(20);

/// The call, in a thread of its own, waited for at most `PATIENCE`.
fn call(text: &str, params: &Params) -> Result<Outcome, Error> {
    let (tx, rx) = mpsc::channel();
    let (text, params) = (text.to_string(), params.clone());
    thread::spawn(move || {
        let _ = tx.send(grobx::groebner(&text, &params));
    });
    rx.recv_timeout(PATIENCE)
        .unwrap_or_else(|_| panic!("no answer within {PATIENCE:?}"))
}

/// The texts with a denominator 0 and its line: in the first generator and
/// in a later one, a product of fractions, 0/0, leading zeros, a line feed
/// after the `/`, and the edge cases of `OBJECTIVES.md`, _Domain
/// specifics the workflow points at_, each with a 0.
const ZEROS: &[(&str, usize)] = &[
    ("x\n0\nx+1/0\n", 3),
    ("x,y,z\n0\nx-y,\ny+z,\n\nz-3/0\n", 6),
    ("x,y\n0\nx*y-2/3*5/0*x\n", 3),
    ("x\n0\n0/0\n", 3),
    ("x\n0\nx+7/000\n", 3),
    ("x\n0\nx+1/\n0\n", 4),
    ("x,y\n0\nx*y-1/2,\nx,\ny+3/0\n", 5),
    ("x,y\n0\n3*x+1,\n2*y^2-5/0\n", 4),
    ("x,y\n0\nx^2+y,\n0,\nx*y+1/0\n", 5),
    ("x,y\n0\nx-1/3*y^2,\ny^3-2/0\n", 4),
];

fn assert_no_inverse(text: &str, line: usize, got: Result<Outcome, Error>, how: &str) {
    assert!(
        matches!(got, Err(Error::NoInverse { line: l }) if l == line),
        "{how}: {got:?}\n{text}"
    );
}

/// q-zero-denominator.1, .2: no bound, seeds 1 to 3.
#[test]
fn a_denominator_0_names_its_line() {
    for &(text, line) in ZEROS {
        for seed in 1..=3 {
            assert_no_inverse(text, line, call(text, &params(seed)), "no bound");
        }
    }
}

/// q-zero-denominator.1, .2: a deadline of 60 s, cofactors, and the
/// checker give the same error, and no `stopped`.
#[test]
fn with_a_bound() {
    for &(text, line) in ZEROS {
        let mut deadline = params(1);
        deadline.deadline = Some(Duration::from_secs(60));
        assert_no_inverse(text, line, call(text, &deadline), "deadline");
        assert_no_inverse(text, line, call(text, &with_cofactors(1)), "cofactors");
        let mut checker = params(1);
        checker.checker = Some(CheckerMode::Exact);
        assert_no_inverse(text, line, call(text, &checker), "checker");
    }
}

/// q-zero-denominator.3: the read's first error is the call's.
#[test]
fn an_earlier_error_comes_first() {
    let got = call("x\n0\nx+y,\nx-1/0\n", &params(1));
    assert!(
        matches!(got, Err(Error::UnknownVariable { line: 3, ref name }) if name == "y"),
        "{got:?}"
    );
    let got = call("x\n0\nx+*1,\nx-1/0\n", &params(1));
    assert!(matches!(got, Err(Error::Syntax { line: 3, .. })), "{got:?}");
    let text = "x\n0\nx-1/0,\nx+*1\n";
    assert_no_inverse(text, 3, call(text, &params(1)), "before a syntax error");
}

/// Denominators that are not 0, leading zeros among them, and the edge
/// cases without their 0, with the bases `kept_tests.rs` writes.
#[test]
fn denominators_that_are_not_0() {
    for (text, want) in [
        ("x\n0\nx+1/10\n", "[10*x+1]:\n"),
        ("x\n0\nx-1/0010\n", "[10*x-1]:\n"),
        ("x,y\n0\nx*y-1/2,\nx,\ny+3\n", "[1]:\n"),
        ("x,y\n0\n3*x+1,\n2*y^2-5\n", "[3*x+1,\n2*y^2-5]:\n"),
        ("x,y\n0\nx^2+y,\n0,\nx*y+1\n", "[y^2-x,\nx*y+1,\nx^2+y]:\n"),
        (
            "x,y\n0\nx-1/3*y^2,\ny^3-2\n",
            "[y^2-3*x,\n3*x*y-2,\n9*x^2-2*y]:\n",
        ),
    ] {
        let outcome = call(text, &params(1)).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
        assert_eq!(outcome.basis.as_deref(), Some(want), "{text}");
    }
}

/// Over GF(p) a denominator 0 or 0 mod p is `NoInverse` at its line, as
/// before.
#[test]
fn over_gfp_as_before() {
    for text in ["x\n7\nx+1/0\n", "x\n7\nx+1/7\n"] {
        assert_no_inverse(text, 3, call(text, &params(1)), "GF(7)");
    }
}

/// q-zero-denominator.1: the message names both causes.
#[test]
fn the_message() {
    assert_eq!(
        Error::NoInverse { line: 3 }.to_string(),
        "line 3: a denominator that is 0, or 0 mod p"
    );
}
