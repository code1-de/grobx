//! Hand-built results over GF(p) and Q on which the cheap mode reaches the
//! exact mode's result, and a false row that one point lets pass under
//! most seeds: rows the cheap mode checks exactly, a false row that
//! vanishes at most points, degrees and exponents past a `u32` and the
//! power table, zero denominators, and malformed or misshapen rows.

use std::num::NonZeroU32;

use grobx_checker::{Error, Flaw, Mode, Order, Part, Verdict, check, check_cofactors};

fn cheap(k: u32, seed: u64) -> Mode {
    Mode::Cheap {
        points: NonZeroU32::new(k).expect("k is at least 1"),
        seed,
    }
}

/// Both functions give the same in both modes, with k points under eight
/// seeds, and `check_cofactors` the same twice under one seed. Returns the
/// rows stage's result.
fn agree(instance: &str, basis: &str, cofactors: &str, k: u32) -> Result<Result<(), Flaw>, Error> {
    let full = check(instance, basis, cofactors, &Order::Grevlex, Mode::Exact);
    let rows = check_cofactors(instance, basis, cofactors, &Order::Grevlex, Mode::Exact);
    for seed in 0..8 {
        let mode = cheap(k, seed);
        let context = format!("{instance:?} {basis:?} {cofactors:?} {mode:?}");
        let got = check(instance, basis, cofactors, &Order::Grevlex, mode);
        assert_eq!(got, full, "check, {context}");
        let got = check_cofactors(instance, basis, cofactors, &Order::Grevlex, mode);
        assert_eq!(got, rows, "check_cofactors, {context}");
        let again = check_cofactors(instance, basis, cofactors, &Order::Grevlex, mode);
        assert_eq!(again, got, "the same seed twice, {context}");
    }
    rows
}

/// Over GF(2), g = f = x with the row x: the difference x² − x vanishes at
/// both points of GF(2). Its degree reaches p, so the row is checked exactly
/// (checker-cheap.3).
#[test]
fn degree_reaches_p_gf2() {
    let rows = agree("x\n2\nx\n", "[x]:\n", "[[x]]:\n", 64);
    assert_eq!(rows, Ok(Err(Flaw::Cofactors(0))));
}

/// Over GF(7), the row x^6: the difference x^7 − x vanishes on GF(7).
#[test]
fn degree_reaches_p_gf7() {
    let rows = agree("x\n7\nx\n", "[x]:\n", "[[x^6]]:\n", 288);
    assert_eq!(rows, Ok(Err(Flaw::Cofactors(0))));
}

const MANY_ROOTS: (&str, &str, &str) = ("x\n7\nx\n", "[1]:\n", "[[x^5]]:\n");

/// Over GF(7), F = {x}, G = {1}, the row x^5: the difference x^6 − 1, of
/// degree 6 < 7, vanishes at 6 of the 7 points. 288 points bound its
/// passing by (6/7)^288 < 2^-64 (checker-cheap.16).
#[test]
fn many_roots_gf7() {
    let (instance, basis, cofactors) = MANY_ROOTS;
    let rows = agree(instance, basis, cofactors, 288);
    assert_eq!(rows, Ok(Err(Flaw::Cofactors(0))));
}

/// The same row at one point passes under about 6/7 of the seeds: the
/// points are drawn from all of GF(7). Of 700 seeds, 600 are expected, with
/// a standard deviation of 9.3.
#[test]
fn many_roots_gf7_one_point() {
    let (instance, basis, cofactors) = MANY_ROOTS;
    let mut passed = 0;
    for seed in 0..700 {
        let got = check_cofactors(instance, basis, cofactors, &Order::Grevlex, cheap(1, seed));
        match got {
            Ok(Ok(())) => passed += 1,
            Ok(Err(Flaw::Cofactors(0))) => {}
            other @ (Ok(Err(_)) | Err(_)) => panic!("seed {seed}: {other:?}"),
        }
        let again = check_cofactors(instance, basis, cofactors, &Order::Grevlex, cheap(1, seed));
        assert_eq!(again, got, "seed {seed}: the same seed twice");
    }
    assert!((540..=660).contains(&passed), "{passed} of 700 passed");
}

/// F = {x^2147483647}, of the largest degree an instance's term may have,
/// G that element twice or once, and the row x^2147483649 as G's last: the
/// product passes `u32`, an `Overflow` at the row's line, 2 or 1, in both
/// modes (checker-cheap.4). The values of F and G
/// at the points need x^2147483647.
#[test]
fn degree_past_u32() {
    for p in [65521, 0] {
        let instance = format!("x\n{p}\nx^2147483647\n");
        let basis = "[x^2147483647]:\n";
        let cofactors = "[[1],\n[x^2147483649]]:\n";
        let expected = Err(Error::Overflow {
            part: Part::Cofactors,
            line: 2,
        });
        let got = agree(&instance, "[x^2147483647,\nx^2147483647]:\n", cofactors, 4);
        assert_eq!(got, expected, "p = {p}");
        let got = agree(&instance, basis, "[[x^2147483649]]:\n", 4);
        assert_eq!(
            got,
            Err(Error::Overflow {
                part: Part::Cofactors,
                line: 1
            }),
            "p = {p}"
        );
    }
}

/// F = G = {x^5000 − 1}: the powers past the table (checker-cheap.10). The
/// row 1 holds; the row 1 + x^3000 leaves x^8000 − x^3000, of degree 8000.
#[test]
fn exponents_past_the_table() {
    for (p, k) in [(65521, 22), (0, 4)] {
        let instance = format!("x\n{p}\nx^5000-1\n");
        let basis = "[x^5000-1]:\n";
        assert_eq!(agree(&instance, basis, "[[1]]:\n", k), Ok(Ok(())), "p = {p}");
        assert_eq!(
            agree(&instance, basis, "[[1+x^3000]]:\n", k),
            Ok(Err(Flaw::Cofactors(0))),
            "p = {p}"
        );
    }
}

/// A zero denominator in a row is `NoInverse` in both modes, `0/0`, the
/// entry of a zero generator, and a row past the last element included. A
/// syntax error after it in the same row comes later in both
/// (checker-cheap.17).
#[test]
fn zero_denominators() {
    let no_inverse = |line| {
        Err(Error::NoInverse {
            part: Part::Cofactors,
            line,
        })
    };
    let cases = [
        ("x\n7\nx\n", "[x]:\n", "[[1/7]]:\n", 1),
        ("x\n7\nx\n", "[x]:\n", "[[1],\n[0/7]]:\n", 2),
        ("x\n0\nx\n", "[x]:\n", "[[1/0]]:\n", 1),
        ("x\n0\nx\n", "[x]:\n", "[[0/0]]:\n", 1),
        ("x\n0\nx\n", "[x]:\n", "[[1+1/0*x]]:\n", 1),
        ("x,y\n0\nx,\n0\n", "[x]:\n", "[[1,\n1/0]]:\n", 2),
        ("x,y\n0\nx,\n0\n", "[x]:\n", "[[1,0],\n[0,0/0]]:\n", 2),
        ("x\n7\nx\n", "[x]:\n", "[[1/7+]]:\n", 1),
        ("x\n0\nx\n", "[x]:\n", "[[1/0+]]:\n", 1),
    ];
    for (instance, basis, cofactors, line) in cases {
        let got = agree(instance, basis, cofactors, 4);
        assert_eq!(got, no_inverse(line), "{instance:?} {cofactors:?}");
    }
}

/// Malformed cofactor texts are errors in both modes, a malformed row past
/// the last element among them, which both modes read.
#[test]
fn malformed_rows() {
    let bad = [
        "[[1]]",
        "[[z]]:",
        "[[x^]]:",
        "[[x^4294967296]]:",
        "[[x^4294967295*x]]:",
        "[[1/]]:",
        "[[1.5]]:",
        "[[x/2]]:",
        "[[1/-2]]:",
        "[1]:",
        "[[1]]: 1",
        "[[1],\n[x+]]:",
        "[[1]\n[1]]:",
        "[[1,,0]]:",
        "",
    ];
    for p in [7, 0] {
        let instance = format!("x\n{p}\nx\n");
        for cofactors in bad {
            let got = agree(&instance, "[x]:\n", cofactors, 23);
            assert!(got.is_err(), "p = {p}, {cofactors:?}: {got:?}");
        }
    }
}

/// A false row ends the check before a malformed one is read, in both
/// modes. At 23 points over GF(7), the false row's difference x passes with
/// probability (1/7)^23 < 2^-64.
#[test]
fn false_row_before_a_malformed_one() {
    for p in [7, 0] {
        let instance = format!("x\n{p}\nx\n");
        let got = agree(&instance, "[x]:\n", "[[2],\n[x+]]:\n", 23);
        assert_eq!(got, Ok(Err(Flaw::Cofactors(0))), "p = {p}");
    }
}

/// Spellings of the rows that denote the same matrix: like terms that cancel,
/// a coefficient that is 0 mod p or a fraction equal to 1, and zeros past
/// 2^64.
#[test]
fn row_spellings() {
    let instance = "x,y\n7\nx+y,\ny^2+1\n";
    let basis = "[x+y,\ny^2+1]:\n";
    for cofactors in [
        "[[1,x-x],[0,1]]:",
        "[[-6,0],[0,1+7*x^9*y^9]]:",
        "[[8,7],[0,1]]:",
        "[[1,0],[0,3/3]]:",
    ] {
        assert_eq!(agree(instance, basis, cofactors, 23), Ok(Ok(())), "{cofactors:?}");
    }
    let instance = "x,y\n0\nx+y,\ny^2+1\n";
    for cofactors in [
        "[[1,x-x],[0,1]]:",
        "[[1/1,0/5],[0,2/2]]:",
        "[[3/3,0],[0,1+0*x^9*y^9]]:",
        "[[1,100000000000000000000000000-100000000000000000000000000],[0,1]]:",
        "[[-1/2+3/2,0],[0,1]]:",
    ] {
        assert_eq!(agree(instance, basis, cofactors, 4), Ok(Ok(())), "{cofactors:?}");
    }
}

/// A row of the wrong length, a row too many, and a row too few.
#[test]
fn shape() {
    for p in [7, 0] {
        let instance = format!("x\n{p}\nx\n");
        for cofactors in ["[[1,0]]:\n", "[[1],\n[0]]:\n", "[]:\n", "[[]]:\n"] {
            let got = agree(&instance, "[x]:\n", cofactors, 23);
            assert_eq!(got, Ok(Err(Flaw::Shape)), "p = {p}, {cofactors:?}");
        }
    }
}

/// A zero element's row is read and not checked, in either mode.
#[test]
fn zero_element() {
    for p in [7, 0] {
        let instance = format!("x\n{p}\nx\n");
        let got = agree(&instance, "[0,\nx]:\n", "[[x^9+1],\n[1]]:\n", 23);
        assert_eq!(got, Ok(Ok(())), "p = {p}");
    }
}

/// F with no generators: the empty row expresses 0, not x.
#[test]
fn no_generators() {
    for p in [7, 0] {
        let instance = format!("x\n{p}\n");
        let got = agree(&instance, "[x]:\n", "[[]]:\n", 23);
        assert_eq!(got, Ok(Err(Flaw::Cofactors(0))), "p = {p}");
        assert_eq!(agree(&instance, "[]:\n", "[]:\n", 23), Ok(Ok(())), "p = {p}");
    }
}

/// 2^64 − 59, the largest prime below 2^64, which Groebner.jl does not
/// take: residues drawn below it (checker-cheap.14). The second matrix
/// leaves x·(y − 3) in row 1.
#[test]
fn largest_prime_below_2_64() {
    let instance = "x,y\n18446744073709551557\nx-2,\ny-3\n";
    let basis = "[y-3,\nx-2]:\n";
    assert_eq!(
        check(instance, basis, "[[0,1],\n[1,0]]:\n", &Order::Grevlex, Mode::Exact),
        Ok(Verdict {
            groebner: Ok(()),
            reduced: Ok(())
        })
    );
    assert_eq!(agree(instance, basis, "[[0,1],\n[1,0]]:\n", 2), Ok(Ok(())));
    assert_eq!(
        agree(instance, basis, "[[0,1],\n[1,x]]:\n", 2),
        Ok(Err(Flaw::Cofactors(1)))
    );
}

/// One point on a true result, under many seeds.
#[test]
fn one_point() {
    for p in [65521, 0] {
        let instance = format!("x,y\n{p}\nx+y,\ny^2+1\n");
        let basis = "[x+y,\ny^2+1]:\n";
        for seed in 0..100 {
            let rows = "[[1,0],\n[0,1]]:\n";
            let got = check(&instance, basis, rows, &Order::Grevlex, cheap(1, seed));
            assert_eq!(
                got,
                Ok(Verdict {
                    groebner: Ok(()),
                    reduced: Ok(())
                }),
                "p = {p}, seed {seed}"
            );
        }
    }
}
