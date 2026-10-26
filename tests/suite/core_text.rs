//! Sub-item 1: the reader of instances, through `grobx::text::generators`,
//! which writes the generators back monic in the canonical spelling. The
//! grammar is checker-gfp.3's (f4-gfp.43).

use grobx::Error;
use grobx::text::generators;

use crate::common::{P, P31, dense, random_square, random_system};

fn read(text: &str) -> String {
    generators(text).unwrap_or_else(|e| panic!("{text:?}: {e:?}"))
}

fn inverse(a: u128, p: u128) -> u128 {
    // a^(p − 2) mod p
    let (mut base, mut e, mut r) = (a % p, p - 2, 1u128);
    while e > 0 {
        if e & 1 == 1 {
            r = r * base % p;
        }
        base = base * base % p;
        e >>= 1;
    }
    r
}

#[test]
fn the_canonical_spelling() {
    assert_eq!(read("x,y\n65521\nx^2+y,\nx*y+1\n"), "[x^2+y,\nx*y+1]:\n");
    assert_eq!(read("x,y,z\n65521\nz+y^2*x+3\n"), "[x*y^2+z+3]:\n");
    assert_eq!(read("x,y\n7\n-x+1\n"), "[x+6]:\n");
    assert_eq!(read("x,y\n7\n3\n"), "[1]:\n");
    assert_eq!(read("x\n65521\nx^65535\n"), "[x^65535]:\n");
}

#[test]
fn whitespace_between_tokens() {
    assert_eq!(
        read("x , y\n 7 \n  x *y - 1 ,\n y^ 2 + 3 * x\n"),
        "[x*y+6,\ny^2+3*x]:\n"
    );
    assert_eq!(read("x,y\n65521\nx\n*\ny\n+\n1"), "[x*y+1]:\n");
}

#[test]
fn like_terms_and_repeated_factors() {
    // 3xy + x, made monic with 3⁻¹ = 5 over GF(7)
    assert_eq!(read("x,y\n7\nx*y+2*y*x+x\n"), "[x*y+5*x]:\n");
    assert_eq!(read("x,y\n7\nx*x*y\n"), "[x^2*y]:\n");
    assert_eq!(read("x,y\n7\nx^2*x*y^0\n"), "[x^3]:\n");
    assert_eq!(read("x,y\n65521\n1+y+x\n"), "[x+y+1]:\n");
    assert_eq!(read("x,y\n7\nx+8*x\n"), "[x]:\n");
}

#[test]
fn zero_generators_vanish() {
    assert_eq!(read("x\n7\nx-x,\nx+1\n"), "[x+1]:\n");
    assert_eq!(read("x\n7\n7*x\n"), "[]:\n");
    assert_eq!(read("x,y\n65521\n0\n"), "[]:\n");
    assert_eq!(read("x,y\n65521\n0,\n0\n"), "[]:\n");
    assert_eq!(read("x,y\n65521\n"), "[]:\n");
    assert_eq!(read("x,y\n65521\nx*y-x*y,\n0,\ny\n"), "[y]:\n");
}

#[test]
fn fractions_and_long_integers() {
    // 4x + 5 over GF(7), made monic: x + 5·4⁻¹ = x + 3
    assert_eq!(read("x\n7\n1/2*x+1/3\n"), "[x+3]:\n");
    assert_eq!(read("x\n7\n2/4*x\n"), "[x]:\n");
    let big = 123_456_789_012_345_678_901_234_567_890_u128;
    let p = u128::from(P);
    let c = big % p;
    let want = format!("[x+{constant}]:\n", constant = inverse(c, p));
    assert_eq!(read(&format!("x\n{P}\n{big}*x+1\n")), want);
    let q = u128::from(P31);
    let want = format!(
        "[x+{constant}]:\n",
        constant = (q - 1) * inverse(big % q, q) % q
    );
    assert_eq!(read(&format!("x\n{P31}\n{big}*x-1\n")), want);
}

#[test]
fn errors_name_their_line() {
    let err = |text: &str| generators(text).expect_err(text);
    assert!(matches!(err("x\n7\nx+\n"), Error::Syntax { line: 3, .. }));
    assert!(matches!(
        err("x\n7\nx,\nx+\n"),
        Error::Syntax { line: 4, .. }
    ));
    assert!(matches!(err("x\n7\nx,,x\n"), Error::Syntax { line: 3, .. }));
    assert!(matches!(err("x\n7\nx^\n"), Error::Syntax { line: 3, .. }));
    assert!(matches!(err("x,\n7\nx\n"), Error::Syntax { line: 1, .. }));
    assert!(matches!(
        err("x\nseven\nx\n"),
        Error::Syntax { line: 2, .. }
    ));
    assert!(matches!(
        err("x\n7\nx+y\n"),
        Error::UnknownVariable { line: 3, ref name } if name == "y"
    ));
    assert!(matches!(
        err("x,y,x\n7\nx\n"),
        Error::RepeatedVariable { ref name } if name == "x"
    ));
    assert!(matches!(err("x\n7\nx+1/7\n"), Error::NoInverse { line: 3 }));
    assert!(matches!(err("x\n7\nx+1/0\n"), Error::NoInverse { line: 3 }));
    assert!(matches!(
        err("x\n7\nx^2147483648\n"),
        Error::Overflow { line: Some(3) }
    ));
    assert!(matches!(
        err("x,y\n7\nx,\nx^1073741824*y^1073741824\n"),
        Error::Overflow { line: Some(4) }
    ));
    assert!(matches!(
        err("x,y\n7\nx^1073741824*y*x^1073741824\n"),
        Error::Overflow { line: Some(3) }
    ));
}

/// The reader takes the primes below 2^64 (f4-gfp.9, f4-gfp.10,
/// gfp-large-primes.10), and not 0: the engine takes Q, but
/// `text::generators` reads no instance over Q (q-lift.35). 2^61 − 1, a
/// prime, left the list with `gfp-large-primes`, which reads it.
#[test]
fn characteristics_the_reader_does_not_take() {
    for p in [
        "0",
        "1",
        "4",
        "65535",
        "2147483648",
        "4294967297",
        "18446744073709551615",
        "18446744073709551616",
    ] {
        let text = format!("x\n{p}\nx\n");
        assert!(
            matches!(generators(&text), Err(Error::Characteristic { .. })),
            "characteristic {p}: {got:?}",
            got = generators(&text)
        );
    }
}

/// Random, random square, and dense instances over GF(p) read, and their
/// generators, written as an instance of their own, read back to the same
/// text.
#[test]
fn generated_systems_read_back() {
    let mut texts = Vec::new();
    for p in [7, P, P31] {
        for seed in 0..3 {
            texts.push(random_system(p, seed));
            texts.push(random_square(p, seed));
        }
        texts.push(dense(3, 3, p, 1));
    }
    for text in texts {
        let written = read(&text);
        let mut lines = text.splitn(3, '\n');
        let (names, p) = (lines.next().unwrap(), lines.next().unwrap());
        let body = written
            .strip_prefix('[')
            .unwrap()
            .strip_suffix("]:\n")
            .unwrap();
        assert!(!body.is_empty(), "no generators in\n{text}");
        assert_eq!(read(&format!("{names}\n{p}\n{body}\n")), written, "{text}");
    }
}
