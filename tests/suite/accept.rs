//! The item's test code: the zero and the unit ideal, input already a
//! Gröbner basis, a zero generator, an eliminable variable, duplicates and
//! scalar multiples, one variable, small primes and the largest below 2^31,
//! coefficients as written, and 40 random systems over five primes, each
//! computed with cofactors and accepted by the checker in the exact mode (`OBJECTIVES.md`, _Domain specifics the
//! workflow points at_: the oracle).

use crate::common::{P, PRIMES, accept, dense, random_system};

/// The seed of every call of `accept` here.
const SEED: u64 = 0x00ac_ce97;

#[test]
fn the_zero_ideal() {
    accept("zero", "x,y\n65521\n0\n", Some("[]:\n"), SEED);
    accept("zeros", "x,y\n7\n0,\n0\n", Some("[]:\n"), SEED);
    accept("none", "x,y\n65521\n", Some("[]:\n"), SEED);
}

#[test]
fn the_unit_ideal() {
    accept("unit", "x,y\n65521\nx*y-1,\nx\n", Some("[1]:\n"), SEED);
    accept("unit-p2", "x\n2\nx,\nx+1\n", Some("[1]:\n"), SEED);
    accept("unit-constant", "x,y\n65521\n5\n", Some("[1]:\n"), SEED);
}

#[test]
fn input_already_a_groebner_basis() {
    accept(
        "gb",
        "x,y\n65521\nx+y,\ny^2+1\n",
        Some("[x+y,\ny^2+1]:\n"),
        SEED,
    );
    let text = dense(3, 3, P, 1);
    let first = accept("cubic-3", &text, None, SEED);
    let names = text.lines().next().unwrap();
    let elements = first
        .trim_end()
        .trim_start_matches('[')
        .trim_end_matches("]:");
    accept(
        "gb-cubic-3",
        &format!("{names}\n{P}\n{elements}\n"),
        Some(&first),
        SEED,
    );
}

#[test]
fn a_zero_generator() {
    accept(
        "zero-generator",
        "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n",
        None,
        SEED,
    );
}

#[test]
fn an_eliminable_variable() {
    accept(
        "eliminable-first",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        None,
        SEED,
    );
    accept(
        "eliminable-last",
        "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        None,
        SEED,
    );
    accept(
        "eliminable-linear",
        "x,y,z\n65521\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n",
        None,
        SEED,
    );
}

#[test]
fn duplicates_and_scalar_multiples() {
    let want = accept(
        "tiny",
        "x,y\n65521\nx^2+y,\nx*y+1\n",
        Some("[y^2+65520*x,\nx*y+1,\nx^2+y]:\n"),
        SEED,
    );
    accept(
        "duplicates",
        "x,y\n65521\nx^2+y,\nx^2+y,\nx*y+1\n",
        Some(&want),
        SEED,
    );
    accept(
        "multiples",
        "x,y\n65521\n2*x^2+2*y,\nx*y+1,\n3*x*y+3\n",
        Some(&want),
        SEED,
    );
}

#[test]
fn one_variable() {
    accept(
        "gcd",
        &format!("x\n{P}\nx^3-1,\nx^2-1\n"),
        Some("[x+65520]:\n"),
        SEED,
    );
}

#[test]
fn small_primes_and_the_largest() {
    accept("p2", "x,y,z\n2\nx*y+z,\nx*z+y,\ny*z+x\n", None, SEED);
    accept(
        "p3",
        "x,y,z\n3\nx^2+y*z+1,\ny^2+x*z,\nz^2+x*y+2\n",
        None,
        SEED,
    );
    accept("p7", "x,y,z\n7\nx^3-y*z,\ny^3-x*z,\nz^3-x*y\n", None, SEED);
    accept(
        "p31",
        "x,y,z\n2147483647\nx^2+123456789*y*z+1,\ny^2-x*z,\nz^2+x*y+2\n",
        None,
        SEED,
    );
}

#[test]
fn coefficients_as_written() {
    accept(
        "long",
        "x,y\n65521\n123456789012345678901234567890*x*y+1,\nx^2-y\n",
        None,
        SEED,
    );
    accept("fractions", "x,y\n7\n1/2*x^2+1/3*y,\nx*y-1/5\n", None, SEED);
    accept("spelling", "x,y\n65521\n1+y*x*x , y^2 - x*x\n", None, SEED);
}

#[test]
fn random_systems() {
    for p in PRIMES {
        for seed in 0..8 {
            accept(
                &format!("random-p{p}-{seed}"),
                &random_system(p, 1000 + seed),
                None,
                SEED,
            );
        }
    }
}
