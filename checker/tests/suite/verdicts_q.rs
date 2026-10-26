//! Hand-built results whose verdicts are known, over Q.

use grobx_checker::{Flaw, Mode, Order, Verdict, check};

fn verdict(instance: &str, basis: &str, cofactors: &str) -> Verdict {
    check(instance, basis, cofactors, &Order::Grevlex, Mode::Exact)
        .unwrap_or_else(|e| panic!("refused {basis:?} with {cofactors:?}: {e:?}"))
}

fn holds() -> Verdict {
    Verdict {
        groebner: Ok(()),
        reduced: Ok(()),
    }
}

/// Over Q, S(xy + 65521y², x²) reduces to −65521²·y³, which no leading term
/// divides. Over GF(65521) the same text is {xy, x²}, a Gröbner basis.
#[test]
fn groebner_modulo_a_prime_only() {
    let basis = "[x*y+65521*y^2,\nx^2]:\n";
    let cofactors = "[[0,1],\n[1,0]]:\n";
    let v = verdict("x,y\n0\nx^2,\nx*y+65521*y^2\n", basis, cofactors);
    assert_eq!(v.groebner, Err(Flaw::SPair(0, 1)));
    assert_eq!(v.reduced, Ok(()));
    let v = verdict("x,y\n65521\nx^2,\nx*y+65521*y^2\n", basis, cofactors);
    assert_eq!(v, holds());
}

/// 2^64, which a `u64` would wrap to 0.
#[test]
fn groebner_modulo_2_64_only() {
    let v = verdict(
        "x,y\n0\nx^2,\nx*y+18446744073709551616*y^2\n",
        "[x*y+18446744073709551616*y^2,\nx^2]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(v.groebner, Err(Flaw::SPair(0, 1)));
    assert_eq!(v.reduced, Ok(()));
}

const MODULAR: &str = "x,y\n0\nx^2,\nx*y+65521*y^2\n";
const MODULAR_BASIS: &str = "[x*y+65521*y^2,\nx^2,\ny^3]:\n";

/// The reduced basis over Q, as Groebner.jl gives it: y³ =
/// y/65521²·x² + (−x/65521² + y/65521)·(xy + 65521y²).
#[test]
fn fraction_cofactors() {
    let v = verdict(
        MODULAR,
        MODULAR_BASIS,
        "[[0,1],\n[1,0],\n[1/4293001441*y,-1/4293001441*x+1/65521*y]]:\n",
    );
    assert_eq!(v, holds());
}

/// The last row off by x²y/10^30.
#[test]
fn cofactor_off_by_a_tiny_fraction() {
    let v = verdict(
        MODULAR,
        MODULAR_BASIS,
        "[[0,1],\n[1,0],\n[1/4293001441*y+1/1000000000000000000000000000000*y,-1/4293001441*x+1/65521*y]]:\n",
    );
    assert_eq!(v.groebner, Err(Flaw::Cofactors(2)));
    assert_eq!(v.reduced, Ok(()));
}

const HUGE: &str = "x,y\n0\nx-340282366920938463463374607431768211457*y,\ny^2-1180591620717411303427\n";
const HUGE_COFACTORS: &str = "[[1,0],\n[0,1]]:\n";

/// Coefficients 2^128 + 1 and 2^70 + 3.
#[test]
fn coefficients_past_2_128() {
    let v = verdict(
        HUGE,
        "[x-340282366920938463463374607431768211457*y,\ny^2-1180591620717411303427]:\n",
        HUGE_COFACTORS,
    );
    assert_eq!(v, holds());
}

/// A coefficient off by 2^64 leaves 2^64·y when the first generator is
/// reduced.
#[test]
fn coefficient_off_by_2_64() {
    let v = verdict(
        HUGE,
        "[x-340282366920938463481821351505477763073*y,\ny^2-1180591620717411303427]:\n",
        HUGE_COFACTORS,
    );
    assert_eq!(v.groebner, Err(Flaw::Generator(0)));
    assert_eq!(v.reduced, Ok(()));
}

#[test]
fn coefficient_off_by_2_128() {
    let v = verdict(
        HUGE,
        "[x-680564733841876926926749214863536422913*y,\ny^2-1180591620717411303427]:\n",
        HUGE_COFACTORS,
    );
    assert_eq!(v.groebner, Err(Flaw::Generator(0)));
    assert_eq!(v.reduced, Ok(()));
}

const INSTANCE: &str = "x,y\n0\nx+y,\ny^2+1\n";

/// A Gröbner basis whose element is not scaled as the canonical text: a common
/// factor, a negative leading coefficient, fractions.
#[test]
fn not_scaled() {
    let cases = [
        ("[2*x+2*y,\ny^2+1]:\n", "[[2,0],\n[0,1]]:\n", 0),
        ("[-x-y,\ny^2+1]:\n", "[[-1,0],\n[0,1]]:\n", 0),
        ("[1/2*x+1/2*y,\ny^2+1]:\n", "[[1/2,0],\n[0,1]]:\n", 0),
        ("[x+y,\n1/3*y^2+1/3]:\n", "[[1,0],\n[0,1/3]]:\n", 1),
    ];
    for (basis, cofactors, i) in cases {
        let v = verdict(INSTANCE, basis, cofactors);
        assert_eq!(v.groebner, Ok(()), "{basis:?}");
        assert_eq!(v.reduced, Err(Flaw::Scale(i)), "{basis:?}");
    }
}

#[test]
fn large_common_factor() {
    let v = verdict(
        "x\n0\nx\n",
        "[10000000000000000000000000000000000000000*x]:\n",
        "[[10000000000000000000000000000000000000000]]:\n",
    );
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(0)));
}

/// 6·(x/2 + y/3) = 3x + 2y and 4·(y² − 1/4) = 4y² − 1. Groebner.jl takes no
/// fraction in an instance, so this case is built by hand.
#[test]
fn fractions_in_the_instance() {
    let v = verdict(
        "x,y\n0\n1/2*x+1/3*y,\ny^2-1/4\n",
        "[3*x+2*y,\n4*y^2-1]:\n",
        "[[6,0],\n[0,4]]:\n",
    );
    assert_eq!(v, holds());
}

/// S(x² − y, xy − 1) = x − y², whose leading term y² no leading term divides.
#[test]
fn not_groebner() {
    let v = verdict(
        "x,y\n0\nx^2-y,\nx*y-1\n",
        "[x*y-1,\nx^2-y]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(v.groebner, Err(Flaw::SPair(0, 1)));
    assert_eq!(v.reduced, Ok(()));
}

/// Every pair's lcm is xyz, and the third element's leading term divides it.
/// A chain criterion that skipped every such pair would accept.
#[test]
fn chain_criterion_is_not_circular() {
    let basis = "[x*y+z,\nx*z+y,\ny*z+x]:\n";
    let v = verdict(
        "x,y,z\n0\nx*y+z,\nx*z+y,\ny*z+x\n",
        basis,
        "[[1,0,0],\n[0,1,0],\n[0,0,1]]:\n",
    );
    assert!(matches!(v.groebner, Err(Flaw::SPair(_, _))), "{v:?}");
    assert_eq!(v.reduced, Ok(()));
}

#[test]
fn divisible_tail() {
    let v = verdict(
        "x,y\n0\nx^2+y,\ny\n",
        "[y,\nx^2+y]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Divisible { element: 1, by: 0 }));
}

#[test]
fn equal_leading_terms() {
    let v = verdict("x\n0\nx\n", "[x,\nx]:\n", "[[1],\n[1]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert!(matches!(v.reduced, Err(Flaw::Divisible { .. })), "{v:?}");
}

/// A zero element adds nothing to the ideal, and is not scaled.
#[test]
fn zero_element() {
    let v = verdict("x\n0\nx\n", "[x,\n0]:\n", "[[1],\n[0]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(1)));
}

/// The zero ideal as Groebner.jl writes it.
#[test]
fn zero_ideal_with_zero_element() {
    let v = verdict("x,y\n0\n0\n", "[0]:\n", "[[0]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(0)));
}

#[test]
fn generator_outside() {
    let v = verdict("x,y\n0\nx,\ny\n", "[x]:\n", "[[1,0]]:\n");
    assert_eq!(v.groebner, Err(Flaw::Generator(1)));
    assert_eq!(v.reduced, Ok(()));
}

#[test]
fn another_ideal() {
    let v = verdict("x,y\n0\nx\n", "[y]:\n", "[[0]]:\n");
    assert!(v.groebner.is_err(), "{v:?}");
}

#[test]
fn shape() {
    for cofactors in ["[[1],\n[0]]:\n", "[[1,0]]:\n", "[]:\n", "[[]]:\n"] {
        let v = verdict("x\n0\nx\n", "[x]:\n", cofactors);
        assert_eq!(v.groebner, Err(Flaw::Shape), "{cofactors:?}");
        assert_eq!(v.reduced, Ok(()), "{cofactors:?}");
    }
}

#[test]
fn wrong_cofactor() {
    for cofactors in ["[[2]]:\n", "[[1/2]]:\n"] {
        let v = verdict("x\n0\nx\n", "[x]:\n", cofactors);
        assert_eq!(v.groebner, Err(Flaw::Cofactors(0)), "{cofactors:?}");
    }
}

/// F with no generator at all generates the zero ideal.
#[test]
fn no_generators() {
    let v = verdict("x,y\n0\n", "[]:\n", "[]:\n");
    assert_eq!(v, holds());
    let v = verdict("x,y\n0\n", "[1]:\n", "[[]]:\n");
    assert_eq!(v.groebner, Err(Flaw::Cofactors(0)));
}

/// 1 = −x + (x + 1), and {1, x} is a Gröbner basis that is not reduced.
#[test]
fn unit_ideal_not_reduced() {
    let v = verdict("x\n0\nx,\nx+1\n", "[1,\nx]:\n", "[[-1,1],\n[1,0]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Divisible { element: 1, by: 0 }));
}

/// 1 = −3/2·(2x) + (3x + 1).
#[test]
fn unit_ideal_from_fractions() {
    let v = verdict("x\n0\n2*x,\n3*x+1\n", "[1]:\n", "[[-3/2,1]]:\n");
    assert_eq!(v, holds());
}

#[test]
fn positive_dimensional() {
    let v = verdict(
        "x,y,z\n0\nx*y,\nx*z\n",
        "[x*z,\nx*y]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(v, holds());
}

/// A zero denominator is an error in each of the three texts.
#[test]
fn zero_denominator() {
    assert!(check("x\n0\n1/0*x\n", "[x]:\n", "[[1]]:\n", &Order::Grevlex, Mode::Exact).is_err());
    assert!(check("x\n0\nx\n", "[1/0*x]:\n", "[[1]]:\n", &Order::Grevlex, Mode::Exact).is_err());
    assert!(check("x\n0\nx\n", "[x]:\n", "[[1/0]]:\n", &Order::Grevlex, Mode::Exact).is_err());
}
