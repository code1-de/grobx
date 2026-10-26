//! Hand-built results whose verdicts are known, over GF(p).

use grobx_checker::{Flaw, Mode, Order, Verdict, check};

fn verdict(instance: &str, basis: &str, cofactors: &str) -> Verdict {
    check(instance, basis, cofactors, &Order::Grevlex, Mode::Exact)
        .unwrap_or_else(|e| panic!("refused {basis:?} with {cofactors:?}: {e:?}"))
}

/// S(x^2 − y, xy − 1) = x − y^2, whose leading term y^2 no leading term
/// divides.
#[test]
fn not_groebner() {
    let v = verdict(
        "x,y\n7\nx^2-y,\nx*y-1\n",
        "[x*y+6,\nx^2+6*y]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(v.groebner, Err(Flaw::SPair(0, 1)));
    assert_eq!(v.reduced, Ok(()));
}

/// Every pair's lcm is xyz, and the third element's leading term divides it.
/// A chain criterion that skipped every such pair would accept; the
/// S-polynomial z^2 − y^2 of the first two does not reduce.
#[test]
fn chain_criterion_is_not_circular() {
    let basis = "[x*y+z,\nx*z+y,\ny*z+x]:\n";
    let v = verdict(
        "x,y,z\n7\nx*y+z,\nx*z+y,\ny*z+x\n",
        basis,
        "[[1,0,0],\n[0,1,0],\n[0,0,1]]:\n",
    );
    assert!(matches!(v.groebner, Err(Flaw::SPair(_, _))), "{v:?}");
    assert_eq!(v.reduced, Ok(()));
}

#[test]
fn divisible_tail() {
    let v = verdict(
        "x,y\n7\nx^2+y,\ny\n",
        "[y,\nx^2+y]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Divisible { element: 1, by: 0 }));
}

#[test]
fn equal_leading_terms() {
    let v = verdict("x\n7\nx\n", "[x,\nx]:\n", "[[1],\n[1]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert!(matches!(v.reduced, Err(Flaw::Divisible { .. })), "{v:?}");
}

#[test]
fn not_monic() {
    let v = verdict("x\n7\nx\n", "[3*x]:\n", "[[3]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(0)));
}

/// A zero element adds nothing to the ideal, and is not monic.
#[test]
fn zero_element() {
    let v = verdict("x\n7\nx\n", "[x,\n0]:\n", "[[1],\n[0]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(1)));
}

/// The zero ideal as Groebner.jl writes it.
#[test]
fn zero_ideal_with_zero_element() {
    let v = verdict("x,y\n7\n0\n", "[0]:\n", "[[0]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Scale(0)));
}

#[test]
fn generator_outside() {
    let v = verdict("x,y\n7\nx,\ny\n", "[x]:\n", "[[1,0]]:\n");
    assert_eq!(v.groebner, Err(Flaw::Generator(1)));
    assert_eq!(v.reduced, Ok(()));
}

#[test]
fn another_ideal() {
    let v = verdict("x,y\n7\nx\n", "[y]:\n", "[[0]]:\n");
    assert!(v.groebner.is_err(), "{v:?}");
}

#[test]
fn shape() {
    for cofactors in ["[[1],\n[0]]:\n", "[[1,0]]:\n", "[]:\n", "[[]]:\n"] {
        let v = verdict("x\n7\nx\n", "[x]:\n", cofactors);
        assert_eq!(v.groebner, Err(Flaw::Shape), "{cofactors:?}");
        assert_eq!(v.reduced, Ok(()), "{cofactors:?}");
    }
}

#[test]
fn wrong_cofactor() {
    let v = verdict("x\n7\nx\n", "[x]:\n", "[[2]]:\n");
    assert_eq!(v.groebner, Err(Flaw::Cofactors(0)));
}

/// F with no generator at all generates the zero ideal.
#[test]
fn no_generators() {
    let v = verdict("x,y\n7\n", "[]:\n", "[]:\n");
    assert_eq!(
        v,
        Verdict {
            groebner: Ok(()),
            reduced: Ok(())
        }
    );
    let v = verdict("x,y\n7\n", "[1]:\n", "[[]]:\n");
    assert_eq!(v.groebner, Err(Flaw::Cofactors(0)));
}

/// 1 = 6·x + (x + 1) over GF(7), and {1, x} is a Gröbner basis that is not
/// reduced.
#[test]
fn unit_ideal_not_reduced() {
    let v = verdict("x\n7\nx,\nx+1\n", "[1,\nx]:\n", "[[6,1],\n[1,0]]:\n");
    assert_eq!(v.groebner, Ok(()));
    assert_eq!(v.reduced, Err(Flaw::Divisible { element: 1, by: 0 }));
}

/// 2^64 − 59, the largest prime below 2^64, above what Groebner.jl takes.
/// (p − 1)^2 ≡ 1 makes the product of two residues near p.
#[test]
fn largest_prime_below_2_64() {
    let instance = "x,y\n18446744073709551557\nx-18446744073709551556*y,\ny^2-2\n";
    let basis = "[x+y,\ny^2+18446744073709551555]:\n";
    let cofactors = "[[18446744073709551556*18446744073709551556,0],\n[0,1]]:\n";
    let v = verdict(instance, basis, cofactors);
    assert_eq!(
        v,
        Verdict {
            groebner: Ok(()),
            reduced: Ok(())
        }
    );
}

/// instance-grammar-agreement.3: 6 is not prime, and the check returns an
/// error.
#[test]
fn composite_characteristic() {
    assert!(
        check(
            "x\n6\n2*x\n",
            "[2*x]:\n",
            "[[1]]:\n",
            &Order::Grevlex,
            Mode::Exact
        )
        .is_err()
    );
}

#[test]
fn positive_dimensional() {
    let v = verdict(
        "x,y,z\n7\nx*y,\nx*z\n",
        "[x*z,\nx*y]:\n",
        "[[0,1],\n[1,0]]:\n",
    );
    assert_eq!(
        v,
        Verdict {
            groebner: Ok(()),
            reduced: Ok(())
        }
    );
}
