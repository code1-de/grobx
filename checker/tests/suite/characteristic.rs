//! instance-grammar-agreement, the checker's test code: a characteristic
//! that is neither 0 nor a prime below 2^64 is refused as the instance is
//! read, as a call refuses it (instance-grammar-agreement.3, .4; plan.md,
//! _Test code_, 1).

use grobx_checker::{Error, Mode, Order, Verdict, check};

/// The check of F = {x} over the characteristic `n`, with G = {x} and the
/// cofactor 1.
fn over(n: &str) -> Result<Verdict, Error> {
    check(
        &format!("x\n{n}\nx\n"),
        "[x]:\n",
        "[[1]]:\n",
        &Order::Grevlex,
        Mode::Exact,
    )
}

fn holds() -> Verdict {
    Verdict {
        groebner: Ok(()),
        reduced: Ok(()),
    }
}

fn refused() -> Result<Verdict, Error> {
    Err(Error::Characteristic { line: 2 })
}

/// Whether n is prime, by trial division.
fn prime(n: u64) -> bool {
    n >= 2
        && (2..)
            .take_while(|d| d * d <= n)
            .all(|d| !n.is_multiple_of(d))
}

/// Every n below 2^16: a prime and 0 are read, 1 and every composite are
/// refused.
#[test]
fn the_characteristics_below_2_16() {
    let mut failures = Vec::new();
    for n in 0u64..1 << 16 {
        let want = if n == 0 || prime(n) {
            Ok(holds())
        } else {
            refused()
        };
        let got = over(&n.to_string());
        if got != want {
            failures.push(format!("{n}: {got:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

/// The primes and composites of the plan's edge cases, most past 2^16:
/// Carmichael numbers, strong pseudoprimes to the small bases, squares and
/// products of primes near 2^32, and the ends of the range.
#[test]
fn the_characteristics_past_2_16() {
    let primes = [
        "65521",
        "2147483647",
        "4294967279",
        "4294967291",
        "2305843009213693951",
        "18446744073709551557",
        "0018446744073709551557",
    ];
    let composites = [
        // 2047 is a strong pseudoprime to base 2; 561 and 1105 are
        // Carmichael numbers.
        "561",
        "1105",
        "2047",
        "65535",
        "4294967297",
        // sorenson-2017 §4.1: strong pseudoprimes found in the search for
        // ψ₁₂, 747451·149491·34233211 and 787711·214831·1360591.
        "3825123056546413051",
        "230245660726188031",
        // 4294967291² and 4294967291·4294967279, below 2^64.
        "18446744030759878681",
        "18446743979220271189",
        "4611686018427387904",
        "18446744073709551615",
    ];
    let past = ["18446744073709551616", "1000000000000000000000000000000"];
    for n in primes {
        assert_eq!(over(n), Ok(holds()), "{n}");
    }
    for n in composites.iter().chain(&past) {
        assert_eq!(over(n), refused(), "{n}");
    }
}

/// The refusal comes before the generators are read: a generator the
/// grammar refuses after a composite characteristic is not what the check
/// returns.
#[test]
fn the_characteristic_before_the_generators() {
    let got = check(
        "x\n6\nx+,\n",
        "[x]:\n",
        "[[1]]:\n",
        &Order::Grevlex,
        Mode::Exact,
    );
    assert_eq!(got, refused());
}
