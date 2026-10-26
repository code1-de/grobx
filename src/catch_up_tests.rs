//! q-crt-time: the lift over Q gives the basis of an element kept early,
//! and the checker accepts its results with cofactors. The searches for
//! the cases of `kept_unchecked_tests.rs`, which `chain_tests.rs` runs
//! too, are here, with the helpers of `chain_tests.rs`'s own searches.

use std::num::NonZeroUsize;
use std::time::Duration;

use grobx_checker::{Mode, Order, check};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Pow, ToPrimitive};

use crate::lift::{Report, lift_text};
use crate::rational::{Rational, bound, reconstruct};
use crate::systems::{dense, random_square};
use crate::{MonomialOrder, Outcome, Params, Threads};

/// Primes near 2^31, lucky for every case of `chain_tests.rs` and
/// `kept_unchecked_tests.rs`, which take them from here.
pub(crate) const L1: u64 = 2_147_483_647;
pub(crate) const L2: u64 = 2_147_483_629;
pub(crate) const L3: u64 = 2_147_483_587;

/// Every lift here with a deadline of 60 s, so that a lift that never
/// completes fails its test.
fn params(seed: u64) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    p.deadline = Some(Duration::from_secs(60));
    p
}

fn with_cofactors(seed: u64) -> Params {
    let mut p = params(seed);
    p.cofactors = true;
    p
}

/// The outcome of the lift of `text`, which must hold a basis.
fn lift(text: &str, params: &Params) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text(text, params, Vec::new(), &mut report)
        .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    outcome
}

fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// B = 10^30 + 57, a constant that Wang's reconstruction takes four replay
/// primes of [2^61, 2^62) to reconstruct, about seven near 2^31, and the
/// chain's two, about four near 2^31, so that its element x − B stays open
/// while another is kept.
pub(crate) fn b() -> BigUint {
    BigUint::from(10u32).pow(30u32) + 57u32
}

/// −B, the coefficient of x − B past its leading one.
pub(crate) fn minus_b() -> BigInt {
    -BigInt::from(b())
}

/// The primes from 101 below `limit`, by trial division.
pub(crate) fn small_primes(limit: u64) -> Vec<u64> {
    (101..limit)
        .filter(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0))
        .collect()
}

/// a mod m, in [0, m).
pub(crate) fn image(a: &BigInt, m: &BigUint) -> BigUint {
    a.mod_floor(&BigInt::from(m.clone()))
        .to_biguint()
        .expect("a floor remainder is not negative")
}

/// Wang's reconstruction of a mod m.
pub(crate) fn wang(a: &BigInt, m: &BigUint) -> Option<Rational> {
    reconstruct(&image(a, m), m, &bound(m))
}

/// Whether r reduces to a mod p: p divides no denominator of r, and r ≡ a.
pub(crate) fn fits(r: &Rational, a: &BigInt, p: u64) -> bool {
    r.residue(p) == image(a, &BigUint::from(p)).to_u64()
}

/// Whether x − B's element is not kept at p₂ from its residue modulo p₁:
/// Wang's reconstruction of −B mod p₁ is None or does not fit −B at p₂.
/// Otherwise both elements of a case of `kept_unchecked_tests.rs` or
/// `chain_tests.rs` could be kept at p₂, and the lifting would be complete
/// there.
pub(crate) fn b_open(p1: u64, p2: u64) -> bool {
    wang(&minus_b(), &BigUint::from(p1)).is_none_or(|s| !fits(&s, &minus_b(), p2))
}

/// An integer c from 1,000,003 and primes p₁, p₂, p₃ from 101 below 1200,
/// distinct and none dividing c, such that Wang's reconstruction r of
/// c mod p₁ is not c and fits c at p₂, while r does not fit c at p₃ and p₃
/// divides no denominator of r. Wang's reconstruction of c mod p₁·p₂ is
/// None or does not fit c at p₃ either, and x − B's element is not kept at
/// p₂ (`b_open`). Returns (c, p₁, p₂, p₃).
pub(crate) fn wrong_keep() -> (i64, u64, u64, u64) {
    let primes = small_primes(1200);
    for c in 1_000_003i64.. {
        let a = BigInt::from(c);
        let divides = |p: u64| (c as u64).is_multiple_of(p);
        for &p1 in &primes {
            if divides(p1) {
                continue;
            }
            let Some(r) = wang(&a, &BigUint::from(p1)) else {
                continue;
            };
            if r == Rational::new(a.clone(), BigInt::from(1)).unwrap() {
                continue;
            }
            for &p2 in &primes {
                if p2 == p1 || divides(p2) || !fits(&r, &a, p2) || !b_open(p1, p2) {
                    continue;
                }
                let m = BigUint::from(p1 * p2);
                let p3 = primes.iter().copied().find(|&p3| {
                    p3 != p1
                        && p3 != p2
                        && !divides(p3)
                        && r.residue(p3).is_some()
                        && !fits(&r, &a, p3)
                        && wang(&a, &m).is_none_or(|s| !fits(&s, &a, p3))
                });
                if let Some(p3) = p3 {
                    return (c, p1, p2, p3);
                }
            }
        }
    }
    unreachable!()
}

/// q-crt-time.1: the lift gives the basis of x − B and y − 1/2, whose
/// element y − 1/2 is kept before x − B.
#[test]
fn an_element_kept_early_through_the_lift() {
    let b = b();
    let text = format!("x,y\n0\nx-{b},\ny-1/2\n");
    let outcome = lift(&text, &params(1));
    assert_eq!(basis(&outcome), format!("[2*y-1,\nx-{b}]:\n"));
}

/// q-crt-time.1: the checker accepts the lift's basis and cofactors in
/// the exact mode on a dense cubic system and on random square systems
/// over Q.
#[test]
fn generated_systems_through_the_lift_with_cofactors() {
    let mut cases = vec![("cubic-3".to_string(), dense(3, 3, 0, 1))];
    for seed in 0..3 {
        cases.push((format!("square-{seed}"), random_square(0, seed)));
    }
    for (name, text) in &cases {
        let outcome = lift(text, &with_cofactors(7));
        let cofactors = outcome.cofactors.as_deref().expect("cofactors");
        let v = check(
            text,
            basis(&outcome),
            cofactors,
            &Order::Grevlex,
            Mode::Exact,
        )
        .unwrap_or_else(|e| panic!("{name}: the checker cannot read the result: {e:?}"));
        assert!(
            v.groebner.is_ok() && v.reduced.is_ok(),
            "{name}: the checker rejects the result: {v:?}"
        );
    }
}

/// `OBJECTIVES.md`, _Domain specifics the workflow points at_, over Q with cofactors: the checker
/// accepts each result in the exact mode.
#[test]
fn the_edge_cases_through_the_lift() {
    for (case, text) in [
        ("F = (0)", "x,y\n0\n0\n"),
        ("F = (0, 0)", "x,y\n0\n0,\n0\n"),
        ("no generator", "x,y\n0\n"),
        ("the unit ideal", "x,y\n0\nx*y-1/2,\nx,\ny+3\n"),
        ("a Gröbner basis already", "x,y\n0\n3*x+1,\n2*y^2-5\n"),
        (
            "the zero polynomial among the generators",
            "x,y\n0\nx^2+y,\n0,\nx*y+1\n",
        ),
        ("an eliminable variable", "x,y\n0\nx-1/3*y^2,\ny^3-2\n"),
    ] {
        let outcome = lift(text, &with_cofactors(3));
        let cofactors = outcome.cofactors.as_deref().expect("cofactors");
        let v = check(
            text,
            basis(&outcome),
            cofactors,
            &Order::Grevlex,
            Mode::Exact,
        )
        .unwrap_or_else(|e| panic!("{case}: the checker cannot read the result: {e:?}"));
        assert!(
            v.groebner.is_ok() && v.reduced.is_ok(),
            "{case}: the checker rejects the result: {v:?}\n{basis}\n{cofactors}",
            basis = basis(&outcome)
        );
    }
}
