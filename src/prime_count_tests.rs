//! q-prime-count-time, the item: the lift that reconstructs over each
//! part's chain denominator returns what the lift of Wang's reconstruction
//! alone returns, from fewer primes where the rationals' numerators and
//! denominators differ in size, where an element's denominator is a
//! multiple of the one before, and where an element's coefficients share
//! their denominator (plan.md, _Test code_).

use grobx_checker::Order;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::One;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::catch_up_tests::L1;
use crate::chain_tests::{
    EDGE, assert_accepted, basis, both_reports_as, generated, params, with_cofactors,
};
use crate::field::is_prime_64;
use crate::lift::{Replays, Report};
use crate::mqrr_tests::of_bits;
use crate::shortcut_tests::{image, product};
use crate::{Outcome, Params};

/// As `chain_tests::both_reports`, the replays alone: an attempt follows
/// every prime, as the counts here assume, where the replays in pairs
/// attempt once a pair (q-lift-rest-time.2).
fn both(case: &str, text: &str, params: &Params, forced: &[u64]) -> (Outcome, Report, Report) {
    both_reports_as(case, text, params, forced, Replays::Alone)
}

/// A random odd integer of exactly `bits` bits prime to every one of
/// `others`.
pub(crate) fn coprime(rng: &mut Xoshiro256PlusPlus, bits: u64, others: &[&BigUint]) -> BigUint {
    loop {
        let x = of_bits(rng, bits) | BigUint::one();
        if others.iter().all(|o| x.gcd(o).is_one()) {
            return x;
        }
    }
}

/// q-prime-count-time.1, .5 to .7: one coefficient n/d whose numerator
/// has 600 bits and denominator 31, and the other way round. MQRR asks
/// for about 600 + 31 + T bits of M, eleven primes of [2^61, 2^62), and
/// the attempt at the twelfth keeps it; Wang's bound asks for 1201 bits,
/// twenty primes.
#[test]
fn lopsided_rationals_take_fewer_primes() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x10b5);
    for (bn, bd) in [(600, 31), (31, 600)] {
        let d = coprime(&mut rng, bd, &[]);
        let n = coprime(&mut rng, bn, &[&d]);
        let text = format!("x\n0\n{d}*x-{n}\n");
        let case = format!("n of {bn} bits, d of {bd}");
        let (outcome, chain, wang) = both(&case, &text, &params(1), &[]);
        assert_eq!(basis(&outcome), format!("[{d}*x-{n}]:\n"), "{case}");
        assert!(chain.combined.len() <= 13, "{case}: {chain:?}");
        assert!(wang.combined.len() >= 20, "{case}: {wang:?}");
    }
}

/// q-prime-count-time.2: d·x + a·y + b·z with a, b, and d of 400 bits.
/// Once a/d is reconstructed, b over the chain's d is an integer of 400
/// bits that the scaled residue takes without a Euclidean step: every try
/// of a/d ran the Euclidean algorithm, and b was tried once.
#[test]
fn the_chain_carries_an_elements_other_coefficients() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc4a2);
    let d = coprime(&mut rng, 400, &[]);
    let a = coprime(&mut rng, 400, &[&d]);
    let b = coprime(&mut rng, 400, &[&d]);
    let text = format!("x,y,z\n0\n{d}*x+{a}*y+{b}*z\n");
    let (outcome, chain, _) = both("the chain", &text, &params(2), &[]);
    assert_eq!(basis(&outcome), format!("[{d}*x+{a}*y+{b}*z]:\n"));
    assert_eq!(chain.reconstructed, chain.euclid + 1, "{chain:?}");
}

/// q-prime-count-time.8: y + n₁/d₁ and x + n₂/(d₁·d₂), n₁ and d₁ of 200
/// bits, n₂ of 400, d₂ of 200. y's element comes first, and its d₁ is the
/// guess for x's: u·d₁ is n₂/d₂, which MQRR finds from about 600 + T bits
/// of M, eleven primes, at an attempt of odd count by the twelfth or the
/// thirteenth. Over u alone MQRR asks for 800 + T bits, and Wang's bound
/// for 801: fourteen primes, and the attempt at the fifteenth.
///
/// Beside it, y + n₁/d₁ and x + n₃/d₃ with d₃ of 31 bits prime to d₁ and
/// n₃ of 600: the guess d₁ makes x's rational n₃·d₁/d₃, of 831 bits, and
/// u alone at every other prime finds n₃/d₃ from about 631 + T bits,
/// eleven primes, by the thirteenth. Wang's bound asks for 1201 bits.
#[test]
fn the_previous_elements_denominator() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9e55);
    let d1 = coprime(&mut rng, 200, &[]);
    let n1 = coprime(&mut rng, 200, &[&d1]);
    let d2 = coprime(&mut rng, 200, &[&d1]);
    let dd = &d1 * &d2;
    let n2 = coprime(&mut rng, 400, &[&dd]);
    let text = format!("x,y\n0\n{dd}*x+{n2},\n{d1}*y+{n1}\n");
    let (outcome, chain, wang) = both("the guess", &text, &params(3), &[]);
    assert_eq!(basis(&outcome), format!("[{d1}*y+{n1},\n{dd}*x+{n2}]:\n"));
    assert!(chain.combined.len() <= 13, "{chain:?}");
    assert!(wang.combined.len() >= 14, "{wang:?}");

    let d3 = coprime(&mut rng, 31, &[&d1]);
    let n3 = coprime(&mut rng, 600, &[&d3]);
    let text = format!("x,y\n0\n{d3}*x+{n3},\n{d1}*y+{n1}\n");
    let (outcome, chain, wang) = both("an unrelated guess", &text, &params(3), &[]);
    assert_eq!(basis(&outcome), format!("[{d1}*y+{n1},\n{d3}*x+{n3}]:\n"));
    assert!(chain.combined.len() <= 13, "{chain:?}");
    assert!(wang.combined.len() >= 20, "{wang:?}");
}

/// q-prime-count-time.1, .2; `OBJECTIVES.md`'s input already a Gröbner
/// basis: x + y/d + z/(d·e), d of 160 bits and e of 60, prime to each
/// other. 1/d is reconstructed from three primes, and z's 1/e over the
/// chain's d at the same attempt makes the chain's last d·e, of 220 bits,
/// more limbs than M's three: its residue at the attempt's prime takes
/// every limb, and the attempt at the fourth prime keeps the element.
/// With cofactors, the cofactor 1/(d·e) waits for a fourth prime of M.
/// Wang's bound asks for 441 bits.
#[test]
fn a_chain_denominator_past_the_modulus() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xde1e);
    let d = coprime(&mut rng, 160, &[]);
    let e = coprime(&mut rng, 60, &[&d]);
    let de = &d * &e;
    let text = format!("x,y,z\n0\n{de}*x+{e}*y+z\n");
    for params in [params(4), with_cofactors(4)] {
        let (outcome, chain, wang) = both("past the modulus", &text, &params, &[]);
        assert_eq!(basis(&outcome), format!("[{de}*x+{e}*y+z]:\n"));
        let most = if params.cofactors { 5 } else { 4 };
        assert!(chain.combined.len() <= most, "{chain:?}");
        assert!(wang.combined.len() >= 8, "{wang:?}");
        if params.cofactors {
            assert_accepted("past the modulus", &text, &outcome, &Order::Grevlex);
        }
    }
}

/// The least prime at or above n.
fn prime_from(n: u64) -> u64 {
    (n..).find(|&q| is_prime_64(q)).unwrap()
}

/// q-prime-count-time.1, .2, q-rational-time.10: a chain's last that p
/// divides, and one that is no unit modulo M. In y + cz + ew, c is the
/// integer below p₁·p₂·p₃ that is n/p₄ modulo it, n of 16 bits, and the
/// primes of M are p₁ to p₅, p₄ of 21 bits and the rest of 41. The
/// attempt at p₃ takes the wrong n/p₄ for c from p₁·p₂, which fits c at
/// p₃, and fails on e, so the chain's last is p₄.
///
/// e of 70 bits: at the attempt at p₄, e·p₄ is the scaled residue's over
/// the last, whose residue at p₄ is 0, so e in lowest terms is checked.
/// The completion's check of n/p₄ fails there and clears the element,
/// and the attempt at p₅ keeps it: five primes. e of 110 bits: e·p₄ is
/// not small at p₁·p₂·p₃, and at the attempt at p₅ the last is no unit
/// modulo M, so e is scaled by 1 and taken over 1, and n/p₄ fails at p₅:
/// six primes. Scaled by the last, e·p₄ would not be small, and the
/// element would wait a prime more.
#[test]
fn a_chain_denominator_that_a_prime_of_m_divides() {
    let wide = |offset: u64| prime_from((1 << 40) + offset);
    let (p1, p2, p3, p4, p5) = (
        wide(1000),
        wide(5000),
        wide(9000),
        prime_from((1 << 20) + 100),
        wide(13_000),
    );
    let n = BigInt::from((1u32 << 15) + 3);
    let d = BigUint::from(p4);
    let c = image(&n, &d, &product(&[p1, p2, p3])).unwrap();
    assert_ne!(
        image(&n, &d, &BigUint::from(p5)),
        Some(&c % p5),
        "n/p₄ fits c at p₅"
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xd1f);
    for (bits, primes) in [(70, 5), (110, 6)] {
        let e = of_bits(&mut rng, bits);
        let text = format!("y,z,w\n0\ny+{c}*z+{e}*w\n");
        let case = format!("e of {bits} bits");
        let (outcome, chain, _) = both(&case, &text, &params(6), &[L1, p1, p2, p3, p4, p5]);
        assert_eq!(basis(&outcome), format!("[y+{c}*z+{e}*w]:\n"), "{case}");
        assert_eq!(
            chain.combined[..5],
            [p1, p2, p3, p4, p5],
            "{case}: {chain:?}"
        );
        assert_eq!(chain.combined.len(), primes, "{case}: {chain:?}");
    }
}

/// q-prime-count-time.3, .4: at one prime of [2^61, 2^62), T is
/// ⌈log₂ 62⌉ + 10 = 16, so an integer below 2^44 is the scaled residue's
/// at the attempt at the second prime, and one past 2^46, whose quotient
/// ⌊(M − c)/c⌋ after a first of 1 stays below 2^16, waits for the third.
/// Wang's bound takes neither from one prime.
#[test]
fn the_threshold_at_one_prime() {
    for (c, primes) in [((1u64 << 44) - 1, 2), ((1u64 << 46) + 1, 3)] {
        let text = format!("x\n0\nx-{c}\n");
        let case = format!("x − {c}");
        let (outcome, chain, wang) = both(&case, &text, &params(5), &[]);
        assert_eq!(basis(&outcome), format!("[x-{c}]:\n"), "{case}");
        assert_eq!(chain.combined.len(), primes, "{case}: {chain:?}");
        assert!(wang.combined.len() >= 3, "{case}: {wang:?}");
    }
}

/// q-prime-count-time.1 to .8 on the generated systems over Q, dense cubic
/// ones of 2 and 3 variables and random square ones: one outcome, without
/// cofactors and with, where the checker accepts in the exact mode. No
/// system takes more than one prime past Wang's lift, and all together
/// take at most three quarters of its primes.
#[test]
fn generated_systems_through_both_lifts() {
    let (mut ours, mut theirs) = (0, 0);
    for (case, text) in generated() {
        for (params, cofactors) in [(params(7), false), (with_cofactors(7), true)] {
            let (outcome, chain, wang) = both(&case, &text, &params, &[]);
            if cofactors {
                assert_accepted(&case, &text, &outcome, &Order::Grevlex);
            }
            assert!(
                chain.combined.len() <= wang.combined.len() + 1,
                "{case}: {chain:?}\n{wang:?}"
            );
            ours += chain.combined.len();
            theirs += wang.combined.len();
        }
    }
    assert!(4 * ours <= 3 * theirs, "{ours} primes against {theirs}");
}

/// `OBJECTIVES.md`'s edge cases over Q with cofactors, through both
/// lifts: one outcome, and the checker accepts in the exact mode.
#[test]
fn the_edge_cases() {
    for (case, text) in EDGE {
        let (outcome, _, _) = both(case, text, &with_cofactors(8), &[]);
        assert_accepted(case, text, &outcome, &Order::Grevlex);
    }
}
