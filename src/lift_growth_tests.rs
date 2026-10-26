//! q-lift-growth-time, the item: the lift that keeps each coefficient's
//! images and converts them at an attempt returns the outcome it returned
//! from residues modulo M, over hundreds of primes, where an image lacks
//! a term, at primes below 2^31 among wide ones, and on the edge cases
//! (plan.md, _Test code_).

use grobx_checker::Order;
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::One;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::catch_up_tests::{L1, L2, L3};
use crate::chain_tests::{
    EDGE, assert_accepted, basis, both_reports as both, generated, params, with_cofactors,
};
use crate::field::is_prime_64;
use crate::lehmer::MARGIN;
use crate::mqrr_tests::of_bits;

/// The least prime at or above n.
fn prime_from(n: u64) -> u64 {
    (n..).find(|&q| is_prime_64(q)).unwrap()
}

/// The k least primes from 2^61 + `offset` on.
pub(crate) fn wide(k: usize, offset: u64) -> Vec<u64> {
    let mut ps: Vec<u64> = Vec::with_capacity(k);
    let mut n = (1 << 61) + offset;
    while ps.len() < k {
        let p = prime_from(n);
        ps.push(p);
        n = p + 2;
    }
    ps
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

/// The fewest and the most primes of [2^61, 2^62) whose product M passes
/// 2^`bits` with |v|·2^(T+1) < M for |v| < 2^`bits`, T the threshold of
/// M: the attempt that keeps v comes at the prime after them, so the lift
/// combines one more.
pub(crate) fn primes_for(bits: u64) -> (usize, usize) {
    // T = ⌈log₂ L(M)⌉ + MARGIN, L(M) between bits and bits + 62·2.
    let t = u64::from((bits + 2 * 62).bit_width()) + MARGIN;
    let need = bits + t + 1;
    let primes = |bits: u64| usize::try_from(need.div_ceil(bits)).unwrap();
    (primes(62), primes(61))
}

/// Test 7, q-lift-growth-time.1, .2, .4, .7: x − n and x + n, n of
/// 20,000 bits: the basis, from as many primes as the scaled residue over
/// 1 asks for and one more, and the lift of Wang's reconstruction alone
/// returns the same outcome.
#[test]
fn an_integer_of_many_primes() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6e07);
    let n = of_bits(&mut rng, 20_000);
    let (fewest, most) = primes_for(20_000);
    for sign in ['-', '+'] {
        let text = format!("x\n0\nx{sign}{n}\n");
        let case = format!("x {sign} n");
        let (outcome, chain, _) = both(&case, &text, &params(9), &[]);
        assert_eq!(basis(&outcome), format!("[x{sign}{n}]:\n"), "{case}");
        assert!(
            (fewest + 1..=most + 2).contains(&chain.combined.len()),
            "{case}: {len} primes, {fewest} to {most} asked for",
            len = chain.combined.len()
        );
    }
}

/// Test 8, q-lift-growth-time.2, .4, .5: d·x + a·y + b·z, d, a, and b of
/// 6,000 bits prime to each other: every attempt at a/d ran MQRR, and b
/// over the chain's d took the scaled residue, so the coefficients tried
/// are one more than the Euclidean runs. Beside it the guess: y + n₁/d₁
/// and x + n₂/(d₁·d₂), n₁, d₁, and n₂ of 2,000 bits and d₂ of 31: x's
/// element keeps within two primes of y's, as u·d₁ is n₂/d₂.
#[test]
fn a_rational_of_many_primes() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7a71);
    let d = coprime(&mut rng, 6_000, &[]);
    let a = coprime(&mut rng, 6_000, &[&d]);
    let b = coprime(&mut rng, 6_000, &[&d]);
    let text = format!("x,y,z\n0\n{d}*x+{a}*y+{b}*z\n");
    let (outcome, chain, _) = both("6,000 bits", &text, &params(10), &[]);
    assert_eq!(basis(&outcome), format!("[{d}*x+{a}*y+{b}*z]:\n"));
    assert_eq!(chain.reconstructed, chain.euclid + 1, "{chain:?}");
    let (_, most) = primes_for(12_000);
    assert!(chain.combined.len() <= most + 2, "{chain:?}");

    let d1 = coprime(&mut rng, 2_000, &[]);
    let n1 = coprime(&mut rng, 2_000, &[&d1]);
    let d2 = coprime(&mut rng, 31, &[&d1]);
    let dd = &d1 * &d2;
    let n2 = coprime(&mut rng, 2_000, &[&dd]);
    let text = format!("x,y\n0\n{dd}*x+{n2},\n{d1}*y+{n1}\n");
    let (outcome, chain, _) = both("the guess", &text, &params(10), &[]);
    assert_eq!(basis(&outcome), format!("[{d1}*y+{n1},\n{dd}*x+{n2}]:\n"));
    let (_, most) = primes_for(4_000);
    assert!(chain.combined.len() <= most + 3, "{chain:?}");
}

/// Test 9, q-lift-growth-time.1, .8: y + c·z + e·w, c a multiple of the
/// second forced wide prime and e of the first, each of about 300 bits,
/// over the learn prime L1 and two forced primes of [2^61, 2^62), then
/// drawn ones: the first image lacks w's term, which the part gains at the
/// second with an image 0 at the first, and the second lacks z's, appended
/// as 0. The basis, with cofactors and without; the checker accepts with
/// them.
#[test]
fn images_that_lack_a_term() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1ac7);
    let ws = wide(2, 7_000);
    let c = of_bits(&mut rng, 240) * ws[1];
    let e = of_bits(&mut rng, 240) * ws[0];
    let text = format!("y,z,w\n0\ny+{c}*z+{e}*w\n");
    let forced = [L1, ws[0], ws[1]];
    for params in [params(11), with_cofactors(11)] {
        let case = format!("cofactors {cofactors}", cofactors = params.cofactors);
        let (outcome, chain, _) = both(&case, &text, &params, &forced);
        assert_eq!(basis(&outcome), format!("[y+{c}*z+{e}*w]:\n"), "{case}");
        assert_eq!(chain.combined[..2], ws[..2], "{case}: {chain:?}");
        if params.cofactors {
            assert_accepted(&case, &text, &outcome, &Order::Grevlex);
        }
    }
}

/// Test 10: `OBJECTIVES.md`'s edge cases over ℚ, with cofactors and
/// without, through both lifts: one outcome, and the checker accepts with
/// cofactors in the exact mode.
#[test]
fn the_edge_cases() {
    for (case, text) in EDGE {
        both(case, text, &params(12), &[]);
        let (outcome, _, _) = both(case, text, &with_cofactors(12), &[]);
        assert_accepted(case, text, &outcome, &Order::Grevlex);
    }
}

/// Test 11, q-lift-growth-time.2, .3, .6: the generated systems over ℚ
/// with primes below 2^31 forced among primes of [2^61, 2^62), with
/// cofactors and without: one outcome through both lifts, and the checker
/// accepts with cofactors.
#[test]
fn generated_systems_at_mixed_primes() {
    let ws = wide(3, 91_000);
    let forced = [L1, L2, ws[0], L3, ws[1], ws[2]];
    for (case, text) in generated() {
        both(&case, &text, &params(13), &forced);
        let (outcome, _, _) = both(&case, &text, &with_cofactors(13), &forced);
        assert_accepted(&case, &text, &outcome, &Order::Grevlex);
    }
}
