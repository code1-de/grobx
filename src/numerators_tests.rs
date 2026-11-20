//! q-teardown-stop-latency's test code, the numerators on one vector of limbs: what a
//! `Numerators` reads back, in runs and in chunks, after a clear, its residues through `Radix`
//! against num-bigint's, and the limb helpers of `limbs.rs`
//! (`iterations/0140-q-teardown-stop-latency/plan.md`, _Test code_, `numerators_tests.rs`).

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use rand_core::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::limbs::{big, bits, limb};
use crate::numerators::Numerators;
use crate::rational::{Radix, residue};
use crate::shared_tests::{below, rng};
use crate::text::limbs;

/// The primes the residues are taken modulo: a small one, GF(65521)'s, 2^31 − 1, and 2^61 − 1,
/// the largest of the lift's primes' size, below 2^62.
const PRIMES: [u64; 4] = [101, 65521, 2_147_483_647, 2_305_843_009_213_693_951];

/// The most limbs of a random numerator.
const LIMBS: u64 = 40;

/// The random numerators test 1 pushes beside the fixed ones.
const RANDOM: usize = 200;

/// A random integer of 1 to `LIMBS` limbs, its top limb not 0, and a random sign.
fn numerator(rng: &mut Xoshiro256PlusPlus) -> BigInt {
    let n = 1 + below(rng, LIMBS);
    let mut digits: Vec<u32> = (0..2 * n).map(|_| rng.next_u32()).collect();
    if let Some(top) = digits.last_mut() {
        *top |= 1;
    }
    let sign = if below(rng, 2) == 0 {
        Sign::Plus
    } else {
        Sign::Minus
    };
    BigInt::from_biguint(sign, BigUint::new(digits))
}

/// The numerators of test 1, each with its chain index: 0, ±1, ±(2^64 − 1), ±2^64,
/// ±(2^128 + 1), and `RANDOM` random ones, the indices not falling.
fn pushed() -> Vec<(BigInt, u32)> {
    let one = BigInt::one();
    let word = BigInt::from(u64::MAX);
    let past = BigInt::one() << 64u32;
    let wide = (BigInt::one() << 128u32) + 1u32;
    let mut out: Vec<BigInt> = vec![
        BigInt::zero(),
        one.clone(),
        -one,
        word.clone(),
        -word,
        past.clone(),
        -past,
        wide.clone(),
        -wide,
    ];
    let mut r = rng(0x7ea4);
    out.extend((0..RANDOM).map(|_| numerator(&mut r)));
    let mut den = 0u32;
    out.into_iter()
        .map(|n| {
            if below(&mut r, 4) == 0 {
                den += 1;
            }
            (n, den)
        })
        .collect()
}

/// A store of `numbers`, pushed in order.
fn store(numbers: &[(BigInt, u32)]) -> Numerators {
    let mut s = Numerators::new();
    for (n, den) in numbers {
        s.push(n, *den);
    }
    s
}

/// Whether `s` reads back `numbers` in order: Err with the first that differs.
fn reads_back(s: &Numerators, numbers: &[(BigInt, u32)]) -> Result<(), String> {
    if s.len() != numbers.len() || s.run().len() != numbers.len() {
        return Err(format!(
            "{len} and a run of {run}, not {n}",
            len = s.len(),
            run = s.run().len(),
            n = numbers.len()
        ));
    }
    for (t, ((num, den), (want, want_den))) in s.run().iter().zip(numbers).enumerate() {
        let digits: Vec<u64> = want.magnitude().iter_u64_digits().collect();
        let negative = match want.sign() {
            Sign::Minus => true,
            Sign::NoSign | Sign::Plus => false,
        };
        let got = num.to_bigint();
        if got != *want
            || den != *want_den
            || num.bits() != want.bits()
            || num.limbs() != limbs(want.magnitude())
            || num.is_negative() != negative
            || num.magnitude() != digits.as_slice()
        {
            return Err(format!(
                "term {t}: {got} over {den}, {bits} bits, {limbs} limbs, not {want} over {want_den}",
                bits = num.bits(),
                limbs = num.limbs()
            ));
        }
    }
    let last = s.last().map(|(n, den)| (n.to_bigint(), den));
    if last.as_ref() != numbers.last().map(|(n, den)| (n.clone(), *den)).as_ref() {
        return Err(format!("the last is {last:?}"));
    }
    Ok(())
}

/// Test 1: every numerator pushed reads back, with its sign, bits, limbs, and chain index; a fresh
/// store is empty (q-teardown-stop-latency.1, .3).
#[test]
fn numerators_read_back() {
    let fresh = Numerators::new();
    assert!(fresh.is_empty() && fresh.last().is_none());
    assert_eq!(fresh.len(), 0);
    let numbers = pushed();
    let s = store(&numbers);
    assert!(!s.is_empty());
    if let Err(e) = reads_back(&s, &numbers) {
        panic!("{e}");
    }
}

/// Test 2: the chunks of a run, of every size from 1 to one past its length, hold its numerators
/// in order, every chunk but the last of that size, none empty.
#[test]
fn chunks_cover_the_run() {
    let numbers = pushed();
    let s = store(&numbers);
    let run = s.run();
    for size in 1..=run.len() + 1 {
        let chunks: Vec<_> = run.chunks(size).collect();
        let k = chunks.len();
        for (i, chunk) in chunks.iter().enumerate() {
            let len = chunk.len();
            assert!(len > 0, "size {size}: chunk {i} is empty");
            assert!(
                len == size || (i + 1 == k && len < size),
                "size {size}: chunk {i} of {k} holds {len}"
            );
        }
        let read: Vec<(BigInt, u32)> = chunks
            .iter()
            .flat_map(|c| c.iter().map(|(n, den)| (n.to_bigint(), den)))
            .collect();
        assert_eq!(read, numbers, "size {size}");
    }
}

/// Test 3: a cleared store is empty, and the numerators pushed after the clear read back.
#[test]
fn clear_and_fill_again() {
    let numbers = pushed();
    let mut s = store(&numbers);
    s.clear();
    assert!(s.is_empty() && s.last().is_none());
    assert_eq!((s.len(), s.run().len()), (0, 0));
    for (n, den) in &numbers {
        s.push(n, *den);
    }
    if let Err(e) = reads_back(&s, &numbers) {
        panic!("after a clear: {e}");
    }
}

/// Test 4: `Radix::fits_over` and `Radix::over` of a numerator through its view agree with those
/// through its `BigInt`, and `Radix::residue` of its magnitude with `residue_big` and the free
/// `residue`, at radices of 1 to 8 powers, numerators past the powers among them
/// (q-teardown-stop-latency.7).
#[test]
fn residues_agree() {
    let numbers = pushed();
    let s = store(&numbers);
    let mut r = rng(0x7ea5);
    for &p in &PRIMES {
        for len in 1..=8 {
            let radix = Radix::new(p, len);
            for ((num, _), (want, _)) in s.run().iter().zip(&numbers) {
                let magnitude = want.magnitude();
                let m = radix.residue(num.magnitude());
                assert_eq!(
                    m,
                    radix.residue_big(magnitude),
                    "{want} mod {p}, {len} powers"
                );
                assert_eq!(m, residue(magnitude, p), "{want} mod {p}, by Horner's rule");
                let di = 1 + below(&mut r, p - 1);
                assert_eq!(
                    radix.over(&num, di),
                    radix.over(want, di),
                    "{want}·{di} mod {p}"
                );
                let c = below(&mut r, p);
                let dr = below(&mut r, p);
                for (c, dr) in [(c, dr), (radix.over(want, 1), 1), (c, 0)] {
                    assert_eq!(
                        radix.fits_over(&num, dr, c),
                        radix.fits_over(want, dr, c),
                        "{want} over a residue {dr} at {c} mod {p}"
                    );
                }
            }
        }
    }
}

/// Test 5: `limbs::big` of limbs is the integer they are, `limbs::bits` its bit length, and
/// `limbs::limb` 0 past the top.
#[test]
fn limb_helpers() {
    let mut r = rng(0x7ea6);
    for n in 0..=12 {
        for _ in 0..20 {
            let mut x: Vec<u64> = (0..n).map(|_| r.next_u64()).collect();
            if let Some(top) = x.last_mut() {
                *top |= 1 << below(&mut r, 64);
            }
            let b = big(&x);
            let digits: Vec<u64> = b.iter_u64_digits().collect();
            assert_eq!(digits, x, "{n} limbs");
            assert_eq!(bits(&x), b.bits(), "{b}");
            assert_eq!(limb(&x, n), 0, "past the top of {b}");
            for (i, &l) in x.iter().enumerate() {
                assert_eq!(limb(&x, i), l, "limb {i} of {b}");
            }
        }
    }
}
