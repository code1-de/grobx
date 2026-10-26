//! q-basis-text-time, the item's unit tests: `push_big` writes every
//! integer as num-bigint writes it, on both sides of the passes' bound,
//! the leaves', and the splits' powers of ten; at the edges of a split's
//! quotient and remainder; and from threads that make the levels at once
//! (plan.md, _Test code_, 1 to 3).

use std::thread;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::shortcut_tests::integer;
use crate::text::{digits_bound, push_big};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// 10^19, a chunk's radix.
fn chunk_radix() -> BigUint {
    BigUint::from(10_000_000_000_000_000_000u64)
}

/// 10^(19m), the radix of m chunks.
fn chunks_radix(m: usize) -> BigUint {
    num_traits::pow(chunk_radix(), m)
}

/// A drawn integer of exactly `limbs` limbs: its top bit at a drawn place
/// of its top limb.
fn of_limbs(rng: &mut Xoshiro256PlusPlus, limbs: u64) -> BigUint {
    let bits = 64 * (limbs - 1) + 1 + rng.next_u64() % 64;
    (BigUint::one() << (bits - 1)) + integer(rng, bits - 1)
}

/// A drawn integer below `bound`, of up to its bits.
fn below(rng: &mut Xoshiro256PlusPlus, bound: &BigUint) -> BigUint {
    integer(rng, bound.bits() + 64) % bound
}

/// `push_big` after a prefix writes the prefix and what num-bigint's
/// `to_str_radix(10)` writes, its digits within `digits_bound` of the bits.
fn assert_written(x: &BigUint) {
    let mut out = b"pre".to_vec();
    push_big(&mut out, x);
    let want = format!("pre{decimal}", decimal = x.to_str_radix(10));
    assert!(
        out == want.as_bytes(),
        "{bits} bits: written {written}, want {want}",
        bits = x.bits(),
        written = String::from_utf8_lossy(&out)
    );
    assert!(
        digits_bound(x.bits()) >= out.len() - 3,
        "{bits} bits, {digits} digits",
        bits = x.bits(),
        digits = out.len() - 3
    );
}

/// Test 1, q-basis-text-time.1, .2, .8, and .9: the integers around the
/// passes' bound of 12 limbs, around 10^(19m) for m on both sides of the
/// leaves' 8 chunks and of the powers of two that splits take, with zero
/// chunks between a quotient and a remainder, and drawn at every limb count
/// to 300 and past it to 2,100.
#[test]
fn integers_as_num_bigint_writes_them() {
    let one = BigUint::one();
    let mut values = vec![
        BigUint::zero(),
        one.clone(),
        chunk_radix() - 1u32,
        chunk_radix(),
    ];
    let limb_counts = (1..=40).chain([
        63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
    ]);
    for n in limb_counts {
        let power = &one << (64 * n);
        values.push(&power - 1u32);
        values.push(&power + 1u32);
        values.push(power);
    }
    let powers_of_two = (7..=11).flat_map(|i| [(1 << i) - 1, 1 << i, (1 << i) + 1]);
    for m in (1..=70).chain(powers_of_two) {
        let power = chunks_radix(m);
        values.push(&power - 1u32);
        values.push(&power + 1u32);
        values.push(power);
    }
    for i in 3..=10 {
        let d_i = chunks_radix(1 << i);
        let cs = [one.clone(), BigUint::from(7u32), chunk_radix() - 1u32];
        let ds = [BigUint::zero(), one.clone(), chunk_radix(), &d_i - 1u32];
        for c in &cs {
            for d in &ds {
                values.push(&d_i * c + d);
            }
        }
    }
    let mut r = rng(0x0b17);
    for limbs in 1..=300 {
        for _ in 0..4 {
            values.push(of_limbs(&mut r, limbs));
        }
    }
    for limbs in (301..=2100).step_by(37) {
        values.push(of_limbs(&mut r, limbs));
    }
    for x in &values {
        assert_written(x);
    }
}

/// Test 2, q-basis-text-time.3 and .4: for D = 10^(19·2^i), the integers
/// of [D, D²) whose splits by D take a quotient or a remainder at its
/// edge, D − 1 or 0, and drawn ones.
#[test]
fn the_splits_at_their_edges() {
    let mut r = rng(0x5b17);
    for i in 3..=9 {
        let d = chunks_radix(1 << i);
        let square = &d * &d;
        let mut values = vec![d.clone(), &d + 1u32, &square - &d, &square - 1u32];
        let mut cs = vec![BigUint::one(), &d - 1u32];
        cs.extend((0..10).map(|_| below(&mut r, &d)));
        for c in &cs {
            values.push(&d * c);
            values.push(&d * c + &d - 1u32);
        }
        for _ in 0..50 {
            values.push(&d + below(&mut r, &(&square - &d)));
        }
        for x in &values {
            assert!(*x >= d && *x < square, "{i}: x in [D, D²)");
            assert_written(x);
        }
    }
}

/// Test 3, q-basis-text-time.6: eight threads at once, each writing thirty
/// drawn integers of 1,500 to 2,100 limbs, write each as num-bigint does,
/// whichever thread makes the levels they take.
#[test]
fn threads_share_the_levels() {
    thread::scope(|scope| {
        for t in 0..8u64 {
            scope.spawn(move || {
                let mut r = rng(0x7e5d + t);
                for _ in 0..30 {
                    let limbs = 1500 + r.next_u64() % 601;
                    assert_written(&of_limbs(&mut r, limbs));
                }
            });
        }
    });
}
