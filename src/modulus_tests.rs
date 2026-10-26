//! q-modulus-time, the lift's test code: the replays' primes of
//! [2^61, 2^62) beside the learn runs' of [2^30, 2^31), `Gf64::prime`,
//! the radix's counted sum and the Chinese remainder step at such primes,
//! and the lift whose replays take them, with its unlucky paths at a wide
//! prime (plan.md, _Test code_, 1 to 6 and 8 to 11: test 5 without
//! `steps_from`, which result-time removed, and tests 6 and 11 on generated
//! systems without reference digests).

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::time::Duration;

use grobx_checker::{Mode, Order, check};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::{Field, Gf64, is_prime_64};
use crate::id::MonoId;
use crate::lift::{Primes, Report, lift_text};
use crate::mixed_radix_tests::converted;
use crate::rational::{Crt, Radix};
use crate::systems::{dense, random_square};
use crate::{MonomialOrder, Outcome, Params, Threads};

/// `lift_tests.rs`'s primes near 2^31, lucky for every system here, and
/// the first of its two primes near 2^30.5.
pub(crate) const L1: u64 = 2_147_483_647;
pub(crate) const L2: u64 = 2_147_483_629;
const L3: u64 = 2_147_483_587;
const U1: u64 = 1_518_500_279;
/// Primes past 2^61 + 2^60, which the test code makes unlucky where it
/// needs to.
pub(crate) const W1: u64 = 3_458_764_513_820_540_933;
pub(crate) const W2: u64 = 3_458_764_513_820_541_017;
const W3: u64 = 3_458_764_513_820_541_041;
/// The least prime past 2^31, 2^32, 2^61, and 2^62, and the largest below
/// 2^62.
const PAST_31: u64 = 2_147_483_659;
const PAST_32: u64 = 4_294_967_311;
const PAST_61: u64 = 2_305_843_009_213_693_967;
const BELOW_62: u64 = 4_611_686_018_427_387_847;
const PAST_62: u64 = 4_611_686_018_427_388_039;

const NARROW: std::ops::Range<u64> = 1 << 30..1 << 31;
const WIDE: std::ops::Range<u64> = 1 << 61..1 << 62;

/// Every lift's deadline, far past what each takes: a fault that keeps the
/// lift from completing stops it there and fails the test. A fault that
/// makes the replays misfit ends it in `Error::Drops` before.
const DEADLINE: Duration = Duration::from_secs(60);

fn params(seed: u64) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    p.deadline = Some(DEADLINE);
    p
}

fn with_cofactors(seed: u64) -> Params {
    let mut p = params(seed);
    p.cofactors = true;
    p
}

/// The lift of `text` with the primes `forced` first: its outcome, which
/// must hold a basis, and its report.
fn lift(text: &str, params: &Params, forced: &[u64]) -> (Outcome, Report) {
    let mut report = Report::default();
    let outcome = lift_text(text, params, forced.to_vec(), &mut report)
        .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    (outcome, report)
}

fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// The checker accepts the lift's outcome over Q in grevlex, in the exact
/// mode.
fn assert_accepted(case: &str, text: &str, outcome: &Outcome) {
    let cofactors = outcome.cofactors.as_deref().expect("cofactors");
    let v = check(
        text,
        basis(outcome),
        cofactors,
        &Order::Grevlex,
        Mode::Exact,
    )
    .unwrap_or_else(|e| panic!("{case}: the checker cannot read the result: {e:?}"));
    assert!(
        v.groebner.is_ok() && v.reduced.is_ok(),
        "{case}: the checker rejects the result: {v:?}"
    );
}

/// x on limbs, the least significant first, no zero limb at the top.
fn limbs(x: &BigUint) -> Vec<u64> {
    x.to_u64_digits()
}

/// The integer of limbs `x`.
fn big(x: &[u64]) -> BigUint {
    x.iter()
        .rev()
        .fold(BigUint::zero(), |a, &l| (a << 64u32) + BigUint::from(l))
}

/// A random integer below 2^bits.
fn integer(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let mut x = BigUint::zero();
    for _ in 0..bits.div_ceil(64) {
        x = (x << 64u32) + BigUint::from(rng.next_u64());
    }
    x >> (bits.div_ceil(64) * 64 - bits)
}

// --- the primes ---

/// q-modulus-time.1, .2, q-lift.7: `next` draws the learn runs' primes of
/// [2^30, 2^31), `wide` the replays' of [2^61, 2^62), from one generator
/// and one set of used primes; the seed decides the sequence.
#[test]
fn the_two_draws() {
    let sequence = |seed: u64| -> Vec<u64> {
        let mut primes = Primes::new(seed, Vec::new());
        // A learn prime, then replays, now and then another learn prime.
        (0..300)
            .map(|i| {
                if i % 7 == 0 {
                    let p = primes.next();
                    assert!(NARROW.contains(&p) && is_prime_64(p), "next: {p}");
                    p
                } else {
                    let p = primes.wide();
                    assert!(WIDE.contains(&p) && is_prime_64(p), "wide: {p}");
                    p
                }
            })
            .collect()
    };
    for seed in 1..=4 {
        let drawn = sequence(seed);
        let distinct: BTreeSet<u64> = drawn.iter().copied().collect();
        assert_eq!(distinct.len(), drawn.len(), "seed {seed}: a prime twice");
        assert_eq!(drawn, sequence(seed), "seed {seed}");
    }
    assert_ne!(sequence(1), sequence(2));
}

/// q-lift.36, q-modulus-time.9: the forced primes come first, in order,
/// for either draw, a narrow one through `wide` among them.
#[test]
fn forced_primes_come_first() {
    let mut primes = Primes::new(7, vec![L1, W1, U1, W2]);
    assert_eq!(primes.next(), L1);
    assert_eq!(primes.wide(), W1);
    assert_eq!(primes.wide(), U1);
    assert_eq!(primes.wide(), W2);
    let forced = [L1, W1, U1, W2];
    for _ in 0..50 {
        let p = primes.wide();
        assert!(WIDE.contains(&p) && !forced.contains(&p), "{p}");
        let q = primes.next();
        assert!(NARROW.contains(&q) && !forced.contains(&q), "{q}");
    }
}

/// q-modulus-time.8: `Gf64::prime` is the field `Gf64::new` builds, on
/// both sides of 2^62 and of 2^63.
#[test]
fn the_prime_is_the_tested_field() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6d31);
    for p in [
        PAST_31,
        PAST_61,
        BELOW_62,
        PAST_62,
        9_223_372_036_854_775_783,
        18_446_744_073_709_551_557,
    ] {
        let (a, b) = (Gf64::prime(p), Gf64::new(p).unwrap());
        assert_eq!(a.characteristic(), p);
        assert_eq!(b.characteristic(), p);
        for _ in 0..100 {
            let (x, y) = (rng.next_u64() % p, rng.next_u64() % p);
            assert_eq!(
                a.mul(a.coef(x), a.coef(y)),
                b.mul(b.coef(x), b.coef(y)),
                "GF({p})"
            );
            assert_eq!(a.inv(a.coef(x)), b.inv(b.coef(x)), "GF({p})");
        }
        let cols: Vec<MonoId> = (0..9).map(MonoId::at).collect();
        let coefs: Vec<u64> = cols.iter().map(|_| rng.next_u64() % p).collect();
        let (mut da, mut db) = (vec![0u64; 9], vec![0u64; 9]);
        // SAFETY: the columns are 0 to 8, the dense rows' entries, one
        // coefficient each.
        unsafe {
            a.load(&mut da, &cols, &coefs);
            b.load(&mut db, &cols, &coefs);
            for m in [0, 1, p - 1, rng.next_u64() % p] {
                a.sub_mul(&mut da, &cols, &coefs, a.coef(m));
                b.sub_mul(&mut db, &cols, &coefs, b.coef(m));
            }
        }
        for c in 0..9 {
            assert_eq!(
                a.take(&mut da[c]),
                b.take(&mut db[c]),
                "GF({p}), column {c}"
            );
        }
    }
}

// --- the radix and the Chinese remainder step ---

/// q-modulus-time.6, .7: the counted sum is num-bigint's remainder at
/// primes from 2^31 − 1 to the largest below 2^62, one drawn by `wide`
/// among them, the limbs of 2^64 − 1 among the numbers.
#[test]
fn the_radix_at_every_width() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6d32);
    let drawn = Primes::new(0x6d32, Vec::new()).wide();
    for p in [L1, PAST_32, PAST_61, drawn, BELOW_62] {
        for len in [1usize, 2, 3, 4, 5, 64, 1000, 4096] {
            let radix = Radix::new(p, len);
            assert_eq!(radix.residue(&[]), 0, "0 mod {p}");
            let mut xs = vec![vec![u64::MAX; len], vec![u64::MAX; len.div_ceil(2)]];
            for _ in 0..4 {
                let bits = rng.next_u64() % (64 * len as u64) + 1;
                xs.push(limbs(&integer(&mut rng, bits)));
            }
            for x in &xs {
                let want = (big(x) % BigUint::from(p)).to_u64().unwrap();
                assert_eq!(radix.residue(x), want, "{len} limbs mod {p}", len = x.len());
                assert_eq!(
                    radix.residue_big(&big(x)),
                    want,
                    "{len} limbs mod {p}",
                    len = x.len()
                );
            }
        }
    }
}

/// q-modulus-time.1, q-lift-growth-time.1: the images at narrow and wide
/// primes in turn convert to the residue modulo M·p after each step.
#[test]
fn steps_at_wide_primes() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6d33);
    let used = [L1, W1, W2, L2, W3];
    let xs: Vec<BigUint> = (0..50)
        .map(|i| integer(&mut rng, [1, 64, 200, 320][i % 4]))
        .collect();
    let mut crt = Crt::new();
    let mut images: Vec<Vec<u64>> = vec![Vec::new(); xs.len()];
    for (k, &p) in used.iter().enumerate() {
        let step = crt.step(p, 0);
        let m = crt.modulus().clone();
        for (x, images) in xs.iter().zip(&mut images) {
            images.push((x % BigUint::from(p)).to_u64().unwrap());
        }
        crt.push(step);
        let mp = crt.modulus().clone();
        assert_eq!(mp, &m * BigUint::from(p));
        for (x, images) in xs.iter().zip(&images) {
            let r = converted(&crt, images);
            assert!(r < mp, "step {k}: the residue passes M·p");
            assert_eq!(r, x % &mp, "step {k}");
        }
    }
}

// --- the lift ---

/// q-modulus-time.1, .2: the learn runs and the verifications at primes of
/// [2^30, 2^31), the replays at primes of [2^61, 2^62).
#[test]
fn the_lift_replays_at_wide_primes() {
    for seed in 1..=3 {
        for (name, text) in [
            ("cubic", dense(2, 3, 0, seed)),
            ("square", random_square(0, seed)),
        ] {
            let (_, report) = lift(&text, &params(seed), &[]);
            assert!(
                !report.combined.is_empty(),
                "{name}, seed {seed}: {report:?}"
            );
            for p in &report.learned {
                assert!(NARROW.contains(p), "{name}, seed {seed}: learned {p}");
            }
            for (q, _) in &report.verified {
                assert!(NARROW.contains(q), "{name}, seed {seed}: verified {q}");
            }
            for p in &report.combined {
                assert!(
                    WIDE.contains(p) && is_prime_64(*p),
                    "{name}, seed {seed}: combined {p}"
                );
            }
        }
    }
}

/// q-lift.21 at a wide prime: a replay prime of a denominator is skipped.
#[test]
fn a_wide_prime_of_a_denominator() {
    let text = format!("x,y\n0\nx-1/{W1},\ny^2-x\n");
    let (outcome, report) = lift(&text, &params(1), &[L1, W1]);
    assert_eq!(report.learned, vec![L1], "{report:?}");
    assert!(report.skipped.contains(&W1), "{report:?}");
    assert!(!report.combined.contains(&W1), "{report:?}");
    assert_eq!(basis(&outcome), format!("[{W1}*x-1,\n{W1}*y^2-1]:\n"));
}

/// q-lift.19, .20 at wide primes: the row that makes y reduces to zero
/// mod W1 and mod W2. One misfit is dropped; two in a row drop the record,
/// and the next prime learns at a draw of [2^30, 2^31).
#[test]
fn an_unlucky_wide_prime() {
    let n = u128::from(W1) * u128::from(W2);
    let text = format!("x,y\n0\nx+{n},\nx*y\n");
    let (outcome, report) = lift(&text, &params(1), &[L1, W1]);
    assert_eq!(report.learned, vec![L1], "{report:?}");
    assert_eq!(report.misfit, vec![W1], "{report:?}");
    assert!(!report.combined.contains(&W1), "{report:?}");
    assert_eq!(basis(&outcome), format!("[y,\nx+{n}]:\n"));

    let (outcome, report) = lift(&text, &params(1), &[L1, W1, W2]);
    assert_eq!(report.misfit[..2], [W1, W2], "{report:?}");
    assert_eq!(report.learned.len(), 2, "{report:?}");
    assert_eq!(report.learned[0], L1, "{report:?}");
    assert!(NARROW.contains(&report.learned[1]), "{report:?}");
    assert_eq!(basis(&outcome), format!("[y,\nx+{n}]:\n"));
}

/// q-lift.18 at a wide prime: over W1 the first element loses its term in
/// y, as the first prime combined and as the third.
#[test]
fn a_vanishing_coefficient_at_a_wide_prime() {
    let text = format!("x,y\n0\nx-{W1}*y,\ny^2-1\n");
    for forced in [[L1, W1, W2, W3], [L1, W2, W3, W1]] {
        let (outcome, report) = lift(&text, &params(1), &forced);
        assert!(report.combined.contains(&W1), "{forced:?}: {report:?}");
        assert!(report.misfit.is_empty(), "{forced:?}: {report:?}");
        assert_eq!(
            basis(&outcome),
            format!("[x-{W1}*y,\ny^2-1]:\n"),
            "{forced:?}"
        );
    }
}

/// q-modulus-time.9: forced narrow and wide replay primes in turn, each in
/// its field, combine into one modulus; with cofactors too.
#[test]
fn narrow_and_wide_replays_mix() {
    let forced = [L1, L2, W1, L3, W2];
    for seed in 1..=2 {
        let case = format!("dense(2, 3, 0, {seed})");
        let text = dense(2, 3, 0, seed);
        let (plain, report) = lift(&text, &params(1), &forced);
        assert_eq!(report.learned, vec![L1], "{case}: {report:?}");
        assert!(report.combined.len() >= 4, "{case}: {report:?}");
        assert_eq!(report.combined[..4], [L2, W1, L3, W2], "{case}: {report:?}");
        let (outcome, report) = lift(&text, &with_cofactors(1), &forced);
        assert_eq!(report.combined[..4], [L2, W1, L3, W2], "{case}: {report:?}");
        assert_eq!(basis(&outcome), basis(&plain), "{case}, cofactors");
        assert_accepted(&case, &text, &outcome);
    }
}
