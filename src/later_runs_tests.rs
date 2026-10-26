//! q-later-runs-time, the lift's test code: `Gf64Pair`'s lanes against
//! `Gf64` at each prime, the pairs that do not fit, the lift in pairs
//! against the lift alone, and an unlucky first prime (plan.md, _Test
//! code_, 1 and 3 to 5). Tests 2 and 6, a pair replay's images and a
//! coefficient 0 at one prime of a pair, gave way to `lift_rest_tests.rs`'
//! tests 1 and 2, as a pair replay hands over one pair image
//! (q-lift-rest-time.1).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::entry::{Imaged, image_in};
use crate::field::{Field, Gf31, Gf64, Gf64Pair, is_prime_64};
use crate::id::MonoId;
use crate::learn_pair_tests::systems;
use crate::lift::{Image, Learns, Reconstruct, Replays, Report, lift_text_as};
use crate::shape::Kept;
use crate::text::Header;
use crate::trace::{NoLearner, NoTrace, Record, Recording};
use crate::{Error, MonomialOrder, Outcome, Params, Threads};

/// `modulus_tests.rs`'s primes and `lift_tests.rs`'s U2: near 2^31, lucky
/// for every instance here; near 2^30.5, which the cases make unlucky; and
/// past 2^61 + 2^60, which the cases make unlucky where they need to.
pub(crate) const L1: u64 = 2_147_483_647;
const U1: u64 = 1_518_500_279;
const U2: u64 = 1_518_500_293;
pub(crate) const W1: u64 = 3_458_764_513_820_540_933;
pub(crate) const W2: u64 = 3_458_764_513_820_541_017;
pub(crate) const W3: u64 = 3_458_764_513_820_541_041;
/// The least prime past 2^61 and the largest below 2^62.
const PAST_61: u64 = 2_305_843_009_213_693_967;
const BELOW_62: u64 = 4_611_686_018_427_387_847;

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

fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// The record of a learn run over `text`, an instance over Q, at the
/// prime `p` below 2^31.
pub(crate) fn learn(text: &str, params: &Params, p: u64) -> Record {
    let header = Header::read(text).unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    let mut recording = Recording::new();
    let got = image_in(
        Gf31::prime(p),
        text,
        params,
        None,
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        &mut recording,
        None,
    )
    .unwrap_or_else(|e| panic!("the learn run at {p} failed: {e:?}\non\n{text}"));
    assert!(
        matches!(got, Imaged::Image(_)),
        "the learn run at {p} stopped"
    );
    recording.finish(named)
}

/// The replay of `record` over `text` in `field`: from or into the shape
/// `kept` holds where `kept` is Some, the full replay where it is None.
pub(crate) fn replay<F: Field>(
    field: F,
    text: &str,
    params: &Params,
    record: &Record,
    kept: Option<&mut Kept>,
) -> Result<Imaged, Error> {
    let header = Header::read(text).unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    image_in(
        field,
        text,
        params,
        Some(record),
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        &mut NoLearner,
        kept,
    )
}

/// The image of a replay alone at the prime `p` of [2^61, 2^62).
pub(crate) fn alone(text: &str, params: &Params, record: &Record, p: u64) -> Image {
    match replay(Gf64::prime(p), text, params, record, None) {
        Ok(Imaged::Image(image)) => image,
        other @ (Ok(Imaged::Lanes(_) | Imaged::Pair(_) | Imaged::Stopped(_)) | Err(_)) => panic!(
            "the replay at {p} returned no image: {what}",
            what = what(&other)
        ),
    }
}

fn what(got: &Result<Imaged, Error>) -> String {
    match got {
        Ok(Imaged::Image(_)) => "an image".to_string(),
        Ok(Imaged::Lanes(images)) => format!("{len} lanes", len = images.len()),
        Ok(Imaged::Pair(_)) => "a pair image".to_string(),
        Ok(Imaged::Stopped(o)) => format!("a stop: {o:?}"),
        Err(e) => format!("{e:?}"),
    }
}

/// The lift of `text` with the primes `forced` first, its replays as
/// `replays` says: its outcome, which must hold a basis.
fn lift(text: &str, params: &Params, forced: &[u64], replays: Replays) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text_as(
        text,
        params,
        forced.to_vec(),
        Reconstruct::Chain,
        replays,
        Learns::Alone,
        &mut NoTrace,
        &mut report,
    )
    .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    outcome
}

/// x mod p, x written in decimal.
fn residue_of(digits: &str, p: u64) -> u64 {
    (BigUint::parse_bytes(digits.as_bytes(), 10).unwrap() % p)
        .to_u64()
        .unwrap()
}

// --- 1: the lanes ---

/// Test 1, q-later-runs-time.1, .3: each lane of `Gf64Pair` computes as
/// `Gf64` at its prime, its dense entries bit for bit as `Gf64`'s, in
/// [0, 2p) between two operations.
#[test]
fn each_lane_is_gf64() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1a4e);
    for (p, q) in [(W1, W2), (PAST_61, BELOW_62), (BELOW_62, PAST_61)] {
        assert!(is_prime_64(p) && is_prime_64(q));
        let f = Gf64Pair::primes(p, q);
        let g = [Gf64::prime(p), Gf64::prime(q)];
        let primes = [p, q];
        let draw =
            |rng: &mut Xoshiro256PlusPlus| -> [u64; 2] { [rng.next_u64() % p, rng.next_u64() % q] };
        for _ in 0..2000 {
            let (a, b) = (draw(&mut rng), draw(&mut rng));
            let v = rng.next_u64();
            assert_eq!(f.coef(v), [g[0].coef(v), g[1].coef(v)]);
            assert_eq!(f.add(a, b), [g[0].add(a[0], b[0]), g[1].add(a[1], b[1])]);
            assert_eq!(f.sub(a, b), [g[0].sub(a[0], b[0]), g[1].sub(a[1], b[1])]);
            assert_eq!(f.neg(a), [g[0].neg(a[0]), g[1].neg(a[1])]);
            assert_eq!(f.mul(a, b), [g[0].mul(a[0], b[0]), g[1].mul(a[1], b[1])]);
            for k in 0..2 {
                assert_eq!(f.lane(a, k), a[k]);
            }
            let both = a[0] != 0 && a[1] != 0;
            assert_eq!(f.is_unit(a), both, "{a:?}");
            match f.inv(a) {
                Some(i) => {
                    assert!(both, "{a:?}");
                    assert_eq!(i, [g[0].inv(a[0]).unwrap(), g[1].inv(a[1]).unwrap()]);
                }
                None => assert!(!both, "{a:?}"),
            }
        }
        for k in 0..2 {
            let mut lane_zero = [1, 1];
            lane_zero[k] = 0;
            assert!(!f.is_unit(lane_zero));
            assert!(f.inv(lane_zero).is_none());
        }
        assert!(!f.is_unit([0, 0]));
        assert_eq!(<Gf64Pair as Field>::LANES, 2);
        assert_eq!(<Gf64 as Field>::LANES, 1);
        for len in [1, 18, 19, 20, 38, 39, 40, 60] {
            let digits: String = (0..len)
                .map(|_| char::from(b'0' + (rng.next_u64() % 10) as u8))
                .collect();
            assert_eq!(
                f.decimal(&digits),
                [residue_of(&digits, p), residue_of(&digits, q)],
                "{digits}"
            );
            assert_eq!(g[0].decimal(&digits), residue_of(&digits, p), "{digits}");
        }
        // Rows into one dense row: the pair's, and each lane's alone.
        let width = 97;
        let mut dense = vec![[0u64; 2]; width];
        let mut single = [vec![0u64; width], vec![0u64; width]];
        let start: Vec<MonoId> = (0..width as u32)
            .filter(|c| c % 3 == 0)
            .map(MonoId::from_raw)
            .collect();
        let start_coefs: Vec<[u64; 2]> = start.iter().map(|_| draw(&mut rng)).collect();
        // SAFETY: every column lies below the width, a coefficient each,
        // and the dense rows are zero where the row loads.
        unsafe {
            f.load(&mut dense, &start, &start_coefs);
            for k in 0..2 {
                let lane: Vec<u64> = start_coefs.iter().map(|c| c[k]).collect();
                g[k].load(&mut single[k], &start, &lane);
            }
        }
        for _ in 0..300 {
            let cols: Vec<MonoId> = (0..width as u32)
                .filter(|_| rng.next_u64() % 4 == 0)
                .map(MonoId::from_raw)
                .collect();
            let coefs: Vec<[u64; 2]> = cols.iter().map(|_| draw(&mut rng)).collect();
            let c = draw(&mut rng);
            // SAFETY: as above.
            unsafe {
                f.sub_mul(&mut dense, &cols, &coefs, c);
                for k in 0..2 {
                    let lane: Vec<u64> = coefs.iter().map(|v| v[k]).collect();
                    g[k].sub_mul(&mut single[k], &cols, &lane, c[k]);
                }
            }
            for col in 0..width {
                for k in 0..2 {
                    assert_eq!(dense[col][k], single[k][col], "column {col}, lane {k}");
                    assert!(dense[col][k] < 2 * primes[k], "column {col}, lane {k}");
                }
            }
        }
        for col in 0..width {
            let mut e = dense[col];
            let got = f.take(&mut e);
            assert_eq!(e, [0, 0]);
            let want = [
                g[0].take(&mut single[0][col]),
                g[1].take(&mut single[1][col]),
            ];
            assert_eq!(got, want, "column {col}");
        }
    }
}

// --- 3: pairs that do not fit ---

/// Test 3, q-later-runs-time.4: a pair does not fit where a generator's
/// leading coefficient or a counted row's leading entry is 0 at one of its
/// primes, nor reads a text with a denominator 0 at one; each prime alone
/// meets what it meets now, W1 alone on the text with a denominator 0
/// left out.
#[test]
fn pairs_that_do_not_fit() {
    let p = params(1);

    // W1·x + y: at W1 the generator's leading monomial is y.
    let text = format!("x,y\n0\n{W1}*x+y,\nx^2-1\n");
    let record = learn(&text, &p, L1);
    let got = replay(Gf64Pair::primes(W1, W2), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::Misfit { step: 0 })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64::prime(W1), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::Misfit { .. })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64::prime(W2), &text, &p, &record, None);
    assert!(
        matches!(got, Ok(Imaged::Image(_))),
        "{what}",
        what = what(&got)
    );

    // x + W1·W2, x·y: the row that makes y reduces to zero at W1.
    let n = u128::from(W1) * u128::from(W2);
    let text = format!("x,y\n0\nx+{n},\nx*y\n");
    let record = learn(&text, &p, L1);
    let got = replay(Gf64Pair::primes(W1, W3), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::Misfit { .. })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64Pair::primes(W3, W1), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::Misfit { .. })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64::prime(W1), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::Misfit { .. })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64::prime(W3), &text, &p, &record, None);
    assert!(
        matches!(got, Ok(Imaged::Image(_))),
        "{what}",
        what = what(&got)
    );

    // x − 1/W1: a denominator 0 at W1.
    let text = format!("x,y\n0\nx-1/{W1},\ny^2-x\n");
    let record = learn(&text, &p, L1);
    let got = replay(Gf64Pair::primes(W1, W2), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::NoInverse { .. })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64Pair::primes(W2, W1), &text, &p, &record, None);
    assert!(
        matches!(got, Err(Error::NoInverse { .. })),
        "{what}",
        what = what(&got)
    );
    let got = replay(Gf64::prime(W2), &text, &p, &record, None);
    assert!(
        matches!(got, Ok(Imaged::Image(_))),
        "{what}",
        what = what(&got)
    );

    // Through the lift, in pairs: the bases of x + W1·W2, x·y and of the
    // text with a denominator 0 at W1.
    let n = u128::from(W1) * u128::from(W2);
    let text = format!("x,y\n0\nx+{n},\nx*y\n");
    let outcome = lift(&text, &p, &[L1, W1, W3], Replays::Pairs);
    assert_eq!(basis(&outcome), format!("[y,\nx+{n}]:\n"));
    let text = format!("x,y\n0\nx-1/{W1},\ny^2-x\n");
    let outcome = lift(&text, &p, &[L1, W1, W2], Replays::Pairs);
    assert_eq!(basis(&outcome), format!("[{W1}*x-1,\n{W1}*y^2-1]:\n"));
}

// --- 4: the lift in pairs against the lift alone ---

/// Test 4, q-later-runs-time.5, .9: the lift in pairs under seeds 1
/// and 2 returns the basis of the lift alone.
#[test]
fn pairs_against_alone() {
    for (name, text) in systems() {
        let alone = lift(&text, &params(1), &[], Replays::Alone);
        for seed in 1..=2 {
            let paired = lift(&text, &params(seed), &[], Replays::Pairs);
            assert_eq!(paired.basis, alone.basis, "{name}, seed {seed}");
        }
    }
}

/// Test 4, the edge cases of `kept_tests.rs`'s
/// `the_edge_cases_through_the_lift`: their bases in both modes.
#[test]
fn the_edge_cases_in_both_modes() {
    let cases = [
        ("the zero ideal", "x,y\n0\n0\n", "[]:\n"),
        ("F = (0), two zeros", "x,y\n0\n0,\n0\n", "[]:\n"),
        ("no generator", "x,y\n0\n", "[]:\n"),
        ("the unit ideal", "x,y\n0\nx*y-1/2,\nx,\ny+3\n", "[1]:\n"),
        ("a constant input", "x,y\n0\nx+y,\n7\n", "[1]:\n"),
        (
            "a Gröbner basis already",
            "x,y\n0\n3*x+1,\n2*y^2-5\n",
            "[3*x+1,\n2*y^2-5]:\n",
        ),
        (
            "a zero generator",
            "x,y\n0\nx^2+y,\n0,\nx*y+1\n",
            "[y^2-x,\nx*y+1,\nx^2+y]:\n",
        ),
        (
            "an eliminable variable",
            "x,y\n0\nx-1/3*y^2,\ny^3-2\n",
            "[y^2-3*x,\n3*x*y-2,\n9*x^2-2*y]:\n",
        ),
    ];
    for (case, text, expected) in cases {
        let outcome = lift(text, &params(1), &[], Replays::Alone);
        assert_eq!(basis(&outcome), expected, "{case}, alone");
        let outcome = lift(text, &params(1), &[], Replays::Pairs);
        assert_eq!(basis(&outcome), expected, "{case}, in pairs");
    }
}

// --- 5: an unlucky first prime ---

/// Test 5, q-later-runs-time.8, pauer-1992 §4, example 4: over U1 the
/// basis is [x]. With U1 and U2 forced first, both lifts return
/// [y, x + N].
#[test]
fn an_unlucky_first_prime() {
    let n = U1 * U2;
    let text = format!("x,y\n0\nx+{n},\nx*y\n");
    for replays in [Replays::Pairs, Replays::Alone] {
        let outcome = lift(&text, &params(1), &[U1, U2], replays);
        assert_eq!(basis(&outcome), format!("[y,\nx+{n}]:\n"), "{replays:?}");
    }
}
