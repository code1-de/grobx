//! q-lift-rest-time, the item's unit tests: a pair replay's image on one
//! list of terms against the replays alone, MQRR's first pass at a large
//! quotient near the threshold against the oracle, an element whose
//! denominators do not nest, and the lift in pairs against the lift alone
//! (`iterations/0088-q-lift-rest-time/plan.md`, _Test code_, 1 to 5).

use std::fmt::Write;

use grobx_checker::Order;
use num_bigint::{BigInt, BigUint};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::attempt_tests::{agrees, small};
use crate::chain_tests::{assert_accepted, basis, lift, params, with_cofactors};
use crate::entry::Imaged;
use crate::field::Gf64Pair;
use crate::later_runs_tests::{L1, W1, W2, W3, alone, learn, replay};
use crate::learn_pair_tests::systems;
use crate::lift::{Image, Learns, PairImage, Reconstruct, Replays, Report, Sparse, lift_text_as};
use crate::mqrr_tests::{continued, primes};
use crate::prime_count_tests::coprime;
use crate::shape::Kept;
use crate::shortcut_tests::{image, product};
use crate::trace::{NoTrace, Record};
use crate::{Outcome, Params};

/// The pair image of a replay of `record` at (p, q).
fn pair_image(
    text: &str,
    params: &Params,
    record: &Record,
    (p, q): (u64, u64),
    kept: Option<&mut Kept>,
) -> PairImage {
    match replay(Gf64Pair::primes(p, q), text, params, record, kept) {
        Ok(Imaged::Pair(image)) => image,
        Ok(_) => panic!("the pair ({p}, {q}) returned no pair image"),
        Err(e) => panic!("the pair ({p}, {q}) failed: {e:?}"),
    }
}

/// The pair image's terms per element, n exponents each, n from the
/// image's leading monomials.
fn variables(image: &PairImage) -> usize {
    image.leads.len() / image.first.len().max(1)
}

/// The image at the pair's prime of lane `k`: per element the terms whose
/// coefficient there is not 0, with those coefficients, as a replay alone
/// at that prime hands them over.
fn lane(image: &PairImage, k: usize) -> Image {
    let n = variables(image);
    let basis = image
        .first
        .iter()
        .zip(&image.second)
        .map(|(first, second)| {
            assert_eq!(
                first.coefs.len(),
                second.len(),
                "a coefficient a term at each prime"
            );
            assert_eq!(
                first.exponents.len(),
                n * first.coefs.len(),
                "n exponents a term"
            );
            let coefs = if k == 0 { &first.coefs } else { second };
            let mut sparse = Sparse {
                exponents: Vec::new(),
                coefs: Vec::new(),
            };
            for (t, &c) in coefs.iter().enumerate() {
                if c != 0 {
                    sparse
                        .exponents
                        .extend_from_slice(&first.exponents[t * n..(t + 1) * n]);
                    sparse.coefs.push(c);
                }
            }
            sparse
        })
        .collect();
    Image {
        leads: image.leads.clone(),
        basis,
        cofactors: Vec::new(),
        stats: image.stats,
    }
}

// --- 1: a pair image's lanes ---

/// Test 1, q-lift-rest-time.1: a pair replay at (W1, W2) returns one pair
/// image, whose lane at each prime, its terms 0 there left out, is the
/// replay alone at that prime: full, into a fresh `Kept`, and from the
/// `Kept` it filled at (W3, W1). Every element holds its terms once, with
/// a coefficient at each prime.
#[test]
fn pair_image_lanes() {
    for (name, text) in systems() {
        let p = params(1);
        let record = learn(&text, &p, L1);
        let image = pair_image(&text, &p, &record, (W1, W2), None);
        assert_eq!(
            lane(&image, 0),
            alone(&text, &p, &record, W1),
            "{name} at W1"
        );
        assert_eq!(
            lane(&image, 1),
            alone(&text, &p, &record, W2),
            "{name} at W2"
        );

        let mut kept = Kept::default();
        let image = pair_image(&text, &p, &record, (W1, W2), Some(&mut kept));
        assert_eq!(
            lane(&image, 0),
            alone(&text, &p, &record, W1),
            "{name} at W1, shaping"
        );
        assert_eq!(
            lane(&image, 1),
            alone(&text, &p, &record, W2),
            "{name} at W2, shaping"
        );
        let image = pair_image(&text, &p, &record, (W3, W1), Some(&mut kept));
        assert_eq!(
            lane(&image, 0),
            alone(&text, &p, &record, W3),
            "{name} at W3, shaped"
        );
        assert_eq!(
            lane(&image, 1),
            alone(&text, &p, &record, W1),
            "{name} at W1, shaped"
        );
    }
}

// --- 2: a coefficient 0 at one prime of a pair ---

/// Test 2, q-lift-rest-time.1, .2: over W1 the first element of
/// `x − W1·y, y^2 − 1` loses its term in y. The pair image at (W1, W2)
/// holds the term with coefficient 0 at W1 and W2 − W1 at W2, and the
/// lifts with W1 forced as the second prime or the fourth, in pairs and
/// alone, return the basis.
#[test]
fn a_zero_kept_at_one_prime() {
    let text = format!("x,y\n0\nx-{W1}*y,\ny^2-1\n");
    let p = params(1);
    let record = learn(&text, &p, L1);
    let image = pair_image(&text, &p, &record, (W1, W2), None);
    assert_eq!(image.first[0].exponents, [0, 1], "{image:?}");
    assert_eq!(image.first[0].coefs, [0], "{image:?}");
    assert_eq!(image.second[0], [W2 - W1], "{image:?}");
    for forced in [[L1, W1, W2, W3], [L1, W2, W3, W1]] {
        for replays in [Replays::Pairs, Replays::Alone] {
            let (outcome, _) = lift_in(&text, &p, &forced, replays);
            assert_eq!(
                basis(&outcome),
                format!("[x-{W1}*y,\ny^2-1]:\n"),
                "{forced:?}, {replays:?}"
            );
        }
    }
}

/// The lift of `text` with its replays as `replays` says and the learn at
/// one prime, the primes `forced` first: its outcome, which must hold a
/// basis, and its report.
fn lift_in(text: &str, params: &Params, forced: &[u64], replays: Replays) -> (Outcome, Report) {
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
    (outcome, report)
}

// --- 3: MQRR's first pass at a large quotient near the threshold ---

/// Test 3, q-lift-rest-time.4: the first quotient of at least 2^t a step
/// the leading parts do not give, past 2^64, at thresholds of 64 to 130:
/// the quotient 2^(t−1), 2^t − 1, 2^t, 2^t + 1, 2^(t+1) − 1, 2^(t+1), and
/// 2^(t+80) after 300 quotients of 1 to 3 and before 300 more, where the
/// remainders' bit lengths differ by t − 1 to t + 81 at that step;
/// `Mqrr::first` gives the oracle's answer, the rational before it or
/// `Nothing`. And rationals of drawn sizes modulo products of 4, 8, and
/// 16 primes of [2^61, 2^62) at the threshold 64 and past it, the large
/// quotient between 2^(t−2) and 2^(t+400).
#[test]
fn first_without_the_division() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1d17);
    let one = BigUint::from(1u32);
    for t in [64u64, 65, 80, 130] {
        let near = [
            &one << (t - 1),
            (&one << t) - 1u32,
            &one << t,
            (&one << t) + 1u32,
            (&one << (t + 1)) - 1u32,
            &one << (t + 1),
            &one << (t + 80),
        ];
        for (i, q) in near.into_iter().enumerate() {
            let mut a = small(&mut rng, 300);
            a.push(q);
            a.extend(small(&mut rng, 300));
            a.push(BigUint::from(2u32));
            let (v, m) = continued(&a);
            agrees(&m, &v, t, &format!("t = {t}, quotient {i}"));
        }
    }
    for k in [4, 8, 16] {
        let ps = primes(&mut rng, k, 61);
        let m = product(&ps);
        for t in [64, 70, 100] {
            for gap in [t - 2, t - 1, t, t + 1, t + 2, t + 80, t + 400] {
                let Some(bits) = m.bits().checked_sub(gap) else {
                    continue;
                };
                if bits < 2 {
                    continue;
                }
                let bd = 1 + bits / 3;
                let d = coprime(&mut rng, bd, &[&m]);
                let n = coprime(&mut rng, bits - bd, &[&d]);
                for n in [BigInt::from(n.clone()), -BigInt::from(n)] {
                    if let Some(v) = image(&n, &d, &m) {
                        agrees(&m, &v, t, &format!("{k} primes, t = {t}, gap {gap}"));
                    }
                }
            }
        }
    }
}

// --- 4: an element whose denominators do not nest ---

/// D·x + a₁·y₁ + … + a₆·y₆ over ℚ for pairwise coprime dᵢ of 40 bits, D
/// their product, and aᵢ = nᵢ·D/dᵢ: the monic element's coefficients are
/// nᵢ/dᵢ, whose denominators do not nest. n₁ holds 900 bits, n₂ to n₅
/// 150, and n₆ `last` bits. The text and its basis text, the generator
/// itself.
fn unnested(rng: &mut Xoshiro256PlusPlus, last: u64) -> (String, String) {
    let mut d: Vec<BigUint> = Vec::new();
    for _ in 0..6 {
        let others: Vec<&BigUint> = d.iter().collect();
        let di = coprime(rng, 40, &others);
        d.push(di);
    }
    let big_d = d.iter().fold(BigUint::from(1u32), |acc, x| acc * x);
    let bits = [900, 150, 150, 150, 150, last];
    let mut terms = format!("{big_d}*x");
    for (i, (di, b)) in d.iter().zip(bits).enumerate() {
        let n = coprime(rng, b, &[di]);
        let a = n * (&big_d / di);
        write!(terms, "+{a}*y{k}", k = i + 1).unwrap();
    }
    let names = "x,y1,y2,y3,y4,y5,y6";
    (format!("{names}\n0\n{terms}\n"), format!("[{terms}]:\n"))
}

/// Test 4, q-lift-rest-time.5 to .9: `unnested` with n₆ of 150 bits,
/// within its part's need, and of 700, past the need and `SLACK`, under
/// seeds 1 to 4: the lift in pairs returns the generator as its basis
/// text. With cofactors, whose replays run alone, the same text, and the
/// exact checker accepts the cofactors.
#[test]
fn unnested_parts() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9a57);
    for last in [150, 700] {
        let (text, want) = unnested(&mut rng, last);
        for seed in 1..=4 {
            let case = format!("n₆ of {last} bits, seed {seed}");
            let (outcome, _) = lift(&text, &params(seed), &[], Reconstruct::Chain);
            assert_eq!(basis(&outcome), want, "{case}");
            let (outcome, _) = lift(&text, &with_cofactors(seed), &[], Reconstruct::Chain);
            assert_eq!(basis(&outcome), want, "{case}, with cofactors");
            assert_accepted(&case, &text, &outcome, &Order::Grevlex);
        }
    }
}

// --- 5: the lift in pairs against the lift alone ---

/// The primes the lift in pairs may combine past the lift alone where the
/// last element kept decides when the lift ends. An attempt once a pair
/// spares no replay, but the guess alternates by the attempt
/// (q-lift-rest-time.3):
/// in pairs it runs where M holds 1, 5, 9, … primes, and the attempt
/// without it where M holds 3, 7, 11, …, against every odd and every even
/// count of primes alone. An element whose guess serves it waits two
/// primes more at most, and one whose guess does not, three. The plan's
/// bound of one prime holds only where every pair attempt tries both,
/// which took more time than the alternation (log 0088-2).
const PAIRED: usize = 3;

/// Test 5, q-lift-rest-time.2, .3: y + n₁/d₁ beside x + n₂/(d₁·d₂), whose
/// guess d₁ serves x's first coefficient, beside x + n₃/d₃, whose guess
/// does not, and `unnested`, under seeds 1 to 8: the lift in pairs and
/// the lift alone return one basis text, and the lift in pairs combines
/// at most `PAIRED` primes more than the lift alone.
#[test]
fn pairs_against_alone() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9e55);
    let d1 = coprime(&mut rng, 200, &[]);
    let n1 = coprime(&mut rng, 200, &[&d1]);
    let d2 = coprime(&mut rng, 200, &[&d1]);
    let dd = &d1 * &d2;
    let n2 = coprime(&mut rng, 400, &[&dd]);
    let d3 = coprime(&mut rng, 31, &[&d1]);
    let n3 = coprime(&mut rng, 600, &[&d3]);
    let texts = [
        format!("x,y\n0\n{dd}*x+{n2},\n{d1}*y+{n1}\n"),
        format!("x,y\n0\n{d3}*x+{n3},\n{d1}*y+{n1}\n"),
        unnested(&mut rng, 150).0,
    ];
    for (i, text) in texts.iter().enumerate() {
        for seed in 1..=8 {
            let p = params(seed);
            let (paired, pairs) = lift_in(text, &p, &[], Replays::Pairs);
            let (single, alone) = lift_in(text, &p, &[], Replays::Alone);
            assert_eq!(basis(&paired), basis(&single), "system {i}, seed {seed}");
            assert!(
                pairs.combined.len() <= alone.combined.len() + PAIRED,
                "system {i}, seed {seed}: {in_pairs} primes in pairs, {alone} alone",
                in_pairs = pairs.combined.len(),
                alone = alone.combined.len()
            );
        }
    }
}
