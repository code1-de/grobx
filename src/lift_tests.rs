//! q-lift, sub-item 2: the image of a run, and the lift over Q, with
//! unlucky primes forced first where a case needs one.

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use grobx_checker::{Mode, Order, check};
use num_bigint::BigUint;

use crate::budget::Budget;
use crate::entry::{Imaged, image_in};
use crate::field::Gf31;
use crate::lift::{Image, Report, Sparse, lift_text};
use crate::rational::{Rational, bound, reconstruct};
use crate::systems::{dense, random_square};
use crate::text::Header;
use crate::trace::{NoLearner, NoTrace};
use crate::{Cancel, MonomialOrder, Outcome, Params, Stop, Threads};

/// Two primes near 2^30.5 that the test code makes unlucky.
const U1: u64 = 1_518_500_279;
const U2: u64 = 1_518_500_293;
/// Primes near 2^31, lucky for every instance here.
const L1: u64 = 2_147_483_647;
const L2: u64 = 2_147_483_629;
const L3: u64 = 2_147_483_587;

fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

fn with_cofactors(seed: u64) -> Params {
    let mut p = params(seed);
    p.cofactors = true;
    p
}

/// The outcome of the lift of `text` with the primes `forced` first, which
/// must hold a basis.
fn lift(text: &str, params: &Params, forced: &[u64]) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text(text, params, forced.to_vec(), &mut report)
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
        "{case}: the checker rejects the result: {v:?}\n{basis}\n{cofactors}",
        basis = basis(outcome)
    );
}

// --- images ---

/// A polynomial of a canonical text over GF(p), read: exponent vectors and
/// coefficients, as `Sparse` holds them.
fn read_poly(poly: &str, names: &[&str]) -> Sparse {
    let mut out = Sparse {
        exponents: Vec::new(),
        coefs: Vec::new(),
    };
    if poly == "0" {
        return out;
    }
    for term in poly.split('+') {
        let mut coef = 1u64;
        let mut e = vec![0u32; names.len()];
        for factor in term.split('*') {
            if factor.starts_with(|c: char| c.is_ascii_digit()) {
                coef = factor.parse().unwrap();
                continue;
            }
            let (name, k) = factor.split_once('^').unwrap_or((factor, "1"));
            let i = names.iter().position(|n| *n == name).unwrap();
            e[i] += k.parse::<u32>().unwrap();
        }
        out.exponents.extend(e);
        out.coefs.push(coef);
    }
    out
}

/// The polynomials of a canonical basis text over GF(p).
fn read_basis(text: &str, names: &[&str]) -> Vec<Sparse> {
    let inner = text
        .strip_prefix('[')
        .unwrap()
        .strip_suffix("]:\n")
        .unwrap();
    if inner.is_empty() {
        return Vec::new();
    }
    inner.split(",\n").map(|p| read_poly(p, names)).collect()
}

/// The rows of a cofactor matrix text over GF(p), each its entries.
fn read_cofactors(text: &str, names: &[&str]) -> Vec<Vec<Sparse>> {
    let inner = text
        .strip_prefix('[')
        .unwrap()
        .strip_suffix("]:\n")
        .unwrap();
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(",\n")
        .map(|row| {
            let row = row.strip_prefix('[').unwrap().strip_suffix(']').unwrap();
            row.split(',').map(|e| read_poly(e, names)).collect()
        })
        .collect()
}

/// The image of a run over `text`'s own prime, below 2^31.
fn image(text: &str, params: &Params) -> Image {
    let header = Header::read(text).unwrap();
    let p: u64 = header.characteristic.parse().unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    let got = image_in(
        Gf31::new(p).unwrap(),
        text,
        params,
        None,
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        &mut NoLearner,
        None,
    )
    .unwrap();
    match got {
        Imaged::Image(image) => image,
        Imaged::Lanes(_) | Imaged::Pair(_) => unreachable!("a run over GF(p) makes one image"),
        Imaged::Stopped(o) => panic!("the run stopped: {o:?}"),
    }
}

/// q-lift.29: the image holds what the texts of the same call write, the
/// leading monomials apart from the other terms (result-time.7).
#[test]
fn images_match_the_texts() {
    let random = "x,y,z\n65521\n3*x^2*y-7*z+1,\nx*y*z-2*y^2+5,\nz^3-x+4*y\n";
    let cases = [
        dense(3, 3, 65521, 1),
        random_square(65521, 1),
        random.to_string(),
    ];
    for text in &cases {
        let names: Vec<&str> = text.lines().next().unwrap().split(',').collect();
        for params in [params(3), with_cofactors(3)] {
            let outcome = crate::groebner(text, &params).unwrap();
            let image = image(text, &params);
            let read = read_basis(basis(&outcome), &names);
            let n = names.len();
            assert!(read.iter().all(|g| g.coefs[0] == 1), "monic");
            let leads: Vec<u32> = read
                .iter()
                .flat_map(|g| g.exponents[..n].to_vec())
                .collect();
            let tails: Vec<Sparse> = read
                .iter()
                .map(|g| Sparse {
                    exponents: g.exponents[n..].to_vec(),
                    coefs: g.coefs[1..].to_vec(),
                })
                .collect();
            assert_eq!(image.leads, leads);
            assert_eq!(image.basis, tails);
            assert_eq!(image.stats, outcome.stats);
            match &outcome.cofactors {
                Some(c) => assert_eq!(image.cofactors, read_cofactors(c, &names)),
                None => assert_eq!(image.cofactors.len(), 0, "cofactors untracked"),
            }
        }
    }
}

// --- the lift ---

/// q-lift.21: with a prime of the input's denominators forced first, the
/// lift gives the basis.
#[test]
fn a_prime_of_a_denominator() {
    let text = format!("x,y\n0\nx-1/{U1},\ny^2-x\n");
    let outcome = lift(&text, &params(1), &[U1]);
    assert_eq!(basis(&outcome), format!("[{U1}*x-1,\n{U1}*y^2-1]:\n"));
}

/// q-later-runs-time.8: pauer-1992 §4, example 4. Over U1 the basis is
/// [x], and the replays of its record at lucky primes fit. With U1 and U2
/// forced first, the lift gives [y, x + N].
#[test]
fn an_unlucky_learn_prime_whose_replays_fit() {
    let n = U1 * U2;
    let text = format!("x,y\n0\nx+{n},\nx*y\n");
    let outcome = lift(&text, &params(1), &[U1, U2]);
    assert_eq!(basis(&outcome), format!("[y,\nx+{n}]:\n"));
}

/// q-lift.20: over U1 the first generator's leading monomial is y, and
/// every replay at a lucky prime misfits at its inputs. With U1 forced
/// first, the lift gives the basis.
#[test]
fn an_unlucky_learn_prime_whose_replays_misfit() {
    let text = format!("x,y\n0\n{U1}*x+y,\nx^2-1\n");
    let outcome = lift(&text, &params(1), &[U1]);
    let square = u128::from(U1) * u128::from(U1);
    assert_eq!(basis(&outcome), format!("[{U1}*x+y,\ny^2-{square}]:\n"));
}

/// q-lift.19: the row that makes y reduces to zero over U1. With U1 forced
/// as the first replay prime, the lift gives the basis.
#[test]
fn an_unlucky_replay_prime() {
    let n = U1 * U2;
    let text = format!("x,y\n0\nx+{n},\nx*y\n");
    let outcome = lift(&text, &params(1), &[L1, U1]);
    assert_eq!(basis(&outcome), format!("[y,\nx+{n}]:\n"));
}

/// q-lift.18: over U1 the first element loses its term in y. With U1
/// forced as the first prime combined and as the third, the lift gives the
/// basis.
#[test]
fn a_vanishing_coefficient() {
    let text = format!("x,y\n0\nx-{U1}*y,\ny^2-1\n");
    for forced in [[L1, U1, L2, L3], [L1, L2, L3, U1]] {
        let outcome = lift(&text, &params(1), &forced);
        assert_eq!(
            basis(&outcome),
            format!("[x-{U1}*y,\ny^2-1]:\n"),
            "{forced:?}"
        );
    }
}

/// The primes below `limit`, from 101, by trial division.
fn small_primes(limit: u64) -> Vec<u64> {
    (101..limit)
        .filter(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0))
        .collect()
}

// q-lift.40 design -- the test code's premature reconstruction comes from small forced primes p₁, p₂ and an integer c the test searches for, Wang's reconstruction of −c mod p₁ a rational r ≠ −c with r ≡ −c mod p₂, not no test of a verification that fails on coefficients (not measured)
/// q-lift.40: an integer c and small primes p₁ ≠ p₂, neither dividing c,
/// such that Wang's reconstruction of −c mod p₁ is a rational r ≠ −c with
/// r ≡ −c mod p₂.
fn early_case() -> (u64, u64, u64) {
    let primes = small_primes(1200);
    for c in 1_000_003u64.. {
        for &p1 in &primes {
            if c % p1 == 0 {
                continue;
            }
            let m = BigUint::from(p1);
            let u = BigUint::from(p1 - c % p1);
            let Some(r) = reconstruct(&u, &m, &bound(&m)) else {
                continue;
            };
            if r == Rational::new((-(c as i64)).into(), 1.into()).unwrap() {
                continue;
            }
            for &p2 in &primes {
                if p2 != p1 && c % p2 != 0 && r.residue(p2) == Some(p2 - c % p2) {
                    return (c, p1, p2);
                }
            }
        }
    }
    unreachable!()
}

/// q-lift.40's case: Wang's reconstruction from p₁ fits p₂. With p₁ and
/// p₂ forced as the first primes combined, the lift gives x − c. The
/// chain's MQRR takes no rational from a modulus below 2^11, so the lift
/// keeps nothing early, and result-time.16's drop is not reached.
#[test]
fn an_early_reconstruction() {
    let (c, p1, p2) = early_case();
    let text = format!("x\n0\nx-{c}\n");
    let outcome = lift(&text, &params(1), &[L1, p1, p2]);
    assert_eq!(basis(&outcome), format!("[x-{c}]:\n"), "{p1}, {p2}");
}

/// q-lift.25: 3 divides the lifted basis's denominator. With 3 forced as
/// the verification prime, the lift gives the basis.
#[test]
fn a_verification_prime_of_a_denominator() {
    let outcome = lift("x\n0\n3*x-1\n", &params(1), &[L1, L2, L3, 3]);
    assert_eq!(basis(&outcome), "[3*x-1]:\n");
}

/// The edge cases of `OBJECTIVES.md` over Q, a case with fractions, and a
/// dense cubic system, with cofactors: the checker accepts, and the basis is the
/// one without cofactors.
#[test]
fn cofactors_are_lifted() {
    let cases = [
        ("zero", "x,y\n0\n0\n".to_string()),
        ("zeros", "x,y\n0\n0,\n0\n".to_string()),
        ("none", "x,y\n0\n".to_string()),
        ("unit", "x,y\n0\nx*y-1/2,\nx,\ny+3\n".to_string()),
        ("basis-already", "x,y\n0\n3*x+1,\n2*y^2-5\n".to_string()),
        ("zero-generator", "x,y\n0\nx^2+y,\n0,\nx*y+1\n".to_string()),
        ("eliminable", "x,y\n0\nx-1/3*y^2,\ny^3-2\n".to_string()),
        ("fractions", "x,y\n0\n2*x^2+3*y-1/2,\nx*y+5/3\n".to_string()),
        ("cubic-3", dense(3, 3, 0, 1)),
    ];
    for (case, text) in &cases {
        let outcome = lift(text, &with_cofactors(5), &[]);
        assert_accepted(case, text, &outcome);
        let plain = lift(text, &params(5), &[]);
        assert_eq!(basis(&plain), basis(&outcome), "{case}");
        assert_eq!(plain.cofactors, None, "{case}");
    }
}

/// q-lift.33: the deadline and the cancel flag stop the lift.
#[test]
fn bounds_stop_the_lift() {
    let text = dense(5, 3, 0, 1);
    let mut p = params(1);
    p.deadline = Some(Duration::from_millis(50));
    let start = Instant::now();
    let outcome = lift_text(&text, &p, Vec::new(), &mut Report::default()).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Deadline));
    assert_eq!(outcome.basis, None);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{elapsed:?}",
        elapsed = start.elapsed()
    );

    let mut p = params(1);
    let cancel = Cancel::new();
    cancel.set();
    p.cancel = Some(cancel);
    let outcome = lift_text(&text, &p, Vec::new(), &mut Report::default()).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel));
    assert_eq!(outcome.basis, None);
}

/// q-lift.7, q-lift.28: one seed gives one outcome, and every seed gives
/// the basis.
#[test]
fn the_seed_decides() {
    let text = dense(3, 3, 0, 1);
    let a = lift(&text, &params(7), &[]);
    let b = lift(&text, &params(7), &[]);
    assert_eq!(a, b);
    for seed in 1..=4 {
        let c = lift(&text, &params(seed), &[]);
        assert_eq!(basis(&c), basis(&a), "seed {seed}");
    }
}
