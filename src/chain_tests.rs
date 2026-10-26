//! q-rational-time, sub-item 2: the lift over the chain, its rationals
//! over a chain of denominators, returns what the lift of
//! `Wang::reconstruct` alone returns: the same outcome (plan.md, _Test
//! code_). q-prime-count-time's chain lift combines fewer primes, so the
//! two reports differ (iteration 0076's plan.md, _The test code of earlier
//! items_).

use std::fmt::Write;
use std::num::NonZeroUsize;
use std::time::Duration;

use grobx_checker::{Mode, Order, check};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::catch_up_tests::{
    L1, L2, L3, b, b_open, fits, image, minus_b, small_primes, wang, wrong_keep,
};
use crate::lift::{Learns, Reconstruct, Replays, Report, lift_text_as};
use crate::rational::Rational;
use crate::systems::{dense, random_square};
use crate::trace::NoTrace;
use crate::{MonomialOrder, Outcome, Params, Threads};

/// More primes near 2^31, beside `catch_up_tests.rs`'s L1 to L3.
const L4: u64 = 2_147_483_579;
const L5: u64 = 2_147_483_563;
/// A prime near 2^31 that no other case here forces.
const P: u64 = 2_147_483_549;

/// Every lift here in `order` with a deadline of 60 s.
fn params_in(order: MonomialOrder, seed: u64, cofactors: bool) -> Params {
    let mut p = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), seed);
    p.deadline = Some(Duration::from_secs(60));
    p.cofactors = cofactors;
    p
}

pub(crate) fn params(seed: u64) -> Params {
    params_in(MonomialOrder::Grevlex, seed, false)
}

pub(crate) fn with_cofactors(seed: u64) -> Params {
    params_in(MonomialOrder::Grevlex, seed, true)
}

/// The lift of `text` under `how`, with the primes `forced` first: its
/// outcome, which must hold a basis, and its report.
pub(crate) fn lift(
    text: &str,
    params: &Params,
    forced: &[u64],
    how: Reconstruct,
) -> (Outcome, Report) {
    lift_as(text, params, forced, how, Replays::Pairs)
}

/// As `lift`, the replays as `replays` says.
fn lift_as(
    text: &str,
    params: &Params,
    forced: &[u64],
    how: Reconstruct,
    replays: Replays,
) -> (Outcome, Report) {
    let mut report = Report::default();
    let outcome = lift_text_as(
        text,
        params,
        forced.to_vec(),
        how,
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

/// q-rational-time.8: the lift over the chain and the lift of
/// `Wang::reconstruct` alone, with the primes `forced` first, return the
/// same outcome. Under `Reconstruct::Wang`, `euclid` is every coefficient
/// tried, and under `Reconstruct::Chain` at most that. Returns the outcome
/// and both reports, the chain's first.
pub(crate) fn both_reports(
    case: &str,
    text: &str,
    params: &Params,
    forced: &[u64],
) -> (Outcome, Report, Report) {
    both_reports_as(case, text, params, forced, Replays::Pairs)
}

/// As `both_reports`, the replays as `replays` says.
pub(crate) fn both_reports_as(
    case: &str,
    text: &str,
    params: &Params,
    forced: &[u64],
    replays: Replays,
) -> (Outcome, Report, Report) {
    let (outcome, chain) = lift_as(text, params, forced, Reconstruct::Chain, replays);
    let (theirs, wang) = lift_as(text, params, forced, Reconstruct::Wang, replays);
    assert_eq!(outcome, theirs, "{case}: the outcomes differ");
    assert_eq!(
        wang.euclid, wang.reconstructed,
        "{case}: the lift of Wang's reconstruction alone took the scaled residue: {wang:?}"
    );
    assert!(
        chain.euclid <= chain.reconstructed,
        "{case}: more Euclidean runs than coefficients tried: {chain:?}"
    );
    (outcome, chain, wang)
}

/// As `both_reports`: the outcome and the chain's report.
fn both(case: &str, text: &str, params: &Params, forced: &[u64]) -> (Outcome, Report) {
    let (outcome, chain, _) = both_reports(case, text, params, forced);
    (outcome, chain)
}

pub(crate) fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// The checker accepts the outcome's basis and cofactors in the exact mode
/// in `order`.
pub(crate) fn assert_accepted(case: &str, text: &str, outcome: &Outcome, order: &Order) {
    let cofactors = outcome.cofactors.as_deref().expect("cofactors");
    let v = check(text, basis(outcome), cofactors, order, Mode::Exact)
        .unwrap_or_else(|e| panic!("{case}: the checker cannot read the result: {e:?}"));
    assert!(
        v.groebner.is_ok() && v.reduced.is_ok(),
        "{case}: the checker rejects the result: {v:?}\n{basis}\n{cofactors}",
        basis = basis(outcome)
    );
}

/// The generated systems over Q of the tests below: dense cubic ones of 2
/// and 3 variables, whose bases have large rationals, and random square
/// ones.
pub(crate) fn generated() -> Vec<(String, String)> {
    let mut systems = Vec::new();
    for seed in 1..=2 {
        systems.push((format!("dense(2, 3, 0, {seed})"), dense(2, 3, 0, seed)));
        systems.push((format!("dense(3, 3, 0, {seed})"), dense(3, 3, 0, seed)));
        systems.push((format!("random_square(0, {seed})"), random_square(0, seed)));
    }
    systems
}

/// q-rational-time.4 to .6: generated systems through both lifts: the same
/// outcome.
#[test]
fn the_two_lifts_agree_on_generated_systems() {
    for (case, text) in generated() {
        both(&case, &text, &params(7), &[]);
    }
}

/// q-rational-time.4 to .6 on the cofactor parts: each has a chain of its
/// own, and the cofactor text is the same. The checker accepts.
#[test]
fn the_two_lifts_agree_with_cofactors() {
    for (case, text) in generated() {
        let mut p = with_cofactors(7);
        p.checker = Some(crate::CheckerMode::Exact);
        let (outcome, _) = both(&case, &text, &p, &[]);
        assert_eq!(
            outcome.certified,
            Some(true),
            "{case}: {flaw:?}",
            flaw = outcome.flaw
        );
        assert_accepted(&case, &text, &outcome, &Order::Grevlex);
    }
}

/// A random integer of 1 to `bits` bits.
fn integer(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let b = 1 + rng.next_u64() % bits;
    let x = BigUint::from(rng.next_u64()) << 64u32 | BigUint::from(rng.next_u64());
    (x >> (128 - b)).max(BigUint::one())
}

/// A random system over Q in 2 or 3 variables: 2 or 3 generators of
/// degree up to 2, each of 1 to 4 terms whose coefficients are fractions
/// n/d, n and d of 1 to 20 bits, either sign. A generator spells each
/// monomial once.
fn random_system(rng: &mut Xoshiro256PlusPlus) -> String {
    let names = ["x", "y", "z"];
    let n = 2 + (rng.next_u64() % 2) as usize;
    let generators: Vec<String> = (0..2 + rng.next_u64() % 2)
        .map(|_| {
            let mut poly = String::new();
            let mut seen: Vec<Vec<u64>> = Vec::new();
            for _ in 0..=(rng.next_u64() % 4) {
                let mut degree = rng.next_u64() % 3;
                let mut exponents = vec![0u64; n];
                for e in &mut exponents {
                    *e = rng.next_u64() % (degree + 1);
                    degree -= *e;
                }
                if seen.contains(&exponents) {
                    continue;
                }
                let (num, den) = (integer(rng, 20), integer(rng, 20));
                if rng.next_u64() % 2 == 1 {
                    poly.push('-');
                } else if !poly.is_empty() {
                    poly.push('+');
                }
                write!(poly, "{num}/{den}").unwrap();
                for (name, &e) in names.iter().zip(&exponents) {
                    if e > 0 {
                        poly.push('*');
                        poly.push_str(name);
                        if e > 1 {
                            write!(poly, "^{e}").unwrap();
                        }
                    }
                }
                seen.push(exponents);
            }
            poly
        })
        .collect();
    format!(
        "{names}\n0\n{generators}\n",
        names = names[..n].join(","),
        generators = generators.join(",\n")
    )
}

/// q-rational-time.4 to .6 on random systems whose coefficients are
/// fractions, so that elements gain new denominators: equal outcomes, with
/// cofactors and without, in grevlex, and for a quarter of
/// them in lex and in the block order of one variable, with cofactors and
/// the exact checker, which certifies.
#[test]
fn the_two_lifts_agree_on_random_systems() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc4a1);
    for i in 0..40 {
        let text = random_system(&mut rng);
        let case = format!("system {i}:\n{text}");
        both(&case, &text, &params(i), &[]);
        let (outcome, _) = both(&case, &text, &with_cofactors(i), &[]);
        assert_accepted(&case, &text, &outcome, &Order::Grevlex);
        if i % 4 == 0 {
            for order in [MonomialOrder::Lex, MonomialOrder::Block(vec![1])] {
                let mut p = params_in(order.clone(), i, true);
                p.checker = Some(crate::CheckerMode::Exact);
                let (outcome, _) = both(&case, &text, &p, &[]);
                assert_eq!(
                    outcome.certified,
                    Some(true),
                    "{case}\nin {order:?}: {flaw:?}",
                    flaw = outcome.flaw
                );
            }
        }
    }
}

/// q-rational-time.4, .5: the cases of `kept_unchecked_tests.rs`, with
/// the same searches and forced primes: a kept element with a wrong
/// rational beside an open one, the one element kept with a wrong
/// rational, a term a kept element's image gains, and a term it lacks.
/// The searches are on Wang's reconstruction, so the wrong keep's
/// verification is read from the Wang lift's report.
#[test]
fn the_two_lifts_agree_on_the_kept_element_cases() {
    let b = b();
    let (c, p1, p2, p3) = wrong_keep();
    let text = format!("x,y\n0\nx-{b},\ny+{c}\n");
    let (outcome, _) = both("a wrong keep", &text, &params(1), &[L1, p1, p2, p3]);
    assert_eq!(basis(&outcome), format!("[y+{c},\nx-{b}]:\n"));

    let text = format!("x\n0\nx+{c}\n");
    let (outcome, _, report) = both_reports(
        "the one element kept wrongly",
        &text,
        &params(1),
        &[L1, p1, p2, p3],
    );
    assert_eq!(basis(&outcome), format!("[x+{c}]:\n"));
    assert_eq!(report.verified[0], (p3, false), "{report:?}");

    let primes = small_primes(1200);
    let (q1, q2) = primes
        .iter()
        .flat_map(|&q1| primes.iter().map(move |&q2| (q1, q2)))
        .find(|&(q1, q2)| q1 != q2 && b_open(q1, q2))
        .unwrap();
    let q3 = primes
        .iter()
        .copied()
        .find(|&q| q != q1 && q != q2)
        .unwrap();
    let n = q1 * q2;
    let text = format!("x,y,z,w\n0\nx-{b},\ny-z+{n}*w\n");
    let (outcome, _) = both("a gained term", &text, &params(1), &[L1, q1, q2, q3]);
    assert_eq!(basis(&outcome), format!("[y-z+{n}*w,\nx-{b}]:\n"));

    let m = BigUint::from(L2) * L3;
    let q = small_primes(1200)
        .into_iter()
        .find(|&q| {
            image(&minus_b(), &BigUint::from(q)) != BigUint::from(0u32)
                && wang(&minus_b(), &m).is_none_or(|s| !fits(&s, &minus_b(), q))
        })
        .unwrap();
    let text = format!("x,y,z\n0\nx-{b},\ny+{q}*z\n");
    let (outcome, _) = both("a lacking term", &text, &params(1), &[L1, L2, L3, q]);
    assert_eq!(basis(&outcome), format!("[y+{q}*z,\nx-{b}]:\n"));
}

/// An integer c from 1,000,003 and distinct primes p₁ to p₄ from 101 below
/// 2000, none dividing c, such that Wang's reconstruction of c modulo p₁
/// does not fit c at p₂, Wang's reconstruction n/d of c modulo p₁·p₂ fits
/// c at p₃, p₄ divides d, and Wang's reconstruction of B modulo p₁·p₂ does
/// not fit B at p₃. Returns (c, p₁, p₂, p₃, p₄).
fn chain_case() -> (i64, u64, u64, u64, u64) {
    let primes = small_primes(2000);
    let big_b = BigInt::from(b());
    for c in 1_000_003i64.. {
        let a = BigInt::from(c);
        let divides = |p: u64| (c as u64).is_multiple_of(p);
        for &p1 in &primes {
            if divides(p1) {
                continue;
            }
            let first = wang(&a, &BigUint::from(p1));
            for &p2 in &primes {
                if p2 == p1 || divides(p2) || first.as_ref().is_some_and(|r| fits(r, &a, p2)) {
                    continue;
                }
                let m = BigUint::from(p1 * p2);
                let Some(r) = wang(&a, &m) else {
                    continue;
                };
                let Some(p4) = primes.iter().copied().find(|&p4| {
                    p4 != p1 && p4 != p2 && !divides(p4) && (&r.den % p4).to_u64() == Some(0)
                }) else {
                    continue;
                };
                let p3 = primes.iter().copied().find(|&p3| {
                    ![p1, p2, p4].contains(&p3)
                        && !divides(p3)
                        && fits(&r, &a, p3)
                        && wang(&big_b, &m).is_none_or(|s| !fits(&s, &big_b, p3))
                });
                if let Some(p3) = p3 {
                    return (c, p1, p2, p3, p4);
                }
            }
        }
    }
    unreachable!()
}

/// q-rational-time.10: in the lift of Wang's reconstruction alone, the
/// attempt at p₃ keeps n/d for z and fails on w, so the chain of the
/// element's part holds d when p₄ joins M, until the completion's check of
/// n/d fails and clears it. The lift over the chain, whose threshold asks
/// more of M than these primes give, need not pass there, and returns the
/// same outcome
/// (`prime_count_tests::a_chain_denominator_that_a_prime_of_m_divides`
/// takes its own case).
#[test]
fn a_prime_divides_a_chain_denominator() {
    let (c, p1, p2, p3, p4) = chain_case();
    let b = b();
    let text = format!("y,z,w\n0\ny+{c}*z+{b}*w\n");
    let case = format!("c = {c}, {p1}, {p2}, {p3}, {p4}");
    let (outcome, report) = both(&case, &text, &params(1), &[L1, p1, p2, p3, p4]);
    assert_eq!(basis(&outcome), format!("[y+{c}*z+{b}*w]:\n"), "{case}");
    assert_eq!(report.combined[..4], [p1, p2, p3, p4], "{case}: {report:?}");
}

/// Integers c and e and distinct primes p₁ to p₅ from 101 below 2000, none
/// dividing c or e, such that Wang's reconstruction of c modulo p₁ does
/// not fit c at p₂, Wang's reconstruction of c modulo p₁·p₂ is n/p₄ for
/// some n and fits c at p₃ and at p₅, Wang's reconstruction of e modulo
/// p₁·p₂ does not fit e at p₃, and e is at most Wang's bound for p₁·p₂·p₃.
/// c is built as (n + K·p₁·p₂)/p₄ with p₃·p₅ dividing K. Returns (c, e,
/// [p₁, p₂, p₃, p₄, p₅]).
fn attempt_case() -> (u64, u64, [u64; 5]) {
    let primes = small_primes(2000);
    let bound = |m: u64| crate::rational::bound(&BigUint::from(m)).to_u64().unwrap();
    let p1 = primes[0];
    for &p2 in &primes[1..] {
        let m = p1 * p2;
        let others = || primes.iter().copied().filter(move |&p| p != p1 && p != p2);
        let Some(p4) = others().find(|&p| p <= bound(m)) else {
            continue;
        };
        let mut rest = others().filter(|&p| p != p4);
        let (p3, p5) = (rest.next().unwrap(), rest.next().unwrap());
        let ps = [p1, p2, p3, p4, p5];
        for n in (1..=bound(m)).filter(|n| !n.is_multiple_of(p4)) {
            let k = (1..p4)
                .find(|k| (n + p3 * p5 * k * m).is_multiple_of(p4))
                .unwrap();
            let c = (n + p3 * p5 * k * m) / p4;
            let a = BigInt::from(c);
            let wangs = Rational {
                num: BigInt::from(n),
                den: BigUint::from(p4),
            };
            if ps.iter().any(|&p| c.is_multiple_of(p))
                || wang(&a, &BigUint::from(p1)).is_some_and(|r| fits(&r, &a, p2))
                || wang(&a, &BigUint::from(m)) != Some(wangs.clone())
                || !fits(&wangs, &a, p3)
                || !fits(&wangs, &a, p5)
            {
                continue;
            }
            let e = (bound(m) + 1..=bound(m * p3)).find(|&e| {
                let x = BigInt::from(e);
                ps.iter().all(|&p| !e.is_multiple_of(p))
                    && wang(&x, &BigUint::from(m)).is_none_or(|s| !fits(&s, &x, p3))
            });
            if let Some(e) = e {
                return (c, e, ps);
            }
        }
    }
    unreachable!()
}

/// q-rational-time.4, .10: in y + 2t + cz + ew + x, the attempt at p₂ of
/// the lift of Wang's reconstruction alone keeps 2 for t and fails on z,
/// and the attempt at p₃ keeps n/p₄ for z and fails on w. So the chain of
/// the element's part holds p₄ at the attempt at p₄, before the
/// completion's check of n/p₄ fails and clears the element. The lift over
/// the chain, whose threshold asks more of M than these primes give, need
/// not pass there, and returns the same outcome
/// (`prime_count_tests::a_chain_denominator_that_a_prime_of_m_divides`
/// takes its own case).
#[test]
fn a_prime_divides_the_chain_at_the_attempt() {
    let (c, e, ps) = attempt_case();
    let text = format!("y,t,z,w,x\n0\ny+2*t+{c}*z+{e}*w+x\n");
    let case = format!("c = {c}, e = {e}, {ps:?}");
    let mut forced = vec![L1];
    forced.extend(ps);
    let (outcome, report) = both(&case, &text, &params(1), &forced);
    assert_eq!(
        basis(&outcome),
        format!("[y+2*t+{c}*z+{e}*w+x]:\n"),
        "{case}"
    );
    assert_eq!(report.combined[..5], ps, "{case}: {report:?}");
}

/// q-rational-time.4: P divides the last denominator of x − y/P's
/// element, so the verification skips P as it skips it now, and the next
/// prime verifies. −1/P reconstructs by MQRR from two primes near 2^31,
/// at the attempt at L4.
#[test]
fn the_verification_prime_divides_a_denominator() {
    let text = format!("x,y\n0\n{P}*x-y,\ny^2-1\n");
    let (outcome, report) = both(
        "a denominator P",
        &text,
        &params(1),
        &[L1, L2, L3, L4, L5, P],
    );
    assert_eq!(basis(&outcome), format!("[{P}*x-y,\ny^2-1]:\n"));
    assert_eq!(report.combined, [L2, L3, L4], "{report:?}");
    assert!(report.skipped.contains(&P), "{report:?}");
    assert!(
        report.verified.len() == 1 && report.verified[0].1,
        "{report:?}"
    );
}

/// `OBJECTIVES.md`, _Domain specifics the workflow points at_, over Q:
/// the edge cases, and elements of integers, whose chain is [1], and of
/// negative numerators.
pub(crate) const EDGE: [(&str, &str); 9] = [
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
    ("integers alone", "x,y\n0\nx^2-y-7,\ny^2+x-11\n"),
    (
        "negative numerators",
        "x,y\n0\nx^2+5/7*y-3/7,\ny^2-2/9*x+4/9\n",
    ),
];

/// `OBJECTIVES.md`, _Domain specifics the workflow points at_, over Q with cofactors, through
/// both lifts: the same outcome, and the checker accepts each
/// result in the exact mode. Among them, elements of integers, whose chain
/// is [1].
#[test]
fn the_edge_cases_through_both_lifts() {
    for (case, text) in EDGE {
        let (outcome, _) = both(case, text, &with_cofactors(3), &[]);
        assert_accepted(case, text, &outcome, &Order::Grevlex);
    }
}
