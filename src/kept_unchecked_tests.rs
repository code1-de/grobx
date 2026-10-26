//! result-time, the lift's kept elements: the lift's basis where an element
//! is kept with a wrong rational, where a term is gained after the keep,
//! and where a term lacks after it (plan.md, _Test code_, 8 to 11). The
//! cases are built by the searches of `catch_up_tests.rs`, which are on
//! Wang's reconstruction at primes below 1200, where MQRR's threshold
//! takes no rational but 0: the lifts here run Wang's reconstruction alone
//! (iteration 0076's plan.md, _The test code of earlier items_).

use std::num::NonZeroUsize;
use std::time::Duration;

use num_bigint::BigUint;

use crate::catch_up_tests::{
    L1, L2, L3, b, b_open, fits, image, minus_b, small_primes, wang, wrong_keep,
};
use crate::lift::{Reconstruct, Report, lift_text_with};
use crate::{MonomialOrder, Outcome, Params, Threads};

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

/// The lift of Wang's reconstruction alone of `text` with the primes
/// `forced` first: its outcome, which must hold a basis.
fn lift(text: &str, forced: &[u64]) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text_with(
        text,
        &params(1),
        forced.to_vec(),
        Reconstruct::Wang,
        &mut report,
    )
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

/// result-time.13, .16: y's element is kept at p₂ with a wrong rational
/// while x − B's is open, and the one element x + c is kept at p₂ with a
/// wrong rational. The lift returns each basis.
#[test]
fn a_wrong_keep_gives_the_basis() {
    let (c, p1, p2, p3) = wrong_keep();
    let b = b();
    let text = format!("x,y\n0\nx-{b},\ny+{c}\n");
    let outcome = lift(&text, &[L1, p1, p2, p3]);
    assert_eq!(
        basis(&outcome),
        format!("[y+{c},\nx-{b}]:\n"),
        "c = {c}, {p1}, {p2}, {p3}"
    );

    let text = format!("x\n0\nx+{c}\n");
    let outcome = lift(&text, &[L1, p1, p2, p3]);
    assert_eq!(
        basis(&outcome),
        format!("[x+{c}]:\n"),
        "c = {c}, {p1}, {p2}, {p3}"
    );
}

/// result-time.15, .16: N = q₁·q₂ vanishes at q₁ and q₂, so y's element is
/// kept at q₂ without w while x − B's is open. The lift returns the basis
/// with w.
#[test]
fn a_gained_term_after_the_keep_is_not_read() {
    let primes = small_primes(1200);
    // x − B's element is not kept at q₂ (`b_open`).
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
    let b = b();
    let text = format!("x,y,z,w\n0\nx-{b},\ny-z+{n}*w\n");
    let outcome = lift(&text, &[L1, q1, q2, q3]);
    assert_eq!(
        basis(&outcome),
        format!("[y-z+{n}*w,\nx-{b}]:\n"),
        "{q1}, {q2}, {q3}"
    );
}

/// result-time.13 to .15: y's element is kept at L3, and its image lacks z
/// at q, the first prime from 101 that divides no B and at which x − B's
/// element is not kept from its residue modulo L2·L3. The lift returns the
/// basis with z.
#[test]
fn a_lacking_term_after_the_keep_changes_nothing() {
    let b = b();
    let m = BigUint::from(L2) * L3;
    let q = small_primes(1200)
        .into_iter()
        .find(|&q| {
            image(&minus_b(), &BigUint::from(q)) != BigUint::from(0u32)
                && wang(&minus_b(), &m).is_none_or(|s| !fits(&s, &minus_b(), q))
        })
        .unwrap();
    let text = format!("x,y,z\n0\nx-{b},\ny+{q}*z\n");
    let outcome = lift(&text, &[L1, L2, L3, q]);
    assert_eq!(basis(&outcome), format!("[y+{q}*z,\nx-{b}]:\n"), "{q}");
}

/// result-time.16: x + c kept wrongly at p₂ twice, each time from the
/// same primes after a learn run. The lift returns the basis.
#[test]
fn a_wrong_keep_twice_gives_the_basis() {
    let (c, p1, p2, p3) = wrong_keep();
    let text = format!("x\n0\nx+{c}\n");
    let outcome = lift(&text, &[L1, p1, p2, p3, L2, p1, p2, p3]);
    assert_eq!(
        basis(&outcome),
        format!("[x+{c}]:\n"),
        "c = {c}, {p1}, {p2}, {p3}"
    );
}
