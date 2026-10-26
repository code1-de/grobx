//! Agree (`OBJECTIVES.md`, _Checker_): on an accepted case and on every
//! mutant of it, the cheap mode reaches the exact mode's verdict. The mutants
//! are those Reject draws with the same seed (`common::accept_and_reject`).


use std::num::NonZeroU32;

use grobx_checker::{Error, Flaw, Mode, Order, Verdict, check, check_cofactors};

use crate::common::{Case, Rng, mutations};

// checker-cheap.16 design -- the test code chooses k per case from p and the case's degree bound (not measured)
/// The number of points that bounds the chance of a false row of degree at
/// most `d` passing them by 2^-64 over GF(p): the least k with
/// (d′/p)^k ≤ 2^-64, d′ = min(d, p − 1) and at least 1. A row of degree p or
/// more is checked exactly (checker-cheap.3). Over Q (p = 0), 4.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "k is the ceiling of a positive bound below 2^32, and f64 holds p and d closely enough for its log"
)]
pub fn points(p: u64, d: u64) -> NonZeroU32 {
    let k = if p == 0 {
        4
    } else {
        let d = d.clamp(1, p - 1) as f64;
        (64.0 / (p as f64 / d).log2()).ceil() as u32
    };
    NonZeroU32::new(k.max(1)).expect("k is at least 1")
}

/// The cheap mode with `points` points and a seed for the variant numbered
/// `n` of the case seeded `seed`.
pub fn cheap(points: NonZeroU32, seed: u64, n: u64) -> Mode {
    Mode::Cheap {
        points,
        seed: seed ^ (n + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    }
}

/// Calls `each` on the case's texts, then on every mutant's, with the label
/// and number of each; a mutant is built just before its call and dropped
/// after it.
fn variants(case: &Case, seed: u64, mut each: impl FnMut(&str, u64, &str, &str)) {
    each("Accept", 0, &case.basis_text(), &case.cofactor_text());
    let mut rng = Rng::new(seed);
    for (n, mutation) in mutations(case, &mut rng).into_iter().enumerate() {
        let (basis, cofactors) = mutation.apply(case);
        each(&format!("{mutation:?}"), n as u64 + 1, &basis, &cofactors);
    }
}

/// The k Agree uses on `case` and its mutants: a mutation raises a degree by
/// at most 3.
fn case_points(case: &Case) -> NonZeroU32 {
    points(case.prime(), case.degree_bound() + 3)
}

/// Whether the rows stage's result `rows` fits the whole check's verdict
/// `full` in the same mode (`check_cofactors`): where the generators reduce
/// to zero, the rows stage reports the flaw `check` reports for the rows, and
/// nothing where `check` found none in them.
pub fn consistent(full: &Verdict, rows: &Result<(), Flaw>) -> bool {
    match &full.groebner {
        Err(Flaw::Generator(_)) => true,
        Ok(()) | Err(Flaw::SPair(..)) => rows.is_ok(),
        Err(flaw) => rows.as_ref().err() == Some(flaw),
    }
}

fn refused<T>(what: &str, result: Result<T, Error>) -> Result<T, String> {
    result.map_err(|e| format!("{what} refused a well-formed result: {e:?}"))
}

/// The whole check and the rows stage alone, on the case and every mutant:
/// in each, the cheap mode's verdict is the exact mode's, and in each mode
/// the two are consistent. The case's verdict holds in both.
pub fn agree_full(case: &Case, seed: u64) {
    let k = case_points(case);
    let mut failures = Vec::new();
    variants(case, seed, |label, n, basis, cofactors| {
        let mode = cheap(k, seed, n);
        let run = |mode: Mode| -> Result<(Verdict, Result<(), Flaw>), String> {
            let full = refused(
                "check",
                check(&case.instance, basis, cofactors, &Order::Grevlex, mode),
            )?;
            let rows = refused(
                "check_cofactors",
                check_cofactors(&case.instance, basis, cofactors, &Order::Grevlex, mode),
            )?;
            Ok((full, rows))
        };
        let (exact, got) = match (run(Mode::Exact), run(mode)) {
            (Ok(exact), Ok(got)) => (exact, got),
            (exact, got) => {
                failures.push(format!("{label}: exact {exact:?}, cheap {got:?} ({mode:?})"));
                return;
            }
        };
        if n == 0 && (exact.0.groebner.is_err() || exact.0.reduced.is_err()) {
            failures.push(format!(
                "{label}: the exact mode says {verdict:?}",
                verdict = exact.0
            ));
        }
        if got != exact {
            failures.push(format!("{label}: exact {exact:?}, cheap {got:?} ({mode:?})"));
        }
        for (name, (full, rows)) in [("exact", &exact), ("cheap", &got)] {
            if !consistent(full, rows) {
                failures.push(format!(
                    "{label}: in the {name} mode check says {full:?} and check_cofactors {rows:?}"
                ));
            }
        }
    });
    assert!(
        failures.is_empty(),
        "{name} (k = {k}):\n{failures}",
        name = case.name,
        failures = failures.join("\n")
    );
}
