//! `check_polled`, which item f4-gfp adds in its sub-item 3 (f4-gfp.26):
//! with a poll that never stops it rules as `check` does, on Accept cases
//! and their mutants in both modes; a poll that answers true stops the
//! check with `Ok(None)` and is not called again; and it is called at least
//! once per generator, per cofactor row, and per pair of elements.

use std::num::NonZeroU32;

use grobx_checker::{Mode, Order, check, check_polled};

use crate::common::{Case, Rng, mutations, quadrics, random_system};

fn cases() -> Vec<Case> {
    let mut out = vec![Case::solved("quadrics-p65521", &quadrics(65521))];
    for seed in 1..=4 {
        out.push(Case::solved(
            &format!("random-p7-{seed}"),
            &random_system(7, seed),
        ));
    }
    out
}

fn modes() -> [Mode; 2] {
    [
        Mode::Exact,
        Mode::Cheap {
            points: NonZeroU32::new(4).unwrap(),
            seed: 5,
        },
    ]
}

#[test]
#[ignore = "calls groebner-jl"]
fn a_poll_that_never_stops_changes_nothing() {
    for case in cases() {
        let mut texts = vec![(case.basis_text(), case.cofactor_text())];
        let mut rng = Rng::new(0x9011);
        for mutation in mutations(&case, &mut rng) {
            texts.push(mutation.apply(&case));
        }
        for (basis, cofactors) in &texts {
            for mode in modes() {
                let want = check(&case.instance, basis, cofactors, &Order::Grevlex, mode);
                let got = check_polled(
                    &case.instance,
                    basis,
                    cofactors,
                    &Order::Grevlex,
                    mode,
                    &mut || false,
                );
                assert_eq!(got, want.map(Some), "{name} in {mode:?}", name = case.name);
            }
        }
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn a_poll_that_answers_true_stops_the_check() {
    for case in cases() {
        let (basis, cofactors) = (case.basis_text(), case.cofactor_text());
        let mut total = 0;
        check_polled(
            &case.instance,
            &basis,
            &cofactors,
            &Order::Grevlex,
            Mode::Exact,
            &mut || {
                total += 1;
                false
            },
        )
        .unwrap();
        assert!(
            total > 0,
            "{name}: the check never polled",
            name = case.name
        );
        for k in [1, 2, 10, total / 2, total] {
            if k == 0 || k > total {
                continue;
            }
            let mut calls = 0;
            let got = check_polled(
                &case.instance,
                &basis,
                &cofactors,
                &Order::Grevlex,
                Mode::Exact,
                &mut || {
                    calls += 1;
                    calls >= k
                },
            );
            assert_eq!(
                got,
                Ok(None),
                "{name}: stopped at poll {k}",
                name = case.name
            );
            assert_eq!(
                calls,
                k,
                "{name}: the poll is not called after it answered true",
                name = case.name
            );
        }
    }
}

/// An Accept case runs every stage: the generators, the cofactor rows, and
/// every pair of elements.
#[test]
#[ignore = "calls groebner-jl"]
fn the_poll_comes_once_per_step_at_least() {
    for case in cases() {
        let (basis, cofactors) = (case.basis_text(), case.cofactor_text());
        let mut total = 0usize;
        let verdict = check_polled(
            &case.instance,
            &basis,
            &cofactors,
            &Order::Grevlex,
            Mode::Exact,
            &mut || {
                total += 1;
                false
            },
        )
        .unwrap()
        .unwrap();
        assert!(
            verdict.groebner.is_ok() && verdict.reduced.is_ok(),
            "{name}",
            name = case.name
        );
        let n = case.basis.len();
        let at_least = case.generators().len() + case.rows.len() + n * n.saturating_sub(1) / 2;
        assert!(
            total >= at_least,
            "{name}: {total} polls for {at_least} steps",
            name = case.name
        );
    }
}
