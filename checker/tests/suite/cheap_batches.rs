//! checker-points-abort, the checker's test code (plan.md, _Test code_,
//! tests 1 to 4, on `common::quadrics` and random systems in place of the
//! instances the plan names): the cheap mode's points in batches give the result of
//! one batch of k points, over GF(p) and Q, on accepted cases, their
//! mutants, and the edge cases; and a check at 2^32 − 1 points stops at
//! the poll.

use std::num::{NonZeroU32, NonZeroUsize};

use grobx_checker::{
    Flaw, Mode, Order, cheap_moduli, check_cofactors, check_cofactors_batched, check_polled,
};

use crate::agree;
use crate::common::{
    Case, Rng, matrix_text, mutations, plus, quadrics, random_system, random_system_q,
};

/// The batch sizes tests 1 and 2 run, from one point to a batch.
const PER: [usize; 4] = [1, 2, 3, 5];

/// `common::quadrics` over characteristic `p`, solved by the oracle.
fn solved_quadrics(p: u64) -> Case {
    Case::solved(&format!("quadrics-p{p}"), &quadrics(p))
}

/// Thirteen of the edge cases of `agree_edge.rs` (`OBJECTIVES.md`, _Domain
/// specifics the workflow points at_): the zero ideal, F = (0) with two
/// zeros, the unit ideal,
/// input already a Gröbner basis, a zero generator, and an eliminable
/// variable, over GF(p) and Q.
fn edges() -> Vec<Case> {
    [
        ("edge-zero", "x,y\n65521\n0\n"),
        ("edge-q-zero", "x,y\n0\n0\n"),
        ("edge-zeros", "x,y\n7\n0,\n0\n"),
        ("edge-q-zeros", "x,y\n0\n0,\n0\n"),
        ("edge-unit", "x,y\n65521\nx*y-1,\nx\n"),
        ("edge-unit-p2", "x\n2\nx,\nx+1\n"),
        ("edge-q-unit", "x,y\n0\nx*y-1,\nx\n"),
        ("edge-gb", "x,y\n65521\nx+y,\ny^2+1\n"),
        ("edge-q-gb", "x,y\n0\nx+y,\ny^2+1\n"),
        ("edge-zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
        ("edge-q-zero-generator", "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n"),
        (
            "edge-eliminable",
            "x,y,z\n65521\nx-y^2-3*z,\ny^3-z,\nz^2-1\n",
        ),
        ("edge-q-eliminable", "x,y,z\n0\nx-y^2-3*z,\ny^3-z,\nz^2-1\n"),
    ]
    .iter()
    .map(|(name, instance)| Case::solved(name, instance))
    .collect()
}

/// On the case and on every mutant Reject draws for it under `seed`, at
/// k = 1, 7, and the k Agree takes, every batch size of `PER` gives what
/// `check_cofactors` gives with every point in one batch.
fn batches_agree(case: &Case, seed: u64) {
    let mut texts = vec![(
        "Accept".to_string(),
        case.basis_text(),
        case.cofactor_text(),
    )];
    let mut rng = Rng::new(seed);
    for mutation in mutations(case, &mut rng) {
        let (basis, cofactors) = mutation.apply(case);
        texts.push((format!("{mutation:?}"), basis, cofactors));
    }
    let agreed = agree::points(case.prime(), case.degree_bound() + 3).get();
    for (label, basis, cofactors) in &texts {
        for k in [1, 7, agreed] {
            let points = NonZeroU32::new(k).unwrap();
            let mode = Mode::Cheap { points, seed };
            let want = check_cofactors(&case.instance, basis, cofactors, &Order::Grevlex, mode);
            for per in PER {
                let got = check_cofactors_batched(
                    &case.instance,
                    basis,
                    cofactors,
                    &Order::Grevlex,
                    points,
                    seed,
                    NonZeroUsize::new(per).unwrap(),
                );
                assert_eq!(
                    got,
                    want,
                    "{name}, {label}, k = {k}, {per} points to a batch",
                    name = case.name
                );
            }
        }
    }
}

/// Test 1 over GF(p).
#[test]
#[ignore = "calls groebner-jl"]
fn batches_give_the_result_of_one_gfp() {
    batches_agree(&solved_quadrics(65521), 0xba7c);
    for seed in 1..=4 {
        let case = Case::solved(&format!("random-p7-{seed}"), &random_system(7, seed));
        batches_agree(&case, 0xba7c ^ seed);
    }
}

/// Test 1 over Q.
#[test]
#[ignore = "calls groebner-jl"]
fn batches_give_the_result_of_one_q() {
    batches_agree(&solved_quadrics(0), 0xba7d);
    for seed in 1..=4 {
        let case = Case::solved(
            &format!("random-q-1-3-{seed}"),
            &random_system_q(1..=3, seed),
        );
        batches_agree(&case, 0xba7d ^ seed);
    }
}

/// Test 2.
#[test]
#[ignore = "calls groebner-jl"]
fn edge_cases_in_batches() {
    for case in edges() {
        batches_agree(&case, 0xed9e ^ case.name.len() as u64);
    }
}

/// Test 3. Over Q, row 0's first entry of `common::quadrics` gains x/q − x/q for
/// the modulus q of the first, the fourth, or the sixth of six points: at
/// that point a denominator of the row vanishes, and the row is checked
/// exactly, true as it was. A row made false, row 0 or row 2, stays false.
/// Batches of 1, 2, 4, and 6 points give what one batch gives, and that is
/// the exact mode's result.
#[test]
#[ignore = "calls groebner-jl"]
fn a_denominator_that_vanishes_in_a_later_batch() {
    let case = solved_quadrics(0);
    let basis = case.basis_text();
    let k = NonZeroU32::new(6).unwrap();
    let seed = 0x7a15;
    let moduli = cheap_moduli(&case.instance, &basis, &Order::Grevlex, k, seed)
        .expect("the quadrics over Q read");
    assert_eq!(moduli.len(), 6, "a modulus per point");
    let v = &case.variables()[0];
    for j in [0, 3, 5] {
        let q = moduli[j];
        for (label, false_row) in [
            ("true", None),
            ("row 0 false", Some(0)),
            ("row 2 false", Some(2)),
        ] {
            let rows: Vec<String> = case
                .rows
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut entries: Vec<String> =
                        r.split(',').map(|c| c.trim().to_string()).collect();
                    if i == 0 {
                        entries[0] = plus(&entries[0], &format!("1/{q}*{v}-1/{q}*{v}"));
                    }
                    if false_row == Some(i) {
                        entries[1] = plus(&entries[1], "1");
                    }
                    entries.join(",")
                })
                .collect();
            let cofactors = matrix_text(rows.iter().map(String::as_str));
            let mode = Mode::Cheap { points: k, seed };
            let want = check_cofactors(&case.instance, &basis, &cofactors, &Order::Grevlex, mode);
            let exact = check_cofactors(
                &case.instance,
                &basis,
                &cofactors,
                &Order::Grevlex,
                Mode::Exact,
            );
            let expected = match false_row {
                None => Ok(Ok(())),
                Some(i) => Ok(Err(Flaw::Cofactors(i))),
            };
            assert_eq!(exact, expected, "{label}, the modulus of point {j}, exact");
            assert_eq!(
                want, expected,
                "{label}, the modulus of point {j}, one batch"
            );
            for per in [1, 2, 4, 6] {
                let got = check_cofactors_batched(
                    &case.instance,
                    &basis,
                    &cofactors,
                    &Order::Grevlex,
                    k,
                    seed,
                    NonZeroUsize::new(per).unwrap(),
                );
                assert_eq!(
                    got, want,
                    "{label}, the modulus of point {j}, {per} points to a batch"
                );
            }
        }
    }
}

/// The plan's edge case of a row that fails at a point of a later batch,
/// before a row that fails in the first, which the mutants of test 1 need
/// not draw. Over GF(7), F = {x} and G = {1, x}. Row 0, x^5, leaves
/// x^6 − 1, which fails at x = 0 alone. Row 1, 2, leaves x, which fails
/// wherever x ≠ 0. The first point has x ≠ 0, so one batch of it names
/// row 1. Of 64 points one has x = 0, so all 64 name row 0, in batches of
/// 1, 2, 7, 63, and 64 points.
#[test]
fn a_row_that_fails_in_a_later_batch() {
    let (instance, basis) = ("x\n7\nx\n", "[1,\nx]:\n");
    let cofactors = matrix_text(["x^5", "2"].into_iter());
    let seed = 0x1a7e;
    let one = Mode::Cheap {
        points: NonZeroU32::new(1).unwrap(),
        seed,
    };
    assert_eq!(
        check_cofactors(instance, basis, &cofactors, &Order::Grevlex, one),
        Ok(Err(Flaw::Cofactors(1))),
        "the first point has x ≠ 0"
    );
    let k = NonZeroU32::new(64).unwrap();
    let mode = Mode::Cheap { points: k, seed };
    let want = check_cofactors(instance, basis, &cofactors, &Order::Grevlex, mode);
    assert_eq!(want, Ok(Err(Flaw::Cofactors(0))), "a point has x = 0");
    for per in [1, 2, 7, 63, 64] {
        let got = check_cofactors_batched(
            instance,
            basis,
            &cofactors,
            &Order::Grevlex,
            k,
            seed,
            NonZeroUsize::new(per).unwrap(),
        );
        assert_eq!(got, want, "{per} points to a batch");
    }
}

/// Test 4. At 2^32 − 1 points the check stops where the poll answers true
/// at its 20,000th call, and returns.
#[test]
#[ignore = "calls groebner-jl"]
fn many_points_stop_at_the_poll() {
    let mut cases = vec![solved_quadrics(65521), solved_quadrics(0)];
    cases.extend(edges());
    for case in cases {
        let mut calls = 0u32;
        let got = check_polled(
            &case.instance,
            &case.basis_text(),
            &case.cofactor_text(),
            &Order::Grevlex,
            Mode::Cheap {
                points: NonZeroU32::MAX,
                seed: 0x5709,
            },
            &mut || {
                calls += 1;
                calls >= 20_000
            },
        );
        assert_eq!(got, Ok(None), "{name}", name = case.name);
        assert_eq!(
            calls,
            20_000,
            "{name}: the poll is not asked again",
            name = case.name
        );
    }
}
