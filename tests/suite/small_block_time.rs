//! small-block-time, the item's test code through the crate: calls over
//! GF(p) and over ℚ under three seeds, the edge cases in three orders over
//! both fields, the checker inside the call, and the memory ceiling
//! (plan.md, _Test code_, 6 to 9).

use grobx::{CheckerMode, Stop};
use grobx_checker::Mode;

use crate::common::{
    EDGE_GENERATORS, FAR_DEADLINE, P, dense, params, params_in, random_square, run,
    with_cofactors_in,
};
use crate::orders::accepted_in;
use crate::q::accepted_q;

/// Test 6, small-block-time.1: calls over GF(65521) and over ℚ return one
/// basis under three seeds.
#[test]
fn three_seeds() {
    for text in [
        dense(4, 3, P, 1),
        random_square(P, 1),
        dense(3, 3, 0, 1),
        random_square(0, 1),
    ] {
        let bases: Vec<_> = (1..=3)
            .map(|seed| {
                let mut p = params(seed);
                p.deadline = Some(FAR_DEADLINE);
                run(&text, &p).basis
            })
            .collect();
        assert_eq!(bases[0], bases[1], "seeds 1 and 2\n{text}");
        assert_eq!(bases[0], bases[2], "seeds 1 and 3\n{text}");
    }
}

/// Test 7, small-block-time.1: the edge cases over GF(65521), over
/// GF(2^31 − 1), and over ℚ, in grevlex, lex, and `block:1`: each basis
/// canonical and accepted by the checker in the exact mode, with
/// cofactors, and the call without cofactors returns the same basis.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE_GENERATORS {
        for characteristic in ["65521", "2147483647", "0"] {
            let text = format!("x,y,z\n{characteristic}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                let checked = if characteristic == "0" {
                    accepted_q(case, &text, order)
                } else {
                    accepted_in(case, &text, order, Mode::Exact)
                };
                let mut p = params_in(order, 1);
                p.deadline = Some(FAR_DEADLINE);
                let plain = run(&text, &p);
                assert_eq!(
                    plain.basis, checked.basis,
                    "{case} over {characteristic} in {order}"
                );
            }
        }
    }
}

/// Test 8, small-block-time.1: dense cubic and random square systems over both
/// fields with cofactors, the checker inside the call in the exact mode:
/// each result certified.
#[test]
fn the_checker_inside_the_call() {
    for text in [
        dense(3, 3, P, 1),
        random_square(P, 2),
        dense(3, 3, 0, 1),
        random_square(0, 2),
    ] {
        let mut p = with_cofactors_in("grevlex", 5);
        p.checker = Some(CheckerMode::Exact);
        p.deadline = Some(FAR_DEADLINE);
        let outcome = run(&text, &p);
        assert_eq!(
            outcome.certified,
            Some(true),
            "{flaw:?}\n{text}",
            flaw = outcome.flaw
        );
    }
}

/// Test 9, small-block-time.6: a memory ceiling of 1 byte stops a long call
/// over GF(65521) and one over ℚ with `Stop::Memory`, and a ceiling of 1 TiB
/// leaves the result of a short call over each as the call without one.
#[test]
fn the_memory_ceiling() {
    for (field, long, short) in [
        ("GF(65521)", dense(8, 3, P, 1), dense(4, 3, P, 1)),
        ("ℚ", dense(5, 3, 0, 1), dense(3, 3, 0, 1)),
    ] {
        let mut tight = params(1);
        tight.memory = Some(1);
        let outcome = grobx::groebner(&long, &tight).unwrap();
        assert!(outcome.basis.is_none(), "{field}: {outcome:?}");
        assert_eq!(outcome.stopped, Some(Stop::Memory), "{field}");
        let mut loose = params(1);
        loose.memory = Some(1 << 40);
        assert_eq!(run(&short, &loose), run(&short, &params(1)), "{field}");
    }
}
