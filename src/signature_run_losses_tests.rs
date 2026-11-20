//! signature-run-losses, the item's crate tests: with the large-round test
//! lowered to every round and every handover carried, calls return the
//! basis of the call with the signatures off, in grevlex, lex, and
//! `block:1`, over GF(p) and ℚ, and the edge cases of `OBJECTIVES.md`
//! stand (`iterations/0112-signature-run-losses/plan.md`, _Test code_, 5
//! to 7).

use std::num::NonZeroUsize;
use std::time::Duration;

use crate::f4::{CARRY_LEAST, LARGE_LEAST};
use crate::shared_tests::systems;
use crate::{Params, Stop, Threads};

/// The primes of test 5: below the combinations' bound, 65521, 2^31 − 1,
/// the least prime past 2^31, whose field is `Gf64`, and ℚ.
const PRIMES: [u64; 5] = [7, 65521, 2_147_483_647, 2_147_483_659, 0];

/// The seeds of each call.
const SEEDS: [u64; 3] = [1, 2, 3];

/// The edge cases of `OBJECTIVES.md` on x, y, z: name and generators.
const EDGE: [(&str, &str); 7] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// The parameters of a call in `order` under `seed`, the signatures on or
/// off.
fn params(order: &str, seed: u64, signatures: bool) -> Params {
    let mut p = Params::new(
        order.parse().expect("a name of the family"),
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    p.signatures = signatures;
    p
}

/// The basis text of the call, with the large-round test reading every
/// round that made an element and every handover carried.
fn carried(text: &str, params: &Params) -> Option<String> {
    LARGE_LEAST.set(Some(1));
    CARRY_LEAST.set(Some(0));
    let outcome = crate::groebner(text, params);
    LARGE_LEAST.set(None);
    CARRY_LEAST.set(None);
    outcome.unwrap_or_else(|e| panic!("{e:?}\n{text}")).basis
}

/// The basis text of the call with the signatures off.
fn plain(text: &str, order: &str, seed: u64) -> Option<String> {
    crate::groebner(text, &params(order, seed, false))
        .unwrap_or_else(|e| panic!("{e:?}\n{text}"))
        .basis
}

/// Test 5: signature-run-losses.1 to .4 and .7. Over each prime and ℚ,
/// under three seeds, every call carried returns the basis of the call
/// with the signatures off.
#[test]
fn carried_calls_give_the_plain_basis() {
    for p in PRIMES {
        for (i, text) in systems(p, 6, 3).iter().enumerate() {
            for seed in SEEDS {
                assert_eq!(
                    carried(text, &params("grevlex", seed, true)),
                    plain(text, "grevlex", seed),
                    "p {p}, system {i}, seed {seed}\n{text}"
                );
            }
        }
    }
}

/// Test 6: the edge cases over 65521 and ℚ in grevlex, lex, and
/// `block:1`: each call carried returns the basis of the call with the
/// signatures off, and with a deadline of zero stops with `Deadline`.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        for p in [65521, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                assert_eq!(
                    carried(&text, &params(order, 1, true)),
                    plain(&text, order, 1),
                    "{case}, p {p}, {order}"
                );
                let mut bounded = params(order, 1, true);
                bounded.deadline = Some(Duration::ZERO);
                LARGE_LEAST.set(Some(1));
                CARRY_LEAST.set(Some(0));
                let outcome = crate::groebner(&text, &bounded);
                LARGE_LEAST.set(None);
                CARRY_LEAST.set(None);
                let outcome = outcome.unwrap_or_else(|e| panic!("{case}: {e:?}"));
                assert_eq!(
                    outcome.stopped,
                    Some(Stop::Deadline),
                    "{case}, p {p}, {order}"
                );
                assert_eq!(outcome.basis, None, "{case}, p {p}, {order}");
            }
        }
    }
}

/// Test 7: signature-run-losses.1. Lex calls of test 5's systems over
/// 65521, whose grevlex run carries before FGLM: the lex basis of the
/// call with the signatures off.
#[test]
fn carried_lex_calls_give_the_plain_basis() {
    for (i, text) in systems(65521, 6, 3).iter().enumerate() {
        assert_eq!(
            carried(text, &params("lex", 1, true)),
            plain(text, "lex", 1),
            "system {i}\n{text}"
        );
    }
}
