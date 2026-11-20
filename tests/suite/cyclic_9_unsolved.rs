//! cyclic-9-unsolved, the item's integration tests: under the production
//! gate of the large-round test, a call with the signatures on returns the
//! basis of the call with them off, canonical, and the basis the exact
//! checker accepts with cofactors (`iterations/0093-cyclic-9-unsolved/plan.md`,
//! _Test code_, 6 and 7).

use grobx_checker::Mode;

use crate::common::{EDGE_GENERATORS, GF31_PRIMES, random_square, random_system, signatures_in};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// In grevlex over GF(p): the call with the signatures on returns the basis
/// of the call with them off, canonical, and the basis of the call with
/// cofactors, which the exact checker accepts.
fn same(case: &str, instance: &str) {
    let basis = |signatures: bool| {
        grobx::groebner(instance, &signatures_in("grevlex", signatures, 0xc9c9))
            .unwrap_or_else(|e| panic!("{case}, signatures {signatures}: {e:?}\n{instance}"))
            .basis
    };
    let on = basis(true);
    assert!(on.is_some(), "{case}: no basis\n{instance}");
    assert_eq!(on, basis(false), "{case}: signatures off\n{instance}");
    let on = on.unwrap();
    assert_canonical(case, instance, &on, "grevlex");
    let accepted = accepted_in(case, instance, "grevlex", Mode::Exact);
    assert_eq!(
        accepted.basis.as_deref(),
        Some(on.as_str()),
        "{case}: cofactors\n{instance}"
    );
}

/// Test 6: the edge cases of `OBJECTIVES.md` over 7, 65521, and 2^31 − 1 in
/// grevlex, and in lex and `block:1` with cofactors; over ℚ in grevlex.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE_GENERATORS {
        for p in GF31_PRIMES {
            let instance = format!("x,y,z\n{p}\n{generators}\n");
            let name = format!("{case} over {p}");
            same(&name, &instance);
            for order in ["lex", "block:1"] {
                accepted_in(&name, &instance, order, Mode::Exact);
            }
        }
        accepted_q(
            &format!("{case} over ℚ"),
            &format!("x,y,z\n0\n{generators}\n"),
            "grevlex",
        );
    }
}

/// Test 7: cyclic-9-unsolved.1 to .6 under `LARGE`. Random systems and
/// random square systems, seeds 0 to 11, over the three primes in grevlex,
/// and over ℚ.
#[test]
fn random_systems_signatures_on_and_off() {
    for seed in 0..12 {
        for p in GF31_PRIMES {
            same(
                &format!("random system {seed} over {p}"),
                &random_system(p, seed),
            );
            same(
                &format!("random square {seed} over {p}"),
                &random_square(p, seed),
            );
        }
        accepted_q(
            &format!("random system {seed} over ℚ"),
            &random_system(0, seed),
            "grevlex",
        );
    }
}
