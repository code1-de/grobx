//! gf2-update-time, the item's test code: the edge cases of `OBJECTIVES.md`
//! over GF(2) in three orders, generated boolean systems through the plain
//! run with cofactors, and inputs whose leading monomials divide one another
//! (`iterations/0130-gf2-update-time/plan.md`, _Test code_, 1 to 3).

use grobx_checker::Mode;

use crate::common::systems::boolean;
use crate::common::{ORDERS, assert_accepted_in, params_in, run};
use crate::gf2_reduction_time::EDGE;
use crate::orders::{Family, assert_canonical, names};

/// The parameters of a call in `order` under `seed`, with signatures and the
/// echelon as given, without cofactors.
fn params_with(order: &str, seed: u64, signatures: bool, echelon: bool) -> grobx::Params {
    let mut p = params_in(order, seed);
    p.signatures = signatures;
    p.echelon = echelon;
    p
}

/// The basis of the call of `text` in `order` with cofactors and signatures
/// off, which runs the plain run and tracks its cofactors: canonical in the
/// order and accepted by the checker in the exact mode.
fn plain_with_cofactors(name: &str, text: &str, order: &str, seed: u64) -> String {
    let mut p = params_with(order, seed, false, true);
    p.cofactors = true;
    let outcome = run(text, &p);
    let basis = outcome.basis.clone().unwrap();
    assert_canonical(name, text, &basis, order);
    let checker = Family::named(order).checker(names(text).len());
    assert_accepted_in(name, text, &outcome, &checker, Mode::Exact);
    basis
}

/// Test 1: the edge cases over GF(2) on x, y, z: in grevlex the basis `EDGE`
/// writes, in lex and `block:1` a canonical basis, each with signatures on
/// and off and the echelon on and off; with cofactors and signatures off,
/// the result accepted by the checker in the exact mode with the basis of
/// the call without cofactors.
#[test]
fn edge_cases_over_gf2() {
    for (case, generators, want) in EDGE {
        let text = format!("x,y,z\n2\n{generators}\n");
        for order in ORDERS {
            let name = format!("{case}, {order}");
            let first = run(&text, &params_with(order, 0xb130, true, true))
                .basis
                .unwrap();
            assert_canonical(&name, &text, &first, order);
            if order == "grevlex" {
                assert_eq!(first, want, "{name}");
            }
            for signatures in [true, false] {
                for echelon in [true, false] {
                    let basis = run(&text, &params_with(order, 0xb130, signatures, echelon))
                        .basis
                        .unwrap();
                    assert_eq!(
                        basis, first,
                        "{name}: signatures {signatures}, echelon {echelon}"
                    );
                }
            }
            let with = plain_with_cofactors(&name, &text, order, 0xb130);
            assert_eq!(with, first, "{name}: with cofactors");
        }
    }
}

/// Test 2: boolean systems over GF(2) for n = 6 to 12, m = 2n and
/// m = n + 2: the call with cofactors and signatures off accepted by the
/// checker in the exact mode, its basis the default call's and that of the
/// call with the echelon off.
#[test]
fn boolean_systems_through_the_plain_run() {
    for n in 6..=12usize {
        for m in [2 * n, n + 2] {
            let seed = 0xb140 + (n * 32 + m) as u64;
            let text = boolean(n, m, seed);
            let name = format!("boolean({n}, {m}, {seed:#x})");
            let with = plain_with_cofactors(&name, &text, "grevlex", seed);
            let default = run(&text, &params_with("grevlex", seed, true, true))
                .basis
                .unwrap();
            assert_eq!(default, with, "{name}: the default call");
            let unreduced = run(&text, &params_with("grevlex", seed, true, false))
                .basis
                .unwrap();
            assert_eq!(unreduced, with, "{name}: the echelon off");
        }
    }
}

/// Test 3: inputs whose leading monomials divide one another, so that with
/// the echelon off an input enters divided by one before it: on x, y, z, w
/// the generators x·y·z + y, x·y + z, x·y + x + w, and x + y·w with the field
/// equations, in grevlex, lex, and `block:2`; and `boolean(n, n + 2, seed)`
/// for n = 6 to 9 with x1²·(x2² + x2) plus its first generator added, of
/// leading monomial x1²·x2², which leaves the ideal as it is. The call with
/// cofactors and signatures off, which takes its inputs as read, is
/// accepted by the checker in the exact mode, its basis that of the call
/// with the echelon on.
#[test]
fn divided_inputs() {
    let text = "x,y,z,w\n2\nx*y*z+y,\nx*y+z,\nx*y+x+w,\nx+y*w,\n\
                x^2+x,\ny^2+y,\nz^2+z,\nw^2+w\n";
    for order in ["grevlex", "lex", "block:2"] {
        let name = format!("four generators, {order}");
        let with = plain_with_cofactors(&name, text, order, 0xb150);
        let echelon = run(text, &params_with(order, 0xb150, false, true))
            .basis
            .unwrap();
        assert_eq!(with, echelon, "{name}");
    }
    for n in 6..=9usize {
        let seed = 0xb160 + n as u64;
        let system = boolean(n, n + 2, seed);
        let (head, generators) = system.split_once("\n2\n").unwrap();
        let first = generators.split(',').next().unwrap().trim();
        let text = format!("{head}\n2\nx1^2*x2^2+x1^2*x2+{first},\n{generators}");
        let name = format!(
            "boolean({n}, {m}, {seed:#x}) and a divided input",
            m = n + 2
        );
        let with = plain_with_cofactors(&name, &text, "grevlex", seed);
        let echelon = run(&text, &params_with("grevlex", seed, false, true))
            .basis
            .unwrap();
        assert_eq!(with, echelon, "{name}");
    }
}
