//! gf2-signature-losses, the item's test code: the edge cases of
//! `OBJECTIVES.md` over GF(2) in three orders, generated boolean systems
//! against the calls without signatures and with cofactors, and the first
//! run line of a call by its field, cofactors, signatures, and order
//! (`iterations/0129-gf2-signature-losses/plan.md`, _Test code_, 1 to 3).

use grobx::CheckerMode;
use grobx_checker::Mode;

use crate::common::systems::{boolean, random_sparse};
use crate::common::{ORDERS, P, assert_accepted, params_in, run, with_cofactors_in};
use crate::gf2_reduction_time::EDGE;
use crate::orders::assert_canonical;
use crate::report::{of, report};

/// The parameters of a call in `order` under `seed`, signatures as
/// `signatures` says.
fn params_with(order: &str, seed: u64, signatures: bool) -> grobx::Params {
    let mut p = params_in(order, seed);
    p.signatures = signatures;
    p
}

/// Test 1: the edge cases over GF(2) on x, y, z in grevlex, lex, and
/// `block:1`: the default call's basis canonical, and the basis of the
/// call without signatures.
#[test]
fn edge_cases_by_engine() {
    for (case, generators, _) in EDGE {
        let text = format!("x,y,z\n2\n{generators}\n");
        for order in ORDERS {
            let name = format!("{case}, {order}");
            let basis = run(&text, &params_with(order, 0xa2d0, true)).basis.unwrap();
            assert_canonical(&name, &text, &basis, order);
            let without = run(&text, &params_with(order, 0xa2d0, false))
                .basis
                .unwrap();
            assert_eq!(basis, without, "{name}");
        }
    }
}

/// Test 2: boolean systems over GF(2) for n = 6 to 12, m = 2n and
/// m = n + 2: the default call's basis is that of the call without
/// signatures, and the call with cofactors is accepted by the checker in
/// the exact mode with the same basis.
#[test]
fn boolean_systems_by_engine() {
    for n in 6..=12usize {
        for m in [2 * n, n + 2] {
            let seed = 0xa2e0 + (n * 32 + m) as u64;
            let text = boolean(n, m, seed);
            let name = format!("boolean({n}, {m}, {seed:#x})");
            let basis = run(&text, &params_with("grevlex", seed, true))
                .basis
                .unwrap();
            let without = run(&text, &params_with("grevlex", seed, false))
                .basis
                .unwrap();
            assert_eq!(basis, without, "{name}: without signatures");
            let outcome = run(&text, &with_cofactors_in("grevlex", seed));
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(
                outcome.basis.as_deref(),
                Some(basis.as_str()),
                "{name}: with cofactors"
            );
        }
    }
}

/// The `engine` and `engine-by` of the first run line of the call of
/// `text` under `p`, with its report on.
fn first_run(text: &str, mut p: grobx::Params) -> (String, String) {
    p.report = true;
    let lines = report(&run(text, &p));
    let runs = of(&lines, "run");
    let first = runs.first().expect("a run line");
    (
        first.value("engine").to_string(),
        first.value("engine-by").to_string(),
    )
}

/// Test 3: the first run line names `engine=plain engine-by=field` over
/// GF(2) in grevlex and in lex without cofactors; `engine=signature` over
/// GF(2) with cofactors and with the checker; `engine=plain
/// engine-by=signatures-off` with signatures off; `engine=plain
/// engine-by=order` in a block order; and `engine=signature
/// engine-by=signatures` over GF(65521).
#[test]
fn the_first_run_line() {
    let seed = 0xa2f0;
    let text = boolean(8, 16, seed);
    let pair = |e: &str, by: &str| (e.to_string(), by.to_string());
    for order in ["grevlex", "lex"] {
        assert_eq!(
            first_run(&text, params_with(order, seed, true)),
            pair("plain", "field"),
            "{order}"
        );
    }
    assert_eq!(
        first_run(&text, with_cofactors_in("grevlex", seed)).0,
        "signature",
        "with cofactors"
    );
    let mut checked = params_with("grevlex", seed, true);
    checked.checker = Some(CheckerMode::Exact);
    assert_eq!(first_run(&text, checked).0, "signature", "with the checker");
    assert_eq!(
        first_run(&text, params_with("grevlex", seed, false)),
        pair("plain", "signatures-off"),
        "signatures off"
    );
    assert_eq!(
        first_run(&text, params_with("block:4", seed, true)),
        pair("plain", "order"),
        "a block order"
    );
    assert_eq!(
        first_run(
            &random_sparse(6, P, seed),
            params_with("grevlex", seed, true)
        ),
        pair("signature", "signatures"),
        "over GF(65521)"
    );
}
