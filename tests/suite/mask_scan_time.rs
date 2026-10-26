//! mask-scan-time, the item's test code: random systems through the plain
//! run, signatures off and on, in three orders over four fields, the
//! checker on their results with cofactors, and the edge cases of
//! `OBJECTIVES.md` (`iterations/0073-mask-scan-time/plan.md`, _Test code_,
//! 3 and 4).

use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted_in, dense, params_in, random_square, random_system, run,
    systems::random_binomials, with_cofactors_in,
};
use crate::many_variables_time::{EDGE, P62};
use crate::orders::{Family, names};
use crate::row_scan_time::agree;

/// The orders of tests 3 and 4: grevlex, lex, and a block order.
const ORDERS: [&str; 3] = ["grevlex", "lex", "block:1"];

/// With cofactors and signatures off in `order`, so that the plain run
/// computes: the checker accepts the result in the order in the exact
/// mode.
fn assert_plain_accepted(case: &str, text: &str, order: &str, seed: u64) {
    let mut params = with_cofactors_in(order, seed);
    params.signatures = false;
    let outcome = run(text, &params);
    let checker_order = Family::named(order).checker(names(text).len());
    assert_accepted_in(case, text, &outcome, &checker_order, Mode::Exact);
}

/// Test 3, mask-scan-time.1 to .6: `random_system` in grevlex, and
/// `random_square`, `dense(2, 3, ·)`, and `random_binomials` in grevlex,
/// lex, and `block:1`, seeds 0 to 3, over 65521, 2^31 − 1, 2^62 − 57, and
/// ℚ: the basis texts with signatures off and on are one; over 65521 and
/// ℚ, with cofactors and signatures off, the checker accepts the result in
/// its order in the exact mode.
#[test]
fn systems_by_the_plain_run() {
    for p in [P, P31, P62, 0] {
        for seed in 0..4 {
            let systems = [
                ("random", random_system(p, seed), &ORDERS[..1]),
                ("square", random_square(p, seed), &ORDERS[..]),
                ("cubic", dense(2, 3, p, seed), &ORDERS[..]),
                ("binomials", random_binomials(p, seed), &ORDERS[..]),
            ];
            for (name, text, orders) in systems {
                for &order in orders {
                    let case = format!("{name} {seed} over {p} in {order}");
                    agree(&case, &text, &params_in(order, seed));
                    if p == P || p == 0 {
                        assert_plain_accepted(&case, &text, order, seed);
                    }
                }
            }
        }
    }
}

/// Test 4: the edge cases on a and b over 65521, 2^31 − 1, 2^62 − 57, and
/// ℚ, in grevlex, lex, and `block:1`: one basis text with signatures off
/// and on, the bases known before the call as written; over 65521 and ℚ,
/// with cofactors and signatures off, the checker accepts the result in its
/// order in the exact mode.
#[test]
fn the_edge_cases() {
    for p in [P, P31, P62, 0] {
        for (case, generators, known) in EDGE {
            let text = format!("a,b\n{p}\n{generators}\n");
            for order in ORDERS {
                let case = format!("{case} over {p} in {order}");
                let got = agree(&case, &text, &params_in(order, 1));
                if let Some(known) = known {
                    assert_eq!(got, known, "{case}");
                }
                if p == P || p == 0 {
                    assert_plain_accepted(&case, &text, order, 1);
                }
            }
        }
    }
}
