//! outgrown-round-size, the item's test code: random and dense systems
//! with signatures off and on over five fields, the checker on their
//! results with cofactors, and the edge cases of `OBJECTIVES.md`
//! (`iterations/0075-outgrown-round-size/plan.md`, _Test code_, 4 and 5).

use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted_in, dense, params_in, random_square, random_system, run,
    with_cofactors_in,
};
use crate::many_variables_time::{EDGE, P62};
use crate::orders::{Family, names};
use crate::row_scan_time::agree;

/// The fields of tests 4 and 5: 65521, 7, whose plain run reduces each row
/// alone, 2^31 − 1, 2^62 − 57, and ℚ, whose learn runs listen.
const FIELDS: [u64; 5] = [P, 7, P31, P62, 0];

/// With cofactors in `order`: the checker accepts the result in the order
/// in the exact mode. Over ℚ in lex the cofactors of a dense cubic system
/// on three variables take seconds (log 0084-2, `q-lex-cofactors`), so
/// test 4 asks this in grevlex alone.
fn assert_accepted_with_cofactors(case: &str, text: &str, order: &str, seed: u64) {
    let outcome = run(text, &with_cofactors_in(order, seed));
    let checker_order = Family::named(order).checker(names(text).len());
    assert_accepted_in(case, text, &outcome, &checker_order, Mode::Exact);
}

/// Test 4, outgrown-round-size.1 to .5: `random_system` in grevlex, and
/// `random_square` and `dense(m, 3, ·)` in grevlex and lex, m from 2 to 3,
/// seeds 0 to 3, over the five fields: the
/// basis texts with signatures off and on are one; over 65521 and ℚ, with
/// cofactors in grevlex, the checker accepts the result in the exact mode.
#[test]
fn systems_by_both_runs() {
    for p in FIELDS {
        for seed in 0..4 {
            let mut systems = vec![
                (
                    "random".to_string(),
                    random_system(p, seed),
                    &["grevlex"][..],
                ),
                (
                    "square".to_string(),
                    random_square(p, seed),
                    &["grevlex", "lex"][..],
                ),
            ];
            for m in 2..=3 {
                systems.push((
                    format!("cubic-{m}"),
                    dense(m, 3, p, seed),
                    &["grevlex", "lex"][..],
                ));
            }
            for (name, text, orders) in systems {
                for &order in orders {
                    let case = format!("{name} {seed} over {p} in {order}");
                    agree(&case, &text, &params_in(order, seed));
                    if (p == P || p == 0) && order == "grevlex" {
                        assert_accepted_with_cofactors(&case, &text, order, seed);
                    }
                }
            }
        }
    }
}

/// Test 5: the edge cases on a and b over the five fields, in grevlex,
/// lex, and `block:1`: one basis text with signatures off and on, the bases
/// known before the call as written; over 65521 and ℚ, with cofactors, the
/// checker accepts the result in its order in the exact mode.
#[test]
fn the_edge_cases() {
    for p in FIELDS {
        for (case, generators, known) in EDGE {
            let text = format!("a,b\n{p}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                let case = format!("{case} over {p} in {order}");
                let got = agree(&case, &text, &params_in(order, 1));
                if let Some(known) = known {
                    assert_eq!(got, known, "{case}");
                }
                if p == P || p == 0 {
                    assert_accepted_with_cofactors(&case, &text, order, 1);
                }
            }
        }
    }
}
