//! row-scan-time, the item's test code: random binomial systems, whose
//! rows hold two entries in wide matrices, with signatures off and on in
//! three orders over four fields, the checker on their results with
//! cofactors, and the edge cases of `OBJECTIVES.md`
//! (`iterations/0072-row-scan-time/plan.md`, _Test code_, 7 and 8).

use grobx::Params;
use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted, params_in, run, systems::random_binomials, with_cofactors_in,
};
use crate::many_variables_time::{EDGE, P62};

/// The basis texts of `text` under `params` with signatures off and on:
/// one text. Returns it.
pub(crate) fn agree(case: &str, text: &str, params: &Params) -> String {
    let mut plain = params.clone();
    plain.signatures = false;
    let mut signatures = params.clone();
    signatures.signatures = true;
    let got = run(text, &plain).basis.expect("a basis");
    let want = run(text, &signatures).basis.expect("a basis");
    assert_eq!(got, want, "{case}: signatures off and on\n{text}");
    got
}

/// The checker accepts the result with cofactors in grevlex, signatures
/// off and on, in the exact mode.
fn assert_both_accepted(case: &str, text: &str, seed: u64) {
    for signatures in [false, true] {
        let mut params = with_cofactors_in("grevlex", seed);
        params.signatures = signatures;
        let outcome = run(text, &params);
        assert_accepted(
            &format!("{case}, signatures {signatures}"),
            text,
            &outcome,
            Mode::Exact,
        );
    }
}

/// Test 7, row-scan-time.1 to .10: `random_binomials`, seeds 0 to 5, over
/// 65521, 2^31 − 1, 2^62 − 57, and ℚ, in grevlex, lex, and block:1: the
/// basis texts with signatures off and on are one; with cofactors in
/// grevlex over 65521 and ℚ the checker accepts both results in the exact
/// mode.
#[test]
fn binomial_systems() {
    for p in [P, P31, P62, 0] {
        for seed in 0..6 {
            let text = random_binomials(p, seed);
            for order in ["grevlex", "lex", "block:1"] {
                let case = format!("binomials {seed} over {p} in {order}");
                agree(&case, &text, &params_in(order, seed));
            }
            if p == P || p == 0 {
                assert_both_accepted(&format!("binomials {seed} over {p}"), &text, seed);
            }
        }
    }
}

/// Test 8: the edge cases on a and b over 65521 and ℚ, signatures off and
/// on: one basis text, the bases known before the call as written, and
/// with cofactors the checker accepts both results in the exact mode.
#[test]
fn the_edge_cases() {
    for p in [P, 0] {
        for (case, generators, known) in EDGE {
            let text = format!("a,b\n{p}\n{generators}\n");
            let case = format!("{case} over {p}");
            let got = agree(&case, &text, &params_in("grevlex", 1));
            if let Some(known) = known {
                assert_eq!(got, known, "{case}");
            }
            assert_both_accepted(&case, &text, 1);
        }
    }
}
