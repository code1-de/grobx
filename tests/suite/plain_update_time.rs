//! plain-update-time, the item's test code: calls whose plain run updates
//! its pair set from the start, against the same calls with signatures,
//! the checker on their results, and the edge cases of `OBJECTIVES.md`
//! (`iterations/0071-plain-update-time/plan.md`, _Test code_, 5 and 6).

use grobx::Params;
use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted, params_in, random_square, random_system, run, with_cofactors_in,
};
use crate::many_variables_time::{EDGE, P62};

/// The basis texts of `text` under `params` with signatures off, the plain
/// run from the start, and on: one text. Returns it.
fn agree(case: &str, text: &str, params: &Params) -> String {
    let mut plain = params.clone();
    plain.signatures = false;
    let mut signatures = params.clone();
    signatures.signatures = true;
    let got = run(text, &plain).basis.expect("a basis");
    let want = run(text, &signatures).basis.expect("a basis");
    assert_eq!(got, want, "{case}: signatures off and on\n{text}");
    got
}

/// The checker accepts the plain run's result with cofactors in grevlex,
/// in the exact mode.
fn assert_plain_accepted(case: &str, text: &str, seed: u64) {
    let mut params = with_cofactors_in("grevlex", seed);
    params.signatures = false;
    let outcome = run(text, &params);
    assert_accepted(case, text, &outcome, Mode::Exact);
}

/// Test 5, plain-update-time.1 to .6: random and random square systems,
/// seeds 0 to 3, over 65521, 2^31 − 1, 2^62 − 57, and ℚ, in grevlex, lex,
/// and block:1: the basis with signatures off is the one with signatures
/// on, the signature run's in grevlex and before FGLM in lex, and the plain
/// run's in block:1 and where F4 runs in lex; with cofactors over 65521 and
/// ℚ the checker accepts the plain run's result.
#[test]
fn plain_and_signature_runs_agree() {
    for p in [P, P31, P62, 0] {
        for seed in 0..4 {
            let systems = [
                (format!("random system {seed}"), random_system(p, seed)),
                (format!("random square {seed}"), random_square(p, seed)),
            ];
            for (case, text) in &systems {
                for order in ["grevlex", "lex", "block:1"] {
                    let case = format!("{case} over {p} in {order}");
                    agree(&case, text, &params_in(order, seed));
                }
                if p == P || p == 0 {
                    assert_plain_accepted(case, text, seed);
                }
            }
        }
    }
}

/// Test 6: the edge cases on a and b over 65521 and ℚ, signatures off and
/// on: one basis text, the bases known before the call as written, and
/// the checker accepts the plain run's result.
#[test]
fn the_edge_cases_without_signatures() {
    for p in [P, 0] {
        for (case, generators, known) in EDGE {
            let text = format!("a,b\n{p}\n{generators}\n");
            let case = format!("{case} over {p}");
            let got = agree(&case, &text, &params_in("grevlex", 1));
            if let Some(known) = known {
                assert_eq!(got, known, "{case}");
            }
            assert_plain_accepted(&case, &text, 1);
        }
    }
}
