//! q-lift, the item's test code: random systems over Q (q-lift.39) in
//! grevlex, lex, and `block:1`, with cofactors that the checker accepts in
//! the exact mode, and Groebner.jl's basis on the generators with their
//! denominators cleared in lex and `block:1` (q-lift.37).

use std::time::Duration;

use crate::common::groebner_jl_reference;
use crate::orders::{Family, names};
use crate::q::{accepted_q, cleared, finished_q, random_q_system, same_set_q, sorted_q_text};

// q-lift.37 design -- the test code's oracles over Q are the checker in the exact mode with the cofactors, and Groebner.jl's lex and block:1 bases on the integer inputs, compared as sets of polynomials, not the checker alone (not measured)
#[test]
fn random_systems_in_grevlex() {
    for seed in 0..6 {
        let case = format!("q-random-{seed}");
        accepted_q(&case, &random_q_system(seed), "grevlex");
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn random_systems_in_lex_and_a_block_order() {
    let mut finished = 0;
    let mut run = 0;
    for seed in 0..20 {
        let text = random_q_system(seed);
        for order in ["lex", "block:1"] {
            run += 1;
            let case = format!("q-random-{seed}");
            let Some(outcome) = finished_q(&case, &text, order, Duration::from_secs(10)) else {
                continue;
            };
            finished += 1;
            let names = names(&text);
            let family = Family::named(order);
            let theirs = groebner_jl_reference(
                &format!("{case}-cleared"),
                &cleared(&text),
                &["--order", order],
                |basis| sorted_q_text(basis, &names, &family),
            )
            .unwrap_or_else(|| panic!("{case}: Groebner.jl gives no basis in {order}"));
            same_set_q(
                &format!("{case} in {order}"),
                outcome.basis.as_deref().unwrap(),
                &theirs,
                &names,
                &family,
            );
        }
    }
    assert!(10 * finished >= 9 * run, "{finished} of {run} finished");
}
