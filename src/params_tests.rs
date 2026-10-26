//! `Params::default` and `random_seed`.

use crate::{MonomialOrder, Params, Threads, groebner};

#[test]
fn default() {
    let mut failures = Vec::new();
    let params = Params::default();
    let expected = Params::new(MonomialOrder::Grevlex, Threads::All, params.seed);
    if format!("{params:?}") != format!("{expected:?}") {
        failures.push(format!("default {params:?}, expected {expected:?}"));
    }
    // A collision of two 64-bit draws would fail this once in 2^64 runs.
    let other = Params::default();
    if other.seed == params.seed {
        failures.push(format!(
            "two defaults share the seed {seed}",
            seed = params.seed
        ));
    }
    let text = "x,y\n0\nx^2+y,\nx*y+1";
    match groebner(text, &params) {
        Ok(outcome) => {
            let basis = outcome.basis.unwrap_or_default();
            if basis != "[y^2-x,\nx*y+1,\nx^2+y]:\n" {
                failures.push(format!("x^2+y, x*y+1 over Q under the defaults: {basis:?}"));
            }
        }
        Err(e) => failures.push(format!("x^2+y, x*y+1 over Q under the defaults: {e}")),
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn random_seed() {
    let draws: Vec<u64> = (0..8).map(|_| super::random_seed()).collect();
    let mut sorted = draws.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), draws.len(), "repeated draws: {draws:?}");
}
