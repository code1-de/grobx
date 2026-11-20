//! A call that asks for the cofactors or the checker in a build without
//! the feature that computes them is refused with `Error::NotBuilt`
//! (`call::built`); a build with the feature returns them. The test runs
//! in every build of the features, so that `cargo test --lib not_built`
//! under `--no-default-features` and under
//! `--no-default-features --features cofactors` checks the refusals.

use crate::{CheckerMode, Error, MonomialOrder, Params, Threads, groebner};

/// One instance over GF(p) and over Q: the refusal comes before the
/// header is read, and the checker runs in `Run::finish` and in the lift.
const TEXTS: [&str; 2] = ["x,y\n65521\nx^2-y,\ny^2-x\n", "x,y\n0\nx^2-y,\ny^2-x\n"];

/// The feature a call refuses for, the cofactors' first, by the features
/// of this build.
fn refused(cofactors: bool, checker: Option<CheckerMode>) -> Option<&'static str> {
    if cofactors && !cfg!(feature = "cofactors") {
        Some("cofactors")
    } else if checker.is_some() && !cfg!(feature = "checker") {
        Some("checker")
    } else {
        None
    }
}

#[test]
fn built_refuses_what_the_build_leaves_out() {
    let mut failures = Vec::new();
    let exact = Some(CheckerMode::Exact);
    for text in TEXTS {
        for (cofactors, checker) in [(false, None), (true, None), (false, exact), (true, exact)] {
            let mut params = Params::new(MonomialOrder::Grevlex, Threads::All, 1);
            params.cofactors = cofactors;
            params.checker = checker;
            let case = format!("{text:?} cofactors={cofactors} checker={checker:?}");
            match (groebner(text, &params), refused(cofactors, checker)) {
                (Err(Error::NotBuilt { feature }), Some(want)) if feature == want => {}
                (Ok(outcome), None) => {
                    if outcome.basis.is_none() {
                        failures.push(format!("{case}: no basis"));
                    }
                    let (got, certified) = (&outcome.cofactors, outcome.certified);
                    if got.is_some() != cofactors {
                        failures.push(format!("{case}: cofactors {got:?}"));
                    }
                    if certified != checker.map(|_| true) {
                        failures.push(format!("{case}: certified {certified:?}"));
                    }
                }
                (got, want) => failures.push(format!("{case}: {got:?}, refused for {want:?}")),
            }
        }
    }
    let e = Error::NotBuilt { feature: "checker" };
    let (name, said) = (e.name(), e.to_string());
    if name != "NotBuilt" {
        failures.push(format!("name {name:?}"));
    }
    let message =
        "checker: not implemented in this build, which leaves out the Cargo feature checker";
    if said != message {
        failures.push(format!("message {said:?}"));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
