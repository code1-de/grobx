//! fglm-lex, the item's test code: a lex record learned over one prime and
//! replayed in lex over another gives the lex call's basis there, for a
//! zero-dimensional ideal and for one that is not. With cofactors the
//! checker accepts the replay's result. A lex record replayed in another
//! order does not fit (fglm-lex.15, plan.md, _Edge cases_).

use grobx::Error;
use grobx::trace::{NoTrace, Record};
use grobx_checker::Mode;

use crate::common::{
    P31, assert_accepted_in, dense, params_in, random_square, run, with_cofactors_in,
};
use crate::orders::{Family, names};

/// The instance with its characteristic, the second line, replaced by p.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let variables = lines.next().unwrap();
    lines.next();
    let rest = lines.next().unwrap_or("");
    format!("{variables}\n{p}\n{rest}")
}

/// The record of a learn run in `order` that returns a basis, the
/// generators in reduced row echelon form where `echelon`.
fn learn(name: &str, text: &str, order: &str, echelon: bool) -> Record {
    let mut params = params_in(order, 1);
    params.echelon = echelon;
    let (outcome, record) = grobx::learn(text, &params, &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: learn failed in {order}: {e:?}"));
    assert!(outcome.basis.is_some(), "{name}: the learn run stopped");
    record.unwrap_or_else(|| panic!("{name}: no record"))
}

const TINY: &str = "x,y\n65521\nx^2+y,\nx*y+1\n";

/// Zero-dimensional ideals over GF(65521), each with its name.
fn zero_dimensional_cases() -> Vec<(String, String)> {
    let mut cases = vec![
        ("tiny".to_string(), TINY.to_string()),
        ("cubic-3".to_string(), dense(3, 3, 65521, 1)),
    ];
    for seed in 0..5 {
        cases.push((format!("square-{seed}"), random_square(65521, seed)));
    }
    cases
}

/// A lex record of a zero-dimensional ideal, replayed in lex over another
/// prime, gives the lex call's basis there, and with cofactors the checker
/// accepts it.
#[test]
fn a_zero_dimensional_lex_record_replays_in_lex() {
    for (name, text) in zero_dimensional_cases() {
        // A record the replay with cofactors fits: the echelon off
        // (input-redundancy-time.8).
        let record = learn(&name, &text, "lex", false);
        let there = over(&text, P31);
        let replayed = grobx::replay(&there, &params_in("lex", 1), &record, &mut NoTrace)
            .unwrap_or_else(|e| panic!("{name}: the replay failed: {e:?}"));
        assert_eq!(
            replayed.basis,
            run(&there, &params_in("lex", 1)).basis,
            "{name}"
        );
        let with = grobx::replay(&there, &with_cofactors_in("lex", 2), &record, &mut NoTrace)
            .unwrap_or_else(|e| panic!("{name}: the replay with cofactors failed: {e:?}"));
        assert_eq!(with.basis, replayed.basis, "{name}");
        let order = Family::Lex.checker(names(&text).len());
        assert_accepted_in(&name, &there, &with, &order, Mode::Exact);
    }
}

/// A lex record of an ideal that is not zero-dimensional, replayed in lex
/// over another prime, gives the lex call's basis there.
#[test]
fn a_positive_dimensional_lex_record_replays_in_lex() {
    let text = "x,y,z\n65521\nx*y-z,\ny^2-x\n";
    let record = learn("xy-z", text, "lex", true);
    let there = over(text, P31);
    let replayed = grobx::replay(&there, &params_in("lex", 1), &record, &mut NoTrace).unwrap();
    assert_eq!(replayed.basis, run(&there, &params_in("lex", 1)).basis);
}

/// A lex record replayed in another order does not fit.
#[test]
fn a_lex_record_does_not_fit_another_order() {
    let record = learn("tiny", TINY, "lex", true);
    for order in ["grevlex", "block:1"] {
        let got = grobx::replay(TINY, &params_in(order, 1), &record, &mut NoTrace);
        assert!(
            matches!(got, Err(Error::Misfit { step: 0 })),
            "in {order}: {got:?}"
        );
    }
}
