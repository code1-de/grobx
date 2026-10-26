//! order-family, the item's test code: a record holds its order. Learned in
//! lex over one prime and replayed in lex over another, it gives the lex
//! run's basis there; replayed under another order, it does not fit
//! (order-family.15, plan.md, _Edge cases_).

use grobx::Error;
use grobx::trace::{NoTrace, Record};

use crate::common::{P31, dense, params_in, random_system, run};

/// The instance with its characteristic, the second line, replaced by p.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let variables = lines.next().unwrap();
    lines.next();
    let rest = lines.next().unwrap_or("");
    format!("{variables}\n{p}\n{rest}")
}

fn learn(name: &str, text: &str, order: &str) -> Record {
    let (outcome, record) = grobx::learn(text, &params_in(order, 1), &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: learn failed in {order}: {e:?}"));
    assert!(outcome.basis.is_some(), "{name}: the learn run stopped");
    record.unwrap_or_else(|| panic!("{name}: no record"))
}

const TINY: &str = "x,y\n65521\nx^2+y,\nx*y+1\n";

#[test]
fn a_lex_record_replays_in_lex() {
    let mut cases = vec![
        ("tiny".to_string(), TINY.to_string()),
        ("cubic-3".to_string(), dense(3, 3, 65521, 1)),
    ];
    for seed in 0..5 {
        cases.push((format!("random-{seed}"), random_system(65521, seed)));
    }
    for (name, text) in cases {
        for order in ["lex", "block:1"] {
            let record = learn(&name, &text, order);
            let there = over(&text, P31);
            let replayed = grobx::replay(&there, &params_in(order, 1), &record, &mut NoTrace)
                .unwrap_or_else(|e| panic!("{name} in {order}: the replay failed: {e:?}"));
            assert_eq!(
                replayed.basis,
                run(&there, &params_in(order, 1)).basis,
                "{name} in {order}"
            );
        }
    }
}

/// The tiny instance's generators have the leading monomials x² and xy in
/// grevlex and in lex, so the inputs alone do not tell the orders apart.
#[test]
fn a_record_does_not_fit_another_order() {
    for (learned, replayed) in [
        ("lex", "grevlex"),
        ("grevlex", "lex"),
        ("grevlex", "block:1"),
        ("block:1", "grevlex"),
    ] {
        let record = learn("tiny", TINY, learned);
        let got = grobx::replay(TINY, &params_in(replayed, 1), &record, &mut NoTrace);
        assert!(
            matches!(got, Err(Error::Misfit { step: 0 })),
            "learned in {learned}, replayed in {replayed}: {got:?}"
        );
    }
}
