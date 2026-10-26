//! order-family, the item's test code: generated systems over GF(65521) in
//! lex and in `block:k`, k = ⌊n/2⌋. Each is computed with cofactors,
//! canonical in its order, and accepted by the checker in the cheap mode at
//! the points checker-cheap.16 chooses. Each lex basis is the
//! `block:1,…,1` basis.

use std::num::NonZeroU32;

use grobx_checker::Mode;

use crate::common::{
    P, assert_accepted_in, degree_bound, dense, params_in, points, random_square, run,
    with_cofactors_in,
};
use crate::orders::{Family, assert_canonical, lex_as_blocks, names};

fn tuning(name: &str, text: &str, order: &str) {
    let n = names(text).len();
    let outcome = run(text, &with_cofactors_in(order, 0x0de7));
    let ours = outcome.basis.clone().unwrap();
    assert_canonical(name, text, &ours, order);
    let k = points(P, degree_bound(text, &outcome));
    let mode = Mode::Cheap {
        points: NonZeroU32::new(k).unwrap(),
        seed: 0x0de7 ^ name.len() as u64,
    };
    let checker = Family::named(order).checker(n);
    assert_accepted_in(name, text, &outcome, &checker, mode);
    if order == "lex" {
        let blocks = lex_as_blocks(n);
        let same = run(text, &params_in(&blocks, 1)).basis.unwrap();
        assert_eq!(ours, same, "{name}: lex and {blocks}");
    }
}

/// A dense cubic system on two variables and random square systems over
/// GF(65521), each with its name.
fn systems() -> Vec<(String, String)> {
    let mut out = vec![("cubic-2".to_string(), dense(2, 3, P, 2))];
    // test-mq-systems.12 design -- in_lex runs a system on three variables in block:1,1 as well, so it takes the random square system of degrees 2, 3, and 2, where the map's dense cubic one on three variables runs for seconds (log 0084-3)
    for seed in [1, 2, 3, 4, 27] {
        out.push((format!("square-{seed}"), random_square(P, seed)));
    }
    out
}

#[test]
fn in_lex() {
    for (name, text) in systems() {
        tuning(&name, &text, "lex");
    }
}

#[test]
fn in_two_blocks() {
    for (name, text) in systems() {
        let order = format!("block:{first}", first = names(&text).len() / 2);
        tuning(&name, &text, &order);
    }
}
