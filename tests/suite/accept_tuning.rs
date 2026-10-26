//! The item's test code: generated systems over GF(65521), each computed
//! with cofactors and accepted by the checker in the cheap mode, k chosen as
//! checker-cheap.16 chooses it.

use std::num::NonZeroU32;

use grobx_checker::Mode;

use crate::common::{
    P, assert_accepted, degree_bound, dense, points, random_square, run, with_cofactors,
};

fn accept(name: &str, text: &str) {
    let outcome = run(text, &with_cofactors(0x7a11));
    let k = points(P, degree_bound(text, &outcome));
    let mode = Mode::Cheap {
        points: NonZeroU32::new(k).unwrap(),
        seed: 0xc4ec ^ name.len() as u64,
    };
    assert_accepted(name, text, &outcome, mode);
}

/// Dense cubic systems on two to four variables and random square systems:
/// the checker accepts each with its cofactors in the cheap mode.
#[test]
fn the_cheap_mode_accepts() {
    let mut cases: Vec<(String, String)> = (2..=4)
        .map(|m| (format!("cubic-{m}"), dense(m, 3, P, m as u64)))
        .collect();
    for seed in 1..=3 {
        cases.push((format!("square-{seed}"), random_square(P, seed)));
    }
    for (name, text) in cases {
        accept(&name, &text);
    }
}
