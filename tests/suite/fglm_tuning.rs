//! fglm-lex, the item's test code: dense systems over GF(65521) in lex,
//! zero-dimensional on most seeds. Each is computed with cofactors,
//! canonical in lex, and accepted by the checker in the cheap mode, k chosen
//! as checker-cheap.16 chooses it.

use std::num::NonZeroU32;

use grobx_checker::Mode;

use crate::common::{P, assert_accepted_in, degree_bound, dense, points, run, with_cofactors_in};
use crate::orders::{Family, assert_canonical, names};

fn in_lex(name: &str, text: &str) {
    let outcome = run(text, &with_cofactors_in("lex", 0x0de7));
    let ours = outcome.basis.clone().unwrap();
    assert_canonical(name, text, &ours, "lex");
    let k = points(P, degree_bound(text, &outcome));
    let mode = Mode::Cheap {
        points: NonZeroU32::new(k).unwrap(),
        seed: 0x0de7 ^ name.len() as u64,
    };
    assert_accepted_in(
        name,
        text,
        &outcome,
        &Family::Lex.checker(names(text).len()),
        mode,
    );
}

/// Dense cubic systems on two and three variables in lex.
#[test]
fn dense_systems_in_lex() {
    for m in 2..=3 {
        in_lex(&format!("cubic-{m}"), &dense(m, 3, P, m as u64));
    }
}
