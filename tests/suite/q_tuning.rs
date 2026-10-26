//! q-lift, the item's test code: generated systems over Q, each accepted
//! with its cofactors in the exact mode in grevlex and in lex. The call in
//! lex without cofactors gives the same basis.

use crate::common::{dense, params_in, random_square, run};
use crate::q::accepted_q;

/// The systems over Q here, each with a name of its own.
pub(crate) fn systems() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for seed in [1, 2] {
        out.push((format!("cubic-2-{seed}"), dense(2, 3, 0, seed)));
    }
    for seed in [1, 3] {
        out.push((format!("square-{seed}"), random_square(0, seed)));
    }
    out
}

/// The systems over Q with cofactors: the checker accepts in the exact
/// mode.
#[test]
fn with_cofactors() {
    for (name, text) in systems() {
        accepted_q(&name, &text, "grevlex");
    }
}

/// The systems over Q in lex: the checker accepts the call with cofactors
/// in the exact mode, and the call without cofactors gives its basis.
#[test]
fn in_lex() {
    for (name, text) in systems() {
        let checked = accepted_q(&name, &text, "lex");
        let plain = run(&text, &params_in("lex", 0x9e37_79b9));
        assert_eq!(plain.basis, checked.basis, "{name}");
    }
}
