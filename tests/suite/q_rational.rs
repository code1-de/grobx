//! q-rational-time, the item's test code: calls over Q with cofactors are
//! accepted by the checker in the exact mode, in grevlex and in lex (plan.md,
//! _Test code_).

use grobx_checker::Mode;

use crate::common::{assert_accepted, dense, params_in, random_square, run, with_cofactors};
use crate::q::accepted_q;

/// The seed of every call here.
const SEED: u64 = 0x9a7;

/// Calls over Q with cofactors (q-rational-time.4 to .6): the checker
/// accepts each result.
#[test]
fn accepted_with_cofactors() {
    for (name, text) in [
        ("cubic-3", dense(3, 3, 0, 4)),
        ("square-3", random_square(0, 3)),
    ] {
        let outcome = run(&text, &with_cofactors(SEED));
        assert_accepted(name, &text, &outcome, Mode::Exact);
    }
}

/// Two systems over Q in lex, as `q_tuning.rs`'s `in_lex` checks them: the
/// checker accepts the call with cofactors in the exact mode, and the call
/// without cofactors gives its basis.
#[test]
fn in_lex() {
    for (name, text) in [
        ("cubic-2", dense(2, 3, 0, 5)),
        ("square-3", random_square(0, 3)),
    ] {
        let checked = accepted_q(name, &text, "lex");
        let plain = run(&text, &params_in("lex", SEED));
        assert_eq!(plain.basis, checked.basis, "{name}");
    }
}
