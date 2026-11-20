//! cyclic-gap, the item's test code: the edge cases of `OBJECTIVES.md` and
//! the plain run in `block:2` and before FGLM in lex
//! (`iterations/0066-cyclic-gap/plan.md`,
//! _Test code_, 5 and 9).

use grobx_checker::Mode;

use crate::common::{
    EDGE_BASES, P, P31, assert_accepted, params, params_in, random_square, run, with_cofactors,
};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// Test 5: the edge cases over 65521, 7, 2^31 − 1, and ℚ in grevlex,
/// signatures off and on. Over 7 each row is reduced on its own, from
/// 65521 the plain run combines rows. The basis with no cofactors is
/// canonical and is the call's with cofactors, which the checker accepts
/// in the exact mode; over 65521 both are as written, and over ℚ the basis
/// is the accepted one of `accepted_q`.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE_BASES {
        for p in [P, 7, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [false, true] {
                let name = format!("{case} over {p}, signatures {signatures}");
                let mut plain = params(0x6a96);
                plain.signatures = signatures;
                let basis = run(&text, &plain).basis.unwrap();
                let mut tracked = with_cofactors(0x6a96);
                tracked.signatures = signatures;
                let outcome = run(&text, &tracked);
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                assert_canonical(&name, &text, &basis, "grevlex");
                assert_eq!(Some(&basis), outcome.basis.as_ref(), "{name}");
                if p == P {
                    assert_eq!(basis, want, "{name}");
                }
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        let accepted = accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
        for signatures in [false, true] {
            let mut plain = params(0x6a97);
            plain.signatures = signatures;
            assert_eq!(
                run(&text, &plain).basis,
                accepted.basis,
                "{case} over ℚ, signatures {signatures}"
            );
        }
    }
}

/// Test 9: the plain run in `block:2`, and in lex the plain run's grevlex
/// basis changed by FGLM. A random square system on three variables over 65521 in
/// lex and in `block:2`, signatures off, under two seeds: the
/// basis is that of the call with cofactors that the checker accepts in
/// the exact mode.
#[test]
fn outside_grevlex() {
    // test-mq-systems.12 design-checked -- a call that F4 runs in block:2 takes the random square system of degrees 2, 3, and 2, where the map's dense cubic one on three variables runs for seconds (log 0084-3)
    let text = random_square(P, 27);
    for order in ["lex", "block:2"] {
        let name = format!("square-27-p65521 in {order}");
        let accepted = accepted_in(&name, &text, order, Mode::Exact);
        for seed in [0x6a9c, 0x6a9d] {
            let mut p = params_in(order, seed);
            p.signatures = false;
            let basis = run(&text, &p).basis;
            assert_eq!(basis, accepted.basis, "{name}, seed {seed:#x}");
        }
    }
}
