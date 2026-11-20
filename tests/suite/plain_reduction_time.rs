//! plain-reduction-time, the item's test code: the edge cases of
//! `OBJECTIVES.md` with signatures off and on, and a deadline that stops a
//! long call (`iterations/0062-plain-reduction-time/plan.md`, _Test code_,
//! 7 and 11).

use std::time::{Duration, Instant};

use grobx::Stop;
use grobx_checker::Mode;

use crate::common::{EDGE_BASES, P, P31, assert_accepted, dense, params, run, with_cofactors};
use crate::orders::assert_canonical;
use crate::q::accepted_q;

/// Test 7: the edge cases over 65521, 2^31 − 1, and ℚ in grevlex, with
/// signatures off and on. With no cofactors and signatures off the plain
/// run combines rows, with signatures on the signature run computes, and
/// each basis is the call's with cofactors, which the checker accepts
/// in the exact mode; over 65521 both are as written, and over ℚ the
/// basis is the accepted one of `accepted_q`.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE_BASES {
        for p in [P, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [false, true] {
                let name = format!("{case} over {p}, signatures {signatures}");
                let mut plain = params(0x62e0);
                plain.signatures = signatures;
                let combined = run(&text, &plain).basis.unwrap();
                let mut tracked = with_cofactors(0x62e0);
                tracked.signatures = signatures;
                let outcome = run(&text, &tracked);
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                assert_canonical(&name, &text, &combined, "grevlex");
                assert_eq!(Some(&combined), outcome.basis.as_ref(), "{name}");
                if p == P {
                    assert_eq!(combined, want, "{name}");
                }
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        let accepted = accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
        for signatures in [false, true] {
            let mut plain = params(0x62e1);
            plain.signatures = signatures;
            assert_eq!(
                run(&text, &plain).basis,
                accepted.basis,
                "{case} over ℚ, signatures {signatures}"
            );
        }
    }
}

/// Test 11: a dense cubic system over GF(65521) on eight variables, under seed
/// 1 with a deadline of 300 ms: the call stops
/// with `Deadline`, no basis, and returns within 1 s past the deadline.
#[test]
fn a_deadline_stops_a_long_call() {
    const DEADLINE: Duration = Duration::from_millis(300);
    let text = dense(8, 3, P, 1);
    let mut p = params(1);
    p.deadline = Some(DEADLINE);
    let start = Instant::now();
    let outcome = grobx::groebner(&text, &p).expect("the call runs");
    let took = start.elapsed();
    assert_eq!(outcome.stopped, Some(Stop::Deadline), "cubic-8");
    assert_eq!(outcome.basis, None, "cubic-8");
    assert!(
        took < DEADLINE + Duration::from_secs(1),
        "cubic-8: returned {late:?} past the deadline",
        late = took.saturating_sub(DEADLINE)
    );
}
