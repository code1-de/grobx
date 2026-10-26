//! text-stop-latency, the item's test code. A call with cofactors stops
//! soon past its deadline and soon after a cancel, both read in its run;
//! generous bounds change no text (text-stop-latency.1 to .9); and
//! the edge cases of `OBJECTIVES.md` hold (plan.md, _Test code_, 6 and 8,
//! and 5 and 7 cut to stops in the run).

use std::num::NonZeroU32;
use std::thread;
use std::time::{Duration, Instant};

use grobx::{Cancel, CheckerMode, Params, Stop};
use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted, degree_bound, dense, params, params_in, points, random_square, run,
    with_cofactors, with_cofactors_in,
};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// Every bound set, none of them reached.
fn generous(mut p: Params) -> Params {
    p.deadline = Some(Duration::from_secs(3600));
    p.cancel = Some(Cancel::new());
    p.memory = Some(1 << 40);
    p
}

/// The long calls of tests 5 and 7, over GF(65521) and over ℚ: each runs
/// for seconds without a bound.
fn long() -> [(&'static str, String); 2] {
    [
        ("cubic-8-p65521", dense(8, 3, P, 1)),
        ("cubic-5-p0", dense(5, 3, 0, 1)),
    ]
}

/// Test 5: a call with cofactors and a deadline of 200 ms stops in its run
/// with `Deadline`, with no text, within 0.5 s past its deadline.
#[test]
fn a_call_with_cofactors_stops_soon_past_the_deadline() {
    let deadline = Duration::from_millis(200);
    for (name, text) in long() {
        let mut p = with_cofactors(1);
        p.deadline = Some(deadline);
        let start = Instant::now();
        let outcome = grobx::groebner(&text, &p).expect("the call runs");
        let took = start.elapsed();
        assert_eq!(outcome.stopped, Some(Stop::Deadline), "{name}");
        assert_eq!((outcome.basis, outcome.cofactors), (None, None), "{name}");
        assert!(
            took < deadline + Duration::from_millis(500),
            "{name}: returned {late:?} past the deadline",
            late = took.saturating_sub(deadline)
        );
    }
}

/// The systems of test 6: two over GF(65521) and two over ℚ.
fn systems() -> [(&'static str, String); 4] {
    [
        ("cubic-3-p65521", dense(3, 3, P, 1)),
        ("square-p65521", random_square(P, 2)),
        ("cubic-2-p0", dense(2, 3, 0, 1)),
        ("square-p0", random_square(0, 2)),
    ]
}

/// Test 6, .1 to .9: with a deadline of an hour, a cancel flag never
/// raised, and a memory ceiling of 2^40 bytes, under seeds 1 and 2, each
/// basis and each cofactor text is that of the call without bounds, and
/// over GF(65521) the cheap checker accepts the result. With the exact
/// checker the call on cubic-3-p65521 is certified.
#[test]
fn generous_bounds_change_no_text() {
    for (name, text) in systems() {
        let over_q = text.lines().nth(1) == Some("0");
        for seed in [1, 2] {
            let outcome = run(&text, &generous(params(seed)));
            let plain = run(&text, &params(seed));
            assert_eq!(outcome.basis, plain.basis, "{name}, seed {seed}");
            let bounded = run(&text, &generous(with_cofactors(seed)));
            let free = run(&text, &with_cofactors(seed));
            assert_eq!(bounded.basis, free.basis, "{name}, seed {seed}");
            assert_eq!(bounded.cofactors, free.cofactors, "{name}, seed {seed}");
            if !over_q {
                let k = points(P, degree_bound(&text, &bounded));
                let mode = Mode::Cheap {
                    points: NonZeroU32::new(k).unwrap(),
                    seed: 0x7e47 ^ seed,
                };
                assert_accepted(name, &text, &bounded, mode);
            }
        }
    }
    let text = dense(3, 3, P, 1);
    let mut p = generous(with_cofactors(1));
    p.checker = Some(CheckerMode::Exact);
    let outcome = run(&text, &p);
    assert_eq!(outcome.certified, Some(true), "cubic-3-p65521");
}

/// Test 7: a cancel raised from another thread 100 ms into a long call
/// with cofactors over ℚ stops the call, in its run, with `Cancel` within
/// 0.5 s of the raise.
#[test]
fn a_cancel_stops_a_call_with_cofactors_soon() {
    let [_, (name, text)] = long();
    let at = Duration::from_millis(100);
    let cancel = Cancel::new();
    let mut p = with_cofactors(1);
    p.cancel = Some(cancel.clone());
    let stopper = thread::spawn(move || {
        thread::sleep(at);
        cancel.set();
        Instant::now()
    });
    let outcome = grobx::groebner(&text, &p).expect("the call runs");
    let ended = Instant::now();
    let raised = stopper.join().unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Cancel), "{name}");
    assert!(
        ended.saturating_duration_since(raised) < Duration::from_millis(500),
        "{took:?} after the raise",
        took = ended.saturating_duration_since(raised)
    );
}

/// The edge cases of `OBJECTIVES.md` on x, y, z: name and generators.
const EDGE: [(&str, &str); 7] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
];

/// Test 8, .1: the edge cases over GF(65521), GF(2^31 − 1), and ℚ, in
/// grevlex, lex, and `block:1`. With generous bounds each basis and
/// cofactor text is the one without bounds, and the basis is canonical; a
/// call with cofactors and no bounds is accepted by the exact checker; with
/// a deadline of zero each call stops with `Deadline`.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        for p in [P, P31, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                let name = format!("{case} over {p} in {order}");
                let free = run(&text, &params_in(order, 0x7e47)).basis.unwrap();
                let bounded = run(&text, &generous(params_in(order, 0x7e47)));
                assert_eq!(bounded.basis.as_deref(), Some(free.as_str()), "{name}");
                if p == 0 {
                    accepted_q(&name, &text, order);
                } else {
                    assert_canonical(&name, &text, &free, order);
                    accepted_in(&name, &text, order, Mode::Exact);
                }
                let with = run(&text, &generous(with_cofactors_in(order, 0x7e47)));
                let without = run(&text, &with_cofactors_in(order, 0x7e47));
                assert_eq!(with.basis.as_deref(), Some(free.as_str()), "{name}");
                assert_eq!(with.cofactors, without.cofactors, "{name}");

                let mut zero = params_in(order, 0x7e47);
                zero.deadline = Some(Duration::ZERO);
                let outcome = grobx::groebner(&text, &zero).expect("the call runs");
                assert_eq!(outcome.stopped, Some(Stop::Deadline), "{name}");
                assert_eq!(outcome.basis, None, "{name}");
            }
        }
    }
}
