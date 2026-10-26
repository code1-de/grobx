//! cofactor-stop-latency, the item's test code (cofactor-stop-latency.1 to
//! .8). Calls with cofactors stop soon past their deadline and soon after a
//! cancel; generous bounds change no result; and the edge cases of
//! `OBJECTIVES.md` hold (plan.md, _Test code_, 6 to 9).

use std::num::NonZeroU32;
use std::thread;
use std::time::{Duration, Instant};

use grobx::{Cancel, Params, Stop};
use grobx_checker::{Mode, Order};

use crate::common::{
    P, P31, assert_accepted_in, degree_bound, dense, points, random_square, run, with_cofactors,
    with_cofactors_in,
};
use crate::orders::accepted_in;
use crate::q::accepted_q;

/// Every bound set, none of them reached.
fn generous(mut p: Params) -> Params {
    p.deadline = Some(Duration::from_secs(3600));
    p.cancel = Some(Cancel::new());
    p.memory = Some(1 << 40);
    p
}

/// The long calls of tests 6 and 8, over GF(65521) and over ℚ: each runs
/// for seconds without a bound.
fn long() -> [(&'static str, String); 2] {
    [
        ("cubic-8-p65521", dense(8, 3, P, 1)),
        ("cubic-5-p0", dense(5, 3, 0, 1)),
    ]
}

/// Test 6, .1 to .8: a call with cofactors with a deadline of 100, 200, or
/// 300 ms stops with `Deadline`, with no basis and no cofactors, within
/// 0.5 s past the deadline.
#[test]
fn calls_with_cofactors_stop_soon_past_the_deadline() {
    for (name, text) in long() {
        for ms in [100, 200, 300] {
            let deadline = Duration::from_millis(ms);
            let mut p = with_cofactors(1);
            p.deadline = Some(deadline);
            let start = Instant::now();
            let outcome = grobx::groebner(&text, &p).expect("the call runs");
            let ended = start.elapsed();
            assert_eq!(outcome.stopped, Some(Stop::Deadline), "{name} at {ms} ms");
            assert_eq!(
                (outcome.basis, outcome.cofactors),
                (None, None),
                "{name} at {ms} ms"
            );
            assert!(
                ended < deadline + Duration::from_millis(500),
                "{name} at {ms} ms: returned {late:?} past the deadline",
                late = ended.saturating_sub(deadline)
            );
        }
    }
}

/// The calls of test 7: name, system, and order.
fn systems() -> [(&'static str, String, &'static str); 4] {
    [
        ("cubic-2-p0", dense(2, 3, 0, 1), "grevlex"),
        ("square-p0", random_square(0, 3), "grevlex"),
        ("cubic-3-p65521", dense(3, 3, P, 1), "lex"),
        ("cubic-2-p65521", dense(2, 3, P, 2), "lex"),
    ]
}

/// Test 7, .1 to .8: with a deadline of an hour, a cancel flag never
/// raised, and a memory ceiling of 2^40 bytes, under seeds 1 and 2, each
/// call with cofactors returns the basis and the cofactor text of the same
/// call without bounds. Over GF(65521) the cheap checker accepts each.
#[test]
fn generous_bounds_change_no_result() {
    for (stem, text, order) in systems() {
        for seed in [1, 2] {
            let bounded = run(&text, &generous(with_cofactors_in(order, seed)));
            let free = run(&text, &with_cofactors_in(order, seed));
            assert_eq!(bounded.basis, free.basis, "{stem} in {order}, seed {seed}");
            assert_eq!(
                bounded.cofactors, free.cofactors,
                "{stem} in {order}, seed {seed}"
            );
            if !stem.ends_with("-p0") {
                let k = points(P, degree_bound(&text, &bounded));
                let mode = Mode::Cheap {
                    points: NonZeroU32::new(k).unwrap(),
                    seed: 0x6373 ^ seed,
                };
                let checked = if order == "lex" {
                    Order::Lex
                } else {
                    Order::Grevlex
                };
                assert_accepted_in(stem, &text, &bounded, &checked, mode);
            }
        }
    }
}

/// Test 8, .1: a cancel raised from another thread 100 ms into a long call
/// with cofactors over ℚ stops the call with `Cancel` within 0.5 s of the
/// raise.
#[test]
fn a_cancel_stops_a_call_with_cofactors_soon() {
    let [_, (stem, text)] = long();
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
    assert_eq!(outcome.stopped, Some(Stop::Cancel), "{stem}");
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

/// Test 9, .1: the edge cases over GF(65521), GF(2^31 − 1), and ℚ, in
/// grevlex, lex, and `block:1`, with cofactors. With generous bounds each
/// basis and cofactor text is the one without bounds, and the exact
/// checker accepts the result; with a deadline of zero each call stops
/// with `Deadline`.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        for p in [P, P31, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                let name = format!("{case} over {p} in {order}");
                let free = if p == 0 {
                    accepted_q(&name, &text, order)
                } else {
                    accepted_in(&name, &text, order, Mode::Exact)
                };
                let bounded = run(&text, &generous(with_cofactors_in(order, 0x0de7)));
                assert_eq!(bounded.basis, free.basis, "{name}");
                assert_eq!(bounded.cofactors, free.cofactors, "{name}");

                let mut zero = with_cofactors_in(order, 0x0de7);
                zero.deadline = Some(Duration::ZERO);
                let outcome = grobx::groebner(&text, &zero).expect("the call runs");
                assert_eq!(outcome.stopped, Some(Stop::Deadline), "{name}");
                assert_eq!((outcome.basis, outcome.cofactors), (None, None), "{name}");
            }
        }
    }
}
