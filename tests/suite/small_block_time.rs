//! small-block-time, the item's test code through the crate: the suite
//! runs under `grobx::Allocator`, calls over GF(p) and over ℚ under three
//! seeds, the edge cases in three orders over both fields, the checker
//! inside the call, and the memory ceiling (plan.md, _Test code_, 6 to 9).

use std::time::Duration;

use grobx::{CheckerMode, Stop};
use grobx_checker::Mode;

use crate::common::{P, dense, params, params_in, random_square, run, with_cofactors_in};
use crate::orders::accepted_in;
use crate::q::accepted_q;

/// The deadline of the calls here that set one, far past what each takes.
const DEADLINE: Duration = Duration::from_secs(60);

#[cfg(target_os = "linux")]
unsafe extern "C" {
    /// mimalloc's `mi_is_in_heap_region`: whether the pointer lies in
    /// mimalloc's heap region.
    fn mi_is_in_heap_region(p: *const std::ffi::c_void) -> bool;
}

#[cfg(target_os = "linux")]
fn mimalloc_holds<T>(p: *const T) -> bool {
    // SAFETY: mi_is_in_heap_region reads mimalloc's own bookkeeping alone.
    unsafe { mi_is_in_heap_region(p.cast()) }
}

/// Test 6, small-block-time.1, .2, .4: the suite's binary runs under
/// `grobx::Allocator`, a block of at most `grobx::SMALL` bytes in
/// mimalloc's heap region and a larger one outside it, on Linux; and
/// calls over GF(65521) and over ℚ return one basis under three seeds.
#[test]
fn the_allocator_and_three_seeds() {
    #[cfg(target_os = "linux")]
    {
        let small = std::hint::black_box(vec![0u8; grobx::SMALL]);
        let large = std::hint::black_box(vec![0u8; grobx::SMALL + 1]);
        assert!(mimalloc_holds(small.as_ptr()));
        assert!(!mimalloc_holds(large.as_ptr()));
    }
    for text in [
        dense(4, 3, P, 1),
        random_square(P, 1),
        dense(3, 3, 0, 1),
        random_square(0, 1),
    ] {
        let bases: Vec<_> = (1..=3)
            .map(|seed| {
                let mut p = params(seed);
                p.deadline = Some(DEADLINE);
                run(&text, &p).basis
            })
            .collect();
        assert_eq!(bases[0], bases[1], "seeds 1 and 2\n{text}");
        assert_eq!(bases[0], bases[2], "seeds 1 and 3\n{text}");
    }
}

/// The edge cases of `OBJECTIVES.md` on x, y, z, their generators over
/// any field.
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

/// Test 7, small-block-time.1: the edge cases over GF(65521), over
/// GF(2^31 − 1), and over ℚ, in grevlex, lex, and `block:1`: each basis
/// canonical and accepted by the checker in the exact mode, with
/// cofactors, and the call without cofactors returns the same basis.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE {
        for characteristic in ["65521", "2147483647", "0"] {
            let text = format!("x,y,z\n{characteristic}\n{generators}\n");
            for order in ["grevlex", "lex", "block:1"] {
                let checked = if characteristic == "0" {
                    accepted_q(case, &text, order)
                } else {
                    accepted_in(case, &text, order, Mode::Exact)
                };
                let mut p = params_in(order, 1);
                p.deadline = Some(DEADLINE);
                let plain = run(&text, &p);
                assert_eq!(
                    plain.basis, checked.basis,
                    "{case} over {characteristic} in {order}"
                );
            }
        }
    }
}

/// Test 8, small-block-time.1: dense cubic and random square systems over both
/// fields with cofactors, the checker inside the call in the exact mode:
/// each result certified.
#[test]
fn the_checker_inside_the_call() {
    for text in [
        dense(3, 3, P, 1),
        random_square(P, 2),
        dense(3, 3, 0, 1),
        random_square(0, 2),
    ] {
        let mut p = with_cofactors_in("grevlex", 5);
        p.checker = Some(CheckerMode::Exact);
        p.deadline = Some(DEADLINE);
        let outcome = run(&text, &p);
        assert_eq!(
            outcome.certified,
            Some(true),
            "{flaw:?}\n{text}",
            flaw = outcome.flaw
        );
    }
}

/// Test 9, small-block-time.6: a memory ceiling of 1 byte stops a long call
/// over GF(65521) and one over ℚ with `Stop::Memory`, and a ceiling of 1 TiB
/// leaves the result of a short call over each as the call without one.
#[test]
fn the_memory_ceiling() {
    for (field, long, short) in [
        ("GF(65521)", dense(8, 3, P, 1), dense(4, 3, P, 1)),
        ("ℚ", dense(5, 3, 0, 1), dense(3, 3, 0, 1)),
    ] {
        let mut tight = params(1);
        tight.memory = Some(1);
        let outcome = grobx::groebner(&long, &tight).unwrap();
        assert!(outcome.basis.is_none(), "{field}: {outcome:?}");
        assert_eq!(outcome.stopped, Some(Stop::Memory), "{field}");
        let mut loose = params(1);
        loose.memory = Some(1 << 40);
        assert_eq!(run(&short, &loose), run(&short, &params(1)), "{field}");
    }
}
