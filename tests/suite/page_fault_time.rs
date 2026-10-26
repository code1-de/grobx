//! page-fault-time, the item's integration tests: after a call, glibc's
//! malloc serves the blocks below `grobx::MMAP_THRESHOLD` from the heap,
//! whatever the call returned, in every thread that called; and the edge
//! cases return what they returned before
//! (`iterations/0090-page-fault-time/plan.md`, _Test code_, 1 to 5).

use std::thread;

use grobx::{Error, MMAP_THRESHOLD, TOP_PAD};
use grobx_checker::Mode;

use crate::common::{P, assert_accepted, params, run, with_cofactors};
use crate::heap::{HEAP_EXCESS, MAPPED_EXCESS};
use crate::q::accepted_q;
use crate::run_generators_time::EDGE;

/// One system whose call sets the parameters.
const SYSTEM: &str = "x,y\n65521\nx^2+y,\nx*y+1\n";

/// What `malloc_usable_size` gives for a block of `size` bytes, less
/// `size`.
fn excess(size: usize) -> u64 {
    // SAFETY: the block is malloc's, read by malloc_usable_size and freed
    // once; no byte of it is read.
    unsafe {
        let p = libc::malloc(size);
        assert!(!p.is_null(), "malloc({size})");
        let usable = libc::malloc_usable_size(p);
        libc::free(p);
        u64::try_from(usable - size).unwrap()
    }
}

/// The bytes of `MMAP_THRESHOLD` and `TOP_PAD`, which are `c_int`s.
fn bytes(n: libc::c_int) -> usize {
    usize::try_from(n).unwrap()
}

/// Test 1's blocks, in the calling thread: below the threshold from the
/// heap, and one no free space of the heap holds in a mapping.
fn assert_padded(case: &str) {
    for size in [1 << 20, bytes(MMAP_THRESHOLD) - (64 << 10)] {
        let e = excess(size);
        assert!(e < HEAP_EXCESS, "{case}: a block of {size} bytes took a mapping ({e} past it)");
    }
    let size = 4 * bytes(TOP_PAD);
    let e = excess(size);
    assert!(e >= MAPPED_EXCESS, "{case}: a block of {size} bytes came from the heap ({e} past it)");
}

/// Test 1, page-fault-time.4 and .5: after a call, the blocks below a
/// huge page come from the heap.
#[test]
fn blocks_below_a_huge_page_come_from_the_heap() {
    let outcome = run(SYSTEM, &params(1));
    assert!(outcome.basis.is_some());
    assert_padded("after one call");
}

/// Test 2, page-fault-time.6: the edge cases over GF(65521) and over ℚ,
/// each with cofactors and accepted by the exact checker, return what they
/// returned before, and the blocks hold after each.
#[test]
fn the_edge_cases() {
    for (case, gens) in EDGE {
        let instance = format!("x,y,z\n{P}\n{gens}\n");
        let outcome = run(&instance, &with_cofactors(1));
        assert_accepted(case, &instance, &outcome, Mode::Exact);
        assert_padded(case);
        accepted_q(case, &format!("x,y,z\n0\n{gens}\n"), "grevlex");
        assert_padded(case);
    }
    let zero = run(&format!("x,y,z\n{P}\n0\n"), &params(1));
    assert_eq!(zero.basis.as_deref(), Some("[]:\n"));
    let unit = run(&format!("x,y,z\n{P}\nx*z-1,\nx\n"), &params(1));
    assert_eq!(unit.basis.as_deref(), Some("[1]:\n"));
    let gb = run(&format!("x,y\n{P}\nx+y,\ny^2+1\n"), &params(1));
    assert_eq!(gb.basis.as_deref(), Some("[x+y,\ny^2+1]:\n"));
}

/// Test 3, page-fault-time.6: a call that returns an error has set the
/// parameters before it read its text.
#[test]
fn an_error_sets_them_too() {
    let got = grobx::groebner("x,y\n65521\nx^^2\n", &params(1));
    assert!(matches!(got, Err(Error::Syntax { .. })), "{got:?}");
    assert_padded("after an error");
}

/// Test 4, page-fault-time.6: eight threads call at once, get one basis,
/// and each sees the parameters set after its call.
#[test]
fn calls_from_eight_threads() {
    let bases: Vec<Option<String>> = thread::scope(|s| {
        let calls: Vec<_> = (0..8)
            .map(|k| {
                s.spawn(move || {
                    let outcome = run(SYSTEM, &params(k + 1));
                    assert_padded(&format!("thread {k}"));
                    outcome.basis
                })
            })
            .collect();
        calls.into_iter().map(|c| c.join().unwrap()).collect()
    });
    assert!(bases[0].is_some());
    assert!(bases.iter().all(|b| *b == bases[0]), "{bases:?}");
}

/// Test 5, page-fault-time.7: the error names `mallopt` and the
/// parameter it refused.
#[test]
fn the_error_names_its_parameter() {
    let text = Error::Malloc { param: "M_TOP_PAD" }.to_string();
    assert!(text.contains("mallopt") && text.contains("M_TOP_PAD"), "{text}");
}
