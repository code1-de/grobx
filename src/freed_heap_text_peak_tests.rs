//! freed-heap-text-peak, the item's test code: a call that gives the
//! allocators' free memory back before its texts returns what the same
//! call returns without, over GF(p) and ℚ, in every order, with cofactors
//! and the checker, on the edge cases, under a memory ceiling, and under
//! deadlines; and the free memory below the heap's top goes back
//! (`iterations/0116-freed-heap-text-peak/plan.md`, _Test code_, 1 to 3).
//!
//! A pair of calls is the same call with `TEXTS_UNTRIMMED` set, the trim
//! off, and unset.

use std::time::Duration;

use crate::budget::TEXTS_UNTRIMMED;
use crate::call::built;
#[cfg(target_os = "linux")]
use crate::child_tests::measures_here;
#[cfg(target_os = "linux")]
use crate::os::resident;
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::FIELDS;
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_sparse};
use crate::{CheckerMode, MonomialOrder, Outcome, Params, Stop};

/// The call of `text` with the trim before the texts off where `untrimmed`.
fn call(text: &str, params: &Params, untrimmed: bool) -> Outcome {
    TEXTS_UNTRIMMED.set(untrimmed);
    let outcome = crate::groebner(text, params);
    TEXTS_UNTRIMMED.set(false);
    outcome.unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
}

/// Asserts that the call of `text` returns with the trim what it returns
/// without, and returns the outcome with the trim.
fn pair(name: &str, text: &str, params: &Params) -> Outcome {
    let without = call(text, params, true);
    let with = call(text, params, false);
    assert_eq!(with.basis, without.basis, "{name}: the bases");
    assert_eq!(
        with.cofactors, without.cofactors,
        "{name}: the cofactor texts"
    );
    assert_eq!(with.stopped, without.stopped, "{name}: the stops");
    assert_eq!(with.certified, without.certified, "{name}: the verdicts");
    assert_eq!(with.flaw, without.flaw, "{name}: the flaws");
    assert_eq!(with.stats, without.stats, "{name}: the stats");
    with
}

/// Test 1, freed-heap-text-peak.1 to .6: every call returns with the trim
/// what it returns without: random sparse and dense systems and the edge
/// cases over 7, 65521, 2^31 − 1, 2^61 − 1, and ℚ, the signatures off and
/// on; a dense system and an eliminable one in lex and `block:1`; with
/// cofactors and the exact checker; and under a memory ceiling.
#[test]
fn every_result_stays() {
    for p in FIELDS {
        let mut texts: Vec<(String, String)> = Vec::new();
        for s in 1..=3 {
            texts.push((format!("sparse4-{s}"), random_sparse(4, p, s)));
            texts.push((format!("sparse5-{s}"), random_sparse(5, p, s)));
            texts.push((format!("dense33-{s}"), dense(3, 3, p, s)));
        }
        for (k, gens) in EDGE.iter().enumerate() {
            texts.push((format!("edge {k}"), format!("x,y,z\n{p}\n{gens}\n")));
        }
        for (system, text) in &texts {
            for signatures in [false, true] {
                let mut params = reported(MonomialOrder::Grevlex);
                params.signatures = signatures;
                let name = format!("{system} over {p}, signatures {signatures}");
                let outcome = pair(&name, text, &params);
                assert!(outcome.basis.is_some(), "{name}: a basis");
            }
        }
    }
    let eliminable = "x,y,z\n7\nx+z,\nz^2+1\n";
    for order in ["lex", "block:1"] {
        let params = reported(order.parse().unwrap());
        for (name, text) in [
            ("dense33-p7-4", dense(3, 3, 7, 4)),
            ("eliminable", eliminable.to_string()),
        ] {
            pair(&format!("{name} in {order}"), &text, &params);
        }
    }
    for p in [65521, 0] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        if built(&params).is_err() {
            continue;
        }
        let name = format!("sparse4-p{p}-1 with cofactors");
        let outcome = pair(&name, &random_sparse(4, p, 1), &params);
        assert!(outcome.cofactors.is_some(), "{name}: the cofactor text");
        assert_eq!(outcome.certified, Some(true), "{name}: certified");
    }
    let mut params = reported(MonomialOrder::Grevlex);
    params.memory = Some(crate::os::resident() + (1 << 30));
    pair(
        "sparse5-p65521-1 under a ceiling",
        &random_sparse(5, 65521, 1),
        &params,
    );
}

/// The blocks of test 2, each of `BLOCK` bytes.
#[cfg(target_os = "linux")]
const BLOCKS: usize = 1024;

/// 64 KiB: below glibc's mmap threshold, so each block comes from glibc's
/// heap.
#[cfg(target_os = "linux")]
const BLOCK: usize = 64 << 10;

/// Test 2, freed-heap-text-peak.1 and .2: in a child process of this test
/// binary, which runs this test alone, 1,024 blocks of 64 KiB written and
/// every other one freed leave 32 MiB free below the heap's top. A call
/// with the trim off leaves the resident set where it was, and the same
/// call with the trim takes it at least 16 MiB lower: over 65521 in
/// grevlex, over ℚ, and in lex. The parent runs the child.
#[cfg(target_os = "linux")]
#[test]
fn the_free_memory_goes_back_before_the_texts() {
    if !measures_here("freed_heap_text_peak_tests::the_free_memory_goes_back_before_the_texts") {
        return;
    }
    for (name, text, order) in [
        (
            "sparse4 over 65521",
            random_sparse(4, 65521, 1),
            MonomialOrder::Grevlex,
        ),
        (
            "sparse4 over Q",
            random_sparse(4, 0, 1),
            MonomialOrder::Grevlex,
        ),
        (
            "dense33 over 7 in lex",
            dense(3, 3, 7, 4),
            MonomialOrder::Lex,
        ),
    ] {
        let params = reported(order);
        // A block of nonzero bytes is written whole, so its pages are
        // resident.
        let mut blocks: Vec<Vec<u8>> = (0..BLOCKS).map(|_| vec![1; BLOCK]).collect();
        for block in blocks.iter_mut().skip(1).step_by(2) {
            *block = Vec::new();
        }
        let without = call(&text, &params, true);
        let untrimmed = resident();
        let with = call(&text, &params, false);
        let trimmed = resident();
        assert_eq!(with.basis, without.basis, "{name}: the bases");
        assert!(
            trimmed + (16 << 20) <= untrimmed,
            "{name}: the resident set was {untrimmed} bytes after the call without the trim and {trimmed} after the call with it"
        );
        drop(std::hint::black_box(blocks));
    }
}

/// Test 3, the edge case of a deadline: under deadlines of 0 to 32 ms,
/// with the trim and without, a call returns its full basis or stops at
/// the deadline with no basis.
#[test]
fn a_stop_at_the_texts() {
    for p in [65521, 0] {
        let text = if p == 0 {
            random_sparse(4, p, 1)
        } else {
            random_sparse(5, p, 1)
        };
        let full = call(&text, &reported(MonomialOrder::Grevlex), false)
            .basis
            .unwrap();
        for untrimmed in [false, true] {
            for ms in [0, 1, 2, 4, 8, 16, 32] {
                let mut params = reported(MonomialOrder::Grevlex);
                params.deadline = Some(Duration::from_millis(ms));
                let outcome = call(&text, &params, untrimmed);
                let name = format!("{ms} ms over {p}, untrimmed {untrimmed}");
                match outcome.stopped {
                    Some(Stop::Deadline) => assert!(outcome.basis.is_none(), "{name}: {outcome:?}"),
                    None => assert_eq!(outcome.basis.unwrap(), full, "{name}"),
                    Some(Stop::Cancel | Stop::Memory) => panic!("{name}: {outcome:?}"),
                }
            }
        }
    }
}
