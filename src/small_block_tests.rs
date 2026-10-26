//! The item's crate test code (`small-block-time`): `Allocator` serves the
//! requests of at most `SMALL` bytes from mimalloc and the larger ones from
//! the system's malloc, at every alignment; a resize across `SMALL` moves a
//! block with its bytes; the crate's unit tests run under it; and the
//! budget's trim gives mimalloc's free memory back (small-block-time.1 to
//! .4, .6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::alloc::{GlobalAlloc, Layout};

use crate::blocks::mimalloc_holds;
use crate::child_tests;
use crate::os::resident;
use crate::{Allocator, SMALL};

/// The alignments tests 1 to 3 take a block at.
const ALIGNS: [usize; 6] = [1, 8, 16, 64, 512, 4096];

/// The byte at offset i of a block written under `salt`.
fn byte(i: usize, salt: u8) -> u8 {
    (i as u8).wrapping_mul(31).wrapping_add(salt)
}

/// Writes the block's first `len` bytes under `salt`.
///
/// # Safety
///
/// p is valid for `len` bytes.
unsafe fn write(p: *mut u8, len: usize, salt: u8) {
    for i in 0..len {
        // SAFETY: the caller's guarantee.
        unsafe { *p.add(i) = byte(i, salt) };
    }
}

/// Whether the block's first `len` bytes are those `write` wrote under
/// `salt`.
///
/// # Safety
///
/// p is valid for `len` bytes.
unsafe fn holds(p: *const u8, len: usize, salt: u8) -> bool {
    // SAFETY: the caller's guarantee.
    (0..len).all(|i| unsafe { *p.add(i) } == byte(i, salt))
}

fn layout(size: usize, align: usize) -> Layout {
    Layout::from_size_align(size, align).unwrap()
}

/// Test 1, small-block-time.1, .2: a block of at most `SMALL` bytes lies
/// in mimalloc's heap region and a larger one outside it, at every
/// alignment, each aligned, and each of its bytes written and read back.
#[test]
fn the_small_blocks_are_mimallocs() {
    for align in ALIGNS {
        for size in [1, 7, 8, 100, SMALL - 1, SMALL, SMALL + 1, 4096, 1 << 20] {
            let l = layout(size, align);
            // SAFETY: the layout's size is above zero.
            let p = unsafe { Allocator.alloc(l) };
            assert!(!p.is_null(), "size {size}, align {align}");
            assert_eq!(p as usize % align, 0, "size {size}, align {align}");
            assert_eq!(
                mimalloc_holds(p),
                size <= SMALL,
                "size {size}, align {align}"
            );
            // SAFETY: the block holds `size` bytes, and goes back at its
            // layout.
            unsafe {
                write(p, size, 3);
                assert!(holds(p, size, 3), "size {size}, align {align}");
                Allocator.dealloc(p, l);
            }
        }
    }
}

/// Test 2, small-block-time.1: a zeroed block is zero on both sides of
/// `SMALL`, where a block of its layout was written and freed before it.
#[test]
fn a_zeroed_block_is_zero_on_both_sides() {
    for align in ALIGNS {
        for size in [1, 64, SMALL, SMALL + 1, 1 << 16] {
            let l = layout(size, align);
            // SAFETY: the layout's size is above zero; each block holds
            // `size` bytes and goes back at its layout.
            unsafe {
                let dirty = Allocator.alloc(l);
                assert!(!dirty.is_null());
                write(dirty, size, 0x55);
                Allocator.dealloc(dirty, l);
                let p = Allocator.alloc_zeroed(l);
                assert!(!p.is_null(), "size {size}, align {align}");
                assert_eq!(p as usize % align, 0);
                assert_eq!(mimalloc_holds(p), size <= SMALL);
                assert!(
                    (0..size).all(|i| *p.add(i) == 0),
                    "size {size}, align {align}"
                );
                Allocator.dealloc(p, l);
            }
        }
    }
}

/// Test 3, small-block-time.3: a block resized up and down across
/// `SMALL`, and within each side, keeps the bytes of the shorter of its
/// two lengths, at every alignment, and lies with the allocator its new
/// size names.
#[test]
fn a_resize_across_small_moves_the_block_with_its_bytes() {
    let sizes = [
        5000,
        600,
        SMALL,
        SMALL + 1,
        3000,
        1 << 20,
        16,
        SMALL + 1,
        SMALL,
        1,
    ];
    for align in ALIGNS {
        let mut l = layout(100, align);
        // SAFETY: the layout's size is above zero.
        let mut p = unsafe { Allocator.alloc(l) };
        assert!(!p.is_null());
        // SAFETY: the block holds 100 bytes.
        unsafe { write(p, 100, 1) };
        let mut salt = 1;
        for next in sizes {
            // SAFETY: p was served at l, and next is above zero.
            let q = unsafe { Allocator.realloc(p, l, next) };
            assert!(
                !q.is_null(),
                "{size} to {next}, align {align}",
                size = l.size()
            );
            assert_eq!(q as usize % align, 0);
            let kept = l.size().min(next);
            // SAFETY: the resized block holds `next` bytes, at least
            // `kept`.
            assert!(
                unsafe { holds(q, kept, salt) },
                "{size} to {next}, align {align}",
                size = l.size()
            );
            assert_eq!(
                mimalloc_holds(q),
                next <= SMALL,
                "{size} to {next}, align {align}",
                size = l.size()
            );
            salt = salt.wrapping_add(1);
            // SAFETY: as above.
            unsafe { write(q, next, salt) };
            (p, l) = (q, layout(next, align));
        }
        // SAFETY: p was last resized to l.
        unsafe { Allocator.dealloc(p, l) };
    }
}

/// Test 4, small-block-time.4: the crate's unit tests run under
/// `Allocator`. A vector of a few bytes lies in mimalloc's heap region and
/// one of 1 MiB outside it. A vector grown entry by entry from empty to
/// 1 MiB, past `SMALL`, and shrunk back below it keeps every entry.
#[test]
fn the_unit_tests_run_under_the_allocator() {
    let small = std::hint::black_box(vec![1u8; 16]);
    assert!(mimalloc_holds(small.as_ptr()));
    let large = std::hint::black_box(vec![1u8; 1 << 20]);
    assert!(!mimalloc_holds(large.as_ptr()));
    let entry = |k: usize| (k as u32).wrapping_mul(2_654_435_761);
    let mut v: Vec<u32> = Vec::new();
    for k in 0..1 << 18 {
        v.push(entry(k));
    }
    assert!(!mimalloc_holds(v.as_ptr()));
    assert!(v.iter().enumerate().all(|(k, &x)| x == entry(k)));
    v.truncate(100);
    v.shrink_to_fit();
    assert!(mimalloc_holds(v.as_ptr()));
    assert!(v.iter().enumerate().all(|(k, &x)| x == entry(k)));
}

/// Test 5, small-block-time.6: in a child process of this test binary,
/// which runs this test alone, 65,536 blocks of 1000 bytes, 62.5 MiB,
/// written and freed, leave the resident set at least 32 MiB higher than the budget's
/// trim does. The parent runs the child.
#[test]
fn the_trim_gives_mimallocs_free_memory_back() {
    if child_tests::running() {
        let blocks: Vec<Vec<u8>> = (0..1 << 16).map(|k| vec![k as u8; 1000]).collect();
        assert!(blocks.iter().all(|b| mimalloc_holds(b.as_ptr())));
        drop(std::hint::black_box(blocks));
        let freed = resident();
        crate::budget::trim();
        let trimmed = resident();
        assert!(
            trimmed + (32 << 20) <= freed,
            "the resident set was {freed} bytes after the free and {trimmed} after the trim"
        );
        return;
    }
    child_tests::run("small_block_tests::the_trim_gives_mimallocs_free_memory_back");
}
