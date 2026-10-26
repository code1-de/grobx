//! page-fault-time, the item's test code for the C interface (plan.md,
//! _Test code_, test 6): a C program whose main thread calls the library
//! sees its main heap grow by `grobx::TOP_PAD` after the call and by
//! glibc's default pad before it, and a block of 1 MiB come from the heap
//! after the call where it took a mapping before.

use std::collections::HashMap;
use std::fs;

use grobx::{MonomialOrder, TOP_PAD};

use crate::common::{Args, program, run, scratch, write_instance};
use crate::heap::{HEAP_EXCESS, MAPPED_EXCESS};
use crate::systems::dense;

/// Test 6, page-fault-time.1 to .5: `heap.c` over GF(65521) and over ℚ.
#[test]
fn the_main_heap_grows_by_the_pad() {
    let pad = u64::try_from(TOP_PAD).unwrap();
    for (stem, text) in [("p65521", dense(3, 3, 65521, 1)), ("q", dense(3, 3, 0, 1))] {
        let dir = scratch(&format!("page_fault_time-{stem}"));
        let out = dir.join("out");
        fs::create_dir_all(&out).unwrap();
        let exe = program("heap.c");
        let instance = write_instance(&dir, &text);
        let ran = run(&exe, &[instance.display().to_string(), out.display().to_string()]);
        let lines: HashMap<String, u64> = String::from_utf8_lossy(&ran.stdout)
            .lines()
            .map(|l| {
                let (k, v) = l.split_once(' ').unwrap();
                (k.to_string(), v.parse().unwrap())
            })
            .collect();
        let at = |k: &str| *lines.get(k).unwrap_or_else(|| panic!("{stem}: no line {k}"));
        assert_eq!(at("status"), 0, "{stem}: the call's status");
        let (before, after) = (at("before"), at("after"));
        assert!(before < pad, "{stem}: the heap grew by {before} before the call");
        assert!(after >= pad, "{stem}: the heap grew by {after} after the call");
        assert!(
            at("excess-before-1m") >= MAPPED_EXCESS,
            "{stem}: a 1 MiB block came from the heap before the call"
        );
        assert!(at("excess-1m") < HEAP_EXCESS, "{stem}: a 1 MiB block took a mapping after the call");
        assert!(
            at("excess-64m") >= MAPPED_EXCESS,
            "{stem}: a 64 MiB block came from the heap after the call"
        );
        let basis = fs::read_to_string(out.join("basis")).unwrap();
        let args = Args::new(MonomialOrder::Grevlex, 1);
        let want = grobx::groebner(&text, &args.params()).unwrap().basis;
        assert_eq!(Some(basis), want, "{stem}");
    }
}
