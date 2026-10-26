//! small-block-time, the item's test code for the C interface (plan.md,
//! _Test code_, test 11): a C program that calls the library runs under
//! `grobx::Allocator`, mimalloc serving the library's small blocks, and is
//! handed back the basis the crate's call gives.

use std::fs;
use std::process::Command;

use grobx::MonomialOrder;

use crate::common::{Args, program, scratch, write_instance};
use crate::systems::dense;

/// Test 11, small-block-time.4: the C driver with `MIMALLOC_VERBOSE=1`:
/// mimalloc reports on standard error the memory it reserves to serve a
/// block, and the basis the driver wrote is the crate's for the same seed,
/// over GF(65521) and over ℚ.
#[test]
fn the_library_runs_under_the_allocator() {
    for (stem, text) in [("p65521", dense(4, 3, 65521, 1)), ("q", dense(3, 3, 0, 1))] {
        let dir = scratch(&format!("small_block_time-{stem}"));
        let out = dir.join("out");
        fs::create_dir_all(&out).unwrap();
        let exe = program("driver.c");
        let args = Args::new(MonomialOrder::Grevlex, 3);
        let ran = Command::new(&exe)
            .args(args.argv(&write_instance(&dir, &text), &out))
            .env("MIMALLOC_VERBOSE", "1")
            .output()
            .unwrap_or_else(|e| panic!("{exe}: {e}", exe = exe.display()));
        let stderr = String::from_utf8_lossy(&ran.stderr);
        assert!(ran.status.success(), "{stem}: {ran:?}");
        assert!(
            stderr.lines().any(|l| l.starts_with("mimalloc: reserved")),
            "{stem}: mimalloc served no block:\n{stderr}"
        );
        let basis = fs::read_to_string(out.join("basis")).unwrap();
        let want = grobx::groebner(&text, &args.params()).unwrap().basis;
        assert_eq!(Some(basis), want, "{stem}");
    }
}
