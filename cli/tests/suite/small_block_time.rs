//! small-block-time, the item's test code for the CLI (plan.md, _Test
//! code_, test 10): the binary runs under `grobx::Allocator`, mimalloc
//! serving its small blocks, and writes the basis the crate's call gives.

use std::time::Duration;

use crate::common::{command, crate_call, finish, scratch, write_instance};
use crate::systems::dense;

/// Test 10, small-block-time.4: with `MIMALLOC_VERBOSE=1` mimalloc reports
/// on standard error the memory it reserves to serve a block, and the
/// basis on standard output is the crate's for the same seed, over
/// GF(65521) and over ℚ.
#[test]
fn the_binary_runs_under_the_allocator() {
    let dir = scratch("small-block-time");
    for (stem, text) in [("p65521", dense(4, 3, 65521, 1)), ("q", dense(3, 3, 0, 1))] {
        let file = write_instance(&dir, stem, &text);
        let mut cmd = command(&["--seed".as_ref(), "3".as_ref(), file.as_os_str()]);
        cmd.env("MIMALLOC_VERBOSE", "1");
        let ran = finish(
            cmd.spawn().expect("the binary starts"),
            Duration::from_secs(120),
        );
        assert_eq!(ran.code(), 0, "{stem}: {ran:?}");
        assert!(
            ran.stderr
                .lines()
                .any(|l| l.starts_with("mimalloc: reserved")),
            "{stem}: mimalloc served no block:\n{stderr}",
            stderr = ran.stderr
        );
        let params = grobx::Params::new(
            grobx::MonomialOrder::Grevlex,
            grobx::Threads::Count(std::num::NonZeroUsize::MIN),
            3,
        );
        assert_eq!(
            Some(ran.stdout.as_str()),
            crate_call(&text, &params).basis.as_deref(),
            "{stem}"
        );
    }
}
