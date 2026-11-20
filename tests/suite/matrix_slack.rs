//! matrix-slack, the item's test code: the edge cases of `OBJECTIVES.md`,
//! whose matrices take no reservation; the checker in the cheap mode on a
//! dense system; and a memory ceiling
//! the reservations' address space passes and their written pages do not
//! (plan.md, _Test code_, 6, 8, and 9).

use std::num::NonZeroU32;
#[cfg(target_os = "linux")]
use std::process::Command;

use grobx::CheckerMode;
use grobx_checker::Mode;

#[cfg(target_os = "linux")]
use crate::common::resident;
use crate::common::{EDGE_BASES, P, assert_accepted, dense, params, run, with_cofactors};
use crate::orders::assert_canonical;
use crate::q::accepted_q;

/// Test 6: the edge cases in grevlex, over 65521 with signatures on and
/// off and over ℚ with them on: each basis canonical and accepted in the exact mode, and over
/// 65521 as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE_BASES {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut params = with_cofactors(0x51a0);
            params.signatures = signatures;
            let outcome = run(&text, &params);
            let basis = outcome.basis.as_deref().unwrap();
            let name = format!("{case} over {P}, signatures {signatures}");
            assert_canonical(&name, &text, basis, "grevlex");
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(basis, want, "{name}");
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}

/// Test 8: the oracle. A dense cubic system over GF(65521) on four variables with
/// the checker in the cheap mode at two points: certified.
#[test]
fn the_checker_certifies_in_the_cheap_mode() {
    let mut p = params(0x51a3);
    p.checker = Some(CheckerMode::Cheap {
        points: NonZeroU32::new(2).unwrap(),
    });
    let outcome = run(&dense(4, 3, P, 1), &p);
    assert_eq!(
        outcome.certified,
        Some(true),
        "cubic-4: {flaw:?}",
        flaw = outcome.flaw
    );
}

/// The variable that tells a child process to run the call.
#[cfg(target_os = "linux")]
const CHILD: &str = "GROBX_SLACK_CHILD";

/// Test 9: in a child process of the suite's binary, which runs this test
/// alone, a dense quartic system over GF(65521) on five variables, whose largest
/// matrix passes `RESERVE_PAST` entries, returns a basis under a memory
/// ceiling of the resident set before the call plus 24 MiB. A reservation
/// spans 32 MiB. The parent runs the child.
#[cfg(target_os = "linux")]
#[test]
fn a_ceiling_past_the_written_pages_holds() {
    if std::env::var_os(CHILD).is_some() {
        // test-mq-systems.8 design-checked -- the dense quartic system on five variables, whose largest matrix passes RESERVE_PAST under the test's seed where the map's dense cubic one on six variables stays below it (log 0084-1)
        let text = dense(5, 4, P, 2);
        let mut p = params(0x51a4);
        p.memory = Some(resident() + (24 << 20));
        let outcome = grobx::groebner(&text, &p).unwrap();
        assert_eq!(outcome.stopped, None, "quartic-5: stopped");
        assert!(outcome.basis.is_some(), "quartic-5: no basis");
        return;
    }
    let exe = std::env::current_exe().expect("the test binary knows its path");
    let out = Command::new(&exe)
        .args([
            "matrix_slack::a_ceiling_past_the_written_pages_holds",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap_or_else(|e| panic!("{exe}: {e}", exe = exe.display()));
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the child failed: {out:?}");
    assert!(said.contains("1 passed"), "the child ran no test: {said}");
}
