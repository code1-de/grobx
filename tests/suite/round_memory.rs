//! round-memory-time, the item's test code: the edge cases of
//! `OBJECTIVES.md` in grevlex, lex, and a block order; the same call twice
//! over GF(65521) and ℚ; the checker on a dense system; and ceilings near
//! the resident set (plan.md, _Test code_, 6 to 9).

use std::num::NonZeroU32;
#[cfg(target_os = "linux")]
use std::process::Command;

use grobx::CheckerMode;
#[cfg(target_os = "linux")]
use grobx::Stop;
use grobx_checker::Mode;

#[cfg(target_os = "linux")]
use crate::common::resident;
use crate::common::{
    EDGE_BASES, P, assert_accepted, dense, params, random_square, run, with_cofactors,
};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::accepted_q;

/// Test 6: the edge cases in grevlex over 65521 with signatures on and
/// off and over ℚ, and in lex and `block:1` over 65521, whose plain runs
/// back-substitute their rows where the lex call does not go through
/// grevlex and FGLM: each basis canonical and accepted in the
/// exact mode, and in grevlex over 65521 as written.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE_BASES {
        let text = format!("x,y,z\n{P}\n{generators}\n");
        for signatures in [true, false] {
            let mut params = with_cofactors(0x0e10);
            params.signatures = signatures;
            let outcome = run(&text, &params);
            let basis = outcome.basis.as_deref().unwrap();
            let name = format!("{case} over {P}, signatures {signatures}");
            assert_canonical(&name, &text, basis, "grevlex");
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(basis, want, "{name}");
        }
        for order in ["lex", "block:1"] {
            accepted_in(&format!("{case} over {P}"), &text, order, Mode::Exact);
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        accepted_q(&format!("{case} over ℚ"), &text, "grevlex");
    }
}

/// Test 7: dense cubic and random square systems over GF(65521) with signatures
/// on and off, and over ℚ. The same call twice in one process returns the
/// same basis and the same stats, and over GF(65521) signatures on and off
/// give one basis.
#[test]
fn the_same_call_twice_agrees() {
    for text in [dense(4, 3, P, 1), dense(4, 3, P, 2), random_square(P, 3)] {
        let mut bases = Vec::new();
        for signatures in [true, false] {
            let mut p = params(0x0e11);
            p.signatures = signatures;
            let first = run(&text, &p);
            let second = run(&text, &p);
            let case = format!("signatures {signatures}\n{text}");
            assert_eq!(first.basis, second.basis, "{case}");
            assert_eq!(first.stats, second.stats, "{case}");
            bases.push(first.basis);
        }
        assert_eq!(bases[0], bases[1], "signatures on and off\n{text}");
    }
    for text in [dense(3, 3, 0, 1), random_square(0, 3)] {
        let first = run(&text, &params(0x0e12));
        let second = run(&text, &params(0x0e12));
        assert_eq!(first.basis, second.basis, "{text}");
        assert_eq!(first.stats, second.stats, "{text}");
    }
}

/// Test 8: the oracle. A dense cubic system over GF(65521) with the checker in
/// the cheap mode at two points: certified.
#[test]
fn the_checker_certifies_a_run() {
    let mut p = params(0x0e13);
    p.checker = Some(CheckerMode::Cheap {
        points: NonZeroU32::new(2).unwrap(),
    });
    let outcome = run(&dense(4, 3, P, 1), &p);
    assert_eq!(
        outcome.certified,
        Some(true),
        "{flaw:?}",
        flaw = outcome.flaw
    );
}

/// The variable that tells a child process to run the calls.
#[cfg(target_os = "linux")]
const CHILD: &str = "GROBX_ROUND_MEMORY_CHILD";

/// Test 9: in a child process of the suite's binary, which runs this test
/// alone, a dense cubic system over GF(65521) under ceilings of the resident set
/// before the call plus 4, 8, and 16 MiB, with signatures on and off: each
/// call returns the basis of the call without a ceiling or stops on the
/// memory bound, and none returns another basis. The parent runs the
/// child.
#[cfg(target_os = "linux")]
#[test]
fn ceilings_near_the_resident_set() {
    if std::env::var_os(CHILD).is_some() {
        let text = dense(6, 3, P, 1);
        let free = run(&text, &params(0x0e14)).basis;
        for mib in [4u64, 8, 16] {
            for signatures in [true, false] {
                let mut p = params(0x0e14);
                p.signatures = signatures;
                p.memory = Some(resident() + (mib << 20));
                let outcome = grobx::groebner(&text, &p).unwrap();
                let case = format!("under {mib} MiB, signatures {signatures}");
                match outcome.basis {
                    Some(_) => assert_eq!(outcome.basis, free, "{case}"),
                    None => assert_eq!(outcome.stopped, Some(Stop::Memory), "{case}"),
                }
            }
        }
        return;
    }
    let exe = std::env::current_exe().expect("the test binary knows its path");
    let out = Command::new(&exe)
        .args([
            "round_memory::ceilings_near_the_resident_set",
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
