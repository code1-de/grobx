//! c-interface, the item's test code for the calls: a C program calls
//! `grobx_groebner` over GF(p) and over ℚ with every parameter of
//! `OBJECTIVES.md`, _Call parameters_, and hands back what the crate's
//! entry point returns for the same arguments.

use std::num::NonZeroU32;
use std::process::Command;
use std::time::{Duration, Instant};

use grobx::CheckerMode;
use grobx::MonomialOrder::{Block, Grevlex, Lex};
use grobxc::*;

use crate::common::{Args, Raise, assert_matches, drive, program, scratch};
use crate::systems::{dense, random_square};

fn cheap(points: u32) -> CheckerMode {
    CheckerMode::Cheap {
        points: NonZeroU32::new(points).unwrap(),
    }
}

/// Six cases on generated systems through the C driver against the crate:
/// status, stopped, certified, the basis text, the cofactor text, and the
/// flaw; the digest by `grobx_digest`.
#[test]
fn every_parameter_matches_the_crate() {
    let cases: Vec<(&str, String, Args)> = vec![
        // Grevlex over GF(p).
        ("grevlex-p", dense(3, 3, 65521, 1), Args::new(Grevlex, 3)),
        // Lex over GF(p), every thread, the cheap checker, cofactors.
        (
            "lex-p",
            dense(3, 3, 65521, 2),
            Args {
                threads: 0,
                checker: Some(cheap(4)),
                cofactors: true,
                ..Args::new(Lex, 5)
            },
        ),
        // A block order, the exact checker.
        (
            "block-p",
            random_square(65521, 3),
            Args {
                checker: Some(CheckerMode::Exact),
                cofactors: true,
                ..Args::new(Block(vec![1]), 7)
            },
        ),
        // Over Q with every bound set and none reached.
        (
            "bounds-q",
            dense(2, 3, 0, 1),
            Args {
                deadline: Some(600_000),
                memory: Some(1 << 40),
                checker: Some(CheckerMode::Exact),
                cofactors: true,
                cancel: Raise::Never,
                ..Args::new(Grevlex, 9)
            },
        ),
        // Lex over Q, the largest seed.
        ("lex-q", dense(2, 3, 0, 2), Args::new(Lex, u64::MAX)),
        // Cofactors over Q without the checker.
        (
            "cofactors-q",
            random_square(0, 1),
            Args {
                cofactors: true,
                ..Args::new(Grevlex, 0)
            },
        ),
    ];
    for (i, (case, text, args)) in cases.iter().enumerate() {
        let name = format!("parameter-{i}-{case}");
        let got = assert_matches(&name, text, args);
        assert_eq!(got.status, GROBX_OK, "{name}");
        assert!(got.basis.is_some(), "{name}: no basis");
        assert_eq!(got.cofactors.is_some(), args.cofactors, "{name}");
        if args.checker.is_some() {
            assert_eq!(
                got.certified,
                GROBX_CERTIFIED,
                "{name}: {flaw:?}",
                flaw = got.flaw
            );
        }
    }
}

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, over GF(7) and
/// over Q, with the exact checker and cofactors, against the crate.
#[test]
fn the_edge_cases_match_the_crate() {
    let cases = [
        ("zero-ideal", "x,y\n7\n", Grevlex, Some("[]:\n")),
        ("f-is-zero", "x,y\n7\n0\n", Grevlex, Some("[]:\n")),
        ("unit-ideal", "x,y\n7\nx,\nx-1\n", Grevlex, Some("[1]:\n")),
        ("already-a-basis", "x,y\n7\nx,\ny\n", Grevlex, None),
        (
            "zero-generator",
            "x,y\n7\nx^2+y,\n0,\nx*y-1\n",
            Grevlex,
            None,
        ),
        (
            "eliminable",
            "t,x,y\n7\nx-t^2,\ny-t^3\n",
            Block(vec![1]),
            None,
        ),
    ];
    for (name, text, order, basis) in cases {
        for (field, text) in [
            ("p7", text.to_string()),
            ("q", text.replacen("\n7\n", "\n0\n", 1)),
        ] {
            let name = format!("edge-{name}-{field}");
            let args = Args {
                checker: Some(CheckerMode::Exact),
                cofactors: true,
                ..Args::new(order.clone(), 1)
            };
            let got = assert_matches(&name, &text, &args);
            assert_eq!(got.status, GROBX_OK, "{name}: {error:?}", error = got.error);
            assert_eq!(
                got.certified,
                GROBX_CERTIFIED,
                "{name}: {flaw:?}",
                flaw = got.flaw
            );
            if let Some(basis) = basis {
                assert_eq!(got.basis.as_deref(), Some(basis), "{name}");
            }
        }
    }
}

/// Every error of the crate a call reaches but `Memory`, `Size`, and
/// `Drops`: its status and the crate's message, and no other text.
#[test]
fn the_errors_of_the_crate_match() {
    // test-mq-systems.10 design -- the order error of block:5 takes the dense cubic system on five variables, all of which block:5 puts in one block, not the map's on three (not measured)
    let five = dense(5, 3, 65521, 1);
    let cases = [
        ("empty", "", Grevlex, GROBX_SYNTAX),
        ("unknown", "x\n7\nx+y\n", Grevlex, GROBX_UNKNOWN_VARIABLE),
        ("repeated", "x,x\n7\nx\n", Grevlex, GROBX_REPEATED_VARIABLE),
        ("characteristic", "x\n4\nx\n", Grevlex, GROBX_CHARACTERISTIC),
        ("overflow", "x\n7\nx^2147483648\n", Grevlex, GROBX_OVERFLOW),
        ("no-inverse", "x\n7\n1/7*x\n", Grevlex, GROBX_NO_INVERSE),
        ("block-5", five.as_str(), Block(vec![5]), GROBX_ORDER),
    ];
    for (name, text, order, status) in cases {
        let name = format!("error-{name}");
        let got = assert_matches(&name, text, &Args::new(order, 1));
        assert_eq!(got.status, status, "{name}: {error:?}", error = got.error);
        assert!(got.error.is_some(), "{name}: a message");
        assert_eq!(
            (&got.basis, &got.cofactors, &got.flaw),
            (&None, &None, &None),
            "{name}"
        );
    }
}

/// A deadline of 0 and a memory ceiling of 1 byte stop the call, and so
/// does a flag raised before it: status `GROBX_OK`, the bound in
/// `stopped`, and no basis, as the crate gives. The calls run on
/// `dense(8, 3, 65521, 1)` and `dense(5, 3, 0, 1)`, which run for seconds
/// without the bound.
#[test]
fn a_bound_stops_the_call() {
    let long = dense(8, 3, 65521, 1);
    let cases = [
        (
            "deadline",
            long.clone(),
            Args {
                deadline: Some(0),
                cofactors: true,
                ..Args::new(Grevlex, 1)
            },
            GROBX_STOP_DEADLINE,
        ),
        (
            "memory",
            long.clone(),
            Args {
                memory: Some(1),
                cofactors: true,
                ..Args::new(Grevlex, 1)
            },
            GROBX_STOP_MEMORY,
        ),
        (
            "cancel",
            long,
            Args {
                cancel: Raise::Before,
                ..Args::new(Grevlex, 1)
            },
            GROBX_STOP_CANCEL,
        ),
        (
            "cancel-q",
            dense(5, 3, 0, 1),
            Args {
                cancel: Raise::Before,
                ..Args::new(Grevlex, 1)
            },
            GROBX_STOP_CANCEL,
        ),
    ];
    for (name, text, args, stopped) in cases {
        let name = format!("bound-{name}");
        let got = assert_matches(&name, &text, &args);
        assert_eq!(got.status, GROBX_OK, "{name}");
        assert_eq!(got.stopped, stopped, "{name}");
        assert_eq!((&got.basis, &got.cofactors), (&None, &None), "{name}");
    }
}

/// A flag that a thread of the C program raises 300 ms into the call stops
/// `dense(8, 3, 65521, 1)`, which runs for seconds, at its next check.
#[test]
fn a_flag_raised_by_another_thread_stops_the_call() {
    let args = Args {
        cancel: Raise::After(300),
        ..Args::new(Grevlex, 1)
    };
    let start = Instant::now();
    let (got, digest, _) = drive(&scratch("cancel-thread"), &dense(8, 3, 65521, 1), &args);
    let took = start.elapsed();
    assert_eq!(got.status, GROBX_OK, "{error:?}", error = got.error);
    assert_eq!(got.stopped, GROBX_STOP_CANCEL);
    assert_eq!((&got.basis, &digest), (&None, &None));
    assert!(took < Duration::from_secs(60), "{took:?}");
}

/// The driver under valgrind's memcheck, which frees each result twice and
/// frees NULL: no leak, no invalid free, on a result with a basis and
/// cofactors, on an error's, and on a stopped call's.
#[test]
#[ignore = "calls valgrind"]
fn the_result_frees_what_it_holds() {
    let driver = program("driver.c");
    let cases = [
        (
            "basis",
            dense(2, 3, 65521, 1),
            Args {
                checker: Some(CheckerMode::Exact),
                cofactors: true,
                ..Args::new(Grevlex, 1)
            },
        ),
        (
            "error",
            // test-mq-systems.10 design -- the order error of block:5 takes the dense cubic system on five variables, all of which block:5 puts in one block, not the map's on three (not measured)
            dense(5, 3, 65521, 1),
            Args::new(Block(vec![5]), 1),
        ),
        (
            "stopped",
            dense(8, 3, 65521, 1),
            Args {
                deadline: Some(0),
                ..Args::new(Grevlex, 1)
            },
        ),
    ];
    for (name, text, args) in cases {
        let dir = scratch(&format!("valgrind-{name}"));
        let file = dir.join("instance.ms");
        std::fs::write(&file, &text).unwrap();
        let out_dir = dir.join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        let out = Command::new("valgrind")
            .args([
                "--leak-check=full",
                "--errors-for-leak-kinds=definite,indirect",
                "--error-exitcode=7",
                "-q",
            ])
            .arg(&driver)
            .args(args.argv(&file, &out_dir))
            .output()
            .unwrap_or_else(|e| panic!("valgrind does not run ({e}): the test code needs it"));
        assert!(
            out.status.success(),
            "{name}: {status}\n{stderr}",
            status = out.status,
            stderr = String::from_utf8_lossy(&out.stderr)
        );
    }
}
