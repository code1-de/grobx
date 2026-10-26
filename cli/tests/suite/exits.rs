//! The item's test code, cli: the exit statuses and what goes to standard
//! error (cli.6, .7, .13 to .17).

use std::ffi::OsString;
use std::fs;
use std::time::{Duration, Instant};

use grobx::{CheckerMode, MonomialOrder, Outcome, Params, Stats, Stop, Threads};
use grobx_cli::{DONE, ERROR, REJECTED, STOPPED, Usage, diagnostic, status};

use crate::common::{crate_call, grobx, scratch, write_instance};
use crate::systems::dense;

/// The arguments `args` with the paths `files` after them.
fn with(args: &[&str], files: &[&std::path::Path]) -> Vec<OsString> {
    args.iter()
        .map(OsString::from)
        .chain(files.iter().map(|f| f.as_os_str().to_owned()))
        .collect()
}

/// The run is on `dense(8, 3, 65521, 1)`, which runs for seconds without
/// the bound.
#[test]
fn a_bound_that_stops_the_run_exits_1() {
    let mid = write_instance(&scratch("bound"), "long", &dense(8, 3, 65521, 1));
    for (switch, value, why) in [("--deadline", "0", "deadline"), ("--memory", "1", "memory")] {
        let ran = grobx(&with(&["--seed", "1", switch, value], &[&mid]), "");
        assert_eq!(ran.code(), 1, "{switch}: {ran:?}");
        assert_eq!(ran.stdout, "", "{switch}");
        assert_eq!(ran.stderr, format!("grobx: stopped: {why}\n"), "{switch}");

        let dir = scratch(&format!("bound-{why}"));
        let (basis, cofactors) = (dir.join("basis.txt"), dir.join("cofactors.txt"));
        let mut args = with(&["--seed", "1", switch, value, "--checker", "exact"], &[]);
        args.extend(["--output".into(), basis.clone().into_os_string()]);
        args.extend(["--cofactors".into(), cofactors.clone().into_os_string()]);
        args.push(mid.clone().into_os_string());
        let ran = grobx(&args, "");
        assert_eq!(ran.code(), 1, "{switch} with files: {ran:?}");
        assert_eq!(ran.stdout, "", "{switch} with files");
        assert_eq!(
            ran.stderr,
            format!("grobx: stopped: {why}\n"),
            "{switch} with files"
        );
        assert_eq!(
            fs::read_to_string(&basis).unwrap(),
            "",
            "{switch}: the output file"
        );
        assert_eq!(
            fs::read_to_string(&cofactors).unwrap(),
            "",
            "{switch}: the cofactor file"
        );
    }
}

#[test]
fn usage_errors_exit_2() {
    let tiny = write_instance(&scratch("usage"), "tiny", &dense(2, 3, 65521, 1));
    let cases: &[&[&str]] = &[
        &["--bogus", "--seed", "1"],
        &["-x", "--seed", "1"],
        &["-t", "1", "--seed", "1"],
        &["--seed"],
        &["--seed", "1", "--order"],
        &["--threads", "x", "--seed", "1"],
        &["--seed", "-1"],
        &["--seed", "+1"],
        &["--seed", "18446744073709551616"],
        &["--seed", ""],
        &["--deadline", "1.5", "--seed", "1"],
        &["--memory", "1G", "--seed", "1"],
        &["--checker", "0", "--seed", "1"],
        &["--checker", "sometimes", "--seed", "1"],
        &["--seed", "1", "--seed", "2"],
        &["--seed=1", "--order=lex", "--order", "grevlex"],
        &["--seed", "1", "--order", "deglex"],
        &["--seed", "1", "--order=block:0"],
        &["--help=x"],
        &["--version=x"],
        &["--bogus", "--help"],
    ];
    for args in cases {
        // The operand comes first, so that a switch can stand last.
        let mut operand_first = vec![tiny.clone().into_os_string()];
        operand_first.extend(args.iter().map(OsString::from));
        let ran = grobx(&operand_first, "");
        assert_eq!(ran.code(), 2, "{args:?}: {ran:?}");
        assert_eq!(ran.stdout, "", "{args:?}");
        assert!(ran.stderr.starts_with("grobx: "), "{args:?}: {ran:?}");
        assert!(
            ran.stderr.ends_with("grobx: --help lists the switches\n"),
            "{args:?}: {ran:?}"
        );
        assert_eq!(ran.stderr.lines().count(), 2, "{args:?}: {ran:?}");
    }
    let ran = grobx(&with(&["--seed", "1"], &[&tiny, &tiny]), "");
    assert_eq!(ran.code(), 2, "two operands: {ran:?}");
    assert_eq!(ran.stdout, "");
    assert!(
        ran.stderr.ends_with("grobx: --help lists the switches\n"),
        "{ran:?}"
    );
}

fn usage(args: &[&str]) -> String {
    match grobx_cli::parse(args.iter().map(OsString::from)) {
        Err(Usage(m)) => m,
        Ok(c) => panic!("{args:?} reads as {c:?}"),
    }
}

#[test]
fn the_usage_messages() {
    assert_eq!(usage(&["--bogus"]), "unknown switch --bogus");
    assert_eq!(usage(&["-t", "1"]), "unknown switch -t");
    assert_eq!(usage(&["--help=x"]), "--help takes no value");
    assert_eq!(usage(&["--seed"]), "--seed takes a value");
    assert_eq!(usage(&["--seed", "1", "--seed", "2"]), "--seed given twice");
    assert_eq!(
        usage(&["--seed", "x"]),
        "--seed \"x\": not a decimal number from 0 to 18446744073709551615"
    );
    assert_eq!(
        usage(&["--threads", "-1", "--seed", "1"]),
        format!(
            "--threads \"-1\": not a decimal number from 0 to {max}",
            max = usize::MAX
        )
    );
    for value in ["1.5", "1.5s", "10ms", "s", "10S", "-1s", "5124095576031h"] {
        assert_eq!(
            usage(&["--deadline", value]),
            format!(
                "--deadline {value:?}: a decimal number of milliseconds, or one followed by s, m, or h, \
                 up to {max} milliseconds",
                max = u64::MAX
            )
        );
    }
    assert_eq!(
        usage(&["--checker", "sometimes"]),
        "--checker \"sometimes\": exact, or a decimal number of points from 1 to 4294967295"
    );
    assert_eq!(
        usage(&["--checker", "0"]),
        "--checker \"0\": exact, or a decimal number of points from 1 to 4294967295"
    );
    assert_eq!(
        usage(&["--order", "deglex"]),
        "--order \"deglex\": grevlex, lex, or block:k1,...,km"
    );
    assert_eq!(
        usage(&["a.ms", "b.ms", "--seed", "1"]),
        "more than one instance: a.ms and b.ms"
    );
}

#[test]
fn an_unreadable_instance_exits_2() {
    let dir = scratch("unreadable");
    let missing = dir.join("missing.ms");
    let binary = dir.join("binary.ms");
    fs::write(&binary, [0xff, 0xfe]).unwrap();
    for file in [&missing, &dir, &binary] {
        let ran = grobx(&with(&["--seed", "1"], &[file]), "");
        assert_eq!(ran.code(), 2, "{file}: {ran:?}", file = file.display());
        assert_eq!(ran.stdout, "", "{file}", file = file.display());
        let opening = format!("grobx: {file}: ", file = file.display());
        assert!(
            ran.stderr.starts_with(&opening),
            "{file}: {ran:?}",
            file = file.display()
        );
        assert_eq!(ran.stderr.lines().count(), 1, "{ran:?}");
    }
}

#[test]
fn an_instance_the_crate_refuses_exits_2() {
    let dir = scratch("refused");
    // test-mq-systems.10 design -- the order error of block:5 takes the dense cubic system on five variables, all of which block:5 puts in one block, not the map's on three (not measured)
    let five = dense(5, 3, 65521, 1);
    let cases = [
        ("", "grevlex"),
        ("x\n7\nx+y\n", "grevlex"),
        ("x\n4\nx\n", "grevlex"),
        ("x,y\n7\nx^2+,\ny\n", "grevlex"),
        (five.as_str(), "block:5"),
    ];
    for (i, (text, order)) in cases.iter().enumerate() {
        let named = order.parse().expect("a name of the family");
        let params = Params::new(named, Threads::Count(std::num::NonZeroUsize::MIN), 1);
        let e = grobx::groebner(text, &params).expect_err("the crate refuses the case");
        let file = dir.join(format!("case-{i}.ms"));
        fs::write(&file, text).unwrap();
        let ran = grobx(&["--seed", "1", "--order", order], text);
        assert_eq!(ran.code(), 2, "{text:?} {order}: {ran:?}");
        assert_eq!(ran.stdout, "");
        assert_eq!(ran.stderr, format!("grobx: standard input: {e}\n"));
        let ran = grobx(&with(&["--seed", "1", "--order", order], &[&file]), "");
        assert_eq!(ran.code(), 2, "{text:?} {order}: {ran:?}");
        assert_eq!(ran.stdout, "");
        assert_eq!(
            ran.stderr,
            format!("grobx: {file}: {e}\n", file = file.display())
        );
    }
}

#[test]
fn an_output_that_cannot_be_created_exits_2_before_the_call() {
    let dir = scratch("uncreatable");
    let nowhere = dir.join("missing").join("out.txt");
    // A long call: dense(8, 3, 65521, 1) runs for seconds.
    let long = write_instance(&dir, "long", &dense(8, 3, 65521, 1));
    for switch in ["--output", "--cofactors"] {
        let start = Instant::now();
        let ran = grobx(&with(&["--seed", "1", switch], &[&nowhere, &long]), "");
        let took = start.elapsed();
        assert_eq!(ran.code(), 2, "{switch}: {ran:?}");
        assert_eq!(ran.stdout, "");
        let opening = format!("grobx: {nowhere}: ", nowhere = nowhere.display());
        assert!(ran.stderr.starts_with(&opening), "{switch}: {ran:?}");
        assert!(
            took < Duration::from_secs(10),
            "{switch}: {took:?}, the call ran"
        );
    }
}

fn outcome(stopped: Option<Stop>, certified: Option<bool>, flaw: Option<&str>) -> Outcome {
    Outcome {
        basis: stopped.is_none().then(|| "[1]:\n".to_string()),
        stopped,
        cofactors: None,
        certified,
        flaw: flaw.map(str::to_string),
        stats: Stats::default(),
        report: None,
    }
}

#[test]
fn the_exit_status_reads_the_outcome() {
    let cases = [
        (outcome(None, None, None), DONE, None),
        (
            outcome(Some(Stop::Deadline), None, None),
            STOPPED,
            Some("stopped: deadline"),
        ),
        (
            outcome(Some(Stop::Memory), None, None),
            STOPPED,
            Some("stopped: memory"),
        ),
        (
            outcome(Some(Stop::Cancel), None, None),
            STOPPED,
            Some("stopped: cancel"),
        ),
        (
            outcome(None, Some(true), None),
            DONE,
            Some("checker: certified"),
        ),
        (
            outcome(None, Some(false), Some("f")),
            REJECTED,
            Some("checker: rejected: f"),
        ),
        (
            outcome(None, Some(false), None),
            REJECTED,
            Some("checker: rejected"),
        ),
    ];
    for (o, code, line) in cases {
        assert_eq!(status(&o), code, "{o:?}");
        assert_eq!(diagnostic(&o).as_deref(), line, "{o:?}");
    }
    assert_eq!((DONE, STOPPED, ERROR, REJECTED), (0, 1, 2, 3));
    // A call with the checker on that the engine finishes is certified.
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(std::num::NonZeroUsize::MIN),
        1,
    );
    p.checker = Some(CheckerMode::Exact);
    let o = crate_call("x,y\n7\nx^2+y,\nx*y-1\n", &p);
    assert_eq!(
        (status(&o), diagnostic(&o).as_deref()),
        (DONE, Some("checker: certified"))
    );
}
