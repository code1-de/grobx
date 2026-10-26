//! The item's test code, cli: every switch but `--report`, which
//! `call_report.rs` tests, on an instance from a file and
//! from standard input, against the crate's entry point (cli.1, .3 to .5,
//! .7 to .13).

use std::ffi::OsString;
use std::fs;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::time::Duration;

use grobx::{CheckerMode, MonomialOrder, Params, Threads};
use grobx_cli::{Command, Input, Run};

use crate::common::{crate_call, grobx, grobx_in, scratch, verdict, write_instance};
use crate::systems::{dense, random_square};

/// One case of `every_switch_matches_the_crate`: its name, its instance
/// text, its arguments but the paths of its files, the files it names, and
/// the parameters the crate takes for them.
struct Case {
    name: &'static str,
    text: String,
    args: &'static [&'static str],
    cofactors: Option<&'static str>,
    output: Option<&'static str>,
    params: Params,
}

fn params(order: &str, threads: usize, seed: u64) -> Params {
    let threads = match NonZeroUsize::new(threads) {
        None => Threads::All,
        Some(n) => Threads::Count(n),
    };
    Params::new(order.parse().expect("a name of the family"), threads, seed)
}

fn cases() -> Vec<Case> {
    let mut block = params("block:1", 1, 7);
    block.checker = Some(CheckerMode::Cheap {
        points: NonZeroU32::new(4).unwrap(),
    });
    block.cofactors = true;
    let mut bounded = params("grevlex", 1, 9);
    bounded.checker = Some(CheckerMode::Exact);
    bounded.cofactors = true;
    bounded.deadline = Some(Duration::from_millis(600_000));
    bounded.memory = Some(1 << 40);
    let mut cofactors = params("grevlex", 1, 0);
    cofactors.cofactors = true;
    vec![
        // The defaults, grevlex and one thread, over GF(p).
        Case {
            name: "grevlex-p",
            text: dense(3, 3, 65521, 1),
            args: &["--seed", "3"],
            cofactors: None,
            output: None,
            params: params("grevlex", 1, 3),
        },
        // Lex over GF(p), all threads.
        Case {
            name: "lex-p",
            text: dense(3, 3, 65521, 2),
            args: &["--order", "lex", "--threads", "0", "--seed", "5"],
            cofactors: None,
            output: None,
            params: params("lex", 0, 5),
        },
        // A block order, the cheap checker, every value after `=`.
        Case {
            name: "block-p",
            text: random_square(65521, 3),
            args: &["--order=block:1", "--seed=7", "--checker=4"],
            cofactors: Some("--cofactors="),
            output: None,
            params: block,
        },
        // Over Q with every bound set and none reached.
        Case {
            name: "bounds-q",
            text: dense(2, 3, 0, 1),
            args: &[
                "--seed",
                "9",
                "--checker",
                "exact",
                "--deadline",
                "600000",
                "--memory",
                "1099511627776",
            ],
            cofactors: Some("--cofactors"),
            output: Some("--output"),
            params: bounded,
        },
        // Lex over Q, the largest seed.
        Case {
            name: "lex-q",
            text: dense(2, 3, 0, 2),
            args: &["--order", "lex", "--seed", "18446744073709551615"],
            cofactors: None,
            output: None,
            params: params("lex", 1, u64::MAX),
        },
        // Cofactors over Q without the checker.
        Case {
            name: "cofactors-q",
            text: random_square(0, 1),
            args: &["--seed", "0"],
            cofactors: Some("--cofactors"),
            output: None,
            params: cofactors,
        },
    ]
}

/// The arguments of `case` with its files in `dir`, and `operand` last
/// where it is given.
fn arguments(case: &Case, dir: &std::path::Path, operand: Option<OsString>) -> Vec<OsString> {
    let mut args: Vec<OsString> = case.args.iter().map(OsString::from).collect();
    for (switch, name) in [
        (case.cofactors, "cofactors.txt"),
        (case.output, "basis.txt"),
    ] {
        let Some(switch) = switch else { continue };
        let file = dir.join(name);
        if let Some(name) = switch.strip_suffix('=') {
            let mut arg = OsString::from(format!("{name}="));
            arg.push(file.as_os_str());
            args.push(arg);
        } else {
            args.push(switch.into());
            args.push(file.into_os_string());
        }
    }
    args.extend(operand);
    args
}

#[test]
fn every_switch_matches_the_crate() {
    for (i, case) in cases().iter().enumerate() {
        let text = case.text.as_str();
        let expected = crate_call(text, &case.params);
        let basis = expected.basis.clone().expect("the crate's call finishes");
        assert_eq!(expected.cofactors.is_some(), case.cofactors.is_some());
        for way in ["path", "stdin", "dash"] {
            let dir = scratch(&format!("every-switch-{i}-{way}"));
            let (operand, stdin) = match way {
                "path" => (
                    Some(write_instance(&dir, "instance", text).into_os_string()),
                    "",
                ),
                "stdin" => (None, text),
                _ => (Some("-".into()), text),
            };
            let ran = grobx(&arguments(case, &dir, operand), stdin);
            let what = format!(
                "{name} {args:?} by {way}",
                name = case.name,
                args = case.args
            );
            assert_eq!(ran.code(), 0, "{what}: {ran:?}");
            assert_eq!(ran.stderr, verdict(&expected), "{what}");
            match case.output {
                Some(_) => {
                    assert_eq!(ran.stdout, "", "{what}");
                    assert_eq!(
                        fs::read_to_string(dir.join("basis.txt")).unwrap(),
                        basis,
                        "{what}"
                    );
                }
                None => assert_eq!(ran.stdout, basis, "{what}"),
            }
            if let Some(cofactors) = &expected.cofactors {
                let written = fs::read_to_string(dir.join("cofactors.txt")).unwrap();
                assert_eq!(&written, cofactors, "{what}");
            }
        }
    }
}

/// `OBJECTIVES.md`'s edge cases, over GF(7) and over Q, each with its
/// order: the CLI writes what the crate returns.
#[test]
fn the_edge_cases_match_the_crate() {
    let cases = [
        ("x,y", "", "grevlex"),
        ("x,y", "0\n", "grevlex"),
        ("x,y", "x,\nx-1\n", "grevlex"),
        ("x,y", "x,\ny\n", "grevlex"),
        ("x,y", "x^2+y,\n0,\nx*y-1\n", "grevlex"),
        ("t,x,y", "x-t^2,\ny-t^3\n", "block:1"),
    ];
    for characteristic in ["7", "0"] {
        for (i, (vars, generators, order)) in cases.iter().enumerate() {
            let text = format!("{vars}\n{characteristic}\n{generators}");
            let mut p = params(order, 1, 11);
            p.checker = Some(CheckerMode::Exact);
            p.cofactors = true;
            let expected = crate_call(&text, &p);
            assert_eq!(expected.certified, Some(true), "{text:?}");
            let dir = scratch(&format!("edge-{characteristic}-{i}"));
            let cofactors = dir.join("cofactors.txt");
            let ran = grobx(
                &[
                    OsString::from("--order"),
                    OsString::from(order),
                    "--seed".into(),
                    "11".into(),
                    "--checker".into(),
                    "exact".into(),
                    "--cofactors".into(),
                    cofactors.clone().into_os_string(),
                ],
                &text,
            );
            assert_eq!(ran.code(), 0, "{text:?}: {ran:?}");
            assert_eq!(Some(ran.stdout), expected.basis, "{text:?}");
            assert_eq!(ran.stderr, "grobx: checker: certified\n", "{text:?}");
            assert_eq!(
                Some(fs::read_to_string(&cofactors).unwrap()),
                expected.cofactors
            );
        }
    }
    // The zero ideal without the checker, and a text whose last line has no
    // line feed.
    let ran = grobx(&["--seed", "1"], "x,y\n7\n");
    assert_eq!((ran.code(), ran.stdout.as_str()), (0, "[]:\n"));
    let ran = grobx(&["--seed", "1"], "x,y\n7\nx^2+y,\nx*y-1");
    assert_eq!(
        (ran.code(), ran.stdout.as_str()),
        (0, "[y^2+x,\nx*y+6,\nx^2+y]:\n")
    );
}

fn parse(args: &[&str]) -> Command {
    grobx_cli::parse(args.iter().map(OsString::from)).unwrap_or_else(|e| panic!("{args:?}: {e:?}"))
}

fn run(args: &[&str]) -> Run {
    match parse(args) {
        Command::Run(run) => *run,
        other @ (Command::Help | Command::Version) => panic!("{args:?}: {other:?}"),
    }
}

#[test]
fn the_switches_fill_the_params() {
    let r = run(&["--seed", "1"]);
    let p = &r.params;
    assert_eq!(p.order, MonomialOrder::Grevlex);
    assert_eq!(p.threads, Threads::Count(NonZeroUsize::MIN));
    assert_eq!(p.seed, 1);
    assert!(!r.drawn);
    assert_eq!(r.args, ["--seed", "1"]);
    assert_eq!(p.deadline, None);
    assert!(p.cancel.is_none());
    assert_eq!(p.memory, None);
    assert_eq!(p.checker, None);
    assert!(!p.cofactors);
    assert!(p.signatures);
    assert!(matches!(r.input, Input::Stdin));
    assert_eq!(r.output, None);
    assert_eq!(r.cofactors, None);
    assert!(!p.report);
    assert_eq!(r.report, None);

    let spaced = [
        "--order",
        "block:2",
        "--threads",
        "0",
        "--seed",
        "18446744073709551615",
        "--deadline",
        "1500",
        "--memory",
        "4096",
        "--checker",
        "3",
        "--cofactors",
        "c.txt",
        "--output",
        "b.txt",
        "f.ms",
    ];
    let joined: Vec<String> = spaced[..16]
        .chunks(2)
        .map(|kv| format!("{key}={value}", key = kv[0], value = kv[1]))
        .chain(["f.ms".to_string()])
        .collect();
    let joined: Vec<&str> = joined.iter().map(String::as_str).collect();
    for args in [&spaced[..], &joined[..]] {
        let r = run(args);
        let p = &r.params;
        assert_eq!(p.order, MonomialOrder::Block(vec![2]), "{args:?}");
        assert_eq!(p.threads, Threads::All, "{args:?}");
        assert_eq!(p.seed, u64::MAX, "{args:?}");
        assert_eq!(p.deadline, Some(Duration::from_millis(1500)), "{args:?}");
        assert!(p.cancel.is_none(), "{args:?}");
        assert_eq!(p.memory, Some(4096), "{args:?}");
        assert_eq!(
            p.checker,
            Some(CheckerMode::Cheap {
                points: NonZeroU32::new(3).unwrap()
            }),
            "{args:?}"
        );
        assert!(p.cofactors, "{args:?}");
        assert!(p.signatures, "{args:?}");
        assert!(
            matches!(&r.input, Input::File(f) if f == &PathBuf::from("f.ms")),
            "{args:?}"
        );
        assert_eq!(r.output, Some(PathBuf::from("b.txt")), "{args:?}");
        assert_eq!(r.cofactors, Some(PathBuf::from("c.txt")), "{args:?}");
    }
    let r = run(&[
        "--checker",
        "exact",
        "--threads",
        "8",
        "--seed",
        "0",
        "--memory",
        "0",
    ]);
    assert_eq!(r.params.checker, Some(CheckerMode::Exact));
    assert_eq!(
        r.params.threads,
        Threads::Count(NonZeroUsize::new(8).unwrap())
    );
    assert_eq!(r.params.memory, Some(0));
    assert!(!r.params.cofactors);

    for (value, ms) in [
        ("0s", 0),
        ("10s", 10_000),
        ("10m", 600_000),
        ("10h", 36_000_000),
        ("5124095576030h", 18_446_744_073_708_000_000),
        ("18446744073709551615", u64::MAX),
    ] {
        let r = run(&["--deadline", value]);
        assert_eq!(
            r.params.deadline,
            Some(Duration::from_millis(ms)),
            "{value}"
        );
    }

    // Two draws that agree come once in 2^64.
    let (a, b) = (run(&[]), run(&["f.ms"]));
    assert!(a.drawn && b.drawn);
    assert_ne!(a.params.seed, b.params.seed);
    assert_eq!(b.args, ["f.ms"]);
}

#[test]
fn the_header() {
    let header = |args: &[&str]| grobx_cli::header(&run(args));
    let version = format!("version {version}\n", version = grobx::VERSION);
    let cases: &[(&[&str], &str)] = &[
        (&[], "grobx"),
        (
            &["--seed=1", "--report", "r.txt", "a/b-c_d.ms"],
            "grobx --seed=1 --report r.txt a/b-c_d.ms",
        ),
        (
            &["--report", "r x.txt", "--", ""],
            "grobx --report 'r x.txt' -- ''",
        ),
        (&["it's.ms"], r"grobx 'it'\''s.ms'"),
        (&["grün.ms"], "grobx 'grün.ms'"),
        (&["--output", "$HOME", "a*"], "grobx --output '$HOME' 'a*'"),
    ];
    let mut failed = Vec::new();
    for (args, command) in cases {
        let want = format!("{version}command {command}\n");
        let got = header(args);
        if got != want {
            failed.push(format!("{args:?}: got {got:?}, want {want:?}"));
        }
    }
    assert!(failed.is_empty(), "{failed}", failed = failed.join("\n"));
}

#[test]
fn operands() {
    assert!(matches!(run(&["--seed", "1", "-"]).input, Input::Stdin));
    assert!(matches!(run(&["-", "--seed", "1"]).input, Input::Stdin));
    let file = |args: &[&str]| match run(args).input {
        Input::File(f) => f,
        Input::Stdin => panic!("{args:?}: standard input"),
    };
    assert_eq!(file(&["--seed", "1", "f.ms"]), PathBuf::from("f.ms"));
    assert_eq!(file(&["f.ms", "--seed", "1"]), PathBuf::from("f.ms"));
    assert_eq!(
        file(&["--seed", "1", "--", "-x.ms"]),
        PathBuf::from("-x.ms")
    );
    assert_eq!(
        file(&["--seed", "1", "--", "--seed"]),
        PathBuf::from("--seed")
    );

    let dir = scratch("operands");
    fs::write(dir.join("-tiny.ms"), "x,y\n7\nx^2+y,\nx*y-1\n").unwrap();
    let ran = grobx_in(&["--seed", "1", "--", "-tiny.ms"], "", Some(&dir));
    assert_eq!(ran.code(), 0, "{ran:?}");
    assert_eq!(ran.stdout, "[y^2+x,\nx*y+6,\nx^2+y]:\n");
}

const SWITCHES: [&str; 11] = [
    "--order",
    "--threads",
    "--seed",
    "--deadline",
    "--memory",
    "--checker",
    "--cofactors",
    "--report",
    "--output",
    "--help",
    "--version",
];

#[test]
fn help_lists_every_switch() {
    let long = grobx(&["--help"], "");
    assert_eq!(long.code(), 0, "{long:?}");
    assert_eq!(long.stderr, "");
    let hello = format!(
        "grobx {version} -- reduced Gröbner bases. Copyright Code1 GmbH.\n\nUsage: ",
        version = grobx::VERSION
    );
    assert!(
        long.stdout.starts_with(&hello),
        "{stdout}",
        stdout = long.stdout
    );
    for switch in SWITCHES {
        assert!(long.stdout.contains(switch), "--help names {switch}");
    }
    assert!(long.stdout.contains("-h,"));
    let short = grobx(&["-h"], "");
    assert_eq!(
        (short.code(), &short.stdout, &short.stderr),
        (0, &long.stdout, &String::new())
    );
    let late = grobx(&["--seed", "1", "--help"], "");
    assert_eq!((late.code(), &late.stdout), (0, &long.stdout));
    assert!(matches!(
        parse(&["--seed", "1", "--help", "--bogus"]),
        Command::Help
    ));
    assert!(matches!(parse(&["--version", "--help"]), Command::Version));
    // A value spelled like a switch is the value; --output's is any text.
    assert!(matches!(
        parse(&["--output", "--help", "--version"]),
        Command::Version
    ));
}

#[test]
fn version_prints_the_crate_version() {
    let ran = grobx(&["--version"], "");
    assert_eq!(ran.code(), 0, "{ran:?}");
    assert_eq!(
        ran.stdout,
        format!("grobx {version}\n", version = grobx::VERSION)
    );
    assert_eq!(ran.stderr, "");
}
