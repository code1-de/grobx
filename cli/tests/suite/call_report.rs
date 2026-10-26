//! call-report, sub-item 3: the report through the CLI's switch `--report
//! FILE` (`iterations/0064-call-report/plan.md`, _Test code_, 22, and
//! call-report.18).

use std::ffi::OsString;
use std::fs;

use grobx_cli::Command;

use crate::common::{grobx, scratch, write_instance};
use crate::systems::dense;

/// The header the binary writes above the report for `args`.
fn header<S: AsRef<str>>(args: &[S]) -> String {
    match grobx_cli::parse(args.iter().map(|a| OsString::from(a.as_ref()))) {
        Ok(Command::Run(run)) => grobx_cli::header(&run),
        other @ (Ok(Command::Help | Command::Version) | Err(_)) => panic!("{other:?}"),
    }
}

/// The kinds of a report's lines below the header's two, which each hold
/// a time and a kind.
fn kinds(report: &str) -> Vec<&str> {
    assert!(report.ends_with('\n'), "{report:?}");
    report
        .lines()
        .skip(2)
        .map(|l| {
            let mut words = l.split(' ');
            let time = words.next().unwrap();
            assert!(time.contains('.'), "{l}");
            words.next().expect("a kind")
        })
        .collect()
}

/// `--report FILE` writes the report to FILE, the instance read from a
/// file or from standard input, beside the basis on standard output; no
/// switch, no file; an error writes none. Over GF(65521) and over ℚ.
#[test]
fn the_switch_writes_the_report() {
    let dir = scratch("call-report");
    for (stem, text) in [("p65521", dense(4, 3, 65521, 1)), ("q", dense(3, 3, 0, 1))] {
        let file = dir.join(format!("{stem}.report"));
        let input = write_instance(&dir, stem, &text).display().to_string();
        for from_stdin in [false, true] {
            let _ = fs::remove_file(&file);
            let report = format!("--report={report}", report = file.display());
            let mut args = vec!["--seed".to_string(), "3".to_string(), report];
            let stdin = if from_stdin {
                text.clone()
            } else {
                args.push(input.clone());
                String::new()
            };
            let ran = grobx(&args, &stdin);
            assert_eq!(ran.code(), 0, "{stem}: {ran:?}");
            let text = fs::read_to_string(&file).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(text.starts_with(&header(&args)), "{stem}: {text}");
            let k = kinds(&text);
            assert_eq!(k.first(), Some(&"call"), "{stem}");
            assert_eq!(k.last(), Some(&"end"), "{stem}");
            let end = text.lines().last().unwrap();
            assert!(
                end.contains(&format!(" bytes={bytes} ", bytes = ran.stdout.len())),
                "{stem}: the basis on standard output: {end}"
            );
        }
        let _ = fs::remove_file(&file);
        let ran = grobx(&["--seed", "3", input.as_str()], "");
        assert_eq!(ran.code(), 0, "{stem}");
        assert!(!file.exists(), "{stem}: a report without the switch");
    }
    let file = dir.join("error.report");
    let report = format!("--report={report}", report = file.display());
    let ran = grobx(&["--seed", "3", report.as_str()], "x\n7\nx^2147483648\n");
    assert_eq!(ran.code(), 2, "{ran:?}");
    assert!(!file.exists(), "an error writes no report");
}

/// A deadline that stops the call: exit 1, and a report that ends with the
/// stop. `--help` lists the switch. The call runs on
/// `dense(8, 3, 65521, 1)`, which runs for seconds without the deadline.
#[test]
fn a_stopped_call_writes_its_report() {
    let dir = scratch("call-report-stopped");
    let file = dir.join("long.report");
    let report = format!("--report={report}", report = file.display());
    let input = write_instance(&dir, "long", &dense(8, 3, 65521, 1))
        .display()
        .to_string();
    let args = [
        "--seed",
        "3",
        "--deadline",
        "50",
        report.as_str(),
        input.as_str(),
    ];
    let ran = grobx(&args, "");
    assert_eq!(ran.code(), 1, "{ran:?}");
    let text = fs::read_to_string(&file).unwrap();
    assert!(text.starts_with(&header(&args)), "{text}");
    let k = kinds(&text);
    assert_eq!(&k[k.len() - 2..], ["stop", "end"], "{text}");
    assert!(
        text.lines().last().unwrap().contains(" result=stopped "),
        "{text}"
    );

    let help = grobx(&["--help"], "");
    assert_eq!(help.code(), 0);
    assert!(
        help.stdout.contains("--report FILE"),
        "{stdout}",
        stdout = help.stdout
    );
}

/// Without `--seed`, the drawn seed goes to standard error and is the one
/// the report's call line names.
#[test]
fn a_drawn_seed_is_written() {
    let dir = scratch("call-report-drawn");
    let file = dir.join("dense.report");
    let report = format!("--report={report}", report = file.display());
    let input = write_instance(&dir, "dense", &dense(3, 3, 65521, 1))
        .display()
        .to_string();
    let args = [report.as_str(), input.as_str()];
    let ran = grobx(&args, "");
    assert_eq!(ran.code(), 0, "{ran:?}");
    let seed = ran
        .stderr
        .strip_prefix("grobx: seed ")
        .and_then(|s| s.strip_suffix('\n'))
        .unwrap_or_else(|| panic!("{ran:?}"));
    assert!(seed.parse::<u64>().is_ok(), "{ran:?}");
    let text = fs::read_to_string(&file).unwrap();
    assert!(text.starts_with(&header(&args)), "{text}");
    let call = text.lines().nth(2).unwrap();
    assert!(call.contains(&format!(" seed={seed} ")), "{call}");
}
