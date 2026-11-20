//! cli-report-device, the CLI's test code: a report on a device or a pipe
//! is the report there, and the call keeps its exit status
//! (`iterations/0099-cli-report-device/plan.md`, _Test code_, 1 to 5).

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::os::unix::fs::{FileTypeExt, symlink};
use std::path::Path;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::common::{
    HEADER_LINES, command, fifo, finish, grobx, header, kinds, scratch, write_instance,
};
use crate::systems::dense;

/// The two instances of the device and pipe tests: a dense system, and the
/// zero ideal, whose report is the shortest and whose basis is `[]:`.
fn instances() -> [(&'static str, String); 2] {
    [
        ("dense", dense(4, 3, 65521, 1)),
        ("zero", "x,y\n65521\n0\n".to_string()),
    ]
}

/// Reads the FIFO at `path` to its end on a thread of its own, which
/// blocks until the binary opens the FIFO to write.
fn reader(path: &Path) -> JoinHandle<String> {
    let path = path.to_path_buf();
    thread::spawn(move || fs::read_to_string(&path).unwrap())
}

/// The kinds of `report`'s lines, owned.
fn owned_kinds(report: &str) -> Vec<String> {
    kinds(report).iter().map(|k| (*k).to_string()).collect()
}

/// The report the binary writes to a regular file for `instance` under
/// seed 3, by the kinds of its lines.
fn file_kinds(dir: &Path, stem: &str, instance: &str) -> Vec<String> {
    let file = dir.join(format!("{stem}.report"));
    let report = format!("--report={report}", report = file.display());
    let ran = grobx(&["--seed", "3", report.as_str(), instance], "");
    assert_eq!(ran.code(), 0, "{stem}: {ran:?}");
    owned_kinds(&fs::read_to_string(&file).unwrap())
}

/// Test 1: cli-report-device.1. `--report /dev/null` on the dense system
/// and on the zero ideal: exit status 0, the basis on standard output as
/// without the switch, nothing on standard error, and `/dev/null` a
/// character device after.
#[test]
fn a_report_on_a_device_keeps_the_call_status() {
    let dir = scratch("cli-report-device-null");
    for (stem, text) in instances() {
        let input = write_instance(&dir, stem, &text).display().to_string();
        let plain = grobx(&["--seed", "3", input.as_str()], "");
        assert_eq!(plain.code(), 0, "{stem}: {plain:?}");
        let ran = grobx(
            &["--seed", "3", "--report", "/dev/null", input.as_str()],
            "",
        );
        assert_eq!(ran.code(), 0, "{stem}: {ran:?}");
        assert_eq!(ran.stdout, plain.stdout, "{stem}");
        assert_eq!(ran.stderr, "", "{stem}");
    }
    let null = fs::metadata("/dev/null").unwrap();
    assert!(null.file_type().is_char_device(), "/dev/null: {null:?}");
}

/// Test 2: cli-report-device.1. `--report` on a FIFO read to its end, on
/// the dense system and on the zero ideal: exit status 0, nothing on
/// standard error, and the text read is the header, then the report once,
/// its kinds those of the report the same call writes to a regular file.
#[test]
fn a_report_on_a_pipe_is_the_report() {
    let dir = scratch("cli-report-device-pipe");
    for (stem, text) in instances() {
        let input = write_instance(&dir, stem, &text).display().to_string();
        let expected = file_kinds(&dir, stem, &input);
        let pipe = fifo(&dir, &format!("{stem}.fifo"));
        let read = reader(&pipe);
        let report = format!("--report={report}", report = pipe.display());
        let args = ["--seed", "3", report.as_str(), input.as_str()];
        let ran = grobx(&args, "");
        assert_eq!(ran.code(), 0, "{stem}: {ran:?}");
        assert_eq!(ran.stderr, "", "{stem}");
        let text = read.join().unwrap();
        assert!(text.starts_with(&header(&args)), "{stem}: {text}");
        let versions = text.lines().filter(|l| l.starts_with("version ")).count();
        assert_eq!(versions, 1, "{stem}: {text}");
        let got = owned_kinds(&text);
        assert_eq!(
            got.iter().filter(|k| *k == "end").count(),
            1,
            "{stem}: {text}"
        );
        assert_eq!(got, expected, "{stem}: {text}");
    }
}

/// Test 3: cli-report-device.1. A call `--deadline 0` stops, its report on
/// a FIFO read to its end: exit status 1, `grobx: stopped: deadline` the
/// one line on standard error, and the text read holds `stop
/// bound=deadline` and ends with `end result=stopped`. The call is on
/// `dense(8, 3, 65521, 1)`, which runs for seconds without the bound.
#[test]
fn a_stopped_call_reports_on_a_pipe() {
    let dir = scratch("cli-report-device-stopped");
    let input = write_instance(&dir, "long", &dense(8, 3, 65521, 1))
        .display()
        .to_string();
    let pipe = fifo(&dir, "long.fifo");
    let read = reader(&pipe);
    let report = format!("--report={report}", report = pipe.display());
    let args = [
        "--seed",
        "3",
        "--deadline",
        "0",
        report.as_str(),
        input.as_str(),
    ];
    let ran = grobx(&args, "");
    assert_eq!(ran.code(), 1, "{ran:?}");
    assert_eq!(ran.stderr, "grobx: stopped: deadline\n", "{ran:?}");
    let text = read.join().unwrap();
    assert!(text.starts_with(&header(&args)), "{text}");
    let k = kinds(&text);
    assert_eq!(&k[k.len() - 2..], ["stop", "end"], "{text}");
    assert!(text.contains(" stop bound=deadline\n"), "{text}");
    assert!(
        text.lines().last().unwrap().contains(" result=stopped "),
        "{text}"
    );
}

/// Test 4: cli-report-device.2. The report is a FIFO whose reader goes
/// away after the header, while the call runs on `dense(8, 3, 65521, 1)`
/// under `--deadline 500`: standard error is one line, the sink's EPIPE,
/// and the exit status is 2, the report on the pipe being incomplete.
#[test]
fn a_lost_line_on_a_pipe_is_said_once() {
    let dir = scratch("cli-report-device-lost");
    let instance = write_instance(&dir, "long", &dense(8, 3, 65521, 1));
    let report = fifo(&dir, "long.report");
    let child = command(&[
        OsStr::new("--seed"),
        OsStr::new("1"),
        OsStr::new("--deadline"),
        OsStr::new("500"),
        OsStr::new("--report"),
        report.as_os_str(),
        instance.as_os_str(),
    ])
    .spawn()
    .expect("the binary starts");
    let mut reader = BufReader::new(File::open(&report).unwrap());
    let mut line = String::new();
    for _ in 0..HEADER_LINES {
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(line.ends_with('\n'), "the header: {line:?}");
    }
    drop(reader);
    let ran = finish(child, Duration::from_secs(60));
    let said = format!(
        "grobx: {path}: {epipe}\n",
        path = report.display(),
        epipe = io::Error::from_raw_os_error(libc::EPIPE)
    );
    assert_eq!(ran.stderr, said, "{ran:?}");
    assert_eq!(ran.code(), 2, "{ran:?}");
}

/// Test 5: cli-report-device.3. `--report` names a symbolic link to a
/// regular file: exit status 0, the target holds the header and the
/// report, its kinds those of the report written to a regular file, and
/// the link stays a link.
#[test]
fn a_link_to_a_regular_file_is_rewritten() {
    let dir = scratch("cli-report-device-link");
    let input = write_instance(&dir, "dense", &dense(4, 3, 65521, 1))
        .display()
        .to_string();
    let expected = file_kinds(&dir, "dense", &input);
    let target = dir.join("target.report");
    fs::write(&target, "stale\n").unwrap();
    let link = dir.join("link.report");
    symlink(&target, &link).unwrap();
    let report = format!("--report={report}", report = link.display());
    let args = ["--seed", "3", report.as_str(), input.as_str()];
    let ran = grobx(&args, "");
    assert_eq!(ran.code(), 0, "{ran:?}");
    assert_eq!(ran.stderr, "", "{ran:?}");
    let text = fs::read_to_string(&target).unwrap();
    assert!(text.starts_with(&header(&args)), "{text}");
    assert_eq!(owned_kinds(&text), expected, "{text}");
    let kind = fs::symlink_metadata(&link).unwrap().file_type();
    assert!(kind.is_symlink(), "{link}: {kind:?}", link = link.display());
}
