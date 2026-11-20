//! silent-fallbacks, the CLI's test code: a report line the call could not
//! write is said, and a refused call leaves a device in its report's place
//! without a word (`iterations/0094-silent-fallbacks/plan.md`, _Test code_,
//! 5 and 6).

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::os::unix::fs::FileTypeExt;
use std::path::Path;
use std::time::Duration;

use crate::common::{HEADER_LINES, command, fifo, finish, grobx, scratch, write_instance};
use crate::systems::dense;

/// Test 5: silent-fallbacks.6. The report is a FIFO whose reader goes away
/// after the header, while the call runs on `dense(8, 3, 65521, 1)`, which
/// runs for seconds and which the deadline stops: the first line on
/// standard error says the sink's error, and the exit status is 2, since
/// the report could not be written whole.
#[test]
fn a_lost_report_line_is_said() {
    let dir = scratch("silent-fallbacks-sink");
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
        "grobx: {path}: {epipe}",
        path = report.display(),
        epipe = io::Error::from_raw_os_error(libc::EPIPE)
    );
    assert_eq!(ran.stderr.lines().next(), Some(said.as_str()), "{ran:?}");
    assert_eq!(ran.code(), 2, "{ran:?}");
}

/// Test 6: silent-fallbacks.4, error-report.2. A call the crate refuses,
/// its report a regular file or `/dev/null`: exit status 2 and the call's
/// error the one line on standard error; the regular file stays, ending
/// with the `error` line, and `/dev/null` stays a character device.
#[test]
fn a_refused_call_leaves_a_device() {
    let dir = scratch("silent-fallbacks-refused");
    let file = dir.join("refused.report");
    let refused = "x\n7\nx^2147483648\n";
    for report in [file.as_path(), Path::new("/dev/null")] {
        let ran = grobx(
            &[
                OsStr::new("--seed"),
                OsStr::new("1"),
                OsStr::new("--report"),
                report.as_os_str(),
            ],
            refused,
        );
        let name = report.display();
        assert_eq!(ran.code(), 2, "{name}: {ran:?}");
        assert_eq!(ran.stderr.lines().count(), 1, "{name}: {ran:?}");
        assert!(
            ran.stderr.starts_with("grobx: standard input: "),
            "{name}: {ran:?}"
        );
    }
    let text =
        fs::read_to_string(&file).unwrap_or_else(|e| panic!("the refused call's report: {e}"));
    let last = text.lines().last().unwrap_or_default();
    assert!(
        last.contains(" error error=Overflow "),
        "the refused call's report: {text}"
    );
    let null = fs::metadata("/dev/null").unwrap();
    assert!(null.file_type().is_char_device(), "/dev/null: {null:?}");
}
