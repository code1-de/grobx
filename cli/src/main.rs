//! The binary `grobx`: reads an instance, calls `grobx::groebner`, and
//! writes what it returns (`MANUAL.md`, _CLI_). `grobx_cli` reads the
//! arguments and the outcome, and `interrupt` takes SIGINT.

mod interrupt;

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::ExitCode;

use grobx_cli::{Command, ERROR, Input, Run, Usage};

// small-block-time.4 design-checked -- the CLI sets the crate's allocator as its global allocator, not the crate setting it for every program that links it (not measured)
#[global_allocator]
static GLOBAL: grobx::Allocator = grobx::Allocator;

fn main() -> ExitCode {
    let code = match grobx_cli::parse(std::env::args_os().skip(1)) {
        Err(Usage(message)) => {
            say(&message);
            say("--help lists the switches");
            ERROR
        }
        Ok(Command::Help) => print(&grobx_cli::help()),
        Ok(Command::Version) => print(&grobx_cli::version()),
        Ok(Command::Run(run)) => execute(*run),
    };
    ExitCode::from(code)
}

// cli.15 practice-checked: GNU error message format -- every message to standard error is one line, grobx: and the message, not the message alone
/// Writes `grobx: `, `message`, and a line feed to standard error, and
/// ignores a failure to write there.
fn say(message: &str) {
    let line = format!("grobx: {message}\n");
    #[allow(
        clippy::let_underscore_must_use,
        reason = "standard error is where a failure to write would be said"
    )]
    let _: io::Result<()> = io::stderr().write_all(line.as_bytes());
}

/// Writes `text` to standard output: `DONE`, or `ERROR` once it has said
/// why it could not.
fn print(text: &str) -> u8 {
    match out(text) {
        Ok(()) => grobx_cli::DONE,
        Err(()) => ERROR,
    }
}

/// Writes `text` to standard output: `Err` once it has said why it could
/// not.
fn out(text: &str) -> Result<(), ()> {
    put(&mut io::stdout().lock(), text).map_err(|e| {
        say(&format!("standard output: {e}"));
    })
}

fn put(to: &mut impl Write, text: &str) -> io::Result<()> {
    to.write_all(text.as_bytes())?;
    to.flush()
}

/// Reads the instance, starts the watch on SIGINT, creates the output
/// files, calls the crate, and writes what it returns. Returns the exit status, `ERROR` once it has
/// said what failed.
fn execute(mut run: Run) -> u8 {
    let (name, read) = match &run.input {
        Input::File(path) => (path.display().to_string(), fs::read(path)),
        Input::Stdin => {
            let mut bytes = Vec::new();
            let read = io::stdin().lock().read_to_end(&mut bytes);
            ("standard input".to_string(), read.map(|_| bytes))
        }
    };
    let bytes = match read {
        Ok(bytes) => bytes,
        Err(e) => {
            say(&format!("{name}: {e}"));
            return ERROR;
        }
    };
    let Ok(text) = String::from_utf8(bytes) else {
        say(&format!("{name}: the instance is not UTF-8"));
        return ERROR;
    };
    // cli.21 design-checked -- the watch starts once the instance is read, and SIGINT's default action ends a process that waits on its input, not the watch first in main with a SIGINT during the read held for the call (not measured)
    let cancel = grobx::Cancel::new();
    if let Err(e) = interrupt::watch(&cancel) {
        say(&format!("SIGINT: {e}"));
        return ERROR;
    }
    // cli.17 design-checked -- the output files are created after the instance is read and before the call, not after the call (not measured)
    let Ok(output) = create(run.output.as_deref()) else {
        return ERROR;
    };
    let Ok(cofactors) = create(run.cofactors.as_deref()) else {
        return ERROR;
    };
    // call-report.25 design-checked -- the report's file is created with the other output files before the call (cli.17), not created after the call returns an outcome (not measured)
    // error-report.2 design-checked -- where the call returns an error, the report's file stays, ending in the error line, not removed (not measured)
    let Ok(report) = create(run.report.as_deref()) else {
        return ERROR;
    };
    let Ok(regular) = regular(report.as_ref().zip(run.report.as_deref())) else {
        return ERROR;
    };
    if run.drawn {
        say(&format!("seed {seed}", seed = run.params.seed));
    }
    // A call that never returns, killed or hung, leaves its report so far.
    if let (Some(file), Some(path)) = (&report, &run.report) {
        if write(file, path, &grobx_cli::header(&run)).is_err() {
            return ERROR;
        }
        match file.try_clone() {
            Ok(clone) => run.params.report_sink = Some(grobx::ReportSink::new(clone)),
            Err(e) => {
                say(&format!("{path}: {e}", path = path.display()));
                return ERROR;
            }
        }
    }
    run.params.cancel = Some(cancel);
    let outcome = match grobx::groebner(&text, &run.params) {
        Ok(outcome) => outcome,
        // cli.16 practice-checked: GNU error message format (program: file: message) -- an error the crate returns is written as the input's name and the error's Display, not with the error's variant first (not measured)
        Err(e) => {
            say(&format!("{name}: {e}"));
            return ERROR;
        }
    };
    if let Some(basis) = &outcome.basis {
        let written = match output.as_ref().zip(run.output.as_ref()) {
            Some((file, path)) => write(file, path, basis),
            None => out(basis),
        };
        if written.is_err() {
            return ERROR;
        }
    }
    if let (Some(file), Some(path), Some(text)) = (&cofactors, &run.cofactors, &outcome.cofactors)
        && write(file, path, text).is_err()
    {
        return ERROR;
    }
    // silent-fallbacks.6 design-checked -- a line the sink lost is said before a regular file is written whole again, and the exit status stays the call's where that succeeds, not exit status 2 at once, nor silence where the rewrite succeeds, nor a writer of the CLI's own that says it at the failed write
    if let (Some(path), Some(e)) = (&run.report, &outcome.sink_error) {
        say(&format!("{path}: {e}", path = path.display()));
    }
    // cli-report-device.1 design-checked -- a report file that is no regular file keeps the lines the sink streamed to it, and nothing follows its end line, not the report written whole again after them, since a device or a pipe cannot be truncated, nor one write at the call's end alone, which a reader that watches the call would wait on
    // cli-report-device.2 design-checked -- a line lost on such a file exits 2, the report there being incomplete and beyond repair, not the call's exit status with the loss said alone
    // Only where the sink lost a line: the streamed file holds every line else, and the `alloc`
    // lines, which the sink alone takes, would be lost to a rewrite from the outcome's text.
    if let (Some(file), Some(path), Some(text)) = (&report, &run.report, &outcome.report)
        && outcome.sink_error.is_some()
    {
        if !regular {
            return ERROR;
        }
        // Written whole again: a write the sink lost is in the outcome's text.
        if rewrite(file, path, &(grobx_cli::header(&run) + text)).is_err() {
            return ERROR;
        }
    }
    if let Some(line) = grobx_cli::diagnostic(&outcome) {
        say(&line);
    }
    grobx_cli::status(&outcome)
}

/// Creates the file at `path` where there is one: `Err` once it has said
/// why it could not.
fn create(path: Option<&Path>) -> Result<Option<File>, ()> {
    let Some(path) = path else { return Ok(None) };
    File::create(path).map(Some).map_err(|e| {
        say(&format!("{path}: {e}", path = path.display()));
    })
}

// cli-report-device.3 practice-checked: TOCTOU avoidance (fstat of the open file, not stat of the path) -- the report file's kind is read from the open file's metadata, not from fs::metadata or symlink_metadata of the path, which can be replaced between the check and the write, and whose link's own kind is not the file's
/// Whether the report file is a regular file, which the CLI writes whole
/// again after the call; false where there is none. `Err` once it has
/// said why the kind could not be read.
fn regular(report: Option<(&File, &Path)>) -> Result<bool, ()> {
    let Some((file, path)) = report else {
        return Ok(false);
    };
    file.metadata().map(|m| m.is_file()).map_err(|e| {
        say(&format!("{path}: {e}", path = path.display()));
    })
}

/// Writes `text` to `file`, created at `path`: `Err` once it has said why
/// it could not.
fn write(mut file: &File, path: &Path, text: &str) -> Result<(), ()> {
    put(&mut file, text).map_err(|e| say(&format!("{path}: {e}", path = path.display())))
}

/// Replaces what the file at `path` holds with `text`.
fn rewrite(mut file: &File, path: &Path, text: &str) -> Result<(), ()> {
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)))
        .map_err(|e| say(&format!("{path}: {e}", path = path.display())))?;
    write(file, path, text)
}
