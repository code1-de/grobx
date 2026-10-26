//! The reading of the CLI's arguments and of an outcome. `parse` reads the
//! switches into a `grobx::Params` and the paths, `header` writes the
//! report file's first lines, and `status` and `diagnostic` read an
//! outcome into an exit status and a line for standard error. The binary `grobx` reads the files, calls the crate,
//! writes, and takes SIGINT (`MANUAL.md`, _CLI_).

use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use grobx::{CheckerMode, MonomialOrder, Outcome, Params, Threads, random_seed};

/// The basis was written, and the checker accepted it or was not asked; or
/// the help or the version was printed.
pub const DONE: u8 = 0;
/// A bound stopped the call: its deadline, its memory ceiling, or SIGINT.
pub const STOPPED: u8 = 1;
/// An error: the watch on SIGINT, the arguments, the instance, an
/// output file or standard output, or a call the crate refuses.
pub const ERROR: u8 = 2;
// cli.14 design -- a result the checker rejects exits 3, with its basis and cofactors written, not 0 with the verdict on standard error alone (not measured)
/// The checker rejected the result, which was written all the same.
pub const REJECTED: u8 = 3;

/// What `--help` prints below its first line and the empty line after it.
const USAGE: &str = "\
Usage: grobx [SWITCH]... [FILE]

Computes the reduced Gröbner basis of the instance in FILE, or in standard
input where FILE is absent or -, and writes the basis text to standard
output.

The instance text is msolve's input format. Line 1 names the variables,
separated by commas, the largest first. Line 2 is the characteristic: a
prime below 2^64, or 0 for the rationals. The generators follow,
separated by commas:

  x,y,z
  65521
  x^2 + y*z - 1,
  2*x*y - 1/3*z,
  z^3 - x

Switches:
  --order NAME      grevlex, lex, or block:k1,...,km; default grevlex
  --threads N       how many threads the call may use, 0 for all; default 1
  --seed N          the seed of every random choice, 0 to 2^64-1; drawn at
                    random and written to standard error where absent
  --deadline TIME   a time limit on the call: milliseconds, or a number with
                    s, m, or h for seconds, minutes, or hours
  --memory BYTES    a ceiling on the process's resident set, in bytes
  --checker MODE    the checker rules on the result: exact, or a number of
                    random points for its cheap mode
  --cofactors FILE  writes the cofactor text to FILE; cofactors will make
                    the call considerably slower
  --report FILE     writes the call report to FILE: a log of every step of
                    the computation, with its sizes and times, which shows
                    where the call spends its time; a stopped call writes
                    one too
  --output FILE     writes the basis text to FILE, not to standard output
  -h, --help        prints this help and exits
  --version         prints the version and exits

A switch takes its value as --name VALUE or --name=VALUE. -- ends the
switches. Ctrl-C ends grobx while it reads the instance. After that,
Ctrl-C stops the call at its next check, and a second Ctrl-C ends the
process.

Exit status: 0 the basis, the help, or the version was written, 1 a bound
or Ctrl-C stopped the call, 2 an error, 3 the checker rejected the result.
";

/// `grobx` and the crate version, which `--version` and `--help` start
/// with.
fn title() -> String {
    format!("grobx {version}", version = grobx::VERSION)
}

// cli.8 practice: GNU --version convention -- --version prints grobx, the version, and a line feed, not the version alone
/// What `--version` prints.
#[must_use]
pub fn version() -> String {
    title() + "\n"
}

/// What `--help` prints.
#[must_use]
pub fn help() -> String {
    format!(
        "{title} -- reduced Gröbner bases. Copyright Code1 GmbH.\n\n{USAGE}",
        title = title()
    )
}

/// Where the instance text comes from.
#[derive(Debug)]
pub enum Input {
    Stdin,
    File(PathBuf),
}

/// A call to run: the parameters the switches set, cancel None, the
/// paths, and the arguments.
#[derive(Debug)]
pub struct Run {
    pub params: Params,
    /// Whether `params.seed` was drawn at random, `--seed` being absent.
    pub drawn: bool,
    /// The arguments after the program's name, as given.
    pub args: Vec<OsString>,
    pub input: Input,
    /// The file the basis text goes to; standard output where None.
    pub output: Option<PathBuf>,
    /// The file the cofactor text goes to.
    pub cofactors: Option<PathBuf>,
    /// The file the call report goes to.
    pub report: Option<PathBuf>,
}

/// What the arguments ask for.
#[derive(Debug)]
pub enum Command {
    Help,
    Version,
    Run(Box<Run>),
}

/// Why the arguments cannot be read: the message, without the program's
/// name.
#[derive(Debug, PartialEq, Eq)]
pub struct Usage(pub String);

/// The switches that take a value.
// cli.11 design -- field and prime are the instance text's, and no switch sets them, not a switch --prime that replaces its second line (not measured)
// cli.12 design -- no switch turns Params::signatures off, not a switch for it (not measured)
// input-redundancy-time.10 design -- no keyword, switch, or field sets Params::echelon: the call runs with it on, as with signatures, not a keyword, a switch, and a field for it (not measured)
#[derive(Clone, Copy)]
enum Switch {
    Order,
    Threads,
    Seed,
    Deadline,
    Memory,
    Checker,
    Cofactors,
    Report,
    Output,
}

// cli.4 design -- every switch is long and -h is the one short switch, not short switches as msolve's -f, -o, and -t (not measured)
const SWITCHES: [(&str, Switch); 9] = [
    ("--order", Switch::Order),
    ("--threads", Switch::Threads),
    ("--seed", Switch::Seed),
    ("--deadline", Switch::Deadline),
    ("--memory", Switch::Memory),
    ("--checker", Switch::Checker),
    ("--cofactors", Switch::Cofactors),
    // call-report.18 design -- the CLI's --report FILE writes the report text to FILE, a stopped call's too, not the report on standard error, which carries the checker's verdict, nor on standard output, which carries the basis (not measured)
    ("--report", Switch::Report),
    ("--output", Switch::Output),
];

/// The values the switches read so far.
#[derive(Default)]
struct Values {
    order: Option<MonomialOrder>,
    threads: Option<Threads>,
    seed: Option<u64>,
    deadline: Option<Duration>,
    memory: Option<u64>,
    checker: Option<CheckerMode>,
    cofactors: Option<PathBuf>,
    report: Option<PathBuf>,
    output: Option<PathBuf>,
}

impl Values {
    /// Whether `switch` has its value.
    fn has(&self, switch: Switch) -> bool {
        match switch {
            Switch::Order => self.order.is_some(),
            Switch::Threads => self.threads.is_some(),
            Switch::Seed => self.seed.is_some(),
            Switch::Deadline => self.deadline.is_some(),
            Switch::Memory => self.memory.is_some(),
            Switch::Checker => self.checker.is_some(),
            Switch::Cofactors => self.cofactors.is_some(),
            Switch::Report => self.report.is_some(),
            Switch::Output => self.output.is_some(),
        }
    }

    /// Reads `value` as the value of `switch`, named `name`.
    fn set(&mut self, switch: Switch, name: &str, value: OsString) -> Result<(), Usage> {
        match switch {
            Switch::Order => self.order = Some(order(name, &text(name, value)?)?),
            // f4-gfp.31 design -- threads = 0 means all the machine has, not None (not measured)
            Switch::Threads => {
                let n = number(name, &text(name, value)?, usize::MAX)?;
                self.threads = Some(NonZeroUsize::new(n).map_or(Threads::All, Threads::Count));
            }
            Switch::Seed => self.seed = Some(number(name, &text(name, value)?, u64::MAX)?),
            Switch::Deadline => self.deadline = Some(deadline(name, &text(name, value)?)?),
            Switch::Memory => self.memory = Some(number(name, &text(name, value)?, u64::MAX)?),
            Switch::Checker => self.checker = Some(checker(name, &text(name, value)?)?),
            Switch::Cofactors => self.cofactors = Some(value.into()),
            Switch::Report => self.report = Some(value.into()),
            Switch::Output => self.output = Some(value.into()),
        }
        Ok(())
    }
}

/// The value `value` of the switch `name` as text; a path alone may be
/// other than UTF-8.
fn text(name: &str, value: OsString) -> Result<String, Usage> {
    value
        .into_string()
        .map_err(|_| Usage(format!("{name}: the value is not UTF-8")))
}

// cli.13 design -- a number is decimal digits alone, with no sign, no space, and no suffix but a deadline's s, m, or h, in the units of grobx.groebner's keywords where it has none, not suffixes K, M, and G and a deadline in seconds (not measured)
/// The number `value` of the switch `name`: one or more ASCII digits that
/// lie in `T`, whose largest value is `max`.
fn number<T: FromStr + Display + Copy>(name: &str, value: &str, max: T) -> Result<T, Usage> {
    digits(value).ok_or_else(|| {
        Usage(format!(
            "{name} {value:?}: not a decimal number from 0 to {max}"
        ))
    })
}

/// `value` as a `T` where it is one or more ASCII digits that lie in `T`.
fn digits<T: FromStr>(value: &str) -> Option<T> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

/// The suffixes of `--deadline`, each with its milliseconds.
const DEADLINE_UNITS: [(char, u64); 3] = [('s', 1_000), ('m', 60_000), ('h', 3_600_000)];

/// The value of `--deadline`: decimal digits, in milliseconds, or followed
/// by `s`, `m`, or `h` for seconds, minutes, or hours, at most 2^64 − 1
/// milliseconds in all.
fn deadline(name: &str, value: &str) -> Result<Duration, Usage> {
    let (count, unit) = DEADLINE_UNITS
        .iter()
        .find_map(|&(suffix, ms)| value.strip_suffix(suffix).map(|count| (count, ms)))
        .unwrap_or((value, 1));
    digits::<u64>(count)
        .and_then(|n| n.checked_mul(unit))
        .map(Duration::from_millis)
        .ok_or_else(|| {
            Usage(format!(
                "{name} {value:?}: a decimal number of milliseconds, or one followed by s, m, \
                 or h, up to {max} milliseconds",
                max = u64::MAX
            ))
        })
}

/// The value of `--checker`: `exact`, or a number of points from 1 to
/// 2^32 − 1.
fn checker(name: &str, value: &str) -> Result<CheckerMode, Usage> {
    if value == "exact" {
        return Ok(CheckerMode::Exact);
    }
    match digits(value).and_then(NonZeroU32::new) {
        Some(points) => Ok(CheckerMode::Cheap { points }),
        None => Err(Usage(format!(
            "{name} {value:?}: exact, or a decimal number of points from 1 to {max}",
            max = u32::MAX
        ))),
    }
}

/// The value of `--order`: `grevlex`, `lex`, or `block:k1,…,km`. Whether a
/// block order's sizes fit the variables, the call finds.
fn order(name: &str, value: &str) -> Result<MonomialOrder, Usage> {
    value.parse().map_err(|_| {
        Usage(format!(
            "{name} {value:?}: grevlex, lex, or block:k1,...,km"
        ))
    })
}

// cli.2 design -- the switches are read by code of the CLI's own, not by clap (not measured)
/// The command the arguments after the program's name ask for, read in
/// order.
///
/// # Errors
///
/// `Usage` where a switch is unknown, given twice, or lacks or misreads
/// its value, where `--help` or `--version` is given a value, or where
/// more than one FILE is given.
pub fn parse<I: IntoIterator<Item = OsString>>(args: I) -> Result<Command, Usage> {
    let given: Vec<OsString> = args.into_iter().collect();
    let mut args = given.iter().cloned();
    let mut values = Values::default();
    let mut operand: Option<OsString> = None;
    let mut ended = false;
    while let Some(arg) = args.next() {
        let bytes = arg.as_encoded_bytes();
        if ended || bytes == b"-" || !bytes.starts_with(b"-") {
            if let Some(first) = &operand {
                return Err(Usage(format!(
                    "more than one instance: {first} and {arg}",
                    first = first.to_string_lossy(),
                    arg = arg.to_string_lossy()
                )));
            }
            operand = Some(arg);
            continue;
        }
        // cli.5 practice: POSIX end-of-options marker -- -- ends the switches and every argument after it is an operand, not a path that begins with - written ./-x
        if bytes == b"--" {
            ended = true;
            continue;
        }
        // cli.3 practice: GNU long options -- a switch takes its value as --name value or --name=value, not in one form alone
        let (name, inline) = match bytes.iter().position(|&b| b == b'=') {
            Some(at) if bytes.starts_with(b"--") => {
                // SAFETY: the bytes start right after `=`, and bytes of
                // as_encoded_bytes may be split right after any ASCII byte
                // (OsStr::from_encoded_bytes_unchecked, Safety).
                let value = unsafe { OsStr::from_encoded_bytes_unchecked(&bytes[at + 1..]) };
                (&bytes[..at], Some(value.to_owned()))
            }
            Some(_) | None => (bytes, None),
        };
        // cli.7 design -- --help and --version act where the reading reaches them, and an error before them is returned, not either one anywhere on the line before every error (not measured)
        let command = match name {
            b"--help" | b"-h" => Some(Command::Help),
            b"--version" => Some(Command::Version),
            _ => None,
        };
        if let Some(command) = command {
            if inline.is_some() {
                let name = String::from_utf8_lossy(name);
                return Err(Usage(format!("{name} takes no value")));
            }
            return Ok(command);
        }
        let Some(&(name, switch)) = SWITCHES.iter().find(|(n, _)| n.as_bytes() == name) else {
            return Err(Usage(format!(
                "unknown switch {arg}",
                arg = arg.to_string_lossy()
            )));
        };
        // cli.6 design -- a switch given twice is a usage error, not the last one counting (not measured)
        if values.has(switch) {
            return Err(Usage(format!("{name} given twice")));
        }
        let value = match inline {
            Some(value) => value,
            None => args
                .next()
                .ok_or_else(|| Usage(format!("{name} takes a value")))?,
        };
        values.set(switch, name, value)?;
    }
    let (seed, drawn) = match values.seed {
        Some(seed) => (seed, false),
        None => (random_seed(), true),
    };
    // cli.9 design -- --order defaults to grevlex, as grobx.groebner's keyword does, not a required switch (not measured)
    let order = values.order.unwrap_or(MonomialOrder::Grevlex);
    // cli.10 design -- --threads defaults to 1, not 0, all the machine has (not measured)
    let threads = values.threads.unwrap_or(Threads::Count(NonZeroUsize::MIN));
    let mut params = Params::new(order, threads, seed);
    params.deadline = values.deadline;
    params.memory = values.memory;
    params.checker = values.checker;
    params.cofactors = values.cofactors.is_some();
    params.report = values.report.is_some();
    let input = match operand {
        None => Input::Stdin,
        Some(operand) if operand == "-" => Input::Stdin,
        Some(operand) => Input::File(operand.into()),
    };
    Ok(Command::Run(Box::new(Run {
        params,
        drawn,
        args: given,
        input,
        output: values.output,
        cofactors: values.cofactors,
        report: values.report,
    })))
}

/// The lines the report file opens with, before the call report's: the
/// crate version, and the command line of `run` as given.
#[must_use]
pub fn header(run: &Run) -> String {
    let mut command = String::from("grobx");
    for arg in &run.args {
        command.push(' ');
        command.push_str(&quote(&arg.to_string_lossy()));
    }
    format!(
        "version {version}\ncommand {command}\n",
        version = grobx::VERSION
    )
}

/// `arg` as a shell word: as it is where it is not empty and every
/// character is safe, in single quotes otherwise.
fn quote(arg: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "-_./:=,+@%".contains(c);
    if !arg.is_empty() && arg.chars().all(safe) {
        return arg.to_string();
    }
    format!("'{escaped}'", escaped = arg.replace('\'', "'\\''"))
}

/// The exit status of a call that returned `outcome`.
#[must_use]
pub fn status(outcome: &Outcome) -> u8 {
    if outcome.stopped.is_some() {
        return STOPPED;
    }
    match outcome.certified {
        Some(false) => REJECTED,
        Some(true) | None => DONE,
    }
}

/// The line standard error gets for `outcome`, without the program's name
/// and without a line feed; None where it gets none. The names of the
/// bounds are the Python result's `stopped`.
#[must_use]
pub fn diagnostic(outcome: &Outcome) -> Option<String> {
    if let Some(stop) = outcome.stopped {
        let bound = stop.name();
        return Some(format!("stopped: {bound}"));
    }
    match outcome.certified {
        None => None,
        Some(true) => Some("checker: certified".to_string()),
        Some(false) => Some(match &outcome.flaw {
            Some(flaw) => format!("checker: rejected: {flaw}"),
            None => "checker: rejected".to_string(),
        }),
    }
}
