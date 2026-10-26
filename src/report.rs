//! The call report (`Params::report`): one text line per event of a call,
//! in the order the call met them, and the times of the stages of the
//! round, or other step, in progress. The report reads and never decides:
//! no hook draws from a generator or changes the order of work.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__cpuid;
use std::fmt::{self, Write};
use std::time::{Duration, Instant};

use crate::ReportSink;

// call-report.5 cited: pearce-2026 §2.2, Table 1 -- the share of F4's time spent in select, symbolic, encode, reduce, backsub, decode, and update differs by instance, symbolic 15.4% to 58.9% and reduce 0.8% to 62.0%; §2.1 outlines a round in those stages
/// A stage of a step of the call whose time the report gives. A round's
/// are `Select` to `Update`, in the order a round runs them, and `Record`
/// before `Reduce` as well (call-report.22).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// The pairs selected and their rows, or the signature run's
    /// reductants.
    Select,
    /// Symbolic preprocessing: the rows' products and the reducers, or a
    /// replay's rows multiplied out.
    Symbolic,
    /// The columns sorted and mapped (`matrix::columns`).
    Encode,
    /// The reduction but its back-substitution.
    Reduce,
    Backsub,
    // call-report.6 design -- the trace's and the learner's hooks of a round are a stage of their own, not folded into decode (not measured)
    /// The trace's and the learner's hooks of the round's matrix, and the
    /// rows that counted listed and sorted for the trace (f4-gfp.52).
    Record,
    /// The made rows as polynomials with their cofactors composed, and in
    /// the signature run the zero rows' syzygy signatures and the
    /// handover's tests.
    Decode,
    /// The new elements entered, with their pairs.
    Update,
    /// The generators read into a run's table.
    Read,
    /// The generators put in reduced row echelon form (`echelon::echelon`).
    Echelon,
    /// FGLM's staircase.
    Find,
    /// FGLM's change of the basis.
    Change,
    /// A run's image for the lift.
    Image,
    /// A text written.
    Text,
    /// The cofactor text over ℚ, which the lift writes element by element
    /// beside the basis text (`Text`).
    Cofactors,
    /// The checker.
    Check,
    // call-report.15 cited: monagan-2015 §2.6, Table 4 -- the time over ℚ broken into the Gröbner bases modulo p, the Chinese remaindering, and the rational reconstruction; parisse-2019 §2 -- the lift is a loop of three steps, a basis modulo a prime, Chinese remaindering, and reconstruction attempts
    /// A prime's image merged into the terms of the lift's open elements.
    Merge,
    /// The lift's attempts at the rationals of its open elements.
    Reconstruct,
    /// The Chinese remainder step of a prime, which makes its constants of
    /// the conversion, and the append of its image to the open elements'
    /// images (q-lift-growth-time.11).
    Crt,
    /// The lifted basis checked against the image at the verification's
    /// prime.
    Verify,
}

/// The stages.
const STAGES: usize = Stage::Verify as usize + 1;

// call-report.8 design -- a time is seconds with six decimals, not integer nanoseconds (not measured)
/// A duration as the report writes it: seconds, a point, and six decimals,
/// truncated to the microsecond.
pub(crate) struct Secs(pub(crate) Duration);

impl fmt::Display for Secs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let us = self.0.as_micros();
        write!(
            f,
            "{secs}.{micros:06}",
            secs = us / 1_000_000,
            micros = us % 1_000_000
        )
    }
}

/// ` key=value`, or nothing where the value is None: a field that does not
/// apply is left out.
pub(crate) struct Opt<'a, T>(pub(crate) &'a str, pub(crate) Option<T>);

impl<T: fmt::Display> fmt::Display for Opt<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.1 {
            Some(v) => write!(f, " {key}={v}", key = self.0),
            None => Ok(()),
        }
    }
}

/// `on` or `off`.
pub(crate) fn on(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

/// The processor's brand string from CPUID, its words joined by `_` so
/// the field holds no space; None off x86-64 or where CPUID has no brand.
pub(crate) fn cpu() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return None;
        }
        let bytes: Vec<u8> = (0x8000_0002..=0x8000_0004)
            .map(__cpuid)
            .flat_map(|r| [r.eax, r.ebx, r.ecx, r.edx])
            .flat_map(u32::to_le_bytes)
            .collect();
        let brand = String::from_utf8_lossy(&bytes);
        let words: Vec<&str> = brand.trim_matches('\0').split_whitespace().collect();
        if words.is_empty() {
            None
        } else {
            Some(words.join("_"))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// What a round's line holds besides its counts' growth and its stage
/// times, as the run knows it once `Stats` has counted the round.
pub(crate) struct Round {
    pub(crate) number: u32,
    pub(crate) degree: u32,
    pub(crate) pairs: usize,
    pub(crate) rows: usize,
    pub(crate) given: usize,
    pub(crate) columns: usize,
    pub(crate) entries: usize,
    pub(crate) pivots: usize,
    pub(crate) zero: u64,
    /// None where the plain run combined rows, which names none.
    pub(crate) zero_rows: Option<usize>,
    pub(crate) reductions: u64,
    pub(crate) bands: bool,
    /// The signature run's pending signatures taken since its last round,
    /// and those of them that gave no reductant.
    pub(crate) pending: Option<(usize, usize)>,
    /// The plain run's pairs left after the round: after its update, or
    /// after its selection where it made a constant.
    pub(crate) left: Option<usize>,
}

// call-report.1 design -- the report is text, a line per event in the order met: seconds since the call's start, a kind, and key=value fields, not JSON Lines nor a typed list of events in the outcome (not measured)
/// The report of one call: its lines so far, and the stage times of the
/// step in progress.
pub(crate) struct Report {
    /// The call's start, `entry::call`'s.
    start: Instant,
    /// The lines so far.
    text: String,
    /// Takes each line as it is written, until a write fails.
    sink: Option<ReportSink>,
    /// Where the last stage ended.
    last: Instant,
    /// The stage times of the step in progress, by `Stage`.
    times: [Duration; STAGES],
    /// The budget's counts, subtracted and added, when the step opened.
    counts: (u64, u64),
    /// The current run's `run` line.
    run: Instant,
    /// The runs started so far.
    runs: u32,
    /// A round `Stats` has counted whose line is not written yet, with its
    /// counts' growth.
    round: Option<(Round, (u64, u64))>,
    /// The resident set in bytes when the report began.
    rss: usize,
}

impl Report {
    /// The report of a call that started at `start`, with no line, each
    /// line also written to `sink` where there is one.
    pub(crate) fn new(start: Instant, sink: Option<ReportSink>) -> Report {
        Report {
            start,
            text: String::new(),
            sink,
            last: start,
            times: [Duration::ZERO; STAGES],
            counts: (0, 0),
            run: start,
            runs: 0,
            round: None,
            rss: crate::blocks::rss().0,
        }
    }

    /// The resident set in bytes when the report began, and the process's
    /// peak resident set now.
    pub(crate) fn rss(&self) -> (usize, usize) {
        (self.rss, crate::blocks::rss().1)
    }

    /// Writes the line of kind `kind`: the time since the call's start, the
    /// kind, and `fields`, which the caller formats as ` key=value` pairs.
    /// A counted round whose line is not written yet is written first.
    pub(crate) fn line(&mut self, kind: &str, fields: fmt::Arguments<'_>) {
        self.flush();
        self.write(kind, fields);
    }

    fn write(&mut self, kind: &str, fields: fmt::Arguments<'_>) {
        let time = Secs(self.start.elapsed());
        let from = self.text.len();
        #[allow(
            clippy::let_underscore_must_use,
            reason = "a String takes every write, and no Display written here fails"
        )]
        let _: fmt::Result = writeln!(self.text, "{time} {kind}{fields}");
        if let Some(sink) = &self.sink
            && sink.put(&self.text[from..]).is_err()
        {
            self.sink = None;
        }
    }

    /// Writes the `run` line of the next run, its number first, and starts
    /// its time.
    pub(crate) fn run(&mut self, fields: fmt::Arguments<'_>) {
        self.flush();
        self.runs += 1;
        self.run = Instant::now();
        let number = self.runs;
        self.write("run", format_args!(" number={number}{fields}"));
    }

    /// The current run's number.
    pub(crate) fn runs(&self) -> u32 {
        self.runs
    }

    /// The time since the current run's `run` line.
    pub(crate) fn since_run(&self) -> Duration {
        self.run.elapsed()
    }

    /// The time since the call's start.
    pub(crate) fn since_start(&self) -> Duration {
        self.start.elapsed()
    }

    /// Opens a step: every stage time zero, its start now, and the
    /// budget's counts `counts` noted.
    pub(crate) fn open(&mut self, counts: (u64, u64)) {
        self.times = [Duration::ZERO; STAGES];
        self.counts = counts;
        self.last = Instant::now();
    }

    // call-report.7 design -- a stage's time is the time between two marks, each a read of the monotonic clock, not the CPU's cycle counter nor sampling (not measured)
    /// Ends `stage`: the time since the last mark, or since `open`, is
    /// added to it.
    pub(crate) fn mark(&mut self, stage: Stage) {
        let now = Instant::now();
        self.times[stage as usize] += now - self.last;
        self.last = now;
    }

    /// The time of `stage` in the open step.
    pub(crate) fn time(&self, stage: Stage) -> Duration {
        self.times[stage as usize]
    }

    /// How much the budget's counts `counts` grew since the step opened.
    pub(crate) fn grown(&self, counts: (u64, u64)) -> (u64, u64) {
        (counts.0 - self.counts.0, counts.1 - self.counts.1)
    }

    // call-report.19 design -- every round Stats counts gets its line, a round a halt ends in its decode or its update before the next line, not no line for such a round (not measured)
    /// Notes the round `Stats` has counted, with the budget's counts
    /// `counts`: its line is written by `end_round`, or before the next
    /// line where a halt ends the round first.
    pub(crate) fn counted(&mut self, round: Round, counts: (u64, u64)) {
        self.flush();
        let grown = self.grown(counts);
        self.round = Some((round, grown));
    }

    /// Writes the counted round's line, with `left` the pairs the update
    /// left, or the selection where the round made a constant, where the run
    /// is the plain run.
    pub(crate) fn end_round(&mut self, left: Option<usize>) {
        if let Some((round, _)) = &mut self.round {
            round.left = left;
        }
        self.flush();
    }

    /// Writes the counted round's line where there is one, its stages' times
    /// as they stand.
    fn flush(&mut self) {
        let Some((r, (subtracted, added))) = self.round.take() else {
            return;
        };
        let t = self.times.map(Secs);
        let pending = r.pending.map(|p| p.0);
        let dropped = r.pending.map(|p| p.1);
        self.write(
            "round",
            format_args!(
                " number={number} degree={degree} pairs={pairs} rows={rows} given={given} \
                 columns={columns} entries={entries} pivots={pivots} zero={zero}{zero_rows} \
                 reductions={reductions} subtracted={subtracted} added={added} \
                 bands={bands}{pending}{dropped}{left} select={select} symbolic={symbolic} \
                 encode={encode} reduce={reduce} backsub={backsub} record={record} \
                 decode={decode} update={update}",
                number = r.number,
                degree = r.degree,
                pairs = r.pairs,
                rows = r.rows,
                given = r.given,
                columns = r.columns,
                entries = r.entries,
                pivots = r.pivots,
                zero = r.zero,
                zero_rows = Opt("zero-rows", r.zero_rows),
                reductions = r.reductions,
                bands = if r.bands { "yes" } else { "no" },
                pending = Opt("pending", pending),
                dropped = Opt("dropped", dropped),
                left = Opt("left", r.left),
                select = t[Stage::Select as usize],
                symbolic = t[Stage::Symbolic as usize],
                encode = t[Stage::Encode as usize],
                reduce = t[Stage::Reduce as usize],
                backsub = t[Stage::Backsub as usize],
                record = t[Stage::Record as usize],
                decode = t[Stage::Decode as usize],
                update = t[Stage::Update as usize],
            ),
        );
    }

    /// The lines, in the order written.
    pub(crate) fn finish(mut self) -> String {
        self.flush();
        self.text
    }
}
