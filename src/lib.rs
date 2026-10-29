//! grobx, a Gröbner basis library.
//!
//! The contracts of the core are public modules: the field, the monomial
//! format, the monomial order, the module of the rows, the cofactor
//! tracking, and the trace.

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::let_underscore_must_use,
        clippy::let_underscore_untyped,
        clippy::many_single_char_names,
        clippy::maybe_infinite_iter,
        clippy::panic,
        clippy::too_many_lines,
        clippy::undocumented_unsafe_blocks,
        clippy::unreachable,
        clippy::unwrap_used,
        clippy::wildcard_enum_match_arm,
        reason = "test code, where these lints carry no weight"
    )
)]

#[cfg(test)]
mod attempt_tests;
#[cfg(test)]
mod band_tests;
mod blocks;
mod budget;
#[cfg(test)]
mod build_tests;
#[cfg(test)]
mod catch_up_tests;
#[cfg(test)]
mod chain_tests;
#[cfg(test)]
mod child_tests;
#[cfg(test)]
mod cofactor_stop_tests;
pub mod cofactors;
#[cfg(test)]
mod combine_tests;
#[cfg(test)]
mod cyclic_gap_tests;
#[cfg(test)]
mod denominator_tests;
#[cfg(test)]
mod divisibility_tests;
#[cfg(test)]
mod early_handover_tests;
mod echelon;
#[cfg(test)]
mod echelon_replay_tests;
mod entry;
mod f4;
mod fglm;
#[cfg(test)]
mod fglm_tests;
pub mod field;
#[cfg(test)]
mod format_tests;
#[cfg(test)]
mod handover_tests;
#[cfg(test)]
mod handover_time_tests;
pub mod id;
#[cfg(test)]
mod input_redundancy_tests;
#[cfg(test)]
mod kept_tests;
#[cfg(test)]
mod kept_unchecked_tests;
#[cfg(test)]
mod key_tests;
#[cfg(test)]
mod later_runs_tests;
#[cfg(test)]
mod learn_pair_tests;
mod lehmer;
#[cfg(test)]
mod lehmer_tests;
mod lift;
#[cfg(test)]
mod lift_growth_tests;
#[cfg(test)]
mod lift_rest_tests;
#[cfg(test)]
mod lift_tests;
#[cfg(test)]
mod lookup_tests;
#[cfg(test)]
mod many_variables_tests;
#[cfg(test)]
mod mask_scan_tests;
mod matrix;
#[cfg(test)]
mod matrix_stop_tests;
#[cfg(test)]
mod misfit_loop_tests;
#[cfg(test)]
mod mixed_radix_tests;
pub mod module;
#[cfg(test)]
mod modulus_tests;
pub mod monomial;
#[cfg(test)]
mod mqrr_tests;
mod narrow;
pub mod order;
mod os;
#[cfg(test)]
mod outgrown_round_tests;
#[cfg(test)]
mod output_tests;
#[cfg(test)]
mod packed_tests;
#[cfg(test)]
mod pair_tests;
#[cfg(test)]
mod pair_work_ratio_tests;
#[cfg(test)]
mod pair_work_tests;
mod pairs;
#[cfg(test)]
mod params_tests;
#[cfg(test)]
mod parsed_tests;
#[cfg(test)]
mod plain_band_tests;
#[cfg(test)]
mod plain_update_tests;
#[cfg(test)]
mod plain_zero_tests;
mod poly;
#[cfg(test)]
mod prefix_tests;
#[cfg(test)]
mod prime_count_tests;
#[cfg(test)]
mod q_basis_text_tests;
#[cfg(test)]
mod radix_tests;
mod rational;
#[cfg(test)]
mod rational_tests;
#[cfg(test)]
mod read_tests;
#[cfg(test)]
mod record_tests;
mod report;
#[cfg(test)]
mod report_lift_tests;
#[cfg(test)]
mod report_tests;
#[cfg(test)]
mod round_memory_tests;
#[cfg(test)]
mod row_scan_tests;
mod shape;
#[cfg(test)]
mod shape_tests;
#[cfg(test)]
mod shortcut_tests;
mod sig;
#[cfg(test)]
mod sig_tests;
#[cfg(test)]
mod signature_zero_tests;
#[cfg(test)]
mod slack_tests;
#[cfg(all(test, target_os = "linux"))]
mod small_block_tests;
#[cfg(test)]
#[path = "../tests/common/systems.rs"]
mod systems;
mod table;
pub mod text;
#[cfg(test)]
mod text_copy_peak_tests;
#[cfg(test)]
mod text_stop_tests;
#[cfg(test)]
mod text_time_tests;
pub mod trace;
#[cfg(test)]
mod zero_rows_tests;

use std::fmt;
use std::hash::{BuildHasher, RandomState};
use std::io::Write;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

pub use blocks::{Allocator, MMAP_THRESHOLD, SMALL, TOP_PAD};
pub use order::MonomialOrder;

// small-block-time.4 design -- the crate's unit tests set the crate's allocator as their global allocator, not the crate setting it for every program that links it (not measured)
#[cfg(test)]
#[global_allocator]
static GLOBAL: Allocator = Allocator;

/// The crate version, which the Python module carries as `__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The reduced Gröbner basis of the instance `text` under `params`, as the
/// canonical basis text, computed by F4, and in lex by F4 in grevlex and
/// FGLM where the ideal is zero-dimensional and its degree at most
/// `fglm::MAX_STAIRCASE` (`MonomialOrder::Lex`). Over Q,
/// characteristic 0, the call learns at two primes below 2^31 at once and
/// replays at
/// primes of [2^61, 2^62), drawn from its seed, and lifts the replays'
/// bases to Q by the Chinese remainder theorem
/// and rational reconstruction; it returns the lifted basis once the basis
/// at one more prime agrees with it, the learn run's second, or a fresh
/// prime where the learn run fell back to its first prime alone or the
/// second divides a denominator of the lifted basis, and the checker, where
/// the call asks for it, rules on it with the lifted cofactors.
///
/// The first call of a process, of this entry point or of any other, sets
/// glibc's `M_TOP_PAD` to `TOP_PAD` and `M_MMAP_THRESHOLD` to
/// `MMAP_THRESHOLD` for the process, which keeps them after the call.
///
/// # Errors
///
/// Every `Error` but `Misfit`: a text that breaks the grammar, a
/// characteristic or order that does not fit, a degree overflow, a count
/// past what a u32 id numbers, a denominator with no inverse, a memory
/// ceiling with no resident set, over Q the lift's drops, and a `mallopt`
/// that refused the first call's parameters.
pub fn groebner(text: &str, params: &Params) -> Result<Outcome, Error> {
    entry::groebner(text, params, &mut trace::NoTrace, &mut trace::NoLearner)
}

/// As `groebner`, and `trace` hears of every matrix the run's rounds and
/// its inter-reduction reduce, not of the generators' echelon form. A lex
/// call's trace hears the grevlex run's matrices, then the lex run's where
/// the call starts over in lex. A call that starts over in a wider monomial
/// format (`monomial`) hears every run's matrices in turn, the run the
/// overflow stopped included, and so does a call whose signature run hands
/// over (`Params::signatures`). A call over Q hears the matrices of every
/// run in turn, a pair learn's and a pair replay's once for both their
/// primes. A run whose trace listens
/// (`trace::Trace::LISTENS`) reduces each row of its matrices on its own.
///
/// # Errors
///
/// As `groebner`.
pub fn groebner_with<T: trace::Trace>(
    text: &str,
    params: &Params,
    trace: &mut T,
) -> Result<Outcome, Error> {
    entry::groebner(text, params, trace, &mut trace::NoLearner)
}

// trace-replay.13 cited: demin-2024 §3 -- Groebner.jl learns the trace of a computation once and applies it by a function of its own, groebner_apply!
/// As `groebner_with`, and, where the run returns a basis, the record of
/// its matrices that `replay` replays over another prime. A lex call's
/// record is that of the run that made the basis: the grevlex run's where
/// FGLM changed its basis (`trace::Record::changed`), else the lex run's. A
/// call that starts over in a wider monomial format records its last run,
/// and one whose signature run hands over the run it hands over to.
/// A learn run reduces each row of its matrices on its own, save that
/// where its trace does not listen and it tracks no cofactors, its plain
/// run rules groups of rows zero by tests of random combinations where p
/// is above 3, and the record names those rows as rows that reduced to
/// zero (`Stats`). Over GF(p) alone: characteristic 0 is
/// `Error::Characteristic`.
///
/// # Errors
///
/// As `groebner`, and `Error::Characteristic` for characteristic 0.
pub fn learn<T: trace::Trace>(
    text: &str,
    params: &Params,
    trace: &mut T,
) -> Result<(Outcome, Option<trace::Record>), Error> {
    let mut recording = trace::Recording::new();
    let outcome = entry::groebner(text, params, trace, &mut recording)?;
    // trace-replay.15 design -- a learn run that returns no basis returns no record, not the record of the rounds that finished (not measured)
    let record = outcome
        .basis
        .is_some()
        .then(|| recording.finish(params.order.clone()));
    Ok((outcome, record))
}

/// The reduced Gröbner basis of `text` over its own prime, from the rows
/// `record` holds, as `groebner` returns it; `trace` hears of every matrix
/// the replay reduces. `Params::signatures` does not matter to a replay.
/// A changed record replays its grevlex steps and changes the basis to lex.
/// `Error::Misfit` where the record does not fit, `Error::Misfit { step: 0
/// }` where it was learned in another order than `Params::order` names, or
/// with the echelon where the replay tracks cofactors.
/// Over GF(p) alone: characteristic 0 is `Error::Characteristic`.
///
/// # Errors
///
/// As `learn`, and `Error::Misfit` where the record does not fit.
pub fn replay<T: trace::Trace>(
    text: &str,
    params: &Params,
    record: &trace::Record,
    trace: &mut T,
) -> Result<Outcome, Error> {
    entry::replay(text, params, record, trace)
}

/// SHA-256 of the text, 64 lowercase hex digits.
#[must_use]
pub fn digest(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for b in Sha256::digest(text.as_bytes()) {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 15) as usize] as char);
    }
    out
}

/// The parameters of a call (`OBJECTIVES.md`, _Call parameters_).
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool is a switch of the call, which a caller sets by name"
)]
pub struct Params {
    /// The monomial order. The first variable is the largest in each. Over
    /// Q each prime's run computes in the order as it does over GF(p).
    pub order: MonomialOrder,
    pub threads: Threads,
    /// The one source of every random choice in the call.
    pub seed: u64,
    /// A time limit on the whole call, from its start.
    pub deadline: Option<Duration>,
    pub cancel: Option<Cancel>,
    /// A ceiling on the process's resident set, in bytes.
    pub memory: Option<u64>,
    pub checker: Option<CheckerMode>,
    /// Whether the outcome carries the cofactors.
    pub cofactors: bool,
    // signature-pruning.18 design -- a switch keeps f4-gfp's engine beside the signature run (not measured)
    /// Whether the run prunes pairs by the signatures of its rows. On by
    /// default: the signature run, which may hand the call over to the
    /// engine of f4-gfp, started over, where the run tracks no cofactors:
    /// over GF(p) where the call asks for no cofactors and no checker, and
    /// over Q in every learn run and in the verification's run. It decides
    /// at the first round that makes an element whose signature degree
    /// passes its degree by 2 or more, where that round made 25 elements or
    /// more, or 100 where the plain run would reduce each row alone, at the
    /// first by 4 or more, at every round that made an element, and at
    /// every round whose update would form 1,000 S-pairs or more
    /// (early-handover-time.1, .2, and .5 to .7, threshold-recheck.16,
    /// input-redundancy-time.7, handover-time.1 to .5, .7, and .8,
    /// outgrown-round-size.1 to .5, pair-work-handover.1, .2, and .4 to .6,
    /// pair-work-ratio.1 and .2). Off runs
    /// the engine of f4-gfp. In lex it decides the
    /// grevlex run's engine, as in grevlex. In a block order, and in lex
    /// where F4 runs in lex, the call runs the engine of f4-gfp, whatever
    /// this says.
    pub signatures: bool,
    // input-redundancy-time.8 design -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
    /// Whether the runs take the generators in reduced row echelon form, as
    /// vectors over their monomials (`echelon::echelon`). On by default. A
    /// run that tracks cofactors takes them as read whatever this says, and
    /// over Q so does every run of a call that asks for cofactors or the
    /// checker. A replay that tracks none takes them as its record's learn
    /// run took them, whatever this says.
    pub echelon: bool,
    /// Whether the outcome carries the call report (`Outcome::report`).
    pub report: bool,
    /// Where `report` is on, a writer that takes each report line as the
    /// call meets it, flushed after the line, so a call that never returns
    /// leaves its report so far. The first write error ends the writing;
    /// `Outcome::report` holds every line either way.
    pub report_sink: Option<ReportSink>,
}

impl Params {
    /// The order, threads, and seed as given, every bound absent, no checker, no
    /// cofactors, signatures on, the echelon on, and no report.
    #[must_use]
    pub fn new(order: MonomialOrder, threads: Threads, seed: u64) -> Params {
        Params {
            order,
            threads,
            seed,
            deadline: None,
            cancel: None,
            memory: None,
            checker: None,
            cofactors: false,
            signatures: true,
            echelon: true,
            report: false,
            report_sink: None,
        }
    }
}

impl Default for Params {
    /// Grevlex, all the machine's threads, and a seed from `random_seed`,
    /// which `seed` holds for a replay; the rest as `Params::new` sets it.
    fn default() -> Params {
        Params::new(MonomialOrder::Grevlex, Threads::All, random_seed())
    }
}

/// A seed from the operating system's random source, which `RandomState`
/// takes its keys from.
#[must_use]
pub fn random_seed() -> u64 {
    RandomState::new().hash_one(())
}

/// How many threads a call may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    /// All the machine has.
    All,
    Count(NonZeroUsize),
}

/// How the checker rules on the ⟨G⟩ ⊆ ⟨F⟩ direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckerMode {
    Exact,
    /// Both sides evaluated at `points` random points.
    Cheap {
        points: NonZeroU32,
    },
}

/// A flag another thread raises to stop a call.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    #[must_use]
    pub fn new() -> Cancel {
        Cancel::default()
    }

    pub fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    #[must_use]
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// The writer `Params::report_sink` names.
#[derive(Clone)]
pub struct ReportSink(Arc<Mutex<dyn Write + Send>>);

impl ReportSink {
    pub fn new(writer: impl Write + Send + 'static) -> ReportSink {
        ReportSink(Arc::new(Mutex::new(writer)))
    }

    /// Writes `line` and flushes.
    pub(crate) fn put(&self, line: &str) -> std::io::Result<()> {
        let mut w = self
            .0
            .lock()
            .map_err(|_| std::io::Error::other("report sink poisoned"))?;
        w.write_all(line.as_bytes())?;
        w.flush()
    }

    /// As `put`, where the writer is free: the allocator writes its lines
    /// (`blocks::heard`) by this, since its thread may hold the writer as
    /// the request comes, and waiting for it would never end. A busy
    /// writer is `None`.
    pub(crate) fn try_put(&self, line: &str) -> Option<std::io::Result<()>> {
        let mut w = self.0.try_lock().ok()?;
        Some(w.write_all(line.as_bytes()).and_then(|()| w.flush()))
    }
}

impl fmt::Debug for ReportSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ReportSink")
    }
}

/// What a call returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The canonical basis text, None where a bound stopped the run.
    pub basis: Option<String>,
    pub stopped: Option<Stop>,
    /// The cofactor matrix as text, where the call asked for it.
    pub cofactors: Option<String>,
    /// The checker's verdict, None where it did not rule: the call did not
    /// ask for it, or a bound stopped the call.
    pub certified: Option<bool>,
    /// The checker's reason where it rejected the result.
    pub flaw: Option<String>,
    /// What the run did; zero where the run did not reach a round.
    pub stats: Stats,
    /// The call report where the call asked for it (`Params::report`): a
    /// line per event of the call, in the order the call met them, from
    /// its `call` line to its `end` line. A call that returns an error
    /// returns no report.
    pub report: Option<String>,
}

// signature-pruning.19 design -- the outcome carries the counts of its rounds (not measured)
/// What a run did, counted over its rounds; the inter-reduction's matrix is
/// not counted. A run a bound stopped counts the rounds whose matrices it
/// reduced. A lex
/// call counts the run that made its basis: the grevlex run where FGLM
/// changed its basis, the lex run where the call started over. A call that
/// starts over in a wider monomial format counts its last run, and one
/// whose signature run hands over the run it hands over to. A call over Q
/// counts the learn run whose record its replays replayed, and one a bound
/// stopped the run the bound stopped, or that learn run where the stop came
/// between two primes or after the last, and zero where no record stood at
/// the stop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The matrices the rounds reduced.
    pub rounds: u32,
    /// The reductions run: each row on its own where the run replays a
    /// record, the trace listens, or the call tracks cofactors, and in a
    /// learn's signature run. A learn's plain run reduces its rows and the
    /// random combinations of its tests where p, the first lane's prime, is
    /// above 3. Else the plain run reduces random combinations of rows
    /// where p is 46341 or more, and the signature run its rows and the
    /// random combinations of its tests where the field has one lane and p
    /// is above 3, each combination a reduction.
    pub reduced: u64,
    /// The reductions of those that reduced to zero.
    pub zero: u64,
}

/// The bound that stopped a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Deadline,
    Cancel,
    Memory,
}

impl Stop {
    /// The bound's name, as the call report, the CLI, and the Python
    /// module write it: "deadline", "cancel", or "memory".
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Stop::Deadline => "deadline",
            Stop::Cancel => "cancel",
            Stop::Memory => "memory",
        }
    }
}

/// Why a call returns no outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The instance text breaks the grammar at `line`, counted from 1.
    Syntax {
        line: usize,
        expected: &'static str,
    },
    UnknownVariable {
        line: usize,
        name: String,
    },
    RepeatedVariable {
        name: String,
    },
    /// Neither 0, for Q, nor a prime below 2^64; or 0 where the call learns
    /// or replays.
    Characteristic {
        value: String,
    },
    /// A degree past the widest monomial format, 2^31 − 1: in the instance
    /// at `line`, or in the run (`None`), the cofactors included.
    Overflow {
        line: Option<usize>,
    },
    /// More generators, terms, numeric factors, or monomials in the
    /// instance, or monomials, rows, terms, elements, or record steps in
    /// the run, than a u32 id numbers, 2^32 − 1.
    Size,
    // q-zero-denominator.1 design -- over Q a denominator 0 is NoInverse, the error a denominator 0 mod p is over GF(p), not a variant of its own or a syntax error (not measured)
    /// A denominator at `line` that is 0 mod p over GF(p), or 0 over Q.
    NoInverse {
        line: usize,
    },
    /// `name` names no order of the family (`MonomialOrder::from_str`), or
    /// it is a block order whose sizes do not fit the instance's variables,
    /// which the call finds once the variables are read.
    Order {
        name: String,
    },
    /// A memory ceiling where the resident set cannot be read.
    Memory,
    // trace-replay.7 design -- a replay that does not fit is an error that names its step, not an outcome without a basis whose field says why (not measured)
    /// A replay's record does not fit the instance over its prime: at the
    /// inputs, under another order than the record's, or where the replay
    /// tracks cofactors of a record whose learn run took the echelon, step
    /// 0, or at the step of the record counted from 1. A
    /// changed record whose replayed basis FGLM cannot change does not fit
    /// at the step one past its last.
    Misfit {
        step: u32,
    },
    // lift-misfit-loop.5 design -- the lift's drops at the bound are an error of their own that counts them by why, not Error::Misfit again, an outcome without a basis whose stopped names a new Stop, or the variant without its counts (not measured)
    /// The lift over Q dropped `DROPS` records in the call: `misfits` after
    /// two misfits in a row, `leads` at a verification whose leading
    /// monomials differ, and `verification` at one at which an element
    /// differs. Random primes make it improbable, and a defect of the lift
    /// makes it certain.
    Drops {
        misfits: u32,
        leads: u32,
        verification: u32,
    },
    /// glibc's `mallopt` refused the value the first call of the process
    /// sets for `param`, `"M_TOP_PAD"` or `"M_MMAP_THRESHOLD"` (`TOP_PAD`,
    /// `MMAP_THRESHOLD`), and every later call of the process returns it
    /// too. glibc takes both values, so a malloc that is not glibc's makes
    /// it.
    Malloc {
        param: &'static str,
    },
}

impl Error {
    /// The variant's name, which the call report's `error` line and the
    /// Python module's `grobx.Error` give.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Error::Syntax { .. } => "Syntax",
            Error::UnknownVariable { .. } => "UnknownVariable",
            Error::RepeatedVariable { .. } => "RepeatedVariable",
            Error::Characteristic { .. } => "Characteristic",
            Error::Overflow { .. } => "Overflow",
            Error::Size => "Size",
            Error::NoInverse { .. } => "NoInverse",
            Error::Order { .. } => "Order",
            Error::Memory => "Memory",
            Error::Misfit { .. } => "Misfit",
            Error::Drops { .. } => "Drops",
            Error::Malloc { .. } => "Malloc",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Syntax { line, expected } => write!(f, "line {line}: expected {expected}"),
            Error::UnknownVariable { line, name } => {
                write!(f, "line {line}: unknown variable {name}")
            }
            Error::RepeatedVariable { name } => write!(f, "variable {name} named twice"),
            Error::Characteristic { value } => write!(
                f,
                "characteristic {value}: the engine takes 0 for Q and the primes below 2^64, \
                 and learns and replays over the primes alone"
            ),
            Error::Overflow { line: Some(line) } => {
                write!(f, "line {line}: a degree past the widest monomial format")
            }
            Error::Overflow { line: None } => write!(f, "a degree past the widest monomial format"),
            Error::Size => write!(
                f,
                "more generators, terms, numeric factors, monomials, rows, elements, or record \
                 steps than a u32 id numbers, 2^32 - 1"
            ),
            Error::NoInverse { line } => {
                write!(f, "line {line}: a denominator that is 0, or 0 mod p")
            }
            Error::Order { name } => {
                write!(
                    f,
                    "monomial order {name:?}: no order of the family on these variables"
                )
            }
            Error::Memory => write!(f, "a memory ceiling where the resident set cannot be read"),
            Error::Misfit { step } => write!(f, "the record does not fit at step {step}"),
            Error::Drops {
                misfits,
                leads,
                verification,
            } => {
                let n = u64::from(*misfits) + u64::from(*leads) + u64::from(*verification);
                write!(
                    f,
                    "the lift over Q dropped {n} records: {misfits} after two misfits in a row, \
                     {leads} at a verification whose leading monomials differ, and \
                     {verification} at one at which an element differs"
                )
            }
            Error::Malloc { param } => write!(
                f,
                "glibc's mallopt refused the value of {param} that the first call of the \
                 process sets"
            ),
        }
    }
}

impl std::error::Error for Error {}
