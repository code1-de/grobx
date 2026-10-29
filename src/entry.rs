//! The entry point: reads the instance's header and the parameters, picks
//! the monomorphization of the run, the field's, the order's, and the
//! monomial format's among them, and hands the result to the checker
//! where the call asks for it. A run whose degree passes its format
//! starts over in the next wider,
//! and a signature run that hands over starts over in the plain run. A run
//! for the lift over Q returns its result as an image (`lift::Image`), or
//! over a field of more than one lane an image per lane, without
//! cofactors. Where the call asks for the call report, the entry point
//! writes its lines of the call, of each run's start, inputs, echelon, and
//! end, and of the image, the texts, FGLM, and the checker.

use std::fmt;
use std::time::Instant;

use grobx_checker::Mode;
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::blocks;
use crate::budget::Budget;
use crate::cofactors::{Cofactors, NoCofactors, Tracked};
use crate::f4::{Elements, HANDOVER, Halt};
use crate::fglm;
use crate::field::Field;
use crate::id::MonoId;
use crate::lift::{self, Image, PairImage, Primes, Report, Sparse};
use crate::matrix;
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::narrow;
use crate::order::{Block, Grevlex, Lex, MonomialOrder, Order};
use crate::poly::Poly;
use crate::report::{self as call_report, Opt, Secs, Stage, cpu, field_value, on};
use crate::shape::{Built, Kept, Shape, Shaping};
use crate::sig::Pairs;
use crate::table::Table;
use crate::text::{Header, Prime, write_basis};
use crate::trace::{Learner, NoLearner, Record, Trace};
use crate::{CheckerMode, Error, Outcome, Params, Stats, Stop, Threads};

/// A format of the family by the width of its lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Width {
    W8,
    W16,
    W32,
}

impl Width {
    /// The width of a lane in bits.
    fn bits(self) -> u32 {
        match self {
            Width::W8 => 8,
            Width::W16 => 16,
            Width::W32 => 32,
        }
    }

    /// The next wider width, None for W32.
    pub(crate) fn wider(self) -> Option<Width> {
        match self {
            Width::W8 => Some(Width::W16),
            Width::W16 => Some(Width::W32),
            Width::W32 => None,
        }
    }
}

/// The outcome of a run in the widths W8 to W32; `learner` hears what the
/// run records.
pub(crate) fn groebner<T: Trace, L: Learner>(
    text: &str,
    params: &Params,
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    groebner_in(text, params, Width::W8, Width::W32, trace, learner)
}

// monomial-formats.14 design -- the widths a call may take are a crate-internal parameter of the entry point, not a field of Params (not measured)
/// As `groebner`, in the widths from `first` to `last` over GF(p): the
/// call starts in `first`, and over in the next wider width where a degree
/// passes the one it runs in. `Error::Overflow` where a degree passes
/// `last`. Over Q the lift's runs take every width, whatever these say.
pub(crate) fn groebner_in<T: Trace, L: Learner>(
    text: &str,
    params: &Params,
    first: Width,
    last: Width,
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    call(text, params, None, (first, last), trace, learner)
}

/// The outcome of a replay of `record`, in the widths W8 to W32.
pub(crate) fn replay<T: Trace>(
    text: &str,
    params: &Params,
    record: &Record,
    trace: &mut T,
) -> Result<Outcome, Error> {
    replay_in(text, params, record, Width::W8, Width::W32, trace)
}

/// As `replay`, in the widths from `first` to `last`.
pub(crate) fn replay_in<T: Trace>(
    text: &str,
    params: &Params,
    record: &Record,
    first: Width,
    last: Width,
    trace: &mut T,
) -> Result<Outcome, Error> {
    call(
        text,
        params,
        Some(record),
        (first, last),
        trace,
        &mut NoLearner,
    )
}

/// The outcome of a run, or of a replay where there is a record, in the
/// widths from `first` to `last`. Over Q, the outcome of the lift
/// (`lift::lift`), whose runs take the widths W8 to W32;
/// `Error::Characteristic` where there is a record or `learner` listens.
/// It sets glibc's parameters first (`blocks::pad_heap`), so that the
/// call's runs, the lift's among them, run on the padded heap, and a
/// refusal returns before the call allocates. An error ends the call
/// report with its `error` line (`report_error`).
fn call<T: Trace, L: Learner>(
    text: &str,
    params: &Params,
    record: Option<&Record>,
    (first, last): (Width, Width),
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    debug_assert!(first <= last);
    let start = Instant::now();
    let mut held: Option<Budget> = None;
    let called = (|| {
        blocks::pad_heap()?;
        let _watch = match &params.report_sink {
            Some(sink) if params.report => Some(blocks::watch(start, sink.clone())),
            Some(_) | None => None,
        };
        let budget = held.insert(Budget::new(start, params)?);
        call_budgeted(text, params, record, (first, last), budget, trace, learner)
    })();
    if let Err(e) = &called {
        report_error(held.as_mut(), params, start, e);
    }
    called
}

// error-report.1 design -- a call that returns an error ends its report with an error line, which the sink keeps, not the report dropped with the error (not measured)
/// The call report's `error` line of `e`: the variant's name and its
/// message as one value (`field_value`). The budget's report writes it,
/// and where the error came before there was one, a report of its own on
/// the call's sink.
pub(crate) fn report_error(
    budget: Option<&mut Budget>,
    params: &Params,
    start: Instant,
    e: &Error,
) {
    let mut own = None;
    let report = match budget.and_then(Budget::report) {
        Some(report) => report,
        None if params.report => {
            own.insert(call_report::Report::new(start, params.report_sink.clone()))
        }
        None => return,
    };
    report.line(
        "error",
        format_args!(
            " error={name} message={message}",
            name = e.name(),
            message = field_value(&e.to_string())
        ),
    );
}

/// `call` once its budget stands.
fn call_budgeted<T: Trace, L: Learner>(
    text: &str,
    params: &Params,
    record: Option<&Record>,
    (first, last): (Width, Width),
    budget: &mut Budget,
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    let named = &params.order;
    let entry = if L::LISTENS {
        "learn"
    } else if record.is_some() {
        "replay"
    } else {
        "groebner"
    };
    report_call(budget, params, entry, T::LISTENS);
    // f4-gfp.30 design -- the run takes one thread, whatever threads allows, not several that share the reduction of a matrix's rows (not measured)
    let _: Threads = params.threads;

    let header = Header::read(text)?;
    let outcome = if header.rational() {
        // q-lift.30 design -- learn and replay refuse characteristic 0, and the lift's records stay inside the call, not learn over Q returning the record of its learn prime (not measured)
        if record.is_some() || L::LISTENS {
            return Err(Error::Characteristic {
                value: header.characteristic.to_string(),
            });
        }
        report_read(budget, &header, "q");
        let mut primes = Primes::new(params.seed, Vec::new());
        let mut report = Report::default();
        lift::lift(
            text,
            params,
            named,
            &header,
            budget,
            trace,
            &mut primes,
            lift::Reconstruct::Chain,
            lift::Replays::Pairs,
            // q-learn-time.10 design -- every learn of the lift starts as a pair learn, the first and each that learns afresh, in every order, with or without cofactors, not the pair learn in grevlex alone nor the first learn run alone a pair (not measured)
            lift::Learns::Pairs,
            &mut report,
        )?
    } else {
        let widths = (first, last);
        // gfp-large-primes.3 practice: monomorphization -- the entry point matches the field once and runs a copy of the call compiled for Gf31 or Gf64, not one field type that holds which of the two it is and matches at each operation
        match header.field()? {
            Prime::Gf31(field) => {
                report_read(budget, &header, "gf31");
                call_in(
                    field, text, params, record, widths, named, &header, budget, trace, learner,
                )?
            }
            Prime::Gf64(field) => {
                report_read(budget, &header, "gf64");
                call_in(
                    field, text, params, record, widths, named, &header, budget, trace, learner,
                )?
            }
        }
    };
    Ok(report_end(budget, outcome))
}

/// The call report's `call` line, where `budget` holds the report: the
/// parameters as the call reads them, the entry point `entry`, whether the
/// trace listens, and the processor's brand.
pub(crate) fn report_call(budget: &mut Budget, params: &Params, entry: &str, listens: bool) {
    let Some(report) = budget.report() else {
        return;
    };
    let threads = match params.threads {
        Threads::All => "all".to_string(),
        Threads::Count(k) => k.to_string(),
    };
    let (checker, points) = match params.checker {
        None => ("off", None),
        Some(CheckerMode::Exact) => ("exact", None),
        Some(CheckerMode::Cheap { points }) => ("cheap", Some(points)),
    };
    let deadline = params.deadline.map(|d| d.as_millis());
    report.line(
        "call",
        format_args!(
            " entry={entry} order={order} seed={seed} threads={threads} \
             signatures={signatures} echelon={echelon} cofactors={cofactors} \
             checker={checker}{points} deadline={deadline} memory={memory} cancel={cancel} \
             trace={trace} cpu={cpu}",
            order = params.order,
            seed = params.seed,
            signatures = on(params.signatures),
            echelon = on(params.echelon),
            cofactors = on(params.cofactors),
            points = Opt("points", points),
            deadline = Or(deadline),
            memory = Or(params.memory),
            cancel = on(params.cancel.is_some()),
            trace = if listens { "listens" } else { "off" },
            cpu = Or(cpu()),
        ),
    );
}

/// The call report's `read` line: the header as read, and the field it
/// decided, `gf31`, `gf64`, or `q`.
pub(crate) fn report_read(budget: &mut Budget, header: &Header, field: &str) {
    if let Some(report) = budget.report() {
        report.line(
            "read",
            format_args!(
                " variables={variables} characteristic={characteristic} field={field}",
                variables = header.names.len(),
                characteristic = header.characteristic
            ),
        );
    }
}

// call-report.10 design -- entry::call puts the report into the outcome, once, at the end, not each site that builds an outcome (not measured)
/// `outcome` with the call report where `budget` holds one: its `stop`
/// line where a bound stopped the call, then its `end` line, and the
/// report taken out of the budget into `Outcome::report`.
pub(crate) fn report_end(budget: &mut Budget, mut outcome: Outcome) -> Outcome {
    if let Some(report) = budget.report() {
        if let Some(stop) = outcome.stopped {
            let bound = stop.name();
            report.line("stop", format_args!(" bound={bound}"));
        }
        let result = if outcome.basis.is_some() {
            "basis"
        } else {
            "stopped"
        };
        let bytes = outcome.basis.as_ref().map(String::len);
        let time = Secs(report.since_start());
        let (start, peak) = report.rss();
        report.line(
            "end",
            format_args!(
                " result={result}{bytes} time={time} process-rss-start={start} \
                 process-rss-peak={peak} process-rss-grown={grown}",
                bytes = Opt("bytes", bytes),
                grown = peak.saturating_sub(start)
            ),
        );
    }
    outcome.report = budget.take_report().map(call_report::Report::finish);
    outcome
}

/// A value of the call report, or `none` where it is absent.
struct Or<T>(Option<T>);

impl<T: fmt::Display> fmt::Display for Or<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "none"),
        }
    }
}

/// The outcome of a call in the field `field`: what `call` did after it
/// read the header (gfp-large-primes.3).
#[allow(
    clippy::too_many_arguments,
    reason = "the field, the instance and what was read of it, the call's parameters, record, and order, and the budget, trace, and learner the run borrows, none of which belongs with another"
)]
fn call_in<F: Field, T: Trace, L: Learner>(
    field: F,
    text: &str,
    params: &Params,
    record: Option<&Record>,
    widths: (Width, Width),
    named: &MonomialOrder,
    header: &Header,
    budget: &mut Budget,
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    let made = run_in(
        field, text, params, record, widths, false, named, header, budget, trace, learner, None,
    )?;
    match made {
        Made::Outcome(outcome) => Ok(outcome),
        #[allow(
            clippy::unreachable,
            reason = "`finish` makes an image only where `image` holds, false here, and every call over GF(p) meets this match in the tests"
        )]
        Made::Image(_) | Made::Lanes(_) | Made::Pair(_) => {
            unreachable!("a run asked for no image made one")
        }
    }
}

/// A run's result for the lift: its image, or the outcome of a run a
/// bound stopped.
pub(crate) enum Imaged {
    Image(Image),
    /// A learn run over a field of more than one lane: an image per lane,
    /// in lane order (q-later-runs-time.1).
    Lanes(Vec<Image>),
    /// A replay over two primes at once: its basis at both, on one list of
    /// terms (q-lift-rest-time.1).
    Pair(PairImage),
    Stopped(Outcome),
}

/// A run in the field `field` as `call_in` runs it, the widths W8 to W32,
/// whose reduced basis, and cofactors where `params` asks for cofactors,
/// become an `Image`, or over a field of more than one lane, without
/// cofactors, an image per lane for a learn run and a `PairImage` for a
/// replay; `params.checker` is not read. A replay where there is
/// a record, else a run that tells `learner` what it records. A replay
/// that tracks no cofactors, where `kept` is Some, runs from the shape kept
/// there, or keeps the shape of what it builds where none is kept.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, the instance and its header, the call's parameters, record, and order, the budget, trace, and learner the run borrows, and the kept shape, none of which belongs with another"
)]
pub(crate) fn image_in<F: Field, T: Trace, L: Learner>(
    field: F,
    text: &str,
    params: &Params,
    record: Option<&Record>,
    named: &MonomialOrder,
    header: &Header,
    budget: &mut Budget,
    trace: &mut T,
    learner: &mut L,
    kept: Option<&mut Kept>,
) -> Result<Imaged, Error> {
    let widths = (Width::W8, Width::W32);
    let made = run_in(
        field, text, params, record, widths, true, named, header, budget, trace, learner, kept,
    )?;
    Ok(match made {
        Made::Image(image) => Imaged::Image(image),
        Made::Lanes(images) => Imaged::Lanes(images),
        Made::Pair(image) => Imaged::Pair(image),
        // A run that makes an image returns an outcome where a bound
        // stopped it alone.
        Made::Outcome(outcome) => Imaged::Stopped(outcome),
    })
}

/// What `call_in` and `image_in` run: the run in the field `field`, which
/// makes an image where `image` is true, and a replay's shape in `kept`.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, the instance and what was read of it, the call's parameters, record, and order, whether it makes an image, the budget, trace, and learner the run borrows, and the kept shape, none of which belongs with another"
)]
#[allow(
    clippy::expect_used,
    reason = "Ran::Done(None) comes only from a run whose basis FGLM changes, and run_in runs none"
)]
fn run_in<F: Field, T: Trace, L: Learner>(
    field: F,
    text: &str,
    params: &Params,
    record: Option<&Record>,
    (first, last): (Width, Width),
    image: bool,
    named: &MonomialOrder,
    header: &Header,
    budget: &mut Budget,
    trace: &mut T,
    learner: &mut L,
    kept: Option<&mut Kept>,
) -> Result<Made, Error> {
    let refused = || Error::Order {
        name: params.order.to_string(),
    };
    let run = Run {
        text,
        params,
        names: &header.names,
        field,
        record,
        named,
        change: false,
        image,
        first,
        last,
        order: (),
    };
    // order-family.4 practice: monomorphization -- the entry point matches the order once and runs a copy of the run compiled for Grevlex, Lex, or Block, not one order type that holds which of the three it is and matches at each comparison
    let made = match named {
        MonomialOrder::Grevlex => {
            run.with(Grevlex)
                .widening(header, trace, learner, budget, kept, Why::Start)?
        }
        MonomialOrder::Lex => return run.lex(header, trace, learner, budget, kept),
        // fglm-lex.19 design -- FGLM is wired for lex alone, and a block order keeps F4 in the block order, not FGLM for every order of the family but grevlex where the ideal is zero-dimensional (log 0011-2)
        MonomialOrder::Block(sizes) => {
            let block = Block::new(sizes, header.names.len()).ok_or_else(refused)?;
            run.with(block)
                .widening(header, trace, learner, budget, kept, Why::Start)?
        }
    };
    Ok(made.expect("a run whose basis FGLM does not change returns an outcome"))
}

/// What a call reads besides its monomorphized parts, and its field and
/// order.
struct Run<'a, F, O> {
    text: &'a str,
    params: &'a Params,
    names: &'a [&'a str],
    field: F,
    /// The record a replay replays, None for a run.
    record: Option<&'a Record>,
    /// The order as the call named it.
    named: &'a MonomialOrder,
    /// Whether the run is a lex call's grevlex run, whose basis FGLM
    /// changes to lex.
    change: bool,
    /// Whether the run makes an image for the lift in place of the texts.
    image: bool,
    /// The narrowest width the run may take, where it starts, save a replay
    /// from a kept shape, which starts in the shape's width (q-replay-time.10).
    first: Width,
    /// The widest width the run may take.
    last: Width,
    order: O,
}

impl<'a, F: Field> Run<'a, F, ()> {
    /// The run in the order `order`.
    fn with<O>(&self, order: O) -> Run<'a, F, O> {
        Run {
            text: self.text,
            params: self.params,
            names: self.names,
            field: self.field,
            record: self.record,
            named: self.named,
            change: self.change,
            image: self.image,
            first: self.first,
            last: self.last,
            order,
        }
    }

    // fglm-lex.1 cited: eder-2013a §5 -- in such cases it is more efficient to compute a Gröbner basis w.r.t. the graded reverse-lexicographical ordering and then to use a Gröbner conversion via FGLM; faugere-1993 §1 and abstract -- for the lexicographical basis it is usually much faster to compute the basis for the degree ordering and to use the algorithm than a direct computation by Buchberger's algorithm
    // fglm-lex.15 cited: berthomieu-2021 §4.5 -- (i) run the F4 tracer modulo p₀, (ii) run fglm on the computed Gröbner basis, repeated for several primes applying the tracer learnt from p₀; berthomieu-2022 §6.3 -- F4 with a tracer for the computation of G_drl modulo each prime, then the change of order step
    /// A lex call: the grevlex run, or the replay of a changed record's
    /// grevlex steps, and its basis changed to lex by FGLM where the ideal is
    /// zero-dimensional and its staircase holds at most
    /// `fglm::MAX_STAIRCASE` monomials.
    /// Otherwise a run starts over in lex, a changed record does not fit
    /// (fglm-lex.20), and a record not changed replays in lex.
    #[allow(
        clippy::expect_used,
        reason = "Ran::Done(None) comes only from a run whose basis FGLM changes, and the widening here runs none"
    )]
    fn lex<T: Trace, L: Learner>(
        &self,
        header: &Header,
        trace: &mut T,
        learner: &mut L,
        budget: &mut Budget,
        mut kept: Option<&mut Kept>,
    ) -> Result<Made, Error> {
        let mut why = Why::Start;
        if self.record.is_none_or(Record::changed) {
            let grevlex = Run {
                change: true,
                ..self.with(Grevlex)
            };
            let made = grevlex.widening(
                header,
                trace,
                learner,
                budget,
                kept.as_deref_mut(),
                Why::Start,
            )?;
            if let Some(made) = made {
                return Ok(made);
            }
            // fglm-lex.20 design -- a changed record whose replayed grevlex basis is not zero-dimensional, or whose staircase passes the bound, does not fit at the step one past the record's last, not a panic (not measured)
            if let Some(record) = self.record {
                return Err(Error::Misfit {
                    step: narrow::id(record.steps.len()).map_err(|_| Error::Size)? + 1,
                });
            }
            why = Why::Lex;
        }
        // fglm-lex.3 design -- a lex call on an ideal that is not zero-dimensional returns its reduced lex basis, not an error (not measured)
        // fglm-lex.4 design -- that basis comes from F4 in lex, not from the Gröbner walk from the grevlex basis (not measured)
        // fglm-lex.5 design -- the lex run starts from the instance's generators, the call started over, not F4 in lex from the grevlex basis just computed (not measured)
        // fglm-lex.27 design -- the lex run starts once the grevlex run has returned and given back its table, basis, and cofactors, not from inside the grevlex run's solve with them still held (not measured)
        // monomial-formats.13 design -- the lex run picks its format from the input afresh and starts at the narrowest width, not in the width the grevlex run ended in (not measured)
        let made = self
            .with(Lex)
            .widening(header, trace, learner, budget, kept, why)?;
        Ok(made.expect("a run whose basis FGLM does not change returns an outcome"))
    }
}

impl<F: Field, O: Order<Dense8> + Order<Dense16> + Order<Dense32>> Run<'_, F, O> {
    // monomial-formats.5 cited: monagan-2007 §2.1 -- it is especially easy to determine an optimal packing for these orders using bounds on the total degree; demin-2024 §4.1 -- the monomial representation is selected at the beginning of the computation
    // monomial-formats.6 design -- the run finds its format by reading the input in each format in turn from the narrowest, the reader's overflow naming a term past it, not a pass over the text for its largest term degree before the read (not measured)
    // monomial-formats.7 cited: bachmann-1998 §3 -- ideally switch dynamically from one exponent size to the next larger one whenever it becomes necessary; with Singular this cannot be done yet
    // monomial-formats.8 design -- the run starts over from its input in the wider format, under the same seed and the budget it has left, not its state converted to the wider format and continued from where it overflowed (log 0013-1)
    // monomial-formats.12 cited: demin-2024 §4.1 -- the monomial representation is selected at the beginning of the computation, then the specialized functions that implement monomial arithmetic are just-in-time compiled and inlined, which relocates the work from running time to compile time; bachmann-1998 §3 -- the versions differ only in comparisons realized by function pointers, each a function call, instead of one inlined routine
    /// The run in the widths from `first` to `last`, each width a copy of
    /// the run compiled for its format: it starts in `first`, and where it
    /// returns `Error::Overflow`, from its input or from the run, over in
    /// the next wider width. In `last`, its `Error::Overflow` is returned.
    /// Where the signature run hands over, the call starts over in the
    /// plain run in the same width, and every later start is the plain
    /// run's. A replay that tracks no cofactors, where `kept` is Some,
    /// starts in the width of the shape kept there where that is wider, and
    /// where its terms differ from the shape's, over in `first` without the
    /// shape. None as `Ran::Done` holds it. The first start is for `why`,
    /// and each later one for what ended the start before it.
    fn widening<T: Trace, L: Learner>(
        &self,
        header: &Header,
        trace: &mut T,
        learner: &mut L,
        budget: &mut Budget,
        kept: Option<&mut Kept>,
        mut why: Why,
    ) -> Result<Option<Made>, Error> {
        // q-replay-time.7 design -- only a replay without cofactors keeps a shape or runs from one, and a replay that tracks cofactors builds its matrices from the record, not the cofactor tracking's matrices and table kept as well (not measured)
        let mut kept = kept.filter(|_| self.record.is_some() && !self.tracks());
        // q-replay-time.10 design -- a replay where a shape is kept starts in the shape's width, not the widening from the narrowest width that every run starts (monomial-formats.6) (not measured)
        let mut width = kept
            .as_ref()
            .and_then(|k| k.width())
            .filter(|&w| self.first < w && w <= self.last)
            .unwrap_or(self.first);
        let mut plain = false;
        loop {
            let shape = kept.as_deref_mut();
            let got = match width {
                Width::W8 => self.start::<Dense8, T, L>(
                    header, plain, trace, learner, budget, shape, width, why,
                ),
                Width::W16 => self.start::<Dense16, T, L>(
                    header, plain, trace, learner, budget, shape, width, why,
                ),
                Width::W32 => self.start::<Dense32, T, L>(
                    header, plain, trace, learner, budget, shape, width, why,
                ),
            };
            match got {
                // gfp-unsolved.16 design -- the handover starts the call over from its input in the plain run, a new table and cofactor tracking, the same seed, and the budget left, not the plain run from the signature run's basis (log 0014-1)
                // q-learn-time.13 design -- a pair learn that hands over runs its plain run over both lanes, each row on its own or ruled zero by a test (learn-zero-time.1), not the pair learn halting at the handover with the plain run at p alone and the verification's run at a fresh prime (log 0052-1)
                Ok(Ran::Handover) => (plain, why) = (true, Why::Handover),
                // q-replay-time.5 design -- a replay whose terms differ from the shape starts over at its prime as a full replay, from its input, in a new table, not the full replay continued from the step where the terms differed, nor the prime dropped as a misfit (not measured)
                // q-replay-time.6 design -- the full replay that a difference started keeps what it builds, and its shape replaces the kept one, not the kept shape kept and the full replay keeping nothing (not measured)
                Ok(Ran::Refit) => {
                    if let Some(kept) = kept.as_deref_mut() {
                        kept.clear();
                    }
                    (width, why) = (self.first, Why::Refit);
                }
                Ok(Ran::Done(made)) => return Ok(made),
                Err(e @ Error::Overflow { .. }) => {
                    match width.wider().filter(|&w| w <= self.last) {
                        Some(wider) => (width, why) = (wider, Why::Wider),
                        None => return Err(e),
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Whether the run tracks cofactors: where the call asks for cofactors,
    /// or for the checker in a run that makes no image.
    fn tracks(&self) -> bool {
        self.params.cofactors || (!self.image && self.params.checker.is_some())
    }

    // input-redundancy-time.8 design -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
    /// Whether the run takes its generators in reduced row echelon form
    /// (`echelon`): where it tracks no cofactors, and its record, or where
    /// it has none `Params::echelon`, asks for it.
    fn echelons(&self) -> bool {
        !self.tracks() && self.record.map_or(self.params.echelon, |r| r.echelon)
    }

    /// Reads the generators into a table of the format `M`, of the width
    /// `width`, in the run's order, takes them in reduced row echelon form
    /// where the run does (`echelons`), and solves, with the cofactor tracking
    /// where the run tracks cofactors (`tracks`), in the plain run where
    /// `plain` is true. Where `kept` is Some, the run is a replay: where a
    /// shape is kept there in `M`, the generators are read into its table
    /// and the replay runs from it. Otherwise a replay that tracks no
    /// cofactors keeps what it builds, and its shape is put there where it
    /// returns an image; one that tracks cofactors keeps nothing, so no
    /// shape is kept for it to run from. `Error::Overflow` where a degree passes
    /// `M`. The run starts for `why`, which its line in the call report
    /// names.
    #[allow(
        clippy::too_many_arguments,
        reason = "the header, the run's kind, width, and cause, the trace, learner, and budget it borrows, and the kept shape, none of which belongs with another"
    )]
    fn start<M: Format, T: Trace, L: Learner>(
        &self,
        header: &Header,
        plain: bool,
        trace: &mut T,
        learner: &mut L,
        budget: &mut Budget,
        mut kept: Option<&mut Kept>,
        width: Width,
        why: Why,
    ) -> Result<Ran, Error>
    where
        O: Order<M>,
    {
        let params = self.params;
        if let Some(kept) = kept.as_deref_mut()
            && let Some(Shape { table, built }) = kept.get::<M>()
        {
            self.report_run::<M, NoCofactors, T, L>(budget, plain, width, why, Some("use"));
            // q-replay-time.3 cited: parisse-2013 §4, steps 5 and 6 -- the lists of all possible monomials are recorded for further primes during the first prime's computation, and a row holds monomial indices relative to that ordered list
            let generators = self.read(header, table, budget)?;
            let generators = match self.echelon(table, generators, learner, budget) {
                Ok(generators) => generators,
                Err(halt) => return halted(halt, Stats::default()),
            };
            // A replay draws no combination.
            let ran = self.solve(
                table,
                generators,
                plain,
                &mut NoCofactors,
                trace,
                learner,
                budget,
                0,
                0,
                Shaping::Use(built),
            );
            match ran {
                Ok(Ran::Refit) => {}
                Ok(Ran::Done(_) | Ran::Handover) | Err(_) => kept.used += 1,
            }
            return ran;
        }
        let format = M::new(self.names.len());
        // f4-gfp.38 design -- the hash multipliers come from rand_xoshiro's Xoshiro256PlusPlus seeded by seed_from_u64, not from a generator written for the engine (not measured)
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(params.seed);
        let mut table = Table::new(format, &mut rng);
        // A replay that tracks no cofactors where a shape may be kept keeps
        // what it builds.
        let shape = self.record.map(|_| {
            if kept.is_some() && !self.tracks() {
                "take"
            } else {
                "off"
            }
        });
        if self.tracks() {
            self.report_run::<M, Tracked<F, M>, T, L>(budget, plain, width, why, shape);
        } else {
            self.report_run::<M, NoCofactors, T, L>(budget, plain, width, why, shape);
        }
        // order-family.13 cited: faugere-1999 §2.5 -- to homogenize the list of polynomials is not efficient for big systems since parasite solutions are also computed; giovini-1991 §3 -- the strategy of homogenizing the basis is never very good
        // The run computes on the generators, not homogenized, in the call's order.
        let generators = self.read(header, &mut table, budget)?;
        let generators = match self.echelon(&mut table, generators, learner, budget) {
            Ok(generators) => generators,
            Err(halt) => return halted(halt, Stats::default()),
        };
        let field = self.field;
        // gfp-unsolved.7 design -- a combination's coefficients are uniform in [1, p), from a Xoshiro256PlusPlus seeded by the call's generator after its other draws, not a generator of its own seeded by a constant or coefficients in [0, p) (not measured)
        if self.tracks() {
            // monomial-formats.9 design -- the cofactor tracking's table takes the run's format, and an overflow in either table starts the call over, not a format of its own for the cofactor table widened apart from the run's (log 0013-1)
            let mut cof = Tracked::new(field, format, generators.len(), &mut rng);
            // f4-gfp.51 design -- the checker's seed is drawn from the call's generator after the cofactor table's multipliers, not the call's seed passed as it is (not measured)
            let seed = rng.next_u64();
            let combine = rng.next_u64();
            self.solve(
                &mut table,
                generators,
                plain,
                &mut cof,
                trace,
                learner,
                budget,
                seed,
                combine,
                Shaping::Off,
            )
        } else if let Some(kept) = kept {
            let combine = rng.next_u64();
            let mut built = Built::default();
            // q-replay-time.2 cited: berthomieu-2022a §3.4, Tracer over Q -- the second modular computation learns, with exact linear algebra, all polynomial data the F4 matrices need to give the nonzero information
            let ran = self.solve(
                &mut table,
                generators,
                plain,
                &mut NoCofactors,
                trace,
                learner,
                budget,
                0,
                combine,
                Shaping::Take(&mut built),
            )?;
            match &ran {
                Ran::Done(Some(Made::Image(_) | Made::Lanes(_) | Made::Pair(_))) => {
                    kept.put(width, Shape { table, built });
                }
                Ran::Done(None | Some(Made::Outcome(_))) | Ran::Handover | Ran::Refit => {}
            }
            Ok(ran)
        } else {
            let combine = rng.next_u64();
            self.solve(
                &mut table,
                generators,
                plain,
                &mut NoCofactors,
                trace,
                learner,
                budget,
                0,
                combine,
                Shaping::Off,
            )
        }
    }

    // input-redundancy-time.9 design -- the echelon has a line of the call report and a stage of its own, not its figures on the inputs line and its time in the read's (not measured)
    /// The run's generators in reduced row echelon form where `echelons`
    /// holds, else as read; `learner` hears which. A replay follows its
    /// record's leading monomials (`echelon::replay`), and any other run
    /// chooses its pivots (`echelon::echelon`). The call report gets the
    /// `echelon` line, or where the echelon halts the run's `ran` line.
    fn echelon<M: Format, L: Learner>(
        &self,
        table: &mut Table<M>,
        generators: Vec<Poly<F, Rank1>>,
        learner: &mut L,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, Rank1>>, Halt>
    where
        O: Order<M>,
    {
        let echelons = self.echelons();
        learner.echelon(echelons);
        if !echelons {
            return Ok(generators);
        }
        budget.open();
        let before = budget.counts().0;
        let count = generators.len();
        let (field, order) = (&self.field, &self.order);
        let echelon = match self.record {
            // run-generators-time.5 cited: berthomieu-2021 §4.3 -- the following calls of F4 apply the trace: generate the matrix with the already computed polynomials using the information from the trace
            Some(record) => {
                crate::echelon::replay(field, table, order, generators, &record.inputs, budget)
            }
            None => crate::echelon::echelon(field, table, order, generators, budget),
        };
        budget.mark(Stage::Echelon);
        let subtracted = budget.counts().0 - before;
        match echelon {
            Ok(e) => {
                if let Some(report) = budget.report() {
                    let rows = e.rows.iter().filter(|g| !g.is_zero()).count();
                    let time = Secs(report.time(Stage::Echelon));
                    report.line(
                        "echelon",
                        format_args!(
                            " generators={count} rows={rows} columns={columns} \
                             subtracted={subtracted} time={time}",
                            columns = e.columns
                        ),
                    );
                }
                Ok(e.rows)
            }
            Err(halt) => {
                let step = match halt {
                    Halt::Misfit(step) => Some(step),
                    Halt::Overflow | Halt::Size | Halt::Stop(_) | Halt::Handover => None,
                };
                let at = Some("echelon");
                report_ran(
                    budget,
                    halt_name(halt),
                    at,
                    step,
                    &Stats::default(),
                    None,
                    None,
                );
                Err(halt)
            }
        }
    }

    /// The generators read into `table` in the run's order, the read timed
    /// for the call report: its `inputs` line, or where the read overflows
    /// the format or meets a denominator that is 0 mod p, the run's `ran`
    /// line.
    fn read<M: Format>(
        &self,
        header: &Header,
        table: &mut Table<M>,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, Rank1>>, Error>
    where
        O: Order<M>,
    {
        budget.open();
        let read = header.generators(&self.field, table, &self.order);
        budget.mark(Stage::Read);
        let at = Some("read");
        match &read {
            Ok(generators) => {
                if let Some(report) = budget.report() {
                    let nonzero = generators.iter().filter(|g| !g.is_zero()).count();
                    let terms: usize = generators.iter().map(Poly::len).sum();
                    let time = Secs(report.time(Stage::Read));
                    report.line(
                        "inputs",
                        format_args!(
                            " generators={generators} nonzero={nonzero} terms={terms} read={time}",
                            generators = generators.len()
                        ),
                    );
                }
            }
            Err(Error::Overflow { .. }) => {
                report_ran(budget, "overflow", at, None, &Stats::default(), None, None);
            }
            Err(Error::NoInverse { .. }) => {
                report_ran(
                    budget,
                    "denominator",
                    at,
                    None,
                    &Stats::default(),
                    None,
                    None,
                );
            }
            Err(_) => {}
        }
        read
    }

    // order-family.14 cited: eder-2013a §4 -- ordering critical pairs by sugar degree, signature-based computations slow down by a large factor; §5 -- signature-based computations w.r.t. not degree compatible orderings are rather inefficient in sorting critical pairs, and grevlex followed by FGLM or a Gröbner walk is more efficient
    // fglm-lex.2 cited: eder-2013a §5 -- signature-based computations w.r.t. not degree compatible orderings are rather inefficient in sorting critical pairs; compute w.r.t. grevlex and then convert via FGLM
    /// The engine of a run in the format `M`, after a handover where
    /// `plain` is true: the replay where there is a record, else the
    /// signature run where the call asks for it, the order is graded, and
    /// no handover was, else the plain run. A lex call's grevlex run is a
    /// run in grevlex, the signature run where the call asks for it.
    fn engine<M: Format>(&self, plain: bool) -> Engine<'_>
    where
        O: Order<M>,
    {
        if let Some(record) = self.record {
            Engine::Replay(record)
        } else if self.params.signatures && <O as Order<M>>::GRADED && !plain {
            Engine::Signature
        } else {
            Engine::Plain
        }
    }

    // call-report.11 design -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
    /// The call report's `run` line of a run in the format `M`, of the
    /// width `width`, with the cofactor tracking `C`, after a handover where
    /// `plain` is true, started for `why`; `shape` is a replay's, `use`,
    /// `take`, or `off`. The run's role, its engine and the reduction's mode
    /// each with what decided it, by the functions the engine decides by
    /// (`engine`, `f4::combines`, `f4::plain_combines`, `f4::learn_tests`,
    /// `matrix::tests_rows`, `matrix::zero_combinations`).
    fn report_run<M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>(
        &self,
        budget: &mut Budget,
        plain: bool,
        width: Width,
        why: Why,
        shape: Option<&str>,
    ) where
        O: Order<M>,
    {
        let Some(report) = budget.report() else {
            return;
        };
        let role = if self.record.is_some() {
            "replay"
        } else if L::LISTENS {
            "learn"
        } else if self.image {
            "verify"
        } else {
            "call"
        };
        let engine = self.engine::<M>(plain);
        // call-report.21 design -- where signatures are off and the order is not graded, the run line names signatures-off, the condition `Run::engine` tests first, not order (not measured)
        let (name, by) = match engine {
            Engine::Replay(_) => ("replay", "record"),
            Engine::Signature => ("signature", "signatures"),
            Engine::Plain if plain => ("plain", "handover"),
            Engine::Plain if !self.params.signatures => ("plain", "signatures-off"),
            Engine::Plain => ("plain", "order"),
        };
        let field = &self.field;
        let p = field.characteristic();
        // call-report.20 design -- a run line reads each lane's prime from the residue of −1 there, p − 1, not a method added to the field contract for the report (not measured)
        let minus_one = field.neg(field.coef(1));
        let primes: Vec<String> = (0..F::LANES)
            .map(|k| (field.lane(minus_one, k) + 1).to_string())
            .collect();
        // `Gf31` and `Gf31Pair` take the primes below 2^31, the others those
        // above (`text::Prime`, `lift::Primes`).
        let small = p < 1 << 31;
        let kind = match (F::LANES, small) {
            (1, true) => "gf31",
            (1, false) => "gf64",
            (_, true) => "gf31pair",
            (_, false) => "gf64pair",
        };
        let order = match self.named {
            _ if self.change => "grevlex",
            MonomialOrder::Grevlex => "grevlex",
            MonomialOrder::Lex => "lex",
            MonomialOrder::Block(_) => "block",
        };
        let combines = crate::f4::combines::<F, M, C, T, L>();
        let tests = crate::f4::learn_tests::<F, M, C, T, L>();
        let (reduce, reduce_by) = match engine {
            Engine::Replay(_) => ("rows", Some("record")),
            Engine::Signature if combines && matrix::tests_rows::<F>(p) => ("tests", None),
            Engine::Plain if crate::f4::plain_combines::<F, M, C, T, L>(p) => {
                ("combinations", None)
            }
            // q-zero-work-time.5 design -- the run line names a learn's plain run that takes tests `reduce=tests`, from the function `f4::run` decides by, not `reduce=rows reduce-by=learner` as for a learn's signature run (not measured)
            Engine::Plain if tests && matrix::zero_combinations(p).is_some() => ("tests", None),
            Engine::Signature | Engine::Plain => {
                let lanes = engine.signature() && F::LANES > 1;
                // q-zero-work-time.6 design -- the learner is a reason for rows in a learn's signature run alone (learn-zero-time.3), and a learn's plain run at p = 2 or 3 names the prime, not the learner as the first reason in every learn run (not measured)
                let learner = L::LISTENS && engine.signature();
                let by = [
                    (T::LISTENS, "trace"),
                    (learner, "learner"),
                    (C::TRACKS, "cofactors"),
                    (lanes, "lanes"),
                ]
                .into_iter()
                .find_map(|(holds, by)| holds.then_some(by));
                ("rows", Some(by.unwrap_or("prime")))
            }
        };
        // gfp-unsolved.17 design -- a call that tracks cofactors does not hand over: the signature run computes it to the end, not the handover with cofactors (log 0014-1)
        let handover = engine.signature().then_some(on(!C::TRACKS));
        report.run(format_args!(
            " role={role} engine={name} engine-by={by} field={kind} primes={primes} width={width} \
             order={order} cofactors={cofactors} reduce={reduce}{reduce_by}{handover}{shape} \
             why={why}",
            primes = primes.join(","),
            width = width.bits(),
            cofactors = on(C::TRACKS),
            reduce_by = Opt("reduce-by", reduce_by),
            handover = Opt("handover", handover),
            shape = Opt("shape", shape),
            why = why.name(),
        ));
    }

    /// Runs F4, the plain run where `plain` is true, or replays the record
    /// where there is one, under `shaping`, changes the basis to lex where
    /// the run is a lex call's grevlex run, and finishes the outcome; `seed`
    /// draws the checker's points, and `combine` the combinations of the
    /// plain run and of the signature run. `Ran::Refit` where a replay from
    /// a shape meets an element whose terms differ from the shape's. The
    /// call report gets the run's `ran` line, and a lex call's grevlex run
    /// its `fglm` line.
    #[allow(
        clippy::too_many_arguments,
        reason = "the table, generators, and cofactors of the run, its kind, the trace, learner, and budget it borrows, its seeds, and its shaping, none of which belongs with another"
    )]
    fn solve<M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>(
        &self,
        table: &mut Table<M>,
        generators: Vec<Poly<F, Rank1>>,
        plain: bool,
        cof: &mut C,
        trace: &mut T,
        learner: &mut L,
        budget: &mut Budget,
        seed: u64,
        combine: u64,
        shaping: Shaping<'_>,
    ) -> Result<Ran, Error>
    where
        O: Order<M>,
    {
        let mut stats = Stats::default();
        let field = &self.field;
        let order = &self.order;
        let engine = self.engine::<M>(plain);
        // The signature run's S-pairs.
        let mut fates = Pairs::new(true);
        let run = match engine {
            Engine::Replay(record) => {
                // order-family.15 design -- a record holds the order it was learned in, and a replay under another order does not fit, not a replay that runs in the record's order whatever the call names (not measured)
                // input-redundancy-time.8 design -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
                if record.order == *self.named && !(record.echelon && C::TRACKS) {
                    let replayed = crate::f4::replay(
                        field, table, order, generators, record, cof, trace, budget, &mut stats,
                        shaping,
                    );
                    if let Some(run) = replayed.transpose() {
                        run
                    } else {
                        report_ran(budget, "refit", None, None, &stats, None, None);
                        return Ok(Ran::Refit);
                    }
                } else {
                    Err(Halt::Misfit(0))
                }
            }
            Engine::Signature => {
                // gfp-unsolved.17 design -- a call that tracks cofactors does not hand over: the signature run computes it to the end, not the handover with cofactors (log 0014-1)
                let handover = (!C::TRACKS).then_some(HANDOVER);
                crate::f4::run_sig(
                    field, table, order, generators, cof, trace, budget, &mut stats, learner,
                    handover, &mut fates, combine,
                )
            }
            Engine::Plain => crate::f4::run(
                field, table, order, generators, cof, trace, budget, &mut stats, learner, combine,
            ),
        };
        if budget.report().is_some() {
            let (outcome, at, step, elements) = match &run {
                Ok(elements) => ("basis", None, None, Some(elements.len())),
                Err(Halt::Handover) => ("handover", None, None, None),
                Err(Halt::Overflow) => ("overflow", Some("run"), None, None),
                Err(Halt::Size) => ("size", None, None, None),
                Err(Halt::Stop(_)) => ("stop", None, None, None),
                Err(Halt::Misfit(step)) => ("misfit", None, Some(*step), None),
            };
            let fates = engine.signature().then_some(&fates);
            report_ran(budget, outcome, at, step, &stats, elements, fates);
        }
        let elements = match run {
            Ok(elements) => elements,
            Err(halt) => return halted(halt, stats),
        };
        if !self.change {
            let made = self.finish(table, elements, order, cof, stats, budget, seed);
            return Ok(Ran::Done(Some(made)));
        }
        let leads: Vec<MonoId> = elements.iter().map(|(g, _)| g.lead()).collect();
        budget.open();
        let found = fglm::staircase(table, order, &leads, budget);
        budget.mark(Stage::Find);
        let staircase = match found {
            Ok(Some(staircase)) => staircase,
            Ok(None) => {
                if budget.report().is_some() {
                    let verdict = if fglm::zero_dimensional(table, &leads) {
                        "past-bound"
                    } else {
                        "positive-dimensional"
                    };
                    report_fglm(budget, verdict, None);
                }
                return Ok(Ran::Done(None));
            }
            Err(halt) => {
                report_fglm(budget, halt_name(halt), None);
                return halted(halt, stats);
            }
        };
        let changed = fglm::change(field, table, order, &Lex, &staircase, elements, cof, budget);
        budget.mark(Stage::Change);
        let verdict = changed
            .as_ref()
            .map_or_else(|&h| halt_name(h), |_| "change");
        report_fglm(budget, verdict, Some(staircase.len()));
        let elements = match changed {
            Ok(elements) => elements,
            Err(halt) => return halted(halt, stats),
        };
        learner.changed();
        // fglm-lex.18 design -- a lex call's stats are those of the run that made its basis, the grevlex run's where FGLM changed it, not the sum of both runs where the call started over (not measured)
        let made = self.finish(table, elements, &Lex, cof, stats, budget, seed);
        Ok(Ran::Done(Some(made)))
    }

    /// The outcome of the reduced basis `elements`: the basis text and the
    /// cofactor text, each written in `texts`, the call's order, and the
    /// checker's verdict where the call asks for it; `seed` draws the
    /// checker's points. Where the run makes an image, the image of
    /// `elements` in `texts` in place of the outcome, or over a field of
    /// more than one lane, without cofactors, an image per lane for a learn
    /// run and a pair image for a replay (`pair_image`). The texts
    /// spend units of `budget` as they are written (`write_basis`,
    /// `Cofactors::write`), and the image's cofactors as they are sorted
    /// (`Cofactors::image`); where the budget's check stops the call, the
    /// outcome is the stop, with no text and no image. The call report
    /// gets the image's line, or a line per text and the checker's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the table, elements, and cofactors of the run, the order of its texts, its counts, the budget, and the seed, none of which belongs with another"
    )]
    fn finish<M: Format, P: Order<M>, C: Cofactors<F, M>>(
        &self,
        table: &Table<M>,
        elements: Elements<F, Rank1, C::Element>,
        texts: &P,
        cof: &C,
        stats: Stats,
        budget: &mut Budget,
        seed: u64,
    ) -> Made {
        let (polys, rows): (Vec<_>, Vec<_>) = elements.into_iter().unzip();
        budget.open();
        if self.image && F::LANES > 1 {
            // A run of several lanes is a replay or a learn run, neither of
            // which tracks cofactors (q-later-runs-time.6, q-lift.12).
            debug_assert!(!C::TRACKS, "no cofactors over several lanes");
            let leads = leads(table, &polys);
            if self.record.is_some() {
                debug_assert_eq!(F::LANES, 2, "a pair replay runs over two primes");
                let image = pair_image(table, &self.field, &polys, leads, stats);
                budget.mark(Stage::Image);
                report_image(budget, polys.len());
                return Made::Pair(image);
            }
            let mut lanes: Vec<Vec<Sparse>> = vec![Vec::with_capacity(polys.len()); F::LANES];
            for g in &polys {
                for (k, sparse) in sparse_lanes(table, &self.field, g).into_iter().enumerate() {
                    lanes[k].push(sparse);
                }
            }
            let images = lanes
                .into_iter()
                .map(|basis| Image {
                    leads: leads.clone(),
                    basis,
                    cofactors: Vec::new(),
                    stats,
                })
                .collect();
            budget.mark(Stage::Image);
            report_image(budget, polys.len());
            return Made::Lanes(images);
        }
        if self.image {
            let basis = polys
                .iter()
                .map(|g| sparse(table, &self.field, g))
                .collect();
            // cofactor-stop-latency.1 design -- every stage that works on cofactors spends units of work on the call's budget as it works, the absorb's merge and append over Q among them, and a stage stops where the budget's check stops the call, not the ticks per row and generator and per part, and nothing for the image and the merge (log 0058-1)
            let cofactors = match cof.image(texts, &rows, budget) {
                Ok(cofactors) => cofactors,
                Err(why) => return Made::Outcome(stopped(why, stats)),
            };
            budget.mark(Stage::Image);
            report_image(budget, polys.len());
            return Made::Image(Image {
                leads: leads(table, &polys),
                basis,
                cofactors,
                stats,
            });
        }
        // text-stop-latency.1 design -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
        let mut basis = String::new();
        let written = write_basis(
            &mut basis,
            self.names,
            table,
            &self.field,
            polys.iter(),
            budget,
        );
        if let Err(why) = written {
            return Made::Outcome(stopped(why, stats));
        }
        budget.mark(Stage::Text);
        report_text(budget, "basis", basis.len(), Stage::Text);
        let params = self.params;
        let mut cofactors = None;
        if self.tracks() {
            budget.open();
            let mut text = String::new();
            if let Err(why) = cof.write(&mut text, self.names, texts, &rows, budget) {
                return Made::Outcome(stopped(why, stats));
            }
            budget.mark(Stage::Text);
            report_text(budget, "cofactors", text.len(), Stage::Text);
            cofactors = Some(text);
        }
        // text-stop-latency.11 design -- no full check of the texts' own follows the last text, not a full check after the texts (not measured)
        // The checker, where the call asks for it, comes after the texts
        // and ticks the budget as it polls: a stop there drops them.
        let (mut certified, mut flaw) = (None, None);
        if let Some(mode) = params.checker {
            #[allow(
                clippy::expect_used,
                reason = "past the image's return, a call with a checker tracks cofactors"
            )]
            let matrix = cofactors.as_deref().expect("the checker's cofactors");
            let n = self.names.len();
            budget.open();
            let (verdict, why) =
                match certify(self.text, &basis, matrix, self.named, n, mode, seed, budget) {
                    Ok(checked) => checked,
                    Err(why) => return Made::Outcome(stopped(why, stats)),
                };
            (certified, flaw) = (Some(verdict), why);
            budget.mark(Stage::Check);
            report_check(budget, mode, verdict);
        }
        Made::Outcome(Outcome {
            basis: Some(basis),
            stopped: None,
            cofactors: if params.cofactors { cofactors } else { None },
            certified,
            flaw,
            stats,
            report: None,
        })
    }
}

// result-time.7 design -- the image carries each element's leading monomial apart from its other terms, and the lift takes the other terms as they are, not the whole element, whose exponents and coefficients the lift moves down by the leading term (log 0033-1, 0042-1)
/// The leading monomials of the reduced basis `polys`, n exponents each.
fn leads<F: Field, M: Format>(table: &Table<M>, polys: &[Poly<F, Rank1>]) -> Vec<u32> {
    let mut leads = Vec::with_capacity(polys.len() * table.format().variables());
    for g in polys {
        // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
        table.exponents(g.monomials[0], &mut leads);
    }
    leads
}

// result-time.8 design -- the image hands over exponents, unpacked from the run's words, not the run's monomial ids with the lift reading exponents only where an image's terms differ from a part's (log 0042-1)
/// The monic polynomial `poly` of the run's table `table` as the lift reads
/// it: its terms but the leading one (result-time.7).
fn sparse<F: Field, M: Format>(table: &Table<M>, field: &F, poly: &Poly<F, Rank1>) -> Sparse {
    debug_assert_eq!(
        field.value(poly.coefs[0]),
        1,
        "an element of the image is monic"
    );
    let n = table.format().variables();
    // result-time.10 design -- the exponents go into a vector zeroed when it is allocated, which `Table::unpack` overwrites, not a vector of unset length written through its spare capacity (log 0042-1)
    let mut exponents = vec![0; (poly.len() - 1) * n];
    for (&m, e) in poly.monomials[1..]
        .iter()
        .zip(exponents.chunks_exact_mut(n))
    {
        // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
        table.unpack(m, e);
    }
    let coefs = poly.coefs[1..].iter().map(|&c| field.value(c)).collect();
    Sparse { exponents, coefs }
}

// q-lift-rest-time.1 cited: dekleine-2001 §2 -- a polynomial mapped to Z_p1 × … × Z_pb[x] is stored as its monomials, each with its coefficients at the b primes; §1 -- the polynomial operations are done simultaneously on b primes to amortize the monomial cost
/// The reduced basis `polys` of a pair replay, each element monic at both
/// lanes of `field`, with its leading monomials `leads`, as the lift reads
/// it: per element its terms but the leading one, each term's exponents
/// read from the table and written once, with its residue at the first
/// lane, 0 among them, and at the second beside.
fn pair_image<F: Field, M: Format>(
    table: &Table<M>,
    field: &F,
    polys: &[Poly<F, Rank1>],
    leads: Vec<u32>,
    stats: Stats,
) -> PairImage {
    debug_assert_eq!(F::LANES, 2, "a pair image holds two lanes");
    let n = table.format().variables();
    let mut first = Vec::with_capacity(polys.len());
    let mut second = Vec::with_capacity(polys.len());
    for poly in polys {
        debug_assert!(
            (0..F::LANES).all(|k| field.lane(poly.coefs[0], k) == 1),
            "an element of the image is monic at each lane"
        );
        let terms = poly.len() - 1;
        let mut exponents = vec![0; terms * n];
        let mut coefs = Vec::with_capacity(terms);
        let mut at_second = Vec::with_capacity(terms);
        let monomials = poly.monomials[1..].iter().zip(&poly.coefs[1..]);
        for ((&m, &c), e) in monomials.zip(exponents.chunks_exact_mut(n)) {
            // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
            table.unpack(m, e);
            coefs.push(field.lane(c, 0));
            at_second.push(field.lane(c, 1));
        }
        first.push(Sparse { exponents, coefs });
        second.push(at_second);
    }
    PairImage {
        leads,
        first,
        second,
        stats,
    }
}

// q-later-runs-time.11 design -- a pair's images read each term's exponents from the table once for every lane, and each leaves out the terms whose residue at its prime is 0, not each lane's image read from the table on its own (log 0039-1)
/// The polynomial `poly`, monic at each lane of `field`, as the lift reads
/// it at each lane: per lane its terms but the leading one whose residue
/// there is not 0, each term's exponents read from the table once for
/// every lane. The learn at two primes makes its images so.
fn sparse_lanes<F: Field, M: Format>(
    table: &Table<M>,
    field: &F,
    poly: &Poly<F, Rank1>,
) -> Vec<Sparse> {
    debug_assert!(
        (0..F::LANES).all(|k| field.lane(poly.coefs[0], k) == 1),
        "an element of the image is monic at each lane"
    );
    let n = table.format().variables();
    let mut lanes: Vec<Sparse> = (0..F::LANES)
        .map(|_| Sparse {
            exponents: Vec::with_capacity((poly.len() - 1) * n),
            coefs: Vec::with_capacity(poly.len() - 1),
        })
        .collect();
    let mut term = vec![0u32; n];
    for (&m, &c) in poly.monomials[1..].iter().zip(&poly.coefs[1..]) {
        // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
        table.unpack(m, &mut term);
        for (k, lane) in lanes.iter_mut().enumerate() {
            let v = field.lane(c, k);
            if v != 0 {
                lane.exponents.extend_from_slice(&term);
                lane.coefs.push(v);
            }
        }
    }
    lanes
}

/// The checker's verdict on `basis` and the cofactor matrix `cofactors`
/// of the instance `text`, in the order `named` on n variables, in the
/// mode `mode`, its cheap mode's points drawn from `seed`: `certified`
/// and `flaw` as `Outcome` holds them, or the bound that stopped the
/// check. `Run::finish` calls it, and so does the lift.
#[allow(
    clippy::too_many_arguments,
    reason = "the instance, basis, and cofactors as texts, the order and its count of variables, the checker's mode and seed, and the budget, none of which belongs with another"
)]
pub(crate) fn certify(
    text: &str,
    basis: &str,
    cofactors: &str,
    named: &MonomialOrder,
    n: usize,
    mode: CheckerMode,
    seed: u64,
    budget: &mut Budget,
) -> Result<(bool, Option<String>), Stop> {
    let mode = match mode {
        CheckerMode::Exact => Mode::Exact,
        CheckerMode::Cheap { points } => Mode::Cheap { points, seed },
    };
    // f4-gfp.26 practice: cooperative cancellation -- the checker calls a poll between its steps and stops where it answers true, not run on a thread of its own that is abandoned at the deadline
    let mut why = None;
    let mut poll = || match budget.tick() {
        Ok(()) => false,
        Err(stop) => {
            why = Some(stop);
            true
        }
    };
    let order = checker_order(named, n);
    let verdict = grobx_checker::check_polled(text, basis, cofactors, &order, mode, &mut poll);
    if let Some(why) = why {
        return Err(why);
    }
    // f4-gfp.35 design -- a result the checker rejects is returned with certified false and the flaw, not raised as an error (not measured)
    Ok(match verdict {
        Ok(Some(v)) => {
            let flaws: Vec<String> = [("groebner", v.groebner), ("reduced", v.reduced)]
                .into_iter()
                .filter_map(|(verdict, r)| r.err().map(|f| format!("{verdict}: {f:?}")))
                .collect();
            let flaw = (!flaws.is_empty()).then(|| flaws.join("; "));
            (flaws.is_empty(), flaw)
        }
        #[allow(
            clippy::unreachable,
            reason = "`check_polled` returns `Ok(None)` only where `poll` returned true, which set `why`, and debug-asserts it"
        )]
        Ok(None) => unreachable!("the checker stopped where the budget did not"),
        Err(e) => (
            false,
            Some(format!("the checker cannot read the result: {e:?}")),
        ),
    })
}

/// What a run that returns makes: the outcome, or the image where the run
/// makes one: over a field of more than one lane an image per lane for a
/// learn run, and a pair image for a replay.
enum Made {
    Outcome(Outcome),
    Image(Image),
    Lanes(Vec<Image>),
    Pair(PairImage),
}

/// Why a run starts, as its line in the call report names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Why {
    /// The call's first run, or over ℚ the lift's run at a prime.
    Start,
    /// The run before passed its width.
    Wider,
    /// The signature run before handed over.
    Handover,
    /// The replay before ran from a kept shape and met other terms.
    Refit,
    /// The grevlex run before was a lex call's whose basis FGLM does not
    /// change.
    Lex,
}

impl Why {
    fn name(self) -> &'static str {
        match self {
            Why::Start => "start",
            Why::Wider => "wider",
            Why::Handover => "handover",
            Why::Refit => "refit",
            Why::Lex => "lex",
        }
    }
}

// rule-breaches.16 design -- the replay engine holds the record it replays, and a match on the engine names each engine once, not a match on the engine and the record whose arm for a replay without a record no run reaches (not measured)
/// The engine a run computes with (`Run::engine`).
#[derive(Clone, Copy, Debug)]
enum Engine<'a> {
    /// The replay of this record.
    Replay(&'a Record),
    Signature,
    Plain,
}

impl Engine<'_> {
    /// Whether the engine is the signature run.
    fn signature(self) -> bool {
        match self {
            Engine::Signature => true,
            Engine::Replay(_) | Engine::Plain => false,
        }
    }
}

/// The call report's `ran` line, where `budget` holds the report: the run
/// ended by `outcome`, where it overflowed or met a denominator 0 mod p
/// `at` the read or in the run, or where the echelon ended it `at` the
/// echelon, at
/// `step` where it did not fit, with its `stats`, the elements of its
/// basis where it made one, and the signature run's S-pairs `fates`, and
/// its time since its `run` line.
fn report_ran(
    budget: &mut Budget,
    outcome: &str,
    at: Option<&str>,
    step: Option<u32>,
    stats: &Stats,
    elements: Option<usize>,
    fates: Option<&Pairs>,
) {
    let Some(report) = budget.report() else {
        return;
    };
    let number = report.runs();
    let time = Secs(report.since_run());
    report.line(
        "ran",
        format_args!(
            " number={number} outcome={outcome}{at}{step} rounds={rounds} reduced={reduced} \
             zero={zero}{elements} time={time}{fates}",
            at = Opt("at", at),
            step = Opt("step", step),
            rounds = stats.rounds,
            reduced = stats.reduced,
            zero = stats.zero,
            elements = Opt("elements", elements),
            fates = Fates(fates),
        ),
    );
}

// call-report.13 design -- the signature run's S-pair fates, sig::Pairs, are reported once, on its ran line, not per round (not measured)
/// The signature run's S-pairs as its `ran` line writes them: what became
/// of each S-pair over the run (`sig::Pairs`), `overflow` as
/// `past-format`. Nothing for another engine's run.
struct Fates<'a>(Option<&'a Pairs>);

impl fmt::Display for Fates<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(p) = self.0 else {
            return Ok(());
        };
        write!(
            f,
            " considered={considered} equal={equal} high={high} low={low} syzygy={syzygy} \
             koszul={koszul} queued={queued} past-format={past_format}",
            considered = p.considered,
            equal = p.equal,
            high = p.high,
            low = p.low,
            syzygy = p.syzygy,
            koszul = p.koszul,
            queued = p.queued,
            past_format = p.overflow
        )
    }
}

/// The call report's `fglm` line of a lex call's grevlex basis: the
/// verdict, the staircase's size where it was found, and the times of
/// the staircase and, where it was found, of the change.
fn report_fglm(budget: &mut Budget, verdict: &str, staircase: Option<usize>) {
    let Some(report) = budget.report() else {
        return;
    };
    let find = Secs(report.time(Stage::Find));
    let change = staircase.map(|_| Secs(report.time(Stage::Change)));
    report.line(
        "fglm",
        format_args!(
            " verdict={verdict}{staircase} find={find}{change}",
            staircase = Opt("staircase", staircase),
            change = Opt("change", change)
        ),
    );
}

/// What the call report calls a halt: `overflow`, `size`, `stop`,
/// `misfit`, or `handover`.
fn halt_name(halt: Halt) -> &'static str {
    match halt {
        Halt::Overflow => "overflow",
        Halt::Size => "size",
        Halt::Stop(_) => "stop",
        Halt::Misfit(_) => "misfit",
        Halt::Handover => "handover",
    }
}

/// The call report's `image` line: a run's image of `elements` elements,
/// with its time.
fn report_image(budget: &mut Budget, elements: usize) {
    if let Some(report) = budget.report() {
        let time = Secs(report.time(Stage::Image));
        report.line("image", format_args!(" elements={elements} time={time}"));
    }
}

/// The call report's `text` line: the text of `kind`, `basis` or
/// `cofactors`, written in `bytes` bytes, with its time, that of `stage`.
pub(crate) fn report_text(budget: &mut Budget, kind: &str, bytes: usize, stage: Stage) {
    if let Some(report) = budget.report() {
        let time = Secs(report.time(stage));
        report.line(
            "text",
            format_args!(" kind={kind} bytes={bytes} time={time}"),
        );
    }
}

/// The call report's `check` line: the checker in `mode` ruled, and
/// certified the result where `certified`, with its time.
pub(crate) fn report_check(budget: &mut Budget, mode: CheckerMode, certified: bool) {
    if let Some(report) = budget.report() {
        let (name, points) = match mode {
            CheckerMode::Exact => ("exact", None),
            CheckerMode::Cheap { points } => ("cheap", Some(points)),
        };
        let verdict = if certified { "certified" } else { "rejected" };
        let time = Secs(report.time(Stage::Check));
        report.line(
            "check",
            format_args!(
                " mode={name}{points} verdict={verdict} time={time}",
                points = Opt("points", points)
            ),
        );
    }
}

/// How a run in one format ends where no error ends the call.
enum Ran {
    /// The outcome or the image, None where the run is a lex call's
    /// grevlex run whose basis FGLM does not change: the ideal is not
    /// zero-dimensional, or its staircase holds more than
    /// `fglm::MAX_STAIRCASE` monomials.
    Done(Option<Made>),
    /// The signature run handed the call over to the plain run.
    Handover,
    /// A replay from a kept shape met an element whose terms differ from
    /// the shape's.
    Refit,
}

/// The checker's name for the call's order on n variables
/// (checker-orders.2, .3): a block order by its matrix, for each block in
/// turn the rows of grevlex as monagan-2007 §2.1 writes them, placed in the
/// block's columns: a row of ones, then −1 at the block's last variable, at
/// the one before it, and so on to its second.
fn checker_order(named: &MonomialOrder, n: usize) -> grobx_checker::Order {
    match named {
        MonomialOrder::Grevlex => grobx_checker::Order::Grevlex,
        MonomialOrder::Lex => grobx_checker::Order::Lex,
        MonomialOrder::Block(sizes) => {
            #[allow(
                clippy::expect_used,
                reason = "run_in took the same sizes for the same n"
            )]
            let block = Block::new(sizes, n).expect("a block order's sizes are fitted");
            let mut rows = Vec::with_capacity(n);
            for (s, e) in block.blocks(n) {
                let mut ones = vec![0; n];
                ones[s..e].fill(1);
                rows.push(ones);
                for i in (s + 1..e).rev() {
                    let mut row = vec![0; n];
                    row[i] = -1;
                    rows.push(row);
                }
            }
            grobx_checker::Order::Matrix(rows)
        }
    }
}

/// What a run that halted made, with the counts of the rounds whose
/// matrices it reduced:
/// a handover, an error, or the outcome of a run a bound stopped.
fn halted(halt: Halt, stats: Stats) -> Result<Ran, Error> {
    match halt {
        Halt::Overflow => Err(Error::Overflow { line: None }),
        Halt::Size => Err(Error::Size),
        Halt::Stop(why) => Ok(Ran::Done(Some(Made::Outcome(stopped(why, stats))))),
        Halt::Misfit(step) => Err(Error::Misfit { step }),
        Halt::Handover => Ok(Ran::Handover),
    }
}

/// The outcome of a run a bound stopped, with the counts of the rounds
/// whose matrices it reduced.
pub(crate) fn stopped(why: Stop, stats: Stats) -> Outcome {
    Outcome {
        basis: None,
        stopped: Some(why),
        cofactors: None,
        certified: None,
        flaw: None,
        stats,
        report: None,
    }
}
