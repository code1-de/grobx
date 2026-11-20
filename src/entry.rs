//! The entry into a run, for a call over GF(p) (`call::call`) and for the
//! lift over Q: picks the monomorphization of the run, the order's and the
//! monomial format's, and hands the result to the checker
//! where the call asks for it. A run whose degree passes its format
//! starts over in the next wider,
//! and a signature run that hands over goes on in the plain run from its
//! work, or starts over there (`f4::run_sig`). A run
//! for the lift over Q returns its result as an image (`Image`), or
//! over a field of more than one lane an image per lane, without
//! cofactors. Where the call asks for the call report, this module
//! writes its lines of the call, of each run's start, inputs, echelon, and
//! end, and of the image, the texts, FGLM, and the checker.

use std::fmt;

#[cfg(feature = "checker")]
use grobx_checker::Mode;
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, PROBE, Site, trim_before_texts};
#[cfg(feature = "cofactors")]
use crate::cofactors::Tracked;
use crate::cofactors::{Cofactors, NoCofactors};
use crate::f4::{Carried, HANDOVER, SigEnd};
use crate::fglm;
use crate::field::{Field, Gf2};
use crate::halt::Halt;
use crate::id::MonoId;
use crate::matrix;
use crate::module::Rank1;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format, WIDTHS, Width};
use crate::narrow;
use crate::order::{Block, Deglex, Grevlex, Lex, MonomialOrder, Order};
use crate::poly::{Elements, Exponents, Poly, Sparse};
use crate::report::{Opt, Or, Secs, Stage, cpu, on};
use crate::shape::{Built, Kept, Shape, Shaping};
use crate::sig::Pairs;
use crate::sort::RUN;
use crate::table::Table;
use crate::text::{Header, Unread, write_basis};
use crate::trace::{Learner, Record, Trace};
use crate::{CheckerMode, Error, Outcome, Params, Stats, Stop, Threads};

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
            deadline = Or(deadline, "none"),
            memory = Or(params.memory, "none"),
            cancel = on(params.cancel.is_some()),
            trace = if listens { "listens" } else { "off" },
            cpu = Or(cpu(), "none"),
        ),
    );
}

/// The call report's `read` line: the header as read, and the field it
/// decided, `gf2`, `gf31`, `gf64`, or `q`.
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

// call-report.10 design-checked -- call::call puts the report into the outcome, once, at the end, not each site that builds an outcome (not measured)
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
        // report-peak-inherited.4 design-checked -- where the resident set or the peak cannot be read the end line carries no process-rss field, not zeros, mimalloc's figures, or an error
        let rss = report.rss();
        report.line(
            "end",
            format_args!(
                " result={result}{bytes} time={time}{start}{peak}{grown}",
                bytes = Opt("bytes", bytes),
                start = Opt("process-rss-start", rss.map(|(s, _)| s)),
                peak = Opt("process-rss-peak", rss.map(|(_, p)| p)),
                grown = Opt("process-rss-grown", rss.map(|(s, p)| p.saturating_sub(s))),
            ),
        );
    }
    if let Some(report) = budget.take_report() {
        let (text, sink_error) = report.finish();
        outcome.report = Some(text);
        outcome.sink_error = sink_error;
    }
    outcome
}

/// The outcome of a call in the field `field`: what `call::call` did after it
/// read the header (gfp-large-primes.3).
#[allow(
    clippy::too_many_arguments,
    reason = "the field, the instance and what was read of it, the call's parameters, record, and order, and the budget, trace, and learner the run borrows, none of which belongs with another"
)]
pub(crate) fn call_in<F: Field, T: Trace, L: Learner>(
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

// q-lift.29 design-checked -- a run's result reaches the lift as an image, exponent vectors and residues read from the run's tables, not the run's canonical basis text and cofactor text read back (not measured)
/// One run's result over one prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Image {
    /// The leading monomials of the reduced basis, by increasing leading
    /// monomial, n exponents each.
    pub(crate) leads: Vec<u32>,
    /// The reduced basis, each element monic, by increasing leading
    /// monomial, without its leading term.
    pub(crate) basis: Vec<Sparse>,
    /// Per element, per generator of the instance, the cofactor; empty
    /// where the run tracked none.
    pub(crate) cofactors: Vec<Vec<Sparse>>,
    pub(crate) stats: Stats,
}

// q-lift-rest-time.1 cited-checked: dekleine-2001 §2, §1 -- a polynomial mapped to Z_p1 × … × Z_pb[x] is stored as its monomials once, each with its coefficients at the b primes beside it, and the polynomial operations are done on the b primes at once to amortize the monomial cost over the arithmetic
/// A pair replay's reduced basis at its two primes, without cofactors:
/// per element its terms but the leading one, once, with their
/// coefficients at the first prime, a coefficient 0 where the residue
/// there is 0, and their coefficients at the second beside.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PairImage {
    /// As `Image::leads`: at both primes.
    pub(crate) leads: Vec<u32>,
    /// Per element, its terms and their coefficients at the first prime.
    pub(crate) first: Vec<Sparse>,
    /// Per element, the coefficients of `first`'s terms at the second
    /// prime.
    pub(crate) second: Vec<Vec<u64>>,
    pub(crate) stats: Stats,
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

/// A run in the field `field` as `call_in` runs it, the widths `WIDTHS`,
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
    let made = run_in(
        field, text, params, record, WIDTHS, true, named, header, budget, trace, learner, kept,
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

/// The narrowest width a run over the field `F` takes: `W4` where the
/// field starts narrow (gf2-monomial-time.2), else `W8`.
fn narrowest<F: Field>() -> Width {
    if F::STARTS_NARROW {
        Width::W4
    } else {
        Width::W8
    }
}

/// What `call_in` and `image_in` run: the run in the field `field`, which
/// makes an image where `image` is true, and a replay's shape in `kept`. It
/// takes the widths from `first` to `last`, neither below the field's
/// narrowest width (`narrowest`).
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
        // gf2-monomial-time.8 design-checked -- a run raises both ends of the widths it is given to its field's narrowest width, so that a run over GF(p) asked for `W4` runs in `W8`, not an error where a caller names a width below the field's narrowest (log 0136-2)
        first: first.max(narrowest::<F>()),
        last: last.max(narrowest::<F>()),
        order: (),
    };
    // order-family.4 practice-checked: monomorphization -- the entry point matches the order once and runs a copy of the run compiled for Grevlex, Deglex, Lex, or Block, not one order type that holds which of them it is and matches at each comparison
    let made = match named {
        MonomialOrder::Grevlex => {
            run.with(Grevlex)
                .widening(header, trace, learner, budget, kept, Why::Start)?
        }
        // deglex-order.1 design-checked -- a deglex call runs F4 in deglex from its input, as a grevlex call does, not the grevlex basis changed to deglex by FGLM where the ideal is zero-dimensional, as a lex call does (log 0157-1)
        MonomialOrder::Deglex => {
            run.with(Deglex)
                .widening(header, trace, learner, budget, kept, Why::Start)?
        }
        MonomialOrder::Lex => return run.lex(header, trace, learner, budget, kept),
        // fglm-lex.19 design-checked -- FGLM is wired for lex alone, and a block order keeps F4 in the block order, not FGLM for every order of the family but grevlex where the ideal is zero-dimensional (log 0011-2)
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

    // fglm-lex.1 cited-checked: eder-2013a §5 -- in such cases it is more efficient to compute a Gröbner basis w.r.t. the graded reverse-lexicographical ordering and then to use a Gröbner conversion via FGLM; faugere-1993 §1 and abstract -- for the lexicographical basis it is usually much faster to compute the basis for the degree ordering and to use the algorithm than a direct computation by Buchberger's algorithm
    // fglm-lex.15 design-checked -- a record of a lex call whose basis FGLM changed holds the grevlex run's steps and says so, and its replay replays them in grevlex and changes the replayed basis, not a lex call in trace mode running F4 in lex so that the record's steps are lex's (not measured)
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
            // fglm-lex.20 design-checked -- a changed record whose replayed grevlex basis is not zero-dimensional, or whose staircase passes the bound, does not fit at the step one past the record's last, not a panic (not measured)
            if let Some(record) = self.record {
                return Err(Error::Misfit {
                    step: narrow::id(record.steps.len()).map_err(|_| Error::Size)? + 1,
                });
            }
            why = Why::Lex;
        }
        // fglm-lex.3 design-checked -- a lex call on an ideal that is not zero-dimensional returns its reduced lex basis, not an error (not measured)
        // fglm-lex.4 design-checked -- that basis comes from F4 in lex, not from the Gröbner walk from the grevlex basis (not measured)
        // fglm-lex.5 design-checked -- the lex run starts from the instance's generators, the call started over, not F4 in lex from the grevlex basis just computed (not measured)
        // fglm-lex.27 design-checked -- the lex run starts once the grevlex run has returned and given back its table, basis, and cofactors, not from inside the grevlex run's solve with them still held (not measured)
        // monomial-formats.13 design-checked -- the lex run picks its format from the input afresh and starts at the narrowest width, not in the width the grevlex run ended in (not measured)
        let made = self
            .with(Lex)
            .widening(header, trace, learner, budget, kept, why)?;
        Ok(made.expect("a run whose basis FGLM does not change returns an outcome"))
    }
}

impl<F: Field, O: Order<Dense4> + Order<Dense8> + Order<Dense16> + Order<Dense32>> Run<'_, F, O> {
    // monomial-formats.5 cited-checked: monagan-2007 §2.1 -- it is especially easy to determine an optimal packing for these orders using bounds on the total degree; demin-2024 §4.1 -- the monomial representation is selected at the beginning of the computation
    // monomial-formats.6 design-checked -- the run finds its format by reading the input in each format in turn from the narrowest, the reader's overflow naming a term past it, not a pass over the text for its largest term degree before the read (not measured)
    // monomial-formats.7 cited-checked: bachmann-1998 §3 -- ideally switch dynamically from one exponent size to the next larger one whenever it becomes necessary; with Singular this cannot be done yet
    // monomial-formats.8 design-checked -- the run starts over from its input in the wider format, under the same seed and the budget it has left, not its state converted to the wider format and continued from where it overflowed (log 0013-1)
    // monomial-formats.12 cited-checked: demin-2024 §4.1 -- the monomial representation is selected at the beginning of the computation, then the specialized functions that implement monomial arithmetic are just-in-time compiled and inlined, which relocates the work from running time to compile time; bachmann-1998 §3 -- the versions differ only in comparisons realized by function pointers, each a function call, instead of one inlined routine
    /// The run in the widths from `first` to `last`, each width a copy of
    /// the run compiled for its format: it starts in `first`, and where it
    /// returns `Error::Overflow`, from its input or from the run, over in
    /// the next wider width. In `last`, its `Error::Overflow` is returned.
    /// Where the signature run hands over and does not carry its work to
    /// the plain run (`f4::run_sig`), the call starts over in the plain run
    /// in the same width, and every later start is the plain run's. Where
    /// the bounded run passes its bound on its row work (`Ran::Outworked`),
    /// the call starts over in the signature run in the same width, and
    /// every later start is the signature run's (`After`). A
    /// replay that tracks no cofactors, where `kept` is Some,
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
        // q-replay-time.7 design-checked -- only a replay without cofactors keeps a shape or runs from one, and a replay that tracks cofactors builds its matrices from the record, not the cofactor tracking's matrices and table kept as well (not measured)
        let mut kept = kept.filter(|_| self.record.is_some() && !self.tracks());
        // q-replay-time.10 design-checked -- a replay where a shape is kept starts in the shape's width, not the widening from the narrowest width that every run starts (monomial-formats.6) (not measured)
        let mut width = kept
            .as_ref()
            .and_then(|k| k.width())
            .filter(|&w| self.first < w && w <= self.last)
            .unwrap_or(self.first);
        let mut after = After::Start;
        loop {
            let shape = kept.as_deref_mut();
            let got = match width {
                // gf2-monomial-time.6 design-checked -- the run's copy for `Dense4` stands under the field's constant `STARTS_NARROW`, so that a field that starts in `Dense8` compiles none, not a copy in every field's run, which grew the release CLI's text several times as much as GF(2)'s copy alone (log 0136-1, 0136-2)
                Width::W4 if F::STARTS_NARROW => self.start::<Dense4, T, L>(
                    header, after, trace, learner, budget, shape, width, why,
                ),
                #[allow(
                    clippy::unreachable,
                    reason = "a run starts in its field's narrowest width or wider (`run_in`), and widths only widen from there"
                )]
                Width::W4 => unreachable!("a run in W4 over a field that starts in W8"),
                Width::W8 => self.start::<Dense8, T, L>(
                    header, after, trace, learner, budget, shape, width, why,
                ),
                Width::W16 => self.start::<Dense16, T, L>(
                    header, after, trace, learner, budget, shape, width, why,
                ),
                Width::W32 => self.start::<Dense32, T, L>(
                    header, after, trace, learner, budget, shape, width, why,
                ),
            };
            match got {
                // gfp-unsolved.16 design-checked -- a handover the signature run does not carry (signature-run-losses.1) starts the call over from its input in the plain run, a new table and cofactor tracking, the same seed, and the budget left, not the plain run from the signature run's basis (log 0014-1)
                // q-learn-time.13 design-checked -- a pair learn that hands over runs its plain run over both lanes, each row on its own or ruled zero by a test (learn-zero-time.1), not the pair learn halting at the handover with the plain run at p alone and the verification's run at a fresh prime (log 0052-1)
                Ok(Ran::Handover) => (after, why) = (After::Handover, Why::Handover),
                // gf2-signature-losses.2 design-checked -- a run past its bound on the row work starts the call over from its input in the signature run, a new table, the same seed, and the budget left, not the run to its end, its elements carried into the signature run, nor both engines run in turns (log 0129-1)
                Ok(Ran::Outworked) => (after, why) = (After::Outworked, Why::Outworked),
                // q-replay-time.5 design-checked -- a replay whose terms differ from the shape starts over at its prime as a full replay, from its input, in a new table, not the full replay continued from the step where the terms differed, nor the prime dropped as a misfit (not measured)
                // q-replay-time.6 design-checked -- the full replay that a difference started keeps what it builds, and its shape replaces the kept one, not the kept shape kept and the full replay keeping nothing (not measured)
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

    // f4-gfp.38 design-checked -- the hash multipliers come from rand_xoshiro's Xoshiro256PlusPlus seeded by seed_from_u64, not from a generator written for the engine (not measured)
    /// An empty table of the format `format` whose hash multipliers are the
    /// first draws of the call's generator, seeded by the call's seed, and
    /// the generator after them.
    fn table<M: Format>(&self, format: M) -> (Table<M>, Xoshiro256PlusPlus) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.params.seed);
        let table = Table::new(format, &mut rng);
        (table, rng)
    }

    /// Whether the run tracks cofactors: where the call asks for cofactors,
    /// or for the checker in a run that makes no image.
    fn tracks(&self) -> bool {
        self.params.cofactors || (!self.image && self.params.checker.is_some())
    }

    // input-redundancy-time.8 design-checked -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors and, over a field that packs, where it is no bounded or plain run (gf2-echelon-losses.1), a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
    // gf2-echelon-losses.1 design-checked -- where the rounds reduce in packed rows, a run in the bounded or the plain engine takes its generators as read, without the substitution and the echelon, and the signature run takes them as over the other fields, not every run as read, the substitution or the echelon alone left out, nor the linear generators' echelon alone (log 0173-1, 0173-2)
    // gf2-echelon-losses.3 design-checked -- the rule asks `Field::PACKS`, the constant the packed reduction asks, not the characteristic 2 read at run time nor a parameter of the call (not measured)
    // gf2-echelon-losses.4 design-checked -- the plain engine takes its generators as read in every order over such a field, in a block order and in a lex call's F4 in lex too, not in the graded orders alone (not measured)
    /// Whether the run in the engine `engine` takes its generators in
    /// reduced row echelon form (`echelon`): where the engine is no bounded
    /// or plain run over a field that packs (`Field::PACKS`), the run tracks
    /// no cofactors, and its record, or where it has none `Params::echelon`,
    /// asks for it.
    fn echelons(&self, engine: Engine<'_>) -> bool {
        let packed = match engine {
            Engine::Bounded | Engine::Plain => F::PACKS,
            Engine::Signature | Engine::Replay(_) => false,
        };
        !packed && !self.tracks() && self.record.map_or(self.params.echelon, |r| r.echelon)
    }

    /// Reads the generators into a table of the format `M`, of the width
    /// `width`, in the run's order, takes them in reduced row echelon form
    /// where the run does (`echelons`), and solves, with the cofactor tracking
    /// where the run tracks cofactors (`tracks`), in the engine that follows
    /// `after` (`engine`). Where `kept` is Some, the run is a replay: where a
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
        after: After,
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
        if let Some(kept) = kept.as_deref_mut()
            && let Some(Shape { table, built }) = kept.get::<M>()
        {
            self.report_run::<M, NoCofactors, T, L>(budget, after, width, why, Some("use"));
            // q-replay-time.3 cited-checked: parisse-2013 §4, steps 5 and 6 -- the lists of all possible monomials are recorded for further primes during the first prime's computation, and a row holds monomial indices relative to that ordered list
            let generators = match self.read(header, table, budget) {
                Ok(generators) => generators,
                Err(unread) => return unread_ran(unread),
            };
            let engine = self.engine::<M>(after);
            let generators = match self.echelon(table, generators, engine, learner, budget) {
                Ok(generators) => generators,
                Err(halt) => return halted(halt, Stats::default()),
            };
            // A replay draws no combination.
            let ran = self.solve(
                table,
                generators,
                after,
                &mut NoCofactors,
                trace,
                learner,
                budget,
                0,
                0,
                Shaping::Use(built),
                width,
            );
            match ran {
                Ok(Ran::Refit) => {}
                Ok(Ran::Done(_) | Ran::Handover | Ran::Outworked) | Err(_) => kept.used += 1,
            }
            return ran;
        }
        let format = M::new(self.names.len());
        let (mut table, mut rng) = self.table(format);
        // A replay that tracks no cofactors where a shape may be kept keeps
        // what it builds.
        let shape = self.record.map(|_| {
            if kept.is_some() && !self.tracks() {
                "take"
            } else {
                "off"
            }
        });
        #[cfg(not(feature = "cofactors"))]
        debug_assert!(
            !self.tracks(),
            "call::built refuses the cofactors and the checker in a build without them"
        );
        if self.tracks() {
            #[cfg(feature = "cofactors")]
            self.report_run::<M, Tracked<F, M>, T, L>(budget, after, width, why, shape);
        } else {
            self.report_run::<M, NoCofactors, T, L>(budget, after, width, why, shape);
        }
        // order-family.13 cited-checked: faugere-1999 §2.5 -- to homogenize the list of polynomials is not efficient for big systems since parasite solutions are also computed; giovini-1991 §3 -- the strategy of homogenizing the basis is never very good
        // The run computes on the generators, not homogenized, in the call's order.
        let generators = match self.read(header, &mut table, budget) {
            Ok(generators) => generators,
            Err(unread) => return unread_ran(unread),
        };
        let engine = self.engine::<M>(after);
        let generators = match self.echelon(&mut table, generators, engine, learner, budget) {
            Ok(generators) => generators,
            Err(halt) => return halted(halt, Stats::default()),
        };
        // gfp-unsolved.7 design-checked -- a combination's coefficients are uniform in [1, p), from a Xoshiro256PlusPlus seeded by the call's generator after its other draws, not a generator of its own seeded by a constant or coefficients in [0, p) (not measured)
        #[cfg(feature = "cofactors")]
        if self.tracks() {
            // monomial-formats.9 design-checked -- the cofactor tracking's table takes the run's format, and an overflow in either table starts the call over, not a format of its own for the cofactor table widened apart from the run's (log 0013-1)
            let mut cof = Tracked::new(self.field, format, generators.len(), &mut rng);
            // f4-gfp.51 design-checked -- the checker's seed is drawn from the call's generator after the cofactor table's multipliers, not the call's seed passed as it is (not measured)
            let seed = rng.next_u64();
            let combine = rng.next_u64();
            return self.solve(
                &mut table,
                generators,
                after,
                &mut cof,
                trace,
                learner,
                budget,
                seed,
                combine,
                Shaping::Off,
                width,
            );
        }
        if let Some(kept) = kept {
            let combine = rng.next_u64();
            let mut built = Built::default();
            // q-replay-time.2 design-checked -- the shape comes from the first replay of a record, which keeps the matrices it builds, not the learn run's matrices with the rows the record leaves out taken out of them (log 0022-3)
            let ran = self.solve(
                &mut table,
                generators,
                after,
                &mut NoCofactors,
                trace,
                learner,
                budget,
                0,
                combine,
                Shaping::Take(&mut built),
                width,
            )?;
            match &ran {
                Ran::Done(Some(made @ (Made::Image(_) | Made::Lanes(_) | Made::Pair(_)))) => {
                    if let Err(why) = kept.put(width, Shape { table, built }, budget) {
                        return halted(Halt::Stop(why), made.stats());
                    }
                }
                Ran::Done(None | Some(Made::Outcome(_)))
                | Ran::Handover
                | Ran::Outworked
                | Ran::Refit => {}
            }
            Ok(ran)
        } else {
            let combine = rng.next_u64();
            self.solve(
                &mut table,
                generators,
                after,
                &mut NoCofactors,
                trace,
                learner,
                budget,
                0,
                combine,
                Shaping::Off,
                width,
            )
        }
    }

    // input-redundancy-time.9 design-checked -- the echelon has a line of the call report and a stage of its own, not its figures on the inputs line and its time in the read's (not measured)
    // gf2-echelon-losses.2 design-checked -- the run's engine, from `start`'s `after`, comes in for `echelons` to decide by, not a test of its own at `start`'s calls of the echelon (not measured)
    /// The generators of the run in the engine `engine` in reduced row
    /// echelon form where `echelons` holds for it, else as read; `learner`
    /// hears which. Before its echelon such a run substitutes its linear
    /// generators' pivots in its other generators (`substitute`). A replay
    /// follows its record's leading monomials (`echelon::replay`), and any
    /// other run chooses its pivots (`echelon::echelon`). Over GF(2), in a
    /// run that neither learns nor replays, the rows that are the squares
    /// of linear rows less their tails' field equations leave as zero
    /// (`echelon::squares_out`). The call report
    /// gets the `echelon` line, or where the echelon halts the run's `ran`
    /// line.
    fn echelon<M: Format, L: Learner>(
        &self,
        table: &mut Table<M>,
        generators: Vec<Poly<F, Rank1>>,
        engine: Engine<'_>,
        learner: &mut L,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, Rank1>>, Halt>
    where
        O: Order<M>,
    {
        let echelons = self.echelons(engine);
        learner.echelon(echelons);
        if !echelons {
            return Ok(generators);
        }
        // ccode-newton-gap.17 design-checked -- the substitution runs at each start of a run, as the read and the echelon do, after a handover that starts over and in a wider format among them, not the substitutes kept across a start over at the same prime (log 0118-1)
        let generators = if crate::substitute::substitutes() {
            self.substitute(table, generators, budget)?
        } else {
            generators
        };
        budget.open();
        let before = budget.counts().0;
        let count = generators.len();
        let (field, order) = (&self.field, &self.order);
        let echelon = match self.record {
            // run-generators-time.5 design-checked -- a replay whose record's learn took the echelon follows the record, the generators it names nonzero the pivot rows of their recorded leading monomials' columns and the others zero, not the pivot chosen again at each prime by `echelon`'s rule (log 0087-3, 0087-4)
            Some(record) => {
                crate::echelon::replay(field, table, order, generators, &record.inputs, budget)
            }
            None => crate::echelon::echelon(field, table, order, generators, budget),
        };
        // bivium-rest-gap.8 design-checked -- the squares' rule runs where the run neither learns nor replays, not in every run over GF(2), where a learn records the rows it left out as zero rows of the echelon, which the replay's echelon cannot make zero, so the replay of the record misfits (log 0181-2)
        let leaves = field.characteristic() == Gf2::P
            && !L::LISTENS
            && self.record.is_none()
            && crate::echelon::squares_out_on();
        let echelon = echelon.and_then(|mut e| {
            let squares = if leaves {
                crate::echelon::squares_out(table, &mut e.rows, budget)?
            } else {
                0
            };
            Ok((e, squares))
        });
        budget.mark(Stage::Echelon);
        let subtracted = budget.counts().0 - before;
        match echelon {
            // bivium-rest-gap.4 design-checked -- the echelon's line counts the rows the squares' rule left out, `squares`, and its rows are the nonzero rows the run takes, not the rank before the rule nor the rows left out unreported (not measured)
            Ok((e, squares)) => {
                if let Some(report) = budget.report() {
                    let rows = e.rows.iter().filter(|g| !g.is_zero()).count();
                    let time = Secs(report.time(Stage::Echelon));
                    report.line(
                        "echelon",
                        format_args!(
                            " generators={count} rows={rows} squares={squares} \
                             columns={columns} subtracted={subtracted} time={time}",
                            columns = e.columns
                        ),
                    );
                }
                Ok(e.rows)
            }
            Err(halt) => {
                let step = match halt {
                    Halt::Misfit(step) => Some(step),
                    Halt::Overflow
                    | Halt::Size
                    | Halt::Stop(_)
                    | Halt::Handover
                    | Halt::Outworked => None,
                };
                let at = Some("echelon");
                report_ran(
                    budget,
                    halt_name(halt),
                    at,
                    step,
                    None,
                    &Stats::default(),
                    None,
                    None,
                );
                Err(halt)
            }
        }
    }

    // ccode-newton-gap.10 design-checked -- the substitution runs where the run takes its echelon (Run::echelons), with no parameter of its own, not a Params field and a record flag of its own (not measured)
    // ccode-newton-gap.11 design-checked -- the call report writes a substitute line after inputs where the run substituted a pivot, its time in a stage of its own, and a halt in it writes ran with at=substitute, not the figures on the echelon line and the time in the echelon's stage (not measured)
    /// The generators with the linear generators' pivots substituted in the
    /// others (`substitute::substitute`). The call report gets the
    /// `substitute` line where a pivot was substituted, or where the
    /// substitution halts the run's `ran` line.
    fn substitute<M: Format>(
        &self,
        table: &mut Table<M>,
        generators: Vec<Poly<F, Rank1>>,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, Rank1>>, Halt>
    where
        O: Order<M>,
    {
        budget.open();
        let substituted =
            crate::substitute::substitute(&self.field, table, &self.order, generators, budget);
        budget.mark(Stage::Substitute);
        match substituted {
            Ok(s) => {
                if s.pivots > 0
                    && let Some(report) = budget.report()
                {
                    let time = Secs(report.time(Stage::Substitute));
                    report.line(
                        "substitute",
                        format_args!(
                            " linear={linear} pivots={pivots} changed={changed} stayed={stayed} \
                             terms={terms} after={after} time={time}",
                            linear = s.linear,
                            pivots = s.pivots,
                            changed = s.changed,
                            stayed = s.stayed,
                            terms = s.terms,
                            after = s.after
                        ),
                    );
                }
                Ok(s.generators)
            }
            Err(halt) => {
                let step = match halt {
                    Halt::Misfit(step) => Some(step),
                    Halt::Overflow
                    | Halt::Size
                    | Halt::Stop(_)
                    | Halt::Handover
                    | Halt::Outworked => None,
                };
                report_ran(
                    budget,
                    halt_name(halt),
                    Some("substitute"),
                    step,
                    None,
                    &Stats::default(),
                    None,
                    None,
                );
                Err(halt)
            }
        }
    }

    /// The generators read into `table` in the run's order, the read spent
    /// on `budget` (`Header::generators`) and timed for the call report: its
    /// `inputs` line, or where the read overflows the format, meets a
    /// denominator that is 0 mod p, or stops, the run's `ran` line.
    fn read<M: Format>(
        &self,
        header: &Header,
        table: &mut Table<M>,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, Rank1>>, Unread>
    where
        O: Order<M>,
    {
        budget.open();
        let read = header.generators(&self.field, table, &self.order, budget);
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
            Err(Unread::Error(Error::Overflow { .. })) => {
                report_ran(
                    budget,
                    "overflow",
                    at,
                    None,
                    None,
                    &Stats::default(),
                    None,
                    None,
                );
            }
            Err(Unread::Error(Error::NoInverse { .. })) => {
                report_ran(
                    budget,
                    "denominator",
                    at,
                    None,
                    None,
                    &Stats::default(),
                    None,
                    None,
                );
            }
            Err(Unread::Stop(_)) => {
                report_ran(
                    budget,
                    "stop",
                    at,
                    None,
                    None,
                    &Stats::default(),
                    None,
                    None,
                );
            }
            Err(Unread::Error(
                Error::Syntax { .. }
                | Error::UnknownVariable { .. }
                | Error::RepeatedVariable { .. }
                | Error::Characteristic { .. }
                | Error::Size
                | Error::Order { .. }
                | Error::Memory
                | Error::Clock
                | Error::Misfit { .. }
                | Error::Drops { .. }
                | Error::NotBuilt { .. },
            )) => {}
        }
        read
    }

    // order-family.14 cited-checked: eder-2013a §4 -- ordering critical pairs by sugar degree, signature-based computations slow down by a large factor; §5 -- signature-based computations w.r.t. not degree compatible orderings are rather inefficient in sorting critical pairs, and grevlex followed by FGLM or a Gröbner walk is more efficient
    // fglm-lex.2 design-checked -- the grevlex run of a lex call runs as a grevlex call's, the signature run where `Params::signatures` is on, not f4-gfp's engine whatever `signatures` says, as order-family.14 has it in an order that is not graded (log 0012-2)
    // gf2-signature-losses.1 design-checked -- over a field that packs, a call that tracks no cofactors and asks for signatures in a graded order starts in the plain run under a bound on its row work, not the signature run first as over the other fields, nor the plain run first over every field (log 0129-1, 0129-3)
    // gf2-signature-losses.8 design-checked -- a run that tracks cofactors, for the call's cofactors or its checker, starts in the signature run, not the bounded run there too (not measured)
    /// The engine of a run in the format `M` that follows `after`: the
    /// replay where there is a record; else, where the call asks for
    /// signatures and the order is graded, the bounded run where nothing
    /// came before, the field packs (`Field::PACKS`), and the run tracks no
    /// cofactors (`tracks`), the signature run where nothing came before or
    /// the bounded run passed its bound, and the plain run after a
    /// handover; else the plain run. A lex call's grevlex run is a run in
    /// grevlex.
    fn engine<M: Format>(&self, after: After) -> Engine<'_>
    where
        O: Order<M>,
    {
        if let Some(record) = self.record {
            return Engine::Replay(record);
        }
        if !(self.params.signatures && <O as Order<M>>::GRADED) {
            return Engine::Plain;
        }
        match after {
            After::Start if F::PACKS && !self.tracks() => Engine::Bounded,
            After::Start | After::Outworked => Engine::Signature,
            After::Handover => Engine::Plain,
        }
    }

    // call-report.11 design-checked -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
    /// The call report's `run` line of a run in the format `M`, of the
    /// width `width`, with the cofactor tracking `C`, in the engine that
    /// follows `after`, started for `why`; `shape` is a replay's, `use`,
    /// `take`, or `off`. The run's role, its engine and the reduction's mode
    /// each with what decided it, by the functions the engine decides by
    /// (`engine`, `f4::combines`, `f4::plain_combines`, `f4::learn_tests`,
    /// `matrix::tests_rows`, `matrix::zero_combinations`).
    fn report_run<M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>(
        &self,
        budget: &mut Budget,
        after: After,
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
        let engine = self.engine::<M>(after);
        // call-report.21 design-checked -- where signatures are off and the order is not graded, the run line names signatures-off, the condition `Run::engine` tests first, not order (not measured)
        let (name, by) = match engine {
            Engine::Replay(_) => ("replay", "record"),
            Engine::Signature => ("signature", "signatures"),
            Engine::Bounded => ("plain", "field"),
            Engine::Plain => match after {
                After::Handover => ("plain", "handover"),
                After::Start | After::Outworked if !self.params.signatures => {
                    ("plain", "signatures-off")
                }
                After::Start | After::Outworked => ("plain", "order"),
            },
        };
        let field = &self.field;
        let p = field.characteristic();
        // call-report.20 design-checked -- a run line reads each lane's prime from the residue of −1 there, p − 1, not a method added to the field contract for the report (not measured)
        let minus_one = field.neg(field.coef(1));
        let primes: Vec<String> = (0..F::LANES)
            .map(|k| (field.lane(minus_one, k) + 1).to_string())
            .collect();
        // `Gf2` takes the prime 2, `Gf31` and `Gf31Pair` the other primes
        // below 2^31, and the others those above (`text::Prime`,
        // `lift::Primes`).
        let small = p < 1 << 31;
        let kind = match (F::LANES, small) {
            // gf2-reduction-time.18 design-checked -- the run line's field kind is gf2 where the field has one lane and its prime is 2, as text::Prime takes Gf2 for it, not a name added to the field contract for the report (not measured)
            (1, true) if p == Gf2::P => "gf2",
            (1, true) => "gf31",
            (1, false) => "gf64",
            (_, true) => "gf31pair",
            (_, false) => "gf64pair",
        };
        let order = match self.named {
            _ if self.change => "grevlex",
            MonomialOrder::Grevlex => "grevlex",
            MonomialOrder::Deglex => "deglex",
            MonomialOrder::Lex => "lex",
            MonomialOrder::Block(_) => "block",
        };
        let combines = crate::f4::combines::<F, M, C, T, L>();
        let tests = crate::f4::learn_tests::<F, M, C, T, L>();
        let (reduce, reduce_by) = match engine {
            Engine::Replay(_) => ("rows", Some("record")),
            Engine::Signature if combines && matrix::tests_rows::<F>(p) => ("tests", None),
            Engine::Plain | Engine::Bounded if crate::f4::plain_combines::<F, M, C, T, L>(p) => {
                ("combinations", None)
            }
            // gf2-forced-zeros-time.13 design-checked -- the run line names `reduce=sketches` where the plain run over GF(2) may take them, from the function the reduction's arm agrees with, not `reduce=rows reduce-by=prime` (not measured)
            Engine::Plain | Engine::Bounded if crate::f4::plain_sketches::<F, M, C, T, L>() => {
                ("sketches", None)
            }
            // q-zero-work-time.5 design-checked -- the run line names a learn's plain run that takes tests `reduce=tests`, from the function `f4::run` decides by, not `reduce=rows reduce-by=learner` as for a learn's signature run (not measured)
            Engine::Plain | Engine::Bounded if tests && matrix::zero_combinations(p).is_some() => {
                ("tests", None)
            }
            Engine::Signature | Engine::Plain | Engine::Bounded => {
                let lanes = engine.signature() && F::LANES > 1;
                // q-zero-work-time.6 design-checked -- the learner is a reason for rows in a learn's signature run alone (learn-zero-time.3), and a learn's plain run at p = 2 or 3 names the prime, not the learner as the first reason in every learn run (not measured)
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
        // gfp-unsolved.17 design-checked -- a call that tracks cofactors does not hand over: the signature run computes it to the end, not the handover with cofactors (log 0014-1)
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

    /// Runs F4 in the engine that follows `after` (`engine`), the bounded
    /// run under its bound on its row work, or replays the record
    /// where there is one, under `shaping`, changes the basis to lex where
    /// the run is a lex call's grevlex run, and finishes the outcome; `seed`
    /// draws the checker's points, and `combine` the combinations of the
    /// plain run and of the signature run. Where the signature run carries
    /// its work at a handover, the plain run goes on from it (`carried`),
    /// and the outcome counts the rounds of both. `Ran::Refit` where a replay from
    /// a shape meets an element whose terms differ from the shape's. The
    /// call report gets the run's `ran` line, the carried run's `run` and
    /// `ran` lines, and a lex call's grevlex run its `fglm` line. `width` is
    /// the run's, which a carried run's `run` line names.
    #[allow(
        clippy::too_many_arguments,
        reason = "the table, generators, and cofactors of the run, its kind, the trace, learner, and budget it borrows, its seeds, and its shaping, none of which belongs with another"
    )]
    fn solve<M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>(
        &self,
        table: &mut Table<M>,
        generators: Vec<Poly<F, Rank1>>,
        after: After,
        cof: &mut C,
        trace: &mut T,
        learner: &mut L,
        budget: &mut Budget,
        seed: u64,
        combine: u64,
        shaping: Shaping<'_>,
        width: Width,
    ) -> Result<Ran, Error>
    where
        O: Order<M>,
    {
        let mut stats = Stats::default();
        let field = &self.field;
        let order = &self.order;
        let engine = self.engine::<M>(after);
        // The signature run's S-pairs.
        let mut fates = Pairs::new(true);
        // The signature run's counts where the plain run carried on from it.
        let mut before = None;
        // The bounded run's row work and bound (gf2-bound-losses.7).
        let mut worked = None;
        let run = match engine {
            Engine::Replay(record) => {
                // order-family.15 design-checked -- a record holds the order it was learned in, and a replay under another order does not fit, not a replay that runs in the record's order whatever the call names (not measured)
                // input-redundancy-time.8 design-checked -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
                if record.order == *self.named && !(record.echelon && C::TRACKS) {
                    let replayed = crate::f4::replay(
                        field, table, order, generators, record, cof, trace, budget, &mut stats,
                        shaping,
                    );
                    if let Some(run) = replayed.transpose() {
                        run
                    } else {
                        report_ran(budget, "refit", None, None, None, &stats, None, None);
                        return Ok(Ran::Refit);
                    }
                } else {
                    Err(Halt::Misfit(0))
                }
            }
            Engine::Signature => {
                // gfp-unsolved.17 design-checked -- a call that tracks cofactors does not hand over: the signature run computes it to the end, not the handover with cofactors (log 0014-1)
                let handover = (!C::TRACKS).then_some(HANDOVER);
                let ended = crate::f4::run_sig(
                    field, table, order, generators, cof, trace, budget, &mut stats, learner,
                    handover, &mut fates, combine,
                );
                match ended {
                    Ok(SigEnd::Basis(elements)) => Ok(elements),
                    // signature-run-losses.9 design-checked -- the outcome counts the rounds of the signature run and of the plain run that carried on from it, and each run's `ran` line its own, not the plain run's rounds alone as after a start over (not measured)
                    Ok(SigEnd::Carry(carried)) => {
                        report_ran(
                            budget,
                            "carry",
                            None,
                            None,
                            None,
                            &stats,
                            None,
                            Some(&fates),
                        );
                        before = Some(std::mem::take(&mut stats));
                        self.carried(
                            table, carried, cof, trace, learner, budget, &mut stats, combine, width,
                        )
                    }
                    Err(halt) => Err(halt),
                }
            }
            Engine::Plain => crate::f4::run(
                field, table, order, generators, cof, trace, budget, &mut stats, learner, combine,
            ),
            Engine::Bounded => {
                let from = budget.row_work();
                let bound = budget.bound(PLAIN_WORK);
                let run = crate::f4::run(
                    field, table, order, generators, cof, trace, budget, &mut stats, learner,
                    combine,
                );
                budget.unbound();
                worked = Some(Worked {
                    work: budget.row_work() - from,
                    bound,
                });
                run
            }
        };
        if budget.report().is_some() {
            let (outcome, at, step, elements, worked) = match &run {
                Ok(elements) => ("basis", None, None, Some(elements.len()), None),
                Err(Halt::Handover) => ("handover", None, None, None, None),
                // gf2-signature-losses.9 design-checked -- the bounded run's run line names engine=plain engine-by=field, its ran line outcome=outworked, and the next run line why=outworked, not a line of its own for the bound (not measured)
                // gf2-bound-losses.7 design-checked -- the outworked run's ran line names the row work the run spent and its bound, work= and bound=, so that a halt by the projection reads work at most the bound, not a field that names the rule that halted it, nor no field (not measured)
                Err(Halt::Outworked) => ("outworked", None, None, None, worked),
                Err(Halt::Overflow) => ("overflow", Some("run"), None, None, None),
                Err(Halt::Size) => ("size", None, None, None, None),
                Err(Halt::Stop(_)) => ("stop", None, None, None, None),
                Err(Halt::Misfit(step)) => ("misfit", None, Some(*step), None, None),
            };
            let fates = (engine.signature() && before.is_none()).then_some(&fates);
            report_ran(budget, outcome, at, step, worked, &stats, elements, fates);
        }
        if let Some(before) = before {
            stats = before.and(stats);
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
                    let verdict = match fglm::zero_dimensional(table, &leads, budget) {
                        Ok(true) => "past-bound",
                        Ok(false) => "positive-dimensional",
                        Err(why) => {
                            report_fglm(budget, halt_name(Halt::Stop(why)), None);
                            return halted(Halt::Stop(why), stats);
                        }
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
        // fglm-lex.18 design-checked -- a lex call's stats are those of the run that made its basis, the grevlex run's where FGLM changed it, not the sum of both runs where the call started over (not measured)
        let made = self.finish(table, elements, &Lex, cof, stats, budget, seed);
        Ok(Ran::Done(Some(made)))
    }

    // signature-run-losses.8 design-checked -- the call takes the signature run's carry, writes the run lines, makes the fresh table, and runs the plain run from it, not the signature run calling the plain run from inside (not measured)
    /// The plain run from the signature run's carry `carried`, which `table`
    /// holds (signature-run-losses.1): the carried elements move into a
    /// fresh table that takes `table`'s place, drawn as `start` draws a
    /// run's (signature-run-losses.5); the call report gets the plain run's
    /// `run` line; and the plain run goes on from the carry
    /// (`f4::run_carried`), its rounds counted in `stats`, its combinations
    /// drawn by `combine`. `width` is the run's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the table, carry, and cofactors of the run, the trace, learner, and budget it borrows, its counts, its seed, and its width, none of which belongs with another"
    )]
    fn carried<M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>(
        &self,
        table: &mut Table<M>,
        mut carried: Carried<F, Rank1, C::Element>,
        cof: &mut C,
        trace: &mut T,
        learner: &mut L,
        budget: &mut Budget,
        stats: &mut Stats,
        combine: u64,
        width: Width,
    ) -> Result<Elements<F, Rank1, C::Element>, Halt>
    where
        O: Order<M>,
    {
        let (mut fresh, _) = self.table(*table.format());
        crate::f4::carry_table(table, &mut fresh, &mut carried, budget)?;
        *table = fresh;
        self.report_run::<M, C, T, L>(budget, After::Handover, width, Why::Carry, None);
        crate::f4::run_carried(
            &self.field,
            table,
            &self.order,
            carried,
            cof,
            trace,
            budget,
            stats,
            learner,
            combine,
        )
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
        if self.image {
            return match self.image_of(table, &polys, &rows, texts, cof, stats, budget) {
                Ok(made) => made,
                Err(why) => Made::Outcome(stopped(why, stats)),
            };
        }
        // freed-heap-text-peak.1 design-checked -- once a call's result is complete, before its basis text, the call gives the allocators' free memory back, not at no point, which leaves what the run freed resident beside the text, nor at every step of the call (log 0116-1)
        // freed-heap-text-peak.5 design-checked -- the basis elements stay until the basis text is written, not each dropped after its text with the free memory given back each 8 MiB dropped (log 0116-1)
        trim_before_texts();
        // call-stop-latency.73 design-checked -- the trim before the texts is followed by a full check at Site::Text, so that the allocator's trim ends the stretch it falls in, not the trim left between the result's last spend and the texts' first check (log 0138-8)
        if let Err(why) = budget.check(|| {}, Site::Text) {
            return Made::Outcome(stopped(why, stats));
        }
        // text-stop-latency.1 design-checked -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
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
        // text-stop-latency.11 design-checked -- no full check of the texts' own follows the last text, not a full check after the texts (not measured)
        // The checker, where the call asks for it, comes after the texts
        // and ticks the budget as it polls: a stop there drops them.
        let checked = check(
            self.text,
            &basis,
            cofactors.as_deref(),
            self.named,
            self.names.len(),
            params.checker,
            || seed,
            budget,
        );
        let (certified, flaw) = match checked {
            Ok(checked) => checked,
            Err(why) => return Made::Outcome(stopped(why, stats)),
        };
        Made::Outcome(Outcome {
            basis: Some(basis),
            stopped: None,
            cofactors: if params.cofactors { cofactors } else { None },
            certified,
            flaw,
            stats,
            report: None,
            sink_error: None,
        })
    }

    // call-stop-latency.28 design-checked -- the images of a run's result spend a unit per term and exponent they read and write, PROBE a monomial read from the table, each lane's copy of the leads a unit a word, and a kept shape's copy a unit an entry, as they go, not unspent before the lift's next check (log 0138-6)
    /// The image of the reduced basis `polys`, with the cofactors `rows` in
    /// `texts`, or over a field of more than one lane, without cofactors, an
    /// image per lane for a learn run and a pair image for a replay
    /// (`pair_image`). Spent at `Site::Images` as `leads`, `sparse`,
    /// `sparse_lanes`, and `pair_image` say, each lane's copy of the
    /// leading monomials a unit an exponent, and the cofactors as
    /// `Cofactors::image` spends. The stop where the budget's check stops
    /// the call.
    #[allow(
        clippy::too_many_arguments,
        reason = "the table, the elements and their cofactors, the order of the cofactors' image, the tracking, the counts, and the budget, none of which belongs with another"
    )]
    fn image_of<M: Format, P: Order<M>, C: Cofactors<F, M>>(
        &self,
        table: &Table<M>,
        polys: &[Poly<F, Rank1>],
        rows: &[C::Element],
        texts: &P,
        cof: &C,
        stats: Stats,
        budget: &mut Budget,
    ) -> Result<Made, Stop> {
        if F::LANES > 1 {
            // A run of several lanes is a replay or a learn run, neither of
            // which tracks cofactors (q-later-runs-time.6, q-lift.12).
            debug_assert!(!C::TRACKS, "no cofactors over several lanes");
            let leads = leads(table, polys, budget)?;
            if self.record.is_some() {
                debug_assert_eq!(F::LANES, 2, "a pair replay runs over two primes");
                let image = pair_image(table, &self.field, polys, leads, stats, budget)?;
                budget.mark(Stage::Image);
                report_image(budget, polys.len());
                return Ok(Made::Pair(image));
            }
            let mut lanes: Vec<Vec<Sparse>> = vec![Vec::with_capacity(polys.len()); F::LANES];
            for g in polys {
                let sparse = sparse_lanes(table, &self.field, g, budget)?;
                for (k, sparse) in sparse.into_iter().enumerate() {
                    lanes[k].push(sparse);
                }
            }
            let mut images = Vec::with_capacity(F::LANES);
            for basis in lanes {
                images.push(Image {
                    leads: leads.clone(),
                    basis,
                    cofactors: Vec::new(),
                    stats,
                });
                budget.spend(leads.len(), Site::Images)?;
            }
            budget.mark(Stage::Image);
            report_image(budget, polys.len());
            return Ok(Made::Lanes(images));
        }
        let mut basis = Vec::with_capacity(polys.len());
        for g in polys {
            basis.push(sparse(table, &self.field, g, budget)?);
        }
        // cofactor-stop-latency.1 design-checked -- every stage that works on cofactors spends units of work on the call's budget as it works, the absorb's merge and append over Q among them, and a stage stops where the budget's check stops the call, not the ticks per row and generator and per part, and nothing for the image and the merge (log 0058-1)
        let cofactors = cof.image(texts, rows, budget)?;
        budget.mark(Stage::Image);
        report_image(budget, polys.len());
        Ok(Made::Image(Image {
            leads: leads(table, polys, budget)?,
            basis,
            cofactors,
            stats,
        }))
    }
}

// result-time.7 design-checked -- the image carries each element's leading monomial apart from its other terms, and the lift takes the other terms as they are, not the whole element, whose exponents and coefficients the lift moves down by the leading term (log 0033-1, 0042-1)
// q-image-exponents-peak.9 design-checked -- the leading monomials stay in `u32`, one per element against a part's many terms, not in bytes too (not measured)
/// The leading monomials of the reduced basis `polys`, n exponents each,
/// spent at `Site::Images` `PROBE` units and its exponents a unit each an
/// element, `sort::RUN` elements at a time. A stop where the budget's
/// check stops the call.
fn leads<F: Field, M: Format>(
    table: &Table<M>,
    polys: &[Poly<F, Rank1>],
    budget: &mut Budget,
) -> Result<Vec<u32>, Stop> {
    let n = table.format().variables();
    let mut leads = Vec::with_capacity(polys.len() * n);
    for chunk in polys.chunks(RUN) {
        for g in chunk {
            // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
            table.exponents(g.monomials[0], &mut leads);
        }
        budget.spend(chunk.len() * (PROBE + n), Site::Images)?;
    }
    Ok(leads)
}

/// The units of work an image spends per term on `n` variables: its
/// monomial read from the table, `PROBE`, and its exponents written and
/// copied, a unit each.
fn term_units(n: usize) -> usize {
    PROBE + 2 * n
}

// result-time.8 design-checked -- the image hands over exponents, unpacked from the run's words, not the run's monomial ids with the lift reading exponents only where an image's terms differ from a part's (log 0042-1)
/// The monic polynomial `poly` of the run's table `table` as the lift reads
/// it: its terms but the leading one (result-time.7). Each term spends
/// `term_units` and a unit for its coefficient at `Site::Images`,
/// `sort::RUN` terms at a time. A stop where the budget's check stops the
/// call.
fn sparse<F: Field, M: Format>(
    table: &Table<M>,
    field: &F,
    poly: &Poly<F, Rank1>,
    budget: &mut Budget,
) -> Result<Sparse, Stop> {
    debug_assert_eq!(
        field.value(poly.coefs[0]),
        1,
        "an element of the image is monic"
    );
    let n = table.format().variables();
    let mut exponents = Exponents::with_capacity((poly.len() - 1) * n);
    let mut coefs = Vec::with_capacity(poly.len() - 1);
    let mut term = vec![0; n];
    let terms = poly.monomials[1..]
        .chunks(RUN)
        .zip(poly.coefs[1..].chunks(RUN));
    for (ms, cs) in terms {
        for &m in ms {
            // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
            table.unpack(m, &mut term);
            exponents.extend_from_slice(&term);
        }
        coefs.extend(cs.iter().map(|&c| field.value(c)));
        budget.spend(ms.len() * (1 + term_units(n)), Site::Images)?;
    }
    Ok(Sparse { exponents, coefs })
}

// q-lift-rest-time.1 cited-checked: dekleine-2001 §2, §1 -- a polynomial mapped to Z_p1 × … × Z_pb[x] is stored as its monomials once, each with its coefficients at the b primes beside it, and the polynomial operations are done on the b primes at once to amortize the monomial cost over the arithmetic
/// The reduced basis `polys` of a pair replay, each element monic at both
/// lanes of `field`, with its leading monomials `leads`, as the lift reads
/// it: per element its terms but the leading one, each term's exponents
/// read from the table and written once, with its residue at the first
/// lane, 0 among them, and at the second beside.
/// Each term spends `term_units` and a unit for each lane's coefficient at
/// `Site::Images`, `sort::RUN` terms at a time. A stop where the budget's
/// check stops the call.
fn pair_image<F: Field, M: Format>(
    table: &Table<M>,
    field: &F,
    polys: &[Poly<F, Rank1>],
    leads: Vec<u32>,
    stats: Stats,
    budget: &mut Budget,
) -> Result<PairImage, Stop> {
    debug_assert_eq!(F::LANES, 2, "a pair image holds two lanes");
    let n = table.format().variables();
    let mut first = Vec::with_capacity(polys.len());
    let mut second = Vec::with_capacity(polys.len());
    let mut term = vec![0; n];
    for poly in polys {
        debug_assert!(
            (0..F::LANES).all(|k| field.lane(poly.coefs[0], k) == 1),
            "an element of the image is monic at each lane"
        );
        let terms = poly.len() - 1;
        let mut exponents = Exponents::with_capacity(terms * n);
        let mut coefs = Vec::with_capacity(terms);
        let mut at_second = Vec::with_capacity(terms);
        let chunks = poly.monomials[1..]
            .chunks(RUN)
            .zip(poly.coefs[1..].chunks(RUN));
        for (ms, cs) in chunks {
            for (&m, &c) in ms.iter().zip(cs) {
                // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
                table.unpack(m, &mut term);
                exponents.extend_from_slice(&term);
                coefs.push(field.lane(c, 0));
                at_second.push(field.lane(c, 1));
            }
            budget.spend(ms.len() * (F::LANES + term_units(n)), Site::Images)?;
        }
        first.push(Sparse { exponents, coefs });
        second.push(at_second);
    }
    Ok(PairImage {
        leads,
        first,
        second,
        stats,
    })
}

// q-later-runs-time.11 design-checked -- a pair's images read each term's exponents from the table once for every lane, and each leaves out the terms whose residue at its prime is 0, not each lane's image read from the table on its own (log 0039-1)
/// The polynomial `poly`, monic at each lane of `field`, as the lift reads
/// it at each lane: per lane its terms but the leading one whose residue
/// there is not 0, each term's exponents read from the table once for
/// every lane. The learn at two primes makes its images so. Each term
/// spends `term_units` and per further lane its exponents' copy and its
/// coefficient, a unit each, at `Site::Images`, `sort::RUN` terms at a
/// time. A stop where the budget's check stops the call.
fn sparse_lanes<F: Field, M: Format>(
    table: &Table<M>,
    field: &F,
    poly: &Poly<F, Rank1>,
    budget: &mut Budget,
) -> Result<Vec<Sparse>, Stop> {
    debug_assert!(
        (0..F::LANES).all(|k| field.lane(poly.coefs[0], k) == 1),
        "an element of the image is monic at each lane"
    );
    let n = table.format().variables();
    let mut lanes: Vec<Sparse> = (0..F::LANES)
        .map(|_| Sparse {
            exponents: Exponents::with_capacity((poly.len() - 1) * n),
            coefs: Vec::with_capacity(poly.len() - 1),
        })
        .collect();
    let mut term = vec![0u32; n];
    let chunks = poly.monomials[1..]
        .chunks(RUN)
        .zip(poly.coefs[1..].chunks(RUN));
    for (ms, cs) in chunks {
        for (&m, &c) in ms.iter().zip(cs) {
            // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
            table.unpack(m, &mut term);
            for (k, lane) in lanes.iter_mut().enumerate() {
                let v = field.lane(c, k);
                if v != 0 {
                    lane.exponents.extend_from_slice(&term);
                    lane.coefs.push(v);
                }
            }
        }
        let lanes = F::LANES * (1 + n) - n;
        budget.spend(ms.len() * (lanes + term_units(n)), Site::Images)?;
    }
    Ok(lanes)
}

/// The checker's verdict where the call asks for it, in the mode
/// `checker` (`certify`), its cheap mode's points drawn from `seed()`:
/// `certified` and `flaw` as `Outcome` holds them, with the checker's
/// stage and its line of the call report, or the bound that stopped the
/// check. (None, None), and no draw, where the call asks for none.
/// `Run::finish` calls it, and so does the lift.
#[cfg(feature = "checker")]
#[allow(
    clippy::too_many_arguments,
    reason = "the instance, basis, and cofactors as texts, the order and its count of variables, the checker's mode and seed, and the budget, none of which belongs with another"
)]
pub(crate) fn check(
    text: &str,
    basis: &str,
    cofactors: Option<&str>,
    named: &MonomialOrder,
    n: usize,
    checker: Option<CheckerMode>,
    seed: impl FnOnce() -> u64,
    budget: &mut Budget,
) -> Result<(Option<bool>, Option<String>), Stop> {
    let Some(mode) = checker else {
        return Ok((None, None));
    };
    #[allow(
        clippy::expect_used,
        reason = "a call with a checker has the cofactors: Run::finish past the image's return tracks them, and the lift lifts them"
    )]
    let matrix = cofactors.expect("the checker's cofactors");
    let seed = seed();
    budget.open();
    let (verdict, flaw) = certify(text, basis, matrix, named, n, mode, seed, budget)?;
    budget.mark(Stage::Check);
    report_check(budget, mode, verdict);
    Ok((Some(verdict), flaw))
}

/// A build without the feature `checker` certifies nothing: (None, None),
/// since `call::built` refuses a call that asks for the checker.
#[cfg(not(feature = "checker"))]
#[allow(
    clippy::too_many_arguments,
    reason = "the signature of the build with the checker, which its callers take"
)]
#[allow(
    clippy::unnecessary_wraps,
    reason = "the build with the checker returns the bound that stopped it"
)]
pub(crate) fn check(
    _text: &str,
    _basis: &str,
    _cofactors: Option<&str>,
    _named: &MonomialOrder,
    _n: usize,
    checker: Option<CheckerMode>,
    _seed: impl FnOnce() -> u64,
    _budget: &mut Budget,
) -> Result<(Option<bool>, Option<String>), Stop> {
    debug_assert!(
        checker.is_none(),
        "call::built refuses the checker in a build without it"
    );
    Ok((None, None))
}

/// The checker's verdict on `basis` and the cofactor matrix `cofactors`
/// of the instance `text`, in the order `named` on n variables, in the
/// mode `mode`, its cheap mode's points drawn from `seed`: `certified`
/// and `flaw` as `Outcome` holds them, or the bound that stopped the
/// check.
#[cfg(feature = "checker")]
#[allow(
    clippy::too_many_arguments,
    reason = "the instance, basis, and cofactors as texts, the order and its count of variables, the checker's mode and seed, and the budget, none of which belongs with another"
)]
fn certify(
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
    let order = checker_order(named, n, budget)?;
    // f4-gfp.26 practice-checked: cooperative cancellation -- the checker calls a poll between its steps and stops where it answers true, not run on a thread of its own that is abandoned at the deadline
    let mut why = None;
    let mut poll = || match budget.tick(Site::Checker) {
        Ok(()) => false,
        Err(stop) => {
            why = Some(stop);
            true
        }
    };
    let verdict = grobx_checker::check_polled(text, basis, cofactors, &order, mode, &mut poll);
    if let Some(why) = why {
        return Err(why);
    }
    // f4-gfp.35 design-checked -- a result the checker rejects is returned with certified false and the flaw, not raised as an error (not measured)
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

impl Made {
    /// The counts of the run that made it, which each lane's image holds
    /// alike.
    fn stats(&self) -> Stats {
        match self {
            Made::Outcome(outcome) => outcome.stats,
            Made::Image(image) => image.stats,
            Made::Lanes(images) => {
                debug_assert!(!images.is_empty(), "an image per lane");
                images.first().map_or_else(Stats::default, |i| i.stats)
            }
            Made::Pair(image) => image.stats,
        }
    }
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
    /// The run before passed the bound on its row work.
    Outworked,
    /// The signature run before handed over its work, which the plain run
    /// goes on from (signature-run-losses.1).
    Carry,
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
            Why::Outworked => "outworked",
            Why::Carry => "carry",
            Why::Refit => "refit",
            Why::Lex => "lex",
        }
    }
}

// gf2-signature-losses.7 design-checked -- the bound holds for the run that starts the call alone, and what came before persists over the widening's later starts, so that a start in a wider format after the bound runs the signature run and the plain run after a handover runs without the bound, not each start bounded afresh, nor the bound kept over the plain run after a handover (not measured)
/// What came before a run's start in the widening, which decides its
/// engine with the call (`Run::engine`). It persists over the widening's
/// later starts.
#[derive(Clone, Copy, Debug)]
enum After {
    /// Nothing: the run is the call's first engine.
    Start,
    /// The signature run handed over: the plain run.
    Handover,
    /// The bounded run passed its bound on its row work: the signature run.
    Outworked,
}

// gf2-signature-losses.4 tuned-checked -- the bounded run halts once its row work passes 2^32, not 2^31 or 2^33 (log 0129-1, 0129-3)
/// The row work (`Budget::bound`) past which the bounded run halts, done
/// or projected over a round in sparse rows (`Budget::projected_past`),
/// and the call starts over in the signature run: 2^32. In session 0129-3 the
/// three bounds differed on the two calls of the tuning set whose plain
/// run passes them alone, where 2^33 took more time and 2^31 less, by
/// less than the runs' seeds moved the mean; 2^31 lies within a few
/// percent of the most row work a plain run spent where it took less time
/// than the signature run.
const PLAIN_WORK: u64 = 1 << 32;

// rule-breaches.16 practice-checked: make illegal states unrepresentable -- the replay engine holds the record it replays, and a match on the engine names each engine once, not a match on the engine and the record whose arm for a replay without a record no run reaches (not measured)
/// The engine a run computes with (`Run::engine`).
#[derive(Clone, Copy, Debug)]
enum Engine<'a> {
    /// The replay of this record.
    Replay(&'a Record),
    Signature,
    Plain,
    /// The plain run under a bound of `PLAIN_WORK` on its row work.
    Bounded,
}

impl Engine<'_> {
    /// Whether the engine is the signature run.
    fn signature(self) -> bool {
        match self {
            Engine::Signature => true,
            Engine::Replay(_) | Engine::Plain | Engine::Bounded => false,
        }
    }
}

/// The call report's `ran` line, where `budget` holds the report: the run
/// ended by `outcome`, where it overflowed or met a denominator 0 mod p
/// `at` the read or in the run, or where the echelon ended it `at` the
/// echelon, at
/// `step` where it did not fit, the bounded run's row work and bound
/// `worked` where its row work halted it, with its `stats`, the elements
/// of its basis where it made one, and the signature run's S-pairs
/// `fates`, and its time since its `run` line.
#[allow(
    clippy::too_many_arguments,
    reason = "the budget, the outcome, where and at which step the run ended, the row work that halted it, its counts, its elements, and the S-pairs' fates, each a field of the line of its own"
)]
fn report_ran(
    budget: &mut Budget,
    outcome: &str,
    at: Option<&str>,
    step: Option<u32>,
    worked: Option<Worked>,
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
            " number={number} outcome={outcome}{at}{step}{work}{bound} rounds={rounds} \
             reduced={reduced} zero={zero}{elements} time={time}{fates}",
            at = Opt("at", at),
            step = Opt("step", step),
            work = Opt("work", worked.map(|w| w.work)),
            bound = Opt("bound", worked.map(|w| w.bound)),
            rounds = stats.rounds,
            reduced = stats.reduced,
            zero = stats.zero,
            elements = Opt("elements", elements),
            fates = Fates(fates),
        ),
    );
}

/// The bounded run's row work and the units of its bound
/// (`Budget::bound`), both from the run's start, which its `ran` line
/// names where the run halted outworked (gf2-bound-losses.7).
#[derive(Clone, Copy, Debug)]
struct Worked {
    work: u64,
    bound: u64,
}

// call-report.13 design-checked -- the signature run's S-pair fates, sig::Pairs, are reported once, on its ran line, not per round (not measured)
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
/// `misfit`, `handover`, or `outworked`.
fn halt_name(halt: Halt) -> &'static str {
    match halt {
        Halt::Overflow => "overflow",
        Halt::Size => "size",
        Halt::Stop(_) => "stop",
        Halt::Misfit(_) => "misfit",
        Halt::Handover => "handover",
        Halt::Outworked => "outworked",
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
#[cfg(feature = "checker")]
fn report_check(budget: &mut Budget, mode: CheckerMode, certified: bool) {
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
    /// The run passed the bound on its row work (`Halt::Outworked`).
    Outworked,
    /// A replay from a kept shape met an element whose terms differ from
    /// the shape's.
    Refit,
}

/// The checker's name for the call's order on n variables
/// (checker-orders.2, .3): deglex by its matrix, a row of ones, then 1 at the
/// first variable, at the second, and so on to the one before the last; a
/// block order by its matrix, for each block in
/// turn the rows of grevlex as monagan-2007 §2.1 writes them, placed in the
/// block's columns: a row of ones, then −1 at the block's last variable, at
/// the one before it, and so on to its second. Each row spends a unit an
/// entry at `Site::Checker`; a stop where the budget's check stops the
/// call.
#[cfg(feature = "checker")]
fn checker_order(
    named: &MonomialOrder,
    n: usize,
    budget: &mut Budget,
) -> Result<grobx_checker::Order, Stop> {
    Ok(match named {
        MonomialOrder::Grevlex => grobx_checker::Order::Grevlex,
        // deglex-order.3 design-checked -- the checker takes deglex as the matrix order of a row of ones and the unit rows of the first n − 1 variables, as it takes a block order, not an order of the checker's own for deglex (log 0157-1)
        MonomialOrder::Deglex => {
            let mut rows = Vec::with_capacity(n);
            rows.push(vec![1; n]);
            budget.spend(n, Site::Checker)?;
            for i in 0..n.saturating_sub(1) {
                let mut row = vec![0; n];
                row[i] = 1;
                rows.push(row);
                budget.spend(n, Site::Checker)?;
            }
            grobx_checker::Order::Matrix(rows)
        }
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
                budget.spend(n, Site::Checker)?;
                for i in (s + 1..e).rev() {
                    let mut row = vec![0; n];
                    row[i] = -1;
                    rows.push(row);
                    budget.spend(n, Site::Checker)?;
                }
            }
            grobx_checker::Order::Matrix(rows)
        }
    })
}

/// What a run that halted made, with the counts of the rounds whose
/// matrices it reduced:
/// a handover, a run past the bound on its row work, an error, or the
/// outcome of a run a bound stopped.
fn halted(halt: Halt, stats: Stats) -> Result<Ran, Error> {
    match halt {
        Halt::Overflow => Err(Error::Overflow { line: None }),
        Halt::Size => Err(Error::Size),
        Halt::Stop(why) => Ok(Ran::Done(Some(Made::Outcome(stopped(why, stats))))),
        Halt::Misfit(step) => Err(Error::Misfit { step }),
        Halt::Handover => Ok(Ran::Handover),
        Halt::Outworked => Ok(Ran::Outworked),
    }
}

/// What a run whose read returned no generators made: the text's error,
/// or the outcome of a run the read's stop stopped, before any round.
fn unread_ran(unread: Unread) -> Result<Ran, Error> {
    match unread {
        Unread::Error(error) => Err(error),
        Unread::Stop(why) => halted(Halt::Stop(why), Stats::default()),
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
        sink_error: None,
    }
}
