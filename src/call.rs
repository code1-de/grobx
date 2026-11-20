//! A call: the crate's entry points `groebner` and `replay`. A call opens
//! its budget and its call report, reads the instance's header,
//! and runs over GF(p) in the field's copy of the run (`entry::call_in`),
//! or the lift over Q (`lift::lift`).

use std::time::Instant;

use crate::blocks;
use crate::budget::{self, Budget};
use crate::entry::{call_in, report_call, report_end, report_read, stopped};
use crate::lift::{self, Primes, Report};
use crate::monomial::{WIDTHS, Width};
use crate::report::{self as call_report, field_value};
use crate::text::{Header, Prime, Unread};
use crate::trace::{Learner, NoLearner, Record, Trace};
use crate::{Error, Outcome, Params, Stats, Threads};

/// The outcome of a run in the widths `WIDTHS`; `learner` hears what the
/// run records.
pub(crate) fn groebner<T: Trace, L: Learner>(
    text: &str,
    params: &Params,
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    groebner_in(text, params, WIDTHS.0, WIDTHS.1, trace, learner)
}

// monomial-formats.14 design-checked -- the widths a call may take are a crate-internal parameter of the entry point, not a field of Params (not measured)
/// As `groebner`, in the widths from `first` to `last` over GF(p), each
/// raised to the field's narrowest width where it lies below: the call
/// starts in `first`, and over in the next wider width where a degree
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

/// The outcome of a replay of `record`, in the widths `WIDTHS`.
pub(crate) fn replay<T: Trace>(
    text: &str,
    params: &Params,
    record: &Record,
    trace: &mut T,
) -> Result<Outcome, Error> {
    replay_in(text, params, record, WIDTHS.0, WIDTHS.1, trace)
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
/// (`lift::lift`), whose runs take the widths `WIDTHS`;
/// `Error::Characteristic` where there is a record or `learner` listens.
/// An error ends the call report with its `error` line (`report_error`).
fn call<T: Trace, L: Learner>(
    text: &str,
    params: &Params,
    record: Option<&Record>,
    (first, last): (Width, Width),
    trace: &mut T,
    learner: &mut L,
) -> Result<Outcome, Error> {
    debug_assert!(first <= last);
    let start = budget::start(params);
    let mut held: Option<Budget> = None;
    let called = (|| {
        let _watch = match (&params.report_sink, start) {
            (Some(sink), Some(start)) if params.report => Some(blocks::watch(start, sink.clone())),
            (Some(_) | None, Some(_) | None) => None,
        };
        let budget = held.insert(Budget::new(start, params)?);
        call_budgeted(text, params, record, (first, last), budget, trace, learner)
    })();
    if let Err(e) = &called {
        report_error(held.as_mut(), params, start, e);
    }
    called
}

// error-report.1 design-checked -- a call that returns an error ends its report with an error line, which the sink keeps, not the report dropped with the error (not measured)
/// The call report's `error` line of `e`: the variant's name and its
/// message as one value (`field_value`). The budget's report writes it,
/// and where the error came before there was one, a report of its own on
/// the call's sink. With no `start` there is no report: the system has no
/// clock, and `e` is `Error::Clock`.
pub(crate) fn report_error(
    budget: Option<&mut Budget>,
    params: &Params,
    start: Option<Instant>,
    e: &Error,
) {
    let mut own = None;
    let report = match (budget.and_then(Budget::report), start) {
        (Some(report), Some(_) | None) => report,
        (None, Some(start)) if params.report => {
            own.insert(call_report::Report::new(start, params.report_sink.clone()))
        }
        (None, Some(_) | None) => return,
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

/// `Error::NotBuilt` where the call asks for what the build leaves out:
/// the cofactors without the feature `cofactors`, the checker without
/// `checker`. Past this check a build without `cofactors` tracks none and
/// one without `checker` certifies nothing. The unit tests skip a case
/// this refuses.
pub(crate) fn built(params: &Params) -> Result<(), Error> {
    if params.cofactors && !cfg!(feature = "cofactors") {
        return Err(Error::NotBuilt {
            feature: "cofactors",
        });
    }
    if params.checker.is_some() && !cfg!(feature = "checker") {
        return Err(Error::NotBuilt { feature: "checker" });
    }
    Ok(())
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
    built(params)?;
    // f4-gfp.30 design-checked -- the run takes one thread, whatever threads allows, not several that share the reduction of a matrix's rows (not measured)
    let _: Threads = params.threads;

    let header = match Header::read(text, budget) {
        Ok(header) => header,
        Err(Unread::Error(error)) => return Err(error),
        Err(Unread::Stop(why)) => return Ok(report_end(budget, stopped(why, Stats::default()))),
    };
    let outcome = if header.rational() {
        // q-lift.30 design-checked -- learn and replay refuse characteristic 0, and the lift's records stay inside the call, not learn over Q returning the record of its learn prime (not measured)
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
            // q-learn-time.10 design-checked -- every learn of the lift starts as a pair learn, the first and each that learns afresh, in every order, with or without cofactors, not the pair learn in grevlex alone nor the first learn run alone a pair (not measured)
            lift::Learns::Pairs,
            &mut report,
        )?
    } else {
        let widths = (first, last);
        // gfp-large-primes.3 practice-checked: monomorphization -- the entry point matches the field once and runs a copy of the call compiled for Gf2, Gf31, or Gf64, not one field type that holds which of them it is and matches at each operation
        match header.field()? {
            Prime::Gf2(field) => {
                report_read(budget, &header, "gf2");
                call_in(
                    field, text, params, record, widths, named, &header, budget, trace, learner,
                )?
            }
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
