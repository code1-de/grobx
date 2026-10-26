//! Sub-item 3: the trace hooks (f4-gfp.39) with the trace off and with a
//! recording trace of the test's own.

use grobx::id::RowId;
use grobx::trace::{NoTrace, RowSource, Trace};

use crate::common::{P, basis, dense, params, random_system};

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Matrix(Vec<RowSource>),
    Counted(Vec<RowId>),
}

#[derive(Default)]
struct Recorder(Vec<Event>);

impl Trace for Recorder {
    fn matrix(&mut self, rows: &[RowSource]) {
        self.0.push(Event::Matrix(rows.to_vec()));
    }

    fn counted(&mut self, rows: &[RowId]) {
        self.0.push(Event::Counted(rows.to_vec()));
    }
}

fn record(text: &str) -> (String, Vec<Event>) {
    let mut recorder = Recorder::default();
    let outcome = grobx::groebner_with(text, &params(1), &mut recorder).expect("the call runs");
    (outcome.basis.expect("a basis"), recorder.0)
}

/// matrix and counted alternate, the counted rows lie in their matrix and
/// increase.
fn assert_well_formed(name: &str, events: &[Event]) {
    assert!(
        events.len().is_multiple_of(2),
        "{name}: {len} events",
        len = events.len()
    );
    for pair in events.chunks(2) {
        let (rows, counted) = match (&pair[0], &pair[1]) {
            (Event::Matrix(rows), Event::Counted(counted)) => (rows, counted),
            (Event::Matrix(_) | Event::Counted(_), Event::Matrix(_) | Event::Counted(_)) => {
                panic!("{name}: matrix and counted do not alternate: {pair:?}")
            }
        };
        assert!(
            counted.windows(2).all(|w| w[0] < w[1]),
            "{name}: {counted:?}"
        );
        assert!(
            counted.iter().all(|&r| r.index() < rows.len()),
            "{name}: {counted:?} of {len}",
            len = rows.len()
        );
    }
}

#[test]
fn no_trace_is_zero_sized() {
    assert_eq!(std::mem::size_of::<NoTrace>(), 0);
}

#[test]
fn a_recording_trace_hears_the_matrices() {
    let text = "x,y\n65521\nx^2+y,\nx*y+1\n";
    let (got, events) = record(text);
    assert_eq!(got, basis(text));
    assert_well_formed("tiny", &events);
    assert!(
        events.len() >= 2,
        "the tiny instance needs a matrix: {events:?}"
    );
    let counted: usize = events
        .iter()
        .map(|e| match e {
            Event::Counted(c) => c.len(),
            Event::Matrix(_) => 0,
        })
        .sum();
    assert!(
        counted >= 1,
        "the tiny instance makes a new element: {events:?}"
    );
}

#[test]
fn the_trace_changes_no_basis() {
    for text in [
        dense(3, 3, P, 1),
        dense(4, 3, P, 2),
        random_system(7, 3),
        random_system(65521, 4),
        "x,y\n65521\n0\n".to_string(),
        "x,y\n65521\nx,\ny\n".to_string(),
    ] {
        let (got, events) = record(&text);
        assert_eq!(got, basis(&text), "{text}");
        assert_well_formed(&text, &events);
    }
}
