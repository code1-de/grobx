//! Sub-item 3: the trace hooks (f4-gfp.39) with the trace off and with a
//! recording trace of the test's own.

use grobx::trace::NoTrace;

use crate::common::{Event, P, Recorder, assert_well_formed, basis, dense, params, random_system};

fn record(text: &str) -> (String, Vec<Event>) {
    let mut recorder = Recorder::default();
    let outcome = grobx::groebner_with(text, &params(1), &mut recorder).expect("the call runs");
    (outcome.basis.expect("a basis"), recorder.0)
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
