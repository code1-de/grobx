//! call-stop-latency, sub-item 5's test code: the reading of the instance, its names, its mapping,
//! the substitution, the echelon, the trace, the images, and the texts each stop a call from
//! inside, and a stop inside the reading returns the call stopped before any run
//! (`iterations/0138-call-stop-latency/plan.md`, _Test code_, `call_stop_read_tests.rs`).

use std::fmt::Write;

use crate::budget::{CHECKED, CHECKS_FORCED, STOP_FORCED, Site};
use crate::call_stop_sites_tests::{Case, forced, stops_at};
use crate::systems::{placed, random_system, term};
use crate::trace::NoTrace;
use crate::{Stop, learn};

/// An instance over GF(65521) on two variables whose ten generators hold 500 terms each, of 25
/// monomials of degree at most 8 in all, so that its text is long and its basis quick.
fn long_instance() -> String {
    let names = ["a", "b"];
    let mut text = String::from("a,b\n65521\n");
    for g in 0..10u32 {
        let mut poly = String::new();
        for k in 0..500u32 {
            let e = [(k + g) % 5, (k / 5 + 2 * g) % 5];
            if k > 0 {
                poly.push('+');
            }
            poly.push_str(&term(u64::from(1 + (g * 31 + k) % 65520), &names, &e));
        }
        let comma = if g < 9 { "," } else { "" };
        writeln!(text, "{poly}{comma}").unwrap();
    }
    text
}

/// The reading's, the start's, the trace's, and the texts' sites, each with a call that reaches
/// it: a long instance; a system placed among 2,000 variable names; a system with a linear
/// generator; a call over ℚ, whose lift takes the images; a call with cofactors over many
/// variables, whose texts write them.
pub(crate) fn cases() -> Vec<(Site, Case)> {
    let three = "x,y,z\n65521\nx*y-z,\ny^2-x*z+1,\nz^3-y\n";
    let many = placed(three, &[0, 700, 1999], 2_000);
    let linear = "x,y,z,w\n65521\nx+2*y-z,\ny*z-w^2+3,\nz^2-x*w\n";
    vec![
        (Site::Parse, Case::new(long_instance(), "grevlex", false)),
        (Site::Map, Case::new(long_instance(), "grevlex", false)),
        (Site::Names, Case::new(many.clone(), "grevlex", false)),
        (Site::Substitute, Case::new(linear, "grevlex", false)),
        (Site::Echelon, Case::new(long_instance(), "grevlex", false)),
        (
            Site::Images,
            Case::new(random_system(0, 9), "grevlex", false),
        ),
        (Site::Text, Case::new(many, "grevlex", true)),
    ]
}

/// Test 1: every site of the reading, the start, and the texts stops a call from inside it, the
/// call's result unchanged with every spend checking (call-stop-latency.24 to .29).
#[test]
fn every_site_of_the_reading_stops() {
    let mut failures = Vec::new();
    for (site, case) in cases() {
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 2, call-stop-latency.24: a stop forced at the parse's first check returns the call
/// stopped by the deadline, with no basis and no round run.
#[test]
fn a_stop_inside_the_reading() {
    let case = Case::new(long_instance(), "grevlex", false);
    let (got, sites) = forced(&case, Some(Site::Parse));
    let got = got.expect("the call stopped in its reading");
    assert_eq!(got.stopped, Some(Stop::Deadline));
    assert!(got.basis.is_none(), "a basis after a stop in the reading");
    assert_eq!(got.stats.rounds, 0, "a round run before the reading's stop");
    assert_eq!(sites.last(), Some(&Site::Parse));
}

/// Test 3, call-stop-latency.27: a learn over GF(65521) checks at the trace's site and stops
/// there.
#[test]
fn a_learn_stops_in_its_trace() {
    let case = Case::new(random_system(65521, 10), "grevlex", false);
    CHECKED.with_borrow_mut(Vec::clear);
    CHECKS_FORCED.set(true);
    STOP_FORCED.set(Some(Site::Trace));
    let got = learn(&case.text, &case.params, &mut NoTrace);
    CHECKS_FORCED.set(false);
    STOP_FORCED.set(None);
    let (got, record) = got.expect("the learn");
    assert_eq!(
        got.stopped,
        Some(Stop::Deadline),
        "a learn stopped in its trace"
    );
    assert!(record.is_none(), "a stopped learn returned a record");
    let sites = CHECKED.with_borrow_mut(std::mem::take);
    assert_eq!(sites.last(), Some(&Site::Trace));
}
