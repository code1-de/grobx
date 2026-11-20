//! call-stop-latency, sub-item 4's test code: the signature run's entry of an element, its
//! searches, its queue, its Koszul loop, and the carry each stop a call from inside, and the queue
//! pops each pending signature once, its repeats dropped
//! (`iterations/0138-call-stop-latency/plan.md`, _Test code_, `call_stop_sig_tests.rs`).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::f4::{CARRY_LEAST, LARGE_LEAST};
use crate::id::MonoId;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::queue::Queue;
use crate::shared_tests::{below, rng};
use crate::sig::Sig;
use crate::systems::random_system;
use crate::table::Table;

/// The signature run's sites, each with a call that reaches it: a random system over GF(65521)
/// with cofactors, which runs the signature run to its end, as a call with cofactors does not hand
/// over; and for the carry the system without cofactors, where `LARGE_LEAST` and `CARRY_LEAST`
/// hold 1, so that its first round hands over and carries the run's work.
pub(crate) fn cases() -> Vec<(Site, Case)> {
    let sig = || Case::new(random_system(65521, 5), "grevlex", true);
    vec![
        (Site::SigEnter, sig()),
        (Site::Antichain, sig()),
        (Site::Reducer, sig()),
        (Site::Queue, sig()),
        (Site::Koszul, sig()),
        (
            Site::Carry,
            Case::new(random_system(65521, 5), "grevlex", false),
        ),
    ]
}

/// Test 1: every site of the signature run stops a call from inside it, the call's result
/// unchanged with every spend checking (call-stop-latency.19 to .23).
#[test]
fn every_site_of_the_signature_run_stops() {
    let mut failures = Vec::new();
    LARGE_LEAST.set(Some(1));
    CARRY_LEAST.set(Some(1));
    for (site, case) in cases() {
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    LARGE_LEAST.set(None);
    CARRY_LEAST.set(None);
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 2, call-stop-latency.22: the queue without its set pops every pending signature once, in
/// increasing order of signature by degree, however often it was pushed.
#[test]
fn the_queue_drops_its_repeats() {
    let mut t = Table::new(Dense16::new(4), &mut rng(47));
    let mut u = Budget::unbounded();
    let mut r = rng(53);
    let monos: Vec<MonoId> = (0..300)
        .map(|_| {
            let e: Vec<u32> = (0..4).map(|_| below(&mut r, 4) as u32).collect();
            t.intern(&e, &mut u).unwrap()
        })
        .collect();
    let mut q = Queue::new();
    let mut pushed = std::collections::BTreeSet::new();
    for _ in 0..2_000 {
        let s = Sig {
            monomial: monos[below(&mut r, monos.len() as u64) as usize],
            index: below(&mut r, 3) as u32,
        };
        q.push(&t, s);
        pushed.insert((s.index, s.monomial));
    }
    let mut popped = Vec::new();
    while !q.is_empty() {
        let round = q.pop_least(&t, &Grevlex, &mut u).unwrap();
        for w in round.windows(2) {
            assert!(
                w[0].cmp(w[1], &t, &Grevlex).is_lt(),
                "a round's signatures in increasing order, each once"
            );
        }
        let degrees: std::collections::HashSet<u32> =
            round.iter().map(|s| t.degree(s.monomial)).collect();
        assert_eq!(degrees.len(), 1, "a round of one degree");
        popped.extend(round.iter().map(|s| (s.index, s.monomial)));
    }
    let once: std::collections::BTreeSet<_> = popped.iter().copied().collect();
    assert_eq!(once.len(), popped.len(), "a signature popped twice");
    assert_eq!(once, pushed, "the signatures popped are those pushed");
}
