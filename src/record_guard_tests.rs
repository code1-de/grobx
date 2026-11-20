//! The replay's guards against a record that is not a learn run's: each tampered record, replayed
//! on the instance and prime it was learned on, is a misfit, never a panic or a basis.

use crate::shared_tests::params;
use crate::systems::dense;
use crate::trace::{NoTrace, Record};
use crate::{Error, learn, replay};

/// A tampering of a record, by name.
type Tamper = (&'static str, fn(&mut Record));

/// One tampering per guard of the replay.
const TAMPERS: &[Tamper] = &[
    ("a generator entered that is none", |r| {
        r.entered[0] = u32::MAX;
    }),
    ("a generator entered twice", |r| {
        let first = r.entered[0];
        r.entered.push(first);
    }),
    ("a step's entry past its leads", |r| {
        r.steps[0].entered.push(u32::MAX);
    }),
    ("a step's row entered twice", |r| {
        let first = r.steps[0].entered[0];
        r.steps[0].entered.push(first);
    }),
    ("a step's lead moved", |r| r.steps[0].leads[0] += 1),
    ("an input lead past every format", |r| {
        let lead = r
            .inputs
            .iter_mut()
            .flatten()
            .next()
            .expect("a nonzero input");
        lead[0] = u32::MAX;
    }),
];

#[test]
fn a_tampered_record_misfits() {
    let text = dense(3, 3, 65521, 1);
    let mut failures = Vec::new();
    for signatures in [false, true] {
        let mut p = params(1);
        p.signatures = signatures;
        let (_, record) = learn(&text, &p, &mut NoTrace).expect("the learn call");
        let record = record.expect("a record");
        assert!(
            !record.steps.is_empty() && !record.steps[0].entered.is_empty(),
            "signatures {signatures}: a record with a step"
        );
        for (name, tamper) in TAMPERS {
            let mut tampered = record.clone();
            tamper(&mut tampered);
            match replay(&text, &p, &tampered, &mut NoTrace) {
                Err(Error::Misfit { .. }) => {}
                got @ (Ok(_) | Err(_)) => {
                    failures.push(format!("{name}, signatures {signatures}: {got:?}"));
                }
            }
        }
        // A misfit names its step of the record, counted from 1, a step before
        // it that builds no matrix counted too.
        let mut unbuilt = record.steps[0].clone();
        unbuilt.elements.clear();
        unbuilt.multipliers.clear();
        unbuilt.reductees.clear();
        unbuilt.leads.clear();
        unbuilt.entered.clear();
        for (before, want) in [(false, 1), (true, 2)] {
            let mut tampered = record.clone();
            if before {
                tampered.steps.insert(0, unbuilt.clone());
            }
            tampered.steps[want - 1].entered.push(u32::MAX);
            match replay(&text, &p, &tampered, &mut NoTrace) {
                Err(Error::Misfit { step }) if step as usize == want => {}
                got @ (Ok(_) | Err(_)) => failures.push(format!(
                    "the misfit's step, an unbuilt step before {before}, signatures {signatures}: {got:?}"
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
