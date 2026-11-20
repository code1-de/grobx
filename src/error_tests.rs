//! `Error`'s names, which the call report's `error` line and the Python module's `grobx.Error`
//! give, and its messages.

use std::collections::BTreeSet;

use crate::Error;

/// One error of every variant, each with its name and the values its message must carry.
fn every() -> Vec<(Error, &'static str, Vec<&'static str>)> {
    vec![
        (
            Error::Syntax {
                line: 7,
                expected: "a comma",
            },
            "Syntax",
            vec!["7", "a comma"],
        ),
        (
            Error::UnknownVariable {
                line: 8,
                name: "zz".to_string(),
            },
            "UnknownVariable",
            vec!["8", "zz"],
        ),
        (
            Error::RepeatedVariable {
                name: "yy".to_string(),
            },
            "RepeatedVariable",
            vec!["yy"],
        ),
        (
            Error::Characteristic {
                value: "12".to_string(),
            },
            "Characteristic",
            vec!["12"],
        ),
        (Error::Overflow { line: Some(9) }, "Overflow", vec!["9"]),
        (Error::Overflow { line: None }, "Overflow", vec![]),
        (Error::Size, "Size", vec![]),
        (Error::NoInverse { line: 10 }, "NoInverse", vec!["10"]),
        (
            Error::Order {
                name: "block:9".to_string(),
            },
            "Order",
            vec!["block:9"],
        ),
        (Error::Memory, "Memory", vec![]),
        (Error::Clock, "Clock", vec![]),
        (Error::Misfit { step: 11 }, "Misfit", vec!["11"]),
        (
            Error::Drops {
                misfits: 2,
                leads: 3,
                verification: 4,
            },
            "Drops",
            vec!["9 records", "2 after", "3 at", "4 at"],
        ),
        (
            Error::NotBuilt { feature: "checker" },
            "NotBuilt",
            vec!["checker"],
        ),
    ]
}

/// Every variant names itself.
#[test]
fn every_variant_names_itself() {
    let mut failures = Vec::new();
    for (error, name, _) in every() {
        if error.name() != name {
            failures.push(format!("{error:?}: {got}", got = error.name()));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every message carries its values, and no two errors share one.
#[test]
fn every_message_carries_its_values() {
    let mut failures = Vec::new();
    let mut said = BTreeSet::new();
    for (error, _, values) in every() {
        let message = error.to_string();
        for value in values {
            if !message.contains(value) {
                failures.push(format!("{error:?} lacks {value:?}: {message}"));
            }
        }
        if !said.insert(message.clone()) {
            failures.push(format!("{error:?} repeats {message}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
