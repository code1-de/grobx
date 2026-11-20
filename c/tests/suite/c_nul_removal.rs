//! c-nul-removal, the item's test code: a text of a C result that holds a NUL byte fails the call
//! with `GROBX_ERROR` and a message that names the text's field and the NUL's place; an error's
//! message that holds one is replaced by such a message.

use std::ffi::{CStr, c_char};
use std::num::NonZeroUsize;

use grobx::{CheckerMode, MonomialOrder, Outcome, Params, Threads, groebner};
use grobxc::{
    GROBX_ERROR, GROBX_STOP_NONE, GROBX_UNCHECKED, failed, grobx_result_free, handed_back,
};

use crate::systems::dense;

/// The flaw the test code puts into an outcome, whose checker certified it.
const FLAW: &str = "a flaw of the test code";

/// The fields of a C result that hold a text of the outcome, in the order `handed_back` searches
/// them.
const FIELDS: [&str; 4] = ["basis", "cofactors", "flaw", "report"];

/// The places a test puts a NUL byte at in a text of `len` bytes, `len` at least 1: its first
/// byte, its middle, and its last.
const PLACES: [fn(usize) -> usize; 3] = [|_| 0, |len| len / 2, |len| len - 1];

/// A NUL byte put into the text of a field, at a place of `PLACES`.
type Nul = (&'static str, fn(usize) -> usize);

/// The NUL bytes of a case, and the field its refusal names.
type Case = (Vec<Nul>, &'static str);

/// The message of a text of the field `field` whose first NUL byte is at `at`.
fn holds_nul(field: &str, at: usize) -> String {
    format!("the {field} text holds a NUL byte at byte {at}")
}

/// `text` with a NUL byte in place of its byte at `at`; still UTF-8, as the crate's texts are
/// ASCII.
fn nul_at(text: &str, at: usize) -> String {
    let mut bytes = text.as_bytes().to_vec();
    bytes[at] = 0;
    String::from_utf8(bytes).expect("the crate's texts are ASCII")
}

/// The outcome of a call with cofactors, the exact checker, and the call report, its flaw set
/// to `FLAW`, so that every field of `FIELDS` holds a text.
fn outcome() -> Outcome {
    let mut params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    params.cofactors = true;
    params.checker = Some(CheckerMode::Exact);
    params.report = true;
    let mut got = groebner(&dense(3, 3, 65521, 1), &params).unwrap();
    got.flaw = Some(FLAW.to_string());
    got
}

/// The text of `field` in `outcome`.
fn text_of<'a>(outcome: &'a mut Outcome, field: &str) -> &'a mut Option<String> {
    match field {
        "basis" => &mut outcome.basis,
        "cofactors" => &mut outcome.cofactors,
        "flaw" => &mut outcome.flaw,
        "report" => &mut outcome.report,
        other => panic!("no text field {other}"),
    }
}

/// The text at `ptr`, None where it is NULL.
fn read(ptr: *mut c_char) -> Option<String> {
    // SAFETY: `ptr` is NULL or a text of a result that `handed_back` or `failed` made, not freed.
    (!ptr.is_null()).then(|| unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string())
}

/// c-nul-removal.1, .2, and .4: an outcome with no NUL byte becomes the C result with each text
/// as the outcome holds it; a NUL at the first, the middle, or the last byte of the basis, the
/// cofactors, the flaw, or the report fails it with `GROBX_ERROR` and the message that names that
/// field and the place; where two texts hold one, the message names the earlier of `FIELDS`.
#[test]
fn handed_back_refuses_a_nul() {
    let mut failures = Vec::new();
    let mut plain = outcome();
    let want: Vec<Option<String>> = FIELDS
        .iter()
        .map(|f| text_of(&mut plain, f).clone())
        .collect();
    match handed_back(plain) {
        Ok(mut result) => {
            let got = [result.basis, result.cofactors, result.flaw, result.report].map(read);
            for ((field, got), want) in FIELDS.iter().zip(got).zip(&want) {
                if got.as_ref() != want.as_ref() {
                    failures.push(format!("no NUL: the {field} text {got:?}, not {want:?}"));
                }
            }
            if !result.error.is_null() {
                failures.push("no NUL: an error text".to_string());
            }
            // SAFETY: `result` is a result `handed_back` made, its texts not freed.
            unsafe { grobx_result_free(&raw mut result) };
        }
        Err(refused) => failures.push(format!("no NUL: refused with {refused:?}")),
    }
    let mut cases: Vec<Case> = Vec::new();
    for field in FIELDS {
        for place in PLACES {
            cases.push((vec![(field, place)], field));
        }
    }
    cases.push((vec![("flaw", PLACES[2]), ("report", PLACES[0])], "flaw"));
    cases.push((vec![("basis", PLACES[1]), ("report", PLACES[1])], "basis"));
    for (nuls, named) in cases {
        let mut given = outcome();
        let mut want_at = None;
        for &(field, place) in &nuls {
            let text = text_of(&mut given, field)
                .as_mut()
                .expect("every field holds a text");
            let at = place(text.len());
            *text = nul_at(text, at);
            if field == named {
                want_at = Some(at);
            }
        }
        let want = (
            GROBX_ERROR,
            holds_nul(named, want_at.expect("the named field holds a NUL")),
        );
        match handed_back(given) {
            Ok(mut result) => {
                failures.push(format!("{nuls:?}: no refusal"));
                // SAFETY: `result` is a result `handed_back` made, its texts not freed.
                unsafe { grobx_result_free(&raw mut result) };
            }
            Err(got) if got != want => failures.push(format!("{nuls:?}: {got:?}, not {want:?}")),
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// c-nul-removal.5: the result of a call that returned no outcome holds the message as its error
/// text and nothing else, an empty one among them; a message with a NUL at its first, middle, or
/// last byte becomes the message that names the error text and the place.
#[test]
fn failed_replaces_a_message_with_a_nul() {
    let mut failures = Vec::new();
    let message = "the call failed: a message of the test code";
    let mut cases = vec![
        (String::new(), String::new()),
        (message.to_string(), message.to_string()),
    ];
    for place in PLACES {
        let at = place(message.len());
        cases.push((nul_at(message, at), holds_nul("error", at)));
    }
    for (given, want) in cases {
        let mut result = failed(given.clone());
        let got = read(result.error);
        if got.as_deref() != Some(want.as_str()) {
            failures.push(format!("{given:?}: the error text {got:?}, not {want:?}"));
        }
        let texts = [result.basis, result.cofactors, result.flaw, result.report];
        if texts.iter().any(|p| !p.is_null()) {
            failures.push(format!("{given:?}: a text beside the error"));
        }
        let (stopped, certified) = (result.stopped, result.certified);
        if stopped != GROBX_STOP_NONE || certified != GROBX_UNCHECKED {
            failures.push(format!(
                "{given:?}: stopped {stopped}, certified {certified}"
            ));
        }
        // SAFETY: `result` is a result `failed` made, its texts not freed.
        unsafe { grobx_result_free(&raw mut result) };
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
