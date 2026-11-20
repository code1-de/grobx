//! grobx in a browser: the crate built for wasm32-unknown-unknown, called
//! from JavaScript through three exports.
//!
//! The caller asks `grobx_input` for a buffer of the request's length in
//! bytes, writes the request there as UTF-8, and calls `grobx_run`, which
//! returns the length of the answer `grobx_output` points to, UTF-8 too.
//! A pointer holds until the next call of an export. The memory may grow
//! while a call runs, so the caller takes a fresh view of it after each.
//! The module imports nothing.
//!
//! The request is three lines and the instance text (`MANUAL.md`,
//! _Instance text_): the order's name (`MonomialOrder::from_str`), the seed
//! in decimal digits, and `1` or `0` for whether the answer carries the
//! cofactors. The call takes one thread, no bound, no checker, and no
//! report: a browser has no clock and gives no resident set
//! (`Error::Clock`, `Error::Memory`), and the page stops a call by ending
//! the worker that runs it.
//!
//! The answer is a JSON object. `version` is the crate version. Then either
//! `basis` and `cofactors` (`MANUAL.md`, _Basis text_ and _Cofactor text_),
//! the latter null unless asked, and `rounds`, `reduced`, and `zero`
//! (`Stats`); or `error`, what refused the request or the call, the
//! variant's name first where the crate refused it.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::num::NonZeroUsize;

use grobx::{Error, MonomialOrder, Params, Threads};

thread_local! {
    /// The request, as `grobx_input` sized it and the caller wrote it.
    static INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// The answer of the last `grobx_run`.
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// A buffer of `len` zero bytes, which the caller fills with the request.
// SAFETY: the module defines no other symbol of this name.
#[unsafe(no_mangle)]
pub extern "C" fn grobx_input(len: usize) -> *mut u8 {
    INPUT.with_borrow_mut(|input| {
        input.clear();
        input.resize(len, 0);
        input.as_mut_ptr()
    })
}

/// Answers the request in the buffer, which it consumes, and returns the
/// answer's length in bytes.
// SAFETY: the module defines no other symbol of this name.
#[unsafe(no_mangle)]
pub extern "C" fn grobx_run() -> usize {
    let answer = answer(&INPUT.take()).into_bytes();
    OUTPUT.with_borrow_mut(|output| {
        *output = answer;
        output.len()
    })
}

/// The answer of the last `grobx_run`.
// SAFETY: the module defines no other symbol of this name.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn grobx_output() -> *const u8 {
    OUTPUT.with_borrow(Vec::as_ptr)
}

/// The answer to `request`, as the crate's documentation gives both.
fn answer(request: &[u8]) -> String {
    let mut out = String::from("{\"version\":");
    push_json(&mut out, grobx::VERSION);
    match read(request).and_then(|(params, text)| call(text, &params)) {
        Ok(fields) => out.push_str(&fields),
        Err(why) => {
            out.push_str(",\"error\":");
            push_json(&mut out, &why);
        }
    }
    out.push('}');
    out
}

/// The parameters and the instance text of `request`, or what is wrong
/// with it.
fn read(request: &[u8]) -> Result<(Params, &str), String> {
    let request =
        std::str::from_utf8(request).map_err(|e| format!("the request is not UTF-8: {e}"))?;
    let mut lines = request.splitn(4, '\n');
    let mut next = |what: &str| {
        lines
            .next()
            .ok_or_else(|| format!("the request ends before the {what}"))
    };
    let order: MonomialOrder = next("order")?.parse().map_err(|e| refused(&e))?;
    let seed = seed(next("seed")?)?;
    let cofactors = match next("cofactors")? {
        "1" => true,
        "0" => false,
        other => return Err(format!("cofactors {other:?}: 1 or 0")),
    };
    let text = next("instance")?;
    let mut params = Params::new(order, Threads::Count(NonZeroUsize::MIN), seed);
    params.cofactors = cofactors;
    Ok((params, text))
}

/// The seed `digits` names: decimal digits alone, as the CLI reads a
/// number, from 0 to 2^64 − 1.
fn seed(digits: &str) -> Result<u64, String> {
    // `u64`'s parse takes a sign, which a seed has not.
    let all = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
    all.then(|| digits.parse().ok())
        .flatten()
        .ok_or_else(|| format!("seed {digits:?}: decimal digits for 0 to 2^64 - 1"))
}

/// The answer's fields after `version` of the call on `text`, or why it
/// gave no basis.
fn call(text: &str, params: &Params) -> Result<String, String> {
    let outcome = grobx::groebner(text, params).map_err(|e| refused(&e))?;
    if let Some(stop) = outcome.stopped {
        return Err(format!("the call stopped: {name}", name = stop.name()));
    }
    let basis = outcome
        .basis
        .ok_or_else(|| "the call gave no basis".to_string())?;
    let mut out = String::from(",\"basis\":");
    push_json(&mut out, &basis);
    out.push_str(",\"cofactors\":");
    match &outcome.cofactors {
        Some(cofactors) => push_json(&mut out, cofactors),
        None => out.push_str("null"),
    }
    let stats = outcome.stats;
    for (name, count) in [
        ("rounds", u64::from(stats.rounds)),
        ("reduced", stats.reduced),
        ("zero", stats.zero),
    ] {
        out.push_str(",\"");
        out.push_str(name);
        out.push_str("\":");
        out.push_str(&count.to_string());
    }
    Ok(out)
}

/// The crate's error, its variant's name first, as the Python module's
/// `grobx.Error` gives it.
fn refused(e: &Error) -> String {
    format!("{name}: {e}", name = e.name())
}

/// Appends `s` to `out` as a JSON string (RFC 8259, section 7): in quotes,
/// the quote, the backslash, and the control characters below U+0020
/// escaped.
fn push_json(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            #[allow(clippy::expect_used, reason = "a String takes every write")]
            c if c < ' ' => write!(out, "\\u{code:04x}", code = u32::from(c))
                .expect("a String takes every write"),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test code, where these lints carry no weight"
)]
mod tests {
    use serde_json::Value;

    use super::*;

    /// The manual's instance over GF(7) (`MANUAL.md`, _Basis text_).
    const GF7: &str = "x,y\n7\nx^2+y,\nx*y-1\n";

    /// The answer to `request`, which is JSON whatever the request.
    fn parsed(request: &[u8]) -> Value {
        let text = answer(request);
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?} is no JSON: {e}"))
    }

    /// The request of `order`, `seed`, `cofactors`, and `text`.
    fn request(order: &str, seed: &str, cofactors: &str, text: &str) -> Vec<u8> {
        format!("{order}\n{seed}\n{cofactors}\n{text}").into_bytes()
    }

    /// The error the answer to `request` gives, or a failure where it gives
    /// none or a basis.
    fn error(request: &[u8], failures: &mut Vec<String>) -> String {
        let answer = parsed(request);
        if answer.get("basis").is_some() {
            failures.push(format!("{answer} has a basis beside its error"));
        }
        if let Some(error) = answer["error"].as_str() {
            error.to_string()
        } else {
            failures.push(format!("{answer} gives no error"));
            String::new()
        }
    }

    /// The answer carries what `grobx::groebner` returns for the same
    /// parameters, in every order and over GF(p) and Q, with the cofactors
    /// where asked, and refuses every malformed request with a message.
    #[test]
    fn answer_carries_the_call_or_why_not() {
        let mut failures = Vec::new();
        let cases = [
            ("grevlex", "7", "0", GF7),
            ("grevlex", "7", "1", GF7),
            ("deglex", "7", "1", GF7),
            ("lex", "0", "0", GF7),
            ("block:1", "18446744073709551615", "1", GF7),
            (
                "grevlex",
                "3",
                "1",
                "x,y,z\n0\nx^2+1/3*y*z-1,\n2*x*y-z,\nz^3-x\n",
            ),
            (
                "lex",
                "3",
                "0",
                "x,y,z\n65521\nx+y+z,\nx*y+y*z+z*x,\nx*y*z-1\n",
            ),
            ("grevlex", "1", "0", "x\n7\n"),
            ("grevlex", "1", "1", "x,y\n7\nx,\n1\n"),
        ];
        for (order, seed, cofactors, text) in cases {
            let answer = parsed(&request(order, seed, cofactors, text));
            let mut params = Params::new(
                order.parse().unwrap(),
                Threads::Count(NonZeroUsize::MIN),
                seed.parse().unwrap(),
            );
            params.cofactors = cofactors == "1";
            let want = grobx::groebner(text, &params).unwrap();
            let fields = [
                ("version", Value::from(grobx::VERSION)),
                ("basis", Value::from(want.basis.clone())),
                ("cofactors", Value::from(want.cofactors.clone())),
                ("rounds", Value::from(want.stats.rounds)),
                ("reduced", Value::from(want.stats.reduced)),
                ("zero", Value::from(want.stats.zero)),
            ];
            for (name, value) in fields {
                if answer[name] != value {
                    failures.push(format!(
                        "{order} {text:?}: {name} {got} for {value}",
                        got = answer[name]
                    ));
                }
            }
            if answer.get("error").is_some() {
                failures.push(format!("{order} {text:?}: {answer} has an error"));
            }
        }
        let refusals: [(&[u8], &str); 16] = [
            (b"", "Order: monomial order \"\""),
            (b"grevlex", "the request ends before the seed"),
            (b"grevlex\n1", "the request ends before the cofactors"),
            (b"grevlex\n1\n0", "the request ends before the instance"),
            (
                &request("degneglex", "1", "0", GF7),
                "Order: monomial order \"degneglex\"",
            ),
            (
                &request("bl\"o\\ck", "1", "0", GF7),
                "Order: monomial order \"bl\\\"o\\\\ck\"",
            ),
            (
                &request("block:5", "1", "0", GF7),
                "Order: monomial order \"block:5\"",
            ),
            (
                &request("grevlex\r", "1", "0", GF7),
                "Order: monomial order \"grevlex\\r\"",
            ),
            (
                &request("grevlex", "", "0", GF7),
                "seed \"\": decimal digits",
            ),
            (
                &request("grevlex", "-1", "0", GF7),
                "seed \"-1\": decimal digits",
            ),
            (
                &request("grevlex", "+1", "0", GF7),
                "seed \"+1\": decimal digits",
            ),
            (
                &request("grevlex", "18446744073709551616", "0", GF7),
                "seed \"18446744073709551616\"",
            ),
            (
                &request("grevlex", "1", "yes", GF7),
                "cofactors \"yes\": 1 or 0",
            ),
            (
                &request("grevlex", "1", "0", "x,y\n7\nx^2+\n"),
                "Syntax: line 3",
            ),
            (
                &request("grevlex", "1", "0", "x\u{200b},y\n7\nx\n"),
                "Syntax: line 1",
            ),
            (b"grevlex\n1\n0\nx\n7\n\xff", "the request is not UTF-8"),
        ];
        for (request, want) in refusals {
            let got = error(request, &mut failures);
            if !got.starts_with(want) {
                failures.push(format!("{request:?}: {got:?}, not {want:?}…"));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// A string reads back as itself from its JSON, whatever it holds.
    #[test]
    fn push_json_escapes_what_json_does_not_take() {
        let mut failures = Vec::new();
        let texts = [
            "",
            "plain",
            "\"quoted\"",
            "back\\slash",
            "line\nfeed\r\n",
            "\u{0}\u{1}\u{1f}\t\u{7f}",
            "ℚ GF(p) ü 𝔽",
            "\u{200b}zero\u{200d}width",
            "\u{202e}rtl \u{5d0}\u{5d1}",
            "\u{2028}\u{2029}",
        ];
        for text in texts {
            let mut out = String::new();
            push_json(&mut out, text);
            match serde_json::from_str::<String>(&out) {
                Ok(back) if back == text => {}
                Ok(back) => failures.push(format!("{text:?}: {out} reads back as {back:?}")),
                Err(e) => failures.push(format!("{text:?}: {out} is no JSON string: {e}")),
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// The buffer is `len` zero bytes, empty included, and replaces the
    /// last.
    #[test]
    fn grobx_input_sizes_the_request() {
        let mut failures = Vec::new();
        for len in [5, 0, 3] {
            let p = grobx_input(len);
            INPUT.with_borrow(|input| {
                if input.len() != len
                    || input.iter().any(|&b| b != 0)
                    || input.as_ptr() != p.cast_const()
                {
                    failures.push(format!(
                        "{len}: {input:?} at {at:p}, given {p:p}",
                        at = input.as_ptr()
                    ));
                }
            });
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// The run consumes the request and leaves its answer, whose length it
    /// returns, where `grobx_output` points.
    #[test]
    fn grobx_run_answers_the_request() {
        let mut failures = Vec::new();
        for request in [request("grevlex", "1", "0", GF7), b"lex".to_vec()] {
            let p = grobx_input(request.len());
            INPUT.with_borrow_mut(|input| input.copy_from_slice(&request));
            assert_eq!(p, INPUT.with_borrow_mut(Vec::as_mut_ptr));
            let len = grobx_run();
            let want = answer(&request).into_bytes();
            OUTPUT.with_borrow(|output| {
                if *output != want || output.len() != len || output.as_ptr() != grobx_output() {
                    failures.push(format!("{request:?}: {len} bytes, {output:?}"));
                }
            });
            if INPUT.with_borrow(|input| !input.is_empty()) {
                failures.push(format!("{request:?}: the request stays"));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
