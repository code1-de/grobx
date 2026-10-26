//! instance-grammar-agreement, the checker's test code: the checker reads
//! the instance text as a call does. Whitespace is ASCII space, tab, line
//! feed, form feed, and carriage return (instance-grammar-agreement.2); an
//! instance term past degree 2^31 − 1 is refused once it is read, and a
//! basis text keeps the bound of `u32` (instance-grammar-agreement.5, .6;
//! plan.md, _Test code_, 2).

use grobx_checker::{Error, Mode, Order, Part, check, generators};

/// The instance x² + 3/2·y, x·y − 1 over GF(7), `{b}` standing where
/// blanks may stand in its first two lines and `{w}` where whitespace may
/// stand among its generators.
const TEMPLATE: &str =
    "{b}x{b},{b}y{b}\n{b}7{b}\n{w}x{w}^{w}2{w}+{w}3{w}/{w}2{w}*{w}y{w},{w}x{w}*{w}y{w}-{w}1{w}";

/// Its generators as the engine writes them: monic, 3/2 = 5 mod 7.
const GENERATORS: &str = "[x^2+5*y,\nx*y+6]:\n";

fn filled(b: &str, w: &str) -> String {
    TEMPLATE.replace("{b}", b).replace("{w}", w)
}

#[test]
fn every_whitespace_between_every_two_tokens() {
    assert_eq!(generators(&filled("", "")).as_deref(), Ok(GENERATORS));
    let blanks = [" ", "\t", "\r", "\x0c", " \t\r\x0c"];
    let spaces = [" ", "\t", "\r", "\x0c", "\n", " \r\n\t\x0c"];
    for b in blanks {
        for w in spaces {
            let text = filled(b, w);
            assert_eq!(generators(&text).as_deref(), Ok(GENERATORS), "{text:?}");
        }
    }
}

#[test]
fn other_bytes_are_no_whitespace() {
    for bad in ["\x0b", "\0", "\u{a0}", "\u{2003}"] {
        let text = format!("x,y\n7\nx{bad}^2+y,\nx*y-1\n");
        assert!(
            matches!(
                generators(&text),
                Err(Error::Syntax {
                    part: Part::Instance,
                    ..
                })
            ),
            "{text:?}: {read:?}",
            read = generators(&text)
        );
        let text = format!("x,{bad}y\n7\nx^2+y,\nx*y-1\n");
        assert!(
            matches!(
                generators(&text),
                Err(Error::Syntax {
                    part: Part::Instance,
                    ..
                })
            ),
            "{text:?}: {read:?}",
            read = generators(&text)
        );
    }
}

#[test]
fn an_instance_term_past_2_31_is_refused() {
    assert_eq!(
        generators("x\n7\nx^2147483647\n").as_deref(),
        Ok("[x^2147483647]:\n")
    );
    assert_eq!(
        generators("x,y\n7\nx^2147483646*y\n").as_deref(),
        Ok("[x^2147483646*y]:\n")
    );
    for text in [
        "x\n7\nx^2147483648\n",
        "x\n7\nx^2147483647*x\n",
        "x,y\n7\ny^1073741824*x^1073741824\n",
        "x\n7\nx^4294967295\n",
        "x\n7\nx^4294967296\n",
        "x\n7\nx^99999999999999999999\n",
        "x\n7\n0*x^2147483648\n",
        "x\n7\nx^2147483648*0\n",
        "x\n7\nx^2147483647\n*x\n",
        "x\n0\nx^2147483648\n",
    ] {
        let got = generators(text);
        let got = if text.contains("\n0\n") {
            // Over Q the hook refuses the characteristic; the check reads.
            check(text, "[]:\n", "[]:\n", &Order::Grevlex, Mode::Exact).map(|_| String::new())
        } else {
            got
        };
        assert!(
            matches!(
                got,
                Err(Error::Overflow {
                    part: Part::Instance,
                    ..
                })
            ),
            "{text:?}: {got:?}"
        );
    }
}

/// The degree is tested once the term is read (instance-grammar-agreement.6),
/// so a later factor's error is the refusal, as the engine's.
#[test]
fn a_later_error_in_the_term_comes_first() {
    assert_eq!(
        generators("x,y\n7\nx^99999999999*z\n"),
        Err(Error::UnknownVariable {
            part: Part::Instance,
            line: 3,
            name: "z".to_string(),
        })
    );
    for text in [
        "x,y\n7\nx^99999999999*\n",
        "x,y\n7\nx^99999999999*y^\n",
        "x,y\n7\nx^4294967296*1/\n",
    ] {
        let got = generators(text);
        assert!(
            matches!(
                got,
                Err(Error::Syntax {
                    part: Part::Instance,
                    ..
                })
            ),
            "{text:?}: {got:?}"
        );
    }
    assert_eq!(
        generators("x,y\n7\nx^99999999999*1/7\n"),
        Err(Error::NoInverse {
            part: Part::Instance,
            line: 3,
        })
    );
}

/// A basis text keeps the bound of `u32` (instance-grammar-agreement.5):
/// x^2147483648 is read and the check rules. x does not reduce modulo it,
/// so the cofactor text is not reached.
#[test]
fn a_basis_text_past_2_31_is_read() {
    let got = check(
        "x\n7\nx\n",
        "[x^2147483648]:\n",
        "[[x^2147483647]]:\n",
        &Order::Grevlex,
        Mode::Exact,
    );
    assert!(got.is_ok(), "{got:?}");
}

/// Over Q the hook refuses the characteristic, as the engine's
/// `text::generators` does.
#[test]
fn the_hook_refuses_q() {
    assert_eq!(
        generators("x,y\n0\nx^2+y\n"),
        Err(Error::Characteristic { line: 2 })
    );
}
