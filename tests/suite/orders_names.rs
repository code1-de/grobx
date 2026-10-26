//! order-family, the item's test code: the names `order` takes and those it
//! refuses (plan.md, _Interface_ and _Edge cases_).

use grobx::{Error, MonomialOrder};

use crate::common::{params, params_in, run};

/// Three variables, which every name below fits or does not fit by its
/// sizes alone.
const THREE: &str = "x,y,z\n65521\nx*y-z,\ny^2-x*z,\nz^3-x\n";

/// Whether the name is refused: by its parse, or by the call on `text`.
fn refused(text: &str, name: &str) -> bool {
    let called = name.parse::<MonomialOrder>().and_then(|order| {
        let mut p = params(1);
        p.order = order;
        grobx::groebner(text, &p)
    });
    match called {
        Err(Error::Order { .. }) => true,
        Ok(_) | Err(_) => false,
    }
}

#[test]
fn the_names_of_the_family() {
    for name in ["grevlex", "lex", "block:1", "block:2", "block:1,1"] {
        assert!(run(THREE, &params_in(name, 1)).basis.is_some(), "{name}");
    }
}

#[test]
fn names_outside_the_family_are_refused() {
    for name in [
        "",
        "Lex",
        "LEX",
        "deglex",
        "degrevlex",
        "grevlex ",
        " lex",
        "block",
        "block:",
        "block:0",
        "block:1,",
        "block:,1",
        "block:1,,1",
        "block:-1",
        "block:+1",
        "block: 1",
        "block:1 ",
        "block:x",
        "block:1,0",
        "block:1;1",
        "block:99999999999999999999999",
    ] {
        assert!(refused(THREE, name), "{name:?}");
    }
}

#[test]
fn the_sizes_leave_a_last_block() {
    for name in [
        "block:3",
        "block:4",
        "block:2,1",
        "block:1,2",
        "block:1,1,1",
    ] {
        assert!(refused(THREE, name), "{name:?} on three variables");
    }
    let one = "x\n65521\nx^3-1,\nx^2-1\n";
    for name in ["block:1", "block:2"] {
        assert!(refused(one, name), "{name:?} on one variable");
    }
}

/// A name that does not parse is refused before the text is read. A block
/// order's sizes are fitted once the header is read, so a header that does
/// not parse is reported first.
#[test]
fn when_a_name_is_refused() {
    let broken = "x,x\n65521\nx\n";
    assert!(refused(broken, "deglex"));
    assert!(matches!(
        grobx::groebner(broken, &params_in("block:5", 1)),
        Err(Error::RepeatedVariable { .. })
    ));
}
