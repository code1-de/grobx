//! q-zero-denominator, the item's test code through the C interface: a
//! denominator 0 over ℚ is `GROBX_NO_INVERSE` with a message, as the crate
//! maps it (plan.md, _Test code_).

use grobx::MonomialOrder;
use grobxc::GROBX_NO_INVERSE;

use crate::common::{Args, assert_matches};

/// With a deadline of 60 s, which the code before the item reached, so
/// that it fails rather than hangs.
#[test]
fn a_denominator_0_is_no_inverse() {
    for (name, text) in [
        ("first", "x\n0\nx+1/0\n"),
        ("later", "x,y\n0\nx+1/2,\ny-1/0\n"),
    ] {
        let mut args = Args::new(MonomialOrder::Grevlex, 1);
        args.deadline = Some(60_000);
        let got = assert_matches(&format!("zero-denominator-{name}"), text, &args);
        assert_eq!(
            got.status,
            GROBX_NO_INVERSE,
            "{name}: {error:?}",
            error = got.error
        );
        assert!(got.error.is_some(), "{name}: a message");
        assert_eq!(
            (&got.basis, &got.cofactors, &got.flaw),
            (&None, &None, &None),
            "{name}"
        );
    }
}
