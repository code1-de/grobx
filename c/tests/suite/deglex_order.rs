//! deglex-order, the item's test code through the C interface:
//! `GROBX_DEGLEX` is 3, and a call under it hands back what the crate's call
//! in deglex returns, the checker's verdict among it
//! (`iterations/0157-deglex-order/plan.md`, _Test code_, 13).

use grobx::CheckerMode;
use grobx::MonomialOrder::Deglex;
use grobxc::*;

use crate::common::{Args, assert_matches};
use crate::systems::{dense, random_square};

/// Test 13: the driver under `GROBX_DEGLEX` against the crate, over
/// GF(65521), GF(2), and ℚ, with the exact checker and with cofactors.
#[test]
fn the_c_interface_in_deglex() {
    assert_eq!(GROBX_DEGLEX, 3);
    let cases = [
        ("deglex-hand", "x,y,z\n65521\nx*z-y^2,\ny^3-1\n".to_string()),
        ("deglex-p", dense(3, 3, 65521, 1)),
        ("deglex-p2", dense(3, 3, 2, 1)),
        ("deglex-q", random_square(0, 2)),
    ];
    for (name, text) in &cases {
        let checked = Args {
            checker: Some(CheckerMode::Exact),
            ..Args::new(Deglex, 3)
        };
        let got = assert_matches(name, text, &checked);
        assert_eq!(got.certified, GROBX_CERTIFIED, "{name}");
        let with = Args {
            cofactors: true,
            ..Args::new(Deglex, 5)
        };
        assert_matches(&format!("{name}-cofactors"), text, &with);
    }
    let got = assert_matches("deglex-hand-text", &cases[0].1, &Args::new(Deglex, 1));
    assert_eq!(got.basis.as_deref(), Some("[x*z+65520*y^2,\ny^3+65520]:\n"));
}
