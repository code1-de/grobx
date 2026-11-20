//! deglex-order, the item's test code through the CLI: `--order deglex`
//! writes the basis the crate's call in deglex returns, and the checker's
//! verdict on it, certified; `--help` names the order
//! (`iterations/0157-deglex-order/plan.md`, _Test code_, 12).

use std::num::NonZeroUsize;

use grobx::{CheckerMode, MonomialOrder, Params, Threads};

use crate::common::{crate_call, grobx, verdict};
use crate::systems::{dense, random_square, random_system};

/// Test 12: the CLI's basis and verdict in deglex are the crate's, over
/// GF(65521), GF(2), and ℚ, where deglex and grevlex part.
#[test]
fn the_cli_in_deglex() {
    let mut cases = vec![
        "x,y,z\n65521\nx*z-y^2,\ny^3-1\n".to_string(),
        dense(3, 3, 2, 1),
        random_square(0, 2),
    ];
    for seed in 0..3 {
        cases.push(random_system(65521, seed));
    }
    let mut params = Params::new(MonomialOrder::Deglex, Threads::Count(NonZeroUsize::MIN), 1);
    params.checker = Some(CheckerMode::Exact);
    for text in &cases {
        let want = crate_call(text, &params);
        let ran = grobx(
            &["--seed", "1", "--order", "deglex", "--checker", "exact"],
            text,
        );
        assert_eq!(ran.code(), 0, "{text}: {ran:?}");
        assert_eq!(Some(ran.stdout.as_str()), want.basis.as_deref(), "{text}");
        assert_eq!(ran.stderr, verdict(&want), "{text}");
        assert_eq!(ran.stderr, "grobx: checker: certified\n", "{text}");
    }
    let grevlex = grobx(&["--seed", "1", "--order", "grevlex"], &cases[0]);
    let deglex = grobx(&["--seed", "1", "--order", "deglex"], &cases[0]);
    assert_eq!(deglex.stdout, "[x*z+65520*y^2,\ny^3+65520]:\n");
    assert_ne!(grevlex.stdout, deglex.stdout);
    let help = grobx(&["--help"], "");
    assert!(
        help.stdout
            .contains("--order NAME      grevlex, deglex, lex, or block:k1,...,km"),
        "{help:?}"
    );
}
