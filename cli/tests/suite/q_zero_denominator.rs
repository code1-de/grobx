//! q-zero-denominator, the item's test code through the CLI: a denominator
//! 0 over ℚ on standard input exits 2 with the error's line on standard
//! error, rather than run on, with and without a deadline (plan.md, _Test
//! code_).

use crate::common::grobx;

/// `finish` kills a run past 120 s: the code before the item ended none
/// without a bound.
#[test]
fn a_denominator_0_exits_2() {
    for args in [
        &["--seed", "1"][..],
        &["--seed", "1", "--deadline", "60000"][..],
    ] {
        let ran = grobx(args, "x\n0\nx+1/0\n");
        assert_eq!(ran.code(), 2, "{args:?}: {ran:?}");
        assert_eq!(
            ran.stderr, "grobx: standard input: line 3: a denominator that is 0, or 0 mod p\n",
            "{args:?}"
        );
        assert!(ran.stdout.is_empty(), "{args:?}: {ran:?}");
    }
}
