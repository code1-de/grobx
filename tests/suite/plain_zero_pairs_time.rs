//! plain-zero-pairs-time, the item's test code: calls whose plain run's B
//! criterion drops the pairs a coprime pair settles, against the calls
//! with signatures and the checker
//! (`iterations/0164-plain-zero-pairs-time/plan.md`, _Test code_, 2 and
//! 3).

use grobx_checker::Mode;

use crate::common::systems::{boolean, cubics};
use crate::common::{EDGE_GENERATORS, P, agree, assert_accepted, run, with};
use crate::large::over;

/// The plain run's basis of `text` is the signature run's, and the plain
/// run's call with cofactors returns it too, which the checker certifies
/// in its exact mode.
fn settled(name: &str, text: &str, seed: u64) {
    let plain = with(seed, false, false);
    let basis = agree(name, text, &plain);
    let outcome = run(text, &with(seed, false, true));
    assert_accepted(name, text, &outcome, Mode::Exact);
    assert_eq!(
        outcome.basis.as_deref(),
        Some(basis.as_str()),
        "{name}: with cofactors"
    );
}

/// Test 2 (plain-zero-pairs-time.1 to .3): the edge cases of
/// `OBJECTIVES.md` on x, y, z over GF(2), GF(65521), and ℚ.
#[test]
fn edge_cases_with_the_settled_pairs() {
    for (case, generators) in EDGE_GENERATORS {
        for p in [2, P, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            settled(&format!("{case}, p = {p}"), &text, 0x5e77);
        }
    }
}

/// Test 3 (plain-zero-pairs-time.1 to .3): systems whose plain run drops
/// pairs a coprime pair settles, boolean systems of 6, 8, and 10 variables
/// and 2n generators and cubic ones of 6 and 7 variables and n generators,
/// each beside its field equations, over GF(2), GF(65521), and ℚ, seeds 0
/// to 2.
#[test]
fn systems_with_the_settled_pairs() {
    for seed in 0..3u64 {
        let mut texts: Vec<(String, String)> = Vec::new();
        for n in [6, 8, 10usize] {
            let m = 2 * n;
            texts.push((
                format!("boolean({n}, {m})"),
                boolean(n, m, 0x5e70 + 16 * seed + n as u64),
            ));
        }
        for n in [6, 7usize] {
            texts.push((
                format!("cubics({n}, {n})"),
                cubics(n, n, 0x5e71 + 16 * seed + n as u64),
            ));
        }
        for (name, text) in &texts {
            for p in [2, P, 0] {
                settled(
                    &format!("{name} seed {seed}, p = {p}"),
                    &over(text, p),
                    seed,
                );
            }
        }
    }
}
