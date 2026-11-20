//! bivium-gap, the item's test code: calls whose signature run tests the
//! inputs' Koszul signatures by their leading monomials, on the systems'
//! own variables and placed on 150, against the calls without signatures
//! and the checker (`iterations/0163-bivium-gap/plan.md`, _Test code_, 2
//! and 3).

use grobx_checker::Mode;

use crate::common::systems::{boolean, placed, random_sparse};
use crate::common::{EDGE_GENERATORS, P, assert_accepted, params, run, with_cofactors};

/// The variables a system is placed on: past 128, where the signatures
/// hold the inputs' leading monomials by their first variable.
const WIDE: usize = 150;

/// `text` with its variables placed on `WIDE` variables, the k-th at
/// 7k + 3.
fn wide(text: &str) -> String {
    let names = text.split_once('\n').expect("a line of variables").0;
    let places: Vec<usize> = (0..names.split(',').count()).map(|k| 7 * k + 3).collect();
    placed(text, &places, WIDE)
}

/// `text` over the field of characteristic `p`.
fn over(text: &str, p: u64) -> String {
    let (names, rest) = text.split_once('\n').expect("a line of variables");
    let generators = rest.split_once('\n').expect("a characteristic").1;
    format!("{names}\n{p}\n{generators}")
}

/// The call of `text` with signatures and the call without return one
/// basis, and so does the call with cofactors, which the checker certifies
/// in its exact mode.
fn agree(name: &str, text: &str, seed: u64) {
    let with = params(seed);
    let mut without = params(seed);
    without.signatures = false;
    let basis = run(text, &with)
        .basis
        .unwrap_or_else(|| panic!("{name}: a basis"));
    let plain = run(text, &without).basis;
    assert_eq!(
        plain.as_deref(),
        Some(basis.as_str()),
        "{name}: without signatures"
    );
    let outcome = run(text, &with_cofactors(seed));
    assert_accepted(name, text, &outcome, Mode::Exact);
    assert_eq!(
        outcome.basis.as_deref(),
        Some(basis.as_str()),
        "{name}: with cofactors"
    );
}

/// Test 2 (bivium-gap.1 to .6): the edge cases of `OBJECTIVES.md` on x, y,
/// z over GF(2), GF(65521), and ℚ, on their own variables and placed on
/// 150.
#[test]
fn edge_cases_with_the_inputs_koszul_signatures() {
    for (case, generators) in EDGE_GENERATORS {
        for p in [2, P, 0] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            agree(&format!("{case}, p = {p}"), &text, 0xb1_1a);
            agree(
                &format!("{case}, p = {p}, on {WIDE}"),
                &wide(&text),
                0xb1_1b,
            );
        }
    }
}

/// Test 3 (bivium-gap.1 to .6): systems of many inputs, boolean systems of
/// 6, 8, and 10 variables and 2n generators beside their field
/// equations over GF(2), GF(65521), and ℚ, and random sparse systems of 3
/// and 4 variables over GF(65521) and of 3 over ℚ, seeds 0 to 2, on their
/// own variables and placed on 150.
#[test]
fn systems_of_many_inputs_with_the_inputs_koszul_signatures() {
    for n in [6, 8, 10usize] {
        for seed in 0..3u64 {
            let text = boolean(n, 2 * n, 0xb1_20 + 16 * seed + n as u64);
            for p in [2, P, 0] {
                let name = format!("boolean({n}, {m}) seed {seed}, p = {p}", m = 2 * n);
                let text = over(&text, p);
                agree(&name, &text, seed);
                agree(&format!("{name}, on {WIDE}"), &wide(&text), seed);
            }
        }
    }
    for (n, p) in [(3, P), (4, P), (3, 0)] {
        for seed in 0..3u64 {
            let text = random_sparse(n, p, 0xb1_30 + seed);
            let name = format!("random_sparse({n}) seed {seed}, p = {p}");
            agree(&name, &text, seed);
            agree(&format!("{name}, on {WIDE}"), &wide(&text), seed);
        }
    }
}
