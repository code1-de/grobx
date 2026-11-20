//! plain-zero-pairs-time, the item's test code: the B criterion drops a
//! pair where the new leading monomial's pair with one of its elements has
//! the pair's lcm and coprime leading monomials, in every format, with the
//! pair set's indexes and without them
//! (`iterations/0164-plain-zero-pairs-time/plan.md`, _Test code_, 1).

use crate::divisibility_tests::Engine;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::order::Grevlex;
use crate::pairs::INDEX_FORCED;

/// The variables a case's leads are placed on, and the place of each of
/// the case's four: 4 to stay within every format's mask, 40 past 32,
/// where a mask bit stands for several variables, and 150 past 128, where
/// a support bit does.
const WIDTHS: [(usize, [usize; 4]); 3] = [
    (4, [0, 1, 2, 3]),
    (40, [3, 17, 30, 38]),
    (150, [5, 70, 131, 149]),
];

/// A case: its leads on a, b, c, d, entered in order, the pairs (i, j) of
/// the set after the last, and what the case shows.
struct Case {
    leads: &'static [[u32; 4]],
    pairs: &'static [(u32, u32)],
    shows: &'static str,
}

const CASES: [Case; 4] = [
    Case {
        leads: &[[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]],
        pairs: &[(0, 2)],
        shows: "ab, ac, b: (1, 2) is coprime at abc, (0, 2) of lcm ab, so (0, 1) drops",
    },
    Case {
        leads: &[[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]],
        pairs: &[(1, 2)],
        shows: "ac, ab, b: (0, 2) is coprime at abc, (1, 2) of lcm ab, so (0, 1) drops",
    },
    Case {
        leads: &[[1, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 0]],
        pairs: &[(0, 1), (0, 2)],
        shows: "ab, acd, bc: (1, 2) has the lcm abcd and shares c, so (0, 1) stays",
    },
    Case {
        leads: &[[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]],
        pairs: &[(0, 1), (1, 2)],
        shows: "ab, bc, ac: both new pairs have the lcm abc and share a variable, so (0, 1) stays",
    },
];

/// The case's pairs in format M on `width` variables, the leads' variables
/// at `places`, with the pair set's indexes as `indexes` says.
fn pairs_in<M: Format>(
    case: &Case,
    width: usize,
    places: [usize; 4],
    indexes: bool,
) -> Vec<(u32, u32)> {
    INDEX_FORCED.set(Some(indexes));
    let mut engine: Engine<M> = Engine::new(width, 0x05e7_71ed);
    for short in case.leads {
        let mut lead = vec![0u32; width];
        for (&e, &v) in short.iter().zip(&places) {
            lead[v] = e;
        }
        let sugar = short.iter().sum();
        let (got, _) = engine.enter(&Grevlex, &lead, sugar);
        let shows = case.shows;
        assert_eq!(got, Ok(()), "{shows}");
    }
    INDEX_FORCED.set(None);
    let mut pairs: Vec<(u32, u32)> = engine.written().iter().map(|p| (p.i, p.j)).collect();
    pairs.sort_unstable();
    pairs
}

/// Each case in format M on each width, with the indexes and without.
fn cases_in<M: Format>() {
    for case in &CASES {
        for (width, places) in WIDTHS {
            for indexes in [false, true] {
                let got = pairs_in::<M>(case, width, places, indexes);
                let shows = case.shows;
                assert_eq!(
                    got, case.pairs,
                    "{shows} on {width} variables, indexes {indexes}"
                );
            }
        }
    }
}

// plain-zero-pairs-time.4 design-checked -- the item's test code checks the update on hand-built leads in every format, on 4, 40, and 150 variables, with the pair set's indexes and without them, beside the tests that hold the update to `divisibility_tests.rs`'s reference, which takes the rule, and the calls against the signature run and the checker, not the calls alone, which leave every result as it is where the B criterion drops fewer pairs than it may (log 0164-1)
/// Test 1 (plain-zero-pairs-time.1 to .3): the B criterion drops the pair
/// (0, 1) where the new lead's pair with one of its elements has its lcm
/// and coprime leading monomials and the other a smaller lcm, on either
/// side, and keeps it where that pair shares a variable or both have its
/// lcm, in `Dense4`, `Dense8`, `Dense16`, and `Dense32`, on 4, 40, and 150
/// variables, with the pair set's indexes and without them.
#[test]
fn a_coprime_pair_settles_its_lcm() {
    cases_in::<Dense4>();
    cases_in::<Dense8>();
    cases_in::<Dense16>();
    cases_in::<Dense32>();
}
