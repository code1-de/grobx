//! fallen-pair-losses, the crate's test code: in grevlex the update counts a
//! fallen element's pairs one degree past their lcms' only until the first
//! update that begins with more than `DEFER_LIVE` live pairs, whatever the
//! set holds after it, and a pair deferred before keeps its sugar
//! (fallen-pair-losses.1, .2; `iterations/0111-fallen-pair-losses/plan.md`,
//! _Test code_, 3).

use crate::budget::Budget;
use crate::cooper_rest_gap_tests::{AT_BOUND, LCM_DEGREE, STAR, product, star_and_path};
use crate::divisibility_tests::Engine;
use crate::monomial::Dense16;
use crate::order::Grevlex;
use crate::pairs::{self, DEFER_LIVE};

/// The variable y, which every element of the star holds.
const Y: usize = STAR;

/// The variable z.
const Z: usize = STAR + 1;

/// The path's first variable, `v_1`.
const V1: usize = STAR + 2;

/// The path's second variable, `v_2`.
const V2: usize = STAR + 3;

/// The sugars of element t's pairs with the elements before it.
fn sugars_of(e: &Engine<Dense16>, t: usize) -> Vec<u32> {
    let t = u32::try_from(t).unwrap();
    e.written()
        .iter()
        .filter(|p| p.j == t)
        .map(|p| p.sugar)
        .collect()
}

/// The star and a path of `path` elements, then y·z at its degree, which
/// does not fall; then the selection of every pair, all of lcm degree 3;
/// then `z·v_1`, fallen by one, whose two pairs, with y·z and with `v_1·v_2`,
/// are its only ones. The live pairs before y·z's update, and the sugars
/// of `z·v_1`'s pairs.
fn after_selection(path: usize) -> (usize, Vec<u32>) {
    let mut e = star_and_path(path);
    let live = e.pairs.len();
    e.enter(&Grevlex, &product(&[Y, Z]), 2).0.unwrap();
    let taken = pairs::select(
        &mut e.pairs,
        &mut e.table,
        &Grevlex,
        &mut Budget::unbounded(),
    )
    .unwrap();
    assert!(
        taken.iter().all(|p| p.sugar == LCM_DEGREE),
        "path {path}: every pair at its lcm's degree"
    );
    assert!(
        e.pairs.is_empty(),
        "path {path}: the selection takes every pair"
    );
    e.enter(&Grevlex, &product(&[Z, V1]), LCM_DEGREE).0.unwrap();
    let sugars = sugars_of(&e, STAR + path + 1);
    assert_eq!(sugars.len(), 2, "path {path}: z·v_1's pairs");
    (live, sugars)
}

/// Test 3, fallen-pair-losses.1: once an update began past the bound, `z·v_1`,
/// fallen by one, makes its pairs at their lcms' degree though the set is
/// empty; where no update began past it, one degree past.
#[test]
fn the_deferral_stops_for_good() {
    let (live, sugars) = after_selection(AT_BOUND + 1);
    assert_eq!(live, DEFER_LIVE + 1, "y·z's update begins past the bound");
    assert_eq!(sugars, vec![LCM_DEGREE; 2], "fallen after the latch");

    let (live, sugars) = after_selection(AT_BOUND);
    assert_eq!(live, DEFER_LIVE, "y·z's update begins at the bound");
    assert_eq!(sugars, vec![LCM_DEGREE + 1; 2], "fallen, no latch");
}

/// Test 3, fallen-pair-losses.2: y·z, fallen by one at the bound, makes its
/// pairs one degree past their lcms'; the update of `z·v_1` begins past the
/// bound and closes the latch, and y·z's pairs keep their sugar. The
/// selection then takes every pair at its lcm's degree and leaves y·z's,
/// and `z·v_2`, fallen by one with the set below the bound, makes its pairs
/// at their lcms' degree.
#[test]
fn deferred_pairs_keep_their_sugar() {
    let mut e = star_and_path(AT_BOUND);
    let yz = STAR + AT_BOUND;
    e.enter(&Grevlex, &product(&[Y, Z]), LCM_DEGREE).0.unwrap();
    assert_eq!(
        sugars_of(&e, yz),
        vec![LCM_DEGREE + 1; STAR],
        "y·z fallen at the bound"
    );
    assert!(e.pairs.len() > DEFER_LIVE, "the set past the bound");
    e.enter(&Grevlex, &product(&[Z, V1]), 2).0.unwrap();
    assert_eq!(
        sugars_of(&e, yz),
        vec![LCM_DEGREE + 1; STAR],
        "y·z's pairs after the latch"
    );
    assert!(
        sugars_of(&e, yz + 1).iter().all(|&s| s == LCM_DEGREE),
        "z·v_1's pairs, not fallen"
    );

    pairs::select(
        &mut e.pairs,
        &mut e.table,
        &Grevlex,
        &mut Budget::unbounded(),
    )
    .unwrap();
    assert!(
        e.pairs.iter().all(|p| p.sugar == LCM_DEGREE + 1),
        "the selection leaves y·z's pairs"
    );
    assert!(e.pairs.len() <= DEFER_LIVE, "the set below the bound");
    e.enter(&Grevlex, &product(&[Z, V2]), LCM_DEGREE).0.unwrap();
    let sugars = sugars_of(&e, yz + 2);
    assert!(!sugars.is_empty(), "z·v_2's pairs");
    assert!(
        sugars.iter().all(|&s| s == LCM_DEGREE),
        "z·v_2 fallen after the latch: {sugars:?}"
    );
}
