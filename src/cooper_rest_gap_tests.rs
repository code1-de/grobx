//! cooper-rest-gap, the crate's test code: in grevlex the update counts a
//! fallen element's pairs one degree past their lcms' while the set holds
//! at most `DEFER_LIVE` live pairs when the update begins, and at their
//! lcms' past it (cooper-rest-gap.1, .2). Written in session 0110-2,
//! beside the item's integration tests, whose bases do not show the bound.

use crate::divisibility_tests::Engine;
use crate::monomial::Dense16;
use crate::order::Grevlex;
use crate::pairs::DEFER_LIVE;

/// The `x_i·y`, whose pairs, of lcm `x_i·x_j·y`, all stay: 120 pairs.
pub(crate) const STAR: usize = 16;

/// The path's variables, `v_1` to `v_11`, for its longest path of 10
/// elements, `DEFER_LIVE` + 1 live pairs with the star's.
pub(crate) const PATH: usize = 11;

/// The variables: `x_1` to `x_16`, y, z, and the path's.
pub(crate) const VARIABLES: usize = STAR + 2 + PATH;

/// The degree of every lcm of the test, `x_i·x_j·y`,
/// `v_k·v_{k+1}·v_{k+2}`, and `x_i·y·z`.
pub(crate) const LCM_DEGREE: u32 = 3;

/// The exponents of the product of the variables at `places`.
pub(crate) fn product(places: &[usize]) -> Vec<u32> {
    let mut e = vec![0; VARIABLES];
    for &k in places {
        e[k] += 1;
    }
    e
}

/// The path of elements that brings the star's set to `DEFER_LIVE` live
/// pairs: each element of the path after the first makes one pair.
pub(crate) const AT_BOUND: usize = DEFER_LIVE - STAR * (STAR - 1) / 2 + 1;

/// The star's `x_i·y`, then a path of `path` elements `v_k·v_{k+1}`, each
/// making one pair with the one before and coprime to the star, all
/// entered at their degree.
pub(crate) fn star_and_path(path: usize) -> Engine<Dense16> {
    let (y, v) = (STAR, STAR + 2);
    let mut e: Engine<Dense16> = Engine::new(VARIABLES, 0xc00e);
    for i in 0..STAR {
        e.enter(&Grevlex, &product(&[i, y]), 2).0.unwrap();
    }
    for k in 0..path {
        e.enter(&Grevlex, &product(&[v + k, v + k + 1]), 2)
            .0
            .unwrap();
    }
    e
}

/// `star_and_path(path)`, then y·z, entered at `sugar`. Its pairs with
/// the star, of lcm `x_i·y·z`, are its only ones. The live pairs before
/// y·z's update, and the sugars of y·z's pairs.
fn probe(path: usize, sugar: u32) -> (usize, Vec<u32>) {
    let (y, z) = (STAR, STAR + 1);
    let mut e = star_and_path(path);
    let live = e.pairs.len();
    let written = e.written();
    assert!(
        written.iter().all(|p| p.sugar == LCM_DEGREE),
        "path {path}: the pairs of elements that did not fall"
    );
    e.enter(&Grevlex, &product(&[y, z]), sugar).0.unwrap();
    let t = u32::try_from(STAR + path).unwrap();
    let sugars: Vec<u32> = e
        .written()
        .iter()
        .filter(|p| p.j == t)
        .map(|p| p.sugar)
        .collect();
    assert_eq!(sugars.len(), STAR, "path {path}: y·z's pairs with the star");
    (live, sugars)
}

/// Test 4: at `DEFER_LIVE` live pairs y·z, fallen by one, makes pairs one
/// degree past their lcms'; at one more, and where y·z did not fall, at
/// their lcms'.
#[test]
fn the_deferral_stops_past_its_bound() {
    let at = AT_BOUND;
    let (live, sugars) = probe(at, LCM_DEGREE);
    assert_eq!(live, DEFER_LIVE, "the path of {at}");
    assert_eq!(sugars, vec![LCM_DEGREE + 1; STAR], "fallen at the bound");

    let (live, sugars) = probe(at + 1, LCM_DEGREE);
    assert_eq!(live, DEFER_LIVE + 1, "the path of {n}", n = at + 1);
    assert_eq!(sugars, vec![LCM_DEGREE; STAR], "fallen past the bound");

    let (_, sugars) = probe(at, 2);
    assert_eq!(sugars, vec![LCM_DEGREE; STAR], "not fallen at the bound");
}
