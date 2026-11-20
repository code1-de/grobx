//! echelon-lane-zero, taken into `q-denominator-line`'s pick: the
//! generators' reduced row echelon form over a pair of primes holds no
//! entry zero at both lanes, each lane of it is the echelon over that
//! lane's prime, and a pair learn whose lanes fit no longer misfits for
//! such an entry (`iterations/0104-q-denominator-line/plan.md`, _Test
//! code_, 6 to 9).

use crate::echelon::{Echelon, echelon, replay};
use crate::echelon_replay_tests::leads;
use crate::entry::Imaged;
use crate::field::{Field, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::halt::Halt;
use crate::input_redundancy_tests::{read, table};
use crate::learn_pair_tests::run_in;
use crate::module::Rank1;
use crate::monomial::Dense16;
use crate::order::Grevlex;
use crate::poly::Poly;
use crate::shared_tests::{LEARN0, LEARN1, REPLAY0, REPLAY1, budget, params, what};
use crate::table::Table;
use crate::trace::NoLearner;

/// `echelon` over `field` of `generators` on `names` in grevlex, with the
/// table it read them into, or the halt.
fn rows<F: Field>(
    field: &F,
    names: &str,
    generators: &str,
) -> Result<(Table<Dense16>, Echelon<F, Rank1>), Halt> {
    let mut t = table(names);
    let g = read(field, &mut t, &Grevlex, names, generators);
    let e = echelon(field, &mut t, &Grevlex, g, &mut budget())?;
    Ok((t, e))
}

/// Lane `lane` of `row`: per monomial its exponents and the lane's
/// coefficient, the monomials whose coefficient is 0 at the lane left out.
fn terms<F: Field>(
    field: &F,
    t: &Table<Dense16>,
    row: &Poly<F, Rank1>,
    lane: usize,
) -> Vec<(Vec<u32>, u64)> {
    row.monomials
        .iter()
        .zip(&row.coefs)
        .filter_map(|(&m, &c)| {
            let v = field.lane(c, lane);
            (v != 0).then(|| {
                let mut x = Vec::new();
                t.exponents(m, &mut x);
                (x, v)
            })
        })
        .collect()
}

/// The echelon over `pair` of `generators`: no coefficient zero at both
/// lanes, and lane k of every row the row of the echelon over `lanes[k]`,
/// the field at the pair's k-th prime. The pair's rows and their table.
fn lanes_agree<P: Field, F: Field>(
    case: &str,
    pair: &P,
    lanes: &[F; 2],
    names: &str,
    generators: &str,
) -> (Table<Dense16>, Echelon<P, Rank1>) {
    let (tp, ep) =
        rows(pair, names, generators).unwrap_or_else(|h| panic!("{case}: the pair halted: {h:?}"));
    for (k, row) in ep.rows.iter().enumerate() {
        for &c in &row.coefs {
            assert!(
                pair.lane(c, 0) != 0 || pair.lane(c, 1) != 0,
                "{case}: row {k} holds a coefficient zero at both lanes"
            );
        }
    }
    for (lane, field) in lanes.iter().enumerate() {
        let (tl, el) = rows(field, names, generators)
            .unwrap_or_else(|h| panic!("{case}: lane {lane} halted: {h:?}"));
        assert_eq!(ep.rows.len(), el.rows.len(), "{case}: lane {lane}'s rows");
        for (k, (rp, rl)) in ep.rows.iter().zip(&el.rows).enumerate() {
            assert_eq!(
                terms(pair, &tp, rp, lane),
                terms(field, &tl, rl, 0),
                "{case}: row {k} at lane {lane}\n{generators}"
            );
        }
    }
    (tp, ep)
}

/// Tests 6 and 7 over `pair` at the primes (p0, p1), `lanes` the fields at
/// each: the generators `x^2+p1*z, p0*x^2+y`, whose second row's leading
/// coefficient is 0 at lane 0 and whose pivot's entry at z is 0 at lane 1;
/// the same with `w` for `y`, where the entry at z would lead; and with
/// `p0*w`, where the lanes' echelons differ in their leads and the pair
/// misfits.
fn six_and_seven<P: Field, F: Field>(case: &str, pair: &P, lanes: &[F; 2], (p0, p1): (u64, u64)) {
    // Test 6: the row `y` holds one term, not `y` and an explicit zero at z.
    let generators = format!("x^2+{p1}*z, {p0}*x^2+y");
    let (_, e) = lanes_agree(case, pair, lanes, "x,y,z", &generators);
    assert_eq!(e.rows.len(), 2, "{case}: two rows\n{generators}");
    assert_eq!(
        e.rows[0].monomials.len(),
        2,
        "{case}: x^2+c*z\n{generators}"
    );
    assert_eq!(
        e.rows[1].monomials.len(),
        1,
        "{case}: y alone\n{generators}"
    );
    // Test 7: the entry at z would lead the second row.
    let generators = format!("x^2+{p1}*z, {p0}*x^2+w");
    let (t, e) = lanes_agree(case, pair, lanes, "x,y,z,w", &generators);
    assert_eq!(e.rows.len(), 2, "{case}: two rows\n{generators}");
    assert_eq!(
        e.rows[1].monomials.len(),
        1,
        "{case}: w alone\n{generators}"
    );
    // The replay from the pair's record gives the pair's rows.
    let want = leads(&t, &e);
    let mut t2 = table("x,y,z,w");
    let g = read(pair, &mut t2, &Grevlex, "x,y,z,w", &generators);
    let got = replay(pair, &mut t2, &Grevlex, g, &want, &mut budget())
        .unwrap_or_else(|h| panic!("{case}: the replay halted: {h:?}\n{generators}"));
    assert_eq!(got.columns, e.columns, "{case}: the replay's columns");
    for (k, (r, w)) in got.rows.iter().zip(&e.rows).enumerate() {
        assert_eq!(
            r.coefs, w.coefs,
            "{case}: the replay's row {k}\n{generators}"
        );
        assert_eq!(
            r.monomials.len(),
            w.monomials.len(),
            "{case}: the replay's row {k}'s terms"
        );
    }
    // The lanes' echelons differ in their leads: the second generator is 0
    // at lane 0 and `w` at lane 1 after the pivot's subtraction.
    let generators = format!("x^2+{p1}*z, {p0}*x^2+{p0}*w");
    let got = rows(pair, "x,y,z,w", &generators);
    assert!(
        matches!(got, Err(Halt::Misfit(0))),
        "{case}: the lanes' leads differ, and the pair misfits\n{generators}"
    );
}

/// Tests 6 and 7, echelon-lane-zero.1, over `Gf31Pair` at (LEARN0,
/// LEARN1).
#[test]
fn no_entry_zero_at_both_lanes_of_gf31pair() {
    six_and_seven(
        "Gf31Pair",
        &Gf31Pair::primes(LEARN0, LEARN1),
        &[Gf31::prime(LEARN0), Gf31::prime(LEARN1)],
        (LEARN0, LEARN1),
    );
}

/// Test 8, echelon-lane-zero.1: tests 6 and 7 over `Gf64Pair` at
/// (REPLAY0, REPLAY1).
#[test]
fn no_entry_zero_at_both_lanes_of_gf64pair() {
    six_and_seven(
        "Gf64Pair",
        &Gf64Pair::primes(REPLAY0, REPLAY1),
        &[Gf64::prime(REPLAY0), Gf64::prime(REPLAY1)],
        (REPLAY0, REPLAY1),
    );
}

/// Test 9, echelon-lane-zero.1: a pair learn through `image_in` over
/// `Gf31Pair` at (LEARN0, LEARN1) on test 7's generators gives an image per
/// lane, each the learn run's at its prime, where the code before the change
/// halted the pair with a misfit.
#[test]
fn a_pair_learn_whose_lanes_fit() {
    let text = format!("x,y,z,w\n0\nx^2+{LEARN1}*z,\n{LEARN0}*x^2+w\n");
    let params = params(1);
    let got = run_in(
        Gf31Pair::primes(LEARN0, LEARN1),
        &text,
        &params,
        &mut NoLearner,
    );
    let Ok(Imaged::Lanes(images)) = got else {
        panic!("the pair learn returned {got}\n{text}", got = what(&got));
    };
    assert_eq!(images.len(), 2, "an image per lane\n{text}");
    for (lane, p) in [LEARN0, LEARN1].into_iter().enumerate() {
        let alone = run_in(Gf31::prime(p), &text, &params, &mut NoLearner);
        let Ok(Imaged::Image(image)) = alone else {
            panic!(
                "the learn at {p} returned {got}\n{text}",
                got = what(&alone)
            );
        };
        assert_eq!(
            images[lane].leads, image.leads,
            "lane {lane}'s leads\n{text}"
        );
        assert_eq!(
            images[lane].basis, image.basis,
            "lane {lane}'s basis\n{text}"
        );
    }
}
