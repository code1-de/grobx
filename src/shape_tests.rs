//! q-replay-time, sub-item 1: a replay from the shape its record's first
//! replay kept returns the full replay's image at its prime, where an
//! element's terms differ from the shape, in lex, and in `block:1`, and so
//! does a replay with cofactors, which keeps no shape (plan.md, _Test
//! code_). The oracle is the full replay at the same prime, `kept` None.

use std::num::NonZeroUsize;
use std::time::Instant;

use crate::budget::Budget;
use crate::entry::{Imaged, image_in};
use crate::field::Gf31;
use crate::lift::Image;
use crate::shape::Kept;
use crate::systems::{dense, random_square, random_system};
use crate::text::Header;
use crate::trace::{NoLearner, NoTrace, Record, Recording};
use crate::{Error, Params, Threads};

/// A prime near 2^30.5 that the cases below make a coefficient vanish at.
const U1: u64 = 1_518_500_279;
/// Primes near 2^31, at which no coefficient below vanishes.
const L1: u64 = 2_147_483_647;
const L2: u64 = 2_147_483_629;
const L3: u64 = 2_147_483_587;
const L4: u64 = 2_147_483_579;

fn params_in(order: &str) -> Params {
    let order = order.parse().expect("a name of the family");
    Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), 7)
}

fn params() -> Params {
    params_in("grevlex")
}

/// The record of a learn run over `text`, an instance over Q, at the prime
/// `p`.
fn learn(text: &str, params: &Params, p: u64) -> Record {
    let header = Header::read(text).unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    let mut recording = Recording::new();
    let got = image_in(
        Gf31::prime(p),
        text,
        params,
        None,
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        &mut recording,
        None,
    )
    .unwrap_or_else(|e| panic!("the learn run at {p} failed: {e:?}\non\n{text}"));
    assert!(
        matches!(got, Imaged::Image(_)),
        "the learn run at {p} stopped"
    );
    recording.finish(named)
}

/// The replay of `record` over `text` at the prime `p`: from or into the
/// shape `kept` holds where `kept` is Some, the full replay where it is
/// None.
fn replay(
    text: &str,
    params: &Params,
    record: &Record,
    p: u64,
    kept: Option<&mut Kept>,
) -> Result<Image, Error> {
    let header = Header::read(text).unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    let got = image_in(
        Gf31::prime(p),
        text,
        params,
        Some(record),
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        &mut NoLearner,
        kept,
    )?;
    match got {
        Imaged::Image(image) => Ok(image),
        Imaged::Lanes(_) | Imaged::Pair(_) => {
            unreachable!("a replay over GF(p) makes one image")
        }
        Imaged::Stopped(o) => panic!("the replay at {p} stopped: {o:?}"),
    }
}

/// The replay at `p` with `kept` equals the full replay at `p`: both an
/// image, the same, or both the same error.
fn same_as_full(case: &str, text: &str, params: &Params, record: &Record, p: u64, kept: &mut Kept) {
    let shaped = replay(text, params, record, p, Some(kept));
    let full = replay(text, params, record, p, None);
    assert_eq!(shaped, full, "{case}: the replay at {p}");
}

/// q-replay-time.1, .2: random systems, random square systems, and
/// dense cubic systems over Q, each learned at L1. The replays at L2, L3, and L4
/// with one `Kept` return the full replay's image.
#[test]
fn a_shaped_replay_equals_the_full_replay() {
    let mut cases = Vec::new();
    for seed in 1..=3 {
        cases.push((format!("random system {seed}"), random_system(0, seed)));
        cases.push((format!("random square {seed}"), random_square(0, seed)));
    }
    cases.push(("cubic 2".to_string(), dense(2, 3, 0, 1)));
    cases.push(("cubic 3".to_string(), dense(3, 3, 0, 1)));
    for (case, text) in &cases {
        let p = params();
        let record = learn(text, &p, L1);
        let mut kept = Kept::default();
        for q in [L2, L3, L4] {
            same_as_full(case, text, &p, &record, q, &mut kept);
        }
    }
}

/// Learns at L1 under `p`, then replays at L2, U1, L3, and L4 with one
/// `Kept`, each the full replay's image (q-replay-time.4, .5).
fn refits(case: &str, text: &str, p: &Params) {
    let record = learn(text, p, L1);
    let mut kept = Kept::default();
    for q in [L2, U1, L3, L4] {
        same_as_full(case, text, p, &record, q, &mut kept);
    }
}

/// q-replay-time.4: a generator's term in y vanishes modulo U1.
#[test]
fn an_input_term_that_vanishes() {
    let text = format!("x,y,z\n0\nx^2+{U1}*y+z,\ny^2+z\n");
    refits("an input term", &text, &params());
}

/// q-replay-time.4: the first step makes y − U1·z over Q, y modulo U1, and
/// the leading monomials are x² and y at every prime. The generators keep
/// their terms at U1, so the difference lies in a made row. The run takes
/// the generators as read: their echelon would make y − U1·z itself.
#[test]
fn an_element_term_that_vanishes() {
    let a = U1 + 1;
    let text = format!("x,y,z\n0\nx^2+{a}*z,\nx^2+y+z\n");
    let mut p = params();
    p.echelon = false;
    let record = learn(&text, &p, L1);
    let vanished = replay(&text, &p, &record, U1, None).expect("the full replay at U1 fits");
    let full = replay(&text, &p, &record, L2, None).unwrap();
    let leads =
        |image: &Image| -> Vec<Vec<u32>> { image.leads.chunks(3).map(<[u32]>::to_vec).collect() };
    assert_eq!(leads(&vanished), leads(&full), "the same leading monomials");
    let terms = |image: &Image| -> usize { image.basis.iter().map(|g| g.coefs.len()).sum() };
    assert!(terms(&vanished) < terms(&full), "a term vanishes at U1");
    refits("an element term", &text, &p);
}

/// The inter-reduction makes x − U1·z over Q, x modulo U1, from inputs that
/// keep their terms at U1. The replays at L2, U1, and L3 with one `Kept`
/// return the full replay's image. The run takes the generators as
/// read: their echelon would make x − U1·z itself.
#[test]
fn an_inter_reduced_term_that_vanishes() {
    let a = U1 + 1;
    let text = format!("x,y,z\n0\nx+y+z,\ny+{a}*z\n");
    let mut p = params();
    p.echelon = false;
    let record = learn(&text, &p, L1);
    let vanished = replay(&text, &p, &record, U1, None).unwrap();
    let full = replay(&text, &p, &record, L2, None).unwrap();
    let terms = |image: &Image| -> usize { image.basis.iter().map(|g| g.coefs.len()).sum() };
    assert!(terms(&vanished) < terms(&full), "a term vanishes at U1");
    let mut kept = Kept::default();
    for q in [L2, U1, L3] {
        same_as_full("an inter-reduced term", &text, &p, &record, q, &mut kept);
    }
}

/// q-replay-time.4: over Q the first step makes an element led by y²,
/// modulo U1 one led by x. The replay from the shape misfits at U1 as the
/// full replay does, at the same step.
#[test]
fn a_misfit_from_the_shape_is_the_full_replays() {
    let a = U1 + 1;
    let text = format!("x,y\n0\nx^2+{a}*y^2+y,\nx^2+y^2+x\n");
    // The generators as read: in echelon form they misfit at the inputs
    // (input-redundancy-time.8).
    let mut p = params();
    p.echelon = false;
    let record = learn(&text, &p, L1);
    let mut kept = Kept::default();
    same_as_full("misfit", &text, &p, &record, L2, &mut kept);
    let full = replay(&text, &p, &record, U1, None);
    assert!(
        matches!(full, Err(Error::Misfit { step }) if step > 0),
        "the full replay misfits at a step: {full:?}"
    );
    let shaped = replay(&text, &p, &record, U1, Some(&mut kept));
    assert_eq!(shaped, full);
}

/// q-replay-time.7: a replay that tracks cofactors, of a dense cubic system
/// over Q, returns the full replay's image with its cofactors.
#[test]
fn a_replay_with_cofactors_equals_the_full_replay() {
    let text = dense(2, 3, 0, 1);
    // The lift's learn runs track no cofactors (q-lift.12), and take the
    // generators as read where its replays track cofactors
    // (input-redundancy-time.8).
    let mut p = params();
    p.echelon = false;
    let record = learn(&text, &p, L1);
    p.cofactors = true;
    let mut kept = Kept::default();
    for q in [L2, L3] {
        same_as_full("cofactors", &text, &p, &record, q, &mut kept);
        let image = replay(&text, &p, &record, q, None).unwrap();
        assert!(
            !image.cofactors.is_empty(),
            "the replay at {q} tracks cofactors"
        );
    }
}

/// `OBJECTIVES.md`, _Domain specifics the workflow points at_: each edge
/// case's replays at L2, L3, and L4 with one `Kept` return the full
/// replay's image.
#[test]
fn the_edge_cases_replay_from_the_shape() {
    let cases = [
        ("the zero ideal", "x,y\n0\n0\n"),
        ("F = (0), two zeros", "x,y\n0\n0,\n0\n"),
        ("a constant input", "x,y\n0\nx+y,\n7\n"),
        ("a constant made in a step", "x,y\n0\nx*y-1/2,\nx,\ny+3\n"),
        ("a Gröbner basis already", "x,y\n0\n3*x+1,\n2*y^2-5\n"),
        ("a zero generator", "x,y\n0\nx^2+y,\n0,\nx*y+1\n"),
        ("an eliminable variable", "x,y\n0\nx-1/3*y^2,\ny^3-2\n"),
    ];
    for (case, text) in cases {
        let p = params();
        let record = learn(text, &p, L1);
        let mut kept = Kept::default();
        for q in [L2, L3, L4] {
            same_as_full(case, text, &p, &record, q, &mut kept);
        }
    }
}

/// The replays at L2, L3, and L4 with one `Kept` of a record whose replays
/// overflow 8-bit lanes return the full replay's image.
#[test]
fn a_replay_past_8_bit_lanes() {
    let text = "x,y\n0\nx^300-y,\nx*y-1\n";
    let p = params();
    let record = learn(text, &p, L1);
    let mut kept = Kept::default();
    for q in [L2, L3, L4] {
        same_as_full("wide", text, &p, &record, q, &mut kept);
    }
}

/// The replays in lex and in `block:1` of a dense cubic and a random
/// square system over Q return the full replay's image.
#[test]
fn lex_and_block_replay_from_the_shape() {
    let cases = [
        ("cubic 2", dense(2, 3, 0, 1)),
        ("random square 2", random_square(0, 2)),
    ];
    for order in ["lex", "block:1"] {
        for (case, text) in &cases {
            let p = params_in(order);
            let record = learn(text, &p, L1);
            let mut kept = Kept::default();
            for q in [L2, L3, L4] {
                same_as_full(&format!("{case} in {order}"), text, &p, &record, q, &mut kept);
            }
        }
    }
}
