//! q-image-exponents-peak, the item's test code: a call over ℚ whose
//! images and parts hold each exponent in a byte returns what the same
//! call returns with each in a `u32`, in every order, with cofactors and
//! the checker, on the edge cases, under a memory ceiling, and under
//! deadlines; the exponent vectors by value and their widening; the images
//! a run hands over; and a merge across the two widths
//! (`iterations/0120-q-image-exponents-peak/plan.md`, _Test code_, 1 to 5).
//!
//! A pair of calls is the same call with `EXPONENTS_IN_WORDS` set, every
//! vector in `u32`, and unset.

use std::time::{Duration, Instant};

use crate::budget::Budget;
use crate::call::built;
use crate::entry::Image;
use crate::later_runs_tests::alone;
use crate::lift::{Part, append, merge};
use crate::lift_rest_tests::pair_image;
use crate::poly::{EXPONENTS_IN_WORDS, Exponents, Sparse};
use crate::rebuilt_reducer_tests::reported;
use crate::shared_tests::{L1, W1, W2, learn, lift_params};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_sparse};
use crate::{CheckerMode, MonomialOrder, Outcome, Params, Stop};

/// Two systems over x, y, z whose second element's tail holds an exponent
/// of 300, past a byte, and 255, the most a byte holds; both compute in
/// `Dense16`.
const WIDE: [&str; 2] = [
    "x,y,z\n0\nx*y^300+5*x^300+7,\nz-2\n",
    "x,y,z\n0\nx*y^255+3*x^255+1,\nz-2\n",
];

/// A system over x alone.
const ONE: &str = "x\n0\nx^3-2*x+5\n";

/// A system over ten variables, whose monomials take two words of
/// `Dense8`.
const TEN: &str = "x1,x2,x3,x4,x5,x6,x7,x8,x9,x10\n0\n\
                   x1*x2-3*x10+x4,\n\
                   x3^2+x4*x5-7*x9,\n\
                   x6*x7*x8-2*x9+1,\n\
                   x10^2-5*x1+x2*x3,\n\
                   x5*x6-x7+2\n";

/// The call of `text` with every exponent vector in `u32` where `words`.
fn call(text: &str, params: &Params, words: bool) -> Outcome {
    EXPONENTS_IN_WORDS.set(words);
    let outcome = crate::groebner(text, params);
    EXPONENTS_IN_WORDS.set(false);
    outcome.unwrap_or_else(|e| panic!("{e:?} on\n{text}"))
}

/// Asserts that the call of `text` returns with bytes what it returns
/// with `u32`, and returns the outcome with bytes.
fn pair(name: &str, text: &str, params: &Params) -> Outcome {
    let words = call(text, params, true);
    let bytes = call(text, params, false);
    assert_eq!(bytes.basis, words.basis, "{name}: the bases");
    assert_eq!(
        bytes.cofactors, words.cofactors,
        "{name}: the cofactor texts"
    );
    assert_eq!(bytes.stopped, words.stopped, "{name}: the stops");
    assert_eq!(bytes.certified, words.certified, "{name}: the verdicts");
    assert_eq!(bytes.flaw, words.flaw, "{name}: the flaws");
    assert_eq!(bytes.stats, words.stats, "{name}: the stats");
    bytes
}

/// The exponents of `terms`, in a vector that starts in bytes.
fn exponents(terms: &[&[u32]]) -> Exponents {
    let mut e = Exponents::new();
    for term in terms {
        e.extend_from_slice(term);
    }
    e
}

/// An image's polynomial of `terms` with the coefficients `coefs`.
fn sparse(terms: &[&[u32]], coefs: &[u64]) -> Sparse {
    Sparse {
        exponents: exponents(terms),
        coefs: coefs.to_vec(),
    }
}

/// Test 1, q-image-exponents-peak.1 to .3, .5, .6, .8 to .10: every call
/// over ℚ returns with bytes what it returns with `u32`: random sparse and
/// dense systems, the edge cases, `WIDE`, `ONE`, and `TEN`, the signatures
/// off and on; a dense system and an eliminable one in lex and `block:1`;
/// with cofactors and the exact checker; and under a memory ceiling.
#[test]
fn every_result_stays() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for s in 1..=3 {
        texts.push((format!("sparse4-{s}"), random_sparse(4, 0, s)));
        texts.push((format!("sparse5-{s}"), random_sparse(5, 0, s)));
        texts.push((format!("dense33-{s}"), dense(3, 3, 0, s)));
    }
    for (k, gens) in EDGE.iter().enumerate() {
        texts.push((format!("edge {k}"), format!("x,y,z\n0\n{gens}\n")));
    }
    for (k, text) in WIDE.iter().enumerate() {
        texts.push((format!("wide {k}"), (*text).to_string()));
    }
    texts.push(("one".to_string(), ONE.to_string()));
    texts.push(("ten".to_string(), TEN.to_string()));
    for (system, text) in &texts {
        for signatures in [false, true] {
            let mut params = reported(MonomialOrder::Grevlex);
            params.signatures = signatures;
            let name = format!("{system}, signatures {signatures}");
            let outcome = pair(&name, text, &params);
            assert!(outcome.basis.is_some(), "{name}: a basis");
        }
    }
    let eliminable = "x,y,z\n0\nx+z,\nz^2+1\n";
    for order in ["lex", "block:1"] {
        let params = reported(order.parse().unwrap());
        for (name, text) in [
            ("dense33-4", dense(3, 3, 0, 4)),
            ("eliminable", eliminable.to_string()),
        ] {
            pair(&format!("{name} in {order}"), &text, &params);
        }
    }
    for (name, text) in [
        ("sparse4-1", random_sparse(4, 0, 1)),
        ("wide 0", WIDE[0].to_string()),
    ] {
        let mut params = reported(MonomialOrder::Grevlex);
        params.cofactors = true;
        params.checker = Some(CheckerMode::Exact);
        if built(&params).is_err() {
            continue;
        }
        let name = format!("{name} with cofactors");
        let outcome = pair(&name, &text, &params);
        assert!(outcome.cofactors.is_some(), "{name}: the cofactor text");
        assert_eq!(outcome.certified, Some(true), "{name}: certified");
    }
    let mut params = reported(MonomialOrder::Grevlex);
    params.memory = Some(crate::os::resident() + (1 << 30));
    pair(
        "sparse5-1 under a ceiling",
        &random_sparse(5, 0, 1),
        &params,
    );
}

/// Test 2, q-image-exponents-peak.1 to .4, .6: a vector holds bytes while
/// every exponent fits one and widens whole at the first that does not,
/// every value kept; a term taken from another vector widens it only where
/// its values pass a byte; vectors and terms are equal by value across the
/// widths; and with `EXPONENTS_IN_WORDS` set a vector holds `u32`.
#[test]
fn exponents_by_value() {
    let mut e = exponents(&[&[1, 2, 255], &[0, 0, 7]]);
    assert_eq!(e.len(), 6);
    assert_eq!(e.bytes(), 6, "a byte an exponent");
    assert_eq!(e.values().collect::<Vec<u32>>(), [1, 2, 255, 0, 0, 7]);
    let mut term = [0u32; 3];
    e.widen_into(3..6, &mut term);
    assert_eq!(term, [0, 0, 7]);

    e.extend_from_slice(&[256, 0, 1]);
    assert_eq!(e.len(), 9);
    assert_eq!(e.bytes(), 36, "four bytes an exponent past 255");
    assert_eq!(
        e.values().collect::<Vec<u32>>(),
        [1, 2, 255, 0, 0, 7, 256, 0, 1],
        "every value kept"
    );
    e.widen_into(0..3, &mut term);
    assert_eq!(term, [1, 2, 255]);

    // A term of the wide vector whose values fit a byte, and one past it.
    let mut narrow = exponents(&[&[3, 3, 3]]);
    narrow.extend_from(&e, 3..6);
    assert_eq!(narrow.bytes(), 6, "a term that fits leaves bytes");
    assert_eq!(narrow.values().collect::<Vec<u32>>(), [3, 3, 3, 0, 0, 7]);
    narrow.extend_from(&e, 6..9);
    assert_eq!(narrow.bytes(), 36, "a term past a byte widens");
    assert_eq!(
        narrow.values().collect::<Vec<u32>>(),
        [3, 3, 3, 0, 0, 7, 256, 0, 1]
    );

    // Equality by value across the widths.
    let bytes = exponents(&[&[1, 2, 255], &[0, 0, 7]]);
    EXPONENTS_IN_WORDS.set(true);
    let words = exponents(&[&[1, 2, 255], &[0, 0, 7]]);
    let changed = exponents(&[&[1, 2, 255], &[0, 0, 8]]);
    EXPONENTS_IN_WORDS.set(false);
    assert_eq!(bytes.bytes(), 6);
    assert_eq!(
        words.bytes(),
        24,
        "EXPONENTS_IN_WORDS: four bytes an exponent"
    );
    assert_eq!(bytes, words);
    assert_eq!(words, bytes);
    assert!(bytes.same(0..3, &words, 0..3));
    assert!(bytes.same(3..6, &words, 3..6));
    assert!(!bytes.same(0..3, &words, 3..6));
    assert_ne!(bytes, changed, "one value changed");
    assert!(!bytes.same(3..6, &changed, 3..6));
    assert!(bytes.same(0..3, &changed, 0..3));
    assert_ne!(bytes, exponents(&[&[1, 2, 255]]), "another length");
    assert_eq!(Exponents::new(), Exponents::new());
    assert_eq!(Exponents::new().len(), 0);
}

/// The element of `image`'s basis whose tail holds an exponent past 255,
/// and the others.
fn widths(image: &Image) -> (Vec<&Sparse>, Vec<&Sparse>) {
    image
        .basis
        .iter()
        .partition(|g| g.exponents.values().any(|e| e > 255))
}

/// Test 3, q-image-exponents-peak.1, .2, .9: the images a replay alone and
/// a pair replay hand over hold a byte an exponent on a dense system, and
/// four bytes on the element of `WIDE[0]` whose tail holds x^300, the other
/// element in bytes; the leading monomials stay `u32`. With
/// `EXPONENTS_IN_WORDS` set, the dense system's take four bytes, with the
/// same values.
#[test]
fn an_image_holds_bytes() {
    let p = lift_params(1);
    let text = dense(3, 3, 0, 1);
    let record = learn(&text, &p, L1);
    let image = alone(&text, &p, &record, W1);
    assert!(!image.basis.is_empty(), "{image:?}");
    for g in &image.basis {
        assert_eq!(g.exponents.bytes(), g.exponents.len(), "{g:?}");
        assert_eq!(g.exponents.len(), 3 * g.coefs.len(), "{g:?}");
    }
    let pair = pair_image(&text, &p, &record, (W1, W2), None);
    for g in &pair.first {
        assert_eq!(g.exponents.bytes(), g.exponents.len(), "{g:?}");
    }
    EXPONENTS_IN_WORDS.set(true);
    let words = alone(&text, &p, &record, W1);
    EXPONENTS_IN_WORDS.set(false);
    for (w, b) in words.basis.iter().zip(&image.basis) {
        assert_eq!(w.exponents.bytes(), 4 * w.exponents.len(), "{w:?}");
        assert_eq!(
            w.exponents.values().collect::<Vec<u32>>(),
            b.exponents.values().collect::<Vec<u32>>()
        );
        assert_eq!(w.coefs, b.coefs);
    }
    assert_eq!(words.leads, image.leads);

    let text = WIDE[0];
    let record = learn(text, &p, L1);
    let image = alone(text, &p, &record, W1);
    let (wide, narrow) = widths(&image);
    assert_eq!(wide.len(), 1, "{image:?}");
    assert_eq!(wide[0].exponents.bytes(), 4 * wide[0].exponents.len());
    assert!(wide[0].exponents.values().any(|e| e == 300));
    assert!(!narrow.is_empty(), "{image:?}");
    for g in narrow {
        assert_eq!(g.exponents.bytes(), g.exponents.len(), "{g:?}");
    }
    assert!(
        image.leads.as_chunks::<3>().0.contains(&[1, 300, 0]),
        "the leading monomial x·y^300 in u32: {image:?}"
    );
    let pair = pair_image(text, &p, &record, (W1, W2), None);
    let wide = pair
        .first
        .iter()
        .filter(|g| g.exponents.values().any(|e| e > 255))
        .count();
    assert_eq!(wide, 1, "{pair:?}");
}

/// x^2 over x, y.
const X2: [u32; 2] = [2, 0];
/// x·y over x, y.
const XY: [u32; 2] = [1, 1];
/// y^2 over x, y.
const Y2: [u32; 2] = [0, 2];
/// x^300 over x, y, an exponent past a byte.
const X300: [u32; 2] = [300, 0];

// hygiene.19 design-checked -- this copy of `budget` stays, under the lift's deadline of `lift_params`; not the shared `budget`, which has none, and not one `fn` that takes the parameters (not measured)
/// A budget no merge here exhausts, under the lift's deadline.
fn budget() -> Budget {
    Budget::new(Instant::now(), &lift_params(1)).unwrap()
}

/// Test 4, q-image-exponents-peak.2, .6, .7: a part in bytes that takes a
/// term past 255 widens, and the image takes its merged terms by value; a
/// part in `u32` and an image in bytes of the same terms merge to no gain,
/// spending what the same merge into a part in bytes spends.
#[test]
fn a_merge_across_widths() {
    let named = MonomialOrder::Grevlex;
    let mut part = Part::new();
    let mut first = sparse(&[&X2, &XY, &Y2], &[3, 4, 5]);
    assert!(merge(&mut part, &mut first, None, &named, 2, &mut budget()).unwrap());
    append(&mut part, &first.coefs, &mut budget()).unwrap();
    let mut second = sparse(&[&X300, &X2, &Y2], &[6, 7, 8]);
    assert!(
        merge(&mut part, &mut second, None, &named, 2, &mut budget()).unwrap(),
        "x^300 is new to the part"
    );
    assert_eq!(
        second.exponents.values().collect::<Vec<u32>>(),
        [300, 0, 2, 0, 1, 1, 0, 2],
        "the image takes the part's terms"
    );
    assert_eq!(second.exponents.bytes(), 32, "four bytes an exponent");
    assert_eq!(second.coefs, [6, 7, 0, 8], "0 at the term the image lacked");
    let images: Vec<Vec<u64>> = part.images().map(<[u64]>::to_vec).collect();
    assert_eq!(images, [vec![0], vec![3], vec![4], vec![5]]);

    EXPONENTS_IN_WORDS.set(true);
    let mut words = Part::new();
    let mut image = sparse(&[&X2, &XY, &Y2], &[3, 4, 5]);
    assert!(merge(&mut words, &mut image, None, &named, 2, &mut budget()).unwrap());
    EXPONENTS_IN_WORDS.set(false);
    assert_eq!(image.exponents.bytes(), 24, "the part's terms in u32");
    let mut bytes = Part::new();
    let mut image = sparse(&[&X2, &XY, &Y2], &[3, 4, 5]);
    assert!(merge(&mut bytes, &mut image, None, &named, 2, &mut budget()).unwrap());

    let same = sparse(&[&X2, &XY, &Y2], &[9, 10, 11]);
    let (mut a, mut b) = (same.clone(), same.clone());
    let (mut spent_words, mut spent_bytes) = (budget(), budget());
    assert!(!merge(&mut words, &mut a, None, &named, 2, &mut spent_words).unwrap());
    assert!(!merge(&mut bytes, &mut b, None, &named, 2, &mut spent_bytes).unwrap());
    assert_eq!(a, same, "the image as it was");
    assert_eq!(a.exponents.bytes(), 6, "in bytes");
    assert_eq!(b, same);
    assert_eq!(spent_words.spent(), spent_bytes.spent(), "the same spend");
    assert!(spent_bytes.spent() >= 1, "the comparison spends");
}

/// Test 5, q-image-exponents-peak.7: under deadlines of 0 to 32 ms, with
/// bytes and with `u32`, a call over ℚ returns its full basis or stops at
/// the deadline with no basis.
#[test]
fn a_stop_in_the_lift() {
    for text in [random_sparse(4, 0, 1), WIDE[0].to_string()] {
        let full = call(&text, &reported(MonomialOrder::Grevlex), false)
            .basis
            .unwrap();
        for words in [false, true] {
            for ms in [0, 1, 2, 4, 8, 16, 32] {
                let mut params = reported(MonomialOrder::Grevlex);
                params.deadline = Some(Duration::from_millis(ms));
                let outcome = call(&text, &params, words);
                let name = format!("{ms} ms, words {words}, on\n{text}");
                match outcome.stopped {
                    Some(Stop::Deadline) => assert!(outcome.basis.is_none(), "{name}: {outcome:?}"),
                    None => assert_eq!(outcome.basis.unwrap(), full, "{name}"),
                    Some(Stop::Cancel | Stop::Memory) => panic!("{name}: {outcome:?}"),
                }
            }
        }
    }
}
