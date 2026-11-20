//! pair-update-rest-time, sub-item 1, the crate's test code: the lanes'
//! sum by one multiplication, the word loops per word count, `G_min`'s
//! arrays with members leaving, and the kept pairs' scan
//! (`iterations/0107-pair-update-rest-time/plan.md`, _Test code_, 1, 2, 6,
//! and 7).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::divisibility_tests::{Engine, RefPair, Reference, degree, lcm, random_lanes, written};
use crate::monomial::{Dense8, Dense16, Dense32, Format, Overflow};
use crate::order::{Grevlex, Lex, Order};
use crate::pairs;
use crate::shared_tests::{below, coprime, divides, rng};
use crate::table::find_passing;

/// a / b, where b divides a.
fn quotient(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The variable counts of the tests: one to eleven words in `Dense8`, one
/// to nine in `Dense16`, and one to nine in `Dense32`, the word boundaries
/// on both sides.
const VARIABLES: [usize; 23] = [
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 40, 48, 55, 56, 57, 64, 65, 70,
];

/// The words of `e` in `format`, a monomial of the format.
fn encoded<M: Format>(format: &M, e: &[u32]) -> Vec<M::Word> {
    let mut w = vec![M::Word::default(); format.words()];
    format
        .encode(e, &mut w)
        .unwrap_or_else(|_| panic!("{e:?} within the format"));
    w
}

/// Test 1 for one format: on each count of `VARIABLES`, 200 random pairs
/// of monomials of degree at most `MAX_DEGREE`, their `lcm`'s degree or
/// Overflow, `lcm_degree`, `div`'s degree, and `mul_quotient`'s degree or
/// Overflow, each against the exponents summed; and the extremes.
fn check_degrees_on<M: Format>(seed: u64) {
    let mut rng = rng(seed);
    let top = M::MAX_DEGREE;
    for &n in &VARIABLES {
        let format = M::new(n);
        let mut out = vec![M::Word::default(); format.words()];
        for _ in 0..200 {
            let a = random_lanes(&mut rng, n, top, Some(top));
            let b = random_lanes(&mut rng, n, top, Some(top));
            let (wa, wb) = (encoded(&format, &a), encoded(&format, &b));
            let l = lcm(&a, &b);
            let dl = degree(&l);
            let case = format!("n {n}, {a:?} and {b:?}");
            // lcm: the degree exactly up to twice the bound, so Overflow
            // is told right.
            let got = format.lcm(&wa, &wb, &mut out);
            if dl <= top {
                assert_eq!(got, Ok(dl), "{case}: the lcm's degree");
                assert_eq!(out, encoded(&format, &l), "{case}: the lcm's words");
            } else {
                assert_eq!(got, Err(Overflow), "{case}: the lcm past the format");
            }
            assert_eq!(format.lcm_degree(&wa, &wb), dl, "{case}: lcm_degree");
            // div: the lcm by a, where the lcm fits.
            if dl <= top {
                let wl = encoded(&format, &l);
                let d = format.div(&wl, &wa, &mut out);
                assert_eq!(d, dl - degree(&a), "{case}: the quotient's degree");
                assert_eq!(
                    out,
                    encoded(&format, &quotient(&l, &a)),
                    "{case}: the quotient"
                );
            }
            // mul_quotient: s·lcm(x, y)/x, s a third monomial.
            let s = random_lanes(&mut rng, n, top, Some(top));
            let ws = encoded(&format, &s);
            let want: Vec<u32> = s
                .iter()
                .zip(&a)
                .zip(&b)
                .map(|((&s, &x), &y)| s + y.saturating_sub(x))
                .collect();
            let dw = degree(&want);
            let got = format.mul_quotient(&ws, &wa, &wb, &mut out);
            if dw <= top {
                assert_eq!(got, Ok(dw), "{case}, s {s:?}: mul_quotient's degree");
                assert_eq!(
                    out,
                    encoded(&format, &want),
                    "{case}, s {s:?}: mul_quotient"
                );
            } else {
                assert_eq!(
                    got,
                    Err(Overflow),
                    "{case}, s {s:?}: mul_quotient past the format"
                );
            }
        }
        // The extremes: one variable at the bound in each of two monomials
        // on different variables, whose lcm's lanes sum to twice the bound;
        // and the same variable, whose lcm is the monomial itself.
        if n >= 2 {
            let mut a = vec![0u32; n];
            let mut b = vec![0u32; n];
            a[0] = top;
            b[n - 1] = top;
            let (wa, wb) = (encoded(&format, &a), encoded(&format, &b));
            assert_eq!(
                format.lcm_degree(&wa, &wb),
                2 * top,
                "n {n}: twice the bound"
            );
            assert_eq!(
                format.lcm(&wa, &wb, &mut out),
                Err(Overflow),
                "n {n}: the lcm at twice the bound overflows"
            );
            assert_eq!(format.lcm_degree(&wa, &wa), top, "n {n}: the bound itself");
            assert_eq!(
                format.lcm(&wa, &wa, &mut out),
                Ok(top),
                "n {n}: the lcm of a with a"
            );
        }
    }
}

/// Test 1: pair-update-rest-time.2, in each format.
#[test]
fn the_lanes_sum_by_one_multiplication() {
    check_degrees_on::<Dense8>(0xa107_0001);
    check_degrees_on::<Dense16>(0xa107_0002);
    check_degrees_on::<Dense32>(0xa107_0003);
}

/// Test 2 for one format: on each count of `VARIABLES`, 300 random pairs:
/// `lcm`, `divides`, `coprime`, `lcm_degree`, `div`, `mul`, and
/// `is_product` against the references on exponent vectors.
fn check_loops_on<M: Format>(seed: u64) {
    let mut rng = rng(seed);
    let top = M::MAX_DEGREE;
    for &n in &VARIABLES {
        let format = M::new(n);
        let w = format.words();
        let mut out = vec![M::Word::default(); w];
        for _ in 0..300 {
            // Half the draws share the degree's room, so that products fit.
            let half = top / 2;
            let a = random_lanes(&mut rng, n, half, Some(half));
            let b = random_lanes(&mut rng, n, half, Some(half));
            let (wa, wb) = (encoded(&format, &a), encoded(&format, &b));
            let case = format!("n {n}, {a:?} and {b:?}");
            let l = lcm(&a, &b);
            assert_eq!(
                format.lcm(&wa, &wb, &mut out),
                Ok(degree(&l)),
                "{case}: lcm"
            );
            assert_eq!(out, encoded(&format, &l), "{case}: the lcm's words");
            assert_eq!(format.divides(&wa, &wb), divides(&a, &b), "{case}: a | b");
            assert_eq!(format.divides(&wb, &wa), divides(&b, &a), "{case}: b | a");
            assert!(format.divides(&wa, &out), "{case}: a | lcm");
            assert_eq!(format.coprime(&wa, &wb), coprime(&a, &b), "{case}: coprime");
            assert_eq!(
                format.lcm_degree(&wa, &wb),
                degree(&l),
                "{case}: lcm_degree"
            );
            let wl = out.clone();
            let d = format.div(&wl, &wb, &mut out);
            assert_eq!(d, degree(&l) - degree(&b), "{case}: div's degree");
            assert_eq!(out, encoded(&format, &quotient(&l, &b)), "{case}: div");
            let product: Vec<u32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            let got = format.mul(&wa, degree(&a), &wb, degree(&b), &mut out);
            assert_eq!(got, Ok(degree(&product)), "{case}: mul's degree");
            assert_eq!(out, encoded(&format, &product), "{case}: mul");
            assert!(format.is_product(&wa, &wb, &out), "{case}: is_product");
            // A product disturbed in one exponent is no product.
            let k = below(&mut rng, n as u64) as usize;
            let mut other = product.clone();
            other[k] += 1;
            if degree(&other) <= top {
                let wo = encoded(&format, &other);
                assert!(!format.is_product(&wa, &wb, &wo), "{case}: not a product");
            }
        }
    }
}

/// Test 2: pair-update-rest-time.3, in each format.
#[test]
fn the_word_loops_per_word_count() {
    check_loops_on::<Dense8>(0xa107_0011);
    check_loops_on::<Dense16>(0xa107_0012);
    check_loops_on::<Dense32>(0xa107_0013);
}

/// A lead that divides one of `leads`, drawn at random: a random subset
/// of its exponents, at least one lowered; None where `leads` is empty.
fn divisor_of(rng: &mut Xoshiro256PlusPlus, leads: &[Vec<u32>]) -> Option<Vec<u32>> {
    if leads.is_empty() {
        return None;
    }
    let base = &leads[below(rng, leads.len() as u64) as usize];
    let mut d = base.clone();
    let mut lowered = false;
    for x in &mut d {
        if *x > 0 && below(rng, 2) == 0 {
            *x -= 1 + below(rng, u64::from(*x)) as u32;
            lowered = true;
        }
    }
    if !lowered && let Some(x) = d.iter_mut().find(|x| **x > 0) {
        *x -= 1;
        lowered = true;
    }
    lowered.then_some(d)
}

/// Test 6 in one format and order: random leads entered one at a time
/// into the engine and the reference, every third one a divisor of an
/// earlier lead where one exists, so that members leave `G_min` and the
/// arrays are rebuilt, with selections between the updates: the same
/// pairs after each, and the arrays in `G_min`'s order.
fn check_leaving_members_on<M: Format, O: Order<M>>(order: &O, seed: u64, n: usize) {
    let mut rng = rng(seed);
    for world in 0..30 {
        let mut engine: Engine<M> = Engine::new(n, seed ^ world);
        let mut reference = Reference::new(O::GRADED, M::MAX_DEGREE);
        let mut leads: Vec<Vec<u32>> = Vec::new();
        let elements = 12 + below(&mut rng, 30) as usize;
        for e in 0..elements {
            let divisor = (e % 3 == 2).then(|| divisor_of(&mut rng, &leads)).flatten();
            let lead = divisor.unwrap_or_else(|| {
                let most = 2 + below(&mut rng, 6) as u32;
                random_lanes(&mut rng, n, 4, Some(most))
            });
            let sugar = degree(&lead) + below(&mut rng, 3) as u32;
            engine.enter(order, &lead, sugar).0.unwrap();
            reference.update(lead.clone(), sugar).unwrap();
            leads.push(lead.clone());
            let case = format!("world {world}, element {e}, lead {lead:?}");
            assert_eq!(engine.written(), reference.pairs, "{case}: the pairs");
            // The arrays read G_min in its order: the ids, leads, and masks
            // of the members, as the basis and the table hold them.
            let (ids, masks) = engine.min_arrays();
            let members = engine.min_members();
            assert_eq!(ids, members, "{case}: the arrays' members");
            let want: Vec<u32> = members.iter().map(|&i| engine.lead_mask(i)).collect();
            assert_eq!(masks, want, "{case}: the arrays' masks");
            if below(&mut rng, 4) == 0 {
                // The selection on an empty set takes nothing.
                let got = pairs::select(
                    &mut engine.pairs,
                    &mut engine.table,
                    order,
                    &mut Budget::unbounded(),
                )
                .unwrap();
                let want: Vec<RefPair> = match reference.pairs.iter().map(|p| p.sugar).min() {
                    None => Vec::new(),
                    Some(least) => {
                        let (taken, rest): (Vec<RefPair>, Vec<RefPair>) =
                            reference.pairs.drain(..).partition(|p| p.sugar == least);
                        reference.pairs = rest;
                        taken
                    }
                };
                assert_eq!(written(&engine.table, &got), want, "{case}: the selection");
            }
        }
    }
}

/// Test 6: pair-update-rest-time.4 and .5, in grevlex and lex, on 4 and
/// on 40 variables.
#[test]
fn members_leave_and_the_arrays_follow() {
    check_leaving_members_on::<Dense16, _>(&Grevlex, 0xa107_0061, 4);
    check_leaving_members_on::<Dense8, _>(&Lex, 0xa107_0062, 40);
    check_leaving_members_on::<Dense8, _>(&Grevlex, 0xa107_0063, 40);
}

/// Test 7: pair-update-rest-time.5. Random kept sets of 1 to 70 lcms
/// and a new pair's lcm: `find_passing` over the kept masks with the
/// divisibility test on the candidates gives the first kept lcm that
/// divides, as the scan one at a time does, on 3 and 40 variables.
#[test]
fn the_kept_pairs_scan_eight_at_a_time() {
    let mut rng = rng(0xa107_0071);
    for &n in &[3usize, 40] {
        let format = Dense8::new(n);
        for round in 0..200 {
            let count = 1 + below(&mut rng, 70) as usize;
            let kept: Vec<Vec<u32>> = (0..count)
                .map(|_| random_lanes(&mut rng, n, 3, Some(6)))
                .collect();
            let kept_words: Vec<Vec<u64>> = kept.iter().map(|e| encoded(&format, e)).collect();
            let kept_masks: Vec<u32> = kept_words.iter().map(|w| format.mask(w)).collect();
            // The new lcm: often a multiple of a kept one.
            let new = if below(&mut rng, 2) == 0 {
                let base = &kept[below(&mut rng, count as u64) as usize];
                let extra = random_lanes(&mut rng, n, 2, Some(3));
                base.iter().zip(&extra).map(|(x, y)| x + y).collect()
            } else {
                random_lanes(&mut rng, n, 3, Some(8))
            };
            let new_words = encoded(&format, &new);
            let mask = format.mask(&new_words);
            let want = kept.iter().position(|q| divides(q, &new));
            let got = find_passing(&kept_masks, mask, |q| {
                format.divides(&kept_words[q], &new_words)
            });
            assert_eq!(
                got, want,
                "n {n}, round {round}: kept {kept:?}, new {new:?}"
            );
        }
    }
}
