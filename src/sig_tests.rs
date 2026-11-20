//! signature-pruning, sub-item 1: the bookkeeping of signatures in `sig.rs`
//! against brute force on exponent vectors (plan.md, _Test code_).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::queue::Queue;
use crate::shared_tests::{
    SIG_INDICES, SIG_VARIABLES, below, divides, exponents, grevlex, id, minus, rng, sig_cmp,
    sig_divides, sig_table,
};
use crate::sig::{Sig, Signatures, cmp_ratio, pair};
use crate::table::{Limit, Table};

/// A random exponent vector of total degree at most `degree`.
fn random_exps(rng: &mut Xoshiro256PlusPlus, degree: u64) -> Vec<u32> {
    let mut left = below(rng, degree + 1);
    let mut e = vec![0u32; SIG_VARIABLES];
    for slot in &mut e {
        let x = below(rng, left + 1);
        *slot = x as u32;
        left -= x;
    }
    e
}

/// A random signature: monomial of degree at most `degree`, index below
/// `SIG_INDICES`.
fn random_sig(rng: &mut Xoshiro256PlusPlus, table: &mut Table<Dense16>, degree: u64) -> Sig {
    let e = random_exps(rng, degree);
    Sig {
        monomial: id(table, &e),
        index: below(rng, u64::from(SIG_INDICES)) as u32,
    }
}

/// A monic polynomial whose leading monomial has exponents `lead`, with up
/// to `tail` further terms below it.
fn random_poly(
    rng: &mut Xoshiro256PlusPlus,
    table: &mut Table<Dense16>,
    field: &Gf31,
    lead: &[u32],
    tail: u64,
) -> Poly<Gf31, Rank1> {
    let l = id(table, lead);
    let mut monomials = vec![l];
    for _ in 0..below(rng, tail + 1) {
        let d: u64 = lead.iter().map(|&x| u64::from(x)).sum();
        let m = id(table, &random_exps(rng, d));
        if table.cmp(&Grevlex, m, l).is_lt() && !monomials.contains(&m) {
            monomials.push(m);
        }
    }
    monomials.sort_by(|&a, &b| table.cmp(&Grevlex, b, a));
    let coefs = monomials
        .iter()
        .enumerate()
        .map(|(k, _)| field.coef(if k == 0 { 1 } else { 1 + below(rng, 1000) }))
        .collect();
    Poly {
        monomials,
        coefs,
        position: (),
    }
}

/// A basis of `count` random elements, each with a random signature, and
/// for each its (signature exponents, index, lead exponents, terms).
struct World {
    table: Table<Dense16>,
    basis: Basis<Gf31, Rank1>,
    sigs: Signatures,
    elements: Vec<(Vec<u32>, u32, Vec<u32>, usize)>,
}

fn world(seed: u64, count: usize) -> World {
    let mut rng = rng(seed);
    let mut table = sig_table(seed);
    let field = Gf31::new(65521).unwrap();
    let mut basis = Basis::new();
    let mut sigs = Signatures::new(SIG_INDICES as usize);
    let mut elements = Vec::new();
    for _ in 0..count {
        let lead = random_exps(&mut rng, 4);
        let poly = random_poly(&mut rng, &mut table, &field, &lead, 3);
        let terms = poly.len();
        let sig = random_sig(&mut rng, &mut table, 6);
        basis.push(poly, false).unwrap();
        sigs.push(&table, &Grevlex, &basis, sig, &mut Budget::unbounded())
            .unwrap();
        elements.push((exponents(&table, sig.monomial), sig.index, lead, terms));
    }
    World {
        table,
        basis,
        sigs,
        elements,
    }
}

/// The sig-lead ratio of element e, as a difference vector.
fn ratio(w: &World, e: usize) -> Vec<i64> {
    minus(&w.elements[e].0, &w.elements[e].2)
}

#[test]
fn signatures_compare_by_schreyers_order() {
    let mut rng = rng(1);
    let mut table = sig_table(1);
    for _ in 0..2000 {
        let a = random_sig(&mut rng, &mut table, 5);
        let b = random_sig(&mut rng, &mut table, 5);
        let (ea, eb) = (exponents(&table, a.monomial), exponents(&table, b.monomial));
        assert_eq!(
            a.cmp(b, &table, &Grevlex),
            sig_cmp((&ea, a.index), (&eb, b.index)),
            "{ea:?}·e{ia} against {eb:?}·e{ib}",
            ia = a.index,
            ib = b.index
        );
        assert_eq!(
            a.divides(b, &table),
            sig_divides((&ea, a.index), (&eb, b.index)),
            "{ea:?}·e{ia} | {eb:?}·e{ib}",
            ia = a.index,
            ib = b.index
        );
    }
}

#[test]
fn a_signature_times_a_monomial() {
    let mut rng = rng(2);
    let mut table = sig_table(2);
    for _ in 0..500 {
        let s = random_sig(&mut rng, &mut table, 5);
        let te = random_exps(&mut rng, 4);
        let t = id(&mut table, &te);
        let p = s.mul(t, &mut table, &mut Budget::unbounded()).unwrap();
        let want: Vec<u32> = exponents(&table, s.monomial)
            .iter()
            .zip(&te)
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(exponents(&table, p.monomial), want);
        assert_eq!(p.index, s.index);
        assert_eq!(p.degree(&table), want.iter().sum::<u32>());
    }
    let big = Sig {
        monomial: id(&mut table, &[20000, 0, 0]),
        index: 0,
    };
    let t = id(&mut table, &[0, 15000, 0]);
    assert_eq!(
        big.mul(t, &mut table, &mut Budget::unbounded()),
        Err(Limit::Overflow)
    );
}

#[test]
fn ratios_compare_as_difference_vectors() {
    let mut rng = rng(3);
    let mut table = sig_table(3);
    for _ in 0..2000 {
        let v: Vec<Vec<u32>> = (0..4).map(|_| random_exps(&mut rng, 5)).collect();
        let ids: Vec<MonoId> = v.iter().map(|e| id(&mut table, e)).collect();
        let got = cmp_ratio(&table, &Grevlex, (ids[0], ids[1]), (ids[2], ids[3]));
        let want = grevlex(&minus(&v[0], &v[1]), &minus(&v[2], &v[3]));
        assert_eq!(got, want, "{v:?}");
    }
    // s_a·l_b passes the format's degree bound, and its lanes do not.
    let s = id(&mut table, &[20000, 0, 0]);
    let l = id(&mut table, &[0, 15000, 0]);
    let one = id(&mut table, &[0, 0, 0]);
    assert_eq!(
        cmp_ratio(&table, &Grevlex, (s, one), (one, l)),
        Ordering::Greater
    );
}

#[test]
fn the_reductant_has_the_largest_ratio() {
    for seed in 0..20 {
        let mut w = world(seed, 40);
        let mut rng = rng(seed ^ 0xa11);
        for _ in 0..200 {
            let sig = random_sig(&mut rng, &mut w.table, 9);
            let se = exponents(&w.table, sig.monomial);
            // Of the elements whose signature divides sig, the largest
            // ratio, ties to the larger signature.
            let mut want: Option<usize> = None;
            for e in 0..w.elements.len() {
                let (s, i, _, _) = &w.elements[e];
                if !sig_divides((s, *i), (&se, sig.index)) {
                    continue;
                }
                let better = match want {
                    None => true,
                    Some(b) => match grevlex(&ratio(&w, e), &ratio(&w, b)) {
                        Ordering::Greater => true,
                        Ordering::Less => false,
                        Ordering::Equal => {
                            let (sb, ib, _, _) = &w.elements[b];
                            sig_cmp((s, *i), (sb, *ib)).is_gt()
                        }
                    },
                };
                if better {
                    want = Some(e);
                }
            }
            let got = w
                .sigs
                .reductant(&w.table, &Grevlex, sig, &mut Budget::unbounded())
                .unwrap()
                .map(ElementId::index);
            match (got, want) {
                (Some(g), Some(b)) => assert!(
                    g == b
                        || (grevlex(&ratio(&w, g), &ratio(&w, b)).is_eq()
                            && w.elements[g].0 == w.elements[b].0
                            && w.elements[g].1 == w.elements[b].1),
                    "seed {seed}: {se:?}·e{index}: got {g}, want {b}",
                    index = sig.index
                ),
                (Some(_), None) | (None, Some(_) | None) => {
                    assert_eq!(got, want, "seed {seed}: {se:?}·e{index}", index = sig.index);
                }
            }
        }
    }
}

#[test]
fn the_reducer_has_the_least_signature() {
    for seed in 0..20 {
        let mut w = world(seed, 40);
        let mut rng = rng(seed ^ 0xb22);
        // The order of reducers: (ratio, index, terms, element).
        let key = |w: &World, e: usize| (ratio(w, e), w.elements[e].1, w.elements[e].3, e);
        let less = |w: &World, a: usize, b: usize| {
            let (ka, kb) = (key(w, a), key(w, b));
            grevlex(&ka.0, &kb.0)
                .then(ka.1.cmp(&kb.1))
                .then(ka.2.cmp(&kb.2))
                .then(ka.3.cmp(&kb.3))
                .is_lt()
        };
        for _ in 0..200 {
            let me = random_exps(&mut rng, 7);
            let m = id(&mut w.table, &me);
            let mut want: Option<usize> = None;
            for e in 0..w.elements.len() {
                if divides(&w.elements[e].2, &me) && want.is_none_or(|b| less(&w, e, b)) {
                    want = Some(e);
                }
            }
            let got = w.sigs.reducer(&w.table, m).0.map(ElementId::index);
            assert_eq!(got, want, "seed {seed}: {me:?}");
        }
        for a in 0..w.elements.len() {
            for b in 0..w.elements.len() {
                let want = grevlex(&ratio(&w, a), &ratio(&w, b))
                    .then(w.elements[a].1.cmp(&w.elements[b].1))
                    .is_lt();
                assert_eq!(
                    w.sigs.before(ElementId::at(a), ElementId::at(b)),
                    want,
                    "seed {seed}: {a} before {b}"
                );
            }
        }
    }
}

#[test]
fn syzygy_signatures_are_known_with_their_multiples() {
    for seed in 0..20 {
        let mut rng = rng(seed ^ 0xc33);
        let mut table = sig_table(seed);
        let mut sigs = Signatures::new(SIG_INDICES as usize);
        let mut pushed: Vec<(Vec<u32>, u32)> = Vec::new();
        for _ in 0..300 {
            let s = random_sig(&mut rng, &mut table, 6);
            let se = exponents(&table, s.monomial);
            let known = pushed
                .iter()
                .any(|(e, i)| sig_divides((e, *i), (&se, s.index)));
            assert_eq!(
                sigs.is_syzygy(&mut table, s, &mut Budget::unbounded())
                    .unwrap(),
                known,
                "seed {seed}: {se:?}"
            );
            if below(&mut rng, 3) == 0 {
                assert_eq!(
                    sigs.push_syzygy(&mut table, s, &mut Budget::unbounded())
                        .unwrap(),
                    !known,
                    "seed {seed}: {se:?}"
                );
                pushed.push((se, s.index));
                assert!(
                    sigs.is_syzygy(&mut table, s, &mut Budget::unbounded())
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn the_pair_has_the_larger_multiple_signature() {
    for seed in 0..20 {
        let mut w = world(seed, 30);
        for g in 0..w.elements.len() {
            for h in 0..w.elements.len() {
                if g == h {
                    continue;
                }
                let (sg, ig, lg, _) = w.elements[g].clone();
                let (sh, ih, lh, _) = w.elements[h].clone();
                let lcm: Vec<u32> = lg.iter().zip(&lh).map(|(a, b)| *a.max(b)).collect();
                let mg: Vec<u32> = (0..SIG_VARIABLES).map(|k| lcm[k] - lg[k] + sg[k]).collect();
                let mh: Vec<u32> = (0..SIG_VARIABLES).map(|k| lcm[k] - lh[k] + sh[k]).collect();
                let coprime = lg.iter().zip(&lh).all(|(a, b)| *a == 0 || *b == 0);
                let want = match sig_cmp((&mg, ig), (&mh, ih)) {
                    Ordering::Equal => None,
                    Ordering::Greater => Some((mg, ig, coprime)),
                    Ordering::Less => Some((mh, ih, coprime)),
                };
                let (ge, he) = (ElementId::at(g), ElementId::at(h));
                let got = pair(&mut w.table, &w.basis, &w.sigs, ge, he)
                    .unwrap()
                    .map(|f| {
                        let format = *w.table.format();
                        let formed: Vec<u32> = (0..SIG_VARIABLES)
                            .map(|k| format.exponent(w.table.formed(), k))
                            .collect();
                        let (lg, lh) = (w.basis.lead(ge), w.basis.lead(he));
                        (formed, f.index, w.table.coprime(lg, lh))
                    });
                assert_eq!(got, want, "seed {seed}: pair {g}, {h}");
            }
        }
    }
}

#[test]
fn the_queue_keeps_every_pending_signature_once() {
    for seed in 0..30 {
        let mut rng = rng(seed ^ 0xd44);
        let mut table = sig_table(seed);
        let mut queue = Queue::new();
        let mut model: Vec<(Vec<u32>, u32)> = Vec::new();
        for _ in 0..400 {
            if below(&mut rng, 5) < 4 {
                let s = random_sig(&mut rng, &mut table, 6);
                let se = exponents(&table, s.monomial);
                queue.push(&table, s);
                if !model.contains(&(se.clone(), s.index)) {
                    model.push((se, s.index));
                }
            } else {
                let got: Vec<(Vec<u32>, u32)> = queue
                    .pop_least(&table, &Grevlex, &mut Budget::unbounded())
                    .unwrap()
                    .iter()
                    .map(|s| (exponents(&table, s.monomial), s.index))
                    .collect();
                let mut want: Vec<(Vec<u32>, u32)> = Vec::new();
                if let Some(d) = model.iter().map(|(e, _)| e.iter().sum::<u32>()).min() {
                    want = model
                        .iter()
                        .filter(|(e, _)| e.iter().sum::<u32>() == d)
                        .cloned()
                        .collect();
                    model.retain(|(e, _)| e.iter().sum::<u32>() != d);
                    want.sort_by(|a, b| sig_cmp((&a.0, a.1), (&b.0, b.1)));
                }
                assert_eq!(got, want, "seed {seed}");
            }
            assert_eq!(queue.len(), model.len(), "seed {seed}");
            assert_eq!(queue.is_empty(), model.is_empty());
        }
    }
}
