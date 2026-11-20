//! cofactor-stop-latency, the crate's test code: the compositions, the
//! lifts, a replay's cofactor image, the absorb's merge, and its append
//! spend units of work and stop inside their work, and a part's images are
//! appended in place (plan.md, _Test code_, 1 to 5).

use num_bigint::BigUint;
use rand_core::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::{CofactorRow, Cofactors, ENTRY, TERM, TERMS, Tracked};
use crate::field::{Field, Gf31};
use crate::halt::Halt;
use crate::id::{ElementId, KeepId, MonoId, RowId, RowSource};
use crate::lift::{COMBINED, LIMB, MERGED, Part, Primes, append, merge};
use crate::mixed_radix_tests::converted;
use crate::monomial::{Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::poly::{Exponents, Sparse};
use crate::rational::Crt;
use crate::shared_tests::{P, budget, passed_after, rng};
use crate::sort::{KEY, RUN};
use crate::table::{Table, monomial_units};
use crate::{MonomialOrder, Stop};

/// The variables of every table here.
const N: usize = 8;

/// The units a multiplier's interning into the cofactor table spends
/// (call-stop-latency.65).
const INTERNED: usize = monomial_units(N);

/// A table of `Dense16` on eight variables holding `count` distinct
/// monomials of degree 1 or more, returned in the order made.
fn monomials(count: usize, seed: u64) -> (Table<Dense16>, Vec<MonoId>) {
    let mut table = Table::new(Dense16::new(N), &mut rng(seed));
    let mut ids = Vec::with_capacity(count);
    'all: for a in 0..64u32 {
        for b in 0..64u32 {
            for c in 0..64u32 {
                if ids.len() == count {
                    break 'all;
                }
                ids.push(
                    table
                        .intern(&[a, b, c, 1, 0, 2, 0, 1], &mut Budget::unbounded())
                        .unwrap(),
                );
            }
        }
    }
    assert_eq!(ids.len(), count);
    (table, ids)
}

/// A coefficient of GF(65521) drawn in [1, p).
fn coef(field: &Gf31, r: &mut Xoshiro256PlusPlus) -> u32 {
    field.coef(1 + r.next_u64() % (P - 1))
}

/// A tracking of two generators over GF(65521), and the element of
/// cofactors (Σ `c_u·u`, 0): its first entry holds the terms `q` of the run's
/// table `table`, its second none.
fn element_of(
    field: Gf31,
    table: &Table<Dense16>,
    q: &[(MonoId, u32)],
    seed: u64,
) -> (Tracked<Gf31, Dense16>, CofactorRow<Gf31>) {
    let mut cof = Tracked::new(field, Dense16::new(N), 2, &mut rng(seed));
    let g0 = cof
        .input(0, field.coef(1), &mut Budget::unbounded())
        .expect("a budget without bounds never stops");
    let row = cof
        .lift(&[(0, q.to_vec())], &[g0], table, &mut budget())
        .unwrap();
    (cof, row)
}

/// The cofactor text of `elements`, written with no bound.
fn text_of(cof: &Tracked<Gf31, Dense16>, elements: &[CofactorRow<Gf31>]) -> String {
    let names: Vec<String> = (0..N).map(|i| format!("x{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut out = String::new();
    cof.write(
        &mut out,
        &names,
        &Grevlex,
        elements,
        &mut Budget::unbounded(),
    )
    .unwrap();
    out
}

/// The entry lengths of test 1 and 2.
fn lengths() -> [usize; 6] {
    [0, 1, TERMS - 1, TERMS, TERMS + 1, 3 * TERMS + 7]
}

/// Test 1, cofactor-stop-latency.1 to .3: a composition of one row returns
/// its element's cofactors, and spends `ENTRY` an entry and `TERM` a term
/// beside its multiplier's interning (call-stop-latency.65);
/// with the budget passed after one chunk it stops once an entry holds
/// `TERMS` terms or more: inside the entry where it holds more, at the next
/// entry where it holds `TERMS`; a row of 1,000 empty entries spends `ENTRY`
/// each beside the interning.
#[test]
fn a_composition_spends_its_terms() {
    let field = Gf31::prime(P);
    let (mut table, ids) = monomials(3 * TERMS + 7, 0x6373);
    // Row 0 of the matrix: element 0 times the monomial 1.
    let unit = table.intern(&[0; N], &mut Budget::unbounded()).unwrap();
    let rows = [RowSource {
        element: ElementId::at(0),
        multiplier: unit,
    }];
    let mut r = rng(1);
    for len in lengths() {
        let q: Vec<(MonoId, u32)> = ids[..len]
            .iter()
            .map(|&u| (u, coef(&field, &mut r)))
            .collect();
        let (mut cof, row) = element_of(field, &table, &q, 2);
        let want = text_of(&cof, std::slice::from_ref(&row));
        cof.push(row);

        cof.matrix(1);
        cof.start(RowId::at(0));
        cof.keep(&mut Budget::unbounded())
            .expect("a budget without bounds never stops");
        let mut b = budget();
        let got = cof.compose(KeepId::at(0), &rows, &table, &mut b).unwrap();
        assert_eq!(
            text_of(&cof, std::slice::from_ref(&got)),
            want,
            "{len} terms"
        );
        assert_eq!(
            b.spent(),
            (INTERNED + 2 * ENTRY + TERM * len) as u64,
            "{len} terms"
        );

        let units = (INTERNED + ENTRY + TERM * TERMS + 1) as u64;
        let mut b = passed_after(units);
        let stopped = cof.compose(KeepId::at(0), &rows, &table, &mut b);
        if len < TERMS {
            assert!(stopped.is_ok(), "{len} terms fit the units");
        } else {
            assert_eq!(
                stopped.err(),
                Some(Halt::Stop(Stop::Deadline)),
                "{len} terms"
            );
            assert!(
                b.spent() <= (INTERNED + 2 * ENTRY + 2 * TERM * TERMS) as u64,
                "{len} terms: {spent} units",
                spent = b.spent()
            );
        }
    }

    let mut wide = Tracked::new(field, Dense16::new(N), 1000, &mut rng(3));
    let zeros = wide.lift(&[], &[], &table, &mut budget()).unwrap();
    wide.push(zeros);
    wide.matrix(1);
    wide.start(RowId::at(0));
    wide.keep(&mut Budget::unbounded())
        .expect("a budget without bounds never stops");
    let mut b = budget();
    wide.compose(KeepId::at(0), &rows, &table, &mut b).unwrap();
    assert_eq!(b.spent(), (INTERNED + 1000 * ENTRY) as u64);
}

/// Test 2, cofactor-stop-latency.1 to .3: a lift by one quotient term m·c
/// returns m·c times its element's cofactors, and spends as a composition
/// does, its quotient's monomial interned as a composition's multiplier; with
/// the budget passed after one chunk it stops once an entry
/// holds `TERMS` terms or more: inside the entry where it holds more, at the
/// next entry where it holds `TERMS`.
#[test]
fn a_lift_spends_its_terms() {
    let field = Gf31::prime(P);
    let (mut table, ids) = monomials(3 * TERMS + 7, 0x6c66);
    let m = table
        .intern(&[1, 0, 0, 0, 1, 0, 0, 0], &mut Budget::unbounded())
        .unwrap();
    let c = field.coef(7);
    let mut r = rng(4);
    for len in lengths() {
        let q: Vec<(MonoId, u32)> = ids[..len]
            .iter()
            .map(|&u| (u, coef(&field, &mut r)))
            .collect();
        let (mut cof, row) = element_of(field, &table, &q, 5);
        // m·c·Σ c_u·u, the reference element, lifted from the input.
        let scaled: Vec<(MonoId, u32)> = q
            .iter()
            .map(|&(u, a)| {
                (
                    table.mul(m, u, &mut Budget::unbounded()).unwrap(),
                    field.mul(c, a),
                )
            })
            .collect();
        let g0 = cof
            .input(0, field.coef(1), &mut Budget::unbounded())
            .expect("a budget without bounds never stops");
        let want_row = cof
            .lift(&[(0, scaled)], &[g0], &table, &mut budget())
            .unwrap();
        let want = text_of(&cof, std::slice::from_ref(&want_row));

        let quotient = [(0, vec![(m, c)])];
        let mut b = budget();
        let got = cof
            .lift(&quotient, std::slice::from_ref(&row), &table, &mut b)
            .unwrap();
        assert_eq!(
            text_of(&cof, std::slice::from_ref(&got)),
            want,
            "{len} terms"
        );
        assert_eq!(
            b.spent(),
            (INTERNED + 2 * ENTRY + TERM * len) as u64,
            "{len} terms"
        );

        let units = (INTERNED + ENTRY + TERM * TERMS + 1) as u64;
        let mut b = passed_after(units);
        let stopped = cof.lift(&quotient, std::slice::from_ref(&row), &table, &mut b);
        if len < TERMS {
            assert!(stopped.is_ok(), "{len} terms fit the units");
        } else {
            assert_eq!(
                stopped.err(),
                Some(Halt::Stop(Stop::Deadline)),
                "{len} terms"
            );
            assert!(b.spent() <= (INTERNED + 2 * ENTRY + 2 * TERM * TERMS) as u64);
        }
    }
}

/// The reference image of one entry: its terms sorted by decreasing
/// monomial of `order` in the run's table, their exponents, and their
/// coefficients' values.
fn reference_entry<O: Order<Dense16>>(
    table: &Table<Dense16>,
    field: &Gf31,
    order: &O,
    terms: &[(MonoId, u32)],
) -> Sparse {
    let mut sorted = terms.to_vec();
    sorted.sort_unstable_by(|a, b| table.cmp(order, b.0, a.0));
    let mut sparse = Sparse {
        exponents: Exponents::new(),
        coefs: Vec::new(),
    };
    for (u, c) in sorted {
        sparse
            .exponents
            .extend((0..N).map(|i| table.exponent(u, i)));
        sparse.coefs.push(field.value(c));
    }
    sparse
}

/// `image` in `order` against the reference, its spends, and its stop.
fn images_as_sorted<O: Order<Dense16>>(
    order: &O,
    table: &Table<Dense16>,
    field: &Gf31,
    entries: &[[Vec<(MonoId, u32)>; 2]],
) {
    let mut cof = Tracked::new(*field, Dense16::new(N), 2, &mut rng(6));
    let inputs = [0, 1].map(|j| {
        cof.input(j, field.coef(1), &mut Budget::unbounded())
            .expect("a budget without bounds never stops")
    });
    let elements: Vec<CofactorRow<Gf31>> = entries
        .iter()
        .map(|[e0, e1]| {
            let quotients = [(0, e0.clone()), (1, e1.clone())];
            cof.lift(&quotients, &inputs, table, &mut budget()).unwrap()
        })
        .collect();
    let mut b = budget();
    let got = cof.image(order, &elements, &mut b).unwrap();
    let want: Vec<Vec<Sparse>> = entries
        .iter()
        .map(|es| {
            es.iter()
                .map(|e| reference_entry(table, field, order, e))
                .collect()
        })
        .collect();
    assert_eq!(got.len(), want.len(), "{order:?}");
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.len(), w.len(), "{order:?}");
        for (gs, ws) in g.iter().zip(w) {
            assert_eq!(gs.exponents, ws.exponents, "{order:?}");
            assert_eq!(gs.coefs, ws.coefs, "{order:?}");
        }
    }
    let terms: usize = entries.iter().flatten().map(Vec::len).sum();
    assert!(
        b.spent() >= (2 * entries.len() * ENTRY + terms * KEY) as u64,
        "{order:?}: {spent} units",
        spent = b.spent()
    );

    let mut b = passed_after(1);
    assert_eq!(
        cof.image(order, &elements, &mut b).err(),
        Some(Stop::Deadline)
    );
    assert_eq!(b.checks(), 1, "{order:?}");
}

/// Test 3, cofactor-stop-latency.1 and .4: a replay's cofactor image sorts
/// each entry by decreasing monomial in grevlex, lex, and a block order,
/// spends `ENTRY` an entry and `KEY` a term at least, and with the budget
/// passed stops at its first spend.
#[test]
#[allow(
    clippy::cast_possible_truncation,
    reason = "test code: the draw of a swap narrows to an index below the monomials' count"
)]
fn the_image_sorts_and_spends() {
    let field = Gf31::prime(P);
    let (table, ids) = monomials(RUN + 3, 0x696d);
    let mut r = rng(7);
    let mut shuffled = ids.clone();
    for i in (1..shuffled.len()).rev() {
        let j = (r.next_u64() % (i as u64 + 1)) as usize;
        shuffled.swap(i, j);
    }
    let long: Vec<(MonoId, u32)> = shuffled[..=RUN]
        .iter()
        .map(|&u| (u, coef(&field, &mut r)))
        .collect();
    let entries = [
        [long, Vec::new()],
        [
            vec![(shuffled[RUN + 1], coef(&field, &mut r))],
            vec![(shuffled[RUN + 2], coef(&field, &mut r))],
        ],
    ];
    images_as_sorted(&Grevlex, &table, &field, &entries);
    images_as_sorted(&Lex, &table, &field, &entries);
    images_as_sorted(&Block::new(&[3], N).unwrap(), &table, &field, &entries);
}

/// The exponents of `table`'s monomials `ids`, distinct, on eight
/// variables, in decreasing grevlex order.
fn decreasing(table: &Table<Dense16>, ids: &[MonoId]) -> Vec<Vec<u32>> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable_by(|&a, &b| table.cmp(&Grevlex, b, a));
    sorted
        .iter()
        .map(|&u| (0..N).map(|i| table.exponent(u, i)).collect())
        .collect()
}

/// An image of these terms, in the order given.
fn image_of(terms: &[(Vec<u32>, u64)]) -> Sparse {
    Sparse {
        exponents: terms.iter().flat_map(|t| t.0.clone()).collect(),
        coefs: terms.iter().map(|t| t.1).collect(),
    }
}

/// Test 4, cofactor-stop-latency.1 and .5: the merge into an empty part
/// gains every term with no image; a merge of equal exponents gains
/// nothing and spends; a merge that gains two terms, and lacks one where
/// the part holds two or more, holds the union in decreasing order, the old
/// images kept and the new 0, and
/// spends `MERGED` a term written at least; with the budget passed it
/// stops.
#[test]
fn the_merge_spends_and_merges() {
    let named = MonomialOrder::Grevlex;
    let count = 3 * COMBINED + 5;
    let (table, ids) = monomials(2 * count + 4, 0x6d67);
    let all = decreasing(&table, &ids);
    let mut r = rng(8);
    for n in [0, 1, count] {
        // The image's terms: every second monomial of `all`, n of them.
        let terms: Vec<(Vec<u32>, u64)> = all
            .iter()
            .step_by(2)
            .take(n)
            .map(|e| (e.clone(), 1 + r.next_u64() % (P - 1)))
            .collect();
        assert_eq!(terms.len(), n);
        let mut part = Part::new();
        let mut image = image_of(&terms);
        let gained = merge(&mut part, &mut image, None, &named, N, &mut budget()).unwrap();
        assert_eq!(gained, n > 0, "{n} terms");
        assert_eq!(part.len(), n);
        assert!(part.images().all(<[u64]>::is_empty), "no image yet");
        assert_eq!(image.coefs, terms.iter().map(|t| t.1).collect::<Vec<_>>());
        if n == 0 {
            continue;
        }

        let mut same = image_of(&terms);
        let mut b = budget();
        assert!(!merge(&mut part, &mut same, None, &named, N, &mut b).unwrap());
        assert!(b.spent() >= 1, "the comparison spends");

        // The images: the coefficients, at one prime.
        append(&mut part, &image.coefs, &mut budget()).unwrap();
        let old: Vec<Vec<u64>> = part.images().map(<[u64]>::to_vec).collect();
        assert_eq!(old, terms.iter().map(|t| vec![t.1]).collect::<Vec<_>>());

        // An image that lacks its second term, where it has two, and gains
        // `all[1]`, after its first term, and the last monomial of `all`,
        // past its last.
        let mut gains: Vec<(Vec<u32>, u64)> = terms.clone();
        let lacked = if n > 1 { Some(gains.remove(1)) } else { None };
        let new = [all[1].clone(), all[all.len() - 1].clone()];
        for e in &new {
            if !terms.iter().any(|t| t.0 == *e) {
                gains.push((e.clone(), 5));
            }
        }
        gains.sort_by(|a, b| {
            let pa = all.iter().position(|e| *e == a.0).unwrap();
            let pb = all.iter().position(|e| *e == b.0).unwrap();
            pa.cmp(&pb)
        });
        let mut image = image_of(&gains);
        let mut b = budget();
        assert!(merge(&mut part, &mut image, None, &named, N, &mut b).unwrap());
        let mut union: Vec<Vec<u32>> = terms.iter().map(|t| t.0.clone()).collect();
        for t in &gains {
            if !union.contains(&t.0) {
                union.push(t.0.clone());
            }
        }
        union.sort_by_key(|e| all.iter().position(|x| x == e).unwrap());
        assert_eq!(part.len(), union.len());
        assert_eq!(
            image.exponents.values().collect::<Vec<u32>>(),
            union.concat()
        );
        for ((e, x), &c) in union.iter().zip(part.images()).zip(&image.coefs) {
            match terms.iter().position(|t| t.0 == *e) {
                Some(k) => assert_eq!(x, &old[k][..], "an old term's images are kept"),
                None => assert_eq!(x, &[0][..], "a new term's images are 0"),
            }
            let given = gains.iter().find(|t| t.0 == *e).map_or(0, |t| t.1);
            assert_eq!(c, given, "0 where the image lacked the term");
        }
        if let Some(t) = &lacked {
            assert!(union.contains(&t.0));
        }
        assert!(
            b.spent() >= (MERGED * union.len()) as u64,
            "{spent} units",
            spent = b.spent()
        );

        let mut other = image_of(&terms[..n - 1]);
        let mut b = passed_after(1);
        assert_eq!(
            merge(&mut part, &mut other, None, &named, N, &mut b),
            Err(Stop::Deadline)
        );
    }
}

/// Test 5, cofactor-stop-latency.1 and .6 to .8, q-lift-growth-time.8 and
/// .10: a part's images appended in place at 3, 5, and four primes below
/// 2^62 are every image's coefficient at every prime through the
/// capacity's doublings, and convert to a residue below M that reduces to
/// each; the append spends `LIMB` a word written, the moves included; with
/// the budget passed it stops.
#[test]
fn the_append_writes_in_place() {
    let named = MonomialOrder::Grevlex;
    let count = 3 * COMBINED + 5;
    let (table, ids) = monomials(count, 0x636d);
    let all = decreasing(&table, &ids);
    for n in [0, 1, COMBINED, count] {
        let mut r = rng(9 + n as u64);
        let mut part = Part::new();
        let terms: Vec<(Vec<u32>, u64)> = all[..n].iter().map(|e| (e.clone(), 0)).collect();
        merge(
            &mut part,
            &mut image_of(&terms),
            None,
            &named,
            N,
            &mut budget(),
        )
        .unwrap();
        let mut crt = Crt::new();
        let mut primes = Primes::new(n as u64, vec![3, 5]);
        let mut images: Vec<(u64, Vec<u64>)> = Vec::new();
        for k in 0..6usize {
            let p = primes.wide();
            // Some coefficients 0, the others drawn below p.
            let coefs: Vec<u64> = (0..n)
                .map(|i| if i % 5 == 0 { 0 } else { r.next_u64() % p })
                .collect();
            let mut b = budget();
            append(&mut part, &coefs, &mut b).unwrap();
            // The capacity, 1 at first, doubles where k primes fill it.
            let moved = if k.is_power_of_two() { k } else { 0 };
            assert!(
                b.spent() >= (n * LIMB * (moved + 1)) as u64,
                "{n} terms at {k} primes: {spent} units",
                spent = b.spent()
            );
            let step = crt.step(p, 1);
            crt.push(step);
            images.push((p, coefs));
            let m = crt.modulus();
            assert_eq!(part.len(), n);
            for (i, x) in part.images().enumerate() {
                let want: Vec<u64> = images.iter().map(|(_, coefs)| coefs[i]).collect();
                assert_eq!(x, &want[..], "term {i} at {k} primes");
                let v = converted(&crt, x);
                assert!(v < *m, "below M");
                for (q, coefs) in &images {
                    assert_eq!(&v % *q, BigUint::from(coefs[i]), "at {q}");
                }
            }
        }
    }

    let mut part = Part::new();
    let terms: Vec<(Vec<u32>, u64)> = all.iter().map(|e| (e.clone(), 0)).collect();
    merge(
        &mut part,
        &mut image_of(&terms),
        None,
        &named,
        N,
        &mut budget(),
    )
    .unwrap();
    let coefs = vec![1; all.len()];
    let mut b = passed_after(1);
    assert_eq!(append(&mut part, &coefs, &mut b), Err(Stop::Deadline));
}
