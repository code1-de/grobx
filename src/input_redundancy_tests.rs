//! input-redundancy-time, the crate's test code: the generators of a run in
//! reduced row echelon form (`echelon::echelon`, input-redundancy-time.1 to
//! .6), and the early round's tests on a round that reduced no row to zero
//! (.7) (`iterations/0068-input-redundancy-time/plan.md`, _Test code_, 1
//! to 4, and the early round).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, draws, and residues mod p"
)]

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Write;
use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::echelon::{Echelon, echelon};
use crate::f4::{Halt, Handover};
use crate::field::{Field, Gf31, Gf31Pair, Gf64};
use crate::id::MonoId;
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::poly::Poly;
use crate::systems::Rng;
use crate::table::Table;
use crate::text::Header;
use crate::{Cancel, MonomialOrder, Params, Stop, Threads};

const P: u64 = 65521;
/// The largest prime below 2^31.
const P31: u64 = 2_147_483_647;
/// A second prime below 2^31.
const Q31: u64 = 2_147_483_629;
/// The Mersenne prime 2^61 − 1, past 2^31.
const P61: u64 = (1 << 61) - 1;

fn params() -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    )
}

pub(crate) fn budget() -> Budget {
    Budget::new(Instant::now(), &params()).unwrap()
}

/// A table for the variables `names`.
pub(crate) fn table(names: &str) -> Table<Dense16> {
    let n = names.split(',').count();
    Table::new(Dense16::new(n), &mut Xoshiro256PlusPlus::seed_from_u64(1))
}

/// The generators `generators` on the variables `names`, read over `field`
/// in `order` into `table`.
pub(crate) fn read<F: Field, O: Order<Dense16>>(
    field: &F,
    table: &mut Table<Dense16>,
    order: &O,
    names: &str,
    generators: &str,
) -> Vec<Poly<F, Rank1>> {
    let text = format!("{names}\n0\n{generators}\n");
    let header = Header::read(&text).unwrap();
    header.generators(field, table, order).unwrap()
}

/// `echelon` on the generators, which must succeed.
fn rows<F: Field, O: Order<Dense16>>(
    field: &F,
    table: &mut Table<Dense16>,
    order: &O,
    generators: Vec<Poly<F, Rank1>>,
) -> Echelon<F, Rank1> {
    echelon(field, table, order, generators, &mut budget())
        .unwrap_or_else(|h| panic!("the echelon halted: {h:?}"))
}

/// Test 1's check: the rows of `echelon` on `generators` over GF(p) in
/// `order` are those `want` writes, read into the same table, one per
/// generator, a zero row as `0`; `columns` are the generators' distinct
/// monomials.
fn check_case<O: Order<Dense16>>(
    case: &str,
    order: &O,
    names: &str,
    generators: &str,
    want: &str,
    columns: usize,
) {
    let field = Gf31::prime(P);
    let mut t = table(names);
    let g = read(&field, &mut t, order, names, generators);
    let count = g.len();
    let e = rows(&field, &mut t, order, g);
    assert_eq!(e.columns, columns, "{case}: the columns");
    assert_eq!(e.rows.len(), count, "{case}: one row per generator");
    let want = read(&field, &mut t, order, names, want);
    assert_eq!(e.rows.len(), want.len(), "{case}");
    for (k, (got, w)) in e.rows.iter().zip(&want).enumerate() {
        assert_eq!(
            (&got.monomials, &got.coefs),
            (&w.monomials, &w.coefs),
            "{case}: row {k}"
        );
    }
}

/// Test 1, input-redundancy-time.1 to .4: cases by hand over GF(65521).
#[test]
fn the_cases_by_hand() {
    // .4: the sparsest row leading at x is its pivot.
    check_case(
        "the sparsest",
        &Grevlex,
        "x,y,z,w",
        "x+y+z+w,\nx+y",
        "z+w,\nx+y",
        4,
    );
    // A tie goes to the least position; the back pass cancels y in row 0.
    check_case("a tie", &Grevlex, "x,y,z", "x+y,\nx+z", "x+z,\ny-z", 3);
    // A constant: the row 1, and no other row keeps a constant term.
    check_case(
        "a constant",
        &Grevlex,
        "x,y",
        "x+1,\nx+y,\n1",
        "x,\ny,\n1",
        3,
    );
    // .3: dependence leaves zero in the dependent generator's position.
    check_case("a multiple", &Grevlex, "x,y", "x+y,\n2*x+2*y", "x+y,\n0", 2);
    check_case(
        "a sum",
        &Grevlex,
        "x,y,z",
        "x+y,\ny+z,\nx+2*y+z",
        "x-z,\ny+z,\n0",
        3,
    );
    check_case("a duplicate", &Grevlex, "x,y", "x+y,\nx+y", "x+y,\n0", 2);
    check_case("no generator", &Grevlex, "x,y", "", "", 0);
    check_case("every generator zero", &Grevlex, "x,y", "0,\n0", "0,\n0", 0);
    check_case(
        "a zero among them",
        &Grevlex,
        "x,y,z",
        "x+y,\n0,\ny+z",
        "x-z,\n0,\ny+z",
        3,
    );
    check_case(
        "in echelon form already",
        &Grevlex,
        "x,y,z",
        "x+z,\ny+2*z",
        "x+z,\ny+2*z",
        3,
    );
    // Four linear forms led by s1, as on ccode-newton40t6, beside two of
    // degree 3 that share no monomial with them: the forms lead at s1 to
    // s4 with tails in s5, s6, and 1, and the others stay.
    check_case(
        "four linear forms led by s1",
        &Grevlex,
        "s1,s2,s3,s4,s5,s6",
        "s1+2*s2+3*s3+4*s4+5*s5+6*s6+7,\n\
         s1+3*s2+5*s3+7*s4+11*s5+13*s6+17,\n\
         s1+4*s2+9*s3+16*s4+25*s5+36*s6+49,\n\
         s1+8*s2+27*s3+64*s4+125*s5+216*s6+343,\n\
         s1^2*s6+s2*s3*s4+s5^3,\n\
         s2^2*s5+s3*s4*s6+s1*s5^2",
        "s1+65512*s5+65509*s6+65495,\n\
         s2+16*s5+27*s6+60,\n\
         s3+65507*s5+65493*s6+65460,\n\
         s4+6*s5+12*s6+24,\n\
         s1^2*s6+s2*s3*s4+s5^3,\n\
         s2^2*s5+s3*s4*s6+s1*s5^2",
        13,
    );
    // The leads follow the order: y^2 in grevlex, x in lex and block:1.
    let gens = "x+y^2,\ny^2+z";
    check_case("grevlex", &Grevlex, "x,y,z", gens, "y^2+z,\nx-z", 3);
    check_case("lex", &Lex, "x,y,z", gens, "x-z,\ny^2+z", 3);
    let block = Block::new(&[1], 3).unwrap();
    check_case("block:1", &block, "x,y,z", gens, "x-z,\ny^2+z", 3);
}

/// The rank of the vectors `rows` over GF(p), by Gaussian elimination on
/// their values, independent of the field's code.
fn rank(mut rows: Vec<Vec<u64>>, p: u64) -> usize {
    let mul = |a: u64, b: u64| (u128::from(a) * u128::from(b) % u128::from(p)) as u64;
    let inv = |a: u64| {
        let (mut b, mut e, mut r) = (a, p - 2, 1u64);
        while e > 0 {
            if e & 1 == 1 {
                r = mul(r, b);
            }
            b = mul(b, b);
            e >>= 1;
        }
        r
    };
    let width = rows.first().map_or(0, Vec::len);
    let mut r = 0;
    for c in 0..width {
        let Some(k) = (r..rows.len()).find(|&k| rows[k][c] != 0) else {
            continue;
        };
        rows.swap(r, k);
        let s = inv(rows[r][c]);
        for v in &mut rows[r] {
            *v = mul(*v, s);
        }
        let pivot = rows[r].clone();
        for (k, row) in rows.iter_mut().enumerate() {
            if k != r && row[c] != 0 {
                let a = row[c];
                for (v, &w) in row.iter_mut().zip(&pivot) {
                    *v = (*v + p - mul(a, w)) % p;
                }
            }
        }
        r += 1;
    }
    r
}

/// The polynomials `polys` as dense vectors of their coefficients' values
/// over the monomials `index` numbers.
fn dense<F: Field>(
    field: &F,
    polys: &[Poly<F, Rank1>],
    index: &HashMap<MonoId, usize>,
) -> Vec<Vec<u64>> {
    polys
        .iter()
        .map(|g| {
            let mut v = vec![0; index.len()];
            for (m, &c) in g.monomials.iter().zip(&g.coefs) {
                v[index[m]] = field.value(c);
            }
            v
        })
        .collect()
}

/// A random system on x1, …, x4 with linear redundancy: 12 sparse
/// generators of degree 2 or less, then the sum of two drawn among them, a
/// multiple of one by a scalar, a copy of one with a term added, a
/// duplicate, and a zero generator, shuffled.
fn redundant(p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(13) ^ 0x7ed);
    let monomials: Vec<String> = {
        let names = ["x1", "x2", "x3", "x4"];
        let mut m = vec![String::new()];
        for (i, a) in names.iter().enumerate() {
            m.push((*a).to_string());
            for b in &names[i..] {
                m.push(format!("{a}*{b}"));
            }
        }
        m
    };
    let coef = |rng: &mut Rng| 1 + rng.below(p - 1);
    let mut gens: Vec<Vec<(usize, u64)>> = Vec::new();
    for _ in 0..12 {
        let k = 1 + rng.below(5) as usize;
        let mut g: Vec<(usize, u64)> = Vec::new();
        for _ in 0..k {
            let m = rng.below(monomials.len() as u64) as usize;
            if g.iter().all(|&(n, _)| n != m) {
                g.push((m, coef(&mut rng)));
            }
        }
        gens.push(g);
    }
    let pick = |rng: &mut Rng, gens: &Vec<Vec<(usize, u64)>>| {
        gens[rng.below(gens.len() as u64) as usize].clone()
    };
    let (a, b) = (pick(&mut rng, &gens), pick(&mut rng, &gens));
    let mut sum: Vec<(usize, u64)> = a.clone();
    for (m, c) in b {
        match sum.iter_mut().find(|(n, _)| *n == m) {
            Some(t) => t.1 = ((u128::from(t.1) + u128::from(c)) % u128::from(p)) as u64,
            None => sum.push((m, c)),
        }
    }
    gens.push(sum);
    let s = coef(&mut rng);
    gens.push(
        pick(&mut rng, &gens)
            .into_iter()
            .map(|(m, c)| (m, (u128::from(c) * u128::from(s) % u128::from(p)) as u64))
            .collect(),
    );
    let mut shared = pick(&mut rng, &gens);
    shared.push((rng.below(monomials.len() as u64) as usize, coef(&mut rng)));
    gens.push(shared);
    gens.push(pick(&mut rng, &gens));
    gens.push(Vec::new());
    for k in (1..gens.len()).rev() {
        let j = rng.below(k as u64 + 1) as usize;
        gens.swap(k, j);
    }
    let mut text = String::new();
    for (k, g) in gens.iter().enumerate() {
        if k > 0 {
            text.push_str(",\n");
        }
        let mut terms = String::new();
        for &(m, c) in g {
            if c == 0 {
                continue;
            }
            if monomials[m].is_empty() {
                write!(terms, "+{c}").unwrap();
            } else {
                write!(terms, "+{c}*{monomial}", monomial = monomials[m]).unwrap();
            }
        }
        text.push_str(if terms.is_empty() { "0" } else { &terms[1..] });
    }
    text
}

/// Test 2's checks on the rows of `echelon` over `field` of the
/// generators `g`: one row per generator; every nonzero row monic; the
/// leads distinct, and no row holds another's lead; the nonzero rows as
/// many as the generators' rank, and the rank of the rows and the
/// generators together the same.
fn check_rows<F: Field>(
    case: &str,
    field: &F,
    p: u64,
    g: &[Poly<F, Rank1>],
    e: &Echelon<F, Rank1>,
) {
    assert_eq!(e.rows.len(), g.len(), "{case}");
    let one = field.coef(1);
    let mut leads = Vec::new();
    for (k, r) in e.rows.iter().enumerate() {
        if r.is_zero() {
            continue;
        }
        assert_eq!(r.coefs[0], one, "{case}: row {k} is not monic");
        leads.push(r.lead());
    }
    let mut sorted = leads.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), leads.len(), "{case}: two rows share a lead");
    for (k, r) in e.rows.iter().enumerate() {
        for m in r.monomials.iter().skip(1) {
            assert!(
                !leads.contains(m),
                "{case}: row {k} holds another row's lead"
            );
        }
    }
    let mut index = HashMap::new();
    for m in g.iter().chain(&e.rows).flat_map(|q| q.monomials.iter()) {
        let n = index.len();
        index.entry(*m).or_insert(n);
    }
    let gens = dense(field, g, &index);
    let rows = dense(field, &e.rows, &index);
    let r = rank(gens.clone(), p);
    assert_eq!(leads.len(), r, "{case}: the nonzero rows against the rank");
    assert_eq!(rank(rows.clone(), p), r, "{case}: the rows' rank");
    let both: Vec<Vec<u64>> = gens.into_iter().chain(rows).collect();
    assert_eq!(
        rank(both, p),
        r,
        "{case}: the rows leave the generators' span"
    );
}

/// Test 2, .1 to .4: the rows of random systems with linear redundancy
/// over 65521, 7, and 2^31 − 1 in grevlex and lex, and over 2^61 − 1 in
/// grevlex.
#[test]
fn random_systems_keep_their_span() {
    for seed in 0..24 {
        for p in [P, 7, P31] {
            let text = redundant(p, seed);
            let field = Gf31::prime(p);
            let mut t = table("x1,x2,x3,x4");
            let g = read(&field, &mut t, &Grevlex, "x1,x2,x3,x4", &text);
            let e = rows(&field, &mut t, &Grevlex, g.clone());
            check_rows(&format!("{p}, seed {seed}"), &field, p, &g, &e);
            let mut t = table("x1,x2,x3,x4");
            let g = read(&field, &mut t, &Lex, "x1,x2,x3,x4", &text);
            let e = rows(&field, &mut t, &Lex, g.clone());
            check_rows(&format!("{p} in lex, seed {seed}"), &field, p, &g, &e);
        }
        let text = redundant(P61, seed);
        let field = Gf64::prime(P61);
        let mut t = table("x1,x2,x3,x4");
        let g = read(&field, &mut t, &Grevlex, "x1,x2,x3,x4", &text);
        let e = rows(&field, &mut t, &Grevlex, g.clone());
        check_rows(&format!("2^61 - 1, seed {seed}"), &field, P61, &g, &e);
    }
}

/// Test 3: over a pair of primes, a pivot whose leading coefficient is 0
/// at one lane is a misfit; where nothing vanishes, each lane's rows are
/// those of the prime's own field.
#[test]
fn a_pair_of_primes() {
    let pair = Gf31Pair::primes(P31, Q31);
    let mut t = table("x,y,z");
    let g = read(
        &pair,
        &mut t,
        &Grevlex,
        "x,y,z",
        &format!("x+y,\n2*x+2*y+{P31}*z"),
    );
    assert_eq!(
        echelon(&pair, &mut t, &Grevlex, g, &mut budget()).err(),
        Some(Halt::Misfit(0))
    );
    let gens = "x+y,\n2*x+3*y+5*z,\nx+z,\n3*x+7*y^2+z";
    let mut t = table("x,y,z");
    let g = read(&pair, &mut t, &Grevlex, "x,y,z", gens);
    let e = rows(&pair, &mut t, &Grevlex, g);
    for (lane, p) in [P31, Q31].into_iter().enumerate() {
        let field = Gf31::prime(p);
        let one = read(&field, &mut t, &Grevlex, "x,y,z", gens);
        let e1 = rows(&field, &mut t, &Grevlex, one);
        assert_eq!(e.rows.len(), e1.rows.len());
        for (a, b) in e.rows.iter().zip(&e1.rows) {
            assert_eq!(a.monomials, b.monomials, "lane {lane}");
            let values: Vec<u64> = a.coefs.iter().map(|&c| pair.lane(c, lane)).collect();
            let want: Vec<u64> = b.coefs.iter().map(|&c| field.value(c)).collect();
            assert_eq!(values, want, "lane {lane}");
        }
    }
}

/// Test 4: a stop. 300 dense generators over 400 monomials of degree 5 or
/// less in six variables, with the budget's cancel raised: a stop; with it
/// lowered, the rows.
#[test]
fn a_stop_in_the_echelon() {
    let names = "a,b,c,d,e,f";
    let vars: Vec<&str> = names.split(',').collect();
    // The exponent vectors of degree 5 or less, the first 400 of them.
    let mut exponents: Vec<Vec<u32>> = vec![Vec::new()];
    for _ in 0..6 {
        exponents = exponents
            .into_iter()
            .flat_map(|e| {
                let used: u32 = e.iter().sum();
                (0..=5 - used).map(move |k| {
                    let mut f = e.clone();
                    f.push(k);
                    f
                })
            })
            .collect();
    }
    let monomials: Vec<String> = exponents
        .iter()
        .take(400)
        .map(|e| {
            let mut text = String::new();
            for (v, &k) in vars.iter().zip(e) {
                match k {
                    0 => {}
                    1 => write!(text, "*{v}").unwrap(),
                    k => write!(text, "*{v}^{k}").unwrap(),
                }
            }
            text
        })
        .collect();
    let mut rng = Rng::new(0x5709);
    let mut gens = Vec::new();
    for _ in 0..300 {
        let mut g = String::new();
        for m in &monomials {
            if rng.below(4) == 0 {
                write!(g, "+{coef}{m}", coef = 1 + rng.below(P - 1)).unwrap();
            }
        }
        gens.push(g[1..].to_string());
    }
    let text = gens.join(",\n");
    let field = Gf31::prime(P);
    let mut t = table(names);
    let g = read(&field, &mut t, &Grevlex, names, &text);
    let mut cancelled = params();
    let cancel = Cancel::new();
    cancel.set();
    cancelled.cancel = Some(cancel);
    let mut b = Budget::new(Instant::now(), &cancelled).unwrap();
    assert_eq!(
        echelon(&field, &mut t, &Grevlex, g.clone(), &mut b).err(),
        Some(Halt::Stop(Stop::Cancel))
    );
    let e = rows(&field, &mut t, &Grevlex, g.clone());
    check_rows("dense", &field, P, &g, &e);
}

/// input-redundancy-time.7: an early round of `EARLY_ROUND_ALONE`
/// elements and no zero, where the plain run would reduce each row alone,
/// hands over by the zero test, before the count; one of fewer elements
/// does not hand over.
#[test]
fn an_early_round_without_a_zero() {
    let calls = Cell::new(0);
    let count = || {
        calls.set(calls.get() + 1);
        0
    };
    assert!(Handover::new(false).after_round(true, false, 100, 0, count));
    assert_eq!(calls.get(), 0, "the zero test hands over before the count");
    assert!(!Handover::new(false).after_round(true, false, 99, 0, || 0));
    // minrank-ks10r3k15-p65521's early round in echelon form.
    assert!(Handover::new(false).after_round(true, false, 4165, 0, || 0));
    // An early round that is the gap round hands over.
    assert!(Handover::new(false).after_round(true, true, 100, 0, || 0));
    // A round that is no early round: the outgrown test alone.
    assert!(!Handover::new(false).after_round(false, false, 100, 0, || 0));
}
