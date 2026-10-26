//! text-stop-latency, the crate's test code: the texts spend units of work
//! as they are written and stop inside their work, the cofactor entries
//! sort in runs, and the cofactor text stops inside an entry (plan.md,
//! _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::{Cofactors, ENTRY, Tracked, sort_ids};
use crate::field::{Field, Gf31};
use crate::id::MonoId;
use crate::lift::{SCALE, scale_part};
use crate::matrix::RUN;
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::poly::Poly;
use crate::rational::{Rational, write_cofactor_q, write_poly_q};
use crate::table::Table;
use crate::text::{CHUNK, decimal_units, term_longest, write_basis, write_poly};
use crate::{MonomialOrder, Params, Stop, Threads};

const P: u64 = 65521;

fn params() -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    )
}

/// A budget with no bound.
fn generous() -> Budget {
    Budget::new(Instant::now(), &params()).unwrap()
}

/// A budget whose deadline passed before the test began, and which has not
/// been checked.
fn passed() -> Budget {
    let mut p = params();
    p.deadline = Some(Duration::from_millis(1));
    Budget::new(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        &p,
    )
    .unwrap()
}

/// `passed` with `units` of work between two full checks.
fn passed_after(units: u64) -> Budget {
    let mut b = passed();
    b.set_work(units);
    b
}

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// The text of a term with coefficient c on these names and exponents, in
/// the canonical spelling: the reference the writers are checked against.
fn term_text(names: &[&str], e: &[u32], c: &str) -> String {
    let monomial: Vec<String> = names
        .iter()
        .zip(e)
        .filter(|&(_, &k)| k > 0)
        .map(|(n, &k)| {
            if k == 1 {
                n.to_string()
            } else {
                format!("{n}^{k}")
            }
        })
        .collect();
    if monomial.is_empty() {
        return c.to_string();
    }
    if c == "1" {
        monomial.join("*")
    } else {
        format!("{c}*{monomial}", monomial = monomial.join("*"))
    }
}

/// The reference text of a polynomial over GF(p), its terms in the order
/// given: `0` where it has none.
fn poly_text(
    names: &[&str],
    table: &Table<Dense16>,
    field: &Gf31,
    terms: &[(MonoId, u32)],
) -> String {
    if terms.is_empty() {
        return "0".to_string();
    }
    let n = names.len();
    terms
        .iter()
        .map(|&(m, c)| {
            let e: Vec<u32> = (0..n).map(|i| table.exponent(m, i)).collect();
            term_text(names, &e, &field.value(c).to_string())
        })
        .collect::<Vec<_>>()
        .join("+")
}

fn terms_of(p: &Poly<Gf31, Rank1>) -> Vec<(MonoId, u32)> {
    p.monomials
        .iter()
        .copied()
        .zip(p.coefs.iter().copied())
        .collect()
}

/// The first `count` monomials x^a·y^b of degree 1 or more, by degree.
fn exponents(count: usize) -> Vec<[u32; 2]> {
    let mut out = Vec::new();
    let mut t = 1;
    while out.len() < count {
        for a in (0..=t).rev() {
            if out.len() < count {
                out.push([a, t - a]);
            }
        }
        t += 1;
    }
    out
}

/// A polynomial over GF(65521) of these monomials, each coefficient drawn
/// in [1, p).
fn poly(
    table: &mut Table<Dense16>,
    field: &Gf31,
    es: &[[u32; 2]],
    r: &mut Xoshiro256PlusPlus,
) -> Poly<Gf31, Rank1> {
    let mut terms: Vec<(MonoId, u32)> = es
        .iter()
        .map(|e| {
            (
                table.intern(e).unwrap(),
                field.coef(1 + r.next_u64() % (P - 1)),
            )
        })
        .collect();
    Poly::from_terms(&mut terms, field, table, &Grevlex, ())
}

/// Test 1, text-stop-latency.1 and .2: the basis text and a polynomial's
/// text over GF(p) are the reference's and spend their terms' bytes; with
/// the budget passed they stop inside the first long polynomial; a memory
/// ceiling is read at the same checks.
#[test]
fn the_texts_over_gfp_spend_their_bytes() {
    let field = Gf31::prime(P);
    let names = ["x", "y"];
    let mut table = Table::new(Dense16::new(2), &mut rng(1));
    let mut r = rng(2);
    let p1 = poly(&mut table, &field, &exponents(2 * CHUNK + 5), &mut r);
    let p2 = poly(&mut table, &field, &exponents(CHUNK), &mut r);
    let one_id = table.intern(&[0, 0]).unwrap();
    let one = Poly::from_terms(
        &mut vec![(one_id, field.coef(1))],
        &field,
        &table,
        &Grevlex,
        (),
    );
    assert_eq!((p1.len(), p2.len()), (2 * CHUNK + 5, CHUNK));
    let polys = [p1, p2, one];
    let texts: Vec<String> = polys
        .iter()
        .map(|p| poly_text(&names, &table, &field, &terms_of(p)))
        .collect();
    let want = format!("[{texts}]:\n", texts = texts.join(",\n"));

    let mut b = generous();
    let mut out = String::new();
    write_basis(&mut out, &names, &table, &field, polys.iter(), &mut b).unwrap();
    assert_eq!(out, want);
    let terms: usize = texts.iter().map(String::len).sum();
    assert_eq!(
        b.spent() as usize,
        terms,
        "the terms' bytes, no bracket or separator"
    );

    let mut b = generous();
    let mut out = String::new();
    write_poly(&mut out, &names, &table, &field, &polys[0], &mut b).unwrap();
    assert_eq!(out, texts[0]);
    assert_eq!(b.spent() as usize, out.len());

    let zero: Poly<Gf31, Rank1> = Poly {
        monomials: Vec::new(),
        coefs: Vec::new(),
        position: (),
    };
    let mut b = generous();
    let mut out = String::new();
    write_poly(&mut out, &names, &table, &field, &zero, &mut b).unwrap();
    assert_eq!((out.as_str(), b.spent()), ("0", 0));

    let bound = 4096 + CHUNK * term_longest(&names);
    let mut b = passed_after(4096);
    let mut out = String::new();
    assert_eq!(
        write_basis(&mut out, &names, &table, &field, polys.iter(), &mut b),
        Err(Stop::Deadline)
    );
    assert!(
        b.spent() as usize <= bound,
        "{spent} units",
        spent = b.spent()
    );
    assert!(out.len() < 1 + texts[0].len(), "the stop comes inside P1");
    let mut b = passed_after(4096);
    let mut out = String::new();
    assert_eq!(
        write_poly(&mut out, &names, &table, &field, &polys[0], &mut b),
        Err(Stop::Deadline)
    );
    assert!(b.spent() as usize <= bound && out.len() < texts[0].len());

    let mut p = params();
    p.memory = Some(1);
    let mut b = Budget::new(Instant::now(), &p).unwrap();
    b.set_work(4096);
    let mut out = String::new();
    assert_eq!(
        write_basis(&mut out, &names, &table, &field, polys.iter(), &mut b),
        Err(Stop::Memory)
    );
}

/// A number of `limbs` limbs, its top limb not 0.
fn number(limbs: usize, r: &mut Xoshiro256PlusPlus) -> BigUint {
    let mut v: Vec<u64> = (0..limbs).map(|_| r.next_u64()).collect();
    if let Some(top) = v.last_mut() {
        *top |= 1 << 63;
    }
    let mut x = BigUint::zero();
    for &limb in v.iter().rev() {
        x = (x << 64u32) + BigUint::from(limb);
    }
    x
}

/// The reference text over ℚ of these terms, (n, d) each with d None for 1,
/// one entry per term in order, with its sign or `+`; the terms of
/// coefficient 0 as empty entries.
fn terms_q(names: &[&str], exponents: &[u32], coefs: &[(BigInt, Option<BigUint>)]) -> Vec<String> {
    let n = names.len();
    let mut written = false;
    let mut out = Vec::new();
    for (k, (num, den)) in coefs.iter().enumerate() {
        if num.is_zero() {
            out.push(String::new());
            continue;
        }
        let sign = match num.sign() {
            Sign::Minus => "-",
            Sign::NoSign | Sign::Plus if written => "+",
            Sign::NoSign | Sign::Plus => "",
        };
        written = true;
        let e = &exponents[k * n..(k + 1) * n];
        let abs = num.abs().to_string();
        let c = match den {
            Some(d) => format!("{abs}/{d}"),
            None => abs,
        };
        let body = if e.iter().all(|&x| x == 0) || den.is_some() || c != "1" {
            if e.iter().all(|&x| x == 0) {
                c
            } else {
                format!("{c}*{term}", term = term_text(names, e, "1"))
            }
        } else {
            term_text(names, e, "1")
        };
        out.push(format!("{sign}{body}"));
    }
    out
}

/// Test 2, text-stop-latency.3 and .4: `decimal_units` is the square of the
/// limbs; the texts over ℚ are the reference's, and each term spends its
/// bytes and the square of its integers' limbs; with the budget passed the
/// text holds its first term alone; the scaling spends SCALE·(m² + 4) a
/// term and stops at its first.
#[test]
fn the_texts_over_q_spend_their_terms() {
    assert_eq!(decimal_units(&BigUint::zero()), 0);
    assert_eq!(decimal_units(&BigUint::one()), 1);
    assert_eq!(decimal_units(&BigUint::from(u64::MAX)), 1);
    assert_eq!(decimal_units(&(BigUint::one() << 64u32)), 4);
    assert_eq!(
        decimal_units(&((BigUint::one() << (64 * 300u32)) - 1u32)),
        90_000
    );

    let names = ["x", "y"];
    let exponents = [3, 0, 2, 1, 1, 1, 1, 0, 0, 1, 0, 0];
    let mut r = rng(3);
    let mut coefs: Vec<(BigInt, Option<BigUint>)> = Vec::new();
    for (k, limbs) in [1, 2, 0, 64, 65, 300].into_iter().enumerate() {
        let sign = if k % 2 == 1 { Sign::Minus } else { Sign::Plus };
        let num = BigInt::from_biguint(sign, number(limbs, &mut r));
        coefs.push((num, None));
    }
    let want = terms_q(&names, &exponents, &coefs);
    let ints: Vec<BigInt> = coefs.iter().map(|(n, _)| n.clone()).collect();
    let mut b = generous();
    let mut out = String::new();
    write_poly_q(&mut out, &names, &exponents, &ints, &mut b).unwrap();
    assert_eq!(out, want.concat());
    let units: usize = want
        .iter()
        .zip(&ints)
        .map(|(t, n)| {
            if n.is_zero() {
                0
            } else {
                t.len() + decimal_units(n.magnitude())
            }
        })
        .sum();
    assert_eq!(b.spent() as usize, units);
    let mut b = passed_after(1);
    let mut out = String::new();
    assert_eq!(
        write_poly_q(&mut out, &names, &exponents, &ints, &mut b),
        Err(Stop::Deadline)
    );
    assert_eq!(out, want[0]);

    let mut q: Vec<(BigInt, Option<BigUint>)> = Vec::new();
    // A denominator of 0 limbs stands for 1, which the text leaves out.
    for (k, (num, den)) in [(1, 1), (2, 2), (0, 1), (64, 65), (65, 0), (3, 2)]
        .into_iter()
        .enumerate()
    {
        let sign = if k % 2 == 0 { Sign::Minus } else { Sign::Plus };
        let n = BigInt::from_biguint(sign, number(num, &mut r));
        let d = number(den, &mut r);
        q.push((n, (!d.is_zero()).then_some(d)));
    }
    let want = terms_q(&names, &exponents, &q);
    let rats: Vec<Rational> = q
        .iter()
        .map(|(n, d)| Rational {
            num: n.clone(),
            den: d.clone().unwrap_or_else(BigUint::one),
        })
        .collect();
    let mut b = generous();
    let mut out = String::new();
    write_cofactor_q(&mut out, &names, &exponents, &rats, &mut b).unwrap();
    assert_eq!(out, want.concat());
    let units: usize = want
        .iter()
        .zip(&q)
        .map(|(t, (n, d))| {
            if n.is_zero() {
                0
            } else {
                t.len() + decimal_units(n.magnitude()) + d.as_ref().map_or(0, decimal_units)
            }
        })
        .sum();
    assert_eq!(b.spent() as usize, units);
    let mut b = passed_after(1);
    let mut out = String::new();
    assert_eq!(
        write_cofactor_q(&mut out, &names, &exponents, &rats, &mut b),
        Err(Stop::Deadline)
    );
    assert_eq!(out, want[0]);

    // num/dens[k] times l = 6 over the chain 1, 2, 6, and a term whose
    // denominator has seven limbs.
    let l = BigInt::from(6);
    let big = number(7, &mut r);
    let dens = vec![
        BigUint::from(1u32),
        BigUint::from(2u32),
        BigUint::from(6u32),
        big.clone(),
    ];
    let terms = vec![
        (BigInt::from(5), 0),
        (BigInt::from(-3), 1),
        (BigInt::from(7), 2),
        (BigInt::from(1), 3),
    ];
    let mut b = generous();
    let got = scale_part(&l, &terms, &dens, &mut b).unwrap();
    let want: Vec<Rational> = terms
        .iter()
        .map(|(n, k)| Rational::new(&l * n, BigInt::from(dens[*k as usize].clone())).unwrap())
        .collect();
    assert_eq!(got, want);
    assert_eq!(
        got[1],
        Rational::new(BigInt::from(-9), BigInt::one()).unwrap()
    );
    // m = 1 + 1 for the first three, and 7 for the last.
    assert_eq!(b.spent() as usize, SCALE * (3 * (4 + 4) + (49 + 4)));
    let mut b = passed_after(1);
    assert_eq!(scale_part(&l, &terms, &dens, &mut b), Err(Stop::Deadline));
    assert_eq!(b.spent() as usize, SCALE * 8, "the first term's spend");
}

/// `n` distinct monomials of `Dense16` on eight variables, of degree 24 or
/// less, in the random order drawn, in a table of their own.
fn monomials(n: usize, seed: u64) -> (Table<Dense16>, Vec<MonoId>) {
    let mut t = Table::new(Dense16::new(8), &mut rng(seed));
    let mut r = rng(seed ^ 0x6d6f);
    let mut taken = HashSet::new();
    let mut ids = Vec::with_capacity(n);
    while ids.len() < n {
        let mut e = [0u32; 8];
        let mut d = (r.next_u64() % 25) as u32;
        for x in &mut e {
            let k = (r.next_u64() % (u64::from(d) + 1)) as u32;
            *x = k;
            d -= k;
        }
        let id = t.intern(&e).unwrap();
        if taken.insert(id) {
            ids.push(id);
        }
    }
    (t, ids)
}

/// `n` distinct monomials on eight variables whose last five exponents are
/// 1 to 5 and whose first three sum to 442, in a random order: their
/// degrees agree, and so do the top 96 bits of their grevlex keys.
fn tied(n: usize, seed: u64) -> (Table<Dense16>, Vec<MonoId>) {
    let mut t = Table::new(Dense16::new(8), &mut rng(seed));
    let mut ids = Vec::new();
    'all: for a in 0..=442u32 {
        for b in 0..=442 - a {
            if ids.len() == n {
                break 'all;
            }
            let e = [a, b, 442 - a - b, 1, 2, 3, 4, 5];
            ids.push(t.intern(&e).unwrap());
        }
    }
    assert_eq!(ids.len(), n);
    let mut r = rng(seed ^ 0x7469);
    for i in (1..n).rev() {
        let j = (r.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    (t, ids)
}

/// `sort_ids` returns the comparison sort's order, and checks at least once
/// a run past `RUN`.
fn sorts_as_compared<O: Order<Dense16>>(order: &O, t: &Table<Dense16>, ids: &[MonoId]) {
    let mut b = generous();
    let at = sort_ids(t, order, ids, &mut b).unwrap();
    let got: Vec<MonoId> = at.iter().map(|&i| ids[i as usize]).collect();
    let mut want = ids.to_vec();
    want.sort_unstable_by(|&a, &b| t.cmp(order, b, a));
    assert_eq!(got, want, "{order:?}, {len} ids", len = ids.len());
    if ids.len() > RUN {
        assert!(
            b.checks() >= ids.len().div_ceil(RUN) as u64,
            "a check a run"
        );
    }
}

/// Test 3, text-stop-latency.6 to .9: the cofactor entries' sort returns the
/// comparison sort's order in grevlex, lex, and a block order, around
/// `RUN` and past it, and on a run of equal keys past `RUN`; with the
/// deadline passed it stops at its first spend in grevlex and at its first
/// run in lex.
#[test]
fn the_cofactor_entries_sort_in_runs() {
    let (t, ids) = monomials(3 * RUN + 7, 0x7473);
    let block = Block::new(&[3], 8).unwrap();
    for len in [0, 1, RUN - 1, RUN, RUN + 1, ids.len()] {
        sorts_as_compared(&Grevlex, &t, &ids[..len]);
        sorts_as_compared(&Lex, &t, &ids[..len]);
        sorts_as_compared(&block, &t, &ids[..len]);
    }
    let (tt, ties) = tied(3 * RUN + 7, 0x7469);
    let top = tt.key(&Grevlex, ties[0]) >> 32;
    assert!(
        ties.iter().all(|&u| tt.key(&Grevlex, u) >> 32 == top),
        "one run of equal keys"
    );
    sorts_as_compared(&Grevlex, &tt, &ties);

    let mut b = passed_after(1);
    assert_eq!(sort_ids(&t, &Grevlex, &ids, &mut b), Err(Stop::Deadline));
    assert_eq!(b.checks(), 1);
    let mut b = passed();
    assert_eq!(sort_ids(&t, &Lex, &ids, &mut b), Err(Stop::Deadline));
    assert_eq!(b.checks(), 1, "the first run's check");
}

/// Test 4, text-stop-latency.1, .5, and .6: a cofactor text with an entry of
/// 3·RUN + 7 terms is the reference's and spends ENTRY an entry at least;
/// with the budget passed it stops before the entry's text is complete; a
/// row of 1,000 zero entries spends ENTRY each.
#[test]
fn the_cofactor_text_stops_inside_an_entry() {
    let field = Gf31::prime(P);
    let (table, ids) = monomials(3 * RUN + 7, 0x636f);
    let names: Vec<String> = (0..8).map(|i| format!("x{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut cof = Tracked::new(field, Dense16::new(8), 2, &mut rng(4));
    let g0 = cof.input(0, field.coef(1));
    let mut r = rng(5);
    let q: Vec<(MonoId, u32)> = ids
        .iter()
        .map(|&u| (u, field.coef(1 + r.next_u64() % (P - 1))))
        .collect();
    let row = cof
        .lift(&[(0, q.clone())], &[g0], &table, &mut generous())
        .unwrap();
    let mut sorted = q;
    sorted.sort_unstable_by(|a, b| table.cmp(&Grevlex, b.0, a.0));
    let want = format!(
        "[[{poly},0]]:\n",
        poly = poly_text(&names, &table, &field, &sorted)
    );

    let mut b = generous();
    let mut out = String::new();
    cof.write(
        &mut out,
        &names,
        &Grevlex,
        std::slice::from_ref(&row),
        &mut b,
    )
    .unwrap();
    assert_eq!(out, want);
    assert!(b.spent() >= 2 * ENTRY as u64);

    let mut out = String::new();
    assert_eq!(
        cof.write(
            &mut out,
            &names,
            &Grevlex,
            std::slice::from_ref(&row),
            &mut passed_after(4096)
        ),
        Err(Stop::Deadline)
    );
    assert!(
        out.len() + 6 < want.len(),
        "the stop comes inside the entry"
    );

    let mut wide = Tracked::new(field, Dense16::new(8), 1000, &mut rng(6));
    let zeros = wide.lift(&[], &[], &table, &mut generous()).unwrap();
    let mut b = generous();
    let mut out = String::new();
    wide.write(
        &mut out,
        &names,
        &Grevlex,
        std::slice::from_ref(&zeros),
        &mut b,
    )
    .unwrap();
    assert_eq!(
        out,
        format!("[[{zeros}]]:\n", zeros = vec!["0"; 1000].join(","))
    );
    assert_eq!(b.spent(), 1000 * ENTRY as u64);
}
