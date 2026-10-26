//! run-generators-time, sub-item 2's test code: a replay's generators in
//! reduced row echelon form from its record's leading monomials
//! (`echelon::replay`, run-generators-time.5 to .8)
//! (`iterations/0087-run-generators-time/plan.md`, _Test code_, 4 to 6).
//! Where the record fits, the replay gives `echelon`'s rows; where it does
//! not, `Halt::Misfit(0)`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::fmt::Write;

use crate::echelon::{Echelon, echelon, replay};
use crate::f4::Halt;
use crate::field::{Field, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::input_redundancy_tests::{budget, read, table};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Grevlex, Lex, Order};
use crate::systems::Rng;
use crate::table::Table;

const P: u64 = 65521;
/// The largest prime below 2^31.
const P31: u64 = 2_147_483_647;
/// The Mersenne prime 2^61 − 1.
const P61: u64 = (1 << 61) - 1;
/// Two primes of [2^30, 2^31), a pair learn's.
const L0: u64 = 1_945_203_761;
const L1: u64 = 1_875_939_437;
/// Two primes of [2^61, 2^62), a pair replay's.
const R0: u64 = 2_536_775_269_993_504_451;
const R1: u64 = 4_026_501_963_682_786_037;

/// A record's leading monomials of `echelon`'s rows: per row the exponents
/// of its first monomial, None for a zero row.
fn leads<F: Field>(table: &Table<Dense16>, e: &Echelon<F, Rank1>) -> Vec<Option<Vec<u32>>> {
    let n = table.format().variables();
    e.rows
        .iter()
        .map(|g| {
            (!g.is_zero()).then(|| {
                let mut x = vec![0; n];
                table.unpack(g.lead(), &mut x);
                x
            })
        })
        .collect()
}

/// `echelon` over `field` of `generators` on `names` in `order`, with the
/// table it read them into; None where it halts.
fn rows<F: Field, O: Order<Dense16>>(
    field: &F,
    order: &O,
    names: &str,
    generators: &str,
) -> Option<(Table<Dense16>, Echelon<F, Rank1>)> {
    let mut t = table(names);
    let g = read(field, &mut t, order, names, generators);
    let e = echelon(field, &mut t, order, g, &mut budget()).ok()?;
    Some((t, e))
}

/// Tests 4 and 6's check: where `echelon` over A and over B give the same
/// leads, `replay` over B with A's leads gives `echelon`'s rows over B and
/// its columns. Whether the leads agreed and the check ran.
fn agrees<A: Field, B: Field, O: Order<Dense16>>(
    case: &str,
    a: &A,
    b: &B,
    order: &O,
    names: &str,
    generators: &str,
) -> bool {
    let Some((ta, ea)) = rows(a, order, names, generators) else {
        return false;
    };
    let Some((tb, eb)) = rows(b, order, names, generators) else {
        return false;
    };
    let want = leads(&ta, &ea);
    if want != leads(&tb, &eb) {
        return false;
    }
    let mut t = table(names);
    let g = read(b, &mut t, order, names, generators);
    let got = replay(b, &mut t, order, g, &want, &mut budget())
        .unwrap_or_else(|h| panic!("{case}: the replay halted: {h:?}\n{generators}"));
    assert_eq!(got.columns, eb.columns, "{case}: the columns");
    assert_eq!(
        got.rows.len(),
        eb.rows.len(),
        "{case}: one row per generator"
    );
    for (k, (r, w)) in got.rows.iter().zip(&eb.rows).enumerate() {
        assert_eq!(
            r.coefs, w.coefs,
            "{case}: row {k}'s coefficients\n{generators}"
        );
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (&u, &v) in r.monomials.iter().zip(&w.monomials) {
            x.clear();
            y.clear();
            t.exponents(u, &mut x);
            tb.exponents(v, &mut y);
            assert_eq!(x, y, "{case}: row {k}'s monomials\n{generators}");
        }
        assert_eq!(
            r.monomials.len(),
            w.monomials.len(),
            "{case}: row {k}'s terms"
        );
    }
    true
}

/// A drawn system on x1, …, xn with linear redundancy over every prime of
/// the test: 4 to 9 generators of up to 6 terms of degree at most 2, with
/// coefficients below 1000 and a sign; then the sum of two, 3 times one,
/// one with a term added, a duplicate, and a zero generator, shuffled.
fn redundant(rng: &mut Rng, n: usize) -> String {
    let mut monomials: Vec<Vec<u32>> = vec![vec![0; n]];
    for i in 0..n {
        let mut e = vec![0; n];
        e[i] = 1;
        monomials.push(e.clone());
        for j in i..n {
            let mut f = e.clone();
            f[j] += 1;
            monomials.push(f);
        }
    }
    let coef = |rng: &mut Rng| {
        let c = 1 + rng.below(999) as i64;
        if rng.below(2) == 1 { -c } else { c }
    };
    let mut gens: Vec<Vec<(usize, i64)>> = Vec::new();
    for _ in 0..4 + rng.below(6) {
        let mut g: Vec<(usize, i64)> = Vec::new();
        for _ in 0..=rng.below(6) {
            let m = rng.below(monomials.len() as u64) as usize;
            if g.iter().all(|&(k, _)| k != m) {
                g.push((m, coef(rng)));
            }
        }
        gens.push(g);
    }
    let pick = |rng: &mut Rng, gens: &Vec<Vec<(usize, i64)>>| {
        gens[rng.below(gens.len() as u64) as usize].clone()
    };
    let (x, y) = (pick(rng, &gens), pick(rng, &gens));
    let mut sum = x.clone();
    for (m, c) in y {
        match sum.iter_mut().find(|(k, _)| *k == m) {
            Some(t) => t.1 += c,
            None => sum.push((m, c)),
        }
    }
    sum.retain(|&(_, c)| c != 0);
    gens.push(sum);
    let triple = pick(rng, &gens)
        .into_iter()
        .map(|(m, c)| (m, 3 * c))
        .collect();
    gens.push(triple);
    let mut more = pick(rng, &gens);
    let m = rng.below(monomials.len() as u64) as usize;
    if more.iter().all(|&(k, _)| k != m) {
        more.push((m, coef(rng)));
    }
    gens.push(more);
    let copy = pick(rng, &gens);
    gens.push(copy);
    gens.push(Vec::new());
    for k in (1..gens.len()).rev() {
        let j = rng.below(k as u64 + 1) as usize;
        gens.swap(k, j);
    }
    write_out(n, &monomials, &gens)
}

/// A drawn sparse linear system: 2 to 40 generators over 3 to 30
/// monomials of x1, …, x4 of degree at most 3, each of 1 to 5 terms with
/// coefficients below 1000 and a sign.
fn sparse(rng: &mut Rng) -> String {
    let n = 4;
    let mut all: Vec<Vec<u32>> = Vec::new();
    for a in 0..=3u32 {
        for b in 0..=3 - a {
            for c in 0..=3 - a - b {
                for d in 0..=3 - a - b - c {
                    all.push(vec![a, b, c, d]);
                }
            }
        }
    }
    let k = 3 + rng.below(28) as usize;
    let mut monomials = Vec::new();
    for _ in 0..k {
        monomials.push(all[rng.below(all.len() as u64) as usize].clone());
    }
    let mut gens = Vec::new();
    for _ in 0..2 + rng.below(39) {
        let mut g: Vec<(usize, i64)> = Vec::new();
        for _ in 0..=rng.below(5) {
            let m = rng.below(monomials.len() as u64) as usize;
            if g.iter().all(|&(j, _)| monomials[j] != monomials[m]) {
                let c = 1 + rng.below(999) as i64;
                g.push((m, if rng.below(2) == 1 { -c } else { c }));
            }
        }
        gens.push(g);
    }
    write_out(n, &monomials, &gens)
}

/// A drawn system over 70 to 252 monomials of x1, …, x5 of degree at most
/// 5, each in some generator: 8 to 40 generators, each of up to 40 more
/// terms, and 2 to 6 of 2 or 3 terms, with coefficients below 1000 and a
/// sign; then the sum of two and a multiple of one, shuffled.
fn wide(rng: &mut Rng) -> String {
    let n = 5;
    let mut all: Vec<Vec<u32>> = Vec::new();
    for a in 0..=5u32 {
        for b in 0..=5 - a {
            for c in 0..=5 - a - b {
                for d in 0..=5 - a - b - c {
                    for e in 0..=5 - a - b - c - d {
                        all.push(vec![a, b, c, d, e]);
                    }
                }
            }
        }
    }
    for k in (1..all.len()).rev() {
        let j = rng.below(k as u64 + 1) as usize;
        all.swap(k, j);
    }
    all.truncate(70 + rng.below(all.len() as u64 - 69) as usize);
    let coef = |rng: &mut Rng| {
        let c = 1 + rng.below(999) as i64;
        if rng.below(2) == 1 { -c } else { c }
    };
    let mut gens: Vec<Vec<(usize, i64)>> = vec![Vec::new(); 8 + rng.below(33) as usize];
    for m in 0..all.len() {
        let k = rng.below(gens.len() as u64) as usize;
        let c = coef(rng);
        gens[k].push((m, c));
    }
    for g in &mut gens {
        for _ in 0..rng.below(41) {
            let m = rng.below(all.len() as u64) as usize;
            if g.iter().all(|&(k, _)| k != m) {
                let c = coef(rng);
                g.push((m, c));
            }
        }
    }
    for _ in 0..2 + rng.below(5) {
        let mut g: Vec<(usize, i64)> = Vec::new();
        for _ in 0..2 + rng.below(2) {
            let m = rng.below(all.len() as u64) as usize;
            if g.iter().all(|&(k, _)| k != m) {
                g.push((m, coef(rng)));
            }
        }
        gens.push(g);
    }
    let (x, y) = (
        rng.below(gens.len() as u64) as usize,
        rng.below(gens.len() as u64) as usize,
    );
    let mut sum = gens[x].clone();
    for &(m, c) in &gens[y] {
        match sum.iter_mut().find(|(k, _)| *k == m) {
            Some(t) => t.1 += c,
            None => sum.push((m, c)),
        }
    }
    sum.retain(|&(_, c)| c != 0);
    gens.push(sum);
    let z = rng.below(gens.len() as u64) as usize;
    let multiple = gens[z].iter().map(|&(m, c)| (m, -7 * c)).collect();
    gens.push(multiple);
    for k in (1..gens.len()).rev() {
        let j = rng.below(k as u64 + 1) as usize;
        gens.swap(k, j);
    }
    write_out(n, &all, &gens)
}

/// The generators' text on x1, …, xn: each term a coefficient and its
/// monomial's variables, a generator without terms `0`.
fn write_out(n: usize, monomials: &[Vec<u32>], gens: &[Vec<(usize, i64)>]) -> String {
    let mut text = String::new();
    for (k, g) in gens.iter().enumerate() {
        if k > 0 {
            text.push_str(",\n");
        }
        if g.is_empty() {
            text.push('0');
        }
        for (j, &(m, c)) in g.iter().enumerate() {
            if c < 0 {
                text.push('-');
            } else if j > 0 {
                text.push('+');
            }
            write!(text, "{abs}", abs = c.unsigned_abs()).unwrap();
            for (i, &e) in monomials[m].iter().enumerate().take(n) {
                match e {
                    0 => {}
                    1 => write!(text, "*x{k}", k = i + 1).unwrap(),
                    e => write!(text, "*x{k}^{e}", k = i + 1).unwrap(),
                }
            }
        }
    }
    text
}

/// x1, …, xn.
fn names(n: usize) -> String {
    (1..=n)
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Test 4, run-generators-time.5 and .6: drawn systems over every pair of
/// five fields, in grevlex and lex. Most pairs' leads agree, and each that
/// agrees replays to `echelon`'s rows.
#[test]
fn replays_as_the_echelon() {
    let mut ran = 0;
    let mut cases = 0;
    for seed in 0..40u64 {
        let mut rng = Rng::new(seed ^ 0x00ec_4e10);
        let n = 3 + (seed % 3) as usize;
        let systems = [
            (names(n), redundant(&mut rng, n)),
            (names(4), sparse(&mut rng)),
        ];
        for (names, text) in &systems {
            let case = format!("seed {seed}");
            macro_rules! pair {
                ($a:expr, $b:expr) => {{
                    let (a, b) = ($a, $b);
                    cases += 2;
                    ran += usize::from(agrees(&case, &a, &b, &Grevlex, names, text));
                    ran += usize::from(agrees(&case, &a, &b, &Lex, names, text));
                }};
            }
            pair!(Gf31::prime(P), Gf31::prime(P));
            pair!(Gf31::prime(P), Gf64::prime(P61));
            pair!(Gf31::prime(P31), Gf31::prime(P));
            pair!(Gf31Pair::primes(L0, L1), Gf64Pair::primes(R0, R1));
            pair!(Gf31Pair::primes(L0, L1), Gf31::prime(P31));
            pair!(Gf64::prime(P61), Gf64Pair::primes(R0, R1));
            pair!(Gf64Pair::primes(R0, R1), Gf31Pair::primes(L0, L1));
        }
    }
    assert!(ran * 10 >= cases * 9, "{ran} of {cases} cases agreed");
}

/// run-generators-time.10 to .12: drawn systems whose 70 to 252 columns
/// fill several words of the replay's bit set, their rows dense or sparse
/// in their span, over a pair of fields and a pair of pairs. Most pairs'
/// leads agree, and each that agrees replays to `echelon`'s rows.
#[test]
fn replays_over_many_columns() {
    let names = names(5);
    let mut ran = 0;
    let mut cases = 0;
    for seed in 0..20u64 {
        let mut rng = Rng::new(seed ^ 0x0077_1de5);
        let text = wide(&mut rng);
        let case = format!("seed {seed}");
        cases += 2;
        ran += usize::from(agrees(
            &case,
            &Gf31::prime(P),
            &Gf64::prime(P61),
            &Grevlex,
            &names,
            &text,
        ));
        ran += usize::from(agrees(
            &case,
            &Gf31Pair::primes(L0, L1),
            &Gf64Pair::primes(R0, R1),
            &Grevlex,
            &names,
            &text,
        ));
    }
    assert!(ran * 10 >= cases * 9, "{ran} of {cases} cases agreed");
}

/// `replay` over `field` of `generators` on `names` with the leads `leads`
/// halts with `Halt::Misfit(0)`.
fn misfits<F: Field>(
    case: &str,
    field: &F,
    names: &str,
    generators: &str,
    leads: &[Option<Vec<u32>>],
) {
    let mut t = table(names);
    let g = read(field, &mut t, &Grevlex, names, generators);
    match replay(field, &mut t, &Grevlex, g, leads, &mut budget()) {
        Err(Halt::Misfit(0)) => {}
        Err(h) => panic!("{case}: halted with {h:?}, not a misfit"),
        Ok(e) => panic!("{case}: no misfit but rows {rows:?}", rows = e.rows),
    }
}

/// Test 5, run-generators-time.5, .7, and .8: records that do not fit.
#[test]
fn replays_that_misfit() {
    let f = Gf31::prime(P);
    let (x, y, z) = (vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]);
    // A zero row that leaves an entry.
    misfits(
        "a zero row",
        &f,
        "x,y,z",
        "x+y,\nx+2*y",
        &[Some(x.clone()), None],
    );
    // A pivot row that leads before its recorded column.
    misfits(
        "a lead after the row's own",
        &f,
        "x,y,z",
        "x+y,\ny",
        &[Some(y.clone()), Some(x.clone())],
    );
    // A recorded monomial no generator holds.
    misfits(
        "a lead no generator holds",
        &f,
        "x,y,z",
        "x+y,\ny",
        &[Some(z.clone()), Some(y.clone())],
    );
    // Two generators recorded at one column.
    misfits(
        "two pivot rows at one column",
        &f,
        "x,y,z",
        "x+y,\nx+2*y",
        &[Some(x.clone()), Some(x.clone())],
    );
    // A record of another count.
    misfits(
        "one lead short",
        &f,
        "x,y,z",
        "x+y,\ny+z",
        &[Some(x.clone())],
    );
    // A leading coefficient 0 at the first lane of a pair.
    misfits(
        "a lane's zero",
        &Gf64Pair::primes(R0, R1),
        "x,y,z",
        &format!("{R0}*x+y,\ny+z"),
        &[Some(x.clone()), Some(y.clone())],
    );
    // x + y and x + (c + 1)·y, c = 65521·3: over GF(65521) the second
    // generator is the first, and the record leaves it zero; over GF(2^31 −
    // 1) it is not, and the replay misfits.
    let text = "x+y,\nx+196564*y";
    let (ta, ea) = rows(&f, &Grevlex, "x,y,z", text).unwrap();
    let learned = leads(&ta, &ea);
    assert_eq!(learned, vec![Some(x.clone()), None]);
    misfits(
        "an unlucky learn prime",
        &Gf31::prime(P31),
        "x,y,z",
        text,
        &learned,
    );
}

/// Test 6, run-generators-time.5 and .6: the edge cases, each as test 4
/// checks it over GF(65521) and GF(2^61 − 1), and over a pair of pairs.
#[test]
fn replay_edge_cases() {
    let cases: [(&str, &str, &str); 8] = [
        ("no generator", "x,y,z", ""),
        ("every generator zero", "x,y,z", "0,\n0"),
        ("a constant among them", "x,y,z", "x+1,\nx+y,\n1"),
        ("in echelon form already", "x,y,z", "x+z,\ny+2*z"),
        (
            "every generator a pivot row",
            "x,y,z,w",
            "x+y+z+w,\ny+2*z,\nz-w,\nw+5",
        ),
        (
            "one pivot row and its multiples",
            "x,y,z",
            "x+y,\n2*x+2*y,\n-3*x-3*y",
        ),
        (
            "forward rows that hold later pivot columns",
            "x,y,z,w",
            "x+y+z+w,\ny+z+3*w,\nx+2*z,\nz+w^2",
        ),
        (
            "an eliminable variable",
            "x,y,z",
            "x-y^2-z,\ny*z-1,\nz^2-y,\nx+y",
        ),
    ];
    for (case, names, text) in cases {
        assert!(
            agrees(
                case,
                &Gf31::prime(P),
                &Gf64::prime(P61),
                &Grevlex,
                names,
                text
            ),
            "{case}: the leads differ"
        );
        assert!(
            agrees(
                case,
                &Gf31Pair::primes(L0, L1),
                &Gf64Pair::primes(R0, R1),
                &Grevlex,
                names,
                text
            ),
            "{case}: the leads differ over the pairs"
        );
    }
}
