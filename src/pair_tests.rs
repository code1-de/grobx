//! signature-pair-time, the sub-item: the S-pair's signature formed in the
//! table's scratch words, the base divisors and their triangle, the query
//! of the known syzygy signatures, and the signature run with base
//! divisors on and off, which returns one basis or one halt (plan.md,
//! _Test code_, on generated systems and without its counts).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;
use std::fmt::Write;
use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::f4::{self, Elements, Halt};
use crate::field::{Field, Gf31};
use crate::id::{ElementId, MonoId};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format, Overflow};
use crate::order::Grevlex;
use crate::poly::{Basis, Poly};
use crate::sig::{self, Base, BaseDivisors, Formed, Pairs, Sig, Signatures, Triangle};
use crate::systems::{dense, random_system};
use crate::table::Table;
use crate::text::{Header, write_basis};
use crate::trace::{NoLearner, NoTrace};
use crate::{MonomialOrder, Params, Stats, Threads};

/// The variables of the random worlds.
const N: usize = 3;
/// The indices of their signatures.
const INDICES: u32 = 2;

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn below(rng: &mut Xoshiro256PlusPlus, n: u64) -> u64 {
    rng.next_u64() % n
}

/// A random exponent vector of `n` variables and total degree at most
/// `degree`.
fn random_exps(rng: &mut Xoshiro256PlusPlus, n: usize, degree: u64) -> Vec<u32> {
    let mut left = below(rng, degree + 1);
    let mut e = vec![0u32; n];
    for slot in &mut e {
        let x = below(rng, left + 1);
        *slot = x as u32;
        left -= x;
    }
    // Not always the first variables first.
    let shift = below(rng, n as u64) as usize;
    e.rotate_right(shift);
    e
}

fn divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

fn signed(e: &[u32]) -> Vec<i64> {
    e.iter().map(|&x| i64::from(x)).collect()
}

fn minus(a: &[u32], b: &[u32]) -> Vec<i64> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) - i64::from(y))
        .collect()
}

/// Grevlex on exponent vectors of any sign.
fn grevlex(a: &[i64], b: &[i64]) -> Ordering {
    let (da, db): (i64, i64) = (a.iter().sum(), b.iter().sum());
    da.cmp(&db).then_with(|| {
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return b[i].cmp(&a[i]);
            }
        }
        Ordering::Equal
    })
}

/// Schreyer's order on (exponents, index).
fn sig_cmp(a: (&[u32], u32), b: (&[u32], u32)) -> Ordering {
    grevlex(&signed(a.0), &signed(b.0)).then(a.1.cmp(&b.1))
}

fn sig_divides(a: (&[u32], u32), b: (&[u32], u32)) -> bool {
    a.1 == b.1 && divides(a.0, b.0)
}

// ---------------------------------------------------------------------
// The quotient on words.

/// s·lcm(x, y)/x on exponent vectors, and its degree.
fn quotient(s: &[u32], x: &[u32], y: &[u32]) -> (Vec<u32>, u64) {
    let e: Vec<u32> = (0..s.len()).map(|k| s[k] + y[k] - x[k].min(y[k])).collect();
    let d = e.iter().map(|&v| u64::from(v)).sum();
    (e, d)
}

fn words<M: Format>(format: &M, e: &[u32]) -> Vec<M::Word> {
    let mut w = vec![M::Word::default(); format.words()];
    format.encode(e, &mut w).expect("the test's exponents fit");
    w
}

fn check_quotients<M: Format>(format: M, seed: u64) {
    let n = format.variables();
    let mut rng = rng(seed);
    // Degrees up to half the bound, at most 600, so that s, x, and y fit;
    // the quotient, of degree at most deg s + deg y, stays within it.
    let most = (u64::from(M::MAX_DEGREE) / 2).min(600);
    for _ in 0..3000 {
        let (s, x, y) = (
            random_exps(&mut rng, n, most),
            random_exps(&mut rng, n, most),
            random_exps(&mut rng, n, most),
        );
        let mut out = vec![M::Word::default(); format.words()];
        let got = format.mul_quotient(
            &words(&format, &s),
            &words(&format, &x),
            &words(&format, &y),
            &mut out,
        );
        let (want, degree) = quotient(&s, &x, &y);
        if degree > u64::from(M::MAX_DEGREE) {
            assert_eq!(got, Err(Overflow), "{s:?} {x:?} {y:?}");
            continue;
        }
        assert_eq!(got, Ok(degree as u32), "{s:?} {x:?} {y:?}");
        let read: Vec<u32> = (0..n).map(|k| format.exponent(&out, k)).collect();
        assert_eq!(read, want, "{s:?} {x:?} {y:?}");
    }
}

/// The bound exactly: s with one exponent at `MAX_DEGREE`, and a quotient
/// one past it.
fn check_bound<M: Format>(format: M) {
    let n = format.variables();
    let max = M::MAX_DEGREE;
    let zero = vec![0u32; n];
    let mut s = zero.clone();
    s[0] = max;
    let mut out = vec![M::Word::default(); format.words()];
    let w0 = words(&format, &zero);
    assert_eq!(
        format.mul_quotient(&words(&format, &s), &w0, &w0, &mut out),
        Ok(max)
    );
    assert_eq!(format.exponent(&out, 0), max);
    let mut y = zero.clone();
    y[0] = 1;
    assert_eq!(
        format.mul_quotient(&words(&format, &s), &w0, &words(&format, &y), &mut out),
        Err(Overflow),
        "one exponent past the word's bound"
    );
    if n > 1 {
        let mut s = zero.clone();
        s[0] = max - 1;
        let mut y = zero.clone();
        y[1] = 1;
        assert_eq!(
            format.mul_quotient(&words(&format, &s), &w0, &words(&format, &y), &mut out),
            Ok(max)
        );
        y[1] = 2;
        assert_eq!(
            format.mul_quotient(&words(&format, &s), &w0, &words(&format, &y), &mut out),
            Err(Overflow),
            "the degree one past the bound"
        );
        // y divides x: the quotient is s alone.
        let mut x = zero.clone();
        x[1] = 5;
        y[1] = 3;
        assert_eq!(
            format.mul_quotient(
                &words(&format, &s),
                &words(&format, &x),
                &words(&format, &y),
                &mut out
            ),
            Ok(max - 1)
        );
    }
}

#[test]
fn the_quotient_is_formed_exponent_by_exponent() {
    for n in [1, 3, 40] {
        check_quotients(Dense8::new(n), 1);
        check_quotients(Dense16::new(n), 2);
        check_quotients(Dense32::new(n), 3);
        check_bound(Dense8::new(n));
        check_bound(Dense16::new(n));
        check_bound(Dense32::new(n));
    }
}

// ---------------------------------------------------------------------
// A world of elements with signatures, built as `sig_tests.rs` builds its
// world, each random exponent vector rotated.

fn table(seed: u64) -> Table<Dense16> {
    Table::new(Dense16::new(N), &mut rng(seed ^ 0x5197))
}

fn id(table: &mut Table<Dense16>, exps: &[u32]) -> MonoId {
    table.intern(exps).unwrap()
}

fn exps(table: &Table<Dense16>, id: MonoId) -> Vec<u32> {
    (0..N).map(|i| table.exponent(id, i)).collect()
}

fn random_sig(rng: &mut Xoshiro256PlusPlus, table: &mut Table<Dense16>, degree: u64) -> Sig {
    let e = random_exps(rng, N, degree);
    Sig {
        monomial: id(table, &e),
        index: below(rng, u64::from(INDICES)) as u32,
    }
}

/// A monic polynomial with leading monomial `lead` and up to `tail` terms
/// below it.
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
        let m = id(table, &random_exps(rng, N, d));
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

/// Per element: signature exponents, index, and leading exponents.
struct World {
    table: Table<Dense16>,
    basis: Basis<Gf31, Rank1>,
    sigs: Signatures,
    elements: Vec<(Vec<u32>, u32, Vec<u32>)>,
}

fn world(seed: u64, count: usize) -> World {
    let mut rng = rng(seed);
    let mut table = table(seed);
    let field = Gf31::new(65521).unwrap();
    let mut basis = Basis::new();
    let mut sigs = Signatures::new(INDICES as usize);
    let mut elements = Vec::new();
    for _ in 0..count {
        let lead = random_exps(&mut rng, N, 4);
        let poly = random_poly(&mut rng, &mut table, &field, &lead, 3);
        let sig = random_sig(&mut rng, &mut table, 6);
        basis.push(poly, false).unwrap();
        sigs.push(&table, &Grevlex, &basis, sig).unwrap();
        elements.push((exps(&table, sig.monomial), sig.index, lead));
    }
    World {
        table,
        basis,
        sigs,
        elements,
    }
}

/// The sig-lead ratio of element e, with its index.
fn ratio(w: &World, e: usize) -> (Vec<i64>, u32) {
    (minus(&w.elements[e].0, &w.elements[e].2), w.elements[e].1)
}

fn ratio_cmp(w: &World, a: usize, b: usize) -> Ordering {
    let (ra, ia) = ratio(w, a);
    let (rb, ib) = ratio(w, b);
    grevlex(&ra, &rb).then(ia.cmp(&ib))
}

/// The signature of S(g, h) on exponent vectors: the larger multiple, None
/// where the two are equal.
fn pair_sig(w: &World, g: usize, h: usize) -> Option<(Vec<u32>, u32)> {
    let (sg, ig, lg) = w.elements[g].clone();
    let (sh, ih, lh) = w.elements[h].clone();
    let lcm: Vec<u32> = lg.iter().zip(&lh).map(|(a, b)| *a.max(b)).collect();
    let mg: Vec<u32> = (0..N).map(|k| lcm[k] - lg[k] + sg[k]).collect();
    let mh: Vec<u32> = (0..N).map(|k| lcm[k] - lh[k] + sh[k]).collect();
    match sig_cmp((&mg, ig), (&mh, ih)) {
        Ordering::Equal => None,
        Ordering::Greater => Some((mg, ig)),
        Ordering::Less => Some((mh, ih)),
    }
}

#[test]
fn the_formed_signature_is_the_pairs() {
    for seed in 0..20 {
        let mut w = world(seed, 30);
        for g in 0..w.elements.len() {
            for h in 0..w.elements.len() {
                if g == h {
                    continue;
                }
                let want = pair_sig(&w, g, h);
                let (ge, he) = (ElementId::at(g), ElementId::at(h));
                let got = sig::pair(&mut w.table, &w.basis, &w.sigs, ge, he).unwrap();
                match (got, want) {
                    (None, None) => {}
                    (Some(f), Some((e, index))) => {
                        let format = *w.table.format();
                        let formed: Vec<u32> = (0..N)
                            .map(|k| format.exponent(w.table.formed(), k))
                            .collect();
                        assert_eq!(formed, e, "seed {seed}: pair {g}, {h}");
                        assert_eq!(f.index, index, "seed {seed}: pair {g}, {h}");
                        assert_eq!(f.degree, e.iter().sum::<u32>(), "seed {seed}");
                        assert_eq!(f.mask, format.mask(w.table.formed()), "seed {seed}");
                        let interned = w.table.intern_formed(f.degree).unwrap();
                        assert_eq!(exps(&w.table, interned), e, "seed {seed}");
                        assert_eq!(interned, id(&mut w.table, &e), "seed {seed}");
                        assert_eq!(w.table.mask(interned), f.mask, "seed {seed}");
                    }
                    (got, want) => panic!(
                        "seed {seed}: pair {g}, {h}: got {got:?}, want {want:?}",
                        got = got.map(|f| (f.index, f.degree))
                    ),
                }
            }
        }
    }
}

/// Brute force: the element of largest ratio among those before t that
/// `candidate` admits, compared as `before` compares them.
fn largest(w: &World, t: usize, candidate: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for e in 0..t {
        if !candidate(e) {
            continue;
        }
        match best.first().map(|&b| ratio_cmp(w, e, b)) {
            None | Some(Ordering::Greater) => best = vec![e],
            Some(Ordering::Equal) => best.push(e),
            Some(Ordering::Less) => {}
        }
    }
    best
}

/// x^v of roune-2012 Theorem 8 for the low ratio base divisor a of t,
/// `Dense16::MAX_DEGREE` for an infinite exponent (monomial-time.12).
fn x_v(w: &World, a: usize, t: usize) -> Vec<u32> {
    let (sa, _, la) = &w.elements[a];
    let (st, _, lt) = &w.elements[t];
    (0..N)
        .map(|k| {
            let p = la[k] + st[k] - sa[k];
            if lt[k] <= p {
                Dense16::MAX_DEGREE
            } else {
                p.max(la[k])
            }
        })
        .collect()
}

#[test]
fn the_base_divisors_have_the_largest_ratio() {
    for seed in 0..20 {
        let w = world(seed ^ 0xbd, 30);
        for t in 0..w.elements.len() {
            let got = BaseDivisors::of(&w.table, &w.basis, &w.sigs, ElementId::at(t));
            let high = largest(&w, t, |e| divides(&w.elements[e].2, &w.elements[t].2));
            match got.high {
                None => assert!(high.is_empty(), "seed {seed}: t {t}: {high:?}"),
                Some(a) => assert!(
                    high.contains(&a.index()),
                    "seed {seed}: t {t}: {a} not in {high:?}"
                ),
            }
            let (st, it, _) = &w.elements[t];
            let low = largest(&w, t, |e| {
                sig_divides((&w.elements[e].0, w.elements[e].1), (st, *it))
            });
            match &got.low {
                None => assert!(low.is_empty(), "seed {seed}: t {t}: {low:?}"),
                Some((a, v)) => {
                    assert!(
                        low.contains(&a.index()),
                        "seed {seed}: t {t}: {a} not in {low:?}"
                    );
                    let format = w.table.format();
                    let v: Vec<u32> = (0..N).map(|k| format.exponent(v, k)).collect();
                    assert_eq!(v, x_v(&w, a.index(), t), "seed {seed}: t {t}");
                }
            }
        }
    }
}

/// A triangle of `n` elements with every bit set.
fn full(n: usize) -> Triangle {
    let mut triangle = Triangle::new();
    triangle.grow(n);
    for j in 0..n {
        for i in 0..j {
            triangle.set(ElementId::at(i), ElementId::at(j));
        }
    }
    triangle
}

#[test]
fn the_base_divisors_eliminate_only_multiples() {
    let mut low_cases = (0usize, 0usize);
    for seed in 0..40 {
        let w = world(seed ^ 0xe1, 30);
        let n = w.elements.len();
        let all = full(n);
        let mut none = Triangle::new();
        none.grow(n);
        for t in 0..n {
            let te = ElementId::at(t);
            let divisors = BaseDivisors::of(&w.table, &w.basis, &w.sigs, te);
            for g in 0..t {
                let ge = ElementId::at(g);
                let got = divisors.eliminates(&w.table, &w.basis, &w.sigs, &all, te, ge);
                if let Some(base) = &got {
                    let a = match base {
                        Base::High => divisors.high.unwrap(),
                        Base::Low => divisors.low.as_ref().unwrap().0,
                    }
                    .index();
                    let (sa, ia) =
                        pair_sig(&w, a, g).expect("a regular S-pair of the base divisor");
                    let (st, it) = pair_sig(&w, t, g).expect("a regular S-pair");
                    assert!(
                        sig_divides((&sa, ia), (&st, it)),
                        "seed {seed}: S({a}, {g}) does not divide S({t}, {g})"
                    );
                }
                assert!(
                    divisors
                        .eliminates(&w.table, &w.basis, &w.sigs, &none, te, ge)
                        .is_none(),
                    "seed {seed}: no bit set, S({t}, {g}) eliminated"
                );
            }
            // The low ratio base divisor alone: Theorem 8's "if and only if"
            // where the ratio conditions hold.
            let Some((a, v)) = divisors.low.clone() else {
                continue;
            };
            let alone = BaseDivisors {
                high: None,
                low: Some((a, v)),
            };
            let a = a.index();
            for g in 0..t {
                if g == a || ratio_cmp(&w, g, a).is_ge() || ratio_cmp(&w, g, t).is_ge() {
                    continue;
                }
                let (sa, ia) = pair_sig(&w, a, g).unwrap();
                let (st, it) = pair_sig(&w, t, g).unwrap();
                let want = sig_divides((&sa, ia), (&st, it));
                let got = alone.eliminates(
                    &w.table,
                    &w.basis,
                    &w.sigs,
                    &all,
                    ElementId::at(t),
                    ElementId::at(g),
                );
                assert_eq!(
                    matches!(got, Some(Base::Low)),
                    want,
                    "seed {seed}: t {t}, g {g}, α′ {a}"
                );
                low_cases.0 += 1;
                low_cases.1 += usize::from(want);
            }
        }
    }
    assert!(
        low_cases.1 > 0 && low_cases.1 < low_cases.0,
        "{low_cases:?}"
    );
}

#[test]
fn the_triangle_holds_each_pair_once() {
    let mut rng = rng(5);
    let n = 200usize;
    let mut triangle = Triangle::new();
    triangle.grow(n);
    let mut model = vec![vec![false; n]; n];
    for _ in 0..3000 {
        let i = below(&mut rng, n as u64) as usize;
        let j = below(&mut rng, n as u64) as usize;
        if i == j {
            continue;
        }
        triangle.set(ElementId::at(i), ElementId::at(j));
        model[i][j] = true;
        model[j][i] = true;
    }
    for (i, row) in model.iter().enumerate() {
        for (j, &set) in row.iter().enumerate() {
            if i != j {
                assert_eq!(
                    triangle.get(ElementId::at(i), ElementId::at(j)),
                    set,
                    "{i}, {j}"
                );
            }
        }
    }
}

#[test]
fn the_syzygy_query_agrees_with_a_scan() {
    for seed in 0..20 {
        let mut rng = rng(seed ^ 0x5c4);
        let mut table = table(seed);
        let one = id(&mut table, &[0, 0, 0]);
        let mut sigs = Signatures::new(INDICES as usize);
        let mut pushed: Vec<(Vec<u32>, u32)> = Vec::new();
        for _ in 0..2000 {
            let s = random_sig(&mut rng, &mut table, 6);
            let se = exps(&table, s.monomial);
            let known = pushed
                .iter()
                .any(|(e, i)| sig_divides((e, *i), (&se, s.index)));
            for _ in 0..2 {
                assert_eq!(sigs.is_syzygy(&table, s), known, "seed {seed}: {se:?}");
                let degree = table.form_quotient(s.monomial, one, one).unwrap();
                let formed = Formed {
                    index: s.index,
                    degree,
                    mask: table.format().mask(table.formed()),
                };
                assert_eq!(degree, se.iter().sum::<u32>());
                assert_eq!(
                    sigs.is_syzygy_formed(&table, &formed),
                    known,
                    "seed {seed}: {se:?}"
                );
            }
            if below(&mut rng, 4) == 0 {
                assert_eq!(sigs.push_syzygy(&table, s), !known, "seed {seed}: {se:?}");
                pushed.push((se, s.index));
            }
        }
    }
}

// ---------------------------------------------------------------------
// The signature run with base divisors on and off.

fn budget() -> Budget {
    let params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    Budget::new(Instant::now(), &params).unwrap()
}

/// The signature run over `text`, in grevlex, in the format `M`, with base
/// divisors where `bases`, to its end: the basis text, or the halt.
fn run<M: Format>(text: &str, bases: bool) -> Result<String, Halt> {
    let header = Header::read(text).unwrap();
    let field = header.gf31().unwrap();
    let mut table = Table::new(
        M::new(header.names.len()),
        &mut Xoshiro256PlusPlus::seed_from_u64(1),
    );
    let generators: Vec<Poly<Gf31, Rank1>> =
        header.generators(&field, &mut table, &Grevlex).unwrap();
    let mut stats = Stats::default();
    let mut pairs = Pairs::new(bases);
    let got: Result<Elements<Gf31, Rank1, ()>, Halt> = f4::run_sig(
        &field,
        &mut table,
        &Grevlex,
        generators,
        &mut NoCofactors,
        &mut NoTrace,
        &mut budget(),
        &mut stats,
        &mut NoLearner,
        None,
        &mut pairs,
        1,
    );
    got.map(|elements| {
        let mut out = String::new();
        write_basis(
            &mut out,
            &header.names,
            &table,
            &field,
            elements.iter().map(|(g, ())| g),
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        out
    })
}

/// The runs over `text` with base divisors and without return one basis,
/// or one halt; the run with them returns it.
fn agree(name: &str, text: &str) -> Result<String, Halt> {
    let on = run::<Dense16>(text, true);
    assert_eq!(on, run::<Dense16>(text, false), "{name}: the bases");
    on
}

/// A random system in one variable over GF(65521).
fn univariate(seed: u64) -> String {
    let mut rng = rng(seed ^ 0x1);
    let count = 1 + below(&mut rng, 4);
    let generators: Vec<String> = (0..count)
        .map(|_| {
            let mut poly = String::new();
            for _ in 0..=below(&mut rng, 4) {
                let c = 1 + below(&mut rng, 65520);
                let e = below(&mut rng, 9);
                if !poly.is_empty() {
                    poly.push('+');
                }
                match e {
                    0 => poly.push_str(&c.to_string()),
                    1 => write!(poly, "{c}*x").unwrap(),
                    _ => write!(poly, "{c}*x^{e}").unwrap(),
                }
            }
            poly
        })
        .collect();
    format!(
        "x\n65521\n{generators}\n",
        generators = generators.join(",\n")
    )
}

#[test]
fn base_divisors_keep_the_basis() {
    for seed in 0..300 {
        let _ = agree(&format!("random-{seed}"), &random_system(65521, seed));
    }
    for seed in 0..50 {
        let _ = agree(&format!("univariate-{seed}"), &univariate(seed));
    }
    for m in 3..=4 {
        let name = format!("cubic-{m}");
        let got = agree(&name, &dense(m, 3, 65521, m as u64));
        assert!(got.is_ok(), "{name}: {got:?}");
    }
}

/// The edge cases of plan.md, with the basis text where it is written.
const EDGE: &[(&str, &str, Option<&str>)] = &[
    ("zero", "x,y\n65521\n0\n", Some("[]:\n")),
    ("zeros", "x,y\n7\n0,\n0\n", Some("[]:\n")),
    ("none", "x,y\n65521\n", Some("[]:\n")),
    ("unit", "x,y\n65521\nx*y-1,\nx\n", Some("[1]:\n")),
    ("unit-constant", "x,y\n65521\n5\n", Some("[1]:\n")),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
        Some("[1]:\n"),
    ),
    ("gb", "x,y\n65521\nx+y,\ny^2+1\n", Some("[x+y,\ny^2+1]:\n")),
    (
        "gb-swapped",
        "x,y\n65521\ny^2+1,\nx+y\n",
        Some("[x+y,\ny^2+1]:\n"),
    ),
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n", None),
    (
        "eliminable",
        "x,y,z\n65521\nx-y^2-z,\ny^3-z,\nz^2-y*z-1\n",
        None,
    ),
    (
        "one-variable",
        "x\n65521\nx^5-1,\nx^3-1\n",
        Some("[x+65520]:\n"),
    ),
    (
        "one-generator",
        "x,y\n65521\nx^2*y+y^3-1\n",
        Some("[x^2*y+y^3+65520]:\n"),
    ),
];

#[test]
fn the_edge_cases_through_both() {
    for (name, text, want) in EDGE {
        let on = agree(name, text);
        if let Some(want) = want {
            assert_eq!(on.as_deref(), Ok(*want), "{name}");
        }
    }
}

/// In `Dense8`, a run whose S-pair passes the degree bound halts with
/// `Halt::Overflow`; a run whose pair past the bound is coprime returns
/// its basis.
#[test]
fn an_overflow_halts_the_run() {
    let shared = "x,y\n65521\nx^100-y,\nx^50*y^30-1\n";
    let coprime = "x,y\n65521\nx^100-1,\ny^60-1\n";
    for bases in [true, false] {
        let got = run::<Dense8>(shared, bases);
        assert!(matches!(got, Err(Halt::Overflow)), "{bases}: {got:?}");
        let got = run::<Dense8>(coprime, bases);
        assert_eq!(
            got.as_deref(),
            Ok("[y^60+65520,\nx^100+65520]:\n"),
            "{bases}"
        );
    }
}
