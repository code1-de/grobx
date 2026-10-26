//! monomial-time, the sub-item: every operation of the three formats on
//! packed words but `unpack`, `mul_word`, and `grevlex_key`, which
//! `output_tests.rs`, `build_tests.rs`, and `key_tests.rs` test, gives what
//! an exponent-by-exponent oracle gives, at the lane borders and at each format's bound, and the table agrees with
//! interning in every width (plan.md, _Test code_).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::divisibility_tests::mask_of;
use crate::id::MonoId;
use crate::monomial::{Dense8, Dense16, Dense32, Format, Overflow};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::table::{Limit, Table};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn below(rng: &mut Xoshiro256PlusPlus, n: u64) -> u64 {
    rng.next_u64() % n
}

/// The bits of a lane of format `M`: 8, 16, or 32 (monomial-time.1).
fn bits<M: Format>() -> u32 {
    (M::MAX_DEGREE + 1).trailing_zeros() + 1
}

/// Exponents per word of format `M`.
fn lanes<M: Format>() -> usize {
    64 / bits::<M>() as usize
}

/// The variable counts of the plan's _Edge cases_ for a format of L lanes.
fn counts(l: usize) -> Vec<usize> {
    let mut n = vec![
        1,
        l - 1,
        l,
        l + 1,
        2 * l - 1,
        2 * l,
        2 * l + 1,
        31,
        32,
        33,
        54,
        64,
        65,
    ];
    n.retain(|&k| k >= 1);
    n.sort_unstable();
    n.dedup();
    n
}

// --- the oracle, exponent by exponent ---

fn degree(e: &[u32]) -> u64 {
    e.iter().map(|&x| u64::from(x)).sum()
}

/// The words of exponents `e` in the layout of monomial-time.1 and .11:
/// variable i in lane i mod L of word ⌊i/L⌋, the low lanes first, the
/// lanes past the last variable 0.
fn layout<M: Format>(e: &[u32]) -> Vec<u64> {
    let (b, l) = (bits::<M>(), lanes::<M>());
    let mut w = vec![0u64; e.len().div_ceil(l)];
    for (i, &x) in e.iter().enumerate() {
        w[i / l] |= u64::from(x) << (b as usize * (i % l));
    }
    w
}

fn revlex(a: &[u32], b: &[u32]) -> Ordering {
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return b[i].cmp(&a[i]);
        }
    }
    Ordering::Equal
}

fn lex(a: &[u32], b: &[u32]) -> Ordering {
    for i in 0..a.len() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

fn grevlex(a: &[u32], b: &[u32]) -> Ordering {
    degree(a).cmp(&degree(b)).then_with(|| revlex(a, b))
}

/// The block order of `Block::new(&[k], n)`: the first k variables, then
/// the rest, each block by its degree, then reverse lexicographically.
fn block(k: usize, a: &[u32], b: &[u32]) -> Ordering {
    for (s, e) in [(0, k), (k, a.len())] {
        let o = grevlex(&a[s..e], &b[s..e]);
        if o.is_ne() {
            return o;
        }
    }
    Ordering::Equal
}

// --- draws ---

/// A vector of n exponents of degree at most `cap`, spread over the
/// variables in steps of at most `step`.
fn draw(rng: &mut Xoshiro256PlusPlus, n: usize, cap: u64, step: u64) -> Vec<u32> {
    let mut left = below(rng, cap + 1);
    let mut e = vec![0u32; n];
    for _ in 0..2 * n + 4 {
        if left == 0 {
            break;
        }
        let i = below(rng, n as u64) as usize;
        let k = below(rng, left.min(step) + 1);
        e[i] += k as u32;
        left -= k;
    }
    e
}

/// A pair of the four kinds of the plan's test 1: small, near the bound,
/// divisor and multiple, and a borrow between neighbour lanes. Each is a
/// monomial of the format.
fn pair<M: Format>(rng: &mut Xoshiro256PlusPlus, n: usize, kind: u64) -> (Vec<u32>, Vec<u32>) {
    let max = u64::from(M::MAX_DEGREE);
    match kind {
        0 => (draw(rng, n, max / 2, 4), draw(rng, n, max / 2, 4)),
        1 => {
            // One exponent of a near the bound, a's degree within 3 of it,
            // and b of at most the degree left.
            let mut a = draw(rng, n, max / 8, 3);
            let i = below(rng, n as u64) as usize;
            let room = max - degree(&a);
            a[i] += (room - below(rng, room.min(3) + 1)) as u32;
            let b = draw(rng, n, max - degree(&a).min(max), 3);
            (a, b)
        }
        2 => {
            let a = draw(rng, n, max / 4, 3);
            let mut b = a.clone();
            let mut left = max - degree(&a);
            for x in &mut b {
                let k = below(rng, left.min(2) + 1);
                *x += k as u32;
                left -= k;
            }
            if below(rng, 2) == 0 { (a, b) } else { (b, a) }
        }
        _ => {
            // b below a in one lane, above it in the neighbours.
            let mut a = draw(rng, n, max / 4, 3);
            let i = below(rng, n as u64) as usize;
            if a[i] == 0 {
                a[i] = 1;
            }
            let mut b = a.clone();
            b[i] -= 1;
            let mut left = max.saturating_sub(degree(&b));
            for j in [i.wrapping_sub(1), i + 1] {
                if j < n && left >= 5 {
                    b[j] += 5;
                    left -= 5;
                }
            }
            (a, b)
        }
    }
}

fn mono<M: Format>(f: &M, e: &[u32]) -> (Vec<M::Word>, u32) {
    let mut m = vec![M::Word::default(); f.words()];
    let d = f.encode(e, &mut m).expect("a monomial of the format");
    (m, d)
}

fn exps<M: Format>(f: &M, m: &[M::Word]) -> Vec<u32> {
    (0..f.variables()).map(|i| f.exponent(m, i)).collect()
}

// --- test 1 ---

/// Every operation on a and b, each a monomial of the format, against the
/// oracle, but `unpack`, `mul_word`, and `grevlex_key`.
fn agree<M: Format<Word = u64>>(f: &M, a: &[u32], b: &[u32], case: &str) {
    let n = a.len();
    let max = u64::from(M::MAX_DEGREE);
    let (ma, da) = mono(f, a);
    let (mb, db) = mono(f, b);
    assert_eq!(ma, layout::<M>(a), "{case}: the layout of a");
    assert_eq!(exps(f, &ma), a, "{case}: a read back");
    assert_eq!(u64::from(da), degree(a), "{case}");
    let mut out = vec![0u64; f.words()];

    let product: Vec<u32> = a.iter().zip(b).map(|(x, y)| x + y).collect();
    match f.mul(&ma, da, &mb, db, &mut out) {
        Ok(d) => {
            assert!(degree(&product) <= max, "{case}: a product past the bound");
            assert_eq!(u64::from(d), degree(&product), "{case}");
            assert_eq!(out, layout::<M>(&product), "{case}: a·b");
            let (mp, _) = mono(f, &product);
            assert!(f.is_product(&ma, &mb, &mp), "{case}: a·b is the product");
            assert!(f.is_product(&mb, &ma, &mp), "{case}: b·a is the product");
            // A product one off in a lane is not the product.
            for i in [0, n / 2, n - 1] {
                let mut c = product.clone();
                if degree(&c) < max {
                    c[i] += 1;
                    let (mc, _) = mono(f, &c);
                    assert!(!f.is_product(&ma, &mb, &mc), "{case}: one more at {i}");
                }
                let mut c = product.clone();
                if c[i] > 0 {
                    c[i] -= 1;
                    let (mc, _) = mono(f, &c);
                    assert!(!f.is_product(&ma, &mb, &mc), "{case}: one less at {i}");
                }
            }
            assert_eq!(f.div(&mp, &mb, &mut out), da, "{case}: a·b / b");
            assert_eq!(out, ma, "{case}: a·b / b is a");
        }
        Err(Overflow) => assert!(degree(&product) > max, "{case}: a product within the bound"),
    }

    let divides = a.iter().zip(b).all(|(x, y)| x <= y);
    assert_eq!(f.divides(&ma, &mb), divides, "{case}: a | b");
    assert_eq!(
        f.divides(&mb, &ma),
        b.iter().zip(a).all(|(x, y)| x <= y),
        "{case}: b | a"
    );
    if divides {
        let quotient: Vec<u32> = b.iter().zip(a).map(|(y, x)| y - x).collect();
        assert_eq!(
            u64::from(f.div(&mb, &ma, &mut out)),
            degree(&quotient),
            "{case}"
        );
        assert_eq!(out, layout::<M>(&quotient), "{case}: b / a");
    }

    let lcm: Vec<u32> = a.iter().zip(b).map(|(x, y)| *x.max(y)).collect();
    assert_eq!(
        u64::from(f.lcm_degree(&ma, &mb)),
        degree(&lcm),
        "{case}: lcm_degree"
    );
    match f.lcm(&ma, &mb, &mut out) {
        Ok(d) => {
            assert_eq!(u64::from(d), degree(&lcm), "{case}");
            assert_eq!(out, layout::<M>(&lcm), "{case}: lcm");
        }
        Err(Overflow) => assert!(degree(&lcm) > max, "{case}: an lcm within the bound"),
    }

    let coprime = a.iter().zip(b).all(|(&x, &y)| x == 0 || y == 0);
    assert_eq!(f.coprime(&ma, &mb), coprime, "{case}: coprime");

    assert_eq!(f.cmp_revlex(&ma, &mb), revlex(a, b), "{case}: revlex");
    assert_eq!(f.cmp_lex(&ma, &mb), lex(a, b), "{case}: lex");
    assert_eq!(
        Grevlex.cmp(f, &ma, da, &mb, db),
        grevlex(a, b),
        "{case}: grevlex"
    );
    assert_eq!(Lex.cmp(f, &ma, da, &mb, db), lex(a, b), "{case}: Lex");
    for k in [1, n / 2, n - 1] {
        if k >= 1 && k < n {
            let order = Block::new(&[k], n).unwrap();
            assert_eq!(
                order.cmp(f, &ma, da, &mb, db),
                block(k, a, b),
                "{case}: block:{k}"
            );
        }
    }
    assert_eq!(f.mask(&ma), mask_of(a), "{case}: mask");

    // The S-pair's signature s·lcm(x, y)/x with s = a·b where it holds.
    if degree(&product) <= max {
        let (ms, _) = mono(f, &product);
        let want: Vec<u32> = (0..n)
            .map(|i| product[i] + b[i].saturating_sub(a[i]))
            .collect();
        match f.mul_quotient(&ms, &ma, &mb, &mut out) {
            Ok(d) => {
                assert_eq!(u64::from(d), degree(&want), "{case}");
                assert_eq!(out, layout::<M>(&want), "{case}: s·lcm/x");
            }
            Err(Overflow) => assert!(degree(&want) > max, "{case}: a signature within the bound"),
        }
    }

    // A bound of exponents at most MAX, whatever their sum: x^v of
    // monomial-time.12.
    let v: Vec<u32> = b
        .iter()
        .enumerate()
        .map(|(i, &y)| if i % 3 == 0 { M::MAX_DEGREE } else { y })
        .collect();
    let mut packed = vec![0u64; f.words()];
    f.pack(&v, &mut packed);
    assert_eq!(packed, layout::<M>(&v), "{case}: pack");
    assert_eq!(
        f.divides(&ma, &packed),
        a.iter().zip(&v).all(|(x, y)| x <= y),
        "{case}: a | v"
    );
}

fn operations<M: Format<Word = u64>>(seed: u64) {
    let mut rng = rng(seed);
    for n in counts(lanes::<M>()) {
        let f = M::new(n);
        assert_eq!(f.words(), n.div_ceil(lanes::<M>()), "{n} variables");
        for k in 0..400 {
            let kind = k % 4;
            let (a, b) = pair::<M>(&mut rng, n, kind);
            agree(
                &f,
                &a,
                &b,
                &format!("{f:?}, {n} variables, kind {kind}: {a:?} {b:?}"),
            );
        }
    }
}

#[test]
fn the_operations_agree_with_the_exponents() {
    operations::<Dense8>(1);
    operations::<Dense16>(2);
    operations::<Dense32>(3);
}

// --- test 2 ---

fn bound<M: Format<Word = u64>>() {
    let max = M::MAX_DEGREE;
    let l = lanes::<M>();
    let n = 2 * l + 1;
    let f = M::new(n);
    let mut out = vec![0u64; f.words()];
    let unit = |i: usize, e: u32| {
        let mut v = vec![0u32; n];
        v[i] = e;
        v
    };
    for i in 0..n {
        let j = (i + 1) % n;
        let (top, d) = mono(&f, &unit(i, max));
        assert_eq!(d, max, "{f:?} lane {i}");
        assert_eq!(top, layout::<M>(&unit(i, max)), "{f:?} lane {i}");
        assert_eq!(f.exponent(&top, i), max, "{f:?} lane {i}");
        assert_eq!(
            f.encode(&unit(i, max + 1), &mut out),
            Err(Overflow),
            "{f:?} lane {i}"
        );
        let (below_top, _) = mono(&f, &unit(i, max - 1));
        assert!(f.divides(&top, &top), "{f:?} lane {i}");
        assert!(f.divides(&below_top, &top), "{f:?} lane {i}");
        assert!(!f.divides(&top, &below_top), "{f:?} lane {i}");
        let (other, _) = mono(&f, &unit(j, 1));
        assert!(
            !f.divides(&top, &other) && !f.divides(&other, &top),
            "{f:?} lane {i}"
        );
        // Products to exactly the bound, and one past it.
        let h = max / 2 + 1;
        let (xh, dh) = mono(&f, &unit(i, h));
        let (xl, dl) = mono(&f, &unit(i, max - h));
        assert_eq!(f.mul(&xh, dh, &xl, dl, &mut out), Ok(max), "{f:?} lane {i}");
        assert_eq!(out, top, "{f:?} lane {i}");
        let (yh, dyh) = mono(&f, &unit(j, h));
        assert_eq!(
            f.mul(&xh, dh, &yh, dyh, &mut out),
            Err(Overflow),
            "{f:?} lane {i}"
        );
        assert_eq!(
            f.mul(&xh, dh, &xh, dh, &mut out),
            Err(Overflow),
            "{f:?} lane {i}"
        );
        // lcms to the bound and past it.
        assert_eq!(f.lcm(&top, &xh, &mut out), Ok(max), "{f:?} lane {i}");
        assert_eq!(out, top, "{f:?} lane {i}");
        assert_eq!(
            f.lcm(&top, &other, &mut out),
            Err(Overflow),
            "{f:?} lane {i}"
        );
        assert_eq!(f.lcm_degree(&top, &other), max + 1, "{f:?} lane {i}");
        // A signature past the bound in a lane, and so in its degree.
        let (one, _) = mono(&f, &vec![0u32; n]);
        let (xi, _) = mono(&f, &unit(i, 1));
        assert_eq!(
            f.mul_quotient(&top, &one, &xi, &mut out),
            Err(Overflow),
            "{f:?} lane {i}"
        );
        assert_eq!(
            f.mul_quotient(&top, &one, &other, &mut out),
            Err(Overflow),
            "{f:?} lane {i}"
        );
        assert_eq!(
            f.mul_quotient(&top, &xi, &xi, &mut out),
            Ok(max),
            "{f:?} lane {i}"
        );
        assert_eq!(out, top, "{f:?} lane {i}");
        // A packed bound of MAX everywhere admits every monomial.
        let mut all = vec![0u64; f.words()];
        f.pack(&vec![max; n], &mut all);
        assert!(f.divides(&top, &all), "{f:?} lane {i}");
        assert!(
            f.coprime(&top, &other) && !f.coprime(&top, &xh),
            "{f:?} lane {i}"
        );
    }
}

#[test]
fn the_bound_of_each_format() {
    assert_eq!(Dense8::MAX_DEGREE, 127);
    assert_eq!(Dense16::MAX_DEGREE, 32767);
    assert_eq!(Dense32::MAX_DEGREE, 2_147_483_647);
    bound::<Dense8>();
    bound::<Dense16>();
    bound::<Dense32>();
}

// --- test 3 ---

fn borrows<M: Format<Word = u64>>() {
    let l = lanes::<M>();
    let n = 2 * l + 1;
    let f = M::new(n);
    for i in 0..n {
        let a = vec![5u32; n];
        let mut b = a.clone();
        b[i] = 4;
        for j in [i.wrapping_sub(1), i + 1] {
            if j < n {
                b[j] = 12;
            }
        }
        let (ma, _) = mono(&f, &a);
        let (mb, _) = mono(&f, &b);
        assert!(
            !f.divides(&ma, &mb),
            "{f:?}: lane {i} fails, its neighbours pass"
        );
        b[i] = 5;
        let (mb, _) = mono(&f, &b);
        assert!(f.divides(&ma, &mb), "{f:?}: every lane passes");
        // The lane below fails and borrows from lane i, which passes.
        if i > 0 {
            let mut c = vec![5u32; n];
            c[i - 1] = 4;
            c[i] = 6;
            let (mc, _) = mono(&f, &c);
            assert!(
                !f.divides(&ma, &mc),
                "{f:?}: lane {lane} fails below lane {i}",
                lane = i - 1
            );
        }
    }
}

#[test]
fn a_borrow_does_not_hide_a_failing_lane() {
    borrows::<Dense8>();
    borrows::<Dense16>();
    borrows::<Dense32>();
}

// --- test 4 ---

fn table_agrees<M: Format<Word = u64>>(seed: u64) {
    let max = u64::from(M::MAX_DEGREE);
    for n in [7, 8, 9, 17, 54] {
        let mut t = Table::new(M::new(n), &mut rng(seed ^ n as u64));
        let mut draws = rng(seed.rotate_left(17) ^ n as u64);
        let mut ids = Vec::new();
        for k in 0..300 {
            let e = draw(&mut draws, n, if k % 7 == 0 { max } else { max / 2 }, 4);
            let id = t.intern(&e).unwrap();
            ids.push((e, id));
        }
        let back =
            |t: &Table<M>, id: MonoId| -> Vec<u32> { (0..n).map(|i| t.exponent(id, i)).collect() };
        for w in ids.windows(2) {
            let ((a, x), (b, y)) = (&w[0], &w[1]);
            let (x, y) = (*x, *y);
            let case = format!("{n} variables: {a:?} {b:?}");
            let product: Vec<u32> = a.iter().zip(b).map(|(p, q)| p + q).collect();
            let len = t.len();
            if let Ok(p) = t.mul(x, y) {
                assert_eq!(back(&t, p), product, "{case}");
                assert_eq!(t.intern(&product), Ok(p), "{case}");
                assert_eq!(t.div(p, y), Ok(x), "{case}");
                assert_eq!(t.div(p, x), Ok(y), "{case}");
            } else {
                assert!(degree(&product) > max, "{case}");
                assert_eq!(t.len(), len, "{case}: an overflow stores nothing");
            }
            let lcm: Vec<u32> = a.iter().zip(b).map(|(p, q)| *p.max(q)).collect();
            assert_eq!(u64::from(t.lcm_degree(x, y)), degree(&lcm), "{case}");
            match t.lcm(x, y) {
                Ok(l) => {
                    assert_eq!(back(&t, l), lcm, "{case}");
                    assert!(t.divides(x, l) && t.divides(y, l), "{case}");
                }
                Err(Limit::Overflow) => assert!(degree(&lcm) > max, "{case}"),
                Err(Limit::Size) => panic!("{case}: the table is full"),
            }
            assert_eq!(
                t.divides(x, y),
                a.iter().zip(b).all(|(p, q)| p <= q),
                "{case}"
            );
            assert_eq!(
                t.coprime(x, y),
                a.iter().zip(b).all(|(&p, &q)| p == 0 || q == 0),
                "{case}"
            );
            assert_eq!(t.cmp(&Grevlex, x, y), grevlex(a, b), "{case}");
            assert_eq!(t.cmp(&Lex, x, y), lex(a, b), "{case}");
        }
        for (e, id) in &ids {
            assert_eq!(
                &back(&t, *id),
                e,
                "{n} variables: an id still names its monomial"
            );
        }
    }
}

#[test]
fn the_table_agrees_with_interning_in_every_width() {
    table_agrees::<Dense8>(11);
    table_agrees::<Dense16>(12);
    table_agrees::<Dense32>(13);
}

// --- test 5 ---

fn infinity<M: Format<Word = u64>>(seed: u64) {
    let max = M::MAX_DEGREE;
    let mut rng = rng(seed);
    for n in counts(lanes::<M>()) {
        let f = M::new(n);
        let mut all = vec![0u64; f.words()];
        f.pack(&vec![max; n], &mut all);
        let mut none = vec![0u64; f.words()];
        f.pack(&vec![0; n], &mut none);
        let (one, _) = mono(&f, &vec![0u32; n]);
        for _ in 0..100 {
            let mut m = draw(&mut rng, n, u64::from(max) / 2, 4);
            // An exponent at the bound now and then.
            if below(&mut rng, 4) == 0 {
                m.fill(0);
                m[below(&mut rng, n as u64) as usize] = max;
            }
            let (mm, _) = mono(&f, &m);
            assert!(f.divides(&mm, &all), "{f:?}: {m:?} | x^∞");
            assert_eq!(f.divides(&mm, &none), mm == one, "{f:?}: {m:?} | 1");
            let v: Vec<u32> = (0..n)
                .map(|_| {
                    if below(&mut rng, 3) == 0 {
                        max
                    } else {
                        below(&mut rng, 6) as u32
                    }
                })
                .collect();
            let mut packed = vec![0u64; f.words()];
            f.pack(&v, &mut packed);
            assert_eq!(
                f.divides(&mm, &packed),
                m.iter().zip(&v).all(|(x, y)| x <= y),
                "{f:?}: {m:?} | {v:?}"
            );
        }
    }
}

#[test]
fn the_base_divisor_bound_holds_infinity() {
    infinity::<Dense8>(21);
    infinity::<Dense16>(22);
    infinity::<Dense32>(23);
}
