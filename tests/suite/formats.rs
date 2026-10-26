//! monomial-formats: the three monomial formats at their bounds, and calls
//! whose degrees pass a format's bound, through the public API (plan.md,
//! _Test code_). Past degree 65535 the oracle is the checker in the exact
//! mode, with the bases written here (monomial-formats.16).

// monomial-formats.16 design -- past degree 65535 the test code's oracle is the checker in the exact mode, with bases written by hand, not msolve's basis (not measured)
use std::cmp::Ordering;

use grobx::monomial::{Dense8, Dense16, Dense32, Format, Overflow};
use grobx::order::{Block, Grevlex, Lex, Order};
use grobx::{Error, Params};
use grobx_checker::Mode;

use crate::common::{Rng, params, params_in, run};
use crate::orders::accepted_in;

/// The monomial with these exponents and its degree.
fn mono<F: Format>(f: &F, exponents: &[u32]) -> (Vec<F::Word>, u32) {
    let mut m = vec![F::Word::default(); f.words()];
    let d = f.encode(exponents, &mut m).expect("within the format");
    (m, d)
}

fn exponents<F: Format>(f: &F, m: &[F::Word]) -> Vec<u32> {
    (0..f.variables()).map(|i| f.exponent(m, i)).collect()
}

#[test]
fn the_bounds() {
    assert_eq!(Dense8::MAX_DEGREE, 127);
    assert_eq!(Dense16::MAX_DEGREE, 32767);
    assert_eq!(Dense32::MAX_DEGREE, 2_147_483_647);
}

/// Each format reads back what it encodes, at 0 and at its bound.
fn reads_back<F: Format>() {
    let max = F::MAX_DEGREE;
    let f = F::new(4);
    assert_eq!(f.variables(), 4);
    for e in [
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [max, 0, 0, 0],
        [0, 0, 0, max],
        [max / 4, max / 4, max / 4, max - 3 * (max / 4)],
    ] {
        let (m, d) = mono(&f, &e);
        assert_eq!(exponents(&f, &m), e, "{f:?}");
        assert_eq!(
            u64::from(d),
            e.iter().map(|&x| u64::from(x)).sum::<u64>(),
            "{f:?}"
        );
    }
}

#[test]
fn each_format_reads_back() {
    reads_back::<Dense8>();
    reads_back::<Dense16>();
    reads_back::<Dense32>();
}

/// `encode`, `mul`, and `lcm` at the bound hold it, and one past it is an
/// overflow; `div` below it.
fn bound<F: Format>() {
    let max = F::MAX_DEGREE;
    let half = max / 2 + 1; // half + half = max + 1, max being odd
    let f = F::new(2);
    let mut out = vec![F::Word::default(); f.words()];
    assert_eq!(f.encode(&[max, 0], &mut out), Ok(max), "{f:?}");
    assert_eq!(f.encode(&[0, max], &mut out), Ok(max), "{f:?}");
    assert_eq!(f.encode(&[max + 1, 0], &mut out), Err(Overflow), "{f:?}");
    assert_eq!(f.encode(&[half, half], &mut out), Err(Overflow), "{f:?}");
    assert_eq!(f.encode(&[u32::MAX, 0], &mut out), Err(Overflow), "{f:?}");
    assert_eq!(
        f.encode(&[u32::MAX, u32::MAX], &mut out),
        Err(Overflow),
        "{f:?}"
    );

    let (xh, dxh) = mono(&f, &[half, 0]);
    let (yh, dyh) = mono(&f, &[0, half]);
    let (xl, dxl) = mono(&f, &[max - half, 0]);
    assert_eq!(f.mul(&xh, dxh, &yh, dyh, &mut out), Err(Overflow), "{f:?}");
    assert_eq!(f.mul(&xh, dxh, &xh, dxh, &mut out), Err(Overflow), "{f:?}");
    assert_eq!(f.mul(&xh, dxh, &xl, dxl, &mut out), Ok(max), "{f:?}");
    assert_eq!(exponents(&f, &out), [max, 0], "{f:?}");
    assert_eq!(f.lcm(&xh, &yh, &mut out), Err(Overflow), "{f:?}");
    let (xm, _) = mono(&f, &[max, 0]);
    assert_eq!(f.lcm(&xm, &xh, &mut out), Ok(max), "{f:?}");
    assert_eq!(exponents(&f, &out), [max, 0], "{f:?}");
    assert_eq!(f.div(&xm, &xh, &mut out), max - half, "{f:?}");
    assert_eq!(exponents(&f, &out), [max - half, 0], "{f:?}");
    assert!(f.divides(&xh, &xm) && !f.divides(&xm, &xh), "{f:?}");
    assert!(f.is_product(&xh, &xl, &xm), "{f:?}");
}

#[test]
fn each_format_detects_its_bound() {
    bound::<Dense8>();
    bound::<Dense16>();
    bound::<Dense32>();
}

/// A vector of n exponents whose degree is at most `degree`.
fn draw(rng: &mut Rng, n: usize, degree: u64) -> Vec<u32> {
    let mut left = rng.below(degree + 1);
    let mut e = vec![0u32; n];
    for _ in 0..2 * n {
        let i = rng.below(n as u64) as usize;
        let k = rng.below(left.min(9) + 1);
        e[i] += k as u32;
        left -= k;
    }
    e
}

/// What one format computes on a and b: the exponents of a·b and of
/// lcm(a, b), whether a divides b, whether a·b is their product, and the
/// masks of a and b. The quotient a·b / b is a.
fn operations<F: Format>(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>, bool, bool, u32, u32) {
    let f = F::new(a.len());
    let (ma, da) = mono(&f, a);
    let (mb, db) = mono(&f, b);
    let mut out = vec![F::Word::default(); f.words()];
    f.mul(&ma, da, &mb, db, &mut out).unwrap();
    let product = exponents(&f, &out);
    let (mp, _) = mono(&f, &product);
    assert_eq!(f.div(&mp, &mb, &mut out), da, "{f:?}");
    assert_eq!(exponents(&f, &out), a, "{f:?}");
    f.lcm(&ma, &mb, &mut out).unwrap();
    let lcm = exponents(&f, &out);
    (
        product,
        lcm,
        f.divides(&ma, &mb),
        f.is_product(&ma, &mb, &mp),
        f.mask(&ma),
        f.mask(&mb),
    )
}

/// Products, quotients, lcms, divisibility, and masks agree across the
/// formats on vectors all three hold, and a mask never rejects a divisor,
/// on 1 to 54 variables. The degrees keep a·b within `Dense8`.
#[test]
fn the_formats_agree() {
    let mut rng = Rng::new(0xf0a7);
    for n in [1, 2, 3, 5, 16, 31, 32, 33, 54] {
        let mut divisors = 0;
        let degree = 60.min((127 - 2 * n as u64) / 2);
        for _ in 0..1500 {
            let a = draw(&mut rng, n, degree);
            let b: Vec<u32> = if rng.below(2) == 0 {
                a.iter().map(|&e| e + rng.below(3) as u32).collect()
            } else {
                draw(&mut rng, n, degree)
            };
            let narrow = operations::<Dense8>(&a, &b);
            assert_eq!(narrow, operations::<Dense16>(&a, &b), "{a:?} {b:?}");
            assert_eq!(narrow, operations::<Dense32>(&a, &b), "{a:?} {b:?}");
            let (_, lcm, divides, product, mask_a, mask_b) = narrow;
            assert!(product);
            let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| *x.max(y)).collect();
            assert_eq!(lcm, want);
            assert_eq!(divides, a.iter().zip(&b).all(|(x, y)| x <= y));
            if divides {
                divisors += 1;
                assert_eq!(mask_a & !mask_b, 0, "{n} variables: {a:?} | {b:?}");
            }
        }
        assert!(divisors > 100, "{n} variables: too few divisors drawn");
    }
}

fn cmp<F: Format, O: Order<F>>(order: &O, a: &[u32], b: &[u32]) -> Ordering {
    let f = F::new(a.len());
    let (ma, da) = mono(&f, a);
    let (mb, db) = mono(&f, b);
    order.cmp(&f, &ma, da, &mb, db)
}

/// Grevlex, lex, and a block order compare alike in every format, and
/// grevlex respects products in `Dense8`, and in `Dense32` at a thousand
/// times the exponents.
#[test]
fn the_orders_agree_across_the_formats() {
    let mut rng = Rng::new(0x0bde);
    let block = Block::new(&[2], 4).unwrap();
    for _ in 0..3000 {
        let (a, b, c) = (
            draw(&mut rng, 4, 80),
            draw(&mut rng, 4, 80),
            draw(&mut rng, 4, 80),
        );
        for (x, y) in [(&a, &b), (&b, &c), (&a, &a)] {
            let g = cmp::<Dense8, _>(&Grevlex, x, y);
            assert_eq!(g, cmp::<Dense16, _>(&Grevlex, x, y), "{x:?} {y:?}");
            assert_eq!(g, cmp::<Dense32, _>(&Grevlex, x, y), "{x:?} {y:?}");
            let l = cmp::<Dense8, _>(&Lex, x, y);
            assert_eq!(l, cmp::<Dense16, _>(&Lex, x, y), "{x:?} {y:?}");
            assert_eq!(l, cmp::<Dense32, _>(&Lex, x, y), "{x:?} {y:?}");
            let k = cmp::<Dense8, _>(&block, x, y);
            assert_eq!(k, cmp::<Dense16, _>(&block, x, y), "{x:?} {y:?}");
            assert_eq!(k, cmp::<Dense32, _>(&block, x, y), "{x:?} {y:?}");
        }
        let ac: Vec<u32> = a.iter().zip(&c).map(|(x, y)| x + y).collect();
        let bc: Vec<u32> = b.iter().zip(&c).map(|(x, y)| x + y).collect();
        assert_eq!(
            cmp::<Dense8, _>(&Grevlex, &a, &b),
            cmp::<Dense8, _>(&Grevlex, &ac, &bc)
        );
        let wide = |v: &[u32]| -> Vec<u32> { v.iter().map(|&e| e * 1000).collect() };
        assert_eq!(
            cmp::<Dense32, _>(&Grevlex, &wide(&a), &wide(&b)),
            cmp::<Dense32, _>(&Grevlex, &wide(&ac), &wide(&bc))
        );
    }
}

/// Inputs with a term past `Dense8`'s bound 127 and past `Dense16`'s 32767,
/// and at 2^31 − 1, are held, and the checker accepts their bases with
/// cofactors; `text::generators` reads them.
#[test]
fn the_input_picks_a_wider_format() {
    let cases = [
        (
            "x\n65521\nx^256-1\n",
            "[x^256+65520]:\n",
            "[x^256+65520]:\n",
        ),
        (
            "x\n65521\nx^65535-1\n",
            "[x^65535+65520]:\n",
            "[x^65535+65520]:\n",
        ),
        (
            "x\n65521\nx^65536-1\n",
            "[x^65536+65520]:\n",
            "[x^65536+65520]:\n",
        ),
        (
            "x,y\n65521\nx^2147483647-y\n",
            "[x^2147483647+65520*y]:\n",
            "[x^2147483647+65520*y]:\n",
        ),
        (
            "x,y\n7\nx^1073741823*y^1073741824+1\n",
            "[x^1073741823*y^1073741824+1]:\n",
            "[x^1073741823*y^1073741824+1]:\n",
        ),
    ];
    for (text, basis, generators) in cases {
        let outcome = accepted_in("input", text, "grevlex", Mode::Exact);
        assert_eq!(outcome.basis.as_deref(), Some(basis), "{text:?}");
        assert_eq!(
            grobx::text::generators(text).unwrap(),
            generators,
            "{text:?}"
        );
    }
}

/// Past 2^31 − 1 in the input, the line; in the run, none.
#[test]
fn past_the_widest_format() {
    let input = [
        ("x\n7\nx^2147483648\n", 3),
        ("x,y\n7\nx,\nx^1073741824*y^1073741824\n", 4),
        ("x,y\n7\nx^1073741824*y*x^1073741824\n", 3),
        ("x\n7\nx^99999999999999999999999\n", 3),
    ];
    for (text, line) in input {
        let got = grobx::groebner(text, &params(1));
        assert!(
            matches!(got, Err(Error::Overflow { line: Some(l) }) if l == line),
            "{text:?}: {got:?}"
        );
        let read = grobx::text::generators(text);
        assert!(
            matches!(read, Err(Error::Overflow { line: Some(l) }) if l == line),
            "{text:?}: {read:?}"
        );
    }
    let text = "x,y\n65521\nx^1500000000*y-1,\nx*y^1500000000-1\n";
    let mut calls: Vec<Params> = ["grevlex", "lex", "block:1"]
        .iter()
        .map(|o| params_in(o, 1))
        .collect();
    let mut off = params(1);
    off.signatures = false;
    calls.push(off);
    let mut cof = params(1);
    cof.cofactors = true;
    calls.push(cof);
    for p in calls {
        let got = grobx::groebner(text, &p);
        assert!(
            matches!(got, Err(Error::Overflow { line: None })),
            "{order} signatures {signatures} cofactors {cofactors}: {got:?}",
            order = p.order,
            signatures = p.signatures,
            cofactors = p.cofactors
        );
    }
}

/// The pair's lcm has degree 200, past `Dense8`'s 127, and 40000, past
/// `Dense16`'s 32767, where each input fits: each call goes on and returns
/// the basis written here, signatures on and off, accepted with its
/// cofactors.
#[test]
fn a_run_goes_on_past_127_and_32767() {
    let cases = [
        (
            "x,y\n65521\nx^100*y-1,\nx*y^100-1\n",
            "[x^99+65520*y^99,\nx*y^100+65520,\ny^199+65520*x^98]:\n",
        ),
        (
            "x,y\n65521\nx^20000*y-1,\nx*y^20000-1\n",
            "[x^19999+65520*y^19999,\nx*y^20000+65520,\ny^39999+65520*x^19998]:\n",
        ),
    ];
    for (text, want) in cases {
        for signatures in [true, false] {
            let mut p = params(1);
            p.signatures = signatures;
            assert_eq!(run(text, &p).basis.as_deref(), Some(want), "{text:?}");
        }
        let outcome = accepted_in("past", text, "grevlex", Mode::Exact);
        assert_eq!(outcome.basis.as_deref(), Some(want), "{text:?}");
    }
}

/// The lex basis holds y^90000 − y, past `Dense16`'s degree 32767: the call
/// goes on in `Dense32`, in lex and in `block:1`, which is lex on two
/// variables. D = 90000 passes the change's 65535, and F4 in lex makes the
/// basis (monomial-formats.11). The grevlex basis is the input.
#[test]
fn lex_goes_on_past_65535() {
    let text = "x,y\n65521\nx^300-y,\ny^300-x\n";
    for order in ["lex", "block:1"] {
        let outcome = accepted_in("x300", text, order, Mode::Exact);
        assert_eq!(
            outcome.basis.as_deref(),
            Some("[y^90000+65520*y,\nx+65520*y^300]:\n"),
            "{order}"
        );
    }
    assert_eq!(
        run(text, &params_in("grevlex", 1)).basis.unwrap(),
        "[y^300+65520*x,\nx^300+65520*y]:\n"
    );
}

/// D = 127 fits `Dense8`, and D = 128 does not: each lex call returns the
/// basis written here, accepted in the exact mode, whether or not the call
/// starts over.
#[test]
fn lex_on_a_staircase_of_127_and_of_128() {
    let cases = [
        ("x,y\n65521\nx-1,\ny^127-1\n", "[y^127+65520,\nx+65520]:\n"),
        ("x,y\n65521\nx^8-1,\ny^16-1\n", "[y^16+65520,\nx^8+65520]:\n"),
    ];
    for (text, want) in cases {
        let outcome = accepted_in("staircase", text, "lex", Mode::Exact);
        assert_eq!(outcome.basis.as_deref(), Some(want), "{text:?}");
    }
}

/// A record of a run that went on in `Dense16` replays over another prime,
/// and its replay goes on as the run did.
#[test]
fn a_replay_goes_on_too() {
    let text = "x,y\n65521\nx^100*y-1,\nx*y^100-1\n";
    let (learned, record) = grobx::learn(text, &params(1), &mut grobx::trace::NoTrace).unwrap();
    let record = record.expect("a basis has a record");
    let other = text.replace("65521", "2147483647");
    let replayed = grobx::replay(&other, &params(2), &record, &mut grobx::trace::NoTrace).unwrap();
    assert_eq!(replayed.basis, run(&other, &params(2)).basis);
    assert_eq!(
        learned.basis.as_deref(),
        Some("[x^99+65520*y^99,\nx*y^100+65520,\ny^199+65520*x^98]:\n")
    );
}
