//! Sub-item 1: the monomial contract with `Dense16`, dense `u16`
//! exponents whose overflow the total degree detects (f4-gfp.11 to
//! f4-gfp.13); the divisor mask (f4-gfp.17, f4-gfp.18); grevlex in the order
//! family; and the module of rank 1 (f4-gfp.40).

use std::cmp::Ordering;

use grobx::module::{Module, Rank1};
use grobx::monomial::{Dense16, Format, Overflow};
use grobx::order::{Grevlex, Order};

use crate::common::Rng;

type Word = <Dense16 as Format>::Word;

fn mono(f: &Dense16, exponents: &[u32]) -> (Vec<Word>, u32) {
    let mut m = vec![Word::default(); f.words()];
    let d = f.encode(exponents, &mut m).expect("within the format");
    (m, d)
}

fn exponents(f: &Dense16, m: &[Word]) -> Vec<u32> {
    (0..f.variables()).map(|i| f.exponent(m, i)).collect()
}

#[test]
fn encode_reads_back() {
    let f = Dense16::new(4);
    assert_eq!(f.variables(), 4);
    for e in [
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [32767, 0, 0, 0],
        [0, 0, 0, 32767],
        [100, 200, 300, 400],
    ] {
        let (m, d) = mono(&f, &e);
        assert_eq!(exponents(&f, &m), e);
        assert_eq!(d, e.iter().sum::<u32>());
    }
}

#[test]
fn products_quotients_and_lcms() {
    let f = Dense16::new(3);
    let (a, da) = mono(&f, &[2, 1, 0]); // x²y
    let (b, db) = mono(&f, &[1, 0, 1]); // xz
    let mut out = vec![Word::default(); f.words()];
    assert_eq!(f.mul(&a, da, &b, db, &mut out), Ok(5));
    assert_eq!(exponents(&f, &out), [3, 1, 1]);
    let (c, _) = mono(&f, &[3, 1, 1]);
    assert_eq!(f.div(&c, &b, &mut out), 3);
    assert_eq!(exponents(&f, &out), [2, 1, 0]);
    let (e, _) = mono(&f, &[1, 3, 1]); // xy³z
    assert_eq!(f.lcm(&a, &e, &mut out), Ok(6));
    assert_eq!(exponents(&f, &out), [2, 3, 1]);
    let (one, _) = mono(&f, &[0, 0, 0]);
    assert!(f.divides(&one, &a));
    assert!(f.divides(&b, &c));
    assert!(f.divides(&a, &a));
    assert!(!f.divides(&a, &b));
    assert!(!f.divides(&a, &one));
}

#[test]
fn degrees_past_32767_overflow() {
    let f = Dense16::new(2);
    let mut out = vec![Word::default(); f.words()];
    assert_eq!(f.encode(&[32767, 0], &mut out), Ok(32767));
    assert_eq!(f.encode(&[32768, 0], &mut out), Err(Overflow));
    assert_eq!(f.encode(&[20000, 15000], &mut out), Err(Overflow));
    assert_eq!(f.encode(&[u32::MAX, 0], &mut out), Err(Overflow));
    let (x20, d20) = mono(&f, &[20000, 0]);
    let (y15, d15) = mono(&f, &[0, 15000]);
    assert_eq!(f.mul(&x20, d20, &y15, d15, &mut out), Err(Overflow));
    let (x15, d15x) = mono(&f, &[15000, 0]);
    let (x17, d17) = mono(&f, &[17767, 0]);
    assert_eq!(f.mul(&x15, d15x, &x17, d17, &mut out), Ok(32767));
    assert_eq!(exponents(&f, &out), [32767, 0]);
    let (y20, _) = mono(&f, &[0, 20000]);
    assert_eq!(f.lcm(&x20, &y20, &mut out), Err(Overflow));
    let (x32767, _) = mono(&f, &[32767, 0]);
    assert_eq!(f.lcm(&x32767, &x32767, &mut out), Ok(32767));
    assert_eq!(Dense16::MAX_DEGREE, 32767);
}

/// a | b implies mask(a) & !mask(b) == 0, for few variables and for more
/// than the mask's 32 bits.
#[test]
fn the_mask_never_rejects_a_divisor() {
    let mut rng = Rng::new(0xd1f);
    for n in [1, 2, 3, 5, 16, 31, 32, 33, 54] {
        let f = Dense16::new(n);
        let mut divisors = 0;
        for _ in 0..3000 {
            let a: Vec<u32> = (0..n).map(|_| rng.below(4) as u32).collect();
            let b: Vec<u32> = a
                .iter()
                .map(|&e| {
                    if rng.below(4 * n as u64) == 0 {
                        rng.below(5) as u32
                    } else {
                        e + rng.below(3) as u32
                    }
                })
                .collect();
            let (ma, _) = mono(&f, &a);
            let (mb, _) = mono(&f, &b);
            let divides = a.iter().zip(&b).all(|(x, y)| x <= y);
            assert_eq!(f.divides(&ma, &mb), divides, "{a:?} | {b:?}");
            if divides {
                divisors += 1;
                assert_eq!(
                    f.mask(&ma) & !f.mask(&mb),
                    0,
                    "{n} variables: {a:?} | {b:?}"
                );
            }
        }
        assert!(divisors > 100, "{n} variables: too few divisors drawn");
    }
}

fn cmp(f: &Dense16, a: &[u32], b: &[u32]) -> Ordering {
    let (ma, da) = mono(f, a);
    let (mb, db) = mono(f, b);
    Grevlex.cmp(f, &ma, da, &mb, db)
}

/// x > y > z, the first variable the largest.
#[test]
fn grevlex_orders_the_textbook_examples() {
    let f = Dense16::new(3);
    let descending = [
        [3, 0, 0],
        [1, 2, 0], // xy² before x²z: the last differing variable decides
        [2, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ];
    for (i, a) in descending.iter().enumerate() {
        assert_eq!(cmp(&f, a, a), Ordering::Equal);
        for b in &descending[i + 1..] {
            assert_eq!(cmp(&f, a, b), Ordering::Greater, "{a:?} > {b:?}");
            assert_eq!(cmp(&f, b, a), Ordering::Less, "{b:?} < {a:?}");
        }
    }
}

/// Grevlex is a monomial order: it respects products.
#[test]
fn grevlex_respects_products() {
    let mut rng = Rng::new(0x9e7);
    let f = Dense16::new(4);
    let mut out_a = vec![Word::default(); f.words()];
    let mut out_b = vec![Word::default(); f.words()];
    for _ in 0..2000 {
        let draw = |rng: &mut Rng| -> Vec<u32> { (0..4).map(|_| rng.below(5) as u32).collect() };
        let (a, b, c) = (draw(&mut rng), draw(&mut rng), draw(&mut rng));
        let (ma, da) = mono(&f, &a);
        let (mb, db) = mono(&f, &b);
        let (mc, dc) = mono(&f, &c);
        let dac = f.mul(&ma, da, &mc, dc, &mut out_a).unwrap();
        let dbc = f.mul(&mb, db, &mc, dc, &mut out_b).unwrap();
        assert_eq!(
            Grevlex.cmp(&f, &ma, da, &mb, db),
            Grevlex.cmp(&f, &out_a, dac, &out_b, dbc),
            "{a:?}, {b:?} times {c:?}"
        );
    }
}

#[test]
fn rank_one_positions_cost_nothing() {
    assert_eq!(Rank1.rank(), 1);
    assert_eq!(std::mem::size_of::<<Rank1 as Module>::Position>(), 0);
}
