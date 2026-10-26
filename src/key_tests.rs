//! matrix-sort-time, the key's and the sorts' test code: the grevlex key
//! as written, its order against grevlex's comparison, and the two sorts of
//! a matrix against the comparison sort (plan.md, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;
use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::id::{MonoId, RowId};
use crate::matrix::{by_signature, sort_decreasing};
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::sig::Sig;
use crate::table::Table;
use crate::{MonomialOrder, Params, Threads};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// A budget with no bound, for the sorts' spends.
fn budget() -> Budget {
    let params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    Budget::new(Instant::now(), &params).unwrap()
}

/// The grevlex key of the monomial with these exponents, in the format `M`
/// on as many variables.
fn key<M: Format>(exponents: &[u32]) -> u128 {
    let f = M::new(exponents.len());
    let mut m = vec![M::Word::default(); f.words()];
    let degree = f.encode(exponents, &mut m).expect("within the format");
    let key = f.grevlex_key(&m, degree);
    assert_eq!(
        <Grevlex as Order<M>>::key(&Grevlex, &f, &m, degree),
        key,
        "{exponents:?}"
    );
    key
}

/// Keys written out by hand: the degree, then each exponent from the last
/// variable down, complemented, a lane's bits each, the first 96 bits.
#[test]
fn the_key_as_written() {
    // x^2·y^3·z^4: the degree 9, then ~4, ~3, ~2.
    assert_eq!(key::<Dense8>(&[2, 3, 4]), 0x09_FB_FC_FD << 96);
    assert_eq!(key::<Dense16>(&[2, 3, 4]), 0x0009_FFFB_FFFC_FFFD << 64);
    // In `Dense32` the key is cut after ~3.
    assert_eq!(
        key::<Dense32>(&[2, 3, 4]),
        0x0000_0009_FFFF_FFFB_FFFF_FFFC << 32
    );
    // The unit monomial: degree 0, every field all ones.
    assert_eq!(key::<Dense8>(&[0, 0, 0]), 0x00_FF_FF_FF << 96);
    assert_eq!(key::<Dense16>(&[0]), 0x0000_FFFF << 96);
    // At the degree bounds.
    assert_eq!(key::<Dense8>(&[127, 0]), 0x7F_FF_80 << 104);
    assert_eq!(key::<Dense16>(&[0, 32767]), 0x7FFF_8000_FFFF << 80);
    assert_eq!(
        key::<Dense32>(&[i32::MAX as u32]),
        0x7FFF_FFFF_8000_0000 << 64
    );
    // Nine variables in `Dense8`, two words, one lane in the last: e_i = i + 1.
    let e: Vec<u32> = (1..=9).collect();
    assert_eq!(key::<Dense8>(&e), 0x2D_F6_F7_F8_F9_FA_FB_FC_FD_FE << 48);
    // Thirteen variables, e_i = i: the degree 78, then ~12 down to ~2; ~1
    // and ~0 are cut.
    let e: Vec<u32> = (0..13).collect();
    assert_eq!(
        key::<Dense8>(&e),
        0x4E_F3_F4_F5_F6_F7_F8_F9_FA_FB_FC_FD << 32
    );
    // Seven variables in `Dense16`, two words: the degree 7, then ~1 for
    // the last five variables; the first two are cut.
    assert_eq!(
        key::<Dense16>(&[1; 7]),
        0x0007_FFFE_FFFE_FFFE_FFFE_FFFE << 32
    );
    const {
        assert!(<Grevlex as Order<Dense8>>::KEYED);
        assert!(!<Lex as Order<Dense8>>::KEYED);
        assert!(!<Block as Order<Dense8>>::KEYED);
    }
}

/// Exponents of n variables: most of the last ones 0, the first ones
/// small, so that many monomials share their degree and their last
/// variables; every exponent at most `top`.
fn draw(r: &mut Xoshiro256PlusPlus, n: usize, top: u32) -> Vec<u32> {
    let spread = r.next_u64().is_multiple_of(3);
    (0..n)
        .map(|i| {
            let x = r.next_u64();
            let e = if i < 3 || spread {
                (x % 4) as u32
            } else {
                u32::from(x.is_multiple_of(6))
            };
            e.min(top)
        })
        .collect()
}

/// A table of format `M` on n variables holding drawn monomials, the unit
/// monomial, and the first and the last variable's powers at the degree
/// bound: their ids, distinct, increasing.
fn monomials<M: Format>(n: usize, count: usize, seed: u64) -> (Table<M>, Vec<MonoId>) {
    let mut t = Table::new(M::new(n), &mut rng(seed));
    let mut r = rng(seed ^ 0x6b65);
    let top = (M::MAX_DEGREE / n as u32).max(1);
    let mut ids = vec![t.intern(&vec![0; n]).unwrap()];
    for i in [0, n - 1] {
        let mut e = vec![0; n];
        e[i] = M::MAX_DEGREE;
        ids.push(t.intern(&e).unwrap());
    }
    for _ in 0..count {
        let e = draw(&mut r, n, top);
        ids.push(t.intern(&e).unwrap());
    }
    ids.sort_unstable();
    ids.dedup();
    (t, ids)
}

/// For every pair of distinct monomials: the smaller key is the smaller
/// monomial in grevlex, and where n lanes of `bits` fit 96 bits the keys
/// differ.
fn orders<M: Format>(n: usize, bits: u32) {
    let (t, ids) = monomials::<M>(n, 300, 0x6f72 ^ n as u64);
    let exact = n as u32 * bits <= 96;
    let mut agree = 0;
    for &a in &ids {
        let ka = t.key(&Grevlex, a);
        assert_eq!(ka as u32, 0, "{n} variables: the low 32 bits");
        for &b in &ids {
            if a == b {
                continue;
            }
            let kb = t.key(&Grevlex, b);
            let want = t.cmp(&Grevlex, a, b);
            match ka.cmp(&kb) {
                Ordering::Equal => {
                    assert!(
                        !exact,
                        "{n} variables of {bits} bits: {a} and {b} share a key"
                    );
                    agree += 1;
                }
                got @ (Ordering::Less | Ordering::Greater) => {
                    assert_eq!(got, want, "{n} variables of {bits} bits: {a} against {b}");
                }
            }
        }
    }
    if !exact {
        assert!(agree > 0, "{n} variables of {bits} bits: no two keys agree");
    }
}

#[test]
fn the_key_orders_as_grevlex() {
    for n in [1, 2, 7, 8, 9, 12, 13, 16, 17, 24, 54] {
        orders::<Dense8>(n, 8);
    }
    for n in [1, 3, 4, 5, 6, 7, 9] {
        orders::<Dense16>(n, 16);
    }
    for n in [1, 2, 3, 4, 5] {
        orders::<Dense32>(n, 32);
    }
}

/// The ids shuffled.
fn shuffled(ids: &[MonoId], r: &mut Xoshiro256PlusPlus) -> Vec<MonoId> {
    let mut ids = ids.to_vec();
    for i in (1..ids.len()).rev() {
        let j = (r.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    ids
}

/// `sort_decreasing` returns the order the comparison sort returns, and
/// `by_signature` the rows sorted by `Sig::cmp` and then by row.
fn sorts<M: Format, O: Order<M>>(order: &O, n: usize, seed: u64) {
    let (mut t, ids) = monomials::<M>(n, 400, seed);
    let mut r = rng(seed ^ 0x736f);
    let mut sets = vec![shuffled(&ids, &mut r)];
    // A set whose keys all agree in their first 96 bits where the key leaves
    // out the first two variables: one degree, only those two varying.
    let agreeing: Vec<MonoId> = (0..=6u32)
        .map(|a| {
            let mut e = vec![0; n];
            (e[0], e[1]) = (6 - a, a);
            t.intern(&e).unwrap()
        })
        .collect();
    sets.push(shuffled(&agreeing, &mut r));
    let all = sets[0].clone();
    sets.push(Vec::new());
    sets.push(all[..1].to_vec());
    sets.push(all[..2].to_vec());
    for set in &sets {
        let mut got = set.clone();
        sort_decreasing(&t, order, &mut got, &mut budget()).unwrap();
        let mut want = set.clone();
        want.sort_by(|&a, &b| t.cmp(order, b, a));
        assert_eq!(
            got,
            want,
            "{order:?} on {n} variables, {len} ids",
            len = set.len()
        );
    }
    // Signatures over few monomials and indices: repeated monomials,
    // repeated indices, and equal signatures.
    let few: Vec<MonoId> = all.iter().copied().take(40).chain(agreeing).collect();
    let sigs: Vec<Sig> = (0..600)
        .map(|_| Sig {
            monomial: few[(r.next_u64() % few.len() as u64) as usize],
            index: (r.next_u64() % 3) as u32,
        })
        .collect();
    for len in [0, 1, 2, 17, sigs.len()] {
        let sigs = &sigs[..len];
        let got = by_signature(&t, order, sigs, &mut budget()).unwrap();
        let mut want: Vec<RowId> = (0..len).map(RowId::at).collect();
        want.sort_by(|&a, &b| {
            sigs[a.index()]
                .cmp(sigs[b.index()], &t, order)
                .then(a.cmp(&b))
        });
        assert_eq!(got, want, "{order:?} on {n} variables, {len} rows");
    }
}

fn each_order<M: Format>(n: usize, seed: u64) {
    sorts::<M, _>(&Grevlex, n, seed);
    sorts::<M, _>(&Lex, n, seed ^ 1);
    let block = Block::new(&[2], n).expect("two variables and a last block");
    sorts::<M, _>(&block, n, seed ^ 2);
}

#[test]
fn the_sorts_agree_with_the_comparison() {
    for n in [3, 9, 13, 17] {
        each_order::<Dense8>(n, 0x100 + n as u64);
        each_order::<Dense16>(n, 0x200 + n as u64);
        each_order::<Dense32>(n, 0x300 + n as u64);
    }
    each_order::<Dense16>(8, 0x208);
}
