//! sort-share-time, sub-item 1, the crate's test code: the two sorts of a
//! matrix on sets whose elements agree in their first 96 bits, and in
//! their first keys and tie keys, against the comparison sort, and the tie
//! key's order among elements whose first keys agree
//! (`iterations/0105-sort-share-time/plan.md`, _Test code_, 1 and 2; the
//! tie key stands where the plan's second key stood, log 0105-2).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;

use rand_core::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, Site};
use crate::id::{MonoId, RowId};
use crate::monomial::{Dense8, Dense16, Dense32, Format, KEY_BITS, PAYLOAD_BITS};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::shared_tests::{budget, rng};
use crate::sig::Sig;
use crate::sort::{by_signature, sort_decreasing};
use crate::table::Table;

/// The variables the first key holds on lanes of `bits` bits, the last
/// ones after the degree's lane, and the lanes of a word.
fn per_key(bits: u32) -> (usize, usize) {
    ((KEY_BITS / bits) as usize - 1, (64 / bits) as usize)
}

/// Exponents of n variables of total degree `degree`, spread over the
/// variables `from..to` alone: monomials of one set agree in their degree
/// and in every variable outside the window, so their keys agree wherever
/// the key reads no variable of the window.
fn in_window(
    r: &mut Xoshiro256PlusPlus,
    n: usize,
    from: usize,
    to: usize,
    degree: u32,
) -> Vec<u32> {
    let mut e = vec![0u32; n];
    for _ in 0..degree {
        let i = from + (r.next_u64() % (to - from) as u64) as usize;
        e[i] += 1;
    }
    e
}

/// A table on n variables holding `count` drawn monomials of each window
/// in `windows`, at degree `degree`, distinct: their ids, and the windows'
/// sets of ids, each shuffled.
fn windows<M: Format>(
    n: usize,
    windows: &[(usize, usize)],
    degree: u32,
    count: usize,
    seed: u64,
) -> (Table<M>, Vec<Vec<MonoId>>) {
    let mut t = Table::new(M::new(n), &mut rng(seed));
    let mut r = rng(seed ^ 0x7469);
    let mut sets = Vec::new();
    for &(from, to) in windows {
        let mut ids: Vec<MonoId> = (0..count)
            .map(|_| {
                t.intern(
                    &in_window(&mut r, n, from, to, degree),
                    &mut Budget::unbounded(),
                )
                .unwrap()
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        for i in (1..ids.len()).rev() {
            let j = (r.next_u64() % (i as u64 + 1)) as usize;
            ids.swap(i, j);
        }
        sets.push(ids);
    }
    (t, sets)
}

/// The elements of `ids` whose first keys agree with another's in grevlex,
/// and with `deep` whose tie keys agree with it too.
fn tied(t: &Table<impl Format>, ids: &[MonoId], deep: bool) -> usize {
    let mut keys: Vec<(u128, u128)> = ids
        .iter()
        .map(|&u| {
            let tie = if deep {
                t.tie_key(&Grevlex, u, None)
            } else {
                0
            };
            (t.key(&Grevlex, u), tie)
        })
        .collect();
    keys.sort_unstable();
    keys.windows(2).filter(|w| w[0] == w[1]).count()
}

/// Test 1: `sort_decreasing` and `by_signature` return the comparison
/// sort's order on sets whose elements agree in their first 96 bits, and
/// in their first keys and tie keys, in every order; and on sets of 0, 1,
/// and 2.
fn sorts_agree<M: Format, O: Order<M>>(order: &O, n: usize, bits: u32, seed: u64, deep: bool) {
    // The first key reads the last variables. Monomials of one degree
    // spread over the variables before them agree in their first keys;
    // spread over the variables before the next word's as well, their tie
    // keys read low words, which agree often; spread over every variable,
    // the first keys decide most.
    let (first, lanes) = per_key(bits);
    let mut w = Vec::new();
    if n > first + lanes {
        w.push((0, n - first - lanes));
    }
    if n > first {
        w.push((0, n - first));
    }
    w.push((0, n));
    let (mut t, sets) = windows::<M>(n, &w, 6, 300, seed);
    let mut all: Vec<MonoId> = sets.concat();
    assert!(
        tied(&t, &all, false) > 0,
        "{n} variables of {bits} bits: no two first keys agree"
    );
    if deep {
        assert!(
            tied(&t, &all, true) > 0,
            "{n} variables of {bits} bits: no two tie keys agree"
        );
    }
    let small: Vec<Vec<MonoId>> = vec![Vec::new(), all[..1].to_vec(), all[..2].to_vec()];
    for set in sets.iter().chain(&small).chain(std::iter::once(&all)) {
        let mut got = set.clone();
        sort_decreasing(&t, order, &mut got, &mut budget(), Site::Columns).unwrap();
        let mut want = set.clone();
        want.sort_by(|&a, &b| t.cmp(order, b, a));
        assert_eq!(
            got,
            want,
            "{order:?} on {n} variables, {len} ids",
            len = set.len()
        );
    }
    // Signatures over the tied monomials: repeated monomials and indices.
    let mut r = rng(seed ^ 0x7369);
    all.truncate(60);
    let unit = t.intern(&vec![0; n], &mut Budget::unbounded()).unwrap();
    all.push(unit);
    let sigs: Vec<Sig> = (0..500)
        .map(|_| Sig {
            monomial: all[(r.next_u64() % all.len() as u64) as usize],
            index: (r.next_u64() % 3) as u32,
        })
        .collect();
    for len in [0, 1, 2, 23, sigs.len()] {
        let sigs = &sigs[..len];
        let got = by_signature(&t, order, sigs, &mut budget(), Site::SigSort).unwrap();
        let mut want: Vec<RowId> = (0..len).map(RowId::at).collect();
        want.sort_by(|&a, &b| {
            sigs[a.index()]
                .cmp(sigs[b.index()], &t, order)
                .then(a.cmp(&b))
        });
        assert_eq!(got, want, "{order:?} on {n} variables, {len} rows");
    }
}

fn each_order<M: Format>(n: usize, bits: u32, seed: u64, deep: bool) {
    sorts_agree::<M, _>(&Grevlex, n, bits, seed, deep);
    sorts_agree::<M, _>(&Lex, n, bits, seed ^ 1, deep);
    let block = Block::new(&[2], n).expect("two variables and a last block");
    sorts_agree::<M, _>(&block, n, bits, seed ^ 2, deep);
}

#[test]
fn the_sorts_agree_with_the_comparison_on_ties() {
    // Where the variables the first key leaves fit one word, the tie key
    // holds every one of them, and two monomials of one degree agree in
    // both only where they are one.
    for (n, deep) in [(13, false), (17, false), (24, true), (54, true)] {
        each_order::<Dense8>(n, 8, 0x5400 + n as u64, deep);
    }
    for (n, deep) in [(9, false), (13, true)] {
        each_order::<Dense16>(n, 16, 0x5500 + n as u64, deep);
    }
    for (n, deep) in [(5, true), (8, true)] {
        each_order::<Dense32>(n, 32, 0x5600 + n as u64, deep);
    }
}

/// The tie key of the monomial with these exponents over its first `below`
/// variables, or over the variables the first key does not hold.
fn tie_key_below<M: Format>(exponents: &[u32], below: Option<usize>) -> u128 {
    let f = M::new(exponents.len());
    let mut m = vec![M::Word::default(); f.words()];
    f.encode(exponents, &mut m).expect("within the format");
    let key = f.grevlex_tie_key(&m, below);
    assert_eq!(
        <Grevlex as Order<M>>::tie_key(&Grevlex, &f, &m, below),
        key,
        "{exponents:?} below {below:?}"
    );
    key
}

/// The tie key of the monomial with these exponents over the variables the
/// first key does not hold.
fn tie_key<M: Format>(exponents: &[u32]) -> u128 {
    tie_key_below::<M>(exponents, None)
}

/// A tie key from its three parts: the complement of the word's index plus
/// one, the complement of the word's lanes below the first key's, and the
/// top 16 bits of the complement of the word below.
fn parts(index: u16, lanes: u64, next: u16) -> u128 {
    (u128::from(index) << 112) | (u128::from(lanes) << 48) | (u128::from(next) << PAYLOAD_BITS)
}

/// Tie keys written out by hand.
#[test]
fn the_tie_key_as_written() {
    let none = u128::MAX << PAYLOAD_BITS;
    // Thirteen variables of 8 bits, e_i = i: the first key holds the
    // degree and variables 12 down to 2, and the tie key reads word 0's
    // lanes of variables 1 and 0, ~1 and ~0, with no word below.
    let e: Vec<u32> = (0..13).collect();
    assert_eq!(
        tie_key::<Dense8>(&e),
        parts(!1, 0xFFFF_FFFF_FFFF_FEFF, 0xFFFF)
    );
    // Variables 12 down to 2 alone: no word has a lane below the key's.
    let mut e = vec![1u32; 13];
    (e[0], e[1]) = (0, 0);
    assert_eq!(tie_key::<Dense8>(&e), none);
    // Three variables fit the first key.
    assert_eq!(tie_key::<Dense8>(&[2, 3, 4]), none);
    assert_eq!(tie_key::<Dense8>(&[0, 0, 0]), none);
    // Twenty-four of 8 bits, e_i = i up to the fifteenth and 0 after: the
    // first key holds variables 23 down to 13, word 1 holds 12 down to 8
    // below it, and word 0's top lanes are variables 7 and 6.
    let e: Vec<u32> = (0..24).map(|i| if i < 16 { i } else { 0 }).collect();
    assert_eq!(
        tie_key::<Dense8>(&e),
        parts(!2, 0xFFFF_FFF3_F4F5_F6F7, 0xF8F9)
    );
    // The same with nothing in word 1: word 0, e_i = i + 1, no word below.
    let e: Vec<u32> = (0..24).map(|i| if i < 8 { i + 1 } else { 0 }).collect();
    assert_eq!(
        tie_key::<Dense8>(&e),
        parts(!1, 0xF7F8_F9FA_FBFC_FDFE, 0xFFFF)
    );
    // Nine variables of 16 bits, e_i = i + 1: the first key holds variables
    // 8 down to 4, and word 0 holds 3 down to 0.
    let e: Vec<u32> = (1..=9).collect();
    assert_eq!(
        tie_key::<Dense16>(&e),
        parts(!1, 0xFFFB_FFFC_FFFD_FFFE, 0xFFFF)
    );
    // Five variables of 32 bits, e_i = i + 1: the first key holds variables
    // 4 and 3, word 1's low lane is variable 2, and word 0's top lane is
    // variable 1, whose top 16 bits are 0.
    assert_eq!(
        tie_key::<Dense32>(&[1, 2, 3, 4, 5]),
        parts(!2, 0xFFFF_FFFF_FFFF_FFFC, 0xFFFF)
    );
    // The unit, on any variables: none.
    assert_eq!(tie_key::<Dense16>(&[0; 13]), none);
    // The levels below a key's word: twenty-four variables of 8 bits, e_i
    // = i up to the fifteenth, read word 1 first, then word 0 whole, and
    // then nothing. Over every variable, word 2's lanes are 0, and word 1
    // is read whole.
    let e: Vec<u32> = (0..24).map(|i| if i < 16 { i } else { 0 }).collect();
    let f = Dense8::new(24);
    let level0 = tie_key::<Dense8>(&e);
    assert_eq!(f.tie_key_next(level0), Some(8));
    let level1 = tie_key_below::<Dense8>(&e, Some(8));
    assert_eq!(level1, parts(!1, 0xF8F9_FAFB_FCFD_FEFF, 0xFFFF));
    assert_eq!(f.tie_key_next(level1), None);
    assert_eq!(f.tie_key_next(none), None);
    assert_eq!(
        tie_key_below::<Dense8>(&e, Some(24)),
        parts(!2, 0xF0F1_F2F3_F4F5_F6F7, 0xF8F9)
    );
}

/// Test 2: among elements whose first keys agree, the tie keys compare as
/// grevlex compares the monomials, or agree where the monomials agree in
/// every lane the tie key reads; where the variables the first key leaves
/// fit one word, they never agree; and the unit's tie key is all ones.
fn tie_keys_order<M: Format>(n: usize, bits: u32, seed: u64) {
    // A window before the first key's variables, whose first keys all
    // agree, and the whole, whose pairs mostly differ there.
    let (first, lanes) = per_key(bits);
    let (t, sets) = windows::<M>(n, &[(0, n - first), (0, n)], 6, 400, seed);
    let ids = sets.concat();
    let (mut ea, mut eb) = (Vec::new(), Vec::new());
    let mut agree = 0;
    let mut decide = 0;
    for &a in &ids {
        for &b in &ids {
            if a == b || t.key(&Grevlex, a) != t.key(&Grevlex, b) {
                continue;
            }
            ea.clear();
            eb.clear();
            t.exponents(a, &mut ea);
            t.exponents(b, &mut eb);
            let (ka, kb) = (tie_key::<M>(&ea), tie_key::<M>(&eb));
            match ka.cmp(&kb) {
                Ordering::Equal => agree += 1,
                got @ (Ordering::Less | Ordering::Greater) => {
                    decide += 1;
                    assert_eq!(got, t.cmp(&Grevlex, a, b), "{n} variables: {a} against {b}");
                }
            }
        }
    }
    assert!(decide > 0, "{n} variables: no tie key decided a tie");
    if n - first <= lanes {
        assert_eq!(agree, 0, "{n} variables: the tie key holds every variable");
    }
    assert_eq!(
        tie_key::<M>(&vec![0; n]),
        u128::MAX << PAYLOAD_BITS,
        "{n} variables of {bits} bits: the unit"
    );
}

#[test]
fn the_tie_key_orders_as_grevlex() {
    for n in [13, 17, 24, 54] {
        tie_keys_order::<Dense8>(n, 8, 0x5700 + n as u64);
    }
    for n in [9, 13] {
        tie_keys_order::<Dense16>(n, 16, 0x5800 + n as u64);
    }
    for n in [5, 8] {
        tie_keys_order::<Dense32>(n, 32, 0x5900 + n as u64);
    }
}
