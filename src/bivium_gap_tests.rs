//! bivium-gap, the crate's test code: the signatures' test of a signature
//! against the inputs' Koszul signatures, and the variables a monomial
//! holds, against brute force on exponent vectors
//! (`iterations/0163-bivium-gap/plan.md`, _Test code_, 1).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::id::MonoId;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::packed_tests::mono;
use crate::shared_tests::{below, divides, rng, sig_divides};
use crate::sig::{Formed, Sig, Signatures};
use crate::table::Table;

/// The variables of the tables the test draws on: few, whose masks hold
/// bits per exponent; past 32, whose masks hold a bit per variable; and
/// past 128, where the inputs' leads are held by their first variable.
const VARIABLES: [usize; 3] = [3, 40, 150];

/// Exponents on n variables: `held` variables drawn, each raised by 1 to
/// `top`.
pub(crate) fn sparse(rng: &mut Xoshiro256PlusPlus, n: usize, held: u64, top: u64) -> Vec<u32> {
    let mut e = vec![0u32; n];
    for _ in 0..held {
        e[below(rng, n as u64) as usize] += 1 + below(rng, top) as u32;
    }
    e
}

/// The exponents of a·b.
pub(crate) fn product(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

// bivium-gap.8 practice-checked: differential testing against a brute-force oracle -- the item's test code checks the signatures' test against brute force on exponent vectors on 3, 40, and 150 variables beside the calls against the checker, not the calls alone, which leave every result as it is where the test finds fewer Koszul signatures than there are (log 0163-1)
/// Test 1 (bivium-gap.1 to .6): on tables of 3, 40, and 150 variables,
/// seeds 0 to 7, inputs whose leading monomials hold one to three
/// variables, and signatures of their indices, a multiple of their input's
/// lead, among them multiples of a Koszul signature: `is_syzygy`, and
/// `is_syzygy_formed` on the signature formed in the table's scratch
/// words, answer whether a pushed syzygy signature or a Koszul signature
/// `lm(f_i)·lm(f_j)·e_j` of an input i below j divides it, and `push_syzygy`
/// whether neither does.
#[test]
fn koszul_signatures_are_known_by_the_inputs_leads() {
    for n in VARIABLES {
        for seed in 0..8u64 {
            let mut rng = rng(seed ^ 0xb1_0e ^ n as u64);
            let mut table = Table::new(Dense16::new(n), &mut rng.clone());
            let mut b = Budget::unbounded();
            let inputs = 4 + below(&mut rng, 24) as usize;
            let leads: Vec<Vec<u32>> = (0..inputs)
                .map(|_| {
                    loop {
                        let held = 1 + below(&mut rng, 3);
                        let e = sparse(&mut rng, n, held, 2);
                        if e.iter().any(|&x| x > 0) {
                            break e;
                        }
                    }
                })
                .collect();
            let ids: Vec<MonoId> = leads
                .iter()
                .map(|e| table.intern(e, &mut b).unwrap())
                .collect();
            let mut sigs = Signatures::new(inputs);
            sigs.set_inputs(&table, ids.clone());
            let mut pushed: Vec<(Vec<u32>, u32)> = Vec::new();
            for draw in 0..300 {
                let j = below(&mut rng, inputs as u64) as usize;
                let held = below(&mut rng, 4);
                let mut u = sparse(&mut rng, n, held, 2);
                if j > 0 && below(&mut rng, 2) == 0 {
                    u = product(&u, &leads[below(&mut rng, j as u64) as usize]);
                }
                let se = product(&u, &leads[j]);
                let s = Sig {
                    monomial: table.intern(&se, &mut b).unwrap(),
                    index: j as u32,
                };
                let koszul = (0..j).any(|i| divides(&product(&leads[i], &leads[j]), &se));
                let known = koszul
                    || pushed
                        .iter()
                        .any(|(e, i)| sig_divides((e, *i), (&se, s.index)));
                let case = format!("n {n}, seed {seed}, draw {draw}: {se:?} of index {j}");
                assert_eq!(
                    sigs.is_syzygy(&mut table, s, &mut b).unwrap(),
                    known,
                    "{case}"
                );
                let degree = table.form_quotient(s.monomial, ids[j], ids[j]).unwrap();
                let formed = Formed {
                    index: s.index,
                    degree,
                    mask: table.format().mask(table.formed()),
                };
                assert_eq!(
                    sigs.is_syzygy_formed(&mut table, &formed, &mut b).unwrap(),
                    known,
                    "{case}, formed"
                );
                if below(&mut rng, 3) == 0 {
                    assert_eq!(
                        sigs.push_syzygy(&mut table, s, &mut b).unwrap(),
                        !known,
                        "{case}, pushed"
                    );
                    pushed.push((se, s.index));
                }
            }
        }
    }
}

/// Test 1 as well: before `set_inputs` no Koszul signature is known, and
/// the first input's signatures have none below them.
#[test]
fn no_koszul_signature_without_the_inputs_or_below_the_first() {
    let rng = rng(0xb1_0f);
    for n in VARIABLES {
        let mut table = Table::new(Dense16::new(n), &mut rng.clone());
        let mut b = Budget::unbounded();
        let mut e = vec![0u32; n];
        e[0] = 1;
        let x = table.intern(&e, &mut b).unwrap();
        e[0] = 2;
        let xx = table.intern(&e, &mut b).unwrap();
        let mut sigs = Signatures::new(2);
        let s = Sig {
            monomial: xx,
            index: 1,
        };
        assert!(
            !sigs.is_syzygy(&mut table, s, &mut b).unwrap(),
            "n {n}: no inputs set"
        );
        sigs.set_inputs(&table, vec![x, x]);
        assert!(
            sigs.is_syzygy(&mut table, s, &mut b).unwrap(),
            "n {n}: x·x·e_1"
        );
        let first = Sig {
            monomial: xx,
            index: 0,
        };
        assert!(
            !sigs.is_syzygy(&mut table, first, &mut b).unwrap(),
            "n {n}: index 0"
        );
    }
}

/// `Format::held` in format M on n variables: the variables of nonzero
/// exponent of drawn monomials, of one variable to seven, in increasing
/// order, and none of the monomial 1; what `out` held before is gone.
fn held_in<M: Format>(n: usize, seed: u64) {
    let f = M::new(n);
    let mut rng = rng(seed);
    let mut out = vec![u32::MAX];
    for draw in 0..200 {
        let held = below(&mut rng, 8);
        let e = sparse(&mut rng, n, held, 1);
        let want: Vec<u32> = (0..n as u32).filter(|&v| e[v as usize] > 0).collect();
        f.held(&mono(&f, &e).0, &mut out);
        assert_eq!(out, want, "{n} variables, draw {draw}: {e:?}");
    }
}

/// Test 1 as well: in every format `held` names the variables a monomial
/// holds, on one variable to 300, across the words' ends.
#[test]
fn held_names_the_variables_of_nonzero_exponent() {
    for n in [1, 3, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 150, 300] {
        let seed = 0xb1_40 + n as u64;
        held_in::<Dense4>(n, seed);
        held_in::<Dense8>(n, seed);
        held_in::<Dense16>(n, seed);
        held_in::<Dense32>(n, seed);
    }
}
