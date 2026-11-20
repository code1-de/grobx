//! wide-format-products-time, the item's test code: products in formats of
//! more than four words, through both doors and by a row, agree with
//! interning whatever the hashes, also after a compaction; and calls in
//! rings of many variables return the basis of the same system in its own
//! variables, which the exact checker certifies
//! (`iterations/0132-wide-format-products-time/plan.md`, _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::{HashMap, HashSet};

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::build_tests::agree_in;
use crate::gf2_start_over_tests::{call, reported};
use crate::id::MonoId;
use crate::lookup_tests::{agree_by, draw};
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::shared_tests::{HASHES, below, exponents, multipliers, rng, table};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{boolean, random_sparse};
use crate::table::{Limit, Live};
use crate::{CheckerMode, groebner};

/// Exponents of n variables: one to six of them at places drawn, each 1
/// to 3, their sum at most `cap`.
fn sparse(r: &mut Xoshiro256PlusPlus, n: usize, cap: u32) -> Vec<u32> {
    let mut e = vec![0u32; n];
    let mut left = cap;
    for _ in 0..=below(r, 6) {
        let x = (1 + below(r, 3) as u32).min(left);
        e[below(r, n as u64) as usize] += x;
        left -= x;
    }
    e
}

/// The draws of tests 1 and 3 on n variables of format `M`: the unit
/// monomial first, then sparse and dense draws in turn, every sum of two
/// within the format's degree bound.
fn draws<M: Format>(n: usize) -> impl FnMut(&mut Xoshiro256PlusPlus) -> Vec<u32> {
    let cap = M::MAX_DEGREE / 2;
    let top = (cap / n as u32).clamp(1, 40);
    let mut k = 0u32;
    move |r| {
        k += 1;
        if k == 1 {
            vec![0; n]
        } else if k.is_multiple_of(2) {
            sparse(r, n, cap)
        } else {
            draw(r, n, top, cap)
        }
    }
}

/// Test 1 on tables of format `M` on each count of variables.
fn products_agree<M: Format>(counts: &[usize], seed: u64) {
    for &n in counts {
        for name in HASHES {
            let case = format!(
                "{words} words of degree bound {max} on {n} variables, {name}",
                words = M::new(n).words(),
                max = M::MAX_DEGREE
            );
            let t = table::<M>(n, multipliers(name, n, seed));
            agree_by(t, seed ^ n as u64, &case, draws::<M>(n));
        }
    }
}

/// Test 1, wide-format-products-time.2, .3, .5 to .9: on formats of five
/// to 8 words and past them, a support bit a word and a group of words, a
/// product through both doors is the id that interning the exponents'
/// sum returns, for sparse and dense monomials and the unit monomial,
/// whatever the hashes; and at the degree bound, past which a product
/// stores nothing.
#[test]
fn products_past_four_words_agree_with_interning() {
    // Dense8: 8 lanes a word, 5 to 138 words.
    products_agree::<Dense8>(
        &[33, 40, 57, 64, 65, 72, 100, 424, 512, 513, 600, 1100],
        0x3f01,
    );
    // Dense16: 4 lanes a word, 5 to 75 words.
    products_agree::<Dense16>(&[17, 32, 33, 300], 0x3f02);
    // Dense32: 2 lanes a word, 5 to 70 words.
    products_agree::<Dense32>(&[9, 16, 17, 140], 0x3f03);
    let max = Dense8::MAX_DEGREE;
    let n = 600;
    for name in HASHES {
        let mut t = table::<Dense8>(n, multipliers(name, n, 0x3f04));
        let mut at = |places: &[(usize, u32)]| {
            let mut e = vec![0; n];
            for &(i, x) in places {
                e[i] = x;
            }
            t.intern(&e, &mut Budget::unbounded()).unwrap()
        };
        let x = at(&[(n - 1, max - 5), (300, 2)]);
        let y = at(&[(0, 3)]);
        let z = at(&[(0, 4)]);
        let mut want = vec![0; n];
        (want[0], want[300], want[n - 1]) = (3, 2, max - 5);
        let p = t.mul(x, y, &mut Budget::unbounded()).unwrap();
        assert_eq!(exponents(&t, p), want, "{name}: at the bound");
        // SAFETY: x, y, and z are ids `intern` returned from this table.
        assert_eq!(
            unsafe { t.mul_unchecked(y, x, &mut Budget::unbounded()) },
            Ok(p),
            "{name}"
        );
        assert_eq!(
            t.intern(&want, &mut Budget::unbounded()),
            Ok(p),
            "{name}: interning"
        );
        let len = t.len();
        assert_eq!(
            t.mul(x, z, &mut Budget::unbounded()),
            Err(Limit::Overflow),
            "{name}"
        );
        // SAFETY: as above.
        assert_eq!(
            unsafe { t.mul_unchecked(z, x, &mut Budget::unbounded()) },
            Err(Limit::Overflow),
            "{name}"
        );
        assert_eq!(t.len(), len, "{name}: an overflow stores nothing");
    }
}

/// Test 2, wide-format-products-time.1, .4: a row's products by
/// `row_products` are those of the lookup per term, on formats of five to
/// eight words, which the copies take, and past eight, which the lookup
/// per term takes with the support check.
#[test]
fn a_rows_products_agree_past_four_words() {
    agree_in::<Dense8>(&[33, 64, 65, 424, 600], 0x3f11);
    agree_in::<Dense16>(&[17, 32, 33, 300], 0x3f12);
    agree_in::<Dense32>(&[9, 16, 17, 140], 0x3f13);
}

/// Test 3, wide-format-products-time.2, .7: a compaction keeps a drawn
/// half of a table past eight words, and the products of kept monomials
/// through both doors are the ids that interning the sums returns, one id
/// per monomial.
#[test]
fn supports_survive_a_compaction() {
    for n in [72, 600] {
        for name in ["drawn", "tags that agree", "one chain"] {
            survives_a_compaction::<Dense8>(n, name, 0x3f21);
        }
    }
}

/// Test 3 on a table of format `M` on n variables under the hashes `name`:
/// 600 monomials drawn and interned, a drawn half of them kept by a
/// compaction, every kept id naming its monomial, and the products of kept
/// monomials through both doors the ids that interning the sums returns,
/// one id per monomial.
pub(crate) fn survives_a_compaction<M: Format>(n: usize, name: &str, seed: u64) {
    let case = format!("{n} variables, {name}");
    let mut t = table::<M>(n, multipliers(name, n, seed));
    let mut r = rng((seed + 1) ^ n as u64);
    let mut next = draws::<M>(n);
    let mut drawn = Vec::new();
    for _ in 0..600 {
        let e = next(&mut r);
        let id = t.intern(&e, &mut Budget::unbounded()).unwrap();
        if !drawn.iter().any(|(_, x)| *x == id) {
            drawn.push((e, id));
        }
    }
    let mut live = Live::new(t.len());
    let kept: Vec<(Vec<u32>, MonoId)> = drawn
        .into_iter()
        .filter(|_| below(&mut r, 2) == 0)
        .collect();
    for (_, id) in &kept {
        live.mark(*id);
    }
    let count = t.compact(&mut live, &mut Budget::unbounded()).unwrap();
    assert_eq!(count, kept.len(), "{case}: the monomials kept");
    let kept: Vec<(Vec<u32>, MonoId)> = kept.into_iter().map(|(e, id)| (e, live.id(id))).collect();
    let mut interned: HashSet<Vec<u32>> = kept.iter().map(|(e, _)| e.clone()).collect();
    for (e, id) in &kept {
        assert_eq!(
            &exponents(&t, *id),
            e,
            "{case}: a kept id names its monomial"
        );
    }
    for k in 0..900 {
        let (a, x) = &kept[below(&mut r, kept.len() as u64) as usize];
        let (b, y) = &kept[below(&mut r, kept.len() as u64) as usize];
        let sum: Vec<u32> = a.iter().zip(b).map(|(p, q)| p + q).collect();
        let product = format!("{case}, product {k}");
        let p = t.mul(*x, *y, &mut Budget::unbounded()).unwrap();
        // SAFETY: x and y are the kept ids the compaction gave.
        assert_eq!(
            unsafe { t.mul_unchecked(*y, *x, &mut Budget::unbounded()) },
            Ok(p),
            "{product}: the two doors"
        );
        assert_eq!(exponents(&t, p), sum, "{product}");
        assert_eq!(
            t.intern(&sum, &mut Budget::unbounded()),
            Ok(p),
            "{product}: interning"
        );
        interned.insert(sum);
    }
    assert_eq!(t.len(), interned.len(), "{case}: one id per monomial");
}

/// The variables, the characteristic, and the generators of an instance
/// text.
fn parts(text: &str) -> (Vec<String>, String, String) {
    let mut lines = text.splitn(3, '\n');
    let variables = lines
        .next()
        .unwrap()
        .split(',')
        .map(str::to_string)
        .collect();
    let characteristic = lines.next().unwrap().to_string();
    (
        variables,
        characteristic,
        lines.next().unwrap_or("").to_string(),
    )
}

/// `text` with every name `names` maps replaced by its image; a name is
/// a letter or underscore followed by letters, digits, and underscores.
fn rename(text: &str, names: &HashMap<String, String>) -> String {
    let mut out = String::new();
    let mut word = String::new();
    for c in text.chars().chain(std::iter::once('\0')) {
        if c.is_ascii_alphanumeric() || c == '_' {
            if word.is_empty() && c.is_ascii_digit() {
                out.push(c);
            } else {
                word.push(c);
            }
            continue;
        }
        if !word.is_empty() {
            out.push_str(names.get(&word).map_or(&word, |w| w));
            word.clear();
        }
        if c != '\0' {
            out.push(c);
        }
    }
    out
}

/// `text`'s system in a ring of `n` variables v1 to vn: its k variables in
/// their order at v_(1 + i·n/k), the others in no generator; and the map
/// from its names to theirs.
fn spread(text: &str, n: usize) -> (String, HashMap<String, String>) {
    let (variables, characteristic, generators) = parts(text);
    let k = variables.len();
    assert!(k <= n, "{k} variables in a ring of {n}");
    let names: HashMap<String, String> = variables
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let at = 1 + i * n / k;
            (x.clone(), format!("v{at}"))
        })
        .collect();
    let ring: Vec<String> = (1..=n).map(|i| format!("v{i}")).collect();
    let wide = format!(
        "{ring}\n{characteristic}\n{generators}",
        ring = ring.join(","),
        generators = rename(&generators, &names)
    );
    (wide, names)
}

/// The basis of the call of `text` with cofactors, which the exact checker
/// certifies.
fn certified(name: &str, text: &str, seed: u64) -> String {
    let mut params = reported(seed, true);
    params.cofactors = true;
    params.checker = Some(CheckerMode::Exact);
    let outcome = groebner(text, &params).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert_eq!(outcome.certified, Some(true), "{name}: certified");
    outcome.basis.unwrap_or_else(|| panic!("{name}: no basis"))
}

/// The systems of test 4 over `p`, by name: two boolean systems over GF(2)
/// or two sparse random ones, and the edge cases on x, y, z.
fn systems(p: u64) -> Vec<(String, String)> {
    let mut systems = Vec::new();
    // boolean(10, 12, 0x3f33) carries under a forced bound of 4^8 units or
    // less, boolean(9, 11, 0x3f31) under none.
    for seed in 0..2u64 {
        let text = match (p, seed) {
            (2, 0) => boolean(9, 11, 0x3f31),
            (2, _) => boolean(10, 12, 0x3f33),
            _ => random_sparse(4, p, 0x3f32 + seed),
        };
        systems.push((format!("system {seed} over {p}"), text));
    }
    for (k, gens) in EDGE.iter().enumerate() {
        systems.push((
            format!("edge case {k} over {p}"),
            format!("x,y,z\n{p}\n{gens}\n"),
        ));
    }
    systems
}

/// Test 4, wide-format-products-time.1 to .4, .7: over GF(2), GF(65521),
/// and ℚ, a system spread through a ring of 40, 72, or 600 variables, 5,
/// 9, and 75 words of Dense8, returns the basis of the system in its own
/// variables, renamed, certified by the exact checker; over GF(2) also
/// under a forced bound of 2^10 units, which starts it over and carries.
/// A system of degree 130 in 600 variables starts in Dense16 and returns
/// its input, certified.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn calls_in_many_variables() {
    let seed = 0x3f41;
    let mut carried = 0;
    for p in [2u64, 65521, 0] {
        for (name, text) in systems(p) {
            let own = certified(&name, &text, seed);
            for n in [40, 72, 600] {
                let (wide, names) = spread(&text, n);
                let case = format!("{name} in {n} variables");
                let want = rename(&own, &names);
                assert_eq!(certified(&case, &wide, seed), want, "{case}");
                let (basis, _) = call(&wide, &reported(seed, true), None);
                assert_eq!(basis, want, "{case}: without cofactors");
                if p == 2 {
                    let (basis, lines) = call(&wide, &reported(seed, true), Some(1 << 10));
                    assert_eq!(basis, want, "{case}: under a forced bound");
                    carried += usize::from(lines.iter().any(|l| l.kind == "carry"));
                }
            }
        }
    }
    assert!(carried > 0, "no call carried in a ring of many variables");
    let ring: Vec<String> = (1..=600).map(|i| format!("v{i}")).collect();
    let text = format!(
        "{ring}\n65521\nv7^130-v590,\nv590^2-1\n",
        ring = ring.join(",")
    );
    let basis = certified("degree 130 in 600 variables", &text, seed);
    assert_eq!(
        basis, "[v590^2+65520,\nv7^130+65520*v590]:\n",
        "degree 130 in 600 variables"
    );
}
