//! text-time, the writers' unit tests: the basis text over GF(p) as the
//! per-term writer writes it, texts of every length around the bounds of
//! the arena, and the integers over ℚ as num-bigint writes them (plan.md,
//! _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::field::{Field, Gf31, Gf64};
use crate::id::MonoId;
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::order::Grevlex;
use crate::poly::Poly;
use crate::table::Table;
use crate::text::{digits_bound, push_big, term_longest, write_basis, write_poly};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// The variable counts of each format, as `output_tests.rs` takes them:
/// whole words, a last word with one lane, and a last word full.
const COUNTS8: [usize; 9] = [1, 2, 7, 8, 9, 16, 17, 24, 54];
const COUNTS16: [usize; 5] = [1, 3, 4, 5, 9];
const COUNTS32: [usize; 4] = [1, 2, 3, 5];

/// The primes of test 1: below 2^31 in `Gf31`, past it in `Gf64`, 2^64 − 59
/// with coefficients of 20 digits.
const PRIMES31: [u64; 4] = [2, 3, 65521, (1 << 31) - 1];
const PRIMES64: [u64; 2] = [(1 << 61) - 1, 18_446_744_073_709_551_557];

/// Drawn exponents of n variables of degree at most `bound`: most 0 to 3,
/// some up to a tenth of the bound.
fn exponents(rng: &mut impl Rng, n: usize, bound: u32) -> Vec<u32> {
    let mut e = vec![0u32; n];
    let mut left = bound;
    for x in &mut e {
        let r = rng.next_u32();
        let want = match r % 8 {
            0..=3 => 0,
            4..=6 => r / 8 % 4,
            _ => (r / 8) % (bound / 10 + 1),
        };
        *x = want.min(left);
        left -= *x;
    }
    e
}

/// A table in the format `M` on n variables that holds the unit monomial,
/// each variable alone at the degree bound, and drawn monomials; its ids,
/// in that order, the table's last id last.
fn table_with<M: Format>(n: usize, seed: u64) -> (Table<M>, Vec<MonoId>) {
    let mut r = rng(seed);
    let mut table = Table::new(M::new(n), &mut r);
    let mut ids = vec![table.intern(&vec![0; n]).unwrap()];
    for i in 0..n {
        let mut e = vec![0; n];
        e[i] = M::MAX_DEGREE;
        ids.push(table.intern(&e).unwrap());
    }
    for _ in 0..300 {
        let e = exponents(&mut r, n, M::MAX_DEGREE);
        let id = table.intern(&e).unwrap();
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    assert_eq!(
        ids.iter().max().unwrap().index(),
        table.len() - 1,
        "the table's last id is among the ids"
    );
    (table, ids)
}

/// Distinct names of one to 42 characters, letters, digits, and
/// underscores, the first a letter.
fn names(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let len = 1 + (i * 7) % 40;
            let mut s = String::from("x");
            while s.len() < len {
                s.push(match s.len() % 3 {
                    0 => '_',
                    1 => char::from(b'0' + (i % 10) as u8),
                    _ => char::from(b'a' + (s.len() % 26) as u8),
                });
            }
            if i > 0 {
                s.push_str(&i.to_string());
            }
            s
        })
        .collect()
}

/// The basis text as the per-term writer writes it: after `prefix`, `[`,
/// each polynomial as `write_poly` writes it, joined by `,` and a line
/// break, and `]:` and a line break.
fn reference<F: Field, M: Format>(
    prefix: &str,
    names: &[&str],
    table: &Table<M>,
    field: &F,
    polys: &[Poly<F, Rank1>],
) -> String {
    let mut out = format!("{prefix}[");
    for (k, g) in polys.iter().enumerate() {
        if k > 0 {
            out.push_str(",\n");
        }
        write_poly(&mut out, names, table, field, g, &mut Budget::unbounded())
            .expect("a budget without bounds never stops");
    }
    out.push_str("]:\n");
    out
}

/// Drawn polynomials over `field` whose monomials come from `pool`, so that
/// terms share monomials across polynomials: coefficients 1, p − 1, and
/// drawn ones.
fn polys_from<F: Field, M: Format>(
    table: &Table<M>,
    pool: &[MonoId],
    field: &F,
    count: usize,
    seed: u64,
) -> Vec<Poly<F, Rank1>> {
    let p = field.characteristic();
    let mut r = rng(seed);
    (0..count)
        .map(|k| {
            let mut terms = Vec::new();
            for _ in 0..=(k * 3) {
                let id = pool[r.next_u32() as usize % pool.len()];
                let v = match r.next_u32() % 3 {
                    0 => 1,
                    1 => p - 1,
                    _ => 1 + r.next_u64() % (p - 1).max(1),
                };
                terms.push((id, field.coef(v)));
            }
            Poly::from_terms(&mut terms, field, table, &Grevlex, ())
        })
        .collect()
}

/// `write_basis` after a prefix writes what `reference` writes, within the
/// reservation of result-time.11.
fn agrees<F: Field, M: Format>(
    case: &str,
    names: &[&str],
    table: &Table<M>,
    field: &F,
    polys: &[Poly<F, Rank1>],
) {
    let want = reference("pre", names, table, field, polys);
    let mut out = String::from("pre");
    write_basis(
        &mut out,
        names,
        table,
        field,
        polys.iter(),
        &mut Budget::unbounded(),
    )
    .expect("a budget without bounds never stops");
    assert_eq!(out, want, "{case}");
    // result-time.11, which the item keeps: the text is reserved once,
    // before its `[`, and its capacity is the prefix's 3 bytes and that
    // reservation, which the text fills at most.
    let terms: usize = polys.iter().map(|g| g.len().max(1)).sum();
    let reserved = 4 + 2 * polys.len().saturating_sub(1) + terms * term_longest(names);
    assert_eq!(out.capacity(), 3 + reserved, "{case}");
    assert!(out.len() <= 3 + reserved, "{case}");
}

fn test_one<F: Field, M: Format>(n: usize, field: F) {
    let (table, ids) = table_with::<M>(n, 0x7e1 + n as u64);
    let owned = names(n);
    let names: Vec<&str> = owned.iter().map(std::string::String::as_str).collect();
    let p = field.characteristic();
    // A pool of a few monomials, shared by every polynomial, and the whole
    // table.
    let few: Vec<MonoId> = ids.iter().copied().take(12.min(ids.len())).collect();
    for (what, pool) in [("a few monomials", &few), ("every monomial", &ids)] {
        let mut polys = polys_from(&table, pool, &field, 14, 0x7e2 + n as u64);
        // A zero polynomial among the others.
        polys.insert(
            3,
            Poly::from_terms(&mut Vec::new(), &field, &table, &Grevlex, ()),
        );
        // The unit monomial and each variable alone at the degree bound, and
        // the table's last id.
        let mut bound: Vec<_> = ids[..=n]
            .iter()
            .map(|&id| (id, field.coef(p - 1)))
            .collect();
        bound.push((MonoId::at(table.len() - 1), field.coef(1)));
        polys.push(Poly::from_terms(&mut bound, &field, &table, &Grevlex, ()));
        agrees(
            &format!("{what}, {n} variables over {p}"),
            &names,
            &table,
            &field,
            &polys,
        );
    }
    // The empty basis and the basis {1}.
    let mut out = String::new();
    write_basis(
        &mut out,
        &names,
        &table,
        &field,
        std::iter::empty::<&Poly<F, Rank1>>(),
        &mut Budget::unbounded(),
    )
    .expect("a budget without bounds never stops");
    assert_eq!(out, "[]:\n", "{n} variables over {p}");
    let one: Poly<F, Rank1> = Poly::from_terms(
        &mut vec![(ids[0], field.coef(1))],
        &field,
        &table,
        &Grevlex,
        (),
    );
    let mut out = String::new();
    write_basis(
        &mut out,
        &names,
        &table,
        &field,
        std::iter::once(&one),
        &mut Budget::unbounded(),
    )
    .expect("a budget without bounds never stops");
    assert_eq!(out, "[1]:\n", "{n} variables over {p}");
}

fn test_in<M: Format>(n: usize) {
    for p in PRIMES31 {
        test_one::<Gf31, M>(n, Gf31::new(p).unwrap());
    }
    for p in PRIMES64 {
        test_one::<Gf64, M>(n, Gf64::new(p).unwrap());
    }
}

/// Test 1, text-time.1, .2, and .4: `write_basis` writes what the per-term
/// writer writes, in each format, over primes below 2^31 and past it, where
/// the polynomials share a few monomials and where they draw from the whole
/// table, with a zero polynomial, the unit monomial, the variables at the
/// bound, and the table's last id; within result-time.11's reservation.
#[test]
fn basis_text_as_the_per_term_writer_writes_it() {
    for n in COUNTS8 {
        test_in::<Dense8>(n);
    }
    for n in COUNTS16 {
        test_in::<Dense16>(n);
    }
    for n in COUNTS32 {
        test_in::<Dense32>(n);
    }
}

/// The text of a monomial as `write_poly` writes it alone, its coefficient
/// 1 and no `+`.
fn monomial_text<F: Field, M: Format>(
    names: &[&str],
    table: &Table<M>,
    field: &F,
    m: MonoId,
) -> String {
    let g: Poly<F, Rank1> =
        Poly::from_terms(&mut vec![(m, field.coef(1))], field, table, &Grevlex, ());
    let mut out = String::new();
    write_poly(&mut out, names, table, field, &g, &mut Budget::unbounded())
        .expect("a budget without bounds never stops");
    out
}

/// Test 2, text-time.3 and .4: on names chosen so that a monomial's text is
/// 1, 31, 32, 33, 254, 255, 256, and 300 bytes long, each such monomial in
/// three polynomials and in one of its own, `write_basis` writes what the per-term writer writes;
/// and terms whose coefficient of 20 digits leaves the copy of 32 bytes
/// no room in the term's reservation.
#[test]
fn texts_of_every_length_around_the_bounds() {
    // Six variables: a name of 1 byte, of 31, of 30, of 29, of 252, and of
    // 300.
    let owned: Vec<String> = [1, 31, 30, 29, 252, 300]
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            let mut s = format!("v{i}");
            while s.len() < len {
                s.push('_');
            }
            if len == 1 {
                s = "x".to_string();
            }
            s
        })
        .collect();
    let names: Vec<&str> = owned.iter().map(std::string::String::as_str).collect();
    let n = names.len();
    let mut r = rng(0x7e3);
    let mut table: Table<Dense16> = Table::new(Dense16::new(n), &mut r);
    let monomial = |table: &mut Table<Dense16>, pairs: &[(usize, u32)]| {
        let mut e = vec![0u32; n];
        for &(i, k) in pairs {
            e[i] = k;
        }
        table.intern(&e).unwrap()
    };
    let wanted: [(&[(usize, u32)], usize); 8] = [
        (&[(0, 1)], 1),           // x
        (&[(1, 1)], 31),          // a name of 31 bytes
        (&[(2, 2)], 32),          // a name of 30 bytes and ^2
        (&[(3, 2), (0, 1)], 33),  // x, *, 29, ^2
        (&[(4, 1), (0, 1)], 254), // x, *, 252
        (&[(4, 2), (0, 1)], 256), // x, *, 252, ^2
        (&[(4, 1), (0, 2)], 256), // x^2, *, 252
        (&[(5, 1)], 300),         // a name of 300 bytes
    ];
    let field = Gf31::new(65521).unwrap();
    let mut ids = Vec::new();
    for (pairs, len) in wanted {
        let m = monomial(&mut table, pairs);
        assert_eq!(
            monomial_text(&names, &table, &field, m).len(),
            len,
            "{pairs:?}"
        );
        ids.push(m);
    }
    // 255 bytes: a name of 252, `^`, and two digits.
    let m = monomial(&mut table, &[(4, 10)]);
    assert_eq!(monomial_text(&names, &table, &field, m).len(), 255);
    ids.push(m);
    let unit = monomial(&mut table, &[]);
    for p in [65521, (1 << 31) - 1] {
        let field = Gf31::new(p).unwrap();
        // Each monomial in three polynomials, with coefficients 1, p − 1,
        // and 2, the second beside the unit monomial.
        let mut polys = Vec::new();
        for (k, v) in [1, p - 1, 2].into_iter().enumerate() {
            let mut terms: Vec<_> = ids.iter().map(|&m| (m, field.coef(v))).collect();
            if k == 1 {
                terms.push((unit, field.coef(v)));
            }
            polys.push(Poly::from_terms(&mut terms, &field, &table, &Grevlex, ()));
        }
        // Each monomial alone, after the first.
        for &m in &ids {
            let mut terms = vec![(m, field.coef(p - 1)), (unit, field.coef(1))];
            polys.push(Poly::from_terms(&mut terms, &field, &table, &Grevlex, ()));
        }
        agrees(
            &format!("texts around the bounds over {p}"),
            &names,
            &table,
            &field,
            &polys,
        );
    }
    // Terms of the longest text on one variable named x: a coefficient of
    // 20 digits, and x at the degree bound, of 10 digits. After a first
    // term, the 22 bytes of `+`, the coefficient, and `*` leave the copy of
    // 32 bytes no room in the term's `Spare` of `term_longest`, 35 bytes.
    let names = ["x"];
    assert!(22 + 32 > term_longest(&names), "the copy has no room");
    let mut table: Table<Dense32> = Table::new(Dense32::new(1), &mut rng(0x7e4));
    let top = table.intern(&[Dense32::MAX_DEGREE]).unwrap();
    let x = table.intern(&[1]).unwrap();
    let unit = table.intern(&[0]).unwrap();
    let field = Gf64::new(18_446_744_073_709_551_557).unwrap();
    let p = field.characteristic();
    let polys: Vec<Poly<Gf64, Rank1>> = [p - 1, p - 2]
        .into_iter()
        .map(|v| {
            let mut terms = vec![
                (top, field.coef(1)),
                (x, field.coef(v)),
                (unit, field.coef(v)),
            ];
            Poly::from_terms(&mut terms, &field, &table, &Grevlex, ())
        })
        .chain(std::iter::once(Poly::from_terms(
            &mut vec![(x, field.coef(1)), (top, field.coef(p - 1))],
            &field,
            &table,
            &Grevlex,
            (),
        )))
        .collect();
    agrees("the longest terms", &names, &table, &field, &polys);
    // The longest first term: all but the `+` of `term_longest`'s 34 bytes
    // of a term and the 1 byte of its separator.
    let mut alone = String::new();
    write_poly(
        &mut alone,
        &names,
        &table,
        &field,
        &polys[2],
        &mut Budget::unbounded(),
    )
    .expect("a budget without bounds never stops");
    let bound = Dense32::MAX_DEGREE;
    assert_eq!(bound.to_string().len(), 10, "x at the bound has 10 digits");
    assert_eq!(alone, format!("18446744073709551556*x^{bound}+x"));
}

/// Test 3, text-time.6, q-basis-text-time.1 and .8: `push_big` after a prefix writes the prefix
/// and what num-bigint's `to_str_radix(10)` writes, on both sides of 64
/// limbs, and its digits stay within `digits_bound` of the bits.
#[test]
fn integers_as_to_str_radix_writes_them() {
    let two64: BigUint = BigUint::one() << 64u32;
    let ten19 = BigUint::from(10u64.pow(19));
    let mut values = vec![
        BigUint::zero(),
        BigUint::one(),
        &two64 - 1u32,
        two64.clone(),
        &two64 + 1u32,
        &ten19 * &two64,
        (BigUint::one() << 128u32) - 1u32,
    ];
    let ten = BigUint::from(10u32);
    let mut p = BigUint::one();
    for _ in 1..=1300 {
        p *= &ten;
        values.push(&p - 1u32);
        values.push(p.clone());
    }
    for k in 1..=70u32 {
        let power = BigUint::one() << (64 * k);
        values.push(&power - 1u32);
        values.push(power);
    }
    let mut r = rng(0x7e5);
    for limbs in 1..=70 {
        for _ in 0..4 {
            let digits: Vec<u32> = (0..2 * limbs).map(|_| r.next_u32()).collect();
            values.push(BigUint::from_slice(&digits));
        }
    }
    for x in values {
        let mut out = b"pre".to_vec();
        push_big(&mut out, &x);
        let want = format!("pre{decimal}", decimal = x.to_str_radix(10));
        assert_eq!(out, want.into_bytes(), "{x}");
        assert!(
            digits_bound(x.bits()) >= out.len() - 3,
            "{x}: {bits} bits, {digits} digits",
            bits = x.bits(),
            digits = out.len() - 3
        );
    }
}
