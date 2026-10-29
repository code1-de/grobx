//! result-time, the writers' test code: the exponents read from whole
//! words and the numbers (plan.md, _Test code_, 1 to 3, sub-item 1); the
//! basis texts over GF(p) and over ℚ against reference writers that read
//! one exponent at a time and format through `format!`, the texts within
//! their reservations, and the image with its leading monomials apart
//! (4 to 7, sub-item 2, in the module `wired`).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::id::MonoId;
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::table::Table;
use crate::text::{digits_bound, push_big, push_decimal};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// The variable counts of each format: whole words, a last word with one
/// lane, and a last word full.
const COUNTS8: [usize; 9] = [1, 2, 7, 8, 9, 16, 17, 24, 54];
const COUNTS16: [usize; 5] = [1, 3, 4, 5, 9];
const COUNTS32: [usize; 4] = [1, 2, 3, 5];

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

/// A table in the format `M` on n variables that holds drawn monomials, the
/// unit monomial, and each variable alone at the degree bound; its ids.
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
        ids.push(table.intern(&e).unwrap());
    }
    (table, ids)
}

fn unpack_agrees<M: Format>(n: usize) {
    let (table, ids) = table_with::<M>(n, 0x0071 + n as u64);
    let mut buf = vec![u32::MAX; n];
    for &id in &ids {
        let want: Vec<u32> = (0..n).map(|i| table.exponent(id, i)).collect();
        table.unpack(id, &mut buf);
        assert_eq!(buf, want, "unpack, {n} variables, id {id}");
        let mut out = vec![7u32, 8, 9];
        table.exponents(id, &mut out);
        assert_eq!(out[..3], [7, 8, 9], "the prefix stays, {n} variables");
        assert_eq!(out[3..], want[..], "exponents, {n} variables, id {id}");
    }
}

/// result-time.1: `Table::unpack` writes what `Table::exponent` reads,
/// variable by variable, and `Table::exponents` appends the same.
#[test]
fn unpack_agrees_with_exponent() {
    for n in COUNTS8 {
        unpack_agrees::<Dense8>(n);
    }
    for n in COUNTS16 {
        unpack_agrees::<Dense16>(n);
    }
    for n in COUNTS32 {
        unpack_agrees::<Dense32>(n);
    }
}

/// result-time.5: `push_decimal` writes what `format!` writes, after a
/// prefix that stays.
#[test]
fn decimals_as_written() {
    let mut values: Vec<u64> = (0..=1000).collect();
    let mut p = 1u64;
    for _ in 1..=19 {
        p *= 10;
        values.extend([p - 1, p, p + 1]);
    }
    values.extend([65520, (1 << 31) - 2, 1 << 32, 1 << 63, u64::MAX]);
    for v in values {
        let mut out = b"pre".to_vec();
        push_decimal(&mut out, v);
        assert_eq!(out, format!("pre{v}").into_bytes(), "{v}");
    }
}

/// q-basis-text-time.1 and .8: `push_big` writes what `BigUint`'s `Display` writes, and
/// `digits_bound` of its bits is at least the digits it wrote.
#[test]
fn big_numbers_as_written() {
    let two64: BigUint = BigUint::one() << 64u32;
    let mut values = vec![
        BigUint::zero(),
        BigUint::one(),
        &two64 - 1u32,
        two64.clone(),
        &two64 + 1u32,
        BigUint::from(10u64.pow(19)),
        BigUint::from(10u128.pow(20)),
    ];
    let ten = BigUint::from(10u32);
    let mut p = BigUint::one();
    for _ in 1..=300 {
        p *= &ten;
        values.push(&p - 1u32);
        values.push(p.clone());
    }
    let mut r = rng(0x0071);
    for limbs in 1..=400 {
        let digits: Vec<u64> = (0..limbs).map(|_| r.next_u64()).collect();
        values.push(BigUint::from_slice(
            &digits
                .iter()
                .flat_map(|d| [*d as u32, (*d >> 32) as u32])
                .collect::<Vec<u32>>(),
        ));
    }
    for x in values {
        let mut out = b"pre".to_vec();
        push_big(&mut out, &x);
        assert_eq!(out, format!("pre{x}").into_bytes(), "{x}");
        assert!(
            digits_bound(x.bits()) >= out.len() - 3,
            "{x}: {bits} bits, {digits} digits",
            bits = x.bits(),
            digits = out.len() - 3
        );
    }
}

// Sub-item 2's tests, of the writers wired.
mod wired {
    use std::fmt::Write;
    use std::time::Instant;

    use num_bigint::{BigInt, BigUint, Sign};
    use num_traits::{One, Zero};
    use rand_core::Rng;

    use super::{COUNTS8, COUNTS16, COUNTS32, rng, table_with};
    use crate::budget::Budget;
    use crate::entry::{Imaged, image_in};
    use crate::field::{Field, Gf31, Gf64};
    use crate::id::MonoId;
    use crate::lift::{Image, Sparse};
    use crate::module::Rank1;
    use crate::monomial::{Dense8, Dense16, Dense32, Format};
    use crate::order::Grevlex;
    use crate::poly::Poly;
    use crate::rational::{Rational, term_bound_q, write_cofactor_q, write_poly_q};
    use crate::systems::dense;
    use crate::table::Table;
    use crate::text::{Header, RESERVE_CAP, term_longest, write_basis};
    use crate::trace::{NoLearner, NoTrace};
    use crate::{MonomialOrder, Params, Threads};

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

    /// A polynomial as the canonical basis text spells it, read through
    /// `Table::exponent` and written through `format!`.
    fn reference_poly<F: Field, M: Format>(
        names: &[&str],
        table: &Table<M>,
        field: &F,
        poly: &Poly<F, Rank1>,
    ) -> String {
        if poly.is_zero() {
            return "0".to_string();
        }
        let mut terms = Vec::new();
        for (&m, &c) in poly.monomials.iter().zip(&poly.coefs) {
            let v = field.value(c);
            let factors: Vec<String> = (0..names.len())
                .filter_map(|i| match table.exponent(m, i) {
                    0 => None,
                    1 => Some(names[i].to_string()),
                    e => Some(format!("{name}^{e}", name = names[i])),
                })
                .collect();
            terms.push(match (factors.is_empty(), v) {
                (true, _) => format!("{v}"),
                (false, 1) => factors.join("*"),
                (false, _) => format!("{v}*{factors}", factors = factors.join("*")),
            });
        }
        terms.join("+")
    }

    /// Drawn polynomials of the table's monomials over `field`: coefficients 1,
    /// p − 1, and drawn ones, among them a term whose monomial is 1 and the
    /// variables at the degree bound.
    fn polys<F: Field, M: Format>(
        table: &Table<M>,
        ids: &[MonoId],
        field: &F,
        seed: u64,
    ) -> Vec<Poly<F, Rank1>> {
        let p = field.characteristic();
        let mut r = rng(seed);
        let mut out = Vec::new();
        for k in 0..12 {
            let mut terms = Vec::new();
            for _ in 0..=(k * 3) {
                let id = ids[r.next_u32() as usize % ids.len()];
                let v = match r.next_u32() % 3 {
                    0 => 1,
                    1 => p - 1,
                    _ => 1 + r.next_u64() % (p - 1).max(1),
                };
                terms.push((id, field.coef(v)));
            }
            out.push(Poly::from_terms(&mut terms, field, table, &Grevlex, ()));
        }
        // The unit monomial and each variable alone at the degree bound.
        let n = table.format().variables();
        let mut terms: Vec<_> = ids[..=n]
            .iter()
            .map(|&id| (id, field.coef(p - 1)))
            .collect();
        out.push(Poly::from_terms(&mut terms, field, table, &Grevlex, ()));
        out
    }

    fn basis_text_agrees<F: Field, M: Format>(n: usize, field: F) {
        let (table, ids) = table_with::<M>(n, 0x0d1 + n as u64);
        let owned = names(n);
        let names: Vec<&str> = owned.iter().map(std::string::String::as_str).collect();
        let mut polys = polys(&table, &ids, &field, 0x0d2 + n as u64);
        // A zero polynomial among others.
        polys.insert(
            3,
            Poly::from_terms(&mut Vec::new(), &field, &table, &Grevlex, ()),
        );
        let want = format!(
            "pre[{polys}]:\n",
            polys = polys
                .iter()
                .map(|g| reference_poly(&names, &table, &field, g))
                .collect::<Vec<_>>()
                .join(",\n")
        );
        let mut out = String::from("pre");
        write_basis(
            &mut out,
            &names,
            &table,
            &field,
            polys.iter(),
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        let p = field.characteristic();
        assert_eq!(out, want, "{n} variables over {p}");
        // result-time.11: the text is reserved once, before its `[`, and grows
        // past that reservation nowhere: its capacity is the prefix's 3 bytes
        // and the reservation, which the text fills at most.
        let terms: usize = polys.iter().map(|g| g.len().max(1)).sum();
        let reserved = 4 + 2 * (polys.len() - 1) + terms * term_longest(&names);
        assert_eq!(out.capacity(), 3 + reserved, "{n} variables over {p}");
        assert!(out.len() <= 3 + reserved, "{n} variables over {p}");
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

    fn basis_text_agrees_in<M: Format>(n: usize) {
        for p in [2, 3, 65521, (1 << 31) - 1] {
            basis_text_agrees::<Gf31, M>(n, Gf31::new(p).unwrap());
        }
        for p in [(1 << 61) - 1, 18_446_744_073_709_551_557] {
            basis_text_agrees::<Gf64, M>(n, Gf64::new(p).unwrap());
        }
    }

    /// result-time.1, .3 to .5, .11: `write_basis` writes what the reference
    /// writer writes, after a prefix that stays, within the reservation it
    /// makes before its first byte. text-reserve.1: a basis whose bound
    /// passes `RESERVE_CAP` reserves the cap alone, and writes the same text.
    #[test]
    fn basis_text_over_gfp_as_a_reference_writes_it() {
        for n in COUNTS8 {
            basis_text_agrees_in::<Dense8>(n);
        }
        for n in COUNTS16 {
            basis_text_agrees_in::<Dense16>(n);
        }
        for n in COUNTS32 {
            basis_text_agrees_in::<Dense32>(n);
        }
        // Copies of the unit monomial and the 17 variables at the degree
        // bound: each term's text is short, its bound counts all 17 names
        // with ten-digit exponents and a twenty-digit coefficient.
        let n = 17;
        let (table, ids) = table_with::<Dense8>(n, 0x0d3);
        let owned = names(n);
        let names: Vec<&str> = owned.iter().map(String::as_str).collect();
        let field = Gf31::new(65521).unwrap();
        let mut terms: Vec<_> = ids[..=n]
            .iter()
            .map(|&id| (id, field.coef(65520)))
            .collect();
        let g = Poly::from_terms(&mut terms, &field, &table, &Grevlex, ());
        let copies = 120_000;
        let bound = 4 + 2 * (copies - 1) + copies * g.len() * term_longest(&names);
        assert!(bound > RESERVE_CAP, "the bound {bound} within the cap");
        let mut out = String::new();
        write_basis(
            &mut out,
            &names,
            &table,
            &field,
            std::iter::repeat_n(&g, copies),
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        assert!(
            out.len() < RESERVE_CAP,
            "the text of {len} bytes past the cap",
            len = out.len()
        );
        let one = reference_poly(&names, &table, &field, &g);
        let want = format!(
            "[{polys}]:\n",
            polys = vec![one.as_str(); copies].join(",\n")
        );
        assert!(out == want, "the capped text differs from the reference's");
        assert_eq!(out.capacity(), RESERVE_CAP, "a bound of {bound}");
    }

    /// Terms over ℚ as the canonical texts spell them: sign, numerator,
    /// denominator where it is not 1, and monomial, through `format!`.
    fn reference_q(names: &[&str], exponents: &[u32], terms: &[(BigInt, BigUint)]) -> String {
        let n = names.len();
        let mut out = String::new();
        let mut written = false;
        for (k, (num, den)) in terms.iter().enumerate() {
            match num.sign() {
                Sign::NoSign => continue,
                Sign::Minus => out.push('-'),
                Sign::Plus if written => out.push('+'),
                Sign::Plus => {}
            }
            written = true;
            let monomial = &exponents[k * n..(k + 1) * n];
            let factors: Vec<String> = (0..n)
                .filter_map(|i| match monomial[i] {
                    0 => None,
                    1 => Some(names[i].to_string()),
                    e => Some(format!("{name}^{e}", name = names[i])),
                })
                .collect();
            let abs = num.magnitude();
            if factors.is_empty() || !den.is_one() || !abs.is_one() {
                out.push_str(&abs.to_string());
                if !den.is_one() {
                    write!(out, "/{den}").unwrap();
                }
                if factors.is_empty() {
                    continue;
                }
                out.push('*');
            }
            out.push_str(&factors.join("*"));
        }
        if !written {
            out.push('0');
        }
        out
    }

    /// A drawn integer: 0, ±1, below 2^64, 2^64 − 1, 2^64, or of up to 1408
    /// bits, 424 digits.
    fn integer(r: &mut impl Rng) -> BigInt {
        let two64: BigInt = BigInt::one() << 64u32;
        let magnitude = match r.next_u32() % 7 {
            0 => BigInt::zero(),
            1 => BigInt::one(),
            2 => BigInt::from(r.next_u64() >> (r.next_u32() % 64)),
            3 => &two64 - BigInt::one(),
            4 => two64,
            _ => {
                let limbs = 1 + r.next_u32() % 22;
                let digits: Vec<u32> = (0..2 * limbs).map(|_| r.next_u32()).collect();
                BigInt::from_biguint(Sign::Plus, BigUint::from_slice(&digits))
            }
        };
        if r.next_u32().is_multiple_of(2) {
            -magnitude
        } else {
            magnitude
        }
    }

    /// result-time.3 and .5, q-basis-text-time.1 and .8: `write_poly_q` and
    /// `write_cofactor_q` write
    /// what the reference writer writes.
    #[test]
    fn basis_text_over_q_as_a_reference_writes_it() {
        let mut r = rng(0x0071);
        for n in [1, 3, 9, 17] {
            let owned = names(n);
            let names: Vec<&str> = owned.iter().map(std::string::String::as_str).collect();
            for k in 0..60 {
                let len = k % 9;
                let mut exponents = Vec::new();
                for t in 0..len {
                    let mut e = vec![0u32; n];
                    if t % 3 != 0 {
                        for x in &mut e {
                            *x = match r.next_u32() % 5 {
                                0 | 1 => 0,
                                2 => 1,
                                3 => r.next_u32() % 20,
                                _ => r.next_u32() % 70000,
                            };
                        }
                    }
                    exponents.extend(e);
                }
                // Over integers, every term's denominator 1.
                let ints: Vec<BigInt> = (0..len).map(|_| integer(&mut r)).collect();
                let terms: Vec<(BigInt, BigUint)> =
                    ints.iter().map(|i| (i.clone(), BigUint::one())).collect();
                let mut out = String::from("pre");
                write_poly_q(
                    &mut out,
                    &names,
                    &exponents,
                    &ints,
                    &mut Budget::unbounded(),
                )
                .expect("a budget without bounds never stops");
                let want = format!(
                    "pre{reference}",
                    reference = reference_q(&names, &exponents, &terms)
                );
                assert_eq!(out, want, "{n} variables, case {k}");
                // Rationals, with denominators below and past 2^64.
                let rats: Vec<Rational> = (0..len)
                    .map(|_| {
                        let mut d = integer(&mut r).magnitude().clone();
                        if d.is_zero() {
                            d = BigUint::from(7u32);
                        }
                        Rational::new(integer(&mut r), BigInt::from(d)).unwrap()
                    })
                    .collect();
                let terms: Vec<(BigInt, BigUint)> = rats
                    .iter()
                    .map(|q| (q.num.clone(), q.den.clone()))
                    .collect();
                let mut out = String::from("pre");
                write_cofactor_q(
                    &mut out,
                    &names,
                    &exponents,
                    &rats,
                    &mut Budget::unbounded(),
                )
                .expect("a budget without bounds never stops");
                let want = format!(
                    "pre{reference}",
                    reference = reference_q(&names, &exponents, &terms)
                );
                assert_eq!(out, want, "{n} variables, rationals, case {k}");
            }
            // No term, and only zero terms.
            let mut out = String::new();
            write_poly_q(&mut out, &names, &[], &[], &mut Budget::unbounded())
                .expect("a budget without bounds never stops");
            assert_eq!(out, "0");
            let mut out = String::new();
            write_poly_q(
                &mut out,
                &names,
                &vec![0; 2 * n],
                &[BigInt::zero(), BigInt::zero()],
                &mut Budget::unbounded(),
            )
            .expect("a budget without bounds never stops");
            assert_eq!(out, "0");
        }
    }

    // --- the image ---

    /// A polynomial of a canonical text over GF(p), read: exponent vectors and
    /// coefficients.
    fn read_poly(poly: &str, names: &[&str]) -> Sparse {
        let mut out = Sparse {
            exponents: Vec::new(),
            coefs: Vec::new(),
        };
        if poly == "0" {
            return out;
        }
        for term in poly.split('+') {
            let mut coef = 1u64;
            let mut e = vec![0u32; names.len()];
            for factor in term.split('*') {
                if factor.starts_with(|c: char| c.is_ascii_digit()) {
                    coef = factor.parse().unwrap();
                    continue;
                }
                let (name, k) = factor.split_once('^').unwrap_or((factor, "1"));
                let i = names.iter().position(|n| *n == name).unwrap();
                e[i] += k.parse::<u32>().unwrap();
            }
            out.exponents.extend(e);
            out.coefs.push(coef);
        }
        out
    }

    /// The polynomials of a canonical basis text, or the rows of a cofactor
    /// text, over GF(p).
    fn read_list(text: &str) -> Vec<&str> {
        let inner = text
            .strip_prefix('[')
            .unwrap()
            .strip_suffix("]:\n")
            .unwrap();
        if inner.is_empty() {
            Vec::new()
        } else {
            inner.split(",\n").collect()
        }
    }

    /// The image of a run over `text`'s own prime, below 2^31.
    fn image(text: &str, params: &Params) -> Image {
        let header = Header::read(text).unwrap();
        let p: u64 = header.characteristic.parse().unwrap();
        let named = params.order.clone();
        let mut budget = Budget::new(Instant::now(), params).unwrap();
        let got = image_in(
            Gf31::new(p).unwrap(),
            text,
            params,
            None,
            &named,
            &header,
            &mut budget,
            &mut NoTrace,
            &mut NoLearner,
            None,
        )
        .unwrap();
        match got {
            Imaged::Image(image) => image,
            Imaged::Lanes(_) | Imaged::Pair(_) => panic!("a run over GF(p) makes one image"),
            Imaged::Stopped(o) => panic!("the run stopped: {o:?}"),
        }
    }

    /// A system on 17 variables over 65521 that finishes at once: each of the
    /// first sixteen variables is a drawn linear polynomial in the next and the
    /// last, and the last is a root of a cubic. Its basis expresses each of
    /// the first sixteen in the last.
    fn seventeen(seed: u64) -> String {
        let mut r = rng(seed);
        let names: Vec<String> = (0..17).map(|i| format!("v{i}")).collect();
        let mut gens = Vec::new();
        for i in 0..16 {
            let (a, b, c) = (
                1 + r.next_u32() % 65520,
                1 + r.next_u32() % 65520,
                r.next_u32() % 65521,
            );
            gens.push(format!(
                "{x}+{a}*{y}+{b}*v16+{c}",
                x = names[i],
                y = names[i + 1]
            ));
        }
        gens.push(format!(
            "v16^3+{c1}*v16+{c0}",
            c1 = 1 + r.next_u32() % 65520,
            c0 = 1 + r.next_u32() % 65520
        ));
        format!(
            "{names}\n65521\n{gens}\n",
            names = names.join(","),
            gens = gens.join(",\n")
        )
    }

    /// result-time.7, .8, .10: the image holds each element's leading exponents in
    /// `leads` and its other terms in `basis`, as the call's basis text gives
    /// them read back, and its cofactors as the cofactor text gives them.
    #[test]
    fn images_hold_the_leads_apart() {
        let with_t = {
            let text = dense(3, 3, 65521, 1);
            let (names, rest) = text.split_once('\n').unwrap();
            format!("{names},t\n{rest},\nt^130-1\n", rest = rest.trim_end())
        };
        let cases = [
            dense(3, 3, 65521, 1),
            dense(3, 3, 65521, 2),
            with_t,
            seventeen(0x0071),
        ];
        for text in &cases {
            let names: Vec<&str> = text.lines().next().unwrap().split(',').collect();
            let n = names.len();
            for cofactors in [false, true] {
                let mut params = Params::new(
                    MonomialOrder::Grevlex,
                    Threads::Count(1.try_into().unwrap()),
                    3,
                );
                params.cofactors = cofactors;
                let outcome = crate::groebner(text, &params).unwrap();
                let basis = outcome.basis.as_deref().unwrap();
                let image = image(text, &params);
                let mut leads = Vec::new();
                let mut tails = Vec::new();
                for poly in read_list(basis) {
                    let g = read_poly(poly, &names);
                    assert_eq!(g.coefs[0], 1, "monic");
                    leads.extend_from_slice(&g.exponents[..n]);
                    tails.push(Sparse {
                        exponents: g.exponents[n..].to_vec(),
                        coefs: g.coefs[1..].to_vec(),
                    });
                }
                assert_eq!(image.leads, leads, "{names}", names = names.join(","));
                assert_eq!(image.basis, tails, "{names}", names = names.join(","));
                assert_eq!(image.stats, outcome.stats);
                match &outcome.cofactors {
                    Some(c) => {
                        let rows: Vec<Vec<Sparse>> = read_list(c)
                            .into_iter()
                            .map(|row| {
                                let row = row.strip_prefix('[').unwrap().strip_suffix(']').unwrap();
                                row.split(',').map(|e| read_poly(e, &names)).collect()
                            })
                            .collect();
                        assert_eq!(image.cofactors, rows);
                    }
                    None => assert_eq!(image.cofactors.len(), 0, "cofactors untracked"),
                }
            }
        }
    }

    /// result-time.12: the bounds of `lift::finish`'s reservation. For drawn
    /// polynomials with integer coefficients written as one basis text, the
    /// sum over their terms of `term_bound_q` of each coefficient's bits,
    /// with 4 bytes for `[`, `]:`, and the line feed and 2 per separator, is
    /// at least the text's length. A coefficient's bits as `finish` counts
    /// them, from a numerator n over a chain denominator d and the chain's
    /// last L, bits(n) + bits(L) − bits(d) + 1, are at least the bits of
    /// n·(L/d).
    #[test]
    fn the_text_over_q_fits_its_reservation() {
        let mut r = rng(0x0072);
        for n in [1, 3, 9, 17] {
            let owned = names(n);
            let names: Vec<&str> = owned.iter().map(std::string::String::as_str).collect();
            let mut text = String::from("[");
            let mut bound = 4;
            for k in 0..40 {
                let len = 1 + k % 9;
                let mut exponents = Vec::new();
                for _ in 0..len {
                    let e: Vec<u32> = (0..n)
                        .map(|_| match r.next_u32() % 4 {
                            0 | 1 => 0,
                            2 => 1,
                            _ => r.next_u32() % 70000,
                        })
                        .collect();
                    exponents.extend(e);
                }
                let ints: Vec<BigInt> = (0..len).map(|_| integer(&mut r)).collect();
                if k > 0 {
                    text.push_str(",\n");
                    bound += 2;
                }
                write_poly_q(
                    &mut text,
                    &names,
                    &exponents,
                    &ints,
                    &mut Budget::unbounded(),
                )
                .expect("a budget without bounds never stops");
                bound += ints
                    .iter()
                    .map(|c| term_bound_q(&names, c.bits()))
                    .sum::<usize>();
            }
            text.push_str("]:\n");
            assert!(
                text.len() <= bound,
                "{n} variables: {len} > {bound}",
                len = text.len()
            );
        }
        for _ in 0..2000 {
            let num = integer(&mut r);
            let mut d = integer(&mut r).magnitude().clone();
            if d.is_zero() {
                d = BigUint::one();
            }
            let f = integer(&mut r).magnitude().clone() + 1u32;
            let l = &d * &f;
            let bits = num.bits() + l.bits() - d.bits() + 1;
            let exact = (&num * BigInt::from(&l / &d)).bits();
            assert!(bits >= exact, "{num} over {d} in {l}: {bits} < {exact}");
        }
    }
}
