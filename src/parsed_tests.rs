//! run-generators-time, sub-item 1's test code: the generators parsed
//! once and mapped to each run's field and table give the polynomials the
//! test makes from the terms it wrote, and the errors in the text's order
//! (`text::Parsed`, run-generators-time.1 to .4)
//! (`iterations/0087-run-generators-time/plan.md`, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::fmt::Write;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Error;
use crate::field::{Field, Gf31, Gf31Pair, Gf64, Gf64Pair};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::poly::Poly;
use crate::systems::Rng;
use crate::table::Table;
use crate::text::Header;

const P: u64 = 65521;
/// The largest prime below 2^31.
const P31: u64 = 2_147_483_647;
/// The Mersenne prime 2^61 − 1.
const P61: u64 = (1 << 61) - 1;
/// Two primes of [2^30, 2^31), a pair learn's.
const L0: u64 = 1_945_203_761;
const L1: u64 = 1_875_939_437;
/// Two primes of [2^61, 2^62), a pair replay's.
const R0: u64 = 2_536_775_269_993_504_451;
const R1: u64 = 4_026_501_963_682_786_037;

/// A table of the format `M` for n variables.
fn table<M: Format>(n: usize) -> Table<M> {
    Table::new(M::new(n), &mut Xoshiro256PlusPlus::seed_from_u64(1))
}

/// The generators of `header` over `field` in `order` into `table`, in
/// the module of rank 1.
fn generators<F: Field, M: Format, O: Order<M>>(
    header: &Header,
    field: &F,
    table: &mut Table<M>,
    order: &O,
) -> Result<Vec<Poly<F, Rank1>>, Error> {
    header.generators(field, table, order)
}

/// A numeric factor as written: a numerator's digits, and a denominator's.
type Factor = (String, Option<String>);

/// A term as drawn: its sign, its exponents, its factors, and the order in
/// which its factors and variables are written, an index into the factors
/// or a variable's index plus the factors' count.
struct Term {
    negative: bool,
    exponents: Vec<u32>,
    factors: Vec<Factor>,
    order: Vec<usize>,
}

/// A literal of `digits` digits, its first digit not 0.
fn literal(rng: &mut Rng, digits: usize) -> String {
    let mut s = String::with_capacity(digits);
    s.push(char::from(b'1' + rng.below(9) as u8));
    for _ in 1..digits {
        s.push(char::from(b'0' + rng.below(10) as u8));
    }
    s
}

/// A drawn numeric factor: an integer below 2^62, of 19, 20, or 100
/// digits, a fraction n/d with d below 65521, which no prime of the test
/// divides, or 65521 itself, which is 0 mod 65521.
fn factor(rng: &mut Rng) -> Factor {
    match rng.below(8) {
        0 => (literal(rng, 19), None),
        1 => ("18446744073709551615".to_string(), None),
        2 => ("18446744073709551616".to_string(), None),
        3 => (literal(rng, 100), None),
        4 => (
            (1 + rng.below(1 << 20)).to_string(),
            Some((1 + rng.below(P - 1)).to_string()),
        ),
        5 => (P.to_string(), None),
        _ => ((1 + rng.below(1 << 62)).to_string(), None),
    }
}

/// A drawn system on n variables: 1 to 6 generators of 1 to 7 terms, each
/// term with 0 to 3 factors and exponents up to 6, and a like term drawn
/// again at times: a copy of an earlier term with its sign flipped, whose
/// sum is 0, or with factors of its own.
fn system(rng: &mut Rng, n: usize) -> Vec<Vec<Term>> {
    let mut gens = Vec::new();
    for _ in 0..=rng.below(6) {
        let mut terms: Vec<Term> = Vec::new();
        for _ in 0..=rng.below(7) {
            let exponents: Vec<u32> = if !terms.is_empty() && rng.below(5) == 0 {
                terms[rng.below(terms.len() as u64) as usize]
                    .exponents
                    .clone()
            } else {
                (0..n)
                    .map(|_| rng.below(7) as u32 * u32::from(rng.below(2) == 0))
                    .collect()
            };
            let factors: Vec<Factor> = (0..rng.below(4)).map(|_| factor(rng)).collect();
            let mut order: Vec<usize> = (0..factors.len()).collect();
            order.extend(
                (0..n)
                    .filter(|&i| exponents[i] > 0)
                    .map(|i| factors.len() + i),
            );
            for k in (1..order.len()).rev() {
                let j = rng.below(k as u64 + 1) as usize;
                order.swap(k, j);
            }
            terms.push(Term {
                negative: rng.below(2) == 1,
                exponents,
                factors,
                order,
            });
        }
        if rng.below(4) == 0 {
            let k = rng.below(terms.len() as u64) as usize;
            let t = &terms[k];
            let copy = Term {
                negative: !t.negative,
                exponents: t.exponents.clone(),
                factors: t.factors.clone(),
                order: t.order.clone(),
            };
            terms.push(copy);
        }
        gens.push(terms);
    }
    gens
}

/// Whitespace between two tokens: none, a space, a line feed, or a line
/// feed and two spaces.
fn space(rng: &mut Rng) -> &'static str {
    ["", "", " ", "\n", "\n  "][rng.below(5) as usize]
}

/// The instance text of `gens` on x1, …, xn, over ℚ.
fn text(rng: &mut Rng, n: usize, gens: &[Vec<Term>]) -> String {
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let mut out = format!("{names}\n0\n", names = names.join(","));
    for (k, g) in gens.iter().enumerate() {
        if k > 0 {
            write!(
                out,
                "{before},{after}",
                before = space(rng),
                after = space(rng)
            )
            .unwrap();
        }
        for (j, t) in g.iter().enumerate() {
            if t.negative {
                write!(out, "-{after}", after = space(rng)).unwrap();
            } else if j > 0 {
                write!(out, "+{after}", after = space(rng)).unwrap();
            }
            if t.order.is_empty() {
                out.push('1');
            }
            for (i, &o) in t.order.iter().enumerate() {
                if i > 0 {
                    write!(
                        out,
                        "{before}*{after}",
                        before = space(rng),
                        after = space(rng)
                    )
                    .unwrap();
                }
                if o < t.factors.len() {
                    let (num, den) = &t.factors[o];
                    out.push_str(num);
                    if let Some(d) = den {
                        write!(
                            out,
                            "{before}/{after}{d}",
                            before = space(rng),
                            after = space(rng)
                        )
                        .unwrap();
                    }
                } else {
                    let v = o - t.factors.len();
                    out.push_str(&names[v]);
                    if t.exponents[v] > 1 {
                        write!(
                            out,
                            "{before}^{after}{exponent}",
                            before = space(rng),
                            after = space(rng),
                            exponent = t.exponents[v]
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    out.push('\n');
    out
}

/// The polynomials the test makes from `gens` over `field` in `order` into
/// `table`: each term's coefficient the product of its factors by the
/// field's own arithmetic, its monomial interned, and each generator by
/// `Poly::from_terms`.
fn made<F: Field, M: Format, O: Order<M>>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    gens: &[Vec<Term>],
) -> Vec<Poly<F, Rank1>> {
    gens.iter()
        .map(|g| {
            let mut terms = Vec::new();
            for t in g {
                let mut c = field.coef(1);
                for (num, den) in &t.factors {
                    c = field.mul(c, field.decimal(num));
                    if let Some(d) = den {
                        c = field.mul(c, field.inv(field.decimal(d)).unwrap());
                    }
                }
                if t.negative {
                    c = field.neg(c);
                }
                let m = table.intern(&t.exponents).unwrap();
                terms.push((m, c));
            }
            Poly::from_terms(&mut terms, field, table, order, ())
        })
        .collect()
}

/// Test 1's check of one text over one field, order, and format.
fn check<F: Field, M: Format, O: Order<M>>(
    case: &str,
    field: &F,
    order: &O,
    n: usize,
    text: &str,
    gens: &[Vec<Term>],
) {
    let header = Header::read(text).unwrap();
    let mut t: Table<M> = table(n);
    let got = generators(&header, field, &mut t, order)
        .unwrap_or_else(|e| panic!("{case}: {e:?}\n{text}"));
    let want = made(field, &mut t, order, gens);
    assert_eq!(got.len(), want.len(), "{case}\n{text}");
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            (&g.monomials, &g.coefs),
            (&w.monomials, &w.coefs),
            "{case}: generator {k}\n{text}"
        );
    }
}

/// Test 1, run-generators-time.1 and .2: drawn texts read over five fields,
/// three orders, and two formats give what the test makes of their terms.
#[test]
fn generators_as_written() {
    for seed in 0..60u64 {
        let mut rng = Rng::new(seed ^ 0x009a_75ed);
        let n = 2 + (seed % 5) as usize;
        let gens = system(&mut rng, n);
        let text = text(&mut rng, n, &gens);
        let case = format!("seed {seed}");
        let block = Block::new(&[n / 2], n).unwrap();
        macro_rules! over {
            ($field:expr) => {{
                let f = $field;
                check::<_, Dense16, _>(&case, &f, &Grevlex, n, &text, &gens);
                check::<_, Dense16, _>(&case, &f, &Lex, n, &text, &gens);
                check::<_, Dense16, _>(&case, &f, &block, n, &text, &gens);
                check::<_, Dense8, _>(&case, &f, &Grevlex, n, &text, &gens);
            }};
        }
        over!(Gf31::prime(P));
        over!(Gf31::prime(P31));
        over!(Gf64::prime(P61));
        over!(Gf31Pair::primes(L0, L1));
        over!(Gf64Pair::primes(R0, R1));
    }
}

/// The error of reading `text` over `field` into a `Dense16` table in
/// grevlex, which must be one.
fn error<F: Field>(field: &F, text: &str) -> Error {
    let header = Header::read(text).unwrap();
    let n = header.names.len();
    let mut t: Table<Dense16> = table(n);
    match generators(&header, field, &mut t, &Grevlex) {
        Ok(g) => panic!("no error but {g:?}\n{text}"),
        Err(e) => e,
    }
}

/// Test 2, run-generators-time.3: each text gives the error of its first
/// break in the text's order, a denominator 0 mod p before a break of the
/// grammar among them.
#[test]
fn errors_in_the_reader_order() {
    let f = Gf31::prime(P);
    // Line 3 x+y, 4 x-y, 5 the denominator 65521, 6 y, 7 the syntax error.
    let text = "x,y\n0\nx+y,\nx-y,\n1/65521*x,\ny,\nx+*y\n";
    assert_eq!(error(&f, text), Error::NoInverse { line: 5 });
    // The syntax error at line 4, before the denominator at line 6.
    let text = "x,y\n0\nx+y,\nx+*y,\ny,\n1/65521*x\n";
    assert!(
        matches!(error(&f, text), Error::Syntax { line: 4, .. }),
        "{text}"
    );
    // A denominator 0 mod p before an unknown variable in one term.
    let text = "x,y\n0\nx+y,\n1/65521*x*q\n";
    assert_eq!(error(&f, text), Error::NoInverse { line: 4 });
    // The unknown variable before the denominator in one term.
    let text = "x,y\n0\nx+y,\nq*1/65521*x\n";
    assert!(
        matches!(error(&f, text), Error::UnknownVariable { line: 4, ref name } if name == "q"),
        "{text}"
    );
    // A denominator 65521 written over two lines: the denominator's line.
    let text = "x,y\n0\nx+y,\n3\n/\n65521*x\n";
    assert_eq!(error(&f, text), Error::NoInverse { line: 6 });
    // A degree of 200 at line 5: past Dense8, within Dense16.
    let text = "x,y\n0\nx+y,\ny^2,\nx^150*y^50+1\n";
    let header = Header::read(text).unwrap();
    let mut t8: Table<Dense8> = table(2);
    let e = generators(&header, &f, &mut t8, &Grevlex).unwrap_err();
    assert_eq!(e, Error::Overflow { line: Some(5) });
    let mut t16: Table<Dense16> = table(2);
    assert_eq!(
        generators(&header, &f, &mut t16, &Grevlex).unwrap().len(),
        3
    );
    // A denominator that the first lane's prime of a pair divides.
    let text = format!("x,y\n0\nx+y,\nx-1/{L0}*y\n");
    assert_eq!(
        error(&Gf31Pair::primes(L0, L1), &text),
        Error::NoInverse { line: 4 }
    );
    // A degree past Dense32, which the parse's own table does not hold,
    // before and after a denominator 0 mod p, and written twice.
    let text = "x,y\n0\nx+y,\nx^2147483648*y+1/65521\n";
    assert_eq!(error(&f, text), Error::Overflow { line: Some(4) });
    let text = "x,y\n0\nx+y,\n1/65521*x^2147483648*y\n";
    assert_eq!(error(&f, text), Error::NoInverse { line: 4 });
    let text = "x,y\n0\nx+y,\nx^99999999999999999999999,\nx^99999999999999999999999\n";
    assert_eq!(error(&f, text), Error::Overflow { line: Some(4) });
    // A denominator 0 mod p after a term past Dense8, in Dense8: the
    // overflow comes first, at its term's line.
    let text = "x,y\n0\nx^200,\n1/65521*y\n";
    let header = Header::read(text).unwrap();
    let mut t8: Table<Dense8> = table(2);
    assert_eq!(
        generators(&header, &f, &mut t8, &Grevlex).unwrap_err(),
        Error::Overflow { line: Some(3) }
    );
}

/// The polynomials of a fresh header of `text` over `field` in grevlex
/// into a fresh table of the format `M`.
fn fresh<F: Field, M: Format>(field: &F, text: &str) -> Result<Vec<Poly<F, Rank1>>, Error> {
    let header = Header::read(text).unwrap();
    let mut t: Table<M> = table(header.names.len());
    generators(&header, field, &mut t, &Grevlex)
}

/// `got` and `want` hold the same coefficients and as many terms.
fn same<F: Field>(case: &str, got: &[Poly<F, Rank1>], want: &[Poly<F, Rank1>]) {
    assert_eq!(got.len(), want.len(), "{case}");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            (&g.monomials, &g.coefs),
            (&w.monomials, &w.coefs),
            "{case}: {k}"
        );
    }
}

/// Test 3, run-generators-time.1 and .4: one header read over several
/// fields and formats in turn, each into a fresh table, gives what a fresh
/// header gives each time.
#[test]
fn one_header_many_fields() {
    for seed in 0..20u64 {
        let mut rng = Rng::new(seed ^ 0x000e_4ead);
        let n = 3;
        let gens = system(&mut rng, n);
        let text = text(&mut rng, n, &gens);
        let header = Header::read(&text).unwrap();
        let case = format!("seed {seed}");
        let (a, b, c) = (Gf31::prime(P), Gf64Pair::primes(R0, R1), Gf31::prime(P31));
        let mut t: Table<Dense16> = table(n);
        same(
            &case,
            &generators(&header, &a, &mut t, &Grevlex).unwrap(),
            &fresh::<_, Dense16>(&a, &text).unwrap(),
        );
        let mut t: Table<Dense16> = table(n);
        same(
            &case,
            &generators(&header, &b, &mut t, &Grevlex).unwrap(),
            &fresh::<_, Dense16>(&b, &text).unwrap(),
        );
        let mut t: Table<Dense16> = table(n);
        same(
            &case,
            &generators(&header, &c, &mut t, &Grevlex).unwrap(),
            &fresh::<_, Dense16>(&c, &text).unwrap(),
        );
        let mut t: Table<Dense16> = table(n);
        same(
            &case,
            &generators(&header, &a, &mut t, &Grevlex).unwrap(),
            &fresh::<_, Dense16>(&a, &text).unwrap(),
        );
    }
    // An overflow in Dense8, then the same header in Dense16.
    let text = "x,y\n0\nx^100*y^100+3*x,\n2/7*y-x\n";
    let header = Header::read(text).unwrap();
    let f = Gf31::prime(P);
    let mut t8: Table<Dense8> = table(2);
    assert_eq!(
        generators(&header, &f, &mut t8, &Grevlex).unwrap_err(),
        Error::Overflow { line: Some(3) }
    );
    let mut t16: Table<Dense16> = table(2);
    same(
        "Dense16 after Dense8",
        &generators(&header, &f, &mut t16, &Grevlex).unwrap(),
        &fresh::<_, Dense16>(&f, text).unwrap(),
    );
}
