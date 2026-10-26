//! The instance reader and the canonical basis text.
//!
//! An instance is in msolve's input format: a line of variable names, a
//! line with the characteristic, then the generators, separated by commas.

use std::cell::OnceCell;
use std::io;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::OnceLock;

use num_bigint::BigUint;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{self, Budget};
use crate::field::{Field, Gf31, Gf64, U64_DIGITS, decimal_value};
use crate::id::MonoId;
use crate::lehmer;
use crate::module::{Module, Rank1};
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::narrow::{self, Narrow};
use crate::order::{Grevlex, Order};
use crate::poly::Poly;
use crate::table::{Limit, Table};
use crate::{Error, Stop};

// q-lift.35 design -- the test code's reader refuses characteristic 0 as before, not the generators over Q written monic (not measured)
/// The instance's nonzero generators, each monic with its terms in
/// decreasing grevlex order, in the instance's order, as a bracketed list
/// in the canonical basis text's spelling. For the test code of the reader.
/// The instance is read in `Dense8`, `Dense16`, and `Dense32` in turn, as a
/// call reads it, and the first that holds it writes the text.
/// `Error::Characteristic` unless the characteristic is a prime below 2^64:
/// it reads no instance over Q.
///
/// # Errors
///
/// As `groebner` reading the instance, and `Error::Characteristic` for
/// characteristic 0.
pub fn generators(text: &str) -> Result<String, Error> {
    match generators_in::<Dense8>(text) {
        Err(Error::Overflow { .. }) => {}
        got @ (Ok(_) | Err(_)) => return got,
    }
    match generators_in::<Dense16>(text) {
        Err(Error::Overflow { .. }) => {}
        got @ (Ok(_) | Err(_)) => return got,
    }
    generators_in::<Dense32>(text)
}

/// As `generators`, in the format `M` alone.
fn generators_in<M: Format>(text: &str) -> Result<String, Error> {
    let header = Header::read(text)?;
    match header.field()? {
        Prime::Gf31(field) => generators_over::<M, _>(&header, field),
        Prime::Gf64(field) => generators_over::<M, _>(&header, field),
    }
}

/// As `generators_in`, over the field `field` of the header.
fn generators_over<M: Format, F: Field>(header: &Header, field: F) -> Result<String, Error> {
    let format = M::new(header.names.len());
    // No call, so no seed: the multipliers decide no result.
    let mut table = Table::new(format, &mut Xoshiro256PlusPlus::seed_from_u64(NO_SEED));
    let mut polys: Vec<Poly<F, Rank1>> = header.generators(&field, &mut table, &Grevlex)?;
    polys.retain(|p| !p.is_zero());
    for p in &mut polys {
        p.make_monic(&field);
    }
    let mut out = String::new();
    // No call, so no bounds.
    #[allow(
        clippy::expect_used,
        reason = "a budget without bounds never stops, and the writing fails only by a stop"
    )]
    write_basis(
        &mut out,
        &header.names,
        &table,
        &field,
        polys.iter(),
        &mut Budget::unbounded(),
    )
    .expect("a budget without bounds never stops");
    Ok(out)
}

/// The seed of a table whose hash multipliers decide no result: a table
/// outside a call, and the parse's, which numbers the monomials in the order
/// the text first writes them.
const NO_SEED: u64 = 0;

// gfp-large-primes.2 cited: demin-2024 §4.2 -- all three flavours store the elements as machine integers, and the bit size of integers varies depending on the size of p
/// The field a characteristic names (gfp-large-primes.2).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Prime {
    /// A prime below 2^31.
    Gf31(Gf31),
    /// A prime from 2^31 to 2^64.
    Gf64(Gf64),
}

/// An instance's first two lines, and where its generators start.
pub(crate) struct Header<'a> {
    pub(crate) names: Vec<&'a str>,
    /// The characteristic's digits as written.
    pub(crate) characteristic: &'a str,
    /// The text after the characteristic's line.
    rest: Cursor<'a>,
    /// The generators parsed, once `parsed` has been called.
    parsed: OnceCell<Parsed<'a>>,
}

impl<'a> Header<'a> {
    pub(crate) fn read(text: &'a str) -> Result<Header<'a>, Error> {
        let mut c = Cursor::new(text);
        let mut names: Vec<&str> = Vec::new();
        loop {
            c.skip_blanks();
            let name = c.name().ok_or_else(|| c.syntax("a variable name"))?;
            if names.contains(&name) {
                return Err(Error::RepeatedVariable {
                    name: name.to_string(),
                });
            }
            names.push(name);
            c.skip_blanks();
            match c.peek() {
                Some(b',') => c.bump(),
                Some(b'\n') => break,
                Some(_) | None => return Err(c.syntax("a comma or a line feed")),
            }
        }
        c.bump(); // the first line feed
        c.skip_blanks();
        let characteristic = c.integer().ok_or_else(|| c.syntax("the characteristic"))?;
        c.skip_blanks();
        match c.peek() {
            Some(b'\n') => c.bump(),
            None => {}
            Some(_) => return Err(c.syntax("a line feed")),
        }
        Ok(Header {
            names,
            characteristic,
            rest: c,
            parsed: OnceCell::new(),
        })
    }

    /// Whether the characteristic is 0: its digits, leading zeros trimmed,
    /// are none.
    pub(crate) fn rational(&self) -> bool {
        self.characteristic.trim_start_matches('0').is_empty()
    }

    /// The characteristic as a field of the engine: `Error::Characteristic`
    /// unless it is a prime below 2^64.
    pub(crate) fn field(&self) -> Result<Prime, Error> {
        let digits = self.characteristic.trim_start_matches('0');
        let p = digits
            .parse::<u64>()
            .ok()
            .or(digits.is_empty().then_some(0));
        let field = p.and_then(|p| {
            Gf31::new(p)
                .map(Prime::Gf31)
                .or_else(|| Gf64::new(p).map(Prime::Gf64))
        });
        field.ok_or_else(|| Error::Characteristic {
            value: self.characteristic.to_string(),
        })
    }

    /// As `field`, `Error::Characteristic` unless the prime lies below
    /// 2^31. Some of the crate's unit tests read their field through it.
    #[cfg(test)]
    pub(crate) fn gf31(&self) -> Result<Gf31, Error> {
        match self.field()? {
            Prime::Gf31(field) => Ok(field),
            Prime::Gf64(_) => Err(Error::Characteristic {
                value: self.characteristic.to_string(),
            }),
        }
    }

    // q-zero-denominator.2 design -- the lift tells a denominator 0 from a prime that divides a denominator by the line of the text's first denominator 0, not by a reader that knows the instance is over Q (not measured)
    /// The line of the first denominator 0 among the generators: a `/`,
    /// whitespace, and decimal digits that are all `0`, at the digits'
    /// line. None where no denominator is 0. `Cursor::factor` takes a
    /// denominator with the same steps, and the `NoInverse` of
    /// `Parsed::map` names the same line.
    pub(crate) fn zero_denominator(&self) -> Option<usize> {
        let mut c = self.rest.clone();
        while let Some(b) = c.peek() {
            c.bump();
            if b != b'/' {
                continue;
            }
            c.skip_space();
            if c.integer().is_some_and(|d| d.bytes().all(|b| b == b'0')) {
                return Some(c.last);
            }
        }
        None
    }

    /// Every generator, zero ones included, in the instance's order, over
    /// `field` in `table`, each with its terms in `order`: the parsed form
    /// mapped to them (`Parsed::map`). The first error of the text over
    /// this field and format; `Error::Size` where the generators, their
    /// terms, their numeric factors, or their distinct monomials are more
    /// than a u32 id numbers.
    pub(crate) fn generators<F: Field, M: Format, O: Order<M>, D: Module>(
        &self,
        field: &F,
        table: &mut Table<M>,
        order: &O,
    ) -> Result<Vec<Poly<F, D>>, Error> {
        self.parsed().map(field, table, order)
    }

    // run-generators-time.1 design -- the header parses the generators once per call into a form free of the field and the run's table, and each run maps that form to its field and table, not each run reading the instance text again (log 0087-1)
    // run-generators-time.4 design -- the parsed form is made at the first run's read, the first call of `Header::parsed`, and kept in the header, in a OnceCell, not made in `Header::read` nor by the entry point and passed beside the header (not measured)
    /// The generators as parsed: parsed at the first call and kept for
    /// every later one. `Parsed::map` takes them to a run's field and table.
    pub(crate) fn parsed(&self) -> &Parsed<'a> {
        self.parsed.get_or_init(|| Parser::parse(self))
    }
}

/// The names with their variable's index, sorted for binary search.
fn sorted_names<'a>(names: &[&'a str]) -> Vec<(&'a str, usize)> {
    let mut sorted: Vec<_> = names.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    sorted.sort_unstable();
    sorted
}

/// A position in the text, with its line.
#[derive(Clone)]
struct Cursor<'a> {
    text: &'a str,
    pos: usize,
    /// The line of `pos`, from 1.
    line: usize,
    /// The line of the last token read.
    last: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Cursor<'a> {
        Cursor {
            text,
            pos: 0,
            line: 1,
            last: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    /// Steps over one byte, a line feed among them.
    fn bump(&mut self) {
        if self.peek() == Some(b'\n') {
            self.line += 1;
        } else {
            self.last = self.line;
        }
        self.pos += 1;
    }

    /// Skips ASCII whitespace but the line feed.
    fn skip_blanks(&mut self) {
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_whitespace() && b != b'\n')
        {
            self.pos += 1;
        }
    }

    /// Skips ASCII whitespace, line feeds included.
    fn skip_space(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_whitespace()) {
            if self.peek() == Some(b'\n') {
                self.line += 1;
            }
            self.pos += 1;
        }
    }

    /// The run of bytes from `pos` that `class` accepts, the first of them
    /// accepted by `first`; None where the first is not.
    fn run(&mut self, first: fn(u8) -> bool, class: fn(u8) -> bool) -> Option<&'a str> {
        if !self.peek().is_some_and(first) {
            return None;
        }
        let start = self.pos;
        while self.peek().is_some_and(class) {
            self.pos += 1;
        }
        self.last = self.line;
        Some(&self.text[start..self.pos])
    }

    fn name(&mut self) -> Option<&'a str> {
        self.run(
            |b| b.is_ascii_alphabetic() || b == b'_',
            |b| b.is_ascii_alphanumeric() || b == b'_',
        )
    }

    fn integer(&mut self) -> Option<&'a str> {
        self.run(|b| b.is_ascii_digit(), |b| b.is_ascii_digit())
    }

    /// Whether a sign stands here and is a minus, None where none does; the
    /// sign is read.
    fn sign(&mut self) -> Option<bool> {
        let negative = match self.peek() {
            Some(b'+') => false,
            Some(b'-') => true,
            Some(_) | None => return None,
        };
        self.bump();
        Some(negative)
    }

    /// A numeric factor, `n` or `n/d` with whitespace around the `/`, its
    /// denominator at the line of the denominator's digits; None where no
    /// digit stands here.
    fn factor(&mut self) -> Result<Option<Factor<'a>>, Error> {
        let Some(digits) = self.integer() else {
            return Ok(None);
        };
        let numerator = Literal::new(digits);
        self.skip_space();
        if self.peek() != Some(b'/') {
            return Ok(Some(Factor {
                numerator,
                denominator: None,
            }));
        }
        self.bump();
        self.skip_space();
        let digits = self.integer().ok_or_else(|| self.syntax("a denominator"))?;
        Ok(Some(Factor {
            numerator,
            denominator: Some((Literal::new(digits), self.last)),
        }))
    }

    /// A power of a variable, `x` or `x^e` with whitespace around the `^`:
    /// the variable's index and e, 1 where no `^` stands; None where no name
    /// stands here. `names` is sorted by `sorted_names`.
    /// `Error::UnknownVariable` at the name's line where it is none of them.
    fn power(&mut self, names: &[(&str, usize)]) -> Result<Option<(usize, u64)>, Error> {
        let Some(name) = self.name() else {
            return Ok(None);
        };
        let i = match names.binary_search_by(|(n, _)| n.cmp(&name)) {
            Ok(k) => names[k].1,
            Err(_) => {
                return Err(Error::UnknownVariable {
                    line: self.last,
                    name: name.to_string(),
                });
            }
        };
        self.skip_space();
        if self.peek() != Some(b'^') {
            return Ok(Some((i, 1)));
        }
        self.bump();
        self.skip_space();
        let digits = self.integer().ok_or_else(|| self.syntax("an exponent"))?;
        // A value past u64 passes every format.
        Ok(Some((i, digits.parse().unwrap_or(u64::MAX))))
    }

    /// A syntax error where `expected` was expected: at the line of the text
    /// there, or at the end of the text, at the line of the last token.
    fn syntax(&self, expected: &'static str) -> Error {
        let line = if self.peek().is_none() {
            self.last
        } else {
            self.line
        };
        Error::Syntax { line, expected }
    }
}

/// Raises the exponent of variable i by e, cut at `u32::MAX`.
fn raise(exponents: &mut [u32], i: usize, e: u64) {
    debug_assert!(
        i < exponents.len(),
        "variable {i} of {len}",
        len = exponents.len()
    );
    exponents[i] = u64::from(exponents[i])
        .saturating_add(e)
        .min(u64::from(u32::MAX))
        .narrow();
}

/// The error of a term at `line` whose monomial reaches a limit of the
/// table.
fn term_error(limit: Limit, line: usize) -> Error {
    match limit {
        Limit::Overflow => Error::Overflow { line: Some(line) },
        Limit::Size => Error::Size,
    }
}

// run-generators-time.2 design -- a literal of at most nineteen digits is kept as its value, whose residue is `Field::coef` of it, and a longer one as its digits, whose residue `Field::decimal` takes, not every literal's digits kept and read at each run nor every literal as a big integer's limbs (log 0087-1)
/// A numeric literal of the instance text: the value of at most
/// `U64_DIGITS` digits, or the digits of a longer one.
#[derive(Clone, Copy, Debug)]
enum Literal<'a> {
    Small(u64),
    Digits(&'a str),
}

impl<'a> Literal<'a> {
    /// The literal these decimal digits write.
    fn new(digits: &'a str) -> Literal<'a> {
        if digits.len() <= U64_DIGITS {
            Literal::Small(decimal_value(digits.as_bytes()))
        } else {
            Literal::Digits(digits)
        }
    }

    /// Its residue at every lane of the field, `Field::decimal` of its
    /// digits.
    fn residue<F: Field>(self, field: &F) -> F::Coef {
        match self {
            Literal::Small(v) => field.coef(v),
            Literal::Digits(digits) => field.decimal(digits),
        }
    }
}

/// A numeric factor of a term, `n` or `n/d`: its numerator, and its
/// denominator with the line of the denominator's digits.
#[derive(Clone, Copy, Debug)]
struct Factor<'a> {
    numerator: Literal<'a>,
    denominator: Option<(Literal<'a>, usize)>,
}

impl Factor<'_> {
    /// Its value in the field: the numerator times the inverse of the
    /// denominator, `Error::NoInverse` at the denominator's line where that
    /// is 0 at a lane.
    fn value<F: Field>(&self, field: &F) -> Result<F::Coef, Error> {
        let numerator = self.numerator.residue(field);
        let Some((denominator, line)) = self.denominator else {
            return Ok(numerator);
        };
        let inverse = field
            .inv(denominator.residue(field))
            .ok_or(Error::NoInverse { line })?;
        Ok(field.mul(numerator, inverse))
    }
}

/// A term as parsed: its sign, its factors as a range of
/// `Parsed::factors`, its monomial's id in `Parsed::monomials`, and the
/// line it starts at. The id is `MonoId::NONE` where the monomial's degree
/// passes `Dense32`, the widest format, and so every format.
struct Term {
    negative: bool,
    factors: Range<u32>,
    monomial: MonoId,
    line: usize,
}

/// The generators as parsed once, free of a run's field and table: per
/// generator a range of `terms`. Where the text breaks the grammar,
/// `error` holds the parse's error and the factors the broken term read
/// before the break, a range of `factors`, and the last generator holds
/// the terms the broken generator read before it.
pub(crate) struct Parsed<'a> {
    generators: Vec<Range<u32>>,
    terms: Vec<Term>,
    factors: Vec<Factor<'a>>,
    // run-generators-time.9 design -- the parse numbers the text's distinct monomials in a table of its own in `Dense32`, in the order the text first writes them, and each run interns each of them once, at its first term, not each term's exponents kept and interned at every run (log 0087-2)
    /// The terms' distinct monomials, numbered in the order the text first
    /// writes them.
    monomials: Table<Dense32>,
    error: Option<(Error, Range<u32>)>,
}

impl Parsed<'_> {
    /// The generators over `field` in `table`, zero ones included, in the
    /// text's order, each with its terms in `order`, like terms added and
    /// zero terms dropped; or the first error of the text. The terms are
    /// taken in the text's order, each its factors' values in turn, then
    /// its monomial, so the error is the first in the text over this field
    /// and format. A monomial is interned at its first term, so `table`
    /// numbers the monomials in the order the text first writes them.
    pub(crate) fn map<F: Field, M: Format, O: Order<M>, D: Module>(
        &self,
        field: &F,
        table: &mut Table<M>,
        order: &O,
    ) -> Result<Vec<Poly<F, D>>, Error> {
        let n = self.monomials.format().variables();
        debug_assert_eq!(
            table.format().variables(),
            n,
            "a table of other variables than the text's"
        );
        let mut ids = vec![MonoId::NONE; self.monomials.len()];
        let mut exponents = vec![0; n];
        let mut polys = Vec::with_capacity(self.generators.len());
        let mut terms = Vec::new();
        for g in &self.generators {
            for term in &self.terms[g.start as usize..g.end as usize] {
                let coef = self.coefficient(&term.factors, field)?;
                let m = self.monomial(term, table, &mut ids, &mut exponents)?;
                terms.push((m, if term.negative { field.neg(coef) } else { coef }));
            }
            polys.push(Poly::from_terms(
                &mut terms,
                field,
                table,
                order,
                D::Position::default(),
            ));
        }
        // run-generators-time.3 design -- a text that breaks the grammar keeps the terms before the break and the broken term's factors, which the map takes before it returns the parse's error, so that a denominator 0 mod p before the break stays the error, not the parse's error returned at once nor the reader kept for such texts (not measured)
        if let Some((error, broken)) = &self.error {
            self.coefficient(broken, field)?;
            return Err(error.clone());
        }
        Ok(polys)
    }

    /// The product of the values of these factors, 1 for none, or the
    /// first factor's error.
    fn coefficient<F: Field>(&self, factors: &Range<u32>, field: &F) -> Result<F::Coef, Error> {
        let mut factors = self.factors[factors.start as usize..factors.end as usize].iter();
        let Some(first) = factors.next() else {
            return Ok(field.coef(1));
        };
        factors.try_fold(first.value(field)?, |coef, factor| {
            Ok(field.mul(coef, factor.value(field)?))
        })
    }

    /// The id in `table` of the term's monomial, interned at the monomial's
    /// first term: `ids` holds per id of `monomials` its id in `table`,
    /// `MonoId::NONE` before, and `exponents` is scratch of one exponent per
    /// variable.
    fn monomial<M: Format>(
        &self,
        term: &Term,
        table: &mut Table<M>,
        ids: &mut [MonoId],
        exponents: &mut [u32],
    ) -> Result<MonoId, Error> {
        if term.monomial == MonoId::NONE {
            return Err(Error::Overflow {
                line: Some(term.line),
            });
        }
        let known = ids[term.monomial.index()];
        if known != MonoId::NONE {
            return Ok(known);
        }
        self.monomials.unpack(term.monomial, exponents);
        let id = table
            .intern(exponents)
            .map_err(|limit| term_error(limit, term.line))?;
        ids[term.monomial.index()] = id;
        Ok(id)
    }
}

// f4-gfp.43 design -- the reader takes the tolerant grammar of checker-gfp.3, whitespace between any tokens, terms in any order, like terms combined, not the corpus's layout alone (not measured)
/// Parses the generators into a `Parsed`: per term its sign, its numeric
/// factors, and its monomial's number, free of a run's field and table.
struct Parser<'a> {
    c: Cursor<'a>,
    names: Vec<(&'a str, usize)>,
    /// The current term's exponents.
    exponents: Vec<u32>,
    parsed: Parsed<'a>,
}

impl<'a> Parser<'a> {
    /// The generators of the text after the header's characteristic line.
    fn parse(header: &Header<'a>) -> Parsed<'a> {
        let n = header.names.len();
        let mut parser = Parser {
            c: header.rest.clone(),
            names: sorted_names(&header.names),
            exponents: vec![0; n],
            parsed: Parsed {
                generators: Vec::new(),
                terms: Vec::new(),
                factors: Vec::new(),
                monomials: Table::new(
                    Dense32::new(n),
                    &mut Xoshiro256PlusPlus::seed_from_u64(NO_SEED),
                ),
                error: None,
            },
        };
        if let Err(error) = parser.generators() {
            parser.broken(error);
        }
        parser.parsed
    }

    /// The generators: `Error::Size` where they are more than a u32 id
    /// numbers.
    fn generators(&mut self) -> Result<(), Error> {
        self.c.skip_space();
        if self.c.peek().is_none() {
            return Ok(());
        }
        loop {
            narrow::id(self.parsed.generators.len()).map_err(|_| Error::Size)?;
            self.poly()?;
            self.c.skip_space();
            match self.c.peek() {
                None => return Ok(()),
                Some(b',') => self.c.bump(),
                Some(_) => return Err(self.c.syntax("a sign, a *, a comma, or the end")),
            }
        }
    }

    fn poly(&mut self) -> Result<(), Error> {
        let start = self.parsed.terms.len().narrow();
        self.c.skip_space();
        let mut negative = self.c.sign().unwrap_or(false);
        loop {
            self.term(negative)?;
            self.c.skip_space();
            match self.c.sign() {
                Some(n) => negative = n,
                None => break,
            }
        }
        let end = self.parsed.terms.len().narrow();
        self.parsed.generators.push(start..end);
        Ok(())
    }

    /// Parses one term and adds it to `terms`: `Error::Size` where the
    /// terms, their factors, or their distinct monomials are more than a
    /// u32 id numbers.
    fn term(&mut self, negative: bool) -> Result<(), Error> {
        self.exponents.fill(0);
        self.c.skip_space();
        let line = self.c.line;
        loop {
            self.c.skip_space();
            if let Some(factor) = self.c.factor()? {
                narrow::id(self.parsed.factors.len()).map_err(|_| Error::Size)?;
                self.parsed.factors.push(factor);
            } else if let Some((i, e)) = self.c.power(&self.names)? {
                raise(&mut self.exponents, i, e);
            } else {
                return Err(self.c.syntax("a term"));
            }
            self.c.skip_space();
            if self.c.peek() == Some(b'*') {
                self.c.bump();
            } else {
                break;
            }
        }
        let monomial = match self.parsed.monomials.intern(&self.exponents) {
            Ok(id) => id,
            Err(Limit::Overflow) => MonoId::NONE,
            Err(Limit::Size) => return Err(Error::Size),
        };
        narrow::id(self.parsed.terms.len()).map_err(|_| Error::Size)?;
        let first = self.parsed.terms.last().map_or(0, |t| t.factors.end);
        let end = self.parsed.factors.len().narrow();
        self.parsed.terms.push(Term {
            negative,
            factors: first..end,
            monomial,
            line,
        });
        Ok(())
    }

    /// Keeps the parse's error, the terms the broken generator read before
    /// it as the last generator, and the factors the broken term read.
    fn broken(&mut self, error: Error) {
        let parsed = &mut self.parsed;
        let start = parsed.generators.last().map_or(0, |g| g.end);
        let end = parsed.terms.len().narrow();
        if start < end {
            parsed.generators.push(start..end);
        }
        let first = parsed.terms.last().map_or(0, |t| t.factors.end);
        parsed.error = Some((error, first..parsed.factors.len().narrow()));
    }
}

// text-stop-latency.2 design -- a writer over GF(p) spends the bytes of each chunk of CHUNK, 2^10, terms after it, the inner loop over a chunk's terms as before, not a spend per term nor a test of the term's index per term to decide the spend (log 0057-1)
/// The terms a writer over GF(p) writes between two spends of the bytes it
/// wrote.
pub(crate) const CHUNK: usize = 1 << 10;

// text-stop-latency.3 design -- over Q each term spends, right after it, its bytes plus the square of the limbs of its numerator and of its denominator, not its bytes alone nor a spend per polynomial (log 0057-1)
/// The units of work of writing x in decimal beside its bytes: the square
/// of its limbs, 0 for 0.
pub(crate) fn decimal_units(x: &BigUint) -> usize {
    let limbs = limbs(x);
    limbs * limbs
}

/// x's limbs of 64 bits, 0 for 0.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a number in memory has fewer limbs than usize counts"
)]
pub(crate) fn limbs(x: &BigUint) -> usize {
    x.bits().div_ceil(64) as usize
}

// text-stop-latency.1 design -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
/// Writes the polynomials as the canonical basis text: `[`, the polynomials
/// separated by `,` and a line feed, `]:` and a line feed. Each polynomial
/// spends a unit of `budget` per byte of its terms, after every `CHUNK`
/// terms and at its end; the brackets, the separators, and a zero
/// polynomial's `0` spend nothing. A stop where the budget's check stops
/// the call, the text then cut short.
pub(crate) fn write_basis<'p, F: Field, M: Format, D: Module + 'p>(
    out: &mut String,
    names: &[&str],
    table: &Table<M>,
    field: &F,
    polys: impl Iterator<Item = &'p Poly<F, D>> + Clone,
    budget: &mut Budget,
) -> Result<(), Stop> {
    // result-time.11 design -- the basis text over GF(p) is reserved once, before its `[`, at the sum of its terms' longest texts and its brackets and separators, not grown by doubling from the first term's reservation or as a String grows (log 0042-1)
    let k = polys.clone().count();
    let terms: usize = polys.clone().map(|g| g.len().max(1)).sum();
    out.reserve(4 + 2 * k.saturating_sub(1) + terms * term_longest(names));
    out.push('[');
    let mut texts = Texts::new(table.len(), names.len());
    for (k, p) in polys.enumerate() {
        if k > 0 {
            out.push_str(",\n");
        }
        write_kept(out, names, table, field, p, &mut texts, budget)?;
    }
    out.push_str("]:\n");
    Ok(())
}

// text-time.1 design -- the basis text over GF(p) writes each distinct monomial's text once per call, into an arena, and copies it from there for each further term with that monomial, not each term's monomial from its exponents (log 0048-1)
// text-time.2 design -- the arena's index is a vector of one u32 per id of the table, zeroed for each basis text, the offset of the text's length byte in the arena, not a hash map from id to offset or 8 bytes an id with the length beside the offset (log 0048-1)
// text-time.3 design -- a monomial whose text passes 255 bytes is not kept, and each of its terms writes it from its exponents, not a length of two bytes or more (not measured)
/// The texts of the monomials one basis text has written, by id of the
/// table: an arena of bytes and an index into it.
struct Texts {
    /// Per id, the offset in `arena` of its text's length byte; 0 where
    /// the arena holds no text of it.
    at: Vec<u32>,
    /// A zero byte that no entry names, then per text kept its length byte
    /// and its bytes, then `COPY` zero bytes.
    arena: Vec<u8>,
    /// The exponents of the monomial last written from them.
    e: Vec<u32>,
}

/// The bytes the copy of a short text takes at once (text-time.4).
const COPY: usize = 32;

impl Texts {
    /// No text yet, for a table of `monomials` ids on `variables` variables.
    fn new(monomials: usize, variables: usize) -> Texts {
        Texts {
            at: vec![0; monomials],
            arena: vec![0; 1 + COPY],
            e: vec![0; variables],
        }
    }

    /// The offset of m's length byte in the arena, its text written there
    /// the first time. None where the text is not kept: `e` then holds m's
    /// exponents.
    #[inline(always)]
    fn text<M: Format>(&mut self, table: &Table<M>, names: &[&str], m: MonoId) -> Option<usize> {
        let at = self.at[m.index()];
        if at != 0 {
            return Some(at as usize);
        }
        self.keep(table, names, m)
    }

    /// Writes m's text past the arena's last text, and keeps it where its
    /// length fits a byte and its offset a u32.
    #[inline(never)]
    fn keep<M: Format>(&mut self, table: &Table<M>, names: &[&str], m: MonoId) -> Option<usize> {
        table.unpack(m, &mut self.e);
        let at = self.arena.len() - COPY;
        self.arena.truncate(at);
        self.arena.push(0);
        let e = &self.e;
        // SAFETY: `monomial` writes at most what `monomial_longest` counts.
        unsafe {
            spare(&mut self.arena, monomial_longest(names), |w| {
                w.monomial(names, e);
            });
        };
        let n = self.arena.len() - at - 1;
        if n > u8::MAX as usize || at > u32::MAX as usize {
            self.arena.truncate(at);
            self.arena.resize(at + COPY, 0);
            return None;
        }
        self.arena[at] = n.narrow();
        self.arena.resize(self.arena.len() + COPY, 0);
        self.at[m.index()] = at.narrow();
        Some(at)
    }
}

/// Writes a polynomial as `write_poly` writes it and spends as it spends,
/// each monomial's text from `texts`.
fn write_kept<F: Field, M: Format, D: Module>(
    out: &mut String,
    names: &[&str],
    table: &Table<M>,
    field: &F,
    poly: &Poly<F, D>,
    texts: &mut Texts,
    budget: &mut Budget,
) -> Result<(), Stop> {
    if poly.is_zero() {
        out.push('0');
        return Ok(());
    }
    // result-time.3 design -- the text is written as bytes into the vector under the String, ASCII and each name whole, not characters pushed into the String or a vector checked by `String::from_utf8` at the end (log 0042-1)
    // SAFETY: every byte written is ASCII or a byte of a name, and each name
    // is a `&str` written whole, so the String stays UTF-8.
    let bytes = unsafe { out.as_mut_vec() };
    let longest = term_longest(names);
    // text-stop-latency.2 design -- a writer over GF(p) spends the bytes of each chunk of CHUNK, 2^10, terms after it, the inner loop over a chunk's terms as before, not a spend per term nor a test of the term's index per term to decide the spend (log 0057-1)
    let chunks = poly.monomials.chunks(CHUNK).zip(poly.coefs.chunks(CHUNK));
    for (j, (ms, cs)) in chunks.enumerate() {
        let mark = bytes.len();
        for (i, (&m, &c)) in ms.iter().zip(cs).enumerate() {
            let k = j * CHUNK + i;
            let v = field.value(c);
            let unit = table.degree(m) == 0;
            let kept = if unit {
                None
            } else {
                texts.text(table, names, m)
            };
            let write = |w: &mut Spare<'_>| {
                if k > 0 {
                    w.byte(b'+');
                }
                if unit {
                    w.decimal(v);
                    return;
                }
                if v != 1 {
                    w.decimal(v);
                    w.byte(b'*');
                }
                match kept {
                    Some(at) => w.kept(&texts.arena[at..]),
                    None => w.monomial(names, &texts.e),
                }
            };
            // SAFETY: a term is at most a `+`, a coefficient below 2^64, a
            // `*`, and the monomial, which `term_longest` counts; `kept`
            // counts the text's bytes alone and copies past them only inside
            // the reservation.
            unsafe { spare(bytes, longest, write) };
        }
        budget.spend(bytes.len() - mark)?;
    }
    Ok(())
}

// text-time.9 design -- the cofactor texts over GF(p) write each monomial from its exponents at each term, not through the basis text's arena of the monomials' texts (not measured)
// text-stop-latency.1 design -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
/// Writes a polynomial in the canonical spelling, `0` for zero: its terms
/// joined by `+`, each its coefficient in [1, p), then `*` and the monomial,
/// where the coefficient 1 and the `*` are left out before a monomial and
/// the monomial 1 after a coefficient. Spends a unit of `budget` per byte of
/// its terms, after every `CHUNK` terms and at its end; the `0` of zero
/// spends nothing. A stop where the budget's check stops the call, the
/// text then cut short.
pub(crate) fn write_poly<F: Field, M: Format, D: Module>(
    out: &mut String,
    names: &[&str],
    table: &Table<M>,
    field: &F,
    poly: &Poly<F, D>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    if poly.is_zero() {
        out.push('0');
        return Ok(());
    }
    // result-time.3 design -- the text is written as bytes into the vector under the String, ASCII and each name whole, not characters pushed into the String or a vector checked by `String::from_utf8` at the end (log 0042-1)
    // SAFETY: every byte written is ASCII or a byte of a name, and each name
    // is a `&str` written whole, so the String stays UTF-8.
    let bytes = unsafe { out.as_mut_vec() };
    let mut e = vec![0u32; table.format().variables()];
    let longest = term_longest(names);
    // text-stop-latency.2 design -- a writer over GF(p) spends the bytes of each chunk of CHUNK, 2^10, terms after it, the inner loop over a chunk's terms as before, not a spend per term nor a test of the term's index per term to decide the spend (log 0057-1)
    let chunks = poly.monomials.chunks(CHUNK).zip(poly.coefs.chunks(CHUNK));
    for (j, (ms, cs)) in chunks.enumerate() {
        let mark = bytes.len();
        for (i, (&m, &c)) in ms.iter().zip(cs).enumerate() {
            let k = j * CHUNK + i;
            let v = field.value(c);
            let write = |w: &mut Spare<'_>| {
                if k > 0 {
                    w.byte(b'+');
                }
                if table.degree(m) == 0 {
                    w.decimal(v);
                    return;
                }
                if v != 1 {
                    w.decimal(v);
                    w.byte(b'*');
                }
                // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
                table.unpack(m, &mut e);
                w.monomial(names, &e);
            };
            // SAFETY: a term is at most a `+`, a coefficient below 2^64, a
            // `*`, and the monomial, which `term_longest` counts.
            unsafe { spare(bytes, longest, write) };
        }
        budget.spend(bytes.len() - mark)?;
    }
    Ok(())
}

/// The most decimal digits of a u64: 20.
const U64_LONGEST: usize = u64::MAX.ilog10() as usize + 1;

/// The most decimal digits of a u32: 10.
const U32_LONGEST: usize = u32::MAX.ilog10() as usize + 1;

/// The bytes of a term's text beside its coefficient and its monomial: a
/// sign or a `+`, and a `*`.
pub(crate) const TERM_MARKS: usize = 2;

/// The bytes of a variable's text in a monomial beside its name and its
/// exponent: a `*` and a `^`.
const POWER_MARKS: usize = 2;

/// The longest text of a term over GF(p) on these names: a `+`, a
/// coefficient below 2^64 of at most 20 digits, and a `*`, then per
/// variable what `monomial_longest` counts.
pub(crate) fn term_longest(names: &[&str]) -> usize {
    TERM_MARKS + U64_LONGEST + monomial_longest(names)
}

/// The longest text of a monomial on these names: per variable its name, a
/// `*`, a `^`, and an exponent below 2^32 of at most 10 digits.
pub(crate) fn monomial_longest(names: &[&str]) -> usize {
    names
        .iter()
        .map(|name| name.len() + POWER_MARKS + U32_LONGEST)
        .sum()
}

// result-time.4 design -- each term is written through a writer over the vector's spare capacity, reserved for the term's longest text, without bounds checks outside the test profile, and the vector's length is set once per term, not a push per byte that checks the capacity and stores the length (log 0033-1, 0042-1)
/// A writer of bytes into a vector's spare capacity, from its start, that
/// counts what it wrote. Its writes have no bounds check outside the test
/// profile, whose debug assertions check them.
pub(crate) struct Spare<'a> {
    /// The spare capacity reserved, its first `len` bytes written.
    spare: &'a mut [MaybeUninit<u8>],
    len: usize,
}

/// Reserves `longest` bytes in `out`, lets `write` write at most that
/// many through a `Spare`, and makes them part of `out`.
///
/// # Safety
///
/// `write` writes at most `longest` bytes.
#[inline(always)]
pub(crate) unsafe fn spare(out: &mut Vec<u8>, longest: usize, write: impl FnOnce(&mut Spare<'_>)) {
    out.reserve(longest);
    let len = out.len();
    let mut w = Spare {
        spare: &mut out.spare_capacity_mut()[..longest],
        len: 0,
    };
    write(&mut w);
    let written = w.len;
    // SAFETY: `reserve` made room for `longest` bytes past `len`, and the
    // writer has written the first `written` of them, at most `longest`.
    unsafe { out.set_len(len + written) };
}

// result-time.5 practice: two-digit lookup table -- a number below 2^64 is written two digits at a time from a table of the hundred pairs, not one digit at a time into a buffer read back through `str::from_utf8`
/// The hundred pairs of digits `00` to `99`, two bytes each.
const PAIRS: [u8; 200] = {
    let mut pairs = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        pairs[2 * i] = b"0123456789"[i / 10];
        pairs[2 * i + 1] = b"0123456789"[i % 10];
        i += 1;
    }
    pairs
};

impl Spare<'_> {
    #[inline(always)]
    pub(crate) fn byte(&mut self, b: u8) {
        debug_assert!(
            self.len < self.spare.len(),
            "a byte past the {reserved} reserved",
            reserved = self.spare.len()
        );
        // SAFETY: `spare`'s caller guarantees that its writer writes at most
        // the bytes reserved, the length of `spare`.
        unsafe { self.spare.get_unchecked_mut(self.len).write(b) };
        self.len += 1;
    }

    #[inline(always)]
    pub(crate) fn bytes(&mut self, bs: &[u8]) {
        debug_assert!(
            self.len + bs.len() <= self.spare.len(),
            "{over} bytes past the {reserved} reserved",
            over = self.len + bs.len() - self.spare.len(),
            reserved = self.spare.len()
        );
        // SAFETY: as in `byte`; `bs` is no part of the spare capacity, which
        // `spare` borrows mutably.
        unsafe {
            let to = self.spare.as_mut_ptr().add(self.len).cast::<u8>();
            std::ptr::copy_nonoverlapping(bs.as_ptr(), to, bs.len());
        }
        self.len += bs.len();
    }

    // text-time.4 design -- a kept text of at most 32 bytes is copied by one copy of the constant 32 bytes where the term's writer holds 32 bytes past what it has written, and every other by its length, not every copy by its length or a reservation 32 bytes larger per term (log 0048-1)
    /// The text whose length byte starts `from`, which holds the text and
    /// `COPY` bytes past it. A text of at most `COPY` bytes is copied by one
    /// copy of `COPY` bytes where the reservation holds them past what is
    /// written, and its length alone counts as written.
    #[inline(always)]
    fn kept(&mut self, from: &[u8]) {
        let n = from[0] as usize;
        let text = &from[1..];
        debug_assert!(
            text.len() >= n + COPY,
            "the arena ends before the text's copy"
        );
        if n <= COPY && self.spare.len() - self.len >= COPY {
            // SAFETY: `text` holds at least `COPY` bytes, and the spare
            // capacity `COPY` past `len`; `text` is no part of the spare
            // capacity, which `spare` borrows mutably.
            unsafe {
                let to = self.spare.as_mut_ptr().add(self.len).cast::<u8>();
                std::ptr::copy_nonoverlapping(text.as_ptr(), to, COPY);
            }
            self.len += n;
        } else {
            self.bytes(&text[..n]);
        }
    }

    /// v in decimal, at most 20 bytes.
    #[inline(always)]
    pub(crate) fn decimal(&mut self, v: u64) {
        if v < 10 {
            self.byte(b'0' + Narrow::<u8>::narrow(v));
            return;
        }
        let mut digits = [0u8; U64_LONGEST];
        let mut k = digits.len();
        let mut v = v;
        while v >= 100 {
            let r = (v % 100) as usize;
            v /= 100;
            k -= 2;
            digits[k..k + 2].copy_from_slice(&PAIRS[2 * r..2 * r + 2]);
        }
        if v >= 10 {
            #[allow(clippy::cast_possible_truncation, reason = "v lies in 10..100")]
            let r = v as usize;
            k -= 2;
            digits[k..k + 2].copy_from_slice(&PAIRS[2 * r..2 * r + 2]);
        } else {
            k -= 1;
            digits[k] = b'0' + Narrow::<u8>::narrow(v);
        }
        self.bytes(&digits[k..]);
    }

    /// The monomial of these exponents, one per variable, not all 0: its
    /// variables of nonzero exponent joined by `*`, each as its name, then
    /// `^` and the exponent where that is not 1. At most what
    /// `monomial_longest` counts.
    #[inline(always)]
    pub(crate) fn monomial(&mut self, names: &[&str], exponents: &[u32]) {
        debug_assert_eq!(names.len(), exponents.len(), "a name per exponent");
        debug_assert!(exponents.iter().any(|&e| e > 0), "the monomial is not 1");
        let mut first = true;
        for (name, &e) in names.iter().zip(exponents) {
            if e == 0 {
                continue;
            }
            if !first {
                self.byte(b'*');
            }
            first = false;
            self.bytes(name.as_bytes());
            if e > 1 {
                self.byte(b'^');
                self.decimal(u64::from(e));
            }
        }
    }

    /// v below 10^19 as exactly 19 digits, zeros first: nine pairs of
    /// digits and one digit.
    #[inline(always)]
    fn chunk(&mut self, v: u64) {
        debug_assert!(v < TEN19, "{v} has more than {U64_DIGITS} digits");
        let mut digits = [0u8; U64_DIGITS];
        let mut v = v;
        for k in (0..U64_DIGITS / 2).rev() {
            let r = (v % 100) as usize;
            v /= 100;
            digits[1 + 2 * k..3 + 2 * k].copy_from_slice(&PAIRS[2 * r..2 * r + 2]);
        }
        digits[0] = b'0' + Narrow::<u8>::narrow(v);
        self.bytes(&digits);
    }
}

/// Appends v in decimal.
pub(crate) fn push_decimal(out: &mut Vec<u8>, v: u64) {
    // SAFETY: a number below 2^64 has at most 20 digits.
    unsafe { spare(out, U64_LONGEST, |w| w.decimal(v)) };
}

// result-time.9 design -- the writers of a number and of a monomial stand in text.rs, and the writer over Q calls them, not the writer of a large number in rational.rs beside the writer over Q (log 0033-1)
/// Appends x in decimal.
pub(crate) fn push_big(out: &mut Vec<u8>, x: &BigUint) {
    let mut limbs = x.iter_u64_digits();
    let Some(low) = limbs.next() else {
        out.push(b'0');
        return;
    };
    match limbs.next() {
        None => push_decimal(out, low),
        Some(_) if x.bits() <= 64 * SPLIT_LIMBS as u64 => push_passes(out, x),
        // q-basis-text-time.1 cited: brent-2010 §1.7.2, Algorithm 1.26 -- for (Q, R) = DivRem(A, B^k), A is written as Q followed by R padded to k digits, in O(M(n) log n), where IntegerOutput's divisions by B take O(n²)
        Some(_) => push_split(out, x),
    }
}

// q-basis-text-time.8 tuned -- SPLIT_LIMBS is 12: an integer of 2 to 12 limbs is written by the passes as text-time.5 wrote it, not the passes up to 2, 4, 8, 16, or 24 limbs (log 0086-1)
/// The most limbs of an integer `push_passes` writes. `push_split` writes
/// a larger one.
const SPLIT_LIMBS: usize = 12;

/// The most chunks of nineteen digits of an integer of `SPLIT_LIMBS`
/// limbs. An integer of n limbs has at most ⌈64n·log₁₀2/19⌉ chunks, at
/// most n + 1 for n up to 71.
const PASS_CHUNKS: usize = SPLIT_LIMBS + 1;

/// 10^19, the largest power of ten below 2^64.
#[allow(
    clippy::cast_possible_truncation,
    reason = "U64_DIGITS, 19, fits a u32"
)]
const TEN19: u64 = 10u64.pow(U64_DIGITS as u32);

/// Appends x, of 2 to `SPLIT_LIMBS` limbs, in decimal by `passes`.
fn push_passes(out: &mut Vec<u8>, x: &BigUint) {
    debug_assert!(
        (2..=SPLIT_LIMBS).contains(&x.iter_u64_digits().len()),
        "{limbs} limbs outside the passes' 2 to {SPLIT_LIMBS}",
        limbs = x.iter_u64_digits().len()
    );
    let mut limbs = [0u64; SPLIT_LIMBS];
    let mut live = 0;
    for (index, limb) in x.iter_u64_digits().enumerate() {
        limbs[index] = limb;
        live = index + 1;
    }
    let mut chunks = [0u64; PASS_CHUNKS];
    let count = passes(&mut limbs[..live], &mut chunks);
    push_chunks(out, &chunks[PASS_CHUNKS - count..]);
}

/// Divides x, on `limbs`, least significant first, with no zero limb at
/// the top, by 10^19 in passes until it is 0. Each pass runs from the top
/// limb down, and its remainder is x's next chunk of nineteen digits, the
/// least significant first. The chunks fill the end of `chunks`, the most
/// significant first, and their count is returned. x is below
/// 10^(19·chunks.len()).
fn passes(limbs: &mut [u64], chunks: &mut [u64]) -> usize {
    debug_assert!(limbs.last() != Some(&0), "a zero limb at the top");
    let mut live = limbs.len();
    let mut count = 0;
    while live > 0 {
        let mut remainder = 0u64;
        for limb in limbs[..live].iter_mut().rev() {
            // text-time.6 design -- each step divides a u128 by the constant 10^19, which the compiler computes, not by Möller and Granlund's 2 by 1 division with a precomputed reciprocal (log 0048-1)
            let dividend = u128::from(remainder) << 64 | u128::from(*limb);
            *limb = (dividend / u128::from(TEN19)).narrow();
            remainder = (dividend % u128::from(TEN19)).narrow();
        }
        count += 1;
        chunks[chunks.len() - count] = remainder;
        while live > 0 && limbs[live - 1] == 0 {
            live -= 1;
        }
    }
    count
}

/// Appends the integer of these chunks of nineteen digits, the most
/// significant first: the first chunk that is not 0 in decimal, and each
/// after it as nineteen digits. Where every chunk is 0 it appends `0`.
fn push_chunks(out: &mut Vec<u8>, chunks: &[u64]) {
    let Some(first) = chunks.iter().position(|&chunk| chunk != 0) else {
        out.push(b'0');
        return;
    };
    let (top, rest) = (chunks[first], &chunks[first + 1..]);
    // SAFETY: a number below 2^64 has at most 20 digits, and each chunk
    // after it 19.
    unsafe {
        spare(out, U64_LONGEST + U64_DIGITS * rest.len(), |writer| {
            writer.decimal(top);
            for &chunk in rest {
                writer.chunk(chunk);
            }
        });
    };
}

// q-basis-text-time.9 tuned -- a part of at most LEAF_CHUNKS, 8, chunks is written by the passes, exactly its chunks with zeros at the top, not leaves of 2, 3, 4, 6, 16, or 32 chunks nor leaves by a fraction whose chunks come by multiplications (log 0086-1)
/// The most chunks of a part `leaf` writes. A part of 8 chunks is below
/// 10^152 < 2^512, so of at most 8 limbs.
const LEAF_CHUNKS: usize = 8;

// q-basis-text-time.6 design -- the powers D_j = 10^(19·2^j) and their reciprocals are made once per process, at the first integer that needs level j, and kept in a static array of `OnceLock`s for every later call and thread, not made per call, per text, or per integer (not measured)
/// A level of the splits: the power Dⱼ = 10^(19·2^j) and its reciprocal
/// μⱼ = ⌊2^(128l)/Dⱼ⌋, l the limbs of Dⱼ, each on limbs, least
/// significant first. Dⱼ has at most 2^j limbs, as 10^19 < 2^64. μⱼ has
/// l + 1 limbs: 2^(64(l − 1)) < Dⱼ < 2^(64l).
struct Level {
    power: Vec<u64>,
    reciprocal: Vec<u64>,
}

/// The levels `LEVEL` holds: a part of k chunks takes the level j with
/// 2^j < k, and k, a usize, is below 2^64.
const LEVELS: usize = usize::BITS as usize;

/// Level j, made by the first integer that takes it.
static LEVEL: [OnceLock<Level>; LEVELS] = [const { OnceLock::new() }; LEVELS];

/// Level j, made at its first use.
fn level(j: usize) -> &'static Level {
    debug_assert!(j < LEVELS, "level {j} of {LEVELS}");
    #[allow(
        clippy::expect_used,
        reason = "j < LEVELS, as 2^j is below a count of chunks, a usize"
    )]
    let cell = LEVEL
        .get(j)
        .expect("2^j is below a count of chunks, a usize");
    cell.get_or_init(|| {
        let power = num_traits::pow(BigUint::from(TEN19), 1 << j);
        let limbs = power.iter_u64_digits().len();
        let reciprocal = (BigUint::from(1u32) << (128 * limbs)) / &power;
        Level {
            power: power.iter_u64_digits().collect(),
            reciprocal: reciprocal.iter_u64_digits().collect(),
        }
    })
}

// q-basis-text-time.2 design -- a part of k chunks splits at h = 2^j, the largest power of two below k, by D_j, not at ⌈k/2⌉ by a power and a reciprocal per h (log 0086-1)
/// The level of a part of k chunks, k ≥ 2: the j with 2^j < k ≤ 2^(j+1).
fn level_of(k: usize) -> usize {
    debug_assert!(k >= 2, "a part of {k} chunks splits at no level");
    (k - 1).ilog2() as usize
}

/// The scratch of the splits from level j down: at each level i on the
/// way a quotient and a remainder of lᵢ + 1 limbs, and at the deepest a
/// product's columns of at most lᵢ + 3, with lᵢ ≤ 2^i: the sum over
/// i ≤ j of 2·(2^i + 1), and 2^j + 3, at most 5·2^j + 2j + 3 limbs.
fn scratch_limbs(j: usize) -> usize {
    (5 << j) + 2 * j + 3
}

// q-basis-text-time.1 cited: brent-2010 §1.7.2, Algorithm 1.26 -- for (Q, R) = DivRem(A, B^k), A is written as Q followed by R padded to k digits, in O(M(n) log n), where IntegerOutput's divisions by B take O(n²)
// q-basis-text-time.7 design -- the scratch of a conversion, x's limbs, its chunks, and the splits' parts and columns, is one vector allocated per integer of more than SPLIT_LIMBS limbs, not a scratch kept from integer to integer, thread-local or a converter's own (not measured)
/// Appends x, of more than `SPLIT_LIMBS` limbs, in decimal: its k chunks
/// of nineteen digits by `split`, k = ⌊bits/63⌋ + 1, so that
/// 10^(19k) > 2^(63k) > x.
fn push_split(out: &mut Vec<u8>, x: &BigUint) {
    debug_assert!(
        x.bits() > 64 * SPLIT_LIMBS as u64,
        "{bits} bits within the passes' {SPLIT_LIMBS} limbs",
        bits = x.bits()
    );
    let n = x.iter_u64_digits().len();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a number in memory has fewer bits than usize counts"
    )]
    let k = (x.bits() / 63) as usize + 1;
    let mut scratch = vec![0u64; n + k + scratch_limbs(level_of(k))];
    let (limbs, rest) = scratch.split_at_mut(n);
    for (limb, v) in limbs.iter_mut().zip(x.iter_u64_digits()) {
        *limb = v;
    }
    let (chunks, rest) = rest.split_at_mut(k);
    split(limbs, chunks, rest);
    push_chunks(out, chunks);
}

/// Writes x's chunks of nineteen digits into `chunks`, the most
/// significant first, 0 at the top where x has fewer. x, on limbs, least
/// significant first, is below 10^(19k), k = `chunks.len()`, and
/// `scratch` holds `scratch_limbs` of k's level. A part of more than
/// `LEAF_CHUNKS` chunks, Dⱼ its level's power, is x = q·Dⱼ + r: q's
/// k − 2^j chunks, then r's 2^j.
fn split(x: &[u64], chunks: &mut [u64], scratch: &mut [u64]) {
    let x = trimmed(x);
    let k = chunks.len();
    if x.is_empty() {
        chunks.fill(0);
        return;
    }
    if k <= LEAF_CHUNKS {
        leaf(x, chunks);
        return;
    }
    let j = level_of(k);
    debug_assert!(
        scratch.len() >= scratch_limbs(j),
        "{len} limbs of scratch for level {j}",
        len = scratch.len()
    );
    let Level { power, reciprocal } = level(j);
    let (high, low) = chunks.split_at_mut(k - (1 << j));
    if lehmer::cmp(x, power).is_lt() {
        high.fill(0);
        split(x, low, scratch);
        return;
    }
    let (quotient, rest) = scratch.split_at_mut(power.len() + 1);
    let (remainder, rest) = rest.split_at_mut(power.len() + 1);
    divide(x, power, reciprocal, (quotient, remainder), rest);
    split(quotient, high, rest);
    split(remainder, low, rest);
}

// q-basis-text-time.3 cited: barrett-1987 §Longmod procedure, Diagram Five, via vanderhoeven-2016 §2.2.1, Function 6 and Proposition 1 -- with R = b^(2n) div M, the n + 1 leading digits of W times R, their product's leading digits times M subtracted from W modulo b^(n+1), leave x below 3M, and at most two subtractions of M give W mod M
/// x = q·D + r with r < D, for D ≤ x < D², D on l limbs and μ its level's
/// reciprocal, into q and r of l + 1 limbs each; `scratch` holds l + 3
/// limbs. q starts at Barrett's quotient, ⌊b·μ/2^(64(l+1))⌋ for
/// b = ⌊x/2^(64(l−1))⌋, and r at x − q·D modulo 2^(64(l+1)), which holds
/// it: with s = 64(l − 1) and t = 64(l + 1), Proposition 1 puts Barrett's
/// quotient at most 2 below ⌊x/D⌋, and the columns left out at most 1
/// more. D is subtracted and q raised while r is D or more.
fn divide(
    x: &[u64],
    power: &[u64],
    reciprocal: &[u64],
    (q, r): (&mut [u64], &mut [u64]),
    scratch: &mut [u64],
) {
    let l = power.len();
    debug_assert!(
        x.len() >= l && x.len() <= 2 * l && q.len() == l + 1 && r.len() == l + 1,
        "x of {x_len} limbs, D of {l}, q of {q_len}, r of {r_len}",
        x_len = x.len(),
        q_len = q.len(),
        r_len = r.len()
    );
    let high = &x[l - 1..];
    // q-basis-text-time.4 cited: barrett-1987 §Longmod procedure -- each long multiplication of the reduction calculates only half its product, since the other half is not required
    let product = &mut scratch[..high.len() + reciprocal.len() - (l - 1)];
    columns(high, reciprocal, l - 1, product);
    // The columns from l + 1 up, the product shifted down by t bits.
    let quotient = &product[2..];
    q[..quotient.len()].copy_from_slice(quotient);
    q[quotient.len()..].fill(0);
    let low = &x[..x.len().min(l + 1)];
    r[..low.len()].copy_from_slice(low);
    r[low.len()..].fill(0);
    let live = trimmed(q);
    if !live.is_empty() {
        let product = &mut scratch[..=l];
        columns(live, power, 0, product);
        sub_in_place(r, product);
    }
    let mut corrections = 0;
    while lehmer::cmp(trimmed(r), power).is_ge() {
        sub_in_place(r, power);
        increment(q);
        corrections += 1;
    }
    debug_assert!(
        corrections <= 3,
        "Barrett's quotient {corrections} below x's"
    );
}

// q-basis-text-time.5 practice: Comba multiplication -- the products of a split are taken column by column, each column's products summed into a three-word accumulator and its low word stored, not row by row with each row's carries through the result in memory, nor by num-bigint's product, which allocates each
/// The limbs `from` to `from + out.len()` of the sum of aᵢ·bₖ·2^(64(i + k))
/// over i + k ≥ `from`, into `out`: the columns of a·b from `from` up, the
/// columns below and their carries left out. Each column's products are
/// summed into a three-word accumulator, whose low word is the column's
/// limb and whose rest carries into the next column. A column past the
/// last product holds the carry, and 0 after it.
fn columns(a: &[u64], b: &[u64], from: usize, out: &mut [u64]) {
    debug_assert!(!a.is_empty() && !b.is_empty(), "a product of no limbs");
    let mut low = 0u128;
    let mut high = 0u64;
    for (column, limb) in (from..).zip(out.iter_mut()) {
        // a_i·b_(column − i) for i from `first` to `end`, exclusive.
        let first = (column + 1).saturating_sub(b.len());
        let end = a.len().min(column + 1);
        if first < end {
            let b_part = &b[column + 1 - end..=column - first];
            for (&ai, &bi) in a[first..end].iter().zip(b_part.iter().rev()) {
                let (sum, carried) = low.overflowing_add(u128::from(ai) * u128::from(bi));
                low = sum;
                high += u64::from(carried);
            }
        }
        *limb = (low & u128::from(u64::MAX)).narrow();
        low = low >> 64 | u128::from(high) << 64;
        high = 0;
    }
}

/// r − y modulo 2^(64·|r|) into r, y of at most r's limbs.
fn sub_in_place(r: &mut [u64], y: &[u64]) {
    debug_assert!(
        y.len() <= r.len(),
        "{y_len} limbs from {r_len}",
        y_len = y.len(),
        r_len = r.len()
    );
    let (head, tail) = r.split_at_mut(y.len());
    let mut borrow = false;
    for (limb, &v) in head.iter_mut().zip(y) {
        let (difference, b1) = limb.overflowing_sub(v);
        let (difference, b2) = difference.overflowing_sub(u64::from(borrow));
        *limb = difference;
        borrow = b1 || b2;
    }
    for limb in tail {
        if !borrow {
            break;
        }
        (*limb, borrow) = limb.overflowing_sub(1);
    }
}

/// q + 1 into q, which holds it.
fn increment(q: &mut [u64]) {
    let carried = q.iter_mut().all(|limb| {
        let carried;
        (*limb, carried) = limb.overflowing_add(1);
        carried
    });
    debug_assert!(!carried, "q + 1 past q's {len} limbs", len = q.len());
}

/// Writes x's chunks into `chunks`, the most significant first, 0 at the
/// top where x has fewer, by `passes`: x, of at most `LEAF_CHUNKS` limbs
/// and no zero limb at the top, is below 10^(19·chunks.len()), and
/// `chunks` holds at most `LEAF_CHUNKS`.
fn leaf(x: &[u64], chunks: &mut [u64]) {
    debug_assert!(
        x.len() <= LEAF_CHUNKS && chunks.len() <= LEAF_CHUNKS,
        "a leaf of {limbs} limbs and {chunks} chunks",
        limbs = x.len(),
        chunks = chunks.len()
    );
    let mut limbs = [0u64; LEAF_CHUNKS];
    let limbs = &mut limbs[..x.len()];
    limbs.copy_from_slice(x);
    let count = passes(limbs, chunks);
    let top = chunks.len() - count;
    chunks[..top].fill(0);
}

/// x without its zero limbs at the top.
fn trimmed(x: &[u64]) -> &[u64] {
    let live = x
        .iter()
        .rposition(|&limb| limb != 0)
        .map_or(0, |top| top + 1);
    &x[..live]
}

// text-copy-peak.1 design -- a text is handed over a chunk at a time, and the whole pages of it behind each chunk are given back to the system, not handed over whole and its pages given back when it is dropped, nor handed over after the measured call, nor written by the core into the receiver's memory (log 0081-1)
// text-copy-peak.4 design -- the pages are given back by madvise(MADV_DONTNEED), not by shrinking the vector, which the allocator may move or keep whole (not measured)
// text-copy-peak.6 design -- each chunk is checked ASCII before it is handed, and a byte past ASCII ends the hand-over with an error, not the text trusted unchecked, nor the whole text checked before the first chunk (not measured)
// text-copy-peak.10 design -- the hand-over and its madvise are a function of the core's text, which the Python module calls with its copy, not the module calling madvise itself (not measured)
/// Hands `text` to `put` in chunks of at most `chunk` bytes, in order,
/// each with its offset in the text, and gives back to the system the
/// whole pages of the text behind each chunk, so that the text and what
/// `put` builds of it stand about once. Each chunk is checked ASCII before
/// it is handed. The text is consumed: a page given back reads as zero.
///
/// # Errors
///
/// `InvalidData` where a byte is past ASCII, the chunk that holds it not
/// handed; `Unsupported` where the page size cannot be read; and the
/// error of `madvise` where it fails.
pub fn hand_over(
    text: String,
    chunk: NonZeroUsize,
    mut put: impl FnMut(usize, &[u8]),
) -> io::Result<()> {
    // text-copy-peak.9 practice: page size from sysconf -- the page size is the system's, read by sysconf, not 4096 as a constant
    let page = budget::page_size()
        .and_then(|p| usize::try_from(p).ok())
        .filter(|p| p.is_power_of_two())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::Unsupported, "the page size cannot be read")
        })?;
    let mut bytes = text.into_bytes();
    let len = bytes.len();
    let base = bytes.as_ptr() as usize;
    // The offset of the first whole page not given back.
    let mut given = base.next_multiple_of(page) - base;
    let mut from = 0;
    while from < len {
        let to = len.min(from.saturating_add(chunk.get()));
        let part = &bytes[from..to];
        if !part.is_ascii() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the text's chunk at byte {from} holds a byte past ASCII"),
            ));
        }
        put(from, part);
        // The end of the last whole page within bytes[..to].
        let end = ((base + to) / page * page).saturating_sub(base);
        if end > given {
            // SAFETY: bytes[given..end] are whole pages of the vector's
            // buffer, which this function owns, and `put` has been handed
            // them; nothing reads them again before the drop. madvise
            // changes no memory but those pages' bytes, which a vector of
            // bytes may hold whatever they are: zero in private anonymous
            // memory (madvise-2, MADV_DONTNEED).
            let r = unsafe {
                libc::madvise(
                    bytes.as_mut_ptr().add(given).cast(),
                    end - given,
                    libc::MADV_DONTNEED,
                )
            };
            if r != 0 {
                return Err(io::Error::last_os_error());
            }
            given = end;
        }
        from = to;
    }
    Ok(())
}

/// 30103/100000, numerator and denominator, a fraction just above log₁₀ 2
/// = 0.30102999….
const LOG10_2: (u64, u64) = (30103, 100_000);

/// At least the decimal digits of a number of `bits` bits:
/// ⌊bits·30103/100000⌋ + 1, by `LOG10_2`. 30103/100000 exceeds log₁₀ 2, so
/// the bound is at least ⌊bits·log₁₀ 2⌋ + 1, the most digits such a number
/// has.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a number in memory has fewer digits than usize counts"
)]
pub(crate) fn digits_bound(bits: u64) -> usize {
    let (num, den) = LOG10_2;
    (bits * num / den + 1) as usize
}

/// Appends the monomial of these exponents, one per variable, not all 0:
/// its variables of nonzero exponent joined by `*`, each as its name,
/// then `^` and the exponent where that is not 1.
pub(crate) fn push_monomial(out: &mut Vec<u8>, names: &[&str], exponents: &[u32]) {
    debug_assert_eq!(exponents.len(), names.len());
    // SAFETY: `monomial` writes at most what `monomial_longest` counts.
    unsafe {
        spare(out, monomial_longest(names), |w| {
            w.monomial(names, exponents);
        });
    };
}
