//! Reads the instance, the basis, and the cofactor matrix.
//!
//! ```text
//! instance := names NL integer (NL polys?)?    polys may span lines
//! names    := name ("," name)*                  distinct names
//! basis    := comment* "[" polys? "]" ":"
//! matrix   := comment* "[" (row ("," row)*)? "]" ":"
//! row      := "[" polys? "]"
//! polys    := poly ("," poly)*
//! poly     := sign? term (sign term)*
//! term     := factor ("*" factor)*
//! factor   := integer ("/" integer)? | name ("^" integer)?
//! comment  := "#" anything up to the line break
//! name     := [A-Za-z_][A-Za-z0-9_]*
//! ```
//!
//! Spaces, tabs, line feeds, form feeds, and carriage returns may stand
//! between any two tokens. In the instance, the first line feed ends the
//! names and the second ends the characteristic. Nothing but whitespace may
//! follow the closing `:`.
//!
//! The characteristic is 0 or a prime below 2^64. A term's degree is at
//! most 2^31 − 1 in the instance and at most `u32::MAX` in the basis and the
//! cofactor matrix.

// checker-gfp.3 design -- the grammar is tolerant: whitespace anywhere, a term any product of numbers, fractions, and powers, like terms combined, terms and elements in any order, comment lines before the bracket (not measured)

use std::collections::HashMap;

use crate::field;
use crate::narrow::Narrow;
use crate::order::Cmp;
use crate::poll::Poll;
use crate::{Error, Order, Part};

// checker-stop-latency.8 design -- the reading counts 1 + b/16 units a term after it, b its bytes, and ends where the poll stops, the polynomial it reads unfinished and a reading error past the term the stop falls in not reached, not the texts read without a poll (log 0059-1)
/// Bytes of a term's text that count as one unit of work beside the term's
/// own (`Poll`).
const BYTES: usize = 16;

/// The cap of an exponent as it is read: past `u32::MAX`, which every
/// degree bound of the texts keeps, so that a capped exponent still
/// overflows, and ten times it with a digit added stays within a u64.
const EXPONENT_CAP: u64 = 1 << 33;

/// The cap of a characteristic as it is read: past 2^64, so that a capped
/// characteristic is still refused, and ten times it with a digit added
/// stays within a u128.
const CHARACTERISTIC_CAP: u128 = 1 << 65;

// checker-stop-latency.9 design -- the `Numbers` functions take the poll as a parameter, not a poll held in `Eval` behind a `RefCell` or a `Numbers` wrapper that holds the poll for the exact reading (not measured)
/// The arithmetic a side of the checker reads its numbers with: GF(p)'s
/// residues, Q's fractions, or in the cheap mode a row's values at the
/// points (`cheap::Eval`). Each function counts the work of its arithmetic
/// with `poll` before it does it: over Q a product, a quotient, and a gcd
/// as `rational::cost` and `rational::gcd_cost` count them, save the
/// quotients by a term's gcd, and in `finish` the gcd of the content and
/// the denominator and the quotients by it, over GF(p) an
/// inverse `gfp::INVERSE` units, in the cheap mode a term at a point a
/// unit, and 19 digits of a numerator, and over Q of a denominator, a unit
/// at each point over Q and once for all points over GF(p), over Q with a
/// denominator's inverse `gfp::INVERSE` units at
/// each point, and `finish` the comparisons of its order test. Once `poll`
/// has stopped, what it computes means nothing.
pub trait Numbers {
    /// A term's coefficient while the term is read.
    type Term;
    /// A polynomial's terms as they are read.
    type Terms: Default;
    /// A polynomial once read.
    type Poly: Default;
    fn one(&self) -> Self::Term;
    /// c times the integer whose decimal digits are `digits`.
    fn times(&self, c: &mut Self::Term, digits: &[u8], poll: &mut Poll);
    /// c over the integer whose decimal digits are `digits`; None where that
    /// integer has no inverse.
    fn over(&self, c: &mut Self::Term, digits: &[u8], poll: &mut Poll) -> Option<()>;
    /// Appends c·m, negated where `negative`, unless c is zero.
    fn push(
        &self,
        terms: &mut Self::Terms,
        m: &[u32],
        c: Self::Term,
        negative: bool,
        poll: &mut Poll,
    );
    /// The sum of `terms` into `out`, its terms in the ring's order, and
    /// `terms` emptied.
    fn finish(&self, terms: &mut Self::Terms, ring: &Ring, out: &mut Self::Poly, poll: &mut Poll);
}

/// The variables and the characteristic of an instance, and the order the
/// check rules in.
pub struct Ring {
    /// p for GF(p), a prime below 2^64, or 0 for Q.
    pub characteristic: u64,
    /// The width of a monomial: the degree and one exponent per variable.
    pub w: usize,
    pub order: Cmp,
    names: HashMap<Vec<u8>, usize>,
}

impl Ring {
    /// The variables' names, in the instance's order.
    pub fn names(&self) -> Vec<&[u8]> {
        let mut names = vec![&[][..]; self.names.len()];
        for (name, &v) in &self.names {
            names[v] = name;
        }
        names
    }
}

/// An instance read up to its generators.
pub struct Header<'a> {
    pub ring: Ring,
    body: &'a [u8],
}

struct Cursor<'a> {
    text: &'a [u8],
    pos: usize,
    line: usize,
    part: Part,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a [u8], part: Part, line: usize) -> Cursor<'a> {
        Cursor {
            text,
            pos: 0,
            line,
            part,
        }
    }

    // instance-grammar-agreement.1 design -- the checker takes the engine's reading of the instance text, and the engine's reader stays as it is (not measured)
    // instance-grammar-agreement.2 design -- whitespace is ASCII space, tab, line feed, form feed, and carriage return in the instance, the basis, and the cofactor text alike, not the form feed in the instance alone (not measured)
    fn skip_space(&mut self) {
        while let Some(&b) = self.text.get(self.pos) {
            match b {
                b'\n' => self.line += 1,
                b' ' | b'\t' | b'\x0c' | b'\r' => {}
                _ => break,
            }
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_space();
        self.text.get(self.pos).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, b: u8, expected: &'static str) -> Result<(), Error> {
        if self.eat(b) {
            Ok(())
        } else {
            Err(self.syntax(expected))
        }
    }

    fn syntax(&self, expected: &'static str) -> Error {
        Error::Syntax {
            part: self.part,
            line: self.line,
            expected,
        }
    }

    fn at_digit(&mut self) -> bool {
        self.peek().is_some_and(|b| b.is_ascii_digit())
    }

    fn comments(&mut self) {
        while self.peek() == Some(b'#') {
            while self.text.get(self.pos).is_some_and(|&b| b != b'\n') {
                self.pos += 1;
            }
        }
    }

    fn end(&mut self) -> Result<(), Error> {
        match self.peek() {
            None => Ok(()),
            Some(_) => Err(self.syntax("nothing after the closing `:`")),
        }
    }

    /// The decimal digits at the cursor.
    fn digits(&mut self) -> &'a [u8] {
        let start = self.pos;
        while self.text.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        &self.text[start..self.pos]
    }

    /// The integer at the cursor as an exponent, at most `EXPONENT_CAP`.
    fn exponent(&mut self) -> u64 {
        let mut acc = 0u64;
        while let Some(&b) = self.text.get(self.pos) {
            if !b.is_ascii_digit() {
                break;
            }
            acc = (acc * 10 + u64::from(b - b'0')).min(EXPONENT_CAP);
            self.pos += 1;
        }
        acc
    }

    fn overflow(&self) -> Error {
        Error::Overflow {
            part: self.part,
            line: self.line,
        }
    }

    /// The name at the cursor, which starts with a letter or `_`.
    fn name(&mut self) -> &'a [u8] {
        let start = self.pos;
        while self
            .text
            .get(self.pos)
            .is_some_and(|&b| b.is_ascii_alphanumeric() || b == b'_')
        {
            self.pos += 1;
        }
        &self.text[start..self.pos]
    }

    fn at_name(&mut self) -> bool {
        self.peek()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
    }

    /// One term, appended to `terms` unless its coefficient is zero. A
    /// degree past 2^31 − 1 in the instance, or past `u32` in the basis and
    /// the cofactors, is `Error::Overflow` once the term is read.
    fn term<N: Numbers>(
        &mut self,
        ring: &Ring,
        numbers: &N,
        negative: bool,
        exps: &mut Vec<u32>,
        terms: &mut N::Terms,
        poll: &mut Poll,
    ) -> Result<(), Error> {
        let start = self.pos;
        exps.clear();
        exps.resize(ring.w, 0);
        let mut c = numbers.one();
        // instance-grammar-agreement.6 design -- the exponents and the degree are summed without a bound while the factors are read, and the bound is tested once the term is read, not at the first exponent or partial degree past `u32` (not measured)
        let mut degree = 0u64;
        loop {
            if self.at_digit() {
                numbers.times(&mut c, self.digits(), poll);
                if self.eat(b'/') {
                    if !self.at_digit() {
                        return Err(self.syntax("a denominator"));
                    }
                    let d = self.digits();
                    numbers.over(&mut c, d, poll).ok_or(Error::NoInverse {
                        part: self.part,
                        line: self.line,
                    })?;
                }
            } else if self.at_name() {
                let name = self.name();
                let Some(&v) = ring.names.get(name) else {
                    return Err(Error::UnknownVariable {
                        part: self.part,
                        line: self.line,
                        name: String::from_utf8_lossy(name).into_owned(),
                    });
                };
                let e = if self.eat(b'^') {
                    if !self.at_digit() {
                        return Err(self.syntax("an exponent"));
                    }
                    self.exponent()
                } else {
                    1
                };
                degree = degree.saturating_add(e);
                exps[1 + v] = exps[1 + v].saturating_add(e.min(u64::from(u32::MAX)).narrow());
            } else {
                return Err(self.syntax("a number or a variable"));
            }
            if !self.eat(b'*') {
                break;
            }
        }
        // instance-grammar-agreement.5 design -- an instance term's degree is at most 2^31 − 1, the engine's widest format's bound, and the basis and the cofactor text keep `u32`'s, not `u32`'s in the instance as well or 2^31 − 1 in every text (not measured)
        let bound = match self.part {
            Part::Instance => (1 << 31) - 1,
            Part::Basis | Part::Cofactors => u64::from(u32::MAX),
        };
        if degree > bound {
            return Err(self.overflow());
        }
        // Every exponent is at most the degree, so none saturated.
        exps[0] = degree.narrow();
        numbers.push(terms, exps, c, negative, poll);
        poll.work(1 + (self.pos - start) / BYTES);
        Ok(())
    }

    // checker-stop-latency.8 design -- the reading counts 1 + b/16 units a term after it, b its bytes, and ends where the poll stops, the polynomial it reads unfinished and a reading error past the term the stop falls in not reached, not the texts read without a poll (log 0059-1)
    /// One polynomial into `out`. Where `poll` stops, the reading ends
    /// there, and `out` means nothing.
    fn poly<N: Numbers>(
        &mut self,
        ring: &Ring,
        numbers: &N,
        exps: &mut Vec<u32>,
        terms: &mut N::Terms,
        out: &mut N::Poly,
        poll: &mut Poll,
    ) -> Result<(), Error> {
        let mut negative = false;
        match self.peek() {
            Some(b'+') => self.pos += 1,
            Some(b'-') => {
                self.pos += 1;
                negative = true;
            }
            Some(_) | None => {}
        }
        self.term(ring, numbers, negative, exps, terms, poll)?;
        loop {
            if poll.stopped() {
                return Ok(());
            }
            match self.peek() {
                Some(b'+') => negative = false,
                Some(b'-') => negative = true,
                Some(_) | None => break,
            }
            self.pos += 1;
            self.term(ring, numbers, negative, exps, terms, poll)?;
        }
        numbers.finish(terms, ring, out, poll);
        Ok(())
    }

    /// `poly ("," poly)*`, each with the line it starts on. Where `poll`
    /// stops, the reading ends there.
    fn polys<N: Numbers>(
        &mut self,
        ring: &Ring,
        numbers: &N,
        out: &mut Vec<N::Poly>,
        lines: &mut Vec<usize>,
        poll: &mut Poll,
    ) -> Result<(), Error> {
        let mut exps = Vec::new();
        let mut terms = N::Terms::default();
        loop {
            self.skip_space();
            lines.push(self.line);
            let mut p = N::Poly::default();
            self.poly(ring, numbers, &mut exps, &mut terms, &mut p, poll)?;
            if poll.stopped() {
                return Ok(());
            }
            out.push(p);
            if !self.eat(b',') {
                return Ok(());
            }
        }
    }
}

impl Cursor<'_> {
    /// One row, `"[" polys? "]"`, into the front of `entries`: how many
    /// entries it has. Where `poll` stops, the reading ends there, and the
    /// row means nothing.
    fn row<N: Numbers>(
        &mut self,
        ring: &Ring,
        numbers: &N,
        exps: &mut Vec<u32>,
        terms: &mut N::Terms,
        entries: &mut Vec<N::Poly>,
        poll: &mut Poll,
    ) -> Result<usize, Error> {
        self.expect(b'[', "`[`")?;
        let mut k = 0;
        if !self.eat(b']') {
            loop {
                if entries.len() == k {
                    entries.push(N::Poly::default());
                }
                self.poly(ring, numbers, exps, terms, &mut entries[k], poll)?;
                k += 1;
                if poll.stopped() {
                    return Ok(k);
                }
                if !self.eat(b',') {
                    break;
                }
            }
            self.expect(b']', "`,` or `]`")?;
        }
        Ok(k)
    }
}

/// The variables and the characteristic of an instance, and `order`
/// resolved against its variables.
pub fn header<'a>(text: &'a str, order: &Order) -> Result<Header<'a>, Error> {
    let text = text.as_bytes();
    let Some(first) = text.iter().position(|&b| b == b'\n') else {
        return Err(Error::Syntax {
            part: Part::Instance,
            line: 1,
            expected: "a line break after the variables",
        });
    };
    let mut names = HashMap::new();
    let mut cur = Cursor::new(&text[..first], Part::Instance, 1);
    loop {
        if !cur.at_name() {
            return Err(cur.syntax("a variable"));
        }
        let name = cur.name();
        if names.insert(name.to_vec(), names.len()).is_some() {
            return Err(Error::RepeatedVariable {
                line: 1,
                name: String::from_utf8_lossy(name).into_owned(),
            });
        }
        if !cur.eat(b',') {
            break;
        }
    }
    if cur.peek().is_some() {
        return Err(cur.syntax("`,` or the end of the line"));
    }

    let rest = &text[first + 1..];
    let second = rest.iter().position(|&b| b == b'\n').unwrap_or(rest.len());
    let mut cur = Cursor::new(&rest[..second], Part::Instance, 2);
    if !cur.at_digit() {
        return Err(cur.syntax("the characteristic"));
    }
    let mut p = 0u128;
    while let Some(&b) = cur.text.get(cur.pos) {
        if !b.is_ascii_digit() {
            break;
        }
        p = (p * 10 + u128::from(b - b'0')).min(CHARACTERISTIC_CAP);
        cur.pos += 1;
    }
    if cur.peek().is_some() {
        return Err(cur.syntax("the end of the line after the characteristic"));
    }
    // instance-grammar-agreement.3 design -- a characteristic that is neither 0 nor a prime below 2^64 is refused as the instance is read, not left untested with a missing inverse an error where it is needed (checker-gfp.5, withdrawn) (not measured)
    if p != 0 && !(p < 1 << 64 && field::is_prime(p.narrow())) {
        return Err(Error::Characteristic { line: 2 });
    }
    let ring = Ring {
        characteristic: p.narrow(),
        w: names.len() + 1,
        order: Cmp::new(order, names.len())?,
        names,
    };
    let body = rest.get(second + 1..).unwrap_or(&[]);
    Ok(Header { ring, body })
}

/// The generators of an instance, read from the third line on. Where
/// `poll` stops, the reading ends there, and the generators mean nothing.
pub fn generators<N: Numbers>(
    header: &Header,
    numbers: &N,
    poll: &mut Poll,
) -> Result<Vec<N::Poly>, Error> {
    let mut cur = Cursor::new(header.body, Part::Instance, 3);
    let mut generators = Vec::new();
    if cur.peek().is_some() {
        cur.polys(
            &header.ring,
            numbers,
            &mut generators,
            &mut Vec::new(),
            poll,
        )?;
    }
    if poll.stopped() {
        return Ok(generators);
    }
    if cur.peek().is_some() {
        return Err(cur.syntax("`,` or the end of the instance"));
    }
    Ok(generators)
}

/// The basis, each element with the line it starts on. Where `poll`
/// stops, the reading ends there, and the basis means nothing.
pub fn basis<N: Numbers>(
    text: &str,
    ring: &Ring,
    numbers: &N,
    poll: &mut Poll,
) -> Result<(Vec<N::Poly>, Vec<usize>), Error> {
    let mut cur = Cursor::new(text.as_bytes(), Part::Basis, 1);
    cur.comments();
    cur.expect(b'[', "`[`")?;
    let (mut elements, mut lines) = (Vec::new(), Vec::new());
    if !cur.eat(b']') {
        cur.polys(ring, numbers, &mut elements, &mut lines, poll)?;
        if poll.stopped() {
            return Ok((elements, lines));
        }
        cur.expect(b']', "`,` or `]`")?;
    }
    cur.expect(b':', "`:`")?;
    cur.end()?;
    Ok((elements, lines))
}

// checker-gfp.17 design -- the cofactor matrix is parsed and checked one row at a time and never held whole (log 0001-1)
/// The rows of the cofactor matrix, read one at a time.
pub struct Rows<'a, N: Numbers> {
    cur: Cursor<'a>,
    first: bool,
    done: bool,
    start: Start,
    exps: Vec<u32>,
    terms: N::Terms,
}

/// Where a row of the cofactor matrix starts: its `[` and the line of it.
#[derive(Clone, Copy, Debug)]
pub struct Start {
    pos: usize,
    line: usize,
}

impl<'a, N: Numbers> Rows<'a, N> {
    pub fn new(text: &'a str) -> Result<Rows<'a, N>, Error> {
        let mut cur = Cursor::new(text.as_bytes(), Part::Cofactors, 1);
        cur.comments();
        cur.expect(b'[', "`[`")?;
        let done = cur.eat(b']');
        if done {
            cur.expect(b':', "`:`")?;
            cur.end()?;
        }
        Ok(Rows {
            cur,
            first: true,
            done,
            start: Start { pos: 0, line: 1 },
            exps: Vec::new(),
            terms: N::Terms::default(),
        })
    }

    /// Where the row `next` read last starts.
    pub fn start(&self) -> Start {
        self.start
    }

    /// Reads the next row into the front of `entries`, and returns the line it
    /// starts on and how many entries it has; None after the last row.
    /// Where `poll` stops, the reading ends there, and the row means
    /// nothing.
    pub fn next(
        &mut self,
        ring: &Ring,
        numbers: &N,
        entries: &mut Vec<N::Poly>,
        poll: &mut Poll,
    ) -> Result<Option<(usize, usize)>, Error> {
        if self.done {
            return Ok(None);
        }
        if !self.first && !self.cur.eat(b',') {
            self.cur.expect(b']', "`,` or `]`")?;
            self.cur.expect(b':', "`:`")?;
            self.cur.end()?;
            self.done = true;
            return Ok(None);
        }
        self.first = false;
        self.cur.skip_space();
        let line = self.cur.line;
        self.start = Start {
            pos: self.cur.pos,
            line,
        };
        let k = self.cur.row(
            ring,
            numbers,
            &mut self.exps,
            &mut self.terms,
            entries,
            poll,
        )?;
        Ok(Some((line, k)))
    }
}

/// Reads the row of the cofactor matrix `text` that starts at `start` again,
/// with `numbers`, into the front of `entries`: how many entries it has.
/// Where `poll` stops, the reading ends there, and the row means nothing.
pub fn row<N: Numbers>(
    text: &str,
    start: Start,
    ring: &Ring,
    numbers: &N,
    entries: &mut Vec<N::Poly>,
    poll: &mut Poll,
) -> Result<usize, Error> {
    let mut cur = Cursor::new(text.as_bytes(), Part::Cofactors, start.line);
    cur.pos = start.pos;
    cur.row(
        ring,
        numbers,
        &mut Vec::new(),
        &mut N::Terms::default(),
        entries,
        poll,
    )
}
