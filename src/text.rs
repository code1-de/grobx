//! The instance reader and the canonical basis text.
//!
//! An instance is in msolve's input format: a line of variable names, a
//! line with the characteristic, then the generators, separated by commas.

#[cfg(test)]
use std::cell::Cell;
use std::cell::OnceCell;
use std::io;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::OnceLock;

use num_bigint::BigUint;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, INVERSE, PROBE, Site};
use crate::field::{Field, Gf2, Gf31, Gf64, U64_DIGITS, decimal_value};
use crate::id::MonoId;
use crate::limbs;
use crate::module::{Module, Rank1};
use crate::monomial::{Dense8, Dense16, Dense32, Format};
use crate::narrow::{self, Narrow};
use crate::order::{Grevlex, Order};
use crate::os;
use crate::poly::Poly;
use crate::sort;
use crate::table::{LOAD, Limit, Table, monomial_units, multipliers, sum_hash, touched_units};
use crate::{Error, Stop};

// q-lift.35 design-checked -- the test code's reader refuses characteristic 0 as before, not the generators over Q written monic (not measured)
/// The instance's nonzero generators, each monic with its terms in
/// decreasing grevlex order, in the instance's order, as a bracketed list
/// in the canonical basis text's spelling. For the test code of the reader.
/// The instance is read in `Dense8`, `Dense16`, and `Dense32` in turn, as a
/// call over a field that starts in `Dense8` reads it, and the first that
/// holds it writes the text, which every format writes alike.
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
    // No call, so no bounds.
    let mut budget = Budget::unbounded();
    let header = Header::read(text, &mut budget).map_err(Unread::unbounded)?;
    match header.field()? {
        Prime::Gf2(field) => generators_over::<M, _>(&header, field, &mut budget),
        Prime::Gf31(field) => generators_over::<M, _>(&header, field, &mut budget),
        Prime::Gf64(field) => generators_over::<M, _>(&header, field, &mut budget),
    }
}

/// As `generators_in`, over the field `field` of the header, under
/// `budget`, which has no bounds.
fn generators_over<M: Format, F: Field>(
    header: &Header,
    field: F,
    budget: &mut Budget,
) -> Result<String, Error> {
    let format = M::new(header.names.len());
    // No call, so no seed: the multipliers decide no result.
    let mut table = Table::new(format, &mut Xoshiro256PlusPlus::seed_from_u64(NO_SEED));
    let mut polys: Vec<Poly<F, Rank1>> = header
        .generators(&field, &mut table, &Grevlex, budget)
        .map_err(Unread::unbounded)?;
    polys.retain(|p| !p.is_zero());
    for p in &mut polys {
        p.make_monic(&field);
    }
    let mut out = String::new();
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
        budget,
    )
    .expect("a budget without bounds never stops");
    Ok(out)
}

/// The seed of hash multipliers that decide no result: a table's outside a
/// call, and the parse's for the text's distinct monomials, which it numbers
/// in the order the text first writes them.
const NO_SEED: u64 = 0;

// gfp-large-primes.2 design-checked -- `Gf31` stays for the primes below 2^31, and the entry point picks `Gf31` or `Gf64` from the characteristic, not `Gf64` for every prime (log 0015-1, 0015-2)
/// The field a characteristic names (gfp-large-primes.2).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Prime {
    /// The prime 2 (gf2-reduction-time.1).
    Gf2(Gf2),
    /// A prime below 2^31 but 2.
    Gf31(Gf31),
    /// A prime from 2^31 to 2^64.
    Gf64(Gf64),
}

/// An instance's first two lines, and where its generators start.
pub(crate) struct Header<'a> {
    pub(crate) names: Vec<&'a str>,
    /// The names indexed for the parse's lookups.
    index: HashIndex,
    /// The characteristic's digits as written.
    pub(crate) characteristic: &'a str,
    /// The text after the characteristic's line.
    rest: Cursor<'a>,
    /// The generators parsed, once `parsed` has been called.
    parsed: OnceCell<Parsed<'a>>,
}

impl<'a> Header<'a> {
    // call-stop-latency.24 design-checked -- the reading of the instance takes the call's budget and spends as it goes, the header's bytes and names compared, the parse's bytes and monomials, and the map's terms, factors, digits, and monomials, and a stop inside it returns the call stopped, not the reading unspent before the call's first check (log 0138-6)
    /// The header of `text`: its names, indexed, and its characteristic,
    /// the bytes of its lines spent on `budget` a unit each and each name's
    /// entry into the index `PROBE` units, at `Site::Names`, and a growth
    /// of the index as `HashIndex::grow` spends. The text's error, or the
    /// stop where the budget's check stops the call.
    pub(crate) fn read(text: &'a str, budget: &mut Budget) -> Result<Header<'a>, Unread> {
        let mut c = Cursor::new(text);
        let mut names: Vec<&str> = Vec::new();
        let mut index = HashIndex::new();
        // The bytes spent so far.
        let mut spent = 0;
        loop {
            c.skip_blanks();
            let name = c.name().ok_or_else(|| c.syntax("a variable name"))?;
            budget.spend(c.pos - spent + PROBE, Site::Names)?;
            spent = c.pos;
            // gf2-input-time.2 design-checked -- the header's test for a repeated name looks the name up in the names' index as the names enter it, not a comparison with every earlier name (log 0158-3)
            let hash = name_hash(name);
            match index.find(hash, |i| names[i] == name) {
                Ok(_) => {
                    return Err(Error::RepeatedVariable {
                        name: name.to_string(),
                    }
                    .into());
                }
                Err(slot) => {
                    names.push(name);
                    narrow::id(names.len()).map_err(|_| Error::Size)?;
                    index.insert(slot, hash, names.len(), budget, Site::Names, |i| {
                        name_hash(names[i])
                    })?;
                }
            }
            c.skip_blanks();
            match c.peek() {
                Some(b',') => c.bump(),
                Some(b'\n') => break,
                Some(_) | None => return Err(c.syntax("a comma or a line feed").into()),
            }
        }
        c.bump(); // the first line feed
        c.skip_blanks();
        let characteristic = c.integer().ok_or_else(|| c.syntax("the characteristic"))?;
        c.skip_blanks();
        match c.peek() {
            Some(b'\n') => c.bump(),
            None => {}
            Some(_) => return Err(c.syntax("a line feed").into()),
        }
        budget.spend(c.pos - spent, Site::Names)?;
        Ok(Header {
            names,
            index,
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
    /// unless it is a prime below 2^64. The prime 2 takes `Gf2`.
    pub(crate) fn field(&self) -> Result<Prime, Error> {
        let digits = self.characteristic.trim_start_matches('0');
        let p = digits
            .parse::<u64>()
            .ok()
            .or(digits.is_empty().then_some(0));
        let field = p.and_then(|p| {
            if p == Gf2::P {
                return Some(Prime::Gf2(Gf2::new()));
            }
            Gf31::new(p)
                .map(Prime::Gf31)
                .or_else(|| Gf64::new(p).map(Prime::Gf64))
        });
        field.ok_or_else(|| Error::Characteristic {
            value: self.characteristic.to_string(),
        })
    }

    /// As `field`, `Error::Characteristic` unless the prime lies below
    /// 2^31, and `Gf31` at 2 for the prime 2. Some of the crate's unit
    /// tests read their field through it.
    #[cfg(test)]
    pub(crate) fn gf31(&self) -> Result<Gf31, Error> {
        match self.field()? {
            Prime::Gf31(field) => Ok(field),
            Prime::Gf2(_) => Gf31::new(Gf2::P).ok_or_else(|| Error::Characteristic {
                value: self.characteristic.to_string(),
            }),
            Prime::Gf64(_) => Err(Error::Characteristic {
                value: self.characteristic.to_string(),
            }),
        }
    }

    /// The line of the denominator 0 the parse stopped at, the `NoInverse`
    /// of `Cursor::factor`, at the digits' line: None where the parse ran
    /// through or stopped at another error, which comes first in the text.
    /// The lift tells a read's `NoInverse` for a denominator 0 from one for
    /// a prime that divides a denominator by it (q-denominator-line.2). A
    /// read's `NoInverse` comes from the parsed form, so it follows the
    /// parse.
    // call-stop-latency.64 design-checked -- the parse's denominator 0 is read from the parse a read made, which every NoInverse follows, not a parse of its own under a budget the lift's test of the read's error would hold (not measured)
    pub(crate) fn zero_denominator(&self) -> Option<usize> {
        let parsed = self.parsed.get();
        debug_assert!(parsed.is_some(), "a NoInverse before the parse");
        let (error, _) = parsed?.error.as_ref()?;
        match error {
            Error::NoInverse { line } => Some(*line),
            Error::Syntax { .. }
            | Error::UnknownVariable { .. }
            | Error::RepeatedVariable { .. }
            | Error::Characteristic { .. }
            | Error::Overflow { .. }
            | Error::Size
            | Error::Order { .. }
            | Error::Memory
            | Error::Clock
            | Error::Misfit { .. }
            | Error::Drops { .. }
            | Error::NotBuilt { .. } => None,
        }
    }

    /// Every generator, zero ones included, in the instance's order, over
    /// `field` in `table`, each with its terms in `order`: the parsed form
    /// mapped to them (`Parsed::map`), the parse and the map spent on
    /// `budget`. The first error of the text over this field and format;
    /// `Error::Size` where the generators, their terms, their numeric
    /// factors, or their distinct monomials are more than a u32 id numbers,
    /// or a term starts at a line from `MINUS` on; or the stop where the
    /// budget's check stops the call.
    pub(crate) fn generators<F: Field, M: Format, O: Order<M>, D: Module>(
        &self,
        field: &F,
        table: &mut Table<M>,
        order: &O,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, D>>, Unread> {
        self.parsed(budget)?.map(field, table, order, budget)
    }

    // run-generators-time.1 design-checked -- the header parses the generators once per call into a form free of the field and the run's table, and each run maps that form to its field and table, not each run reading the instance text again (log 0087-1)
    // run-generators-time.4 practice-checked: lazy initialization -- the parsed form is made at the first run's read, the first call of `Header::parsed`, and kept in the header, in a OnceCell, not made in `Header::read` nor by the entry point and passed beside the header (not measured)
    /// The generators as parsed: parsed at the first call, spent on
    /// `budget`, and kept for every later one. `Parsed::map` takes them to a
    /// run's field and table. The stop where the budget's check stops the
    /// call, with nothing kept.
    pub(crate) fn parsed(&self, budget: &mut Budget) -> Result<&Parsed<'a>, Stop> {
        if let Some(parsed) = self.parsed.get() {
            return Ok(parsed);
        }
        let parsed = Parser::parse(self, budget)?;
        Ok(self.parsed.get_or_init(|| parsed))
    }
}

// call-stop-latency.63 design-checked -- a reading returns its stop beside the text's errors in a type of its own, Unread, which a call turns into its stopped outcome, not a variant of the public Error, which names no stop, nor a Result of a Result (not measured)
/// Why a reading of the instance returns no result: the text's error, or
/// the stop of the call's budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Unread {
    Error(Error),
    Stop(Stop),
}

impl From<Error> for Unread {
    fn from(error: Error) -> Unread {
        Unread::Error(error)
    }
}

impl From<Stop> for Unread {
    fn from(stop: Stop) -> Unread {
        Unread::Stop(stop)
    }
}

impl Unread {
    /// The error of a reading under a budget without bounds, which never
    /// stops.
    #[allow(
        clippy::unreachable,
        reason = "a budget without bounds never stops, and its callers read under one alone"
    )]
    pub(crate) fn unbounded(self) -> Error {
        match self {
            Unread::Error(error) => error,
            Unread::Stop(stop) => unreachable!("a budget without bounds stopped: {stop:?}"),
        }
    }
}

// gf2-input-time.3 practice-checked: hash table with linear probing -- a variable's name is found in a table of the names' indices under a hash of its bytes, the hash's top 32 bits beside each index and compared before the name, not by a binary search over the sorted names, whose string comparisons held much of the parse once its terms interned over their variables alone (log 0158-1, 0158-3)
// instance-read-peak.2 design-checked -- the parse's distinct monomials are found by the index the names are found by, entries numbered from 0 under a hash the caller takes and compared by the caller's test, not an index of their own (log 0175-1, 0175-2)
/// Entries numbered from 0, found by a 64-bit hash of each and the caller's
/// test of an entry: per slot an entry's number plus one below the hash's
/// top 32 bits, 0 where none, probed linearly from the hash's low bits, the
/// slots a power of two in number and at least `LOAD` times the entries.
/// The variables' names and the parse's distinct monomials are indexed so.
#[derive(Clone)]
struct HashIndex {
    slots: Vec<u64>,
}

/// The slots of an empty hash index.
const INDEX_SLOTS: usize = 1 << 4;

impl HashIndex {
    fn new() -> HashIndex {
        HashIndex {
            slots: vec![0; INDEX_SLOTS],
        }
    }

    // gf2-input-time.5 design-checked -- a name's first slot is its hash's low bits, not its top bits, which FNV-1a mixes little on names of a few bytes, so that the names of an instance of many variables clustered (log 0158-1, 0158-3)
    /// The first slot of an entry of hash `hash`, and its tag: the hash's
    /// low bits, and its top 32 bits.
    fn home(&self, hash: u64) -> (usize, u64) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the mask keeps the slot below the index's size"
        )]
        let slot = hash as usize & (self.slots.len() - 1);
        (slot, hash >> u32::BITS << u32::BITS)
    }

    /// The number of the entry of hash `hash` for which `same` holds; or
    /// the empty slot where it would stand. The tags are compared before
    /// `same` is asked.
    #[inline(always)]
    fn find(&self, hash: u64, same: impl Fn(usize) -> bool) -> Result<usize, usize> {
        let last = self.slots.len() - 1;
        let (mut slot, tag) = self.home(hash);
        loop {
            let s = self.slots[slot];
            if s == 0 {
                return Err(slot);
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the low 32 bits hold an entry's number plus one"
            )]
            let i = (s as u32) as usize - 1;
            if s >> u32::BITS << u32::BITS == tag && same(i) {
                return Ok(i);
            }
            slot = (slot + 1) & last;
        }
    }

    /// Enters entry `entries − 1`, of hash `hash`, at the empty slot `slot`
    /// that `find` returned for it, the entries before it entered, and grows
    /// the index where that takes it past half full; `hash_of` gives each
    /// entry's hash. The entries number at most `u32::MAX`, which the
    /// caller checks. The stop where the budget's check stops the call in a
    /// growth, spent at `site`.
    fn insert(
        &mut self,
        slot: usize,
        hash: u64,
        entries: usize,
        budget: &mut Budget,
        site: Site,
        hash_of: impl Fn(usize) -> u64,
    ) -> Result<(), Stop> {
        debug_assert!(slot < self.slots.len(), "slot {slot}");
        debug_assert!(
            entries > 0 && u32::try_from(entries).is_ok(),
            "{entries} entries"
        );
        let (_, tag) = self.home(hash);
        self.slots[slot] = tag | entries as u64;
        if LOAD * entries > self.slots.len() {
            self.grow(entries, budget, site, hash_of)?;
        }
        Ok(())
    }

    /// Doubles the slots and enters each of the `entries` again by its hash,
    /// `hash_of`: `PROBE` units an entry, at `site`.
    fn grow(
        &mut self,
        entries: usize,
        budget: &mut Budget,
        site: Site,
        hash_of: impl Fn(usize) -> u64,
    ) -> Result<(), Stop> {
        let doubled = 2 * self.slots.len();
        self.slots.clear();
        self.slots.resize(doubled, 0);
        let last = doubled - 1;
        for i in 0..entries {
            let (mut slot, tag) = self.home(hash_of(i));
            while self.slots[slot] != 0 {
                slot = (slot + 1) & last;
            }
            self.slots[slot] = tag | (i as u64 + 1);
        }
        budget.spend(entries * PROBE, site)
    }
}

// gf2-input-time.4 practice-checked: FNV-1a -- a name's hash is the 64-bit FNV-1a hash of its bytes, not a hash of the engine's own
/// The 64-bit FNV-1a hash of `name`'s bytes.
fn name_hash(name: &str) -> u64 {
    name.bytes().fold(FNV_OFFSET, fnv_step)
}

/// FNV-1a's offset basis, the hash of no byte.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a's hash `h` of some bytes extended by the byte `b`.
#[inline(always)]
fn fnv_step(h: u64, b: u8) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    (h ^ u64::from(b)).wrapping_mul(PRIME)
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
        self.run(name_first, name_byte)
    }

    // instance-read-time.4 design-checked -- a name's hash is taken as its bytes are scanned, and the names' index finds the name by that hash, not the name scanned and then hashed in a second pass over its bytes (log 0160-1, 0165-1, 0165-2)
    /// A name and its hash (`name_hash`), taken in one pass over its bytes;
    /// None where no name stands here.
    fn hashed_name(&mut self) -> Option<(&'a str, u64)> {
        let bytes = self.text.as_bytes();
        let start = self.pos;
        if !bytes.get(start).copied().is_some_and(name_first) {
            return None;
        }
        let mut hash = FNV_OFFSET;
        let mut end = start;
        while let Some(&b) = bytes.get(end).filter(|&&b| name_byte(b)) {
            hash = fnv_step(hash, b);
            end += 1;
        }
        self.pos = end;
        self.last = self.line;
        Some((&self.text[start..end], hash))
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

    /// A numeric factor, `n` or `n/d` (`denominator`); None where no digit
    /// stands here.
    fn factor(&mut self) -> Result<Option<Factor<'a>>, Error> {
        let Some(digits) = self.integer() else {
            return Ok(None);
        };
        Ok(Some(Factor {
            numerator: Literal::new(digits),
            denominator: self.denominator()?,
        }))
    }

    /// A denominator, `/d` with whitespace around the `/`: d and the line of
    /// its digits; None where no `/` stands here. `Error::NoInverse` at the
    /// digits' line where they are all `0`.
    fn denominator(&mut self) -> Result<Option<(Literal<'a>, usize)>, Error> {
        self.skip_space();
        if self.peek() != Some(b'/') {
            return Ok(None);
        }
        self.bump();
        self.skip_space();
        let digits = self.integer().ok_or_else(|| self.syntax("a denominator"))?;
        // q-denominator-line.1 design-checked -- a denominator whose digits are all 0 is the parse's error, NoInverse at the digits' line, which every read returns at its place in the text's order whatever its prime, not a scan of the text for the first /0 after a read's NoInverse, which names a line and not the error a read reaches first on it
        if digits.bytes().all(|b| b == b'0') {
            return Err(Error::NoInverse { line: self.last });
        }
        Ok(Some((Literal::new(digits), self.last)))
    }

    /// A power of a variable, `x` or `x^e` with whitespace around the `^`:
    /// the variable's index and e, 1 where no `^` stands; None where no name
    /// stands here. `index` indexes `names`.
    /// `Error::UnknownVariable` at the name's line where it is none of them.
    fn power(&mut self, names: &[&str], index: &HashIndex) -> Result<Option<(usize, u64)>, Error> {
        let Some((name, hash)) = self.hashed_name() else {
            return Ok(None);
        };
        let Ok(i) = index.find(hash, |i| names[i] == name) else {
            return Err(Error::UnknownVariable {
                line: self.last,
                name: name.to_string(),
            });
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

/// Whether a name may begin with the byte b: a letter or `_`.
fn name_first(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Whether a name may hold the byte b after its first: a letter, a digit,
/// or `_`.
fn name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
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
/// table, or the stop that a growth of the table met.
fn term_error(limit: Limit, line: usize) -> Unread {
    match limit {
        Limit::Overflow => Unread::Error(Error::Overflow { line: Some(line) }),
        Limit::Size => Unread::Error(Error::Size),
        Limit::Stop(stop) => Unread::Stop(stop),
    }
}

// run-generators-time.2 design-checked -- a literal of at most nineteen digits is kept as its value, whose residue is `Field::coef` of it, and a longer one as its digits, whose residue `Field::decimal` takes, not every literal's digits kept and read at each run nor every literal as a big integer's limbs (log 0087-1)
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

    /// The units of work of its residue at one lane: `PROBE`, a remainder,
    /// for a value, and for digits a unit each and `PROBE` per
    /// `U64_DIGITS` of them (`field::decimal_residue`).
    fn units(self) -> usize {
        match self {
            Literal::Small(_) => PROBE,
            Literal::Digits(digits) => digits.len() + digits.len().div_ceil(U64_DIGITS) * PROBE,
        }
    }
}

/// A numeric factor of a term, `n` or `n/d`, or `1/d` for a denominator
/// after another factor: its numerator, and its denominator with the line
/// of the denominator's digits.
#[derive(Clone, Copy, Debug)]
struct Factor<'a> {
    numerator: Literal<'a>,
    denominator: Option<(Literal<'a>, usize)>,
}

impl Factor<'_> {
    /// Its value in the field: the numerator times the inverse of the
    /// denominator, `Error::NoInverse` at the denominator's line where the
    /// field has no inverse of it, a prime of the field that divides it.
    /// The parse refuses a denominator that is 0 (`Cursor::factor`).
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

    // call-stop-latency.61 design-checked -- a numeric factor's value spends PROBE units a literal, a remainder, its digits past a word's a unit each and PROBE per word, PROBE a product, and INVERSE a denominator's inversion, at each lane, not a unit per term and digit (log 0138-6)
    /// The units of work of its value at one lane, beside the product that
    /// takes it into its term's coefficient: its literals' residues, and
    /// for a denominator its inversion and the product.
    fn units(&self) -> usize {
        let numerator = self.numerator.units();
        match self.denominator {
            None => numerator,
            Some((denominator, _)) => numerator + denominator.units() + INVERSE + PROBE,
        }
    }
}

#[cfg(test)]
thread_local! {
    // instance-read-peak.9 design-checked -- the test profile can give every distinct monomial a parse meets the hash 0, so that the test code reaches the index's comparison of the variables each monomial names, which random multipliers leave untaken, not that comparison tested on an index of its own (not measured)
    /// Where true, every distinct monomial a parse on this thread meets
    /// hashes to 0 (`monomial_hash`).
    pub(crate) static ONE_HASH: Cell<bool> = const { Cell::new(false) };
}

/// The hash of a distinct monomial of the parse from its variables and
/// their exponents, each variable once: `table::sum_hash` under the
/// parse's multipliers, and 0 in the test profile where `ONE_HASH` holds.
fn monomial_hash(multipliers: &[u64], exponents: impl Iterator<Item = (usize, u32)>) -> u64 {
    #[cfg(test)]
    if ONE_HASH.get() {
        return 0;
    }
    sum_hash(multipliers, exponents)
}

// instance-read-peak.5 design-checked -- a parsed term takes 12 bytes, the end of its factors' range, its monomial's number, and its line with its sign in the line's top bit, the range's start the end of the term before, not 24 bytes of a range, a line in a usize, and a sign beside them, as the call keeps every term for its runs (log 0175-1, 0175-2)
/// A term as parsed: the end of its factors' range in `Parsed::factors`,
/// which starts where the term before's ends; its monomial's number among
/// the text's distinct monomials; and the line it starts at, with its sign
/// in the line's top bit, `MINUS`, set for a minus. The number is
/// `MonoId::NONE` where the monomial's degree passes `Dense32`, the widest
/// format, and so every format.
#[derive(Clone, Copy)]
pub(crate) struct Term {
    pub(crate) end: u32,
    pub(crate) monomial: MonoId,
    line: u32,
}

/// The bit of a term's line that marks a minus (`Term`).
pub(crate) const MINUS: u32 = 1 << (u32::BITS - 1);

impl Term {
    /// The term whose factors end at `end`, of monomial `monomial`, starting
    /// at `line`, a minus where `negative`: `Error::Size` where the line
    /// reaches `MINUS`.
    pub(crate) fn new(
        end: u32,
        monomial: MonoId,
        line: usize,
        negative: bool,
    ) -> Result<Term, Error> {
        let line = u32::try_from(line)
            .ok()
            .filter(|&l| l < MINUS)
            .ok_or(Error::Size)?;
        let sign = if negative { MINUS } else { 0 };
        Ok(Term {
            end,
            monomial,
            line: line | sign,
        })
    }

    /// Whether the term's sign is a minus.
    pub(crate) fn negative(self) -> bool {
        self.line & MINUS != 0
    }

    /// The line the term starts at.
    pub(crate) fn line(self) -> usize {
        (self.line & !MINUS) as usize
    }
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
    // run-generators-time.9 design-checked -- the parse numbers the text's distinct monomials in the order the text first writes them, and each run interns each of them once, at its first term, not each term's exponents kept and interned at every run (log 0087-2); amended by instance-read-peak.1: the numbers by `index`, in place of a table of their own in `Dense32`
    /// The text's variables.
    variables: usize,
    // instance-read-peak.3 design-checked -- a distinct monomial's hash is Σ r_i e_i over the variables its term names, the table's own hash, its multipliers drawn as a table draws them, not a hash of the variables' and exponents' bytes nor a mix function of a variable's index, so that every spelling of a monomial hashes alike (not measured)
    /// The distinct monomials' hash multipliers, a variable's each, drawn as
    /// a table's from the seed `NO_SEED`.
    multipliers: Vec<u64>,
    // instance-read-peak.1 practice-checked: sparse monomial representation -- the parse finds a term's monomial among the text's distinct monomials by its hash and the variables and exponents each one's first term named, `named`, with no exponent vector kept for any, not each distinct monomial's exponents over every variable kept in a table in `Dense32`, four bytes a variable, which the call keeps for its runs (log 0175-1, 0175-2)
    /// The distinct monomials by number, under the hashes of their `named`
    /// exponents.
    index: HashIndex,
    // instance-read-time.3 practice-checked: sparse monomial representation -- each distinct monomial keeps the variables its first term names and their exponents, from which a run interns it over those variables alone (`Table::intern_touched`, gf2-input-time.1), not its exponents unpacked from `Dense32` and interned over every variable of the format (log 0160-1, 0165-1, 0165-2)
    /// Per distinct monomial, by its number, the variables its first term
    /// names with their exponents: `named[named_at[k]..named_at[k + 1]]`.
    named: Vec<(u32, u32)>,
    /// The bounds of each distinct monomial's variables in `named`, one
    /// more than the monomials, from 0.
    named_at: Vec<u32>,
    error: Option<(Error, Range<u32>)>,
}

impl Parsed<'_> {
    /// The generators over `field` in `table`, zero ones included, in the
    /// text's order, each with its terms in `order`, like terms added and
    /// zero terms dropped; or the first error of the text. The terms are
    /// taken in the text's order, each its factors' values in turn, then
    /// its monomial, so the error is the first in the text over this field
    /// and format. A monomial is interned at its first term, over the
    /// variables that term names, so `table` numbers the monomials in the
    /// order the text first writes them. Once every term is taken, the
    /// text's distinct monomials are sorted into decreasing `order` once,
    /// and the terms are ordered by their monomials' ranks in that sort
    /// (instance-read-time.1), then by their generators, by two counting
    /// sorts (instance-read-peak.7). The map spends on `budget` at
    /// `Site::Map`: the ids' array a unit an id, each term `PROBE` units and
    /// its coefficient's (`coefficient_units`), each monomial interned what
    /// `touched_units` says, the ranks what `ranks` spends, the counting
    /// sorts what `sort::counting` spends, and each generator's sum what
    /// `Poly::from_sorted` spends. The stop where the budget's check stops
    /// the call.
    pub(crate) fn map<F: Field, M: Format, O: Order<M>, D: Module>(
        &self,
        field: &F,
        table: &mut Table<M>,
        order: &O,
        budget: &mut Budget,
    ) -> Result<Vec<Poly<F, D>>, Unread> {
        let n = self.variables;
        debug_assert_eq!(
            table.format().variables(),
            n,
            "a table of other variables than the text's"
        );
        let mut ids = vec![MonoId::NONE; self.named_at.len() - 1];
        budget.spend(ids.len(), Site::Map)?;
        let mut exponents = vec![0; n];
        let mut touched = Vec::new();
        // Every term's generator, its monomial's number among the distinct
        // monomials, and its coefficient, generator after generator.
        let mut terms: Vec<(u32, u32, F::Coef)> = Vec::with_capacity(self.terms.len());
        let mut start = 0;
        for (g, range) in (0u32..).zip(&self.generators) {
            for &term in &self.terms[range.start as usize..range.end as usize] {
                let factors = start..term.end;
                start = term.end;
                let coef = self.coefficient(&factors, field)?;
                self.monomial(term, table, &mut ids, &mut exponents, &mut touched, budget)?;
                let coef = if term.negative() {
                    field.neg(coef)
                } else {
                    coef
                };
                terms.push((g, term.monomial.raw(), coef));
                budget.spend(PROBE + self.coefficient_units::<F>(&factors), Site::Map)?;
            }
        }
        // run-generators-time.3 design-checked -- a text that breaks the grammar keeps the terms before the break and the broken term's factors, which the map takes before it returns the parse's error, so that a denominator 0 mod p before the break stays the error, not the parse's error returned at once nor the reader kept for such texts (not measured)
        if let Some((error, broken)) = &self.error {
            self.coefficient(broken, field)?;
            return Err(error.clone().into());
        }
        let rank = ranks(&ids, table, order, budget)?;
        // instance-read-peak.7 practice-checked: LSD radix sort -- the terms' numbers are ordered by a stable counting sort by their monomials' ranks, then by a stable counting sort by their generators, which leaves each generator's terms by rank, not each generator's terms sorted by a comparison of their ranks (log 0175-1, 0175-2)
        // instance-read-peak.8 practice-checked: indirect sort -- the counting sorts order the terms' numbers, four bytes a term, and each generator takes its terms through them, not the terms themselves moved, whose coefficient over ℚ takes 16 bytes in `Gf64Pair` (log 0175-1, 0175-2)
        let mut counts = Vec::new();
        let mut numbers: Vec<u32> = (0..terms.len()).map(Narrow::narrow).collect();
        let mut by_rank = Vec::new();
        sort::counting(
            &numbers,
            |&t| rank[terms[t as usize].1 as usize] as usize,
            ids.len(),
            &mut counts,
            &mut by_rank,
            budget,
            Site::Map,
        )?;
        sort::counting(
            &by_rank,
            |&t| terms[t as usize].0 as usize,
            self.generators.len(),
            &mut counts,
            &mut numbers,
            budget,
            Site::Map,
        )?;
        drop(by_rank);
        let mut polys = Vec::with_capacity(self.generators.len());
        let mut sorted = Vec::new();
        // The generators' ranges follow each other from 0 over every term,
        // so the sort by generator leaves a generator's terms at its range.
        for g in &self.generators {
            sorted.clear();
            sorted.extend(numbers[g.start as usize..g.end as usize].iter().map(|&t| {
                let (_, m, coef) = terms[t as usize];
                (ids[m as usize], coef)
            }));
            polys.push(Poly::from_sorted(
                &mut sorted,
                field,
                D::Position::default(),
                budget,
                Site::Map,
            )?);
        }
        Ok(polys)
    }

    /// The units of work of the coefficient of these factors over the
    /// field `F`: per factor its value's and a product's, `PROBE`, at each
    /// lane (`Factor::units`).
    fn coefficient_units<F: Field>(&self, factors: &Range<u32>) -> usize {
        let factors = &self.factors[factors.start as usize..factors.end as usize];
        let lane: usize = factors.iter().map(|f| f.units() + PROBE).sum();
        F::LANES * lane
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
    /// first term over the variables that term names (`named`,
    /// `Table::intern_touched`) and spent there as `touched_units` says, at
    /// `Site::Map`: `ids` holds per id of `monomials` its id in `table`,
    /// `MonoId::NONE` before, `exponents` is scratch of one exponent per
    /// variable, 0 before and after, and `touched` scratch of the variables
    /// named. The stop where the budget's check stops the call.
    fn monomial<M: Format>(
        &self,
        term: Term,
        table: &mut Table<M>,
        ids: &mut [MonoId],
        exponents: &mut [u32],
        touched: &mut Vec<usize>,
        budget: &mut Budget,
    ) -> Result<MonoId, Unread> {
        if term.monomial == MonoId::NONE {
            return Err(Error::Overflow {
                line: Some(term.line()),
            }
            .into());
        }
        let k = term.monomial.index();
        let known = ids[k];
        if known != MonoId::NONE {
            return Ok(known);
        }
        let named = &self.named[self.named_at[k] as usize..self.named_at[k + 1] as usize];
        touched.clear();
        for &(i, e) in named {
            exponents[i as usize] = e;
            touched.push(i as usize);
        }
        let before = table.len();
        let interned = table.intern_touched(exponents, touched, budget);
        for &i in touched.iter() {
            exponents[i] = 0;
        }
        let id = interned.map_err(|limit| term_error(limit, term.line()))?;
        let units = touched_units(touched.len(), table.format().words(), table.len() > before);
        budget.spend(units, Site::Map)?;
        ids[k] = id;
        Ok(id)
    }
}

// instance-read-time.1 cited-checked: bachmann-1998 §4, attributes it to Macaulay [4] -- a monomial is compared by its rank, one integer whose order is the monomial order
// instance-read-time.2 design-checked -- a monomial's rank is found by a binary search of its id among the text's ids sorted by id, not an array of ranks indexed by the run table's ids, which a replay over ℚ, reading into a table the run before it built, would size past the text's monomials, nor a hash map from id to rank (log 0160-1, 0165-1, 0165-2)
/// Per number k of the text's distinct monomials, the rank of its monomial
/// `ids[k]` of `table` in decreasing `order` among them, 0 the largest:
/// `ids` sorted once (`sort::sort_decreasing`), and each id found by a
/// search among them sorted by id (`sort::sort_runs`), all spent at
/// `Site::Map`, a search `PROBE` units. The stop where the budget's check
/// stops the call.
fn ranks<M: Format, O: Order<M>>(
    ids: &[MonoId],
    table: &Table<M>,
    order: &O,
    budget: &mut Budget,
) -> Result<Vec<u32>, Stop> {
    debug_assert!(
        ids.iter().all(|&m| m != MonoId::NONE),
        "every monomial of the text interned"
    );
    let mut sorted = ids.to_vec();
    sort::sort_decreasing(table, order, &mut sorted, budget, Site::Map)?;
    // The text's numbers by id: the ids are distinct, as the numbers are.
    let mut by_id: Vec<(MonoId, u32)> = ids.iter().zip(0..).map(|(&m, k)| (m, k)).collect();
    sort::sort_runs(&mut by_id, |a, b| a.0.cmp(&b.0), budget, Site::Map)?;
    let mut rank = vec![0; ids.len()];
    for (chunk, start) in sorted.chunks(sort::RUN).zip((0u32..).step_by(sort::RUN)) {
        for (&m, r) in chunk.iter().zip(start..) {
            #[allow(
                clippy::expect_used,
                reason = "`sorted` holds the ids of `by_id`, each once"
            )]
            let at = by_id
                .binary_search_by(|e| e.0.cmp(&m))
                .expect("a sorted id among the text's ids");
            rank[by_id[at].1 as usize] = r;
        }
        budget.spend(chunk.len() * PROBE, Site::Map)?;
    }
    Ok(rank)
}

// f4-gfp.43 design-checked -- the reader takes the tolerant grammar of checker-gfp.3, whitespace between any tokens, terms in any order, like terms combined, not the corpus's layout alone (not measured)
/// Parses the generators into a `Parsed`: per term its sign, its numeric
/// factors, and its monomial's number, free of a run's field and table.
struct Parser<'h, 'a> {
    c: Cursor<'a>,
    names: &'h [&'a str],
    index: &'h HashIndex,
    /// The current term's exponents, 0 but at `touched`.
    exponents: Vec<u32>,
    /// The variables of the current term's nonzero exponents, each once.
    touched: Vec<usize>,
    parsed: Parsed<'a>,
    /// The cursor's position at the last spend of the bytes it passed.
    spent: usize,
}

impl<'h, 'a> Parser<'h, 'a> {
    /// The generators of the text after the header's characteristic line,
    /// spent on `budget` at `Site::Parse`: the bytes the cursor passes a
    /// unit each, after each term and at the end, and each term's monomial
    /// as `monomial_units` says of the variables it names. The stop where
    /// the budget's check stops the call.
    fn parse(header: &'h Header<'a>, budget: &mut Budget) -> Result<Parsed<'a>, Stop> {
        let n = header.names.len();
        let mut parser = Parser {
            c: header.rest.clone(),
            names: &header.names,
            index: &header.index,
            exponents: vec![0; n],
            touched: Vec::new(),
            parsed: Parsed {
                generators: Vec::new(),
                terms: Vec::new(),
                factors: Vec::new(),
                variables: n,
                multipliers: multipliers(n, &mut Xoshiro256PlusPlus::seed_from_u64(NO_SEED)),
                index: HashIndex::new(),
                named: Vec::new(),
                named_at: vec![0],
                error: None,
            },
            spent: header.rest.pos,
        };
        match parser.generators(budget) {
            Ok(()) => {}
            Err(Unread::Error(error)) => parser.broken(error),
            Err(Unread::Stop(stop)) => return Err(stop),
        }
        budget.spend(parser.c.pos - parser.spent, Site::Parse)?;
        Ok(parser.parsed)
    }

    /// The generators: `Error::Size` where they are more than a u32 id
    /// numbers.
    fn generators(&mut self, budget: &mut Budget) -> Result<(), Unread> {
        self.c.skip_space();
        if self.c.peek().is_none() {
            return Ok(());
        }
        loop {
            narrow::id(self.parsed.generators.len()).map_err(|_| Error::Size)?;
            self.poly(budget)?;
            self.c.skip_space();
            match self.c.peek() {
                None => return Ok(()),
                Some(b',') => self.c.bump(),
                Some(_) => {
                    return Err(self
                        .c
                        .syntax("a sign, a *, a /, a comma, or the end")
                        .into());
                }
            }
        }
    }

    fn poly(&mut self, budget: &mut Budget) -> Result<(), Unread> {
        let start = self.parsed.terms.len().narrow();
        self.c.skip_space();
        let mut negative = self.c.sign().unwrap_or(false);
        loop {
            self.term(negative, budget)?;
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

    /// Parses one term and adds it to `terms`, and spends the bytes since
    /// the last spend and its monomial: `Error::Size` where the terms,
    /// their factors, or their distinct monomials are more than a u32 id
    /// numbers, or the term's line reaches `MINUS` (`Term::new`).
    fn term(&mut self, negative: bool, budget: &mut Budget) -> Result<(), Unread> {
        for &i in &self.touched {
            self.exponents[i] = 0;
        }
        self.touched.clear();
        self.c.skip_space();
        let line = self.c.line;
        loop {
            self.c.skip_space();
            if let Some(factor) = self.c.factor()? {
                narrow::id(self.parsed.factors.len()).map_err(|_| Error::Size)?;
                self.parsed.factors.push(factor);
            } else if let Some((i, e)) = self.c.power(self.names, self.index)? {
                if self.exponents[i] == 0 && e > 0 {
                    self.touched.push(i);
                }
                raise(&mut self.exponents, i, e);
            } else {
                return Err(self.c.syntax("a term").into());
            }
            // instance-read-time.5 design-checked -- after a factor the term reads the next byte: a `*` steps to the next factor, and a sign, a comma, or the text's end ends the term, with no scan of spaces and denominators, which every other byte takes, not those scans after every factor (log 0160-1, 0165-1, 0165-2)
            match self.c.peek() {
                Some(b'*') => {
                    self.c.bump();
                    continue;
                }
                Some(b'+' | b'-' | b',') | None => break,
                Some(_) => {}
            }
            // Any factor, a power among them, may be followed by denominators,
            // each a factor 1/d of the term: `y/3`, `y^2/3*x`, `1/2/3`.
            while let Some(denominator) = self.c.denominator()? {
                narrow::id(self.parsed.factors.len()).map_err(|_| Error::Size)?;
                self.parsed.factors.push(Factor {
                    numerator: Literal::Small(1),
                    denominator: Some(denominator),
                });
            }
            self.c.skip_space();
            if self.c.peek() == Some(b'*') {
                self.c.bump();
            } else {
                break;
            }
        }
        let monomial = self.distinct(budget)?;
        narrow::id(self.parsed.terms.len()).map_err(|_| Error::Size)?;
        let end = self.parsed.factors.len().narrow();
        self.parsed
            .terms
            .push(Term::new(end, monomial, line, negative)?);
        let bytes = self.c.pos - self.spent;
        self.spent = self.c.pos;
        budget.spend(bytes + monomial_units(self.touched.len()), Site::Parse)?;
        Ok(())
    }

    /// The number of the current term's monomial among the text's distinct
    /// monomials, in the order the text first writes them: found by its hash
    /// over the variables the term names (`monomial_hash`) and the test
    /// that a distinct monomial names as many variables with the term's
    /// exponents, and, where none does, numbered next, its variables and
    /// exponents appended to `named`. `MonoId::NONE` where its degree passes
    /// `Dense32`. `Error::Size` where the distinct monomials, or their
    /// variables, are more than a u32 id numbers; the stop where the
    /// budget's check stops the call in a growth of the index, spent at
    /// `Site::Parse`.
    fn distinct(&mut self, budget: &mut Budget) -> Result<MonoId, Unread> {
        let (exponents, touched) = (&self.exponents, &self.touched);
        let degree: u64 = touched.iter().map(|&i| u64::from(exponents[i])).sum();
        if degree > u64::from(Dense32::MAX_DEGREE) {
            return Ok(MonoId::NONE);
        }
        let p = &mut self.parsed;
        let hash = monomial_hash(&p.multipliers, touched.iter().map(|&i| (i, exponents[i])));
        let found = p.index.find(hash, |k| {
            let named = &p.named[p.named_at[k] as usize..p.named_at[k + 1] as usize];
            named.len() == touched.len() && named.iter().all(|&(v, e)| exponents[v as usize] == e)
        });
        let slot = match found {
            Ok(k) => return Ok(MonoId::at(k)),
            Err(slot) => slot,
        };
        let distinct = p.named_at.len();
        let id = MonoId::new(distinct - 1).map_err(|_| Error::Size)?;
        // The variables number below 2^32, as the names' index does, and an
        // exponent past u32::MAX is cut there (`raise`).
        p.named
            .extend(touched.iter().map(|&i| (i.narrow(), exponents[i])));
        let at = narrow::id(p.named.len()).map_err(|_| Error::Size)?;
        p.named_at.push(at);
        p.index
            .insert(slot, hash, distinct, budget, Site::Parse, |k| {
                let named = &p.named[p.named_at[k] as usize..p.named_at[k + 1] as usize];
                monomial_hash(&p.multipliers, named.iter().map(|&(v, e)| (v as usize, e)))
            })?;
        Ok(id)
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
        let first = parsed.terms.last().map_or(0, |t| t.end);
        parsed.error = Some((error, first..parsed.factors.len().narrow()));
    }
}

// text-stop-latency.2 design-checked -- a writer over GF(p) spends the bytes of each chunk of CHUNK, 2^10, terms after it, the inner loop over a chunk's terms as before, not a spend per term nor a test of the term's index per term to decide the spend (log 0057-1)
/// The terms a writer over GF(p) writes between two spends of the bytes it
/// wrote.
pub(crate) const CHUNK: usize = 1 << 10;

// text-stop-latency.3 design-checked -- over Q each term spends, right after it, its bytes plus the square of the limbs of its numerator and of its denominator, not its bytes alone nor a spend per polynomial (log 0057-1)
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

// text-stop-latency.1 design-checked -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
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
    // result-time.11 design-checked -- the basis text over GF(p) is reserved once, before its `[`, at the sum of its terms' longest texts and its brackets and separators, not grown by doubling from the first term's reservation or as a String grows (log 0042-1)
    let k = polys.clone().count();
    let terms: usize = polys.clone().map(|g| g.len().max(1)).sum();
    out.reserve(reservation(
        (4 + 2 * k.saturating_sub(1)).saturating_add(terms.saturating_mul(term_longest(names))),
    ));
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

// text-time.1 practice-checked: memoization -- the basis text over GF(p) writes each distinct monomial's text once per call, into an arena, and copies it from there for each further term with that monomial, not each term's monomial from its exponents (log 0048-1)
// text-time.2 practice-checked: direct-address table -- the arena's index is a vector of one u32 per id of the table, zeroed for each basis text, the offset of the text's length byte in the arena, not a hash map from id to offset or 8 bytes an id with the length beside the offset (log 0048-1)
// text-time.3 design-checked -- a monomial whose text passes 255 bytes is not kept, and each of its terms writes it from its exponents, not a length of two bytes or more (not measured)
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
    /// The variables of the monomials written from their exponents since
    /// the last `visited`, which the writer spends.
    variables: usize,
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
            variables: 0,
        }
    }

    /// The variables of the monomials written from their exponents since
    /// the last call.
    fn visited(&mut self) -> usize {
        std::mem::take(&mut self.variables)
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
        self.variables += self.e.len();
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

/// Writes a polynomial as `write_poly` writes it, each monomial's text from
/// `texts`, and spends its bytes as it spends them, beside the variables of
/// each monomial `texts` writes from its exponents, a unit each.
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
    // result-time.3 design-checked -- the text is written as bytes into the vector under the String, ASCII and each name whole, not characters pushed into the String or a vector checked by `String::from_utf8` at the end (log 0042-1)
    // SAFETY: every byte written is ASCII or a byte of a name, and each name
    // is a `&str` written whole, so the String stays UTF-8.
    let bytes = unsafe { out.as_mut_vec() };
    let longest = term_longest(names);
    // text-stop-latency.2 design-checked -- a writer over GF(p) spends the bytes of each chunk of CHUNK, 2^10, terms after it, the inner loop over a chunk's terms as before, not a spend per term nor a test of the term's index per term to decide the spend (log 0057-1)
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
        // call-stop-latency.29 design-checked -- a text spends the variables of each monomial it writes from its exponents, a basis text's first writing of each and each term of a cofactor text or a text over Q, beside its bytes, not its bytes alone (text-stop-latency.1, .2), as a monomial's writing scans every variable and writes the few it holds (log 0138-6)
        budget.spend(bytes.len() - mark + texts.visited(), Site::Text)?;
    }
    Ok(())
}

// text-time.9 design-checked -- the cofactor texts over GF(p) write each monomial from its exponents at each term, not through the basis text's arena of the monomials' texts (not measured)
// text-stop-latency.1 design-checked -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
/// Writes a polynomial in the canonical spelling, `0` for zero: its terms
/// joined by `+`, each its coefficient in [1, p), then `*` and the monomial,
/// where the coefficient 1 and the `*` are left out before a monomial and
/// the monomial 1 after a coefficient. Spends a unit of `budget` per byte of
/// its terms and per variable of each term, whose monomial it writes from
/// its exponents, after every `CHUNK` terms and at its end; the `0` of zero
/// spends nothing. A stop where the budget's check stops the call, the
/// text then cut short.
#[cfg(any(feature = "cofactors", test))]
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
    // result-time.3 design-checked -- the text is written as bytes into the vector under the String, ASCII and each name whole, not characters pushed into the String or a vector checked by `String::from_utf8` at the end (log 0042-1)
    // SAFETY: every byte written is ASCII or a byte of a name, and each name
    // is a `&str` written whole, so the String stays UTF-8.
    let bytes = unsafe { out.as_mut_vec() };
    let mut e = vec![0u32; table.format().variables()];
    let longest = term_longest(names);
    // text-stop-latency.2 design-checked -- a writer over GF(p) spends the bytes of each chunk of CHUNK, 2^10, terms after it, the inner loop over a chunk's terms as before, not a spend per term nor a test of the term's index per term to decide the spend (log 0057-1)
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
                // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
                table.unpack(m, &mut e);
                w.monomial(names, &e);
            };
            // SAFETY: a term is at most a `+`, a coefficient below 2^64, a
            // `*`, and the monomial, which `term_longest` counts.
            unsafe { spare(bytes, longest, write) };
        }
        // call-stop-latency.29 design-checked -- a text spends the variables of each monomial it writes from its exponents, a basis text's first writing of each and each term of a cofactor text or a text over Q, beside its bytes, not its bytes alone (text-stop-latency.1, .2), as a monomial's writing scans every variable and writes the few it holds (log 0138-6)
        budget.spend(bytes.len() - mark + ms.len() * e.len(), Site::Text)?;
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

// text-reserve.1 design-checked -- a basis text reserves at once its bound or RESERVE_CAP, whichever is less, and grows past it as its terms reserve their room, not its whole bound, which counts a coefficient of the widest field and ten digits for each variable's exponent in every term (not measured)
// The whole bound passed the machine's memory in one request on
// katsura-16 over p65521, whose basis text fits it, and the allocator's
// refusal aborted the process (the call report of grobx 0.73.2).
/// The bytes a basis text reserves at once at most: 1 GiB. A text past it
/// grows by doubling, so that its largest request stays within about twice
/// its length.
pub(crate) const RESERVE_CAP: usize = 1 << 30;

/// The bytes a basis text of at most `bound` bytes reserves at once.
pub(crate) fn reservation(bound: usize) -> usize {
    bound.min(RESERVE_CAP)
}

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

// result-time.4 design-checked -- each term is written through a writer over the vector's spare capacity, reserved for the term's longest text, without bounds checks outside the test profile, and the vector's length is set once per term, not a push per byte that checks the capacity and stores the length (log 0033-1, 0042-1)
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

// result-time.5 practice-checked: two-digit lookup table -- a number below 2^64 is written two digits at a time from a table of the hundred pairs, not one digit at a time into a buffer read back through `str::from_utf8`
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

    // text-time.4 design-checked -- a kept text of at most 32 bytes is copied by one copy of the constant 32 bytes where the term's writer holds 32 bytes past what it has written, and every other by its length, not every copy by its length or a reservation 32 bytes larger per term (log 0048-1)
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

// result-time.9 design-checked -- the writers of a number and of a monomial stand in text.rs, and the writer over Q calls them, not the writer of a large number in rational.rs beside the writer over Q (log 0033-1)
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
        // q-basis-text-time.1 cited-checked: brent-2010 §1.7.2, Algorithm 1.26 -- for (Q, R) = DivRem(A, B^k), A is written as Q followed by R padded to k digits, in O(M(n) log n), where IntegerOutput's divisions by B take O(n²)
        Some(_) => push_split(out, x),
    }
}

// q-basis-text-time.8 tuned-checked -- SPLIT_LIMBS is 12: an integer of 2 to 12 limbs is written by the passes as text-time.5 wrote it, not the passes up to 2, 4, 8, 16, or 24 limbs (log 0086-1)
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
            // text-time.6 design-checked -- each step divides a u128 by the constant 10^19, which the compiler computes, not by Möller and Granlund's 2 by 1 division with a precomputed reciprocal (log 0048-1)
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

// q-basis-text-time.9 tuned-checked -- a part of at most LEAF_CHUNKS, 8, chunks is written by the passes, exactly its chunks with zeros at the top, not leaves of 2, 3, 4, 6, 16, or 32 chunks nor leaves by a fraction whose chunks come by multiplications (log 0086-1)
/// The most chunks of a part `leaf` writes. A part of 8 chunks is below
/// 10^152 < 2^512, so of at most 8 limbs.
const LEAF_CHUNKS: usize = 8;

// q-basis-text-time.6 practice-checked: lazy initialization -- the powers D_j = 10^(19·2^j) and their reciprocals are made once per process, at the first integer that needs level j, and kept in a static array of `OnceLock`s for every later call and thread, not made per call, per text, or per integer (not measured)
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

// q-basis-text-time.2 design-checked -- a part of k chunks splits at h = 2^j, the largest power of two below k, by D_j, not at ⌈k/2⌉ by a power and a reciprocal per h (log 0086-1)
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

// q-basis-text-time.1 cited-checked: brent-2010 §1.7.2, Algorithm 1.26 -- for (Q, R) = DivRem(A, B^k), A is written as Q followed by R padded to k digits, in O(M(n) log n), where IntegerOutput's divisions by B take O(n²)
// q-basis-text-time.7 design-checked -- the scratch of a conversion, x's limbs, its chunks, and the splits' parts and columns, is one vector allocated per integer of more than SPLIT_LIMBS limbs, not a scratch kept from integer to integer, thread-local or a converter's own (not measured)
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
    if limbs::cmp(x, power).is_lt() {
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

// q-basis-text-time.3 cited-checked: barrett-1987 §Longmod procedure, Diagram Five, via vanderhoeven-2016 §2.2.1, Function 6 and Proposition 1 -- with R = b^(2n) div M, the n + 1 leading digits of W times R, their product's leading digits times M subtracted from W modulo b^(n+1), leave x below 3M, and at most two subtractions of M give W mod M
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
    // q-basis-text-time.4 cited-checked: barrett-1987 §Longmod procedure -- each long multiplication of the reduction calculates only half its product, since the other half is not required
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
    while limbs::cmp(trimmed(r), power).is_ge() {
        sub_in_place(r, power);
        increment(q);
        corrections += 1;
    }
    debug_assert!(
        corrections <= 3,
        "Barrett's quotient {corrections} below x's"
    );
}

// q-basis-text-time.5 practice-checked: Comba multiplication -- the products of a split are taken column by column, each column's products summed into a three-word accumulator and its low word stored, not row by row with each row's carries through the result in memory, nor by num-bigint's product, which allocates each
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
pub(crate) fn sub_in_place(r: &mut [u64], y: &[u64]) {
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

// text-copy-peak.1 design-checked -- a text is handed over a chunk at a time, and the whole pages of it behind each chunk are given back to the system, not handed over whole and its pages given back when it is dropped, nor handed over after the measured call, nor written by the core into the receiver's memory (log 0081-1)
// text-copy-peak.4 design-checked -- the pages are given back by madvise(MADV_DONTNEED), not by shrinking the vector, which the allocator may move or keep whole (not measured)
// text-copy-peak.6 design-checked -- each chunk is checked ASCII before it is handed, and a byte past ASCII ends the hand-over with an error, not the text trusted unchecked, nor the whole text checked before the first chunk (not measured)
// text-copy-peak.10 design-checked -- the hand-over and its madvise are a function of the core's text, which the Python module calls with its copy, not the module calling madvise itself (not measured)
/// Hands `text` to `put` in chunks of at most `chunk` bytes, in order,
/// each with its offset in the text, and gives back to the system the
/// whole pages of the text behind each chunk, so that the text and what
/// `put` builds of it stand about once. Each chunk is checked ASCII before
/// it is handed. The text is consumed.
///
/// # Errors
///
/// `InvalidData` where a byte is past ASCII, the chunk that holds it not
/// handed; `Unsupported` where the page size cannot be read; and the
/// error of `os::discard` where the system refuses pages back.
pub fn hand_over(
    text: String,
    chunk: NonZeroUsize,
    mut put: impl FnMut(usize, &[u8]),
) -> io::Result<()> {
    // text-copy-peak.9 design-checked -- the page size is the system's, read by sysconf, not 4096 as a constant (not measured)
    let page = os::page_size()
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
            // `put` has been handed bytes[given..end], and nothing reads
            // them again before the drop.
            os::discard(&mut bytes[given..end], page)?;
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
