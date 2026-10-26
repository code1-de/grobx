//! The rows stage in the cheap mode: the row of every nonzero element of G
//! is checked at random points instead of expanded (`OBJECTIVES.md`,
//! _Checker_), save the rows `Mode::Cheap` names. `Mode::Cheap` states the
//! bound it keeps.

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::check::{Read, Side, Stage, exact_row};
use crate::field::{DIGITS, Field};
use crate::gfp::INVERSE;
use crate::parse::{self, Numbers, Ring, Rows};
use crate::poll::Poll;
use crate::{Error, Flaw};

/// The checker's own constant, "checker" in ASCII.
const SALT: u64 = 0x0063_6865_636b_6572;

/// The largest exponent a power table holds.
const TABLE: u32 = 1024;

/// A point: its modulus, its coordinates, and the values of F and G at it.
pub struct Point {
    pub field: Field,
    coords: Vec<u64>,
    /// Per variable the powers a⁰, a¹, …, a^E of its coordinate, back to back.
    powers: Vec<u64>,
    /// E + 1.
    stride: usize,
    /// The value of each generator.
    f: Vec<u64>,
    /// The value of each element of G as read.
    g: Vec<u64>,
}

impl Point {
    /// The value of the monomial m.
    #[inline]
    pub fn mono(&self, m: &[u32]) -> u64 {
        let mut value = None;
        for (v, &e) in m[1..].iter().enumerate() {
            if e == 0 {
                continue;
            }
            let power = if (e as usize) < self.stride {
                self.powers[v * self.stride + e as usize]
            } else {
                self.field.pow(self.coords[v], u64::from(e))
            };
            value = Some(match value {
                None => power,
                Some(x) => self.field.mul(x, power),
            });
        }
        value.unwrap_or(1)
    }

    /// c times the value of the monomial whose variable factors are
    /// `factors`, each a variable and its exponent.
    #[inline]
    fn term(&self, c: u64, factors: &[(usize, u32)]) -> u64 {
        let mut value = c;
        for &(v, e) in factors {
            let power = if (e as usize) < self.stride {
                self.powers[v * self.stride + e as usize]
            } else {
                self.field.pow(self.coords[v], u64::from(e))
            };
            value = self.field.mul(value, power);
        }
        value
    }

    /// The value of a polynomial given as its monomials, back to back, and
    /// its coefficients reduced modulo the point's modulus.
    pub fn value(&self, monos: &[u32], coefs: impl Iterator<Item = u64>) -> u64 {
        let w = self.coords.len() + 1;
        let mut acc = 0;
        for (m, c) in monos.chunks_exact(w).zip(coefs) {
            acc = self.field.add(acc, self.field.mul(c, self.mono(m)));
        }
        acc
    }
}

// checker-cheap.14 practice: rejection sampling -- a residue is drawn by rejection, a draw below 2^64 mod m drawn again, not taken mod m at once
/// A residue in [0, m), every one equally likely.
fn uniform(rng: &mut Xoshiro256PlusPlus, m: u64) -> u64 {
    debug_assert!(m > 0, "m > 0");
    let r = m.wrapping_neg() % m;
    loop {
        let x = rng.next_u64();
        if x >= r {
            return x % m;
        }
    }
}

/// The primes below 2^16.
fn small_primes() -> &'static [u64] {
    static PRIMES: OnceLock<Vec<u64>> = OnceLock::new();
    PRIMES.get_or_init(|| {
        let mut composite = vec![false; 1 << 16];
        let mut primes = Vec::new();
        for p in 2..composite.len() {
            if composite[p] {
                continue;
            }
            primes.push(p as u64);
            for multiple in (p * p..composite.len()).step_by(p) {
                composite[multiple] = true;
            }
        }
        primes
    })
}

/// The least modulus of a point over Q, 2^31, and of the primes drawn
/// (checker-cheap.6).
const LEAST_MODULUS: u64 = 1 << 31;

// checker-cheap.6 design -- the primes are drawn uniformly from [2^31, 2^32) and tested by trial division by the primes below 2^16, not taken near 2^62 and tested by Miller–Rabin (not measured)
/// A prime drawn uniformly from those in [2^31, 2^32). Every composite below
/// 2^32 has a prime factor below 2^16.
fn prime(rng: &mut Xoshiro256PlusPlus) -> u64 {
    loop {
        // The draw's top 31 bits, an offset below `LEAST_MODULUS`.
        let x = LEAST_MODULUS + (rng.next_u64() >> (u64::BITS - LEAST_MODULUS.trailing_zeros()));
        if small_primes().iter().all(|&p| !x.is_multiple_of(p)) {
            return x;
        }
    }
}

// checker-points-abort.2 design -- a batch holds as many points as fit a budget in bytes, counted from a point's arrays and a row's entries at it, and at least one, not a fixed number of points (not measured)
// checker-points-abort.3 design -- the budget is 2^24 bytes, not 2^20 or 2^26 (not measured)
/// The bytes the points of a batch and a row's entries at them may take.
const BUDGET: usize = 1 << 24;

// checker-points-abort.1 cited: demin-2026 §4.3, Remark _Memory consumption_ -- a larger batch of primes raises the memory, Katsura-13 from 2.7 GB at N = 1 to 29.3 GB at N = 32, and large batches are unattractive
/// The cheap mode's points, drawn from the seed a batch at a time. The
/// points of the batches, one after the other, are the points one batch of
/// them all holds: the generator runs on from batch to batch.
struct Draws {
    rng: Xoshiro256PlusPlus,
    characteristic: u64,
    /// The number of variables.
    n: usize,
    /// The power table's stride, E + 1.
    stride: usize,
}

impl Draws {
    /// The draws from `seed`. A term of F and G is a unit of work for
    /// `poll`; where it stops, the draws mean nothing.
    fn new<S: Side>(side: &S, ring: &Ring, seed: u64, poll: &mut Poll) -> Draws {
        // checker-cheap.13 design -- the generator is seeded with the seed xor a constant of the checker's own, not with the seed as passed (not measured)
        let generator = Xoshiro256PlusPlus::seed_from_u64(seed ^ SALT);
        // checker-cheap.10 design -- each point keeps a table of its coordinates' powers up to the largest exponent in F and G, at most 1024, not square-and-multiply for every factor (not measured)
        let largest = (0..side.generators())
            .map(|j| side.generator_monos(j))
            .chain((0..side.len()).map(|i| side.monos(i)))
            .take_while(|monos| !poll.work(monos.len() / ring.w))
            .flat_map(|monos| monos.chunks_exact(ring.w))
            .flat_map(|m| m[1..].iter().copied())
            .max()
            .unwrap_or(0)
            .min(TABLE);
        Draws {
            rng: generator,
            characteristic: ring.characteristic,
            n: ring.w - 1,
            stride: largest as usize + 1,
        }
    }

    /// The bytes one point and a row's entries at it take: the point's
    /// coordinates, powers, and values of F and G, a value per entry of a
    /// row of one entry per generator, and a coefficient's residue.
    fn point_bytes<S: Side>(&self, side: &S) -> usize {
        let words = self.n + self.n * self.stride + 2 * side.generators() + side.len() + 1;
        8 * words + size_of::<Point>()
    }

    /// The points of a batch: as many as fit `BUDGET`, at least one.
    fn per<S: Side>(&self, side: &S) -> usize {
        (BUDGET / self.point_bytes(side)).max(1)
    }

    /// The next `count` points, in order, each with the values of F and G
    /// at it; None where `poll`, asked once per draw, a point drawn again
    /// included, stops.
    fn next<S: Side>(&mut self, side: &S, count: usize, poll: &mut Poll) -> Option<Vec<Point>> {
        let mut points = Vec::with_capacity(count);
        while points.len() < count {
            if poll.step() {
                return None;
            }
            // checker-cheap.5 cited: parisse-2013 §2 _Probabilistic checking algorithm_ -- the check runs modulo several primes and stops once the inverse of their product is below a fixed ε
            let field = match self.characteristic {
                0 => Field::new(prime(&mut self.rng)),
                p => Field::new(p),
            };
            let coords: Vec<u64> = (0..self.n)
                .map(|_| uniform(&mut self.rng, field.modulus()))
                .collect();
            let mut powers = Vec::with_capacity(self.n * self.stride);
            for &a in &coords {
                let mut x = 1;
                for _ in 0..self.stride {
                    powers.push(x);
                    x = field.mul(x, a);
                }
            }
            let mut point = Point {
                field,
                coords,
                powers,
                stride: self.stride,
                f: Vec::new(),
                g: Vec::new(),
            };
            let (mut f, mut g) = (Vec::new(), Vec::new());
            // checker-cheap.7 cited: monagan-2004 _Remark 3_ -- if a prime divides a denominator of a coefficient of the polynomial being reconstructed, the modular algorithm must either not use it or discard it
            let defined = side.values(&point, &mut f, &mut g, poll);
            if poll.stopped() {
                return None;
            }
            if defined {
                point.f = f;
                point.g = g;
                points.push(point);
            }
        }
        Some(points)
    }
}

/// The moduli of the k points the cheap mode draws from `seed`, in order.
pub fn moduli<S: Side>(side: &S, ring: &Ring, k: u32, seed: u64) -> Vec<u64> {
    let never = &mut || false;
    let poll = &mut Poll::new(never);
    let mut draws = Draws::new(side, ring, seed, poll);
    let per = draws.per(side);
    let mut moduli = Vec::new();
    let mut left = k as usize;
    while left > 0 {
        let count = left.min(per);
        left -= count;
        #[allow(
            clippy::expect_used,
            reason = "next is None only where the poll stops, and this poll never answers true"
        )]
        let points = draws
            .next(side, count, poll)
            .expect("a poll that never stops does not stop the draws");
        moduli.extend(points.iter().map(|point| point.field.modulus()));
    }
    moduli
}

// checker-cheap.9 design -- a row's entries are evaluated at the points as they are read, not read into polynomials and evaluated (not measured)
// checker-stop-latency.9 design -- the `Numbers` functions take the poll as a parameter, not a poll held in `Eval` behind a `RefCell` or a `Numbers` wrapper that holds the poll for the exact reading (not measured)
/// The numbers the cheap mode reads a row with: each term's value at the
/// points is added to its entry's as the term is read. Once the check's
/// poll has stopped, no term is evaluated.
struct Eval<'a> {
    points: &'a [Point],
    /// Over Q every point has a modulus of its own; over GF(p) all share
    /// the first point's.
    q: bool,
    /// The residues of the last term, kept for the next.
    spare: Cell<Vec<u64>>,
    /// The variable factors of the last term, kept for the next.
    factors: Cell<Vec<(usize, u32)>>,
}

/// A term's coefficient while the term is read.
struct Coef {
    /// Its residue modulo each modulus: over GF(p) one, over Q one per point.
    res: Vec<u64>,
    /// Over Q: a factor was the integer 0.
    zero: bool,
    /// Over Q: a denominator read before any factor 0 is 0 modulo a
    /// point's prime.
    vanished: bool,
}

/// An entry of a row as read.
#[derive(Default)]
struct Entry {
    /// Its value at each point; empty while no term has been added.
    values: Vec<u64>,
    /// The largest degree of its nonzero terms.
    degree: Option<u32>,
    /// Over Q: a denominator of a term, read before any factor 0 of it, is
    /// 0 modulo a point's prime.
    vanished: bool,
}

// checker-points-abort.7 design -- the poll is asked as a row's terms are evaluated at the points, a term at a point a unit, and 19 digits of a numerator, and over Q of a denominator, a unit at each point over Q and once for the batch over GF(p), not once per row alone (log 0047-2)
impl Numbers for Eval<'_> {
    type Term = Coef;
    type Terms = Entry;
    type Poly = Entry;

    fn one(&self) -> Coef {
        let mut res = self.spare.take();
        res.clear();
        res.resize(if self.q { self.points.len() } else { 1 }, 1);
        Coef {
            res,
            zero: false,
            vanished: false,
        }
    }

    fn times(&self, c: &mut Coef, digits: &[u8], poll: &mut Poll) {
        if self.q && digits.iter().all(|&b| b == b'0') {
            c.zero = true;
        }
        if c.zero || poll.work(c.res.len() * digits.len().div_ceil(DIGITS)) {
            return;
        }
        for (r, point) in c.res.iter_mut().zip(self.points) {
            *r = point.field.mul(*r, point.field.decimal(digits));
        }
    }

    fn over(&self, c: &mut Coef, digits: &[u8], poll: &mut Poll) -> Option<()> {
        if !self.q {
            if poll.work(INVERSE) {
                return Some(());
            }
            let field = self.points[0].field;
            let inverse = field.inv(field.decimal(digits))?;
            c.res[0] = field.mul(c.res[0], inverse);
            return Some(());
        }
        // checker-cheap.17 design -- over Q a denominator that is the integer 0 is refused as it is read, not marked and refused when the row is read again (not measured)
        if digits.iter().all(|&b| b == b'0') {
            return None;
        }
        if c.zero || poll.work(c.res.len() * (digits.len().div_ceil(DIGITS) + INVERSE)) {
            return Some(());
        }
        for (r, point) in c.res.iter_mut().zip(self.points) {
            match point.field.inv(point.field.decimal(digits)) {
                Some(inverse) => *r = point.field.mul(*r, inverse),
                None => c.vanished = true,
            }
        }
        Some(())
    }

    fn push(&self, entry: &mut Entry, m: &[u32], c: Coef, negative: bool, poll: &mut Poll) {
        let Coef {
            res,
            zero,
            vanished,
        } = c;
        // Over GF(p) a residue of 0 is a zero coefficient.
        let zero = zero || !self.q && res[0] == 0;
        if vanished {
            entry.vanished = true;
        } else if !zero && !entry.vanished && !poll.work(self.points.len()) {
            entry.degree = entry.degree.max(Some(m[0]));
            if entry.values.is_empty() {
                entry.values.resize(self.points.len(), 0);
            }
            let mut factors = self.factors.take();
            factors.clear();
            factors.extend(
                m[1..]
                    .iter()
                    .enumerate()
                    .filter(|&(_, &e)| e > 0)
                    .map(|(v, &e)| (v, e)),
            );
            for (t, (point, value)) in self.points.iter().zip(&mut entry.values).enumerate() {
                let field = point.field;
                let x = point.term(res[if self.q { t } else { 0 }], &factors);
                *value = field.add(*value, if negative { field.neg(x) } else { x });
            }
            self.factors.set(factors);
        }
        self.spare.set(res);
    }

    fn finish(&self, entry: &mut Entry, _: &Ring, out: &mut Entry, _: &mut Poll) {
        if entry.values.is_empty() {
            entry.values.resize(self.points.len(), 0);
        }
        std::mem::swap(entry, out);
        entry.values.clear();
        entry.degree = None;
        entry.vanished = false;
    }
}

/// The largest degree of a term of the polynomial whose monomials, `w`
/// words each, are `monos`; None where it has no term.
fn degree(monos: &[u32], w: usize) -> Option<u32> {
    debug_assert!(
        w > 0 && monos.len().is_multiple_of(w),
        "whole monomials of w words"
    );
    monos.chunks_exact(w).map(|m| m[0]).max()
}

/// The degree bound of a row: the largest of deg gᵢ, `g`, and of deg fⱼ
/// plus the entry's largest degree, over the nonzero fⱼ whose entry has a
/// nonzero term.
fn bound(g: u32, degrees: &[Option<u32>], row: &[Entry]) -> u64 {
    let mut d = u64::from(g);
    for (f, entry) in degrees.iter().zip(row) {
        if let (Some(f), Some(e)) = (f, entry.degree) {
            d = d.max(u64::from(*f) + u64::from(e));
        }
    }
    d
}

/// The entries of a row a unit of work counts at each point, beside the
/// unit of the point itself, before the row is checked at the points
/// (checker-stop-latency.12).
const ENTRIES_PER_UNIT: usize = 4;

/// The rows stage at k points drawn from `seed`: the row of every nonzero
/// element in the cofactor matrix `text` expresses it in F at every point,
/// or the first flaw. A zero element's row is read and its shape checked.
/// The points come in batches of `per`, or of as many as fit `BUDGET`
/// where `per` is None, and the result is the result of one batch of k
/// points. `poll` is asked once per draw, once per row in each batch,
/// and as the row's terms are evaluated.
pub fn rows<S: Side>(
    side: &mut S,
    ring: &Ring,
    text: &str,
    k: u32,
    seed: u64,
    per: Option<NonZeroUsize>,
    poll: &mut Poll,
) -> Result<Stage, Error> {
    let mut draws = Draws::new(side, ring, seed, poll);
    if poll.stopped() {
        return Ok(None);
    }
    let per = per.map_or_else(|| draws.per(side), NonZeroUsize::get);
    // checker-cheap.3 design -- a row whose degree bound reaches the smallest modulus is read again and checked exactly, not at points in an extension field (not measured)
    // checker-points-abort.5 design -- the bound is p over GF(p) and 2^31 over Q, at or below every modulus a point can have, not the least modulus among the drawn points, which is known only once every point is drawn (not measured)
    let least = match ring.characteristic {
        0 => LEAST_MODULUS,
        p => p,
    };
    // In an order that is not degree compatible the leading term need not
    // have the largest degree.
    let degrees: Vec<Option<u32>> = (0..side.generators())
        .map(|j| degree(side.generator_monos(j), ring.w))
        .collect();
    let mut first: Option<Event> = None;
    // Per row read, whether it was checked exactly.
    let mut checked: Vec<bool> = Vec::new();
    let mut exact: Vec<Read<S>> = Vec::new();
    let mut left = k as usize;
    // checker-points-abort.1 cited: demin-2026 §4.3, Remark _Memory consumption_ -- a larger batch of primes raises the memory, Katsura-13 from 2.7 GB at N = 1 to 29.3 GB at N = 32, and large batches are unattractive
    while left > 0 {
        let count = left.min(per);
        left -= count;
        let Some(points) = draws.next(side, count, poll) else {
            return Ok(None);
        };
        let eval = Eval {
            points: &points,
            q: ring.characteristic == 0,
            spare: Cell::default(),
            factors: Cell::default(),
        };
        let mut rows = Rows::new(text)?;
        let mut entries = Vec::new();
        for i in 0.. {
            // The rows past the earliest event, and its own row unless it
            // failed at a point, cannot change the result.
            if first
                .as_ref()
                .is_some_and(|e| e.row < i || e.row == i && !e.point)
            {
                break;
            }
            let next = rows.next(ring, &eval, &mut entries, poll);
            if poll.stopped() {
                return Ok(None);
            }
            let (line, k) = match next {
                Ok(Some(row)) => row,
                Ok(None) => {
                    if i != side.len() {
                        record(&mut first, Event::at(i, Ok(Flaw::Shape)));
                    }
                    break;
                }
                Err(error) => {
                    record(&mut first, Event::at(i, Err(error)));
                    break;
                }
            };
            if poll.step() {
                return Ok(None);
            }
            if checked.len() == i {
                checked.push(false);
            }
            // A row checked exactly in an earlier batch is passed over.
            if checked[i] {
                continue;
            }
            let row = &entries[..k];
            // checker-cheap.8 design -- a row in which a denominator vanishes modulo a point's prime is read again and checked exactly, not every row again at a new point (not measured)
            let again = if row.iter().any(|entry| entry.vanished) {
                true
            } else {
                // A row that failed at a point of an earlier batch, and
                // vanished at no point of this one, stays failed.
                if first.as_ref().is_some_and(|e| e.row == i) {
                    break;
                }
                // Rows past the last element are read, and counted after the
                // last row.
                if i >= side.len() {
                    continue;
                }
                if k != side.generators() {
                    record(&mut first, Event::at(i, Ok(Flaw::Shape)));
                    break;
                }
                let Some(g) = degree(side.monos(i), ring.w) else {
                    continue;
                };
                let d = bound(g, &degrees, row);
                // checker-cheap.4 design -- a row whose degree bound passes u32 is read again and checked exactly, not refused with Overflow at once (not measured)
                d > u64::from(u32::MAX) || d >= least
            };
            if again {
                checked[i] = true;
                let flaw = parse::row(text, rows.start(), ring, side.numbers(), &mut exact, poll)
                    .and_then(|k| exact_row(side, i, line, &exact[..k], poll));
                if poll.stopped() {
                    return Ok(None);
                }
                match flaw {
                    Ok(None) => {}
                    Ok(Some(flaw)) => {
                        record(&mut first, Event::at(i, Ok(flaw)));
                        break;
                    }
                    Err(error) => {
                        record(&mut first, Event::at(i, Err(error)));
                        break;
                    }
                }
            } else if poll.work(points.len() * (1 + k / ENTRIES_PER_UNIT)) {
                return Ok(None);
            } else if !at_points(&points, &degrees, row, i) {
                record(
                    &mut first,
                    Event {
                        row: i,
                        outcome: Ok(Flaw::Cofactors(i)),
                        point: true,
                    },
                );
                break;
            }
        }
    }
    match first {
        None => Ok(Some(Ok(()))),
        Some(Event {
            outcome: Ok(flaw), ..
        }) => Ok(Some(Err(flaw))),
        Some(Event {
            outcome: Err(error),
            ..
        }) => Err(error),
    }
}

/// An event of the rows stage: the flaw or the error at a row, the end of
/// the rows counting as the row after the last.
struct Event {
    row: usize,
    outcome: Result<Flaw, Error>,
    /// The row failed at a point. A later point at which a denominator of
    /// the row vanishes has it read again and checked exactly.
    point: bool,
}

impl Event {
    fn at(row: usize, outcome: Result<Flaw, Error>) -> Event {
        Event {
            row,
            outcome,
            point: false,
        }
    }
}

// checker-points-abort.4 design -- the result is the earliest event in row order over all k points, the row that failed at a point read again in later batches, not each batch's first flaw as it comes (not measured)
/// Records `event` where it lies before the earliest event so far, or at
/// that event's row where that row failed at a point. A pass reads no row
/// past the earliest event, so every event it meets is one of these.
fn record(first: &mut Option<Event>, event: Event) {
    debug_assert!(
        first
            .as_ref()
            .is_none_or(|e| event.row < e.row || event.row == e.row && e.point)
    );
    *first = Some(event);
}

/// Whether the row of the nonzero element i, read as `row`, expresses it
/// in F at every point of `points`.
fn at_points(points: &[Point], degrees: &[Option<u32>], row: &[Entry], i: usize) -> bool {
    debug_assert_eq!(row.len(), degrees.len(), "an entry per generator");
    debug_assert!(
        points.first().is_none_or(|p| i < p.g.len()),
        "i is an element of G"
    );
    points.iter().enumerate().all(|(t, point)| {
        let field = point.field;
        let mut s = field.neg(point.g[i]);
        for (j, entry) in row.iter().enumerate() {
            if degrees[j].is_some() {
                s = field.add(s, field.mul(entry.values[t], point.f[j]));
            }
        }
        s == 0
    })
}
