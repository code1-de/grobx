//! The two verdicts: whether G is a Gröbner basis of ⟨F⟩, and whether G is
//! reduced. The driver here is the same over both fields. Each field's side
//! does the arithmetic: `gfp.rs` over GF(p), `q.rs` over Q.

use std::num::NonZeroUsize;

use crate::cheap::{self, Point};
use crate::divisors::{Divisors, MASKS};
use crate::field::Field;
use crate::gfp::Gfp;
use crate::parse::{self, Header, Numbers, Ring, Rows};
use crate::poll::Poll;
use crate::poly::{self, Overflow};
use crate::q::Q;
use crate::rational::Rationals;
use crate::{Error, Flaw, Mode, Order, Part, Verdict};

/// The verdict, or None where the poll stopped the check.
pub fn check(
    instance: &str,
    basis: &str,
    cofactors: &str,
    order: &Order,
    mode: Mode,
    poll: &mut Poll,
) -> Result<Option<Verdict>, Error> {
    let header = parse::header(instance, order)?;
    match header.ring.characteristic {
        0 => run::<Q>(&header, Rationals, basis, cofactors, mode, poll),
        p => run::<Gfp>(&header, Field::new(p), basis, cofactors, mode, poll),
    }
}

/// The rows stage alone; the cheap mode draws its points in batches of
/// `per`, or of as many as fit its budget where `per` is None.
pub fn check_cofactors(
    instance: &str,
    basis: &str,
    cofactors: &str,
    order: &Order,
    mode: Mode,
    per: Option<NonZeroUsize>,
) -> Result<Result<(), Flaw>, Error> {
    let header = parse::header(instance, order)?;
    match header.ring.characteristic {
        0 => run_rows::<Q>(&header, Rationals, basis, cofactors, mode, per),
        p => run_rows::<Gfp>(&header, Field::new(p), basis, cofactors, mode, per),
    }
}

/// The moduli of the k points the cheap mode draws from `seed`.
pub fn cheap_moduli(
    instance: &str,
    basis: &str,
    order: &Order,
    k: u32,
    seed: u64,
) -> Result<Vec<u64>, Error> {
    let header = parse::header(instance, order)?;
    match header.ring.characteristic {
        0 => moduli::<Q>(&header, Rationals, basis, k, seed),
        p => moduli::<Gfp>(&header, Field::new(p), basis, k, seed),
    }
}

/// A stage's result: the first flaw or none, or None where the poll stopped
/// the check.
pub type Stage = Option<Result<(), Flaw>>;

/// A polynomial as a side reads it.
pub type Read<S> = <<S as Side>::Numbers as Numbers>::Poly;

// checker-q.14 design -- the checks are one driver generic over the field's arithmetic (not measured)
/// The arithmetic of the checks over one field. The driver keeps the rest:
/// the order of the checks, the shape of the cofactor matrix, zero elements,
/// the pairs and their criteria, and divisibility.
pub trait Side: Sized {
    type Numbers: Numbers;

    /// The side for the generators and the elements of G as read, in the
    /// ring's order. A term is a unit of work for `poll`: over GF(p) a term
    /// of an element made monic, over Q a term of a polynomial whose units
    /// at a point it computes. Where it stops, the side means nothing.
    fn new(
        numbers: Self::Numbers,
        ring: &Ring,
        generators: Vec<Read<Self>>,
        elements: Vec<Read<Self>>,
        poll: &mut Poll,
    ) -> Self;

    fn numbers(&self) -> &Self::Numbers;

    /// How many elements G has.
    fn len(&self) -> usize;

    /// The monomials of element i, back to back, the leading one first; none
    /// where element i is zero.
    fn monos(&self, i: usize) -> &[u32];

    /// How many generators F has.
    fn generators(&self) -> usize;

    /// The monomials of generator j, back to back, the leading one first;
    /// none where generator j is zero.
    fn generator_monos(&self, j: usize) -> &[u32];

    // checker-stop-latency.12 design -- the values of F and G at a point count a polynomial's units before it, over GF(p) a unit a term, over Q a unit a term, one per 8 of each coefficient's limbs, and its fraction's limbs, computed once per polynomial, not a point's values counted as one step or the units summed at every point (log 0059-1)
    /// The value at `point` of every generator into `generators` and of
    /// every element of G, as read, into `elements`; false where a
    /// denominator of one of them has no inverse modulo the point's modulus.
    /// A polynomial's terms are work for `poll`, counted before it is
    /// evaluated: a unit a term, over Q with one per 8 of a coefficient's
    /// limbs and its fraction's limbs. Where it stops, the answer means
    /// nothing.
    fn values(
        &self,
        point: &Point,
        generators: &mut Vec<u64>,
        elements: &mut Vec<u64>,
        poll: &mut Poll,
    ) -> bool;

    /// Whether element i is nonzero and scaled as the canonical basis text.
    fn scaled(&self, i: usize) -> bool;

    /// Whether generator j reduces to zero modulo G, or the element of G
    /// whose multiple in a reduction step passed `u32`. The sum's terms,
    /// its products over Q, and the divisor queries are work for `poll`;
    /// where it stops, the answer means nothing.
    fn generator(&mut self, j: usize, index: &mut Divisors, poll: &mut Poll)
    -> Result<bool, usize>;

    /// Whether `entries`, one per generator, express the nonzero element i
    /// in F. The sum's terms, and over Q its products and the scaling of a
    /// cofactor's terms, are work for `poll`; where it stops, the answer
    /// means nothing.
    fn row(&mut self, i: usize, entries: &[Read<Self>], poll: &mut Poll) -> Result<bool, Overflow>;

    /// Whether the S-polynomial of the nonzero elements i and j, whose leading
    /// monomials have the lcm `lcm`, reduces to zero modulo G, or the element
    /// of G whose multiple passed `u32` in it or in a reduction step. The
    /// sum's terms, its products over Q, and the divisor queries are work for
    /// `poll`; where it stops, the answer means nothing.
    fn spair(
        &mut self,
        i: usize,
        j: usize,
        lcm: &[u32],
        index: &mut Divisors,
        poll: &mut Poll,
    ) -> Result<bool, usize>;
}

/// The side for the generators of the instance and the elements of `basis`,
/// and the line each element starts on; None where `poll` stopped the
/// reading or the side's making.
fn side<S: Side>(
    header: &Header,
    numbers: S::Numbers,
    basis: &str,
    poll: &mut Poll,
) -> Result<Option<(S, Vec<usize>)>, Error> {
    let ring = &header.ring;
    let generators = parse::generators(header, &numbers, poll)?;
    if poll.stopped() {
        return Ok(None);
    }
    let (elements, lines) = parse::basis(basis, ring, &numbers, poll)?;
    if poll.stopped() {
        return Ok(None);
    }
    let side = S::new(numbers, ring, generators, elements, poll);
    if poll.stopped() {
        return Ok(None);
    }
    Ok(Some((side, lines)))
}

fn run<S: Side>(
    header: &Header,
    numbers: S::Numbers,
    basis: &str,
    cofactors: &str,
    mode: Mode,
    poll: &mut Poll,
) -> Result<Option<Verdict>, Error> {
    let ring = &header.ring;
    let Some((mut side, lines)) = side::<S>(header, numbers, basis, poll)? else {
        return Ok(None);
    };
    let mut index = Divisors::new(ring.w, (0..side.len()).map(|i| side.monos(i)));
    let Some(reduced) = reduced(&side, &mut index, ring.w, poll) else {
        return Ok(None);
    };
    let Some(groebner) = groebner(&mut side, &mut index, ring, &lines, cofactors, mode, poll)?
    else {
        return Ok(None);
    };
    // A poll that answered true in the last step stops the check too.
    if poll.stopped() {
        return Ok(None);
    }
    Ok(Some(Verdict { groebner, reduced }))
}

// checker-cheap.15 design -- the rows stage runs alone, in either mode, beside the full check, not the full check as the only entry point (log 0001-2)
#[allow(
    clippy::expect_used,
    reason = "the reading and the rows stop only where the poll stops, and this poll never answers true"
)]
fn run_rows<S: Side>(
    header: &Header,
    numbers: S::Numbers,
    basis: &str,
    cofactors: &str,
    mode: Mode,
    per: Option<NonZeroUsize>,
) -> Result<Result<(), Flaw>, Error> {
    let never = &mut || false;
    let (mut side, _) = side::<S>(header, numbers, basis, &mut Poll::new(never))?
        .expect("a poll that never stops does not stop the reading");
    let stage = rows(
        &mut side,
        &header.ring,
        cofactors,
        mode,
        per,
        &mut Poll::new(never),
    )?;
    Ok(stage.expect("a poll that never stops does not stop the check"))
}

fn moduli<S: Side>(
    header: &Header,
    numbers: S::Numbers,
    basis: &str,
    k: u32,
    seed: u64,
) -> Result<Vec<u64>, Error> {
    let never = &mut || false;
    #[allow(
        clippy::expect_used,
        reason = "the reading stops only where the poll stops, and this poll never answers true"
    )]
    let (side, _) = side::<S>(header, numbers, basis, &mut Poll::new(never))?
        .expect("a poll that never stops does not stop the reading");
    Ok(cheap::moduli(&side, &header.ring, k, seed))
}

fn reduced<S: Side>(side: &S, index: &mut Divisors, w: usize, poll: &mut Poll) -> Stage {
    for i in 0..side.len() {
        if poll.step() {
            return None;
        }
        // checker-gfp.18 design -- a zero element of G is left out of the Gröbner checks and fails the reduced verdict (not measured)
        if !side.scaled(i) {
            return Some(Err(Flaw::Scale(i)));
        }
        let monos = side.monos(i);
        // checker-stop-latency.7 design -- a divisor query the memo does not answer counts 1 + n/32 units before its scan, n the divmasks of the index, the chain criterion 1 + n/32 a pair, the memo's growth a unit per monomial it moves, a block of 4096 at a time, and an answer from the memo nothing, not a unit per query or the growth counted at once (log 0059-1)
        if poll.work(1 + index.masks.len() / MASKS) {
            return None;
        }
        if let Some(by) = index.divisor_other(&monos[..w], i) {
            return Some(Err(Flaw::Divisible { element: i, by }));
        }
        for m in monos[w..].chunks_exact(w) {
            if let Some(by) = index.query(m, poll) {
                return Some(Err(Flaw::Divisible { element: i, by }));
            }
            if poll.stopped() {
                return None;
            }
        }
    }
    Some(Ok(()))
}

// checker-gfp.16 design -- the Gröbner checks run generators, cofactor rows, S-pairs, and the first flaw ends the check (not measured)
fn groebner<S: Side>(
    side: &mut S,
    index: &mut Divisors,
    ring: &Ring,
    lines: &[usize],
    cofactors: &str,
    mode: Mode,
    poll: &mut Poll,
) -> Result<Stage, Error> {
    // ⟨F⟩ ⊆ ⟨G⟩: every generator reduces to zero modulo G.
    for j in 0..side.generators() {
        if poll.step() {
            return Ok(None);
        }
        let zero = side
            .generator(j, index, poll)
            .map_err(|u| overflow(lines, u))?;
        if poll.stopped() {
            return Ok(None);
        }
        if !zero {
            return Ok(Some(Err(Flaw::Generator(j))));
        }
    }

    // ⟨G⟩ ⊆ ⟨F⟩: the row of every nonzero element of G expresses it in F.
    match rows(side, ring, cofactors, mode, None, poll)? {
        Some(Ok(())) => {}
        stage @ (Some(Err(_)) | None) => return Ok(stage),
    }

    pairs(side, index, lines, poll)
}

/// The rows stage: the row of every nonzero element in the cofactor matrix
/// expresses it in F, or the first flaw. A zero element's row is read and
/// its shape checked. The cheap mode draws its points in batches of `per`,
/// or of as many as fit its budget where `per` is None.
fn rows<S: Side>(
    side: &mut S,
    ring: &Ring,
    cofactors: &str,
    mode: Mode,
    per: Option<NonZeroUsize>,
    poll: &mut Poll,
) -> Result<Stage, Error> {
    match mode {
        Mode::Exact => exact_rows(side, ring, cofactors, poll),
        Mode::Cheap { points, seed } => {
            cheap::rows(side, ring, cofactors, points.get(), seed, per, poll)
        }
    }
}

fn exact_rows<S: Side>(
    side: &mut S,
    ring: &Ring,
    cofactors: &str,
    poll: &mut Poll,
) -> Result<Stage, Error> {
    let mut rows = Rows::<S::Numbers>::new(cofactors)?;
    let mut entries = Vec::new();
    let mut count = 0;
    while let Some((line, k)) = rows.next(ring, side.numbers(), &mut entries, poll)? {
        if poll.step() {
            return Ok(None);
        }
        let i = count;
        count += 1;
        let flaw = exact_row(side, i, line, &entries[..k], poll)?;
        if poll.stopped() {
            return Ok(None);
        }
        if let Some(flaw) = flaw {
            return Ok(Some(Err(flaw)));
        }
    }
    if count != side.len() {
        return Ok(Some(Err(Flaw::Shape)));
    }
    Ok(Some(Ok(())))
}

/// Row i, read as `entries` from `line` on, ruled on exactly: the flaw it
/// shows, or None. Where `poll` stops, the answer means nothing.
pub fn exact_row<S: Side>(
    side: &mut S,
    i: usize,
    line: usize,
    entries: &[Read<S>],
    poll: &mut Poll,
) -> Result<Option<Flaw>, Error> {
    // Rows past the last element are read, and counted after the last row.
    if i >= side.len() {
        return Ok(None);
    }
    if entries.len() != side.generators() {
        return Ok(Some(Flaw::Shape));
    }
    if side.monos(i).is_empty() {
        return Ok(None);
    }
    let expressed = side.row(i, entries, poll).map_err(|_| Error::Overflow {
        part: Part::Cofactors,
        line,
    })?;
    Ok((!expressed).then_some(Flaw::Cofactors(i)))
}

// checker-gfp.9 cited: buchberger-2006 Lemma 5.13, via buchberger-1985 Theorem 6.2 -- a basis is a Gröbner basis exactly when the S-polynomials of all pairs reduce to zero
/// G is a Gröbner basis: every S-polynomial the criteria do not excuse
/// reduces to zero.
fn pairs<S: Side>(
    side: &mut S,
    index: &mut Divisors,
    lines: &[usize],
    poll: &mut Poll,
) -> Result<Stage, Error> {
    let live = index.elements.clone();
    let mut lcm = Vec::new();
    // checker-gfp.24 design -- the pairs are visited in the order of their positions in G, i then j (not measured)
    for a in 0..live.len() {
        for b in a + 1..live.len() {
            if poll.step() {
                return Ok(None);
            }
            let (i, j) = (live[a], live[b]);
            let (lt_i, lt_j) = (index.lt(a), index.lt(b));
            // checker-gfp.10 cited: buchberger-2006 Lemma 5.11, via buchberger-1985 Lemma 6.4 -- when the lcm of the leading power products is their product, the S-polynomial reduces to zero
            if index.masks[a] & index.masks[b] == 0 && poly::coprime(lt_i, lt_j) {
                continue;
            }
            lcm.clear();
            poly::lcm_into(lt_i, lt_j, &mut lcm).map_err(|_| Error::Overflow {
                part: Part::Basis,
                line: lines[j],
            })?;
            // checker-stop-latency.7 design -- a divisor query the memo does not answer counts 1 + n/32 units before its scan, n the divmasks of the index, the chain criterion 1 + n/32 a pair, the memo's growth a unit per monomial it moves, a block of 4096 at a time, and an answer from the memo nothing, not a unit per query or the growth counted at once (log 0059-1)
            if poll.work(1 + live.len() / MASKS) {
                return Ok(None);
            }
            if index.chain(a, b, &lcm) {
                continue;
            }
            let zero = side
                .spair(i, j, &lcm, index, poll)
                .map_err(|u| overflow(lines, u))?;
            if poll.stopped() {
                return Ok(None);
            }
            if !zero {
                return Ok(Some(Err(Flaw::SPair(i, j))));
            }
        }
    }
    Ok(Some(Ok(())))
}

// checker-orders.12 cited: bigatti-2025 §1.1, Remark 5 -- xy >_Lex z^1000: Lex is not degree compatible
/// A multiple of element u of G past `u32`, in a reduction step or an
/// S-polynomial. In an order that is not degree compatible a tail term can
/// have a larger degree than its leading term.
fn overflow(lines: &[usize], u: usize) -> Error {
    Error::Overflow {
        part: Part::Basis,
        line: lines[u],
    }
}
