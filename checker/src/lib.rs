//! The checker: rules on a result, an instance with its basis and cofactors.
//! It shares no code with the rest of grobx (`OBJECTIVES.md`, _Checker_).
//!
//! Over GF(p) and over Q, in the monomial order the call names, the checker
//! gives two verdicts. G is a Gröbner basis of ⟨F⟩ when every generator of F
//! reduces to zero modulo G, the row of every nonzero element of G in the
//! cofactor matrix expresses it in F, and every
//! S-polynomial the criteria do not excuse reduces to zero modulo G. G is
//! reduced when every element is nonzero and scaled as the canonical basis text, and no term of an
//! element is divisible by the leading term of another.
//!
//! Over Q every reduction is exact. An element is scaled over GF(p) when it
//! is monic, and over Q when its coefficients are integers with no common
//! factor and its leading coefficient is positive. Leading terms are taken in
//! the order the call names.

mod cheap;
mod check;
mod divisors;
mod field;
mod gfp;
mod heap;
mod narrow;
mod order;
mod parse;
mod poll;
mod poly;
mod q;
mod rational;

use std::num::{NonZeroU32, NonZeroUsize};

// checker-gfp.2 design -- the checker takes the instance, the basis, and the cofactors as text (not measured)
// checker-orders.1 design -- the order is a parameter of check, check_polled, and check_cofactors, which every call states, not functions of their own beside three that keep grevlex (not measured)
/// Rules on the basis `basis` of the instance `instance`, with the cofactor
/// matrix `cofactors`: row i holds, for every generator j of the instance,
/// the cofactor of generator j in element i.
///
/// The instance is in msolve's input format, the basis a bracketed list of
/// polynomials closed by `:`, and the matrix a bracketed list of such rows.
/// Element i is the i-th polynomial of the basis text, generator j the j-th
/// of the instance, both counted from 0.
///
/// The instance's characteristic chooses the field: 0 is Q, and a prime p
/// below 2^64 is GF(p). Over Q every number is read exactly, save a
/// cofactor row's in the cheap mode, which it reduces modulo each point's
/// prime where it does not check the row exactly.
///
/// The instance is read as `grobx::groebner` reads it: what that call
/// refuses, `check` refuses with an error of the same kind, save more
/// generators or monomials than a u32 id numbers, which `check` reads.
///
/// Leading terms, reductions, and the scale of an element are taken in the
/// monomial order `order`. A matrix that is not an order on the instance's
/// variables is refused with `Error::Order` once the variables and the
/// characteristic are read.
///
/// # Errors
///
/// An `Error` where the instance or the basis text does not read, or the
/// cofactor text as far as the check reads it: the rows' stage runs once
/// every generator reduces to zero, and reads up to the first flaw. An
/// `Error` too where `order` is
/// not an order on the instance's variables, or `Error::Overflow` where a
/// product the check forms passes `u32`: a multiple in a reduction or an
/// S-polynomial, an lcm of two leading monomials, or a cofactor's term
/// times a generator's.
#[allow(
    clippy::expect_used,
    clippy::missing_panics_doc,
    reason = "the expect cannot fail: a poll that never answers true never stops the check"
)]
pub fn check(
    instance: &str,
    basis: &str,
    cofactors: &str,
    order: &Order,
    mode: Mode,
) -> Result<Verdict, Error> {
    let verdict = check::check(
        instance,
        basis,
        cofactors,
        order,
        mode,
        &mut poll::Poll::new(&mut || false),
    )?;
    Ok(verdict.expect("a poll that never stops does not stop the check"))
}

// f4-gfp.26 practice: cooperative cancellation -- the checker calls a poll between its steps and stops where it answers true, not run on a thread of its own that is abandoned at the deadline
// checker-stop-latency.1 design -- the check counts its work in units, a unit about the work of one term, and calls its poll once per 4096 units beside its poll per element, generator, row, pair, and point, not per 4096 reduction steps whatever a step's work (log 0059-1)
/// As `check`, and `poll` is called as the check's work goes: at least once per
/// element of G in the reduced verdict, per generator, per row of the
/// cofactor matrix, per pair of nonzero elements i < j, per point of the cheap mode,
/// and once per 4096 units of work. A unit is about the work of one term:
/// a term read, a term written into a sum over GF(p), a product the sum
/// over Q takes, a monomial evaluated at a point. Over Q a product, a
/// quotient, and a gcd of integers count as many units as their limbs ask,
/// save the gcd of an S-pair's two leading coefficients and that of a
/// polynomial's content and denominator, and the quotients by these and by
/// a term's gcd.
/// Work is counted before it is done, save a term read, counted after it.
/// In the cheap mode a term of a cofactor row evaluated at a point counts
/// as a unit, and so do 19 digits of the row's numerators, and
/// over Q of its denominators, reduced modulo a point's modulus, and a
/// denominator's inverse 8 units more; over GF(p) a number is reduced once
/// for all of a batch's points. A term or a number is counted at all of
/// the points it is evaluated at before it is evaluated.
/// Where it returns true, the check stops, the
/// result is `Ok(None)`, and `poll` is not called again. A reading error
/// in the term the stop falls in is still returned, save in a cofactor row
/// the cheap mode evaluates, and one past that term is not reached.
///
/// # Errors
///
/// As `check`.
pub fn check_polled(
    instance: &str,
    basis: &str,
    cofactors: &str,
    order: &Order,
    mode: Mode,
    poll: &mut dyn FnMut() -> bool,
) -> Result<Option<Verdict>, Error> {
    let poll = &mut poll::Poll::new(poll);
    let verdict = check::check(instance, basis, cofactors, order, mode, poll);
    debug_assert!(
        !matches!(verdict, Ok(None)) || poll.stopped(),
        "the check stopped where the poll did not"
    );
    verdict
}

/// Rules on the ⟨G⟩ ⊆ ⟨F⟩ direction alone, in the mode `mode`: whether the
/// row of every nonzero element of G in the cofactor matrix expresses it in
/// F. It reads the
/// three texts as `check` reads them, refuses what `check` refuses in
/// reading them, and runs no other check.
///
/// The result is `Ok(())` where every such row expresses its element, else the
/// first `Shape` or `Cofactors(i)`. Where the generators reduce to zero,
/// this is the flaw `check` reports in the same mode, or `Ok(())` where
/// `check` reports an `SPair` or none.
///
/// # Errors
///
/// As `check`.
pub fn check_cofactors(
    instance: &str,
    basis: &str,
    cofactors: &str,
    order: &Order,
    mode: Mode,
) -> Result<Result<(), Flaw>, Error> {
    check::check_cofactors(instance, basis, cofactors, order, mode, None)
}

// checker-points-abort.6 design -- the test code reaches a batch size and the drawn moduli through two public functions hidden from the documentation, not through unit tests inside the crate on hand-built texts alone (not measured)
/// As `check_cofactors` in the cheap mode at `points` points drawn from
/// `seed`, with `per` points to a batch. For the test code.
#[doc(hidden)]
pub fn check_cofactors_batched(
    instance: &str,
    basis: &str,
    cofactors: &str,
    order: &Order,
    points: NonZeroU32,
    seed: u64,
    per: NonZeroUsize,
) -> Result<Result<(), Flaw>, Error> {
    let mode = Mode::Cheap { points, seed };
    check::check_cofactors(instance, basis, cofactors, order, mode, Some(per))
}

// checker-points-abort.6 design -- the test code reaches a batch size and the drawn moduli through two public functions hidden from the documentation, not through unit tests inside the crate on hand-built texts alone (not measured)
/// The modulus of each of the `points` points the cheap mode draws from
/// `seed` for the instance and the basis, in order. For the test code.
#[doc(hidden)]
pub fn cheap_moduli(
    instance: &str,
    basis: &str,
    order: &Order,
    points: NonZeroU32,
    seed: u64,
) -> Result<Vec<u64>, Error> {
    check::cheap_moduli(instance, basis, order, points.get(), seed)
}

// instance-grammar-agreement.7 design -- the test code compares the generators through a public function hidden from the documentation, the counterpart of `grobx::text::generators`, not the checker's verdict on the engine's result alone or a comparison inside the checker with a list the engine wrote (not measured)
/// The instance's nonzero generators over GF(p), each monic with its terms
/// in decreasing grevlex order, in the instance's order, as a bracketed
/// list in the canonical basis text's spelling. For the test code: the
/// counterpart of the engine's `grobx::text::generators`.
/// `Error::Characteristic` unless the characteristic is a prime below 2^64:
/// it reads no instance over Q.
#[doc(hidden)]
pub fn generators(instance: &str) -> Result<String, Error> {
    let header = parse::header(instance, &Order::Grevlex)?;
    let p = header.ring.characteristic;
    if p == 0 {
        return Err(Error::Characteristic { line: 2 });
    }
    let field = field::Field::new(p);
    let polys = parse::generators(&header, &field, &mut poll::Poll::new(&mut || false))?;
    Ok(gfp::text(field, &header.ring, &polys))
}

// checker-orders.2 cited: robbiano-1985 Theorems 4 and 5, via bachmann-1998 §2.1 -- every term ordering is given by orthogonal u_1, …, u_s ∈ R^n, s ≤ n: v > w ⇔ (v·u_1, …, v·u_s) >_lex (w·u_1, …, w·u_s)
/// A monomial order on the instance's variables, numbered in the order the
/// instance lists them, the first numbered 0.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Order {
    /// a > b where deg a > deg b, or the degrees are equal and the last
    /// variable whose exponents differ has the smaller exponent in a.
    Grevlex,
    /// a > b where the first variable whose exponents differ has the larger
    /// exponent in a.
    Lex,
    /// The order of the n × n integer matrix M whose rows these are: a > b
    /// where the first nonzero entry of M·(a − b) is positive. Column j
    /// belongs to variable j. M is nonsingular, and the first nonzero entry
    /// of every column is positive. Block and elimination orders are given
    /// this way.
    Matrix(Vec<Vec<i64>>),
}

/// How the ⟨G⟩ ⊆ ⟨F⟩ direction is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Mode {
    /// The row of every nonzero gᵢ is expanded: Σⱼ Cᵢⱼ·fⱼ − gᵢ is zero.
    Exact,
    // checker-cheap.1 design -- the cheap mode is a variant of Mode that carries k and the seed, not a seed parameter of check beside the mode (not measured)
    // checker-cheap.2 design -- k is a NonZeroU32, not a u32 with an error for k = 0 (not measured)
    /// The row of every nonzero gᵢ is checked at `points` random points
    /// (`OBJECTIVES.md`, _Checker_), and every random choice is drawn from `seed`. The
    /// generators, the S-pairs, and the reduced verdict are checked as in
    /// `Exact`. The same arguments give the same result.
    ///
    /// The points are drawn in batches, and the cofactor matrix is read
    /// once per batch, at the batch's points. A batch holds as many points
    /// as fit 16 MiB, and at least one. A point counts with its
    /// coordinates, its power table, the values of F and G at it, a row's
    /// values at it, and a coefficient's residue. The
    /// result is that of one batch of all the points.
    ///
    /// A true row passes at every point. A row's degree bound d is the
    /// largest of deg gᵢ and, over the nonzero fⱼ whose entry Cᵢⱼ has a
    /// nonzero term, deg fⱼ plus the largest degree of the entry's nonzero
    /// terms as written. A row with d past `u32` is checked exactly.
    ///
    /// Over GF(p), p prime, every coordinate of a point is uniform in GF(p).
    /// A false row with d < p passes k points with probability at most
    /// (d/p)^k. A row with d ≥ p is checked exactly.
    ///
    /// Over Q every point has a prime q of its own, uniform among the
    /// N = 98 182 656 primes in [2^31, 2^32), and its coordinates are uniform
    /// modulo q. Let u/v be a nonzero coefficient of Σⱼ Cᵢⱼ·fⱼ − gᵢ in lowest
    /// terms, and b the bit length of u. A false row passes one point with
    /// probability at most b/(31·N) + d/2^31, and k points with at most the
    /// k-th power of that. A point whose prime divides a denominator of F or
    /// G is drawn again, which lowers N by at most the bit length of those
    /// denominators over 31. A row with d ≥ 2^31 is checked exactly. A row
    /// with a denominator that the prime of a
    /// point divides is checked exactly, save a denominator that follows a
    /// factor 0 of its term, which adds nothing.
    Cheap { points: NonZeroU32, seed: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// Whether G is a Gröbner basis of ⟨F⟩, or the first flaw found.
    pub groebner: Result<(), Flaw>,
    /// Whether G is reduced, or the first flaw found.
    pub reduced: Result<(), Flaw>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Flaw {
    /// Generator j does not reduce to zero modulo G.
    Generator(usize),
    /// The cofactor matrix does not have one row per element of G, or a row
    /// does not have one entry per generator.
    Shape,
    /// Row i does not express element i in F.
    Cofactors(usize),
    /// The S-polynomial of elements i < j does not reduce to zero modulo G.
    SPair(usize, usize),
    /// Element i is zero or not scaled as the canonical basis text: over
    /// GF(p) monic, over Q with integer coefficients that have no common
    /// factor and a positive leading coefficient, leading in the order the
    /// call names.
    Scale(usize),
    /// A term of `element` is divisible by the leading term of `by`.
    Divisible { element: usize, by: usize },
}

/// The text an error was found in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Instance,
    Basis,
    Cofactors,
}

/// Why the checker cannot rule. `line` counts from 1.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The text does not follow the grammar: `expected` should stand at `line`.
    Syntax {
        part: Part,
        line: usize,
        expected: &'static str,
    },
    /// A name the instance does not list.
    UnknownVariable {
        part: Part,
        line: usize,
        name: String,
    },
    /// A name the instance lists twice.
    RepeatedVariable { line: usize, name: String },
    /// The characteristic is neither 0 nor a prime below 2^64, or it is 0
    /// where `generators` reads it.
    Characteristic { line: usize },
    /// A term of the instance whose degree passes 2^31 − 1, or an exponent
    /// or a degree past `u32` in the basis, the cofactors, or a product.
    Overflow { part: Part, line: usize },
    /// A denominator has no inverse: it is 0 mod p, or over Q it is 0.
    NoInverse { part: Part, line: usize },
    /// The order is a matrix that is not n × n for the instance's n
    /// variables, is singular, or has a column whose first nonzero entry is
    /// negative.
    Order,
}
