//! The lift over Q: a call over Q runs the engine over primes, one run at
//! a time. A learn run over two primes below 2^31 at once records the rows
//! that counted at the first, and returns the reduced basis at the second,
//! the verification's prime. Replays of the record at further primes, of
//! [2^61, 2^62), return the reduced basis there, and its cofactors where
//! the call tracks them. Each open coefficient keeps its image at every
//! prime, and an attempt converts its images to its residue modulo the
//! product of the primes by the Chinese remainder theorem, in their mixed
//! radix, and reconstructs it as a rational, element by element. Once a
//! coefficient of an element has fitted in an attempt, a further
//! coefficient of a part that holds a rational is first converted and
//! reconstructed modulo the product of the first primes its part needs. A
//! kept element is neither checked nor combined at the primes after its
//! keep, and its images there are not read. A replay that tracks no
//! cofactors, of a record FGLM did not change, runs over two primes of
//! [2^61, 2^62) at once, and its image holds each term once with its
//! coefficients at both: the lift merges it once, absorbs the first prime
//! without an attempt, and attempts at the second, at the next turn of its
//! loop. Once every element is reconstructed, the learn
//! run's basis at its second prime verifies the result, which is written as
//! the canonical basis text over Q. A learn run whose two primes disagree,
//! or whose read meets a denominator 0 at either, falls back to its first
//! prime alone, and its verification is a run at a fresh prime, as it is
//! where the second prime divides a denominator of the lifted basis. Where
//! the verification finds an element that differs, the lift learns afresh.
//! Where the lift has dropped `DROPS` records in the call, at misfits or at
//! verifications alike, the call returns `Error::Drops`.
//! Where the call asks for the call report, the lift writes what became of
//! each prime, in the order of use, and the lines of its texts and checker.
//! A pair's second prime whose image the lift never takes, where the first
//! ends the lift or drops the record, stands on no line of its own: on the
//! pair's `run` line, and on its `pair` line where the pair fell back.

use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};
use std::fmt;

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::ENTRY;
use crate::entry::{Imaged, certify, image_in, report_check, report_text, stopped};
#[cfg(test)]
use crate::entry::{report_call, report_end, report_read};
use crate::field::{Field, Gf31, Gf31Pair, Gf64, Gf64Pair, is_prime_64, mul_mod, shoup_factor};
#[cfg(test)]
use crate::lehmer::Wang;
use crate::lehmer::{First, Mqrr, Scaled, gcd};
use crate::monomial::{Dense32, Format};
use crate::narrow::Narrow;
use crate::order::{Block, Grevlex, Lex, MonomialOrder, Order};
use crate::rational::{
    Converted, Crt, CrtStep, Radix, Rational, residue, scale_images, term_bound_q,
    write_cofactor_q, write_poly_q,
};
use crate::report::{Opt, Secs, Stage, on};
use crate::shape::Kept;
use crate::text::{Header, decimal_units, limbs, reservation};
use crate::trace::{NoLearner, Record, Recording, Trace};
use crate::{Error, Outcome, Params, Stats, Stop};

// The lift's matches on `Imaged`, and its pair learn's `[Image; 2]`, take
// these lanes: `image_in` makes an image per lane for a learn run over a
// field of more than one, and a pair image for a replay over `Gf64Pair`.
const _: () = assert!(
    Gf31::LANES == 1 && Gf64::LANES == 1 && Gf31Pair::LANES == 2 && Gf64Pair::LANES == 2,
    "the lift's fields have the lanes its matches take"
);

/// A polynomial as the lift reads it: its terms' exponent vectors, n per
/// term, in decreasing order of the call's order, and their coefficients
/// in [0, p), 0 only where `merge` gave an image a term it lacked or where
/// a pair image's term vanishes at its first prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sparse {
    pub(crate) exponents: Vec<u32>,
    pub(crate) coefs: Vec<u64>,
}

// q-lift.29 design -- a run's result reaches the lift as an image, exponent vectors and residues read from the run's tables, not the run's canonical basis text and cofactor text read back (not measured)
/// One run's result over one prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Image {
    /// The leading monomials of the reduced basis, by increasing leading
    /// monomial, n exponents each.
    pub(crate) leads: Vec<u32>,
    /// The reduced basis, each element monic, by increasing leading
    /// monomial, without its leading term.
    pub(crate) basis: Vec<Sparse>,
    /// Per element, per generator of the instance, the cofactor; empty
    /// where the run tracked none.
    pub(crate) cofactors: Vec<Vec<Sparse>>,
    pub(crate) stats: Stats,
}

// q-lift-rest-time.1 cited: dekleine-2001 §2 -- a polynomial mapped to Z_p1 × … × Z_pb[x] is stored as its monomials, each with its coefficients at the b primes; §1 -- the polynomial operations are done simultaneously on b primes to amortize the monomial cost
/// A pair replay's reduced basis at its two primes, without cofactors:
/// per element its terms but the leading one, once, with their
/// coefficients at the first prime, a coefficient 0 where the residue
/// there is 0, and their coefficients at the second beside.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PairImage {
    /// As `Image::leads`: at both primes.
    pub(crate) leads: Vec<u32>,
    /// Per element, its terms and their coefficients at the first prime.
    pub(crate) first: Vec<Sparse>,
    /// Per element, the coefficients of `first`'s terms at the second
    /// prime.
    pub(crate) second: Vec<Vec<u64>>,
    pub(crate) stats: Stats,
}

// q-lift.6 cited: berthomieu-2021 §4.5 -- picking randomly a prime number p₀ in the interval ]2^30, 2^31[; idrees-2011 §2 -- pTestSB randomly chooses a prime number not used in the previous computations
// q-lift.7 practice: random prime by incremental search -- a prime is the least prime at or above a uniform draw from its range, [2^30, 2^31) or [2^61, 2^62), drawn again where none lies below the range's top or it was used, the draws from Xoshiro256PlusPlus seeded by seed_from_u64, not a uniform draw among the primes by odd numbers drawn until one is prime
// q-lift.36 design -- the lift takes a crate-internal list of primes that come first, and reports what it did with each prime, not unlucky primes met by drawn primes alone (not measured)
/// Where the lift's primes come from: the forced primes first, in order,
/// then draws.
pub(crate) struct Primes {
    rng: Xoshiro256PlusPlus,
    /// The forced primes not yet taken, the next one last.
    forced: Vec<u64>,
    /// Every prime taken.
    used: BTreeSet<u64>,
}

impl Primes {
    /// Draws from `Xoshiro256PlusPlus` seeded by `seed_from_u64(seed)`;
    /// `forced` are primes below 2^62, and those `next` takes below 2^31.
    pub(crate) fn new(seed: u64, mut forced: Vec<u64>) -> Primes {
        forced.reverse();
        Primes {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            forced,
            used: BTreeSet::new(),
        }
    }

    // q-lift.5 cited: berthomieu-2021 §4.5 -- one starts by picking randomly a prime number p₀ in the interval ]2^30, 2^31[; demin-2024 §4.3 -- in the multi-modular runs we use 31-bit primes
    // q-lift.8 design -- the lift tests a candidate prime by gfp-large-primes.10's strong probable-prime test and builds its field without testing it again, not Gf31::new's trial division at each candidate and each run (log 0016-2)
    // q-zero-denominator.4 design -- next draws until an unused prime turns up, with no bound, as only a prime that divides a nonzero denominator, of the text or at a verification of the lifted basis, is skipped, not an error, None, or a budget check once every prime of the range is used (not measured)
    // q-modulus-time.2 design -- the learn run and the verification's run keep their primes from [2^30, 2^31), in Gf31Pair or Gf31, as their images are not combined, not every run of the lift at a prime from [2^61, 2^62) (log 0036-1)
    /// The next forced prime, else the least prime at or above a uniform
    /// draw from [2^30, 2^31), drawn again where there is none below 2^31
    /// or it was used before: the prime of a learn run and of a
    /// verification. A forced prime here lies below 2^31.
    pub(crate) fn next(&mut self) -> u64 {
        if let Some(p) = self.forced.pop() {
            debug_assert!(p < 1 << 31 && is_prime_64(p), "{p} is a prime below 2^31");
            self.used.insert(p);
            return p;
        }
        loop {
            // The top 30 of 64 bits, past 2^30. No even number past 2 is
            // prime.
            let draw = (1 << 30) | (self.rng.next_u64() >> 34);
            let mut n = draw | 1;
            while n < 1 << 31 && !is_prime_64(n) {
                n += 2;
            }
            if n < 1 << 31 && self.used.insert(n) {
                return n;
            }
        }
    }

    // q-modulus-time.1 cited: pearce-2026 §2 -- F4 modulo a machine prime p < 2^63, with Chinese remaindering and rational reconstruction expected to compute bases over ℚ; §3, Table 2 -- the cost of 63-bit primes is reasonable, 1.00 to 1.77 times the time at p = 2^31 − 1
    // q-modulus-time.4 cited: harvey-2014 §3 -- the redundant representation requires the stronger condition p < β/4
    /// The next forced prime, else the least prime at or above a uniform
    /// draw from [2^61, 2^62), drawn again where there is none below 2^62
    /// or it was used before: the prime of a replay. A forced prime here
    /// lies below 2^62.
    pub(crate) fn wide(&mut self) -> u64 {
        if let Some(p) = self.forced.pop() {
            debug_assert!(p < 1 << 62 && is_prime_64(p), "{p} is a prime below 2^62");
            self.used.insert(p);
            return p;
        }
        loop {
            // The top 61 of 64 bits, past 2^61.
            let draw = (1 << 61) | (self.rng.next_u64() >> 3);
            let mut n = draw | 1;
            while n < 1 << 62 && !is_prime_64(n) {
                n += 2;
            }
            if n < 1 << 62 && self.used.insert(n) {
                return n;
            }
        }
    }

    /// The generator's next 64 bits (q-lift.34).
    pub(crate) fn draw(&mut self) -> u64 {
        self.rng.next_u64()
    }
}

/// What the lift did with each prime, in order of use (q-lift.36), and
/// counts of its work. A pair replay's second prime is in no list where the
/// lift never takes its image: its image goes unused, or its replay alone
/// never runs. A pair learn's second prime is in `unpaired` where the pair
/// fell back, in `skipped` or `verified` once the verification reaches it,
/// and in no list where the record is dropped before.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Report {
    /// The primes of the learn runs that made a record.
    pub(crate) learned: Vec<u64>,
    /// The primes whose replay was combined.
    pub(crate) combined: Vec<u64>,
    /// The primes whose replay did not fit.
    pub(crate) misfit: Vec<u64>,
    /// The primes skipped: a denominator of the input, or of the lifted
    /// basis at a verification, is 0 mod p.
    pub(crate) skipped: Vec<u64>,
    /// The verification primes, each with whether the lifted basis
    /// passed there.
    pub(crate) verified: Vec<(u64, bool)>,
    // q-learn-time.12 design -- the report lists the pair learns that fell back, each (p, q), not the fallbacks counted in misfit or skipped (not measured)
    /// The pair learns that fell back to their first prime alone, each as
    /// (p, q): their lanes disagreed, or a denominator of the input is 0
    /// at p or at q.
    pub(crate) unpaired: Vec<(u64, u64)>,
    // q-reconstruct-time.11 design -- the report counts every rational reconstruction the lift runs, so that the test code bounds the work exactly, not the test code timing the call (not measured)
    /// Every rational reconstruction the lift ran, one per coefficient
    /// tried, whatever it returned, by the scaled residue, by MQRR, or by
    /// Wang's reconstruction.
    pub(crate) reconstructed: u64,
    // q-rational-time.8 design -- the report counts the coefficients tried that ran the Euclidean algorithm, and a test switch makes a lift run Wang's reconstruction on every coefficient, so that the test code compares the two lifts exactly, not the test code timing the call or counts frozen in it (not measured)
    /// The coefficients tried that ran the Euclidean algorithm: MQRR's first
    /// pass under `Reconstruct::Chain`, and Wang's reconstruction under
    /// `Reconstruct::Wang`.
    pub(crate) euclid: u64,
    // q-replay-time.12 design -- Kept counts the replays that ran from a shape and the shapes taken, and the report carries the counts, not the test code timing the call (not measured)
    // q-later-runs-time.10 design -- the report counts a pair replay once in shaped and in reshaped, which count runs, not a count per prime (not measured)
    /// The replays that ran from a kept shape (`Kept::used`), a pair
    /// replay once.
    pub(crate) shaped: u64,
    /// The replays that kept a shape (`Kept::taken`), a pair replay once.
    pub(crate) reshaped: u64,
    // q-crt-time.9 design -- the report counts the images the lift appends, one per open coefficient and prime, not the test code timing the call (not measured)
    /// Every image appended: one per open coefficient and prime.
    pub(crate) steps: u64,
}

// q-rational-time.8 design -- the report counts the coefficients tried that ran the Euclidean algorithm, and a test switch makes a lift run Wang's reconstruction on every coefficient, so that the test code compares the two lifts exactly, not the test code timing the call or counts frozen in it (not measured)
/// How an attempt reconstructs a coefficient. The test code compares the
/// lift over the chain with the lift of `Wang::reconstruct` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reconstruct {
    /// The scaled residue over the part's chain denominator, or over the
    /// previous element's at a part's first coefficient at every other
    /// prime, and MQRR where it is not small.
    Chain,
    /// `Wang::reconstruct` on every coefficient.
    #[cfg(test)]
    Wang,
}

// q-later-runs-time.9 design -- the lift takes how its replays run, in pairs or one prime at a time, as a crate-internal parameter beside Reconstruct, the call running pairs and the test code comparing the two lifts, not the test code timing the call or counts frozen in it (not measured)
/// How the lift runs a replay that tracks no cofactors, of a record FGLM
/// did not change: two primes at once, or one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Replays {
    /// Two primes of [2^61, 2^62) at once, in `Gf64Pair`.
    Pairs,
    /// One prime at a time.
    #[cfg(test)]
    Alone,
}

// q-learn-time.11 design -- the lift takes how it learns as a crate-internal parameter beside Replays, the call learning in pairs and the test code of earlier items, through lift_text and lift_text_with, alone, not the forced primes of the earlier test code reordered for the pair (not measured)
/// How the lift learns its record: over two primes at once, the second
/// the verification's, or at one prime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Learns {
    /// The learn prime and the verification's prime of [2^30, 2^31) at
    /// once, in `Gf31Pair`.
    Pairs,
    /// One prime, and the verification's run at a fresh prime.
    #[cfg(test)]
    Alone,
}

/// What the lift takes next at its replays' primes.
enum Next {
    /// The second prime of a pair that did not fit, whose text has a
    /// denominator 0 at either prime, or whose primes are not both of
    /// [2^61, 2^62): it replays alone.
    Alone(u64),
    /// The second prime of a pair image the lift merged and absorbed at its
    /// first: per element, its coefficients there, aligned with its part's
    /// terms, none for an element kept before.
    Second(u64, Vec<Vec<u64>>),
    /// The second prime of a pair image whose leading monomials are not the
    /// learn run's: a misfit there too.
    Leads(u64),
}

/// What a turn of the lift's loop absorbs at its prime.
enum Absorb {
    /// A replay's image at the prime alone.
    Image(Image),
    /// A pair image, absorbed at its first prime; its second, q, waits.
    First(PairImage, u64),
    /// A pair's coefficients at its second prime (`Next::Second`).
    Second(Vec<Vec<u64>>),
}

impl Absorb {
    /// The image's elements, and the parts of each: the basis polynomial,
    /// and a cofactor per generator where the replay tracked them. A pair
    /// replay tracks none.
    fn shape(&self) -> (usize, usize) {
        match self {
            Absorb::Image(image) => (
                image.basis.len(),
                1 + image.cofactors.first().map_or(0, Vec::len),
            ),
            Absorb::First(image, _) => (image.first.len(), 1),
            Absorb::Second(coefs) => (coefs.len(), 1),
        }
    }
}

/// The record the replays replay, and what its learn run returned.
struct Learned {
    record: Record,
    /// The leading monomials of its basis, n exponents each.
    leads: Vec<u32>,
    stats: Stats,
}

impl Learned {
    fn new(record: Record, image: Image) -> Learned {
        Learned {
            record,
            leads: image.leads,
            stats: image.stats,
        }
    }
}

// lift-misfit-loop.4 design -- DROPS is 4: the tuning set's calls over Q dropped no record (log 0069-1, _Step 3_), so four drops at random primes are improbable, not 2 or 3, the error sooner under a defect, nor 8 or more (log 0069-1)
/// The records a call's lift drops before it returns `Error::Drops`.
const DROPS: u32 = 4;

/// The misfits in a row that drop a record (q-lift.20).
const MISFITS: u32 = 2;

/// Why the lift dropped a record, as the call report's `drop` line names
/// it.
#[derive(Clone, Copy)]
enum Why {
    /// Two misfits in a row (q-lift.20).
    Misfits,
    /// A verification whose leading monomials differ (q-later-runs-time.8).
    Leads,
    /// A verification at which an element differs (result-time.16).
    Verification,
}

impl Why {
    /// The word of the `drop` line.
    fn name(self) -> &'static str {
        match self {
            Why::Misfits => "misfits",
            Why::Leads => "leads",
            Why::Verification => "verification",
        }
    }
}

// lift-misfit-loop.2 design -- every drop counts, after two misfits in a row and at a verification whose leading monomials or an element differ, and a skip at a denominator or a pair learn that falls back drops nothing, not the drops at misfits alone nor every learn run (not measured)
// lift-misfit-loop.3 design -- the count is the call's and never resets, not the drops in a row of records that combined no image, nor a count reset where a replay's image is combined (not measured)
/// The records the lift dropped in the call, by why.
#[derive(Default)]
struct Drops {
    misfits: u32,
    leads: u32,
    verification: u32,
}

impl Drops {
    /// One drop more for `why`: `Error::Drops` where the drops reach
    /// `DROPS`.
    fn count(&mut self, why: Why) -> Result<(), Error> {
        let Drops {
            misfits,
            leads,
            verification,
        } = self;
        debug_assert!(
            *misfits + *leads + *verification < DROPS,
            "the call returned at the drop that reached DROPS"
        );
        match why {
            Why::Misfits => *misfits += 1,
            Why::Leads => *leads += 1,
            Why::Verification => *verification += 1,
        }
        if *misfits + *leads + *verification < DROPS {
            return Ok(());
        }
        Err(Error::Drops {
            misfits: *misfits,
            leads: *leads,
            verification: *verification,
        })
    }
}

// q-lift.27 design -- the lift computes one run at a time on one thread, a pair replay one run over two primes, whatever threads allows, as f4-gfp.30 does, not the primes on several threads at once (not measured)
/// The outcome of a call over Q: the lift of the instance `text`. `named`
/// and `header` are the call's, read; every prime's run takes `budget` and
/// tells `trace` its matrices. `primes` gives the primes, and `report`
/// hears what the lift did with each. An attempt reconstructs as `how`
/// says, a replay that tracks no cofactors, of a record FGLM did not
/// change, runs as `replays` says, and a learn run as `learns` says. A
/// replay runs from the shape the record's last full replay kept, until the
/// record is dropped or the verification runs. The call report gets the
/// `lift` line first, and after each prime's run what became of the prime,
/// save a pair's second prime whose image it never takes.
#[allow(
    clippy::too_many_arguments,
    reason = "the instance, its parameters, order, and header, the budget, trace, primes, and report the lift borrows, and its choices of the reconstruction, the replays, and the learn runs, none of which belongs with another"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the loop over the primes in one piece, until a split of it is measured"
)]
pub(crate) fn lift<T: Trace>(
    text: &str,
    params: &Params,
    named: &MonomialOrder,
    header: &Header,
    budget: &mut Budget,
    trace: &mut T,
    primes: &mut Primes,
    how: Reconstruct,
    replays: Replays,
    learns: Learns,
    report: &mut Report,
) -> Result<Outcome, Error> {
    let n = header.names.len();
    let with_cofactors = params.cofactors || params.checker.is_some();
    // q-lift.12 design -- the learn runs track no cofactors, and the replays track them where the call asks for cofactors or the checker, not learn runs with cofactors (log 0014-3, 0016-1)
    // q-lift.28 design -- every prime's run takes the call's seed, not a seed per prime from the lift's generator (not measured)
    // input-redundancy-time.8 design -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
    let echelon = params.echelon && !with_cofactors;
    let learn = Params {
        cofactors: false,
        checker: None,
        echelon,
        ..params.clone()
    };
    let replay = Params {
        cofactors: with_cofactors,
        checker: None,
        echelon,
        ..params.clone()
    };
    let mut learned: Option<Learned> = None;
    // The pair learn's image at its second prime, with that prime, until
    // the verification takes it.
    let mut second: Option<(u64, Image)> = None;
    let mut lifting: Option<Lifting> = None;
    let mut misfits = 0;
    // q-replay-time.8 design -- the lift keeps one shape, with its record, dropped where the record is dropped or replaced and given back by the full check before each prime where the resident set passes the ceiling, not a shape kept past its record or held through a memory stop (not measured)
    let mut kept = Kept::default();
    // q-later-runs-time.5 design -- the second prime's image of a pair waits and is taken at the next turn of the loop, after the budget's check, as a replay's image is, and a record dropped drops it, not both images taken at once before the test of completion (log 0039-1)
    let mut pending: VecDeque<Next> = VecDeque::new();
    // q-later-runs-time.6 design -- a replay that tracks cofactors, and the replay of a record FGLM changed, run one prime at a time, not pairs there too with the cofactor tracking's rows and images per lane and FGLM's matrices over the pair (not measured)
    let pairs = match replays {
        Replays::Pairs => !with_cofactors,
        #[cfg(test)]
        Replays::Alone => false,
    };
    let learn_pairs = match learns {
        Learns::Pairs => true,
        #[cfg(test)]
        Learns::Alone => false,
    };
    let runs = |two: bool| if two { "pairs" } else { "alone" };
    // call-report.24 design -- the lift line's replays names how the lift runs the replays of a record FGLM did not change, in pairs where the call's parameter asks for pairs and no cofactors are tracked, not the parameter as passed (not measured)
    line(
        budget,
        "lift",
        format_args!(
            " learns={learns} replays={replays} track={track}",
            learns = runs(learn_pairs),
            replays = runs(pairs),
            track = on(with_cofactors)
        ),
    );
    // lift-misfit-loop.1 design -- the lift counts the records it drops in the call, and the drop that brings the count to DROPS returns Error::Drops after its entries in the report and its drop line, not the budget bounding the learn runs alone (q-lift.26, withdrawn) (log 0068-1)
    let mut drops = Drops::default();
    'turns: loop {
        // q-lift.33 design -- the budget is checked in full before each prime, and the work on the images between two primes ticks it (q-lift.42), not a check between primes alone (not measured)
        // q-replay-time.8 design -- the full check before each prime gives the shape back where the resident set passes the ceiling, and the next replay builds it again (not measured)
        if let Err(why) = budget.check(|| kept.clear()) {
            let stats = learned.as_ref().map_or(Stats::default(), |l| l.stats);
            return Ok(stopped(why, stats));
        }
        let Some(current) = &learned else {
            let p = primes.next();
            if learn_pairs {
                // q-learn-time.9 design -- the pair's second prime is the next prime of the lift's Primes, drawn right after the first, forced primes included, not a generator of its own for the verification's primes (not measured)
                let q = primes.next();
                let mut recording = Recording::new();
                // q-learn-time.1 cited: graebe-1993 §4, via demin-2026 §4.2 -- a bimodular arithmetic adds almost nothing to the computing time of the single modular case and ends with high probability at two lucky primes; dekleine-2001 §1 and §3, independently, via demin-2026 §4.2 -- computing over Z_p1 × … × Z_pb amortizes the monomial cost over the arithmetic of the primes
                // q-lift.31 design -- a call over Q tells its trace every matrix of every run in turn, not the learn run's matrices alone (not measured)
                let got = image_in(
                    Gf31Pair::primes(p, q),
                    text,
                    &learn,
                    None,
                    named,
                    header,
                    budget,
                    trace,
                    &mut recording,
                    None,
                );
                match got {
                    // q-zero-denominator.2 design -- a read's NoInverse at the line of the text's first denominator 0 is the call's error, not a reader that knows the instance is over Q; at any other line the pair falls back (q-learn-time.6) (not measured)
                    Err(Error::NoInverse { line }) if header.zero_denominator() == Some(line) => {
                        return Err(Error::NoInverse { line });
                    }
                    // q-learn-time.6 design -- a pair learn that halts on a disagreement, or whose read meets a denominator 0 at p or q on a line other than the text's first denominator 0, falls back to the learn run at p alone, and the verification's run at a fresh prime follows the lift, not both primes dropped and a new pair learned, nor the prime whose lane was 0 dropped and the learn run at the other (not measured)
                    Err(Error::Misfit { .. }) => {
                        report_unpaired(budget, report, (p, q), "disagree");
                    }
                    Err(Error::NoInverse { .. }) => {
                        report_unpaired(budget, report, (p, q), "denominator");
                    }
                    Err(e) => return Err(e),
                    Ok(Imaged::Stopped(outcome)) => return Ok(outcome),
                    Ok(Imaged::Lanes(images)) => {
                        #[allow(
                            clippy::expect_used,
                            reason = "`image_in` makes an image per lane, and `Gf31Pair` has two as the assert at the top checks"
                        )]
                        let [first, image]: [Image; 2] =
                            images.try_into().expect("an image per prime of the pair");
                        learned = Some(Learned::new(recording.finish(named.clone()), first));
                        second = Some((q, image));
                        report_learn(budget, Lanes(p, Some(q)), "recorded", None);
                        report.learned.push(p);
                        misfits = 0;
                        continue;
                    }
                    #[allow(
                        clippy::unreachable,
                        reason = "`image_in` makes an image per lane for a learn run over `Gf31Pair`, of two lanes as the assert at the top checks"
                    )]
                    Ok(Imaged::Image(_) | Imaged::Pair(_)) => {
                        unreachable!("the pair learn at ({p}, {q}) made no image per lane")
                    }
                }
            }
            let mut recording = Recording::new();
            // q-lift.4 cited: berthomieu-2021 §4.3 and §4.5 -- learn from the first computation modulo p₀ with exact linear algebra, and apply the trace learnt from p₀ for the other primes; parisse-2013 §3, independently -- learning from the first prime discards for the next primes the s-polys that reduced to 0; traverso-1989 §The Gröbner trace modular algorithm, steps (2) and (7), via demin-2024 §4.4, independently -- the Gröbner basis modulo p₀ is computed with its trace, and the basis modulo every other prime using the trace
            // q-lift.31 design -- a call over Q tells its trace every matrix of every run in turn, not the learn run's matrices alone (not measured)
            let got = image_in(
                Gf31::prime(p),
                text,
                &learn,
                None,
                named,
                header,
                budget,
                trace,
                &mut recording,
                None,
            );
            match got {
                // q-zero-denominator.2 design -- a read's NoInverse at the line of the text's first denominator 0 is the call's error, and one at any other line skips the prime, not a reader that knows the instance is over Q (not measured)
                // q-zero-denominator.3 design -- the text is scanned for a denominator 0 only after a prime's read returned NoInverse, not before the first prime (not measured)
                // A replay and the verification read a text this read took
                // whole, which holds no denominator 0, so they keep their skips.
                Err(Error::NoInverse { line }) if header.zero_denominator() == Some(line) => {
                    return Err(Error::NoInverse { line });
                }
                // q-lift.21 cited: boehm-2015 §3 -- a bad prime of type-1, where the reduction of the input is not defined, e.g. at the prime factors of a denominator, is detected and rejected from the beginning; idrees-2011 §1 -- I_p is written for a p that divides no denominator of the coefficients
                Err(Error::NoInverse { .. }) => {
                    report_learn(budget, Lanes(p, None), "skipped", Some("denominator"));
                    report.skipped.push(p);
                }
                Err(e) => return Err(e),
                Ok(Imaged::Stopped(outcome)) => return Ok(outcome),
                Ok(Imaged::Image(image)) => {
                    learned = Some(Learned::new(recording.finish(named.clone()), image));
                    report_learn(budget, Lanes(p, None), "recorded", None);
                    report.learned.push(p);
                    misfits = 0;
                }
                #[allow(
                    clippy::unreachable,
                    reason = "`image_in` makes one image over `Gf31`, of one lane as the assert at the top checks"
                )]
                Ok(Imaged::Lanes(_) | Imaged::Pair(_)) => {
                    unreachable!("the learn run at {p} made more than one image")
                }
            }
            continue;
        };
        // q-lift.4 cited: berthomieu-2021 §4.3 and §4.5 -- learn from the first computation modulo p₀ with exact linear algebra, and apply the trace learnt from p₀ for the other primes
        // q-lift.31 design -- a call over Q tells its trace every matrix of every run in turn, not the learn run's matrices alone (not measured)
        let record = &current.record;
        let alone = |p: u64, budget: &mut Budget, trace: &mut T, kept: &mut Kept| {
            replay_at(p, text, &replay, record, named, header, budget, trace, kept)
        };
        // What the turn absorbs at p, or why p is a misfit: the replay did
        // not fit the record, or its leading monomials are not the learn
        // run's.
        let (p, image) = 'took: {
            let (p, got) = match pending.pop_front() {
                // q-lift-rest-time.2 cited: dekleine-2001 §3.2 -- if the digits of agreement do not suffice, we get another batch of primes and continue: the reconstruction is tried once a batch; demin-2024 §4.3, independently -- the modular images are computed in batches and the reconstruction attempted after each
                Some(Next::Second(q, coefs)) => break 'took (q, Ok(Absorb::Second(coefs))),
                Some(Next::Leads(q)) => break 'took (q, Err("leads")),
                Some(Next::Alone(p)) => (p, alone(p, budget, trace, &mut kept)),
                None if pairs && !record.changed() => 'pair: {
                    let (p, q) = (primes.wide(), primes.wide());
                    // q-later-runs-time.3 design -- the pair's primes both lie in [2^61, 2^62), each lane computing as Gf64 below 2^62, not pairs of any two primes with each lane in the row operation of its range or in Gf31 (not measured)
                    let wide = |r: u64| (1 << 61..1 << 62).contains(&r);
                    let both = (wide(p) && wide(q)).then(|| {
                        replay_pair_at(
                            p, q, text, &replay, record, named, header, budget, trace, &mut kept,
                        )
                    });
                    let why = match both {
                        Some(Ok(Imaged::Pair(image))) if image.leads == current.leads => {
                            break 'took (p, Ok(Absorb::First(image, q)));
                        }
                        // A pair's leading monomials are those of both its
                        // primes.
                        Some(Ok(Imaged::Pair(_))) => {
                            pending.push_back(Next::Leads(q));
                            break 'took (p, Err("leads"));
                        }
                        // q-later-runs-time.4 design -- a pair fits only where every leading coefficient the replay inverts, of a pivot of the generators' echelon form or, where the record took none, of a nonzero generator, and every counted row's leading entry is a unit at both primes, and where it does not, or a denominator of the text is 0 at either prime, each prime replays alone, the second after the lift has taken the first's image, not the pair a misfit at both primes nor the lane that fits taken from the pair's run (not measured)
                        None => "range",
                        Some(Err(Error::Misfit { .. })) => "misfit",
                        Some(Err(Error::NoInverse { .. })) => "denominator",
                        Some(other) => break 'pair (p, other),
                    };
                    line(
                        budget,
                        "pair",
                        format_args!(
                            " primes={lanes} outcome=alone why={why}",
                            lanes = Lanes(p, Some(q))
                        ),
                    );
                    pending.push_back(Next::Alone(q));
                    (p, alone(p, budget, trace, &mut kept))
                }
                None => {
                    let p = primes.wide();
                    (p, alone(p, budget, trace, &mut kept))
                }
            };
            let image = match got {
                // q-lift.21 cited: boehm-2015 §3 -- a bad prime of type-1, where the reduction of the input is not defined, e.g. at the prime factors of a denominator, is detected and rejected from the beginning
                Err(Error::NoInverse { .. }) => {
                    line(budget, "skip", format_args!(" prime={p} why=denominator"));
                    report.skipped.push(p);
                    continue 'turns;
                }
                Err(Error::Misfit { .. }) => Err("record"),
                Err(e) => return Err(e),
                Ok(Imaged::Stopped(outcome)) => return Ok(outcome),
                Ok(Imaged::Image(image)) if image.leads == current.leads => {
                    Ok(Absorb::Image(image))
                }
                Ok(Imaged::Image(_)) => Err("leads"),
                #[allow(
                    clippy::unreachable,
                    reason = "`alone` replays over `Gf31` or `Gf64`, of one lane as the assert at the top checks, and a pair's image is taken above"
                )]
                Ok(Imaged::Lanes(_) | Imaged::Pair(_)) => {
                    unreachable!("the replay at {p} made more than one image")
                }
            };
            (p, image)
        };
        // A replay that skips its prime stops at its read, before it uses or
        // keeps a shape.
        report.shaped = kept.used;
        report.reshaped = kept.taken;
        let image = match image {
            Ok(image) => image,
            Err(why) => {
                // q-lift.19 cited: boehm-2015 §3 -- for a bad prime of type-2 the computation in characteristic p cannot be carried through; detected before the Chinese remainder step, it does not influence the final lifting
                line(budget, "misfit", format_args!(" prime={p} why={why}"));
                report.misfit.push(p);
                misfits += 1;
                // q-lift.20 design -- two misfits in a row drop the record and what its replays lifted, and the next prime learns afresh, not a new learn run at the first misfit nor never (not measured)
                if misfits == MISFITS {
                    (learned, lifting, misfits) = (None, None, 0);
                    kept.clear();
                    pending.clear();
                    // q-learn-time.8 design -- the second lane's image is kept beside the lift from the learn until the verification, and dropped wherever the record is dropped, not only its leading monomials kept and the verification's run at q once the lift is complete (log 0052-1)
                    second = None;
                    drop_record(budget, &mut drops, Why::Misfits)?;
                }
                continue;
            }
        };
        misfits = 0;
        report.combined.push(p);
        let lifted =
            lifting.get_or_insert_with(|| Lifting::new(&current.leads, n, image.shape(), how));
        // The open elements and the attempts before the absorb, where the
        // call report is on.
        let before = budget
            .report()
            .map(|_| (lifted.open(), report.reconstructed));
        budget.open();
        // q-lift.10 design -- each replay's image is combined once the replay returns, one prime at a time, not images computed in batches and combined together (not measured)
        // q-lift.11 design -- the replays' images alone are combined, not the learn run's basis as the first image (not measured)
        let absorbed = match image {
            Absorb::Image(image) => lifted.absorb(parts(image), p, named, n, budget, report),
            // q-lift-rest-time.2 cited: dekleine-2001 §3.2 -- if the digits of agreement do not suffice, we get another batch of primes and continue: the reconstruction is tried once a batch; demin-2024 §4.3, independently -- the modular images are computed in batches and the reconstruction attempted after each
            Absorb::First(image, q) => lifted
                .absorb_first(image, p, named, n, budget, report)
                .map(|coefs| pending.push_back(Next::Second(q, coefs))),
            Absorb::Second(coefs) => lifted.absorb_second(&coefs, p, budget, report),
        };
        if let Err(why) = absorbed {
            return Ok(stopped(why, current.stats));
        }
        if let Some((open, attempts)) = before {
            report_combine(budget, p, lifted, open, report.reconstructed - attempts);
        }
        let Some(lifted) = lifted.complete() else {
            continue;
        };
        // q-replay-time.15 design -- the shape is dropped before the verification's run, whose matrices would stand beside it, not the shape kept through the verification (log 0022-3)
        kept.clear();
        // q-lift.22 cited: idrees-2011 §2 -- pTestSB, at a random prime p not used before, is positive iff (G mod p) is a standard basis of I_p; demin-2024 §4.3 -- a relaxed randomized check: the recovered basis is a Gröbner basis modulo a randomly chosen prime; berthomieu-2021 §4.5 -- until a solution whose image by reduction to some prime p coincides with the output there
        let (q, image) = loop {
            // q-learn-time.7 design -- the verification takes the second lane's image where q divides no denominator of the lifted basis, and otherwise skips q and runs at a fresh prime, not the lift learning afresh where q divides a denominator (not measured)
            // The verification takes the image, so a record dropped after it
            // leaves none behind (q-learn-time.8).
            if let Some((q, image)) = second.take() {
                // q-lift.25 cited: arnold-2003 Theorem 5.12 -- if p is a lucky prime, then p does not divide d(G)
                if !lifted.divides_a_denominator(q) {
                    break (q, image);
                }
                line(
                    budget,
                    "skip",
                    format_args!(" prime={q} why=lifted-denominator"),
                );
                report.skipped.push(q);
            }
            let q = primes.next();
            // q-lift.25 cited: arnold-2003 Theorem 5.12 -- if p is a lucky prime, then p does not divide d(G)
            if lifted.divides_a_denominator(q) {
                line(
                    budget,
                    "skip",
                    format_args!(" prime={q} why=lifted-denominator"),
                );
                report.skipped.push(q);
                continue;
            }
            // q-later-runs-time.7 design -- the verification's run at its fresh prime records nothing, a run with no learner apart from the record whose plain run combines rows where a call over GF(q) would, not a learn run at q (q-lift.23, withdrawn), nor a replay of the record; idrees-2011 §2's pTestSB computes the standard basis of I_p at a fresh prime and is silent on a trace (log 0037-1, 0039-1)
            // q-lift.31 design -- a call over Q tells its trace every matrix of every run in turn, not the learn run's matrices alone (not measured)
            let got = image_in(
                Gf31::prime(q),
                text,
                &learn,
                None,
                named,
                header,
                budget,
                trace,
                &mut NoLearner,
                None,
            );
            match got {
                // q-lift.21 cited: boehm-2015 §3 -- a bad prime of type-1, where the reduction of the input is not defined, e.g. at the prime factors of a denominator, is detected and rejected from the beginning
                Err(Error::NoInverse { .. }) => {
                    line(budget, "skip", format_args!(" prime={q} why=denominator"));
                    report.skipped.push(q);
                }
                Err(e) => return Err(e),
                Ok(Imaged::Stopped(outcome)) => return Ok(outcome),
                Ok(Imaged::Image(image)) => break (q, image),
                #[allow(
                    clippy::unreachable,
                    reason = "`image_in` makes one image over `Gf31`, of one lane as the assert at the top checks"
                )]
                Ok(Imaged::Lanes(_) | Imaged::Pair(_)) => {
                    unreachable!("the run at {q} made more than one image")
                }
            }
        };
        // q-later-runs-time.8 cited: parisse-2013 §4, Modular algorithm, step 3 -- where a prime's output has other leading terms than the outputs reconstructed so far, clearing all learning data is probably a good idea; where only coefficients differ, result-time.16
        if image.leads != current.leads {
            report_verify(budget, q, "leads-differ", false);
            report.verified.push((q, false));
            drop_record(budget, &mut drops, Why::Leads)?;
            (learned, lifting, misfits) = (None, None, 0);
            pending.clear();
            continue;
        }
        budget.open();
        let passed = lifted.verify(&image, q, n);
        budget.mark(Stage::Verify);
        report_verify(budget, q, if passed { "passed" } else { "failed" }, true);
        report.verified.push((q, passed));
        // result-time.16 design -- a verification at which an element differs drops the record and the lifting, and the next prime learns afresh, as one whose leading monomials differ does, not the elements that differ reopened by a catch-up and the lift going on at the next prime (not measured)
        if !passed {
            drop_record(budget, &mut drops, Why::Verification)?;
            (learned, lifting, misfits) = (None, None, 0);
            pending.clear();
            continue;
        }
        // q-lift.32 design -- a call over Q's stats are those of the learn run whose record the result comes from, not sums over every run (not measured)
        let stats = current.stats;
        return Ok(finish(
            lifted,
            text,
            params,
            named,
            &header.names,
            stats,
            budget,
            primes,
        ));
    }
}

// call-report.14 design -- the lift writes a line per prime whose image or skip it takes, in the order of use, with what became of it and why, not the lists of lift::Report at the end (not measured)
/// The call report's line of `kind` with `fields`, where `budget` holds
/// the report.
fn line(budget: &mut Budget, kind: &str, fields: fmt::Arguments<'_>) {
    if let Some(report) = budget.report() {
        report.line(kind, fields);
    }
}

/// A prime, or a pair's two primes joined by `,`, as the call report
/// writes them.
#[derive(Clone, Copy)]
struct Lanes(u64, Option<u64>);

impl fmt::Display for Lanes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.1 {
            Some(q) => write!(f, "{p},{q}", p = self.0),
            None => write!(f, "{p}", p = self.0),
        }
    }
}

/// The call report's `learn` line: the learn at `primes` came to
/// `outcome`, `recorded`, `unpaired`, or `skipped`, for `why`.
fn report_learn(budget: &mut Budget, primes: Lanes, outcome: &str, why: Option<&str>) {
    line(
        budget,
        "learn",
        format_args!(
            " primes={primes} outcome={outcome}{why}",
            why = Opt("why", why)
        ),
    );
}

/// The pair (p, q) learns no record together, for `why`: its `learn` line,
/// and its place among the report's unpaired pairs.
fn report_unpaired(budget: &mut Budget, report: &mut Report, (p, q): (u64, u64), why: &str) {
    report_learn(budget, Lanes(p, Some(q)), "unpaired", Some(why));
    report.unpaired.push((p, q));
}

/// The call report's `drop` line for `why`, and the drop counted in
/// `drops`: `Error::Drops` where it brings them to `DROPS`.
fn drop_record(budget: &mut Budget, drops: &mut Drops, why: Why) -> Result<(), Error> {
    let name = why.name();
    line(budget, "drop", format_args!(" why={name}"));
    drops.count(why)
}

// call-report.15 cited: monagan-2015 §2.6, Table 4 -- the time over ℚ broken into the Gröbner bases modulo p, the Chinese remaindering, and the rational reconstruction; parisse-2019 §2 -- the lift is a loop of three steps, a basis modulo a prime, Chinese remaindering, and reconstruction attempts
// q-lift-growth-time.11 design -- the combine line keeps its fields, crt timing the step and the append and reconstruct the attempts with their conversions, not a field of the conversion's own (not measured)
/// The call report's `combine` line: the image at p absorbed into
/// `lifting`, whose elements were `open` before, after `attempts`
/// reconstructions, with the times of its merge, of its attempts with
/// their conversions, and of its step and append.
fn report_combine(budget: &mut Budget, p: u64, lifting: &Lifting, open: usize, attempts: u64) {
    let Some(report) = budget.report() else {
        return;
    };
    let primes = lifting.crt.primes().len();
    let kept = lifting.elements.len() - lifting.open();
    let [merge, reconstruct, crt] =
        [Stage::Merge, Stage::Reconstruct, Stage::Crt].map(|s| Secs(report.time(s)));
    report.line(
        "combine",
        format_args!(
            " prime={p} primes={primes} open={open} kept={kept} attempts={attempts} \
             merge={merge} reconstruct={reconstruct} crt={crt}"
        ),
    );
}

/// The call report's `verify` line: the lifted basis at q came to
/// `outcome`, with the time of `Lifting::verify` where `timed`.
fn report_verify(budget: &mut Budget, q: u64, outcome: &str, timed: bool) {
    let Some(report) = budget.report() else {
        return;
    };
    let time = timed.then(|| Secs(report.time(Stage::Verify)));
    report.line(
        "verify",
        format_args!(
            " prime={q} outcome={outcome}{time}",
            time = Opt("time", time)
        ),
    );
}

// q-modulus-time.9 design -- a replay's field follows its prime, Gf31 below 2^31 and Gf64 from 2^31, so that a forced prime below 2^31 replays in Gf31, not every replay in Gf64 nor forced primes past 2^31 alone for the replays (not measured)
/// The replay of `record` at p, in `Gf31` where p < 2^31 and in `Gf64`
/// from 2^31, as `image_in` runs it, from the shape in `kept` or keeping
/// one there where it tracks no cofactors.
#[allow(
    clippy::too_many_arguments,
    reason = "the prime, the instance, its parameters, record, order, and header, and the budget, trace, and kept shape the replay borrows, none of which belongs with another"
)]
fn replay_at<T: Trace>(
    p: u64,
    text: &str,
    params: &Params,
    record: &Record,
    named: &MonomialOrder,
    header: &Header,
    budget: &mut Budget,
    trace: &mut T,
    kept: &mut Kept,
) -> Result<Imaged, Error> {
    let (record, learner, kept) = (Some(record), &mut NoLearner, Some(kept));
    if p < 1 << 31 {
        let field = Gf31::prime(p);
        image_in(
            field, text, params, record, named, header, budget, trace, learner, kept,
        )
    } else {
        let field = Gf64::prime(p);
        image_in(
            field, text, params, record, named, header, budget, trace, learner, kept,
        )
    }
}

/// The replay of `record` at p and q at once, in `Gf64Pair`, as
/// `image_in` runs it, from the shape in `kept` or keeping one there.
#[allow(
    clippy::too_many_arguments,
    reason = "the two primes, the instance, its parameters, record, order, and header, and the budget, trace, and kept shape the replay borrows, none of which belongs with another"
)]
fn replay_pair_at<T: Trace>(
    p: u64,
    q: u64,
    text: &str,
    params: &Params,
    record: &Record,
    named: &MonomialOrder,
    header: &Header,
    budget: &mut Budget,
    trace: &mut T,
    kept: &mut Kept,
) -> Result<Imaged, Error> {
    image_in(
        Gf64Pair::primes(p, q),
        text,
        params,
        Some(record),
        named,
        header,
        budget,
        trace,
        &mut NoLearner,
        Some(kept),
    )
}

/// An image's elements as the lift reads them: per element its parts,
/// part 0 the basis polynomial but its leading term, part j generator j's
/// cofactor, j from 1.
fn parts(image: Image) -> Vec<Vec<Sparse>> {
    let mut cofactors = image.cofactors.into_iter();
    image
        .basis
        .into_iter()
        .map(|g| {
            let mut parts = vec![g];
            parts.extend(cofactors.next().unwrap_or_default());
            parts
        })
        .collect()
}

/// What the replays of one record lifted.
struct Lifting {
    crt: Crt,
    elements: Vec<Element>,
    how: Reconstruct,
    /// The most limbs of a numerator or a chain denominator any element
    /// has held: every radix that reduces them has as many powers. One
    /// that an attempt makes past them takes `Radix::residue_big`'s Horner
    /// rule.
    width: usize,
    scratch: Scratch,
    /// The absorbs that attempted (q-lift-rest-time.3).
    attempts: u64,
    prefixes: Prefixes,
}

// rule-breaches.6 design -- the verification and the texts take a view of a lifting that keeps every element, which `Lifting::complete` alone makes, not a variant of `Error` for a lifting with an open element, nor a debug assertion per element (not measured)
/// A lifting with a prime combined that keeps every element. Only
/// `Lifting::complete` makes one.
#[derive(Clone, Copy)]
struct Complete<'a>(&'a Lifting);

// q-lift-rest-time.8 design -- a prefix holds a multiple of PREFIX, 4, primes and at most k − 4 of M's k, and the lifting keeps the MQRR of each prefix it took, made once from the product of its primes, not a prefix of any length with its MQRR made at each coefficient (log 0088-1)
/// The primes a prefix of M grows by, and the fewest of M's primes past a
/// prefix.
const PREFIX: usize = 4;

// q-lift-rest-time.7 tuned -- a prefix holds SLACK, 128 bits, past its part's need, so that a coefficient larger than the ones before by up to two primes still fits there, not no slack nor one prime's 61 bits (log 0088-1)
/// The bits past a part's need a prefix holds.
const SLACK: u64 = 128;

// q-lift-rest-time.5 invented -- once a coefficient of an element fitted in an attempt, a further coefficient of a part that holds a rational is converted from its images at the first j primes and reconstructed modulo their product, j from the part's need, and taken modulo M where it does not fit there, not every coefficient modulo M, nor the prefix from the attempt's first coefficient on, which converts every failing probe twice (log 0088-1)
/// The prefixes of M's primes an attempt reconstructs a coefficient
/// modulo. The lifting's primes only grow, so a prefix's product never
/// changes.
#[derive(Default)]
struct Prefixes {
    // q-lift-rest-time.9 design -- a prefix's bits are bounded below by the sum of L(p) − 1 over its primes, kept as primes join, as a prime of L bits is at least 2^(L−1), not the product's exact length made per prefix nor a sum of log₂ in floating point (not measured)
    /// At j − 1, for the first j of M's primes, Σ (L(pᵢ) − 1) over them: a
    /// lower bound on the bits of their product. Extended as primes join.
    bits: Vec<u64>,
    /// At j/`PREFIX` − 1, MQRR modulo the product of the first j primes,
    /// made at its first use.
    mqrr: Vec<Option<Mqrr>>,
}

impl Prefixes {
    /// The fewest of `primes`, M's, j a multiple of `PREFIX`, whose bound
    /// in `bits` reaches `need`, where j + `PREFIX` ≤ the primes of M; None
    /// elsewhere.
    fn length(&mut self, primes: &[u64], need: u64) -> Option<usize> {
        for &p in &primes[self.bits.len()..] {
            let before = self.bits.last().copied().unwrap_or(0);
            self.bits.push(before + u64::from(p.bit_width()) - 1);
        }
        let j = (self.bits.partition_point(|&b| b < need) + 1).next_multiple_of(PREFIX);
        (j + PREFIX <= primes.len()).then_some(j)
    }

    /// MQRR modulo the product of `primes[..j]`, j a multiple of `PREFIX`
    /// that `length` gave for `primes`, M's.
    fn mqrr(&mut self, primes: &[u64], j: usize) -> &Mqrr {
        debug_assert!(
            j > 0 && j.is_multiple_of(PREFIX) && j <= primes.len(),
            "a prefix of {j} of {len} primes",
            len = primes.len()
        );
        let at = j / PREFIX - 1;
        if self.mqrr.len() <= at {
            self.mqrr.resize_with(at + 1, || None);
        }
        self.mqrr[at].get_or_insert_with(|| {
            let product = primes[..j].iter().fold(BigUint::one(), |m, &p| m * p);
            Mqrr::new(&product)
        })
    }
}

/// An element's coefficients at one prime, per part, aligned with the
/// part's terms: what an attempt checks and the append appends.
trait Aligned {
    /// Part j's coefficients.
    fn part(&self, j: usize) -> &[u64];
}

/// A replay alone's element, merged: its parts.
impl Aligned for Vec<Sparse> {
    fn part(&self, j: usize) -> &[u64] {
        &self[j].coefs
    }
}

/// A pair image's element at its first prime, merged: one part.
impl Aligned for Sparse {
    fn part(&self, j: usize) -> &[u64] {
        debug_assert_eq!(j, 0, "a pair image's element has one part");
        &self.coefs
    }
}

/// A pair image's element at its second prime, aligned: one part.
impl Aligned for Vec<u64> {
    fn part(&self, j: usize) -> &[u64] {
        debug_assert_eq!(j, 0, "a pair image's element has one part");
        self
    }
}

// q-lift-growth-time.9 cited: monagan-2007 §2.2 -- buffers reused rather than freed amortize the cost of memory allocation
/// The attempts' buffers, kept from attempt to attempt.
#[derive(Default)]
struct Scratch {
    /// The images of u·l.
    scaled: Vec<u64>,
    /// The mixed-radix digits of the conversion.
    digits: Vec<u64>,
    // q-crt-time.6 design -- a residue lies on 64-bit limbs of the lift's own, as lehmer.rs holds a number, not num-bigint's BigUint (log 0023-1)
    /// The limbs the conversion's form names.
    limbs: Vec<u64>,
}

/// A conversion of `Crt`: `Crt::convert` or `Crt::convert_small`.
type Conversion = fn(&Crt, &[u64], &mut Vec<u64>, &mut Vec<u64>) -> (Converted, usize);

impl Scratch {
    /// The conversion of the images of u·l at the primes of `crt`'s M by
    /// `conversion`: `images` u's, and `by` l's residues at those primes
    /// and their Shoup factors, or None for l = 1. Its form, with its limbs
    /// in `limbs`, and whether it ended below the top digit. The
    /// conversion's multiply-adds are spent after it; a stop where the
    /// budget's check stops the call.
    fn convert(
        &mut self,
        images: &[u64],
        by: Option<(&[u64], &[u64])>,
        crt: &Crt,
        conversion: Conversion,
        budget: &mut Budget,
    ) -> Result<(Converted, bool), Stop> {
        let v = match by {
            Some((residues, factors)) => {
                scale_images(images, residues, factors, crt.primes(), &mut self.scaled);
                &self.scaled[..]
            }
            None => images,
        };
        let (form, products) = conversion(crt, v, &mut self.digits, &mut self.limbs);
        let ended = self.digits.len() < v.len();
        // q-lift-growth-time.10 design -- the append spends LIMB units a word it writes, its moves included, COMBINED coefficients at a time, and an attempt spends its conversion's multiply-adds after the conversion, not the combination's LIMB·(w′ + 4) a coefficient (cofactor-stop-latency.6), nor conversions that tick alone (not measured)
        budget.spend(products)?;
        Ok((form, ended))
    }
}

// q-lift.17 design -- an element is its basis polynomial and, where tracked, its row of cofactors, reconstructed, cleared, and kept as one, not the cofactors lifted apart once the basis is kept (log 0016-1)
/// An element of the basis over Q: its basis polynomial and, where the
/// call tracks cofactors, its cofactor row.
struct Element {
    /// The leading monomial's exponents; its coefficient is 1.
    lead: Vec<u32>,
    /// Part 0: the basis polynomial's terms but the leading one. Part j:
    /// generator j's cofactor, j from 1.
    parts: Vec<Part>,
    /// Per part, the rationals of its first terms, in order. They form a
    /// prefix over the parts: a part holds rationals only where every part
    /// before it holds one per term. Each fit the image at the prime after
    /// the modulus it was reconstructed from.
    rationals: Vec<Rationals>,
    /// Whether every part holds one rational per term, and they all fit
    /// the image at the prime where the last of them was reconstructed.
    /// The element is open where not. Only the verification checks a kept
    /// element again.
    kept: bool,
}

/// The terms of a part, in decreasing order, and while its element is
/// open the images of each coefficient at the lifting's primes, all in one
/// vector. A kept element's parts hold no image.
pub(crate) struct Part {
    exponents: Vec<u32>,
    // q-lift-growth-time.1 design -- each open coefficient keeps its image at every prime combined, one word a prime, and no residue modulo M is formed before an attempt asks for one, not the residue modulo M made by a Chinese remainder step of every open coefficient at every prime (log 0083-1)
    // cofactor-stop-latency.7 design -- a part's images lie in one vector, each term's on the words of a capacity, not a vector per term (log 0058-1)
    /// Per term, its images at the lifting's first `count` primes, in
    /// their order: term t's at `images[t·cap..t·cap + count]`.
    images: Vec<u64>,
    /// The words of a term: a power of two, at least 1 and `count`.
    cap: usize,
    /// The primes combined into the part.
    count: usize,
}

impl Part {
    /// No term.
    pub(crate) fn new() -> Part {
        Part {
            exponents: Vec::new(),
            images: Vec::new(),
            cap: 1,
            count: 0,
        }
    }

    /// The terms whose images are held: every term while the element is
    /// open.
    pub(crate) fn len(&self) -> usize {
        self.images.len() / self.cap
    }

    /// Each term's images, at the first `count` primes.
    pub(crate) fn images(&self) -> impl Iterator<Item = &[u64]> {
        let count = self.count;
        self.images
            .chunks_exact(self.cap)
            .map(move |term| &term[..count])
    }
}

// q-rational-time.4 design -- a part keeps its rationals as numerators over a chain of denominators, 1 and then each the lcm of the one before and a denominator an attempt found, a numerator over the chain's last at its reconstruction and unreduced, not each rational in lowest terms with a denominator of its own (log 0028-1)
/// A part's rationals: numerators over a chain of denominators. The
/// chain's last is the lcm of the denominators of the part's rationals in
/// lowest terms.
#[derive(Clone)]
struct Rationals {
    /// The chain: dens[0] = 1, and each next the lcm of the one before
    /// and the denominator of a rational in lowest terms an attempt found
    /// for the part. Each divides the next.
    dens: Vec<BigUint>,
    /// Per term, in order, its numerator and the index of its denominator
    /// in `dens`: the rational num / dens[den], not in lowest terms. The
    /// indices do not fall from one term to the next.
    terms: Vec<(BigInt, u32)>,
    // q-lift-growth-time.5 design -- the residues of a part's chain's last at the lifting's primes, and their Shoup factors, are kept with the part's rationals, taken once per chain denominator and prime, and the unit test of the chain's last reads them, not the residues taken at each coefficient's attempt (not measured)
    /// The last denominator's residues at the lifting's first primes, in
    /// their order: those `extend` has taken, and those a growth by the
    /// denominator MQRR found carried from the last before (`grow`). A
    /// growth by the lcm empties them. None while the chain is [1].
    residues: Vec<u64>,
    /// Each residue's Shoup factor, `shoup_factor(residue, p)`.
    factors: Vec<u64>,
    /// Whether the last denominator is a unit modulo the product of the
    /// primes `coprime` was last given. Where it is not, the attempt
    /// scales the part's images by 1 until the part is cleared.
    unit: bool,
    /// The most limbs of a numerator or a denominator here.
    width: usize,
    // q-lift-rest-time.6 design -- a part's need is the bits its second coefficient asked for, then the largest over its coefficients from the second on, as the part's first coefficient reconstructs the element's denominator and the later ones a small factor of it beside their numerators, not the last coefficient's need alone, which fails where a larger coefficient follows, nor every coefficient's, the first's among them, which asks for about twice (log 0088-1)
    /// The bits a modulus needs for the part's next coefficient
    /// (`note_need`): 0 while the part holds no rational.
    need: u64,
}

impl Rationals {
    /// No term, and the chain [1].
    fn new() -> Rationals {
        Rationals {
            dens: vec![BigUint::one()],
            terms: Vec::new(),
            residues: Vec::new(),
            factors: Vec::new(),
            unit: true,
            width: 1,
            need: 0,
        }
    }

    /// No term, and the chain [1].
    fn clear(&mut self) {
        self.dens.truncate(1);
        self.terms.clear();
        self.residues.clear();
        self.factors.clear();
        self.unit = true;
        self.width = 1;
        self.need = 0;
    }

    /// The need after the fit of the part's last term, t its index, `chain`
    /// the chain's length and its last's bits before the fit, and T MQRR's
    /// threshold modulo M: the bits the term asked for, L(n′) + L(f) + T +
    /// 2 for its numerator n′ over the chain's last and the factor f the
    /// chain grew by, where t ≤ 1, and the larger of that and the need
    /// before past t = 1. L(f) is L(last after) − L(last before) + 1 where
    /// the chain grew, else 0. By monagan-2004 §3, Lemma 1, the quotient
    /// after n′/f exceeds m/(3·|n′|·f), so a modulus of these bits finds it.
    fn note_need(&mut self, (dens, bits): (usize, u64), t: u64) {
        let numerator = self.terms.last().map_or(0, |(num, _)| num.bits());
        let factor = if self.dens.len() > dens {
            self.last().bits() - bits + 1
        } else {
            0
        };
        let asked = numerator + factor + t + 2;
        self.need = if self.terms.len() <= 2 {
            asked
        } else {
            self.need.max(asked)
        };
    }

    fn len(&self) -> usize {
        self.terms.len()
    }

    /// The chain's last denominator.
    #[allow(
        clippy::expect_used,
        reason = "dens starts as [1], clear keeps its first, and push alone adds"
    )]
    fn last(&self) -> &BigUint {
        self.dens.last().expect("the chain starts at 1")
    }

    // q-rational-time.10 cited: collins-1995 §4.3 -- method II: for each prime divisor p of M, check whether p divides B, ignoring any remaining primes as soon as a zero remainder is computed
    /// Whether the last denominator is a unit modulo the product of
    /// `primes`, the lifting's primes: its residues at the primes past
    /// those taken (`extend`), until one is 0. A denominator an attempt
    /// finds is prime to the primes it was reconstructed modulo: it divides
    /// l·f, l the unit the residue was scaled by and f MQRR's, whose gcd
    /// with M divides n′ and so is 1. One found modulo a prefix of M's
    /// primes is taken only where f is a unit at the rest (`push_found`).
    /// Its lcm with the chain's last is thus a unit wherever the last is.
    fn coprime(&mut self, primes: &[u64]) -> bool {
        if self.unit && self.dens.len() > 1 {
            self.unit = self.extend(primes);
        }
        self.unit
    }

    /// Takes the last denominator's residues, and their Shoup factors, at
    /// the primes of `primes` past those taken, in order, until one is 0:
    /// whether none is.
    fn extend(&mut self, primes: &[u64]) -> bool {
        let Rationals {
            dens,
            residues,
            factors,
            ..
        } = self;
        #[allow(
            clippy::expect_used,
            reason = "dens starts as [1], clear keeps its first, and push alone adds"
        )]
        let last = dens.last().expect("the chain starts at 1");
        for &p in &primes[residues.len()..] {
            let r = residue(last, p);
            if r == 0 {
                return false;
            }
            residues.push(r);
            factors.push(shoup_factor(r, p));
        }
        true
    }

    /// The rationals whose chain's last is the guess for the part of the
    /// next element (q-prime-count-time.8), where that last is not 1 and is
    /// a unit modulo the product of `primes` (`coprime`): its residues at
    /// every prime of `primes` are taken.
    fn guess(&mut self, primes: &[u64]) -> Option<&Rationals> {
        (self.dens.len() > 1 && self.coprime(primes)).then_some(&*self)
    }

    /// l's residues at `primes`, M's primes, and their Shoup factors, l as
    /// `over` names it, those of the part's chain's last taken where they
    /// are not: None for l = 1.
    fn by<'s>(&'s mut self, over: &'s Over<'_>, primes: &[u64]) -> Option<(&'s [u64], &'s [u64])> {
        match over {
            Over::Last(_) if self.dens.len() == 1 => None,
            Over::Last(_) => {
                // The last is a unit modulo M where the one before it was
                // (`coprime`), so none of its residues is 0.
                let unit = self.extend(primes);
                debug_assert!(unit, "the chain's last is a unit modulo M");
                Some((&self.residues, &self.factors))
            }
            Over::Guess(guess) => Some((&guess.residues, &guess.factors)),
            Over::One => None,
        }
    }

    /// The last denominator's residue at p, `radix`'s prime, which `at`
    /// keeps with the chain's length it was taken at: taken again where
    /// the chain has grown since.
    fn last_residue(&self, at: &mut (usize, u64), radix: &Radix) -> u64 {
        if at.0 != self.dens.len() {
            *at = (self.dens.len(), radix.residue_big(self.last()));
        }
        at.1
    }

    /// The term v over the last denominator.
    fn push_over_last(&mut self, v: BigInt) {
        self.width = self.width.max(limbs(v.magnitude()));
        let last = self.dens.len() - 1;
        self.terms.push((v, last.narrow()));
    }

    /// The term n/d, in lowest terms: n·(L/d) over the last denominator L
    /// where d divides it, else n·(L'/d) over L' = lcm(L, d), which the
    /// chain gains.
    fn push_rational(&mut self, r: Rational) {
        let (q, rem) = self.last().div_rem(&r.den);
        let num = if rem.is_zero() {
            r.num * BigInt::from(q)
        } else {
            // q-rational-time.9 cited: collins-1995 §4.3 -- method I, gcd(A, B) computed by the algorithm of Section 3, used throughout
            let g = gcd(self.last(), &r.den);
            // L' = (L/g)·d, and L'/d = L/g.
            let f = self.last() / g;
            let next = &f * &r.den;
            self.width = self.width.max(limbs(&next));
            self.dens.push(next);
            self.residues.clear();
            self.factors.clear();
            r.num * BigInt::from(f)
        };
        self.width = self.width.max(limbs(num.magnitude()));
        let last = self.dens.len() - 1;
        self.terms.push((num, last.narrow()));
    }

    // q-attempt-time.2 design -- the residues of the chain's next l·f at M's primes, and their Shoup factors, are those of l, taken, times f's residues, not l·f's residues taken again from its limbs at the next coefficient (log 0085-1)
    /// The chain gains its next, l·f, l its last, and the term n over it.
    /// The residues taken become l·f's at their primes, the first of
    /// `primes`: l's times f's. f is a unit modulo their product, so none
    /// becomes 0.
    fn grow(&mut self, f: &BigUint, n: BigInt, primes: &[u64]) {
        let next = self.last() * f;
        let taken = self.residues.iter_mut().zip(&mut self.factors);
        for ((r, factor), &p) in taken.zip(primes) {
            *r = mul_mod(*r, residue(f, p), p);
            debug_assert_ne!(*r, 0, "f is a unit modulo {p}");
            *factor = shoup_factor(*r, p);
        }
        self.width = self.width.max(limbs(&next)).max(limbs(n.magnitude()));
        self.dens.push(next);
        let last = self.dens.len() - 1;
        self.terms.push((n, last.narrow()));
    }

    /// Whether r, in lowest terms, reduces to c at p, `radix`'s prime; it
    /// is pushed where it does.
    fn push_if_fits(&mut self, r: Rational, c: u64, radix: &Radix) -> bool {
        // q-crt-time.8 design -- a rational n/d fits c at p where d mod p ≠ 0 and n mod p = c·(d mod p) mod p, not n·d⁻¹ mod p by an extended Euclid compared with c (log 0023-1)
        let fits = radix.fits(&r, c);
        if fits {
            self.push_rational(r);
        }
        fits
    }

    // q-prime-count-time.1 cited: monagan-2004 §3 -- MQRR outputs n/d with high probability once m is only a modest number of bits longer than 2|n|d, where N = D = ⌊√(m/2)⌋ requires up to twice as many primes as are necessary; traverso-1989 §Reconstruction algorithms -- the coefficients are lifted one at a time, and the others are multiplied by the denominator found
    /// The next coefficient under `Reconstruct::Chain`, whose image at p,
    /// `at`'s prime, is c, and whose v = u·l mod M, u its residue modulo M
    /// and l as `over` names it, is `v`, as `mqrr.classify` gives it.
    /// Where v is small the rational is v/l. Else it is n′/(l·f) for MQRR's
    /// n′/f on v: first the rational before its first quotient of at least
    /// 2^T, and where that does not fit, the one before its largest.
    /// Whether one was found that fits c at p, and it was pushed. Where
    /// none was, nothing changed but the residue at p `over` keeps.
    /// `euclid` counts MQRR's runs. M is `modulo`'s: all of the lifting's
    /// primes, or a prefix of them.
    fn take(
        &mut self,
        v: Scaled,
        c: u64,
        over: &mut Over,
        modulo: Modulo,
        at: &At,
        euclid: &mut u64,
    ) -> bool {
        let mqrr = modulo.mqrr;
        let radix = at.radix;
        match v {
            // q-prime-count-time.2 cited: traverso-1989 §Reconstruction algorithms -- the coefficients are lifted one at a time and the others multiplied by the denominator found, and the remaining coefficients that fall below the bound are lifted with denominator 1
            Scaled::Small(small) => match over {
                Over::Last(last) => {
                    let l = self.last_residue(last, radix);
                    // Where p divides L, the rational in lowest terms is
                    // checked, as one MQRR finds is.
                    let fits = if l != 0 {
                        radix.fits_over(&small, l, c)
                    } else {
                        radix.fits(&lowest(&small, self.last()), c)
                    };
                    if fits {
                        self.push_over_last(small);
                    }
                    fits
                }
                Over::Guess(guess) => self.push_if_fits(lowest(&small, guess.last()), c, radix),
                Over::One => self.push_if_fits(lowest(&small, &self.dens[0]), c, radix),
            },
            Scaled::Residue(residue) => {
                *euclid += 1;
                // q-attempt-time.3 invented -- MQRR's first pass ends at the first quotient, in the order of the steps, that reaches 2^T, and the attempt takes the rational before it where it fits at p, MQRR running to its largest quotient only where it does not, not the first pass to the remainder below the largest quotient at every coefficient, nor the pass ended at a quotient past √M (log 0085-1)
                match mqrr.first(&residue) {
                    First::Nothing => return false,
                    First::Rational(found) => {
                        if self.push_found(found, c, over, modulo.rest, at) {
                            return true;
                        }
                    }
                    First::Shared => {}
                }
                mqrr.reconstruct(&residue)
                    .is_some_and(|found| self.push_found(found, c, over, modulo.rest, at))
            }
        }
    }

    /// The coefficient n′/(l·f) for MQRR's `found` = n′/f on v = u·l mod
    /// M, l as `over` names it, pushed where it fits c at p, `at`'s prime:
    /// whether it was. M is a prefix of the lifting's primes, `rest` the
    /// primes past it, none where M is all of them. Where it was not,
    /// nothing changed but the residue at p `over` keeps.
    fn push_found(
        &mut self,
        found: Rational,
        c: u64,
        over: &mut Over,
        rest: &[u64],
        at: &At,
    ) -> bool {
        let radix = at.radix;
        match over {
            // q-lift-rest-time.10 design -- a denominator f that MQRR found modulo a prefix of M's primes is taken only where it is a unit at each prime past the prefix, so that the chain's last stays a unit modulo M (`coprime`), not f taken as found, whose residue 0 at a later prime would leave the chain's last no unit there (not measured)
            Over::Last(_) if rest.iter().any(|&q| residue(&found.den, q) == 0) => false,
            Over::Last(last) => {
                let l = self.last_residue(last, radix);
                // q-attempt-time.1 cited: traverso-1989 §Reconstruction algorithms -- a better version lifts the coefficients one at a time, and multiplies the other coefficients by the denominator that one has found at this step
                // For the coefficient a/d in lowest terms and h = gcd(l, d),
                // n′ = a·l/h and f = d/h, so lcm(l, d) = l·f, over which the
                // coefficient is n′.
                if found.den.is_one() {
                    if l != 0 {
                        let fits = radix.fits_over(&found.num, l, c);
                        if fits {
                            self.push_over_last(found.num);
                        }
                        return fits;
                    }
                } else {
                    let next = radix.mul(l, radix.residue_big(&found.den));
                    if next != 0 {
                        let fits = radix.fits_over(&found.num, next, c);
                        if fits {
                            self.grow(&found.den, found.num, at.crt.primes());
                            **last = (self.dens.len(), next);
                        }
                        return fits;
                    }
                }
                // p divides l·f: the rational in lowest terms is checked.
                let rational = lowest(&found.num, &(self.last() * &found.den));
                self.push_if_fits(rational, c, radix)
            }
            Over::Guess(guess) => {
                let rational = lowest(&found.num, &(guess.last() * &found.den));
                self.push_if_fits(rational, c, radix)
            }
            Over::One => self.push_if_fits(found, c, radix),
        }
    }

    // q-rational-time.5 design -- the checks at a prime take each chain denominator's residue once per part, and each numerator's once per term, not a residue of the denominator per term (log 0028-1)
    /// Whether the first `coefs.len()` rationals reduce mod p, `radix`'s
    /// prime, to `coefs`, the image's coefficients of their terms: each
    /// chain denominator's residue is taken once.
    fn fits(&self, coefs: &[u64], radix: &Radix) -> bool {
        debug_assert!(coefs.len() <= self.terms.len());
        let mut at = (u32::MAX, 0);
        self.terms.iter().zip(coefs).all(|((num, den), &c)| {
            if at.0 != *den {
                at = (*den, radix.residue_big(&self.dens[*den as usize]));
            }
            radix.fits_over(num, at.1, c)
        })
    }

    // q-rational-time.5 design -- the verification takes each chain denominator's inverse once per part and prime, and each numerator's residue once per term, not an inverse per term (log 0028-1)
    /// Each rational's residue mod p, `radix`'s prime: None where p
    /// divides its chain denominator. Each chain denominator's inverse is
    /// taken once.
    fn images<'a>(&'a self, radix: &'a Radix) -> impl Iterator<Item = Option<u64>> + 'a {
        let mut at = (u32::MAX, None);
        self.terms.iter().map(move |(num, den)| {
            if at.0 != *den {
                at = (*den, radix.inverse(&self.dens[*den as usize]));
            }
            at.1.map(|di| radix.over(num, di))
        })
    }

    // q-rational-time.6 design -- the basis text's integers are the numerators times the last denominator over their own, and the leading integer is the last denominator, not the lcm of every term's denominator folded by num-integer's gcd and a division per term (log 0028-1)
    /// Each rational times the last denominator L, an integer:
    /// num·(L/dens[den]), the quotient taken once per chain denominator.
    fn integers(&self) -> impl Iterator<Item = BigInt> + '_ {
        let last: u32 = (self.dens.len() - 1).narrow();
        let mut at = (u32::MAX, BigInt::one());
        self.terms.iter().map(move |(num, den)| {
            if *den == last {
                return num.clone();
            }
            if at.0 != *den {
                at = (*den, BigInt::from(self.last() / &self.dens[*den as usize]));
            }
            num * &at.1
        })
    }
}

impl Lifting {
    /// No prime combined: per leading monomial of `leads` an open element,
    /// `shape` the elements and the parts of each, every part without a
    /// term. Its attempts reconstruct as `how` says.
    fn new(
        leads: &[u32],
        n: usize,
        (elements, parts): (usize, usize),
        how: Reconstruct,
    ) -> Lifting {
        debug_assert_eq!(leads.len(), elements * n, "n exponents per element");
        let elements = (0..elements)
            .map(|i| Element {
                lead: leads[i * n..(i + 1) * n].to_vec(),
                parts: (0..parts).map(|_| Part::new()).collect(),
                rationals: vec![Rationals::new(); parts],
                kept: false,
            })
            .collect();
        Lifting {
            crt: Crt::new(),
            elements,
            how,
            width: 1,
            scratch: Scratch::default(),
            attempts: 0,
            prefixes: Prefixes::default(),
        }
    }

    /// Absorbs the replay's image at p, as `parts` gives it: merges the
    /// terms of the open elements, reconstructs the open ones from their
    /// images at the primes of M in order until one fails
    /// (`attempt_open`), and appends the image at p to the open elements'
    /// images. A kept element's image is not read. `report` counts the
    /// reconstructions and the images appended. The merge, the attempts'
    /// conversions, and the append spend units of `budget` as they work
    /// (`merge`, `Scratch::convert`, `append`). The call report's stages
    /// `Merge`, `Reconstruct`, and `Crt` take their times, `Crt` the step's
    /// and the append's. The bound that stopped it, where one did.
    fn absorb(
        &mut self,
        mut image: Vec<Vec<Sparse>>,
        p: u64,
        named: &MonomialOrder,
        n: usize,
        budget: &mut Budget,
        report: &mut Report,
    ) -> Result<(), Stop> {
        let step = self.crt.step(p, self.width);
        budget.mark(Stage::Crt);
        // q-lift.18 cited: parisse-2019 §1 -- the basis has about 20 millions monomials, 19825931 or a little less for some primes, because of cancellation
        // A part takes the union of its images' terms; a term an image lacks
        // reads coefficient 0 at its prime.
        for (element, parts) in self.elements.iter_mut().zip(&mut image) {
            // result-time.15 design -- the lift reads no image of a kept element, so an image that lacks a term of a kept element, or holds a term it lacks, changes nothing, not a catch-up before the merge with the element reopened where the image gains a term and checked where it lacks one (not measured)
            if element.kept {
                continue;
            }
            let mut gained = false;
            for (part, sparse) in element.parts.iter_mut().zip(parts) {
                gained |= merge(part, sparse, None, named, n, budget)?;
            }
            if gained {
                element.clear();
            }
        }
        budget.mark(Stage::Merge);
        if !self.crt.primes().is_empty() {
            self.attempt_open(&image, &step, budget, report)?;
        }
        budget.mark(Stage::Reconstruct);
        self.append_open(&image, budget, report)?;
        self.crt.push(step);
        budget.mark(Stage::Crt);
        Ok(())
    }

    // q-lift-rest-time.2 cited: dekleine-2001 §3.2 -- if the digits of agreement do not suffice, we get another batch of primes and continue: the reconstruction is tried once a batch; demin-2024 §4.3, independently -- the modular images are computed in batches and the reconstruction attempted after each
    /// Absorbs a pair image at its first prime p, as `absorb` does, with no
    /// attempt: merges each open element's terms with the image's, and the
    /// second prime's coefficients take the merged terms beside the
    /// first's; appends the image at p. Per element, its coefficients at
    /// the second prime, aligned with its part's terms, for
    /// `absorb_second`, and none for an element kept before. The bound that
    /// stopped it, where one did.
    fn absorb_first(
        &mut self,
        image: PairImage,
        p: u64,
        named: &MonomialOrder,
        n: usize,
        budget: &mut Budget,
        report: &mut Report,
    ) -> Result<Vec<Vec<u64>>, Stop> {
        let PairImage {
            mut first,
            mut second,
            ..
        } = image;
        let step = self.crt.step(p, self.width);
        budget.mark(Stage::Crt);
        let images = self.elements.iter_mut().zip(&mut first).zip(&mut second);
        for ((element, sparse), at_second) in images {
            // result-time.15 design -- the lift reads no image of a kept element, so an image that lacks a term of a kept element, or holds a term it lacks, changes nothing, not a catch-up before the merge with the element reopened where the image gains a term and checked where it lacks one (not measured)
            if element.kept {
                *at_second = Vec::new();
                continue;
            }
            debug_assert_eq!(element.parts.len(), 1, "a pair replay tracks no cofactors");
            if merge(
                &mut element.parts[0],
                sparse,
                Some(at_second),
                named,
                n,
                budget,
            )? {
                element.clear();
            }
        }
        budget.mark(Stage::Merge);
        self.append_open(&first, budget, report)?;
        self.crt.push(step);
        budget.mark(Stage::Crt);
        Ok(second)
    }

    /// Absorbs a pair image at its second prime q, whose coefficients
    /// `absorb_first` aligned with the parts' terms: the attempts at q from
    /// the open elements on (`attempt_open`), then the append, as `absorb`
    /// runs them. The bound that stopped it, where one did.
    fn absorb_second(
        &mut self,
        coefs: &[Vec<u64>],
        q: u64,
        budget: &mut Budget,
        report: &mut Report,
    ) -> Result<(), Stop> {
        let step = self.crt.step(q, self.width);
        budget.mark(Stage::Crt);
        self.attempt_open(coefs, &step, budget, report)?;
        budget.mark(Stage::Reconstruct);
        self.append_open(coefs, budget, report)?;
        self.crt.push(step);
        budget.mark(Stage::Crt);
        Ok(())
    }

    /// The attempts at `step`'s prime, from M: from the first open element
    /// on, each open element is reconstructed from its images and checked
    /// against `image`, its coefficients there, until one fails. `report`
    /// counts the reconstructions. M holds a prime. The bound that stopped
    /// them, where one did.
    fn attempt_open<E: Aligned>(
        &mut self,
        image: &[E],
        step: &CrtStep,
        budget: &mut Budget,
        report: &mut Report,
    ) -> Result<(), Stop> {
        let modulus = self.crt.modulus();
        debug_assert!(modulus.bits() > 1, "M holds a prime");
        // q-lift.15 cited: parisse-2019 §2 -- an attempt is made to reconstruct the first basis element that is not yet reconstructed and check that it fits with the last prime; if successful, try the next one until reconstruction fails; parisse-2013 §4 step 4 -- reconstruct with all primes but the last one, and check the value modulo the last prime
        let at = At {
            by: match self.how {
                Reconstruct::Chain => By::Chain(Mqrr::new(modulus)),
                #[cfg(test)]
                Reconstruct::Wang => By::Wang(Wang::new(modulus)),
            },
            radix: step.radix(),
            crt: &self.crt,
            guess: self.attempts.is_multiple_of(2),
        };
        self.attempts += 1;
        // The element before the one attempted, which is kept.
        let mut previous: Option<&mut Element> = None;
        for (element, coefs) in self.elements.iter_mut().zip(image) {
            // result-time.13 design -- the lift checks no kept element at the primes after its keep, the verification its one check after the attempt's, not every kept element's rationals checked against the image at every further prime (log 0042-1)
            if !element.kept {
                let kept = element.attempt(
                    coefs,
                    &at,
                    &mut self.scratch,
                    &mut self.prefixes,
                    previous.take(),
                    budget,
                    report,
                )?;
                self.width = self.width.max(element.width());
                if !kept {
                    break;
                }
            }
            previous = Some(element);
        }
        Ok(())
    }

    /// Appends `image`, the coefficients at the next prime aligned with the
    /// parts' terms, to the open elements' images (`append`). `report`
    /// counts the images appended. The bound that stopped it, where one
    /// did.
    fn append_open<E: Aligned>(
        &mut self,
        image: &[E],
        budget: &mut Budget,
        report: &mut Report,
    ) -> Result<(), Stop> {
        for (element, coefs) in self.elements.iter_mut().zip(image) {
            // q-crt-time.1 design -- a kept element is left out of the append, not every element's images appended at every prime (log 0023-1)
            if element.kept {
                continue;
            }
            for (j, part) in element.parts.iter_mut().enumerate() {
                // q-lift.42 design -- the append ticks the budget once per part of an element, the reconstruction once per coefficient, not once per element (log 0016-2)
                budget.tick()?;
                report.steps += part.len() as u64;
                append(part, coefs.part(j), budget)?;
            }
        }
        Ok(())
    }

    /// The elements not kept.
    fn open(&self) -> usize {
        self.elements.iter().filter(|e| !e.kept).count()
    }

    /// The lifting as `Complete` where a prime is combined and every
    /// element kept, None where not.
    fn complete(&self) -> Option<Complete<'_>> {
        let done = !self.crt.primes().is_empty() && self.elements.iter().all(|e| e.kept);
        done.then_some(Complete(self))
    }
}

impl Complete<'_> {
    /// Whether q divides a denominator of a kept basis polynomial: the last
    /// of its chain.
    fn divides_a_denominator(self, q: u64) -> bool {
        self.0
            .elements
            .iter()
            .any(|e| residue(e.rationals[0].last(), q) == 0)
    }

    /// Whether each kept basis polynomial modulo q, the terms whose residue
    /// is 0 left out, is the element of `image` without its leading term, a
    /// run's reduced basis at q whose leading monomials are the lifting's.
    fn verify(self, image: &Image, q: u64, n: usize) -> bool {
        let radix = Radix::new(q, self.0.width);
        for (element, g) in self.0.elements.iter().zip(&image.basis) {
            let exponents = &element.parts[0].exponents;
            let mut k = 0;
            for (t, c) in element.rationals[0].images(&radix).enumerate() {
                #[allow(
                    clippy::expect_used,
                    reason = "the verification runs where q divides no denominator, and each divides the last"
                )]
                let c = c.expect("q divides no denominator of the basis");
                if c == 0 {
                    continue;
                }
                let term = &exponents[t * n..(t + 1) * n];
                if k == g.coefs.len() || g.coefs[k] != c || g.exponents[k * n..(k + 1) * n] != *term
                {
                    return false;
                }
                k += 1;
            }
            if k != g.coefs.len() {
                return false;
            }
        }
        true
    }
}

/// What an attempt at p reads: how it reconstructs modulo M, p's radix,
/// the primes of M with their conversion, and whether it takes the guess
/// at a part's first coefficient (q-prime-count-time.8): at the lifting's
/// attempts of even index, from 0 (q-lift-rest-time.3).
struct At<'a> {
    by: By,
    radix: &'a Radix,
    crt: &'a Crt,
    guess: bool,
}

/// What an attempt reconstructs a coefficient modulo: M, or the product of
/// a prefix of M's primes (q-lift-rest-time.5).
#[derive(Clone, Copy)]
struct Modulo<'a> {
    mqrr: &'a Mqrr,
    /// M's primes past the prefix, none for M.
    rest: &'a [u64],
}

/// An attempt's reconstruction as `Reconstruct` names it, prepared once
/// modulo M.
enum By {
    /// `Reconstruct::Chain`.
    Chain(Mqrr),
    /// `Reconstruct::Wang`.
    #[cfg(test)]
    Wang(Wang),
}

/// What the attempt scales a coefficient's images by, l.
enum Over<'a> {
    /// The part's chain's last, a unit modulo M, with its residue at p and
    /// the chain's length where it was taken.
    Last(&'a mut (usize, u64)),
    /// The previous element's rationals of the part, whose chain's last is
    /// not 1 and a unit modulo M, with its residues at M's primes, at the
    /// part's first coefficient (q-prime-count-time.8).
    Guess(&'a Rationals),
    /// 1, where the part's chain's last is no unit modulo M.
    One,
}

impl Element {
    // q-reconstruct-time.10 design -- a rational that fails the completion check clears every rational of its element, not the rationals from the failing one on with those before it kept (log 0018-1)
    /// Empties every part's rationals and its chain: the element is open,
    /// and its next attempt starts at its first coefficient.
    fn clear(&mut self) {
        self.rationals.iter_mut().for_each(Rationals::clear);
        self.kept = false;
    }

    /// The most limbs of a number its rationals hold.
    fn width(&self) -> usize {
        self.rationals.iter().map(|r| r.width).max().unwrap_or(1)
    }

    /// The attempt at p, `at`'s prime, on an open element, whose parts hold
    /// the images at `at`'s primes of M and whose coefficients at p, merged,
    /// are `image`: each coefficient past the element's rationals, part 0
    /// first, converted in `scratch`, reconstructed, and checked at p. Over
    /// the chain, once a coefficient of the element fitted in the attempt,
    /// a further coefficient of a part that holds a rational is first
    /// converted and reconstructed modulo the product of the first primes
    /// of M its part needs, a prefix `prefixes` keeps. Otherwise, or where
    /// it does not fit there, it is converted at M's primes: the conversion
    /// ends at its first digit 0 or pⱼ − 1 below the top, and where the
    /// value it gives does not fit, the coefficient is converted in full
    /// and taken again. `previous` is the kept element before it, whose
    /// chain denominators are the guesses at its parts' first coefficients
    /// where `at` takes the guess. Whether the element is kept now.
    /// `report` counts the reconstructions and those that ran the Euclidean
    /// algorithm.
    #[allow(
        clippy::too_many_arguments,
        reason = "the attempt's inputs, its two buffers, and the call's budget and report, none of which belongs with another"
    )]
    fn attempt(
        &mut self,
        image: &impl Aligned,
        at: &At,
        scratch: &mut Scratch,
        prefixes: &mut Prefixes,
        mut previous: Option<&mut Element>,
        budget: &mut Budget,
        report: &mut Report,
    ) -> Result<bool, Stop> {
        let (radix, primes) = (at.radix, at.crt.primes());
        // The rationals of earlier attempts: every part before j, and part
        // j's first t.
        let j = self
            .rationals
            .iter()
            .zip(&self.parts)
            .position(|(r, part)| r.len() < part.len())
            .unwrap_or(self.parts.len());
        let t = self.rationals.get(j).map_or(0, Rationals::len);
        // Whether a coefficient of the element fitted in this attempt.
        let mut fitted = false;
        let parts = self.parts.iter().zip(&mut self.rationals);
        for (index, (part, rationals)) in parts.enumerate().skip(j) {
            // q-rational-time.10 cited: collins-1995 §4.3 -- method II: for each prime divisor p of M, check whether p divides B, ignoring any remaining primes as soon as a zero remainder is computed
            let unit = rationals.coprime(primes);
            // q-prime-count-time.8 design -- at a part's first coefficient, at every other attempt, the residue is scaled by the previous element's chain denominator of the part, not by 1 alone, the guess at every prime, both at every prime with the guess where 1 fails, nor the lcm of every element's denominator before (log 0076-1)
            // q-lift-rest-time.3 design -- the guess at a part's first coefficient is taken at the lifting's attempts of even index, counted from 0, which alternates by the pair where the attempts run once a pair and agrees with M of an odd number of primes where every prime attempts, not at M of an odd number of primes (q-prime-count-time.8 as written), at which every attempt once a pair would guess (log 0088-1)
            let mut guess = if rationals.terms.is_empty() && at.guess {
                previous
                    .as_deref_mut()
                    .and_then(|e| e.rationals.get_mut(index))
                    .and_then(|r| r.guess(primes))
            } else {
                None
            };
            let mut last = (0, 0);
            // q-reconstruct-time.8 design -- an attempt goes on at the first coefficient past the element's rationals, which earlier attempts reconstructed and checked, not the element reconstructed again from its first coefficient at each prime (log 0018-1)
            let terms = part.images().zip(image.part(index));
            for (u, &c) in terms.skip(rationals.len()) {
                // q-lift.42 design -- the append ticks the budget once per part of an element, the reconstruction once per coefficient, not once per element (log 0016-2)
                budget.tick()?;
                report.reconstructed += 1;
                let fits = match &at.by {
                    By::Chain(mqrr) => {
                        let mut over = match guess.take() {
                            Some(guess) => Over::Guess(guess),
                            None if unit => Over::Last(&mut last),
                            None => Over::One,
                        };
                        // The chain before the coefficient, for its need.
                        let chain = (rationals.dens.len(), rationals.last().bits());
                        // The coefficient's runs of the Euclidean algorithm.
                        let mut euclid = 0;
                        let mut fits = false;
                        // q-lift-rest-time.5 invented -- once a coefficient of an element fitted in an attempt, a further coefficient of a part that holds a rational is converted from its images at the first j primes and reconstructed modulo their product, j from the part's need, and taken modulo M where it does not fit there, not every coefficient modulo M, nor the prefix from the attempt's first coefficient on, which converts every failing probe twice (log 0088-1)
                        if fitted
                            && rationals.need > 0
                            && let Some(j) = prefixes.length(primes, rationals.need + SLACK)
                        {
                            let by = rationals.by(&over, primes);
                            let (form, _) =
                                scratch.convert(&u[..j], by, at.crt, Crt::convert_small, budget)?;
                            let modulo = Modulo {
                                mqrr: prefixes.mqrr(primes, j),
                                rest: &primes[j..],
                            };
                            let v = modulo.mqrr.classify(form, &scratch.limbs);
                            fits = rationals.take(v, c, &mut over, modulo, at, &mut euclid);
                        }
                        let modulo = Modulo { mqrr, rest: &[] };
                        if !fits {
                            let by = rationals.by(&over, primes);
                            // q-attempt-time.6 design -- every conversion of an attempt over the chain ends at its first digit 0 or p_j − 1 below the top, those whose v MQRR then takes among them, not the ended conversion only where the part's chain's last is past 1 (not measured)
                            let (form, ended) =
                                scratch.convert(u, by, at.crt, Crt::convert_small, budget)?;
                            let v = mqrr.classify(form, &scratch.limbs);
                            fits = rationals.take(v, c, &mut over, modulo, at, &mut euclid);
                            // q-attempt-time.5 design -- where the small value of a conversion that ended below the top does not fit at p, the attempt converts the same images in full and takes the coefficient again, not the attempt failed as a misfit fails it (not measured)
                            if !fits && ended {
                                let by = rationals.by(&over, primes);
                                let (form, _) =
                                    scratch.convert(u, by, at.crt, Crt::convert, budget)?;
                                let v = mqrr.classify(form, &scratch.limbs);
                                fits = rationals.take(v, c, &mut over, modulo, at, &mut euclid);
                            }
                        }
                        report.euclid += u64::from(euclid > 0);
                        if fits {
                            rationals.note_need(chain, mqrr.threshold());
                        }
                        fits
                    }
                    #[cfg(test)]
                    By::Wang(wang) => {
                        report.euclid += 1;
                        let (form, _) = scratch.convert(u, None, at.crt, Crt::convert, budget)?;
                        wang.reconstruct_converted(form, &scratch.limbs)
                            .is_some_and(|r| rationals.push_if_fits(r, c, radix))
                    }
                };
                // q-lift.16 cited: collins-1995 §2 -- we abort the attempt upon encountering any coefficient for which the three conditions cannot be satisfied
                if !fits {
                    return Ok(false);
                }
                fitted = true;
            }
        }
        // q-reconstruct-time.9 design -- the rationals of earlier attempts are checked at p once the element's last coefficient reconstructs, and no more, not each rational checked at every prime from its reconstruction on (not measured)
        let earlier = self.rationals[..j]
            .iter()
            .enumerate()
            .all(|(i, r)| r.fits(image.part(i), radix))
            && (j == self.parts.len() || self.rationals[j].fits(&image.part(j)[..t], radix));
        if earlier {
            self.kept = true;
            // result-time.14 design -- a kept element's images are dropped at its keep, not kept for a catch-up (log 0042-1)
            for part in &mut self.parts {
                part.images = Vec::new();
            }
        } else {
            self.clear();
        }
        Ok(self.kept)
    }
}

/// v/l in lowest terms, l > 0.
fn lowest(v: &BigInt, l: &BigUint) -> Rational {
    // q-rational-time.9 cited: collins-1995 §4.3 -- method I, gcd(A, B) computed by the algorithm of Section 3, used throughout
    let g = gcd(v.magnitude(), l);
    Rational {
        num: v / BigInt::from(g.clone()),
        den: l / g,
    }
}

// cofactor-stop-latency.1 design -- every stage that works on cofactors spends units of work on the call's budget as it works, the absorb's merge and append over Q among them, and a stage stops where the budget's check stops the call, not the ticks per row and generator and per part, and nothing for the image and the merge (log 0058-1)
/// Merges the image's terms `image` into the part's: the part takes the
/// union of both in decreasing order, a term new to it with images 0, its
/// coefficients at the primes combined since every earlier image lacked it.
/// The image takes the part's terms, coefficient 0 where it lacked one, and
/// so does `second`, a pair image's coefficients of the image's terms at
/// its second prime, where there is one.
/// Whether the part gained a term. The comparison of the part's exponents
/// with the image's spends a unit of `budget` per 8 words, `COMPARED`
/// words at a time, and the union `COMBINED`·(`MERGED` + w) units before
/// each `COMBINED` terms it writes, w the words of a term's images. A stop
/// where the budget's check stops the call, the part and the image then as
/// they were.
pub(crate) fn merge(
    part: &mut Part,
    image: &mut Sparse,
    second: Option<&mut Vec<u64>>,
    named: &MonomialOrder,
    n: usize,
    budget: &mut Budget,
) -> Result<bool, Stop> {
    debug_assert!(
        second.as_ref().is_none_or(|s| s.len() == image.coefs.len()),
        "a coefficient a term at the second prime"
    );
    // The comparison reads both parts' exponents.
    let same = equal_spent(&part.exponents, &image.exponents, budget)?;
    if same {
        return Ok(false);
    }
    match named {
        MonomialOrder::Grevlex => merge_in(part, image, second, &Grevlex, n, budget),
        MonomialOrder::Lex => merge_in(part, image, second, &Lex, n, budget),
        MonomialOrder::Block(sizes) => {
            #[allow(
                clippy::expect_used,
                reason = "run_in took the same sizes for the same n"
            )]
            let block = Block::new(sizes, n).expect("the run took the order's sizes");
            merge_in(part, image, second, &block, n, budget)
        }
    }
}

// cofactor-stop-latency.1 design -- every stage that works on cofactors spends units of work on the call's budget as it works, the absorb's merge and append over Q among them, and a stage stops where the budget's check stops the call, not the ticks per row and generator and per part, and nothing for the image and the merge (log 0058-1)
// cofactor-stop-latency.8 design -- the append writes a part's images in place, the vector resized and the terms moved from the last to the first, not a new vector per part and prime (log 0058-1)
// q-lift-growth-time.8 design -- a part's images lie term by term in one vector, each term at a capacity that doubles once the primes reach it, the vector resized in place and the terms moved from the last to the first, not one vector per prime of the image's coefficients, every term grown by a word at every prime, nor a fresh vector at each doubling (log 0083-1)
/// Appends `coefs`, the image's coefficients of the part's terms at the
/// next prime, to the part's images, in place: where the primes combined
/// fill the capacity, it doubles, the vector resized and each term's
/// images moved to their new place from the last term to the first, so
/// that none is written over before it moves. Spends `LIMB` units of
/// `budget` a word written, the moves included, `COMBINED` coefficients
/// at a time. A stop where the budget's check stops the call, the part's
/// images then partly moved and appended, for the call to drop.
pub(crate) fn append(part: &mut Part, coefs: &[u64], budget: &mut Budget) -> Result<(), Stop> {
    let (len, old, count) = (part.len(), part.cap, part.count);
    debug_assert_eq!(len, coefs.len(), "a coefficient a term");
    let cap = if count == old { 2 * old } else { old };
    part.images.resize(len * cap, 0);
    // A term's words written: its image at p, and its images moved where
    // the capacity doubles.
    let words = if cap == old { 1 } else { count + 1 };
    let mut done = 0;
    for t in (0..len).rev() {
        // Term t moves to t·cap, past the old places of the terms before
        // it, which end at t·old.
        if cap != old {
            part.images.copy_within(t * old..t * old + count, t * cap);
        }
        part.images[t * cap + count] = coefs[t];
        done += 1;
        if done == COMBINED {
            done = 0;
            // cofactor-stop-latency.6 design -- the append spends LIMB, 2, units a word it writes after each COMBINED, 2^10, coefficients, beside the tick per part (q-lift.42), not the tick per part alone (log 0058-1)
            // q-lift-growth-time.10 design -- the append spends LIMB units a word it writes, its moves included, COMBINED coefficients at a time, and an attempt spends its conversion's multiply-adds after the conversion, not the combination's LIMB·(w′ + 4) a coefficient (cofactor-stop-latency.6), nor conversions that tick alone (not measured)
            budget.spend(COMBINED * LIMB * words)?;
        }
    }
    part.cap = cap;
    part.count = count + 1;
    // cofactor-stop-latency.6 design -- the append spends LIMB, 2, units a word it writes after each COMBINED, 2^10, coefficients, beside the tick per part (q-lift.42), not the tick per part alone (log 0058-1)
    budget.spend(done * LIMB * words)
}

// cofactor-stop-latency.5 design -- the merge's comparison of a part's exponents with the image's spends a unit per 8 words, COMPARED, 2^16, words at a time, and the merge MERGED + w units a term, MERGED = 16, w the words of a term's images, COMBINED terms at a time, not the comparison and the merge unspent (log 0058-1)
/// Units of work per term a merge writes, beside its images' words.
pub(crate) const MERGED: usize = 16;

// cofactor-stop-latency.6 design -- the append spends LIMB, 2, units a word it writes after each COMBINED, 2^10, coefficients, beside the tick per part (q-lift.42), not the tick per part alone (log 0058-1)
/// Units of work per word the append writes.
pub(crate) const LIMB: usize = 2;

// cofactor-stop-latency.6 design -- the append spends LIMB, 2, units a word it writes after each COMBINED, 2^10, coefficients, beside the tick per part (q-lift.42), not the tick per part alone (log 0058-1)
/// The terms a merge writes, and the coefficients an append appends,
/// between two spends.
pub(crate) const COMBINED: usize = 1 << 10;

// cofactor-stop-latency.5 design -- the merge's comparison of a part's exponents with the image's spends a unit per 8 words, COMPARED, 2^16, words at a time, and the merge MERGED + w units a term, MERGED = 16, w the words of a term's images, COMBINED terms at a time, not the comparison and the merge unspent (log 0058-1)
/// The words of exponents a merge compares between two spends.
const COMPARED: usize = 1 << 16;

// cofactor-stop-latency.5 design -- the merge's comparison of a part's exponents with the image's spends a unit per 8 words, COMPARED, 2^16, words at a time, and the merge MERGED + w units a term, MERGED = 16, w the words of a term's images, COMBINED terms at a time, not the comparison and the merge unspent (log 0058-1)
/// Whether a and b are equal, compared `COMPARED` words at a time, each
/// chunk found equal spending a unit of `budget` per 8 words and one more.
fn equal_spent(a: &[u32], b: &[u32], budget: &mut Budget) -> Result<bool, Stop> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (x, y) in a.chunks(COMPARED).zip(b.chunks(COMPARED)) {
        if x != y {
            return Ok(false);
        }
        budget.spend(x.len() / 8 + 1)?;
    }
    Ok(true)
}

/// As `merge`, in the order `order`. Two exponent vectors are packed into
/// `Dense32`'s words (`Format::pack`) before they are compared.
fn merge_in<O: Order<Dense32>>(
    part: &mut Part,
    image: &mut Sparse,
    second: Option<&mut Vec<u64>>,
    order: &O,
    variables: usize,
    budget: &mut Budget,
) -> Result<bool, Stop> {
    let format = Dense32::new(variables);
    let degree = |monomial: &[u32]| monomial.iter().sum::<u32>();
    let (mut part_words, mut image_words) =
        (vec![0u64; format.words()], vec![0u64; format.words()]);
    let (part_len, image_len) = (part.len(), image.coefs.len());
    let w = part.cap;
    let mut exponents = Vec::with_capacity((part_len + image_len) * variables);
    let mut images: Vec<u64> = Vec::with_capacity((part_len + image_len) * w);
    let mut coefs = Vec::with_capacity(part_len + image_len);
    // The second prime's coefficients, and beside them their new places.
    let mut second = second.map(|s| (s, Vec::with_capacity(part_len + image_len)));
    let (mut part_at, mut image_at) = (0, 0);
    let mut gained = false;
    while part_at < part_len || image_at < image_len {
        // cofactor-stop-latency.5 design -- the merge's comparison of a part's exponents with the image's spends a unit per 8 words, COMPARED, 2^16, words at a time, and the merge MERGED + w units a term, MERGED = 16, w the words of a term's images, COMBINED terms at a time, not the comparison and the merge unspent (log 0058-1)
        if coefs.len() % COMBINED == 0 {
            budget.spend(COMBINED * (MERGED + w))?;
        }
        let part_term = part_at * variables..(part_at + 1) * variables;
        let image_term = image_at * variables..(image_at + 1) * variables;
        let next = if image_at == image_len {
            Ordering::Greater
        } else if part_at == part_len {
            Ordering::Less
        } else {
            let part_exponents = &part.exponents[part_term.clone()];
            let image_exponents = &image.exponents[image_term.clone()];
            format.pack(part_exponents, &mut part_words);
            format.pack(image_exponents, &mut image_words);
            order.cmp(
                &format,
                &part_words,
                degree(part_exponents),
                &image_words,
                degree(image_exponents),
            )
        };
        // The image's term the merged term takes its coefficients from.
        let taken = match next {
            Ordering::Greater => {
                exponents.extend_from_slice(&part.exponents[part_term]);
                images.extend_from_slice(&part.images[part_at * w..(part_at + 1) * w]);
                part_at += 1;
                None
            }
            Ordering::Less => {
                exponents.extend_from_slice(&image.exponents[image_term]);
                images.resize(images.len() + w, 0);
                image_at += 1;
                gained = true;
                Some(image_at - 1)
            }
            Ordering::Equal => {
                exponents.extend_from_slice(&part.exponents[part_term]);
                images.extend_from_slice(&part.images[part_at * w..(part_at + 1) * w]);
                part_at += 1;
                image_at += 1;
                Some(image_at - 1)
            }
        };
        coefs.push(taken.map_or(0, |t| image.coefs[t]));
        if let Some((old, new)) = &mut second {
            new.push(taken.map_or(0, |t| old[t]));
        }
    }
    part.images = images;
    image.exponents.clone_from(&exponents);
    image.coefs = coefs;
    if let Some((old, new)) = second {
        *old = new;
    }
    part.exponents = exponents;
    Ok(gained)
}

// text-stop-latency.4 design -- the scaling of a cofactor term over Q spends SCALE·(m² + 4) units before it, SCALE = 16, m the larger of the limbs of the scaled numerator's factors summed and of the denominator, not nothing nor its bytes alone (log 0057-1)
/// Units of work per square limb of the scaling of a cofactor term over Q.
pub(crate) const SCALE: usize = 16;

// text-stop-latency.4 design -- the scaling of a cofactor term over Q spends SCALE·(m² + 4) units before it, SCALE = 16, m the larger of the limbs of the scaled numerator's factors summed and of the denominator, not nothing nor its bytes alone (log 0057-1)
/// The terms num/dens[k] of a cofactor part times l, each in lowest terms.
/// Each term spends `SCALE`·(m² + 4) units of `budget` before its scaling,
/// m the larger of the limbs of l and num summed and of dens[k]. A stop
/// where the budget's check stops the call.
#[allow(
    clippy::expect_used,
    reason = "every denominator of the chain is positive"
)]
pub(crate) fn scale_part(
    l: &BigInt,
    terms: &[(BigInt, u32)],
    dens: &[BigUint],
    budget: &mut Budget,
) -> Result<Vec<Rational>, Stop> {
    terms
        .iter()
        .map(|(num, k)| {
            let den = &dens[*k as usize];
            let m = (limbs(l.magnitude()) + limbs(num.magnitude())).max(limbs(den));
            budget.spend(SCALE * (m * m + 4))?;
            let den = BigInt::from(den.clone());
            Ok(Rational::new(l * num, den).expect("a denominator is positive"))
        })
        .collect()
}

/// The outcome of a complete lifting: each element scaled to integers with
/// no common factor, the leading one positive, as the canonical basis text;
/// where the call tracks cofactors, each row times the element's factor as
/// the cofactor matrix; and the checker's verdict where the call asks for
/// it. `stats` are the outcome's. The texts spend units of `budget` as they
/// are written: each integer of the basis text past its element's leading
/// one the square of its limbs as it is computed, then what `write_poly_q` spends; each cofactor entry
/// `ENTRY` units, its scaling what `scale_part` spends, then what
/// `write_cofactor_q` spends. Where the budget's check stops the call, the
/// outcome is the stop, with no text. The call report gets a line per text
/// and the checker's.
#[allow(
    clippy::too_many_arguments,
    reason = "the complete lifting, the instance, its parameters, order, and names, its counts, and the budget and primes it borrows, none of which belongs with another"
)]
fn finish(
    Complete(lifting): Complete<'_>,
    text: &str,
    params: &Params,
    order: &MonomialOrder,
    names: &[&str],
    stats: Stats,
    budget: &mut Budget,
    primes: &mut Primes,
) -> Outcome {
    let with_cofactors = params.cofactors || params.checker.is_some();
    budget.open();
    // result-time.12 design -- the basis text over Q is reserved once, before its `[`, at the sum over its terms of the monomial's longest text, a sign and a `*`, and the digits its integer's bits allow, not grown by doubling as a String grows (not measured)
    // The integer of a term num/dens[k] is num·(L/dens[k]), of at most
    // bits(num) + bits(L) − bits(dens[k]) + 1 bits, and the leading one is L.
    let mut bound = 4 + 2 * lifting.elements.len().saturating_sub(1);
    for element in &lifting.elements {
        let r = &element.rationals[0];
        let bits = r.last().bits();
        bound += term_bound_q(names, bits);
        for (num, den) in &r.terms {
            bound += term_bound_q(names, num.bits() + bits - r.dens[*den as usize].bits() + 1);
        }
    }
    let mut basis = String::with_capacity(reservation(bound));
    basis.push('[');
    let mut matrix = String::from("[");
    let mut exponents = Vec::new();
    let mut ints = Vec::new();
    for (i, element) in lifting.elements.iter().enumerate() {
        let kept = &element.rationals;
        if i > 0 {
            basis.push_str(",\n");
        }
        exponents.clear();
        exponents.extend_from_slice(&element.lead);
        exponents.extend_from_slice(&element.parts[0].exponents);
        // q-rational-time.6 design -- the basis text's integers are the numerators times the last denominator over their own, and the leading integer is the last denominator, not the lcm of every term's denominator folded by num-integer's gcd and a division per term (log 0028-1)
        // The last denominator L is the lcm of the denominators in lowest
        // terms, so the integers are those of `rational::scale`, with no
        // common factor.
        let l = BigInt::from(kept[0].last().clone());
        ints.clear();
        ints.push(l.clone());
        // text-stop-latency.12 design -- the integers of an element's basis text over Q are computed one at a time, each spending the square of its limbs as it is computed, not the integers computed for the whole element before its first spend (log 0057-1)
        for x in kept[0].integers() {
            if let Err(why) = budget.spend(decimal_units(x.magnitude())) {
                return stopped(why, stats);
            }
            ints.push(x);
        }
        // text-stop-latency.1 design -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
        if let Err(why) = write_poly_q(&mut basis, names, &exponents, &ints, budget) {
            return stopped(why, stats);
        }
        // call-report.23 design -- the basis text and the cofactor text over ℚ, written element by element in one loop, are timed apart by a mark after each element's part of each, not one time for both texts nor the texts written one after the other (not measured)
        budget.mark(Stage::Text);
        if !with_cofactors {
            continue;
        }
        if i > 0 {
            matrix.push_str(",\n");
        }
        matrix.push('[');
        for (j, (part, rationals)) in element.parts.iter().zip(kept).enumerate().skip(1) {
            if j > 1 {
                matrix.push(',');
            }
            // text-stop-latency.5 design -- each entry of a cofactor text, over GF(p) and over Q, spends ENTRY, 16, units beside its terms, not a tick per entry nor nothing (not measured)
            let written = budget
                .spend(ENTRY)
                .and_then(|()| scale_part(&l, &rationals.terms, &rationals.dens, budget))
                .and_then(|scaled| {
                    write_cofactor_q(&mut matrix, names, &part.exponents, &scaled, budget)
                });
            // text-stop-latency.1 design -- every text spends a unit of work a byte on the call's budget as it is written, and a writer stops where the budget's check stops the call, not a tick per polynomial or entry nor one full check after each text (log 0057-1)
            if let Err(why) = written {
                return stopped(why, stats);
            }
        }
        matrix.push(']');
        budget.mark(Stage::Cofactors);
    }
    basis.push_str("]:\n");
    matrix.push_str("]:\n");
    report_text(budget, "basis", basis.len(), Stage::Text);
    if with_cofactors {
        report_text(budget, "cofactors", matrix.len(), Stage::Cofactors);
    }
    // text-stop-latency.11 design -- no full check of the texts' own follows the last text, not a full check after the texts (not measured)
    // The checker, where the call asks for it, comes after the texts
    // and ticks the budget as it polls: a stop there drops them.
    let (mut certified, mut flaw) = (None, None);
    if let Some(mode) = params.checker {
        // q-lift.34 design -- the checker's seed over Q is the lift's generator's next draw after the verification, not the call's seed passed as it is (not measured)
        let seed = primes.draw();
        budget.open();
        let (verdict, why) = match certify(
            text,
            &basis,
            &matrix,
            order,
            names.len(),
            mode,
            seed,
            budget,
        ) {
            Ok(checked) => checked,
            Err(why) => return stopped(why, stats),
        };
        (certified, flaw) = (Some(verdict), why);
        budget.mark(Stage::Check);
        report_check(budget, mode, verdict);
    }
    Outcome {
        basis: Some(basis),
        stopped: None,
        cofactors: params.cofactors.then_some(matrix),
        certified,
        flaw,
        stats,
        report: None,
    }
}

/// As a call over Q, with the primes `forced` first: the order named, the
/// budget started, the header read, and `lift`, with `NoTrace`, the
/// reconstruction over the chain, and the learn runs at one prime each
/// (q-learn-time.11). For the crate's unit tests.
#[cfg(test)]
pub(crate) fn lift_text(
    text: &str,
    params: &Params,
    forced: Vec<u64>,
    report: &mut Report,
) -> Result<Outcome, Error> {
    lift_text_with(text, params, forced, Reconstruct::Chain, report)
}

// q-rational-time.8 design -- the report counts the coefficients tried that ran the Euclidean algorithm, and a test switch makes a lift run Wang's reconstruction on every coefficient, so that the test code compares the two lifts exactly, not the test code timing the call or counts frozen in it (not measured)
/// As `lift_text`, with the attempts reconstructing as `how` says. The
/// learn runs run at one prime each (q-learn-time.11).
#[cfg(test)]
pub(crate) fn lift_text_with(
    text: &str,
    params: &Params,
    forced: Vec<u64>,
    how: Reconstruct,
    report: &mut Report,
) -> Result<Outcome, Error> {
    lift_text_as(
        text,
        params,
        forced,
        how,
        Replays::Pairs,
        Learns::Alone,
        &mut crate::trace::NoTrace,
        report,
    )
}

/// As `lift_text_with`, the replays run as `replays` says, the learn runs
/// as `learns` says, and every run telling `trace` its matrices. The call
/// report's lines of the call, its read, and its end are those
/// `entry::call` writes. For the crate's unit tests.
#[cfg(test)]
#[allow(
    clippy::too_many_arguments,
    reason = "the instance, its parameters and forced primes, the choices of the reconstruction, the replays, and the learn runs, and the trace and report the test reads, none of which belongs with another"
)]
pub(crate) fn lift_text_as<T: Trace>(
    text: &str,
    params: &Params,
    forced: Vec<u64>,
    how: Reconstruct,
    replays: Replays,
    learns: Learns,
    trace: &mut T,
    report: &mut Report,
) -> Result<Outcome, Error> {
    let start = std::time::Instant::now();
    let mut budget = Budget::new(start, params)?;
    report_call(&mut budget, params, "groebner", T::LISTENS);
    let header = Header::read(text)?;
    report_read(&mut budget, &header, "q");
    let mut primes = Primes::new(params.seed, forced);
    let outcome = lift(
        text,
        params,
        &params.order,
        &header,
        &mut budget,
        trace,
        &mut primes,
        how,
        replays,
        learns,
        report,
    )?;
    Ok(report_end(&mut budget, outcome))
}
