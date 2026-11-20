//! The helpers that two or more of the root package's unit test files take, each defined once
//! here and taken by `use` (`test-helper-copies`; `iterations/0147-hygiene/plan.md`, _Approach_).

// hygiene.9 design-checked -- the unit tests' shared helpers stand in this one file, not each kept in the first file that defines it and made `pub(crate)` (not measured)
// hygiene.20 design-checked -- the helpers and types that this file's definitions name and that stand once in another test file, `reported`, `of_bits`, `RefPair`, `Line`, and `lines`, are taken from there, not moved here with their importers (not measured)

use std::cmp::Ordering;
use std::collections::HashSet;
use std::convert::Infallible;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use rand_core::{Rng, SeedableRng, TryRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::divisibility_tests::RefPair;
use crate::entry::{Imaged, image_in};
use crate::field::{BAND, Field, Gf31, is_prime_64};
use crate::halt::Halt;
use crate::id::MonoId;
use crate::lift::{Report, lift_text};
use crate::matrix::{self, Buffers, Combine, Matrix, Reduced, Reduction};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::mqrr_tests::of_bits;
use crate::poly::Basis;
use crate::rebuilt_reducer_tests::reported;
use crate::report_tests::{Line, lines};
use crate::systems::{dense, random_square, random_system};
use crate::table::Table;
use crate::text::Header;
use crate::trace::{NoTrace, Record, Recording};
use crate::{Error, MonomialOrder, Outcome, Params, Threads};

/// 65521, the largest prime below 2^16.
pub(crate) const P: u64 = 65521;
/// 2^31 − 1, the largest prime below 2^31.
pub(crate) const P31: u64 = 2_147_483_647;
/// The Mersenne prime 2^61 − 1, past 2^31.
pub(crate) const P61: u64 = (1 << 61) - 1;

/// A field of each coefficient width and ℚ: the primes 7 and 65521 below 2^16, 2^31 − 1 below
/// 2^31, 2^61 − 1 past it, and 0 for ℚ.
pub(crate) const FIELDS: [u64; 5] = [7, 65521, 2_147_483_647, 2_305_843_009_213_693_951, 0];

/// Three primes of `Gf31`: 7, below the combinations' bound of 46341; 65521, below 2^16, where the
/// row operations delay their reduction; and 2^31 − 1, the largest below 2^31, whose band corrects
/// each product.
pub(crate) const GF31_PRIMES: [u64; 3] = [7, 65521, 2_147_483_647];

/// The primes of the combinations: below 2^16 and the largest below 2^31, each past the
/// combinations' bound of 46341.
pub(crate) const COMBINING: [u64; 2] = [65521, 2_147_483_647];

/// The primes of the random systems: 2, 3, 7, 65521, and 2^31 − 1.
pub(crate) const RANDOM_PRIMES: [u64; 5] = [2, 3, 7, 65521, 2_147_483_647];

/// The four largest primes below 2^31, from the top down, lucky for every case of the lift's
/// tests that take them.
pub(crate) const L1: u64 = 2_147_483_647;
pub(crate) const L2: u64 = 2_147_483_629;
pub(crate) const L3: u64 = 2_147_483_587;
pub(crate) const L4: u64 = 2_147_483_579;
/// The two least primes past 2^30.5, which the lift's tests that take them make unlucky: a case
/// makes a coefficient vanish at one of them.
pub(crate) const U1: u64 = 1_518_500_279;
pub(crate) const U2: u64 = 1_518_500_293;
/// The three least primes past 2^61 + 2^60, which the lift's tests that take them make unlucky
/// where they need to.
pub(crate) const W1: u64 = 3_458_764_513_820_540_933;
pub(crate) const W2: u64 = 3_458_764_513_820_541_017;
pub(crate) const W3: u64 = 3_458_764_513_820_541_041;
/// The least prime past 2^61 and the largest below 2^62.
pub(crate) const PAST_61: u64 = 2_305_843_009_213_693_967;
pub(crate) const BELOW_62: u64 = 4_611_686_018_427_387_847;

/// Two primes of [2^30, 2^31), the lanes of a pair learn's `Gf31Pair`.
pub(crate) const LEARN0: u64 = 1_945_203_761;
pub(crate) const LEARN1: u64 = 1_875_939_437;
/// Two primes of [2^61, 2^62), the lanes of a pair replay's `Gf64Pair`.
pub(crate) const REPLAY0: u64 = 2_536_775_269_993_504_451;
pub(crate) const REPLAY1: u64 = 4_026_501_963_682_786_037;

/// Every lift's deadline in the lift's tests, far past what each takes: a fault that keeps the
/// lift from completing, or has it learn afresh without end, stops it there and fails the test. A
/// fault that makes the replays misfit, or makes the lift learn afresh, ends it in `Error::Drops`
/// before.
pub(crate) const LIFT_DEADLINE: Duration = Duration::from_secs(60);

// hygiene.16 design-checked -- `format_tests.rs`' 200 ms deadline, which the copies of the GF(2) tests name as theirs, joins them here though its tokens differ, not left in `format_tests.rs` as a constant of its own (not measured)
/// The deadline of each call of a case that a test compares in two ways, in two widths or with the
/// indexes and without: a case whose call it stops is left out of the comparison.
pub(crate) const CASE_DEADLINE: Duration = Duration::from_millis(200);

/// The split turned off in the test profile: `SPLIT_BOUNDS` with a least that no pair set reaches.
pub(crate) const OFF: Option<(usize, usize)> = Some((usize::MAX, 1));

/// The variable counts of each format: whole words, a last word with one lane, and a last word
/// full.
pub(crate) const COUNTS8: [usize; 9] = [1, 2, 7, 8, 9, 16, 17, 24, 54];
pub(crate) const COUNTS16: [usize; 5] = [1, 3, 4, 5, 9];
pub(crate) const COUNTS32: [usize; 4] = [1, 2, 3, 5];

/// The hashes of a table's edge cases, by name (`iterations/0031-product-lookup-time/plan.md`,
/// _Edge cases_).
pub(crate) const HASHES: [&str; 4] = [
    "drawn",
    // Every hash's low 32 bits are 0: every slot's tag agrees.
    "tags that agree",
    // Every hash is 0: one probe chain from slot 0.
    "one chain",
    // Every hash is 2^64 minus the degree, the unit monomial's 0: every
    // other home slot is the last, and every chain from there wraps past
    // the end.
    "chains that wrap",
];

// hygiene.15 design-checked -- the signature worlds' variables take a name of their own beside `divisor_units_tests.rs`' `N`, the variables of its generator, whose tokens agree with theirs, not one constant for both (not measured)
/// The variables of the signature tests' random worlds.
pub(crate) const SIG_VARIABLES: usize = 3;
/// The indices of their signatures.
pub(crate) const SIG_INDICES: u32 = 2;

/// Hands out the given words in turn, then 0: the hash multipliers of a table, which
/// `Table::new` draws one per variable, in order.
pub(crate) struct Given(pub(crate) std::vec::IntoIter<u64>);

impl TryRng for Given {
    type Error = Infallible;

    #[allow(
        clippy::cast_possible_truncation,
        reason = "the low half of the word is the word drawn"
    )]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.try_next_u64()? as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.0.next().unwrap_or(0))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        dst.fill(0);
        Ok(())
    }
}

// hygiene.17 design-checked -- the two files' random matrix shares its fields and its reduction here, and each file keeps its own constructor under a name of its own, not one constructor for both, as the two differ beyond a number literal (not measured)
/// A random matrix over GF(p) on four variables: its field, its basis of random monic
/// polynomials, and the matrix of the rows drawn from the basis with the reducers symbolic
/// preprocessing adds. `combine_tests.rs` draws it by `Random::drawn` and `plain_band_tests.rs` by
/// `Random::shaped`.
pub(crate) struct Random {
    pub(crate) field: Gf31,
    pub(crate) basis: Basis<Gf31, Rank1>,
    pub(crate) matrix: Matrix,
}

impl Random {
    /// The plain run's reduction of the matrix without cofactors, under a budget with no bound.
    pub(crate) fn reduce(&self, combine: Option<&mut Combine>) -> Reduction<Gf31> {
        self.reduce_in(combine, &mut budget()).unwrap()
    }

    /// As `reduce`, under `b`.
    pub(crate) fn reduce_in(
        &self,
        combine: Option<&mut Combine>,
        b: &mut Budget,
    ) -> Result<Reduction<Gf31>, Halt> {
        matrix::reduce::<Gf31, Dense16, Rank1, NoCofactors>(
            &self.field,
            &self.basis,
            &self.matrix,
            &mut NoCofactors,
            b,
            true,
            combine,
            &mut Buffers::new(),
            None,
            None,
        )
    }
}

// hygiene.18 design-checked -- groups of copies whose helpers give the same for every caller became one definition here though their tokens differ: `below` of `r` and of `rng`, `budget` of four groups and `generous`, `params` and `params()` as `params(1)`, `passed_after` written through `passed` and out, `lift_params` of the 60 s deadline written out, of `LIFT_DEADLINE`, and of `chain_tests.rs`' `params`, and `exponents` and `exps`; not one shared definition under a name of its own for each group of tokens (not measured)
/// The parameters of a call in grevlex on one thread under `seed`, every bound absent.
pub(crate) fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

/// `params(seed)` with the lift's deadline, `LIFT_DEADLINE`.
pub(crate) fn lift_params(seed: u64) -> Params {
    let mut p = params(seed);
    p.deadline = Some(LIFT_DEADLINE);
    p
}

/// `lift_params(seed)` with cofactors.
pub(crate) fn with_cofactors(seed: u64) -> Params {
    let mut p = lift_params(seed);
    p.cofactors = true;
    p
}

/// A budget with no bound.
pub(crate) fn budget() -> Budget {
    Budget::new(Instant::now(), &params(1)).unwrap()
}

/// A budget with no bound under `rebuilt_reducer_tests::reported` in grevlex: its report on, the
/// signatures off.
pub(crate) fn reported_budget() -> Budget {
    Budget::new(Instant::now(), &reported(MonomialOrder::Grevlex)).unwrap()
}

/// A budget whose deadline passed before the test began, and which has not
/// been checked.
pub(crate) fn passed() -> Budget {
    let mut p = params(1);
    p.deadline = Some(Duration::from_millis(1));
    Budget::new(
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        &p,
    )
    .unwrap()
}

/// `passed` with `units` of work between two full checks.
pub(crate) fn passed_after(units: u64) -> Budget {
    let mut b = passed();
    b.set_work(units);
    b
}

/// The generator of a test's draws under `seed`.
pub(crate) fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// A draw of `rng` below `n`.
pub(crate) fn below(rng: &mut Xoshiro256PlusPlus, n: u64) -> u64 {
    rng.next_u64() % n
}

/// Drawn exponents of n variables of degree at most `bound`: most 0 to 3,
/// some up to a tenth of the bound.
pub(crate) fn drawn_exponents(rng: &mut impl Rng, n: usize, bound: u32) -> Vec<u32> {
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

/// The exponent vectors that agree with `e` before `at` and spend `left` from `at` on, appended
/// to `out`.
pub(crate) fn fill(e: &mut [u32], at: usize, left: u32, out: &mut Vec<Vec<u32>>) {
    if at == e.len() - 1 {
        e[at] = left;
        out.push(e.to_vec());
        return;
    }
    for k in 0..=left {
        e[at] = k;
        fill(e, at + 1, left - k, out);
    }
}

/// Whether a divides b, exponent by exponent.
pub(crate) fn divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

/// Whether a and b share no variable.
pub(crate) fn coprime(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| *x == 0 || *y == 0)
}

/// The exponents as signed integers.
pub(crate) fn signed(e: &[u32]) -> Vec<i64> {
    e.iter().map(|&x| i64::from(x)).collect()
}

/// a − b, exponent by exponent.
pub(crate) fn minus(a: &[u32], b: &[u32]) -> Vec<i64> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) - i64::from(y))
        .collect()
}

/// Grevlex on exponent vectors of any sign: the degree first, then the last
/// variable where they differ, the smaller exponent the larger monomial.
pub(crate) fn grevlex(a: &[i64], b: &[i64]) -> Ordering {
    let (da, db): (i64, i64) = (a.iter().sum(), b.iter().sum());
    da.cmp(&db).then_with(|| {
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return b[i].cmp(&a[i]);
            }
        }
        Ordering::Equal
    })
}

/// Schreyer's order on (exponents, index) (signature-pruning.2).
pub(crate) fn sig_cmp(a: (&[u32], u32), b: (&[u32], u32)) -> Ordering {
    grevlex(&signed(a.0), &signed(b.0)).then(a.1.cmp(&b.1))
}

/// Whether the signature a divides b: one index, and a's exponents divide b's.
pub(crate) fn sig_divides(a: (&[u32], u32), b: (&[u32], u32)) -> bool {
    a.1 == b.1 && divides(a.0, b.0)
}

/// The n multipliers of the hashes `name`.
pub(crate) fn multipliers(name: &str, n: usize, seed: u64) -> Vec<u64> {
    let mut r = rng(seed ^ n as u64);
    (0..n as u64)
        .map(|i| match name {
            "drawn" => r.next_u64(),
            "tags that agree" => (i + 1) << 32,
            "one chain" => 0,
            "chains that wrap" => u64::MAX,
            // Every hash's low 31 bits are 0, its bit 31 the parity of
            // Σ (i + 1)·e_i: every slot's tag agrees in its low 31 bits.
            "tags that agree in 31 bits" => (i + 1) << 31,
            // Every hash is −2^52 times the degree and a little: the home
            // slots of every degree but 0 lie among the slot array's last.
            "homes near the end" => (u64::MAX << 52) | (i + 1),
            _ => unreachable!("{name}"),
        })
        .collect()
}

/// A table on `n` variables of format `M` whose hash multipliers are
/// `multipliers`.
pub(crate) fn table<M: Format>(n: usize, multipliers: Vec<u64>) -> Table<M> {
    Table::new(M::new(n), &mut Given(multipliers.into_iter()))
}

/// A table of the signature tests' random worlds, on `SIG_VARIABLES` variables, its hash
/// multipliers drawn under `seed`.
pub(crate) fn sig_table(seed: u64) -> Table<Dense16> {
    Table::new(Dense16::new(SIG_VARIABLES), &mut rng(seed ^ 0x5197))
}

/// The id of the monomial `exps` in `table`, interned there where it is not.
pub(crate) fn id(table: &mut Table<Dense16>, exps: &[u32]) -> MonoId {
    table.intern(exps, &mut Budget::unbounded()).unwrap()
}

/// The exponents of the monomial `id` of `t`, one per variable.
pub(crate) fn exponents<M: Format>(t: &Table<M>, id: MonoId) -> Vec<u32> {
    (0..t.format().variables())
        .map(|i| t.exponent(id, i))
        .collect()
}

// hygiene.10 practice-checked: Parameterize Function (Fowler's refactoring) -- the two files' copies, which differ in the variables alone, are one `fn` that takes them, not one copy's six variables for both files' tests (not measured)
/// `n` distinct monomials of `Dense16` on `variables` variables, of degree
/// 24 or less, in the random order drawn, in a table of their own.
#[allow(
    clippy::cast_possible_truncation,
    reason = "each draw lies below 25 or below the degree left plus one, which a u32 holds"
)]
pub(crate) fn monomials(variables: usize, n: usize, seed: u64) -> (Table<Dense16>, Vec<MonoId>) {
    let mut t = Table::new(Dense16::new(variables), &mut rng(seed));
    let mut r = rng(seed ^ 0x6d6f);
    let mut taken = HashSet::new();
    let mut ids = Vec::with_capacity(n);
    while ids.len() < n {
        let mut e = vec![0u32; variables];
        let mut d = (r.next_u64() % 25) as u32;
        for x in &mut e {
            let k = (r.next_u64() % (u64::from(d) + 1)) as u32;
            *x = k;
            d -= k;
        }
        let id = t.intern(&e, &mut Budget::unbounded()).unwrap();
        if taken.insert(id) {
            ids.push(id);
        }
    }
    (t, ids)
}

/// A random integer below 2^bits.
pub(crate) fn integer(rng: &mut Xoshiro256PlusPlus, bits: u64) -> BigUint {
    let mut x = BigUint::zero();
    for _ in 0..bits.div_ceil(64) {
        x = (x << 64u32) + BigUint::from(rng.next_u64());
    }
    x >> (bits.div_ceil(64) * 64 - bits)
}

/// A random odd integer of exactly `bits` bits prime to every one of
/// `others`.
pub(crate) fn drawn_coprime(
    rng: &mut Xoshiro256PlusPlus,
    bits: u64,
    others: &[&BigUint],
) -> BigUint {
    loop {
        let x = of_bits(rng, bits) | BigUint::one();
        if others.iter().all(|o| x.gcd(o).is_one()) {
            return x;
        }
    }
}

/// x on limbs, the least significant first, no zero limb at the top.
pub(crate) fn limbs(x: &BigUint) -> Vec<u64> {
    x.to_u64_digits()
}

/// The product of `primes`.
pub(crate) fn product(primes: &[u64]) -> BigUint {
    primes.iter().fold(BigUint::one(), |m, &p| m * p)
}

/// n·d⁻¹ mod m; None where d is no unit mod m.
pub(crate) fn image(n: &BigInt, d: &BigUint, m: &BigUint) -> Option<BigUint> {
    let m = BigInt::from(m.clone());
    let e = BigInt::from(d.clone()).extended_gcd(&m);
    if !e.gcd.is_one() {
        return None;
    }
    (n * e.x).mod_floor(&m).to_biguint()
}

/// x mod p, x written in decimal.
pub(crate) fn residue_of(digits: &str, p: u64) -> u64 {
    (BigUint::parse_bytes(digits.as_bytes(), 10).unwrap() % p)
        .to_u64()
        .unwrap()
}

/// A random prime of [2^30, 2^31) not among `used`, by `Gf31::new`'s test.
pub(crate) fn prime(rng: &mut Xoshiro256PlusPlus, used: &[u64]) -> u64 {
    loop {
        let mut c = (1u64 << 30) + (rng.next_u64() >> 34);
        while c < 1 << 31 && Gf31::new(c).is_none() {
            c += 1;
        }
        if c < 1 << 31 && !used.contains(&c) {
            return c;
        }
    }
}

/// The least prime at or above n.
pub(crate) fn prime_from(n: u64) -> u64 {
    (n..).find(|&q| is_prime_64(q)).unwrap()
}

/// The primes from 101 below `limit`, by trial division.
pub(crate) fn small_primes(limit: u64) -> Vec<u64> {
    (101..limit)
        .filter(|&n| (2..).take_while(|d| d * d <= n).all(|d| n % d != 0))
        .collect()
}

/// The lift of `text` with the primes `forced` first: its outcome, which
/// must hold a basis.
pub(crate) fn lift(text: &str, params: &Params, forced: &[u64]) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text(text, params, forced.to_vec(), &mut report)
        .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    outcome
}

/// The basis of `outcome`, which must hold one.
pub(crate) fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// The record of a learn run over `text`, an instance over Q, at the
/// prime `p` below 2^31.
pub(crate) fn learn(text: &str, params: &Params, p: u64) -> Record {
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    let mut recording = Recording::new();
    let got = image_in(
        Gf31::prime(p),
        text,
        params,
        None,
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        &mut recording,
        None,
    )
    .unwrap_or_else(|e| panic!("the learn run at {p} failed: {e:?}\non\n{text}"));
    assert!(
        matches!(got, Imaged::Image(_)),
        "the learn run at {p} stopped"
    );
    recording.finish(named)
}

/// What `image_in` returned, for a message.
pub(crate) fn what(got: &Result<Imaged, Error>) -> String {
    match got {
        Ok(Imaged::Image(_)) => "an image".to_string(),
        Ok(Imaged::Lanes(images)) => format!("{len} lanes", len = images.len()),
        Ok(Imaged::Pair(_)) => "a pair image".to_string(),
        Ok(Imaged::Stopped(o)) => format!("a stop: {o:?}"),
        Err(e) => format!("{e:?}"),
    }
}

/// The made rows, each as its columns and coefficients, by leading column.
pub(crate) fn rows(reduction: &Reduction<Gf31>) -> Vec<(Vec<MonoId>, Vec<u32>)> {
    let mut made: Vec<&Reduced<Gf31>> = reduction.made.iter().collect();
    made.sort_by_key(|r| r.cols[0]);
    made.iter()
        .map(|r| (r.cols.clone(), r.coefs.clone()))
        .collect()
}

/// Every entry of the band and of the dense rows read by `take`: the
/// values agree, and both hold zeros after.
pub(crate) fn read_all(
    f: &Gf31,
    band: &mut [[i64; BAND]],
    dense: &mut [Vec<i64>; BAND],
    case: &str,
) {
    for (c, entry) in band.iter_mut().enumerate() {
        for k in 0..BAND {
            let a = f.take(&mut entry[k]);
            let b = f.take(&mut dense[k][c]);
            assert_eq!(a, b, "{case}: column {c}, band row {k}");
        }
    }
    assert!(band.iter().all(|e| e.iter().all(|&x| x == 0)), "{case}");
}

/// The reference's selection: the pairs of least sugar, in its order,
/// taken out of `pairs`, the rest left in its order.
pub(crate) fn select_reference(pairs: &mut Vec<RefPair>) -> Vec<RefPair> {
    let Some(d) = pairs.iter().map(|p| p.sugar).min() else {
        return Vec::new();
    };
    let (selected, rest): (Vec<RefPair>, Vec<RefPair>) =
        pairs.drain(..).partition(|p| p.sugar == d);
    *pairs = rest;
    selected
}

/// The report's lines of `outcome`.
pub(crate) fn report(outcome: &Outcome) -> Vec<Line> {
    lines(
        outcome
            .report
            .as_deref()
            .expect("a call with the report on"),
    )
}

/// The `round` lines of the plain runs: those after a `run` line whose
/// engine is `plain`.
pub(crate) fn plain_rounds(lines: &[Line]) -> Vec<&Line> {
    let mut plain = false;
    let mut out = Vec::new();
    for l in lines {
        if l.kind == "run" {
            plain = l.get("engine") == Some("plain");
        } else if l.kind == "round" && plain {
            out.push(l);
        }
    }
    out
}

// hygiene.10 practice-checked: Parameterize Function (Fowler's refactoring) -- the three files' copies, which differ in their counts of seeds alone, are one `fn` that takes them, not one copy's counts for every file's tests (not measured)
/// Random systems and random square systems over `p` of the seeds below
/// `seeds`, and dense cubics on three variables of the seeds below
/// `cubics`.
pub(crate) fn systems(p: u64, seeds: u64, cubics: u64) -> Vec<String> {
    let mut texts: Vec<String> = (0..seeds)
        .flat_map(|seed| [random_system(p, seed), random_square(p, seed)])
        .collect();
    texts.extend((0..cubics).map(|seed| dense(3, 3, p, seed)));
    texts
}
