//! q-learn-time, the lift's test code: `Gf31Pair`'s lanes against `Gf31`
//! at each prime, a pair learn against the learn run at its first prime
//! and the run at its second, the disagreements that halt a pair learn,
//! the lift in pairs against the lift alone, the fallbacks, and the edge
//! cases in both modes (plan.md, _Test code_, 1 to 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::entry::{Imaged, image_in};
use crate::field::{Field, Gf31, Gf31Pair};
use crate::id::MonoId;
use crate::lift::{Image, Learns, Reconstruct, Replays, Report, lift_text_as};
use crate::systems::{dense, random_square};
use crate::text::Header;
use crate::trace::{NoLearner, NoTrace, Record, Recording};
use crate::{Error, MonomialOrder, Outcome, Params, Threads};

/// Primes below 2^31 from the top down, and the least past 2^30.
const L1: u64 = 2_147_483_647;
const L2: u64 = 2_147_483_629;
const L3: u64 = 2_147_483_587;
const L4: u64 = 2_147_483_579;
const N1: u64 = 1_073_741_827;

/// Every lift's deadline, far past what each takes: a fault that keeps the
/// lift from completing stops it there and fails the test. A fault that
/// makes it learn afresh ends it in `Error::Drops` before.
const DEADLINE: Duration = Duration::from_secs(60);

/// The systems over Q of tests 2 and 4 in grevlex, here and in
/// `later_runs_tests.rs`: random square systems on two to four variables,
/// and dense cubic systems on two and three.
pub(crate) fn systems() -> Vec<(String, String)> {
    let mut list: Vec<(String, String)> = (1..=3)
        .map(|seed| (format!("square-{seed}"), random_square(0, seed)))
        .collect();
    for m in [2, 3] {
        list.push((format!("cubic-{m}"), dense(m, 3, 0, m as u64)));
    }
    list
}

// test-mq-systems.11 design -- the system in lex and block:2 is random_square(0, 0), three variables of degrees 3, 2, and 1, not the dense cubic system on three variables, whose lift in block:2 with cofactors runs past DEADLINE (log 0084-2)
/// The system over Q of tests 2 and 4 in lex and `block:2`: three
/// variables, of which `block:2` leaves one to its second block.
fn ordered() -> String {
    random_square(0, 0)
}

fn params_in(order: &str, seed: u64) -> Params {
    let mut p = Params::new(
        order
            .parse::<MonomialOrder>()
            .expect("a name of the family"),
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    p.deadline = Some(DEADLINE);
    p
}

fn params(seed: u64) -> Params {
    params_in("grevlex", seed)
}

fn with_cofactors_in(order: &str, seed: u64) -> Params {
    let mut p = params_in(order, seed);
    p.cofactors = true;
    p
}

fn basis(outcome: &Outcome) -> &str {
    outcome.basis.as_deref().unwrap()
}

/// What `image_in` returned, for a message.
fn what(got: &Result<Imaged, Error>) -> String {
    match got {
        Ok(Imaged::Image(_)) => "an image".to_string(),
        Ok(Imaged::Lanes(images)) => format!("{len} lanes", len = images.len()),
        Ok(Imaged::Pair(_)) => "a pair image".to_string(),
        Ok(Imaged::Stopped(o)) => format!("a stop: {o:?}"),
        Err(e) => format!("{e:?}"),
    }
}

/// A run over `text`, an instance over Q, in `field`, with no record,
/// telling `learner` what it records.
fn run_in<F: Field, L: crate::trace::Learner>(
    field: F,
    text: &str,
    params: &Params,
    learner: &mut L,
) -> Result<Imaged, Error> {
    let header = Header::read(text).unwrap();
    let named = params.order.clone();
    let mut budget = Budget::new(Instant::now(), params).unwrap();
    image_in(
        field,
        text,
        params,
        None,
        &named,
        &header,
        &mut budget,
        &mut NoTrace,
        learner,
        None,
    )
}

/// The pair learn at (p, q): its two images and its record, or what it
/// returned instead.
fn pair_learn(
    text: &str,
    params: &Params,
    (p, q): (u64, u64),
) -> Result<(Image, Image, Record), Error> {
    let mut recording = Recording::new();
    match run_in(Gf31Pair::primes(p, q), text, params, &mut recording) {
        Ok(Imaged::Lanes(mut images)) => {
            assert_eq!(images.len(), 2, "the pair ({p}, {q}): an image per lane");
            let second = images.pop().unwrap();
            let first = images.pop().unwrap();
            Ok((first, second, recording.finish(params.order.clone())))
        }
        Err(e) => Err(e),
        other @ Ok(Imaged::Image(_) | Imaged::Pair(_) | Imaged::Stopped(_)) => panic!(
            "the pair ({p}, {q}) returned no lanes: {what}",
            what = what(&other)
        ),
    }
}

/// The learn run at the prime p alone: its image and its record.
fn learn_alone(text: &str, params: &Params, p: u64) -> (Image, Record) {
    let mut recording = Recording::new();
    match run_in(Gf31::prime(p), text, params, &mut recording) {
        Ok(Imaged::Image(image)) => (image, recording.finish(params.order.clone())),
        other @ (Ok(Imaged::Lanes(_) | Imaged::Pair(_) | Imaged::Stopped(_)) | Err(_)) => panic!(
            "the learn run at {p} returned no image: {what}",
            what = what(&other)
        ),
    }
}

/// The run at the prime q with no learner, as the verification's.
fn run_at(text: &str, params: &Params, q: u64) -> Image {
    match run_in(Gf31::prime(q), text, params, &mut NoLearner) {
        Ok(Imaged::Image(image)) => image,
        other @ (Ok(Imaged::Lanes(_) | Imaged::Pair(_) | Imaged::Stopped(_)) | Err(_)) => panic!(
            "the run at {q} returned no image: {what}",
            what = what(&other)
        ),
    }
}

/// The lift of `text` with the primes `forced` first, learning as `learns`
/// says, the replays in pairs: its outcome, which must hold a basis.
fn lift(text: &str, params: &Params, forced: &[u64], learns: Learns) -> Outcome {
    let mut report = Report::default();
    let outcome = lift_text_as(
        text,
        params,
        forced.to_vec(),
        Reconstruct::Chain,
        Replays::Pairs,
        learns,
        &mut NoTrace,
        &mut report,
    )
    .unwrap_or_else(|e| panic!("the lift failed: {e:?}\non\n{text}\n{report:?}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "the lift stopped: {outcome:?}\n{report:?}"
    );
    outcome
}

/// x mod p, x written in decimal.
fn residue_of(digits: &str, p: u64) -> u64 {
    (BigUint::parse_bytes(digits.as_bytes(), 10).unwrap() % p)
        .to_u64()
        .unwrap()
}

// --- 1: the lanes ---

/// Test 1, q-learn-time.2, .3: each lane of `Gf31Pair` computes as `Gf31`
/// at its prime, below 2^16 and past it: every operation, the inverse
/// None exactly where a lane is 0, and a dense row's values after rows
/// loaded and subtracted, each entry in [0, `p_k²`) between two operations.
#[test]
fn each_lane_is_gf31() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x31a2);
    for (p, q) in [(L1, N1), (N1, L2), (65521, 7), (3, L1)] {
        let f = Gf31Pair::primes(p, q);
        let g = [Gf31::prime(p), Gf31::prime(q)];
        let primes = [p, q];
        let draw = |rng: &mut Xoshiro256PlusPlus| -> [u32; 2] {
            [(rng.next_u64() % p) as u32, (rng.next_u64() % q) as u32]
        };
        for _ in 0..2000 {
            let (a, b) = (draw(&mut rng), draw(&mut rng));
            let v = rng.next_u64();
            assert_eq!(f.coef(v), [g[0].coef(v), g[1].coef(v)]);
            assert_eq!(f.add(a, b), [g[0].add(a[0], b[0]), g[1].add(a[1], b[1])]);
            assert_eq!(f.sub(a, b), [g[0].sub(a[0], b[0]), g[1].sub(a[1], b[1])]);
            assert_eq!(f.neg(a), [g[0].neg(a[0]), g[1].neg(a[1])]);
            assert_eq!(f.mul(a, b), [g[0].mul(a[0], b[0]), g[1].mul(a[1], b[1])]);
            assert_eq!(f.value(a), u64::from(a[0]));
            for k in 0..2 {
                assert_eq!(f.lane(a, k), u64::from(a[k]));
            }
            let both = a[0] != 0 && a[1] != 0;
            assert_eq!(f.is_unit(a), both, "{a:?}");
            match f.inv(a) {
                Some(i) => {
                    assert!(both, "{a:?}");
                    assert_eq!(i, [g[0].inv(a[0]).unwrap(), g[1].inv(a[1]).unwrap()]);
                }
                None => assert!(!both, "{a:?}"),
            }
        }
        for k in 0..2 {
            let mut lane_zero = [1, 1];
            lane_zero[k] = 0;
            assert!(!f.is_unit(lane_zero), "({p}, {q}), lane {k} at 0");
            assert!(f.inv(lane_zero).is_none(), "({p}, {q}), lane {k} at 0");
        }
        assert!(f.inv([0, 0]).is_none());
        assert_eq!(f.characteristic(), p);
        assert_eq!(<Gf31Pair as Field>::LANES, 2);
        for len in [1, 18, 19, 20, 38, 39, 40, 60] {
            let digits: String = (0..len)
                .map(|_| char::from(b'0' + (rng.next_u64() % 10) as u8))
                .collect();
            assert_eq!(
                f.decimal(&digits),
                [residue_of(&digits, p) as u32, residue_of(&digits, q) as u32],
                "({p}, {q}): {digits}"
            );
        }
        // Rows into one dense row: the pair's, and each lane's alone.
        let width = 97;
        let mut dense = vec![[0i64; 2]; width];
        let mut single = [vec![0i64; width], vec![0i64; width]];
        let start: Vec<MonoId> = (0..width as u32)
            .filter(|c| c % 3 == 0)
            .map(MonoId::from_raw)
            .collect();
        let start_coefs: Vec<[u32; 2]> = start.iter().map(|_| draw(&mut rng)).collect();
        // SAFETY: every column lies below the width, a coefficient each,
        // and the dense rows are zero where the row loads.
        unsafe {
            f.load(&mut dense, &start, &start_coefs);
            for k in 0..2 {
                let lane: Vec<u32> = start_coefs.iter().map(|c| c[k]).collect();
                g[k].load(&mut single[k], &start, &lane);
            }
        }
        for _ in 0..200 {
            let cols: Vec<MonoId> = (0..width as u32)
                .filter(|_| rng.next_u64() % 4 == 0)
                .map(MonoId::from_raw)
                .collect();
            let coefs: Vec<[u32; 2]> = cols.iter().map(|_| draw(&mut rng)).collect();
            let c = draw(&mut rng);
            // SAFETY: as above.
            unsafe {
                f.sub_mul(&mut dense, &cols, &coefs, c);
                for k in 0..2 {
                    let lane: Vec<u32> = coefs.iter().map(|v| v[k]).collect();
                    g[k].sub_mul(&mut single[k], &cols, &lane, c[k]);
                }
            }
            for (col, e) in dense.iter().enumerate() {
                for k in 0..2 {
                    let bound = (primes[k] * primes[k]) as i64;
                    assert!(
                        (0..bound).contains(&e[k]),
                        "({p}, {q}): column {col}, lane {k}, {entry}",
                        entry = e[k]
                    );
                }
            }
        }
        for col in 0..width {
            let mut e = dense[col];
            let got = f.take(&mut e);
            assert_eq!(e, [0, 0], "({p}, {q}): column {col}");
            let want = [
                g[0].take(&mut single[0][col]),
                g[1].take(&mut single[1][col]),
            ];
            assert_eq!(got, want, "({p}, {q}): column {col}");
        }
        let mut zero = [0i64; 2];
        assert_eq!(f.take(&mut zero), [0, 0]);
    }
}

// --- 2: a pair learn against the runs alone ---

/// The pair learn at (p, q) against the learn run at p alone and the run
/// at q with no learner: the first lane the learn run's leading monomials
/// and basis, the record the learn run's, and the second lane the run at
/// q's leading monomials and basis.
fn same_as_alone(case: &str, text: &str, params: &Params, (p, q): (u64, u64)) {
    let (first, second, record) = pair_learn(text, params, (p, q))
        .unwrap_or_else(|e| panic!("{case}: the pair learn at ({p}, {q}) failed: {e:?}"));
    let (alone, alone_record) = learn_alone(text, params, p);
    assert_eq!(record, alone_record, "{case}: the records");
    assert_eq!(first.leads, alone.leads, "{case}: the first lane's leads");
    assert_eq!(first.basis, alone.basis, "{case}: the first lane's basis");
    assert!(
        first.cofactors.is_empty() && second.cofactors.is_empty(),
        "{case}"
    );
    let at_q = run_at(text, params, q);
    assert_eq!(second.leads, at_q.leads, "{case}: the second lane's leads");
    assert_eq!(second.basis, at_q.basis, "{case}: the second lane's basis");
}

/// Test 2, q-learn-time.1, .10: a pair learn is the learn run at its first
/// prime and the run at its second, in grevlex, in lex, whose bases FGLM
/// changes, and in `block:2`; and in grevlex with the primes swapped.
#[test]
fn a_pair_learn_is_the_two_runs() {
    for (name, text) in systems() {
        same_as_alone(&name, &text, &params(1), (L1, L2));
        same_as_alone(&format!("{name}, swapped"), &text, &params(1), (L2, L1));
    }
    let text = ordered();
    for order in ["lex", "block:2"] {
        same_as_alone(order, &text, &params_in(order, 1), (L3, N1));
    }
}

// --- 3: disagreements ---

/// The texts whose pair learn at (L1, L2) disagrees: each case, its text,
/// its order, and which prime breaks it.
fn disagreements() -> Vec<(&'static str, String, &'static str, u64)> {
    vec![
        (
            "a leading coefficient q",
            format!("x,y,z\n0\n{L2}*x^2+y,\nx*y-1\n"),
            "grevlex",
            L2,
        ),
        (
            "a leading coefficient p",
            format!("x,y,z\n0\n{L1}*x^2+y,\nx*y-1\n"),
            "grevlex",
            L1,
        ),
        (
            "a row zero at q alone",
            format!("x,y,z\n0\nx*y-1,\nx*y+{L2}*z-1\n"),
            "grevlex",
            L2,
        ),
        (
            "a row zero at p alone",
            format!("x,y,z\n0\nx*y-1,\nx*y+{L1}*z-1\n"),
            "grevlex",
            L1,
        ),
        (
            "an FGLM pivot q",
            format!("x,y\n0\ny^2-{L2}*x,\nx^2-1\n"),
            "lex",
            L2,
        ),
    ]
}

/// Test 3, q-learn-time.4, .5: the pair learn at (L1, L2) halts with a
/// misfit at step 0 where a leading coefficient the run inverts is 0 at
/// one prime alone: of an input, of a row the generators' echelon form
/// keeps, and of a pivot of FGLM's echelon form. The learn run at L1 alone returns an
/// image on each, and the run at L2 alone on those L1 breaks.
#[test]
fn disagreements_halt_the_pair_learn() {
    for (case, text, order, breaks) in disagreements() {
        let p = params_in(order, 1);
        let got = pair_learn(&text, &p, (L1, L2));
        assert!(
            matches!(got, Err(Error::Misfit { step: 0 })),
            "{case}: {got:?}",
            got = got.map(|_| "two lanes")
        );
        learn_alone(&text, &p, L1);
        if breaks == L1 {
            run_at(&text, &p, L2);
        }
    }
    // The primes swapped: the lane at 0 is the first.
    let (case, text, order, _) = &disagreements()[0];
    let got = pair_learn(text, &params_in(order, 1), (L2, L1));
    assert!(
        matches!(got, Err(Error::Misfit { step: 0 })),
        "{case}, swapped: {got:?}",
        got = got.map(|_| "two lanes")
    );
}

// --- 4: the lift in pairs against the lift alone ---

/// Test 4, q-learn-time.1, .10: both lifts return the same basis
/// and cofactors, and the lift in pairs with its first two primes forced
/// returns the basis of the lift alone.
#[test]
fn pairs_against_alone() {
    for (name, text) in systems() {
        for seed in 1..=2 {
            for cofactors in [false, true] {
                let mut p = params(seed);
                p.cofactors = cofactors;
                let case = format!("{name}, seed {seed}, cofactors {cofactors}");
                let alone = lift(&text, &p, &[], Learns::Alone);
                let paired = lift(&text, &p, &[], Learns::Pairs);
                assert_eq!(paired.basis, alone.basis, "{case}");
                assert_eq!(paired.cofactors, alone.cofactors, "{case}");
            }
        }
        let alone = lift(&text, &params(1), &[], Learns::Alone);
        let forced = lift(&text, &params(1), &[L1, L2], Learns::Pairs);
        assert_eq!(forced.basis, alone.basis, "{name}: L1 and L2 forced");
    }
    let text = ordered();
    for order in ["lex", "block:2"] {
        for p in [params_in(order, 1), with_cofactors_in(order, 1)] {
            let case = format!("{order}, cofactors {cofactors}", cofactors = p.cofactors);
            let alone = lift(&text, &p, &[], Learns::Alone);
            let paired = lift(&text, &p, &[], Learns::Pairs);
            assert_eq!(paired.basis, alone.basis, "{case}");
            assert_eq!(paired.cofactors, alone.cofactors, "{case}");
        }
        let alone = lift(&text, &params_in(order, 1), &[], Learns::Alone);
        let forced = lift(&text, &params_in(order, 1), &[L3, L4], Learns::Pairs);
        assert_eq!(forced.basis, alone.basis, "{order}: L3 and L4 forced");
    }
}

// --- 5: the fallbacks ---

/// Test 5, q-learn-time.6: on a text whose pair learn at (L1, L2)
/// disagrees, or whose read meets a denominator 0 at L1 or L2, the lift
/// in pairs with L1 and L2 forced returns the basis the lift alone
/// returns, and on a denominator 0 the error it returns.
#[test]
fn the_fallbacks() {
    for (case, text, order, _) in disagreements() {
        let p = params_in(order, 1);
        let alone = lift(&text, &p, &[], Learns::Alone);
        let paired = lift(&text, &p, &[L1, L2], Learns::Pairs);
        assert_eq!(paired.basis, alone.basis, "{case}");
    }

    // A row zero at L1 alone.
    let text = format!("x,y,z\n0\nx*y-1,\nx*y+{L1}*z-1\n");
    let outcome = lift(&text, &params(1), &[L1, L2], Learns::Pairs);
    assert_eq!(basis(&outcome), "[z,\nx*y-1]:\n");

    // A denominator that L2 divides, and one that L1 divides.
    for d in [L2, L1] {
        let text = format!("x,y\n0\nx-1/{d}*y,\ny^2-x\n");
        let alone = lift(&text, &params(1), &[], Learns::Alone);
        let paired = lift(&text, &params(1), &[L1, L2], Learns::Pairs);
        assert_eq!(paired.basis, alone.basis, "a denominator {d}");
    }

    // A denominator 0: the error the lift alone returns.
    let text = "x,y\n0\nx-1/0*y,\ny^2-x\n";
    let errors: Vec<Error> = [Learns::Alone, Learns::Pairs]
        .into_iter()
        .map(|learns| {
            let mut report = Report::default();
            lift_text_as(
                text,
                &params(1),
                vec![L1, L2],
                Reconstruct::Chain,
                Replays::Pairs,
                learns,
                &mut NoTrace,
                &mut report,
            )
            .expect_err("a denominator 0")
        })
        .collect();
    assert!(matches!(errors[0], Error::NoInverse { .. }), "{errors:?}");
    assert_eq!(errors[0], errors[1]);
}

// --- 6: the edge cases ---

/// Test 6: the edge cases of `OBJECTIVES.md` on x, y, z over Q, in grevlex,
/// lex, and `block:1`, with and without cofactors: both lifts return the
/// same basis and cofactors.
#[test]
fn the_edge_cases_in_both_modes() {
    let cases = [
        ("the zero ideal", "0"),
        ("F = (0, 0)", "0,\n0"),
        ("no generator", ""),
        ("the unit ideal", "x*z-1,\nx"),
        ("already a Gröbner basis", "x+z,\nz^2+1"),
        (
            "the zero polynomial among the generators",
            "x*y-z,\n0,\ny^2-x",
        ),
        ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
        ("a denominator", "x-1/3*y^2,\ny^3-2,\nz-y"),
    ];
    for (case, generators) in cases {
        let text = format!("x,y,z\n0\n{generators}\n");
        for order in ["grevlex", "lex", "block:1"] {
            for p in [params_in(order, 1), with_cofactors_in(order, 1)] {
                let name = format!(
                    "{case} in {order}, cofactors {cofactors}",
                    cofactors = p.cofactors
                );
                let alone = lift(&text, &p, &[], Learns::Alone);
                let paired = lift(&text, &p, &[], Learns::Pairs);
                assert_eq!(paired.basis, alone.basis, "{name}");
                assert_eq!(paired.cofactors, alone.cofactors, "{name}");
            }
        }
    }
}
