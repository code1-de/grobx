//! The item's test code, `gfp-large-primes`: `grobx::groebner` over the
//! primes from 2^31 to 2^64 (plan.md, _Test code_). Every result computed
//! with cofactors is accepted by the checker (`OBJECTIVES.md`, _Domain
//! specifics the workflow points at_: the oracle).

use std::num::NonZeroU32;
use std::time::Duration;

use grobx::trace::NoTrace;
use grobx::{CheckerMode, Error, Params};
use grobx_checker::Mode;

use crate::common::{
    assert_accepted, degree_bound, dense, params, points, random_square, random_system, run,
    with_cofactors,
};
use crate::large::{LARGE, LARGEST, LEAST, over};
use crate::orders::finished_in;

/// The parameters of a measured call with signatures off: with no
/// cofactors the plain run reduces the rows in random combinations
/// (gfp-unsolved.2), which every prime of `LARGE` takes.
fn plain(seed: u64) -> Params {
    let mut p = params(seed);
    p.signatures = false;
    p
}

/// Computes `text` with cofactors, has the checker accept it in the exact
/// mode, compares it with `want` where given, and returns the basis.
fn accept(name: &str, text: &str, want: Option<&str>) -> String {
    let outcome = run(text, &with_cofactors(0x1a29e));
    assert_accepted(name, text, &outcome, Mode::Exact);
    let basis = outcome.basis.unwrap();
    if let Some(want) = want {
        assert_eq!(basis, want, "{name}");
    }
    basis
}

// --- the edge cases of OBJECTIVES.md ---

#[test]
fn the_zero_ideal() {
    for p in LARGE {
        accept(
            &format!("zero-{p}"),
            &format!("x,y\n{p}\n0\n"),
            Some("[]:\n"),
        );
        accept(
            &format!("zeros-{p}"),
            &format!("x,y\n{p}\n0,\n0\n"),
            Some("[]:\n"),
        );
        accept(&format!("none-{p}"), &format!("x,y\n{p}\n"), Some("[]:\n"));
    }
}

#[test]
fn the_unit_ideal() {
    for p in LARGE {
        accept(
            &format!("unit-{p}"),
            &format!("x,y\n{p}\nx*y-1,\nx\n"),
            Some("[1]:\n"),
        );
        accept(
            &format!("unit-x-{p}"),
            &format!("x\n{p}\nx,\nx+1\n"),
            Some("[1]:\n"),
        );
        accept(
            &format!("unit-constant-{p}"),
            &format!("x,y\n{p}\n5\n"),
            Some("[1]:\n"),
        );
    }
}

#[test]
fn input_already_a_groebner_basis() {
    for p in LARGE {
        accept(
            &format!("gb-{p}"),
            &format!("x,y\n{p}\nx+y,\ny^2+1\n"),
            Some("[x+y,\ny^2+1]:\n"),
        );
        let text = dense(3, 3, p, 1);
        let basis = accept(&format!("cubic-3-{p}"), &text, None);
        let elements = basis
            .trim_end()
            .trim_start_matches('[')
            .trim_end_matches("]:");
        accept(
            &format!("gb-cubic-3-{p}"),
            &format!("x1,x2,x3\n{p}\n{elements}\n"),
            Some(&basis),
        );
    }
}

/// The zero generator leaves the ideal as it is: the basis is that of the
/// generators without it.
#[test]
fn a_zero_generator() {
    for p in LARGE {
        let name = format!("zero-generator-{p}");
        let basis = accept(&name, &format!("x,y,z\n{p}\nx*y-z,\n0,\ny^2-x\n"), None);
        let without = format!("x,y,z\n{p}\nx*y-z,\ny^2-x\n");
        assert_eq!(basis, accept(&format!("{name}-without"), &without, None));
    }
}

#[test]
fn an_eliminable_variable() {
    for p in LARGE {
        for (case, text) in [
            ("first", format!("x,y,z\n{p}\nx-y^2-z,\ny*z-1,\nz^2-y\n")),
            ("last", format!("y,z,x\n{p}\nx-y^2-z,\ny*z-1,\nz^2-y\n")),
            (
                "linear",
                format!("x,y,z\n{p}\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n"),
            ),
        ] {
            accept(&format!("eliminable-{case}-{p}"), &text, None);
        }
    }
}

// --- coefficients and characteristics as written ---

/// The value of the decimal digits mod p, computed here in `u128`.
fn residue(digits: &str, p: u64) -> u64 {
    digits.bytes().fold(0u128, |acc, b| {
        (acc * 10 + u128::from(b - b'0')) % u128::from(p)
    }) as u64
}

/// a^(p − 2) mod p, the inverse of a unit by Fermat, computed here.
fn inverse(a: u64, p: u64) -> u64 {
    let q = u128::from(p);
    let (mut base, mut e, mut r) = (u128::from(a) % q, p - 2, 1u128);
    while e > 0 {
        if e & 1 == 1 {
            r = r * base % q;
        }
        base = base * base % q;
        e >>= 1;
    }
    r as u64
}

/// Coefficients past p and past 2^64, negative ones, and fractions give
/// the basis of the text with every coefficient reduced mod p here.
#[test]
fn coefficients_as_written() {
    let big = "123456789012345678901234567890123456789";
    let two_64 = "18446744073709551616";
    for p in [LEAST, 9_223_372_036_854_775_783, LARGEST] {
        let q = u128::from(p);
        let past = (q + 5).to_string();
        let text = format!("x,y\n{p}\n{past}*x^2+{two_64}*y-{big},\nx*y-1/3*y+5/7\n");
        let neg = |v: u64| ((q - u128::from(v)) % q) as u64;
        let reduced = format!(
            "x,y\n{p}\n5*x^2+{two_64_mod_p}*y+{minus_big},\nx*y+{minus_third}*y+{five_sevenths}\n",
            two_64_mod_p = residue(two_64, p),
            minus_big = neg(residue(big, p)),
            minus_third = neg(inverse(3, p)),
            five_sevenths = (5 * u128::from(inverse(7, p)) % q) as u64,
        );
        let name = format!("written-{p}");
        let basis = accept(&name, &text, None);
        assert_eq!(basis, run(&reduced, &params(3)).basis.unwrap(), "{name}");
        // A denominator that is 0 mod p.
        for d in [q, 2 * q] {
            let text = format!("x\n{p}\n1/{d}*x+1\n");
            assert!(
                matches!(
                    grobx::groebner(&text, &params(1)),
                    Err(Error::NoInverse { .. })
                ),
                "1/{d} over {p}"
            );
        }
    }
}

/// Composites from 2^31 to 2^64, 2^64 and past, and a number of 30 digits
/// are `Error::Characteristic`, as written, with a message that names
/// 2^64; a prime with leading zeros is read.
#[test]
fn what_the_engine_refuses() {
    for c in [
        "2147483648",
        "4294967297",
        "2301745249",
        "3825123056546413051",
        "18446744030759878681",
        "18446744073709551615",
        "18446744073709551616",
        "18446744073709551629",
        "123456789012345678901234567890",
    ] {
        let text = format!("x,y\n{c}\nx^2+y,\nx*y+1\n");
        match grobx::groebner(&text, &params(1)) {
            Err(e @ Error::Characteristic { .. }) => {
                assert!(matches!(&e, Error::Characteristic { value } if value == c));
                assert!(e.to_string().contains("2^64"), "{e}");
            }
            other => panic!("characteristic {c}: {other:?}"),
        }
    }
    let tiny = |c: &str| format!("x,y\n{c}\nx^2+y,\nx*y+1\n");
    let want = format!(
        "[y^2+{minus_one}*x,\nx*y+1,\nx^2+y]:\n",
        minus_one = LARGEST - 1
    );
    assert_eq!(
        run(&tiny(&LARGEST.to_string()), &params(1)).basis.unwrap(),
        want
    );
    assert_eq!(
        run(&tiny(&format!("0000{LARGEST}")), &params(1))
            .basis
            .unwrap(),
        want
    );
}

// --- random systems, orders, generated systems, and replays ---

/// Random systems over p, with cofactors and accepted in the exact mode;
/// the call without cofactors and the call with signatures off return the
/// same basis.
fn random_systems_over(p: u64) {
    for seed in 0..4 {
        let text = random_system(p, 0x1a7e + seed);
        let name = format!("random-p{p}-{seed}");
        let basis = accept(&name, &text, None);
        let without = run(&text, &params(seed)).basis.unwrap();
        assert_eq!(without, basis, "{name} without cofactors");
        let combined = run(&text, &plain(seed)).basis.unwrap();
        assert_eq!(combined, basis, "{name} with signatures off");
    }
}

/// The systems read over the large primes, over 65521 as written: dense
/// cubic systems on two and three variables and random square systems.
fn systems() -> Vec<(String, String)> {
    let mut list = Vec::new();
    for m in [2, 3] {
        list.push((format!("cubic-{m}"), dense(m, 3, 65521, m as u64)));
    }
    for seed in 0..3 {
        list.push((format!("square-{seed}"), random_square(65521, seed)));
    }
    list
}

/// Each of `systems` read over p, with cofactors and accepted in the cheap
/// mode, k chosen as checker-cheap.16 chooses it, and the same basis with
/// signatures off.
fn systems_over(p: u64) {
    for (stem, text) in systems() {
        let text = over(&text, p);
        let name = format!("{stem}-over-{p}");
        let outcome = run(&text, &with_cofactors(0x7a11));
        let k = points(p, degree_bound(&text, &outcome));
        let mode = Mode::Cheap {
            points: NonZeroU32::new(k).unwrap(),
            seed: 0xc4ec ^ p,
        };
        assert_accepted(&name, &text, &outcome, mode);
        let basis = outcome.basis.unwrap();
        let combined = run(&text, &plain(1)).basis.unwrap();
        assert_eq!(combined, basis, "{name} with signatures off");
    }
}

/// A record of `text` learned over `from` replays over `to` and returns the
/// basis a run over `to` returns.
fn assert_replays(name: &str, text: &str, from: u64, to: u64) {
    let learned = over(text, from);
    let (_, record) = grobx::learn(&learned, &params(5), &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: learn over {from}: {e:?}"));
    let record = record.unwrap_or_else(|| panic!("{name}: no record over {from}"));
    let other = over(text, to);
    let replayed = grobx::replay(&other, &params(5), &record, &mut NoTrace)
        .unwrap_or_else(|e| panic!("{name}: replay over {to} of {from}'s record: {e:?}"));
    assert_eq!(
        replayed.basis,
        run(&other, &params(5)).basis,
        "{name}: replay over {to} of {from}'s record"
    );
}

/// A record of each of `systems` learned over 65521 replays over p.
fn replays_over(p: u64) {
    for (stem, text) in systems() {
        assert_replays(&stem, &text, 65521, p);
    }
}

macro_rules! over_each_prime {
    ($($name:ident => $k:expr;)*) => {
        mod random_systems {
            $(#[test] fn $name() { super::random_systems_over(super::LARGE[$k]); })*
        }
        mod generated_systems {
            $(#[test] fn $name() { super::systems_over(super::LARGE[$k]); })*
        }
        mod replays_from_65521 {
            $(#[test] fn $name() { super::replays_over(super::LARGE[$k]); })*
        }
    };
}

over_each_prime! {
    least_past_2_31 => 0;
    below_2_32 => 1;
    past_2_32 => 2;
    below_2_62 => 3;
    below_2_63 => 4;
    past_2_63 => 5;
    largest_below_2_64 => 6;
}

/// A record learned over 2^64 − 59 replays over 65521, which `Gf31` takes,
/// and over the least prime past 2^31.
#[test]
fn replays_from_the_largest_prime() {
    for (stem, text) in systems() {
        for to in [65521, LEAST] {
            assert_replays(&stem, &text, LARGEST, to);
        }
    }
}

const DEADLINE: Duration = Duration::from_secs(10);

/// Random square systems over every prime of `LARGE` in lex, changed by
/// FGLM where the ideal is zero-dimensional, and in `block:1`, with
/// cofactors and a deadline of 10 s, canonical in the order and accepted
/// by the checker in it; nine in ten finish (fglm-lex.24, order-family.19).
#[test]
fn lex_and_a_block_order() {
    for order in ["lex", "block:1"] {
        let (mut finished, mut total) = (0, 0);
        for p in LARGE {
            for seed in 0..6 {
                let text = random_square(p, seed);
                let name = format!("square-p{p}-{seed}");
                total += 1;
                if finished_in(&name, &text, order, Mode::Exact, DEADLINE).is_some() {
                    finished += 1;
                }
            }
        }
        assert!(
            finished * 10 >= total * 9,
            "{order}: only {finished} of {total} finished"
        );
    }
}

/// The checker parameter certifies a result over a large prime in both
/// modes.
#[test]
fn the_checker_parameter() {
    for p in [LEAST, LARGEST] {
        let text = dense(3, 3, p, 1);
        for mode in [
            CheckerMode::Exact,
            CheckerMode::Cheap {
                points: NonZeroU32::new(4).unwrap(),
            },
        ] {
            let mut params = params(9);
            params.checker = Some(mode);
            let outcome = run(&text, &params);
            assert_eq!(
                outcome.certified,
                Some(true),
                "{p}, {mode:?}: {flaw:?}",
                flaw = outcome.flaw
            );
        }
    }
}

/// `grobx::text::generators` over a large prime: each generator monic, its
/// coefficients in [0, p), over each prime of `LARGE` and over 2^61 − 1.
#[test]
fn generators_over_a_large_prime() {
    for p in LARGE.into_iter().chain([(1 << 61) - 1]) {
        let q = u128::from(p);
        let text = format!(
            "x,y\n{p}\n2*x^2+{past}*y,\n{minus_one}*x*y-1\n",
            past = q + 3,
            minus_one = p - 1
        );
        // x^2 + 3·2⁻¹·y with 2⁻¹ = (p + 1)/2, and −x·y − 1 made monic,
        // x·y + 1.
        let half = q.div_ceil(2);
        let want = format!(
            "[x^2+{three_halves}*y,\nx*y+1]:\n",
            three_halves = 3 * half % q
        );
        assert_eq!(grobx::text::generators(&text).unwrap(), want, "over {p}");
    }
}
