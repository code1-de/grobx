//! q-denominator-line, the read's error over ℚ: the parse's `NoInverse` at
//! a denominator 0, the same error under every first prime and every
//! seed, the reader's order over GF(p), and `Header::zero_denominator`
//! from the parse (`iterations/0104-q-denominator-line/plan.md`, _Test
//! code_, 1 to 5). `echelon-lane-zero`'s tests are `echelon_lane_tests.rs`.

use std::fmt::Debug;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::denominator_tests::{TEXTS, least_prime};
use crate::field::{Gf31, is_prime_64};
use crate::lift::{Learns, Reconstruct, Replays, Report, lift_text, lift_text_as};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Format};
use crate::order::Grevlex;
use crate::shared_tests::params;
use crate::table::Table;
use crate::text::{Header, Unread, generators};
use crate::trace::NoTrace;
use crate::{Error, groebner};

/// The `n` primes after `p`. The texts here have the denominators P, 2, 3,
/// and 10, so a prime after P divides none of them.
fn primes_after(p: u64, n: usize) -> Vec<u64> {
    (p + 1..).filter(|&q| is_prime_64(q)).take(n).collect()
}

/// The error a read at a prime that divides no denominator of the text
/// returns.
#[derive(Clone, Copy, Debug)]
enum Want {
    NoInverse(usize),
    Unknown(usize, &'static str),
    Syntax(usize),
    Overflow(usize),
}

impl Want {
    /// Asserts that `got` is this error, `context` named in the message.
    fn check<T: Debug>(self, got: &Result<T, Error>, context: &str) {
        match self {
            Want::NoInverse(line) => assert!(
                matches!(got, Err(Error::NoInverse { line: l }) if *l == line),
                "{context}: NoInverse at {line}, got {got:?}"
            ),
            Want::Unknown(line, name) => assert!(
                matches!(
                    got,
                    Err(Error::UnknownVariable { line: l, name: n }) if *l == line && n == name
                ),
                "{context}: UnknownVariable {name} at {line}, got {got:?}"
            ),
            Want::Syntax(line) => assert!(
                matches!(got, Err(Error::Syntax { line: l, .. }) if *l == line),
                "{context}: Syntax at {line}, got {got:?}"
            ),
            Want::Overflow(line) => assert!(
                matches!(got, Err(Error::Overflow { line: Some(l) }) if *l == line),
                "{context}: Overflow at {line}, got {got:?}"
            ),
        }
    }
}

/// The texts whose line holds a denominator `p` beside an earlier error or
/// a `/0`: each with the error a read at a prime that divides no
/// denominator returns, and whether a read at `p` stops before the parse's
/// error, so that the lift skips `p`.
fn cases(p: u64) -> Vec<(String, Want, bool)> {
    vec![
        (
            format!("x,y\n0\n1/{p}*x+w+1/0\n"),
            Want::Unknown(3, "w"),
            true,
        ),
        (format!("x,y\n0\n1/{p}*x+y/x+1/0\n"), Want::Syntax(3), true),
        (
            format!("x,y\n0\nx+1/{p}*y-1/0\n"),
            Want::NoInverse(3),
            false,
        ),
        (
            format!("x,y\n0\nx+1/{p},\ny-1/0\n"),
            Want::NoInverse(4),
            true,
        ),
        (
            format!("x,y\n0\nw+1/{p}*x+1/0\n"),
            Want::Unknown(3, "w"),
            false,
        ),
        (
            format!("x,y\n0\nx+1/{p},\ny+w+1/0\n"),
            Want::Unknown(4, "w"),
            true,
        ),
    ]
}

/// Test 1, q-denominator-line.1 and .2: through `lift_text` with the first
/// prime forced to P, then to Q, the texts of `cases`: one error under both
/// first primes, the one a read at Q returns, and P skipped where its read
/// stops before the parse's error. The first text through pair learns at
/// (P, Q): the pair unpaired for the denominator, then P skipped alone.
#[test]
fn one_error_under_either_first_prime() {
    let p = least_prime();
    let q = primes_after(p, 1)[0];
    let cases = cases(p);
    for (text, want, stops_at_p) in &cases {
        for (first, skipped) in [(p, *stops_at_p), (q, false)] {
            let mut report = Report::default();
            let got = lift_text(text, &params(1), vec![first], &mut report);
            want.check(&got, &format!("{text}under {first}"));
            let expect: Vec<u64> = if skipped { vec![p] } else { vec![] };
            assert_eq!(report.skipped, expect, "{text}under {first}: {report:?}");
        }
    }
    let (text, want, _) = &cases[0];
    let mut report = Report::default();
    let got = lift_text_as(
        text,
        &params(1),
        vec![p, q],
        Reconstruct::Chain,
        Replays::Pairs,
        Learns::Pairs,
        &mut NoTrace,
        &mut report,
    );
    want.check(&got, &format!("{text}in pairs"));
    assert_eq!(report.unpaired, vec![(p, q)], "{text}in pairs: {report:?}");
    assert_eq!(report.skipped, vec![p], "{text}in pairs: {report:?}");
}

/// Test 2, q-denominator-line.1: through `groebner` over ℚ under seeds 1 to
/// 8, the texts of `cases` with P as their denominator: one error each,
/// the error of test 1.
#[test]
fn one_error_under_every_seed() {
    let p = least_prime();
    for (text, want, _) in &cases(p) {
        for seed in 1..=8 {
            let got = groebner(text, &params(seed));
            want.check(&got, &format!("{text}seed {seed}"));
        }
    }
}

/// Test 3, q-denominator-line.1: the texts of `denominator_tests::TEXTS`
/// and the edge texts, the unit ideal's `1/0`, the zero ideal's `0/0`, and
/// a `/0` after the zero polynomial or after a generator, through
/// `lift_text` under P, Q, and R: `NoInverse` at the line named under each
/// first prime, no prime skipped, and the texts without a `/0` lift one
/// basis under each.
#[test]
fn a_denominator_0_under_every_first_prime() {
    let p = least_prime();
    let after = primes_after(p, 2);
    let edge: [(&str, Option<usize>); 4] = [
        ("x\n0\n1/0\n", Some(3)),
        ("x\n0\n0/0\n", Some(3)),
        ("x\n0\n0,\nx+1/0\n", Some(4)),
        ("x\n0\nx,\nx+1/0\n", Some(4)),
    ];
    for &(text, line) in TEXTS.iter().chain(&edge) {
        let mut bases = Vec::new();
        for first in [p, after[0], after[1]] {
            let mut report = Report::default();
            let got = lift_text(text, &params(1), vec![first], &mut report);
            match line {
                Some(line) => Want::NoInverse(line).check(&got, &format!("{text}under {first}")),
                None => bases.push(
                    got.unwrap_or_else(|e| panic!("{text}under {first}: {e:?}"))
                        .basis,
                ),
            }
            assert!(report.skipped.is_empty(), "{text}under {first}: {report:?}");
        }
        assert!(bases.windows(2).all(|w| w[0] == w[1]), "{text}: {bases:?}");
    }
}

/// Test 4: over GF(p) the reader's order is unchanged. Through
/// `text::generators`, which widens the format past an overflow: a `/0`
/// before an unknown variable is `NoInverse`, after one `UnknownVariable`,
/// after a syntax error `Syntax`, after a term past `Dense8` `NoInverse` at
/// its own line, and a `/0` whose digits stand two lines after the `/`
/// `NoInverse` at the digits' line. In `Dense8` itself the overflow comes
/// first, at its term's line.
#[test]
fn the_reader_order_over_gfp() {
    let cases: [(&str, Want); 5] = [
        ("x,y\n65521\n1/0*x+q\n", Want::NoInverse(3)),
        ("x,y\n65521\nq+1/0\n", Want::Unknown(3, "q")),
        ("x,y\n65521\nx+*y+1/0\n", Want::Syntax(3)),
        ("x,y\n65521\nx^200,\n1/0*y\n", Want::NoInverse(4)),
        ("x,y\n65521\nx+1/\n\n000\n", Want::NoInverse(5)),
    ];
    for (text, want) in cases {
        let got = generators(text);
        want.check(&got, text);
    }
    let text = "x,y\n65521\nx^200,\n1/0*y\n";
    let header = Header::read(text, &mut Budget::unbounded()).unwrap();
    let field = Gf31::prime(65521);
    let mut t8: Table<Dense8> =
        Table::new(Dense8::new(2), &mut Xoshiro256PlusPlus::seed_from_u64(1));
    let got = header
        .generators::<Gf31, Dense8, Grevlex, Rank1>(
            &field,
            &mut t8,
            &Grevlex,
            &mut Budget::unbounded(),
        )
        .map_err(Unread::unbounded);
    Want::Overflow(3).check(&got, &format!("{text}in Dense8"));
    let mut t16: Table<Dense16> =
        Table::new(Dense16::new(2), &mut Xoshiro256PlusPlus::seed_from_u64(1));
    let got = header
        .generators::<Gf31, Dense16, Grevlex, Rank1>(
            &field,
            &mut t16,
            &Grevlex,
            &mut Budget::unbounded(),
        )
        .map_err(Unread::unbounded);
    Want::NoInverse(4).check(&got, &format!("{text}in Dense16"));
}

/// Test 5, q-denominator-line.1: `Header::zero_denominator` is the line of
/// the denominator 0 the parse stopped at: None where an unknown variable
/// or a syntax error stands before the `/0` on its line, Some where the
/// `/0` is the parse's error, and the line `TEXTS` names on each of its
/// texts.
#[test]
fn the_parse_names_the_denominator_0() {
    let p = least_prime();
    let cases = [
        (format!("x,y\n0\n1/{p}*x+w+1/0\n"), None),
        (format!("x,y\n0\n1/{p}*x+y/x+1/0\n"), None),
        (format!("x,y\n0\nx+1/{p}*y-1/0\n"), Some(3)),
    ];
    for (text, line) in &cases {
        let header = Header::read(text, &mut Budget::unbounded()).unwrap();
        header.parsed(&mut Budget::unbounded()).unwrap();
        assert_eq!(header.zero_denominator(), *line, "{text}");
    }
    for &(text, line) in TEXTS {
        let header = Header::read(text, &mut Budget::unbounded()).unwrap();
        header.parsed(&mut Budget::unbounded()).unwrap();
        assert_eq!(header.zero_denominator(), line, "{text}");
    }
}
