//! q-zero-denominator, the scan and the lift: the text's first denominator
//! 0 by its line, the line the reader gives over GF(7), and the lift's
//! answer where a prime of its own divides a denominator beside a 0
//! (plan.md, _Test code_).

use std::num::NonZeroUsize;

use crate::field::is_prime_64;
use crate::lift::{Report, lift_text};
use crate::text::Header;
use crate::{Error, MonomialOrder, Params, Threads};

fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

/// P, the least prime above 2^30, which the lift is made to take first.
fn least_prime() -> u64 {
    ((1u64 << 30) + 1..)
        .find(|&n| is_prime_64(n))
        .expect("a prime above 2^30")
}

/// The texts over ℚ and the line of each one's first denominator 0.
const TEXTS: &[(&str, Option<usize>)] = &[
    ("x\n0\nx+1/0\n", Some(3)),
    ("x,y\n0\nx+1/3,\ny-1/0\n", Some(4)),
    ("x\n0\nx+7/000\n", Some(3)),
    ("x\n0\n2/3*5/0*x\n", Some(3)),
    ("x\n0\n0/0\n", Some(3)),
    ("x\n0\nx+1/\n\n 0\n", Some(5)),
    ("x\n0\nx-1/0,\nx+1/0\n", Some(3)),
    ("x\n0\nx+1/10\n", None),
    ("x\n0\nx+1/0010\n", None),
    ("x\n0\nx\n", None),
    ("x\n0\n", None),
];

/// q-zero-denominator.2, .3: the scan gives the line of the first
/// denominator 0, at its digits, and the reader over GF(7) stops at that
/// line with `NoInverse`, as it does at every prime that divides no
/// denominator before it.
#[test]
fn the_first_denominator_0() {
    for &(text, line) in TEXTS {
        let header = Header::read(text).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
        assert!(header.rational(), "{text}");
        assert_eq!(header.zero_denominator(), line, "{text}");
        let over7 = text.replacen("\n0\n", "\n7\n", 1);
        match line {
            Some(line) => assert!(
                matches!(crate::text::generators(&over7), Err(Error::NoInverse { line: l }) if l == line),
                "{over7}: {generators:?}",
                generators = crate::text::generators(&over7)
            ),
            None => assert!(crate::text::generators(&over7).is_ok(), "{over7}"),
        }
    }
}

/// q-zero-denominator.2: the lift's first prime forced to P. A denominator
/// P skips P, and a denominator 0 returns `NoInverse` at its line: after
/// the skip where P's denominator comes first, at P where the 0 comes
/// first or shares the line. A denominator P beside none that is 0 skips
/// P and lifts the basis.
#[test]
fn a_prime_that_divides_a_denominator_beside_a_0() {
    let p = least_prime();
    let cases: [(String, Result<String, usize>, Vec<u64>); 4] = [
        (format!("x,y\n0\nx+1/{p},\ny-1/0\n"), Err(4), vec![p]),
        (format!("x,y\n0\nx-1/0,\ny+1/{p}\n"), Err(3), vec![]),
        (format!("x,y\n0\nx+1/{p}*y-1/0\n"), Err(3), vec![]),
        (
            format!("x,y\n0\nx+1/{p},\ny-1/2\n"),
            Ok(format!("[2*y-1,\n{p}*x+1]:\n")),
            vec![p],
        ),
    ];
    for (text, want, skipped) in cases {
        let mut report = Report::default();
        let got = lift_text(&text, &params(1), vec![p], &mut report);
        match want {
            Err(line) => assert!(
                matches!(got, Err(Error::NoInverse { line: l }) if l == line),
                "{text}: {got:?}"
            ),
            Ok(basis) => assert_eq!(
                got.unwrap_or_else(|e| panic!("{text}: {e:?}"))
                    .basis
                    .as_deref(),
                Some(basis.as_str()),
                "{text}"
            ),
        }
        assert_eq!(report.skipped, skipped, "{text}: {report:?}");
    }
}
