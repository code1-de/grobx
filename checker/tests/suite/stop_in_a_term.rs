//! silent-fallbacks, the checker's test code: a stop that falls in the
//! inverse of a term's denominator returns the term's reading error, as
//! `check_polled` states (`iterations/0094-silent-fallbacks/plan.md`,
//! _Test code_, 7).

use grobx_checker::{Error, Mode, Order, Part, check_polled};

/// The primes of the test: below 2^16, and the largest below 2^31.
const PRIMES: [u64; 2] = [65521, (1 << 31) - 1];

/// The counts of terms ahead of the term whose denominator is p: around
/// the 4096 units after which the check first asks its poll, a term of
/// fewer than 16 bytes a unit.
const AHEAD: std::ops::RangeInclusive<usize> = 4080..=4100;

/// The line of the instance that holds the generator.
const LINE: usize = 3;

/// Test 7: silent-fallbacks.9. The generator x^1 + x^2 + … + x^k + 1/p·y,
/// in the exact mode: with a poll that never stops, the check returns
/// `NoInverse` after C calls of it; with a poll that answers true at its
/// C-th call, the check stops or returns `NoInverse`, and returns it for
/// the k whose C-th call falls in the inverse of 1/p.
#[test]
fn a_stop_in_a_terms_inverse_keeps_its_error() {
    let error = Error::NoInverse {
        part: Part::Instance,
        line: LINE,
    };
    for p in PRIMES {
        let mut kept = 0;
        for k in AHEAD {
            let terms: Vec<String> = (1..=k).map(|i| format!("x^{i}")).collect();
            let instance = format!("x,y\n{p}\n{terms}+1/{p}*y\n", terms = terms.join("+"));
            let check = |poll: &mut dyn FnMut() -> bool| {
                check_polled(
                    &instance,
                    "[1]:\n",
                    "[[1]]:\n",
                    &Order::Grevlex,
                    Mode::Exact,
                    poll,
                )
            };
            let mut total = 0;
            let got = check(&mut || {
                total += 1;
                false
            });
            assert_eq!(got, Err(error.clone()), "{p}, {k} terms");
            if total == 0 {
                continue;
            }
            let mut calls = 0;
            let got = check(&mut || {
                calls += 1;
                calls >= total
            });
            match got {
                Ok(None) => {}
                Err(e) => {
                    assert_eq!(e, error, "{p}, {k} terms, stopped at poll {total}");
                    kept += 1;
                }
                Ok(Some(v)) => panic!("{p}, {k} terms: a verdict past the stop: {v:?}"),
            }
        }
        assert!(kept > 0, "{p}: no stop fell in the inverse");
    }
}
