//! Test code for the primes from 2^31 to 2^64 (`gfp-large-primes`): the
//! primes the test code runs on, an instance read over another prime, and
//! GNU `factor` as the oracle on primality. Nothing here leans on the
//! engine.


use std::process::Command;

// gfp-large-primes.13 practice: boundary-value testing -- the test code's primes are the least past 2^31, those on either side of 2^32 and of 2^63, the largest below 2^62, and the largest below 2^64, not random primes
/// The primes of the test code: 2^31 + 11, 2^32 − 5, 2^32 + 15, 2^62 − 57,
/// 2^63 − 25, 2^63 + 29, and 2^64 − 59.
pub const LARGE: [u64; 7] = [
    2_147_483_659,
    4_294_967_291,
    4_294_967_311,
    4_611_686_018_427_387_847,
    9_223_372_036_854_775_783,
    9_223_372_036_854_775_837,
    18_446_744_073_709_551_557,
];

/// The least prime past 2^31.
pub const LEAST: u64 = 2_147_483_659;
/// The largest prime below 2^64.
pub const LARGEST: u64 = 18_446_744_073_709_551_557;

/// The instance `text` with its characteristic line replaced by `p`.
pub fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let names = lines.next().expect("a variable line");
    let _ = lines.next();
    let rest = lines.next().unwrap_or("");
    format!("{names}\n{p}\n{rest}")
}

// gfp-large-primes.11 design -- the test code's oracle on primality is GNU coreutils' factor, run as a black box, not a list of primes and composites written by hand (not measured)
/// Whether each of `numbers` is prime, as GNU `factor` says: one factor,
/// the number itself. `factor` is run once per thousand of them.
pub fn factor_says_prime(numbers: &[u64]) -> Vec<bool> {
    let mut verdicts = Vec::with_capacity(numbers.len());
    for chunk in numbers.chunks(1000) {
        let out = Command::new("factor")
            .args(chunk.iter().map(u64::to_string))
            .output()
            .unwrap_or_else(|e| panic!("factor does not run ({e}): the test code needs it"));
        assert!(out.status.success(), "factor failed: {out:?}");
        let text = String::from_utf8(out.stdout).expect("factor prints UTF-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), chunk.len(), "factor prints a line a number");
        for (&n, line) in chunk.iter().zip(lines) {
            let (head, factors) = line
                .split_once(':')
                .unwrap_or_else(|| panic!("factor's line {line:?}"));
            assert_eq!(head.trim(), n.to_string(), "factor's line {line:?}");
            let factors: Vec<u128> = factors
                .split_whitespace()
                .map(|f| f.parse().unwrap())
                .collect();
            // The factors multiply to n, so the line is about n.
            if n >= 2 {
                assert_eq!(factors.iter().product::<u128>(), u128::from(n), "{line:?}");
            }
            verdicts.push(factors.len() == 1 && factors[0] == u128::from(n));
        }
    }
    verdicts
}

