//! test-mq-systems, sub-item 1: `random_square` writes no system of dense
//! quadrics and keeps every other draw, the dense cubic
//! systems that replace the dense quadrics are zero-dimensional of the
//! degree they are sized by, and no unit test of the crate calls `dense` of
//! degree 2 (`iterations/0084-test-mq-systems/plan.md`, _Test code_, 3 to
//! 6).

use grobx_checker::Mode;

use crate::common::{
    P, P31, assert_accepted, dense, dense_quadric_calls, max_degree, params, random_square, run,
    sha256, with_cofactors,
};
use crate::large::LARGE;

/// The generators of an instance text.
fn generators(text: &str) -> impl Iterator<Item = &str> {
    let generators = text.splitn(3, '\n').nth(2).expect("generators");
    generators.trim_end().split(",\n")
}

/// The degree of each generator of an instance text.
fn degrees(text: &str) -> Vec<u64> {
    generators(text).map(max_degree).collect()
}

/// Whether an instance text is a system of dense quadrics, what the `mq`
/// family's construction writes (test-mq-systems.1): every generator of
/// degree 2 with every monomial of degree at most 2, its terms joined by
/// `+` as `random_square` joins them. Over a small prime a draw leaves out
/// the monomials whose coefficient came out 0.
fn dense_quadrics(text: &str) -> bool {
    let n = text
        .lines()
        .next()
        .expect("a variable line")
        .split(',')
        .count();
    let monomials = (n + 1) * (n + 2) / 2;
    generators(text).all(|g| max_degree(g) == 2 && g.split('+').count() == monomials)
}

/// Test 3: over every field the tests take `random_square` over, and the
/// seeds 0 to 63, it writes no system of dense quadrics (test-mq-systems.7).
#[test]
fn random_square_draws_no_quadrics_alone() {
    for p in [0, 2, 3, 7, P, P31].into_iter().chain(LARGE) {
        for seed in 0..64 {
            let text = random_square(p, seed);
            assert!(
                !dense_quadrics(&text),
                "random_square({p}, {seed}) is a system of dense quadrics:\n{text}"
            );
        }
    }
}

/// The SHA-256 of `random_square`'s text at the pairs of a prime and a seed
/// that test code chose before test-mq-systems.7, among them
/// `matrix_stop_latency.rs`' (2^31 − 1, 3) and (2^31 − 1, 7) (log 0055-2).
const KEPT: [(u64, u64, &str); 5] = [
    (
        P31,
        3,
        "387d9823d8f7de7edadc171bf9de196f08c1e099ad3d88e3f9d2c11decfb4ed2",
    ),
    (
        P31,
        7,
        "bf22371248afeb958bbc5ebd25896110981431c5676272a9368db695c3076982",
    ),
    (
        0,
        1,
        "6177bb8393fc44d1930c6020e5f3203ce85c5cfb34ddf591ddace4f5fe69e3f7",
    ),
    (
        P,
        1,
        "10e0b8423eba8c8d9245b569eda77535de2c86822e22033c9eec4cf53050f5a0",
    ),
    (
        0,
        3,
        "12c888f61f2f623aea356daa08ced8f84fb9e296d298e1f1820c51c2a8f94b99",
    ),
];

/// Test 4: `random_square` writes the text it wrote before
/// test-mq-systems.7 where its draw holds a generator of a degree other
/// than 2, and (65521, 10), whose draw was two quadrics, holds a quadric
/// and a cubic.
#[test]
fn random_square_keeps_its_other_draws() {
    for (p, seed, digest) in KEPT {
        assert_eq!(
            sha256(&random_square(p, seed)),
            digest,
            "random_square({p}, {seed})"
        );
    }
    assert_eq!(degrees(&random_square(P, 10)), [2, 3]);
}

/// The exponent vector of the leading monomial of each element of a basis
/// text in grevlex on `names`: the first term of each polynomial.
fn leads(basis: &str, names: &[&str]) -> Vec<Vec<u32>> {
    let inner = basis
        .strip_prefix('[')
        .and_then(|b| b.strip_suffix("]:\n"))
        .expect("a basis text");
    inner
        .split(",\n")
        .map(|poly| {
            let end = poly[1..].find(['+', '-']).map_or(poly.len(), |i| i + 1);
            let mut e = vec![0u32; names.len()];
            for factor in poly[..end].split('*') {
                if factor.starts_with(|c: char| c.is_ascii_digit()) {
                    continue;
                }
                let (name, k) = factor.split_once('^').unwrap_or((factor, "1"));
                let i = names
                    .iter()
                    .position(|n| *n == name)
                    .unwrap_or_else(|| panic!("{name} is no variable of {names:?}"));
                e[i] = k.parse().expect("an exponent");
            }
            e
        })
        .collect()
}

/// The largest exponent of a variable the staircase walk takes before it
/// rules the ideal not zero-dimensional.
const STAIRCASE_REACH: u32 = 64;

/// The monomials no leading monomial of `leads` divides that agree with `e`
/// before position `i`, where `e` is zero from `i` on: the staircase's
/// monomials, counted variable by variable. A monomial past a leading one
/// in one variable stays past it, so each variable's walk ends at its first
/// monomial in the ideal.
fn staircase(leads: &[Vec<u32>], e: &mut [u32], i: usize) -> u64 {
    if i == e.len() {
        return 1;
    }
    let mut count = 0;
    while !leads
        .iter()
        .any(|l| l.iter().zip(e.iter()).all(|(a, b)| a <= b))
    {
        assert!(e[i] < STAIRCASE_REACH, "no pure power of variable {i}");
        count += staircase(leads, e, i + 1);
        e[i] += 1;
    }
    e[i] = 0;
    count
}

/// Test 5: dense cubic systems on m variables, m from 2 to 5 over 65521 and
/// from 2 to 3 over ℚ, seeds 1 to 3, are zero-dimensional of degree 3^m, the
/// size test-mq-systems.4 takes them by, and the checker accepts them with
/// their cofactors in the exact mode, m up to 4 over 65521. The dense
/// quartic system of test-mq-systems.8 is of degree 4^5.
#[test]
fn dense_cubics_have_the_bezout_degree() {
    let mut cases: Vec<(usize, u32, u64, u64)> = Vec::new();
    for seed in 1..=3 {
        cases.extend((2..=5).map(|m| (m, 3, P, seed)));
        cases.extend((2..=3).map(|m| (m, 3, 0, seed)));
    }
    cases.push((5, 4, P, 2));
    for (m, d, p, seed) in cases {
        let text = dense(m, d, p, seed);
        let name = format!("dense({m}, {d}, {p}, {seed})");
        let names: Vec<&str> = text.lines().next().unwrap().split(',').collect();
        let basis = run(&text, &params(seed)).basis.unwrap();
        let degree = staircase(&leads(&basis, &names), &mut vec![0; m], 0);
        assert_eq!(
            degree,
            u64::from(d).pow(u32::try_from(m).unwrap()),
            "{name}"
        );
        if d == 3 && (p == 0 || m <= 4) {
            let outcome = run(&text, &with_cofactors(seed));
            assert_accepted(&name, &text, &outcome, Mode::Exact);
            assert_eq!(outcome.basis.as_deref(), Some(basis.as_str()), "{name}");
        }
    }
}

/// Test 6: no file of the crate's unit tests calls `dense` of degree 2.
#[test]
fn the_unit_tests_write_no_dense_quadrics() {
    let calls = dense_quadric_calls(&["src"]);
    assert!(calls.is_empty(), "dense quadrics at {calls:#?}");
}
