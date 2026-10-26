//! Test code for the call over Q (`q-lift`): a basis text over Q read into
//! its terms, the canonical form over Q in an order, random systems over Q,
//! their generators with the denominators cleared, and a call with cofactors that the checker accepts. Nothing
//! here leans on the engine's arithmetic.

use std::cmp::Ordering;
use std::fmt::Write;
use std::time::Duration;

use grobx::{Outcome, Stop};
use grobx_checker::Mode;

use crate::common::{Rng, assert_accepted_in, run, sha256, terms_text, with_cofactors_in};
use crate::orders::{Family, names};

/// A polynomial of a basis text over Q: its terms as written, each its
/// exponent vector and its coefficient as a signed decimal integer, `1`
/// and `-1` spelled out.
pub type TermsQ = Vec<(Vec<u64>, String)>;

/// The terms of a polynomial over Q as the canonical text writes it:
/// terms joined by `+` or `-`, each an optional coefficient, then the
/// variables joined by `*`.
fn read_poly_q(poly: &str, names: &[String]) -> TermsQ {
    let mut terms = Vec::new();
    let mut start = 0;
    let bytes = poly.as_bytes();
    for i in 1..=bytes.len() {
        if i < bytes.len() && bytes[i] != b'+' && bytes[i] != b'-' {
            continue;
        }
        let term = &poly[start..i];
        start = i;
        let (sign, body) = match term.as_bytes()[0] {
            b'-' => ("-", &term[1..]),
            b'+' => ("", &term[1..]),
            _ => ("", term),
        };
        let mut coef = "1".to_string();
        let mut exponents = vec![0u64; names.len()];
        for factor in body.split('*') {
            if factor.starts_with(|c: char| c.is_ascii_digit()) {
                assert!(
                    factor.bytes().all(|b| b.is_ascii_digit()),
                    "a coefficient over Q is an integer: {term} in {poly}"
                );
                coef = factor.to_string();
                continue;
            }
            let (name, e) = factor.split_once('^').unwrap_or((factor, "1"));
            let i = names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("{name} in {term}"));
            exponents[i] += e.parse::<u64>().unwrap();
        }
        terms.push((exponents, format!("{sign}{coef}")));
    }
    terms
}

/// The polynomials of a basis text over Q, each as its terms in the order
/// written.
pub fn read_basis_q(text: &str, names: &[String]) -> Vec<TermsQ> {
    let inner = text
        .strip_prefix('[')
        .and_then(|t| t.strip_suffix("]:\n"))
        .unwrap_or_else(|| panic!("no basis text: {text:?}"));
    if inner.is_empty() {
        return Vec::new();
    }
    inner.split(",\n").map(|p| read_poly_q(p, names)).collect()
}

/// The basis text over Q is canonical in `name`'s order: each
/// polynomial's terms in decreasing order, its first coefficient positive,
/// and the polynomials in increasing order of their leading monomials
/// (`OBJECTIVES.md`, _Measurement_). The checker rules on the coefficients
/// having no common factor.
pub fn assert_canonical_q(case: &str, instance: &str, text: &str, name: &str) {
    let family = Family::named(name);
    let polys = read_basis_q(text, &names(instance));
    for poly in &polys {
        assert!(
            !poly.is_empty(),
            "{case} in {name}: an empty polynomial\n{text}"
        );
        assert!(
            !poly[0].1.starts_with('-'),
            "{case} in {name}: a negative leading coefficient\n{text}"
        );
        for w in poly.windows(2) {
            assert_eq!(
                family.cmp(&w[0].0, &w[1].0),
                Ordering::Greater,
                "{case} in {name}: terms not decreasing\n{text}"
            );
        }
    }
    for w in polys.windows(2) {
        assert_eq!(
            family.cmp(&w[0][0].0, &w[1][0].0),
            Ordering::Less,
            "{case} in {name}: leading monomials not increasing\n{text}"
        );
    }
}

/// The outcome of a call over Q in the order `name` with cofactors, its
/// basis text canonical in the order and accepted by the checker in it,
/// in the exact mode.
pub fn accepted_q(case: &str, instance: &str, name: &str) -> Outcome {
    let outcome = run(instance, &with_cofactors_in(name, 0x0de7));
    check_q(case, instance, name, &outcome);
    outcome
}

/// As `accepted_q`, with a deadline: None where the deadline stopped the
/// call (order-family.19).
pub fn finished_q(case: &str, instance: &str, name: &str, deadline: Duration) -> Option<Outcome> {
    let mut params = with_cofactors_in(name, 0x0de7);
    params.deadline = Some(deadline);
    let outcome = grobx::groebner(instance, &params)
        .unwrap_or_else(|e| panic!("{case} in {name}: {e:?}\n{instance}"));
    if outcome.basis.is_none() {
        assert_eq!(outcome.stopped, Some(Stop::Deadline), "{case} in {name}");
        eprintln!("{case} in {name}: stopped by the deadline");
        return None;
    }
    check_q(case, instance, name, &outcome);
    Some(outcome)
}

fn check_q(case: &str, instance: &str, name: &str, outcome: &Outcome) {
    let basis = outcome.basis.as_deref().unwrap();
    assert_canonical_q(case, instance, basis, name);
    let order = Family::named(name).checker(names(instance).len());
    assert_accepted_in(case, instance, outcome, &order, Mode::Exact);
}

/// A basis text over Q as the text a digest is taken of: each polynomial's
/// terms in decreasing order in `family` and its sign taken so that its
/// leading coefficient is positive, the polynomials sorted.
pub fn sorted_q_text(text: &str, names: &[String], family: &Family) -> String {
    let mut polys: Vec<TermsQ> = read_basis_q(text, names)
        .into_iter()
        .map(|mut poly| {
            poly.sort_by(|a, b| family.cmp(&b.0, &a.0));
            if poly[0].1.starts_with('-') {
                for (_, c) in &mut poly {
                    *c = match c.strip_prefix('-') {
                        Some(abs) => abs.to_string(),
                        None => format!("-{c}"),
                    };
                }
            }
            poly
        })
        .collect();
    polys.sort();
    terms_text(&polys)
}

/// Our basis text over Q and another program's are the same set of
/// polynomials, each polynomial's terms read in any order and its sign
/// taken so that its leading coefficient in `family` is positive: `theirs`
/// is the digest of their `sorted_q_text`.
pub fn same_set_q(case: &str, ours: &str, theirs: &str, names: &[String], family: &Family) {
    assert_eq!(
        sha256(&sorted_q_text(ours, names, family)),
        theirs,
        "{case}: the bases differ\nours:\n{ours}"
    );
}

// q-lift.39 design -- the random systems over Q have random_system's shapes, coefficients up to 10^6 with a random sign, one term in three with a denominator up to 1000, and each monomial once in a generator (not measured)
/// A random instance over Q: two to four of the variables x, y, z, w, one
/// generator up to one more than the variables, each of one to five terms
/// of degree four or less with a coefficient in [1, 10^6], a random sign,
/// and on one term in three a denominator in [2, 1000]. A generator spells
/// each monomial once: msolve keeps the last coefficient of a monomial
/// spelled twice over Q (`LEARNINGS.md`, _msolve as an oracle: terms that
/// are 0 mod p_).
pub fn random_q_system(seed: u64) -> String {
    let mut rng = Rng::new(seed ^ 0x0051_7a11);
    let names = ["x", "y", "z", "w"];
    let n = 2 + rng.below(3) as usize;
    let generators: Vec<String> = (0..=rng.below(n as u64 + 1))
        .map(|_| {
            let mut poly = String::new();
            let mut used: Vec<Vec<u64>> = Vec::new();
            for _ in 0..=rng.below(5) {
                let negative = rng.below(2) == 1;
                let mut coef = (1 + rng.below(1_000_000)).to_string();
                if rng.below(3) == 0 {
                    write!(coef, "/{denominator}", denominator = 2 + rng.below(999)).unwrap();
                }
                let mut degree = rng.below(5);
                let mut exponents = vec![0u64; n];
                for e in &mut exponents {
                    *e = rng.below(degree + 1);
                    degree -= *e;
                }
                if used.contains(&exponents) {
                    continue;
                }
                if negative {
                    poly.push('-');
                } else if !poly.is_empty() {
                    poly.push('+');
                }
                poly.push_str(&coef);
                for (name, &e) in names.iter().zip(&exponents) {
                    if e > 0 {
                        poly.push('*');
                        poly.push_str(name);
                        if e > 1 {
                            write!(poly, "^{e}").unwrap();
                        }
                    }
                }
                used.push(exponents);
            }
            poly
        })
        .collect();
    format!(
        "{variables}\n0\n{generators}\n",
        variables = names[..n].join(","),
        generators = generators.join(",\n")
    )
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// The instance with each generator multiplied by the lcm of its
/// denominators, which generates the same ideal over Q: for Groebner.jl,
/// which reads no fraction (log 0016-1, _Step 2_). Each term is an
/// optional sign, an optional coefficient `n` or `n/d` of at most 19
/// digits each, and variables joined by `*`.
pub fn cleared(text: &str) -> String {
    let mut lines = text.splitn(3, '\n');
    let (variables, characteristic) = (lines.next().unwrap(), lines.next().unwrap());
    let generators: Vec<String> = lines
        .next()
        .unwrap_or("")
        .split(',')
        .map(|g| g.chars().filter(|c| !c.is_whitespace()).collect::<String>())
        .filter(|g| !g.is_empty())
        .map(|g| {
            // Each term: its numerator, denominator, and monomial as written.
            let mut terms: Vec<(i128, i128, String)> = Vec::new();
            let bytes = g.as_bytes();
            let mut start = 0;
            for i in 1..=bytes.len() {
                if i < bytes.len() && bytes[i] != b'+' && bytes[i] != b'-' {
                    continue;
                }
                let term = &g[start..i];
                start = i;
                let (negative, body) = match term.as_bytes()[0] {
                    b'-' => (true, &term[1..]),
                    b'+' => (false, &term[1..]),
                    _ => (false, term),
                };
                let (mut num, mut den) = (1i128, 1i128);
                let mut monomial = Vec::new();
                for factor in body.split('*') {
                    if factor.starts_with(|c: char| c.is_ascii_digit()) {
                        let (n, d) = factor.split_once('/').unwrap_or((factor, "1"));
                        num *= n.parse::<i128>().unwrap();
                        den *= d.parse::<i128>().unwrap();
                    } else {
                        monomial.push(factor);
                    }
                }
                terms.push((if negative { -num } else { num }, den, monomial.join("*")));
            }
            let l = terms.iter().fold(1i128, |l, t| l / gcd(l, t.1) * t.1);
            let mut out = String::new();
            for (num, den, monomial) in terms {
                let c = num * (l / den);
                if c >= 0 && !out.is_empty() {
                    out.push('+');
                }
                out.push_str(&c.to_string());
                if !monomial.is_empty() {
                    out.push('*');
                    out.push_str(&monomial);
                }
            }
            out
        })
        .collect();
    format!(
        "{variables}\n{characteristic}\n{generators}\n",
        generators = generators.join(",\n")
    )
}
