//! The generated systems of the test code (`WORKFLOW.md`, step 5): random
//! systems, random square systems, dense systems, random sparse systems,
//! and random binomial systems, over GF(p) and over
//! ℚ, where p is 0. The engine's unit tests and the suites of the C
//! interface, the CLI, and the Python extension declare this file by
//! `#[path]`, and the engine's suite declares it in `common`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::fmt::Write;

/// A splitmix64 generator: the tests draw from fixed seeds, so a failure
/// comes back on the next run.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The bound of a drawn coefficient over ℚ.
const Q_BOUND: u64 = 1 << 20;

/// The bound of a drawn coefficient: p, or `Q_BOUND` over ℚ.
fn bound(p: u64) -> u64 {
    if p == 0 { Q_BOUND } else { p }
}

/// The exponent vectors on n variables of total degree at most d, in the
/// order of their digits in [0, d], the first variable's the most
/// significant.
fn exponents(n: usize, d: u32) -> Vec<Vec<u32>> {
    let mut all = Vec::new();
    let mut e = vec![0u32; n];
    loop {
        if e.iter().sum::<u32>() <= d {
            all.push(e.clone());
        }
        // The next vector of digits, the last variable's the least
        // significant.
        let mut i = n;
        while i > 0 {
            i -= 1;
            if e[i] < d {
                e[i] += 1;
                break;
            }
            e[i] = 0;
        }
        if e.iter().all(|&k| k == 0) {
            break;
        }
    }
    all
}

/// The term c times the monomial of exponents `e` on `names`.
fn term<S: AsRef<str>>(c: u64, names: &[S], e: &[u32]) -> String {
    let mut term = c.to_string();
    for (name, &k) in names.iter().zip(e) {
        if k > 0 {
            term.push('*');
            term.push_str(name.as_ref());
            if k > 1 {
                write!(term, "^{k}").unwrap();
            }
        }
    }
    term
}

/// A random instance over GF(p), or ℚ where p is 0: two to four of the
/// variables x, y, z, w, one generator up to one more than the variables,
/// each of one to five terms of degree four or less with a coefficient in
/// [1, `bound(p)`) and a random sign.
#[allow(
    dead_code,
    reason = "the suites of the C interface, the CLI, and the Python extension declare this file and do not call it"
)]
pub fn random_system(p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(23));
    let names = ["x", "y", "z", "w"];
    let n = 2 + rng.below(3) as usize;
    let generators: Vec<String> = (0..=rng.below(n as u64 + 1))
        .map(|_| {
            let mut poly = String::new();
            for _ in 0..=rng.below(5) {
                let sign = if rng.below(2) == 0 { "+" } else { "-" };
                if !poly.is_empty() || sign == "-" {
                    poly.push_str(sign);
                }
                poly.push_str(&(1 + rng.below(bound(p) - 1)).to_string());
                let mut degree = rng.below(5);
                for name in &names[..n] {
                    let e = rng.below(degree + 1);
                    degree -= e;
                    if e > 0 {
                        poly.push('*');
                        poly.push_str(name);
                        if e > 1 {
                            write!(poly, "^{e}").unwrap();
                        }
                    }
                }
            }
            poly
        })
        .collect();
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names[..n].join(","),
        generators = generators.join(",\n")
    )
}

// fglm-lex.26 design -- the random cases of the change are square systems as well, n generators on n variables with every monomial up to each generator's degree, not random_system alone (log 0012-1)
/// A random square instance over GF(p), or ℚ where p is 0: two to four of
/// the variables x, y, z, w and as many generators, each of a degree from 1
/// to 3 with a coefficient in [0, `bound(p)`) for every monomial of at most that degree, the
/// exponent vectors in the order of `exponents`; zero terms are left out.
/// Where every degree drawn is 2, the last generator is of degree 3. Its
/// ideal is zero-dimensional on most seeds.
#[allow(
    dead_code,
    reason = "the Python extension's suite declares this file and does not call it"
)]
pub fn random_square(p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(29) ^ 0x5157);
    let names = ["x", "y", "z", "w"];
    let n = 2 + rng.below(3) as usize;
    let mut generators = Vec::new();
    let mut quadrics = true;
    for i in 0..n {
        let mut d = 1 + rng.below(3) as u32;
        quadrics &= d == 2;
        // test-mq-systems.7 design -- a draw of quadrics alone, the mq family's system, gets a cubic last generator, and every other draw keeps its system (not measured)
        if quadrics && i == n - 1 {
            d = 3;
        }
        let mut terms = Vec::new();
        for e in exponents(n, d) {
            let c = rng.below(bound(p));
            if c != 0 {
                terms.push(term(c, &names, &e));
            }
        }
        generators.push(if terms.is_empty() {
            "0".to_string()
        } else {
            terms.join("+")
        });
    }
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names[..n].join(","),
        generators = generators.join(",\n")
    )
}

// test-mq-systems.4 design -- a test takes dense(m, 3) for n dense quadrics, m = round(n · ln 2 / ln 3) and at least 2, whose degree 3^m, a regular system's (bardet-2015 §1.4), is the nearest to 2^n (log 0084-1)
// test-mq-systems.5 design -- a call over ℚ that a test stops while it runs takes dense(5, 3, 0, seed), one size above the map's dense(4, 3, 0, seed), which finishes before the tests' stops (log 0084-1)
// test-mq-systems.6 design -- a loop over the sizes n in a..=b runs over m in map(a)..=map(b), each size once, its seed following m where it followed n (not measured)
/// A dense instance over GF(p), or ℚ where p is 0: n generators on the
/// variables x1, …, xn, each with every monomial of degree at most d, in
/// the order of `exponents`, and a coefficient in [1, `bound(p)`). Its ideal is zero-dimensional
/// of degree d^n on most seeds, so n and d set how long a call takes.
///
/// It panics at d = 2, whose systems are the `mq` family's, which test code
/// takes at no size (`WORKFLOW.md`, step 5). The comments above say how a
/// test sizes a dense system in their place.
pub fn dense(n: usize, d: u32, p: u64, seed: u64) -> String {
    // test-mq-systems.1 design -- a system is of a family of the instance corpus where the family's construction writes it, every generator as written: for mq every generator with every monomial of degree at most 2 and a random coefficient, whatever the counts of generators and variables (not measured)
    // test-mq-systems.2 design -- every call of degree 2 gave way, those that stop a call, read a report, or compute in lex among them, since a lex call checks the grevlex basis FGLM changes (not measured)
    // test-mq-systems.3 design -- dense refuses degree 2 by an assertion, so that a test that calls it fails at once with the rule in its message, beside the scan of the sources (not measured)
    assert!(
        d != 2,
        "dense of degree 2 writes the mq family's systems, which test code takes at no size (WORKFLOW.md, step 5)"
    );
    let mut rng = Rng::new(seed ^ p.rotate_left(31) ^ 0xd5e5);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let monomials = exponents(n, d);
    let generators: Vec<String> = (0..n)
        .map(|_| {
            let terms: Vec<String> = monomials
                .iter()
                .map(|e| term(1 + rng.below(bound(p) - 1), &names, e))
                .collect();
            terms.join("+")
        })
        .collect();
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// A random sparse instance over GF(p), or ℚ where p is 0: n generators on
/// the variables x1, …, xn, each of four to eight terms of degree 1 to 3,
/// every variable of a term drawn on its own, and a constant term, each
/// with a coefficient in [1, `bound(p)`). Its F4 rounds hold many rows that
/// reduce to zero.
#[allow(
    dead_code,
    reason = "the engine's unit tests and the suites of the C interface, the CLI, and the Python extension declare this file and do not call it"
)]
pub fn random_sparse(n: usize, p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(37) ^ 0x5a75);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let generators: Vec<String> = (0..n)
        .map(|_| {
            let mut terms: Vec<String> = (0..4 + rng.below(5))
                .map(|_| {
                    let mut e = vec![0u32; n];
                    for _ in 0..=rng.below(3) {
                        e[rng.below(n as u64) as usize] += 1;
                    }
                    term(1 + rng.below(bound(p) - 1), &names, &e)
                })
                .collect();
            terms.push((1 + rng.below(bound(p) - 1)).to_string());
            terms.join("+")
        })
        .collect();
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// A random binomial instance over GF(p), or ℚ where p is 0: the eight
/// variables a to h and 8 to 12 generators, each m₁ − c·m₂ for random
/// monomials m₁ ≠ m₂ of one degree, 3 or 4 for the whole instance, and c
/// in [1, `bound(p)`). Its matrices' rows hold two entries and reduce
/// along chains across the columns (`row-scan-time`).
#[allow(
    dead_code,
    reason = "the engine's unit tests and the suites of the C interface, the CLI, and the Python extension declare this file and do not call it"
)]
pub fn random_binomials(p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(29) ^ 0xb170);
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let degree = 3 + rng.below(2);
    let count = 8 + rng.below(5) as usize;
    let monomial = |rng: &mut Rng| {
        let mut e = [0u32; 8];
        for _ in 0..degree {
            e[rng.below(8) as usize] += 1;
        }
        e
    };
    let mut generators = Vec::new();
    while generators.len() < count {
        let (m1, m2) = (monomial(&mut rng), monomial(&mut rng));
        if m1 == m2 {
            continue;
        }
        let c = 1 + rng.below(bound(p) - 1);
        generators.push(format!(
            "{first}-{second}",
            first = term(1, &names, &m1),
            second = term(c, &names, &m2)
        ));
    }
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}
