//! The generated systems of the test code (`WORKFLOW.md`, step 5): random
//! systems, random square systems, dense systems, random sparse systems,
//! random boolean systems over GF(2), and random binomial systems, over
//! GF(p) and over ℚ, where p is 0. The engine's unit tests and the suites
//! of the C interface, the CLI, and the Python extension declare this file
//! by `#[path]`, and the engine's suite declares it in `common`.

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
pub const Q_BOUND: u64 = 1 << 20;

/// The bound of a drawn coefficient: p, or `Q_BOUND` over ℚ.
pub fn bound(p: u64) -> u64 {
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
pub fn term<S: AsRef<str>>(c: u64, names: &[S], e: &[u32]) -> String {
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

// fglm-lex.26 design-checked -- the random cases of the change are square systems as well, n generators on n variables with every monomial up to each generator's degree, not random_system alone (log 0012-1)
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
        // test-mq-systems.7 design-checked -- a draw of quadrics alone, the mq family's system, gets a cubic last generator, and every other draw keeps its system (not measured)
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

// test-mq-systems.4 design-checked -- a test takes dense(m, 3) for n dense quadrics, m = round(n · ln 2 / ln 3) and at least 2, whose degree 3^m, a regular system's (bardet-2015 §1.4), is the nearest to 2^n (log 0084-1)
// test-mq-systems.5 tuned-checked -- a call over ℚ that a test stops while it runs takes dense(5, 3, 0, seed), one size above the map's dense(4, 3, 0, seed), which finishes before the tests' stops (log 0084-1)
// test-mq-systems.6 design-checked -- a loop over the sizes n in a..=b runs over m in map(a)..=map(b), each size once, its seed following m where it followed n (not measured)
/// A dense instance over GF(p), or ℚ where p is 0: n generators on the
/// variables x1, …, xn, each with every monomial of degree at most d, in
/// the order of `exponents`, and a coefficient in [1, `bound(p)`). Its ideal is zero-dimensional
/// of degree d^n on most seeds, so n and d set how long a call takes.
///
/// It panics at d = 2, whose systems are the `mq` family's, which test code
/// takes at no size (`WORKFLOW.md`, step 5). The comments above say how a
/// test sizes a dense system in their place.
pub fn dense(n: usize, d: u32, p: u64, seed: u64) -> String {
    // test-mq-systems.1 design-checked -- a system is of a family of the instance corpus where the family's construction writes it, every generator as written: for mq every generator with every monomial of degree at most 2 and a random coefficient, whatever the counts of generators and variables (not measured)
    // test-mq-systems.2 design-checked -- every call of degree 2 gave way, those that stop a call, read a report, or compute in lex among them, since a lex call checks the grevlex basis FGLM changes (not measured)
    // test-mq-systems.3 design-checked -- dense refuses degree 2 by an assertion, so that a test that calls it fails at once with the rule in its message, beside the scan of the sources (not measured)
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

/// A random boolean instance over GF(2): m generators on the variables
/// x1, …, xn, each of 3 to 8 terms of degree 1 to 3, every variable of a
/// term drawn on its own, and the constant term that makes it vanish at a
/// point drawn first, then the field equations `x_i^2 + x_i`. Its ideal is
/// radical, and the drawn point is among its zeros. mqbool's and hfe's
/// generators are quadrics, sat's are clauses, and bivium's and sr's are
/// ciphers', so the system is of no family of the instance corpus
/// (`WORKFLOW.md`, step 5).
#[allow(
    dead_code,
    reason = "the engine's unit tests and the suites of the C interface, the CLI, and the Python extension declare this file and do not call it, and the engine's suite calls it from gf2_reduction_time.rs"
)]
pub fn boolean(n: usize, m: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ 0xb001_ea05);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let point: Vec<bool> = (0..n).map(|_| rng.below(2) == 1).collect();
    let mut generators: Vec<String> = (0..m)
        .map(|_| {
            // The generator's value at the point, mod 2.
            let mut value = false;
            let mut terms = Vec::new();
            for _ in 0..3 + rng.below(6) {
                let mut e = vec![0u32; n];
                for _ in 0..=rng.below(3) {
                    e[rng.below(n as u64) as usize] += 1;
                }
                value ^= e.iter().zip(&point).all(|(&k, &at)| k == 0 || at);
                terms.push(term(1, &names, &e));
            }
            if value {
                terms.push("1".to_string());
            }
            terms.join("+")
        })
        .collect();
    generators.extend(names.iter().map(|x| format!("{x}^2+{x}")));
    format!(
        "{variables}\n2\n{generators}\n",
        variables = names.join(","),
        generators = generators.join(",\n")
    )
}

/// The instance `text` on the n variables x1, …, xn: its variables, in
/// the order its first line names them, become x{p + 1} for the places p
/// of `places`, each below n and increasing, so that each keeps its rank
/// in the variable order; the other variables appear in no generator. Its
/// generators name variables, numbers, and the signs and marks between
/// them alone.
#[allow(
    dead_code,
    reason = "the engine's suite and the suites of the C interface, the CLI, and the Python extension declare this file and do not call it"
)]
pub fn placed(text: &str, places: &[usize], n: usize) -> String {
    let (names, rest) = text.split_once('\n').expect("a line of variables");
    let (characteristic, generators) = rest.split_once('\n').expect("a characteristic");
    let names: Vec<&str> = names.split(',').collect();
    assert_eq!(names.len(), places.len(), "a place per variable");
    assert!(
        places.windows(2).all(|w| w[0] < w[1]) && places.last().is_none_or(|&p| p < n),
        "increasing places below {n}"
    );
    let mut renamed = String::new();
    let mut chars = generators.chars().peekable();
    while let Some(c) = chars.next() {
        if !c.is_ascii_alphabetic() {
            renamed.push(c);
            continue;
        }
        let mut name = String::from(c);
        while let Some(&d) = chars.peek().filter(|d| d.is_ascii_alphanumeric()) {
            name.push(d);
            chars.next();
        }
        let i = names
            .iter()
            .position(|&x| x == name)
            .unwrap_or_else(|| panic!("{name} is no variable of {names:?}"));
        write!(renamed, "x{place}", place = places[i] + 1).expect("a String takes every write");
    }
    let variables: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    format!(
        "{variables}\n{characteristic}\n{renamed}",
        variables = variables.join(",")
    )
}

/// A dense boolean instance over GF(2): the variables x1, …, xn and m
/// generators, each square-free monomial of degree 3 or less a term with
/// probability 1/2, and the constant 1 where the other terms sum to 1 at a
/// drawn point, which every generator then vanishes at; then the field
/// equations xi^2 + xi. Its rounds hold rows of 16 entries or more in the
/// mean from n = 8 on.
#[allow(
    dead_code,
    reason = "the suites of the C interface, the CLI, and the Python extension declare this file and do not call it"
)]
pub fn cubics(n: usize, m: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ 0xc0b1);
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let point: Vec<bool> = (0..n).map(|_| rng.below(2) == 1).collect();
    // The square-free monomials of degree 1 to 3, as sets of variables.
    let mut sets: Vec<Vec<usize>> = Vec::new();
    for a in 0..n {
        sets.push(vec![a]);
        for b in a + 1..n {
            sets.push(vec![a, b]);
            for c in b + 1..n {
                sets.push(vec![a, b, c]);
            }
        }
    }
    let mut generators: Vec<String> = (0..m)
        .map(|_| {
            // The generator's value at the point, mod 2.
            let mut value = false;
            let mut terms = Vec::new();
            for set in &sets {
                if rng.below(2) == 0 {
                    continue;
                }
                value ^= set.iter().all(|&v| point[v]);
                let factors: Vec<&str> = set.iter().map(|&v| names[v].as_str()).collect();
                terms.push(factors.join("*"));
            }
            if value {
                terms.push("1".to_string());
            }
            if terms.is_empty() {
                "0".to_string()
            } else {
                terms.join("+")
            }
        })
        .collect();
    generators.extend(names.iter().map(|x| format!("{x}^2+{x}")));
    format!(
        "{variables}\n2\n{generators}\n",
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
