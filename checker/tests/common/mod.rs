//! Test code shared by the checker's tests: the oracle and its cache, the
//! cases it yields and a bound on their degrees, the mutants built from them
//! as text, and random and hand-written systems over GF(p) and Q.
//!
//! The mutants are built without the checker's own parser and arithmetic.
//! Every mutant is a string operation on Groebner.jl's text (checker-gfp.22), so this code
//! needs no arithmetic that could err.


use std::fmt::Write;
use std::fs;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use grobx_checker::{Mode, Order, Verdict, check};

// --- the oracle ---

fn oracle_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let out = Command::new("groebner-jl").arg("--version").output().unwrap_or_else(|e| {
            panic!("groebner-jl does not run ({e}): the Accept cases need it (AUTONOMOUS.md, _What is installed_)")
        });
        assert!(out.status.success(), "groebner-jl --version failed: {out:?}");
        String::from_utf8(out.stdout).expect("groebner-jl --version prints UTF-8").trim().to_string()
    })
}

fn cache_dir() -> PathBuf {
    let version: String = oracle_version()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("oracle")
        .join(version);
    fs::create_dir_all(&dir).expect("the oracle cache directory can be created");
    dir
}

/// Writes `text` to `path` through a temporary file, so that a reader never
/// sees half of it.
fn write_atomic(path: &PathBuf, text: &str) {
    let tmp = path.with_extension(format!(
        "tmp-{process}-{thread:?}",
        process = std::process::id(),
        thread = std::thread::current().id()
    ));
    fs::write(&tmp, text).expect("the oracle cache can be written");
    fs::rename(&tmp, path).expect("the oracle cache can be written");
}

/// What `groebner-jl --cofactors` prints for `instance`, from the cache where
/// the cache holds the same instance text under the same Groebner.jl version
/// (checker-gfp.19).
// checker-gfp.19 design -- the oracle's output is cached under the cargo target directory, not committed (log 0001-1)
pub fn oracle(name: &str, instance: &str) -> String {
    oracle_with(name, instance, &[])
}

/// What `groebner-jl <args> --cofactors` prints for `instance`, cached as
/// `oracle` caches it, under a file name that carries the arguments.
pub fn oracle_with(name: &str, instance: &str, args: &[&str]) -> String {
    let dir = cache_dir();
    let tag: String = args
        .join(" ")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let stem = if tag.is_empty() {
        name.to_string()
    } else {
        format!("{name}~{tag}")
    };
    let input = dir.join(format!("{stem}.ms"));
    let output = dir.join(format!("{stem}.out"));
    if fs::read_to_string(&input).is_ok_and(|cached| cached == instance)
        && let Ok(out) = fs::read_to_string(&output)
    {
        return out;
    }
    let run = dir.join(format!(
        "{stem}.run-{process}-{thread:?}.ms",
        process = std::process::id(),
        thread = std::thread::current().id()
    ));
    fs::write(&run, instance).expect("the instance can be written for groebner-jl");
    let out = Command::new("groebner-jl").args(args).arg("--cofactors").arg(&run).output().unwrap_or_else(|e| {
        panic!("groebner-jl does not run ({e}): the Accept cases need it (AUTONOMOUS.md, _What is installed_)")
    });
    let _ = fs::remove_file(&run);
    assert!(
        out.status.success(),
        "groebner-jl {args:?} --cofactors failed on {name}: {status}\n{stderr}",
        status = out.status,
        stderr = String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("groebner-jl prints UTF-8");
    // The output first, then the instance: a crash between the two leaves an
    // instance that does not match, and the next run computes again.
    write_atomic(&output, &text);
    write_atomic(&input, instance);
    text
}

// --- cases ---

/// A result to rule on: an instance, a basis, and its cofactor matrix, each
/// element and each row kept as its own text.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub instance: String,
    pub basis: Vec<String>,
    /// The inside of each row, between its brackets: entries joined by commas.
    pub rows: Vec<String>,
}

/// The inside of the outermost brackets of `text`, which must end in `]:`,
/// after leading comment lines.
fn bracketed(text: &str) -> &str {
    let mut rest = text.trim_start();
    while rest.starts_with('#') {
        rest = rest.split_once('\n').map_or("", |(_, r)| r).trim_start();
    }
    let rest = rest.trim_end();
    let inner = rest
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(':'))
        .and_then(|r| r.trim_end().strip_suffix(']'));
    inner.unwrap_or_else(|| panic!("not a bracketed list: {text:.200}"))
}

fn split_list(inner: &str) -> Vec<String> {
    if inner.trim().is_empty() {
        Vec::new()
    } else {
        inner.split(',').map(|s| s.trim().to_string()).collect()
    }
}

/// The rows of a matrix `[[a,b],[c,d]]:`, each as the text between its
/// brackets.
fn split_rows(inner: &str) -> Vec<String> {
    let mut rows = Vec::new();
    let mut rest = inner.trim();
    while !rest.is_empty() {
        let open = rest
            .strip_prefix('[')
            .unwrap_or_else(|| panic!("a row does not open: {rest:.200}"));
        let close = open
            .find(']')
            .unwrap_or_else(|| panic!("a row does not close: {rest:.200}"));
        rows.push(open[..close].trim().to_string());
        rest = open[close + 1..].trim_start();
        rest = rest.strip_prefix(',').unwrap_or(rest).trim_start();
    }
    rows
}

impl Case {
    /// The case Groebner.jl's output describes. Zero elements and their rows
    /// are dropped: Groebner.jl writes `[0]` for the zero ideal, whose reduced
    /// basis is empty.
    pub fn from_oracle(name: &str, instance: &str, output: &str) -> Case {
        let end = output
            .find("]:")
            .unwrap_or_else(|| panic!("{name}: no basis in the oracle's output"))
            + 2;
        let basis = split_list(bracketed(&output[..end]));
        let rows = split_rows(bracketed(&output[end..]));
        assert_eq!(
            basis.len(),
            rows.len(),
            "{name}: the oracle wrote a row per element"
        );
        let (basis, rows) = basis
            .into_iter()
            .zip(rows)
            .filter(|(g, _)| g != "0")
            .unzip();
        Case {
            name: name.to_string(),
            instance: instance.to_string(),
            basis,
            rows,
        }
    }

    /// Runs the oracle on `instance` and reads its output.
    pub fn solved(name: &str, instance: &str) -> Case {
        let output = oracle(name, instance);
        Case::from_oracle(name, instance, &output)
    }

    /// The characteristic: a prime, or 0 for Q.
    pub fn prime(&self) -> u64 {
        let line = self
            .instance
            .lines()
            .nth(1)
            .expect("an instance has a characteristic line");
        line.trim().parse().expect("the characteristic is a number")
    }

    pub fn variables(&self) -> Vec<String> {
        let line = self
            .instance
            .lines()
            .next()
            .expect("an instance has a variable line");
        line.split(',').map(|s| s.trim().to_string()).collect()
    }

    /// The generators of the instance, each as its text.
    pub fn generators(&self) -> Vec<String> {
        let body: Vec<&str> = self.instance.lines().skip(2).collect();
        split_list(&body.join("\n"))
    }

    pub fn basis_text(&self) -> String {
        basis_text(&self.basis)
    }

    pub fn cofactor_text(&self) -> String {
        matrix_text(self.rows.iter().map(String::as_str))
    }

    pub fn rule(&self) -> Verdict {
        rule(
            &self.name,
            &self.instance,
            &self.basis_text(),
            &self.cofactor_text(),
        )
    }

    pub fn is_unit(&self) -> bool {
        self.basis.len() == 1 && self.basis[0] == "1"
    }

    /// A bound on the degree of Σⱼ `C_ij·f_j` − `g_i` for every row i: the
    /// cofactors' largest degree plus the generators', or the basis's
    /// largest degree where that is larger.
    pub fn degree_bound(&self) -> u64 {
        let rows = self.rows.iter().map(|r| max_degree(r)).max().unwrap_or(0);
        let generators = self
            .generators()
            .iter()
            .map(|f| max_degree(f))
            .max()
            .unwrap_or(0);
        let basis = self.basis.iter().map(|g| max_degree(g)).max().unwrap_or(0);
        (rows + generators).max(basis)
    }
}

/// The largest total degree of a term in `text`, read from its variable
/// factors: a name counts 1, or the exponent after its `^`. Numbers outside
/// exponents count nothing. A sign, a comma, a bracket, and a line break end
/// a term.
pub fn max_degree(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let (mut max, mut degree, mut i) = (0, 0, 0);
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphabetic() || b == b'_' {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if bytes.get(i) == Some(&b'^') {
                let start = i + 1;
                i = start;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                degree += text[start..i].parse::<u64>().unwrap_or_else(|_| {
                    panic!("an exponent after `^`: {rest:.200}", rest = &text[start..])
                });
            } else {
                degree += 1;
            }
            continue;
        }
        if matches!(b, b'+' | b'-' | b',' | b'[' | b']' | b'\n') {
            max = max.max(degree);
            degree = 0;
        }
        i += 1;
    }
    max.max(degree)
}

/// The canonical spelling of a list of polynomials: `[`, the elements joined
/// by a comma and a line break, `]:` and a line break.
pub fn basis_text<S: AsRef<str>>(elements: &[S]) -> String {
    let mut text = String::from("[");
    for (i, g) in elements.iter().enumerate() {
        if i > 0 {
            text.push_str(",\n");
        }
        text.push_str(g.as_ref());
    }
    text.push_str("]:\n");
    text
}

pub fn matrix_text<'a>(rows: impl Iterator<Item = &'a str>) -> String {
    let mut text = String::from("[");
    for (i, row) in rows.enumerate() {
        if i > 0 {
            text.push_str(",\n");
        }
        text.push('[');
        text.push_str(row);
        text.push(']');
    }
    text.push_str("]:\n");
    text
}

/// The checker's verdict in exact mode; an error fails the test.
pub fn rule(name: &str, instance: &str, basis: &str, cofactors: &str) -> Verdict {
    check(instance, basis, cofactors, &Order::Grevlex, Mode::Exact)
        .unwrap_or_else(|e| panic!("{name}: the checker refused a well-formed result: {e:?}"))
}

// --- text surgery ---

/// The terms of a polynomial's text, each as its sign (true for minus) and
/// its body without the sign.
pub fn terms(poly: &str) -> Vec<(bool, String)> {
    let mut out = Vec::new();
    let mut negative = false;
    let mut body = String::new();
    for c in poly.chars().filter(|c| !c.is_whitespace()) {
        if (c == '+' || c == '-') && !body.is_empty() {
            out.push((negative, std::mem::take(&mut body)));
            negative = c == '-';
        } else if c == '+' || c == '-' {
            negative ^= c == '-';
        } else {
            body.push(c);
        }
    }
    assert!(
        !body.is_empty(),
        "a polynomial ends in a sign or is empty: {poly:?}"
    );
    out.push((negative, body));
    out
}

/// `factor` times the polynomial `poly`, by putting `factor*` in front of
/// every term.
pub fn times(factor: &str, poly: &str) -> String {
    let mut out = String::new();
    for (negative, body) in terms(poly) {
        out.push(if negative { '-' } else { '+' });
        out.push_str(factor);
        out.push('*');
        out.push_str(&body);
    }
    out
}

/// `poly` plus `addend`, by appending the addend's terms.
pub fn plus(poly: &str, addend: &str) -> String {
    let mut out = poly.to_string();
    for (negative, body) in terms(addend) {
        out.push(if negative { '-' } else { '+' });
        out.push_str(&body);
    }
    out
}

pub fn is_zero_text(poly: &str) -> bool {
    poly.trim() == "0"
}

// --- mutants ---

/// A mutation of an accepted case (`OBJECTIVES.md`, _Checker_, Reject).
#[derive(Clone, Debug)]
pub enum Mutation {
    /// Element k and row k are removed.
    Dropped { k: usize },
    /// `δ·m·g_k` with the row `δ·m·C_k` is put first.
    IdealElement { k: usize, factor: String },
    /// `δ·m·g_k` + ε with the row `δ·m·C_k` is put first; over the empty basis, ε
    /// with a zero row.
    OutsideElement {
        k: Option<usize>,
        factor: String,
        epsilon: String,
    },
    /// δ times term t of `g_k` is added to `g_k`.
    Coefficient { k: usize, t: usize, delta: String },
    /// δ·m is added to the cofactor of generator j in row k.
    Cofactor { k: usize, j: usize, factor: String },
    /// G becomes {1}, with row 0 of C or a zero row.
    One,
}

/// What the checker must say about a mutant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    NotGroebner,
    GroebnerNotReduced,
    /// The mutant is the case itself, and so is the verdict.
    Unchanged,
}

/// A splitmix64 generator: the tests draw from a fixed seed, so a failure
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

    /// A number in [0, n), n > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A number in [1, p), p ≥ 2.
    pub fn unit(&mut self, p: u64) -> u64 {
        1 + self.below(p - 1)
    }

    pub fn index(&mut self, n: usize) -> usize {
        self.below(n as u64) as usize
    }
}

// checker-q.16 design -- over Q, δ and ε are positive fractions a/b, 1 ≤ a ≤ 2^62, 1 ≤ b ≤ 2^20 (not measured)
/// A nonzero number to scale by, as text: over GF(p) a residue in [1, p),
/// over Q (p = 0) a positive fraction a/b with 1 ≤ a ≤ 2^62 and
/// 1 ≤ b ≤ 2^20. Over GF(p) it draws what `Rng::unit` draws.
pub fn number(rng: &mut Rng, p: u64) -> String {
    if p > 0 {
        return rng.unit(p).to_string();
    }
    let a = 1 + rng.below(1 << 62);
    let b = 1 + rng.below(1 << 20);
    format!("{a}/{b}")
}

/// A random monomial of degree 0 to 3 in `variables`, as text.
pub fn monomial(rng: &mut Rng, variables: &[String]) -> String {
    let mut exponents = vec![0u32; variables.len()];
    for _ in 0..rng.below(4) {
        exponents[rng.index(variables.len())] += 1;
    }
    let factors: Vec<String> = variables
        .iter()
        .zip(&exponents)
        .filter(|(_, e)| **e > 0)
        .map(|(v, e)| {
            if *e == 1 {
                v.clone()
            } else {
                format!("{v}^{e}")
            }
        })
        .collect();
    if factors.is_empty() {
        "1".to_string()
    } else {
        factors.join("*")
    }
}

/// Every mutation that applies to `case`, each with its own random choices.
pub fn mutations(case: &Case, rng: &mut Rng) -> Vec<Mutation> {
    let p = case.prime();
    let variables = case.variables();
    let n = case.basis.len();
    let mut out = Vec::new();
    let factor = |rng: &mut Rng| {
        format!(
            "{coefficient}*{monomial}",
            coefficient = number(rng, p),
            monomial = monomial(rng, &variables)
        )
    };
    if n > 0 {
        out.push(Mutation::Dropped { k: rng.index(n) });
        out.push(Mutation::IdealElement {
            k: rng.index(n),
            factor: factor(rng),
        });
        let k = rng.index(n);
        out.push(Mutation::Coefficient {
            k,
            t: rng.index(terms(&case.basis[k]).len()),
            delta: number(rng, p),
        });
    }
    if !case.is_unit() {
        let k = if n > 0 { Some(rng.index(n)) } else { None };
        out.push(Mutation::OutsideElement {
            k,
            factor: factor(rng),
            epsilon: number(rng, p),
        });
    }
    let generators = case.generators();
    let live: Vec<usize> = (0..generators.len())
        .filter(|&j| !is_zero_text(&generators[j]))
        .collect();
    if n > 0 && !live.is_empty() {
        let j = live[rng.index(live.len())];
        out.push(Mutation::Cofactor {
            k: rng.index(n),
            j,
            factor: factor(rng),
        });
    }
    out.push(Mutation::One);
    out
}

impl Mutation {
    pub fn expect(&self, case: &Case) -> Expect {
        match self {
            Mutation::IdealElement { .. } => Expect::GroebnerNotReduced,
            Mutation::One if case.is_unit() => Expect::Unchanged,
            Mutation::Dropped { .. }
            | Mutation::OutsideElement { .. }
            | Mutation::Coefficient { .. }
            | Mutation::Cofactor { .. }
            | Mutation::One => Expect::NotGroebner,
        }
    }

    // checker-gfp.22 design -- mutants are built as text: factors put in front of every term, a term appended (not measured)
    /// The mutant's basis text and cofactor text.
    pub fn apply(&self, case: &Case) -> (String, String) {
        let width = case.generators().len();
        let zero_row = vec!["0"; width].join(",");
        let row_times = |factor: &str, row: &str| -> String {
            if width == 0 {
                return String::new();
            }
            row.split(',')
                .map(|c| {
                    if is_zero_text(c) {
                        "0".to_string()
                    } else {
                        times(factor, c)
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        match self {
            Mutation::Dropped { k } => {
                let basis: Vec<&String> = case
                    .basis
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i != k)
                    .map(|(_, g)| g)
                    .collect();
                let rows = case
                    .rows
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i != k)
                    .map(|(_, r)| r.as_str());
                (basis_text(&basis), matrix_text(rows))
            }
            Mutation::IdealElement { k, factor } => {
                let h = times(factor, &case.basis[*k]);
                let row = row_times(factor, &case.rows[*k]);
                let mut basis = vec![h];
                basis.extend(case.basis.iter().cloned());
                let rows =
                    std::iter::once(row.as_str()).chain(case.rows.iter().map(String::as_str));
                (basis_text(&basis), matrix_text(rows))
            }
            Mutation::OutsideElement { k, factor, epsilon } => {
                let (h, row) = match k {
                    Some(k) => (
                        plus(&times(factor, &case.basis[*k]), epsilon),
                        row_times(factor, &case.rows[*k]),
                    ),
                    None => (epsilon.clone(), zero_row.clone()),
                };
                let mut basis = vec![h];
                basis.extend(case.basis.iter().cloned());
                let rows =
                    std::iter::once(row.as_str()).chain(case.rows.iter().map(String::as_str));
                (basis_text(&basis), matrix_text(rows))
            }
            Mutation::Coefficient { k, t, delta } => {
                let (negative, body) = terms(&case.basis[*k]).swap_remove(*t);
                let sign = if negative { '-' } else { '+' };
                let mut basis = case.basis.clone();
                basis[*k] = format!("{element}{sign}{delta}*{body}", element = case.basis[*k]);
                (basis_text(&basis), case.cofactor_text())
            }
            Mutation::Cofactor { k, j, factor } => {
                let mut entries: Vec<String> = case.rows[*k]
                    .split(',')
                    .map(|c| c.trim().to_string())
                    .collect();
                entries[*j] = plus(&entries[*j], factor);
                let row = entries.join(",");
                let rows = case
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(i, r)| if i == *k { row.as_str() } else { r.as_str() });
                (case.basis_text(), matrix_text(rows))
            }
            Mutation::One => {
                if case.is_unit() {
                    return (case.basis_text(), case.cofactor_text());
                }
                let row = case.rows.first().cloned().unwrap_or(zero_row);
                (
                    basis_text(&["1"]),
                    matrix_text(std::iter::once(row.as_str())),
                )
            }
        }
    }
}

/// Accept: the checker says the case is a reduced Gröbner basis. Reject:
/// every mutant gets its predicted verdict. A refusal of the case itself
/// fails the test at once; every other failure is gathered into one
/// message.
pub fn accept_and_reject(case: &Case, seed: u64) {
    let mut failures = Vec::new();
    let verdict = case.rule();
    if verdict.groebner.is_err() || verdict.reduced.is_err() {
        failures.push(format!("Accept: {verdict:?}"));
    }
    let mut rng = Rng::new(seed);
    for mutation in mutations(case, &mut rng) {
        let (basis, cofactors) = mutation.apply(case);
        let got = match check(
            &case.instance,
            &basis,
            &cofactors,
            &Order::Grevlex,
            Mode::Exact,
        ) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{mutation:?}: refused: {e:?}"));
                continue;
            }
        };
        let ok = match mutation.expect(case) {
            Expect::NotGroebner => got.groebner.is_err(),
            Expect::GroebnerNotReduced => got.groebner.is_ok() && got.reduced.is_err(),
            Expect::Unchanged => got == verdict,
        };
        if !ok {
            failures.push(format!(
                "{mutation:?}: expected {expected:?}, got {got:?}",
                expected = mutation.expect(case)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{name}:\n{failures}",
        name = case.name,
        failures = failures.join("\n")
    );
}

// --- random systems ---

/// A random instance over GF(p): two to four of the variables x, y, z, w,
/// one generator up to one per variable, so that the ideal is seldom the unit
/// ideal, each of one to five distinct terms of degree three or less with a
/// coefficient in [1, p) and a random sign: two to five monomials are drawn,
/// and a repeated one is dropped.
pub fn random_system(p: u64, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(17));
    random_instance(&mut rng, p, |rng| rng.unit(p).to_string())
}

// checker-q.17 design -- random systems over Q have integer coefficients of 1 to 3 digits, or of 20 to 25 (not measured)
/// A random instance over Q, shaped as `random_system`'s, whose coefficients
/// are integers with a number of digits in `digits`. Groebner.jl takes no
/// fraction in an instance.
pub fn random_system_q(digits: RangeInclusive<usize>, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ (*digits.end() as u64).rotate_left(29));
    random_instance(&mut rng, 0, |rng| {
        let span = (digits.end() - digits.start() + 1) as u64;
        let n = digits.start() + rng.below(span) as usize;
        let mut text = (1 + rng.below(9)).to_string();
        for _ in 1..n {
            text.push_str(&rng.below(10).to_string());
        }
        text
    })
}

/// The shape both random systems share, each coefficient from `coefficient`.
fn random_instance(
    rng: &mut Rng,
    characteristic: u64,
    mut coefficient: impl FnMut(&mut Rng) -> String,
) -> String {
    let names = ["x", "y", "z", "w"];
    let n = 2 + rng.index(3);
    let variables: Vec<String> = names[..n].iter().map(std::string::ToString::to_string).collect();
    let mut generators = Vec::new();
    for _ in 0..=rng.index(n) {
        let mut seen = Vec::new();
        let mut poly = String::new();
        for _ in 0..2 + rng.index(4) {
            let m = monomial(rng, &variables);
            if seen.contains(&m) {
                continue;
            }
            seen.push(m.clone());
            let sign = if rng.below(2) == 0 { '+' } else { '-' };
            if !poly.is_empty() || sign == '-' {
                poly.push(sign);
            }
            write!(poly, "{coefficient}*{m}", coefficient = coefficient(rng)).unwrap();
        }
        generators.push(poly);
    }
    format!(
        "{variables}\n{characteristic}\n{generators}\n",
        variables = variables.join(","),
        generators = generators.join(",\n")
    )
}

/// Three quadrics in x, y, z over characteristic `p`, 0 for Q: a hand-written
/// zero-dimensional instance whose reduced basis holds several elements in
/// grevlex, lex, and `block:1`.
pub fn quadrics(p: u64) -> String {
    format!("x,y,z\n{p}\nx^2+y*z-2*z,\ny^2-x*z+3*x,\nz^2+x*y-5*y\n")
}
