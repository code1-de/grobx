//! Test code shared by the engine's tests: the oracles, the checker, msolve,
//! and Groebner.jl's digests, the generated systems, and the
//! degree bound that sizes the checker's cheap mode.
//!
//! The digest here comes from sha2 directly, not from `grobx::digest`, and
//! msolve's output is converted to the canonical text by string operations
//! alone, so that no oracle leans on the engine.

pub mod sources;
pub mod systems;

use std::collections::BTreeMap;
use std::fmt::{self, Write};
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use grobx::{MonomialOrder, Outcome, Params, Threads};
use grobx_checker::{Mode, Order, Verdict, check};
use sha2::{Digest, Sha256};

pub use systems::{Rng, dense, random_square, random_system};

/// The digests of Groebner.jl's bases (`groebner_jl_reference`).
pub const GROEBNER_JL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../groebner-jl.json");

pub const P: u64 = 65521;
/// The largest prime below 2^31, the largest `Gf31` takes (f4-gfp.9).
pub const P31: u64 = 2_147_483_647;
/// The primes of the random systems.
pub const PRIMES: [u64; 5] = [2, 3, 7, P, P31];

pub fn sha256(text: &str) -> String {
    let mut hex = String::with_capacity(64);
    for b in Sha256::digest(text.as_bytes()) {
        write!(hex, "{b:02x}").unwrap();
    }
    hex
}

// --- calls ---

/// A trace that listens and keeps nothing: a run under it reduces each row
/// of its matrices on its own (`grobx::trace::Trace::LISTENS`).
pub struct Listening;

impl grobx::trace::Trace for Listening {
    fn matrix(&mut self, _: &[grobx::trace::RowSource]) {}

    fn counted(&mut self, _: &[grobx::id::RowId]) {}
}

/// The parameters of a measured call (`OBJECTIVES.md`, _Measurement_), every
/// bound absent.
pub fn params(seed: u64) -> Params {
    Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    )
}

pub fn with_cofactors(seed: u64) -> Params {
    let mut p = params(seed);
    p.cofactors = true;
    p
}

/// `params` in the monomial order named `order`.
pub fn params_in(order: &str, seed: u64) -> Params {
    let mut p = params(seed);
    p.order = order.parse().expect("a name of the family");
    p
}

/// `with_cofactors` in the monomial order named `order`.
pub fn with_cofactors_in(order: &str, seed: u64) -> Params {
    let mut p = params_in(order, seed);
    p.cofactors = true;
    p
}

/// The outcome of a call that must succeed and must not be stopped.
pub fn run(instance: &str, params: &Params) -> Outcome {
    let outcome = grobx::groebner(instance, params)
        .unwrap_or_else(|e| panic!("groebner failed: {e:?}\non\n{instance}"));
    assert!(
        outcome.stopped.is_none() && outcome.basis.is_some(),
        "the call stopped: {outcome:?}"
    );
    outcome
}

pub fn basis(instance: &str) -> String {
    run(instance, &params(1)).basis.unwrap()
}

// --- the checker ---

/// The checker's verdict on an outcome computed with cofactors in `order`.
pub fn verdict_in(instance: &str, outcome: &Outcome, order: &Order, mode: Mode) -> Verdict {
    let basis = outcome.basis.as_deref().expect("a basis");
    let cofactors = outcome.cofactors.as_deref().expect("cofactors");
    check(instance, basis, cofactors, order, mode).unwrap_or_else(|e| {
        panic!("the checker cannot read the result: {e:?}\n{basis}\n{cofactors}")
    })
}

/// The checker accepts the outcome: a Gröbner basis of the instance's
/// ideal, and reduced.
pub fn assert_accepted(name: &str, instance: &str, outcome: &Outcome, mode: Mode) {
    assert_accepted_in(name, instance, outcome, &Order::Grevlex, mode);
}

/// The checker accepts the outcome in `order`: a Gröbner basis of the
/// instance's ideal, and reduced.
pub fn assert_accepted_in(
    name: &str,
    instance: &str,
    outcome: &Outcome,
    order: &Order,
    mode: Mode,
) {
    let v = verdict_in(instance, outcome, order, mode);
    assert!(
        v.groebner.is_ok() && v.reduced.is_ok(),
        "{name}: the checker rejects the result in {order:?}: {v:?}\n\
         {instance}\n{basis}\n{cofactors}",
        basis = outcome.basis.as_deref().unwrap_or(""),
        cofactors = outcome.cofactors.as_deref().unwrap_or("")
    );
}

/// The largest total degree of a term in `text`, read from its variable
/// factors alone.
pub fn max_degree(text: &str) -> u64 {
    let mut best = 0;
    let mut term = 0u64;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_alphabetic() || c == '_' {
            while chars
                .peek()
                .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
            {
                chars.next();
            }
            let mut e = 1;
            if chars.peek() == Some(&'^') {
                chars.next();
                let mut digits = String::new();
                while chars.peek().is_some_and(char::is_ascii_digit) {
                    digits.push(chars.next().unwrap());
                }
                e = digits.parse().unwrap();
            }
            term += e;
        } else if matches!(c, '+' | '-' | ',' | '[' | ']' | '\n') {
            best = best.max(term);
            term = 0;
        }
    }
    best.max(term)
}

/// A bound on the degree of every cofactor row's difference: the basis's
/// largest degree, or the cofactors' plus the generators'.
pub fn degree_bound(instance: &str, outcome: &Outcome) -> u64 {
    let generators = instance.splitn(3, '\n').nth(2).unwrap_or("");
    let basis = max_degree(outcome.basis.as_deref().unwrap_or(""));
    let rows = max_degree(outcome.cofactors.as_deref().unwrap_or("")) + max_degree(generators);
    basis.max(rows)
}

/// The points of the cheap mode as checker-cheap.16 chooses them over GF(p):
/// the least k with (d′/p)^k ≤ 2^−64, d′ = min(d, p − 1), at least 1.
pub fn points(p: u64, d: u64) -> u32 {
    let d = d.clamp(1, p - 1) as f64;
    let per_point = (p as f64 / d).log2();
    ((64.0 / per_point).ceil() as u32).max(1)
}

// --- msolve, the second oracle (f4-gfp.41) ---

fn msolve_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let out = Command::new("msolve").arg("-V").output().unwrap_or_else(|e| {
            panic!("msolve does not run ({e}): the test code needs it (AUTONOMOUS.md, _What is installed_)")
        });
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(!text.is_empty(), "msolve -V printed nothing: {out:?}");
        text
    })
}

fn msolve_cache() -> PathBuf {
    let version: String = msolve_version()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("msolve")
        .join(version);
    fs::create_dir_all(&dir).expect("the msolve cache directory can be created");
    dir
}

// f4-gfp.41 design -- msolve, run as a black box, is a second oracle on random systems (not measured)
/// msolve's reduced grevlex basis of `instance`, in the canonical text; None
/// where msolve refuses the instance, as it does the zero ideal, or fails.
/// The output is cached under `name` and msolve's version, and used where
/// the cached instance is the one msolve reads.
pub fn msolve(name: &str, instance: &str) -> Option<String> {
    msolve_with(name, instance, &[])
}

// order-family.16 design -- in a two-block order the second oracle is msolve's -e k on the text msolve_spelling writes, every coefficient reduced mod p (not measured)
/// msolve's reduced basis of `instance` with `args` before its own, as
/// `msolve` returns it: `["-e", "k"]` gives the basis in two grevlex blocks,
/// the first k variables in the first, and msolve writes its terms in that
/// order. msolve 0.10.1's `-e` misreads coefficients past p (log 0011-1,
/// _The oracles in lex and in block orders_), which `msolve_spelling` reduces. The output is cached under a
/// name that carries the arguments.
pub fn msolve_with(name: &str, instance: &str, args: &[&str]) -> Option<String> {
    let spelled = msolve_spelling(instance);
    let instance = spelled.as_str();
    let dir = msolve_cache();
    let name = case_name(name, args);
    let input = dir.join(format!("{name}.ms"));
    let output = dir.join(format!("{name}.out"));
    let cached = fs::read_to_string(&input).is_ok_and(|text| text == instance);
    if !cached {
        let run = dir.join(format!(
            "{name}.run-{process}-{thread:?}.ms",
            process = std::process::id(),
            thread = std::thread::current().id()
        ));
        let out = run.with_extension("out");
        fs::write(&run, instance).unwrap();
        let status = Command::new("msolve")
            .args(args)
            .args(["-g", "2", "-t", "1", "-f"])
            .arg(&run)
            .arg("-o")
            .arg(&out)
            .output()
            .expect("msolve runs");
        // A refusal, or a failure, leaves no basis, and the cache keeps that.
        let text = if status.status.success() {
            fs::read_to_string(&out).unwrap_or_default()
        } else {
            String::new()
        };
        let _ = fs::remove_file(&run);
        let _ = fs::remove_file(&out);
        fs::write(&output, &text).unwrap();
        fs::write(&input, instance).unwrap();
    }
    let text = fs::read_to_string(&output).unwrap();
    canonical_from_msolve(&text)
}

/// The instance as msolve reads it right: like terms combined mod p, and
/// terms whose coefficient is 0 mod p and zero generators left out. msolve
/// 0.7.5 returns wrong bases where a generator is 0 (log 0004-1, step 5) and
/// where a term's coefficient is 0 mod p, as `y*z^2-1-1` or `y*z^2-2` over
/// GF(2) (log 0004-3). Every coefficient is reduced mod p: msolve 0.10.1's
/// `-e` misreads coefficients past p (log 0011-1, _The oracles in lex and
/// in block orders_). A text with a
/// fraction, a characteristic below 2, or a factor this does not read is
/// returned as it is.
fn msolve_spelling(instance: &str) -> String {
    let mut lines = instance.splitn(3, '\n');
    let (Some(names), Some(p), generators) = (lines.next(), lines.next(), lines.next()) else {
        return instance.to_string();
    };
    let names: Vec<&str> = names.split(',').map(str::trim).collect();
    let p: u128 = match p.trim().parse() {
        Ok(p) if p >= 2 => p,
        Ok(_) | Err(_) => return instance.to_string(),
    };
    let generators: String = generators
        .unwrap_or("")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if generators.contains('/') {
        return instance.to_string();
    }
    let mut polys: Vec<String> = Vec::new();
    for generator in generators.split(',').filter(|g| !g.is_empty()) {
        // The terms, each its exponents and its coefficient mod p, in the
        // order they first stand.
        let mut terms: Vec<(Vec<u64>, u128)> = Vec::new();
        let mut rest = generator;
        while !rest.is_empty() {
            let negative = rest.starts_with('-');
            rest = rest.strip_prefix(['+', '-']).unwrap_or(rest);
            let end = rest.find(['+', '-']).unwrap_or(rest.len());
            let (term, tail) = rest.split_at(end);
            rest = tail;
            let mut coef = 1u128;
            let mut exponents = vec![0u64; names.len()];
            for factor in term.split('*') {
                if factor.starts_with(|c: char| c.is_ascii_digit()) {
                    let Ok(v) = factor.parse::<u128>() else {
                        return instance.to_string();
                    };
                    coef = coef * (v % p) % p;
                } else {
                    let (name, e) = factor.split_once('^').unwrap_or((factor, "1"));
                    let (Some(i), Ok(e)) =
                        (names.iter().position(|n| *n == name), e.parse::<u64>())
                    else {
                        return instance.to_string();
                    };
                    exponents[i] += e;
                }
            }
            if negative {
                coef = (p - coef) % p;
            }
            match terms.iter_mut().find(|(m, _)| *m == exponents) {
                Some((_, c)) => *c = (*c + coef) % p,
                None => terms.push((exponents, coef)),
            }
        }
        let poly: Vec<String> = terms
            .iter()
            .filter(|(_, c)| *c != 0)
            .map(|(m, c)| {
                let mut term = c.to_string();
                for (name, &e) in names.iter().zip(m) {
                    match e {
                        0 => {}
                        1 => write!(term, "*{name}").unwrap(),
                        _ => write!(term, "*{name}^{e}").unwrap(),
                    }
                }
                term
            })
            .collect();
        if !poly.is_empty() {
            polys.push(poly.join("+"));
        }
    }
    format!(
        "{variables}\n{p}\n{generators}\n",
        variables = names.join(","),
        generators = polys.join(",\n")
    )
}

/// msolve's basis file in the canonical text: comment lines dropped, `^1`
/// and a coefficient `1*` before a monomial left out. None where the file
/// holds no bracketed basis.
pub fn canonical_from_msolve(text: &str) -> Option<String> {
    let body: String = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let body = body.trim();
    let inner = body
        .strip_prefix('[')?
        .strip_suffix(":")?
        .trim_end()
        .strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some("[]:\n".to_string());
    }
    let polys: Vec<String> = inner.split(',').map(|p| canonical_poly(p.trim())).collect();
    Some(format!("[{polys}]:\n", polys = polys.join(",\n")))
}

fn canonical_poly(poly: &str) -> String {
    let mut out = String::new();
    let mut term = String::new();
    let flush = |term: &mut String, out: &mut String| {
        if term.is_empty() {
            return;
        }
        let (sign, body) = match term.as_bytes()[0] {
            b'+' | b'-' => (&term[..1], &term[1..]),
            _ => ("", &term[..]),
        };
        let mut factors: Vec<String> = body
            .split('*')
            .map(|f| f.strip_suffix("^1").unwrap_or(f).to_string())
            .collect();
        if factors.len() > 1 && factors[0] == "1" {
            factors.remove(0);
        }
        let sign = if out.is_empty() && sign == "+" {
            ""
        } else {
            sign
        };
        out.push_str(sign);
        out.push_str(&factors.join("*"));
        term.clear();
    };
    for c in poly.chars() {
        if (c == '+' || c == '-') && !term.is_empty() && !term.ends_with('^') {
            flush(&mut term, &mut out);
        }
        if !c.is_whitespace() {
            term.push(c);
        }
    }
    flush(&mut term, &mut out);
    out
}

// --- Groebner.jl, the oracle in lex (fglm-lex.21) ---

fn groebner_jl_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let out = Command::new("groebner-jl")
            .arg("--version")
            .output()
            .unwrap_or_else(|e| {
                panic!("groebner-jl does not run ({e}), and groebner-jl.json holds no digest for the case: the test code needs one of them (AUTONOMOUS.md, _What is installed_)")
            });
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(
            out.status.success() && !text.is_empty(),
            "groebner-jl --version failed: {out:?}"
        );
        text
    })
}

fn groebner_jl_cache() -> PathBuf {
    let version: String = groebner_jl_version()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("groebner-jl")
        .join(version);
    fs::create_dir_all(&dir).expect("the Groebner.jl cache directory can be created");
    dir
}

// fglm-lex.21 design -- the test code's second oracle in lex is Groebner.jl's --order lex, run as a black box and cached, compared as sets of polynomials whose terms the test code sorts in lex, not msolve or the block:1,…,1 run (log 0012-1)
/// The reduced basis `groebner-jl <args> <instance>` prints, its first
/// bracketed list through `]:` and a line feed, as Groebner.jl writes it:
/// each polynomial's terms in grevlex, whatever the order
/// (`AUTONOMOUS.md`, _What is installed_). None where it fails. The output
/// is cached under the instance's text, its case, and Groebner.jl's
/// version.
fn groebner_jl_with(name: &str, instance: &str, args: &[&str]) -> Option<String> {
    let dir = groebner_jl_cache();
    let name = case_name(name, args);
    let input = dir.join(format!("{name}.ms"));
    let output = dir.join(format!("{name}.out"));
    let cached = fs::read_to_string(&input).is_ok_and(|text| text == instance);
    if !cached {
        let run = dir.join(format!(
            "{name}.run-{process}-{thread:?}.ms",
            process = std::process::id(),
            thread = std::thread::current().id()
        ));
        fs::write(&run, instance).unwrap();
        let out = Command::new("groebner-jl")
            .args(args)
            .arg(&run)
            .output()
            .expect("groebner-jl runs");
        // A failure leaves no basis, and the cache keeps that.
        let text = if out.status.success() {
            String::from_utf8(out.stdout).expect("groebner-jl prints UTF-8")
        } else {
            String::new()
        };
        let _ = fs::remove_file(&run);
        fs::write(&output, &text).unwrap();
        fs::write(&input, instance).unwrap();
    }
    let text = fs::read_to_string(&output).unwrap();
    let end = text.find("]:")?;
    let basis = text[..end].trim_start();
    basis.starts_with('[').then(|| format!("{basis}]:\n"))
}

/// A case of a reference program: `name` and the arguments it is run with,
/// each as `.` and the argument, any character but an ASCII letter or
/// digit as `_`.
fn case_name(name: &str, args: &[&str]) -> String {
    let mut case = name.to_string();
    for a in args {
        case.push('.');
        case.extend(
            a.chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }),
        );
    }
    case
}

/// What groebner-jl.json holds: per case, the digest of its instance under
/// `instance`, and under `basis` the digest of Groebner.jl's basis as
/// `normal` writes it.
type GroebnerJl = BTreeMap<String, BTreeMap<String, String>>;

fn groebner_jl_digests() -> &'static Mutex<GroebnerJl> {
    static DIGESTS: OnceLock<Mutex<GroebnerJl>> = OnceLock::new();
    DIGESTS.get_or_init(|| {
        let text = fs::read_to_string(GROEBNER_JL)
            .expect("groebner-jl.json is at the root of the repository");
        let digests = serde_json::from_str(&text)
            .expect("groebner-jl.json maps cases to the digests of instance and basis");
        Mutex::new(digests)
    })
}

/// The digest of `normal` of the basis `groebner_jl_with` gives, so that a
/// machine without Groebner.jl runs the test code: read from groebner-jl.json
/// where it holds the case for this instance, else computed by groebner-jl
/// and written there. None where Groebner.jl fails, which is not written.
pub fn groebner_jl_reference(
    name: &str,
    instance: &str,
    args: &[&str],
    normal: impl Fn(&str) -> String,
) -> Option<String> {
    let case = case_name(name, args);
    let input = sha256(instance);
    let held = groebner_jl_digests();
    if let Some(entry) = held.lock().unwrap().get(&case)
        && entry.get("instance") == Some(&input)
    {
        let basis = entry.get("basis");
        return Some(basis.expect("an entry holds its basis's digest").clone());
    }
    // groebner-jl runs without the lock, so cases on other threads go on.
    let basis = sha256(&normal(&groebner_jl_with(name, instance, args)?));
    let mut digests = held.lock().unwrap();
    let entry = [("instance", input), ("basis", basis.clone())];
    digests.insert(case, entry.map(|(k, v)| (k.to_string(), v)).into());
    let text = serde_json::to_string_pretty(&*digests).expect("the digests serialize");
    fs::write(GROEBNER_JL, format!("{text}\n")).expect("groebner-jl.json can be written");
    Some(basis)
}

/// Sorted polynomials as the text a digest is taken of: one per line, each
/// term its coefficient, a colon, and its exponents joined by commas.
pub fn terms_text<C: fmt::Display>(polys: &[Vec<(Vec<u64>, C)>]) -> String {
    let term = |(e, c): &(Vec<u64>, C)| {
        let e: Vec<String> = e.iter().map(u64::to_string).collect();
        format!("{c}:{exponents}", exponents = e.join(","))
    };
    let lines: Vec<String> = polys
        .iter()
        .map(|p| p.iter().map(term).collect::<Vec<_>>().join(" "))
        .collect();
    lines.join("\n") + "\n"
}

// --- the test code's sources ---

/// Every call `dense(` in the `.rs` files under the directories `dirs` of
/// `grobx/` whose second argument is the literal 2, as `<file>:<line>`:
/// the systems of the instance corpus's `mq` family (test-mq-systems.1).
/// A line's text from `//` on is a comment and left out.
pub fn dense_quadric_calls(dirs: &[&str]) -> Vec<String> {
    let mut found = Vec::new();
    for source in sources::rust_sources(dirs) {
        let code: Vec<&str> = source
            .text
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect();
        let code = code.join("\n");
        for (at, _) in code.match_indices("dense(") {
            let named = code[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
            if !named && second_argument(&code[at + "dense(".len()..]) == Some("2") {
                let line = code[..at].matches('\n').count() + 1;
                found.push((source.path.clone(), line));
            }
        }
    }
    found.sort();
    found
        .into_iter()
        .map(|(file, line)| format!("{file}:{line}"))
        .collect()
}

/// The second argument, trimmed, of a call whose argument list `rest`
/// starts: the text between its first and second comma, or its closing
/// parenthesis, at the call's own depth of brackets.
fn second_argument(rest: &str) -> Option<&str> {
    let mut depth = 0usize;
    let mut start = None;
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' if depth == 0 => return start.map(|s| rest[s..i].trim()),
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => match start {
                None => start = Some(i + 1),
                Some(s) => return Some(rest[s..i].trim()),
            },
            _ => {}
        }
    }
    None
}
