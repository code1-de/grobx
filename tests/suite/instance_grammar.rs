//! instance-grammar-agreement, the item's test code: the engine and the
//! checker read the same instance texts. What one reads, the other reads to
//! the same generators, and what one refuses, the other refuses, with an
//! error of the same kind (plan.md, _Test code_, 3 to 6).

use std::time::Duration;

use grobx::CheckerMode;
use grobx_checker::{Mode, Order, Part};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::common::{params, with_cofactors};

// instance-grammar-agreement.8 design -- two refusals are compared by their kind, not by their line (not measured)
/// What a reader made of a text: its generators' text, the text read
/// whole where no hook writes the generators, or a refusal of one kind.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Reading {
    Generators(String),
    Whole,
    Refused(&'static str),
}

fn engine_kind(e: &grobx::Error) -> &'static str {
    match e {
        grobx::Error::Syntax { .. } => "syntax",
        grobx::Error::UnknownVariable { .. } => "unknown variable",
        grobx::Error::RepeatedVariable { .. } => "repeated variable",
        grobx::Error::Characteristic { .. } => "characteristic",
        grobx::Error::Overflow { .. } => "overflow",
        grobx::Error::NoInverse { .. } => "no inverse",
        _ => "another error",
    }
}

fn checker_kind(e: &grobx_checker::Error) -> &'static str {
    match e {
        grobx_checker::Error::Syntax { .. } => "syntax",
        grobx_checker::Error::UnknownVariable { .. } => "unknown variable",
        grobx_checker::Error::RepeatedVariable { .. } => "repeated variable",
        grobx_checker::Error::Characteristic { .. } => "characteristic",
        grobx_checker::Error::Overflow { .. } => "overflow",
        grobx_checker::Error::NoInverse { .. } => "no inverse",
        _ => "another error",
    }
}

/// Whether the checker's error is the instance's.
fn of_the_instance(e: &grobx_checker::Error) -> bool {
    match e {
        grobx_checker::Error::Syntax { part, .. }
        | grobx_checker::Error::UnknownVariable { part, .. }
        | grobx_checker::Error::Overflow { part, .. }
        | grobx_checker::Error::NoInverse { part, .. } => match part {
            Part::Instance => true,
            Part::Basis | Part::Cofactors => false,
        },
        grobx_checker::Error::RepeatedVariable { .. }
        | grobx_checker::Error::Characteristic { .. } => true,
        _ => false,
    }
}

/// The engine's generators of a text over GF(p): `grobx::text::generators`.
fn engine_over_p(text: &str) -> Reading {
    match grobx::text::generators(text) {
        Ok(t) => Reading::Generators(t),
        Err(e) => Reading::Refused(engine_kind(&e)),
    }
}

/// The checker's generators of a text over GF(p):
/// `grobx_checker::generators`, the counterpart of the engine's
/// (instance-grammar-agreement.7).
fn checker_over_p(text: &str) -> Reading {
    match grobx_checker::generators(text) {
        Ok(t) => Reading::Generators(t),
        Err(e) => Reading::Refused(checker_kind(&e)),
    }
}

/// The engine's reading of a text over Q, by a call under `deadline`: a
/// basis, a stop at the deadline, and an overflow during the run all read
/// the text.
fn engine_over_q(text: &str, deadline: Duration) -> Reading {
    let mut p = params(0x9a3);
    p.deadline = Some(deadline);
    match grobx::groebner(text, &p) {
        Ok(_) | Err(grobx::Error::Overflow { line: None }) => Reading::Whole,
        Err(e) => Reading::Refused(engine_kind(&e)),
    }
}

/// The checker's reading of a text over Q, by a check with no element and
/// no row: an error of the instance refuses it.
fn checker_over_q(text: &str) -> Reading {
    match grobx_checker::check(text, "[]:\n", "[]:\n", &Order::Grevlex, Mode::Exact) {
        Err(e) if of_the_instance(&e) => Reading::Refused(checker_kind(&e)),
        Ok(_) | Err(_) => Reading::Whole,
    }
}

/// The texts of some of the plan's edge cases over a prime.
fn cases_over_p() -> Vec<String> {
    let mut texts: Vec<String> = [
        // OBJECTIVES.md's edge cases.
        "x,y\n7\n",
        "x,y\n7",
        "x,y\n7\n\n  \n",
        "x,y\n7\n0\n",
        "x,y\n7\nx*y-1,\nx\n",
        "x,y\n7\nx+y,\ny^2+1\n",
        "x,y,z\n7\nx*y-z,\n0,\ny^2-x\n",
        "x,y,z\n7\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        // Whitespace.
        "x,y\r\n7\r\nx^2+y,\r\nx*y-1\r\n",
        "x,y\n7\nx^2+y,\x0cx*y-1\n",
        "\x0cx,\x0cy\x0c\n\x0c7\x0c\n\x0cx\x0c^\x0c2\x0c+\x0c1\x0c/\x0c2\x0c*\x0cy\x0c\n",
        "\tx ,\ty \r\n 7\t\n\tx\t*\ty\n",
        "x,y\n7\nx\x0b+y\n",
        "x,y\n7\nx\0+y\n",
        "x,y\n7\nx\u{a0}+y\n",
        "x,y\n7\nx\u{2003}+y\n",
        "x,\x0by\n7\nx\n",
        // Line 1.
        "x_1,_y2,Z9\n7\nx_1*_y2-Z9^3\n",
        "x,x\n7\nx\n",
        "1x,y\n7\ny\n",
        "x y\n7\nx\n",
        "x,y",
        "\n7\nx\n",
        "x,\n7\nx\n",
        // Line 2.
        "x\n007\nx^2+1\n",
        "x\n\n7\nx\n",
        "x\n+7\nx\n",
        "x\n7 x\nx\n",
        "x\n7a\nx\n",
        // Signs.
        "x\n7\n-x\n",
        "x\n7\n+x\n",
        "x\n7\n--x\n",
        "x\n7\nx+-1\n",
        "x\n7\nx-+1\n",
        "x\n7\n-\n",
        // Numbers and fractions.
        "x,y\n7\n2*x*3*x^2-1/2*y*4\n",
        "x\n7\n0/5*x+1\n",
        "x\n7\nx+1/0\n",
        "x\n7\nx+1/7\n",
        "x\n7\nx+1/14\n",
        "x\n7\nx/2\n",
        "x\n7\n1/2/3*x\n",
        "x\n7\n1/ 3*x\n",
        "x\n7\n1/\n",
        "x\n7\n123456789012345678901234567890*x+1\n",
        "x\n2\nx+1+1\n",
        // Variables and exponents.
        "x,y\n7\nx*x^2*y*y\n",
        "x,y\n7\nx*z\n",
        "x,y\n7\n2x\n",
        "x,y\n7\nx y\n",
        "x\n7\nx^ 2\n",
        "x\n7\nx ^2\n",
        "x\n7\nx^+2\n",
        "x\n7\nx^-1\n",
        "x\n7\nx^\n",
        "x\n7\nx^0\n",
        "x\n7\nx^007\n",
        // Commas and what follows the generators.
        "x,y\n7\nx,,y\n",
        "x,y\n7\nx,\n",
        "x,y\n7\n,x\n",
        "x,y\n7\nx;\n",
        "x,y\n7\nx\n)\n",
        // The degree.
        "x\n7\nx^2147483647\n",
        "x\n7\nx^2147483648\n",
        "x,y\n7\nx^1073741824*y^1073741824\n",
        "x\n7\nx^4294967295\n",
        "x\n7\nx^4294967296\n",
        "x\n7\nx^99999999999999999999\n",
        "x\n7\n0*x^2147483648\n",
        "x,y\n7\nx^99999999999*z\n",
        "x,y\n7\nx^99999999999*\n",
        "x,y\n7\nx^99999999999*1/7\n",
    ]
    .iter()
    .map(std::string::ToString::to_string)
    .collect();
    // The characteristic.
    for p in [
        "1",
        "2",
        "3",
        "4",
        "9",
        "65521",
        "65535",
        "561",
        "2047",
        "2147483647",
        "2305843009213693951",
        "3825123056546413051",
        "18446744030759878681",
        "18446744073709551557",
        "18446744073709551615",
        "18446744073709551616",
        "1000000000000000000000000000000",
    ] {
        texts.push(format!("x,y\n{p}\nx^2+y,\nx*y-1\n"));
        texts.push(format!("x,y\n{p}\nx^2+,\n"));
    }
    texts
}

/// Test 3: over a prime, the two readers give the same generators or the
/// same kind of refusal.
#[test]
fn the_grammars_cases_over_a_prime() {
    let failures: Vec<String> = cases_over_p()
        .iter()
        .filter_map(|text| {
            let (e, c) = (engine_over_p(text), checker_over_p(text));
            (e != c).then(|| format!("{text:?}: engine {e:?}, checker {c:?}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

/// The texts of some of the plan's edge cases over Q.
fn cases_over_q() -> Vec<&'static str> {
    vec![
        "x,y\n0\n",
        "x,y\n0\n0\n",
        "x,y\n0\nx*y-1,\nx\n",
        "x,y\n0\nx+y,\ny^2+1\n",
        "x,y,z\n0\nx*y-z,\n0,\ny^2-x\n",
        "x,y,z\n0\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        "x,y\n00\n1/2*x^2-3/4*y,\nx*y+5\n",
        "x,y\n0\n123456789012345678901234567890/7*x-y^3,\nx*y-1/3\n",
        "x,y\n0\nx^2+y,\x0cx*y-1\n",
        "x,y\n0\nx^2+y,\x0bx*y-1\n",
        "x\n0\nx+1/0\n",
        "x\n0\nx+1/000\n",
        "x\n0\nx+1/\n",
        "x\n0\nx^2147483647\n",
        "x\n0\nx^2147483648\n",
        "x,y\n0\nx^99999999999*z\n",
        "x,y\n0\nx*z\n",
        "x,x\n0\nx\n",
    ]
}

/// Test 4: over Q, the call and the check read the same texts and refuse
/// the others with one kind; every text both read whole, called with
/// cofactors and the exact checker, returns a basis the checker certifies.
#[test]
fn the_grammars_cases_over_q() {
    let mut failures = Vec::new();
    for text in cases_over_q() {
        let (e, c) = (
            engine_over_q(text, Duration::from_secs(1)),
            checker_over_q(text),
        );
        if e != c {
            failures.push(format!("{text:?}: engine {e:?}, checker {c:?}"));
            continue;
        }
        match e {
            Reading::Whole => {
                let mut p = with_cofactors(0x9a3);
                p.checker = Some(CheckerMode::Exact);
                let outcome = grobx::groebner(text, &p).unwrap();
                match outcome.certified {
                    Some(true) => {}
                    Some(false) | None => {
                        failures.push(format!("{text:?}: not certified, {outcome:?}"));
                    }
                }
            }
            Reading::Generators(_) | Reading::Refused(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

// instance-grammar-agreement.9 practice: mutation-based fuzzing -- the random texts are seed instances mutated by insertions, deletions, and replacements from a pool, not texts a generator of the grammar makes
/// The seed instances of the random texts.
const SEEDS: [&str; 9] = [
    "x,y\n7\nx^2+y,\nx*y-1\n",
    "x,y,z\n65521\nx^2 + y*z - 1,\n2*x*y - 1/3*z,\nz^3 - x\n",
    "a_1,b2\n0\n1/2*a_1^2-3*b2,\na_1*b2+5\n",
    "x\n2\nx^3+x+1\n",
    "x,y\n0\n-x+2/4*y^2,\n+y\n",
    "x,y\n0\n123456789012345678901234567890/7*x-y^3,\n x*y*x - 1/0000003\n",
    " x , y \r\n 00007 \r\n x ^ 2 + y , \r\n x * y - 1 \r\n",
    "x\n18446744073709551557\nx^2-2\n",
    "x,y\n7\n0,\n0*x+y,\nx*0*y\n",
];

/// What a mutation inserts or writes over a character.
const POOL: [&str; 38] = [
    " ",
    "\t",
    "\r",
    "\n",
    "\x0b",
    "\x0c",
    "\0",
    "+",
    "-",
    "*",
    "/",
    "^",
    ",",
    "0",
    "1",
    "7",
    "9",
    "x",
    "y",
    "z",
    "_",
    "\u{a0}",
    "\u{2003}",
    "#",
    ";",
    ".",
    "(",
    ")",
    "2147483647",
    "2147483648",
    "4294967295",
    "4294967296",
    "18446744073709551615",
    "18446744073709551616",
    "18446744073709551557",
    "9",
    "4",
    "1",
];

/// A seed instance with one to five edits, each a string of `POOL`
/// inserted, a character deleted, or a character written over by a string
/// of `POOL`; a deletion or an overwrite drawn past the last character does
/// nothing.
fn mutated(rng: &mut Xoshiro256PlusPlus) -> String {
    let below = |rng: &mut Xoshiro256PlusPlus, n: usize| (rng.next_u64() % n as u64) as usize;
    let seed = SEEDS[below(rng, SEEDS.len())];
    let mut chars: Vec<String> = seed.chars().map(|c| c.to_string()).collect();
    for _ in 0..=below(rng, 5) {
        let i = below(rng, chars.len() + 1);
        match below(rng, 3) {
            0 => chars.insert(i, POOL[below(rng, POOL.len())].to_string()),
            1 if i < chars.len() => {
                chars.remove(i);
            }
            _ if i < chars.len() => chars[i] = POOL[below(rng, POOL.len())].to_string(),
            _ => {}
        }
    }
    chars.concat()
}

/// Whether the text's second line is the characteristic 0: digits that are
/// all `0`, between ASCII whitespace.
fn over_q(text: &str) -> bool {
    let second = text.split('\n').nth(1).unwrap_or("");
    let digits = second.trim_matches(|c: char| c.is_ascii_whitespace());
    !digits.is_empty() && digits.bytes().all(|b| b == b'0')
}

/// Test 5: random texts under the seeds 1, 2, and 3. Every text is read
/// by both hooks; one over Q is read by a call under a deadline of 100 ms
/// and by a check as well.
#[test]
fn random_texts() {
    let mut failures = Vec::new();
    for seed in [1, 2, 3] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        for _ in 0..5_000 {
            let text = mutated(&mut rng);
            let (e, c) = (engine_over_p(&text), checker_over_p(&text));
            if e != c {
                failures.push(format!(
                    "seed {seed}, {text:?}: engine {e:?}, checker {c:?}"
                ));
            }
            if over_q(&text) {
                let (e, c) = (
                    engine_over_q(&text, Duration::from_millis(100)),
                    checker_over_q(&text),
                );
                if e != c {
                    failures.push(format!(
                        "seed {seed}, {text:?} over Q: engine {e:?}, checker {c:?}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{failures}",
        failures = failures.join("\n")
    );
}

/// Test 6: a form feed between two generators. The call returns the basis
/// and the exact checker certifies it.
#[test]
fn a_form_feed_through_the_crate() {
    let mut p = with_cofactors(0x9a3);
    p.checker = Some(CheckerMode::Exact);
    let outcome = grobx::groebner("x,y\n7\nx^2+y,\x0cx*y-1\n", &p).unwrap();
    assert_eq!(outcome.basis.as_deref(), Some("[y^2+x,\nx*y+6,\nx^2+y]:\n"));
    assert_eq!(outcome.certified, Some(true), "{outcome:?}");
}
