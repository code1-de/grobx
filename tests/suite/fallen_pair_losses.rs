//! fallen-pair-losses, the item's integration tests: in grevlex a call whose
//! plain run counts a fallen element's pairs one degree past their lcms'
//! only until its pair set has grown past its bound returns the basis of
//! the call with the signatures on, canonical, and over GF(p) the basis the
//! exact checker accepts with cofactors; in lex and in `block:1` the exact
//! checker accepts the call's basis
//! (`iterations/0111-fallen-pair-losses/plan.md`, _Test code_, 1 and 2).

use grobx_checker::Mode;

use crate::common::{EDGE_GENERATORS, FALLING_SEEDS, P, params_in, signatures_in};
use crate::cooper_rest_gap::{PRIMES, falling};
use crate::orders::accepted_in;
use crate::q::assert_canonical_q;
use crate::redundant_round_pairs::same;

/// The sizes of the falling systems that pass the bound: variables,
/// generators, and the top degree, three generators past the variables.
/// Over each prime and over ℚ their plain runs pass the bound of 128 live
/// pairs, and most do so with deferred pairs pending and enter fallen
/// elements after it at updates that begin with fewer (log 0111-1,
/// _Step 5_). Their ideals are the unit ideal; less their values at
/// (1, …, 1), the ideal of a common root.
const SIZES: [(usize, usize, u32); 2] = [(5, 8, 4), (6, 9, 3)];

/// The size of the falling systems whose own bases are given back as
/// inputs: their runs stay below the bound, and their bases pass it as
/// inputs.
const BASIS_SIZE: (usize, usize, u32) = (5, 5, 3);

/// The edge cases of `EDGE_GENERATORS` that make no pair: the zero ideal, F = (0, 0),
/// and no generator.
const PAIRLESS: usize = 3;

/// The reduced basis text of the unit ideal.
const UNIT: &str = "[1]:\n";

/// `instance` with `generator` after its generators.
fn with_generator(instance: &str, generator: &str) -> String {
    format!("{body},\n{generator}\n", body = instance.trim_end())
}

/// `instance` with the variable w after its variables and the generator
/// x·y − w, which w alone eliminates.
fn with_eliminable(instance: &str) -> String {
    let (variables, rest) = instance.split_once('\n').unwrap();
    with_generator(&format!("{variables},w\n{rest}"), "x*y-w")
}

/// The falling system `instance` over GF(p), or ℚ where p is 0, each
/// generator less its value at (1, …, 1), the sum of its coefficients:
/// every term of `falling` carries its coefficient first.
fn vanishing_at_one(instance: &str, p: u64) -> String {
    let mut lines = instance.lines();
    let (variables, field) = (lines.next().unwrap(), lines.next().unwrap());
    let body = lines.collect::<Vec<_>>().join("\n");
    let generators: Vec<String> = body
        .split(",\n")
        .map(|g| {
            let sum: u64 = g
                .split('+')
                .map(|t| t.split('*').next().unwrap().parse::<u64>().unwrap())
                .sum();
            let value = if p == 0 { sum } else { sum % p };
            format!("{g}-{value}")
        })
        .collect();
    format!(
        "{variables}\n{field}\n{generators}\n",
        generators = generators.join(",\n")
    )
}

/// The basis text of `instance` in grevlex.
fn basis_of(case: &str, instance: &str) -> String {
    grobx::groebner(instance, &params_in("grevlex", 0xfa11))
        .unwrap_or_else(|e| panic!("{case}: {e:?}\n{instance}"))
        .basis
        .unwrap_or_else(|| panic!("{case}: no basis\n{instance}"))
}

/// The reduced basis of `instance` in grevlex, given back as an instance
/// over its variables and field: an input already a Gröbner basis.
fn its_basis(case: &str, instance: &str) -> String {
    let basis = basis_of(case, instance);
    let mut lines = instance.lines();
    let (variables, field) = (lines.next().unwrap(), lines.next().unwrap());
    let polynomials = basis
        .strip_prefix('[')
        .and_then(|b| b.strip_suffix("]:\n"))
        .unwrap_or_else(|| panic!("{case}: basis text {basis:?}"));
    format!("{variables}\n{field}\n{polynomials}\n")
}

/// Over ℚ in grevlex: the call with the signatures off returns the basis
/// of the call with them on, canonical. The calls with cofactors of these
/// systems take minutes over ℚ.
fn same_q(case: &str, instance: &str) -> String {
    let basis = |signatures: bool| {
        grobx::groebner(instance, &signatures_in("grevlex", signatures, 0x5e1f))
            .unwrap_or_else(|e| panic!("{case}, signatures {signatures}: {e:?}\n{instance}"))
            .basis
            .unwrap_or_else(|| panic!("{case}, signatures {signatures}: no basis\n{instance}"))
    };
    let off = basis(false);
    assert_eq!(off, basis(true), "{case}: signatures on\n{instance}");
    assert_canonical_q(case, instance, &off, "grevlex");
    off
}

/// The name of the falling system of size (n, k, top) and seed over p.
fn named(n: usize, k: usize, top: u32, seed: u64, p: u64) -> String {
    let field = if p == 0 {
        "ℚ".to_string()
    } else {
        p.to_string()
    };
    format!("{k} falling generators on {n} variables, degree {top}, seed {seed}, over {field}")
}

/// Test 1: the edge cases that make no pair over the four primes; every
/// edge case of `EDGE_GENERATORS` in lex and `block:1` over 65521; and in the falling
/// systems of `SIZES` over the four primes and over ℚ, the unit ideal, the
/// zero polynomial among the generators, and an eliminable variable; the
/// bases of the falling systems of `BASIS_SIZE` given back as inputs over
/// the four primes.
#[test]
fn the_edge_cases() {
    for (case, generators) in &EDGE_GENERATORS[..PAIRLESS] {
        for p in PRIMES {
            same(
                &format!("{case} over {p}"),
                &format!("x,y,z\n{p}\n{generators}\n"),
            );
        }
    }
    for (case, generators) in EDGE_GENERATORS {
        for order in ["lex", "block:1"] {
            accepted_in(
                &format!("{case} over {P}"),
                &format!("x,y,z\n{P}\n{generators}\n"),
                order,
                Mode::Exact,
            );
        }
    }
    for seed in 0..FALLING_SEEDS {
        for (n, k, top) in SIZES {
            for p in PRIMES {
                let name = named(n, k, top, seed, p);
                let instance = falling(p, n, k, top, seed);
                let case = format!("{name}, the unit ideal");
                same(&case, &instance);
                assert_eq!(basis_of(&case, &instance), UNIT, "{case}\n{instance}");
                same(
                    &format!("{name}, the zero polynomial among them"),
                    &with_generator(&instance, "0"),
                );
                same(
                    &format!("{name}, an eliminable variable"),
                    &with_eliminable(&instance),
                );
            }
            let name = named(n, k, top, seed, 0);
            let instance = falling(0, n, k, top, seed);
            let case = format!("{name}, the unit ideal");
            assert_eq!(same_q(&case, &instance), UNIT, "{case}\n{instance}");
            same_q(
                &format!("{name}, the zero polynomial among them"),
                &with_generator(&instance, "0"),
            );
            same_q(
                &format!("{name}, an eliminable variable"),
                &with_eliminable(&instance),
            );
        }
        let (n, k, top) = BASIS_SIZE;
        for p in PRIMES {
            let case = format!(
                "{name}, already a Gröbner basis",
                name = named(n, k, top, seed, p)
            );
            same(&case, &its_basis(&case, &falling(p, n, k, top, seed)));
        }
    }
}

/// Test 2: the falling systems of `SIZES` less their values at (1, …, 1)
/// over the four primes and over ℚ, seeds 0 to `FALLING_SEEDS` − 1.
#[test]
fn falling_systems_with_a_common_root() {
    for seed in 0..FALLING_SEEDS {
        for (n, k, top) in SIZES {
            for p in PRIMES {
                same(
                    &format!("{name}, a common root", name = named(n, k, top, seed, p)),
                    &vanishing_at_one(&falling(p, n, k, top, seed), p),
                );
            }
            same_q(
                &format!("{name}, a common root", name = named(n, k, top, seed, 0)),
                &vanishing_at_one(&falling(0, n, k, top, seed), 0),
            );
        }
    }
}
