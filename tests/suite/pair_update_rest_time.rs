//! pair-update-rest-time, the item's integration tests: with the update's
//! lcms interned at selection, its arrays, and its loops per word count, a
//! call returns the basis of the call with the signatures off, canonical,
//! which the exact checker accepts with cofactors, on the edge cases,
//! random systems, and systems in many variables, over GF(p) and ℚ; and
//! every round line of the report names a `select` and an `update` time
//! (`iterations/0107-pair-update-rest-time/plan.md`, _Test code_, 8).

use grobx::Params;

use crate::common::{EDGE_GENERATORS, GF31_PRIMES, P, params_in, random_square, random_system};
use crate::q::accepted_q;
use crate::redundant_round_pairs::same;

/// The copies of `copies`, each on four variables: 4, 12, 36, 40, and 72
/// variables, one, two, five, five, and nine words in `Dense8`.
const COPIES: [usize; 5] = [1, 3, 9, 10, 18];

/// `k` random square systems, copy i of seed `seed` + i on the variables
/// x(4i+1) to x(4i+4), in one instance on 4k variables over `p`: every
/// pair between two copies is coprime, and `G_min` holds every copy's
/// elements. The plan's random sparse systems on 33 to 70 variables, n
/// generators in n variables, do not compute in milliseconds (log 0107-3,
/// _Step 6_).
fn copies(k: usize, p: u64, seed: u64) -> String {
    let names: Vec<String> = (1..=4 * k).map(|i| format!("x{i}")).collect();
    let mut gens: Vec<String> = Vec::new();
    for i in 0..k {
        let text = random_square(p, seed + i as u64);
        let mut lines = text.lines();
        let own: Vec<char> = lines
            .next()
            .expect("a variable line")
            .split(',')
            .map(|v| v.chars().next().expect("a name"))
            .collect();
        lines.next().expect("the characteristic");
        for g in lines {
            let mut out = String::new();
            for c in g.trim_end_matches(',').chars() {
                match own.iter().position(|&v| v == c) {
                    Some(at) => out.push_str(&names[4 * i + at]),
                    None => out.push(c),
                }
            }
            gens.push(out);
        }
    }
    let (names, gens) = (names.join(","), gens.join(",\n"));
    format!("{names}\n{p}\n{gens}\n")
}

/// A chain of binomials in n variables, `x_i·x_{i+1} − x_{i+2}` for i
/// below n − 2, and `x_{n−1}² − 1`: many pairs of one lcm, and members
/// that leave `G_min`.
fn chain(n: usize, p: u64) -> String {
    let names: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    let mut gens: Vec<String> = (0..n - 2)
        .map(|i| {
            let (a, b, c) = (&names[i], &names[i + 1], &names[i + 2]);
            format!("{a}*{b}-{c}")
        })
        .collect();
    let last = &names[n - 1];
    gens.push(format!("{last}^2-1"));
    let (names, gens) = (names.join(","), gens.join(",\n"));
    format!("{names}\n{p}\n{gens}\n")
}

/// The edge cases of `OBJECTIVES.md` and random systems over each prime,
/// and over ℚ.
#[test]
fn the_edge_cases_and_random_systems() {
    for (case, gens) in EDGE_GENERATORS {
        for p in GF31_PRIMES {
            same(
                &format!("{case} over {p}"),
                &format!("x,y,z\n{p}\n{gens}\n"),
            );
        }
        accepted_q(case, &format!("x,y,z\n0\n{gens}\n"), "grevlex");
    }
    for seed in 1..=6 {
        for p in GF31_PRIMES {
            same(
                &format!("random system {seed} over {p}"),
                &random_system(p, seed),
            );
            same(
                &format!("random square {seed} over {p}"),
                &random_square(p, seed),
            );
        }
        accepted_q(
            &format!("random system {seed}"),
            &random_system(0, seed),
            "grevlex",
        );
    }
}

/// Copies of random square systems and chains in many variables over
/// GF(65521) and ℚ.
#[test]
fn many_variables() {
    for k in COPIES {
        let n = 4 * k;
        for seed in 1..=2 {
            same(
                &format!("{k} copies on {n} variables, seed {seed}"),
                &copies(k, P, seed),
            );
        }
        accepted_q(
            &format!("{k} copies on {n} variables over Q"),
            &copies(k, 0, 1),
            "grevlex",
        );
    }
    for n in [40, 70] {
        same(&format!("chain of {n}"), &chain(n, P));
        accepted_q(&format!("chain of {n} over Q"), &chain(n, 0), "grevlex");
    }
}

/// The report's round lines name a `select` and an `update` time, the
/// stages the interning moved between, on ten copies on 40 variables and
/// a random square system.
#[test]
fn the_round_lines_name_the_stages() {
    // Seed 4: the square system of seed 3, in x and y, holds a linear
    // generator, whose substitution (ccode-newton-gap) leaves its run no
    // round.
    for text in [copies(10, P, 3), random_square(P, 4)] {
        let mut params: Params = params_in("grevlex", 0xa107);
        params.report = true;
        params.signatures = false;
        let outcome = grobx::groebner(&text, &params).unwrap();
        let report = outcome.report.unwrap();
        let rounds: Vec<&str> = report.lines().filter(|l| l.contains(" round ")).collect();
        assert!(!rounds.is_empty(), "no round line\n{text}");
        for line in rounds {
            assert!(
                line.contains(" select=") && line.contains(" update="),
                "a round line without its stages: {line}"
            );
        }
    }
}
