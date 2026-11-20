//! cooper-rest-gap, the item's integration tests: in grevlex a call whose
//! plain run counts a fallen element's pairs one degree past their lcms'
//! while the pair set is small returns the basis of the call with the
//! signatures on, canonical, and the basis the exact checker accepts with
//! cofactors; in lex, in `block:1`, and over ℚ the exact checker accepts
//! the call's basis (`iterations/0110-cooper-rest-gap/plan.md`, _Test
//! code_, 1 to 3).

use grobx_checker::Mode;

use crate::common::systems::{bound, term};
use crate::common::{EDGE_GENERATORS, FALLING_SEEDS, P, P31, Rng, random_square, random_system};
use crate::large::LEAST;
use crate::orders::accepted_in;
use crate::q::accepted_q;
use crate::redundant_round_pairs::same;

/// The primes of the tests: one below the combinations' bound, one that
/// delays its subtractions, the largest below 2^31, and one past it.
pub(crate) const PRIMES: [u64; 4] = [7, P, P31, LEAST];

/// The variables of a falling system, which takes the first n of them.
const NAMES: [&str; 7] = ["x", "y", "z", "u", "v", "s", "t"];

/// The sizes of the falling systems over GF(p): variables, generators,
/// and the top degree. Their pair sets stay below the deferral's bound of
/// 128 live pairs, and six generators of degree 3 on five variables pass
/// it.
const SIZES: [(usize, usize, u32); 5] = [(3, 3, 2), (3, 3, 3), (4, 4, 3), (5, 5, 3), (5, 6, 3)];

/// The sizes of the falling systems over ℚ, whose calls with cofactors
/// each take milliseconds.
const SIZES_Q: [(usize, usize, u32); 4] = [(3, 3, 2), (3, 3, 3), (4, 4, 2), (5, 5, 2)];

/// Exponents on `n` variables of total degree `degree`, each unit of it
/// given to a variable drawn from `rng`.
fn spread(rng: &mut Rng, n: usize, degree: u32) -> Vec<u32> {
    let mut e = vec![0; n];
    for _ in 0..degree {
        e[rng.below(n as u64) as usize] += 1;
    }
    e
}

/// An inhomogeneous system over GF(p), or ℚ where p is 0, on the first
/// `n` of `NAMES`: `k` generators, each two or three terms of the degree
/// `top` and three to five terms of lower degrees, every coefficient in
/// [1, `bound(p)`), drawn from `seed`. Its reductions bring elements whose
/// leading monomials lie below the degree of the round that made them.
pub(crate) fn falling(p: u64, n: usize, k: usize, top: u32, seed: u64) -> String {
    let mut rng = Rng::new(seed ^ p.rotate_left(17) ^ 0xc00e);
    let names = &NAMES[..n];
    let generators: Vec<String> = (0..k)
        .map(|_| {
            let tops = 2 + rng.below(2);
            let lows = 3 + rng.below(3);
            let terms: Vec<String> = (0..tops + lows)
                .map(|t| {
                    let degree = if t < tops {
                        top
                    } else {
                        rng.below(u64::from(top)) as u32
                    };
                    let e = spread(&mut rng, n, degree);
                    term(1 + rng.below(bound(p) - 1), names, &e)
                })
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

/// Test 1: the edge cases of `OBJECTIVES.md` over the four primes in
/// grevlex, and in lex and `block:1` with cofactors; over ℚ in grevlex.
#[test]
fn the_edge_cases() {
    for (case, generators) in EDGE_GENERATORS {
        for p in PRIMES {
            let instance = format!("x,y,z\n{p}\n{generators}\n");
            let name = format!("{case} over {p}");
            same(&name, &instance);
            for order in ["lex", "block:1"] {
                accepted_in(&name, &instance, order, Mode::Exact);
            }
        }
        accepted_q(
            &format!("{case} over ℚ"),
            &format!("x,y,z\n0\n{generators}\n"),
            "grevlex",
        );
    }
}

/// Test 2: random systems, seeds 0 to 23, and random square systems,
/// seeds 0 to 11, over the four primes in grevlex; random systems over ℚ.
#[test]
fn random_systems() {
    for seed in 0..24 {
        for p in PRIMES {
            same(
                &format!("random system {seed} over {p}"),
                &random_system(p, seed),
            );
            if seed < 12 {
                same(
                    &format!("random square {seed} over {p}"),
                    &random_square(p, seed),
                );
            }
        }
        accepted_q(
            &format!("random system {seed} over ℚ"),
            &random_system(0, seed),
            "grevlex",
        );
    }
}

/// Test 3: inhomogeneous systems of every size of `SIZES` over the four
/// primes in grevlex, and of `SIZES_Q` over ℚ, seeds 0 to `FALLING_SEEDS` − 1.
#[test]
fn falling_systems() {
    for seed in 0..FALLING_SEEDS {
        for (n, k, top) in SIZES {
            for p in PRIMES {
                same(
                    &format!(
                        "{k} falling generators on {n} variables, degree {top}, seed {seed}, over {p}"
                    ),
                    &falling(p, n, k, top, seed),
                );
            }
        }
        for (n, k, top) in SIZES_Q {
            accepted_q(
                &format!(
                    "{k} falling generators on {n} variables, degree {top}, seed {seed}, over ℚ"
                ),
                &falling(0, n, k, top, seed),
                "grevlex",
            );
        }
    }
}
