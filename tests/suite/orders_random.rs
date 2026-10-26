//! order-family, the item's test code: random systems over every prime of
//! `PRIMES` in lex, in two-block orders, and in three-block orders, each
//! computed with cofactors and a deadline of 1 s (order-family.19). Every
//! result is canonical in its order and accepted by the checker in the
//! exact mode. Each two-block basis is msolve's `-e k`, and each lex basis
//! is the `block:1,…,1` basis where that call finishes within a deadline of
//! 200 ms (plan.md, _Test code_; fglm-lex.24).

use std::time::Duration;

use grobx_checker::Mode;

use crate::common::{PRIMES, Rng, msolve_with, params_in, random_system};
use crate::orders::{finished_in, lex_as_blocks, names};

const SEEDS: u64 = 2;
const DEADLINE: Duration = Duration::from_secs(1);
/// The deadline of the `block:1,…,1` call.
const BLOCKS_DEADLINE: Duration = Duration::from_millis(200);

#[test]
fn lex_on_random_systems() {
    let mut finished = 0;
    let mut cases = 0;
    for p in PRIMES {
        for seed in 0..SEEDS {
            cases += 1;
            let text = random_system(p, seed);
            let name = format!("orders-random-p{p}-{seed}");
            let Some(outcome) = finished_in(&name, &text, "lex", Mode::Exact, DEADLINE) else {
                continue;
            };
            let lex = outcome.basis.unwrap();
            let blocks = lex_as_blocks(names(&text).len());
            // fglm-lex.24: a lex call on a zero-dimensional ideal finishes
            // where F4 in a block order may not, so the block call takes
            // a deadline too, and the comparison is made where it finishes.
            let mut p = params_in(&blocks, 1);
            p.deadline = Some(BLOCKS_DEADLINE);
            let same = grobx::groebner(&text, &p)
                .unwrap_or_else(|e| panic!("{name} in {blocks}: {e:?}\n{text}"))
                .basis;
            if let Some(same) = same {
                assert_eq!(lex, same, "{name}: lex and {blocks}\n{text}");
            }
            finished += 1;
        }
    }
    assert!(
        10 * finished >= 9 * cases,
        "only {finished} of {cases} lex runs finished"
    );
}

#[test]
#[ignore = "calls msolve"]
fn two_blocks_on_random_systems() {
    let mut compared = 0;
    let mut cases = 0;
    for p in PRIMES {
        for seed in 0..SEEDS {
            cases += 1;
            let text = random_system(p, seed);
            let n = names(&text).len() as u64;
            let k = 1 + Rng::new(seed ^ p ^ 0xb10c).below(n - 1);
            let order = format!("block:{k}");
            let name = format!("orders-random-p{p}-{seed}");
            let Some(outcome) = finished_in(&name, &text, &order, Mode::Exact, DEADLINE) else {
                continue;
            };
            let ours = outcome.basis.unwrap();
            let e = k.to_string();
            if let Some(theirs) = msolve_with(&name, &text, &["-e", &e]) {
                assert_eq!(
                    ours, theirs,
                    "{name} in {order} against msolve -e {k}\n{text}"
                );
                compared += 1;
            }
        }
    }
    assert!(
        10 * compared >= 9 * cases,
        "only {compared} of {cases} systems compared with msolve"
    );
}

/// Three blocks on three and four variables: sizes drawn so that each block
/// holds a variable. On three variables that is lex.
#[test]
fn three_blocks_on_random_systems() {
    let mut finished = 0;
    let mut cases = 0;
    for p in PRIMES {
        for seed in 0..2 * SEEDS {
            let text = random_system(p, seed);
            let n = names(&text).len() as u64;
            if n < 3 {
                continue;
            }
            cases += 1;
            let mut rng = Rng::new(seed ^ p ^ 0x3b10c);
            let first = 1 + rng.below(n - 2);
            let second = 1 + rng.below(n - 1 - first);
            let order = format!("block:{first},{second}");
            let name = format!("orders-random-p{p}-{seed}");
            if finished_in(&name, &text, &order, Mode::Exact, DEADLINE).is_some() {
                finished += 1;
            }
        }
    }
    assert!(
        10 * finished >= 9 * cases,
        "only {finished} of {cases} three-block runs finished"
    );
}
