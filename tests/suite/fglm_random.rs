//! fglm-lex, the item's test code: random square systems and random systems
//! over every prime of `PRIMES` in lex, each computed with cofactors and a
//! deadline of 1 s, canonical in lex, and accepted by the checker in the
//! exact mode. Each is the `block:1,…,1` basis where that call finishes
//! within a deadline of 200 ms (fglm-lex.24, .26).

use grobx_checker::Mode;

use crate::common::{
    BLOCKS_DEADLINE, LEX_DEADLINE, LEX_SEEDS, PRIMES, params_in, random_square, random_system,
};
use crate::orders::{finished_in, lex_as_blocks, names};

// fglm-lex.24 design-checked -- the random cases run in lex with a deadline, at least 9 in 10 finish, and each is compared with the block:1,…,1 call where that call finishes within its own deadline (not measured)
/// The systems `draw` draws over every prime, in lex: how many finished,
/// and how many ran.
fn in_lex(kind: &str, draw: fn(u64, u64) -> String) -> (u64, u64) {
    let mut finished = 0;
    let mut cases = 0;
    for p in PRIMES {
        for seed in 0..LEX_SEEDS {
            cases += 1;
            let text = draw(p, seed);
            let name = format!("fglm-{kind}-p{p}-{seed}");
            let Some(outcome) = finished_in(&name, &text, "lex", Mode::Exact, LEX_DEADLINE) else {
                continue;
            };
            finished += 1;
            let blocks = lex_as_blocks(names(&text).len());
            let mut p = params_in(&blocks, 1);
            p.deadline = Some(BLOCKS_DEADLINE);
            let f4 = grobx::groebner(&text, &p)
                .unwrap_or_else(|e| panic!("{name} in {blocks}: {e:?}\n{text}"));
            if let Some(f4) = f4.basis {
                assert_eq!(
                    outcome.basis.unwrap(),
                    f4,
                    "{name}: lex and {blocks}\n{text}"
                );
            }
        }
    }
    (finished, cases)
}

#[test]
fn square_systems() {
    let (finished, cases) = in_lex("square", random_square);
    assert!(
        10 * finished >= 9 * cases,
        "only {finished} of {cases} square systems finished"
    );
}

#[test]
fn random_systems() {
    let (finished, cases) = in_lex("random", random_system);
    assert!(
        10 * finished >= 9 * cases,
        "only {finished} of {cases} random systems finished"
    );
}
