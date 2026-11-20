//! rebuilt-reducer-time, sub-item 1, the suite's test code: random sparse
//! systems on four and five variables computed with cofactors and accepted
//! by the checker in the exact mode over every prime, and over ℚ
//! (`iterations/0102-rebuilt-reducer-time/plan.md`, _Test code_, 8).

use grobx_checker::Mode;

use crate::common::{PRIMES, assert_accepted, run, systems::random_sparse, with_cofactors};
use crate::q::accepted_q;

#[test]
fn random_sparse_systems_are_accepted() {
    for p in PRIMES {
        for n in [4, 5] {
            for seed in 1..=4 {
                let name = format!("sparse{n}-p{p}-{seed}");
                let text = random_sparse(n, p, seed);
                let outcome = run(&text, &with_cofactors(0x0eb1 + seed));
                assert_accepted(&name, &text, &outcome, Mode::Exact);
            }
        }
    }
    for n in [4, 5] {
        for seed in 1..=4 {
            accepted_q(
                &format!("sparse{n}-q-{seed}"),
                &random_sparse(n, 0, seed),
                "grevlex",
            );
        }
    }
}
