//! column-index-peak, the item's test code in the suite: random sparse
//! systems on four and five variables computed with cofactors and accepted
//! by the checker in the exact mode over every prime, and over ℚ, their
//! matrices' columns in 2 bytes where the round's monomials allow
//! (`iterations/0115-column-index-peak/plan.md`, _Test code_, 14).

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
                let outcome = run(&text, &with_cofactors(0x0c1d + seed));
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
