//! q-modulus-time, the item's test code: `Gf64`'s row operation on
//! entries in [0, 2p) below 2^62 (q-modulus-time.3, .4, .5), and calls
//! over ℚ whose replays take primes of [2^61, 2^62) (q-modulus-time.1):
//! the edge cases, in grevlex, lex, and `block:1`, and dense systems
//! (plan.md, _Test code_, 12 to 14).

use std::time::Duration;

use grobx::Params;
use grobx::field::{Field, Gf64};
use grobx::id::MonoId;
use grobx_checker::Mode;

use crate::common::{assert_accepted, dense, params, run, with_cofactors};
use crate::q::finished_q;
use crate::reduction_time::agree;

/// Every call's deadline, far past what each takes: a fault that keeps the
/// lift from completing stops the call there and fails the test. A fault
/// that makes the replays misfit ends it in `Error::Drops` before.
const DEADLINE: Duration = Duration::from_secs(120);

/// `params` or `with_cofactors` with the deadline.
fn bounded(mut params: Params) -> Params {
    params.deadline = Some(DEADLINE);
    params
}

/// The least prime past 2^61, the largest below 2^62, the least past
/// 2^62, and the least past 2^31: `Gf64`'s redundant row operation at
/// both ends of its range, and its row operation with the correction past
/// it.
const PRIMES: [u64; 4] = [
    2_305_843_009_213_693_967,
    4_611_686_018_427_387_847,
    4_611_686_018_427_388_039,
    2_147_483_659,
];

/// The edge cases of `OBJECTIVES.md` on x, y, z over ℚ: name and
/// generators. The first seven are `matrix_sort_time.rs`'s; the last
/// three hold reduced bases of integers of 97 to 107 bits.
const EDGE: [(&str, &str); 10] = [
    ("the zero ideal", "0"),
    ("F = (0, 0)", "0,\n0"),
    ("no generator", ""),
    ("the unit ideal", "x*z-1,\nx"),
    ("already a Gröbner basis", "x+z,\nz^2+1"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
    ),
    ("an eliminable variable", "x-y^2-z,\ny*z-1,\nz^2-y"),
    (
        "already a Gröbner basis, wide",
        "x+1/12345678901234567890123*z,\nz^2+98765432109876543210987654321",
    ),
    (
        "the zero polynomial among the generators, wide",
        "3*x*y-1/1000000007*z,\n0,\ny^2-1180591620717411303424*x",
    ),
    (
        "an eliminable variable, wide",
        "x-y^2-1/97*z,\ny*z-1000000000000000000000000000000,\nz^2-y",
    ),
];

/// q-modulus-time.3, .4, .5: loads, subtractions, and takes on dense rows
/// of 1 to 12 entries against the sums in `u128`, and a million
/// subtractions of (p − 1)·(p − 1) from one entry, at each of `PRIMES`.
#[test]
fn the_redundant_row_operation() {
    for p in PRIMES {
        let f = Gf64::new(p).unwrap();
        agree(f, 0x6d34);
        let top = f.coef(p - 1);
        let mut dense = vec![<Gf64 as Field>::Acc::default(); 1];
        // SAFETY: column 0 is the dense row's one entry.
        unsafe { f.load(&mut dense, &[MonoId::from_raw(0)], &[top]) };
        for _ in 0..1_000_000 {
            // SAFETY: as above.
            unsafe { f.sub_mul(&mut dense, &[MonoId::from_raw(0)], &[top], top) };
        }
        // (p − 1) − 10^6·(p − 1)² ≡ −1 − 10^6 mod p
        let q = u128::from(p);
        let want = (q - 1 + q - 1_000_000 % q) % q;
        assert_eq!(u128::from(f.value(f.take(&mut dense[0]))), want, "GF({p})");
        assert_eq!(f.value(f.take(&mut dense[0])), 0, "GF({p}): take leaves 0");
    }
}

/// q-modulus-time.1: the edge cases over ℚ in grevlex, lex, and
/// `block:1`, each canonical and accepted by the checker in the exact mode
/// with its cofactors (`q::finished_q`, as `q::accepted_q` with the
/// deadline); in grevlex the call without cofactors returns the same basis.
#[test]
fn the_edge_cases_over_q() {
    for (case, generators) in EDGE {
        let text = format!("x,y,z\n0\n{generators}\n");
        for order in ["grevlex", "lex", "block:1"] {
            let outcome = finished_q(&format!("{case} over ℚ"), &text, order, DEADLINE)
                .unwrap_or_else(|| panic!("{case} in {order}: stopped by the deadline"));
            if order == "grevlex" {
                let plain = run(&text, &bounded(params(0x6d35)));
                assert_eq!(plain.basis, outcome.basis, "{case}");
            }
        }
    }
}

/// q-modulus-time.1: dense cubic systems over ℚ with cofactors, each
/// accepted in the exact mode.
#[test]
fn dense_systems_over_q() {
    for seed in 1..=3 {
        let text = dense(3, 3, 0, seed);
        let outcome = run(&text, &bounded(with_cofactors(1)));
        assert_accepted(&format!("cubic, seed {seed}"), &text, &outcome, Mode::Exact);
    }
}
