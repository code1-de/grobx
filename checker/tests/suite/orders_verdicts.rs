//! Cases of `checker-orders` (its plan, _Edge cases_), hand-built and from
//! Groebner.jl: matrices
//! that are refused and matrices that are taken, the same texts ruled in two
//! orders, the leading coefficient in the order, products past `u32` in an
//! order that is not degree compatible, the cheap mode's degree bound, and
//! matrices of grevlex and lex giving those orders' verdicts.

use std::num::NonZeroU32;

use grobx_checker::{
    Error, Flaw, Mode, Order, Part, Verdict, check, check_cofactors, check_polled,
};

use crate::common::{Case, quadrics, random_system, random_system_q};
use crate::orders::{
    Jl, accept_and_reject_in, grevlex_matrix, grevlex_matrix_nonnegative, lex_matrix,
    lex_matrix_heavy, same_verdicts, solved_in, solved_permuted, variables,
};

/// A true case on x and y over GF(65521) and over Q: F = G = {x}, the row 1.
const TRUE: [(&str, &str, &str); 2] = [
    ("x,y\n65521\nx\n", "[x]:\n", "[[1]]:\n"),
    ("x,y\n0\nx\n", "[x]:\n", "[[1]]:\n"),
];

/// The same on x, y, and z.
const TRUE3: [(&str, &str, &str); 2] = [
    ("x,y,z\n65521\nx\n", "[x]:\n", "[[1]]:\n"),
    ("x,y,z\n0\nx\n", "[x]:\n", "[[1]]:\n"),
];

fn cheap(k: u32, seed: u64) -> Mode {
    Mode::Cheap {
        points: NonZeroU32::new(k).expect("k is at least 1"),
        seed,
    }
}

/// The exact verdict; an error fails the test.
fn verdict(instance: &str, basis: &str, cofactors: &str, order: &Order) -> Verdict {
    check(instance, basis, cofactors, order, Mode::Exact)
        .unwrap_or_else(|e| panic!("{order:?} refused {instance:?}, {basis:?}: {e:?}"))
}

fn accepted(v: &Verdict) -> bool {
    v.groebner.is_ok() && v.reduced.is_ok()
}

/// `order` is refused with `Error::Order` by all three functions, in both
/// modes, on each of `cases`.
fn refused(order: &Order, cases: &[(&str, &str, &str)]) {
    for &(instance, basis, cofactors) in cases {
        for mode in [Mode::Exact, cheap(4, 7)] {
            assert_eq!(
                check(instance, basis, cofactors, order, mode),
                Err(Error::Order),
                "check, {order:?}, {instance:?}"
            );
            assert_eq!(
                check_cofactors(instance, basis, cofactors, order, mode),
                Err(Error::Order),
                "check_cofactors, {order:?}, {instance:?}"
            );
            let mut never = || false;
            assert_eq!(
                check_polled(instance, basis, cofactors, order, mode, &mut never),
                Err(Error::Order),
                "check_polled, {order:?}, {instance:?}"
            );
        }
    }
}

/// `order` is taken, and the true cases are accepted in it.
fn taken(order: &Order, cases: &[(&str, &str, &str)]) {
    for &(instance, basis, cofactors) in cases {
        let v = verdict(instance, basis, cofactors, order);
        assert!(accepted(&v), "{order:?}, {instance:?}: {v:?}");
    }
}

#[test]
fn a_matrix_of_the_wrong_size_is_refused() {
    refused(&Order::Matrix(vec![vec![1, 0, 0], vec![0, 1, 0]]), &TRUE);
    refused(&Order::Matrix(vec![vec![1, 0]]), &TRUE);
    refused(&Order::Matrix(vec![]), &TRUE);
    refused(
        &Order::Matrix(vec![vec![1, 0], vec![0, 1], vec![1, 1]]),
        &TRUE,
    );
    refused(&Order::Matrix(vec![vec![1, 0], vec![0]]), &TRUE);
    refused(&Order::Matrix(vec![vec![1, 0], vec![0, 1]]), &TRUE3);
}

#[test]
fn a_singular_matrix_is_refused() {
    refused(&Order::Matrix(vec![vec![1, 1], vec![2, 2]]), &TRUE);
    // The second column is zero.
    refused(&Order::Matrix(vec![vec![1, 0], vec![1, 0]]), &TRUE);
    // The third row is the sum of the first two, and every column's first
    // nonzero entry is positive.
    refused(
        &Order::Matrix(vec![vec![1, 1, 0], vec![0, 1, 1], vec![1, 2, 1]]),
        &TRUE3,
    );
}

#[test]
fn a_negative_first_entry_is_refused() {
    // Each is nonsingular.
    refused(&Order::Matrix(vec![vec![1, -1], vec![0, 1]]), &TRUE);
    refused(&Order::Matrix(vec![vec![-1, 0], vec![0, 1]]), &TRUE);
    refused(&Order::Matrix(vec![vec![0, 1], vec![-1, 0]]), &TRUE);
}

#[test]
fn matrices_that_fit_are_taken() {
    // Lex with y > x, grevlex, a weighted order, lex with a weight of
    // i64::MAX, grevlex with y > x with a weight of i64::MIN, and lex with
    // z > x > y on three variables.
    taken(&Order::Matrix(vec![vec![0, 1], vec![1, 0]]), &TRUE);
    taken(&Order::Matrix(vec![vec![1, 1], vec![0, -1]]), &TRUE);
    taken(&Order::Matrix(vec![vec![2, 1], vec![0, 1]]), &TRUE);
    taken(&Order::Matrix(vec![vec![i64::MAX, 1], vec![0, 1]]), &TRUE);
    taken(&Order::Matrix(vec![vec![1, 1], vec![i64::MIN, 0]]), &TRUE);
    taken(
        &Order::Matrix(vec![vec![0, 0, 1], vec![1, 0, 0], vec![0, 1, 0]]),
        &TRUE3,
    );
}

#[test]
fn one_variable() {
    let cases = [
        ("x\n65521\nx^2-1\n", "[x^2-1]:\n", "[[1]]:\n"),
        ("x\n0\nx^2-1\n", "[x^2-1]:\n", "[[1]]:\n"),
    ];
    taken(&Order::Matrix(vec![vec![1]]), &cases);
    taken(&Order::Matrix(vec![vec![2]]), &cases);
    taken(&Order::Lex, &cases);
    refused(&Order::Matrix(vec![vec![0]]), &cases);
    refused(&Order::Matrix(vec![vec![-1]]), &cases);
}

/// F = {x − y², y³ − 1}, with its reduced bases in lex and in grevlex as
/// Groebner.jl 0.10.10 wrote them, and their cofactors: over GF(65521),
/// then over Q.
const TWO_BASES: [(&str, &str, &str, &str, &str); 2] = [
    (
        "x,y\n65521\nx-y^2,\ny^3-1\n",
        "[y^3+65520,\n65520*y^2+x]:\n",
        "[[0,1],\n[1,0]]:\n",
        "[y^2+65520*x,\nx*y+65520,\nx^2+65520*y]:\n",
        "[[65520,0],\n[y,1],\n[y^2+x,y]]:\n",
    ),
    (
        "x,y\n0\nx-y^2,\ny^3-1\n",
        "[y^3-1,\n-y^2+x]:\n",
        "[[0,1],\n[1,0]]:\n",
        "[y^2-x,\nx*y-1,\nx^2-y]:\n",
        "[[-1,0],\n[y,1],\n[y^2+x,y]]:\n",
    ),
];

#[test]
fn the_order_decides_the_verdict() {
    let rejected = |v: &Verdict| v.groebner.is_err() && v.reduced.is_err();
    for (instance, lex, lex_rows, grevlex, grevlex_rows) in TWO_BASES {
        for order in [Order::Lex, Order::Matrix(lex_matrix(2))] {
            let v = verdict(instance, lex, lex_rows, &order);
            assert!(accepted(&v), "the lex basis in {order:?}: {v:?}");
            let v = verdict(instance, grevlex, grevlex_rows, &order);
            assert!(rejected(&v), "the grevlex basis in {order:?}: {v:?}");
        }
        for order in [Order::Grevlex, Order::Matrix(grevlex_matrix(2))] {
            let v = verdict(instance, grevlex, grevlex_rows, &order);
            assert!(accepted(&v), "the grevlex basis in {order:?}: {v:?}");
            let v = verdict(instance, lex, lex_rows, &order);
            assert!(rejected(&v), "the lex basis in {order:?}: {v:?}");
        }
    }
}

#[test]
fn the_leading_coefficient_follows_the_order() {
    // Over GF(65521), x + 2y² is monic in lex, and its leading coefficient
    // in grevlex is 2. Over Q, x − y² has the leading coefficient 1 in lex
    // and −1 in grevlex.
    for (instance, basis) in [
        ("x,y\n65521\nx+2*y^2\n", "[x+2*y^2]:\n"),
        ("x,y\n0\nx-y^2\n", "[x-y^2]:\n"),
    ] {
        let v = verdict(instance, basis, "[[1]]:\n", &Order::Lex);
        assert!(accepted(&v), "{instance:?} in lex: {v:?}");
        let v = verdict(instance, basis, "[[1]]:\n", &Order::Grevlex);
        assert_eq!(v.groebner, Ok(()), "{instance:?} in grevlex");
        assert_eq!(v.reduced, Err(Flaw::Scale(0)), "{instance:?} in grevlex");
    }
}

#[test]
fn a_reduction_step_past_u32_is_refused() {
    // In lex, x² reduces by x − y^2147483648 to x·y^2147483648, and that to
    // y^4294967296.
    for p in ["65521", "0"] {
        let instance = format!("x,y\n{p}\nx^2\n");
        let (basis, rows) = ("[x-y^2147483648]:\n", "[[0]]:\n");
        for mode in [Mode::Exact, cheap(4, 11)] {
            assert_eq!(
                check(&instance, basis, rows, &Order::Lex, mode),
                Err(Error::Overflow {
                    part: Part::Basis,
                    line: 1
                }),
                "over {p}, {mode:?}"
            );
        }
        // In grevlex the leading term is y^2147483648, which x² is not
        // divisible by.
        let v = verdict(&instance, basis, rows, &Order::Grevlex);
        assert_eq!(v.groebner, Err(Flaw::Generator(0)), "over {p}");
    }
}

#[test]
fn an_s_polynomial_past_u32_is_refused() {
    // N = 2^31 − 1, the largest degree of an instance's term
    // (instance-grammar-agreement.5). In lex, the S-polynomial of
    // x²·z² − x·y^(N−1) and x − y^N leads with x·y^N·z², of degree 2^31 + 2.
    // Whichever element reduces it or the first generator, a multiple of
    // the second holds y^(2N)·z², of degree 2^32, and the second starts on
    // line 2.
    let (first, second) = ("x^2*z^2-x*y^2147483646", "x-y^2147483647");
    for p in ["65521", "0"] {
        let instance = format!("x,y,z\n{p}\n{first},\n{second}\n");
        let basis = format!("[{first},\n{second}]:\n");
        let rows = "[[1,0],\n[0,1]]:\n";
        assert_eq!(
            check(&instance, &basis, rows, &Order::Lex, Mode::Exact),
            Err(Error::Overflow {
                part: Part::Basis,
                line: 2
            }),
            "over {p}"
        );
        // In grevlex the leading terms are x·y^(N−1) and y^N, and every
        // product stays within u32: the check rules.
        assert!(
            check(&instance, &basis, rows, &Order::Grevlex, Mode::Exact).is_ok(),
            "over {p}"
        );
    }
}

#[test]
fn a_later_term_of_a_row_past_u32_is_refused() {
    // In lex the entry x + y^2147483649 stands x first. Its first product
    // with the generator y^2147483647, of the largest degree an instance's
    // term takes (instance-grammar-agreement.5), has degree 2^31, its
    // second 2^32.
    for p in ["65521", "0"] {
        let instance = format!("x,y\n{p}\ny^2147483647\n");
        let (basis, rows) = ("[x]:\n", "[[x+y^2147483649]]:\n");
        for mode in [Mode::Exact, cheap(4, 13)] {
            assert_eq!(
                check_cofactors(&instance, basis, rows, &Order::Lex, mode),
                Err(Error::Overflow {
                    part: Part::Cofactors,
                    line: 1
                }),
                "over {p}, {mode:?}"
            );
        }
    }
}

#[test]
fn the_cheap_bound_takes_the_largest_degree() {
    // Over GF(7) in lex, x + y⁷ − y has the leading term x, and its degree 7
    // is a tail term's. The difference of each row, y − y⁷ or y⁷ − y,
    // vanishes on all of GF(7)², so only the exact check finds it.
    let cases = [
        ("x,y\n7\nx\n", "[x+y^7-y]:\n", "[[1]]:\n"),
        ("x,y\n7\nx+y^7-y\n", "[x]:\n", "[[1]]:\n"),
    ];
    for (instance, basis, rows) in cases {
        for order in [Order::Lex, Order::Matrix(lex_matrix(2))] {
            assert_eq!(
                check_cofactors(instance, basis, rows, &order, Mode::Exact),
                Ok(Err(Flaw::Cofactors(0))),
                "{instance:?}, {basis:?}, {order:?}"
            );
            for seed in 0..20 {
                for k in [1, 4] {
                    let mode = cheap(k, seed);
                    assert_eq!(
                        check_cofactors(instance, basis, rows, &order, mode),
                        Ok(Err(Flaw::Cofactors(0))),
                        "{instance:?}, {basis:?}, {order:?}, {mode:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn check_polled_stops_in_a_matrix_order() {
    let (instance, _, _, grevlex, grevlex_rows) = TWO_BASES[0];
    let mut calls = 0;
    let mut stop = || {
        calls += 1;
        true
    };
    let order = Order::Matrix(grevlex_matrix(2));
    let got = check_polled(
        instance,
        grevlex,
        grevlex_rows,
        &order,
        Mode::Exact,
        &mut stop,
    );
    assert_eq!(got, Ok(None));
    assert_eq!(calls, 1, "the poll is not called again once it said stop");
}

/// The grevlex cases: Groebner.jl's results for the random systems over
/// GF(65521) and Q with small coefficients, and for `common::quadrics` over
/// both.
fn grevlex_cases() -> Vec<(Case, u64)> {
    let mut cases = Vec::new();
    for seed in 1..=10 {
        let name = format!("random-p65521-{seed}");
        cases.push((Case::solved(&name, &random_system(65521, seed)), seed));
        let name = format!("random-q-small-{seed}");
        cases.push((Case::solved(&name, &random_system_q(1..=3, seed)), seed));
    }
    for p in [65521, 0] {
        cases.push((Case::solved(&format!("quadrics-p{p}"), &quadrics(p)), 5));
    }
    cases
}

#[test]
#[ignore = "calls groebner-jl"]
fn matrices_of_grevlex_give_its_verdicts() {
    for (case, seed) in grevlex_cases() {
        let n = case.variables().len();
        same_verdicts(
            &case,
            &Order::Grevlex,
            &Order::Matrix(grevlex_matrix(n)),
            seed,
        );
        same_verdicts(
            &case,
            &Order::Grevlex,
            &Order::Matrix(grevlex_matrix_nonnegative(n)),
            seed,
        );
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn matrices_of_lex_give_its_verdicts() {
    for seed in 1..=10 {
        for (name, instance) in [
            (format!("random-p65521-{seed}"), random_system(65521, seed)),
            (
                format!("random-q-small-{seed}"),
                random_system_q(1..=3, seed),
            ),
        ] {
            let (case, order) = solved_in(&name, &instance, Jl::Lex);
            let n = variables(&instance);
            same_verdicts(&case, &order, &Order::Matrix(lex_matrix(n)), seed);
            same_verdicts(&case, &order, &Order::Matrix(lex_matrix_heavy(n)), seed);
        }
    }
}

#[test]
#[ignore = "calls groebner-jl"]
fn a_weight_at_the_low_end_of_i64() {
    // [[1,1],[i64::MIN,0]] is grevlex with y > x: Groebner.jl's grevlex
    // result for the instance with its two variables swapped is accepted in
    // it, and every mutant gets its verdict.
    let mut systems = 0;
    for seed in 1..=60 {
        for (name, instance) in [
            (format!("random-p65521-{seed}"), random_system(65521, seed)),
            (
                format!("random-q-small-{seed}"),
                random_system_q(1..=3, seed),
            ),
        ] {
            if variables(&instance) != 2 {
                continue;
            }
            systems += 1;
            let (case, order) =
                solved_permuted(&format!("{name}-swapped"), &instance, Jl::Grevlex, &[1, 0]);
            let heavy = Order::Matrix(vec![vec![1, 1], vec![i64::MIN, 0]]);
            accept_and_reject_in(&case, &heavy, seed);
            same_verdicts(&case, &order, &heavy, seed);
        }
    }
    assert!(systems >= 10, "{systems} random systems on two variables");
}
