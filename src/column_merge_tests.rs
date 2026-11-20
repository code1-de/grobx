//! sort-share-time, the item's test code: the matrices with kept rows,
//! whose columns come merged from the matrix before's order and whose kept
//! rows map through it, stay as the build without kept rows makes them,
//! at the matrix level and end to end, and a stop inside such a build
//! (`iterations/0105-sort-share-time/plan.md`, _Test code_, 3 to 5).

use std::cmp::Ordering;
use std::time::Duration;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::field::Gf31;
use crate::id::RowSource;
use crate::matrix::{self, Buffers, KEPT_OFF, Matrix};
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::Basis;
use crate::rebuilt_reducer_tests::{agree, element, halves, pair, reported, same, systems};
use crate::split_piece_time_tests::EDGE;
use crate::table::Table;
use crate::{MonomialOrder, Stop};

fn budget() -> Budget {
    Budget::new(std::time::Instant::now(), &reported(MonomialOrder::Grevlex)).unwrap()
}

/// The basis of `a_kept_row_is_taken_and_the_rest_compacted`: leads x²,
/// y², z², xy, and w² in grevlex.
fn basis(table: &mut Table<Dense16>, field: &Gf31) -> Basis<Gf31, Rank1> {
    let mut basis: Basis<Gf31, Rank1> = Basis::new();
    for terms in [
        [([2, 0, 0, 0], 1), ([0, 1, 1, 0], 3)],
        [([0, 2, 0, 0], 1), ([1, 0, 0, 1], 5)],
        [([0, 0, 2, 0], 1), ([0, 0, 0, 1], 7)],
        [([1, 1, 0, 0], 1), ([0, 0, 1, 1], 11)],
        [([0, 0, 0, 2], 1), ([1, 0, 0, 0], 13)],
    ] {
        let g = element(table, field, &terms);
        basis.push_min(table, g, &mut Budget::unbounded()).unwrap();
    }
    basis
}

/// The build of `given` with the keeping off, in buffers of its own: what
/// every build must equal.
fn reference(
    table: &mut Table<Dense16>,
    basis: &Basis<Gf31, Rank1>,
    given: &[RowSource],
) -> Matrix {
    KEPT_OFF.set(true);
    let mut bufs: Buffers<Gf31> = Buffers::new();
    let m = matrix::build(
        table,
        basis,
        &Grevlex,
        given.to_vec(),
        &mut bufs,
        &mut budget(),
    )
    .unwrap();
    KEPT_OFF.set(false);
    m
}

/// Test 3, sort-share-time.1 to .4: four builds in a row, each with the
/// matrix before kept, equal the builds without: new columns among old
/// ones (round 2), no new column (round 3), and a matrix before wider than
/// the round (round 4).
#[test]
fn the_merged_columns_are_the_sorted_ones() {
    let field = Gf31::new(65521).unwrap();
    let mut table = Table::new(Dense16::new(4), &mut Xoshiro256PlusPlus::seed_from_u64(7));
    let basis = basis(&mut table, &field);
    let first: Vec<RowSource> = [
        halves(&mut table, &basis, 0, 3),
        halves(&mut table, &basis, 1, 3),
    ]
    .concat();
    let mut second = first.clone();
    second.extend(halves(&mut table, &basis, 2, 4));
    let third = second.clone();
    let fourth: Vec<RowSource> = halves(&mut table, &basis, 2, 4).to_vec();
    let mut bufs: Buffers<Gf31> = Buffers::new();
    for (k, given) in [first, second, third, fourth].iter().enumerate() {
        let round = u32::try_from(k).unwrap() + 1;
        let m = matrix::build(
            &mut table,
            &basis,
            &Grevlex,
            given.clone(),
            &mut bufs,
            &mut budget(),
        )
        .unwrap();
        let want = reference(&mut table, &basis, given);
        same(&format!("round {round}"), &m, &want);
        if round > 1 {
            assert!(m.kept() > 0 || round == 4, "round {round} took no kept row");
        }
        for (c, &u) in m.monomials.iter().enumerate() {
            if c > 0 {
                assert!(
                    matches!(
                        table.cmp(&Grevlex, m.monomials[c - 1], u),
                        Ordering::Greater
                    ),
                    "round {round}: column {c} out of order"
                );
            }
        }
        bufs.keep(m);
    }
}

/// A system of 14 variables whose grevlex keys tie: generators of degree
/// 2 whose leads share variables, `x_i·x_{i+1} + x_{i+2}²`, and the last
/// closing the chain.
fn chain14() -> String {
    let names: Vec<String> = (1..=14).map(|i| format!("x{i}")).collect();
    let mut gens = Vec::new();
    for i in 0..12 {
        let (a, b, c) = (&names[i], &names[i + 1], &names[i + 2]);
        gens.push(format!("{a}*{b}+{c}^2"));
    }
    let (a, b, c) = (&names[12], &names[13], &names[0]);
    gens.push(format!("{a}*{b}+{c}^2"));
    let (a, b) = (&names[13], &names[1]);
    gens.push(format!("{a}^2+{b}"));
    let (variables, generators) = (names.join(","), gens.join(",\n"));
    format!("{variables}\n65521\n{generators}")
}

/// Test 4: end to end, the bases, the round lines, and the reductions
/// counted with the keeping on equal those with it off, on the small
/// systems over GF(65521) and ℚ in grevlex, lex, and `block:2`, on the
/// edge cases, and on a system of 14 variables whose keys tie.
#[test]
fn the_calls_stay() {
    let mut cases: Vec<(String, String)> = systems(65521);
    cases.extend(systems(0));
    for p in [65521, 0] {
        cases.extend(EDGE.iter().enumerate().map(|(k, gens)| {
            (
                format!("edge {k} over {p}"),
                format!("x,y,z\n{p}\n{gens}\n"),
            )
        }));
    }
    cases.push(("chain14".to_string(), chain14()));
    for order in ["grevlex", "lex", "block:2"] {
        let order: MonomialOrder = order.parse().unwrap();
        let mut params = reported(order.clone());
        params.deadline = Some(Duration::from_secs(20));
        // The system of 14 variables runs in grevlex alone: lex and the
        // block order take its keys by the unkeyed sort. The sparse systems
        // over ℚ compute in grevlex alone, their lift in lex combining
        // thousands of primes; and the block call, which computes in the
        // block order itself, takes seconds on the dense cubics over ℚ and
        // on the sparse systems on five variables.
        let runs = |name: &str| match order {
            MonomialOrder::Grevlex | MonomialOrder::Deglex => true,
            MonomialOrder::Lex => {
                name != "chain14" && !(name.starts_with("sparse") && name.contains("-p0-"))
            }
            MonomialOrder::Block(_) => {
                name.starts_with("sparse4-p65521")
                    || name.starts_with("dense33-p65521")
                    || name.starts_with("edge")
            }
        };
        for (name, text) in &cases {
            if !runs(name) {
                continue;
            }
            let (off, on) = pair(text, &params);
            agree(&format!("{name} in {order}"), &off, &on);
        }
    }
}

/// Test 5: a deadline that stops inside a build with kept rows returns
/// `Deadline` and no basis.
#[test]
fn a_stop_inside_a_build_with_kept_rows() {
    let mut params = reported(MonomialOrder::Grevlex);
    params.deadline = Some(Duration::from_micros(300));
    let text = chain14();
    let outcome = crate::groebner(&text, &params).unwrap();
    assert_eq!(outcome.stopped, Some(Stop::Deadline), "{outcome:?}");
    assert!(outcome.basis.is_none());
}
