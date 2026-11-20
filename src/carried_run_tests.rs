//! signature-run-losses, sub-item 1: the plain run from carried elements,
//! unwired. The bound of a carry, the pairs a bound leaves out, the
//! carry's fresh table, and the plain run from a carry
//! (`iterations/0112-signature-run-losses/plan.md`, _Test code_, 1 to 4).

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stats;
use crate::budget::Budget;
use crate::cofactors::NoCofactors;
use crate::divisibility_tests::{Engine, RefPair};
use crate::f4::{self, Carried};
use crate::field::Gf31;
use crate::handover_tests::Call;
use crate::id::MonoId;
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::Poly;
use crate::shared_tests::GF31_PRIMES;
use crate::systems::{random_square, random_system};
use crate::table::Table;
use crate::trace::{NoLearner, NoTrace};

/// The seeds of tests 2 to 4.
pub(crate) const SEEDS: u64 = 8;

/// A bound at or past the sugar of every pair of test 4's systems, whose
/// degrees stay below it.
const PAST_EVERY_SUGAR: u32 = 1_000;

/// The most elements test 2 enters: their pairs stay below `DEFER_LIVE`,
/// so that both pair sets defer alike.
const MOST: u64 = 12;

fn id(table: &mut Table<Dense16>, e: [u32; 3]) -> MonoId {
    table.intern(&e, &mut Budget::unbounded()).unwrap()
}

/// Test 1: signature-run-losses.4. The round's degree where no older
/// leading monomial divides a made one; one less where one divides it at
/// a multiplier whose degree plus the older element's sugar is at most
/// the round's degree; the round's degree where the sum passes it, and
/// where a made leading monomial divides an older one alone.
#[test]
fn the_bound_reads_older_divisors() {
    let mut table = Table::new(Dense16::new(3), &mut Xoshiro256PlusPlus::seed_from_u64(1));
    let x2 = id(&mut table, [2, 0, 0]);
    let x3 = id(&mut table, [3, 0, 0]);
    let y3 = id(&mut table, [0, 3, 0]);
    let x2y = id(&mut table, [2, 1, 0]);
    let xyz2 = id(&mut table, [1, 1, 2]);
    assert_eq!(
        f4::carry_bound(
            &table,
            5,
            &[(x2, 2), (y3, 3)],
            &[xyz2],
            &mut Budget::unbounded()
        )
        .unwrap(),
        5
    );
    // x^2 divides x^2·y at the multiplier y, of degree 1.
    assert_eq!(
        f4::carry_bound(&table, 4, &[(x2, 3)], &[x2y], &mut Budget::unbounded()).unwrap(),
        3,
        "the sum at the degree"
    );
    assert_eq!(
        f4::carry_bound(&table, 4, &[(x2, 2)], &[x2y], &mut Budget::unbounded()).unwrap(),
        3,
        "the sum below it"
    );
    assert_eq!(
        f4::carry_bound(&table, 4, &[(x2, 4)], &[x2y], &mut Budget::unbounded()).unwrap(),
        4,
        "the sum past it"
    );
    assert_eq!(
        f4::carry_bound(&table, 4, &[(x3, 3)], &[x2], &mut Budget::unbounded()).unwrap(),
        4,
        "the made one divides"
    );
    assert_eq!(
        f4::carry_bound(&table, 4, &[], &[x2], &mut Budget::unbounded()).unwrap(),
        4,
        "no older element"
    );
}

/// A random leading monomial on `n` variables of degree 1 to 5.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a draw below n, a variable's index"
)]
fn random_lead(rng: &mut Xoshiro256PlusPlus, n: usize) -> Vec<u32> {
    let degree = 1 + rng.next_u64() % 5;
    let mut e = vec![0u32; n];
    for _ in 0..degree {
        e[(rng.next_u64() % n as u64) as usize] += 1;
    }
    e
}

/// Test 2: signature-run-losses.3. Random leading monomials entered with
/// their sugars into two pair sets, one with a bound and one without:
/// each pair of the first is one of the second, and the second's pairs
/// the first lacks are those whose sugar,
/// `max(s_i − deg lm g_i, s_j − deg lm g_j) + deg lcm`, is at most the
/// bound.
#[test]
fn a_bound_leaves_out_the_covered_pairs() {
    let mut left_out = 0;
    for seed in 0..SEEDS * 5 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5e1 ^ seed);
        let n = 3 + (rng.next_u64() % 3) as usize;
        let count = 4 + rng.next_u64() % (MOST - 3);
        let bound = 2 + (rng.next_u64() % 6) as u32;
        let mut free: Engine<Dense16> = Engine::new(n, seed);
        let mut bounded: Engine<Dense16> = Engine::new(n, seed);
        let mut sugars = Vec::new();
        let mut degrees = Vec::new();
        for _ in 0..count {
            let lead = random_lead(&mut rng, n);
            let degree: u32 = lead.iter().sum();
            let sugar = degree + (rng.next_u64() % 3) as u32;
            free.enter_with(&Grevlex, &lead, sugar, 0).0.unwrap();
            bounded.enter_with(&Grevlex, &lead, sugar, bound).0.unwrap();
            sugars.push(sugar);
            degrees.push(degree);
        }
        let sugar = |p: &RefPair| {
            let (i, j) = (p.i as usize, p.j as usize);
            let fall = (sugars[i] - degrees[i]).max(sugars[j] - degrees[j]);
            fall + p.lcm.iter().sum::<u32>()
        };
        let by_elements = |mut pairs: Vec<RefPair>| {
            pairs.sort_by_key(|p| (p.i, p.j));
            pairs
        };
        let all = by_elements(free.written());
        let want: Vec<RefPair> = all.iter().filter(|p| sugar(p) > bound).cloned().collect();
        left_out += all.len() - want.len();
        assert_eq!(
            by_elements(bounded.written()),
            want,
            "seed {seed}, bound {bound}"
        );
    }
    assert!(left_out > 0, "no pair at or below a bound");
}

/// The nonzero generators of `call`, monic, each with its top degree as
/// its sugar.
fn monic_inputs(call: &mut Call) -> Vec<(Poly<Gf31, Rank1>, u32, ())> {
    let field = call.field;
    call.generators()
        .into_iter()
        .filter(|g| !g.is_zero())
        .map(|mut g| {
            g.make_monic(&field);
            let top = call.table.degree(g.lead());
            (g, top, ())
        })
        .collect()
}

/// A carry of `elements` with the bound `bound`.
fn carry(elements: Vec<(Poly<Gf31, Rank1>, u32, ())>, bound: u32) -> Carried<Gf31, Rank1, ()> {
    Carried {
        elements,
        bound,
        round: 1,
        made: 0,
        work: 0,
    }
}

/// Test 3: signature-run-losses.5. The carried elements in a fresh table
/// keep their terms' exponent vectors, their coefficients, and the order
/// of their terms, and the fresh table holds their monomials and no
/// other.
#[test]
fn the_fresh_table_keeps_the_elements() {
    for p in GF31_PRIMES {
        for seed in 0..SEEDS {
            let text = random_system(p, seed);
            let mut call = Call::new(&text);
            let mut carried = carry(monic_inputs(&mut call), 0);
            let terms = |table: &Table<Dense16>, carried: &Carried<Gf31, Rank1, ()>| {
                carried
                    .elements
                    .iter()
                    .map(|(g, _, ())| {
                        let exponents: Vec<Vec<u32>> = g
                            .monomials
                            .iter()
                            .map(|&m| {
                                let mut e = Vec::new();
                                table.exponents(m, &mut e);
                                e
                            })
                            .collect();
                        (exponents, g.coefs.clone())
                    })
                    .collect::<Vec<_>>()
            };
            let before = terms(&call.table, &carried);
            let mut fresh = Table::new(
                Dense16::new(call.table.format().variables()),
                &mut Xoshiro256PlusPlus::seed_from_u64(seed ^ 0xf5e),
            );
            f4::carry_table(
                &call.table,
                &mut fresh,
                &mut carried,
                &mut Budget::unbounded(),
            )
            .unwrap();
            assert_eq!(
                terms(&fresh, &carried),
                before,
                "p {p}, seed {seed}\n{text}"
            );
            let mut distinct: Vec<Vec<u32>> = before.into_iter().flat_map(|(e, _)| e).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(fresh.len(), distinct.len(), "p {p}, seed {seed}\n{text}");
        }
    }
}

/// The basis text of the plain run from `carried` over `call`'s table.
fn carried_basis(call: &mut Call, carried: Carried<Gf31, Rank1, ()>) -> String {
    let elements = f4::run_carried(
        &call.field,
        &mut call.table,
        &Grevlex,
        carried,
        &mut NoCofactors,
        &mut NoTrace,
        &mut Budget::unbounded(),
        &mut Stats::default(),
        &mut NoLearner,
        1,
    )
    .unwrap();
    call.text_of(&elements)
}

/// Test 4: signature-run-losses.1, .3. Random systems and random square
/// systems over each prime: the plain run from a carry of the monic
/// inputs at their top degree with the bound 0, and from a carry of the
/// inputs and their reduced basis with a bound past every pair's sugar,
/// gives the reduced basis of the run from the input.
#[test]
fn a_carried_run_gives_the_plain_basis() {
    let mut carried_bases = 0;
    for p in GF31_PRIMES {
        for seed in 0..SEEDS {
            for text in [random_system(p, seed), random_square(p, seed)] {
                let want = Call::new(&text).plain();
                let mut call = Call::new(&text);
                let inputs = monic_inputs(&mut call);
                if inputs
                    .iter()
                    .any(|(g, _, ())| call.table.degree(g.lead()) == 0)
                {
                    // A constant input ends the run before any carry.
                    continue;
                }
                assert_eq!(
                    carried_basis(&mut call, carry(inputs, 0)),
                    want,
                    "inputs\n{text}"
                );
                let mut call = Call::new(&text);
                let mut elements = monic_inputs(&mut call);
                let generators = call.generators();
                let basis = f4::run(
                    &call.field,
                    &mut call.table,
                    &Grevlex,
                    generators,
                    &mut NoCofactors,
                    &mut NoTrace,
                    &mut Budget::unbounded(),
                    &mut Stats::default(),
                    &mut NoLearner,
                    1,
                )
                .unwrap();
                if basis.iter().any(|(g, ())| call.table.degree(g.lead()) == 0) {
                    continue;
                }
                elements.extend(basis.into_iter().map(|(g, ())| {
                    let top = call.table.degree(g.lead());
                    (g, top, ())
                }));
                // The run's rounds stamped the table's monomials, and a run
                // starts in a table that has built no matrix: the carry's
                // fresh table.
                let mut carried = carry(elements, PAST_EVERY_SUGAR);
                let mut fresh = Table::new(
                    Dense16::new(call.table.format().variables()),
                    &mut Xoshiro256PlusPlus::seed_from_u64(1),
                );
                f4::carry_table(
                    &call.table,
                    &mut fresh,
                    &mut carried,
                    &mut Budget::unbounded(),
                )
                .unwrap();
                call.table = fresh;
                assert_eq!(
                    carried_basis(&mut call, carried),
                    want,
                    "the basis carried\n{text}"
                );
                carried_bases += 1;
            }
        }
    }
    assert!(carried_bases > 0, "no system whose basis is carried");
}
