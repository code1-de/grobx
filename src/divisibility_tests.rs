//! divisibility-time, the crate's test code: the divisor masks against
//! masks built from the exponents, `find_passing` against a scan of every
//! mask, and the pair update against joux-2011's Algorithm 2 written out
//! on exponent vectors (plan.md, _Test code_, 1 to 6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::ops::RangeInclusive;

use rand_core::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::field::{Field, Gf31};
use crate::halt::Halt;
use crate::id::{ElementId, MonoId};
use crate::module::Rank1;
use crate::monomial::{Dense8, Dense16, Dense32, Format, Overflow};
use crate::order::{Grevlex, Lex, Order};
use crate::pairs::{self, DEFER_LIVE, NewPairs, Pair, PairSet};
use crate::poly::{Basis, Poly};
use crate::shared_tests::{below, coprime, divides, rng};
use crate::table::{Limit, Table, find_passing};

/// The mask of these exponents as the format states it: up to 32
/// variables, bit k of variable i set where the exponent passes k, with
/// 32 / n bits a variable; past 32, bit i mod 32 set where the exponent of
/// variable i is not 0.
pub(crate) fn mask_of(exponents: &[u32]) -> u32 {
    let n = exponents.len();
    if n == 0 {
        return 0;
    }
    if n > 32 {
        return exponents
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e > 0)
            .fold(0, |mask, (i, _)| mask | 1 << (i % 32));
    }
    let bits = 32 / n as u32;
    let mut mask = 0u32;
    for (i, &e) in exponents.iter().enumerate() {
        let set = e.min(bits);
        mask |= (((1u64 << set) - 1) as u32) << (i as u32 * bits);
    }
    mask
}

/// Random exponents for `n` variables, each at most `top`: about a third
/// of them 0, a sixth at `top`, a sixth from 0 to `top`, and the rest from
/// 1 to 4, and the sum at most `sum`, or at most `u32::MAX` where `sum` is
/// None: in `Dense32` two lanes at `top` leave every later lane 0 or 1.
pub(crate) fn random_lanes(
    rng: &mut Xoshiro256PlusPlus,
    n: usize,
    top: u32,
    sum: Option<u32>,
) -> Vec<u32> {
    let mut left = sum.unwrap_or(u32::MAX);
    let mut e = vec![0u32; n];
    for x in &mut e {
        let v = match below(rng, 6) {
            0 | 1 => 0,
            2 => top,
            3 => below(rng, u64::from(top) + 1) as u32,
            _ => 1 + below(rng, 4) as u32,
        };
        let v = v.min(top).min(left);
        *x = v;
        left -= v;
    }
    // Not always the first variables first.
    let shift = below(rng, n.max(1) as u64) as usize;
    e.rotate_right(shift);
    e
}

/// Test 1 for one format on `ns` variables, `per` draws each: its masks
/// against `mask_of`, on monomials of the table's kind and on packed lanes
/// of a sum up to `u32::MAX`, and the divisor property.
pub(crate) fn check_masks_on<M: Format>(seed: u64, ns: RangeInclusive<usize>, per: usize) {
    let mut rng = rng(seed);
    for n in ns {
        let format = M::new(n);
        let top = M::MAX_DEGREE;
        let mut words = vec![M::Word::default(); format.words()];
        for _ in 0..per {
            // Lanes of a sum up to u32::MAX, each at most the degree bound.
            let e = random_lanes(&mut rng, n, top, None);
            format.pack(&e, &mut words);
            assert_eq!(format.mask(&words), mask_of(&e), "n {n}, lanes {e:?}");
            // A monomial of the table's kind: its degree at most the bound.
            let a = random_lanes(&mut rng, n, top, Some(top));
            format.encode(&a, &mut words).unwrap();
            let mask_a = format.mask(&words);
            assert_eq!(mask_a, mask_of(&a), "n {n}, exponents {a:?}");
            // A multiple of it, still within the bound.
            let room = top - a.iter().sum::<u32>();
            let c = random_lanes(&mut rng, n, room, Some(room));
            let b: Vec<u32> = a.iter().zip(&c).map(|(x, y)| x + y).collect();
            format.encode(&b, &mut words).unwrap();
            let mask_b = format.mask(&words);
            assert_eq!(mask_b, mask_of(&b), "n {n}, exponents {b:?}");
            assert_eq!(mask_a & !mask_b, 0, "n {n}: {a:?} divides {b:?}");
        }
    }
}

/// Test 1: divisibility-time.5, .6.
#[test]
fn masks_agree_with_the_exponents() {
    check_masks_on::<Dense8>(0xd1f1_0001, 0..=70, 2000);
    check_masks_on::<Dense16>(0xd1f1_0002, 0..=70, 2000);
    check_masks_on::<Dense32>(0xd1f1_0003, 0..=70, 2000);
}

/// Test 2: divisibility-time.7. `find_passing` against a scan of every
/// mask: the same answer, and the rule hears the passing masks in
/// increasing order up to the one it accepts.
#[test]
fn the_search_agrees_with_a_scan() {
    let mut rng = rng(0xd1f1_0004);
    for len in 0..=41usize {
        for round in 0..300 {
            // Masks of a few bits, and a query of many.
            let masks: Vec<u32> = (0..len)
                .map(|_| {
                    let mut m = 0u32;
                    for _ in 0..below(&mut rng, 5) {
                        m |= 1 << below(&mut rng, 32);
                    }
                    m
                })
                .collect();
            let mut query = 0u32;
            for _ in 0..below(&mut rng, 40) {
                query |= 1 << below(&mut rng, 32);
            }
            // The rule accepts a random subset, none in some rounds.
            let salt = rng.next_u64();
            let none = round % 5 == 0;
            let accepts =
                |k: usize| !none && (salt >> (k % 64)) & 1 == 1 && !(salt >> 7).is_multiple_of(3);
            let passing: Vec<usize> = (0..len).filter(|&k| masks[k] & !query == 0).collect();
            let want = passing.iter().copied().find(|&k| accepts(k));
            let mut heard = Vec::new();
            let got = find_passing(&masks, query, |k| {
                heard.push(k);
                accepts(k)
            });
            assert_eq!(got, want, "len {len}, round {round}");
            let expected: Vec<usize> = match want {
                Some(w) => passing.iter().copied().filter(|&k| k <= w).collect(),
                None => passing.clone(),
            };
            assert_eq!(heard, expected, "len {len}, round {round}: the masks heard");
        }
    }
}

/// A pair as joux-2011's Algorithm 2 keeps it, on exponent vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RefPair {
    pub(crate) i: u32,
    pub(crate) j: u32,
    pub(crate) lcm: Vec<u32>,
    pub(crate) degree: u32,
    pub(crate) sugar: u32,
}

pub(crate) fn lcm(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| *x.max(y)).collect()
}

pub(crate) fn degree(e: &[u32]) -> u32 {
    e.iter().sum()
}

/// The update as joux-2011 §2.2's Algorithm 2 states it, with the code
/// before's `B_t`, which also drops a pair where the pair of the new lead
/// and one of its elements has its lcm and coprime leading monomials
/// (plain-zero-pairs-time.1), sugar, and Overflow, and in a graded order
/// the pair of a fallen element one degree past its lcm's until the first
/// update that begins with more than `DEFER_LIVE` pairs
/// (cooper-rest-gap.1, .2, fallen-pair-losses.1): the reference of test 3.
pub(crate) struct Reference {
    graded: bool,
    bound: u32,
    leads: Vec<Vec<u32>>,
    sugar: Vec<u32>,
    min: Vec<u32>,
    pub(crate) pairs: Vec<RefPair>,
    /// No update so far began with more than `DEFER_LIVE` pairs.
    defers: bool,
}

impl Reference {
    pub(crate) fn new(graded: bool, bound: u32) -> Reference {
        Reference {
            graded,
            bound,
            leads: Vec::new(),
            sugar: Vec::new(),
            min: Vec::new(),
            pairs: Vec::new(),
            defers: true,
        }
    }

    /// The members of `G_min`, by index, in the order the update keeps them.
    pub(crate) fn min_members(&self) -> &[u32] {
        &self.min
    }

    /// The sugar of the pair (i, t) of lcm degree `degree`.
    fn pair_sugar(&self, i: u32, t_lead: &[u32], t_sugar: u32, degree: u32) -> u32 {
        let si = self.sugar[i as usize] - self::degree(&self.leads[i as usize]);
        let st = t_sugar - self::degree(t_lead);
        let fall = si.max(st);
        if !self.graded {
            return fall.saturating_add(degree);
        }
        if fall > 0 && self.defers {
            return degree + 1;
        }
        degree
    }

    pub(crate) fn update(&mut self, lead: Vec<u32>, sugar: u32) -> Result<(), Overflow> {
        let t = self.leads.len() as u32;
        if self.pairs.len() > DEFER_LIVE {
            self.defers = false;
        }
        let leads = &self.leads;
        // B_t.
        self.pairs.retain(|p| {
            if !divides(&lead, &p.lcm) {
                return true;
            }
            let open = |l: &[u32]| degree(&lcm(l, &lead)) == p.degree && !coprime(l, &lead);
            open(&leads[p.i as usize]) || open(&leads[p.j as usize])
        });
        // P0 and P1.
        let mut p0: Vec<RefPair> = Vec::new();
        let mut p1: Vec<RefPair> = Vec::new();
        for &i in &self.min {
            let li = &self.leads[i as usize];
            let l = lcm(li, &lead);
            let d = degree(&l);
            let s = self.pair_sugar(i, &lead, sugar, d);
            let pair = RefPair {
                i,
                j: t,
                lcm: l,
                degree: d,
                sugar: s,
            };
            if coprime(li, &lead) {
                if d <= self.bound {
                    p0.push(pair);
                }
            } else {
                if d > self.bound {
                    return Err(Overflow);
                }
                p1.push(pair);
            }
        }
        // M and F, lines 14 to 19.
        let mut kept: Vec<RefPair> = Vec::new();
        for k in 0..p1.len() {
            let p = &p1[k];
            let div = |q: &RefPair| divides(&q.lcm, &p.lcm);
            if !p0.iter().any(div) && !p1[k + 1..].iter().any(div) && !kept.iter().any(div) {
                kept.push(p.clone());
            }
        }
        self.pairs.extend(kept);
        // G_min, lines 21 to 27.
        let divided = self
            .min
            .iter()
            .any(|&i| divides(&self.leads[i as usize], &lead));
        if !divided {
            let leads = &self.leads;
            self.min.retain(|&i| !divides(&lead, &leads[i as usize]));
            self.min.push(t);
        }
        self.leads.push(lead);
        self.sugar.push(sugar);
        Ok(())
    }
}

pub(crate) fn exponents<M: Format>(table: &Table<M>, id: MonoId) -> Vec<u32> {
    (0..table.format().variables())
        .map(|i| table.exponent(id, i))
        .collect()
}

/// The monic monomial of these exponents as a polynomial of the table.
pub(crate) fn monomial<M: Format>(
    table: &mut Table<M>,
    field: &Gf31,
    e: &[u32],
) -> Poly<Gf31, Rank1> {
    Poly {
        monomials: vec![table.intern(e, &mut Budget::unbounded()).unwrap()],
        coefs: vec![field.coef(1)],
        position: (),
    }
}

/// The pairs as the reference writes them, their lcms read from `table`.
pub(crate) fn written<'a, M: Format>(
    table: &Table<M>,
    pairs: impl IntoIterator<Item = &'a Pair>,
) -> Vec<RefPair> {
    pairs
        .into_iter()
        .map(|p| RefPair {
            i: p.i.index() as u32,
            j: p.j.index() as u32,
            lcm: exponents(table, p.lcm),
            degree: p.degree,
            sugar: p.sugar,
        })
        .collect()
}

/// The engine's side of tests 3 to 6: a table, a basis, its pairs, its
/// sugars, and the update's buffers.
pub(crate) struct Engine<M: Format> {
    pub(crate) table: Table<M>,
    basis: Basis<Gf31, Rank1>,
    pub(crate) pairs: PairSet<M::Word>,
    sugar: Vec<u32>,
    new: NewPairs<M::Word>,
    field: Gf31,
}

impl<M: Format> Engine<M> {
    pub(crate) fn new(n: usize, seed: u64) -> Engine<M> {
        let table = Table::new(M::new(n), &mut rng(seed));
        Engine {
            pairs: PairSet::new(table.format().words()),
            table,
            basis: Basis::new(),
            sugar: Vec::new(),
            new: NewPairs::new(),
            field: Gf31::new(65521).unwrap(),
        }
    }

    /// Enters the element of leading monomial `lead` and sugar `sugar`:
    /// the update's result, and the ids the table gained in it.
    pub(crate) fn enter<O: Order<M>>(
        &mut self,
        order: &O,
        lead: &[u32],
        sugar: u32,
    ) -> (Result<(), Limit>, std::ops::Range<u32>) {
        self.enter_with(order, lead, sugar, 0)
    }

    /// As `enter`, the update leaving out the pairs whose sugar is at
    /// most `covered` (signature-run-losses.3).
    pub(crate) fn enter_with<O: Order<M>>(
        &mut self,
        order: &O,
        lead: &[u32],
        sugar: u32,
        covered: u32,
    ) -> (Result<(), Limit>, std::ops::Range<u32>) {
        let (got, ids) = self.enter_under(order, lead, sugar, covered, &mut Budget::unbounded());
        let got = got.map_err(|halt| match halt {
            Halt::Overflow => Limit::Overflow,
            Halt::Size => Limit::Size,
            Halt::Stop(_) | Halt::Misfit(_) | Halt::Handover | Halt::Outworked => {
                panic!("an update under an unbounded budget halted: {halt:?}")
            }
        });
        (got, ids)
    }

    /// As `enter_with`, the update spending `budget`
    /// (call-stop-latency.1).
    pub(crate) fn enter_under<O: Order<M>>(
        &mut self,
        order: &O,
        lead: &[u32],
        sugar: u32,
        covered: u32,
        budget: &mut Budget,
    ) -> (Result<(), Halt>, std::ops::Range<u32>) {
        let h = monomial(&mut self.table, &self.field, lead);
        self.sugar.push(sugar);
        let before = self.table.len() as u32;
        let got = pairs::update(
            &self.table,
            order,
            &mut self.basis,
            &mut self.pairs,
            &self.sugar,
            h,
            &mut self.new,
            covered,
            budget,
        );
        (got, before..self.table.len() as u32)
    }

    /// The members of `G_min`.
    pub(crate) fn min_len(&self) -> usize {
        self.basis.min().len()
    }

    /// The pairs as the reference writes them, their lcms read from the
    /// set's words: a live pair's lcm is interned by no update
    /// (pair-update-rest-time.1).
    pub(crate) fn written(&self) -> Vec<RefPair> {
        let format = self.table.format();
        let mut e = vec![0u32; format.variables()];
        self.pairs
            .with_words()
            .map(|(p, words)| {
                format.unpack(words, &mut e);
                RefPair {
                    i: p.i.index() as u32,
                    j: p.j.index() as u32,
                    lcm: e.clone(),
                    degree: p.degree,
                    sugar: p.sugar,
                }
            })
            .collect()
    }

    /// `G_min`'s members' ids and masks as the update's arrays hold them,
    /// owned.
    pub(crate) fn min_arrays(&self) -> (Vec<ElementId>, Vec<u32>) {
        self.new.min_arrays()
    }

    /// `G_min` as the basis holds it.
    pub(crate) fn min_members(&self) -> &[ElementId] {
        self.basis.min()
    }

    /// The mask of element i's leading monomial.
    pub(crate) fn lead_mask(&self, i: ElementId) -> u32 {
        self.table.mask(self.basis.lead(i))
    }
}

/// Tests 3 and 4 in one order: random leading monomials entered one at a
/// time into the engine and the reference, on `variables(world)`
/// variables in each of 60 worlds.
pub(crate) fn check_updates_on<O: Order<Dense16>>(
    order: &O,
    seed: u64,
    variables: impl Fn(u64) -> usize,
) {
    let mut rng = rng(seed);
    for world in 0..60 {
        let n = variables(world);
        let mut engine: Engine<Dense16> = Engine::new(n, seed ^ world);
        let mut reference = Reference::new(O::GRADED, Dense16::MAX_DEGREE);
        let elements = 10 + below(&mut rng, 30) as usize;
        for e in 0..elements {
            // Small degrees on few variables: many coprime pairs and many
            // equal lcms.
            let most = 1 + below(&mut rng, 5) as u32;
            let lead = random_lanes(&mut rng, n, 3, Some(most));
            let sugar = degree(&lead) + below(&mut rng, 4) as u32;
            let t = ElementId::at(engine.basis.len());
            let (got, gained) = engine.enter(order, &lead, sugar);
            got.unwrap();
            reference.update(lead.clone(), sugar).unwrap();
            assert_eq!(
                engine.written(),
                reference.pairs,
                "world {world}, element {e}, lead {lead:?}"
            );
            let min: Vec<u32> = engine
                .min_members()
                .iter()
                .map(|i| i.index() as u32)
                .collect();
            assert_eq!(
                min,
                reference.min_members(),
                "world {world}, element {e}, lead {lead:?}: G_min"
            );
            // Test 4: every monomial the update interned is a kept pair's
            // lcm; since pair-update-rest-time.1 the update interns none,
            // the kept pairs' lcms going into the set as words.
            assert!(
                gained.is_empty(),
                "world {world}, element {e}: the update interned {monomials:?}",
                monomials = gained
                    .clone()
                    .map(|id| exponents(&engine.table, MonoId::from_raw(id)))
                    .collect::<Vec<_>>()
            );
            assert!(
                engine
                    .pairs
                    .iter()
                    .all(|p| p.j != t || p.lcm == MonoId::NONE),
                "world {world}, element {e}: a kept pair's lcm is interned"
            );
        }
    }
}

/// Tests 3 and 4: divisibility-time.1 to .3, in a graded order and in one
/// that is not.
#[test]
fn the_update_keeps_the_pairs_of_algorithm_2() {
    let few = |world: u64| 2 + (world % 5) as usize;
    check_updates_on(&Grevlex, 0xd1f1_0005, few);
    check_updates_on(&Lex, 0xd1f1_0006, few);
}

/// Test 5: divisibility-time.1, .3, by hand on x, y, z.
#[test]
fn the_update_by_hand() {
    // An element entered into an empty G_min, then one coprime to it.
    let mut e: Engine<Dense16> = Engine::new(3, 1);
    e.enter(&Grevlex, &[1, 0, 0], 1).0.unwrap();
    assert!(e.pairs.is_empty(), "an empty G_min makes no pair");
    e.enter(&Grevlex, &[0, 1, 0], 1).0.unwrap();
    assert!(e.pairs.is_empty(), "a coprime pair is dropped");

    // Equal lcms: x²y, x²z, then yz; the later pair stands.
    let mut e: Engine<Dense16> = Engine::new(3, 2);
    e.enter(&Grevlex, &[2, 1, 0], 3).0.unwrap();
    e.enter(&Grevlex, &[2, 0, 1], 3).0.unwrap();
    e.enter(&Grevlex, &[0, 1, 1], 2).0.unwrap();
    let want = vec![
        RefPair {
            i: 0,
            j: 1,
            lcm: vec![2, 1, 1],
            degree: 4,
            sugar: 4,
        },
        RefPair {
            i: 1,
            j: 2,
            lcm: vec![2, 1, 1],
            degree: 4,
            sugar: 4,
        },
    ];
    assert_eq!(e.written(), want, "of two equal lcms the later pair stands");

    // A proper divisor: xyz, y²z, then x²y; x²yz divides x²y²z.
    let mut e: Engine<Dense16> = Engine::new(3, 3);
    e.enter(&Grevlex, &[1, 1, 1], 3).0.unwrap();
    e.enter(&Grevlex, &[0, 2, 1], 3).0.unwrap();
    e.enter(&Grevlex, &[2, 1, 0], 3).0.unwrap();
    let want = vec![
        RefPair {
            i: 0,
            j: 1,
            lcm: vec![1, 2, 1],
            degree: 4,
            sugar: 4,
        },
        RefPair {
            i: 0,
            j: 2,
            lcm: vec![2, 1, 1],
            degree: 4,
            sugar: 4,
        },
    ];
    assert_eq!(
        e.written(),
        want,
        "a pair whose lcm a smaller one divides goes"
    );
}

/// Test 6: divisibility-time.1. In `Dense8`, whose degree bound is 127, a
/// coprime pair past the bound is dropped. A pair of P1 past it waits for
/// its selection (gf2-monomial-time.3; `gf2_monomial_time_tests.rs`, test
/// 9).
#[test]
fn an_lcm_past_the_format() {
    let mut e: Engine<Dense8> = Engine::new(2, 5);
    e.enter(&Grevlex, &[100, 0], 100).0.unwrap();
    let (got, _) = e.enter(&Grevlex, &[0, 120], 120);
    assert_eq!(got, Ok(()), "x^100 and y^120: coprime");
    assert!(e.pairs.is_empty(), "the coprime pair is dropped");
}
