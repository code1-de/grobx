//! gf2-update-time, sub-item 1's test code: the order in which the plain
//! run enters elements, by the field, with the learn's record in step; and
//! calls over GF(2) whose carried elements enter the plain run in that
//! order (`iterations/0130-gf2-update-time/plan.md`, _Test code_, 4 and 5).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small draws"
)]

use std::cmp::Ordering;

use crate::budget::Budget;
use crate::field::{Field, Gf2, Gf31};
use crate::gf2_start_over_tests::{call, reported};
use crate::id::MonoId;
use crate::module::Rank1;
use crate::monomial::{Dense8, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::pairs;
use crate::poly::Poly;
use crate::shared_tests::{below, rng};
use crate::systems::boolean;
use crate::table::Table;
use crate::trace;

/// The variables of test 4's monomials.
const N: usize = 5;

/// The monomials of test 4: 48 drawn under `seed`, each exponent below 3,
/// interned in `table`, so that some are drawn twice and some divide others.
fn drawn(table: &mut Table<Dense8>, seed: u64) -> Vec<MonoId> {
    let mut r = rng(seed);
    (0..48)
        .map(|_| {
            let e: Vec<u32> = (0..N).map(|_| below(&mut r, 3) as u32).collect();
            table.intern(&e, &mut Budget::unbounded()).unwrap()
        })
        .collect()
}

/// The monic monomials `ids` as polynomials over `field`.
fn monomials<F: Field>(field: &F, ids: &[MonoId]) -> Vec<Poly<F, Rank1>> {
    ids.iter()
        .map(|&m| Poly {
            monomials: vec![m],
            coefs: vec![field.coef(1)],
            position: (),
        })
        .collect()
}

/// Whether `entered`, places in `ids`, lists them as `want` orders two
/// monomials, ties in the places' order.
fn in_order<O: Order<Dense8>>(
    table: &Table<Dense8>,
    order: &O,
    ids: &[MonoId],
    entered: &[u32],
    want: Ordering,
) -> bool {
    entered.windows(2).all(|w| {
        let (a, b) = (ids[w[0] as usize], ids[w[1] as usize]);
        let got = table.cmp(order, a, b);
        if got.is_eq() {
            w[0] < w[1]
        } else {
            got == want
        }
    })
}

/// Test 4 in one order: `pairs::entering` over GF(2) is the order's
/// comparison and over GF(65521) its reverse, and `trace::inputs` gives
/// `entered` in that order over each field, ties in the generators' order.
fn the_order_in<O: Order<Dense8>>(name: &str, order: &O, seed: u64) {
    let mut table = Table::new(Dense8::new(N), &mut rng(seed));
    let ids = drawn(&mut table, seed);
    for &a in &ids {
        for &b in &ids {
            assert_eq!(
                pairs::entering::<Gf2, _, _>(&table, order, a, b),
                table.cmp(order, a, b),
                "{name}: over GF(2)"
            );
            assert_eq!(
                pairs::entering::<Gf31, _, _>(&table, order, a, b),
                table.cmp(order, b, a),
                "{name}: over GF(65521)"
            );
        }
    }
    let unbounded = &mut Budget::unbounded();
    let (_, _, entered, _) = trace::inputs(
        &table,
        order,
        &monomials(&Gf2::new(), &ids),
        true,
        true,
        unbounded,
    )
    .unwrap();
    assert_eq!(entered.len(), ids.len(), "{name}");
    assert!(
        in_order(&table, order, &ids, &entered, Ordering::Less),
        "{name}: the record over GF(2) {entered:?}"
    );
    let gf31 = Gf31::new(65521).unwrap();
    let (_, _, entered, _) = trace::inputs(
        &table,
        order,
        &monomials(&gf31, &ids),
        true,
        true,
        unbounded,
    )
    .unwrap();
    assert!(
        in_order(&table, order, &ids, &entered, Ordering::Greater),
        "{name}: the record over GF(65521) {entered:?}"
    );
}

/// Test 4: the entering order by the field in grevlex, lex, and
/// `block:2`, `enter`'s and the record's alike.
#[test]
fn the_entering_order() {
    the_order_in("grevlex", &Grevlex, 0xb170);
    the_order_in("lex", &Lex, 0xb171);
    the_order_in("block:2", &Block::new(&[2], N).unwrap(), 0xb172);
}

/// Test 5: calls over GF(2) on `boolean(n, m, seed)` for n = 6 to 13,
/// m = n + 2 and m = 2n, with `WORK_FORCED` at Some(0), so that the plain
/// run starts over in the signature run: one basis, that of the call with
/// signatures off; among them a call whose report holds a `carry` line, so
/// that carried elements entered the plain run, asserted present.
#[test]
fn the_carried_elements() {
    let mut carried = 0;
    for n in 6..=13usize {
        for m in [n + 2, 2 * n] {
            let seed = 0xb180 + n as u64;
            let text = boolean(n, m, seed);
            let name = format!("boolean({n}, {m}, {seed:#x})");
            let (want, _) = call(&text, &reported(seed, false), None);
            let (basis, lines) = call(&text, &reported(seed, true), Some(0));
            assert_eq!(basis, want, "{name}");
            if lines.iter().any(|l| l.kind == "carry") {
                carried += 1;
            }
        }
    }
    assert!(carried > 0, "no call carried after its start over");
}
