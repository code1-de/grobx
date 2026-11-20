//! deglex-order, the item's test code on the order itself: deglex on
//! exponent vectors in every format against its definition, and its name
//! (`iterations/0157-deglex-order/plan.md`, _Test code_, 1 and 2).

use std::cmp::Ordering;

use crate::Error;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::order::{Deglex, Grevlex, MonomialOrder, Order};

/// Every exponent vector on n variables of total degree at most d.
fn vectors(n: usize, d: u32) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|v: Vec<u32>| {
                let used: u32 = v.iter().sum();
                (0..=d - used).map(move |e| {
                    let mut w = v.clone();
                    w.push(e);
                    w
                })
            })
            .collect();
    }
    out
}

/// a against b in deglex by its definition: the larger total degree, then
/// the larger exponent at the first variable where they differ.
fn by_definition(a: &[u32], b: &[u32]) -> Ordering {
    let (da, db): (u32, u32) = (a.iter().sum(), b.iter().sum());
    da.cmp(&db).then_with(|| a.cmp(b))
}

/// a against b in `order` on the format of their length.
fn compared<M: Format, O: Order<M>>(order: &O, a: &[u32], b: &[u32]) -> Ordering {
    let format = M::new(a.len());
    let (mut x, mut y) = (
        vec![M::Word::default(); format.words()],
        vec![M::Word::default(); format.words()],
    );
    let dx = format.encode(a, &mut x).unwrap();
    let dy = format.encode(b, &mut y).unwrap();
    order.cmp(&format, &x, dx, &y, dy)
}

/// Deglex against its definition on every pair of vectors of degree at most
/// 3, on 1 to 4 variables and, spread over several words, on 17 and 33,
/// where each vector holds at most two nonzero exponents.
fn against_the_definition<M: Format>() {
    let mut sets: Vec<Vec<Vec<u32>>> = (1..=4).map(|n| vectors(n, 3)).collect();
    for n in [17, 33] {
        let mut set = vec![vec![0; n]];
        for i in 0..n {
            for j in i..n {
                for (ei, ej) in [(1, 0), (1, 1), (2, 1), (1, 2)] {
                    let mut v = vec![0; n];
                    v[i] += ei;
                    v[j] += ej;
                    set.push(v);
                }
            }
        }
        sets.push(set);
    }
    for set in &sets {
        for a in set {
            for b in set {
                assert_eq!(
                    compared::<M, _>(&Deglex, a, b),
                    by_definition(a, b),
                    "{a:?} {b:?}"
                );
            }
        }
    }
}

/// Test 1: deglex in every format is its definition; on three variables xz
/// is above y² in deglex and below it in grevlex.
#[test]
fn deglex_is_its_definition() {
    against_the_definition::<Dense4>();
    against_the_definition::<Dense8>();
    against_the_definition::<Dense16>();
    against_the_definition::<Dense32>();
    let (xz, yy) = ([1, 0, 1], [0, 2, 0]);
    assert_eq!(compared::<Dense16, _>(&Deglex, &xz, &yy), Ordering::Greater);
    assert_eq!(compared::<Dense16, _>(&Grevlex, &xz, &yy), Ordering::Less);
}

/// Test 2: `deglex` names the order and the order writes it; names near it
/// name no order.
#[test]
fn the_name_deglex() {
    assert_eq!(
        "deglex".parse::<MonomialOrder>().ok(),
        Some(MonomialOrder::Deglex)
    );
    assert_eq!(MonomialOrder::Deglex.to_string(), "deglex");
    for name in [
        "Deglex",
        "DEGLEX",
        "deglex ",
        " deglex",
        "deglex:1",
        "deg lex",
        "grlex",
        "degneglex",
    ] {
        assert!(
            matches!(name.parse::<MonomialOrder>(), Err(Error::Order { name: n }) if n == name),
            "{name:?}"
        );
    }
}
