//! trace-replay, sub-item 1: the record `learn` returns, its variables,
//! inputs, order of entry, and steps, read against the instance as parsed
//! and the basis as returned.

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;
use std::num::NonZeroUsize;

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::field::Gf31;
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::Poly;
use crate::systems::{dense, random_system};
use crate::table::Table;
use crate::text::Header;
use crate::trace::{NoTrace, Record};
use crate::{MonomialOrder, Outcome, Params, Threads};

/// The primes of the random systems.
const PRIMES: [u64; 5] = [2, 3, 7, 65521, 2_147_483_647];

/// The edge cases of plan.md.
const EDGE: &[&str] = &[
    "x,y\n65521\n0\n",
    "x,y\n65521\n",
    "x,y\n65521\nx*y-1,\nx\n",
    "x,y\n65521\n1\n",
    "x,y\n65521\nx^2+y,\nx*y+1,\n7\n",
    "x,y\n65521\nx+y,\ny^2+1\n",
    "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n",
    "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
];

/// A tiny instance, dense cubic systems, the edge cases, and random
/// systems.
fn instances() -> Vec<(String, String)> {
    let mut list = vec![(
        "tiny".to_string(),
        "x,y\n65521\nx^2+y,\nx*y+1\n".to_string(),
    )];
    for m in 2..=3 {
        list.push((format!("cubic-{m}"), dense(m, 3, 65521, 1)));
    }
    for (k, text) in EDGE.iter().enumerate() {
        list.push((format!("edge-{k}"), text.to_string()));
    }
    for p in PRIMES {
        for seed in 0..2 {
            list.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
    }
    list
}

fn params(signatures: bool) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    p.signatures = signatures;
    p
}

/// The outcome and the record of a learn run that returns a basis.
fn learned(name: &str, text: &str, signatures: bool) -> (Outcome, Record) {
    // The generators as parsed: the echelon off (input-redundancy-time.8).
    let mut p = params(signatures);
    p.echelon = false;
    let (outcome, record) =
        crate::learn(text, &p, &mut NoTrace).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert!(outcome.basis.is_some(), "{name}: no basis");
    (
        outcome,
        record.unwrap_or_else(|| panic!("{name}: no record")),
    )
}

/// Per generator as parsed, the exponents of its leading monomial, None
/// where it is zero.
fn generator_leads(text: &str) -> Vec<Option<Vec<u32>>> {
    let header = Header::read(text).unwrap();
    let field = header.gf31().unwrap();
    let n = header.names.len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    let mut table = Table::new(Dense16::new(n), &mut rng);
    let generators: Vec<Poly<Gf31, Rank1>> =
        header.generators(&field, &mut table, &Grevlex).unwrap();
    generators
        .iter()
        .map(|g| (!g.is_zero()).then(|| (0..n).map(|i| table.exponent(g.lead(), i)).collect()))
        .collect()
}

/// Grevlex on exponent vectors.
fn grevlex(a: &[u32], b: &[u32]) -> Ordering {
    let (da, db): (u32, u32) = (a.iter().sum(), b.iter().sum());
    da.cmp(&db).then_with(|| {
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return b[i].cmp(&a[i]);
            }
        }
        Ordering::Equal
    })
}

fn divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

fn mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// The n exponents at position k of a flat vector.
fn at(flat: &[u32], n: usize, k: usize) -> &[u32] {
    &flat[k * n..(k + 1) * n]
}

/// Every element's leading exponents by its index in the basis: the
/// inputs in their order of entry, then each step's new elements in theirs.
fn element_leads(record: &Record) -> Vec<Vec<u32>> {
    let n = record.variables;
    let mut leads = Vec::new();
    for &j in &record.entered {
        leads.push(
            record.inputs[j as usize]
                .clone()
                .expect("an input that entered is nonzero"),
        );
    }
    for step in &record.steps {
        for &k in &step.entered {
            leads.push(at(&step.leads, n, k as usize).to_vec());
        }
    }
    leads
}

/// The leading exponents of each polynomial of a basis text over GF(p),
/// read from its first term.
fn basis_leads(text: &str, basis: &str) -> Vec<Vec<u32>> {
    let names: Vec<&str> = text
        .lines()
        .next()
        .unwrap()
        .split(',')
        .map(str::trim)
        .collect();
    let body = basis
        .strip_prefix('[')
        .unwrap()
        .strip_suffix("]:\n")
        .unwrap();
    if body.is_empty() {
        return Vec::new();
    }
    body.split(",\n")
        .map(|poly| {
            let end = poly[1..].find(['+', '-']).map_or(poly.len(), |i| i + 1);
            let term = &poly[..end];
            let mut e = vec![0; names.len()];
            if term != "1" {
                for factor in term.split('*') {
                    let (name, power) = factor
                        .split_once('^')
                        .map_or((factor, 1), |(v, p)| (v, p.parse().unwrap()));
                    let i = names.iter().position(|v| *v == name).unwrap();
                    e[i] += power;
                }
            }
            e
        })
        .collect()
}

#[test]
fn the_inputs_are_recorded() {
    for (name, text) in instances() {
        let leads = generator_leads(&text);
        for signatures in [false, true] {
            let (_, record) = learned(&name, &text, signatures);
            let n = text.lines().next().unwrap().split(',').count();
            assert_eq!(record.variables, n, "{name}");
            assert_eq!(record.inputs, leads, "{name}: the generators' leads");
            let want: Vec<u32> = (0..leads.len() as u32)
                .filter(|&j| leads[j as usize].is_some())
                .collect();
            let mut entered = record.entered.clone();
            entered.sort_unstable();
            assert_eq!(
                entered, want,
                "{name}, signatures {signatures}: the nonzero inputs enter"
            );
        }
    }
}

#[test]
fn every_step_is_whole() {
    for (name, text) in instances() {
        for signatures in [false, true] {
            let (_, record) = learned(&name, &text, signatures);
            let name = format!("{name}, signatures {signatures}");
            let n = record.variables;
            let mut made = record.entered.len();
            for (s, step) in record.steps.iter().enumerate() {
                let rows = step.elements.len();
                assert!(!step.reductees.is_empty(), "{name}: step {s} counts no row");
                assert_eq!(step.multipliers.len(), n * rows, "{name}: step {s}");
                assert_eq!(
                    step.leads.len(),
                    n * step.reductees.len(),
                    "{name}: step {s}"
                );
                let mut seen = vec![false; rows];
                for &r in &step.reductees {
                    assert!(r.index() < rows, "{name}: step {s}, row {r} of {rows}");
                    assert!(!seen[r.index()], "{name}: step {s}, row {r} twice");
                    seen[r.index()] = true;
                }
                let mut entered = step.entered.clone();
                entered.sort_unstable();
                assert!(
                    entered.iter().copied().eq(0..step.reductees.len() as u32),
                    "{name}: step {s}, order of entry {entered:?}",
                    entered = step.entered
                );
                for &g in &step.elements {
                    assert!(
                        g.index() < made,
                        "{name}: step {s} names element {g} of {made}"
                    );
                }
                made += step.reductees.len();
            }
        }
    }
}

#[test]
fn the_leading_monomials_agree() {
    for (name, text) in instances() {
        for signatures in [false, true] {
            let (_, record) = learned(&name, &text, signatures);
            let name = format!("{name}, signatures {signatures}");
            let n = record.variables;
            let leads = element_leads(&record);
            for (s, step) in record.steps.iter().enumerate() {
                let row_lead = |r: usize| {
                    mul(
                        at(&step.multipliers, n, r),
                        &leads[step.elements[r].index()],
                    )
                };
                let mut reductee = vec![false; step.elements.len()];
                for &r in &step.reductees {
                    reductee[r.index()] = true;
                }
                let pivots: Vec<Vec<u32>> = (0..step.elements.len())
                    .filter(|&r| !reductee[r])
                    .map(row_lead)
                    .collect();
                let mut distinct = pivots.clone();
                distinct.sort();
                distinct.dedup();
                assert_eq!(
                    distinct.len(),
                    pivots.len(),
                    "{name}: step {s}: two pivot rows share a leading monomial"
                );
                let mut results: Vec<Vec<u32>> = Vec::new();
                for (k, &r) in step.reductees.iter().enumerate() {
                    let lead = row_lead(r.index());
                    let result = at(&step.leads, n, k).to_vec();
                    let hits = pivots.iter().filter(|p| **p == lead).count()
                        + results.iter().filter(|q| **q == lead).count();
                    assert_eq!(
                        hits, 1,
                        "{name}: step {s}, row {r}: {hits} pivots for its leading monomial"
                    );
                    assert_eq!(
                        grevlex(&result, &lead),
                        Ordering::Less,
                        "{name}: step {s}, row {r} reduced to {result:?} from {lead:?}"
                    );
                    assert!(
                        !pivots.contains(&result),
                        "{name}: step {s}, row {r} reduced to a pivot's monomial"
                    );
                    assert!(
                        !results.contains(&result),
                        "{name}: step {s}, row {r} reduced to an earlier result's monomial"
                    );
                    results.push(result);
                }
            }
        }
    }
}

#[test]
fn the_record_names_the_basis() {
    for (name, text) in instances() {
        for signatures in [false, true] {
            let (outcome, record) = learned(&name, &text, signatures);
            let mut min: Vec<Vec<u32>> = Vec::new();
            for lead in element_leads(&record) {
                if min.iter().any(|m| divides(m, &lead)) {
                    continue;
                }
                min.retain(|m| !divides(&lead, m));
                min.push(lead);
            }
            min.sort_by(|a, b| grevlex(a, b));
            let basis = outcome.basis.as_deref().unwrap();
            assert_eq!(
                min,
                basis_leads(&text, basis),
                "{name}, signatures {signatures}: G_min of the record\n{basis}"
            );
        }
    }
}
