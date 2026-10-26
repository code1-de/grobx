//! gfp-unsolved, sub-item 1: the plain run, the engine with signatures off,
//! through the public API: the edge cases and random systems against the
//! signature run, the
//! checker on its cofactors, one outcome per seed, its record, and the
//! delayed subtraction (plan.md, _Test code_).

use grobx::field::{Field, Gf31};
use grobx::id::MonoId;
use grobx::trace::NoTrace;
use grobx::{Outcome, Params};
use grobx_checker::Mode;

use crate::common::{
    P, P31, Rng, assert_accepted, dense, params, random_square, random_system, run, with_cofactors,
};

/// The parameters of a call with signatures off.
fn plain(seed: u64) -> Params {
    let mut p = params(seed);
    p.signatures = false;
    p
}

/// `plain` with cofactors.
fn plain_with_cofactors(seed: u64) -> Params {
    let mut p = with_cofactors(seed);
    p.signatures = false;
    p
}

/// The instance `text` with its characteristic replaced by p.
fn over(text: &str, p: u64) -> String {
    let mut lines = text.splitn(3, '\n');
    let names = lines.next().unwrap();
    let _ = lines.next();
    let rest = lines.next().unwrap_or("");
    format!("{names}\n{p}\n{rest}")
}

/// The edge cases of `OBJECTIVES.md`, _Domain specifics the workflow points at_, over 65521.
const EDGE: &[(&str, &str)] = &[
    ("zero-ideal", "x,y\n65521\n0,\n0\n"),
    ("f-is-zero", "x,y\n65521\n0\n"),
    ("unit", "x,y\n65521\nx*y-1,\nx\n"),
    (
        "unit-from-a-round",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
    ),
    ("already-a-basis", "x,y\n65521\nx+y,\ny^2+1\n"),
    ("zero-generator", "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n"),
    ("eliminable", "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n"),
];

/// The primes of the edge cases: 7 reduces each row on its own, 65521 and
/// 2147483647 combine (gfp-unsolved.5).
const PRIMES: [u64; 3] = [7, P, P31];

#[test]
fn the_edge_cases_without_signatures() {
    for &(name, text) in EDGE {
        for p in PRIMES {
            let text = over(text, p);
            let outcome = run(&text, &plain_with_cofactors(1));
            assert_eq!(
                outcome.basis,
                run(&text, &params(1)).basis,
                "{name} over {p}: the signature run's basis"
            );
            assert_eq!(
                outcome.basis,
                run(&text, &plain(1)).basis,
                "{name} over {p}: the plain run without cofactors"
            );
            assert_accepted(&format!("{name}-p{p}"), &text, &outcome, Mode::Exact);
        }
    }
}

#[test]
fn random_systems_without_signatures() {
    for p in [P, P31] {
        for seed in 0..30 {
            let text = random_system(p, seed);
            let outcome = run(&text, &plain_with_cofactors(seed));
            assert_eq!(
                outcome.basis,
                run(&text, &params(seed)).basis,
                "random-p{p}-{seed}: the signature run's basis\n{text}"
            );
            assert_accepted(&format!("random-p{p}-{seed}"), &text, &outcome, Mode::Exact);
        }
    }
}

/// Generated systems over 65521: name and text.
fn systems() -> [(&'static str, String); 3] {
    [
        ("cubic-4", dense(4, 3, P, 1)),
        ("cubic-3", dense(3, 3, P, 2)),
        ("square", random_square(P, 3)),
    ]
}

/// gfp-unsolved.7: the seed decides the combinations. One seed gives one
/// outcome, its counts included; two seeds give one basis.
#[test]
fn one_seed_one_outcome() {
    for (stem, text) in systems() {
        let first: Outcome = run(&text, &plain(7));
        assert_eq!(first, run(&text, &plain(7)), "{stem}");
        assert_eq!(first.basis, run(&text, &plain(8)).basis, "{stem}");
        let with = run(&text, &plain_with_cofactors(7));
        assert_eq!(with, run(&text, &plain_with_cofactors(7)), "{stem}");
    }
}

/// gfp-unsolved.11: a record of the plain run in grevlex replays over its
/// prime and over another.
#[test]
fn a_plain_record_replays() {
    for (stem, text) in systems() {
        let (outcome, record) = grobx::learn(&text, &plain(1), &mut NoTrace).unwrap();
        let record = record.expect("a record");
        let again = grobx::replay(&text, &plain(2), &record, &mut NoTrace).unwrap();
        assert_eq!(again.basis, outcome.basis, "{stem}");
        let other = over(&text, P31);
        let replayed = grobx::replay(&other, &plain(2), &record, &mut NoTrace).unwrap();
        assert_eq!(
            replayed.basis,
            run(&other, &plain(1)).basis,
            "{stem} over {P31}"
        );
    }
}

/// gfp-unsolved.10: below 2^16 the dense row's entries take every
/// subtraction unreduced, and `take` returns the residue; at and above it
/// they are corrected as before. Either way the residues are those computed
/// in `u128`.
#[test]
fn the_delayed_subtraction() {
    for p in [65521, 32749, 65537, 2_147_483_647] {
        let field = Gf31::new(p).unwrap();
        let width = 64;
        let mut rng = Rng::new(p ^ 0xde1a);
        let mut dense = vec![<Gf31 as Field>::Acc::default(); width];
        let mut want = vec![0u128; width];
        let start: Vec<MonoId> = (0..width as u32).map(MonoId::from_raw).collect();
        let values: Vec<u32> = (0..width).map(|_| rng.below(p) as u32).collect();
        // SAFETY: the columns are 0 to width − 1, the dense row's entries,
        // one value for each.
        unsafe { field.load(&mut dense, &start, &values) };
        for (w, &v) in want.iter_mut().zip(&values) {
            *w = u128::from(v);
        }
        let p128 = u128::from(p);
        for round in 0..200_000 {
            // Every hundredth subtraction falls on column 0 alone, so that
            // one entry takes 2000 more than the others.
            let cols: Vec<MonoId> = if round % 100 == 0 {
                vec![MonoId::from_raw(0)]
            } else {
                let mut c: Vec<MonoId> = (0..=rng.below(8))
                    .map(|_| MonoId::from_raw(rng.below(width as u64) as u32))
                    .collect();
                c.sort_unstable();
                c.dedup();
                c
            };
            let coefs: Vec<u32> = cols.iter().map(|_| rng.below(p) as u32).collect();
            let c = rng.below(p) as u32;
            // SAFETY: every column is drawn below `width`, one coefficient
            // for each.
            unsafe { field.sub_mul(&mut dense, &cols, &coefs, c) };
            for (&col, &v) in cols.iter().zip(&coefs) {
                let product = u128::from(c) * u128::from(v) % p128;
                want[col.index()] = (want[col.index()] + p128 - product) % p128;
            }
        }
        for col in 0..width {
            let got = field.take(&mut dense[col]);
            assert_eq!(u128::from(got), want[col], "p {p}, column {col}");
            assert_eq!(
                field.take(&mut dense[col]),
                0,
                "p {p}, column {col} after take"
            );
        }
    }
}
