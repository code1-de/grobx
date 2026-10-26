//! reduction-time, the item's test code: the row operations that write the
//! dense row without a bounds check, `Gf31`'s four entries per turn
//! (reduction-time.1 to .5), against the same sums in `u128`; and calls
//! whose matrices they reduce, over primes on both sides of 2^16 and over
//! `Gf64`.

use std::num::NonZeroU32;

use grobx::field::{Field, Gf31, Gf64};
use grobx::id::MonoId;
use grobx_checker::Mode;

use crate::common::{
    P, P31, Rng, assert_accepted, degree_bound, dense, params, points, random_square, run,
    with_cofactors,
};
use crate::large::{LARGEST, LEAST, over};
use crate::orders::{accepted_in, assert_canonical};

/// The primes of `Gf31`: two below 2^16 besides 65521, and two above it.
const GF31: [u64; 5] = [2, 3, P, 65537, P31];
/// The largest prime of `Gf64`'s pre-inverse, 2^63 − 25.
const PRE_INVERSE: u64 = 9_223_372_036_854_775_783;

/// A coefficient of [0, p): p − 1 one time in four, else uniform.
fn draw(rng: &mut Rng, p: u64) -> u64 {
    if rng.below(4) == 0 {
        p - 1
    } else {
        rng.below(p)
    }
}

/// A sparse row over a dense row of `width` entries: 0 to 9 distinct
/// columns, increasing, ending at the last entry where `last` holds, and
/// their values.
fn row(rng: &mut Rng, width: usize, p: u64, last: bool) -> (Vec<MonoId>, Vec<u64>) {
    let k = rng.below(width.min(9) as u64 + 1) as usize;
    let mut all: Vec<u32> = (0..width as u32).collect();
    for i in 0..k {
        let j = i + rng.below((width - i) as u64) as usize;
        all.swap(i, j);
    }
    let mut cols = all[..k].to_vec();
    if last && k > 0 && !cols.contains(&(width as u32 - 1)) {
        cols[0] = width as u32 - 1;
    }
    cols.sort_unstable();
    let vals = cols.iter().map(|_| draw(rng, p)).collect();
    (cols.into_iter().map(MonoId::from_raw).collect(), vals)
}

/// Loads, subtracts, and takes on dense rows of 1 to 12 entries, against
/// the sums in `u128`; every entry reads 0 after its take.
pub(crate) fn agree<F: Field>(f: F, seed: u64) {
    let p = f.characteristic();
    let q = u128::from(p);
    let mut rng = Rng::new(seed ^ p);
    for width in 1..=12usize {
        for case in 0..30 {
            let mut dense = vec![F::Acc::default(); width];
            let mut want = vec![0u128; width];
            let (cols, vals) = row(&mut rng, width, p, case % 3 == 0);
            let coefs: Vec<F::Coef> = vals.iter().map(|&v| f.coef(v)).collect();
            // SAFETY: `row` draws its columns below `width`, the dense
            // row's length, one coefficient for each.
            unsafe { f.load(&mut dense, &cols, &coefs) };
            for (&c, &v) in cols.iter().zip(&vals) {
                want[c.index()] = u128::from(v);
            }
            for _ in 0..rng.below(200) {
                let last = rng.below(3) == 0;
                let (cols, vals) = row(&mut rng, width, p, last);
                let coefs: Vec<F::Coef> = vals.iter().map(|&v| f.coef(v)).collect();
                let m = draw(&mut rng, p);
                // SAFETY: as above.
                unsafe { f.sub_mul(&mut dense, &cols, &coefs, f.coef(m)) };
                for (&c, &v) in cols.iter().zip(&vals) {
                    let product = u128::from(m) * u128::from(v) % q;
                    want[c.index()] = (want[c.index()] + q - product) % q;
                }
            }
            for c in 0..width {
                assert_eq!(
                    u128::from(f.value(f.take(&mut dense[c]))),
                    want[c],
                    "GF({p}), width {width}, case {case}, column {c}"
                );
            }
            for entry in &mut dense {
                assert_eq!(f.value(f.take(entry)), 0, "GF({p}): take leaves 0");
            }
        }
    }
}

/// reduction-time.1, .2, .5: rows of every length of the unrolled loop's
/// tail, the dense row's last entry among their columns, and the largest
/// products, in both loops of `Gf31` and all three of `Gf64`.
#[test]
fn the_row_operations_agree_with_u128() {
    for p in GF31 {
        agree(Gf31::new(p).unwrap(), 0x5ed0);
    }
    for p in [LEAST, PRE_INVERSE, LARGEST] {
        agree(Gf64::new(p).unwrap(), 0x5ed0);
    }
}

/// reduction-time.5: 100,000 subtractions of (p − 1)·(p − 1) from every
/// entry of a row of four, five, and seven entries, below and above 2^16
/// (gfp-unsolved.10, f4-gfp.8).
#[test]
fn many_subtractions_from_one_entry() {
    const TIMES: u128 = 100_000;
    for p in [P, 65537, P31] {
        let f = Gf31::new(p).unwrap();
        let top = f.coef(p - 1);
        for len in [4usize, 5, 7] {
            let cols: Vec<MonoId> = (0..len as u32).map(MonoId::from_raw).collect();
            let coefs = vec![top; len];
            let mut dense = vec![<Gf31 as Field>::Acc::default(); len];
            // SAFETY: the columns are 0 to len − 1, the dense row's
            // entries, one coefficient for each.
            unsafe { f.load(&mut dense, &cols, &coefs) };
            for _ in 0..TIMES {
                // SAFETY: as above.
                unsafe { f.sub_mul(&mut dense, &cols, &coefs, top) };
            }
            // (p − 1) − TIMES·(p − 1)² ≡ −1 − TIMES mod p
            let q = u128::from(p);
            let want = (q - 1 + q - TIMES % q) % q;
            for (c, entry) in dense.iter_mut().enumerate() {
                assert_eq!(
                    u128::from(f.value(f.take(entry))),
                    want,
                    "GF({p}), {len} entries, column {c}"
                );
            }
        }
    }
}

/// The edge cases of `OBJECTIVES.md` on x, y, z: name, generators, and the
/// basis over GF(65521).
const EDGE: [(&str, &str, &str); 7] = [
    ("the zero ideal", "0", "[]:\n"),
    ("F = (0, 0)", "0,\n0", "[]:\n"),
    ("no generator", "", "[]:\n"),
    ("the unit ideal", "x*z-1,\nx", "[1]:\n"),
    ("already a Gröbner basis", "x+z,\nz^2+1", "[x+z,\nz^2+1]:\n"),
    (
        "the zero polynomial among the generators",
        "x*y-z,\n0,\ny^2-x",
        "[y^2+65520*x,\nx*y+65520*z,\nx^2+65520*y*z]:\n",
    ),
    (
        "an eliminable variable",
        "x-y^2-z,\ny*z-1,\nz^2-y",
        "[x+65519*z,\nz^2+65520*y,\ny*z+65520,\ny^2+65520*z]:\n",
    ),
];

/// The edge cases over 65521 and 65537, the two loops of `Gf31`, over
/// 2^31 − 1, and over `Gf64`'s least prime, signatures on and off: each
/// basis canonical and accepted by the checker in the exact mode, and over
/// 65521 as written.
#[test]
fn the_edge_cases_on_both_sides_of_2_to_the_16() {
    for p in [P, 65537, P31, LEAST] {
        for (case, generators, want) in EDGE {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut params = with_cofactors(0x0de7);
                params.signatures = signatures;
                let outcome = run(&text, &params);
                let basis = outcome.basis.as_deref().unwrap();
                let name = format!("{case} over {p}, signatures {signatures}");
                assert_canonical(&name, &text, basis, "grevlex");
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                if p == P {
                    assert_eq!(basis, want, "{name}");
                }
            }
        }
    }
}

/// Generated systems read over 65537, 2^31 − 1, and `Gf64`'s least prime:
/// the plain run without cofactors, which reduces random combinations of
/// rows (gfp-unsolved.2), returns the basis of the plain run with
/// cofactors, which reduces each row on its own (gfp-unsolved.8), and the
/// checker accepts the latter in its cheap mode.
#[test]
fn combinations_and_rows_agree_over_large_primes() {
    for (stem, system) in [
        ("cubic-3", dense(3, 3, P, 1)),
        ("square", random_square(P, 4)),
    ] {
        for p in [65537, P31, LEAST] {
            let text = over(&system, p);
            let name = format!("{stem} over {p}");
            let mut plain = params(11);
            plain.signatures = false;
            let combined = run(&text, &plain);
            let mut tracked = with_cofactors(11);
            tracked.signatures = false;
            let rows = run(&text, &tracked);
            assert_eq!(combined.basis, rows.basis, "{name}");
            let k = points(p, degree_bound(&text, &rows));
            let mode = Mode::Cheap {
                points: NonZeroU32::new(k).unwrap(),
                seed: 0x0de7 ^ p,
            };
            assert_accepted(&name, &text, &rows, mode);
        }
    }
}

/// Zero-dimensional dense cubic systems in lex over 65521 and 2^31 − 1: FGLM's
/// row operations change the grevlex basis, and the checker accepts the
/// lex basis with its cofactors in the exact mode.
#[test]
fn lex_through_fglm_over_both_fields() {
    for (stem, system) in [
        ("cubic-3", dense(3, 3, P, 1)),
        ("cubic-2", dense(2, 3, P, 2)),
    ] {
        for p in [P, P31] {
            let text = over(&system, p);
            accepted_in(&format!("{stem} over {p}"), &text, "lex", Mode::Exact);
        }
    }
}
