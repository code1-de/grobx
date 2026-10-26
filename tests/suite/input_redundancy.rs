//! input-redundancy-time, the item's test code (input-redundancy-time.1 to
//! .6, .8, .9): the edge cases of `OBJECTIVES.md`, and systems with linear
//! redundancy among their generators, over GF(p) and ℚ, in grevlex and
//! outside it, with the call report's `echelon` line
//! (`iterations/0068-input-redundancy-time/plan.md`, _Test code_, 6 to 9).

use std::collections::BTreeSet;
use std::fmt::Write;

use grobx::{Outcome, Params};
use grobx_checker::Mode;

use crate::common::{P, P31, Rng, assert_accepted, params, run, with_cofactors};
use crate::orders::{accepted_in, assert_canonical};
use crate::q::{accepted_q, assert_canonical_q};
use crate::report::{Line, of, report as lines_of, untimed};

/// The edge cases of `OBJECTIVES.md` on x, y, z: name, generators, and the
/// basis over GF(65521) in grevlex.
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

/// The bound of a drawn coefficient over ℚ.
const Q_BOUND: u64 = 1 << 20;

/// A generator: its terms, each an exponent vector and a coefficient.
pub(crate) type Terms = Vec<(Vec<u32>, i64)>;

/// A system with linear redundancy, over GF(p), or over ℚ where p is 0, on
/// n of the variables x1, x2, …, n 3 or 4 by seed: two linear forms L0 and
/// L1 and n − 2 quadratics, each with every monomial up to its degree, so
/// the ideal is zero-dimensional on most seeds; then L0 plus the first
/// quadratic, which shares its leading monomial; 3·L1; 2·L0 + 5·L1, a third
/// form led by x1 and dependent; `x_n·L0 + L1`, which holds x1 in its tail;
/// and L0 again. The ideal is that of L0, L1, and the quadratics. Each
/// generator is the list of its terms, an exponent vector and a
/// coefficient: a drawn one below p, or below `Q_BOUND` with a sign over
/// ℚ, where the derived generators' coefficients over ℚ are their sums
/// and multiples.
pub(crate) fn redundant(p: u64, seed: u64) -> (Vec<String>, Vec<Terms>) {
    let mut rng = Rng::new(seed ^ p.rotate_left(29) ^ 0x1d0e);
    let n = 3 + (seed % 2) as usize;
    let modulus = i64::try_from(p).unwrap();
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    let mut coef = || -> i64 {
        if p == 0 {
            let c = 1 + i64::try_from(rng.below(Q_BOUND - 1)).unwrap();
            if rng.below(2) == 1 { -c } else { c }
        } else {
            1 + i64::try_from(rng.below(p - 1)).unwrap()
        }
    };
    // Every exponent vector on n variables of degree at most d.
    let exponents = |d: u32| -> Vec<Vec<u32>> {
        let mut all: Vec<Vec<u32>> = vec![Vec::new()];
        for _ in 0..n {
            all = all
                .into_iter()
                .flat_map(|e| {
                    let used: u32 = e.iter().sum();
                    (0..=d - used).map(move |k| {
                        let mut f = e.clone();
                        f.push(k);
                        f
                    })
                })
                .collect();
        }
        all
    };
    let mut dense = |d: u32| -> Terms { exponents(d).into_iter().map(|e| (e, coef())).collect() };
    let mut gens = vec![dense(1), dense(1)];
    for _ in 0..n - 2 {
        gens.push(dense(2));
    }
    // a + s·b, the terms that cancel left out.
    let add = |a: &[(Vec<u32>, i64)], b: &[(Vec<u32>, i64)], s: i64| {
        let mut sum: Terms = a.to_vec();
        for (e, c) in b {
            let c = if p == 0 {
                c * s
            } else {
                (c * s).rem_euclid(modulus)
            };
            match sum.iter_mut().find(|(f, _)| f == e) {
                Some(t) => {
                    t.1 = if p == 0 {
                        t.1 + c
                    } else {
                        (t.1 + c).rem_euclid(modulus)
                    };
                }
                None => sum.push((e.clone(), c)),
            }
        }
        sum.retain(|(_, c)| *c != 0);
        sum
    };
    let (l0, l1, q0) = (gens[0].clone(), gens[1].clone(), gens[2].clone());
    gens.push(add(&l0, &q0, 1));
    gens.push(add(&[], &l1, 3));
    gens.push(add(&add(&[], &l0, 2), &l1, 5));
    let shifted: Terms = l0
        .iter()
        .map(|(e, c)| {
            let mut f = e.clone();
            f[n - 1] += 1;
            (f, *c)
        })
        .collect();
    gens.push(add(&shifted, &l1, 1));
    gens.push(l0);
    gens.retain(|g| !g.is_empty());
    (names, gens)
}

/// The instance text of `redundant`'s system.
pub(crate) fn text(p: u64, names: &[String], gens: &[Terms]) -> String {
    let mut out = format!("{variables}\n{p}\n", variables = names.join(","));
    for (k, g) in gens.iter().enumerate() {
        if k > 0 {
            out.push_str(",\n");
        }
        let mut poly = String::new();
        for (e, c) in g {
            write!(
                poly,
                "{sign}{magnitude}",
                sign = if *c < 0 { "-" } else { "+" },
                magnitude = c.unsigned_abs()
            )
            .unwrap();
            for (name, &k) in names.iter().zip(e) {
                match k {
                    0 => {}
                    1 => write!(poly, "*{name}").unwrap(),
                    k => write!(poly, "*{name}^{k}").unwrap(),
                }
            }
        }
        out.push_str(poly.trim_start_matches('+'));
    }
    out.push('\n');
    out
}

/// The rank over GF(p) of the generators as vectors over their monomials,
/// by the test's own Gaussian elimination.
fn rank(p: u64, gens: &[Terms]) -> u64 {
    let monomials: BTreeSet<&Vec<u32>> = gens.iter().flatten().map(|(e, _)| e).collect();
    let index: Vec<&Vec<u32>> = monomials.into_iter().collect();
    let mut rows: Vec<Vec<u64>> = gens
        .iter()
        .map(|g| {
            let mut v = vec![0u64; index.len()];
            for (e, c) in g {
                let k = index.binary_search(&e).unwrap();
                v[k] = u64::try_from(c.rem_euclid(i64::try_from(p).unwrap())).unwrap();
            }
            v
        })
        .collect();
    let mul = |a: u64, b: u64| (u128::from(a) * u128::from(b) % u128::from(p)) as u64;
    let inv = |a: u64| {
        let (mut b, mut e, mut r) = (a, p - 2, 1u64);
        while e > 0 {
            if e & 1 == 1 {
                r = mul(r, b);
            }
            b = mul(b, b);
            e >>= 1;
        }
        r
    };
    let mut r = 0;
    for c in 0..index.len() {
        let Some(k) = (r..rows.len()).find(|&k| rows[k][c] != 0) else {
            continue;
        };
        rows.swap(r, k);
        let s = inv(rows[r][c]);
        for v in &mut rows[r] {
            *v = mul(*v, s);
        }
        let pivot = rows[r].clone();
        for (k, row) in rows.iter_mut().enumerate() {
            if k != r && row[c] != 0 {
                let a = row[c];
                for (v, &w) in row.iter_mut().zip(&pivot) {
                    *v = (*v + p - mul(a, w)) % p;
                }
            }
        }
        r += 1;
    }
    r as u64
}

/// The distinct monomials of the generators.
fn columns(gens: &[Terms]) -> u64 {
    gens.iter()
        .flatten()
        .map(|(e, _)| e)
        .collect::<BTreeSet<_>>()
        .len() as u64
}

/// `params` with the report on.
fn reported(params: Params) -> Params {
    let mut params = params;
    params.report = true;
    params
}

/// The `echelon` lines of the call's report, each checked: right after its
/// run's `inputs` line, with `generators` the count that line read,
/// `rows` at most its `nonzero`, and `subtracted` and `time` read. Every
/// run of the
/// call has one.
fn echelon_lines(case: &str, outcome: &Outcome) -> Vec<Line> {
    let lines = lines_of(outcome);
    let mut found = Vec::new();
    for (k, line) in lines.iter().enumerate() {
        if line.kind == "inputs" {
            let next = &lines[k + 1];
            assert_eq!(next.kind, "echelon", "{case}: {next:?}");
            assert_eq!(
                next.num("generators"),
                line.num("generators"),
                "{case}: {next:?}"
            );
            assert!(next.num("rows") <= line.num("nonzero"), "{case}: {next:?}");
            next.num("subtracted");
            next.secs("time");
            found.push(next.clone());
        }
    }
    assert_eq!(
        found.len(),
        of(&lines, "run").len(),
        "{case}: one line per run"
    );
    found
}

/// Test 6, input-redundancy-time.1, .8: the edge cases over 65521, 7, and
/// 2^31 − 1 in grevlex, signatures on and off, with the report: each basis
/// canonical, as written over 65521, and the one with cofactors, whose
/// generators stay as read, which the checker accepts in the exact mode;
/// in lex and `block:1` and over ℚ, the same.
#[test]
fn the_edge_cases() {
    for (case, generators, want) in EDGE {
        for p in [P, 7, P31] {
            let text = format!("x,y,z\n{p}\n{generators}\n");
            for signatures in [true, false] {
                let mut plain = reported(params(0x1d0e));
                plain.signatures = signatures;
                let outcome = run(&text, &plain);
                let name = format!("{case} over {p}, signatures {signatures}");
                echelon_lines(&name, &outcome);
                let basis = outcome.basis.unwrap();
                let mut with = with_cofactors(0x1d0e);
                with.signatures = signatures;
                let outcome = run(&text, &with);
                assert_eq!(outcome.basis.as_deref(), Some(basis.as_str()), "{name}");
                assert_canonical(&name, &text, &basis, "grevlex");
                assert_accepted(&name, &text, &outcome, Mode::Exact);
                if p == P {
                    assert_eq!(basis, want, "{name}");
                }
            }
            for order in ["lex", "block:1"] {
                let name = format!("{case} over {p} in {order}");
                let with = accepted_in(&name, &text, order, Mode::Exact);
                let mut plain = reported(params(0x1d0e));
                plain.order = order.parse().unwrap();
                let outcome = run(&text, &plain);
                echelon_lines(&name, &outcome);
                assert_eq!(outcome.basis, with.basis, "{name}");
            }
        }
        let text = format!("x,y,z\n0\n{generators}\n");
        let name = format!("{case} over ℚ");
        let with = accepted_q(&name, &text, "grevlex");
        let outcome = run(&text, &reported(params(0x1d0e)));
        echelon_lines(&name, &outcome);
        assert_eq!(outcome.basis, with.basis, "{name}");
    }
}

/// Test 7, input-redundancy-time.1 to .4: the redundant systems over 65521,
/// 7, 2^31 − 1, and ℚ, signatures on and off, the echelon on and off, 8
/// seeds each: each basis canonical and the one with cofactors, which the
/// checker accepts in the exact mode.
#[test]
fn redundant_systems() {
    for p in [P, 7, P31, 0] {
        for seed in 0..8 {
            let (names, gens) = redundant(p, seed);
            let instance = text(p, &names, &gens);
            let case = format!("redundant over {p}, seed {seed}");
            let with = run(&instance, &with_cofactors(seed));
            let basis = with.basis.clone().unwrap();
            if p == 0 {
                assert_canonical_q(&case, &instance, &basis, "grevlex");
                assert_eq!(accepted_q(&case, &instance, "grevlex").basis, with.basis);
            } else {
                assert_canonical(&case, &instance, &basis, "grevlex");
                assert_accepted(&case, &instance, &with, Mode::Exact);
            }
            for signatures in [true, false] {
                for echelon in [true, false] {
                    let mut plain = params(seed);
                    plain.signatures = signatures;
                    plain.echelon = echelon;
                    let outcome = run(&instance, &plain);
                    assert_eq!(
                        outcome.basis.as_deref(),
                        Some(basis.as_str()),
                        "{case}, signatures {signatures}, echelon {echelon}"
                    );
                }
            }
        }
    }
}

/// Test 8, input-redundancy-time.9, .8: the report's `echelon` lines. Over
/// GF(p) each holds the generators' rank as `rows` and their distinct
/// monomials as `columns`; over ℚ every run's line holds the same `rows`
/// and `columns`, the learn's, the replays', and the verification's where
/// it runs at a prime of its own; a
/// call with cofactors, with the checker, or with the echelon off has
/// none over GF(p), and over ℚ a call with cofactors has none; the call
/// line names the parameter.
#[test]
fn the_echelon_line() {
    for p in [P, 7, P31] {
        for seed in 0..8 {
            let (names, gens) = redundant(p, seed);
            let instance = text(p, &names, &gens);
            let case = format!("redundant over {p}, seed {seed}");
            let outcome = run(&instance, &reported(params(seed)));
            for line in echelon_lines(&case, &outcome) {
                assert_eq!(
                    line.num("generators"),
                    gens.len() as u64,
                    "{case}: {line:?}"
                );
                assert_eq!(line.num("rows"), rank(p, &gens), "{case}: {line:?}");
                assert_eq!(line.num("columns"), columns(&gens), "{case}: {line:?}");
            }
            let mut with = reported(with_cofactors(seed));
            with.signatures = seed % 2 == 0;
            let lines = lines_of(&run(&instance, &with));
            assert!(of(&lines, "echelon").is_empty(), "{case} with cofactors");
            let mut checked = reported(params(seed));
            checked.checker = Some(grobx::CheckerMode::Exact);
            let lines = lines_of(&run(&instance, &checked));
            assert!(of(&lines, "echelon").is_empty(), "{case} with the checker");
            let mut off = reported(params(seed));
            off.echelon = false;
            let lines = lines_of(&run(&instance, &off));
            assert!(
                of(&lines, "echelon").is_empty(),
                "{case} with the echelon off"
            );
            assert_eq!(
                lines[0].value("echelon"),
                "off",
                "{case}: {call:?}",
                call = lines[0]
            );
        }
    }
    for seed in 0..8 {
        let (names, gens) = redundant(0, seed);
        let instance = text(0, &names, &gens);
        let case = format!("redundant over ℚ, seed {seed}");
        let outcome = run(&instance, &reported(params(seed)));
        assert_eq!(lines_of(&outcome)[0].value("echelon"), "on", "{case}");
        let lines = echelon_lines(&case, &outcome);
        assert!(lines.len() >= 2, "{case}: a learn and a replay at least");
        let mut with = reported(with_cofactors(seed));
        with.signatures = seed % 2 == 0;
        let all = lines_of(&run(&instance, &with));
        assert!(of(&all, "echelon").is_empty(), "{case} with cofactors");
        for line in &lines {
            assert_eq!(
                line.num("generators"),
                gens.len() as u64,
                "{case}: {line:?}"
            );
            assert_eq!(
                (line.num("rows"), line.num("columns")),
                (lines[0].num("rows"), lines[0].num("columns")),
                "{case}: {line:?}"
            );
            assert_eq!(line.num("columns"), columns(&gens), "{case}: {line:?}");
        }
    }
}

/// Test 9: one seed, one path. Two calls under one seed write the same
/// report but for its times and resident sets, over 65521 and ℚ.
#[test]
fn one_seed_one_report() {
    for p in [P, 0] {
        for seed in 0..4 {
            let (names, gens) = redundant(p, seed);
            let instance = text(p, &names, &gens);
            let a = lines_of(&run(&instance, &reported(params(seed))));
            let b = lines_of(&run(&instance, &reported(params(seed))));
            assert_eq!(untimed(&a), untimed(&b), "over {p}, seed {seed}");
        }
    }
}
