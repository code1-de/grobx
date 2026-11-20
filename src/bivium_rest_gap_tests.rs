//! bivium-rest-gap, the item's test code: over GF(2) the rows of the
//! echelon that are the squares of linear rows less their tails' field
//! equations leave the run's generators, and the signature run records a
//! coprime S-pair's Koszul signature with no search and no drop, every
//! result as it was (`iterations/0181-bivium-rest-gap/plan.md`, _Test
//! code_, 1 to 4); a learn over GF(2) and the replay of its record return
//! the basis, which the rule in a learn broke (test 5, bivium-rest-gap.8,
//! log 0181-2).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::bivium_gap_tests::{product, sparse};
use crate::budget::Budget;
use crate::echelon::{SQUARES_OUT, echelon, squares_out};
use crate::f4::KOSZUL_SEARCHED;
use crate::field::Gf2;
use crate::gf2_bound_losses_tests::certified;
use crate::gf2_input_time_tests::{EDGE, sparse_pivots};
use crate::gf2_start_over_tests::{call, forcing, reported};
use crate::id::MonoId;
use crate::input_redundancy_tests::{read, table};
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::report_tests::Line;
use crate::shared_tests::{P, below, divides, rng, sig_divides};
use crate::sig::{Formed, Sig, Signatures};
use crate::systems::random_system;
use crate::table::Table;
use crate::trace::NoTrace;

/// The edge cases over x, y, z with the field equations of x, y, and z,
/// and the systems whose squares the rule leaves out or keeps, each with
/// the rows the rule leaves out of its echelon over GF(2): a square of a
/// linear row of two variables, of one variable alone, and of a variable
/// and a constant; two squares; and a square whose tail's variable lacks
/// its field equation, which stays.
const SQUARES: [(&str, usize); 6] = [
    ("x+y+1,\nx^2+x,\ny^2+y", 1),
    ("x,\nx^2+x,\ny^2+y", 1),
    ("x+1,\nx^2+x,\ny*z+z", 1),
    ("x+y,\ny+1,\nx^2+x,\ny^2+y", 2),
    ("x+y+z,\nx^2+x,\ny^2+y", 0),
    ("x+y,\nx^2+x,\ny^2+y,\nz^2+z,\nx*z+y", 1),
];

/// The systems over GF(2) of tests 2 and 4, each with its name: systems of
/// 8 to 16 variables whose linear generators lead some, their quadrics, and
/// every field equation (`sparse_pivots`); the edge cases over x, y, z
/// with the field equations of x, y, and z; and `SQUARES`.
fn gf2_systems() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (seed, (n, k)) in (0xb7e0u64..).zip([(8, 2), (10, 3), (12, 5), (14, 6), (16, 8)]) {
        out.push((
            format!("sparse_pivots({n}, {k}, {q}, 2, {seed:#x})", q = 2 * n),
            sparse_pivots(n, k, 2 * n, 2, seed),
        ));
    }
    let fields = "x^2+x,\ny^2+y,\nz^2+z";
    for (k, g) in EDGE.iter().enumerate() {
        let gens = if g.is_empty() {
            fields.to_owned()
        } else {
            format!("{g},\n{fields}")
        };
        out.push((
            format!("edge {k} with the field equations"),
            format!("x,y,z\n2\n{gens}\n"),
        ));
        out.push((format!("edge {k}"), format!("x,y,z\n2\n{g}\n")));
    }
    for (k, (g, _)) in SQUARES.iter().enumerate() {
        out.push((format!("squares {k}"), format!("x,y,z\n2\n{g}\n")));
    }
    out
}

/// The rows the rule left out in the call of the report `lines`: the
/// `squares` of each of its `echelon` lines, in order.
fn squares(name: &str, lines: &[Line]) -> Vec<u64> {
    lines
        .iter()
        .filter(|l| l.kind == "echelon")
        .map(|l| {
            l.get("squares")
                .unwrap_or_else(|| panic!("{name}: an echelon line without squares: {l:?}"))
                .parse::<u64>()
                .unwrap()
        })
        .collect()
}

/// The instance `text` over GF(2) over p in place of 2, each field
/// equation `v^2+v` of a variable v written `v^2-v`.
fn over(p: u64, text: &str) -> String {
    let mut parts = text.splitn(3, '\n');
    let names = parts.next().unwrap();
    assert_eq!(parts.next(), Some("2"), "{text:?}: over GF(2)");
    let rest = parts.next().unwrap().trim_end_matches('\n');
    let fields: Vec<String> = names.split(',').map(|v| format!("{v}^2+{v}")).collect();
    let gens: Vec<String> = rest
        .split(",\n")
        .map(|g| {
            if fields.iter().any(|f| f == g) {
                g.replacen('+', "-", 1)
            } else {
                g.to_owned()
            }
        })
        .collect();
    format!("{names}\n{p}\n{gens}\n", gens = gens.join(",\n"))
}

/// What `f` returns with `SQUARES_OUT` at `on` and `KOSZUL_SEARCHED` at
/// `searched` on this thread, both back to their defaults after.
fn with<T>(on: bool, searched: bool, f: impl FnOnce() -> T) -> T {
    SQUARES_OUT.set(on);
    KOSZUL_SEARCHED.set(searched);
    let got = f();
    SQUARES_OUT.set(true);
    KOSZUL_SEARCHED.set(false);
    got
}

/// Test 1 (bivium-rest-gap.1 to .3): the generators of `SQUARES` and of
/// the edge cases with the field equations, over GF(2) in grevlex, in
/// reduced row echelon form: `squares_out` leaves out as zero the rows
/// `SQUARES` names, each a row `x_v^2 + L` whose tail is the tail of the
/// linear row `x_v + L`, every variable of that tail with its row
/// `x_j^2 + x_j`; every other row stays as the echelon left it, at its
/// position; and on the edge cases with the field equations it counts the
/// rows it leaves out.
#[test]
fn the_squares_of_linear_rows_leave() {
    let names = "x,y,z";
    for (k, (g, out)) in SQUARES.iter().enumerate() {
        let mut t = table(names);
        let gens = read(&Gf2, &mut t, &Grevlex, names, g);
        let mut b = Budget::unbounded();
        let e = echelon(&Gf2, &mut t, &Grevlex, gens, &mut b).unwrap();
        let before: Vec<Vec<MonoId>> = e.rows.iter().map(|r| r.monomials.clone()).collect();
        let mut rows = e.rows;
        let left = squares_out(&t, &mut rows, &mut b).unwrap();
        assert_eq!(left, *out, "squares {k}: the rows left out");
        let mut zeroed = 0;
        for (i, (row, was)) in rows.iter().zip(&before).enumerate() {
            if row.is_zero() && !was.is_empty() {
                zeroed += 1;
                let lead = t.monomial(was[0]);
                let degree = t.degree(was[0]);
                let mut held = Vec::new();
                t.format().held(lead, &mut held);
                assert!(
                    degree == 2 && held.len() == 1,
                    "squares {k}: row {i} left out, {was:?}, is no square of a variable"
                );
                continue;
            }
            assert_eq!(&row.monomials, was, "squares {k}: row {i} stays");
        }
        assert_eq!(zeroed, *out, "squares {k}: the rows zeroed");
    }
    for (k, g) in EDGE.iter().enumerate() {
        let g = if g.is_empty() {
            "x^2+x,\ny^2+y,\nz^2+z".to_owned()
        } else {
            format!("{g},\nx^2+x,\ny^2+y,\nz^2+z")
        };
        let mut t = table(names);
        let gens = read(&Gf2, &mut t, &Grevlex, names, &g);
        let mut b = Budget::unbounded();
        let mut rows = echelon(&Gf2, &mut t, &Grevlex, gens, &mut b).unwrap().rows;
        let nonzero = rows.iter().filter(|r| !r.is_zero()).count();
        let left = squares_out(&t, &mut rows, &mut b).unwrap();
        let after = rows.iter().filter(|r| !r.is_zero()).count();
        assert_eq!(
            nonzero - after,
            left,
            "edge {k}: the rows left out are counted"
        );
    }
}

/// Test 2 (bivium-rest-gap.1 to .4): the calls over GF(2) in grevlex of
/// `gf2_systems`, the bounded run's bound forced to 0 so that the
/// signature run follows it, with the rule and without (`SQUARES_OUT`):
/// both return the basis of the call with the signatures off, which the
/// call with cofactors and the exact checker returns and certifies; the
/// rule leaves out at least one row on each `sparse_pivots` system and, on
/// each `echelon` line of `SQUARES`, no more than test 1 does, and without
/// it none; and over
/// GF(65521) the calls of the same systems with `x^2 - x` in place of
/// `x^2 + x` leave out none.
#[test]
fn calls_with_the_squares_left_out() {
    let mut left = 0;
    for (name, text) in gf2_systems() {
        let (want, _) = call(&text, &reported(1, false), None);
        let (on, lines) = with(true, false, || call(&text, &reported(1, true), Some(0)));
        let (off, off_lines) = with(false, false, || call(&text, &reported(1, true), Some(0)));
        assert_eq!(on, want, "{name}: with the rule");
        assert_eq!(off, want, "{name}: without the rule");
        assert_eq!(certified(&name, &text, 1), want, "{name}: certified");
        assert!(
            squares(&name, &off_lines).iter().all(|&q| q == 0),
            "{name}: none left out without the rule"
        );
        let got = squares(&name, &lines);
        if name.starts_with("sparse_pivots") {
            assert!(
                got.iter().any(|&q| q > 0),
                "{name}: a linear generator's square left out"
            );
        }
        // `SQUARES`' counts are test 1's, of the echelon alone: before it the
        // signature run's substitution may put a linear row's constant in
        // a field equation, which then is zero.
        if let Some(k) = name.strip_prefix("squares ") {
            let k: usize = k.parse().unwrap();
            assert!(
                got.iter().all(|&q| q <= SQUARES[k].1 as u64),
                "{name}: the squares left out, {got:?}"
            );
        }
        left += got.iter().sum::<u64>();
        let over_p = over(P, &text);
        let (want, _) = call(&over_p, &reported(1, false), None);
        let (got, lines) = call(&over_p, &reported(1, true), None);
        assert_eq!(got, want, "{name} over {P}");
        assert!(
            squares(&name, &lines).iter().all(|&q| q == 0),
            "{name} over {P}: none left out"
        );
    }
    assert!(left > 0, "no row was left out");
}

/// Test 3 (bivium-rest-gap.5, .6): on tables of 3, 40, and 150 variables,
/// seeds 0 to 7, inputs whose leading monomials hold one to three
/// variables, and drawn signatures of their indices, a multiple of their
/// input's lead, among them multiples of a Koszul signature and of one
/// recorded before: two signatures of the same inputs record each
/// signature no known one divides, one by `push_syzygy` and the other by
/// `push_coprime`, and after every record `is_syzygy` and
/// `is_syzygy_formed` answer alike in both, as brute force over the
/// recorded signatures and the Koszul signatures `lm(f_i)·lm(f_j)·e_j`,
/// i below j, answers.
#[test]
fn coprime_records_answer_as_the_searched_ones() {
    for n in [3usize, 40, 150] {
        for seed in 0..8u64 {
            let mut rng = rng(seed ^ 0xb7e1 ^ n as u64);
            let mut t = Table::new(Dense16::new(n), &mut rng.clone());
            let mut b = Budget::unbounded();
            let inputs = 4 + below(&mut rng, 24) as usize;
            let leads: Vec<Vec<u32>> = (0..inputs)
                .map(|_| {
                    loop {
                        let held = 1 + below(&mut rng, 3);
                        let e = sparse(&mut rng, n, held, 2);
                        if e.iter().any(|&x| x > 0) {
                            break e;
                        }
                    }
                })
                .collect();
            let ids: Vec<MonoId> = leads.iter().map(|e| t.intern(e, &mut b).unwrap()).collect();
            let (mut searched, mut coprime) = (Signatures::new(inputs), Signatures::new(inputs));
            searched.set_inputs(&t, ids.clone());
            coprime.set_inputs(&t, ids.clone());
            let mut recorded: Vec<(Vec<u32>, u32)> = Vec::new();
            for draw in 0..300 {
                let j = below(&mut rng, inputs as u64) as usize;
                let held = below(&mut rng, 4);
                let mut u = sparse(&mut rng, n, held, 2);
                match below(&mut rng, 3) {
                    0 if j > 0 => u = product(&u, &leads[below(&mut rng, j as u64) as usize]),
                    1 if !recorded.is_empty() => {
                        let (e, i) = &recorded[below(&mut rng, recorded.len() as u64) as usize];
                        if *i == j as u32 {
                            // A multiple of a recorded signature, its quotient by
                            // the lead the product below puts back.
                            u = e
                                .iter()
                                .zip(&leads[j])
                                .zip(&u)
                                .map(|((x, l), y)| x - l + y)
                                .collect();
                        }
                    }
                    _ => {}
                }
                let se = product(&u, &leads[j]);
                let s = Sig {
                    monomial: t.intern(&se, &mut b).unwrap(),
                    index: j as u32,
                };
                let known = (0..j).any(|i| divides(&product(&leads[i], &leads[j]), &se))
                    || recorded
                        .iter()
                        .any(|(e, i)| sig_divides((e, *i), (&se, s.index)));
                let case = format!("n {n}, seed {seed}, draw {draw}: {se:?} of index {j}");
                let degree = t.form_quotient(s.monomial, ids[j], ids[j]).unwrap();
                let formed = Formed {
                    index: s.index,
                    degree,
                    mask: t.format().mask(t.formed()),
                };
                for (sigs, how) in [(&mut searched, "searched"), (&mut coprime, "coprime")] {
                    assert_eq!(
                        sigs.is_syzygy(&mut t, s, &mut b).unwrap(),
                        known,
                        "{case}, {how}"
                    );
                    assert_eq!(
                        sigs.is_syzygy_formed(&mut t, &formed, &mut b).unwrap(),
                        known,
                        "{case}, {how}, formed"
                    );
                }
                if !known && below(&mut rng, 2) == 0 {
                    assert!(
                        searched.push_syzygy(&mut t, s, &mut b).unwrap(),
                        "{case}, pushed"
                    );
                    coprime.push_coprime(&t, s);
                    recorded.push((se, s.index));
                }
            }
        }
    }
}

/// Test 4 (bivium-rest-gap.5, .6): calls with the signatures on, over
/// GF(65521) and GF(7) on `random_system`, seeds 0 to 23, over ℚ on
/// `random_system`, seeds 0 to 11, and over GF(2) on `gf2_systems` with the
/// bounded run's bound forced to 0, each with the coprime S-pairs' Koszul
/// signatures recorded by `push_coprime` and by `push_syzygy`
/// (`KOSZUL_SEARCHED`): both return the basis of the call with the
/// signatures off, and over GF(65521) the call with cofactors and the
/// exact checker returns it and certifies it.
#[test]
fn calls_with_the_coprime_records() {
    let mut cases: Vec<(String, String, Option<u64>)> = Vec::new();
    for p in [P, 7] {
        for seed in 0..24u64 {
            cases.push((
                format!("random_system({p}, {seed})"),
                random_system(p, seed),
                None,
            ));
        }
    }
    for seed in 0..12u64 {
        cases.push((
            format!("random_system(0, {seed})"),
            random_system(0, seed),
            None,
        ));
    }
    for (name, text) in gf2_systems() {
        cases.push((name, text, Some(0)));
    }
    for (name, text, forced) in cases {
        let (want, _) = call(&text, &reported(1, false), None);
        let (coprime, _) = with(true, false, || call(&text, &reported(1, true), forced));
        let (searched, _) = with(true, true, || call(&text, &reported(1, true), forced));
        assert_eq!(coprime, want, "{name}: recorded by push_coprime");
        assert_eq!(searched, want, "{name}: recorded by push_syzygy");
        if name.starts_with(&format!("random_system({P}")) {
            assert_eq!(certified(&name, &text, 1), want, "{name}: certified");
        }
    }
}

/// Test 5 (bivium-rest-gap.8): over GF(2) on `gf2_systems`, the bounded
/// run's bound forced to 0 so that the signature run makes the record, a
/// learn and the replay of its record on the same system: both return the
/// basis of the call with the signatures off, the replay with no misfit.
#[test]
fn learns_and_replays_over_gf2() {
    for (name, text) in gf2_systems() {
        let (want, _) = call(&text, &reported(1, false), None);
        let params = reported(1, true);
        let (learned, record) = forcing(Some(0), || crate::learn(&text, &params, &mut NoTrace))
            .unwrap_or_else(|e| panic!("{name}: the learn failed: {e:?}"));
        assert_eq!(
            learned.basis.as_deref(),
            Some(want.as_str()),
            "{name}: learned"
        );
        let record = record.unwrap_or_else(|| panic!("{name}: no record"));
        let replayed = forcing(Some(0), || {
            crate::replay(&text, &params, &record, &mut NoTrace)
        })
        .unwrap_or_else(|e| panic!("{name}: the replay failed: {e:?}"));
        assert_eq!(
            replayed.basis.as_deref(),
            Some(want.as_str()),
            "{name}: replayed"
        );
    }
}
