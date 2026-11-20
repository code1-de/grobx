//! linear-substitution-time, the item's test code: the substitution's rule
//! with its terms as keys and as the run's words; a generator that stays at
//! a long sum on the way; a call with the keys against the same call with
//! the words and without the substitution; the keys' bounds; and a stop in
//! the keyed sums (`iterations/0169-linear-substitution-time/plan.md`,
//! _Test code_, 1 to 5).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices and draws"
)]

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::ccode_newton_gap_tests::call;
use crate::field::{Field, Gf2, Gf31};
use crate::gf2_input_time_tests::{
    EDGE, Pivots, Terms, add_into, drawn, pivots, read_into, sparse_pivots, substitutions_in,
    terms_of, through, written,
};
use crate::gf2_start_over_tests::forcing;
use crate::monomial::{Dense8, Dense16, Format};
use crate::order::Grevlex;
use crate::rebuilt_reducer_tests::{TIMES, reported};
use crate::report_tests::Line;
use crate::shared_tests::{below, report, rng};
use crate::substitute::{INDICES, KEY_LANES, KEY_LIMIT, keyed, substitute};
use crate::table::Table;
use crate::{MonomialOrder, Outcome, Params};

/// `case` with every generator past degree `KEY_LANES` left out, and, for
/// two pivots or more, x1²·x2² + x_(k+1)² added, whose Horner's rule skips
/// a power at each level and squares the values: a case the keys take.
fn within_keys(mut case: Pivots) -> Pivots {
    let degree = |e: &Vec<u32>| e.iter().sum::<u32>();
    case.others
        .retain(|g| g.keys().all(|e| degree(e) <= KEY_LANES));
    let k = case.linear.len();
    if k >= 2 {
        let mut e = vec![0; case.n];
        e[0] = 2;
        e[1] = 2;
        let mut square = vec![0; case.n];
        square[k] = 2;
        case.others.push(Terms::from([(e, 1), (square, 1)]));
    }
    case
}

/// Test 1's check of one case in the format M over `field`: with
/// `INDICES` on and off, `substitute` keeps gf2-input-time's amended rule
/// (`gf2_input_time_tests::substitutions_in`), and the two return the same
/// generators and counts.
fn both_encodings<F: Field, M: Format>(field: &F, case: &Pivots, name: &str) {
    INDICES.set(true);
    let keys = substitutions_in::<F, M>(field, case, &format!("{name}, keys"));
    INDICES.set(false);
    let words = substitutions_in::<F, M>(field, case, &format!("{name}, words"));
    INDICES.set(true);
    assert_eq!(keys, words, "{name}: the keys against the words");
}

/// Test 1, linear-substitution-time.1, .2, .5 to .7, .9: `substitute` over
/// GF(2) and 65521, in `Dense8` and `Dense16`, on gf2-input-time's test-1
/// shapes, each as drawn, which holds a generator of degree 5 and takes
/// the run's words, and within the keys, of degree `KEY_LANES` at most; and
/// on shapes of degree 4 and 5: the rule holds with the keys and with the
/// words, and both return the same generators and counts. Shape 4 holds
/// 40 pivots, and its generators that hold every pivot sort by the slices.
#[test]
fn the_rule_in_both_encodings() {
    let shapes = [
        (6, 1, 1, 8, 4),
        (10, 3, 4, 12, 4),
        (24, 12, 16, 16, 3),
        (60, 40, 4, 24, 2),
        (12, 2, 12, 4, 4),
        (16, 3, 6, 16, 3),
        (16, 3, 6, 16, 4),
    ];
    for (s, &(n, k, width, count, degree)) in (1u64..).zip(&shapes) {
        for p in [2u64, 65521] {
            let drawn_case = || pivots(0x0169 + s, n, k, width, count, degree, p);
            for (form, case) in [
                ("as drawn", drawn_case()),
                ("within the keys", within_keys(drawn_case())),
            ] {
                let name = format!("shape {s} over {p}, {form}");
                if p == 2 {
                    both_encodings::<_, Dense8>(&Gf2::new(), &case, &format!("{name}, Dense8"));
                    both_encodings::<_, Dense16>(&Gf2::new(), &case, &format!("{name}, Dense16"));
                } else {
                    let field = Gf31::new(p).unwrap();
                    both_encodings::<_, Dense8>(&field, &case, &format!("{name}, Dense8"));
                    both_encodings::<_, Dense16>(&field, &case, &format!("{name}, Dense16"));
                }
            }
        }
    }
}

/// Test 2's case over p: the pivots x1 and x2, both of value −(x3 + x4);
/// g = x1·x2 − x2², whose expansion is 0; and h = x1·x5 + x3·x5 + x4·x5 +
/// x6², whose expansion is x6².
fn long_sum(p: u64) -> String {
    format!("x1,x2,x3,x4,x5,x6\n{p}\nx1+x3+x4,\nx2+x3+x4,\nx1*x2-x2^2,\nx1*x5+x3*x5+x4*x5+x6^2\n")
}

/// Test 2 over `field`, with `INDICES` at `keys`.
fn long_sum_in<F: Field>(field: &F, p: u64, keys: bool) {
    INDICES.set(keys);
    let mut table = Table::new(
        Dense8::new(6),
        &mut Xoshiro256PlusPlus::seed_from_u64(0x0169),
    );
    let before = read_into(field, &mut table, &long_sum(p));
    let got = substitute(
        field,
        &mut table,
        &Grevlex,
        before.clone(),
        &mut Budget::unbounded(),
    );
    INDICES.set(true);
    let got = got.unwrap();
    let name = format!("over {p}, keys {keys}");
    assert_eq!(
        terms_of(field, &table, &got.generators[2]),
        terms_of(field, &table, &before[2]),
        "{name}: g stays at its sum of two terms, though its expansion is 0"
    );
    let mut square = vec![0; 6];
    square[5] = 2;
    assert_eq!(
        terms_of(field, &table, &got.generators[3]),
        Terms::from([(square, 1)]),
        "{name}: h takes x6²"
    );
    assert_eq!(
        [got.linear, got.pivots, got.changed, got.stayed],
        [2, 2, 1, 1],
        "{name}: the counts"
    );
}

/// Test 2, linear-substitution-time.1: over GF(2) and 65521, with the keys
/// and with the words, g's substitution ends at the sum Horner's rule
/// drains at x2's level, the value −(x3 + x4), which holds as many terms as
/// g, and g stays as read though its expansion is 0; h, of one pivot to
/// the power 1, drains its substitute alone and takes x6².
#[test]
fn a_long_sum_keeps_its_generator() {
    for keys in [true, false] {
        long_sum_in(&Gf2::new(), 2, keys);
        long_sum_in(&Gf31::new(65521).unwrap(), 65521, keys);
    }
}

/// `a_longer_substitute_keeps_its_generator` over `field`, with `INDICES` at `keys`.
fn longer_substitute_in<F: Field>(field: &F, p: u64, keys: bool) {
    INDICES.set(keys);
    let mut table = Table::new(
        Dense8::new(3),
        &mut Xoshiro256PlusPlus::seed_from_u64(0x0169),
    );
    let before = read_into(field, &mut table, &format!("x,y,z\n{p}\nx-y-z,\nx^2-1\n"));
    let got = substitute(
        field,
        &mut table,
        &Grevlex,
        before.clone(),
        &mut Budget::unbounded(),
    );
    INDICES.set(true);
    let got = got.unwrap();
    let name = format!("over {p}, keys {keys}");
    assert_eq!(
        terms_of(field, &table, &got.generators[1]),
        terms_of(field, &table, &before[1]),
        "{name}: g stays as read"
    );
    assert_eq!(
        [got.linear, got.pivots, got.changed, got.stayed],
        [1, 1, 0, 1],
        "{name}: the counts"
    );
}

/// The pivot x of value y + z and g = x² − 1, whose substitute y² + 2yz + z² − 1 holds more terms
/// than g: over GF(2) and 65521, with the keys and with the words, g stays as read.
#[test]
fn a_longer_substitute_keeps_its_generator() {
    for keys in [true, false] {
        longer_substitute_in(&Gf2::new(), 2, keys);
        longer_substitute_in(&Gf31::new(65521).unwrap(), 65521, keys);
    }
}

/// A system over p on n variables that vanishes at a drawn point: k linear
/// generators, each a pivot plus a sparse form in the variables past k, and
/// `count` generators of degree at most `degree` in the pivots and the
/// variables past k; over GF(2) the field equations of every variable.
fn planted(n: usize, k: usize, count: usize, degree: u32, p: u64, seed: u64) -> String {
    let mut r = rng(seed);
    let rest: Vec<usize> = (k..n).collect();
    let a: Vec<u64> = (0..n).map(|_| below(&mut r, p)).collect();
    let mut gens = Vec::new();
    for i in 0..k {
        let terms = 1 + below(&mut r, 4) as usize;
        let mut form = drawn(&mut r, &rest, n, 1, terms, p);
        let mut lead = vec![0; n];
        lead[i] = 1;
        form.insert(lead, 1);
        gens.push(written(&through(form, &a, p)));
    }
    let all: Vec<usize> = (0..n).collect();
    for _ in 0..count {
        let mut g = drawn(&mut r, &all, n, degree, 5, p);
        let mut top = vec![0; n];
        for _ in 0..degree {
            top[all[below(&mut r, n as u64) as usize]] += 1;
        }
        add_into(&mut g, &Terms::from([(top, 1)]), 1, p);
        gens.push(written(&through(g, &a, p)));
    }
    if p == 2 {
        for i in 1..=n {
            gens.push(format!("x{i}^2+x{i}"));
        }
    }
    let names: Vec<String> = (1..=n).map(|i| format!("x{i}")).collect();
    format!(
        "{names}\n{p}\n{gens}\n",
        names = names.join(","),
        gens = gens.join(",\n")
    )
}

/// The report's lines but their times and the `end` line: a field of a
/// stage's name or `time`, and any field whose value is a decimal with a
/// point, as the lift's `merge`, `reconstruct`, and `crt` are.
pub(crate) fn untimed(outcome: &Outcome) -> Vec<Line> {
    let timed = |k: &str, v: &str| {
        TIMES.contains(&k) || k == "time" || (v.contains('.') && v.parse::<f64>().is_ok())
    };
    report(outcome)
        .into_iter()
        .filter(|l| l.kind != "end")
        .map(|mut l| {
            l.time = 0.0;
            l.fields.retain(|(k, v)| !timed(k, v));
            l
        })
        .collect()
}

/// The call of `text` with the substitution, `INDICES` at `keys`.
fn with_keys(text: &str, params: &Params, keys: bool) -> Outcome {
    INDICES.set(keys);
    let outcome = call(text, params, true);
    INDICES.set(true);
    outcome
}

/// Test 3, linear-substitution-time.1 to .9: calls over GF(2), 65521, and 7
/// on gf2-input-time's sparse-pivot systems and edge cases, over ℚ on the
/// edge cases, on planted systems of degree 4 that hold linear generators,
/// and on one of degree 5, the signatures off and on: with the keys a call
/// returns the basis and the report, but its times, that it returns with
/// the words, and the basis it returns without the substitution.
#[test]
fn calls_return_what_they_return_with_the_words() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for p in [2u64, 65521, 7, 0] {
        for (k, g) in EDGE.iter().enumerate() {
            texts.push((format!("edge {k} over {p}"), format!("x,y,z\n{p}\n{g}\n")));
        }
    }
    let shapes = [
        (2, 12, 4, 10),
        (2, 20, 8, 16),
        (7, 12, 4, 16),
        (65521, 16, 6, 24),
    ];
    for (s, &(p, n, k, quadrics)) in (1u64..).zip(&shapes) {
        texts.push((
            format!("sparse pivots {n}/{k} over {p}"),
            sparse_pivots(n, k, quadrics, p, 0x0169 + s),
        ));
    }
    let planted_shapes = [
        (2, 12, 4, 8, 4),
        (2, 14, 6, 10, 3),
        (65521, 8, 3, 6, 4),
        (2, 10, 3, 6, 5),
    ];
    for (s, &(p, n, k, count, degree)) in (1u64..).zip(&planted_shapes) {
        texts.push((
            format!("planted {n}/{k} of degree {degree} over {p}"),
            planted(n, k, count, degree, p, 0x1690 + s),
        ));
    }
    for (name, text) in &texts {
        for signatures in [false, true] {
            let mut params = reported(MonomialOrder::Grevlex);
            params.signatures = signatures;
            let keys = with_keys(text, &params, true);
            let words = with_keys(text, &params, false);
            let off = call(text, &params, false);
            let name = format!("{name}, signatures {signatures}");
            assert!(keys.basis.is_some(), "{name}: a basis");
            assert_eq!(
                keys.basis, words.basis,
                "{name}: the keys' basis and the words'"
            );
            assert_eq!(untimed(&keys), untimed(&words), "{name}: the reports");
            assert_eq!(
                keys.basis, off.basis,
                "{name}: the basis without the substitution"
            );
        }
    }
}

/// Test 4, linear-substitution-time.2, .3: the keys hold a run of fewer
/// than `KEY_LIMIT` variables whose generators have degree `KEY_LANES` at
/// most, and no other.
#[test]
fn the_keys_by_variables_and_degree() {
    for degree in 0..=KEY_LANES {
        assert!(keyed(1, degree), "1 variable, degree {degree}");
        assert!(keyed(124, degree), "124 variables, degree {degree}");
    }
    assert!(keyed(KEY_LIMIT - 1, KEY_LANES), "KEY_LIMIT − 1 variables");
    assert!(!keyed(124, KEY_LANES + 1), "degree past KEY_LANES");
    assert!(!keyed(KEY_LIMIT, 2), "KEY_LIMIT variables");
    assert!(!keyed(KEY_LIMIT + 1, 2), "past KEY_LIMIT variables");
}

/// Test 5, linear-substitution-time.1, .2: a call on a sparse-pivot system
/// over GF(2), whose keyed sums grow and whose pivots' field equations end
/// their substitutions at long sums, and one on a planted system of degree
/// 5, which takes the words, stops at the first check at
/// `Site::Substitute`, in the signature run that a bound of 0 starts the
/// call over in, and returns what it returns without the forcing, with the
/// keys and with the words.
#[test]
fn stops_in_the_keyed_sums() {
    let texts = [
        sparse_pivots(30, 12, 24, 2, 0x5ab6),
        planted(12, 4, 8, 5, 2, 0x5ab7),
    ];
    for keys in [true, false] {
        for text in &texts {
            INDICES.set(keys);
            // Over GF(2) the signature run alone substitutes.
            let got = forcing(Some(0), || {
                stops_at(Site::Substitute, &Case::new(text.clone(), "grevlex", false))
            });
            INDICES.set(true);
            got.unwrap_or_else(|e| panic!("keys {keys}: {e}"));
        }
    }
}
