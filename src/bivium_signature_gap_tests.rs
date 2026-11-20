//! bivium-signature-gap, the item's test code: the signature run's
//! comparison of two sig-lead ratios on whole words, the gathers of a
//! monomial's lanes over the words that hold a variable, and the hash of the
//! table's scratch words over their variables, each against the exponents it
//! stands for, in every format and on few and on hundreds of variables; and
//! calls over hundreds of variables with signatures on and off, which return
//! one basis that the checker certifies
//! (`iterations/0183-bivium-signature-gap/plan.md`, _Test code_, 1 to 4).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::cmp::Ordering;

use crate::bivium_gap_tests::sparse;
use crate::budget::Budget;
use crate::divisibility_tests::{mask_of, random_lanes};
use crate::gf2_bound_losses_tests::certified;
use crate::gf2_input_time_tests::{EDGE, sparse_pivots};
use crate::gf2_start_over_tests::{call, reported};
use crate::gf2_update_rest_time_tests::support_of;
use crate::id::MonoId;
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::order::{Block, Deglex, Grevlex, Lex};
use crate::shared_tests::{below, exponents, grevlex, rng};
use crate::sig::cmp_ratio;
use crate::table::{Table, sum_hash};

/// The counts of variables tests 1 to 3 take: few, past 32, past
/// `KOSZUL_INDEX_VARIABLES`, and bivium's.
const VARIABLE_COUNTS: [usize; 6] = [1, 3, 17, 40, 150, 423];

/// The exponents of a·b, each lane the sum of two, as i64.
fn sum(a: &[u32], b: &[u32]) -> Vec<i64> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) + i64::from(y))
        .collect()
}

/// The orders test 1 compares under, by name.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Grevlex,
    Deglex,
    Lex,
    Block,
}

/// The order `kind` on exponent vectors of any size: grevlex the degree and
/// then the last differing variable's smaller exponent; deglex the degree
/// and then the first differing variable's larger exponent; lex that alone;
/// and `Block::new(&[k], n)` grevlex on the first k variables and then on
/// the rest.
fn ordered(kind: Kind, k: usize, a: &[i64], b: &[i64]) -> Ordering {
    match kind {
        Kind::Grevlex => grevlex(a, b),
        Kind::Deglex => {
            let (da, db): (i64, i64) = (a.iter().sum(), b.iter().sum());
            da.cmp(&db).then_with(|| a.cmp(b))
        }
        Kind::Lex => a.cmp(b),
        Kind::Block => grevlex(&a[..k], &b[..k]).then_with(|| grevlex(&a[k..], &b[k..])),
    }
}

/// `cmp_ratio` under `kind`, with the block order's first block of k.
fn compared<M: Format>(
    table: &Table<M>,
    kind: Kind,
    k: usize,
    a: (MonoId, MonoId),
    b: (MonoId, MonoId),
) -> Ordering {
    let n = table.format().variables();
    match kind {
        Kind::Grevlex => cmp_ratio(table, &Grevlex, a, b),
        Kind::Deglex => cmp_ratio(table, &Deglex, a, b),
        Kind::Lex => cmp_ratio(table, &Lex, a, b),
        Kind::Block => {
            let order = Block::new(&[k], n).unwrap_or_else(|| panic!("block:{k} on {n}"));
            cmp_ratio(table, &order, a, b)
        }
    }
}

/// The monomial of exponents e in `table`.
fn interned<M: Format>(table: &mut Table<M>, e: &[u32]) -> MonoId {
    table
        .intern(e, &mut Budget::unbounded())
        .unwrap_or_else(|limit| panic!("{e:?}: {limit:?}"))
}

/// Test 1 for one format: on `VARIABLE_COUNTS` variables but past the
/// format's reach, `per` draws each, four monomials of the table of degree
/// at most `MAX_DEGREE`, a of the first ratio's signature, b its lead, c
/// the second's signature, d its lead: under each order the comparison is
/// the order's on the exponents of a·d and c·b, whatever their degrees, and
/// a common factor of a and b leaves it as it is.
fn check_ratios<M: Format>(seed: u64, per: usize) {
    let mut r = rng(seed);
    let top = M::MAX_DEGREE;
    for n in VARIABLE_COUNTS {
        let mut table = Table::new(M::new(n), &mut rng(seed ^ n as u64));
        let k = (n / 2).max(1);
        for _ in 0..per {
            let draw: Vec<Vec<u32>> = (0..4)
                .map(|_| random_lanes(&mut r, n, top, Some(top)))
                .collect();
            let ids: Vec<MonoId> = draw.iter().map(|e| interned(&mut table, e)).collect();
            let (left, right) = (sum(&draw[0], &draw[3]), sum(&draw[2], &draw[1]));
            for kind in [Kind::Grevlex, Kind::Deglex, Kind::Lex, Kind::Block] {
                // A block order takes two blocks.
                let skip = match kind {
                    Kind::Block => n < 2,
                    Kind::Grevlex | Kind::Deglex | Kind::Lex => false,
                };
                if skip {
                    continue;
                }
                let got = compared(&table, kind, k, (ids[0], ids[1]), (ids[2], ids[3]));
                assert_eq!(
                    got,
                    ordered(kind, k, &left, &right),
                    "{kind:?} on {n}: {draw:?}"
                );
            }
            // A common factor g of a and b, where g·a fits the format.
            let room = top - draw[0].iter().chain(&draw[1]).sum::<u32>().min(top);
            let g = random_lanes(&mut r, n, room / 2, Some(room / 2));
            let ga: Vec<u32> = draw[0].iter().zip(&g).map(|(x, y)| x + y).collect();
            let gb: Vec<u32> = draw[1].iter().zip(&g).map(|(x, y)| x + y).collect();
            let (ga, gb) = (interned(&mut table, &ga), interned(&mut table, &gb));
            assert_eq!(
                cmp_ratio(&table, &Grevlex, (ga, gb), (ids[2], ids[3])),
                cmp_ratio(&table, &Grevlex, (ids[0], ids[1]), (ids[2], ids[3])),
                "a common factor on {n}: {draw:?}, {g:?}"
            );
        }
        // Products of twice the degree bound: x_1^top·x_n^top against 1·1.
        let mut a = vec![0u32; n];
        a[0] = top;
        let mut d = vec![0u32; n];
        d[n - 1] = top;
        let one = vec![0u32; n];
        let (a, d, one) = (
            interned(&mut table, &a),
            interned(&mut table, &d),
            interned(&mut table, &one),
        );
        assert_eq!(
            cmp_ratio(&table, &Grevlex, (a, one), (one, d)),
            Ordering::Greater,
            "x_1^{top}·x_{n}^{top} on {n}"
        );
        assert_eq!(
            cmp_ratio(&table, &Lex, (one, a), (d, one)),
            Ordering::Less,
            "1·1 against x_{n}^{top}·x_1^{top} in lex on {n}"
        );
    }
}

/// Test 1: bivium-signature-gap.1. The comparison of two sig-lead ratios
/// is the order's on the two products' exponents.
#[test]
fn ratios_compare_on_whole_words() {
    check_ratios::<Dense4>(0xb5_0001, 300);
    check_ratios::<Dense8>(0xb5_0002, 300);
    check_ratios::<Dense16>(0xb5_0003, 300);
    check_ratios::<Dense32>(0xb5_0004, 300);
}

/// The counts of variables test 2 takes beside 0 to 70: past 64, 128, and
/// 256 variables, about the word boundaries of every format.
const MORE_COUNTS: [usize; 9] = [99, 127, 128, 129, 150, 255, 256, 423, 513];

/// Exponents on n variables whose nonzero ones stand in `words` of the
/// format's words of `lanes` lanes, drawn, the first and the last word
/// among the draws: each such word holds one to four variables of
/// exponent 1 to `top`.
fn in_words(
    r: &mut rand_xoshiro::Xoshiro256PlusPlus,
    n: usize,
    lanes: usize,
    words: usize,
    top: u32,
) -> Vec<u32> {
    let mut e = vec![0u32; n];
    if n == 0 {
        return e;
    }
    let count = n.div_ceil(lanes);
    for t in 0..words {
        let w = match t {
            0 => 0,
            1 => count - 1,
            _ => below(r, count as u64) as usize,
        };
        for _ in 0..=below(r, 4) {
            let v = (w * lanes + below(r, lanes as u64) as usize).min(n - 1);
            e[v] = 1 + below(r, u64::from(top)) as u32;
        }
    }
    e
}

/// Test 2 for one format: on 0 to 70 variables and `MORE_COUNTS`, `per`
/// draws each, lanes of no word, one, two, or more words of the format,
/// each lane at most the degree bound: the mask is `mask_of`'s, the
/// support `support_of`'s, and the variables `held` names those of nonzero
/// exponent, increasing.
fn check_gathers<M: Format>(seed: u64, per: usize) {
    let mut r = rng(seed);
    let top = M::MAX_DEGREE;
    for n in (0..=70).chain(MORE_COUNTS) {
        let format = M::new(n);
        // The variables of the format's first word.
        let lanes = (0..n).take_while(|&v| format.unit(v).0 == 0).count().max(1);
        let mut words = vec![M::Word::default(); format.words()];
        let mut held = Vec::new();
        for t in 0..per {
            let e = in_words(&mut r, n, lanes, t % 5, top.min(4));
            format.pack(&e, &mut words);
            assert_eq!(format.mask(&words), mask_of(&e), "mask on {n}: {e:?}");
            assert_eq!(
                format.support(&words),
                support_of(&e),
                "support on {n}: {e:?}"
            );
            format.held(&words, &mut held);
            let want: Vec<u32> = (0..n).filter(|&i| e[i] > 0).map(|i| i as u32).collect();
            assert_eq!(held, want, "held on {n}: {e:?}");
        }
    }
}

/// Test 2: bivium-signature-gap.2 and .4. The gathers of a monomial's
/// lanes pass over its words of no variable and give what every word's
/// gather gives.
#[test]
fn gathers_pass_words_of_no_variable() {
    check_gathers::<Dense4>(0xb5_0011, 200);
    check_gathers::<Dense8>(0xb5_0012, 200);
    check_gathers::<Dense16>(0xb5_0013, 200);
    check_gathers::<Dense32>(0xb5_0014, 200);
}

/// The hash of exponents e in `table` as `sum_hash` states it.
fn hash_of<M: Format>(table: &Table<M>, e: &[u32]) -> u64 {
    let multipliers: Vec<u64> = (0..e.len()).map(|i| table.multiplier(i)).collect();
    sum_hash(
        &multipliers,
        e.iter()
            .enumerate()
            .filter(|&(_, &x)| x > 0)
            .map(|(i, &x)| (i, x)),
    )
}

/// Test 3 for one format: on `VARIABLE_COUNTS` variables, `per` draws
/// each, sparse monomials s, x, y of the table, each of one to three
/// variables: the signature s·y/gcd(x, y) formed in the scratch words and
/// interned takes the hash `sum_hash` gives its exponents and the id their
/// interning gives; and the monomial 1 and one of the last variable alone
/// take theirs.
fn check_hashes<M: Format>(seed: u64, per: usize) {
    let mut r = rng(seed);
    let top = u64::from(M::MAX_DEGREE.min(3));
    for n in VARIABLE_COUNTS {
        let mut table = Table::new(M::new(n), &mut rng(seed ^ n as u64));
        let mut b = Budget::unbounded();
        for _ in 0..per {
            let held = 1 + below(&mut r, 3);
            let draw: Vec<Vec<u32>> = (0..3).map(|_| sparse(&mut r, n, held, top)).collect();
            let want: Vec<u32> = (0..n)
                .map(|i| draw[0][i] + draw[2][i] - draw[1][i].min(draw[2][i]))
                .collect();
            if want.iter().sum::<u32>() > M::MAX_DEGREE
                || draw.iter().any(|e| e.iter().sum::<u32>() > M::MAX_DEGREE)
            {
                continue;
            }
            let ids: Vec<MonoId> = draw.iter().map(|e| interned(&mut table, e)).collect();
            for (id, e) in ids.iter().zip(&draw) {
                assert_eq!(
                    table.hash(*id),
                    hash_of(&table, e),
                    "interned on {n}: {e:?}"
                );
            }
            let degree = table
                .form_quotient(ids[0], ids[1], ids[2])
                .unwrap_or_else(|o| panic!("{draw:?}: {o:?}"));
            let formed = table
                .intern_formed(degree, &mut b)
                .unwrap_or_else(|limit| panic!("{draw:?}: {limit:?}"));
            assert_eq!(exponents(&table, formed), want, "formed on {n}: {draw:?}");
            assert_eq!(
                table.hash(formed),
                hash_of(&table, &want),
                "its hash on {n}"
            );
            assert_eq!(interned(&mut table, &want), formed, "its id on {n}");
        }
        let one = vec![0u32; n];
        let mut last = vec![0u32; n];
        last[n - 1] = 1;
        for e in [one, last] {
            let id = interned(&mut table, &e);
            assert_eq!(table.hash(id), hash_of(&table, &e), "{e:?} on {n}");
        }
    }
}

/// Test 3: bivium-signature-gap.3. The hash of the scratch words is the sum
/// of their variables' terms.
#[test]
fn the_scratch_hash_sums_the_exponents() {
    check_hashes::<Dense4>(0xb5_0021, 400);
    check_hashes::<Dense8>(0xb5_0022, 400);
    check_hashes::<Dense16>(0xb5_0023, 400);
    check_hashes::<Dense32>(0xb5_0024, 400);
}

/// The system `text` over n variables named v1 to vn, with x, y, and z
/// taking the places 4, n/2, and n among them.
fn spread(n: usize, characteristic: u64, gens: &str) -> String {
    let names: Vec<String> = (1..=n)
        .map(|i| match i {
            4 => "x".to_owned(),
            _ if i == n / 2 => "y".to_owned(),
            _ if i == n => "z".to_owned(),
            _ => format!("v{i}"),
        })
        .collect();
    format!(
        "{names}\n{characteristic}\n{gens}\n",
        names = names.join(",")
    )
}

/// The systems of test 4, each with its name: over GF(2) and GF(65521)
/// systems of 40, 140, and 200 variables whose linear generators lead all
/// but 8 and whose quadrics hold those 8 and two pivots (`sparse_pivots`);
/// the edge cases over x, y, z spread over 140 variables, over GF(2) with
/// and without the field equations of x, y, and z; and over GF(2) a
/// generator of degree 8, which `Dense4` does not hold.
fn systems() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (seed, n) in (0xb5_0030u64..).zip([40, 140, 200]) {
        for p in [2, 65521] {
            out.push((
                format!("sparse_pivots({n}, {k}, 12, {p}, {seed:#x})", k = n - 8),
                sparse_pivots(n, n - 8, 12, p, seed),
            ));
        }
    }
    let fields = "x^2+x,\ny^2+y,\nz^2+z";
    for (k, g) in EDGE.iter().enumerate() {
        for p in [2, 65521] {
            out.push((format!("edge {k} over {p} on 140"), spread(140, p, g)));
        }
        let gens = if g.is_empty() {
            fields.to_owned()
        } else {
            format!("{g},\n{fields}")
        };
        out.push((
            format!("edge {k} with the field equations on 140"),
            spread(140, 2, &gens),
        ));
    }
    out.push((
        "degree 8 on 140".to_owned(),
        spread(
            140,
            2,
            "v1*v2*v3*x*v5*v6*v7*v8+y,\ny*z+x,\nx^2+x,\ny^2+y,\nz^2+z",
        ),
    ));
    out
}

/// Test 4: bivium-signature-gap.1 to .4 and the acceptance criterion. Each
/// system of `systems` returns one basis with signatures on, the bound on
/// the plain run's row work forced to 0 so that a call over GF(2) starts
/// over in the signature run, and with signatures off, and the checker
/// certifies it.
#[test]
fn calls_over_hundreds_of_variables() {
    for (name, text) in systems() {
        let (on, _) = call(&text, &reported(1, true), Some(0));
        let (off, _) = call(&text, &reported(1, false), None);
        assert_eq!(on, off, "{name}: signatures on against off");
        assert_eq!(certified(&name, &text, 1), on, "{name}: certified");
    }
}
