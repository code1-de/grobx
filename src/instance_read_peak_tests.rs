//! instance-read-peak, the item's test code in the crate: drawn texts read
//! with their distinct monomials found by the parse's hash index, with every
//! monomial's hash forced to one value as well, give what the test makes of
//! their terms; one monomial in every spelling is one distinct monomial; the
//! touched encoding writes the words the full encoding writes; the counting
//! sort orders as a stable sort; a parsed term keeps its line and sign;
//! calls on the edge cases return their certified bases; and a stop in the
//! parse's and the map's new loops
//! (`iterations/0175-instance-read-peak/plan.md`, _Test code_, 1 to 7).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::Error;
use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::field::{Gf2, Gf31, Gf64Pair};
use crate::id::MonoId;
use crate::instance_read_time_tests::{checked, exps, first_written, read, sparse};
use crate::monomial::{Dense4, Dense8, Dense16, Dense32, Format};
use crate::order::{Block, Grevlex, Lex};
use crate::parsed_tests::{Term, check, generators, table, text};
use crate::shared_tests::{P, REPLAY0, REPLAY1};
use crate::sort;
use crate::systems::Rng;
use crate::table::Table;
use crate::text::{self, Header, ONE_HASH};

/// Drawn texts of the shapes the counting sorts and the index's growth
/// meet: `sparse`'s on n variables, or one generator of many terms, or
/// many generators of a few terms, each term on 0 to 3 of the variables,
/// a copy of an earlier term at times.
fn shaped(rng: &mut Rng, n: usize, shape: u64) -> Vec<Vec<Term>> {
    let (gens, terms) = match shape {
        0 => return sparse(rng, n),
        1 => (1, 600),
        _ => (300, 3),
    };
    let mut out: Vec<Vec<Term>> = Vec::new();
    for _ in 0..gens {
        let mut g: Vec<Term> = Vec::new();
        for _ in 0..=rng.below(terms) {
            let mut exponents = vec![0; n];
            if !g.is_empty() && rng.below(5) == 0 {
                exponents.clone_from(&g[rng.below(g.len() as u64) as usize].exponents);
            } else {
                for _ in 0..rng.below(4) {
                    exponents[rng.below(n as u64) as usize] = 1 + rng.below(3) as u32;
                }
            }
            let mut order: Vec<usize> = (0..n).filter(|&i| exponents[i] > 0).collect();
            for k in (1..order.len()).rev() {
                order.swap(k, rng.below(k as u64 + 1) as usize);
            }
            g.push(Term {
                negative: rng.below(2) == 1,
                exponents,
                factors: Vec::new(),
                order,
            });
        }
        out.push(g);
    }
    out
}

/// Test 1, instance-read-peak.1 to .3, .7, and .8: drawn texts on 3, 40,
/// 130, 300, and 600 variables, of `shaped`'s shapes, read over three
/// fields, three orders, and two formats, give what the test makes of
/// their terms; a fresh table numbers their monomials in the order the
/// text first writes them.
#[test]
fn drawn_texts_as_written() {
    for seed in 0..45u64 {
        let mut rng = Rng::new(seed ^ 0x0175_1ead);
        let n = [3, 40, 130, 300, 600][(seed % 5) as usize];
        let gens = shaped(&mut rng, n, seed % 3);
        let text = text(&mut rng, n, &gens);
        let case = format!(
            "seed {seed}, {n} variables, shape {shape}",
            shape = seed % 3
        );
        let block = Block::new(&[n / 2], n).unwrap();
        macro_rules! over {
            ($field:expr) => {{
                let f = $field;
                check::<_, Dense16, _>(&case, &f, &Grevlex, n, &text, &gens);
                check::<_, Dense16, _>(&case, &f, &Lex, n, &text, &gens);
                check::<_, Dense16, _>(&case, &f, &block, n, &text, &gens);
                check::<_, Dense8, _>(&case, &f, &Grevlex, n, &text, &gens);
            }};
        }
        over!(Gf31::prime(P));
        over!(Gf2::new());
        over!(Gf64Pair::primes(REPLAY0, REPLAY1));
        let header = Header::read(&text, &mut Budget::unbounded()).unwrap();
        let mut t: Table<Dense16> = table(n);
        generators(&header, &Gf31::prime(P), &mut t, &Grevlex).unwrap();
        for (k, e) in first_written(&gens).iter().enumerate() {
            let id = t.intern(e, &mut Budget::unbounded()).unwrap();
            assert_eq!(id, MonoId::at(k), "{case}: monomial {k}\n{text}");
        }
    }
}

/// Test 2, instance-read-peak.1 and .9: with every distinct monomial's
/// hash forced to one value (`text::ONE_HASH`), so that each lookup meets
/// every distinct monomial before it in the index, drawn texts on 40 and
/// 130 variables read as test 1 reads them, and the index's growths keep
/// every monomial.
#[test]
fn one_hash_reads_alike() {
    ONE_HASH.set(true);
    for seed in 0..12u64 {
        let mut rng = Rng::new(seed ^ 0x0175_0002);
        let n = [40, 130][(seed % 2) as usize];
        let gens = shaped(&mut rng, n, seed % 3);
        let text = text(&mut rng, n, &gens);
        let case = format!("seed {seed}, {n} variables, one hash");
        check::<_, Dense16, _>(&case, &Gf31::prime(P), &Grevlex, n, &text, &gens);
        check::<_, Dense8, _>(&case, &Gf2::new(), &Grevlex, n, &text, &gens);
    }
    ONE_HASH.set(false);
}

/// Test 3, instance-read-peak.1: one monomial in every spelling, its
/// variables in each order, a variable repeated, and an exponent of 1 or
/// 0 written, is one distinct monomial: a generator of its five spellings
/// over GF(65521) reads as the one term 5·x1·x2^2, beside a generator
/// whose spellings of x1·x2 and x2·x1 cancel, which leaves x1·x3^2; and
/// with every hash forced to one value alike.
#[test]
fn spellings_are_one_monomial() {
    let text = "x1,x2,x3\n65521\n\
                x1*x2^2+x2*x1*x2+x2^2*x1+x2*x2*x1^1+x3^0*x1*x2^2,\n\
                x1*x2-x2*x1+x3^2*x1\n";
    let want = vec![vec![(vec![1, 2, 0], 5)], vec![(vec![1, 0, 2], 1)]];
    assert_eq!(read(text).unwrap(), want);
    ONE_HASH.set(true);
    let got = read(text);
    ONE_HASH.set(false);
    assert_eq!(got.unwrap(), want);
}

/// A degree and words that an encoding wrote, `Err(())` where it
/// overflowed.
type Encoded<W> = Result<(u32, Vec<W>), ()>;

/// The degree and the words of the exponents `e` in format `M` on
/// `e.len()` variables, by `encode` and by `encode_touched` over its
/// nonzero variables, whose words hold every lane's bits before, which the
/// touched encoding clears.
fn both<M: Format>(e: &[u32]) -> (Encoded<M::Word>, Encoded<M::Word>) {
    let f = M::new(e.len());
    let touched: Vec<usize> = (0..e.len()).filter(|&i| e[i] > 0).collect();
    let mut full = vec![M::Word::default(); f.words()];
    let full = f.encode(e, &mut full).map(|d| (d, full)).map_err(|_| ());
    let mut words = vec![M::Word::default(); f.words()];
    f.lanes(&(0..e.len()).collect::<Vec<_>>(), &mut words);
    let touched = f
        .encode_touched(e, &touched, &mut words)
        .map(|d| (d, words))
        .map_err(|_| ());
    (full, touched)
}

/// Test 4, instance-read-peak.4: `encode_touched` writes the words and
/// returns the degree that `encode` does, and overflows where it does, in
/// `Dense4`, `Dense8`, `Dense16`, and `Dense32`, on 1 to 600 variables, 0
/// to 5 of them nonzero, at and past each format's largest degree; and a
/// table interning by the touched variables gives the ids that interning
/// the full exponents gives.
#[test]
fn touched_encoding_as_full() {
    let mut rng = Rng::new(0x0175_0004);
    for n in [1, 15, 16, 17, 31, 32, 33, 64, 65, 130, 600] {
        for _ in 0..40 {
            let mut e = vec![0u32; n];
            for _ in 0..rng.below(6) {
                e[rng.below(n as u64) as usize] = 1 + rng.below(4) as u32;
            }
            let (a, b) = both::<Dense4>(&e);
            assert_eq!(a, b, "Dense4, {e:?}");
            let (a, b) = both::<Dense8>(&e);
            assert_eq!(a, b, "Dense8, {e:?}");
            let (a, b) = both::<Dense16>(&e);
            assert_eq!(a, b, "Dense16, {e:?}");
            let (a, b) = both::<Dense32>(&e);
            assert_eq!(a, b, "Dense32, {e:?}");
        }
        for top in [
            <Dense4 as Format>::MAX_DEGREE,
            <Dense8 as Format>::MAX_DEGREE,
        ] {
            for d in [top, top + 1] {
                let mut e = vec![0u32; n];
                e[n - 1] = d / 2;
                e[0] += d - d / 2;
                let (a, b) = both::<Dense4>(&e);
                assert_eq!(a, b, "{n} variables, degree {d}");
                let (a, b) = both::<Dense8>(&e);
                assert_eq!(a, b, "{n} variables, degree {d}");
            }
        }
        let mut by_touched: Table<Dense8> = table(n);
        let mut by_full: Table<Dense8> = table(n);
        for _ in 0..60 {
            let mut e = vec![0u32; n];
            for _ in 0..rng.below(4) {
                e[rng.below(n as u64) as usize] = 1 + rng.below(3) as u32;
            }
            let touched: Vec<usize> = (0..n).filter(|&i| e[i] > 0).collect();
            let a = by_touched
                .intern_touched(&e, &touched, &mut Budget::unbounded())
                .unwrap();
            let b = by_full.intern(&e, &mut Budget::unbounded()).unwrap();
            assert_eq!(a, b, "{n} variables, {e:?}");
            assert_eq!(exps(&by_touched, a), e, "{n} variables");
        }
    }
}

/// Test 5, instance-read-peak.7: `sort::counting` orders drawn items by
/// their keys below 1, 2, 7, and 1,000 as a stable sort by key does, none
/// and one item among them, with its counts reused from a larger sort.
#[test]
fn counting_sort_as_stable() {
    let mut rng = Rng::new(0x0175_0005);
    let mut counts = Vec::new();
    let mut out = Vec::new();
    for keys in [1usize, 2, 7, 1_000] {
        for len in [0usize, 1, 2, 9, 300, 5_000] {
            let items: Vec<(usize, u32)> = (0..len as u32)
                .map(|i| (rng.below(keys as u64) as usize, i))
                .collect();
            sort::counting(
                &items,
                |&(k, _)| k,
                keys,
                &mut counts,
                &mut out,
                &mut Budget::unbounded(),
                Site::Map,
            )
            .unwrap();
            let mut want = items.clone();
            want.sort_by_key(|&(k, _)| k);
            assert_eq!(out, want, "{keys} keys, {len} items");
        }
    }
}

/// Test 6, instance-read-peak.5: a parsed term keeps its factors' end, its
/// monomial, its line, and its sign; a line from `MINUS` on is
/// `Error::Size`; and through a text, the signs of 200 terms, every third a
/// minus, and the line of a term whose degree passes `Dense32` read back.
#[test]
fn terms_keep_line_and_sign() {
    for (end, line, negative) in [(0, 1, false), (7, 1, true), (u32::MAX, (1 << 31) - 1, true)] {
        let t = text::Term::new(end, MonoId::at(5), line, negative).unwrap();
        assert_eq!(
            (t.end, t.monomial, t.line(), t.negative()),
            (end, MonoId::at(5), line, negative)
        );
    }
    for line in [1usize << 31, u32::MAX as usize, usize::MAX] {
        let got = text::Term::new(0, MonoId::at(0), line, false).map(|_| ());
        assert_eq!(got, Err(Error::Size), "line {line}");
    }
    let terms: Vec<String> = (1..=200)
        .map(|i| format!("{sign}{i}*x^{i}", sign = if i % 3 == 0 { "-" } else { "+" }))
        .collect();
    let gens = read(&format!("x\n65521\n{terms}\n", terms = terms.concat())).unwrap();
    let want: Vec<(Vec<u32>, u64)> = (1..=200u64)
        .rev()
        .map(|i| (vec![i as u32], if i % 3 == 0 { P - i } else { i }))
        .collect();
    assert_eq!(gens, vec![want]);
    let past = u64::from(<Dense32 as Format>::MAX_DEGREE);
    let got = read(&format!("x,y\n65521\nx+y,\n\nx*y+x^{past}*y\n"));
    assert!(
        matches!(got, Err(Error::Overflow { line: Some(5) })),
        "{got:?}"
    );
}

/// Test 7: calls on the edge cases over GF(2), GF(65521), and ℚ, with the
/// exact checker, on 300 variables whose terms spell their monomials in
/// other orders, return their bases certified: the zero ideal, F = (0),
/// the unit ideal, a basis already, the zero polynomial among the
/// generators, and an eliminable variable; and a call whose parse grows
/// its index past 2^10 distinct monomials and whose map counts them stops
/// at the first check of `Site::Parse` and of `Site::Map`
/// (`call_stop_sites_tests::stops_at`).
#[test]
fn edge_cases_and_stops() {
    let names: Vec<String> = (1..=300).map(|i| format!("v{i}")).collect();
    let names = names.join(",");
    for p in [2, 65521, 0] {
        for (gens, want) in [
            ("0", Some("[]:\n")),
            ("0,0", Some("[]:\n")),
            ("v1*v300+1,v300*v1-v1,v1", Some("[1]:\n")),
            ("v2^2*v1-v300,v300^2-v7", None),
            ("v1*v300-v300*v1,v1^2*v2+v2*v1*v1,v300^3", None),
            ("v1-v300*1-v7,v7*v1^2*v300+v300*v7^2", None),
        ] {
            let text = format!("{names}\n{p}\n{gens}\n");
            let out = checked(&text);
            assert_eq!(out.certified, Some(true), "{text}: {out:?}");
            if let Some(want) = want {
                assert_eq!(out.basis.as_deref(), Some(want), "{text}");
            }
        }
    }
    let mut rng = Rng::new(0x0175_0007);
    let mut gens = Vec::new();
    for _ in 0..40 {
        let terms: Vec<String> = (0..60)
            .map(|_| {
                let a = 1 + rng.below(300);
                let b = 1 + rng.below(300);
                format!("v{a}*v{b}")
            })
            .collect();
        gens.push(terms.join("+"));
    }
    let points: Vec<String> = (1..=300).map(|i| format!("v{i}")).collect();
    let text = format!(
        "{names}\n2\n{gens},\n{points}\n",
        gens = gens.join(",\n"),
        points = points.join(",\n")
    );
    stops_at(Site::Parse, &Case::new(text.clone(), "grevlex", false)).unwrap();
    stops_at(Site::Map, &Case::new(text, "grevlex", false)).unwrap();
}
