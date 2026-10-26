//! plain-zero-time, the crate's test code: the blocks of the plain run's
//! combinations, a block after one that made no pivot grown to twice its
//! rows and a grown block stopped past `GROWN_PIVOTS` pivots, against a
//! reference of the test's own; the edges of the blocks' sizes; and a
//! deadline inside a grown block
//! (`iterations/0077-plain-zero-time/plan.md`, _Test code_, 1 to 3).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use crate::Stop;
use crate::cofactors::NoCofactors;
use crate::cyclic_gap_tests::{Echelon, Random, Shape, led, long, rows, with_rest};
use crate::f4::Halt;
use crate::field::{BAND, Field, Gf31};
use crate::id::RowId;
use crate::matrix::{self, BLOCK, Combine, GROWN_PIVOTS, Reduction, ZEROS};
use crate::monomial::Dense16;
use crate::plain_band_tests::passed_after;
use crate::systems::Rng;
use crate::trace::RowSource;

/// The primes of the combinations: below 2^16 and the largest below 2^31,
/// each past the combinations' bound of 46341.
const COMBINING: [u64; 2] = [65521, 2_147_483_647];

/// A block of the item's rule as the reference made it: its first row,
/// its rows, the pivots it made, whether it holds more than `BLOCK` rows,
/// and whether it stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    pub(crate) first: u32,
    pub(crate) rows: usize,
    pub(crate) pivots: usize,
    pub(crate) grown: bool,
    pub(crate) stopped: bool,
}

/// Row r's entries: column and coefficient.
fn sparse(m: &Random, r: usize) -> Vec<(usize, u64)> {
    let coefs = &m.basis.element(m.matrix.rows[r].element).coefs;
    m.matrix
        .row(RowId::at(r))
        .iter()
        .zip(coefs)
        .map(|(&c, &a)| (c.index(), u64::from(a)))
        .collect()
}

/// The reference of the item's blocks (plan, _Algorithms_), on the test's
/// own elimination: the pivot rows held in an `Echelon`, then the rows to
/// reduce in the order of cyclic-gap.1 cut into blocks of `BLOCK` rows, or
/// twice the rows of the block before where that block made no pivot.
/// Each combination is a combination of the block's rows under
/// coefficients drawn from `seed` in [1, p), a pivot where it does not
/// reduce to zero; where the matrix takes bands a band of min(`BAND`, n −
/// pivots) combinations, its lanes one after another. A block of more than
/// `BLOCK` rows that has made more than `GROWN_PIVOTS` pivots stops before
/// its next band or combination, and the blocks after it take its rows
/// again from its first. The blocks in the order reduced.
pub(crate) fn blocks(m: &Random, seed: u64) -> Vec<Block> {
    let p = m.field.characteristic();
    let width = m.matrix.columns();
    let (pivot_rows, rest) = m.split();
    let mut e = Echelon::new(p, width);
    for &r in &pivot_rows {
        e.insert(m.dense(r));
    }
    let entries: Vec<Vec<(usize, u64)>> = rest.iter().map(|&r| sparse(m, r)).collect();
    let bands = matrix::takes_bands::<Gf31, Dense16, NoCofactors>(&m.matrix);
    let mut rng = Rng::new(seed ^ 0x9e0b);
    // Whether a combination of `block`'s rows reduces to zero, held where
    // it does not.
    let mut combination = |e: &mut Echelon, block: &[Vec<(usize, u64)>]| {
        let mut v = vec![0u64; width];
        for row in block {
            let k = 1 + rng.below(p - 1);
            for &(c, a) in row {
                v[c] = ((u128::from(v[c]) + u128::from(k) * u128::from(a)) % u128::from(p)) as u64;
            }
        }
        e.insert(v)
    };
    let mut out = Vec::new();
    let (mut start, mut size) = (0, BLOCK);
    while start < rest.len() {
        let first = start;
        let end = rest.len().min(start + size);
        let block = &entries[start..end];
        start = end;
        let n = block.len();
        let grown = n > BLOCK;
        let (mut pivots, mut done, mut stopped) = (0, 0, false);
        while bands && done == 0 && n - pivots >= 2 {
            if grown && pivots > GROWN_PIVOTS {
                stopped = true;
                break;
            }
            for _ in 0..BAND.min(n - pivots) {
                if combination(&mut e, block) {
                    done += 1;
                } else {
                    pivots += 1;
                }
            }
        }
        while !stopped && pivots < n && done < ZEROS {
            if grown && pivots > GROWN_PIVOTS {
                stopped = true;
                break;
            }
            if combination(&mut e, block) {
                done += 1;
            } else {
                pivots += 1;
            }
        }
        out.push(Block {
            first: rest[first] as u32,
            rows: n,
            pivots,
            grown,
            stopped,
        });
        if stopped {
            start = first;
            size = BLOCK;
        } else if pivots == 0 {
            size = 2 * n;
        } else {
            size = BLOCK;
        }
    }
    out
}

/// The combinations: every made row names the first row of a block of
/// `want`, as many as the block's pivots, in block order.
pub(crate) fn assert_blocks(name: &str, reduction: &Reduction<Gf31>, want: &[Block]) {
    let got: Vec<u32> = reduction
        .made
        .iter()
        .map(|r| r.row.index() as u32)
        .collect();
    let mut expected = Vec::new();
    for b in want {
        expected.extend(std::iter::repeat_n(b.first, b.pivots));
    }
    assert_eq!(got, expected, "{name}: the made rows by block");
}

/// A matrix of `shape` at p whose rows to reduce that lead farthest right
/// lie in the span of its pivot rows: every element of `G_min` given
/// `copies` + 1 times with multiplier 1, every copy past the first a row
/// to reduce that the first reduces to zero, beside the shape's S-pair
/// halves, which lead at their lcms, farther left.
fn zeros_first(p: u64, seed: u64, shape: &Shape, copies: usize) -> Random {
    Random::with_rows(p, seed, shape, |table, basis, mut rows| {
        let one = table.intern(&[0; 4]).unwrap();
        for &g in basis.min() {
            for _ in 0..=copies {
                rows.push(RowSource {
                    element: g,
                    multiplier: one,
                });
            }
        }
        rows
    })
}

/// The matrix of test 1 at p for `seed`, of long rows where `is_long`.
fn case(p: u64, seed: u64, is_long: bool) -> Random {
    let shape = if is_long {
        long(6 + (seed % 5) as usize, 4 + (seed * 7 % 48) as usize, 0, 0)
    } else {
        led(12, 4 + (seed * 11 % 64) as usize)
    };
    let copies = if is_long {
        6 + (seed * 5 % 40) as usize
    } else {
        3 + (seed * 3 % 30) as usize
    };
    zeros_first(p, seed, &shape, copies)
}

/// Whether the matrix's plain run takes bands.
fn banded(m: &Random) -> bool {
    matrix::takes_bands::<Gf31, Dense16, NoCofactors>(&m.matrix)
}

/// `m`'s combinations under `seed` against `want`: the made rows by block,
/// and the made rows by leading column those of each row on its own.
fn assert_case(name: &str, m: &Random, want: &[Block], seed: u64) {
    let combined = m.reduce(Some(&mut Combine::new(0x9e0c ^ seed)));
    assert_blocks(name, &combined, want);
    assert_eq!(rows(&combined), rows(&m.reduce(None)), "{name}");
}

/// Test 1: plain-zero-time.1 to .8. On matrices whose rows to reduce that
/// lead farthest right lie in the span of the pivot rows, of short rows
/// and of long rows, at 65521 and 2^31 − 1: every made row names the first
/// row of a block of the item's rule, as many as its pivots, in block
/// order; and the made rows are those of each row on its own. Among the
/// cases, grown blocks, grown blocks that end with pivots, stopped blocks,
/// in bands and one combination at a time.
#[test]
fn grown_blocks_name_their_first_rows() {
    let (mut grown, mut ended, mut stopped, mut in_bands, mut alone) = (0, 0, 0, 0, 0);
    for p in COMBINING {
        for seed in 0..40u64 {
            for is_long in [false, true] {
                let m = case(p, seed, is_long);
                let name = format!("p {p}, seed {seed}, long {is_long}");
                let want = blocks(&m, seed);
                assert_case(&name, &m, &want, seed);
                if want.iter().any(|b| b.grown) {
                    grown += 1;
                    if banded(&m) {
                        in_bands += 1;
                    } else {
                        alone += 1;
                    }
                }
                ended += usize::from(want.iter().any(|b| b.grown && !b.stopped && b.pivots > 0));
                stopped += usize::from(want.iter().any(|b| b.stopped));
            }
        }
    }
    assert!(grown >= 6, "{grown} cases with a grown block");
    assert!(
        ended >= 6,
        "{ended} cases with a grown block that ends with pivots"
    );
    assert!(stopped >= 6, "{stopped} cases with a stopped block");
    assert!(
        in_bands >= 6,
        "{in_bands} cases with a grown block in bands"
    );
    assert!(
        alone >= 6,
        "{alone} cases with a grown block one combination at a time"
    );
}

/// The first matrix of test 1's family at 65521, of long rows where
/// `is_long`, whose reference holds `wanted`, among seeds 0 to 399, and
/// its seed.
fn found(is_long: bool, what: &str, wanted: impl Fn(&[Block]) -> bool) -> (String, Random, u64) {
    (0..400u64)
        .find_map(|seed| {
            let m = case(65521, seed, is_long);
            wanted(&blocks(&m, seed)).then(|| (format!("{what}, seed {seed}"), m, seed))
        })
        .unwrap_or_else(|| panic!("no matrix with {what}"))
}

/// A grown block of `pivots` pivots that ended, or that stopped.
fn grown_with(pivots: usize, stopped: bool) -> impl Fn(&[Block]) -> bool {
    move |want| {
        want.iter()
            .any(|b| b.grown && b.pivots == pivots && b.stopped == stopped)
    }
}

/// Test 2: the edges of the blocks' sizes at 65521: no row to reduce; one;
/// `BLOCK`, one block; `BLOCK` + 1 rows whose first block makes no pivot,
/// the second the one row left; a stretch of rank 0 across blocks of 64,
/// 128, and the rows left; grown blocks of 16 pivots that end and of 17
/// that stop, one combination at a time; in bands, grown blocks of 16, 17,
/// and 18 pivots that end and of 19 and 20 that stop; and a stopped block
/// whose rows the next blocks take again, one of which makes no pivot and
/// doubles the one after. `GROWN_PIVOTS` is 16, the reference holds each
/// edge, and each case is as test 1 asserts.
#[test]
fn the_edges_of_the_sizes() {
    assert_eq!(GROWN_PIVOTS, 16, "plain-zero-time.6");
    let p = 65521;
    let mut cases: Vec<(String, Random, u64)> = Vec::new();
    let no_row = Random::with_rows(p, 0, &led(5, 0), |table, basis, _| {
        let one = table.intern(&[0; 4]).unwrap();
        basis
            .min()
            .iter()
            .map(|&g| RowSource {
                element: g,
                multiplier: one,
            })
            .collect()
    });
    assert!(no_row.split().1.is_empty(), "a row to reduce");
    cases.push(("no row".to_string(), no_row, 0));
    // n rows to reduce, every one a copy of an element of `G_min`, and
    // the reference's blocks as (rows, pivots).
    let mut sized = |what: &str, n: usize, sizes: &[(usize, usize)]| {
        let m = with_rest(p, 1, &led(4, 0), n).unwrap_or_else(|| panic!("no matrix of {what}"));
        let want: Vec<(usize, usize)> = blocks(&m, 1).iter().map(|b| (b.rows, b.pivots)).collect();
        assert_eq!(want, sizes, "{what}: the reference's blocks");
        cases.push((what.to_string(), m, 1));
    };
    sized("one row", 1, &[(1, 0)]);
    sized("BLOCK rows", BLOCK, &[(BLOCK, 0)]);
    sized("BLOCK + 1 rows", BLOCK + 1, &[(BLOCK, 0), (1, 0)]);
    sized(
        "a stretch of rank 0",
        3 * BLOCK + 100,
        &[(BLOCK, 0), (2 * BLOCK, 0), (100, 0)],
    );
    cases.push(found(
        false,
        "16 pivots, ended, one at a time",
        grown_with(GROWN_PIVOTS, false),
    ));
    cases.push(found(
        false,
        "17 pivots, stopped, one at a time",
        grown_with(GROWN_PIVOTS + 1, true),
    ));
    for pivots in GROWN_PIVOTS..GROWN_PIVOTS + BAND - 1 {
        cases.push(found(
            true,
            &format!("{pivots} pivots, ended, in bands"),
            grown_with(pivots, false),
        ));
    }
    for pivots in GROWN_PIVOTS + BAND - 1..=GROWN_PIVOTS + BAND {
        cases.push(found(
            true,
            &format!("{pivots} pivots, stopped, in bands"),
            grown_with(pivots, true),
        ));
    }
    for is_long in [false, true] {
        cases.push(found(
            is_long,
            &format!("a doubling after a stop, long {is_long}"),
            |want| {
                want.windows(3).any(|w| {
                    w[0].stopped
                        && w[1].first == w[0].first
                        && w[1].rows == BLOCK
                        && w[1].pivots == 0
                        && w[2].grown
                })
            },
        ));
    }
    assert_eq!(cases.len(), 14);
    for (name, m, seed) in &cases {
        let want = blocks(m, *seed);
        assert_case(name, m, &want, *seed);
    }
}

/// Test 3: a deadline inside the grown blocks. Matrices of test 1 at
/// 65521 whose reference holds a grown block, of short rows and of long
/// rows; a deadline passed and unchecked, the setup's units and `T` more
/// between two full checks, for `T` an eighth to seven eighths of the
/// units past the setup's that the reduction spends unbounded: the
/// reduction stops with `Deadline` after one full check and at most `T` +
/// `BAND`·W units past the setup's, W the matrix's width, which bounds a
/// row the reduction made as well as the matrix's rows.
#[test]
fn a_deadline_stops_a_grown_block() {
    let mut cases = 0;
    for seed in 0..40u64 {
        for is_long in [false, true] {
            let m = case(65521, seed, is_long);
            if !blocks(&m, seed).iter().any(|b| b.grown) {
                continue;
            }
            let setup = (m.matrix.rows.len() + m.matrix.columns()) as u64;
            let width = m.matrix.columns() as u64;
            let mut b = crate::cyclic_gap_tests::budget();
            m.reduce_in(Some(&mut Combine::new(seed)), &mut b).unwrap();
            let past = b.spent() - setup;
            for eighth in 1..8 {
                let t = past * eighth / 8;
                let name = format!("seed {seed}, long {is_long}, T {t}");
                let mut b = passed_after(setup + t);
                let got = m.reduce_in(Some(&mut Combine::new(seed)), &mut b);
                assert!(
                    matches!(got, Err(Halt::Stop(Stop::Deadline))),
                    "{name}: not stopped by the deadline"
                );
                assert_eq!(b.checks(), 1, "{name}");
                assert!(
                    b.spent() <= setup + t + BAND as u64 * width,
                    "{name}: {spent} spent before the stop, {past} past the setup in all",
                    spent = b.spent()
                );
            }
            cases += 1;
        }
    }
    assert!(cases >= 12, "{cases} matrices with a grown block");
}
