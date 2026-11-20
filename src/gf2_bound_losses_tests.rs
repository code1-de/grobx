//! gf2-bound-losses, the item's test code: calls over GF(2) whose bounded
//! plain run halts where its row work, projected over a round in sparse
//! rows, passes the bound, and starts over in the signature run, return
//! the basis the exact checker certifies; no round in packed rows halts by
//! the projection; and the projection's arithmetic
//! (`iterations/0131-gf2-bound-losses/plan.md`, _Test code_, 1 to 3).

use crate::budget::{Budget, Site};
use crate::gf2_start_over_tests::{call, reported};
use crate::packed::PACK_FORCED;
use crate::report_tests::Line;
use crate::split_piece_time_tests::EDGE;
use crate::systems::boolean;
use crate::{CheckerMode, groebner};

/// The forced bounds of tests 1 and 2: 4^k units for k = 0 to 15, a bound
/// of `u64::MAX` units, and none.
pub(crate) fn bounds() -> Vec<Option<u64>> {
    let mut bounds: Vec<Option<u64>> = (0..16).map(|k| Some(1u64 << (2 * k))).collect();
    bounds.extend([Some(u64::MAX), None]);
    bounds
}

/// The systems of tests 1 and 2 over GF(2): `boolean(n, m, seed)` for n =
/// 9 to 13, m = n + 2 and m = 2n, by name, and where `edge` holds the
/// edge cases on x, y, z.
pub(crate) fn systems(edge: bool) -> Vec<(String, String)> {
    let mut systems = Vec::new();
    for n in 9..=13usize {
        for m in [n + 2, 2 * n] {
            let seed = 0xb0b0 + (n * 32 + m) as u64;
            systems.push((format!("boolean({n}, {m}, {seed:#x})"), boolean(n, m, seed)));
        }
    }
    if edge {
        for (k, gens) in EDGE.iter().enumerate() {
            systems.push((format!("edge case {k}"), format!("x,y,z\n2\n{gens}\n")));
        }
    }
    systems
}

/// The basis of the call of `text` with cofactors and signatures off,
/// which the exact checker certifies.
pub(crate) fn certified(name: &str, text: &str, seed: u64) -> String {
    let mut params = reported(seed, false);
    params.cofactors = true;
    params.checker = Some(CheckerMode::Exact);
    let outcome = groebner(text, &params).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert_eq!(outcome.certified, Some(true), "{name}: certified");
    outcome.basis.unwrap()
}

/// What the report of a call under a forced bound shows: the `outworked`
/// lines' row work and bound, and whether a carry line stands.
#[derive(Default)]
pub(crate) struct Halts {
    /// Per `outworked` line, its `work` and `bound`.
    pub(crate) outworked: Vec<(u64, u64)>,
    pub(crate) carried: bool,
}

/// The halts `lines` report, each `outworked` line's `bound` asserted
/// `forced`'s units.
pub(crate) fn halts(case: &str, lines: &[Line], forced: Option<u64>) -> Halts {
    let mut halts = Halts::default();
    for line in lines {
        if line.is("ran", &[("outcome", "outworked")]) {
            let field = |key: &str| -> u64 {
                line.get(key)
                    .unwrap_or_else(|| panic!("{case}: no {key} in {line:?}"))
                    .parse()
                    .unwrap_or_else(|e| panic!("{case}: {key} in {line:?}: {e}"))
            };
            let (work, bound) = (field("work"), field("bound"));
            assert_eq!(Some(bound), forced, "{case}: the bound of {line:?}");
            halts.outworked.push((work, bound));
        }
        if line.kind == "carry" {
            halts.carried = true;
        }
    }
    halts
}

/// Test 1: gf2-bound-losses.1, .3, .4, .5, .7. Over GF(2), on test 1's
/// systems under each forced bound, the call returns the basis that the
/// call with cofactors and signatures off returns certified by the exact
/// checker. Every `outworked` line names the forced units as its `bound`
/// and the row work as its `work`. Among the calls, one halts by the
/// projection, its `work` at most its `bound`; one past the bound; and one
/// carries after its start over, which moves its elements by the map. At a
/// bound of `u64::MAX` and at none no call halts.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn the_projected_halt() {
    let (mut projected, mut past, mut carried) = (0, 0, 0);
    for (name, text) in systems(true) {
        let seed = 0xb0c1;
        let want = certified(&name, &text, seed);
        for forced in bounds() {
            let case = format!("{name}, forced {forced:?}");
            let (basis, lines) = call(&text, &reported(seed, true), forced);
            assert_eq!(basis, want, "{case}");
            let halts = halts(&case, &lines, forced);
            // A bound of `u64::MAX` units, and none, halt no call.
            let unbounded = match forced {
                Some(units) => units == u64::MAX,
                None => true,
            };
            if unbounded {
                let outworked = &halts.outworked;
                assert!(outworked.is_empty(), "{case}: {outworked:?}");
            }
            projected += halts.outworked.iter().filter(|(w, b)| w <= b).count();
            past += halts.outworked.iter().filter(|(w, b)| w > b).count();
            carried += usize::from(halts.carried);
        }
    }
    assert!(projected > 0, "no halt by the projection");
    assert!(past > 0, "no halt past the bound");
    assert!(carried > 0, "no carry after a start over");
}

/// Test 2: gf2-bound-losses.2. Test 1's boolean systems with every round
/// over GF(2) in packed rows (`PACK_FORCED`): each call returns test 1's
/// basis, and every `outworked` line's `work` lies past its `bound`. An
/// `outworked` line stands among the calls.
#[test]
#[cfg_attr(not(feature = "checker"), ignore = "the build leaves out the checker")]
fn no_projection_in_packed_rounds() {
    let mut outworked = 0;
    for (name, text) in systems(false) {
        let seed = 0xb0c2;
        let want = certified(&name, &text, seed);
        PACK_FORCED.set(Some(true));
        for forced in bounds() {
            let case = format!("{name}, packed, forced {forced:?}");
            let (basis, lines) = call(&text, &reported(seed, true), forced);
            assert_eq!(basis, want, "{case}");
            let halts = halts(&case, &lines, forced);
            for (work, bound) in &halts.outworked {
                assert!(work > bound, "{case}: work {work} at most bound {bound}");
            }
            outworked += halts.outworked.len();
        }
        PACK_FORCED.set(None);
    }
    assert!(outworked > 0, "no halt in packed rows");
}

/// The rows and the rows begun of test 3: every pair up to `ROWS` rows.
const ROWS: usize = 120;

/// A budget without a bound whose row work is `s`.
pub(crate) fn worked(s: u64) -> Budget {
    let mut budget = Budget::unbounded();
    budget
        .spend_subtracted(usize::try_from(s).unwrap(), Site::Row)
        .expect("an unbounded budget stops no call");
    budget
}

/// Test 3: gf2-bound-losses.1. A budget bounded at u units after s units
/// of row work, and r units more: `projected_past(s, reached, rows)` holds
/// exactly where s + r · rows / reached, rounded down, passes s + u, at
/// every rows and reached up to `ROWS`. A budget without a bound, and one
/// bounded at `u64::MAX`, which a projection past `u64::MAX` does not
/// pass, never project past.
#[test]
fn the_projection() {
    let cases: [(u64, u64, u64); 6] = [
        (0, 1000, 1),
        (0, 1000, 999),
        (17, 1 << 20, 1 << 14),
        (1 << 30, 1 << 40, 1 << 33),
        (1 << 40, 1 << 30, 1 << 30),
        (5, 3, 0),
    ];
    for (s, u, r) in cases {
        let mut budget = worked(s);
        assert_eq!(budget.bound(u), u, "the units of the bound");
        budget
            .spend_subtracted(usize::try_from(r).unwrap(), Site::Row)
            .expect("an unbounded budget stops no call");
        assert_eq!(budget.row_work(), s + r, "the row work");
        for rows in 1..=ROWS {
            for reached in 1..=rows {
                let projected = u128::from(s) + u128::from(r) * rows as u128 / reached as u128;
                assert_eq!(
                    budget.projected_past(s, reached, rows),
                    projected > u128::from(s + u),
                    "s {s}, u {u}, r {r}, {reached} of {rows} rows"
                );
            }
        }
    }
    let mut unbound = worked(1 << 20);
    unbound
        .spend_subtracted(1 << 20, Site::Row)
        .expect("an unbounded budget stops no call");
    assert!(!unbound.projected_past(0, 1, ROWS), "no bound");
    let mut most = worked(3);
    most.bound(u64::MAX);
    most.spend_subtracted(1 << 62, Site::Row)
        .expect("an unbounded budget stops no call");
    assert!(!most.projected_past(3, 1, 1 << 20), "a bound of u64::MAX");
}
