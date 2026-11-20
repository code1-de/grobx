//! gf2-signature-losses, sub-item 2's test code: calls over GF(2) whose
//! plain run passes its bound start over in the signature run, with the
//! basis of the call without signatures
//! (`iterations/0129-gf2-signature-losses/plan.md`, _Test code_, 7).

use std::num::NonZeroUsize;

use crate::budget::WORK_FORCED;
use crate::report_tests::{Line, lines};
use crate::systems::boolean;
use crate::{MonomialOrder, Params, Threads};

/// The parameters of a call in grevlex on one thread under `seed`, with
/// its report, and signatures as `signatures` says.
pub(crate) fn reported(seed: u64, signatures: bool) -> Params {
    let mut params = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        seed,
    );
    params.report = true;
    params.signatures = signatures;
    params
}

/// What `f` returns with `WORK_FORCED` at `forced` on this thread, and
/// `WORK_FORCED` back to None after.
pub(crate) fn forcing<T>(forced: Option<u64>, f: impl FnOnce() -> T) -> T {
    WORK_FORCED.set(forced);
    let got = f();
    WORK_FORCED.set(None);
    got
}

/// The basis text and the report's lines of the call of `text` under
/// `params`, with `WORK_FORCED` at `forced` (`forcing`).
pub(crate) fn call(text: &str, params: &Params, forced: Option<u64>) -> (String, Vec<Line>) {
    let outcome = forcing(forced, || crate::groebner(text, params));
    let outcome = outcome.unwrap_or_else(|e| panic!("the call failed: {e:?}"));
    assert!(outcome.stopped.is_none(), "the call stopped: {outcome:?}");
    let report = outcome.report.clone().unwrap();
    (outcome.basis.unwrap(), lines(&report))
}

/// A `run`, `ran`, or `carry` line by its kind and the fields that tell
/// the runs apart: the engine, or a `ran` line's outcome; `engine-by`; and
/// `why`.
type RunLine = (String, Option<String>, Option<String>, Option<String>);

/// The report's `run`, `ran`, and `carry` lines in their order.
fn runs(lines: &[Line]) -> Vec<RunLine> {
    lines
        .iter()
        .filter(|l| matches!(l.kind.as_str(), "run" | "ran" | "carry"))
        .map(|l| {
            let get = |k: &str| l.get(k).map(str::to_string);
            let first = if l.kind == "ran" {
                get("outcome")
            } else {
                get("engine")
            };
            (l.kind.clone(), first, get("engine-by"), get("why"))
        })
        .collect()
}

/// Test 7: calls over GF(2) on `boolean(n, 2n, seed)` for n = 6 to 10,
/// with `WORK_FORCED` at `Some(0)`, at `Some(u64::MAX)`, and at None: one
/// basis, that of the call without signatures. At Some(0) the first run
/// line is the plain run's by its field, its `ran` line names
/// `outworked`, and the next run line is the signature run's for
/// `outworked`; at `Some(u64::MAX)` and at None no line names `outworked`.
/// Among the calls at Some(0) a signature run hands over, by a carry or a
/// start over in the plain run, asserted present.
#[test]
fn the_start_over() {
    let mut handed = 0;
    for n in 6..=10usize {
        let seed = 0xa2c0 + n as u64;
        let text = boolean(n, 2 * n, seed);
        let name = format!("boolean({n}, {m}, {seed:#x})", m = 2 * n);
        let (want, _) = call(&text, &reported(seed, false), None);
        for forced in [Some(0), Some(u64::MAX), None] {
            let case = format!("{name}, forced {forced:?}");
            let (basis, lines) = call(&text, &reported(seed, true), forced);
            assert_eq!(basis, want, "{case}");
            let runs = runs(&lines);
            let outworked = runs.iter().any(|(_, first, _, why)| {
                first.as_deref() == Some("outworked") || why.as_deref() == Some("outworked")
            });
            let first = runs
                .first()
                .unwrap_or_else(|| panic!("{case}: no run line"));
            assert_eq!(
                (first.0.as_str(), first.1.as_deref(), first.2.as_deref()),
                ("run", Some("plain"), Some("field")),
                "{case}: the first run line"
            );
            if forced != Some(0) {
                assert!(!outworked, "{case}: {runs:?}");
                continue;
            }
            let ran = runs.iter().position(|r| r.0 == "ran").unwrap();
            assert_eq!(
                runs[ran].1.as_deref(),
                Some("outworked"),
                "{case}: {runs:?}"
            );
            let next = &runs[ran + 1];
            assert_eq!(
                (next.0.as_str(), next.1.as_deref(), next.3.as_deref()),
                ("run", Some("signature"), Some("outworked")),
                "{case}: the start over's run line"
            );
            if runs[ran + 2..]
                .iter()
                .any(|r| r.0 == "carry" || r.2.as_deref() == Some("handover"))
            {
                handed += 1;
            }
        }
    }
    assert!(
        handed > 0,
        "no signature run after a start over handed over"
    );
}
