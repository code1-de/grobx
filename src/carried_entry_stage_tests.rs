//! carried-entry-stage's test code: the call report's `enter` line, the time of a run's entry of
//! the elements it starts from, the plain run's of its inputs and of its carried elements and the
//! signature run's of its inputs (`iterations/0182-carried-entry-stage/plan.md`, _Test code_).

use std::num::NonZeroUsize;

use crate::carried_run_tests::SEEDS;
use crate::gf2_input_time_tests::EDGE;
use crate::gf2_start_over_tests::{call, reported};
use crate::report_tests::Line;
use crate::systems::{boolean, random_system};
use crate::{MonomialOrder, Params, Threads};

/// The kinds that stand in a run between its `run` line and its entry.
const BEFORE: [&str; 4] = ["inputs", "substitute", "echelon", "carry"];

/// The slack of two times of the report, each truncated to the microsecond.
const SLACK: f64 = 2e-6;

/// The variables of test 3's system, each with its square as a generator.
const SQUARES: usize = 300;

/// A run's entry as its report reads: the run's engine, whether the run went on from a carry,
/// and its `enter` line's elements and pairs where it has one.
struct Entry {
    engine: String,
    carried: bool,
    /// The run's first line is `constant round=0`: an input is constant.
    constant: bool,
    entered: Option<(u64, u64)>,
}

/// The integer of `key` on `line`, which must have it.
fn num(case: &str, line: &Line, key: &str) -> u64 {
    line.get(key)
        .unwrap_or_else(|| panic!("{case}: no {key} on {line:?}"))
        .parse()
        .unwrap_or_else(|e| panic!("{case}: {key} on {line:?}: {e}"))
}

/// The time of `key` on `line`, in seconds.
fn secs(case: &str, line: &Line, key: &str) -> f64 {
    line.get(key)
        .unwrap_or_else(|| panic!("{case}: no {key} on {line:?}"))
        .parse()
        .unwrap_or_else(|e| panic!("{case}: {key} on {line:?}: {e}"))
}

/// The entry of `run`, a run's lines from its `run` line to its `ran` line, checked. A replay
/// and a run whose input is constant write no `enter` line; every other run writes at most one,
/// and one where it reached a round or ended with a basis, a carry, or a handover. The line
/// stands after the run's lines of `BEFORE` and before every other line of the run. It names as
/// elements the carry's where the run went on from one, else the rows of the run's echelon line
/// where it has one, else the nonzero generators of its inputs line. Its time is at most the
/// time from the `run` line to it and at most the run's `ran` time. A plain run's first round
/// selects at most its pairs, and a signature run's run queues at least its pairs.
fn entry(case: &str, run: &[Line]) -> Entry {
    let head = &run[0];
    let ran = run.last().unwrap();
    let engine = head
        .get("engine")
        .unwrap_or_else(|| panic!("{case}: no engine on {head:?}"))
        .to_string();
    let carry = run.iter().find(|l| l.kind == "carry");
    let constant = run.iter().any(|l| l.is("constant", &[("round", "0")]));
    let enters: Vec<usize> = (0..run.len()).filter(|&k| run[k].kind == "enter").collect();
    assert!(
        enters.len() <= 1,
        "{case}: enter lines {enters:?} in {run:?}"
    );
    let round = run.iter().find(|l| l.kind == "round");
    if engine == "replay" || constant {
        assert!(enters.is_empty(), "{case}: an enter line in {run:?}");
    }
    let ended = match ran.get("outcome") {
        Some(outcome) => ["basis", "carry", "handover"].contains(&outcome),
        None => false,
    };
    if engine != "replay" && !constant && (ended || round.is_some()) {
        assert_eq!(enters.len(), 1, "{case}: no enter line in {run:?}");
    }
    let mut entered = None;
    if let Some(&k) = enters.first() {
        let line = &run[k];
        assert!(
            run[1..k].iter().all(|l| BEFORE.contains(&l.kind.as_str())),
            "{case}: a line before the entry in {run:?}"
        );
        assert!(
            run[k + 1..]
                .iter()
                .all(|l| !BEFORE.contains(&l.kind.as_str())),
            "{case}: a line of the run's start after the entry in {run:?}"
        );
        let elements = num(case, line, "elements");
        let echelon = run[1..k].iter().find(|l| l.kind == "echelon");
        let inputs = run[1..k].iter().find(|l| l.kind == "inputs");
        let want = carry
            .map(|c| num(case, c, "elements"))
            .or_else(|| echelon.map(|e| num(case, e, "rows")))
            .or_else(|| inputs.map(|l| num(case, l, "nonzero")))
            .unwrap_or_else(|| panic!("{case}: no carry, echelon, or inputs line in {run:?}"));
        assert_eq!(elements, want, "{case}: the elements of {line:?}");
        let time = secs(case, line, "time");
        assert!(
            time <= line.time - head.time + SLACK,
            "{case}: {line:?} past its run's start {head:?}"
        );
        assert!(
            time <= secs(case, ran, "time") + SLACK,
            "{case}: {line:?} past its run's time {ran:?}"
        );
        let pairs = num(case, line, "pairs");
        match engine.as_str() {
            "plain" => {
                if let Some(r) = round {
                    assert!(
                        num(case, r, "pairs") <= pairs,
                        "{case}: {r:?} after {line:?}"
                    );
                }
            }
            "signature" => {
                if let Some(queued) = ran.get("queued") {
                    let queued: u64 = queued.parse().unwrap();
                    assert!(pairs <= queued, "{case}: {line:?} past {ran:?}");
                }
            }
            other => panic!("{case}: an enter line in a run of engine {other}"),
        }
        entered = Some((elements, pairs));
    }
    Entry {
        engine,
        carried: carry.is_some(),
        constant,
        entered,
    }
}

/// The entries of the runs of `lines`, each checked (`entry`).
fn entries(case: &str, lines: &[Line]) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(start) = lines[at..].iter().position(|l| l.kind == "run") {
        let start = at + start;
        let end = start
            + lines[start..]
                .iter()
                .position(|l| l.kind == "ran")
                .unwrap_or_else(|| panic!("{case}: a run without its ran line"));
        out.push(entry(case, &lines[start..=end]));
        at = end + 1;
    }
    assert!(
        lines[at..].iter().all(|l| l.kind != "enter"),
        "{case}: an enter line outside a run"
    );
    out
}

/// The parameters of a call in `order` on one thread under `seed`, with its report, signatures
/// as `signatures` says, and cofactors as `cofactors` says.
fn params(order: MonomialOrder, seed: u64, signatures: bool, cofactors: bool) -> Params {
    let mut params = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), seed);
    params.report = true;
    params.signatures = signatures;
    params.cofactors = cofactors;
    params
}

/// Test 1: carried-entry-stage.1, .2, .5, .6. Over GF(2), GF(65521), and ℚ, on `EDGE` in x, y,
/// and z and on `SEEDS` of `random_system`'s systems, in grevlex with signatures on and off and
/// with cofactors, and over GF(65521) in lex: every run's entry meets `entry`'s checks. Among the
/// calls a plain run enters, a signature run enters, a run's input is constant, and over ℚ a
/// replay runs, each asserted present.
#[test]
fn every_run_enters_once() {
    let (mut plain, mut signature, mut constant, mut replay) = (0, 0, 0, 0);
    for p in [2u64, 65521, 0] {
        let mut texts: Vec<String> = EDGE.iter().map(|g| format!("x,y,z\n{p}\n{g}\n")).collect();
        texts.extend((0..SEEDS).map(|seed| random_system(p, 0xc4e0 + seed)));
        let mut orders = vec![MonomialOrder::Grevlex];
        if p == 65521 {
            orders.push(MonomialOrder::Lex);
        }
        for (k, text) in texts.iter().enumerate() {
            for order in &orders {
                for (signatures, cofactors) in [(true, false), (false, false), (true, true)] {
                    let case = format!("p={p}, text {k}, {order:?}, {signatures}, {cofactors}");
                    let given = params(order.clone(), 0xc4e1, signatures, cofactors);
                    let (_, lines) = call(text, &given, None);
                    for e in entries(&case, &lines) {
                        let entered = e.entered.is_some();
                        plain += usize::from(e.engine == "plain" && entered);
                        signature += usize::from(e.engine == "signature" && entered);
                        constant += usize::from(e.constant);
                        replay += usize::from(e.engine == "replay");
                    }
                }
            }
        }
    }
    assert!(plain > 0, "no plain run entered");
    assert!(signature > 0, "no signature run entered");
    assert!(constant > 0, "no input was constant");
    assert!(replay > 0, "no replay ran");
}

/// Test 2: carried-entry-stage.1, .2, the item's acceptance criterion. Over GF(2) on
/// `boolean(n, m, seed)` for n = 6 to 13, m = n + 2 and m = 2n, with `WORK_FORCED` at Some(0), so
/// that the plain run starts over in the signature run: every run's entry meets `entry`'s
/// checks, and a plain run that goes on from a carry enters the carry's elements in a time at
/// most its run's. Such a run is asserted present.
#[test]
fn the_carried_entry() {
    let mut carried = 0;
    for n in 6..=13usize {
        for m in [n + 2, 2 * n] {
            let seed = 0xc4f0 + n as u64;
            let text = boolean(n, m, seed);
            let case = format!("boolean({n}, {m}, {seed:#x})");
            let (_, lines) = call(&text, &reported(seed, true), Some(0));
            for e in entries(&case, &lines) {
                if e.carried {
                    assert!(e.entered.is_some(), "{case}: the carry entered nothing");
                    carried += 1;
                }
            }
        }
    }
    assert!(carried > 0, "no call carried after its start over");
}

/// Test 3: carried-entry-stage.3, .4. Over GF(65521), the squares of `SQUARES` variables, every
/// pair of them coprime, with signatures on and off: the run enters every square, leaves no
/// pair, and its entry's time is above zero.
#[test]
fn the_entry_has_its_time() {
    let names: Vec<String> = (1..=SQUARES).map(|i| format!("v{i}")).collect();
    let squares: Vec<String> = names.iter().map(|v| format!("{v}^2")).collect();
    let text = format!(
        "{ring}\n65521\n{generators}\n",
        ring = names.join(","),
        generators = squares.join(",\n")
    );
    for signatures in [true, false] {
        let case = format!("squares, signatures {signatures}");
        let (_, lines) = call(&text, &reported(0xc501, signatures), None);
        let entered: Vec<(u64, u64)> = entries(&case, &lines)
            .iter()
            .filter_map(|e| e.entered)
            .collect();
        assert_eq!(entered, [(SQUARES as u64, 0)], "{case}");
        let line = lines.iter().find(|l| l.kind == "enter").unwrap();
        assert!(secs(&case, line, "time") > 0.0, "{case}: {line:?}");
    }
}
