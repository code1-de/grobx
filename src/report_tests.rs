//! call-report, sub-item 1, the crate's test code: the report's lines and
//! stage times alone, the budget's counts of the units spent as subtracted
//! and as added, and what the handover's tests read
//! (`iterations/0064-call-report/plan.md`, _Test code_, 1, 2, and 4).

use std::num::NonZeroUsize;
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::budget::Budget;
use crate::f4::Handover;
use crate::report::{Report, Stage};
use crate::{MonomialOrder, Params, Threads};

/// One line of a report: its time, its kind, and its fields.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Line {
    pub(crate) time: f64,
    pub(crate) kind: String,
    pub(crate) fields: Vec<(String, String)>,
}

impl Line {
    pub(crate) fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub(crate) fn is(&self, kind: &str, fields: &[(&str, &str)]) -> bool {
        self.kind == kind && fields.iter().all(|&(k, v)| self.get(k) == Some(v))
    }
}

/// Whether `v` is a time: digits, a point, and six digits.
fn is_time(v: &str) -> bool {
    v.split_once('.').is_some_and(|(s, f)| {
        !s.is_empty()
            && s.bytes().all(|b| b.is_ascii_digit())
            && f.len() == 6
            && f.bytes().all(|b| b.is_ascii_digit())
    })
}

/// The lines of the report `text`, each checked against the grammar of
/// `plan.md`, _The report text_, and their times checked never to
/// decrease.
pub(crate) fn lines(text: &str) -> Vec<Line> {
    assert!(text.is_empty() || text.ends_with('\n'), "{text:?}");
    let mut out: Vec<Line> = Vec::new();
    for line in text.lines() {
        let mut words = line.split(' ');
        let time = words.next().unwrap();
        assert!(is_time(time), "the time of {line:?}");
        let kind = words.next().expect("a kind");
        assert!(
            !kind.is_empty() && kind.bytes().all(|b| b.is_ascii_lowercase()),
            "the kind of {line:?}"
        );
        let mut fields = Vec::new();
        for word in words {
            let (k, v) = word.split_once('=').expect("a field");
            assert!(
                !k.is_empty() && k.bytes().all(|b| b.is_ascii_lowercase() || b == b'-'),
                "the key {k:?} of {line:?}"
            );
            assert!(
                !v.is_empty() && !v.contains('='),
                "the value {v:?} of {line:?}"
            );
            fields.push((k.to_string(), v.to_string()));
        }
        let time: f64 = time.parse().unwrap();
        if let Some(last) = out.last() {
            assert!(last.time <= time, "the time falls at {line:?}");
        }
        out.push(Line {
            time,
            kind: kind.to_string(),
            fields,
        });
    }
    out
}

/// Test 1, call-report.1, .7, .8: a report alone. Its lines follow the
/// grammar in the order written, with times that never decrease; a time is
/// seconds truncated to six decimals; `mark` adds the time since the last
/// mark, or since `open`, to its stage alone; `open` zeroes every stage and
/// notes the counts that `grown` measures from.
#[test]
fn a_report_alone() {
    let start = Instant::now();
    let mut report = Report::new(start, None);
    report.line(
        "call",
        format_args!(" entry=groebner seed={seed}", seed = 7),
    );
    report.line("read", format_args!(" variables=3 characteristic=65521"));
    report.open((10, 20));
    sleep(Duration::from_millis(3));
    report.mark(Stage::Select);
    let select = report.time(Stage::Select);
    assert!(select >= Duration::from_millis(3), "{select:?}");
    assert_eq!(report.time(Stage::Reduce), Duration::ZERO);
    sleep(Duration::from_millis(2));
    report.mark(Stage::Reduce);
    assert_eq!(
        report.time(Stage::Select),
        select,
        "a mark adds to its stage alone"
    );
    assert!(report.time(Stage::Reduce) >= Duration::from_millis(2));
    report.mark(Stage::Select);
    assert!(report.time(Stage::Select) >= select, "a stage adds up");
    assert_eq!(report.grown((15, 26)), (5, 6));
    report.open((15, 26));
    for stage in [Stage::Select, Stage::Reduce, Stage::Update, Stage::Text] {
        assert_eq!(report.time(stage), Duration::ZERO, "open zeroes {stage:?}");
    }
    assert_eq!(report.grown((15, 26)), (0, 0));
    report.line("end", format_args!(" result=basis bytes=12"));
    let text = report.finish();
    let read = lines(&text);
    let kinds: Vec<&str> = read.iter().map(|l| l.kind.as_str()).collect();
    assert_eq!(kinds, ["call", "read", "end"], "{text}");
    assert!(
        read[0].is("call", &[("entry", "groebner"), ("seed", "7")]),
        "{text}"
    );
    assert!(read[2].time >= 0.005, "{text}");
    assert!(
        read[2].time <= start.elapsed().as_secs_f64() + 1e-6,
        "{text}"
    );

    assert_eq!(
        crate::report::Secs(Duration::from_nanos(1_234_567_890)).to_string(),
        "1.234567"
    );
    assert_eq!(crate::report::Secs(Duration::ZERO).to_string(), "0.000000");
    assert_eq!(
        crate::report::Secs(Duration::from_micros(61_000_001)).to_string(),
        "61.000001"
    );
}

/// The `call` line's `cpu`: a value the line's grammar takes, and on
/// x86-64 always present, since every x86-64 part has the brand leaves.
#[test]
fn the_cpu_brand() {
    let brand = crate::report::cpu();
    if cfg!(target_arch = "x86_64") {
        let brand = brand.expect("x86-64 has a brand string");
        assert!(!brand.is_empty(), "{brand:?}");
        assert!(
            !brand.contains(char::is_whitespace) && !brand.contains('=') && !brand.contains('\0'),
            "{brand:?}"
        );
    } else {
        assert_eq!(brand, None);
    }
}

/// The `end` line's resident sets: 64 MiB written after the report began
/// raise the peak above the start by at least that much.
#[test]
fn the_resident_sets() {
    let report = Report::new(Instant::now(), None);
    let size = 64 << 20;
    let block = vec![1u8; size];
    let (start, peak) = report.rss();
    assert!(start > 0, "no resident set at the start");
    assert!(
        peak - start >= size,
        "peak {peak} start {start} below {size}"
    );
    drop(std::hint::black_box(block));
}

fn budget() -> Budget {
    let p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    Budget::new(Instant::now(), &p).unwrap()
}

/// Test 2, call-report.4: the budget counts the units spent as subtracted
/// and as added, apart from those spent otherwise, across full checks.
#[test]
fn the_budget_counts_its_row_operations() {
    let mut b = budget();
    b.set_work(100);
    assert_eq!(b.counts(), (0, 0));
    let mut want = (0u64, 0u64);
    let mut all = 0u64;
    for k in 0..1000usize {
        let units = 1 + k % 37;
        match k % 3 {
            0 => {
                b.spend(units).unwrap();
            }
            1 => {
                b.spend_subtracted(units).unwrap();
                want.0 += units as u64;
            }
            _ => {
                b.spend_added(units).unwrap();
                want.1 += units as u64;
            }
        }
        all += units as u64;
    }
    assert_eq!(b.counts(), want);
    assert_eq!(b.spent(), all, "every unit is spent");
    assert!(b.checks() > 100, "the counts cross full checks");
}

/// `after_round` on a new or given `Handover`, with a `divisible` that
/// returns `n`, and what its tests read.
fn read(
    h: &mut Handover,
    early: bool,
    gap: bool,
    (made, zeros, n): (usize, usize, usize),
) -> (
    bool,
    Option<bool>,
    Option<bool>,
    Option<bool>,
    Option<usize>,
) {
    let handed = h.after_round(early, gap, made, zeros, || n);
    let t = h.tests();
    (handed, t.early, t.gap, t.outgrown, t.divisible)
}

/// Test 4, call-report.12: what `Handover::after_round` records of the
/// tests that read a round, by early-handover-time.5 and handover-time.2
/// to .5 and .7: the early test
/// reads the early round, the gap
/// test the gap round where the early test stayed, the outgrown test
/// every round that made an element where neither handed over
/// (outgrown-round-size.4), and the count where a test reads it. Every
/// `Handover` here is one whose plain run would reduce each row alone.
#[test]
fn the_handover_records_its_tests() {
    // No test reads a round that made no element.
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, false, false, (0, 0, 0)),
        (false, None, None, None, None)
    );
    // The outgrown test reads a small round that is neither early nor a
    // gap round: 99 in 99 hand over.
    assert_eq!(
        read(&mut Handover::new(false), false, false, (99, 1, 99)),
        (true, None, None, Some(true), Some(99))
    );
    // A round of 100 elements: 79 in 100 stay, 80 hand over.
    assert_eq!(
        read(&mut h, false, false, (100, 0, 79)),
        (false, None, None, Some(false), Some(79))
    );
    assert_eq!(
        read(&mut h, false, false, (100, 0, 80)),
        (true, None, None, Some(true), Some(80))
    );
    // The early round: too few zeros hand over before the count.
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, true, false, (100, 24, 100)),
        (true, Some(true), None, None, None)
    );
    // The early round stays, and the outgrown test reads the same count.
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, true, false, (100, 25, 60)),
        (false, Some(false), None, Some(false), Some(60))
    );
    // A second early round: the early test reads no more.
    assert_eq!(
        read(&mut h, true, false, (100, 25, 61)),
        (false, None, None, Some(false), Some(61))
    );
    // The gap round: fewer zeros than half the elements hand over.
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, false, true, (100, 49, 0)),
        (true, None, Some(true), None, None)
    );
    // Half of them stay unless more than 6 in 10 are divisible, and the
    // outgrown test reads the same count.
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, false, true, (99, 50, 59)),
        (false, None, Some(false), Some(false), Some(59))
    );
    // The gap round decides once: the gap test reads no more, and the
    // outgrown test hands over.
    assert_eq!(
        read(&mut h, false, true, (99, 0, 99)),
        (true, None, None, Some(true), Some(99))
    );
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, false, true, (99, 50, 60)),
        (true, None, Some(true), None, Some(60))
    );
    // One round both early and the gap round: the early test first, then
    // the gap test, then the outgrown test, on one count.
    let mut h = Handover::new(false);
    assert_eq!(
        read(&mut h, true, true, (100, 50, 60)),
        (false, Some(false), Some(false), Some(false), Some(60))
    );
}
