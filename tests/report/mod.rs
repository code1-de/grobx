//! call-report's test code, shared by its integration tests: a report text
//! read into its lines by the grammar of
//! `iterations/0064-call-report/plan.md`, _The report text_, and the checks
//! the tests make on every report: the order of its lines and its counts
//! against the outcome's.

use grobx::{Outcome, Stats};

/// One line of a report: its time, its kind, and its fields in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub time: f64,
    pub kind: String,
    pub fields: Vec<(String, String)>,
}

impl Line {
    /// The value of `key`, None where the line has no such field.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The value of `key`, which the line must have.
    pub fn value(&self, key: &str) -> &str {
        self.get(key)
            .unwrap_or_else(|| panic!("no field {key} on the line {self:?}"))
    }

    /// The integer of `key`.
    pub fn num(&self, key: &str) -> u64 {
        let v = self.value(key);
        v.parse()
            .unwrap_or_else(|_| panic!("{key}={v} is no integer on the line {self:?}"))
    }

    /// The time of `key`, in seconds.
    pub fn secs(&self, key: &str) -> f64 {
        let v = self.value(key);
        assert!(is_time(v), "{key}={v} is no time on the line {self:?}");
        v.parse().unwrap()
    }

    /// Whether the line is of `kind` and holds every field of `fields`.
    pub fn is(&self, kind: &str, fields: &[(&str, &str)]) -> bool {
        self.kind == kind && fields.iter().all(|&(k, v)| self.get(k) == Some(v))
    }
}

/// The keys of a round's eight stage times (call-report.5, .6).
pub const STAGES: [&str; 8] = [
    "select", "symbolic", "encode", "reduce", "backsub", "record", "decode", "update",
];

/// The kinds that stand between a `run` line and its `ran` line, and only
/// there.
const IN_RUN: [&str; 6] = [
    "inputs", "echelon", "round", "handover", "constant", "inter",
];

/// Whether `v` is a time: digits, a point, and six digits.
pub fn is_time(v: &str) -> bool {
    match v.split_once('.') {
        Some((s, f)) => {
            !s.is_empty()
                && s.bytes().all(|b| b.is_ascii_digit())
                && f.len() == 6
                && f.bytes().all(|b| b.is_ascii_digit())
        }
        None => false,
    }
}

/// The lines of `text`, each checked against the grammar: a time, a kind of
/// lowercase letters, then fields `key=value` with keys of lowercase
/// letters and `-`, values of no space and no `=`, each line ended by a
/// line feed.
pub fn parse(text: &str) -> Vec<Line> {
    assert!(
        text.is_empty() || text.ends_with('\n'),
        "the report does not end with a line feed: {text:?}"
    );
    let mut lines = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let n = n + 1;
        assert!(!line.contains('\t'), "line {n} holds a tab: {line:?}");
        let mut words = line.split(' ');
        let time = words.next().unwrap();
        assert!(is_time(time), "line {n}: the time {time:?} in {line:?}");
        let kind = words
            .next()
            .unwrap_or_else(|| panic!("line {n}: no kind in {line:?}"));
        assert!(
            !kind.is_empty() && kind.bytes().all(|b| b.is_ascii_lowercase()),
            "line {n}: the kind {kind:?} in {line:?}"
        );
        let mut fields = Vec::new();
        for word in words {
            let (key, value) = word
                .split_once('=')
                .unwrap_or_else(|| panic!("line {n}: {word:?} is no field in {line:?}"));
            assert!(
                !key.is_empty() && key.bytes().all(|b| b.is_ascii_lowercase() || b == b'-'),
                "line {n}: the key {key:?} in {line:?}"
            );
            assert!(
                !value.is_empty() && !value.contains('='),
                "line {n}: the value {value:?} in {line:?}"
            );
            assert!(
                fields.iter().all(|(k, _): &(String, String)| k != key),
                "line {n}: {key} twice in {line:?}"
            );
            fields.push((key.to_string(), value.to_string()));
        }
        lines.push(Line {
            time: time.parse().unwrap(),
            kind: kind.to_string(),
            fields,
        });
    }
    lines
}

/// The report of `outcome`, which must hold one, parsed, and its order
/// checked: `call` first, `read` second, `end` last; times that never
/// decrease; every `run` closed by its `ran`, of the same number, before the
/// next `run`; the run numbers 1, 2, …; the kinds of `IN_RUN` inside a run
/// alone, every other kind outside; and `stop` where the outcome stopped,
/// right before `end`, whose `result` says the same.
pub fn report(outcome: &Outcome) -> Vec<Line> {
    let text = outcome
        .report
        .as_deref()
        .unwrap_or_else(|| panic!("the call asked for the report and has none: {outcome:?}"));
    let lines = parse(text);
    assert!(lines.len() >= 3, "too few lines:\n{text}");
    assert_eq!(lines[0].kind, "call", "the first line:\n{text}");
    assert_eq!(lines[1].kind, "read", "the second line:\n{text}");
    let end = lines.last().unwrap();
    assert_eq!(end.kind, "end", "the last line:\n{text}");
    for w in lines.windows(2) {
        assert!(w[0].time <= w[1].time, "the time falls: {w:?}\n{text}");
    }
    let mut open: Option<u64> = None;
    let mut runs = 0;
    for line in &lines {
        match line.kind.as_str() {
            "run" => {
                assert_eq!(open, None, "a run inside a run: {line:?}\n{text}");
                runs += 1;
                assert_eq!(line.num("number"), runs, "the runs' numbers:\n{text}");
                open = Some(runs);
            }
            "ran" => {
                assert_eq!(open, Some(line.num("number")), "{line:?}\n{text}");
                open = None;
            }
            kind if IN_RUN.contains(&kind) => {
                assert!(open.is_some(), "{kind} outside a run: {line:?}\n{text}");
            }
            kind => assert!(open.is_none(), "{kind} inside a run: {line:?}\n{text}"),
        }
    }
    assert_eq!(open, None, "a run without its ran:\n{text}");
    if let Some(stop) = outcome.stopped {
        let bound = format!("{stop:?}").to_lowercase();
        let stop_line = &lines[lines.len() - 2];
        assert!(
            stop_line.is("stop", &[("bound", &bound)]),
            "{stop_line:?}\n{text}"
        );
        assert!(end.is("end", &[("result", "stopped")]), "{end:?}\n{text}");
        assert!(outcome.basis.is_none());
    } else {
        assert!(lines.iter().all(|l| l.kind != "stop"), "a stop:\n{text}");
        let basis = outcome
            .basis
            .as_deref()
            .expect("a basis where no bound stopped");
        assert!(end.is("end", &[("result", "basis")]), "{end:?}\n{text}");
        assert_eq!(end.num("bytes"), basis.len() as u64, "{end:?}");
    }
    end.secs("time");
    lines
}

/// The lines from run `number`'s `run` line to its `ran` line, both
/// included.
pub fn run_lines(lines: &[Line], number: u64) -> &[Line] {
    let n = number.to_string();
    let a = lines
        .iter()
        .position(|l| l.is("run", &[("number", &n)]))
        .unwrap_or_else(|| panic!("no run {number}"));
    let b = lines
        .iter()
        .position(|l| l.is("ran", &[("number", &n)]))
        .unwrap_or_else(|| panic!("no ran {number}"));
    &lines[a..=b]
}

/// The run's `round` lines: the counts each holds are the run's `Stats`
/// summed (`rounds`, `reductions`, `zero`), and its `ran` line holds them.
/// A run a bound stopped has a line for every round it counted
/// (call-report.19). Its check allows one round more, and leaves its sums
/// unchecked.
pub fn assert_run_counts(run: &[Line]) {
    let ran = run.last().unwrap();
    let rounds: Vec<&Line> = run.iter().filter(|l| l.kind == "round").collect();
    let stopped = ran.get("outcome") == Some("stop");
    let counted = ran.num("rounds");
    if stopped {
        let written = rounds.len() as u64;
        assert!(
            counted == written || counted == written + 1,
            "{counted} rounds counted, {written} written: {run:?}"
        );
        return;
    }
    assert_eq!(counted, rounds.len() as u64, "{run:?}");
    let reduced: u64 = rounds.iter().map(|l| l.num("reductions")).sum();
    let zero: u64 = rounds.iter().map(|l| l.num("zero")).sum();
    assert_eq!(ran.num("reduced"), reduced, "{run:?}");
    assert_eq!(ran.num("zero"), zero, "{run:?}");
    for (k, line) in rounds.iter().enumerate() {
        assert_eq!(line.num("number"), k as u64 + 1, "{line:?}");
        assert!(line.num("pivots") <= line.num("reductions"), "{line:?}");
        assert!(line.num("zero") <= line.num("reductions"), "{line:?}");
        assert!(line.num("given") <= line.num("rows"), "{line:?}");
        for key in STAGES {
            line.secs(key);
        }
    }
}

/// Over GF(p): the last `ran` line whose outcome is `basis` or `stop`
/// holds `stats`, and every run's counts agree with its `round` lines.
pub fn assert_stats(lines: &[Line], stats: &Stats) {
    for ran in lines.iter().filter(|l| l.kind == "ran") {
        assert_run_counts(run_lines(lines, ran.num("number")));
    }
    let last = lines
        .iter()
        .rev()
        .find(|l| l.is("ran", &[("outcome", "basis")]) || l.is("ran", &[("outcome", "stop")]))
        .unwrap_or_else(|| panic!("no run made a basis or stopped: {lines:?}"));
    assert_eq!(
        (last.num("rounds"), last.num("reduced"), last.num("zero")),
        (u64::from(stats.rounds), stats.reduced, stats.zero),
        "{last:?}"
    );
}

/// Every round's eight stage times add up to at most the time between its
/// line and the line before it, plus 10 µs for the rounding of nine times
/// to the microsecond.
pub fn assert_stage_times(lines: &[Line]) {
    for w in lines.windows(2) {
        if w[1].kind != "round" {
            continue;
        }
        let sum: f64 = STAGES.iter().map(|k| w[1].secs(k)).sum();
        assert!(
            sum <= w[1].time - w[0].time + 10e-6,
            "the stages take {sum} s of {span} s: {line:?}",
            span = w[1].time - w[0].time,
            line = w[1]
        );
    }
}

/// The lines with every time and resident set taken out: each line's kind
/// and its fields whose value holds no point and whose key is no
/// `process-rss-`. A time holds a point, and so may the `call` line's
/// `cpu`, which goes with the times.
pub fn untimed(lines: &[Line]) -> Vec<(String, Vec<(String, String)>)> {
    lines
        .iter()
        .map(|l| {
            let fields = l
                .fields
                .iter()
                .filter(|(k, v)| !v.contains('.') && !k.starts_with("process-rss-"))
                .cloned()
                .collect();
            (l.kind.clone(), fields)
        })
        .collect()
}

/// The lines of `kind`.
pub fn of<'a>(lines: &'a [Line], kind: &str) -> Vec<&'a Line> {
    lines.iter().filter(|l| l.kind == kind).collect()
}

/// `outcome` with its report taken out, for comparison with a call that
/// asked for none.
pub fn without_report(outcome: &Outcome) -> Outcome {
    Outcome {
        report: None,
        ..outcome.clone()
    }
}
