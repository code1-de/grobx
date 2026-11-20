//! checker-long-row-memory, the checker's test code
//! (`iterations/0109-checker-long-row-memory/plan.md`, _Test code_): a
//! row of the cofactor text with more entries than F has generators. In
//! the cheap mode the check's memory stays within its batch budget,
//! whatever the row's length, and both modes give the row `Shape`, or the
//! reading error an entry past the generators holds.

use std::num::{NonZeroU32, NonZeroUsize};
use std::process::Command;

use grobx_checker::{
    Error, Flaw, Mode, Order, Part, cheap_moduli, check_cofactors, check_cofactors_batched,
};
use num_bigint::BigUint;

use crate::cheap_verdicts::agree;
use crate::common::PER;

/// The variable that tells a child process the characteristic and the
/// number of points of the check it runs (test 1).
const CHILD: &str = "GROBX_CHECKER_LONG_ROW_CHILD";

/// The entries of test 1's row, the item's acceptance criterion.
const ENTRIES: usize = 100_000;

/// The points of test 1's check at many points.
const MANY: u32 = 100_000;

/// How far the peak of the check at `MANY` points may lie above the peak of
/// the check at one point, in KiB: 32 MiB.
const ALLOWED_KIB: u64 = 32 << 10;

/// The child's address space, in KiB: 4 GiB, so that a check whose memory
/// grows with the row's length fails its allocation and not the machine.
const ADDRESS_SPACE_KIB: u64 = 4 << 20;

/// F = {x}, G = {x}, and the row of `entries` entries `[1,0,…,0]`.
fn long_row(p: u64, entries: usize) -> (String, &'static str, String) {
    let mut row = String::from("[[1");
    for _ in 1..entries {
        row.push_str(",0");
    }
    row.push_str("]]:\n");
    (format!("x\n{p}\nx\n"), "[x]:\n", row)
}

/// The peak resident set of this process, `VmHWM` of `/proc/self/status`,
/// in KiB.
fn peak_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .expect("a VmHWM line");
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// Runs test 1 in a child process of the suite's binary, under an address
/// space of `ADDRESS_SPACE_KIB` and without a core file, on the check over
/// characteristic `p` at `points` points: the child's peak in KiB.
fn child_peak(p: u64, points: u32) -> u64 {
    let exe = std::env::current_exe().expect("the test binary knows its path");
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "ulimit -c 0 && ulimit -v {ADDRESS_SPACE_KIB} && exec \"$0\" \"$@\""
        ))
        .arg(&exe)
        .args([
            "long_rows::a_long_row_at_many_points",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, format!("{p} {points}"))
        .output()
        .unwrap_or_else(|e| panic!("sh -c {exe}: {e}", exe = exe.display()));
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "p = {p}, {points} points: the child failed: {out:?}"
    );
    assert!(said.contains("1 passed"), "the child ran no test: {said}");
    // libtest prints the test's name and the child's line on one line.
    let (_, peak) = said
        .split_once("peak-kib=")
        .unwrap_or_else(|| panic!("the child said no peak: {said}"));
    peak.split_whitespace()
        .next()
        .and_then(|kib| kib.parse().ok())
        .unwrap_or_else(|| panic!("the child's peak is no number: {said}"))
}

/// Test 1, the acceptance criterion: F = {x}, G = {x}, and a row of
/// 100,000 entries, in the cheap mode under seed 1 over GF(65521) and Q.
/// In a child process of its own, the check at one point and the check at
/// 100,000 points each return `Shape`, and the second's peak resident set
/// lies within 32 MiB of the first's.
#[test]
fn a_long_row_at_many_points() {
    if let Some(job) = std::env::var_os(CHILD) {
        let job = job.into_string().expect("the job is text");
        let (p, points) = job.split_once(' ').expect("the job is `<p> <points>`");
        let p: u64 = p.parse().expect("p is a number");
        let points: u32 = points.parse().expect("points is a number");
        let (instance, basis, row) = long_row(p, ENTRIES);
        let mode = Mode::Cheap {
            points: NonZeroU32::new(points).expect("points is at least 1"),
            seed: 1,
        };
        let got = check_cofactors(&instance, basis, &row, &Order::Grevlex, mode);
        assert_eq!(got, Ok(Err(Flaw::Shape)), "p = {p}, {points} points");
        println!("peak-kib={peak}", peak = peak_kib());
        return;
    }
    for p in [65521, 0] {
        let one = child_peak(p, 1);
        let many = child_peak(p, MANY);
        assert!(
            many <= one + ALLOWED_KIB,
            "p = {p}: the peak at {MANY} points, {many} KiB, passes the peak at one point, {one} KiB, by more than {ALLOWED_KIB} KiB"
        );
    }
}

/// The batched check at 7 points under seed 3 gives the exact mode's rows
/// stage for every batch size of `PER`.
fn batched(instance: &str, basis: &str, cofactors: &str) {
    let exact = check_cofactors(instance, basis, cofactors, &Order::Grevlex, Mode::Exact);
    let k = NonZeroU32::new(7).expect("7 is at least 1");
    for per in PER {
        let per = NonZeroUsize::new(per).expect("a batch holds a point");
        let got = check_cofactors_batched(instance, basis, cofactors, &Order::Grevlex, k, 3, per);
        assert_eq!(
            got, exact,
            "{per} points a batch: {instance:?} {cofactors:?}"
        );
    }
}

/// `agree` at 23 points, then `batched`: the rows stage's result.
fn both(instance: &str, basis: &str, cofactors: &str) -> Result<Result<(), Flaw>, Error> {
    let got = agree(instance, basis, cofactors, 23);
    batched(instance, basis, cofactors);
    got
}

/// The row `[1,0,…,0]` of `entries` entries.
fn zeros_after_one(entries: usize) -> String {
    let mut row = String::from("[1");
    for _ in 1..entries {
        row.push_str(",0");
    }
    row.push(']');
    row
}

/// Test 2: over GF(7), GF(65521), and Q, F = {x} and G = {x}, a row of 2,
/// 3, and 1,000 entries, the long row after a true one past the last
/// element, and a true row after a long one past the last; and F with no
/// generator, G = {x}, and a row of 1 and 1,000 entries: `Shape` in both
/// modes, at every batch size.
#[test]
fn long_rows_are_misshapen() {
    for p in [7, 65521, 0] {
        let instance = format!("x\n{p}\nx\n");
        let mut texts: Vec<String> = [2, 3, 1000]
            .iter()
            .map(|&n| format!("[{row}]:\n", row = zeros_after_one(n)))
            .collect();
        texts.push(format!("[[1],\n{row}]:\n", row = zeros_after_one(1000)));
        texts.push(format!("[{row},\n[1]]:\n", row = zeros_after_one(1000)));
        for cofactors in &texts {
            let got = both(&instance, "[x]:\n", cofactors);
            assert_eq!(got, Ok(Err(Flaw::Shape)), "p = {p}, {cofactors:?}");
        }
        let instance = format!("x\n{p}\n");
        for n in [1, 1000] {
            let cofactors = format!("[{row}]:\n", row = zeros_after_one(n));
            let got = both(&instance, "[x]:\n", &cofactors);
            assert_eq!(got, Ok(Err(Flaw::Shape)), "p = {p}, {cofactors:?}");
        }
    }
}

/// Test 3: an entry past the generators that does not read is the error
/// of its reading in both modes, the same as the exact mode's, at every
/// batch size: a name F does not list, a syntax error, an exponent past
/// `u32`, a denominator 0 mod p or over Q the integer 0, the entry 1,000
/// entries in, and the entry in a row past the last element.
#[test]
fn reading_errors_past_the_generators() {
    let unknown = |line| Error::UnknownVariable {
        part: Part::Cofactors,
        line,
        name: "z".to_string(),
    };
    let no_inverse = |line| Error::NoInverse {
        part: Part::Cofactors,
        line,
    };
    let overflow = |line| Error::Overflow {
        part: Part::Cofactors,
        line,
    };
    let far = zeros_after_one(1000);
    let far = &far[..far.len() - 1];
    for p in [7, 65521, 0] {
        let instance = format!("x\n{p}\nx\n");
        let denominator = if p == 0 { "1/0" } else { &format!("1/{p}") };
        let cases = [
            ("[[1,0,z]]:\n".to_string(), Some(unknown(1))),
            ("[[1,0,\nz]]:\n".to_string(), Some(unknown(2))),
            ("[[1,0,x^4294967296]]:\n".to_string(), Some(overflow(1))),
            (format!("[[1,0,{denominator}]]:\n"), Some(no_inverse(1))),
            (format!("[[1,\n0,\n{denominator}]]:\n"), Some(no_inverse(3))),
            (format!("[{far},z]]:\n"), Some(unknown(1))),
            ("[[1],\n[0,0,z]]:\n".to_string(), Some(unknown(2))),
            ("[[1,0,*]]:\n".to_string(), None),
            ("[[1,0,x+]]:\n".to_string(), None),
            (format!("[{far},1/]]:\n"), None),
        ];
        for (cofactors, error) in cases {
            let got = both(&instance, "[x]:\n", &cofactors);
            match error {
                Some(error) => assert_eq!(got, Err(error), "p = {p}, {cofactors:?}"),
                None => assert!(
                    matches!(
                        got,
                        Err(Error::Syntax {
                            part: Part::Cofactors,
                            ..
                        })
                    ),
                    "p = {p}, {cofactors:?}: {got:?}"
                ),
            }
        }
    }
}

/// The edge cases (`OBJECTIVES.md`, _Domain specifics the workflow points
/// at_) as an instance, its reduced basis, and its cofactor rows, written
/// for characteristic `p`: the zero ideal, F = (0) with two zeros, the
/// unit ideal, an input already a Gröbner basis, a zero generator, and an
/// eliminable variable.
fn edges(p: u64) -> Vec<(String, &'static str, Vec<&'static str>)> {
    [
        ("x,y\n{p}\n0\n", "[]:\n", vec![]),
        ("x,y\n{p}\n0,\n0\n", "[]:\n", vec![]),
        ("x,y\n{p}\nx*y-1,\nx\n", "[1]:\n", vec!["[-1,y]"]),
        ("x,y\n{p}\nx,\ny\n", "[y,\nx]:\n", vec!["[0,1]", "[1,0]"]),
        ("x,y\n{p}\nx,\n0\n", "[x]:\n", vec!["[1,0]"]),
        (
            "x,y\n{p}\nx-y,\ny^2-1\n",
            "[x-y,\ny^2-1]:\n",
            vec!["[1,0]", "[0,1]"],
        ),
    ]
    .into_iter()
    .map(|(instance, basis, rows)| (instance.replace("{p}", &p.to_string()), basis, rows))
    .collect()
}

/// The matrix of `rows`, each row text as it stands.
fn matrix(rows: &[String]) -> String {
    format!("[{rows}]:\n", rows = rows.join(",\n"))
}

/// Test 4: each edge case over GF(7), GF(65521), and Q is certified in
/// both modes as written, and is `Shape` in both modes, at every batch
/// size, with its last row one entry and 1,000 entries longer, and with a
/// row past the last element of one entry more than F has generators.
#[test]
fn edge_cases_with_long_rows() {
    for p in [7, 65521, 0] {
        for (instance, basis, rows) in edges(p) {
            let rows: Vec<String> = rows.iter().map(ToString::to_string).collect();
            let generators = instance.lines().skip(2).count();
            let got = both(&instance, basis, &matrix(&rows));
            assert_eq!(got, Ok(Ok(())), "{instance:?} as written");
            let mut longer = Vec::new();
            if let Some(last) = rows.last() {
                for more in [1, 1000] {
                    let mut long = rows.clone();
                    long.pop();
                    long.push(format!(
                        "{row}{zeros}]",
                        row = &last[..last.len() - 1],
                        zeros = ",0".repeat(more)
                    ));
                    longer.push(long);
                }
            }
            let mut past = rows.clone();
            past.push(format!("[1{zeros}]", zeros = ",0".repeat(generators)));
            longer.push(past);
            for long in longer {
                let cofactors = matrix(&long);
                let got = both(&instance, basis, &cofactors);
                assert_eq!(got, Ok(Err(Flaw::Shape)), "{instance:?} {cofactors:?}");
            }
        }
    }
}

/// Test 5: over Q, F = {x} and G = {x}, a row whose entry past the
/// generators has a denominator that the modulus of every point `both`
/// draws divides, and a long row whose first entry has it: `Shape` in both
/// modes.
#[test]
fn vanishing_denominators_past_the_generators() {
    let (instance, basis) = ("x\n0\nx\n", "[x]:\n");
    let k = NonZeroU32::new(23).expect("23 is at least 1");
    let mut product = BigUint::from(1u32);
    for seed in 0..8 {
        let moduli =
            cheap_moduli(instance, basis, &Order::Grevlex, k, seed).expect("F = {x} over Q reads");
        for q in moduli {
            product *= q;
        }
    }
    for cofactors in [
        format!("[[1,0,1/{product}]]:\n"),
        format!("[[1+1/{product}-1/{product},0]]:\n"),
    ] {
        let got = both(instance, basis, &cofactors);
        assert_eq!(got, Ok(Err(Flaw::Shape)), "{cofactors:?}");
    }
}
