//! unreached-test-paths, the item's test code through the CLI
//! (`iterations/0151-unreached-test-paths/plan.md`, _Test code_, tests 3 and 4): the checker's
//! points stop at a deadline that falls in the check, and a SIGINT that the process inherited
//! ignored stays ignored while its call runs (cli.20).

#[cfg(unix)]
use std::ffi::OsStr;
#[cfg(unix)]
use std::fs;
use std::num::{NonZeroU32, NonZeroUsize};
use std::time::{Duration, Instant};

use grobx::{CheckerMode, MonomialOrder, Params, Threads};

use crate::common::{crate_call, grobx, verdict};
#[cfg(unix)]
use crate::common::{finish, ignoring_sigint, scratch, sigint, until, write_instance};
use crate::systems::dense;

/// How long a call that its deadline stops may take, its start included.
const PATIENCE: Duration = Duration::from_secs(10);

// unreached-test-paths.2 design-checked -- test 4's SIGINT comes once the process has spent IN_THE_CALL of CPU time, not after a sleep from its start, which a loaded machine can leave inside the read, nor once the watch blocks SIGINT, which a process that inherited SIGINT ignored never does (cli.20) (log 0151-1)
/// The CPU time a process has spent once its call runs: its read of an instance file takes a
/// thousandth of it.
#[cfg(unix)]
const IN_THE_CALL: Duration = Duration::from_millis(200);

// unreached-test-paths.1 design-checked -- the checker's points meet their deadline on a system whose run ends in milliseconds, beside the same call at 2 points, certified, not on the system of `checker_points.rs`, whose run the deadline stops before the check starts, nor with a later deadline, which a run on a loaded machine can outlast (log 0151-1)
/// The systems whose run ends in milliseconds, so that a deadline of 300 ms falls in the
/// checker's points: a dense cubic system on three variables over GF(65521) and over ℚ.
fn quick() -> [(&'static str, String); 2] {
    [
        ("cubic-3-p65521", dense(3, 3, 65521, 1)),
        ("cubic-3-p0", dense(3, 3, 0, 1)),
    ]
}

/// The crate's call of `text` in grevlex on one thread under seed 7, with the cheap checker at
/// `points` where Some.
fn params(points: Option<u32>) -> Params {
    let mut p = Params::new(MonomialOrder::Grevlex, Threads::Count(NonZeroUsize::MIN), 7);
    p.checker = points.map(|k| CheckerMode::Cheap {
        points: NonZeroU32::new(k).expect("a count of points is not 0"),
    });
    p
}

/// Test 3: `--checker 4294967295 --deadline 300` on each quick system stops in the check: exit 1,
/// `grobx: stopped: deadline`, no basis, within `PATIENCE`. `--checker 2` with the same deadline
/// writes the crate's basis and its verdict, certified.
#[test]
fn the_checkers_points_stop_at_the_deadline() {
    for (name, text) in quick() {
        let start = Instant::now();
        let ran = grobx(
            &[
                "--seed",
                "7",
                "--checker",
                "4294967295",
                "--deadline",
                "300",
            ],
            &text,
        );
        let took = start.elapsed();
        assert_eq!(ran.code(), 1, "{name}: {ran:?}");
        assert_eq!(ran.stderr, "grobx: stopped: deadline\n", "{name}: {ran:?}");
        assert!(ran.stdout.is_empty(), "{name}: {ran:?}");
        assert!(took < PATIENCE, "{name}: stopped after {took:?}");
        let ran = grobx(
            &["--seed", "7", "--checker", "2", "--deadline", "300"],
            &text,
        );
        let want = crate_call(&text, &params(Some(2)));
        assert_eq!(want.certified, Some(true), "{name}: {want:?}");
        assert_eq!(ran.code(), 0, "{name}: {ran:?}");
        assert_eq!(ran.stderr, verdict(&want), "{name}: {ran:?}");
        assert_eq!(Some(ran.stdout), want.basis, "{name}");
    }
}

/// The CPU time, user and system, that process `pid` has spent: fields 14 and 15 of
/// `/proc/<pid>/stat`, which count clock ticks.
#[cfg(unix)]
fn cpu_time(pid: u32) -> Duration {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    // The fields after the command name, which stands in parentheses and may hold spaces, start
    // with field 3.
    let (_, rest) = stat
        .rsplit_once(')')
        .expect("the command name's parenthesis");
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks: u64 = fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap();
    // SAFETY: sysconf reads a configuration value and changes nothing.
    let per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let per_second = u64::try_from(per_second).expect("clock ticks a second");
    Duration::from_millis(ticks * 1000 / per_second)
}

/// Test 4, cli.20: a process that inherited SIGINT ignored, on an instance file of
/// `dense(8, 3, 65521, 1)`, whose call runs for seconds, takes a SIGINT once it has spent
/// `IN_THE_CALL` of CPU time, inside its call. The call goes on: exit 0, the crate's basis, and
/// nothing on standard error.
#[cfg(unix)]
#[test]
fn an_inherited_ignored_sigint_stays_ignored_in_the_call() {
    let text = dense(8, 3, 65521, 1);
    let long = write_instance(&scratch("sigint-ignored-call"), "long", &text);
    let child = ignoring_sigint(&[OsStr::new("--seed"), "1".as_ref(), long.as_os_str()])
        .spawn()
        .expect("the binary starts");
    let pid = child.id();
    until("the call running", || cpu_time(pid) >= IN_THE_CALL);
    sigint(&child);
    let ran = finish(child, Duration::from_secs(120));
    let mut p = params(None);
    p.seed = 1;
    assert_eq!(ran.code(), 0, "{ran:?}");
    assert_eq!(ran.stderr, "", "{ran:?}");
    assert_eq!(Some(ran.stdout), crate_call(&text, &p).basis);
}
