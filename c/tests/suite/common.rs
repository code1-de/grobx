//! Test code shared by the C interface's tests: the library and the
//! programs built for this test's profile, an instance file in a scratch
//! directory, a run of the C driver, and what the crate's entry point gives
//! for the same arguments.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use grobx::{CheckerMode, Error, MonomialOrder, Outcome, Params, Stop, Threads};
use grobxc::*;

/// The directory of `grobx.h`.
pub const HEADER_DIR: &str = env!("CARGO_MANIFEST_DIR");
/// The C and C++ programs of the test code.
pub const PROGRAMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/programs");

/// The directory of `libgrobxc.so`: `deps/`, where the test binary lies,
/// and where cargo builds the cdylib with it.
pub fn lib_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let exe = std::env::current_exe().unwrap();
        let dir = exe.parent().unwrap().to_path_buf();
        assert!(
            dir.join("libgrobxc.so").is_file(),
            "{dir}/libgrobxc.so",
            dir = dir.display()
        );
        dir
    })
}

/// The program `programs/<file>`, compiled once per run against
/// `grobx.h` and linked against `libgrobxc.so`: a `.c` file by `cc`, a
/// `.cpp` file by `c++`, every warning an error.
pub fn program(file: &str) -> PathBuf {
    static BUILT: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();
    let mut built = BUILT.get_or_init(Default::default).lock().unwrap();
    if let Some(exe) = built.get(file) {
        return exe.clone();
    }
    let lib = lib_dir();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("c-programs");
    fs::create_dir_all(&dir).unwrap();
    let (stem, cpp) = match file.strip_suffix(".cpp") {
        Some(stem) => (stem, true),
        None => (file.strip_suffix(".c").expect("a .c or .cpp file"), false),
    };
    let exe = dir.join(stem);
    let mut cmd = Command::new(if cpp { "c++" } else { "cc" });
    if cpp {
        cmd.args(["-std=c++17"]);
    } else {
        cmd.args(["-std=c11", "-pthread"]);
    }
    cmd.args(["-Wall", "-Wextra", "-Werror", "-pedantic", "-I", HEADER_DIR])
        .arg(Path::new(PROGRAMS).join(file))
        .arg("-L")
        .arg(lib)
        .arg(format!("-Wl,-rpath,{lib}", lib = lib.display()))
        .arg("-lgrobxc")
        .arg("-o")
        .arg(&exe);
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("the compiler does not run ({e}): the test code needs it"));
    assert!(
        out.status.success(),
        "{file} does not compile: {status}\n{stderr}",
        status = out.status,
        stderr = String::from_utf8_lossy(&out.stderr)
    );
    built.insert(file.to_string(), exe.clone());
    exe
}

/// Runs `exe` with `args` and fails where it does not exit 0.
pub fn run(exe: &Path, args: &[String]) -> Output {
    let out = Command::new(exe)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("{exe}: {e}", exe = exe.display()));
    assert!(
        out.status.success(),
        "{exe} {args:?}: {status}\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
        exe = exe.display(),
        status = out.status,
        stdout = String::from_utf8_lossy(&out.stdout),
        stderr = String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// A directory of this test's own, empty, under cargo's scratch directory.
pub fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("c").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The file `instance.ms` in `dir`, holding `text`.
pub fn write_instance(dir: &Path, text: &str) -> PathBuf {
    let file = dir.join("instance.ms");
    fs::write(&file, text).unwrap();
    file
}

/// When the driver raises the cancel flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Raise {
    /// No flag.
    None,
    /// A flag that is never raised.
    Never,
    /// Raised before the call.
    Before,
    /// Raised by a thread of its own after this many milliseconds.
    After(u64),
}

/// The driver's arguments, and the `Params` they stand for.
#[derive(Clone, Debug)]
pub struct Args {
    pub order: MonomialOrder,
    pub threads: usize,
    pub seed: u64,
    pub deadline: Option<u64>,
    pub memory: Option<u64>,
    pub checker: Option<CheckerMode>,
    pub cofactors: bool,
    pub cancel: Raise,
}

impl Args {
    /// `order`, one thread, and `seed`, nothing else.
    pub fn new(order: MonomialOrder, seed: u64) -> Args {
        Args {
            order,
            threads: 1,
            seed,
            deadline: None,
            memory: None,
            checker: None,
            cofactors: false,
            cancel: Raise::None,
        }
    }

    /// The crate's parameters for these arguments, and a flag raised before
    /// the call where the driver raises one before.
    pub fn params(&self) -> Params {
        let threads = match std::num::NonZeroUsize::new(self.threads) {
            None => Threads::All,
            Some(n) => Threads::Count(n),
        };
        let mut params = Params::new(self.order.clone(), threads, self.seed);
        params.deadline = self.deadline.map(Duration::from_millis);
        params.memory = self.memory;
        params.checker = self.checker;
        params.cofactors = self.cofactors;
        match self.cancel {
            Raise::Before => {
                let cancel = grobx::Cancel::new();
                cancel.set();
                params.cancel = Some(cancel);
            }
            Raise::None | Raise::Never | Raise::After(_) => {}
        }
        params
    }

    /// The driver's arguments for the instance file `instance` and the
    /// out dir `out`: the order as its `GROBX_*` value and the block sizes
    /// joined by commas, `-` for an order without blocks.
    pub fn argv(&self, instance: &Path, out: &Path) -> Vec<String> {
        let or_dash = |v: Option<u64>| v.map_or("-".to_string(), |v| v.to_string());
        let (order, blocks) = match &self.order {
            MonomialOrder::Grevlex => (GROBX_GREVLEX, "-".to_string()),
            MonomialOrder::Lex => (GROBX_LEX, "-".to_string()),
            MonomialOrder::Block(sizes) => (
                GROBX_BLOCK,
                sizes
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        };
        vec![
            instance.display().to_string(),
            order.to_string(),
            blocks,
            self.threads.to_string(),
            self.seed.to_string(),
            or_dash(self.deadline),
            or_dash(self.memory),
            match self.checker {
                None => "-".to_string(),
                Some(CheckerMode::Exact) => "exact".to_string(),
                Some(CheckerMode::Cheap { points }) => points.to_string(),
            },
            if self.cofactors { "1" } else { "0" }.to_string(),
            match self.cancel {
                Raise::None => "-".to_string(),
                Raise::Never => "never".to_string(),
                Raise::Before => "set".to_string(),
                Raise::After(ms) => ms.to_string(),
            },
            out.display().to_string(),
        ]
    }
}

/// What a call through the C interface handed back, or what the crate's
/// call gives as the library maps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Driven {
    pub status: i32,
    pub stopped: i32,
    pub certified: i32,
    pub basis: Option<String>,
    pub cofactors: Option<String>,
    pub flaw: Option<String>,
    pub error: Option<String>,
}

/// The driver's run on `text` with `args`, in the scratch directory
/// `dir`: what it handed back, the basis text's digest by `grobx_digest`,
/// and `grobx_version()`.
pub fn drive(dir: &Path, text: &str, args: &Args) -> (Driven, Option<String>, String) {
    let instance = write_instance(dir, text);
    let out = dir.join("out");
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).unwrap();
    let ran = run(&program("driver.c"), &args.argv(&instance, &out));
    let stdout = String::from_utf8(ran.stdout).unwrap();
    let line = |key: &str| -> String {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key} ")))
            .unwrap_or_else(|| panic!("no line {key}: {stdout}"))
            .to_string()
    };
    let file = |name: &str| fs::read_to_string(out.join(name)).ok();
    let digest = Some(line("digest")).filter(|d| d != "-");
    let driven = Driven {
        status: line("status").parse().unwrap(),
        stopped: line("stopped").parse().unwrap(),
        certified: line("certified").parse().unwrap(),
        basis: file("basis"),
        cofactors: file("cofactors"),
        flaw: file("flaw"),
        error: file("error"),
    };
    (driven, digest, line("version"))
}

/// What the C interface hands back where the crate's call returns `got`.
pub fn mapped(got: Result<Outcome, Error>) -> Driven {
    match got {
        Ok(o) => Driven {
            status: GROBX_OK,
            stopped: match o.stopped {
                None => GROBX_STOP_NONE,
                Some(Stop::Deadline) => GROBX_STOP_DEADLINE,
                Some(Stop::Cancel) => GROBX_STOP_CANCEL,
                Some(Stop::Memory) => GROBX_STOP_MEMORY,
            },
            certified: match o.certified {
                None => GROBX_UNCHECKED,
                Some(false) => GROBX_REJECTED,
                Some(true) => GROBX_CERTIFIED,
            },
            basis: o.basis,
            cofactors: o.cofactors,
            flaw: o.flaw,
            error: None,
        },
        Err(e) => Driven {
            status: match e {
                Error::Syntax { .. } => GROBX_SYNTAX,
                Error::UnknownVariable { .. } => GROBX_UNKNOWN_VARIABLE,
                Error::RepeatedVariable { .. } => GROBX_REPEATED_VARIABLE,
                Error::Characteristic { .. } => GROBX_CHARACTERISTIC,
                Error::Overflow { .. } => GROBX_OVERFLOW,
                Error::Size => GROBX_SIZE,
                Error::NoInverse { .. } => GROBX_NO_INVERSE,
                Error::Order { .. } => GROBX_ORDER,
                Error::Memory => GROBX_MEMORY,
                Error::Drops { .. } => GROBX_DROPS,
                _ => GROBX_ERROR,
            },
            stopped: GROBX_STOP_NONE,
            certified: GROBX_UNCHECKED,
            basis: None,
            cofactors: None,
            flaw: None,
            error: Some(e.to_string()),
        },
    }
}

/// What the crate's entry point gives for `text` with `args`, as the C
/// interface hands it back.
pub fn expected(text: &str, args: &Args) -> Driven {
    mapped(grobx::groebner(text, &args.params()))
}

/// Runs the driver on `text` with `args` and asserts that it hands back
/// what the crate gives, and the basis text's digest by `grobx_digest`.
/// Returns what it handed back.
pub fn assert_matches(name: &str, text: &str, args: &Args) -> Driven {
    let dir = scratch(name);
    let (got, digest, _) = drive(&dir, text, args);
    let want = expected(text, args);
    assert_eq!(got, want, "{name}: {args:?}");
    assert_eq!(
        digest,
        got.basis.as_deref().map(grobx::digest),
        "{name}: grobx_digest"
    );
    got
}
