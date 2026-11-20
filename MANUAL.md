# grobx manual

grobx computes reduced Gröbner bases of polynomial ideals over prime fields
GF(p), p a prime below 2^64, and over the rationals ℚ. It is a Rust crate
with a Python extension module, a command-line program, `grobx`, and a C
interface, the library `grobxc` and its header `grobx.h`. All
four take the ideal's generators as text and return the basis as text.
The Maple package `Grobx` runs the command-line program (_Maple_ below).
The Sage module `grobx_sage` calls the Python module (_Sage_ below).
The WebAssembly module `grobx-wasm` runs the crate in a web page (_Browser_ below).

grobx runs on Linux and Windows on x86-64 and on macOS on Apple Silicon. The memory ceiling
reads the resident set from `/proc/self/statm` on Linux, by `proc_pidinfo`
on macOS, and as the working set on Windows, so a call with a ceiling
returns `Error::Memory` where the system gives none.
The call report reads the process's peak from `/proc/self/status` on
Linux, by `getrusage` on macOS, and as the peak working set on Windows,
and its `end` line holds no resident-set field where the system gives
none (_Report text_ below).
In a browser, as `wasm32-unknown-unknown`, the system gives no resident set and no clock, so a call
there with a ceiling returns `Error::Memory`, and one with a deadline or the call report
`Error::Clock`.

This manual is for a caller of the library. _Interfaces_ says how to call
it, _Formats_ what goes in and what comes out, and _Computation_ how a call
behaves. _Design_ says which big decisions shape the engine and why.

## Interfaces

### Rust

The crate is `grobx`. Its entry points take the instance text and a
`Params`:

```rust
let text = "x,y,z\n0\nx+y+z,\nx*y+y*z+z*x,\nx*y*z-1\n";
let outcome = grobx::groebner(text, &grobx::Params::default())?;
assert_eq!(outcome.basis.as_deref(), Some("[x+y+z,\ny^2+y*z+z^2,\nz^3-1]:\n"));
```

- `groebner(text, &params) -> Result<Outcome, Error>` computes the reduced
  Gröbner basis of the instance `text` (_Instance text_ below).
- `groebner_with(text, &params, &mut trace)` is `groebner`, and the `trace::Trace` it is given
  (`trace::NoTrace` listens to nothing) hears the source of every row of every matrix
  the run's rounds and its inter-reduction build, the generators' echelon
  form not among them, and which rows made a new element. A trace that listens
  has the run reduce each row of its matrices on its own, so `stats` may
  differ from `groebner`'s. Over ℚ it hears the matrices of every run of
  the lift in turn, a pair learn's and a pair replay's once for both their
  primes.
- `learn(text, &params, &mut trace)` is `groebner_with`, and it returns, beside the outcome,
  the `trace::Record` of the matrices that made new elements, or `None` where the call
  returns no basis. Over GF(p) alone.
  Where `trace` does not listen and the call tracks no cofactors, the
  learn's plain run rules rows zero by random combinations (_Rows ruled
  zero by random combinations_ below), so `stats` may differ from a
  learn's whose trace listens.
- `replay(text, &params, &record, &mut trace)` computes the basis of the same
  system over another prime from the record, building only the rows the
  record names. It returns `Error::Misfit` where the record does not fit the
  system over that prime. Over GF(p) alone.
- `digest(text) -> String` is SHA-256 of the text, 64 lowercase hex digits.
- `text::hand_over(text, chunk, put) -> io::Result<()>` hands the `String`
  `text` to `put(offset, bytes)` in chunks of at most `chunk` bytes, in
  order, and gives back to the system the whole pages of the text behind
  each chunk, so that the text and what `put` builds of it stand about
  once. It checks each chunk ASCII before it hands it. It returns
  `InvalidData` at a chunk that holds a byte past ASCII, which it does not
  hand, `Unsupported` where the page size cannot be read, and the system's
  error where it refuses the pages back. The Python module copies its long texts by
  it (_Python_ below).
- `VERSION` is the crate version.

`Params::new(order, threads, seed)` sets every bound absent, no checker, no
cofactors, signatures on, the echelon on, and no report. `Params::default()` is
`Params::new(MonomialOrder::Grevlex, Threads::All, random_seed())`, and
`random_seed()` draws a seed from the operating system's random source. Its fields:

| Field | Type | Meaning |
|---|---|---|
| `order` | `MonomialOrder` | The monomial order: `Grevlex`, `Deglex`, `Lex`, or `Block(vec![k1, …, km])` (_Monomial orders_ below). |
| `threads` | `Threads::All` or `Threads::Count(NonZeroUsize)` | How many threads the call may use. The engine computes on one thread whatever this allows. |
| `seed` | `u64` | The one source of every random choice in the call. The same text, parameters, and seed compute the same basis along the same path. A drawn seed is read back from the field to run the call again. |
| `deadline` | `Option<Duration>` | A time limit on the whole call, the checker included. |
| `cancel` | `Option<Cancel>` | A flag another thread raises to stop the call. |
| `memory` | `Option<u64>` | A ceiling on the process's resident set, in bytes. |
| `checker` | `Option<CheckerMode>` | Has the checker rule on the result: `Exact`, or `Cheap { points }` at that many random points (_Checker_ below). |
| `cofactors` | `bool` | Whether the outcome carries the cofactor matrix (_Cofactor text_ below). |
| `signatures` | `bool` | Whether a run prunes pairs by signatures. |
| `echelon` | `bool` | Whether a run takes the generators in reduced row echelon form before it computes. |
| `report` | `bool` | Whether the outcome carries the call report (_Report text_ below). |
| `report_sink` | `Option<ReportSink>` | Where `report` is on, `ReportSink::new(writer)` of a `Write + Send + 'static` that takes each report line as the call meets it, flushed after the line, so a call that dies leaves its report so far. The first write error ends the writing, and the outcome's `sink_error` holds it; the outcome's report holds every line either way, save the `alloc` lines, which the sink alone takes (_Report text_). |

`signatures` chooses the engine. On, the run prunes pairs by the signatures
of its rows and hands the call to the plain F4 engine, started over, where
the signatures stop paying. Over GF(p), a call that asks for cofactors or the checker
never hands over. Over ℚ the learn runs and the verification's run of such a call
track no cofactors and may hand over.
Over GF(2), a call that asks for neither may start in the plain F4 engine (_Signatures_, **The
plain run first over GF(2)**).
Off, the call runs the plain F4 engine. A block order,
and a lex call that runs F4 in lex, run the plain engine whatever
`signatures` says.

`echelon` puts the generators in reduced row echelon form, as vectors
over their monomials, and substitutes the variables its linear generators lead, before a run
computes (_Matrices and reduction_, **The inputs**, **The linear generators substituted**). A run
that tracks cofactors takes the generators as read whatever `echelon` says, and so does every run
over ℚ of a call that asks for cofactors or the checker, and the plain F4 runs over GF(2). A replay
takes them as its record's learn run took them, and a replay that tracks cofactors of a record
whose learn run took the echelon returns `Error::Misfit`. Off, every run but a replay takes the
generators as read.

`Cancel::new()` makes a flag. `set()` raises it, and `is_set()` reads it.
A call checks the flag lazily, as it checks the deadline: it stops at its
next check, not at once.

The call returns an `Outcome`:

| Field | Type | Meaning |
|---|---|---|
| `basis` | `Option<String>` | The basis text (_Basis text_ below); `None` where a bound stopped the call. |
| `stopped` | `Option<Stop>`: `Deadline`, `Cancel`, or `Memory` | The bound that stopped the call. |
| `cofactors` | `Option<String>` | The cofactor text, where the call asked for it and finished. |
| `certified` | `Option<bool>` | The checker's verdict; `None` where the checker did not rule: the call did not ask for it, or a bound stopped the call. |
| `flaw` | `Option<String>` | Why the checker rejected the result. |
| `stats` | `Stats` | What the run did: `rounds`, the matrices its rounds reduced, the inter-reduction's not counted; `reduced`, the reductions it ran, rows or random combinations of rows where the plain run or the signature run combines or tests them; `zero`, those that reduced to zero. Where the plain run goes on from the signature run's work, both runs count. |
| `report` | `Option<String>` | The call report (_Report text_ below), where the call asked for it; `None` where it did not. |
| `sink_error` | `Option<String>` | The message of the first error `report_sink`'s writer returned, after which the call wrote no line to it; `None` where every line reached it, or the call had no sink or no report. |

`Stop::name()` is the bound's name, `"deadline"`, `"cancel"`, or
`"memory"`, as the call report, the Python module, and the CLI write it.

A rejected result is returned with `certified` false. It is not an error.
A call that returns an `Error` returns no report.

`Allocator` hands every request to the system's allocator and writes the
call report's `alloc` lines (_Report text_ below).
The Python module, the C library, and the CLI set it as their
global allocator. A Rust program sets it by

```rust
#[global_allocator]
static GLOBAL: grobx::Allocator = grobx::Allocator;
```

A program that sets no global allocator runs the crate under its own, and
its call reports have no `alloc` line.

On Linux, a call whose result is complete gives glibc's free memory back to the system before it
writes its texts, by `malloc_trim(0)` where the C library is glibc (_Free memory given back before
the texts_ below). Memory the caller freed
before the call goes back with the run's, and the process's next allocations take their pages
afresh. A call that a bound stops before its texts does not, save that a memory stop gives the free
memory back (_Bounds_ below).

The Cargo features `cofactors` and `checker` are on by default. A build without `cofactors` leaves
out the cofactor tracking, which takes a copy of every run of its own, and one without `checker`
leaves out the checker and its crate `grobx-checker`. `checker` takes `cofactors` along, since the
checker rules on the cofactors. A call that asks for what its build leaves out returns
`Error::NotBuilt` before it reads the instance. The Python module, the C library, the CLI, and the
browser's module pass both features on: `--no-default-features` builds any of them without either,
and `--no-default-features --features cofactors` without the checker alone.

### Python

The Python extension module `grobx` wraps `groebner`:

```python
import grobx

r = grobx.groebner("x,y,z\n0\nx+y+z,\nx*y+y*z+z*x,\nx*y*z-1\n")
r.basis    # '[x+y+z,\ny^2+y*z+z^2,\nz^3-1]:\n'
r.digest   # SHA-256 of r.basis, 64 lowercase hex digits
```

`grobx.groebner(text, *, order="grevlex", threads=0, seed=None, deadline=None, cancel=None,
memory=None, checker=None, cofactors=False, report=False, report_file=None)`
takes the fields of `Params`
(_Rust_ above) as keywords. Every parameter but `text` is a keyword, and
`seed=None` draws one, as `Params::default()` does. Where a keyword's type
differs from the field's:

| Keyword | Type |
|---|---|
| `order` | `str`, the order's name (_Monomial orders_ below) |
| `threads` | `int`; 0 is all the machine has |
| `seed` | `int`, 0 to 2^64 − 1, or `None` |
| `deadline` | `int` in milliseconds, or `None` |
| `cancel` | `grobx.Cancel` or `None` |
| `checker` | `"exact"`, an `int` from 1 to 2^32 − 1 for the cheap mode's points, or `None` |
| `report_file` | a path, `str` or `os.PathLike`, or `None`; the file `report_sink` writes to |

`signatures` and `echelon` are no keywords: a Python call runs with both
on. `report_file` turns `report` on. The call creates the file once it has read `order` and
`checker`, which raise before it, and writes each report line there as it meets it, unbuffered,
so a call that raises or never returns leaves its report so far. A call
that raises leaves it ending with its `error` line. A file that cannot be
created raises `OSError`.

The call releases the GIL while it computes, so another Python thread can
raise `cancel`. `grobx.Cancel()` has the methods of `Cancel`.

The call returns a `grobx.Result`. Its read-only attributes are the fields
of `Outcome` but `stats`, with `stopped` as the string `"deadline"`,
`"cancel"`, or `"memory"`. Two more attributes: `digest` is SHA-256 of
`basis`, computed when it is read, and `seed` is the seed the call ran under.

The call makes `basis` and `cofactors` `str`s after the core's call has returned, outside its
bounds, a long one by `text::hand_over`, so that the call's peak holds each text about once.
A `str` that cannot be made raises Python's error, `MemoryError` among them, a text that cannot be
handed over raises `OSError`, and a `str` that does not read back as its text raises `RuntimeError`.

An instance or a parameter the call cannot take raises `grobx.Error`, a
subclass of `ValueError`. Its message opens with the name of the error and
a colon, `Syntax:`, `UnknownVariable:`, and so on (_Errors_ below). A
`checker` that is a `bool`, or neither `"exact"` nor an integer from 1 to
2^32 − 1, raises
`ValueError`, `TypeError`, or `OverflowError`. An integer keyword out of its
type's range, such as `seed=-1`, raises `OverflowError`.

`grobx.__version__` is the crate version.

### CLI

The command-line program `grobx`, the binary of the package `grobx-cli`,
calls `groebner` once:

```sh
grobx instance.ms > basis.txt
printf 'x,y,z\n0\nx+y+z,\nx*y+y*z+z*x,\nx*y*z-1\n' | grobx --checker exact --cofactors c.txt
```

`grobx [SWITCH]... [FILE]` reads the instance text from FILE, or from
standard input where FILE is absent or `-`, and writes the basis text to
standard output. At most one FILE is given. Field and prime are the
instance text's second line.

| Switch | Value | Default | Sets |
|---|---|---|---|
| `--order NAME` | `grevlex`, `deglex`, `lex`, or `block:k1,…,km` | `grevlex` | `order` |
| `--threads N` | a number of threads; 0 is all the machine has | 1 | `threads` |
| `--seed N` | 0 to 2^64 − 1 | drawn at random, and written to standard error | `seed` |
| `--deadline TIME` | milliseconds, or a number followed by `s`, `m`, or `h` for seconds, minutes, or hours, at most 2^64 − 1 milliseconds | none | `deadline` |
| `--memory BYTES` | bytes, or a number followed by `k`, `m`, `g`, or `t`, in either case, for KiB, MiB, GiB, or TiB, at most 2^64 − 1 bytes | none | `memory` |
| `--checker MODE` | `exact`, or a number of points from 1 to 2^32 − 1 | none | `checker` |
| `--cofactors FILE` | a path | none | `cofactors`, and the file the cofactor text goes to |
| `--report FILE` | a path | none | `report`, and the file the call report goes to |
| `--output FILE` | a path | standard output | the file the basis text goes to |
| `-h`, `--help` | | | prints `grobx`, the crate version, and the copyright of Code1 GmbH on one line, then the usage, the instance format with an example, the switches, what Ctrl-C does, and the exit statuses, and exits 0 |
| `-V`, `--version` | | | prints `grobx` and the crate version and exits 0 |

The switches set the fields of `Params` (_Rust_ above) that the keywords of
`grobx.groebner` set, save `cancel`, which SIGINT raises (below), and
`signatures` and `echelon` stay on. A number is decimal digits
alone, with no sign and no suffix but those of `--deadline` and `--memory`.
A switch takes its value as `--name
VALUE` or `--name=VALUE`. In the first form VALUE is the next argument,
whatever it begins with. `--` ends the switches: `grobx -- -x.ms`
reads the file `-x.ms`. A switch given twice is an error. The arguments are
read in order, and `--help` and `--version` act where the reading reaches
them: what follows is not read.

The files `--output` and `--cofactors` name are created after the instance
is read and before the call. A run that writes no basis, stopped or
refused, leaves them empty.
The file `--report` names is created with them. Each report line goes
there as the call meets it, so a process that dies leaves the report so
far, and a call that returns writes nothing after its `end` line. Where
`grobx` cannot write a line there while the call runs, it says so, and
the exit status stays the call's where the file is a regular file, which
it then writes whole again from the outcome's report, without `alloc` lines,
and is 2 where the file is a device or a pipe, whose report is then
incomplete. A
stopped call writes its report there. A call the crate refuses leaves
its report there, ending with its `error` line.
Two lines stand above the call report (_Report text_ below): `version`
and the crate version, then `command grobx` and the arguments as given,
each in single quotes where it is empty or holds more than ASCII letters,
digits, and `-_./:=,+@%`. An argument that is not UTF-8 stands with U+FFFD
for each sequence that is not. The lines read:

```
version <a.b.c>
command grobx --seed 7 --report r.txt instance.ms
0.000042 call entry=groebner order=grevlex seed=7 ...
```

Each message to standard error is one line that opens with `grobx: `,
save where a path or an argument it echoes holds a line break:

| When | Line |
|---|---|
| `--seed` is absent, before the call. | `grobx: seed <n>`, the seed drawn |
| A bound stopped the call. | `grobx: stopped: deadline`, `memory`, or `cancel` |
| The checker certified the result. | `grobx: checker: certified` |
| The checker rejected the result. | `grobx: checker: rejected`, and `: ` and its flaw where it names one |
| The arguments cannot be read. | what is wrong, then `grobx: --help lists the switches` |
| The instance cannot be read, or is not UTF-8. | `grobx: <input>: <reason>`, `<input>` the path or `standard input` |
| An output file cannot be created or written, or the report file's kind cannot be read. | `grobx: <path>: <reason>` |
| A report line cannot be written to the report file while the call runs. | `grobx: <path>: <reason>` |
| Standard output cannot be written. | `grobx: standard output: <reason>` |
| The watch on SIGINT cannot start, or fails once started, or a second SIGINT cannot end the process. | `grobx: SIGINT: <reason>` |
| The call returns an `Error`. | `grobx: <input>: <message>`, `<input>` the path or `standard input` |

| Exit status | When |
|---|---|
| 0 | The basis was written, and the checker accepted it or was not asked; or the help or the version was printed. |
| 1 | A bound stopped the call: `--deadline`, `--memory`, or SIGINT. |
| 2 | An error: the watch on SIGINT, the arguments, the instance, an output file or standard output, or an instance or order the call refuses, `Error::Memory` among them. |
| 3 | The checker rejected the result. The basis was written, and the cofactors where `--cofactors` names a file. |

While `grobx` reads the instance, SIGINT ends the process by SIGINT.
After that, SIGINT raises the call's cancel flag, so the call stops at its next check
and `grobx` exits 1. A SIGINT that comes before the call, while `grobx`
opens its output files, stops the call at its first check. A second
SIGINT ends the process by SIGINT, with no exit status. Where SIGINT is
ignored when `grobx` starts, as a shell has it for a command run in the
background, `grobx` leaves it ignored.
Where the watch fails once started, or a second SIGINT cannot end the
process, `grobx` says why and exits 2.
On Windows, Ctrl-C does what SIGINT does here. A second Ctrl-C ends the
process as Ctrl-C does, and a Ctrl-C that is ignored when `grobx` starts,
as in a new process group, stays ignored.

### C

The package `grobx-c` builds the shared library `libgrobxc.so` on Linux,
`libgrobxc.dylib` on macOS, and `grobxc.dll` on Windows:
`cargo build --release -p grobx-c` writes it to
`target/release/`. Its header is `c/grobx.h`. A C program
includes the header and links with `-lgrobxc`, on Windows against the
import library `grobxc.dll.lib`, and there finds the DLL beside itself or
on the `PATH`. A C++ program includes the
same header as it stands.

```c
#include "grobx.h"

grobx_params p = grobx_params_new(GROBX_GREVLEX, 1, 1);
p.deadline = 60000;
p.checker = GROBX_CHECKER_EXACT;
p.cofactors = true;
grobx_result r;
int status = grobx_groebner("x,y,z\n0\nx+y+z,\nx*y+y*z+z*x,\nx*y*z-1\n", &p, &r);
/* status GROBX_OK, r.basis "[x+y+z,\ny^2+y*z+z^2,\nz^3-1]:\n",
   r.certified GROBX_CERTIFIED */
grobx_result_free(&r);
```

`grobx_groebner(text, &params, &result)` computes what `grobx.groebner`
computes for the same arguments (_Python_ above). `text` is the instance
text, NUL-terminated.

`grobx_params_new(order, threads, seed)` returns a `grobx_params` with the
order, threads, and seed given, no blocks, every bound absent, no checker, no
cofactors, and no report. `grobx_params_default()` is
`grobx_params_new(GROBX_GREVLEX, 0, s)` with s drawn as `random_seed()`
draws it, the C form of `Params::default()`; `seed` holds it to run the call again.
The caller sets the other fields by name. The fields are those
of `Params` (_Rust_ above). Where a field's type differs:

| Field | Type |
|---|---|
| `order` | `uint32_t`: `GROBX_GREVLEX`, `GROBX_DEGLEX`, `GROBX_LEX`, or `GROBX_BLOCK` |
| `blocks`, `nblocks` | `const size_t *` and `size_t`: under `GROBX_BLOCK`, the sizes k1, …, km of `Block` (_Monomial orders_ below); `{2, 3}` is `Block(vec![2, 3])` |
| `threads` | `size_t`; 0 is all the machine has |
| `deadline` | `uint64_t` in milliseconds; `GROBX_NONE`, `UINT64_MAX`, is none |
| `cancel` | `grobx_cancel *`; NULL is none |
| `memory` | `uint64_t` in bytes; `GROBX_NONE` is none |
| `checker` | `uint32_t`: `GROBX_CHECKER_OFF`, `GROBX_CHECKER_EXACT`, or `GROBX_CHECKER_CHEAP` |
| `points` | `uint32_t`: the cheap mode's points, from 1 to 2^32 − 1; read in the cheap mode alone |

`signatures`, `echelon`, and `report_sink` are no fields: a C call runs
with both on, as a Python call does, and with no sink.

The call writes a `grobx_result`. Its fields are those of `Outcome` but
`stats` and `sink_error`, and one more, `error`:

| Field | Type |
|---|---|
| `basis` | `char *`; NULL where a bound stopped the call or the status is not `GROBX_OK` |
| `cofactors` | `char *`; NULL where the call did not ask for them or did not finish |
| `stopped` | `int`: `GROBX_STOP_NONE`, `GROBX_STOP_DEADLINE`, `GROBX_STOP_CANCEL`, or `GROBX_STOP_MEMORY` |
| `certified` | `int`: `GROBX_UNCHECKED`, −1, where the checker did not rule: the call did not ask for it, a bound stopped it, or it failed; `GROBX_REJECTED`, 0; or `GROBX_CERTIFIED`, 1 |
| `flaw` | `char *`; why the checker rejected the result, or NULL |
| `error` | `char *`; the error's message where the status is not `GROBX_OK`, else NULL |
| `report` | `char *`; the call report where the call asked for it and the status is `GROBX_OK`, else NULL |

Every text of the result is NUL-terminated and the library's. The call
writes every field of the result, whatever its status, and neither reads
nor frees what the result held before. `grobx_result_free(&r)` frees every
text of `r` and sets it NULL. The caller frees the result after every
call, an error's included. A second free frees nothing, and
`grobx_result_free(NULL)` does nothing.

The call makes its texts C strings after the core's call has returned, outside its bounds. The
crate's texts hold no NUL byte; one that did would fail the call with `GROBX_ERROR`, `error` naming
the text and the byte.

The call returns a status code:

| Status | Value | When |
|---|---|---|
| `GROBX_OK` | 0 | The call returned a result, one a bound stopped among them. |
| `GROBX_SYNTAX` | 1 | `Error::Syntax` |
| `GROBX_UNKNOWN_VARIABLE` | 2 | `Error::UnknownVariable` |
| `GROBX_REPEATED_VARIABLE` | 3 | `Error::RepeatedVariable` |
| `GROBX_CHARACTERISTIC` | 4 | `Error::Characteristic` |
| `GROBX_OVERFLOW` | 5 | `Error::Overflow` |
| `GROBX_NO_INVERSE` | 6 | `Error::NoInverse` |
| `GROBX_ORDER` | 7 | `Error::Order` |
| `GROBX_MEMORY` | 8 | `Error::Memory` |
| `GROBX_ERROR` | 9 | Any other `Error`, or a text of the result that holds a NUL byte |
| `GROBX_ARGUMENT` | 10 | An argument the interface refuses before the call: `text`, `params`, or `result` NULL; `text` not UTF-8; `order` none of the four orders; under `GROBX_BLOCK`, `blocks` NULL while `nblocks` is not 0; `checker` none of the three modes; or the cheap mode with `points` 0. |
| `GROBX_PANIC` | 11 | The call panicked, a defect of grobx. The process's panic hook hears of the panic too, and by default prints it on standard error. |
| `GROBX_SIZE` | 12 | `Error::Size` |
| `GROBX_DROPS` | 13 | `Error::Drops` |
| `GROBX_NOT_BUILT` | 14 | `Error::NotBuilt` |

Where the status is not `GROBX_OK`, `error` holds the message: an
`Error`'s as the CLI writes it, or what the interface refused, or the
panic's. A NULL `result` returns `GROBX_ARGUMENT` and writes nothing.

`grobx_cancel_new()` makes a flag. `grobx_cancel_set(c)` raises it,
`grobx_cancel_is_set(c)` reads it, and `grobx_cancel_free(c)` frees it.
Any thread may raise the flag while a call runs, and the call stops at its
next check. The caller frees the flag once no call uses it and no thread
raises it. Each of the three takes NULL and does nothing, and
`grobx_cancel_is_set(NULL)` is false.

`grobx_digest(text, digest)` writes SHA-256 of the NUL-terminated `text`,
64 lowercase hex digits and a NUL, to `char digest[65]`. It returns
`GROBX_OK`, or `GROBX_ARGUMENT` where an argument is NULL or the text is
not UTF-8.

`grobx_version()` returns the crate version, a static NUL-terminated text.

### Maple

The Maple package `Grobx`, the one file `maple/Grobx.mpl`, runs the
command-line program `grobx` (_CLI_ above). Its `Grobx:-Basis` is called as
Maple's `Groebner:-Basis` is, for the orders and fields grobx computes in,
and raises an error for anything else:

```maple
read "/path/to/Grobx.mpl":
with(Grobx):
Basis([x + y + z, x*y + y*z + z*x, x*y*z - 1], tdeg(x, y, z));
# [x + y + z, y^2 + y*z + z^2, z^3 - 1]
```

`maple/README.md` describes the package: where it finds the program, its
orders and options, and its errors.

### Sage

The Python module `grobx_sage`, the one file `sage/grobx_sage.py` of the PyPI package `grobx-sage`,
calls the Python module
(_Python_ above). Its `groebner_basis(I)` is called as Sage's `I.groebner_basis()` is, for the
orders and fields grobx computes in, and raises an error for anything else:

```sage
import grobx_sage
R.<x,y,z> = PolynomialRing(QQ)
grobx_sage.groebner_basis(R.ideal(x + y + z, x*y + y*z + z*x, x*y*z - 1))
# [z^3 - 1, y^2 + y*z + z^2, x + y + z]
```

`sage/README.md` describes the module: its setup, orders and keywords, and its errors.

### Browser

The workspace member `grobx-wasm`, in `wasm/`, builds the crate for a web page:

```sh
cargo build -p grobx-wasm --target wasm32-unknown-unknown --profile wasm
```

The module `grobx_wasm.wasm` imports nothing and exports its memory and three functions.
`grobx_input(len)` returns a buffer of `len` bytes, which the caller fills with the request,
`grobx_run()` answers it and returns the answer's length in bytes, and `grobx_output()` points at
the answer. Both are UTF-8, and a pointer holds until the next call. The memory may grow while a
call runs, so the caller takes a fresh view of it after each. The request is the order's
name, the seed in decimal digits, and `1` or `0` for the cofactors, a line each, then the instance
text. The answer is a JSON object: `version`, then either `basis`, `cofactors`, null where not
asked, `rounds`, `reduced`, and `zero`, or `error`, what went wrong, opening with the variant's
name where the crate refused the call. A call
runs on one thread, with no bound, no checker, and no report. grobx.com's `/live` runs the module
in a Web Worker.

Built with `--no-default-features` (_Rust_ above), the module answers a request with `1` for the
cofactors with the error `NotBuilt`.

### Errors

`Error` is `#[non_exhaustive]`. Its `Display` is its message, and `Error::name()` the variant's
name, which the call report's `error` line, `grobx.Error`, and the browser's answer open with. The
C interface returns each `Error` as a status code (_C_ above).

| `Error` | Raised where |
|---|---|
| `Syntax { line, expected }` | The instance text breaks the grammar at `line`, counted from 1. |
| `UnknownVariable { line, name }` | A generator names a variable the first line does not. |
| `RepeatedVariable { name }` | The first line names a variable twice. |
| `Characteristic { value }` | The characteristic is neither 0 nor a prime below 2^64, or is 0 in `learn` or `replay`. |
| `Overflow { line }` | A degree passes the widest monomial format, in the instance at `line` or during the run, or a pair's sugar passes 2^32 − 1 during the run. |
| `Size` | The instance or the run holds more generators, terms, numeric factors, monomials, rows, elements, or record steps than a u32 id numbers, 2^32 − 1, or the instance holds a term at a line past 2^31 − 1. |
| `NoInverse { line }` | A denominator at `line` is 0 mod p over GF(p), or is 0 over ℚ. |
| `Order { name }` | `name` names no order (`str::parse`), or it is a block order whose sizes do not fit the variables. |
| `Memory` | A memory ceiling is asked for where the resident set cannot be read. |
| `Clock` | A deadline or the call report is asked for where the system has no clock: wasm32 in a browser (_Browser_ above). |
| `Misfit { step }` | A replay's record does not fit at `step`; step 0 is the inputs, a record learned in another order, a record learned with the echelon where the replay tracks cofactors, or the change of order of a record FGLM changed. A changed record whose replayed grevlex basis FGLM cannot change does not fit at the step one past the record's last. |
| `Drops { misfits, leads, verification }` | The lift over ℚ dropped 4 records in the call: `misfits` after 2 misfits in a row, `leads` at a verification whose leading monomials differ, and `verification` at one at which an element differs. Random primes make it improbable, and a defect of grobx makes it certain. |
| `NotBuilt { feature }` | The call asks for the cofactors or the checker, and the crate was built without the Cargo feature `feature`, `cofactors` or `checker`, that computes them (_Rust_ above). The message says `not implemented in this build`. |

## Formats

### Instance text

The instance text is msolve's input format:

```
x,y,z
65521
x^2 + y*z - 1,
2*x*y - 1/3*z,
z^3 - x
```

- **Line 1:** the variable names, separated by commas, the largest variable
  first. A name is an ASCII letter or `_`, then ASCII letters, digits, and
  `_`.
- **Line 2:** the characteristic in decimal digits. A prime p below 2^64
  computes over GF(p). 0 computes over ℚ.
- **The rest:** the generators, separated by commas. Whitespace, line feeds
  included, may stand between any two tokens. No generators is the zero
  ideal.

Whitespace is the ASCII space, tab, line feed, form feed, and carriage
return. On lines 1 and 2 any of them but the line feed may stand between
and around the tokens.

A generator is a sum of terms, each preceded by `+` or `-`; the first term's
sign may be left out. A term is a product of factors joined by `*`. A
factor is a decimal integer or a variable with an optional exponent `^e`,
and any factor may be followed by denominators `/d`, decimal integers that
divide the term. A term may hold several numbers
and repeat a variable, and like terms are combined: `2*x*3*x^2` is `6*x^3`.
A term's degree is at most 2^31 − 1, its coefficient zero or not; a term
past it is `Overflow` once the term is read.
`1/3*y`, `y/3`, and `y*1/3` are one term, and `1/2/3*y` is `1/6*y`; a
denominator is a number, so `x/y` is a syntax error. Over GF(p) every number is read mod p.
A denominator 0, its digits all `0`, is `NoInverse` at the line of those
digits, over ℚ as over GF(p). The parse finds it, so an error earlier in
the text, on the same line included, comes first.

### Basis text

The basis text is canonical: one ideal and one order give one text, so the
digest of two texts compares two bases.

```
[y^2+x,
x*y+6,
x^2+y]:
```

- `[`, the elements separated by `,` and a line feed, then `]:` and a line
  feed. No spaces anywhere.
- The elements stand in increasing order of their leading monomials, and the
  terms of each in decreasing order, in the call's monomial order.
- A term is its coefficient, `*`, and its monomial. The coefficient 1 and
  the `*` after it are left out, and so is the monomial 1. A monomial joins
  its variables by `*` in the instance's variable order. The exponent 1 is
  left out.
- Over GF(p) every element is monic, its coefficients in [0, p) and written
  without a sign; `x*y+6` is x·y − 1 over GF(7).
- Over ℚ every element has integer coefficients with no common factor, and
  its leading coefficient is positive.
- Terms are joined by `+`. Over ℚ a negative coefficient is written as `-`
  and its absolute value, in place of the `+`, and an absolute value 1 is
  left out as the coefficient 1 is: `x^2-y`.
- The zero ideal's basis is `[]:`, and the unit ideal's is `[1]:`.

### Cofactor text

The cofactor text expresses every element of the basis in the generators:
row i holds, for each generator j of the instance, the cofactor c_ij with
g_i = Σ_j c_ij·f_j.

```
[[y,6*x],
[0,1],
[1,0]]:
```

- `[`, one row per element of the basis in the order of the basis text,
  separated by `,` and a line feed, then `]:` and a line feed.
- A row is `[`, one polynomial per generator in the instance's order,
  separated by `,`, then `]`. A zero cofactor is `0`. Zero generators have
  their column.
- Each polynomial is spelled as in the basis text, terms in decreasing order.
  Over ℚ a coefficient may be a fraction and may be negative:
  `[[0,1/3],\n[1/2,0]]:\n` for the generators `2*x`, `3*y`.

The example above is the basis of _Basis text_: y² + x = y·(x² + y) +
6x·(xy − 1) over GF(7).

### Report text

The call report follows a call from its instance text to its result: one
line per event, in the order the call met the events. A call writes it
where `Params::report` asks for it (_Rust_ above).

```
report = *line
line   = time SP kind *(SP key "=" value) LF
time   = 1*DIGIT "." 6DIGIT
kind   = 1*LOWER
key    = 1*(LOWER / "-")
value  = 1*(any byte but SP, LF, and "=")
```

- `time` is the seconds since the call started, truncated to the
  microsecond. The times never decrease from line to line.
- A value is a decimal integer, a duration in seconds with six decimals,
  a word, or a list of those joined by `,`. A field that does not apply
  to the event is left out.
- The first line's kind is `call`, and the last line's is `end`, or
  `error` where the call returns an error. An error of the memory
  ceiling comes before the report begins, and its report is its `error`
  line alone.

The call `grobx.groebner("x,y\n7\nx^2+y,\nx*y-1\n", seed=1, threads=1, report=True)` through
Python:

```
0.000032 call entry=groebner order=grevlex seed=1 threads=1 signatures=on echelon=on cofactors=off checker=off deadline=none memory=none cancel=off trace=off cpu=Intel(R)_Core(TM)_i7-14700K
0.000037 read variables=2 characteristic=7 field=gf31
0.000051 run number=1 role=call engine=signature engine-by=signatures field=gf31 primes=7 width=8 order=grevlex cofactors=off reduce=tests handover=on why=start
0.000082 inputs generators=2 nonzero=2 terms=4 read=0.000029
0.000097 echelon generators=2 rows=2 squares=0 columns=4 subtracted=0 time=0.000008
0.000119 enter elements=2 pairs=1 time=0.000018
0.000149 round number=1 degree=3 pairs=1 rows=2 given=1 kept=0 columns=3 entries=4 index=2 pivots=1 zero=0 zero-rows=0 reductions=1 subtracted=2 added=0 bands=no packed=no pending=1 dropped=0 select=0.000005 symbolic=0.000004 encode=0.000002 reduce=0.000010 backsub=0.000000 record=0.000001 decode=0.000003 update=0.000001
0.000151 handover round=1 outgrown=stay made=1 zero=0 divisible=0 widest=1 verdict=stay
0.000169 inter rows=3 given=3 columns=6 entries=6 index=2 subtracted=0 elements=3 symbolic=0.000006 encode=0.000000 reduce=0.000003 decode=0.000005
0.000173 ran number=1 outcome=basis rounds=1 reduced=1 zero=0 elements=3 time=0.000121 considered=3 equal=0 high=0 low=0 syzygy=2 koszul=0 queued=1 past-format=0
0.000185 text kind=basis bytes=23 time=0.000009
0.000208 end result=basis bytes=23 time=0.000187 process-rss-start=9691136 process-rss-peak=13762560 process-rss-grown=4071424
```

Below, each kind with its fields in the order its line writes them. A
"time" is a duration in seconds. `on|off` and the like list a field's
values.

**The call.**

- `call entry=<groebner|learn|replay> order=<name> seed=<n>
  threads=<all|k> signatures=<on|off> echelon=<on|off> cofactors=<on|off>
  checker=<off|exact|cheap> points=<k> deadline=<ms|none>
  memory=<bytes|none> cancel=<on|off> trace=<listens|off>
  cpu=<brand|none>`: the
  parameters as the call reads them. `entry` is `learn` for `learn`,
  `replay` for `replay`, and `groebner` for `groebner` and
  `groebner_with`. `order` is the order's name (_Monomial orders_ below).
  `points` stands where `checker` is `cheap`. `trace` is `listens` where
  the trace of `groebner_with`, `learn`, or `replay` hears the matrices.
  `cpu` is the processor's CPUID brand string up to its first NUL byte, its words joined by `_`,
  and `none` off x86-64, where the processor has no brand leaves, or where the brand holds no word.
- `read variables=<n> characteristic=<p> field=<gf2|gf31|gf64|q>`: the first
  two lines of the instance text as read. `gf2` computes over GF(2), `gf31` over another prime
  below 2^31, `gf64` over a larger prime, and `q` over ℚ, through the
  lift (_Over ℚ_ below).
- `stop bound=<deadline|cancel|memory>`: a bound stopped the call, the
  outcome's `stopped`.
- `end result=<basis|stopped> bytes=<b> time=<t> process-rss-start=<b>
  process-rss-peak=<b> process-rss-grown=<b>`: the call's end, after
  `stop` where there is one. `bytes` is the basis text's length, where
  there is a basis, and `time` the whole call's.
  `process-rss-start` is the process's resident set at the call's start,
  read as the memory ceiling reads it.
  `process-rss-peak` is the process's peak resident set at the end,
  on Linux `VmHWM` of `/proc/self/status`, the kernel's high-water mark of the
  process's own image since the process began, or since its owner last
  reset it through `/proc/self/clear_refs`; on macOS getrusage's
  `ru_maxrss`, and on Windows the peak working set. `process-rss-grown` is the
  peak less the start. All three are in bytes. The call resets no
  high-water mark, so a call after a larger one in the same process
  reads that one's peak. On Linux and Windows the peak is the process's
  own, whatever resident set the process that started it held; on macOS
  that is not known. Where the system gives no figure, the line holds
  none of the three fields.
- `error error=<variant> message=<text>`: the call returns the error
  `Error::<variant>`, whose message is `message`, its words joined by `_`
  and each `=` an `_`.
- `alloc kind=<alloc|zeroed|realloc> bytes=<b> from=<b>
  result=<served|refused> stack=<frame>,…`: a request for a block of 1
  GiB or more (`REPORTED`), or one the allocator refused, which then aborts
  the process. Only `grobx::Allocator` writes it, and only to
  `report_sink`, not to the outcome's report. `from` is the old block's
  bytes where `realloc` resizes one. `stack` gives on Linux the request's
  return addresses, each as its object's file name, `+0x`, and the
  offset into the object, which `addr2line -f -C -e <object>` maps to a
  function. A request that comes while the sink writes another line has
  no line.

**A run.** Each run of the engine writes a `run` line as it starts and a
`ran` line as it ends, and its rounds' lines between them. Over ℚ every
run of the lift does.

- `run number=<k> role=<call|learn|replay|verify>
  engine=<signature|plain|replay>
  engine-by=<signatures|signatures-off|order|handover|field|record>
  field=<gf2|gf31|gf64|gf31pair|gf64pair> primes=<p[,q]> width=<4|8|16|32>
  order=<grevlex|deglex|lex|block> cofactors=<on|off>
  reduce=<rows|combinations|tests|sketches>
  reduce-by=<record|trace|learner|cofactors|lanes|prime>
  handover=<on|off> shape=<use|take|off>
  why=<start|wider|handover|outworked|carry|refit|lex>`.
  - `number` counts the call's runs from 1, the lift's among them.
  - `role`: `replay` where the run replays a record, `learn` where it
    records one, `verify` where it is the lift's run at the
    verification's prime, and `call` for any other.
  - `engine` and `engine-by`, why that engine: `signatures`, the
    call's `signatures` on and a graded order; `signatures-off`;
    `order`, a block order or F4 in lex; `handover`, the plain run after
    the signature run handed over; `field`, the plain run under a bound
    on its row work that starts a call over GF(2) (_Signatures_, **The
    plain run first over GF(2)**, below); `record`, a replay.
  - `field` and `primes`: the field the run computes in and its primes,
    two where a pair of primes shares the run. `width` is the bits of
    each exponent of the monomial format. `order` is the order the run
    computes in: `grevlex` for a lex call's grevlex run, whose basis
    FGLM is to change.
  - `reduce`: `combinations` where the plain run reduces random
    combinations of rows, `tests` where the signature run or a learn's
    plain run rules rows zero by tests, `sketches` where the plain run over GF(2) rules rows zero
    by sketches in its rounds of long rows (_Rows ruled zero by sketches over GF(2)_ in _Matrices
    and reduction_ below), else `rows`, each row reduced
    alone. `reduce-by` stands
    with `rows` and names the first reason that holds, in this order: a
    replay, a trace that listens, a learn's signature run, cofactors tracked, a
    signature run over a pair of primes, or a prime too small for the
    combinations or the tests.
  - `handover` stands for the signature run: `off` where it tracks
    cofactors and so never hands over. `shape` stands for a replay:
    `use` where it runs from a kept shape of the record's matrices,
    `take` where it keeps one, else `off`. The lift alone keeps shapes.
  - `why`: `start`, the call's first run or the lift's run at a prime;
    `wider`, the run before passed its monomial format; `handover`;
    `outworked`, the plain run before passed its bound on its row work;
    `carry`, the signature run before handed its work over, which the
    plain run goes on from (_Signatures_, **The plain run goes on from
    the signature run's work**, below);
    `refit`, a replay from a kept shape met other terms; `lex`, a lex
    call whose ideal FGLM does not take, which runs F4 in lex.
- `inputs generators=<m> nonzero=<k> terms=<t> read=<time>`: the
  generators read into the run's format, the terms counted over all,
  and the time of the read. The call's first read parses the text, and
  every later one maps the parsed generators (_Over ℚ_, **The
  generators parsed once**, below).
- `substitute linear=<l> pivots=<k> changed=<c> stayed=<s> terms=<t> after=<a> time=<time>`: after
  `inputs`, where the run takes the generators in reduced row echelon form, over GF(2) no bounded
  or plain run, and substituted at least
  one variable that its linear generators lead in its other generators (_Matrices and reduction_,
  **The linear generators substituted**, below): the generators of degree 1; the variables
  substituted; the generators that took their substitutes, and those that held such a variable and
  stayed as read, a sum on the way to their substitutes, or the substitutes themselves, no shorter;
  the terms of the changed generators before and
  after; and its time.
- `echelon generators=<m> rows=<r> squares=<q> columns=<c> subtracted=<s>
  time=<time>`: after `inputs` and `substitute`, where the run takes the generators in
  reduced row echelon form (`Params::echelon`, _Rust_ above), over GF(2) no bounded or plain run:
  the
  generators read, zero ones among them; the nonzero rows the run takes, their rank less the rows
  left out; the rows left out over GF(2) as squares of linear rows (_Matrices and reduction_, **The
  inputs**, below), 0 in a run that learns or replays and over every other field;
  the matrix's columns, the generators' distinct monomials; the entries
  its row operations subtracted, as a round's `subtracted` counts them;
  and its time.
- `round number=<k> degree=<d> pairs=<p> rows=<r> given=<g> kept=<n>
  columns=<c> entries=<e> index=<2|4> pivots=<v> zero=<z> zero-rows=<y> reductions=<x>
  subtracted=<s> added=<a> bands=<yes|no> packed=<w|no> mode=<blocks|rows> sketches=<k>
  pending=<q> dropped=<o> deferred=<f> left=<l> select=<time> symbolic=<time> encode=<time>
  reduce=<time>
  backsub=<time> record=<time> decode=<time> update=<time>`: a round of
  the run, one line per round.
  - `degree`: the sugar of the plain run's pairs, in a graded order their lcms' degree or one
    more for a fallen element's pairs (_Pairs_ below), the degree of the
    signature run's signatures, or the degree of a replay's first
    column. `pairs`: the plain run's pairs, the signature run's
    reductants, or a replay's rows the record counts.
  - `rows`, `given`, `kept`, `columns`, and `entries`: the matrix's rows, of
    which `given` were given, ahead of the reducers symbolic preprocessing
    added, of which `kept` were taken from the matrix before as built,
    its columns, and its nonzero entries. `index`: the bytes of an entry's
    column, 2 where the build kept the matrix narrow, 4 where it was wide
    (_Columns of 2 bytes_ below).
  - `pivots`: the rows the reduction made; `zero`: the reductions to
    zero; `reductions`: the reductions run.
    `zero-rows`: the rows known to reduce to zero, left out where the
    round's `mode` is `blocks`. Where the line names `sketches`, `zero-rows` less `zero` gives the
    rows a sketch ruled zero.
  - `subtracted` and `added`: the entries the round's row operations
    subtracted, and those they added into combinations, as the budget
    spends them (_Limits_ below).
  - `bands`: whether the round's matrix takes bands of rows (_Matrices and reduction_ below).
  - `packed`: the words of the K columns where the round reduced its rows in packed rows over
    GF(2), and `no` where it did not (_The early pivots reduced once, in packed rows_ in _Matrices
    and reduction_ below).
  - `mode` stands for the plain run where it combines rows: how the round
    reduced its rows to reduce after their first block of random
    combinations, `blocks` as random combinations of blocks, or `rows` each
    on its own, with the tests of rows that reduce to zero. A round that
    probed reduced 8 rows of its second block each on its own before it
    chose, in either mode (_Short rows reduced each on its
    own, chosen per round_ in _Matrices and reduction_ below).
  - `sketches` stands for the plain run over GF(2) where the round's rows were tested by sketches:
    the sketches drawn, 0 where no two rows reduced to zero one after another with rows left (_Rows
    ruled zero by sketches over GF(2)_ in _Matrices and reduction_ below).
  - `pending` and `dropped` stand for the signature run: the signatures
    taken since its last round, and those that gave no reductant.
    `deferred` and `left` stand for the plain run: the pairs of the
    round's sugar its selection left in the pair set, 0 where it took
    them all (_The pairs of a large round_ below), and the pairs left
    after the round.
  - The eight times split the round:
    `select`, the pairs, their lcms interned, and their rows, or the
    reductants; `symbolic`,
    symbolic preprocessing; `encode`, the columns sorted and mapped;
    `reduce`, the reduction but its back-substitution; `backsub`;
    `record`, the trace's and the learn run's hooks; `decode`, the rows
    made into polynomials, with the signature run's syzygies and its
    handover's tests; `update`, the new elements and their pairs. A
    replay from a kept shape builds no matrix, and its `symbolic` and
    `encode` are 0.
  - A round that a bound or an overflow ends after it was counted
    writes its line before the next line, its times as they stood, and
    `left` left out.
- `handover round=<k> early=<stay|hand-over> gap=<stay|hand-over>
  outgrown=<stay|hand-over> large=<stay|hand-over>
  pair-work=<stay|hand-over> made=<m> zero=<z>
  divisible=<d> pairs=<n> widest=<w>
  verdict=<stay|hand-over>`: after every round that made an element in a
  signature run that may hand over, which the outgrown test reads, each
  test that read it with its verdict, the early test at the early rounds and the large-round test
  at the large ones (_Signatures_ below); the elements
  the round made, its rows reduced to zero, the count a test read where
  one did, the S-pairs the round's update would form where the pair-work
  test read them, the round's widest gap, and whether the run hands over.
- `carry round=<k> elements=<n> made=<m> bound=<b> work=<w>
  monomials=<c>`: before the first round of a plain run that goes on
  from the signature run's work: the signature run's handover round, the
  elements carried, of which the round made `made`, the sugar at and below
  which the pairs of two carried elements are left out, the entries the
  signature run subtracted, and the monomials of the plain run's fresh
  table.
- `enter elements=<n> pairs=<p> time=<time>`: after a plain or a signature run's `inputs`,
  `substitute`, `echelon`, and `carry` lines and before its every other line, where the run has
  entered the elements it starts from: the elements entered, the carried elements where the run goes
  on from a carry, else the rows of the run's echelon, else its nonzero generators; the pairs they
  left, the plain run's pair set or the S-pairs the signature run queued; and the time from the
  run's first budget check to the entry's end, the elements made monic, their sugars or their
  signatures, and their pairs formed among it. A replay, which enters its inputs with no pairs,
  writes none. Nor does a run whose input is constant, or one that a bound or a halt ends before its
  entry ends, whose `ran` line's `time` holds the entry so far.
- `constant round=<k>`: the run made a constant, and the basis is
  `[1]:`. Round 0 is an input that is constant.
- `compact round=<k> monomials=<n> kept=<m> time=<time>`: after the line of
  round k of a plain run, where the run compacted its monomial table at
  the round's end (_Monomials_, **The plain run's table compacted**,
  below): the monomials the table held, those it kept, and the
  compaction's time.
- `tails round=<k> members=<n> linear=<l> given=<g> changed=<c> terms=<t> after=<a> rows=<r>
  columns=<w> time=<time>`: before the line of round k of a plain run, where the run inter-reduced
  `G_min` at the round's start (_Matrices and reduction_, **The basis inter-reduced where an element
  of degree 1 joins it**, below): `G_min`'s members, those of degree 1, the members it took as the
  matrix's given rows, the members whose polynomials changed, every member's terms before and after,
  the inter-reduction's matrix's rows and columns, 0 and 0 where it took no given row and built no
  matrix, and its time.
- `inter rows=<r> given=<g> columns=<c> entries=<e> index=<2|4>
  subtracted=<s> elements=<n> symbolic=<time> encode=<time> reduce=<time>
  decode=<time>`: the matrix that inter-reduces the basis, the elements
  it made, and the times of its stages, `decode` with the elements
  sorted. `rows` to `index` are the round line's.
- `ran number=<k>
  outcome=<basis|handover|outworked|carry|overflow|size|denominator|refit|misfit|stop>
  at=<read|substitute|echelon|run> step=<s> work=<w> bound=<b> rounds=<k> reduced=<x> zero=<z>
  elements=<n>
  time=<time> considered=<n> equal=<n> high=<n> low=<n> syzygy=<n>
  koszul=<n> queued=<n> past-format=<n>`: the run's end. `overflow`:
  the read or the run passed the monomial format. `denominator`: the
  read met a denominator that is 0 mod the run's prime. `at` stands for
  both, `read` or `run`, and as `echelon` where the generators' reduced
  row echelon form ended the run: `misfit`, `stop`, or `size`. It stands as `substitute`, with the
  same outcomes, where the substitution of the linear generators ended the run, and as `read`
  for `stop` where the read did.
  `step` stands for `misfit`: the replay's step that did not fit,
  or 0 where the lanes of a pair's run disagree. `size`: a count of the
  run passed what a u32 id numbers (`Error::Size`).
  `carry`: the signature run handed its work over to the plain run,
  which goes on from it. `outworked`: the plain run that starts a call
  over GF(2) passed its bound on its row work, and the call starts over
  in the signature run.
  `work` and `bound` stand for `outworked`: the row work the run spent and the units of its bound,
  `work` at most `bound` where the row work projected over a round in sparse rows passed the bound
  (_Signatures_, **The plain run first over GF(2)**).
  `rounds`, `reduced`, and `zero` are the run's `stats`, and `elements`
  the reduced basis's, where the run made one. A plain run that goes on
  from a carry writes its own counts, and the outcome's `stats` are the
  sum of both runs'. `time` runs from the
  `run` line. The signature run adds what became of its S-pairs over
  the run: considered, ruled out by an equal sig-lead ratio and index,
  by the high or the low ratio base divisor, by a known syzygy
  signature, or by coprime leading monomials, queued, and past the
  format.
- `fglm verdict=<change|past-bound|positive-dimensional|overflow|size|misfit|stop>
  staircase=<D> find=<time> change=<time>`: a lex call's grevlex basis.
  `change` where FGLM changed it to lex, `past-bound` and
  `positive-dimensional` where FGLM does not apply: F4 then runs in lex,
  or a replay of a changed record does not fit,
  else the halt of the staircase or of the change. `staircase` and
  `change` stand where the staircase was found: its size and the
  change's time.
- `image elements=<n> time=<time>`: a run of the lift made its image.
- `text kind=<basis|cofactors> bytes=<b> time=<time>`: a text written.
- `check mode=<exact|cheap> points=<k> verdict=<certified|rejected>
  time=<time>`: the checker ruled, `points` in the cheap mode alone. Where a bound stops the
  check, `stop` follows instead.

**The lift over ℚ.** After each prime's run the lift writes what became
of the prime, in the order it used them. A pair's second prime whose
image the lift never takes, where the first ends the lift or drops the
record, stands on no line of its own: on the pair's `run` line, and on
its `pair` line where the pair fell back:

- `lift learns=pairs replays=<pairs|alone> track=<on|off>`:
  once, first. `replays` is `pairs` where two primes share each replay
  of a record FGLM did not change: the call tracks no cofactors. A
  record FGLM changed replays one prime at a time even so.
  `track` is whether the replays track cofactors.
- `learn primes=<p[,q]> outcome=<recorded|unpaired|skipped>
  why=<disagree|denominator> skipped=<n>`: a learn run's record. `unpaired`: a learn
  at two primes fell back to the first alone, for `why`. `skipped`: the
  prime divides a denominator of the text.
  `skipped=<n>`, where the outcome is `recorded`: the counted rows of the
  learn run that its replays skip (_Trace and replay_, _The rows the
  output depends on_).
- `pair primes=<p,q> outcome=alone why=<misfit|denominator>`: a
  replay at two primes runs each alone.
- `skip prime=<p> why=<denominator|lifted-denominator>`: a prime that
  divides a denominator of the text, or the verification's prime that
  divides one of the lifted basis.
- `misfit prime=<p> why=<record|leads>`: the replay did not fit the
  record, or its leading monomials differ from the learn run's.
- `drop why=<misfits|leads|verification>`: the record and the lifting
  dropped, after 2 misfits in a row, or at a verification whose
  leading monomials differ or at which an element differs. The next
  prime learns afresh, save after the fourth drop, where the call returns `Drops`.
- `combine prime=<p> primes=<k> open=<n> kept=<n> attempts=<a>
  merge=<time> reconstruct=<time> crt=<time>`: the image at p combined:
  the primes combined so far, the elements open before and kept after,
  the rational reconstructions tried, and the times of the merge, of the
  attempts with their conversions, and of the prime's Chinese remainder
  step with the append of its image. At a pair's first prime the lift
  merges the pair's image and attempts nothing, `attempts=0`, and at its
  second it merges nothing.
- `verify prime=<q> outcome=<passed|failed|leads-differ> time=<time>`:
  the lifted basis checked at q, `time` where the leading monomials
  agree.

A pair's two primes share its run's lines. The texts, `check`, `stop`,
and `end` follow as over GF(p), with `kind=cofactors` timed apart from
`kind=basis`.

### Monomial orders

An order is a `MonomialOrder` in Rust and its name in Python. A name parses
to its `MonomialOrder` by `str::parse`, which returns `Error::Order` where it
names none, and `to_string` gives back its name, each size without
leading zeros.

- `Grevlex`, named `"grevlex"`, is the graded reverse lexicographic order.
- `Deglex`, named `"deglex"`, is the degree lexicographic order: a > b where deg a > deg b, or the
  degrees are equal and the first variable whose exponents differ has the larger exponent in a. The
  call runs F4 in deglex as it does in grevlex, the signature run where `signatures` asks for it,
  and changes no order.
- `Lex`, named `"lex"`, is the lexicographic order. The call computes the grevlex basis first.
  Where the ideal is zero-dimensional and its degree is at most `MAX_STAIRCASE`, 65535, it changes
  that basis to lex by FGLM; otherwise it runs F4 in lex as well.
- `Block(vec![k1, …, km])`, named `"block:k1,…,km"`, splits the variables,
  in the instance's order, into
  blocks of k1, …, km variables and a last block of the rest. Each block is
  ordered by grevlex, and the blocks are compared in turn. The sizes are
  decimal digits, separated by commas, with no sign and no space. Each is at
  least 1, and they sum below the number of variables. `"block:k"`
  eliminates the first k variables, as Groebner.jl's `block:k` and msolve's
  `-e k` do.

Each order has the one name above. grobx reads no other program's name for it, since such a name
can mean another order elsewhere: `tdeg` is grevlex in Maple and deglex in Giac. Other programs name
the orders as follows.

| Program | grevlex | deglex | lex |
|---|---|---|---|
| CoCoA | `DegRevLex` | `DegLex` | `Lex` |
| Giac | `revlex` | `tdeg` | `plex` |
| Groebner.jl | `DegRevLex` | `DegLex` | `Lex` |
| Macaulay2 | `GRevLex` | `GLex` | `Lex` |
| Magma | `"grevlex"` | `"glex"` | `"lex"` |
| Maple | `tdeg` | `grlex` | `plex` |
| Mathematica | `DegreeReverseLexicographic` | `DegreeLexicographic` | `Lexicographic` |
| Oscar | `degrevlex` | `deglex` | `lex` |
| Sage | `degrevlex`, `dp` | `deglex` | `lex`, `lp` |
| Singular | `dp` | `Dp` | `lp` |
| SymPy | `grevlex` | `grlex` | `lex` |

The first variable of the instance is the largest in every order.

## Computation

### Over ℚ

A call over ℚ learns at two primes below 2^31 at once and computes the basis over
primes of [2^61, 2^62), each drawn from the
seed, and lifts it to ℚ by the Chinese remainder theorem and rational
reconstruction. It returns the lifted basis once the basis at one more prime
agrees with it: the learn run's second, or a fresh prime below 2^31 where
the learn run fell back to its first prime alone or its second divides a
denominator of the lifted basis. With the checker on, the checker rules on the lifted basis
and its lifted cofactors exactly over ℚ or at points, as the call asks.

### Bounds

A deadline, a cancel flag, and a memory ceiling each stop the call. The
result then carries no basis and names the bound in `stopped`. The call
checks its bounds lazily, so it stops at its next check, not at once. The
memory ceiling is read from the resident set at those checks, so a call may
pass it briefly. Before the ceiling stops a call, the call gives back the
memory glibc's malloc holds free, and at the start of a round or of the
inter-reduction, or over ℚ before a prime, the memory it can rebuild. A
hard ceiling is the operating system's: `ulimit` or
a cgroup.

A call checks its bounds as it reads the instance, at the start of each run,
at the start of each round, after
every 64 of its rows or other steps, and once the call has done `WORK`, 2^23, units of work since
the last check. Every loop whose work grows with the input spends units as it goes, a unit about an
entry of a row operation or a byte of a text, so that a check comes within `WORK` units of any
loop's work; the checker takes a step per 4096 of its own units, a unit about the work of one term
(_Limits_). No check follows a call's last text but the checker's, where the call asks for it.

### Checker

The checker is a separate crate, `grobx-checker`, that shares no code with
the engine. It gives two verdicts:

- **Gröbner basis of ⟨F⟩.** Every generator reduces to zero modulo G, every
  S-polynomial the criteria do not excuse reduces to zero, and the row of
  every nonzero element of G in the cofactor matrix expresses it in F. The first two give
  ⟨F⟩ ⊆ ⟨G⟩ and that G is a Gröbner basis, the third ⟨G⟩ ⊆ ⟨F⟩.
- **Reduced.** Every element is nonzero and scaled as the basis text
  scales it, and no term of an element is divisible by the leading term of
  another.

`certified` is true where both hold. The exact mode expands the row of
every nonzero element. The cheap mode instead evaluates both sides of
g_i = Σ_j c_ij·f_j at k random points and compares them. A false identity
of total degree d survives one point over GF(p) with probability at most
d/p, and k points with at most (d/p)^k. Over ℚ the points are taken modulo
random primes. The checker needs the cofactors, so a call with the checker
on tracks them, and its time includes tracking and checking.

The cheap mode draws its points in batches of as many points as fit 16 MiB
and at least one, and reads the
cofactor text once per batch. Its memory grows neither with k nor with a
row's entries, and the result is that of all k points at once. A row's
entries past one per generator, which make the row `Shape`, are read at no
point. A call at any k from 1 to
2^32 − 1 returns its verdict or stops at its bounds. Over ℚ a row whose
degree bound reaches `LEAST_MODULUS`, 2^31, is checked exactly.

`grobx_checker::check(instance, basis, cofactors, order, mode)` runs the
checker on texts from any source: the instance text, a basis text, and a
cofactor text as above. It reads the instance text as `grobx::groebner`
does: what that call refuses, it refuses with an error of the same kind,
save more generators or monomials than a u32 id numbers, which it reads.
Its `Order` is `Grevlex`, `Lex`, or a nonsingular n × n
integer weight matrix, the first nonzero entry of each column positive,
which gives block and elimination orders.

## Design

This part describes how the engine computes a basis.

### A call, end to end

1. The reader reads the instance into the narrowest monomial format that holds it, and the entry
   point picks the field and the order once and the format at each start, each combination its
   own compiled copy of the engine. The call parses the generators once, and each run maps them to
   its field and format. Before it computes, a run that tracks no cofactors substitutes the
   variables its linear generators lead in its other generators and puts the generators in reduced
   row echelon form (`Params::echelon`), save the plain F4 runs over GF(2) and, over ℚ, the runs of
   a call that asks for cofactors or the checker. A replay does as its record says.
2. Over GF(p) in grevlex and in deglex, the signature run computes the basis. Where it
   stops paying, it hands the call to the plain F4 run, which goes on
   from the signature run's work where that work was large and no trace, learner, or cofactor
   tracking listens, and starts over otherwise. Over GF(2) a call without cofactors or the checker
   starts in the plain run, and starts over in the signature run where the plain run's row
   work passes a bound.
3. Both runs build one matrix per round and reduce it. The plain run
   inter-reduces its basis before a round where an element of degree 1 has joined it since, over
   GF(2) once their count has grown by a quarter (_Matrices and reduction_ below). A last matrix
   inter-reduces the basis, and the writer writes the canonical text (_The result_ below).
4. In lex, the call computes the grevlex basis and changes it to lex by
   FGLM, or, where FGLM does not apply, runs F4 in lex as well.
5. Over ℚ, the call learns at two primes at once, replays at more primes,
   lifts the images to ℚ, and stops once one more prime agrees, as a rule
   the learn run's second.

### Monomials

- **One table per run.** A monomial is a u32 id into a hash table that lives for the whole run.
  The hash is additive, Σ r_i·e_i with r_i drawn from the seed, so a product's hash is the sum of
  its factors' hashes, and a reducer row's products are looked up by a sum of two hashes. A run
  that starts over builds a new table, and so does the plain run that
  goes on from the signature run's work, into which it carries that work's monomials alone.
- **The plain run's table compacted.** At a round's end, once the table has interned
  `COMPACT_GROWTH`, 2^14, monomials since its last compaction, the plain run keeps only the
  monomials its basis and the matrix it keeps for the next build hold, so the table grows with
  the basis rather than with every round's matrix. A run whose trace listens does not compact,
  since a trace names each row's multiplier by its id. The signature run and the replays keep
  every monomial.
- **Divisor masks.** Every monomial carries a 32-bit mask with fixed thresholds, and every
  divisibility test between monomials of the table reads the masks before the exponents. Up to 32
  variables each variable takes ⌊32/n⌋ bits, bit k set where its exponent passes k. Past 32
  variables variable i sets bit i mod 32 where its exponent is not 0, so a bit stands for several
  variables. Two monomials whose masks share no bit share no variable. Up to 32 variables, masks
  that share a bit mean a shared variable. Past 32 the exponents decide. The mask of an lcm is the
  OR of its factors' masks (_Pairs_ below).
- **Dense formats with a restart.** A monomial is its total degree and one lane per variable of
  4, 8, 16, or 32 bits: `Dense4`, `Dense8`, `Dense16`, and `Dense32`, of degree at most 7, 127,
  32767, and 2^31 − 1. A call starts in the narrowest format that reads its input: `Dense4` over
  GF(2), `Dense8` over every other field. Where a product would pass the format's degree bound,
  the call starts over from its input in the next wider format, under the same seed and the
  budget left, so narrow lanes cost nothing in correctness.
- **Packed words.** The lanes stand in 64-bit words, and no exponent passes the degree bound, so
  every lane keeps its top bit clear. A product is the word sum, a divisibility test subtracts the
  words and reads the top bits, and an lcm takes each lane's maximum from the top bits: each is a
  few word operations whatever the lane width. A block order reads lane by lane.
- **Supports past eight words.** A ring takes a format of more than eight words past 128
  variables in `Dense4`, past 64 in `Dense8`, past 32 in `Dense16`, and past 16 in `Dense32`. There
  each monomial keeps its support, a 64-bit mask of the groups of its words that hold a nonzero
  word, and a product is checked on the groups the factors' supports mark alone, so a sparse
  monomial of hundreds of variables costs the words that hold its variables.
- **One compiled copy per field, format, and order.** Cofactor tracking and the trace are type
  parameters whose empty versions compile to nothing, so a call that asks for neither pays nothing
  for them.

### Pairs

- **Gebauer–Möller at insertion.** The criteria B_t, M, F, and the product criterion run as each
  element enters the basis, in the variant that ignores sugar. B_t departs from Gebauer and
  Möller's, which keeps a pair (i, j) wherever (i, t) or (j, t) has its lcm: it drops (i, j), whose
  lcm lm(t) divides, also where one of (i, t) and (j, t) has coprime leading monomials, since a
  pair the product criterion settles goes on account of no other pair (roune-2012 §D.1).
- **The entering order by the field.** The plain run enters each round's new elements, and the
  inputs, one at a time through the update. Over every field but GF(2) they go by decreasing
  leading monomial, so that an element is redundant exactly when a later one's leading monomial
  divides its own. Over GF(2) they go by increasing leading monomial, so that of two elements
  whose leading monomials divide one another the smaller enters first and the larger joins no
  G_min. The replays over ℚ enter in the order the learn's record names.
- **The new pairs by increasing degree.** The product criterion drops a new pair of coprime leading
  monomials before its lcm is formed. M and F run over the other new pairs by increasing lcm
  degree, each against the pairs kept so far, and of each lcm the last pair in G_min's order stays.
  Only a kept pair's lcm enters the monomial table, at the selection that takes the pair.
- **A divided element's one pair.** Where the leading monomial of a member g of G_min divides
  lm(h), lcm(g, h) is lm(h) itself, and it divides the lcm of every other new pair of h
  (eder-2012 §2.4, Lemma 2.4.1), so M and F keep one new pair of h. The update searches G_min for
  the last such member before it forms any pair, and forms that pair alone.
- **The criteria by masks and indexes.** The pair set keeps each pair's lcm mask beside it, and
  B_t reads a pair's lcm only where its mask passes. M and F compare lcms by their colons over
  lm(h): where a key bit stands for one variable, a kept pair whose colon is one or two variables
  stands in bit masks, so a new pair finds such a divisor without a scan, and only the kept pairs
  of other colons are scanned. Past 32
  variables, where a mask bit stands for several variables and the masks pass most pairs, the plain
  run keeps bitmap indexes of the pair set's lcms and of G_min's leading monomials, a bit set per
  bit of a 128-bit support, so B_t, the new pairs, and the search for a divided element's divisor
  visit only the candidates whose supports admit them.
- **Selection by degree in grevlex and deglex, by sugar elsewhere.** In a graded order a round
  takes every
  pair whose lcm has the least total degree, a fallen element's pair counted one degree past it
  until the pair set has grown past its bound (below). In lex and block orders a round takes every
  pair of least sugar. A degree or sugar of many pairs is split (_The pairs of a large round_
  below). A new element's sugar is the larger of the round's sugar and its own largest total
  degree. The input is never homogenized. A pair's sugar past 2^32 − 1 is `Overflow`, as an lcm
  past the format is.
- **A fallen element's pairs one degree later in a graded order.** An element fell where its sugar
  lies above its leading monomial's degree: the round that made it had a higher degree than its
  leading monomial. In a graded order a pair one of whose elements fell counts one degree past its
  lcm's, so the elements of low degree a fall brings enter, and their criteria drop the pairs they
  make redundant, before those pairs of the round's degree are reduced, until the first update that
  begins with more than `DEFER_LIVE`, 128, live pairs in the set ends the step for the rest of the
  run.
- **The pairs of a large round.** Where the least sugar holds more than `SPLIT_LEAST`, 2,048, live
  pairs, the plain run's selection takes the `SPLIT_TAKE`, 512, of them whose lcms are least in the
  run's order and leaves the rest in the set, where the new elements' B_t may drop them before a
  later round takes them. No pair a split leaves goes but by the criteria, so the basis is the one
  the whole round would give. Over every field but GF(2) a sugar splits once, and the next
  selection takes the rest whole. Over GF(2) the sugar splits again, piece by piece, while it holds
  more than `SPLIT_TAKE` live pairs and at least half the set, and the last piece made elements from
  fewer than half its pairs or made an element that fell: the first shows that most rows of the
  whole sugar would reduce to zero, the second that the new elements' pairs of lower sugar will
  take most of the sugar's rest away. The round line's `deferred` counts the pairs a selection left
  (_Report text_ above).

### Matrices and reduction

- **The inputs.** Before a run computes, it puts its generators in reduced row echelon form, as
  vectors over their monomials, where it takes the echelon (_A call, end to end_ above): a forward
  pass that pivots each column on its sparsest row, and a back pass in dense rows of the field. A
  generator that depends on the rows that won columns before it becomes zero. Over GF(2) the echelon
  also leaves out, as zero, each row x_v^2 + L that a linear row x_v + L and the field equations
  x_j^2 + x_j of L's variables put in the ideal of the others (`echelon::squares_out`): such a row
  is (x_v + L)^2 + Σ_j (x_j^2 + x_j), and the signature run would reduce it to zero at its own
  signature only after forming its S-pairs with every other input. A run that learns or replays
  leaves no row out, since a replay's echelon must reduce to zero each row its record names zero.
- **The linear generators substituted.** A run that takes its generators in reduced row echelon
  form (**The inputs** above) first reduces its other generators by its linear ones, those whose
  every term has degree at most 1. Each row of the linear generators' reduced row echelon form that
  leads with a variable gives that variable a value, minus the row's other terms, and every other
  generator has those variables replaced by their values, by Horner's rule nested over the ones it
  holds. The substitute takes the generator's place where it has fewer terms, and a substitution
  that reaches a sum as long as the generator ends there, the generator kept as read. The linear
  generators stay, every variable stays in the ring, and the substitution runs once per run.
- **A replay's echelon from its record.** A replay whose record's learn run took the echelon takes
  its pivots from the record: each generator the record names nonzero is reduced once by the pivot
  rows of the columns before its own, and each it names zero is reduced in full, an entry left a
  misfit at the inputs.
- **Rows from the basis alone.** Each round builds its rows from the current basis, with no
  Simplify. A row the matrix before built under the same (element, multiplier) is taken from it as
  built, not reduced. A row t·g holds only column indices and reads its coefficients from g. In the
  plain run the pivot of each column is the sparsest candidate. The signature run's is in
  _Signatures_.
- **The basis inter-reduced where an element of degree 1 joins it.** A plain round makes its new
  elements fully reduced by its own matrix's pivots, but an element's tail is not reduced by the
  elements later rounds make, and every reducer row taken from it carries that tail. An element of
  degree 1 is the one whose leading variable divides many monomials of those rows. So at the start
  of a round where `G_min` holds more elements of degree 1 than at its last inter-reduction, over
  GF(2) a quarter more (`TAILS_GROWTH_PART`), the plain run inter-reduces `G_min`, and each element
  whose tail changes keeps its leading monomial, sugar, and pairs. The strategy is Buchberger's,
  each new polynomial reducing the others (buchberger-1985 §6.4 (2), after buchberger-2006 §9.4),
  taken where an element of degree 1 joins alone, since after every round the inter-reductions
  cost more than they spare. Only the elements whose tails hold a term that a leading monomial
  joined since divides are given rows, by the record of when each element was last reduced
  (makarim-2017 §4.1, via hauke-2022 §2.2). A run whose trace or learner listens, or that tracks
  cofactors, keeps its elements as made.
- **The divisor search.** Symbolic preprocessing in the plain run takes, for each monomial new to
  the round but the given rows' leading monomials, the element of G_min of fewest terms whose
  leading monomial divides it, ties to the element made first, by divisor masks read eight at a
  time. Past `FILED_PAST`, 64, variables, where the masks pass most elements, G_min is filed by a
  bit of each leading monomial's support, and the search visits only the files of the monomial's
  support bits.
- **Columns of 2 bytes.** A matrix's rows hold column indices alone, their coefficients their
  elements'. Where the round's monomials number at most `NARROW_COLUMNS`, 2^16, an entry takes 2
  bytes, and past that 4, so a narrow matrix's rows take half the memory. The round and inter lines
  name the width, `index`.
- **Rows to reduce from the right.** The plain run's pivot rows stay unreduced. It reduces each row
  without a pivot fully, through a dense accumulator, by the pivots and the new pivots before it,
  and a nonzero result becomes a new pivot. The rows go by decreasing leading column, so the short
  sweeps on the right make the round's new pivots first, and a block that leads far left finds its
  rows in the span of the rows before it. The run then back-substitutes the new rows, so a round
  ends in reduced row echelon form in every order. Over `Gf31` the accumulator holds `i64` and
  reduces an entry mod p once, when the loop reaches its column.
- **Random combinations of row blocks.** The plain run cuts the rows without a pivot into blocks of
  `BLOCK`, 64, consecutive rows from the right, and reduces random combinations of each block until
  `ZEROS`, 2, of them reduce to zero or the block has made as many pivots as it has rows, so a
  block whose rows lie in the span of the pivots costs two reductions, not one per row. A block
  that made no pivot doubles the next, and a grown block stops once it has made more than
  `GROWN_PIVOTS`, 16, pivots. The run reduces each row on its own where p lies below
  `COMBINES_FROM`, 46341, the least p with p² ≥ `ZERO_BOUND`, 2^31, where the call tracks
  cofactors, and where a trace or a learn run listens; a learn run whose trace does not listen and
  that tracks no cofactors tests those rows (_Rows ruled zero by random combinations_ below).
- **Short rows reduced each on its own, chosen per round.** A plain round that combines rows,
  whose matrix takes no bands and holds `BLOCK` + `PROBE`, 72, rows to reduce or more, reduces the
  rows after its first block either as random combinations of blocks or each on its own with the
  tests of _Rows ruled zero by random combinations_ below. A probe decides: it reduces 8 rows of
  the second block each on its own and compares their cost with that of the first block's
  combinations, weighed by the share of the run's rows that made pivots. The run probes again
  where the rows to reduce double or halve, or after `REPROBE`, 4, rounds. The reduced row echelon
  form is the same in either mode, so the basis stays. The round line names the mode (_Report
  text_ above).
- **Reductants in bands of four.** Over `Gf31` and `Gf2`, where no cofactors are tracked, `BAND`,
  4, rows are reduced in one sweep over the columns, which reads each pivot row once for the four:
  a signature round's consecutive reductants and the plain run's random combinations in a matrix
  whose rows hold `BAND_ENTRIES`, 64, entries or more in the mean, and the back-substitution and
  the final inter-reduction in every matrix. Every band row comes out entry for entry as one row at
  a time makes it. A round that reduces in packed rows takes no band.
- **The early pivots reduced once, in packed rows.** Over GF(2), where no cofactors are tracked, a
  round whose matrix holds `PACKED_ENTRIES`, 100,000, entries or more, in rows of
  `PACKED_ROW_ENTRIES`, 16, entries or more in the mean, reduces its early pivots, those set before
  its first row is reduced, once among themselves, to rows over the other columns, the K columns,
  packed 64 to a word, where an addition is an exclusive or. A row to reduce then adds the packed
  row of each early pivot it meets, so its reduction by them is word operations. The rows made,
  the rows that reduce to zero, and the pivots are those of the sparse reduction. The K columns
  take at most `PACKED_WORDS`, 256, words, and the early pivots' packed rows at most
  `PACKED_TABLE`, 256 MiB. The round line's `packed` names the words.
- **The back-substitution in packed rows, by tables of row sums.** A plain round that reduces in
  packed rows back-substitutes its made rows in them, `TABLE_ROWS`, 8, at a time: a table holds the
  2^8 sums of a block's rows, and each later row adds the one sum its bits at the block's leading
  columns name, in place of up to eight rows. This is Bard's method of four Russians inversion,
  M4RI (bard-2006 §3 and §3.1).
- **Rows ruled zero by sketches over GF(2).** Over GF(2) the plain run reduces each row on its own,
  as 2 lies below `COMBINES_FROM`, and most rows reduce to zero. Once `STREAK`, 2, rows of a round
  have reduced to zero one after another, the reducer draws a sketch: a random 128-bit word per
  column without a pivot, and per pivot column the exclusive or of the words over its pivot row's
  tail. A row's sketch, the exclusive or over its columns, is 0 for a row in the span of the
  pivots and uniform outside it. Reduced by an echelon of the sketches of the rows made since, a
  residue of 0 rules the row zero without its reduction. A row that does not reduce to zero passes
  with probability 2^(e − 128) for e residues in the echelon, so the sketch is drawn again at
  `SKETCH_ECHELON`, 97, residues, which keeps that below 2^−31. The words come from the call's
  seed. A round takes sketches where its run may combine rows, with no trace, learner, or
  cofactors, and where its rows hold `SKETCH_ROW_ENTRIES`, 16, entries or more in the mean.

### Fields

- **Three fields, chosen once per call over GF(p).** `Gf2` takes the prime 2, `Gf31` the other
  primes below 2^31 with `u32` coefficients, and `Gf64` the primes from 2^31 to 2^64 with `u64`
  coefficients, a row operation multiplying by the multiplier's pre-inverse below 2^63 and by a
  `u128` remainder past it. A call over ℚ runs its learn runs in `Gf31Pair`, over two primes at
  once, its fallbacks in `Gf31`, and its replays in `Gf64` or `Gf64Pair` (_Over ℚ_).
- **A field of bytes for GF(2).** Over GF(2) a coefficient is a byte, and a row operation is an
  exclusive or that reads no coefficient of its pivot row, every stored coefficient being 1.

### Signatures

- **A batched F4 over signatures.** Each round takes every pending signature of least degree whose
  row a·g an element before g in the order (sig-lead ratio, index) top-reduces (lairez-2022 §5.7,
  Algorithm 8), and reduces the rows in one matrix, in Schreyer's order on signatures.
- **Syzygies from three sources.** The run learns a syzygy signature from every reduction to zero,
  from the Koszul syzygies of the inputs, and from coprime leading monomials, and queues no
  signature they divide. A new syzygy does not prune the queue. It keeps the minimal ones of the
  reductions to zero, and the inputs' Koszul signatures lm(f_i)·lm(f_j)·e_j, i < j, it tests from
  the inputs' leading monomials alone: Faugère's criterion, against the inputs of lower index, not
  every element of lower index as in F5.
- **The S-pairs of a new element.** When an element enters, the run picks two earlier elements of
  largest sig-lead ratio as its base divisors: one whose leading monomial divides the new element's,
  and one whose signature divides its signature. A triangle of bits records which S-pairs are known
  to have a syzygy signature, and an S-pair of the new element goes without its signature formed
  where a base divisor's S-pair with the same element has its bit set and roune-2012 Theorem 7 or 8
  makes that signature divide the S-pair's. An S-pair whose two elements have the same (sig-lead
  ratio, index) is dropped. The base divisors eliminate only S-pairs the signature criterion
  eliminates, so every round reduces the same rows. A full check at a round's start or the run's end
  that finds the resident set past the ceiling gives the triangle back, and the run goes on without
  base divisors. An S-pair they would have eliminated is then formed, and where its signature
  passes the format's degree bound and its leading monomials are not coprime, the call starts over
  in the next wider format.
- **Rewriting by the sig-lead ratio.** The row at a signature σ is a·g for the g of largest
  sig-lead ratio whose signature divides σ. Symbolic preprocessing in the signature run takes, for
  each monomial new to the round, the first element in the order (sig-lead ratio, index, terms)
  whose leading monomial divides it, and adds no reducer whose signature reaches the round's
  largest reductant signature: the monomial then has no reducer. Two ratios s_a/l_a and s_b/l_b
  compare as s_a·l_b against s_b·l_a, word sums (_Packed words_ above) that carry no lane into the
  next even past the degree bound, so no comparison restarts the call.
- **The F4-RB matrix.** Only the reductant is a given row, and symbolic preprocessing supplies the
  rest, a reducer row the matrix before built taken from it as built. Rows are reduced in
  increasing signature order. A reducer row whose column already has a pivot is dropped where the
  row before it has its signature, and is reduced as a reductant otherwise.
- **Rows ruled zero by random combinations.** Within a stretch of a signature round's rows that no
  reducer row setting a pivot interrupts, every row is reduced by the same pivots. Once `STREAK`,
  2, rows of a stretch have reduced to zero one after another, a test takes the rest of it: where s
  random combinations of its rows each reduce to zero, every row joins the syzygies. s is the least
  integer with (p − 1)^s ≥ `ZERO_BOUND`, 2^31, 2 at 65521, so a test passes a row that does not
  reduce to zero with probability at most 2^−31. After a failed test the tests take blocks of
  `FIRST`, 8, rows, doubling up to `BLOCK`, 64. No test runs at p = 2 or 3, where cofactors are
  tracked, where a trace listens, or in a learn's signature run; over GF(2) the plain run rules rows
  zero by sketches instead (_Matrices and reduction_ above). A learn run's plain run, where its
  trace does not listen and it tracks no cofactors, tests its rows to reduce, each reduced on its
  own, the same way, and a row a passing test rules zero makes no pivot, so the record is the one
  each row on its own makes (_The record_ below). Over `Gf31Pair` a combination reduces to zero
  where it does at both primes.
- **The handover.** The signature run decides once at the gap round: the first round that makes an
  element whose signature degree passes its leading degree by `HANDOVER`, 4, or more. The gap round
  hands over where it reduced fewer rows to zero than `GAP_ZEROS`, a half, of the elements it made,
  or where more than `DIVISIBLE`, 6 in 10, of them are divisible: a leading monomial of the basis
  before the round divides theirs. Otherwise the signature run keeps the call. The outgrown test
  reads every round that made an element, and hands over where `OUTGROWN`, 8 in 10, or more of them
  are divisible. At a handover the call starts over in the plain run from its input, where the
  signature run does not carry its work (**The plain run goes on from the signature run's work**
  below). A run that tracks cofactors never hands over.
- **The early round.** The signature run decides once at its early round as well: the first round
  that makes an element whose signature degree passes its leading degree by `EARLY`, 2, or more.
  Where that round made its least elements or more, it hands over where it reduced fewer rows to
  zero than `EARLY_ZEROS`, a quarter, of the elements it made, or where more than `DIVISIBLE`, 6 in
  10, of them are divisible. Otherwise the run goes on, and no later round decides by these tests.
  A round that reduced no row to zero is read as well. Where the early round is the gap round, its
  tests run first, and the gap round's on the same counts. The least is `EARLY_ROUND`, 25, where
  the plain run after a handover would combine rows, and `EARLY_ROUND_ALONE`, 100, where it would
  reduce each row alone: where it listens, tracks cofactors, or runs at a prime below 46341.
- **The pair work.** Where the early, the gap, the outgrown, and the large-round tests kept the run,
  the signature run reads every round whose update would form `PAIRS`, 1,000, S-pairs or more, from
  its first round on. Where m elements enter a basis of B, the update forms m·B + m·(m − 1)/2
  S-pairs. The round hands over where they reach `PAIR_WORK`, 1, times the entries its row
  operations subtracted, before the update forms any.
- **The large round.** Where the early, the gap, and the outgrown tests kept the run, the signature
  run reads every round of `LARGE`, 150, elements or more. The round hands over where
  `LARGE_DIVISIBLE`, a quarter, or more of its elements are divisible and it reduced fewer rows to
  zero than `LARGE_ZEROS`, a tenth, of them. The pair-work test reads the same round after it, where
  it kept the run. The count of divisible elements is the one the outgrown test reads.
- **The plain run goes on from the signature run's work.** Where neither the trace nor the learner
  listens, no cofactors are tracked, and the signature run has subtracted `CARRY_WORK`, 100,000,
  entries or more since it started, the plain run goes on from that work in place of the input. It
  takes the signature run's elements, and the elements its handover round
  made. Each element's sugar is its signature's degree, and the round's elements take the round's
  degree. The update leaves out each pair of two carried elements whose sugar is at most the
  carry's bound: the round's degree d, or d − 1 where an older element's leading monomial divides a
  made element's at a multiplier whose degree plus the older element's sugar is at most d. The
  carried elements enter as an input's do, in the field's entering order (_The entering order by
  the field_ above), and the plain run takes a fresh table into which their monomials alone move.
  Over ℚ a learn's runs listen and a replay runs no signature run, so they start over, and the
  verification's run at a fresh prime may carry.
- **The plain run first over GF(2).** Over GF(2), whose rounds reduce in packed rows
  (`Field::PACKS`), a call that tracks no cofactors and asks for signatures in a graded order starts
  in the plain run under a bound on its row work, the bounded run, which takes the generators as
  read. Once the row work, the entries the sparse row operations subtract and add and the words the
  packed rows XOR, passes `PLAIN_WORK`, 2^32, the run halts, and the call starts over from its input
  in the signature run, with the same seed and the budget left. Every later start of the call is the
  signature run's, or after its handover the plain run's, and neither runs under the bound. A run
  that tracks cofactors starts in the signature run.
- **The bounded run halts at its projected row work.** The bounded run also halts where its row
  work so far, plus the round's row work projected over the round's rows to reduce and, where the
  round takes a piece of its sugar, over the whole sugar, passes the bound, so a round whose work
  would pass it halts the run early in the round, not at its end. The projection holds in rounds
  that reduce in sparse rows alone, once the reduction has begun `PROJECTED_ROWS`, a block of 64,
  of the round's rows.

### Lex and FGLM

- **Grevlex, then FGLM.** In lex the call computes the grevlex basis and changes it to lex by FGLM.
- **F4 in lex as the fallback.** Where the ideal is not zero-dimensional, or its staircase passes
  `MAX_STAIRCASE`, the call starts over in lex from its input, after the grevlex run has given back
  its memory.
- **A staircase past the format's degree bound.** A staircase of more
  monomials than the format's degree bound, 7 in `Dense4`, 127 in `Dense8`, and 32767 in `Dense16`,
  starts the call over from its input in the next wider format, its grevlex run included, so a lex
  call may compute its grevlex basis once per format it passes through.
- **Matphi.** Every border monomial's normal form is computed once, before the change.
- **NewBasis.** Candidates wait in a heap under lex, and linear dependence is tested against a
  semi-echelon form that carries its combinations. For n variables and a staircase of D monomials,
  Matphi costs O(n·D³) operations (faugere-1993), and the change as a whole as much.
- **Cofactors in lex.** Each lex element is divided by the grevlex basis, and the quotients are
  composed with its cofactors.

### Trace and replay

- **The record.** A learn run reduces each row on its own and records only the matrices that made a
  new element, and of each the rows a replay needs: the counted rows and, to closure, the pivot row
  of each column a needed row holds. Per row the record holds its element and multiplier, in the
  learn run's order, and per counted row the leading monomial it must reach. A replay builds those
  rows straight from the record, with no pairs and no symbolic preprocessing, and repeats the learn
  run's arithmetic. A learn run's plain run takes its rows to reduce by decreasing leading column,
  so its counted rows are those of the shortest sweeps. Where no trace listens, the learn's plain
  run rules rows zero by tests (_Rows ruled zero by random combinations_ above). A row a test rules
  zero is recorded as a row that reduced to zero.
- **Misfits.** A replay checks the generators and the leading monomial of every counted row it
  reduces, and stops at the first that differs with `Error::Misfit`.
- **Lex records.** A lex record whose basis FGLM changed holds the grevlex steps and a flag, and its
  replay runs FGLM after them. Any other lex record holds the lex run's steps.
- **The rows the output depends on.** Once the rounds of a learn run of the lift end, its record
  keeps of each step only the counted rows the run's output depends on, and the replays skip the
  others. The output is `G_min` of the run's basis, which the inter-reduction reduces, or the
  constant the run's last step made. From the last step to the first, an element is needed where
  it lies in `G_min` or where a kept row of a later step multiplies it, and a counted row is kept
  where its element is needed or where it reduced to the constant, closed within its step over the
  rows that hold the pivots of its columns. A skipped row's element enters a replay's basis as its
  leading monomial alone, so that each later element keeps its id and `G_min` is the learn run's.
  `grobx::learn` returns its record whole. A replay checks no row it skips, so over a prime where a
  skipped row alone does not fit, a replay of a cut record returns a basis that is not the
  system's, where a replay of a whole record returns `Error::Misfit`. Over ℚ the verification
  guards the lift.

### The result

- **Each monomial's text written once.** The basis text over GF(p) writes each distinct monomial's
  text once per call and copies it for each further term with that monomial.
- **Integers over ℚ by divide and conquer.** An integer past `SPLIT_LIMBS`, 12, limbs is written by
  splitting it at a power 10^(19·2^j), the quotient by Barrett's reciprocal, down to parts of at
  most `LEAF_CHUNKS`, 8, chunks of nineteen digits, each written by repeated division by 10^19.

### Over ℚ

- **Learn once, replay many.** The lift learns at one prime and replays at the others. Each prime
  is the least prime at or above a random draw from the seed.
- **The generators parsed once.** A call parses the generators once, into terms that name their
  monomial among the text's distinct monomials and keep their numeric factors, and each run maps
  that form to its prime and its table, the distinct monomials sorted once into the run's order. A
  literal of at most nineteen digits is kept as its value, and a longer one as its digits. A text
  that breaks the grammar keeps the terms before the break, so that a denominator 0 mod p before
  the break stays the error (_A denominator 0_ below).
- **Wide primes for the replays.** A replay's prime lies in [2^61, 2^62) and its run in `Gf64`, so
  each replay adds about 61 bits to the modulus, while the learn runs and the verifications keep
  their primes below 2^31.
- **Two primes per replay.** A replay that tracks no cofactors, of a record FGLM did not change,
  runs over two primes of [2^61, 2^62) at once, in `Gf64Pair`: the primes share the replay's rows,
  columns, and pivots, and a row operation does both primes' arithmetic in one pass over the row.
  The lift takes the first prime's image without an attempt at reconstruction and attempts at the
  second, from the modulus times both. Where
  a leading coefficient the replay inverts is 0 at one prime, or a denominator vanishes at either,
  each prime replays alone.
- **Unlucky primes.** The lift skips a prime that divides a denominator, and drops one whose replay
  misfits. Two misfits in a row drop the record, and the next prime learns afresh. A verification
  whose leading monomials or an element differ drops the record too (_One more prime ends the
  lift_ below). The fourth drop in a call returns `Error::Drops`, whatever the drops' whys. A skip
  and a pair learn that falls back drop no record.
- **A denominator 0.** The parse stops at a denominator whose digits are all `0` with `NoInverse`
  at its line, as it stops at a break of the grammar. A read at any prime returns it at its place
  in the text's order, after the errors over that prime of the terms before it. Where a learn run's
  read stops with `NoInverse` at the line of the parse's `NoInverse`, the call returns the error. A
  read's `NoInverse` at any other line names a prime that divides a nonzero denominator: a pair
  learn falls back to the learn run at its first prime alone, and a learn run at one prime skips
  its prime. The error is the one a denominator 0 mod p is over GF(p), and the checker's reader
  refuses the text alike.
- **One more prime ends the lift.** Once every element is reconstructed, the basis of a full run at
  one more prime must agree: the learn run's at its second prime, or a run of its own at a fresh
  prime where the learn run fell back or its second prime divides a denominator of the lifted basis
  (_The verification beside the learn run_ below). The returned basis is correct with high
  probability. A proof over ℚ is the checker's. A run of its own records nothing: it runs as a call
  over GF(q) runs, apart from the record, and its plain run combines rows. Where its leading
  monomials or only its coefficients differ from the lifted basis, the lift drops the record and
  what it lifted, and the next prime learns afresh.
- **The verification beside the learn run.** Every learn of the lift starts over two primes of
  [2^30, 2^31) at once, in `Gf31Pair`: the learn prime and the verification's, drawn right after
  it. The two primes share the run's pairs, symbolic preprocessing, matrices, and handover, and a
  row operation does both primes' arithmetic in one pass over the row. The record is the first
  prime's. The basis at the second is kept until every element is reconstructed, and verifies the
  lifted basis in place of a run of its own. Where a leading coefficient the run inverts is 0 at
  one prime alone, the run halts, the lift learns at the first prime alone, and the verification
  runs at a fresh prime, as where a denominator of the input is 0 at either prime.
- **Reconstruction over the chain.** The coefficients of an element share their denominators. A
  part keeps its rationals as numerators over a chain of denominators, each the lcm of the one
  before and the denominator of a rational an attempt found. An attempt scales a coefficient's
  images by l, l the part's chain's last, and converts them to v = u·l mod M, u the coefficient's
  residue modulo M, taken in (−M/2, M/2]. Where |v|·2^(T+1) < M, the rational is v/l, with no
  Euclidean step, so a coefficient whose denominator the chain already holds costs no
  reconstruction. Elsewhere the maximal quotient rational reconstruction, MQRR, takes the rational
  n′/f before the largest quotient of the Euclidean algorithm on (M, v mod M), where that quotient
  is at least 2^T, and the coefficient is n′/(l·f); the chain's next is l·f. T is ⌈log₂ L(M)⌉ +
  `MARGIN`, L the bit length and `MARGIN` 10. MQRR runs in Lehmer's form on 128-bit leading parts,
  and an attempt first takes the rational before the first quotient of at least 2^T where it fits
  at the next prime, running to the largest quotient only where it does not.
- **Element by element, incrementally.** Elements are reconstructed in increasing order of their
  leading monomials, and an attempt resumes at the coefficient where the last one stopped. An
  element's rationals are checked at the prime after the modulus they were reconstructed from, and
  once it is kept, at no further prime but the verification's (_One more prime ends the lift_
  above).
- **Kept elements leave the combination.** Once an element is kept, the lift neither combines nor
  checks it, reads none of its images, and drops those it holds. An image that gains or lacks a
  term of a kept element changes nothing. The verification, at the learn run's second prime or at a
  fresh one, is a kept element's one check after its keep.
- **Images and their conversion.** Each open coefficient keeps its image at every prime combined,
  one word a prime, and no residue modulo M is formed before an attempt asks for one. An attempt
  takes v's mixed-radix digits from the images and then v by Horner's rule, and ends the
  conversion early at a digit that shows v small or close to M, converting in full only where the
  value so found does not fit at the next prime. The check of a rational n/d against c at a prime
  compares n with c·d, with no inverse.
- **A coefficient at the primes it needs.** Once a coefficient of an element fitted in an attempt,
  a further coefficient of its part is reconstructed from its images at the first j primes of M
  alone, j the fewest multiple of `PREFIX`, 4, whose bits reach those the part's earlier
  coefficients asked for plus `SLACK`, 128 bits, and modulo M only where it does not fit there, so
  a coefficient of few bits costs a reconstruction of few bits however large M has grown.
- **Cofactors over ℚ.** The learn runs and the verification's run track no cofactors, so their
  signature run may hand over. The replays track them where the call asks for cofactors or the
  checker, and a cofactor row is lifted with its element.

### Checker

- **Independent and streaming.** The checker is its own crate and reads texts. It holds G and one
  row of the cofactor matrix at a time.
- **What it checks.** Two verdicts. G is reduced: every element is nonzero and scaled as the
  canonical text, and no term of an element is divisible by another's leading term. G is a Gröbner
  basis of ⟨F⟩, in turn: the generators reduce to zero, the rows of the cofactor matrix hold,
  and every S-pair the product and chain criteria do not excuse reduces to zero.
- **Plain arithmetic.** `u64` residues with `u128` products for every p, a geobucket, and divisor
  masks.
- **The instance read as the engine reads it.** The checker reads the texts by a reader of its own
  that refuses what the engine's refuses, with the same error.
- **Exact over ℚ.** The checker reduces over the integers.
- **The cheap mode.** Only the rows stage is evaluated at points, and the rest stays exact. Over ℚ
  each point has a prime of its own, and a row whose degree bound reaches `LEAST_MODULUS`, 2^31,
  below every prime a point can have, is checked exactly. The result is the earliest flaw in row
  order over all the points.

### Memory

- **Free memory given back before the texts.** Once a call's result is complete, before its basis
  text, the call gives glibc's free memory back by `trim`, `malloc_trim(0)`, as a memory stop does
  (_Bounds_ above). A run that makes an
  image for the lift gives nothing back, and neither does a call that a bound stops before its
  texts.

### Limits

- **One thread.** Every run computes on one thread, whatever `threads` allows.
- **The budget.** A run ticks its budget once per row or step, and every loop whose work grows with
  the input spends units of work on it, per iteration or per chunk of at most `RUN`, 2^15,
  iterations, a unit bounding about a nanosecond of the loop's slowest case: an entry of a row
  operation, a byte of a text, a word read, and `PROBE`, 16, for a random access into an array past
  the caches, with weights of their own for products, sorts, cofactor terms, and reconstructions.
  Every 64 ticks or `WORK`, 2^23, units, and at the start of each round, the run checks its
  deadline, cancel flag, and memory ceiling in full, so a check comes within `WORK` units of any
  loop's work. `budget::Site` names every loop that spends and its units. The checker counts units
  of its own, about a term each, and ticks once per 4096 of them.
- **The memory ceiling.** The run reads the resident set from `/proc/self/statm` on Linux, by
  `proc_pidinfo` on macOS, and as the working set on Windows. Before a memory stop it returns free
  memory to the system, and where the check at the start of a round finds the resident set past
  the ceiling, it first shrinks its arrays and gives back its matrices' buffers. Where the resident
  set can no longer be read after the start, the run counts it as past the ceiling.
- **Address space.** A matrix whose columns pass
  1 MiB reserves 32 MiB of address space, of which the resident set holds
  only the pages the run's matrices have written. The memory ceiling and a cgroup's limit
  count the resident set. `ulimit -v` counts the address space, the whole
  reservation among it.
