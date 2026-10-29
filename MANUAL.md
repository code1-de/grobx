# grobx manual

grobx computes reduced Gröbner bases of polynomial ideals over prime fields
GF(p), p a prime below 2^64, and over the rationals ℚ. It is a Rust crate
with a Python extension module, a command-line program, `grobx`, and a C
interface, the library `grobxc` and its header `grobx.h`. All
four take the ideal's generators as text and return the basis as text.
The Maple package `Grobx` runs the command-line program (_Maple_ below).

grobx runs on Linux and Windows on x86-64 and on macOS on Apple Silicon. The memory ceiling
reads the resident set from `/proc/self/statm` on Linux, by `proc_pidinfo`
on macOS, and as the working set on Windows, so a call with a ceiling
returns `Error::Memory` where the system gives none.

This manual is for a caller of the library. _Interfaces_ says how to call
it, _Formats_ what goes in and what comes out, and _Computation_ how a call
behaves. _Design_ says which big decisions shape the engine and why.

## Interfaces

### Rust

The crate is `grobx`. Its entry points take the instance text and a
`Params`:

```rust
let outcome = grobx::groebner("x,y\n7\nx^2+y,\nx*y-1\n", &grobx::Params::default())?;
assert_eq!(outcome.basis.as_deref(), Some("[y^2+x,\nx*y+6,\nx^2+y]:\n"));
```

- `groebner(text, &params) -> Result<Outcome, Error>` computes the reduced
  Gröbner basis of the instance `text` (_Instance text_ below).
- `groebner_with(text, &params, &mut trace)` is `groebner`, and the
  `trace::Trace` it is given hears the source of every row of every matrix
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
  hand, `Unsupported` where the page size cannot be read, and the error of
  `madvise` where that fails. The Python module copies its long texts by
  it (_Python_ below).
- `VERSION` is the crate version.

`Params::new(order, threads, seed)` sets every bound absent, no checker, no
cofactors, signatures on, the echelon on, and no report. `Params::default()` is
`Params::new(MonomialOrder::Grevlex, Threads::All, random_seed())`, and
`random_seed()` draws a seed from the operating system's random source. Its fields:

| Field | Type | Meaning |
|---|---|---|
| `order` | `MonomialOrder` | The monomial order: `Grevlex`, `Lex`, or `Block(vec![k1, …, km])` (_Monomial orders_ below). |
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
| `report_sink` | `Option<ReportSink>` | Where `report` is on, a writer that takes each report line as the call meets it, flushed after the line, so a call that dies leaves its report so far. The first write error ends the writing; the outcome's report holds every line either way, save the `alloc` lines, which the sink alone takes (_Report text_). |

`signatures` chooses the engine. On, the run prunes pairs by the signatures
of its rows and hands the call to the plain F4 engine, started over, where
the signatures stop paying. Over GF(p), a call that asks for cofactors or the checker
never hands over. Over ℚ the learn runs and the verification's run of such a call
track no cofactors and may hand over.
Off, the call runs the plain F4 engine. A block order,
and a lex call that runs F4 in lex, run the plain engine whatever
`signatures` says.

`echelon` puts the generators in reduced row echelon form, as vectors
over their monomials, before a run computes (_Matrices and reduction_,
**The inputs**). A run that tracks cofactors takes them as read whatever
`echelon` says. Over ℚ, every run of a call that asks for cofactors or the
checker takes them as read. A replay takes them as its record's learn run
took them, and a replay that tracks cofactors of a record whose learn run
took the echelon returns `Error::Misfit`. Off, every run but a replay
takes the generators as read.

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
| `stats` | `Stats` | What the run did: `rounds`, the matrices its rounds reduced, the inter-reduction's not counted; `reduced`, the reductions it ran, rows or random combinations of rows where the plain run or the signature run combines or tests them; `zero`, those that reduced to zero. |
| `report` | `Option<String>` | The call report (_Report text_ below), where the call asked for it; `None` where it did not. |

`Stop::name()` is the bound's name, `"deadline"`, `"cancel"`, or
`"memory"`, as the call report, the Python module, and the CLI write it.

A rejected result is returned with `certified` false. It is not an error.
A call that returns an `Error` returns no report.

`Allocator` serves a request of at most `SMALL`, 1024 bytes, from mimalloc
and a larger one from the system's malloc (_Small blocks from mimalloc_
below), on Linux; on Windows and macOS the system's allocator serves both.
The Python module, the C library, and the CLI set it as their
global allocator. A Rust program sets it by

```rust
#[global_allocator]
static GLOBAL: grobx::Allocator = grobx::Allocator;
```

A program that sets no global allocator runs the crate under its own.

The first call of a process, of any entry point, sets two parameters of
glibc's malloc for the process by `mallopt`: `M_TOP_PAD` to `TOP_PAD`,
16 MiB, and `M_MMAP_THRESHOLD` to `MMAP_THRESHOLD`, 2 MiB (_The heap
padded at the first call_ below). The process keeps them after the call.
They replace a caller's own setting of the two from before the first call,
and a caller's `mallopt` after the first call stays. Where `mallopt`
refuses either, that call and every later one of the process return
`Error::Malloc`. Where the C library is not glibc, a call sets nothing.

### Python

The Python extension module `grobx` wraps `groebner`:

```python
import grobx

r = grobx.groebner("x,y\n7\nx^2+y,\nx*y-1\n")
r.basis    # '[y^2+x,\nx*y+6,\nx^2+y]:\n'
r.digest   # SHA-256 of r.basis, 64 lowercase hex digits
```

`grobx.groebner(text, *, order="grevlex", threads=0, seed=None, deadline=None, cancel=None,
memory=None, checker=None, cofactors=False, report=False, report_file=None)` takes the fields of `Params`
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
on. `report_file` turns `report` on, and the call creates the file at
its start and writes each report line there as it meets it, unbuffered,
so a call that raises or never returns leaves its report so far. A call
that raises leaves it ending with its `error` line. A file that cannot be
created raises `OSError`.

The call releases the GIL while it computes, so another Python thread can
raise `cancel`. `grobx.Cancel()` has the methods of `Cancel`. The first
call sets glibc's malloc parameters for the Python process (_Rust_ above).

The call returns a `grobx.Result`. Its read-only attributes are the fields
of `Outcome` but `stats`, with `stopped` as the string `"deadline"`,
`"cancel"`, or `"memory"`. Two more attributes: `digest` is SHA-256 of
`basis`, computed when it is read, and `seed` is the seed the call ran under.

The call makes `basis` and `cofactors` `str`s before it returns. A text of
at most `TEXT_CHUNK`, 256 KiB, is copied into its `str` at once. A longer
one is copied 256 KiB at a time by `text::hand_over`, which gives back the
core's text behind each copy, so that the call's peak holds the text
about once. The `str` is then
read back at its first, middle, and last character. A `str` that cannot be
made raises Python's error, `MemoryError` among them, a text that cannot
be handed over raises `OSError`, and a `str` that reads back otherwise
than its text raises `RuntimeError`. `report` is copied at once.

An instance or a parameter the call cannot take raises `grobx.Error`, a
subclass of `ValueError`. Its message opens with the name of the error and
a colon, `Syntax:`, `UnknownVariable:`, and so on (_Errors_ below).
`Error::Malloc` raises it too, with the message opening `Error:`. A
`checker` that is a `bool`, or neither `"exact"` nor an integer from 1 to
2^32 − 1, raises
`ValueError`, `TypeError`, or `OverflowError`. An integer keyword out of its
type's range, such as `seed=-1`, raises `OverflowError`.

`grobx.__version__` is the crate version.

### CLI

The command-line program `grobx`, the binary of the package `grobx-cli`,
calls `groebner` once, which sets glibc's malloc parameters (_Rust_
above):

```sh
grobx instance.ms > basis.txt
printf 'x,y\n7\nx^2+y,\nx*y-1\n' | grobx --checker exact --cofactors c.txt
```

`grobx [SWITCH]... [FILE]` reads the instance text from FILE, or from
standard input where FILE is absent or `-`, and writes the basis text to
standard output. At most one FILE is given. Field and prime are the
instance text's second line.

| Switch | Value | Default | Sets |
|---|---|---|---|
| `--order NAME` | `grevlex`, `lex`, or `block:k1,…,km` | `grevlex` | `order` |
| `--threads N` | a number of threads; 0 is all the machine has | 1 | `threads` |
| `--seed N` | 0 to 2^64 − 1 | drawn at random, and written to standard error | `seed` |
| `--deadline TIME` | milliseconds, or a number followed by `s`, `m`, or `h` for seconds, minutes, or hours, at most 2^64 − 1 milliseconds | none | `deadline` |
| `--memory BYTES` | bytes | none | `memory` |
| `--checker MODE` | `exact`, or a number of points from 1 to 2^32 − 1 | none | `checker` |
| `--cofactors FILE` | a path | none | `cofactors`, and the file the cofactor text goes to |
| `--report FILE` | a path | none | `report`, and the file the call report goes to |
| `--output FILE` | a path | standard output | the file the basis text goes to |
| `-h`, `--help` | | | prints `grobx`, the crate version, and the copyright of Code1 GmbH on one line, then the usage, the instance format with an example, the switches, what Ctrl-C does, and the exit statuses, and exits 0 |
| `--version` | | | prints `grobx` and the crate version and exits 0 |

The switches set the fields of `Params` (_Rust_ above) that the keywords of
`grobx.groebner` set, save `cancel`, which SIGINT raises (below), and
`signatures` and `echelon` stay on. A number is decimal digits
alone, with no sign and no suffix but the `s`, `m`, or `h` of `--deadline`.
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
far; a call that returns writes the file whole again at its end. A
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
| An output file cannot be created or written, or a refused call's report file cannot be removed. | `grobx: <path>: <reason>` |
| Standard output cannot be written. | `grobx: standard output: <reason>` |
| The watch on SIGINT cannot start. | `grobx: SIGINT: <reason>` |
| The call returns an `Error`. | `grobx: <input>: <message>`, `<input>` the path or `standard input` |

| Exit status | When |
|---|---|
| 0 | The basis was written, and the checker accepted it or was not asked; or the help or the version was printed. |
| 1 | A bound stopped the call: `--deadline`, `--memory`, or SIGINT. |
| 2 | An error: the watch on SIGINT, the arguments, the instance, an output file or standard output, or an instance or order the call refuses, `Error::Memory` and `Error::Malloc` among them. |
| 3 | The checker rejected the result. The basis was written, and the cofactors where `--cofactors` names a file. |

While `grobx` reads the instance, SIGINT ends the process by SIGINT.
After that, SIGINT raises the call's cancel flag, so the call stops at its next check
and `grobx` exits 1. A SIGINT that comes before the call, while `grobx`
opens its output files, stops the call at its first check. A second
SIGINT ends the process by SIGINT, with no exit status. Where SIGINT is
ignored when `grobx` starts, as a shell has it for a command run in the
background, `grobx` leaves it ignored.
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
int status = grobx_groebner("x,y\n7\nx^2+y,\nx*y-1\n", &p, &r);
/* status GROBX_OK, r.basis "[y^2+x,\nx*y+6,\nx^2+y]:\n",
   r.certified GROBX_CERTIFIED */
grobx_result_free(&r);
```

`grobx_groebner(text, &params, &result)` computes what `grobx.groebner`
computes for the same arguments (_Python_ above). `text` is the instance
text, NUL-terminated. The first call sets glibc's malloc parameters for
the caller's process (_Rust_ above).

`grobx_params_new(order, threads, seed)` returns a `grobx_params` with the
order, threads, and seed given, no blocks, every bound absent, no checker, no
cofactors, and no report. `grobx_params_default()` is
`grobx_params_new(GROBX_GREVLEX, 0, s)` with s drawn as `random_seed()`
draws it, the C form of `Params::default()`; `seed` holds it for a replay.
The caller sets the other fields by name. The fields are those
of `Params` (_Rust_ above). Where a field's type differs:

| Field | Type |
|---|---|
| `order` | `uint32_t`: `GROBX_GREVLEX`, `GROBX_LEX`, or `GROBX_BLOCK` |
| `blocks`, `nblocks` | `const size_t *` and `size_t`: under `GROBX_BLOCK`, the sizes k1, …, km of `Block` (_Monomial orders_ below); `{2, 3}` is `Block(vec![2, 3])` |
| `threads` | `size_t`; 0 is all the machine has |
| `deadline` | `uint64_t` in milliseconds; `GROBX_NONE`, `UINT64_MAX`, is none |
| `cancel` | `grobx_cancel *`; NULL is none |
| `memory` | `uint64_t` in bytes; `GROBX_NONE` is none |
| `checker` | `uint32_t`: `GROBX_CHECKER_OFF`, `GROBX_CHECKER_EXACT`, or `GROBX_CHECKER_CHEAP` |
| `points` | `uint32_t`: the cheap mode's points, from 1 to 2^32 − 1; read in the cheap mode alone |

`signatures`, `echelon`, and `report_sink` are no fields: a C call runs
with both on, and with no sink, as a Python call does.

The call writes a `grobx_result`. Its fields are those of `Outcome` but
`stats`, and one more, `error`:

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
| `GROBX_ERROR` | 9 | Any other `Error`, `Error::Malloc` among them |
| `GROBX_ARGUMENT` | 10 | An argument the interface refuses before the call: `text`, `params`, or `result` NULL; `text` not UTF-8; `order` none of the three orders; under `GROBX_BLOCK`, `blocks` NULL while `nblocks` is not 0; `checker` none of the three modes; or the cheap mode with `points` 0. |
| `GROBX_PANIC` | 11 | The call panicked, a defect of grobx. The process's panic hook hears of the panic too, and by default prints it on standard error. |
| `GROBX_SIZE` | 12 | `Error::Size` |
| `GROBX_DROPS` | 13 | `Error::Drops` |

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
Basis([x^2 + y, x*y - 1], tdeg(x, y), characteristic = 7);
# [y^2 + x, x*y + 6, x^2 + y]
```

`maple/README.md` describes the package: where it finds the program, its
orders and options, and its errors.

### Errors

The C interface returns each `Error` as a status code (_C_ above).

| `Error` | Raised where |
|---|---|
| `Syntax { line, expected }` | The instance text breaks the grammar at `line`, counted from 1. |
| `UnknownVariable { line, name }` | A generator names a variable the first line does not. |
| `RepeatedVariable { name }` | The first line names a variable twice. |
| `Characteristic { value }` | The characteristic is neither 0 nor a prime below 2^64, or is 0 in `learn` or `replay`. |
| `Overflow { line }` | A degree passes the widest monomial format, in the instance at `line` or during the run. |
| `Size` | The instance or the run holds more generators, terms, numeric factors, monomials, rows, elements, or record steps than a u32 id numbers, 2^32 − 1. |
| `NoInverse { line }` | A denominator at `line` is 0 mod p over GF(p), or is 0 over ℚ. |
| `Order { name }` | `name` names no order (`str::parse`), or it is a block order whose sizes do not fit the variables. |
| `Memory` | A memory ceiling is asked for where the resident set cannot be read. |
| `Misfit { step }` | A replay's record does not fit at `step`; step 0 is the inputs, a record learned in another order, a record learned with the echelon where the replay tracks cofactors, or the change of order of a record FGLM changed. A changed record whose replayed grevlex basis FGLM cannot change does not fit at the step one past the record's last. |
| `Drops { misfits, leads, verification }` | The lift over ℚ dropped `DROPS`, 4, records in the call: `misfits` after `MISFITS`, 2, misfits in a row, `leads` at a verification whose leading monomials differ, and `verification` at one at which an element differs. Random primes make it improbable, and a defect of grobx makes it certain. |
| `Malloc { param }` | `mallopt` refused the value the first call of the process sets for `param`, `"M_TOP_PAD"` or `"M_MMAP_THRESHOLD"`; every later call of the process returns it too. glibc takes both values, so a malloc that is not glibc's makes it. |

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
factor is a decimal integer, a fraction `a/b` of two decimal integers, or a
variable with an optional exponent `^e`. A term may hold several numbers
and repeat a variable, and like terms are combined: `2*x*3*x^2` is `6*x^3`.
A term's degree is at most 2^31 − 1, its coefficient zero or not; a term
past it is `Overflow` once the term is read.
A denominator stands only after a number: `1/3*y` and `y*1/3` are read,
`y/3` is a syntax error. Over GF(p) every number is read mod p.
A denominator 0, its digits all `0`, is `NoInverse` at the line of those
digits, over ℚ as over GF(p).

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
  `error` where the call returns an error. An error before the report
  began, of glibc's parameters or of the memory ceiling, is a report of
  its `error` line alone.

The call of _Rust_ above with seed 1, one thread, and the report on,
through Python:

```
0.000030 call entry=groebner order=grevlex seed=1 threads=1 signatures=on echelon=on cofactors=off checker=off deadline=none memory=none cancel=off trace=off cpu=Intel(R)_Core(TM)_i7-14700K
0.000034 read variables=2 characteristic=7 field=gf31
0.000048 run number=1 role=call engine=signature engine-by=signatures field=gf31 primes=7 width=8 order=grevlex cofactors=off reduce=tests handover=on why=start
0.000063 inputs generators=2 nonzero=2 terms=4 read=0.000013
0.000075 echelon generators=2 rows=2 columns=4 subtracted=0 time=0.000011
0.000109 round number=1 degree=3 pairs=1 rows=2 given=1 columns=3 entries=4 pivots=1 zero=0 zero-rows=0 reductions=1 subtracted=2 added=0 bands=no pending=1 dropped=0 select=0.000003 symbolic=0.000003 encode=0.000000 reduce=0.000006 backsub=0.000000 record=0.000000 decode=0.000002 update=0.000001
0.000111 handover round=1 outgrown=stay made=1 zero=0 divisible=0 widest=1 verdict=stay
0.000121 inter rows=3 given=3 columns=6 entries=6 subtracted=0 elements=3 symbolic=0.000003 encode=0.000000 reduce=0.000001 decode=0.000002
0.000123 ran number=1 outcome=basis rounds=1 reduced=1 zero=0 elements=3 time=0.000074 considered=3 equal=0 high=0 low=0 syzygy=2 koszul=0 queued=1 past-format=0
0.000130 text kind=basis bytes=23 time=0.000004
0.000132 end result=basis bytes=23 time=0.000132 process-rss-start=3080192 process-rss-peak=15695872 process-rss-grown=12615680
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
  `cpu` is the processor's CPUID brand string, its words joined by `_`,
  and `none` off x86-64.
- `read variables=<n> characteristic=<p> field=<gf31|gf64|q>`: the first
  two lines of the instance text as read. `gf31` computes over a prime
  below 2^31, `gf64` over a larger prime, and `q` over ℚ, through the
  lift (_Over ℚ_ below).
- `stop bound=<deadline|cancel|memory>`: a bound stopped the call, the
  outcome's `stopped`.
- `end result=<basis|stopped> bytes=<b> time=<t> process-rss-start=<b>
  process-rss-peak=<b> process-rss-grown=<b>`: the call's end, after
  `stop` where there is one. `bytes` is the basis text's length, where
  there is a basis, and `time` the whole call's.
  `process-rss-start` is the process's resident set at the call's start,
  `process-rss-peak` its peak resident set at the end, and
  `process-rss-grown` the peak less the start, all in bytes as mimalloc's `mi_process_info` reads them
  on Linux, and on Windows and macOS as the system reads them, 0 where it cannot.
  The peak is the process's, so a call after a larger one in the same
  process reads that one's.
- `error error=<variant> message=<text>`: the call returns the error
  `Error::<variant>`, whose message is `message`, its words joined by `_`
  and each `=` an `_`.
- `alloc kind=<alloc|zeroed|realloc> bytes=<b> from=<b>
  result=<served|refused> stack=<frame>,…`: a request for a block of 1
  GiB or more (`LARGE`), or one the allocator refused, which then aborts
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
  engine-by=<signatures|signatures-off|order|handover|record>
  field=<gf31|gf64|gf31pair|gf64pair> primes=<p[,q]> width=<8|16|32>
  order=<grevlex|lex|block> cofactors=<on|off>
  reduce=<rows|combinations|tests>
  reduce-by=<record|trace|learner|cofactors|lanes|prime>
  handover=<on|off> shape=<use|take|off>
  why=<start|wider|handover|refit|lex>`.
  - `number` counts the call's runs from 1, the lift's among them.
  - `role`: `replay` where the run replays a record, `learn` where it
    records one, `verify` where it is the lift's run at the
    verification's prime, and `call` for any other.
  - `engine` and `engine-by`, why that engine: `signatures`, the
    call's `signatures` on and a graded order; `signatures-off`;
    `order`, a block order or F4 in lex; `handover`, the plain run after
    the signature run handed over; `record`, a replay.
  - `field` and `primes`: the field the run computes in and its primes,
    two where a pair of primes shares the run. `width` is the bits of
    each exponent of the monomial format. `order` is the order the run
    computes in: `grevlex` for a lex call's grevlex run, whose basis
    FGLM is to change.
  - `reduce`: `combinations` where the plain run reduces random
    combinations of rows, `tests` where the signature run or a learn's
    plain run rules rows zero by tests, else `rows`, each row reduced
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
    `refit`, a replay from a kept shape met other terms; `lex`, a lex
    call whose ideal FGLM does not take, which runs F4 in lex.
- `inputs generators=<m> nonzero=<k> terms=<t> read=<time>`: the
  generators read into the run's format, the terms counted over all,
  and the time of the read. The call's first read parses the text, and
  every later one maps the parsed generators (_Over ℚ_, **The
  generators parsed once**, below).
- `echelon generators=<m> rows=<r> columns=<c> subtracted=<s>
  time=<time>`: after `inputs`, where the run takes the generators in
  reduced row echelon form (`Params::echelon`, _Rust_ above): the
  generators read, zero ones among them; the nonzero rows, their rank;
  the matrix's columns, the generators' distinct monomials; the entries
  its row operations subtracted, as a round's `subtracted` counts them;
  and its time.
- `round number=<k> degree=<d> pairs=<p> rows=<r> given=<g> columns=<c>
  entries=<e> pivots=<v> zero=<z> zero-rows=<y> reductions=<x>
  subtracted=<s> added=<a> bands=<yes|no> pending=<q> dropped=<o>
  left=<l> select=<time> symbolic=<time> encode=<time> reduce=<time>
  backsub=<time> record=<time> decode=<time> update=<time>`: a round of
  the run, one line per round `stats.rounds` counts.
  - `degree`: the sugar of the plain run's pairs, the degree of the
    signature run's signatures, or the degree of a replay's first
    column. `pairs`: the plain run's pairs, the signature run's
    reductants, or a replay's rows the record counts.
  - `rows`, `given`, `columns`, and `entries`: the matrix's rows, of
    which `given` were given, ahead of the reducers symbolic preprocessing
    added, its columns, and its nonzero entries.
  - `pivots`: the rows the reduction made; `zero`: the reductions to
    zero; `reductions`: the reductions run. `zero` and `reductions` add
    up to `stats.zero` and `stats.reduced`.
    `zero-rows`: the rows known to reduce to zero, left out where the
    plain run combined rows.
  - `subtracted` and `added`: the entries the round's row operations
    subtracted, and those they added into combinations, as the budget
    spends them: a pivot row's length, its leading entry among them, per
    subtraction, and a row's length per band row of a band it joins, the
    lanes a block leaves at coefficient 0 among them.
  - `bands`: whether the round's matrix takes bands of rows: the field
    has them, no cofactors are tracked, the rows hold 64 entries or more
    in the mean, the run is no replay, and in the plain run it combines
    rows. A row that does
    not stand with another goes alone even so.
  - `pending` and `dropped` stand for the signature run: the signatures
    taken since its last round, and those that gave no reductant.
    `left` stands for the plain run: the pairs left after the round.
  - The eight times split the round (_The call report_ below):
    `select`, the pairs and their rows, or the reductants; `symbolic`,
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
  outgrown=<stay|hand-over> pair-work=<stay|hand-over> made=<m> zero=<z>
  divisible=<d> pairs=<n> widest=<w>
  verdict=<stay|hand-over>`: after every round that made an element in a
  signature run that may hand over, which the outgrown test reads, each
  test that read it with its verdict,
  the early test at every early round, `stay` at one of fewer than its
  least: `EARLY_ROUND`, 25, elements where the plain run after a handover
  would combine rows, and `EARLY_ROUND_ALONE`, 100, where not; the elements
  the round made, its rows reduced to zero, the count a test read where
  one did, the S-pairs the round's update would form where the pair-work
  test read them, the round's widest gap, and whether the run hands over.
- `constant round=<k>`: the run made a constant, and the basis is
  `[1]:`. Round 0 is an input that is constant.
- `inter rows=<r> given=<g> columns=<c> entries=<e> subtracted=<s>
  elements=<n> symbolic=<time> encode=<time> reduce=<time>
  decode=<time>`: the matrix that inter-reduces the basis, the elements
  it made, and the times of its stages, `decode` with the elements
  sorted.
- `ran number=<k>
  outcome=<basis|handover|overflow|size|denominator|refit|misfit|stop>
  at=<read|echelon|run> step=<s> rounds=<k> reduced=<x> zero=<z> elements=<n>
  time=<time> considered=<n> equal=<n> high=<n> low=<n> syzygy=<n>
  koszul=<n> queued=<n> past-format=<n>`: the run's end. `overflow`:
  the read or the run passed the monomial format. `denominator`: the
  read met a denominator that is 0 mod the run's prime. `at` stands for
  both, `read` or `run`, and as `echelon` where the generators' reduced
  row echelon form ended the run: `misfit`, `stop`, or `size`.
  `step` stands for `misfit`: the replay's step that did not fit,
  or 0 where the lanes of a pair's run disagree. `size`: a count of the
  run passed what a u32 id numbers (`Error::Size`).
  `rounds`, `reduced`, and `zero` are the run's `stats`, and `elements`
  the reduced basis's, where the run made one. `time` runs from the
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
  time=<time>`: the checker ruled. Where a bound stops the check, `stop`
  follows instead.

**The lift over ℚ.** After each prime's run the lift writes what became
of the prime, in the order it used them. A pair's second prime whose
image the lift never takes, where the first ends the lift or drops the
record, stands on no line of its own: on the pair's `run` line, and on
its `pair` line where the pair fell back:

- `lift learns=<pairs|alone> replays=<pairs|alone> track=<on|off>`:
  once, first. `replays` is `pairs` where two primes share each replay
  of a record FGLM did not change: the call tracks no cofactors. A
  record FGLM changed replays one prime at a time even so.
  `track` is whether the replays track cofactors.
- `learn primes=<p[,q]> outcome=<recorded|unpaired|skipped>
  why=<disagree|denominator>`: a learn run's record. `unpaired`: a learn
  at two primes fell back to the first alone, for `why`. `skipped`: the
  prime divides a denominator of the text.
- `pair primes=<p,q> outcome=alone why=<misfit|denominator|range>`: a
  replay at two primes runs each alone, `range` where a prime lies
  outside [2^61, 2^62).
- `skip prime=<p> why=<denominator|lifted-denominator>`: a prime that
  divides a denominator of the text, or the verification's prime that
  divides one of the lifted basis.
- `misfit prime=<p> why=<record|leads>`: the replay did not fit the
  record, or its leading monomials differ from the learn run's.
- `drop why=<misfits|leads|verification>`: the record and the lifting
  dropped, after `MISFITS`, 2, misfits in a row, or at a verification whose
  leading monomials differ or at which an element differs. The next
  prime learns afresh, save after the drop that reaches `DROPS`, where
  the call returns `Drops` and no report.
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
- `Lex`, named `"lex"`, is the lexicographic order. Where the ideal is zero-dimensional
  and its degree is at most `MAX_STAIRCASE`, 65535, the call computes the
  grevlex basis and changes it to lex by FGLM. Otherwise it runs F4 in lex.
- `Block(vec![k1, …, km])`, named `"block:k1,…,km"`, splits the variables,
  in the instance's order, into
  blocks of k1, …, km variables and a last block of the rest. Each block is
  ordered by grevlex, and the blocks are compared in turn. The sizes are
  decimal digits, separated by commas, with no sign and no space. Each is at
  least 1, and they sum below the number of variables. `"block:k"`
  eliminates the first k variables, as Groebner.jl's `block:k` and msolve's
  `-e k` do.

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
An instance with a denominator 0 returns `NoInverse` at once, at the
line of the first, unless an error earlier in the text, a refused order or
memory ceiling, or a bound comes first.
Where the lift drops `DROPS`, 4, records, each after `MISFITS`, 2, replays that
misfit or at a verification that fails, the call returns `Drops`.

### Bounds

A deadline, a cancel flag, and a memory ceiling each stop the call. The
result then carries no basis and names the bound in `stopped`. The call
checks its bounds lazily, so it stops at its next check, not at once. The
memory ceiling is read from the resident set at those checks, so a call may
pass it briefly. Before the ceiling stops a call, the call gives back the
memory glibc's malloc and mimalloc hold free, and at the start of a round or of the
inter-reduction, or over ℚ before a prime, the memory it can rebuild. A
hard ceiling is the operating system's: `ulimit` or
a cgroup.

A call checks its bounds at the start of each run, once the run has read the
instance, at the start of each round, after
every 64 of its rows or other steps, and once a round's matrix has done
`WORK`, 2^23, units of work since the last check, a unit an entry of a
row operation (_Limits_, _Units of work_), or once its texts have done as
many, a unit a byte (_Limits_, _Texts' units of work_), or once its work on
the cofactors and on the lift's images over ℚ has done as many
(_Limits_, _Cofactors' units of work_). The checker's steps include one
per 4096 of its own units of work, a unit about the work of one term
(_Limits_, _The checker's units of work_). Through the Python module on
an i7-14700K, 357 calls with a deadline at 0.1 to 0.9 of their time
stopped a median of 0.0018 s past it and at most 0.052 s past. On the
tuning set of iterations 0067 on, 259 calls with a deadline at 0.3 to
0.9 of their time stopped a median of 0.0014 s past it and at most
0.062 s past, satlib.uuf50_8-p65521 at a deadline of 64 ms. A
deadline inside a basis text over ℚ of 138 MB stopped its call within
0.010 s past. On 220 calls with cofactors or in lex through the CLI, the
work on cofactors went at most 0.066 s between two checks, and a call
stopped outside its checker returned within 0.048 s of the check that
stopped it. No check follows a call's last text but the checker's, where the
call asks for it, and the Python module copies
each text into a Python string after the call, a long one in chunks
(_Python_ above): 0.027 s for a text of
220.6 MB.

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
cofactor text once per batch. Its memory does not grow with k, and the
result is that of all k points at once. A call at any k from 1 to
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

This part names the decisions that shape what a call costs and what it
returns. Each names its decision ids. The comment at the site in the source
carries the id, its provenance, and the source it cites; `grep -rn
'<id> '` finds it. The plan of the item the id is named after holds the
full reasons, the alternatives rejected, and what was measured.

### A call, end to end

1. The reader reads the instance into the narrowest monomial format that
   holds it, and the entry point picks the field, the format, and the order
   once. Each combination is its own compiled copy of the engine. The
   call parses the generators once, and each run maps them to its field
   and format. A run
   that tracks no cofactors puts the generators in reduced row echelon
   form before it computes, where `Params::echelon` asks for it, as it
   does by default. A replay does as its record says, and over ℚ no run
   does where the call asks for cofactors or the checker.
2. Over GF(p) in grevlex, the signature run computes the basis. Where it
   stops paying, it hands the call to the plain F4 run, which starts over.
3. Both runs build one matrix per round and reduce it. A last matrix
   inter-reduces the basis, and the writer writes the canonical text as
   bytes, into a buffer reserved once (_The result_ below).
4. In lex, the call computes the grevlex basis and changes it to lex by
   FGLM, or runs F4 in lex where FGLM does not apply.
5. Over ℚ, the call learns at two primes at once, replays at more primes,
   lifts the images to ℚ, and stops once one more prime agrees, as a rule
   the learn run's second.

### Monomials

- **One table per run** (f4-gfp.14, .15, .16, .19, .44;
  threshold-recheck.14, .15). A monomial is a
  `MonoId`, a u32 id, into a hash table that lives for the whole run and frees
  nothing. A run that starts over builds a new table, and over ℚ a replay
  may read into the table that a full replay before it built.
  A run that tracks cofactors keeps their monomials in a second table of
  the same format (f4-gfp.24).
  The hash is additive, Σ r_i·e_i with r_i drawn from the seed, so
  a product's hash is the sum of its factors' hashes. The table probes
  linearly and takes the slot from the hash's top bits. Its slot array
  starts at `SLOTS`, 2^10, slots and doubles once it holds fewer than
  `LOAD`, 2, slots per monomial. At the machine's own clock, outside the measuring condition,
  2^9 and 2^11 slots at the start each took +0.002 in the mean ln of the
  time over the 101 instances of the tuning set but cyclic-9-p65521, and a
  load of 4 and of 6 in 10 −0.006 and +0.010, the peaks +0.014 and −0.012,
  each within the margin of the improvement test. Monagan and Pearce
  found that freeing monomials did not pay and that additive hashing made
  multiplication faster (monagan-2015 §2.1).
- **Divisor masks** (f4-gfp.17, .18; many-variables-time.1–.4). Every
  monomial carries a 32-bit mask
  with fixed thresholds, and every divisibility test between monomials of
  the table reads the masks before the exponents (roune-2012 §4.2,
  demin-2024 §4.1). Up to 32 variables each variable takes ⌊32/n⌋ bits,
  bit k set where its exponent passes k. Past 32 variables variable i sets
  bit i mod 32 where its exponent is not 0, so every variable reaches the
  mask and a bit stands for several, as demin-2024 §4.1 compresses the
  mask. The alternative rejected reads the first 32 variables alone, as
  msolve (berthomieu-2021 §4.1 (2)) and roune-2012 §C.3.2 do, and passes
  every test between monomials that differ past them alone. A bit for a
  run of consecutive variables is rejected too: variable i at bit i mod 32
  lets a word's lanes land on consecutive bits, the mask still formed a
  word at a time. The mask stays 32 bits wide, where roune-2012 §C.3.2
  found 64-bit masks slower. Two monomials whose masks share no bit share
  no variable. Masks that share a bit mean a shared variable up to 32
  variables, and past 32 the exponents decide. The mask of an lcm is the
  OR of its factors' masks (_Pairs_ below). In the measuring
  condition, against the first 32 variables alone, the tuning set took
  0.968 of the time in the geomean over its 102 instances. The 17 on more
  than 32 variables, 35 to 60, took 0.841 of the time, the other 85 0.995.
  Two of the 17, toric-veronese5d3 and toricsat-veronese5d3 over ℚ, and
  toricsat-veronese3d6 among the other 85 fell at the same instructions:
  the build placed the reduction's scan of the dense row within 64 bytes,
  where the build measured against placed it across a 64-byte boundary.
  The other 15 of the 17 took 0.834 of the time.
  Over GF(65521) biomdcl-220 took 0.147 s against 0.231 s, at half the
  instructions, sat-pret60-25 0.069 s against 0.100 s, and
  sudoku-n9u56s5 0.700 s against 0.995 s.
- **Masks from whole words** (divisibility-time.5, .6). A mask of one or
  two bits a variable, on 11 variables or more, is formed a word at a
  time: bit 0 of each variable from the top bits that mark the lanes that
  are not 0, bit 1 from those that mark the lanes of 2 or more, gathered
  into the mask's bits by one multiplication, by two shifts in `Dense32`.
  A mask of three bits a variable or more is formed lane by lane from each word. The alternative
  rejected reads each variable through `Format::exponent`, which finds its
  word and lane again. The S-pairs' signatures take a mask each, millions
  in a call on chandra. On chandra-11 over GF(65521), seed 1, a mask of
  two bits a variable took 131 instructions for each of 1.96 million
  signatures, 7.4 % of the call's, and chandra-13 took 6.33 s against
  7.53 s with divisibility-time.1 to .7 together (_Pairs_ below).
- **Dense formats with a restart** (monomial-formats.1, .3–.9;
  monomial-time.4). A monomial is its total degree and one lane per
  variable of 8, 16, or 32 bits: `Dense8`, `Dense16`, and `Dense32`, of
  degree at most 127, 32767, and 2^31 − 1. A call starts in the narrowest
  format that reads its input. A product past the format's degree bound
  starts the call over from its input in the next wider format, under the
  same seed and the budget left. A restart doubles the work done before
  it: chandra-12 in lex took 17.6 s against 8.3 s.
- **Packed words** (monomial-time.1–.12). The lanes stand in 64-bit
  words, the low lanes first, and the lanes past the last variable hold
  0. No exponent passes the degree bound, so every lane keeps its top bit
  clear. A product is the word sum, a product check compares the words,
  a divisibility test subtracts the words and reads the top bits, and
  grevlex on equal degrees compares the words from the last (bachmann-1998
  §3, Lemma 1). Lex compares the first differing word at its lowest
  differing lane. lcm and the S-pair's signature take each lane's maximum
  or minimum from the top bits, their degree is the sum of the lanes, and
  a block order reads lane by lane. The low ratio base divisor's bound x^v
  is packed words of the format, an infinite exponent held as the degree
  bound, and its test is the divisibility test. One word per variable,
  operated on one exponent at a time, is the alternative rejected. A
  divisibility test on exponents of the full 8 or 16 bits, which would
  keep the bounds 255 and 65535, took 1.025 to 1.085 times the guard
  bit's time on chandra-12, noon-9, and biomd-333. Against one word per
  variable, the tuning set took 0.892 of the time in the geomean over its
  116 instances, 0.849 over GF(65521) and 0.949 over ℚ. Over GF(65521),
  biomd-147 took 6.34 s against 11.45 s and sparse-16 0.331 s against
  0.579 s, and the calls executed 0.874 of the instructions in the
  geomean.
- **One compiled copy per field, format, and order** (order-family.4,
  monomial-formats.12, gfp-large-primes.3, f4-gfp.25). The entry point
  matches each once. Cofactor tracking and the trace are type parameters
  whose empty versions compile to nothing. In the checker, a match at each
  comparison cost 4–5 %. The price is size: the extension module grew from
  1.98 to 3.43 MB with the three formats, and from 3.66 to 5.17 MB with
  `Gf64`.

### Pairs

- **Gebauer–Möller at insertion** (f4-gfp.1, .2; order-family.11). The
  criteria B_t, M, F, and the product criterion run as each element enters
  the basis, in the variant that ignores sugar (gebauer-1988 §4,
  giovini-1991 Appendix).
- **The new pairs tested by degree** (divisibility-time.1–.4). A new pair
  whose leading monomials are coprime gets no lcm, and no other new pair
  is tested against it: its lcm divides another new pair's only where one
  leading monomial of G_min divides another, and none does. The product
  criterion drops it. The other new pairs' lcms are formed in the
  update's own words, with their degrees and masks, and M and F run over
  them by increasing degree, each pair against the pairs kept so far. A
  pair goes where a kept pair's lcm divides its own. Of each lcm the last
  pair in G_min's order stays, as joux-2011 §2.2's Algorithm 2 keeps it,
  and only the kept pairs' lcms enter the table. The alternative rejected
  is Algorithm 2 as written: every new pair's lcm interned and tested
  against every other new pair, in time quadratic in their number. The
  update's buffers stay with the run, and a memory check that finds the
  resident set past the ceiling gives them back. Over GF(65521)
  biomd-333 took 0.351 s against 0.778 s and peaked at 26.6 MiB against
  99.4 MiB, and over ℚ 0.902 s against 1.754 s at 30.3 MiB against
  108.3 MiB. With divisibility-time.5 to .7, against Algorithm 2 as
  written, the masks per variable, and the scan inlined, the tuning set
  took 0.963 of the time in the geomean over its 116 instances, 0.953
  over GF(65521) and 0.972 over ℚ, and executed 0.940 of the
  instructions over GF(65521). biomd took 0.81 of the time over its 15
  instances and chandra 0.95 over its 11.
- **The pair set's masks beside its pairs** (plain-update-time.1–.6). The
  plain run's pair set holds its pairs and, in an array beside them, the
  complement of each pair's lcm mask. The B criterion tests the new
  leading monomial's mask against eight of them at a time, by the block
  scan of the divisor search (_The divisor search in the order of terms_ below),
  and reads a pair's exponents only where its mask passes. The
  complements turn the B criterion's test, the lead's mask within the
  lcm's, into the scan's. A pair the B criterion drops keeps its place,
  its complement set to all ones, which no nonconstant monomial's mask
  passes, and leaves the set at the round's selection. A new pair's lcm
  mask is the OR of its two leading monomials' masks. The new pairs whose
  leading monomials are not coprime are ordered by their lcm's degree by
  a counting sort, stable in G_min's order, where the degrees span at
  most as many values as there are such pairs, and by a comparison sort
  past that. Every pair, its place in
  the set, every selection, and every matrix stay as they were. The
  alternatives rejected: each pair's mask read from the table by its
  lcm's id, or kept in the pair; the masks as they are, with a block
  scan of their own; a dropped pair removed by `Vec::retain` at each
  update, which moves every pair after it, or swapped with the last,
  which reorders the set and with it the multipliers' ids and the matrix;
  the lcm's mask formed from its exponents; and a comparison sort of the
  new pairs at every update. In the measuring condition, against the B
  criterion by `Vec::retain` over the table's masks and the comparison
  sort, the tuning set took 0.969 of the time in the geomean over its 102
  instances and executed 0.965 of the instructions. Over GF(65521)
  satlib.uuf50_8 took 0.279 s against 0.482 s, brent-223-r11-fix9s1
  0.081 s against 0.114 s, and toric-veronese3d7 0.172 s against
  0.219 s; over ℚ satlib.uuf50_8 took 0.317 s against 0.520 s.
- **Selection by degree in grevlex, by sugar elsewhere** (f4-gfp.3;
  order-family.7–.10, .13). In a graded order a round takes every pair
  whose lcm has the least total degree. In lex and block orders a round
  takes every pair of least sugar. A new element's sugar is the larger of
  the round's sugar and its own largest total degree, which no source
  states (order-family.9). The input is never homogenized. Faugère found
  sugar less efficient for F4 in a degree order (faugere-1999 §2.5), and
  Giovini et al. found the normal strategy poor in lex (giovini-1991 §1).

### Matrices and reduction

- **The inputs** (input-redundancy-time.1–.6, .8, .9). Before a run
  computes, `echelon::echelon` puts its generators in reduced row echelon
  form: the matrix of their coefficients, a column per monomial in
  decreasing order and a row per generator. The forward pass goes over
  the columns from the first. Of the rows that lead at a column, the
  sparsest, ties to the least position, is its pivot (bouillaguet-2016
  §2.3), made monic, and every other row there is reduced by it, a merge
  of the two sparse rows. The back pass goes over the pivot columns from
  the last, and reduces each pivot row that holds another pivot column by
  the final pivot rows in a dense row of the field (heisinger-2025 §6.1,
  martani-2012 §1.2). A row stands at the position of the generator it
  came from, and a generator that depends on the rows that won columns
  before it becomes zero. The rows generate the same ideal, so the basis stays. buchberger-1985
  §6.4 reduces the polynomials modulo each other at the start of the
  algorithm by normal forms; this reduction is linear, one elimination
  over the generators' own monomials. ccode-newton40t6-p65521's four
  linear generators all lead at s1, and the signature run, which reduces
  a row by rows of smaller signature alone, kept s2 to s4 in their tails:
  its round 2 built 4,496,388 columns, every monomial of degree 35 or
  less, in 1.60 s of a 1.80 s call; from the generators in echelon form
  its round 1 of degree 35 built 412,062 rows (log 0068-1). The row of
  least position as the pivot took
  minrank-sm10r3k6-p0's 2,100 generators over 2,506 monomials 2.2 s, the
  sparsest 0.074 s, and one pass that subtracts each new pivot row from
  the pivot rows before it 0.49 s. Merges in place of the dense row took
  critmin-n9f2x2g2-p0's 122 echelons, one per pair of primes, 0.72 s of
  the call against 0.61 s. Written outside the call over the tuning set,
  rows by decreasing leading monomial in place of the generators'
  positions took −0.061 in the mean ln against −0.069, and by increasing
  +0.030. A run that tracks cofactors takes the generators as read, and
  over ℚ so does every run of a call that asks for cofactors or the
  checker (.8): each row's cofactors would be a combination of the
  generators' (`Params::echelon`, _Rust_ above). The echelon has a line
  of the call report and a stage of its own (.9). In the measuring
  condition, with the early round of _Signatures_ reading a round without
  a zero (input-redundancy-time.7), the call took −0.062 in the mean ln
  over the 102 instances of the tuning set against a noise threshold of
  0.020, and the peaks −0.060 in the mild form (log 0068-2).
  ccode-newton40t6-p65521 took 0.221 s against 1.882 s, at a peak of
  56.0 MiB against 598.7 MiB; biomdcl-123-p0 −0.82, rabl-biomd-315
  −0.78, poseidon-t2F2P3 −0.77, and rabl-biomd-123 −0.74. Over GF(65521)
  the 58 instances took −0.095, over ℚ the 44 −0.019. The echelon changes
  a run's path: distgeo-loop6r-s2-p0 took +0.73, its plain run's record
  replaying 17 rounds in place of 13, and sat-pret60-25 +0.37, its
  signature run handing over after round 2 in place of round 1.
  A replay takes its pivots from its record (_A replay's echelon from
  its record_ below).
- **A replay's echelon from its record** (run-generators-time.5–.8,
  .10–.12). A replay whose record's learn run took the echelon follows
  the record, as a trace's later calls build their matrices from it
  (berthomieu-2021 §4.3): each generator the record names nonzero is the
  pivot row of its recorded leading monomial's column, which interning
  the monomial finds (.8), and every other generator must reduce to zero
  (.5). The pivot rows go by increasing column. Each is reduced once in a
  dense row of the field by the pivot rows of the columns before its
  own, by `Field::sub_mul`, and made monic (.6; martani-2012 §1.2,
  monagan-2015 §2.2). A pivot row that leads at its column already is
  made monic in place (.11; monagan-2015 §2.2, heisinger-2025 §6.1). The
  dense row's columns come in increasing order from a bit set read a
  word at a time (.10), and a pivot row whose columns span fewer words
  than they number keeps its words, which each subtraction of it ORs
  into the set (.12). Each generator the record names zero is reduced in
  full, and an entry left is a misfit at the inputs, with no random
  combination and no probability of error (.7). The back pass is the
  echelon's. Where the echelon at the replay's prime gives the recorded
  leading monomials, the replay gives its rows and subtracts as many
  entries. Outside the measuring condition, the replay's forward pass
  took 12.8 ms on minrank-sm10r3k6-p0 against the learn's 65.9 ms, and
  0.92 ms on critmin-n9f2x2g2-p0 against 3.09 ms. Reading the dense row
  entry by entry, as heisinger-2025 §6.1 and monagan-2015 §2.2 read
  their buffers, took 2.53 ms on sudoku-n9u55s6-p0 against 0.51 ms by
  the bit set, and a bit set per column at every subtraction 1.44 ms on
  critmin (log 0087-3). The merges of the learn's forward pass and of
  every call over GF(p) stay.
- **Rows from the basis alone** (f4-gfp.4, .5, .20, .45; order-family.12).
  Each round builds its rows from the current basis: no Simplify, no
  matrix kept from round to round (monagan-2015 §2). A row t·g holds only
  column indices and reads its coefficients from g. In the plain run the
  pivot of each column is the sparsest candidate; the signature run's is
  in _Signatures_.
- **The divisor search in the order of terms** (matrix-build-time.8,
  mask-scan-time.1–.6). Symbolic preprocessing in the plain run looks,
  for each monomial new to the round but the given rows' leading
  monomials, for the element of G_min of fewest terms whose leading
  monomial divides it, ties to the element made first. At each round
  `Divisors::new` sorts G_min's elements by their terms, ties by their
  index, which is the order made, and takes their leading monomials and
  masks in that order. The sort spends nothing from the budget. The
  search reads eight masks at a time into a bit set of candidates, so
  that a block whose eight masks all fail costs no branch per element.
  It tests the candidates' divisibility in order and stops at the first
  that divides. No element before it divides the monomial with fewer
  terms, or with as many and made before it, so it is the element the
  rule names, and every matrix stays as the search in the order made
  built it. A search that finds no divisor scans every mask. Scanned in
  the order made with the best candidate kept, a search that found a
  divisor scanned as far as one that found none: of Gonnet-83_1-p65521's
  454,990 searches, seed 1, 80 % found a divisor, the first in the order
  of terms at a mean position of 288 in a G_min of 1,210 (log 0073-1).
  Three alternatives were measured against this search at the machine's
  own clock and rejected (log 0073-1): two over the tuning set but
  cyclic-9-p65521, one in single calls. A record per monomial of the basis's length at its last
  search without a divisor, so that its next search scans only the
  elements made since, as monagan-2015 §2.1 keeps an index into the
  basis in each monomial, took mean d +0.0016. Per variable and exponent
  a bit set of the elements whose exponent is at most it, ANDed 64
  elements at a time over the variables where the monomial's exponent
  lies below the largest, took Gonnet-83_1-p65521 0.172 s against
  0.161 s in single calls. A step of 32 masks by one OR of their four
  blocks' tests before the block's own took mean d +0.0105. In the
  measuring condition the call took mean d −0.0220 over the 102
  instances of the tuning set against the search in the order made,
  noise threshold 0.0039 (log 0073-2): Gonnet-83_1-p65521 0.261 s
  against 0.373 s, toricsat-veronese5d3-p0 −0.193, biomdcl-147-p65521
  −0.190, toric-veronese5d3-p0 −0.186, biomdcl-220-p65521 −0.184, and
  Geometry.Chou.337_1-p65521 −0.183.
- **The columns in a reservation past 1 MiB** (matrix-slack.1–.5, .7, .8). A
  matrix holds every row's columns back to back in one `Vec<MonoId>`, the
  run's column vector, and the columns grow by doubling. Before the row whose products
  would take them past `RESERVE_PAST`, 2^18 entries, `matrix::make_room`
  reserves `RESERVATION`, 2^23 entries or 32 MiB, or the columns and the
  row where they take more. The columns move into the reservation once,
  and the process wires its pages as the rows write them, as f4ncgb's
  stores are wired page by page (heisinger-2025 §5). glibc's malloc maps
  a request of 32 MiB outside its heap where its free memory cannot serve
  it. Past `RESERVATION` the columns grow by doubling again. The
  reservation stays in the run's buffers for its next matrix and goes back
  where its rounds end, and a matrix a replay
  keeps over ℚ shrinks its columns to their length (q-replay-time.14).
  The matrix's rows, offsets, and monomials and the monomial table
  keep growing by doubling. Outside the
  measuring condition, over 112 instances, a reservation before every
  matrix's first row took time +0.036 in the mean ln of the quotient, a
  reservation from 2^16 entries took +0.008 for memory −0.075, and one
  from 2^20 entries memory −0.043. The columns in blocks of 4096 to 2^20
  entries, each row in one block, raised katsura-13-p65521's peak by
  +0.265 in ln, and one array per thread that every matrix takes kept the
  pages of the largest columns it had held. Against the columns growing
  by doubling throughout, the tuning set took 0.885 of the memory in the
  geomean over its 112 instances with a memory reading, and 1.002 of the
  time over its 116. The 43 instances whose peak passed 20 MiB took
  −0.144 in the mean ln of the memory quotient, −0.203 over GF(65521)
  and −0.056 over ℚ, and +0.009 in time. Over GF(65521) reimer-7 peaked
  at 23.5 MiB against 38.2 MiB, mqpd-11 at 55.2 against 87.6, and
  chandra-13 at 210.8 against 264.6. mqpd-9 over GF(65521), whose largest
  matrix holds 600,075 entries, peaked at 5.9 MiB against 4.8 MiB. The 51
  instances below 5 MiB took −0.135, most of it the extension's file
  pages, which the code's layout moves.
- **Pivot rows stay unreduced** (f4-gfp.6, .7). The plain run reduces the
  rows without a pivot each fully, through a dense
  accumulator, by the pivots and the new pivots before it. A nonzero result
  becomes a new pivot (boyer-2016 §4.3, monagan-2015 §2.2).
- **Rows to reduce from the right** (cyclic-gap.1). The plain run takes the
  rows without a pivot by decreasing leading column, those of one leading
  column by decreasing row index, alone or in blocks. monagan-2015 §2.2
  examines them by increasing leading column. A row on its own, or a
  combination of a block, sweeps from its leading column to the matrix's
  last and subtracts the pivot row of every column it holds an entry at.
  Taken from the right, the short sweeps make the round's new pivots
  first, and those pivots, which lead far right, are short rows to
  subtract. The round's rank is the same in either order. A block that
  leads far left then finds its rows in the span of the rows before it and
  is done after `ZEROS` combinations reduce to zero, where taken first it
  would make its pivots by long sweeps. With each row on its own and no
  test the counts stay: every row is reduced once, and as many reduce to zero as in
  any order. Which rows reduce to zero changes. Under seed 1
  cyclic-8-p65521's plain run subtracted 538.6 million entries in 1,946
  combinations, 437 to zero, against 1,042.5 million in 1,967, 458 to
  zero, by increasing leading column. In the measuring condition the
  tuning set took 0.977 of the time in the geomean over its 116
  instances: 0.948 over the 49 over ℚ it solves, and 0.999 over the 63
  over GF(65521). cyclic-8-p0 took 10.05 s against 19.48 s, reimer-7-p0
  1.43 s against 2.58 s, biomd-103-p0 7.51 s against 11.18 s, and
  cyclic-8-p65521 0.443 s against 0.584 s. The peaks took 0.979. The
  sparsest rows first, as demin-2024 §4.2 reads, lost to it on five
  instances in session 0066-1.
- **Reduced row echelon form in every order** (gfp-unsolved.1;
  order-family.18). The plain run back-substitutes the new rows. reimer-8
  took 17.9 s with it against 55.2 s without, and chandra-8 in `block:4`
  0.16 s against 20 s.
- **Random combinations of row blocks** (gfp-unsolved.2–.8, .18, .19;
  cyclic-gap.2; plain-zero-time.1–.9). The
  plain run cuts the rows without a pivot into blocks of consecutive rows and
  reduces random combinations of each block, until `ZEROS` of them reduce
  to zero or the block has made as many pivots as it has rows
  (monagan-2017 §2, berthomieu-2021 §4.2). The blocks are cut from the
  order of _Rows to reduce from the right_ above: the first holds the
  `BLOCK` rows that lead farthest right, the last the rest at the left end
  (cyclic-gap.2). A block's combinations sweep from its last row's leading
  column. The plain run reduces each row on its own
  where p lies below `COMBINES_FROM`, 46341, the least p with p² ≥ `ZERO_BOUND`,
  2^31, where the call tracks cofactors, and where a trace or a
  learn run listens. A learn run whose trace does not listen and that
  tracks no cofactors tests those rows (_Rows ruled zero by random
  combinations_ below, learn-zero-time.1).
  reimer-8 took 9.2 s against 20.6 s. One zero would
  leave a block short with probability about 1/p, a wrong basis every few
  hundred runs. Combining with cofactors took biomd-333 past 18 min and
  16.5 GB against 0.8 s.
  Over `Gf31`, in a matrix whose rows hold `BAND_ENTRIES`, 64, entries or
  more in the mean, a block of n rows forms min(`BAND`, n − pivots)
  combinations at once in the band of _Reductants in bands of four_
  below, pivots the rows the block has made, and reduces them in one
  sweep, which reads each pivot row once for the band's combinations
  (plain-reduction-time.1–.6). One pass over each block row adds it to
  every band row, its coefficients drawn together in band-row order. The
  block takes bands while none of its combinations has reduced to zero
  and two pivots or more are left to make, and then one combination at a
  time. A band row is reduced by the band rows before it, so every
  combination comes out as one at a time makes it. A band's lanes after
  its second zero are combinations one at a time would not take: a block
  of rank r < n takes up to r + 4 combinations where one at a time takes
  r + 2, and `Stats` counts every lane. A row added to a band spends its
  length `BAND` times. Bands in every matrix, a prototype, executed
  +0.093 in the ln of biomd-333-p65521's instructions, whose rows hold
  two or three entries. Against one combination at a time, the tuning
  set took 0.981 of the time in the geomean over its 116 instances,
  0.965 over the 63 over GF(65521) both solve and 1.001 over the 49 over
  ℚ. Over GF(65521) cyclic-8 took 0.571 s against 0.945 s, reimer-8
  5.36 s against 8.64 s, biomd-103 0.054 s against 0.084 s, cyclic-7
  0.036 s against 0.048 s, and reimer-7 0.247 s against 0.310 s. They
  executed 0.92 to 1.01 of the instructions: the time fell where the
  sweep read each pivot row once for the band. reimer-8 peaked at
  341.1 MiB against 336.8 MiB.
  A block holds `BLOCK` rows, or twice the rows of the block before it
  where that block made no pivot, or the rows left where they are fewer
  (plain-zero-time.1, .4, .5, .7, .8). A block of more than `BLOCK` rows
  is reduced as every block is, and stops before its next band or
  combination once it has made more than `GROWN_PIVOTS`, 16, pivots. Its
  pivots stay, and the blocks after it take its rows again from its
  first, the first of them `BLOCK` rows, each next one sized as any block
  is (.2, .3, .6). A stretch of rows in the span of
  the pivots thus costs a band of zeros, or two combinations, per block
  of a doubling run of blocks, where it cost that per `BLOCK` rows. A
  block takes a band after a band whose combinations all made pivots
  (.9). demin-2026b §2.3, Proposition 1, bounds the chance that a block
  misses rank whatever its size. eco-13-p65521's round 26, 4,778 given
  rows and no pivot, reduced 24 combinations in 0.039 s, against 172
  in 0.257 s with every block of `BLOCK` rows, seed 1 at the machine's
  own clock. In step 7 of iteration 0078, with the learn's tests of
  _Rows ruled zero by random combinations_ below, against every block of
  `BLOCK` rows and each row of a learn's plain run on its own, the tuning
  set took 0.970 of the time in the geomean over its 102 instances, 0.969
  over the 57 over GF(65521) that finished and 0.970 over the 44 over ℚ.
  Over GF(65521) rab-eco-12
  took 1.485 s against 1.811 s, phc-rps10 0.786 s against 0.947 s, and
  eco-13 1.843 s against 2.215 s. Rejected, each against every block of
  `BLOCK` rows in the mean ln of the time over the 57 at the machine's
  own clock (log 0077-1, _Step 3_): two combinations that rule the
  doubled rows zero, or hand them to blocks of `BLOCK` where one does not
  reduce to zero, −0.0241 one at a time and −0.0323 in a band of two,
  where grown blocks stopped past 8 pivots took −0.0365; each block ended
  at its first zero and the round closed by two combinations of all its
  rows, −0.0165; four times the rows, −0.0216 against twice's −0.0325,
  sat-pret60-25 +0.171; no stop, −0.0241, Curves.curve6_30 +0.276; a stop
  past 4, 8, or 32 pivots, −0.0298, −0.0326, and −0.0319 against 16's
  −0.0341; and growth in the matrices that take bands alone, −0.0328
  against growth in every matrix's −0.0333. A cap of 1024 rows and growth
  after a block of one or two pivots moved the time within the noise. One
  combination alone after a band whose combinations all made pivots took
  eco-13 1.308 s against 1.180 s, single calls.
- **Delayed reduction mod p** (f4-gfp.8; gfp-unsolved.10). Over `Gf31` the
  accumulator holds `i64` and reduces an entry once, when the loop reaches
  its column (monagan-2015 §2.2, demin-2024 §4.2).
- **The scan of the marked columns** (row-scan-time.1–.10). A reducer
  scans its dense row in one of two ways, chosen before each row. The
  dense scan passes every column from the row's start to the matrix's
  width. The marked scan keeps a bit per column, `MARK_BITS`, 64, columns
  a word, sets the bit of every column a load, an addition into a
  combination, or a row operation writes, and visits the set bits alone,
  in increasing column order, each cleared as it is read (.1, .6). Every
  column that holds an entry has its bit set, so the two scans take the
  same entries in the same order and make the same rows, and no count of
  the call report changes. A row takes the marked scan where the columns
  the dense scans of the rows before it would pass exceed `MARKED_RATIO`,
  2, times the entries their loads, additions, and row operations wrote,
  once `MARKED_ROWS`, 4, rows are counted (.2). Both counts are halved
  every `MARKED_DECAY`, 64, rows, so that they weigh the recent rows (.3).
  The entries written are the budget's counts of the entries subtracted
  and added (call-report.4), read once a row, so that a band's entries
  count with the next row, and the load's length (.4).
  The row operations stay the fields' `sub_mul`, and a pass of its own
  marks the subtracted row's columns after it (.5). The marks are the
  reducer's own, a word per 64 columns of its matrix, made at its first
  marked row: 1/64 of `Gf31`'s dense row (.8). The signature run's test
  of a combination takes the scan a row would take, and where it meets
  an entry in a column without a pivot it zeroes the entries at the bits
  left (.9). The back-substitution and the bands keep the dense scan
  (.7). Rejected: the reach of the row in the graph of the pivot rows,
  found by a depth-first search before the numerical work
  (gilbert-1988 §2, Lemma 1, via bouillaguet-2016 §2.3), which assumes
  no cancellation and follows a chain of binomial rows past the column
  where its entries cancel; the dense scan alone, which heisinger-2025
  §6.1 Remark 1 reports ahead of tracked nonzero positions in its
  matrices; the marked scan for the rest of a matrix once its counts
  first choose it, under which poseidon-t2F2P3-p65521's inter-reduction,
  rows of 8,981 entries in the mean, took 0.202 s against 0.081 s; a
  ratio of 8, which left colour-paley17-k4-p0 at +0.003 in the ln of the
  time where the marked scan for good took −0.229, and a ratio of 1,
  whose highest instance took +0.077 where a ratio of 2 took +0.029; and
  a count kept at each row operation of the scan, under which
  ccode-newton40t6-p65521's round 1 of 412,062 rows reduced in 0.0189 s
  against 0.0151 s. Those figures were taken outside the measuring
  condition (log 0072-1; the +0.003 is plan 0072's, row-scan-time.2). On
  toric-veronese5d3-p0 the learn's plain run
  reduced rows of 2 entries in a matrix of 68,121 columns, and the dense
  scan executed 0.835 of the call's instructions. In the measuring
  condition, run `20261016T100129Z-e7cc58d2`, the tuning set took 0.949
  of the time in the geomean over its 102 instances, 0.902 over the 44
  over ℚ and 0.987 over the 57 over GF(65521) both solve:
  toric-veronese5d3-p0 0.310 s against 1.037 s, toricsat-veronese5d3-p0
  0.330 s against 1.068 s, toricsat-veronese3d6-p0 0.080 s against
  0.184 s, sudoku-n9u55s6-p0 0.279 s against 0.409 s, and noon-9-p0
  2.15 s against 2.68 s. The calls executed 0.923 of the instructions in
  the geomean, and their peak memory took 0.990.
- **Reductants in bands of four** (row-traffic-time.1–.13). A signature
  round over `Gf31` that tracks no cofactors, in a matrix whose rows hold
  `BAND_ENTRIES`, 64, entries or more in the mean, reduces each run of
  consecutive reductants, no reducer row between them, in bands of
  `BAND`, 4, rows. A band is reduced in one sweep over the columns, which
  reads each pivot row once for the band's rows, where the reduction of
  one row at a time reads it again for each (martani-2012 §3.5.1,
  boyer-2016 §4.2). The band's dense rows are one array whose element
  holds their four entries of a column, 32 bytes. A pivot row that one
  band row alone needs is subtracted from that row alone (boyer-2016
  §4.2). The others are subtracted from the four rows by SSE2's
  `pmuludq`, two rows a multiply, which baseline x86-64 has
  (berthomieu-2021 §4.1). A band row that meets the leading column an
  earlier band row took in the sweep is reduced by that row through a
  relation between their values, applied at each later column without a
  pivot (faugere-2010 §4.4). Every band row comes out as the reduction of
  one row at a time makes it, entry for entry, so no result, and no stat
  or record of a learn run, changes. Where a signature round tests its
  rows, a band moves where the test starts, and with it the counts in
  `Stats` (signature-zero-time.13). `Gf64` and `Gf64Pair`, whose row
  operation multiplies by Shoup's product, take no band, and a run that
  tracks cofactors takes none. `Gf31Pair`, the field of a pair learn
  over ℚ, takes none either, as its band would hold 64 bytes a column
  (row-traffic-time.13). The band and its rows' work vectors come from
  the run's buffers and go from round to round (monagan-2007 §2.2). A
  reduction's first band zeroes the band whole, up to its matrix's width,
  and trusts no band the buffers held to be zero (row-traffic-time.10).
  Each band row is built in two work vectors of its own and copied at its
  length (row-traffic-time.11, monagan-2015 §2.2). Each pivot row a band
  subtracts spends its length once per band row it is subtracted from,
  right after the subtraction (row-traffic-time.12), so that `WORK` units
  bound a band's time as they bound a row's, not the ticks of its rows
  alone.
  The plain run's random combinations take the band too, in matrices of
  rows as long (_Random combinations of row blocks_ above). There a
  band's combinations after its second zero are ones that one combination
  at a time would not reduce, and the stats count them.
  Each reduction of mq-12-p65521's four
  widest matrices subtracted 8.3 to 9.1 million of their 9.2 to 9.8
  million entries, at 0.36 to 0.52 ns an entry. On katsura-12 the bands
  read 488 million pivot entries where one row at a time read 1,595
  million. Outside the measuring condition, over GF(65521), single
  calls: bands of two rows took katsura-13 4.380 s against four's
  3.501 s; bands of eight took mq-11 0.359 s against 0.396 s, and eco-12
  0.378 s against 0.349 s. Bands in every matrix, every pivot row
  subtracted from the four rows, took cyclic-7 +0.120 in the ln of the
  time quotient and biomd-365 +0.114, three calls each. With bands from
  128 entries a row biomd-103 lost its −0.073 to +0.019. Against the
  reduction of one row at a time, the tuning set took 0.926 of the time
  in the geomean over its 116 instances, 0.867 over the 63 over GF(65521)
  both solve and 1.002 over the 49 over ℚ. Over GF(65521) mq-12 took
  3.96 s against 9.69 s, mqpd-12 3.70 s against 8.22 s, katsura-13
  5.51 s against 10.71 s, and eco-13 3.35 s against 3.84 s. katsura-13
  peaked at 228.2 MiB against 224.7 MiB, and katsura-12 at 65.1 MiB
  against 53.7 MiB.
- **Sorts on keys** (matrix-sort-time.1–.8). A matrix sorts its columns,
  the round's monomials, into decreasing order, and a signature round
  sorts its rows by signature. In grevlex each sort keys its elements as
  it starts. The key is the degree, then each exponent from the last
  variable down, complemented in its lane's bits, and its integer order is
  grevlex (monagan-2007 §2.1). It is built from whole words. An element is
  one `u128`, the key's first `KEY_BITS`, 96, bits above the monomial's id or the row,
  and the sort compares integers. Elements whose first 96 bits agree are
  sorted by the order's comparison. Past 12 variables in `Dense8`, 6 in
  `Dense16`, and 3 in `Dense32` two monomials of one degree can agree
  there. The keys are dropped when the sort returns. SINGULAR compares
  monomials by their rank, one integer whose order is the monomial order,
  and measured it 0.9 to 3.7 times as fast as a comparison of exponent
  vectors (bachmann-1998 §4). Lex and the block orders keep the comparison
  sort, which reads both monomials' degrees and words from the table at
  each comparison, and sort more than `RUN`, 2^15, elements in runs
  (_Limits_, _Units of work_). In grevlex the two comparison sorts took 6.6% of the
  instructions in the mean over 81 instances and 52.6% of sparse-18's
  over GF(65521). A
  key built lane by lane through `Format::exponent` took 2.7 times the
  instructions of the key from words on sparse-18, and an element of
  24 bytes, the key's first 128 bits beside the id, raised sparse-18's
  peak by 47 MiB. The probe of the lookup per term, `Table::find`, is
  `#[inline(always)]`: a build that left it to the compiler called it out
  of line, and henrion-7 over GF(65521) executed 15.1% more instructions.
  Against the comparison sorts, the tuning set took 0.967 of the time in
  the geomean over its 116 instances, 0.952 over GF(65521) and 0.985 over
  ℚ, and executed 0.950 of the instructions over GF(65521). Over
  GF(65521) sparse-12 took 9.7 ms against 13.3 ms, and sparse-18 1.75 s
  against 2.13 s. sparse-18 peaked at 270.0 MiB against 248.2 MiB. A
  build of the comparison sorts peaked at 266.4 MiB on it in another run:
  its peak moves by up to 19 MiB between builds of the same code.

### Fields

- **Two fields, chosen once per call over GF(p)** (gfp-large-primes.1–.7,
  .10). `Gf31`
  takes the primes below 2^31 with `u32` coefficients. `Gf64` takes the
  primes from 2^31 to 2^64 with `u64` coefficients. A call over ℚ runs
  its learn runs in `Gf31Pair`, each over the learn prime and the
  verification's, its fallbacks in `Gf31`, and its replays in `Gf64`
  or `Gf64Pair` (_Over ℚ_). A row operation in
  `Gf64` below 2^63 computes the multiplier's pre-inverse once and each
  product in one word, from 2^62 with one correction (vanderhoeven-2016 §2.2,
  harvey-2014 §2). At and past 2^63 each product is a `u128` remainder,
  because the two-word pre-inverse form took 2.0 to 2.5 times as long. The
  corrections of a `Gf64` row operation are branch-free; branches took 2.3 to 3.7 times as long.
  `Gf64` tests primality by the strong probable-prime test; trial division
  took 7.7 s at 2^64 − 59, the test 42 µs.
- **Entries up to 2p below 2^62** (q-modulus-time.3–.5). Below 2^62 a
  `Gf64` row operation keeps the dense row's entries in [0, 2p). It
  subtracts each product v·c − q·p, which lies in [0, 2p), without its
  correction, and adds 2p where the subtraction borrows. The read of an
  entry takes p off where the entry is p or more. The redundant form
  needs p below a quarter of the word (harvey-2014 §3). The loop takes
  one entry per turn. Outside the measuring condition, over 45 instances
  over ℚ, the calls took 0.978 of their time with the correction per
  product at primes of [2^62, 2^63) in the geomean, and one entry per
  turn took 1.001 of the time
  of four, within the noise.

### Signatures

- **A batched F4 over signatures** (signature-pruning.1–.3). Each round
  takes every pending signature of least degree and reduces its rows in
  one matrix, in Schreyer's order on signatures (lairez-2022 §5.7 Thm 62,
  gao-2016 §3).
- **Syzygies from three sources** (signature-pruning.5, .6, .16, .25, .26;
  gfp-unsolved.14). The run learns a syzygy signature from every reduction
  to zero, from the Koszul syzygies of the inputs, and from coprime leading
  monomials. It keeps the minimal ones and queues no signature they divide
  (eder-2017 §6.1, §7.2, roune-2012 §3). A new syzygy does not prune the
  queue: pruning took 32 % of chandra-11's instructions.
- **The S-pairs of a new element** (signature-pair-time.1–.13). When an
  element enters, the run picks two earlier elements of largest sig-lead
  ratio as its base divisors: one whose leading monomial divides the new
  element's, and one whose signature divides its signature. A triangle of
  bits, one per pair of elements, records which S-pairs are known to have
  a syzygy signature. An S-pair of the new element goes without its
  signature formed where a base divisor's S-pair with the same element
  has its bit set and roune-2012 Theorem 7 or 8 makes that signature
  divide the S-pair's (roune-2012 §3.5). An S-pair whose two elements
  have the same (sig-lead ratio, index) is dropped. Every other S-pair's
  signature is formed word by word in the table's scratch words and
  tested there.
  Of these, only a signature the run queues or enters as a Koszul
  syzygy's is interned. A query of the known syzygy signatures tests their masks
  eight at a time and tries the member that answered its last yes first.
  The base divisors eliminate only S-pairs the signature criterion
  eliminates, so every round reduces the same rows. A memory check at the
  start of a round that finds the resident set past the ceiling gives the
  triangle back, and the run goes on without
  base divisors
  (roune-2012 Appendix B.2). The base divisors eliminated 5,183,115 of
  chandra-12's 13,099,521 S-pairs and 247,747 of biomd-365's 435,711.
  chandra-13 over GF(65521) took 8.84 s against 16.47 s, biomd-365
  0.073 s against 0.174 s, and the tuning set 0.917 of the time in the
  geometric mean.
- **Rewriting by the sig-lead ratio** (signature-pruning.4, .10, .21, .23,
  .27). The row at a signature σ is a·g for the g of largest sig-lead
  ratio whose signature divides σ (arri-2011 §7.2, eder-2017 §13).
  Symbolic preprocessing in the signature run takes, for each monomial
  new to the round, the first element of the reducer order, (sig-lead
  ratio, index, terms, element), whose leading monomial divides it
  (signature-pruning.10).
- **The F4-RB matrix** (signature-pruning.8, .9, .11–.14;
  gfp-unsolved.12, .13). Only the reductant is a given row, and symbolic
  preprocessing supplies the rest. Rows are reduced in increasing
  signature order. A reducer row whose column already has a pivot is
  dropped where the row before it has its signature (eder-2017 §6.2),
  and is reduced as a reductant otherwise (eder-2017 §13, §14.3). cyclic-8 took 1.4 s
  against 42 s with this, and two instances returned wrong bases without
  gfp-unsolved.13.
- **Rows ruled zero by random combinations** (signature-zero-time.1–.14;
  learn-zero-time.1–.6; zero-rows-time.1–.4).
  A run is a stretch of a signature round's rows, in signature order, that
  no reducer row setting a pivot interrupts. Every row of a run is reduced
  by the same pivots and by the rows the run makes before it. Where
  `STREAK`, 2, rows of a run have reduced to zero
  one after another, alone or in a band, a test takes the rest of the run, up to the next
  reducer row that sets a pivot, past the band's last row where the streak ends in a band
  (signature-zero-time.13). Where s random combinations of the rest's
  rows each reduce to zero by the run's pivots, every row of the rest
  joins the syzygies (monagan-2017 §2). s is the least integer with
  (p − 1)^s ≥ `ZERO_BOUND`, 2^31 (`zero_combinations`, demin-2026b §2.5), 2 at 65521:
  a test passes over a row that does not reduce to zero with probability
  at most 2^−31. A combination stops at its first entry in a column
  without a pivot, which shows it nonzero, and a failed test leaves its
  rows to be reduced as the run's other rows are, in bands where the round takes them.
  The s combinations are reduced one after another in the dense row, not in a band,
  so a failed test builds none past the one that stops (signature-zero-time.14).
  After a failed test the run's tests
  take blocks of `FIRST`, 8, rows, doubling after each block that passes
  up to `BLOCK`, 64. A group of fewer than `GROUP_ROWS`·s, 2s, rows takes no test. Each
  combination is a reduction in `Stats` and a tick of the budget. No test
  runs at p = 2 or p = 3, in a call that tracks cofactors, where a
  trace listens, or in a learn's signature run (learn-zero-time.3).
  Over ℚ the verification's run at a fresh prime takes them,
  where the verification runs one (q-learn-time.7, q-later-runs-time.7).
  The rows that reduce to zero took 0.44 to 0.76 of the signature run's
  subtracted entries on chandra, eco, mqpd, mq, and katsura over
  GF(65521), without the bands, and in mq and mqpd they stand in one tail at each round's end.
  With the bands they took 0.31 to 0.39 of mq's calls from mq-10 on, most of it in bands,
  at the machine's own clock.
  In the measuring condition the tuning set took 0.972 of the time in the
  geometric mean over its 116 instances, 0.954 over the 63 solved over GF(65521)
  and 0.994 over the 49 solved over ℚ:
  mq −0.240, mqpd −0.160, and katsura −0.078 in the mean
  ln over GF(65521), and the 94 instances outside mq, mqpd, katsura, and henrion −0.003.
  mq-12 took 2.53 s against 3.96 s, and katsura-13 4.87 s against
  5.51 s. A test after each zero, the whole rest at every test, blocks
  without a cap, and combinations reduced to their end took −0.0003,
  −0.0010, +0.0031, and +0.0127 against its prototype over GF(65521), at
  the machine's own clock.
  A learn run's plain run takes the tests where its trace does not listen
  and the call tracks no cofactors (learn-zero-time.1, .4, .5;
  zero-rows-time.1). It reduces each row on its own, by decreasing leading
  column (_Rows to reduce from the right_ above), and sets every pivot row
  before its first, so its rows to reduce are one run. Once `STREAK` of
  them have reduced to zero one after another, its first test takes the
  rows to the matrix's end, and after a failed test blocks of `FIRST` rows
  doubling up to `BLOCK` (learn-zero-time.2, zero-rows-time.4). A group
  reduces from the leftmost leading column among its rows, its last row's.
  A failed test leaves its rows to be reduced each on its own until the
  next streak (zero-rows-time.2, .3). A row a passing test rules zero makes
  no pivot, as it would reduced on its own, so the record is the one each
  row on its own makes (_The record_ below). The tests' coefficients come
  from the run's seed. Over `Gf31Pair` s is the first prime's, 2
  (learn-zero-time.6): one draw gives both lanes their coefficients, and
  a combination reduces to zero where it does at both primes. With each
  row on its own and no test, the pair learn's plain run of
  Geometry.Chou.337_1-p0 reduced 12,047 rows, 10,924 of them to zero, and
  its zero rows took about 0.540 of the call, each round's reduction time
  split by its rows, seed 1 at the machine's own clock. Against the
  learn's tests,
  over the 44 instances over ℚ at the machine's own clock (log 0078-1,
  _Step 3_): a test of the next block right after the next row that makes
  a pivot took +0.0047 in the mean ln of the time, a failed group's first
  half tested again +0.0049, and a test after each row that reduces to
  zero +0.0038. In step 7 of iteration 0078, beside the grown blocks of
  _Random combinations of row blocks_ above, the 44 instances over ℚ took
  0.970 of the time in the geomean: latin-n4-p0 0.292 s against 0.413 s,
  Geometry.Chou.337_1-p0 3.13 s against 4.37 s, cyclic-7-p0 0.226 s
  against 0.281 s, and cyclic-8-p0 8.63 s against 10.08 s. Tests in the
  learn's signature run as well took −0.0001
  over the 49 instances over ℚ of the tuning set of iteration 0063
  against the plain run's tests alone, at the machine's own clock
  (learn-zero-time.3).
- **On by default** (signature-pruning.18). Against the plain run the
  signature run took 0.53 of the time in the geometric mean over 53
  instances and solved six it did not. It failed on cyclic-8 and reimer-7,
  which the plain run solved, and it lost badly on others: reimer-6
  15.9×, biomd-103 14.8×.
- **The handover** (handover-time.1–.5, .7, .8; outgrown-round-size.4;
  gfp-unsolved.16, .17). The
  signature run decides once at the gap round: the first round that makes
  an element whose signature degree passes its leading degree by
  `HANDOVER`, 4, or more. The gap round hands over where it reduced fewer rows to
  zero than `GAP_ZEROS`, a half, of the elements it made, or where more than
  `DIVISIBLE`, 6 in 10, of them are divisible: a leading monomial of the
  basis before the round divides
  theirs. Otherwise the signature run keeps the call. The outgrown test
  reads every round that made an element, and hands over where `OUTGROWN`,
  8 in 10, or more of them are divisible. At a handover the call starts over in the plain run
  from its input. At their gap rounds eco reduced 0.66 to 0.96 rows to
  zero per element made, and reimer and cyclic 0.06 to 0.33. biomd-103 and
  biomd-333 reduced 2.07 and 2.10, and 0.96 and 0.70 of their elements were
  divisible. Kept to its end, reimer-4 to -8's signature runs took −0.031 to +4.784
  in ln over GF(65521) against a handover at the gap round. biomd-147
  reaches no gap of 3: its signature run made 1,532 elements against the
  plain run's 428, and it hands over at its sixth round, of 390 elements,
  0.85 of them divisible. Over rounds of 100 elements or more the largest
  divisible share on an instance the signature run wins was biomd-200's
  0.66, on the tuning set of iterations up to 0066. On the tuning set of
  iterations 0067 on, with the early round's least at 100, the outgrown
  test at every round took −0.0212 in the mean ln of the time over its
  101 instances but cyclic-9-p65521 against rounds of 100 elements or
  more, at the machine's own clock, and at rounds of 50, 25, 10, or 5
  elements or more +0.0022, −0.0084, −0.0170 and −0.0185, and −0.0214
  (log 0075-1). Geometry.Chou.64_1-p65521 took about −1.0 where the
  test read rounds of 25 elements or fewer, Geometry.Chou.139_1-p0 −0.93
  of 10 or fewer, and biomdcl-123-p0 −0.17 of 5 or fewer, and no
  instance rose past +0.065. A
  round that made no element has no share to read. In the measuring
  condition, with the early round's leasts of **The early round** below,
  Geometry.Chou.64_1-p65521 took 0.050 s against 0.122 s, at a peak of
  6.8 MiB against 28.7 MiB, Geometry.Chou.139_1-p0 −0.86 in ln, and
  biomdcl-123-p0 −0.18 (log 0075-2).
  In the measuring condition, against a handover at every gap round,
  the two tests took −0.063 in the mean ln over the 116 instances of the
  tuning set. eco took −0.237 to −0.592 over GF(65521) and −0.515 to
  −0.710 over ℚ, eco-13 4.244 s against 5.377 s, and biomd-147 over
  GF(65521) 0.750 s against 4.728 s, at a peak of 63.7 MiB against 384.4 MiB.
  The other 104 took +0.001 in the
  mean. Under a handover at every gap round, the restart lost at most 0.84 s on the tuning set,
  and 18.6 s on cyclic-9. A call that tracks cofactors never hands over:
  the plain run's cofactors for eco-9 took 562 MB against 13.5 MB. No
  source hands a signature run over during a call. roune-2012 Appendix B.4
  finds that SB should probably not be used where its signature Gröbner
  basis is known to be much larger than the minimal one.
- **The early round** (early-handover-time.1, .2, .5–.7;
  outgrown-round-size.1–.3, .5; threshold-recheck.16;
  input-redundancy-time.7). The signature run
  decides once at its early round as well: the first round that makes an
  element whose signature degree passes its leading degree by `EARLY`, 2,
  or more. Where that round made its least elements or more, it
  hands over where it reduced fewer rows to zero
  than `EARLY_ZEROS`, a quarter, of the elements it made, or where more than
  `DIVISIBLE`, 6 in 10, of them are divisible, the gap round's share. At the machine's own clock,
  outside the measuring condition, 5 and 7 in 10 took +0.001 and −0.002
  in the mean ln of the time over the 101 instances of the tuning set but
  cyclic-9-p65521, within the margin. Otherwise the run goes on, and no later round
  decides by these tests. A round that reduced no row to zero is read as
  well. With its generators in reduced row echelon form (_Matrices and
  reduction_, **The inputs**), minrank-ks10r3k15-p65521's early round made
  4,165 elements and reduced none; read only where a row reduced to zero,
  as before, its update formed 11.1 million S-pairs in 19.9 s and the
  call ran past 120 s, where plan 0068 gives the plain run from the start 1.31 s. With
  the generators as read, the same round reduced 219 rows to zero and
  handed over (log 0068-1). Where the early round is the gap round, its
  tests run first, and the gap round's on the same counts. The early
  round of biomd-103-p65521, its fourth, ended at 0.003 s of the call,
  and its gap round, its seventh, at 0.49 s. biomd-103's early round
  reduced 15 rows to zero for 148 elements, and 260 of biomd-365's 412
  were divisible. biomd-315, which the signature run wins, reduced 0.60
  rows to zero per element at its early round, and 0.43 of its elements
  were divisible. The tests at every round of a gap of 2 or 3 handed
  biomd-200-p65521 over, +2.31 in ln. The tests at every early round,
  whatever its size, handed noon over, +1.23 in the mean ln over
  GF(65521). In the measuring condition, against the gap round alone,
  the early round took −0.034 in the mean ln over the 116 instances of
  the tuning set, and the peaks −0.040 in the mild form.
  biomd-103-p65521 took 0.083 s against 0.834 s, at a peak of 5.2 MiB
  against 27.4 MiB, biomd-103-p0 −0.11, and biomd-365 −0.50 over
  GF(65521) and −0.45 over ℚ. reimer-8-p65521 hands over at its eighth
  round in place of its ninth. The other 108 instances took −0.006 in
  the mean.
  The least is `EARLY_ROUND`, 25, where the plain run after a handover
  would combine rows, and `EARLY_ROUND_ALONE`, 100, where it would reduce
  each row alone: where it listens, tracks cofactors, or runs at a prime
  below 46341 (`f4::plain_combines`). The run's hooks and its prime fix
  the least when the run starts. A plain run that reduces each row alone
  reduces each row that reduces to zero, save those a learn's tests rule
  zero (_Rows ruled zero by random combinations_ above), work the signature run's
  criteria spare. In iteration 0075, before the learn took its tests,
  eco-11-p0's call, a pair learn, took 0.409 s and 2,918 reductions to
  zero in its plain run, where its signature run took
  0.096 s and 585, and a handover at its early round of 45 elements took
  +0.61 in ln (log 0075-1). At the machine's own clock, with the outgrown
  test at every round, these leasts took −0.0453 and −0.0438 in the mean
  ln of the time over the 101 instances of the tuning set but
  cyclic-9-p65521. Where the plain run combines rows, 1, 10, and 50 took
  −0.0406, −0.0455, and −0.0397, and 100, the outgrown test alone,
  −0.0212. Where it reduces each row alone, 50 took −0.0436. One least of
  25 for the early round and the outgrown test took −0.0292. At these
  leasts chandra-12-p65521 took −0.76 and chandra-11-p65521 −0.74 to
  −0.75. At 25 in every run, with the outgrown test at rounds of 100
  elements or more, eco-13-p65521 took −0.45 and critpt-n6f4g2-p65521
  −0.29. Their early rounds made 45 to 66 elements and reduced 0 to 3
  rows to zero.
  rab-eco-12-p65521 took +0.31: its early round reads as eco-13's and
  chandra-12's, and no count the four tests read tells them apart. In
  the measuring condition, against the rows of `0.60.0-10c69c5`, whose
  early round's tests and outgrown test read rounds of 100 elements or
  more, the leasts and the outgrown test at every round took −0.0419 in
  the mean ln of the time over the 102 instances of the tuning set, at a
  noise threshold of 0.0070, and −0.0453 of the peaks in the mild form
  (log 0075-2). chandra-12-p65521 took 0.509 s against 1.109 s,
  chandra-11-p65521 −0.73, eco-13-p65521 −0.42, critpt-n6f4g2-p65521
  −0.32, and phc-rps10-p65521 −0.18; rab-eco-12-p65521 +0.27, and
  eco-11-p0 and chandra-11-p0 +0.001. No source decides between the two
  engines from a run's first rounds, nor sets the size of a round a test
  of the handover reads.
- **The pair work** (pair-work-handover.1, .2, .4–.6; pair-work-ratio.1,
  .2; pair-work-handover.3 withdrawn). Where the early, the gap,
  and the outgrown tests kept the run, the signature run reads every round
  whose update would form `PAIRS`, 1,000, S-pairs or more, from its first
  round on. Where m elements enter a basis of B, the update forms
  m·B + m·(m − 1)/2 S-pairs. The round hands over where they reach
  `PAIR_WORK`, 1, times the entries its row operations subtracted, before
  the update forms any. An S-pair costs the update a formed signature and
  divisibility tests against the base divisors and the known syzygy
  signatures. An entry subtracted costs the reduction one multiply-add.
  satlib.uuf50_8's round 1 reads 36,232 S-pairs against 5,617 entries.
  Without the test, its round 2 read 2,881,655 against 49,373, and its
  update took 5.57 s, and the update of its round 3 had not finished at
  120 s. sat-pret60-25's round 1 reads 29,211 against 8,553, and brent-223's
  572,364 against 1,905. In iteration 0067, at a multiple of 4, over the
  tuning set, seed 1, the test read 426
  rounds and handed 17 calls over. Of the rounds it kept,
  Gonnet-83_1-p65521's third read 2.44, and the round after it handed over
  at 4.47. No other round it kept read more than 1.74. The S-pairs alone
  do not tell: noon-9-p0's rounds of 1.3 million S-pairs and more, where
  the signature run takes less time than the plain run from the start,
  subtracted more entries than that. The S-pairs against the entries of
  the round's matrix read 5 to 11 on colour-g18m76s1, whose time goes into
  its reductions. Without the bound of `PAIRS`, Geometry.Chou.459_1-p0's
  round 1 would hand over at 75 S-pairs, 4.69 times its entries, where the
  signature run took less time than the plain run from the start by 0.15
  in ln. In iteration 0067, prototypes at the machine's own clock took about the same over
  the tuning set with a multiple of 2 as with 4, and kept sian-daisymamil4's,
  biomdcl-220's, and Gonnet-83_1's runs with 8. A bound of 10,000 S-pairs
  kept biomdcl-220's and distgeo-fw-laman10's. In iteration 0067, in the measuring condition,
  against the three tests alone, the pair-work test took −0.277 in the
  mean ln over the 102 instances of the tuning set, and the peaks −0.147
  in the mild form. satlib.uuf50_8 took 0.61 s over GF(65521) and 0.65 s
  over ℚ, where the three tests alone hit the time limit. sat-pret60-25
  took 0.068 s against 1.022 s, and sian-daisymamil4 0.074 s against
  1.895 s, at a peak of 4.6 MiB against 44.7 MiB. The other 85 instances took
  −0.007 in the mean.
  In the call reports of seed 1 at the multiple of 4 (log 0080-2), the
  rounds the test kept where the plain run from the start takes less
  time read 1.27 to 3.67 S-pairs an entry: colour-g18m76s1-k4-p0's round
  1 3.67, sat-pret60-25-p65521's round 1 3.42, Geometry.Chou.459_1-p0's
  round 4 1.69, Geometry.Chou.460_1-p65521's round 2 1.68,
  colour-paley17-k4-p0's round 1 1.55, and phc-i1-p0's round 4 1.27,
  where neither engine wins. Of the calls the signature run takes in
  less time, noon-9-p0's round 9 read the most, 0.884, then noon-8-p0's
  round 3, 0.778. In screens at the machine's own clock, outside the
  measuring condition, against 4, the multiple 1 took −0.0290 and −0.0301
  in the mean ln of the time over the 101 instances of the tuning set but
  cyclic-9-p65521, 2 took −0.0252 and −0.0229, and 1/2 −0.0245, where
  noon-8-p0 and noon-9-p0 hand over at +0.45 and +0.60. At 1, a least of
  300 S-pairs took −0.0048 and one of 100 +0.0182: biomd-200-p65521,
  biomd-705-p65521, rabl-biomd-315-p65521, noon-8-p0, and noon-9-p0 hand
  over at +0.44 to +2.15.
  In the measuring condition, against the rows of `0.64.0-ba9057e`, whose
  multiple is 4, the multiple 1 took −0.0298 in the mean ln of the time
  over the 102 instances of the tuning set, at a noise threshold of
  0.0016, and −0.0303 of the peaks in the mild form: −0.052 over the 44
  over ℚ and −0.013 over the 58 over GF(65521) (log 0080-3).
  colour-g18m76s1-k4-p0 took 0.179 s against 1.210 s, at a peak of 10.4
  MiB against 52.1 MiB, colour-paley17-k4-p0 −0.35,
  Geometry.Chou.460_1-p65521 −0.31, sat-pret60-25-p65521 −0.25, and
  Geometry.Chou.459_1-p0 −0.16, at a peak of 112.1 MiB against 198.4
  MiB. The other 97 took −0.001 in the mean.
  No source hands a run over by its S-pairs against
  its reduction. roune-2012 §D.3 proposes to compare two algorithms' work
  by counts of divisor queries, monomial multiplications and comparisons,
  and ground field operations, in place of counts of reductions, and
  reports no such comparison.

### Lex and FGLM

- **Grevlex, then FGLM** (fglm-lex.1, .2). F4 in lex ran past 30 s on
  instances whose grevlex basis takes under 0.1 s. FGLM changes a basis of
  degree D in at most 0.016 s up to cyclic-7 (D = 924), and in 19.2 s on
  sparse-12 (D = 4096) (faugere-1993 §1, eder-2013a §5).
- **F4 in lex as the fallback** (fglm-lex.3–.5, .27; monomial-formats.11).
  Where the ideal is not zero-dimensional, or its staircase passes
  `MAX_STAIRCASE`, the call starts over in lex from its input, after the
  grevlex run has given back its memory. The fallback wastes the grevlex
  run.
- **Matphi** (fglm-lex.7). Every border monomial's normal form is
  computed once, before the change. faugere-1993 prefers normal forms on
  demand for the coefficient growth over ℚ, which GF(p) does not have.
- **NewBasis** (fglm-lex.8–.11). Candidates wait in a heap under lex, and
  linear dependence is tested against a semi-echelon form that carries its
  combinations (faugere-1993 §4, Thm 5.1). The change costs about D³/2
  multiply-adds.
- **Cofactors in lex** (fglm-lex.12–.14). Each lex element is divided by
  the grevlex basis, and the quotients are composed with its cofactors
  (buchberger-1985 §6.7). A run that tracks no cofactors pays nothing for it.

### Trace and replay

- **The record** (trace-replay.1–.4, .8, .9, .12, .16, .17;
  gfp-unsolved.6, .11). A learn run reduces each row on its own and records
  only the matrices that made a new element, and of each the rows a
  replay needs: the counted rows and, to closure, the pivot row of each
  column a needed row holds. Per row the record holds its element and
  multiplier, in the learn run's order, and per counted row the leading
  monomial it must reach. A replay builds those rows straight from the
  record, with no pairs and no symbolic preprocessing, and repeats the
  learn run's arithmetic (traverso-1989, berthomieu-2021 §4.3, demin-2024 §4.4,
  parisse-2013 §4). A learn run's plain run takes its rows to reduce by
  decreasing leading column (cyclic-gap.1), so its counted rows are those
  of the shortest sweeps, and a replay reduces those. Under seed 1 each
  replay of cyclic-8-p0 subtracted 280.8 million entries against 674.1
  million by increasing leading column, of cyclic-7-p0 10.2 million
  against 21.0 million, and of reimer-6-p0 0.77 million against 3.36
  million.
  Where no trace listens, the learn's plain run rules rows zero by tests
  (_Rows ruled zero by random combinations_ above). A row a test rules
  zero is recorded as a row that reduced to zero: it is no counted row and
  makes no pivot, as when it reduces to zero on its own
  (learn-zero-time.1).
- **Misfits** (trace-replay.6, .7). A replay checks the generators and
  every counted row's leading monomial, and stops at the first that differs
  with `Error::Misfit`.
- **Lex records** (fglm-lex.15). A lex record whose basis FGLM changed
  holds the grevlex steps and a flag, and its replay runs FGLM after them
  (berthomieu-2021 §4.5). Any other lex record holds the lex run's steps.

### The result

- **Each monomial's text written once** (text-time.1–.4, .9). The basis
  text over GF(p) writes each distinct monomial's text once per call into
  an arena, a length byte before it, and copies it from there for each
  further term with that monomial. An index of one `u32` per monomial of
  the run's table, zeroed for each basis text, holds the offset of the
  text's length byte, 0 where the arena holds no text of it: 6.5 MB at
  sparse-18-p65521's 1,636,791 monomials. A text of at most `COPY`, 32
  bytes, is copied by one copy of 32 bytes where the term's reservation
  holds 32 bytes past what the term has written, and its length alone
  counts as written. The reservation stays the one above. A text past 255 bytes is
  not kept, and each of its terms writes it from its exponents. The
  cofactor text over GF(p) writes each monomial from its exponents. A
  monomial recurs across a basis: noon-9-p65521's 2,399,314 terms hold
  22,638 distinct monomials. In the measuring condition, with the entry
  below, a call took 0.974 of the time in the geomean over the tuning
  set's 116 instances, and 0.963 over the 64 over GF(65521). noon-9 over
  GF(65521) took 0.812, 0.607 s against 0.747 s.
- **Integers over ℚ by divide and conquer** (q-basis-text-time.1–.9;
  text-time.6, .8). An integer below 2^64 is written as above, and one of
  2 to `SPLIT_LIMBS`, 12, limbs by passes that each divide its limbs by
  10^19, from the top limb down, the remainder of each pass the next
  nineteen digits. Each step divides a `u128` by the constant 10^19, as
  the compiler computes it. A larger integer x of k chunks of nineteen
  digits, k = ⌊bits/63⌋ + 1, splits at h = 2^j, the largest power of two
  below k: x = q·Dⱼ + r with Dⱼ = 10^(19·2^j), and its text is q's k − h
  chunks followed by r's h (brent-2010 §1.7.2, Algorithm 1.26). Each part
  splits the same way, down to parts
  of at most `LEAF_CHUNKS`, 8, chunks, which the passes write with their
  zero chunks at the top. A split's quotient is Barrett's (barrett-1987,
  _Longmod procedure_): x's limbs from
  l − 1 up times μⱼ = ⌊2^(128l)/Dⱼ⌋, l the limbs of Dⱼ, shifted down by
  l + 1 limbs. Dⱼ is subtracted from the remainder at most three times
  (vanderhoeven-2016 §2.2.1, Proposition 1). The quotient's product takes
  its columns from l − 1 up alone, which lowers the quotient by at most
  one more, as barrett-1987 calculates half of each product, and both
  products sum each column into a three-word
  accumulator. A level's power and reciprocal are made once per process,
  by the first integer that takes the level, and kept for every later
  call and thread. Each integer past 12 limbs takes one vector for its
  limbs, its chunks, and its splits' parts and columns. The passes run
  about k²/2 division steps, each on the chain of the one before, and
  the splits about as many multiply-adds, which no chain holds up. In a
  microbenchmark of the integers of 20 digits or more of the bases of
  seed 1, the splits wrote biomd-123-p0's 2.8 times as fast as the
  passes up to 64 limbs with num-bigint's `to_str_radix(10)` past them,
  mq-8-p0's 2.7 times, and anemoi-l1R2a17-p0's 3.8 times. At 16 limbs the
  passes took 385 ns against `to_str_radix`'s 679 ns. The passes up to 2,
  8, 12, 16, and 24 limbs took 0.549, 0.517, 0.511, 0.511, and 0.513 s
  over nine instances whose integers lie near the bounds. Leaves of 2, 4,
  and 8 chunks took 0.758, 0.693, and 0.669 s over six instances, and in
  a second sweep leaves of 8, 16, and 32 chunks 0.664, 0.702, and
  0.864 s. In step 7's measuring run a call over ℚ took 0.941 of the
  time in the geomean over the tuning set's 44 instances over ℚ,
  jacsurf-symmetroid8-aff 0.775, mqpd-8 0.788, and biomd-123 0.812, and
  the 58 over GF(65521), whose path the entry does not touch, 0.977. Step
  9's measuring run took 0.941 over ℚ and 0.981 over GF(65521). Over
  GF(65521) the build moved the time: the compiler emitted
  `Table::row_products` otherwise (log 0086-3). Each
  monomial over ℚ is written from its exponents at each term.

### Over ℚ

- **Learn once, replay many** (q-lift.4–.11, .27, .28). The lift learns at
  one prime and replays at the others. Each prime is the
  least prime at or above a random draw from the seed. A replay took 0.2
  to 20 ms where a learn run took up to 1.08 s.
- **The generators parsed once** (run-generators-time.1–.4, .9). A call
  parses the generators at its first run's read and keeps them in the
  header (.1, .4): per term its sign, its numeric factors, and the number
  of its monomial in a `Dense32` table of the text's distinct monomials,
  numbered in the order the text first writes them (.9). Each run maps
  that form to its field and its table and interns each monomial at its
  first term. A literal of at most nineteen digits is kept as its value,
  whose residue is one remainder per lane, and a longer one as its
  digits (.2). The map takes the terms in the text's order, so each
  generator and each error is that of a read at the run's prime. A text
  that breaks the grammar keeps the terms before the break, so that a
  denominator 0 mod p before the break stays the error (.3), which the
  lift reads (_A denominator 0_ below). Each term's exponents kept and
  interned at every run would have taken 16.7 MB beside
  minrank-sm10r3k6-p0's 2.6 MB text, whose 154,502 terms over 27
  variables hold 2,506 monomials. Outside the measuring condition, a
  run's map took 0.15 to 0.38 of a read of the text, and the first run's
  parse and map 0.95 to 1.43, the most on texts of 48 and 55 variables
  whose monomials are mostly distinct (log 0087-2). No source in `lit/`
  reads an instance once for several primes.
- **Wide primes for the replays** (q-modulus-time.1, .2, .8, .9). A
  replay's prime lies in [2^61, 2^62) and its run in `Gf64`, or in
  `Gf64Pair` beside a second prime (_Two primes per replay_ below). It adds
  twice the bits of a prime below 2^31 to the modulus, so a lift
  combines about half as many images, and the appends, the conversions,
  the attempts, and the checks fall with them. A run at a prime past
  2^62 took 1.01 to 1.70 times its time at 2^31 − 1 (log 0036-1), and
  pearce-2026 §3, Table 2, finds 1.00 to 1.77 at 2^63 − 25.
  The learn runs and the verifications keep their primes below 2^31 in
  `Gf31Pair` and `Gf31`: their images are not combined. A replay's field is built for
  the prime its draw tested, without a second test. A prime the lift is
  given below 2^31 replays in `Gf31`. Against replays below 2^31, the
  calls over ℚ took 0.833 of the time in the geomean over the tuning
  set's 52 instances over ℚ, and executed 0.821 of the instructions over
  the 49 they solve. mq-8 took 16.1 s against 25.1 s, biomd-123 18.3 s
  against 27.6 s, and chandra-11 3.05 s against 4.27 s. cyclic-5, whose
  lift combines too few primes to halve, took 1.019 of the time. The
  peak rose by 1.7 % in the geomean, cyclic-8's by 15.3 MiB to
  120.8 MiB: a replay's coefficients take 8 bytes.
- **Two primes per replay** (q-later-runs-time.1–.6, .11). A replay that
  tracks no cofactors, of a record FGLM did not change, runs over two
  primes of [2^61, 2^62) at once, in `Gf64Pair`. Its coefficient is a pair
  of residues, one per prime, and every operation runs on each residue as
  `Gf64` runs it. The two primes share the replay's rows, columns, and
  pivots and the scan of its dense row, and a row operation does both
  primes' arithmetic in one pass over the row (graebe-1993 §4 and
  dekleine-2001 §1, via demin-2026 §4.2; demin-2024 §4.4).
  The run reads each term's exponents once and hands
  the lift one image of both primes (_The lift once a pair_ below).
  The lift takes the second prime at its next turn, after the
  budget's check, so it combines the primes one at a time and stops at
  the prime that completes it. A pair fits only where every leading
  coefficient the replay inverts, of a pivot of the generators' echelon
  form or, where the record took none, of a generator, and every counted
  row's leading entry is nonzero at both primes. Where it does not, or where a denominator vanishes at
  either prime, each prime replays alone and meets its misfit or skip
  there. A replay that tracks cofactors and the replay of a record FGLM
  changed run one prime at a time. Outside the measuring condition, over
  the 49 instances over ℚ that finish within 60 s, the pairs took 0.971 of
  the time in the geomean, and 0.952 over the 23 whose lift combines 14
  primes or more. Four primes per replay took 0.992 of the time of two
  over ℚ, and cyclic-8 peaked at 151.5 MiB against 119.4 MiB: a dense row
  takes 8 bytes a column per prime. In the measuring condition the pairs
  and the verification's run with no learner (_One more prime ends the
  lift_ below) together took 0.934 of the time in the geomean over the 49
  instances over ℚ that finish within 30 s. eco-11 took 1.337 s against
  1.639 s, and reimer-7 2.688 s against 3.101 s, its peak 124.4 MiB
  against 99.3 MiB.
- **Unlucky primes** (q-lift.19–.21; lift-misfit-loop.1–.5). The lift skips a prime that
  divides a denominator, and drops one whose replay misfits. Two misfits in
  a row drop the record, and the next prime learns afresh
  (boehm-2015 §3, pauer-1992).
  A verification whose leading monomials or an element differ drops the
  record too (_One more prime ends the lift_ below). The lift drops at most
  `DROPS`, 4, records in a call: the drop that reaches it returns
  `Error::Drops`, which counts the drops by why. Every drop counts,
  whatever its why, and the count never resets. A skip and a pair learn
  that falls back drop no record. Without the bound a defect that makes
  every replay misfit learns afresh without end: a stub of iteration 0068
  wrote 1.3 million lines of the call report in 3 s, and the disk ran full
  (log 0068-1). The tuning set's 44 instances over ℚ under 23 seeds,
  1,012 calls, dropped no record (log 0069-1). The few bad primes of an
  instance are not met where the primes are chosen at random (boehm-2015
  §3), and traverso-1989 bounds the probability that random primes are
  equally bad. A bound of 2 or 3 would end a defective call one or two
  learn runs sooner. A drop is an error rather than a `Stop`, which names
  a bound the caller set.
- **A denominator 0** (q-zero-denominator.1–.4). A denominator 0 is 0
  mod every prime, so no prime is good. Where a learn run's read
  stops with `NoInverse`, the lift scans the text for its first
  denominator 0. Where that stands on the line the read names, the call
  returns the error. Any other line names a prime that divides a nonzero
  denominator: a pair learn falls back to the learn run at its first
  prime alone (q-learn-time.6), and a learn run at one prime skips its
  prime. The error is the one a
  denominator 0 mod p is over GF(p), rather than a variant of its own,
  and the checker's reader refuses the text alike. A scan before the
  first prime would put the denominator 0 ahead of an earlier error and
  cost every call over ℚ. The supply of primes stays unbounded: a text
  whose nonzero denominators every prime of [2^30, 2^31) divides holds
  over 450 million digits of them.
- **One more prime ends the lift** (q-lift.22, .25; q-later-runs-time.7,
  .8). Once every element is
  reconstructed, the basis of a full run at one more prime must agree: the
  learn run's at its second prime, or a run of its own at a fresh prime
  where the learn run fell back or its second prime divides a denominator
  of the lifted basis (_The verification beside the learn run_ below). The returned
  basis is correct with high probability. A proof over ℚ is the checker's
  (idrees-2011 §1–2, demin-2024 §4.3).
  A run of its own records nothing: it runs as a call over GF(q) runs, apart from
  the record, and its plain run combines rows. Where its leading
  monomials differ from the record's, the lift drops the record and what
  it lifted, and the next prime learns afresh (parisse-2013 §4). Where
  only coefficients differ, the lift drops the record and what it lifted
  too, and the next prime learns afresh (result-time.16): no prime checked
  a kept element since its keep, so its residues cannot be rebuilt.
  Outside the measuring condition, against a learn run at q, the calls
  over ℚ took 0.969 of the time in the geomean over 49 instances, and
  eco-11 took 0.814.
- **The verification beside the learn run** (q-learn-time.1–.15). Every
  learn of the lift starts over two primes of [2^30, 2^31) at once, in
  `Gf31Pair`: the learn prime and the verification's, drawn right after
  it. Its coefficient is a pair of residues, and every operation runs on
  each as `Gf31` runs it past 2^16. The two primes share the run's pairs,
  symbolic preprocessing, matrices, and handover, and a row operation
  does both primes' arithmetic in one pass over the row, four entries per
  turn (graebe-1993 §4, dekleine-2001 §1 and §3). The record is the first
  prime's. The basis at the second is kept beside the lift until every
  element is reconstructed, and verifies the lifted basis in place of a
  run of its own. Where a leading coefficient the run inverts, of a pivot
  of the generators' echelon form, of an input, a row a reduction keeps,
  or a vector that joins FGLM's echelon form, is 0 at one prime alone,
  the run halts (dekleine-2001 §3). The
  lift then learns at the first prime alone, and the verification runs
  at a fresh prime, as where a denominator of the input is 0 at either
  prime. The run at one prime and the verification's run of its own built
  the same pairs and matrices twice: they took 0.198 and 0.180 of a call
  over ℚ in the mean over 40 instances, callgrind, seed 1. In the
  measuring condition the calls over ℚ took 0.877 of the time in the
  geomean over the 49 that finish within 30 s, and the tuning set 0.946
  over its 116 instances. biomd-365 took 0.0605 s against 0.1019 s, and
  biomd-333 0.618 s against 0.868 s. reimer-7 and cyclic-8, whose learn
  runs hand over, took 1.016 and 1.015 of their time: their plain run
  reduced each row on its own at both primes, where the verification's
  run of its own combined rows. The second prime's basis, held through
  the replays, took the peaks over ℚ up by 5.7 % in the geomean,
  katsura-11's to 235.8 MiB from 215.5 MiB. Outside the measuring
  condition, one entry per turn took biomd-333 0.52 s against 0.40 s
  for four.
- **Reconstruction over the chain** (q-prime-count-time.1–.8;
  q-rational-time.4–.10; q-reconstruct-time.1, .2, .4–.6, .12, .13;
  q-lift.14; q-attempt-time.1–.3). The coefficients of an element share their denominators. A
  part keeps its rationals as numerators over a chain of denominators,
  each the lcm of the one before and the denominator of a rational an
  attempt found. An attempt scales a coefficient's images by l's, l the
  part's chain's last, and converts them to v = u·l mod M, u the
  coefficient's residue modulo M, taken in (−M/2, M/2] (_Images and their
  conversion_ below). Where
  |v|·2^(T+1) < M, the rational is v/l, with no Euclidean step
  (traverso-1989 §Reconstruction algorithms). Elsewhere the maximal
  quotient rational reconstruction, MQRR, takes from v mod M, in [1, M),
  the rational n′/f before the first largest quotient of the Euclidean
  algorithm on (M, v mod M),
  where that quotient is at least 2^T, and the coefficient is n′/(l·f)
  (monagan-2004 §3). The chain's next is l·f, over which the coefficient's
  numerator is n′, with no gcd: for the coefficient a/d in lowest terms and
  h = gcd(l, d), n′ = a·l/h and f = d/h, so lcm(l, d) = l·f
  (traverso-1989 §Reconstruction algorithms). l·f's residues at M's primes
  are l's, kept, times f's. Where p divides l·f, n′/(l·f) is brought to
  lowest terms, and the chain gains the lcm of l and its denominator.
  Over l the rational left has a small denominator, and
  MQRR asks for about |n′|·f·2^T < M, where Wang's bound, N = D =
  ⌊√(M/2)⌋, asks for 2·max(|n|, d)² < M of each coefficient. T is
  ⌈log₂ L(M)⌉ + `MARGIN`, L the bit length and `MARGIN` 10: the median
  largest quotient's log₂ grows as log₂ log₂ M, and past it each bit
  halves the count of larger ones (monagan-2004 §3 and §3.1). MQRR runs
  in two passes, both in Lehmer's form on 128-bit leading parts
  (collins-1995 §4.1): the remainders alone to the step of the largest
  quotient, ended where the remainder falls below 2^T or to the largest
  quotient found, and then the remainders and the cosequence to that
  step. An attempt first ends the first pass at the first quotient of at
  least 2^T, in the order of the steps, and takes the rational before it
  where its terms are coprime and it fits at p: a rational small against
  M is followed by a quotient past M/(3·|t|·r) (monagan-2004 §3, Lemma 1).
  MQRR runs to its largest quotient only where that rational does not
  fit, and where no quotient reaches 2^T, the attempt fails after the one
  pass. A rational whose numerator and denominator are not coprime is
  refused (collins-1995 §2). At a part's first coefficient, at every
  other attempt, l is the previous element's
  chain's last of the part: an element's denominator is often a multiple
  of the one before. l is a unit modulo M, which its residue at each
  prime of M says (collins-1995 §4.3), and where the part's last is not,
  l is 1. Those residues are kept and scale the images.
  The checks at a prime take each chain denominator's residue
  once, and a number with more limbs than the step's powers by Horner's
  rule. The basis text's integers are the numerators times the chain's
  last over their own, with no lcm to compute.

  In the bases' texts, the largest integer of an element, which bounds
  what a known common denominator asks for, has 0.500 of the bits of M
  that Wang's bound asks for on mq-7-p0 and mqpd-8-p0, and MQRR over the
  chain asks for 0.50 to 0.55 of them on eleven of the 28 instances over
  ℚ that combine more than three primes (log 0076-1). At the machine's
  own clock, three seeds per instance over the 44 instances over ℚ, the
  58 over GF(65521) at 0 in the mean over the 102, MQRR over the chain
  alone took time mean d −0.0950, and with the guess at every other prime
  −0.1119, both at `MARGIN` 10; `MARGIN` 6 took −0.1126 and 16 −0.1081.
  At a fixed T of 32 bits, MQRR alone took −0.0885, with the guess where
  1 fails −0.1007, and with the guess at every other prime −0.1026. In
  the measuring condition the tuning set took 0.8979 of the time in the
  geomean over its 102 instances, mean d −0.259 over the 44 over ℚ:
  biomd-123-p0 5.62 s against 13.39 s, mq-8-p0 6.20 s against 12.56 s,
  jacsurf-symmetroid8-aff-p0 1.15 s against 2.34 s, phc-fourbar-p0
  0.789 s against 1.496 s, and mq-7-p0 0.399 s against 0.739 s. The
  peaks over ℚ fell, noon-9-p0's to 649.1 MiB from 666.4 MiB and
  katsura-11-p0's to 180.1 MiB from 194.8 MiB, and biomd-123-p0's rose
  to 382.4 MiB from 374.1 MiB.

  The chain grown by l·f, MQRR's first quotient of at least 2^T tried
  first, and the conversion ended at its first digit 0 or its prime less 1
  below the top (_Images and their conversion_ below) together took, in
  the measuring condition, 0.9620 of the time in the geomean over the
  tuning set's 102 instances: mean d −0.0869 over the 44 over ℚ, at
  −0.0706 of their instructions, and −0.0021 over the 58 over GF(65521).
  biomdcl-123-p0 took 0.067 s against 0.127 s, biomdcl-315-p0 0.92 s
  against 1.57 s, phc-fourbar-p0 0.38 s against 0.55 s, and phc-i1-p0
  0.126 s against 0.178 s (log 0085-2). At the machine's own clock over
  the 44 over ℚ, each against the build before it, the chain's next as
  l·f with no gcd took mean d −0.036, MQRR's first large quotient −0.012,
  the ended conversion −0.017, and l·f's residues from l's −0.021 (log
  0085-1). The peaks over ℚ rose by 0.0054 in the plain mean ln, and by
  0.0091 in another run of the same code. powerflow-ring6c0v1s1-p0's
  stood at 39.9 MiB in one run and at 48.4 MiB in the other, against
  46.9 MiB, and chandra-11-p0's at 137.4 MiB and at 125.0 MiB, against
  125.3 MiB, each level held by all three trials of its run (log 0085-3).

  Before, the lift ran Wang's reconstruction with N = D = ⌊√(M/2)⌋
  (wang-1981 §4; q-lift.13, withdrawn), and first Wang's rational over
  the chain's last, where v/L in lowest terms lay within Wang's bound
  (q-rational-time.1–.3, .11, withdrawn). With the plain Euclid,
  reconstruction took 21 to 86 % of a call over ℚ on 11 of 13 instances
  profiled, and the Lehmer form took 1/6 to 1/74 of its time from 4 to
  750 primes, and 1/1.5 at one prime. Over the chain's last, Wang's
  reconstruction ran on 0.1% to 2.4% of the coefficients tried on
  sparse-10 to -16, 10% to 27% on katsura-9 to -11 and chandra-9 to -11,
  and 74% on biomd-333. In the measuring condition mqpd-7 then took
  0.296 s against 0.515 s, mq-7 1.454 s against 2.134 s, katsura-11
  6.90 s against 8.65 s, and chandra-11 4.50 s against 5.65 s, mq-8 and
  biomd-123 finished within 30 s, in 26.1 s and 28.2 s, and the tuning set
  of then took 0.916 of the time in the geomean. MQRR on each
  coefficient alone, in lowest terms, asks for 0.99 or more of Wang's
  bits on mq-5 to -7, mqpd-5 to -8, cyclic-8, katsura-10 and -11, and
  biomd-103, whose numerators and denominators have one size (log
  0048-1), and demin-2024 §4.3 found MQRR not effective on Groebner.jl's
  benchmarks. The test code's lift runs Wang's reconstruction on every
  coefficient (q-reconstruct-time.3, .7), and the lift over the chain
  returns its outcomes.
- **Element by element, incrementally** (q-lift.15, .16, .18;
  q-reconstruct-time.8–.10). Elements are reconstructed in increasing
  order of their leading monomials, and an attempt resumes at the
  coefficient where the last one stopped. An element's rationals are
  checked at the prime after the modulus they were reconstructed from,
  and once it is kept, at no further prime but the verification's (_One
  more prime ends the lift_ above). With the Lehmer form, the two
  took mq-6 from 3.50 s to 0.60 s, and mq-7 finishes in 10.3 s.
- **Kept elements leave the combination** (q-crt-time.1, .9;
  result-time.13–.15). Once an element is kept, the lift neither combines
  nor checks it, reads none of its images, and drops those it holds. An
  image that gains or lacks a term of a kept element changes nothing.
  The verification, at the learn run's second prime or at a fresh one, is
  a kept element's one check after its keep (_One more prime ends the lift_ above). Combining every
  element at every prime, 74 to 85 % of the combination on mq-7, mq-8,
  mqpd-8, and biomd-123 went to elements already kept. With result-time's writers
  of _The result_, in the measuring condition, a call over ℚ took 0.955
  of the time in the geomean over the 49 instances that finish, and
  0.929 over the 10 past 1 s. mq-8 took 12.67 s against 15.66 s and
  biomd-123 14.98 s against 17.25 s. The peaks fell by 12 % in the
  geomean over the 49: biomd-123's 107.4 MiB lower, at 436.8 MiB, and
  mq-8's 67.9 MiB lower.
- **Images and their conversion** (q-lift.9; q-lift-growth-time.1–.12;
  q-crt-time.4–.6, .8; cofactor-stop-latency.7, .8; q-attempt-time.4–.6). Each open coefficient
  keeps its image at every prime combined, one word a prime, and no
  residue modulo M is formed before an attempt asks for one. A part's
  images lie in one vector, term by term, each term on a capacity of words
  that doubles once the primes fill it: the vector is resized in place and
  the terms move from the last to the first. An append thus writes one
  word a coefficient, a doubling moves each word once, and a stopped call
  drops a block per part. Each prime's Chinese remainder step makes the
  residues of the products of the primes before it at that prime, and the
  lifting keeps them as a triangle, k(k + 1)/2 words over k primes, with
  each prime's inverse of the product before it. An attempt multiplies a
  coefficient's images by l's residues at the same primes, by Shoup's
  product with a factor made once per prime and chain denominator
  (vanderhoeven-2016 §2.2.2), so that they are the images of v = u·l mod M.
  The conversion takes v's mixed-radix digits, each from v's image at its
  prime and the digits before it, and then v on limbs by Horner's rule
  (garner-1959, _Conversion from a residue code_).
  Where the top digit is 0, the rule starts at the highest digit that is
  not 0, and where it is the last prime less 1, the rule runs over the
  digits of M − 1 − v. A numerator over the chain's last, of about half
  M's bits, thus reads only its own digits. An attempt's conversion ends
  at the first digit below the top that is 0 or its prime less 1, and the
  rule reads the digits before it as it reads them below a top digit 0 or
  the last prime less 1: a value below the product of the primes before
  that digit has every digit from there on 0, and one within it of M every
  digit its prime less 1. A numerator over the chain's last thus computes
  only its own digits. Any other value meets such a digit with a
  probability of about 2/p a digit, and its ended conversion gives a wrong
  value. Such a digit stays as primes join, so where the value of an ended
  conversion does not fit at p, the attempt converts the images in full
  and takes the coefficient again. MQRR's bounds lie on limbs, and
  v is compared with them there. The attempts' buffers stay from attempt
  to attempt (monagan-2007 §2.2). Over k primes the digits take k(k − 1)/2
  word products and the rule at most as many. No source in `lit/` keeps
  the images and converts them at an attempt (log 0083-1), or ends a
  conversion at a digit (log 0085-1), and Garner's
  paper is requested.

  In the measuring condition the tuning set took 0.9615 of the time in the
  geomean over its 102 instances: mean d −0.0994 over the 44 over ℚ, and
  +0.0063 over the 58 over GF(65521), at −0.0005 of their instructions.
  phc-fourbar-p0
  took 0.551 s against 0.790 s, mqpd-8-p0 0.927 s against 1.326 s,
  biomd-123-p0 4.00 s against 5.66 s, and mq-8-p0 4.47 s against 6.20 s.
  The peaks over ℚ rose by 0.002 in the plain mean ln, and fell by 0.002 in
  another run of the same code. phc-stewgou40-p0's rose to 9.3 and 9.4 MiB
  from 5.8 MiB. chandra-11-p0's stood at 137.4 MiB in one run and at
  125.1 MiB in the other, against 129.4 MiB, each level held by all three
  trials of its run. Outside the measuring condition powerflow-ring6c0v1s1-p0's
  rose by glibc's heap, left untrimmed after the lift's blocks are freed
  (log 0083-1). mq-8-p0's fell to 106.3 MiB from 120.0 MiB. At the
  machine's own clock, the digits' sums cut at a run of top digits 0 or
  p − 1 by a telescoping sum took mean d −0.0008 against this conversion
  over the 44 over ℚ, within the noise. A capacity grown by half, a vector
  per prime, and a fresh vector at each doubling peaked within 2 MiB of
  each other (log 0083-1).

  Before, each open coefficient kept its residue modulo M on M's limbs,
  and a Chinese remainder step at every prime formed x + M·t in one pass
  over them (collins-1995 §4.2; q-crt-time.7, withdrawn), the vector
  resized to the limbs of M·p a residue. The scaling was num-bigint's
  product u·l and its remainder by M (q-rational-time.7, withdrawn). A
  vector per residue then left chandra-11-p0 0.243 s in the lift's drop
  after its stopping check. A new vector per part and prime raised
  sparse-14-p0's minor faults from about 12,000 to 22,000 to 25,000, and
  the resized vector to about 14,500 to 15,200 (release builds on an
  i7-14700K, outside the measuring condition). A number's residue modulo p
  is the sum of its limbs times the powers 2^(64i) mod p, accumulated in
  `u128` and reduced once (dumas-2008 §2.1.2). Each addition that wraps
  the sum counts one 2^128, whose residue is taken once per prime
  (q-modulus-time.6, .7), and a mixed-radix digit's sum is taken alike.
  The check of a rational n/d against c compares n with c·d modulo p, with
  no inverse. With the entry above in its first form, which checked kept
  elements at every prime, the residues on limbs took, one call each,
  release build, outside the measuring condition: mq-7 1.33 s against
  6.50 s, mqpd-8 4.88 s against 19.7 s, and biomd-123 25.5 s against
  86.5 s, and mq-8 and biomd-200 finished within 120 s, in 25.9 s and
  74.5 s.
- **The lift once a pair, at the primes a coefficient needs**
  (q-lift-rest-time.1–.10). A pair replay hands the lift each element's
  terms once, with their coefficients at both primes beside them, a
  coefficient 0 where a residue is 0 (dekleine-2001 §2). The lift merges
  the terms into the parts once, the second prime's coefficients taking
  the merged terms beside the first's, and appends the first prime's
  without an attempt: the second prime's image exists already, so an
  attempt there spares no replay (dekleine-2001 §3.2, demin-2024 §4.3).
  The attempts run at the second prime, from M times the first, at the
  lift's next turn. The guess at a part's first coefficient is taken at
  the lift's attempts of even index (_Reconstruction over the chain_
  above), which alternates it by the pair. A pair replay, which attempts
  where M holds an odd number of primes, would otherwise take the guess
  at every attempt.
  MQRR's first pass ends at a step its Lehmer runs do not give, without
  the step's division, where the remainders' bit lengths differ by more
  than T: the step's quotient is at least 2^(L(A₁)−L(A₂)−1), and the
  pass reads of it only whether it reaches 2^T.
  Once a coefficient of an element fitted in an attempt, a further
  coefficient of a part that holds a rational is converted from its
  images at the first j primes of M and reconstructed modulo their
  product. j is the fewest multiple of `PREFIX`, 4, whose primes' bits,
  bounded below by Σ (L(pᵢ) − 1), reach the part's need and `SLACK`, 128
  bits, where j + 4 is at most M's primes. A part's need is the bits its
  second coefficient asked for, L(n′) + L(f) + T + 2 for its numerator n′
  over the chain's last and the factor f the chain grew by, then the
  largest over its later coefficients: by monagan-2004 §3, Lemma 1, a
  modulus of these bits finds n′/f. The first coefficient reconstructs
  the element's denominator and asks for about twice. Where the
  coefficient does not fit at the prefix, it is taken modulo M, and a
  denominator found modulo the prefix is taken only where it is a unit at
  M's other primes. The lifting keeps each prefix's MQRR.
  An element kept behind another holds integers of about half M's bits,
  and the conversion and the Euclidean algorithm cost the square of the
  primes they read. No source in `lit/` reconstructs modulo part of the
  primes combined, estimates the modulus from the coefficients before,
  or takes a quotient's size from the remainders' lengths (log 0088-1).

  With the generators parsed once and a replay's echelon from its record
  (_A replay's echelon from its record_ above), in the measuring condition, the
  tuning set took 0.9779 of the time in the geomean over its 102
  instances: mean d −0.0599 over the 44 over ℚ, at −0.0490 of their
  instructions, and +0.0061 over the 58 over GF(65521), at +0.0009 of
  theirs. biomdcl-315-p0 took 0.627 s against 0.804 s, phc-stewgou40-p0
  0.217 s against 0.262 s, phc-fourbar-p0 0.262 s against 0.316 s, and
  phc-i1-p0 0.102 s against 0.118 s; critmin-n9f2x2g2-p0 1.17 s against
  1.82 s and minrank-sm10r3k6-p0 0.241 s against 0.375 s, by the
  generators and the echelon (log 0088-2). The peaks over ℚ fell by
  0.0118 in the plain mean ln, noon-9-p0's to 550.6 MiB from 641.7 MiB,
  and minrank-sm10r3k6-p0's rose to 32.9 MiB from 23.1 MiB by its parsed
  form. Step 9's measuring run took 0.9784 of the time, 0.9428 over ℚ,
  and its peaks over ℚ fell by 0.0204 in the plain mean ln (log 0088-3).
  At the machine's own clock, three seeds over the 44 over ℚ, the
  58 over GF(65521) at 0 in the mean over the 102: the attempt once a
  pair took −0.0046, MQRR's first pass without the division −0.0024, the
  prefix −0.0041, the three −0.0081, and with the pair's image on one
  list of terms −0.0096 (log 0088-1). The guess and then none at every
  attempt once a pair took +0.0020 against the alternation (log 0088-2).
  With it a lift in pairs combines at most one prime more than a lift of
  replays alone, and with the alternation up to three: the guess runs at
  every fourth prime where alone it runs at every second.
- **Cofactors over ℚ** (q-lift.12, .17). The learn runs and the
  verification's run track no cofactors, so their signature run may hand
  over. The replays track them where
  the call asks for cofactors or the checker, and a cofactor row is lifted
  with its element.

### Checker

- **Independent and streaming** (checker-gfp.1–.3, .17). The checker is
  its own crate and reads texts. It holds G and one row of the cofactor
  matrix at a time. Cofactor texts reach hundreds of megabytes.
- **What it checks** (checker-gfp.9–.11, .14, .16; checker-orders.13). In
  turn: the generators reduce to zero, the rows of the cofactor matrix
  hold, and every S-pair the product and chain criteria do not excuse
  reduces to zero (buchberger-1979, gebauer-1988 §4.3). Over GF(65521) the
  S-pairs took 48 to 93 % of the check (log 0001-2). Over ℚ they took
  under half on 7 of 9 instances, and reading the texts took most of
  mq-6's and mqpd-7's check (log 0002-2).
- **Plain arithmetic** (checker-gfp.4, .7, .8, .12, .13, .15). `u64`
  residues with `u128` products for every p, a geobucket, and divisor
  masks.
- **The instance read as the engine reads it**
  (instance-grammar-agreement.1–.6). The checker takes the engine's
  reading, and the engine's reader, which every call runs, stays as it is.
  Whitespace is the form feed too, in all three texts. A characteristic
  that is neither 0 nor a prime below 2^64 is `Characteristic` as the
  instance is read: a strong probable prime test to each of the first
  twelve prime bases, after trial division by them, decides every `u64`,
  since the least composite that passes all twelve lies past 2^64
  (sorenson-2017 §1, miller-1976). An instance term past degree 2^31 − 1,
  `Dense32`'s bound, is `Overflow`, and the basis and the cofactors keep
  `u32`'s. The bound is tested once the term is read, so a later factor's
  unknown variable or syntax error is the refusal, as it is the engine's.
  A differential run over 100,000 mutated texts found 346 texts with a
  form feed the engine read and the checker refused, 1,330 with a
  characteristic of 1 or a composite the checker read, and 27 with a term
  of degree from 2^31 to 2^32 − 1 the checker read (log 0050-1).
- **Exact over ℚ** (checker-q.1–.11, .18). The checker reduces over the
  integers. A check modulo primes could pass a set that is a Gröbner basis
  mod p but not over ℚ, such as {x², xy + 65521y²}.
- **The cheap mode** (checker-cheap.3–.10, .13–.15). Only the rows stage
  is evaluated at points; the rest stays exact. Over ℚ each point has a
  prime of its own. It took 12 to 90 % of the exact rows stage over
  GF(65521) and 4 to 27 % over ℚ.
- **The cheap mode's batches** (checker-points-abort.1–.5, .7). The points
  are drawn in batches that fit 2^24 bytes, and at least one point, a point counted with its
  coordinates, its power table, the values of F and G at it, a row's values at it, and a
  coefficient's residue. Drawing
  every point before the rows are read, the alternative, reserved room for
  all k: at 2^32 − 1 points on cyclic-5 that was 481 GB, and the
  allocator's failure aborted the process. A larger batch raises the
  memory for no gain in throughput (demin-2026 §4.3). A batch held 484
  points on chandra-13-p65521, the fewest of the tuning set of iteration
  0047, and 6,418 in the median, so every k up to 484 was one batch there
  (log 0047-2). The result is the
  earliest flaw or reading error in row order over all k points, where each batch's first
  flaw could name a later row than one batch of all the points. Over ℚ a
  row is checked exactly where its degree bound reaches `LEAST_MODULUS`, 2^31, below every
  prime a point can have, so no pass over all the points finds the least
  prime first. The poll is asked as the terms are evaluated, a term at a
  point counting as a unit of work. Asked once per row alone, it let
  the test profile's call at 2^32 − 1 points on cyclic-5 over ℚ with a
  deadline of 300 ms stop after 62 s: the release CLI's call took 2.27 s at 1,000
  points, and a batch there holds 22,075.

### Memory

- **Small blocks from mimalloc** (small-block-time.1–.7). `Allocator`
  serves a request of at most `SMALL`, 1024 bytes, from mimalloc, and a
  larger one from glibc's malloc. num-bigint's integers of up to 128
  limbs are among the small blocks. The large blocks, a matrix's columns
  among them, stay glibc's, as _The columns in a reservation past 1 MiB_
  above has them. A block is freed and resized by the allocator its
  layout's size names, with no lookup of the pointer, and a resize across
  `SMALL` moves the block to the other allocator. The crate exports the
  allocator and does not set it, since a program has one global
  allocator: the Python module, the C library, and the CLI each set it.
  Before a memory stop, `mi_collect` gives mimalloc's free memory back
  beside glibc's `malloc_trim`. mimalloc is the crate `mimalloc` 0.1.52
  with its default features, mimalloc 3; its features `v2`,
  `local_dynamic_tls`, and `no_thp` measured within the noise of it.
  Outside the measuring condition, mimalloc for every request lost on the
  calls below 1 ms over GF(65521), sparse-8 +0.44 in the ln of the time,
  and raised the large calls' peaks; a bound of 256 bytes gained less
  than one of 1024, and a bound of 4096 raised the peaks. In the
  measuring condition a call took 0.972 of the time in the geomean over
  the tuning set's 116 instances. Over the 112 that both builds solve,
  it took 0.989 over the 63 over GF(65521), 0.949 over the 49 over ℚ,
  and 0.951 over the 35 below 10 ms, and its peak memory took 0.954.
- **mimalloc on Linux alone** (port.26). mimalloc serves the small blocks
  on Linux, where small-block-time was measured. On Windows and macOS the
  system's allocator serves every request, and `mi_collect` is not
  called. mimalloc 3.3.2 failed with an access violation as it reserved
  its first arena, before `main`, in a Windows 11 guest under KVM.
- **The heap padded at the first call** (page-fault-time.1–.9). A call's
  first writes to memory new to the process fault page by page, and the
  kernel's time for the faults is part of the call's. The first call of a
  process, at its entry, sets glibc's `M_TOP_PAD` to `TOP_PAD`, 16 MiB,
  and `M_MMAP_THRESHOLD` to `MMAP_THRESHOLD`, 2 MiB, by `mallopt`, once,
  kept by a `OnceLock`. The main heap then grows by at least 16 MiB a
  step and keeps 16 MiB free at its top when it is trimmed. Where the
  kernel runs transparent huge pages `always`, it backs the heap's
  aligned 2 MiB extents with huge pages at their first write, one fault
  where there were 512. Setting the pad disables glibc's dynamic mmap
  threshold, which would then stay at 128 KiB (mallopt-3): every block of
  128 KiB or more that the heap's free space cannot hold would take a
  mapping, whose pages the kernel zeroes and faults afresh. With the
  threshold at 2 MiB a block below a huge page comes from the heap, and a
  larger one the heap cannot hold takes a mapping of its own, which goes
  back to the system at its free. glibc pads the main arena's growth
  alone: a thread's arena grows by steps of its own, and the threshold
  holds in every arena. Where the C library is not glibc, a call sets
  nothing. Rejected: the parameters through `GLIBC_TUNABLES` or
  `MALLOC_TOP_PAD_`, which take effect only where they are defined before
  the process's first allocation (mallopt-3), and a library loads after
  it; set at every call, which would undo a caller's own setting; set at
  the load of the Python module, the C library, or the CLI, which a
  program that links the crate would miss; a refusal ignored or a panic;
  a pad of 4, 32, or 64 MiB; a threshold of 8 or 32 MiB, or left as the
  pad leaves it; and the blocks of 4 MiB or more in 2 MiB aligned
  mappings of the crate's own with `MADV_HUGEPAGE`, which leaves the
  smaller blocks' faults as they were. No keyword, switch, or field turns
  the setting off. Outside the measuring condition, the CLI with the
  setting before the call, one call each over 101 instances of the
  tuning set, eight at a time, took the time −0.0229 in the mean ln
  against the base, the peaks +0.0008 in the mild form of the memory
  test with six instances past their allowance, and the faults 0.249 of
  the base's. The pad alone took −0.0226,
  with faults 0.417 of the base's; with a threshold of 8 MiB −0.0280, the
  peaks +0.0039 and nine instances past their allowance; a pad of 64 MiB
  −0.0244 and −0.0116 in two runs, with 25 past it; and the mappings with
  `MADV_HUGEPAGE` +0.0007. In the measuring condition a call took 0.977
  of the time in the geomean over the tuning set's 102 instances, mean
  ln −0.0233: the cycles −0.0238, the user CPU time −0.0129, and the
  system's share of the CPU time 0.022 in the mean against 0.032. Its
  peak memory took +0.123 in the plain mean ln over 101, and −0.0011 in
  the mild form with 16 MiB allowed. Three instances over ℚ passed their
  allowance, anemoi-l1R2a17-p0 the most, 79.0 MiB against 51.7 MiB.

### CLI

- **Defaults** (cli.9, .10, .12; input-redundancy-time.10). `--order` defaults to grevlex, the order
  of the basis text's measurement, as `grobx.groebner`'s keyword does.
  `--threads` defaults to 1, as msolve's `-t` does, where 0 would
  allow all the machine has; the engine computes on one thread whatever it
  allows. No switch turns `signatures` or `echelon` off, as no keyword
  of `grobx.groebner` does, so `grobx` runs the engine of every measured
  call.
- **Exit status 3** (cli.14). A result the checker rejects exits 3, its
  basis and cofactors written, where exit 0 would leave the ruling to
  standard error alone. A rejected result is no error of the call, so it
  is not 2.
- **Files before the call** (cli.17). The output files are created before
  the call, where creating them after it would lose a long run to a path
  that cannot be written.
- **SIGINT** (cli.18–.21). Once the instance is read, SIGINT is blocked in
  every thread, and a thread of its own takes it by `sigwait` and raises
  the cancel flag. A signal handler, the alternative, may call
  async-signal-safe functions alone and cannot reach the flag. A second
  SIGINT restores the default action and raises SIGINT, so that the
  parent sees the process end by the signal, where `exit(130)` would hide
  it. An ignored SIGINT stays ignored, where watching it would take a
  SIGINT the parent meant to keep from a command run in the background.
  Before the instance is read, SIGINT keeps its default action, so that it
  ends a `grobx` that waits on its input, where a watch from the start of
  `main` would hold it for a call that has not begun.
  On Windows (port.7, .8) a console control handler takes Ctrl-C and
  raises the flag, and it passes a second Ctrl-C on to the system's
  handler, which ends the process.

### C interface

- **A struct the caller fills** (c-interface.1–.3, .19;
  input-redundancy-time.10). The order, the threads, and the seed are
  `grobx_params_new`'s arguments, as in `Params::new`, and
  `grobx_params_default` sets them as `Params::default` does. The rest
  are fields set by name, as keywords are. An
  opaque handle with a setter per parameter, the alternative, would keep
  the ABI stable across versions, and the header and the library ship
  together. An absent bound is `GROBX_NONE`, where
  a flag beside each value would let a value set without its flag pass
  unread. The checker is a mode and its points, so every count from 1 to
  2^32 − 1 can be asked. `signatures` and `echelon` are no fields, so a C
  call runs the engine of every measured call.
- **Integers checked at the boundary** (c-interface.4, .20). `checker` is a
  `uint32_t` the library checks, not a Rust enum, since a value outside an
  enum's variants is undefined behaviour and C lets any integer into the
  field. A text that is not UTF-8 is `GROBX_ARGUMENT` before the call.
- **One result, one free** (c-interface.5, .6, .9, .16, .17). The result is
  the caller's struct, and the call writes all of it whatever its status.
  The error's message is a field of it, where a thread-local last error
  would keep state from one call to the next. `grobx_result_free` frees
  every text and sets it NULL, so one free follows every call, and a
  second does no harm.
- **Status codes** (c-interface.7, .8, .18). A status other than
  `GROBX_OK` is where Python raises. A call a bound stopped is `GROBX_OK`
  with `stopped` set, as Python returns a result there, where a status per
  bound would read as the CLI's exit status 1. Each variant of `Error` a
  call reaches has a status of its own, and `GROBX_ERROR` stands for any
  other, since `Error` is non-exhaustive.
- **A panic caught** (c-interface.10, .24). The call runs inside
  `catch_unwind`, and a panic returns `GROBX_PANIC` with its message. A
  panic that leaves an `extern "C"` function aborts the process, and the
  library runs inside its caller's. The library sets no panic hook of its
  own: the hook is the process's, shared with the caller's own Rust code.

### The call report

- **Text, a line per event** (call-report.1, .8). The report is text,
  one line per event with its time and `key=value` fields (_Report text_
  above), where JSON Lines or a typed list of events in the outcome
  would need a parser to read. One text crosses the Rust, Python, C, and
  CLI interfaces unchanged. Times are seconds with six decimals, where
  integer nanoseconds would be read by a person less easily. A
  microsecond resolves a round's stage.
- **Carried by the budget** (call-report.2, .3, .7, .9, .10). The report
  travels inside the call's budget, which every stage takes already,
  where a parameter of its own would change about fifteen signatures.
  Each hook tests at run time whether the report is on, where a type
  parameter would compile every run twice. The hooks stand per round,
  per stage, per run, per prime, and over ℚ per element of the texts,
  never per row operation. A stage's time is
  the time between two reads of the monotonic clock, the clock the
  budget reads. `entry::call` puts the report into the outcome once, at
  the end. A call that returns an `Error` returns no report: `Error` is
  a plain enum compared by value, and most errors come from the text
  before any run.
- **Counts in every call** (call-report.4). The budget counts the entries
  the row operations subtract and those they add into combinations, in
  every call, beside the spends that are there already (_Units of work_
  below). Counts kept with the report on alone would test per row
  operation.
- **A round's stages** (call-report.5, .6, .22). A round's time is split
  into `select`, `symbolic`, `encode`, `reduce`, `backsub`, `decode`, and
  `update`, the parts of pearce-2026 §2.1. Its §2.2, Table 1, finds their
  shares differ by instance: symbolic preprocessing took 15.4% to 58.9%
  of F4's time and the reduction 0.8% to 62.0%, so one time per round
  would hide where it goes. The eighth stage, `record`, holds the trace's
  and the learn run's hooks, both of the trace's among them, work a call
  without a trace does not do, and the rows that counted listed and
  sorted for the trace, which every round does (f4-gfp.52).
- **Every decision with what decided it** (call-report.11, .12, .13,
  .19–.21; pair-work-handover.7). The `run` line names the engine and the reduction's mode,
  each with its reason, read from the functions the engine decides by,
  where a mode stated per round would repeat it. `engine-by` names
  `signatures-off` where the order is not graded either, the first
  condition the engine reads. Each lane's prime is read as the residue
  of −1 there plus one, so the field's interface stays as it is. Every
  round at which a handover test reads the round gets a `handover` line,
  a verdict to stay among them. The pair-work test's verdict and the S-pairs
  it read stand on that line, not on one of their own. The signature run's S-pair fates stand
  once, on its `ran` line, as the run keeps them. Every round `stats`
  counts gets its line, one a halt ends in its decode or update too: a
  search over random systems of three binomials found one whose 8-bit
  plain run overflows in the update after round 20, and its report gives
  that round's figures.
- **The lift's lines** (call-report.14, .15, .23, .24). The lift writes a
  line per prime whose image or skip it takes, in the order it used
  them, with what became of the
  prime, where `lift::Report`'s lists at the end lose the order: a
  misfit, a drop, and the learn after it. A combined prime's time is
  split into the merge, the attempts at reconstruction, and the Chinese
  remaindering, and its image is timed apart. monagan-2015 §2.6, Table 4,
  breaks the time over ℚ into the bases modulo p, the Chinese
  remaindering, and the reconstruction, and parisse-2019 §2 runs the lift
  as a loop of those three steps. The cofactor text over ℚ, which the lift writes element
  by element beside the basis text, is timed apart by marks between
  them, so the order of the work stays. The `lift` line's `replays`
  says what the lift does, `alone` where cofactors are tracked whatever
  the parameter asks, save that a record FGLM changed replays alone
  under `replays=pairs`.
- **The interfaces** (call-report.16–.18, .25). Python's `report` is a
  `str` made inside the call, as the basis is, where a list of parsed
  lines or a callback per line would fix a format beside the text. The
  C interface appends `report` to both structs, and `grobx_result_free`
  frees it with the other texts. The CLI writes the report to the file
  `--report` names, where standard error carries the checker's verdict
  and standard output the basis. The file is created before the call
  with the other output files (cli.17), so a long run is not lost to a
  path that cannot be written, and a call the crate refuses removes it
  where it is a regular file.
- **The cost with the report off.** Built in the release profile, one
  call each without the report under callgrind, the branch with the
  report executed 0.9947 to 1.0005 of the instructions of the build
  before it on six calls over GF(65521), katsura-9, cyclic-6, and eco-10
  with signatures on and off (log 0064-2). The lift's lines took 1.0000
  to 1.0003 of the instructions of the branch with the report over GF(p)
  alone on six calls
  over ℚ, cyclic-6, katsura-7, reimer-5, eco-8, noon-5, and katsura-7
  with cofactors and the cheap checker (log 0064-3). In the measuring
  condition the tuning set's 116 instances took 1.0024 of the time of
  the build before it in the geomean, and the 112 both solve executed
  0.9979 of its instructions (log 0064-4).

### Limits

- **One thread** (f4-gfp.30, q-lift.27). Every run computes on one thread,
  whatever `threads` allows.
- **The budget** (f4-gfp.27–.29, .50). A run ticks its budget once per row
  and checks its deadline, cancel flag, and memory ceiling in full every
  so many ticks, once a round's matrix, a text, or the work on cofactors or images has spent
  `WORK` units of work (_Units of work_, _Texts' units of work_, _Cofactors' units of work_
  below), and at the start of each round.
  It reads the resident set
  from `/proc/self/statm` on Linux, by `proc_pidinfo` on macOS, and as
  the working set on Windows (port.3, .4). Before a memory
  stop it returns free memory to the system, and at the start of a round
  it first shrinks its arrays and gives back its matrices' buffers. Where
  the resident set can no longer be read after the start, the run counts it as past the ceiling.
- **Units of work** (matrix-stop-latency.1–.9). Beside its ticks the
  budget counts units of work, a unit an entry of a row operation or a
  byte of a text, and
  checks in full once 64 ticks or `WORK`, 2^23, units have passed since
  the last full check, whichever comes first. A row's work ranges over
  six orders of magnitude, so no count of rows bounds the time between
  two checks: one combination of a random square system over
  GF(2^31 − 1) in `block:1,1,1` subtracted about 170 million entries.
  2^23 entries take 3.7 ms over `Gf31` to 14 ms over `Gf31Pair` on an
  i7-14700K, where a full check reads the clock in 10 ns and
  `/proc/self/statm` in 0.63 µs. Each row operation of a reduction, a
  back-substitution, or a combination spends its row's length right
  after it (matrix-stop-latency.3), in a band once per band row it is
  subtracted from (row-traffic-time.12). A row added to a band's
  combinations spends its length `BAND` times, right after its pass
  (plain-reduction-time.6). A count kept inside the reduction of
  a row and spent past 2^12 raised noon-7-p65521's instructions by 7.9%.
  Symbolic preprocessing spends `PRODUCT`, 16, units a product, every row
  it multiplies, the given rows included (matrix-stop-latency.5), and the
  columns' mapping a unit an entry. A reduction's setup spends its
  matrix's rows and columns once, after its reducer is made and any pivots
  it sets before its first row (matrix-stop-latency.8). In grevlex a sort spends `KEY`, 16, units an
  element once its keys are built and `SORTED`, 64, once sorted
  (threshold-recheck.12, .13), and
  stays one sort (matrix-stop-latency.7): sorted in runs,
  sparse-18-p65521 peaked 20.7 MiB higher. A sort without keys, in lex
  and the block orders, sorts more than `RUN`, 2^15, elements in runs
  merged pairwise, with a full check after each run and wherever the
  merge's buffer reaches a multiple of `RUN` as it compares, the rest of a
  run copied without one (matrix-stop-latency.6). One comparison sort of a
  random square system's columns in `block:1,1,1` went 0.799 s. The scan
  of the dense row is not spent, as the rows' ticks bound it
  (matrix-stop-latency.4). Nor are the divisor search of symbolic
  preprocessing and the signature run's reducer search
  (matrix-stop-latency.9), nor the divisor search's sort of G_min
  (mask-scan-time.6). FGLM's own steps and the
  checker spend no units and tick alone, the checker once per 4096 units of
  its own (_The checker's units of work_ below). The lift's attempts over ℚ
  tick once per coefficient, and each spends its conversion's multiply-adds
  after it, a unit each (q-lift-growth-time.10): an attempt at 2,000 primes
  takes about 4 million. The texts, the cofactors, and the lift's
  merge and append over ℚ spend theirs (_Texts' units of work_, _Cofactors'
  units of work_ below). Release builds on an
  i7-14700K, timed between every two full checks under seeds 1 and 2:
  the longest gap of a round's matrix was 0.069 s on the tuning set's
  instances past 50 ms in grevlex, a keyed sort of sparse-18-p65521's
  1.26 million rows by signature, and 0.020 s on 50 random square
  systems in block orders. Against matrices that tick per row alone, the
  tuning set took 1.0005 of the time and 0.9992 of the peak memory in
  the geomean over the 112 instances both solve, and executed 1.0017 of
  the instructions.
- **Texts' units of work** (text-stop-latency.1–.12). Every text a call
  returns, the call report aside, spends units of work on its budget as it is written, a unit a
  byte, and a stop while a text is written drops the texts and returns
  the stop with no basis. One polynomial of biomd-123-p0's basis text
  holds 53 MB, and one cofactor entry of sparse-16-p65521 5,848,126
  terms, so no count of polynomials or entries bounds the time between
  two checks. A writer over GF(p) spends the bytes of each `CHUNK`,
  2^10, terms after them, with no test per term of whether to spend in
  the loop over a chunk (text-stop-latency.2): such a test per term raised
  `write_basis`'s instructions on noon-7-p65521 from 8.42 to 8.92
  million. Over ℚ each term spends its bytes and the square of the limbs
  of its numerator and of its denominator (.3). The decimal conversion
  took 0.79 to 1.26 ns per square limb, and one integer of mq-8-p0's
  basis text has 9,225 digits. Each integer of a basis text over ℚ past
  its element's leading one
  spends the square of its limbs once more as it is computed (.12). The
  scaling of a cofactor term over ℚ to lowest terms spends `SCALE`·(m² +
  4) units before it, `SCALE` = 16, m the larger of the limbs of the
  scaled numerator's two factors summed and of the denominator (.4): the
  scaling took 19.36 s of mq-7-p0's call with cofactors. Each entry of a
  cofactor text spends `ENTRY`, 16, units beside its terms, so a row of
  zero entries is spent as well (.5). A cofactor entry over GF(p) is
  sorted by its monomial ids, and its terms follow by a gather (.9): in
  grevlex by their keys, each run of equal keys by the order's
  comparison (.7), and in lex and the block orders by the comparison.
  Each sort is `sort_runs`'s, in runs past `RUN` elements, and the keys
  are built and the terms gathered `RUN` at a time, each spent (.6, .8).
  The keys' sort is one function for every format and order. Sorted in
  runs by the comparison alone, sparse-16-p65521's cofactor text took
  5.06 and 5.53 s against 4.22 and 4.80 s for one sort. Sorted by keys
  it took 3.98 s against 4.57 s, and katsura-12-p65521's 3.06 s against
  3.91 s. No full check follows the last text but the checker's, so a call
  that asks for no checker and finishes its texts returns them (.11).
  The Python module's copy of each text
  into a Python string and the C interface's handover of its texts come
  after the call and are not checked (.10): the copy took 0.027 s for
  220.6 MB (_Python_ above). Release builds on an
  i7-14700K, 403 calls under fresh seeds,
  the tuning set's instances past 50 ms in grevlex, FGLM's cases in lex,
  50 random square systems, and calls with cofactors and the checker:
  the longest gap of a text between two full checks was 0.038 s,
  eco-10-p0's texts over ℚ with cofactors, and 0.027 s over GF(p),
  katsura-12-p65521's cofactor text of 1.02 GB. Texts that spent no
  units went 0.885 s on biomd-123-p0's basis text, 5.598 s on
  sparse-16-p65521's cofactor text, and 20.16 s on mq-7-p0's texts with
  cofactors. In the measuring condition the tuning set took 1.0069 of the
  time and 0.9981 of the peak memory against texts that spend no units,
  in the geomean over the 112 instances both solve, and executed 0.9991
  of the instructions.
- **Cofactors' units of work** (cofactor-stop-latency.1–.8). Every stage
  that works on cofactors spends units of work on the call's budget as it
  works, and over ℚ so do the lift's merge and append, with
  cofactors or without. A stop in any of them returns the stop with no
  basis. One cofactor entry of sparse-16-p65521 holds 5,848,126 terms,
  and cyclic-7-p0's cofactor image 137 million, so no count of rows,
  entries, or parts bounds the time between two checks (.1). A
  composition and FGLM's lift spend `ENTRY`, 16, units an entry they
  read, in place of a tick per row and generator (.2), and `TERM`, 64,
  units a term after each `TERMS`, 2^12, terms of an entry (.3). A term
  took 71 to 79 ns in a benchmark and about 156 ns in sparse-16-p65521's
  signature run, where 16 units a term went 0.082 s between two checks
  and a spend per entry of its whole length 0.256 s. A replay's cofactor
  image sorts each entry by its monomial ids as a cofactor text does, and
  gathers its terms `RUN` at a time, spending `ENTRY` an entry and `KEY` a
  term (.4). The merge compares a part's exponents with the image's
  `COMPARED`, 2^16, words at a time, a unit per 8 words, and writes the
  union spending `MERGED` + w units a term, `MERGED` = 16, w the
  words of a term's images, `COMBINED`, 2^10, terms at a time (.5). The
  append spends `LIMB`, 2, units a word it writes, the moves of a doubling
  included, after each `COMBINED` coefficients, beside its tick per part
  (.6; q-lift-growth-time.10). A part's images lie in one
  vector, appended in place (.7, .8; _Over ℚ_, _Images and their conversion_), so a
  stopped call frees a block per part. Release builds on an i7-14700K,
  220 calls under fresh seeds, FGLM's cases in lex, 50 random square
  systems in lex and in block orders, and calls with cofactors and the
  checker: the longest gap of a cofactor stage between two full checks
  was 0.066 s, sparse-16-p65521's signature run's compositions, then
  0.052 s, cyclic-7-p0's cofactor image. Outside the checker a stopped
  call returned within 0.048 s of the check that stopped it. The
  checker's rows stage, before it counted units of work, went on 1.27 s on
  sparse-16-p65521 (_The checker's units of work_ below). Stages that ticked per
  row and generator or per part, or not at all, went 4.150 s in
  cyclic-7-p0's image and merge, 0.774 s in sparse-16-p65521's
  inter-reduction's compositions, and 0.587 s in biomd-123-p0's
  combination, and cyclic-7-p0 returned 0.265 s after its stopping check.
  In the measuring condition the tuning set took 0.9943 of the time and
  0.9989 of the peak memory against stages that tick alone and a
  vector per residue, in the geomean over the 112 instances both solve,
  and executed 1.0003 of the instructions. Over the 49 over ℚ it took
  0.9918 of the time and 0.9488 of the peak memory.
- **The checker's units of work** (checker-stop-latency.1–.12). The
  checker counts its work in units, a unit about the work of one term,
  and calls its poll once per `STEPS`, 4096, units, beside its call per
  element of G in the reduced verdict, generator, row, pair, and point
  (.1). Each call ticks the budget, so a full check comes within 64
  calls, 2^18 units. One cofactor entry of sparse-16-p65521 holds
  5,848,126 terms, so no count of elements, rows, or pairs bounds the
  time between two checks: a poll per step and per 4096 reduction steps
  let its exact rows stage go 8.818 s between two full checks. Units
  are counted before their work is done, save a term read (.8), and n of them call the poll
  once per 4096 they pass (.2). The geobucket builds a summand and
  merges a block of at most `STRIDE`, 4096, terms of each operand at a
  time, a unit a term, each block counted before it and the merge's
  loop cut to the block's terms (.3): a count per term in the merge
  loop took mq-10-p65521's call with the exact checker from 39.5 s to
  43.5 s. An extraction counts a unit per bucket a turn (.4). A merge
  whose output lacks room allocates it afresh (.11): `reserve` on a
  cleared vector copies its old contents, 0.013 s in one call. Over ℚ
  the heap counts 1 + 2·l·w/`WORDS` units a product it takes, l the
  bits of its length, w a key's words, and `WORDS` 16, beside the
  product's cost (.5). A product or a quotient of integers counts 2 +
  la·lb/16 units, la and lb their limbs, and a gcd or an lcm
  64·(1 + L) + 2·L², L the larger's limbs (.6): num-bigint's gcd shifts a bit at a
  time and took 37.7 µs at 32 limbs, where a product took 0.53 µs. A
  divisor query the memo does not answer counts 1 + n/`MASKS` units
  before its scan, n the divmasks of the index and `MASKS` 32, the memo's
  growth a unit a
  monomial it moves, and an answer from the memo none (.7). The texts
  are read at 1 + b/`BYTES` units a term, b its bytes and `BYTES` 16,
  counted after the term, and the reading ends at a stop, so an error
  past the term the stop falls in is not reached (.8, .9). A polynomial read is tested for its
  order a block of 4096 comparisons at a time (.10). The values of F
  and G at a point count a unit a term, over ℚ one more per 8 of a
  coefficient's limbs and its fraction's limbs, computed once per
  polynomial (.12). Release builds on an i7-14700K, through the CLI
  under `deadline-latency`'s instrument, 111 calls with cofactors and
  the checker under fresh seeds: 17 instances exact and at 1,000
  points, each call that returned again with its deadline at 0.25, 0.5,
  and 0.75 of the way through its check, and katsura-12-p65521 and
  mq-11-p65521 checked exactly. The longest gap of the checker between
  two full checks was 0.0266 s, eco-10-p0's S-pair stage, and over
  GF(65521) 0.0205 s, sparse-16-p65521's exact rows stage. A call
  stopped inside the checker returned within 0.0349 s of its stopping
  check, katsura-12-p65521. A checker that polled per step and per 4096
  reduction steps went on 1.141 s after its stopping check, at 1,000
  points on sparse-16-p65521. Against that checker, over 12 calls with
  cofactors and the checker through the CLI, three rounds, the calls
  took 1.0067 of the time in the geomean, and the check on
  cyclic-7-p65521 executed 1.009 of the instructions. The checker's
  merge loop moves with the build's code placement: two builds of the
  same work took mq-10-p65521 checked exactly 39.0 and 43.4 s.
- **Address space** (matrix-slack.1, .3). A matrix whose columns pass
  1 MiB reserves 32 MiB of address space, of which the resident set holds
  only the pages the run's matrices have written. The memory ceiling and a cgroup's limit
  count the resident set. `ulimit -v` counts the address space, the whole
  reservation among it.
