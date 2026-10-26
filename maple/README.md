# Grobx for Maple

The Maple package `Grobx` computes reduced Gröbner bases with grobx's
command-line program. Its `Grobx:-Basis` is called as Maple's
`Groebner:-Basis` is, for the orders and fields grobx computes in, and
raises an error for anything else.

```maple
read "/path/to/Grobx.mpl":
with(Grobx):
Basis([x^2 + y, x*y - 1], tdeg(x, y), characteristic = 7);
# [y^2 + x, x*y + 6, x^2 + y]
```

## Setup

The package is the one file `Grobx.mpl`, which `read` loads. It runs the
program `grobx` (`MANUAL.md`, _CLI_), which `cargo build --release -p grobx-cli`
writes to `target/release/grobx`. The package takes
the program from the first of:

1. the path `Grobx:-SetProgram(path)` names;
2. the path the environment variable `GROBX` names;
3. `grobx`, found by the shell's `PATH`.

## Basis

`Basis(F, T, options)` returns the reduced Gröbner basis of the polynomials
`F`, a list or a set, in the order `T`. The elements stand in increasing
order of their leading monomials. Over GF(p) every element is monic, its
coefficients in [0, p). Over ℚ every element has integer coefficients with
no common factor and a positive leading coefficient.

| Order `T` | grobx's order |
|---|---|
| `tdeg(x, y, …)` | grevlex |
| `plex(x, y, …)` | lex |
| `lexdeg([x, …], [y, …], …)` | block: grevlex in each block, the blocks in turn |
| `prod(tdeg(x, …), tdeg(y, …), …)` | block, as `lexdeg` |

The order's variables are the variables of the call. The coefficients of
`F` are rational numbers.

| Option | Value | Default |
|---|---|---|
| `characteristic` | 0 for ℚ, or a prime below 2^64 | 0 |
| `output` | `basis`, or `extended` for the basis and its cofactors | `basis` |
| `threads` | how many threads the call may use; 0 is all the machine has | grobx's, one |
| `seed` | the seed of every random choice, below 2^64 | drawn at random |
| `deadline` | a time limit on the whole call, in milliseconds | `infinity` |
| `memory` | a bound on the resident set of grobx's process, in bytes | `infinity` |
| `checker` | `exact`, or the number of random points of the cheap mode, below 2^32 | `false` |

`output = extended` returns `G, C`, where `G[i]` is the dot product of
`C[i]` with `F`, as `Groebner:-Basis` returns them.

`checker` has grobx's checker certify the result: that `G` is a Gröbner
basis of the ideal of `F`, and reduced. The cheap mode evaluates the
cofactor identities at random points instead of expanding them.

With `infolevel[Grobx] >= 1` the call prints what grobx wrote to standard
error, the seed it drew and the checker's verdict among it, and grobx's
call report.

A call writes its files into a directory of its own,
`grobx-<process id>-<call>` in Maple's temporary directory, runs `grobx`
there, and removes the files it wrote and then the directory. It runs the
program through `sh` on Unix and through `cmd.exe` on Windows.

## Errors

`Basis` raises an error for:

- any other order: `grlex`, `wdeg`, `'matrix'`, the `_min` orders, `prod`
  with a block that is not `tdeg`, and `MonomialOrder` objects;
- an order with no variable, a variable twice, or an empty block;
- a polynomial with a coefficient outside ℚ (`RootOf`, a float) or a name
  outside the order's variables (a parameter);
- a characteristic that is not 0 and not a prime below 2^64;
- any option `Groebner:-Basis` takes and `Basis` does not, `method` among
  them, and an ideal from `PolynomialIdeals` in place of `F`;
- an error of grobx, such as a denominator that is 0 mod p, prefixed
  `grobx:`;
- a deadline or memory ceiling that stopped the call;
- a result the checker rejected, with the checker's reason;
- a program `grobx` that cannot run, with its exit status.

## Test

`make test` runs `test.mpl` under Maple, `MAPLE` naming the installation
(`~/maple2026` by default), on the program `GROBX` names
(`../target/release/grobx` by default). It compares `Grobx:-Basis` with
`Groebner:-Basis` on small systems over ℚ, GF(65521), and a prime above
2^63, in every order. It checks the cofactors and the checker, that every
input grobx does not compute raises an error, and that the calls leave no
file open and no directory behind.

`maple-run.sh` runs a Maple script for make. Maple's command line exits 0
after an error, so the script also fails a run on its output.
