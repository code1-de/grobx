# grobx

grobx computes reduced Gröbner bases of polynomial ideals over prime fields
GF(p), p a prime below 2^64, and over the rationals ℚ. GF(2) takes a path of
its own, with rows packed into machine words, and is much faster than the
other primes. grobx is a Rust crate
with a command-line program, a Python extension module, and a C interface.
All four take the ideal's generators as text and return the basis as text.

Over ℚ and over GF(p), grobx is 7× faster than Maple, msolve, or Macaulay2.
[grobx.com/benchmarks.html](https://grobx.com/benchmarks.html) gives the time and memory of every
instance.

grobx runs on Linux and Windows on x86-64. It builds for macOS on Apple
Silicon, where it is untested.
It also builds for the browser, `wasm32-unknown-unknown`, as the module in `wasm/`, which
[grobx.com/live.html](https://grobx.com/live.html) runs.

## Rust

```sh
cargo add grobx
```

```rust
let outcome = grobx::groebner("x,y\n7\nx^2+y,\nx*y-1\n", &grobx::Params::default())?;
assert_eq!(outcome.basis.as_deref(), Some("[y^2+x,\nx*y+6,\nx^2+y]:\n"));
```

The input is msolve's format: the variables on line 1, the characteristic
on line 2, a prime below 2^64 for GF(p) or 0 for ℚ, then the generators,
separated by commas.

## Python

```sh
pip install grobx
```

The wheels are for Python 3.12 and later on Linux and Windows on x86-64 and on macOS on Apple
Silicon.

```python
import grobx

r = grobx.groebner("x,y\n7\nx^2+y,\nx*y-1\n")
r.basis    # '[y^2+x,\nx*y+6,\nx^2+y]:\n'
```

## Sage

```sh
sage -pip install grobx-sage
```

```sage
import grobx_sage
R.<x,y> = PolynomialRing(GF(7))
grobx_sage.groebner_basis(R.ideal(x^2 + y, x*y - 1))
# [x^2 + y, x*y - 1, y^2 + x]
```

[sage/README.md](sage/README.md) describes the orders and keywords it takes.

## Command line, C, and Maple

These build from a clone of the repository:

```sh
git clone https://github.com/code1-de/grobx
cd grobx
```

The command-line program:

```sh
cargo build --release -p grobx-cli
target/release/grobx instance.ms > basis.txt
```

The C library: `cargo build --release -p grobx-c` writes
`target/release/libgrobxc.so`, `libgrobxc.dylib` on macOS, or `grobxc.dll`
on Windows. A C program includes `c/grobx.h` and links with `-lgrobxc`, on
Windows against the import library `grobxc.dll.lib`.

The Maple package `maple/Grobx.mpl`, which runs the command-line program
from the `PATH` or from the path in `GROBX`:

```maple
read "maple/Grobx.mpl":
with(Grobx):
Basis([x^2 + y, x*y - 1], tdeg(x, y), characteristic = 7);
# [y^2 + x, x*y + 6, x^2 + y]
```

[maple/README.md](maple/README.md) describes it.

## Documentation

[MANUAL.md](MANUAL.md) describes every interface, the input and output
formats, how a call behaves, and the design of the engine.
[grobx.com](https://grobx.com) has the benchmarks and the instance corpus.

## License

GPL-3.0-only, see [LICENSE](LICENSE). Copyright Code1 GmbH.

Code1 GmbH also offers grobx under a commercial license.
