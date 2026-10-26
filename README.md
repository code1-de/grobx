# grobx

grobx computes reduced Gröbner bases of polynomial ideals over prime fields
GF(p), p a prime below 2^64, and over the rationals ℚ. It is a Rust crate
with a command-line program, a Python extension module, and a C interface.
All four take the ideal's generators as text and return the basis as text.

grobx runs on Linux and Windows on x86-64. It builds for macOS on Apple
Silicon, where it is untested.

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

## Command line, Python, C, and Maple

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

The Python extension module, for Python 3.12 and later:

```sh
cargo build --release -p grobx-python
cp target/release/libgrobx.so grobx.so
```

On macOS the library is `libgrobx.dylib`, copied as `grobx.so`; on Windows
it is `grobx.dll`, copied as `grobx.pyd`.

```python
import grobx

r = grobx.groebner("x,y\n7\nx^2+y,\nx*y-1\n")
r.basis    # '[y^2+x,\nx*y+6,\nx^2+y]:\n'
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
