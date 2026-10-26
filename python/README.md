# grobx

grobx computes reduced Gröbner bases of polynomial ideals over prime fields
GF(p), p a prime below 2^64, and over the rationals ℚ. This is its Python
module. It takes the ideal's generators as text and returns the basis as
text.

The package is a wheel for Linux on x86-64 and CPython 3.12 and later.

## Install

```sh
pip install grobx
```

## Example

```python
import grobx

r = grobx.groebner("x,y\n7\nx^2+y,\nx*y-1\n")
r.basis    # '[y^2+x,\nx*y+6,\nx^2+y]:\n'
```

The input is msolve's format: the variables on line 1, separated by
commas, the largest first; the characteristic on line 2, a prime below
2^64 for GF(p) or 0 for ℚ; then the generators, separated by commas. Over
ℚ:

```python
r = grobx.groebner("x,y,z\n0\nx+y+z,\nx*y+y*z+z*x,\nx*y*z-1\n")
print(r.basis)
# [x+y+z,
# y^2+y*z+z^2,
# z^3-1]:
```

## The call

```python
grobx.groebner(text, *, order="grevlex", threads=0, seed=None, deadline=None, cancel=None,
               memory=None, checker=None, cofactors=False, report=False)
```

| Keyword | Meaning |
|---|---|
| `order` | `"grevlex"`, `"lex"`, or `"block:k1,…,km"`: grevlex blocks of k1, …, km variables and a last block of the rest |
| `threads` | How many threads the call may use, 0 for all; the engine computes on one |
| `seed` | The one source of every random choice, 0 to 2^64 − 1; `None` draws one |
| `deadline` | A time limit on the whole call, in milliseconds |
| `cancel` | A `grobx.Cancel` that another thread raises by `set()` to stop the call |
| `memory` | A ceiling on the process's resident set, in bytes |
| `checker` | Has the checker rule on the result: `"exact"`, or the number of random points of the cheap mode |
| `cofactors` | Whether the result carries the cofactor matrix |
| `report` | Whether the result carries the call report |

The call releases the GIL while it computes. It returns a `grobx.Result`:

| Attribute | Meaning |
|---|---|
| `basis` | The basis text; `None` where a bound stopped the call |
| `stopped` | The bound that stopped the call, `"deadline"`, `"cancel"`, or `"memory"`; else `None` |
| `certified` | The checker's verdict; `None` where it did not rule |
| `flaw` | Why the checker rejected the result |
| `cofactors` | The cofactor text, where the call asked for it |
| `report` | The call report, where the call asked for it |
| `digest` | SHA-256 of `basis`, 64 lowercase hex digits |
| `seed` | The seed the call ran under |

An instance or a parameter the call cannot take raises `grobx.Error`, a
subclass of `ValueError`.

## Documentation

The [manual](https://github.com/code1-de/grobx/blob/master/MANUAL.md#python)
describes the formats, the bounds, the checker, and the engine.
[grobx.com](https://grobx.com) has the benchmarks and the instance corpus,
and [GitHub](https://github.com/code1-de/grobx) the source.

## License

GPL-3.0-only. Copyright Code1 GmbH.

Code1 GmbH also offers grobx under a commercial license.
