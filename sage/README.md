# Grobx for Sage

The Python module `grobx_sage` computes reduced Gröbner bases of Sage's polynomial ideals with
grobx's Python module. Its `groebner_basis(I)` is called as Sage's `I.groebner_basis()` is, for the
fields and orders grobx computes in, returns what Sage's returns, and raises an error for anything
else.

Over ℚ and over GF(p), grobx is 4× faster than Maple and msolve.
[grobx.com/benchmarks.html](https://grobx.com/benchmarks.html) gives the time and memory of every
instance.

```sage
import grobx_sage
R.<x,y> = PolynomialRing(GF(7))
grobx_sage.groebner_basis(R.ideal(x^2 + y, x*y - 1))
# [x^2 + y, x*y - 1, y^2 + x]
```

## Setup

`sage -pip install grobx-sage` installs the module `grobx_sage` from PyPI into Sage's Python, and
grobx's Python module with it. `make dist` builds the package from this directory into `dist/`: the
sdist, and the wheel `grobx_sage-<version>-py3-none-any.whl`, which `sage -pip install` takes as
well.

## groebner_basis

`groebner_basis(I, *, threads=None, seed=None, deadline=None, memory=None, checker=None)` returns
the reduced Gröbner basis of the ideal `I` in its ring's term order. The ring is a multivariate
polynomial ring over ℚ or GF(p), p a prime below 2^64, or a Boolean polynomial ring (_Boolean
rings_ below). The basis is an immutable
`PolynomialSequence` of monic polynomials of that ring, the largest leading monomial first, as Sage
returns it. The zero ideal's basis is `[0]`.

| Term order of the ring | grobx's order |
|---|---|
| `degrevlex`, Sage's default, or `dp` | grevlex |
| `deglex` | deglex |
| `lex` or `lp` | lex |
| `degrevlex(k1),…,degrevlex(km)` | `block:k1,…,k(m−1)`: grevlex in each block, the blocks in turn |

The sum `TermOrder("degrevlex", k1) + … + TermOrder("degrevlex", km)` is the same block order, and
so is `dp(k1),…,dp(km)`. Sage lowercases an order's name, so its `Dp` is `dp`, which is grevlex.

The keywords are those of `grobx.groebner` (grobx's `python/README.md`, _The call_). A keyword left
at `None` takes grobx's default.

| Keyword | Value |
|---|---|
| `threads` | how many threads the call may use; 0 is all the machine has |
| `seed` | the seed of every random choice, below 2^64 |
| `deadline` | a time limit on the whole call, in milliseconds |
| `memory` | a ceiling on the resident set of Sage's process, in bytes |
| `checker` | `"exact"`, or the number of random points of the cheap mode |

`checker` has grobx's checker certify the result: that the basis is a Gröbner basis of `I`, and
reduced.

Ctrl-C raises grobx's cancel flag. `groebner_basis` raises `KeyboardInterrupt` once grobx has
stopped.

The ideal reaches grobx as instance text, its variables renamed v0, v1, …, and the basis comes back
as basis text (grobx's `MANUAL.md`, _Formats_). grobx reads ASCII names only, and a Sage variable
may carry any Python identifier.

## Boolean rings

A Boolean polynomial ring `B = BooleanPolynomialRing(n, names, order=…)` is its cover ring
`B.cover_ring()`, GF(2)[x] in the same term order, modulo the field equations x² + x.
`groebner_basis` computes the basis of the ideal and the field equations of all variables in the
cover ring, maps it into `B`, and drops the field equations, which map to 0 there. The result is
PolyBoRi's `I.groebner_basis()`. The orders are those of _groebner_basis_ above. The zero ideal's
basis is `[0]`, where Sage's own `groebner_basis` raises `IndexError`.

## Errors

`groebner_basis` raises:

- `TypeError` for anything but an ideal of a multivariate polynomial ring, an ideal of
  `PolynomialRing(K, 'x')` among them; `PolynomialRing(K, 'x', 1)` is the multivariate ring in one
  variable;
- `ValueError` for a base ring other than ℚ and GF(p), such as ZZ, RR, GF(p^k), a number field, or a
  fraction field;
- `ValueError` for any other order: `degneglex`, the negative orders, the weighted orders, matrix
  orders, and a block order with a block that is not `degrevlex`;
- `ValueError`, `grobx.Error` among them, for what grobx rejects: a prime above 2^64, or a keyword's
  value;
- `RuntimeError` where a deadline or a memory ceiling stopped the call, and where the checker
  rejected the basis, with the checker's reason.

## Test

`make test` builds the wheel and grobx's wheel from this tree, the latter by maturin and zig,
installs both under Sage in the Docker image `SAGE_IMAGE` (`sagemath/sagemath:10.10` by default),
and runs `test_grobx_sage.py` against the installed module. It compares `groebner_basis`
with Sage's `groebner_basis` on small systems over ℚ, GF(2), GF(65521), and GF(2^64 − 59) and on
random Boolean systems, in `degrevlex`, `dp`, `Dp`, `deglex`, `lp`, `lex`, and three block orders,
and on rings with Unicode names, one variable, and twelve variables. It
checks the keywords, that every input grobx does not compute raises its error, and that a deadline
and a Ctrl-C stop grobx on cyclic-9.

The wheels and the test reach the container as a tar on standard input. The container's user `sage`
cannot read a bind mount of a home directory that is closed to others.
