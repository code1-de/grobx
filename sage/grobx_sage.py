"""grobx_sage -- reduced Gröbner bases of Sage's polynomial ideals, computed by grobx.

`groebner_basis(I)` is called as Sage's `I.groebner_basis()` is and returns what Sage's returns, for the
fields and orders grobx computes in. The ideal reaches grobx as instance text, and the basis comes back as
basis text (grobx's `MANUAL.md`, _Formats_). The text names the variables v0, v1, …: grobx reads ASCII
names only, and a Sage variable may carry any Python identifier.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import grobx
from sage.rings.polynomial.multi_polynomial_ideal import MPolynomialIdeal
from sage.rings.polynomial.multi_polynomial_ring_base import MPolynomialRing_base
from sage.rings.polynomial.multi_polynomial_sequence import PolynomialSequence
from sage.rings.polynomial.pbori.pbori import BooleanPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

if TYPE_CHECKING:
    from sage.rings.polynomial.multi_polynomial import MPolynomial
    from sage.rings.polynomial.multi_polynomial_sequence import PolynomialSequence_generic
    from sage.rings.polynomial.term_order import TermOrder

# The variables of the instance text are v0, v1, …
VARIABLE = "v"
# How often, in seconds, the calling thread wakes while grobx computes, to see a Ctrl-C.
POLL = 0.1
# grobx's name of each term order, by the name Sage's TermOrder keeps. Sage keeps Singular's dp and lp as given,
# and lowercases Dp to dp, so a ring in order="Dp" is grevlex in Sage as it is here.
ORDERS = {"degrevlex": "grevlex", "dp": "grevlex", "deglex": "deglex", "lex": "lex", "lp": "lex"}


def groebner_basis(
    ideal: MPolynomialIdeal,
    *,
    threads: int | None = None,
    seed: int | None = None,
    deadline: int | None = None,
    memory: int | None = None,
    checker: str | int | None = None,
) -> PolynomialSequence_generic:
    """The reduced Gröbner basis of `ideal` in its ring's term order, as `ideal.groebner_basis()` gives it.

    The ring is a multivariate polynomial ring over QQ or GF(p), p a prime below 2^64, or a Boolean
    polynomial ring, in degrevlex, deglex, lex, or a block order of degrevlex blocks. The basis is an immutable
    sequence of monic polynomials of that ring, the largest leading monomial first. The zero ideal's basis is
    [0]. The keywords are those of `grobx.groebner`, and one left at None takes grobx's default.

    Raises TypeError where `ideal` is no ideal of a multivariate polynomial ring, ValueError for a ring grobx
    does not compute in or an argument grobx rejects, RuntimeError where a deadline or memory ceiling stopped
    the call or the checker rejected the basis, and KeyboardInterrupt on Ctrl-C once grobx has stopped.
    """
    if not isinstance(ideal, MPolynomialIdeal):
        raise TypeError(
            "groebner_basis takes an ideal R.ideal(F) of a multivariate polynomial ring, "
            f"PolynomialRing(K, 'x', 1) for one variable, not {type(ideal).__name__}"
        )
    ring = ideal.ring()
    if not isinstance(ring, MPolynomialRing_base):
        raise ValueError(f"grobx computes in rings PolynomialRing(K, names), not in {ring}")
    given = {"threads": threads, "seed": seed, "deadline": deadline, "memory": memory, "checker": checker}
    options = {k: v for k, v in given.items() if v is not None}
    if isinstance(ring, BooleanPolynomialRing):
        return boolean_basis(ideal, options)
    return ring_basis(ideal, options)


def ring_basis(ideal: MPolynomialIdeal, options: dict[str, int | str]) -> PolynomialSequence_generic:
    """The basis `groebner_basis` returns for `ideal`, an ideal of a ring other than a Boolean ring."""
    ring = ideal.ring()
    p = characteristic(ring)
    order = grobx_order(ring.term_order())
    return basis_sequence(compute(instance_text(ideal, p), order, options), ring)


def boolean_basis(ideal: MPolynomialIdeal, options: dict[str, int | str]) -> PolynomialSequence_generic:
    """The basis `groebner_basis` returns for `ideal`, an ideal of a Boolean ring.

    The Boolean ring is its cover ring GF(2)[x], in the same term order, modulo the field equations
    x^2 + x. The cover ring's reduced basis of the ideal and the field equations holds the field equation of
    each variable that no other leading monomial divides, and multilinear elements. The field equations map
    to 0 in the Boolean ring, and the multilinear elements are the Boolean basis, in Sage's order.
    """
    ring = ideal.ring()
    cover = ring.cover_ring()
    lifted = cover.ideal([cover(f) for f in ideal.gens()] + [x**2 + x for x in cover.gens()])
    elements = [g for g in map(ring, ring_basis(lifted, options)) if g != 0]
    # Sage's own Boolean groebner_basis raises IndexError on the zero ideal; [0] is the ordinary rings' answer.
    return PolynomialSequence(ring, elements or [ring(0)], immutable=True)


def characteristic(ring: MPolynomialRing_base) -> int:
    """The characteristic of `ring`'s base field, QQ or GF(p).

    grobx checks the bound p < 2^64.
    """
    field = ring.base_ring()
    if not field.is_prime_field():
        raise ValueError(f"grobx computes over QQ and GF(p), p a prime, not over {field}")
    return int(field.characteristic())


def grobx_order(order: TermOrder) -> str:
    """grobx's name of the term order `order`."""
    name = order.name()
    if name in ORDERS:
        return ORDERS[name]
    if name == "block" and all(ORDERS.get(b.name()) == "grevlex" for b in order.blocks()):
        return "block:" + ",".join(str(len(b)) for b in order.blocks()[:-1])
    raise ValueError(f"grobx computes in degrevlex, deglex, lex, and blocks of degrevlex, not in {order}")


def instance_text(ideal: MPolynomialIdeal, p: int) -> str:
    """The instance text of `ideal` over the prime field of characteristic `p`."""
    names = ",".join(f"{VARIABLE}{i}" for i in range(ideal.ring().ngens()))
    generators = ",\n".join(generator_text(f, p) for f in ideal.gens())
    return f"{names}\n{p}\n{generators}\n"


def generator_text(f: MPolynomial, p: int) -> str:
    """`f` as a generator of the instance text over the prime field of characteristic `p`."""
    terms: list[str] = []
    for exponents, c in f.monomial_coefficients().items():
        if p == 0:
            sign = "-" if c < 0 else "+"
            numerator, denominator = abs(c.numerator()), c.denominator()
            number = f"{numerator}" if denominator == 1 else f"{numerator}/{denominator}"
        else:
            sign, number = "+", f"{int(c)}"
        monomial = "*".join(f"{VARIABLE}{i}" if e == 1 else f"{VARIABLE}{i}^{e}" for i, e in enumerate(exponents) if e)
        terms.append(f"{sign}{number}*{monomial}" if monomial else f"{sign}{number}")
    return "".join(terms) or "0"


def compute(text: str, order: str, options: dict[str, int | str]) -> str:
    """The basis text grobx computes for the instance text `text` in grobx's order `order`."""
    cancel = grobx.Cancel()
    outcome: list[grobx.Result | BaseException] = []

    def run() -> None:
        try:
            outcome.append(grobx.groebner(text, order=order, cancel=cancel, **options))
        except BaseException as e:  # raised again in the calling thread
            outcome.append(e)

    # A thread blocked inside grobx sees no Ctrl-C until grobx returns. The call runs in a worker, and a
    # Ctrl-C in the calling thread raises grobx's cancel flag.
    worker = threading.Thread(target=run, name="grobx", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(POLL)
    except KeyboardInterrupt:
        cancel.set()
        worker.join()
        raise
    [result] = outcome
    if isinstance(result, BaseException):
        raise result
    if result.stopped is not None:
        raise RuntimeError(f"grobx: the {result.stopped} bound stopped the call")
    if result.certified is False:
        raise RuntimeError(f"grobx: the checker rejected the basis: {result.flaw}")
    if result.basis is None:
        raise RuntimeError("grobx returned no basis and named no bound that stopped it")
    return result.basis


def basis_sequence(text: str, ring: MPolynomialRing_base) -> PolynomialSequence_generic:
    """The basis text `text` as Sage gives a basis of an ideal of `ring`."""
    if not (text.startswith("[") and text.endswith("]:\n")):
        raise RuntimeError(f"grobx returned a basis text out of its format: {text[:80]!r}")
    body = text[1:-3]
    if not body:
        return PolynomialSequence(ring, [ring(0)], immutable=True)
    names = [f"{VARIABLE}{i}" for i in range(ring.ngens())]
    parse = PolynomialRing(ring.base_ring(), names, len(names))
    # grobx writes the smallest leading monomial first, and over QQ with integer coefficients.
    elements = [ring(parse(e).monomial_coefficients()) for e in reversed(body.split(",\n"))]
    return PolynomialSequence(ring, [g * ~g.lc() for g in elements], immutable=True)
