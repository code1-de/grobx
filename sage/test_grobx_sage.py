"""test_grobx_sage -- grobx_sage.groebner_basis against Sage's own groebner_basis.

Runs under Sage's Python with grobx installed (README.md, _Test_).
"""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import TYPE_CHECKING

import grobx
from sage.all import (
    GF,
    QQ,
    RR,
    ZZ,
    BooleanPolynomialRing,
    NumberField,
    PolynomialRing,
    TermOrder,
    next_prime,
    set_random_seed,
)
from sage.rings.ideal import Cyclic, Katsura

import grobx_sage

if TYPE_CHECKING:
    from sage.rings.polynomial.multi_polynomial_ideal import MPolynomialIdeal

# 2^64 - 59 is the largest prime below 2^64.
FIELDS = (QQ, GF(65521), GF(2**64 - 59))
# Orders of four variables.
ORDERS = (
    "degrevlex",
    "dp",
    "Dp",
    "deglex",
    "lex",
    "lp",
    "degrevlex(2),degrevlex(2)",
    "degrevlex(1),degrevlex(3)",
    "dp(2),dp(2)",
)
# Random Boolean systems per order, under a fixed seed so that a failure repeats.
BOOLEAN_SYSTEMS = 10
BOOLEAN_SEED = 1
# grobx takes tens of seconds on cyclic-9 over GF(65521), so a bound or a Ctrl-C stops it before it ends.
HARD = 9
DEADLINE_MS = 200
INTERRUPT_AFTER_S = 0.5
# How long grobx may take to stop after a Ctrl-C.
CANCEL_LIMIT_S = 10.0


def test_groebner_basis() -> None:
    failures: list[str] = []

    def agree(ideal: MPolynomialIdeal, what: str, **options: int | str) -> None:
        try:
            got = grobx_sage.groebner_basis(ideal, **options)
        except Exception as e:
            failures.append(f"{what}: raised {e!r}")
            return
        want = ideal.groebner_basis()
        if list(got) != list(want):
            failures.append(f"{what}: {list(got)} != Sage's {list(want)}")
        if got.ring() is not ideal.ring() or not got.is_immutable():
            failures.append(f"{what}: ring {got.ring()}, immutable {got.is_immutable()}")

    def raises(kind: type[BaseException], ideal: object, what: str, **options: int | str) -> None:
        try:
            grobx_sage.groebner_basis(ideal, **options)
        except Exception as e:
            if type(e) is not kind:
                failures.append(f"{what}: raised {e!r}, not {kind.__name__}")
            return
        failures.append(f"{what}: raised nothing, not {kind.__name__}")

    for field in FIELDS:
        for order in ORDERS:
            R = PolynomialRing(field, "a,b,c,d", order=order)
            a, b, c, d = R.gens()
            where = f"{field}, {order}"
            agree(Cyclic(R), f"cyclic-4 over {where}")
            agree(Katsura(R), f"katsura-3 over {where}")
            fractions = [
                a**2 - field(1) / 2 * b * c + 3,
                field(2) / 3 * a * b - d,
                c**2 - a + d**2,
                b * d - field(1) / 5,
            ]
            agree(R.ideal(fractions), f"fractions over {where}")
            agree(R.ideal([a * b - 1, a]), f"the unit ideal over {where}")
            agree(R.ideal(R(0)), f"the zero ideal over {where}")
            agree(R.ideal([]), f"no generators over {where}")

    U = PolynomialRing(QQ, "α,β,γ")
    α, β, γ = U.gens()
    agree(U.ideal([α**2 - β, β * γ - 1, γ**2 - α]), "Unicode names")
    x = PolynomialRing(GF(7), "x", 1).gen()
    agree(x.parent().ideal([x**3 - 1, x**2 - 1]), "one variable")
    W = PolynomialRing(GF(65521), 12, "w")
    w = W.gens()
    agree(W.ideal([w[i] - w[i + 1] for i in range(11)] + [w[11] ** 2 - 2]), "twelve variables")

    K = Katsura(PolynomialRing(QQ, "a,b,c,d"))
    agree(K, "the exact checker", checker="exact")
    agree(K, "the cheap checker", checker=16)
    agree(K, "a seed and one thread", seed=1, threads=1)

    raises(TypeError, list(K.gens()), "a list")
    t = PolynomialRing(QQ, "t").gen()
    raises(TypeError, t.parent().ideal([t**2]), "a univariate ring")
    for field in (GF(4), ZZ, RR, NumberField(t**2 + 1, "i"), PolynomialRing(QQ, "s").fraction_field()):
        raises(ValueError, PolynomialRing(field, "a,b").ideal([field(1)]), f"over {field}")
    raises(grobx.Error, PolynomialRing(GF(next_prime(2**64)), "a,b").ideal([1]), "over GF(p), p above 2^64")
    for order in (
        "degneglex",
        "negdegrevlex",
        "degrevlex(2),lex(2)",
        "degrevlex(2),deglex(2)",
        "deglex(2),deglex(2)",
        "lex(2),lex(2)",
        TermOrder("wdegrevlex", (1, 2, 3, 4)),
    ):
        raises(ValueError, PolynomialRing(QQ, "a,b,c,d", order=order).ideal([1]), f"order {order}")
    B = BooleanPolynomialRing(2, "a,b", order="degneglex")
    raises(ValueError, B.ideal([B.gen(0) * B.gen(1)]), "a Boolean ring in degneglex")
    raises(ValueError, K, "an unknown checker", checker="bogus")

    set_random_seed(BOOLEAN_SEED)
    for order in ORDERS:
        R2 = PolynomialRing(GF(2), "a,b,c,d", order=order)
        agree(Cyclic(R2), f"cyclic-4 over GF(2), {order}")
        agree(Katsura(R2), f"katsura-3 over GF(2), {order}")
        B = BooleanPolynomialRing(4, "a,b,c,d", order=order)
        for k in range(BOOLEAN_SYSTEMS):
            gens = [B.random_element(degree=2, terms=4) for _ in range(2)]
            agree(B.ideal(gens), f"Boolean system {k} {gens}, {order}")
        a = B.gen(0)
        agree(B.ideal([a + 1, a]), f"the Boolean unit ideal, {order}")
        # Sage's own Boolean groebner_basis raises IndexError on the zero ideal.
        for zero in (B.ideal(B(0)), B.ideal([])):
            if list(got := grobx_sage.groebner_basis(zero)) != [0]:
                failures.append(f"the Boolean zero ideal, {order}: {list(got)}, not [0]")
    B1 = BooleanPolynomialRing(1, "x")
    agree(B1.ideal([B1.gen(0) + 1]), "a Boolean ring in one variable")

    hard = Cyclic(PolynomialRing(GF(65521), HARD, "x"))
    raises(RuntimeError, hard, f"cyclic-{HARD} under a deadline", deadline=DEADLINE_MS)
    timer = threading.Timer(INTERRUPT_AFTER_S, os.kill, (os.getpid(), signal.SIGINT))
    start = time.monotonic()
    timer.start()
    try:
        grobx_sage.groebner_basis(hard)
        failures.append(f"cyclic-{HARD} under Ctrl-C: finished")
    except KeyboardInterrupt:
        if (elapsed := time.monotonic() - start) > INTERRUPT_AFTER_S + CANCEL_LIMIT_S:
            failures.append(f"cyclic-{HARD} under Ctrl-C: stopped only after {elapsed:.1f} s")
    finally:
        timer.cancel()
    if any(th.name == "grobx" and th.is_alive() for th in threading.enumerate()):
        failures.append(f"cyclic-{HARD} under Ctrl-C: grobx still runs")

    assert not failures, "\n".join(failures)
