# test.mpl -- checks Grobx:-Basis against Groebner:-Basis on tiny systems,
# and that every input grobx does not compute raises an error. Run by
# `make test` in grobx/maple/, which passes it on its last line alone.

read "Grobx.mpl":
failures := []:
Leftovers := () -> select(d -> StringTools:-IsPrefix("grobx-", d),
                          FileTools:-ListDirectory(FileTools:-TemporaryDirectory())):
leftovers := Leftovers():
opened := iostatus()[1]:

Fail := proc(what)
    global failures;
    failures := [op(failures), what];
    printf("FAIL %s\n", what);
end proc:

# Each element divided by its leading coefficient, over Q or mod p.
Monic := proc(G, T, p)
    local g, c;
    [seq(proc(g)
             c := Groebner:-LeadingCoefficient(g, T);
             `if`(p = 0, expand(g / c), expand(g * (1 / c mod p)) mod p)
         end proc(g), g in G)]
end proc:

Reference := proc(F, T, p)
    `if`(p = 0, Groebner:-Basis(F, T), Groebner:-Basis(F, T, characteristic = p))
end proc:

Agree := proc(label, F, T, p)
    local mine, theirs, G, C, i, j, r;
    try
        mine := Grobx:-Basis(F, T, characteristic = p);
    catch:
        Fail(sprintf("%s: %s", label, StringTools:-FormatMessage(lastexception[2 .. -1])));
        return;
    end try;
    theirs := Reference(F, T, p);
    if Monic(mine, T, p) <> Monic(theirs, T, p) then
        Fail(sprintf("%s: %a differs from Groebner:-Basis %a", label, mine, theirs));
    end if;
    G, C := Grobx:-Basis(F, T, characteristic = p, output = extended);
    if G <> mine then
        Fail(sprintf("%s: output = extended changes the basis to %a", label, G));
    end if;
    for i to nops(G) do
        r := expand(G[i] - add(C[i][j] * F[j], j = 1 .. nops(F)));
        if `if`(p = 0, r, r mod p) <> 0 then
            Fail(sprintf("%s: cofactor row %d leaves %a", label, i, r));
        end if;
    end do;
    for i in [exact, 3] do
        try
            if Grobx:-Basis(F, T, characteristic = p, checker = i) <> mine then
                Fail(sprintf("%s: checker = %a changes the basis", label, i));
            end if;
        catch:
            Fail(sprintf("%s, checker = %a: %s", label, i, StringTools:-FormatMessage(lastexception[2 .. -1])));
        end try;
    end do;
end proc:

Refused := proc(label, f, needle)
    local message;
    try
        f();
        Fail(sprintf("%s: no error", label));
    catch:
        message := StringTools:-FormatMessage(lastexception[2 .. -1]);
        if StringTools:-Search(needle, message) = 0 then
            Fail(sprintf("%s: the error %s names no %s", label, message, needle));
        end if;
    end try;
end proc:

systems := [
    "manual" = [x^2 + y, x*y - 1],
    "cyclic-3" = [x + y + z, x*y + y*z + z*x, x*y*z - 1],
    "zero-dim" = [x^2 + y^2 + z^2 - 1, x - y, y - z^2],
    "positive-dim" = [x*y - z, x*z - y]
]:
orders := [tdeg(x, y, z), plex(x, y, z), lexdeg([x], [y, z]), prod(tdeg(x, y), tdeg(z))]:
primes := [0, 65521, nextprime(2^63)]:
for s in systems do
    for T in orders do
        for p in primes do
            Agree(sprintf("%s %a p=%d", lhs(s), T, p), rhs(s), T, p);
        end do;
    end do;
end do:

for p in [0, 65521] do
    Agree(sprintf("zero ideal p=%d", p), [], tdeg(x, y), p);
    Agree(sprintf("F = (0) p=%d", p), [0], tdeg(x, y), p);
    Agree(sprintf("zero generator p=%d", p), [0, x - 1, y^2 - x], tdeg(x, y), p);
    Agree(sprintf("unit ideal p=%d", p), [x, x - 1], tdeg(x, y), p);
    Agree(sprintf("already a basis p=%d", p), [y^2 + x, x*y + 6, x^2 + y], tdeg(x, y), p);
    Agree(sprintf("eliminable variable p=%d", p), [t - x^2, t - y^3], lexdeg([t], [x, y]), p);
    Agree(sprintf("set input p=%d", p), {x^2 + y, x*y - 1}, tdeg(x, y), p);
    Agree(sprintf("indexed variables p=%d", p), [a[1]^2 - a[2], a[1]*a[2] - 1], tdeg(a[1], a[2]), p);
    Agree(sprintf("a variable named as a renamed one p=%d", p), [_grobx2^2 - x, x*_grobx2 - 1], tdeg(x, _grobx2), p);
    Agree(sprintf("rational coefficients p=%d", p), [1/3*x^2 - 2/5*y, 7/2*x*y - 1], tdeg(x, y), p);
end do:

Refused("grlex", () -> Grobx:-Basis([x - y], grlex(x, y)), "takes the orders");
Refused("wdeg", () -> Grobx:-Basis([x - y], wdeg([1, 2], [x, y])), "takes the orders");
Refused("plex_min", () -> Grobx:-Basis([x - y], plex_min(x, y)), "takes the orders");
Refused("prod with plex", () -> Grobx:-Basis([x - y], prod(plex(x), tdeg(y))), "takes the orders");
Refused("no variable", () -> Grobx:-Basis([1], tdeg()), "at least one variable");
Refused("empty block", () -> Grobx:-Basis([x - y], lexdeg([x, y], [])), "no empty block");
Refused("repeated variable", () -> Grobx:-Basis([x - y], tdeg(x, y, x)), "each variable once");
Refused("parameter", () -> Grobx:-Basis([a*x + y], tdeg(x, y)), "rational coefficients");
Refused("RootOf", () -> Grobx:-Basis([RootOf(_Z^2 - 2)*x + y], tdeg(x, y)), "rational coefficients");
Refused("float", () -> Grobx:-Basis([0.5*x + y], tdeg(x, y)), "rational coefficients");
Refused("not a polynomial", () -> Grobx:-Basis([1/x + y], tdeg(x, y)), "rational coefficients");
Refused("composite characteristic", () -> Grobx:-Basis([x - y], tdeg(x, y), characteristic = 6), "prime p below 2^64");
Refused("characteristic past 2^64", () -> Grobx:-Basis([x - y], tdeg(x, y), characteristic = nextprime(2^64)),
        "prime p below 2^64");
Refused("denominator 0 mod p", () -> Grobx:-Basis([x/7 + y], tdeg(x, y), characteristic = 7), "grobx:");
Refused("method", () -> Grobx:-Basis([x - y], tdeg(x, y), method = fgb), "method");
Refused("seed past 2^64", () -> Grobx:-Basis([x - y], tdeg(x, y), seed = 2^64), "seed below 2^64");
Refused("points past 2^32", () -> Grobx:-Basis([x - y], tdeg(x, y), checker = 2^32), "points below 2^32");
Refused("PolynomialIdeal", () -> Grobx:-Basis(PolynomialIdeals:-PolynomialIdeal([x - y]), tdeg(x, y)), "invalid input");
Refused("deadline", () -> Grobx:-Basis([x + y + z, x*y + y*z + z*x, x*y*z - 1], tdeg(x, y, z), deadline = 0),
        "the deadline stopped the call");

try
    if Grobx:-Basis([x^2 + y, x*y - 1], tdeg(x, y), characteristic = 7, seed = 1)
       <> Grobx:-Basis([x^2 + y, x*y - 1], tdeg(x, y), characteristic = 7, seed = 2^64 - 1) then
        Fail("the seed changes the basis");
    end if;
catch:
    Fail(sprintf("seed: %s", StringTools:-FormatMessage(lastexception[2 .. -1])));
end try:

Grobx:-SetProgram("/nonexistent/grobx"):
Refused("no program", () -> Grobx:-Basis([x - y], tdeg(x, y)), "exited with status 127");
# A quote the shell's quoting broke on would exit 2, sh's syntax error.
Grobx:-SetProgram("/nonexistent/it's/grobx"):
Refused("a quote in the program's path", () -> Grobx:-Basis([x - y], tdeg(x, y)), "exited with status 127");
Grobx:-SetProgram(getenv("GROBX")):

if {op(Leftovers())} <> {op(leftovers)} then
    Fail(sprintf("the calls left %a in the temporary directory", {op(Leftovers())} minus {op(leftovers)}));
end if:
if iostatus()[1] <> opened then
    Fail(sprintf("the calls left %d files open", iostatus()[1] - opened));
end if:

if failures = [] then
    printf("test.mpl: all passed\n");
else
    printf("test.mpl: %d failed\n", nops(failures));
    `quit`(1);
end if:
