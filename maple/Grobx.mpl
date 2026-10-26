# Grobx.mpl -- the Maple package Grobx: reduced Groebner bases computed by
# grobx's command-line program.
#
# Grobx:-Basis(F, T, opts) is called as Groebner:-Basis is, for the orders
# and fields grobx computes in, and raises an error for anything else. It
# writes the instance text into a directory of its own, runs `grobx` on it,
# and reads the basis text and the cofactor text back by parse: both are
# Maple syntax. The variables go to grobx as _grobx1, _grobx2, ..., which
# both accept however the variables are named; x[1] and the protected D and
# I are no names of the instance grammar.

Grobx := module()
    option package;
    export Basis, SetProgram;
    local program, calls, Program, Quote, Command, OrderOf, Message;

    program := FAIL;
    calls := 0;

    # The path of the program grobx. Without it, GROBX names it, and without
    # that the shell finds grobx by PATH.
    SetProgram := proc(path::string)
        program := path;
        path
    end proc;

    Program := proc()
        local env;
        if program <> FAIL then
            return program;
        end if;
        # getenv returns NULL for a variable that is not set.
        env := [getenv("GROBX")];
        if env <> [] and env <> [""] then
            return env[1];
        end if;
        "grobx"
    end proc;

    # s as one word of the command line ssystem hands on: to sh on Unix,
    # which takes '...' literally, and to cmd.exe on Windows, which takes
    # "..." and has no escape for a " inside it.
    Quote := proc(s)
        if kernelopts(platform) <> "windows" then
            return cat("'", StringTools:-SubstituteAll(s, "'", "'\\''"), "'");
        end if;
        if StringTools:-Search("\"", s) > 0 then
            error "Grobx: cannot pass %1 to cmd.exe, which takes no \" inside a word", s;
        end if;
        cat("\"", s, "\"")
    end proc;

    # The command line that runs words with standard error into err. cmd.exe
    # /c strips the first and the last quote of a line that starts with one,
    # so on Windows the line stands in one more pair.
    Command := proc(words, err)
        local line;
        line := cat(StringTools:-Join(map(Quote, words), " "), " 2> ", Quote(err));
        `if`(kernelopts(platform) = "windows", cat("\"", line, "\""), line)
    end proc;

    # The order's name in grobx and its variables.
    OrderOf := proc(T)
        local blocks, B;
        if type(T, 'specfunc(name, tdeg)') then
            return "grevlex", [op(T)];
        elif type(T, 'specfunc(name, plex)') then
            return "lex", [op(T)];
        elif type(T, 'specfunc(list(name), lexdeg)') then
            blocks := [op(T)];
        elif type(T, 'specfunc(specfunc(name, tdeg), prod)') then
            blocks := map(B -> [op(B)], [op(T)]);
        else
            error "Grobx:-Basis takes the orders tdeg, plex, lexdeg, and prod of tdeg orders, not %1", T;
        end if;
        if blocks = [] or member([], blocks) then
            error "Grobx:-Basis takes no empty block: %1", T;
        end if;
        if nops(blocks) = 1 then
            return "grevlex", blocks[1];
        end if;
        return cat("block:", StringTools:-Join(map(B -> convert(nops(B), string), blocks[1 .. -2]), ",")),
               [seq(op(B), B in blocks)];
    end proc;

    # What grobx wrote to standard error, its "grobx: " prefixes and the
    # path of the instance taken out.
    Message := proc(err, instance)
        local s;
        if not FileTools:-Exists(err) then
            return "";
        end if;
        # ReadFile of an empty file returns no string.
        s := FileTools:-Text:-ReadFile(err);
        if not type(s, string) then
            return "";
        end if;
        s := StringTools:-Trim(s);
        s := StringTools:-SubstituteAll(s, cat(instance, ": "), "");
        StringTools:-SubstituteAll(s, "grobx: ", "")
    end proc;

    Basis := proc(F::{list, set}, T,
                  {characteristic::nonnegint := 0,
                   output::identical(basis, extended) := basis,
                   threads::{nonnegint, identical(FAIL)} := FAIL,
                   seed::{nonnegint, identical(FAIL)} := FAIL,
                   deadline::{nonnegint, identical(infinity)} := infinity,
                   memory::{nonnegint, identical(infinity)} := infinity,
                   checker::{identical(false, exact), posint} := false}, $)
        local order, vars, n, i, f, new, forward, back, text, verbose, dir, instance, basis, cofactors,
              report, err, argv, status, message, stops, G, C, s;
        order, vars := OrderOf(T);
        n := nops(vars);
        if n = 0 then
            error "Grobx:-Basis takes an order on at least one variable, not %1", T;
        end if;
        if nops({op(vars)}) < n then
            error "Grobx:-Basis takes each variable once: %1", T;
        end if;
        if characteristic <> 0 and not (characteristic < 2^64 and isprime(characteristic)) then
            error "Grobx:-Basis computes over Q and over GF(p) for a prime p below 2^64, not in characteristic %1",
                  characteristic;
        end if;
        for f in F do
            if not type(f, polynom(rational, vars)) then
                error "Grobx:-Basis takes polynomials in %1 with rational coefficients; %2 is none", vars, f;
            end if;
        end do;
        if seed <> FAIL and seed >= 2^64 then
            error "Grobx:-Basis takes a seed below 2^64, not %1", seed;
        end if;
        if deadline <> infinity and deadline >= 2^64 then
            error "Grobx:-Basis takes a deadline below 2^64 milliseconds, not %1", deadline;
        end if;
        if memory <> infinity and memory >= 2^64 then
            error "Grobx:-Basis takes a memory ceiling below 2^64 bytes, not %1", memory;
        end if;
        if type(checker, posint) and checker >= 2^32 then
            error "Grobx:-Basis takes the checker's points below 2^32, not %1", checker;
        end if;

        new := [seq(cat(`_grobx`, i), i = 1 .. n)];
        forward := {seq(vars[i] = new[i], i = 1 .. n)};
        back := {seq(new[i] = vars[i], i = 1 .. n)};
        text := cat(StringTools:-Join(map(convert, new, string), ","), "\n", characteristic, "\n",
                    StringTools:-Join([seq(convert(expand(eval(f, forward)), string), f in F)], ",\n"), "\n");
        verbose := evalb(`if`(assigned(infolevel['Grobx']), infolevel['Grobx'], 0) >= 1);

        # A directory of the call's own, named by the process and the call:
        # TemporaryFilename gives every new session the same names. Only the
        # files named here are removed from it, and then the empty directory.
        calls := calls + 1;
        dir := FileTools:-JoinPath([FileTools:-TemporaryDirectory(), cat("grobx-", kernelopts(pid), "-", calls)]);
        FileTools:-MakeDirectory(dir);
        instance, basis, cofactors, report, err := seq(FileTools:-JoinPath([dir, s]),
            s in ["instance.ms", "basis.txt", "cofactors.txt", "report.txt", "stderr.txt"]);
        try
            FileTools:-Text:-WriteFile(instance, text);
            argv := ["--order", order, "--output", basis];
            if threads <> FAIL then
                argv := [op(argv), "--threads", convert(threads, string)];
            end if;
            if seed <> FAIL then
                argv := [op(argv), "--seed", convert(seed, string)];
            end if;
            if deadline <> infinity then
                argv := [op(argv), "--deadline", convert(deadline, string)];
            end if;
            if memory <> infinity then
                argv := [op(argv), "--memory", convert(memory, string)];
            end if;
            if checker <> false then
                argv := [op(argv), "--checker", convert(checker, string)];
            end if;
            if output = extended then
                argv := [op(argv), "--cofactors", cofactors];
            end if;
            if verbose then
                argv := [op(argv), "--report", report];
            end if;
            argv := [op(argv), "--", instance];
            status := ssystem(Command([Program(), op(argv)], err))[1];
            message := Message(err, instance);
            if status = 1 then
                stops := table(["stopped: deadline" = "the deadline", "stopped: memory" = "the memory ceiling",
                                "stopped: cancel" = "an interrupt"]);
                s := select(l -> assigned(stops[l]), StringTools:-Split(message, "\n"));
                error "Grobx:-Basis: %1 stopped the call", `if`(s = [], message, stops[s[1]]);
            elif status = 3 then
                error "Grobx:-Basis: the checker rejected the basis: %1", message;
            elif status = 2 then
                error "grobx: %1", message;
            elif status <> 0 then
                error "Grobx:-Basis: %1 exited with status %2: %3", Program(), status, message;
            end if;
            if verbose then
                printf("%s\n", message);
                printf("%s", FileTools:-Text:-ReadFile(report));
            end if;
            G := eval(parse(StringTools:-TrimRight(FileTools:-Text:-ReadFile(basis))[1 .. -2]), back);
            if output = extended then
                C := eval(parse(StringTools:-TrimRight(FileTools:-Text:-ReadFile(cofactors))[1 .. -2]), back);
            end if;
        finally
            for s in [instance, basis, cofactors, report, err] do
                if FileTools:-Exists(s) then
                    FileTools:-Remove(s);
                end if;
            end do;
            FileTools:-RemoveDirectory(dir);
        end try;
        if output = extended then
            return G, C;
        end if;
        G
    end proc;
end module:
