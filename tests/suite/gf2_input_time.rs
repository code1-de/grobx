//! gf2-input-time, the item's test code in the suite: the engine reads
//! texts of many variables, whose names take several shapes and whose
//! terms name their variables in any order, to the generators the checker
//! reads; and it refuses a repeated name in the header and an unknown name
//! in a generator as the checker does, with the name
//! (`iterations/0158-gf2-input-time/plan.md`, _Test code_, 5 and 6).

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// A draw of `rng` below `n`.
fn below(rng: &mut Xoshiro256PlusPlus, n: usize) -> usize {
    (rng.next_u64() % n as u64) as usize
}

/// The names of n variables in the shape `shape`: `x1` on; letters in
/// base 26 after a `v`; a prefix chain `a`, `a1`, `a11`, …; names with
/// underscores; and names of mixed case.
fn names(shape: usize, n: usize) -> Vec<String> {
    (0..n)
        .map(|i| match shape {
            0 => format!("x{i}", i = i + 1),
            1 => {
                let mut s = String::from("v");
                let mut k = i;
                loop {
                    s.push(char::from(b'a' + (k % 26) as u8));
                    k /= 26;
                    if k == 0 {
                        break s;
                    }
                }
            }
            2 => format!("a{ones}", ones = "1".repeat(i)),
            3 => format!("_{i}_x"),
            _ => format!("Xy{i}Z"),
        })
        .collect()
}

/// A random text over p on `names`: up to eight generators of up to twelve
/// terms, each term a sign, at times a numeric factor, and up to four
/// powers of drawn names, a name at times twice, with exponents 0 to 3,
/// written with or without `^1` and spaces around the `*`.
fn text(names: &[String], p: u64, rng: &mut Xoshiro256PlusPlus) -> String {
    let mut gens = Vec::new();
    for _ in 0..=below(rng, 8) {
        let mut g = String::new();
        for t in 0..=below(rng, 12) {
            g.push_str(if below(rng, 2) == 0 { "+" } else { "-" });
            let mut factors = Vec::new();
            if below(rng, 2) == 0 || t == 0 {
                factors.push((1 + below(rng, 9)).to_string());
            }
            for _ in 0..below(rng, 5) {
                let name = &names[below(rng, names.len())];
                factors.push(match below(rng, 5) {
                    0 => format!("{name}^0"),
                    1 => format!("{name}^1"),
                    2 => format!("{name}^{e}", e = 2 + below(rng, 2)),
                    3 => format!("{name}*{name}"),
                    _ => name.clone(),
                });
            }
            if factors.is_empty() {
                factors.push("1".to_string());
            }
            let join = if below(rng, 4) == 0 { " * " } else { "*" };
            g.push_str(&factors.join(join));
        }
        gens.push(g);
    }
    format!(
        "{names}\n{p}\n{gens}\n",
        names = names.join(","),
        gens = gens.join(",\n")
    )
}

/// Test 5, gf2-input-time.1 to .5: over 2, 7, and 65521, random texts on
/// 1 to 300 variables of each shape, the counts around the growths of the
/// engine's index of the names among them, and texts of like terms written
/// apart and of degrees past every format: the engine's generators are the
/// checker's, or both refuse the text with an error of one kind.
#[test]
fn many_names_read_as_the_checker_reads_them() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0158);
    let mut texts = Vec::new();
    for shape in 0..5 {
        for n in [1, 2, 7, 8, 9, 15, 16, 17, 33, 64, 65, 300] {
            let names = names(shape, n);
            for p in [2, 7, 65521] {
                for _ in 0..3 {
                    texts.push(text(&names, p, &mut rng));
                }
            }
        }
    }
    for g in [
        "x*y+y*x",
        "x*y+y*x+x^1*y^1",
        "y*x^0*x+x*y*y^0",
        "x*x^2*x^0-x^3",
        "x^0",
        "x^0*y^0+1",
        "x^2147483647*y",
        "x^2147483647*x",
        "x^4294967295*x^5",
        "y*x^2147483648",
    ] {
        for p in [2, 3, 7] {
            texts.push(format!("x,y\n{p}\n{g}\n"));
        }
    }
    for t in &texts {
        let engine = grobx::text::generators(t).map_err(|e| kind(&format!("{e:?}")));
        let checker = grobx_checker::generators(t).map_err(|e| kind(&format!("{e:?}")));
        assert_eq!(engine, checker, "{t}");
    }
}

/// The kind of an error by its name: the text of its variant up to the
/// first brace or parenthesis.
fn kind(debug: &str) -> String {
    debug
        .split(['{', '(', ' '])
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Test 6, gf2-input-time.2 to .5: a header that repeats a name at its
/// place 2 to 300, after names of each shape, is refused as a repeated
/// variable with that name by the engine and the checker; and a generator
/// that names a known name with a byte more, a byte less, or another case
/// is refused as an unknown variable, by the engine at the generator's
/// line with that name.
#[test]
fn refusals_as_the_checker_refuses_them() {
    for shape in 0..5 {
        for n in [2, 8, 9, 16, 17, 33, 300] {
            let mut names = names(shape, n);
            let repeated = names[(n - 1) / 2].clone();
            names[n - 1] = repeated.clone();
            let t = format!("{names}\n7\n1\n", names = names.join(","));
            let got = grobx::text::generators(&t);
            assert!(
                matches!(&got, Err(grobx::Error::RepeatedVariable { name }) if *name == repeated),
                "shape {shape}, {n} names: {got:?}"
            );
            assert!(
                matches!(
                    grobx_checker::generators(&t),
                    Err(grobx_checker::Error::RepeatedVariable { .. })
                ),
                "shape {shape}, {n} names: the checker refuses the repeat"
            );
        }
        let names = names(shape, 40);
        let known = &names[17];
        let unknown = [
            format!("{known}q"),
            known[..known.len() - 1].to_string(),
            known.to_uppercase(),
        ];
        for u in unknown
            .iter()
            .filter(|u| !names.contains(u) && !u.is_empty())
        {
            let t = format!(
                "{names}\n7\n{known}+1,\n{known}*{u}+2\n",
                names = names.join(",")
            );
            let got = grobx::text::generators(&t);
            assert!(
                matches!(&got, Err(grobx::Error::UnknownVariable { line: 4, name }) if name == u),
                "{t}: {got:?}"
            );
            assert!(
                matches!(
                    grobx_checker::generators(&t),
                    Err(grobx_checker::Error::UnknownVariable { .. })
                ),
                "{t}: the checker refuses the name"
            );
        }
    }
}
