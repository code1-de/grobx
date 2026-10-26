//! text-copy-peak, the item's test code through the Python module
//! (`iterations/0081-text-copy-peak/plan.md`, _Test code_, 3 to 5): basis
//! texts of every length around a power of two from a page to 2 MiB, long
//! cofactor texts, and the edge cases, each the text the core returns,
//! byte for byte, and accepted by the checker.

use std::fmt::Write;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use grobx_checker::{Mode, Order, check};
use grobx_core::{MonomialOrder, Params, Threads};

use crate::systems::Rng;
use crate::wrapper::python;

/// The degree of each generator's leading term, past every other term's.
const D: u32 = 70;

/// The scratch directory of the test named `name`, made empty.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("text-copy-peak")
        .join(name);
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The exponents of x, y, z of total degree below `D`, in decreasing
/// grevlex order: by degree, then the smaller exponent of the last
/// variable that differs first.
fn lower() -> Vec<[u32; 3]> {
    let mut all = Vec::new();
    for a in 0..D {
        for b in 0..D - a {
            for c in 0..D - a - b {
                all.push([a, b, c]);
            }
        }
    }
    all.sort_by(|e, f| {
        let (de, df) = (e.iter().sum::<u32>(), f.iter().sum::<u32>());
        df.cmp(&de).then_with(|| {
            let i = (0..3).rev().find(|&i| e[i] != f[i]).unwrap();
            e[i].cmp(&f[i])
        })
    });
    all
}

/// The monomial of exponents `e` on x, y, z as the canonical text spells
/// it, empty for 1.
fn monomial(e: [u32; 3]) -> String {
    let mut out = Vec::new();
    for (name, k) in ["x", "y", "z"].into_iter().zip(e) {
        match k {
            0 => {}
            1 => out.push(name.to_string()),
            _ => out.push(format!("{name}^{k}")),
        }
    }
    out.join("*")
}

/// A polynomial's canonical text: `v^D` for the variable at `lead`, then
/// each of `terms`, a coefficient and a monomial, in their order.
fn poly(lead: usize, terms: &[(u64, [u32; 3])]) -> String {
    let mut e = [0; 3];
    e[lead] = D;
    let mut out = monomial(e);
    for &(c, e) in terms {
        let m = monomial(e);
        if m.is_empty() {
            write!(out, "+{c}").unwrap();
        } else {
            write!(out, "+{c}*{m}").unwrap();
        }
    }
    out
}

/// The monomials of `lower` and the length of their text in a
/// polynomial: per monomial a `+`, a coefficient of one digit, and the
/// monomial with its `*`, summed over the first n, for each n.
struct Lower {
    exponents: Vec<[u32; 3]>,
    prefix: Vec<usize>,
}

impl Lower {
    fn new() -> Lower {
        let exponents = lower();
        let mut prefix = vec![0];
        for &e in &exponents {
            let m = monomial(e).len();
            let term = 2 + if m == 0 { 0 } else { m + 1 };
            prefix.push(prefix.last().unwrap() + term);
        }
        Lower { exponents, prefix }
    }

    /// The basis text's length where each generator's terms past the
    /// leading one take `terms` bytes: the leading monomials, the terms,
    /// the brackets, and the separators.
    fn text(terms: usize) -> usize {
        3 * (monomial([D, 0, 0]).len() + terms) + 1 + 2 * 2 + 3
    }
}

/// An instance over GF(p), or ℚ where p is 0, whose generators are its
/// reduced basis, and that basis's canonical text, `len` bytes long, at
/// least 1024: x^D, y^D, and z^D, each plus the same first n monomials of
/// `lower`, each with a drawn coefficient of one or two digits, both below
/// p, and the coefficients of two digits as many as make up `len`. The
/// leading monomials are pairwise prime and no other term is divisible by
/// one, so the generators are a reduced basis.
fn reduced(lower: &Lower, p: u64, len: usize, seed: u64) -> (String, String) {
    assert!(len >= 1024);
    let n = lower
        .prefix
        .partition_point(|&terms| Lower::text(terms) <= len)
        - 1;
    let wide = len - Lower::text(lower.prefix[n]);
    assert!(wide <= 3 * n, "{len} bytes: {n} terms, {wide} more digits");
    let mut rng = Rng::new(seed);
    let mut polys = Vec::new();
    for v in 0..3 {
        let terms: Vec<(u64, [u32; 3])> = lower.exponents[..n]
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let c = if v * n + i < wide {
                    10 + rng.below(90)
                } else {
                    2 + rng.below(8)
                };
                (c, e)
            })
            .collect();
        polys.push(poly(v, &terms));
    }
    let instance = format!("x,y,z\n{p}\n{polys}\n", polys = polys.join(",\n"));
    // Increasing leading monomials: z^D, y^D, x^D.
    let basis = format!(
        "[{third},\n{second},\n{first}]:\n",
        third = polys[2],
        second = polys[1],
        first = polys[0]
    );
    assert_eq!(basis.len(), len, "the text's length");
    (instance, basis)
}

/// The parameters of the module's calls in the scripts: grevlex, one
/// thread, seed 1, and cofactors where asked.
fn params(cofactors: bool) -> Params {
    let mut p = Params::new(MonomialOrder::Grevlex, Threads::Count(NonZeroUsize::MIN), 1);
    p.cofactors = cofactors;
    p
}

/// Runs each instance of `cases` through the module, seed 1, one thread,
/// with cofactors where asked, and returns its basis text and its
/// cofactor text as Python returned them, written to files in `dir` and
/// read back. The script asserts that each call returned a basis, and
/// that `digest` is SHA-256 of it.
fn through_python(dir: &Path, cases: &[(String, bool)]) -> Vec<(String, Option<String>)> {
    let mut manifest = String::new();
    for (k, (instance, cofactors)) in cases.iter().enumerate() {
        let at = dir.join(format!("{k}.ms"));
        fs::write(&at, instance).unwrap();
        writeln!(
            manifest,
            "{at}\t{out}\t{cofactors}",
            at = at.display(),
            out = dir.join(format!("{k}")).display(),
            cofactors = u8::from(*cofactors)
        )
        .unwrap();
    }
    let list = dir.join("cases.txt");
    fs::write(&list, manifest).unwrap();
    python(&format!(
        r#"
import hashlib
import grobx
for line in open({list:?}).read().splitlines():
    instance, out, cofactors = line.split("\t")
    r = grobx.groebner(open(instance).read(), order="grevlex", threads=1, seed=1, cofactors=cofactors == "1")
    assert r.stopped is None and isinstance(r.basis, str), (instance, r.stopped)
    assert r.digest == hashlib.sha256(r.basis.encode()).hexdigest(), instance
    open(out + ".basis", "w", newline="").write(r.basis)
    if r.cofactors is not None:
        open(out + ".cofactors", "w", newline="").write(r.cofactors)
"#,
        list = list.display().to_string()
    ));
    (0..cases.len())
        .map(|k| {
            let out = dir.join(format!("{k}"));
            let basis = fs::read_to_string(out.with_extension("basis")).unwrap();
            let cofactors = fs::read_to_string(out.with_extension("cofactors")).ok();
            (basis, cofactors)
        })
        .collect()
}

/// Asserts that each case's texts through the module are those the core
/// returns for the same call, and, where the case asks for cofactors,
/// that the checker accepts them in the exact mode.
fn assert_as_the_core(dir: &Path, cases: &[(String, bool)]) {
    let got = through_python(dir, cases);
    for ((instance, cofactors), (basis, cofactor_text)) in cases.iter().zip(got) {
        let outcome = grobx_core::groebner(instance, &params(*cofactors)).unwrap();
        assert!(
            outcome.basis.as_deref() == Some(basis.as_str()),
            "the basis of\n{instance}"
        );
        assert!(
            outcome.cofactors == cofactor_text,
            "the cofactors of\n{instance}"
        );
        if *cofactors {
            let cofactor_text = cofactor_text.unwrap();
            let v = check(
                instance,
                &basis,
                &cofactor_text,
                &Order::Grevlex,
                Mode::Exact,
            )
            .unwrap();
            assert!(
                v.groebner.is_ok() && v.reduced.is_ok(),
                "the checker rejects {v:?} on\n{instance}"
            );
        }
    }
}

/// Test 3, text-copy-peak.1, .3, and .5: over GF(65521), basis texts of
/// 2^k − 1, 2^k, and 2^k + 1 bytes for k from 12 to 21, around a page and
/// around every chunk of a power of two from a page to 2 MiB, 2^18 among
/// them; over ℚ for k from 12 to 19. Each text through the module is the
/// reduced basis, byte for byte.
#[test]
fn basis_texts_of_every_length_through_python() {
    let dir = scratch("lengths");
    let lower = Lower::new();
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for (p, top) in [(65521, 21), (0, 19)] {
        for k in 12..=top {
            for len in [(1 << k) - 1, 1 << k, (1 << k) + 1] {
                let (instance, basis) = reduced(&lower, p, len, len as u64 ^ p);
                cases.push((instance, false));
                want.push(basis);
            }
        }
    }
    let got = through_python(&dir, &cases);
    for ((basis, _), want) in got.iter().zip(&want) {
        assert!(
            basis == want,
            "a text of {wanted} bytes reads {read} bytes",
            wanted = want.len(),
            read = basis.len()
        );
    }
}

/// An instance over GF(p), or ℚ where p is 0, whose cofactor text is long:
/// x^D + 2 + m·(z^D + 3), y^D + 5, and z^D + 3, m the sum of the
/// monomials of degree below `degree` with drawn coefficients of one
/// digit. Its reduced basis is z^D + 3, y^D + 5, and x^D + 2, the first
/// the generator less m times the third.
fn long_cofactors(p: u64, degree: u32, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let mut m = String::new();
    for e in lower()
        .into_iter()
        .filter(|e| e.iter().sum::<u32>() < degree)
    {
        let c = 2 + rng.below(8);
        let mono = monomial(e);
        let factor = if mono.is_empty() {
            String::new()
        } else {
            format!("*{mono}")
        };
        write!(m, "+{c}{factor}*z^{D}+{three_c}{factor}", three_c = 3 * c).unwrap();
    }
    format!("x,y,z\n{p}\nx^{D}+2{m},\ny^{D}+5,\nz^{D}+3\n")
}

/// Test 4, text-copy-peak.1 and .8: cofactor texts past 2^20 bytes over
/// GF(65521) and past 2^18 over ℚ through the module are those the core
/// returns, and the checker accepts them with the basis.
#[test]
fn long_cofactor_texts_through_python() {
    let dir = scratch("cofactors");
    let cases = vec![
        (long_cofactors(65521, D, 1), true),
        (long_cofactors(0, 56, 2), true),
    ];
    let got = through_python(&dir, &cases);
    assert!(
        got[0].1.as_ref().unwrap().len() > 1 << 20,
        "{bytes} bytes",
        bytes = got[0].1.as_ref().unwrap().len()
    );
    assert!(
        got[1].1.as_ref().unwrap().len() > 1 << 18,
        "{bytes} bytes",
        bytes = got[1].1.as_ref().unwrap().len()
    );
    assert_as_the_core(&dir, &cases);
}

/// Test 5, text-copy-peak.5: the edge cases of `OBJECTIVES.md` over
/// GF(65521) and ℚ, with cofactors and without: no generator, F = (0),
/// F = (0, 0), the unit ideal, input already a Gröbner basis, the zero
/// polynomial among the generators, and an eliminable variable. Each text
/// through the module is the core's, and the checker accepts it.
#[test]
fn the_edge_cases_through_python() {
    let dir = scratch("edges");
    let mut cases = Vec::new();
    for p in [65521, 0] {
        for body in [
            "",
            "0",
            "0,\n0",
            "x*y-1,\nx,\ny",
            "x^2+y,\ny^3+1",
            "x^2-y,\n0,\nx*y-1",
            "x-y^2-z,\ny^2-z^3+1,\nz^2-y",
        ] {
            let instance = format!("x,y,z\n{p}\n{body}\n");
            cases.push((instance.clone(), false));
            cases.push((instance, true));
        }
    }
    assert_as_the_core(&dir, &cases);
}
