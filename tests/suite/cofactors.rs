//! Sub-item 3: cofactor tracking (f4-gfp.23, f4-gfp.25). The checker in
//! the exact mode is the oracle: every row of the cofactor matrix must
//! express its element of the basis in the generators.

use grobx::cofactors::NoCofactors;
use grobx_checker::Mode;

use crate::common::{PRIMES, assert_accepted, params, random_system, run, with_cofactors};

/// The rows of a cofactor text, each as its entries' texts.
fn rows(text: &str) -> Vec<Vec<String>> {
    let inner = text
        .trim_end()
        .strip_prefix('[')
        .and_then(|t| t.strip_suffix("]:"))
        .unwrap_or_else(|| panic!("not a bracketed matrix: {text:?}"));
    let mut out = Vec::new();
    let mut depth = 0;
    let mut row = String::new();
    for c in inner.chars() {
        match c {
            '[' => {
                depth += 1;
                row.clear();
            }
            ']' => {
                depth -= 1;
                out.push(row.split(',').map(|e| e.trim().to_string()).collect());
            }
            _ if depth == 1 => row.push(c),
            _ => {}
        }
    }
    out
}

fn elements(basis: &str) -> usize {
    let inner = basis
        .trim_end()
        .trim_start_matches('[')
        .trim_end_matches("]:");
    if inner.is_empty() {
        0
    } else {
        inner.split(',').count()
    }
}

#[test]
fn no_cofactors_is_zero_sized() {
    assert_eq!(std::mem::size_of::<NoCofactors>(), 0);
}

#[test]
fn cofactors_come_where_asked() {
    let text = "x,y\n65521\nx^2+y,\nx*y+1\n";
    assert_eq!(run(text, &params(1)).cofactors, None);
    let outcome = run(text, &with_cofactors(1));
    let matrix = outcome
        .cofactors
        .as_deref()
        .expect("cofactors were asked for");
    let rows = rows(matrix);
    assert_eq!(rows.len(), elements(outcome.basis.as_deref().unwrap()));
    assert!(rows.iter().all(|r| r.len() == 2), "{matrix}");
    assert!(matrix.ends_with("]:\n"), "{matrix:?}");
    assert_accepted("tiny", text, &outcome, Mode::Exact);
}

#[test]
fn a_zero_generator_has_a_zero_column() {
    let text = "x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n";
    let outcome = run(text, &with_cofactors(1));
    let rows = rows(outcome.cofactors.as_deref().unwrap());
    assert_ne!(rows.len(), 0, "the cofactor text's rows");
    for row in &rows {
        assert_eq!(row.len(), 3, "{row:?}");
        assert_eq!(row[1], "0", "the zero generator's entry: {row:?}");
    }
    assert_accepted("zero-generator", text, &outcome, Mode::Exact);
}

#[test]
fn the_zero_and_the_unit_ideal() {
    for text in ["x,y\n65521\n0\n", "x,y\n65521\n0,\n0\n", "x,y\n65521\n"] {
        let outcome = run(text, &with_cofactors(1));
        assert_eq!(outcome.basis.as_deref(), Some("[]:\n"));
        assert_eq!(outcome.cofactors.as_deref(), Some("[]:\n"));
        assert_accepted("zero", text, &outcome, Mode::Exact);
    }
    for text in [
        "x,y\n65521\nx*y-1,\nx\n",
        "x\n2\nx,\nx+1\n",
        "x,y\n65521\n5\n",
    ] {
        let outcome = run(text, &with_cofactors(1));
        assert_eq!(outcome.basis.as_deref(), Some("[1]:\n"));
        assert_accepted(text, text, &outcome, Mode::Exact);
    }
}

#[test]
fn hand_cases_are_accepted() {
    for text in [
        "x,y\n65521\nx+y,\ny^2+1\n",
        "x,y\n65521\n2*x^2+2*y,\nx*y+1,\n3*x*y+3\n",
        "x\n65521\nx^3-1,\nx^2-1\n",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        "x,y,z\n2\nx*y+z,\nx*z+y,\ny*z+x\n",
        "x,y,z\n2147483647\nx^2+123456789*y*z+1,\ny^2-x*z,\nz^2+x*y+2\n",
        "x,y\n7\n1/2*x^2+1/3*y,\nx*y-1/5\n",
    ] {
        assert_accepted(text, text, &run(text, &with_cofactors(1)), Mode::Exact);
    }
}

#[test]
fn random_systems_are_accepted() {
    for p in PRIMES {
        for seed in 0..12 {
            let text = random_system(p, 100 + seed);
            assert_accepted(
                &format!("random p{p} {seed}"),
                &text,
                &run(&text, &with_cofactors(1)),
                Mode::Exact,
            );
        }
    }
}

/// The same inputs under the same seed compute the same thing
/// (`OBJECTIVES.md`, _Call parameters_).
#[test]
fn the_same_seed_gives_the_same_cofactors() {
    let text = random_system(65521, 7);
    let a = run(&text, &with_cofactors(42));
    let b = run(&text, &with_cofactors(42));
    assert_eq!(a, b);
    let c = run(&text, &with_cofactors(43));
    assert_eq!(a.basis, c.basis);
}
