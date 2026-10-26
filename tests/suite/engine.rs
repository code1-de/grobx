//! Sub-item 2: the F4 engine through `groebner`, without cofactors. Its
//! bases are compared with hand-computed texts and with msolve's on random
//! systems (f4-gfp.41).

use grobx::{Error, MonomialOrder};

use crate::common::{P, PRIMES, basis, dense, msolve, params, random_system, run, sha256};

const TINY: &str = "x,y\n65521\nx^2+y,\nx*y+1\n";

#[test]
fn the_tiny_instance() {
    assert_eq!(basis(TINY), "[y^2+65520*x,\nx*y+1,\nx^2+y]:\n");
}

#[test]
fn the_zero_ideal() {
    for text in [
        "x,y\n65521\n0\n",
        "x,y\n7\n0,\n0\n",
        "x,y\n65521\n",
        "x\n2\nx+x\n",
    ] {
        assert_eq!(basis(text), "[]:\n", "{text:?}");
    }
}

#[test]
fn the_unit_ideal() {
    for text in [
        "x,y\n65521\nx*y-1,\nx\n",
        "x\n2\nx,\nx+1\n",
        "x,y\n65521\n5\n",
        "x,y\n65521\nx^2+y,\nx*y+1,\n7\n",
        "x,y,z\n65521\nx*y-1,\nx*z,\ny*z-1,\nz^2\n",
    ] {
        assert_eq!(basis(text), "[1]:\n", "{text:?}");
    }
}

#[test]
fn input_already_a_groebner_basis() {
    assert_eq!(basis("x,y\n65521\nx+y,\ny^2+1\n"), "[x+y,\ny^2+1]:\n");
    assert_eq!(basis("x,y\n65521\ny^2+1,\nx+y\n"), "[x+y,\ny^2+1]:\n");
    // a reduced basis is its own reduced basis
    let text = dense(3, 3, P, 1);
    let reduced = basis(&text);
    let names = text.lines().next().unwrap();
    let elements = reduced
        .trim_end()
        .trim_start_matches('[')
        .trim_end_matches("]:");
    let again = format!("{names}\n{P}\n{elements}\n");
    assert_eq!(basis(&again), reduced);
}

/// msolve 0.7.5 returns a wrong basis where a generator is 0 (log 0004-1,
/// step 5), so msolve sees the instance without it.
#[test]
#[ignore = "calls msolve"]
fn a_zero_generator_changes_nothing() {
    let with = basis("x,y,z\n65521\nx*y-z,\n0,\ny^2-x\n");
    let without = basis("x,y,z\n65521\nx*y-z,\ny^2-x\n");
    assert_eq!(with, without);
    assert_eq!(
        Some(without.clone()),
        msolve("engine-zero-generator", "x,y,z\n65521\nx*y-z,\ny^2-x\n")
    );
}

#[test]
#[ignore = "calls msolve"]
fn an_eliminable_variable() {
    for (name, text) in [
        (
            "engine-eliminable-first",
            "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        ),
        (
            "engine-eliminable-last",
            "y,z,x\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        ),
        (
            "engine-eliminable-linear",
            "x,y,z\n65521\nx-2*y+3*z-5,\nx*y-z^2,\ny^3-x*z+1\n",
        ),
    ] {
        assert_eq!(Some(basis(text)), msolve(name, text), "{name}");
    }
}

#[test]
fn duplicates_and_scalar_multiples() {
    let want = basis(TINY);
    assert_eq!(basis("x,y\n65521\nx^2+y,\nx^2+y,\nx*y+1\n"), want);
    assert_eq!(basis("x,y\n65521\n2*x^2+2*y,\nx*y+1,\n3*x*y+3\n"), want);
    assert_eq!(basis("x,y\n65521\nx*y+1,\nx^2+y\n"), want);
}

#[test]
fn one_variable_gives_the_gcd() {
    assert_eq!(basis("x\n65521\nx^3-1,\nx^2-1\n"), "[x+65520]:\n");
    assert_eq!(basis("x\n7\nx^4-1,\nx^6-1\n"), "[x^2+6]:\n");
    assert_eq!(basis("x\n65521\nx^5+x+1\n"), "[x^5+x+1]:\n");
}

#[test]
#[ignore = "calls msolve"]
fn small_primes_and_the_largest() {
    for (name, text) in [
        ("engine-p2", "x,y,z\n2\nx*y+z,\nx*z+y,\ny*z+x\n"),
        ("engine-p3", "x,y,z\n3\nx^2+y*z+1,\ny^2+x*z,\nz^2+x*y+2\n"),
        ("engine-p7", "x,y,z\n7\nx^3-y*z,\ny^3-x*z,\nz^3-x*y\n"),
        (
            "engine-p31",
            "x,y,z\n2147483647\nx^2+123456789*y*z+1,\ny^2-x*z,\nz^2+x*y+2\n",
        ),
    ] {
        assert_eq!(Some(basis(text)), msolve(name, text), "{name}");
    }
}

/// Random systems over the five primes, against msolve. msolve refuses the
/// zero ideal, which a system is where the terms of each generator cancel;
/// the count of the systems compared leaves room for those.
#[test]
#[ignore = "calls msolve"]
fn random_systems_agree_with_msolve() {
    let mut compared = 0;
    for p in PRIMES {
        for seed in 0..30 {
            let text = random_system(p, seed);
            let name = format!("engine-random-p{p}-{seed}");
            let ours = basis(&text);
            if let Some(theirs) = msolve(&name, &text) {
                assert_eq!(ours, theirs, "{name}:\n{text}");
                compared += 1;
            }
        }
    }
    assert!(
        compared >= 140,
        "msolve took only {compared} of 150 systems"
    );
}

/// Coprime leading monomials, whose lcm x^20000·y^20000 `Dense16` cannot
/// hold: the plain run's product criterion drops the pair, and the
/// signature run the S-pair, whose Koszul signature it cannot hold either.
/// The input is its own basis. `formats.rs` holds x^20000·y − 1 and
/// x·y^20000 − 1, whose pair is not coprime (monomial-formats.7).
#[test]
fn a_coprime_pair_past_the_format() {
    for signatures in [true, false] {
        let mut p = params(1);
        p.signatures = signatures;
        let outcome = run("x,y\n65521\nx^20000-y,\ny^20000-x\n", &p);
        assert_eq!(
            outcome.basis.as_deref(),
            Some("[y^20000+65520*x,\nx^20000+65520*y]:\n"),
            "signatures {signatures}"
        );
    }
}

#[test]
fn what_the_engine_refuses() {
    let order = |name: &str| name.parse::<MonomialOrder>();
    assert!(matches!(order("deglex"), Err(Error::Order { .. })));
    assert!(matches!(order("degrevlex"), Err(Error::Order { .. })));
    // 0, Q, is taken since q-lift; 4 is composite, and 2^31 is no prime.
    for p in ["4", "2147483648"] {
        let text = format!("x,y\n{p}\nx^2+y,\nx*y+1\n");
        assert!(
            matches!(
                grobx::groebner(&text, &params(1)),
                Err(Error::Characteristic { .. })
            ),
            "characteristic {p}"
        );
    }
    assert!(matches!(
        grobx::groebner("x\n7\nx+y\n", &params(1)),
        Err(Error::UnknownVariable { .. })
    ));
}

#[test]
fn the_digest_is_sha256() {
    for text in ["", "[]:\n", "[1]:\n", &basis(TINY)] {
        assert_eq!(grobx::digest(text), sha256(text));
    }
}
