//! fglm-lex, sub-item 1: the change of order in `fglm.rs`, called directly, against
//! bigatti-2025's printed cases, round trips, F4 in lex, the division
//! multiplied out, and the checker on the lifted cofactors (plan.md, _Test
//! code_, on generated systems in place of the plan's instances, and at the
//! formats' degree bounds as monomial-time set them).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::Budget;
use crate::cofactors::{Cofactors, NoCofactors, Tracked};
use crate::f4::{self, Elements, Halt};
use crate::fglm::{self, Staircase};
use crate::field::{Field, Gf31};
use crate::id::MonoId;
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::{Block, Grevlex, Lex, Order};
use crate::poly::Poly;
use crate::systems::{dense, random_square, random_system};
use crate::table::Table;
use crate::text::{Header, write_basis};
use crate::trace::{NoLearner, NoTrace};
use crate::{Cancel, MonomialOrder, Params, Stats, Stop, Threads};

/// The primes of the random systems.
pub(crate) const PRIMES: [u64; 5] = [2, 3, 7, 65521, 2_147_483_647];

/// Seeds per prime and generator.
const SEEDS: u64 = 20;

/// The parameters of a call in grevlex, with a deadline where given.
fn params(deadline: Option<Duration>) -> Params {
    let mut p = Params::new(
        MonomialOrder::Grevlex,
        Threads::Count(NonZeroUsize::new(1).unwrap()),
        1,
    );
    p.deadline = deadline;
    p
}

fn budget(deadline: Option<Duration>) -> Budget {
    Budget::new(Instant::now(), &params(deadline)).unwrap()
}

/// A call's parts: the field, the monomial table, and the variable names.
struct Call {
    text: String,
    field: Gf31,
    table: Table<Dense16>,
    names: Vec<String>,
}

impl Call {
    fn new(text: &str) -> Call {
        let header = Header::read(text).unwrap();
        let field = header.gf31().unwrap();
        let n = header.names.len();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        Call {
            text: text.to_string(),
            field,
            table: Table::new(Dense16::new(n), &mut rng),
            names: header
                .names
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
        }
    }

    fn n(&self) -> usize {
        self.names.len()
    }

    /// The generators in `order`.
    fn generators<O: Order<Dense16>>(&mut self, order: &O) -> Vec<Poly<Gf31, Rank1>> {
        let header = Header::read(&self.text).unwrap();
        header
            .generators(&self.field, &mut self.table, order)
            .unwrap()
    }

    /// The reduced basis in `order` by F4, f4-gfp's engine, with the
    /// cofactors `cof` tracks; None where the deadline or an overflow
    /// stopped it.
    fn f4<O: Order<Dense16>, C: Cofactors<Gf31, Dense16>>(
        &mut self,
        order: &O,
        cof: &mut C,
        deadline: Option<Duration>,
    ) -> Option<Elements<Gf31, Rank1, C::Element>> {
        let generators = self.generators(order);
        let mut stats = Stats::default();
        let got = f4::run(
            &self.field,
            &mut self.table,
            order,
            generators,
            cof,
            &mut NoTrace,
            &mut budget(deadline),
            &mut stats,
            &mut NoLearner,
            1,
        );
        match got {
            Ok(elements) => Some(elements),
            Err(Halt::Stop(Stop::Deadline) | Halt::Overflow) => None,
            Err(e) => panic!("{e:?}\n{text}", text = self.text),
        }
    }

    /// The reduced grevlex basis, with no cofactors.
    fn grevlex(&mut self) -> Vec<Poly<Gf31, Rank1>> {
        let elements = self.f4(&Grevlex, &mut NoCofactors, None).unwrap();
        elements.into_iter().map(|(g, ())| g).collect()
    }

    fn leads(polys: &[Poly<Gf31, Rank1>]) -> Vec<MonoId> {
        polys.iter().map(super::poly::Poly::lead).collect()
    }

    /// The staircase of `polys`, a reduced basis in `order`.
    fn staircase<O: Order<Dense16>>(
        &mut self,
        order: &O,
        polys: &[Poly<Gf31, Rank1>],
    ) -> Option<Staircase> {
        fglm::staircase(
            &mut self.table,
            order,
            &Call::leads(polys),
            &mut budget(None),
        )
        .unwrap()
    }

    /// The reduced basis in `target` from `polys`, the reduced basis in
    /// `source`, by the change, with no cofactors; None where the staircase
    /// is None.
    fn change<O1: Order<Dense16>, O2: Order<Dense16>>(
        &mut self,
        source: &O1,
        target: &O2,
        polys: &[Poly<Gf31, Rank1>],
    ) -> Option<Vec<Poly<Gf31, Rank1>>> {
        let staircase = self.staircase(source, polys)?;
        let elements = polys.iter().map(|g| (g.clone(), ())).collect();
        let changed = fglm::change(
            &self.field,
            &mut self.table,
            source,
            target,
            &staircase,
            elements,
            &mut NoCofactors,
            &mut budget(None),
        )
        .unwrap();
        Some(changed.into_iter().map(|(g, ())| g).collect())
    }

    /// A monomial's exponents.
    fn exponents(&self, m: MonoId) -> Vec<u32> {
        (0..self.n()).map(|i| self.table.exponent(m, i)).collect()
    }

    /// A polynomial as its terms, exponents and coefficient, in its order.
    fn terms(&self, g: &Poly<Gf31, Rank1>) -> Vec<(Vec<u32>, u64)> {
        g.monomials
            .iter()
            .zip(&g.coefs)
            .map(|(&m, &c)| (self.exponents(m), self.field.value(c)))
            .collect()
    }

    fn all_terms(&self, polys: &[Poly<Gf31, Rank1>]) -> Vec<Vec<(Vec<u32>, u64)>> {
        polys.iter().map(|g| self.terms(g)).collect()
    }

    fn text_of(&self, polys: &[Poly<Gf31, Rank1>]) -> String {
        let names: Vec<&str> = self.names.iter().map(std::string::String::as_str).collect();
        let mut out = String::new();
        write_basis(
            &mut out,
            &names,
            &self.table,
            &self.field,
            polys.iter(),
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        out
    }
}

/// The monomial with these exponents, interned.
fn monomial(call: &mut Call, e: &[u32]) -> MonoId {
    call.table.intern(e).unwrap()
}

#[test]
fn zero_dimensional_by_the_pure_powers() {
    let cases: &[(&[&[u32]], bool)] = &[
        (&[&[2, 0], &[0, 1]], true),
        (&[&[1, 0], &[0, 3]], true),
        (&[&[0, 0]], true),
        (&[&[1, 1]], false),
        (&[&[2, 0]], false),
        (&[&[2, 0], &[1, 1]], false),
        (&[], false),
    ];
    for &(leads, want) in cases {
        let mut call = Call::new("x,y\n65521\nx\n");
        let ids: Vec<MonoId> = leads.iter().map(|e| monomial(&mut call, e)).collect();
        assert_eq!(fglm::zero_dimensional(&call.table, &ids), want, "{leads:?}");
    }
}

// fglm-lex.23 cited: bigatti-2025 Example 38, Example 41, Example 46, and §2.7 -- a staircase and its border, two multiplication matrices over GF(101), and a DRL basis changed to lex, each printed with every coefficient
/// bigatti-2025 Examples 38 and 41: the staircase and the border of
/// G = {xy² − y, y⁴ − x², x³ − y³} in `DegRevLex` over GF(101).
#[test]
fn the_staircase_of_example_38() {
    let mut call = Call::new("x,y\n101\nx*y^2-y,\ny^4-x^2,\nx^3-y^3\n");
    let g = call.grevlex();
    assert_eq!(
        call.text_of(&g),
        "[x*y^2+100*y,\nx^3+100*y^3,\ny^4+100*x^2]:\n"
    );
    let st = call.staircase(&Grevlex, &g).unwrap();
    assert_eq!(st.len(), 8);
    let b: Vec<Vec<u32>> = st.monomials().iter().map(|&m| call.exponents(m)).collect();
    assert_eq!(
        b,
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 2],
            [1, 1],
            [2, 0],
            [0, 3],
            [2, 1]
        ]
    );
    let border: Vec<Vec<u32>> = st.border().iter().map(|&m| call.exponents(m)).collect();
    assert_eq!(border, [[1, 2], [3, 0], [0, 4], [1, 3], [2, 2], [3, 1]]);
}

/// bigatti-2025 Example 46: the normal forms of the border of
/// G = {y² + 34x + y + 2, x² + xy + 2y} over B = [1, y, x, xy], the columns
/// of M⁽¹⁾ and M⁽²⁾ that are not unit vectors. A basis with a tail
/// monomial outside B is a misfit.
#[test]
fn the_multiplication_table_of_example_46() {
    let mut call = Call::new("x,y\n101\ny^2+34*x+y+2,\nx^2+x*y+2*y\n");
    let g = call.grevlex();
    let st = call.staircase(&Grevlex, &g).unwrap();
    let b: Vec<Vec<u32>> = st.monomials().iter().map(|&m| call.exponents(m)).collect();
    assert_eq!(b, [[0, 0], [0, 1], [1, 0], [1, 1]]);
    let border: Vec<Vec<u32>> = st.border().iter().map(|&m| call.exponents(m)).collect();
    assert_eq!(border, [[0, 2], [2, 0], [1, 2], [2, 1]]);
    let forms =
        fglm::border_forms(&call.field, &mut call.table, &st, &g, &mut budget(None)).unwrap();
    let dense = |k: usize| {
        let mut v = [0u64; 4];
        for (&c, &a) in forms[k].cols.iter().zip(&forms[k].coefs) {
            v[c.index()] = call.field.value(a);
        }
        v
    };
    // y² = y·w₂, M⁽²⁾ column 2: −2w₁ − w₂ − 34w₃
    assert_eq!(dense(0), [99, 100, 67, 0]);
    // x² = x·w₃, M⁽¹⁾ column 3: −2w₂ − w₄
    assert_eq!(dense(1), [0, 99, 0, 100]);
    // xy² = y·w₄, M⁽²⁾ column 4: −33w₂ − 2w₃ + 33w₄
    assert_eq!(dense(2), [0, 68, 99, 33]);
    // x²y = x·w₄, M⁽¹⁾ column 4: 4w₁ + 35w₂ − 31w₃ − 33w₄
    assert_eq!(dense(3), [4, 35, 70, 68]);

    // y², the first element's leading monomial, in the second's tail.
    let mut unreduced = Call::new("x,y\n101\ny^2+34*x+y+2,\nx^2+x*y+y^2+2*y\n");
    let g = unreduced.generators(&Grevlex);
    let st = unreduced.staircase(&Grevlex, &g).unwrap();
    let got = fglm::border_forms(
        &unreduced.field,
        &mut unreduced.table,
        &st,
        &g,
        &mut budget(None),
    );
    assert!(
        matches!(got, Err(Halt::Misfit(0))),
        "{err:?}",
        err = got.err()
    );
}

/// bigatti-2025 §2.7: Example 46's basis changed from DRL to lex.
#[test]
fn the_toy_example() {
    let mut call = Call::new("x,y\n101\ny^2+34*x+y+2,\nx^2+x*y+2*y\n");
    let g = call.grevlex();
    let lex = call.change(&Grevlex, &Lex, &g).unwrap();
    assert_eq!(
        call.text_of(&lex),
        "[y^4+69*y^3+72*y^2+26*y+4,\nx+3*y^2+3*y+6]:\n"
    );
}

/// In `Dense16` the staircase holds at most `Dense16::MAX_DEGREE`
/// monomials, and a larger one is `Halt::Overflow` (monomial-formats.10).
#[test]
fn the_staircase_stops_past_the_degree_bound() {
    let mut call = Call::new("x,y\n65521\nx^128,\ny^256\n");
    let g = call.grevlex();
    let leads = Call::leads(&g);
    let got = fglm::staircase(&mut call.table, &Grevlex, &leads, &mut budget(None));
    assert!(matches!(got, Err(Halt::Overflow)), "D = 32768");
    let mut call = Call::new("x,y\n65521\nx^151,\ny^217\n");
    let g = call.grevlex();
    let st = call.staircase(&Grevlex, &g).unwrap();
    assert_eq!(st.len(), 32767);
    assert_eq!(st.len() as u32, Dense16::MAX_DEGREE);
}

#[test]
fn not_zero_dimensional_has_no_staircase() {
    for text in [
        "x,y\n65521\nx*y-1\n",
        "x,y\n65521\nx^2\n",
        "x,y,z\n65521\nx*y-z,\ny^2-x\n",
        "x,y\n65521\n0\n",
    ] {
        let mut call = Call::new(text);
        let g = call.grevlex();
        assert!(call.staircase(&Grevlex, &g).is_none(), "{text:?}");
    }
}

/// The unit ideal, D = 0, and a point, D = 1.
#[test]
fn the_unit_ideal_and_a_point() {
    for text in [
        "x,y\n65521\nx*y-1,\nx\n",
        "x,y\n65521\n5\n",
        "x,y\n2\nx,\nx+1\n",
    ] {
        let mut call = Call::new(text);
        let g = call.grevlex();
        let st = call.staircase(&Grevlex, &g).unwrap();
        assert_eq!(st.len(), 0, "{text:?}");
        let lex = call.change(&Grevlex, &Lex, &g).unwrap();
        assert_eq!(call.text_of(&lex), "[1]:\n", "{text:?}");
    }
    let mut call = Call::new("x,y\n65521\nx-3,\ny-5\n");
    let g = call.grevlex();
    assert_eq!(call.staircase(&Grevlex, &g).unwrap().len(), 1);
    let lex = call.change(&Grevlex, &Lex, &g).unwrap();
    assert_eq!(call.text_of(&lex), "[y+65516,\nx+65518]:\n");
}

/// Hand-computed lex bases.
#[test]
fn hand_computed_changes() {
    let cases = [
        ("x,y\n65521\nx^2,\ny^2\n", "[y^2,\nx^2]:\n"),
        ("x,y\n65521\nx^2,\nx*y,\ny^2\n", "[y^2,\nx*y,\nx^2]:\n"),
        (
            "x,y\n65521\nx-y^2,\ny^3-1\n",
            "[y^3+65520,\nx+65520*y^2]:\n",
        ),
        ("x,y\n65521\nx^2+y,\nx*y+1\n", "[y^3+1,\nx+65520*y^2]:\n"),
        (
            "x,y,z\n65521\nx^20-y,\ny^20-z,\nz^20-x\n",
            "[z^8000+65520*z,\ny+65520*z^400,\nx+65520*z^20]:\n",
        ),
        ("x\n65521\nx^3-1,\nx^2-1\n", "[x+65520]:\n"),
    ];
    for (text, want) in cases {
        let mut call = Call::new(text);
        let g = call.grevlex();
        let lex = call.change(&Grevlex, &Lex, &g).unwrap();
        assert_eq!(call.text_of(&lex), want, "{text:?}");
    }
}

/// The instances of the round trips and the comparisons.
fn systems() -> Vec<(String, String)> {
    let mut list = Vec::new();
    for p in PRIMES {
        for seed in 0..SEEDS {
            list.push((format!("square-p{p}-{seed}"), random_square(p, seed)));
            list.push((format!("random-p{p}-{seed}"), random_system(p, seed)));
        }
    }
    list
}

/// Grevlex to lex to grevlex, and grevlex to `block:1` to grevlex, give the
/// grevlex basis back, where the ideal is zero-dimensional.
#[test]
fn round_trips() {
    let mut changed = 0;
    for (name, text) in systems() {
        let mut call = Call::new(&text);
        let g = call.grevlex();
        let Some(lex) = call.change(&Grevlex, &Lex, &g) else {
            continue;
        };
        let back = call.change(&Lex, &Grevlex, &lex).unwrap();
        assert_eq!(
            call.all_terms(&back),
            call.all_terms(&g),
            "{name}: lex\n{text}"
        );
        if call.n() >= 2 {
            let block = Block::new(&[1], call.n()).unwrap();
            let there = call.change(&Grevlex, &block, &g).unwrap();
            let back = call.change(&block, &Grevlex, &there).unwrap();
            assert_eq!(
                call.all_terms(&back),
                call.all_terms(&g),
                "{name}: block:1\n{text}"
            );
        }
        changed += 1;
    }
    assert!(changed >= 100, "only {changed} systems were changed");
}

/// The change gives F4's lex basis, where F4 in lex finishes within
/// 100 ms.
#[test]
fn the_change_is_f4_in_lex() {
    let mut zero_dimensional = 0;
    let mut compared = 0;
    for (name, text) in systems() {
        let mut call = Call::new(&text);
        let g = call.grevlex();
        let Some(lex) = call.change(&Grevlex, &Lex, &g) else {
            continue;
        };
        zero_dimensional += 1;
        let mut other = Call::new(&text);
        let Some(f4) = other.f4(&Lex, &mut NoCofactors, Some(Duration::from_millis(100))) else {
            continue;
        };
        let f4: Vec<Poly<Gf31, Rank1>> = f4.into_iter().map(|(g, ())| g).collect();
        assert_eq!(call.all_terms(&lex), other.all_terms(&f4), "{name}\n{text}");
        compared += 1;
    }
    assert!(
        2 * compared >= zero_dimensional,
        "only {compared} of {zero_dimensional} compared"
    );
}

/// For every element g of every lex basis, Σ `q_u·g_u` over the quotients of
/// its division by the grevlex basis is g.
#[test]
fn the_division_multiplied_out() {
    for (name, text) in systems() {
        let mut call = Call::new(&text);
        let g = call.grevlex();
        let Some(lex) = call.change(&Grevlex, &Lex, &g) else {
            continue;
        };
        let p = call.field.characteristic();
        for h in &lex {
            let quotients = fglm::divide(
                &call.field,
                &mut call.table,
                &Grevlex,
                h,
                &g,
                &mut budget(None),
            )
            .unwrap();
            let mut sum: HashMap<Vec<u32>, u64> = HashMap::new();
            for (u, q) in &quotients {
                for &(s, a) in q {
                    let s = call.exponents(s);
                    for (t, b) in call.terms(&g[*u as usize]) {
                        let st: Vec<u32> = s.iter().zip(&t).map(|(x, y)| x + y).collect();
                        let a = u128::from(call.field.value(a));
                        let e = sum.entry(st).or_insert(0);
                        *e = ((u128::from(*e) + a * u128::from(b)) % u128::from(p)) as u64;
                    }
                }
            }
            sum.retain(|_, c| *c != 0);
            let want: HashMap<Vec<u32>, u64> = call.terms(h).into_iter().collect();
            assert_eq!(
                sum,
                want,
                "{name}: {terms:?}\n{text}",
                terms = call.terms(h)
            );
        }
    }
}

/// With `Tracked`, the lifted cofactors of the changed basis express it in
/// the instance's generators, and the checker accepts the result in lex.
#[test]
fn the_lifted_cofactors() {
    let mut cases: Vec<(String, String)> = [
        "x,y\n101\ny^2+34*x+y+2,\nx^2+x*y+2*y\n",
        "x,y\n65521\nx^2+y,\nx*y+1\n",
        "x,y\n65521\nx*y-1,\nx\n",
        "x,y\n65521\nx-3,\ny-5\n",
        "x,y,z\n65521\nx-y^2-z,\ny*z-1,\nz^2-y\n",
        "x,y,z\n65521\nx^20-y,\ny^20-z,\nz^20-x\n",
    ]
    .iter()
    .enumerate()
    .map(|(k, t)| (format!("hand-{k}"), t.to_string()))
    .collect();
    for m in [2, 3] {
        cases.push((format!("cubic-{m}"), dense(m, 3, 65521, m as u64)));
    }
    for p in PRIMES {
        for seed in 0..10 {
            cases.push((format!("square-p{p}-{seed}"), random_square(p, seed)));
        }
    }
    let mut lifted = 0;
    for (name, text) in cases {
        let mut call = Call::new(&text);
        let generators = call.generators(&Grevlex).len();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let mut cof = Tracked::new(call.field, Dense16::new(call.n()), generators, &mut rng);
        let elements = call.f4(&Grevlex, &mut cof, None).unwrap();
        let polys: Vec<Poly<Gf31, Rank1>> = elements.iter().map(|(g, _)| g.clone()).collect();
        let Some(st) = call.staircase(&Grevlex, &polys) else {
            continue;
        };
        let changed = fglm::change(
            &call.field,
            &mut call.table,
            &Grevlex,
            &Lex,
            &st,
            elements,
            &mut cof,
            &mut budget(None),
        )
        .unwrap();
        let (polys, rows): (Vec<_>, Vec<_>) = changed.into_iter().unzip();
        let basis = call.text_of(&polys);
        let names: Vec<&str> = call.names.iter().map(std::string::String::as_str).collect();
        let mut cofactors = String::new();
        cof.write(
            &mut cofactors,
            &names,
            &Lex,
            &rows,
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        let verdict = grobx_checker::check(
            &text,
            &basis,
            &cofactors,
            &grobx_checker::Order::Lex,
            grobx_checker::Mode::Exact,
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}\n{basis}\n{cofactors}"));
        assert_eq!(verdict.groebner, Ok(()), "{name}\n{text}\n{basis}");
        assert_eq!(verdict.reduced, Ok(()), "{name}\n{text}\n{basis}");
        lifted += 1;
    }
    assert!(lifted >= 50, "only {lifted} cases lifted");
}

/// A change checks the budget at its start: a raised cancel flag stops it.
#[test]
fn a_cancel_stops_the_change() {
    let mut call = Call::new(&dense(3, 3, 65521, 1));
    let g = call.grevlex();
    let st = call.staircase(&Grevlex, &g).unwrap();
    let mut p = params(None);
    let cancel = Cancel::new();
    cancel.set();
    p.cancel = Some(cancel);
    let mut stopped = Budget::new(Instant::now(), &p).unwrap();
    let elements = g.iter().map(|h| (h.clone(), ())).collect();
    let got = fglm::change(
        &call.field,
        &mut call.table,
        &Grevlex,
        &Lex,
        &st,
        elements,
        &mut NoCofactors,
        &mut stopped,
    );
    assert!(
        matches!(got, Err(Halt::Stop(Stop::Cancel))),
        "{got:?}",
        got = got.map(|e| e.len())
    );
}
