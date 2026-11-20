//! gfp-unsolved, sub-item 2: the signature run with F4-RB's reducer rows
//! and the queue pruned late gives the plain run's basis, and the checker
//! accepts its cofactors.

use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::Stats;
use crate::budget::Budget;
use crate::cofactors::NoCofactors;
#[cfg(feature = "cofactors")]
use crate::cofactors::{Cofactors, Tracked};
use crate::f4::{self, SigEnd};
use crate::field::Gf31;
#[cfg(feature = "cofactors")]
use crate::halt::Halt;
use crate::module::Rank1;
use crate::monomial::{Dense16, Format};
use crate::order::Grevlex;
use crate::poly::{Elements, Poly};
use crate::shared_tests::budget;
#[cfg(feature = "cofactors")]
use crate::sig::Pairs;
#[cfg(feature = "cofactors")]
use crate::systems::{dense, random_system};
use crate::table::Table;
use crate::text::{Header, Unread, write_basis};
use crate::trace::{NoLearner, NoTrace};

/// The basis of a signature run without the handover, which carries
/// nothing.
pub(crate) fn sig_basis<E>(end: SigEnd<Gf31, Rank1, E>) -> Elements<Gf31, Rank1, E> {
    match end {
        SigEnd::Basis(elements) => elements,
        SigEnd::Carry(_) => panic!("a signature run without the handover carried"),
    }
}

/// The parts of a call over `text`, in grevlex, in `Dense16`.
pub(crate) struct Call {
    text: String,
    pub(crate) field: Gf31,
    pub(crate) table: Table<Dense16>,
}

impl Call {
    pub(crate) fn new(text: &str) -> Call {
        let header = Header::read(text, &mut Budget::unbounded()).unwrap();
        let field = header.gf31().unwrap();
        let table = Table::new(
            Dense16::new(header.names.len()),
            &mut Xoshiro256PlusPlus::seed_from_u64(1),
        );
        Call {
            text: text.to_string(),
            field,
            table,
        }
    }

    pub(crate) fn generators(&mut self) -> Vec<Poly<Gf31, Rank1>> {
        let header = Header::read(&self.text, &mut Budget::unbounded()).unwrap();
        header
            .generators(
                &self.field,
                &mut self.table,
                &Grevlex,
                &mut Budget::unbounded(),
            )
            .map_err(Unread::unbounded)
            .unwrap()
    }

    /// The signature run, without the handover.
    #[cfg(feature = "cofactors")]
    fn sig<C: Cofactors<Gf31, Dense16>>(
        &mut self,
        cof: &mut C,
    ) -> Result<Elements<Gf31, Rank1, C::Element>, Halt> {
        let generators = self.generators();
        f4::run_sig(
            &self.field,
            &mut self.table,
            &Grevlex,
            generators,
            cof,
            &mut NoTrace,
            &mut budget(),
            &mut Stats::default(),
            &mut NoLearner,
            None,
            &mut Pairs::new(true),
            1,
        )
        .map(sig_basis)
    }

    /// The plain run's reduced basis.
    pub(crate) fn plain(&mut self) -> String {
        let generators = self.generators();
        let elements = f4::run(
            &self.field,
            &mut self.table,
            &Grevlex,
            generators,
            &mut NoCofactors,
            &mut NoTrace,
            &mut budget(),
            &mut Stats::default(),
            &mut NoLearner,
            1,
        )
        .unwrap();
        self.text_of(&elements)
    }

    pub(crate) fn text_of<E>(&self, elements: &Elements<Gf31, Rank1, E>) -> String {
        let header = Header::read(&self.text, &mut Budget::unbounded()).unwrap();
        let mut out = String::new();
        write_basis(
            &mut out,
            &header.names,
            &self.table,
            &self.field,
            elements.iter().map(|(g, _)| g),
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        out
    }
}

/// gfp-unsolved.12, .13: the signature run alone, reducer rows reduced to
/// new elements and to syzygy signatures, gives the plain run's basis, and
/// the checker accepts its cofactors in the exact mode.
#[cfg(feature = "cofactors")]
#[test]
fn reducer_rows_reduce_to_elements_and_syzygies() {
    let mut cases: Vec<(String, String)> = (2..=3)
        .map(|m| (format!("cubic-{m}"), dense(m, 3, 65521, 1)))
        .collect();
    for seed in 0..6 {
        cases.push((format!("random-{seed}"), random_system(65521, seed)));
    }
    for (name, text) in &cases {
        let header = Header::read(text, &mut Budget::unbounded()).unwrap();
        let names = header.names.clone();
        let mut call = Call::new(text);
        let generators = call.generators().len();
        let mut cof = Tracked::new(
            call.field,
            Dense16::new(names.len()),
            generators,
            &mut Xoshiro256PlusPlus::seed_from_u64(2),
        );
        let elements = call
            .sig(&mut cof)
            .unwrap_or_else(|h| panic!("{name}: {h:?}"));
        let basis = call.text_of(&elements);
        assert_eq!(
            basis,
            Call::new(text).plain(),
            "{name}: the plain run's basis"
        );
        let rows: Vec<_> = elements.into_iter().map(|(_, row)| row).collect();
        let mut cofactors = String::new();
        cof.write(
            &mut cofactors,
            &names,
            &Grevlex,
            &rows,
            &mut Budget::unbounded(),
        )
        .expect("a budget without bounds never stops");
        let verdict = grobx_checker::check(
            text,
            &basis,
            &cofactors,
            &grobx_checker::Order::Grevlex,
            grobx_checker::Mode::Exact,
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(verdict.groebner, Ok(()), "{name}\n{text}\n{basis}");
        assert_eq!(verdict.reduced, Ok(()), "{name}\n{text}\n{basis}");
    }
}
