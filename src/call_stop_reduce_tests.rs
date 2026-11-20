//! call-stop-latency, sub-item 3's test code: the reductions' dense scans, the back-substitution in
//! bands, the inter-reduction, the setup's sort, and the signature reduction's scan of a run each
//! stop a call from inside, and a back-substitution in bands spends as it goes
//! (`iterations/0138-call-stop-latency/plan.md`, _Test code_, `call_stop_reduce_tests.rs`).

use crate::Stop;
use crate::back_substitution_tests::reduce;
use crate::band_tests::Random;
use crate::budget::{Budget, Site};
use crate::call_stop_sites_tests::{Case, spends_as_it_goes, stops_at};
use crate::field::Gf31;
use crate::halt::Halt;
use crate::matrix::Buffers;
use crate::systems::{boolean, random_sparse, random_system};

/// The reductions' sites, each with a call that reaches it: random systems over GF(31), whose
/// back-substitution takes bands, and GF(65521); a boolean system over GF(2); a sparse system over
/// GF(65521) without cofactors, whose signature rounds test runs of zero rows.
pub(crate) fn cases() -> Vec<(Site, Case)> {
    vec![
        (
            Site::Scan,
            Case::new(random_system(65521, 5), "grevlex", false),
        ),
        (
            Site::BackBands,
            Case::new(random_system(31, 6), "grevlex", false),
        ),
        (
            Site::Inter,
            Case::new(random_system(31, 1), "grevlex", false),
        ),
        (Site::Setup, Case::new(boolean(12, 16, 2), "grevlex", false)),
        (
            Site::SigScan,
            Case::new(random_sparse(4, 65521, 4), "grevlex", false),
        ),
    ]
}

/// Test 1: every site of the reductions stops a call from inside it, the call's result unchanged
/// with every spend checking (call-stop-latency.16 to .18).
#[test]
fn every_site_of_the_reductions_stops() {
    let mut failures = Vec::new();
    for (site, case) in cases() {
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 2, call-stop-latency.16, .17: a reduction of a matrix of hundreds of columns whose made
/// rows need back-substitution, in bands, spends its sweeps as it goes and stops inside under a
/// passed deadline.
#[test]
fn a_back_substitution_in_bands_spends() {
    let m = Random::of(Gf31::new(7).unwrap(), 1, (40, 900, 0), (30, 7));
    let back = |b: &mut Budget| -> Result<usize, Stop> {
        match reduce(&m, true, true, b, &mut Buffers::new()) {
            Ok(made) => Ok(made.len()),
            Err(Halt::Stop(s)) => Err(s),
            Err(
                halt @ (Halt::Overflow
                | Halt::Size
                | Halt::Misfit(_)
                | Halt::Handover
                | Halt::Outworked),
            ) => panic!("the reduction: {halt:?}"),
        }
    };
    if let Err(e) = spends_as_it_goes(Site::BackBands, 1 << 10, 4, back) {
        panic!("{e}");
    }
}
