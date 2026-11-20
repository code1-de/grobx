//! The divisor search of the plain run's symbolic preprocessing: of the
//! members of `G_min` whose leading monomials divide a monomial, the one of
//! fewest terms; past `FILED_PAST` variables over `G_min`'s members filed
//! by a bit of their leads' supports.

#[cfg(test)]
use std::cell::Cell;

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::field::Field;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::{Format, SUPPORT_BITS};
use crate::narrow::Narrow;
use crate::poly::Basis;
use crate::sort::{RUN, sort_runs};
use crate::table::{Table, find_passing, search_units};

#[cfg(test)]
thread_local! {
    // gf2-symbolic-time.6 design-checked -- in the test profile a thread-local fixes whether the divisor search files G_min's members, so that test code compares the files and the scan on one G_min and one call at any count of variables, not a parameter of the call nor formats padded past FILED_PAST variables (not measured)
    /// The test profile's choice on this thread, where Some: whether
    /// `Divisors` files `G_min`'s members, in place of the variables'
    /// count (`filed`).
    pub(crate) static FILED_FORCED: Cell<Option<bool>> = const { Cell::new(None) };
}

// gf2-symbolic-time.4 tuned-checked -- the members are filed past 64 variables, where every bit of the divisor mask stands for two variables or more, not past the mask's 32 bits, where the calls of 33 to 60 variables took longer on the whole at the same instructions, their searches finding a divisor early in the scan, and the toric calls' files mispredicting their branches (log 0176-1, 0176-2)
/// The variables past which `Divisors` files `G_min`'s members
/// (gf2-symbolic-time.4): two to each bit of the divisor mask.
pub(crate) const FILED_PAST: usize = 2 * u32::BITS as usize;

/// Whether `Divisors` files `G_min`'s members on a format of n variables:
/// past `FILED_PAST`. In the test profile `FILED_FORCED` decides where it
/// holds a choice.
fn filed(n: usize) -> bool {
    let past = n > FILED_PAST;
    #[cfg(test)]
    let past = FILED_FORCED.get().unwrap_or(past);
    past
}

// mask-scan-time.1 design-checked -- G_min stands in increasing order of its elements' terms, ties in the order made, so that the first divisor the search finds is the rule's and the search stops there, not every candidate tested in the order made and the one of fewest terms kept (log 0073-1)
// mask-scan-time.2 design-checked -- the order is made at each round by a sort of G_min, not kept in the basis as elements join and leave G_min (not measured)
/// `G_min`'s leading monomials with their masks, in increasing order of
/// their elements' terms, ties in the order made, for the divisor search
/// of symbolic preprocessing; past `FILED_PAST` variables, their places
/// filed by a bit of their supports.
pub(crate) struct Divisors {
    element: Vec<ElementId>,
    lead: Vec<MonoId>,
    mask: Vec<u32>,
    files: Option<Files>,
}

impl Divisors {
    // call-stop-latency.14 design-checked -- the divisor search of symbolic preprocessing spends a unit per block of G_min's masks a search may test, and the signature run's reducer search per block of the basis's, a search that finds none as one that finds one, not unspent (matrix-stop-latency.9), where stretches of searches that found none went 0.38 s between two checks on PHP(8,7) (log 0138-1)
    /// `G_min`'s order for the search, spent at `Site::Divisors`: the sort
    /// as `sort_runs` spends, and the leads and their masks `budget::PROBE` units
    /// an element, `RUN` at a time; past `FILED_PAST` variables the files
    /// as `Files::of` spends. A stop where the budget's check stops the
    /// call.
    pub(crate) fn new<F: Field, M: Format, D: Module>(
        table: &Table<M>,
        basis: &Basis<F, D>,
        budget: &mut Budget,
    ) -> Result<Divisors, Stop> {
        Divisors::of(table, basis, basis.min().to_vec(), budget)
    }

    /// As `new`, over the members `element` of `G_min` alone.
    pub(crate) fn of<F: Field, M: Format, D: Module>(
        table: &Table<M>,
        basis: &Basis<F, D>,
        mut element: Vec<ElementId>,
        budget: &mut Budget,
    ) -> Result<Divisors, Stop> {
        // The index is the order made, which tells every two elements apart.
        sort_runs(
            &mut element,
            |&a, &b| (basis.length(a), a).cmp(&(basis.length(b), b)),
            budget,
            Site::Divisors,
        )?;
        let (mut lead, mut mask) = (
            Vec::with_capacity(element.len()),
            Vec::with_capacity(element.len()),
        );
        for chunk in element.chunks(RUN) {
            lead.extend(chunk.iter().map(|&i| basis.lead(i)));
            mask.extend(chunk.iter().map(|&i| table.mask(basis.lead(i))));
            budget.spend(chunk.len() * PROBE, Site::Divisors)?;
        }
        let files = if filed(table.format().variables()) {
            Some(Files::of(table, &lead, &mask, budget)?)
        } else {
            None
        };
        Ok(Divisors {
            element,
            lead,
            mask,
            files,
        })
    }

    // f4-gfp.5 design-checked -- where several rows can be a column's pivot, the one of fewest terms is, among the reducers whose leading monomial divides the column's and among the pair rows of one leading column, ties to the element made first and the first pair row, not the first divisor in the order of G and the first pair row (not measured)
    // matrix-build-time.8 design-checked -- G_min's masks are tested eight at a time before the divisibility test runs on the candidates in order, not one mask test per element (log 0037-1)
    // table-read-time.9 design-checked -- the formed signature's factors, the reducer search's entries, the known syzygy signatures' ids, and the plain run's divisor search keep their bounds checks, not read unchecked as well (log 0045-1)
    // mask-scan-time.3 design-checked -- a search that finds no divisor leaves no record, not per monomial the basis's length at that search, so that its next search scans only the elements made since (log 0043-1, 0073-1)
    // mask-scan-time.4 design-checked -- the search tests G_min's masks, then the candidates' divisibility, not per variable and exponent a bit set of the elements whose exponent is at most it, ANDed over the variables where the monomial's exponent lies below the largest (log 0073-1)
    /// The element of `G_min` with the fewest terms whose leading monomial
    /// divides u, ties to the element made first, None where none does; and
    /// the units the search spends: a unit per block of masks it may test
    /// and `PROBE` units a candidate whose divisibility it tested
    /// (`table::search_units`), and past `FILED_PAST` variables a unit per
    /// file it visits besides (`Files::find`).
    pub(crate) fn find<M: Format>(
        &self,
        table: &Table<M>,
        u: MonoId,
    ) -> (Option<ElementId>, usize) {
        // gf2-symbolic-time.8 design-checked -- the scan and the filed search are functions of their own, each with its own count of the candidates tested, not one closure shared by both, which on the calls the files leave executed more instructions a search than the scan alone (log 0176-2)
        match &self.files {
            Some(files) => self.find_filed(files, table, u),
            None => self.scan(table, u),
        }
    }

    /// `find` past `FILED_PAST` variables: the files of `files` that
    /// `Files::find` visits.
    fn find_filed<M: Format>(
        &self,
        files: &Files,
        table: &Table<M>,
        u: MonoId,
    ) -> (Option<ElementId>, usize) {
        let words = table.monomial(u);
        let mut tested = 0;
        let (found, visited) = files.find(table.format().support(words), table.mask(u), &mut |k| {
            tested += 1;
            self.divides(table, k, words)
        });
        (
            found.map(|k| self.element[k]),
            visited + search_units(0, tested),
        )
    }

    /// `find` up to `FILED_PAST` variables: the scan of every mask of
    /// `G_min`.
    fn scan<M: Format>(&self, table: &Table<M>, u: MonoId) -> (Option<ElementId>, usize) {
        let words = table.monomial(u);
        let mut tested = 0;
        let found = find_passing(&self.mask, table.mask(u), |k| {
            tested += 1;
            self.divides(table, k, words)
        })
        .map(|k| self.element[k]);
        (found, search_units(self.mask.len(), tested))
    }

    /// Whether the lead at place k divides the monomial of `words`.
    fn divides<M: Format>(&self, table: &Table<M>, k: usize, words: &[M::Word]) -> bool {
        table.format().divides(table.monomial(self.lead[k]), words)
    }
}

/// The key of the members whose leads have no variable: the constant
/// alone, whose file every search visits (gf2-symbolic-time.3).
const EMPTY: usize = SUPPORT_BITS;

// gf2-symbolic-time.1 transferred-checked: information retrieval, inverted index -- carried into the divisor search: past FILED_PAST variables G_min's places stand in files, each place under one bit of its lead's support, and a search visits the files of its monomial's support bits alone, not the scan of every mask (matrix-build-time.8), whose masks, a bit for several variables, pass many members whose leads do not divide, and a search that finds no divisor reads every mask of G_min (log 0176-1, 0176-2)
// gf2-symbolic-time.5 design-checked -- each file's masks are tested eight at a time by the scan of the masks (matrix-build-time.8), not each place's support tested exactly, which over sat's files of 60 variables executed more instructions a place than the scan spends a mask (log 0176-1, 0176-2)
/// `G_min`'s places in the search order filed by key
/// (gf2-symbolic-time.1): each under one bit of its lead's support, the
/// bit the fewest of the leads hold (.2), and a place whose lead has no
/// variable under `EMPTY` (.3). Every member whose lead divides a monomial
/// lies in a file of a bit of the monomial's support or in `EMPTY`'s, so
/// the files of those bits hold the search's answer.
struct Files {
    /// Per key, from 0 to `EMPTY`, the start of its file in `places` and
    /// `masks`, and after them the end of `EMPTY`'s.
    starts: Vec<u32>,
    /// The files' places, back to back by key, each file's increasing.
    places: Vec<u32>,
    /// `masks[e]`, the mask of the lead at place `places[e]`.
    masks: Vec<u32>,
}

impl Files {
    /// The files of the places of `lead`, whose masks `mask` holds, by key:
    /// a counting sort by key, stable in the places' order. Spent at
    /// `Site::Divisors`, `RUN` places at a time: a lead's support its words
    /// and `PROBE` units, a unit per bit of a support counted and per bit
    /// its key reads, and `PROBE` units a place filed. A stop where the
    /// budget's check stops the call.
    fn of<M: Format>(
        table: &Table<M>,
        lead: &[MonoId],
        mask: &[u32],
        budget: &mut Budget,
    ) -> Result<Files, Stop> {
        debug_assert_eq!(lead.len(), mask.len(), "a mask per lead");
        let format = table.format();
        let w = format.words();
        // The leads that hold each bit.
        let mut held = [0u32; SUPPORT_BITS];
        let mut supports = Vec::with_capacity(lead.len());
        for chunk in lead.chunks(RUN) {
            let mut bits_read = 0;
            for &l in chunk {
                let support = format.support(table.monomial(l));
                let mut bits = support;
                while bits != 0 {
                    held[bits.trailing_zeros() as usize] += 1;
                    bits &= bits - 1;
                    bits_read += 1;
                }
                supports.push(support);
            }
            budget.spend(chunk.len() * (PROBE + w) + bits_read, Site::Divisors)?;
        }
        // starts[key + 1] counts the places of the key, then the prefix
        // sums make it the end of the key's file.
        let mut starts = vec![0u32; EMPTY + 2];
        let mut keys: Vec<u8> = Vec::with_capacity(lead.len());
        for chunk in supports.chunks(RUN) {
            let mut bits_read = 0;
            for &support in chunk {
                let (k, read) = key(support, &held);
                starts[k + 1] += 1;
                keys.push(k.narrow());
                bits_read += read;
            }
            budget.spend(chunk.len() + bits_read, Site::Divisors)?;
        }
        for k in 0..=EMPTY {
            starts[k + 1] += starts[k];
        }
        let mut next = starts.clone();
        let mut places = vec![0u32; lead.len()];
        let mut masks = vec![0u32; lead.len()];
        for (c, chunk) in keys.chunks(RUN).enumerate() {
            for (j, &k) in chunk.iter().enumerate() {
                let place = c * RUN + j;
                let at = &mut next[usize::from(k)];
                places[*at as usize] = place.narrow();
                masks[*at as usize] = mask[place];
                *at += 1;
            }
            budget.spend(chunk.len() * PROBE, Site::Divisors)?;
        }
        Ok(Files {
            starts,
            places,
            masks,
        })
    }

    // gf2-symbolic-time.7 design-checked -- a filed search spends a unit per file it visits and per block of the file's masks, beside PROBE units a candidate, not the units of the scan of all of G_min's masks (call-stop-latency.14), which it does not read (log 0176-1, 0176-2)
    /// The least place, in the search order, among those filed under the
    /// bits of `support`, a monomial's, and under `EMPTY`, whose lead
    /// `divides` the monomial; None where none does. Each file's masks are
    /// tested against the monomial's `mask` eight at a time, and `divides`
    /// runs on the candidates in the file's order: the file ends at its
    /// first divisor and at a place past the least found. And the units of
    /// the files it visits: a unit a file and a unit per block of its masks
    /// (`table::search_units`).
    fn find(
        &self,
        support: u128,
        mask: u32,
        divides: &mut impl FnMut(usize) -> bool,
    ) -> (Option<usize>, usize) {
        let mut best: Option<usize> = None;
        let mut units = 0;
        let mut visit = |key: usize| {
            let (lo, hi) = (self.starts[key] as usize, self.starts[key + 1] as usize);
            units += 1 + search_units(hi - lo, 0);
            let places = &self.places[lo..hi];
            find_passing(&self.masks[lo..hi], mask, |j| {
                let k = places[j] as usize;
                if best.is_some_and(|b| k > b) {
                    return true;
                }
                let found = divides(k);
                if found {
                    best = Some(k);
                }
                found
            });
        };
        visit(EMPTY);
        let mut bits = support;
        while bits != 0 {
            visit(bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
        (best, units)
    }
}

// gf2-symbolic-time.2 design-checked -- a place's key is the bit of its lead's support that the fewest of G_min's leads hold, ties to the lowest, not the lowest bit, so that the bits most leads hold, which most monomials searched hold too, have short files (log 0176-1, 0176-2)
// gf2-symbolic-time.3 design-checked -- a lead of empty support, the constant, is filed under a key of its own that every search visits, not under bit 0, which a search of a monomial without variable 0 would pass by (log 0176-1, 0176-2)
/// The key of a lead of support `support`, where `held[b]` of the leads
/// hold bit b: the bit of `support` the fewest hold, ties to the lowest,
/// and `EMPTY` for an empty support; and the bits it read.
fn key(support: u128, held: &[u32; SUPPORT_BITS]) -> (usize, usize) {
    let mut best = EMPTY;
    let mut read = 0;
    let mut bits = support;
    while bits != 0 {
        let b = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        read += 1;
        if best == EMPTY || held[b] < held[best] {
            best = b;
        }
    }
    (best, read)
}
