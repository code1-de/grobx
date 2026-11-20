//! The sorts of monomials and of rows by a monomial order: keyed where the
//! order has keys, by pieces past `PIECE` keys, in runs where it has none,
//! each spending its work on the budget as it goes.

#[cfg(test)]
use std::cell::Cell;
use std::cmp::Ordering;

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::field::Field;
use crate::id::{MonoId, RowId};
use crate::module::Module;
use crate::monomial::{Format, PAYLOAD_BITS};
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::Poly;
use crate::sig::Sig;
use crate::table::Table;

// threshold-recheck.13 design-checked -- a keyed sort spends KEY, 16, units a key as its keys are built, not none, as building a key reads every exponent word of its element; at these spends no call of run O, the tuning set's instances past 0.1 s but cyclic-9-p65521, returned past 0.1 s after its deadline (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0056-1, 0074-2)
/// Units of work per key a keyed sort builds, and per term a cofactor
/// entry gathers into its sorted order.
pub(crate) const KEY: usize = 16;

// threshold-recheck.12 design-checked -- an element of a keyed sort, or of a sort of at most RUN elements, spends SORTED, 64, units, not one unit as an entry of a row operation spends, as a sort makes about log2 n comparisons an element; at these spends no call of run O, the tuning set's instances past 0.1 s but cyclic-9-p65521, returned past 0.1 s after its deadline (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0056-1, 0074-2)
/// Units of work per element of a keyed sort, and of a sort of at most
/// `RUN` elements by `sort_runs`.
pub(crate) const SORTED: usize = 64;

// matrix-stop-latency.6 design-checked -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
/// Elements per run of `sort_runs`, and per chunk of the work spent `RUN`
/// elements at a time.
pub(crate) const RUN: usize = 1 << 15;

// call-stop-latency.9 design-checked -- a keyed sort of more than PIECE keys sorts by pieces, each pass spent a unit a key RUN keys at a time and each piece sorted as one sort spent after it, not one sort spent after it (matrix-stop-latency.7) nor runs merged pairwise into a buffer, which raised sparse-18-p65521's peak (log 0056-1, 0138-1, 0138-2)
// call-stop-latency.43 tuned-checked -- PIECE is 2^20 keys, not 2^18, the plan's: one sort of 2^20 keys of random monomials went tens of milliseconds, and the splits made the sorts from 2^18 to 2^20 keys slower than one sort each, where the tuning set's largest sorts lie (log 0138-2)
/// Keys past which a keyed sort sorts by pieces (`keys`).
pub(crate) const PIECE: usize = 1 << 20;

#[cfg(test)]
thread_local! {
    /// Where Some, the keys past which `keys` sorts by pieces on this
    /// thread, in place of `PIECE`.
    pub(crate) static PIECE_FORCED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The keys past which `keys` sorts by pieces: `PIECE`, or in the test
/// profile `PIECE_FORCED` where it holds a count.
fn piece_len() -> usize {
    #[cfg(test)]
    if let Some(len) = PIECE_FORCED.get() {
        return len;
    }
    PIECE
}

/// `items` into `out` by increasing `key`, which lies below `keys`, items
/// of one key in their order in `items`: a counting sort, `counts` its
/// scratch. Each pass spends at `site` a unit an item, or a key in the
/// counts' prefix sums, after each `RUN`; a stop where the budget's check
/// stops the call, `out` then in no order. The items number at most
/// `u32::MAX`.
pub(crate) fn counting<T: Copy>(
    items: &[T],
    key: impl Fn(&T) -> usize,
    keys: usize,
    counts: &mut Vec<u32>,
    out: &mut Vec<T>,
    budget: &mut Budget,
    site: Site,
) -> Result<(), Stop> {
    debug_assert!(
        u32::try_from(items.len()).is_ok(),
        "{len} items",
        len = items.len()
    );
    out.clear();
    let Some(&first) = items.first() else {
        return Ok(());
    };
    // counts[k + 1] counts key k, and after the prefix sums counts[k] is
    // the first place of key k in `out`.
    counts.clear();
    counts.resize(keys + 1, 0);
    for chunk in items.chunks(RUN) {
        for item in chunk {
            counts[key(item) + 1] += 1;
        }
        budget.spend(chunk.len(), site)?;
    }
    for start in (1..=keys).step_by(RUN) {
        let end = (start + RUN).min(keys + 1);
        for k in start..end {
            counts[k] += counts[k - 1];
        }
        budget.spend(end - start, site)?;
    }
    out.resize(items.len(), first);
    for chunk in items.chunks(RUN) {
        for &item in chunk {
            let next = &mut counts[key(&item)];
            out[*next as usize] = item;
            *next += 1;
        }
        budget.spend(chunk.len(), site)?;
    }
    Ok(())
}

// matrix-sort-time.1 cited-checked: bachmann-1998 §4, attributes it to Macaulay [4] -- a monomial is compared by its rank, one integer whose order is the monomial order
// matrix-sort-time.2 practice-checked: decorate-sort-undecorate -- each element of a sort is keyed as the sort starts, and the keys are dropped when it returns, not a key stored in the table per monomial
/// Sorts the monomials `ids` of the table into decreasing order, spent at
/// `site`. In an order without keys the sort is `sort_runs`'s. In a keyed
/// order it is `keyed_decreasing`'s, and the ids written back spend a unit
/// each, `RUN` at a time. A stop where the budget's check stops the call.
pub(crate) fn sort_decreasing<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    ids: &mut [MonoId],
    budget: &mut Budget,
    site: Site,
) -> Result<(), Stop> {
    // matrix-stop-latency.6 design-checked -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
    if !O::KEYED {
        return sort_runs(ids, |&a, &b| table.cmp(order, b, a), budget, site);
    }
    let keyed = keyed_decreasing(table, order, ids, budget, site)?;
    for (chunk, sorted) in ids.chunks_mut(RUN).zip(keyed.chunks(RUN)) {
        for (u, &k) in chunk.iter_mut().zip(sorted) {
            *u = MonoId::from_raw(payload(k));
        }
        budget.spend(chunk.len(), site)?;
    }
    Ok(())
}

/// The monomials `ids` of the table keyed, `Order::key` above each id, and
/// sorted into decreasing order by `keys`, the ties by `ties`, spent at
/// `site`: the keys built `RUN` at a time, `KEY` units each, and the sort
/// spent as `keys` and `ties` spend. A stop where the budget's check stops
/// the call. In a keyed order alone.
pub(crate) fn keyed_decreasing<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    ids: &[MonoId],
    budget: &mut Budget,
    site: Site,
) -> Result<Vec<u128>, Stop> {
    debug_assert!(O::KEYED, "a keyed sort");
    let mut keyed: Vec<u128> = Vec::with_capacity(ids.len());
    for chunk in ids.chunks(RUN) {
        keyed.extend(
            chunk
                .iter()
                .map(|&u| table.key(order, u) | u128::from(u.raw())),
        );
        budget.spend(chunk.len() * KEY, site)?;
    }
    let cmp = |a: u32, b: u32| table.cmp(order, MonoId::from_raw(b), MonoId::from_raw(a));
    keys(&mut keyed, true, budget, site, |piece, budget| {
        ties(
            table,
            order,
            piece,
            true,
            MonoId::from_raw,
            cmp,
            budget,
            site,
        )
    })?;
    Ok(keyed)
}

/// The rows of a matrix whose signatures are `sigs`, in increasing order
/// of signature, the rows of one signature in increasing order. The sort
/// spends as `sort_decreasing`'s does, at `site`, and a stop where the
/// budget's check stops the call.
pub(crate) fn by_signature<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    sigs: &[Sig],
    budget: &mut Budget,
    site: Site,
) -> Result<Vec<RowId>, Stop> {
    let cmp = |a: RowId, b: RowId| {
        sigs[a.index()]
            .cmp(sigs[b.index()], table, order)
            .then(a.cmp(&b))
    };
    // matrix-stop-latency.6 design-checked -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
    if !O::KEYED {
        let mut rows: Vec<RowId> = (0..sigs.len()).map(RowId::at).collect();
        sort_runs(&mut rows, |&a, &b| cmp(a, b), budget, site)?;
        return Ok(rows);
    }
    let mut keyed: Vec<u128> = Vec::with_capacity(sigs.len());
    for (k, chunk) in sigs.chunks(RUN).enumerate() {
        keyed.extend(
            chunk
                .iter()
                .enumerate()
                .map(|(r, s)| table.key(order, s.monomial) | (k * RUN + r) as u128),
        );
        budget.spend(chunk.len() * KEY, site)?;
    }
    let row = |k: u32| RowId::at(k as usize);
    keys(&mut keyed, false, budget, site, |piece, budget| {
        ties(
            table,
            order,
            piece,
            false,
            |r| sigs[row(r).index()].monomial,
            |a, b| cmp(row(a), row(b)),
            budget,
            site,
        )
    })?;
    let mut rows = Vec::with_capacity(keyed.len());
    for chunk in keyed.chunks(RUN) {
        rows.extend(chunk.iter().map(|&k| row(payload(k))));
        budget.spend(chunk.len(), site)?;
    }
    Ok(rows)
}

// call-stop-latency.9 design-checked -- a keyed sort of more than PIECE keys sorts by pieces, each pass spent a unit a key RUN keys at a time and each piece sorted as one sort spent after it, not one sort spent after it (matrix-stop-latency.7) nor runs merged pairwise into a buffer, which raised sparse-18-p65521's peak (log 0056-1, 0138-1, 0138-2)
/// Sorts `v` increasing, or decreasing where `decreasing`, spent at `site`,
/// and hands it to `piece` piece by piece, in order, each piece once it is
/// sorted. `piece` spends its own work. At `PIECE` keys or fewer `v` is one
/// sort, spent `SORTED` units a key after it, and one piece. Past `PIECE`
/// the sort goes by pieces (`by_pieces`): a piece then holds `PIECE` keys or
/// fewer, or keys that agree in every bit above their payloads, and every
/// run of keys that agree above their payloads lies in one piece. No piece
/// is empty. A stop where the budget's check stops the call, `v` then in no
/// order.
pub(crate) fn keys(
    v: &mut [u128],
    decreasing: bool,
    budget: &mut Budget,
    site: Site,
    mut piece: impl FnMut(&mut [u128], &mut Budget) -> Result<(), Stop>,
) -> Result<(), Stop> {
    if v.is_empty() {
        return Ok(());
    }
    by_pieces(v, decreasing, PAYLOAD_BITS, budget, site, &mut piece)
}

// call-stop-latency.38 design-checked -- a piece past PIECE whose keys agree above their payloads is sorted by pieces on its whole keys and handed to the callback whole, not left unsorted for ties, so that keys gives sort_unstable's order and no run of equal keys spans two pieces (not measured)
/// `keys` of `v`, whose keys are told apart by their bits from `shift` up,
/// their part: a piece of `piece_len()` keys or fewer is sorted as one; a
/// larger one is split (`split`) around the part of the median of a sample
/// (`pivot`), the parts no larger than it to one side, and each side taken
/// the same way, in order; where every part is the median's, the piece is
/// one, sorted first on its whole keys where `shift` is not 0. `v` is not
/// empty.
fn by_pieces<F>(
    v: &mut [u128],
    decreasing: bool,
    shift: u32,
    budget: &mut Budget,
    site: Site,
    piece: &mut F,
) -> Result<(), Stop>
where
    F: FnMut(&mut [u128], &mut Budget) -> Result<(), Stop>,
{
    debug_assert!(!v.is_empty(), "a piece holds keys");
    if v.len() <= piece_len() {
        if decreasing {
            v.sort_unstable_by(|a, b| b.cmp(a));
        } else {
            v.sort_unstable();
        }
        budget.spend(v.len() * SORTED, site)?;
        return piece(v, budget);
    }
    let p = pivot(v, shift);
    // The first side takes the parts up to the median's, or from it down
    // where `decreasing`; where that is every key, the parts before the
    // median's alone.
    let ahead = |k: u128, with: bool| match (k >> shift).cmp(&p) {
        Ordering::Less => !decreasing,
        Ordering::Equal => with,
        Ordering::Greater => decreasing,
    };
    let mut at = split(v, |k| ahead(k, true), budget, site)?;
    if at == v.len() {
        at = split(v, |k| ahead(k, false), budget, site)?;
    }
    if at == 0 {
        if shift > 0 {
            by_pieces(v, decreasing, 0, budget, site, &mut no_piece)?;
        }
        return piece(v, budget);
    }
    let (first, rest) = v.split_at_mut(at);
    by_pieces(first, decreasing, shift, budget, site, piece)?;
    by_pieces(rest, decreasing, shift, budget, site, piece)
}

/// A piece of a sort that hands its pieces to nobody: nothing done.
#[allow(
    clippy::unnecessary_wraps,
    reason = "the type of every piece's callback, which may stop"
)]
pub(crate) fn no_piece(_: &mut [u128], _: &mut Budget) -> Result<(), Stop> {
    Ok(())
}

/// Keys of a sample whose median part is a split's pivot (`pivot`).
const SAMPLE: usize = 65;

/// The median of the parts, the bits from `shift` up, of `SAMPLE` keys of
/// `v` taken at even steps, some taken twice where `v` holds fewer. `v` is
/// not empty.
fn pivot(v: &[u128], shift: u32) -> u128 {
    debug_assert!(!v.is_empty(), "a key to sample");
    let step = (v.len() / SAMPLE).max(1);
    let mut sample: [u128; SAMPLE] =
        std::array::from_fn(|i| v[(i * step).min(v.len() - 1)] >> shift);
    sample.sort_unstable();
    sample[SAMPLE / 2]
}

// call-stop-latency.42 practice-checked: quicksort partition -- a piece past PIECE is split in place around the median of a sample, the keys with their parts to one side or the other, not partitioned by the RADIX bits below the keys' common prefix by cycles (call-stop-latency.34), which sorted grevlex keys of random monomials slower than one sort, as their sparse exponents fill few buckets (log 0138-2)
/// Moves the keys of `v` that `ahead` takes before the others, in place,
/// in one pass with no branch on `ahead`, a unit a key spent at `site`
/// after each `RUN` keys: the count `ahead` took. A stop where the budget's
/// check stops the call, `v` then a permutation of its keys.
fn split(
    v: &mut [u128],
    ahead: impl Fn(u128) -> bool,
    budget: &mut Budget,
    site: Site,
) -> Result<usize, Stop> {
    // The keys before `at` are taken, those from `at` to the key looked at
    // are not, and each key looked at changes place with the first not
    // taken.
    let mut at = 0;
    for start in (0..v.len()).step_by(RUN) {
        let end = (start + RUN).min(v.len());
        for i in start..end {
            v.swap(i, at);
            at += usize::from(ahead(v[at]));
        }
        budget.spend(end - start, site)?;
    }
    Ok(at)
}

// call-stop-latency.62 design-checked -- a polynomial's terms in a call are sorted by sort_runs and combined by Poly::from_sorted, both spent, in sort.rs, which reaches poly already, not by the sort of Poly::from_terms, unspent, nor a sort in poly.rs, which would close a cycle of modules through sig (log 0138-6)
/// The polynomial of `terms`, in any order: sorted into decreasing `order`
/// by `sort_runs`, like terms next to each other, and made by
/// `Poly::from_sorted`, spent at `site`. `terms` is left empty. A stop
/// where the budget's check stops the call.
pub(crate) fn poly<F: Field, M: Format, O: Order<M>, D: Module>(
    terms: &mut Vec<(MonoId, F::Coef)>,
    field: &F,
    table: &Table<M>,
    order: &O,
    position: D::Position,
    budget: &mut Budget,
    site: Site,
) -> Result<Poly<F, D>, Stop> {
    sort_runs(terms, |a, b| table.cmp(order, b.0, a.0), budget, site)?;
    Poly::from_sorted(terms, field, position, budget, site)
}

// matrix-stop-latency.6 design-checked -- a sort without keys sorts past RUN, 2^15, elements in runs, merged pairwise, a full check after each run and wherever the merge's buffer reaches a multiple of RUN as it compares, the rest of a run copied without one, not one sort spent after it nor keys for the block orders and lex (log 0056-1)
/// Sorts `v` by `cmp` into the order `sort_unstable_by` returns where no
/// two of its elements are equal under `cmp`, and into increasing order,
/// equal elements next to each other, where some are; spent and checked
/// at `site`. At most
/// `RUN` elements are sorted at once and spend `SORTED` units an element.
/// More are sorted in runs of `RUN` elements, a full check after each, and
/// the runs are merged pairwise into a buffer of `v`'s length, a full check
/// each time an element taken by comparison brings the buffer to a
/// multiple of `RUN`, the rest of a run copied without one. A stop where
/// the budget's check stops the call, `v` then in no order.
pub(crate) fn sort_runs<T: Copy>(
    v: &mut [T],
    mut cmp: impl FnMut(&T, &T) -> Ordering,
    budget: &mut Budget,
    site: Site,
) -> Result<(), Stop> {
    let n = v.len();
    if n <= RUN {
        v.sort_unstable_by(&mut cmp);
        return budget.spend(n * SORTED, site);
    }
    for run in v.chunks_mut(RUN) {
        run.sort_unstable_by(&mut cmp);
        budget.check(|| {}, site)?;
    }
    let mut buf: Vec<T> = Vec::with_capacity(n);
    let mut width = RUN;
    while width < n {
        buf.clear();
        let mut start = 0;
        while start < n {
            let mid = (start + width).min(n);
            let end = (start + 2 * width).min(n);
            let (mut i, mut j) = (start, mid);
            while i < mid && j < end {
                // The left run's element first on a tie, as no two compare
                // equal.
                if cmp(&v[j], &v[i]).is_lt() {
                    buf.push(v[j]);
                    j += 1;
                } else {
                    buf.push(v[i]);
                    i += 1;
                }
                if buf.len().is_multiple_of(RUN) {
                    budget.check(|| {}, site)?;
                }
            }
            buf.extend_from_slice(&v[i..mid]);
            buf.extend_from_slice(&v[j..end]);
            start = end;
        }
        v.copy_from_slice(&buf);
        width *= 2;
    }
    Ok(())
}

// matrix-sort-time.5 design-checked -- an element of a keyed sort is one u128, the key's first 96 bits above a 32-bit payload, the id or the row, not the key's first 128 bits beside the payload in 24 bytes (log 0032-1)
// sort-share-time.9 practice-checked: small-input cutoff to a simpler method -- a run of fewer than TIE_KEYED, 8, elements is sorted by the order's comparison alone, not keyed as the longer runs are: its keys cost more than its few comparisons, on the runs of two to four rows of one signature monomial most (log 0105-2, _Step 6_)
/// Elements a run of a keyed sort's ties has at least for its elements to
/// take tie keys.
const TIE_KEYED: usize = 8;

// sort-share-time.8 invented-checked -- elements whose first 96 bits agree are sorted by tie keys built per element of the run, the last word with a nonzero lane below the key's lanes by its index, its lanes, and the top of the word below, again below that word where they agree, level by level, and by the order's comparison where the keys read every word, not by the comparison alone (matrix-sort-time.6) nor by the key's next 96 bits, which lie in lanes that are zero on the sparse monomials of the long ties, so that the comparison kept its work and the key cost its build (log 0105-2, _Step 6_)
// call-stop-latency.10 design-checked -- ties spends KEY units per tie key built and SORTED per element sorted, level by level and run by run, and a unit per element its scans for runs visit, RUN at a time, at its sort's site, not the ties unspent inside the sort's one spend (log 0138-1)
/// Sorts each run of `keyed` whose elements agree in their top 96 bits,
/// their payloads, the low 32 bits, naming monomials through `monomial`:
/// a run of `TIE_KEYED` elements or more is keyed by the monomials' tie
/// keys into a scratch vector and sorted on them by `keys`, decreasing
/// where `decreasing`; a run that agrees there too is keyed again over the
/// variables `Order::tie_key_next` names, while it names some; and the
/// rest, and a run shorter than `TIE_KEYED`, by `cmp` on the payloads, by
/// `sort_runs`. The scratch vector and the worklist of runs are one per
/// sort, reused across the runs. The work is spent at `site` as it goes:
/// a unit per element a scan for runs visits and per element written back,
/// `RUN` at a time, `KEY` units per tie key built, and the sorts as `keys`
/// and `sort_runs` spend. A stop where the budget's check stops the call,
/// `keyed` then in no order.
#[allow(
    clippy::too_many_arguments,
    reason = "the table and the order the tie keys come from, the keys, their direction, the payloads' monomials and comparison, and the budget with its site, none of which belongs with another"
)]
fn ties<M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    keyed: &mut [u128],
    decreasing: bool,
    monomial: impl Fn(u32) -> MonoId,
    mut cmp: impl FnMut(u32, u32) -> Ordering,
    budget: &mut Budget,
    site: Site,
) -> Result<(), Stop> {
    let mut scratch: Vec<u128> = Vec::new();
    // The runs of `scratch` whose tie keys agree, each with the variables
    // its next keys read.
    let mut work: Vec<(usize, usize, Option<usize>)> = Vec::new();
    let tie =
        |p: u32, below: Option<usize>| table.tie_key(order, monomial(p), below) | u128::from(p);
    let mut visited = 0;
    for run in keyed.chunk_by_mut(|&a, &b| same_key(a, b)) {
        visited += run.len();
        if visited >= RUN {
            budget.spend(visited, site)?;
            visited = 0;
        }
        if run.len() < 2 {
            continue;
        }
        if run.len() < TIE_KEYED {
            sort_runs(run, |&a, &b| cmp(payload(a), payload(b)), budget, site)?;
            continue;
        }
        scratch.clear();
        scratch.extend(run.iter().map(|&k| payload(k)).map(u128::from));
        work.push((0, scratch.len(), None));
        while let Some((start, end, below)) = work.pop() {
            let part = &mut scratch[start..end];
            for chunk in part.chunks_mut(RUN) {
                for e in chunk.iter_mut() {
                    *e = tie(payload(*e), below);
                }
                budget.spend(chunk.len() * KEY, site)?;
            }
            // call-stop-latency.11 design-checked -- every other sort whose length grows with the input sorts by keys where its keys fit a u128, and by sort_runs where it compares, not sort_unstable spent after it (not measured)
            keys(part, decreasing, budget, site, no_piece)?;
            let (mut at, mut scanned) = (start, 0);
            for tied in part.chunk_by_mut(|&a, &b| same_key(a, b)) {
                scanned += tied.len();
                if scanned >= RUN {
                    budget.spend(scanned, site)?;
                    scanned = 0;
                }
                let next = order.tie_key_next(table.format(), tied[0]);
                match next {
                    Some(below) if tied.len() >= TIE_KEYED => {
                        work.push((at, at + tied.len(), Some(below)));
                    }
                    Some(_) | None => {
                        if tied.len() > 1 {
                            sort_runs(tied, |&a, &b| cmp(payload(a), payload(b)), budget, site)?;
                        }
                    }
                }
                at += tied.len();
            }
            budget.spend(scanned, site)?;
        }
        // The run's first key, above its payloads.
        let first = (run[0] >> PAYLOAD_BITS) << PAYLOAD_BITS;
        for (chunk, sorted) in run.chunks_mut(RUN).zip(scratch.chunks(RUN)) {
            for (k, &s) in chunk.iter_mut().zip(sorted) {
                *k = first | u128::from(payload(s));
            }
            budget.spend(chunk.len(), site)?;
        }
    }
    budget.spend(visited, site)
}

/// The payload of an element of a keyed sort, its low 32 bits.
pub(crate) fn payload(k: u128) -> u32 {
    (k & u128::from(u32::MAX)).narrow()
}

const _: () = assert!(PAYLOAD_BITS == u32::BITS, "a payload is a u32");

/// Whether two elements of a keyed sort agree in their keys above their
/// payloads.
#[inline]
pub(crate) fn same_key(a: u128, b: u128) -> bool {
    a >> PAYLOAD_BITS == b >> PAYLOAD_BITS
}
