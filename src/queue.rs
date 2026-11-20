//! The pending signatures of a signature run: the signatures of the S-pairs
//! it keeps until a round takes them (lairez-2022 §5.7, Algorithm 8).

use std::collections::BTreeMap;

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::id::MonoId;
use crate::monomial::Format;
use crate::narrow::Narrow;
use crate::order::Order;
use crate::sig::Sig;
use crate::sort::{RUN, by_signature, keys, no_piece, payload};
use crate::table::Table;

/// The pending signatures.
pub(crate) struct Queue {
    // call-stop-latency.22 design-checked -- the queue keeps no hash set: a signature pushed twice stands twice until a round takes its degree, sorts the popped, and drops the repeats, not the hash set of signature-pruning.26, whose growth rehashes every pending signature between two checks, nor a set of the run's own beside the table (log 0138-5)
    // call-stop-latency.57 practice-checked: bucket queue -- the pending signatures stand in a list per degree, which a round takes whole, not a list per index that each round scans whole for the least degree and again to take it out, where the repeats stand as well
    /// Per degree its pending signatures, in the order pushed, a signature
    /// pushed twice standing twice. No list is empty.
    by_degree: BTreeMap<u32, Vec<Sig>>,
}

impl Queue {
    pub(crate) fn new() -> Queue {
        Queue {
            by_degree: BTreeMap::new(),
        }
    }

    // signature-pruning.25 design-checked -- the queue keeps every S-pair signature it is given, once, and a pending signature that another divides stays, not dropped for the pending one that divides it as signature-pruning.7 did (log 0005-3)
    /// Adds `sig`, a signature of the table. A signature pending already
    /// stands twice until a round takes it, and `pop_least` drops the
    /// repeat. A pending signature that divides another stays beside it, and
    /// the run drops a multiple of a known syzygy signature when a round
    /// takes it (gfp-unsolved.14).
    pub(crate) fn push<M: Format>(&mut self, table: &Table<M>, sig: Sig) {
        self.by_degree
            .entry(sig.degree(table))
            .or_default()
            .push(sig);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_degree.is_empty()
    }

    /// Gives back the slack of the growable arrays.
    pub(crate) fn shrink_to_fit(&mut self) {
        for list in self.by_degree.values_mut() {
            list.shrink_to_fit();
        }
    }

    /// The pending signatures, each once however often it was pushed.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        let pending: std::collections::HashSet<&Sig> = self.by_degree.values().flatten().collect();
        pending.len()
    }

    // signature-pruning.3 cited-checked: eder-2013a §3 -- signature-based algorithms should always be implemented without a purely polynomial degree preselection; §4, Theorem 12 -- sig-deg(f) equals the sugar degree of poly(f)
    // call-stop-latency.21 design-checked -- the queue's selection spends the pending signatures it visits, a round's alone, not unspent after the round's check (log 0138-5)
    // call-stop-latency.58 design-checked -- the signatures a round takes are told apart by their ids, sorted as integers, before the distinct are sorted by signature, not every signature taken sorted by signature and the repeats dropped after, as a round of some calls takes many times more repeats than distinct signatures (log 0138-5)
    /// Takes out every pending signature of the least degree, in increasing
    /// order of signature, each once. Their ids, keyed a unit each, are
    /// sorted as keys (`sort::keys`) and the repeats dropped a unit a key,
    /// and the distinct are sorted by `sort::by_signature` and taken out a
    /// unit each, `RUN` at a time, each sort spending its own, all at
    /// `Site::Queue`. A stop where the budget's check stops the call, the
    /// signatures taken then dropped.
    pub(crate) fn pop_least<M: Format, O: Order<M>>(
        &mut self,
        table: &Table<M>,
        order: &O,
        budget: &mut Budget,
    ) -> Result<Vec<Sig>, Stop> {
        let Some((_, taken)) = self.by_degree.pop_first() else {
            return Ok(Vec::new());
        };
        let mut ids: Vec<u128> = Vec::with_capacity(taken.len());
        for chunk in taken.chunks(RUN) {
            ids.extend(
                chunk
                    .iter()
                    .map(|s| (u128::from(s.index) << u32::BITS) | u128::from(s.monomial.raw())),
            );
            budget.spend(chunk.len(), Site::Queue)?;
        }
        drop(taken);
        keys(&mut ids, false, budget, Site::Queue, no_piece)?;
        let mut distinct: Vec<Sig> = Vec::with_capacity(ids.len());
        let mut previous = None;
        for chunk in ids.chunks(RUN) {
            for &key in chunk {
                if previous != Some(key) {
                    distinct.push(Sig {
                        monomial: MonoId::from_raw(payload(key)),
                        index: (key >> u32::BITS).narrow(),
                    });
                }
                previous = Some(key);
            }
            budget.spend(chunk.len(), Site::Queue)?;
        }
        let rows = by_signature(table, order, &distinct, budget, Site::Queue)?;
        let mut out = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(RUN) {
            out.extend(chunk.iter().map(|r| distinct[r.index()]));
            budget.spend(chunk.len(), Site::Queue)?;
        }
        Ok(out)
    }
}
