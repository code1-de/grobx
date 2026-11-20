//! Critical pairs: the update that prunes them as each element enters the
//! basis, and the selection of a round's pairs.

#[cfg(test)]
use std::cell::Cell;
use std::cmp::Ordering;

use crate::Stop;
use crate::bitmap::Bitmap;
use crate::budget::{Budget, PROBE, Site};
use crate::field::Field;
use crate::halt::Halt;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::{ANY, Format, SUPPORT_BITS};
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::{Basis, Poly};
use crate::sort::{RUN, counting, sort_runs};
use crate::table::{Limit, Table, coprime_masked, find_passing};

/// A critical pair of basis elements i < j, with the id of the lcm of their
/// leading monomials, its degree, and the pair's sugar: in a graded order
/// the lcm's degree, or one more for a pair of a fallen element while the
/// set has not grown past its bound (`update`). While the pair lies in the set its `lcm` is
/// `MonoId::NONE`, and the set holds the lcm's words; the selection that
/// takes the pair interns them and sets the id
/// (pair-update-rest-time.1). The lcm's degree may pass the format's
/// degree bound while the pair lies in the set (gf2-monomial-time.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pair {
    pub(crate) i: ElementId,
    pub(crate) j: ElementId,
    pub(crate) lcm: MonoId,
    pub(crate) degree: u32,
    pub(crate) sugar: u32,
}

/// The complement that marks a dead place, whose pair the B criterion
/// dropped or a selection took: no mask of a leading monomial but a
/// constant's passes it (`PairSet::excluded`).
const DROPPED: u32 = u32::MAX;

#[cfg(test)]
thread_local! {
    // gf2-update-rest-time.10 design-checked -- in the test profile a thread-local fixes whether a pair set and its update keep their bitmap indexes, so that test code compares the two on one update and one call, not a parameter of the call nor the variables padded past 32 (not measured)
    /// The test profile's choice on this thread, where Some: whether a pair
    /// set that holds no pair yet keeps its indexes, in place of the
    /// variables' count (`indexed`).
    pub(crate) static INDEX_FORCED: Cell<Option<bool>> = const { Cell::new(None) };
    // gf2-chain-criteria-time.8 design-checked -- in the test profile two thread-locals, true by default, turn a divided element's one pair and the colon masks off, so that test code compares the update and a call with and without them, not a parameter of the call, which a caller has no use for (not measured)
    /// The test profile's choice on this thread: whether the update keeps a
    /// divided element's one pair (gf2-chain-criteria-time.1 to .3), or
    /// forms every pair of P1 and `Basis::push_min` scans `G_min` itself.
    pub(crate) static ONE_PAIR: Cell<bool> = const { Cell::new(true) };
    /// The test profile's choice on this thread: whether the criteria M and
    /// F keep the kept colons of one and two variables in masks
    /// (gf2-chain-criteria-time.4, .5), or scan every kept pair.
    pub(crate) static COLON_MASKS: Cell<bool> = const { Cell::new(true) };
    // gf2-pair-update-time.6 design-checked -- in the test profile a thread-local, None by default, forces the kept colons of one variable found before the new pairs are formed at every update whose keys hold colons, or at none, in place of the count the update before visited, so that test code compares the update and a call with and without them, not a parameter of the call, which a caller has no use for (not measured)
    /// The test profile's choice on this thread, where Some: whether every
    /// update whose keys hold colons finds the kept colons of one variable
    /// before it forms the new pairs (gf2-pair-update-time.1), in place of
    /// the members of P1 the update before visited (`finds_first`).
    pub(crate) static COLONS_FIRST: Cell<Option<bool>> = const { Cell::new(None) };
    // pair-index-upkeep-time.8 design-checked -- in the test profile two thread-locals, true by default, turn the pair set's holes and G_min's tombstones off, so that test code compares the update, the selection, and a call with and without them, not a parameter of the call, which a caller has no use for (not measured)
    /// The test profile's choice on this thread: whether a push fills the
    /// pair set's holes and a selection compacts the set only where its
    /// holes outnumber its live pairs (pair-index-upkeep-time.1, .3), or no
    /// push fills a hole and every selection compacts the set.
    pub(crate) static HOLES: Cell<bool> = const { Cell::new(true) };
    /// The test profile's choice on this thread: whether the members that
    /// leave `G_min` stay in the update's arrays as tombstones where the set
    /// indexes and are found by `G_min`'s index (pair-index-upkeep-time.4,
    /// .6), or the arrays are rebuilt where members leave and
    /// `Basis::push_min_known` scans `G_min` for them.
    pub(crate) static TOMBS: Cell<bool> = const { Cell::new(true) };
}

/// Whether a push fills the pair set's holes (pair-index-upkeep-time.1):
/// in the test profile `HOLES`.
fn fills() -> bool {
    #[cfg(not(test))]
    let fills = true;
    #[cfg(test)]
    let fills = HOLES.get();
    fills
}

/// Whether the members that leave `G_min` stay as tombstones where the set
/// indexes (pair-index-upkeep-time.4): in the test profile `TOMBS`.
fn entombs() -> bool {
    #[cfg(not(test))]
    let entombs = true;
    #[cfg(test)]
    let entombs = TOMBS.get();
    entombs
}

// gf2-update-rest-time.3 tuned-checked -- the indexes run where the format's variables pass the divisor mask's 32 bits, not at every count, where to 32 variables the masks are exact and the index adds its upkeep alone: over ℚ and GF(65521) the index at every count took the calls of 32 variables or fewer longer and those of more variables less long (log 0137-1, 0137-2)
/// Whether the plain run's update and selection keep their bitmap indexes
/// on a format of n variables (gf2-update-rest-time.1, .7): past the
/// divisor mask's 32 bits, where a mask bit stands for several variables
/// (many-variables-time.2). In the test profile `INDEX_FORCED` decides
/// where it holds a choice.
pub(crate) fn indexed(n: usize) -> bool {
    let past = n > u32::BITS as usize;
    #[cfg(test)]
    let past = INDEX_FORCED.get().unwrap_or(past);
    past
}

/// Whether a pair set keeps its bitmap index, which its first update
/// decides (`indexed`).
enum Indexing {
    /// No update has run.
    Undecided,
    /// The B criterion scans the masks.
    Off,
    /// The index over the places, by the lcms' supports, and per element,
    /// by its id, its leading monomial's support, from which a rebuild
    /// takes a pair's (gf2-update-rest-time.4).
    On { index: Bitmap, supports: Vec<u128> },
}

// pair-update-rest-time.1 practice-checked: lazy evaluation -- a kept pair's lcm is interned by the selection that takes it, the pair set holding the lcm's words beside the pair until then, not interned at the update, where the B criterion drops most kept pairs before a selection on the calls whose update weighs most; nor interned at the update and the words kept too; nor the pair without its words and the lcm formed again from the leads at the selection, which the B criterion reads at every passing pair
/// The plain run's pair set: the pairs at their places, and beside them
/// their lcms' words and their lcm masks' complements; the dead places,
/// which pushes fill (`holes`); the pairs the last selection left of its
/// sugar (`select`); and past 32 variables a bitmap index over the places
/// by the lcms' supports (`indexed`). The set's order is its pairs' keys
/// (`key`), whatever their places. `W` is the format's word.
pub(crate) struct PairSet<W> {
    /// The pairs at their places, the dead ones among them: those the B
    /// criterion dropped and those a selection took, until a push fills
    /// their place or the set compacts.
    pairs: Vec<Pair>,
    /// Per place, its lcm's words, `w` each.
    words: Vec<W>,
    /// The format's word count.
    w: usize,
    /// Per place, the complement of its lcm's divisor mask; `DROPPED` at a
    /// dead place. A pair's lcm is no constant, so its mask is not 0
    /// (`Format::mask`), and no live pair's complement is `DROPPED`.
    excluded: Vec<u32>,
    // pair-index-upkeep-time.1 practice-checked: free list -- the set's dead places, those the B criterion dropped and those a selection took, go onto a free list that pushes fill before the arrays grow, a dead place's index bits cleared and a filled place's set, and a selection compacts the set no more, not the set compacted and its index rebuilt at every selection (gf2-update-rest-time.4), nor the dead places left until a compaction at a share of them with the index's bits as they stand, whose dead places lengthen the B criterion's intersections and which each selection still passes over (log 0171-1, 0171-2)
    /// The dead places, which pushes fill from the last
    /// (pair-index-upkeep-time.9): every place but the live pairs'.
    holes: Vec<usize>,
    /// The live pairs.
    live: usize,
    /// The live pairs of the last selection's sugar that it left in the
    /// set: 0 where it took them all.
    deferred: usize,
    /// The sugar of the last split; None before one, after a selection
    /// whose least sugar lies above it, and after the set was emptied
    /// (split-piece-time.2).
    split: Option<u32>,
    /// The note of the last piece of the split `split` names, where the
    /// plain run noted it (`note_piece`); None from each split on until its
    /// first note (gf2-wide-round-time.1).
    piece: Option<Piece>,
    /// Whether a fallen element's pair counts one degree past its lcm's in a
    /// graded order: set at the set's creation, cleared for good by the first
    /// update that begins with more than `DEFER_LIVE` live pairs
    /// (fallen-pair-losses.1).
    defers: bool,
    /// Whether the set keeps its bitmap index, and the index.
    indexing: Indexing,
}

/// What the plain run noted of a piece of a split (`PairSet::note_piece`).
#[derive(Clone, Copy, Debug)]
struct Piece {
    /// The pairs the piece took.
    taken: usize,
    /// The elements its round made.
    made: usize,
    /// Those of them that fell (`fall`).
    fallen: usize,
}

impl<W: Copy> PairSet<W> {
    /// An empty set for a format of `w` words.
    pub(crate) fn new(w: usize) -> PairSet<W> {
        PairSet {
            pairs: Vec::new(),
            words: Vec::new(),
            w,
            excluded: Vec::new(),
            holes: Vec::new(),
            live: 0,
            deferred: 0,
            split: None,
            piece: None,
            defers: true,
            indexing: Indexing::Undecided,
        }
    }

    /// The live pairs of the last selection's sugar that it left in the
    /// set, 0 where it took them all (redundant-round-pairs.1).
    pub(crate) fn deferred(&self) -> usize {
        self.deferred
    }

    /// Notes that the round of the last selection, a piece of a split, took
    /// `taken` pairs and made `made` elements, `fallen` of which fell, which
    /// the next selection of the split's sugar reads (gf2-wide-round-time.1,
    /// gf2-build-time.1).
    pub(crate) fn note_piece(&mut self, taken: usize, made: usize, fallen: usize) {
        debug_assert!(
            self.deferred > 0 && self.split.is_some(),
            "a note of a piece where the last selection split no sugar"
        );
        debug_assert!(made <= taken, "{made} elements of {taken} pairs");
        debug_assert!(fallen <= made, "{fallen} fallen of {made} elements");
        self.piece = Some(Piece {
            taken,
            made,
            fallen,
        });
    }

    /// The live pairs.
    pub(crate) fn len(&self) -> usize {
        self.live
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The live pairs' places, in the set's order (`key`).
    #[cfg(test)]
    pub(crate) fn places(&self) -> Vec<usize> {
        let mut places: Vec<usize> = (0..self.pairs.len())
            .filter(|&k| self.excluded[k] != DROPPED)
            .collect();
        places.sort_unstable_by_key(|&k| key(&self.pairs[k]));
        places
    }

    /// The live pairs, in the set's order.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Pair> {
        self.places().into_iter().map(|k| &self.pairs[k])
    }

    /// The live pairs with their lcms' words, in the set's order.
    #[cfg(test)]
    pub(crate) fn with_words(&self) -> impl Iterator<Item = (&Pair, &[W])> {
        self.places()
            .into_iter()
            .map(|k| (&self.pairs[k], self.lcm_words(k)))
    }

    /// The lcm's words of the pair at place `k`, among the pairs the
    /// arrays hold.
    fn lcm_words(&self, k: usize) -> &[W] {
        debug_assert!(
            k < self.pairs.len(),
            "place {k} of {n}",
            n = self.pairs.len()
        );
        &self.words[k * self.w..(k + 1) * self.w]
    }

    /// The places the arrays hold, the holes among them.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.pairs.len()
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.pairs.shrink_to_fit();
        self.words.shrink_to_fit();
        self.excluded.shrink_to_fit();
        self.holes.shrink_to_fit();
        match &mut self.indexing {
            Indexing::On { index, supports } => {
                index.shrink_to_fit();
                supports.shrink_to_fit();
            }
            Indexing::Undecided | Indexing::Off => {}
        }
    }

    /// Drops every pair: the run needs none of them reduced. Whether the
    /// set defers stays as it is, the run's (fallen-pair-losses.1).
    pub(crate) fn clear(&mut self) {
        self.pairs.clear();
        self.words.clear();
        self.excluded.clear();
        self.holes.clear();
        self.live = 0;
        self.deferred = 0;
        self.split = None;
        match &mut self.indexing {
            Indexing::On { index, .. } => index.clear(),
            Indexing::Undecided | Indexing::Off => {}
        }
    }

    /// Places `pair`, whose lcm has the words `lcm` and the divisor mask
    /// `mask`, at the last hole, or past the last place where the set holds
    /// none or the test profile's `HOLES` is false; its `lcm` id is
    /// `MonoId::NONE`. Where the set indexes, `support` is the lcm's
    /// support, which joins the index at the pair's place; None else.
    pub(crate) fn push(&mut self, pair: Pair, mask: u32, lcm: &[W], support: Option<u128>) {
        debug_assert_ne!(mask, 0, "a pair's lcm is no constant");
        debug_assert_eq!(pair.lcm, MonoId::NONE, "a live pair's lcm is not interned");
        debug_assert_eq!(lcm.len(), self.w, "the format's words");
        debug_assert_eq!(
            support.is_some(),
            matches!(self.indexing, Indexing::On { .. }),
            "a support where the set indexes"
        );
        // pair-index-upkeep-time.9 design-checked -- the holes are a stack, a push filling the place vacated last, not the lowest place first by a heap nor the place vacated first: the places vacated last are those the update's B criterion dropped and read last (log 0171-2)
        let hole = if fills() { self.holes.pop() } else { None };
        let k = if let Some(k) = hole {
            self.pairs[k] = pair;
            self.words[k * self.w..(k + 1) * self.w].copy_from_slice(lcm);
            self.excluded[k] = !mask;
            k
        } else {
            self.pairs.push(pair);
            self.words.extend_from_slice(lcm);
            self.excluded.push(!mask);
            self.pairs.len() - 1
        };
        self.live += 1;
        match &mut self.indexing {
            Indexing::On { index, .. } => {
                if let Some(support) = support {
                    index.set(k, support);
                }
            }
            Indexing::Undecided | Indexing::Off => {}
        }
    }

    // pair-index-upkeep-time.1 practice-checked: free list -- the set's dead places, those the B criterion dropped and those a selection took, go onto a free list that pushes fill before the arrays grow, a dead place's index bits cleared and a filled place's set, and a selection compacts the set no more, not the set compacted and its index rebuilt at every selection (gf2-update-rest-time.4), nor the dead places left until a compaction at a share of them with the index's bits as they stand, whose dead places lengthen the B criterion's intersections and which each selection still passes over (log 0171-1, 0171-2)
    /// The live pair at place `k` dies: its complement `DROPPED`, its place
    /// a hole, and where the set indexes its place out of the index, by its
    /// lcm's support, the union of its elements' supports
    /// (pair-index-upkeep-time.1).
    fn vacate(&mut self, k: usize) {
        debug_assert!(
            k < self.pairs.len() && self.excluded[k] != DROPPED,
            "place {k} of {n} is live",
            n = self.pairs.len()
        );
        self.excluded[k] = DROPPED;
        self.live -= 1;
        self.holes.push(k);
        match &mut self.indexing {
            Indexing::On { index, supports } => {
                let p = self.pairs[k];
                index.unset(k, supports[p.i.index()] | supports[p.j.index()]);
            }
            Indexing::Undecided | Indexing::Off => {}
        }
    }

    // pair-index-upkeep-time.7 design-checked -- a vacated place spends a unit, and where the set indexes 2 · PROBE units and a unit per set of the index besides, its elements' supports read and its bits cleared, as a place of the index's rebuild spends, RUN places at a time at the update's and the selection's site; the search for the members that leave spends as Bitmap::holding spends, 1 + w units a place it visits, and a member entombed a unit per set of the index; not the vacated places unspent, which a selection of a large piece or an update whose B criterion drops many pairs vacates by the thousand between two checks (log 0171-2)
    /// Vacates the live pairs at `places` (`vacate`), spent at `site` as it
    /// goes, `RUN` places at a time: a unit a place, and where the set
    /// indexes `2 · PROBE` units and a unit per set of the index besides. A
    /// stop where the budget's check stops the call, the set then part
    /// updated, which the stopped run drops.
    fn vacate_all(
        &mut self,
        places: &[usize],
        budget: &mut Budget,
        site: Site,
    ) -> Result<(), Stop> {
        let per = match &self.indexing {
            Indexing::On { index, .. } => 1 + 2 * PROBE + index.bits(),
            Indexing::Undecided | Indexing::Off => 1,
        };
        for chunk in places.chunks(RUN) {
            for &k in chunk {
                self.vacate(k);
            }
            budget.spend(chunk.len() * per, site)?;
        }
        Ok(())
    }

    // gf2-update-rest-time.4 design-checked -- the index is rebuilt where the set compacts, each pair's lcm's support the union of its elements' leading monomials' supports, recorded per element, not a support recorded per place, which took sat-r75-320-s1's peak higher (log 0137-1, 0137-2); amended in session 0171-2 by pair-index-upkeep-time.1 and .3: the set compacts where its holes outnumber its live pairs, not at every selection
    /// Moves the live pairs to the front in their places' order, their
    /// words and complements with them, and empties `holes`; where the set
    /// indexes, its index is rebuilt over them. Spent at `Site::Select` as
    /// it goes, `RUN` places at a time: a unit a place and its lcm's words a
    /// pair moved, and the index's rebuild a unit per place and support bit
    /// and `2 · PROBE` units a place, its elements' supports read
    /// (call-stop-latency.12). A stop where the budget's check stops the
    /// call, the set then part compacted, which the stopped run drops.
    fn compact(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let w = self.w;
        let mut free = 0;
        for start in (0..self.pairs.len()).step_by(RUN) {
            let end = (start + RUN).min(self.pairs.len());
            let before = free;
            for k in start..end {
                if self.excluded[k] == DROPPED {
                    continue;
                }
                self.pairs[free] = self.pairs[k];
                self.excluded[free] = self.excluded[k];
                self.words.copy_within(k * w..(k + 1) * w, free * w);
                free += 1;
            }
            budget.spend(end - start + (free - before) * w, Site::Select)?;
        }
        debug_assert_eq!(free, self.live, "the live pairs moved");
        self.pairs.truncate(free);
        self.words.truncate(free * w);
        self.excluded.truncate(free);
        self.holes.clear();
        match &mut self.indexing {
            Indexing::On { index, supports } => {
                let set = &self.pairs;
                index.rebuild(
                    set.len(),
                    |k| supports[set[k].i.index()] | supports[set[k].j.index()],
                    2 * PROBE,
                    budget,
                    Site::Select,
                )?;
            }
            Indexing::Undecided | Indexing::Off => {}
        }
        Ok(())
    }

    /// Decides at the set's first update whether it keeps its index
    /// (`indexed`), on a format of `n` variables, and where it does
    /// records the support `lead` of element t, entered now: the support
    /// of each element before it first, from the table, where the set has
    /// not recorded it, as for the basis a carried run starts from,
    /// `PROBE` units and its words an element spent at `Site::Update`,
    /// `RUN` elements at a time. The lead's support at t's place in
    /// `element_supports`. A stop where the budget's check stops the call.
    fn index_element<F: Field, M: Format<Word = W>, D: Module>(
        &mut self,
        table: &Table<M>,
        basis: &Basis<F, D>,
        t: ElementId,
        lead: &[W],
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        let format = table.format();
        match self.indexing {
            Indexing::Undecided => {
                self.indexing = if indexed(format.variables()) {
                    Indexing::On {
                        index: Bitmap::new(format.variables().min(SUPPORT_BITS)),
                        supports: Vec::new(),
                    }
                } else {
                    Indexing::Off
                };
            }
            Indexing::Off | Indexing::On { .. } => {}
        }
        match &mut self.indexing {
            Indexing::On { supports, .. } => {
                let t = t.index();
                // gf2-update-rest-time.11 design-checked -- element t's support is recorded at place t, a support recorded at t or past it replaced, not a debug_assert! that none stands there, so that a stale support never leaves a member that shares a variable with the lead out of the new pairs (log 0137-2)
                for start in (supports.len()..t).step_by(RUN) {
                    let end = (start + RUN).min(t);
                    for e in start..end {
                        let lead = table.monomial(basis.lead(ElementId::at(e)));
                        supports.push(format.support(lead));
                    }
                    budget.spend((end - start) * (PROBE + format.words()), Site::Update)?;
                }
                supports.truncate(t);
                supports.push(format.support(lead));
            }
            Indexing::Undecided | Indexing::Off => {}
        }
        Ok(())
    }

    /// Where the set indexes, the supports of the elements' leading
    /// monomials by id, to the last element entered; None else.
    fn element_supports(&self) -> Option<&[u128]> {
        match &self.indexing {
            Indexing::On { supports, .. } => Some(supports),
            Indexing::Undecided | Indexing::Off => None,
        }
    }

    /// Vacates the live pair at place `k`, as the B criterion vacates a
    /// pair it drops.
    #[cfg(test)]
    pub(crate) fn drop_at(&mut self, k: usize) {
        assert_ne!(self.excluded[k], DROPPED, "the pair at {k} is live");
        self.vacate(k);
    }
}

// divisibility-time.4 design-checked -- the update's buffers for its new pairs stay with the run and are cleared at each element entered, not allocated per element entered (log 0043-1)
/// The update's buffers, of its new pairs and of the pairs the B criterion
/// drops, kept by the run from one element entered to the next.
pub(crate) struct NewPairs<W> {
    /// The lcm of each new pair of P1, by its place among them in `G_min`'s
    /// order, the format's words each.
    words: Vec<W>,
    /// The element of each new pair of P1, by its place.
    elements: Vec<ElementId>,
    /// Per new pair of P1: its lcm's degree, its place, and its lcm's
    /// mask.
    order: Vec<(u32, u32, u32)>,
    /// `order` by increasing degree, ties by place.
    sorted: Vec<(u32, u32, u32)>,
    /// The counting sort's counts, then the next place of each degree
    /// (`by_degree`).
    counts: Vec<u32>,
    /// The pairs kept so far: place, lcm's mask, and lcm's degree.
    kept: Vec<(u32, u32, u32)>,
    /// The places in the pair set of the pairs the B criterion drops at
    /// this update.
    dropped: Vec<usize>,
    /// `G_min`'s members, in its order: `Basis::min` as it stands, with the
    /// tombstones of `min_dead` among them, or empty after `release`, and
    /// the next update rebuilds the arrays (pair-update-rest-time.4).
    min_ids: Vec<ElementId>,
    /// The members' leading monomials' masks, by place in `min_ids`.
    min_masks: Vec<u32>,
    /// The members' leading monomials' words, the format's words each, by
    /// place in `min_ids`.
    min_words: Vec<W>,
    /// Where the pair set indexes: per support bit a bit set of the places
    /// in `min_ids` whose member's leading monomial's support holds it
    /// (gf2-update-rest-time.7); None else.
    min_index: Option<Bitmap>,
    /// Where the pair set indexes: the members' leading monomials'
    /// supports, by place in `min_ids`.
    min_supports: Vec<u128>,
    // pair-index-upkeep-time.4 practice-checked: tombstones -- where the pair set indexes, a member that leaves G_min stays at its place in the arrays as a tombstone, its index bits cleared, and the arrays are rebuilt where the tombstones reach the live members, not the arrays and index rebuilt wherever members leave (pair-update-rest-time.4): the index names the live members alone to the new pairs and to the divisor's search (log 0171-1, 0171-2)
    /// Per place in `min_ids`, whether its member left `G_min` and stands
    /// as a tombstone, which `G_min`'s index names no more
    /// (pair-index-upkeep-time.4): none where the pair set does not index.
    min_dead: Vec<bool>,
    /// The tombstones of `min_dead`.
    tombs: usize,
    /// The places in `min_ids` of the members the element entering makes
    /// redundant, in `G_min`'s order (`leaving`).
    leaving: Vec<usize>,
    /// Their ids, by place in `leaving`.
    left: Vec<ElementId>,
    /// Where the pair set indexes: the lcm's support of each new pair of
    /// P1, by its place.
    supports: Vec<u128>,
    /// The kept pairs' keys (`Keys`), in `kept`'s order while the criteria
    /// M and F run, before they sort `kept` by place
    /// (gf2-update-rest-time.9, gf2-chain-criteria-time.5).
    kept_keys: Vec<u128>,
    /// The bits of the kept colons of one variable outside the lead's key,
    /// one bit each, while the criteria M and F run
    /// (gf2-chain-criteria-time.4).
    ones: u128,
    /// Per key bit of `ones`, the index in `kept` of the pair whose colon
    /// it is; `SUPPORT_BITS` entries once the criteria ran, or none.
    one_at: Vec<u32>,
    /// Per key bit, the other bits of the kept colons of two bits outside
    /// the lead's key that hold it; `SUPPORT_BITS` entries once the
    /// criteria ran, or none.
    twos: Vec<u128>,
    /// The bits whose entry in `twos` is not 0, which the next criteria
    /// clear.
    twos_held: u128,
    /// The kept colons of `twos`: their lower bit, their higher bit, and
    /// the index in `kept` of the pair whose colon it is.
    two_at: Vec<(u32, u32, u32)>,
    /// The indexes in `kept` of the other kept pairs, which each pair of P1
    /// the masks pass scans.
    listed: Vec<u32>,
    /// The chunks of the B criterion's intersections (`Bitmap::holding`) and
    /// of the visits of `G_min`'s index that form the new pairs
    /// (`Bitmap::meeting_less`).
    buffer: Vec<u64>,
    /// Where the pair set does not index: the places in `G_min`'s arrays of
    /// the members of P1, in its order, gathered before their pairs are
    /// formed (gf2-pair-update-time.4).
    members: Vec<u32>,
    /// Where the pair set indexes and the kept colons of one variable are
    /// found first: the places of the members whose colon is one bit
    /// outside the lead's key, in `G_min`'s order, whose pairs are formed
    /// whatever the kept colons (gf2-pair-update-time.2, .5).
    singles: Vec<usize>,
    /// The members of P1 the last update visited (gf2-pair-update-time.3).
    visited: usize,
}

impl<W: Copy + Eq> NewPairs<W> {
    pub(crate) fn new() -> NewPairs<W> {
        NewPairs {
            words: Vec::new(),
            elements: Vec::new(),
            order: Vec::new(),
            sorted: Vec::new(),
            counts: Vec::new(),
            kept: Vec::new(),
            dropped: Vec::new(),
            min_ids: Vec::new(),
            min_masks: Vec::new(),
            min_words: Vec::new(),
            min_index: None,
            min_supports: Vec::new(),
            min_dead: Vec::new(),
            tombs: 0,
            leaving: Vec::new(),
            left: Vec::new(),
            supports: Vec::new(),
            kept_keys: Vec::new(),
            ones: 0,
            one_at: Vec::new(),
            twos: Vec::new(),
            twos_held: 0,
            two_at: Vec::new(),
            listed: Vec::new(),
            buffer: Vec::new(),
            members: Vec::new(),
            singles: Vec::new(),
            visited: 0,
        }
    }

    /// Gives back the buffers' memory. The next update rebuilds `G_min`'s
    /// arrays, and finds the kept colons first as `visited` rules.
    pub(crate) fn release(&mut self) {
        self.words = Vec::new();
        self.elements = Vec::new();
        self.order = Vec::new();
        self.sorted = Vec::new();
        self.counts = Vec::new();
        self.kept = Vec::new();
        self.dropped = Vec::new();
        self.min_ids = Vec::new();
        self.min_masks = Vec::new();
        self.min_words = Vec::new();
        self.min_index = None;
        self.min_supports = Vec::new();
        self.min_dead = Vec::new();
        self.tombs = 0;
        self.leaving = Vec::new();
        self.left = Vec::new();
        self.supports = Vec::new();
        self.kept_keys = Vec::new();
        self.ones = 0;
        self.one_at = Vec::new();
        self.twos = Vec::new();
        self.twos_held = 0;
        self.two_at = Vec::new();
        self.listed = Vec::new();
        self.buffer = Vec::new();
        self.members = Vec::new();
        self.singles = Vec::new();
    }

    /// Fills `G_min`'s arrays from the basis, and where the pair set
    /// indexes, `supports` the elements' supports by id, the members'
    /// supports and their index. Spent at `Site::Update` as it goes:
    /// `PROBE` units and its words a member, `RUN` members at a time, and
    /// the index's rebuild a unit per member and support bit. A stop where
    /// the budget's check stops the call, the arrays then part filled and
    /// the index set aside, which the next update rebuilds.
    fn rebuild_min<F: Field, M: Format<Word = W>, D: Module>(
        &mut self,
        table: &Table<M>,
        basis: &Basis<F, D>,
        supports: Option<&[u128]>,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        // The index stands aside while the members are appended, and is
        // rebuilt over them at once.
        let index = self.min_index.take();
        self.min_ids.clear();
        self.min_masks.clear();
        self.min_words.clear();
        self.min_supports.clear();
        self.min_dead.clear();
        self.tombs = 0;
        let w = table.format().words();
        for chunk in basis.min().chunks(RUN) {
            for &i in chunk {
                let support = supports.map(|s| s[i.index()]);
                self.append_min(
                    i,
                    table.mask(basis.lead(i)),
                    table.monomial(basis.lead(i)),
                    support,
                );
            }
            budget.spend(chunk.len() * (PROBE + w), Site::Update)?;
        }
        if supports.is_some() {
            let bits = table.format().variables().min(SUPPORT_BITS);
            let mut index = index.unwrap_or_else(|| Bitmap::new(bits));
            let members = &self.min_supports;
            index.rebuild(members.len(), |k| members[k], 1, budget, Site::Update)?;
            self.min_index = Some(index);
        }
        Ok(())
    }

    /// Appends the member `id` of `G_min`, whose leading monomial has the
    /// mask `mask`, the words `words`, and where the pair set indexes the
    /// support `support`, which joins the members' index.
    fn append_min(&mut self, id: ElementId, mask: u32, words: &[W], support: Option<u128>) {
        let k = self.min_ids.len();
        self.min_dead.push(false);
        self.min_ids.push(id);
        self.min_masks.push(mask);
        self.min_words.extend_from_slice(words);
        if let Some(support) = support {
            self.min_supports.push(support);
            if let Some(index) = self.min_index.as_mut() {
                index.set(k, support);
            }
        }
    }

    // pair-index-upkeep-time.7 design-checked -- a vacated place spends a unit, and where the set indexes 2 · PROBE units and a unit per set of the index besides, its elements' supports read and its bits cleared, as a place of the index's rebuild spends, RUN places at a time at the update's and the selection's site; the search for the members that leave spends as Bitmap::holding spends, 1 + w units a place it visits, and a member entombed a unit per set of the index; not the vacated places unspent, which a selection of a large piece or an update whose B criterion drops many pairs vacates by the thousand between two checks (log 0171-2)
    /// Marks the members of `leaving` dead, tombstones whose places leave
    /// `G_min`'s index by their supports (pair-index-upkeep-time.4). Spent
    /// at `Site::Update` as it goes, `RUN` members at a time, a unit per set
    /// of the index a member (pair-index-upkeep-time.7); a stop where the
    /// budget's check stops the call.
    fn entomb(&mut self, budget: &mut Budget) -> Result<(), Stop> {
        let NewPairs {
            min_dead,
            tombs,
            min_index,
            min_supports,
            leaving,
            ..
        } = self;
        let per = min_index.as_ref().map_or(1, Bitmap::bits);
        for chunk in leaving.chunks(RUN) {
            for &k in chunk {
                debug_assert!(!min_dead[k], "member {k} is live");
                min_dead[k] = true;
                *tombs += 1;
                if let Some(index) = min_index.as_mut() {
                    index.unset(k, min_supports[k]);
                }
            }
            budget.spend(chunk.len() * per, Site::Update)?;
        }
        Ok(())
    }

    /// The places in `min_ids` of the live members, in `G_min`'s order.
    fn live_min(&self) -> impl Iterator<Item = usize> {
        (0..self.min_ids.len()).filter(|&k| !self.min_dead[k])
    }

    /// Whether the arrays hold `G_min`'s members in its order beside their
    /// tombstones, with their masks and words, and where the pair set
    /// indexes, `supports` the elements' supports by id, with their supports
    /// beside an index; and no tombstone where it does not: the test
    /// profile's check.
    fn min_in_step<F: Field, M: Format<Word = W>, D: Module>(
        &self,
        table: &Table<M>,
        basis: &Basis<F, D>,
        supports: Option<&[u128]>,
    ) -> bool {
        let w = table.format().words();
        let indexed = supports.is_none_or(|s| {
            self.min_index.is_some()
                && self.min_supports.len() == self.min_ids.len()
                && self.live_min().all(|k| {
                    let (i, m) = (self.min_ids[k], self.min_supports[k]);
                    s[i.index()] == m && m == table.format().support(table.monomial(basis.lead(i)))
                })
        });
        self.live_min()
            .map(|k| self.min_ids[k])
            .eq(basis.min().iter().copied())
            && self.min_dead.len() == self.min_ids.len()
            && self.min_dead.iter().filter(|&&dead| dead).count() == self.tombs
            && (supports.is_some() || self.tombs == 0)
            && self.min_masks.len() == self.min_ids.len()
            && self.min_words.len() == w * self.min_ids.len()
            && self.live_min().all(|k| {
                let lead = basis.lead(self.min_ids[k]);
                self.min_masks[k] == table.mask(lead)
                    && self.min_words[k * w..(k + 1) * w] == *table.monomial(lead)
            })
            && indexed
    }

    /// `G_min`'s live members' ids and their leading monomials' masks, as
    /// the arrays hold them.
    #[cfg(test)]
    pub(crate) fn min_arrays(&self) -> (Vec<ElementId>, Vec<u32>) {
        (
            self.live_min().map(|k| self.min_ids[k]).collect(),
            self.live_min().map(|k| self.min_masks[k]).collect(),
        )
    }
}

// pair-update-rest-time.3 cited-checked:demin-2024 §4.1 -- the specialized functions that implement monomial arithmetic are compiled and inlined, which relocates the work from running time to compile time; bachmann-1998 §3 -- loop unrolling and pointer arithmetic
/// `$op::<N, …>(args)` for a format of `$words` words, N from one to
/// `UNROLLED`, and `$op::<ANY, …>(args)` for every other count, where `…`
/// are the type parameters after N, each `_`: the update's loops, and the
/// substitution's (`substitute::substitute`), compiled per word count,
/// through the word operations they call (`Format::divides_n` and the
/// others). The dispatch stands at each loop, once per update, and at no
/// call of a word operation: a dispatch in each operation put the
/// operations out of line at every caller, and inlined it put the update's
/// closures out of line (log 0107-2, _Step 6_).
macro_rules! per_words {
    ($words:expr, $op:ident $(::<$($g:ty),*>)? ($($arg:expr),* $(,)?)) => {
        match $words {
            1 => $op::<1 $($(, $g)*)?>($($arg),*),
            2 => $op::<2 $($(, $g)*)?>($($arg),*),
            3 => $op::<3 $($(, $g)*)?>($($arg),*),
            4 => $op::<4 $($(, $g)*)?>($($arg),*),
            5 => $op::<5 $($(, $g)*)?>($($arg),*),
            6 => $op::<6 $($(, $g)*)?>($($arg),*),
            7 => $op::<7 $($(, $g)*)?>($($arg),*),
            8 => $op::<8 $($(, $g)*)?>($($arg),*),
            _ => {
                debug_assert!(
                    $words == 0 || $words > $crate::monomial::UNROLLED,
                    "a word count past the copies"
                );
                $op::<{ $crate::monomial::ANY } $($(, $g)*)?>($($arg),*)
            }
        }
    };
}
pub(crate) use per_words;

// cooper-rest-gap.2 tuned-checked -- a fallen element's pair counts one degree past its lcm's only while the set holds at most 128 live pairs when the update begins, not at every size of the set, where the deferred pairs gather into one large round, nor while it holds at most 64, 256, 512, or 1,024 (log 0110-1, _Step 0_)
// fallen-pair-losses.1 design-checked -- the step holds until the first update of the run that begins with more than 128 live pairs and never again in that run, not switched by the set's size as each update begins, where a set that hovers near the bound puts pairs of one lcm degree at two degrees and the descent's rounds double, nor with every pending pair sent back to its lcm's degree at each update past the bound (log 0111-1, _Step 0_)
// fallen-pair-losses.3 tuned-checked -- the latch closes at the step's own bound of 128, not at 256, where the deferred pairs gather into larger rounds (log 0111-1, _Step 0_)
/// The most live pairs the set may have held, when the update of a new
/// element begins and at every update of the run before it, for a pair of
/// a fallen element to count one degree past its lcm's in a graded order
/// (cooper-rest-gap.2, fallen-pair-losses.1, .3).
pub(crate) const DEFER_LIVE: usize = 128;

/// The fall of an element of sugar `sugar` whose leading monomial is
/// `lead`: its sugar less its leading monomial's degree. The element fell
/// where its fall is positive.
pub(crate) fn fall<M: Format>(table: &Table<M>, sugar: u32, lead: MonoId) -> u32 {
    let degree = table.degree(lead);
    debug_assert!(
        sugar >= degree,
        "a sugar of {sugar} below its leading monomial's degree {degree}"
    );
    sugar - degree
}

// gf2-update-time.2 design-checked -- the entering order is the field's, increasing over GF(2) and decreasing over every other field, not increasing over every field, which over ℚ and GF(65521) parts the instances and gains nothing in the mean (log 0130-1, 0130-2)
/// The order in which the plain run enters an element of leading monomial
/// `a` and one of leading monomial `b`, `Less` where the first enters
/// first: increasing where `F::ENTERS_ASCENDING` holds (gf2-update-time.1),
/// else decreasing (f4-gfp.2). `Plain::enter` sorts by it, and so does the
/// learn's record of the inputs and of each step (`trace::inputs`,
/// `trace::step`), which the replays enter in.
pub(crate) fn entering<F: Field, M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    a: MonoId,
    b: MonoId,
) -> Ordering {
    if F::ENTERS_ASCENDING {
        table.cmp(order, a, b)
    } else {
        table.cmp(order, b, a)
    }
}

// f4-gfp.1 cited-checked: gebauer-1988 §1, §4.4, §4.6 -- the three criteria are independent of the pairs considered before, which speeds up their tests, and redundant elements are cancelled as new ones enter
/// Enters the monic nonzero element `h` into the basis and updates the pair
/// set, as gebauer-1988 §4.4 `updatePairs` and §4.6 in the form of joux-2011
/// §2.2 Algorithm 2. Beyond them, the B criterion drops the pairs that a
/// pair of coprime leading monomials settles (`drops`,
/// plain-zero-pairs-time.1). Where a member of `G_min` divides lm(h), the
/// new pairs are the one pair M and F keep, which `last_divisor` finds
/// before any other is formed (`one_pair`, gf2-chain-criteria-time.1, .2),
/// and `G_min` takes the finding (`Basis::push_min_known`). Elsewhere M
/// and F read the colons of one and two variables off masks where a key's
/// bit stands for one variable (`criteria`, `Keys`). `sugar` holds the
/// sugar of every element, `h`'s last.
/// An element fell where its sugar lies above its leading monomial's
/// degree. In a graded order a new pair counts at its lcm's degree, or one
/// more where one of its elements fell and the set held at most
/// `DEFER_LIVE` live pairs when this update and every update of the run
/// before it began (cooper-rest-gap.1, .2, fallen-pair-losses.1); a pair
/// keeps its sugar (fallen-pair-losses.2). In another order a new pair
/// counts at its sugar (order-family.7). A pair the criteria keep whose
/// sugar in giovini-1991's sense, its fall plus its lcm's degree, is at
/// most `covered` is left out: its S-polynomial has a standard
/// representation (signature-run-losses.3). `covered` 0 leaves none out,
/// as an lcm's degree is at least 1. `new` holds the
/// update's buffers, of its new pairs and of the places of the pairs the
/// B criterion drops. The update interns nothing: a kept pair's
/// lcm goes into the set as words (pair-update-rest-time.1), whatever its
/// degree, which the selection that takes the pair checks
/// (gf2-monomial-time.3). Overflow where a kept pair's sugar passes
/// `u32::MAX`, before that pair is pushed; Size where the basis is full.
/// The update spends its work on `budget` at `Site::Update` as it goes, its
/// scans `RUN` places or members at a time, and `h`'s entry into the basis
/// at `Site::Enter` (`Basis::push_min_known`); a stop where the budget's check
/// stops the call, the set and the buffers then part updated, which the
/// stopped run drops. Where the set indexes
/// (`indexed`), the B criterion takes its candidates from the set's index
/// and the new pairs their members from `G_min`'s, and every pair pushed
/// joins the set's index (gf2-update-rest-time.1, .7). The pairs the B
/// criterion drops leave their places as holes, which the pairs pushed fill
/// (pair-index-upkeep-time.1), and where the set indexes the members h
/// makes redundant stay in `G_min`'s arrays as tombstones (`enter_min`).
#[allow(
    clippy::too_many_arguments,
    reason = "the table, order, basis, pair set, sugars, and element of the update, its buffers, the sugar its pairs must pass, and the call's budget, none of which belongs with another"
)]
pub(crate) fn update<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &Table<M>,
    _: &O,
    basis: &mut Basis<F, D>,
    pairs: &mut PairSet<M::Word>,
    sugar: &[u32],
    h: Poly<F, D>,
    new: &mut NewPairs<M::Word>,
    covered: u32,
    budget: &mut Budget,
) -> Result<(), Halt> {
    debug_assert!(!h.is_zero(), "h is nonzero");
    debug_assert!(
        sugar.len() == basis.len() + 1,
        "a sugar per element, h's last"
    );
    debug_assert!(
        sugar[basis.len()] >= table.degree(h.lead()),
        "h's sugar is at least its leading monomial's degree"
    );
    let t = ElementId::new(basis.len())?;
    let lead = h.lead();
    // order-family.7 cited-checked: giovini-1991 §1, §4 -- the normal selection strategy works fine for a degree-compatible term-ordering but is quite bad for the lexicographic one; the sugar option is usually the best, especially with Lex, and a selection by the degree of the S-polynomial was sometimes considerably worse; faugere-1999 §2.5, Definition 2.9, after [Gio91] -- the sugar degree of the polynomials of F4
    // The pair (i, t) with lcm τ has the sugar max(s_i − deg lm(g_i), s_t −
    // deg lm(h)) + deg τ (giovini-1991, Appendix). An element's sugar is at
    // least its leading monomial's degree, so no subtraction wraps, and
    // the larger of the two differences is the pair's fall.
    // silent-fallbacks.8 design-checked -- a sugar past u32::MAX is Limit::Overflow, which halts the run as an lcm past the format does, not the sugar saturated, a u64 sugar, or a Limit variant of its own
    // cooper-rest-gap.1 tuned-checked -- in a graded order a pair one of whose elements fell counts at the lcm's degree plus one, not at the lcm's degree (f4-gfp.3), at the sugar, at the lcm's degree plus the fall capped at 2, nor plus one only at a fall of 2 or more, of 3 or more, or of exactly 1, or only where the lcm's degree reaches the fallen element's sugar (log 0110-1, _Step 0_)
    // fallen-pair-losses.1 design-checked -- the first update that begins past the bound clears the flag for the rest of the run, not a test of the set's size at each update
    // fallen-pair-losses.2 design-checked -- a pair deferred before the latch closes keeps its sugar, the lcm's degree plus one, not every pending pair sent back to its lcm's degree, which loses the gain of the pairs deferred before the set grew past the bound (log 0111-1, _Step 0_)
    if pairs.live > DEFER_LIVE {
        pairs.defers = false;
    }
    let defers = pairs.defers;
    let pair_fall = |table: &Table<M>, i: ElementId| {
        fall(table, sugar[i.index()], basis.lead(i)).max(fall(table, sugar[t.index()], lead))
    };
    let pair_sugar = |fall: u32, degree: u32| {
        if !O::GRADED {
            return fall.checked_add(degree).ok_or(Limit::Overflow);
        }
        if fall > 0 && defers {
            return degree.checked_add(1).ok_or(Limit::Overflow);
        }
        Ok(degree)
    };

    let format = *table.format();
    let w = format.words();
    let lead_words = table.monomial(lead);
    let lead_mask = table.mask(lead);
    new.dropped.clear();
    // call-stop-latency.13 design-checked -- the update spends its scans as they go, RUN places or members at a time, and the criteria M and F a unit per kept pair tested, not one spend once the update is done (call-stop-latency.1), whose one update's scan of a set of millions of places is one stretch between two checks (not measured)
    pairs.index_element(table, basis, t, lead_words, budget)?;
    // gf2-update-time.5 design-checked -- an element a member of G_min divides goes through the whole update as any element does, the B criterion over the set, its new pairs with G_min, and M and F, not its B criterion left out, where a pair the criterion drops would stay in the set until its selection (log 0130-1)
    // gf2-update-time.6 cited-checked: gebauer-1988 §4.7 (1) -- the pair set usually holds more pairs where they are entered before criterion 1 or 2 tests them than in updatePairs, where all possible tests are done before a pair is entered
    let lead_support = match &pairs.indexing {
        Indexing::On { index, supports } => {
            let support = supports[t.index()];
            per_words!(
                w,
                b_criterion_indexed::<_, _, _>(
                    &format, table, basis, pairs, index, lead_words, support, new, budget
                )
            )?;
            Some(support)
        }
        Indexing::Undecided | Indexing::Off => {
            per_words!(
                w,
                b_criterion::<_, _, _>(
                    &format, table, basis, pairs, lead_words, lead_mask, new, budget
                )
            )?;
            None
        }
    };
    // plain-update-time.3 practice-checked: lazy deletion -- a pair the B criterion drops keeps its place, its mask's complement set to all ones, not removed by `Vec::retain` at each update, nor swapped with the last pair, which reorders the set and with it the multipliers' ids and the matrix; amended in session 0171-2 by pair-index-upkeep-time.1: the place is a hole that a later push fills, and the set's order is its pairs' keys
    pairs.vacate_all(&new.dropped, budget, Site::Update)?;

    // pair-update-rest-time.4 design-checked -- G_min's leading monomials' ids, masks, and words stand in arrays of the update's buffers in G_min's order, which the new pairs' loop reads in order, not read from the table by each member's id, a gather per member
    if new.min_ids.len() - new.tombs != basis.min().len()
        || lead_support.is_some() != new.min_index.is_some()
    {
        new.rebuild_min(table, basis, pairs.element_supports(), budget)?;
    }
    debug_assert!(
        new.min_in_step(table, basis, pairs.element_supports()),
        "the arrays hold G_min's members in its order"
    );
    #[cfg(not(test))]
    let searched = true;
    #[cfg(test)]
    let searched = ONE_PAIR.get();
    let divisor = if searched {
        per_words!(
            w,
            last_divisor::<_>(&format, new, lead_words, lead_mask, lead_support, budget)
        )?
    } else {
        None
    };
    let lead_degree = table.degree(lead);
    match divisor {
        // gf2-chain-criteria-time.1 design-checked -- where a member of G_min of nonconstant leading monomial divides lm(h), h's new pairs are the one pair M and F keep, with the last such member in G_min's order, of lcm lm(h), and no pair of P1 is formed, sorted, or tested, not every pair of P1 formed and run through M and F (divisibility-time.2, .3) (log 0170-1, 0170-2)
        Some(Divisor::Member(k)) => {
            one_pair(new, k, lead_words, lead_mask, lead_support, lead_degree);
        }
        // The constant divides lm(h) and shares no variable with it: its
        // pair is the product criterion's (.1).
        Some(Divisor::Unit) => new.kept.clear(),
        None => {
            let (keys, lead_key) = keys(&format, lead_mask, lead_support);
            per_words!(
                w,
                new_pairs::<_>(
                    &format,
                    new,
                    lead_words,
                    lead_mask,
                    lead_support,
                    keys,
                    lead_degree,
                    budget
                )
            )?;
            by_degree(&new.order, &mut new.counts, &mut new.sorted, budget)?;
            per_words!(
                w,
                criteria::<_>(&format, new, keys, lead_key, lead_degree, budget)
            )?;
        }
    }
    for chunk in new.kept.chunks(RUN) {
        for &(place, mask, degree) in chunk {
            let i = new.elements[place as usize];
            let fall = pair_fall(table, i);
            // signature-run-losses.3 cited-checked: eder-2013a §2, Theorem 2.4 -- where each pair's S-polynomial is non-minimal or has a standard representation, poly(G) is a Gröbner basis; eder-2017 §4.2, Theorem -- a basis whose regular S-pairs below T s-reduce to zero is a signature Gröbner basis up to T
            // The criteria ran before: a pair they keep for its lcm stands for
            // the others of that lcm, and one left out here is settled.
            if fall.checked_add(degree).is_some_and(|s| s <= covered) {
                continue;
            }
            let sugar = pair_sugar(fall, degree)?;
            let at = place as usize * w;
            pairs.push(
                Pair {
                    i,
                    j: t,
                    lcm: MonoId::NONE,
                    degree,
                    sugar,
                },
                mask,
                &new.words[at..at + w],
                lead_support.map(|_| new.supports[place as usize]),
            );
        }
        // A kept pair reads its element's lead's degree from the table and
        // writes its lcm's words into the set.
        budget.spend(chunk.len() * (PROBE + w), Site::Update)?;
    }

    let supports = pairs.element_supports();
    enter_min(
        table,
        basis,
        supports,
        h,
        new,
        divisor,
        searched,
        lead_support,
        budget,
    )
}

// f4-gfp.2 cited-checked: gebauer-1988 §4.6, via joux-2011 §2.2 Algorithm 2 -- with the inputs ordered by decreasing head term, an element is redundant exactly when a later one's head term divides its own
/// The update's last part: enters `h` into the basis and `G_min` and keeps
/// `G_min`'s arrays in step. `divisor` is what `last_divisor` found where
/// the update `searched`; `supports` and `lead_support` are the elements'
/// supports by id and h's where the pair set indexes. Where no member
/// divides lm(h), the update searched, and the set indexes, the members h
/// makes redundant are found by `G_min`'s index (`leaving`,
/// pair-index-upkeep-time.6) and stay in the arrays as tombstones, which
/// are rebuilt where the tombstones reach the live members before h
/// (pair-index-upkeep-time.4). Elsewhere the arrays are rebuilt where
/// members left (pair-update-rest-time.4, pair-index-upkeep-time.5). Spent
/// as the update spends; Size where the basis is full, and a stop where
/// the budget's check stops the call.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, basis, supports, and element of the update, its buffers, what its search found and whether it searched, the lead's support, and the call's budget, none of which belongs with another"
)]
fn enter_min<F: Field, M: Format, D: Module>(
    table: &Table<M>,
    basis: &mut Basis<F, D>,
    supports: Option<&[u128]>,
    h: Poly<F, D>,
    new: &mut NewPairs<M::Word>,
    divisor: Option<Divisor>,
    searched: bool,
    lead_support: Option<u128>,
    budget: &mut Budget,
) -> Result<(), Halt> {
    let format = *table.format();
    let lead = h.lead();
    let (lead_words, lead_mask) = (table.monomial(lead), table.mask(lead));
    // The lead's support where the members that leave are found by the
    // index and entombed.
    let tombs_by = lead_support.filter(|_| divisor.is_none() && searched && entombs());
    let entombed = tombs_by.is_some();
    // Over GF(2) the elements enter by increasing leading monomial
    // (gf2-update-time.1), and a member of G_min may divide h's: h then
    // joins no G_min, and no member leaves it.
    let entered = match divisor {
        // gf2-chain-criteria-time.3: the search's finding stands for
        // push_min's scan.
        Some(Divisor::Member(_) | Divisor::Unit) => basis.push_min_known(table, h, true, budget)?,
        None => match tombs_by {
            Some(support) => {
                per_words!(
                    format.words(),
                    leaving::<_>(&format, new, lead_words, support, budget)
                )?;
                basis.push_min_leaving(table, h, &new.left, budget)?
            }
            None if searched => basis.push_min_known(table, h, false, budget)?,
            None => basis.push_min(table, h, budget)?,
        },
    };
    debug_assert!(
        entered.joined || entered.left == 0,
        "members leave G_min only for an element that joins it"
    );
    // pair-index-upkeep-time.4 practice-checked: tombstones -- where the pair set indexes, a member that leaves G_min stays at its place in the arrays as a tombstone, its index bits cleared, and the arrays are rebuilt where the tombstones reach the live members, not the arrays and index rebuilt wherever members leave (pair-update-rest-time.4): the index names the live members alone to the new pairs and to the divisor's search (log 0171-1, 0171-2)
    if entombed && entered.left != 0 {
        new.entomb(budget)?;
    }
    // pair-index-upkeep-time.5 design-checked -- where the set does not index, the arrays are rebuilt where members leave, not tombstones at every count of variables, whose places last_divisor's and new_pairs' loops would read and test each without the index (log 0171-2)
    // The arrays follow G_min: the new member appended where it joined,
    // and where members left, the arrays rebuilt but where the tombstones
    // stay below the live members before h.
    let live = new.min_ids.len() - new.tombs;
    if entered.left != 0 && !(entombed && new.tombs < live) {
        new.rebuild_min(table, basis, supports, budget)?;
    } else if entered.joined {
        new.append_min(entered.id, lead_mask, lead_words, lead_support);
    }
    Ok(())
}

// call-stop-latency.47 design-checked -- the update's units bound its work's slowest case: a place scanned a unit, a place the B criterion may test TESTED, 3 · PROBE, units, a member of G_min visited a unit and its words and a new pair formed PROBE units and its lcm's words, a kept pair a pair of P1 may test a unit and its words, the kept pairs pushed PROBE units and their words, and G_min's arrays rebuilt PROBE units and its words a member, not a unit per place and member alone (call-stop-latency.1), which leaves untold the reads of the table at random that a place passing its mask is tested by (not measured)
/// Units a place the B criterion may test spends: its lcm's words read at
/// its place, and its two elements' leading monomials read from the table,
/// `PROBE` each.
const TESTED: usize = 3 * PROBE;

// call-stop-latency.50 design-checked -- the criteria M and F spend once per TESTERS, 64, pairs of P1, not a spend per chunk of RUN pairs, whose tests a kept list of thousands would take past a check's interval (log 0138-3); amended in session 0170-2: a chunk's units are gf2-chain-criteria-time.9's, in place of the most kept pairs each pair may have tested
/// The pairs of P1 the criteria M and F test between two spends.
const TESTERS: usize = 64;

// plain-update-time.1 cited-checked: drepper-2007 §6.2.1 -- where a frequently run job over a big array of records reads only some of their fields, the others are loaded into the caches for nothing, and the record is better split, the fields the job reads stored in a structure of their own
// plain-update-time.2 design-checked -- the array holds the masks' complements, so that the reducer search's block scan tests the lead's mask against them, not the masks as they are with a block scan of their own for the reverse inclusion (not measured)
/// The B criterion for a format of N words (`per_words`): `B_t` drops a
/// pair (i, j) whose lcm the lead `lead_words` divides where each of
/// lcm(i, t) and lcm(j, t) divides its lcm strictly or is the lcm of a
/// pair the product criterion drops (`drops`). The places of the pairs
/// it drops go to `new.dropped`. `find_passing` visits k where
/// `excluded[k] & lead_mask == 0`: the lead's mask passes the lcm's, which
/// LM(h) | lcm needs. A dropped pair passes where the lead is a constant
/// alone, which the run never enters (f4-gfp.22); the scan skips it, so
/// that it stays total. The scan goes `RUN` places at a time, each chunk
/// spending a unit a place and `TESTED` units a pair tested at
/// `Site::Update`; a stop where the budget's check stops the call.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, basis, and pair set the criterion reads, the lead's words and mask, the update's buffers, and the call's budget, none of which belongs with another"
)]
fn b_criterion<const N: usize, F: Field, M: Format, D: Module>(
    format: &M,
    table: &Table<M>,
    basis: &Basis<F, D>,
    pairs: &PairSet<M::Word>,
    lead_words: &[M::Word],
    lead_mask: u32,
    new: &mut NewPairs<M::Word>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    // A chunk of RUN masks holds whole blocks of the scan's eight.
    for (c, chunk) in pairs.excluded.chunks(RUN).enumerate() {
        let base = c * RUN;
        let mut tested = 0;
        find_passing(chunk, !lead_mask, |k| {
            tested += 1;
            if drops::<N, _, _, _>(format, table, basis, pairs, lead_words, base + k) {
                new.dropped.push(base + k);
            }
            false
        });
        budget.spend(chunk.len() + tested * TESTED, Site::Update)?;
    }
    Ok(())
}

// gf2-update-rest-time.1 practice-checked: bitmap index -- the B criterion takes its candidates from a bit set per support bit over the pair set's places, the sets of the new leading monomial's support intersected, not the scan of the masks' complements, which past 32 variables pass a pair for each mask bit that stands for several variables, nor reducer-search-time.4's bit set per mask bit, whose folded bits pass several times the candidates the supports pass (log 0131-1, 0137-1)
/// The B criterion by the pair set's index (`Indexing::On`), for a format
/// of N words: as `b_criterion`, over the places whose lcm's support holds
/// every bit of the lead's support `lead_support`, which every lcm the lead
/// divides does, in increasing order, so that the same pairs drop. The
/// index spends its intersection at `Site::Update` as it goes, and
/// `TESTED` units a place it may test (`Bitmap::holding`); a stop where the
/// budget's check stops the call.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, basis, pair set, and index the criterion reads, the lead's words and support, the update's buffers, and the call's budget, none of which belongs with another"
)]
fn b_criterion_indexed<const N: usize, F: Field, M: Format, D: Module>(
    format: &M,
    table: &Table<M>,
    basis: &Basis<F, D>,
    pairs: &PairSet<M::Word>,
    index: &Bitmap,
    lead_words: &[M::Word],
    lead_support: u128,
    new: &mut NewPairs<M::Word>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let NewPairs {
        buffer, dropped, ..
    } = new;
    index.holding(
        lead_support,
        pairs.pairs.len(),
        buffer,
        TESTED,
        budget,
        Site::Update,
        |k| {
            if drops::<N, _, _, _>(format, table, basis, pairs, lead_words, k) {
                dropped.push(k);
            }
        },
    )
}

// plain-zero-pairs-time.1 cited-checked: roune-2012 §D.1 -- where in(c) divides lcm(in(a), in(b)), (a, b) is eliminated right away if each of lcm(a, c) and lcm(b, c) differs from lcm(a, b) or its pair is eliminated, the relatively prime criterion's pairs among them, with no possibility of a circular argument
// plain-zero-pairs-time.3 design-checked -- the settled pairs drop over every field, not over GF(2) alone, where most of the gain lies (log 0164-1, 0164-2)
/// Whether `B_t` drops the pair at place k, for a format of N words: the
/// pair is live, the lead `lead_words` divides its lcm, and each of the
/// pairs (i, t) and (j, t) has an lcm that differs from the pair's or
/// leading monomials that are coprime. Both lcms divide the pair's, so
/// they differ from it exactly where their degree is smaller.
#[inline(always)]
fn drops<const N: usize, F: Field, M: Format, D: Module>(
    format: &M,
    table: &Table<M>,
    basis: &Basis<F, D>,
    pairs: &PairSet<M::Word>,
    lead_words: &[M::Word],
    k: usize,
) -> bool {
    if pairs.excluded[k] == DROPPED {
        return false;
    }
    let p = pairs.pairs[k];
    if !format.divides_n::<N>(lead_words, pairs.lcm_words(k)) {
        return false;
    }
    let (li, lj) = (
        table.monomial(basis.lead(p.i)),
        table.monomial(basis.lead(p.j)),
    );
    // plain-zero-pairs-time.2 design-checked -- a pair of lcm equal to the pair's counts as settled where the product criterion drops it alone, not where the update's criteria M and F drop it or where it was reduced, roune-2012 §D.1's whole rule, which needs each pair's fate recorded and an order among the pairs of one lcm (log 0164-1, 0164-2)
    let settled = |l: &[M::Word]| {
        format.lcm_degree_n::<N>(l, lead_words) != p.degree || format.coprime_n::<N>(l, lead_words)
    };
    settled(li) && settled(lj)
}

// pair-index-upkeep-time.6 practice-checked: bitmap index -- where the pair set indexes, the members an element makes redundant are found by G_min's index, the live members whose supports hold the lead's, tested by division, and handed to the basis's push and to the tombstones, not push_min_known's scan of every member through the table, nor a scan of the arrays for the members the basis marked redundant (log 0171-1, 0171-2)
/// The members of `G_min` the lead `lead_words`, of support `lead_support`,
/// makes redundant, for a format of N words (`per_words`): the live members
/// whose leading monomials the lead divides, their places in `new.leaving`
/// and their ids in `new.left`, in `G_min`'s order. They are among the
/// places `G_min`'s index names as holding every bit of the lead's support,
/// each tested live and then by division. Spent at `Site::Update` as
/// `Bitmap::holding` spends, 1 + w units a place it visits
/// (pair-index-upkeep-time.7); a stop where the budget's check stops the
/// call.
// pair-index-upkeep-time.7 design-checked -- a vacated place spends a unit, and where the set indexes 2 · PROBE units and a unit per set of the index besides, its elements' supports read and its bits cleared, as a place of the index's rebuild spends, RUN places at a time at the update's and the selection's site; the search for the members that leave spends as Bitmap::holding spends, 1 + w units a place it visits, and a member entombed a unit per set of the index; not the vacated places unspent, which a selection of a large piece or an update whose B criterion drops many pairs vacates by the thousand between two checks (log 0171-2)
#[allow(
    clippy::expect_used,
    reason = "the update finds the leaving members where the pair set indexes, and its arrays' check before rebuilds G_min's index wherever it stands missing there"
)]
fn leaving<const N: usize, M: Format>(
    format: &M,
    new: &mut NewPairs<M::Word>,
    lead_words: &[M::Word],
    lead_support: u128,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let w = format.words();
    let NewPairs {
        min_ids,
        min_words,
        min_index,
        min_dead,
        buffer,
        leaving,
        left,
        ..
    } = new;
    leaving.clear();
    left.clear();
    let index = min_index
        .as_ref()
        .expect("G_min's index stands where the pair set indexes");
    index.holding(
        lead_support,
        min_ids.len(),
        buffer,
        1 + w,
        budget,
        Site::Update,
        |k| {
            if !min_dead[k] && format.divides_n::<N>(lead_words, &min_words[k * w..(k + 1) * w]) {
                leaving.push(k);
                left.push(min_ids[k]);
            }
        },
    )
}

/// What the search for a divided element's pair found (`last_divisor`).
#[derive(Clone, Copy, Debug)]
enum Divisor {
    /// The place in `G_min`'s arrays of the last member, in `G_min`'s
    /// order, whose leading monomial divides the lead.
    Member(usize),
    /// `G_min` is the constant alone, which divides every lead.
    Unit,
}

// gf2-chain-criteria-time.2 design-checked -- a divided element's divisor is searched before any pair of P1 is formed, from the last member of G_min's order, among the members the index names where the set indexes (Bitmap::meeting_last), and the search stops at the first divisor, not each member tested for division as P1 is formed with no pair formed after the first divisor, where the divisors stand late in G_min's order, among the members that joined last, so that most of P1 was formed before its divisor (log 0170-1, 0170-2)
/// The search for a divided element's pair, for a format of N words
/// (`per_words`): `Unit` where `G_min` is one member of mask 0, the
/// constant; else the place in `G_min`'s arrays of the last member whose
/// leading monomial divides the lead `lead_words`, None where none does.
/// A member of mask 0 is no member the search returns. Where the pair set
/// indexes, `lead_support` the lead's support, the members are those
/// `G_min`'s index names as sharing a bit with it, each tested by its
/// support and then its words; elsewhere every member from the last, by its
/// mask and then its words. A stop where the budget's check stops the
/// call.
fn last_divisor<const N: usize, M: Format>(
    format: &M,
    new: &NewPairs<M::Word>,
    lead_words: &[M::Word],
    lead_mask: u32,
    lead_support: Option<u128>,
    budget: &mut Budget,
) -> Result<Option<Divisor>, Stop> {
    let w = format.words();
    let NewPairs {
        min_masks,
        min_words,
        min_index,
        min_supports,
        ..
    } = new;
    if let [0] = min_masks[..] {
        return Ok(Some(Divisor::Unit));
    }
    let divides = |k: usize| format.divides_n::<N>(&min_words[k * w..(k + 1) * w], lead_words);
    debug_assert!(
        lead_support.is_none() || min_index.is_some(),
        "G_min's index stands where the pair set indexes"
    );
    // gf2-chain-criteria-time.9 design-checked -- the search for a divided element's divisor spends a unit per word of each index set it reads and a unit and the words per member it tests, or a unit a member and the words per member whose mask passes, and the criteria M and F a unit a pair, a unit a bit of the walk of the masks of two variables, and a unit and the words per pair of the kept list's rest tested, each as it goes, not a unit and the words per pair of the whole kept list (call-stop-latency.50), as most pairs test no kept pair's words (log 0170-2)
    if let (Some(support), Some(index)) = (lead_support, min_index.as_ref()) {
        let found = index.meeting_last(support, 1 + w, budget, Site::Update, |k| {
            min_supports[k] & !support == 0 && divides(k)
        })?;
        return Ok(found.map(Divisor::Member));
    }
    let mut end = min_masks.len();
    while end > 0 {
        let start = end.saturating_sub(RUN);
        let mut passed = 0;
        let found = (start..end).rev().find(|&k| {
            let mask = min_masks[k];
            if mask == 0 || mask & !lead_mask != 0 {
                return false;
            }
            passed += 1;
            divides(k)
        });
        let visited = found.map_or(end - start, |k| end - k);
        budget.spend(visited + passed * w, Site::Update)?;
        if let Some(k) = found {
            return Ok(Some(Divisor::Member(k)));
        }
        end = start;
    }
    Ok(None)
}

/// A divided element's P1 and kept pairs (gf2-chain-criteria-time.1): the
/// one pair with the member at place k of `G_min`'s arrays, whose leading
/// monomial divides the lead `lead_words`, of mask `lead_mask`, degree
/// `lead_degree`, and where the pair set indexes support `lead_support`.
/// The pair's lcm is the lead. lm(g) | lm(h) makes lcm(g, h) = lm(h) divide
/// every lcm(i, h) (eder-2012 §2.4, Lemma 2.4.1), so M drops every pair of
/// P1 of a larger lcm, and of the pairs of lcm lm(h), those of the members
/// that divide it, F keeps the last in `G_min`'s order, the one
/// `last_divisor` finds.
fn one_pair<W: Copy>(
    new: &mut NewPairs<W>,
    k: usize,
    lead_words: &[W],
    lead_mask: u32,
    lead_support: Option<u128>,
    lead_degree: u32,
) {
    debug_assert!(k < new.min_ids.len(), "member {k} of G_min");
    debug_assert_eq!(
        new.min_masks[k] & !lead_mask,
        0,
        "a divisor's mask lies in its multiple's"
    );
    new.words.clear();
    new.words.extend_from_slice(lead_words);
    new.elements.clear();
    new.elements.push(new.min_ids[k]);
    new.supports.clear();
    new.supports.extend(lead_support);
    new.kept.clear();
    new.kept.push((0, lead_mask, lead_degree));
}

// gf2-pair-update-time.3 tuned-checked -- an update finds the kept colons of one variable first where the update before visited more than 128 members of P1, not at every update whose keys hold colons, which took more instructions on sr's calls, whose P1 is small and thinned by no such colon, nor past 16, 32, 64, or 256 members (log 0180-1)
/// The members of P1 the update before must have visited for an update to
/// find the kept colons of one variable first (gf2-pair-update-time.3).
const FIRST_LEAST: usize = 128;

/// Whether an update finds the kept colons of one variable before it
/// forms the new pairs (gf2-pair-update-time.1), where the update before
/// visited `visited` members of P1: past `FIRST_LEAST`, or in the test
/// profile what `COLONS_FIRST` holds where Some.
fn finds_first(visited: usize) -> bool {
    let first = visited > FIRST_LEAST;
    #[cfg(test)]
    let first = COLONS_FIRST.get().unwrap_or(first);
    first
}

/// The new pairs with `G_min` for a format of N words (`per_words`): the
/// coprime ones P0 are dropped, and the others, P1, get their lcms,
/// elements, and (degree, place, mask) in `new`, read off `G_min`'s arrays
/// in its order (pair-update-rest-time.4). An lcm's degree may pass the
/// format's degree bound (gf2-monomial-time.3). Where the pair set indexes,
/// `lead_support` the lead's support, the members are those `G_min`'s index
/// names as sharing a bit with it (gf2-update-rest-time.7), in its order,
/// and each new pair of P1 gets its lcm's support too; elsewhere every
/// member is tested, and P1's members are gathered before their pairs are
/// formed (gf2-pair-update-time.4).
///
/// Where the keys `keys` hold colons (`Keys::colons`) and the update before
/// visited more than `FIRST_LEAST` members of P1, the update first finds the
/// kept colons of one variable outside the lead's key: the bits of the
/// members' colons of one bit whose lcm with the lead has the lead's degree
/// `lead_degree` plus one, which `criteria` would keep in `ones`. It forms
/// no pair whose colon holds one of those bits and another bit besides,
/// which M drops for that kept colon (gf2-pair-update-time.1); where the set
/// indexes, the index leaves those members out by their sets, and the
/// members of a colon of one bit stay (.2, .5).
///
/// Spent at `Site::Update` as the members are visited, `RUN` at a time or
/// as the index spends (`Bitmap::meeting`, `Bitmap::meeting_less`). Where
/// the set indexes, a member the forming visits spends its slowest case, a
/// unit and its words read and a new pair of P1 formed, `PROBE` units and
/// its lcm's words; where the kept colons are found first, a member the pass
/// that finds them visits a unit and its words (gf2-pair-update-time.7).
/// Elsewhere a member the pass gathers spends a unit and its words read and
/// its lcm's degree, its words again, and a pair formed from the buffer
/// `PROBE` units and its lcm's words. A stop where the budget's check stops
/// the call.
#[allow(
    clippy::too_many_arguments,
    reason = "the format, the update's buffers, the lead's words, mask, support, and degree, the criteria's keys, and the call's budget, none of which belongs with another"
)]
fn new_pairs<const N: usize, M: Format>(
    format: &M,
    new: &mut NewPairs<M::Word>,
    lead_words: &[M::Word],
    lead_mask: u32,
    lead_support: Option<u128>,
    keys: Keys,
    lead_degree: u32,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let w = format.words();
    let NewPairs {
        words,
        elements,
        order,
        supports,
        min_ids,
        min_masks,
        min_words,
        min_index,
        min_supports,
        buffer,
        members,
        singles,
        visited,
        ..
    } = new;
    words.clear();
    elements.clear();
    order.clear();
    supports.clear();
    // gf2-pair-queries-time.3 practice-checked: preallocation -- the new pairs' vectors are reserved for every member of G_min before the members are visited, each new pair one of them, not grown as the pairs are pushed (log 0179-1, 0179-2)
    let count = min_ids.len();
    words.reserve(count * w);
    elements.reserve(count);
    order.reserve(count);
    if lead_support.is_some() {
        supports.reserve(count);
    }
    // The units of a member visited, its slowest case: a unit and its words
    // read, and a new pair of P1 formed.
    let formed = PROBE + w;
    let per = 1 + w + formed;
    // The units of a member the unindexed pass gathers, its slowest case: a
    // unit and its words read, and its lcm's degree.
    let gathered = 1 + 2 * w;
    let member = |k: usize| &min_words[k * w..(k + 1) * w];
    // The bits of the colon of the pair of the member at place k outside
    // the lead's key, as `criteria` reads them where the keys hold colons:
    // its support's where the set indexes, its mask's elsewhere.
    let colon_of = |k: usize| match lead_support {
        Some(support) => min_supports[k] & !support,
        None => u128::from(min_masks[k] & !lead_mask),
    };
    debug_assert!(
        lead_support.is_none() || min_index.is_some(),
        "G_min's index stands where the pair set indexes"
    );
    // gf2-pair-update-time.1 design-checked -- the update finds the kept colons of one variable outside the lead's key first, the bits of the colons of one bit whose lcm with the lead has the lead's degree plus one, and forms no pair of P1 whose colon holds one of them and another bit, which M drops for that colon, not every pair of P1 formed, sorted by degree, and dropped by `ones` in `criteria` (gf2-chain-criteria-time.4), nor the lcm words formed for the kept pairs alone with every pair of P1 sorted and tested (gf2-chain-criteria-time.7), where such a colon drops most of P1 on sat, hfe, and mqbool (log 0180-1)
    let first = keys.colons() && finds_first(*visited);
    let mut ones = 0;
    let mut seen = 0;
    singles.clear();
    members.clear();
    if let (Some(support), Some(index)) = (lead_support, min_index.as_ref()) {
        if first {
            // gf2-pair-update-time.5 design-checked -- the pass keeps every member whose colon is one bit for the forming and tests the lcm's degree only for a bit not yet among the kept colons, not only the members of a kept colon kept, each one-bit colon's degree tested, which took sat-r60-256-s1-p2 more instructions (log 0180-1)
            // gf2-pair-update-time.7 design-checked -- the pass that finds the kept colons spends a unit and the member's words a member it visits, as the index spends, its slowest case the lcm's degree read, not left unspent beside the forming's units, which a set of thousands of members would take past a check's interval (not measured)
            index.meeting(support, 1 + w, budget, Site::Update, |k| {
                seen += 1;
                let colon = min_supports[k] & !support;
                if one_bit(colon) {
                    singles.push(k);
                    if colon & ones == 0
                        && format.lcm_degree_n::<N>(member(k), lead_words) == lead_degree + 1
                    {
                        ones |= colon;
                    }
                }
            })?;
        }
    } else {
        // gf2-pair-update-time.4 design-checked -- where the set does not index, one pass over G_min's members gathers P1's members into a buffer, the kept colons found among them, and the pairs are formed from the buffer, not a pass that finds the kept colons and a second over every member that forms the pairs, which took colour-g18m76s1-k4-p0, whose P1 is a small part of G_min, more instructions (log 0180-1)
        for start in (0..min_ids.len()).step_by(RUN) {
            let end = (start + RUN).min(min_ids.len());
            for (k, &mask_i) in (start..end).zip(&min_masks[start..end]) {
                let li = member(k);
                // divisibility-time.1 design-checked -- a new pair whose leading monomials are coprime gets no lcm, and no other new pair is tested against it, not P0's lcms formed and tested against every pair of P1 (log 0043-1)
                // lm(g_i)·lm(h) divides lcm(lm(g_j), lm(h)) exactly where lm(g_i)
                // divides lm(g_j), which no two leading monomials of G_min do. The
                // product criterion drops P0, whatever its lcm's degree.
                if coprime_masked::<N, M>(format, mask_i, li, lead_mask, lead_words) {
                    continue;
                }
                members.push(k.narrow());
                if first {
                    let colon = u128::from(mask_i & !lead_mask);
                    if one_bit(colon)
                        && colon & ones == 0
                        && format.lcm_degree_n::<N>(li, lead_words) == lead_degree + 1
                    {
                        ones |= colon;
                    }
                }
            }
            budget.spend((end - start) * gathered, Site::Update)?;
        }
    }
    // The new pair of the member at place k, P0's left out where `tested`.
    let mut form = |k: usize, tested: bool| {
        let mask_i = min_masks[k];
        let li = member(k);
        if tested && coprime_masked::<N, M>(format, mask_i, li, lead_mask, lead_words) {
            return;
        }
        // A kept colon of one variable divides this pair's colon properly:
        // M drops the pair (gf2-pair-update-time.1).
        if ones != 0 {
            let colon = colon_of(k);
            if colon & ones != 0 && !one_bit(colon) {
                return;
            }
        }
        let place = elements.len().narrow();
        // gf2-chain-criteria-time.7 design-checked -- each pair of P1 keeps its lcm's words formed as the pair is made, not the words formed for the kept pairs and the pairs that scan the kept list's rest alone, with a pair's degree read off the supports where the member's leading monomial has no square, whose gain lay within the spread of the calls it does not reach, for a degree per member and a second path to the words (log 0170-1, 0170-2)
        // divisibility-time.2 design-checked -- the lcms of the new pairs of P1 are formed in the update's own words with their degrees and masks, and only the kept pairs' lcms are interned, not every new pair's lcm interned before the criteria (log 0043-1)
        // gf2-pair-queries-time.2 design-checked -- a new pair's lcm is formed in an array of N words on the stack and appended to the update's words, not the words grown by zeros and the lcm written over them, which cost a pair of P1 more instructions and time; the copy for any word count, whose N is no array's length, grows the words by zeros and writes the lcm over them (log 0179-1, 0179-2)
        let degree = if N == ANY {
            let start = words.len();
            words.resize(start + w, M::Word::default());
            format.lcm_words_n::<N>(li, lead_words, &mut words[start..])
        } else {
            let mut lcm = [M::Word::default(); N];
            let degree = format.lcm_words_n::<N>(li, lead_words, &mut lcm);
            words.extend_from_slice(&lcm);
            degree
        };
        // plain-update-time.4 design-checked -- a new pair's lcm mask is the OR of its leading monomials' masks, read from the table, not the mask formed from the lcm's words (log 0071-1)
        let mask = mask_i | lead_mask;
        debug_assert_eq!(
            mask,
            format.mask(&words[words.len() - w..]),
            "mask(lcm(a, b)) = mask(a) | mask(b)"
        );
        elements.push(min_ids[k]);
        order.push((degree, place, mask));
        if let Some(support) = lead_support {
            supports.push(min_supports[k] | support);
        }
    };
    if let (Some(support), Some(index)) = (lead_support, min_index.as_ref()) {
        // gf2-update-rest-time.7 practice-checked: bitmap index -- G_min's members stand in a bit set per support bit over their places, and the new pairs are formed with the members the union of the lead's bits' sets names, in G_min's order, not with every member, each tested for coprimality, where most members share no variable with the lead on the calls whose update weighs most (log 0137-1)
        // gf2-update-rest-time.8 design-checked -- past SUPPORT_BITS variables a member the index names is tested for coprimality as every member is, for a support bit then stands for several variables, not taken as P1's on the bit it shares (log 0137-2)
        let folded = format.variables() > SUPPORT_BITS;
        index.meeting_less(
            support,
            ones,
            singles,
            buffer,
            per,
            budget,
            Site::Update,
            |k| {
                form(k, folded);
            },
        )?;
        *visited = if first { seen } else { elements.len() };
    } else {
        for chunk in members.chunks(RUN) {
            for &k in chunk {
                form(k as usize, false);
            }
            budget.spend(chunk.len() * formed, Site::Update)?;
        }
        *visited = members.len();
    }
    Ok(())
}

// gf2-chain-criteria-time.5 design-checked -- a pair's key in the criteria M and F is past 32 variables to SUPPORT_BITS with the index its lcm's support, a bit a variable, and up to 32 variables its lcm's divisor mask whole, each bit one variable's exponent at a threshold, so that the bits of a colon name its variables; elsewhere a bit stands for several variables and no colon goes into the masks; not the masks of the colons where the set indexes alone, which left the calls of 32 variables or fewer to the scan, nor up to 32 variables each variable's first mask bit alone, which leaves a colon of a variable of the lead to the scan (log 0170-1, 0170-2)
/// What a pair's key is in the criteria M and F (`criteria`): bits of its
/// lcm, of which a divisor's lie in its multiple's. Where each bit stands
/// for one variable, the colon lcm(i, h)/lm(h) of a pair (i, h) holds no
/// more bits of a variable outside the lead's key than its exponent, and
/// the kept colons of one and two variables go into masks
/// (gf2-chain-criteria-time.4).
#[derive(Clone, Copy, Debug)]
enum Keys {
    /// The lcm's support, each bit one variable: where the set indexes, to
    /// `SUPPORT_BITS` variables.
    Supports,
    /// The lcm's divisor mask, each bit one variable's exponent reaching a
    /// threshold of its own: up to 32 variables, where the set does not
    /// index (`Format::mask`).
    Masks,
    /// A bit for several variables, or the colon masks off in the test
    /// profile: the lcm's support where the set indexes, `supports`, else
    /// its mask. Every kept pair is scanned.
    Folded { supports: bool },
}

impl Keys {
    /// Whether the kept colons go into masks.
    fn colons(self) -> bool {
        match self {
            Keys::Supports | Keys::Masks => true,
            Keys::Folded { .. } => false,
        }
    }
}

// gf2-chain-criteria-time.10 design-checked -- where the set indexes on 32 variables or fewer, which the test profile alone forces, a pair's key is its lcm's support, as gf2-update-rest-time.9 tests a kept pair where the set indexes, not its divisor mask (not measured)
/// The keys of an update's criteria M and F on `format`, whose lead has the
/// mask `lead_mask` and, where the pair set indexes, the support
/// `lead_support`, and the lead's key. In the test profile `COLON_MASKS`
/// false takes `Keys::Folded` at every count of variables.
fn keys<M: Format>(format: &M, lead_mask: u32, lead_support: Option<u128>) -> (Keys, u128) {
    #[cfg(not(test))]
    let colons = true;
    #[cfg(test)]
    let colons = COLON_MASKS.get();
    let n = format.variables();
    match lead_support {
        Some(support) if colons && n <= SUPPORT_BITS => (Keys::Supports, support),
        Some(support) => (Keys::Folded { supports: true }, support),
        None if colons && n <= u32::BITS as usize => (Keys::Masks, u128::from(lead_mask)),
        None => (Keys::Folded { supports: false }, u128::from(lead_mask)),
    }
}

// gf2-chain-criteria-time.6 practice-checked: clearing the lowest set bit, x & (x − 1) -- a colon's one bit or two outside the lead's key is told by x & (x − 1), not by count_ones, which the baseline x86-64 build computes with no instruction of its own (log 0170-2)
/// Whether x holds exactly one bit.
#[allow(
    clippy::manual_is_power_of_two,
    reason = "u128::is_power_of_two counts the ones, which the baseline x86-64 build computes with no instruction of its own (gf2-chain-criteria-time.6)"
)]
fn one_bit(x: u128) -> bool {
    x != 0 && x & (x - 1) == 0
}

/// Whether x holds exactly two bits.
fn two_bits(x: u128) -> bool {
    x != 0 && one_bit(x & (x - 1))
}

/// The lower and the higher bit of x, which holds two.
fn two_of(x: u128) -> (u32, u32) {
    debug_assert!(two_bits(x), "{x:#x} holds two bits");
    (x.trailing_zeros(), u128::BITS - 1 - x.leading_zeros())
}

// order-family.11 design-checked -- the Gebauer–Möller criteria of update run as they are, whatever the sugar of the pairs they drop, the sloppy variant, not the fussy variant, in which a pair goes only where the pairs that replace it have no larger sugar (not measured)
// divisibility-time.3 design-checked -- M and F run over P1 by increasing lcm degree, ties in G_min's order, each pair against the pairs kept so far, not each pair against the rest of P1 and the kept pairs in G_min's order (log 0043-1)
// gf2-chain-criteria-time.4 design-checked -- the criteria M and F keep the kept pairs whose colon lcm(i, h)/lm(h) is one variable outside the lead's key in a mask, and those whose colon is two such variables or the square of one in a mask per key bit, each with its index in the kept list, and scan the other kept pairs alone, where each bit of a key stands for one variable, not every kept pair scanned (pair-update-rest-time.5), where a kept colon of one variable is the first divisor of most pairs a kept pair drops, nor session 0160-1's hash set of the kept colons with every proper divisor of a colon looked up, whose lookups at random places took more time than its instructions spared (log 0170-1, 0170-2)
/// The criteria M and F for a format of N words (`per_words`), over P1 by
/// increasing degree, `new.sorted`: a pair of P1 stays, in P2, where no
/// pair of the rest of P1 or of P2 has an lcm that divides its lcm,
/// whatever the sugars. By increasing degree, a pair goes where a kept
/// pair's lcm divides its own: a proper divisor has a smaller degree and
/// came before. Where the two degrees are equal, so are the lcms, and the
/// pair takes the kept pair's place: of each lcm the last pair in `G_min`'s
/// order stays. P2 is `new.kept`, by increasing place. A pair's key is
/// `keys`'s, the lead's `lead_key`, and the lead's degree `lead_degree`.
///
/// Where a key's bit stands for one variable (`Keys::colons`), the pair
/// (i, h)'s colon c = lcm(i, h)/lm(h), of degree deg lcm(i, h) − deg lm(h),
/// has the bits C, its key's outside the lead's, each naming a variable
/// that divides c, at most as many of a variable as its exponent in c
/// (`Format::mask`). A kept pair whose colon has degree 1 and one bit in C,
/// a variable x, stands in `new.ones`; one whose colon has degree 2 and
/// two bits a and b in C, x·y or x², stands in `new.twos` at a and
/// at b, each holding the other's bit; every other kept pair stands in
/// `new.listed`. A pair goes where its C meets `ones`, else where a bit of
/// its C has a `twos` entry that meets its C, else where a kept pair of
/// `listed` whose key lies in its key has an lcm that divides its own.
/// Each test is exact: a kept colon in `ones` or `twos` divides c exactly
/// where its bits lie in C, and a colon of degree 1 or 2 whose C holds as
/// many bits as its degree is the kept one's where it holds the kept one's
/// bits. The kept lcms divide none of one another, so a pair meets at most
/// one kept pair of its own lcm, and where it meets one it meets no proper
/// divisor. Elsewhere every kept pair stands in `listed`.
///
/// The pairs of P1 spend at `Site::Update` `TESTERS` at a time: a unit a
/// pair, a unit a bit the walk of `twos` visits, and a unit and the lcm's
/// words per pair of `listed` tested (gf2-chain-criteria-time.9); P2's
/// sort by place as `sort_runs` spends. A stop where the budget's check
/// stops the call.
fn criteria<const N: usize, M: Format>(
    format: &M,
    new: &mut NewPairs<M::Word>,
    keys: Keys,
    lead_key: u128,
    lead_degree: u32,
    budget: &mut Budget,
) -> Result<(), Stop> {
    let w = format.words();
    let NewPairs {
        words,
        sorted,
        supports,
        kept,
        kept_keys,
        ones,
        one_at,
        twos,
        twos_held,
        two_at,
        listed,
        ..
    } = new;
    kept.clear();
    kept_keys.clear();
    two_at.clear();
    listed.clear();
    *ones = 0;
    let mut held = *twos_held;
    while held != 0 {
        twos[held.trailing_zeros() as usize] = 0;
        held &= held - 1;
    }
    *twos_held = 0;
    one_at.resize(SUPPORT_BITS, 0);
    twos.resize(SUPPORT_BITS, 0);
    let colons = keys.colons();
    for chunk in sorted.chunks(TESTERS) {
        let mut units = chunk.len();
        for &(degree, place, mask) in chunk {
            // gf2-update-rest-time.9 design-checked -- where the pair set indexes, a kept pair's lcm is tested for division where its support lies in the new pair's, not where its mask does, whose folded bits pass more kept pairs to the test (log 0137-1, 0137-2)
            let key = match keys {
                Keys::Supports | Keys::Folded { supports: true } => supports[place as usize],
                Keys::Masks | Keys::Folded { supports: false } => u128::from(mask),
            };
            debug_assert!(degree >= lead_degree, "an lcm's degree reaches the lead's");
            let colon = key & !lead_key;
            let colon_degree = degree - lead_degree;
            if colons {
                let one = colon & *ones;
                if one != 0 {
                    // The kept colon x divides this one, and is this
                    // one where it has degree 1.
                    if colon_degree == 1 {
                        kept[one_at[one.trailing_zeros() as usize] as usize].0 = place;
                    }
                    continue;
                }
                let mut walk = colon & *twos_held;
                let mut met = false;
                while walk != 0 {
                    units += 1;
                    if twos[walk.trailing_zeros() as usize] & colon != 0 {
                        met = true;
                        break;
                    }
                    walk &= walk - 1;
                }
                if met {
                    if colon_degree == 2 && two_bits(colon) {
                        let (a, b) = two_of(colon);
                        #[allow(
                            clippy::expect_used,
                            reason = "a kept colon of two bits within a colon of two bits holds the colon's two bits, and two_at holds every kept colon of twos"
                        )]
                        let &(_, _, k) = two_at.iter().find(|&&(p, q, _)| (p, q) == (a, b)).expect(
                            "a colon of two bits that a kept colon of two divides is the kept one",
                        );
                        kept[k as usize].0 = place;
                    }
                    continue;
                }
            }
            let at = place as usize * w;
            let lcm = &words[at..at + w];
            let divides = |k: u32| {
                let (q, _, _) = kept[k as usize];
                let at = q as usize * w;
                kept_keys[k as usize] & !key == 0 && format.divides_n::<N>(&words[at..at + w], lcm)
            };
            // pair-update-rest-time.5 design-checked -- the criteria M and F scan the kept pairs one at a time, the mask test inline and the exit at the first divisor, not by the block scan over the kept pairs' masks eight at a time, which took more instructions and time on the kept lists of a few dozen pairs (log 0107-2, _Step 6_; amended in step 6)
            let found = listed.iter().position(|&k| divides(k));
            units += found.map_or(listed.len(), |p| p + 1) * (1 + w);
            if let Some(p) = found {
                let q = &mut kept[listed[p] as usize];
                if q.2 == degree {
                    q.0 = place;
                }
            } else {
                let k = kept.len().narrow();
                kept.push((place, mask, degree));
                kept_keys.push(key);
                if colons && colon_degree == 1 && one_bit(colon) {
                    *ones |= colon;
                    one_at[colon.trailing_zeros() as usize] = k;
                } else if colons && colon_degree == 2 && two_bits(colon) {
                    let (a, b) = two_of(colon);
                    twos[a as usize] |= 1 << b;
                    twos[b as usize] |= 1 << a;
                    *twos_held |= colon;
                    two_at.push((a, b, k));
                } else {
                    listed.push(k);
                }
            }
        }
        // gf2-chain-criteria-time.9 design-checked -- the search for a divided element's divisor spends a unit per word of each index set it reads and a unit and the words per member it tests, or a unit a member and the words per member whose mask passes, and the criteria M and F a unit a pair, a unit a bit of the walk of the masks of two variables, and a unit and the words per pair of the kept list's rest tested, each as it goes, not a unit and the words per pair of the whole kept list (call-stop-latency.50), as most pairs test no kept pair's words (log 0170-2)
        budget.spend(units, Site::Update)?;
    }
    sort_runs(kept, |a, b| a.0.cmp(&b.0), budget, Site::Update)
}

// plain-update-time.5 practice-checked: counting sort -- the new pairs of P1 are ordered by degree by a counting sort, stable in G_min's order, not by a comparison sort on degree, place, and mask, nor on a 64-bit key of degree and place
// plain-update-time.6 design-checked -- the counting sort runs where the degrees span at most as many values as P1 holds pairs, and a comparison sort past that, not the counting sort at every span, nor a fixed bound on the span (not measured)
/// `order`'s entries, (degree, place, mask) by increasing place, to
/// `sorted` by increasing degree, ties by place. A counting sort by degree
/// (`sort::counting`) keeps the places' order within a degree. It runs
/// where the degrees span at most as many values as `order` holds entries,
/// so that `counts` is no longer than `order` and one more. Each pass
/// spends a unit an entry at `Site::Update` after each `RUN` entries, and
/// a comparison sort as `sort_runs` spends; a stop where the budget's check
/// stops the call.
fn by_degree(
    order: &[(u32, u32, u32)],
    counts: &mut Vec<u32>,
    sorted: &mut Vec<(u32, u32, u32)>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    debug_assert!(
        order.windows(2).all(|e| e[0].1 < e[1].1),
        "the places rise in G_min's order"
    );
    sorted.clear();
    let Some(&(first, _, _)) = order.first() else {
        return Ok(());
    };
    let (mut lo, mut hi) = (first, first);
    for chunk in order.chunks(RUN) {
        for &(d, _, _) in chunk {
            (lo, hi) = (lo.min(d), hi.max(d));
        }
        budget.spend(chunk.len(), Site::Update)?;
    }
    let span = (hi - lo) as usize + 1;
    if span > order.len() {
        for chunk in order.chunks(RUN) {
            sorted.extend_from_slice(chunk);
            budget.spend(chunk.len(), Site::Update)?;
        }
        // The places tell every two entries apart.
        return sort_runs(sorted, Ord::cmp, budget, Site::Update);
    }
    counting(
        order,
        |&(d, _, _)| (d - lo) as usize,
        span,
        counts,
        sorted,
        budget,
        Site::Update,
    )
}

// redundant-round-pairs.3 tuned-checked -- a selection splits a sugar of more than 2,048 live pairs, not of more than 512, the pairs a split takes, nor 3,072 or 4,096 (log 0095-1, _Step 3_)
/// The most live pairs of the least sugar a selection takes whole
/// (redundant-round-pairs.3), but where the repeat of a split holds, whose
/// least is `SPLIT_TAKE` (gf2-split-rest-time.1).
pub(crate) const SPLIT_LEAST: usize = 2_048;

// redundant-round-pairs.2 tuned-checked -- a split takes 512 pairs, not 256, 768, or 1,024 (log 0095-1, _Step 3_)
// split-piece-time.4 tuned-checked -- a split of the least lcms takes 512 pairs past 2,048 as the split of the first pairs did, not a piece of 1,024 nor a split past 1,024 pairs (log 0096-1, _Step 3_)
/// The pairs a selection that splits its sugar takes
/// (redundant-round-pairs.2).
pub(crate) const SPLIT_TAKE: usize = 512;

// gf2-wide-round-time.4 tuned-checked -- a piece's yield is low below half its pairs, not below a quarter, which the first pieces of sr's wide sugar, about a third, do not pass (log 0154-1, 0154-2)
/// A piece's yield is low where its round made fewer elements than its
/// pairs over `LOW_YIELD` (gf2-wide-round-time.4).
const LOW_YIELD: usize = 2;

// gf2-wide-round-time.2 tuned-checked -- the split sugar splits again only where it holds at least half the set's live pairs, not wherever its last piece's yield is low, where the selections of many pieces each pass over a set many times the sugar (log 0154-1, 0154-2)
/// The split sugar splits again only where its live pairs number at least
/// the set's over `AGAIN_SHARE` (gf2-wide-round-time.2).
const AGAIN_SHARE: usize = 2;

#[cfg(test)]
thread_local! {
    /// The test profile's least and take of the split on this thread, where
    /// Some; None leaves `SPLIT_LEAST` and `SPLIT_TAKE`.
    pub(crate) static SPLIT_BOUNDS: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// The most live pairs of the least sugar a selection takes whole, and the
/// pairs a split takes: `SPLIT_LEAST` and `SPLIT_TAKE`, or in the test
/// profile what `SPLIT_BOUNDS` holds.
fn split_bounds() -> (usize, usize) {
    #[cfg(test)]
    if let Some(bounds) = SPLIT_BOUNDS.get() {
        return bounds;
    }
    (SPLIT_LEAST, SPLIT_TAKE)
}

// f4-gfp.3 cited-checked: faugere-1999 §2.5 -- the best selection tested takes all pairs of minimal total degree; with the sugar degree the algorithm was less efficient
// order-family.8 cited-checked: faugere-1999 §2.5 -- the best function tested takes all critical pairs of minimal total degree; with the sugar degree the algorithm was less efficient
// redundant-round-pairs.1 cited-checked: huang-2010 §3.1.1 -- with fewer pairs selected each loop produces new basis elements, the new basis has some pairs removed by Buchberger's criteria, and the running time is not slowed as much; kurokawa-2023 §3.3, Algorithm 7, attributes it to Ito 2021 [45] -- a subset of 256 of the least degree's pairs is reduced at a time, the others left among the pairs; design: the split only where the least sugar holds more than SPLIT_LEAST live pairs
// redundant-round-pairs.4 design-checked -- the selection right after a split takes every pair of its least sugar, whatever their count, not a split at every selection past the bound; over GF(2) amended by gf2-wide-round-time.1
// split-piece-time.1 cited-checked: buchberger-1979 §4, via giovini-1991 §1 -- choose the pair whose lcm of head terms is minimal with respect to the term order, else a pair a criterion would spare later may be reduced; invented: the rule applied to the piece of a split round, not the first pairs in the set's order (redundant-round-pairs.5), the pairs of greatest lcm, nor half of each
// split-piece-time.2 design-checked -- a selection whose least sugar is the last split's takes it whole until a selection's least sugar lies above it, not a split at every selection past the bound but the one right after a split, nor a second piece of the first pairs where the piece made few pivots (log 0096-1, _Step 3_); over GF(2) amended by gf2-wide-round-time.1
/// Takes the pairs of least sugar out of `pairs`: in a graded order those
/// whose lcm has the least total degree, a pair of a fallen element
/// counted one degree past its lcm's while the set had not grown past its
/// bound (`update`).
/// Where that sugar holds more than `SPLIT_LEAST` live pairs, the last
/// selection took its sugar whole, and the sugar is not the last split's,
/// the selection takes the `SPLIT_TAKE` of them whose lcms are least in
/// `order`, ties in the set's order, and leaves the rest, whose count
/// `PairSet::deferred` then reads. It splits the last split's sugar again
/// so, right after a split too, where the last piece's note
/// (`PairSet::note_piece`) has it make fewer elements than its pairs over
/// `LOW_YIELD` or an element that fell (`fall`), and the sugar's live
/// pairs number at least the set's over `AGAIN_SHARE`
/// (gf2-wide-round-time.1, .2, .4, gf2-build-time.1, .2), wherever the
/// sugar holds more than `SPLIT_TAKE` live pairs (gf2-split-rest-time.1, .2);
/// the pairs the round's new elements make redundant leave the set by the
/// B criterion before the next selection. A selection whose least sugar
/// lies above the last split's forgets that split. The selection takes
/// its pairs in the set's order (`key`, pair-index-upkeep-time.2), and
/// their places become holes, as the places of the pairs the B criterion
/// dropped are (pair-index-upkeep-time.1). The pairs taken get their lcms'
/// ids, interned here from the set's words in that order
/// (pair-update-rest-time.1). Where the holes then outnumber the live
/// pairs, the set compacts, and where it indexes, its index is rebuilt over
/// the pairs left (`PairSet::compact`, pair-index-upkeep-time.3). Overflow
/// where a pair taken has an lcm past the format's degree bound
/// (gf2-monomial-time.3), Size where the table is full, and a stop where
/// interning an lcm grows the table or the selection's spend meets the
/// budget's check stopping the call; the set is then left as the run drops
/// it. The selection spends at `Site::Select` as it goes, `RUN` places at
/// a time: a unit a place of each pass and `PROBE` units an lcm interned;
/// the sort of the places taken as `sort_runs` spends; the least lcms of a
/// split as `least_lcms` spends; the places vacated as
/// `PairSet::vacate_all` spends; and the compaction as `PairSet::compact`
/// spends (call-stop-latency.12).
pub(crate) fn select<M: Format, O: Order<M>>(
    pairs: &mut PairSet<M::Word>,
    table: &mut Table<M>,
    order: &O,
    budget: &mut Budget,
) -> Result<Vec<Pair>, Limit> {
    debug_assert_eq!(pairs.w, table.format().words(), "the set's format");
    // call-stop-latency.12 design-checked -- the pair selection spends a unit a place per pass, PROBE units an lcm it interns beside the table's own spends, and the index's rebuild a unit per place and support bit, as it goes, not no spend after the round's check, which a set of millions of places passes three times and indexes again between two checks (not measured)
    // The least sugar and its live pairs, in one pass.
    let mut least: Option<(u32, usize)> = None;
    for (sugars, excluded) in pairs.pairs.chunks(RUN).zip(pairs.excluded.chunks(RUN)) {
        for (p, _) in sugars.iter().zip(excluded).filter(|&(_, &x)| x != DROPPED) {
            least = match least {
                None => Some((p.sugar, 1)),
                Some((d, n)) => match p.sugar.cmp(&d) {
                    Ordering::Less => Some((p.sugar, 1)),
                    Ordering::Equal => Some((d, n + 1)),
                    Ordering::Greater => Some((d, n)),
                },
            };
        }
        budget.spend(sugars.len(), Site::Select)?;
    }
    let Some((d, n)) = least else {
        pairs.clear();
        return Ok(Vec::new());
    };
    if pairs.split.is_some_and(|s| d > s) {
        pairs.split = None;
    }
    let (most, take) = split_bounds();
    debug_assert!(take >= 1, "a split takes a pair");
    debug_assert!(
        n <= pairs.live,
        "{n} pairs of the sugar of {live} live",
        live = pairs.live
    );
    // gf2-wide-round-time.1 cited-checked: makarim-2026 §5.4 -- when a relatively small number of the critical pairs would be sufficient, the F4 algorithm wastes a significant amount of computation and memory, which small batches of pairs do not; sakata-2026 §5.5, Remark 5, independently -- a large selection makes the first matrix excessively long, a small one increases the loops and the matrix constructions; kurokawa-2023 §3.3, Algorithm 7, line 22, attributes it to Ito 2021 [45] -- a subset of the highest degree so far that made fewer new polynomials than its pairs drops the degree's other pairs, which reduce to zero with a high probability; design: the split sugar split again while its last piece made elements from fewer than half its pairs, its pairs not dropped, not taken whole after its first piece (split-piece-time.2, redundant-round-pairs.4), nor split at every selection past the bound (log 0154-1, 0154-2); its least amended by gf2-split-rest-time.1, its rule by gf2-build-time.1
    // gf2-build-time.1 cited-checked: sakata-2026 §5 and §5.5 -- the phase in which the degrees of the S-polynomials decrease is computed by MF4, whose SelectPairLimited takes a bounded number of pairs a loop while a loop makes new elements; kurokawa-2023 §3.3, Algorithm 7, line 19, attributes it to Ito 2021 [45] -- the reduction of a degree's subsets breaks off once a new polynomial's degree is below it; design: the split sugar split again after a piece whose elements fell, whatever its yield, not taken whole after a piece of full yield, which sent mqbool-18's rest of 15,143 pairs into one round of 29,800 rows (log 0159-1, 0159-2)
    // gf2-build-time.2 tuned-checked -- one element that fell splits the sugar again, not half the piece's elements, under which bivium-114's second piece, a quarter of its elements fallen at a yield of half, sent its rest whole into a round the bound halts after its build (log 0159-1, 0159-2)
    let again = pairs.split == Some(d)
        && pairs
            .piece
            .is_some_and(|p| LOW_YIELD * p.made < p.taken || p.fallen > 0)
        && AGAIN_SHARE * n >= pairs.live;
    // gf2-split-rest-time.1 tuned-checked -- where the repeat's rule holds, the split sugar splits again past SPLIT_TAKE live pairs, the piece's size, not past the first split's least SPLIT_LEAST, under which the rest of 2,048 or fewer pairs after a piece of low yield went whole into one round (log 0156-1, 0156-2)
    // gf2-split-rest-time.2 design-checked -- the repeat's least is SPLIT_TAKE, no constant of its own, not a least between the piece and SPLIT_LEAST such as 1,024, which took sr the same path and the 16 no further (log 0156-1, 0156-2)
    let least = if again { take } else { most };
    let piece = if n > least && (again || (pairs.deferred == 0 && pairs.split != Some(d))) {
        if pairs.split != Some(d) {
            pairs.piece = None;
        }
        pairs.split = Some(d);
        Some(least_lcms(
            pairs,
            d,
            take.min(n),
            table.format(),
            order,
            budget,
        )?)
    } else {
        None
    };
    let places = if let Some(places) = piece {
        places
    } else {
        let mut places = Vec::with_capacity(n);
        for start in (0..pairs.pairs.len()).step_by(RUN) {
            let end = (start + RUN).min(pairs.pairs.len());
            places.extend(
                (start..end).filter(|&k| pairs.excluded[k] != DROPPED && pairs.pairs[k].sugar == d),
            );
            budget.spend(end - start, Site::Select)?;
        }
        let set = &pairs.pairs;
        sort_runs(
            &mut places,
            |&a, &b| key(&set[a]).cmp(&key(&set[b])),
            budget,
            Site::Select,
        )?;
        places
    };
    let mut selected = Vec::with_capacity(places.len());
    for chunk in places.chunks(RUN) {
        for &k in chunk {
            let p = pairs.pairs[k];
            // gf2-monomial-time.3 cited-checked: bachmann-1998 §2.2 -- checks on bounds of the exponent values can be realized by degree bound checks in the outer loops of the computation, as during the S-pair selection, which makes exponent value checks in the subsequent monomial operations unnecessary
            // A new pair's lcm past the bound lies in the set until a
            // selection takes it: an update that halted at its forming
            // started mqbool over in `Dense8`, where no round of the run
            // passes the bound (log 0136-1).
            if p.degree > M::MAX_DEGREE {
                return Err(Limit::Overflow);
            }
            let lcm = table.intern_words(pairs.lcm_words(k), p.degree, budget)?;
            selected.push(Pair { lcm, ..p });
        }
        budget.spend(chunk.len() * PROBE, Site::Select)?;
    }
    pairs.vacate_all(&places, budget, Site::Select)?;
    // pair-index-upkeep-time.3 cited-checked: maier-2018 §5.4 -- tombstones fill up a table and slow its find queries, which migrating every element that is no tombstone into a new table solves: the set compacts and rebuilds its index after a selection where its holes outnumber its live pairs, not never, where each intersection of the B criterion reads the index's words over every place
    if !fills() || pairs.holes.len() > pairs.live {
        pairs.compact(budget)?;
    }
    pairs.deferred = n - selected.len();
    debug_assert!(
        !selected.is_empty() && selected.len() <= n,
        "{taken} of {n} pairs taken",
        taken = selected.len()
    );
    Ok(selected)
}

// split-piece-time.3 practice-checked: quickselect -- the piece is found by a partition of the sugar's places at the piece's size and a sort of the piece, not by a sort of every place, and by `Order::cmp` on the set's words in every order, not by the keys of a keyed order
// call-stop-latency.46 design-checked -- the piece is found RUN places at a time, the take least of the pairs kept so far and the chunk's partitioned at the piece's size, each chunk spent, not one partition of every place of the sugar spent before or after it, which on a sugar of millions of pairs is one stretch between two checks; the take least of a total order are those of its chunks' least, so the piece is the same (not measured)
/// The places in `pairs` of the `take` live pairs of sugar `d` whose lcms
/// are least in `order`, ties in the set's order, in the set's order
/// (split-piece-time.1, .3, pair-index-upkeep-time.2). The lcms are
/// compared on the set's words (pair-update-rest-time.1). The places are taken `RUN` at a time, the
/// `take` least of those kept and the chunk's kept after each, a unit a
/// place and `PROBE` units a place partitioned spent at `Site::Select`; a
/// stop where the budget's check stops the call.
fn least_lcms<M: Format, O: Order<M>>(
    pairs: &PairSet<M::Word>,
    d: u32,
    take: usize,
    format: &M,
    order: &O,
    budget: &mut Budget,
) -> Result<Vec<usize>, Stop> {
    debug_assert!(take >= 1, "a piece holds a pair");
    let lcm = |at: usize| (pairs.lcm_words(at), pairs.pairs[at].degree);
    let by_key = |a: usize, b: usize| key(&pairs.pairs[a]).cmp(&key(&pairs.pairs[b]));
    let cmp = |&a: &usize, &b: &usize| {
        let ((la, da), (lb, db)) = (lcm(a), lcm(b));
        order.cmp(format, la, da, lb, db).then_with(|| by_key(a, b))
    };
    let mut places: Vec<usize> = Vec::with_capacity(take + RUN);
    for start in (0..pairs.pairs.len()).step_by(RUN) {
        let end = (start + RUN).min(pairs.pairs.len());
        places.extend(
            (start..end).filter(|&k| pairs.excluded[k] != DROPPED && pairs.pairs[k].sugar == d),
        );
        let partitioned = if places.len() > take { places.len() } else { 0 };
        if partitioned > 0 {
            places.select_nth_unstable_by(take - 1, &cmp);
            places.truncate(take);
        }
        budget.spend(end - start + partitioned * PROBE, Site::Select)?;
    }
    debug_assert!(
        places.len() == take,
        "{take} places of the sugar's, {n} found",
        n = places.len()
    );
    places.sort_unstable_by(|&a, &b| by_key(a, b));
    Ok(places)
}

// pair-index-upkeep-time.2 design-checked -- the set's order is its pairs' key, j above i, which the selection takes its places by and the piece's least lcms break their ties by, not a sequence number per place set at its push, 8 bytes a place, nor the places in no order, under which the holes would change the matrix's rows: the update pushes in (j, i) order, j the basis's next id and one update's kept pairs in G_min's order of increasing i, so the key reads the order the places held while every selection compacted the set (log 0171-1, 0171-2)
/// A pair's place in the set's order: `j` above `i`.
fn key(p: &Pair) -> u64 {
    (u64::from(p.j.raw()) << 32) | u64::from(p.i.raw())
}
