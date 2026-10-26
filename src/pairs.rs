//! Critical pairs: the update that prunes them as each element enters the
//! basis, and the selection of a round's pairs.

use crate::field::Field;
use crate::id::{ElementId, MonoId};
use crate::module::Module;
use crate::monomial::Format;
use crate::narrow::Narrow;
use crate::order::Order;
use crate::poly::{Basis, Poly};
use crate::table::{Limit, Table, find_passing};

/// A critical pair of basis elements i < j, with the id of the lcm of their
/// leading monomials, its degree, and the pair's sugar: in a graded order
/// the lcm's degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pair {
    pub(crate) i: ElementId,
    pub(crate) j: ElementId,
    pub(crate) lcm: MonoId,
    pub(crate) degree: u32,
    pub(crate) sugar: u32,
}

/// The complement that marks a pair the B criterion dropped: no mask of a
/// leading monomial but a constant's passes it (`PairSet::excluded`).
const DROPPED: u32 = u32::MAX;

/// The plain run's pair set: the pairs, and beside them their lcm masks'
/// complements.
pub(crate) struct PairSet {
    /// The pairs, in the order the update entered them, those the B
    /// criterion dropped among them until the next selection.
    pairs: Vec<Pair>,
    /// Per pair, the complement of its lcm's divisor mask; `DROPPED` for a
    /// pair the B criterion dropped. A pair's lcm is no constant, so its
    /// mask is not 0 (`Format::mask`), and no live pair's complement is
    /// `DROPPED`.
    excluded: Vec<u32>,
    /// The pairs not dropped.
    live: usize,
}

impl PairSet {
    pub(crate) fn new() -> PairSet {
        PairSet {
            pairs: Vec::new(),
            excluded: Vec::new(),
            live: 0,
        }
    }

    /// The pairs not dropped.
    pub(crate) fn len(&self) -> usize {
        self.live
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The pairs not dropped, in their order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Pair> {
        self.pairs
            .iter()
            .zip(&self.excluded)
            .filter(|&(_, &x)| x != DROPPED)
            .map(|(p, _)| p)
    }

    /// The pairs the arrays hold, those dropped since the last selection
    /// among them.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.pairs.len()
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.pairs.shrink_to_fit();
        self.excluded.shrink_to_fit();
    }

    /// Appends `pair`, whose lcm has the divisor mask `mask`.
    fn push(&mut self, pair: Pair, mask: u32) {
        debug_assert_ne!(mask, 0, "a pair's lcm is no constant");
        self.pairs.push(pair);
        self.excluded.push(!mask);
        self.live += 1;
    }
}

// divisibility-time.4 design -- the update's buffers for its new pairs stay with the run and are cleared at each element entered, not allocated per element entered (log 0043-1)
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
}

impl<W> NewPairs<W> {
    pub(crate) fn new() -> NewPairs<W> {
        NewPairs {
            words: Vec::new(),
            elements: Vec::new(),
            order: Vec::new(),
            sorted: Vec::new(),
            counts: Vec::new(),
            kept: Vec::new(),
            dropped: Vec::new(),
        }
    }

    /// Gives back the buffers' memory.
    pub(crate) fn release(&mut self) {
        self.words = Vec::new();
        self.elements = Vec::new();
        self.order = Vec::new();
        self.sorted = Vec::new();
        self.counts = Vec::new();
        self.kept = Vec::new();
        self.dropped = Vec::new();
    }
}

// f4-gfp.1 cited: gebauer-1988 §1, §4.4, §4.6 -- the three criteria are independent of the pairs considered before, which speeds up their tests, and redundant elements are cancelled as new ones enter
/// Enters the monic nonzero element `h` into the basis and updates the pair
/// set, as gebauer-1988 §4.4 `updatePairs` and §4.6 in the form of joux-2011
/// §2.2 Algorithm 2. Where the order is not graded, `sugar` holds the sugar
/// of every element, `h`'s last; in a graded order it is not read. `new`
/// holds the update's buffers, of its new pairs and of the places of the
/// pairs the B criterion drops. Overflow where the lcm of a new pair
/// that is not coprime passes the format, before any lcm is interned; Size
/// where the basis or the table is full.
pub(crate) fn update<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    _: &O,
    basis: &mut Basis<F, D>,
    pairs: &mut PairSet,
    sugar: &[u32],
    h: Poly<F, D>,
    new: &mut NewPairs<M::Word>,
) -> Result<(), Limit> {
    debug_assert!(!h.is_zero(), "h is nonzero");
    debug_assert!(
        O::GRADED || sugar.len() == basis.len() + 1,
        "a sugar per element, h's last"
    );
    debug_assert!(
        O::GRADED || sugar[basis.len()] >= table.degree(h.lead()),
        "h's sugar is at least its leading monomial's degree"
    );
    let t = ElementId::new(basis.len())?;
    let lead = h.lead();
    // order-family.7 cited: giovini-1991 §1 -- the normal selection strategy works fine if the term-ordering is degree-compatible, but is quite bad for the lexicographic one; §4 -- the sugar option is the best one, especially with Lex, and a strategy by the degree of the S-polynomial was sometimes considerably worse; faugere-1999 §2.5, Definition 2.9, after [Gio91] -- the sugar degree of the polynomials of F4
    // The pair (i, t) with lcm τ has the sugar max(s_i − deg lm(g_i), s_t −
    // deg lm(h)) + deg τ (giovini-1991, Appendix). An element's sugar is at
    // least its leading monomial's degree, so no subtraction wraps.
    let pair_sugar = |table: &Table<M>, i: ElementId, degree: u32| {
        if O::GRADED {
            return degree;
        }
        let si = sugar[i.index()] - table.degree(basis.lead(i));
        let st = sugar[t.index()] - table.degree(lead);
        si.max(st).saturating_add(degree)
    };

    // B_t: a pair (i, j) whose lcm LM(h) divides goes where lcm(i, t) and
    // lcm(j, t) both divide its lcm strictly. Both divide it, so they differ
    // from it exactly where their degree is smaller.
    let format = *table.format();
    let lead_words = table.monomial(lead);
    let lead_mask = table.mask(lead);
    new.dropped.clear();
    // plain-update-time.1 design -- the B criterion tests the new leading monomial's mask against the pairs' lcm masks, kept in an array of their own beside the pairs, eight at a time, and reads a pair's words only where its mask passes, not each pair's mask read from the table by its lcm's id, nor the mask kept in each pair (log 0070-1, 0071-1)
    // plain-update-time.2 design -- the array holds the masks' complements, so that the reducer search's block scan tests the lead's mask against them, not the masks as they are with a block scan of their own for the reverse inclusion (not measured)
    // `find_passing` visits k where excluded[k] & lead_mask == 0: the lead's
    // mask passes the lcm's, which LM(h) | lcm needs. A dropped pair passes
    // where the lead is a constant alone, which the run never enters
    // (f4-gfp.22); the update skips it, so that it stays total.
    find_passing(&pairs.excluded, !lead_mask, |k| {
        if pairs.excluded[k] == DROPPED {
            return false;
        }
        let p = pairs.pairs[k];
        if format.divides(lead_words, table.monomial(p.lcm)) {
            let di = table.lcm_degree(basis.lead(p.i), lead);
            let dj = table.lcm_degree(basis.lead(p.j), lead);
            if di != p.degree && dj != p.degree {
                new.dropped.push(k);
            }
        }
        false
    });
    // plain-update-time.3 practice: lazy deletion -- a pair the B criterion drops keeps its place, its mask's complement set to all ones, and leaves the arrays at the round's selection, not removed by `Vec::retain` at each update, nor swapped with the last pair, which reorders the set and with it the multipliers' ids and the matrix
    for &k in &new.dropped {
        pairs.excluded[k] = DROPPED;
    }
    pairs.live -= new.dropped.len();

    // The new pairs with G_min: the coprime ones P0, the others P1.
    let w = format.words();
    new.words.clear();
    new.elements.clear();
    new.order.clear();
    new.kept.clear();
    for &i in basis.min() {
        let li = basis.lead(i);
        // divisibility-time.1 design -- a new pair whose leading monomials are coprime gets no lcm, and no other new pair is tested against it, not P0's lcms formed and tested against every pair of P1 (log 0043-1)
        // lm(g_i)·lm(h) divides lcm(lm(g_j), lm(h)) exactly where lm(g_i)
        // divides lm(g_j), which no two leading monomials of G_min do. The
        // product criterion drops P0, whatever its lcm's degree.
        if table.coprime(li, lead) {
            continue;
        }
        let place = new.elements.len().narrow();
        let start = new.words.len();
        new.words.resize(start + w, M::Word::default());
        // divisibility-time.2 design -- the lcms of the new pairs of P1 are formed in the update's own words with their degrees and masks, and only the kept pairs' lcms are interned, not every new pair's lcm interned before the criteria (log 0043-1)
        let degree = table.lcm_to(li, lead, &mut new.words[start..])?;
        // plain-update-time.4 design -- a new pair's lcm mask is the OR of its leading monomials' masks, read from the table, not the mask formed from the lcm's words (log 0071-1)
        let mask = table.mask(li) | lead_mask;
        debug_assert_eq!(
            mask,
            format.mask(&new.words[start..]),
            "mask(lcm(a, b)) = mask(a) | mask(b)"
        );
        new.elements.push(i);
        new.order.push((degree, place, mask));
    }

    // order-family.11 cited: giovini-1991, Appendix -- both variants were tested, in most cases there was no difference, and the sloppy variant is the one used to build the tables
    // divisibility-time.3 design -- M and F run over P1 by increasing lcm degree, ties in G_min's order, each pair against the pairs kept so far, not each pair against the rest of P1 and the kept pairs in G_min's order (log 0043-1)
    // M and F: a pair of P1 stays, in P2, where no pair of the rest of P1 or
    // of P2 has an lcm that divides its lcm, whatever the sugars. By
    // increasing degree, a pair goes where a kept pair's lcm divides its
    // own: a proper divisor has a smaller degree and came before. Where the
    // two degrees are equal, so are the lcms, and the pair takes the kept
    // pair's place: of each lcm the last pair in G_min's order stays.
    by_degree(&new.order, &mut new.counts, &mut new.sorted);
    for &(degree, place, mask) in &new.sorted {
        let at = place as usize * w;
        let lcm = &new.words[at..at + w];
        let words = &new.words;
        let divisor = new.kept.iter_mut().find(|(q, q_mask, _)| {
            let at = *q as usize * w;
            q_mask & !mask == 0 && format.divides(&words[at..at + w], lcm)
        });
        match divisor {
            None => new.kept.push((place, mask, degree)),
            Some(q) if q.2 == degree => q.0 = place,
            Some(_) => {}
        }
    }
    new.kept.sort_unstable_by_key(|&(place, _, _)| place);
    for &(place, mask, degree) in &new.kept {
        let i = new.elements[place as usize];
        let at = place as usize * w;
        let lcm = table.intern_words(&new.words[at..at + w], degree)?;
        pairs.push(
            Pair {
                i,
                j: t,
                lcm,
                degree,
                sugar: pair_sugar(table, i, degree),
            },
            mask,
        );
    }

    // f4-gfp.2 cited: gebauer-1988 §4.6, via joux-2011 §2.2 Algorithm 2 -- with the inputs ordered by decreasing head term, an element is redundant exactly when a later one's head term divides its own
    basis.push_min(table, h)?;
    Ok(())
}

// plain-update-time.5 practice: counting sort -- the new pairs of P1 are ordered by degree by a counting sort, stable in G_min's order, not by a comparison sort on degree, place, and mask, nor on a 64-bit key of degree and place
// plain-update-time.6 design -- the counting sort runs where the degrees span at most as many values as P1 holds pairs, and a comparison sort past that, not the counting sort at every span, nor a fixed bound on the span (not measured)
/// `order`'s entries, (degree, place, mask) by increasing place, to
/// `sorted` by increasing degree, ties by place. A counting sort by degree
/// keeps the places' order within a degree. It runs where the degrees span
/// at most as many values as `order` holds entries, so that `counts` is no
/// longer than `order` and one more.
fn by_degree(order: &[(u32, u32, u32)], counts: &mut Vec<u32>, sorted: &mut Vec<(u32, u32, u32)>) {
    debug_assert!(
        order.windows(2).all(|e| e[0].1 < e[1].1),
        "the places rise in G_min's order"
    );
    sorted.clear();
    let Some(&(first, _, _)) = order.first() else {
        return;
    };
    let (lo, hi) = order.iter().fold((first, first), |(lo, hi), &(d, _, _)| {
        (lo.min(d), hi.max(d))
    });
    let span = (hi - lo) as usize + 1;
    if span > order.len() {
        sorted.extend_from_slice(order);
        sorted.sort_unstable();
        return;
    }
    // counts[d − lo + 1] counts degree d, and after the prefix sums
    // counts[d − lo] is the first place of degree d in `sorted`.
    counts.clear();
    counts.resize(span + 1, 0);
    for &(d, _, _) in order {
        counts[(d - lo) as usize + 1] += 1;
    }
    for k in 1..=span {
        counts[k] += counts[k - 1];
    }
    sorted.resize(order.len(), (0, 0, 0));
    for &entry in order {
        let next = &mut counts[(entry.0 - lo) as usize];
        sorted[*next as usize] = entry;
        *next += 1;
    }
}

// f4-gfp.3 cited: faugere-1999 §2.5 -- the best selection tested takes all pairs of minimal total degree; with the sugar degree the algorithm was less efficient
// order-family.8 cited: faugere-1999 §2.5 -- the best function tested takes all critical pairs of minimal total degree; with the sugar degree the algorithm was less efficient
/// Takes the pairs of least sugar out of `pairs`: in a graded order those
/// whose lcm has the least total degree. The pairs the B criterion dropped
/// leave the set here (plain-update-time.3), and the selection and the
/// pairs left keep the set's order.
pub(crate) fn select(pairs: &mut PairSet) -> Vec<Pair> {
    let Some(d) = pairs.iter().map(|p| p.sugar).min() else {
        pairs.pairs.clear();
        pairs.excluded.clear();
        return Vec::new();
    };
    let mut selected = Vec::new();
    let mut free = 0;
    for k in 0..pairs.pairs.len() {
        let (p, x) = (pairs.pairs[k], pairs.excluded[k]);
        if x == DROPPED {
            continue;
        }
        if p.sugar == d {
            selected.push(p);
        } else {
            pairs.pairs[free] = p;
            pairs.excluded[free] = x;
            free += 1;
        }
    }
    pairs.pairs.truncate(free);
    pairs.excluded.truncate(free);
    pairs.live = free;
    selected
}
