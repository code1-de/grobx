//! A bitmap index over places, by the supports of the monomials that stand
//! there (gf2-update-rest-time.1, .5 to .7): per support bit a bit set of
//! the places whose support holds it, so that the places whose support
//! holds every bit of a query's, or shares a bit with it, are read off the
//! sets of the query's bits alone. The plain run's pair set keeps one over
//! its pairs' lcms, and its update one over `G_min`'s leading monomials
//! (`pairs`), which it also searches from the last place
//! (gf2-chain-criteria-time.2). A place that leaves is taken out of its
//! sets, and one that joins put in, between the rebuilds
//! (pair-index-upkeep-time.1, .4).

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::monomial::SUPPORT_BITS;
use crate::sort::RUN;

// gf2-update-rest-time.6 practice-checked: strip-mining -- an intersection ANDs the sets a chunk of 256 words at a time into a buffer, set by set, not the sets' words ANDed place word by place word, whose index arithmetic per set took more instructions than the chunks and their candidates (log 0137-1, 0137-2)
/// The words of a set an intersection or a visit with places left out
/// takes at a time, 16,384 places: the buffer's 2 KiB a half stay in the
/// first-level cache beside the sets' chunks.
const CHUNK: usize = 256;

/// The index: per support bit below its bits, the set of the places whose
/// support holds the bit, place k at bit k mod 64 of word k / 64.
pub(crate) struct Bitmap {
    // gf2-update-rest-time.5 design-checked -- each set stands in a vector of its own, not every set in one array of a stride of words each, whose growth copies every set into a new array beside the old and took sat-r75-320-s1's peak higher (log 0137-1, 0137-2)
    /// The sets, each `words` long.
    sets: Vec<Vec<u64>>,
    /// The words of each set.
    words: usize,
}

impl Bitmap {
    /// An index over no place of supports of `bits` bits, at most
    /// `SUPPORT_BITS`: no support it is given holds a bit past them.
    pub(crate) fn new(bits: usize) -> Bitmap {
        debug_assert!(
            (1..=SUPPORT_BITS).contains(&bits),
            "{bits} bits, 1 to {SUPPORT_BITS}"
        );
        Bitmap {
            sets: (0..bits).map(|_| Vec::new()).collect(),
            words: 0,
        }
    }

    /// Places k's support `support`: its place joins the set of each of its
    /// bits. The sets grow to cover k.
    pub(crate) fn set(&mut self, k: usize, support: u128) {
        let w = k / 64;
        if w >= self.words {
            self.words = w + 1;
            for set in &mut self.sets {
                set.resize(self.words, 0);
            }
        }
        self.mark(k, support);
    }

    /// `set` at a place the sets cover.
    #[inline]
    fn mark(&mut self, k: usize, support: u128) {
        debug_assert!(k / 64 < self.words, "place {k} past the sets");
        debug_assert!(
            self.sets.len() == SUPPORT_BITS || support >> self.sets.len() == 0,
            "a support past the index's {bits} bits",
            bits = self.sets.len()
        );
        let bit = 1 << (k % 64);
        let mut x = support;
        while x != 0 {
            self.sets[x.trailing_zeros() as usize][k / 64] |= bit;
            x &= x - 1;
        }
    }

    /// Takes place k out of the set of each bit of `support`, at a place
    /// the sets cover: the place left, where `support` is the one `set`
    /// placed there (pair-index-upkeep-time.1). It spends nothing; its
    /// callers spend.
    pub(crate) fn unset(&mut self, k: usize, support: u128) {
        debug_assert!(k / 64 < self.words, "place {k} past the sets");
        debug_assert!(
            self.sets.len() == SUPPORT_BITS || support >> self.sets.len() == 0,
            "a support past the index's {bits} bits",
            bits = self.sets.len()
        );
        let keep = !(1 << (k % 64));
        let mut x = support;
        while x != 0 {
            self.sets[x.trailing_zeros() as usize][k / 64] &= keep;
            x &= x - 1;
        }
    }

    /// The support bits the index has a set for.
    pub(crate) fn bits(&self) -> usize {
        self.sets.len()
    }

    /// Empties every set; the sets keep their memory.
    pub(crate) fn clear(&mut self) {
        for set in &mut self.sets {
            set.clear();
        }
        self.words = 0;
    }

    /// Gives back the sets' slack.
    pub(crate) fn shrink_to_fit(&mut self) {
        for set in &mut self.sets {
            set.shrink_to_fit();
        }
    }

    // call-stop-latency.45 design-checked -- the index's rebuild, intersection, and union spend on the call's budget as they go, a chunk at a time: the sets' words a unit each, and each place the chunk may mark or visit the caller's units per place, and a unit per set it may mark, an upper bound with no count per place, not their callers' spends before or after them, which the index's own work would pass between two checks on a set of millions of places, nor a count of the places visited and the bits marked, which took sat-r60-256-s1-p2's update and selection to more instructions (log 0138-3)
    /// The index over the places below `places` afresh, place k's support
    /// `support(k)`, spent at `site` as it goes: each set's words a unit
    /// each, then `per` units a place, the cost of `support`, and a unit per
    /// set it may mark, `RUN` places at a time. A stop where the budget's
    /// check stops the call, the index then part built over sets of their
    /// full length.
    pub(crate) fn rebuild(
        &mut self,
        places: usize,
        support: impl Fn(usize) -> u128,
        per: usize,
        budget: &mut Budget,
        site: Site,
    ) -> Result<(), Stop> {
        self.words = places.div_ceil(64);
        for set in &mut self.sets {
            set.clear();
            set.resize(self.words, 0);
            budget.spend(self.words, site)?;
        }
        let place = per + self.sets.len();
        for start in (0..places).step_by(RUN) {
            let end = (start + RUN).min(places);
            for k in start..end {
                self.mark(k, support(k));
            }
            budget.spend((end - start) * place, site)?;
        }
        Ok(())
    }

    /// Visits in increasing order every place below `places` whose support
    /// holds each bit of `query`: every place below `places` where `query`
    /// is 0. The intersection takes its chunks in `buffer`. Spent at `site`
    /// as it goes, a chunk at a time: a unit per word of a set it reads, and
    /// `per` units, the cost of `visit`, per place the chunk holds. A stop
    /// where the budget's check stops the call, the places after it not
    /// visited.
    #[allow(
        clippy::too_many_arguments,
        reason = "the query, the places, and the buffer of the intersection, the units of a visit, the call's budget and its site, and the visit, none of which belongs with another"
    )]
    pub(crate) fn holding(
        &self,
        query: u128,
        places: usize,
        buffer: &mut Vec<u64>,
        per: usize,
        budget: &mut Budget,
        site: Site,
        mut visit: impl FnMut(usize),
    ) -> Result<(), Stop> {
        debug_assert!(
            self.sets.len() == SUPPORT_BITS || query >> self.sets.len() == 0,
            "a query past the index's {bits} bits",
            bits = self.sets.len()
        );
        if query == 0 {
            for start in (0..places).step_by(RUN) {
                let end = (start + RUN).min(places);
                (start..end).for_each(&mut visit);
                budget.spend((end - start) * per, site)?;
            }
            return Ok(());
        }
        let (bits, n) = bits_of(query);
        let first = &self.sets[bits[0]];
        let rest = &bits[1..n];
        buffer.resize(CHUNK, 0);
        let words = self.words.min(places.div_ceil(64));
        let mut base = 0;
        while base < words {
            let len = CHUNK.min(words - base);
            let chunk = &mut buffer[..len];
            chunk.copy_from_slice(&first[base..base + len]);
            for &b in rest {
                for (a, &x) in chunk.iter_mut().zip(&self.sets[b][base..base + len]) {
                    *a &= x;
                }
            }
            // gf2-pair-queries-time.1 design-checked -- the places of the chunk's last word past `places` are cleared once before the visit, and a word of the intersection is visited only where it is nonzero, not each word's place and its mask against `places` computed as it is visited, which cost the B criterion's mostly zero words of an intersection more instructions and time (log 0179-1, 0179-2)
            // The chunk's last word holds places past `places` only where it
            // is the last of `words`, whose places begin below `places`.
            let end = (base + len) * 64;
            if end > places {
                chunk[len - 1] &= (1 << (places + 64 - end)) - 1;
            }
            for (j, &x) in chunk.iter().enumerate() {
                if x != 0 {
                    visit_bits(x, (base + j) * 64, &mut visit);
                }
            }
            budget.spend(len * (n + 64 * per), site)?;
            base += len;
        }
        Ok(())
    }

    /// Visits in increasing order every place whose support shares a bit
    /// with `query`. Spent at `site` as it goes, `CHUNK` words of the sets
    /// at a time: a unit per word of a set it reads, and `per` units, the
    /// cost of `visit`, per place the chunk holds. A stop where the
    /// budget's check stops the call, the places after it not visited.
    pub(crate) fn meeting(
        &self,
        query: u128,
        per: usize,
        budget: &mut Budget,
        site: Site,
        mut visit: impl FnMut(usize),
    ) -> Result<(), Stop> {
        debug_assert!(
            self.sets.len() == SUPPORT_BITS || query >> self.sets.len() == 0,
            "a query past the index's {bits} bits",
            bits = self.sets.len()
        );
        let (bits, n) = bits_of(query);
        let bits = &bits[..n];
        for base in (0..self.words).step_by(CHUNK) {
            let end = (base + CHUNK).min(self.words);
            for w in base..end {
                let x = bits.iter().fold(0, |a, &b| a | self.sets[b][w]);
                visit_bits(x, w * 64, &mut visit);
            }
            budget.spend((end - base) * (n + 64 * per), site)?;
        }
        Ok(())
    }

    // gf2-pair-update-time.2 design-checked -- the forming's visit takes the places the query names less those whose support holds a bit of `less`, a chunk of words at a time, the query's sets and those of `less` ORed each into a buffer set by set, with the places of `extra` added back, not each member visited and tested by its colon, whose dropped members' visits cost the update more instructions, nor the sets of `less` ORed a word at a time as `meeting` ORs the query's, whose bounds checks on a dozen kept colons' sets took more instructions than the members they left out, nor `meeting` itself taking `less` and `extra` by chunks, which took sat-r60-256-s1-p2's pass that finds the kept colons more instructions a member (log 0180-1)
    /// Visits in increasing order every place whose support shares a bit
    /// with `query` and holds no bit of `less`, and every place of `extra`,
    /// increasing, whatever its support; a place both name is visited once.
    /// A chunk of `CHUNK` words at a time, the sets of `query`'s bits and of
    /// `less`'s are each joined by an OR, set by set, in a half of `buffer`.
    /// Spent at `site` as it goes: a unit per word of a set it reads, and
    /// `per` units, the cost of `visit`, per place the chunk holds. A stop
    /// where the budget's check stops the call, the places after it not
    /// visited.
    #[allow(
        clippy::too_many_arguments,
        reason = "the query, the places it leaves out and adds, the buffer of the sets' chunks, the units of a visit, the call's budget and its site, and the visit, none of which belongs with another"
    )]
    pub(crate) fn meeting_less(
        &self,
        query: u128,
        less: u128,
        extra: &[usize],
        buffer: &mut Vec<u64>,
        per: usize,
        budget: &mut Budget,
        site: Site,
        mut visit: impl FnMut(usize),
    ) -> Result<(), Stop> {
        debug_assert!(
            self.sets.len() == SUPPORT_BITS || (query | less) >> self.sets.len() == 0,
            "a query past the index's {bits} bits",
            bits = self.sets.len()
        );
        debug_assert!(
            extra.windows(2).all(|p| p[0] < p[1])
                && extra.last().is_none_or(|&k| k < self.words * 64),
            "the places added increase within the index"
        );
        let (n, m) = (query.count_ones() as usize, less.count_ones() as usize);
        buffer.resize(2 * CHUNK, 0);
        let (named, left) = buffer.split_at_mut(CHUNK);
        let mut next = 0;
        for base in (0..self.words).step_by(CHUNK) {
            let len = CHUNK.min(self.words - base);
            let (named, left) = (&mut named[..len], &mut left[..len]);
            for (into, of) in [(&mut *named, query), (&mut *left, less)] {
                into.fill(0);
                let mut rest = of;
                while rest != 0 {
                    let b = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    for (a, &x) in into.iter_mut().zip(&self.sets[b][base..base + len]) {
                        *a |= x;
                    }
                }
            }
            for (j, (&a, &o)) in named.iter().zip(left.iter()).enumerate() {
                let w = base + j;
                let mut x = a & !o;
                while next < extra.len() && extra[next] < (w + 1) * 64 {
                    x |= 1 << (extra[next] - w * 64);
                    next += 1;
                }
                visit_bits(x, w * 64, &mut visit);
            }
            budget.spend(len * (n + m + 64 * per), site)?;
        }
        Ok(())
    }

    // gf2-chain-criteria-time.2 design-checked -- the search for a divided element's divisor takes the places from the last, each word's from its highest bit, and stops at the first that passes, not every place visited in increasing order, where the divisors stand among the members that joined last (log 0170-1, 0170-2)
    /// The last place whose support shares a bit with `query` and passes
    /// `test`: the words taken from the last, each word's places from its
    /// highest bit, and the search ends at the first place that passes.
    /// Spent at `site` as it goes, `CHUNK` words of the sets at a time: a
    /// unit per word of a set it reads, and `per` units, the cost of
    /// `test`, per place it tests (gf2-chain-criteria-time.9). A stop where
    /// the budget's check stops the call, the places before it not tested.
    pub(crate) fn meeting_last(
        &self,
        query: u128,
        per: usize,
        budget: &mut Budget,
        site: Site,
        mut test: impl FnMut(usize) -> bool,
    ) -> Result<Option<usize>, Stop> {
        debug_assert!(
            self.sets.len() == SUPPORT_BITS || query >> self.sets.len() == 0,
            "a query past the index's {bits} bits",
            bits = self.sets.len()
        );
        let (bits, n) = bits_of(query);
        let bits = &bits[..n];
        let mut end = self.words;
        while end > 0 {
            let base = end.saturating_sub(CHUNK);
            let (mut read, mut tested) = (0, 0);
            let mut found = None;
            'words: for w in (base..end).rev() {
                read += 1;
                let mut x = bits.iter().fold(0, |a, &b| a | self.sets[b][w]);
                while x != 0 {
                    let i = u64::BITS - 1 - x.leading_zeros();
                    let k = w * 64 + i as usize;
                    tested += 1;
                    if test(k) {
                        found = Some(k);
                        break 'words;
                    }
                    x ^= 1 << i;
                }
            }
            budget.spend(read * n + tested * per, site)?;
            if found.is_some() {
                return Ok(found);
            }
            end = base;
        }
        Ok(None)
    }
}

/// The bits set in x, increasing, and their count.
fn bits_of(x: u128) -> ([usize; SUPPORT_BITS], usize) {
    let mut bits = [0; SUPPORT_BITS];
    let mut n = 0;
    let mut x = x;
    while x != 0 {
        bits[n] = x.trailing_zeros() as usize;
        n += 1;
        x &= x - 1;
    }
    (bits, n)
}

/// Visits from + i for every bit i set in x, increasing.
#[inline]
fn visit_bits(x: u64, from: usize, visit: &mut impl FnMut(usize)) {
    let mut x = x;
    while x != 0 {
        visit(from + x.trailing_zeros() as usize);
        x &= x - 1;
    }
}
