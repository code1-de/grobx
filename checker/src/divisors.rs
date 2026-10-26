//! The leading monomials of G, indexed for divisor queries and the chain
//! criterion.

use crate::narrow::Narrow;
use crate::poll::Poll;
use crate::poly;

// checker-stop-latency.7 design -- a divisor query the memo does not answer counts 1 + n/32 units before its scan, n the divmasks of the index, the chain criterion 1 + n/32 a pair, the memo's growth a unit per monomial it moves, a block of 4096 at a time, and an answer from the memo nothing, not a unit per query or the growth counted at once (log 0059-1)
/// Divmasks of a scan that count as one unit of work (`Poll`).
pub const MASKS: usize = 32;

/// Monomials the memo moves between two questions to the poll.
const STRIDE: usize = 4096;

/// The memo's slots at the start, a power of two.
const SLOTS: usize = 1 << 10;

/// The multiplier of the memo's hash: ⌊2^64/φ⌋, φ the golden ratio.
const MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

// checker-gfp.12 cited: roune-2012, divisor queries -- divmasks eliminate around 98% of all divisibility checks in a monomial list
/// The leading monomials of the nonzero elements of G, each with its
/// divmask, in the order of G.
pub struct Divisors {
    w: usize,
    lts: Vec<u32>,
    pub masks: Vec<u64>,
    /// The position in G of each entry.
    pub elements: Vec<usize>,
    memo: Memo,
}

impl Divisors {
    /// The index of G, given each element's monomials back to back, the
    /// leading one first; a zero element has none.
    pub fn new<'a>(w: usize, elements: impl Iterator<Item = &'a [u32]>) -> Divisors {
        let mut index = Divisors {
            w,
            lts: Vec::new(),
            masks: Vec::new(),
            elements: Vec::new(),
            memo: Memo::new(w),
        };
        for (i, monos) in elements.enumerate() {
            if !monos.is_empty() {
                let lt = &monos[..w];
                index.lts.extend_from_slice(lt);
                index.masks.push(poly::mask(lt));
                index.elements.push(i);
            }
        }
        index
    }

    pub fn lt(&self, a: usize) -> &[u32] {
        &self.lts[a * self.w..(a + 1) * self.w]
    }

    // checker-gfp.23 design -- a divisor query answers with the first element in the order of G whose leading monomial divides m (log 0001-2)
    /// The first entry, of an element other than `skip`, whose leading
    /// monomial divides m.
    fn scan(&self, m: &[u32], skip: Option<usize>) -> Option<usize> {
        let mask = poly::mask(m);
        (0..self.masks.len()).find(|&a| {
            self.masks[a] & !mask == 0
                && Some(self.elements[a]) != skip
                && poly::divides(self.lt(a), m)
        })
    }

    /// Some element of G whose leading monomial divides m. A query the
    /// memo does not answer is a unit of work for `poll`, its scan one more
    /// per `MASKS` divmasks of the index, however few the scan reaches, and
    /// the memo's growth one per monomial it
    /// moves; where it stops, the answer is None and means nothing.
    pub fn query(&mut self, m: &[u32], poll: &mut Poll) -> Option<usize> {
        // checker-gfp.13 practice: memoization -- the answer to a divisor query is kept per monomial for the rest of the check
        // checker-stop-latency.7 design -- a divisor query the memo does not answer counts 1 + n/32 units before its scan, n the divmasks of the index, the chain criterion 1 + n/32 a pair, the memo's growth a unit per monomial it moves, a block of 4096 at a time, and an answer from the memo nothing, not a unit per query or the growth counted at once (log 0059-1)
        let h = Memo::hash(m);
        let a = match self.memo.get(m, h) {
            Ok(a) => a,
            Err(slot) => {
                if poll.work(1 + self.masks.len() / MASKS) {
                    return None;
                }
                let a = match self.scan(m, None) {
                    None => u32::MAX,
                    // An element the memo cannot number is answered past it.
                    Some(i) => match u32::try_from(i) {
                        Ok(a) if a != u32::MAX => a,
                        Ok(_) | Err(_) => return Some(self.elements[i]),
                    },
                };
                if !self.memo.insert(m, slot, a, poll) {
                    return None;
                }
                a
            }
        };
        (a != u32::MAX).then(|| self.elements[a as usize])
    }

    /// An element of G other than element i whose leading monomial divides m.
    pub fn divisor_other(&self, m: &[u32], i: usize) -> Option<usize> {
        self.scan(m, Some(i)).map(|a| self.elements[a])
    }

    // checker-gfp.11 cited: buchberger-1979 §5 Criterion 3, with the selection of least lcm of §4; gebauer-1988 §4.3 -- the criterion is sound when the pairs are ordered by when they leave B
    // checker-orders.13 cited: buchberger-1985 §6.2 -- an arbitrary admissible ordering $<_T$ is fixed before Theorem 6.2, Lemma 6.4, and Theorem 6.5; gebauer-1988 §2.1 -- T is totally ordered by any admissible $<_T$
    // The pairs that excuse {a, b} have lcms that divide L, and a proper
    // divisor of L precedes L in every monomial order.
    /// Whether the pair {a, b}, a < b, with lcm L is excused: some third
    /// entry c has lt(c) | L, and the pairs {a, c} and {c, b} both precede
    /// {a, b}. Pairs are ordered by their lcm in grevlex, then by the
    /// smaller position, then by the larger, whatever the order the basis is
    /// in.
    pub fn chain(&self, a: usize, b: usize, lcm: &[u32]) -> bool {
        debug_assert!(a < b && b < self.masks.len(), "entries a < b of the index");
        debug_assert_eq!(lcm.len(), self.w, "the lcm is a monomial");
        let mask = poly::mask(lcm);
        let degree = lcm[0];
        // lt(c) | L makes lcm(a, c) and lcm(c, b) divisors of L: each is L
        // exactly where its degree is L's.
        let lcm_degree = |x: &[u32], y: &[u32]| -> u64 {
            x[1..]
                .iter()
                .zip(&y[1..])
                .map(|(s, t)| u64::from(*s.max(t)))
                .sum()
        };
        (0..self.masks.len()).any(|c| {
            if c == a || c == b || self.masks[c] & !mask != 0 {
                return false;
            }
            let lt_c = self.lt(c);
            if !poly::divides(lt_c, lcm) {
                return false;
            }
            // {a, c} precedes {a, b} where lcm(a, c) < L, or where it is L
            // and c < b. {c, b} precedes {a, b} where lcm(c, b) < L, or
            // where it is L and c < a.
            let before_ac = c < b || lcm_degree(self.lt(a), lt_c) != u64::from(degree);
            let before_cb = c < a || lcm_degree(lt_c, self.lt(b)) != u64::from(degree);
            before_ac && before_cb
        })
    }
}

/// The answers to divisor queries, by monomial: an open-addressing table
/// whose keys are kept back to back.
struct Memo {
    w: usize,
    keys: Vec<u32>,
    values: Vec<u32>,
    /// One more than the entry's number, or 0 where empty.
    slots: Vec<u32>,
}

impl Memo {
    fn new(w: usize) -> Memo {
        Memo {
            w,
            keys: Vec::new(),
            values: Vec::new(),
            slots: vec![0; SLOTS],
        }
    }

    fn hash(m: &[u32]) -> u64 {
        let mut h = 0u64;
        for &x in m {
            h = (h ^ u64::from(x)).wrapping_mul(MULTIPLIER);
            h ^= h >> 32;
        }
        h
    }

    /// The value kept for m, or the empty slot where it would go.
    fn get(&self, m: &[u32], h: u64) -> Result<u32, usize> {
        let mask = self.slots.len() - 1;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the mask keeps the slot below the table's size"
        )]
        let mut s = h as usize & mask;
        loop {
            let e = self.slots[s];
            if e == 0 {
                return Err(s);
            }
            let e = (e - 1) as usize;
            if &self.keys[e * self.w..(e + 1) * self.w] == m {
                return Ok(self.values[e]);
            }
            s = (s + 1) & mask;
        }
    }

    // checker-stop-latency.7 design -- a divisor query the memo does not answer counts 1 + n/32 units before its scan, n the divmasks of the index, the chain criterion 1 + n/32 a pair, the memo's growth a unit per monomial it moves, a block of 4096 at a time, and an answer from the memo nothing, not a unit per query or the growth counted at once (log 0059-1)
    /// Keeps `value` for m in the empty slot `slot`. Where the table grows,
    /// a monomial it moves is a unit of work for `poll`, asked a block of
    /// at most `STRIDE` at a time; false where it stops, and the table
    /// then means nothing.
    fn insert(&mut self, m: &[u32], slot: usize, value: u32, poll: &mut Poll) -> bool {
        // A memo whose next entry no u32 slot numbers keeps no more: the
        // queries it does not answer scan again.
        let Some(entry) = u32::try_from(self.values.len())
            .ok()
            .and_then(|e| e.checked_add(1))
        else {
            return true;
        };
        self.keys.extend_from_slice(m);
        self.values.push(value);
        self.slots[slot] = entry;
        if 2 * self.values.len() > self.slots.len() {
            let mut slots = vec![0u32; 2 * self.slots.len()];
            let mask = slots.len() - 1;
            for e in 0..self.values.len() {
                if e % STRIDE == 0 && poll.work(STRIDE.min(self.values.len() - e)) {
                    return false;
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "the mask keeps the slot below the table's size"
                )]
                let mut s = Memo::hash(&self.keys[e * self.w..(e + 1) * self.w]) as usize & mask;
                while slots[s] != 0 {
                    s = (s + 1) & mask;
                }
                slots[s] = Narrow::<u32>::narrow(e) + 1;
            }
            self.slots = slots;
        }
        true
    }
}
