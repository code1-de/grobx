//! Q: a sum of products, merged in a heap of streams.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::order::{Cmp, by_order};
use crate::poll::Poll;
use crate::poly::{self, Overflow, Poly};
use crate::rational::{cost, gcd_cost};

/// Words of the keys a sift compares that count as one unit of work
/// (`Poll`).
const WORDS: usize = 16;

// checker-q.7 cited: monagan-2007 §1.2 -- dividing by {g₁, …, gₛ} with a heap of quotient terms and pointers into the divisors; johnson-1974 §3, §6 -- the heap merges the products, and division builds the quotient term by term while subtracting its product; monagan-2011 §2.3 -- fraction-free merging does Θ(qf + q²g) integer multiplications
/// A sum of streams, each the product of a quotient q = num/den, a monomial,
/// and the terms of an integer polynomial from a position on. The heap yields
/// the sum's terms in decreasing order.
pub struct Heap<'a> {
    w: usize,
    order: &'a Cmp,
    streams: Vec<Stream<'a>>,
    /// The monomial each stream is multiplied by, `w` words per stream.
    shifts: Vec<u32>,
    /// The monomial of each stream's next product, `w` words per stream.
    keys: Vec<u32>,
    /// The streams with a product left, a binary max-heap by key.
    live: Vec<usize>,
    /// The streams without one, whose places a new stream takes.
    free: Vec<usize>,
    // checker-q.8 cited: monagan-2011 §2.3 -- fractions are scaled to a common denominator updated with each quotient term; all coefficient arithmetic is delayed to scale each term once
    /// The common denominator s of the quotients: every den divides it.
    s: BigInt,
    /// Counts the changes of s.
    version: u64,
    acc: BigInt,
}

struct Stream<'a> {
    r: &'a Poly<BigInt>,
    /// The element of G that r is, where the caller named one.
    element: Option<usize>,
    next: usize,
    num: BigInt,
    den: BigInt,
    /// (s/den)·num, for the s of `version`.
    scaled: BigInt,
    version: u64,
}

impl<'a> Heap<'a> {
    pub fn new(w: usize, order: &'a Cmp) -> Heap<'a> {
        Heap {
            w,
            order,
            streams: Vec::new(),
            shifts: Vec::new(),
            keys: Vec::new(),
            live: Vec::new(),
            free: Vec::new(),
            s: BigInt::one(),
            version: 0,
            acc: BigInt::zero(),
        }
    }

    /// The common denominator.
    pub fn s(&self) -> &BigInt {
        &self.s
    }

    /// Adds (num/den)·t·r, r's terms from `start` on; den > 0. `element` is
    /// the element of G that r is, where the caller names one: `next` names
    /// it where a later product passes `u32`. The stream's turn in the heap is work for
    /// `poll` as `depth` counts it, the test of s by den as `cost` counts it,
    /// and the lcm that updates s as `gcd_cost` counts it; where it stops,
    /// nothing is added.
    #[allow(
        clippy::too_many_arguments,
        reason = "a stream's polynomial, multiplier, and start, its coefficient as numerator and denominator, the element it is, and the poll, none of which belongs with another"
    )]
    pub fn add(
        &mut self,
        r: &'a Poly<BigInt>,
        t: &[u32],
        start: usize,
        num: BigInt,
        den: BigInt,
        element: Option<usize>,
        poll: &mut Poll,
    ) -> Result<(), Overflow> {
        debug_assert!(den > BigInt::ZERO, "den > 0");
        debug_assert_eq!(t.len(), self.w, "t is a monomial");
        if start >= r.len() || num.is_zero() {
            return Ok(());
        }
        if poll.work(self.depth() + if den.is_one() { 0 } else { cost(&self.s, &den) }) {
            return Ok(());
        }
        let w = self.w;
        let k = if let Some(k) = self.free.pop() {
            k
        } else {
            self.streams.push(Stream {
                r,
                element,
                next: 0,
                num: BigInt::zero(),
                den: BigInt::one(),
                scaled: BigInt::zero(),
                version: 0,
            });
            self.shifts.resize(self.shifts.len() + w, 0);
            self.keys.resize(self.keys.len() + w, 0);
            self.streams.len() - 1
        };
        let mut key = Vec::with_capacity(w);
        let product = poly::mul_into(t, r.mono(start, w), &mut key);
        if let Err(overflow) = product {
            self.free.push(k);
            return Err(overflow);
        }
        self.keys[k * w..(k + 1) * w].copy_from_slice(&key);
        self.shifts[k * w..(k + 1) * w].copy_from_slice(t);
        // checker-q.9 design -- a quotient whose numerator lc(g) does not divide is brought to lowest terms, and s becomes the lcm of the denominators (log 0002-1)
        if !den.is_one() && !self.s.is_multiple_of(&den) {
            if poll.work(gcd_cost(&self.s, &den)) {
                self.free.push(k);
                return Ok(());
            }
            self.s = self.s.lcm(&den);
            self.version += 1;
        }
        let stream = &mut self.streams[k];
        stream.r = r;
        stream.element = element;
        stream.next = start;
        stream.num = num;
        stream.den = den;
        // A version before the heap's current one marks the scaled numerator
        // stale.
        stream.version = self.version.wrapping_sub(1);
        self.live.push(k);
        self.sift_up(self.live.len() - 1);
        Ok(())
    }

    /// The sum's next term below those yielded so far: its monomial into `m`,
    /// and C, where the term is C/s. None where no term is left. Where the
    /// product of a stream's later term passes `u32`, the error is the
    /// stream's element of G, and the heap is of no further use. Each
    /// product taken is work for `poll` as `cost` counts it, and its turn at
    /// the top as `depth` counts it, and a stream's scaling to a new common
    /// denominator as `cost` counts it; where it stops, the answer is None
    /// and means nothing.
    pub fn next(
        &mut self,
        m: &mut Vec<u32>,
        poll: &mut Poll,
    ) -> Result<Option<BigInt>, Option<usize>> {
        let w = self.w;
        loop {
            let Some(&top) = self.live.first() else {
                return Ok(None);
            };
            // The products of one monomial come off a heap of about one
            // length.
            let depth = self.depth();
            m.clear();
            m.extend_from_slice(self.key(top));
            self.acc.set_zero();
            while let Some(&k) = self.live.first() {
                if self.key(k) != m.as_slice() {
                    break;
                }
                let stream = &mut self.streams[k];
                if stream.version != self.version {
                    if poll.work(cost(&self.s, &stream.den) + cost(&self.s, &stream.num)) {
                        return Ok(None);
                    }
                    stream.scaled = if stream.den == self.s {
                        stream.num.clone()
                    } else {
                        &self.s / &stream.den * &stream.num
                    };
                    stream.version = self.version;
                }
                if poll.work(depth + cost(&stream.scaled, &stream.r.coefs[stream.next])) {
                    return Ok(None);
                }
                self.acc += &stream.scaled * &stream.r.coefs[stream.next];
                stream.next += 1;
                if stream.next < stream.r.len() {
                    let mono = stream.r.mono(stream.next, w);
                    let shift = &self.shifts[k * w..(k + 1) * w];
                    // checker-orders.12 cited: bigatti-2025 §1.1, Remark 5 -- xy >_Lex z^1000: Lex is not degree compatible
                    // A later term of r may have a higher degree than the
                    // first. Every exponent is at most its degree, so a
                    // degree that fits leaves every exponent sum fitting too.
                    let Some(degree) = shift[0].checked_add(mono[0]) else {
                        return Err(stream.element);
                    };
                    let key = &mut self.keys[k * w..(k + 1) * w];
                    key[0] = degree;
                    for (slot, (s, e)) in key[1..].iter_mut().zip(shift[1..].iter().zip(&mono[1..]))
                    {
                        *slot = s + e;
                    }
                    self.sift_down(0);
                } else {
                    self.free.push(k);
                    if let Some(last) = self.live.pop()
                        && !self.live.is_empty()
                    {
                        self.live[0] = last;
                        self.sift_down(0);
                    }
                }
            }
            if !self.acc.is_zero() {
                return Ok(Some(std::mem::take(&mut self.acc)));
            }
        }
    }

    // checker-stop-latency.5 design -- the heap counts, per product it takes, 1 + 2·l·w/16 units, l the bits of its length and w a key's words, beside the product's cost, not a unit a product (log 0059-1)
    /// The units of work of a stream's turn at the top of the heap: one,
    /// and the words of the keys a sift compares, one unit per `WORDS`.
    fn depth(&self) -> usize {
        let levels = self.live.len().bit_width() as usize;
        1 + 2 * levels * self.w / WORDS
    }

    fn key(&self, k: usize) -> &[u32] {
        &self.keys[k * self.w..(k + 1) * self.w]
    }

    fn sift_up(&mut self, i: usize) {
        let (heap, keys, w) = (&mut self.live, &self.keys, self.w);
        by_order!(self.order, cmp => sift_up(heap, keys, w, i, cmp));
    }

    fn sift_down(&mut self, i: usize) {
        let (heap, keys, w) = (&mut self.live, &self.keys, self.w);
        by_order!(self.order, cmp => sift_down(heap, keys, w, i, cmp));
    }
}

/// Whether stream a's key, in `keys` at `w` words per stream, is greater
/// than stream b's by `cmp`.
#[inline(always)]
fn greater(
    keys: &[u32],
    w: usize,
    cmp: impl Fn(&[u32], &[u32]) -> Ordering,
    a: usize,
    b: usize,
) -> bool {
    cmp(&keys[a * w..(a + 1) * w], &keys[b * w..(b + 1) * w]).is_gt()
}

/// Moves the stream at position i of the max-heap `heap` up to its place.
#[inline(always)]
fn sift_up(
    heap: &mut [usize],
    keys: &[u32],
    w: usize,
    mut i: usize,
    cmp: impl Fn(&[u32], &[u32]) -> Ordering,
) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if !greater(keys, w, &cmp, heap[i], heap[parent]) {
            break;
        }
        heap.swap(i, parent);
        i = parent;
    }
}

/// Moves the stream at position i of the max-heap `heap` down to its place.
#[inline(always)]
fn sift_down(
    heap: &mut [usize],
    keys: &[u32],
    w: usize,
    mut i: usize,
    cmp: impl Fn(&[u32], &[u32]) -> Ordering,
) {
    let n = heap.len();
    loop {
        let (left, right) = (2 * i + 1, 2 * i + 2);
        let mut largest = i;
        if left < n && greater(keys, w, &cmp, heap[left], heap[largest]) {
            largest = left;
        }
        if right < n && greater(keys, w, &cmp, heap[right], heap[largest]) {
            largest = right;
        }
        if largest == i {
            return;
        }
        heap.swap(i, largest);
        i = largest;
    }
}
