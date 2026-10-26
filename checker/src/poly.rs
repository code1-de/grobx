//! Monomials, polynomials over either field, and the geobucket over GF(p),
//! in the order of the check (`order.rs`).
//!
//! A monomial in n variables is n+1 `u32`: the total degree, then the
//! exponents in the instance's variable order.

use std::cmp::Ordering;

use crate::field::Field;
use crate::order::{Cmp, by_order};
use crate::poll::Poll;

/// A degree or an exponent went past `u32`.
#[derive(Clone, Copy, Debug)]
pub struct Overflow;

/// a·b into `out`.
#[inline]
pub fn mul_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> Result<(), Overflow> {
    // Every exponent is at most its degree, so a degree that fits leaves
    // every exponent sum fitting too.
    let degree = a[0].checked_add(b[0]).ok_or(Overflow)?;
    out.push(degree);
    out.extend(a[1..].iter().zip(&b[1..]).map(|(x, y)| x + y));
    Ok(())
}

/// Whether a divides b.
#[inline]
pub fn divides(a: &[u32], b: &[u32]) -> bool {
    a[0] <= b[0] && a[1..].iter().zip(&b[1..]).all(|(x, y)| x <= y)
}

/// b / a into `out`, where a divides b.
pub fn div_into(b: &[u32], a: &[u32], out: &mut Vec<u32>) {
    out.extend(b.iter().zip(a).map(|(x, y)| x - y));
}

/// lcm(a, b) into `out`.
pub fn lcm_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> Result<(), Overflow> {
    let start = out.len();
    out.push(0);
    let mut degree = 0u32;
    for (x, y) in a[1..].iter().zip(&b[1..]) {
        let e = *x.max(y);
        degree = degree.checked_add(e).ok_or(Overflow)?;
        out.push(e);
    }
    out[start] = degree;
    Ok(())
}

/// One bit per variable, set where the exponent is positive, for the first
/// 64 variables. a divides b only where mask(a) & !mask(b) == 0.
#[inline]
pub fn mask(m: &[u32]) -> u64 {
    let mut bits = 0u64;
    for (v, e) in m[1..].iter().take(64).enumerate() {
        if *e > 0 {
            bits |= 1 << v;
        }
    }
    bits
}

/// Whether a and b share no variable.
pub fn coprime(a: &[u32], b: &[u32]) -> bool {
    a[1..].iter().zip(&b[1..]).all(|(x, y)| *x == 0 || *y == 0)
}

// checker-gfp.7 design -- a polynomial is two flat arrays, the monomials back to back and the coefficients (not measured)
/// A polynomial: its terms, no coefficient zero, in strictly decreasing
/// order, except inside the geobucket, which keeps them increasing, and
/// while it is read, before `normalize`.
/// `w` is the width of a monomial, n+1. The coefficients are residues over
/// GF(p) and integers over Q.
#[derive(Clone, Debug)]
pub struct Poly<C> {
    pub monos: Vec<u32>,
    pub coefs: Vec<C>,
}

impl<C> Default for Poly<C> {
    fn default() -> Poly<C> {
        Poly {
            monos: Vec::new(),
            coefs: Vec::new(),
        }
    }
}

impl<C> Poly<C> {
    pub fn len(&self) -> usize {
        self.coefs.len()
    }

    pub fn is_zero(&self) -> bool {
        self.coefs.is_empty()
    }

    pub fn clear(&mut self) {
        self.monos.clear();
        self.coefs.clear();
    }

    #[inline]
    pub fn mono(&self, i: usize, w: usize) -> &[u32] {
        &self.monos[i * w..(i + 1) * w]
    }

    pub fn push(&mut self, m: &[u32], c: C) {
        self.monos.extend_from_slice(m);
        self.coefs.push(c);
    }

    // checker-stop-latency.10 design -- the order of the terms read is tested a block of 4096 comparisons at a time, each block counted before it, and terms out of order are sorted in one count, not the whole test counted in one count before it (log 0059-1)
    /// Sorts terms given in any order into decreasing order and combines
    /// like terms with `add`, dropping those that cancel. A comparison is a
    /// unit of work for `poll`, counted before it is made, the test of the
    /// order a block of at most `STRIDE` at a time and the sort at once;
    /// where it stops, the terms are left as they stand.
    pub fn normalize(
        &mut self,
        w: usize,
        order: &Cmp,
        add: impl Fn(&mut C, C),
        is_zero: impl Fn(&C) -> bool,
        poll: &mut Poll,
    ) {
        let n = self.len();
        let mut ordered = true;
        let mut start = 1;
        while ordered && start < n {
            let end = n.min(start + STRIDE);
            if poll.work(end - start) {
                return;
            }
            ordered = by_order!(order, cmp => {
                (start..end).all(|i| cmp(self.mono(i - 1, w), self.mono(i, w)).is_gt())
            });
            start = end;
        }
        if ordered {
            return;
        }
        if poll.work(n * n.bit_width() as usize) {
            return;
        }
        let mut sorted: Vec<usize> = (0..n).collect();
        by_order!(order, cmp => {
            sorted.sort_by(|&i, &j| cmp(self.mono(j, w), self.mono(i, w)));
        });
        let monos = std::mem::take(&mut self.monos);
        let mut coefs: Vec<Option<C>> = std::mem::take(&mut self.coefs)
            .into_iter()
            .map(Some)
            .collect();
        let mut out: Poly<C> = Poly::default();
        for i in sorted {
            let m = &monos[i * w..(i + 1) * w];
            #[allow(
                clippy::expect_used,
                reason = "sorted is a permutation of the term indices"
            )]
            let c = coefs[i].take().expect("each term is taken once");
            let last = out.len();
            if last > 0 && out.mono(last - 1, w) == m {
                add(&mut out.coefs[last - 1], c);
            } else {
                out.push(m, c);
            }
        }
        for (i, c) in out.coefs.into_iter().enumerate() {
            if !is_zero(&c) {
                self.monos.extend_from_slice(&out.monos[i * w..(i + 1) * w]);
                self.coefs.push(c);
            }
        }
    }
}

// checker-gfp.8 cited: yan-1998 §4, §5 _Choice of d_ -- buckets of capacity d^i, summands added only to buckets of comparable size; d = 4 worked best
/// A sum of polynomials kept in buckets of capacity 4^(i+1), each a
/// polynomial in increasing order, so that its leading term is its last.
pub struct Geobucket {
    field: Field,
    w: usize,
    order: Cmp,
    buckets: Vec<Poly<u64>>,
    /// The summand being added, increasing.
    summand: Poly<u64>,
    merged: Poly<u64>,
}

fn capacity(i: usize) -> usize {
    4usize.saturating_pow(u32::try_from(i + 1).unwrap_or(u32::MAX))
}

impl Geobucket {
    pub fn new(field: Field, w: usize, order: Cmp) -> Geobucket {
        Geobucket {
            field,
            w,
            order,
            buckets: Vec::new(),
            summand: Poly::default(),
            merged: Poly::default(),
        }
    }

    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
    }

    // checker-stop-latency.3 design -- the geobucket builds a summand and merges a block of at most 4096 terms of each operand at a time, each block counted before it, the merge's loop over the block's terms cut as slices of their own, not a merge counted at once or a count per term (log 0059-1)
    /// Adds c·m·f, where c is nonzero and f is decreasing, leaving out f's
    /// first `skip` terms. Each term written into the summand, and each
    /// term a merge passes, is a unit of work for `poll`; where it stops, the sum is left
    /// as it stands and means nothing.
    pub fn add(
        &mut self,
        c: u64,
        m: &[u32],
        f: &Poly<u64>,
        skip: usize,
        poll: &mut Poll,
    ) -> Result<(), Overflow> {
        let w = self.w;
        let field = self.field;
        self.summand.clear();
        // The terms from the last, a block of at most `STRIDE` at a time.
        let mut end = f.len();
        while end > skip {
            let start = skip.max(end.saturating_sub(STRIDE));
            if poll.work(end - start) {
                return Ok(());
            }
            for t in (start..end).rev() {
                let coef = if c == 1 {
                    f.coefs[t]
                } else {
                    field.mul(c, f.coefs[t])
                };
                // c and f's coefficients are nonzero residues mod a prime, so
                // their product is nonzero.
                debug_assert_ne!(coef, 0);
                mul_into(m, f.mono(t, w), &mut self.summand.monos)?;
                self.summand.coefs.push(coef);
            }
            end = start;
        }
        self.add_summand(poll);
        Ok(())
    }

    fn add_summand(&mut self, poll: &mut Poll) {
        let len = self.summand.len();
        if len == 0 {
            return;
        }
        let mut i = 0;
        while capacity(i) < len {
            i += 1;
        }
        if self.buckets.len() <= i {
            self.buckets.resize_with(i + 1, Poly::default);
        }
        std::mem::swap(&mut self.merged, &mut self.summand);
        loop {
            merge(
                self.field,
                self.w,
                &self.order,
                &self.buckets[i],
                &self.merged,
                &mut self.summand,
                poll,
            );
            if poll.stopped() {
                return;
            }
            std::mem::swap(&mut self.merged, &mut self.summand);
            self.buckets[i].clear();
            if self.merged.len() <= capacity(i) {
                std::mem::swap(&mut self.buckets[i], &mut self.merged);
                return;
            }
            i += 1;
            if self.buckets.len() <= i {
                self.buckets.push(Poly::default());
            }
        }
    }

    // checker-stop-latency.4 design -- an extraction counts a unit per bucket each turn, before it, not one step whatever it cancels (not measured)
    /// Removes the leading term and returns its coefficient, with its
    /// monomial in `m`; None where the sum is zero. Each turn, which takes
    /// out a term of a bucket at least, is a unit of work for `poll` per
    /// bucket; where it stops, the answer is None and means nothing.
    pub fn lead(&mut self, m: &mut Vec<u32>, poll: &mut Poll) -> Option<u64> {
        let w = self.w;
        loop {
            if poll.work(self.buckets.len()) {
                return None;
            }
            let buckets = &self.buckets;
            let j = by_order!(&self.order, cmp => top_by(buckets, w, cmp))?;
            let lead = self.buckets[j].len() - 1;
            m.clear();
            m.extend_from_slice(self.buckets[j].mono(lead, w));
            let mut c = 0;
            for b in &mut self.buckets {
                let len = b.len();
                if len > 0 && b.mono(len - 1, w) == m.as_slice() {
                    c = self.field.add(c, b.coefs[len - 1]);
                    b.coefs.pop();
                    b.monos.truncate((len - 1) * w);
                }
            }
            if c != 0 {
                return Some(c);
            }
        }
    }
}

/// The bucket whose last term is the largest by `cmp`; None where every
/// bucket is empty.
#[inline(always)]
fn top_by(
    buckets: &[Poly<u64>],
    w: usize,
    cmp: impl Fn(&[u32], &[u32]) -> Ordering,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, b) in buckets.iter().enumerate() {
        if b.is_zero() {
            continue;
        }
        let top = b.mono(b.len() - 1, w);
        match best {
            Some(j) => {
                let other = &buckets[j];
                if cmp(top, other.mono(other.len() - 1, w)).is_gt() {
                    best = Some(i);
                }
            }
            None => best = Some(i),
        }
    }
    best
}

// checker-stop-latency.3 design -- the geobucket builds a summand and merges a block of at most 4096 terms of each operand at a time, each block counted before it, the merge's loop over the block's terms cut as slices of their own, not a merge counted at once or a count per term (log 0059-1)
/// Terms of a geobucket's work, or order tests of `normalize`, between two
/// questions to the poll.
const STRIDE: usize = 4096;

/// a + b into `out`, all three increasing. A term passed is a unit of work
/// for `poll`, asked before each block of at most `STRIDE` terms of a and
/// as many of b, which counts them all; where it stops, `out` means
/// nothing.
fn merge(
    field: Field,
    w: usize,
    order: &Cmp,
    a: &Poly<u64>,
    b: &Poly<u64>,
    out: &mut Poly<u64>,
    poll: &mut Poll,
) {
    by_order!(order, cmp => merge_by(field, w, cmp, a, b, out, poll));
}

// checker-stop-latency.3 design -- the geobucket builds a summand and merges a block of at most 4096 terms of each operand at a time, each block counted before it, the merge's loop over the block's terms cut as slices of their own, not a merge counted at once or a count per term (log 0059-1)
/// `merge`, the monomials compared by `cmp`.
#[inline(always)]
fn merge_by(
    field: Field,
    w: usize,
    cmp: impl Fn(&[u32], &[u32]) -> Ordering,
    a: &Poly<u64>,
    b: &Poly<u64>,
    out: &mut Poly<u64>,
    poll: &mut Poll,
) {
    // checker-stop-latency.11 design -- an output whose monomials lack room has its two vectors allocated afresh, not grown by `reserve`, which copies the contents they held before they were cleared (log 0059-1)
    out.clear();
    if out.monos.capacity() < a.monos.len() + b.monos.len() {
        out.monos = Vec::with_capacity(a.monos.len() + b.monos.len());
        out.coefs = Vec::with_capacity(a.len() + b.len());
    }
    let (mut a_at, mut b_at) = (0, 0);
    while a_at < a.len() && b_at < b.len() {
        let (a_end, b_end) = (a.len().min(a_at + STRIDE), b.len().min(b_at + STRIDE));
        if poll.work(a_end - a_at + b_end - b_at) {
            return;
        }
        // The block's terms as slices of their own, whose lengths bound
        // every index of the loop.
        let (a_monos, a_coefs) = (&a.monos[..a_end * w], &a.coefs[..a_end]);
        let (b_monos, b_coefs) = (&b.monos[..b_end * w], &b.coefs[..b_end]);
        while a_at < a_coefs.len() && b_at < b_coefs.len() {
            let a_mono = &a_monos[a_at * w..(a_at + 1) * w];
            let b_mono = &b_monos[b_at * w..(b_at + 1) * w];
            match cmp(a_mono, b_mono) {
                Ordering::Less => {
                    out.monos.extend_from_slice(a_mono);
                    out.coefs.push(a_coefs[a_at]);
                    a_at += 1;
                }
                Ordering::Greater => {
                    out.monos.extend_from_slice(b_mono);
                    out.coefs.push(b_coefs[b_at]);
                    b_at += 1;
                }
                Ordering::Equal => {
                    let sum = field.add(a_coefs[a_at], b_coefs[b_at]);
                    if sum != 0 {
                        out.monos.extend_from_slice(a_mono);
                        out.coefs.push(sum);
                    }
                    a_at += 1;
                    b_at += 1;
                }
            }
        }
    }
    // The rest of one of the two, a block at a time.
    for (remaining, mut at) in [(a, a_at), (b, b_at)] {
        while at < remaining.len() {
            let end = remaining.len().min(at + STRIDE);
            if poll.work(end - at) {
                return;
            }
            out.monos
                .extend_from_slice(&remaining.monos[at * w..end * w]);
            out.coefs.extend_from_slice(&remaining.coefs[at..end]);
            at = end;
        }
    }
}
