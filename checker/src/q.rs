//! Q's side of the checks: exact reductions of integer polynomials, their
//! products merged in the heap.

use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::cheap::Point;
use crate::check::Side;
use crate::divisors::Divisors;
use crate::field::Field;
use crate::heap::Heap;
use crate::order::Cmp;
use crate::parse::Ring;
use crate::poll::Poll;
use crate::poly::{self, Overflow};
use crate::rational::{QPoly, Rationals, cost, gcd_cost, limbs};

/// Limbs of a residue's pass that count as one unit of work (`Poll`).
const LIMBS: usize = 8;

pub struct Q {
    w: usize,
    order: Cmp,
    /// F, each generator as its content times its primitive part.
    generators: Vec<QPoly>,
    /// G likewise. The primitive parts are the reducers.
    elements: Vec<QPoly>,
    // checker-stop-latency.12 design -- the values of F and G at a point count a polynomial's units before it, over GF(p) a unit a term, over Q a unit a term, one per 8 of each coefficient's limbs, and its fraction's limbs, computed once per polynomial, not a point's values counted as one step or the units summed at every point (log 0059-1)
    /// The units of work of each generator's value at a point, then of each
    /// element's: a unit a term, one per `LIMBS` of its coefficient's
    /// limbs, and its fraction's limbs.
    units: Vec<usize>,
    one: Vec<u32>,
    monomial: Vec<u32>,
    quotient: Vec<u32>,
}

impl Side for Q {
    type Numbers = Rationals;

    fn new(
        _: Rationals,
        ring: &Ring,
        generators: Vec<QPoly>,
        elements: Vec<QPoly>,
        poll: &mut Poll,
    ) -> Q {
        let mut units = Vec::with_capacity(generators.len() + elements.len());
        for f in generators.iter().chain(&elements) {
            if poll.work(f.r.len()) {
                break;
            }
            let terms: usize = f.r.coefs.iter().map(|c| 1 + limbs(c) / LIMBS).sum();
            units.push(terms + limbs(&f.num) + limbs(&f.den));
        }
        Q {
            w: ring.w,
            order: ring.order.clone(),
            generators,
            elements,
            units,
            one: vec![0; ring.w],
            monomial: Vec::new(),
            quotient: Vec::new(),
        }
    }

    fn numbers(&self) -> &Rationals {
        &Rationals
    }

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn monos(&self, i: usize) -> &[u32] {
        &self.elements[i].r.monos
    }

    fn generators(&self) -> usize {
        self.generators.len()
    }

    fn generator_monos(&self, j: usize) -> &[u32] {
        &self.generators[j].r.monos
    }

    fn values(
        &self,
        point: &Point,
        generators: &mut Vec<u64>,
        elements: &mut Vec<u64>,
        poll: &mut Poll,
    ) -> bool {
        let field = point.field;
        let mut units = self.units.iter();
        for (out, polys) in [(generators, &self.generators), (elements, &self.elements)] {
            out.clear();
            for f in polys {
                #[allow(
                    clippy::expect_used,
                    reason = "Q::new fills the units of every polynomial unless the poll stops, and then side() returns None"
                )]
                let cost = *units
                    .next()
                    .expect("a side made whole has every polynomial's units");
                // A residue takes a pass over its limbs.
                if poll.work(cost) {
                    return false;
                }
                // (num/den)·r is defined at the point where den is a unit.
                let Some(inverse) = field.inv(residue(field, &f.den)) else {
                    return false;
                };
                let r = point.value(&f.r.monos, f.r.coefs.iter().map(|c| residue(field, c)));
                out.push(field.mul(field.mul(residue(field, &f.num), inverse), r));
            }
        }
        true
    }

    fn scaled(&self, i: usize) -> bool {
        self.elements[i].is_primitive()
    }

    fn generator(
        &mut self,
        j: usize,
        index: &mut Divisors,
        poll: &mut Poll,
    ) -> Result<bool, usize> {
        let elements = &self.elements;
        let mut heap = Heap::new(self.w, &self.order);
        #[allow(clippy::expect_used, reason = "a monomial times 1 passes no u32")]
        heap.add(
            &self.generators[j].r,
            &self.one,
            0,
            BigInt::one(),
            BigInt::one(),
            None,
            poll,
        )
        .expect("a monomial times 1 does not overflow");
        let (monomial, quotient) = (&mut self.monomial, &mut self.quotient);
        reduces_to_zero(elements, self.w, &mut heap, index, monomial, quotient, poll)
    }

    // checker-q.12 cited: monagan-2007 §1 -- the heap algorithms are as fast as geobuckets but use far less memory
    fn row(
        &mut self,
        element: usize,
        entries: &[QPoly],
        poll: &mut Poll,
    ) -> Result<bool, Overflow> {
        let w = self.w;
        let mut heap = Heap::new(w, &self.order);
        // checker-q.13 cited: johnson-1974 §3, §6 -- the heap holds one element per term of the smaller polynomial, which multiplication is free to choose
        // The streams are the generator's terms, whichever factor is
        // shorter: a generator is taken as the shorter factor of a row.
        for (c, f) in entries.iter().zip(&self.generators) {
            if c.r.is_zero() || f.r.is_zero() {
                continue;
            }
            // c·f = (num/den)·R·r with the fraction in lowest terms, since
            // each of the two fractions is.
            if poll.work(
                gcd_cost(&c.num, &f.den) + gcd_cost(&f.num, &c.den) + 4 * cost(&c.num, &f.num),
            ) {
                return Ok(false);
            }
            let g1 = c.num.gcd(&f.den);
            let g2 = f.num.gcd(&c.den);
            let num = (&c.num / &g1) * (&f.num / &g2);
            let den = (&c.den / &g2) * (&f.den / &g1);
            for term in 0..f.r.len() {
                let coef = &f.r.coefs[term];
                if poll.work(gcd_cost(coef, &den) + cost(coef, &num)) {
                    return Ok(false);
                }
                let common = coef.gcd(&den);
                let (term_num, term_den) = if common.is_one() {
                    (coef * &num, den.clone())
                } else {
                    (coef / &common * &num, &den / &common)
                };
                heap.add(&c.r, f.r.mono(term, w), 0, term_num, term_den, None, poll)?;
            }
        }
        let basis_element = &self.elements[element];
        heap.add(
            &basis_element.r,
            &self.one,
            0,
            -&basis_element.num,
            basis_element.den.clone(),
            None,
            poll,
        )?;
        // A later term of an entry times a generator's term may pass `u32`.
        let next = heap.next(&mut self.monomial, poll).map_err(|_| Overflow)?;
        Ok(next.is_none())
    }

    // checker-q.10 cited: parisse-2013 §2 -- the s-poly is reduced in Z, multiplied by appropriate leading coefficients
    fn spair(
        &mut self,
        i: usize,
        j: usize,
        lcm: &[u32],
        index: &mut Divisors,
        poll: &mut Poll,
    ) -> Result<bool, usize> {
        let w = self.w;
        let elements = &self.elements;
        let (first, second) = (&elements[i].r, &elements[j].r);
        let gamma = first.coefs[0].gcd(&second.coefs[0]);
        let mut heap = Heap::new(w, &self.order);
        for (element, element_poly, multiplier) in [
            (i, first, &second.coefs[0] / &gamma),
            (j, second, -(&first.coefs[0] / &gamma)),
        ] {
            self.quotient.clear();
            poly::div_into(lcm, element_poly.mono(0, w), &mut self.quotient);
            heap.add(
                element_poly,
                &self.quotient,
                1,
                multiplier,
                BigInt::one(),
                Some(element),
                poll,
            )
            .map_err(|_| element)?;
        }
        reduces_to_zero(
            elements,
            w,
            &mut heap,
            index,
            &mut self.monomial,
            &mut self.quotient,
            poll,
        )
    }
}

/// n mod p.
fn residue(field: Field, n: &BigInt) -> u64 {
    let r = field.limbs(n.iter_u64_digits().rev());
    match n.sign() {
        Sign::Minus => field.neg(r),
        Sign::NoSign | Sign::Plus => r,
    }
}

// checker-q.4 cited: parisse-2013 §2 _Checking a reconstructed Groebner basis_ -- the deterministic check reduces the s-polys to 0 over Q; the probabilistic one reduces them modulo primes, with an error probability
// checker-q.5 cited: parisse-2013 §4 -- integer certification is significantly faster than modular certification for Cyclic examples, and almost as fast for Katsura
// checker-gfp.14 cited: buchberger-1979 §2 -- the definition of reduction, and of a Gröbner basis as one modulo which every element of the ideal reduces to zero
/// Top-reduces the heap's sum modulo G, exactly over Q: whether it reaches
/// zero, or the element of G whose multiple passed `u32`. Its work is
/// counted for `poll` as it is done; where it stops, the answer means
/// nothing.
fn reduces_to_zero<'a>(
    elements: &'a [QPoly],
    w: usize,
    heap: &mut Heap<'a>,
    index: &mut Divisors,
    monomial: &mut Vec<u32>,
    quotient: &mut Vec<u32>,
    poll: &mut Poll,
) -> Result<bool, usize> {
    loop {
        #[allow(
            clippy::expect_used,
            reason = "a stream with no element is the generator times 1, which passes no u32"
        )]
        let next = heap
            .next(monomial, poll)
            .map_err(|element| element.expect("the terms of a polynomial as read fit `u32`"))?;
        let Some(c) = next else {
            return Ok(true);
        };
        let Some(reducer) = index.query(monomial, poll) else {
            return Ok(false);
        };
        let reducer_poly = &elements[reducer].r;
        let a = &reducer_poly.coefs[0];
        quotient.clear();
        poly::div_into(monomial, reducer_poly.mono(0, w), quotient);
        // The term is C/s. The quotient −C/(s·a) cancels it with the
        // reducer's leading product, which is left out.
        let s = heap.s();
        // checker-q.9 design -- a quotient whose numerator lc(g) does not divide is brought to lowest terms, and s becomes the lcm of the denominators (log 0002-1)
        let (num, den) = if a.is_one() {
            (-c, s.clone())
        } else {
            if poll.work(cost(&c, a)) {
                return Ok(false);
            }
            let (quotient, remainder) = c.div_rem(a);
            if remainder.is_zero() {
                (-quotient, s.clone())
            } else {
                if poll.work(cost(s, a)) {
                    return Ok(false);
                }
                let sa = s * a;
                if poll.work(gcd_cost(&c, &sa) + 2 * cost(&c, &sa)) {
                    return Ok(false);
                }
                let common = c.gcd(&sa);
                (-(c / &common), sa / common)
            }
        };
        // checker-orders.12 cited: bigatti-2025 §1.1, Remark 5 -- xy >_Lex z^1000: Lex is not degree compatible
        // A tail term may have a larger degree than the leading term, so a
        // key may pass the degree of `monomial` and `u32`.
        heap.add(reducer_poly, quotient, 1, num, den, Some(reducer), poll)
            .map_err(|_| reducer)?;
    }
}
