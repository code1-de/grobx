//! Q's numbers as the parser reads them, and a polynomial over Q as a
//! rational times its primitive part.

use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::field::DIGITS;
use crate::parse::{Numbers, Ring};
use crate::poll::Poll;
use crate::poly::Poly;

/// Products of two limbs that count as one unit of work (`Poll`).
const LIMBS: usize = 16;

/// The limbs of |n|, 64 bits each.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a number in memory has fewer bits than usize counts"
)]
pub fn limbs(n: &BigInt) -> usize {
    (n.bits() as usize).div_ceil(64)
}

// checker-stop-latency.6 design -- a product or a quotient of a and b over Q counts 2 + la·lb/16 units, la and lb their limbs, and a gcd or an lcm 64·(1 + L) + 2·L², L the larger's limbs, not a gcd counted as a product (log 0059-1)
/// The units of work of a product or a quotient of a and b: two, and one
/// per `LIMBS` products of a limb of a by a limb of b.
pub fn cost(a: &BigInt, b: &BigInt) -> usize {
    2 + limbs(a) * limbs(b) / LIMBS
}

// checker-stop-latency.6 design -- a product or a quotient of a and b over Q counts 2 + la·lb/16 units, la and lb their limbs, and a gcd or an lcm 64·(1 + L) + 2·L², L the larger's limbs, not a gcd counted as a product (log 0059-1)
/// The units of work of a gcd or an lcm of a and b, L the limbs of the
/// larger: 64 per limb and one more, and 2 per square of a limb. The gcd
/// shifts its operands a bit at a time.
pub fn gcd_cost(a: &BigInt, b: &BigInt) -> usize {
    let l = limbs(a).max(limbs(b));
    64 * (1 + l) + 2 * l * l
}

/// The units of work of an operation on a and on an integer of `digits`
/// decimal digits, as `cost`.
fn digits_cost(a: &BigInt, digits: usize) -> usize {
    2 + limbs(a) * digits.div_ceil(DIGITS) / LIMBS
}

/// Q's numbers, read exactly.
pub struct Rationals;

/// A term's coefficient n/d while the term is read, d > 0.
pub struct Fraction {
    n: BigInt,
    d: BigInt,
}

/// A polynomial's terms over Q as read: per term its monomial and its
/// coefficient n/d in lowest terms, d > 0.
#[derive(Default)]
pub struct Terms {
    monos: Vec<u32>,
    nums: Vec<BigInt>,
    dens: Vec<BigInt>,
}

// checker-q.6 cited: monagan-2011 §2.3 -- rational arithmetic does about ten times the integer operations; fraction-free algorithms scale fractions to a common denominator
// checker-q.11 cited: brickenstein-2010 §4 -- over Q coefficients are kept in Z, and the content is extracted from an element used to reduce others
// checker-q.18 practice: content and primitive part -- a polynomial over Q is held once, as a fraction times its primitive part
/// A polynomial over Q, (num/den)·r. r is an integer polynomial whose
/// coefficients have no common factor and whose leading coefficient is
/// positive, its primitive part. num/den is in lowest terms and den > 0. The
/// zero polynomial has no terms, num = 0, and den = 1.
#[derive(Clone, Debug)]
pub struct QPoly {
    pub r: Poly<BigInt>,
    pub num: BigInt,
    pub den: BigInt,
}

impl Default for QPoly {
    fn default() -> QPoly {
        QPoly {
            r: Poly::default(),
            num: BigInt::zero(),
            den: BigInt::one(),
        }
    }
}

impl QPoly {
    /// Whether the polynomial is its own primitive part: integer
    /// coefficients with no common factor and a positive leading one.
    pub fn is_primitive(&self) -> bool {
        !self.r.is_zero() && self.num.is_one() && self.den.is_one()
    }
}

/// The integer whose decimal digits are `digits`.
#[allow(
    clippy::expect_used,
    reason = "the reader calls it only after at_digit(), on nonempty ASCII digits"
)]
pub fn integer(digits: &[u8]) -> BigInt {
    BigInt::parse_bytes(digits, 10).expect("decimal digits are an integer")
}

impl Numbers for Rationals {
    type Term = Fraction;
    type Terms = Terms;
    type Poly = QPoly;

    fn one(&self) -> Fraction {
        Fraction {
            n: BigInt::one(),
            d: BigInt::one(),
        }
    }

    fn times(&self, c: &mut Fraction, digits: &[u8], poll: &mut Poll) {
        if poll.work(digits_cost(&c.n, digits.len())) {
            return;
        }
        c.n *= integer(digits);
    }

    fn over(&self, c: &mut Fraction, digits: &[u8], poll: &mut Poll) -> Option<()> {
        let d = integer(digits);
        if d.is_zero() {
            return None;
        }
        if poll.work(cost(&c.d, &d)) {
            return Some(());
        }
        c.d *= d;
        Some(())
    }

    // checker-q.15 design -- a term's coefficient is read as a fraction, and the polynomial is brought over one denominator when it ends (not measured)
    fn push(&self, terms: &mut Terms, m: &[u32], c: Fraction, negative: bool, poll: &mut Poll) {
        let Fraction { mut n, mut d } = c;
        if n.is_zero() {
            return;
        }
        if !d.is_one() && !poll.work(gcd_cost(&n, &d)) {
            let g = n.gcd(&d);
            if !g.is_one() {
                n /= &g;
                d /= &g;
            }
        }
        if negative {
            n = -n;
        }
        terms.monos.extend_from_slice(m);
        terms.nums.push(n);
        terms.dens.push(d);
    }

    fn finish(&self, terms: &mut Terms, ring: &Ring, out: &mut QPoly, poll: &mut Poll) {
        // D, the lcm of the denominators; every numerator put over it.
        let mut big_d = BigInt::one();
        for d in &terms.dens {
            if !d.is_one() {
                if poll.work(cost(&big_d, d)) {
                    return;
                }
                if !big_d.is_multiple_of(d) {
                    if poll.work(gcd_cost(&big_d, d)) {
                        return;
                    }
                    big_d = big_d.lcm(d);
                }
            }
        }
        let mut p = Poly {
            monos: std::mem::take(&mut terms.monos),
            coefs: std::mem::take(&mut terms.nums),
        };
        if !big_d.is_one() {
            for (n, d) in p.coefs.iter_mut().zip(&terms.dens) {
                if *d != big_d {
                    if poll.work(cost(&big_d, d) + cost(n, &big_d)) {
                        return;
                    }
                    *n *= &big_d / d;
                }
            }
        }
        terms.dens.clear();
        p.normalize(
            ring.w,
            &ring.order,
            |a, b| *a += b,
            num_traits::Zero::is_zero,
            poll,
        );
        if poll.stopped() {
            return;
        }
        *out = primitive(p, big_d, poll);
    }
}

/// p/d as its content times its primitive part, d > 0. The content's gcd
/// with each coefficient is work for `poll` as `gcd_cost` counts it, and
/// each coefficient's quotient by the content as `cost` counts it; where
/// it stops, the result means nothing.
pub fn primitive(mut p: Poly<BigInt>, d: BigInt, poll: &mut Poll) -> QPoly {
    debug_assert!(d > BigInt::ZERO, "d > 0");
    let Some(lc) = p.coefs.first() else {
        return QPoly::default();
    };
    let negative = match lc.sign() {
        Sign::Minus => true,
        Sign::NoSign | Sign::Plus => false,
    };
    let mut content = BigInt::zero();
    for c in &p.coefs {
        if poll.work(gcd_cost(&content, c)) {
            return QPoly::default();
        }
        content = content.gcd(c);
        if content.is_one() {
            break;
        }
    }
    if negative {
        content = -content;
    }
    if !content.is_one() {
        for c in &mut p.coefs {
            if poll.work(cost(c, &content)) {
                return QPoly::default();
            }
            *c /= &content;
        }
    }
    let g = content.gcd(&d);
    let (num, den) = if g.is_one() {
        (content, d)
    } else {
        (content / &g, d / g)
    };
    QPoly { r: p, num, den }
}
