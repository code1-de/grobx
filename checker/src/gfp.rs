//! GF(p)'s side of the checks: residues as read, monic elements, and the
//! geobucket; and the generators' text for the test code.

use crate::cheap::Point;
use crate::check::Side;
use crate::divisors::Divisors;
use crate::field::Field;
use crate::parse::{Numbers, Ring};
use crate::poll::Poll;
use crate::poly::{self, Geobucket, Overflow, Poly};

/// The units of work of an inverse modulo p (`Poll`).
pub const INVERSE: usize = 8;

impl Numbers for Field {
    type Term = u64;
    type Terms = Poly<u64>;
    type Poly = Poly<u64>;

    fn one(&self) -> u64 {
        1
    }

    fn times(&self, c: &mut u64, digits: &[u8], _: &mut Poll) {
        *c = self.mul(*c, self.decimal(digits));
    }

    fn over(&self, c: &mut u64, digits: &[u8], poll: &mut Poll) -> Option<()> {
        poll.work(INVERSE);
        let inverse = self.inv(self.decimal(digits))?;
        *c = self.mul(*c, inverse);
        Some(())
    }

    fn push(&self, terms: &mut Poly<u64>, m: &[u32], c: u64, negative: bool, _: &mut Poll) {
        let c = if negative { self.neg(c) } else { c };
        if c != 0 {
            terms.push(m, c);
        }
    }

    fn finish(&self, terms: &mut Poly<u64>, ring: &Ring, out: &mut Poly<u64>, poll: &mut Poll) {
        let add = |a: &mut u64, b| *a = self.add(*a, b);
        terms.normalize(ring.w, &ring.order, add, |c| *c == 0, poll);
        std::mem::swap(terms, out);
        terms.clear();
    }
}

/// The nonzero polynomials of `polys`, each made monic, as a bracketed
/// list in the canonical basis text's spelling: `[`, the polynomials
/// separated by `,` and a line feed, `]:` and a line feed. A term is its
/// coefficient, `*`, and its monomial, the coefficient 1 and its `*` left
/// out, and the monomial 1 left out. A monomial joins its variables by `*`,
/// each with `^e` where e > 1. Terms are joined by `+`.
pub fn text(field: Field, ring: &Ring, polys: &[Poly<u64>]) -> String {
    let names = ring.names();
    let mut out = String::from("[");
    for (k, f) in polys.iter().filter(|f| !f.coefs.is_empty()).enumerate() {
        if k > 0 {
            out.push_str(",\n");
        }
        #[allow(
            clippy::expect_used,
            reason = "p is prime, and push and normalize drop every zero coefficient"
        )]
        let inverse = field
            .inv(f.coefs[0])
            .expect("a nonzero residue mod a prime");
        for (t, &c) in f.coefs.iter().enumerate() {
            if t > 0 {
                out.push('+');
            }
            let c = field.mul(c, inverse);
            let m = &f.mono(t, ring.w)[1..];
            let constant = m.iter().all(|&e| e == 0);
            if constant || c != 1 {
                out.push_str(&c.to_string());
                if !constant {
                    out.push('*');
                }
            }
            let mut first = true;
            for (name, &e) in names.iter().zip(m).filter(|(_, e)| **e > 0) {
                if !first {
                    out.push('*');
                }
                first = false;
                out.push_str(&String::from_utf8_lossy(name));
                if e > 1 {
                    out.push('^');
                    out.push_str(&e.to_string());
                }
            }
        }
    }
    out.push_str("]:\n");
    out
}

pub struct Gfp {
    field: Field,
    w: usize,
    generators: Vec<Poly<u64>>,
    /// G, every nonzero element monic.
    elements: Vec<Poly<u64>>,
    /// The leading coefficient of each element as read, 0 for a zero element.
    leading: Vec<u64>,
    bucket: Geobucket,
    one: Vec<u32>,
    m: Vec<u32>,
    q: Vec<u32>,
}

impl Side for Gfp {
    type Numbers = Field;

    fn new(
        field: Field,
        ring: &Ring,
        generators: Vec<Poly<u64>>,
        mut elements: Vec<Poly<u64>>,
        poll: &mut Poll,
    ) -> Gfp {
        // checker-gfp.15 design -- the Gröbner checks run on monic elements, one inverse per element (not measured)
        // Each element is made monic in place and its leading coefficient kept,
        // so that G is held once.
        let mut leading = Vec::with_capacity(elements.len());
        for g in &mut elements {
            let lc = g.coefs.first().copied().unwrap_or(0);
            leading.push(lc);
            if lc > 1 && !poll.work(g.len()) {
                #[allow(clippy::expect_used, reason = "lc lies in 2..p, and p is prime")]
                let inverse = field.inv(lc).expect("a nonzero residue mod a prime");
                for c in &mut g.coefs {
                    *c = field.mul(*c, inverse);
                }
            }
        }
        let w = ring.w;
        Gfp {
            field,
            w,
            generators,
            elements,
            leading,
            bucket: Geobucket::new(field, w, ring.order.clone()),
            one: vec![0; w],
            m: Vec::new(),
            q: Vec::new(),
        }
    }

    fn numbers(&self) -> &Field {
        &self.field
    }

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn monos(&self, i: usize) -> &[u32] {
        &self.elements[i].monos
    }

    fn generators(&self) -> usize {
        self.generators.len()
    }

    fn generator_monos(&self, j: usize) -> &[u32] {
        &self.generators[j].monos
    }

    fn values(
        &self,
        point: &Point,
        generators: &mut Vec<u64>,
        elements: &mut Vec<u64>,
        poll: &mut Poll,
    ) -> bool {
        let value = |f: &Poly<u64>| point.value(&f.monos, f.coefs.iter().copied());
        generators.clear();
        for f in &self.generators {
            if poll.work(f.len()) {
                return false;
            }
            generators.push(value(f));
        }
        elements.clear();
        // An element as read is its leading coefficient times its monic form.
        for (g, &lc) in self.elements.iter().zip(&self.leading) {
            if poll.work(g.len()) {
                return false;
            }
            elements.push(point.field.mul(lc, value(g)));
        }
        true
    }

    fn scaled(&self, i: usize) -> bool {
        self.leading[i] == 1
    }

    fn generator(
        &mut self,
        j: usize,
        index: &mut Divisors,
        poll: &mut Poll,
    ) -> Result<bool, usize> {
        self.bucket.clear();
        #[allow(clippy::expect_used, reason = "a monomial times 1 passes no u32")]
        self.bucket
            .add(1, &self.one, &self.generators[j], 0, poll)
            .expect("a monomial times 1 does not overflow");
        self.reduces_to_zero(index, poll)
    }

    fn row(&mut self, i: usize, entries: &[Poly<u64>], poll: &mut Poll) -> Result<bool, Overflow> {
        let w = self.w;
        self.bucket.clear();
        for (c, f) in entries.iter().zip(&self.generators) {
            for t in 0..f.len() {
                self.bucket.add(f.coefs[t], f.mono(t, w), c, 0, poll)?;
                if poll.stopped() {
                    return Ok(false);
                }
            }
        }
        let minus = self.field.neg(self.leading[i]);
        self.bucket
            .add(minus, &self.one, &self.elements[i], 0, poll)?;
        Ok(self.bucket.lead(&mut self.m, poll).is_none())
    }

    fn spair(
        &mut self,
        i: usize,
        j: usize,
        lcm: &[u32],
        index: &mut Divisors,
        poll: &mut Poll,
    ) -> Result<bool, usize> {
        let w = self.w;
        self.bucket.clear();
        for (u, c) in [(i, 1), (j, self.field.neg(1))] {
            self.q.clear();
            poly::div_into(lcm, self.elements[u].mono(0, w), &mut self.q);
            self.bucket
                .add(c, &self.q, &self.elements[u], 1, poll)
                .map_err(|_| u)?;
        }
        self.reduces_to_zero(index, poll)
    }
}

impl Gfp {
    // checker-gfp.14 cited: buchberger-1979 §2 -- the definition of reduction, and of a Gröbner basis as one modulo which every element of the ideal reduces to zero
    /// Top-reduces the geobucket modulo G: whether it reaches zero, or the
    /// element of G whose multiple passed `u32`. Its work is counted for
    /// `poll` as it is done; where it stops, the answer means nothing.
    fn reduces_to_zero(&mut self, index: &mut Divisors, poll: &mut Poll) -> Result<bool, usize> {
        let w = self.w;
        loop {
            let Some(c) = self.bucket.lead(&mut self.m, poll) else {
                return Ok(true);
            };
            let Some(u) = index.query(&self.m, poll) else {
                return Ok(false);
            };
            self.q.clear();
            poly::div_into(&self.m, self.elements[u].mono(0, w), &mut self.q);
            // The leading term c·m is already taken out, and g_u is monic.
            // checker-orders.12 cited: bigatti-2025 §1.1, Remark 5 -- xy >_Lex z^1000: Lex is not degree compatible
            // A tail term may have a larger degree than the leading term, so
            // a product may pass deg m and `u32`.
            self.bucket
                .add(self.field.neg(c), &self.q, &self.elements[u], 1, poll)
                .map_err(|_| u)?;
        }
    }
}
