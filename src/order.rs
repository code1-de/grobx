//! The order contract: a family of monomial orders over a monomial format.
//! Its members are grevlex, lex, and block orders, and a call names one of
//! them by `Params::order`.

use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::str::FromStr;

use crate::Error;
use crate::monomial::Format;

// order-family.6 design -- Order asks Clone where it asked Copy, and Block holds its ends in a boxed slice, not a fixed array of ends that keeps Copy (not measured)
/// A monomial order. Variables stand in the instance's order.
pub trait Order<M: Format>: Clone + Debug + Send + Sync + 'static {
    /// Whether a monomial of larger total degree is the larger, in every
    /// case: grevlex.
    const GRADED: bool;
    /// Compares a and b, each with its total degree.
    fn cmp(&self, format: &M, a: &[M::Word], da: u32, b: &[M::Word], db: u32) -> Ordering;
    // matrix-sort-time.7 design -- lex and the block orders keep the comparison sort, with no key of their own (not measured)
    /// Whether `key` orders the monomials: of two monomials, the one with
    /// the smaller key is the smaller.
    const KEYED: bool = false;
    /// The key of the monomial a, of degree da, its low 32 bits 0, where
    /// KEYED. 0 where not.
    fn key(&self, _: &M, _: &[M::Word], _: u32) -> u128 {
        0
    }
}

/// The graded reverse lexicographic order: a > b where deg a > deg b, or
/// the degrees are equal and the last variable whose exponents differ has
/// the smaller exponent in a. The first variable is the largest.
#[derive(Clone, Copy, Debug, Default)]
pub struct Grevlex;

impl<M: Format> Order<M> for Grevlex {
    const GRADED: bool = true;

    fn cmp(&self, format: &M, a: &[M::Word], da: u32, b: &[M::Word], db: u32) -> Ordering {
        da.cmp(&db).then_with(|| format.cmp_revlex(a, b))
    }

    const KEYED: bool = true;

    fn key(&self, format: &M, a: &[M::Word], da: u32) -> u128 {
        format.grevlex_key(a, da)
    }
}

/// The lexicographic order: a > b where the first variable whose exponents
/// differ has the larger exponent in a. The first variable is the largest.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lex;

impl<M: Format> Order<M> for Lex {
    const GRADED: bool = false;

    fn cmp(&self, format: &M, a: &[M::Word], _: u32, b: &[M::Word], _: u32) -> Ordering {
        format.cmp_lex(a, b)
    }
}

// order-family.3 design -- inside a block the order is grevlex, not lex or an order named per block (not measured)
/// A block order: blocks of consecutive variables in the instance's order,
/// compared in turn. In the first block where a and b differ, a > b where a
/// has the larger degree in the block, or the degrees are equal and the
/// block's last variable whose exponents differ has the smaller exponent
/// in a. `block:k` is an elimination order for the first k variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The end of every block but the last, increasing. The last block runs
    /// from the last end to the last variable.
    ends: Box<[usize]>,
}

impl Block {
    /// The block order of these sizes on n variables, the remaining
    /// variables the last block: None unless there is a size, every size is
    /// at least 1, and they sum below n.
    #[must_use]
    pub fn new(sizes: &[usize], n: usize) -> Option<Block> {
        let mut ends = Vec::with_capacity(sizes.len());
        let mut end = 0usize;
        for &k in sizes {
            if k == 0 {
                return None;
            }
            end = end.checked_add(k)?;
            ends.push(end);
        }
        (!sizes.is_empty() && end < n).then(|| Block { ends: ends.into() })
    }

    /// The blocks on n variables, each a range of variables.
    pub fn blocks(&self, n: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts.zip(self.ends.iter().copied().chain([n]))
    }
}

impl<M: Format> Order<M> for Block {
    const GRADED: bool = false;

    // order-family.5 design -- a block comparison sums each block's exponents as it compares, not a degree per block stored beside each monomial in the table (not measured)
    // monomial-time.7 design -- a block comparison reads the exponents lane by lane through `Format::exponent`, not each block's degree from packed sums of its lanes (not measured)
    fn cmp(&self, format: &M, a: &[M::Word], _: u32, b: &[M::Word], _: u32) -> Ordering {
        for (s, e) in self.blocks(format.variables()) {
            let (mut da, mut db) = (0u32, 0u32);
            for i in s..e {
                da += format.exponent(a, i);
                db += format.exponent(b, i);
            }
            if da != db {
                return da.cmp(&db);
            }
            for i in (s..e).rev() {
                let (x, y) = (format.exponent(a, i), format.exponent(b, i));
                if x != y {
                    return y.cmp(&x);
                }
            }
        }
        Ordering::Equal
    }
}

// order-family.1 cited: bachmann-1998 §2.1 -- matrix-based descriptions of monomial orderings are very general, but their realization in an actual implementation is usually rather time-consuming
/// An order of the family as a call takes it (`Params::order`), before the
/// instance's variables are known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MonomialOrder {
    /// The graded reverse lexicographic order.
    Grevlex,
    /// The lexicographic order. Where the ideal is zero-dimensional and its
    /// degree at most `fglm::MAX_STAIRCASE`, the call computes the grevlex
    /// basis and changes it to lex by FGLM, and otherwise it runs F4 in lex.
    Lex,
    /// Blocks of k1, …, km consecutive variables in the instance's order,
    /// the remaining variables the last block, each block ordered by grevlex
    /// and the blocks compared in turn. Each size is at least 1, and the
    /// sizes sum below the instance's variables, or the call is
    /// `Error::Order`. `Block(vec![k])` is an elimination order for the
    /// first k variables, Groebner.jl's `block:k` and msolve's `-e k`.
    Block(Vec<usize>),
}

impl FromStr for MonomialOrder {
    type Err = Error;

    // order-family.2 design -- a block order is named block:k1,…,km, the sizes of the first m blocks, not block:k alone, two blocks, as Groebner.jl names it (not measured)
    /// The order `name` names: `grevlex`, `lex`, or `block:k1,…,km`, m ≥ 1
    /// sizes in decimal digits separated by commas, each at least 1.
    /// `Error::Order` where it names none.
    fn from_str(name: &str) -> Result<MonomialOrder, Error> {
        let refused = || Error::Order {
            name: name.to_string(),
        };
        match name {
            "grevlex" => Ok(MonomialOrder::Grevlex),
            "lex" => Ok(MonomialOrder::Lex),
            _ => {
                let sizes = name.strip_prefix("block:").ok_or_else(refused)?;
                let size = |k: &str| {
                    // `usize`'s parse takes a sign, which a size has not.
                    let digits = !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit());
                    digits.then(|| k.parse::<usize>().ok()).flatten()
                };
                let sizes: Option<Vec<usize>> = sizes
                    .split(',')
                    .map(|k| size(k).filter(|&k| k >= 1))
                    .collect();
                sizes.map(MonomialOrder::Block).ok_or_else(refused)
            }
        }
    }
}

/// The order's name, which `from_str` reads back.
impl fmt::Display for MonomialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MonomialOrder::Grevlex => write!(f, "grevlex"),
            MonomialOrder::Lex => write!(f, "lex"),
            MonomialOrder::Block(sizes) => {
                let sizes: Vec<String> = sizes.iter().map(usize::to_string).collect();
                write!(f, "block:{sizes}", sizes = sizes.join(","))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monomial::Dense16;

    /// a against b in the order, exponent vectors of one length.
    fn cmp<O: Order<Dense16>>(order: &O, a: &[u32], b: &[u32]) -> Ordering {
        let format = Dense16::new(a.len());
        let (mut x, mut y) = (vec![0u64; format.words()], vec![0u64; format.words()]);
        let dx = format.encode(a, &mut x).unwrap();
        let dy = format.encode(b, &mut y).unwrap();
        order.cmp(&format, &x, dx, &y, dy)
    }

    #[test]
    fn lex_reads_the_first_variable_first() {
        assert_eq!(cmp(&Lex, &[1, 0, 0], &[0, 9, 9]), Ordering::Greater);
        assert_eq!(cmp(&Lex, &[1, 2, 0], &[1, 1, 7]), Ordering::Greater);
        assert_eq!(cmp(&Lex, &[1, 1, 7], &[1, 1, 7]), Ordering::Equal);
        assert_eq!(cmp(&Lex, &[0, 0, 1], &[0, 1, 0]), Ordering::Less);
    }

    #[test]
    fn a_block_order_compares_its_blocks_in_turn() {
        let b = Block::new(&[1], 3).unwrap();
        // the first block decides by x's degree
        assert_eq!(cmp(&b, &[1, 0, 0], &[0, 5, 5]), Ordering::Greater);
        // then grevlex on y, z
        assert_eq!(cmp(&b, &[1, 2, 0], &[1, 0, 1]), Ordering::Greater);
        assert_eq!(cmp(&b, &[1, 1, 0], &[1, 0, 1]), Ordering::Greater);
        assert_eq!(cmp(&b, &[1, 1, 1], &[1, 1, 1]), Ordering::Equal);
        let b = Block::new(&[2], 4).unwrap();
        // grevlex inside the first block: x·y against x², degree 2 each
        assert_eq!(cmp(&b, &[1, 1, 0, 0], &[2, 0, 0, 0]), Ordering::Less);
        assert_eq!(cmp(&b, &[0, 1, 0, 0], &[0, 0, 3, 3]), Ordering::Greater);
    }

    #[test]
    fn ones_make_lex() {
        let b = Block::new(&[1, 1, 1], 4).unwrap();
        let vectors = [
            [0, 0, 0, 1],
            [0, 0, 2, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 3],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [2, 0, 0, 0],
        ];
        for a in &vectors {
            for c in &vectors {
                assert_eq!(cmp(&b, a, c), cmp(&Lex, a, c), "{a:?} {c:?}");
            }
        }
    }

    #[test]
    fn sizes_fit_or_not() {
        assert!(Block::new(&[1], 2).is_some());
        assert!(Block::new(&[2], 2).is_none());
        assert!(Block::new(&[0], 3).is_none());
        assert!(Block::new(&[], 3).is_none());
        assert!(Block::new(&[usize::MAX, 2], 3).is_none());
        let b = Block::new(&[2, 1], 5).unwrap();
        assert_eq!(b.blocks(5).collect::<Vec<_>>(), [(0, 2), (2, 3), (3, 5)]);
    }

    #[test]
    fn names_parse() {
        let parse = |name: &str| name.parse::<MonomialOrder>().ok();
        assert_eq!(parse("grevlex"), Some(MonomialOrder::Grevlex));
        assert_eq!(parse("lex"), Some(MonomialOrder::Lex));
        assert_eq!(parse("block:3"), Some(MonomialOrder::Block(vec![3])));
        assert_eq!(parse("block:1,20"), Some(MonomialOrder::Block(vec![1, 20])));
        for name in [
            "", "Lex", "block:", "block:+1", "block:1,", "block:0", "block: 1",
        ] {
            assert!(
                matches!(name.parse::<MonomialOrder>(), Err(Error::Order { name: n }) if n == name),
                "{name:?}"
            );
        }
        for name in ["grevlex", "lex", "block:3", "block:1,20"] {
            assert_eq!(parse(name).map(|o| o.to_string()).as_deref(), Some(name));
        }
    }
}
