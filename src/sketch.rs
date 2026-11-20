//! The sketches of a plain round over GF(2) (gf2-forced-zeros-time.1 to .5).
//! A draw gives each column of the round's matrix a word of `BITS` bits,
//! each bit a vector that every pivot row of the round annihilates when it
//! is drawn, uniform among them. A row's sketch is the exclusive or of its
//! columns' words: 0 for a row in the span of those pivots, and for a row
//! outside it uniform. The rows the round makes after the draw join the test
//! by their sketches, kept in an echelon by their top bit. A row whose sketch
//! the echelon reduces to 0, its residue, lies in the span of the pivots at
//! the draw and of the rows made since, save with probability 2^(e − `BITS`)
//! for e sketches in the echelon; a row whose residue is not 0 lies outside
//! it. Every coefficient is 1: the caller's field packs (`Field::PACKS`).
//! The module reads its rows through the caller, and knows no matrix.

use crate::Stop;
use crate::budget::{Budget, Site};
use crate::columns::{Row, by_width};
use crate::id::Column;

// gf2-forced-zeros-time.5 tuned-checked -- a sketch holds 128 bits a column, a u128, not 64 bits with the echelon's cap at 33 (log 0134-1, 0134-2)
/// The bits of a column's word, and of a row's sketch and residue.
pub(crate) const BITS: u32 = u128::BITS;

/// A round's sketch: the columns' words of the last draw, and the echelon
/// of the residues of the rows made since.
pub(crate) struct Sketch {
    /// Per column of the matrix, its word.
    words: Vec<u128>,
    /// Per bit b, the residue in the echelon whose top bit is b, where bit
    /// b of `tops` is set; 0 elsewhere.
    echelon: [u128; BITS as usize],
    /// A bit per residue in the echelon, at its top bit.
    tops: u128,
    /// The residues in the echelon.
    len: usize,
    /// The draws.
    draws: u32,
}

impl Sketch {
    /// A sketch before its first draw: no word, and the echelon empty.
    pub(crate) fn new() -> Sketch {
        Sketch {
            words: Vec::new(),
            echelon: [0; BITS as usize],
            tops: 0,
            len: 0,
            draws: 0,
        }
    }

    /// The draws so far.
    pub(crate) fn draws(&self) -> u32 {
        self.draws
    }

    // gf2-forced-zeros-time.12 design-checked -- a draw spends the tails' entries it reads and the matrix's columns, and a row's test its entries, as units of the budget, none of it row work, not the draw's entries as row work, which the bound on the row work and its projection read (log 0134-2)
    /// Draws the words of `columns` columns afresh, from the last column to
    /// the first: `word()` at a column without a pivot, and where `pivot(c)`
    /// gives the tail of column c's pivot row, the exclusive or of the words
    /// of the tail's columns, each right of c and drawn before. The echelon
    /// empties. Each tail spends its length and the columns spend one unit
    /// each (`Budget::spend`), and a stop where the budget's check stops the
    /// call.
    pub(crate) fn draw<'m>(
        &mut self,
        columns: usize,
        pivot: impl Fn(usize) -> Option<Row<'m>>,
        mut word: impl FnMut() -> u128,
        budget: &mut Budget,
    ) -> Result<(), Stop> {
        self.words.clear();
        self.words.resize(columns, 0);
        for c in (0..columns).rev() {
            let w = match pivot(c) {
                None => word(),
                Some(tail) => {
                    budget.spend(tail.len(), Site::Row)?;
                    let words = &self.words;
                    by_width!(tail, |t| t.iter().fold(0, |a, &d| {
                        debug_assert!(d.column() > c, "a tail right of its lead");
                        a ^ words[d.column()]
                    }))
                }
            };
            self.words[c] = w;
        }
        budget.spend(columns, Site::Row)?;
        self.echelon = [0; BITS as usize];
        self.tops = 0;
        self.len = 0;
        self.draws += 1;
        Ok(())
    }

    /// The residue of the row of the columns `cols`: its sketch reduced by
    /// the echelon, from its top bit down. 0 where the row lies in the span
    /// of the pivots at the draw and the rows made since, save with
    /// probability 2^(e − `BITS`) for e residues in the echelon.
    pub(crate) fn residue(&self, cols: Row<'_>) -> u128 {
        debug_assert!(self.draws > 0, "a residue after a draw");
        let words = &self.words;
        let mut x = by_width!(cols, |c| c.iter().fold(0, |a, &k| a ^ words[k.column()]));
        while x != 0 {
            let top = BITS - 1 - x.leading_zeros();
            if self.tops >> top & 1 == 0 {
                break;
            }
            x ^= self.echelon[top as usize];
        }
        x
    }

    /// Keeps the residue `x` of a row made since the draw in the echelon at
    /// its top bit, which no residue there holds; whether the echelon then
    /// holds `cap` residues.
    pub(crate) fn keep(&mut self, x: u128, cap: usize) -> bool {
        debug_assert!(x != 0, "the residue of a made row is not 0");
        debug_assert!(
            (1..=BITS as usize).contains(&cap),
            "a cap of 1 to BITS residues"
        );
        let top = BITS - 1 - x.leading_zeros();
        debug_assert!(
            self.tops >> top & 1 == 0,
            "a residue's top bit free in the echelon"
        );
        self.echelon[top as usize] = x;
        self.tops |= 1 << top;
        self.len += 1;
        self.len >= cap
    }
}
