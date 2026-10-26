//! The shape of a record's replays: what a full replay of a record built,
//! its table and its matrices, kept for that record's replays at further
//! primes. A replay from the shape reads its generators into the shape's
//! table and reduces the shape's matrices with its own coefficients, and
//! builds none. It checks the terms of every nonzero generator and of
//! every row a step makes against the shape, not those of the
//! inter-reduction's rows.

use std::any::Any;

use crate::entry::Width;
use crate::id::MonoId;
use crate::matrix::Matrix;
use crate::monomial::Format;
use crate::table::Table;

// q-replay-time.3 cited: parisse-2013 §4, steps 5 and 6 -- the lists of all possible monomials are recorded for further primes during the first prime's computation, and a row holds monomial indices relative to that ordered list
/// What a full replay of a record built, kept for that record's replays
/// at further primes: the replay's table, whose ids the matrices
/// name, and the matrices.
pub(crate) struct Shape<M: Format> {
    pub(crate) table: Table<M>,
    pub(crate) built: Built,
}

/// The matrices of one replay and the terms of the elements they were
/// built from.
#[derive(Default)]
pub(crate) struct Built {
    /// Per nonzero generator, in the instance's order, its monomials.
    pub(crate) inputs: Vec<Vec<MonoId>>,
    /// Per step of the record, in order, to the step where the replay
    /// ended.
    pub(crate) steps: Vec<Taken>,
    // q-replay-time.11 cited: berthomieu-2022a §3.4, Tracer over Q -- the computations that apply the learned data need no symbolic preprocessing
    /// The inter-reduction's matrix. None where the replay ended at a
    /// constant element.
    pub(crate) inter: Option<Matrix>,
}

/// One step's matrix as the replay built it, and what its reduction made.
pub(crate) struct Taken {
    pub(crate) matrix: Matrix,
    /// The ids of the monomials the counted rows must reduce to, in row
    /// order.
    pub(crate) leads: Vec<MonoId>,
    /// Per row the reduction made, in the order `matrix::replay` returns
    /// them, its columns.
    pub(crate) made: Vec<Vec<MonoId>>,
}

/// What a replay does with the shape of its matrices.
pub(crate) enum Shaping<'a> {
    /// Builds every matrix from the record, and keeps none.
    Off,
    /// Builds every matrix from the record, and keeps it in the `Built`.
    Take(&'a mut Built),
    /// Takes every matrix from the `Built`, and builds none.
    Use(&'a Built),
}

/// The lift's slot for the shape of its record's replays, and its counts.
#[derive(Default)]
pub(crate) struct Kept {
    // q-replay-time.9 design -- the kept shape is a Box<dyn Any> with its width, which a run takes by a downcast to its format, not an enum with a variant per width (not measured)
    /// The shape, the width of its format, and the `Shape<M>` boxed.
    shape: Option<(Width, Box<dyn Any>)>,
    // q-replay-time.12 design -- Kept counts the replays that ran from a shape and the shapes taken, not the test code timing the call (not measured)
    /// The replays from a kept shape whose solve ran to an end: an image, a
    /// misfit, a stop, or another error. A replay that ends at its read or
    /// at its echelon is not counted.
    pub(crate) used: u64,
    /// The shapes put: the full replays that kept what they built.
    pub(crate) taken: u64,
}

impl Kept {
    /// Drops the shape; the counts stay.
    pub(crate) fn clear(&mut self) {
        self.shape = None;
    }

    /// The width of the shape's format, None where none is kept.
    pub(crate) fn width(&self) -> Option<Width> {
        self.shape.as_ref().map(|&(width, _)| width)
    }

    /// The shape, where one is kept in the format `M`.
    pub(crate) fn get<M: Format>(&mut self) -> Option<&mut Shape<M>> {
        self.shape.as_mut()?.1.downcast_mut::<Shape<M>>()
    }

    /// Keeps `shape`, of the format of `width`, in place of any shape kept,
    /// and counts it. The shape's table gives back its slack.
    pub(crate) fn put<M: Format>(&mut self, width: Width, mut shape: Shape<M>) {
        // q-replay-time.14 design -- a kept matrix gives back the slack of its arrays when it is kept, and the shape's table when the shape is put, not the shape kept as built (log 0022-2)
        shape.table.shrink_to_fit();
        self.shape = Some((width, Box::new(shape)));
        self.taken += 1;
    }
}
