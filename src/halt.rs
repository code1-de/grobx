//! Why a run ends without a basis.

use crate::Stop;
use crate::monomial::Overflow;
use crate::narrow::Size;
use crate::table::Limit;

/// Why a run ends without a basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Halt {
    /// A monomial of the run, or of its cofactors, passes the format.
    Overflow,
    /// A count of the run passes what a u32 id numbers (`narrow::id`).
    Size,
    /// A bound stopped the run.
    Stop(Stop),
    /// A replay's record does not fit: at the inputs, under another order
    /// than the record's, or where the replay tracks cofactors of a record
    /// whose learn run took the echelon, step 0, or at the step of the
    /// record counted from 1.
    /// Also 0 where the lanes of a run over several primes disagree, at
    /// the inputs, in a round's reduction, or in FGLM's change: a leading
    /// coefficient the run inverts is 0 at one lane and not at the other
    /// (q-learn-time.4, .5). Also 0 where FGLM's division of an element by
    /// a replayed basis leaves a remainder (`fglm::divide`), and where a
    /// tail monomial of the basis FGLM changes lies outside its staircase
    /// (`fglm::border_forms`).
    Misfit(u32),
    /// The signature run hands the call to the plain run (handover-time.1).
    Handover,
    // gf2-signature-losses.6 design-checked -- a run past its bound on the row work halts with a variant of the crate's halt, not a variant of the public Stop nor a flag the call reads after a cancel (not measured)
    /// The run's row work passed the bound set on it (`Budget::outworked`).
    Outworked,
}

impl From<Overflow> for Halt {
    fn from(_: Overflow) -> Halt {
        Halt::Overflow
    }
}

impl From<Size> for Halt {
    fn from(_: Size) -> Halt {
        Halt::Size
    }
}

impl From<Limit> for Halt {
    fn from(limit: Limit) -> Halt {
        match limit {
            Limit::Overflow => Halt::Overflow,
            Limit::Size => Halt::Size,
            Limit::Stop(stop) => Halt::Stop(stop),
        }
    }
}

impl From<Stop> for Halt {
    fn from(stop: Stop) -> Halt {
        Halt::Stop(stop)
    }
}
