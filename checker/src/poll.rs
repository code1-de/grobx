//! The caller's poll, which the check calls as its work goes and which can
//! stop it (`check_polled`).

// checker-stop-latency.1 design -- the check counts its work in units, a unit about the work of one term, and calls its poll once per 4096 units beside its poll per element, generator, row, pair, and point, not per 4096 reduction steps whatever a step's work (log 0059-1)
/// Units of work between two calls of the poll. A unit is about the work
/// of one term: a term written into the geobucket, a product the heap
/// yields, a term read, a monomial evaluated at a point.
const STEPS: u64 = 4096;

/// The poll of one check. Once it has answered true, it is not called
/// again, and every later question answers true.
pub struct Poll<'p> {
    callback: &'p mut dyn FnMut() -> bool,
    /// Units of work since the last call.
    steps: u64,
    stopped: bool,
}

impl<'p> Poll<'p> {
    pub fn new(callback: &'p mut dyn FnMut() -> bool) -> Poll<'p> {
        Poll {
            callback,
            steps: 0,
            stopped: false,
        }
    }

    /// A step of the check: calls the poll. Whether the check stops.
    pub fn step(&mut self) -> bool {
        if !self.stopped {
            self.steps = 0;
            self.stopped = (self.callback)();
        }
        self.stopped
    }

    // checker-stop-latency.2 design -- n units are counted before they are done and call the poll once for every 4096 units they pass, not once however many (not measured)
    /// `n` units of work: calls the poll once for every `STEPS` units
    /// counted since its last call, these n among them, until it answers
    /// true. The n units are counted before they are done. Whether the
    /// check stops.
    #[inline]
    pub fn work(&mut self, n: usize) -> bool {
        self.steps += n as u64;
        if self.steps >= STEPS {
            return self.calls();
        }
        self.stopped
    }

    #[cold]
    fn calls(&mut self) -> bool {
        while !self.stopped && self.steps >= STEPS {
            self.steps -= STEPS;
            self.stopped = (self.callback)();
        }
        self.stopped
    }

    /// Whether the poll has answered true.
    #[inline]
    pub fn stopped(&self) -> bool {
        self.stopped
    }
}
