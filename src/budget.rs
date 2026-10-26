//! The budget of a call: its deadline, its cancel flag, and its ceiling on
//! the resident set (`OBJECTIVES.md`, _Call parameters_), the counts of its
//! row operations, and the call report where the call asks for it.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::time::Instant;

use crate::report::{Report, Stage};
use crate::{Cancel, Error, Params, Stop};

/// Ticks from one full check to the next.
const TICKS: u32 = 64;

// matrix-stop-latency.1 design -- the budget counts units of work beside its ticks, and a full check comes once 64 ticks or WORK units have passed since the last, whichever first, not more ticks per check, fewer rows per check, or a clock read per row (log 0056-1)
// matrix-stop-latency.2 design -- a unit is one entry of a row operation, and WORK is 2^23 units, not units of time, nor 2^20 or 2^26 units (log 0056-1)
/// Units of work from one full check to the next. A unit is one entry of
/// a row operation, or a byte of a text (`text::write_basis`). Other work
/// spends by weights of its own: a product of symbolic preprocessing
/// `matrix::PRODUCT` units, an element of a keyed sort `matrix::KEY` and
/// `matrix::SORTED`, a term of a cofactor's composition or lift
/// `cofactors::TERM`, a cofactor entry `cofactors::ENTRY`, a term the
/// lift's merge writes `lift::MERGED` and more, a word the lift's append
/// writes `lift::LIMB` (`lift::append`), a multiply-add of an attempt's
/// conversion over Q one unit, and a cofactor term scaled over Q
/// `lift::SCALE`·(m² + 4).
pub(crate) const WORK: u64 = 1 << 23;

/// The bounds of one call, and the work done since their last full check:
/// ticks, a row or another step of the run each, and units of work
/// (`WORK`), which the matrices spend as they are built, sorted, and
/// reduced, the texts as they are written, the cofactors as they are
/// composed, lifted, and sorted into a replay's image, and over ℚ the
/// lift's images as they are merged and appended and converted at the
/// attempts. A full check comes once either reaches its bound,
/// and every full check resets both. The budget also counts the entries the
/// row operations subtract and those they add into combinations, and holds
/// the call report where the call asks for it.
pub(crate) struct Budget {
    /// None where the call has none, or where it lies past what an `Instant`
    /// holds.
    deadline: Option<Instant>,
    cancel: Option<Cancel>,
    /// The ceiling in bytes, and where the resident set is read.
    memory: Option<(u64, Resident)>,
    /// Ticks left until the next full check.
    left: u32,
    /// Units of work left until the next full check.
    work: u64,
    // call-report.2 design -- the report travels inside the call's budget, which every stage that reports takes already, not a parameter of its own beside it nor a thread-local report (not measured)
    /// The call report, where the call asks for it.
    report: Option<Box<Report>>,
    // call-report.4 design -- the budget counts, every call, the entries the row operations subtract and those they add into combinations, in the spend of each row operation, not counts kept only where the report is on, a count in each reduction's result, nor the reduction's units of work (not measured)
    /// The entries the row operations subtracted since `new`.
    subtracted: u64,
    /// The entries the row operations added into combinations since `new`.
    added: u64,
    /// The units from one full check to the next, `WORK` unless a test set
    /// fewer.
    #[cfg(test)]
    per: u64,
    /// The units spent since `new`.
    #[cfg(test)]
    spent: u64,
    /// The full checks since `new`.
    #[cfg(test)]
    checks: u64,
}

impl Budget {
    /// The budget of a call that started at `start`, with the call report
    /// where `params` asks for it. A ceiling where the resident set cannot
    /// be read is `Error::Memory`.
    pub(crate) fn new(start: Instant, params: &Params) -> Result<Budget, Error> {
        let memory = match params.memory {
            None => None,
            Some(ceiling) => Some((ceiling, Resident::open().ok_or(Error::Memory)?)),
        };
        Ok(Budget {
            deadline: params.deadline.and_then(|d| start.checked_add(d)),
            cancel: params.cancel.clone(),
            memory,
            left: TICKS,
            work: WORK,
            report: params
                .report
                .then(|| Box::new(Report::new(start, params.report_sink.clone()))),
            subtracted: 0,
            added: 0,
            #[cfg(test)]
            per: WORK,
            #[cfg(test)]
            spent: 0,
            #[cfg(test)]
            checks: 0,
        })
    }

    /// A budget with no bound, which never stops: for the writers of the
    /// texts outside a call, and the test code.
    pub(crate) fn unbounded() -> Budget {
        Budget {
            deadline: None,
            cancel: None,
            memory: None,
            left: TICKS,
            work: WORK,
            report: None,
            subtracted: 0,
            added: 0,
            #[cfg(test)]
            per: WORK,
            #[cfg(test)]
            spent: 0,
            #[cfg(test)]
            checks: 0,
        }
    }

    /// The full check: the cancel flag, the deadline, and the ceiling, in
    /// that order. It first resets the ticks and the units of work left
    /// until the next. Where the resident set is above the ceiling,
    /// `release` gives back what the run can rebuild, the allocators give
    /// back what is free, and the resident set is read again.
    pub(crate) fn check(&mut self, release: impl FnOnce()) -> Result<(), Stop> {
        self.left = TICKS;
        #[cfg(not(test))]
        {
            self.work = WORK;
        }
        #[cfg(test)]
        {
            self.work = self.per;
            self.checks += 1;
        }
        if self.cancel.as_ref().is_some_and(Cancel::is_set) {
            return Err(Stop::Cancel);
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(Stop::Deadline);
        }
        if let Some((ceiling, resident)) = &mut self.memory
            && resident.above(*ceiling)
        {
            release();
            trim();
            if resident.above(*ceiling) {
                return Err(Stop::Memory);
            }
        }
        Ok(())
    }

    // f4-gfp.27 design -- the budget is ticked once per row and checked fully every 64 ticks and at the start of every round, not checked at the start of each round alone (not measured)
    /// One row, or another step, of work: a full check where it is the
    /// `TICKS`th tick since
    /// the last, with nothing of the run's to give back.
    #[inline]
    pub(crate) fn tick(&mut self) -> Result<(), Stop> {
        self.left -= 1;
        if self.left == 0 {
            return self.check(|| {});
        }
        Ok(())
    }

    // matrix-stop-latency.1 design -- the budget counts units of work beside its ticks, and a full check comes once 64 ticks or WORK units have passed since the last, whichever first, not more ticks per check, fewer rows per check, or a clock read per row (log 0056-1)
    /// `units` of work done: a full check where they reach the units left
    /// until the next, with nothing of the run's to give back.
    #[inline]
    pub(crate) fn spend(&mut self, units: usize) -> Result<(), Stop> {
        let units = units as u64;
        #[cfg(test)]
        {
            self.spent += units;
        }
        if units < self.work {
            self.work -= units;
            return Ok(());
        }
        self.check(|| {})
    }

    // call-report.4 design -- the budget counts, every call, the entries the row operations subtract and those they add into combinations, in the spend of each row operation, not counts kept only where the report is on, a count in each reduction's result, nor the reduction's units of work (not measured)
    /// `units` entries a row operation subtracted, counted and spent.
    #[inline]
    pub(crate) fn spend_subtracted(&mut self, units: usize) -> Result<(), Stop> {
        self.subtracted += units as u64;
        self.spend(units)
    }

    /// `units` entries a row operation added into a combination, counted
    /// and spent.
    #[inline]
    pub(crate) fn spend_added(&mut self, units: usize) -> Result<(), Stop> {
        self.added += units as u64;
        self.spend(units)
    }

    /// The entries the row operations subtracted, and those they added
    /// into combinations, since `new`.
    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.subtracted, self.added)
    }

    // call-report.3 design -- each hook tests at run time whether the report is on, not a type parameter whose off implementation compiles to nothing, as the trace and the cofactor tracking (f4-gfp.25) (not measured)
    /// The call report, None where the call did not ask for it.
    #[inline]
    pub(crate) fn report(&mut self) -> Option<&mut Report> {
        self.report.as_deref_mut()
    }

    /// Opens a step of the report: its stages zero from now, and the counts
    /// noted. Nothing where the report is off.
    #[inline]
    pub(crate) fn open(&mut self) {
        let counts = self.counts();
        if let Some(report) = &mut self.report {
            report.open(counts);
        }
    }

    /// Ends `stage` of the report's open step (`Report::mark`). Nothing
    /// where the report is off.
    #[inline]
    pub(crate) fn mark(&mut self, stage: Stage) {
        if let Some(report) = &mut self.report {
            report.mark(stage);
        }
    }

    /// Notes the round `round` builds, which `Stats` has counted
    /// (`Report::counted`). Nothing where the report is off, and `round`
    /// is not called.
    #[inline]
    pub(crate) fn counted(&mut self, round: impl FnOnce() -> crate::report::Round) {
        let counts = self.counts();
        if let Some(report) = &mut self.report {
            report.counted(round(), counts);
        }
    }

    /// Writes the counted round's line (`Report::end_round`). Nothing where
    /// the report is off.
    #[inline]
    pub(crate) fn end_round(&mut self, left: Option<usize>) {
        if let Some(report) = &mut self.report {
            report.end_round(left);
        }
    }

    /// The report, taken out of the budget.
    pub(crate) fn take_report(&mut self) -> Option<Report> {
        self.report.take().map(|r| *r)
    }

    /// The units from one full check to the next from now on, and the
    /// units left until the next.
    #[cfg(test)]
    pub(crate) fn set_work(&mut self, units: u64) {
        self.per = units;
        self.work = units;
    }

    /// The units spent since `new`.
    #[cfg(test)]
    pub(crate) fn spent(&self) -> u64 {
        self.spent
    }

    /// The full checks since `new`.
    #[cfg(test)]
    pub(crate) fn checks(&self) -> u64 {
        self.checks
    }
}

// f4-gfp.28 design -- the resident set is read from /proc/self/statm, its second field times the page size, not getrusage's ru_maxrss, a high-water mark (not measured)
/// The process's resident set, read from `/proc/self/statm`.
struct Resident {
    file: File,
    /// The page size in bytes.
    page: u64,
}

impl Resident {
    /// None where the resident set cannot be read.
    fn open() -> Option<Resident> {
        let file = File::open("/proc/self/statm").ok()?;
        let resident = Resident {
            file,
            page: page_size()?,
        };
        resident.bytes()?;
        Some(resident)
    }

    /// The resident set in bytes.
    fn bytes(&self) -> Option<u64> {
        // The fields are seven counts of pages; 128 bytes hold them.
        let mut buf = [0u8; 128];
        let n = self.file.read_at(&mut buf, 0).ok()?;
        let text = std::str::from_utf8(&buf[..n]).ok()?;
        let pages: u64 = text.split_ascii_whitespace().nth(1)?.parse().ok()?;
        pages.checked_mul(self.page)
    }

    // f4-gfp.50 design -- a resident set that cannot be read after the start counts as above the ceiling, not as below it (not measured)
    /// Whether the resident set is above `ceiling`. A resident set that can
    /// no longer be read is.
    fn above(&self, ceiling: u64) -> bool {
        self.bytes().is_none_or(|b| b > ceiling)
    }
}

/// The system's page size in bytes; None where sysconf gives none.
pub(crate) fn page_size() -> Option<u64> {
    // SAFETY: sysconf reads a constant of the system and touches no memory
    // of ours.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).ok().filter(|&p| p > 0)
}

// f4-gfp.29 design -- before a memory stop the run gives back what it can rebuild and glibc's malloc_trim(0) gives free memory back to the system, not the buffers freed alone (not measured)
// small-block-time.6 design -- before a memory stop mimalloc gives its free memory back as well, by mi_collect(true), not glibc's malloc_trim(0) alone (log 0053-1)
/// Gives the allocators' free memory back to the system: glibc's, and
/// mimalloc's (`blocks::collect`).
pub(crate) fn trim() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only returns free memory of glibc's heap to the
    // system; no live allocation moves.
    unsafe {
        libc::malloc_trim(0);
    }
    crate::blocks::collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read at offset 0 reads the resident set anew: 64 MiB written raise
    /// it by about that much.
    #[test]
    fn the_resident_set_follows_allocations() {
        let resident = Resident::open().expect("/proc/self/statm is readable here");
        let before = resident.bytes().unwrap();
        let block = std::hint::black_box(vec![1u8; 64 << 20]);
        let after = resident.bytes().unwrap();
        assert!(after >= before + (60 << 20), "{before} B, then {after} B");
        drop(block);
    }
}
