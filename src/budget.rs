//! The budget of a call: its deadline, its cancel flag, and its ceiling on
//! the resident set (`OBJECTIVES.md`, _Call parameters_), the counts of its
//! row operations, its row work and a bound on it, and the call report
//! where the call asks for it.

#[cfg(test)]
use std::cell::{Cell, RefCell};
use std::time::Instant;

use crate::os::{self, Resident};
use crate::report::{Report, Stage};
use crate::{Cancel, Error, Params, Stop};

/// Ticks from one full check to the next.
const TICKS: u32 = 64;

// matrix-stop-latency.1 design-checked -- the budget counts units of work beside its ticks, and a full check comes once 64 ticks or WORK units have passed since the last, whichever first, not more ticks per check, fewer rows per check, or a clock read per row (log 0056-1)
// matrix-stop-latency.2 design-checked -- a unit is one entry of a row operation, and WORK is 2^23 units, not units of time, nor 2^20 or 2^26 units (log 0056-1)
/// Units of work from one full check to the next. A unit is one entry of
/// a row operation, or a byte of a text (`text::write_basis`). Other work
/// spends by weights of its own: a product of symbolic preprocessing
/// `matrix::PRODUCT` units, an element of a keyed sort `sort::KEY` and
/// `sort::SORTED`, a term of a cofactor's composition or lift
/// `cofactors::TERM`, a cofactor entry `cofactors::ENTRY`, a term the
/// lift's merge writes `lift::MERGED` and more, a word the lift's append
/// writes `lift::LIMB` (`lift::append`), a multiply-add of an attempt's
/// conversion over Q one unit, a cofactor term scaled over Q
/// `lift::SCALE`·(m² + 4), and an id the table places in a growth or a
/// compaction `PROBE`.
pub(crate) const WORK: u64 = 1 << 23;

// call-stop-latency.3 design-checked -- a spend's units bound its work from above at about a nanosecond a unit on its slowest case, a random access into an array past the caches PROBE, 16, units, not the units at the work's typical cost nor WORK lowered (log 0138-1)
/// Units of work of a random access into an array past the caches: a
/// probe, a placement, or a key read at a random place.
pub(crate) const PROBE: usize = 16;

// call-stop-latency.61 design-checked -- a numeric factor's value spends PROBE units a literal, a remainder, its digits past a word's a unit each and PROBE per word, PROBE a product, and INVERSE a denominator's inversion, at each lane, not a unit per term and digit (log 0138-6)
/// Units of work of an inversion in a prime field: the extended Euclidean
/// algorithm on a prime below 2^64, at most twice its bits in steps, each a
/// division of `PROBE` units.
pub(crate) const INVERSE: usize = 2 * u64::BITS as usize * PROBE;

// call-stop-latency.2 practice-checked: cooperative cancellation -- every loop of a call whose work grows with the input spends its work on the budget inside the loop, per iteration or per chunk of at most RUN iterations where an iteration's work is a few nanoseconds, not a spend once before or after the whole loop, a tick per iteration whose work grows, nor no spend (log 0138-1)
// call-stop-latency.5 practice-checked: failpoints -- every spend, tick, and full check names its loop by a Site, and the test profile records the checks' sites, forces a check at every spend and tick, and forces a stop at a site's first check, not the sites named in comments alone nor a test per loop that builds the loop's state by hand (not measured)
/// The loop a spend, a tick, or a full check of the budget belongs to: a
/// variant per loop of a call whose work grows with the input, or per group
/// of loops of one part of the call, its doc comment naming the loops and
/// what they spend. In the test profile the budget records the site of
/// each full check (`CHECKED`, `Budget::checked_at`); outside it no code
/// reads the site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Site {
    /// The instance's header: a unit a byte of its lines, and per name
    /// `PROBE` units, its entry into the names' index, and in each growth
    /// of the index `PROBE` units a name (`text::Header::read`,
    /// `text::HashIndex`).
    Names,
    /// The parse of the instance's generators: a unit a byte it passes,
    /// spent after each term, and per term what finding its monomial among
    /// the text's distinct monomials spends, `table::MONOMIAL_PASSES` units
    /// a variable of the term and `PROBE`; and in each growth of the
    /// distinct monomials' index `PROBE` units a monomial (`text::Parser`,
    /// `table::monomial_units`, `text::HashIndex`).
    Parse,
    /// The parsed generators mapped to a run's field and table: the ids'
    /// array a unit an id; per term `PROBE` units, and its factors' values,
    /// each literal `PROBE` units, its digits past a word's a unit each and
    /// `PROBE` per word, a product `PROBE`, and a denominator's inversion
    /// `INVERSE`, at each lane; its monomial, where it is interned, as
    /// `table::touched_units` says; and the terms' two counting sorts, a
    /// unit a term a pass and a unit a key of their prefix sums
    /// (`text::Parsed::map`, `sort::counting`).
    Map,
    /// The start of a run, plain, carried, signature, or replay, and the
    /// end of its rounds: a full check each (`f4::run`, `run_carried`,
    /// `run_sig`, `replay`).
    Run,
    /// The start of a round of a run, or of a replayed step that builds a
    /// matrix: a full check each.
    Round,
    /// The lift's primes over ℚ: a full check before each (`lift::lift`).
    Prime,
    /// The inputs made monic: `PROBE` units a term scaled
    /// (`f4::monic_inputs`).
    Monic,
    /// The sugars of the elements a round makes or the inputs: `PROBE`
    /// units an element, and in an order without a grading a term
    /// (`f4::with_sugar`).
    Sugar,
    /// The pair selection: its passes over the set's places, a unit a place
    /// and a word moved, the least lcms of a split `PROBE` units a place
    /// compared, each lcm taken `PROBE` units, and the index's rebuild a
    /// unit per place and support bit and `2 · PROBE` a place's support
    /// (`pairs::select`).
    Select,
    /// The rows of the selected pairs: `matrix::PRODUCT` units a row, and
    /// their sort and the repeats' removal (`f4::pair_rows`).
    PairRows,
    /// The rows of symbolic preprocessing: a tick per row a divisor or a
    /// reducer search adds or a replay's record gives, and
    /// `matrix::PRODUCT` units per product a row forms, a given row's
    /// lead forms, or a kept row copies; a narrow column vector widened a
    /// unit an entry (`matrix::build`, `build_sig`, `build_rows`,
    /// `multiply`, `Matrix::take_kept`, `Columns::widen`).
    Products,
    /// The divisor search of symbolic preprocessing and the signature
    /// run's reducer search: a unit per block of masks a search may test,
    /// whether it finds one or not, and `PROBE` units per candidate it
    /// tests, and past `divisors::FILED_PAST` variables the divisor search
    /// a unit per file it visits besides; a reducer the signature run's
    /// build finds `matrix::FORMED` units; `G_min`'s sort by length and its
    /// leads' masks `PROBE` units an element, and past `FILED_PAST` its
    /// files as `Files::of` spends (`matrix::build`, `build_sig`,
    /// `Divisors::new`, `Divisors::find`).
    Divisors,
    /// The matrix before that a build holds: its columns' map and its rows'
    /// order a unit per column and row, and its compaction a unit per row
    /// and per entry moved (`Matrix::into_kept`, `Matrix::drop_kept`).
    Kept,
    /// The columns of a matrix: the keyed sort of its monomials, the keys
    /// and their write-back a unit each, `sort::RUN` at a time; the merge
    /// with the matrix before, `PROBE` units a round monomial looked up and
    /// `sort::KEY` an old column; the numbering `2 · PROBE` units a
    /// monomial (`sort::sort_decreasing`, `matrix::merge_columns`,
    /// `matrix::columns`); the mapping of its rows' entries to the sorted
    /// columns, a unit an entry (`matrix::map_columns`).
    Columns,
    /// The sort of a signature matrix's rows by signature
    /// (`sort::by_signature`), spent as the columns' sort.
    SigSort,
    /// A reduction's setup: the pivots its loops over the rows set or read,
    /// `PROBE` units a row, `sort::RUN` rows at a time, and the rows a
    /// replay or a signature round passes without reducing them `PROBE`
    /// units each; the sorts of the rows to reduce, of the made rows the
    /// back-substitution takes, and of the inter-reduction's rows by
    /// `sort::keys`, their keys `PROBE` units each and the rows written
    /// back a unit each; the reducer's dense row and pivot array the
    /// matrix's columns once made, and its band `field::BAND` units a
    /// column once filled (`matrix::reduce`, `reduce_sig`, `replay`,
    /// `inter_reduce`, `Reducer::back_substitute`, `Reducer::widen_band`).
    Setup,
    /// The rows of a reduction: a tick per row reduced, combination formed,
    /// or made row back-substituted, the entries its row operations subtract
    /// and add a unit each, a forward band's sweep as `BackBands` spends a
    /// back-substitution's, a zero test's group, its least leading column
    /// `PROBE` units a row and the rows it rules zero a unit each, the
    /// packed rows' entries, and the sketch's draws and residues a unit per
    /// entry and column (`matrix::Reducer`, `packed::Rows`,
    /// `sketch::Sketch`).
    Row,
    /// The scans of one row of a reduction: the columns a dense scan
    /// passes, from its start to the matrix's width, or the words of the
    /// marks the marked scan reads, a unit each, once per row; and the
    /// entries of a made row's tail the back-substitution reads for the
    /// leading column of another made row, a unit each
    /// (`Reducer::cancel`, `reduces_to_zero`, `back_substitute_row`,
    /// `needs`).
    Scan,
    /// The back-substitution in bands: a tick per band or row taken, and
    /// each band's sweep, the entries it subtracts a unit each and the
    /// columns it passes `field::BAND` units each, once per sweep
    /// (`Reducer::back_substitute`, `sweep_band`).
    BackBands,
    /// The inter-reduction's given rows: a tick each, the entries of the
    /// tail its test for a pivot reads and of a row taken as it stands a
    /// unit each, a tick per band or row its reductions take, and each
    /// band's sweep as `BackBands` spends (`matrix::inter_reduce`).
    Inter,
    /// The signature round's scan of a run for the rows its test takes:
    /// `PROBE` units a position, `sort::RUN` positions at a time
    /// (`matrix::reduce_sig`).
    SigScan,
    /// The rows a reduction made, decoded: the rows that counted, listed
    /// and sorted, the constant sought `PROBE` units a row, each row's
    /// columns mapped to their monomials a unit an entry, and the
    /// inter-reduction's elements sorted (`f4::rounds`,
    /// `f4::reduced_elements`, `matrix::Reduced::poly`).
    Decode,
    /// The entering of elements into a basis: their sort into the entering
    /// order (`f4::Plain::enter`), and `G_min`'s scans for the members that
    /// divide an element's leading monomial and those it divides, `PROBE`
    /// units a member, and the members that leave a unit each
    /// (`poly::Basis::push_min`).
    Enter,
    /// The pair update, as it goes: the B criterion's scan a unit a place
    /// and `pairs::TESTED` units a pair it may test, the new pairs' scan of
    /// `G_min` and its arrays' rebuild per member, the sort of the new
    /// pairs by degree a unit a pair, the criteria M and F per kept pair a
    /// pair may test, and the kept pairs pushed (`pairs::update`).
    Update,
    /// The signature run's entering of an element: its S-pairs with the
    /// earlier elements, `f4::PAIR` units and the format's words a pair,
    /// `sort::RUN` at a time; its signature's place in the reducer order,
    /// `sig::ratio_units` a comparison and a unit an entry its insertion
    /// moves, and the integers' respacing `PROBE` units an entry; its row of
    /// the triangle a unit a word; and its base divisors' scans of the
    /// earlier elements `PROBE` units an element (`f4::SigRun::enter`,
    /// `sig::Signatures::push`, `respace`, `Triangle::grow`,
    /// `BaseDivisors::of`).
    SigEnter,
    /// The known syzygy signatures: each search, a unit per block of the
    /// members' masks it may test and `PROBE` units per member it tests,
    /// and each new one's scan for the members
    /// it divides, a unit a member (`sig::Antichain::divides`,
    /// `drop_multiples`).
    Antichain,
    /// The signature run's searches for a pending signature a round takes:
    /// its reductant among the elements of its index, `PROBE` units an
    /// element, and the reducer of the reductant's leading monomial, a unit
    /// per block of the basis's masks and `PROBE` units per candidate it
    /// tests; and the handover tests' reducer
    /// searches of a round's elements, the same (`sig::Signatures::reductant`,
    /// `f4::SigRun::reductants`, `f4::run_sig`).
    Reducer,
    /// The pending signatures a round takes: their ids keyed, sorted, and
    /// told apart, and the distinct sorted by signature and taken out, a
    /// unit each and the sorts' own; and each taken signature's reductant's
    /// multiple and lead, `matrix::PRODUCT` units each (`queue::Queue::pop_least`,
    /// `f4::SigRun::reductants`).
    Queue,
    /// The test of a signature against the Koszul syzygy signatures of the
    /// signature run's inputs: the quotient's words, a unit per block of the
    /// inputs' masks or per input visited, and `PROBE` units per input
    /// tested (`sig::Signatures::koszul`).
    Koszul,
    /// The carry of a signature run's work to the plain run: its elements
    /// with their sugars, `PROBE` units an element of the basis and a unit a
    /// made one; the bound's scans of the older elements, `PROBE` units an
    /// element per made one; and the move into a fresh table, its map's
    /// fill a unit an id, and `PROBE` units a term and a unit a word of each
    /// monomial interned (`f4::carry`, `carry_bound`, `carry_table`).
    Carry,
    /// The learn's record: the inputs' leading monomials, a step's rows
    /// and their entries visited, its multipliers' and leads' exponents,
    /// and the sorts of the inputs and the new elements into their order of
    /// entry; the cut's rows, the reaches it follows, and each step's
    /// rewrite; a unit per row, entry, and exponent, and the sorts'
    /// own (`trace::inputs`, `step`, `pivots`, `cut`, `rewrite`).
    Trace,
    /// The echelon of the generators: their monomials gathered a unit a
    /// term and sorted, each column numbered and each term's column read
    /// `PROBE` units; each pass's loop over the columns a unit a column;
    /// the rows scanned, loaded, copied, scaled, and taken a unit an entry,
    /// the dense rows and their marks a unit an entry once made, the words
    /// of the marks a scan passes a unit each, and the entries its row
    /// operations subtract; a replay's recorded leads interned as the
    /// parse's monomials; and the rows mapped back to their monomials a
    /// unit an entry (`echelon`).
    Echelon,
    /// The substitution of a linear system's pivots into the generators:
    /// the generators' test for degree 1 and the linear ones' copies a
    /// unit a term; the pivots' rows a unit per term and variable; each
    /// generator's search for the pivots it holds `PROBE` units and a unit
    /// a pivot per term, `sort::RUN` terms at a time; its split `PROBE`
    /// units and a unit per word and held pivot a term, and its sort;
    /// Horner's levels a unit per term they walk; each product by a value
    /// a tick, per term summed `PROBE` units and two a word, a growth of
    /// the sum `PROBE` units a term, and its drain a unit per term and
    /// word; and the substitute's terms interned `PROBE` units and a unit
    /// a word each and sorted (`substitute`).
    Substitute,
    /// The cofactor tracking: its compositions, sorts, and images, a term
    /// or an entry its weight, a term an image gathers `sort::KEY` units
    /// and its variables, and each multiplier interned as the parse's
    /// monomials; the combinations a row of a reduction starts from, takes
    /// away, and keeps, a unit per row they visit; and an input's entries,
    /// a unit a generator (`cofactors`).
    #[cfg_attr(
        not(feature = "cofactors"),
        allow(
            dead_code,
            reason = "a build without the cofactors spends nothing here; the site stays so that every build numbers the sites alike"
        )
    )]
    Cofactors,
    /// The images of a run's result for the lift: a unit per term and
    /// exponent each image reads and writes, per word of the leading
    /// monomials each lane's image copies, and a unit a word of the arrays
    /// a kept shape's table copies (`entry::leads`, `sparse`,
    /// `sparse_lanes`, `pair_image`, `Run::finish`, `shape::Kept::put`).
    Images,
    /// The lift's appends of a prime's images: a tick per part and
    /// `lift::LIMB` units a word written (`lift::append`).
    Append,
    /// The lift's merges of an image whose terms the part lacks: the
    /// comparison a unit per 8 exponents, `lift::MERGED` plus the words and
    /// `lift::MERGE_PASSES` units a variable a term, and the merged terms'
    /// copy into the image a unit per 8 exponents; the lifting's elements
    /// made, a unit per part and per variable of a leading monomial; and an
    /// image's parts, a unit each (`lift::merge_in`, `Lifting::new`,
    /// `lift::parts`).
    Merge,
    /// The lift's reconstructions over ℚ: a tick per coefficient, the
    /// conversions' multiply-adds a unit each and the reconstruction that
    /// follows `lift::RECONSTRUCTION` units per square of the images
    /// converted; a part's chain residues a unit per prime and limb; and
    /// the earlier rationals' checks a unit per term and limb
    /// (`lift::Element::attempt`, `lift::Scratch::convert`,
    /// `lift::Rationals::fits`).
    Reconstruct,
    /// The verification of a complete lifting: the test of its
    /// denominators at a prime a unit per element and limb, and its basis
    /// compared with a run's a unit per term, per limb of its numerator,
    /// and per 8 exponents, and `INVERSE` a chain denominator inverted
    /// (`lift::Complete::divides_a_denominator`, `lift::Complete::verify`).
    Verify,
    /// The texts' set-up over ℚ: the basis text's and the cofactor text's
    /// reservations a unit a term, and the exponents copied for the writers
    /// of each element and each cofactor a unit an exponent
    /// (`lift::finish`).
    Finish,
    /// The texts of the result: a full check after the allocator's trim
    /// before them (`entry::Run::finish`, `lift::finish`), a unit a byte
    /// written, a unit a variable
    /// of each monomial a basis text writes first and of each term a
    /// cofactor text or a text over ℚ writes, and over ℚ the square of the
    /// limbs of each integer computed and scaled (`text::write_basis`,
    /// `text::write_poly`, `rational`, `lift::finish`).
    Text,
    /// FGLM's change of order: a full check at its start and a tick per
    /// step; the test for zero dimension a unit per exponent it may read;
    /// the staircase's walk `PROBE` units a product and a unit per leading
    /// monomial a new product tests, its sort by `sort_runs`, and its
    /// places renamed a unit each; the border's forms `PROBE` units per
    /// term or quotient looked up; a product's reads `PROBE` units an entry
    /// and its rows a unit an entry; the candidates' insertions and pops
    /// `PROBE` units and a unit per level of the heap; the echelon's loop a
    /// unit per row and its row operations a unit an entry; the dense rows
    /// read back a unit an entry; and the division's steps a unit per
    /// element its search tests, `PROBE` units a product, and a unit per
    /// level of its heap (`fglm`).
    Fglm,
    /// The checker's polls, a tick each, and the rows of a block order's
    /// matrix, a unit an entry (`entry::certify`, `entry::checker_order`).
    #[cfg_attr(
        not(feature = "checker"),
        allow(
            dead_code,
            reason = "a build without the checker spends nothing here; the site stays so that every build numbers the sites alike"
        )
    )]
    Checker,
    /// The table's growth: its new slots a unit each and its ids placed
    /// `PROBE` units each, `table::PLACE` at a time (`Table::grow`).
    Place,
    /// The table's compaction: the ids walked and the words moved a unit
    /// each, its slots emptied a unit each, and its ids placed `PROBE`
    /// units each, `table::PLACE` at a time (`Table::compact`,
    /// `f4::compact`).
    Compact,
}

#[cfg(test)]
impl Site {
    /// Every variant, in the order the enum declares them, so that a
    /// variant as `usize` is its index here.
    pub(crate) const ALL: [Site; 45] = [
        Site::Names,
        Site::Parse,
        Site::Map,
        Site::Run,
        Site::Round,
        Site::Prime,
        Site::Monic,
        Site::Sugar,
        Site::Select,
        Site::PairRows,
        Site::Products,
        Site::Divisors,
        Site::Kept,
        Site::Columns,
        Site::SigSort,
        Site::Setup,
        Site::Row,
        Site::Scan,
        Site::BackBands,
        Site::Inter,
        Site::SigScan,
        Site::Decode,
        Site::Enter,
        Site::Update,
        Site::SigEnter,
        Site::Antichain,
        Site::Reducer,
        Site::Queue,
        Site::Koszul,
        Site::Carry,
        Site::Trace,
        Site::Echelon,
        Site::Substitute,
        Site::Cofactors,
        Site::Images,
        Site::Append,
        Site::Merge,
        Site::Reconstruct,
        Site::Verify,
        Site::Finish,
        Site::Text,
        Site::Fglm,
        Site::Checker,
        Site::Place,
        Site::Compact,
    ];
}

// `Budget::checked_at` counts by `site as usize`, which `Site::ALL` holds in
// order.
#[cfg(test)]
const _: () = {
    let mut i = 0;
    while i < Site::ALL.len() {
        assert!(Site::ALL[i] as usize == i, "Site::ALL in declared order");
        i += 1;
    }
};

/// The bounds of one call, and the work done since their last full check:
/// ticks, a row or another step of the run each, and units of work
/// (`WORK`), which the matrices spend as they are built, sorted, and
/// reduced, the monomial table as it grows and compacts, the texts as they
/// are written, the cofactors as they are
/// composed, lifted, and sorted into a replay's image, and over ℚ the
/// lift's images as they are merged and appended and converted at the
/// attempts. A full check comes once either reaches its bound,
/// and every full check resets both. The budget also counts the entries the
/// row operations subtract and those they add into combinations, and the
/// row work, which a bound may cap (`bound`), and holds the call report
/// where the call asks for it.
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
    // call-report.2 design-checked -- the report travels inside the call's budget, which every stage that reports takes already, not a parameter of its own beside it nor a thread-local report (not measured)
    /// The call report, where the call asks for it.
    report: Option<Box<Report>>,
    // call-report.4 design-checked -- the budget counts, every call, the entries the row operations subtract and those they add into combinations, in the spend of each row operation, not counts kept only where the report is on, a count in each reduction's result, nor the reduction's units of work (not measured)
    /// The entries the row operations subtracted since `new`.
    subtracted: u64,
    /// The entries the row operations added into combinations since `new`.
    added: u64,
    // gf2-signature-losses.3 design-checked -- the row work counts the entries the sparse row operations subtract and add and the words the packed rows XOR, not the budget's units, which spend a packed row's entries (gf2-reduction-time.11), nor its subtracted entries, nor the time (log 0129-1)
    /// The row work since `new`: the entries the sparse row operations
    /// subtracted and added into combinations, and the words of the packed
    /// rows added into the accumulator (`spend_packed`).
    row_work: u64,
    /// The row work past which a run halts (`outworked`), None for none.
    work_bound: Option<u64>,
    /// The pairs the round under reduction took and the live pairs its
    /// sugar held, those it took among them, over which the projection of
    /// its row work is taken (`over_sugar`); (1, 1), the whole sugar,
    /// outside such a reduction.
    share: (usize, usize),
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
    #[cfg(test)]
    forcing: Forcing,
}

/// What the test profile forces on a budget, read from its thread's
/// `CHECKS_FORCED` and `STOP_FORCED` where `Budget::new` makes it, and the
/// full checks it made by site.
#[cfg(test)]
struct Forcing {
    /// Ticks from one full check to the next: `TICKS`, or 1 where every
    /// tick checks.
    ticks: u32,
    /// The site whose first full check stops the call, where one is forced.
    stop: Option<Site>,
    /// Whether the forced stop has come: every later full check stops as
    /// well, as at a deadline passed.
    stopped: bool,
    /// Whether the sites of the full checks go to `CHECKED`.
    recording: bool,
    /// Whether every full check gives back what the run can rebuild, as above the ceiling, and
    /// goes on.
    releasing: bool,
    /// The full checks by site, in `Site::ALL`'s order.
    by_site: [u64; Site::ALL.len()],
}

#[cfg(test)]
impl Forcing {
    /// Nothing forced and nothing recorded.
    fn none() -> Forcing {
        Forcing {
            ticks: TICKS,
            stop: None,
            stopped: false,
            recording: false,
            releasing: false,
            by_site: [0; Site::ALL.len()],
        }
    }

    /// What this thread forces now.
    fn of_thread() -> Forcing {
        let (every, stop) = (CHECKS_FORCED.get(), STOP_FORCED.get());
        Forcing {
            ticks: if every { 1 } else { TICKS },
            stop,
            recording: every || stop.is_some(),
            releasing: RELEASE_FORCED.get(),
            ..Forcing::none()
        }
    }

    // call-stop-latency.39 design-checked -- after the forced stop every later full check stops as well, as at a deadline passed, not the forced site's first check alone (not measured)
    /// A full check at `site`, counted and, where the budget records,
    /// recorded: `Stop::Deadline` where it is the forced site's first, or
    /// comes after it.
    fn checked(&mut self, site: Site) -> Result<(), Stop> {
        self.by_site[site as usize] += 1;
        if self.recording {
            CHECKED.with_borrow_mut(|sites| sites.push(site));
        }
        if self.stopped || self.stop == Some(site) {
            self.stopped = true;
            return Err(Stop::Deadline);
        }
        Ok(())
    }
}

/// The start of a call that `params` asks a deadline or the call report
/// of, where the system has a clock (`os::CLOCK`); None otherwise. A call
/// with neither reads no clock, so it runs where there is none.
pub(crate) fn start(params: &Params) -> Option<Instant> {
    let timed = params.deadline.is_some() || params.report;
    (timed && os::CLOCK).then(Instant::now)
}

impl Budget {
    /// The budget of a call that started at `start`, with the call report
    /// where `params` asks for it. A ceiling where the resident set cannot
    /// be read is `Error::Memory`, and a deadline or the call report with no
    /// start, where the system has no clock (`start`), is `Error::Clock`. In
    /// the test profile it takes what its thread's `CHECKS_FORCED` and
    /// `STOP_FORCED` force.
    pub(crate) fn new(start: impl Into<Option<Instant>>, params: &Params) -> Result<Budget, Error> {
        let start = start.into();
        let memory = match params.memory {
            None => None,
            Some(ceiling) => Some((ceiling, Resident::open().ok_or(Error::Memory)?)),
        };
        #[cfg(not(test))]
        let (left, work) = (TICKS, WORK);
        #[cfg(test)]
        let forcing = Forcing::of_thread();
        // Where every spend checks, no unit is left until the next check.
        #[cfg(test)]
        let (left, work) = (forcing.ticks, if CHECKS_FORCED.get() { 0 } else { WORK });
        let deadline = match params.deadline {
            None => None,
            Some(d) => start.ok_or(Error::Clock)?.checked_add(d),
        };
        let report = if params.report {
            let start = start.ok_or(Error::Clock)?;
            Some(Box::new(Report::new(start, params.report_sink.clone())))
        } else {
            None
        };
        Ok(Budget {
            deadline,
            cancel: params.cancel.clone(),
            memory,
            left,
            work,
            report,
            subtracted: 0,
            added: 0,
            row_work: 0,
            work_bound: None,
            share: (1, 1),
            #[cfg(test)]
            per: work,
            #[cfg(test)]
            spent: 0,
            #[cfg(test)]
            checks: 0,
            #[cfg(test)]
            forcing,
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
            row_work: 0,
            work_bound: None,
            share: (1, 1),
            #[cfg(test)]
            per: WORK,
            #[cfg(test)]
            spent: 0,
            #[cfg(test)]
            checks: 0,
            #[cfg(test)]
            forcing: Forcing::none(),
        }
    }

    /// The full check at `site`: the cancel flag, the deadline, and the
    /// ceiling, in that order. It first resets the ticks and the units of
    /// work left until the next. Where the resident set is above the
    /// ceiling, `release` gives back what the run can rebuild, the
    /// allocators give back what is free, and the resident set is read
    /// again. In the test profile the check is counted at its site, and
    /// stops where a stop is forced (`Forcing::checked`).
    pub(crate) fn check(
        &mut self,
        release: impl FnOnce(),
        #[cfg_attr(
            not(test),
            allow(
                unused_variables,
                reason = "the site is read in the test profile alone"
            )
        )]
        site: Site,
    ) -> Result<(), Stop> {
        #[cfg(not(test))]
        {
            self.left = TICKS;
            self.work = WORK;
        }
        #[cfg(test)]
        {
            self.left = self.forcing.ticks;
            self.work = self.per;
            self.checks += 1;
            self.forcing.checked(site)?;
        }
        if self.cancel.as_ref().is_some_and(Cancel::is_set) {
            return Err(Stop::Cancel);
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(Stop::Deadline);
        }
        #[cfg(not(test))]
        let forced = false;
        #[cfg(test)]
        let forced = self.forcing.releasing;
        let passed = self
            .memory
            .as_ref()
            .is_some_and(|(ceiling, resident)| above(resident, *ceiling));
        if passed || forced {
            #[cfg(test)]
            RELEASES.set(RELEASES.get() + u64::from(forced));
            release();
            trim();
            if let Some((ceiling, resident)) = &self.memory
                && above(resident, *ceiling)
            {
                return Err(Stop::Memory);
            }
        }
        Ok(())
    }

    // f4-gfp.27 design-checked -- the budget is ticked once per row and checked fully every 64 ticks and at the start of every round, not checked at the start of each round alone (not measured)
    /// One row, or another step, of work at `site`: a full check where it
    /// is the `TICKS`th tick since the last, with nothing of the run's to
    /// give back.
    #[inline]
    pub(crate) fn tick(&mut self, site: Site) -> Result<(), Stop> {
        self.left -= 1;
        if self.left == 0 {
            return self.check(|| {}, site);
        }
        Ok(())
    }

    // matrix-stop-latency.1 design-checked -- the budget counts units of work beside its ticks, and a full check comes once 64 ticks or WORK units have passed since the last, whichever first, not more ticks per check, fewer rows per check, or a clock read per row (log 0056-1)
    /// `units` of work done at `site`: a full check where they reach the
    /// units left until the next, with nothing of the run's to give back.
    #[inline]
    pub(crate) fn spend(&mut self, units: usize, site: Site) -> Result<(), Stop> {
        let units = units as u64;
        #[cfg(test)]
        {
            self.spent += units;
        }
        if units < self.work {
            self.work -= units;
            return Ok(());
        }
        self.check(|| {}, site)
    }

    // call-report.4 design-checked -- the budget counts, every call, the entries the row operations subtract and those they add into combinations, in the spend of each row operation, not counts kept only where the report is on, a count in each reduction's result, nor the reduction's units of work (not measured)
    /// `units` entries a row operation subtracted, counted, added to the
    /// row work, and spent at `site`.
    #[inline]
    pub(crate) fn spend_subtracted(&mut self, units: usize, site: Site) -> Result<(), Stop> {
        self.subtracted += units as u64;
        self.row_work += units as u64;
        self.spend(units, site)
    }

    /// `units` entries a row operation added into a combination, counted,
    /// added to the row work, and spent at `site`.
    #[inline]
    pub(crate) fn spend_added(&mut self, units: usize, site: Site) -> Result<(), Stop> {
        self.added += units as u64;
        self.row_work += units as u64;
        self.spend(units, site)
    }

    /// A packed row of `entries` entries in `words` words added into the
    /// accumulator by exclusive or (`packed::Rows`): its entries counted as subtracted and
    /// spent at `Site::Row`, as `spend_subtracted` counts and spends them, and its words
    /// added to the row work.
    #[inline]
    pub(crate) fn spend_packed(&mut self, entries: usize, words: usize) -> Result<(), Stop> {
        self.subtracted += entries as u64;
        self.row_work += words as u64;
        self.spend(entries, Site::Row)
    }

    /// The row work since `new`: the entries the sparse row operations
    /// subtracted and added, and the words of the packed rows added.
    pub(crate) fn row_work(&self) -> u64 {
        self.row_work
    }

    /// Bounds the row work at its value now plus `units`, saturating at
    /// `u64::MAX`, which no row work passes, and returns the units. In the
    /// test profile `WORK_FORCED` gives the units where it holds them.
    pub(crate) fn bound(&mut self, units: u64) -> u64 {
        #[cfg(test)]
        let units = WORK_FORCED.get().unwrap_or(units);
        self.work_bound = Some(self.row_work.saturating_add(units));
        units
    }

    /// Lifts the bound on the row work.
    pub(crate) fn unbound(&mut self) {
        self.work_bound = None;
    }

    /// Whether a bound on the row work is set and the row work lies past it.
    #[inline]
    pub(crate) fn outworked(&self) -> bool {
        self.work_bound.is_some_and(|bound| self.row_work > bound)
    }

    // gf2-build-time.3 design-checked -- a round that takes a piece of its sugar projects its row work over the sugar's live pairs, the piece's and those it leaves, over the piece's, not over the piece's rows alone (gf2-bound-losses.1), under which each of bivium-114's pieces projected below the bound and its bounded run went on to 5.04 s (log 0159-1, 0159-2)
    // gf2-build-time.5 design-checked -- the share holds for the span of the round's reduction, which `over_sugar` opens and closes around it, not a parameter of `matrix::reduce`, which its reducer would carry to the bound's one reader, nor a field the round sets and leaves, which the inter-reduction after the last round would read (log 0159-1, 0159-2)
    /// Runs `reduce` with the projection of the row work
    /// (`projected_past`) taken over a round's sugar: of the `sugar` live
    /// pairs the sugar held, the `piece` the round took (gf2-build-time.3).
    /// Outside `reduce` the projection is taken over the whole sugar again.
    pub(crate) fn over_sugar<R>(
        &mut self,
        piece: usize,
        sugar: usize,
        reduce: impl FnOnce(&mut Budget) -> R,
    ) -> R {
        debug_assert!(
            (1..=sugar).contains(&piece),
            "a piece of {piece} pairs of a sugar of {sugar}"
        );
        let whole = std::mem::replace(&mut self.share, (piece, sugar));
        let reduced = reduce(self);
        self.share = whole;
        reduced
    }

    // gf2-bound-losses.1 design-checked -- a round's row work is projected over its rows to reduce from the rows its reduction has begun, and the run halts where the projection passes the bound, not only where the row work done passes it, nor before a round in sparse rows of 2^24 entries or more, nor with the least sugar split at every selection past its bound (log 0131-1, 0131-2); taken over the round's sugar by gf2-build-time.3
    /// Whether a bound on the row work is set and the row work projected
    /// over a reduction passes it: `start`, the row work when the
    /// reduction's rows began, plus the row work since times `rows`, the
    /// rows it reduces, over `reached`, the rows it has begun, and within
    /// `over_sugar` times the sugar's live pairs over the piece's. The
    /// projection is taken in u128, saturating, and one past `u64::MAX` as
    /// `u64::MAX`, which passes no bound.
    #[inline]
    pub(crate) fn projected_past(&self, start: u64, reached: usize, rows: usize) -> bool {
        debug_assert!(
            start <= self.row_work,
            "the start {start} past the row work"
        );
        debug_assert!(
            (1..=rows).contains(&reached),
            "{reached} rows begun of {rows}"
        );
        let Some(bound) = self.work_bound else {
            return false;
        };
        let (piece, sugar) = self.share;
        let since = u128::from(self.row_work - start);
        let over = reached as u128 * piece as u128;
        let projected = u128::from(start).saturating_add(
            since
                .saturating_mul(rows as u128)
                .saturating_mul(sugar as u128)
                / over,
        );
        projected.min(u128::from(u64::MAX)) > u128::from(bound)
    }

    /// The entries the row operations subtracted, and those they added
    /// into combinations, since `new`.
    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.subtracted, self.added)
    }

    // call-report.3 design-checked -- each hook tests at run time whether the report is on, not a type parameter whose off implementation compiles to nothing, as the trace and the cofactor tracking (f4-gfp.25) (not measured)
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

    /// The full checks at `site` since `new`.
    #[cfg(test)]
    pub(crate) fn checked_at(&self, site: Site) -> u64 {
        self.forcing.by_site[site as usize]
    }
}

// f4-gfp.50 design-checked -- a resident set that cannot be read after the start counts as above the ceiling, not as below it (not measured)
/// Whether the resident set is above `ceiling`. A resident set that can
/// no longer be read is.
fn above(resident: &Resident, ceiling: u64) -> bool {
    resident.bytes().is_none_or(|b| b > ceiling)
}

// f4-gfp.29 design-checked -- before a memory stop the run gives back what it can rebuild and glibc's malloc_trim(0) gives free memory back to the system, not the buffers freed alone (not measured)
/// Gives glibc's free memory back to the system.
pub(crate) fn trim() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only returns free memory of glibc's heap to the
    // system; no live allocation moves.
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(test)]
thread_local! {
    // freed-heap-text-peak.6 design-checked -- in the test profile a thread-local turns the trim before the texts off on its thread, so that the test code compares a call with the trim against the same call without, not a parameter of the call (not measured)
    /// Whether the calls on this thread give nothing back before their
    /// texts (`trim_before_texts`).
    pub(crate) static TEXTS_UNTRIMMED: Cell<bool> = const { Cell::new(false) };

    // gf2-signature-losses.10 design-checked -- in the test profile a thread-local sets the bound's units, so that test code starts calls over at a bound of 0, not a parameter of the call nor the constant lowered (not measured)
    /// Where Some, the units every `Budget::bound` on this thread takes in
    /// place of its caller's.
    pub(crate) static WORK_FORCED: Cell<Option<u64>> = const { Cell::new(None) };

    // call-stop-latency.5 practice-checked: failpoints -- every spend, tick, and full check names its loop by a Site, and the test profile records the checks' sites, forces a check at every spend and tick, and forces a stop at a site's first check, not the sites named in comments alone nor a test per loop that builds the loop's state by hand (not measured)
    /// Where true, a budget `Budget::new` makes on this thread checks in
    /// full at every spend and every tick.
    pub(crate) static CHECKS_FORCED: Cell<bool> = const { Cell::new(false) };

    /// Where Some, a budget `Budget::new` makes on this thread stops with
    /// `Stop::Deadline` at its first full check at that site, and at every
    /// full check after it.
    pub(crate) static STOP_FORCED: Cell<Option<Site>> = const { Cell::new(None) };

    /// Where true, a budget `Budget::new` makes on this thread gives back what the run can
    /// rebuild at every full check, as above the ceiling, and goes on: no run passes its ceiling
    /// in the test code, so this is what runs the release closures.
    pub(crate) static RELEASE_FORCED: Cell<bool> = const { Cell::new(false) };

    /// The releases `RELEASE_FORCED` made on this thread.
    pub(crate) static RELEASES: Cell<u64> = const { Cell::new(0) };

    // call-stop-latency.40 design-checked -- the sites go to CHECKED from the budgets Budget::new made while a check or a stop was forced alone, not from every budget of the thread, whose record would grow through every test of the thread (not measured)
    /// The sites of the full checks, in order, of the budgets `Budget::new`
    /// made on this thread while a check or a stop was forced.
    pub(crate) static CHECKED: RefCell<Vec<Site>> = const { RefCell::new(Vec::new()) };
}

// freed-heap-text-peak.2 design-checked -- the free memory goes back by `trim`, glibc's malloc_trim(0), not a malloc_trim that keeps a pad free at the heap's top, which would keep the pad resident beside a text that takes a mapping of its own (not measured)
// freed-heap-text-peak.3 design-checked -- every call gives its free memory back before its texts, whatever the texts' size, not only a call whose basis text's reservation passes glibc's mmap threshold (log 0116-1)
// freed-heap-text-peak.7 design-checked -- the call report names no figure of the trim, not a field of the `text` line with the resident set given back, which would read the resident set twice a call (not measured)
/// Gives glibc's free memory back (`trim`) once a call's result is
/// complete and its texts are yet to be written: what the run freed below
/// glibc's heap top then no longer stands beside the texts' fresh pages. In
/// the test profile `TEXTS_UNTRIMMED` turns it off on its thread.
pub(crate) fn trim_before_texts() {
    #[cfg(test)]
    if TEXTS_UNTRIMMED.get() {
        return;
    }
    trim();
}
