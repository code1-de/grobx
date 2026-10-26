//! The F4 run (faugere-1999 §2): the inputs enter the basis, each round
//! reduces the matrix of the pairs of least sugar, in a graded order of
//! least degree, and the elements of `G_min` are inter-reduced to the reduced
//! basis.

use std::cell::LazyCell;

use crate::budget::Budget;
use crate::cofactors::Cofactors;
use crate::field::Field;
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{self, Buffers, Combine, Matrix, Reduced, Reduction};
use crate::module::Module;
use crate::monomial::{Format, Overflow};
use crate::narrow::{Narrow, Size};
use crate::order::Order;
use crate::pairs::{self, NewPairs, Pair, PairSet};
use crate::poly::{Basis, Poly};
use crate::report::{Opt, Round, Secs, Stage};
use crate::shape::{Built, Shaping, Taken};
use crate::sig::{self, Base, BaseDivisors, Pairs, Queue, Sig, Signatures, Triangle};
use crate::table::{Limit, Table};
use crate::trace::{self, Learner, Record, RowSource, Step, Trace};
use crate::{Stats, Stop};

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
        }
    }
}

impl From<Stop> for Halt {
    fn from(stop: Stop) -> Halt {
        Halt::Stop(stop)
    }
}

/// The run's elements on their way into the basis, each with its
/// cofactors.
pub(crate) type Elements<F, D, E> = Vec<(Poly<F, D>, E)>;

// f4-gfp.25 practice: monomorphization -- the trace and the cofactor tracking are type parameters of run, and their empty implementations zero-sized types with empty methods, not a runtime flag tested at each hook
// trace-replay.10 design -- the learn run tells the record through a type parameter of run and run_sig whose methods take a closure that builds what is recorded, not hooks added to the public Trace (not measured)
/// The reduced Gröbner basis of the generators, by increasing leading
/// monomial, each element with its cofactors. A round's new rows are
/// inter-reduced (`matrix::reduce`). Where neither the trace nor the
/// learner listens, no cofactors are tracked, and p is at least 46341, the
/// rows of a round's matrix are reduced in random combinations, whose
/// coefficients `seed` draws. Where the learner listens, the trace does
/// not, no cofactors are tracked, and p is above 3, the rows are reduced
/// each on its own
/// and tested for rows that reduce to zero (`matrix::reduce`), the tests'
/// coefficients drawn by `seed`. The budget is checked at the start, at the start of every round,
/// and before the inter-reduction, and wherever the ticks or the units
/// of work of a matrix, or of the cofactors' composition, reach a full
/// check (`budget::Budget`). `stats` counts the rounds as their
/// matrices are reduced, and `learner` hears the inputs and the step of every round. The
/// rounds' matrices take their buffers in turn from one `matrix::Buffers`,
/// which goes back where the rounds end, before the inter-reduction. The
/// call report, where the budget holds one, gets a line per round `stats`
/// counts, one for the inter-reduction, and one where the run makes a
/// constant.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and generators of the run, the cofactors, trace, budget, counts, and learner it borrows, and its seed, none of which belongs with another"
)]
pub(crate) fn run<F, M, O, D, C, T, L>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    cof: &mut C,
    trace: &mut T,
    budget: &mut Budget,
    stats: &mut Stats,
    learner: &mut L,
    seed: u64,
) -> Result<Elements<F, D, C::Element>, Halt>
where
    F: Field,
    M: Format,
    O: Order<M>,
    D: Module,
    C: Cofactors<F, M>,
    T: Trace,
    L: Learner,
{
    budget.check(|| {})?;
    learner.inputs(|| trace::inputs(table, order, &generators, true, true));
    let mut combine = combines::<F, M, C, T, L>().then(|| Combine::new(seed));
    let combined = plain_combines::<F, M, C, T, L>(field.characteristic());
    // learn-zero-time.5 design -- the tests' coefficients come from a Combine seeded by the run's seed, the draw of the plain run's combinations, which a learn run does not take, not a draw of their own from the call's generator (not measured)
    let mut tests = learn_tests::<F, M, C, T, L>().then(|| Combine::new(seed));
    let mut inputs = monic_inputs(field, generators, cof)?;
    if let Some(one) = constant_input(&mut inputs, table, budget) {
        return Ok(one);
    }
    let mut basis: Basis<F, D> = Basis::new();
    let mut pairs = PairSet::new();
    let mut new_pairs: NewPairs<M::Word> = NewPairs::new();
    // Where the order is not graded, every element's sugar, by its index in
    // the basis. A graded order keeps none.
    let mut sugar: Vec<u32> = Vec::new();
    // An input's sugar is the largest total degree of its terms
    // (giovini-1991 §2): the round's sugar 0 lies below every degree.
    enter(
        table,
        order,
        &mut basis,
        &mut pairs,
        &mut new_pairs,
        &mut sugar,
        0,
        cof,
        inputs,
    )?;

    let mut bufs: Buffers<F> = Buffers::new();
    // f4-gfp.4 cited: monagan-2015 §2 -- simplifying as suggested by Faugère gave no speedup in their experiments
    let mut round = 0;
    while !pairs.is_empty() {
        budget.check(|| {
            release(table, &mut basis, &mut pairs, &mut new_pairs);
            bufs.release();
        })?;
        budget.open();
        round += 1;
        let selected = pairs::select(&mut pairs);
        let d = selected[0].sugar;
        let rows = pair_rows(table, &basis, &selected)?;
        budget.mark(Stage::Select);
        let m = matrix::build(table, &basis, order, round, rows, bufs.take_cols(), budget)?;
        trace.matrix(&m.rows);
        // call-report.22 design -- the trace's hook of the matrix before its reduction counts into the report's record stage, not into reduce (not measured)
        budget.mark(Stage::Record);
        cof.matrix(m.rows.len());
        // gfp-unsolved.1 cited: demin-2024 §4.2 -- the rows of D′ are inter-reduced to obtain the reduced row echelon form; boyer-2016 §4.1, independently -- a fully reduced row echelon form reduces the rows reused in later steps, which can be beneficial
        let reduction = matrix::reduce(
            field,
            &basis,
            &m,
            cof,
            budget,
            true,
            combine.as_mut(),
            &mut bufs,
            tests.as_mut(),
        )?;
        budget.mark(Stage::Reduce);
        count(stats, &reduction);
        budget.counted(|| {
            let bands = combined && matrix::takes_bands::<F, M, C>(&m);
            figures(round, d, selected.len(), &m, &reduction, combined, bands)
        });
        learner.step(|| trace::step(table, order, &m, &reduction, true));
        let mut made = reduction.made;
        // f4-gfp.52 design -- the rows that counted are listed and sorted for every matrix, the trace on or off, not only where a constant of the trait says the trace listens (not measured)
        let mut counted: Vec<RowId> = made.iter().map(|r| r.row).collect();
        counted.sort_unstable();
        trace.counted(&counted);
        budget.mark(Stage::Record);
        // f4-gfp.22 design -- a constant element ends the run with the basis {1}, not running until no pair is left (not measured)
        if let Some(one) = constant_made(&mut made, &m, table, cof, budget)? {
            budget.end_round(Some(pairs.len()));
            report_constant(budget, round);
            return Ok(one);
        }
        let mut new = Vec::with_capacity(made.len());
        for r in made {
            let element = cof.compose(r.kept, &m.rows, table, budget)?;
            new.push((r.poly(&m), element));
        }
        budget.mark(Stage::Decode);
        enter(
            table,
            order,
            &mut basis,
            &mut pairs,
            &mut new_pairs,
            &mut sugar,
            d,
            cof,
            new,
        )?;
        budget.mark(Stage::Update);
        budget.end_round(Some(pairs.len()));
        bufs.keep_cols(m);
    }

    // round-memory-time.7 design -- the buffers live for one run's rounds and go back where the rounds end, before the inter-reduction, not buffers that live for the call or for the thread (log 0051-1)
    drop(bufs);
    budget.check(|| release(table, &mut basis, &mut pairs, &mut new_pairs))?;
    reduced_basis(field, table, order, &basis, round + 1, cof, trace, budget)
}

/// The nonzero generators, made monic, each with its cofactors.
/// `Halt::Misfit(0)` where a nonzero generator's leading coefficient has
/// no inverse: over several lanes, 0 at one lane.
fn monic_inputs<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    generators: Vec<Poly<F, D>>,
    cof: &mut C,
) -> Result<Elements<F, D, C::Element>, Halt> {
    let mut inputs = Vec::new();
    for (j, mut g) in generators.into_iter().enumerate() {
        if g.is_zero() {
            continue;
        }
        // q-learn-time.4 cited: dekleine-2001 §3 -- on each division with a nonzero remainder the list of leading coefficients is checked, and a c_i = 0 means the leading term of r is lost under p_i, with very high probability a bad prime; graebe-1993 §4, independently -- a leading coefficient detected as nonzero that is zero mod p_i shows p_i unlucky
        let Some(s) = g.try_monic(field) else {
            return Err(Halt::Misfit(0));
        };
        inputs.push((g, cof.input(j, s)));
    }
    Ok(inputs)
}

/// The run's basis {1} where an input of `inputs` is constant: that input,
/// taken out of `inputs`, after the call report's line of a constant at
/// round 0. None where no input is constant.
fn constant_input<F: Field, M: Format, D: Module, E>(
    inputs: &mut Elements<F, D, E>,
    table: &Table<M>,
    budget: &mut Budget,
) -> Option<Elements<F, D, E>> {
    let k = inputs
        .iter()
        .position(|(g, _)| table.degree(g.lead()) == 0)?;
    report_constant(budget, 0);
    Some(vec![inputs.swap_remove(k)])
}

/// The run's basis {1} where a row of `made`, the rows the matrix `m`
/// made, is constant: that row, with its cofactors, after the round's
/// decode stage is marked. None where no row is constant. A constant has
/// one column, the last, and its cofactors alone are composed.
fn constant_made<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    made: &mut [Reduced<F>],
    m: &Matrix,
    table: &Table<M>,
    cof: &mut C,
    budget: &mut Budget,
) -> Result<Option<Elements<F, D, C::Element>>, Halt> {
    let Some(constant) = made
        .iter_mut()
        .find(|r| table.degree(m.monomials[r.cols[0].index()]) == 0)
    else {
        return Ok(None);
    };
    let element = cof.compose(constant.kept, &m.rows, table, budget)?;
    let one = constant.take_poly(m);
    budget.mark(Stage::Decode);
    Ok(Some(vec![(one, element)]))
}

/// Adds a round's counts to the stats.
fn count<F: Field>(stats: &mut Stats, reduction: &Reduction<F>) {
    stats.rounds += 1;
    stats.reduced += reduction.reductions;
    stats.zero += reduction.zeros;
}

// gfp-unsolved.6 cited: berthomieu-2021 §4.3, Remark 4 -- with the tracer the probabilistic linear algebra cannot be used in the learn run, since it could not detect which specific rows reduce to zero; berthomieu-2022a §3.4, Tracer over Q -- once nothing more is learned, probabilistic linear algebra accelerates the step
// gfp-unsolved.8 design -- a run that tracks cofactors reduces each row on its own, not combinations with their cofactors tracked through a hook that begins an empty combination and adds its rows (log 0014-1)
// signature-zero-time.10 design -- the tests run where no cofactors are tracked, neither the trace nor the learner listens, and the field has one lane, not in a learn run as well (log 0049-1)
// call-report.11 design -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
/// Whether a run's hooks let it reduce rows in random combinations, the
/// plain run (`matrix::combines_rows`), or rule them zero by tests, the
/// signature run (`matrix::tests_rows`): neither the trace nor the learner
/// listens, and no cofactors are tracked.
pub(crate) fn combines<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>() -> bool {
    !T::LISTENS && !L::LISTENS && !C::TRACKS
}

// learn-zero-time.1 invented -- a learn run's plain run, where no cofactors are tracked, rules its rows zero by signature-zero-time.1's test of random combinations, the rows a test rules zero recorded as rows that reduced to zero, not each row reduced on its own (log 0063-1)
// learn-zero-time.4 design -- where the trace listens the plain run reduces each row on its own, as `groebner_with` states, not tests where the trace listens as well (not measured)
// zero-rows-time.1 design -- the learn's plain run takes learn-zero-time's tests beside the plain run's grown blocks of plain-zero-time, not each row of a learn reduced on its own beside the grown blocks, as gfp-unsolved.6 has it, nor the tests without the grown blocks (log 0078-1)
// q-zero-work-time.5 design -- the run line names a learn's plain run that takes tests `reduce=tests`, from this function, which `run` decides by, not `reduce=rows reduce-by=learner` as for a learn's signature run (not measured)
/// Whether a plain run's hooks let it rule rows zero by tests while it
/// reduces each row on its own (`matrix::reduce`), where p takes a test
/// (`matrix::zero_combinations`): the learner listens, the trace does
/// not, and no cofactors are tracked.
pub(crate) fn learn_tests<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>() -> bool {
    !T::LISTENS && L::LISTENS && !C::TRACKS
}

/// Whether the plain run under these hooks combines rows at p: its hooks
/// let it (`combines`), and p takes combinations (`matrix::combines_rows`).
pub(crate) fn plain_combines<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>(
    p: u64,
) -> bool {
    debug_assert!(p >= 2, "p is a field's characteristic, a prime");
    combines::<F, M, C, T, L>() && matrix::combines_rows(p)
}

/// The figures of round `number` for its line in the call report: its
/// degree `degree`, its `pairs` pairs or reductants, its matrix `m` and the
/// reduction of it, which combined rows where `combined` and so named no
/// row that reduced to zero, and whose matrix takes bands where `bands`
/// (`matrix::takes_bands`).
fn figures<F: Field>(
    number: u32,
    degree: u32,
    pairs: usize,
    m: &Matrix,
    reduction: &Reduction<F>,
    combined: bool,
    bands: bool,
) -> Round {
    Round {
        number,
        degree,
        pairs,
        rows: m.rows.len(),
        given: m.given,
        columns: m.columns(),
        entries: m.entries(),
        pivots: reduction.made.len(),
        zero: reduction.zeros,
        zero_rows: (!combined).then_some(reduction.zero.len()),
        reductions: reduction.reductions,
        bands,
        pending: None,
        left: None,
    }
}

/// The call report's line of round `round` where a test of `decision` read
/// it (call-report.12): each test that read it with its verdict, the
/// elements made, the rows reduced to zero, and the widest gap, all three in
/// `read`, the count where a test read it, the S-pairs where the pair-work
/// test read them, and whether the run hands over.
/// Nothing where no test read the round, or `read` is None: the handover
/// did not decide after the round.
// pair-work-handover.7 design -- the handover line names the pair-work test's verdict and the S-pairs it read, not a line of its own, nor the S-pairs left out (not measured)
fn report_handover(
    budget: &mut Budget,
    round: u32,
    decision: &Handover,
    read: Option<(usize, usize, u32)>,
    handed: bool,
) {
    let Some((made, zero, widest)) = read else {
        return;
    };
    let t = decision.tests();
    if t.early.is_none() && t.gap.is_none() && t.outgrown.is_none() && t.pair_work.is_none() {
        return;
    }
    let Some(report) = budget.report() else {
        return;
    };
    let verdict = |h: bool| if h { "hand-over" } else { "stay" };
    report.line(
        "handover",
        format_args!(
            " round={round}{early}{gap}{outgrown}{pair_work} made={made} zero={zero}\
             {divisible}{pairs} widest={widest} verdict={verdict}",
            early = Opt("early", t.early.map(verdict)),
            gap = Opt("gap", t.gap.map(verdict)),
            outgrown = Opt("outgrown", t.outgrown.map(verdict)),
            pair_work = Opt("pair-work", t.pair_work.map(verdict)),
            divisible = Opt("divisible", t.divisible),
            pairs = Opt("pairs", t.pairs),
            verdict = verdict(handed),
        ),
    );
}

/// The call report's line of a run that made a constant in round `round`,
/// round 0 where an input is constant.
fn report_constant(budget: &mut Budget, round: u32) {
    if let Some(report) = budget.report() {
        report.line("constant", format_args!(" round={round}"));
    }
}

/// The call report's line of the inter-reduction's matrix `m`, which made
/// `elements` elements, its step opened before the matrix was built.
fn report_inter(budget: &mut Budget, m: &Matrix, elements: usize) {
    let counts = budget.counts();
    let Some(report) = budget.report() else {
        return;
    };
    let (subtracted, _) = report.grown(counts);
    let [symbolic, encode, reduce, decode] =
        [Stage::Symbolic, Stage::Encode, Stage::Reduce, Stage::Decode]
            .map(|s| Secs(report.time(s)));
    report.line(
        "inter",
        format_args!(
            " rows={rows} given={given} columns={columns} entries={entries} \
             subtracted={subtracted} elements={elements} symbolic={symbolic} encode={encode} \
             reduce={reduce} decode={decode}",
            rows = m.rows.len(),
            given = m.given,
            columns = m.columns(),
            entries = m.entries(),
        ),
    );
}

/// The reduced basis from `G_min`, a Gröbner basis, by the matrix of round
/// `round`, by increasing leading monomial, each element with its
/// cofactors, and the matrix's line in the call report.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and basis, the round, and the cofactors, trace, and budget it borrows, none of which belongs with another"
)]
fn reduced_basis<F, M, O, D, C, T>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    basis: &Basis<F, D>,
    round: u32,
    cof: &mut C,
    trace: &mut T,
    budget: &mut Budget,
) -> Result<Elements<F, D, C::Element>, Halt>
where
    F: Field,
    M: Format,
    O: Order<M>,
    D: Module,
    C: Cofactors<F, M>,
    T: Trace,
{
    budget.open();
    let m = inter_matrix(table, order, basis, round, budget)?;
    let made = inter_reduce(field, basis, &m, cof, trace, budget)?;
    budget.mark(Stage::Reduce);
    let elements = reduced_elements(table, order, &m, made, cof, budget)?;
    budget.mark(Stage::Decode);
    report_inter(budget, &m, elements.len());
    Ok(elements)
}

/// The matrix of round `round` that inter-reduces `G_min`: its elements, in
/// the order made, as the given rows, and their reducers by symbolic
/// preprocessing.
fn inter_matrix<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    order: &O,
    basis: &Basis<F, D>,
    round: u32,
    budget: &mut Budget,
) -> Result<Matrix, Halt> {
    let one = table.intern(&vec![0; table.format().variables()])?;
    let rows = basis
        .min()
        .iter()
        .map(|&i| RowSource {
            element: i,
            multiplier: one,
        })
        .collect();
    matrix::build(table, basis, order, round, rows, Vec::new(), budget)
}

/// The given rows of `m`, the inter-reduction's matrix of `G_min`, reduced
/// (`matrix::inter_reduce`) with buffers of its own, and the matrix told
/// to the trace.
fn inter_reduce<F, M, D, C, T>(
    field: &F,
    basis: &Basis<F, D>,
    m: &Matrix,
    cof: &mut C,
    trace: &mut T,
    budget: &mut Budget,
) -> Result<Vec<Reduced<F>>, Halt>
where
    F: Field,
    M: Format,
    D: Module,
    C: Cofactors<F, M>,
    T: Trace,
{
    trace.matrix(&m.rows);
    cof.matrix(m.rows.len());
    let made = matrix::inter_reduce(field, basis, m, cof, budget, &mut Buffers::new())?;
    trace.counted(&(0..m.given).map(RowId::at).collect::<Vec<RowId>>());
    Ok(made)
}

/// The rows `made` of the inter-reduction's matrix `m` as elements, by
/// increasing leading monomial, each with its cofactors.
fn reduced_elements<F, M, O, D, C>(
    table: &Table<M>,
    order: &O,
    m: &Matrix,
    made: Vec<Reduced<F>>,
    cof: &mut C,
    budget: &mut Budget,
) -> Result<Elements<F, D, C::Element>, Halt>
where
    F: Field,
    M: Format,
    O: Order<M>,
    D: Module,
    C: Cofactors<F, M>,
{
    let mut reduced = Vec::with_capacity(made.len());
    for r in made {
        let element = cof.compose(r.kept, &m.rows, table, budget)?;
        reduced.push((r.poly(m), element));
    }
    reduced.sort_by(|a, b| table.cmp(order, a.0.lead(), b.0.lead()));
    Ok(reduced)
}

// f4-gfp.2 cited: gebauer-1988 §4.6, via joux-2011 §2.2 Algorithm 2 -- with the inputs ordered by decreasing head term, an element is redundant exactly when a later one's head term divides its own
/// Enters the monic elements into the basis through the update, by
/// decreasing leading monomial, and their cofactors into the tracking.
/// Where the order is not graded, each element's sugar joins `sugar`: the
/// larger of `d`, the sugar of the round that made it, and the largest
/// total degree of its terms. `new_pairs` holds the update's buffers.
#[allow(
    clippy::too_many_arguments,
    reason = "the table, order, basis, pairs, and update buffers the elements enter, the sugar and the round's degree, and the cofactors and elements, none of which belongs with another"
)]
fn enter<F: Field, M: Format, O: Order<M>, D: Module, C: Cofactors<F, M>>(
    table: &mut Table<M>,
    order: &O,
    basis: &mut Basis<F, D>,
    pairs: &mut PairSet,
    new_pairs: &mut NewPairs<M::Word>,
    sugar: &mut Vec<u32>,
    d: u32,
    cof: &mut C,
    mut elements: Elements<F, D, C::Element>,
) -> Result<(), Limit> {
    elements.sort_by(|a, b| table.cmp(order, b.0.lead(), a.0.lead()));
    for (h, element) in elements {
        if !O::GRADED {
            // order-family.9 design -- a new element's sugar is the larger of the round's least sugar d and its own largest total degree, not the largest sugar of the matrix's rows (not measured)
            // order-family.10 cited: bigatti-2011 §3, the proposition on the elementary properties of the sugar, (1) -- sugar(U) is componentwise greater than or equal to the degree of U
            let top = h.monomials.iter().map(|&u| table.degree(u)).max();
            sugar.push(d.max(top.unwrap_or(0)));
        }
        pairs::update(table, order, basis, pairs, sugar, h, new_pairs)?;
        cof.push(element);
    }
    Ok(())
}

// f4-gfp.49 design -- inside a matrix a memory check gives back nothing of the run's own and between rounds the run shrinks its growable arrays, not the dense row dropped inside a matrix and allocated again for the next row (not measured)
/// Gives back what the run can rebuild between two rounds: the slack of its
/// table, its basis, and its pairs, and the update's buffers. The matrices'
/// work lists are gone by then, and the run gives its matrix buffers back
/// beside these (`Buffers::release`).
fn release<F: Field, M: Format, D: Module>(
    table: &mut Table<M>,
    basis: &mut Basis<F, D>,
    pairs: &mut PairSet,
    new_pairs: &mut NewPairs<M::Word>,
) {
    table.shrink_to_fit();
    basis.shrink_to_fit();
    pairs.shrink_to_fit();
    new_pairs.release();
}

/// The rows of the selected pairs: `lcm/LM(g_i)` times `g_i` and `lcm/LM(g_j)`
/// times `g_j` for each pair (i, j), each row once.
fn pair_rows<F: Field, M: Format, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    selected: &[Pair],
) -> Result<Vec<RowSource>, Size> {
    let mut rows: Vec<RowSource> = Vec::with_capacity(2 * selected.len());
    for p in selected {
        for k in [p.i, p.j] {
            rows.push(RowSource {
                element: k,
                multiplier: table.div(p.lcm, basis.lead(k))?,
            });
        }
    }
    // The pairs of one lcm that share an element share its row.
    // f4-gfp.45 design -- the pair rows stand by their element's index, so a tie for a pivot among them goes to the element made first, not to the row a pair named first (not measured)
    rows.sort_unstable_by_key(|r| (r.element, r.multiplier));
    rows.dedup();
    Ok(rows)
}

// handover-time.1 invented -- the signature run decides at its gap round once, the first round that makes an element whose signature's degree passes its leading monomial's by 4 or more, not at every such round, nor by the tests of the gap round at every such round (log 0038-1)
/// The gap of the gap round: the first round that makes an element whose
/// signature degree, its sugar degree, passes its leading monomial's
/// degree by `HANDOVER` or more.
pub(crate) const HANDOVER: u32 = 4;

// early-handover-time.1 invented -- the signature run decides at one more round, the first that makes an element whose signature's degree passes its leading monomial's by 2 or more, not at the gap round alone, nor by the gap round's tests at a gap of 2, 3, or 5 for every call (log 0054-1)
// early-handover-time.7 tuned -- the early round's gap is 2, not 3 (log 0054-1)
/// The gap of the early round: the first round that makes an element
/// whose signature degree passes its leading monomial's by `EARLY` or
/// more.
pub(crate) const EARLY: u32 = 2;

// outgrown-round-size.1 design -- the early round's tests and the outgrown test each read rounds of a size of their own, not one least for both (handover-time.6 and early-handover-time.3, withdrawn) (log 0075-1)
// outgrown-round-size.2 tuned -- the early round's tests read a round of 25 elements or more where the plain run would combine rows, not 1, 10, 50, or 100 (log 0075-1)
/// The fewest elements an early round makes for its tests to read it
/// where the plain run after a handover would combine rows
/// (outgrown-round-size.2).
pub(crate) const EARLY_ROUND: usize = 25;

// outgrown-round-size.3 tuned -- where the plain run would reduce each row alone, the early round's tests read a round of 100 elements or more, not 25 as where it combines rows, nor 50 (log 0075-1)
/// The fewest elements an early round makes for its tests to read it
/// where the plain run after a handover would reduce each row alone
/// (outgrown-round-size.3).
pub(crate) const EARLY_ROUND_ALONE: usize = 100;

// pair-work-handover.4 tuned -- the pair-work test reads a round whose update would form 1,000 S-pairs or more, not every round, nor 10,000 or more (log 0067-1)
// pair-work-ratio.2 tuned -- at the multiple of pair-work-ratio.1 the pair-work test reads a round of 1,000 S-pairs or more, as at 4, not of 300 or 100 or more (log 0080-2)
/// The fewest S-pairs a round's update would form for the pair-work test
/// to read the round (pair-work-handover.4, pair-work-ratio.2).
pub(crate) const PAIRS: u64 = 1_000;

// pair-work-ratio.1 tuned -- the pair-work test hands over where the S-pairs reach the entries subtracted, not 4 times them (pair-work-handover.3, withdrawn), 2 times, or half of them (log 0080-2)
/// How many times the entries a round's row operations subtracted the
/// S-pairs its update would form reach for the pair-work test to hand
/// over (pair-work-ratio.1).
pub(crate) const PAIR_WORK: u64 = 1;

/// A share n/d of a count, which a part of it is compared with in
/// integers: part·d against n·whole.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Share {
    pub(crate) n: usize,
    pub(crate) d: usize,
}

impl Share {
    /// Whether `part` lies below this share of `whole`.
    #[inline]
    pub(crate) fn below(self, part: usize, whole: usize) -> bool {
        part * self.d < self.n * whole
    }

    /// Whether `part` lies above this share of `whole`.
    #[inline]
    pub(crate) fn above(self, part: usize, whole: usize) -> bool {
        part * self.d > self.n * whole
    }

    /// Whether `part` reaches this share of `whole`.
    #[inline]
    pub(crate) fn reached(self, part: usize, whole: usize) -> bool {
        part * self.d >= self.n * whole
    }
}

/// The share of its elements below which the rows an early round reduced
/// to zero hand over (early-handover-time.5): a quarter.
pub(crate) const EARLY_ZEROS: Share = Share { n: 1, d: 4 };

/// The share of its elements below which the rows the gap round reduced to
/// zero hand over (handover-time.2): a half.
pub(crate) const GAP_ZEROS: Share = Share { n: 1, d: 2 };

/// The share of its elements above which the divisible elements of the
/// early round and of the gap round hand over (handover-time.4,
/// threshold-recheck.16): 6 in 10.
pub(crate) const DIVISIBLE: Share = Share { n: 6, d: 10 };

/// The share of its elements at or above which the divisible elements of
/// an outgrown round hand over (handover-time.7): 8 in 10.
pub(crate) const OUTGROWN: Share = Share { n: 8, d: 10 };

/// The S-pairs the update forms where `made` new elements enter a basis of
/// `basis` elements: each new element with every element before it,
/// made·basis + made·(made − 1)/2, `u64::MAX` where that passes it.
pub(crate) fn pairs_formed(made: usize, basis: usize) -> u64 {
    // Each product fits u128, and their sum saturates.
    let (made, basis) = (made as u128, basis as u128);
    let pairs = (made * basis).saturating_add(made * made.saturating_sub(1) / 2);
    u64::try_from(pairs).unwrap_or(u64::MAX)
}

/// The handover's state over the rounds of a signature run. The run
/// decides at its early round and at its gap round, each once, at every
/// round that made an element, and at every round whose update would form
/// `PAIRS` S-pairs or more.
pub(crate) struct Handover {
    /// Whether the gap round has been.
    decided: bool,
    /// Whether the early round has been.
    early: bool,
    /// The fewest elements the early round makes for its tests to read it.
    least: usize,
    /// What the tests read of the last round.
    tests: Tests,
}

// call-report.12 design -- every round at which a handover test reads the round gets a line with each test's verdict, not a line for the round that hands over alone (not measured)
/// What the handover's tests read of a round (`Handover::after_round`,
/// `Handover::after_pairs`): per test None where it did not read the
/// round, else whether it handed over, the early test reading every early
/// round and staying at one of fewer elements than its least without its
/// zero and share tests; the count of divisible elements
/// where a test read it, and the S-pairs the round's update would form
/// where the pair-work test read them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tests {
    pub(crate) early: Option<bool>,
    pub(crate) gap: Option<bool>,
    pub(crate) outgrown: Option<bool>,
    pub(crate) pair_work: Option<bool>,
    pub(crate) divisible: Option<usize>,
    pub(crate) pairs: Option<u64>,
}

impl Handover {
    /// No round yet, for a run whose plain run after a handover combines
    /// rows where `combines` (`plain_combines`): its early round's tests
    /// read a round of `EARLY_ROUND` elements or more where `combines`,
    /// and of `EARLY_ROUND_ALONE` or more where not.
    pub(crate) fn new(combines: bool) -> Handover {
        Handover {
            decided: false,
            early: false,
            least: if combines {
                EARLY_ROUND
            } else {
                EARLY_ROUND_ALONE
            },
            tests: Tests::default(),
        }
    }

    /// What the tests read of the round `after_round` last ruled on: the
    /// early test reads the early round, the gap test the gap round where
    /// the early test did not hand over, and the outgrown test a round
    /// that made an element where neither handed over. The pair-work test,
    /// `after_pairs`, reads a round of `PAIRS` S-pairs or more where none
    /// of them handed over.
    pub(crate) fn tests(&self) -> Tests {
        self.tests
    }

    /// Whether the run hands over after a round that made `made`
    /// elements and reduced `zeros` rows to zero, an element of a gap of
    /// `EARLY` or more among them where `early`, and of the run's handover
    /// gap or more where `gap`. An element's gap is its signature degree less its
    /// leading monomial's degree. The first round with `early` is the
    /// early round, and the first with `gap` the gap round. Their tests
    /// run in that order, the early round's first where one round is both,
    /// and then the outgrown test, where the round made an element. The
    /// early round's tests read a round of the `Handover`'s least or more
    /// elements. `divisible` counts the elements whose
    /// leading monomial a leading monomial of the basis divides, and is
    /// called only where a test reads it, at most once.
    pub(crate) fn after_round(
        &mut self,
        early: bool,
        gap: bool,
        made: usize,
        zeros: usize,
        divisible: impl FnOnce() -> usize,
    ) -> bool {
        let count = LazyCell::new(divisible);
        let read = || *count;
        let mut tests = Tests::default();
        let handed = 'tests: {
            // early-handover-time.2 design -- the early round decides once, not every round of a gap of 2 or 3 before the gap round (log 0054-1)
            if early && !self.early {
                self.early = true;
                // input-redundancy-time.7 design -- the early round's tests read a round that reduced no row to zero, as one that reduced some, not only a round that reduced a row to zero (early-handover-time.4) (log 0068-1)
                // early-handover-time.5 tuned -- the early round hands over where it reduced fewer rows to zero than a quarter of the elements it made, not half of them (log 0054-1)
                // early-handover-time.6 cited: roune-2012 Appendix B.4 -- stopping on detecting a Gröbner basis is only a benefit when the signature Gröbner basis is much larger than the minimal Gröbner basis, and if that is known ahead of time, SB should probably not be used in the first place
                // threshold-recheck.16 design -- the early round hands over where more than 6 in 10 of its elements are divisible, the gap round's share (handover-time.4), not 5 or 7 in 10; neither moved the time past the margin on the tuning set (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0054-1, 0074-2)
                let hands = made >= self.least
                    && (EARLY_ZEROS.below(zeros, made) || DIVISIBLE.above(read(), made));
                tests.early = Some(hands);
                if hands {
                    break 'tests true;
                }
            }
            if gap && !self.decided {
                self.decided = true;
                // handover-time.2 invented -- the gap round hands over where it reduced fewer rows to zero than half the elements it made, not by the share of divisible elements alone, nor by the elements made against the round before; eder-2013a §5 finds the signature criteria discard more pairs than Gebauer and Möller's on inhomogeneous input and the reduction suffer, and is silent on the rows a round reduces to zero (log 0038-1)
                // handover-time.3 cited: roune-2012 Appendix B.4 -- stopping on detecting a Gröbner basis is only a benefit when the signature Gröbner basis is much larger than the minimal Gröbner basis, and if that is known ahead of time, SB should probably not be used in the first place
                // handover-time.4 tuned -- most of the gap round's elements is more than 6 in 10, not 5 or 7 in 10 (log 0038-1)
                let hands = GAP_ZEROS.below(zeros, made) || DIVISIBLE.above(read(), made);
                tests.gap = Some(hands);
                if hands {
                    break 'tests true;
                }
            }
            // handover-time.5 cited: roune-2012 Appendix B.4 -- stopping on detecting a Gröbner basis is only a benefit when the signature Gröbner basis is much larger than the minimal Gröbner basis, and if that is known ahead of time, SB should probably not be used in the first place
            // handover-time.7 tuned -- an outgrown round hands over where 8 in 10 or more of its elements are divisible, not 7 or 9 in 10 (log 0038-1)
            // outgrown-round-size.4 tuned -- the outgrown test reads every round that made an element, not a round of 100 (handover-time.6, withdrawn), 50, 25, 10, or 5 elements or more (log 0075-1)
            // A round that made no element has no share to read.
            let outgrown = made > 0;
            let hands = outgrown && OUTGROWN.reached(read(), made);
            tests.outgrown = outgrown.then_some(hands);
            hands
        };
        tests.divisible = LazyCell::get(&count).copied();
        self.tests = tests;
        handed
    }

    // pair-work-handover.1 invented -- a round hands over where the S-pairs its update would form reach a multiple of the entries its row operations subtracted, not by the S-pairs alone, nor against the entries of its matrix (log 0067-1)
    // pair-work-handover.2 design -- the pair-work test reads every round of `PAIRS` S-pairs or more, from the first on, not once at the first round it reads, as the early and the gap round decide (log 0067-2)
    // pair-work-handover.6 design -- the pair-work test runs after the early, the gap, and the outgrown tests, where none of them handed over, not before them (not measured)
    /// Whether the run hands over after a round that `after_round` kept
    /// the run after, called once for that round: the round's update would
    /// form `pairs` S-pairs, and its row operations subtracted `subtracted`
    /// entries. The pair-work test reads a round of `PAIRS` S-pairs or
    /// more, and hands over where they reach `PAIR_WORK` times the entries
    /// subtracted.
    pub(crate) fn after_pairs(&mut self, pairs: u64, subtracted: u64) -> bool {
        if pairs < PAIRS {
            return false;
        }
        let hands = u128::from(pairs) >= u128::from(PAIR_WORK) * u128::from(subtracted);
        self.tests.pair_work = Some(hands);
        self.tests.pairs = Some(pairs);
        hands
    }
}

// signature-pruning.1 cited: lairez-2022 §5.7, Algorithm 8 and Theorem 62 -- a batch of signatures is reduced at a time, modulo AG^{<σ} and the batch's new elements; ars-2005 §1.4.f, independently, via albrecht-2010 §1 and §5 -- a version of F5 close to the form of F4, whose computations are done on sub-matrices
/// As `run`, with the pairs pruned by signatures (plan.md of
/// signature-pruning, _A round_): every element carries a signature, the
/// pending work is a queue of signatures, and each round reduces the
/// reductants of the pending signatures of least degree in one matrix.
/// When the queue is empty, `G_min` is inter-reduced as in `run`. The order
/// is graded. Where `handover` is Some(g), after each round that makes no
/// constant a `Handover` decides. Its early round is the first that makes
/// an element whose signature's degree passes its leading monomial's
/// degree by `EARLY` or more, and its gap round the first by g or more. Its
/// early round's tests read a round of `EARLY_ROUND` elements or more where
/// the plain run under these hooks at the field's prime would combine rows
/// (`plain_combines`), and of `EARLY_ROUND_ALONE` or more where not. Its
/// outgrown test reads every round that made an element. Its
/// pair-work test reads the S-pairs the round's elements would form as they
/// enter against the entries the round's row operations subtracted. On
/// `Halt::Handover` the round is counted and told to the trace and the
/// learner, and none of its elements entered. `pairs` says whether
/// the run starts out eliminating S-pairs by base divisors, which a memory
/// release ends (signature-pair-time.8), and counts what became of every
/// S-pair. The budget is checked where `run` checks it, and where the
/// tick of an element entered reaches a full check. The rounds'
/// matrices take their buffers in turn from the
/// run's `matrix::Buffers`, which go back where the rounds end. Where
/// neither the trace nor the learner listens, no cofactors are tracked,
/// the field has one lane, and p is above 3 (`matrix::tests_rows`),
/// a round's rows that reduce to zero are ruled zero by random
/// combinations where they stand together (`matrix::reduce_sig`), whose
/// coefficients `seed` draws. The call report gets its lines as in `run`,
/// and a line for each round a test of the `Handover` read.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and generators of the run, the cofactors, trace, budget, counts, and learner it borrows, the handover, the S-pairs' fates, and the seed, none of which belongs with another"
)]
pub(crate) fn run_sig<F, M, O, D, C, T, L>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    cof: &mut C,
    trace: &mut T,
    budget: &mut Budget,
    stats: &mut Stats,
    learner: &mut L,
    handover: Option<u32>,
    pairs: &mut Pairs,
    seed: u64,
) -> Result<Elements<F, D, C::Element>, Halt>
where
    F: Field,
    M: Format,
    O: Order<M>,
    D: Module,
    C: Cofactors<F, M>,
    T: Trace,
    L: Learner,
{
    debug_assert!(O::GRADED, "the signature run takes a graded order");
    budget.check(|| {})?;
    learner.inputs(|| trace::inputs(table, order, &generators, false, false));
    // learn-zero-time.3 design -- a learn's signature run keeps no test, signature-zero-time.10 standing, while its plain run takes them (`run`), not the tests in the pair learn's signature run as well (log 0063-1)
    // signature-zero-time.11 design -- the signature run's coefficients come from a Combine seeded by the call's combine draw, the plain run's, which after a handover builds its own from the same draw, not a draw of its own from the call's generator (not measured)
    let mut combine = combines::<F, M, C, T, L>().then(|| Combine::new(seed));
    let mut inputs = monic_inputs(field, generators, cof)?;
    if let Some(one) = constant_input(&mut inputs, table, budget) {
        return Ok(one);
    }
    let mut run = SigRun {
        basis: Basis::new(),
        sigs: Signatures::new(inputs.len()),
        queue: Queue::new(inputs.len()),
        triangle: pairs.bases.then(Triangle::new),
        bufs: Buffers::new(),
    };
    // signature-pruning.5 cited: eder-2017 §7.2, Algorithm RB, line 3 -- the Koszul syzygies of the inputs enter H at the start; roune-2012 §3.3 -- with relatively prime lead terms the S-pair's signature is the Koszul syzygy's, the relatively prime criterion
    // Inputs i < j have the Koszul signature lm(f_i)·lm(f_j)·e_j. One the
    // format cannot hold divides no signature it holds.
    let n: u32 = inputs.len().narrow();
    for j in 0..n {
        for i in 0..j {
            let m = match table.mul(inputs[i as usize].0.lead(), inputs[j as usize].0.lead()) {
                Ok(m) => m,
                Err(Limit::Overflow) => continue,
                Err(Limit::Size) => return Err(Halt::Size),
            };
            let sig = Sig {
                monomial: m,
                index: j,
            };
            run.sigs.push_syzygy(table, sig);
        }
    }
    // signature-pruning.17 design -- the inputs keep the instance's order as their indices (not measured)
    for (i, (g, element)) in inputs.into_iter().enumerate() {
        let sig = Sig {
            monomial: g.lead(),
            index: i.narrow(),
        };
        run.enter(table, order, cof, budget, pairs, g, element, sig)?;
    }

    let mut round = 0;
    // outgrown-round-size.5 design -- the early round's least is fixed when the run's `Handover` is made, from the plain run's mode under the run's hooks and its prime, not read at every round's decision, nor from the hooks alone without the prime (not measured)
    let mut decision = Handover::new(plain_combines::<F, M, C, T, L>(field.characteristic()));
    // The pending signatures taken since the last round, and those of them
    // that gave no reductant.
    let mut taken = (0, 0);
    while !run.queue.is_empty() {
        budget.check(|| run.release(table))?;
        budget.open();
        let (given, popped) = run.reductants(table, order)?;
        taken = (taken.0 + popped, taken.1 + popped - given.len());
        if given.is_empty() {
            continue;
        }
        budget.mark(Stage::Select);
        round += 1;
        let (degree, reductants) = (given[0].1.degree(table), given.len());
        let (m, sigs) = matrix::build_sig(
            table,
            &run.basis,
            &run.sigs,
            order,
            round,
            given,
            run.bufs.take_cols(),
            budget,
        )?;
        trace.matrix(&m.rows);
        // call-report.22 design -- the trace's hook of the matrix before its reduction counts into the report's record stage, not into reduce (not measured)
        budget.mark(Stage::Record);
        cof.matrix(m.rows.len());
        // pair-work-handover.5 design -- the round's work in its reduction is the entries its row operations subtracted, the budget's count across the reduction, not with the entries added into combinations beside them, nor a count of its own in the reduction's result (log 0067-1)
        let before = budget.counts().0;
        let reduction = matrix::reduce_sig(
            field,
            table,
            order,
            &run.basis,
            &m,
            &sigs,
            cof,
            budget,
            &mut run.bufs,
            combine.as_mut(),
        )?;
        // The budget's count only grows.
        let subtracted = budget.counts().0.saturating_sub(before);
        budget.mark(Stage::Reduce);
        count(stats, &reduction);
        budget.counted(|| {
            let bands = matrix::takes_bands::<F, M, C>(&m);
            Round {
                pending: Some(taken),
                ..figures(round, degree, reductants, &m, &reduction, false, bands)
            }
        });
        taken = (0, 0);
        learner.step(|| trace::step(table, order, &m, &reduction, false));
        let mut counted: Vec<RowId> = reduction.made.iter().map(|r| r.row).collect();
        counted.sort_unstable();
        trace.counted(&counted);
        budget.mark(Stage::Record);
        // gfp-unsolved.14 cited: roune-2012 Appendix B.1 -- an S-pair is eliminated early, when made, or late, right before its reduction; trying to eliminate them all every time new information comes in would take a lot of time
        // A pending signature a known syzygy signature divides is dropped
        // when a round takes it (`SigRun::reductants`).
        for &r in &reduction.zero {
            run.sigs.push_syzygy(table, sigs[r.index()]);
        }
        // f4-gfp.22 design -- a constant element ends the run with the basis {1}, not running until no pair is left (not measured)
        let mut made = reduction.made;
        if let Some(one) = constant_made(&mut made, &m, table, cof, budget)? {
            budget.end_round(None);
            report_constant(budget, round);
            return Ok(one);
        }
        // The handover's tests that read the round, with the elements made,
        // the rows reduced to zero, and the widest gap.
        let mut read = None;
        // handover-time.1 invented -- the signature run decides at its gap round once, the first round that makes an element whose signature's degree passes its leading monomial's by 4 or more, not at every such round, nor by the tests of the gap round at every such round (log 0038-1)
        if let Some(g) = handover {
            let lead = |r: &matrix::Reduced<F>| m.monomials[r.cols[0].index()];
            // The round's widest gap, 0 where it made no element.
            let widest = made
                .iter()
                .map(|r| {
                    let sig = sigs[r.row.index()].degree(table);
                    sig.saturating_sub(table.degree(lead(r)))
                })
                .max()
                .unwrap_or(0);
            // early-handover-time.1 invented -- the signature run decides at one more round, the first that makes an element whose signature's degree passes its leading monomial's by 2 or more, not at the gap round alone, nor by the gap round's tests at a gap of 2, 3, or 5 for every call (log 0054-1)
            let early = widest >= EARLY;
            let gap = widest >= g;
            // handover-time.8 design -- an element is counted divisible by the signature run's reducer search over the basis before the round's elements enter, and only where a test reads the count, not with the round's own elements among the divisors, nor by the plain run's search over G_min built for the test (not measured)
            let divisible = || {
                made.iter()
                    .filter(|r| run.sigs.reducer(table, lead(r)).is_some())
                    .count()
            };
            let zeros = reduction.zero.len();
            let handed = decision.after_round(early, gap, made.len(), zeros, divisible)
                || decision.after_pairs(pairs_formed(made.len(), run.basis.len()), subtracted);
            read = Some((made.len(), zeros, widest));
            if handed {
                budget.mark(Stage::Decode);
                budget.end_round(None);
                report_handover(budget, round, &decision, read, true);
                // round-memory-time.11 design -- the plain run after a handover starts with buffers of its own, and the signature run's drop with its halt, not handed to the plain run (not measured)
                return Err(Halt::Handover);
            }
        }
        let mut new = Vec::with_capacity(made.len());
        for r in made {
            let element = cof.compose(r.kept, &m.rows, table, budget)?;
            let sig = sigs[r.row.index()];
            new.push((r.poly(&m), element, sig));
        }
        budget.mark(Stage::Decode);
        // The rows were made in increasing signature order.
        for (h, element, sig) in new {
            run.enter(table, order, cof, budget, pairs, h, element, sig)?;
        }
        budget.mark(Stage::Update);
        budget.end_round(None);
        report_handover(budget, round, &decision, read, false);
        run.bufs.keep_cols(m);
    }

    // round-memory-time.7 design -- the buffers live for one run's rounds and go back where the rounds end, before the inter-reduction, not buffers that live for the call or for the thread (log 0051-1)
    run.bufs.release();
    budget.check(|| run.release(table))?;
    reduced_basis(
        field,
        table,
        order,
        &run.basis,
        round + 1,
        cof,
        trace,
        budget,
    )
}

// trace-replay.1 cited: berthomieu-2021 §4.3, via berthomieu-2022a §3.4 (3) -- the matrices are generated from the trace's polynomials, with no need of handling critical pairs or symbolic preprocessing; traverso-1989 §Gröbner traces, via demin-2024 §2.2, independently -- critical pairs and simplifier sequences are sufficient to reconstruct the algorithm avoiding all useless critical pairs and all division tests; joux-2011 §2.1 (2), independently -- do not maintain nor update a queue of untreated pairs
/// The reduced Gröbner basis of the generators from the rows of `record`,
/// as `run` returns it: the inputs enter the basis, each step's matrix is
/// built from its rows and reduced, and `G_min` is inter-reduced as in `run`.
/// Under `Shaping::Take` the replay keeps what it builds in the `Built`:
/// the nonzero generators' monomials, every matrix with its steps' lead
/// ids, and the columns of every row each step's reduction made. Under
/// `Shaping::Use` the matrices are the `Built`'s, whose table is `table`,
/// and the replay builds none: None at the first input, or row a step
/// made, whose terms differ from the shape's. The budget is checked at the
/// start, at the start of every step, and before the inter-reduction, and
/// in between as in `run`.
/// `Halt::Misfit` where the record does not fit. The steps' matrices take
/// their buffers in turn from one `matrix::Buffers`, which goes back where
/// the steps end, before the inter-reduction. The call report gets its
/// lines as in `run`, a round's per step.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, generators, and record of the replay, the cofactors, trace, budget, and counts it borrows, and its shaping, none of which belongs with another"
)]
pub(crate) fn replay<F, M, O, D, C, T>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    generators: Vec<Poly<F, D>>,
    record: &Record,
    cof: &mut C,
    trace: &mut T,
    budget: &mut Budget,
    stats: &mut Stats,
    shaping: Shaping<'_>,
) -> Result<Option<Elements<F, D, C::Element>>, Halt>
where
    F: Field,
    M: Format,
    O: Order<M>,
    D: Module,
    C: Cofactors<F, M>,
    T: Trace,
{
    let (shape, mut keep) = match shaping {
        Shaping::Off => (None, None),
        Shaping::Take(built) => (None, Some(built)),
        Shaping::Use(built) => (Some(built), None),
    };
    budget.check(|| {})?;
    let n = table.format().variables();
    // trace-replay.6 design -- a replay checks that the record fits, the variables, the generators, which are zero, and their leading monomials, and stops at the first failure, not the leading monomials of the output compared with the learn run's after the replay (not measured)
    // q-later-runs-time.4 design -- a pair fits only where every leading coefficient of the generators it is given, the pivots of their echelon form where the record took it, is a unit at both primes, and where it does not each prime replays alone, not the pair a misfit at both primes nor the lane that fits taken from the pair's run (not measured)
    // run-generators-time.13 design -- the replay checks each generator's leading monomial and coefficient against the record where `echelon::replay` gave the generators from the record too, not the check left out there (not measured)
    let fits = record.variables == n
        && record.inputs.len() == generators.len()
        && generators
            .iter()
            .zip(&record.inputs)
            .all(|(g, lead)| match lead {
                None => g.is_zero(),
                Some(e) => {
                    !g.is_zero()
                        && field.is_unit(g.coefs[0])
                        && (0..n).all(|i| table.exponent(g.lead(), i) == e[i])
                }
            });
    if !fits {
        return Err(Halt::Misfit(0));
    }
    let terms = generators
        .iter()
        .filter(|g| !g.is_zero())
        .map(|g| &g.monomials);
    if let Some(built) = shape {
        // q-replay-time.4 design -- a replay from the shape checks every element's terms against it, each nonzero generator's monomials and the columns of each row each step's reduction makes, and stops at the first that differs, not a shape that holds the union of the terms seen with explicit zero entries and no check (not measured)
        if !terms.eq(&built.inputs) {
            return Ok(None);
        }
    } else if let Some(built) = keep.as_deref_mut() {
        built.inputs = terms.cloned().collect();
    }
    let mut inputs = monic_inputs(field, generators, cof)?;
    if let Some(one) = constant_input(&mut inputs, table, budget) {
        return Ok(Some(one));
    }
    // trace-replay.12 design -- the record holds the order in which the inputs and each step's new elements entered the basis, not a replay that sorts them as the learn run's engine does (not measured)
    // The inputs are the nonzero generators, in the instance's order.
    let nonzero: Vec<u32> = (0..record.inputs.len().narrow())
        .filter(|&j| record.inputs[j as usize].is_some())
        .collect();
    let mut inputs: Vec<Option<_>> = inputs.into_iter().map(Some).collect();
    let mut basis: Basis<F, D> = Basis::new();
    for j in &record.entered {
        let Ok(k) = nonzero.binary_search(j) else {
            return Err(Halt::Misfit(0));
        };
        let Some((g, element)) = inputs.get_mut(k).and_then(Option::take) else {
            return Err(Halt::Misfit(0));
        };
        basis.push_min(table, g)?;
        cof.push(element);
    }

    let mut bufs: Buffers<F> = Buffers::new();
    let mut round = 0;
    for (s, step) in record.steps.iter().enumerate() {
        budget.check(|| {
            table.shrink_to_fit();
            basis.shrink_to_fit();
            bufs.release();
        })?;
        budget.open();
        round += 1;
        // A shape ends before this step only where its replay made a
        // constant, and a replay with its terms makes the same constant: the
        // first arm is not reached.
        let taken = match shape.map(|built| built.steps.get(s)) {
            Some(None) => return Ok(None),
            Some(Some(taken)) => Some(taken),
            None => None,
        };
        let mut owned = None;
        // q-replay-time.1 cited: berthomieu-2022a §3.4, Tracer over Q -- the computations after those that learn apply the learned data, with no critical pairs and no symbolic preprocessing; parisse-2013 §3 -- the symbolic preprocessing can reuse for the next primes the collection of monomials used for the first; demin-2024 §4.4 -- the apply stage caches the order of rows and columns in the matrices across the modular runs
        let (m, leads) = if let Some(taken) = taken {
            (&taken.matrix, &taken.leads[..])
        } else {
            let cols = bufs.take_cols();
            let (m, leads) =
                owned.insert(build_step(table, &basis, order, round, step, cols, budget)?);
            (&*m, &leads[..])
        };
        trace.matrix(&m.rows);
        // call-report.22 design -- the trace's hook of the matrix before its reduction counts into the report's record stage, not into reduce (not measured)
        budget.mark(Stage::Record);
        cof.matrix(m.rows.len());
        // gfp-unsolved.11 design -- a record says whether its run inter-reduced the new rows, and a replay inter-reduces where it says so, not a replay that inter-reduces where the order is not graded (not measured)
        let reduced = matrix::replay(
            field,
            &basis,
            m,
            &step.reductees,
            leads,
            cof,
            budget,
            record.inter_reduced,
            &mut bufs,
        )?;
        budget.mark(Stage::Reduce);
        let Some(mut made) = reduced else {
            return Err(Halt::Misfit(round));
        };
        // q-replay-time.4 design -- a replay from the shape checks every element's terms against it, each nonzero generator's monomials and the columns of each row each step's reduction makes, and stops at the first that differs, not a shape that holds the union of the terms seen with explicit zero entries and no check (not measured)
        if let Some(taken) = taken
            && !same_columns(&made, &taken.made)
        {
            return Ok(None);
        }
        let columns = if keep.is_some() {
            made.iter().map(|r| r.cols.clone()).collect()
        } else {
            Vec::new()
        };
        stats.rounds += 1;
        stats.reduced += made.len() as u64;
        budget.counted(|| Round {
            number: round,
            degree: m.monomials.first().map_or(0, |&u| table.degree(u)),
            pairs: step.reductees.len(),
            rows: m.rows.len(),
            given: m.given,
            columns: m.columns(),
            entries: m.entries(),
            pivots: made.len(),
            zero: 0,
            zero_rows: Some(0),
            reductions: made.len() as u64,
            bands: false,
            pending: None,
            left: None,
        });
        let mut counted = step.reductees.clone();
        counted.sort_unstable();
        trace.counted(&counted);
        budget.mark(Stage::Record);
        // f4-gfp.22 design -- a constant element ends the run with the basis {1}, not running until no pair is left (not measured)
        if let Some(one) = constant_made(&mut made, m, table, cof, budget)? {
            keep_step(keep.as_deref_mut(), owned, columns);
            budget.end_round(None);
            report_constant(budget, round);
            return Ok(Some(one));
        }
        let mut new = Vec::with_capacity(made.len());
        for r in made {
            let element = cof.compose(r.kept, &m.rows, table, budget)?;
            new.push(Some((r.poly(m), element)));
        }
        budget.mark(Stage::Decode);
        if let Some(m) = keep_step(keep.as_deref_mut(), owned, columns) {
            bufs.keep_cols(m);
        }
        for &k in &step.entered {
            let Some((h, element)) = new.get_mut(k as usize).and_then(Option::take) else {
                return Err(Halt::Misfit(round));
            };
            basis.push_min(table, h)?;
            cof.push(element);
        }
        budget.mark(Stage::Update);
        budget.end_round(None);
    }

    // round-memory-time.7 design -- the buffers live for one run's rounds and go back where the rounds end, before the inter-reduction, not buffers that live for the call or for the thread (log 0051-1)
    drop(bufs);
    budget.check(|| {
        table.shrink_to_fit();
        basis.shrink_to_fit();
    })?;
    budget.open();
    // trace-replay.8 cited: joux-2011 §2.2, Alg. 6, line 22 -- F4Remake returns InterReduce(G_min), computed in the replay
    // A shape has no inter-reduction only where its replay made a constant,
    // and a replay with its terms makes the same constant: the first arm is
    // not reached.
    let inter = match shape.map(|built| built.inter.as_ref()) {
        Some(None) => return Ok(None),
        Some(Some(inter)) => Some(inter),
        None => None,
    };
    let mut owned = None;
    let m = match inter {
        // q-replay-time.11 cited: berthomieu-2022a §3.4, Tracer over Q -- the computations that apply the learned data need no symbolic preprocessing
        Some(matrix) => matrix,
        None => &*owned.insert(inter_matrix(table, order, &basis, round + 1, budget)?),
    };
    let made = inter_reduce(field, &basis, m, cof, trace, budget)?;
    budget.mark(Stage::Reduce);
    // q-replay-time.13 design -- a replay from the shape does not check the rows its inter-reduction makes against the shape, not a check of those rows that starts the prime over in full where one differs (log 0022-2)
    let reduced = reduced_elements(table, order, m, made, cof, budget)?;
    budget.mark(Stage::Decode);
    report_inter(budget, m, reduced.len());
    if let Some(built) = keep
        && let Some(mut matrix) = owned
    {
        // q-replay-time.14 design -- a kept matrix gives back the slack of its arrays when it is kept, and the shape's table when the shape is put, not the shape kept as built (log 0022-2)
        matrix.shrink_to_fit();
        built.inter = Some(matrix);
    }
    Ok(Some(reduced))
}

/// The matrix of `step` in round `round`, its rows multiplied out from the
/// record (`matrix::build_rows`) into `cols`, and the ids of the monomials
/// its counted rows must reduce to. Overflow where a product passes the
/// format.
fn build_step<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    round: u32,
    step: &Step,
    cols: Vec<MonoId>,
    budget: &mut Budget,
) -> Result<(Matrix, Vec<MonoId>), Halt> {
    let n = table.format().variables();
    let mut rows = Vec::with_capacity(step.elements.len());
    for (&element, e) in step.elements.iter().zip(step.multipliers.chunks_exact(n)) {
        rows.push(RowSource {
            element,
            multiplier: table.intern(e)?,
        });
    }
    let m = matrix::build_rows(table, basis, order, round, rows, cols, budget)?;
    let mut leads = Vec::with_capacity(step.reductees.len());
    for e in step.leads.chunks_exact(n) {
        leads.push(table.intern(e)?);
    }
    Ok((m, leads))
}

/// Whether the rows `made` have the columns `columns`, row by row.
fn same_columns<F: Field>(made: &[Reduced<F>], columns: &[Vec<MonoId>]) -> bool {
    made.iter().map(|r| &r.cols).eq(columns)
}

// q-replay-time.2 cited: berthomieu-2022a §3.4, Tracer over Q -- the second modular computation learns, with exact linear algebra, all polynomial data the F4 matrices need to give the nonzero information
/// Keeps the step's matrix the replay built, with its lead ids, and the
/// columns of the rows its reduction made, where `keep` is Some. The
/// matrix the replay built and does not keep.
fn keep_step(
    keep: Option<&mut Built>,
    owned: Option<(Matrix, Vec<MonoId>)>,
    made: Vec<Vec<MonoId>>,
) -> Option<Matrix> {
    match (keep, owned) {
        (Some(built), Some((mut matrix, leads))) => {
            // q-replay-time.14 design -- a kept matrix gives back the slack of its arrays when it is kept, and the shape's table when the shape is put, not the shape kept as built (log 0022-2)
            matrix.shrink_to_fit();
            built.steps.push(Taken {
                matrix,
                leads,
                made,
            });
            None
        }
        (None, owned) => owned.map(|(matrix, _)| matrix),
        (Some(_), None) => None,
    }
}

/// The state of a signature run: the basis, every element's signature with
/// the known syzygy signatures, the pending signatures, the triangle of the
/// base divisors, and the buffers of its matrices.
struct SigRun<F: Field, D: Module> {
    basis: Basis<F, D>,
    sigs: Signatures,
    queue: Queue,
    /// Held where the run eliminates S-pairs by base divisors.
    triangle: Option<Triangle>,
    /// The buffers the run's matrices take in turn.
    bufs: Buffers<F>,
}

impl<F: Field, D: Module> SigRun<F, D> {
    // f4-gfp.49 design -- inside a matrix a memory check gives back nothing of the run's own and between rounds the run shrinks its growable arrays, not the dense row dropped inside a matrix and allocated again for the next row (not measured)
    /// Gives back what the run can rebuild between two rounds, the slack of
    /// its growable arrays and its matrix buffers, and the triangle, which it
    /// can do without.
    fn release<M: Format>(&mut self, table: &mut Table<M>) {
        table.shrink_to_fit();
        self.basis.shrink_to_fit();
        self.sigs.shrink_to_fit();
        self.queue.shrink_to_fit();
        self.bufs.release();
        // signature-pair-time.8 cited: roune-2012 Appendix B.2 -- if the computer runs out of memory when using the base divisor technique, then it is possible to simply drop the triangle of bits and stop using the base divisor technique from that point onward
        self.triangle = None;
    }

    /// Enters the monic element `h` with the signature `sig` into the basis,
    /// its cofactors into the tracking, and the signatures of its S-pairs
    /// into the queue, each S-pair counted in `pairs`. Overflow where a
    /// comparison of sig-lead ratios, or a signature that is no Koszul
    /// signature and that no base divisor eliminates, passes the format;
    /// Size where the basis or the table is full.
    /// The element ticks the budget: a stop where its check stops the call.
    #[allow(
        clippy::too_many_arguments,
        reason = "the table, order, cofactors, budget, and pairs the element enters, and the element with its cofactors and signature, none of which belongs with another"
    )]
    fn enter<M: Format, O: Order<M>, C: Cofactors<F, M>>(
        &mut self,
        table: &mut Table<M>,
        order: &O,
        cof: &mut C,
        budget: &mut Budget,
        pairs: &mut Pairs,
        h: Poly<F, D>,
        element: C::Element,
        sig: Sig,
    ) -> Result<(), Halt> {
        let lead = h.lead();
        let new = self.basis.push_min(table, h)?;
        cof.push(element);
        self.sigs.push(table, order, &self.basis, sig)?;
        // signature-pruning.22 design -- every element entered ticks the budget (not measured)
        budget.tick()?;
        let divisors = self.triangle.as_mut().map(|triangle| {
            triangle.grow(new.index() + 1);
            BaseDivisors::of(table, &self.basis, &self.sigs, new)
        });
        // signature-pruning.15 cited: lairez-2022 §5.6.1, Algorithm 5, line 60 -- for h ∈ G: Q ← Q ∪ Σ(g, h) ∪ Σ(h, g)
        for g in (0..new.index()).map(ElementId::at) {
            pairs.considered += 1;
            if let (Some(divisors), Some(triangle)) = (&divisors, &mut self.triangle) {
                // signature-pair-time.3 cited: roune-2012 §3.5 and Appendix B.2, Theorems 18 and 19 -- the base divisor criterion is strictly weaker than the signature criterion, but it is faster to check; for yang1 the base divisors are a 35% performance improvement, and they eliminate 71% of the S-pairs
                let base = divisors.eliminates(table, &self.basis, &self.sigs, triangle, new, g);
                if let Some(base) = base {
                    match base {
                        Base::High => pairs.high += 1,
                        Base::Low => pairs.low += 1,
                    }
                    triangle.set(g, new);
                    continue;
                }
            }
            // signature-pair-time.11 design -- an S-pair a base divisor eliminates is not formed, so its signature passing the format's degree bound does not halt the run (not measured)
            let formed = match sig::pair(table, &self.basis, &self.sigs, g, new) {
                Ok(Some(formed)) => formed,
                Ok(None) => {
                    pairs.equal += 1;
                    continue;
                }
                // A Koszul signature the format cannot hold divides no
                // signature it holds.
                Err(Overflow) if table.coprime(self.basis.lead(g), lead) => {
                    pairs.overflow += 1;
                    continue;
                }
                Err(Overflow) => return Err(Halt::Overflow),
            };
            // signature-pair-time.6 design -- a pair's bit is set when its S-pair is made and a base divisor, the signature criterion, or coprime leading monomials eliminate it, the last where the format holds its signature; a queued S-pair keeps its bit unset (not measured)
            // signature-pruning.6 cited: eder-2017 §6.1, Lemma (syzygy criterion) -- if S is a syzygy signature and S divides T, T is one too; roune-2012 Appendix B.1 -- it is better to eliminate an S-pair early than late
            // A coprime S-pair whose signature a known syzygy signature
            // divides goes here too, and the known ones stay the same.
            if self.sigs.is_syzygy_formed(table, &formed) {
                pairs.syzygy += 1;
                if let Some(triangle) = &mut self.triangle {
                    triangle.set(g, new);
                }
                continue;
            }
            // signature-pair-time.2 design -- a formed signature is interned only where the run keeps it, queued or entered as a Koszul syzygy signature (log 0026-1)
            let s = Sig {
                monomial: table.intern_formed(formed.degree)?,
                index: formed.index,
            };
            if table.coprime(self.basis.lead(g), lead) {
                // signature-pruning.5 cited: eder-2017 §7.2, Algorithm RB, line 3 -- the Koszul syzygies of the inputs enter H at the start; roune-2012 §3.3 -- with relatively prime lead terms the S-pair's signature is the Koszul syzygy's, the relatively prime criterion
                // gfp-unsolved.14 cited: roune-2012 Appendix B.1 -- an S-pair is eliminated early, when made, or late, right before its reduction; trying to eliminate them all every time new information comes in would take a lot of time
                pairs.koszul += 1;
                if let Some(triangle) = &mut self.triangle {
                    triangle.set(g, new);
                }
                self.sigs.push_syzygy(table, s);
                continue;
            }
            pairs.queued += 1;
            self.queue.push(s);
        }
        Ok(())
    }

    /// Takes the pending signatures of least degree out of the queue: the
    /// reductant of each, in increasing order of signature, where the
    /// signature is no known syzygy signature and a multiple of smaller
    /// signature top-reduces the reductant; and how many signatures it took.
    fn reductants<M: Format, O: Order<M>>(
        &mut self,
        table: &mut Table<M>,
        order: &O,
    ) -> Result<(Vec<(RowSource, Sig)>, usize), Limit> {
        let mut given = Vec::new();
        let popped = self.queue.pop_least(table, order);
        let taken = popped.len();
        for s in popped {
            if self.sigs.is_syzygy(table, s) {
                continue;
            }
            // s is the signature of an S-pair, a multiple of one of its
            // halves' signatures.
            #[allow(
                clippy::expect_used,
                reason = "a queued signature is a multiple of an element's signature"
            )]
            let g = self
                .sigs
                .reductant(table, order, s)
                .expect("an element's signature divides a pending one");
            let a = table.div(s.monomial, self.sigs.sig(g).monomial)?;
            let lead = table.mul(a, self.basis.lead(g))?;
            #[allow(
                clippy::expect_used,
                reason = "g's own lead divides a·lead(g), and g is among the reducers"
            )]
            let r = self
                .sigs
                .reducer(table, lead)
                .expect("g's leading monomial divides its multiple's");
            // signature-pruning.9 cited: lairez-2022 §5.7, Algorithm 8, line 94 -- if f is →G-reducible then it joins the reductants
            if !self.sigs.before(r, g) {
                continue;
            }
            given.push((
                RowSource {
                    element: g,
                    multiplier: a,
                },
                s,
            ));
        }
        Ok((given, taken))
    }
}
