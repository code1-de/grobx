//! The F4 run (faugere-1999 §2): the inputs enter the basis, each round
//! reduces the matrix of the pairs of least sugar, in a graded order of
//! least degree, a fallen element's pair counted one degree past its lcm's
//! until the pair set has grown past its bound, and the elements of
//! `G_min` are inter-reduced to the reduced basis.

#[cfg(test)]
use std::cell::Cell;
use std::cell::LazyCell;
use std::cmp::Ordering;

use crate::budget::{Budget, PROBE, Site};
use crate::cofactors::Cofactors;
use crate::divisors::Divisors;
use crate::field::Field;
use crate::halt::Halt;
use crate::id::{ElementId, MonoId, RowId, RowSource};
use crate::matrix::{self, Buffers, Combine, Matrix, Modes, Reduced, Reduction};
use crate::module::Module;
use crate::monomial::{Format, Overflow};
use crate::narrow::{self, Narrow};
use crate::order::Order;
use crate::pairs::{self, NewPairs, Pair, PairSet};
use crate::poly::{Basis, Elements, Poly};
use crate::queue::Queue;
use crate::report::{Opt, Round, RoundMode, Secs, Stage};
use crate::shape::{Built, Shaping, Taken};
use crate::sig::{self, Base, BaseDivisors, Pairs, Sig, Signatures, Triangle};
use crate::sort::{RUN, sort_runs};
use crate::table::{Limit, Live, Table, monomial_units, search_units};
use crate::trace::{self, Learner, Output, Record, Step, Trace};
use crate::{Stats, Stop};

// f4-gfp.25 practice-checked: monomorphization -- the trace and the cofactor tracking are type parameters of run, and their empty implementations zero-sized types with empty methods, not a runtime flag tested at each hook
// trace-replay.10 design-checked -- the learn run tells the record through a type parameter of run and run_sig whose methods take a closure that builds what is recorded, not hooks added to the public Trace (not measured)
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
/// which goes back where the rounds end, before the inter-reduction.
/// Where the trace does not listen, the run compacts its table at a
/// round's end once the table has interned `COMPACT_GROWTH` monomials since
/// the last compaction or since the rounds began (`compact`). Where neither
/// the trace nor the learner listens and no cofactors are tracked, `G_min`
/// is inter-reduced at the start of a round where it holds more members of
/// degree 1 than at its last inter-reduction, and where the rounds reduce
/// in packed rows a `TAILS_GROWTH_PART` more (`tails`). The call report,
/// where the budget holds one, gets a line for the inputs' entry, one per
/// round `stats` counts, one per compaction, one per inter-reduction
/// between the rounds, one for the inter-reduction, and one where the run
/// makes a constant.
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
    budget.check(|| {}, Site::Run)?;
    // carried-entry-stage.3 design-checked -- the entry's step opens at the run's start, after its first check, so that it holds the run's every step before its first round, the learner's hook, the monic inputs, and their sugars beside the update, not the update's calls alone (not measured)
    budget.open();
    learner.inputs(|| trace::inputs(table, order, &generators, true, true, budget))?;
    let mut inputs = monic_inputs(field, generators, cof, budget)?;
    if let Some(one) = constant_input(&mut inputs, table, budget) {
        return Ok(one);
    }
    let mut plain = Plain::new(table);
    // An input's sugar is the largest total degree of its terms
    // (giovini-1991 §2): the round's sugar 0 lies below every degree.
    let inputs = with_sugar(table, order, inputs, 0, budget)?;
    let elements = inputs.len();
    plain.enter(table, order, cof, budget, inputs, 0)?;
    budget.mark(Stage::Enter);
    report_enter(budget, elements, plain.pairs.len() as u64);
    rounds(
        field, table, order, plain, cof, trace, budget, stats, learner, seed,
    )
}

/// The plain run's rounds from `plain`, which holds the elements entered
/// before the first round, and the inter-reduction of its `G_min`, as `run`
/// states them, the rounds numbered from 1.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and state of the run, the cofactors, trace, budget, counts, and learner it borrows, and its seed, none of which belongs with another"
)]
fn rounds<F, M, O, D, C, T, L>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    mut plain: Plain<F, M, D>,
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
    let mut combine = combines::<F, M, C, T, L>().then(|| Combine::new(seed));
    let combined = plain_combines::<F, M, C, T, L>(field.characteristic());
    // learn-zero-time.5 design-checked -- the tests' coefficients come from a Combine seeded by the run's seed, the draw of the plain run's combinations, which a learn run does not take, not a draw of their own from the call's generator (not measured)
    let mut tests = learn_tests::<F, M, C, T, L>().then(|| Combine::new(seed));
    // A run that combines rows chooses a mode per round (`matrix::reduce`).
    let mut modes = combined.then(Modes::new);
    let mut bufs: Buffers<F> = Buffers::new();
    // f4-gfp.4 cited-checked: monagan-2015 §2 -- simplifying as suggested by Faugère gave no speedup in their experiments
    let mut round = 0;
    // The table's monomials after its last compaction, or where the rounds
    // began.
    let mut compacted = table.len();
    // Where G_min was last inter-reduced (`tails`).
    let mut last = Tails::default();
    while !plain.pairs.is_empty() {
        budget.check(
            || {
                plain.release(table);
                bufs.release();
            },
            Site::Round,
        )?;
        if tails_on::<F, M, C, T, L>() {
            tails(
                field,
                table,
                order,
                &mut plain.basis,
                cof,
                &mut bufs,
                &mut last,
                round + 1,
                budget,
            )?;
        }
        budget.open();
        round += 1;
        // redundant-round-pairs.6 design-checked -- the split runs in every plain run, the call's, a learn's whose record the replays follow, one that tracks cofactors, in every order by the least sugar, not in grevlex alone nor in the call's run alone
        let selected = pairs::select(&mut plain.pairs, table, order, budget)?;
        let deferred = plain.pairs.deferred();
        let d = selected[0].sugar;
        let rows = pair_rows(table, &plain.basis, &selected, budget)?;
        budget.mark(Stage::Select);
        let m = matrix::build(table, &plain.basis, order, rows, &mut bufs, budget)?;
        trace.matrix(&m.rows);
        // call-report.22 design-checked -- the trace's hook of the matrix before its reduction counts into the report's record stage, not into reduce (not measured)
        budget.mark(Stage::Record);
        cof.matrix(m.rows.len());
        // gfp-unsolved.1 cited-checked: heisinger-2025 §4, footnote 1 -- in principle a row echelon form suffices, but the reduced row echelon form typically improves the algorithm's overall performance
        // gf2-build-time.4 design-checked -- every round projects its row work over its sugar, the first piece of a split too, not a repeat of a split alone, which took the 16 over GF(2) as far (log 0159-1, 0159-2)
        let reduction = budget.over_sugar(selected.len(), selected.len() + deferred, |budget| {
            matrix::reduce(
                field,
                &plain.basis,
                &m,
                cof,
                budget,
                true,
                combine.as_mut(),
                &mut bufs,
                tests.as_mut(),
                modes.as_mut(),
            )
        })?;
        budget.mark(Stage::Reduce);
        count(stats, &reduction);
        budget.counted(|| Round {
            deferred: Some(deferred),
            ..figures(round, d, selected.len(), &m, &reduction)
        });
        learner.step(|reach| trace::step(table, order, &m, &reduction, true, reach, budget))?;
        let mut made = reduction.made;
        // f4-gfp.52 design-checked -- the rows that counted are listed and sorted for every matrix, the trace on or off, not only where a constant of the trait says the trace listens (not measured)
        let counted = counted_rows(&made, budget)?;
        trace.counted(&counted);
        budget.mark(Stage::Record);
        // f4-gfp.22 design-checked -- a constant element ends the run with the basis {1}, not running until no pair is left (not measured)
        if let Some(one) = constant_made(&mut made, &m, table, cof, budget)? {
            learner.output(Output::Constant, budget)?;
            budget.end_round(Some(plain.pairs.len()));
            report_constant(budget, round);
            return Ok(one);
        }
        let mut new = Vec::with_capacity(made.len());
        for r in made {
            let element = cof.compose(r.kept, &m.rows, table, budget)?;
            new.push((r.poly(&m, budget)?, element));
        }
        budget.mark(Stage::Decode);
        let new = with_sugar(table, order, new, d, budget)?;
        if F::SPLITS_AGAIN && deferred > 0 {
            let fallen = new
                .iter()
                .filter(|(h, sugar, _)| pairs::fall(table, *sugar, h.lead()) > 0)
                .count();
            plain.pairs.note_piece(selected.len(), new.len(), fallen);
        }
        plain.enter(table, order, cof, budget, new, 0)?;
        budget.mark(Stage::Update);
        budget.end_round(Some(plain.pairs.len()));
        bufs.keep(m);
        // dead-monomial-peak.6 design-checked -- the run compacts only where the trace does not listen, as a trace that listens hears each row's multiplier by its id in one numbering for the run, not under every trace (not measured)
        // dead-monomial-peak.7 design-checked -- the plain run compacts, a call's, a learn's, and one that goes on from a carry, in every order, and the signature run, the replays, and the inter-reduction keep their tables, not the signature run's table compacted too, whose signatures, rules, and queues hold ids of it (not measured)
        // dead-monomial-peak.11 design-checked -- where cofactors are tracked the run compacts as without, the tracking holding its monomials in a table of its own (f4-gfp.24) and no id of the run's between two rounds, not no compaction with cofactors (not measured)
        if !T::LISTENS && table.len() >= compacted.saturating_add(compact_growth()) {
            compacted = compact(table, &mut plain.basis, &mut bufs, budget, round)?;
        }
    }

    // round-memory-time.7 design-checked -- the buffers live for one run's rounds and go back where the rounds end, before the inter-reduction, not buffers that live for the call or for the thread (log 0051-1)
    drop(bufs);
    budget.check(|| plain.release(table), Site::Run)?;
    learner.output(Output::Basis(plain.basis.min()), budget)?;
    reduced_basis(field, table, order, &plain.basis, cof, trace, budget)
}

/// The plain run's elements on their way into its basis, each with its
/// sugar and its cofactors.
pub(crate) type Sugared<F, D, E> = Vec<(Poly<F, D>, u32, E)>;

/// The plain run's state between its rounds: the basis, its pairs and the
/// update's buffers, and every element's sugar.
struct Plain<F: Field, M: Format, D: Module> {
    basis: Basis<F, D>,
    pairs: PairSet<M::Word>,
    new_pairs: NewPairs<M::Word>,
    /// Every element's sugar, by its index in the basis.
    sugar: Vec<u32>,
}

impl<F: Field, M: Format, D: Module> Plain<F, M, D> {
    /// No element yet, for a run over `table`.
    fn new(table: &Table<M>) -> Plain<F, M, D> {
        Plain {
            basis: Basis::new(),
            pairs: PairSet::new(table.format().words()),
            new_pairs: NewPairs::new(),
            sugar: Vec::new(),
        }
    }

    // f4-gfp.2 cited-checked: gebauer-1988 §4.6, via joux-2011 §2.2 Algorithm 2 -- with the inputs ordered by decreasing head term, an element is redundant exactly when a later one's head term divides its own
    // gf2-update-time.3 design-checked -- every `enter` takes the field's order, the inputs', the carried elements', and each round's, not each round's elements alone, with the inputs and the carried elements by decreasing leading monomial over GF(2) too (log 0130-1, 0130-2)
    /// Enters the monic elements, each with its sugar, into the basis
    /// through the update, in `pairs::entering`'s order, stably: by
    /// decreasing leading monomial, and over GF(2) by increasing. Their
    /// cofactors go into the tracking. The update leaves out the pairs whose
    /// sugar is at most `covered`, and spends `budget` (`pairs::update`);
    /// the sort spends at `Site::Enter` as `sort_by_places` spends.
    fn enter<O: Order<M>, C: Cofactors<F, M>>(
        &mut self,
        table: &Table<M>,
        order: &O,
        cof: &mut C,
        budget: &mut Budget,
        elements: Sugared<F, D, C::Element>,
        covered: u32,
    ) -> Result<(), Halt> {
        // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
        let elements = sort_by_places(
            elements,
            |a, b| pairs::entering::<F, _, _>(table, order, a.0.lead(), b.0.lead()),
            budget,
            Site::Enter,
        )?;
        for (h, sugar, element) in elements {
            self.sugar.push(sugar);
            pairs::update(
                table,
                order,
                &mut self.basis,
                &mut self.pairs,
                &self.sugar,
                h,
                &mut self.new_pairs,
                covered,
                budget,
            )?;
            cof.push(element);
        }
        Ok(())
    }

    // f4-gfp.49 design-checked -- inside a matrix a memory check gives back nothing of the run's own and between rounds the run shrinks its growable arrays, not the dense row dropped inside a matrix and allocated again for the next row (not measured)
    /// Gives back what the run can rebuild between two rounds: the slack of
    /// its table, its basis, and its pairs, and the update's buffers. The
    /// matrices' work lists are gone by then, and the run gives its matrix
    /// buffers back beside these (`Buffers::release`).
    fn release(&mut self, table: &mut Table<M>) {
        table.shrink_to_fit();
        self.basis.shrink_to_fit();
        self.pairs.shrink_to_fit();
        self.new_pairs.release();
    }
}

/// `elements` made by a round of least sugar `d`, each with its sugar
/// (`sugar_of`); `d` is 0 for the inputs. Each element spends `PROBE` units
/// a degree it reads from the table, its lead's, or in an order without a
/// grading its terms', at `Site::Sugar`; a stop where the budget's check
/// stops the call.
fn with_sugar<F: Field, M: Format, O: Order<M>, D: Module, E>(
    table: &Table<M>,
    order: &O,
    elements: Elements<F, D, E>,
    d: u32,
    budget: &mut Budget,
) -> Result<Sugared<F, D, E>, Stop> {
    // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
    let mut sugared = Vec::with_capacity(elements.len());
    for (h, element) in elements {
        let read = if O::GRADED { 1 } else { h.len() };
        budget.spend(read * PROBE, Site::Sugar)?;
        let sugar = sugar_of(table, order, &h, d);
        sugared.push((h, sugar, element));
    }
    Ok(sugared)
}

// order-family.9 design-checked -- a new element's sugar is the larger of the round's least sugar d and its own largest total degree, not the largest sugar of the matrix's rows (not measured)
// order-family.10 cited-checked: bigatti-2011 §3, the proposition on the elementary properties of the sugar, (1) -- sugar(U) is componentwise greater than or equal to the degree of U
/// The sugar of the element `h` a round of least sugar `d` made: the
/// larger of `d` and the largest total degree of its terms, in a graded
/// order its leading monomial's.
fn sugar_of<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &Table<M>,
    _: &O,
    h: &Poly<F, D>,
    d: u32,
) -> u32 {
    let top = if O::GRADED {
        // In a graded order the leading monomial has the largest total
        // degree.
        Some(table.degree(h.lead()))
    } else {
        h.monomials.iter().map(|&u| table.degree(u)).max()
    };
    d.max(top.unwrap_or(0))
}

/// The signature run's work at a handover that the plain run goes on
/// from (`run_carried`).
pub(crate) struct Carried<F: Field, D: Module, E> {
    /// The signature run's elements in the order they entered, then the
    /// handover round's elements, each monic and no constant, with its
    /// sugar and its cofactors.
    pub(crate) elements: Sugared<F, D, E>,
    /// The sugar at and below which a pair of two carried elements has a
    /// standard representation (`carry_bound`).
    pub(crate) bound: u32,
    /// The handover round's number, for the call report.
    pub(crate) round: u32,
    /// How many of the elements the handover round made, for the call
    /// report.
    pub(crate) made: usize,
    /// The entries the signature run subtracted, for the call report.
    pub(crate) work: u64,
}

/// The reduced Gröbner basis from a carry, as `run` returns it from the
/// generators: the carried elements enter at their sugars, the update
/// leaving out the pairs whose sugar is at most the carry's bound, and the
/// rounds run from them as from an input's elements (`rounds`). The carry
/// runs where neither the trace nor the learner listens and no cofactors
/// are tracked. The carried elements' monomials are ids of `table`, a
/// fresh table that holds them alone (`carry_table`). The call
/// report, where the budget holds one, gets a line of the carry and one of
/// the entry before the rounds'.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and carry of the run, the cofactors, trace, budget, counts, and learner it borrows, and its seed, none of which belongs with another"
)]
pub(crate) fn run_carried<F, M, O, D, C, T, L>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    carried: Carried<F, D, C::Element>,
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
    debug_assert!(
        combines::<F, M, C, T, L>(),
        "a carry runs where neither the trace nor the learner listens and no cofactors are tracked"
    );
    debug_assert!(
        carried.elements.iter().all(|(h, sugar, _)| {
            !h.is_zero()
                && table.degree(h.lead()) > 0
                && *sugar >= table.degree(h.lead())
                && h.coefs[0] == field.coef(1)
        }),
        "every carried element monic and no constant, its sugar at least its leading monomial's degree"
    );
    budget.check(|| {}, Site::Run)?;
    report_carry(budget, &carried, table.len());
    budget.open();
    let mut plain = Plain::new(table);
    let elements = carried.elements.len();
    // signature-run-losses.7 design-checked -- the carried elements enter as an input's elements enter, in `pairs::entering`'s order, all before the first round, not by increasing sugar, ties by decreasing leading monomial (log 0112-1)
    plain.enter(table, order, cof, budget, carried.elements, carried.bound)?;
    budget.mark(Stage::Enter);
    report_enter(budget, elements, plain.pairs.len() as u64);
    rounds(
        field, table, order, plain, cof, trace, budget, stats, learner, seed,
    )
}

// carried-entry-stage.1 design-checked -- a run's entry gets a line of its own after its elements entered and before its first round, not a field of the `carry` line, which is written before the entry, nor time added to the first round's stages (not measured)
// carried-entry-stage.2 design-checked -- the line stands for the plain run's entry of its inputs and of its carried elements and for the signature run's entry of its inputs, not for the carried elements alone, and a replay, which enters its inputs with no pairs, writes none (log 0182-1)
// carried-entry-stage.5 design-checked -- the line names the elements entered and the pairs they left, the plain run's pair set or the S-pairs the signature run queued, beside the step's time, not the time alone (not measured)
// carried-entry-stage.6 design-checked -- a run that ends before its entry ends, by a constant input or a halt, writes no entry line, its time in its `ran` line's, not a line of the entry so far (not measured)
/// The call report's line of a run's entry of `elements` elements, which
/// left `pairs` pairs: the plain run's pair set, or the S-pairs the
/// signature run queued. Its time is the open step's `Stage::Enter`.
fn report_enter(budget: &mut Budget, elements: usize, pairs: u64) {
    let Some(report) = budget.report() else {
        return;
    };
    let time = Secs(report.time(Stage::Enter));
    report.line(
        "enter",
        format_args!(" elements={elements} pairs={pairs} time={time}"),
    );
}

/// The call report's line of the carry `carried` over a table of
/// `monomials` monomials.
fn report_carry<F: Field, D: Module, E>(
    budget: &mut Budget,
    carried: &Carried<F, D, E>,
    monomials: usize,
) {
    let Some(report) = budget.report() else {
        return;
    };
    report.line(
        "carry",
        format_args!(
            " round={round} elements={elements} made={made} bound={bound} work={work} \
             monomials={monomials}",
            round = carried.round,
            elements = carried.elements.len(),
            made = carried.made,
            bound = carried.bound,
            work = carried.work,
        ),
    );
}

// signature-run-losses.5 design-checked -- the plain run takes a fresh table that holds the carried elements' monomials alone, not the signature run's table kept, whose matrices' monomials the plain run's lookups and peak carry (log 0112-1)
// gf2-bound-losses.5 practice-checked: memoization -- each monomial of `table` is interned in `fresh` once, at its first term, and every later term takes its id from a map by its id in `table`, not every term interned anew, its monomial's words hashed (log 0131-1, 0131-2)
// gf2-bound-losses.6 practice-checked: direct-address table -- the map is a vector over `table`'s ids, `MonoId::NONE` for a monomial no term has moved yet, dropped with the carry, not a hash map keyed by the old id (log 0131-2)
// call-stop-latency.23 design-checked -- the carry's copy spends a unit a word and its bound's tests a unit a pair, not no spend from the signature run's end to the plain run's start (log 0138-5)
/// Interns the carried elements' monomials, ids of `table`, in `fresh`, a
/// table of the same format, and puts the ids of `fresh` in their place,
/// each element's terms in their order: each monomial once, at its first
/// term, the later terms by a map from the ids of `table` to those of
/// `fresh`. Size where `fresh` is full, and a stop where a growth of
/// `fresh` meets the budget's check stopping the call. The map's fill
/// spends a unit an id (`matrix::filled`), and each element `PROBE` units a
/// term, its id mapped at a random place, and a unit a word of each
/// monomial it interns, at `Site::Carry`; a stop where the budget's check
/// stops the call.
pub(crate) fn carry_table<F: Field, M: Format, D: Module, E>(
    table: &Table<M>,
    fresh: &mut Table<M>,
    carried: &mut Carried<F, D, E>,
    budget: &mut Budget,
) -> Result<(), Limit> {
    debug_assert_eq!(
        table.format().variables(),
        fresh.format().variables(),
        "the two tables' formats"
    );
    let words = table.format().words();
    let mut moved = matrix::filled(table.len(), MonoId::NONE, budget, Site::Carry)?;
    for (h, _, _) in &mut carried.elements {
        let mut interned = 0;
        for u in &mut h.monomials {
            let to = &mut moved[u.index()];
            if *to == MonoId::NONE {
                *to = fresh.intern_words(table.monomial(*u), table.degree(*u), budget)?;
                interned += 1;
            }
            *u = *to;
        }
        budget.spend(h.monomials.len() * PROBE + interned * words, Site::Carry)?;
    }
    Ok(())
}

// signature-run-losses.4 design-checked -- the bound is the handover round's degree, less one where an older element's leading monomial divides a made element's at a multiplier whose degree plus the older element's sugar is at most the round's degree, not the round's degree wherever no made element's leading monomial divides an older one's, which left rab-katsura-10-p65521 a basis short of elements at every seed (log 0112-1)
// call-stop-latency.23 design-checked -- the carry's copy spends a unit a word and its bound's tests a unit a pair, not no spend from the signature run's end to the plain run's start (log 0138-5)
/// The bound of a carry: `degree`, the handover round's, or `degree` − 1
/// where the leading monomial of an element of `older`, each with its
/// sugar, divides one of `made`, the leading monomials the round made, at a
/// multiplier whose degree plus that sugar is at most `degree`. Such a made
/// element leaves a critical pair of signature degree `degree` that the
/// signature run has not reduced, and every signature below `degree` is
/// processed. Each made element's scan of `older` spends `PROBE` units an
/// element of `older`, a monomial read at a random place, at `Site::Carry`;
/// a stop where the budget's check stops the call.
pub(crate) fn carry_bound<M: Format>(
    table: &Table<M>,
    degree: u32,
    older: &[(MonoId, u32)],
    made: &[MonoId],
    budget: &mut Budget,
) -> Result<u32, Stop> {
    debug_assert!(degree > 0, "a round's degree is at least 1");
    for &m in made {
        let open = older.iter().any(|&(g, sugar)| {
            // g divides m, so the multiplier's degree does not wrap.
            table.divides(g, m)
                && u64::from(table.degree(m) - table.degree(g)) + u64::from(sugar)
                    <= u64::from(degree)
        });
        budget.spend(older.len() * PROBE, Site::Carry)?;
        if open {
            return Ok(degree.saturating_sub(1));
        }
    }
    Ok(degree)
}

// call-stop-latency.48 design-checked -- a term made monic spends PROBE units, its product with the inverse a remainder by p, not a unit a term as a row operation's entry, whose products are reduced once per row (not measured)
/// The nonzero generators, made monic, each with its cofactors.
/// `Halt::Misfit(0)` where a nonzero generator's leading coefficient has
/// no inverse: over several lanes, 0 at one lane. Each generator spends
/// `PROBE` units a term and one more at `Site::Monic`; a stop where the
/// budget's check stops the call.
fn monic_inputs<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    generators: Vec<Poly<F, D>>,
    cof: &mut C,
    budget: &mut Budget,
) -> Result<Elements<F, D, C::Element>, Halt> {
    let mut inputs = Vec::new();
    for (j, mut g) in generators.into_iter().enumerate() {
        // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
        budget.spend(1 + g.len() * PROBE, Site::Monic)?;
        if g.is_zero() {
            continue;
        }
        // q-learn-time.4 cited-checked: dekleine-2001 §3 -- on each division with a nonzero remainder the list of leading coefficients is checked, and a c_i = 0 means the leading term of r is lost under p_i, with very high probability a bad prime; graebe-1993 §4, independently -- a leading coefficient detected as nonzero that is zero mod p_i shows p_i unlucky
        let Some(s) = g.try_monic(field) else {
            return Err(Halt::Misfit(0));
        };
        inputs.push((g, cof.input(j, s, budget)?));
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
/// one column, the last, and its cofactors alone are composed. The search
/// spends `PROBE` units a row, its lead's degree read from the table, at
/// `Site::Decode` after each `RUN` rows; a stop where the budget's check
/// stops the call.
fn constant_made<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    made: &mut [Reduced<F>],
    m: &Matrix,
    table: &Table<M>,
    cof: &mut C,
    budget: &mut Budget,
) -> Result<Option<Elements<F, D, C::Element>>, Halt> {
    let mut found = None;
    for (k, chunk) in made.chunks(RUN).enumerate() {
        found = chunk
            .iter()
            .position(|r| table.degree(m.monomials[r.cols[0].index()]) == 0)
            .map(|j| k * RUN + j);
        budget.spend(chunk.len() * PROBE, Site::Decode)?;
        if found.is_some() {
            break;
        }
    }
    let Some(at) = found else {
        return Ok(None);
    };
    let constant = &mut made[at];
    let element = cof.compose(constant.kept, &m.rows, table, budget)?;
    let one = constant.take_poly(m, budget)?;
    budget.mark(Stage::Decode);
    Ok(Some(vec![(one, element)]))
}

/// The rows of `made` that counted, the rows of the matrix they were made
/// from, increasing: listed a unit a row at `Site::Decode` after each
/// `RUN`, and sorted as `sort_runs` spends; a stop where the budget's
/// check stops the call.
fn counted_rows<F: Field>(made: &[Reduced<F>], budget: &mut Budget) -> Result<Vec<RowId>, Stop> {
    let mut counted: Vec<RowId> = Vec::with_capacity(made.len());
    for chunk in made.chunks(RUN) {
        counted.extend(chunk.iter().map(|r| r.row));
        budget.spend(chunk.len(), Site::Decode)?;
    }
    // Two rows made from one block name its first row, and are equal.
    sort_runs(&mut counted, RowId::cmp, budget, Site::Decode)?;
    Ok(counted)
}

// call-stop-latency.49 practice-checked: in-place permutation by cycle following -- a sort of elements that are no Copy values sorts their places by sort_runs, ties by place, and moves the elements in place along the permutation's cycles, a unit an element, not the elements' stable sort spent after it, nor the elements moved into a second vector beside the first (not measured)
/// `v` sorted by `cmp`, ties in their order in `v`: their places sorted
/// as `sort_runs` spends at `site`, then each element moved to its place
/// along the cycles of the permutation, a unit an element spent after each
/// `RUN`. A stop where the budget's check stops the call, `v` then
/// dropped.
fn sort_by_places<T>(
    mut v: Vec<T>,
    cmp: impl Fn(&T, &T) -> Ordering,
    budget: &mut Budget,
    site: Site,
) -> Result<Vec<T>, Stop> {
    // A place done: no element's, as the places number fewer than it.
    const DONE: u32 = u32::MAX;
    let mut places: Vec<u32> = (0..v.len()).map(Narrow::narrow).collect();
    sort_runs(
        &mut places,
        |&a, &b| cmp(&v[a as usize], &v[b as usize]).then(a.cmp(&b)),
        budget,
        site,
    )?;
    // places[k] holds the place in `v` whose element goes to k. A cycle
    // from k takes each element of it to its place once, and marks the
    // places done.
    for start in (0..v.len()).step_by(RUN) {
        let end = (start + RUN).min(v.len());
        for k in start..end {
            let mut at = k;
            while places[at] != DONE {
                let from = places[at] as usize;
                places[at] = DONE;
                if from == k {
                    break;
                }
                v.swap(at, from);
                at = from;
            }
        }
        budget.spend(end - start, site)?;
    }
    Ok(v)
}

/// Adds a round's counts to the stats.
fn count<F: Field>(stats: &mut Stats, reduction: &Reduction<F>) {
    stats.rounds += 1;
    stats.reduced += reduction.reductions;
    stats.zero += reduction.zeros;
}

// gfp-unsolved.6 cited-checked: berthomieu-2021 §4.3, Remark 4 -- with the tracer the probabilistic linear algebra cannot be used in the learn run, since it could not detect which specific rows reduce to zero; berthomieu-2022a §3.4, Tracer over Q -- once nothing more is learned, probabilistic linear algebra accelerates the step
// gfp-unsolved.8 design-checked -- a run that tracks cofactors reduces each row on its own, not combinations with their cofactors tracked through a hook that begins an empty combination and adds its rows (log 0014-1)
// signature-zero-time.10 design-checked -- the tests run where no cofactors are tracked, neither the trace nor the learner listens, and the field has one lane, not in a learn run as well (log 0049-1)
// call-report.11 design-checked -- the run line names the reduction's mode and what decided it, from functions the engine decides by too, not the mode stated per round nor the engine writing a line of its own (not measured)
/// Whether a run's hooks let it reduce rows in random combinations, the
/// plain run (`matrix::combines_rows`), or rule them zero by tests, the
/// signature run (`matrix::tests_rows`): neither the trace nor the learner
/// listens, and no cofactors are tracked.
pub(crate) fn combines<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>() -> bool {
    !T::LISTENS && !L::LISTENS && !C::TRACKS
}

// learn-zero-time.1 invented-checked -- a learn run's plain run, where no cofactors are tracked, rules its rows zero by signature-zero-time.1's test of random combinations, the rows a test rules zero recorded as rows that reduced to zero, not each row reduced on its own (log 0063-1)
// learn-zero-time.4 design-checked -- where the trace listens the plain run reduces each row on its own, as `groebner_with` states, not tests where the trace listens as well (not measured)
// zero-rows-time.1 design-checked -- the learn's plain run takes learn-zero-time's tests beside the plain run's grown blocks of plain-zero-time, not each row of a learn reduced on its own beside the grown blocks, as gfp-unsolved.6 has it, nor the tests without the grown blocks (log 0078-1)
// q-zero-work-time.5 design-checked -- the run line names a learn's plain run that takes tests `reduce=tests`, from this function, which `run` decides by, not `reduce=rows reduce-by=learner` as for a learn's signature run (not measured)
/// Whether a plain run's hooks let it rule rows zero by tests while it
/// reduces each row on its own (`matrix::reduce`), where p takes a test
/// (`matrix::zero_combinations`): the learner listens, the trace does
/// not, and no cofactors are tracked.
pub(crate) fn learn_tests<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>() -> bool {
    !T::LISTENS && L::LISTENS && !C::TRACKS
}

/// Whether the plain run under these hooks takes the sketches over a field
/// that packs, GF(2), in its rounds whose rows are long enough
/// (`matrix::reduce`): its hooks let it combine rows (`combines`), and the
/// field packs (`Field::PACKS`).
pub(crate) fn plain_sketches<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>() -> bool
{
    combines::<F, M, C, T, L>() && F::PACKS
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
/// reduction of it, whose mode the line names where it combined rows,
/// whether its matrix took bands, and its packed rows' words. The rows that
/// reduced to zero are named but where every row to reduce went into a
/// combination, the blocks' mode.
fn figures<F: Field>(
    number: u32,
    degree: u32,
    pairs: usize,
    m: &Matrix,
    reduction: &Reduction<F>,
) -> Round {
    let zero_rows = match reduction.mode {
        None | Some(RoundMode::Rows) => Some(reduction.zero.len()),
        Some(RoundMode::Blocks) => None,
    };
    Round {
        number,
        degree,
        pairs,
        rows: m.rows.len(),
        given: m.given,
        kept: m.kept(),
        columns: m.columns(),
        entries: m.entries(),
        index: m.index(),
        pivots: reduction.made.len(),
        zero: reduction.zeros,
        zero_rows,
        reductions: reduction.reductions,
        bands: reduction.bands,
        packed: reduction.packed,
        mode: reduction.mode,
        sketches: reduction.sketches,
        pending: None,
        deferred: None,
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
// pair-work-handover.7 design-checked -- the handover line names the pair-work test's verdict and the S-pairs it read, not a line of its own, nor the S-pairs left out (not measured)
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
    if t.early.is_none()
        && t.gap.is_none()
        && t.outgrown.is_none()
        && t.large.is_none()
        && t.pair_work.is_none()
    {
        return;
    }
    let Some(report) = budget.report() else {
        return;
    };
    let verdict = |h: bool| if h { "hand-over" } else { "stay" };
    // cyclic-9-unsolved.7 design-checked -- the handover line names the large-round test's verdict between the outgrown test's and the pair-work test's, not a line of its own
    report.line(
        "handover",
        format_args!(
            " round={round}{early}{gap}{outgrown}{large}{pair_work} made={made} zero={zero}\
             {divisible}{pairs} widest={widest} verdict={verdict}",
            early = Opt("early", t.early.map(verdict)),
            gap = Opt("gap", t.gap.map(verdict)),
            outgrown = Opt("outgrown", t.outgrown.map(verdict)),
            large = Opt("large", t.large.map(verdict)),
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

// dead-monomial-peak.1 practice-checked: garbage collection -- at a round's end the plain run compacts its table to the monomials a later step reads, the basis's elements' monomials and leading monomials and the kept matrix's columns' monomials and rows' multipliers, which take new ids, not every monomial kept for the whole run (f4-gfp.14, from monagan-2015 §2.1, whose garbage collection of monomials did not improve its timings), nor a table of each matrix's own beside the run's (column-monomials-time), which held the reducers' multipliers and the made rows' columns in both (log 0117-1)
// call-stop-latency.8 design-checked -- a compaction spends its moves a unit a word, its slots emptied a unit each, and its ids placed PROBE units each, PLACE at a time, and the run's walks of the basis's and the buffers' ids a unit an id, not no spend (dead-monomial-peak.8) (log 0138-1, 0138-2)
/// Compacts the plain run's `table` after round `round` to the monomials a
/// later step reads (`Table::compact`): those `basis` holds, its elements'
/// and their leading monomials, and those of the matrix `bufs` keep for
/// the next build, its columns' and its rows' multipliers. Each takes its
/// new id there. The walks of the ids spend a unit an id at
/// `Site::Compact` (`Basis::ids_mut`, `Buffers::ids_mut`), and the table's
/// compaction spends its own. The call report, where the budget holds one, gets the
/// compaction's line. The count kept, or the stop where the budget's
/// check stops the call, the table and the ids then part renumbered.
fn compact<F: Field, M: Format, D: Module>(
    table: &mut Table<M>,
    basis: &mut Basis<F, D>,
    bufs: &mut Buffers<F>,
    budget: &mut Budget,
    round: u32,
) -> Result<usize, Stop> {
    budget.open();
    let monomials = table.len();
    let mut live = Live::new(monomials);
    basis.ids_mut(|u| live.mark(*u), budget, Site::Compact)?;
    bufs.ids_mut(|u| live.mark(*u), budget, Site::Compact)?;
    let kept = table.compact(&mut live, budget)?;
    basis.ids_mut(|u| *u = live.id(*u), budget, Site::Compact)?;
    bufs.ids_mut(|u| *u = live.id(*u), budget, Site::Compact)?;
    budget.mark(Stage::Compact);
    report_compact(budget, round, monomials, kept);
    Ok(kept)
}

// dead-monomial-peak.9 design-checked -- the compaction writes a line of its own after the round's, with the round, the monomials before, those kept, and its time in a stage of its own, not fields of the round line, which is written before the compaction, while most rounds compact nothing (not measured)
/// The call report's line of the compaction after round `round` of a
/// table of `monomials` monomials, which kept `kept`.
fn report_compact(budget: &mut Budget, round: u32, monomials: usize, kept: usize) {
    let Some(report) = budget.report() else {
        return;
    };
    let time = Secs(report.time(Stage::Compact));
    report.line(
        "compact",
        format_args!(" round={round} monomials={monomials} kept={kept} time={time}"),
    );
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
            " rows={rows} given={given} columns={columns} entries={entries} index={index} \
             subtracted={subtracted} elements={elements} symbolic={symbolic} encode={encode} \
             reduce={reduce} decode={decode}",
            rows = m.rows.len(),
            given = m.given,
            columns = m.columns(),
            entries = m.entries(),
            index = m.index(),
        ),
    );
}

// linear-tail-time.8 design-checked -- the inter-reduction between the rounds runs in a plain run whose trace and learner listen to nothing and that tracks no cofactors, not in every plain run with a step of the record for the replays and the members' cofactors composed, where the learn's record over ℚ took inter-reduced members that no replay held and the lift dropped its records (log 0166-1, 0166-2)
// linear-tail-time.11 design-checked -- in the test profile a thread-local, TAILS, turns the inter-reduction between the rounds off, not a public switch (not measured)
/// Whether a plain run of these cofactors, trace, and learner inter-reduces
/// `G_min` between its rounds (`tails`): where its hooks let it combine
/// rows (`combines`), neither the trace nor the learner listening and no
/// cofactors tracked, and in the test profile where `TAILS` holds.
fn tails_on<F: Field, M: Format, C: Cofactors<F, M>, T: Trace, L: Learner>() -> bool {
    #[cfg(test)]
    if !TAILS.get() {
        return false;
    }
    combines::<F, M, C, T, L>()
}

/// Where a plain run last inter-reduced `G_min` between its rounds
/// (`tails`), none before its first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tails {
    /// `G_min`'s members of degree 1 then.
    pub(crate) linear: usize,
    /// The basis's length then: the elements from it on joined since.
    pub(crate) since: usize,
}

// gf2-tails-time.2 tuned-checked -- where the rounds reduce in packed rows the members of degree 1 grow by a quarter before the next inter-reduction, not by a half, which took the calls over GF(2) as far at the machine's own clock and sr less far, nor by their count, which took the calls over GF(2) less far (log 0174-1, 0174-2)
/// Where the rounds reduce in packed rows, the members of degree 1 grow by
/// at least their count at the last inter-reduction over this, rounded up,
/// before `G_min` is inter-reduced again (gf2-tails-time.1, .2).
pub(crate) const TAILS_GROWTH_PART: usize = 4;

/// A monomial's mark in `reducible`: a tail term of no member yet seen.
const UNSEEN: u8 = 0;
/// A monomial's mark in `reducible`: a tail term of a member that no
/// leading monomial of `G_min` divides.
const IRREDUCIBLE: u8 = 1;
/// A monomial's mark in `reducible`: a tail term of a member that the
/// leading monomial of a member joined since divides.
const REDUCIBLE: u8 = 2;

/// Whether the plain run spares its inter-reductions between the rounds
/// (gf2-tails-time.1, .4, .5): outside the test profile always, in it where
/// `TAILS_SPARE` holds.
fn spares() -> bool {
    #[cfg(not(test))]
    let spares = true;
    #[cfg(test)]
    let spares = TAILS_SPARE.get();
    spares
}

// gf2-tails-time.4 cited-checked: makarim-2017 §4.1, via hauke-2022 §2.2 -- by numbering each update one tracks for every element between which two updates it was last tail-reduced, and an element of the current generation is tail-irreducible with respect to the current basis and needs no update
// gf2-tails-time.5 cited-checked:makarim-2017 §4.1, via hauke-2022 §2.2 -- an element last tail-reduced at an earlier update is tail-irreducible with respect to the basis of that update, so only the elements joined since can reduce its tail
// gf2-tails-time.6 design-checked -- the marks are a byte per monomial of the table, allocated zeroed for one inter-reduction and dropped after it, not an array the run keeps between its rounds, which would add to the rounds' peaks, nor the table's stamps, which the build's rounds take (log 0174-1, 0174-2)
/// The members of `G_min` that hold a tail term the leading monomial of a
/// member joined since `since` divides, in `G_min`'s order, and the marks
/// of the members' tail terms, by monomial id: `REDUCIBLE` where such a
/// leading monomial divides the term, `IRREDUCIBLE` where none does, and
/// so no leading monomial of `G_min` at all, as every member was reduced
/// by every leading monomial of a member before `since` (`tails`). A
/// member's terms spend a unit each at `Site::Enter`, `RUN` at a time, and
/// each search its masks' blocks and the candidates it tests at
/// `Site::Divisors` (`Divisors::find`); a stop where the budget's check
/// stops the call.
fn reducible<F: Field, M: Format, D: Module>(
    table: &Table<M>,
    basis: &Basis<F, D>,
    since: usize,
    budget: &mut Budget,
) -> Result<(Vec<ElementId>, Vec<u8>), Stop> {
    debug_assert!(
        since <= basis.len(),
        "{since} past the basis's {len}",
        len = basis.len()
    );
    let joined: Vec<ElementId> = basis
        .min()
        .iter()
        .copied()
        .filter(|i| i.index() >= since)
        .collect();
    let divisors = Divisors::of(table, basis, joined, budget)?;
    let mut marks = vec![UNSEEN; table.len()];
    let mut given = Vec::new();
    for &i in basis.min() {
        let monomials = &basis.element(i).monomials;
        let mut holds = false;
        for chunk in monomials[1..].chunks(RUN) {
            for &u in chunk {
                let mark = &mut marks[u.index()];
                if *mark == UNSEEN {
                    let (found, units) = divisors.find(table, u);
                    budget.spend(units, Site::Divisors)?;
                    *mark = if found.is_some() {
                        REDUCIBLE
                    } else {
                        IRREDUCIBLE
                    };
                }
                holds |= *mark == REDUCIBLE;
            }
            budget.spend(chunk.len(), Site::Enter)?;
        }
        if holds {
            given.push(i);
        }
    }
    Ok((given, marks))
}

// linear-tail-time.1 cited-checked: buchberger-2006 §9.4, item 1, via buchberger-1985 §6.4 (2) -- each time a new polynomial is adjoined to the basis, all the other polynomials may be reduced using also the new polynomial, and the representations are reduced at every new run of the part rel
// linear-tail-time.2 invented-checked -- G_min is inter-reduced where a member of degree 1 joined it since its last inter-reduction, not at the start of every round after one that entered an element, which cost time over GF(2), on hfe, sat, and sr most (log 0166-1, 0166-2)
// linear-tail-time.3 design-checked -- the trigger counts G_min's members of degree 1 alone, not those members only where they hold more than two terms each in the mean, which took the 16 over GF(2) as the count alone took them (log 0166-1, 0166-2)
// linear-tail-time.4 design-checked -- every member of G_min is inter-reduced, not its members of degree 1 alone by each other, which saved less time over GF(2) in the measuring condition, on sr above all (log 0166-1, 0166-2)
// linear-tail-time.5 design-checked -- G_min is inter-reduced by the inter-reduction's matrix, `inter_matrix` and `matrix::inter_reduce`, which the reduced basis takes, not a reduction of its own by the members of degree 1 (log 0166-1, 0166-2)
// linear-tail-time.9 design-checked -- the inter-reduction runs at the start of a round, before its selection, which covers the first round of a run from a carry, not after the update of the round that made the member and at the run's start (log 0166-1, 0166-2)
// linear-tail-time.12 design-checked -- the inter-reduction runs over every field, not over GF(2) alone, as it took time off GF(65521) in the measuring condition too (log 0166-1, 0166-2)
// linear-tail-time.13 design-checked -- a member keeps its sugar where its tail is reduced, not the largest of the sugars of the members its reduction took, which the selection would read as a fall (log 0166-1, 0166-2)
// gf2-tails-time.1 design-checked -- where the rounds reduce in packed rows, G_min is inter-reduced once its members of degree 1 have grown by a quarter since its last inter-reduction, not wherever one joined it (linear-tail-time.2), which on sr inter-reduced G_min at most of its rounds as its members of degree 1 joined a few at a time (log 0174-1, 0174-2)
// gf2-tails-time.3 design-checked -- the growth by a quarter holds where the rounds reduce in packed rows alone, Field::PACKS, not over every field, where it took sudoku-n9u56s5-p65521 and the calls over GF(65521) more instructions in the measuring condition (log 0174-1, 0174-2)
/// `G_min` inter-reduced where it holds more members of degree 1 than
/// `last.linear`, and where the rounds reduce in packed rows
/// (`Field::PACKS`) at least `last.linear` over `TAILS_GROWTH_PART`,
/// rounded up, more: the members that hold a tail term the leading monomial
/// of a member joined since `last.since` divides (`reducible`) as the
/// given rows of the inter-reduction's matrix, the other members reducers
/// where symbolic preprocessing takes them, with no divisor search for a
/// tail term that no such leading monomial divides (`build_searched`),
/// reduced by `matrix::inter_reduce`, and each member whose result differs
/// replaced by it in the basis, of the same leading monomial
/// (`Basis::replace`); `last` then holds the members of degree 1 and the
/// basis's length. Where no member holds such a term, no matrix is built
/// and the kept matrix stays. Elsewhere the kept matrix is dropped first
/// (`Buffers::forget`): the inter-reduction's build writes the table's
/// column slots, through which the next build finds a kept row, and a
/// changed member's kept rows are stale. The members of degree 1 of
/// `G_min` only grow: a leading monomial of degree 1 is divided by itself
/// or by a constant alone. The count reads each member's degree, `PROBE`
/// units a member spent `RUN` members at a time at `Site::Enter`; the given
/// rows spend as `reducible`, the matrix and its reduction as the
/// inter-reduction's, and each result compared with its member its terms
/// at `Site::Decode`. The call report's `tails` line, before round `round`.
/// The halt of the matrix's build, or the stop where the budget's check
/// stops the call. The caller tracks no cofactors (`tails_on`).
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and basis, the cofactors and the run's buffers, the last inter-reduction and the round, and the budget, none of which belongs with another"
)]
pub(crate) fn tails<F, M, O, D, C>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    basis: &mut Basis<F, D>,
    cof: &mut C,
    bufs: &mut Buffers<F>,
    last: &mut Tails,
    round: u32,
    budget: &mut Budget,
) -> Result<(), Halt>
where
    F: Field,
    M: Format,
    O: Order<M>,
    D: Module,
    C: Cofactors<F, M>,
{
    debug_assert!(!C::TRACKS, "the inter-reduction composes no cofactors");
    debug_assert!(round >= 1, "the rounds are numbered from 1");
    debug_assert!(
        last.linear <= basis.min().len(),
        "{linear} members of degree 1 past G_min's {len}",
        linear = last.linear,
        len = basis.min().len()
    );
    let mut members = 0;
    for chunk in basis.min().chunks(RUN) {
        members += chunk
            .iter()
            .filter(|&&i| table.degree(basis.lead(i)) == 1)
            .count();
        budget.spend(chunk.len() * PROBE, Site::Enter)?;
    }
    debug_assert!(
        members >= last.linear,
        "{members} members of degree 1 below the {linear} of the last inter-reduction",
        linear = last.linear
    );
    if members <= last.linear {
        return Ok(());
    }
    let spares = spares();
    let wanted = last
        .linear
        .saturating_add(last.linear.div_ceil(TAILS_GROWTH_PART));
    if F::PACKS && spares && members < wanted {
        return Ok(());
    }
    budget.open();
    let (given, marks) = if spares {
        reducible(table, basis, last.since, budget)?
    } else {
        (basis.min().to_vec(), Vec::new())
    };
    *last = Tails {
        linear: members,
        since: basis.len(),
    };
    let mut terms = 0;
    for chunk in basis.min().chunks(RUN) {
        terms += chunk.iter().map(|&i| basis.length(i)).sum::<usize>();
        budget.spend(chunk.len() * PROBE, Site::Enter)?;
    }
    let mut counts = TailsCounts {
        members: basis.min().len(),
        linear: members,
        given: given.len(),
        changed: 0,
        terms,
        after: terms,
    };
    if given.is_empty() {
        budget.mark(Stage::Symbolic);
        report_tails(budget, None, round, &counts);
        return Ok(());
    }
    bufs.forget();
    let rows = unit_rows(table, &given, budget)?;
    let m = matrix::build_searched(
        table,
        basis,
        order,
        rows,
        &mut Buffers::new(),
        budget,
        |u| marks.get(u.index()) != Some(&IRREDUCIBLE),
    )?;
    cof.matrix(m.rows.len());
    let made = matrix::inter_reduce(field, basis, &m, cof, budget, &mut Buffers::new())?;
    budget.mark(Stage::Reduce);
    for r in made {
        debug_assert!(r.row.index() < m.given, "a given row of G_min");
        let i = m.rows[r.row.index()].element;
        let mut poly = r.poly::<D>(&m, budget)?;
        let old = basis.element(i);
        budget.spend(old.len(), Site::Decode)?;
        // A tail entry at a pivot column leaves the row, so a result of the
        // same monomials is the member as it stood.
        if poly.monomials != old.monomials {
            counts.after = counts.after - old.len() + poly.len();
            poly.position = old.position;
            basis.replace(i, poly);
            counts.changed += 1;
        }
    }
    budget.mark(Stage::Decode);
    report_tails(budget, Some(&m), round, &counts);
    Ok(())
}

/// The counts of a `tails` line: `G_min`'s members, those of degree 1, the
/// given rows, the members changed, and the members' terms before and
/// after.
struct TailsCounts {
    members: usize,
    linear: usize,
    given: usize,
    changed: usize,
    terms: usize,
    after: usize,
}

// linear-tail-time.10 design-checked -- the inter-reduction between the rounds writes a line of its own before the round's, with its round, G_min's members, those of degree 1, the members it changed, their terms before and after, its matrix's rows and columns, and its time, not fields of the round line (not measured)
// gf2-tails-time.7 design-checked -- the line names the given rows beside G_min's members, its terms count every member's, and an inter-reduction that builds no matrix writes its line with no rows and no columns, not the given rows' terms alone nor no line (not measured)
/// The call report's line of `tails` before round `round`, `m` its matrix
/// where it built one: the counts of `counts`, the matrix's rows and
/// columns, 0 and 0 where it built none, and the time of the step opened
/// before the given rows were found.
fn report_tails(budget: &mut Budget, m: Option<&Matrix>, round: u32, counts: &TailsCounts) {
    let Some(report) = budget.report() else {
        return;
    };
    let time = Secs(
        [Stage::Symbolic, Stage::Encode, Stage::Reduce, Stage::Decode]
            .into_iter()
            .map(|s| report.time(s))
            .sum(),
    );
    let TailsCounts {
        members,
        linear,
        given,
        changed,
        terms,
        after,
    } = counts;
    report.line(
        "tails",
        format_args!(
            " round={round} members={members} linear={linear} given={given} changed={changed} \
             terms={terms} after={after} rows={rows} columns={columns} time={time}",
            rows = m.map_or(0, |m| m.rows.len()),
            columns = m.map_or(0, Matrix::columns),
        ),
    );
}

/// The reduced basis from `G_min`, a Gröbner basis, by the inter-reduction's
/// matrix (`inter_matrix`), by increasing leading monomial, each element
/// with its cofactors, and the matrix's line in the call report.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, order, and basis, and the cofactors, trace, and budget it borrows, none of which belongs with another"
)]
fn reduced_basis<F, M, O, D, C, T>(
    field: &F,
    table: &mut Table<M>,
    order: &O,
    basis: &Basis<F, D>,
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
    let m = inter_matrix(table, order, basis, budget)?;
    let made = inter_reduce(field, basis, &m, cof, trace, budget)?;
    budget.mark(Stage::Reduce);
    let elements = reduced_elements(table, order, &m, made, cof, budget)?;
    budget.mark(Stage::Decode);
    report_inter(budget, &m, elements.len());
    Ok(elements)
}

/// The matrix that inter-reduces `G_min`: its elements, in the order made,
/// as the given rows, and their reducers by symbolic preprocessing.
pub(crate) fn inter_matrix<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    order: &O,
    basis: &Basis<F, D>,
    budget: &mut Budget,
) -> Result<Matrix, Halt> {
    let rows = unit_rows(table, basis.min(), budget)?;
    // rebuilt-reducer-time.11 design-checked -- the replays' builds and the inter-reduction's take no row of the matrix before, not the replays' rows kept too
    matrix::build(table, basis, order, rows, &mut Buffers::new(), budget)
}

/// The rows of the elements `elements` by the unit monomial, which the
/// table interns; Size where the table is full, and a stop where the
/// budget's check stops the call.
fn unit_rows<M: Format>(
    table: &mut Table<M>,
    elements: &[ElementId],
    budget: &mut Budget,
) -> Result<Vec<RowSource>, Limit> {
    let one = table.intern(&vec![0; table.format().variables()], budget)?;
    Ok(elements
        .iter()
        .map(|&i| RowSource {
            element: i,
            multiplier: one,
        })
        .collect())
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
/// increasing leading monomial, each with its cofactors, decoded and sorted
/// at `Site::Decode` (`Reduced::poly`, `sort_by_places`).
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
        reduced.push((r.poly(m, budget)?, element));
    }
    let by_lead = |a: &(Poly<F, D>, C::Element), b: &(Poly<F, D>, C::Element)| {
        table.cmp(order, a.0.lead(), b.0.lead())
    };
    Ok(sort_by_places(reduced, by_lead, budget, Site::Decode)?)
}

/// The rows of the selected pairs: `lcm/LM(g_i)` times `g_i` and `lcm/LM(g_j)`
/// times `g_j` for each pair (i, j), each row once. Spent at
/// `Site::PairRows` as it goes: `PRODUCT` units a row, a division in the
/// table, after each `RUN` pairs, the sort as `sort_runs` spends, and the
/// repeats' removal a unit a row after each `RUN`; a stop where the
/// budget's check stops the call.
fn pair_rows<F: Field, M: Format, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    selected: &[Pair],
    budget: &mut Budget,
) -> Result<Vec<RowSource>, Limit> {
    // call-stop-latency.15 design-checked -- the pair rows, the columns' permutation and merge, the kept matrix's moves, the decode loops, the entering sort, the sugars, the monic inputs, and push_min spend per element, term, or entry visited, as they go, not unspent between the round's spends (not measured)
    let mut rows: Vec<RowSource> = Vec::with_capacity(2 * selected.len());
    for chunk in selected.chunks(RUN) {
        for p in chunk {
            for k in [p.i, p.j] {
                rows.push(RowSource {
                    element: k,
                    multiplier: table.div(p.lcm, basis.lead(k), budget)?,
                });
            }
        }
        budget.spend(2 * chunk.len() * matrix::PRODUCT, Site::PairRows)?;
    }
    // The pairs of one lcm that share an element share its row.
    // f4-gfp.45 design-checked -- the pair rows stand by their element's index, so a tie for a pivot among them goes to the element made first, not to the row a pair named first (not measured)
    // Two rows that compare equal are equal, so `sort_runs` sorts them as a
    // sort of distinct rows does.
    let key = |r: &RowSource| (r.element, r.multiplier);
    sort_runs(
        &mut rows,
        |a, b| key(a).cmp(&key(b)),
        budget,
        Site::PairRows,
    )?;
    let mut kept = 0;
    for start in (0..rows.len()).step_by(RUN) {
        let end = (start + RUN).min(rows.len());
        for k in start..end {
            if kept == 0 || rows[k] != rows[kept - 1] {
                rows[kept] = rows[k];
                kept += 1;
            }
        }
        budget.spend(end - start, Site::PairRows)?;
    }
    rows.truncate(kept);
    Ok(rows)
}

// handover-time.1 design-checked -- the signature run decides at its gap round once, the first round that makes an element whose signature's degree passes its leading monomial's by 4 or more, not at every such round, nor by the tests of the gap round at every such round (log 0038-1)
/// The gap of the gap round: the first round that makes an element whose
/// signature degree, its sugar degree, passes its leading monomial's
/// degree by `HANDOVER` or more.
pub(crate) const HANDOVER: u32 = 4;

// early-handover-time.1 design-checked -- the signature run decides at one more round, the first that makes an element whose signature's degree passes its leading monomial's by 2 or more, not at the gap round alone, nor by the gap round's tests at a gap of 2, 3, or 5 for every call (log 0054-1)
// early-handover-time.7 tuned-checked -- the early round's gap is 2, not 3 (log 0054-1)
/// The gap of the early round: the first round that makes an element
/// whose signature degree passes its leading monomial's by `EARLY` or
/// more.
pub(crate) const EARLY: u32 = 2;

// outgrown-round-size.1 design-checked -- the early round's tests and the outgrown test each read rounds of a size of their own, not one least for both (handover-time.6 and early-handover-time.3, withdrawn) (log 0075-1)
// outgrown-round-size.2 tuned-checked -- the early round's tests read a round of 25 elements or more where the plain run would combine rows, not 1, 10, 50, or 100 (log 0075-1)
/// The fewest elements an early round makes for its tests to read it
/// where the plain run after a handover would combine rows
/// (outgrown-round-size.2).
pub(crate) const EARLY_ROUND: usize = 25;

// outgrown-round-size.3 tuned-checked -- where the plain run would reduce each row alone, the early round's tests read a round of 100 elements or more, not 25 as where it combines rows, nor 50 (log 0075-1)
/// The fewest elements an early round makes for its tests to read it
/// where the plain run after a handover would reduce each row alone
/// (outgrown-round-size.3).
pub(crate) const EARLY_ROUND_ALONE: usize = 100;

// pair-work-handover.4 tuned-checked -- the pair-work test reads a round whose update would form 1,000 S-pairs or more, not every round, nor 10,000 or more (log 0067-1)
// pair-work-ratio.2 tuned-checked -- at the multiple of pair-work-ratio.1 the pair-work test reads a round of 1,000 S-pairs or more, as at 4, not of 300 or 100 or more (log 0080-2)
/// The fewest S-pairs a round's update would form for the pair-work test
/// to read the round (pair-work-handover.4, pair-work-ratio.2).
pub(crate) const PAIRS: u64 = 1_000;

// pair-work-ratio.1 tuned-checked -- the pair-work test hands over where the S-pairs reach the entries subtracted, not 4 times them (pair-work-handover.3, withdrawn), 2 times, or half of them (log 0080-2)
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

// cyclic-9-unsolved.2 tuned-checked -- the large-round test reads a round of 150 elements or more, not 100, which reads rur-gametwo7-p65521's round 5 as well, nor 200, which hands cyclic-9-p65521 over a round later (log 0093-1, _Step 3_)
/// The fewest elements a round makes for the large-round test to read it
/// (cyclic-9-unsolved.2).
pub(crate) const LARGE: usize = 150;

// cyclic-9-unsolved.3 tuned-checked -- a large round hands over where a quarter or more of its elements are divisible, not 3 in 10, which keeps cyclic-9-p65521's round 9, nor a tenth (log 0093-1, _Step 3_)
/// The share of its elements at or above which the divisible elements of
/// a large round hand over (cyclic-9-unsolved.3): a quarter.
pub(crate) const LARGE_DIVISIBLE: Share = Share { n: 1, d: 4 };

// cyclic-9-unsolved.4 tuned-checked -- a large round hands over where it reduced fewer rows to zero than a tenth of its elements, not a twentieth, which cyclic-9-p65521's round 10 barely meets and cyclic-8-p0's misses, nor a fifth, which comes near the large rounds of a quarter divisible that reduce many rows to zero (log 0093-1, _Step 3_)
/// The share of its elements below which the rows a large round reduced to
/// zero hand over (cyclic-9-unsolved.4): a tenth.
pub(crate) const LARGE_ZEROS: Share = Share { n: 1, d: 10 };

// signature-run-losses.6 tuned-checked -- a handover carries the signature run's work to the plain run where the signature run subtracted 100,000 entries or more, not at every handover, which made ccode-cooper45t4-p65521 and the semaev instances far slower, their plain run off the path the deferral of cooper-rest-gap was tuned on, nor at 10,000, which takes the carry to calls it spares next to nothing, nor at 1,000,000, which leaves out anemoi-l1R2a43-p65521 and phc-rps10-p65521 (log 0112-1)
/// The fewest entries a signature run subtracts before a handover for the
/// plain run to go on from its work (signature-run-losses.6).
pub(crate) const CARRY_WORK: u64 = 100_000;

#[cfg(test)]
thread_local! {
    /// The test profile's least of the large-round test on this thread:
    /// Some(k) reads a round of k elements or more, None `LARGE`.
    pub(crate) static LARGE_LEAST: Cell<Option<usize>> = const { Cell::new(None) };
    /// The test profile's least work of a carry on this thread: Some(w)
    /// carries a handover after w entries or more, None `CARRY_WORK`.
    pub(crate) static CARRY_LEAST: Cell<Option<u64>> = const { Cell::new(None) };
    // dead-monomial-peak.10 design-checked -- in the test profile a thread-local stands in for COMPACT_GROWTH, Some(0) compacting at every round's end and Some(usize::MAX) never, so that test code compares the two on one call, not a parameter of the call nor the constant lowered (not measured)
    /// The test profile's monomials between two compactions on this
    /// thread: Some(k) compacts once the table has interned k monomials
    /// since the last, Some(0) at every round's end, None `COMPACT_GROWTH`.
    pub(crate) static COMPACT_BOUND: Cell<Option<usize>> = const { Cell::new(None) };
    /// The test profile's switch of the inter-reduction between the rounds
    /// on this thread: false turns it off (`tails_on`).
    pub(crate) static TAILS: Cell<bool> = const { Cell::new(true) };
    // gf2-tails-time.8 design-checked -- in the test profile a thread-local, true by default, turns the growth by a quarter, the given rows a joined member reduces, and the searches left out off together, so that test code compares the inter-reductions with and without them on one call, not a parameter of the call nor a thread-local per decision (not measured)
    /// The test profile's switch of the spared inter-reductions on this
    /// thread (gf2-tails-time.1, .4, .5): false inter-reduces wherever a
    /// member of degree 1 joined `G_min`, every member a given row and every
    /// monomial searched (`spares`).
    pub(crate) static TAILS_SPARE: Cell<bool> = const { Cell::new(true) };
    // bivium-rest-gap.7 design-checked -- in the test profile a thread-local, false by default, records a coprime S-pair's Koszul signature by `push_syzygy`'s search and drop, so that test code compares a call with and without `push_coprime`, not a parameter of the call (not measured)
    /// The test profile's switch of the coprime S-pairs' records on this
    /// thread: true takes `Signatures::push_syzygy`, false
    /// `Signatures::push_coprime`.
    pub(crate) static KOSZUL_SEARCHED: Cell<bool> = const { Cell::new(false) };
}

// dead-monomial-peak.4 tuned-checked -- the run compacts once its table has interned 2^14 monomials since its last compaction or since the rounds began, not at every round's end where the table grew, which cost time on calls of many rounds, nor where the table doubled since the last, which leaves it up to twice what it keeps, nor at every round's end from 2^16 monomials on, at 1.25, 1.5, or 2 times the monomials after the last, nor at 2^13 or 2^15 monomials interned (log 0117-1)
/// The monomials a plain run's table interns between two compactions
/// (dead-monomial-peak.4).
pub(crate) const COMPACT_GROWTH: usize = 1 << 14;

/// The fewest elements a round makes for the large-round test to read it:
/// `LARGE`, or in the test profile what `LARGE_LEAST` holds.
fn large_least() -> usize {
    #[cfg(test)]
    if let Some(least) = LARGE_LEAST.get() {
        return least;
    }
    LARGE
}

/// The fewest entries a signature run subtracts before a handover for the
/// plain run to go on from its work: `CARRY_WORK`, or in the test profile
/// what `CARRY_LEAST` holds.
fn carry_least() -> u64 {
    #[cfg(test)]
    if let Some(least) = CARRY_LEAST.get() {
        return least;
    }
    CARRY_WORK
}

/// The monomials a plain run's table interns between two compactions:
/// `COMPACT_GROWTH`, or in the test profile what `COMPACT_BOUND` holds.
fn compact_growth() -> usize {
    #[cfg(test)]
    if let Some(bound) = COMPACT_BOUND.get() {
        return bound;
    }
    COMPACT_GROWTH
}

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
/// round that made an element, at every round of `LARGE` elements or more,
/// and at every round whose update would form `PAIRS` S-pairs or more.
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

// call-report.12 design-checked -- every round at which a handover test reads the round gets a line with each test's verdict, not a line for the round that hands over alone (not measured)
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
    pub(crate) large: Option<bool>,
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
    /// the early test did not hand over, the outgrown test a round that
    /// made an element where neither handed over, and the large-round test
    /// a round of `LARGE` elements or more where none of them handed over.
    /// The pair-work test, `after_pairs`, reads a round of `PAIRS` S-pairs
    /// or more where none of them handed over.
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
    /// then the outgrown test, where the round made an element, and then
    /// the large-round test, where it made `LARGE` elements or more. The
    /// early round's tests read a round of the `Handover`'s least or more
    /// elements. `divisible` counts the elements whose
    /// leading monomial a leading monomial of the basis divides, and is
    /// called only where a test reads it, at most once; a stop where its
    /// count stops the call.
    pub(crate) fn after_round(
        &mut self,
        early: bool,
        gap: bool,
        made: usize,
        zeros: usize,
        divisible: impl FnOnce() -> Result<usize, Stop>,
    ) -> Result<bool, Stop> {
        let count = LazyCell::new(divisible);
        let read = || *count;
        let mut tests = Tests::default();
        let handed = 'tests: {
            // early-handover-time.2 design-checked -- the early round decides once, not every round of a gap of 2 or 3 before the gap round (log 0054-1)
            if early && !self.early {
                self.early = true;
                // input-redundancy-time.7 design-checked -- the early round's tests read a round that reduced no row to zero, as one that reduced some, not only a round that reduced a row to zero (early-handover-time.4) (log 0068-1)
                // early-handover-time.5 design-checked -- the early round hands over where it reduced fewer rows to zero than a quarter of the elements it made, not half of them (log 0054-1)
                // early-handover-time.6 design-checked -- the early round hands over too where more than 6 in 10 of its elements are divisible, the gap round's share test (handover-time.3, .4) counted as handover-time.8 counts them, not by the zero test of early-handover-time.5 alone (log 0054-1)
                // threshold-recheck.16 design-checked -- the early round hands over where more than 6 in 10 of its elements are divisible, the gap round's share (handover-time.4), not 5 or 7 in 10; neither moved the time past the margin on the tuning set (`LEARNINGS.md`, _The engine's constants measured again on the tuning set of iterations 0067 on_) (log 0054-1, 0074-2)
                let hands = made >= self.least
                    && (EARLY_ZEROS.below(zeros, made) || DIVISIBLE.above(read()?, made));
                tests.early = Some(hands);
                if hands {
                    break 'tests true;
                }
            }
            if gap && !self.decided {
                self.decided = true;
                // handover-time.2 design-checked -- the gap round hands over where it reduced fewer rows to zero than half the elements it made, not by the share of divisible elements alone, nor by the elements made against the round before; eder-2013a §5 finds the signature criteria discard more pairs than Gebauer and Möller's on inhomogeneous input and the reduction suffer, and is silent on the rows a round reduces to zero (log 0038-1)
                // handover-time.3 design-checked -- the gap round hands over too where most of its elements are divisible, their leading monomial divisible by a leading monomial of the basis before the round, not by the zero test of handover-time.2 alone (log 0038-1)
                // handover-time.4 tuned-checked -- most of the gap round's elements is more than 6 in 10, not 5 or 7 in 10 (log 0038-1)
                let hands = GAP_ZEROS.below(zeros, made) || DIVISIBLE.above(read()?, made);
                tests.gap = Some(hands);
                if hands {
                    break 'tests true;
                }
            }
            // handover-time.5 design-checked -- a round that makes enough elements hands over where 8 in 10 or more of them are divisible, whether or not it makes a gap element and whether or not the gap round has been, not at the gap alone (log 0038-1)
            // handover-time.7 tuned-checked -- an outgrown round hands over where 8 in 10 or more of its elements are divisible, not 7 or 9 in 10 (log 0038-1)
            // outgrown-round-size.4 tuned-checked -- the outgrown test reads every round that made an element, not a round of 100 (handover-time.6, withdrawn), 50, 25, 10, or 5 elements or more (log 0075-1)
            // A round that made no element has no share to read.
            let outgrown = made > 0;
            let hands = outgrown && OUTGROWN.reached(read()?, made);
            tests.outgrown = outgrown.then_some(hands);
            if hands {
                break 'tests true;
            }
            // cyclic-9-unsolved.1 cited-checked: roune-2012 Appendix B.4 -- stopping on detecting a Gröbner basis is only a benefit when the signature Gröbner basis is much larger than the minimal Gröbner basis, an element whose lead term another's divides being no essential one; design -- a round of many elements hands over where a share of them are divisible and few of its rows reduced to zero, all three together, not the zeros alone nor the divisible share alone
            // cyclic-9-unsolved.5 design-checked -- the large-round test reads every round of its least or more, not once at the first such round
            // cyclic-9-unsolved.6 design-checked -- the large-round test runs after the early, the gap, and the outgrown tests, where none of them handed over, and before the pair-work test
            let large = made >= large_least();
            let hands =
                large && LARGE_ZEROS.below(zeros, made) && LARGE_DIVISIBLE.reached(read()?, made);
            tests.large = large.then_some(hands);
            hands
        };
        tests.divisible = LazyCell::get(&count).copied().transpose()?;
        self.tests = tests;
        Ok(handed)
    }

    // pair-work-handover.1 invented-checked -- a round hands over where the S-pairs its update would form reach a multiple of the entries its row operations subtracted, not by the S-pairs alone, nor against the entries of its matrix (log 0067-1)
    // pair-work-handover.2 design-checked -- the pair-work test reads every round of `PAIRS` S-pairs or more, from the first on, not once at the first round it reads, as the early and the gap round decide (log 0067-2)
    // pair-work-handover.6 design-checked -- the pair-work test runs after the early, the gap, and the outgrown tests, where none of them handed over, not before them (not measured)
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

// signature-pruning.1 cited-checked: lairez-2022 §5.7, Algorithm 8 and Theorem 62 -- a batch of signatures is reduced at a time, modulo AG^{<σ} and the batch's new elements; ars-2005 §1.4.f, independently, via albrecht-2010 §1 and §5 -- a version of F5 close to the form of F4, whose computations are done on sub-matrices
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
/// outgrown test reads every round that made an element, and its
/// large-round test every round of `LARGE` elements or more. Its
/// pair-work test reads the S-pairs the round's elements would form as they
/// enter against the entries the round's row operations subtracted. On
/// a handover the round is counted and told to the trace and the learner,
/// and none of its elements entered. The handover returns a carry to the
/// plain run (`carry`) where neither the trace nor the learner listens, no
/// cofactors are tracked, and the run has subtracted `CARRY_WORK` entries
/// or more since its start, and `Halt::Handover` otherwise. `pairs` says whether
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
) -> Result<SigEnd<F, D, C::Element>, Halt>
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
    budget.check(|| {}, Site::Run)?;
    // The entries subtracted before the run, which its carry leaves out.
    let start = budget.counts().0;
    budget.open();
    learner.inputs(|| trace::inputs(table, order, &generators, false, false, budget))?;
    // learn-zero-time.3 design-checked -- a learn's signature run keeps no test, signature-zero-time.10 standing, while its plain run takes them (`run`), not the tests in the pair learn's signature run as well (log 0063-1)
    // signature-zero-time.11 design-checked -- the signature run's coefficients come from a Combine seeded by the call's combine draw, the plain run's, which after a handover builds its own from the same draw, not a draw of its own from the call's generator (not measured)
    let mut combine = combines::<F, M, C, T, L>().then(|| Combine::new(seed));
    let mut inputs = monic_inputs(field, generators, cof, budget)?;
    if let Some(one) = constant_input(&mut inputs, table, budget) {
        return Ok(SigEnd::Basis(one));
    }
    let mut run = SigRun {
        basis: Basis::new(),
        sigs: Signatures::new(inputs.len()),
        queue: Queue::new(),
        triangle: pairs.bases.then(Triangle::new),
        bufs: Buffers::new(),
    };
    // Inputs i < j have the Koszul signature lm(f_i)·lm(f_j)·e_j, which the
    // signatures test by the inputs' leading monomials (bivium-gap.1).
    run.sigs
        .set_inputs(table, inputs.iter().map(|(g, _)| g.lead()).collect());
    let elements = inputs.len();
    let queued = pairs.queued;
    // signature-pruning.17 design-checked -- the inputs keep the instance's order as their indices (not measured)
    for (i, (g, element)) in inputs.into_iter().enumerate() {
        let sig = Sig {
            monomial: g.lead(),
            index: i.narrow(),
        };
        run.enter(table, order, cof, budget, pairs, g, element, sig)?;
    }
    budget.mark(Stage::Enter);
    report_enter(budget, elements, pairs.queued - queued);

    let mut round = 0;
    // outgrown-round-size.5 design-checked -- the early round's least is fixed when the run's `Handover` is made, from the plain run's mode under the run's hooks and its prime, not read at every round's decision, nor from the hooks alone without the prime (not measured)
    let mut decision = Handover::new(plain_combines::<F, M, C, T, L>(field.characteristic()));
    // The pending signatures taken since the last round, and those of them
    // that gave no reductant.
    let mut taken = (0, 0);
    while !run.queue.is_empty() {
        budget.check(|| run.release(table), Site::Round)?;
        budget.open();
        let (given, popped) = run.reductants(table, order, budget)?;
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
            given,
            &mut run.bufs,
            budget,
        )?;
        trace.matrix(&m.rows);
        // call-report.22 design-checked -- the trace's hook of the matrix before its reduction counts into the report's record stage, not into reduce (not measured)
        budget.mark(Stage::Record);
        cof.matrix(m.rows.len());
        // pair-work-handover.5 design-checked -- the round's work in its reduction is the entries its row operations subtracted, the budget's count across the reduction, not with the entries added into combinations beside them, nor a count of its own in the reduction's result (log 0067-1)
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
        budget.counted(|| Round {
            pending: Some(taken),
            ..figures(round, degree, reductants, &m, &reduction)
        });
        taken = (0, 0);
        learner.step(|reach| trace::step(table, order, &m, &reduction, false, reach, budget))?;
        let counted = counted_rows(&reduction.made, budget)?;
        trace.counted(&counted);
        budget.mark(Stage::Record);
        // gfp-unsolved.14 cited-checked: roune-2012 Appendix B.1 -- an S-pair is eliminated early, when made, or late, right before its reduction; trying to eliminate them all every time new information comes in would take a lot of time
        // A pending signature a known syzygy signature divides is dropped
        // when a round takes it (`SigRun::reductants`).
        for &r in &reduction.zero {
            run.sigs.push_syzygy(table, sigs[r.index()], budget)?;
        }
        // f4-gfp.22 design-checked -- a constant element ends the run with the basis {1}, not running until no pair is left (not measured)
        let mut made = reduction.made;
        if let Some(one) = constant_made(&mut made, &m, table, cof, budget)? {
            learner.output(Output::Constant, budget)?;
            budget.end_round(None);
            report_constant(budget, round);
            return Ok(SigEnd::Basis(one));
        }
        // The handover's tests that read the round, with the elements made,
        // the rows reduced to zero, and the widest gap.
        let mut read = None;
        // handover-time.1 design-checked -- the signature run decides at its gap round once, the first round that makes an element whose signature's degree passes its leading monomial's by 4 or more, not at every such round, nor by the tests of the gap round at every such round (log 0038-1)
        if let Some(g) = handover {
            let lead = |r: &matrix::Reduced<F>| m.monomials[r.cols[0].index()];
            // The round's widest gap, 0 where it made no element, two
            // degrees read at random places a row.
            let mut widest = 0;
            for chunk in made.chunks(RUN) {
                for r in chunk {
                    let sig = sigs[r.row.index()].degree(table);
                    widest = widest.max(sig.saturating_sub(table.degree(lead(r))));
                }
                budget.spend(chunk.len() * 2 * PROBE, Site::Decode)?;
            }
            // early-handover-time.1 design-checked -- the signature run decides at one more round, the first that makes an element whose signature's degree passes its leading monomial's by 2 or more, not at the gap round alone, nor by the gap round's tests at a gap of 2, 3, or 5 for every call (log 0054-1)
            let early = widest >= EARLY;
            let gap = widest >= g;
            // handover-time.8 design-checked -- an element is counted divisible by the signature run's reducer search over the basis before the round's elements enter, and only where a test reads the count, not with the round's own elements among the divisors, nor by the plain run's search over G_min built for the test (not measured)
            let divisible = || -> Result<usize, Stop> {
                let mut count = 0;
                for r in &made {
                    let (found, tested) = run.sigs.reducer(table, lead(r));
                    count += usize::from(found.is_some());
                    budget.spend(search_units(run.sigs.len(), tested), Site::Reducer)?;
                }
                Ok(count)
            };
            let zeros = reduction.zero.len();
            let handed = decision.after_round(early, gap, made.len(), zeros, divisible)?
                || decision.after_pairs(pairs_formed(made.len(), run.basis.len()), subtracted);
            read = Some((made.len(), zeros, widest));
            if handed {
                budget.mark(Stage::Decode);
                budget.end_round(None);
                report_handover(budget, round, &decision, read, true);
                // signature-run-losses.1 design-checked -- at a handover where neither the trace nor the learner listens, no cofactors are tracked, and the signature run has subtracted `CARRY_WORK` entries or more, the plain run goes on from the signature run's elements and the handover round's, not from the input over again (gfp-unsolved.16), nor from the signature run's elements alone with the round reduced again (log 0112-1)
                // The budget's count only grows.
                let work = budget.counts().0.saturating_sub(start);
                if combines::<F, M, C, T, L>()
                    && work >= carry_least()
                    && let Some(untracked) = C::untracked()
                {
                    let made: Vec<Poly<F, D>> = made
                        .into_iter()
                        .map(|r| r.poly(&m, budget))
                        .collect::<Result<_, Stop>>()?;
                    let count = made.len();
                    let (elements, bound) =
                        carry(table, run.basis, &run.sigs, made, degree, untracked, budget)?;
                    return Ok(SigEnd::Carry(Carried {
                        elements,
                        bound,
                        round,
                        made: count,
                        work,
                    }));
                }
                // round-memory-time.11 design-checked -- the plain run after a handover starts with buffers of its own, and the signature run's drop with its halt, not handed to the plain run (not measured)
                return Err(Halt::Handover);
            }
        }
        let mut new = Vec::with_capacity(made.len());
        for r in made {
            let element = cof.compose(r.kept, &m.rows, table, budget)?;
            let sig = sigs[r.row.index()];
            new.push((r.poly(&m, budget)?, element, sig));
        }
        budget.mark(Stage::Decode);
        // The rows were made in increasing signature order.
        for (h, element, sig) in new {
            run.enter(table, order, cof, budget, pairs, h, element, sig)?;
        }
        budget.mark(Stage::Update);
        budget.end_round(None);
        report_handover(budget, round, &decision, read, false);
        run.bufs.keep(m);
    }

    // round-memory-time.7 design-checked -- the buffers live for one run's rounds and go back where the rounds end, before the inter-reduction, not buffers that live for the call or for the thread (log 0051-1)
    run.bufs.release();
    budget.check(|| run.release(table), Site::Run)?;
    learner.output(Output::Basis(run.basis.min()), budget)?;
    let elements = reduced_basis(field, table, order, &run.basis, cof, trace, budget)?;
    Ok(SigEnd::Basis(elements))
}

/// How a signature run ends where no halt ends it: with the reduced basis,
/// or at a handover with a carry to the plain run (signature-run-losses.1).
pub(crate) enum SigEnd<F: Field, D: Module, E> {
    Basis(Elements<F, D, E>),
    Carry(Carried<F, D, E>),
}

// call-stop-latency.23 design-checked -- the carry's copy spends a unit a word and its bound's tests a unit a pair, not no spend from the signature run's end to the plain run's start (log 0138-5)
/// The elements a signature run carries where it hands over after a round
/// of signature degree `degree`, and their bound (`carry_bound`): the
/// elements of `basis` in the order they entered, each at its signature's
/// degree in `sigs`, then `made`, the round's elements, each at `degree`,
/// all with the cofactors `untracked` makes. An element of `basis` spends
/// `PROBE` units, its signature's degree read at a random place, and one of
/// `made` a unit, at `Site::Carry`, and the bound its own; a stop where the
/// budget's check stops the call.
fn carry<F: Field, M: Format, D: Module, E>(
    table: &Table<M>,
    basis: Basis<F, D>,
    sigs: &Signatures,
    made: Vec<Poly<F, D>>,
    degree: u32,
    untracked: fn() -> E,
    budget: &mut Budget,
) -> Result<(Sugared<F, D, E>, u32), Stop> {
    let mut elements: Sugared<F, D, E> = Vec::with_capacity(basis.len() + made.len());
    let mut older: Vec<(MonoId, u32)> = Vec::with_capacity(basis.len());
    for (k, h) in basis.into_elements().into_iter().enumerate() {
        budget.spend(PROBE, Site::Carry)?;
        // signature-run-losses.2 cited-checked: eder-2013a §4, Theorem 4.2 -- the signature degree of a labeled polynomial equals the sugar degree of its polynomial part
        let sugar = sigs.sig(ElementId::at(k)).degree(table);
        older.push((h.lead(), sugar));
        elements.push((h, sugar, untracked()));
    }
    let mut leads: Vec<MonoId> = Vec::with_capacity(made.len());
    for h in made {
        budget.spend(1, Site::Carry)?;
        leads.push(h.lead());
        elements.push((h, degree, untracked()));
    }
    let bound = carry_bound(table, degree, &older, &leads, budget)?;
    Ok((elements, bound))
}

// trace-replay.1 cited-checked: berthomieu-2021 §4.3, via berthomieu-2022a §3.4 (3) -- the matrices are generated from the trace's polynomials, with no need of handling critical pairs or symbolic preprocessing; traverso-1989 §Gröbner traces, via demin-2024 §2.2, independently -- critical pairs and simplifier sequences are sufficient to reconstruct the algorithm avoiding all useless critical pairs and all division tests; joux-2011 §2.1 (2), independently -- do not maintain nor update a queue of untreated pairs
/// The reduced Gröbner basis of the generators from the rows of `record`,
/// as `run` returns it: the inputs enter the basis, each step's matrix is
/// built from its rows and reduced, and `G_min` is inter-reduced as in `run`.
/// A skipped row's element enters as its leading monomial (`enter_skipped`).
/// Under `Shaping::Take` the replay keeps what it builds in the `Built`:
/// the nonzero generators' monomials, every matrix with its steps' lead
/// ids, and the columns of every row each step's reduction made. Under
/// `Shaping::Use` the matrices are the `Built`'s, whose table is `table`,
/// and the replay builds none: None at the first input, or row a step
/// made, whose terms differ from the shape's. The budget is checked at the
/// start, at the start of every step that builds a matrix, and before the
/// inter-reduction, and in between as in `run`.
/// `Halt::Misfit` where the record does not fit. The steps' matrices take
/// their buffers in turn from one `matrix::Buffers`, which goes back where
/// the steps end, before the inter-reduction. The call report gets its
/// lines as in `run`, a round's per step that builds a matrix.
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
    budget.check(|| {}, Site::Run)?;
    let n = table.format().variables();
    // trace-replay.6 design-checked -- a replay checks that the record fits, the variables, the generators, which are zero, and their leading monomials, and stops at the first failure, not the leading monomials of the output compared with the learn run's after the replay (not measured)
    // q-later-runs-time.4 design-checked -- a pair fits only where every leading coefficient of the generators it is given, the pivots of their echelon form where the record took it, is a unit at both primes, and where it does not each prime replays alone, not the pair a misfit at both primes nor the lane that fits taken from the pair's run (not measured)
    // run-generators-time.13 design-checked -- the replay checks each generator's leading monomial and coefficient against the record where `echelon::replay` gave the generators from the record too, not the check left out there (not measured)
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
        // q-replay-time.4 design-checked -- a replay from the shape checks every element's terms against it, each nonzero generator's monomials and the columns of each row each step's reduction makes, and stops at the first that differs, not a shape that holds the union of the terms seen with explicit zero entries and no check (not measured)
        if !terms.eq(&built.inputs) {
            return Ok(None);
        }
    } else if let Some(built) = keep.as_deref_mut() {
        built.inputs = terms.cloned().collect();
    }
    let mut inputs = monic_inputs(field, generators, cof, budget)?;
    if let Some(one) = constant_input(&mut inputs, table, budget) {
        return Ok(Some(one));
    }
    // trace-replay.12 design-checked -- the record holds the order in which the inputs and each step's new elements entered the basis, not a replay that sorts them as the learn run's engine does (not measured)
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
        basis.push_min(table, g, budget)?;
        cof.push(element);
    }

    let mut bufs: Buffers<F> = Buffers::new();
    let mut round = 0;
    for (s, step) in record.steps.iter().enumerate() {
        // A misfit names its step of the record, counted from 1, whether or
        // not the steps before it built a matrix; `round` counts the built
        // ones alone, for the shape and the call report.
        let at = narrow::id(s + 1)?;
        // q-replay-reduction-time.5 design-checked -- a step all of whose counted rows are skipped stays in the record without rows, and the replay builds no matrix for it and enters its elements, not the step dropped and its elements entered by the step before it
        if step.reductees.is_empty() {
            enter_unbuilt(field, table, &mut basis, cof, step, budget, at)?;
            continue;
        }
        // The shape holds a matrix per step that builds one.
        let shaped = round as usize;
        budget.check(
            || {
                table.shrink_to_fit();
                basis.shrink_to_fit();
                bufs.release();
            },
            Site::Round,
        )?;
        budget.open();
        round += 1;
        // A shape ends before this step only where its replay made a
        // constant, and a replay with its terms makes the same constant: the
        // first arm is not reached.
        let taken = match shape.map(|built| built.steps.get(shaped)) {
            Some(None) => return Ok(None),
            Some(Some(taken)) => Some(taken),
            None => None,
        };
        let mut owned = None;
        // q-replay-time.1 cited-checked: berthomieu-2022a §3.4, Tracer over Q -- the computations after those that learn apply the learned data, with no critical pairs and no symbolic preprocessing; parisse-2013 §3 -- the symbolic preprocessing can reuse for the next primes the collection of monomials used for the first; demin-2024 §4.4 -- the apply stage caches the order of rows and columns in the matrices across the modular runs
        let (m, leads) = if let Some(taken) = taken {
            (&taken.matrix, &taken.leads[..])
        } else {
            let (m, leads) =
                owned.insert(build_step(table, &basis, order, step, &mut bufs, budget)?);
            (&*m, &leads[..])
        };
        trace.matrix(&m.rows);
        // call-report.22 design-checked -- the trace's hook of the matrix before its reduction counts into the report's record stage, not into reduce (not measured)
        budget.mark(Stage::Record);
        cof.matrix(m.rows.len());
        // gfp-unsolved.11 design-checked -- a record says whether its run inter-reduced the new rows, and a replay inter-reduces where it says so, not a replay that inter-reduces where the order is not graded (not measured)
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
            return Err(Halt::Misfit(at));
        };
        // q-replay-time.4 design-checked -- a replay from the shape checks every element's terms against it, each nonzero generator's monomials and the columns of each row each step's reduction makes, and stops at the first that differs, not a shape that holds the union of the terms seen with explicit zero entries and no check (not measured)
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
            kept: m.kept(),
            columns: m.columns(),
            entries: m.entries(),
            index: m.index(),
            pivots: made.len(),
            zero: 0,
            zero_rows: Some(0),
            reductions: made.len() as u64,
            bands: false,
            packed: None,
            mode: None,
            sketches: None,
            pending: None,
            deferred: None,
            left: None,
        });
        let mut counted = step.reductees.clone();
        // call-stop-latency.11 design-checked -- every other sort whose length grows with the input sorts by keys where its keys fit a u128, and by sort_runs where it compares, not sort_unstable spent after it (not measured)
        sort_runs(&mut counted, Ord::cmp, budget, Site::Trace)?;
        trace.counted(&counted);
        budget.mark(Stage::Record);
        // f4-gfp.22 design-checked -- a constant element ends the run with the basis {1}, not running until no pair is left (not measured)
        if let Some(one) = constant_made(&mut made, m, table, cof, budget)? {
            keep_step(keep.as_deref_mut(), owned, columns);
            budget.end_round(None);
            report_constant(budget, round);
            return Ok(Some(one));
        }
        let mut new = Vec::with_capacity(made.len());
        for r in made {
            let element = cof.compose(r.kept, &m.rows, table, budget)?;
            new.push((r.poly(m, budget)?, element));
        }
        budget.mark(Stage::Decode);
        let entering = (new, leads);
        enter_step(field, table, &mut basis, cof, step, entering, budget, at)?;
        if let Some(m) = keep_step(keep.as_deref_mut(), owned, columns) {
            bufs.keep(m);
        }
        budget.mark(Stage::Update);
        budget.end_round(None);
    }

    // round-memory-time.7 design-checked -- the buffers live for one run's rounds and go back where the rounds end, before the inter-reduction, not buffers that live for the call or for the thread (log 0051-1)
    drop(bufs);
    budget.check(
        || {
            table.shrink_to_fit();
            basis.shrink_to_fit();
        },
        Site::Run,
    )?;
    budget.open();
    // trace-replay.8 cited-checked: joux-2011 §2.2, Alg. 6, line 22 -- F4Remake returns InterReduce(G_min), computed in the replay
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
        // q-replay-time.11 cited-checked: berthomieu-2022a §3.4, Tracer over Q -- the computations that apply the learned data need no symbolic preprocessing
        Some(matrix) => matrix,
        None => &*owned.insert(inter_matrix(table, order, &basis, budget)?),
    };
    let made = inter_reduce(field, &basis, m, cof, trace, budget)?;
    budget.mark(Stage::Reduce);
    // q-replay-time.13 design-checked -- a replay from the shape does not check the rows its inter-reduction makes against the shape, not a check of those rows that starts the prime over in full where one differs (log 0022-2)
    let reduced = reduced_elements(table, order, m, made, cof, budget)?;
    budget.mark(Stage::Decode);
    report_inter(budget, m, reduced.len());
    if let Some(built) = keep
        && let Some(mut matrix) = owned
    {
        // q-replay-time.14 design-checked -- a kept matrix gives back the slack of its arrays when it is kept, and the shape's table when the shape is put, not the shape kept as built (log 0022-2)
        matrix.shrink_to_fit();
        built.inter = Some(matrix);
    }
    Ok(Some(reduced))
}

/// Enters the elements of `step`, the record's step `at`, which builds no
/// matrix: each a skipped row's leading monomial (`enter_step`).
fn enter_unbuilt<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    table: &mut Table<M>,
    basis: &mut Basis<F, D>,
    cof: &mut C,
    step: &Step,
    budget: &mut Budget,
    at: u32,
) -> Result<(), Halt> {
    let mut leads = Vec::with_capacity(step.entered.len());
    for e in step.leads.chunks_exact(table.format().variables()) {
        leads.push(table.intern(e, budget)?);
    }
    let entering = (Vec::new(), &leads[..]);
    enter_step(field, table, basis, cof, step, entering, budget, at)
}

/// Enters the elements of `step` into `basis` in the order they entered
/// the learn run's: of `entering`, the made rows and the step's lead ids,
/// a made row at a position below the made rows' count, and past it the
/// skipped row's leading monomial at that position (`enter_skipped`).
/// `Halt::Misfit(at)`, `at` the step's place in the record counted from 1,
/// at a position taken twice or past the step's leads.
#[allow(
    clippy::too_many_arguments,
    reason = "the field, table, basis, and cofactors the step enters into, the step and what it made, the budget, and the step a misfit names, none of which belongs with another"
)]
fn enter_step<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    table: &Table<M>,
    basis: &mut Basis<F, D>,
    cof: &mut C,
    step: &Step,
    entering: (Elements<F, D, C::Element>, &[MonoId]),
    budget: &mut Budget,
    at: u32,
) -> Result<(), Halt> {
    let (made, leads) = entering;
    let mut made: Vec<_> = made.into_iter().map(Some).collect();
    for &k in &step.entered {
        let k = k as usize;
        if k >= made.len() {
            let Some(&lead) = leads.get(k) else {
                return Err(Halt::Misfit(at));
            };
            enter_skipped(field, table, basis, cof, lead, budget)?;
            continue;
        }
        let Some((h, element)) = made.get_mut(k).and_then(Option::take) else {
            return Err(Halt::Misfit(at));
        };
        basis.push_min(table, h, budget)?;
        cof.push(element);
    }
    Ok(())
}

// q-replay-reduction-time.4 design-checked -- a skipped counted row's element enters the replay's basis as its leading monomial alone, coefficient 1, with the cofactors of the empty sum, so that each later element keeps its id and G_min is the learn run's, not the record's element ids renumbered without the skipped elements
/// Enters a skipped row's element into `basis`: the monomial `lead` alone,
/// coefficient 1, and the cofactors of the empty sum. No row of a later
/// step multiplies it, and it leaves `G_min` as the learn run's.
fn enter_skipped<F: Field, M: Format, D: Module, C: Cofactors<F, M>>(
    field: &F,
    table: &Table<M>,
    basis: &mut Basis<F, D>,
    cof: &mut C,
    lead: MonoId,
    budget: &mut Budget,
) -> Result<(), Halt> {
    let h = Poly {
        monomials: vec![lead],
        coefs: vec![field.coef(1)],
        position: D::Position::default(),
    };
    basis.push_min(table, h, budget)?;
    let element = cof.lift(&[], &[], table, budget)?;
    cof.push(element);
    Ok(())
}

// call-stop-latency.60 design-checked -- a monomial interned from its exponents spends MONOMIAL_PASSES, 3, units a variable and PROBE, its exponents written, encoded, and hashed and its slot probed, not a unit a term (log 0138-6)
/// The matrix of `step`, its rows multiplied out from the record
/// (`matrix::build_rows`) into `cols`, and the ids of the step's
/// leads: the monomials its counted rows must reduce to, then its skipped
/// rows' leading monomials (`Step::leads`). Each multiplier and lead
/// interned from the record's exponents spends `table::monomial_units`,
/// `RUN` at a time, at `Site::Products`. Overflow where a product passes
/// the format, a stop where the budget's check stops the call.
fn build_step<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &mut Table<M>,
    basis: &Basis<F, D>,
    order: &O,
    step: &Step,
    bufs: &mut Buffers<F>,
    budget: &mut Budget,
) -> Result<(Matrix, Vec<MonoId>), Halt> {
    let n = table.format().variables();
    let mut rows = Vec::with_capacity(step.elements.len());
    let multipliers = step.multipliers.chunks(RUN * n);
    for (elements, exponents) in step.elements.chunks(RUN).zip(multipliers) {
        for (&element, e) in elements.iter().zip(exponents.chunks_exact(n)) {
            rows.push(RowSource {
                element,
                multiplier: table.intern(e, budget)?,
            });
        }
        budget.spend(elements.len() * monomial_units(n), Site::Products)?;
    }
    let m = matrix::build_rows(table, basis, order, rows, bufs, budget)?;
    let mut leads = Vec::with_capacity(step.entered.len());
    for exponents in step.leads.chunks(RUN * n) {
        let before = leads.len();
        for e in exponents.chunks_exact(n) {
            leads.push(table.intern(e, budget)?);
        }
        budget.spend((leads.len() - before) * monomial_units(n), Site::Products)?;
    }
    Ok((m, leads))
}

/// Whether the rows `made` have the columns `columns`, row by row.
fn same_columns<F: Field>(made: &[Reduced<F>], columns: &[Vec<MonoId>]) -> bool {
    made.iter().map(|r| &r.cols).eq(columns)
}

// q-replay-time.2 design-checked -- the shape comes from the first replay of a record, which keeps the matrices it builds, not from the learn run's matrices with the rows the record leaves out taken out of them (log 0022-3)
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
            // q-replay-time.14 design-checked -- a kept matrix gives back the slack of its arrays when it is kept, and the shape's table when the shape is put, not the shape kept as built (log 0022-2)
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

/// Units of an S-pair the signature run's entering of an element considers
/// (`SigRun::enter`), at its slowest case: its reads at random places,
/// `PROBE` units each, of the earlier element's lead by the base divisor's
/// test, of the three monomials its signature is formed from, of the
/// formed signature's slot in the table, and of the two leads' masks by the
/// test of coprimality.
const PAIR: usize = 6 * PROBE;

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
    // f4-gfp.49 design-checked -- inside a matrix a memory check gives back nothing of the run's own and between rounds the run shrinks its growable arrays, not the dense row dropped inside a matrix and allocated again for the next row (not measured)
    /// Gives back what the run can rebuild between two rounds, the slack of
    /// its growable arrays and its matrix buffers, and the triangle, which it
    /// can do without.
    fn release<M: Format>(&mut self, table: &mut Table<M>) {
        table.shrink_to_fit();
        self.basis.shrink_to_fit();
        self.sigs.shrink_to_fit();
        self.queue.shrink_to_fit();
        self.bufs.release();
        // signature-pair-time.8 cited-checked: roune-2012 Appendix B.2 -- if the computer runs out of memory when using the base divisor technique, then it is possible to simply drop the triangle of bits and stop using the base divisor technique from that point onward
        self.triangle = None;
    }

    /// Enters the monic element `h` with the signature `sig` into the basis,
    /// its cofactors into the tracking, and the signatures of its S-pairs
    /// into the queue, each S-pair counted in `pairs`. Overflow where a
    /// signature that is no Koszul signature and that no base divisor
    /// eliminates passes the format; Size where the basis or the table is
    /// full. The S-pairs with the
    /// earlier elements spend `PAIR` units and the format's words a pair at
    /// `Site::SigEnter` after each `RUN`, the searches of the known syzygy
    /// signatures their own at `Site::Antichain` and the tests against the
    /// inputs' Koszul signatures theirs at `Site::Koszul`, and the entering
    /// into the basis, the signatures, the triangle, and the base divisors
    /// their own; a stop where the budget's check stops the call.
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
        let new = self.basis.push_min(table, h, budget)?.id;
        cof.push(element);
        self.sigs.push(table, order, &self.basis, sig, budget)?;
        let divisors = match self.triangle.as_mut() {
            Some(triangle) => {
                triangle.grow(new.index() + 1, budget)?;
                Some(BaseDivisors::of(
                    table,
                    &self.basis,
                    &self.sigs,
                    new,
                    budget,
                )?)
            }
            None => None,
        };
        // call-stop-latency.19 design-checked -- the entering of an element spends PAIR units and the format's words per earlier element it pairs, RUN pairs at a time, and the known syzygy signatures each test scans, as it goes, not one tick per element entered (signature-pruning.22) (log 0138-1, 0138-5)
        let units = PAIR + table.format().words();
        // signature-pruning.15 cited-checked: lairez-2022 §5.6.1, Algorithm 5, line 60 -- for h ∈ G: Q ← Q ∪ Σ(g, h) ∪ Σ(h, g)
        for start in (0..new.index()).step_by(RUN) {
            let end = (start + RUN).min(new.index());
            for g in (start..end).map(ElementId::at) {
                pairs.considered += 1;
                if let (Some(divisors), Some(triangle)) = (&divisors, &mut self.triangle) {
                    // signature-pair-time.3 cited-checked: roune-2012 §3.5 and Appendix B.2, Theorems 18 and 19 -- the base divisor criterion is strictly weaker than the signature criterion, but it is faster to check; for yang1 the base divisors are a 35% performance improvement, and they eliminate 71% of the S-pairs
                    let base =
                        divisors.eliminates(table, &self.basis, &self.sigs, triangle, new, g);
                    if let Some(base) = base {
                        match base {
                            Base::High => pairs.high += 1,
                            Base::Low => pairs.low += 1,
                        }
                        triangle.set(g, new);
                        continue;
                    }
                }
                // signature-pair-time.11 design-checked -- an S-pair a base divisor eliminates is not formed, so its signature passing the format's degree bound does not halt the run (not measured)
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
                    // gf2-monomial-time.9 design-checked -- an S-pair whose signature passes the format halts the signature run, and the call starts over in the next width, not the S-pair kept until its row is formed, as the plain run keeps a pair past the bound until its selection (log 0136-1, 0136-2)
                    Err(Overflow) => return Err(Halt::Overflow),
                };
                // signature-pair-time.6 design-checked -- a pair's bit is set when its S-pair is made and a base divisor, the signature criterion, or coprime leading monomials eliminate it, the last where the format holds its signature; a queued S-pair keeps its bit unset (not measured)
                // signature-pruning.6 cited-checked: eder-2017 §6.1, Lemma (syzygy criterion) -- if S is a syzygy signature and S divides T, T is one too; roune-2012 §B.1 -- it is better to eliminate an S-pair early, as it is created, than late, right before its reduction, since one eliminated late is stored and compared with other S-pairs in the meantime
                // A coprime S-pair whose signature a known syzygy signature
                // divides goes here too, and the known ones stay the same.
                if self.sigs.is_syzygy_formed(table, &formed, budget)? {
                    pairs.syzygy += 1;
                    if let Some(triangle) = &mut self.triangle {
                        triangle.set(g, new);
                    }
                    continue;
                }
                // signature-pair-time.2 design-checked -- a formed signature is interned only where the run keeps it, queued or entered as a Koszul syzygy signature (log 0026-1)
                let s = Sig {
                    monomial: table.intern_formed(formed.degree, budget)?,
                    index: formed.index,
                };
                if table.coprime(self.basis.lead(g), lead) {
                    // signature-pruning.5 cited-checked: roune-2012 §3.3 -- with relatively prime lead terms the S-pair's signature is the Koszul syzygy's, the relatively prime criterion
                    // gfp-unsolved.14 cited-checked: roune-2012 Appendix B.1 -- an S-pair is eliminated early, when made, or late, right before its reduction; trying to eliminate them all every time new information comes in would take a lot of time
                    pairs.koszul += 1;
                    if let Some(triangle) = &mut self.triangle {
                        triangle.set(g, new);
                    }
                    // bivium-rest-gap.5, .6: the formed signature's search
                    // above found no divisor.
                    #[cfg(test)]
                    if KOSZUL_SEARCHED.get() {
                        self.sigs.push_syzygy(table, s, budget)?;
                        continue;
                    }
                    self.sigs.push_coprime(table, s);
                    continue;
                }
                pairs.queued += 1;
                self.queue.push(table, s);
            }
            budget.spend((end - start) * units, Site::SigEnter)?;
        }
        Ok(())
    }

    /// Takes the pending signatures of least degree out of the queue: the
    /// reductant of each, in increasing order of signature, where the
    /// signature is no known syzygy signature and a multiple of smaller
    /// signature top-reduces the reductant; and how many signatures it took.
    /// The queue spends its selection, and each signature taken its
    /// reductant's multiple and lead, `matrix::PRODUCT` units each, at
    /// `Site::Queue`, its reductant search its own, and its reducer search a
    /// unit per block of the basis's masks and `PROBE` units per candidate
    /// it tests (`table::search_units`), at `Site::Reducer`; a stop where the
    /// budget's check stops the call.
    fn reductants<M: Format, O: Order<M>>(
        &mut self,
        table: &mut Table<M>,
        order: &O,
        budget: &mut Budget,
    ) -> Result<(Vec<(RowSource, Sig)>, usize), Limit> {
        let mut given = Vec::new();
        // call-stop-latency.21 design-checked -- the queue's selection spends the pending signatures it visits, a round's alone, not unspent after the round's check (log 0138-5)
        let popped = self.queue.pop_least(table, order, budget)?;
        let taken = popped.len();
        for s in popped {
            if self.sigs.is_syzygy(table, s, budget)? {
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
                .reductant(table, order, s, budget)?
                .expect("an element's signature divides a pending one");
            let a = table.div(s.monomial, self.sigs.sig(g).monomial, budget)?;
            let lead = table.mul(a, self.basis.lead(g), budget)?;
            budget.spend(2 * matrix::PRODUCT, Site::Queue)?;
            let (found, tested) = self.sigs.reducer(table, lead);
            #[allow(
                clippy::expect_used,
                reason = "g's own lead divides a·lead(g), and g is among the reducers"
            )]
            let r = found.expect("g's leading monomial divides its multiple's");
            budget.spend(search_units(self.sigs.len(), tested), Site::Reducer)?;
            // signature-pruning.9 cited-checked: lairez-2022 §5.7, Algorithm 8, line 94 -- if f is →G-reducible then it joins the reductants
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
