//! The trace contract: a run tells its trace which rows each matrix holds
//! and which of them counted. `NoTrace` hears nothing, and with it the hooks
//! compile to nothing.
//!
//! The trace mode: `grobx::learn` takes a `Record` of the rows that counted
//! and the pivot rows they need, which `grobx::replay` replays over another prime. A run tells the record
//! through the crate's `Learner`, and with `NoLearner` it records nothing.
//! A lex record whose basis FGLM changed holds the grevlex run's rows. The
//! lift's records keep only the counted rows the learn run's output
//! depends on (`cut`), and their replays skip the others.

use crate::Stop;
use crate::budget::{Budget, PROBE, Site};
use crate::columns::by_width;
use crate::field::Field;
pub use crate::id::RowSource;
use crate::id::{Column, ElementId, MonoId, RowId};
use crate::matrix::{Matrix, Pivot, Reduction};
use crate::module::Module;
use crate::monomial::Format;
use crate::narrow::Narrow;
use crate::order::{MonomialOrder, Order};
use crate::pairs;
use crate::poly::Poly;
use crate::sort::{RUN, sort_runs};
use crate::table::Table;

// f4-gfp.39 design-checked -- the trace hooks are two calls per matrix, its rows' sources once built and the rows that made a new element once reduced, not no hooks until trace-replay (not measured)
/// What a run tells its trace, for every matrix it reduces: the rounds'
/// and the inter-reduction's.
pub trait Trace {
    // gfp-unsolved.6 cited-checked:berthomieu-2021 §4.3, Remark 4 -- with the tracer the probabilistic linear algebra cannot be used in the learn run, since it could not detect which specific rows reduce to zero; berthomieu-2022a §3.4, Tracer over Q -- once nothing more is learned, probabilistic linear algebra accelerates the step
    /// Whether the trace listens. A run whose trace listens reduces each
    /// row of its matrices on its own, so that `counted` names rows. True
    /// unless a trace says otherwise.
    const LISTENS: bool = true;
    /// A matrix is built: the source of each row, in row order.
    fn matrix(&mut self, rows: &[RowSource]);
    /// The last matrix is reduced: the rows whose reduction made a new
    /// element, increasing. For the inter-reduction's matrix these are the
    /// elements of `G_min`.
    fn counted(&mut self, rows: &[RowId]);
}

/// The trace off.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoTrace;

impl Trace for NoTrace {
    const LISTENS: bool = false;

    #[inline(always)]
    fn matrix(&mut self, _: &[RowSource]) {}

    #[inline(always)]
    fn counted(&mut self, _: &[RowId]) {}
}

// trace-replay.4 design-checked -- the record names each multiplier and each expected leading monomial by its exponents, one u32 per variable, not by monomial ids of the learn run's table copied into the record (not measured)
/// What a run in trace mode recorded: per matrix that made a new element,
/// the rows a replay builds. Learned by `grobx::learn`, replayed by
/// `grobx::replay`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    // order-family.15 design-checked -- a record holds the order it was learned in, and a replay under another order does not fit, not a replay that runs in the record's order whatever the call names (not measured)
    /// The order the learn call named.
    pub(crate) order: MonomialOrder,
    /// The instance's variables.
    pub(crate) variables: usize,
    /// Per generator, the exponents of its leading monomial, None where it
    /// is zero.
    pub(crate) inputs: Vec<Option<Vec<u32>>>,
    // trace-replay.12 design-checked -- the record holds the order in which the inputs and each step's new elements entered the basis, not a replay that sorts them as the learn run's engine does (not measured)
    /// The nonzero generators by index, in the order they entered the
    /// basis.
    pub(crate) entered: Vec<u32>,
    pub(crate) steps: Vec<Step>,
    /// Whether FGLM changed the basis of the run the steps record.
    pub(crate) changed: bool,
    // gfp-unsolved.11 design-checked -- a record says whether its run inter-reduced the new rows, and a replay inter-reduces where it says so, not a replay that inter-reduces where the order is not graded (not measured)
    /// Whether the learn run inter-reduced each step's new rows: the plain
    /// run's in every order, not the signature run's.
    pub(crate) inter_reduced: bool,
    // input-redundancy-time.8 design-checked -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
    /// Whether the learn run took the generators in reduced row echelon
    /// form (`echelon::echelon`).
    pub(crate) echelon: bool,
}

impl Record {
    /// The matrices a replay builds before its inter-reduction.
    #[must_use]
    pub fn matrices(&self) -> usize {
        self.steps.len()
    }

    /// The rows of those matrices.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.steps.iter().map(|s| s.elements.len()).sum()
    }

    /// The rows of those a replay reduces: the counted rows.
    #[must_use]
    pub fn reduced(&self) -> usize {
        self.steps.iter().map(|s| s.reductees.len()).sum()
    }

    /// The counted rows of the learn run a replay skips (`cut`).
    pub(crate) fn skipped(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.entered.len() - s.reductees.len())
            .sum()
    }

    // fglm-lex.15 design-checked -- a record of a lex call whose basis FGLM changed holds the grevlex run's steps and says so, and a replay of it replays them in grevlex and changes the replayed basis, not a lex call in trace mode that runs F4 in lex so that a lex record's steps are lex's (not measured)
    /// Whether the record's steps are a grevlex run whose basis FGLM
    /// changed to the record's order: a lex record of a zero-dimensional
    /// ideal of degree at most `fglm::MAX_STAIRCASE`.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.changed
    }
}

/// One matrix of the learn run, as a replay builds it. A step `cut` left
/// whole holds every counted row; a cut step holds the counted rows the
/// output depends on, and of each other one, skipped, its leading
/// monomial alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Step {
    /// Per row, in the order the learn run's reduction took the rows, its
    /// basis element.
    pub(crate) elements: Vec<ElementId>,
    /// Per row, the exponents of its multiplier, `variables` each.
    pub(crate) multipliers: Vec<u32>,
    /// The counted rows a replay reduces, increasing, which is the order
    /// reduced; none where the step skips every counted row. Every other
    /// row is a pivot from its place on.
    pub(crate) reductees: Vec<RowId>,
    /// Per counted row, in the order of `reductees`, the exponents of the
    /// leading monomial it reduced to, and after them, per skipped row in
    /// the learn run's order, the exponents of its leading monomial;
    /// `variables` each.
    pub(crate) leads: Vec<u32>,
    /// The new elements, as positions in `leads`, in the order they
    /// entered the basis. A position at or past `reductees.len()` names a
    /// skipped row.
    pub(crate) entered: Vec<u32>,
}

/// A step's reach: per row of the step, in the step's row order, the rows
/// of the step that hold the pivots of its columns when the learn run's
/// reduction ended, each once, in `rows[start[r]..start[r + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reach {
    pub(crate) start: Vec<u32>,
    pub(crate) rows: Vec<u32>,
}

/// What a run's reduced basis is made from once its rounds end.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Output<'a> {
    /// `G_min` of the run's basis, which the inter-reduction reduces.
    Basis(&'a [ElementId]),
    /// The constant the run's last step made.
    Constant,
}

/// The inputs' part of a record: the variables, per generator the
/// exponents of its leading monomial or None, the nonzero generators in
/// their order of entry, and whether the run inter-reduces each round's new
/// rows.
pub(crate) type Inputs = (usize, Vec<Option<Vec<u32>>>, Vec<u32>, bool);

// trace-replay.10 design-checked -- the learn run tells the record through a type parameter of run and run_sig whose methods take a closure that builds what is recorded, not hooks added to the public Trace (not measured)
/// What a run tells the record it takes. `inputs` and `step` each get a
/// closure that builds what they record, spent on the call's budget;
/// `NoLearner` never calls it. Each returns the stop where the budget's
/// check stops the call.
pub(crate) trait Learner {
    /// Whether the learner records: `Recording` true, `NoLearner` false
    /// (gfp-unsolved.6).
    const LISTENS: bool;
    /// The variables, the generators' leading monomials, and their order
    /// of entry, as `Record` holds them.
    fn inputs(&mut self, f: impl FnOnce() -> Result<Inputs, Stop>) -> Result<(), Stop>;
    /// The step of a matrix just reduced, and its reach where `f` is told
    /// that the learner keeps it. A step with no counted row is not kept.
    fn step(
        &mut self,
        f: impl FnOnce(bool) -> Result<(Step, Option<Reach>), Stop>,
    ) -> Result<(), Stop>;
    /// The run's rounds ended, and its reduced basis is made from
    /// `output`; the cut, where the learner cuts, spent on `budget`.
    fn output(&mut self, output: Output<'_>, budget: &mut Budget) -> Result<(), Stop>;
    /// The run's basis is changed to the call's order by FGLM.
    fn changed(&mut self);
    /// Whether the run takes the generators in reduced row echelon form.
    fn echelon(&mut self, taken: bool);
}

/// The trace mode off: a zero-sized type whose methods are empty.
pub(crate) struct NoLearner;

impl Learner for NoLearner {
    const LISTENS: bool = false;

    #[inline(always)]
    fn inputs(&mut self, _: impl FnOnce() -> Result<Inputs, Stop>) -> Result<(), Stop> {
        Ok(())
    }

    #[inline(always)]
    fn step(
        &mut self,
        _: impl FnOnce(bool) -> Result<(Step, Option<Reach>), Stop>,
    ) -> Result<(), Stop> {
        Ok(())
    }

    #[inline(always)]
    fn output(&mut self, _: Output<'_>, _: &mut Budget) -> Result<(), Stop> {
        Ok(())
    }

    #[inline(always)]
    fn changed(&mut self) {}

    #[inline(always)]
    fn echelon(&mut self, _: bool) {}
}

/// The trace mode on: the record so far, and where the record is cut, the
/// reach of each of its steps until the run's output decides the cut.
pub(crate) struct Recording {
    record: Record,
    cuts: bool,
    reach: Vec<Reach>,
}

impl Recording {
    /// A recording that keeps every counted row, `grobx::learn`'s.
    pub(crate) fn new() -> Recording {
        Recording {
            record: Record {
                order: MonomialOrder::Grevlex,
                variables: 0,
                inputs: Vec::new(),
                entered: Vec::new(),
                steps: Vec::new(),
                changed: false,
                inter_reduced: false,
                echelon: false,
            },
            cuts: false,
            reach: Vec::new(),
        }
    }

    // q-replay-reduction-time.6 design-checked -- the lift's learns cut their records, and grobx::learn returns its record whole, so that a replay of it checks every counted row, not every record cut
    /// A recording that cuts its record to the counted rows the run's
    /// output depends on (`cut`), the lift's.
    pub(crate) fn cutting() -> Recording {
        Recording {
            cuts: true,
            ..Recording::new()
        }
    }

    /// The record of a call that named `order`.
    pub(crate) fn finish(mut self, order: MonomialOrder) -> Record {
        self.record.order = order;
        self.record
    }
}

impl Learner for Recording {
    const LISTENS: bool = true;

    // fglm-lex.16 design-checked -- a run's inputs start the record afresh, its steps emptied and changed cleared, not a Learner method the entry point calls before it starts over in lex (not measured)
    fn inputs(&mut self, f: impl FnOnce() -> Result<Inputs, Stop>) -> Result<(), Stop> {
        let (variables, inputs, entered, inter_reduced) = f()?;
        let r = &mut self.record;
        r.variables = variables;
        r.inputs = inputs;
        r.entered = entered;
        r.inter_reduced = inter_reduced;
        r.steps.clear();
        r.changed = false;
        self.reach.clear();
        Ok(())
    }

    fn step(
        &mut self,
        f: impl FnOnce(bool) -> Result<(Step, Option<Reach>), Stop>,
    ) -> Result<(), Stop> {
        let (step, reach) = f(self.cuts)?;
        // trace-replay.9 cited-checked:berthomieu-2022a §3.4 (1) -- the first modular computation learns which F4 matrices give new information for the basis
        if !step.reductees.is_empty() {
            self.record.steps.push(step);
            if let Some(reach) = reach {
                self.reach.push(reach);
            }
        }
        Ok(())
    }

    // q-replay-reduction-time.3 design-checked -- the learn records each step's reach as it records the step and decides the cut once its rounds end, not the replay that keeps the shape finding the cut from the shape's matrices for the replays after it
    fn output(&mut self, output: Output<'_>, budget: &mut Budget) -> Result<(), Stop> {
        if self.cuts {
            cut(&mut self.record, &self.reach, output, budget)?;
        }
        self.reach = Vec::new();
        Ok(())
    }

    fn changed(&mut self) {
        self.record.changed = true;
    }

    fn echelon(&mut self, taken: bool) {
        self.record.echelon = taken;
    }
}

/// The exponents of the monomial `id`, one per variable.
fn exponents<M: Format>(table: &Table<M>, id: MonoId, out: &mut Vec<u32>) {
    // result-time.1 design-checked -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
    table.exponents(id, out);
}

/// The inputs' part of the record, from the generators the run takes: as
/// parsed, or in reduced row echelon form where the run takes them so
/// (`Record::echelon`).
/// `by_lead`: the inputs enter in `pairs::entering`'s order, as in
/// `run`, else in the instance's order, as in `run_sig`. `inter_reduced`:
/// the run inter-reduces each round's new rows, as `run` does. Spent at
/// `Site::Trace`: per generator `PROBE` units and its lead's exponents a
/// unit each, `sort::RUN` generators at a time, and their sort into their
/// order of entry. A stop where the budget's check stops the call.
pub(crate) fn inputs<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &Table<M>,
    order: &O,
    generators: &[Poly<F, D>],
    by_lead: bool,
    inter_reduced: bool,
    budget: &mut Budget,
) -> Result<Inputs, Stop> {
    let n = table.format().variables();
    let mut leads = Vec::with_capacity(generators.len());
    for chunk in generators.chunks(RUN) {
        leads.extend(chunk.iter().map(|g| {
            (!g.is_zero()).then(|| {
                let mut e = Vec::new();
                exponents(table, g.lead(), &mut e);
                e
            })
        }));
        budget.spend(chunk.len() * (PROBE + n), Site::Trace)?;
    }
    let mut entered: Vec<u32> = (0..generators.len().narrow())
        .filter(|&j| !generators[j as usize].is_zero())
        .collect();
    if by_lead {
        // As `enter` in f4.rs sorts the monic inputs, stably.
        sort_by_entering::<F, _, _>(
            table,
            order,
            &mut entered,
            |j| generators[j as usize].lead(),
            budget,
        )?;
    }
    Ok((n, leads, entered, inter_reduced))
}

/// Sorts `places` stably by `pairs::entering`'s order of the leading
/// monomials `lead` gives, ties by place, by `sort_runs` at `Site::Trace`.
/// A stop where the budget's check stops the call.
fn sort_by_entering<F: Field, M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    places: &mut [u32],
    lead: impl Fn(u32) -> MonoId,
    budget: &mut Budget,
) -> Result<(), Stop> {
    sort_runs(
        places,
        |&a, &b| pairs::entering::<F, _, _>(table, order, lead(a), lead(b)).then(a.cmp(&b)),
        budget,
        Site::Trace,
    )
}

// trace-replay.2 cited-checked:berthomieu-2021 §4.3 -- store all polynomials and multiples that generate the matrix, then remove those reduced to zero and the reducers only needed for them
// call-stop-latency.27 design-checked -- the learn's record, its inputs, steps, reaches, cut, and rewrites, spends a unit per row, entry, and exponent it visits and the sorts' own, as it goes, not unspent between the round's spends (log 0138-6)
/// The step of the matrix `m` after its reduction, and its reach where
/// `reach` holds. `by_lead` as in `inputs`, for the new elements. Spent at
/// `Site::Trace`: the flags and the index a unit a row once made, each
/// row it visits a unit and its entries a unit each, each taken row's and
/// each made row's monomial `PROBE` units and its exponents a unit each,
/// and the sort of the new elements. A stop where the budget's check stops
/// the call.
pub(crate) fn step<F: Field, M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    m: &Matrix,
    reduction: &Reduction<F>,
    by_lead: bool,
    reach: bool,
    budget: &mut Budget,
) -> Result<(Step, Option<Reach>), Stop> {
    let n = table.format().variables();
    let made = &reduction.made;
    let mut needed = vec![false; m.rows.len()];
    let mut stack: Vec<RowId> = Vec::with_capacity(made.len());
    for r in made {
        needed[r.row.index()] = true;
        stack.push(r.row);
    }
    budget.spend(m.rows.len() + made.len(), Site::Trace)?;
    // trace-replay.3 design-checked -- a pivot row is needed where it is the pivot of a column of a needed row, from the counted rows to closure, not only the pivot rows a counted row's reduction subtracted over the learn prime (not measured)
    // A `New` pivot is a counted row already.
    while let Some(r) = stack.pop() {
        let entries = by_width!(m.row(r), |cols| {
            for &c in cols {
                match reduction.pivot[c.column()] {
                    Pivot::Row(q) if !needed[q.index()] => {
                        needed[q.index()] = true;
                        stack.push(q);
                    }
                    Pivot::Row(_) | Pivot::None | Pivot::New(_) => {}
                }
            }
            cols.len()
        });
        budget.spend(1 + entries, Site::Trace)?;
    }
    // trace-replay.16 cited-checked:demin-2024 §4.4 -- the apply stage mirrors the F4 computation without the redundant rows, and caches the order of rows and columns in the matrices; parisse-2013 §4, step 8, independently -- care must be taken to keep row ordering when learning is active
    // The needed rows in the order the reduction took them. Where it names
    // none, it set every pivot row first, in row order, and then reduced
    // the counted rows in the order of `made`.
    let taken: Vec<RowId> = if reduction.taken.is_empty() {
        for r in made {
            needed[r.row.index()] = false;
        }
        let pivots = (0..m.rows.len())
            .map(RowId::at)
            .filter(|r| needed[r.index()]);
        pivots.chain(made.iter().map(|r| r.row)).collect()
    } else {
        let taken = reduction.taken.iter().copied();
        taken.filter(|r| needed[r.index()]).collect()
    };
    // Each row's place in the step: its row in the replay's matrix.
    let mut index = vec![RowId::NONE; m.rows.len()];
    budget.spend(made.len() + 2 * m.rows.len(), Site::Trace)?;
    let mut elements = Vec::with_capacity(taken.len());
    let mut multipliers = Vec::new();
    for (j, chunk) in taken.chunks(RUN).enumerate() {
        for (i, &r) in chunk.iter().enumerate() {
            let row = m.rows[r.index()];
            index[r.index()] = RowId::at(j * RUN + i);
            elements.push(row.element);
            exponents(table, row.multiplier, &mut multipliers);
        }
        budget.spend(chunk.len() * (PROBE + n), Site::Trace)?;
    }
    let reductees = made.iter().map(|r| index[r.row.index()]).collect();
    let lead = |k: u32| m.monomials[made[k as usize].cols[0].index()];
    let mut leads = Vec::with_capacity(made.len() * n);
    for start in (0..made.len()).step_by(RUN) {
        let end = (start + RUN).min(made.len());
        for k in start..end {
            exponents(table, lead(k.narrow()), &mut leads);
        }
        // Each made row's place and its lead's exponents.
        budget.spend((end - start) * (1 + PROBE + n), Site::Trace)?;
    }
    let mut entered: Vec<u32> = (0..made.len().narrow()).collect();
    if by_lead {
        // As `enter` in f4.rs sorts the new elements, stably.
        sort_by_entering::<F, _, _>(table, order, &mut entered, lead, budget)?;
    }
    let step = Step {
        elements,
        multipliers,
        reductees,
        leads,
        entered,
    };
    let reach = if reach {
        Some(pivots(m, reduction, &taken, &index, budget)?)
    } else {
        None
    };
    Ok((step, reach))
}

// q-replay-reduction-time.2 design-checked -- a row reaches the row that holds the pivot of each of its columns when the reduction ended, a pivot row or the counted row whose reduction set a new pivot there, not the counted rows of a step kept whole where any of them is needed (log 0098-1)
// q-replay-reduction-time.3 design-checked -- the reach is a pass of its own over the taken rows, not recorded inside the pass that closes the needed rows
/// The reach of the rows `taken` of `m`, each row's place in the step at
/// its index in `index`: per taken row, the rows that hold the pivots of
/// its columns when `reduction` ended. Each row spends a unit and its
/// entries a unit each at `Site::Trace`. A stop where the budget's check
/// stops the call.
fn pivots<F: Field>(
    m: &Matrix,
    reduction: &Reduction<F>,
    taken: &[RowId],
    index: &[RowId],
    budget: &mut Budget,
) -> Result<Reach, Stop> {
    let mut start = Vec::with_capacity(taken.len() + 1);
    let mut rows: Vec<u32> = Vec::new();
    start.push(0);
    for (k, &r) in taken.iter().enumerate() {
        let entries = by_width!(m.row(r), |cols| {
            for &c in cols {
                let q = match reduction.pivot[c.column()] {
                    Pivot::Row(q) => q,
                    Pivot::New(j) => reduction.made[j.index()].row,
                    Pivot::None => continue,
                };
                let i = index[q.index()];
                debug_assert!(
                    i != RowId::NONE,
                    "the pivot row {q} of a needed row is needed"
                );
                // A row's columns hold pivots of rows of their own, one each,
                // and the pivot row of its own leading column is the row
                // itself.
                if i.index() != k {
                    rows.push(i.index().narrow());
                }
            }
            cols.len()
        });
        start.push(rows.len().narrow());
        budget.spend(1 + entries, Site::Trace)?;
    }
    Ok(Reach { start, rows })
}

// q-replay-reduction-time.1 transferred-checked: compilers, dead-code elimination -- carried into the replay's record: it keeps of each step only the counted rows the run's output depends on, G_min or the constant, through the rows of the later steps, and a replay skips the others, not every counted row replayed (log 0098-1, 0098-2)
/// Cuts `record`, whose steps' reaches are `reach`, to the counted rows
/// its run's output depends on, made from `output`: an element is needed
/// where it lies in `G_min` or a kept row of a later step multiplies it; a
/// counted row is kept where its element is needed, or under
/// `Output::Constant` where the last step reduced it to the constant, and
/// so is every row the reach of a kept row reaches. The steps are taken
/// from the last to the first, and each is rewritten to its kept rows, its
/// skipped rows' leading monomials after its counted rows' (`Step`). Spent
/// at `Site::Trace`: the elements' flags a unit each once made, and per
/// step its counted rows a unit each and, where the step made the
/// constant, their exponents a unit each, each row it keeps a unit and its
/// reach a unit an entry, its rows a unit each, and its rewrite. A stop
/// where the budget's check stops the call.
pub(crate) fn cut(
    record: &mut Record,
    reach: &[Reach],
    output: Output<'_>,
    budget: &mut Budget,
) -> Result<(), Stop> {
    debug_assert_eq!(reach.len(), record.steps.len(), "a reach per step");
    let n = record.variables;
    // The id of a step's first element: the inputs take the first ids, and
    // each step's elements follow the step's before it.
    let mut first = Vec::with_capacity(record.steps.len());
    let mut ids = record.entered.len();
    for s in &record.steps {
        first.push(ids);
        ids += s.entered.len();
    }
    let mut needed = vec![false; ids];
    let min = match output {
        Output::Basis(min) => min,
        Output::Constant => &[],
    };
    for &e in min {
        debug_assert!(e.index() < ids, "element {e} of G_min of {ids}");
        needed[e.index()] = true;
    }
    budget.spend(record.steps.len() + ids + min.len(), Site::Trace)?;
    let last = record.steps.len().checked_sub(1);
    for (s, (step, reach)) in record.steps.iter_mut().zip(reach).enumerate().rev() {
        let counted = step.reductees.len();
        let mut id = vec![0; counted];
        for (i, &j) in step.entered.iter().enumerate() {
            id[j as usize] = first[s] + i;
        }
        let constant = match output {
            Output::Constant => Some(s) == last,
            Output::Basis(_) => false,
        };
        let mut kept = vec![false; step.elements.len()];
        let mut stack = Vec::new();
        for (j, &r) in step.reductees.iter().enumerate() {
            let one = constant && step.leads[j * n..(j + 1) * n].iter().all(|&e| e == 0);
            if (needed[id[j]] || one) && !kept[r.index()] {
                kept[r.index()] = true;
                stack.push(r.index());
            }
        }
        let exponents = if constant { n } else { 0 };
        budget.spend(counted * (2 + exponents) + step.elements.len(), Site::Trace)?;
        // q-replay-reduction-time.2 design-checked -- a row reaches the row that holds the pivot of each of its columns when the reduction ended, a pivot row or the counted row whose reduction set a new pivot there, not the counted rows of a step kept whole where any of them is needed (log 0098-1)
        while let Some(r) = stack.pop() {
            let reached = &reach.rows[reach.start[r] as usize..reach.start[r + 1] as usize];
            for &q in reached {
                if !kept[q as usize] {
                    kept[q as usize] = true;
                    stack.push(q as usize);
                }
            }
            budget.spend(1 + reached.len(), Site::Trace)?;
        }
        for chunk in step.elements.chunks(RUN).zip(kept.chunks(RUN)) {
            for (&e, &k) in chunk.0.iter().zip(chunk.1) {
                if k {
                    needed[e.index()] = true;
                }
            }
            budget.spend(chunk.0.len(), Site::Trace)?;
        }
        rewrite(step, &kept, n, budget)?;
    }
    Ok(())
}

/// `step` cut to its rows `kept`, each of `n` variables (`Step`). Spent at
/// `Site::Trace`: its arrays a unit an entry once made, and each row a unit
/// and its exponents copied a unit each, `sort::RUN` rows at a time. A stop
/// where the budget's check stops the call.
fn rewrite(step: &mut Step, kept: &[bool], n: usize, budget: &mut Budget) -> Result<(), Stop> {
    debug_assert_eq!(step.entered.len(), step.reductees.len(), "a whole step");
    debug_assert_eq!(step.leads.len(), step.reductees.len() * n, "a whole step");
    let mut place = vec![RowId::NONE; kept.len()];
    budget.spend(kept.len(), Site::Trace)?;
    let mut elements = Vec::new();
    let mut multipliers = Vec::new();
    for (j, chunk) in kept.chunks(RUN).enumerate() {
        for (i, &k) in chunk.iter().enumerate() {
            let r = j * RUN + i;
            if k {
                place[r] = RowId::at(elements.len());
                elements.push(step.elements[r]);
                multipliers.extend_from_slice(&step.multipliers[r * n..(r + 1) * n]);
            }
        }
        budget.spend(chunk.len() * (1 + n), Site::Trace)?;
    }
    let counted = step.reductees.len();
    let stays = step.reductees.iter().filter(|r| kept[r.index()]).count();
    let mut reductees = Vec::with_capacity(stays);
    let mut leads = vec![0; step.leads.len()];
    let mut position = vec![0u32; counted];
    budget.spend(2 * counted + leads.len(), Site::Trace)?;
    let mut skipped = stays;
    for (start, chunk) in step.reductees.chunks(RUN).enumerate() {
        for (i, &r) in chunk.iter().enumerate() {
            let j = start * RUN + i;
            let k = if kept[r.index()] {
                reductees.push(place[r.index()]);
                reductees.len() - 1
            } else {
                skipped += 1;
                skipped - 1
            };
            position[j] = k.narrow();
            leads[k * n..(k + 1) * n].copy_from_slice(&step.leads[j * n..(j + 1) * n]);
        }
        budget.spend(chunk.len() * (1 + n), Site::Trace)?;
    }
    for j in &mut step.entered {
        *j = position[*j as usize];
    }
    budget.spend(step.entered.len(), Site::Trace)?;
    step.elements = elements;
    step.multipliers = multipliers;
    step.reductees = reductees;
    step.leads = leads;
    Ok(())
}
