//! The trace contract: a run tells its trace which rows each matrix holds
//! and which of them counted. `NoTrace` hears nothing, and with it the hooks
//! compile to nothing.
//!
//! The trace mode: `grobx::learn` takes a `Record` of the rows that counted
//! and the pivot rows they need, which `grobx::replay` replays over another prime. A run tells the record
//! through the crate's `Learner`, and with `NoLearner` it records nothing.
//! A lex record whose basis FGLM changed holds the grevlex run's rows.

use crate::field::Field;
use crate::id::{ElementId, MonoId, RowId};
use crate::matrix::{Matrix, Pivot, Reduction};
use crate::module::Module;
use crate::monomial::Format;
use crate::narrow::Narrow;
use crate::order::{MonomialOrder, Order};
use crate::poly::Poly;
use crate::table::Table;

/// Where a row of a matrix comes from: the basis element `element` times the
/// monomial `multiplier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSource {
    /// A basis element.
    pub element: ElementId,
    /// A monomial of the run's table.
    pub multiplier: MonoId,
}

// f4-gfp.39 design -- the trace hooks are two calls per matrix, its rows' sources once built and the rows that made a new element once reduced, not no hooks until trace-replay (not measured)
/// What a run tells its trace, for every matrix it reduces: the rounds'
/// and the inter-reduction's.
pub trait Trace {
    // gfp-unsolved.6 cited: berthomieu-2021 §4.3, Remark 4 -- with the tracer the probabilistic linear algebra cannot be used in the learn run, since it could not detect which specific rows reduce to zero; berthomieu-2022a §3.4, Tracer over Q -- once nothing more is learned, probabilistic linear algebra accelerates the step
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

// trace-replay.4 design -- the record names each multiplier and each expected leading monomial by its exponents, one u32 per variable, not by monomial ids of the learn run's table copied into the record (not measured)
/// What a run in trace mode recorded: per matrix that made a new element,
/// the rows a replay builds. Learned by `grobx::learn`, replayed by
/// `grobx::replay`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    // order-family.15 design -- a record holds the order it was learned in, and a replay under another order does not fit, not a replay that runs in the record's order whatever the call names (not measured)
    /// The order the learn call named.
    pub(crate) order: MonomialOrder,
    /// The instance's variables.
    pub(crate) variables: usize,
    /// Per generator, the exponents of its leading monomial, None where it
    /// is zero.
    pub(crate) inputs: Vec<Option<Vec<u32>>>,
    // trace-replay.12 design -- the record holds the order in which the inputs and each step's new elements entered the basis, not a replay that sorts them as the learn run's engine does (not measured)
    /// The nonzero generators by index, in the order they entered the
    /// basis.
    pub(crate) entered: Vec<u32>,
    pub(crate) steps: Vec<Step>,
    /// Whether FGLM changed the basis of the run the steps record.
    pub(crate) changed: bool,
    // gfp-unsolved.11 design -- a record says whether its run inter-reduced the new rows, and a replay inter-reduces where it says so, not a replay that inter-reduces where the order is not graded (not measured)
    /// Whether the learn run inter-reduced each step's new rows: the plain
    /// run's in every order, not the signature run's.
    pub(crate) inter_reduced: bool,
    // input-redundancy-time.8 design -- the echelon is a parameter of the call, on by default, that a run follows where it tracks no cofactors, a replay as its record says, and the lift turns off for all its runs where the call asks for cofactors or the checker, not each run deciding by its own cofactor tracking (log 0068-1)
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

    // fglm-lex.15 cited: berthomieu-2021 §4.5 -- (i) run the F4 tracer modulo p₀, (ii) run fglm on the computed Gröbner basis, repeated for several primes applying the tracer learnt from p₀; berthomieu-2022 §6.3 -- F4 with a tracer for the computation of G_drl modulo each prime, then the change of order step
    /// Whether the record's steps are a grevlex run whose basis FGLM
    /// changed to the record's order: a lex record of a zero-dimensional
    /// ideal of degree at most `fglm::MAX_STAIRCASE`.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.changed
    }
}

/// One matrix of the learn run, as a replay builds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Step {
    /// Per row, in the order the learn run's reduction took the rows, its
    /// basis element.
    pub(crate) elements: Vec<ElementId>,
    /// Per row, the exponents of its multiplier, `variables` each.
    pub(crate) multipliers: Vec<u32>,
    /// The counted rows, increasing, which is the order reduced. Every
    /// other row is a pivot from its place on.
    pub(crate) reductees: Vec<RowId>,
    /// Per counted row, the exponents of the leading monomial it reduced
    /// to, `variables` each, in the order of `reductees`.
    pub(crate) leads: Vec<u32>,
    /// The new elements, as positions in `reductees`, in the order they
    /// entered the basis.
    pub(crate) entered: Vec<u32>,
}

/// The inputs' part of a record: the variables, per generator the
/// exponents of its leading monomial or None, the nonzero generators in
/// their order of entry, and whether the run inter-reduces each round's new
/// rows.
pub(crate) type Inputs = (usize, Vec<Option<Vec<u32>>>, Vec<u32>, bool);

// trace-replay.10 design -- the learn run tells the record through a type parameter of run and run_sig whose methods take a closure that builds what is recorded, not hooks added to the public Trace (not measured)
/// What a run tells the record it takes. `inputs` and `step` each get a
/// closure that builds what they record; `NoLearner` never calls it.
pub(crate) trait Learner {
    /// Whether the learner records: `Recording` true, `NoLearner` false
    /// (gfp-unsolved.6).
    const LISTENS: bool;
    /// The variables, the generators' leading monomials, and their order
    /// of entry, as `Record` holds them.
    fn inputs(&mut self, f: impl FnOnce() -> Inputs);
    /// The step of a matrix just reduced. A step with no counted row is
    /// not kept.
    fn step(&mut self, f: impl FnOnce() -> Step);
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
    fn inputs(&mut self, _: impl FnOnce() -> Inputs) {}

    #[inline(always)]
    fn step(&mut self, _: impl FnOnce() -> Step) {}

    #[inline(always)]
    fn changed(&mut self) {}

    #[inline(always)]
    fn echelon(&mut self, _: bool) {}
}

/// The trace mode on: the record so far.
pub(crate) struct Recording(Record);

impl Recording {
    pub(crate) fn new() -> Recording {
        Recording(Record {
            order: MonomialOrder::Grevlex,
            variables: 0,
            inputs: Vec::new(),
            entered: Vec::new(),
            steps: Vec::new(),
            changed: false,
            inter_reduced: false,
            echelon: false,
        })
    }

    /// The record of a call that named `order`.
    pub(crate) fn finish(mut self, order: MonomialOrder) -> Record {
        self.0.order = order;
        self.0
    }
}

impl Learner for Recording {
    const LISTENS: bool = true;

    // fglm-lex.16 design -- a run's inputs start the record afresh, its steps emptied and changed cleared, not a Learner method the entry point calls before it starts over in lex (not measured)
    fn inputs(&mut self, f: impl FnOnce() -> Inputs) {
        let (variables, inputs, entered, inter_reduced) = f();
        self.0.variables = variables;
        self.0.inputs = inputs;
        self.0.entered = entered;
        self.0.inter_reduced = inter_reduced;
        self.0.steps.clear();
        self.0.changed = false;
    }

    fn step(&mut self, f: impl FnOnce() -> Step) {
        let step = f();
        // trace-replay.9 cited: berthomieu-2022a §3.4 (1) -- the first modular computation learns which F4 matrices give new information for the basis
        if !step.reductees.is_empty() {
            self.0.steps.push(step);
        }
    }

    fn changed(&mut self) {
        self.0.changed = true;
    }

    fn echelon(&mut self, taken: bool) {
        self.0.echelon = taken;
    }
}

/// The exponents of the monomial `id`, one per variable.
fn exponents<M: Format>(table: &Table<M>, id: MonoId, out: &mut Vec<u32>) {
    // result-time.1 design -- the writers of a result read a monomial's exponents from its whole words, lane after lane, not one variable at a time through `exponent` (log 0033-1, 0042-1)
    table.exponents(id, out);
}

/// The inputs' part of the record, from the generators the run takes: as
/// parsed, or in reduced row echelon form where the run takes them so
/// (`Record::echelon`).
/// `by_lead`: the inputs enter by decreasing leading monomial, as in
/// `run`, else in the instance's order, as in `run_sig`. `inter_reduced`:
/// the run inter-reduces each round's new rows, as `run` does.
pub(crate) fn inputs<F: Field, M: Format, O: Order<M>, D: Module>(
    table: &Table<M>,
    order: &O,
    generators: &[Poly<F, D>],
    by_lead: bool,
    inter_reduced: bool,
) -> Inputs {
    let leads = generators
        .iter()
        .map(|g| {
            (!g.is_zero()).then(|| {
                let mut e = Vec::new();
                exponents(table, g.lead(), &mut e);
                e
            })
        })
        .collect();
    let mut entered: Vec<u32> = (0..generators.len().narrow())
        .filter(|&j| !generators[j as usize].is_zero())
        .collect();
    if by_lead {
        // As `enter` in f4.rs sorts the monic inputs, stably.
        entered.sort_by(|&a, &b| {
            let (a, b) = (&generators[a as usize], &generators[b as usize]);
            table.cmp(order, b.lead(), a.lead())
        });
    }
    (table.format().variables(), leads, entered, inter_reduced)
}

// trace-replay.2 cited: berthomieu-2021 §4.3 -- store all polynomials and multiples that generate the matrix, then remove those reduced to zero and the reducers only needed for them
/// The step of the matrix `m` after its reduction. `by_lead` as in
/// `inputs`, for the new elements.
pub(crate) fn step<F: Field, M: Format, O: Order<M>>(
    table: &Table<M>,
    order: &O,
    m: &Matrix,
    reduction: &Reduction<F>,
    by_lead: bool,
) -> Step {
    let made = &reduction.made;
    let mut needed = vec![false; m.rows.len()];
    let mut stack: Vec<RowId> = Vec::with_capacity(made.len());
    for r in made {
        needed[r.row.index()] = true;
        stack.push(r.row);
    }
    // trace-replay.3 design -- a pivot row is needed where it is the pivot of a column of a needed row, from the counted rows to closure, not only the pivot rows a counted row's reduction subtracted over the learn prime (not measured)
    // A `New` pivot is a counted row already.
    while let Some(r) = stack.pop() {
        for &c in m.row(r) {
            match reduction.pivot[c.index()] {
                Pivot::Row(q) if !needed[q.index()] => {
                    needed[q.index()] = true;
                    stack.push(q);
                }
                Pivot::Row(_) | Pivot::None | Pivot::New(_) => {}
            }
        }
    }
    // trace-replay.16 cited: demin-2024 §4.4 -- the apply stage mirrors the F4 computation without the redundant rows, and caches the order of rows and columns in the matrices; parisse-2013 §4, step 8, independently -- care must be taken to keep row ordering when learning is active
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
    let mut elements = Vec::with_capacity(taken.len());
    let mut multipliers = Vec::new();
    for (k, &r) in taken.iter().enumerate() {
        let row = m.rows[r.index()];
        index[r.index()] = RowId::at(k);
        elements.push(row.element);
        exponents(table, row.multiplier, &mut multipliers);
    }
    let reductees = made.iter().map(|r| index[r.row.index()]).collect();
    let lead = |k: u32| m.monomials[made[k as usize].cols[0].index()];
    let mut leads = Vec::with_capacity(made.len() * table.format().variables());
    for k in 0..made.len().narrow() {
        exponents(table, lead(k), &mut leads);
    }
    let mut entered: Vec<u32> = (0..made.len().narrow()).collect();
    if by_lead {
        // As `enter` in f4.rs sorts the new elements, stably.
        entered.sort_by(|&a, &b| table.cmp(order, lead(b), lead(a)));
    }
    Step {
        elements,
        multipliers,
        reductees,
        leads,
        entered,
    }
}
