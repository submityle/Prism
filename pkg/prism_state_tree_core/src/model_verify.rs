//! Bounded, deterministic verification of a compiled [`Plan`].
//!
//! This module is deliberately independent from the runtime. It treats guards
//! as unknown-but-possible, uses an explicit event alphabet for event-triggered
//! transitions, and models a path as a sequence of structural-entry and
//! transition edges. The result is useful for authoring and CI without adding
//! work to the runtime hot path.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use alloc::vec;
use alloc::vec::Vec;

use crate::model::{NodeKind, Plan, StateId, TransitionTarget, Trigger};
use crate::EventId;

/// Severity assigned to a model-verification finding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DiagnosticSeverity {
    /// The bounded analysis found a structural correctness problem.
    Error,
    /// The model is structurally valid, but the analysis found a risk or gap.
    Warning,
}

/// Stable category identifying a model-verification finding.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DiagnosticCode {
    /// A state is absent from the root-entry and transition closure.
    UnreachableState,
    /// A transition was not exercised by the finite event/path search.
    UncoveredTransition,
    /// A reachable atomic state has no transition enabled by the event alphabet.
    DeadlockCandidate,
    /// The finite eventless graph exceeds the configured cascade threshold.
    EventlessCascadeExceeded,
    /// A reachable eventless cycle can continue without an external event.
    EventlessCascadeUnbounded,
    /// The path output bound prevented exhaustive finite-path enumeration.
    PathEnumerationBound,
    /// The work-item bound prevented exhaustive verification.
    WorkBound,
    /// The diagnostic output bound prevented retaining every finding.
    DiagnosticBound,
}

impl DiagnosticCode {
    /// Returns the stable machine-readable spelling of this code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnreachableState => "STV0001",
            Self::UncoveredTransition => "STV0002",
            Self::DeadlockCandidate => "STV0003",
            Self::EventlessCascadeExceeded => "STV0004",
            Self::EventlessCascadeUnbounded => "STV0005",
            Self::PathEnumerationBound => "STV0097",
            Self::WorkBound => "STV0098",
            Self::DiagnosticBound => "STV0099",
        }
    }

    /// Returns a stable, human-readable explanation of this code.
    pub const fn summary(self) -> &'static str {
        match self {
            Self::UnreachableState => {
                "state is not reachable from root entry paths or a transition target"
            }
            Self::UncoveredTransition => {
                "transition is not exercised by the supplied finite event/path search"
            }
            Self::DeadlockCandidate => {
                "reachable atomic state has no transition enabled by the supplied event alphabet"
            }
            Self::EventlessCascadeExceeded => {
                "eventless cascade depth exceeds the configured verification threshold"
            }
            Self::EventlessCascadeUnbounded => {
                "reachable eventless transitions contain a cycle and may cascade indefinitely"
            }
            Self::PathEnumerationBound => {
                "path output capacity was reached before finite-path enumeration was exhaustive"
            }
            Self::WorkBound => "verification work capacity was reached before analysis completed",
            Self::DiagnosticBound => {
                "diagnostic output capacity was reached before all findings were retained"
            }
        }
    }
}

/// One structured finding produced by [`verify`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Stable finding category.
    pub code: DiagnosticCode,
    /// Finding severity.
    pub severity: DiagnosticSeverity,
    /// Primary state associated with the finding, when applicable.
    pub state: Option<StateId>,
    /// Related state, such as a transition target, when applicable.
    pub related_state: Option<StateId>,
    /// Global declaration order of the related transition, when applicable.
    pub transition: Option<u32>,
    /// Observed or configured depth associated with the finding, when applicable.
    pub depth: Option<usize>,
}

/// Explicit resource and search bounds for [`verify`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerificationLimits {
    /// Maximum number of work items consumed by the complete pass.
    pub max_work_items: usize,
    /// Maximum number of retained diagnostics.
    pub max_diagnostics: usize,
    /// Maximum number of retained finite paths.
    pub max_paths: usize,
    /// Maximum number of edges in one emitted finite path.
    pub max_path_depth: usize,
    /// Threshold at which an eventless cascade finding is emitted.
    pub max_eventless_cascade_depth: usize,
}

impl VerificationLimits {
    /// Creates explicit bounds for one verification pass.
    pub const fn new(
        max_work_items: usize,
        max_diagnostics: usize,
        max_paths: usize,
        max_path_depth: usize,
        max_eventless_cascade_depth: usize,
    ) -> Self {
        Self {
            max_work_items,
            max_diagnostics,
            max_paths,
            max_path_depth,
            max_eventless_cascade_depth,
        }
    }
}

impl Default for VerificationLimits {
    fn default() -> Self {
        Self::new(1_000_000, 256, 4_096, 32, 64)
    }
}

/// The kind of edge represented by one finite-path step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathStepKind {
    /// An implicit entry into a compound initial child or parallel region.
    StructuralEntry,
    /// A declared transition.
    Transition,
}

/// One deterministic edge in an emitted finite path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PathStep {
    /// Source state of this edge.
    pub source: StateId,
    /// Destination state of this edge.
    pub target: StateId,
    /// Whether this is an implicit entry edge or a declared transition.
    pub kind: PathStepKind,
    /// Declaration order of the transition, or `None` for structural entry.
    pub transition: Option<u32>,
    /// Trigger of the transition, or `None` for structural entry.
    pub trigger: Option<Trigger>,
}

/// One finite path emitted by the bounded verifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationPath {
    states: Vec<StateId>,
    steps: Vec<PathStep>,
}

impl VerificationPath {
    /// Returns all states in this path, including its initial state.
    pub fn states(&self) -> &[StateId] {
        &self.states
    }

    /// Returns the structural-entry and transition edges in this path.
    pub fn steps(&self) -> &[PathStep] {
        &self.steps
    }

    /// Returns the number of edges in this path.
    pub const fn depth(&self) -> usize {
        self.steps.len()
    }
}

/// Counts declared transitions and transitions exercised by emitted paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionCoverage {
    /// Number of declared transitions in the plan.
    pub total: usize,
    /// Number of declared transitions present in at least one emitted path.
    pub covered: usize,
}

impl TransitionCoverage {
    /// Returns whether every declared transition was exercised.
    pub const fn is_complete(self) -> bool {
        self.covered == self.total
    }
}

/// Result of one bounded, deterministic model-verification pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationReport {
    diagnostics: Vec<Diagnostic>,
    complete: bool,
    work_items: usize,
    reachable_states: Vec<StateId>,
    covered_transitions: Vec<u32>,
    uncovered_transitions: Vec<u32>,
    deadlock_candidates: Vec<StateId>,
    paths: Vec<VerificationPath>,
    transition_coverage: TransitionCoverage,
    eventless_cascade_depth: Option<usize>,
}

impl VerificationReport {
    /// Returns retained findings in deterministic emission order.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Returns whether the configured work, path, and diagnostic bounds were
    /// sufficient for an exhaustive pass within the requested finite depth.
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns the number of consumed bounded work items.
    pub const fn work_items(&self) -> usize {
        self.work_items
    }

    /// Returns states in deterministic numeric order that are structurally reachable.
    pub fn reachable_states(&self) -> &[StateId] {
        &self.reachable_states
    }

    /// Returns declaration orders of transitions exercised by emitted paths.
    pub fn covered_transitions(&self) -> &[u32] {
        &self.covered_transitions
    }

    /// Returns declaration orders of transitions not exercised by emitted paths.
    pub fn uncovered_transitions(&self) -> &[u32] {
        &self.uncovered_transitions
    }

    /// Returns reachable atomic states that have no transition enabled by the
    /// supplied event alphabet.
    pub fn deadlock_candidates(&self) -> &[StateId] {
        &self.deadlock_candidates
    }

    /// Returns all retained paths in deterministic depth-first order.
    pub fn paths(&self) -> &[VerificationPath] {
        &self.paths
    }

    /// Returns the declared-versus-exercised transition counts.
    pub const fn transition_coverage(&self) -> TransitionCoverage {
        self.transition_coverage
    }

    /// Returns the longest finite eventless cascade observed, or `None` when
    /// a reachable eventless cycle makes the cascade unbounded.
    pub const fn eventless_cascade_depth(&self) -> Option<usize> {
        self.eventless_cascade_depth
    }

    /// Returns whether at least one retained finding has the supplied code.
    pub fn contains_code(&self, code: DiagnosticCode) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code)
    }
}

/// Verifies a compiled plan with an explicit finite event alphabet and bounds.
///
/// Reachability is structural and conservative: initial compound children,
/// parallel regions, and all declared transition targets are considered
/// possible. Transition coverage and deadlock candidates use `events`; a
/// guard identity is treated as potentially true because the public `Plan`
/// API does not contain guard implementations. Paths include implicit entry
/// edges and transitions, and are enumerated up to
/// `limits.max_path_depth`.
pub fn verify(plan: &Plan, events: &[EventId], limits: VerificationLimits) -> VerificationReport {
    let mut builder = ReportBuilder::new(limits);
    let state_count = plan.state_count();
    let mut transition_refs = Vec::new();
    let mut transitions_by_source = vec![Vec::new(); state_count];
    let mut eventless_edges = vec![Vec::new(); state_count];

    for index in 0..state_count {
        if !builder.step() {
            return builder.finish(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                TransitionCoverage {
                    total: 0,
                    covered: 0,
                },
                None,
            );
        }
        let state = StateId(index as u32);
        let Some(node) = plan.node(state) else {
            builder.work_bound();
            return builder.finish(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                TransitionCoverage {
                    total: 0,
                    covered: 0,
                },
                None,
            );
        };
        for transition in &node.transitions {
            if !builder.step() {
                return builder.finish(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    TransitionCoverage {
                        total: transition_refs.len(),
                        covered: 0,
                    },
                    None,
                );
            }
            let target = transition_target(state, transition.target);
            let reference = TransitionRef {
                source: state,
                target,
                history: matches!(transition.target, TransitionTarget::History { .. }),
                declaration_order: transition.declaration_order(),
            };
            if matches!(transition.trigger, Trigger::Eventless) {
                eventless_edges[index].push(EventlessEdge {
                    target,
                    transition: reference.declaration_order,
                });
            }
            transitions_by_source[index].push(reference);
            transition_refs.push(reference);
        }
    }

    let reachable = match structural_reachability(plan, &transitions_by_source, &mut builder) {
        Some(reachable) => reachable,
        None => {
            return builder.finish(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                TransitionCoverage {
                    total: transition_refs.len(),
                    covered: 0,
                },
                None,
            )
        }
    };
    let reachable_states: Vec<_> = reachable
        .iter()
        .enumerate()
        .filter_map(|(index, reachable)| reachable.then_some(StateId(index as u32)))
        .collect();

    for (index, reachable) in reachable.iter().enumerate() {
        if !builder.step() {
            return builder.finish(
                reachable_states,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                TransitionCoverage {
                    total: transition_refs.len(),
                    covered: 0,
                },
                None,
            );
        }
        if !reachable {
            builder.push(Diagnostic {
                code: DiagnosticCode::UnreachableState,
                severity: DiagnosticSeverity::Error,
                state: Some(StateId(index as u32)),
                related_state: None,
                transition: None,
                depth: None,
            });
        }
    }

    let deadlock_candidates = find_deadlocks(plan, &reachable, events, &mut builder);
    let eventless_cascade_depth = eventless_depth(
        &eventless_edges,
        &reachable,
        limits.max_eventless_cascade_depth,
        &mut builder,
    );

    let mut covered_transitions = Vec::new();
    let paths = enumerate_paths(
        plan,
        events,
        limits,
        &reachable,
        &mut covered_transitions,
        &mut builder,
    );
    covered_transitions.sort_unstable();
    covered_transitions.dedup();

    let mut uncovered_transitions = Vec::new();
    for reference in &transition_refs {
        if covered_transitions
            .binary_search(&reference.declaration_order)
            .is_err()
        {
            uncovered_transitions.push(reference.declaration_order);
            builder.push(Diagnostic {
                code: DiagnosticCode::UncoveredTransition,
                severity: DiagnosticSeverity::Warning,
                state: Some(reference.source),
                related_state: Some(reference.target),
                transition: Some(reference.declaration_order),
                depth: None,
            });
        }
    }

    let coverage = TransitionCoverage {
        total: transition_refs.len(),
        covered: covered_transitions.len(),
    };
    builder.finish(
        reachable_states,
        covered_transitions,
        uncovered_transitions,
        deadlock_candidates,
        paths,
        coverage,
        eventless_cascade_depth,
    )
}

fn transition_target(source: StateId, target: TransitionTarget) -> StateId {
    match target {
        TransitionTarget::State(state) | TransitionTarget::History { owner: state, .. } => state,
        TransitionTarget::SelfState => source,
    }
}

fn event_is_supplied(events: &[EventId], event: EventId) -> bool {
    events.contains(&event)
}

fn structural_reachability(
    plan: &Plan,
    transitions_by_source: &[Vec<TransitionRef>],
    builder: &mut ReportBuilder,
) -> Option<Vec<bool>> {
    let count = plan.state_count();
    let mut reachable = vec![false; count];
    let mut queue = Vec::new();
    mark_reachable(plan.root(), &mut reachable, &mut queue);
    let mut cursor = 0;

    while cursor < queue.len() {
        if !builder.step() {
            return None;
        }
        let state = queue[cursor];
        cursor += 1;
        let Some(node) = plan.node(state) else {
            builder.work_bound();
            return None;
        };
        match &node.kind {
            NodeKind::Atomic => {}
            NodeKind::Compound { initial, .. } => {
                mark_reachable(*initial, &mut reachable, &mut queue);
            }
            NodeKind::Parallel { regions, .. } => {
                for region in regions {
                    mark_reachable(*region, &mut reachable, &mut queue);
                }
            }
        }
        for transition in &transitions_by_source[state.0 as usize] {
            if !builder.step() {
                return None;
            }
            mark_reachable(transition.target, &mut reachable, &mut queue);
            if transition.history {
                mark_subtree(plan, transition.target, &mut reachable, &mut queue, builder)?;
            }
        }
    }
    Some(reachable)
}

fn mark_reachable(state: StateId, reachable: &mut [bool], queue: &mut Vec<StateId>) {
    let index = state.0 as usize;
    if !reachable[index] {
        reachable[index] = true;
        queue.push(state);
    }
}

fn mark_subtree(
    plan: &Plan,
    root: StateId,
    reachable: &mut [bool],
    queue: &mut Vec<StateId>,
    builder: &mut ReportBuilder,
) -> Option<()> {
    for index in 0..plan.state_count() {
        if !builder.step() {
            return None;
        }
        let state = StateId(index as u32);
        if state != root && is_descendant(plan, state, root) {
            mark_reachable(state, reachable, queue);
        }
    }
    Some(())
}

fn is_descendant(plan: &Plan, mut state: StateId, ancestor: StateId) -> bool {
    while let Some(parent) = plan.parent(state) {
        if parent == ancestor {
            return true;
        }
        state = parent;
    }
    false
}

fn find_deadlocks(
    plan: &Plan,
    reachable: &[bool],
    events: &[EventId],
    builder: &mut ReportBuilder,
) -> Vec<StateId> {
    let mut candidates = Vec::new();
    for (index, is_reachable) in reachable.iter().enumerate().take(plan.state_count()) {
        if !builder.step() {
            break;
        }
        if !*is_reachable {
            continue;
        }
        let state = StateId(index as u32);
        let Some(node) = plan.node(state) else {
            builder.work_bound();
            break;
        };
        if !matches!(node.kind, NodeKind::Atomic) {
            continue;
        }
        let enabled = node.transitions.iter().any(|transition| {
            matches!(transition.trigger, Trigger::Eventless)
                || matches!(transition.trigger, Trigger::Event(event) if event_is_supplied(events, event))
        });
        if !enabled {
            candidates.push(state);
            builder.push(Diagnostic {
                code: DiagnosticCode::DeadlockCandidate,
                severity: DiagnosticSeverity::Warning,
                state: Some(state),
                related_state: None,
                transition: None,
                depth: None,
            });
        }
    }
    candidates
}

fn eventless_depth(
    edges: &[Vec<EventlessEdge>],
    reachable: &[bool],
    threshold: usize,
    builder: &mut ReportBuilder,
) -> Option<usize> {
    let count = edges.len();
    let mut color = vec![0_u8; count];
    let mut cycle = false;

    for start in 0..count {
        if !reachable[start] || color[start] != 0 {
            continue;
        }
        color[start] = 1;
        let mut stack = vec![(StateId(start as u32), 0_usize)];
        while let Some((state, next_edge)) = stack.last_mut() {
            if !builder.step() {
                return None;
            }
            let source = state.0 as usize;
            if *next_edge == edges[source].len() {
                color[source] = 2;
                stack.pop();
                continue;
            }
            let edge = edges[source][*next_edge];
            *next_edge += 1;
            let target = edge.target.0 as usize;
            if !reachable[target] {
                continue;
            }
            if color[target] == 1 {
                cycle = true;
                builder.push(Diagnostic {
                    code: DiagnosticCode::EventlessCascadeUnbounded,
                    severity: DiagnosticSeverity::Error,
                    state: Some(*state),
                    related_state: Some(edge.target),
                    transition: Some(edge.transition),
                    depth: None,
                });
            } else if color[target] == 0 {
                color[target] = 1;
                stack.push((edge.target, 0));
            }
        }
    }

    if cycle {
        return None;
    }

    let mut indegree = vec![0_usize; count];
    let mut reachable_count = 0;
    for source in 0..count {
        if !reachable[source] {
            continue;
        }
        reachable_count += 1;
        for edge in &edges[source] {
            if reachable[edge.target.0 as usize] {
                indegree[edge.target.0 as usize] += 1;
            }
        }
    }
    let mut ready = Vec::new();
    for index in 0..count {
        if reachable[index] && indegree[index] == 0 {
            ready.push(StateId(index as u32));
        }
    }
    let mut order = Vec::with_capacity(reachable_count);
    while let Some(state) = ready.pop() {
        if !builder.step() {
            return None;
        }
        order.push(state);
        for edge in &edges[state.0 as usize] {
            let target = edge.target.0 as usize;
            if !reachable[target] {
                continue;
            }
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(edge.target);
            }
        }
    }

    if order.len() != reachable_count {
        return None;
    }
    let mut longest = vec![0_usize; count];
    let mut maximum = 0;
    let mut exceeded = false;
    for state in order {
        for edge in &edges[state.0 as usize] {
            let target = edge.target.0 as usize;
            if !reachable[target] {
                continue;
            }
            if !builder.step() {
                return None;
            }
            let depth = longest[state.0 as usize].saturating_add(1);
            if depth > longest[target] {
                longest[target] = depth;
            }
            maximum = maximum.max(depth);
            if depth > threshold && !exceeded {
                exceeded = true;
                builder.push(Diagnostic {
                    code: DiagnosticCode::EventlessCascadeExceeded,
                    severity: DiagnosticSeverity::Warning,
                    state: Some(state),
                    related_state: Some(edge.target),
                    transition: Some(edge.transition),
                    depth: Some(depth),
                });
            }
        }
    }
    Some(maximum)
}

fn enumerate_paths(
    plan: &Plan,
    events: &[EventId],
    limits: VerificationLimits,
    reachable: &[bool],
    covered_transitions: &mut Vec<u32>,
    builder: &mut ReportBuilder,
) -> Vec<VerificationPath> {
    if limits.max_paths == 0 {
        builder.path_bound();
        return Vec::new();
    }
    let root = plan.root();
    if !reachable[root.0 as usize] {
        return Vec::new();
    }

    let mut paths = Vec::new();
    let mut states = vec![root];
    let mut steps = Vec::new();
    let mut frames = vec![PathFrame {
        state: root,
        next_entry: 0,
        next_transition: 0,
    }];
    paths.push(VerificationPath {
        states: states.clone(),
        steps: steps.clone(),
    });

    while let Some(frame) = frames.last_mut() {
        if !builder.step() {
            break;
        }
        if steps.len() >= limits.max_path_depth {
            frames.pop();
            if !steps.is_empty() {
                steps.pop();
                states.pop();
            }
            continue;
        }
        let Some(next) = next_successor(plan, events, reachable, frame) else {
            frames.pop();
            if !steps.is_empty() {
                steps.pop();
                states.pop();
            }
            continue;
        };
        if paths.len() >= limits.max_paths {
            builder.path_bound();
            break;
        }
        if let Some(transition) = next.transition {
            covered_transitions.push(transition);
        }
        steps.push(next.step);
        states.push(next.step.target);
        paths.push(VerificationPath {
            states: states.clone(),
            steps: steps.clone(),
        });
        frames.push(PathFrame {
            state: next.step.target,
            next_entry: 0,
            next_transition: 0,
        });
    }
    paths
}

fn next_successor(
    plan: &Plan,
    events: &[EventId],
    reachable: &[bool],
    frame: &mut PathFrame,
) -> Option<Successor> {
    let node = plan.node(frame.state)?;
    let entry_targets: &[StateId] = match &node.kind {
        NodeKind::Atomic => &[],
        NodeKind::Compound { initial, .. } => core::slice::from_ref(initial),
        NodeKind::Parallel { regions, .. } => regions,
    };
    while frame.next_entry < entry_targets.len() {
        let target = entry_targets[frame.next_entry];
        frame.next_entry += 1;
        if reachable[target.0 as usize] {
            return Some(Successor {
                transition: None,
                step: PathStep {
                    source: frame.state,
                    target,
                    kind: PathStepKind::StructuralEntry,
                    transition: None,
                    trigger: None,
                },
            });
        }
    }
    while frame.next_transition < node.transitions.len() {
        let transition = &node.transitions[frame.next_transition];
        frame.next_transition += 1;
        let enabled = match transition.trigger {
            Trigger::Eventless => true,
            Trigger::Event(event) => event_is_supplied(events, event),
        };
        if !enabled {
            continue;
        }
        let target = transition_target(frame.state, transition.target);
        if !reachable[target.0 as usize] {
            continue;
        }
        return Some(Successor {
            transition: Some(transition.declaration_order()),
            step: PathStep {
                source: frame.state,
                target,
                kind: PathStepKind::Transition,
                transition: Some(transition.declaration_order()),
                trigger: Some(transition.trigger),
            },
        });
    }
    None
}

#[derive(Clone, Copy, Debug)]
struct TransitionRef {
    source: StateId,
    target: StateId,
    history: bool,
    declaration_order: u32,
}

#[derive(Clone, Copy, Debug)]
struct EventlessEdge {
    target: StateId,
    transition: u32,
}

#[derive(Clone, Copy, Debug)]
struct PathFrame {
    state: StateId,
    next_entry: usize,
    next_transition: usize,
}

#[derive(Clone, Copy, Debug)]
struct Successor {
    transition: Option<u32>,
    step: PathStep,
}

struct ReportBuilder {
    diagnostics: Vec<Diagnostic>,
    complete: bool,
    work_items: usize,
    max_work_items: usize,
    max_diagnostics: usize,
    max_findings: usize,
    work_limit_hit: bool,
    path_limit_hit: bool,
    diagnostic_limit_hit: bool,
}

impl ReportBuilder {
    fn new(limits: VerificationLimits) -> Self {
        Self {
            diagnostics: Vec::with_capacity(limits.max_diagnostics.min(256)),
            complete: true,
            work_items: 0,
            max_work_items: limits.max_work_items,
            max_diagnostics: limits.max_diagnostics,
            max_findings: limits.max_diagnostics.saturating_sub(1),
            work_limit_hit: false,
            path_limit_hit: false,
            diagnostic_limit_hit: false,
        }
    }

    fn step(&mut self) -> bool {
        if self.work_items >= self.max_work_items {
            self.work_bound();
            return false;
        }
        self.work_items += 1;
        true
    }

    fn push(&mut self, diagnostic: Diagnostic) {
        if self.diagnostics.len() >= self.max_findings {
            self.complete = false;
            self.diagnostic_limit_hit = true;
        } else {
            self.diagnostics.push(diagnostic);
        }
    }

    fn work_bound(&mut self) {
        self.complete = false;
        self.work_limit_hit = true;
    }

    fn path_bound(&mut self) {
        self.complete = false;
        self.path_limit_hit = true;
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the report fields are kept explicit to preserve deterministic ownership"
    )]
    fn finish(
        mut self,
        reachable_states: Vec<StateId>,
        covered_transitions: Vec<u32>,
        uncovered_transitions: Vec<u32>,
        deadlock_candidates: Vec<StateId>,
        paths: Vec<VerificationPath>,
        transition_coverage: TransitionCoverage,
        eventless_cascade_depth: Option<usize>,
    ) -> VerificationReport {
        if self.work_limit_hit {
            self.append_bound(DiagnosticCode::WorkBound);
        } else if self.path_limit_hit {
            self.append_bound(DiagnosticCode::PathEnumerationBound);
        } else if self.diagnostic_limit_hit {
            self.append_bound(DiagnosticCode::DiagnosticBound);
        }
        VerificationReport {
            diagnostics: self.diagnostics,
            complete: self.complete,
            work_items: self.work_items,
            reachable_states,
            covered_transitions,
            uncovered_transitions,
            deadlock_candidates,
            paths,
            transition_coverage,
            eventless_cascade_depth,
        }
    }

    fn append_bound(&mut self, code: DiagnosticCode) {
        if self.diagnostics.len() < self.max_diagnostics {
            self.diagnostics.push(Diagnostic {
                code,
                severity: DiagnosticSeverity::Error,
                state: None,
                related_state: None,
                transition: None,
                depth: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::model::{StateNode, Transition, TransitionKind, TreeDef};

    fn compile(nodes: Vec<StateNode>) -> Plan {
        TreeDef {
            root: StateId(0),
            nodes,
        }
        .compile()
        .expect("test plan must compile")
    }

    fn transition(source: u32, target: u32, trigger: Trigger) -> Transition {
        Transition::new(
            StateId(source),
            TransitionTarget::State(StateId(target)),
            trigger,
            None,
            0,
            TransitionKind::External,
        )
    }

    #[test]
    fn reports_reachability_coverage_and_atomic_deadlock_candidates() {
        let plan = compile(vec![
            StateNode::compound(
                StateId(0),
                None,
                vec![StateId(1), StateId(2), StateId(3)],
                StateId(1),
                None,
            ),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(transition(
                1,
                2,
                Trigger::Event(EventId(7)),
            )),
            StateNode::atomic(StateId(2), Some(StateId(0))),
            StateNode::atomic(StateId(3), Some(StateId(0))),
        ]);
        let report = verify(
            &plan,
            &[EventId(7)],
            VerificationLimits::new(10_000, 64, 64, 8, 8),
        );

        assert!(report.is_complete());
        assert_eq!(
            report.reachable_states(),
            &[StateId(0), StateId(1), StateId(2)]
        );
        assert_eq!(
            report.transition_coverage(),
            TransitionCoverage {
                total: 1,
                covered: 1
            }
        );
        assert_eq!(report.uncovered_transitions(), &[]);
        assert_eq!(report.deadlock_candidates(), &[StateId(2)]);
        assert!(report.contains_code(DiagnosticCode::UnreachableState));
        assert!(report.contains_code(DiagnosticCode::DeadlockCandidate));
    }

    #[test]
    fn eventless_depth_is_deterministic_and_cycles_are_unbounded() {
        let acyclic = compile(vec![
            StateNode::compound(
                StateId(0),
                None,
                vec![StateId(1), StateId(2), StateId(3)],
                StateId(1),
                None,
            ),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(transition(
                1,
                2,
                Trigger::Eventless,
            )),
            StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(transition(
                2,
                3,
                Trigger::Eventless,
            )),
            StateNode::atomic(StateId(3), Some(StateId(0))),
        ]);
        let report = verify(&acyclic, &[], VerificationLimits::default());
        assert_eq!(report.eventless_cascade_depth(), Some(2));
        assert!(!report.contains_code(DiagnosticCode::EventlessCascadeExceeded));

        let cyclic = compile(vec![
            StateNode::compound(StateId(0), None, vec![StateId(1)], StateId(1), None),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(transition(
                1,
                1,
                Trigger::Eventless,
            )),
        ]);
        let report = verify(&cyclic, &[], VerificationLimits::default());
        assert_eq!(report.eventless_cascade_depth(), None);
        assert!(report.contains_code(DiagnosticCode::EventlessCascadeUnbounded));
    }

    #[test]
    fn finite_paths_are_depth_first_and_respect_event_alphabet() {
        let plan = compile(vec![
            StateNode::compound(
                StateId(0),
                None,
                vec![StateId(1), StateId(2), StateId(3)],
                StateId(1),
                None,
            ),
            StateNode::atomic(StateId(1), Some(StateId(0)))
                .with_transition(transition(1, 2, Trigger::Event(EventId(1))))
                .with_transition(transition(1, 3, Trigger::Event(EventId(2)))),
            StateNode::atomic(StateId(2), Some(StateId(0))),
            StateNode::atomic(StateId(3), Some(StateId(0))),
        ]);
        let report = verify(
            &plan,
            &[EventId(1)],
            VerificationLimits::new(10_000, 64, 16, 2, 8),
        );
        assert_eq!(report.paths().len(), 3);
        assert_eq!(report.paths()[1].states(), &[StateId(0), StateId(1)]);
        assert_eq!(
            report.paths()[2].states(),
            &[StateId(0), StateId(1), StateId(2)]
        );
        assert_eq!(report.covered_transitions(), &[0]);
        assert_eq!(report.uncovered_transitions(), &[1]);
        assert!(report.contains_code(DiagnosticCode::UncoveredTransition));

        let repeat = verify(
            &plan,
            &[EventId(1)],
            VerificationLimits::new(10_000, 64, 16, 2, 8),
        );
        assert_eq!(report, repeat);
    }

    #[test]
    fn covers_transitions_in_each_parallel_region() {
        let plan = compile(vec![
            StateNode::parallel(
                StateId(0),
                None,
                vec![StateId(1), StateId(2)],
                vec![StateId(1), StateId(2)],
                None,
            ),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(transition(
                1,
                1,
                Trigger::Event(EventId(1)),
            )),
            StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(transition(
                2,
                2,
                Trigger::Event(EventId(2)),
            )),
        ]);
        let report = verify(
            &plan,
            &[EventId(1), EventId(2)],
            VerificationLimits::new(10_000, 64, 64, 4, 8),
        );
        assert!(report.is_complete());
        assert_eq!(
            report.reachable_states(),
            &[StateId(0), StateId(1), StateId(2)]
        );
        assert_eq!(
            report.transition_coverage(),
            TransitionCoverage {
                total: 2,
                covered: 2
            }
        );
        assert_eq!(report.uncovered_transitions(), &[]);
    }

    #[test]
    fn diagnostic_limit_is_explicitly_reported() {
        let plan = compile(vec![
            StateNode::compound(
                StateId(0),
                None,
                vec![StateId(1), StateId(2), StateId(3)],
                StateId(1),
                None,
            ),
            StateNode::atomic(StateId(1), Some(StateId(0))),
            StateNode::atomic(StateId(2), Some(StateId(0))),
            StateNode::atomic(StateId(3), Some(StateId(0))),
        ]);
        let report = verify(&plan, &[], VerificationLimits::new(10_000, 1, 64, 4, 8));
        assert!(!report.is_complete());
        assert_eq!(report.diagnostics().len(), 1);
        assert!(report.contains_code(DiagnosticCode::DiagnosticBound));
    }

    #[test]
    fn path_and_work_bounds_make_report_incomplete() {
        let plan = compile(vec![StateNode::atomic(StateId(0), None)
            .with_transition(transition(0, 0, Trigger::Eventless))]);
        let path_limited = verify(&plan, &[], VerificationLimits::new(10_000, 8, 1, 4, 8));
        assert!(!path_limited.is_complete());
        assert!(path_limited.contains_code(DiagnosticCode::PathEnumerationBound));

        let work_limited = verify(&plan, &[], VerificationLimits::new(1, 8, 8, 4, 8));
        assert!(!work_limited.is_complete());
        assert!(work_limited.contains_code(DiagnosticCode::WorkBound));
    }
}
