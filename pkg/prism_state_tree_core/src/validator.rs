//! Bounded, read-only diagnostics for a compiled [`Plan`].
//!
//! The validator deliberately operates on the model that exists in this crate.
//! It does not inspect runtime instances and it does not invent semantics for
//! authoring features that are not represented by [`crate::model`].

use alloc::vec;
use alloc::vec::Vec;

use crate::model::{NodeKind, Plan, StateId, TransitionTarget, Trigger};

/// Severity assigned to a static diagnostic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DiagnosticSeverity {
    /// The model has a potentially unsafe or non-terminating property.
    Error,
    /// The model is valid, but a conservative analysis found a risk.
    Warning,
}

/// Stable code identifying a validator finding.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DiagnosticCode {
    /// A state cannot be reached from the root or any reachable transition.
    St0001,
    /// A reachable eventless cascade can exceed its configured bound.
    St0006,
    /// Parallel regions refer to one history owner from more than one region.
    St0007,
    /// The validator stopped at an explicit work or diagnostic bound.
    St0099,
}

impl DiagnosticCode {
    /// Returns the stable external spelling of this code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::St0001 => "ST0001",
            Self::St0006 => "ST0006",
            Self::St0007 => "ST0007",
            Self::St0099 => "ST0099",
        }
    }
}

/// One structured static finding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Stable diagnostic code.
    pub code: DiagnosticCode,
    /// Diagnostic severity.
    pub severity: DiagnosticSeverity,
    /// Primary state associated with the finding, when applicable.
    pub state: Option<StateId>,
    /// Related state, such as a target or history owner.
    pub related_state: Option<StateId>,
    /// Declaration order of the related transition, when applicable.
    pub transition: Option<u32>,
}

impl Diagnostic {
    /// Returns a stable, human-readable summary for this code.
    pub const fn summary(self) -> &'static str {
        match self.code {
            DiagnosticCode::St0001 => {
                "state is not reachable from the root entry paths or a reachable transition"
            }
            DiagnosticCode::St0006 => {
                "a reachable eventless cascade can exceed the configured bound"
            }
            DiagnosticCode::St0007 => "parallel regions refer to the same history owner",
            DiagnosticCode::St0099 => {
                "the validator bound was reached before a complete report was produced"
            }
        }
    }
}

/// Explicit limits that keep validation work and output bounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidationLimits {
    /// Maximum number of stored diagnostics, including a possible `ST0099`.
    pub max_diagnostics: usize,
    /// Maximum number of model work items examined.
    pub max_work_items: usize,
    /// Maximum allowed eventless edges in one acyclic cascade.
    pub max_eventless_cascade: usize,
}

impl ValidationLimits {
    /// Creates a bounded validation configuration.
    pub const fn new(
        max_diagnostics: usize,
        max_work_items: usize,
        max_eventless_cascade: usize,
    ) -> Self {
        Self {
            max_diagnostics,
            max_work_items,
            max_eventless_cascade,
        }
    }
}

impl Default for ValidationLimits {
    fn default() -> Self {
        Self::new(256, 1_000_000, 64)
    }
}

/// Result of one bounded validation pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationReport {
    diagnostics: Vec<Diagnostic>,
    complete: bool,
    work_items: usize,
}

impl ValidationReport {
    /// Returns all findings retained by the report.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Returns whether the configured work and output bounds were sufficient.
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns the number of model work items consumed by the pass.
    pub const fn work_items(&self) -> usize {
        self.work_items
    }

    /// Returns whether at least one finding has the supplied code.
    pub fn contains_code(&self, code: DiagnosticCode) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code)
    }
}

/// Read-only static validator for a compiled [`Plan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Validator {
    limits: ValidationLimits,
}

impl Validator {
    /// Creates a validator with explicit resource bounds.
    pub const fn new(limits: ValidationLimits) -> Self {
        Self { limits }
    }

    /// Returns the bounds used by this validator.
    pub const fn limits(self) -> ValidationLimits {
        self.limits
    }

    /// Produces deterministic diagnostics without changing the plan.
    pub fn validate(self, plan: &Plan) -> ValidationReport {
        let mut builder = ReportBuilder::new(self.limits);
        let state_count = plan.state_count();
        if state_count > self.limits.max_work_items {
            builder.limit_hit();
            return builder.finish();
        }

        let mut reachable = vec![false; state_count];
        let mut queue = Vec::with_capacity(state_count.min(self.limits.max_work_items));
        mark_state(plan.root(), &mut reachable, &mut queue);

        let mut eventless = vec![Vec::new(); state_count];
        while let Some(state) = queue.pop() {
            if !builder.step() {
                return builder.finish();
            }
            let Some(node) = plan.node(state) else {
                builder.limit_hit();
                return builder.finish();
            };
            match &node.kind {
                NodeKind::Atomic => {}
                NodeKind::Compound { initial, .. } => {
                    mark_state(*initial, &mut reachable, &mut queue);
                }
                NodeKind::Parallel { regions, .. } => {
                    for region in regions {
                        mark_state(*region, &mut reachable, &mut queue);
                    }
                }
            }
            for transition in &node.transitions {
                if !builder.step() {
                    return builder.finish();
                }
                let target = transition_target(state, transition.target);
                if matches!(transition.trigger, Trigger::Eventless) {
                    eventless[state.0 as usize].push(EventlessEdge {
                        target,
                        transition: transition.declaration_order(),
                    });
                }
                if transition.target_is_history() {
                    mark_subtree(plan, target, &mut reachable, &mut queue);
                } else {
                    mark_state(target, &mut reachable, &mut queue);
                }
            }
        }

        for (index, is_reachable) in reachable.iter().enumerate() {
            if !builder.step() {
                return builder.finish();
            }
            if !is_reachable {
                builder.push(Diagnostic {
                    code: DiagnosticCode::St0001,
                    severity: DiagnosticSeverity::Error,
                    state: Some(StateId(index as u32)),
                    related_state: None,
                    transition: None,
                });
            }
        }

        detect_eventless_cycles_and_depth(plan, &reachable, &eventless, self.limits, &mut builder);
        detect_parallel_history_conflicts(plan, &mut builder);
        builder.finish()
    }
}

/// Validates a plan with [`ValidationLimits::default`].
pub fn validate(plan: &Plan) -> ValidationReport {
    Validator::new(ValidationLimits::default()).validate(plan)
}

#[derive(Clone, Copy, Debug)]
struct EventlessEdge {
    target: StateId,
    transition: u32,
}

fn transition_target(source: StateId, target: TransitionTarget) -> StateId {
    match target {
        TransitionTarget::State(state) => state,
        TransitionTarget::History { owner, .. } => owner,
        TransitionTarget::SelfState => source,
    }
}

trait TransitionTargetExt {
    fn target_is_history(&self) -> bool;
}

impl TransitionTargetExt for crate::model::Transition {
    fn target_is_history(&self) -> bool {
        matches!(self.target, TransitionTarget::History { .. })
    }
}

fn mark_state(state: StateId, reachable: &mut [bool], queue: &mut Vec<StateId>) {
    let index = state.0 as usize;
    if !reachable[index] {
        reachable[index] = true;
        queue.push(state);
    }
}

fn mark_subtree(plan: &Plan, root: StateId, reachable: &mut [bool], queue: &mut Vec<StateId>) {
    mark_state(root, reachable, queue);
    for index in 0..plan.state_count() {
        let state = StateId(index as u32);
        if state != root && is_descendant(plan, state, root) {
            mark_state(state, reachable, queue);
        }
    }
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

fn detect_eventless_cycles_and_depth(
    plan: &Plan,
    reachable: &[bool],
    eventless: &[Vec<EventlessEdge>],
    limits: ValidationLimits,
    builder: &mut ReportBuilder,
) {
    let count = plan.state_count();
    let mut color = vec![0_u8; count];
    let mut stack: Vec<(StateId, usize)> = Vec::new();
    let mut reported_cycles = Vec::new();

    for start_index in 0..count {
        if !reachable[start_index] || color[start_index] != 0 {
            continue;
        }
        let start = StateId(start_index as u32);
        color[start_index] = 1;
        stack.push((start, 0));
        while let Some((state, edge_index)) = stack.last_mut() {
            if !builder.step() {
                return;
            }
            let edges = &eventless[state.0 as usize];
            if *edge_index == edges.len() {
                color[state.0 as usize] = 2;
                stack.pop();
                continue;
            }
            let edge = edges[*edge_index];
            *edge_index += 1;
            if !reachable[edge.target.0 as usize] {
                continue;
            }
            if color[edge.target.0 as usize] == 1 {
                if !reported_cycles.contains(&edge.transition) {
                    reported_cycles.push(edge.transition);
                    builder.push(Diagnostic {
                        code: DiagnosticCode::St0006,
                        severity: DiagnosticSeverity::Warning,
                        state: Some(*state),
                        related_state: Some(edge.target),
                        transition: Some(edge.transition),
                    });
                }
            } else if color[edge.target.0 as usize] == 0 {
                color[edge.target.0 as usize] = 1;
                stack.push((edge.target, 0));
            }
        }
    }

    let mut indegree = vec![0_usize; count];
    let mut topo = Vec::with_capacity(count);
    let mut queue = Vec::new();
    for source in 0..count {
        if !reachable[source] {
            continue;
        }
        for edge in &eventless[source] {
            if reachable[edge.target.0 as usize] {
                indegree[edge.target.0 as usize] += 1;
            }
        }
    }
    for source in 0..count {
        if reachable[source] && indegree[source] == 0 {
            queue.push(StateId(source as u32));
        }
    }
    while let Some(state) = queue.pop() {
        if !builder.step() {
            return;
        }
        topo.push(state);
        for edge in &eventless[state.0 as usize] {
            if !reachable[edge.target.0 as usize] {
                continue;
            }
            indegree[edge.target.0 as usize] -= 1;
            if indegree[edge.target.0 as usize] == 0 {
                queue.push(edge.target);
            }
        }
    }
    if topo.len() != reachable.iter().filter(|reachable| **reachable).count() {
        return;
    }

    let mut longest = vec![0_usize; count];
    for state in topo {
        for edge in &eventless[state.0 as usize] {
            if !reachable[edge.target.0 as usize] {
                continue;
            }
            let depth = longest[state.0 as usize] + 1;
            if depth > longest[edge.target.0 as usize] {
                longest[edge.target.0 as usize] = depth;
            }
            if depth > limits.max_eventless_cascade {
                builder.push(Diagnostic {
                    code: DiagnosticCode::St0006,
                    severity: DiagnosticSeverity::Warning,
                    state: Some(state),
                    related_state: Some(edge.target),
                    transition: Some(edge.transition),
                });
            }
        }
    }
}

fn detect_parallel_history_conflicts(plan: &Plan, builder: &mut ReportBuilder) {
    for parallel_index in 0..plan.state_count() {
        if !builder.step() {
            return;
        }
        let parallel = StateId(parallel_index as u32);
        let Some(node) = plan.node(parallel) else {
            return;
        };
        let NodeKind::Parallel { regions, .. } = &node.kind else {
            continue;
        };
        let mut owners: Vec<(StateId, StateId)> = Vec::new();
        for region in regions {
            if !builder.step() {
                return;
            }
            for state_index in 0..plan.state_count() {
                if !builder.step() {
                    return;
                }
                let state = StateId(state_index as u32);
                if state != *region && !is_descendant(plan, state, *region) {
                    continue;
                }
                let Some(state_node) = plan.node(state) else {
                    return;
                };
                for transition in &state_node.transitions {
                    if !builder.step() {
                        return;
                    }
                    let TransitionTarget::History { owner, .. } = transition.target else {
                        continue;
                    };
                    if let Some((_, previous_region)) = owners
                        .iter()
                        .find(|(previous_owner, _)| *previous_owner == owner)
                    {
                        if *previous_region != *region {
                            builder.push(Diagnostic {
                                code: DiagnosticCode::St0007,
                                severity: DiagnosticSeverity::Warning,
                                state: Some(parallel),
                                related_state: Some(owner),
                                transition: Some(transition.declaration_order()),
                            });
                        }
                    } else {
                        owners.push((owner, *region));
                    }
                }
            }
        }
    }
}

struct ReportBuilder {
    diagnostics: Vec<Diagnostic>,
    complete: bool,
    diagnostic_limit: usize,
    total_diagnostic_limit: usize,
    work_limit: usize,
    work_items: usize,
    limit_hit: bool,
}

impl ReportBuilder {
    fn new(limits: ValidationLimits) -> Self {
        Self {
            diagnostics: Vec::with_capacity(limits.max_diagnostics.min(256)),
            complete: true,
            diagnostic_limit: limits.max_diagnostics.saturating_sub(1),
            total_diagnostic_limit: limits.max_diagnostics,
            work_limit: limits.max_work_items,
            work_items: 0,
            limit_hit: false,
        }
    }

    fn step(&mut self) -> bool {
        if self.work_items == self.work_limit {
            self.limit_hit();
            return false;
        }
        self.work_items += 1;
        true
    }

    fn push(&mut self, diagnostic: Diagnostic) {
        if self.diagnostics.len() < self.diagnostic_limit {
            self.diagnostics.push(diagnostic);
        } else {
            self.limit_hit();
        }
    }

    fn limit_hit(&mut self) {
        self.complete = false;
        self.limit_hit = true;
    }

    fn finish(mut self) -> ValidationReport {
        if self.limit_hit
            && self.total_diagnostic_limit > 0
            && self.diagnostics.len() < self.total_diagnostic_limit
        {
            self.diagnostics.push(Diagnostic {
                code: DiagnosticCode::St0099,
                severity: DiagnosticSeverity::Error,
                state: None,
                related_state: None,
                transition: None,
            });
        }
        ValidationReport {
            diagnostics: self.diagnostics,
            complete: self.complete,
            work_items: self.work_items,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::model::{HistoryMode, StateNode, Transition, TransitionKind, TreeDef};
    use crate::EventId;

    fn compile(nodes: Vec<StateNode>) -> Plan {
        TreeDef {
            root: StateId(0),
            nodes,
        }
        .compile()
        .expect("test model must compile")
    }

    fn compound_root(children: Vec<StateId>, initial: StateId) -> StateNode {
        StateNode::compound(StateId(0), None, children, initial, None)
    }

    #[test]
    fn reports_unreachable_state_and_keeps_valid_plan_unchanged() {
        let plan = compile(vec![
            compound_root(vec![StateId(1), StateId(2)], StateId(1)),
            StateNode::atomic(StateId(1), Some(StateId(0))),
            StateNode::atomic(StateId(2), Some(StateId(0))),
        ]);
        let report = validate(&plan);
        assert!(report.is_complete());
        assert!(report.diagnostics().iter().any(|diagnostic| {
            diagnostic.code == DiagnosticCode::St0001 && diagnostic.state == Some(StateId(2))
        }));
        assert_eq!(plan.root(), StateId(0));
        assert_eq!(plan.state_count(), 3);
    }

    #[test]
    fn transition_target_makes_state_reachable() {
        let plan = compile(vec![
            compound_root(vec![StateId(1), StateId(2)], StateId(1)),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
                StateId(1),
                TransitionTarget::State(StateId(2)),
                Trigger::Event(EventId(1)),
                None,
                0,
                TransitionKind::External,
            )),
            StateNode::atomic(StateId(2), Some(StateId(0))),
        ]);
        let report = validate(&plan);
        assert!(!report.contains_code(DiagnosticCode::St0001));
    }

    #[test]
    fn reports_reachable_eventless_cycle() {
        let plan = compile(vec![
            compound_root(vec![StateId(1), StateId(2)], StateId(1)),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
                StateId(1),
                TransitionTarget::State(StateId(2)),
                Trigger::Eventless,
                None,
                0,
                TransitionKind::External,
            )),
            StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(Transition::new(
                StateId(2),
                TransitionTarget::State(StateId(1)),
                Trigger::Eventless,
                None,
                0,
                TransitionKind::External,
            )),
        ]);
        let report = Validator::new(ValidationLimits::new(16, 128, 8)).validate(&plan);
        assert!(report.contains_code(DiagnosticCode::St0006));
        assert!(report.is_complete());
    }

    #[test]
    fn reports_acyclic_eventless_cascade_over_limit() {
        let plan = compile(vec![
            compound_root(vec![StateId(1), StateId(2), StateId(3)], StateId(1)),
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
                StateId(1),
                TransitionTarget::State(StateId(2)),
                Trigger::Eventless,
                None,
                0,
                TransitionKind::External,
            )),
            StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(Transition::new(
                StateId(2),
                TransitionTarget::State(StateId(3)),
                Trigger::Eventless,
                None,
                0,
                TransitionKind::External,
            )),
            StateNode::atomic(StateId(3), Some(StateId(0))),
        ]);
        let report = Validator::new(ValidationLimits::new(16, 128, 1)).validate(&plan);
        assert!(report.contains_code(DiagnosticCode::St0006));
    }

    #[test]
    fn reports_history_owner_shared_by_parallel_regions() {
        let plan = compile(vec![
            compound_root(vec![StateId(1)], StateId(1)),
            StateNode::parallel(
                StateId(1),
                Some(StateId(0)),
                vec![StateId(2), StateId(3)],
                vec![StateId(2), StateId(3)],
                Some(HistoryMode::Deep),
            ),
            StateNode::atomic(StateId(2), Some(StateId(1))).with_transition(Transition::new(
                StateId(2),
                TransitionTarget::History {
                    owner: StateId(1),
                    depth: HistoryMode::Deep,
                },
                Trigger::Event(EventId(1)),
                None,
                0,
                TransitionKind::External,
            )),
            StateNode::atomic(StateId(3), Some(StateId(1))).with_transition(Transition::new(
                StateId(3),
                TransitionTarget::History {
                    owner: StateId(1),
                    depth: HistoryMode::Deep,
                },
                Trigger::Event(EventId(2)),
                None,
                0,
                TransitionKind::External,
            )),
        ]);
        let report = validate(&plan);
        assert!(report.contains_code(DiagnosticCode::St0007));
    }

    #[test]
    fn report_is_bounded_and_marks_incomplete_output() {
        let plan = compile(vec![
            compound_root(
                vec![StateId(1), StateId(2), StateId(3), StateId(4)],
                StateId(1),
            ),
            StateNode::atomic(StateId(1), Some(StateId(0))),
            StateNode::atomic(StateId(2), Some(StateId(0))),
            StateNode::atomic(StateId(3), Some(StateId(0))),
            StateNode::atomic(StateId(4), Some(StateId(0))),
        ]);
        let report = Validator::new(ValidationLimits::new(2, 128, 8)).validate(&plan);
        assert!(!report.is_complete());
        assert!(report.diagnostics().len() <= 2);
        assert!(report.contains_code(DiagnosticCode::St0099));
    }
}
