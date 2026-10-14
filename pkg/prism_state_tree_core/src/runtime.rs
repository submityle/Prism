//! Bounded event delivery and deterministic cursor runtime.
use alloc::{boxed::Box, collections::VecDeque, sync::Arc, vec, vec::Vec};

use crate::model::{
    EventId, GuardId, HistoryMode, NodeKind, Plan, StateId, Transition, TransitionKind,
    TransitionTarget, Trigger,
};

/// A queued event. Events are values and are never re-enqueued by the kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Event {
    /// Event kind used for trigger matching.
    pub id: EventId,
}

impl Event {
    /// Creates an event of the given kind.
    pub const fn new(id: EventId) -> Self {
        Self { id }
    }
}

/// Read-only input supplied to a guard during selection.
#[derive(Clone, Copy, Debug)]
pub struct GuardContext {
    /// Event currently being consumed, or `None` during eventless stabilization.
    pub event: Option<Event>,
    /// State whose transition is being evaluated.
    pub source: StateId,
}

/// A pure boolean transition guard.
pub trait Guard: Send + Sync {
    /// Evaluates without mutating the runtime.
    fn evaluate(&self, context: GuardContext) -> bool;
}

/// A guard that always accepts.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlwaysGuard;

impl Guard for AlwaysGuard {
    fn evaluate(&self, _context: GuardContext) -> bool {
        true
    }
}

/// Lifecycle and transition observation hook.
pub trait LifecycleCallback {
    /// Called after a state is inserted into the active configuration.
    fn on_enter(&mut self, _state: StateId) {}
    /// Called before a state is removed from the active configuration.
    fn on_exit(&mut self, _state: StateId) {}
    /// Called after a transition's lifecycle work has completed.
    fn on_transition(&mut self, _source: StateId, _target: TransitionTarget) {}
}

/// A callback that intentionally observes nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopLifecycle;

impl LifecycleCallback for NoopLifecycle {}

/// Kind of observable runtime log entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogKind {
    /// State entry. Enter
    Enter,
    /// State exit.
    Exit,
    /// A selected transition.
    Transition,
}

/// A deterministic lifecycle/transition record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogEntry {
    /// Entry, exit, or transition record kind.
    pub kind: LogKind,
    /// Microstep ordinal within the current `run` call.
    pub microstep: usize,
    /// State associated with an entry/exit, or transition source.
    pub state: StateId,
    /// Transition source for transition records.
    pub source: Option<StateId>,
    /// Transition target for transition records.
    pub target: Option<TransitionTarget>,
    /// Event consumed by the transition, if any.
    pub event: Option<Event>,
}

/// Event queue overflow error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventQueueError {
    /// The configured capacity is zero.
    ZeroCapacity,
    /// The queue is full; the event was not inserted.
    Full,
}

/// Microstep cascade budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MicrostepLimit(pub usize);

/// Runtime execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    /// An event could not be queued.
    EventQueue(EventQueueError),
    /// Eventless/event-triggered transitions exceeded the run budget.
    MicrostepLimit(MicrostepLimit),
    /// A runtime operation referenced a state outside its plan.
    InvalidState(StateId),
    /// An enabled diagnostic log filled during the call; state changes remain applied.
    LogCapacityExceeded,
}

/// Result of a successful run-to-completion call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepReport {
    /// Number of FIFO events consumed.
    pub events_processed: usize,
    /// Number of selected transitions.
    pub microsteps: usize,
    /// Events left in the bounded queue.
    pub queued_events: usize,
}

/// A validated plan's mutable cursor and bounded input queue.
pub struct Instance {
    pub(crate) plan: Arc<Plan>,
    pub(crate) active: Vec<StateId>,
    pub(crate) history: Vec<Option<Vec<StateId>>>,
    pub(crate) queue: VecDeque<Event>,
    pub(crate) guards: Vec<Option<Box<dyn Guard>>>,
    pub(crate) log: Vec<LogEntry>,
    log_capacity: usize,
    log_overflowed: bool,
    path_scratch: Vec<StateId>,
    pub(crate) event_capacity: usize,
    pub(crate) microstep_limit: MicrostepLimit,
    pub(crate) started: bool,
    pub(crate) current_event: Option<Event>,
}

impl Instance {
    /// Creates an instance with a fixed event capacity and per-run transition budget.
    pub fn new(
        plan: Plan,
        event_capacity: usize,
        microstep_limit: MicrostepLimit,
    ) -> Result<Self, RuntimeError> {
        Self::new_shared(Arc::new(plan), event_capacity, microstep_limit)
    }

    /// Creates an instance using an already shared immutable plan allocation.
    ///
    /// This is the allocation-free plan handoff used by host adapters that
    /// keep a plan as an asset. The plan is retained for the instance lifetime
    /// and is not cloned on subsequent [`Self::run`] calls.
    pub fn new_shared(
        plan: Arc<Plan>,
        event_capacity: usize,
        microstep_limit: MicrostepLimit,
    ) -> Result<Self, RuntimeError> {
        if event_capacity == 0 {
            return Err(RuntimeError::EventQueue(EventQueueError::ZeroCapacity));
        }
        let count = plan.state_count();
        Ok(Self {
            plan,
            active: Vec::with_capacity(count),
            history: vec![None; count],
            queue: VecDeque::with_capacity(event_capacity),
            guards: Vec::new(),
            log: Vec::new(),
            log_capacity: 0,
            log_overflowed: false,
            path_scratch: Vec::with_capacity(count),
            event_capacity,
            microstep_limit,
            started: false,
            current_event: None,
        })
    }

    /// Returns the immutable plan shared by this instance.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    /// Returns active states in deterministic entry order.
    pub fn active_states(&self) -> &[StateId] {
        &self.active
    }
    /// Returns whether a state is active.
    pub fn is_active(&self, state: StateId) -> bool {
        self.active.contains(&state)
    }
    /// Returns the bounded queue's current length.
    pub fn queued_events(&self) -> usize {
        self.queue.len()
    }
    /// Returns the queue capacity.
    pub const fn event_capacity(&self) -> usize {
        self.event_capacity
    }
    /// Returns retained diagnostic entries in emission order. Logging is off by default.
    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }
    /// Enables diagnostics with a fixed preallocated capacity, or disables it with zero.
    /// Changing capacity clears prior entries and overflow status.
    pub fn configure_log_capacity(&mut self, capacity: usize) {
        self.log = Vec::with_capacity(capacity);
        self.log_capacity = capacity;
        self.log_overflowed = false;
    }
    /// Whether an enabled log lost records since its last clear.
    pub const fn log_overflowed(&self) -> bool {
        self.log_overflowed
    }
    /// Clears the observable log and overflow status without changing runtime state.
    pub fn clear_log(&mut self) {
        self.log.clear();
        self.log_overflowed = false;
    }

    fn record(&mut self, entry: LogEntry) {
        if self.log_capacity == 0 {
            return;
        }
        if self.log.len() == self.log_capacity {
            self.log_overflowed = true;
        } else {
            self.log.push(entry);
        }
    }

    fn finish_report(&self, report: StepReport) -> Result<StepReport, RuntimeError> {
        if self.log_overflowed {
            Err(RuntimeError::LogCapacityExceeded)
        } else {
            Ok(report)
        }
    }
    /// Installs or replaces a pure guard at a numeric identity.
    pub fn set_guard<G: Guard + 'static>(&mut self, id: GuardId, guard: G) {
        let index = id.0 as usize;
        if self.guards.len() <= index {
            self.guards.resize_with(index + 1, || None);
        }
        self.guards[index] = Some(Box::new(guard));
    }
    /// Enqueues an event without allowing it to bypass older events.
    pub fn send(&mut self, event: Event) -> Result<(), EventQueueError> {
        if self.queue.len() == self.event_capacity {
            return Err(EventQueueError::Full);
        }
        self.queue.push_back(event);
        Ok(())
    }

    /// Starts the root and performs initial eventless stabilization.
    pub fn start<C: LifecycleCallback>(
        &mut self,
        callback: &mut C,
    ) -> Result<StepReport, RuntimeError> {
        let mut report = StepReport {
            events_processed: 0,
            microsteps: 0,
            queued_events: self.queue.len(),
        };
        self.start_into(callback, &mut report)?;
        report.queued_events = self.queue.len();
        self.finish_report(report)
    }

    fn start_into<C: LifecycleCallback>(
        &mut self,
        callback: &mut C,
        report: &mut StepReport,
    ) -> Result<(), RuntimeError> {
        if !self.started {
            self.started = true;
            self.enter_target(self.plan.root(), None, None, 0, callback);
        }
        self.run_eventless(callback, report)
    }

    /// Consumes queued events FIFO and runs eventless cascades until completion
    /// or the configured microstep budget is exhausted.
    ///
    /// A budget error is deliberately not a transaction rollback: transitions
    /// already applied remain applied. If an event was selected before the
    /// budget was exhausted, it is counted as consumed; an event rejected
    /// before its transition starts remains at the FIFO head for retry. Callers
    /// can inspect [`Self::queued_events`] and the log before retrying with a
    /// larger budget. A budget error does not carry a partial [`StepReport`].
    pub fn run<C: LifecycleCallback>(
        &mut self,
        callback: &mut C,
    ) -> Result<StepReport, RuntimeError> {
        let mut report = StepReport {
            events_processed: 0,
            microsteps: 0,
            queued_events: self.queue.len(),
        };
        self.start_into(callback, &mut report)?;
        while let Some(event) = self.queue.pop_front() {
            self.current_event = Some(event);
            let selected = self.select(Trigger::Event(event.id));
            let result = if let Some(transition) = selected {
                if report.microsteps >= self.microstep_limit.0 {
                    // The transition budget applies before the event is
                    // consumed. Keep the event at the head of the FIFO so a
                    // caller can raise the budget and retry without loss.
                    self.queue.push_front(event);
                    Err(RuntimeError::MicrostepLimit(self.microstep_limit))
                } else {
                    report.events_processed += 1;
                    let result = self.apply(transition, &mut report, callback);
                    // The event is visible while selecting/applying its own
                    // transition, but never leaks into the eventless cascade
                    // that follows it.
                    self.current_event = None;
                    result.and_then(|()| self.run_eventless(callback, &mut report))
                }
            } else {
                report.events_processed += 1;
                self.current_event = None;
                Ok(())
            };
            self.current_event = None;
            result?;
        }
        self.run_eventless(callback, &mut report)?;
        report.queued_events = self.queue.len();
        self.finish_report(report)
    }

    fn run_eventless<C: LifecycleCallback>(
        &mut self,
        callback: &mut C,
        report: &mut StepReport,
    ) -> Result<(), RuntimeError> {
        while let Some(transition) = self.select(Trigger::Eventless) {
            if report.microsteps >= self.microstep_limit.0 {
                return Err(RuntimeError::MicrostepLimit(self.microstep_limit));
            }
            self.apply(transition, report, callback)?;
        }
        Ok(())
    }

    fn select(&self, trigger: Trigger) -> Option<(StateId, usize)> {
        let mut best: Option<(StateId, usize, u16, u32, u32)> = None;
        for &active_state in &self.active {
            let mut source = Some(active_state);
            while let Some(owner) = source {
                let node = &self.plan.nodes[owner.0 as usize];
                for (index, transition) in node.transitions.iter().enumerate() {
                    if transition.trigger != trigger || !self.guard_passes(transition, owner) {
                        continue;
                    }
                    let candidate = (
                        owner,
                        index,
                        transition.priority,
                        transition.declaration_order,
                        self.plan.depths[owner.0 as usize],
                    );
                    let key = (candidate.2, candidate.3, core::cmp::Reverse(candidate.4));
                    let replace = best
                        .as_ref()
                        .map(|old| key < (old.2, old.3, core::cmp::Reverse(old.4)))
                        .unwrap_or(true);
                    if replace {
                        best = Some(candidate);
                    }
                }
                source = self.plan.parent(owner);
            }
        }
        best.map(|(source, index, _, _, _)| (source, index))
    }

    fn guard_passes(&self, transition: &Transition, source: StateId) -> bool {
        match transition.guard {
            None => true,
            Some(id) => self
                .guards
                .get(id.0 as usize)
                .and_then(Option::as_deref)
                .map(|guard| {
                    guard.evaluate(GuardContext {
                        event: self.current_event,
                        source,
                    })
                })
                .unwrap_or(false),
        }
    }

    fn apply<C: LifecycleCallback>(
        &mut self,
        selected: (StateId, usize),
        report: &mut StepReport,
        callback: &mut C,
    ) -> Result<(), RuntimeError> {
        let (source, index) = selected;
        if report.microsteps >= self.microstep_limit.0 {
            return Err(RuntimeError::MicrostepLimit(self.microstep_limit));
        }
        let transition = &self.plan.nodes[source.0 as usize].transitions[index];
        let target = transition.target;
        let kind = transition.kind;
        let effective_target = match target {
            TransitionTarget::SelfState => source,
            TransitionTarget::State(id) | TransitionTarget::History { owner: id, .. } => id,
        };
        if kind == TransitionKind::External {
            let boundary = if self.plan.is_descendant(effective_target, source) {
                // A transition to a strict descendant keeps its source active;
                // an explicit self transition restarts the source.
                if effective_target == source {
                    self.plan.parent(source)
                } else {
                    Some(source)
                }
            } else if self.plan.is_descendant(source, effective_target) {
                // An external transition to an ancestor exits and re-enters
                // that ancestor, not merely the source's branch beneath it.
                self.plan.parent(effective_target)
            } else {
                let lca = self.plan.lca(source, effective_target);
                if matches!(
                    self.plan.nodes[lca.0 as usize].kind,
                    NodeKind::Parallel { .. }
                ) {
                    // A cross-region transition exits the complete parallel
                    // configuration so the target region cannot retain a
                    // stale active branch.
                    self.plan.parent(lca)
                } else {
                    Some(lca)
                }
            };
            if boundary == Some(source) && effective_target != source {
                self.capture_histories(source);
                self.exit_descendants(source, callback, report.microsteps);
            } else {
                let exit_root = self.exit_root(source, boundary);
                self.capture_histories(exit_root);
                self.exit_subtree(exit_root, callback, report.microsteps);
            }
            self.enter_target(
                effective_target,
                boundary,
                match target {
                    TransitionTarget::History { depth, .. } => Some(depth),
                    _ => None,
                },
                report.microsteps,
                callback,
            );
            self.complete_parallel_configuration(report.microsteps, callback);
        }
        self.record(LogEntry {
            kind: LogKind::Transition,
            microstep: report.microsteps,
            state: source,
            source: Some(source),
            target: Some(target),
            event: self.current_event,
        });
        callback.on_transition(source, target);
        report.microsteps += 1;
        Ok(())
    }

    /// Repairs region roots that were removed by a transition owned by an
    /// already-active parallel state. A transition from a parallel owner may
    /// intentionally replace one region's branch while the owner itself stays
    /// active; every other region must still have a live configuration.
    fn complete_parallel_configuration<C: LifecycleCallback>(
        &mut self,
        microstep: usize,
        callback: &mut C,
    ) {
        let mut active_index = 0;
        while active_index < self.active.len() {
            let parallel = self.active[active_index];
            active_index += 1;
            let region_count = match &self.plan.nodes[parallel.0 as usize].kind {
                NodeKind::Parallel { regions, .. } => regions.len(),
                _ => continue,
            };
            for region_index in 0..region_count {
                let region = match &self.plan.nodes[parallel.0 as usize].kind {
                    NodeKind::Parallel { regions, .. } => regions[region_index],
                    _ => unreachable!(),
                };
                if !self.is_active(region) {
                    self.enter_state_default(region, microstep, callback);
                }
            }
        }
    }

    fn capture_histories(&mut self, root: StateId) {
        for index in 0..self.active.len() {
            let owner = self.active[index];
            if !self.plan.is_descendant(owner, root) {
                continue;
            }
            let Some(depth) = self.plan.nodes[owner.0 as usize].kind.history() else {
                continue;
            };
            let saved = self.history[owner.0 as usize].get_or_insert_with(Vec::new);
            saved.clear();
            for &state in &self.active {
                let include = match depth {
                    HistoryMode::Deep => state != owner && self.plan.is_descendant(state, owner),
                    HistoryMode::Shallow => self.plan.parent(state) == Some(owner),
                };
                if include {
                    saved.push(state);
                }
            }
        }
    }

    fn exit_subtree<C: LifecycleCallback>(
        &mut self,
        root: StateId,
        callback: &mut C,
        microstep: usize,
    ) {
        for index in (0..self.active.len()).rev() {
            let state = self.active[index];
            if !self.plan.is_descendant(state, root) {
                continue;
            }
            callback.on_exit(state);
            self.record(LogEntry {
                kind: LogKind::Exit,
                microstep,
                state,
                source: None,
                target: None,
                event: self.current_event,
            });
        }
        self.active.retain(|id| !self.plan.is_descendant(*id, root));
    }

    fn exit_descendants<C: LifecycleCallback>(
        &mut self,
        boundary: StateId,
        callback: &mut C,
        microstep: usize,
    ) {
        for index in (0..self.active.len()).rev() {
            let state = self.active[index];
            if state == boundary || !self.plan.is_descendant(state, boundary) {
                continue;
            }
            callback.on_exit(state);
            self.record(LogEntry {
                kind: LogKind::Exit,
                microstep,
                state,
                source: None,
                target: None,
                event: self.current_event,
            });
        }
        self.active
            .retain(|id| *id == boundary || !self.plan.is_descendant(*id, boundary));
    }

    fn exit_root(&self, source: StateId, boundary: Option<StateId>) -> StateId {
        let Some(boundary) = boundary else {
            return self.plan.root();
        };
        let mut root = source;
        while self.plan.parent(root) != Some(boundary) {
            root = self
                .plan
                .parent(root)
                .expect("validated source has a path to the LCA");
        }
        root
    }

    fn enter_target<C: LifecycleCallback>(
        &mut self,
        target: StateId,
        boundary: Option<StateId>,
        history: Option<HistoryMode>,
        microstep: usize,
        callback: &mut C,
    ) {
        if !self.plan.contains(target) || self.is_active(target) {
            return;
        }
        self.path_scratch.clear();
        let mut cursor = target;
        while !self.is_active(cursor) && Some(cursor) != boundary {
            self.path_scratch.push(cursor);
            cursor = match self.plan.parent(cursor) {
                Some(parent) => parent,
                None => break,
            };
        }
        self.path_scratch.reverse();
        for index in 0..self.path_scratch.len() {
            let state = self.path_scratch[index];
            self.enter_one(state, microstep, callback);
            if history.is_none()
                && matches!(
                    self.plan.nodes[state.0 as usize].kind,
                    NodeKind::Parallel { .. }
                )
            {
                let region_count = match &self.plan.nodes[state.0 as usize].kind {
                    NodeKind::Parallel { regions, .. } => regions.len(),
                    _ => 0,
                };
                for index in 0..region_count {
                    let region = match &self.plan.nodes[state.0 as usize].kind {
                        NodeKind::Parallel { regions, .. } => regions[index],
                        _ => continue,
                    };
                    if !self.plan.is_descendant(target, region) && !self.is_active(region) {
                        self.enter_state_default(region, microstep, callback);
                    }
                }
            }
        }
        if history.is_some() {
            self.restore_children(
                target,
                history.unwrap_or(HistoryMode::Deep),
                microstep,
                callback,
            );
        } else {
            self.initialize_children(target, microstep, callback);
        }
        self.path_scratch.clear();
    }

    fn enter_one<C: LifecycleCallback>(
        &mut self,
        state: StateId,
        microstep: usize,
        callback: &mut C,
    ) {
        self.active.push(state);
        callback.on_enter(state);
        self.record(LogEntry {
            kind: LogKind::Enter,
            microstep,
            state,
            source: None,
            target: None,
            event: self.current_event,
        });
    }

    fn enter_state_default<C: LifecycleCallback>(
        &mut self,
        state: StateId,
        microstep: usize,
        callback: &mut C,
    ) {
        if self.is_active(state) {
            return;
        }
        self.enter_one(state, microstep, callback);
        self.initialize_children(state, microstep, callback);
    }

    fn initialize_children<C: LifecycleCallback>(
        &mut self,
        owner: StateId,
        microstep: usize,
        callback: &mut C,
    ) {
        let region_count = match &self.plan.nodes[owner.0 as usize].kind {
            NodeKind::Atomic => return,
            NodeKind::Compound { initial, .. } => {
                self.enter_state_default(*initial, microstep, callback);
                return;
            }
            NodeKind::Parallel { regions, .. } => regions.len(),
        };
        for index in 0..region_count {
            let region = match &self.plan.nodes[owner.0 as usize].kind {
                NodeKind::Parallel { regions, .. } => regions[index],
                _ => unreachable!(),
            };
            self.enter_state_default(region, microstep, callback);
        }
    }

    fn restore_children<C: LifecycleCallback>(
        &mut self,
        owner: StateId,
        depth: HistoryMode,
        microstep: usize,
        callback: &mut C,
    ) {
        let Some(saved) = &self.history[owner.0 as usize] else {
            self.initialize_children(owner, microstep, callback);
            return;
        };
        match depth {
            HistoryMode::Shallow => {
                for index in 0..saved.len() {
                    let state = self.history[owner.0 as usize].as_ref().unwrap()[index];
                    if self.plan.parent(state) == Some(owner) {
                        self.enter_state_default(state, microstep, callback);
                    }
                }
            }
            HistoryMode::Deep => {
                for index in 0..saved.len() {
                    let state = self.history[owner.0 as usize].as_ref().unwrap()[index];
                    if !self.is_active(state)
                        && self
                            .plan
                            .parent(state)
                            .map(|parent| self.is_active(parent))
                            .unwrap_or(false)
                    {
                        self.enter_one(state, microstep, callback);
                    }
                }
            }
        }
    }
}
