//! Deterministic input tracing and digest-based replay for [`Instance`].
//!
//! This module observes an instance only through its public runtime API. A
//! trace records operation boundaries rather than private microsteps: a start
//! operation, or one event followed by run-to-completion. Replaying the same
//! inputs against the same semantic plan and equivalent guard services lets a
//! caller detect a different active configuration without requiring a
//! debugger, a renderer, or a serialization framework.

use alloc::vec::Vec;

use crate::{
    model::{HistoryMode, NodeKind, Plan, StateId, TransitionTarget, Trigger},
    runtime::{Event, EventQueueError, Instance, LifecycleCallback, RuntimeError, StepReport},
};

const DIGEST_SEED: u128 = 0x6c62272e07bb014262b821756295c58d_u128;
const DIGEST_PRIME: u128 = 0x1000000000000000000013b_u128;

/// A deterministic digest of a plan's public semantic definition.
///
/// The digest is useful for rejecting a trace replayed against a different
/// plan. It is not a cryptographic hash and is not a persistent wire-format
/// compatibility promise.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlanDigest(u128);

impl PlanDigest {
    /// Returns the raw deterministic digest value.
    pub const fn value(self) -> u128 {
        self.0
    }

    /// Computes a digest from the public semantic contents of a compiled plan.
    pub fn from_plan(plan: &Plan) -> Self {
        let mut hash = DIGEST_SEED;
        mix_u32(&mut hash, plan.root().0);
        mix_len(&mut hash, plan.state_count());
        for index in 0..plan.state_count() {
            let id = StateId(index as u32);
            let node = plan.node(id).expect("compiled plan has dense state ids");
            mix_u32(&mut hash, node.id.0);
            mix_option_state(&mut hash, node.parent);
            match &node.kind {
                NodeKind::Atomic => mix_u8(&mut hash, 0),
                NodeKind::Compound {
                    children,
                    initial,
                    history,
                } => {
                    mix_u8(&mut hash, 1);
                    mix_states(&mut hash, children);
                    mix_u32(&mut hash, initial.0);
                    mix_history(&mut hash, *history);
                }
                NodeKind::Parallel {
                    children,
                    regions,
                    history,
                } => {
                    mix_u8(&mut hash, 2);
                    mix_states(&mut hash, children);
                    mix_states(&mut hash, regions);
                    mix_history(&mut hash, *history);
                }
            }
            mix_len(&mut hash, node.transitions.len());
            for transition in &node.transitions {
                mix_u32(&mut hash, transition.source.0);
                match transition.target {
                    TransitionTarget::State(state) => {
                        mix_u8(&mut hash, 0);
                        mix_u32(&mut hash, state.0);
                    }
                    TransitionTarget::History { owner, depth } => {
                        mix_u8(&mut hash, 1);
                        mix_u32(&mut hash, owner.0);
                        mix_history(&mut hash, Some(depth));
                    }
                    TransitionTarget::SelfState => mix_u8(&mut hash, 2),
                }
                match transition.trigger {
                    Trigger::Event(event) => {
                        mix_u8(&mut hash, 0);
                        mix_u32(&mut hash, event.0);
                    }
                    Trigger::Eventless => mix_u8(&mut hash, 1),
                }
                mix_option_u32(&mut hash, transition.guard.map(|guard| guard.0));
                mix_u16(&mut hash, transition.priority);
                mix_u8(
                    &mut hash,
                    match transition.kind {
                        crate::TransitionKind::External => 0,
                        crate::TransitionKind::Internal => 1,
                    },
                );
                mix_u32(&mut hash, transition.declaration_order());
            }
        }
        Self(hash)
    }
}

/// A deterministic digest of an instance's active-state entry order.
///
/// Only the active configuration is included. Pending events, history, guard
/// objects, lifecycle callbacks, and transient runtime details are outside
/// this digest and should be handled by a caller's own trace policy when
/// they are semantically relevant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ActiveStateDigest {
    hash: u128,
    state_count: usize,
}

impl ActiveStateDigest {
    /// Computes a digest from an active-state slice in entry order.
    pub fn from_states(states: &[StateId]) -> Self {
        let mut hash = DIGEST_SEED;
        mix_len(&mut hash, states.len());
        for (index, state) in states.iter().enumerate() {
            mix_len(&mut hash, index);
            mix_u32(&mut hash, state.0);
        }
        Self {
            hash,
            state_count: states.len(),
        }
    }

    /// Returns the raw digest value.
    pub const fn value(self) -> u128 {
        self.hash
    }

    /// Returns the number of states represented by the digest.
    pub const fn state_count(self) -> usize {
        self.state_count
    }
}

/// Computes the active-state digest of an instance through its public API.
pub fn active_state_digest(instance: &Instance) -> ActiveStateDigest {
    ActiveStateDigest::from_states(instance.active_states())
}

/// An input operation represented by a trace record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceInput {
    /// Start the instance and perform its initial eventless stabilization.
    Start,
    /// Enqueue one event and run the instance to completion.
    Event(Event),
}

/// One bounded operation-and-state observation in a [`TraceLog`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceRecord {
    /// Input operation that produced the observed boundary.
    pub input: TraceInput,
    /// Active-state digest after the operation completed.
    pub digest: ActiveStateDigest,
}

/// Failure while capturing a trace operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceLogError {
    /// The trace reached its configured record capacity.
    CapacityExceeded {
        /// Configured maximum number of records.
        capacity: usize,
    },
    /// A record was attempted for a different semantic plan.
    PlanMismatch {
        /// Plan associated with the existing trace.
        expected: PlanDigest,
        /// Plan observed on the instance.
        actual: PlanDigest,
    },
    /// The operation would make the trace omit an already queued input.
    PendingEvents {
        /// Number of events already queued before the operation.
        count: usize,
    },
    /// Enqueuing the input event failed.
    EventQueue(EventQueueError),
    /// The instance operation failed after it was invoked.
    Runtime(RuntimeError),
}

/// A bounded deterministic input trace.
pub struct TraceLog {
    capacity: usize,
    records: Vec<TraceRecord>,
    plan: Option<PlanDigest>,
    overflowed: bool,
}

impl TraceLog {
    /// Creates an empty trace with a fixed maximum record count.
    ///
    /// A zero capacity is valid but cannot capture any record.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            records: Vec::with_capacity(capacity),
            plan: None,
            overflowed: false,
        }
    }

    /// Returns the configured maximum record count.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of captured records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns whether no records have been captured.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns whether the trace has reached its record capacity.
    pub fn is_full(&self) -> bool {
        self.records.len() == self.capacity
    }

    /// Returns whether a capture was rejected because the trace was full.
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Returns the plan digest established by the first captured record.
    pub const fn plan_digest(&self) -> Option<PlanDigest> {
        self.plan
    }

    /// Returns captured records in operation order.
    pub fn records(&self) -> &[TraceRecord] {
        &self.records
    }

    /// Clears all records and allows the trace to establish a new plan.
    pub fn clear(&mut self) {
        self.records.clear();
        self.plan = None;
        self.overflowed = false;
    }

    /// Captures the current state as a start operation.
    ///
    /// This method is for callers that already invoked [`Instance::start`].
    /// For an operation that also executes the instance, use [`Self::start`].
    pub fn record_start(&mut self, instance: &Instance) -> Result<(), TraceLogError> {
        self.record(TraceInput::Start, instance)
    }

    /// Captures the current state as the result of one event input.
    ///
    /// The caller is responsible for having already sent and processed the
    /// event. For a single operation that sends and processes the event, use
    /// [`Self::event`].
    pub fn record_event(&mut self, instance: &Instance, event: Event) -> Result<(), TraceLogError> {
        if instance.queued_events() != 0 {
            return Err(TraceLogError::PendingEvents {
                count: instance.queued_events(),
            });
        }
        self.record(TraceInput::Event(event), instance)
    }

    /// Starts an instance and captures the resulting active-state digest.
    pub fn start<C: LifecycleCallback>(
        &mut self,
        instance: &mut Instance,
        callback: &mut C,
    ) -> Result<StepReport, TraceLogError> {
        self.prepare(instance)?;
        if instance.queued_events() != 0 {
            return Err(TraceLogError::PendingEvents {
                count: instance.queued_events(),
            });
        }
        let report = instance.start(callback).map_err(TraceLogError::Runtime)?;
        self.append_prepared(TraceInput::Start, instance);
        Ok(report)
    }

    /// Sends one event, runs to completion, and captures the resulting digest.
    ///
    /// The operation requires an empty queue before sending so every input in
    /// the resulting trace is self-contained.
    pub fn event<C: LifecycleCallback>(
        &mut self,
        instance: &mut Instance,
        event: Event,
        callback: &mut C,
    ) -> Result<StepReport, TraceLogError> {
        self.prepare(instance)?;
        if instance.queued_events() != 0 {
            return Err(TraceLogError::PendingEvents {
                count: instance.queued_events(),
            });
        }
        instance.send(event).map_err(TraceLogError::EventQueue)?;
        let report = instance.run(callback).map_err(TraceLogError::Runtime)?;
        self.append_prepared(TraceInput::Event(event), instance);
        Ok(report)
    }

    /// Replays every captured input against an instance and checks each digest.
    pub fn replay<C: LifecycleCallback>(
        &self,
        instance: &mut Instance,
        callback: &mut C,
    ) -> Result<ReplayReport, ReplayError> {
        if self.overflowed {
            return Err(ReplayError::TraceOverflowed);
        }
        if let Some(expected) = self.plan {
            let actual = PlanDigest::from_plan(instance.plan());
            if actual != expected {
                return Err(ReplayError::PlanMismatch { expected, actual });
            }
        }

        let mut events_replayed = 0;
        for (record_index, record) in self.records.iter().enumerate() {
            match record.input {
                TraceInput::Start => {
                    instance
                        .start(callback)
                        .map_err(|error| ReplayError::Runtime {
                            record_index,
                            error,
                        })?;
                }
                TraceInput::Event(event) => {
                    instance
                        .send(event)
                        .map_err(|error| ReplayError::EventQueue {
                            record_index,
                            error,
                        })?;
                    instance
                        .run(callback)
                        .map_err(|error| ReplayError::Runtime {
                            record_index,
                            error,
                        })?;
                    events_replayed += 1;
                }
            }
            let actual = active_state_digest(instance);
            if actual != record.digest {
                return Err(ReplayError::Diverged(DivergenceReport {
                    record_index,
                    input: record.input,
                    expected: record.digest,
                    actual,
                }));
            }
        }
        Ok(ReplayReport {
            records_replayed: self.records.len(),
            events_replayed,
        })
    }

    fn record(&mut self, input: TraceInput, instance: &Instance) -> Result<(), TraceLogError> {
        self.prepare(instance)?;
        if matches!(input, TraceInput::Event(_)) && instance.queued_events() != 0 {
            return Err(TraceLogError::PendingEvents {
                count: instance.queued_events(),
            });
        }
        self.append_prepared(input, instance);
        Ok(())
    }

    fn prepare(&mut self, instance: &Instance) -> Result<(), TraceLogError> {
        if self.records.len() == self.capacity {
            self.overflowed = true;
            return Err(TraceLogError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        let actual = PlanDigest::from_plan(instance.plan());
        if let Some(expected) = self.plan
            && expected != actual
        {
            return Err(TraceLogError::PlanMismatch { expected, actual });
        }
        Ok(())
    }

    fn append_prepared(&mut self, input: TraceInput, instance: &Instance) {
        let actual = PlanDigest::from_plan(instance.plan());
        if self.plan.is_none() {
            self.plan = Some(actual);
        }
        self.records.push(TraceRecord {
            input,
            digest: active_state_digest(instance),
        });
    }
}

/// Summary returned after a successful trace replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayReport {
    /// Number of operation records checked.
    pub records_replayed: usize,
    /// Number of event inputs applied.
    pub events_replayed: usize,
}

/// The first active-state mismatch found during replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DivergenceReport {
    /// Zero-based record index at which the mismatch was observed.
    pub record_index: usize,
    /// Input operation associated with the mismatch.
    pub input: TraceInput,
    /// Digest captured while recording.
    pub expected: ActiveStateDigest,
    /// Digest observed during replay.
    pub actual: ActiveStateDigest,
}

/// Failure while replaying a [`TraceLog`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayError {
    /// The trace dropped one or more records after reaching capacity.
    TraceOverflowed,
    /// The target instance uses a different semantic plan.
    PlanMismatch {
        /// Plan associated with the trace.
        expected: PlanDigest,
        /// Plan observed on the replay instance.
        actual: PlanDigest,
    },
    /// Sending an input event failed.
    EventQueue {
        /// Record whose input failed.
        record_index: usize,
        /// Queue error returned by the instance.
        error: EventQueueError,
    },
    /// Running the input operation failed.
    Runtime {
        /// Record whose operation failed.
        record_index: usize,
        /// Runtime error returned by the instance.
        error: RuntimeError,
    },
    /// The operation completed but produced a different active configuration.
    Diverged(DivergenceReport),
}

fn mix_u8(hash: &mut u128, value: u8) {
    *hash ^= u128::from(value);
    *hash = hash.wrapping_mul(DIGEST_PRIME);
}

fn mix_u16(hash: &mut u128, value: u16) {
    for byte in value.to_le_bytes() {
        mix_u8(hash, byte);
    }
}

fn mix_u32(hash: &mut u128, value: u32) {
    for byte in value.to_le_bytes() {
        mix_u8(hash, byte);
    }
}

fn mix_u64(hash: &mut u128, value: u64) {
    for byte in value.to_le_bytes() {
        mix_u8(hash, byte);
    }
}

fn mix_len(hash: &mut u128, value: usize) {
    mix_u64(hash, value as u64);
}

fn mix_states(hash: &mut u128, states: &[StateId]) {
    mix_len(hash, states.len());
    for state in states {
        mix_u32(hash, state.0);
    }
}

fn mix_option_state(hash: &mut u128, state: Option<StateId>) {
    match state {
        Some(state) => {
            mix_u8(hash, 1);
            mix_u32(hash, state.0);
        }
        None => mix_u8(hash, 0),
    }
}

fn mix_option_u32(hash: &mut u128, value: Option<u32>) {
    match value {
        Some(value) => {
            mix_u8(hash, 1);
            mix_u32(hash, value);
        }
        None => mix_u8(hash, 0),
    }
}

fn mix_history(hash: &mut u128, history: Option<HistoryMode>) {
    match history {
        Some(HistoryMode::Shallow) => mix_u8(hash, 1),
        Some(HistoryMode::Deep) => mix_u8(hash, 2),
        None => mix_u8(hash, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AlwaysGuard, MicrostepLimit, NoopLifecycle, StateNode, Transition, TransitionKind, TreeDef,
    };
    use alloc::vec;

    #[derive(Clone, Copy)]
    struct FixedGuard(bool);

    impl crate::Guard for FixedGuard {
        fn evaluate(&self, _context: crate::GuardContext) -> bool {
            self.0
        }
    }

    fn plan_with_guard() -> Plan {
        let root = StateNode::compound(
            StateId(0),
            None,
            vec![StateId(1), StateId(2)],
            StateId(1),
            None,
        );
        let source =
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
                StateId(1),
                TransitionTarget::State(StateId(2)),
                Trigger::Event(crate::EventId(7)),
                Some(crate::GuardId(0)),
                0,
                TransitionKind::External,
            ));
        let target = StateNode::atomic(StateId(2), Some(StateId(0)));
        TreeDef {
            root: StateId(0),
            nodes: vec![root, source, target],
        }
        .compile()
        .unwrap()
    }

    #[test]
    fn bounded_capture_rejects_before_mutating_the_instance() {
        let mut instance = Instance::new(plan_with_guard(), 2, MicrostepLimit(8)).unwrap();
        instance.set_guard(crate::GuardId(0), AlwaysGuard);
        let mut trace = TraceLog::new(1);
        let mut callback = NoopLifecycle;
        trace.start(&mut instance, &mut callback).unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(
            trace.event(&mut instance, Event::new(crate::EventId(7)), &mut callback),
            Err(TraceLogError::CapacityExceeded { capacity: 1 })
        );
        assert_eq!(instance.active_states(), &[StateId(0), StateId(1)]);
        assert_eq!(instance.queued_events(), 0);
        assert!(trace.overflowed());
    }

    #[test]
    fn same_inputs_replay_to_the_same_active_configuration() {
        let mut recorded = Instance::new(plan_with_guard(), 2, MicrostepLimit(8)).unwrap();
        recorded.set_guard(crate::GuardId(0), FixedGuard(true));
        let mut trace = TraceLog::new(4);
        let mut callback = NoopLifecycle;
        trace.start(&mut recorded, &mut callback).unwrap();
        trace
            .event(&mut recorded, Event::new(crate::EventId(7)), &mut callback)
            .unwrap();

        let mut replayed = Instance::new(plan_with_guard(), 2, MicrostepLimit(8)).unwrap();
        replayed.set_guard(crate::GuardId(0), FixedGuard(true));
        let report = trace.replay(&mut replayed, &mut callback).unwrap();
        assert_eq!(
            report,
            ReplayReport {
                records_replayed: 2,
                events_replayed: 1,
            }
        );
        assert_eq!(
            active_state_digest(&recorded),
            active_state_digest(&replayed)
        );
    }

    #[test]
    fn replay_reports_the_first_digest_divergence() {
        let mut recorded = Instance::new(plan_with_guard(), 2, MicrostepLimit(8)).unwrap();
        recorded.set_guard(crate::GuardId(0), FixedGuard(true));
        let mut trace = TraceLog::new(4);
        let mut callback = NoopLifecycle;
        trace.start(&mut recorded, &mut callback).unwrap();
        trace
            .event(&mut recorded, Event::new(crate::EventId(7)), &mut callback)
            .unwrap();

        let mut replayed = Instance::new(plan_with_guard(), 2, MicrostepLimit(8)).unwrap();
        replayed.set_guard(crate::GuardId(0), FixedGuard(false));
        let error = trace.replay(&mut replayed, &mut callback).unwrap_err();
        let ReplayError::Diverged(report) = error else {
            panic!("expected digest divergence, got {error:?}");
        };
        assert_eq!(report.record_index, 1);
        assert_eq!(
            report.input,
            TraceInput::Event(Event::new(crate::EventId(7)))
        );
        assert_ne!(report.expected, report.actual);
        assert_eq!(report.actual, active_state_digest(&replayed));
    }

    #[test]
    fn replay_rejects_a_different_plan_before_running_inputs() {
        let mut recorded = Instance::new(plan_with_guard(), 2, MicrostepLimit(8)).unwrap();
        recorded.set_guard(crate::GuardId(0), FixedGuard(true));
        let mut trace = TraceLog::new(2);
        let mut callback = NoopLifecycle;
        trace.start(&mut recorded, &mut callback).unwrap();

        let other_plan = StateNode::atomic(StateId(0), None);
        let other_plan = TreeDef {
            root: StateId(0),
            nodes: vec![other_plan],
        }
        .compile()
        .unwrap();
        let mut replayed = Instance::new(other_plan, 2, MicrostepLimit(8)).unwrap();
        let error = trace.replay(&mut replayed, &mut callback).unwrap_err();
        assert!(matches!(error, ReplayError::PlanMismatch { .. }));
        assert!(replayed.active_states().is_empty());
    }

    #[test]
    fn active_digest_is_order_sensitive() {
        let first = ActiveStateDigest::from_states(&[StateId(0), StateId(1)]);
        let second = ActiveStateDigest::from_states(&[StateId(1), StateId(0)]);
        assert_ne!(first, second);
        assert_eq!(first.state_count(), 2);
    }
}
