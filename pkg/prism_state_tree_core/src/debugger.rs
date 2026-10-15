//! A bounded, host-neutral debugger contract for StateTree observations.
//!
//! The debugger stores only breakpoint declarations, hit records, migration
//! causes, and optional cursors into a caller-owned [`TraceLog`](crate::TraceLog).
//! It does not execute the state tree, evaluate guards, or own task and
//! migration policy. Every collection has an explicit capacity and this file
//! uses only `alloc` collections.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

use crate::{
    model::GuardId,
    replay::{TraceLog, TraceRecord},
    revision::MigrationKey,
    StateId,
};

/// Maximum number of breakpoint declarations retained by one debugger.
pub const MAX_DEBUGGER_BREAKPOINTS: usize = 1024;

/// Maximum number of hit records retained by one debugger.
pub const MAX_DEBUGGER_HITS: usize = 8192;

/// Maximum number of migration-cause nodes retained by one debugger.
pub const MAX_DEBUGGER_MIGRATION_CAUSES: usize = 4096;

/// Stable host-owned task identity used by task-completion breakpoints.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TaskId(u32);

impl TaskId {
    /// Creates a task identity from a host-defined stable value.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the host-defined stable value.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Terminal result reported by a task-completion observation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TaskOutcome {
    /// The task completed successfully.
    Success,
    /// The task completed with a failure result.
    Failure,
    /// The task was cancelled before successful completion.
    Cancelled,
}

/// A declarative debugger breakpoint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Breakpoint {
    /// Break when a state is entered.
    StateEntered(StateId),
    /// Break when a state is exited.
    StateExited(StateId),
    /// Break when a task reaches a terminal outcome.
    TaskCompleted(TaskId),
    /// Break when a guard is evaluated.
    Guard(GuardId),
    /// Break when a migration for the supplied stable identity is observed.
    Migration(MigrationKey),
}

/// A debugger observation that may match one or more breakpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebugEvent {
    /// A state-entry observation.
    StateEntered(StateId),
    /// A state-exit observation.
    StateExited(StateId),
    /// A terminal task outcome observation.
    TaskCompleted {
        /// Task that reached a terminal outcome.
        task: TaskId,
        /// Terminal result reported by the host.
        outcome: TaskOutcome,
    },
    /// A guard evaluation observation.
    GuardEvaluated {
        /// Guard that was evaluated.
        guard: GuardId,
        /// Result returned by the host guard service.
        passed: bool,
    },
    /// A migration observation with an optional previously recorded cause.
    Migration {
        /// Stable identity being migrated.
        key: MigrationKey,
        /// Parent cause in this debugger's migration-cause store.
        parent: Option<MigrationCauseId>,
    },
}

/// A zero-based cursor into a caller-owned [`TraceLog`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TraceCursor {
    record_index: usize,
}

impl TraceCursor {
    /// Creates a cursor for a trace-record index.
    #[must_use]
    pub const fn new(record_index: usize) -> Self {
        Self { record_index }
    }

    /// Returns the zero-based trace-record index.
    #[must_use]
    pub const fn record_index(self) -> usize {
        self.record_index
    }

    /// Creates a cursor only when the index exists in the supplied trace.
    #[must_use]
    pub fn checked(trace: &TraceLog, record_index: usize) -> Option<Self> {
        if record_index < trace.len() {
            Some(Self::new(record_index))
        } else {
            None
        }
    }

    /// Resolves this cursor against a trace-record slice.
    #[must_use]
    pub fn record(self, records: &[TraceRecord]) -> Option<&TraceRecord> {
        records.get(self.record_index)
    }
}

/// Stable index of a migration-cause node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MigrationCauseId(usize);

impl MigrationCauseId {
    /// Creates an index for a migration-cause node.
    ///
    /// The debugger validates the index when it is used as a parent. Values
    /// created by callers should normally come from [`DebugObservation`].
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the zero-based cause index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// One node in a migration causal chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MigrationCause {
    key: MigrationKey,
    parent: Option<MigrationCauseId>,
    trace: Option<TraceCursor>,
}

impl MigrationCause {
    /// Returns the stable identity associated with this migration step.
    #[must_use]
    pub const fn key(self) -> MigrationKey {
        self.key
    }

    /// Returns the previous cause in the chain, if any.
    #[must_use]
    pub const fn parent(self) -> Option<MigrationCauseId> {
        self.parent
    }

    /// Returns the trace cursor captured for this migration step.
    #[must_use]
    pub const fn trace(self) -> Option<TraceCursor> {
        self.trace
    }
}

/// One breakpoint hit retained by a debugger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HitRecord {
    sequence: u64,
    breakpoint: Breakpoint,
    event: DebugEvent,
    trace: Option<TraceCursor>,
    migration_cause: Option<MigrationCauseId>,
}

impl HitRecord {
    /// Returns the monotonic hit sequence assigned by the debugger.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Returns the breakpoint that matched this hit.
    #[must_use]
    pub const fn breakpoint(self) -> Breakpoint {
        self.breakpoint
    }

    /// Returns the full observation that caused this hit.
    #[must_use]
    pub const fn event(self) -> DebugEvent {
        self.event
    }

    /// Returns the optional trace cursor attached to this hit.
    #[must_use]
    pub const fn trace(self) -> Option<TraceCursor> {
        self.trace
    }

    /// Returns the migration cause created for this observation, if any.
    #[must_use]
    pub const fn migration_cause(self) -> Option<MigrationCauseId> {
        self.migration_cause
    }
}

/// Independent capacities for the debugger's bounded stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DebuggerCapacity {
    breakpoints: usize,
    hits: usize,
    migration_causes: usize,
}

impl DebuggerCapacity {
    /// Creates a debugger capacity declaration.
    #[must_use]
    pub const fn new(breakpoints: usize, hits: usize, migration_causes: usize) -> Self {
        Self {
            breakpoints,
            hits,
            migration_causes,
        }
    }

    /// Returns the breakpoint capacity.
    #[must_use]
    pub const fn breakpoints(self) -> usize {
        self.breakpoints
    }

    /// Returns the hit-record capacity.
    #[must_use]
    pub const fn hits(self) -> usize {
        self.hits
    }

    /// Returns the migration-cause capacity.
    #[must_use]
    pub const fn migration_causes(self) -> usize {
        self.migration_causes
    }
}

/// Which bounded debugger store exceeded its hard contract limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityKind {
    /// Breakpoint declarations.
    Breakpoints,
    /// Hit records.
    Hits,
    /// Migration-cause nodes.
    MigrationCauses,
}

/// Failure while configuring or recording a debugger observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebuggerError {
    /// A requested store capacity exceeds the hard contract limit.
    CapacityLimit {
        /// Store whose limit was exceeded.
        kind: CapacityKind,
        /// Requested capacity.
        requested: usize,
        /// Maximum supported capacity.
        maximum: usize,
    },
    /// Adding a breakpoint would exceed the configured breakpoint capacity.
    BreakpointCapacityExceeded {
        /// Configured breakpoint capacity.
        capacity: usize,
    },
    /// The observation would add more hit records than the configured limit.
    HitCapacityExceeded {
        /// Configured hit-record capacity.
        capacity: usize,
        /// Number of records needed by the observation.
        needed: usize,
    },
    /// A migration observation would exceed the configured cause capacity.
    MigrationCauseCapacityExceeded {
        /// Configured migration-cause capacity.
        capacity: usize,
    },
    /// The breakpoint is already registered.
    DuplicateBreakpoint(Breakpoint),
    /// A migration parent does not identify an already recorded cause.
    UnknownMigrationCause(MigrationCauseId),
}

/// Result details returned after observing one event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DebugObservation {
    hit_count: usize,
    migration_cause: Option<MigrationCauseId>,
}

impl DebugObservation {
    /// Returns the number of breakpoint hits appended by the observation.
    #[must_use]
    pub const fn hit_count(self) -> usize {
        self.hit_count
    }

    /// Returns the migration cause appended by the observation, if any.
    #[must_use]
    pub const fn migration_cause(self) -> Option<MigrationCauseId> {
        self.migration_cause
    }
}

/// A bounded debugger contract over host-supplied semantic observations.
pub struct Debugger {
    capacity: DebuggerCapacity,
    breakpoints: Vec<Breakpoint>,
    hits: Vec<HitRecord>,
    migration_causes: Vec<MigrationCause>,
    next_sequence: u64,
}

impl Debugger {
    /// Creates an empty debugger after validating all hard capacity limits.
    pub fn new(capacity: DebuggerCapacity) -> Result<Self, DebuggerError> {
        validate_capacity(
            CapacityKind::Breakpoints,
            capacity.breakpoints,
            MAX_DEBUGGER_BREAKPOINTS,
        )?;
        validate_capacity(CapacityKind::Hits, capacity.hits, MAX_DEBUGGER_HITS)?;
        validate_capacity(
            CapacityKind::MigrationCauses,
            capacity.migration_causes,
            MAX_DEBUGGER_MIGRATION_CAUSES,
        )?;
        Ok(Self {
            capacity,
            breakpoints: Vec::with_capacity(capacity.breakpoints),
            hits: Vec::with_capacity(capacity.hits),
            migration_causes: Vec::with_capacity(capacity.migration_causes),
            next_sequence: 0,
        })
    }

    /// Creates an empty debugger from three independent capacities.
    pub fn with_capacities(
        breakpoints: usize,
        hits: usize,
        migration_causes: usize,
    ) -> Result<Self, DebuggerError> {
        Self::new(DebuggerCapacity::new(breakpoints, hits, migration_causes))
    }

    /// Returns the configured store capacities.
    #[must_use]
    pub const fn capacity(&self) -> DebuggerCapacity {
        self.capacity
    }

    /// Returns the registered breakpoints in declaration order.
    #[must_use]
    pub fn breakpoints(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Returns retained hits in sequence order.
    #[must_use]
    pub fn hits(&self) -> &[HitRecord] {
        &self.hits
    }

    /// Returns retained migration causes in insertion order.
    #[must_use]
    pub fn migration_causes(&self) -> &[MigrationCause] {
        &self.migration_causes
    }

    /// Adds a breakpoint unless it is already registered.
    pub fn add_breakpoint(&mut self, breakpoint: Breakpoint) -> Result<(), DebuggerError> {
        if self.breakpoints.contains(&breakpoint) {
            return Err(DebuggerError::DuplicateBreakpoint(breakpoint));
        }
        if self.breakpoints.len() == self.capacity.breakpoints {
            return Err(DebuggerError::BreakpointCapacityExceeded {
                capacity: self.capacity.breakpoints,
            });
        }
        self.breakpoints.push(breakpoint);
        Ok(())
    }

    /// Removes a breakpoint and returns whether it was present.
    pub fn remove_breakpoint(&mut self, breakpoint: Breakpoint) -> bool {
        if let Some(index) = self.breakpoints.iter().position(|item| *item == breakpoint) {
            self.breakpoints.remove(index);
            true
        } else {
            false
        }
    }

    /// Removes all registered breakpoints without discarding hit records.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Observes an event without attaching a trace cursor.
    pub fn observe(&mut self, event: DebugEvent) -> Result<DebugObservation, DebuggerError> {
        self.observe_at(event, None)
    }

    /// Observes an event and attaches the supplied trace cursor to its hits.
    pub fn observe_at(
        &mut self,
        event: DebugEvent,
        trace: Option<TraceCursor>,
    ) -> Result<DebugObservation, DebuggerError> {
        let matching = self
            .breakpoints
            .iter()
            .copied()
            .filter(|breakpoint| breakpoint.matches(event))
            .collect::<Vec<_>>();
        if self.hits.len() + matching.len() > self.capacity.hits {
            return Err(DebuggerError::HitCapacityExceeded {
                capacity: self.capacity.hits,
                needed: matching.len(),
            });
        }
        let hit_count = matching.len();

        let migration_cause = if let DebugEvent::Migration { key, parent } = event {
            if let Some(parent) = parent
                && parent.index() >= self.migration_causes.len()
            {
                return Err(DebuggerError::UnknownMigrationCause(parent));
            }
            if self.migration_causes.len() == self.capacity.migration_causes {
                return Err(DebuggerError::MigrationCauseCapacityExceeded {
                    capacity: self.capacity.migration_causes,
                });
            }
            let id = MigrationCauseId::new(self.migration_causes.len());
            self.migration_causes
                .push(MigrationCause { key, parent, trace });
            Some(id)
        } else {
            None
        };

        for breakpoint in matching {
            self.hits.push(HitRecord {
                sequence: self.next_sequence,
                breakpoint,
                event,
                trace,
                migration_cause,
            });
            self.next_sequence = self.next_sequence.wrapping_add(1);
        }
        Ok(DebugObservation {
            hit_count,
            migration_cause,
        })
    }

    /// Returns a migration cause by id.
    #[must_use]
    pub fn migration_cause(&self, id: MigrationCauseId) -> Option<MigrationCause> {
        self.migration_causes.get(id.index()).copied()
    }

    /// Returns an iterator from a migration cause toward its root cause.
    #[must_use]
    pub fn migration_chain(&self, id: MigrationCauseId) -> Option<MigrationChain<'_>> {
        if id.index() < self.migration_causes.len() {
            Some(MigrationChain {
                causes: &self.migration_causes,
                next: Some(id),
            })
        } else {
            None
        }
    }

    /// Clears hit and migration-cause history while keeping breakpoints.
    pub fn clear_history(&mut self) {
        self.hits.clear();
        self.migration_causes.clear();
    }
}

/// Iterator over a migration cause and its ancestors, newest first.
pub struct MigrationChain<'a> {
    causes: &'a [MigrationCause],
    next: Option<MigrationCauseId>,
}

impl Iterator for MigrationChain<'_> {
    type Item = MigrationCause;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.next?;
        let cause = *self.causes.get(id.index())?;
        self.next = cause.parent;
        Some(cause)
    }
}

impl Breakpoint {
    fn matches(self, event: DebugEvent) -> bool {
        match (self, event) {
            (Self::StateEntered(expected), DebugEvent::StateEntered(actual)) => expected == actual,
            (Self::StateExited(expected), DebugEvent::StateExited(actual)) => expected == actual,
            (Self::TaskCompleted(expected), DebugEvent::TaskCompleted { task, .. }) => {
                expected == task
            }
            (Self::Guard(expected), DebugEvent::GuardEvaluated { guard, .. }) => expected == guard,
            (Self::Migration(expected), DebugEvent::Migration { key, .. }) => expected == key,
            _ => false,
        }
    }
}

fn validate_capacity(
    kind: CapacityKind,
    requested: usize,
    maximum: usize,
) -> Result<(), DebuggerError> {
    if requested > maximum {
        Err(DebuggerError::CapacityLimit {
            kind,
            requested,
            maximum,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActiveStateDigest, Event, TraceInput};

    fn key(value: u128) -> MigrationKey {
        MigrationKey::new(value)
    }

    #[test]
    fn all_breakpoint_kinds_match_their_event() {
        let mut debugger = Debugger::with_capacities(5, 5, 1).unwrap();
        debugger
            .add_breakpoint(Breakpoint::StateEntered(StateId(1)))
            .unwrap();
        debugger
            .add_breakpoint(Breakpoint::StateExited(StateId(2)))
            .unwrap();
        debugger
            .add_breakpoint(Breakpoint::TaskCompleted(TaskId::new(3)))
            .unwrap();
        debugger
            .add_breakpoint(Breakpoint::Guard(GuardId(4)))
            .unwrap();
        debugger
            .add_breakpoint(Breakpoint::Migration(key(5)))
            .unwrap();

        assert_eq!(
            debugger
                .observe(DebugEvent::StateEntered(StateId(1)))
                .unwrap()
                .hit_count(),
            1
        );
        assert_eq!(
            debugger
                .observe(DebugEvent::StateExited(StateId(2)))
                .unwrap()
                .hit_count(),
            1
        );
        assert_eq!(
            debugger
                .observe(DebugEvent::TaskCompleted {
                    task: TaskId::new(3),
                    outcome: TaskOutcome::Success,
                })
                .unwrap()
                .hit_count(),
            1
        );
        assert_eq!(
            debugger
                .observe(DebugEvent::GuardEvaluated {
                    guard: GuardId(4),
                    passed: false,
                })
                .unwrap()
                .hit_count(),
            1
        );
        assert_eq!(
            debugger
                .observe(DebugEvent::Migration {
                    key: key(5),
                    parent: None,
                })
                .unwrap()
                .hit_count(),
            1
        );
        assert_eq!(debugger.hits().len(), 5);
    }

    #[test]
    fn migration_chain_keeps_trace_cursors() {
        let mut debugger = Debugger::with_capacities(1, 1, 2).unwrap();
        debugger
            .add_breakpoint(Breakpoint::Migration(key(2)))
            .unwrap();

        let root = debugger
            .observe_at(
                DebugEvent::Migration {
                    key: key(1),
                    parent: None,
                },
                Some(TraceCursor::new(7)),
            )
            .unwrap()
            .migration_cause()
            .unwrap();
        let child = debugger
            .observe_at(
                DebugEvent::Migration {
                    key: key(2),
                    parent: Some(root),
                },
                Some(TraceCursor::new(8)),
            )
            .unwrap();

        let child_id = child.migration_cause().unwrap();
        let chain = debugger
            .migration_chain(child_id)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].key(), key(2));
        assert_eq!(chain[0].trace(), Some(TraceCursor::new(8)));
        assert_eq!(chain[1].key(), key(1));
        assert_eq!(chain[1].trace(), Some(TraceCursor::new(7)));
        assert_eq!(debugger.hits()[0].migration_cause(), Some(child_id));
    }

    #[test]
    fn cursor_resolves_against_trace_records() {
        let record = TraceRecord {
            input: TraceInput::Event(Event::new(crate::EventId(9))),
            digest: ActiveStateDigest::from_states(&[StateId(1)]),
        };
        let cursor = TraceCursor::new(0);
        assert_eq!(cursor.record(&[record]), Some(&record));
        assert_eq!(TraceCursor::new(1).record(&[record]), None);
    }

    #[test]
    fn capacity_errors_are_reported_without_partial_observation() {
        let requested = MAX_DEBUGGER_BREAKPOINTS + 1;
        assert!(matches!(
            Debugger::with_capacities(requested, 0, 0),
            Err(DebuggerError::CapacityLimit {
                kind: CapacityKind::Breakpoints,
                requested: actual,
                maximum: MAX_DEBUGGER_BREAKPOINTS,
            })
                if actual == requested
        ));

        let mut no_breakpoints = Debugger::with_capacities(0, 0, 0).unwrap();
        assert_eq!(
            no_breakpoints.add_breakpoint(Breakpoint::StateEntered(StateId(1))),
            Err(DebuggerError::BreakpointCapacityExceeded { capacity: 0 })
        );

        let mut debugger = Debugger::with_capacities(1, 0, 0).unwrap();
        debugger
            .add_breakpoint(Breakpoint::StateEntered(StateId(1)))
            .unwrap();
        assert_eq!(
            debugger.observe(DebugEvent::StateEntered(StateId(1))),
            Err(DebuggerError::HitCapacityExceeded {
                capacity: 0,
                needed: 1,
            })
        );
        assert!(debugger.hits().is_empty());

        let mut no_causes = Debugger::with_capacities(0, 0, 0).unwrap();
        assert_eq!(
            no_causes.observe(DebugEvent::Migration {
                key: key(1),
                parent: None,
            }),
            Err(DebuggerError::MigrationCauseCapacityExceeded { capacity: 0 })
        );
        assert!(no_causes.migration_causes().is_empty());
    }

    #[test]
    fn invalid_migration_parent_does_not_append_a_cause() {
        let mut debugger = Debugger::with_capacities(0, 0, 1).unwrap();
        assert_eq!(
            debugger.observe(DebugEvent::Migration {
                key: key(1),
                parent: Some(MigrationCauseId::new(4)),
            }),
            Err(DebuggerError::UnknownMigrationCause(MigrationCauseId::new(
                4
            )))
        );
        assert!(debugger.migration_causes().is_empty());
    }
}
