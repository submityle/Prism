//! A deterministic, pure-data StateTree ST0 semantic kernel.
//!
//! The crate implements hierarchical atomic/compound/parallel states,
//! eventless and event-triggered transitions, LCA lifecycle ordering,
//! shallow/deep history, a bounded event queue, and run-to-completion
//! microsteps. It deliberately has no ECS, renderer, networking, editor, or
//! task-system dependency.
#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod model;
mod runtime;
mod snapshot;

pub use model::{
    EventId, GuardId, HistoryMode, NodeKind, Plan, PlanError, StateId, StateNode, Transition,
    TransitionKind, TransitionTarget, TreeDef, Trigger,
};
pub use runtime::{
    AlwaysGuard, Event, EventQueueError, Guard, GuardContext, Instance, LifecycleCallback,
    LogEntry, LogKind, MicrostepLimit, NoopLifecycle, RuntimeError, StepReport,
};
pub use snapshot::{RestoreError, Snapshot, SnapshotCodecError};

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests;
