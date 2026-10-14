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

mod binding;
mod codec;
mod frame;
mod model;
mod runtime;
mod schema;
mod snapshot;

pub use binding::{
    BindingPath, BindingPathError, BindingSlot, DependencyError, DependencyPlan,
    EvaluatorDependency,
};
pub use codec::{
    FrameMigration, FrameSnapshot, SlotCodec, SlotDecl, SlotGuid, SlotType, SlotValue,
};
pub use frame::{FrameError, FrameField, FrameLayout, InstanceFrame};
pub use model::{
    EventId, GuardId, HistoryMode, NodeKind, Plan, PlanError, StateId, StateNode, Transition,
    TransitionKind, TransitionTarget, TreeDef, Trigger,
};
pub use runtime::{
    AlwaysGuard, Event, EventQueueError, Guard, GuardContext, Instance, LifecycleCallback,
    LogEntry, LogKind, MicrostepLimit, NoopLifecycle, RuntimeError, StepReport,
};
pub use schema::{
    ContextRef, Schema, SchemaCapabilities, SchemaDescriptor, SchemaError, SchemaId, SchemaVersion,
};
pub use snapshot::{RestoreError, Snapshot, SnapshotCodecError};

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod binding_tests;
#[cfg(test)]
mod tests;
