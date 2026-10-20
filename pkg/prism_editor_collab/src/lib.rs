//! Deterministic, transport-neutral collaboration data for the Prism editor.
//!
//! The crate deliberately owns no transport, UI, gameplay, ECS, or engine
//! policy. It provides the state and ordering primitives that a transport
//! adapter can replicate:
//!
//! - hybrid logical timestamps and stable command envelopes;
//! - causal buffering, deduplication, and deterministic replay;
//! - field-level last-writer-wins registers with tombstones;
//! - bounded replay history with snapshot-aware recovery;
//! - explicit presence and advisory soft-lock lease lifecycles.
//!
//! Field registers are the CRDT in this crate. A field write wins by
//! `(HLC timestamp, actor id, sequence)`, and deletes are retained as
//! tombstones so an older write cannot resurrect a field. Maps and lists are
//! intentionally represented by separate field paths; no hidden gameplay
//! schema is imposed.

#![forbid(unsafe_code)]

mod clock;
mod command;
mod document;
mod lock;
mod presence;

pub use clock::{ClockError, HlcClock, HlcTimestamp};
pub use command::{
    ActorId, CommandEnvelope, CommandError, CommandId, CommandLimits, CommandPayload, DomainId,
    FieldPath, PathSegment, Value,
};
pub use document::{
    ApplyError, ApplyOutcome, DocumentState, HistoryEntry, HistoryError, HistoryInfo,
    ReceiveOutcome, ReplayError, Replica, ReplicaLimits, Snapshot,
};
pub use lock::{LockError, LockLease, LockRecord, ResourceId, SoftLockRegistry};
pub use presence::{
    PresenceError, PresenceLease, PresenceRecord, PresenceRegistry, PresenceStatus,
};
