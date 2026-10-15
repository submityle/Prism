//! Deterministic persistence primitives for Prism world overrides.
//!
//! The crate is the M5/M8 persistence core described by
//! `docs/prism_world_system_design_zh.md`. It deliberately owns no scene,
//! ECS, filesystem, network, or GPU policy. Its input and output are stable
//! identifiers, field-level override deltas, immutable transactions, and
//! replayable state.
//!
//! The binary codec is self-contained and uses little-endian integers. It
//! does not depend on `serde`, `std`, or any external hashing implementation.
//! The crate is suitable for embedded builds because its only allocation
//! dependency is `alloc`.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod codec;
mod crdt;
#[cfg(feature = "std")]
mod durable;
mod hash;
mod log;
mod timeline;
mod types;

pub use codec::{
    decode_delta, decode_log, decode_snapshot, decode_transaction, encode_delta, encode_log,
    encode_snapshot, encode_transaction, CodecError,
};
pub use crdt::{CrdtError, CrdtState, LwwRegister, MergeKey};
#[cfg(feature = "std")]
pub use durable::{
    FileJournal, JournalError, JournalRecovery, RecoveryReason, SnapshotError, SnapshotStore,
};
pub use hash::{checksum32, hash64};
pub use log::{AppendOnlyLog, LogError, ObjectLogIter};
pub use timeline::{Snapshot, Timeline, TimelineError};
pub use types::{
    ActorId, DeltaError, FieldId, FieldOverride, LogicalClock, ObjectState, OverrideDelta,
    OverrideValue, PersistentId, Transaction, TransactionId, ValueError, WorldState,
};

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "std"))]
mod durable_tests;
