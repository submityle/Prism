//! Reflection-driven field-level network delta (design §24.5) — ✅ delivered.
//!
//! The replication layer (`prism_replication`) replicates each entity by
//! sending, every tick, only the component fields that changed. This module is
//! the reflection-side engine for that: it tracks which fields are dirty, encodes
//! just those fields into a compact byte string, and applies that string onto a
//! remote replica so the replica converges to a full snapshot of the source.
//!
//! The three cooperating pieces:
//!
//! 1. [`DirtyMask`] — a growable bitset over a struct's positional field
//!    indices. Mark bits explicitly as fields are written, or derive the mask
//!    from a previous snapshot with [`DirtyMask::changed`] /
//!    [`DirtyMask::changed_in_plan`] (the latter honouring the design's
//!    `#[reflect(replicate)]`/`#[reflect(no_replicate)]` intent through a
//!    [`ReplicationPlan`](crate::ReplicationPlan)).
//! 2. [`encode_delta`]/[`decode_delta`]/[`apply_delta`] — the compact
//!    `varint(field_id) + value` wire codec, built on the existing reflection
//!    serializer so field values inherit [`StableTypeId`](crate::StableTypeId)
//!    tagging and the §24.8 untrusted-input hardening.
//! 3. [`ReplicationState`] — a stateful baseline tracker that snapshots a value
//!    and emits deltas against it tick after tick.
//!
//! # Full-snapshot parity
//! For any `new` reachable from `old` by [`Reflect::apply`](crate::Reflect::apply),
//! encoding the dirty delta of `new` relative to `old` and applying it onto a
//! clone of `old` yields a value equal to `new` — the delta path and the
//! full-snapshot path agree. This identity (and idempotent re-application) is
//! oracle-checked in `tests_net_delta`.
//!
//! Because each dirty field is sent as a whole value and applied with
//! [`Reflect::apply`](crate::Reflect::apply), the per-field round-trip inherits
//! that operation's documented boundaries (see [`crate::diff`]): leaf, string,
//! nested-struct, and grow-only collection fields round-trip; a field whose
//! value cannot be re-applied surfaces a typed [`DeltaError::Apply`] rather than
//! silently diverging.

mod codec;
mod dirty;
mod session;

pub use codec::{apply_delta, decode_and_apply, decode_delta, encode_delta, FieldDelta};
pub use dirty::DirtyMask;
pub use session::ReplicationState;

use crate::apply::ApplyError;
use crate::ser::{DeserializeError, SerializeError};
use core::fmt;

/// An error raised while tracking, encoding, decoding, or applying a field
/// delta.
#[derive(Debug)]
#[non_exhaustive]
pub enum DeltaError {
    /// A delta operation was requested on a value that is not a named-field
    /// struct.
    NotAStruct,
    /// The two compared structs expose a different number of fields.
    FieldCountMismatch {
        /// The old value's field count.
        old: usize,
        /// The new value's field count.
        new: usize,
    },
    /// A field index named by a mask or delta exceeds the struct's field count.
    FieldIndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The struct's actual field count.
        field_count: usize,
    },
    /// A field value failed to serialize into the delta.
    Serialize(SerializeError),
    /// A field value failed to deserialize against the live target field.
    Deserialize(DeserializeError),
    /// A decoded field value was incompatible with the target field.
    Apply(ApplyError),
    /// The delta byte frame was truncated, malformed, or had trailing bytes.
    Malformed,
}

impl fmt::Display for DeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeltaError::NotAStruct => f.write_str("field delta requires a named-field struct"),
            DeltaError::FieldCountMismatch { old, new } => write!(
                f,
                "struct field count changed between snapshots: {old} -> {new}"
            ),
            DeltaError::FieldIndexOutOfRange { index, field_count } => write!(
                f,
                "field index {index} is out of range for a struct with {field_count} fields"
            ),
            DeltaError::Serialize(err) => write!(f, "failed to encode a delta field: {err}"),
            DeltaError::Deserialize(err) => write!(f, "failed to decode a delta field: {err}"),
            DeltaError::Apply(err) => write!(f, "failed to apply a delta field: {err}"),
            DeltaError::Malformed => f.write_str("delta byte frame is malformed or truncated"),
        }
    }
}

impl core::error::Error for DeltaError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            DeltaError::Serialize(err) => Some(err),
            DeltaError::Deserialize(err) => Some(err),
            DeltaError::Apply(err) => Some(err),
            _ => None,
        }
    }
}
