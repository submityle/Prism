//! Exact scatter deletion snapshots stored in scene-owned override fields.

use alloc::vec::Vec;
use core::{fmt, mem::size_of};

use prism_world_persist::{
    DeltaError, FieldId, OverrideDelta, OverrideValue, PersistentId, WorldState,
};
use prism_world_residency::{
    FieldScatterOutput, ScatterCandidateKey, ScatterDeletionBudget, ScatterDeletionError,
    ScatterDeletionSet, ScatterOutput, SectorCoord, SCATTER_DELETION_VERSION,
};

/// Bytes in the canonical `PSDL`, revision, and key-count header.
pub const SCATTER_DELETION_HEADER_BYTES: usize = 16;
/// Canonical little-endian key width; independent of Rust struct layout.
pub const SCATTER_DELETION_KEY_BYTES: usize = 52;
const MAGIC: [u8; 4] = *b"PSDL";

/// Admission limits for one immutable deletion field, not aggregate process RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterDeletionCodecBudget {
    /// Maximum serialized bytes, including the fixed header.
    pub max_encoded_bytes: usize,
    /// Maximum decoded key count and allocated key Vec payload.
    pub deletions: ScatterDeletionBudget,
}

/// Invalid, noncanonical, or over-budget persisted scatter state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScatterPersistenceError {
    /// The serialized payload exceeds its independent admission limit.
    EncodedByteBudgetExceeded {
        /// Required complete payload size.
        required: usize,
        /// Configured byte limit.
        limit: usize,
    },
    /// The payload is not a scatter-deletion field.
    InvalidMagic,
    /// A different deletion-format revision requires an explicit migration.
    UnsupportedVersion(u32),
    /// Truncation, trailing bytes, or an inconsistent declared key count.
    InvalidLength {
        /// Required complete payload size.
        expected: usize,
        /// Supplied payload size.
        actual: usize,
    },
    /// Key order is not strictly increasing, including duplicate keys.
    NonCanonicalKeys,
    /// The bound field exists but does not contain opaque bytes.
    UnexpectedValueType,
    /// A size or count calculation cannot be represented by this host.
    ArithmeticOverflow,
    /// Fallible reservation of serialized bytes or decoded keys failed.
    AllocationFailed,
    /// Key validation or decoded-memory admission failed.
    Deletions(ScatterDeletionError),
    /// The persistence layer rejected a field delta.
    Delta(DeltaError),
}

impl fmt::Display for ScatterPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncodedByteBudgetExceeded { required, limit } => write!(
                f,
                "scatter deletion payload requires {required} bytes, limit {limit}"
            ),
            Self::InvalidMagic => f.write_str("invalid scatter deletion magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported scatter deletion revision {version}")
            }
            Self::InvalidLength { expected, actual } => write!(
                f,
                "scatter deletion payload length {actual}, expected {expected}"
            ),
            Self::NonCanonicalKeys => f.write_str("scatter deletion keys are not strictly sorted"),
            Self::UnexpectedValueType => f.write_str("scatter deletion field is not bytes"),
            Self::ArithmeticOverflow => f.write_str("scatter deletion size overflow"),
            Self::AllocationFailed => f.write_str("scatter deletion allocation failed"),
            Self::Deletions(error) => write!(f, "scatter deletion keys: {error}"),
            Self::Delta(error) => write!(f, "scatter deletion delta: {error:?}"),
        }
    }
}

impl core::error::Error for ScatterPersistenceError {}

fn encoded_len(count: usize) -> Result<usize, ScatterPersistenceError> {
    count
        .checked_mul(SCATTER_DELETION_KEY_BYTES)
        .and_then(|bytes| bytes.checked_add(SCATTER_DELETION_HEADER_BYTES))
        .ok_or(ScatterPersistenceError::ArithmeticOverflow)
}

fn check_wire_budget(required: usize, limit: usize) -> Result<(), ScatterPersistenceError> {
    if required > limit {
        return Err(ScatterPersistenceError::EncodedByteBudgetExceeded { required, limit });
    }
    Ok(())
}

/// Encodes a canonical full-key snapshot, with no hashing or lossy ID conversion.
///
/// An empty set encodes as the 16-byte header. The Vec's logical length is bounded
/// by `max_encoded_bytes`; allocator rounding/metadata are not an RSS guarantee.
pub fn encode_scatter_deletions(
    deletions: &ScatterDeletionSet,
    max_encoded_bytes: usize,
) -> Result<Vec<u8>, ScatterPersistenceError> {
    let length = encoded_len(deletions.keys().len())?;
    check_wire_budget(length, max_encoded_bytes)?;
    let count = u64::try_from(deletions.keys().len())
        .map_err(|_| ScatterPersistenceError::ArithmeticOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| ScatterPersistenceError::AllocationFailed)?;
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&SCATTER_DELETION_VERSION.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    for key in deletions.keys() {
        bytes.extend_from_slice(&key.kernel_version.to_le_bytes());
        bytes.extend_from_slice(&key.seed.to_le_bytes());
        bytes.extend_from_slice(&key.version.to_le_bytes());
        bytes.extend_from_slice(&key.salt.to_le_bytes());
        bytes.extend_from_slice(&key.grid_spacing_mm.to_le_bytes());
        for coordinate in [key.grid.x, key.grid.y, key.grid.z] {
            bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn read<const N: usize>(bytes: &[u8], offset: usize) -> [u8; N] {
    // Only called after exact header/record size validation, with fixed offsets.
    bytes[offset..offset + N]
        .try_into()
        .expect("validated scatter deletion record")
}

/// Validates format, wire/count/memory limits, complete keys and canonical order.
///
/// Length/count/byte admission is checked before key allocation. Unknown kernel
/// revisions, zero spacing and unreconstructable coordinates are errors. Bad
/// data never becomes an empty deletion set. No world state is mutated.
pub fn decode_scatter_deletions(
    bytes: &[u8],
    budget: ScatterDeletionCodecBudget,
) -> Result<ScatterDeletionSet, ScatterPersistenceError> {
    check_wire_budget(bytes.len(), budget.max_encoded_bytes)?;
    if bytes.len() < SCATTER_DELETION_HEADER_BYTES {
        return Err(ScatterPersistenceError::InvalidLength {
            expected: SCATTER_DELETION_HEADER_BYTES,
            actual: bytes.len(),
        });
    }
    if bytes[..4] != MAGIC {
        return Err(ScatterPersistenceError::InvalidMagic);
    }
    let version = u32::from_le_bytes(read(bytes, 4));
    if version != SCATTER_DELETION_VERSION {
        return Err(ScatterPersistenceError::UnsupportedVersion(version));
    }
    let count = usize::try_from(u64::from_le_bytes(read(bytes, 8)))
        .map_err(|_| ScatterPersistenceError::ArithmeticOverflow)?;
    let expected = encoded_len(count)?;
    if expected != bytes.len() {
        return Err(ScatterPersistenceError::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    if count > budget.deletions.max_keys {
        return Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::KeyBudgetExceeded {
                required: count,
                limit: budget.deletions.max_keys,
            },
        ));
    }
    let key_bytes = count
        .checked_mul(size_of::<ScatterCandidateKey>())
        .ok_or(ScatterPersistenceError::ArithmeticOverflow)?;
    if key_bytes > budget.deletions.max_key_bytes {
        return Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::KeyBytesBudgetExceeded {
                required: key_bytes,
                limit: budget.deletions.max_key_bytes,
            },
        ));
    }
    let mut keys = Vec::new();
    keys.try_reserve_exact(count)
        .map_err(|_| ScatterPersistenceError::AllocationFailed)?;
    // Account for allocator-provided spare key capacity before filling the Vec.
    let allocated = keys
        .capacity()
        .checked_mul(size_of::<ScatterCandidateKey>())
        .ok_or(ScatterPersistenceError::ArithmeticOverflow)?;
    if allocated > budget.deletions.max_key_bytes {
        return Err(ScatterPersistenceError::Deletions(
            ScatterDeletionError::KeyBytesBudgetExceeded {
                required: allocated,
                limit: budget.deletions.max_key_bytes,
            },
        ));
    }
    for record in bytes[SCATTER_DELETION_HEADER_BYTES..].chunks_exact(SCATTER_DELETION_KEY_BYTES) {
        let key = ScatterCandidateKey {
            kernel_version: u32::from_le_bytes(read(record, 0)),
            seed: u64::from_le_bytes(read(record, 4)),
            version: u32::from_le_bytes(read(record, 12)),
            salt: u64::from_le_bytes(read(record, 16)),
            grid_spacing_mm: u32::from_le_bytes(read(record, 24)),
            grid: SectorCoord {
                x: i64::from_le_bytes(read(record, 28)),
                y: i64::from_le_bytes(read(record, 36)),
                z: i64::from_le_bytes(read(record, 44)),
            },
        };
        if keys.last().is_some_and(|previous| previous >= &key) {
            return Err(ScatterPersistenceError::NonCanonicalKeys);
        }
        keys.push(key);
    }
    ScatterDeletionSet::new(keys, budget.deletions).map_err(ScatterPersistenceError::Deletions)
}

/// A caller-selected override field on the scene's authoritative scatter node.
///
/// This is not a candidate's ID. Candidate identity is this node plus its entire
/// structural key; a dense proxy index or truncated hash is never persisted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterDeletionBinding {
    /// Scene-owned persistent identity of the author scatter node.
    pub object: PersistentId,
    /// Caller-owned field/schema identifier for the whole deletion snapshot.
    pub field: FieldId,
}

impl ScatterDeletionBinding {
    /// Reads one authoritative field; absent/deleted fields mean no deletions.
    pub fn read(
        self,
        state: &WorldState,
        budget: ScatterDeletionCodecBudget,
    ) -> Result<ScatterDeletionSnapshot, ScatterPersistenceError> {
        let deletions = match state.get(self.object, self.field) {
            None | Some(OverrideValue::Delete) => {
                ScatterDeletionSet::new(Vec::new(), budget.deletions)
                    .map_err(ScatterPersistenceError::Deletions)?
            }
            Some(OverrideValue::Bytes(bytes)) => decode_scatter_deletions(bytes, budget)?,
            Some(_) => return Err(ScatterPersistenceError::UnexpectedValueType),
        };
        Ok(self.snapshot(deletions))
    }

    /// Explicitly associates an immutable set with this scene node and field.
    pub fn snapshot(self, deletions: ScatterDeletionSet) -> ScatterDeletionSnapshot {
        ScatterDeletionSnapshot {
            binding: self,
            deletions,
        }
    }

    /// Builds a whole-field replacement for `Timeline::commit`, not a transaction.
    ///
    /// The timeline must capture the actual prior field value for undo. Empty
    /// sets remove the field (`Delete`) and need zero serialized payload bytes.
    /// Nonempty sets use the canonical codec. Persistence/CRDT merging remains
    /// field-level LWW, not a per-key OR-set or concurrent union.
    pub fn delta(
        self,
        deletions: &ScatterDeletionSet,
        max_encoded_bytes: usize,
    ) -> Result<OverrideDelta, ScatterPersistenceError> {
        let value = if deletions.keys().is_empty() {
            OverrideValue::Delete
        } else {
            OverrideValue::Bytes(encode_scatter_deletions(deletions, max_encoded_bytes)?)
        };
        OverrideDelta::single(self.field, value).map_err(ScatterPersistenceError::Delta)
    }
}

/// Complete immutable cache dependency: node/field binding and exact deletion set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScatterDeletionSnapshot {
    binding: ScatterDeletionBinding,
    deletions: ScatterDeletionSet,
}

impl ScatterDeletionSnapshot {
    /// Returns the author node/field identity.
    pub const fn binding(&self) -> ScatterDeletionBinding {
        self.binding
    }

    /// Returns the exact full-key deletion snapshot.
    pub const fn deletions(&self) -> &ScatterDeletionSet {
        &self.deletions
    }

    /// Filters a disposable source buffer in place, retaining the complete dependency.
    ///
    /// The embedded source cache key describes upstream generation only. A cache
    /// consumer must additionally compare `snapshot`, including its binding.
    pub fn filter_source(&self, source: ScatterOutput) -> PersistedScatterOutput {
        let filtered = self.deletions.filter_source(source);
        PersistedScatterOutput {
            source: filtered.source,
            snapshot: self.clone(),
            removed: filtered.removed,
        }
    }

    /// Stably compacts points and aligned author attributes without new Vecs.
    pub fn filter_field(
        &self,
        source: FieldScatterOutput,
    ) -> Result<PersistedFieldScatterOutput, ScatterPersistenceError> {
        let filtered = self
            .deletions
            .filter_field(source)
            .map_err(ScatterPersistenceError::Deletions)?;
        Ok(PersistedFieldScatterOutput {
            source: filtered.source,
            snapshot: self.clone(),
            removed: filtered.removed,
        })
    }
}

/// Derived source data plus its additional persistent deletion cache dependency.
#[derive(Debug, Eq, PartialEq)]
pub struct PersistedScatterOutput {
    /// Filtered points; cache key/counters retain their upstream-generation meaning.
    pub source: ScatterOutput,
    /// Full immutable authority snapshot used by the filter.
    pub snapshot: ScatterDeletionSnapshot,
    /// Number of input points removed.
    pub removed: usize,
}

/// Derived author-field data plus its persistent deletion cache dependency.
#[derive(Debug, Eq, PartialEq)]
pub struct PersistedFieldScatterOutput {
    /// Filtered points and attributes; upstream generation key/counters unchanged.
    pub source: FieldScatterOutput,
    /// Full immutable authority snapshot used by the filter.
    pub snapshot: ScatterDeletionSnapshot,
    /// Number of input points removed.
    pub removed: usize,
}
