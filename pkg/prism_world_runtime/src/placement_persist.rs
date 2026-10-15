//! Full placement identities stored only in scene-owned override fields.

use alloc::vec::Vec;
use core::{fmt, mem::size_of};

use prism_world_persist::{
    DeltaError, FieldId, OverrideDelta, OverrideValue, PersistentId, WorldState,
};
use prism_world_residency::{ScatterCandidateKey, SectorCoord};
use prism_world_terrain::{
    filter_heightfield_placement, HeightfieldPlacementDeletionBudget,
    HeightfieldPlacementDeletionError, HeightfieldPlacementDeletionSet, HeightfieldPlacementKey,
    HeightfieldPlacementOutput, HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
};

/// Version of the canonical `PPLD` little-endian wire format.
pub const PLACEMENT_DELETION_CODEC_VERSION: u32 = 1;
/// Four magic bytes, u32 codec version, and u64 key count.
pub const PLACEMENT_DELETION_HEADER_BYTES: usize = 16;
/// Placement version plus all 52 canonical source-key bytes, including grid Y.
pub const PLACEMENT_DELETION_KEY_BYTES: usize = 56;
const MAGIC: [u8; 4] = *b"PPLD";

/// Limits for one serialized field and its immutable decoded key capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementDeletionCodecBudget {
    /// Maximum complete serialized length, including the header.
    pub max_encoded_bytes: usize,
    /// Independent decoded count and actual key Vec capacity limits.
    pub deletions: HeightfieldPlacementDeletionBudget,
}

/// Malformed, incompatible, or over-budget placement deletion state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlacementPersistenceError {
    /// Complete wire length exceeds its independent limit.
    EncodedByteBudgetExceeded {
        /// Required complete bytes.
        required: usize,
        /// Allowed complete bytes.
        limit: usize,
    },
    /// Not a placement-deletion payload; old 3D scatter payloads are rejected.
    InvalidMagic,
    /// Unknown codec revision; migration must be explicit.
    UnsupportedVersion(u32),
    /// Truncated header/keys, inconsistent count, or trailing bytes.
    InvalidLength {
        /// Required complete length.
        expected: usize,
        /// Actual supplied length.
        actual: usize,
    },
    /// Full keys are not strictly ascending, including duplicates.
    NonCanonicalKeys,
    /// An existing bound override is neither bytes nor Delete.
    UnexpectedValueType,
    /// Host count/length/capacity arithmetic overflow.
    ArithmeticOverflow,
    /// Fallible key/wire Vec reservation failed.
    AllocationFailed,
    /// Invalid key, decoded-memory limit, or misaligned output columns.
    Deletions(HeightfieldPlacementDeletionError),
    /// The persistence core rejected an override delta.
    Delta(DeltaError),
}

impl fmt::Display for PlacementPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncodedByteBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "placement deletion payload requires {required} bytes, limit {limit}"
                )
            }
            Self::InvalidMagic => f.write_str("invalid placement deletion magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported placement deletion codec revision {version}")
            }
            Self::InvalidLength { expected, actual } => {
                write!(
                    f,
                    "placement deletion payload length {actual}, expected {expected}"
                )
            }
            Self::NonCanonicalKeys => {
                f.write_str("placement deletion keys are not strictly sorted")
            }
            Self::UnexpectedValueType => f.write_str("placement deletion field is not bytes"),
            Self::ArithmeticOverflow => {
                f.write_str("placement deletion count or byte size overflow")
            }
            Self::AllocationFailed => f.write_str("placement deletion reservation failed"),
            Self::Deletions(error) => write!(f, "placement deletion keys: {error}"),
            Self::Delta(error) => write!(f, "placement deletion delta: {error:?}"),
        }
    }
}

impl core::error::Error for PlacementPersistenceError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Deletions(error) => Some(error),
            _ => None,
        }
    }
}

fn encoded_len(count: usize) -> Result<usize, PlacementPersistenceError> {
    count
        .checked_mul(PLACEMENT_DELETION_KEY_BYTES)
        .and_then(|bytes| bytes.checked_add(PLACEMENT_DELETION_HEADER_BYTES))
        .ok_or(PlacementPersistenceError::ArithmeticOverflow)
}

fn check_wire(required: usize, limit: usize) -> Result<(), PlacementPersistenceError> {
    if required > limit {
        return Err(PlacementPersistenceError::EncodedByteBudgetExceeded { required, limit });
    }
    Ok(())
}

fn check_keys(
    count: usize,
    capacity: usize,
    budget: HeightfieldPlacementDeletionBudget,
) -> Result<(), PlacementPersistenceError> {
    if count > budget.max_keys {
        return Err(PlacementPersistenceError::Deletions(
            HeightfieldPlacementDeletionError::KeyBudgetExceeded {
                required: count,
                limit: budget.max_keys,
            },
        ));
    }
    let required = capacity
        .checked_mul(size_of::<HeightfieldPlacementKey>())
        .ok_or(PlacementPersistenceError::ArithmeticOverflow)?;
    if required > budget.max_key_bytes {
        return Err(PlacementPersistenceError::Deletions(
            HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
                required,
                limit: budget.max_key_bytes,
            },
        ));
    }
    Ok(())
}

/// Encodes full identities without hashes, dense slots, or terrain geometry.
///
/// Wire-length admission precedes one fallible reservation. Empty sets have a
/// 16-byte header. This wire limit bounds length, not allocator overhead or RSS;
/// decoded key capacity is admitted separately when reading.
pub fn encode_placement_deletions(
    deletions: &HeightfieldPlacementDeletionSet,
    max_encoded_bytes: usize,
) -> Result<Vec<u8>, PlacementPersistenceError> {
    let length = encoded_len(deletions.keys().len())?;
    check_wire(length, max_encoded_bytes)?;
    let count = u64::try_from(deletions.keys().len())
        .map_err(|_| PlacementPersistenceError::ArithmeticOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| PlacementPersistenceError::AllocationFailed)?;
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&PLACEMENT_DELETION_CODEC_VERSION.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    for key in deletions.keys() {
        bytes.extend_from_slice(&key.placement_version.to_le_bytes());
        let source = key.source;
        bytes.extend_from_slice(&source.kernel_version.to_le_bytes());
        bytes.extend_from_slice(&source.seed.to_le_bytes());
        bytes.extend_from_slice(&source.version.to_le_bytes());
        bytes.extend_from_slice(&source.salt.to_le_bytes());
        bytes.extend_from_slice(&source.grid_spacing_mm.to_le_bytes());
        for coordinate in [source.grid.x, source.grid.y, source.grid.z] {
            bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn read<const N: usize>(bytes: &[u8], offset: usize) -> [u8; N] {
    // All callers first prove the exact fixed header/record width.
    bytes[offset..offset + N]
        .try_into()
        .expect("validated placement deletion record width")
}

/// Validates exact length, order, full keys and budgets before publishing a set.
///
/// No terrain is required: identity validity is independent of current surface
/// coverage or membership. Unknown placement/source versions, nonzero source Y,
/// zero spacing and unreconstructable raw coordinates fail, never become an
/// empty set. Count and decoded-memory admission precede key allocation.
pub fn decode_placement_deletions(
    bytes: &[u8],
    budget: PlacementDeletionCodecBudget,
) -> Result<HeightfieldPlacementDeletionSet, PlacementPersistenceError> {
    check_wire(bytes.len(), budget.max_encoded_bytes)?;
    if bytes.len() < PLACEMENT_DELETION_HEADER_BYTES {
        return Err(PlacementPersistenceError::InvalidLength {
            expected: PLACEMENT_DELETION_HEADER_BYTES,
            actual: bytes.len(),
        });
    }
    if bytes[..4] != MAGIC {
        return Err(PlacementPersistenceError::InvalidMagic);
    }
    let version = u32::from_le_bytes(read(bytes, 4));
    if version != PLACEMENT_DELETION_CODEC_VERSION {
        return Err(PlacementPersistenceError::UnsupportedVersion(version));
    }
    let count = usize::try_from(u64::from_le_bytes(read(bytes, 8)))
        .map_err(|_| PlacementPersistenceError::ArithmeticOverflow)?;
    let expected = encoded_len(count)?;
    if expected != bytes.len() {
        return Err(PlacementPersistenceError::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    check_keys(count, count, budget.deletions)?;
    let mut keys = Vec::new();
    keys.try_reserve_exact(count)
        .map_err(|_| PlacementPersistenceError::AllocationFailed)?;
    check_keys(count, keys.capacity(), budget.deletions)?;
    for record in
        bytes[PLACEMENT_DELETION_HEADER_BYTES..].chunks_exact(PLACEMENT_DELETION_KEY_BYTES)
    {
        let key = HeightfieldPlacementKey {
            placement_version: u32::from_le_bytes(read(record, 0)),
            source: ScatterCandidateKey {
                kernel_version: u32::from_le_bytes(read(record, 4)),
                seed: u64::from_le_bytes(read(record, 8)),
                version: u32::from_le_bytes(read(record, 16)),
                salt: u64::from_le_bytes(read(record, 20)),
                grid_spacing_mm: u32::from_le_bytes(read(record, 28)),
                grid: SectorCoord {
                    x: i64::from_le_bytes(read(record, 32)),
                    y: i64::from_le_bytes(read(record, 40)),
                    z: i64::from_le_bytes(read(record, 48)),
                },
            },
        };
        if keys.last().is_some_and(|previous| previous >= &key) {
            return Err(PlacementPersistenceError::NonCanonicalKeys);
        }
        keys.push(key);
    }
    HeightfieldPlacementDeletionSet::new(keys, budget.deletions)
        .map_err(PlacementPersistenceError::Deletions)
}

/// Caller-selected field of the scene-owned placement author node, not a point ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementDeletionBinding {
    /// Authoritative scene node namespace for the complete candidate identities.
    pub object: PersistentId,
    /// Schema-owned field for a whole immutable deletion snapshot.
    pub field: FieldId,
}

impl PlacementDeletionBinding {
    /// Reads authoritative override state; absent/Delete fields mean empty.
    pub fn read(
        self,
        state: &WorldState,
        budget: PlacementDeletionCodecBudget,
    ) -> Result<PlacementDeletionSnapshot, PlacementPersistenceError> {
        let deletions = match state.get(self.object, self.field) {
            None | Some(OverrideValue::Delete) => {
                HeightfieldPlacementDeletionSet::new(Vec::new(), budget.deletions)
                    .map_err(PlacementPersistenceError::Deletions)?
            }
            Some(OverrideValue::Bytes(bytes)) => decode_placement_deletions(bytes, budget)?,
            Some(_) => return Err(PlacementPersistenceError::UnexpectedValueType),
        };
        Ok(self.snapshot(deletions))
    }

    /// Explicitly binds a complete set, including currently absent candidates.
    pub fn snapshot(self, deletions: HeightfieldPlacementDeletionSet) -> PlacementDeletionSnapshot {
        PlacementDeletionSnapshot {
            binding: self,
            deletions,
            deletion_version: HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
        }
    }

    /// Builds a whole-field delta for the existing Timeline/persistence owner.
    ///
    /// Empty sets produce Delete without a wire allocation/budget requirement.
    /// This does not commit, journal, union concurrent sets or claim atomicity
    /// between Timeline and FileJournal. Undo must capture the actual prior field.
    pub fn delta(
        self,
        deletions: &HeightfieldPlacementDeletionSet,
        max_encoded_bytes: usize,
    ) -> Result<OverrideDelta, PlacementPersistenceError> {
        let value = if deletions.keys().is_empty() {
            OverrideValue::Delete
        } else {
            OverrideValue::Bytes(encode_placement_deletions(deletions, max_encoded_bytes)?)
        };
        OverrideDelta::single(self.field, value).map_err(PlacementPersistenceError::Delta)
    }
}

/// Complete additional cache dependency; field merges remain field-level LWW.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementDeletionSnapshot {
    binding: PlacementDeletionBinding,
    deletions: HeightfieldPlacementDeletionSet,
    deletion_version: u32,
}

impl PlacementDeletionSnapshot {
    /// Returns the authoritative author node and field namespace.
    pub const fn binding(&self) -> PlacementDeletionBinding {
        self.binding
    }

    /// Returns all immutable full keys, not just those removed this request.
    pub const fn deletions(&self) -> &HeightfieldPlacementDeletionSet {
        &self.deletions
    }

    /// Returns the in-memory filtering semantic revision, independent of codec.
    pub const fn deletion_version(&self) -> u32 {
        self.deletion_version
    }

    /// Stably filters both columns in place and retains the complete dependency.
    ///
    /// Input geometry/cache/counters are not regenerated or rewritten. A cache
    /// consumer must compare this snapshot in addition to the upstream cache.
    pub fn filter(
        &self,
        source: HeightfieldPlacementOutput,
    ) -> Result<PersistedHeightfieldPlacementOutput, PlacementPersistenceError> {
        let filtered = filter_heightfield_placement(source, &self.deletions)
            .map_err(PlacementPersistenceError::Deletions)?;
        Ok(PersistedHeightfieldPlacementOutput {
            source: filtered.source,
            snapshot: self.clone(),
            removed: filtered.removed,
        })
    }
}

/// Disposable projected data plus the full authoritative deletion dependency.
#[derive(Debug, Eq, PartialEq)]
pub struct PersistedHeightfieldPlacementOutput {
    /// Filtered aligned geometry/attributes with upstream cache/counters intact.
    pub source: HeightfieldPlacementOutput,
    /// Full immutable node/field/set/filter-version dependency.
    pub snapshot: PlacementDeletionSnapshot,
    /// Actual source rows removed, not total deletion-set size.
    pub removed: usize,
}
