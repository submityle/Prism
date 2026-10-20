//! Multi-cell residency, stable cross-cell references, and restart snapshots.
//!
//! This module owns only the world-side index.  Scene content remains owned by
//! the scene host; a cell contributes the stable object identities that the
//! host has made resident.  Dense runtime entities, GPU slots, and cell-local
//! ordering are deliberately not used as persistent identity.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use prism_world_persist::PersistentId;
use prism_world_stream::{CellManifest, SectorId};

use crate::SceneHost;

/// Current version of the deterministic cell-index snapshot format.
pub const WORLD_CELL_SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_MAGIC: [u8; 4] = *b"PWCS";
const SNAPSHOT_HEADER_BYTES: usize = 24;
const CELL_RECORD_BYTES: usize = 29;
const OBJECT_ID_BYTES: usize = 16;

/// Limits applied before allocating a decoded world-cell index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldCellStoreLimits {
    /// Maximum simultaneously resident cells.
    pub max_cells: usize,
    /// Maximum stable object identities across all resident cells.
    pub max_objects: usize,
    /// Maximum complete snapshot byte length.
    pub max_snapshot_bytes: usize,
}

impl WorldCellStoreLimits {
    /// Creates explicit limits.  Zero is valid and rejects all insertions.
    #[must_use]
    pub const fn new(max_cells: usize, max_objects: usize, max_snapshot_bytes: usize) -> Self {
        Self {
            max_cells,
            max_objects,
            max_snapshot_bytes,
        }
    }
}

/// A persistent identity that can resolve independently of runtime entities.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StableWorldRef {
    /// Cross-session object identity.
    pub id: PersistentId,
}

impl StableWorldRef {
    /// Creates a stable world reference.
    #[must_use]
    pub const fn new(id: PersistentId) -> Self {
        Self { id }
    }
}

/// The current resident cell of one stable object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedWorldRef {
    /// Resolved stable identity.
    pub reference: StableWorldRef,
    /// Cell currently owning the object.
    pub cell: SectorId,
}

/// Decodes only the stable identities owned by one loaded cell.
pub trait StableRefDecoder {
    /// Decoder-specific failure type.
    type Error;

    /// Returns the complete stable identity set for the cell revision.
    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<Vec<StableWorldRef>, Self::Error>;
}

impl<F, E> StableRefDecoder for F
where
    F: FnMut(&CellManifest, &[u8]) -> Result<Vec<StableWorldRef>, E>,
{
    type Error = E;

    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<Vec<StableWorldRef>, Self::Error> {
        self(manifest, payload)
    }
}

/// A failure from stable-reference decoding, store admission, or the scene host.
#[derive(Debug)]
pub enum StableRefSceneHostError<D, H> {
    /// Stable identity extraction failed before touching either owner.
    Decoder(D),
    /// The decoded identity set conflicts with the resident world index.
    Store(WorldCellStoreError),
    /// The authoritative scene host rejected the lifecycle operation.
    Host(H),
}

impl<D: fmt::Display, H: fmt::Display> fmt::Display for StableRefSceneHostError<D, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(error) => write!(f, "stable reference decoder: {error}"),
            Self::Store(error) => write!(f, "stable reference store: {error}"),
            Self::Host(error) => write!(f, "scene host: {error}"),
        }
    }
}

impl<D, H> core::error::Error for StableRefSceneHostError<D, H>
where
    D: core::error::Error + 'static,
    H: core::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Decoder(error) => Some(error),
            Self::Store(error) => Some(error),
            Self::Host(error) => Some(error),
        }
    }
}

/// Decorates a scene host with atomic stable-reference index publication.
///
/// The decoder is deliberately explicit: the world runtime does not assume a
/// cell payload format. On load/reload, decoding and store validation happen
/// before the authoritative host is called; the index is published only after
/// the host succeeds. On eviction, the host is called first and the index is
/// removed only after success. This preserves the previous index on ordinary
/// host or decoder failures.
pub struct StableRefSceneHost<'a, H: SceneHost, D: StableRefDecoder> {
    host: &'a mut H,
    store: &'a mut WorldCellStore,
    decoder: D,
}

impl<'a, H: SceneHost, D: StableRefDecoder> StableRefSceneHost<'a, H, D> {
    /// Wraps a host, stable-reference store, and payload decoder.
    #[must_use]
    pub fn new(host: &'a mut H, store: &'a mut WorldCellStore, decoder: D) -> Self {
        Self {
            host,
            store,
            decoder,
        }
    }

    /// Returns the decorated store for read-only resolution and persistence.
    #[must_use]
    pub fn store(&self) -> &WorldCellStore {
        self.store
    }
}

impl<H: SceneHost, D: StableRefDecoder> SceneHost for StableRefSceneHost<'_, H, D> {
    type Error = StableRefSceneHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let objects = self
            .decoder
            .decode(manifest, payload)
            .map_err(StableRefSceneHostError::Decoder)?;
        self.store
            .validate_replace(manifest.sector, &objects)
            .map_err(StableRefSceneHostError::Store)?;
        self.host
            .load_cell(manifest, payload)
            .map_err(StableRefSceneHostError::Host)?;
        self.store
            .replace_cell(manifest.sector, &objects)
            .map_err(StableRefSceneHostError::Store)?;
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        if !self.store.contains_cell(sector) {
            return Err(StableRefSceneHostError::Store(
                WorldCellStoreError::CellNotResident { cell: sector },
            ));
        }
        self.host
            .evict_cell(sector)
            .map_err(StableRefSceneHostError::Host)?;
        self.store
            .remove_cell(sector)
            .map_err(StableRefSceneHostError::Store)?;
        Ok(())
    }
}

/// A successful cell replacement summary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellCommit {
    /// Cell that was loaded or replaced.
    pub cell: SectorId,
    /// Number of identities introduced by the new revision.
    pub added_objects: usize,
    /// Number of identities removed from the previous revision.
    pub removed_objects: usize,
}

/// Errors returned without publishing a partial cell-index mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldCellStoreError {
    /// The cell budget is exhausted.
    CellCapacityExceeded {
        /// Configured cell limit.
        capacity: usize,
    },
    /// The object budget is exhausted.
    ObjectCapacityExceeded {
        /// Resulting object count.
        requested: usize,
        /// Configured object limit.
        available: usize,
    },
    /// A cell was requested twice through the insert-only API.
    CellAlreadyResident {
        /// Conflicting cell.
        cell: SectorId,
    },
    /// A stable identity is repeated inside one cell revision.
    DuplicateObject {
        /// Repeated stable identity.
        object: StableWorldRef,
    },
    /// A stable identity is already owned by another resident cell.
    ObjectAlreadyResident {
        /// Conflicting stable identity.
        object: StableWorldRef,
        /// Existing owner cell.
        cell: SectorId,
    },
    /// The requested object or source cell is not resident.
    NotResident {
        /// Missing stable identity.
        object: StableWorldRef,
    },
    /// The relocation destination is not resident.
    DestinationNotResident {
        /// Missing destination cell.
        cell: SectorId,
    },
    /// The requested source cell does not exist.
    CellNotResident {
        /// Missing cell.
        cell: SectorId,
    },
    /// A fallible vector reservation failed.
    AllocationFailed,
    /// Snapshot output or input exceeded its configured bound.
    SnapshotByteBudgetExceeded {
        /// Complete byte requirement.
        required: usize,
        /// Configured byte limit.
        limit: usize,
    },
    /// The snapshot header is too short or the record stream is inconsistent.
    InvalidSnapshotLength {
        /// Expected complete length or next required offset.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// The snapshot magic does not identify this format.
    InvalidSnapshotMagic,
    /// The snapshot revision is not supported.
    UnsupportedSnapshotVersion(u32),
    /// Snapshot cells or object ids are not in canonical order.
    NonCanonicalSnapshot,
    /// The declared object count differs from the record stream.
    ObjectCountMismatch {
        /// Count declared in the snapshot header.
        expected: usize,
        /// Count encountered while decoding records.
        actual: usize,
    },
    /// Snapshot count arithmetic overflowed the host ABI.
    SnapshotArithmeticOverflow,
}

impl fmt::Display for WorldCellStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CellCapacityExceeded { capacity } => {
                write!(f, "resident cell capacity {capacity} exceeded")
            }
            Self::ObjectCapacityExceeded {
                requested,
                available,
            } => {
                write!(
                    f,
                    "requested {requested} objects, only {available} available"
                )
            }
            Self::CellAlreadyResident { cell } => write!(f, "cell {cell:?} is already resident"),
            Self::DuplicateObject { object } => write!(f, "duplicate object {object:?}"),
            Self::ObjectAlreadyResident { object, cell } => {
                write!(f, "object {object:?} is already resident in {cell:?}")
            }
            Self::NotResident { object } => write!(f, "object {object:?} is not resident"),
            Self::DestinationNotResident { cell } => {
                write!(f, "relocation destination {cell:?} is not resident")
            }
            Self::CellNotResident { cell } => write!(f, "cell {cell:?} is not resident"),
            Self::AllocationFailed => f.write_str("world cell index reservation failed"),
            Self::SnapshotByteBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "world cell snapshot needs {required} bytes, limit {limit}"
                )
            }
            Self::InvalidSnapshotLength { expected, actual } => {
                write!(
                    f,
                    "world cell snapshot length {actual}, expected {expected}"
                )
            }
            Self::InvalidSnapshotMagic => f.write_str("invalid world cell snapshot magic"),
            Self::UnsupportedSnapshotVersion(version) => {
                write!(f, "unsupported world cell snapshot version {version}")
            }
            Self::NonCanonicalSnapshot => f.write_str("world cell snapshot is not canonical"),
            Self::ObjectCountMismatch { expected, actual } => {
                write!(
                    f,
                    "world cell snapshot declares {expected} objects, decoded {actual}"
                )
            }
            Self::SnapshotArithmeticOverflow => {
                f.write_str("world cell snapshot arithmetic overflow")
            }
        }
    }
}

impl core::error::Error for WorldCellStoreError {}

/// Bounded resident-cell index with O(log cell) and O(log object) lookups.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorldCellStore {
    limits: WorldCellStoreLimits,
    cells: BTreeMap<SectorId, Vec<StableWorldRef>>,
    owners: BTreeMap<StableWorldRef, SectorId>,
}

impl WorldCellStore {
    /// Creates an empty store without allocating proportional to the limits.
    #[must_use]
    pub const fn new(limits: WorldCellStoreLimits) -> Self {
        Self {
            limits,
            cells: BTreeMap::new(),
            owners: BTreeMap::new(),
        }
    }

    /// Returns the configured limits.
    #[must_use]
    pub const fn limits(&self) -> WorldCellStoreLimits {
        self.limits
    }

    /// Returns the number of resident cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Returns the number of resident objects.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.owners.len()
    }

    /// Returns whether a cell is resident.
    #[must_use]
    pub fn contains_cell(&self, cell: SectorId) -> bool {
        self.cells.contains_key(&cell)
    }

    /// Returns stable objects in canonical stable-id order for one cell.
    #[must_use]
    pub fn cell_objects(&self, cell: SectorId) -> Option<&[StableWorldRef]> {
        self.cells.get(&cell).map(Vec::as_slice)
    }

    /// Resolves a stable reference without consulting runtime entity handles.
    #[must_use]
    pub fn resolve(&self, reference: StableWorldRef) -> Option<ResolvedWorldRef> {
        self.owners
            .get(&reference)
            .copied()
            .map(|cell| ResolvedWorldRef { reference, cell })
    }

    /// Validates an insert-only cell operation before any mutation.
    pub fn validate_insert(
        &self,
        cell: SectorId,
        objects: &[StableWorldRef],
    ) -> Result<(), WorldCellStoreError> {
        if self.contains_cell(cell) {
            return Err(WorldCellStoreError::CellAlreadyResident { cell });
        }
        self.validate_new_cell(cell, objects, false)
    }

    /// Validates a replacement before an external scene host is called.
    pub fn validate_replace(
        &self,
        cell: SectorId,
        objects: &[StableWorldRef],
    ) -> Result<(), WorldCellStoreError> {
        self.validate_new_cell(cell, objects, self.contains_cell(cell))
    }

    /// Inserts a cell and its stable identities atomically from the API's view.
    pub fn insert_cell(
        &mut self,
        cell: SectorId,
        objects: &[StableWorldRef],
    ) -> Result<(), WorldCellStoreError> {
        self.validate_insert(cell, objects)?;
        self.publish_cell(cell, objects);
        Ok(())
    }

    /// Validates and replaces one cell revision, preserving unaffected cells.
    pub fn replace_cell(
        &mut self,
        cell: SectorId,
        objects: &[StableWorldRef],
    ) -> Result<CellCommit, WorldCellStoreError> {
        self.validate_replace(cell, objects)?;
        let old = self.cells.get(&cell).map(Vec::as_slice).unwrap_or(&[]);
        let removed_objects = old
            .iter()
            .filter(|object| !objects.contains(object))
            .count();
        let added_objects = objects
            .iter()
            .filter(|object| !old.contains(object))
            .count();
        if let Some(old_objects) = self.cells.remove(&cell) {
            for object in old_objects {
                self.owners.remove(&object);
            }
        }
        self.publish_cell(cell, objects);
        Ok(CellCommit {
            cell,
            added_objects,
            removed_objects,
        })
    }

    /// Removes a resident cell and returns its stable identities.
    pub fn remove_cell(
        &mut self,
        cell: SectorId,
    ) -> Result<Vec<StableWorldRef>, WorldCellStoreError> {
        let objects = self
            .cells
            .remove(&cell)
            .ok_or(WorldCellStoreError::CellNotResident { cell })?;
        for object in &objects {
            self.owners.remove(object);
        }
        Ok(objects)
    }

    /// Relocates one stable object between two already resident cells.
    pub fn relocate(
        &mut self,
        reference: StableWorldRef,
        destination: SectorId,
    ) -> Result<(), WorldCellStoreError> {
        if !self.contains_cell(destination) {
            return Err(WorldCellStoreError::DestinationNotResident { cell: destination });
        }
        let source = self
            .owners
            .get(&reference)
            .copied()
            .ok_or(WorldCellStoreError::NotResident { object: reference })?;
        if source == destination {
            return Ok(());
        }
        {
            let source_objects = self
                .cells
                .get_mut(&source)
                .ok_or(WorldCellStoreError::CellNotResident { cell: source })?;
            let index = source_objects
                .iter()
                .position(|candidate| *candidate == reference)
                .ok_or(WorldCellStoreError::NotResident { object: reference })?;
            source_objects.remove(index);
        }
        let destination_objects = self
            .cells
            .get_mut(&destination)
            .ok_or(WorldCellStoreError::CellNotResident { cell: destination })?;
        destination_objects.push(reference);
        destination_objects.sort_unstable();
        self.owners.insert(reference, destination);
        Ok(())
    }

    /// Encodes a deterministic restart snapshot with an explicit byte bound.
    pub fn encode_snapshot(&self, max_bytes: usize) -> Result<Vec<u8>, WorldCellStoreError> {
        let length = self.snapshot_len()?;
        if length > max_bytes {
            return Err(WorldCellStoreError::SnapshotByteBudgetExceeded {
                required: length,
                limit: max_bytes,
            });
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| WorldCellStoreError::AllocationFailed)?;
        bytes.extend_from_slice(&SNAPSHOT_MAGIC);
        bytes.extend_from_slice(&WORLD_CELL_SNAPSHOT_VERSION.to_le_bytes());
        let cell_count = u64::try_from(self.cells.len())
            .map_err(|_| WorldCellStoreError::SnapshotArithmeticOverflow)?;
        let object_count = u64::try_from(self.owners.len())
            .map_err(|_| WorldCellStoreError::SnapshotArithmeticOverflow)?;
        bytes.extend_from_slice(&cell_count.to_le_bytes());
        bytes.extend_from_slice(&object_count.to_le_bytes());
        for (cell, objects) in &self.cells {
            bytes.push(cell.level);
            bytes.extend_from_slice(&cell.x.to_le_bytes());
            bytes.extend_from_slice(&cell.y.to_le_bytes());
            bytes.extend_from_slice(&cell.z.to_le_bytes());
            let object_count = u32::try_from(objects.len())
                .map_err(|_| WorldCellStoreError::SnapshotArithmeticOverflow)?;
            bytes.extend_from_slice(&object_count.to_le_bytes());
            for object in objects {
                bytes.extend_from_slice(&object.id.0.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    /// Decodes and validates a restart snapshot before publishing any state.
    pub fn decode_snapshot(
        bytes: &[u8],
        limits: WorldCellStoreLimits,
    ) -> Result<Self, WorldCellStoreError> {
        if bytes.len() > limits.max_snapshot_bytes {
            return Err(WorldCellStoreError::SnapshotByteBudgetExceeded {
                required: bytes.len(),
                limit: limits.max_snapshot_bytes,
            });
        }
        if bytes.len() < SNAPSHOT_HEADER_BYTES {
            return Err(WorldCellStoreError::InvalidSnapshotLength {
                expected: SNAPSHOT_HEADER_BYTES,
                actual: bytes.len(),
            });
        }
        if bytes[..4] != SNAPSHOT_MAGIC {
            return Err(WorldCellStoreError::InvalidSnapshotMagic);
        }
        let version = read_u32(bytes, 4);
        if version != WORLD_CELL_SNAPSHOT_VERSION {
            return Err(WorldCellStoreError::UnsupportedSnapshotVersion(version));
        }
        let cell_count = usize::try_from(read_u64(bytes, 8))
            .map_err(|_| WorldCellStoreError::SnapshotArithmeticOverflow)?;
        let object_count = usize::try_from(read_u64(bytes, 16))
            .map_err(|_| WorldCellStoreError::SnapshotArithmeticOverflow)?;
        if cell_count > limits.max_cells {
            return Err(WorldCellStoreError::CellCapacityExceeded {
                capacity: limits.max_cells,
            });
        }
        if object_count > limits.max_objects {
            return Err(WorldCellStoreError::ObjectCapacityExceeded {
                requested: object_count,
                available: limits.max_objects,
            });
        }
        let mut cursor = SNAPSHOT_HEADER_BYTES;
        let mut store = Self::new(limits);
        let mut previous = None;
        let mut decoded_objects = 0usize;
        for _ in 0..cell_count {
            let cell = SectorId::new(
                read_u8(bytes, &mut cursor)?,
                read_i64(bytes, &mut cursor)?,
                read_i64(bytes, &mut cursor)?,
                read_i64(bytes, &mut cursor)?,
            );
            if previous.is_some_and(|old| old >= cell) {
                return Err(WorldCellStoreError::NonCanonicalSnapshot);
            }
            previous = Some(cell);
            let count = usize::try_from(read_u32_at(bytes, &mut cursor)?)
                .map_err(|_| WorldCellStoreError::SnapshotArithmeticOverflow)?;
            let remaining_declared = object_count.saturating_sub(decoded_objects);
            if count > remaining_declared {
                return Err(WorldCellStoreError::ObjectCountMismatch {
                    expected: object_count,
                    actual: decoded_objects.saturating_add(count),
                });
            }
            let required_bytes = count
                .checked_mul(OBJECT_ID_BYTES)
                .ok_or(WorldCellStoreError::SnapshotArithmeticOverflow)?;
            if required_bytes > bytes.len().saturating_sub(cursor) {
                return Err(WorldCellStoreError::InvalidSnapshotLength {
                    expected: cursor.saturating_add(required_bytes),
                    actual: bytes.len(),
                });
            }
            let mut objects = Vec::new();
            objects
                .try_reserve_exact(count)
                .map_err(|_| WorldCellStoreError::AllocationFailed)?;
            for _ in 0..count {
                objects.push(StableWorldRef::new(PersistentId(read_u128(
                    bytes,
                    &mut cursor,
                )?)));
            }
            if objects.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(WorldCellStoreError::NonCanonicalSnapshot);
            }
            decoded_objects = decoded_objects
                .checked_add(count)
                .ok_or(WorldCellStoreError::SnapshotArithmeticOverflow)?;
            store.insert_cell(cell, &objects)?;
        }
        if decoded_objects != object_count {
            return Err(WorldCellStoreError::ObjectCountMismatch {
                expected: object_count,
                actual: decoded_objects,
            });
        }
        if cursor != bytes.len() {
            return Err(WorldCellStoreError::InvalidSnapshotLength {
                expected: cursor,
                actual: bytes.len(),
            });
        }
        Ok(store)
    }

    #[cfg(feature = "std")]
    /// Atomically writes a restart snapshot beside the target and renames it.
    pub fn save_snapshot(
        &self,
        path: impl AsRef<std::path::Path>,
        max_bytes: usize,
    ) -> Result<(), WorldCellStoreError> {
        use std::fs::{self, OpenOptions};
        use std::io::Write;
        let path = path.as_ref();
        let temporary = path.with_extension("pwcs.tmp");
        let bytes = self.encode_snapshot(max_bytes)?;
        let result = (|| {
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&temporary)
                .map_err(|_| WorldCellStoreError::AllocationFailed)?;
            file.write_all(&bytes)
                .map_err(|_| WorldCellStoreError::AllocationFailed)?;
            file.sync_all()
                .map_err(|_| WorldCellStoreError::AllocationFailed)?;
            fs::rename(&temporary, path).map_err(|_| WorldCellStoreError::AllocationFailed)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    #[cfg(feature = "std")]
    /// Opens and validates a restart snapshot from disk.
    pub fn load_snapshot(
        path: impl AsRef<std::path::Path>,
        limits: WorldCellStoreLimits,
    ) -> Result<Self, WorldCellStoreError> {
        let bytes = std::fs::read(path).map_err(|_| WorldCellStoreError::AllocationFailed)?;
        Self::decode_snapshot(&bytes, limits)
    }

    fn validate_new_cell(
        &self,
        cell: SectorId,
        objects: &[StableWorldRef],
        replacing: bool,
    ) -> Result<(), WorldCellStoreError> {
        if (!replacing || !self.contains_cell(cell)) && self.cells.len() >= self.limits.max_cells {
            return Err(WorldCellStoreError::CellCapacityExceeded {
                capacity: self.limits.max_cells,
            });
        }
        let old = self.cells.get(&cell).map(Vec::as_slice).unwrap_or(&[]);
        if objects.len() > self.limits.max_objects {
            return Err(WorldCellStoreError::ObjectCapacityExceeded {
                requested: objects.len(),
                available: self.limits.max_objects,
            });
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(objects.len())
            .map_err(|_| WorldCellStoreError::AllocationFailed)?;
        for &object in objects {
            if seen.contains(&object) {
                return Err(WorldCellStoreError::DuplicateObject { object });
            }
            seen.push(object);
            if let Some(owner) = self.owners.get(&object).copied()
                && (owner != cell || !old.contains(&object))
            {
                return Err(WorldCellStoreError::ObjectAlreadyResident {
                    object,
                    cell: owner,
                });
            }
        }
        let new_total = self
            .owners
            .len()
            .saturating_sub(old.len())
            .saturating_add(objects.len());
        if new_total > self.limits.max_objects {
            return Err(WorldCellStoreError::ObjectCapacityExceeded {
                requested: new_total,
                available: self.limits.max_objects,
            });
        }
        Ok(())
    }

    fn publish_cell(&mut self, cell: SectorId, objects: &[StableWorldRef]) {
        let mut owned = objects.to_vec();
        owned.sort_unstable();
        self.cells.insert(cell, owned.clone());
        for object in owned {
            self.owners.insert(object, cell);
        }
    }

    fn snapshot_len(&self) -> Result<usize, WorldCellStoreError> {
        let mut length = SNAPSHOT_HEADER_BYTES;
        for objects in self.cells.values() {
            length = length
                .checked_add(CELL_RECORD_BYTES)
                .and_then(|value| value.checked_add(objects.len().checked_mul(OBJECT_ID_BYTES)?))
                .ok_or(WorldCellStoreError::SnapshotArithmeticOverflow)?;
        }
        Ok(length)
    }
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, WorldCellStoreError> {
    let value = *bytes
        .get(*cursor)
        .ok_or(WorldCellStoreError::InvalidSnapshotLength {
            expected: cursor.saturating_add(1),
            actual: bytes.len(),
        })?;
    *cursor += 1;
    Ok(value)
}

fn read_fixed<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], WorldCellStoreError> {
    let end = cursor
        .checked_add(N)
        .ok_or(WorldCellStoreError::SnapshotArithmeticOverflow)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(WorldCellStoreError::InvalidSnapshotLength {
            expected: end,
            actual: bytes.len(),
        })?;
    *cursor = end;
    Ok(slice.try_into().expect("validated fixed snapshot width"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated snapshot header"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated snapshot header"),
    )
}

fn read_u32_at(bytes: &[u8], cursor: &mut usize) -> Result<u32, WorldCellStoreError> {
    Ok(u32::from_le_bytes(read_fixed(bytes, cursor)?))
}

fn read_i64(bytes: &[u8], cursor: &mut usize) -> Result<i64, WorldCellStoreError> {
    Ok(i64::from_le_bytes(read_fixed(bytes, cursor)?))
}

fn read_u128(bytes: &[u8], cursor: &mut usize) -> Result<u128, WorldCellStoreError> {
    Ok(u128::from_le_bytes(read_fixed(bytes, cursor)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_world_stream::{AabbI64, GlobalPos};

    fn limits() -> WorldCellStoreLimits {
        WorldCellStoreLimits::new(8, 32, 4096)
    }

    fn reference(id: u128) -> StableWorldRef {
        StableWorldRef::new(PersistentId(id))
    }

    fn cell(x: i64) -> SectorId {
        SectorId::new(3, x, 0, 0)
    }

    fn manifest(x: i64) -> CellManifest {
        CellManifest {
            sector: cell(x),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1)).unwrap(),
            scene_key: x as u128,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 0,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        }
    }

    #[derive(Debug)]
    struct TestError;

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("test error")
        }
    }

    impl core::error::Error for TestError {}

    #[derive(Default)]
    struct Host {
        loaded: Vec<SectorId>,
        evicted: Vec<SectorId>,
        fail_load: bool,
    }

    impl SceneHost for Host {
        type Error = TestError;

        fn load_cell(&mut self, manifest: &CellManifest, _: &[u8]) -> Result<(), Self::Error> {
            if self.fail_load {
                return Err(TestError);
            }
            self.loaded.push(manifest.sector);
            Ok(())
        }

        fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
            self.evicted.push(sector);
            Ok(())
        }
    }

    #[test]
    fn multiple_cells_resolve_and_relocate_without_dense_identity() {
        let mut store = WorldCellStore::new(limits());
        let a = reference(10);
        let b = reference(20);
        let c = reference(30);
        store.insert_cell(cell(-1), &[a, b]).unwrap();
        store.insert_cell(cell(1), &[c]).unwrap();
        assert_eq!(store.resolve(a).unwrap().cell, cell(-1));
        store.relocate(b, cell(1)).unwrap();
        assert_eq!(store.resolve(b).unwrap().cell, cell(1));
        assert_eq!(store.cell_objects(cell(-1)).unwrap(), &[a]);
        assert_eq!(store.cell_objects(cell(1)).unwrap(), &[b, c]);
        store.remove_cell(cell(-1)).unwrap();
        assert!(store.resolve(a).is_none());
        assert_eq!(store.resolve(b).unwrap().cell, cell(1));
    }

    #[test]
    fn replacement_rejects_cross_cell_duplicate_before_mutation() {
        let mut store = WorldCellStore::new(limits());
        let object = reference(1);
        store.insert_cell(cell(0), &[object]).unwrap();
        let before = store.clone();
        let error = store.replace_cell(cell(1), &[object]).unwrap_err();
        assert!(matches!(
            error,
            WorldCellStoreError::ObjectAlreadyResident { .. }
        ));
        assert_eq!(store, before);
    }

    #[test]
    fn restart_snapshot_is_canonical_and_restores_cross_cell_index() {
        let mut store = WorldCellStore::new(limits());
        store
            .insert_cell(cell(2), &[reference(9), reference(7)])
            .unwrap();
        store.insert_cell(cell(-2), &[reference(5)]).unwrap();
        let bytes = store.encode_snapshot(4096).unwrap();
        let restored = WorldCellStore::decode_snapshot(&bytes, limits()).unwrap();
        assert_eq!(restored.cell_count(), 2);
        assert_eq!(restored.object_count(), 3);
        assert_eq!(restored.resolve(reference(7)).unwrap().cell, cell(2));
        assert_eq!(bytes, restored.encode_snapshot(4096).unwrap());
    }

    #[test]
    fn malformed_snapshot_never_publishes_partial_state() {
        let mut bytes = WorldCellStore::new(limits()).encode_snapshot(4096).unwrap();
        bytes[0] = b'X';
        assert!(matches!(
            WorldCellStore::decode_snapshot(&bytes, limits()),
            Err(WorldCellStoreError::InvalidSnapshotMagic)
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn restart_file_roundtrip_keeps_cross_cell_resolution() {
        let path = std::env::temp_dir().join(format!(
            "prism-world-cell-store-{}.pwcs",
            std::process::id()
        ));
        let mut store = WorldCellStore::new(limits());
        store
            .insert_cell(cell(-4), &[reference(42), reference(7)])
            .unwrap();
        store.insert_cell(cell(8), &[reference(99)]).unwrap();
        store.save_snapshot(&path, 4096).unwrap();
        let restored = WorldCellStore::load_snapshot(&path, limits()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(restored.resolve(reference(7)).unwrap().cell, cell(-4));
        assert_eq!(restored.resolve(reference(99)).unwrap().cell, cell(8));
        assert_eq!(
            restored.encode_snapshot(4096).unwrap(),
            store.encode_snapshot(4096).unwrap()
        );
    }

    #[test]
    fn snapshot_header_counts_are_checked_before_record_allocation() {
        let mut bytes = WorldCellStore::new(limits()).encode_snapshot(4096).unwrap();
        bytes[16..24].copy_from_slice(&1_u64.to_le_bytes());
        assert!(matches!(
            WorldCellStore::decode_snapshot(&bytes, limits()),
            Err(WorldCellStoreError::ObjectCountMismatch {
                expected: 1,
                actual: 0
            })
        ));

        bytes[8..16].copy_from_slice(&9_u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&0_u64.to_le_bytes());
        assert!(matches!(
            WorldCellStore::decode_snapshot(&bytes, limits()),
            Err(WorldCellStoreError::CellCapacityExceeded { capacity: 8 })
        ));
    }

    #[test]
    fn scene_host_adapter_publishes_multiple_cells_and_resolves_cross_cell_refs() {
        let mut host = Host::default();
        let mut store = WorldCellStore::new(limits());
        let a = reference(10);
        let b = reference(20);
        let c = reference(30);
        let mut decoder = move |manifest: &CellManifest, _: &[u8]| {
            Ok::<_, TestError>(match manifest.sector.x {
                -1 => vec![b, a],
                _ => vec![c],
            })
        };
        {
            let mut adapter = StableRefSceneHost::new(&mut host, &mut store, &mut decoder);
            SceneHost::load_cell(&mut adapter, &manifest(-1), b"a").unwrap();
            SceneHost::load_cell(&mut adapter, &manifest(1), b"b").unwrap();
        }
        assert_eq!(store.cell_objects(cell(-1)).unwrap(), &[a, b]);
        assert_eq!(store.resolve(c).unwrap().cell, cell(1));
        store.relocate(b, cell(1)).unwrap();
        assert_eq!(store.resolve(b).unwrap().cell, cell(1));
    }

    #[test]
    fn scene_host_adapter_keeps_old_index_on_reload_failure_and_removes_after_eviction() {
        let mut host = Host::default();
        let mut store = WorldCellStore::new(limits());
        let old = reference(1);
        let new = reference(2);
        let mut decoder = move |_: &CellManifest, payload: &[u8]| {
            Ok::<_, TestError>(if payload == b"new" {
                vec![new]
            } else {
                vec![old]
            })
        };
        {
            let mut adapter = StableRefSceneHost::new(&mut host, &mut store, &mut decoder);
            SceneHost::load_cell(&mut adapter, &manifest(0), b"old").unwrap();
        }
        host.fail_load = true;
        {
            let mut adapter = StableRefSceneHost::new(&mut host, &mut store, &mut decoder);
            assert!(SceneHost::load_cell(&mut adapter, &manifest(0), b"new").is_err());
        }
        assert!(store.resolve(old).is_some());
        assert!(store.resolve(new).is_none());
        {
            let mut adapter = StableRefSceneHost::new(&mut host, &mut store, &mut decoder);
            SceneHost::evict_cell(&mut adapter, cell(0)).unwrap();
        }
        assert!(store.resolve(old).is_none());
    }
}
