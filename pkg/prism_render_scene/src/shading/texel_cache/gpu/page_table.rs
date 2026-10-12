//! CPU snapshot for the shade-page writer's indexed page-table ABI.
//!
//! This is deliberately not the material virtual-texture table. Material VT
//! entries are keyed by `TexturePageKey` and contain `(key words, slot)` for a
//! binary search. The shade writer instead indexes one entry per `page_index`
//! and checks `(atlas_x, atlas_y, generation, epoch)` against its job before
//! writing. A shade `slot` is an atlas allocation identifier; it is neither
//! that `page_index` nor a material-VT physical slot.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::super::key::ShadePageKey;
use super::super::store::ShadePageHandle;

/// One record used to build an immutable shade page-table snapshot.
///
/// The atlas origin is the address consumed by `gpu/atlas.wgsl`. `handle.slot`
/// remains CPU allocation metadata and is intentionally not serialized into
/// the four-word GPU entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadePageTableRecord {
    /// Stable object/material/chart/channel identity of this shade page.
    pub key: ShadePageKey,
    /// Shade-cache atlas allocation, generation, and epoch.
    pub handle: ShadePageHandle,
    /// X origin in atlas texels, matching the GPU writer job ABI.
    pub atlas_x: u32,
    /// Y origin in atlas texels, matching the GPU writer job ABI.
    pub atlas_y: u32,
}

/// Four `u32` words matching `PageEntry` in `gpu/atlas.wgsl`.
///
/// This layout is indexed by the snapshot's deterministic `page_index`; it is
/// not a key/value lookup record and has no material-VT slot field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct GpuShadePageEntry {
    /// Atlas texel X origin.
    pub atlas_x: u32,
    /// Atlas texel Y origin.
    pub atlas_y: u32,
    /// Shade content generation proof.
    pub generation: u32,
    /// Shade epoch proof.
    pub epoch: u32,
}

impl GpuShadePageEntry {
    /// Returns the exact storage-buffer word order used by `PageEntry`.
    #[must_use]
    pub const fn words(self) -> [u32; 4] {
        [self.atlas_x, self.atlas_y, self.generation, self.epoch]
    }
}

/// Failure to build a safe, deterministic shade page-table snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadePageTableError {
    /// The logical key appears more than once.
    DuplicateKey,
    /// Two live shade records claim the same physical shade atlas slot.
    DuplicateShadeSlot,
    /// Two records point at the same atlas origin.
    DuplicateAtlasOrigin,
    /// A record's key or handle does not belong to the snapshot epoch.
    EpochMismatch,
    /// The GPU writer ABI stores generation and epoch as `u32` values.
    GpuProofOutOfRange,
    /// The snapshot contains more entries than can be indexed by a `u32` page index.
    TooManyPages,
}

/// Immutable, key-sorted CPU snapshot for the shade writer page-table ABI.
///
/// Entry position is the GPU `page_index`. It is assigned by key order and is
/// distinct from [`ShadePageHandle::slot`], which identifies the allocation in
/// the shade atlas. The snapshot only includes fully published records supplied
/// by the caller; staging writes must not be passed here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadePageTableSnapshot {
    epoch: u64,
    records: Vec<ShadePageTableRecord>,
    gpu_entries: Vec<GpuShadePageEntry>,
    indices: BTreeMap<ShadePageKey, u32>,
}

impl ShadePageTableSnapshot {
    /// Validates and snapshots published shade records for one GPU epoch.
    ///
    /// Records are sorted by [`ShadePageKey`] before assigning page indices, so
    /// output is independent of input traversal order. A snapshot never accepts
    /// ordinary material-VT bindings: callers must provide shade keys and
    /// shade handles explicitly.
    pub fn try_from_records(
        epoch: u64,
        records: impl IntoIterator<Item = ShadePageTableRecord>,
    ) -> Result<Self, ShadePageTableError> {
        let mut records: Vec<_> = records.into_iter().collect();
        records.sort_unstable_by_key(|record| record.key);
        if records.len() > u32::MAX as usize {
            return Err(ShadePageTableError::TooManyPages);
        }

        let mut keys = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut origins = BTreeSet::new();
        let mut indices = BTreeMap::new();
        let mut gpu_entries = Vec::with_capacity(records.len());

        for (position, record) in records.iter().enumerate() {
            if !keys.insert(record.key) {
                return Err(ShadePageTableError::DuplicateKey);
            }
            if !slots.insert(record.handle.slot) {
                return Err(ShadePageTableError::DuplicateShadeSlot);
            }
            if !origins.insert((record.atlas_x, record.atlas_y)) {
                return Err(ShadePageTableError::DuplicateAtlasOrigin);
            }
            if record.key.material_epoch != epoch || record.handle.epoch != epoch {
                return Err(ShadePageTableError::EpochMismatch);
            }
            let generation = u32::try_from(record.handle.generation)
                .map_err(|_| ShadePageTableError::GpuProofOutOfRange)?;
            let gpu_epoch = u32::try_from(record.handle.epoch)
                .map_err(|_| ShadePageTableError::GpuProofOutOfRange)?;
            let page_index =
                u32::try_from(position).map_err(|_| ShadePageTableError::TooManyPages)?;
            indices.insert(record.key, page_index);
            gpu_entries.push(GpuShadePageEntry {
                atlas_x: record.atlas_x,
                atlas_y: record.atlas_y,
                generation,
                epoch: gpu_epoch,
            });
        }

        Ok(Self {
            epoch,
            records,
            gpu_entries,
            indices,
        })
    }

    /// Epoch represented by the active shade snapshot.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of indexed shade pages in this snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether this snapshot contains no published shade pages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Published shade records in deterministic page-index order.
    #[must_use]
    pub fn records(&self) -> &[ShadePageTableRecord] {
        &self.records
    }

    /// Four-word writer ABI entries in the same order as [`Self::records`].
    #[must_use]
    pub fn gpu_entries(&self) -> &[GpuShadePageEntry] {
        &self.gpu_entries
    }

    /// Flattened `(atlas_x, atlas_y, generation, epoch)` words for upload.
    #[must_use]
    pub fn gpu_words(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.gpu_entries.len() * 4);
        for entry in &self.gpu_entries {
            words.extend_from_slice(&entry.words());
        }
        words
    }

    /// Returns the GPU table index for an exact shade-page identity.
    #[must_use]
    pub fn page_index(&self, key: ShadePageKey) -> Option<u32> {
        self.indices.get(&key).copied()
    }

    /// Revalidates the CPU/GPU alignment invariants of this snapshot.
    #[must_use]
    pub fn validate(&self) -> Result<(), ShadePageTableError> {
        if self.records.len() != self.gpu_entries.len()
            || self.records.len() != self.indices.len()
            || self.records.len() > u32::MAX as usize
        {
            return Err(ShadePageTableError::TooManyPages);
        }
        for (position, (record, gpu)) in self.records.iter().zip(&self.gpu_entries).enumerate() {
            let page_index =
                u32::try_from(position).map_err(|_| ShadePageTableError::TooManyPages)?;
            if self.indices.get(&record.key) != Some(&page_index) {
                return Err(ShadePageTableError::DuplicateKey);
            }
            if record.key.material_epoch != self.epoch || record.handle.epoch != self.epoch {
                return Err(ShadePageTableError::EpochMismatch);
            }
            if u32::try_from(record.handle.generation).ok() != Some(gpu.generation)
                || u32::try_from(record.handle.epoch).ok() != Some(gpu.epoch)
                || record.atlas_x != gpu.atlas_x
                || record.atlas_y != gpu.atlas_y
            {
                return Err(ShadePageTableError::GpuProofOutOfRange);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::key::ShadeChannel;
    use super::*;

    fn record(object_id: u32, slot: u32, x: u32) -> ShadePageTableRecord {
        let epoch = 7;
        ShadePageTableRecord {
            key: ShadePageKey {
                object_id,
                material_id: 3,
                material_epoch: epoch,
                face: 0,
                mip: 0,
                channel: ShadeChannel::LitDiffuse,
                page_x: 0,
                page_y: 0,
            },
            handle: ShadePageHandle {
                slot,
                generation: u64::from(slot + 10),
                epoch,
            },
            atlas_x: x,
            atlas_y: 0,
        }
    }

    #[test]
    fn snapshot_assigns_stable_page_indices_and_emits_writer_abi_words() {
        let snapshot =
            ShadePageTableSnapshot::try_from_records(7, [record(9, 5, 16), record(2, 4, 0)])
                .unwrap();
        assert_eq!(snapshot.page_index(record(2, 4, 0).key), Some(0));
        assert_eq!(snapshot.page_index(record(9, 5, 16).key), Some(1));
        assert_eq!(snapshot.gpu_words(), [0, 0, 14, 7, 16, 0, 15, 7]);
        assert!(snapshot.validate().is_ok());
    }

    #[test]
    fn shade_identity_and_allocation_constraints_are_validated() {
        assert_eq!(
            ShadePageTableSnapshot::try_from_records(7, [record(1, 0, 0), record(2, 0, 8)]),
            Err(ShadePageTableError::DuplicateShadeSlot)
        );
        assert_eq!(
            ShadePageTableSnapshot::try_from_records(7, [record(1, 0, 0), record(1, 1, 8)]),
            Err(ShadePageTableError::DuplicateKey)
        );
        assert_eq!(
            ShadePageTableSnapshot::try_from_records(8, [record(1, 0, 0)]),
            Err(ShadePageTableError::EpochMismatch)
        );
    }

    #[test]
    fn writer_proof_width_is_checked_before_upload() {
        let mut page = record(1, 0, 0);
        page.handle.generation = u64::from(u32::MAX) + 1;
        assert_eq!(
            ShadePageTableSnapshot::try_from_records(7, [page]),
            Err(ShadePageTableError::GpuProofOutOfRange)
        );
    }
}
