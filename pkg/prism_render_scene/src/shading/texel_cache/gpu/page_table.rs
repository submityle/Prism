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

use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_driver::{BufferId, RenderQueue};
use prism_render_driver_wgpu::WgpuQueue;

use crate::GpuCompletionTracker;

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

// -----------------------------------------------------------------------------
// Material virtual-texture page table
// -----------------------------------------------------------------------------

/// Number of `u32` words in one material-VT page-table entry.
pub const MATERIAL_VT_ENTRY_WORDS: usize = 4;

/// Value returned by the material-VT shader for a missing page.
pub const MATERIAL_VT_MISS: u32 = u32::MAX;

/// Texture id reserved by the material-VT shader for a non-virtual texture.
pub const MATERIAL_VT_NON_VIRTUAL_TEXTURE: u32 = u32::MAX;

/// A material-VT page supplied by a real producer before publication.
///
/// `ready` is deliberately not encoded into the shader entry: the WESL
/// contract has only `(texture, mip/layer, page-x/page-y, slot)`. Instead, the
/// adapter refuses to construct a snapshot unless every source page is ready,
/// and the snapshot is only exposed to upload after that validation succeeds.
/// This keeps a partially uploaded page from becoming a binary-search hit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTexturePageRecord {
    /// Virtual page identity used by `material_virtual_texture.wesl`.
    pub key: TexturePageKey,
    /// Physical atlas slot returned by the material-VT producer.
    pub slot: u32,
    /// Producer content generation for this page.
    pub generation: u64,
    /// Material-VT publication epoch for this page.
    pub epoch: u64,
    /// Whether the producer has completed the page upload and made it ready.
    pub ready: bool,
}

/// The exact four-word entry consumed by `material_virtual_texture.wesl`.
///
/// This type is intentionally distinct from [`GpuShadePageEntry`]. Shade
/// entries are indexed by page index and carry atlas coordinates plus proof
/// words; material-VT entries are key-sorted and carry a physical slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct GpuMaterialVirtualTextureEntry {
    /// Source texture id (`w0`).
    pub texture: u32,
    /// Packed mip/layer word (`w1`).
    pub mip_layer: u32,
    /// Packed page-x/page-y word (`w2`).
    pub page_xy: u32,
    /// Physical atlas slot (`w3`).
    pub slot: u32,
}

impl GpuMaterialVirtualTextureEntry {
    /// Returns the exact word order required by the material-VT WESL helper.
    #[must_use]
    pub const fn words(self) -> [u32; MATERIAL_VT_ENTRY_WORDS] {
        [self.texture, self.mip_layer, self.page_xy, self.slot]
    }
}

/// Errors raised while building or publishing a material-VT table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVirtualTexturePageTableError {
    /// A source page was not complete and therefore cannot be published.
    PageNotReady,
    /// A page belongs to another material-VT epoch.
    EpochMismatch,
    /// A page was produced by another content generation.
    GenerationMismatch,
    /// Generation zero denotes an uninitialised producer state.
    ZeroGeneration,
    /// The shader's miss sentinel cannot be a resident physical slot.
    MissSentinelReserved,
    /// The shader uses this texture id for non-virtual material textures.
    NonVirtualTextureReserved,
    /// Two records claim the same virtual key.
    DuplicateKey,
    /// Two records claim the same physical slot in one published snapshot.
    DuplicatePhysicalSlot,
    /// More entries cannot be addressed by the shader's `u32` entry count.
    TooManyPages,
    /// The raw records passed to the strict sorted constructor were unordered.
    KeysNotSorted,
    /// The staged upload does not belong to the adapter's current state.
    InvalidUpload,
    /// A table is already staged or has an upload in flight.
    UploadAlreadyInFlight,
    /// No staged table exists for the requested operation.
    NoStagedUpload,
    /// The submitted queue value is not a valid non-zero timeline value.
    InvalidCompletionValue,
    /// The queue completion proof has not reached the upload's value.
    CompletionNotReached,
    /// The table epoch would move backwards.
    EpochRollback,
    /// A same-epoch table generation would move backwards or repeat.
    GenerationRollback,
    /// A publication serial would wrap and alias an earlier upload.
    PublicationSerialExhausted,
    /// The private snapshot arrays no longer describe one another.
    SnapshotInvariantViolation,
}

/// Immutable, key-sorted material-VT table ready to be copied to a storage
/// buffer.
///
/// The table contains only records that were marked ready by the producer and
/// matched one `(epoch, generation)` pair. It does not claim that a GPU copy
/// has completed; use [`MaterialVirtualTexturePageTableAdapter`] for that
/// upload/publish transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTexturePageTableSnapshot {
    epoch: u64,
    generation: u64,
    records: Vec<MaterialVirtualTexturePageRecord>,
    entries: Vec<GpuMaterialVirtualTextureEntry>,
    words: Vec<u32>,
    indices: BTreeMap<TexturePageKey, u32>,
}

impl MaterialVirtualTexturePageTableSnapshot {
    fn entry_for_record(
        record: &MaterialVirtualTexturePageRecord,
    ) -> GpuMaterialVirtualTextureEntry {
        GpuMaterialVirtualTextureEntry {
            texture: record.key.texture,
            mip_layer: (u32::from(record.key.mip) << 24) | (u32::from(record.key.layer) << 8),
            page_xy: (u32::from(record.key.x) << 16) | u32::from(record.key.y),
            slot: record.slot,
        }
    }

    /// Builds a deterministic snapshot from producer records.
    ///
    /// Input traversal order is irrelevant: records are sorted by the exact
    /// key order used by the shader's unsigned lexicographic binary search.
    /// Every input record must be ready and agree with the requested epoch and
    /// generation; unready records are rejected rather than silently omitted.
    pub fn try_from_records(
        epoch: u64,
        generation: u64,
        records: impl IntoIterator<Item = MaterialVirtualTexturePageRecord>,
    ) -> Result<Self, MaterialVirtualTexturePageTableError> {
        let mut records: Vec<_> = records.into_iter().collect();
        records.sort_unstable_by_key(|record| record.key);
        Self::try_from_sorted_records(epoch, generation, records)
    }

    /// Builds a snapshot while requiring the caller's records to already be
    /// strictly key-sorted.
    ///
    /// This is the boundary for a producer that promises deterministic order;
    /// it is intentionally strict so an accidental unsorted upload cannot
    /// invalidate the shader's binary search.
    pub fn try_from_sorted_records(
        epoch: u64,
        generation: u64,
        records: Vec<MaterialVirtualTexturePageRecord>,
    ) -> Result<Self, MaterialVirtualTexturePageTableError> {
        if generation == 0 {
            return Err(MaterialVirtualTexturePageTableError::ZeroGeneration);
        }
        if records.len() > u32::MAX as usize {
            return Err(MaterialVirtualTexturePageTableError::TooManyPages);
        }

        let mut previous_key = None;
        let mut keys = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut entries = Vec::with_capacity(records.len());
        let word_count = records
            .len()
            .checked_mul(MATERIAL_VT_ENTRY_WORDS)
            .ok_or(MaterialVirtualTexturePageTableError::TooManyPages)?;
        let mut words = Vec::with_capacity(word_count);
        let mut indices = BTreeMap::new();

        for (position, record) in records.iter().enumerate() {
            if previous_key.is_some_and(|previous| previous >= record.key) {
                return Err(if previous_key == Some(record.key) {
                    MaterialVirtualTexturePageTableError::DuplicateKey
                } else {
                    MaterialVirtualTexturePageTableError::KeysNotSorted
                });
            }
            previous_key = Some(record.key);
            if !record.ready {
                return Err(MaterialVirtualTexturePageTableError::PageNotReady);
            }
            if record.epoch != epoch {
                return Err(MaterialVirtualTexturePageTableError::EpochMismatch);
            }
            if record.generation != generation {
                return Err(MaterialVirtualTexturePageTableError::GenerationMismatch);
            }
            if record.slot == MATERIAL_VT_MISS {
                return Err(MaterialVirtualTexturePageTableError::MissSentinelReserved);
            }
            if record.key.texture == MATERIAL_VT_NON_VIRTUAL_TEXTURE {
                return Err(MaterialVirtualTexturePageTableError::NonVirtualTextureReserved);
            }
            if !keys.insert(record.key) {
                return Err(MaterialVirtualTexturePageTableError::DuplicateKey);
            }
            if !slots.insert(record.slot) {
                return Err(MaterialVirtualTexturePageTableError::DuplicatePhysicalSlot);
            }

            let entry = Self::entry_for_record(record);
            let page_index = u32::try_from(position)
                .map_err(|_| MaterialVirtualTexturePageTableError::TooManyPages)?;
            indices.insert(record.key, page_index);
            words.extend_from_slice(&entry.words());
            entries.push(entry);
        }

        let snapshot = Self {
            epoch,
            generation,
            records,
            entries,
            words,
            indices,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Material epoch represented by this snapshot.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Producer generation represented by this snapshot.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of resident material-VT pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the snapshot has no resident pages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Producer records retained for CPU-side publication diagnostics.
    #[must_use]
    pub fn records(&self) -> &[MaterialVirtualTexturePageRecord] {
        &self.records
    }

    /// Typed entries matching the WESL ABI.
    #[must_use]
    pub fn entries(&self) -> &[GpuMaterialVirtualTextureEntry] {
        &self.entries
    }

    /// Flat words to upload verbatim to the WESL storage buffer.
    #[must_use]
    pub fn gpu_words(&self) -> &[u32] {
        &self.words
    }

    /// Exact resident lookup, returning `None` for the shader miss sentinel.
    #[must_use]
    pub fn lookup(&self, key: TexturePageKey) -> Option<u32> {
        self.indices
            .get(&key)
            .and_then(|index| self.entries.get(*index as usize))
            .map(|entry| entry.slot)
    }

    /// Rechecks the CPU table against the material-VT WESL contract.
    #[must_use]
    pub fn validate(&self) -> Result<(), MaterialVirtualTexturePageTableError> {
        if self.generation == 0 {
            return Err(MaterialVirtualTexturePageTableError::ZeroGeneration);
        }
        let Some(expected_word_count) = self.entries.len().checked_mul(MATERIAL_VT_ENTRY_WORDS)
        else {
            return Err(MaterialVirtualTexturePageTableError::SnapshotInvariantViolation);
        };
        if self.records.len() != self.entries.len()
            || self.records.len() != self.indices.len()
            || self.words.len() != expected_word_count
            || self.records.len() > u32::MAX as usize
        {
            return Err(MaterialVirtualTexturePageTableError::SnapshotInvariantViolation);
        }

        let mut previous_key = None;
        let mut slots = BTreeSet::new();
        for (position, ((record, entry), words)) in self
            .records
            .iter()
            .zip(&self.entries)
            .zip(self.words.chunks_exact(MATERIAL_VT_ENTRY_WORDS))
            .enumerate()
        {
            if previous_key.is_some_and(|previous| previous >= record.key) {
                return Err(if previous_key == Some(record.key) {
                    MaterialVirtualTexturePageTableError::DuplicateKey
                } else {
                    MaterialVirtualTexturePageTableError::KeysNotSorted
                });
            }
            previous_key = Some(record.key);
            if !record.ready {
                return Err(MaterialVirtualTexturePageTableError::PageNotReady);
            }
            if record.epoch != self.epoch {
                return Err(MaterialVirtualTexturePageTableError::EpochMismatch);
            }
            if record.generation != self.generation {
                return Err(MaterialVirtualTexturePageTableError::GenerationMismatch);
            }
            if record.slot == MATERIAL_VT_MISS {
                return Err(MaterialVirtualTexturePageTableError::MissSentinelReserved);
            }
            if record.key.texture == MATERIAL_VT_NON_VIRTUAL_TEXTURE {
                return Err(MaterialVirtualTexturePageTableError::NonVirtualTextureReserved);
            }
            if !slots.insert(record.slot) {
                return Err(MaterialVirtualTexturePageTableError::DuplicatePhysicalSlot);
            }
            let Ok(page_index) = u32::try_from(position) else {
                return Err(MaterialVirtualTexturePageTableError::SnapshotInvariantViolation);
            };
            if self.indices.get(&record.key) != Some(&page_index) {
                return Err(MaterialVirtualTexturePageTableError::DuplicateKey);
            }
            let expected = Self::entry_for_record(record);
            if *entry != expected || entry.words() != words {
                return Err(MaterialVirtualTexturePageTableError::InvalidUpload);
            }
        }
        Ok(())
    }
}

/// Upload data returned by the material-VT adapter after CPU validation.
///
/// The render backend records [`Self::gpu_words`] into a storage buffer. This
/// object does not claim the copy has happened; only `submit` followed by a
/// queue completion proof can make it publishable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureUpload {
    serial: u64,
    snapshot: MaterialVirtualTexturePageTableSnapshot,
}

/// The dedicated storage buffer receiving material-VT page-table entries.
///
/// This wrapper keeps the material-VT upload path distinct from the indexed
/// shade-cache table. Construct it only for the storage buffer bound to the
/// material-VT resolver; the adapter does not accept a shade-table snapshot or
/// a bare buffer id directly at its upload boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureTableBuffer {
    buffer: BufferId,
}

impl MaterialVirtualTextureTableBuffer {
    /// Marks a backend buffer as the dedicated material-VT table target.
    ///
    /// The caller owns the backend buffer layout and binding declaration; this
    /// explicit constructor is the point where that resource contract is
    /// asserted. It is intentionally not a `From<BufferId>` conversion, so a
    /// shade-table id cannot flow into an upload call accidentally.
    #[must_use]
    pub const fn new(buffer: BufferId) -> Self {
        Self { buffer }
    }
}

impl MaterialVirtualTextureUpload {
    /// Adapter-local upload serial, useful for backend tracing.
    #[must_use]
    pub const fn serial(&self) -> u64 {
        self.serial
    }

    /// Epoch carried by the upload.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.snapshot.epoch()
    }

    /// Generation carried by the upload.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.snapshot.generation()
    }

    /// Number of entries the backend must expose to the shader.
    #[must_use]
    pub fn entry_count(&self) -> u32 {
        self.snapshot.len() as u32
    }

    /// Exact words for a storage-buffer upload.
    #[must_use]
    pub fn gpu_words(&self) -> &[u32] {
        self.snapshot.gpu_words()
    }

    /// The validated snapshot represented by this upload.
    #[must_use]
    pub fn snapshot(&self) -> &MaterialVirtualTexturePageTableSnapshot {
        &self.snapshot
    }
}

/// A queue timeline proof used to distinguish submitted from completed work.
///
/// Its value is private so callers cannot manufacture readiness by constructing
/// a large number. The adapter obtains it from the completion tracker itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaterialVirtualTextureCompletionProof {
    completed: GpuCompletionValue,
}

impl MaterialVirtualTextureCompletionProof {
    /// Captures the completion value observed by the render queue tracker.
    #[must_use]
    fn from_tracker(tracker: &GpuCompletionTracker) -> Self {
        Self {
            completed: tracker.completed_value(),
        }
    }
}

/// A submitted upload that is not publishable until its queue value completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureSubmission {
    serial: u64,
    completion: GpuCompletionValue,
}

/// A completion-verified table waiting for the atomic active-table swap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureReadyToPublish {
    serial: u64,
    snapshot: MaterialVirtualTexturePageTableSnapshot,
}

impl MaterialVirtualTextureReadyToPublish {
    /// The validated snapshot that will become active on publish.
    #[must_use]
    pub fn snapshot(&self) -> &MaterialVirtualTexturePageTableSnapshot {
        &self.snapshot
    }
}

/// CPU-side upload/publish state machine for the material-VT table.
///
/// This adapter owns no `wgpu` buffer and does not invent page records. A real
/// producer supplies ready records, the backend uploads the returned words,
/// and the queue completion tracker supplies the proof required before the
/// active snapshot is swapped. The existing shade-cache table is not accepted
/// by any of these typed APIs.
#[derive(Clone, Debug, Default)]
pub struct MaterialVirtualTexturePageTableAdapter {
    active: Option<MaterialVirtualTexturePageTableSnapshot>,
    staged: Option<MaterialVirtualTextureUpload>,
    submitted: Option<MaterialVirtualTextureSubmission>,
    completed: Option<MaterialVirtualTextureReadyToPublish>,
    next_serial: u64,
}

impl MaterialVirtualTexturePageTableAdapter {
    /// Creates an adapter with no active material-VT pages.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: None,
            staged: None,
            submitted: None,
            completed: None,
            next_serial: 0,
        }
    }

    /// Returns the last atomically published snapshot, if any.
    #[must_use]
    pub fn active(&self) -> Option<&MaterialVirtualTexturePageTableSnapshot> {
        self.active.as_ref()
    }

    /// Validates and stages a producer snapshot for backend upload.
    pub fn stage(
        &mut self,
        snapshot: MaterialVirtualTexturePageTableSnapshot,
    ) -> Result<MaterialVirtualTextureUpload, MaterialVirtualTexturePageTableError> {
        if self.staged.is_some() || self.submitted.is_some() || self.completed.is_some() {
            return Err(MaterialVirtualTexturePageTableError::UploadAlreadyInFlight);
        }
        snapshot.validate()?;
        if let Some(active) = &self.active {
            if snapshot.epoch < active.epoch {
                return Err(MaterialVirtualTexturePageTableError::EpochRollback);
            }
            if snapshot.epoch == active.epoch && snapshot.generation <= active.generation {
                return Err(MaterialVirtualTexturePageTableError::GenerationRollback);
            }
        }
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or(MaterialVirtualTexturePageTableError::PublicationSerialExhausted)?;
        let upload = MaterialVirtualTextureUpload {
            serial: self.next_serial,
            snapshot,
        };
        self.staged = Some(upload.clone());
        Ok(upload)
    }

    /// Checks that an upload is the currently staged table before touching the
    /// backend queue.
    fn validate_submission(
        &mut self,
        upload: MaterialVirtualTextureUpload,
    ) -> Result<(), MaterialVirtualTexturePageTableError> {
        if self.submitted.is_some() || self.completed.is_some() {
            return Err(MaterialVirtualTexturePageTableError::UploadAlreadyInFlight);
        }
        if self.staged.as_ref() != Some(&upload) {
            return Err(MaterialVirtualTexturePageTableError::InvalidUpload);
        }
        Ok(())
    }

    /// Records a queue timeline value after the real backend upload has been
    /// enqueued. This helper is private so callers cannot supply a fake value.
    fn record_submission(
        &mut self,
        upload: MaterialVirtualTextureUpload,
        completion: GpuCompletionValue,
    ) -> Result<MaterialVirtualTextureSubmission, MaterialVirtualTexturePageTableError> {
        if completion.0 == 0 {
            return Err(MaterialVirtualTexturePageTableError::InvalidCompletionValue);
        }
        let submission = MaterialVirtualTextureSubmission {
            serial: upload.serial,
            completion,
        };
        self.submitted = Some(submission);
        Ok(submission)
    }

    /// Uploads the validated words to the dedicated material-VT buffer and
    /// records the real queue timeline value for that upload.
    ///
    /// `GpuCompletionTracker::track` is called only after `write_buffer`, so a
    /// later completion proof covers the table write as well as earlier queue
    /// work. No producer records are created here and no caller-supplied
    /// completion value is accepted.
    pub fn upload_and_submit(
        &mut self,
        upload: MaterialVirtualTextureUpload,
        target: MaterialVirtualTextureTableBuffer,
        queue: &WgpuQueue,
        tracker: &GpuCompletionTracker,
    ) -> Result<MaterialVirtualTextureSubmission, MaterialVirtualTexturePageTableError> {
        self.validate_submission(upload.clone())?;
        if !upload.gpu_words().is_empty() {
            queue.write_buffer(target.buffer, 0, bytemuck::cast_slice(upload.gpu_words()));
        }
        let completion = tracker.track(queue);
        self.record_submission(upload, completion)
    }

    /// Converts a submitted upload into a publish token only after queue
    /// completion has been observed by the tracker.
    pub fn complete(
        &mut self,
        submission: MaterialVirtualTextureSubmission,
        tracker: &GpuCompletionTracker,
    ) -> Result<MaterialVirtualTextureReadyToPublish, MaterialVirtualTexturePageTableError> {
        self.complete_with_proof(
            submission,
            MaterialVirtualTextureCompletionProof::from_tracker(tracker),
        )
    }

    fn complete_with_proof(
        &mut self,
        submission: MaterialVirtualTextureSubmission,
        proof: MaterialVirtualTextureCompletionProof,
    ) -> Result<MaterialVirtualTextureReadyToPublish, MaterialVirtualTexturePageTableError> {
        if self.submitted != Some(submission) {
            return Err(MaterialVirtualTexturePageTableError::InvalidUpload);
        }
        if proof.completed < submission.completion {
            return Err(MaterialVirtualTexturePageTableError::CompletionNotReached);
        }
        let Some(upload) = self.staged.take() else {
            return Err(MaterialVirtualTexturePageTableError::NoStagedUpload);
        };
        if upload.serial != submission.serial {
            return Err(MaterialVirtualTexturePageTableError::InvalidUpload);
        }
        self.submitted = None;
        let ready = MaterialVirtualTextureReadyToPublish {
            serial: upload.serial,
            snapshot: upload.snapshot,
        };
        self.completed = Some(ready.clone());
        Ok(ready)
    }

    /// Atomically makes the completion-verified table active.
    pub fn publish(
        &mut self,
        ready: MaterialVirtualTextureReadyToPublish,
    ) -> Result<(), MaterialVirtualTexturePageTableError> {
        if self.completed.as_ref() != Some(&ready) {
            return Err(MaterialVirtualTexturePageTableError::InvalidUpload);
        }
        self.active = Some(ready.snapshot);
        self.completed = None;
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

    fn material_key(texture: u32, mip: u8, layer: u16, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture,
            mip,
            layer,
            x,
            y,
        }
    }

    fn material_page(texture: u32, mip: u8, x: u16, slot: u32) -> MaterialVirtualTexturePageRecord {
        MaterialVirtualTexturePageRecord {
            key: material_key(texture, mip, 2, x, 3),
            slot,
            generation: 9,
            epoch: 17,
            ready: true,
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

    #[test]
    fn material_vt_snapshot_is_key_sorted_and_matches_wesl_words() {
        let snapshot = MaterialVirtualTexturePageTableSnapshot::try_from_records(
            17,
            9,
            [material_page(2, 1, 4, 8), material_page(1, 0, 2, 3)],
        )
        .unwrap();

        assert_eq!(snapshot.entries()[0].words(), [1, 2 << 8, 2 << 16 | 3, 3]);
        assert_eq!(
            snapshot.entries()[1].words(),
            [2, (1 << 24) | (2 << 8), 4 << 16 | 3, 8]
        );
        assert_eq!(snapshot.lookup(material_key(1, 0, 2, 2, 3)), Some(3));
        assert_eq!(snapshot.lookup(material_key(7, 0, 0, 0, 0)), None);
        assert_eq!(snapshot.gpu_words().len(), 2 * MATERIAL_VT_ENTRY_WORDS);
        assert!(snapshot.validate().is_ok());
    }

    #[test]
    fn material_vt_strict_constructor_rejects_unsorted_keys() {
        let records = vec![material_page(2, 0, 0, 1), material_page(1, 0, 0, 2)];
        assert_eq!(
            MaterialVirtualTexturePageTableSnapshot::try_from_sorted_records(17, 9, records),
            Err(MaterialVirtualTexturePageTableError::KeysNotSorted)
        );
    }

    #[test]
    fn material_vt_rejects_unready_stale_and_reserved_records() {
        let mut page = material_page(1, 0, 0, 1);
        page.ready = false;
        assert_eq!(
            MaterialVirtualTexturePageTableSnapshot::try_from_records(17, 9, [page]),
            Err(MaterialVirtualTexturePageTableError::PageNotReady)
        );

        let mut page = material_page(1, 0, 0, 1);
        page.epoch = 18;
        assert_eq!(
            MaterialVirtualTexturePageTableSnapshot::try_from_records(17, 9, [page]),
            Err(MaterialVirtualTexturePageTableError::EpochMismatch)
        );

        let mut page = material_page(1, 0, 0, 1);
        page.generation = 8;
        assert_eq!(
            MaterialVirtualTexturePageTableSnapshot::try_from_records(17, 9, [page]),
            Err(MaterialVirtualTexturePageTableError::GenerationMismatch)
        );

        let mut page = material_page(1, 0, 0, MATERIAL_VT_MISS);
        assert_eq!(
            MaterialVirtualTexturePageTableSnapshot::try_from_records(17, 9, [page]),
            Err(MaterialVirtualTexturePageTableError::MissSentinelReserved)
        );
        page.slot = 1;
        page.key.texture = MATERIAL_VT_NON_VIRTUAL_TEXTURE;
        assert_eq!(
            MaterialVirtualTexturePageTableSnapshot::try_from_records(17, 9, [page]),
            Err(MaterialVirtualTexturePageTableError::NonVirtualTextureReserved)
        );
    }

    #[test]
    fn material_vt_adapter_requires_completion_before_publish() {
        let snapshot = MaterialVirtualTexturePageTableSnapshot::try_from_records(
            17,
            9,
            [material_page(1, 0, 0, 4)],
        )
        .unwrap();
        let mut adapter = MaterialVirtualTexturePageTableAdapter::new();
        let upload = adapter.stage(snapshot.clone()).unwrap();
        assert_eq!(upload.entry_count(), 1);
        let submission = adapter
            .record_submission(upload.clone(), GpuCompletionValue(4))
            .unwrap();
        let tracker = GpuCompletionTracker::default();
        assert_eq!(
            adapter.complete(submission, &tracker),
            Err(MaterialVirtualTexturePageTableError::CompletionNotReached)
        );
        assert_eq!(
            adapter.complete_with_proof(
                submission,
                MaterialVirtualTextureCompletionProof {
                    completed: GpuCompletionValue(3),
                },
            ),
            Err(MaterialVirtualTexturePageTableError::CompletionNotReached)
        );
        assert!(adapter.active().is_none());

        let ready = adapter
            .complete_with_proof(
                submission,
                MaterialVirtualTextureCompletionProof {
                    completed: GpuCompletionValue(4),
                },
            )
            .unwrap();
        assert!(adapter.active().is_none());
        adapter.publish(ready).unwrap();
        assert_eq!(adapter.active(), Some(&snapshot));
    }

    #[test]
    fn material_vt_adapter_rejects_epoch_and_generation_rollback() {
        let first = MaterialVirtualTexturePageTableSnapshot::try_from_records(
            17,
            9,
            [material_page(1, 0, 0, 4)],
        )
        .unwrap();
        let mut adapter = MaterialVirtualTexturePageTableAdapter::new();
        let upload = adapter.stage(first).unwrap();
        let submission = adapter
            .record_submission(upload, GpuCompletionValue(1))
            .unwrap();
        let ready = adapter
            .complete_with_proof(
                submission,
                MaterialVirtualTextureCompletionProof {
                    completed: GpuCompletionValue(1),
                },
            )
            .unwrap();
        adapter.publish(ready).unwrap();

        let mut lower_generation = material_page(1, 0, 0, 4);
        lower_generation.generation = 8;
        let lower_generation =
            MaterialVirtualTexturePageTableSnapshot::try_from_records(17, 8, [lower_generation])
                .unwrap();
        assert_eq!(
            adapter.stage(lower_generation),
            Err(MaterialVirtualTexturePageTableError::GenerationRollback)
        );

        let mut lower_epoch = material_page(1, 0, 0, 4);
        lower_epoch.epoch = 16;
        lower_epoch.generation = 10;
        let lower_epoch =
            MaterialVirtualTexturePageTableSnapshot::try_from_records(16, 10, [lower_epoch])
                .unwrap();
        assert_eq!(
            adapter.stage(lower_epoch),
            Err(MaterialVirtualTexturePageTableError::EpochRollback)
        );
    }
}
