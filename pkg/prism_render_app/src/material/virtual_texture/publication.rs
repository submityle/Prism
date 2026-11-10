//! Full-snapshot upload, completion qualification, and dependent-row refresh.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
};

use prism_asset::AssetId;
use prism_ecs::prelude::Resource;
use prism_image::Image;
use prism_render_architecture::texture_streaming::{GpuPageTable, TexturePageKey};
use prism_render_driver::{
    Extent3d, FrameSync, ImageDataLayout, Instance, RenderDevice, RenderQueue, TextureWrite,
};
use prism_render_material::{
    build_material_vt_atlas_tile, GpuMaterialTexture, MaterialRegistry, MaterialSnapshot,
    MaterialTextureResidencyUpdate, MaterialVtAtlasConfig, MaterialVtAtlasTile,
};
use prism_render_runtime::RenderRuntime;

use super::{
    gpu::{check_owner_runtime, GpuSlice},
    NativeVtBinding, NativeVtConfig, NativeVtError, NativeVtGpuConfig, NativeVtSource,
    NativeVtSources,
};
use crate::material::{BindlessSlot, RenderMaterials, TextureSlotHeap};

/// The entire desired residency, not a delta. Slots are physical atlas slots;
/// portable images instead occupy their current bindless slot's atlas tile.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeVtResidency {
    pub pages: Vec<(TexturePageKey, u32)>,
    pub portable: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVtPublish {
    Idle,
    Waiting,
    Published { refreshed_rows: u32 },
}

#[derive(Clone)]
pub(super) struct ResidentSource {
    pub(super) source: Arc<NativeVtSource>,
    slot: BindlessSlot,
    portable: bool,
}

impl ResidentSource {
    fn validate(
        &self,
        sources: &NativeVtSources,
        heap: &TextureSlotHeap,
    ) -> Result<(), NativeVtError> {
        let texture = self.source.layout().descriptor().texture_id();
        if !sources.is_current(&self.source) {
            return Err(NativeVtError::StaleSource(texture));
        }
        if heap.slot_of(self.source.image) != Some(self.slot) {
            return Err(NativeVtError::StaleSlot(texture));
        }
        Ok(())
    }

    fn row(&self, row: &GpuMaterialTexture) -> GpuMaterialTexture {
        if self.portable {
            let descriptor = self.source.layout().descriptor();
            GpuMaterialTexture::with_portable_atlas(
                row.index,
                row.generation,
                row.semantic,
                row.sampler_index,
                descriptor.width(),
                descriptor.height(),
                descriptor.page_size(),
            )
        } else {
            self.source
                .layout()
                .texture_row(row.index, row.generation, row.semantic, row.sampler_index)
                .gpu
        }
    }
}

/// Published ownership needs identity and row metadata, not the source payload.
/// A weak token prevents pointer ABA without pinning replaced image bytes.
#[derive(Clone)]
struct PublishedSource {
    token: Weak<NativeVtSource>,
    image: AssetId<Image>,
    texture: u32,
    slot: BindlessSlot,
    template: GpuMaterialTexture,
}

impl PublishedSource {
    fn new(resident: &ResidentSource) -> Self {
        Self {
            token: Arc::downgrade(&resident.source),
            image: resident.source.image,
            texture: resident.source.layout().descriptor().texture_id(),
            slot: resident.slot,
            template: resident.row(&GpuMaterialTexture::non_virtual(
                resident.slot.index,
                resident.slot.generation,
                0,
                0,
            )),
        }
    }

    fn valid(&self, sources: &NativeVtSources, heap: &TextureSlotHeap) -> bool {
        sources.get(self.texture).is_some_and(|source| {
            // The retained Weak prevents pointer reuse; checking the pointer
            // directly avoids weak-count atomics on quiet-frame checks.
            std::ptr::eq(self.token.as_ptr(), Arc::as_ptr(source))
        }) && heap.slot_of(self.image) == Some(self.slot)
    }

    fn row(&self, row: &GpuMaterialTexture) -> GpuMaterialTexture {
        GpuMaterialTexture {
            semantic: row.semantic,
            sampler_index: row.sampler_index,
            ..self.template
        }
    }
}

// Semantic/sampler are authored state and deliberately not ownership identity.
fn row_identity(row: &GpuMaterialTexture) -> [u32; 6] {
    [
        row.index,
        row.generation,
        row.virtual_texture_id,
        row.virtual_texture_width,
        row.virtual_texture_height,
        row.virtual_texture_page_size,
    ]
}

pub(super) struct PreparedUpload {
    pub(super) sources: Vec<ResidentSource>,
    pub(super) pages: Vec<(TexturePageKey, u32)>,
    pub(super) keys: BTreeSet<TexturePageKey>,
    pub(super) table: GpuPageTable,
    pub(super) config: NativeVtGpuConfig,
}

pub(super) struct Pending {
    pub(super) slice: usize,
    pub(super) sources: Vec<ResidentSource>,
    pub(super) complete: Arc<AtomicBool>,
}

/// Owns two complete, creator-device-specific atlas/table/config snapshots.
/// All mutation is exclusive; run publication/refresh before material buffer
/// sync, and keep a binding lease with every bind group that samples the atlas.
pub struct NativeMaterialVt<D: RenderDevice> {
    /// Rotates on every owner/resource recreation, including same-device
    /// rebuilds. A Weak token prevents identity ABA in scheduler/feedback.
    pub(super) incarnation: Arc<()>,
    pub(super) config: NativeVtConfig,
    pub(super) slices: [Arc<GpuSlice<D>>; 2],
    pub(super) active: Option<usize>,
    active_is_empty: bool,
    pub(super) pending: Option<Pending>,
    /// Monotonic logical publication generation for frame bundle admission.
    publication_generation: u64,
    published: Vec<PublishedSource>,
    // Keep exact lightweight tombstones across publications. An authored row
    // can arrive late with metadata from more than one publication ago.
    owned_rows: BTreeSet<[u32; 6]>,
    row_refresh: Option<(MaterialSnapshot, Vec<bool>)>,
}

impl<D: RenderDevice + Send + Sync + 'static> Resource for NativeMaterialVt<D> {}

impl<D: RenderDevice + Send + Sync + 'static> NativeMaterialVt<D> {
    pub fn new<I: Instance<Device = D> + 'static>(
        runtime: &RenderRuntime<I>,
        config: NativeVtConfig,
    ) -> Result<Self, NativeVtError> {
        if runtime.device_lost().is_some() {
            return Err(NativeVtError::DeviceLost);
        }
        let abandoned = Arc::new(AtomicBool::new(false));
        Ok(Self {
            incarnation: Arc::new(()),
            config,
            slices: [
                GpuSlice::new(
                    runtime.device_handle(),
                    runtime.device_generation(),
                    config,
                    Arc::clone(&abandoned),
                )?,
                GpuSlice::new(
                    runtime.device_handle(),
                    runtime.device_generation(),
                    config,
                    abandoned,
                )?,
            ],
            active: None,
            active_is_empty: false,
            pending: None,
            publication_generation: 0,
            published: Vec::new(),
            owned_rows: BTreeSet::new(),
            row_refresh: None,
        })
    }

    pub fn config(&self) -> NativeVtConfig {
        self.config
    }
    pub fn has_pending_upload(&self) -> bool {
        self.pending.is_some()
    }

    /// Installs the zero-initialized disabled binding used before the first
    /// authored VT source is resident. It performs no queue work and is only
    /// used by the production plugin's cold-start path.
    pub(crate) fn activate_empty(&mut self) {
        if self.active.is_none() && self.pending.is_none() {
            self.active = Some(0);
            self.active_is_empty = true;
            self.publication_generation = 1;
        }
    }

    /// Logical VT publication generation, independent of physical device IDs.
    #[must_use]
    pub fn publication_generation(&self) -> u64 {
        self.publication_generation
    }

    pub(super) fn active_is_empty(&self) -> bool {
        self.active_is_empty
    }

    /// Permanently retire both allocations after known device loss or when
    /// shutdown cannot prove the submitting device idle. Existing leases and
    /// callbacks retain their IDs without issuing unsafe explicit destroys.
    pub(crate) fn abandon_device(&self) {
        for slice in &self.slices {
            slice.abandon();
        }
    }

    /// Replace creator-device resources after recovery, retaining lightweight
    /// row ownership so `refresh_rows` can revoke the old device's metadata.
    /// No binding is available until a new upload really completes. Allocate
    /// both new slices before dropping any previous state; old callbacks and
    /// consumer leases continue to own their original device's slices.
    pub fn recreate_for_device<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
    ) -> Result<(), NativeVtError> {
        let mut replacement = Self::new(runtime, self.config)?;
        replacement.owned_rows = std::mem::take(&mut self.owned_rows);
        *self = replacement;
        Ok(())
    }

    /// CPU validation and tile construction finish before any queue write.
    /// An actual empty submission flushes queued writes; only the subsequent
    /// completion callback qualifies publication, including on null backends.
    /// Residency is a complete snapshot. This does not deduplicate identical
    /// submissions: the scheduler should stage only changes, and retry
    /// `PendingUpload`/`SliceBusy` after pending work/consumer leases progress.
    pub fn stage<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        sources: &NativeVtSources,
        heap: &TextureSlotHeap,
        residency: &NativeVtResidency,
    ) -> Result<(), NativeVtError>
    where
        I::Queue: FrameSync,
    {
        check_owner_runtime(&self.slices, runtime)?;
        if self.pending.is_some() {
            return Err(NativeVtError::PendingUpload);
        }
        let index = self.active.map_or(0, |active| 1 - active);
        if !GpuSlice::idle(&self.slices[index]) {
            return Err(NativeVtError::SliceBusy);
        }
        let prepared = prepare_upload(self.config, sources, heap, residency)?;
        let tiles = prepare_cpu_tiles(self.config, &prepared)?;
        let complete = upload(
            &self.slices[index],
            runtime.queue(),
            self.config,
            &prepared,
            &tiles,
        );
        self.pending = Some(Pending {
            slice: index,
            sources: prepared.sources,
            complete,
        });
        Ok(())
    }

    /// Logical cancellation never discharges GPU work. The inactive resources
    /// stay pinned/busy until their real callback and all consumer leases end.
    pub fn cancel(&mut self) -> bool {
        self.pending.take().is_some()
    }

    /// Source/slot staleness cancels the pending job. Device failures leave it
    /// unpublished; use `recreate_for_device` on device recovery. A row preflight
    /// error keeps the completed job retryable and changes no rows or bindings.
    /// `Waiting` retains the previous active slice. After handling this result
    /// (including a stale-source cancellation), run `refresh_rows` before sync.
    pub fn try_publish<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        sources: &NativeVtSources,
        materials: &mut RenderMaterials,
    ) -> Result<NativeVtPublish, NativeVtError> {
        check_owner_runtime(&self.slices, runtime)?;
        let Some(pending) = &self.pending else {
            return Ok(NativeVtPublish::Idle);
        };
        if !pending.complete.load(Ordering::Acquire) {
            return Ok(NativeVtPublish::Waiting);
        }
        for source in &pending.sources {
            if let Err(error) = source.validate(sources, materials.textures()) {
                self.cancel();
                return Err(error);
            }
        }
        let published: Vec<_> = pending.sources.iter().map(PublishedSource::new).collect();
        let updates = prepare_row_updates(materials.registry(), &self.owned_rows, &published);
        let count = materials
            .registry_mut()
            .update_texture_residency(&updates)
            .map_err(NativeVtError::MaterialRegistry)?;
        let pending = self.pending.take().expect("validated pending upload");
        self.active = Some(pending.slice);
        self.active_is_empty = published.is_empty();
        self.publication_generation = self.publication_generation.wrapping_add(1).max(1);
        self.owned_rows.extend(
            published
                .iter()
                .map(|source| row_identity(&source.template)),
        );
        self.published = published;
        self.row_refresh = Some((
            materials.registry().snapshot(),
            vec![true; self.published.len()],
        ));
        Ok(NativeVtPublish::Published {
            refreshed_rows: count,
        })
    }

    /// Run every frame, even without uploads: new authored rows need metadata,
    /// and replaced/removed sources must lose metadata before the buffer sync.
    /// Invalid sources are not adopted again without a fresh completed upload.
    /// Returns the number of changed material records, not individual texture
    /// entries. On failure no record is changed; callers must not sync/consume
    /// invalid VT rows as though revocation had succeeded.
    pub fn refresh_rows<I: Instance<Device = D> + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        sources: &NativeVtSources,
        materials: &mut RenderMaterials,
    ) -> Result<u32, NativeVtError> {
        check_owner_runtime(&self.slices, runtime)?;
        // Quiet frames check the registry snapshot, source identities and heap
        // leases without rescanning/cloning material texture rows. The registry
        // snapshot itself currently counts occupied slots, so this is not O(1).
        if let Some((snapshot, valid)) = &self.row_refresh
            && *snapshot == materials.registry().snapshot()
            && self.published.iter().zip(valid).all(|(source, was_valid)| {
                source.valid(sources, materials.textures()) == *was_valid
            })
        {
            return Ok(0);
        }
        let valid: Vec<_> = self
            .published
            .iter()
            .map(|source| source.valid(sources, materials.textures()))
            .collect();
        let current: Vec<_> = self
            .published
            .iter()
            .zip(&valid)
            .filter(|(_, valid)| **valid)
            .map(|(source, _)| source.clone())
            .collect();
        let updates = prepare_row_updates(materials.registry(), &self.owned_rows, &current);
        let count = materials
            .registry_mut()
            .update_texture_residency(&updates)
            .map_err(NativeVtError::MaterialRegistry)?;
        self.row_refresh = Some((materials.registry().snapshot(), valid));
        // Keep the old ownership identities for clearing stale rows introduced
        // by a late authored publication. They never authorize sampling again.
        Ok(count)
    }

    /// None before first completion-qualified publication. Stage an empty set
    /// first if a disabled but initialized binding is needed at startup.
    pub fn active_binding<I: Instance<Device = D> + 'static>(
        &self,
        runtime: &RenderRuntime<I>,
    ) -> Result<Option<NativeVtBinding<D>>, NativeVtError> {
        check_owner_runtime(&self.slices, runtime)?;
        Ok(self.active.map(|index| NativeVtBinding {
            owner: Arc::clone(&self.slices[index]),
        }))
    }
}

pub(super) fn prepare_upload(
    config: NativeVtConfig,
    sources: &NativeVtSources,
    heap: &TextureSlotHeap,
    residency: &NativeVtResidency,
) -> Result<PreparedUpload, NativeVtError> {
    let capacity = config.capacity()?;
    if residency.pages.len() > config.max_entries as usize {
        return Err(NativeVtError::TableCapacity);
    }
    let mut keys = BTreeSet::new();
    let mut slots = BTreeSet::new();
    let mut selected = BTreeMap::<u32, ResidentSource>::new();
    for &(key, slot) in &residency.pages {
        if !keys.insert(key) {
            return Err(NativeVtError::DuplicatePage(key));
        }
        reserve_slot(&mut slots, slot, capacity)?;
        if let std::collections::btree_map::Entry::Vacant(entry) = selected.entry(key.texture) {
            entry.insert(select_source(config, sources, heap, key.texture, false)?);
        }
        selected[&key.texture]
            .source
            .layout()
            .page_metadata(key)
            .map_err(NativeVtError::Source)?;
    }
    let mut coarsest_mip = 0;
    for (&texture, resident) in &selected {
        let descriptor = resident.source.layout().descriptor();
        let mip = descriptor.mip_count() - 1;
        for layer in 0..descriptor.layers() {
            let tail = TexturePageKey {
                texture,
                mip,
                layer,
                x: 0,
                y: 0,
            };
            if !keys.contains(&tail) {
                return Err(NativeVtError::MissingCoarsestPage(texture));
            }
        }
        coarsest_mip = coarsest_mip.max(u32::from(mip));
    }
    let mut portable_keys = Vec::new();
    for &texture in &residency.portable {
        if selected.contains_key(&texture) {
            return Err(NativeVtError::InvalidPortableSource(texture));
        }
        let resident = select_source(config, sources, heap, texture, true)?;
        let descriptor = resident.source.layout().descriptor();
        if descriptor.layers() != 1
            || descriptor.width() > config.page_size
            || descriptor.height() > config.page_size
        {
            return Err(NativeVtError::InvalidPortableSource(texture));
        }
        reserve_slot(&mut slots, resident.slot.index, capacity)?;
        portable_keys.push((
            TexturePageKey {
                texture,
                mip: 0,
                layer: 0,
                x: 0,
                y: 0,
            },
            resident.slot.index,
        ));
        selected.insert(texture, resident);
    }
    let pages = residency
        .pages
        .iter()
        .chain(&portable_keys)
        .copied()
        .collect();
    let gpu_config = NativeVtGpuConfig {
        enabled: u32::from(!selected.is_empty()),
        entry_count: residency.pages.len() as u32,
        page_size: config.page_size,
        tile_extent_px: config.geometry()?.tile_extent_px(),
        tiles_per_row: config.tiles_per_row,
        atlas_layers: config.atlas_layers,
        gutter_px: config.gutter_px,
        coarsest_mip,
    };
    Ok(PreparedUpload {
        sources: selected.into_values().collect(),
        pages,
        keys,
        table: GpuPageTable::from_bindings(&residency.pages),
        config: gpu_config,
    })
}

fn prepare_cpu_tiles(
    config: NativeVtConfig,
    prepared: &PreparedUpload,
) -> Result<Vec<MaterialVtAtlasTile>, NativeVtError> {
    let atlas = MaterialVtAtlasConfig::new(config.gutter_px, config.tiles_per_row)
        .map_err(NativeVtError::Upload)?;
    let sources: BTreeMap<_, _> = prepared
        .sources
        .iter()
        .map(|resident| {
            (
                resident.source.layout().descriptor().texture_id(),
                &resident.source,
            )
        })
        .collect();
    prepared
        .pages
        .iter()
        .map(|&(key, slot)| {
            let snapshot = sources[&key.texture]
                .cpu_snapshot()
                .ok_or(NativeVtError::RequiresResidentUpload(key.texture))?;
            build_material_vt_atlas_tile(snapshot, key, slot, &prepared.keys, atlas)
                .map_err(NativeVtError::Upload)
        })
        .collect()
}

fn reserve_slot(slots: &mut BTreeSet<u32>, slot: u32, capacity: u32) -> Result<(), NativeVtError> {
    if slot >= capacity {
        return Err(NativeVtError::SlotOutOfBounds(slot));
    }
    if !slots.insert(slot) {
        return Err(NativeVtError::DuplicateSlot(slot));
    }
    Ok(())
}

fn select_source(
    config: NativeVtConfig,
    sources: &NativeVtSources,
    heap: &TextureSlotHeap,
    texture: u32,
    portable: bool,
) -> Result<ResidentSource, NativeVtError> {
    let source = sources
        .get(texture)
        .ok_or(NativeVtError::MissingSource(texture))?;
    let descriptor = source.layout().descriptor();
    if descriptor.page_size() != config.page_size || descriptor.format() != config.format {
        return Err(NativeVtError::AtlasMismatch(texture));
    }
    let slot = heap
        .slot_of(source.image)
        .ok_or(NativeVtError::StaleSlot(texture))?;
    Ok(ResidentSource {
        source: Arc::clone(source),
        slot,
        portable,
    })
}

fn upload<D: RenderDevice + Send + Sync + 'static>(
    slice: &Arc<GpuSlice<D>>,
    queue: &(impl RenderQueue + FrameSync),
    config: NativeVtConfig,
    prepared: &PreparedUpload,
    tiles: &[MaterialVtAtlasTile],
) -> Arc<AtomicBool> {
    let geometry = config.geometry().expect("prevalidated atlas geometry");
    // If a backend unwinds after a write, this obligation deliberately remains:
    // neither unwinding nor callback abandonment is evidence of GPU completion.
    slice.begin_work();
    for tile in tiles {
        let placement = geometry.slot_placement(tile.slot);
        let extent = tile.layout.stored_tile_extent_px();
        queue.write_texture(
            TextureWrite {
                texture: slice.texture(),
                mip_level: 0,
                origin: [
                    placement.origin_x_px,
                    placement.origin_y_px,
                    placement.layer,
                ],
                size: Extent3d::new_2d(extent, extent),
            },
            &tile.staging_bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(tile.bytes_per_row),
                rows_per_image: Some(extent),
            },
        );
    }
    let empty = [0u32; 4];
    let words = if prepared.table.words().is_empty() {
        &empty[..]
    } else {
        prepared.table.words()
    };
    queue.write_buffer(slice.table(), 0, bytemuck::cast_slice(words));
    queue.write_buffer(slice.config(), 0, bytemuck::bytes_of(&prepared.config));
    drop(queue.submit_tracked(&[]));
    slice.on_completion(queue)
}

/// Prepare only changed texture rows; the registry owns atomic validation and
/// commit. No authored record or runtime tail needs cloning/restoration.
fn prepare_row_updates(
    registry: &MaterialRegistry,
    previous: &BTreeSet<[u32; 6]>,
    next: &[PublishedSource],
) -> Vec<MaterialTextureResidencyUpdate> {
    let next: BTreeMap<_, _> = next
        .iter()
        .map(|source| ((source.slot.index, source.slot.generation), source))
        .collect();
    let mut updates = Vec::new();
    for index in 0..registry.capacity() {
        let Some(record) = registry.record_at(index) else {
            continue;
        };
        for (offset, row) in record.textures.iter().enumerate() {
            let key = (row.index, row.generation);
            let updated = if let Some(source) = next.get(&key) {
                source.row(row)
            } else if previous.contains(&row_identity(row)) {
                GpuMaterialTexture::non_virtual(
                    row.index,
                    row.generation,
                    row.semantic,
                    row.sampler_index,
                )
            } else {
                *row
            };
            if updated != *row {
                updates.push(MaterialTextureResidencyUpdate {
                    handle: record.handle,
                    expected_revision: record.revision,
                    texture_index: offset,
                    expected: *row,
                    replacement: updated,
                });
            }
        }
    }
    updates
}
