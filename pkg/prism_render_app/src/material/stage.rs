//! The material stage: the render world's material registry, and the system
//! that keeps the ABI buffers in step with it.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use prism_asset::AssetId;
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_image::Image;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::{FrameSync, Instance, RenderDevice, RenderQueue};
use prism_render_material::{
    DynamicChannels, MaterialExecutionBinding, MaterialExecutionTables, MaterialRecord,
    MaterialRegistry, MaterialRegistryError, MaterialSnapshot, ShadingPlan,
};

/// The index of the permanent fallback material (slot zero).
pub const FALLBACK_MATERIAL_INDEX: u32 = 0;

use crate::runtime_resource::RenderRuntimeResource;
use crate::{ExtractedNativeMaterials, SceneRegistry};

use super::binding::NativeMaterialTextureBindings;
use super::buffers::{
    ExecutionTableSyncError, MaterialParameterBuffers, MaterialSyncReport,
    NativeExecutionTableBuffers,
};
use super::texture_heap::{
    BindlessSlot, TextureFallback, TextureHeapAcquireError, TextureSlotHeap,
};
use super::virtual_texture::{NativeMaterialVt, NativeVtError, NativeVtScheduler, NativeVtSources};

/// What a material's authored texture reference asks for.
///
/// The caller decides this per slot because it owns the *semantic vocabulary*
/// (`prism_render_material`'s texture semantics live with its Bevy bridge, which
/// the render side does not link): the renderer only has to know whether a slot
/// samples a real image — for which it acquires a bindless slot — or a reserved
/// fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureRequest {
    /// Sample this image; the registry acquires (or references) a bindless slot
    /// for it.
    Image(AssetId<Image>),
    /// Sample the reserved fallback texture this semantic calls for.
    Fallback(TextureFallback),
}

/// Why an authored texture update could not be resolved for a material.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureResolutionError {
    /// The bounded texture heap had no free slot for this semantic.
    CapacityExhausted {
        /// Material row being prepared.
        material_index: u32,
        /// Texture semantic which could not be represented.
        semantic: u32,
    },
}

/// One successful acquisition, not an authored dependency or an overflow
/// fallback. Keep the lease's slot generation so cleanup cannot release a
/// different residency created through the mutable texture-heap API.
#[derive(Clone)]
struct HeldTexture {
    image: AssetId<Image>,
    slot: BindlessSlot,
}

/// The render world's material registry.
///
/// The authoritative records live here, in the render world, because that is
/// where they are needed: the ABI buffers are built from them and a shader reads
/// them every frame. The simulation publishes into this registry through the
/// extract step (the same one-way seam as every other render input), which is
/// why the wrapper — and not a bare `MaterialRegistry` — is the resource: the
/// registry type belongs to `prism_render_material`, so the resource marker has
/// to be implemented on a type local to this crate.
pub struct RenderMaterials {
    registry: MaterialRegistry,
    /// Canonical builtin execution catalog used when extracted records carry
    /// a builtin response plan. Registered plans still require explicit
    /// validated publication and never alias this catalog.
    builtin_execution: Option<MaterialExecutionTables>,
    /// The bindless slot space this registry assigns material textures from.
    textures: TextureSlotHeap,
    /// The images each material currently holds a slot reference for, so a
    /// republish releases exactly what it took and a retirement releases the
    /// rest.
    held: BTreeMap<u32, Vec<HeldTexture>>,
    /// Publication number for the authored texture heap bindings.
    texture_binding_version: u64,
}

impl RenderMaterials {
    /// A registry with room for `capacity` materials.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            registry: MaterialRegistry::new(capacity),
            builtin_execution: None,
            textures: TextureSlotHeap::default(),
            held: BTreeMap::new(),
            texture_binding_version: 0,
        }
    }

    /// Creates the production Native material registry with the canonical
    /// builtin response catalog already installed. This removes the old test
    /// requirement to write SceneRegistry, material rows, and execution tables
    /// independently before the first extracted frame.
    #[must_use]
    pub fn new_with_builtin_execution(capacity: u32) -> Self {
        let mut registry = MaterialRegistry::new(capacity);
        let builtin_execution = MaterialExecutionTables::builtin_with_mask_blends();
        registry
            .install_execution_tables(&builtin_execution)
            .expect("canonical builtin execution catalog must validate");
        Self {
            registry,
            builtin_execution: Some(builtin_execution),
            textures: TextureSlotHeap::default(),
            held: BTreeMap::new(),
            texture_binding_version: 0,
        }
    }

    /// The bindless slot heap.
    #[must_use]
    pub const fn textures(&self) -> &TextureSlotHeap {
        &self.textures
    }

    /// The bindless slot heap, mutably (for a device-limit re-clamp at startup).
    pub fn textures_mut(&mut self) -> &mut TextureSlotHeap {
        &mut self.textures
    }

    /// Clamps the slot space to what the device's texture binding array can
    /// address.
    ///
    /// # Panics
    /// Panics while any image is resident: re-clamping under live rows would
    /// silently invalidate the slots they name. Call it at startup, before
    /// anything is published.
    pub fn set_texture_capacity(&mut self, capacity: u32) {
        self.textures.reset_with_capacity(capacity);
        self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
    }

    /// Resolves a material's authored texture references into ABI rows.
    ///
    /// Each request acquires a bindless slot for its image (reference counted, so
    /// two materials sharing an image share a slot) or names the reserved
    /// fallback for its semantic. The returned rows are what the material's
    /// record carries, and `semantic` is the ABI's own number for the slot — this
    /// layer never interprets it.
    ///
    /// Resolution is transactional: if any authored image cannot acquire a
    /// slot, newly acquired leases are rolled back and the currently published
    /// rows/leases remain unchanged. Successful acquisitions are tracked with
    /// their slot generations; prepare reconciles them with final published rows.
    pub fn resolve_textures(
        &mut self,
        index: u32,
        requests: impl IntoIterator<Item = (u32, TextureRequest)>,
    ) -> Result<Vec<prism_render_material::GpuMaterialTexture>, TextureResolutionError> {
        // Resolve against a private heap snapshot. This releases the old
        // material leases before reacquiring slots, while preserving the
        // published heap and leases if any request exceeds capacity.
        let requests: Vec<_> = requests.into_iter().collect();
        let requested_images: BTreeSet<_> = requests
            .iter()
            .filter_map(|(_, request)| match request {
                TextureRequest::Image(image) => Some(*image),
                TextureRequest::Fallback(_) => None,
            })
            .collect();
        let previous = self.held.get(&index).cloned().unwrap_or_default();
        let mut heap = self.textures.clone();
        // Keep omitted old images resident until the replacement is known to
        // fit. Releasing only images that are being re-resolved both preserves
        // the published material on overflow and advances a same-image slot
        // generation for stale-row rejection.
        for held in &previous {
            if requested_images.contains(&held.image) && heap.slot_of(held.image) == Some(held.slot)
            {
                heap.release(held.image);
            }
        }
        let mut rows = Vec::new();
        let mut images: Vec<HeldTexture> = Vec::new();
        for (semantic, request) in requests {
            let slot = match request {
                TextureRequest::Image(image) => {
                    let slot = match heap.acquire(image) {
                        Ok(slot) => slot,
                        Err(TextureHeapAcquireError::CapacityExhausted) => {
                            return Err(TextureResolutionError::CapacityExhausted {
                                material_index: index,
                                semantic,
                            });
                        }
                    };
                    images.push(HeldTexture { image, slot });
                    slot
                }
                TextureRequest::Fallback(fallback) => TextureSlotHeap::fallback_slot(fallback),
            };
            // The sampler heap is bound in parallel with the texture heap, so a
            // texture's sampler sits at the same bindless slot. The constructor
            // also explicitly disables virtual-texture metadata for this legacy
            // image request, keeping the ABI zero-initialized instead of relying
            // on a struct literal that can drift when the ABI grows.
            rows.push(prism_render_material::GpuMaterialTexture::non_virtual(
                slot.index,
                slot.generation,
                semantic,
                slot.index,
            ));
        }
        for held in &previous {
            if !requested_images.contains(&held.image)
                && heap.slot_of(held.image) == Some(held.slot)
            {
                heap.release(held.image);
            }
        }
        self.textures = heap;
        self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        self.held.remove(&index);
        if !images.is_empty() {
            self.held.insert(index, images);
        }
        Ok(rows)
    }

    /// The bindless slot an image currently occupies, if it is resident.
    #[must_use]
    pub fn texture_slot(&self, image: AssetId<Image>) -> Option<BindlessSlot> {
        self.textures.slot_of(image)
    }

    /// Images actually referenced by final published rows with a live held
    /// lease. Unpublished acquisitions, retired rows and stale generations do
    /// not create streaming demand. This reads only textured material records
    /// and never drains the registry's buffer-upload dirty queue.
    #[must_use]
    pub fn referenced_texture_images(&self) -> BTreeMap<AssetId<Image>, BindlessSlot> {
        let mut images = BTreeMap::new();
        for (&index, held) in &self.held {
            let Some(record) = self.registry.record_at(index) else {
                continue;
            };
            for lease in held {
                if record.textures.iter().any(|row| {
                    row.index == lease.slot.index && row.generation == lease.slot.generation
                }) && self.textures.slot_of(lease.image) == Some(lease.slot)
                {
                    images.insert(lease.image, lease.slot);
                }
            }
        }
        images
    }

    /// Drops every tracked slot reference `index` holds, without releasing a
    /// replacement residency with a different generation. Returns the number
    /// of tracked leases removed, including leases that are already stale.
    pub fn release_textures(&mut self, index: u32) -> usize {
        let Some(images) = self.held.remove(&index) else {
            return 0;
        };
        for held in &images {
            if self.textures.slot_of(held.image) == Some(held.slot) {
                self.textures.release(held.image);
            }
        }
        if !images.is_empty() {
            self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        }
        images.len()
    }

    /// Reconcile ownership with the final published texture rows, rather than
    /// relying on observing an intermediate retirement. A dirty index can be
    /// retired, reclaimed and republished before a prepare pass sees it.
    fn reconcile_textures(&mut self, index: u32) {
        let Some(held) = self.held.remove(&index) else {
            return;
        };
        let mut referenced = BTreeMap::<(u32, u32), usize>::new();
        if let Some(record) = self.registry.record_at(index) {
            for texture in &record.textures {
                *referenced
                    .entry((texture.index, texture.generation))
                    .or_default() += 1;
            }
        }
        let mut retained = Vec::new();
        let mut residency_changed = false;
        for lease in held {
            if self.textures.slot_of(lease.image) != Some(lease.slot) {
                continue;
            }
            let remaining = referenced.get_mut(&(lease.slot.index, lease.slot.generation));
            if let Some(remaining) = remaining.filter(|count| **count != 0) {
                *remaining -= 1;
                retained.push(lease);
            } else {
                residency_changed |= self.textures.release(lease.image);
            }
        }
        if residency_changed {
            // Reconciliation is also the normal retirement path. A released
            // slot changes both the sampled view and its generation table, so
            // binding consumers must rebuild even when no explicit resolve or
            // release_textures call happened.
            self.texture_binding_version = self.texture_binding_version.wrapping_add(1);
        }
        if !retained.is_empty() {
            self.held.insert(index, retained);
        }
    }

    /// Publishes (or replaces) a material record, marking its row dirty.
    ///
    /// # Errors
    /// Propagates the registry's own errors: a handle outside the capacity, or a
    /// revision that would move backwards.
    pub fn publish(&mut self, record: MaterialRecord) -> Result<(), MaterialRegistryError> {
        self.registry.publish(record)
    }

    /// The registry's current epoch and live-material count.
    #[must_use]
    pub fn snapshot(&self) -> MaterialSnapshot {
        self.registry.snapshot()
    }

    /// The registry itself, for systems that only read it.
    #[must_use]
    pub fn registry(&self) -> &MaterialRegistry {
        &self.registry
    }

    /// The registry itself, mutably.
    pub fn registry_mut(&mut self) -> &mut MaterialRegistry {
        &mut self.registry
    }

    /// Resolves only builtin plans from the production catalog. A missing
    /// result is intentional for registered or malformed plans and must remain
    /// fail-closed at queue admission.
    #[must_use]
    pub fn builtin_execution_binding(
        &self,
        plan: &ShadingPlan,
    ) -> Option<MaterialExecutionBinding> {
        self.registry
            .execution_tables()
            .and_then(|tables| tables.binding_for_plan(plan))
    }

    /// Returns the canonical builtin table used when the scene has no custom
    /// execution publication. A scene-owned table is never silently merged into
    /// this catalog; its plan indices and registered link are one generation.
    fn builtin_execution_tables(&self) -> Option<&MaterialExecutionTables> {
        self.builtin_execution.as_ref()
    }

    /// Publication number for the current authored texture heap state.
    #[must_use]
    pub const fn texture_binding_version(&self) -> u64 {
        self.texture_binding_version
    }
}

impl Resource for RenderMaterials {}

/// Reconciles the authoritative main-world material snapshot into both render
/// registries before queue and material upload admission. Stable handles are
/// adopted verbatim; a generation replacement is retired and remains
/// unavailable until the old GPU references have completed.
pub(crate) fn reconcile_native_materials<I>(
    extracted: Res<ExtractedNativeMaterials>,
    mut materials: ResMut<RenderMaterials>,
    mut scene_registry: ResMut<SceneRegistry>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if !extracted.changed && !extracted.execution_tables_changed {
        return;
    }

    if extracted.execution_tables_changed {
        let handles: Vec<_> = (1..materials.registry().capacity())
            .filter_map(|index| {
                materials
                    .registry()
                    .record_at(index)
                    .map(|record| record.handle)
            })
            .collect();
        for handle in handles {
            materials
                .registry_mut()
                .clear_execution_binding(handle)
                .unwrap_or_else(|error| panic!("native execution binding clear failed: {error:?}"));
        }
        let tables = extracted
            .execution_tables
            .clone()
            .or_else(|| materials.builtin_execution_tables().cloned())
            .unwrap_or_else(|| {
                panic!("native material execution tables changed without a verified replacement")
            });
        materials
            .registry_mut()
            .install_execution_tables(&tables)
            .unwrap_or_else(|error| panic!("native execution table install failed: {error:?}"));
    }

    let incoming: BTreeMap<_, _> = extracted
        .records
        .iter()
        .map(|record| (record.handle.index, record))
        .collect();
    let incoming_dynamic: BTreeMap<_, _> = extracted.dynamic.iter().copied().collect();
    let completed = GpuCompletionValue(runtime.runtime().completed_value());
    materials.registry_mut().reclaim_completed(completed);

    // Retire rows absent from the authoritative snapshot, or rows whose stable
    // generation changed. Never replace a live generation in place.
    let current: Vec<_> = (1..materials.registry().capacity())
        .filter_map(|index| materials.registry().record_at(index).cloned())
        .collect();
    let mut retired = BTreeSet::new();
    for record in current {
        let replacement = incoming.get(&record.handle.index);
        if replacement.is_some_and(|next| next.handle == record.handle) {
            continue;
        }
        materials.release_textures(record.handle.index);
        materials
            .registry_mut()
            .retire(
                record.handle,
                GpuCompletionValue(runtime.runtime().next_completion_value()),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "native material retirement failed for {:?}: {error:?}",
                    record.handle
                )
            });
        retired.insert(record.handle.index);
        scene_registry.remove_material(record.handle);
    }

    // Publish only records whose exact handle can be admitted this frame. A
    // generation replacement that is still retiring stays fail-closed until a
    // later extraction observes completed GPU work.
    for record in extracted.records.iter().cloned() {
        if retired.contains(&record.handle.index) {
            continue;
        }
        match materials.registry().get(record.handle) {
            Some(current) if current == &record => {}
            Some(current) if record.revision > current.revision => {
                materials
                    .publish(record.clone())
                    .unwrap_or_else(|error| panic!("native material republish failed: {error:?}"));
            }
            Some(current) => panic!(
                "native material revision did not advance for {:?}: current={}, incoming={}",
                record.handle, current.revision, record.revision
            ),
            None => {
                if materials
                    .registry()
                    .record_at(record.handle.index)
                    .is_some()
                {
                    continue;
                }
                materials
                    .registry_mut()
                    .adopt_and_publish(record.clone())
                    .unwrap_or_else(|error| panic!("native material adoption failed: {error:?}"));
            }
        }
        if let Some(binding) = materials.builtin_execution_binding(&record.response_plan) {
            materials
                .registry_mut()
                .set_execution_binding(record.handle, binding)
                .unwrap_or_else(|error| panic!("builtin execution binding failed: {error:?}"));
        } else {
            materials
                .registry_mut()
                .clear_execution_binding(record.handle)
                .unwrap_or_else(|error| panic!("native execution binding clear failed: {error:?}"));
        }
        let dynamic = incoming_dynamic
            .get(&record.handle)
            .copied()
            .unwrap_or(DynamicChannels::NONE);
        materials
            .registry_mut()
            .set_dynamic_channels(record.handle, dynamic)
            .unwrap_or_else(|error| panic!("native dynamic channel publication failed: {error:?}"));
        scene_registry.insert_material(record);
    }
}

/// Publishes the bindless authored-image arrays after GPU images and material
/// rows have been prepared. Non-bindless devices intentionally publish no
/// direct heap; their shader path must consume a valid portable VT row.
pub(crate) fn prepare_native_texture_bindings<I>(
    mut bindings: ResMut<NativeMaterialTextureBindings>,
    materials: Res<RenderMaterials>,
    images: Res<crate::render_asset::RenderAssets<crate::GpuImage>>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    bindings.prepare(
        runtime.runtime().device_handle(),
        runtime.runtime().queue(),
        runtime.runtime().device_generation(),
        &materials,
        &images,
    );
}

/// The most recent upload pass's report.
#[derive(Resource, Default, Debug)]
pub struct MaterialUploadStats {
    /// What the last pass did.
    pub last: MaterialSyncReport,
    /// Whether the last VT prepare completed the publication/refresh gate.
    pub native_vt_ready: bool,
    /// Missing production owner/scheduler/sources is distinct from upload wait.
    pub native_vt_missing_resources: bool,
    /// Scheduler diagnostics from the last prepare; never silently discarded.
    pub native_vt_errors: Vec<NativeVtError>,
    /// Whether the registry execution-table version is published to GPU.
    pub execution_tables_ready: bool,
    /// Why execution-table publication was refused, if any.
    pub execution_tables_error: Option<ExecutionTableSyncError>,
}

/// Packs the registry's dirty materials into the ABI buffers.
///
/// Runs in [`RenderSystems::PrepareAssets`](crate::RenderSystems::PrepareAssets),
/// before Queue validates material identities and revisions against the header
/// mirror, and before any draw binds the buffers. Only dirty rows are touched,
/// so a frame where no material changed uploads nothing; a dirty index with no
/// record is a retirement, and its row is rewritten as an inactive tombstone
/// while its word run goes back to the heap.
pub(crate) fn sync_material_buffers<I>(
    mut buffers: ResMut<MaterialParameterBuffers>,
    mut execution_buffers: ResMut<NativeExecutionTableBuffers>,
    mut materials: ResMut<RenderMaterials>,
    mut stats: ResMut<MaterialUploadStats>,
    runtime: Res<RenderRuntimeResource<I>>,
    (config, scheduler, owner, sources): (
        Option<Res<crate::config::RenderAppConfig>>,
        Option<Res<NativeVtScheduler>>,
        Option<Res<NativeMaterialVt<I::Device>>>,
        Option<Res<NativeVtSources>>,
    ),
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if !crate::plugin::native_vt_resources_ready(
        config.is_some(),
        scheduler.as_deref(),
        owner.is_some(),
        sources.is_some(),
    ) {
        // In particular, do not drain dirty rows while revocation/publication
        // has failed: the next successful prepare must upload those same rows.
        return;
    }
    let runtime_ref = runtime.runtime();
    buffers.set_vt_publication_generation(
        owner
            .as_ref()
            .map_or(0, |native_vt| native_vt.publication_generation()),
    );
    match execution_buffers.sync_registry(
        runtime_ref.device_handle(),
        runtime_ref.queue(),
        materials.registry(),
    ) {
        Ok(_) => {
            stats.execution_tables_ready = execution_buffers.execution_table_version().is_some();
            stats.execution_tables_error = None;
        }
        Err(ExecutionTableSyncError::TablesUnavailable) => {
            stats.execution_tables_ready = false;
            stats.execution_tables_error = None;
        }
        Err(ExecutionTableSyncError::RegisteredResponseSourceUnavailable) => {
            stats.execution_tables_ready = false;
            stats.execution_tables_error =
                Some(ExecutionTableSyncError::RegisteredResponseSourceUnavailable);
            // Do not upload headers for a new table generation when its
            // registered response cannot be linked. That would pair material
            // rows with the previous execution snapshot and admit a mixed ABI.
            buffers.set_execution_table_version(execution_buffers.execution_table_version());
            return;
        }
    }
    // Execution tables have their own completion-published generation. They
    // may advance while all material bytes stay quiet, so keep the logical
    // material snapshot stamp paired with the latest completed table here.
    buffers.set_execution_table_version(execution_buffers.execution_table_version());
    if let Some(report) = sync_dirty_material_buffers(
        &mut buffers,
        &mut materials,
        runtime_ref.device(),
        runtime_ref.queue(),
    ) {
        stats.last = report;
    }
}

/// The production upload path, shared with headless regression tests without
/// constructing an ECS runtime or maintaining a second copy of runtime state.
fn sync_dirty_material_buffers<D: RenderDevice, Q: RenderQueue>(
    buffers: &mut MaterialParameterBuffers,
    materials: &mut RenderMaterials,
    device: &D,
    queue: &Q,
) -> Option<MaterialSyncReport> {
    let dirty = materials.registry_mut().take_dirty();
    if dirty.is_empty() {
        return None;
    }

    for &index in &dirty {
        materials.reconcile_textures(index);
    }
    let mut report =
        buffers.sync_registry(device, queue, materials.registry(), dirty.iter().copied());
    for index in dirty {
        if index == FALLBACK_MATERIAL_INDEX || materials.registry().record_at(index).is_some() {
            continue;
        }
        // The registry is authoritatively absent, but one or more generations
        // may have been published and retired since the last upload. Remove
        // the generation actually present in the GPU mirror, not a newer
        // registry generation which never reached that mirror.
        let Some(generation) = buffers.header(index).map(|row| row.generation) else {
            continue;
        };
        if buffers.remove(queue, index, generation) {
            report.headers_written += 1;
        }
    }
    Some(report)
}

#[cfg(test)]
#[path = "runtime_tail_tests.rs"]
mod runtime_tail_tests;
