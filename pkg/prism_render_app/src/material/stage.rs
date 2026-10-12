//! The material stage: the render world's material registry, and the system
//! that keeps the ABI buffers in step with it.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_asset::AssetId;
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_image::Image;
use prism_render_driver::{FrameSync, Instance};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{
    MaterialRecord, MaterialRegistry, MaterialRegistryError, MaterialSnapshot,
    fallback_material_record,
};

/// The index of the permanent fallback material (slot zero).
pub const FALLBACK_MATERIAL_INDEX: u32 = 0;

use crate::runtime_resource::RenderRuntimeResource;

use super::buffers::{MaterialParameterBuffers, MaterialSyncReport};
use super::texture_heap::{BindlessSlot, TextureFallback, TextureSlotHeap};

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
    /// The bindless slot space this registry assigns material textures from.
    textures: TextureSlotHeap,
    /// The images each material currently holds a slot reference for, so a
    /// republish releases exactly what it took and a retirement releases the
    /// rest.
    held: BTreeMap<u32, Vec<AssetId<Image>>>,
}

impl RenderMaterials {
    /// A registry with room for `capacity` materials.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            registry: MaterialRegistry::new(capacity),
            textures: TextureSlotHeap::default(),
            held: BTreeMap::new(),
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
    }

    /// Resolves a material's authored texture references into ABI rows.
    ///
    /// Each request acquires a bindless slot for its image (reference counted, so
    /// two materials sharing an image share a slot) or names the reserved
    /// fallback for its semantic. The returned rows are what the material's
    /// record carries, and `semantic` is the ABI's own number for the slot — this
    /// layer never interprets it.
    ///
    /// The registry remembers every image a resolution touched, so republishing
    /// the material releases exactly those references.
    pub fn resolve_textures(
        &mut self,
        index: u32,
        requests: impl IntoIterator<Item = (u32, TextureRequest)>,
    ) -> Vec<prism_render_material::GpuMaterialTexture> {
        self.release_textures(index);
        let mut rows = Vec::new();
        let mut images = Vec::new();
        for (semantic, request) in requests {
            let slot = match request {
                TextureRequest::Image(image) => {
                    images.push(image);
                    self.textures.acquire(image)
                }
                TextureRequest::Fallback(fallback) => TextureSlotHeap::fallback_slot(fallback),
            };
            rows.push(prism_render_material::GpuMaterialTexture {
                index: slot.index,
                generation: slot.generation,
                semantic,
                // The sampler heap is bound in parallel with the texture heap, so
                // a texture's sampler sits at the same bindless slot.
                sampler_index: slot.index,
            });
        }
        if !images.is_empty() {
            self.held.insert(index, images);
        }
        rows
    }

    /// The bindless slot an image currently occupies, if it is resident.
    #[must_use]
    pub fn texture_slot(&self, image: AssetId<Image>) -> Option<BindlessSlot> {
        self.textures.slot_of(image)
    }

    /// Drops every slot reference `index` holds. Called when a material is
    /// republished (its new rows are resolved first) and when it retires.
    pub fn release_textures(&mut self, index: u32) -> usize {
        let Some(images) = self.held.remove(&index) else {
            return 0;
        };
        for image in &images {
            self.textures.release(*image);
        }
        images.len()
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
}

impl Resource for RenderMaterials {}

/// The most recent upload pass's report.
#[derive(Resource, Default, Debug)]
pub struct MaterialUploadStats {
    /// What the last pass did.
    pub last: MaterialSyncReport,
}

/// Packs the registry's dirty materials into the ABI buffers.
///
/// Runs in [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources),
/// before anything reads the buffers or binds them. Only dirty rows are touched,
/// so a frame where no material changed uploads nothing; a dirty index with no
/// record is a retirement, and its row is rewritten as an inactive placeholder
/// while its word run goes back to the heap.
pub(crate) fn sync_material_buffers<I>(
    mut buffers: ResMut<MaterialParameterBuffers>,
    mut materials: ResMut<RenderMaterials>,
    mut stats: ResMut<MaterialUploadStats>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let dirty = materials.registry_mut().take_dirty();
    if dirty.is_empty() {
        return;
    }
    let epoch = materials.registry().snapshot().epoch;
    let device = runtime.runtime().device();
    let queue = runtime.runtime().queue();

    let mut records = Vec::<(u32, MaterialRecord)>::new();
    let mut retired = Vec::<u32>::new();
    for index in dirty {
        // Slot zero is the permanent fallback, not a scene material: it has no
        // published record, and a shader that reads a stale handle must find a
        // valid lit surface there rather than a retired row.
        if index == FALLBACK_MATERIAL_INDEX {
            let handle = GenerationalHandle::new(FALLBACK_MATERIAL_INDEX, 0);
            records.push((index, fallback_material_record(handle, epoch)));
            continue;
        }
        match materials.registry().record_at(index) {
            Some(record) => records.push((index, record.clone())),
            None => retired.push(index),
        }
    }

    let mut report = buffers.sync(device, queue, records, epoch);
    for index in retired {
        let generation = materials.registry().generation_at(index).unwrap_or(0);
        buffers.remove(queue, index, generation);
        // The material is gone, so the bindless slots its rows named go back to
        // the heap — and their generations bump, which is what stops a stale row
        // from sampling whatever lands in the slot next.
        materials.release_textures(index);
        report.headers_written += 1;
    }
    stats.last = report;
}

