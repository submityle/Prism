use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::{HashMap, HashSet};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{
    GpuMaterialTexture, MaterialRegistry, MaterialRegistryError, StandardMaterialTextureResolver,
    TextureSemantic,
};

use super::texture_heap::{BindlessHeapStats, BindlessTextureHeap};
use super::vt_source::{MaterialVtSourceLookup, MaterialVtSourceRegistry};

#[derive(Resource)]
pub(crate) struct RenderMaterialRegistry {
    pub registry: MaterialRegistry,
    pub assets: HashMap<AssetId<StandardMaterial>, GenerationalHandle>,
    pub revisions: HashMap<AssetId<StandardMaterial>, u32>,
    pub dirty_assets: HashSet<AssetId<StandardMaterial>>,
    /// Bounded bindless slot allocator backing every material texture.
    textures: BindlessTextureHeap,
    /// Set of images each published material currently holds a reference to, so
    /// republishing or retiring a material releases exactly the slots it owned.
    material_textures: HashMap<AssetId<StandardMaterial>, Vec<AssetId<Image>>>,
    /// Scratch list the active [`TextureResolver`] pushes acquired images into
    /// while lowering a single material. Drained by `publish_asset`.
    pending_textures: Vec<AssetId<Image>>,
}

impl Default for RenderMaterialRegistry {
    fn default() -> Self {
        Self {
            registry: MaterialRegistry::new(1 << 20),
            assets: HashMap::default(),
            revisions: HashMap::default(),
            dirty_assets: HashSet::default(),
            textures: BindlessTextureHeap::default(),
            material_textures: HashMap::default(),
            pending_textures: Vec::new(),
        }
    }
}

impl RenderMaterialRegistry {
    pub fn material_handle(&self, asset: AssetId<StandardMaterial>) -> Option<GenerationalHandle> {
        self.assets.get(&asset).copied()
    }
    pub fn publish_asset(
        &mut self,
        id: AssetId<StandardMaterial>,
        material: &StandardMaterial,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> Result<GenerationalHandle, MaterialRegistryError> {
        self.publish_asset_with_sources(id, material, completion, None)
    }

    pub fn publish_asset_with_sources(
        &mut self,
        id: AssetId<StandardMaterial>,
        material: &StandardMaterial,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
        sources: Option<&MaterialVtSourceRegistry>,
    ) -> Result<GenerationalHandle, MaterialRegistryError> {
        let (handle, allocated) = match self.assets.get(&id).copied() {
            Some(handle) => (handle, false),
            None => {
                let handle = self
                    .registry
                    .allocate()
                    .map_err(|_| MaterialRegistryError::CapacityExceeded)?;
                self.assets.insert(id, handle);
                (handle, true)
            }
        };
        let revision = self.revisions.get(&id).copied().unwrap_or(0) + 1;
        // Each `resolve` call during the OpenPBR import acquires a bindless slot and
        // records the image here so we can reconcile references afterwards.
        self.pending_textures.clear();
        let record = {
            let mut resolver = self.texture_resolver_with_sources(sources);
            prism_render_material::import_standard_material(
                handle,
                revision,
                material,
                &mut resolver,
            )
        };
        if let Err(error) = self.registry.publish(record) {
            // Roll back the slots this attempt acquired so a failed publish
            // never leaks references.
            for image in core::mem::take(&mut self.pending_textures) {
                self.textures.release(image);
            }
            if allocated {
                self.assets.remove(&id);
                let _ = self.registry.cancel_allocation(handle);
            }
            return Err(error);
        }
        // Success: adopt the freshly acquired texture set and release the set the
        // previous revision held. Acquiring before releasing keeps shared
        // textures resident without a transient generation bump.
        let new_set = core::mem::take(&mut self.pending_textures);
        let previous = self
            .material_textures
            .insert(id, new_set)
            .unwrap_or_default();
        for image in previous {
            // The previous material row may still be referenced by the GPU.
            // Keep its slot quarantined until the submission that could read
            // that row has completed; immediate release would allow a new
            // image to reuse the slot and create a use-after-republish.
            self.textures.retire(image, completion);
        }
        self.revisions.insert(id, revision);
        self.dirty_assets.insert(id);
        Ok(handle)
    }
    pub fn retire_asset(
        &mut self,
        id: AssetId<StandardMaterial>,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> Result<Option<GenerationalHandle>, prism_render_material::MaterialHandleError> {
        let Some(handle) = self.assets.get(&id).copied() else {
            return Ok(None);
        };
        self.registry.retire(handle, completion)?;
        self.assets.remove(&id);
        self.revisions.remove(&id);
        if let Some(images) = self.material_textures.remove(&id) {
            for image in images {
                self.textures.retire(image, completion);
            }
        }
        self.dirty_assets.insert(id);
        Ok(Some(handle))
    }
    pub fn texture_resolver(&mut self) -> TextureResolver<'_> {
        self.texture_resolver_with_sources(None)
    }

    fn texture_resolver_with_sources<'a>(
        &'a mut self,
        sources: Option<&'a MaterialVtSourceRegistry>,
    ) -> TextureResolver<'a> {
        TextureResolver {
            registry: self,
            sources,
        }
    }

    /// Current bindless texture-heap occupancy, surfaced for diagnostics and
    /// consumed by the bindless upload system when it sizes its slot arrays.
    pub fn texture_heap_stats(&self) -> BindlessHeapStats {
        self.textures.stats()
    }

    pub(crate) fn texture_generation_table(&self) -> &[u32] {
        self.textures.generation_table()
    }

    pub(crate) fn texture_generation_version(&self) -> u32 {
        self.textures.generation_version()
    }

    /// Reclaims texture slots whose GPU completion proof has arrived.
    pub fn reclaim_completed_textures(
        &mut self,
        completed: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> u32 {
        self.textures.reclaim_completed(completed)
    }

    /// High-water bound of live bindless slots. The upload path only needs to
    /// materialise `TextureView`s for slots below this value.
    pub fn texture_high_water(&self) -> u32 {
        self.textures.high_water()
    }

    /// Iterates the resident `(slot_index, image)` pairs so the upload system
    /// can place each live image at its assigned bindless slot.
    pub fn texture_slots(&self) -> impl Iterator<Item = (AssetId<Image>, u32)> + '_ {
        self.textures
            .iter_slots()
            .map(|(slot, image)| (image, slot))
    }

    /// Clamps the bindless slot space to `capacity` before any material is
    /// published, so `acquire` can never hand out a slot the bound
    /// `binding_array` cannot address. Only valid while the heap is empty.
    pub fn set_texture_capacity(&mut self, capacity: u32) {
        self.textures.reset_with_capacity(capacity);
    }
}

pub(crate) struct TextureResolver<'a> {
    registry: &'a mut RenderMaterialRegistry,
    sources: Option<&'a MaterialVtSourceRegistry>,
}
impl StandardMaterialTextureResolver for TextureResolver<'_> {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture {
        // Acquire (or reference) the bindless slot and remember the image so the
        // owning material releases exactly what it took on republish/retire.
        let slot = self.registry.textures.acquire(image);
        self.registry.pending_textures.push(image);
        // The sampler heap is bound in parallel with the texture heap, so a
        // texture's sampler lives at the same bindless slot as the texture
        // itself. Reserved slots 0/1/2 carry the default filtering sampler.
        let Some(MaterialVtSourceLookup::Available { snapshot, .. }) = self
            .sources
            .filter(|sources| sources.eligible_for_default_producer(image))
            .and_then(|sources| match sources.lookup(image) {
                MaterialVtSourceLookup::Available { identity, snapshot } => {
                    Some(MaterialVtSourceLookup::Available { identity, snapshot })
                }
                _ => None,
            })
        else {
            return GpuMaterialTexture::non_virtual(
                slot.index,
                slot.generation,
                semantic as u32,
                slot.index,
            );
        };
        let descriptor = snapshot.descriptor();
        GpuMaterialTexture::with_virtual_texture(
            slot.index,
            // `generation` guards the bindless heap slot in the shader. The
            // source revision identifies the CPU snapshot and may advance
            // while the image remains resident in the same heap slot; using
            // that revision here would make an otherwise live VT row stale.
            slot.generation,
            semantic as u32,
            slot.index,
            descriptor.texture_id(),
            descriptor.width(),
            descriptor.height(),
            descriptor.page_size(),
        )
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct PrismMaterialDiagnostics {
    pub active: u32,
    pub created: u32,
    pub updated: u32,
    pub retired: u32,
    pub reclaimed: u32,
    pub errors: u32,
    pub uploaded_rows: u32,
    pub uploaded_bytes: u64,
    pub epoch: u64,
    pub buffer_version: u32,
    pub buffer_rebuilds: u32,
    /// High-water mark of live bindless texture slots (0 when non-bindless).
    pub texture_slots_live: u32,
    /// Total addressable bindless texture slots for the active device.
    pub texture_slots_capacity: u32,
    /// Distinct images currently resident in the bindless heap.
    pub texture_images_resident: u32,
    /// Cumulative bindless `acquire` calls that hit the capacity ceiling and
    /// fell back to the white slot.
    pub texture_overflow: u32,
}
