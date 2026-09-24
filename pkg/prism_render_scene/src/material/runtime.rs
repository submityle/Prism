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
        // Each `resolve` call during lowering acquires a bindless slot and
        // records the image here so we can reconcile references afterwards.
        self.pending_textures.clear();
        let record = {
            let mut resolver = self.texture_resolver();
            prism_render_material::lower_standard_material(
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
        let previous = self.material_textures.insert(id, new_set).unwrap_or_default();
        for image in previous {
            self.textures.release(image);
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
        let Some(handle) = self.assets.remove(&id) else {
            return Ok(None);
        };
        self.registry.retire(handle, completion)?;
        self.revisions.remove(&id);
        if let Some(images) = self.material_textures.remove(&id) {
            for image in images {
                self.textures.release(image);
            }
        }
        self.dirty_assets.insert(id);
        Ok(Some(handle))
    }
    pub fn texture_resolver(&mut self) -> TextureResolver<'_> {
        TextureResolver { registry: self }
    }

    /// Current bindless texture-heap occupancy, surfaced for diagnostics.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Folded into PrismMaterialDiagnostics by the upcoming bindless upload slice."
        )
    )]
    pub fn texture_heap_stats(&self) -> BindlessHeapStats {
        self.textures.stats()
    }
}

pub(crate) struct TextureResolver<'a> {
    registry: &'a mut RenderMaterialRegistry,
}
impl StandardMaterialTextureResolver for TextureResolver<'_> {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture {
        // Acquire (or reference) the bindless slot and remember the image so the
        // owning material releases exactly what it took on republish/retire.
        let slot = self.registry.textures.acquire(image);
        self.registry.pending_textures.push(image);
        GpuMaterialTexture {
            index: slot.index,
            generation: slot.generation,
            semantic: semantic as u32,
            // Sampler binding-array wiring lands in a later slice; every texture
            // shares the default sampler (index 0) until then.
            sampler_index: 0,
        }
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
}
