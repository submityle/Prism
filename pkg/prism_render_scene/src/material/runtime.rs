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

#[derive(Resource)]
pub(crate) struct RenderMaterialRegistry {
    pub registry: MaterialRegistry,
    pub assets: HashMap<AssetId<StandardMaterial>, GenerationalHandle>,
    pub revisions: HashMap<AssetId<StandardMaterial>, u32>,
    pub dirty_assets: HashSet<AssetId<StandardMaterial>>,
    textures: HashMap<AssetId<Image>, GenerationalHandle>,
    next_texture: u32,
}

impl Default for RenderMaterialRegistry {
    fn default() -> Self {
        Self {
            registry: MaterialRegistry::new(1 << 20),
            assets: HashMap::default(),
            revisions: HashMap::default(),
            dirty_assets: HashSet::default(),
            textures: HashMap::default(),
            next_texture: 1,
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
            if allocated {
                self.assets.remove(&id);
                let _ = self.registry.cancel_allocation(handle);
            }
            return Err(error);
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
        self.dirty_assets.insert(id);
        Ok(Some(handle))
    }
    pub fn texture_resolver(&mut self) -> TextureResolver<'_> {
        TextureResolver { registry: self }
    }
}

pub(crate) struct TextureResolver<'a> {
    registry: &'a mut RenderMaterialRegistry,
}
impl StandardMaterialTextureResolver for TextureResolver<'_> {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture {
        let handle = *self.registry.textures.entry(image).or_insert_with(|| {
            let index = self.registry.next_texture;
            self.registry.next_texture += 1;
            GenerationalHandle {
                index,
                generation: 1,
            }
        });
        GpuMaterialTexture {
            index: handle.index,
            generation: handle.generation,
            semantic: semantic as u32,
            sampler_index: handle.index,
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
