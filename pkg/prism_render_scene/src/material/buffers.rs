use super::runtime::RenderMaterialRegistry;
use alloc::sync::Arc;
use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::render_resource::{AtomicPod, AtomicSparseBufferVec, Buffer, BufferUsages};
use prism_render_material::{
    fallback_material_header, inactive_material_header, GpuMaterialHeader, GpuMaterialTexture,
    GpuSurfaceParameters, MAX_MATERIAL_TEXTURES,
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialHeaderRow(pub GpuMaterialHeader);
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialParametersRow(pub GpuSurfaceParameters);
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialTextureRow(pub GpuMaterialTexture);
bevy_render::impl_atomic_pod!(MaterialHeaderRow, MaterialHeaderRowBlob);
bevy_render::impl_atomic_pod!(MaterialParametersRow, MaterialParametersRowBlob);
bevy_render::impl_atomic_pod!(MaterialTextureRow, MaterialTextureRowBlob);

#[derive(Resource)]
pub(crate) struct MaterialGpuBuffers {
    pub headers: AtomicSparseBufferVec<MaterialHeaderRow>,
    pub parameters: AtomicSparseBufferVec<MaterialParametersRow>,
    pub textures: AtomicSparseBufferVec<MaterialTextureRow>,
}

impl FromWorld for MaterialGpuBuffers {
    fn from_world(_: &mut World) -> Self {
        Self {
            headers: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism material headers"),
            ),
            parameters: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism material parameters"),
            ),
            textures: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                Arc::from("prism material textures"),
            ),
        }
    }
}

impl MaterialGpuBuffers {
    pub fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer)> {
        Some((
            self.headers.buffer()?,
            self.parameters.buffer()?,
            self.textures.buffer()?,
        ))
    }
    pub fn apply_dirty(&mut self, runtime: &mut RenderMaterialRegistry) -> (u32, u64) {
        let dirty = runtime.registry.take_dirty();
        for &index in &dirty {
            let generation = runtime.registry.generation_at(index).unwrap_or(0);
            if let Some(record) = runtime.registry.record_at(index) {
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(record.header(
                        index,
                        index * MAX_MATERIAL_TEXTURES as u32,
                        // No serialized closure-graph blob buffer yet; RT/deferred
                        // consumers fall back to the packed axes until it lands.
                        0,
                        runtime.registry.snapshot().epoch,
                    )),
                );
                self.parameters
                    .grow_and_set(index, MaterialParametersRow(record.surface));
                for (slot, texture) in record.fixed_texture_rows().into_iter().enumerate() {
                    self.textures.grow_and_set(
                        index * MAX_MATERIAL_TEXTURES as u32 + slot as u32,
                        MaterialTextureRow(texture),
                    );
                }
            } else if index == 0 {
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(fallback_material_header(runtime.registry.snapshot().epoch)),
                );
                self.parameters.grow_and_set(
                    index,
                    MaterialParametersRow(GpuSurfaceParameters::default()),
                );
                for slot in 0..MAX_MATERIAL_TEXTURES as u32 {
                    self.textures
                        .grow_and_set(slot, MaterialTextureRow::default());
                }
            } else {
                self.headers.grow_and_set(
                    index,
                    MaterialHeaderRow(inactive_material_header(generation)),
                );
                self.parameters.grow_and_set(
                    index,
                    MaterialParametersRow(GpuSurfaceParameters::default()),
                );
                for slot in 0..MAX_MATERIAL_TEXTURES as u32 {
                    self.textures.grow_and_set(
                        index * MAX_MATERIAL_TEXTURES as u32 + slot,
                        MaterialTextureRow::default(),
                    );
                }
            }
        }
        let bytes = dirty.len() as u64
            * (size_of::<GpuMaterialHeader>()
                + size_of::<GpuSurfaceParameters>()
                + MAX_MATERIAL_TEXTURES * size_of::<GpuMaterialTexture>()) as u64;
        (dirty.len() as u32, bytes)
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::world::{FromWorld, World};
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::{Illumination, MaterialRenderClass};

    use super::*;

    #[test]
    fn dirty_rows_publish_fallback_live_and_retired_generations() {
        let mut world = World::new();
        let mut buffers = MaterialGpuBuffers::from_world(&mut world);
        let mut runtime = RenderMaterialRegistry::default();
        let handle = runtime.registry.allocate().unwrap();
        runtime
            .registry
            .publish(prism_render_material::MaterialRecord {
                handle,
                revision: 1,
                domain: prism_render_material::MaterialDomain::Surface,
                render_class: MaterialRenderClass::Opaque,
                illumination: Illumination::Lit,
                features: Default::default(),
                closure_mask: 1,
                surface: GpuSurfaceParameters::default(),
                textures: vec![],
                custom_program: None,
            })
            .unwrap();

        assert_eq!(buffers.apply_dirty(&mut runtime).0, 2);
        runtime
            .registry
            .retire(handle, GpuCompletionValue(1))
            .unwrap();
        assert_eq!(buffers.apply_dirty(&mut runtime).0, 1);
        assert_eq!(
            runtime.registry.generation_at(handle.index),
            Some(handle.generation)
        );
    }
}
