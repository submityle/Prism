use bevy_asset::AssetId;
use bevy_ecs::{prelude::*, system::SystemParam};
use bevy_pbr::StandardMaterial;
use bevy_render::render_resource::{BindGroup, BindGroupLayout, Buffer};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{MaterialSnapshot, FALLBACK_MATERIAL_HANDLE};

use super::{
    bindings::MaterialBindGroup, buffers::MaterialGpuBuffers, runtime::RenderMaterialRegistry,
};

/// Read-only access shared by every Material ABI consumer.
#[derive(SystemParam)]
pub struct MaterialReader<'w> {
    runtime: Res<'w, RenderMaterialRegistry>,
    buffers: Res<'w, MaterialGpuBuffers>,
    bindings: Res<'w, MaterialBindGroup>,
}

impl MaterialReader<'_> {
    pub fn snapshot(&self) -> MaterialSnapshot {
        self.runtime.registry.snapshot()
    }

    pub fn handle(&self, asset: AssetId<StandardMaterial>) -> GenerationalHandle {
        self.runtime
            .material_handle(asset)
            .unwrap_or(FALLBACK_MATERIAL_HANDLE)
    }

    pub fn layout(&self) -> &BindGroupLayout {
        &self.bindings.layout
    }

    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bindings.bind_group.as_ref()
    }

    pub fn buffers(&self) -> Option<MaterialBufferBindings<'_>> {
        let (headers, parameters, textures) = self.buffers.buffers()?;
        Some(MaterialBufferBindings {
            headers,
            parameters,
            textures,
        })
    }
}

pub struct MaterialBufferBindings<'a> {
    pub headers: &'a Buffer,
    pub parameters: &'a Buffer,
    pub textures: &'a Buffer,
}
