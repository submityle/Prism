use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::storage_buffer_read_only_sized, BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, BufferId, ShaderStages},
    renderer::RenderDevice,
};
use core::num::NonZero;
use prism_render_material::{GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters};

use super::{buffers::MaterialGpuBuffers, runtime::RenderMaterialRegistry};

/// Shared binding contract for raster, visibility, shadows, GI, ray, and
/// offline consumers of the unified Material ABI.
#[derive(Resource)]
pub struct MaterialBindGroup {
    pub layout: BindGroupLayout,
    pub layout_descriptor: BindGroupLayoutDescriptor,
    pub bind_group: Option<BindGroup>,
    buffer_ids: Option<[BufferId; 3]>,
    buffer_version: u32,
}

impl FromWorld for MaterialBindGroup {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuMaterialHeader>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuSurfaceParameters>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuMaterialTexture>() as u64),
                ),
            ),
        );
        let layout_descriptor = BindGroupLayoutDescriptor::new("prism materials", &entries);
        Self {
            layout: device.create_bind_group_layout("prism materials", &entries),
            layout_descriptor,
            bind_group: None,
            buffer_ids: None,
            buffer_version: 0,
        }
    }
}

impl MaterialBindGroup {
    pub(crate) fn prepare(
        &mut self,
        device: &RenderDevice,
        buffers: &MaterialGpuBuffers,
        runtime: &RenderMaterialRegistry,
    ) {
        let Some((headers, parameters, textures)) = buffers.buffers() else {
            return;
        };
        let ids = [headers.id(), parameters.id(), textures.id()];
        let version = runtime.registry.snapshot().buffer_version;
        if self.buffer_ids == Some(ids) && self.buffer_version == version {
            return;
        }
        self.bind_group = Some(device.create_bind_group(
            "prism materials",
            &self.layout,
            &BindGroupEntries::sequential((
                headers.as_entire_binding(),
                parameters.as_entire_binding(),
                textures.as_entire_binding(),
            )),
        ));
        self.buffer_ids = Some(ids);
        self.buffer_version = version;
    }
}
