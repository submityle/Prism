use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{binding_types::storage_buffer_read_only, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, ShaderStages},
    renderer::RenderDevice,
};

use super::{
    rows::{RenderGpuSceneBounds, RenderGpuSceneInstance, RenderGpuSceneTransform},
    storage::GpuSceneBuffers,
};

/// Shared bind group containing all stable GPU Scene tables.
#[derive(Resource)]
pub struct GpuSceneBindGroup {
    pub layout: BindGroupLayout,
    pub layout_descriptor: BindGroupLayoutDescriptor,
    pub bind_group: Option<BindGroup>,
    buffer_ids: Option<[bevy_render::render_resource::BufferId; 4]>,
}

impl FromWorld for GpuSceneBindGroup {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only::<RenderGpuSceneInstance>(false),
                storage_buffer_read_only::<RenderGpuSceneTransform>(false),
                storage_buffer_read_only::<RenderGpuSceneTransform>(false),
                storage_buffer_read_only::<RenderGpuSceneBounds>(false),
            ),
        );
        let layout_descriptor = BindGroupLayoutDescriptor::new("prism gpu scene", &entries);
        Self {
            layout: device.create_bind_group_layout("prism gpu scene", &entries),
            layout_descriptor,
            bind_group: None,
            buffer_ids: None,
        }
    }
}

impl GpuSceneBindGroup {
    pub(crate) fn prepare(&mut self, device: &RenderDevice, buffers: &GpuSceneBuffers) {
        let Some((instances, current, previous, bounds)) = buffers.binding_buffers() else {
            return;
        };
        let ids = [instances.id(), current.id(), previous.id(), bounds.id()];
        if self.buffer_ids == Some(ids) {
            return;
        }
        self.bind_group = Some(device.create_bind_group(
            "prism gpu scene",
            &self.layout,
            &BindGroupEntries::sequential((
                instances.as_entire_binding(),
                current.as_entire_binding(),
                previous.as_entire_binding(),
                bounds.as_entire_binding(),
            )),
        ));
        self.buffer_ids = Some(ids);
    }
}
