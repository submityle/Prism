use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{binding_types::storage_buffer_read_only, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, BufferId, ShaderStages},
    renderer::RenderDevice,
};

use super::{buffers::RenderGeometryBuffers, rows::*};

#[derive(Resource)]
pub struct GeometryBindGroup {
    pub(crate) bind_group: Option<BindGroup>,
    pub(crate) layout_descriptor: BindGroupLayoutDescriptor,
    layout: BindGroupLayout,
    ids: Option<[BufferId; 2]>,
}

impl FromWorld for GeometryBindGroup {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only::<RenderGeometryHeader>(false),
                storage_buffer_read_only::<RenderGeometryLod>(false),
            ),
        );
        Self {
            bind_group: None,
            layout_descriptor: BindGroupLayoutDescriptor::new("prism geometry ABI", &entries),
            layout: device.create_bind_group_layout("prism geometry ABI", &entries),
            ids: None,
        }
    }
}

pub(crate) fn prepare_geometry_bind_group(
    device: Res<RenderDevice>,
    buffers: Res<RenderGeometryBuffers>,
    mut bindings: ResMut<GeometryBindGroup>,
) {
    let Some((headers, lods)) = buffers.buffers() else {
        return;
    };
    let ids = [headers.id(), lods.id()];
    if bindings.ids == Some(ids) {
        return;
    }
    bindings.bind_group = Some(device.create_bind_group(
        "prism geometry ABI",
        &bindings.layout,
        &BindGroupEntries::sequential((headers.as_entire_binding(), lods.as_entire_binding())),
    ));
    bindings.ids = Some(ids);
}
