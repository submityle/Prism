//! Storage target bind group for the LUT bake.
use super::{pipeline::SkyMultiscatterPipeline, resources::SkyMultiscatterLut};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::BindGroup, render_resource::BindGroupEntries, renderer::RenderDevice,
};
#[derive(Resource)]
pub(crate) struct SkyMultiscatterBindGroup(pub(crate) BindGroup);
pub(crate) fn prepare_sky_multiscatter_bind_group(
    mut commands: Commands,
    pipeline: Res<SkyMultiscatterPipeline>,
    lut: Res<SkyMultiscatterLut>,
    device: Res<RenderDevice>,
) {
    let group = device.create_bind_group(
        "prism sky multiple-scattering LUT",
        &pipeline.layout,
        &BindGroupEntries::sequential((lut.view(),)),
    );
    commands.insert_resource(SkyMultiscatterBindGroup(group));
}
