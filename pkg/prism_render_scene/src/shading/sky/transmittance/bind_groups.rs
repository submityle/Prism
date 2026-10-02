//! Storage target bind group for the transmittance LUT bake.
use super::{pipeline::SkyTransmittancePipeline, resources::SkyTransmittanceLut};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::BindGroup, render_resource::BindGroupEntries, renderer::RenderDevice,
};
#[derive(Resource)]
pub(crate) struct SkyTransmittanceBindGroup(pub(crate) BindGroup);
pub(crate) fn prepare_sky_transmittance_bind_group(
    mut commands: Commands,
    pipeline: Res<SkyTransmittancePipeline>,
    lut: Res<SkyTransmittanceLut>,
    device: Res<RenderDevice>,
) {
    let group = device.create_bind_group(
        "prism sky transmittance LUT",
        &pipeline.layout,
        &BindGroupEntries::sequential((lut.view(),)),
    );
    commands.insert_resource(SkyTransmittanceBindGroup(group));
}
