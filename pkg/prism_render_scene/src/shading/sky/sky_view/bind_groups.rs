//! Bind group for the sky-view LUT bake: `[sampler, ms_lut, tr_lut, storage]`.
use super::{pipeline::SkyViewPipeline, resources::SkyViewLut};
use super::super::multiscatter::SkyMultiscatterLut;
use super::super::transmittance::SkyTransmittanceLut;
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::BindGroup, render_resource::BindGroupEntries, renderer::RenderDevice,
};
#[derive(Resource)]
pub(crate) struct SkyViewBindGroup(pub(crate) BindGroup);
pub(crate) fn prepare_sky_view_bind_group(
    mut commands: Commands,
    pipeline: Res<SkyViewPipeline>,
    ms_lut: Res<SkyMultiscatterLut>,
    tr_lut: Res<SkyTransmittanceLut>,
    lut: Res<SkyViewLut>,
    device: Res<RenderDevice>,
) {
    let group = device.create_bind_group(
        "prism sky-view LUT",
        &pipeline.layout,
        &BindGroupEntries::sequential((
            &pipeline.sampler,
            ms_lut.view(),
            tr_lut.view(),
            lut.view(),
        )),
    );
    commands.insert_resource(SkyViewBindGroup(group));
}
