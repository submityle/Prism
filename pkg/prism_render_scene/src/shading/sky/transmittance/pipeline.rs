//! Compute pipeline for baking the sky transmittance LUT.
use super::{abi::GpuSkyTransmittanceParams, resources::LUT_FORMAT};
use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{binding_types::texture_storage_2d, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{render_resource::*, renderer::RenderDevice};
use bevy_shader::Shader;
#[derive(Resource)]
pub(crate) struct SkyTransmittancePipeline {
    pub(crate) pipeline: CachedComputePipelineId,
    pub(crate) layout: BindGroupLayout,
}
pub(crate) fn init_sky_transmittance_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    assets: Res<bevy_asset::AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (texture_storage_2d(
            LUT_FORMAT,
            StorageTextureAccess::WriteOnly,
        ),),
    );
    let descriptor = BindGroupLayoutDescriptor::new("prism sky transmittance LUT", &entries);
    let layout = device.create_bind_group_layout("prism sky transmittance LUT", &entries);
    let shader: Handle<Shader> = load_embedded_asset!(
        assets.as_ref(),
        "../../../shaders/sky_transmittance_lut.wesl"
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism sky transmittance LUT".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSkyTransmittanceParams>() as u32,
        shader,
        entry_point: Some("sky_transmittance_lut".into()),
        ..Default::default()
    });
    commands.insert_resource(SkyTransmittancePipeline { pipeline, layout });
}
