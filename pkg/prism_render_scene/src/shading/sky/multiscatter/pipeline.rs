//! Compute pipeline for baking the sky LUT.
use super::{abi::GpuSkyMultiscatterParams, resources::LUT_FORMAT};
use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{binding_types::texture_storage_2d, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{render_resource::*, renderer::RenderDevice};
use bevy_shader::Shader;
#[derive(Resource)]
pub(crate) struct SkyMultiscatterPipeline {
    pub(crate) pipeline: CachedComputePipelineId,
    pub(crate) layout: BindGroupLayout,
}
pub(crate) fn init_sky_multiscatter_pipeline(
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
    let descriptor = BindGroupLayoutDescriptor::new("prism sky multiple-scattering LUT", &entries);
    let layout = device.create_bind_group_layout("prism sky multiple-scattering LUT", &entries);
    let shader: Handle<Shader> = load_embedded_asset!(
        assets.as_ref(),
        "../../../shaders/sky_multiscatter_lut.wesl"
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism sky multiple-scattering LUT".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSkyMultiscatterParams>() as u32,
        shader,
        entry_point: Some("sky_multiscatter_lut".into()),
        ..Default::default()
    });
    commands.insert_resource(SkyMultiscatterPipeline { pipeline, layout });
}
