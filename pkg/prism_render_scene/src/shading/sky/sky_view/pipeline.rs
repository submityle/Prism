//! Compute pipeline for baking the sky-view LUT.
//!
//! The group-0 layout is `[sampler, ms_lut, tr_lut, storage]`: a linear-clamp
//! sampler, the multiple-scattering LUT and the transmittance LUT read as
//! sampled textures, and the sky-view LUT written as a storage image.
use super::{abi::GpuSkyViewParams, resources::LUT_FORMAT};
use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_2d, texture_storage_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{render_resource::*, renderer::RenderDevice};
use bevy_shader::Shader;
#[derive(Resource)]
pub(crate) struct SkyViewPipeline {
    pub(crate) pipeline: CachedComputePipelineId,
    pub(crate) layout: BindGroupLayout,
    pub(crate) sampler: Sampler,
}
pub(crate) fn init_sky_view_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    assets: Res<bevy_asset::AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_storage_2d(LUT_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    );
    let descriptor = BindGroupLayoutDescriptor::new("prism sky-view LUT", &entries);
    let layout = device.create_bind_group_layout("prism sky-view LUT", &entries);
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism sky-view LUT linear-clamp sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });
    let shader: Handle<Shader> =
        load_embedded_asset!(assets.as_ref(), "../../../shaders/sky_view_lut.wesl");
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism sky-view LUT".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSkyViewParams>() as u32,
        shader,
        entry_point: Some("sky_view_lut".into()),
        ..Default::default()
    });
    commands.insert_resource(SkyViewPipeline {
        pipeline,
        layout,
        sampler,
    });
}
