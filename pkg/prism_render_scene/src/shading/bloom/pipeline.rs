//! Bloom compute pipelines, their bind-group layouts and the shared sampler.
//!
//! Five entry points in `bloom.wesl` drive the chain — `bloom_copy_scene`,
//! `bloom_prefilter_downsample`, `bloom_downsample`, `bloom_upsample` and
//! `bloom_combine_pass` — over three distinct group-0 layouts:
//!
//! * `copy` — a plain texture read + storage write (`[texture, storage]`).
//! * `downsample` — a sampled read behind the linear-clamp sampler + storage
//!   write (`[sampler, texture, storage]`); shared by the prefilter and every
//!   deeper downsample.
//! * `merge` — two texture reads (one sampled, one loaded) + storage write
//!   (`[sampler, texture, texture, storage]`); shared by the upsample (coarse
//!   sampled + finer loaded) and the combine (bloom sampled + base loaded).
//!
//! All five share one 32-byte [`GpuBloomParams`] immediate block and the one
//! linear-clamp sampler.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_2d, texture_storage_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        AddressMode, BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor,
        FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        StorageTextureAccess, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::GpuBloomParams;

/// All bloom compute pipelines plus the layouts and sampler their bind groups
/// need.
#[derive(Resource)]
pub(crate) struct BloomPipelines {
    /// `bloom_copy_scene`: full-res copy of `scene_color` into the base target.
    pub(crate) copy: CachedComputePipelineId,
    /// `bloom_prefilter_downsample`: Karis prefilter + partial-Karis first
    /// downsample into the half-res mip 0.
    pub(crate) prefilter: CachedComputePipelineId,
    /// `bloom_downsample`: energy-preserving COD 13-tap downsample.
    pub(crate) downsample: CachedComputePipelineId,
    /// `bloom_upsample`: 3x3 tent upsample of the coarser mip added to the finer.
    pub(crate) upsample: CachedComputePipelineId,
    /// `bloom_combine_pass`: blend the accumulated bloom over the scene base.
    pub(crate) combine: CachedComputePipelineId,
    /// `[texture, storage]` layout for the copy.
    pub(crate) copy_layout: BindGroupLayout,
    /// `[sampler, texture, storage]` layout shared by the prefilter + downsamples.
    pub(crate) downsample_layout: BindGroupLayout,
    /// `[sampler, texture, texture, storage]` layout shared by the upsample +
    /// combine.
    pub(crate) merge_layout: BindGroupLayout,
    /// Linear-clamp sampler used by every sampled read in the chain.
    pub(crate) sampler: Sampler,
}

/// `[texture, storage]` — the copy pass.
fn copy_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `[sampler, texture, storage]` — the prefilter + every downsample.
fn downsample_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `[sampler, texture, texture, storage]` — the upsample + combine.
fn merge_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`BloomPipelines`]: builds the three layouts,
/// the shared linear-clamp sampler, and queues all five compute pipelines.
pub(crate) fn init_bloom_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let copy_entries = copy_layout_entries();
    let downsample_entries = downsample_layout_entries();
    let merge_entries = merge_layout_entries();

    let copy_desc = BindGroupLayoutDescriptor::new("prism bloom copy", &copy_entries);
    let downsample_desc =
        BindGroupLayoutDescriptor::new("prism bloom downsample", &downsample_entries);
    let merge_desc = BindGroupLayoutDescriptor::new("prism bloom merge", &merge_entries);

    let copy_layout = device.create_bind_group_layout("prism bloom copy", &copy_entries);
    let downsample_layout =
        device.create_bind_group_layout("prism bloom downsample", &downsample_entries);
    let merge_layout = device.create_bind_group_layout("prism bloom merge", &merge_entries);

    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism bloom linear-clamp sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/bloom.wesl");
    let immediate_size = size_of::<GpuBloomParams>() as u32;

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism bloom copy".into()),
        layout: vec![copy_desc],
        immediate_size,
        shader: shader.clone(),
        entry_point: Some("bloom_copy_scene".into()),
        ..Default::default()
    });
    let prefilter = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism bloom prefilter".into()),
        layout: vec![downsample_desc.clone()],
        immediate_size,
        shader: shader.clone(),
        entry_point: Some("bloom_prefilter_downsample".into()),
        ..Default::default()
    });
    let downsample = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism bloom downsample".into()),
        layout: vec![downsample_desc],
        immediate_size,
        shader: shader.clone(),
        entry_point: Some("bloom_downsample".into()),
        ..Default::default()
    });
    let upsample = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism bloom upsample".into()),
        layout: vec![merge_desc.clone()],
        immediate_size,
        shader: shader.clone(),
        entry_point: Some("bloom_upsample".into()),
        ..Default::default()
    });
    let combine = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism bloom combine".into()),
        layout: vec![merge_desc],
        immediate_size,
        shader,
        entry_point: Some("bloom_combine_pass".into()),
        ..Default::default()
    });

    commands.insert_resource(BloomPipelines {
        copy,
        prefilter,
        downsample,
        upsample,
        combine,
        copy_layout,
        downsample_layout,
        merge_layout,
        sampler,
    });
}
