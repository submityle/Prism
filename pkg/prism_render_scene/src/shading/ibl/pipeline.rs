//! Compute pipelines + bind-group layouts for the IBL precompute passes.
//!
//! Two kernels feed the resolve's split-sum specular term:
//!
//! * [`BrdfLutPipeline`] runs `shaders/brdf_lut.wesl`.  The kernel reads no
//!   scene state — it derives `n_dot_v` / `roughness` from the storage texture's
//!   dimensions — so a single write-only `Rg16Float` storage texture is all it
//!   binds, with the GGX sample count arriving in a 16-byte immediate block.
//! * [`EnvPrefilterPipeline`] runs `shaders/env_prefilter.wesl`.  It convolves a
//!   source radiance cube into one output mip per dispatch, so it binds the
//!   source `texture_cube` + sampler and a write-only `rgba16float`
//!   `texture_storage_2d_array` target, with the per-mip roughness / sample
//!   count / mip size arriving in a 16-byte immediate block.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_cube, texture_storage_2d, texture_storage_2d_array},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        SamplerBindingType, ShaderStages, StorageTextureAccess, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::{GpuBrdfLutConfig, GpuPrefilterConfig};
use super::resources::{DFG_LUT_FORMAT, PREFILTERED_ENV_FORMAT};

/// Compute pipeline and the single owned bind-group layout for the DFG table
/// precompute.
#[derive(Resource)]
pub(crate) struct BrdfLutPipeline {
    /// `integrate_brdf_lut` compute entry point, specialized against
    /// [`Self::layout`] and the 16-byte immediate config block.
    pub(crate) pipeline: CachedComputePipelineId,
    /// group 0: the write-only `Rg16Float` DFG storage texture.
    pub(crate) layout: BindGroupLayout,
}

/// group-0 layout: one write-only `rg16float` storage texture holding the
/// `(scale, bias)` split-sum pair.
fn brdf_lut_layout_entries() -> BindGroupLayoutEntries<1> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (texture_storage_2d(
            DFG_LUT_FORMAT,
            StorageTextureAccess::WriteOnly,
        ),),
    )
}

/// `RenderStartup` initializer for [`BrdfLutPipeline`].
pub(crate) fn init_brdf_lut_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = brdf_lut_layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism DFG LUT", &entries);
    let layout = device.create_bind_group_layout("prism DFG LUT", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/brdf_lut.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism DFG LUT".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuBrdfLutConfig>() as u32,
        shader,
        entry_point: Some("integrate_brdf_lut".into()),
        ..Default::default()
    });

    commands.insert_resource(BrdfLutPipeline { pipeline, layout });
}

/// Compute pipeline and the owned bind-group layout for the prefiltered
/// radiance precompute.
#[derive(Resource)]
pub(crate) struct EnvPrefilterPipeline {
    /// `prefilter_env_map` compute entry point, specialized against
    /// [`Self::layout`] and the 16-byte per-mip immediate config block.
    pub(crate) pipeline: CachedComputePipelineId,
    /// group 0: source radiance cube + filtering sampler + write-only
    /// `rgba16float` array target.
    pub(crate) layout: BindGroupLayout,
}

/// group-0 layout for the prefilter kernel: the source radiance `texture_cube`
/// (binding 0), a filtering `sampler` (binding 1), and the write-only
/// `rgba16float` `texture_storage_2d_array` output mip (binding 2).
fn env_prefilter_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_cube(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_storage_2d_array(PREFILTERED_ENV_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`EnvPrefilterPipeline`].
pub(crate) fn init_env_prefilter_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = env_prefilter_layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism env prefilter", &entries);
    let layout = device.create_bind_group_layout("prism env prefilter", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/env_prefilter.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism env prefilter".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuPrefilterConfig>() as u32,
        shader,
        entry_point: Some("prefilter_env_map".into()),
        ..Default::default()
    });

    commands.insert_resource(EnvPrefilterPipeline { pipeline, layout });
}
