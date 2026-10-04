//! Compute pipelines and group-0 layouts for the specular-GI *temporal denoise*
//! passes (reproject + history-clamp), plus the `RenderStartup` initializers
//! that queue them.
//!
//! The temporal path is two screen-space compute dispatches that sit between the
//! `spec_gi` reuse resolve and the spatial pre-filter:
//!
//! * **reproject** (`spec_denoise_reproject` entry point in
//!   `shaders/spec_denoise_reproject.wesl`) rebuilds each pixel's world-space
//!   surface from the SSR prepass reverse-Z depth, follows the specular virtual
//!   reflection point back into the previous frame and samples the prior
//!   accumulated history/metadata/luminance planes under a world-space
//!   disocclusion guard, writing the three reprojected planes the clamp reads;
//! * **history-clamp** (`spec_denoise_history_clamp` entry point in
//!   `shaders/spec_denoise_history_clamp.wesl`) fuses that reprojected history
//!   with the current-frame noisy resolve under a colour-AABB clamp and a
//!   normal/roughness consistency gate, advancing the age + dual-rate luminance
//!   EMAs and writing both the next-frame history planes and the denoised
//!   specular the spatial pass filters.
//!
//! Like [`super::pipeline`] and [`super::super::spec_gi::pipeline`], each kernel
//! reads its config from a bound **uniform** at `@group(0) @binding(0)` (not an
//! immediate block), so neither pipeline carries an `immediate_size`. Both WESL
//! files additionally declare a `@group(1)` probe entry used only by the WESL
//! parity harness; the pipeline specializes group 0 and the named entry point
//! only, so the probe group is irrelevant here (exactly as the spatial pass
//! ignores its own test-only bindings).
//!
//! Each layout mirrors its frozen 11-binding group-0 contract exactly: the
//! config uniform (0), the non-filterable float reads and the write-only
//! `rgba16float` storage targets.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_2d, texture_storage_2d, uniform_buffer_sized},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use core::num::NonZero;

use super::resources::SPEC_DENOISE_FILTERED_FORMAT;
use super::temporal_abi::{GpuSpecDenoiseHistoryClampConfig, GpuSpecDenoiseReprojectConfig};

/// Minimum binding size of the reproject `sdr_config` uniform: the frozen
/// 304-byte [`GpuSpecDenoiseReprojectConfig`] ABI the kernel reads at
/// `@binding(0)`. Declared on the layout so a short upload is rejected at
/// bind-group creation rather than read as garbage by the shader.
const SPEC_DENOISE_REPROJECT_CONFIG_SIZE: u64 = size_of::<GpuSpecDenoiseReprojectConfig>() as u64;

/// Minimum binding size of the history-clamp `sdh_config` uniform: the frozen
/// 48-byte [`GpuSpecDenoiseHistoryClampConfig`] ABI the kernel reads at
/// `@binding(0)`.
const SPEC_DENOISE_HISTORY_CLAMP_CONFIG_SIZE: u64 =
    size_of::<GpuSpecDenoiseHistoryClampConfig>() as u64;

/// Compute pipeline for the reproject kernel and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct SpecDenoiseReprojectPipeline {
    /// `spec_denoise_reproject` compute entry point, specialized against the
    /// group-0 layout below. Recorded by the dispatch node.
    reproject: CachedComputePipelineId,
    /// group 0: config uniform (0), seven non-filterable float reads
    /// (normal/roughness 1, SSR hit 2, reverse-Z depth 3, prev history 4, prev
    /// depth 5, prev meta 6, prev luma 7) and three write-only reprojected
    /// targets (8, 9, 10).
    layout: BindGroupLayout,
}

impl SpecDenoiseReprojectPipeline {
    /// The `spec_denoise_reproject` compute pipeline id. Recorded by the
    /// dispatch node.
    pub(crate) fn reproject(&self) -> CachedComputePipelineId {
        self.reproject
    }

    /// group-0 layout the per-view bind group builds against.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// Compute pipeline for the history-clamp kernel and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct SpecDenoiseHistoryClampPipeline {
    /// `spec_denoise_history_clamp` compute entry point, specialized against the
    /// group-0 layout below. Recorded by the dispatch node.
    history_clamp: CachedComputePipelineId,
    /// group 0: config uniform (0), six non-filterable float reads (reprojected
    /// history 1, reprojected meta 2, reprojected luma 3, resolve 4,
    /// normal/roughness 5, reverse-Z depth 6) and four write-only targets (out
    /// history 7, out meta 8, out luma 9, denoised 10).
    layout: BindGroupLayout,
}

impl SpecDenoiseHistoryClampPipeline {
    /// The `spec_denoise_history_clamp` compute pipeline id. Recorded by the
    /// dispatch node.
    pub(crate) fn history_clamp(&self) -> CachedComputePipelineId {
        self.history_clamp
    }

    /// group-0 layout the per-view bind group builds against.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// group-0 layout mirroring `spec_denoise_reproject.wesl`: the 304-byte config
/// uniform, seven non-filterable float reads (all `textureLoad`ed) and three
/// write-only `rgba16float` reprojected targets.
fn reproject_layout_entries() -> BindGroupLayoutEntries<11> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, NonZero::new(SPEC_DENOISE_REPROJECT_CONFIG_SIZE)),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
        ),
    )
}

/// group-0 layout mirroring `spec_denoise_history_clamp.wesl`: the 48-byte
/// config uniform, six non-filterable float reads and four write-only
/// `rgba16float` targets.
fn history_clamp_layout_entries() -> BindGroupLayoutEntries<11> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, NonZero::new(SPEC_DENOISE_HISTORY_CLAMP_CONFIG_SIZE)),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
        ),
    )
}

/// `RenderStartup` initializer for [`SpecDenoiseReprojectPipeline`].
pub(crate) fn init_spec_denoise_reproject_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = reproject_layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism spec_denoise reproject", &entries);
    let layout = device.create_bind_group_layout("prism spec_denoise reproject", &entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/spec_denoise_reproject.wesl"
    );

    // No `immediate_size`: the config travels in the bound uniform at binding 0.
    let reproject = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism spec_denoise reproject".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("spec_denoise_reproject".into()),
        ..Default::default()
    });

    commands.insert_resource(SpecDenoiseReprojectPipeline { reproject, layout });
}

/// `RenderStartup` initializer for [`SpecDenoiseHistoryClampPipeline`].
pub(crate) fn init_spec_denoise_history_clamp_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = history_clamp_layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism spec_denoise history_clamp", &entries);
    let layout = device.create_bind_group_layout("prism spec_denoise history_clamp", &entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/spec_denoise_history_clamp.wesl"
    );

    // No `immediate_size`: the config travels in the bound uniform at binding 0.
    let history_clamp = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism spec_denoise history_clamp".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("spec_denoise_history_clamp".into()),
        ..Default::default()
    });

    commands.insert_resource(SpecDenoiseHistoryClampPipeline {
        history_clamp,
        layout,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproject_config_uniform_min_size_matches_frozen_abi() {
        // The layout declares a non-zero minimum binding size equal to the
        // frozen 304-byte reproject config ABI so a short upload is rejected at
        // bind-group creation instead of being read as garbage by the kernel.
        assert_eq!(SPEC_DENOISE_REPROJECT_CONFIG_SIZE, 304);
        assert_eq!(
            size_of::<GpuSpecDenoiseReprojectConfig>() as u64,
            SPEC_DENOISE_REPROJECT_CONFIG_SIZE
        );
        assert!(NonZero::new(SPEC_DENOISE_REPROJECT_CONFIG_SIZE).is_some());
    }

    #[test]
    fn history_clamp_config_uniform_min_size_matches_frozen_abi() {
        // Same guard for the 48-byte history-clamp config ABI.
        assert_eq!(SPEC_DENOISE_HISTORY_CLAMP_CONFIG_SIZE, 48);
        assert_eq!(
            size_of::<GpuSpecDenoiseHistoryClampConfig>() as u64,
            SPEC_DENOISE_HISTORY_CLAMP_CONFIG_SIZE
        );
        assert!(NonZero::new(SPEC_DENOISE_HISTORY_CLAMP_CONFIG_SIZE).is_some());
    }

    #[test]
    fn reproject_group0_layout_has_eleven_sequential_bindings() {
        // Arity guard: config uniform + 7 texture reads + 3 storage-texture
        // writes = the 11-binding group-0 contract the reproject kernel declares.
        let entries = reproject_layout_entries();
        assert_eq!(entries.len(), 11);
    }

    #[test]
    fn history_clamp_group0_layout_has_eleven_sequential_bindings() {
        // Arity guard: config uniform + 6 texture reads + 4 storage-texture
        // writes = the 11-binding group-0 contract the clamp kernel declares.
        let entries = history_clamp_layout_entries();
        assert_eq!(entries.len(), 11);
    }
}
