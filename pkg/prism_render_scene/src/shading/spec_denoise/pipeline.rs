//! Compute pipeline and group-0 layout for the specular-GI *spatial denoise*
//! pass, plus the `RenderStartup` initializer that queues it.
//!
//! The spatial pass is a single screen-space compute dispatch
//! (`spec_denoise_spatial` entry point in `shaders/spec_denoise_spatial.wesl`):
//! one invocation per framebuffer texel reconstructs the pixel's view-space
//! position from the SSR prepass reverse-Z depth, selects an anisotropic,
//! contact-hardened blur footprint from the surface roughness and the SSR hit
//! distance, then runs the edge-aware cross-bilateral gather (gated on depth,
//! normal and roughness) that matches the CPU golden's `spatial_filter`.
//!
//! Like [`super::super::spec_gi::pipeline`], the kernel reads its
//! [`GpuSpecDenoiseSpatialConfig`] from a bound **uniform** buffer at
//! `@group(0) @binding(0)` (one small uniform per view), not an immediate block.
//! The remaining five bindings mirror the frozen `spec_denoise_spatial.wesl`
//! group-0 contract exactly:
//!
//! * `0` `var<uniform> sds_config: SpecDenoiseSpatialConfig` (112-byte ABI),
//! * `1` `sds_resolved` (the `spec_gi` resolve: rgb specular, a confidence),
//! * `2` `sds_hit` (the SSR trace's per-pixel world-space hit distance),
//! * `3` `sds_nr` (packed view normal + roughness G-buffer),
//! * `4` `sds_depth` (SSR prepass reverse-Z device depth),
//! * `5` `sds_filtered` (write-only `rgba16float` filtered specular + confidence).

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

use super::abi::GpuSpecDenoiseSpatialConfig;
use super::resources::SPEC_DENOISE_FILTERED_FORMAT;

/// Minimum binding size of the `sds_config` uniform: the frozen 112-byte
/// [`GpuSpecDenoiseSpatialConfig`] ABI the kernel reads at `@binding(0)`.
/// Declared on the layout so a short upload is rejected at bind-group creation
/// rather than read as garbage by the shader.
const SPEC_DENOISE_SPATIAL_CONFIG_SIZE: u64 = size_of::<GpuSpecDenoiseSpatialConfig>() as u64;

/// Compute pipeline for the spatial kernel and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct SpecDenoiseSpatialPipeline {
    /// `spec_denoise_spatial` compute entry point, specialized against the
    /// group-0 layout below. Recorded by the dispatch node.
    spatial: CachedComputePipelineId,
    /// group 0: the config uniform (0), the four reads (resolve 1, SSR hit 2,
    /// normal/roughness 3, reverse-Z depth 4) and the write-only filtered
    /// target (5).
    layout: BindGroupLayout,
}

impl SpecDenoiseSpatialPipeline {
    /// The `spec_denoise_spatial` compute pipeline id. Recorded by the dispatch
    /// node.
    pub(crate) fn spatial(&self) -> CachedComputePipelineId {
        self.spatial
    }

    /// group-0 layout the per-view bind group builds against.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// group-0 layout mirroring `spec_denoise_spatial.wesl`: the 112-byte config
/// uniform, four non-filterable float reads (the `spec_gi` resolve, the SSR hit
/// distance, the packed normal/roughness G-buffer and the SSR reverse-Z depth,
/// all `textureLoad`ed) and the write-only `rgba16float` filtered target.
fn layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, NonZero::new(SPEC_DENOISE_SPATIAL_CONFIG_SIZE)),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(
                SPEC_DENOISE_FILTERED_FORMAT,
                StorageTextureAccess::WriteOnly,
            ),
        ),
    )
}

/// `RenderStartup` initializer for [`SpecDenoiseSpatialPipeline`].
pub(crate) fn init_spec_denoise_spatial_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism spec_denoise spatial", &entries);
    let layout = device.create_bind_group_layout("prism spec_denoise spatial", &entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/spec_denoise_spatial.wesl"
    );

    // No `immediate_size`: the config travels in the bound uniform at binding 0,
    // not a push-constant block (mirroring the `spec_gi` reuse pass).
    let spatial = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism spec_denoise spatial".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("spec_denoise_spatial".into()),
        ..Default::default()
    });

    commands.insert_resource(SpecDenoiseSpatialPipeline { spatial, layout });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_uniform_min_size_matches_frozen_abi() {
        // The layout declares a non-zero minimum binding size equal to the
        // frozen 112-byte config ABI so a short upload is rejected at bind-group
        // creation instead of being read as garbage by the kernel.
        assert_eq!(SPEC_DENOISE_SPATIAL_CONFIG_SIZE, 112);
        assert_eq!(
            size_of::<GpuSpecDenoiseSpatialConfig>() as u64,
            SPEC_DENOISE_SPATIAL_CONFIG_SIZE
        );
        assert!(NonZero::new(SPEC_DENOISE_SPATIAL_CONFIG_SIZE).is_some());
    }

    #[test]
    fn group0_layout_has_six_sequential_bindings() {
        // Arity guard: config uniform + 4 texture reads + 1 storage-texture
        // write = the 6-binding group-0 contract the `spec_denoise_spatial.wesl`
        // kernel declares.
        let entries = layout_entries();
        assert_eq!(entries.len(), 6);
    }
}
