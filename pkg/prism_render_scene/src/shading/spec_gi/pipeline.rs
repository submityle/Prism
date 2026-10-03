//! Compute pipeline and group-0 layout for the glossy-specular ReSTIR *reuse*
//! pass, plus the `RenderStartup` initializer that queues it.
//!
//! The reuse pass is a single screen-space compute dispatch (`spec_gi_reuse`
//! entry point in `shaders/spec_gi_reuse.wesl`): one invocation per framebuffer
//! texel reconstructs the pixel's view-space glossy point from the SSR prepass
//! reverse-Z depth + packed `normal_roughness`, streams the current-frame
//! screen-space GGX candidate into a fresh reservoir, temporally merges the
//! same-pixel prior-frame reservoir under the roughness-tightened confidence
//! cap, finalises the unbiased contribution weight and writes both the packed
//! reservoir (ping-pong storage, for next frame) and the resolved specular +
//! confidence (storage texture, for `spec_denoise` and the composite).
//!
//! Unlike [`super::super::ssgi::trace`] and [`super::super::world_restir`] — both
//! of which pass their dispatch config in an immediate (push-constant) block —
//! the reuse kernel reads its [`GpuSpecGiReuseConfig`] from a bound **uniform**
//! buffer at `@group(0) @binding(0)` (so the per-view bind-group slice uploads
//! one small uniform per view). The remaining six bindings mirror the frozen
//! `spec_gi_reuse.wesl` group-0 contract exactly:
//!
//! * `0` `var<uniform> config: SpecGiReuseConfig` (96-byte [`GpuSpecGiReuseConfig`]),
//! * `1` `prior_reservoirs: array<GpuSpecrReservoir>` (read-only, last frame),
//! * `2` `out_reservoirs: array<GpuSpecrReservoir>` (read-write, this frame),
//! * `3` `scene_depth` (SSR prepass reverse-Z depth, `textureLoad`ed),
//! * `4` `normal_roughness` (packed view normal + roughness G-buffer),
//! * `5` `candidate` (the SSR trace's screen-space glossy candidate radiance),
//! * `6` `resolved_out` (write-only `rgba16float` specular + confidence).

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_storage_2d,
            uniform_buffer_sized,
        },
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

use super::abi::GpuSpecGiReuseConfig;
use super::resources::SPEC_GI_RESOLVED_FORMAT;

/// Minimum binding size of the `config` uniform: the frozen 96-byte
/// [`GpuSpecGiReuseConfig`] ABI the kernel reads at `@binding(0)`. Declared on
/// the layout so a short upload is rejected at bind-group creation rather than
/// read as garbage by the shader.
const SPEC_GI_REUSE_CONFIG_SIZE: u64 = size_of::<GpuSpecGiReuseConfig>() as u64;

/// Compute pipeline for the reuse kernel and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct SpecGiReusePipeline {
    /// `spec_gi_reuse` compute entry point, specialized against the group-0
    /// layout below. Recorded by the dispatch node in a follow-up slice.
    reuse: CachedComputePipelineId,
    /// group 0: the config uniform (0), the ping-pong reservoir pair (prior
    /// read-only 1, out read-write 2), the three SSR-rebuilt reads (depth 3,
    /// normal/roughness 4, candidate 5) and the write-only resolved target (6).
    layout: BindGroupLayout,
}

impl SpecGiReusePipeline {
    /// The `spec_gi_reuse` compute pipeline id. Recorded by the dispatch node in
    /// a follow-up slice.
    #[allow(dead_code)] // the dispatch node records this pipeline in a follow-up slice.
    pub(crate) fn reuse(&self) -> CachedComputePipelineId {
        self.reuse
    }

    /// group-0 layout the per-view bind group builds against in a follow-up
    /// slice.
    #[allow(dead_code)] // the bind-group slice builds against this layout next.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// group-0 layout mirroring `spec_gi_reuse.wesl`: the 96-byte config uniform,
/// the read-only prior / read-write out reservoir storage buffers (unsized
/// `array<GpuSpecrReservoir>` at the frozen 64-byte stride), the three
/// non-filterable float reads (SSR depth, packed normal/roughness and the
/// candidate radiance, all `textureLoad`ed), then the write-only `rgba16float`
/// resolved specular + confidence target.
fn layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, NonZero::new(SPEC_GI_REUSE_CONFIG_SIZE)),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SPEC_GI_RESOLVED_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SpecGiReusePipeline`].
#[allow(dead_code)] // registered in `RenderStartup` by the plugin slice (next).
pub(crate) fn init_spec_gi_reuse_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism spec_gi reuse", &entries);
    let layout = device.create_bind_group_layout("prism spec_gi reuse", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/spec_gi_reuse.wesl");

    // No `immediate_size`: the config travels in the bound uniform at binding 0,
    // not a push-constant block (unlike the SSGI / world-space ReSTIR passes).
    let reuse = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism spec_gi reuse".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("spec_gi_reuse".into()),
        ..Default::default()
    });

    commands.insert_resource(SpecGiReusePipeline { reuse, layout });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_uniform_min_size_matches_frozen_abi() {
        // The layout declares a non-zero minimum binding size equal to the
        // frozen 96-byte config ABI so a short upload is rejected at bind-group
        // creation instead of being read as garbage by the kernel.
        assert_eq!(SPEC_GI_REUSE_CONFIG_SIZE, 96);
        assert_eq!(
            size_of::<GpuSpecGiReuseConfig>() as u64,
            SPEC_GI_REUSE_CONFIG_SIZE
        );
        assert!(NonZero::new(SPEC_GI_REUSE_CONFIG_SIZE).is_some());
    }

    #[test]
    fn group0_layout_has_seven_sequential_bindings() {
        // Arity guard: config uniform + 2 reservoir storage buffers + 3 texture
        // reads + 1 storage-texture write = the 7-binding group-0 contract the
        // `spec_gi_reuse.wesl` kernel declares.
        let entries = layout_entries();
        assert_eq!(entries.len(), 7);
    }
}
