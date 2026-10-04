//! Compute pipeline and group-0 layout for the glossy-specular ReSTIR
//! **spatial** reuse pass, plus the `RenderStartup` initializer that queues it.
//!
//! The spatial pass is the second stage of the screen-space glossy ReSTIR
//! pipeline (`spec_gi_spatial` entry point in `shaders/spec_gi_spatial.wesl`).
//! It runs after the temporal [`super::pipeline`] reuse dispatch has written
//! this frame's post-temporal reservoir table and, behind a render-graph
//! barrier, reads that completed table read-only, pools a frame-jittered disc
//! of neighbour reservoirs onto each pixel's GGX lobe through the golden
//! `specr_merge_glossy`, and overwrites the resolved specular+confidence target
//! only (the reservoir history is left pure — the standard screen-space ReSTIR
//! temporal/spatial split that avoids the progressive over-blur a spatial
//! write-back causes).
//!
//! Unlike the reuse pass (which reads its config from a bound uniform), this
//! pass carries its [`GpuSpecGiSpatialParams`] in an **immediate**
//! (push-constant) block — like the SSGI / composite dispatches — so it binds
//! no uniform buffer. The four group-0 bindings mirror the frozen
//! `spec_gi_spatial.wesl` contract exactly:
//!
//! * `0` `reservoirs: array<GpuSpecrReservoir>` (read-only post-temporal snapshot),
//! * `1` `scene_depth` (SSR prepass reverse-Z depth, `textureLoad`ed),
//! * `2` `normal_roughness` (packed view normal + roughness G-buffer),
//! * `3` `resolved_out` (write-only `rgba16float` specular + confidence).

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_read_only_sized, texture_2d, texture_storage_2d},
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

use super::abi::GpuSpecGiSpatialParams;
use super::resources::SPEC_GI_RESOLVED_FORMAT;

/// Compute pipeline for the spatial reuse kernel and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct SpecGiSpatialPipeline {
    /// `spec_gi_spatial` compute entry point, specialized against the group-0
    /// layout below and the 96-byte [`GpuSpecGiSpatialParams`] immediate block.
    spatial: CachedComputePipelineId,
    /// group 0: the read-only post-temporal reservoir snapshot (0), the two
    /// SSR-rebuilt G-buffer reads (depth 1, normal/roughness 2) and the
    /// write-only resolved target (3).
    layout: BindGroupLayout,
}

impl SpecGiSpatialPipeline {
    /// The `spec_gi_spatial` compute pipeline id. Recorded by the dispatch node.
    pub(crate) fn spatial(&self) -> CachedComputePipelineId {
        self.spatial
    }

    /// group-0 layout the per-view bind group builds against.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// group-0 layout mirroring `spec_gi_spatial.wesl`: the read-only reservoir
/// storage buffer (unsized `array<GpuSpecrReservoir>` at the frozen 64-byte
/// stride), the two non-filterable float G-buffer reads (SSR depth and packed
/// normal/roughness, both `textureLoad`ed), then the write-only `rgba16float`
/// resolved specular + confidence target.
fn layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SPEC_GI_RESOLVED_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SpecGiSpatialPipeline`].
pub(crate) fn init_spec_gi_spatial_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism spec_gi spatial", &entries);
    let layout = device.create_bind_group_layout("prism spec_gi spatial", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/spec_gi_spatial.wesl");

    let spatial = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism spec_gi spatial".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSpecGiSpatialParams>() as u32,
        shader,
        entry_point: Some("spec_gi_spatial".into()),
        ..Default::default()
    });

    commands.insert_resource(SpecGiSpatialPipeline { spatial, layout });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group0_layout_has_four_sequential_bindings() {
        // Arity guard: read-only reservoir snapshot + 2 G-buffer reads + 1
        // storage-texture write = the 4-binding group-0 contract the
        // `spec_gi_spatial.wesl` kernel declares.
        let entries = layout_entries();
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn immediate_block_matches_frozen_spatial_abi() {
        // The pipeline reserves exactly the 96-byte spatial params immediate
        // block the kernel reads via `var<immediate> params`.
        assert_eq!(size_of::<GpuSpecGiSpatialParams>(), 96);
    }
}
