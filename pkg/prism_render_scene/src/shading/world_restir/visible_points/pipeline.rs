//! The world-space `ReSTIR` visible-point producer compute pipeline, its owned
//! group-0 layout, and the `RenderStartup` initializer that queues it.
//!
//! `visible_points_main` (`world_restir_visible_points.wesl`) runs one
//! invocation per screen tile. Its group-0 binds the SSR prepass reverse-Z
//! device depth (sampled, 0), the SSR packed view-space `normal_roughness`
//! (sampled, 1) and the per-frame visible-point list (storage read-write, 2):
//! each invocation reconstructs the tile-centre shading point from the depth,
//! lifts it + its normal into world space and writes one
//! [`super::super::abi::GpuWorldRestirInjectPoint`] record. The framebuffer
//! size, the tile grid and the per-point counts arrive in the
//! [`GpuVisiblePointsParams`] immediate block, so no uniform buffer is bound.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_sized, texture_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::GpuVisiblePointsParams;

/// The world-space `ReSTIR` visible-point producer compute pipeline and its
/// owned group-0 layout.
#[derive(Resource)]
pub(crate) struct WorldRestirVisiblePointsPipeline {
    /// `visible_points_main` entry: one invocation per screen tile.
    pipeline: CachedComputePipelineId,
    /// group 0 for `visible_points_main`: the SSR prepass depth (sampled, 0),
    /// the SSR packed `normal_roughness` (sampled, 1) and the per-frame
    /// visible-point list (storage read-write, 2).
    layout: BindGroupLayout,
}

impl WorldRestirVisiblePointsPipeline {
    /// The `visible_points_main` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// group-0 layout for the `visible_points_main` dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// `visible_points_main` group-0 layout: the SSR prepass reverse-Z device depth
/// (sampled, 0) and the SSR packed view-space `normal_roughness` (sampled, 1),
/// both read with `textureLoad` (so non-filterable float is sufficient), and
/// the per-frame visible-point list (storage read-write, 2) at the frozen
/// [`super::super::abi::WORLD_RESTIR_INJECT_POINT_STRIDE`].
fn layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`WorldRestirVisiblePointsPipeline`].
pub(crate) fn init_world_restir_visible_points_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor =
        BindGroupLayoutDescriptor::new("prism world-space ReSTIR visible points", &entries);
    let layout =
        device.create_bind_group_layout("prism world-space ReSTIR visible points", &entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../../../shaders/world_restir_visible_points.wesl"
    );

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR visible points".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuVisiblePointsParams>() as u32,
        shader,
        entry_point: Some("visible_points_main".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldRestirVisiblePointsPipeline { pipeline, layout });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_points_immediate_block_is_the_abi_size() {
        // The pipeline reserves exactly the frozen producer immediate block; a
        // drift here would mismatch `set_immediates` against the shader's
        // `var<immediate>` block.
        assert_eq!(size_of::<GpuVisiblePointsParams>() as u32, 160);
    }
}
