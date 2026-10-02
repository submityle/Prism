//! The light-routing cull compute pipeline, its owned group-0 layout, and the
//! `RenderStartup` initializer that queues it.
//!
//! Mirrors [`super::super::world_space_gi::pipeline`]: one compute entry point
//! (`light_routing_cull_main`) from `light_routing.wesl`, specialized against
//! its group-0 layout and its single immediate block. The layout binds the
//! read-only per-light routing records at binding `0` and the two read-write
//! export buffers — the channel-gated visibility mask (`1`) and the NPR
//! per-layer contribution masks (`2`) — matching the shader's `sequential`
//! `{0, 1, 2}` bindings.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::GpuLightRoutingParams;

/// The light-routing cull compute pipeline and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct LightRoutingPipeline {
    /// `light_routing_cull_main` entry: per-word cluster cull refinement.
    pipeline: CachedComputePipelineId,
    /// group 0 for `light_routing_cull_main`: routing records read-only, the
    /// visibility mask and per-layer masks read-write.
    layout: BindGroupLayout,
}

impl LightRoutingPipeline {
    /// The `light_routing_cull_main` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// group-0 layout for the `light_routing_cull_main` dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// `light_routing_cull_main` group-0 layout: the read-only per-light routing
/// records at binding `0`, then the read-write channel-gated visibility mask
/// (`1`) and NPR per-layer contribution masks (`2`).
fn layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`LightRoutingPipeline`].
pub(crate) fn init_light_routing_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism light routing", &entries);
    let layout = device.create_bind_group_layout("prism light routing", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/light_routing.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism light routing cull".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuLightRoutingParams>() as u32,
        shader,
        entry_point: Some("light_routing_cull_main".into()),
        ..Default::default()
    });

    commands.insert_resource(LightRoutingPipeline { pipeline, layout });
}
