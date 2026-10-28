//! Decal resolve compute pipeline and its exact six-buffer group layout.

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
use prism_render_architecture::render_fabric::{
    DECAL_CLUSTER_INDEX_STRIDE, DECAL_CLUSTER_RANGE_STRIDE, DECAL_INSTANCE_STRIDE,
    DECAL_MATERIAL_STRIDE, DECAL_RECEIVER_STRIDE, DECAL_RESOLVED_RECEIVER_STRIDE,
};
use std::num::NonZeroU64;

/// The production Decal resolve pipeline and group-0 layout.
#[derive(Resource)]
pub(crate) struct DecalResolvePipeline {
    resolve: CachedComputePipelineId,
    layout: BindGroupLayout,
}

impl DecalResolvePipeline {
    pub(crate) fn resolve(&self) -> CachedComputePipelineId {
        self.resolve
    }

    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

fn layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, NonZeroU64::new(DECAL_INSTANCE_STRIDE)),
            storage_buffer_read_only_sized(false, NonZeroU64::new(DECAL_MATERIAL_STRIDE)),
            storage_buffer_read_only_sized(false, NonZeroU64::new(DECAL_RECEIVER_STRIDE)),
            storage_buffer_read_only_sized(false, NonZeroU64::new(DECAL_CLUSTER_RANGE_STRIDE)),
            storage_buffer_read_only_sized(false, NonZeroU64::new(DECAL_CLUSTER_INDEX_STRIDE)),
            storage_buffer_sized(false, NonZeroU64::new(DECAL_RESOLVED_RECEIVER_STRIDE)),
        ),
    )
}

pub(crate) fn init_decal_resolve_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/decal_resolve.wesl");
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism decal resolve", &entries);
    let layout = device.create_bind_group_layout("prism decal resolve", &entries);
    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism decal resolve".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("decal_resolve".into()),
        ..Default::default()
    });
    commands.insert_resource(DecalResolvePipeline { resolve, layout });
}

#[cfg(test)]
pub(crate) fn create_decal_resolve_pipeline(
    device: &RenderDevice,
    cache: &PipelineCache,
    shader: Handle<Shader>,
) -> DecalResolvePipeline {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism decal resolve", &entries);
    let layout = device.create_bind_group_layout("prism decal resolve", &entries);
    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism decal resolve".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("decal_resolve".into()),
        ..Default::default()
    });
    DecalResolvePipeline { resolve, layout }
}
