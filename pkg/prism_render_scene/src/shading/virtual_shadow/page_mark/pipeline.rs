//! Virtual-shadow-map page-request (`page-mark`) compute pipeline and its owned
//! group-0 layout.
//!
//! Second stage of the VSM pipeline, downstream of the receiver-generation pass
//! ([`super::super::pipeline`]) and upstream of the CPU allocator: it consumes
//! this frame's per-pixel receivers and marks, in a camera-snapped resident
//! request bitmap, exactly the clipmap pages the visible surfaces will sample.
//! The GPU twin of the golden
//! [`prism_render_shading::shadow::virtual_sm::generate_page_requests`], sharing
//! the golden [`prism_render_shading::ClipmapConfig`] page addressing.
//!
//! The layout mirrors `shaders/vsm_page_mark.wesl` binding-for-binding:
//!
//! * binding 0 -- the read-only per-pixel receiver `storage` array produced by
//!   the receiver-generation pass;
//! * binding 1 -- the read-write per-window-slot request bitmap
//!   (`array<atomic<u32>>`) this pass marks with `atomicOr`.
//!
//! Unlike the receiver-generation pass, every per-frame constant travels in the
//! [`super::super::abi::GpuVsmPageMarkParams`] push-constant immediate block
//! rather than a uniform, so the pipeline declares a non-zero `immediate_size`
//! and carries no binding-2 uniform.

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

use super::super::abi::GpuVsmPageMarkParams;

/// Compute pipeline and its owned group-0 layout for the page-request pass.
#[derive(Resource)]
pub(crate) struct VsmPageMarkPipeline {
    /// `vsm_mark_pages` compute entry, specialized against the group-0 layout
    /// below and the [`GpuVsmPageMarkParams`] immediate block.
    pipeline: CachedComputePipelineId,
    /// group 0: receiver read, request-bitmap read-write.
    layout: BindGroupLayout,
}

impl VsmPageMarkPipeline {
    /// The `vsm_mark_pages` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// The group-0 bind-group layout shared with the page-mark dispatch's bind
    /// group.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// group-0 layout mirroring `vsm_page_mark.wesl`: the read-only receiver
/// `storage` array (binding 0) and the read-write request bitmap (binding 1).
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`VsmPageMarkPipeline`].
pub(crate) fn init_vsm_page_mark_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism VSM page-mark", &entries);
    let layout = device.create_bind_group_layout("prism VSM page-mark", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/vsm_page_mark.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism VSM page-mark".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("vsm_mark_pages".into()),
        immediate_size: size_of::<GpuVsmPageMarkParams>() as u32,
        ..Default::default()
    });

    commands.insert_resource(VsmPageMarkPipeline { pipeline, layout });
}
