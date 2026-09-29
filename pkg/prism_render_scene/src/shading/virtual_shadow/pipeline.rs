//! Virtual-shadow-map receiver-generation compute pipeline and its owned
//! group-0 layout.
//!
//! Mirrors the pipeline plumbing of the sibling compute passes
//! ([`super::super::exposure::histogram`] for the storage-buffer write,
//! [`super::super::resolve`] for the per-frame uniform): a single
//! `vsm_generate_receivers` entry specialized against a three-binding group-0
//! layout. Unlike the resolve / TAA passes it carries no immediate block --
//! every per-frame constant travels in the [`super::abi::GpuVsmReceiverGenParams`]
//! uniform bound at binding 2 -- so `immediate_size` stays zero.
//!
//! The layout mirrors `shaders/vsm_receiver_gen.wesl` binding-for-binding:
//!
//! * binding 0 -- the camera device depth (`R32Float`, non-filterable float,
//!   `textureLoad`ed), sourced from the SSR geometry prepass;
//! * binding 1 -- the write-only per-pixel receiver `storage` array; and
//! * binding 2 -- the per-frame [`super::abi::GpuVsmReceiverGenParams`] uniform.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_sized, texture_2d, uniform_buffer_sized},
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

/// Compute pipeline and its owned group-0 layout for the receiver-generation
/// pass.
#[derive(Resource)]
pub(crate) struct VsmReceiverGenPipeline {
    /// `vsm_generate_receivers` compute entry, specialized against the group-0
    /// layout below. Carries no immediate block; per-frame state is the
    /// binding-2 uniform.
    pipeline: CachedComputePipelineId,
    /// group 0: camera depth read, receiver storage write, per-frame uniform.
    layout: BindGroupLayout,
}

impl VsmReceiverGenPipeline {
    /// The `vsm_generate_receivers` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// The group-0 bind-group layout shared with the receiver-generation
    /// dispatch's bind group.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// group-0 layout mirroring `vsm_receiver_gen.wesl`: the non-filterable camera
/// depth (`textureLoad`ed, binding 0), the read-write receiver `storage` array
/// (binding 1) and the per-frame uniform (binding 2).
fn layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`VsmReceiverGenPipeline`].
pub(crate) fn init_vsm_receiver_gen_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism VSM receiver-gen", &entries);
    let layout = device.create_bind_group_layout("prism VSM receiver-gen", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/vsm_receiver_gen.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism VSM receiver-gen".into()),
        layout: vec![descriptor],
        shader,
        entry_point: Some("vsm_generate_receivers".into()),
        ..Default::default()
    });

    commands.insert_resource(VsmReceiverGenPipeline { pipeline, layout });
}
