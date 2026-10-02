//! The outline compute pipeline, its owned group-0 layout, and the
//! `RenderStartup` initializer that queues it.
//!
//! Mirrors [`super::super::color_grade::pipeline`]: one compute entry point
//! (`outline_main`) from `outline.wesl`, specialized against its group-0 layout
//! and its single immediate block. The layout binds the pre-exposed scene
//! colour (non-filterable float, `textureLoad`ed per pixel) at binding `0`, the
//! write-only `rgba16float` output at binding `1`, the SSR reverse-Z device
//! depth at binding `2` and the SSR view-space normal at binding `3`, matching
//! the shader's `sequential` `{0, 1, 2, 3}` bindings.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_2d, texture_storage_2d},
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

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::GpuOutlineParams;

/// The outline compute pipeline and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct OutlinePipeline {
    /// `outline_main` entry: full-screen edge detection over the geometry buffer
    /// composited over the scene colour.
    pipeline: CachedComputePipelineId,
    /// group 0 for `outline_main`: scene-colour + depth + normal reads, output
    /// storage write.
    layout: BindGroupLayout,
}

impl OutlinePipeline {
    /// The `outline_main` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// group-0 layout for the `outline_main` dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// `outline_main` group-0 layout: the pre-exposed scene colour (non-filterable
/// float, `textureLoad`ed per pixel) at binding `0`, the write-only
/// `rgba16float` output at binding `1`, the SSR reverse-Z device depth at
/// binding `2` and the SSR view-space normal at binding `3` (both non-filterable
/// floats, `textureLoad`ed across the four-neighbour cross).
fn layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// `RenderStartup` initializer for [`OutlinePipeline`].
pub(crate) fn init_outline_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism outline", &entries);
    let layout = device.create_bind_group_layout("prism outline", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/outline.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism outline".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuOutlineParams>() as u32,
        shader,
        entry_point: Some("outline_main".into()),
        ..Default::default()
    });

    commands.insert_resource(OutlinePipeline { pipeline, layout });
}
