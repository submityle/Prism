//! The tone-map compute pipeline, its owned group-0 layout, and the
//! `RenderStartup` initializer that queues it.
//!
//! Mirrors [`super::super::gamut_map::pipeline`]: one compute entry point
//! (`tonemap_main`) from `tonemap.wesl`, specialized against its group-0 layout
//! and its single immediate block. The layout binds the pre-exposed scene colour
//! (non-filterable float, `textureLoad`ed per pixel) at binding `0` and the
//! write-only `rgba16float` output at binding `1`, matching the shader's
//! `sequential` `{0, 1}` bindings.

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
use super::abi::GpuTonemapParams;

/// The tone-map compute pipeline and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct TonemapPipeline {
    /// `tonemap_main` entry: full-screen tone map of the pre-exposed scene colour.
    pipeline: CachedComputePipelineId,
    /// group 0 for `tonemap_main`: scene-colour read + output storage write.
    layout: BindGroupLayout,
}

impl TonemapPipeline {
    /// The `tonemap_main` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// group-0 layout for the `tonemap_main` dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// `tonemap_main` group-0 layout: the pre-exposed scene colour (non-filterable
/// float, `textureLoad`ed per pixel) at binding `0`, then the write-only
/// `rgba16float` output at binding `1`.
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`TonemapPipeline`].
pub(crate) fn init_tonemap_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism tonemap", &entries);
    let layout = device.create_bind_group_layout("prism tonemap", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/tonemap.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism tonemap".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuTonemapParams>() as u32,
        shader,
        entry_point: Some("tonemap_main".into()),
        ..Default::default()
    });

    commands.insert_resource(TonemapPipeline { pipeline, layout });
}
