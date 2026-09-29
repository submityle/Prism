//! The lens-flare compute pipeline, its owned group-0 layout, and the
//! `RenderStartup` initializer that queues it.
//!
//! Mirrors [`super::super::color_grade::pipeline`]: one compute entry point
//! (`lens_flare_main`) from `lens_flare.wesl`, specialized against its group-0
//! layout and its single immediate block. The layout binds the pre-exposed
//! scene colour (non-filterable float, `textureLoad`ed per pixel — the ghost /
//! halo taps do their own bilinear filtering by hand so no `sampler` is needed)
//! at binding `0` and the write-only `rgba16float` output at binding `1`,
//! matching the shader's `sequential` `{0, 1}` bindings.

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

use super::abi::GpuLensFlareParams;
use super::super::resources::SCENE_COLOR_FORMAT;

/// The lens-flare compute pipeline and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct LensFlarePipeline {
    /// `lens_flare_main` entry: full-screen flare over the scene colour.
    pipeline: CachedComputePipelineId,
    /// group 0 for `lens_flare_main`: scene-colour read + output storage write.
    layout: BindGroupLayout,
}

impl LensFlarePipeline {
    /// The `lens_flare_main` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// group-0 layout for the `lens_flare_main` dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// `lens_flare_main` group-0 layout: the pre-exposed scene colour
/// (non-filterable float, `textureLoad`ed per pixel) at binding `0`, then the
/// write-only `rgba16float` output at binding `1`.
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`LensFlarePipeline`].
pub(crate) fn init_lens_flare_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism lens flare", &entries);
    let layout = device.create_bind_group_layout("prism lens flare", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/lens_flare.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism lens flare".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuLensFlareParams>() as u32,
        shader,
        entry_point: Some("lens_flare_main".into()),
        ..Default::default()
    });

    commands.insert_resource(LensFlarePipeline { pipeline, layout });
}
