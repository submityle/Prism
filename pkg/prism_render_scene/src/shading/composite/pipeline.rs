//! Render pipeline + bind-group layout for the HDR -> view-target composite.
//!
//! The resolve compute pass wrote linear pre-exposure HDR radiance into the
//! per-view `scene_color` storage texture for every *covered* pixel.  This
//! fullscreen pass (`shaders/composite.wesl`) reads that texture back and
//! copies it into the core-3d view target *after* the main pass has cleared
//! the target and drawn any non-Prism geometry, and *before* Bevy's
//! tonemapping node maps it to the display.  Uncovered pixels are `discard`ed
//! so the main pass' clear/background survives.
//!
//! Only one bind group is owned here (**group 0** = the three resolve outputs,
//! all read by integer `textureLoad`).  The pipeline is specialized on the view
//! target's colour format so the fragment stage's single colour target always
//! matches the surface it renders into (HDR `Rgba16Float` on screen, or the
//! target texture's format when rendering to a texture).
//!
//! Following Bevy's own fullscreen passes (`tonemapping`,
//! `background_motion_vectors`), specialization happens in the
//! [`prepare_shading_composite_pipelines`] system during
//! [`RenderSystems::Prepare`](bevy_render::RenderSystems::Prepare); the concrete
//! [`CachedRenderPipelineId`] is stashed on the view as
//! [`ViewCompositePipelineId`] so the graph node stays a thin recorder.

use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_read_only_sized, texture_2d},
        BindGroupLayoutEntries,
    },
    descriptor::{BindGroupLayoutDescriptor, FragmentState, RenderPipelineDescriptor, VertexState},
};
use bevy_render::{
    render_resource::{
        CachedRenderPipelineId, ColorTargetState, ColorWrites, MultisampleState, PipelineCache,
        PrimitiveState, ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines,
        TextureFormat, TextureSampleType,
    },
    view::ExtractedView,
};
use bevy_shader::Shader;

use super::super::resources::ViewVisibilityBuffer;

/// Specialization key for the composite pipeline.
///
/// The only run-time-variable state is the destination colour format: on-screen
/// views composite into the HDR `Rgba16Float` main texture, while
/// render-to-texture views inherit their target's format.  Keying on it keeps
/// the single `@location(0)` colour target byte-compatible with the attachment.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub(crate) struct ShadingCompositeKey {
    /// Format of the view target this pass renders into.
    pub(crate) target_format: TextureFormat,
}

/// Render pipeline specializer + the owned group-0 layout for the composite.
#[derive(Resource)]
pub(crate) struct ShadingCompositePipeline {
    /// group 0: `scene_color` (float) + `visibility_ids`/`visibility_metadata`
    /// (uint), all sampled by integer `textureLoad`, then the exposure-state
    /// storage buffer (read-only) the fragment multiplies radiance by.  Stored as a descriptor so
    /// the concrete [`BindGroupLayout`](bevy_render::render_resource::BindGroupLayout)
    /// is resolved from the [`PipelineCache`] and can never drift from the
    /// pipeline it feeds.
    pub(crate) layout: BindGroupLayoutDescriptor,
    /// The `composite.wesl` module, providing both the fullscreen `vertex` and
    /// the copy-or-discard `fragment` entry points.
    pub(crate) shader: Handle<Shader>,
}

/// The specialized composite pipeline chosen for one view, keyed on its target
/// format.  Present only on views the composite should run for; [`super::node`]
/// reads it to fetch the concrete render pipeline.
#[derive(Component)]
pub(crate) struct ViewCompositePipelineId(pub(crate) CachedRenderPipelineId);

/// group-0 layout: the three resolve outputs (all fragment-visible textures)
/// then the persistent exposure state. `scene_color` is a non-filterable float
/// texture (integer `textureLoad`, no sampler); the two visibility targets are
/// `u32` textures; the exposure state is a read-only storage buffer holding the
/// eye-adaptation multiplier the fragment applies to radiance.
fn composite_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer: builds the group-0 layout descriptor and loads
/// the embedded `composite.wesl` module.  No device dependency beyond the
/// shared asset server, so it can run without ordering against the material or
/// light bind groups.
pub(crate) fn init_shading_composite_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let entries = composite_layout_entries();
    let layout = BindGroupLayoutDescriptor::new("prism composite", &entries);
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/composite.wesl");
    commands.insert_resource(ShadingCompositePipeline { layout, shader });
}

impl SpecializedRenderPipeline for ShadingCompositePipeline {
    type Key = ShadingCompositeKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("prism shading composite".into()),
            layout: vec![self.layout.clone()],
            immediate_size: 0,
            // Fullscreen triangle generated from `@builtin(vertex_index)`; no
            // vertex buffers are bound.
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
                ..Default::default()
            },
            primitive: PrimitiveState::default(),
            // The pass writes straight into the view target's colour attachment
            // and never touches depth, so no depth-stencil state is attached.
            depth_stencil: None,
            // The Prism visibility path only runs when MSAA is disabled, so the
            // composite target is always single-sampled.
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    // Opaque copy: covered pixels overwrite the target, uncovered
                    // pixels are discarded in the shader, so no blending.
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

/// `RenderSystems::Prepare` system specializing the composite pipeline per view.
///
/// Only views that carry a [`ViewVisibilityBuffer`] (i.e. the Prism visibility
/// path is live for them this frame) get a pipeline: MSAA / disabled views
/// never allocate the buffer, so they are skipped here and the graph node finds
/// no [`ViewCompositePipelineId`] to run.  Keying on
/// [`ExtractedView::target_format`] keeps the colour target byte-compatible with
/// whatever surface the view renders into.
pub(crate) fn prepare_shading_composite_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ShadingCompositePipeline>>,
    pipeline: Res<ShadingCompositePipeline>,
    views: Query<(Entity, &ExtractedView), With<ViewVisibilityBuffer>>,
) {
    for (entity, view) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            ShadingCompositeKey {
                target_format: view.target_format,
            },
        );
        commands.entity(entity).insert(ViewCompositePipelineId(id));
    }
}
