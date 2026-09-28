//! Render pipeline + group-0 layout for the WBOIT composite.
//!
//! The transparent forward pass accumulated weighted premultiplied colour into
//! [`ViewOitTargets`] (accumulation `Rgba16Float` + revealage `R16Float`). This
//! fullscreen pass (`shaders/oit.wesl`, entry `composite`) reads both targets
//! back and, for every pixel, emits `vec4(average_rgb, coverage)` as the blend
//! source. The pipeline blend is fixed-function `SrcAlpha`/`OneMinusSrcAlpha`
//! (add) on both colour and alpha, so the hardware reproduces
//! `OitAccumulation::resolve` over the already-composited opaque view target
//! without the shader ever sampling that background. Pixels with no transparent
//! coverage emit coverage 0 and leave the target untouched.
//!
//! Only group 0 is owned here (the two WBOIT targets, both read by integer
//! `textureLoad`). The pipeline is specialized on the view target colour format
//! so the single colour target always matches the surface it blends into. This
//! mirrors [`super::super::composite::pipeline`] closely; the only substantive
//! differences are the two-texture layout and the enabled blend state.

use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{binding_types::texture_2d, BindGroupLayoutEntries},
    descriptor::{
        BindGroupLayoutDescriptor, FragmentState, RenderPipelineDescriptor, VertexState,
    },
};
use bevy_render::{
    render_resource::{
        BlendComponent, BlendFactor, BlendOperation, BlendState, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, MultisampleState, PipelineCache, PrimitiveState,
        ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
        TextureSampleType,
    },
    view::ExtractedView,
};
use bevy_shader::Shader;

use super::targets::ViewOitTargets;

/// Specialization key for the OIT composite pipeline: only the destination
/// colour format varies, exactly as for the opaque composite.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub(crate) struct OitCompositeKey {
    /// Format of the view target this pass blends into.
    pub(crate) target_format: TextureFormat,
}

/// Render pipeline specializer + the owned group-0 layout for the WBOIT
/// composite.
#[derive(Resource)]
pub(crate) struct OitCompositePipeline {
    /// group 0: `oit_accum` (float) + `oit_revealage` (float), both sampled by
    /// integer `textureLoad`. Stored as a descriptor so the concrete layout is
    /// resolved from the [`PipelineCache`] and can never drift from the pipeline
    /// it feeds.
    pub(crate) layout: BindGroupLayoutDescriptor,
    /// The `oit.wesl` module, providing the fullscreen `vertex` and the
    /// `composite` entry points.
    pub(crate) shader: Handle<Shader>,
}

/// The specialized OIT composite pipeline chosen for one view, keyed on its
/// target format. Present only on views the composite should run for;
/// [`super::composite_node`] reads it to fetch the concrete render pipeline.
#[derive(Component)]
pub(crate) struct ViewOitCompositePipelineId(pub(crate) CachedRenderPipelineId);

/// group-0 layout: the two WBOIT targets, both fragment-visible non-filterable
/// float textures read only by integer `textureLoad` (no sampler).
fn oit_composite_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// `RenderStartup` initializer: builds the group-0 layout descriptor and loads
/// the embedded `oit.wesl` module. No device dependency beyond the shared asset
/// server.
pub(crate) fn init_oit_composite_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let entries = oit_composite_layout_entries();
    let layout = BindGroupLayoutDescriptor::new("prism oit composite", &entries);
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/oit.wesl");
    commands.insert_resource(OitCompositePipeline { layout, shader });
}

impl SpecializedRenderPipeline for OitCompositePipeline {
    type Key = OitCompositeKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // SrcAlpha/OneMinusSrcAlpha add on both colour and alpha: with the
        // fragment source (average_rgb, coverage) this evaluates to
        // average*coverage + dst*(1 - coverage), i.e. OitAccumulation::resolve
        // over the bound opaque view target.
        let blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        };
        RenderPipelineDescriptor {
            label: Some("prism oit composite".into()),
            layout: vec![self.layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vertex".into()),
                buffers: vec![],
                ..Default::default()
            },
            primitive: PrimitiveState::default(),
            // The pass blends into the view target colour attachment and never
            // touches depth.
            depth_stencil: None,
            // The Prism visibility path only runs single-sampled.
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("composite".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(blend),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

/// `RenderSystems::Prepare` system specializing the OIT composite pipeline per
/// view. Only views that carry a [`ViewOitTargets`] this frame (the transparent
/// path is live for them) get a pipeline; the rest are skipped and the node
/// finds no [`ViewOitCompositePipelineId`] to run.
pub(crate) fn prepare_oit_composite_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<OitCompositePipeline>>,
    pipeline: Res<OitCompositePipeline>,
    views: Query<(Entity, &ExtractedView), With<ViewOitTargets>>,
) {
    for (entity, view) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            OitCompositeKey {
                target_format: view.target_format,
            },
        );
        commands
            .entity(entity)
            .insert(ViewOitCompositePipelineId(id));
    }
}
