//! The water-surface **raster** render pipelines: the `@vertex`/`@fragment`
//! draw that finally turns the solved water fields into visible pixels.
//!
//! Every pipeline the sibling [`super::pipeline`] slice builds is a `@compute`
//! solver that writes device buffers; none of them draw. This slice builds the
//! missing raster half contracted by
//! [`prism_render_architecture::water::gpu::surface_pass`] and
//! [`prism_render_architecture::water::gpu::surface_bindings`]: one shared
//! vertex stage that displaces the surface grid and four per-frontend fragment
//! stages that shade it through the design §5 lighting-response fork
//! (`PBR`/`NPR`/custom/hybrid). The `WESL` source is
//! `shaders/water_surface_raster.wesl`; its entry-point names are byte-identical
//! to the `CPU` contract
//! ([`SurfaceDrawDescriptor::vertex_entry`](prism_render_architecture::water::gpu::SurfaceDrawDescriptor::vertex_entry)
//! and [`fragment_entry`](prism_render_architecture::water::gpu::SurfaceDrawDescriptor::fragment_entry)).
//!
//! ## Why the pipeline is specialized per view
//!
//! The surface draw composites into whatever colour attachment the active view
//! renders into: the main `HDR` `Rgba16Float` texture on screen, or a
//! render-to-texture view's own format. A render pipeline's colour-target format
//! must be byte-compatible with the attachment it writes, so — exactly like the
//! [`super::super::shading::composite`] pass — the pipeline is a
//! [`SpecializedRenderPipeline`] keyed on
//! ([`ShadingFrontend`], [`ExtractedView::target_format`]). The frontend selects
//! the fragment entry point (the §5 lighting fork); the target format keeps the
//! single `@location(0)` colour target compatible with the view it draws into.
//!
//! Specialization happens in [`prepare_water_surface_pipelines`] during
//! [`RenderSystems::Prepare`](bevy_render::RenderSystems::Prepare), following
//! Bevy's own fullscreen passes and the Prism composite: it specializes all four
//! frontends for each visibility-path view and stashes the concrete
//! [`CachedRenderPipelineId`]s on the view as [`ViewWaterSurfacePipelines`], so
//! the raster draw system (following slice) stays a thin recorder that only
//! fetches the pipeline for a body's frontend.
//!
//! ## Pass state (mirrors the `CPU` contract)
//!
//! * **Target** — the view's colour attachment, drawn after the opaque pass and
//!   before post-processing, exactly like `UE5` Single Layer Water, `Crest`, and
//!   `WaveWorks`
//!   ([`WaterRenderTarget::HdrTransparent`](prism_render_architecture::water::gpu::WaterRenderTarget)).
//! * **Blend** — pre-multiplied alpha
//!   (`src.One`/`dst.OneMinusSrcAlpha`, add) on both colour and alpha, matching
//!   [`SurfaceBlend::PremultipliedAlpha`](prism_render_architecture::water::gpu::SurfaceBlend):
//!   the fragment shader emits already-composited refraction/reflection colour.
//! * **Depth** — tests against the opaque depth so submerged geometry occludes
//!   the surface, but never writes
//!   ([`SurfaceDepth`](prism_render_architecture::water::gpu::SurfaceDepth) with
//!   `write = false`). The contract phrases the test as the logical
//!   [`DepthTest::LessEqual`](prism_render_architecture::water::gpu::DepthTest)
//!   ("nearer or equal passes"); this engine renders reverse-Z, so the logical
//!   "nearer or equal" maps to [`CompareFunction::GreaterEqual`] on the device,
//!   exactly as [`super::super::shading::transparent`] does.
//! * **Culling** — disabled: a water surface is viewed from above *and* below
//!   (the camera crosses the waterline), so both windings must raster.
//!
//! ## Vertex plumbing
//!
//! There is **no vertex buffer**. The vertex stage reads the displaced grid out
//! of the four `@group(0)` storage arrays indexed by `@builtin(vertex_index)`
//! (the value an index-buffer draw pulls per invocation), so the pipeline
//! declares an empty [`VertexState::buffers`]. The draw node binds the index
//! buffer and the storage arrays; this pipeline only fixes their layout.

use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::{BindGroupLayoutDescriptor, FragmentState, RenderPipelineDescriptor, VertexState},
};
use bevy_render::{
    render_resource::PipelineCache,
    render_resource::{
        BlendComponent, BlendFactor, BlendOperation, BlendState, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState,
        FrontFace, MultisampleState, PrimitiveState, PrimitiveTopology, SamplerBindingType,
        ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines, StencilState,
        TextureFormat, TextureSampleType,
    },
    view::ExtractedView,
};
use bevy_shader::Shader;

use prism_render_architecture::water::gpu::plan_surface_draw;
use prism_render_architecture::water::ShadingFrontend;

use super::super::shading::ViewVisibilityBuffer;
use crate::lighting::LightBindGroup;

/// The four shading frontends in a stable order; the index into
/// [`ViewWaterSurfacePipelines::ids`] is this slot. `PBR` first keeps the
/// default-frontend pipeline at slot zero.
const FRONTENDS: [ShadingFrontend; 4] = [
    ShadingFrontend::Pbr,
    ShadingFrontend::Npr,
    ShadingFrontend::Custom,
    ShadingFrontend::Hybrid,
];

/// The slot a frontend occupies in [`ViewWaterSurfacePipelines::ids`].
const fn frontend_slot(frontend: ShadingFrontend) -> usize {
    match frontend {
        ShadingFrontend::Pbr => 0,
        ShadingFrontend::Npr => 1,
        ShadingFrontend::Custom => 2,
        ShadingFrontend::Hybrid => 3,
    }
}

/// Builds the water-surface `@group(0)` layout entries (seven bindings), in the
/// exact `@binding(n)` order
/// [`SurfaceBinding`](prism_render_architecture::water::gpu::SurfaceBinding)
/// declares: the per-view uniform, the four read-only per-vertex storage arrays
/// (`base_positions`/`surface_uvs`/`displacement`/`normal_foam`), the refraction
/// scene-colour texture, and its filtering sampler.
///
/// All seven are declared [`ShaderStages::VERTEX_FRAGMENT`]: the geometry stage
/// reads the uniform and the four storage arrays, the lighting stage reads the
/// uniform plus the refraction texture/sampler, and a visibility superset is a
/// legal and cheaper declaration than two separate masks.
fn surface_layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::VERTEX_FRAGMENT,
        (
            // @binding(0) WaterSurfaceView uniform.
            uniform_buffer_sized(false, None),
            // @binding(1..=4) base_positions / surface_uvs / displacement /
            // normal_foam, one `array<vec4<f32>>` each.
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            // @binding(5) already-rendered opaque scene colour (refraction).
            texture_2d(TextureSampleType::Float { filterable: true }),
            // @binding(6) the filtering sampler the refraction lookup uses.
            sampler(SamplerBindingType::Filtering),
        ),
    )
}

/// Pre-multiplied-alpha blend for the transparent water composite:
/// `dst = src + dst * (1 - src.a)` on both colour and alpha. The fragment
/// shader emits already-composited (pre-multiplied) refraction/reflection
/// colour, matching
/// [`SurfaceBlend::PremultipliedAlpha`](prism_render_architecture::water::gpu::SurfaceBlend)
/// and the `UE5` Single Layer Water / `Crest` compositing path.
const fn premultiplied_alpha_blend() -> BlendState {
    let component = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
    BlendState {
        color: component,
        alpha: component,
    }
}

/// Specialization key for the water-surface raster pipeline.
///
/// Two axes vary at run time: the shading `frontend` selects the fragment entry
/// point (the §5 lighting-response fork), and `target_format` is the colour
/// format of the view target this draw composites into — `Rgba16Float` on an
/// `HDR` screen, or a render-to-texture view's own format. Keying on both keeps
/// the fragment stage forked correctly *and* the single `@location(0)` colour
/// target byte-compatible with the attachment it blends into.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub(crate) struct WaterSurfaceKey {
    /// Lighting-response frontend; selects the fragment entry point.
    pub(crate) frontend: ShadingFrontend,
    /// Format of the view target this pass renders into.
    pub(crate) target_format: TextureFormat,
}

/// Render-pipeline specializer + the owned shared `@group(0)` layout for the
/// water-surface raster draw.
///
/// Built once at `RenderStartup` by [`init_water_surface_pipelines`]; read by
/// [`prepare_water_surface_pipelines`] (which specializes the four frontends per
/// view) and by the raster draw system (following slice), which resolves the
/// concrete bind-group layout from the [`PipelineCache`] via [`Self::layout`] to
/// build each body's per-body bind group.
#[derive(Resource)]
pub(crate) struct WaterSurfacePipelines {
    /// The shared `@group(0)` bind-group layout descriptor every surface draw
    /// binds. Stored as a descriptor (not a concrete handle) so the raster draw
    /// system resolves the live [`BindGroupLayout`](bevy_render::render_resource::BindGroupLayout)
    /// from the [`PipelineCache`] with the *same* descriptor the pipeline
    /// specializes against, and the two can never drift.
    pub(crate) layout: BindGroupLayoutDescriptor,
    /// The engine's shared `@group(1)` light-table layout descriptor (all
    /// directionals + punctuals + the image-based `LightEnvironment`), cloned
    /// from [`LightBindGroup`] so the surface fragment stage reads the *same*
    /// light tables the opaque `shading_resolve` pass does. Resolved to the live
    /// [`BindGroupLayout`](bevy_render::render_resource::BindGroupLayout) through
    /// the [`PipelineCache`] with the same descriptor the resolve pipeline uses,
    /// so the two can never drift.
    pub(crate) light_layout: BindGroupLayoutDescriptor,
    /// The water-surface `@group(2)` virtual-shadow-map layout descriptor (page
    /// table + physical atlas + sampler + params), a re-numbered twin of the
    /// opaque resolve pass's VSM group. Built from
    /// [`super::surface_vsm::vsm_layout_entries`] so the primary directional
    /// light's shadow is sampled from the identical demand-paged atlas. Resolved
    /// to the live [`BindGroupLayout`](bevy_render::render_resource::BindGroupLayout)
    /// through the [`PipelineCache`] with the same descriptor the draw node binds.
    pub(crate) vsm_layout: BindGroupLayoutDescriptor,
    /// The water-surface `@group(3)` screen-space-reflection layout descriptor
    /// (reverse-Z Hi-Z pyramid + march config uniform). Built from
    /// [`super::surface_ssr::ssr_layout_entries`] so the transparent surface can
    /// march the same depth pyramid the opaque `SSR` prepass produces. Resolved
    /// to the live [`BindGroupLayout`](bevy_render::render_resource::BindGroupLayout)
    /// through the [`PipelineCache`] with the same descriptor the draw node binds.
    pub(crate) ssr_layout: BindGroupLayoutDescriptor,
    /// The embedded `water_surface_raster.wesl` module both stages compile from.
    pub(crate) shader: Handle<Shader>,
}

impl SpecializedRenderPipeline for WaterSurfacePipelines {
    type Key = WaterSurfaceKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // The entry points come straight from the deterministic `CPU` contract,
        // so the host pipeline and the device shader can never drift on names.
        let draw = plan_surface_draw(key.frontend);
        RenderPipelineDescriptor {
            label: Some(format!("prism water surface {:?}", key.frontend).into()),
            layout: vec![
                self.layout.clone(),
                self.light_layout.clone(),
                self.vsm_layout.clone(),
                self.ssr_layout.clone(),
            ],
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some(draw.vertex_entry().into()),
                // No vertex buffer: the vertex stage indexes the `@group(0)`
                // storage arrays by `@builtin(vertex_index)`.
                buffers: vec![],
                ..Default::default()
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Ccw,
                // The surface is viewed from above and below (the camera crosses
                // the waterline), so neither winding is culled.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                // Reverse-Z: the logical `LessEqual` ("nearer or equal passes")
                // contract maps to `GreaterEqual` on the device. Never write -
                // the transparent surface must not occlude later transparent
                // draws (contract `SurfaceDepth { write: false }`).
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            // The Prism visibility path only runs when MSAA is disabled, so the
            // surface target is always single-sampled.
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some(draw.fragment_entry().into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(premultiplied_alpha_blend()),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

/// The four specialized water-surface pipelines chosen for one view, one per
/// [`ShadingFrontend`] (indexed by [`frontend_slot`]), all keyed on the view's
/// target format.
///
/// Present only on views the surface draw should run for (the Prism visibility
/// path is live for them this frame). The raster draw system (following slice)
/// reads it to fetch the concrete render pipeline for each body's frontend.
#[derive(Component)]
pub(crate) struct ViewWaterSurfacePipelines {
    /// One cached pipeline id per frontend, in [`FRONTENDS`] / [`frontend_slot`]
    /// order.
    ids: [CachedRenderPipelineId; 4],
}

impl ViewWaterSurfacePipelines {
    /// The specialized pipeline id that shades the given frontend for this view.
    #[must_use]
    pub(crate) fn id_for(&self, frontend: ShadingFrontend) -> CachedRenderPipelineId {
        self.ids[frontend_slot(frontend)]
    }
}

/// `RenderStartup` initializer: builds the shared surface bind-group layout
/// descriptor and loads the embedded raster shader, then inserts the
/// [`WaterSurfacePipelines`] specializer resource.
///
/// The raster shader must be registered as an embedded asset before this runs
/// (see [`super::plugin`]); `load_embedded_asset!` resolves it by its path
/// relative to this file. It also depends on [`LightBindGroup`] already being
/// initialized (the plugin orders this `.after(init_gpu_resource::<LightBindGroup>)`)
/// so the shared light-table layout descriptor can be cloned. The concrete
/// pipelines are specialized later, per view, in
/// [`prepare_water_surface_pipelines`].
pub(crate) fn init_water_surface_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    light_bindings: Res<LightBindGroup>,
) {
    let entries = surface_layout_entries();
    let layout = BindGroupLayoutDescriptor::new("prism water surface", &entries);

    // Reuse the engine's shared light-table layout verbatim so the surface
    // fragment stage binds the identical directional/punctual/environment
    // tables the opaque resolve pass reads - no second light upload.
    let light_layout = light_bindings.layout_descriptor.clone();

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/water_surface_raster.wesl"
    );

    // The `@group(2)` VSM layout is a re-numbered twin of the opaque resolve
    // pass's virtual-shadow-map group; the draw node binds either the resident
    // page table + atlas or the format-correct fallback against this descriptor.
    let vsm_layout = BindGroupLayoutDescriptor::new(
        "prism water surface vsm",
        &super::surface_vsm::vsm_layout_entries(),
    );

    // The `@group(3)` SSR layout: the reverse-Z Hi-Z pyramid plus the march
    // config uniform. The draw node binds either the resident `ViewSsrTextures`
    // pyramid or the 1x1 fallback against this descriptor.
    let ssr_layout = BindGroupLayoutDescriptor::new(
        "prism water surface ssr",
        &super::surface_ssr::ssr_layout_entries(),
    );

    commands.insert_resource(WaterSurfacePipelines {
        layout,
        light_layout,
        vsm_layout,
        ssr_layout,
        shader,
    });
}

/// `RenderSystems::Prepare` system specializing the four surface pipelines per
/// view.
///
/// Only views that carry a [`ViewVisibilityBuffer`] (the Prism visibility path
/// is live for them this frame) get pipelines: MSAA / disabled views never
/// allocate the buffer, so they are skipped here and the raster draw system
/// finds no [`ViewWaterSurfacePipelines`] to run. All four frontends are
/// specialized up front — a view may host several bodies with different
/// frontends — and keyed on [`ExtractedView::target_format`] so each colour
/// target stays byte-compatible with the attachment the view renders into.
pub(crate) fn prepare_water_surface_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<WaterSurfacePipelines>>,
    pipeline: Res<WaterSurfacePipelines>,
    views: Query<(Entity, &ExtractedView), With<ViewVisibilityBuffer>>,
) {
    for (entity, view) in &views {
        let ids = FRONTENDS.map(|frontend| {
            pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                WaterSurfaceKey {
                    frontend,
                    target_format: view.target_format,
                },
            )
        });
        commands
            .entity(entity)
            .insert(ViewWaterSurfacePipelines { ids });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::SurfaceBinding;

    /// A specializer with a stub layout + shader, enough to exercise the pure
    /// `specialize` logic on the `CPU` without a device.
    fn fixture() -> WaterSurfacePipelines {
        let entries = surface_layout_entries();
        // A structurally equivalent stand-in for the engine's shared light-table
        // layout (3x read-only storage, visible to compute+fragment); production
        // clones the real descriptor off `LightBindGroup`.
        let light_entries = BindGroupLayoutEntries::<3>::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        );
        WaterSurfacePipelines {
            layout: BindGroupLayoutDescriptor::new("prism water surface", &entries),
            light_layout: BindGroupLayoutDescriptor::new("prism lights", &light_entries),
            vsm_layout: BindGroupLayoutDescriptor::new(
                "prism water surface vsm",
                &crate::water::surface_vsm::vsm_layout_entries(),
            ),
            ssr_layout: BindGroupLayoutDescriptor::new(
                "prism water surface ssr",
                &crate::water::surface_ssr::ssr_layout_entries(),
            ),
            shader: Handle::default(),
        }
    }

    #[test]
    fn layout_entries_match_the_surface_binding_contract() {
        let entries = surface_layout_entries();
        assert_eq!(entries.len(), SurfaceBinding::ALL.len());
        for (slot, binding) in SurfaceBinding::ALL.iter().enumerate() {
            let entry = &entries[slot];
            assert_eq!(
                entry.binding,
                binding.index(),
                "layout entry {slot} must sit at the shader's @binding slot"
            );
            // Declared as the vertex+fragment superset, so every stage the
            // contract marks as a reader is actually visible.
            if binding.visible_in_vertex() {
                assert!(
                    entry.visibility.contains(ShaderStages::VERTEX),
                    "{binding:?} is read in the vertex stage"
                );
            }
            if binding.visible_in_fragment() {
                assert!(
                    entry.visibility.contains(ShaderStages::FRAGMENT),
                    "{binding:?} is read in the fragment stage"
                );
            }
        }
    }

    #[test]
    fn frontend_slots_are_unique_and_contiguous() {
        let mut seen = [false; 4];
        for frontend in FRONTENDS {
            let slot = frontend_slot(frontend);
            assert!(slot < seen.len());
            assert!(!seen[slot], "frontend slot {slot} assigned twice");
            seen[slot] = true;
        }
        assert!(seen.iter().all(|&s| s), "every slot must be assigned");
    }

    #[test]
    fn premultiplied_blend_reads_destination_by_one_minus_src_alpha() {
        let blend = premultiplied_alpha_blend();
        for component in [blend.color, blend.alpha] {
            assert_eq!(component.src_factor, BlendFactor::One);
            assert_eq!(component.dst_factor, BlendFactor::OneMinusSrcAlpha);
            assert_eq!(component.operation, BlendOperation::Add);
        }
    }

    #[test]
    fn specialize_targets_the_key_format_and_keeps_the_transparent_pass_state() {
        // The colour target must adopt whatever format the view renders into, so
        // a render-to-texture view (here `Rgba8UnormSrgb`) is byte-compatible
        // with its attachment, not pinned to the on-screen `HDR` format.
        let pipelines = fixture();
        let desc = pipelines.specialize(WaterSurfaceKey {
            frontend: ShadingFrontend::Pbr,
            target_format: TextureFormat::Rgba8UnormSrgb,
        });

        let fragment = desc
            .fragment
            .expect("the surface draw has a fragment stage");
        let target = fragment.targets[0]
            .as_ref()
            .expect("the surface draw writes one colour target");
        assert_eq!(target.format, TextureFormat::Rgba8UnormSrgb);
        // Transparent composite: pre-multiplied alpha, reverse-Z depth test with
        // no depth write, both windings rastered.
        assert!(target.blend.is_some());
        let depth = desc.depth_stencil.expect("the surface tests opaque depth");
        assert_eq!(depth.depth_write_enabled, Some(false));
        assert_eq!(depth.depth_compare, Some(CompareFunction::GreaterEqual));
        assert_eq!(desc.primitive.cull_mode, None);
        assert_eq!(desc.primitive.topology, PrimitiveTopology::TriangleList);
        // Four bind-group layouts: the per-body @group(0) surface layout, the
        // shared @group(1) engine light table the fragment stage samples, the
        // @group(2) virtual-shadow-map twin the primary directional light reads,
        // and the @group(3) screen-space-reflection Hi-Z pyramid + march config.
        assert_eq!(desc.layout.len(), 4);
    }

    #[test]
    fn specialize_forks_the_fragment_entry_per_frontend() {
        let pipelines = fixture();
        for frontend in FRONTENDS {
            let desc = pipelines.specialize(WaterSurfaceKey {
                frontend,
                target_format: TextureFormat::Rgba16Float,
            });
            let fragment = desc.fragment.expect("a fragment stage per frontend");
            let expected = plan_surface_draw(frontend).fragment_entry();
            assert_eq!(
                fragment.entry_point.as_deref(),
                Some(expected),
                "{frontend:?} must bind its own fragment entry point"
            );
            // The vertex stage is shared across every frontend.
            assert_eq!(
                desc.vertex.entry_point.as_deref(),
                Some(plan_surface_draw(frontend).vertex_entry())
            );
        }
    }
}
