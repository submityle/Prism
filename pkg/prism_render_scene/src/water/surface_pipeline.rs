//! The water-surface **raster** render pipelines: the `@vertex`/`@fragment`
//! draw that finally turns the solved water fields into visible `HDR` pixels.
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
//! This slice owns only the pipeline *resources* (the shared `@group(0)` layout
//! and the four cached render pipelines, one per frontend). The draw node that
//! allocates each body's vertex/index buffers, builds its per-body bind group,
//! and records `draw_indexed` into the main `HDR` transparent target lands in
//! the following slice; it reads [`WaterSurfacePipelines`] built here.
//!
//! ## Pass state (mirrors the `CPU` contract)
//!
//! * **Target** — the main `HDR` colour ([`WATER_SURFACE_COLOR_FORMAT`],
//!   `Rgba16Float`), drawn after the opaque pass and before post-processing,
//!   exactly like `UE5` Single Layer Water, `Crest`, and `WaveWorks`
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

use bevy_asset::{load_embedded_asset, Handle};
use bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, BlendComponent, BlendFactor, BlendOperation, BlendState,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
        DepthStencilState, FragmentState, FrontFace, MultisampleState, PipelineCache,
        PrimitiveState, PrimitiveTopology, RenderPipelineDescriptor, SamplerBindingType,
        ShaderStages, StencilState, TextureFormat, TextureSampleType, VertexState,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use prism_render_architecture::water::gpu::plan_surface_draw;
use prism_render_architecture::water::ShadingFrontend;

/// Colour format the water-surface draw composites into: the main `HDR` target.
///
/// `Rgba16Float` is the `HDR` colour format every Prism view target carries, so
/// the surface pipeline is layout-compatible with the main pass colour
/// attachment it blends into. The draw node must bind an `HDR` view target.
pub(crate) const WATER_SURFACE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The four shading frontends in a stable order; the index into
/// [`WaterSurfacePipelines::pipelines`] is this slot. `PBR` first keeps the
/// default-frontend pipeline at slot zero.
const FRONTENDS: [ShadingFrontend; 4] = [
    ShadingFrontend::Pbr,
    ShadingFrontend::Npr,
    ShadingFrontend::Custom,
    ShadingFrontend::Hybrid,
];

/// The slot a frontend occupies in [`WaterSurfacePipelines::pipelines`].
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

/// The four cached water-surface render pipelines (one per shading frontend)
/// plus the shared `@group(0)` layout and the raster shader handle.
///
/// Built once at `RenderStartup` by [`init_water_surface_pipelines`]; read by
/// the raster draw node (next slice), which binds [`Self::surface_layout`] into
/// a per-body bind group and keys its draw on [`Self::pipeline_for`].
#[derive(Resource)]
pub(crate) struct WaterSurfacePipelines {
    /// The shared `@group(0)` bind-group layout every surface draw binds.
    // Read by the raster draw node (following slice) to build each body's
    // per-body bind group; nothing in this slice reads it back.
    #[expect(
        dead_code,
        reason = "bound by the water-surface raster draw node that lands in the following slice; this slice only builds the layout"
    )]
    pub(crate) surface_layout: BindGroupLayout,
    /// One cached render pipeline per [`ShadingFrontend`], indexed by
    /// [`frontend_slot`].
    pipelines: [CachedRenderPipelineId; 4],
    /// The embedded `water_surface_raster.wesl` module both stages compile from.
    // Retained so the draw node (and future hot-reload) can resolve the module;
    // not re-read within this slice.
    #[expect(
        dead_code,
        reason = "retained for the raster draw node and shader hot-reload in the following slice"
    )]
    pub(crate) shader: Handle<Shader>,
}

impl WaterSurfacePipelines {
    /// The cached render pipeline that shades the given frontend.
    #[must_use]
    #[expect(
        dead_code,
        reason = "queried by the water-surface raster draw node that lands in the following slice"
    )]
    pub(crate) fn pipeline_for(&self, frontend: ShadingFrontend) -> CachedRenderPipelineId {
        self.pipelines[frontend_slot(frontend)]
    }
}

/// `RenderStartup` initializer: creates the shared surface bind-group layout and
/// queues the four per-frontend render pipelines into the [`PipelineCache`].
///
/// The raster shader must be registered as an embedded asset before this runs
/// (see [`super::plugin`]); `load_embedded_asset!` resolves it by its path
/// relative to this file. Each frontend's `WESL` entry points come straight
/// from the deterministic `CPU` contract
/// ([`plan_surface_draw`]), so the host pipeline and the device shader can never
/// drift on entry-point names.
pub(crate) fn init_water_surface_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = surface_layout_entries();
    // The descriptor the pipeline layout references, and the concrete device
    // handle the draw node binds a per-body bind group against. Mirrors the
    // compute slice, which likewise keeps both forms.
    let layout_descriptor = BindGroupLayoutDescriptor::new("prism water surface", &entries);
    let surface_layout = device.create_bind_group_layout("prism water surface", &entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/water_surface_raster.wesl"
    );

    let blend = premultiplied_alpha_blend();

    // One specialization per frontend. The vertex stage is shared (the displaced
    // mesh is frontend-agnostic, §5); only the fragment entry forks.
    let pipelines = FRONTENDS.map(|frontend| {
        let draw = plan_surface_draw(frontend);
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(format!("prism water surface {frontend:?}").into()),
            layout: vec![layout_descriptor.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: shader.clone(),
                entry_point: Some(draw.vertex_entry().into()),
                // No vertex buffer: the vertex stage indexes the `@group(0)`
                // storage arrays by `@builtin(vertex_index)`.
                buffers: vec![],
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                entry_point: Some(draw.fragment_entry().into()),
                targets: vec![Some(ColorTargetState {
                    format: WATER_SURFACE_COLOR_FORMAT,
                    blend: Some(blend),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
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
            multisample: MultisampleState::default(),
            ..Default::default()
        })
    });

    commands.insert_resource(WaterSurfacePipelines {
        surface_layout,
        pipelines,
        shader,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::SurfaceBinding;

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
    fn surface_color_format_is_hdr() {
        assert_eq!(WATER_SURFACE_COLOR_FORMAT, TextureFormat::Rgba16Float);
    }
}
