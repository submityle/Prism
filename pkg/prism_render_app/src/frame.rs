//! Building the frame's render graph.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;

use prism_ecs::prelude::Resource;
use prism_ecs::world::World;
use prism_render_driver::{
    BufferId, BufferState, BufferUsages, Color, DepthLoadOp, TextureFormat, TextureId,
    TextureState, TextureUsages, TextureViewId,
};
use prism_render_gpudriven::{CompiledGraphGpuGraphImport, CompiledGraphGpuResourceSlot};
use prism_render_graph::{
    BufferDesc, BufferHandle, ImportedBuffer, ImportedTexture, LogicalResource, PassFlags,
    RenderGraph, TextureDesc, TextureHandle,
};

use crate::display_transform_gpu::{DisplayTransformGpuError, OffscreenDisplayTransform};
use crate::draw::{canonical, DrawFunctions, RenderBindGroupBufferAccess};
use crate::lighting_gpu::{LightingGpuPass, OUTPUT_BYTES};
use crate::lighting_trace_gpu::TraceGraphRecorder;
use crate::particle::{EmberGpuAssetRegistry, ParticleGpuState};
use crate::queue::{PhaseItem, RenderQueues};
use crate::sets::RenderPhase;
use crate::target::RenderTarget;
use prism_camera::ClearMode;

mod surface_views;
pub(crate) use surface_views::{
    finalize_surface_view_graph, prepare_surface_view_graph, PreparedSurfaceViewGraph,
    SurfaceViewGraphError, SurfaceViewGraphInput, SurfaceViewGraphStages,
};

/// The host-visible stages represented by one frame graph.
///
/// This is deliberately a receipt, not a control flag. It records the
/// engine-neutral scene/visibility submission, the trace-result boundary, the
/// admitted direct/GI recipes, composite, and the texture presentation consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameGraphStages {
    pub scene_passes: u32,
    /// Actual trace compute passes recorded in this graph (not lighting recipes).
    pub trace_dispatches: u32,
    /// Candidate-build compute passes consuming the trace output.
    pub candidate_build_dispatches: u32,
    pub direct_recipes: u32,
    pub gi_recipes: u32,
    pub composite_passes: u32,
    pub present_passes: u32,
    /// Per-emitter raster passes admitted by the current physical Fabric frame.
    pub particle_passes: u32,
    pub ember_compute_passes: u32,
    pub lighting_dispatches: u32,
    /// Graph texture consumed by the final present pass (OCIO output for windows).
    pub present_source: prism_render_graph::TextureHandle,
}

impl FrameGraphStages {
    fn for_graph(
        scene_passes: u32,
        lighting: Option<LightingGpuPass>,
        trace: Option<TraceGraphRecorder>,
        particle: Option<&ParticleGpuState>,
        ember: Option<&EmberGpuAssetRegistry>,
        present_source: prism_render_graph::TextureHandle,
    ) -> Self {
        if let Some(pass) = lighting {
            let expects_trace = pass.run_gi && pass.candidate_source
                == prism_render_architecture::lighting::contract::GiCandidateSource::SoftwareBvhGpu;
            assert_eq!(
                trace.is_some(),
                expects_trace,
                "host trace must match the compiled candidate source"
            );
            assert_eq!(
                trace.is_some_and(|trace| trace.candidates.is_some()),
                expects_trace,
                "the planned GPU trace requires its candidate-build pass"
            );
        }
        let trace_dispatches = u32::from(trace.is_some());
        let candidate_build_dispatches = u32::from(
            trace.is_some_and(|trace| trace.candidates.is_some())
                && lighting.is_some_and(|pass| pass.run_gi),
        );
        let direct_recipes = lighting.is_some_and(|pass| pass.run_direct) as u32;
        let gi_recipes = lighting.is_some_and(|pass| pass.run_gi) as u32;
        // Direct lighting is evaluated per pixel by composite. Only GI has a
        // separate lighting compute pass in this graph.
        let lighting_dispatches = lighting.map_or(0, |pass| u32::from(pass.run_gi) + 1);
        Self {
            scene_passes,
            trace_dispatches,
            candidate_build_dispatches,
            direct_recipes,
            gi_recipes,
            composite_passes: u32::from(lighting.is_some()),
            present_passes: 0,
            // This receipt describes the headless append only. Native cameras
            // are admitted separately against their own color/depth targets.
            particle_passes: particle
                .filter(|particle| particle.fabric_gpu_frame().is_some())
                .map_or(0, |particle| {
                    u32::try_from(particle.fabric_draw_count())
                        .expect("admitted particle raster passes fit graph capacity")
                }),
            ember_compute_passes: ember.map_or(0, |registry| registry.compute_pass_count() as u32),
            lighting_dispatches,
            present_source,
        }
    }

    pub(crate) fn assert_contract(self) {
        assert!(
            self.scene_passes >= 1,
            "a frame must clear its scene target"
        );
        assert!(
            self.trace_dispatches <= 1,
            "a frame has at most one trace dispatch"
        );
        assert!(
            self.candidate_build_dispatches <= self.trace_dispatches,
            "candidate construction requires a trace dispatch"
        );
        assert!(
            self.candidate_build_dispatches <= self.gi_recipes,
            "candidate construction requires an admitted GI recipe"
        );
        assert!(
            self.composite_passes <= 1,
            "a frame has at most one lighting composite"
        );
        assert!(
            self.direct_recipes <= self.composite_passes,
            "direct lighting is evaluated by the lighting composite"
        );
        assert!(
            self.present_passes <= 1,
            "a frame has at most one presentation"
        );
        assert!(
            (self.particle_passes as usize)
                <= prism_render_graph::capacity::GraphLimits::default().passes,
            "a frame has too many particle raster stages"
        );
        assert!(
            (self.ember_compute_passes as usize)
                <= prism_render_graph::capacity::GraphLimits::default().passes,
            "a frame has too many Ember compute stages"
        );
        assert_eq!(
            self.lighting_dispatches,
            self.gi_recipes + self.composite_passes,
            "lighting dispatch accounting must match recorded passes"
        );
    }
}

/// The reserved native surface draw contract is installed only for the
/// SurfaceEncodeProvider by queue_stage. Neither lighting availability nor an
/// arbitrary group-0 binding implies this six-target raster contract. Custom
/// single-HDR providers use MESH_UNBOUND (or their own draw contract).
fn uses_surface_gbuffer(item: &PhaseItem, functions: &DrawFunctions) -> bool {
    item.dynamic_bind_group_contracts(functions)
        .iter()
        .any(|binding| {
            binding.identity == canonical::SURFACE_MATERIAL
                && binding.index == canonical::SURFACE_GROUP
        })
        || item.bind_groups(functions).iter().any(|binding| {
            binding.identity == canonical::SURFACE_MATERIAL
                && binding.index == canonical::SURFACE_GROUP
        })
}

/// Keep binding sidecars aligned when separating the native opaque producer
/// from single-HDR draws. Forward opaque draws run after composite so it cannot
/// overwrite them; both opaque paths write/test the same reversed-Z depth.
fn split_surface_opaque(
    queues: &RenderQueues,
    functions: &DrawFunctions,
) -> (RenderQueues, RenderQueues) {
    let mut surface = RenderQueues::new();
    let mut hdr = RenderQueues::new();
    for phase in RenderPhase::ORDER {
        for (index, item) in queues.phase(phase).iter().enumerate() {
            let destination =
                if phase == RenderPhase::Opaque && uses_surface_gbuffer(item, functions) {
                    &mut surface
                } else {
                    &mut hdr
                };
            queues.copy_item_to(phase, index, destination);
        }
    }
    (surface, hdr)
}

/// The texel format of the render sub-app's offscreen frame target.
///
/// Scene-linear HDR radiance must survive values greater than one until the
/// output transform applies exposure and display encoding. The display target
/// has its own format; it must not constrain this intermediate scene target.
pub const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The depth format of the render sub-app's frame.
///
/// `Depth32Float` is what reversed-Z wants: the convention is near 1 / far 0
/// tested with `GreaterEqual`, which needs the float buffer's precision near the
/// far plane (see [`DepthStencilState::reverse_z`]).
///
/// [`DepthStencilState::reverse_z`]: prism_render_driver::DepthStencilState::reverse_z
pub const RENDER_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// One owner-published texture that must be imported into the current frame
/// graph before view-only Ember bindings are resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameGraphTextureImport {
    /// Physical texture object owned by the publisher.
    pub texture: TextureId,
    /// The view id consumers use when referring to this import.
    pub view: TextureViewId,
    /// Symbolic description used by render-graph validation.
    pub desc: TextureDesc,
    /// Resource state before this frame begins.
    pub entry_state: TextureState,
    /// Resource state required after this frame finishes.
    pub exit_state: TextureState,
    /// Number of mip levels tracked by the graph.
    pub mip_levels: u32,
    /// Number of array layers tracked by the graph.
    pub array_layers: u32,
}

/// Render-world publication point for owner-to-frame-graph texture handles.
#[derive(Clone, Default)]
pub struct FrameGraphTextureImports {
    entries: BTreeMap<TextureViewId, FrameGraphTextureImport>,
    external_owners: Vec<Arc<dyn Any + Send + Sync>>,
}

impl Resource for FrameGraphTextureImports {}

impl FrameGraphTextureImports {
    /// Pins an immutable external resource generation to every graph built
    /// from this frame's imports. Submission ownership then extends its life
    /// through GPU completion.
    pub fn retain_owner(&mut self, owner: Arc<dyn Any + Send + Sync>) {
        self.external_owners.push(owner);
    }

    /// Publishes or replaces an owner texture import keyed by its view id.
    pub fn set(&mut self, import: FrameGraphTextureImport) {
        self.entries.insert(import.view, import);
    }

    /// Removes an owner texture import without destroying the physical object.
    pub fn remove(&mut self, view: TextureViewId) -> Option<FrameGraphTextureImport> {
        self.entries.remove(&view)
    }

    /// Returns deterministic owner imports in view-id order.
    pub fn iter(&self) -> impl Iterator<Item = &FrameGraphTextureImport> {
        self.entries.values()
    }

    /// Returns the number of published owner imports.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Extracts owner-published texture imports into the render world.
///
/// The physical texture and view remain owned by the publisher. Extraction
/// only copies the immutable per-frame import contract, so a device recovery
/// or owner replacement becomes visible atomically at the next render frame.
pub fn extract_frame_graph_texture_imports(main: &mut World, render: &mut World) {
    render.init_resource::<FrameGraphTextureImports>();
    let snapshot = main
        .get_resource::<FrameGraphTextureImports>()
        .cloned()
        .unwrap_or_default();
    *render.resource_mut::<FrameGraphTextureImports>() = snapshot;
}

/// Builds the frame graph: clear the target, then record the queued draws.
///
/// Native opaque surfaces write the six-target GBuffer independently of lighting
/// availability. Single-HDR providers write the frame target after composite
/// (when present), followed by the remaining phases in [`RenderPhase::ORDER`].
/// A buffer an item binds must be *declared* by
/// the pass that reads it — that is what orders the pass against writers and
/// lets validation see the access — so every bound buffer is imported into the
/// graph once and declared with its vertex, index, uniform, or storage read.
/// Render bind-group buffers come from the same binding contract that records
/// the opaque group handle, so the declaration and command agree by
/// construction.
///
/// The target is **imported** rather than allocated transiently so the same
/// texture backs every frame and its contents survive the graph (which is what
/// lets a test read the frame back), and every pass is flagged
/// [`PassFlags::SIDE_EFFECT`] so the compiler never culls a frame that draws into
/// an externally-owned target.
///
/// An item carrying GPU-written indirect arguments declares that read as well;
/// its [`IndirectArgs`](crate::IndirectArgs) handle must come from *this* graph,
/// which is why a bin stream is filled once the frame graph exists (see the port
/// notes on [`RenderQueues`]).
pub(crate) fn target_graph(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
) -> RenderGraph {
    target_graph_with_trace(target, queues, functions, lighting, None)
}

/// Builds a frame graph with an optional software-BVH trace recorder.
pub(crate) fn target_graph_with_trace(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
) -> RenderGraph {
    target_graph_output_with_trace(target, queues, functions, lighting, trace).0
}

/// The color handle lets the presentation pass consume this exact frame.
/// The trace-aware graph constructor used by the host composition path.
pub(crate) fn target_graph_output_with_trace(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
) -> (RenderGraph, prism_render_graph::TextureHandle) {
    target_graph_output_with_trace_and_particles(target, queues, functions, lighting, trace, None)
}

/// The host graph constructor with an optional GPU-authoritative particle pass.
pub(crate) fn target_graph_output_with_trace_and_particles(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleGpuState>,
) -> (RenderGraph, prism_render_graph::TextureHandle) {
    let (graph, color, _) = target_graph_output_with_trace_and_particles_and_external_views(
        target,
        queues,
        functions,
        lighting,
        trace,
        particle,
        &FrameGraphTextureImports::default(),
    );
    (graph, color)
}

/// Host graph constructor that also publishes the latest handles for imported
/// frame textures, allowing view-only Ember resources to alias those imports.
fn target_graph_output_with_trace_and_particles_and_external_views(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleGpuState>,
    external_views: &FrameGraphTextureImports,
) -> (
    RenderGraph,
    TextureHandle,
    BTreeMap<TextureViewId, TextureHandle>,
) {
    let mut graph = RenderGraph::new(target.viewport().extent());
    let mut buffers = FrameGraphBuffers::default();
    buffers.reserve_target(queues, functions, lighting, trace);
    let mut textures = BTreeMap::new();
    let appended = append_target_graph(
        &mut graph,
        &mut buffers,
        &mut textures,
        target,
        queues,
        functions,
        lighting,
        trace,
        particle,
        ClearMode::ColorAndDepth,
        external_views,
    );
    graph.blackboard_mut().set(appended.stages);
    (graph, appended.color, textures)
}

/// One physical buffer has one graph identity throughout all camera appends.
/// Writes publish their fresh SSA handle before later cameras collect reads.
#[derive(Default)]
struct FrameGraphBuffers {
    handles: BTreeMap<BufferId, BufferHandle>,
    requirements: BTreeMap<BufferId, BufferDesc>,
}

impl FrameGraphBuffers {
    fn require(&mut self, buffer: BufferId, desc: BufferDesc) {
        self.requirements
            .entry(buffer)
            .and_modify(|required| {
                required.size = required.size.max(desc.size);
                required.usage |= desc.usage;
            })
            .or_insert(desc);
    }

    /// Preflight all targets before the first import: even a later camera's
    /// vertex/uniform alias contributes to the physical buffer descriptor.
    fn reserve_target(
        &mut self,
        queues: &RenderQueues,
        functions: &DrawFunctions,
        lighting: Option<LightingGpuPass>,
        trace: Option<TraceGraphRecorder>,
    ) {
        for phase in RenderPhase::ORDER {
            for (index, item) in queues.phase(phase).iter().enumerate() {
                if let Some(external) = queues.external_indirect(phase, index) {
                    self.require(external.args, BufferDesc::new(16, BufferUsages::INDIRECT));
                    for &(buffer, bytes) in &external.vertices {
                        self.require(buffer, BufferDesc::new(bytes, BufferUsages::VERTEX));
                    }
                }
                for buffer in &item.vertices {
                    self.require(*buffer, BufferDesc::new(0, BufferUsages::VERTEX));
                }
                if let Some((buffer, _)) = item.indices {
                    self.require(buffer, BufferDesc::new(0, BufferUsages::INDEX));
                }
                for binding in item
                    .bind_groups(functions)
                    .iter()
                    .chain(queues.dynamic_bind_groups(phase, index).iter())
                {
                    for resource in &binding.buffers {
                        let usage = match resource.access {
                            RenderBindGroupBufferAccess::Uniform => BufferUsages::UNIFORM,
                            RenderBindGroupBufferAccess::Storage => BufferUsages::STORAGE,
                        };
                        self.require(resource.buffer, BufferDesc::new(0, usage));
                    }
                }
            }
        }
        if let Some(pass) = lighting {
            for (buffer, size, usage) in [
                (pass.gi_output, OUTPUT_BYTES, BufferUsages::STORAGE),
                (pass.params, 32, BufferUsages::UNIFORM),
                (pass.directionals, 0, BufferUsages::STORAGE),
                (pass.punctuals, 0, BufferUsages::STORAGE),
                (pass.areas, 0, BufferUsages::STORAGE),
                (
                    pass.candidates,
                    crate::lighting_gpu::GI_CANDIDATE_BUFFER_BYTES,
                    BufferUsages::STORAGE,
                ),
                (
                    pass.surfaces,
                    crate::lighting_gpu::LIGHTING_SURFACE_BUFFER_BYTES,
                    BufferUsages::STORAGE,
                ),
            ] {
                self.require(buffer, BufferDesc::new(size, usage));
            }
            let materials = pass
                .composite_material_buffers()
                .expect("lighting composite requires canonical material buffers");
            for (buffer, size) in [
                (
                    materials.headers.expect("validated material headers"),
                    materials.headers_size_bytes,
                ),
                (
                    materials
                        .parameters
                        .expect("validated material parameters"),
                    materials.parameters_size_bytes,
                ),
                (
                    materials
                        .execution_plans
                        .expect("validated material execution plans"),
                    materials.execution_plans_size_bytes,
                ),
                (
                    materials
                        .execution_programs
                        .expect("validated material execution programs"),
                    materials.execution_programs_size_bytes,
                ),
            ] {
                self.require(buffer, BufferDesc::new(size, BufferUsages::STORAGE));
            }
        }
        if let Some(trace) = trace {
            for buffer in [
                Some(trace.nodes),
                Some(trace.triangles),
                Some(trace.attributes),
                Some(trace.rays),
                Some(trace.hits),
                trace.candidates,
                trace.radiance,
                trace.surfaces,
            ]
            .into_iter()
            .flatten()
            {
                self.require(
                    buffer.id,
                    BufferDesc::new(buffer.size_bytes, BufferUsages::STORAGE),
                );
            }
            if let Some(params) = trace.params {
                self.require(
                    params.id,
                    BufferDesc::new(params.size_bytes, BufferUsages::UNIFORM),
                );
            }
        }
    }

    fn import(
        &mut self,
        graph: &mut RenderGraph,
        name: &str,
        desc: BufferDesc,
        buffer: BufferId,
    ) -> BufferHandle {
        let desc = self.requirements.get(&buffer).cloned().unwrap_or(desc);
        *self.handles.entry(buffer).or_insert_with(|| {
            graph.import_buffer(
                name,
                desc,
                ImportedBuffer {
                    buffer,
                    entry_state: BufferState::initial(),
                },
            )
        })
    }

    fn publish(&mut self, buffer: BufferId, handle: BufferHandle) {
        let previous = self
            .handles
            .get_mut(&buffer)
            .expect("published buffer was imported");
        assert_eq!(
            previous.resource(),
            handle.resource(),
            "physical buffer identity changed"
        );
        assert!(
            previous.version() <= handle.version(),
            "buffer SSA version regressed"
        );
        *previous = handle;
    }
}

struct TargetGraphAppend {
    color: TextureHandle,
    depth: TextureHandle,
    taa_motion: TextureHandle,
    taa_reactive: TextureHandle,
    stages: FrameGraphStages,
}

fn sampled_texture_handles(
    item: &PhaseItem,
    phase: RenderPhase,
    item_index: usize,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    imports: &BTreeMap<TextureViewId, TextureHandle>,
) -> Vec<TextureHandle> {
    let mut handles = Vec::new();
    for binding in item
        .bind_groups(functions)
        .iter()
        .chain(queues.dynamic_bind_groups(phase, item_index).iter())
    {
        for view in &binding.sampled_textures {
            let handle = *imports.get(view).unwrap_or_else(|| {
                panic!(
                    "render bind group {:?} samples TextureViewId {:?} without a FrameGraphTextureImports entry",
                    binding.bind_group,
                    view
                )
            });
            if !handles.contains(&handle) {
                handles.push(handle);
            }
        }
    }
    handles
}

fn retain_queue_binding_owners(
    graph: &mut RenderGraph,
    queues: &RenderQueues,
    functions: &DrawFunctions,
) {
    for phase in RenderPhase::ORDER {
        for (item_index, item) in queues.phase(phase).iter().enumerate() {
            for binding in item
                .bind_groups(functions)
                .iter()
                .chain(queues.dynamic_bind_groups(phase, item_index).iter())
            {
                for owner in &binding.external_owners {
                    graph.retain_external_owner(owner.clone());
                }
            }
        }
    }
}

fn clear_color_or_write(
    builder: &mut prism_render_graph::PassBuilder<'_>,
    handle: TextureHandle,
    clear: ClearMode,
    value: Color,
) -> TextureHandle {
    match clear {
        ClearMode::ColorAndDepth => builder.clear_color_attachment(handle, value),
        ClearMode::DepthOnly | ClearMode::None => {
            builder.color_attachment_load(handle, prism_render_driver::LoadOp::Load)
        }
        ClearMode::Inherit => unreachable!("clear mode is resolved before graph building"),
    }
}

/// Appends one target without creating a second graph or reimporting shared buffers.
fn append_target_graph(
    graph: &mut RenderGraph,
    buffers: &mut FrameGraphBuffers,
    external_texture_handles: &mut BTreeMap<TextureViewId, TextureHandle>,
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleGpuState>,
    clear: ClearMode,
    external_views: &FrameGraphTextureImports,
) -> TargetGraphAppend {
    for owner in &external_views.external_owners {
        graph.retain_external_owner(owner.clone());
    }
    // Persistent camera queues carry physical IDs and owners, NEVER handles
    // from a previous output graph. Resolve on a local copy before splitting.
    let mut local_queues = queues.clone();
    retain_queue_binding_owners(graph, &local_queues, functions);
    local_queues.resolve_external_indirect(|external| {
        graph.retain_external_owner(external.owner.clone());
        crate::queue::IndirectArgs {
            args: buffers.import(
                graph,
                "prism-external-draw-indirect",
                BufferDesc::new(16, BufferUsages::INDIRECT),
                external.args,
            ),
            offset: 0,
            count: 1,
            stride: 16,
        }
    });
    let queues = &local_queues;
    let (surface_queues, hdr_queues) = split_surface_opaque(queues, functions);
    // GBuffer is cleared every frame, even when the last native draw disappears.
    // Without a composite, HDR needs its own initializer as well.
    let scene_passes = 1
        + u32::from(lighting.is_none())
        + u32::from(!surface_queues.phase(RenderPhase::Opaque).is_empty())
        + RenderPhase::ORDER
            .into_iter()
            .filter(|phase| !hdr_queues.phase(*phase).is_empty())
            .count() as u32;
    let mut color = graph.import_texture(
        "prism-render-app-target",
        TextureDesc::color(target.format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::RENDER_ATTACHMENT
                    .union(TextureUsages::COPY_SRC)
                    .union(TextureUsages::TEXTURE_BINDING)
                    .union(TextureUsages::STORAGE_BINDING),
            ),
        ImportedTexture {
            texture: target.texture(),
            default_view: target.view(),
            entry_state: target.color_entry_state(),
            exit_state: target.color_exit_state(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut position_x = graph.import_texture(
        "prism-render-app-gbuffer-position-x",
        TextureDesc::color(target.position_x_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::COPY_SRC
                    .union(TextureUsages::COPY_DST)
                    .union(TextureUsages::TEXTURE_BINDING),
            ),
        ImportedTexture {
            texture: target.position_x_texture(),
            default_view: target.position_x_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut position_y = graph.import_texture(
        "prism-render-app-gbuffer-position-y",
        TextureDesc::color(target.position_y_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::COPY_SRC
                    .union(TextureUsages::COPY_DST)
                    .union(TextureUsages::TEXTURE_BINDING),
            ),
        ImportedTexture {
            texture: target.position_y_texture(),
            default_view: target.position_y_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut position_z = graph.import_texture(
        "prism-render-app-gbuffer-position-z",
        TextureDesc::color(target.position_z_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::COPY_SRC
                    .union(TextureUsages::COPY_DST)
                    .union(TextureUsages::TEXTURE_BINDING),
            ),
        ImportedTexture {
            texture: target.position_z_texture(),
            default_view: target.position_z_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut normal = graph.import_texture(
        "prism-render-app-gbuffer-normal",
        TextureDesc::color(target.normal_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::COPY_SRC
                    .union(TextureUsages::COPY_DST)
                    .union(TextureUsages::TEXTURE_BINDING),
            ),
        ImportedTexture {
            texture: target.normal_texture(),
            default_view: target.normal_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut albedo = graph.import_texture(
        "prism-render-app-gbuffer-albedo",
        TextureDesc::color(target.albedo_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::COPY_SRC
                    .union(TextureUsages::COPY_DST)
                    .union(TextureUsages::TEXTURE_BINDING),
            ),
        ImportedTexture {
            texture: target.albedo_texture(),
            default_view: target.albedo_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut surface_id = graph.import_texture(
        "prism-render-app-gbuffer-surface-id",
        TextureDesc::color(target.surface_id_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(
                TextureUsages::COPY_SRC
                    .union(TextureUsages::COPY_DST)
                    .union(TextureUsages::TEXTURE_BINDING),
            ),
        ImportedTexture {
            texture: target.surface_id_texture(),
            default_view: target.surface_id_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut motion = graph.import_texture(
        "prism-render-app-taa-motion",
        TextureDesc::color(target.taa_motion_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(crate::target::TAA_TEMPORAL_INPUT_USAGES),
        ImportedTexture {
            texture: target.taa_motion_texture(),
            default_view: target.taa_motion_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let mut reactive = graph.import_texture(
        "prism-render-app-taa-reactive",
        TextureDesc::color(target.taa_reactive_format())
            .with_size(prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ))
            .with_usage(crate::target::TAA_TEMPORAL_INPUT_USAGES),
        ImportedTexture {
            texture: target.taa_reactive_texture(),
            default_view: target.taa_reactive_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    // Collect every physical buffer before importing it. An imported buffer's
    // descriptor is the union of the usages this frame declares — the graph
    // neither owns nor sizes it, and the size is the owner's business (a mesh
    // buffer, a scene/material buffer, a bin stream).
    let mut required_usages: BTreeMap<BufferId, BufferUsages> = BTreeMap::new();
    for phase in RenderPhase::ORDER {
        for (item_index, item) in queues.phase(phase).iter().enumerate() {
            for buffer in &item.vertices {
                required_usages
                    .entry(*buffer)
                    .and_modify(|usage| *usage |= BufferUsages::VERTEX)
                    .or_insert(BufferUsages::VERTEX);
            }
            if let Some((indices, _)) = item.indices {
                required_usages
                    .entry(indices)
                    .and_modify(|usage| *usage |= BufferUsages::INDEX)
                    .or_insert(BufferUsages::INDEX);
            }
            for binding in item.bind_groups(functions) {
                for resource in &binding.buffers {
                    let usage = match resource.access {
                        RenderBindGroupBufferAccess::Uniform => BufferUsages::UNIFORM,
                        RenderBindGroupBufferAccess::Storage => BufferUsages::STORAGE,
                    };
                    required_usages
                        .entry(resource.buffer)
                        .and_modify(|declared| *declared |= usage)
                        .or_insert(usage);
                }
            }
            for binding in queues.dynamic_bind_groups(phase, item_index) {
                for resource in &binding.buffers {
                    let usage = match resource.access {
                        RenderBindGroupBufferAccess::Uniform => BufferUsages::UNIFORM,
                        RenderBindGroupBufferAccess::Storage => BufferUsages::STORAGE,
                    };
                    required_usages
                        .entry(resource.buffer)
                        .and_modify(|declared| *declared |= usage)
                        .or_insert(usage);
                }
            }
        }
    }
    for (id, usage) in required_usages {
        buffers.import(graph, "frame-bound-buffer", BufferDesc::new(0, usage), id);
    }

    // A frame that rasterizes geometry needs depth, and the pipelines it
    // specializes must agree with the attachment: the queue stage passes
    // `RENDER_DEPTH_FORMAT` into the variant for exactly this reason.
    let mut depth = graph.import_texture(
        "prism-render-app-depth",
        TextureDesc::depth(target.depth_format()).with_size(
            prism_render_graph::SizeClass::absolute_2d(
                target.viewport().width,
                target.viewport().height,
            ),
        ),
        ImportedTexture {
            texture: target.depth_texture(),
            default_view: target.depth_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::depth_write(),
            mip_levels: 1,
            array_layers: 1,
        },
    );

    // Publish every host-owned view under the graph-local handle created for
    // this frame. View-only Ember bindings can now alias color, G-buffer,
    // depth, or arbitrary owner textures without guessing their descriptors.
    external_texture_handles.insert(target.view(), color);
    external_texture_handles.insert(target.position_x_view(), position_x);
    external_texture_handles.insert(target.position_y_view(), position_y);
    external_texture_handles.insert(target.position_z_view(), position_z);
    external_texture_handles.insert(target.normal_view(), normal);
    external_texture_handles.insert(target.albedo_view(), albedo);
    external_texture_handles.insert(target.surface_id_view(), surface_id);
    external_texture_handles.insert(target.taa_motion_view(), motion);
    external_texture_handles.insert(target.taa_reactive_view(), reactive);
    external_texture_handles.insert(target.depth_view(), depth);
    for (index, import) in external_views.iter().enumerate() {
        if external_texture_handles.contains_key(&import.view) {
            continue;
        }
        let handle = graph.import_texture(
            format!("prism-render-app-external-view-{index}"),
            import.desc.clone(),
            ImportedTexture {
                texture: import.texture,
                default_view: import.view,
                entry_state: import.entry_state,
                exit_state: import.exit_state,
                mip_levels: import.mip_levels.max(1),
                array_layers: import.array_layers.max(1),
            },
        );
        external_texture_handles.insert(import.view, handle);
    }

    // Clear outputs in separate passes: the scene-linear HDR target is 8 B per
    // sample while the compact GBuffer is exactly 32 B per sample, so they may
    // not share one color-attachment set on devices with the WebGPU minimum
    // `maxColorAttachmentBytesPerSample` limit.
    {
        graph.add_raster_pass(
            "prism-render-app-clear-gbuffer",
            PassFlags::SIDE_EFFECT,
            |builder| {
                position_x = clear_color_or_write(builder, position_x, clear, Color::TRANSPARENT);
                position_y = clear_color_or_write(builder, position_y, clear, Color::TRANSPARENT);
                position_z = clear_color_or_write(builder, position_z, clear, Color::TRANSPARENT);
                // Alpha is surface coverage, not an opaque background. Clear
                // explicitly rather than relying on the generic color policy.
                normal = clear_color_or_write(builder, normal, clear, Color::TRANSPARENT);
                albedo = clear_color_or_write(builder, albedo, clear, Color::TRANSPARENT);
                surface_id = clear_color_or_write(builder, surface_id, clear, Color::TRANSPARENT);
                depth = match clear {
                    ClearMode::ColorAndDepth | ClearMode::DepthOnly => {
                        builder.depth_attachment(depth)
                    }
                    ClearMode::None => builder.depth_attachment_load(depth, DepthLoadOp::Load),
                    ClearMode::Inherit => {
                        unreachable!("clear mode is resolved before graph building")
                    }
                };
                |_ctx| {}
            },
        );
    }
    if lighting.is_none() {
        graph.add_raster_pass(
            "prism-render-app-clear-color",
            PassFlags::SIDE_EFFECT,
            |builder| {
                color = clear_color_or_write(builder, color, clear, Color::TRANSPARENT);
                |_ctx| {}
            },
        );
    }

    for phase in [RenderPhase::Opaque] {
        let queues = &surface_queues;
        if queues.phase(phase).is_empty() {
            continue;
        }
        let mut vertex_reads = Vec::new();
        let mut index_reads = Vec::new();
        let mut indirect_reads = Vec::new();
        let mut uniform_reads = Vec::new();
        let mut storage_reads = Vec::new();
        let mut sampled_texture_reads = Vec::new();
        for (item_index, item) in queues.phase(phase).iter().enumerate() {
            sampled_texture_reads.extend(sampled_texture_handles(
                item,
                phase,
                item_index,
                queues,
                functions,
                external_texture_handles,
            ));
            for buffer in &item.vertices {
                if let Some(handle) = buffers.handles.get(buffer) {
                    vertex_reads.push(*handle);
                }
            }
            if let Some((indices, _)) = item.indices
                && let Some(handle) = buffers.handles.get(&indices)
            {
                index_reads.push(*handle);
            }
            if let Some(args) = item.indirect {
                indirect_reads.push(args.args);
            }
            for binding in item.bind_groups(functions) {
                for resource in &binding.buffers {
                    if let Some(handle) = buffers.handles.get(&resource.buffer) {
                        match resource.access {
                            RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                            RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                        }
                    }
                }
            }
            for binding in queues.dynamic_bind_groups(phase, item_index) {
                for resource in &binding.buffers {
                    if let Some(handle) = buffers.handles.get(&resource.buffer) {
                        match resource.access {
                            RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                            RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                        }
                    }
                }
            }
        }
        graph.add_raster_pass(
            format!("prism-render-app-{phase:?}"),
            PassFlags::SIDE_EFFECT,
            |builder| {
                position_x = builder.color_attachment(position_x);
                position_y = builder.color_attachment(position_y);
                position_z = builder.color_attachment(position_z);
                normal = builder.color_attachment(normal);
                albedo = builder.color_attachment(albedo);
                surface_id = builder.color_attachment(surface_id);
                depth = builder.depth_attachment(depth);
                for handle in &vertex_reads {
                    let _ = builder.read_vertex(*handle);
                }
                for handle in &index_reads {
                    let _ = builder.read_index(*handle);
                }
                for handle in &indirect_reads {
                    let _ = builder.read_indirect(*handle);
                }
                for handle in &uniform_reads {
                    let _ = builder.read_uniform(*handle);
                }
                for handle in &storage_reads {
                    let _ = builder.read_storage_buffer(*handle);
                }
                for handle in &sampled_texture_reads {
                    let _ = builder.read_texture(*handle);
                }
                let queues = queues.clone();
                let functions = functions.clone();
                move |ctx| queues.record(phase, ctx, &functions)
            },
        );
    }

    let mut trace_hits = None;
    let mut trace_inputs = None;
    let mut trace_pass_dependency = None;
    if let Some(trace) = trace {
        let nodes = buffers.import(
            graph,
            "prism-lighting-trace-nodes",
            BufferDesc::new(trace.nodes.size_bytes, BufferUsages::STORAGE),
            trace.nodes.id,
        );
        let triangles = buffers.import(
            graph,
            "prism-lighting-trace-triangles",
            BufferDesc::new(trace.triangles.size_bytes, BufferUsages::STORAGE),
            trace.triangles.id,
        );
        let attributes = buffers.import(
            graph,
            "prism-lighting-trace-attributes",
            BufferDesc::new(trace.attributes.size_bytes, BufferUsages::STORAGE),
            trace.attributes.id,
        );
        let rays = buffers.import(
            graph,
            "prism-lighting-trace-rays",
            BufferDesc::new(trace.rays.size_bytes, BufferUsages::STORAGE),
            trace.rays.id,
        );
        let mut hits = buffers.import(
            graph,
            "prism-lighting-trace-hits",
            BufferDesc::new(trace.hits.size_bytes, BufferUsages::STORAGE),
            trace.hits.id,
        );
        let trace_pass = graph
            .try_add_pass_with_dependencies(
                "prism-lighting-trace-dispatch",
                prism_render_graph::pass::PassKind::Compute,
                PassFlags::SIDE_EFFECT,
                Vec::new(),
                |builder| {
                    // The read creates an explicit scene-raster -> trace dependency;
                    // the current BVH kernel does not sample the raster color itself.
                    let _scene_raster_boundary = builder.read_depth_attachment(depth);
                    let _nodes = builder.read_storage_buffer(nodes);
                    let _triangles = builder.read_storage_buffer(triangles);
                    let _attributes = builder.read_storage_buffer(attributes);
                    let _rays = builder.read_storage_buffer(rays);
                    hits = builder.write_storage_buffer(hits);
                    move |context| trace.record_graph(context)
                },
            )
            .expect("trace pass fits the host frame graph");
        buffers.publish(trace.hits.id, hits);
        trace_pass_dependency = Some(trace_pass);
        trace_hits = Some(hits);
        trace_inputs = Some((triangles, attributes, rays));
    }

    let Some(lighting) = lighting else {
        for phase in [
            RenderPhase::Opaque,
            RenderPhase::Transparent,
            RenderPhase::Stylized,
            RenderPhase::Custom,
        ] {
            let queues = &hdr_queues;
            if queues.phase(phase).is_empty() {
                continue;
            }
            let mut vertex_reads = Vec::new();
            let mut index_reads = Vec::new();
            let mut indirect_reads = Vec::new();
            let mut uniform_reads = Vec::new();
            let mut storage_reads = Vec::new();
            let mut sampled_texture_reads = Vec::new();
            for (item_index, item) in queues.phase(phase).iter().enumerate() {
                sampled_texture_reads.extend(sampled_texture_handles(
                    item,
                    phase,
                    item_index,
                    queues,
                    functions,
                    external_texture_handles,
                ));
                for buffer in &item.vertices {
                    if let Some(handle) = buffers.handles.get(buffer) {
                        vertex_reads.push(*handle);
                    }
                }
                if let Some((indices, _)) = item.indices
                    && let Some(handle) = buffers.handles.get(&indices)
                {
                    index_reads.push(*handle);
                }
                if let Some(args) = item.indirect {
                    indirect_reads.push(args.args);
                }
                for binding in item.bind_groups(functions) {
                    for resource in &binding.buffers {
                        if let Some(handle) = buffers.handles.get(&resource.buffer) {
                            match resource.access {
                                RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                                RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                            }
                        }
                    }
                }
                for binding in queues.dynamic_bind_groups(phase, item_index) {
                    for resource in &binding.buffers {
                        if let Some(handle) = buffers.handles.get(&resource.buffer) {
                            match resource.access {
                                RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                                RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                            }
                        }
                    }
                }
            }
            graph.add_raster_pass(
                format!("prism-render-app-{phase:?}"),
                PassFlags::SIDE_EFFECT,
                |builder| {
                    color = builder.color_attachment(color);
                    depth = if phase == RenderPhase::Opaque {
                        builder.depth_attachment(depth)
                    } else {
                        builder.read_depth_attachment(depth)
                    };
                    for handle in &vertex_reads {
                        let _ = builder.read_vertex(*handle);
                    }
                    for handle in &index_reads {
                        let _ = builder.read_index(*handle);
                    }
                    for handle in &indirect_reads {
                        let _ = builder.read_indirect(*handle);
                    }
                    for handle in &uniform_reads {
                        let _ = builder.read_uniform(*handle);
                    }
                    for handle in &storage_reads {
                        let _ = builder.read_storage_buffer(*handle);
                    }
                    for handle in &sampled_texture_reads {
                        let _ = builder.read_texture(*handle);
                    }
                    let queues = queues.clone();
                    let functions = functions.clone();
                    move |ctx| queues.record(phase, ctx, &functions)
                },
            );
        }
        if let Some(particle) = particle {
            (color, depth) = particle
                .append_gpu_to_graph(graph, color, depth)
                .expect("particle GPU graph admission");
        }
        let stages = FrameGraphStages::for_graph(scene_passes, None, trace, particle, None, color);
        stages.assert_contract();
        external_texture_handles.insert(target.view(), color);
        external_texture_handles.insert(target.position_x_view(), position_x);
        external_texture_handles.insert(target.position_y_view(), position_y);
        external_texture_handles.insert(target.position_z_view(), position_z);
        external_texture_handles.insert(target.normal_view(), normal);
        external_texture_handles.insert(target.albedo_view(), albedo);
        external_texture_handles.insert(target.surface_id_view(), surface_id);
        external_texture_handles.insert(target.taa_motion_view(), motion);
        external_texture_handles.insert(target.taa_reactive_view(), reactive);
        external_texture_handles.insert(target.depth_view(), depth);
        return TargetGraphAppend {
            color,
            depth,
            taa_motion: motion,
            taa_reactive: reactive,
            stages,
        };
    };

    let mut gi_output = buffers.import(
        graph,
        "prism-lighting-gi-output",
        BufferDesc::new(OUTPUT_BYTES, BufferUsages::STORAGE),
        lighting.gi_output,
    );
    let params = buffers.import(
        graph,
        "prism-lighting-plan-params",
        BufferDesc::new(32, BufferUsages::UNIFORM),
        lighting.params,
    );
    let directionals = buffers.import(
        graph,
        "prism-lighting-directionals",
        BufferDesc::new(0, BufferUsages::STORAGE),
        lighting.directionals,
    );
    let punctuals = buffers.import(
        graph,
        "prism-lighting-punctuals",
        BufferDesc::new(0, BufferUsages::STORAGE),
        lighting.punctuals,
    );
    let areas = buffers.import(
        graph,
        "prism-lighting-areas",
        BufferDesc::new(0, BufferUsages::STORAGE),
        lighting.areas,
    );
    let mut candidates = buffers.import(
        graph,
        "prism-lighting-gi-candidates",
        BufferDesc::new(
            crate::lighting_gpu::GI_CANDIDATE_BUFFER_BYTES,
            BufferUsages::STORAGE,
        ),
        lighting.candidates,
    );
    let surfaces = buffers.import(
        graph,
        "prism-lighting-surfaces",
        BufferDesc::new(
            crate::lighting_gpu::LIGHTING_SURFACE_BUFFER_BYTES,
            BufferUsages::STORAGE,
        ),
        lighting.surfaces,
    );
    let materials = lighting
        .composite_material_buffers()
        .expect("lighting composite requires canonical material buffers");
    let material_bindings = [
        (
            "prism-material-headers",
            materials.headers.expect("validated material headers"),
            materials.headers_size_bytes,
        ),
        (
            "prism-material-parameters",
            materials
                .parameters
                .expect("validated material parameters"),
            materials.parameters_size_bytes,
        ),
        (
            "prism-material-execution-plans",
            materials
                .execution_plans
                .expect("validated material execution plans"),
            materials.execution_plans_size_bytes,
        ),
        (
            "prism-material-execution-programs",
            materials
                .execution_programs
                .expect("validated material execution programs"),
            materials.execution_programs_size_bytes,
        ),
    ];
    let material_reads = material_bindings.map(|(name, buffer, size)| {
        buffers.import(
            graph,
            name,
            BufferDesc::new(size, BufferUsages::STORAGE),
            buffer,
        )
    });
    if let (Some(trace), Some(hits), Some((trace_triangles, trace_attributes, trace_rays)), true) =
        (trace, trace_hits, trace_inputs, lighting.run_gi)
        && let (
            Some(candidate_output),
            Some(radiance),
            Some(candidate_surfaces),
            Some(candidate_params),
        ) = (
            trace.candidates,
            trace.radiance,
            trace.surfaces,
            trace.params,
        )
    {
        let radiance = buffers.import(
            graph,
            "prism-lighting-trace-radiance",
            BufferDesc::new(radiance.size_bytes, BufferUsages::STORAGE),
            radiance.id,
        );
        assert_eq!(candidate_output.id, lighting.candidates);
        assert_eq!(candidate_surfaces.id, lighting.surfaces);
        assert_eq!(candidate_params.id, lighting.params);
        graph
            .try_add_pass_with_dependencies(
                "prism-lighting-gi-candidate-build",
                prism_render_graph::pass::PassKind::Compute,
                PassFlags::SIDE_EFFECT,
                trace_pass_dependency.into_iter().collect(),
                |builder| {
                    let _rays = builder.read_storage_buffer(trace_rays);
                    let _hits = builder.read_storage_buffer(hits);
                    let _triangles = builder.read_storage_buffer(trace_triangles);
                    let _attributes = builder.read_storage_buffer(trace_attributes);
                    let _radiance = builder.read_storage_buffer(radiance);
                    let _surfaces = builder.read_storage_buffer(surfaces);
                    let _params = builder.read_uniform(params);
                    candidates = builder.write_storage_buffer(candidates);
                    move |context| trace.record_candidate_graph(context)
                },
            )
            .expect("GI candidate pass fits the host frame graph");
    }
    buffers.publish(lighting.candidates, candidates);
    if lighting.run_gi {
        graph.add_compute_pass(
            "prism-lighting-gi-dispatch",
            PassFlags::SIDE_EFFECT,
            |builder| {
                gi_output = builder.write_storage_buffer(gi_output);
                let _params = builder.read_uniform(params);
                let _candidates = builder.read_storage_buffer(candidates);
                let _surfaces = builder.read_storage_buffer(surfaces);
                move |context| {
                    context
                        .set_compute_pipeline(lighting.gi_pipeline)
                        .set_compute_bind_group(0, lighting.gi_bind_group, Vec::new())
                        .dispatch(lighting.workgroups, 1, 1);
                }
            },
        );
    }

    buffers.publish(lighting.gi_output, gi_output);

    let target_bind_group = lighting
        .composite_target_bind_group
        .expect("lighting graph requires a target-aware composite bind group");
    let material_bind_group = lighting
        .composite_material_bind_group
        .expect("lighting composite requires canonical material bind group");
    assert_eq!(
        crate::lighting_gpu::LightingGpuPass::composite_material_group(),
        2,
        "composite material bind group must match the shader's group 2"
    );
    let extent = target.viewport().extent();
    graph.add_compute_pass(
        "prism-lighting-composite-dispatch",
        PassFlags::SIDE_EFFECT,
        |builder| {
            position_x = builder.sample(position_x);
            position_y = builder.sample(position_y);
            position_z = builder.sample(position_z);
            normal = builder.sample(normal);
            albedo = builder.sample(albedo);
            surface_id = builder.sample(surface_id);
            let _gi = builder.read_storage_buffer(gi_output);
            let _params = builder.read_uniform(params);
            let _directionals = builder.read_storage_buffer(directionals);
            let _punctuals = builder.read_storage_buffer(punctuals);
            let _areas = builder.read_storage_buffer(areas);
            assert_eq!(material_reads.len(), 4);
            for handle in material_reads {
                let _ = builder.read_storage_buffer(handle);
            }
            color = builder.write_storage_texture(color);
            move |context| {
                context
                    .set_compute_pipeline(lighting.composite_pipeline)
                    .set_compute_bind_group(0, lighting.composite_bind_group, Vec::new())
                    .set_compute_bind_group(1, target_bind_group, Vec::new());
                context.set_compute_bind_group(
                    crate::lighting_gpu::LightingGpuPass::composite_material_group(),
                    material_bind_group,
                    Vec::new(),
                );
                context.dispatch(
                    extent
                        .width
                        .div_ceil(crate::lighting_gpu::COMPOSITE_WORKGROUP_X),
                    extent
                        .height
                        .div_ceil(crate::lighting_gpu::COMPOSITE_WORKGROUP_Y),
                    1,
                );
            }
        },
    );

    for phase in [
        RenderPhase::Opaque,
        RenderPhase::Transparent,
        RenderPhase::Stylized,
        RenderPhase::Custom,
    ] {
        let queues = &hdr_queues;
        if queues.phase(phase).is_empty() {
            continue;
        }
        let mut vertex_reads = Vec::new();
        let mut index_reads = Vec::new();
        let mut indirect_reads = Vec::new();
        let mut uniform_reads = Vec::new();
        let mut storage_reads = Vec::new();
        let mut sampled_texture_reads = Vec::new();
        for (item_index, item) in queues.phase(phase).iter().enumerate() {
            sampled_texture_reads.extend(sampled_texture_handles(
                item,
                phase,
                item_index,
                queues,
                functions,
                external_texture_handles,
            ));
            for buffer in &item.vertices {
                if let Some(handle) = buffers.handles.get(buffer) {
                    vertex_reads.push(*handle);
                }
            }
            if let Some((indices, _)) = item.indices
                && let Some(handle) = buffers.handles.get(&indices)
            {
                index_reads.push(*handle);
            }
            if let Some(args) = item.indirect {
                indirect_reads.push(args.args);
            }
            for binding in item.bind_groups(functions) {
                for resource in &binding.buffers {
                    if let Some(handle) = buffers.handles.get(&resource.buffer) {
                        match resource.access {
                            RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                            RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                        }
                    }
                }
            }
            for binding in queues.dynamic_bind_groups(phase, item_index) {
                for resource in &binding.buffers {
                    if let Some(handle) = buffers.handles.get(&resource.buffer) {
                        match resource.access {
                            RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                            RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                        }
                    }
                }
            }
        }
        graph.add_raster_pass(
            format!("prism-render-app-{phase:?}"),
            PassFlags::SIDE_EFFECT,
            |builder| {
                color = builder.color_attachment(color);
                depth = if phase == RenderPhase::Opaque {
                    builder.depth_attachment(depth)
                } else {
                    builder.read_depth_attachment(depth)
                };
                for handle in &vertex_reads {
                    let _ = builder.read_vertex(*handle);
                }
                for handle in &index_reads {
                    let _ = builder.read_index(*handle);
                }
                for handle in &indirect_reads {
                    let _ = builder.read_indirect(*handle);
                }
                for handle in &uniform_reads {
                    let _ = builder.read_uniform(*handle);
                }
                for handle in &storage_reads {
                    let _ = builder.read_storage_buffer(*handle);
                }
                for handle in &sampled_texture_reads {
                    let _ = builder.read_texture(*handle);
                }
                let queues = queues.clone();
                let functions = functions.clone();
                move |ctx| queues.record(phase, ctx, &functions)
            },
        );
    }
    if let Some(particle) = particle {
        (color, depth) = particle
            .append_gpu_to_graph(graph, color, depth)
            .expect("particle GPU graph admission");
    }

    let stages =
        FrameGraphStages::for_graph(scene_passes, Some(lighting), trace, particle, None, color);
    stages.assert_contract();
    external_texture_handles.insert(target.view(), color);
    external_texture_handles.insert(target.position_x_view(), position_x);
    external_texture_handles.insert(target.position_y_view(), position_y);
    external_texture_handles.insert(target.position_z_view(), position_z);
    external_texture_handles.insert(target.normal_view(), normal);
    external_texture_handles.insert(target.albedo_view(), albedo);
    external_texture_handles.insert(target.surface_id_view(), surface_id);
    external_texture_handles.insert(target.taa_motion_view(), motion);
    external_texture_handles.insert(target.taa_reactive_view(), reactive);
    external_texture_handles.insert(target.depth_view(), depth);
    TargetGraphAppend {
        color,
        depth,
        taa_motion: motion,
        taa_reactive: reactive,
        stages,
    }
}

/// Builds the headless host frame and appends its display transform to the same
/// graph. The source remains the persistent linear frame target; the returned
/// display texture is the graph's terminal output for readback/offline hosts.
pub(crate) fn target_graph_with_display_transform(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    transform: &OffscreenDisplayTransform,
    queue: &dyn prism_render_driver::RenderQueue,
) -> Result<RenderGraph, DisplayTransformGpuError> {
    target_graph_with_display_transform_and_trace(
        target, queues, functions, lighting, None, transform, queue,
    )
}

/// Trace-aware display-transform graph constructor for render-host composition.
pub(crate) fn target_graph_with_display_transform_and_trace(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    transform: &OffscreenDisplayTransform,
    queue: &dyn prism_render_driver::RenderQueue,
) -> Result<RenderGraph, DisplayTransformGpuError> {
    target_graph_with_display_transform_and_trace_and_particles(
        target, queues, functions, lighting, trace, None, transform, queue,
    )
}

/// Trace-aware display-transform graph constructor with an optional GPU particle pass.
pub(crate) fn target_graph_with_display_transform_and_trace_and_particles(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleGpuState>,
    transform: &OffscreenDisplayTransform,
    queue: &dyn prism_render_driver::RenderQueue,
) -> Result<RenderGraph, DisplayTransformGpuError> {
    target_graph_with_display_transform_and_trace_particles_and_ember(
        target,
        queues,
        functions,
        lighting,
        trace,
        particle,
        None,
        &FrameGraphTextureImports::default(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        0,
        transform,
        queue,
    )
}

/// Trace-aware display-transform graph constructor with optional Ember compute
/// graphs in addition to the GPU-authoritative particle bridge.
pub(crate) fn target_graph_with_display_transform_and_trace_particles_and_ember(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleGpuState>,
    ember: Option<&EmberGpuAssetRegistry>,
    external_views: &FrameGraphTextureImports,
    external_imports: &BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport>,
    // Handles for caller-owned views already imported into this frame graph.
    external_texture_handles: &BTreeMap<TextureViewId, TextureHandle>,
    frame_index: u64,
    transform: &OffscreenDisplayTransform,
    queue: &dyn prism_render_driver::RenderQueue,
) -> Result<RenderGraph, DisplayTransformGpuError> {
    let (mut graph, source, mut host_texture_handles) =
        target_graph_output_with_trace_and_particles_and_external_views(
            target,
            queues,
            functions,
            lighting,
            trace,
            particle,
            external_views,
        );
    for (view, handle) in external_texture_handles {
        host_texture_handles.entry(*view).or_insert(*handle);
    }
    if let Some(ember) = ember {
        ember
            .append_to_graph(
                &mut graph,
                frame_index,
                external_imports,
                &host_texture_handles,
            )
            .map_err(|_| DisplayTransformGpuError::EmberGraph)?;
        let mut stages = *graph
            .blackboard()
            .get::<FrameGraphStages>()
            .expect("the host frame graph must publish its stage receipt");
        stages.ember_compute_passes = ember.compute_pass_count() as u32;
        graph.blackboard_mut().set(stages);
    }
    let output = transform.append(&mut graph, queue, source)?;
    let mut stages = *graph
        .blackboard()
        .get::<FrameGraphStages>()
        .expect("the host frame graph must publish its stage receipt");
    stages.present_source = output;
    stages.assert_contract();
    graph.blackboard_mut().set(stages);
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{
        BindGroupBindingId, DrawFunctionId, DrawFunctions, RenderBindGroupBinding,
        RenderBindGroupBindingContract, RenderBindGroupBuffer,
    };
    use crate::queue::RenderQueues;
    use crate::target::RenderTarget;
    use prism_ecs::world::World;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_driver::{
        BindGroupId, DepthLoadOp, DeviceDescriptor, Extent3d, Instance, LoadOp, RenderDevice,
        RenderPipelineId, RequestAdapterOptions, StoreOp, TextureDescriptor, TextureFormat,
        TextureState, TextureUsages, TextureViewDescriptor, TextureViewId,
    };
    use prism_render_driver_null::NullInstance;
    use prism_render_graph::{ExecutionPlan, ResourceIndex};

    fn null_device() -> prism_render_driver_null::NullDevice {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("the null backend always has an adapter");
        instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("the null backend always creates a device")
            .0
    }

    fn particle_device() -> prism_render_driver_null::NullDevice {
        prism_render_driver_null::NullDevice::new_with_capabilities(
            prism_render_driver::DeviceCapabilities {
                backend: prism_render_driver::Backend::Noop,
                features: prism_render_driver::Features::INDIRECT_DRAW,
                limits: prism_render_driver::Limits::desktop(),
            },
        )
    }

    fn assert_particle_stage_admission(
        target: &RenderTarget,
        particle: Option<&ParticleGpuState>,
        expected_passes: u32,
    ) {
        let (graph, color, _) = target_graph_output_with_trace_and_particles_and_external_views(
            target,
            &RenderQueues::new(),
            &canonical::registry(),
            None,
            None,
            particle,
            &FrameGraphTextureImports::default(),
        );
        let stages = *graph.blackboard().get::<FrameGraphStages>().unwrap();
        stages.assert_contract();
        assert_eq!(stages.particle_passes, expected_passes);
        assert_eq!(stages.present_source, color);
        let plan = graph.compile().expect("admitted particle graph compiles");
        let raster_passes = plan.attachments.iter().flatten().count();
        assert_eq!(
            raster_passes,
            stages.scene_passes as usize + stages.particle_passes as usize,
            "the receipt must count actual per-emitter raster admissions, not GPU state presence"
        );
    }

    #[test]
    fn particle_stage_receipt_is_zero_without_gpu_fabric() {
        let device = particle_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let state = ParticleGpuState::new(&device, std::sync::Arc::new(device.clone()), 1, 4)
            .expect("empty particle GPU state is valid");
        assert!(state.fabric_gpu_frame().is_none());

        assert_particle_stage_admission(&target, None, 0);
        assert_particle_stage_admission(&target, Some(&state), 0);

        state.destroy();
        target.destroy(&device);
    }

    #[test]
    fn particle_stage_receipt_counts_each_admitted_fabric_draw() {
        use prism_app::App;
        use prism_particle_app::{
            extract_particle_simulation_inputs, EmitterRuntimeComponent,
            ExtractedParticleSimulationInputs, ParticleSimulationConfig, ParticleSimulationPlugin,
        };
        use prism_render_architecture::particle::{
            emitter::{BurstSpawn, Emitter, EmitterShape, SpawnParams},
            runtime::{EmitterRuntime, EmitterRuntimeConfig},
            EmitterHandle, Vec3,
        };

        let device = particle_device();
        let queue = prism_render_driver_null::NullQueue::new();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let mut app = App::new();
        app.add_plugins(ParticleSimulationPlugin);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .capacity = 4;
        // One empty emitter contributes no draw. Two live emitters and the
        // remaining GPU spawn capacity each own an independent raster pass.
        for index in 0..3 {
            let mut runtime = EmitterRuntime::new(
                Emitter::new(
                    EmitterHandle(index),
                    EmitterShape::Point,
                    SpawnParams::default(),
                ),
                EmitterRuntimeConfig {
                    capacity: 1,
                    bursts: vec![BurstSpawn {
                        time: 0.1,
                        count: 1,
                    }],
                    lifetime: 10.0,
                    ..Default::default()
                },
                Vec3::ZERO,
                Vec3::ZERO,
            )
            .unwrap();
            if index != 0 {
                runtime.step(0.1);
            }
            app.world_mut().spawn(EmitterRuntimeComponent::new(runtime));
        }
        let mut render = World::new();
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        let mut inputs = render
            .remove_resource::<ExtractedParticleSimulationInputs>()
            .unwrap();
        let mut state =
            ParticleGpuState::new(&device, std::sync::Arc::new(device.clone()), 1, 4).unwrap();
        state
            .prepare_gpu_snapshot(
                &queue,
                &crate::config::RenderAppConfig::default(),
                &mut inputs,
                42,
            )
            .expect("bootstrap admits real physical Fabric bindings");
        assert_eq!(state.fabric_gpu_frame().unwrap().draws().len(), 3);
        assert_particle_stage_admission(&target, Some(&state), 3);

        inputs.enabled = false;
        state
            .prepare_gpu_snapshot(
                &queue,
                &crate::config::RenderAppConfig::default(),
                &mut inputs,
                43,
            )
            .expect("disabled particles revoke their Fabric admission");
        assert!(state.fabric_gpu_frame().is_none());
        assert_particle_stage_admission(&target, Some(&state), 0);

        state.destroy();
        target.destroy(&device);
    }

    // These tests compile the production host graph, but never record/execute
    // its closures. Placeholder pipeline/group ids describe draw contracts;
    // shader execution and numerical readback still require the GPU gates.
    fn attachment_test_item(draw: DrawFunctionId) -> PhaseItem {
        PhaseItem {
            pipeline: RenderPipelineId::from_parts(10, 1),
            draw,
            geometry: GenerationalHandle::new(0, 1),
            vertices: Vec::new(),
            vertex_sizes: Vec::new(),
            indices: None,
            index_size: None,
            first: 0,
            count: 3,
            base_vertex: 0,
            indirect: None,
            lod_level: 0,
            material: GenerationalHandle::new(0, 1),
            scene: GenerationalHandle::new(0, 1),
            order: 0,
        }
    }

    fn attachment_test_binding(identity: BindGroupBindingId, index: u32) -> RenderBindGroupBinding {
        RenderBindGroupBinding {
            identity,
            index,
            bind_group: BindGroupId::from_parts(20, 1),
            dynamic_offsets: vec![0, 256],
            buffers: vec![RenderBindGroupBuffer {
                buffer: BufferId::from_parts(30, 1),
                access: RenderBindGroupBufferAccess::Uniform,
            }],
            sampled_textures: Vec::new(),
            external_owners: Vec::new(),
        }
    }

    fn compile_without_lighting(
        target: &RenderTarget,
        queues: &RenderQueues,
        functions: &DrawFunctions,
    ) -> (ExecutionPlan, BTreeMap<TextureViewId, TextureHandle>) {
        let (graph, color, handles) =
            target_graph_output_with_trace_and_particles_and_external_views(
                target,
                queues,
                functions,
                None,
                None,
                None,
                &FrameGraphTextureImports::default(),
            );
        let stages = graph.blackboard().get::<FrameGraphStages>().unwrap();
        stages.assert_contract();
        assert_eq!(stages.scene_passes as usize, graph.pass_count());
        assert_eq!(stages.lighting_dispatches, 0);
        assert_eq!(stages.composite_passes, 0);
        assert_eq!(stages.present_source, color);
        let plan = graph
            .compile()
            .expect("no-lighting attachment graph compiles");
        assert_eq!(plan.order.len(), graph.pass_count());
        (plan, handles)
    }

    fn assert_no_lighting_attachment_contract(
        target: &RenderTarget,
        plan: &ExecutionPlan,
        handles: &BTreeMap<TextureViewId, TextureHandle>,
    ) -> [ResourceIndex; 6] {
        // f32 XYZ and full u32 identity remain independent planes. Coverage is
        // stored in normal alpha (f16), not the lossy/clamped sRGB8 albedo.
        let formats = [
            target.position_x_format(),
            target.position_y_format(),
            target.position_z_format(),
            target.normal_format(),
            target.albedo_format(),
            target.surface_id_format(),
        ];
        assert_eq!(
            formats,
            [
                TextureFormat::R32Float,
                TextureFormat::R32Float,
                TextureFormat::R32Float,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::R32Uint,
            ]
        );
        assert_eq!(target.format(), TextureFormat::Rgba16Float);
        let gbuffer = [
            target.position_x_view(),
            target.position_y_view(),
            target.position_z_view(),
            target.normal_view(),
            target.albedo_view(),
            target.surface_id_view(),
        ]
        .map(|view| handles[&view].resource());
        assert_eq!(
            gbuffer.len(),
            crate::forward_lit::SURFACE_GBUFFER_TARGET_COUNT,
            "frame graph GBuffer attachment count must match the surface shader ABI"
        );
        let hdr = handles[&target.view()].resource();
        let depth = handles[&target.depth_view()].resource();
        let mut gbuffer_clears = 0;
        let mut hdr_clears = 0;
        for attachments in &plan.attachments {
            let raster = attachments
                .as_ref()
                .expect("no-lighting scene passes are raster");
            let resources: Vec<_> = raster.colors.iter().map(|a| a.resource).collect();
            if resources.as_slice() == gbuffer.as_slice() {
                assert_eq!(
                    raster.colors.len(),
                    crate::forward_lit::SURFACE_GBUFFER_TARGET_COUNT,
                    "opaque raster attachment count must match SurfaceEncodeProvider"
                );
            }
            assert!(
                resources.as_slice() == gbuffer.as_slice() || resources == [hdr],
                "GBuffer and HDR must never share an attachment set"
            );
            // Keep the concrete format accounting beside the ABI assertion:
            // WebGPU charges sRGB8 RGBA 8 attachment bytes, not its 4 storage
            // bytes. Six GBuffer targets cost 4+4+4+8+8+4 = 32; HDR costs 8.
            let bytes: u32 = raster
                .colors
                .iter()
                .map(|attachment| {
                    if attachment.resource == hdr {
                        8
                    } else {
                        [4, 4, 4, 8, 8, 4][gbuffer
                            .iter()
                            .position(|resource| *resource == attachment.resource)
                            .unwrap()]
                    }
                })
                .sum();
            assert!(bytes <= 32);
            assert!(raster.colors.iter().all(|a| a.store == StoreOp::Store));
            if raster
                .colors
                .iter()
                .all(|a| matches!(a.load, LoadOp::Clear(_)))
            {
                if resources.as_slice() == gbuffer.as_slice() {
                    gbuffer_clears += 1;
                    for (index, attachment) in raster.colors.iter().enumerate() {
                        assert_eq!(
                            attachment.load,
                            LoadOp::Clear(Color::TRANSPARENT),
                            "all six GBuffer planes must be initialized to zero; only normal alpha carries coverage"
                        );
                    }
                    let attachment = raster.depth.unwrap();
                    assert_eq!(attachment.resource, depth);
                    assert!(attachment.writes);
                    assert_eq!(attachment.depth_load, DepthLoadOp::Clear(0.0));
                    assert_eq!(attachment.depth_store, StoreOp::Store);
                } else {
                    hdr_clears += 1;
                    assert_eq!(
                        raster.colors[0].load,
                        LoadOp::Clear(Color::TRANSPARENT),
                        "no-lighting HDR initialization must match the production zero clear"
                    );
                    assert!(
                        raster.depth.is_none(),
                        "HDR clear must not clear shared depth twice"
                    );
                }
            } else {
                assert!(raster.colors.iter().all(|a| a.load == LoadOp::Load));
                let attachment = raster.depth.unwrap();
                assert_eq!(attachment.resource, depth);
                assert!(
                    attachment.writes,
                    "both opaque producers must write shared depth"
                );
                assert_eq!(attachment.depth_load, DepthLoadOp::Load);
                assert_eq!(attachment.depth_store, StoreOp::Store);
            }
        }
        assert_eq!(
            gbuffer_clears, 1,
            "reset coverage even when no surface remains"
        );
        assert_eq!(
            hdr_clears, 1,
            "without composite, HDR needs its own initializer"
        );
        gbuffer
    }

    #[test]
    fn no_lighting_empty_frame_clears_six_gbuffer_attachments_and_hdr_separately() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let (plan, handles) =
            compile_without_lighting(&target, &RenderQueues::new(), &canonical::registry());
        assert_eq!(plan.order.len(), 2);
        assert_no_lighting_attachment_contract(&target, &plan, &handles);
        target.destroy(&device);
    }

    #[test]
    fn no_lighting_surface_draw_keeps_six_attachments_and_loads_zero_coverage_clear() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let mut functions = canonical::registry();
        let binding =
            attachment_test_binding(canonical::SURFACE_MATERIAL, canonical::SURFACE_GROUP);
        let static_draw = functions.register_mesh_with_bind_groups(vec![binding.clone()]);
        for draw in [canonical::MESH, static_draw] {
            let mut queues = RenderQueues::new();
            queues.push_with_bind_groups(
                RenderPhase::Opaque,
                attachment_test_item(draw),
                if draw == canonical::MESH {
                    vec![binding.clone()]
                } else {
                    Vec::new()
                },
            );
            let (plan, handles) = compile_without_lighting(&target, &queues, &functions);
            assert_eq!(plan.order.len(), 3);
            let gbuffer = assert_no_lighting_attachment_contract(&target, &plan, &handles);
            let passes: Vec<_> = plan
                .attachments
                .iter()
                .flatten()
                .filter(|raster| {
                    raster.colors.len() == crate::forward_lit::SURFACE_GBUFFER_TARGET_COUNT
                })
                .collect();
            assert_eq!(passes.len(), 2);
            assert_eq!(passes[0].colors[3].load, LoadOp::Clear(Color::TRANSPARENT));
            for (clear, draw) in passes[0].colors.iter().zip(&passes[1].colors) {
                assert_eq!(draw.load, LoadOp::Load);
                assert_eq!(draw.output_version, clear.output_version + 1);
            }
            assert_eq!(
                passes[1]
                    .colors
                    .iter()
                    .map(|a| a.resource)
                    .collect::<Vec<_>>(),
                gbuffer
            );
        }
        target.destroy(&device);
    }

    #[test]
    fn no_lighting_single_hdr_providers_keep_one_target_and_binding_sidecars() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let mut functions = canonical::registry();
        let custom_binding = attachment_test_binding(BindGroupBindingId(73), 0);
        let custom_draw = functions.register_with_bind_groups_and_dynamic_contract(
            crate::mesh_draw::mesh_draw,
            Vec::new(),
            vec![RenderBindGroupBindingContract {
                identity: custom_binding.identity,
                index: 0,
            }],
        );
        let surface_binding =
            attachment_test_binding(canonical::SURFACE_MATERIAL, canonical::SURFACE_GROUP);
        for hdr_draw in [canonical::MESH_UNBOUND, custom_draw] {
            for mixed in [false, true] {
                let mut queues = RenderQueues::new();
                let hdr_item = attachment_test_item(hdr_draw);
                let hdr_bindings = if hdr_draw == custom_draw {
                    vec![custom_binding.clone()]
                } else {
                    Vec::new()
                };
                // Put HDR first: splitting must preserve sidecar association,
                // while depth dependencies still order the surface producer first.
                queues.push_with_bind_groups(
                    RenderPhase::Opaque,
                    hdr_item.clone(),
                    hdr_bindings.clone(),
                );
                if mixed {
                    queues.push_with_bind_groups(
                        RenderPhase::Opaque,
                        attachment_test_item(canonical::MESH),
                        vec![surface_binding.clone()],
                    );
                }
                let (surface, hdr) = split_surface_opaque(&queues, &functions);
                assert_eq!(surface.phase(RenderPhase::Opaque).len(), usize::from(mixed));
                assert_eq!(hdr.phase(RenderPhase::Opaque), &[hdr_item]);
                assert_eq!(
                    hdr.dynamic_bind_groups(RenderPhase::Opaque, 0),
                    hdr_bindings
                );
                if mixed {
                    assert_eq!(
                        surface.dynamic_bind_groups(RenderPhase::Opaque, 0),
                        &[surface_binding.clone()]
                    );
                }
                let (plan, handles) = compile_without_lighting(&target, &queues, &functions);
                assert_eq!(plan.order.len(), 3 + usize::from(mixed));
                assert_no_lighting_attachment_contract(&target, &plan, &handles);
                let hdr_resource = handles[&target.view()].resource();
                let hdr_passes: Vec<_> = plan
                    .attachments
                    .iter()
                    .flatten()
                    .filter(|raster| {
                        raster.colors.len() == 1 && raster.colors[0].resource == hdr_resource
                    })
                    .collect();
                assert_eq!(hdr_passes.len(), 2, "one HDR clear and one HDR opaque pass");
                assert_eq!(hdr_passes[1].colors[0].load, LoadOp::Load);
                assert_eq!(
                    hdr_passes[1].colors[0].output_version,
                    hdr_passes[0].colors[0].output_version + 1
                );
                if mixed {
                    let surface_draw = plan
                        .attachments
                        .iter()
                        .position(|raster| {
                            raster.as_ref().is_some_and(|raster| {
                                raster.colors.len()
                                    == crate::forward_lit::SURFACE_GBUFFER_TARGET_COUNT
                                    && raster.colors[0].load == LoadOp::Load
                            })
                        })
                        .unwrap();
                    let hdr_draw = plan
                        .attachments
                        .iter()
                        .position(|raster| {
                            raster.as_ref().is_some_and(|raster| {
                                raster.colors.len() == 1 && raster.colors[0].load == LoadOp::Load
                            })
                        })
                        .unwrap();
                    assert!(surface_draw < hdr_draw);
                }
            }
        }
        target.destroy(&device);
    }

    #[test]
    fn owner_texture_imports_extract_without_losing_view_metadata() {
        let view = TextureViewId::from_parts(7, 3);
        let import = FrameGraphTextureImport {
            texture: TextureId::from_parts(6, 3),
            view,
            desc: TextureDesc::depth(TextureFormat::Depth32Float)
                .with_size(prism_render_graph::SizeClass::absolute_2d(64, 32)),
            entry_state: TextureState::depth_read(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        };
        let mut main = World::new();
        let mut published = FrameGraphTextureImports::default();
        published.set(import.clone());
        main.insert_resource(published);
        let mut render = World::new();

        extract_frame_graph_texture_imports(&mut main, &mut render);

        let extracted = render.resource::<FrameGraphTextureImports>();
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted.iter().next(), Some(&import));
    }

    #[test]
    fn host_frame_graph_imports_owner_views_and_deduplicates_builtins() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let external_texture = device.create_texture(&TextureDescriptor::new_2d(
            Extent3d::new_2d(4, 4),
            TextureFormat::Rgba8Unorm,
            TextureUsages::TEXTURE_BINDING,
        ));
        let external_view =
            device.create_texture_view(external_texture, &TextureViewDescriptor::default());
        let mut imports = FrameGraphTextureImports::default();
        imports.set(FrameGraphTextureImport {
            texture: external_texture,
            view: external_view,
            desc: TextureDesc::color(TextureFormat::Rgba8Unorm)
                .with_size(prism_render_graph::SizeClass::absolute_2d(4, 4))
                .with_usage(TextureUsages::TEXTURE_BINDING),
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        });
        // A built-in view is intentionally published too: it must resolve to
        // the host's already-imported graph handle, not a duplicate import.
        imports.set(FrameGraphTextureImport {
            texture: target.depth_texture(),
            view: target.depth_view(),
            desc: TextureDesc::color(TextureFormat::Rgba8Unorm),
            entry_state: TextureState::initial(),
            exit_state: TextureState::initial(),
            mip_levels: 1,
            array_layers: 1,
        });

        let (graph, _color, handles) =
            target_graph_output_with_trace_and_particles_and_external_views(
                &target,
                &RenderQueues::default(),
                &DrawFunctions::default(),
                None,
                None,
                None,
                &imports,
            );
        assert_eq!(handles.len(), 9, "eight builtins plus one owner import");
        assert!(handles.contains_key(&external_view));
        assert!(handles.contains_key(&target.depth_view()));
        assert!(graph.compile().is_ok());

        device.destroy_texture_view(external_view);
        device.destroy_texture(external_texture);
        target.destroy(&device);
    }

    #[test]
    fn render_bind_group_sampled_texture_is_a_frame_graph_read() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let texture = device.create_texture(&TextureDescriptor::new_2d(
            Extent3d::new_2d(4, 4),
            TextureFormat::Rgba8Unorm,
            TextureUsages::TEXTURE_BINDING,
        ));
        let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
        let mut imports = FrameGraphTextureImports::default();
        imports.set(FrameGraphTextureImport {
            texture,
            view,
            desc: TextureDesc::color(TextureFormat::Rgba8Unorm)
                .with_size(prism_render_graph::SizeClass::absolute_2d(4, 4))
                .with_usage(TextureUsages::TEXTURE_BINDING),
            entry_state: TextureState::initial(),
            exit_state: TextureState::initial(),
            mip_levels: 1,
            array_layers: 1,
        });
        let mut binding =
            attachment_test_binding(canonical::SURFACE_MATERIAL, canonical::SURFACE_GROUP);
        binding.sampled_textures.push(view);
        let mut queues = RenderQueues::new();
        queues.push_with_bind_groups(
            RenderPhase::Opaque,
            attachment_test_item(canonical::MESH),
            vec![binding],
        );
        let (graph, _, handles) = target_graph_output_with_trace_and_particles_and_external_views(
            &target,
            &queues,
            &canonical::registry(),
            None,
            None,
            None,
            &imports,
        );
        let plan = graph.compile().expect("sampled imported view compiles");
        let resource = handles[&view].resource();
        let surface_pass = plan
            .attachments
            .iter()
            .position(|attachments| {
                attachments.as_ref().is_some_and(|raster| {
                    raster.colors.len() == crate::forward_lit::SURFACE_GBUFFER_TARGET_COUNT
                        && raster.colors[0].load == LoadOp::Load
                })
            })
            .expect("surface draw pass exists");
        assert!(
            plan.barriers[surface_pass].iter().any(|barrier| {
                matches!(barrier, prism_render_graph::GraphBarrier::Texture(texture_barrier)
                if texture_barrier.texture == resource
                    && texture_barrier.after == TextureState::shader_read(
                        prism_render_driver::PipelineStages::FRAGMENT_SHADER
                    ))
            }),
            "surface pass must transition the imported view to sampled-read state"
        );
        device.destroy_texture_view(view);
        device.destroy_texture(texture);
        target.destroy(&device);
    }

    #[test]
    fn render_bind_group_sampled_texture_without_import_is_rejected() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let missing_view = TextureViewId::from_parts(900, 1);
        let mut binding =
            attachment_test_binding(canonical::SURFACE_MATERIAL, canonical::SURFACE_GROUP);
        binding.sampled_textures.push(missing_view);
        let mut queues = RenderQueues::new();
        queues.push_with_bind_groups(
            RenderPhase::Opaque,
            attachment_test_item(canonical::MESH),
            vec![binding],
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            target_graph_output_with_trace_and_particles_and_external_views(
                &target,
                &queues,
                &canonical::registry(),
                None,
                None,
                None,
                &FrameGraphTextureImports::default(),
            )
        }));
        assert!(
            result.is_err(),
            "missing sampled-view imports must fail closed"
        );
        target.destroy(&device);
    }

    #[test]
    fn render_bind_group_external_owner_is_retained_by_each_output_graph() {
        let device = null_device();
        let target = RenderTarget::new(&device, Extent3d::new_2d(16, 8));
        let owner: alloc::sync::Arc<dyn core::any::Any + Send + Sync> =
            alloc::sync::Arc::new(17_u32);
        let weak_owner = alloc::sync::Arc::downgrade(&owner);
        let mut binding =
            attachment_test_binding(canonical::SURFACE_MATERIAL, canonical::SURFACE_GROUP);
        binding.external_owners.push(owner.clone());
        let draw = canonical::MESH;
        let mut queues = RenderQueues::new();
        queues.push_with_bind_groups(
            RenderPhase::Opaque,
            attachment_test_item(draw),
            vec![binding],
        );

        let (first, _, _) = target_graph_output_with_trace_and_particles_and_external_views(
            &target,
            &queues,
            &canonical::registry(),
            None,
            None,
            None,
            &FrameGraphTextureImports::default(),
        );
        let (second, _, _) = target_graph_output_with_trace_and_particles_and_external_views(
            &target,
            &queues,
            &canonical::registry(),
            None,
            None,
            None,
            &FrameGraphTextureImports::default(),
        );

        assert_eq!(first.external_owner_count(), 1);
        assert_eq!(second.external_owner_count(), 1);
        drop(queues);
        drop(owner);
        assert!(weak_owner.upgrade().is_some());
        drop(first);
        assert!(weak_owner.upgrade().is_some());
        drop(second);
        assert!(weak_owner.upgrade().is_none());
        target.destroy(&device);
    }
}
