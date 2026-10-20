//! Building the frame's render graph.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_driver::{
    BufferId, BufferState, BufferUsages, TextureFormat, TextureState, TextureUsages, TextureViewId,
};
use prism_render_gpudriven::ParticleStage;
use prism_render_gpudriven::{CompiledGraphGpuGraphImport, CompiledGraphGpuResourceSlot};
use prism_render_graph::{
    BufferDesc, BufferHandle, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
    TextureHandle,
};

use crate::display_transform_gpu::{DisplayTransformGpuError, OffscreenDisplayTransform};
use crate::draw::{DrawFunctions, RenderBindGroupBufferAccess};
use crate::lighting_gpu::{LightingGpuPass, OUTPUT_BYTES};
use crate::lighting_trace_gpu::TraceGraphRecorder;
use crate::particle::EmberGpuAssetRegistry;
use crate::queue::RenderQueues;
use crate::sets::RenderPhase;
use crate::target::RenderTarget;

/// The host-visible stages represented by one frame graph.
///
/// This is deliberately a receipt, not a control flag. It records the
/// engine-neutral scene/visibility submission, the trace-result boundary, the
/// admitted GI recipe, composite, and the texture that presentation consumes.
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
    pub particle_passes: u32,
    pub ember_compute_passes: u32,
    pub lighting_dispatches: u32,
    /// Graph texture consumed by the final present pass (OCIO output for windows).
    pub present_source: prism_render_graph::TextureHandle,
}

impl FrameGraphStages {
    fn for_graph(
        queues: &RenderQueues,
        lighting: Option<LightingGpuPass>,
        trace: Option<TraceGraphRecorder>,
        particle: Option<&ParticleStage>,
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
        let scene_passes = 1 + RenderPhase::ORDER
            .into_iter()
            .filter(|phase| !queues.phase(*phase).is_empty())
            .count() as u32;
        let trace_dispatches = u32::from(trace.is_some());
        let candidate_build_dispatches = u32::from(
            trace.is_some_and(|trace| trace.candidates.is_some())
                && lighting.is_some_and(|pass| pass.run_gi),
        );
        let direct_recipes = lighting.is_some_and(|pass| pass.run_direct) as u32;
        let gi_recipes = lighting.is_some_and(|pass| pass.run_gi) as u32;
        let lighting_dispatches = lighting.map_or(0, |pass| {
            u32::from(pass.run_direct) + u32::from(pass.run_gi) + 1
        });
        Self {
            scene_passes,
            trace_dispatches,
            candidate_build_dispatches,
            direct_recipes,
            gi_recipes,
            composite_passes: u32::from(lighting.is_some()),
            present_passes: 0,
            particle_passes: u32::from(particle.is_some()),
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
            self.present_passes <= 1,
            "a frame has at most one presentation"
        );
        assert!(
            self.particle_passes <= 1,
            "a frame has at most one particle raster stage"
        );
        assert!(
            (self.ember_compute_passes as usize)
                <= prism_render_graph::capacity::GraphLimits::default().passes,
            "a frame has too many Ember compute stages"
        );
        assert_eq!(
            self.lighting_dispatches,
            self.direct_recipes + self.gi_recipes + self.composite_passes,
            "lighting dispatch accounting must match admitted recipes"
        );
    }
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

/// Builds the frame graph: clear the target, then record the queued draws.
///
/// One raster pass per phase that has items, in [`RenderPhase::ORDER`], each
/// drawing into the frame target. A buffer an item binds must be *declared* by
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

/// The host graph constructor with an optional CPU-authoritative particle pass.
pub(crate) fn target_graph_output_with_trace_and_particles(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleStage>,
) -> (RenderGraph, prism_render_graph::TextureHandle) {
    let (graph, color, _) = target_graph_output_with_trace_and_particles_and_external_views(
        target, queues, functions, lighting, trace, particle,
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
    particle: Option<&ParticleStage>,
) -> (
    RenderGraph,
    TextureHandle,
    BTreeMap<TextureViewId, TextureHandle>,
) {
    let mut graph = RenderGraph::new(target.viewport().extent());
    let mut color = graph.import_texture(
        "prism-render-app-target",
        TextureDesc::color(target.format())
            .with_usage(TextureUsages::COPY_SRC.union(TextureUsages::TEXTURE_BINDING)),
        ImportedTexture {
            texture: target.texture(),
            default_view: target.view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
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
    let mut imported: BTreeMap<BufferId, BufferHandle> = BTreeMap::new();
    for (id, usage) in required_usages {
        let handle = graph.import_buffer(
            "frame-bound-buffer",
            BufferDesc::new(0, usage),
            ImportedBuffer {
                buffer: id,
                entry_state: BufferState::initial(),
            },
        );
        imported.insert(id, handle);
    }

    // A frame that rasterizes geometry needs depth, and the pipelines it
    // specializes must agree with the attachment: the queue stage passes
    // `RENDER_DEPTH_FORMAT` into the variant for exactly this reason.
    let mut depth = graph.import_texture(
        "prism-render-app-depth",
        TextureDesc::depth(target.depth_format()),
        ImportedTexture {
            texture: target.depth_texture(),
            default_view: target.depth_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::depth_write(),
            mip_levels: 1,
            array_layers: 1,
        },
    );

    // Scene ownership is first in the host chain: clear the target, then
    // rasterize the visibility-derived queues. Lighting consumes that scene
    // result after the raster work has been declared, so graph order and the
    // host receipt agree even when a phase is empty.
    graph.add_raster_pass(
        "prism-render-app-clear",
        PassFlags::SIDE_EFFECT,
        |builder| {
            color = builder.color_attachment(color);
            depth = builder.depth_attachment(depth);
            |_ctx| {}
        },
    );

    for phase in RenderPhase::ORDER {
        if queues.phase(phase).is_empty() {
            continue;
        }
        let mut vertex_reads = Vec::new();
        let mut index_reads = Vec::new();
        let mut indirect_reads = Vec::new();
        let mut uniform_reads = Vec::new();
        let mut storage_reads = Vec::new();
        for (item_index, item) in queues.phase(phase).iter().enumerate() {
            for buffer in &item.vertices {
                if let Some(handle) = imported.get(buffer) {
                    vertex_reads.push(*handle);
                }
            }
            if let Some((indices, _)) = item.indices
                && let Some(handle) = imported.get(&indices)
            {
                index_reads.push(*handle);
            }
            if let Some(args) = item.indirect {
                indirect_reads.push(args.args);
            }
            for binding in item.bind_groups(functions) {
                for resource in &binding.buffers {
                    if let Some(handle) = imported.get(&resource.buffer) {
                        match resource.access {
                            RenderBindGroupBufferAccess::Uniform => uniform_reads.push(*handle),
                            RenderBindGroupBufferAccess::Storage => storage_reads.push(*handle),
                        }
                    }
                }
            }
            for binding in queues.dynamic_bind_groups(phase, item_index) {
                for resource in &binding.buffers {
                    if let Some(handle) = imported.get(&resource.buffer) {
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
                let queues = queues.clone();
                let functions = functions.clone();
                move |ctx| queues.record(phase, ctx, &functions)
            },
        );
    }

    if let Some(particle) = particle {
        color = particle.append_render_to_graph(&mut graph, color);
    }

    let mut trace_hits = None;
    let mut trace_inputs = None;
    let mut trace_pass_dependency = None;
    if let Some(trace) = trace {
        let nodes = graph.import_buffer(
            "prism-lighting-trace-nodes",
            BufferDesc::new(trace.nodes.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: trace.nodes.id,
                entry_state: BufferState::initial(),
            },
        );
        let triangles = graph.import_buffer(
            "prism-lighting-trace-triangles",
            BufferDesc::new(trace.triangles.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: trace.triangles.id,
                entry_state: BufferState::initial(),
            },
        );
        let rays = graph.import_buffer(
            "prism-lighting-trace-rays",
            BufferDesc::new(trace.rays.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: trace.rays.id,
                entry_state: BufferState::initial(),
            },
        );
        let mut hits = graph.import_buffer(
            "prism-lighting-trace-hits",
            BufferDesc::new(trace.hits.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: trace.hits.id,
                entry_state: BufferState::initial(),
            },
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
                    let _scene_raster_boundary = builder.sample(color);
                    let _nodes = builder.read_storage_buffer(nodes);
                    let _triangles = builder.read_storage_buffer(triangles);
                    let _rays = builder.read_storage_buffer(rays);
                    hits = builder.write_storage_buffer(hits);
                    move |context| trace.record_graph(context)
                },
            )
            .expect("trace pass fits the host frame graph");
        trace_pass_dependency = Some(trace_pass);
        trace_hits = Some(hits);
        trace_inputs = Some((triangles, rays));
    }

    let Some(lighting) = lighting else {
        let stages = FrameGraphStages::for_graph(queues, None, trace, particle, None, color);
        stages.assert_contract();
        graph.blackboard_mut().set(stages);
        let mut external_texture_handles = BTreeMap::new();
        external_texture_handles.insert(target.view(), color);
        return (graph, color, external_texture_handles);
    };

    let mut direct_output = graph.import_buffer(
        "prism-lighting-direct-output",
        BufferDesc::new(OUTPUT_BYTES, BufferUsages::STORAGE),
        ImportedBuffer {
            buffer: lighting.direct_output,
            entry_state: BufferState::initial(),
        },
    );
    let mut gi_output = graph.import_buffer(
        "prism-lighting-gi-output",
        BufferDesc::new(OUTPUT_BYTES, BufferUsages::STORAGE),
        ImportedBuffer {
            buffer: lighting.gi_output,
            entry_state: BufferState::initial(),
        },
    );
    let params = graph.import_buffer(
        "prism-lighting-plan-params",
        BufferDesc::new(32, BufferUsages::UNIFORM),
        ImportedBuffer {
            buffer: lighting.params,
            entry_state: BufferState::initial(),
        },
    );
    let directionals = graph.import_buffer(
        "prism-lighting-directionals",
        BufferDesc::new(0, BufferUsages::STORAGE),
        ImportedBuffer {
            buffer: lighting.directionals,
            entry_state: BufferState::initial(),
        },
    );
    let punctuals = graph.import_buffer(
        "prism-lighting-punctuals",
        BufferDesc::new(0, BufferUsages::STORAGE),
        ImportedBuffer {
            buffer: lighting.punctuals,
            entry_state: BufferState::initial(),
        },
    );
    let areas = graph.import_buffer(
        "prism-lighting-areas",
        BufferDesc::new(0, BufferUsages::STORAGE),
        ImportedBuffer {
            buffer: lighting.areas,
            entry_state: BufferState::initial(),
        },
    );
    let mut candidates = graph.import_buffer(
        "prism-lighting-gi-candidates",
        BufferDesc::new(
            crate::lighting_gpu::GI_CANDIDATE_BUFFER_BYTES,
            BufferUsages::STORAGE,
        ),
        ImportedBuffer {
            buffer: lighting.candidates,
            entry_state: BufferState::initial(),
        },
    );
    let surfaces = graph.import_buffer(
        "prism-lighting-surfaces",
        BufferDesc::new(
            crate::lighting_gpu::LIGHTING_SURFACE_BUFFER_BYTES,
            BufferUsages::STORAGE,
        ),
        ImportedBuffer {
            buffer: lighting.surfaces,
            entry_state: BufferState::initial(),
        },
    );
    if let (Some(trace), Some(hits), Some((trace_triangles, trace_rays)), true) =
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
        let radiance = graph.import_buffer(
            "prism-lighting-trace-radiance",
            BufferDesc::new(radiance.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: radiance.id,
                entry_state: BufferState::initial(),
            },
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
                    let _radiance = builder.read_storage_buffer(radiance);
                    let _surfaces = builder.read_storage_buffer(surfaces);
                    let _params = builder.read_uniform(params);
                    candidates = builder.write_storage_buffer(candidates);
                    move |context| trace.record_candidate_graph(context)
                },
            )
            .expect("GI candidate pass fits the host frame graph");
    }
    if lighting.run_direct {
        graph.add_compute_pass(
            "prism-lighting-direct-dispatch",
            PassFlags::SIDE_EFFECT,
            |builder| {
                direct_output = builder.write_storage_buffer(direct_output);
                let _params = builder.read_uniform(params);
                let _directionals = builder.read_storage_buffer(directionals);
                let _punctuals = builder.read_storage_buffer(punctuals);
                let _areas = builder.read_storage_buffer(areas);
                let _surfaces = builder.read_storage_buffer(surfaces);
                move |context| {
                    context
                        .set_compute_pipeline(lighting.direct_pipeline)
                        .set_compute_bind_group(0, lighting.direct_bind_group, Vec::new())
                        .dispatch(lighting.workgroups, 1, 1);
                }
            },
        );
    }
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
    compile_error!(
        "lighting composite requires a screen-space surface/radiance image binding; the current LightingGpuPass exposes only per-surface storage samples, so a real Rgba16Float composite cannot be recorded without a new external producer contract"
    );

    let stages = FrameGraphStages::for_graph(queues, Some(lighting), trace, particle, None, color);
    stages.assert_contract();
    graph.blackboard_mut().set(stages);
    let mut external_texture_handles = BTreeMap::new();
    external_texture_handles.insert(target.view(), color);
    (graph, color, external_texture_handles)
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

/// Trace-aware display-transform graph constructor with an optional particle pass.
pub(crate) fn target_graph_with_display_transform_and_trace_and_particles(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleStage>,
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
        &BTreeMap::new(),
        &BTreeMap::new(),
        0,
        transform,
        queue,
    )
}

/// Trace-aware display-transform graph constructor with optional Ember compute
/// graphs in addition to the CPU-authoritative sprite bridge.
pub(crate) fn target_graph_with_display_transform_and_trace_particles_and_ember(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
    lighting: Option<LightingGpuPass>,
    trace: Option<TraceGraphRecorder>,
    particle: Option<&ParticleStage>,
    ember: Option<&EmberGpuAssetRegistry>,
    external_imports: &BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuGraphImport>,
    // Handles for caller-owned views already imported into this frame graph.
    external_texture_handles: &BTreeMap<TextureViewId, TextureHandle>,
    frame_index: u64,
    transform: &OffscreenDisplayTransform,
    queue: &dyn prism_render_driver::RenderQueue,
) -> Result<RenderGraph, DisplayTransformGpuError> {
    let (mut graph, source, mut host_texture_handles) =
        target_graph_output_with_trace_and_particles_and_external_views(
            target, queues, functions, lighting, trace, particle,
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
