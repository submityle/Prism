//! Building the frame's render graph.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_driver::{
    BufferId, BufferState, BufferUsages, TextureFormat, TextureState, TextureUsages,
};
use prism_render_graph::{
    BufferDesc, BufferHandle, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
};

use crate::display_transform_gpu::{DisplayTransformGpuError, OffscreenDisplayTransform};
use crate::draw::DrawFunctions;
use crate::lighting_gpu::{LightingGpuPass, OUTPUT_BYTES};
use crate::lighting_trace_gpu::TraceGraphRecorder;
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
    pub lighting_dispatches: u32,
    /// Graph texture consumed by the final present pass (OCIO output for windows).
    pub present_source: prism_render_graph::TextureHandle,
}

impl FrameGraphStages {
    fn for_graph(
        queues: &RenderQueues,
        lighting: Option<LightingGpuPass>,
        trace: Option<TraceGraphRecorder>,
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
        assert_eq!(
            self.lighting_dispatches,
            self.direct_recipes + self.gi_recipes + self.composite_passes,
            "lighting dispatch accounting must match admitted recipes"
        );
    }
}

/// The texel format of the render sub-app's offscreen frame target.
///
/// `Rgba8Unorm` is the portable baseline every backend can render to and read
/// back, so a frame recorded here is verifiable on any GPU.
pub const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

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
/// graph once and declared as a vertex or index read. The recorder then binds
/// the same ids, so the declaration and the recorded command agree by
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

    // Import each bound buffer once. An imported buffer's descriptor is only the
    // usage this frame declares — the graph neither owns nor sizes it, and the
    // size is the owner's business (a mesh buffer, a scene buffer, a bin stream).
    let mut imported: BTreeMap<BufferId, BufferHandle> = BTreeMap::new();
    for item in RenderPhase::ORDER
        .into_iter()
        .flat_map(|phase| queues.phase(phase))
    {
        let mut bind = |graph: &mut RenderGraph, id: BufferId, usage: BufferUsages| {
            if let Some(handle) = imported.get(&id) {
                return *handle;
            }
            let handle = graph.import_buffer(
                "frame-bound-buffer",
                BufferDesc::new(0, usage),
                ImportedBuffer {
                    buffer: id,
                    entry_state: BufferState::initial(),
                },
            );
            imported.insert(id, handle);
            handle
        };
        for buffer in &item.vertices {
            let _ = bind(&mut graph, *buffer, BufferUsages::VERTEX);
        }
        if let Some((indices, _)) = item.indices {
            let _ = bind(&mut graph, indices, BufferUsages::INDEX);
        }
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
        for item in queues.phase(phase) {
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
        let stages = FrameGraphStages::for_graph(queues, None, trace, color);
        stages.assert_contract();
        graph.blackboard_mut().set(stages);
        return (graph, color);
    };

    let output = graph.import_buffer(
        "prism-lighting-final-output",
        BufferDesc::new(OUTPUT_BYTES, BufferUsages::STORAGE | BufferUsages::COPY_SRC),
        ImportedBuffer {
            buffer: lighting.output,
            entry_state: BufferState::initial(),
        },
    );
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
                let _candidates = builder.read_storage_buffer(candidates);
                move |context| {
                    context
                        .set_compute_pipeline(lighting.gi_pipeline)
                        .set_compute_bind_group(0, lighting.gi_bind_group, Vec::new())
                        .dispatch(lighting.workgroups, 1, 1);
                }
            },
        );
    }
    graph.add_compute_pass(
        "prism-lighting-composite",
        PassFlags::SIDE_EFFECT,
        |builder| {
            let _output = builder.write_storage_buffer(output);
            let _direct_output = builder.read_storage_buffer(direct_output);
            let _gi_output = builder.read_storage_buffer(gi_output);
            let _params = builder.read_uniform(params);
            move |context| {
                context
                    .set_compute_pipeline(lighting.composite_pipeline)
                    .set_compute_bind_group(0, lighting.composite_bind_group, Vec::new())
                    .dispatch(lighting.workgroups, 1, 1);
            }
        },
    );

    let stages = FrameGraphStages::for_graph(queues, Some(lighting), trace, color);
    stages.assert_contract();
    graph.blackboard_mut().set(stages);
    (graph, color)
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
    let (mut graph, source) =
        target_graph_output_with_trace(target, queues, functions, lighting, trace);
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
