//! Lowering for heightfield placement into Prism's executable render graph.
//!
//! The placement kernels remain owned by the caller. This module owns only the
//! graph declaration and command recording contract: callers provide imported
//! RHI resources and pipeline/bind-group identities from the exact device that
//! will record the graph. No backend object is allocated, submitted, or
//! retired here.

use core::fmt;
use std::collections::BTreeMap;
use std::vec::Vec;

use prism_render_driver::{
    BindGroupId, BufferId, BufferUsages, ComputePipelineId, Extent3d, RenderPipelineId,
};
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ExecuteContext, ImportedBuffer, ImportedTexture,
    PassFlags, RenderGraph, TextureDesc,
};

use crate::world_placement_graph::PlacementStream;

/// WGSL for clearing only the instance-count word of a placement indirect
/// argument buffer. Bind group 0 contains that storage buffer at binding 0.
pub const PLACEMENT_INDIRECT_COUNT_RESET_WGSL: &str =
    include_str!("shaders/placement_indirect_count_reset.wgsl");

/// One imported buffer together with its physical byte footprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementImportedBuffer {
    /// Physical RHI buffer and entry state.
    pub imported: ImportedBuffer,
    /// Physical byte size used by graph validation and diagnostics.
    pub bytes: u64,
}

/// One compute stage supplied by the renderer-owned pipeline registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementComputeStage {
    /// Registered compute pipeline on the graph's device table.
    pub pipeline: ComputePipelineId,
    /// Registered bind group compatible with `pipeline`.
    pub bind_group: BindGroupId,
    /// Direct workgroup dimensions. Zero is rejected before graph mutation.
    pub dispatch: [u32; 3],
}

/// The raster consumer for one compacted placement stream.
///
/// The opaque bind group must bind the same compact buffer declared to this
/// pass; with deletion enabled that is the filtered deletion output. The graph
/// can enforce resource barriers but cannot introspect an RHI bind group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementDrawStage {
    /// Registered render pipeline on the graph's device table.
    pub pipeline: RenderPipelineId,
    /// Registered bind group reading placement records and attributes.
    pub bind_group: BindGroupId,
    /// Byte offset of the indirect argument record.
    pub indirect_offset: u64,
}

/// Candidate-dependent placement work.
///
/// Empty placement requests intentionally omit this value. They still execute
/// the reset stage so stale indirect/status state cannot escape into a later
/// frame, but they do not fabricate a nonzero dispatch or an empty draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementRenderWork {
    /// Canonical XZ source generation kernel.
    pub source: PlacementComputeStage,
    /// Exact heightfield projection kernel.
    pub project: PlacementComputeStage,
    /// Author field/compact kernel.
    pub field: PlacementComputeStage,
    /// Fail-closed finalization kernel.
    pub finalize: PlacementComputeStage,
    /// Optional full-key deletion stage. When present, the graph copies the
    /// finalized placement indirect arguments, runs deletion, and routes draw
    /// through the deletion compact/indirect outputs.
    pub deletion: Option<PlacementDeletionWork>,
    /// Production raster consumer.
    pub draw: PlacementDrawStage,
}

/// Resources and stage for filtering a placement stream by canonical full key.
///
/// The bind group must match the full-key deletion shader ABI: request params,
/// placement records, upstream compact indices, upstream indirect args, sorted
/// deletion keys, output compact indices, and output indirect args. The graph
/// declares these accesses explicitly and preserves the upstream placement
/// outputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementDeletionWork {
    /// Metadata from the same uploaded uniform bound by `stage`. Native
    /// producers expose this in their registration snapshot. Like the opaque
    /// bind group, raw RHI callers must not substitute unrelated contents.
    pub request: PlacementDeletionRequest,
    /// Deletion request parameters.
    pub params: PlacementImportedBuffer,
    /// Sorted canonical full-key deletion set.
    pub keys: PlacementImportedBuffer,
    /// Filtered compact source-slot indices.
    pub compact: PlacementImportedBuffer,
    /// Filtered four-word indirect draw arguments.
    pub indirect: PlacementImportedBuffer,
    /// Stage that clears word 1 (instance count) of `indirect` while preserving
    /// the other words. It runs after the argument copy and before filtering.
    pub clear_instance_count: PlacementComputeStage,
    /// Registered deletion compute pipeline and bind group.
    pub stage: PlacementComputeStage,
}

/// Reset-only deletion output for a placement stream with no candidates.
///
/// The graph copies the upstream reset's four-word indirect record, then clears
/// the output instance count. No filter or draw is recorded. The clear bind
/// group must bind `indirect`; opaque RHI bindings cannot be inspected here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementEmptyDeletionWork {
    /// Four-word deletion indirect output, distinct from every other binding.
    pub indirect: PlacementImportedBuffer,
    /// Singleton stage clearing word 1 while preserving the other words.
    pub clear_instance_count: PlacementComputeStage,
}

/// Shader-visible dimensions of one uploaded full-key deletion request.
/// These correspond to uniform words 0..4, not to allocated capacities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementDeletionRequest {
    /// Logical candidate slots, equal to the upstream stream count.
    pub candidate_count: u32,
    /// Canonical uploaded deletion keys, excluding unused allocated slots.
    pub key_count: u32,
    /// Uniform X stride and Y extent used to flatten the 64-lane dispatch.
    pub workgroups: [u32; 2],
}

/// Complete physical resource/pipeline binding for one placement stream.
///
/// The fields intentionally include every buffer visible to the compute and
/// draw bind groups. Declaring all of them keeps the graph's barriers honest;
/// a bind group may not smuggle an undeclared resource into a pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementRenderStream {
    /// Stable world identity and revision used for labels and diagnostics.
    pub stream: PlacementStream,
    /// Source-kernel uniform parameters.
    pub source_params: PlacementImportedBuffer,
    /// Projection-kernel uniform parameters.
    pub project_params: PlacementImportedBuffer,
    /// Author-field uniform parameters.
    pub field_params: PlacementImportedBuffer,
    /// Source/projected placement records.
    pub records: PlacementImportedBuffer,
    /// Exact terrain samples written by projection.
    pub samples: PlacementImportedBuffer,
    /// Final author-field attributes.
    pub attributes: PlacementImportedBuffer,
    /// Compacted source-slot indices.
    pub compact: PlacementImportedBuffer,
    /// Four-word indirect draw arguments.
    pub indirect: PlacementImportedBuffer,
    /// Projection overflow status.
    pub status: PlacementImportedBuffer,
    /// Heightfield samples read by projection.
    pub heights: PlacementImportedBuffer,
    /// Author brush/node records read by the field kernel.
    pub brushes: PlacementImportedBuffer,
    /// Reset kernel clearing transient output state.
    pub reset: PlacementComputeStage,
    /// Candidate-dependent compute and draw work. This must be absent exactly
    /// when `stream.candidate_count == 0`.
    pub work: Option<PlacementRenderWork>,
    /// Optional reset-only deletion output. May only be supplied when
    /// `stream.candidate_count == 0` and `work` is absent.
    pub empty_deletion: Option<PlacementEmptyDeletionWork>,
}

/// Imported scene color target consumed by the placement raster passes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementSceneTarget {
    /// Symbolic target description; it must match the imported native texture.
    pub descriptor: TextureDesc,
    /// Native texture/view and entry/exit states.
    pub imported: ImportedTexture,
}

/// Errors raised before the graph is returned to the renderer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlacementRenderBridgeError {
    /// The same cell was supplied more than once in one frame.
    DuplicateSector { stream: PlacementStream },
    /// A physical buffer cannot be imported with zero bytes.
    ZeroBuffer {
        /// Stream owning the invalid binding.
        stream: PlacementStream,
        /// Binding label.
        resource: &'static str,
    },
    /// A shader-visible resource is smaller than its fixed ABI/budget minimum.
    InvalidBufferSize {
        /// Stream owning the invalid binding.
        stream: PlacementStream,
        /// Binding label.
        resource: &'static str,
        /// Actual physical byte size.
        bytes: u64,
        /// Minimum byte size required by the shader ABI.
        minimum: u64,
    },
    /// A dispatch dimension of zero would silently omit work.
    ZeroDispatch {
        /// Stream owning the invalid stage.
        stream: PlacementStream,
        /// Stage label.
        stage: &'static str,
        /// Invalid dimensions.
        dispatch: [u32; 3],
    },
    /// Indirect argument offsets must be four-byte aligned.
    MisalignedIndirectOffset {
        /// Stream owning the invalid draw.
        stream: PlacementStream,
        /// Invalid byte offset.
        offset: u64,
    },
    /// Placement kernels initialize only the first indirect record.
    UnsupportedIndirectOffset {
        stream: PlacementStream,
        offset: u64,
    },
    /// Shader work dimensions disagree with the uploaded request or its bounds,
    /// or reset-only deletion work was supplied for a nonempty/workful stream.
    InvalidDeletionRequest { stream: PlacementStream },
    /// The singleton reset kernel must not race itself in multiple workgroups.
    InvalidResetDispatch {
        stream: PlacementStream,
        stage: &'static str,
        dispatch: [u32; 3],
    },
    /// A physical allocation is used under multiple names and at least one
    /// usage writes it. Includes aliases between different streams.
    AliasedBuffer {
        stream: PlacementStream,
        resource: &'static str,
        other_stream: Box<PlacementStream>,
        other_resource: &'static str,
    },
    /// Shared read-only imports must agree on physical extent and entry state.
    ConflictingBufferImport {
        stream: PlacementStream,
        resource: &'static str,
    },
    /// Candidate count and candidate-dependent work disagree.
    CandidateWorkMismatch {
        /// Stream owning the inconsistent binding.
        stream: PlacementStream,
        /// Whether candidate-dependent work was supplied.
        work_present: bool,
    },
    /// The underlying executable graph rejected a checked operation.
    Graph(CompileError),
}

impl fmt::Display for PlacementRenderBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSector { stream } => {
                write!(f, "duplicate placement stream for {stream:?}")
            }
            Self::ZeroBuffer { stream, resource } => {
                write!(f, "placement {resource} buffer is empty for {stream:?}")
            }
            Self::InvalidBufferSize {
                stream,
                resource,
                bytes,
                minimum,
            } => write!(
                f,
                "placement {resource} buffer has {bytes} bytes; needs at least {minimum} for {stream:?}"
            ),
            Self::ZeroDispatch {
                stream,
                stage,
                dispatch,
            } => write!(
                f,
                "placement {stage} dispatch {dispatch:?} is zero for {stream:?}"
            ),
            Self::MisalignedIndirectOffset { stream, offset } => write!(
                f,
                "placement indirect offset {offset} is not four-byte aligned for {stream:?}"
            ),
            Self::UnsupportedIndirectOffset { stream, offset } => write!(f,
                "placement indirect offset {offset} must be zero for the fixed ABI in {stream:?}"),
            Self::InvalidDeletionRequest { stream } => write!(f,
                "placement deletion request dimensions, dispatch, or empty-work metadata are inconsistent for {stream:?}"),
            Self::InvalidResetDispatch { stream, stage, dispatch } => write!(f,
                "placement {stage} dispatch {dispatch:?} must be [1, 1, 1] for {stream:?}"),
            Self::AliasedBuffer { stream, resource, other_stream, other_resource } => write!(f,
                "placement {resource} in {stream:?} aliases writable usage {other_resource} in {other_stream:?}"),
            Self::ConflictingBufferImport { stream, resource } => write!(f,
                "placement shared buffer {resource} has inconsistent import metadata in {stream:?}"),
            Self::CandidateWorkMismatch {
                stream,
                work_present,
            } => write!(
                f,
                "placement candidate count {} disagrees with work_present={work_present} for {stream:?}",
                stream.candidate_count
            ),
            Self::Graph(error) => write!(f, "placement render graph lowering failed: {error}"),
        }
    }
}

impl std::error::Error for PlacementRenderBridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CompileError> for PlacementRenderBridgeError {
    fn from(error: CompileError) -> Self {
        Self::Graph(error)
    }
}

/// Builds an executable `RenderGraph` for all supplied placement streams.
///
/// The returned graph is still record-only. The renderer must call
/// `RenderGraph::record`/`execute` with the same RHI device table that owns all
/// imported IDs and must retain native resources until submission completion.
pub fn build_placement_render_graph(
    extent: Extent3d,
    scene: PlacementSceneTarget,
    streams: &[PlacementRenderStream],
) -> Result<RenderGraph, PlacementRenderBridgeError> {
    let mut ordered = streams.to_vec();
    ordered.sort_unstable_by_key(|item| (item.stream.cell, item.stream.revision));
    for pair in ordered.windows(2) {
        if pair[0].stream.cell == pair[1].stream.cell {
            return Err(PlacementRenderBridgeError::DuplicateSector {
                stream: pair[1].stream,
            });
        }
    }

    for item in &ordered {
        validate_stream(item)?;
    }
    validate_physical_buffers(&ordered)?;
    let mut graph = RenderGraph::new(extent);
    let mut scene_handle =
        graph.try_import_texture("world placement scene", scene.descriptor, scene.imported)?;
    let mut imported_buffers = BTreeMap::new();

    for item in ordered {
        let handles = import_stream(&mut graph, &mut imported_buffers, &item)?;

        let (indirect, status) = add_reset_pass(
            &mut graph,
            handles.indirect,
            handles.status,
            item.reset,
            item.stream,
        )?;

        let Some(work) = item.work else {
            if let Some(deletion) = item.empty_deletion {
                let deletion_indirect = import_buffer(
                    &mut graph,
                    &mut imported_buffers,
                    "world placement deletion indirect",
                    deletion.indirect,
                )?;
                // Reuse the upstream reset's SSA output, not a second import
                // of the same physical buffer or its stale entry-state handle.
                let copied_indirect =
                    add_deletion_args_copy(&mut graph, indirect, deletion_indirect, item.stream)?;
                add_deletion_count_reset(
                    &mut graph,
                    copied_indirect,
                    deletion.clear_instance_count,
                    item.stream,
                )?;
            }
            continue;
        };

        let attributes = handles.attributes;
        let compact = handles.compact;
        let source_params = handles.source_params;
        let source_records = handles.records;
        let source_stage = work.source;
        let source_records =
            graph_handle_write(&mut graph, source_params, source_records, source_stage)?;

        let project_params = handles.project_params;
        let heights = handles.heights;
        let samples = handles.samples;
        let project_stage = work.project;
        let (records, samples, status, indirect, attributes) = add_project_pass(
            &mut graph,
            project_params,
            source_records,
            heights,
            samples,
            status,
            indirect,
            attributes,
            project_stage,
            item.stream,
        )?;

        let field_params = handles.field_params;
        let brushes = handles.brushes;
        let field_stage = work.field;
        let (attributes, compact, indirect) = add_field_pass(
            &mut graph,
            field_params,
            records,
            brushes,
            attributes,
            compact,
            indirect,
            field_stage,
            item.stream,
        )?;

        let (records, samples, _status, attributes, indirect) = add_finalize_pass(
            &mut graph,
            project_params,
            records,
            samples,
            status,
            attributes,
            indirect,
            work.finalize,
            item.stream,
        )?;

        let (draw_compact, draw_indirect) = if let Some(deletion) = work.deletion {
            let deletion_handles = import_deletion(&mut graph, &mut imported_buffers, deletion)?;
            let copied_indirect = add_deletion_args_copy(
                &mut graph,
                indirect,
                deletion_handles.indirect,
                item.stream,
            )?;
            let reset_indirect = add_deletion_count_reset(
                &mut graph,
                copied_indirect,
                deletion.clear_instance_count,
                item.stream,
            )?;
            let (filtered_compact, filtered_indirect) = add_deletion_pass(
                &mut graph,
                [records, compact, indirect],
                reset_indirect,
                deletion_handles,
                deletion.stage,
                item.stream,
            )?;
            (filtered_compact, filtered_indirect)
        } else {
            (compact, indirect)
        };

        let draw_records = records;
        let draw_samples = samples;
        let draw_attributes = attributes;
        let draw_stage = work.draw;
        let previous_scene = scene_handle;
        let mut next_scene = previous_scene;
        graph.try_add_raster_pass(
            format!("world placement draw {:?}", item.stream),
            PassFlags::NEVER_CULL,
            |builder| {
                let _ = builder.read_storage_buffer(draw_records);
                let _ = builder.read_storage_buffer(draw_samples);
                let _ = builder.read_storage_buffer(draw_attributes);
                let _ = builder.read_storage_buffer(draw_compact);
                let _ = builder.read_indirect(draw_indirect);
                next_scene = builder.color_attachment(previous_scene);
                move |commands: &mut ExecuteContext<'_>| {
                    commands
                        .set_render_pipeline(draw_stage.pipeline)
                        .set_render_bind_group(0, draw_stage.bind_group, Vec::new())
                        .draw_indirect(draw_indirect, draw_stage.indirect_offset);
                }
            },
        )?;
        scene_handle = next_scene;
    }

    Ok(graph)
}

#[derive(Clone, Copy)]
struct StreamHandles {
    source_params: BufferHandle,
    project_params: BufferHandle,
    field_params: BufferHandle,
    records: BufferHandle,
    samples: BufferHandle,
    attributes: BufferHandle,
    compact: BufferHandle,
    indirect: BufferHandle,
    status: BufferHandle,
    heights: BufferHandle,
    brushes: BufferHandle,
}

#[derive(Clone, Copy)]
struct DeletionHandles {
    params: BufferHandle,
    keys: BufferHandle,
    compact: BufferHandle,
    indirect: BufferHandle,
}

fn import_deletion(
    graph: &mut RenderGraph,
    imported: &mut BTreeMap<BufferId, BufferHandle>,
    work: PlacementDeletionWork,
) -> Result<DeletionHandles, CompileError> {
    let mut import =
        |name, value, graph: &mut RenderGraph| import_buffer(graph, imported, name, value);
    Ok(DeletionHandles {
        params: import("world placement deletion params", work.params, graph)?,
        keys: import("world placement deletion keys", work.keys, graph)?,
        compact: import("world placement deletion compact", work.compact, graph)?,
        indirect: import("world placement deletion indirect", work.indirect, graph)?,
    })
}

fn add_deletion_args_copy(
    graph: &mut RenderGraph,
    source: BufferHandle,
    destination: BufferHandle,
    stream: PlacementStream,
) -> Result<BufferHandle, CompileError> {
    let mut output = destination;
    graph.try_add_transfer_pass(
        format!("world placement deletion args {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            let source = builder.copy_src_buffer(source);
            output = builder.copy_dst_buffer(destination);
            move |commands: &mut ExecuteContext<'_>| {
                commands.copy_buffer_to_buffer(source, 0, output, 0, 16);
            }
        },
    )?;
    Ok(output)
}

fn add_deletion_pass(
    graph: &mut RenderGraph,
    [records, upstream_compact, upstream_indirect]: [BufferHandle; 3],
    copied_indirect: BufferHandle,
    deletion: DeletionHandles,
    stage: PlacementComputeStage,
    stream: PlacementStream,
) -> Result<(BufferHandle, BufferHandle), CompileError> {
    let mut output = (deletion.compact, copied_indirect);
    graph.try_add_compute_pass(
        format!("world placement deletion {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            let _ = builder.read_uniform(deletion.params);
            let _ = builder.read_storage_buffer(records);
            let _ = builder.read_storage_buffer(upstream_compact);
            let _ = builder.read_storage_buffer(upstream_indirect);
            let _ = builder.read_storage_buffer(deletion.keys);
            output.0 = builder.write_storage_buffer(deletion.compact);
            output.1 = builder.write_storage_buffer(copied_indirect);
            move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
        },
    )?;
    Ok(output)
}

fn add_deletion_count_reset(
    graph: &mut RenderGraph,
    indirect: BufferHandle,
    stage: PlacementComputeStage,
    stream: PlacementStream,
) -> Result<BufferHandle, CompileError> {
    let mut output = indirect;
    graph.try_add_compute_pass(
        format!("world placement deletion count reset {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            output = builder.write_storage_buffer(indirect);
            move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
        },
    )?;
    Ok(output)
}

fn import_stream(
    graph: &mut RenderGraph,
    imported: &mut BTreeMap<BufferId, BufferHandle>,
    item: &PlacementRenderStream,
) -> Result<StreamHandles, CompileError> {
    let mut import =
        |name, value, graph: &mut RenderGraph| import_buffer(graph, imported, name, value);
    Ok(StreamHandles {
        source_params: import("world placement source params", item.source_params, graph)?,
        project_params: import("world placement project params", item.project_params, graph)?,
        field_params: import("world placement field params", item.field_params, graph)?,
        records: import("world placement records", item.records, graph)?,
        samples: import("world placement samples", item.samples, graph)?,
        attributes: import("world placement attributes", item.attributes, graph)?,
        compact: import("world placement compact", item.compact, graph)?,
        indirect: import("world placement indirect", item.indirect, graph)?,
        status: import("world placement status", item.status, graph)?,
        heights: import("world placement heights", item.heights, graph)?,
        brushes: import("world placement brushes", item.brushes, graph)?,
    })
}

fn add_reset_pass(
    graph: &mut RenderGraph,
    indirect: BufferHandle,
    status: BufferHandle,
    stage: PlacementComputeStage,
    stream: PlacementStream,
) -> Result<(BufferHandle, BufferHandle), CompileError> {
    let mut out = (indirect, status);
    graph.try_add_compute_pass(
        format!("world placement reset {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            out.0 = builder.write_storage_buffer(indirect);
            out.1 = builder.write_storage_buffer(status);
            move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
        },
    )?;
    Ok(out)
}

fn graph_handle_write(
    graph: &mut RenderGraph,
    params: BufferHandle,
    records: BufferHandle,
    stage: PlacementComputeStage,
) -> Result<BufferHandle, CompileError> {
    let mut output = records;
    graph.try_add_compute_pass("world placement source", PassFlags::NEVER_CULL, |builder| {
        let _ = builder.read_uniform(params);
        output = builder.write_storage_buffer(records);
        move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
    })?;
    Ok(output)
}

fn add_project_pass(
    graph: &mut RenderGraph,
    params: BufferHandle,
    records: BufferHandle,
    heights: BufferHandle,
    samples: BufferHandle,
    status: BufferHandle,
    indirect: BufferHandle,
    attributes: BufferHandle,
    stage: PlacementComputeStage,
    stream: PlacementStream,
) -> Result<
    (
        BufferHandle,
        BufferHandle,
        BufferHandle,
        BufferHandle,
        BufferHandle,
    ),
    CompileError,
> {
    let mut out = (records, samples, status, indirect, attributes);
    graph.try_add_compute_pass(
        format!("world placement project {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            let _ = builder.read_uniform(params);
            out.0 = builder.write_storage_buffer(records);
            let _ = builder.read_storage_buffer(heights);
            out.1 = builder.write_storage_buffer(samples);
            out.2 = builder.write_storage_buffer(status);
            out.3 = builder.write_storage_buffer(indirect);
            out.4 = builder.write_storage_buffer(attributes);
            move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
        },
    )?;
    Ok(out)
}

fn add_field_pass(
    graph: &mut RenderGraph,
    params: BufferHandle,
    records: BufferHandle,
    brushes: BufferHandle,
    attributes: BufferHandle,
    compact: BufferHandle,
    indirect: BufferHandle,
    stage: PlacementComputeStage,
    stream: PlacementStream,
) -> Result<(BufferHandle, BufferHandle, BufferHandle), CompileError> {
    let mut out = (attributes, compact, indirect);
    graph.try_add_compute_pass(
        format!("world placement field {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            let _ = builder.read_uniform(params);
            let _ = builder.read_storage_buffer(records);
            let _ = builder.read_storage_buffer(brushes);
            out.0 = builder.write_storage_buffer(attributes);
            out.1 = builder.write_storage_buffer(compact);
            out.2 = builder.write_storage_buffer(indirect);
            move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
        },
    )?;
    Ok(out)
}

fn add_finalize_pass(
    graph: &mut RenderGraph,
    params: BufferHandle,
    records: BufferHandle,
    samples: BufferHandle,
    status: BufferHandle,
    attributes: BufferHandle,
    indirect: BufferHandle,
    stage: PlacementComputeStage,
    stream: PlacementStream,
) -> Result<
    (
        BufferHandle,
        BufferHandle,
        BufferHandle,
        BufferHandle,
        BufferHandle,
    ),
    CompileError,
> {
    let mut out = (records, samples, status, attributes, indirect);
    graph.try_add_compute_pass(
        format!("world placement finalize {:?}", stream),
        PassFlags::NEVER_CULL,
        |builder| {
            let _ = builder.read_uniform(params);
            out.0 = builder.write_storage_buffer(records);
            out.1 = builder.write_storage_buffer(samples);
            out.2 = builder.write_storage_buffer(status);
            out.3 = builder.write_storage_buffer(attributes);
            out.4 = builder.write_storage_buffer(indirect);
            move |commands: &mut ExecuteContext<'_>| record_compute(commands, stage)
        },
    )?;
    Ok(out)
}

fn record_compute(commands: &mut ExecuteContext<'_>, stage: PlacementComputeStage) {
    commands
        .set_compute_pipeline(stage.pipeline)
        .set_compute_bind_group(0, stage.bind_group, Vec::new())
        .dispatch(stage.dispatch[0], stage.dispatch[1], stage.dispatch[2]);
}

fn import_buffer(
    graph: &mut RenderGraph,
    imported: &mut BTreeMap<BufferId, BufferHandle>,
    name: &'static str,
    value: PlacementImportedBuffer,
) -> Result<BufferHandle, CompileError> {
    if let Some(handle) = imported.get(&value.imported.buffer) {
        return Ok(*handle);
    }
    let handle = graph.try_import_buffer(
        name,
        BufferDesc::new(value.bytes, BufferUsages::NONE),
        value.imported,
    )?;
    imported.insert(value.imported.buffer, handle);
    Ok(handle)
}

/// Reject hidden physical hazards before creating any logical graph resources.
/// A shared immutable resource gets one logical identity; writable resources
/// must have unique physical identities across the entire frame. O(S log S)
/// for S streams (each has a fixed maximum of fifteen bindings).
fn validate_physical_buffers(
    streams: &[PlacementRenderStream],
) -> Result<(), PlacementRenderBridgeError> {
    let mut seen = BTreeMap::new();
    for item in streams {
        let mut check = |resource, binding: PlacementImportedBuffer, writable| {
            if let Some(&(other_stream, other_resource, other_binding, other_writable)) =
                seen.get(&binding.imported.buffer)
            {
                if writable || other_writable {
                    return Err(PlacementRenderBridgeError::AliasedBuffer {
                        stream: item.stream,
                        resource,
                        other_stream: Box::new(other_stream),
                        other_resource,
                    });
                }
                if binding != other_binding {
                    return Err(PlacementRenderBridgeError::ConflictingBufferImport {
                        stream: item.stream,
                        resource,
                    });
                }
            } else {
                seen.insert(
                    binding.imported.buffer,
                    (item.stream, resource, binding, writable),
                );
            }
            Ok(())
        };
        for (name, buffer, writable) in [
            ("source_params", item.source_params, false),
            ("project_params", item.project_params, false),
            ("field_params", item.field_params, false),
            ("records", item.records, true),
            ("samples", item.samples, true),
            ("attributes", item.attributes, true),
            ("compact", item.compact, true),
            ("indirect", item.indirect, true),
            ("status", item.status, true),
            ("heights", item.heights, false),
            ("brushes", item.brushes, false),
        ] {
            check(name, buffer, writable)?;
        }
        if let Some(deletion) = item.work.and_then(|work| work.deletion) {
            for (name, buffer, writable) in [
                ("deletion_params", deletion.params, false),
                ("deletion_keys", deletion.keys, false),
                ("deletion_compact", deletion.compact, true),
                ("deletion_indirect", deletion.indirect, true),
            ] {
                check(name, buffer, writable)?;
            }
        }
        if let Some(deletion) = item.empty_deletion {
            check("deletion_indirect", deletion.indirect, true)?;
        }
    }
    Ok(())
}

fn validate_size(
    item: &PlacementRenderStream,
    resource: &'static str,
    binding: PlacementImportedBuffer,
    minimum: u64,
) -> Result<(), PlacementRenderBridgeError> {
    if binding.bytes < minimum {
        return Err(PlacementRenderBridgeError::InvalidBufferSize {
            stream: item.stream,
            resource,
            bytes: binding.bytes,
            minimum,
        });
    }
    Ok(())
}

fn validate_reset(
    item: &PlacementRenderStream,
    stage: &'static str,
    dispatch: [u32; 3],
) -> Result<(), PlacementRenderBridgeError> {
    if dispatch.contains(&0) {
        return Err(PlacementRenderBridgeError::ZeroDispatch {
            stream: item.stream,
            stage,
            dispatch,
        });
    }
    if dispatch != [1, 1, 1] {
        return Err(PlacementRenderBridgeError::InvalidResetDispatch {
            stream: item.stream,
            stage,
            dispatch,
        });
    }
    Ok(())
}

fn validate_stream(item: &PlacementRenderStream) -> Result<(), PlacementRenderBridgeError> {
    let buffers = [
        ("source_params", item.source_params),
        ("project_params", item.project_params),
        ("field_params", item.field_params),
        ("records", item.records),
        ("samples", item.samples),
        ("attributes", item.attributes),
        ("compact", item.compact),
        ("indirect", item.indirect),
        ("status", item.status),
        ("heights", item.heights),
        ("brushes", item.brushes),
    ];
    for (resource, binding) in buffers {
        if binding.bytes == 0 {
            return Err(PlacementRenderBridgeError::ZeroBuffer {
                stream: item.stream,
                resource,
            });
        }
    }
    if (item.stream.candidate_count == 0) != item.work.is_none() {
        return Err(PlacementRenderBridgeError::CandidateWorkMismatch {
            stream: item.stream,
            work_present: item.work.is_some(),
        });
    }
    if let Some(deletion) = item.empty_deletion {
        if item.stream.candidate_count != 0 || item.work.is_some() {
            return Err(PlacementRenderBridgeError::InvalidDeletionRequest {
                stream: item.stream,
            });
        }
        if deletion.indirect.bytes == 0 {
            return Err(PlacementRenderBridgeError::ZeroBuffer {
                stream: item.stream,
                resource: "deletion_indirect",
            });
        }
        validate_size(item, "deletion_indirect", deletion.indirect, 16)?;
        validate_reset(
            item,
            "deletion_count_reset",
            deletion.clear_instance_count.dispatch,
        )?;
    }
    let slots = u64::from(item.stream.candidate_count.max(1));
    for (resource, binding, minimum) in [
        ("source_params", item.source_params, 192),
        ("project_params", item.project_params, 128),
        ("field_params", item.field_params, 128),
        ("records", item.records, slots * 64),
        ("samples", item.samples, slots * 64),
        ("attributes", item.attributes, slots * 16),
        ("compact", item.compact, slots * 4),
        ("indirect", item.indirect, 16),
        ("status", item.status, 4),
        ("heights", item.heights, 4),
        ("brushes", item.brushes, 64),
    ] {
        validate_size(item, resource, binding, minimum)?;
    }
    validate_reset(item, "reset", item.reset.dispatch)?;
    let Some(work) = item.work else {
        return Ok(());
    };
    for (stage, label) in [
        (work.source, "source"),
        (work.project, "project"),
        (work.field, "field"),
        (work.finalize, "finalize"),
    ] {
        if stage.dispatch.contains(&0) {
            return Err(PlacementRenderBridgeError::ZeroDispatch {
                stream: item.stream,
                stage: label,
                dispatch: stage.dispatch,
            });
        }
    }
    if let Some(deletion) = work.deletion {
        for (resource, binding) in [
            ("deletion_params", deletion.params),
            ("deletion_keys", deletion.keys),
            ("deletion_compact", deletion.compact),
            ("deletion_indirect", deletion.indirect),
        ] {
            if binding.bytes == 0 {
                return Err(PlacementRenderBridgeError::ZeroBuffer {
                    stream: item.stream,
                    resource,
                });
            }
        }
        let candidate_bytes = u64::from(item.stream.candidate_count) * 4;
        for (resource, binding, minimum) in [
            ("deletion_params", deletion.params, 128),
            (
                "deletion_keys",
                deletion.keys,
                56 * u64::from(deletion.request.key_count.max(1)),
            ),
            ("deletion_compact", deletion.compact, candidate_bytes),
            ("deletion_indirect", deletion.indirect, 16),
        ] {
            validate_size(item, resource, binding, minimum)?;
        }
        validate_reset(
            item,
            "deletion_count_reset",
            deletion.clear_instance_count.dispatch,
        )?;
        if deletion.stage.dispatch.contains(&0) {
            return Err(PlacementRenderBridgeError::ZeroDispatch {
                stream: item.stream,
                stage: "deletion",
                dispatch: deletion.stage.dispatch,
            });
        }
        let [x, y, z] = deletion.stage.dispatch;
        let invocations = u64::from(x)
            .checked_mul(u64::from(y))
            .and_then(|n| n.checked_mul(64));
        if deletion.request.candidate_count != item.stream.candidate_count
            || deletion.request.workgroups != [x, y]
            || z != 1
            || invocations.is_none_or(|n| {
                n < u64::from(item.stream.candidate_count) || n > u64::from(u32::MAX) + 1
            })
        {
            return Err(PlacementRenderBridgeError::InvalidDeletionRequest {
                stream: item.stream,
            });
        }
    }
    if work.draw.indirect_offset % 4 != 0 {
        return Err(PlacementRenderBridgeError::MisalignedIndirectOffset {
            stream: item.stream,
            offset: work.draw.indirect_offset,
        });
    }
    if work.draw.indirect_offset != 0 {
        return Err(PlacementRenderBridgeError::UnsupportedIndirectOffset {
            stream: item.stream,
            offset: work.draw.indirect_offset,
        });
    }
    let draw_indirect = work
        .deletion
        .map_or(item.indirect, |deletion| deletion.indirect);
    let Some(indirect_end) = work.draw.indirect_offset.checked_add(16) else {
        return Err(PlacementRenderBridgeError::InvalidBufferSize {
            stream: item.stream,
            resource: "draw_indirect",
            bytes: draw_indirect.bytes,
            minimum: u64::MAX,
        });
    };
    if draw_indirect.bytes < indirect_end {
        return Err(PlacementRenderBridgeError::InvalidBufferSize {
            stream: item.stream,
            resource: "draw_indirect",
            bytes: draw_indirect.bytes,
            minimum: indirect_end,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{
        BufferDescriptor, BufferState, ComputeCommand, CopyCommand, DeviceDescriptor, Instance,
        Pass, RenderCommand, RenderDevice, RequestAdapterOptions, TextureDescriptor,
        TextureDimension, TextureFormat, TextureState, TextureUsages,
    };
    use prism_render_driver_null::{NullInstance, NullQueue};

    fn stream() -> PlacementStream {
        PlacementStream {
            cell: prism_world_stream::SectorId::new(1, 0, 0, 0),
            revision: 3,
            candidate_count: 32,
        }
    }

    fn imported_buffer(device: &impl RenderDevice, bytes: u64) -> PlacementImportedBuffer {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes,
            usage: BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        PlacementImportedBuffer {
            imported: ImportedBuffer {
                buffer,
                entry_state: BufferState::initial(),
            },
            bytes,
        }
    }

    fn item(device: &impl RenderDevice) -> PlacementRenderStream {
        let stage = |id, dispatch| PlacementComputeStage {
            pipeline: ComputePipelineId::from_parts(id, 0),
            bind_group: BindGroupId::from_parts(id, 0),
            dispatch,
        };
        PlacementRenderStream {
            stream: stream(),
            source_params: imported_buffer(device, 192),
            project_params: imported_buffer(device, 128),
            field_params: imported_buffer(device, 128),
            records: imported_buffer(device, 64 * 32),
            samples: imported_buffer(device, 64 * 32),
            attributes: imported_buffer(device, 16 * 32),
            compact: imported_buffer(device, 4 * 32),
            indirect: imported_buffer(device, 16),
            status: imported_buffer(device, 4),
            heights: imported_buffer(device, 16),
            brushes: imported_buffer(device, 64),
            reset: stage(1, [1, 1, 1]),
            work: Some(PlacementRenderWork {
                source: stage(2, [2, 1, 1]),
                project: stage(3, [3, 1, 1]),
                field: stage(4, [4, 1, 1]),
                finalize: stage(5, [5, 1, 1]),
                deletion: None,
                draw: PlacementDrawStage {
                    pipeline: RenderPipelineId::from_parts(6, 0),
                    bind_group: BindGroupId::from_parts(6, 0),
                    indirect_offset: 0,
                },
            }),
            empty_deletion: None,
        }
    }

    fn reset_only_item(device: &impl RenderDevice) -> PlacementRenderStream {
        let mut item = item(device);
        item.stream.candidate_count = 0;
        item.work = None;
        item
    }

    fn with_empty_deletion(device: &impl RenderDevice) -> PlacementRenderStream {
        let mut item = reset_only_item(device);
        item.empty_deletion = Some(PlacementEmptyDeletionWork {
            indirect: imported_buffer(device, 16),
            clear_instance_count: PlacementComputeStage {
                pipeline: ComputePipelineId::from_parts(7, 0),
                bind_group: BindGroupId::from_parts(7, 0),
                dispatch: [1, 1, 1],
            },
        });
        item
    }

    fn with_deletion(device: &impl RenderDevice) -> PlacementRenderStream {
        let mut item = item(device);
        item.records = imported_buffer(device, u64::from(item.stream.candidate_count) * 64);
        item.compact = imported_buffer(device, u64::from(item.stream.candidate_count) * 4);
        let stage = |id, dispatch| PlacementComputeStage {
            pipeline: ComputePipelineId::from_parts(id, 0),
            bind_group: BindGroupId::from_parts(id, 0),
            dispatch,
        };
        item.work.as_mut().unwrap().draw.indirect_offset = 0;
        item.work.as_mut().unwrap().deletion = Some(PlacementDeletionWork {
            request: PlacementDeletionRequest {
                candidate_count: item.stream.candidate_count,
                key_count: 1,
                workgroups: [1, 1],
            },
            params: imported_buffer(device, 128),
            keys: imported_buffer(device, 56),
            compact: imported_buffer(device, 128),
            indirect: imported_buffer(device, 16),
            clear_instance_count: stage(7, [1, 1, 1]),
            stage: stage(8, [1, 1, 1]),
        });
        item
    }

    fn device() -> impl RenderDevice {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter");
        let (device, _queue): (_, NullQueue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device");
        device
    }

    fn scene(device: &impl RenderDevice) -> PlacementSceneTarget {
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d::new_2d(4, 4),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
        });
        let view = device.create_texture_view(texture, &Default::default());
        PlacementSceneTarget {
            descriptor: TextureDesc::color(TextureFormat::Rgba8Unorm),
            imported: ImportedTexture {
                texture,
                default_view: view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        }
    }

    #[test]
    fn lowers_all_compute_stages_and_indirect_draw_to_one_recorded_graph() {
        let device = device();
        let graph =
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[item(&device)])
                .expect("graph lowering");
        let recorded = graph.record(&device).expect("recording");
        assert_eq!(recorded.plan().pass_count(), 6);
        assert_eq!(recorded.commands().passes.len(), 6);
        for (index, (pass, expected_x)) in recorded.commands().passes[..5]
            .iter()
            .zip(1..=5)
            .enumerate()
        {
            let Pass::Compute { commands, .. } = pass else {
                panic!("pass {index} must be compute");
            };
            let [ComputeCommand::SetPipeline(_), ComputeCommand::SetBindGroup { index: 0, .. }, ComputeCommand::Dispatch { x, y: 1, z: 1 }] =
                commands.as_slice()
            else {
                panic!("pass {index} has unexpected compute commands: {commands:?}");
            };
            assert_eq!(*x, expected_x);
        }
        let Pass::Render { commands, .. } = &recorded.commands().passes[5] else {
            panic!("last pass must be render");
        };
        assert!(matches!(
            commands.as_slice(),
            [
                RenderCommand::SetPipeline(_),
                RenderCommand::SetBindGroup { index: 0, .. },
                RenderCommand::DrawIndirect { offset: 0, .. },
            ]
        ));
    }

    #[test]
    fn empty_stream_records_only_the_mandatory_reset_pass() {
        let device = device();
        let graph = build_placement_render_graph(
            Extent3d::new_2d(4, 4),
            scene(&device),
            &[reset_only_item(&device)],
        )
        .expect("reset-only graph lowering");
        let recorded = graph.record(&device).expect("reset-only recording");
        assert_eq!(recorded.plan().pass_count(), 1);
        let [Pass::Compute {
            label, commands, ..
        }] = recorded.commands().passes.as_slice()
        else {
            panic!("empty stream must record exactly one compute pass");
        };
        assert!(label
            .as_deref()
            .is_some_and(|label| label.contains("reset")));
        assert!(matches!(
            commands.as_slice(),
            [
                ComputeCommand::SetPipeline(_),
                ComputeCommand::SetBindGroup { index: 0, .. },
                ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
            ]
        ));
    }

    #[test]
    fn empty_deletion_copies_upstream_reset_then_clears_output_without_filter_or_draw() {
        let device = device();
        let item = with_empty_deletion(&device);
        let deletion = item.empty_deletion.unwrap();
        let graph = build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[item])
            .expect("empty deletion graph lowering");
        let recorded = graph.record(&device).expect("empty deletion recording");
        let plan = recorded.plan();
        assert_eq!(plan.order, [0, 1, 2]);
        // Eleven upstream imports plus one deletion output. In particular the
        // copy source must not be reimported under a new logical identity.
        assert_eq!(plan.used_buffers.len(), 12);
        assert_eq!(plan.used_buffers.iter().filter(|&&used| used).count(), 3);
        assert!(plan.used_textures.iter().all(|&used| !used));
        let [reset_accesses, copy_accesses, clear_accesses] = plan.accesses.as_slice() else {
            panic!("empty output must have exactly three pass declarations");
        };
        assert_eq!(reset_accesses.len(), 2);
        assert_eq!(copy_accesses.len(), 2);
        assert_eq!(clear_accesses.len(), 1);
        let upstream = reset_accesses[0];
        let copy_source = copy_accesses[0];
        let copy_destination = copy_accesses[1];
        let clear = clear_accesses[0];
        assert_eq!(copy_source.resource, upstream.resource);
        assert_eq!(Some(copy_source.input_version), upstream.output_version);
        assert_eq!(copy_source.output_version, None);
        assert_ne!(copy_destination.resource, upstream.resource);
        assert_eq!(clear.resource, copy_destination.resource);
        assert_eq!(Some(clear.input_version), copy_destination.output_version);
        assert!(clear.output_version.is_some());
        for (source, destination, resource) in [(0, 1, upstream.resource), (1, 2, clear.resource)] {
            assert!(plan.logical_barriers.iter().any(|barrier| {
                barrier.source == source
                    && barrier.destination == destination
                    && barrier.resource == resource
            }));
        }

        let [Pass::Compute {
            commands: reset, ..
        }, Pass::Transfer { commands: copy, .. }, Pass::Compute {
            commands: clear, ..
        }] = recorded.commands().passes.as_slice()
        else {
            panic!("empty deletion must only reset, copy, and clear; never filter or draw");
        };
        for (commands, stage) in [(reset, item.reset), (clear, deletion.clear_instance_count)] {
            assert_eq!(
                commands.as_slice(),
                &[
                    ComputeCommand::SetPipeline(stage.pipeline),
                    ComputeCommand::SetBindGroup {
                        index: 0,
                        bind_group: stage.bind_group,
                        dynamic_offsets: Vec::new(),
                    },
                    ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
                ]
            );
        }
        assert_eq!(
            copy.as_slice(),
            &[CopyCommand::CopyBufferToBuffer {
                src: item.indirect.imported.buffer,
                src_offset: 0,
                dst: deletion.indirect.imported.buffer,
                dst_offset: 0,
                size: 16,
            }]
        );
        recorded.release(&device);
    }

    #[test]
    fn rejects_empty_deletion_on_nonempty_or_workful_streams() {
        let device = device();
        for (candidate_count, work_present) in [(0, true), (32, false), (32, true)] {
            let mut invalid = with_empty_deletion(&device);
            invalid.stream.candidate_count = candidate_count;
            if work_present {
                invalid.work = item(&device).work;
            }
            assert!(matches!(
                build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[invalid]),
                Err(PlacementRenderBridgeError::CandidateWorkMismatch { .. })
                    | Err(PlacementRenderBridgeError::InvalidDeletionRequest { .. })
            ));
        }
    }

    #[test]
    fn rejects_empty_deletion_with_short_indirect_or_invalid_clear_dispatch() {
        let device = device();
        let base = with_empty_deletion(&device);
        for bytes in 0..16 {
            let mut item = base;
            item.empty_deletion.as_mut().unwrap().indirect.bytes = bytes;
            let result =
                build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[item]);
            if bytes == 0 {
                assert!(matches!(
                    result,
                    Err(PlacementRenderBridgeError::ZeroBuffer {
                        resource: "deletion_indirect",
                        ..
                    })
                ));
            } else {
                assert!(matches!(
                    result,
                    Err(PlacementRenderBridgeError::InvalidBufferSize {
                        resource: "deletion_indirect",
                        minimum: 16,
                        ..
                    })
                ));
            }
        }
        for dispatch in [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
        ] {
            let mut item = base;
            item.empty_deletion
                .as_mut()
                .unwrap()
                .clear_instance_count
                .dispatch = dispatch;
            let result =
                build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[item]);
            if dispatch.contains(&0) {
                assert!(matches!(
                    result,
                    Err(PlacementRenderBridgeError::ZeroDispatch {
                        stage: "deletion_count_reset",
                        ..
                    })
                ));
            } else {
                assert!(matches!(
                    result,
                    Err(PlacementRenderBridgeError::InvalidResetDispatch {
                        stage: "deletion_count_reset",
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn rejects_empty_deletion_output_aliases_within_and_between_streams() {
        let device = device();
        let base = with_empty_deletion(&device);
        let mut distinct = with_deletion(&device);
        distinct.stream.cell = prism_world_stream::SectorId::new(1, 1, 0, 0);
        for other in BUFFER_NAMES {
            for across_streams in [false, true] {
                if !across_streams && other.starts_with("deletion_") {
                    continue;
                }
                let mut first = base;
                let mut second = distinct;
                let imported = if across_streams {
                    binding_mut(&mut second, other).imported
                } else {
                    binding_mut(&mut first, other).imported
                };
                first.empty_deletion.as_mut().unwrap().indirect.imported = imported;
                let streams = if across_streams {
                    vec![first, second]
                } else {
                    vec![first]
                };
                assert!(
                    matches!(
                        build_placement_render_graph(
                            Extent3d::new_2d(4, 4),
                            scene(&device),
                            &streams,
                        ),
                        Err(PlacementRenderBridgeError::AliasedBuffer { .. })
                    ),
                    "empty deletion aliases {other}, across_streams={across_streams}"
                );
            }
        }

        let mut second_empty = with_empty_deletion(&device);
        second_empty.stream.cell = distinct.stream.cell;
        second_empty.empty_deletion.as_mut().unwrap().indirect =
            base.empty_deletion.unwrap().indirect;
        assert!(matches!(
            build_placement_render_graph(
                Extent3d::new_2d(4, 4),
                scene(&device),
                &[base, second_empty],
            ),
            Err(PlacementRenderBridgeError::AliasedBuffer {
                resource: "deletion_indirect",
                other_resource: "deletion_indirect",
                ..
            })
        ));
    }

    #[test]
    fn deletion_copies_resets_filters_then_draws_from_filtered_indirect() {
        let device = device();
        let item = with_deletion(&device);
        let upstream_indirect = item.indirect.imported.buffer;
        let deletion_indirect = item
            .work
            .unwrap()
            .deletion
            .unwrap()
            .indirect
            .imported
            .buffer;
        let graph = build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[item])
            .expect("deletion graph lowering");
        let recorded = graph.record(&device).expect("deletion graph recording");
        let passes = &recorded.commands().passes;
        assert_eq!(recorded.plan().pass_count(), 9);
        let Pass::Transfer { commands, .. } = &passes[5] else {
            panic!("argument initialization must begin with a transfer copy");
        };
        assert_eq!(
            commands.as_slice(),
            &[CopyCommand::CopyBufferToBuffer {
                src: upstream_indirect,
                src_offset: 0,
                dst: deletion_indirect,
                dst_offset: 0,
                size: 16,
            }]
        );
        for (index, expected_dispatch) in [(6, 1), (7, 1)] {
            let Pass::Compute { commands, .. } = &passes[index] else {
                panic!("deletion preparation/filter pass must be compute");
            };
            assert!(matches!(
                commands.as_slice(),
                [
                    ComputeCommand::SetPipeline(_),
                    ComputeCommand::SetBindGroup { index: 0, .. },
                    ComputeCommand::Dispatch { x, y: 1, z: 1 },
                ] if *x == expected_dispatch
            ));
        }
        let Pass::Render { commands, .. } = passes.last().expect("draw pass") else {
            panic!("last pass must draw");
        };
        assert!(matches!(
            commands.as_slice(),
            [
                RenderCommand::SetPipeline(_),
                RenderCommand::SetBindGroup { index: 0, .. },
                RenderCommand::DrawIndirect { buffer, offset: 0 },
            ] if *buffer == deletion_indirect
        ));
    }

    #[test]
    fn rejects_duplicate_cells_before_graph_lowering() {
        let device = device();
        let first = item(&device);
        let mut second = item(&device);
        second.stream.revision += 1;
        assert!(matches!(
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[first, second],),
            Err(PlacementRenderBridgeError::DuplicateSector { .. })
        ));
    }

    #[test]
    fn rejects_zero_buffers_dispatches_and_misaligned_indirect_offsets() {
        let device = device();

        let mut zero_buffer = item(&device);
        zero_buffer.records.bytes = 0;
        assert!(matches!(
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[zero_buffer],),
            Err(PlacementRenderBridgeError::ZeroBuffer {
                resource: "records",
                ..
            })
        ));

        let mut zero_dispatch = item(&device);
        zero_dispatch.work.as_mut().unwrap().project.dispatch = [1, 0, 1];
        assert!(matches!(
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[zero_dispatch],),
            Err(PlacementRenderBridgeError::ZeroDispatch {
                stage: "project",
                dispatch: [1, 0, 1],
                ..
            })
        ));

        let mut misaligned = item(&device);
        misaligned.work.as_mut().unwrap().draw.indirect_offset = 2;
        assert!(matches!(
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[misaligned],),
            Err(PlacementRenderBridgeError::MisalignedIndirectOffset { offset: 2, .. })
        ));

        let mut missing_work = item(&device);
        missing_work.work = None;
        assert!(matches!(
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[missing_work],),
            Err(PlacementRenderBridgeError::CandidateWorkMismatch {
                work_present: false,
                ..
            })
        ));

        let mut unexpected_work = item(&device);
        unexpected_work.stream.candidate_count = 0;
        assert!(matches!(
            build_placement_render_graph(
                Extent3d::new_2d(4, 4),
                scene(&device),
                &[unexpected_work],
            ),
            Err(PlacementRenderBridgeError::CandidateWorkMismatch {
                work_present: true,
                ..
            })
        ));
    }

    #[test]
    fn two_streams_lower_in_stable_cell_order() {
        let device = device();
        let mut high = item(&device);
        high.stream.cell = prism_world_stream::SectorId::new(0, 2, 0, 0);
        let mut low = item(&device);
        low.stream.cell = prism_world_stream::SectorId::new(0, 1, 0, 0);
        let graph =
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[high, low])
                .expect("two-stream graph lowering");
        let recorded = graph.record(&device).expect("two-stream recording");
        assert_eq!(recorded.commands().passes.len(), 12);
        let labels = recorded
            .commands()
            .passes
            .iter()
            .filter_map(|pass| match pass {
                Pass::Compute { label, .. } => label.as_deref(),
                Pass::Render { descriptor, .. } => descriptor.label.as_deref(),
                Pass::Transfer { .. } => None,
            })
            .collect::<Vec<_>>();
        assert!(labels[0].contains("level: 0, x: 1"));
        assert!(labels[6].contains("level: 0, x: 2"));
    }

    fn binding_mut<'a>(
        item: &'a mut PlacementRenderStream,
        name: &str,
    ) -> &'a mut PlacementImportedBuffer {
        match name {
            "source_params" => &mut item.source_params,
            "project_params" => &mut item.project_params,
            "field_params" => &mut item.field_params,
            "records" => &mut item.records,
            "samples" => &mut item.samples,
            "attributes" => &mut item.attributes,
            "compact" => &mut item.compact,
            "indirect" => &mut item.indirect,
            "status" => &mut item.status,
            "heights" => &mut item.heights,
            "brushes" => &mut item.brushes,
            name => {
                let deletion = item.work.as_mut().unwrap().deletion.as_mut().unwrap();
                match name {
                    "deletion_params" => &mut deletion.params,
                    "deletion_keys" => &mut deletion.keys,
                    "deletion_compact" => &mut deletion.compact,
                    "deletion_indirect" => &mut deletion.indirect,
                    _ => panic!("unknown binding: {name}"),
                }
            }
        }
    }

    const BUFFER_NAMES: [&str; 15] = [
        "source_params",
        "project_params",
        "field_params",
        "records",
        "samples",
        "attributes",
        "compact",
        "indirect",
        "status",
        "heights",
        "brushes",
        "deletion_params",
        "deletion_keys",
        "deletion_compact",
        "deletion_indirect",
    ];
    const WRITABLE_NAMES: [&str; 8] = [
        "records",
        "samples",
        "attributes",
        "compact",
        "indirect",
        "status",
        "deletion_compact",
        "deletion_indirect",
    ];

    #[test]
    fn rejects_every_shader_buffer_below_its_abi_minimum() {
        let device = device();
        let minimums = [
            192,
            128,
            128,
            32 * 64,
            32 * 64,
            32 * 16,
            32 * 4,
            16,
            4,
            4,
            64,
            128,
            56,
            32 * 4,
            16,
        ];
        for (name, minimum) in BUFFER_NAMES.into_iter().zip(minimums) {
            let mut item = with_deletion(&device);
            binding_mut(&mut item, name).bytes = minimum - 1;
            assert!(
                matches!(
                    validate_stream(&item),
                    Err(PlacementRenderBridgeError::InvalidBufferSize {resource, minimum: actual, ..})
                        if resource == name && actual == minimum
                ),
                "{name} must require {minimum} bytes"
            );
        }
    }

    #[test]
    fn deletion_keys_require_the_complete_uploaded_set() {
        let device = device();
        for key_count in [0, 1, 2, u32::MAX] {
            let mut item = with_deletion(&device);
            item.work
                .as_mut()
                .unwrap()
                .deletion
                .as_mut()
                .unwrap()
                .request
                .key_count = key_count;
            let required = 56 * u64::from(key_count.max(1));
            binding_mut(&mut item, "deletion_keys").bytes = required;
            validate_stream(&item).expect("exact key extent");
            binding_mut(&mut item, "deletion_keys").bytes -= 1;
            assert!(
                matches!(validate_stream(&item), Err(PlacementRenderBridgeError::InvalidBufferSize { resource: "deletion_keys", minimum, .. }) if minimum == required)
            );
        }
    }

    #[test]
    fn rejects_incomplete_repeated_mismatched_or_wrapping_deletion_dispatch() {
        let device = device();
        for (candidate_count, dispatch, workgroups) in [
            (128, [1, 1, 1], [1, 1]),
            (32, [1, 1, 2], [1, 1]),
            (32, [2, 2, 1], [1, 2]),
            (32, [2, 2, 1], [2, 1]),
            (32, [65536, 1025, 1], [65536, 1025]),
            (32, [u32::MAX, u32::MAX, 1], [u32::MAX, u32::MAX]),
        ] {
            let mut item = with_deletion(&device);
            item.stream.candidate_count = candidate_count;
            for (name, stride) in [
                ("records", 64),
                ("samples", 64),
                ("attributes", 16),
                ("compact", 4),
                ("deletion_compact", 4),
            ] {
                binding_mut(&mut item, name).bytes = u64::from(candidate_count) * stride;
            }
            let deletion = item.work.as_mut().unwrap().deletion.as_mut().unwrap();
            deletion.request.candidate_count = candidate_count;
            deletion.request.workgroups = workgroups;
            deletion.stage.dispatch = dispatch;
            assert!(
                matches!(
                    validate_stream(&item),
                    Err(PlacementRenderBridgeError::InvalidDeletionRequest { .. })
                ),
                "N={candidate_count}, dispatch={dispatch:?}, uniform groups={workgroups:?}"
            );
        }
        let mut item = with_deletion(&device);
        item.work
            .as_mut()
            .unwrap()
            .deletion
            .as_mut()
            .unwrap()
            .request
            .candidate_count += 1;
        assert!(matches!(
            validate_stream(&item),
            Err(PlacementRenderBridgeError::InvalidDeletionRequest { .. })
        ));
    }

    #[test]
    fn accepts_covered_2d_dispatch_and_exact_u32_index_boundary() {
        let device = device();
        for (count, groups) in [
            (32, [1, 1]),
            (128, [1, 2]),
            (129, [2, 2]),
            (u32::MAX, [65536, 1024]),
        ] {
            let mut item = with_deletion(&device);
            item.stream.candidate_count = count;
            for (name, stride) in [
                ("records", 64),
                ("samples", 64),
                ("attributes", 16),
                ("compact", 4),
                ("deletion_compact", 4),
            ] {
                // Validation-only boundary: no giant physical allocation.
                binding_mut(&mut item, name).bytes = u64::from(count) * stride;
            }
            let deletion = item.work.as_mut().unwrap().deletion.as_mut().unwrap();
            deletion.request.candidate_count = count;
            deletion.request.workgroups = groups;
            deletion.stage.dispatch = [groups[0], groups[1], 1];
            validate_stream(&item).expect("covered non-wrapping dispatch");
        }
    }

    #[test]
    fn rejects_nonzero_indirect_record_even_when_aligned_and_in_bounds() {
        let device = device();
        for deletion_enabled in [false, true] {
            let mut item = if deletion_enabled {
                with_deletion(&device)
            } else {
                item(&device)
            };
            if deletion_enabled {
                binding_mut(&mut item, "deletion_indirect").bytes = 32;
            } else {
                item.indirect.bytes = 32;
            }
            item.work.as_mut().unwrap().draw.indirect_offset = 16;
            assert!(matches!(
                validate_stream(&item),
                Err(PlacementRenderBridgeError::UnsupportedIndirectOffset { offset: 16, .. })
            ));
        }
    }

    #[test]
    fn rejects_duplicate_reset_workgroups() {
        let device = device();
        let mut item = with_deletion(&device);
        item.reset.dispatch = [1, 1, 2];
        assert!(matches!(
            validate_stream(&item),
            Err(PlacementRenderBridgeError::InvalidResetDispatch { stage: "reset", .. })
        ));
        item.reset.dispatch = [1, 1, 1];
        item.work
            .as_mut()
            .unwrap()
            .deletion
            .as_mut()
            .unwrap()
            .clear_instance_count
            .dispatch = [2, 1, 1];
        assert!(matches!(
            validate_stream(&item),
            Err(PlacementRenderBridgeError::InvalidResetDispatch {
                stage: "deletion_count_reset",
                ..
            })
        ));
    }

    #[test]
    fn rejects_all_writable_alias_pairs_within_and_between_streams() {
        let device = device();
        let base = with_deletion(&device);
        let mut distinct = with_deletion(&device);
        distinct.stream.cell = prism_world_stream::SectorId::new(1, 1, 0, 0);
        for writable in WRITABLE_NAMES {
            for other in BUFFER_NAMES {
                for across_streams in [false, true] {
                    if !across_streams && writable == other {
                        continue;
                    }
                    let mut first = base;
                    let mut second = distinct;
                    let source = if across_streams {
                        binding_mut(&mut second, other).imported
                    } else {
                        binding_mut(&mut first, other).imported
                    };
                    binding_mut(&mut first, writable).imported = source;
                    let streams = if across_streams {
                        vec![first, second]
                    } else {
                        vec![first]
                    };
                    assert!(
                        matches!(
                            build_placement_render_graph(
                                Extent3d::new_2d(4, 4),
                                scene(&device),
                                &streams
                            ),
                            Err(PlacementRenderBridgeError::AliasedBuffer { .. })
                        ),
                        "writable={writable}, other={other}, across_streams={across_streams}"
                    );
                }
            }
        }
    }

    #[test]
    fn shares_immutable_imports_but_rejects_inconsistent_metadata() {
        let device = device();
        let first = with_deletion(&device);
        let mut second = with_deletion(&device);
        second.stream.cell = prism_world_stream::SectorId::new(1, 1, 0, 0);
        second.heights = first.heights;
        second.brushes = first.brushes;
        second
            .work
            .as_mut()
            .unwrap()
            .deletion
            .as_mut()
            .unwrap()
            .keys = first.work.unwrap().deletion.unwrap().keys;
        let graph =
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[first, second])
                .expect("shared read-only buffers");
        let recorded = graph.record(&device).expect("shared imports record");
        assert_eq!(recorded.commands().passes.len(), 18);
        recorded.release(&device);
        second.heights.bytes += 4;
        assert!(matches!(
            build_placement_render_graph(Extent3d::new_2d(4, 4), scene(&device), &[first, second]),
            Err(PlacementRenderBridgeError::ConflictingBufferImport {
                resource: "heights",
                ..
            })
        ));

        let mut graph = RenderGraph::new(Extent3d::new_2d(4, 4));
        let mut imported = BTreeMap::new();
        let a = import_buffer(&mut graph, &mut imported, "first", first.heights).unwrap();
        let b = import_buffer(&mut graph, &mut imported, "shared", first.heights).unwrap();
        assert_eq!(a, b);
        assert_eq!(imported.len(), 1);
    }
}
