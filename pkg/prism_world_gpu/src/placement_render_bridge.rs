//! Lowering for heightfield placement into Prism's executable render graph.
//!
//! The placement kernels remain owned by the caller. This module owns only the
//! graph declaration and command recording contract: callers provide imported
//! RHI resources and pipeline/bind-group identities from the exact device that
//! will record the graph. No backend object is allocated, submitted, or
//! retired here.

use core::fmt;
use std::vec::Vec;

use prism_render_driver::{
    BindGroupId, BufferUsages, ComputePipelineId, Extent3d, RenderPipelineId,
};
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ExecuteContext, ImportedBuffer, ImportedTexture,
    PassFlags, RenderGraph, TextureDesc,
};

use crate::world_placement_graph::PlacementStream;

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
    /// Production raster consumer.
    pub draw: PlacementDrawStage,
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
    let mut graph = RenderGraph::new(extent);
    let mut scene_handle =
        graph.try_import_texture("world placement scene", scene.descriptor, scene.imported)?;

    let mut ordered = streams.to_vec();
    ordered.sort_unstable_by_key(|item| (item.stream.cell, item.stream.revision));
    for pair in ordered.windows(2) {
        if pair[0].stream.cell == pair[1].stream.cell {
            return Err(PlacementRenderBridgeError::DuplicateSector {
                stream: pair[1].stream,
            });
        }
    }

    for item in ordered {
        validate_stream(&item)?;
        let handles = import_stream(&mut graph, &item)?;

        let (indirect, status) = add_reset_pass(
            &mut graph,
            handles.indirect,
            handles.status,
            item.reset,
            item.stream,
        )?;

        let Some(work) = item.work else {
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

        let draw_records = records;
        let draw_samples = samples;
        let draw_attributes = attributes;
        let draw_compact = compact;
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
                let _ = builder.read_indirect(indirect);
                next_scene = builder.color_attachment(previous_scene);
                move |commands: &mut ExecuteContext<'_>| {
                    commands
                        .set_render_pipeline(draw_stage.pipeline)
                        .set_render_bind_group(0, draw_stage.bind_group, Vec::new())
                        .draw_indirect(indirect, draw_stage.indirect_offset);
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

fn import_stream(
    graph: &mut RenderGraph,
    item: &PlacementRenderStream,
) -> Result<StreamHandles, CompileError> {
    let import = |name: &'static str, value: PlacementImportedBuffer, graph: &mut RenderGraph| {
        graph.try_import_buffer(
            name,
            BufferDesc::new(value.bytes, BufferUsages::NONE),
            value.imported,
        )
    };
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
    if item.reset.dispatch.contains(&0) {
        return Err(PlacementRenderBridgeError::ZeroDispatch {
            stream: item.stream,
            stage: "reset",
            dispatch: item.reset.dispatch,
        });
    }
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
    if work.draw.indirect_offset % 4 != 0 {
        return Err(PlacementRenderBridgeError::MisalignedIndirectOffset {
            stream: item.stream,
            offset: work.draw.indirect_offset,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{
        BufferDescriptor, BufferState, ComputeCommand, DeviceDescriptor, Instance, Pass,
        RenderCommand, RenderDevice, RequestAdapterOptions, TextureDescriptor, TextureDimension,
        TextureFormat, TextureState, TextureUsages,
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
            source_params: imported_buffer(device, 64),
            project_params: imported_buffer(device, 64),
            field_params: imported_buffer(device, 64),
            records: imported_buffer(device, 64),
            samples: imported_buffer(device, 64),
            attributes: imported_buffer(device, 16),
            compact: imported_buffer(device, 16),
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
                draw: PlacementDrawStage {
                    pipeline: RenderPipelineId::from_parts(6, 0),
                    bind_group: BindGroupId::from_parts(6, 0),
                    indirect_offset: 4,
                },
            }),
        }
    }

    fn reset_only_item(device: &impl RenderDevice) -> PlacementRenderStream {
        let mut item = item(device);
        item.stream.candidate_count = 0;
        item.work = None;
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
                RenderCommand::DrawIndirect { offset: 4, .. },
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
}
