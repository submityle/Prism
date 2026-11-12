//! GPU sort/cull and renderer-matrix resource wiring for particle renderers.
//!
//! This module is intentionally independent from the application frame loop.
//! It owns the RHI resources and declares their RenderGraph dependencies, while
//! the architecture crate remains the owner of renderer and sort contracts.
//! The cull pass writes sort items, the sort passes order those items by the
//! full-precision ordered f32 depth key for native cameras (the contract's
//! quantized key for explicit fixtures), with a particle-index tie break or
//! explicit RibbonId/birth topology for strips. The
//! final passes publish one shared visible-index ABI plus renderer-specific
//! indirect argument records.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use std::sync::{Mutex, Once};

mod camera;
use camera::CAMERA_BYTES;
pub use camera::{ParticleCamera, ParticleCameraError};
mod strip;
pub use strip::{ParticleStripSettings, ParticleStripTessellation};
use strip::{StripGpu, TessellationGpu, STRIP_VERTEX_BYTES};

use prism_render_architecture::particle::indirect_draw::{
    RendererGeometry, INDICES_PER_SEGMENT, SPRITE_QUAD_VERTEX_COUNT,
};
use prism_render_architecture::particle::renderers::{RendererKind, RendererPlan};
use prism_render_architecture::particle::sort_cull::{
    sort_cull_buffer_layout, SortCullBufferLayout, SortCullLayoutError,
};
use prism_render_architecture::particle::SortStrategy;
use prism_render_architecture::render_fabric::ObjectResourceIdentity;
use prism_render_driver as rhi;
use prism_render_driver::{BufferState, PipelineStages};
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ImportedBuffer, ImportedTexture,
    PassFlags, PassKind, RecordedGraph, RenderGraph, SizeClass, TextureDesc, TextureHandle,
};
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, ColorTargetState, CompareFunction,
    ComputePipelineDescriptor, ComputePipelineId, DepthStencilState, FragmentState, IndexFormat,
    MultisampleState, PipelineLayoutDescriptor, PipelineLayoutId, PrimitiveState,
    PrimitiveTopology, RenderDevice, RenderPipelineDescriptor, RenderPipelineId, RenderQueue,
    ShaderModuleDescriptor, ShaderModuleId, ShaderStages, TextureFormat, TextureState,
    TextureUsages, TextureViewDescriptor, VertexState,
};

use crate::gpu::{bind_buffer, buffer_entry};

/// The fixed offscreen target used by the standalone renderer-matrix raster
/// consumer. Callers embedding the stage in a larger frame graph can use the
/// same pipeline/bindings with their own color target.
pub const RENDERER_MATRIX_TARGET_SIZE: u32 = 128;

/// Shared renderer/bridge input stride. Bytes 0..32 retain the original ABI;
/// bytes 32..48 carry RibbonId, birth low/high words and topology flags.
pub const PARTICLE_RENDER_INPUT_BYTES: u64 = 48;

/// Access direction authenticated by a graph-free renderer phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleRendererBufferAccess {
    /// The phase reads the physical range.
    Read,
    /// The phase writes the physical range.
    Write,
    /// The phase reads and writes the physical range.
    ReadWrite,
}

/// Semantic identity for a physical renderer resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleRendererBufferRole {
    /// Renderer input records.
    Input,
    /// GPU-authored input count.
    InputCount,
    /// Emitter route table.
    RouteTable,
    /// Cull frustum planes.
    Frustum,
    /// Camera uniform.
    Camera,
    /// Cull/compact control words.
    Control,
    /// Cull/compact parameters.
    Params,
    /// Bitonic sort source records.
    SortSource,
    /// Bitonic sort destination records.
    SortDestination,
    /// One bitonic sort step uniform.
    SortStep,
    /// Compact visible source indices.
    VisibleIndices,
    /// Strip settings uniform.
    StripParams,
    /// Strip prefix source records.
    StripSource,
    /// Strip prefix destination records.
    StripDestination,
    /// GPU-generated strip vertices.
    StripVertices,
    /// Tessellation settings uniform.
    TessellationParams,
    /// Tessellation scan source records.
    TessellationSource,
    /// Tessellation scan destination records.
    TessellationDestination,
    /// Tessellated subedge geometry.
    TessellationGeometry,
    /// Sprite indirect arguments.
    SpriteArguments,
    /// Indexed indirect arguments.
    IndexedArguments,
    /// Route parameters uniform.
    RouteParams,
    /// Static surface mesh vertex storage.
    SurfaceMeshVertices,
    /// Static surface index buffer.
    SurfaceIndexBuffer,
    /// Surface material uniform.
    SurfaceMaterial,
}

/// A physical buffer range used by one renderer record phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParticleRendererPhysicalBuffer {
    /// Semantic role of the bound range.
    pub role: ParticleRendererBufferRole,
    /// Exact RHI buffer bound by the phase's bind group or draw.
    pub buffer: BufferId,
    /// Byte length admitted for this phase.
    pub bytes: u64,
    /// Access required by this phase.
    pub access: ParticleRendererBufferAccess,
}

/// Stable identity of every physically separate renderer phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleRendererRecordPhaseKind {
    /// Clears control and indirect argument records.
    Reset,
    /// Seeds padded sort records.
    Seed,
    /// Culls source records into the sort list.
    Cull,
    /// One concrete bitonic sort step.
    Sort { step: u32 },
    /// Compacts visible source indices.
    Compact,
    /// Builds strip edge/prefix records.
    StripEdges,
    /// One concrete strip prefix scan step.
    StripScan { step: u32 },
    /// Counts tessellated subedges and clears the fixed geometry budget.
    TessellationCount,
    /// One concrete tessellation prefix scan step.
    TessellationScan { step: u32 },
    /// Emits compact tessellated geometry and final indexed arguments.
    TessellationEmit,
    /// Emits sprite/indexed indirect arguments.
    Args,
    /// Records the GPU-driven raster draw.
    Raster,
}

/// Pipeline selected by a graph-free renderer phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleRendererRecordPipeline {
    /// Compute pipeline and its bind group.
    Compute(ComputePipelineId),
    /// Raster pipeline and its bind group.
    Raster(RenderPipelineId),
}

/// Indirect draw metadata attached to the raster phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParticleRendererDrawMetadata {
    /// Whether the argument record is indexed.
    pub indexed: bool,
    /// Physical argument buffer.
    pub arguments: BufferId,
    /// Byte range containing the one indirect argument record.
    pub arguments_bytes: u64,
    /// Physical index buffer for indexed draws.
    pub index_buffer: Option<BufferId>,
    /// Byte range bound from the index buffer.
    pub index_bytes: u64,
    /// Byte offset of the argument record.
    pub arguments_offset: u64,
}

/// One executable graph-free renderer phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParticleRendererRecordPhase {
    kind: ParticleRendererRecordPhaseKind,
    pipeline: ParticleRendererRecordPipeline,
    bind_group: BindGroupId,
    dispatch: Option<[u32; 3]>,
    draw: Option<ParticleRendererDrawMetadata>,
    indirect_ordinal: Option<usize>,
    index_ordinal: Option<usize>,
    resources: Vec<ParticleRendererPhysicalBuffer>,
}

impl ParticleRendererRecordPhase {
    /// Stable physical phase identity.
    #[must_use]
    pub const fn kind(&self) -> ParticleRendererRecordPhaseKind {
        self.kind
    }

    /// Pipeline selected by this phase.
    #[must_use]
    pub const fn pipeline(&self) -> ParticleRendererRecordPipeline {
        self.pipeline
    }

    /// Bind group selected by this phase.
    #[must_use]
    pub const fn bind_group(&self) -> BindGroupId {
        self.bind_group
    }

    /// Compute dispatch dimensions, or `None` for the raster phase.
    #[must_use]
    pub const fn dispatch(&self) -> Option<[u32; 3]> {
        self.dispatch
    }

    /// Raster indirect draw metadata, or `None` for compute phases.
    #[must_use]
    pub const fn draw(&self) -> Option<ParticleRendererDrawMetadata> {
        self.draw
    }

    /// Exact physical buffers and ranges authenticated for this phase.
    #[must_use]
    pub fn resources(&self) -> &[ParticleRendererPhysicalBuffer] {
        &self.resources
    }
}

/// Error returned before a record-plan command is emitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleRendererRecordError {
    /// The requested phase does not exist.
    InvalidPhase { index: usize },
    /// A compute phase was sent to a raster pass or vice versa.
    WrongPassKind {
        phase: ParticleRendererRecordPhaseKind,
        actual: PassKind,
    },
    /// A draw resource was not declared by the compiled Fabric pass.
    ResourceNotDeclared {
        /// Resource role that failed declaration lookup.
        role: ParticleRendererBufferRole,
    },
}

/// Graph-free command plan for one particle renderer instance.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleRendererRecordPlan {
    capacity: u32,
    route: ParticleRendererRoute,
    phases: Vec<ParticleRendererRecordPhase>,
}

impl ParticleRendererRecordPlan {
    /// Renderer work capacity covered by the plan.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Renderer route covered by the plan.
    #[must_use]
    pub const fn route(&self) -> ParticleRendererRoute {
        self.route
    }

    /// Ordered physical phases, including every sort/scan level.
    #[must_use]
    pub fn phases(&self) -> &[ParticleRendererRecordPhase] {
        &self.phases
    }

    /// Records one phase through an existing Render Fabric runtime context.
    ///
    /// The runtime owns resource admission and synchronization; this helper
    /// contributes only the already-authenticated driver commands.
    pub fn record_runtime_phase(
        &self,
        index: usize,
        identity: ObjectResourceIdentity,
        context: &mut prism_render_architecture::render_fabric::RuntimePassContext<'_, '_>,
    ) -> Result<(), ParticleRendererRecordError> {
        let phase = self
            .phases
            .get(index)
            .ok_or(ParticleRendererRecordError::InvalidPhase { index })?;
        let expected = match phase.kind {
            ParticleRendererRecordPhaseKind::Raster => PassKind::Raster,
            _ => PassKind::Compute,
        };
        if context.commands().kind() != expected {
            return Err(ParticleRendererRecordError::WrongPassKind {
                phase: phase.kind,
                actual: context.commands().kind(),
            });
        }
        match phase.pipeline {
            ParticleRendererRecordPipeline::Compute(pipeline) => {
                let dispatch = phase
                    .dispatch
                    .expect("compute phase must have dispatch metadata");
                context
                    .commands()
                    .set_compute_pipeline(pipeline)
                    .set_compute_bind_group(0, phase.bind_group, Vec::new())
                    .dispatch(dispatch[0], dispatch[1], dispatch[2]);
            }
            ParticleRendererRecordPipeline::Raster(pipeline) => {
                context
                    .commands()
                    .set_render_pipeline(pipeline)
                    .set_render_bind_group(0, phase.bind_group, Vec::new());
                if let Some(draw) = phase.draw {
                    let arguments = phase.indirect_ordinal.ok_or(
                        ParticleRendererRecordError::ResourceNotDeclared {
                            role: if draw.indexed {
                                ParticleRendererBufferRole::IndexedArguments
                            } else {
                                ParticleRendererBufferRole::SpriteArguments
                            },
                        },
                    )?;
                    if draw.indexed {
                        let index_ordinal = phase.index_ordinal.ok_or(
                            ParticleRendererRecordError::ResourceNotDeclared {
                                role: ParticleRendererBufferRole::SurfaceIndexBuffer,
                            },
                        )?;
                        if !context.object_set_index_buffer(
                            identity,
                            index_ordinal,
                            IndexFormat::Uint32,
                            0,
                            core::num::NonZeroU64::new(draw.index_bytes),
                        ) {
                            return Err(ParticleRendererRecordError::ResourceNotDeclared {
                                role: ParticleRendererBufferRole::SurfaceIndexBuffer,
                            });
                        }
                        if !context.object_draw_indexed_indirect(
                            identity,
                            arguments,
                            draw.arguments_offset,
                        ) {
                            return Err(ParticleRendererRecordError::ResourceNotDeclared {
                                role: ParticleRendererBufferRole::IndexedArguments,
                            });
                        }
                    } else if context.object_draw_indirect(
                        identity,
                        arguments,
                        draw.arguments_offset,
                    ) {
                    } else {
                        return Err(ParticleRendererRecordError::ResourceNotDeclared {
                            role: ParticleRendererBufferRole::SpriteArguments,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Private physical resources used by the graph-free strip lowering. The
/// direct graph path retains its own resources in `StripGpu`; this second set
/// lets Fabric record strip work without asking the graph module to create
/// passes or without reaching into the child module's private state.
struct RendererRecordPlanStrip {
    capacity: u32,
    buffers: [BufferId; 2],
    params: BufferId,
    shader: ShaderModuleId,
    layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    init_pipeline: ComputePipelineId,
    scan_pipeline: ComputePipelineId,
    init_group: BindGroupId,
    scan_groups: Vec<BindGroupId>,
    steps: Vec<BufferId>,
    tessellation: Option<RendererRecordPlanTessellation>,
}

struct RendererRecordPlanTessellation {
    settings: ParticleStripTessellation,
    capacity: u32,
    route: ParticleRendererRoute,
    emitter_local: bool,
    geometry: BufferId,
    records: [BufferId; 2],
    params: BufferId,
    shader: ShaderModuleId,
    layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    count_pipeline: ComputePipelineId,
    scan_pipeline: ComputePipelineId,
    emit_pipeline: ComputePipelineId,
    count_group: BindGroupId,
    emit_group: BindGroupId,
    scan_groups: Vec<BindGroupId>,
    steps: Vec<BufferId>,
}

impl RendererRecordPlanStrip {
    fn new<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        route: ParticleRendererRoute,
        input: BufferId,
        input_count: BufferId,
        visible: BufferId,
        control: BufferId,
        camera: BufferId,
        indexed_args: BufferId,
        tessellation: Option<ParticleStripTessellation>,
        emitter_local: bool,
    ) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let buffers = core::array::from_fn(|_| {
            device.create_buffer(&BufferDescriptor::new(
                u64::from(capacity) * STRIP_VERTEX_BYTES,
                storage,
            ))
        });
        let params = device.create_buffer(&BufferDescriptor::new(
            32,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_strip.wgsl"
        )));
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-record-plan-strip-bgl")),
            entries: (0..8)
                .map(|binding| {
                    buffer_entry(
                        binding,
                        match binding {
                            4 | 7 => BufferBindingType::Uniform,
                            6 => BufferBindingType::Storage { read_only: false },
                            _ => BufferBindingType::Storage { read_only: true },
                        },
                        ShaderStages::COMPUTE,
                    )
                })
                .collect(),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-record-plan-strip-pl")),
            bind_group_layouts: vec![layout],
            push_constant_ranges: Vec::new(),
        });
        let init_pipeline = make_compute(device, pipeline_layout, shader, "strip_edges");
        let scan_pipeline = make_compute(device, pipeline_layout, shader, "strip_scan");
        let group = |source, destination, step| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(String::from("particle-record-plan-strip-bg")),
                layout: Some(layout),
                entries: vec![
                    bind_buffer(0, input),
                    bind_buffer(1, visible),
                    bind_buffer(2, control),
                    bind_buffer(3, input_count),
                    bind_buffer(4, params),
                    bind_buffer(5, source),
                    bind_buffer(6, destination),
                    bind_buffer(7, step),
                ],
            })
        };
        let init_step = device.create_buffer(&BufferDescriptor::new(
            16,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let init_group = group(buffers[1], buffers[0], init_step);
        let mut steps = vec![init_step];
        let mut scan_groups = Vec::new();
        let mut offset = 1u32;
        while offset < capacity {
            let step = device.create_buffer(&BufferDescriptor::new(
                16,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ));
            let index = scan_groups.len() % 2;
            scan_groups.push(group(buffers[index], buffers[1 - index], step));
            steps.push(step);
            offset *= 2;
        }
        let tessellation = tessellation.map(|settings| {
            RendererRecordPlanTessellation::new(
                device,
                capacity,
                route,
                input,
                visible,
                control,
                buffers[scan_groups.len() % 2],
                camera,
                indexed_args,
                settings,
                emitter_local,
            )
        });
        Self {
            capacity,
            buffers,
            params,
            shader,
            layout,
            pipeline_layout,
            init_pipeline,
            scan_pipeline,
            init_group,
            scan_groups,
            steps,
            tessellation,
        }
    }

    fn output(&self) -> BufferId {
        self.buffers[self.scan_groups.len() % 2]
    }

    fn initialize_static(&self, queue: &impl RenderQueue) {
        for (index, step) in self.steps.iter().copied().enumerate() {
            let mut bytes = [0; 16];
            bytes[..4].copy_from_slice(&(1u32 << index).to_le_bytes());
            queue.write_buffer(step, 0, &bytes);
        }
        if let Some(tessellation) = &self.tessellation {
            tessellation.initialize_static(queue);
        }
    }

    fn upload(&self, queue: &impl RenderQueue, material: ParticleSurfaceMaterial) {
        let settings = material.strip;
        queue.write_buffer(
            self.params,
            0,
            &record_plan_strip_settings_bytes(settings, self.capacity),
        );
        if let Some(tessellation) = &self.tessellation {
            tessellation.upload(queue, material);
        }
    }

    fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        if let Some(tessellation) = &self.tessellation {
            tessellation.destroy(device);
        }
        device.destroy_bind_group(self.init_group);
        for group in &self.scan_groups {
            device.destroy_bind_group(*group);
        }
        for step in &self.steps {
            device.destroy_buffer(*step);
        }
        for buffer in self.buffers {
            device.destroy_buffer(buffer);
        }
        device.destroy_buffer(self.params);
        device.destroy_compute_pipeline(self.init_pipeline);
        device.destroy_compute_pipeline(self.scan_pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.layout);
        device.destroy_shader_module(self.shader);
    }
}

impl RendererRecordPlanTessellation {
    fn new<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        route: ParticleRendererRoute,
        input: BufferId,
        visible: BufferId,
        control: BufferId,
        strip: BufferId,
        camera: BufferId,
        indexed_args: BufferId,
        settings: ParticleStripTessellation,
        emitter_local: bool,
    ) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let geometry = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * u64::from(settings.max_subdivisions) * 48,
            storage,
        ));
        let records = core::array::from_fn(|_| {
            device.create_buffer(&BufferDescriptor::new(u64::from(capacity) * 16, storage))
        });
        let params = device.create_buffer(&BufferDescriptor::new(
            48,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_tessellate.wgsl"
        )));
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-record-plan-tessellation-bgl")),
            entries: (0..11)
                .map(|binding| {
                    buffer_entry(
                        binding,
                        match binding {
                            4 | 7 | 10 => BufferBindingType::Uniform,
                            6 | 8 | 9 => BufferBindingType::Storage { read_only: false },
                            _ => BufferBindingType::Storage { read_only: true },
                        },
                        ShaderStages::COMPUTE,
                    )
                })
                .collect(),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-record-plan-tessellation-pl")),
            bind_group_layouts: vec![layout],
            push_constant_ranges: Vec::new(),
        });
        let count_pipeline = make_compute(device, pipeline_layout, shader, "tess_count");
        let scan_pipeline = make_compute(device, pipeline_layout, shader, "tess_scan");
        let emit_pipeline = make_compute(device, pipeline_layout, shader, "tess_emit");
        let scan_count = capacity.next_power_of_two().trailing_zeros() as usize;
        let mut steps = Vec::new();
        let step_count = scan_count.max(1);
        for _ in 0..step_count {
            steps.push(device.create_buffer(&BufferDescriptor::new(
                16,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )));
        }
        let group = |source, destination, step| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(String::from("particle-record-plan-tessellation-bg")),
                layout: Some(layout),
                entries: vec![
                    bind_buffer(0, input),
                    bind_buffer(1, visible),
                    bind_buffer(2, control),
                    bind_buffer(3, strip),
                    bind_buffer(4, params),
                    bind_buffer(5, source),
                    bind_buffer(6, destination),
                    bind_buffer(7, camera),
                    bind_buffer(8, geometry),
                    bind_buffer(9, indexed_args),
                    bind_buffer(10, step),
                ],
            })
        };
        let count_group = group(records[1], records[0], steps[0]);
        let emit_group = group(
            records[scan_count % 2],
            records[1 - scan_count % 2],
            steps[0],
        );
        let scan_groups = (0..scan_count)
            .map(|index| group(records[index % 2], records[1 - index % 2], steps[index]))
            .collect();
        Self {
            settings,
            capacity,
            route,
            emitter_local,
            geometry,
            records,
            params,
            shader,
            layout,
            pipeline_layout,
            count_pipeline,
            scan_pipeline,
            emit_pipeline,
            count_group,
            emit_group,
            scan_groups,
            steps,
        }
    }

    fn prefixes(&self) -> BufferId {
        self.records[self.capacity.next_power_of_two().trailing_zeros() as usize % 2]
    }

    fn initialize_static(&self, queue: &impl RenderQueue) {
        for (index, step) in self.steps.iter().copied().enumerate() {
            let mut bytes = [0; 16];
            bytes[..4].copy_from_slice(&(1u32 << index).to_le_bytes());
            queue.write_buffer(step, 0, &bytes);
        }
    }

    fn upload(&self, queue: &impl RenderQueue, material: ParticleSurfaceMaterial) {
        queue.write_buffer(self.params, 0, &record_plan_tessellation_params(self, material));
    }

    fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        for group in self
            .scan_groups
            .iter()
            .chain(core::iter::once(&self.count_group))
            .chain(core::iter::once(&self.emit_group))
        {
            device.destroy_bind_group(*group);
        }
        for record in self.records {
            device.destroy_buffer(record);
        }
        for step in &self.steps {
            device.destroy_buffer(*step);
        }
        device.destroy_buffer(self.params);
        device.destroy_buffer(self.geometry);
        for pipeline in [self.count_pipeline, self.scan_pipeline, self.emit_pipeline] {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.layout);
        device.destroy_shader_module(self.shader);
    }
}

fn record_plan_strip_settings_bytes(settings: ParticleStripSettings, capacity: u32) -> [u8; 32] {
    let gap = settings.max_birth_gap.unwrap_or(0);
    let words = [
        settings.uv_units_per_world_unit.to_bits(),
        settings.max_segment_length.unwrap_or(0.0).to_bits(),
        u32::from(settings.max_segment_length.is_some())
            | (u32::from(settings.max_birth_gap.is_some()) << 1),
        capacity,
        gap as u32,
        (gap >> 32) as u32,
        0,
        0,
    ];
    let mut bytes = [0; 32];
    for (index, word) in words.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn record_plan_tessellation_params(
    tessellation: &RendererRecordPlanTessellation,
    material: ParticleSurfaceMaterial,
) -> [u8; 48] {
    let span = match tessellation.route.geometry {
        RendererGeometry::Ribbon {
            segments_per_chain, ..
        }
        | RendererGeometry::Beam {
            segments_per_chain, ..
        } => segments_per_chain,
        RendererGeometry::Sprite { .. } | RendererGeometry::Mesh { .. } => 0,
    };
    let anchors = match tessellation.route.geometry {
        RendererGeometry::Ribbon { chain_count, .. }
        | RendererGeometry::Beam { chain_count, .. } => {
            if tessellation.emitter_local {
                tessellation.capacity
            } else {
                chain_count
            }
        }
        RendererGeometry::Sprite { .. } | RendererGeometry::Mesh { .. } => 0,
    };
    let words = [
        tessellation.capacity,
        tessellation.settings.max_subdivisions,
        anchors,
        span,
        tessellation.settings.target_segment_pixels.to_bits(),
        tessellation.settings.target_curve_error_pixels.to_bits(),
        material.strip.uv_units_per_world_unit.to_bits(),
        material.ribbon_width.to_bits(),
        if tessellation.emitter_local {
            0
        } else {
            tessellation.route.first_instance()
        },
        u32::from(tessellation.route.kind() == RendererKind::Beam),
        0,
        0,
    ];
    let mut bytes = [0; 48];
    for (index, word) in words.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

const SURFACE_MATERIAL_BYTES: u64 = 64;
const SURFACE_MESH_VERTEX_STRIDE: u64 = 32;
const AUX_RECORD_BYTES: u64 = 32;
const AUX_PARAMS_BYTES: u64 = 16;

/// One source particle record consumed by the GPU cull pass.
///
/// The payload index is kept explicit so a compact render buffer can refer to
/// a stable pool slot even when the source records were assembled from a
/// subrange or a multi-emitter batch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleRenderInput {
    /// World-space bounding-sphere center and radius.
    pub center_radius: [f32; 4],
    /// Linear depth used only by explicit camera-less fixture views. Native
    /// camera views derive sorting and clip depth from world position instead.
    pub depth: f32,
    /// Pool/instance identity used as a depth-sort tie-break, not birth order.
    pub particle_index: u32,
    /// Bit 0 marks an invalid lifetime and excludes the record from rendering.
    /// Other bits are preserved for renderer-defined use.
    pub flags: u32,
    /// Stable emitter identity authored by the simulation-to-render bridge.
    pub emitter_id: u32,
    /// Strip identity within this emitter, at byte offset 32.
    pub ribbon_id: u32,
    /// Immutable unsigned birth sequence, low word then high word at 36/40.
    /// Must describe actual spawn order across pool-slot reuse, without wrap.
    pub birth_sequence: [u32; 2],
    /// Byte offset 44: bit 0 marks explicit topology as valid; other bits zero.
    /// Missing topology never falls back to pool-index or scatter-order strips.
    pub topology_flags: u32,
}

/// One mesh vertex consumed by the particle Mesh renderer.
///
/// The renderer-matrix surface ABI currently carries local position and UV;
/// normals and material attributes remain owned by the shared particle
/// material path.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleMeshVertex {
    /// Local mesh position in three-dimensional world-aligned surface space.
    pub position: [f32; 3],
    /// Mesh UV passed through to the surface fragment stage.
    pub uv: [f32; 2],
}

impl ParticleMeshVertex {
    /// Creates a finite mesh vertex.
    #[must_use]
    pub const fn new(position: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }

    fn validate(self) -> bool {
        self.position
            .iter()
            .chain(self.uv.iter())
            .all(|value| value.is_finite())
    }
}

impl ParticleRenderInput {
    /// Creates a finite, valid source record.
    #[must_use]
    pub const fn new(center: [f32; 3], radius: f32, depth: f32, particle_index: u32) -> Self {
        Self {
            center_radius: [center[0], center[1], center[2], radius],
            depth,
            particle_index,
            flags: 0,
            emitter_id: 0,
            ribbon_id: 0,
            birth_sequence: [0; 2],
            topology_flags: 0,
        }
    }

    /// Associates this sample with a strip and its true, immutable birth order.
    /// Changing ribbon_id starts a distinct chain. Birth must not be a pool slot.
    #[must_use]
    pub const fn with_ribbon_topology(mut self, ribbon_id: u32, birth_sequence: u64) -> Self {
        self.ribbon_id = ribbon_id;
        self.birth_sequence = [birth_sequence as u32, (birth_sequence >> 32) as u32];
        self.topology_flags = 1;
        self
    }

    fn append_bytes(self, bytes: &mut Vec<u8>) {
        for value in self.center_radius {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.depth.to_le_bytes());
        for word in [
            self.particle_index,
            self.flags,
            self.emitter_id,
            self.ribbon_id,
            self.birth_sequence[0],
            self.birth_sequence[1],
            self.topology_flags,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }

    fn validate(self) -> bool {
        self.center_radius.iter().all(|value| value.is_finite())
            && self.depth.is_finite()
            && self.center_radius[3] >= 0.0
            && self.topology_flags <= 1
    }
}

/// Six inward-facing frustum planes, packed as `(normal.xyz, d)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleCullView {
    /// Validated native camera. None selects explicit linear-depth fixture
    /// rendering; a native camera never consumes the bridge's synthetic depth.
    pub camera: Option<ParticleCamera>,
    /// The six planes used by the conservative sphere test.
    pub planes: [[f32; 4]; 6],
    /// Camera position used by the optional distance test.
    pub camera_position: [f32; 3],
    /// A non-positive value disables distance culling.
    pub max_distance: f32,
    /// Near value used by explicit camera-less fixture depth-key quantization.
    /// Native camera sorting uses full f32 linear depth without this clamp.
    pub near_depth: f32,
    /// Far value used by explicit camera-less fixture depth-key quantization.
    pub far_depth: f32,
    /// `true` produces an ascending key for far-to-near compositing.
    pub back_to_front: bool,
}

impl ParticleCullView {
    /// A view that keeps every finite sphere and quantizes depth in `[0, 1]`.
    #[must_use]
    pub const fn unrestricted() -> Self {
        Self {
            camera: None,
            planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            camera_position: [0.0; 3],
            max_distance: 0.0,
            near_depth: 0.0,
            far_depth: 1.0,
            back_to_front: false,
        }
    }

    /// Creates a view whose frustum, sort origin and projection share one
    /// native camera. Sorting uses full f32 linear depth, independent of
    /// near/far. The finite, increasing near/far interval is retained for API
    /// compatibility with explicit fixtures; it does not bound projection.
    pub fn from_camera(
        camera: ParticleCamera,
        near_depth: f32,
        far_depth: f32,
        back_to_front: bool,
    ) -> Result<Self, ParticleRendererGpuError> {
        let view = Self {
            camera: Some(camera),
            planes: camera.planes(),
            camera_position: camera.position(),
            near_depth,
            far_depth,
            back_to_front,
            max_distance: 0.0,
        };
        if !view.validate() {
            return Err(ParticleRendererGpuError::InvalidView);
        }
        Ok(view)
    }

    fn validate(self) -> bool {
        self.planes.iter().flatten().all(|value| value.is_finite())
            && self.camera_position.iter().all(|value| value.is_finite())
            && self.max_distance.is_finite()
            && self.near_depth.is_finite()
            && self.far_depth.is_finite()
            && self.far_depth > self.near_depth
            && (self.far_depth - self.near_depth).is_finite()
    }

    fn frustum_bytes(self) -> [u8; 96] {
        let mut bytes = [0; 96];
        for (index, plane) in self.planes.into_iter().enumerate() {
            for (component, value) in plane.into_iter().enumerate() {
                let offset = (index * 16) + (component * 4);
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn params_bytes(self, capacity: u32) -> [u8; 64] {
        let words = [
            capacity,
            0,
            u32::from(self.back_to_front),
            0,
            self.near_depth.to_bits(),
            self.far_depth.to_bits(),
            self.max_distance.to_bits(),
            0,
            self.camera_position[0].to_bits(),
            self.camera_position[1].to_bits(),
            self.camera_position[2].to_bits(),
            0,
            0,
            0,
            0,
            0,
        ];
        let mut bytes = [0; 64];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The shared renderer route accepted by the GPU stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleRendererRoute {
    /// Geometry family selected by the architecture renderer matrix.
    pub geometry: RendererGeometry,
    /// Sort strategy selected by the architecture sort/cull matrix.
    pub sort: SortStrategy,
}

impl ParticleRendererRoute {
    /// Builds a GPU route from the architecture plan and its geometry contract.
    pub fn from_plan(
        plan: &RendererPlan,
        geometry: RendererGeometry,
    ) -> Result<Self, ParticleRendererRouteError> {
        if plan.kind != geometry.kind() {
            return Err(ParticleRendererRouteError::GeometryKindMismatch {
                plan: plan.kind,
                geometry: geometry.kind(),
            });
        }
        if !matches!(
            plan.kind,
            RendererKind::Sprite | RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam
        ) {
            return Err(ParticleRendererRouteError::UnsupportedRenderer(plan.kind));
        }
        let sort = plan
            .routing
            .map(|routing| routing.sort)
            .ok_or(ParticleRendererRouteError::MissingGeometryRouting)?;
        Self::new(plan.kind, sort, geometry)
    }

    /// Creates a route directly from the frozen renderer contracts.
    pub fn new(
        kind: RendererKind,
        sort: SortStrategy,
        geometry: RendererGeometry,
    ) -> Result<Self, ParticleRendererRouteError> {
        if kind != geometry.kind() {
            return Err(ParticleRendererRouteError::GeometryKindMismatch {
                plan: kind,
                geometry: geometry.kind(),
            });
        }
        if !matches!(
            kind,
            RendererKind::Sprite | RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam
        ) {
            return Err(ParticleRendererRouteError::UnsupportedRenderer(kind));
        }
        if matches!(geometry, RendererGeometry::Mesh { index_count: 0, .. }) {
            return Err(ParticleRendererRouteError::ZeroMeshIndexCount);
        }
        Ok(Self { geometry, sort })
    }

    /// Returns the contract renderer kind.
    #[must_use]
    pub const fn kind(self) -> RendererKind {
        self.geometry.kind()
    }

    fn kind_code(self) -> u32 {
        match self.kind() {
            RendererKind::Sprite => 0,
            RendererKind::Mesh => 1,
            RendererKind::Ribbon => 2,
            RendererKind::Beam => 3,
            RendererKind::Light | RendererKind::Decal | RendererKind::Volume => unreachable!(),
        }
    }

    fn geometry_count(self) -> u32 {
        match self.geometry {
            RendererGeometry::Sprite { alive_count, .. }
            | RendererGeometry::Mesh { alive_count, .. } => alive_count,
            RendererGeometry::Ribbon { chain_count, .. }
            | RendererGeometry::Beam { chain_count, .. } => chain_count,
        }
    }

    fn index_count(self) -> u32 {
        match self.geometry {
            RendererGeometry::Sprite { .. } => 0,
            RendererGeometry::Mesh { index_count, .. } => index_count,
            RendererGeometry::Ribbon {
                segments_per_chain, ..
            }
            | RendererGeometry::Beam {
                segments_per_chain, ..
            } => segments_per_chain.saturating_mul(INDICES_PER_SEGMENT),
        }
    }

    fn first_instance(self) -> u32 {
        match self.geometry {
            RendererGeometry::Sprite { first_instance, .. }
            | RendererGeometry::Mesh { first_instance, .. }
            | RendererGeometry::Ribbon { first_instance, .. }
            | RendererGeometry::Beam { first_instance, .. } => first_instance,
        }
    }

    fn params_bytes(self) -> [u8; 48] {
        let words = [
            self.kind_code(),
            self.geometry_count(),
            self.index_count(),
            self.first_instance(),
            SPRITE_QUAD_VERTEX_COUNT,
            0,
            0,
            0,
        ];
        let mut bytes = [0; 48];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Invalid renderer route input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleRendererRouteError {
    /// The geometry kind and plan kind do not agree.
    GeometryKindMismatch {
        /// Kind selected by the renderer plan.
        plan: RendererKind,
        /// Kind represented by the geometry ABI.
        geometry: RendererKind,
    },
    /// A light/decal/volume plan cannot be sent to this surface route.
    UnsupportedRenderer(RendererKind),
    /// A plan without a geometry routing decision cannot produce a surface pass.
    MissingGeometryRouting,
    /// Mesh rendering needs a nonzero source index count.
    ZeroMeshIndexCount,
}

/// Resource/layout error returned before creating any GPU object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleRendererGpuError {
    /// The source pool has no elements.
    ZeroCapacity,
    /// The requested capacity cannot be rounded to a power of two.
    CapacityOverflow,
    /// An emitter work budget is zero or exceeds the shared input capacity.
    InvalidWorkCapacity,
    /// A required workgroup exceeds the device dispatch limit.
    DispatchTooLarge { workgroups: u32, limit: u32 },
    /// A required resource exceeds one device buffer limit.
    BufferTooLarge {
        /// Resource label used in the error.
        resource: &'static str,
        /// Requested byte count.
        requested: u64,
        /// Device limit.
        limit: u64,
    },
    /// A required compute binding is unavailable.
    BindingLimitTooSmall { requested: u32, limit: u32 },
    /// The architecture sort/cull contract rejected the layout.
    Layout(SortCullLayoutError),
    /// A route is not surface geometry supported by this stage.
    Route(ParticleRendererRouteError),
    /// Host input did not satisfy the finite ABI invariants.
    InvalidInput,
    /// Host view parameters did not satisfy the finite ABI invariants.
    InvalidView,
    /// A custom surface mesh was supplied to a non-Mesh route.
    MeshGeometryUnsupported,
    /// A custom surface mesh did not fit the route-owned buffers.
    MeshGeometryTooLarge,
    /// A custom surface mesh contained an invalid index or non-finite vertex.
    InvalidMeshGeometry,
    /// The route table is empty or exceeds the packed route-index ABI.
    InvalidRouteTable,
}

impl From<SortCullLayoutError> for ParticleRendererGpuError {
    fn from(value: SortCullLayoutError) -> Self {
        Self::Layout(value)
    }
}

impl From<ParticleRendererRouteError> for ParticleRendererGpuError {
    fn from(value: ParticleRendererRouteError) -> Self {
        Self::Route(value)
    }
}

impl ParticleAuxiliaryGpu {
    /// Creates a renderer-specific Light, Decal, or Volume dispatch stage.
    pub fn new<D: RenderDevice + ?Sized>(
        device: &D,
        kind: RendererKind,
        capacity: u32,
    ) -> Result<Self, ParticleRendererGpuError> {
        if capacity == 0 {
            return Err(ParticleRendererGpuError::ZeroCapacity);
        }
        if !matches!(
            kind,
            RendererKind::Light | RendererKind::Decal | RendererKind::Volume
        ) {
            return Err(ParticleRendererGpuError::Route(
                ParticleRendererRouteError::UnsupportedRenderer(kind),
            ));
        }
        let limits = device.capabilities().limits;
        let max_storage = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        for (resource, size) in [
            (
                "particle auxiliary input",
                u64::from(capacity) * PARTICLE_RENDER_INPUT_BYTES,
            ),
            (
                "particle auxiliary output",
                u64::from(capacity) * AUX_RECORD_BYTES,
            ),
        ] {
            if size > max_storage {
                return Err(ParticleRendererGpuError::BufferTooLarge {
                    resource,
                    requested: size,
                    limit: max_storage,
                });
            }
        }
        let workgroups = capacity.div_ceil(64);
        if workgroups > limits.max_compute_workgroups_per_dimension {
            return Err(ParticleRendererGpuError::DispatchTooLarge {
                workgroups,
                limit: limits.max_compute_workgroups_per_dimension,
            });
        }
        let input = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * PARTICLE_RENDER_INPUT_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let output = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * AUX_RECORD_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let params = device.create_buffer(&BufferDescriptor::new(
            AUX_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_auxiliary.wgsl"
        )));
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-auxiliary-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(2, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-auxiliary-pl")),
            bind_group_layouts: vec![layout],
            push_constant_ranges: Vec::new(),
        });
        let entry_point = match kind {
            RendererKind::Light => "emit_lights",
            RendererKind::Decal => "project_decals",
            RendererKind::Volume => "accumulate_volumes",
            _ => unreachable!(),
        };
        let pipeline = make_compute(device, pipeline_layout, shader, entry_point);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-auxiliary-bg")),
            layout: Some(layout),
            entries: vec![
                bind_buffer(0, input),
                bind_buffer(1, output),
                bind_buffer(2, params),
            ],
        });
        Ok(Self {
            kind,
            capacity,
            input,
            output,
            params,
            shader,
            layout,
            pipeline_layout,
            pipeline,
            bind_group,
        })
    }

    /// Uploads particle records and renderer-specific effect parameters.
    pub fn set_inputs(
        &self,
        queue: &impl RenderQueue,
        inputs: &[ParticleRenderInput],
        intensity_or_density: f32,
    ) -> Result<(), ParticleRendererGpuError> {
        if inputs.len() > self.capacity as usize || !intensity_or_density.is_finite() {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        let mut bytes = Vec::with_capacity(inputs.len() * PARTICLE_RENDER_INPUT_BYTES as usize);
        for input in inputs {
            if !input.validate() {
                return Err(ParticleRendererGpuError::InvalidInput);
            }
            input.append_bytes(&mut bytes);
        }
        queue.write_buffer(self.input, 0, &bytes);
        let mode = match self.kind {
            RendererKind::Light => 0.0_f32,
            RendererKind::Decal => 1.0,
            RendererKind::Volume => 2.0,
            _ => unreachable!(),
        };
        let params = [
            inputs.len() as u32,
            mode.to_bits(),
            intensity_or_density.to_bits(),
            0_u32,
        ];
        let mut packed = [0_u8; AUX_PARAMS_BYTES as usize];
        for (i, word) in params.into_iter().enumerate() {
            packed[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        queue.write_buffer(self.params, 0, &packed);
        Ok(())
    }

    /// Adds the selected renderer's concrete compute phase to a graph.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
    ) -> Result<ParticleAuxiliaryGraphOutputs, CompileError> {
        let input = import_buffer(
            graph,
            "particle-aux-input",
            self.input,
            u64::from(self.capacity) * PARTICLE_RENDER_INPUT_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        let output = import_buffer(
            graph,
            "particle-aux-output",
            self.output,
            u64::from(self.capacity) * AUX_RECORD_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let params = import_buffer(
            graph,
            "particle-aux-params",
            self.params,
            AUX_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let groups = self.capacity.div_ceil(64);
        let pipeline = self.pipeline;
        let bind_group = self.bind_group;
        let mut records = output;
        graph.add_compute_pass(
            format!("particle-renderer-{:?}-phase", self.kind),
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(input);
                builder.read_uniform(params);
                records = builder.write_storage_buffer(output);
                move |context| {
                    context
                        .set_compute_pipeline(pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(groups, 1, 1);
                }
            },
        );
        Ok(ParticleAuxiliaryGraphOutputs { records })
    }

    /// Builds a graph containing this renderer's concrete compute phase.
    #[must_use]
    pub fn graph_frame(&self) -> RenderGraph {
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        self.append_to_graph(&mut graph)
            .expect("particle auxiliary graph admission");
        graph
    }

    /// Executes this phase and submits its compute dispatch.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        self.graph_frame().execute(device, queue)
    }

    /// GPU output records written by the selected renderer phase.
    #[must_use]
    pub const fn output(&self) -> BufferId {
        self.output
    }

    /// Releases all GPU resources owned by this auxiliary phase.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        device.destroy_bind_group(self.bind_group);
        device.destroy_compute_pipeline(self.pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.layout);
        device.destroy_shader_module(self.shader);
        for buffer in [self.input, self.output, self.params] {
            device.destroy_buffer(buffer);
        }
    }
}

/// The GPU-backed sort/cull and renderer-argument stage.
pub struct ParticleRendererGpu {
    input_capacity: u32,
    capacity: u32,
    padded_capacity: u32,
    layout: SortCullBufferLayout,
    route: ParticleRendererRoute,
    input: BufferId,
    input_count: BufferId,
    owns_input: bool,
    emitter_source: Option<ParticleEmitterGpuSource>,
    route_table: BufferId,
    material: ParticleSurfaceMaterial,
    frustum: BufferId,
    camera: BufferId,
    // One stage belongs to one view. Capture its rectangle into each recorded
    // raster pass; a different queued view needs independent GPU work/uniforms.
    raster_viewport: Mutex<Option<[u32; 4]>>,
    static_upload: Once,
    params: BufferId,
    control: BufferId,
    sort_a: BufferId,
    sort_b: BufferId,
    visible_indices: BufferId,
    strip: Option<StripGpu>,
    // Final scan output for strips; a small unused binding for Sprite/Mesh.
    strip_vertices: BufferId,
    sprite_args: BufferId,
    indexed_args: BufferId,
    route_params: BufferId,
    cull_shader: ShaderModuleId,
    sort_shader: ShaderModuleId,
    compact_shader: ShaderModuleId,
    args_shader: ShaderModuleId,
    cull_layout: BindGroupLayoutId,
    cull_pipeline_layout: PipelineLayoutId,
    reset_pipeline: ComputePipelineId,
    seed_pipeline: ComputePipelineId,
    cull_pipeline: ComputePipelineId,
    cull_bind_group: BindGroupId,
    sort_layout: BindGroupLayoutId,
    sort_pipeline_layout: PipelineLayoutId,
    sort_pipeline: ComputePipelineId,
    sort_bind_groups: Vec<BindGroupId>,
    sort_steps: Vec<BufferId>,
    compact_layout: BindGroupLayoutId,
    compact_pipeline_layout: PipelineLayoutId,
    compact_pipeline: ComputePipelineId,
    compact_bind_group: BindGroupId,
    args_layout: BindGroupLayoutId,
    args_pipeline_layout: PipelineLayoutId,
    args_pipeline: ComputePipelineId,
    args_bind_group: BindGroupId,
    sorted_output: BufferId,
    surface_target: rhi::TextureId,
    surface_target_view: rhi::TextureViewId,
    surface_mesh_vertices: BufferId,
    surface_index_buffer: BufferId,
    surface_material: BufferId,
    surface_mesh_bytes: Vec<u8>,
    surface_index_bytes: Vec<u8>,
    surface_shader: ShaderModuleId,
    surface_bind_group_layout: BindGroupLayoutId,
    surface_pipeline_layout: PipelineLayoutId,
    depth_bind_group_layout: BindGroupLayoutId,
    hdr_surface_pipeline_layout: PipelineLayoutId,
    surface_pipeline: RenderPipelineId,
    hdr_surface_pipeline: RenderPipelineId,
    surface_bind_group: BindGroupId,
    record_plan_strip: Option<RendererRecordPlanStrip>,
    record_plan_surface_bind_group: BindGroupId,
    owns_record_plan_surface_bind_group: bool,
}

fn apply_viewport(
    context: &mut prism_render_graph::ExecuteContext<'_>,
    viewport: Option<[u32; 4]>,
) {
    if let Some([x, y, width, height]) = viewport {
        context
            .set_viewport(rhi::Viewport {
                x: x as f32,
                y: y as f32,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })
            .set_scissor(rhi::ScissorRect {
                x,
                y,
                width,
                height,
            });
    }
}

/// Borrowed GPU producer resources. Ranges are never uploaded by the renderer.
/// The table ABI is a 16-byte `(registered_count, 0, 0, 0)` header followed by
/// 16-byte `(active_start, particle_count, emitter_id, route_index)` records.
/// Record offsets address the compact input, in registry order, not pool IDs.
/// The input buffer uses PARTICLE_RENDER_INPUT_BYTES (48 bytes) per record;
/// strips require explicit RibbonId/birth metadata with topology_flags == 1.
/// All three buffers must outlive every renderer borrowing this source.
#[derive(Clone, Copy, Debug)]
pub struct ParticleEmitterGpuSource {
    pub input: BufferId,
    pub input_count: BufferId,
    pub route_table: BufferId,
    /// Allocated record capacity, excluding the header.
    pub route_capacity: u32,
    pub route_index: u32,
    pub emitter_id: u32,
}

/// Independently bound surface material for one renderer/emitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleSurfaceMaterial {
    pub scale: [f32; 2],
    pub softness: f32,
    /// Nonnegative fade distance in linear surface depth units; zero disables
    /// scene intersection fading. The standalone target has no scene depth.
    pub depth_fade_distance: f32,
    pub tint: [f32; 4],
    pub ribbon_width: f32,
    /// GPU strip geometry policy, consumed only by Ribbon and Beam routes.
    pub strip: ParticleStripSettings,
}

impl ParticleSurfaceMaterial {
    /// Default material used by the standalone route.
    #[must_use]
    pub fn for_route(route: ParticleRendererRoute) -> Self {
        let bytes = surface_material_bytes(route, ParticleCullView::unrestricted());
        let word = |offset| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        Self {
            scale: [word(0), word(4)],
            softness: word(8),
            depth_fade_distance: word(12),
            tint: [word(16), word(20), word(24), word(28)],
            ribbon_width: word(32),
            strip: ParticleStripSettings::default(),
        }
    }

    fn validate(self) -> bool {
        self.scale
            .iter()
            .chain(self.tint.iter())
            .copied()
            .chain([self.softness, self.ribbon_width, self.depth_fade_distance])
            .all(f32::is_finite)
            && self.scale.iter().all(|value| *value >= 0.0)
            && self.softness >= 0.0
            && self.softness <= 1.0
            && self.ribbon_width >= 0.0
            && self.depth_fade_distance >= 0.0
            && self.tint[3] >= 0.0
            && self.tint[3] <= 1.0
            && self.strip.validate()
    }
}

/// CPU registry metadata only; dynamic ranges/counts remain GPU authoritative.
#[derive(Clone, Copy, Debug)]
pub struct ParticleEmitterRendererConfig {
    pub emitter_id: u32,
    /// Bridge route ordinal, independent of this consumer's registry ordinal.
    /// Multiple renderers may share the same emitter/route pair.
    pub route_index: u32,
    pub renderer: ParticleRendererRoute,
    pub material: ParticleSurfaceMaterial,
}

/// One independent renderer per registry entry, sharing only bridge inputs.
pub struct ParticleEmitterRendererGpu {
    emitters: Vec<ParticleRendererGpu>,
}

/// Per-emitter compute outputs plus the threaded HDR color/depth versions.
pub struct ParticleEmitterRendererGraphOutputs {
    pub emitters: Vec<ParticleRendererGraphOutputs>,
    pub color: TextureHandle,
    pub depth: TextureHandle,
}

impl ParticleEmitterRendererGpu {
    /// Each entry explicitly identifies a bridge route; registry order is draw
    /// order, not route order. Physical inputs are borrowed without copying.
    /// This constructor reserves capacity work items per consumer;
    /// use new_with_budgets to avoid emitter_count * capacity work allocation.
    pub fn new<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        source: ParticleEmitterGpuSource,
        registry: &[ParticleEmitterRendererConfig],
    ) -> Result<Self, ParticleRendererGpuError> {
        Self::new_with_budgets(
            device,
            capacity,
            source,
            registry,
            &vec![capacity; registry.len()],
        )
    }

    /// Allocates local work buffers from guaranteed per-emitter maxima, in
    /// registry order, rather than duplicating the shared input capacity.
    /// Each budget must be nonzero and no greater than input_capacity. The
    /// producer must enforce these maxima; an over-budget GPU range fails
    /// closed, never truncates an arbitrary scatter prefix. Total work memory
    /// is proportional to the sum of power-of-two-rounded budgets, so callers
    /// should budget the sum explicitly. Fan-out consumers may legitimately
    /// reserve more total work than input_capacity without duplicating inputs.
    /// The bridge's 16-byte header/16-byte record ABI is unchanged.
    pub fn new_with_budgets<D: RenderDevice + ?Sized>(
        device: &D,
        input_capacity: u32,
        source: ParticleEmitterGpuSource,
        registry: &[ParticleEmitterRendererConfig],
        work_capacities: &[u32],
    ) -> Result<Self, ParticleRendererGpuError> {
        Self::new_with_budgets_and_tessellation(
            device,
            input_capacity,
            source,
            registry,
            work_capacities,
            &vec![None; registry.len()],
        )
    }

    /// Registry-ordered optional curve policies. Every policy and expanded
    /// device budget is admitted before allocating the first consumer, so an
    /// invalid later consumer cannot leave earlier stage allocations behind.
    /// None preserves straight strips; Some is valid only for Ribbon/Beam.
    pub fn new_with_budgets_and_tessellation<D: RenderDevice + ?Sized>(
        device: &D,
        input_capacity: u32,
        source: ParticleEmitterGpuSource,
        registry: &[ParticleEmitterRendererConfig],
        work_capacities: &[u32],
        tessellation: &[Option<ParticleStripTessellation>],
    ) -> Result<Self, ParticleRendererGpuError> {
        if registry.len() != work_capacities.len() || registry.len() != tessellation.len() {
            return Err(ParticleRendererGpuError::InvalidWorkCapacity);
        }
        if work_capacities
            .iter()
            .any(|&budget| budget == 0 || budget > input_capacity)
        {
            return Err(ParticleRendererGpuError::InvalidWorkCapacity);
        }
        let mut emitter_routes = BTreeMap::new();
        let mut route_emitters = BTreeMap::new();
        for config in registry {
            if config.route_index >= source.route_capacity
                || !config.material.validate()
                || emitter_routes
                    .insert(config.emitter_id, config.route_index)
                    .is_some_and(|prior| prior != config.route_index)
                || route_emitters
                    .insert(config.route_index, config.emitter_id)
                    .is_some_and(|prior| prior != config.emitter_id)
            {
                return Err(ParticleRendererGpuError::InvalidRouteTable);
            }
        }
        for ((config, &capacity), settings) in
            registry.iter().zip(work_capacities).zip(tessellation)
        {
            if let Some(settings) = settings {
                settings.admit(device, capacity, config.renderer)?;
            }
        }
        let mut emitters = Vec::with_capacity(registry.len());
        for (index, config) in registry.iter().enumerate() {
            let stage = ParticleRendererGpu::new_inner(
                device,
                input_capacity,
                work_capacities[index],
                config.renderer,
                Some(ParticleEmitterGpuSource {
                    route_index: config.route_index,
                    emitter_id: config.emitter_id,
                    ..source
                }),
                tessellation[index],
            );
            match stage {
                Ok(mut stage) => {
                    // Every material was validated before allocating stages.
                    stage.material = config.material;
                    emitters.push(stage);
                }
                Err(error) => {
                    for stage in &emitters {
                        stage.destroy(device);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self { emitters })
    }

    /// Registry-ordered stages, including per-emitter indirect/visibility IDs.
    pub fn emitters(&self) -> &[ParticleRendererGpu] {
        &self.emitters
    }

    /// Allows independent mesh geometry and material updates per emitter.
    pub fn emitters_mut(&mut self) -> &mut [ParticleRendererGpu] {
        &mut self.emitters
    }

    pub fn set_view(
        &self,
        queue: &impl RenderQueue,
        view: ParticleCullView,
    ) -> Result<(), ParticleRendererGpuError> {
        if !view.validate() {
            return Err(ParticleRendererGpuError::InvalidView);
        }
        for stage in &self.emitters {
            stage.set_view(queue, view)?;
        }
        Ok(())
    }

    /// Appends independent compute and indirect draws, threading the caller's
    /// initialized HDR SSA through every material. No aggregate/global draw is
    /// emitted. The depth SSA is read-only and unchanged for every emitter.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
        input: BufferHandle,
        input_count: BufferHandle,
        route_table: BufferHandle,
        mut color: TextureHandle,
        mut depth: TextureHandle,
    ) -> Result<ParticleEmitterRendererGraphOutputs, CompileError> {
        let mut outputs = Vec::with_capacity(self.emitters.len());
        for stage in &self.emitters {
            let output = stage.append_compute_to_graph_with_emitter_routes(
                graph,
                input,
                input_count,
                route_table,
            )?;
            (color, depth) = stage.append_raster_to_graph(graph, &output, color, depth)?;
            outputs.push(output);
        }
        Ok(ParticleEmitterRendererGraphOutputs {
            emitters: outputs,
            color,
            depth,
        })
    }

    /// Releases only owned per-emitter work/geometry/material resources.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        for stage in &self.emitters {
            stage.destroy(device);
        }
    }
}

/// Executable non-surface particle phase for Light, Decal, or Volume.
///
/// The stage owns its input/output records and dispatches a renderer-specific
/// WGSL entry point; it is deliberately separate from indexed surface draws.
pub struct ParticleAuxiliaryGpu {
    kind: RendererKind,
    capacity: u32,
    input: BufferId,
    output: BufferId,
    params: BufferId,
    shader: ShaderModuleId,
    layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: ComputePipelineId,
    bind_group: BindGroupId,
}

/// Graph outputs from a real renderer-specific auxiliary dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ParticleAuxiliaryGraphOutputs {
    /// The transformed per-particle light/decal/volume records.
    pub records: BufferHandle,
}

/// SSA versions produced by the renderer-matrix compute passes.
#[derive(Clone, Copy, Debug)]
pub struct ParticleRendererGraphOutputs {
    /// Source particle records consumed by the raster and auxiliary paths.
    pub input: BufferHandle,
    /// Active source-record count, preserving a GPU producer's SSA dependency.
    pub input_count: BufferHandle,
    /// Compact source-record offsets consumed by the vertex shader, not pool IDs.
    pub visible_indices: BufferHandle,
    /// GPU arc-length UVs and discontinuity heads, in visible-rank order.
    /// Present only for Ribbon/Beam; raster must consume this exact SSA version.
    pub strip_vertices: Option<BufferHandle>,
    /// Compact GPU-generated curve subedges, only for tessellated strips.
    pub strip_geometry: Option<BufferHandle>,
    /// Sprite non-indexed indirect arguments.
    pub sprite_args: BufferHandle,
    /// Mesh/Ribbon/Beam indexed indirect arguments.
    pub indexed_args: BufferHandle,
    /// Cull counter and route control values.
    pub control: BufferHandle,
}

impl ParticleRendererGpu {
    /// Creates all persistent buffers, pipelines and bind groups for one route.
    pub fn new<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        route: ParticleRendererRoute,
    ) -> Result<Self, ParticleRendererGpuError> {
        Self::new_inner(device, capacity, capacity, route, None, None)
    }

    /// Opts Ribbon/Beam into GPU screen-space curve tessellation. All expanded
    /// storage is admitted before allocation. The allocation cap is immutable;
    /// subdivision/draw counts are recomputed on the GPU for every camera.
    pub fn new_tessellated<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        route: ParticleRendererRoute,
        settings: ParticleStripTessellation,
    ) -> Result<Self, ParticleRendererGpuError> {
        Self::new_inner(device, capacity, capacity, route, None, Some(settings))
    }

    /// Creates an independent cull/sort/indirect/material consumer for one
    /// registered emitter. Only input/count/table are shared; all work buffers
    /// and geometry/material bindings belong to this stage. GPU counts replace
    /// the route's CPU alive/chain count. Geometry topology remains CPU metadata.
    /// Ribbon/Beam use explicit RibbonId/birth topology, independently of
    /// depth-sort policy, because scatter and reused pool-slot order are invalid.
    /// Emitter-local indirect draws are zero-based even if geometry specifies a
    /// logical first_instance: no shared instance buffer is addressed, and a
    /// nonzero GPU base would require the optional INDIRECT_FIRST_INSTANCE feature.
    pub fn new_emitter<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        route: ParticleRendererRoute,
        source: ParticleEmitterGpuSource,
    ) -> Result<Self, ParticleRendererGpuError> {
        Self::new_emitter_with_budget(device, capacity, capacity, route, source)
    }

    /// Like new_emitter, with an independently bounded local visibility/sort
    /// budget. Source offsets always address the full input_capacity buffer;
    /// range counts greater than work_capacity are rejected on the GPU.
    pub fn new_emitter_with_budget<D: RenderDevice + ?Sized>(
        device: &D,
        input_capacity: u32,
        work_capacity: u32,
        route: ParticleRendererRoute,
        source: ParticleEmitterGpuSource,
    ) -> Result<Self, ParticleRendererGpuError> {
        if source.route_index >= source.route_capacity {
            return Err(ParticleRendererGpuError::InvalidRouteTable);
        }
        Self::new_inner(
            device,
            input_capacity,
            work_capacity,
            route,
            Some(source),
            None,
        )
    }

    /// Emitter-local production variant of new_tessellated. Borrowed producer
    /// inputs stay resident; geometry counts use this consumer's work budget.
    pub fn new_emitter_tessellated_with_budget<D: RenderDevice + ?Sized>(
        device: &D,
        input_capacity: u32,
        work_capacity: u32,
        route: ParticleRendererRoute,
        source: ParticleEmitterGpuSource,
        settings: ParticleStripTessellation,
    ) -> Result<Self, ParticleRendererGpuError> {
        if source.route_index >= source.route_capacity {
            return Err(ParticleRendererGpuError::InvalidRouteTable);
        }
        Self::new_inner(
            device,
            input_capacity,
            work_capacity,
            route,
            Some(source),
            Some(settings),
        )
    }

    fn new_inner<D: RenderDevice + ?Sized>(
        device: &D,
        input_capacity: u32,
        capacity: u32,
        route: ParticleRendererRoute,
        emitter_source: Option<ParticleEmitterGpuSource>,
        tessellation: Option<ParticleStripTessellation>,
    ) -> Result<Self, ParticleRendererGpuError> {
        if input_capacity == 0 {
            return Err(ParticleRendererGpuError::ZeroCapacity);
        }
        if capacity == 0 || capacity > input_capacity {
            return Err(ParticleRendererGpuError::InvalidWorkCapacity);
        }
        if let Some(settings) = tessellation {
            settings.admit(device, capacity, route)?;
        }
        let caps = device.capabilities();
        if caps.limits.max_bind_groups < 2 || caps.limits.max_bindings_per_bind_group < 8 {
            return Err(ParticleRendererGpuError::BindingLimitTooSmall {
                requested: 8,
                limit: caps.limits.max_bindings_per_bind_group,
            });
        }
        if caps.limits.max_compute_invocations_per_workgroup < 64 {
            return Err(ParticleRendererGpuError::BindingLimitTooSmall {
                requested: 64,
                limit: caps.limits.max_compute_invocations_per_workgroup,
            });
        }
        // Six vertex storage bindings (including strip prefixes), and six
        // storage bindings in the strip compute layout. Admit before allocating.
        if caps.limits.max_storage_buffers_per_shader_stage < 6 {
            return Err(ParticleRendererGpuError::BindingLimitTooSmall {
                requested: 6,
                limit: caps.limits.max_storage_buffers_per_shader_stage,
            });
        }
        let padded_capacity = capacity
            .checked_next_power_of_two()
            .ok_or(ParticleRendererGpuError::CapacityOverflow)?;
        let max_storage = caps
            .limits
            .max_storage_buffer_binding_size
            .min(caps.limits.max_buffer_size);
        let layout = sort_cull_buffer_layout(capacity, 256, 64, max_storage)?;
        let workgroups = padded_capacity.div_ceil(64);
        if workgroups > caps.limits.max_compute_workgroups_per_dimension {
            return Err(ParticleRendererGpuError::DispatchTooLarge {
                workgroups,
                limit: caps.limits.max_compute_workgroups_per_dimension,
            });
        }

        let check_buffer = |resource: &'static str, size: u64| {
            if size > max_storage {
                Err(ParticleRendererGpuError::BufferTooLarge {
                    resource,
                    requested: size,
                    limit: max_storage,
                })
            } else {
                Ok(())
            }
        };
        let input_bytes = u64::from(input_capacity) * PARTICLE_RENDER_INPUT_BYTES;
        let sort_bytes = u64::from(padded_capacity) * 8;
        let visible_bytes = u64::from(capacity) * 4;
        let stable_strip = matches!(route.kind(), RendererKind::Ribbon | RendererKind::Beam);
        let strip_bytes = u64::from(if stable_strip { capacity } else { 1 }) * STRIP_VERTEX_BYTES;
        check_buffer("particle render input", input_bytes)?;
        check_buffer("particle render input count", 4)?;
        check_buffer("particle sort items", sort_bytes)?;
        check_buffer("particle visible indices", visible_bytes)?;
        check_buffer("particle strip vertices", strip_bytes)?;
        check_buffer("particle control", 16)?;
        check_buffer("particle renderer arguments", 40)?;
        check_buffer(
            "particle emitter routes",
            emitter_source.map_or(16, |source| 16 + u64::from(source.route_capacity) * 16),
        )?;

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let input = emitter_source.map_or_else(
            || {
                device.create_buffer(&BufferDescriptor::new(
                    input_bytes,
                    storage | BufferUsages::COPY_DST,
                ))
            },
            |source| source.input,
        );
        let input_count = emitter_source.map_or_else(
            || device.create_buffer(&BufferDescriptor::new(4, storage | BufferUsages::COPY_DST)),
            |source| source.input_count,
        );
        let route_table = emitter_source.map_or_else(
            || device.create_buffer(&BufferDescriptor::new(16, storage | BufferUsages::COPY_DST)),
            |source| source.route_table,
        );
        let frustum = device.create_buffer(&BufferDescriptor::new(
            96,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let params = device.create_buffer(&BufferDescriptor::new(
            64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let camera = device.create_buffer(&BufferDescriptor::new(
            CAMERA_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let control = device.create_buffer(&BufferDescriptor::new(16, storage));
        let sort_a = device.create_buffer(&BufferDescriptor::new(sort_bytes, storage));
        let sort_b = device.create_buffer(&BufferDescriptor::new(sort_bytes, storage));
        let visible_indices = device.create_buffer(&BufferDescriptor::new(visible_bytes, storage));
        let sprite_args = device.create_buffer(&BufferDescriptor::new(
            16,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        ));
        let indexed_args = device.create_buffer(&BufferDescriptor::new(
            20,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        ));
        let mut strip = stable_strip.then(|| {
            StripGpu::new(
                device,
                capacity,
                input,
                input_count,
                visible_indices,
                control,
                tessellation.map(|settings| {
                    TessellationGpu::new(
                        device,
                        capacity,
                        route,
                        settings,
                        emitter_source.is_some(),
                    )
                }),
            )
        });
        let strip_vertices = strip.as_ref().map_or_else(
            || device.create_buffer(&BufferDescriptor::new(STRIP_VERTEX_BYTES, storage)),
            StripGpu::output,
        );
        if let Some(tessellation) = strip.as_mut().and_then(|strip| strip.tessellation.as_mut()) {
            tessellation.bind(
                device,
                input,
                visible_indices,
                control,
                strip_vertices,
                camera,
                indexed_args,
            );
        }
        let record_plan_strip = stable_strip.then(|| {
            RendererRecordPlanStrip::new(
                device,
                capacity,
                route,
                input,
                input_count,
                visible_indices,
                control,
                camera,
                indexed_args,
                tessellation,
                emitter_source.is_some(),
            )
        });
        let route_params = device.create_buffer(&BufferDescriptor::new(
            48,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let cull_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_cull.wgsl"
        )));
        let sort_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_sort.wgsl"
        )));
        let compact_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            include_str!("../shaders/particle_renderer_compact.wgsl"),
        ));
        let args_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_args.wgsl"
        )));

        let cull_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-cull-bgl")),
            entries: vec![
                buffer_entry(0, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    3,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(4, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    5,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    6,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(7, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let cull_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-cull-pl")),
            bind_group_layouts: vec![cull_layout],
            push_constant_ranges: Vec::new(),
        });
        let reset_pipeline = make_compute(device, cull_pipeline_layout, cull_shader, "reset");
        let seed_pipeline = make_compute(device, cull_pipeline_layout, cull_shader, "seed");
        let cull_pipeline = make_compute(device, cull_pipeline_layout, cull_shader, "cull");
        let cull_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-cull-bg")),
            layout: Some(cull_layout),
            entries: vec![
                bind_buffer(0, frustum),
                bind_buffer(1, input),
                bind_buffer(2, control),
                bind_buffer(3, sort_a),
                bind_buffer(4, params),
                bind_buffer(5, input_count),
                bind_buffer(6, route_table),
                bind_buffer(7, camera),
            ],
        });

        let sort_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-sort-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(2, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    3,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
            ],
        });
        let sort_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-sort-pl")),
            bind_group_layouts: vec![sort_layout],
            push_constant_ranges: Vec::new(),
        });
        let sort_pipeline = make_compute(device, sort_pipeline_layout, sort_shader, "sort_step");
        let needs_sort = route.sort.is_explicit_sort() || stable_strip;
        let mut sort_bind_groups = Vec::new();
        let mut sort_steps = Vec::new();
        let mut k = 2u32;
        while needs_sort && k <= padded_capacity {
            let mut j = k / 2;
            while j != 0 {
                let step = device.create_buffer(&BufferDescriptor::new(
                    16,
                    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                ));
                let source = if sort_bind_groups.len() % 2 == 0 {
                    sort_a
                } else {
                    sort_b
                };
                let destination = if sort_bind_groups.len() % 2 == 0 {
                    sort_b
                } else {
                    sort_a
                };
                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some(String::from("particle-renderer-sort-bg")),
                    layout: Some(sort_layout),
                    entries: vec![
                        bind_buffer(0, source),
                        bind_buffer(1, destination),
                        bind_buffer(2, step),
                        bind_buffer(3, input),
                    ],
                });
                sort_bind_groups.push(bind_group);
                sort_steps.push(step);
                if j == 1 {
                    break;
                }
                j /= 2;
            }
            if k > padded_capacity / 2 {
                break;
            }
            k *= 2;
        }
        let sorted_output = if sort_bind_groups.len() % 2 == 0 {
            sort_a
        } else {
            sort_b
        };

        let compact_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-compact-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(2, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    3,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
            ],
        });
        let compact_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-compact-pl")),
            bind_group_layouts: vec![compact_layout],
            push_constant_ranges: Vec::new(),
        });
        let compact_pipeline =
            make_compute(device, compact_pipeline_layout, compact_shader, "compact");
        let compact_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-compact-bg")),
            layout: Some(compact_layout),
            entries: vec![
                bind_buffer(0, sorted_output),
                bind_buffer(1, visible_indices),
                bind_buffer(2, params),
                bind_buffer(3, control),
            ],
        });

        let args_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-args-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(3, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let args_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-args-pl")),
            bind_group_layouts: vec![args_layout],
            push_constant_ranges: Vec::new(),
        });
        let args_pipeline = make_compute(device, args_pipeline_layout, args_shader, "emit_args");
        let args_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-args-bg")),
            layout: Some(args_layout),
            entries: vec![
                bind_buffer(0, control),
                bind_buffer(1, sprite_args),
                bind_buffer(2, indexed_args),
                bind_buffer(3, route_params),
            ],
        });

        let surface_target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(RENDERER_MATRIX_TARGET_SIZE, RENDERER_MATRIX_TARGET_SIZE),
            TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let surface_target_view =
            device.create_texture_view(surface_target, &TextureViewDescriptor::default());
        let surface_mesh_vertices = device.create_buffer(&BufferDescriptor::new(
            u64::from(self_index_capacity(route).max(3)) * SURFACE_MESH_VERTEX_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let surface_index_buffer = device.create_buffer(&BufferDescriptor::new(
            u64::from(surface_index_count(route).max(1)) * 4,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
        ));
        let surface_material = device.create_buffer(&BufferDescriptor::new(
            SURFACE_MATERIAL_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let surface_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            include_str!("../shaders/particle_renderer_surface.wgsl"),
        ));
        let surface_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(String::from("particle-renderer-surface-bgl")),
                entries: vec![
                    buffer_entry(
                        0,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
                    buffer_entry(
                        1,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
                    buffer_entry(
                        2,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
                    buffer_entry(
                        3,
                        BufferBindingType::Uniform,
                        ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ),
                    buffer_entry(
                        4,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
                    buffer_entry(
                        5,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
                    buffer_entry(
                        6,
                        BufferBindingType::Uniform,
                        ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ),
                    buffer_entry(
                        7,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
                ],
            });
        let surface_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-surface-pl")),
            bind_group_layouts: vec![surface_bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let surface_pipeline = make_surface_pipeline(
            device,
            surface_pipeline_layout,
            surface_shader,
            route.kind(),
        );
        let depth_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-depth-bgl")),
            entries: vec![rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::Depth,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let hdr_surface_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(String::from("particle-renderer-hdr-surface-pl")),
                bind_group_layouts: vec![surface_bind_group_layout, depth_bind_group_layout],
                push_constant_ranges: Vec::new(),
            });
        let hdr_surface_pipeline = make_surface_pipeline_for_target(
            device,
            hdr_surface_pipeline_layout,
            surface_shader,
            route.kind(),
            TextureFormat::Rgba16Float,
            Some(DepthStencilState::read_only(
                TextureFormat::Depth32Float,
                CompareFunction::GreaterEqual,
            )),
            "fs_depth",
        );
        let surface_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-surface-bg")),
            layout: Some(surface_bind_group_layout),
            entries: vec![
                bind_buffer(0, input),
                bind_buffer(1, visible_indices),
                bind_buffer(
                    2,
                    strip
                        .as_ref()
                        .and_then(|strip| strip.tessellation.as_ref())
                        .map_or(surface_mesh_vertices, |tessellation| tessellation.geometry),
                ),
                bind_buffer(3, surface_material),
                bind_buffer(4, input_count),
                bind_buffer(5, control),
                bind_buffer(6, camera),
                bind_buffer(7, strip_vertices),
            ],
        });
        let (record_plan_surface_bind_group, owns_record_plan_surface_bind_group) =
            if let Some(record_strip) = &record_plan_strip {
                let geometry = record_strip
                    .tessellation
                    .as_ref()
                    .map_or(record_strip.output(), |tessellation| tessellation.geometry);
                (
                    device.create_bind_group(&BindGroupDescriptor {
                        label: Some(String::from("particle-record-plan-surface-bg")),
                        layout: Some(surface_bind_group_layout),
                        entries: vec![
                            bind_buffer(0, input),
                            bind_buffer(1, visible_indices),
                            bind_buffer(2, geometry),
                            bind_buffer(3, surface_material),
                            bind_buffer(4, input_count),
                            bind_buffer(5, control),
                            bind_buffer(6, camera),
                            bind_buffer(7, record_strip.output()),
                        ],
                    }),
                    true,
                )
            } else {
                (surface_bind_group, false)
            };

        Ok(Self {
            input_capacity,
            capacity,
            padded_capacity,
            layout,
            route,
            input,
            input_count,
            owns_input: emitter_source.is_none(),
            emitter_source,
            route_table,
            material: ParticleSurfaceMaterial::for_route(route),
            frustum,
            camera,
            raster_viewport: Mutex::new(None),
            static_upload: Once::new(),
            params,
            control,
            sort_a,
            sort_b,
            visible_indices,
            strip,
            strip_vertices,
            sprite_args,
            indexed_args,
            route_params,
            cull_shader,
            sort_shader,
            compact_shader,
            args_shader,
            cull_layout,
            cull_pipeline_layout,
            reset_pipeline,
            seed_pipeline,
            cull_pipeline,
            cull_bind_group,
            sort_layout,
            sort_pipeline_layout,
            sort_pipeline,
            sort_bind_groups,
            sort_steps,
            compact_layout,
            compact_pipeline_layout,
            compact_pipeline,
            compact_bind_group,
            args_layout,
            args_pipeline_layout,
            args_pipeline,
            args_bind_group,
            sorted_output,
            surface_target,
            surface_target_view,
            surface_mesh_vertices,
            surface_index_buffer,
            surface_material,
            surface_mesh_bytes: surface_mesh_vertex_bytes(route),
            surface_index_bytes: surface_index_bytes(route),
            surface_shader,
            surface_bind_group_layout,
            surface_pipeline_layout,
            depth_bind_group_layout,
            hdr_surface_pipeline_layout,
            surface_pipeline,
            hdr_surface_pipeline,
            surface_bind_group,
            record_plan_strip,
            record_plan_surface_bind_group,
            owns_record_plan_surface_bind_group,
        })
    }

    /// Uploads the source records and the frame's cull/sort constants.
    pub fn set_inputs(
        &self,
        queue: &impl RenderQueue,
        inputs: &[ParticleRenderInput],
        view: ParticleCullView,
    ) -> Result<(), ParticleRendererGpuError> {
        if !self.owns_input {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        if inputs.len() > self.input_capacity as usize {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        if !view.validate() {
            return Err(ParticleRendererGpuError::InvalidView);
        }
        if !inputs.iter().copied().all(ParticleRenderInput::validate) {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        let mut bytes = Vec::with_capacity(inputs.len() * PARTICLE_RENDER_INPUT_BYTES as usize);
        for input in inputs {
            input.append_bytes(&mut bytes);
        }
        queue.write_buffer(self.input, 0, &bytes);
        queue.write_buffer(self.input_count, 0, &(inputs.len() as u32).to_le_bytes());
        self.set_view(queue, view)
    }

    /// Uploads only the view and route state while leaving the source input
    /// buffer untouched. A GPU simulation bridge uses this entry point so the
    /// renderer does not overwrite the bridge-authored input with a CPU
    /// snapshot upload.
    pub fn set_view(
        &self,
        queue: &impl RenderQueue,
        view: ParticleCullView,
    ) -> Result<(), ParticleRendererGpuError> {
        if !view.validate() {
            return Err(ParticleRendererGpuError::InvalidView);
        }
        let mut raster_viewport = self
            .raster_viewport
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.initialize_static(queue);
        queue.write_buffer(self.frustum, 0, &view.frustum_bytes());
        queue.write_buffer(self.camera, 0, &ParticleCamera::bytes(view.camera));
        let mut params = view.params_bytes(self.capacity);
        params[4..8].copy_from_slice(&self.input_capacity.to_le_bytes());
        params[60..64].copy_from_slice(&u32::from(self.stable_strip()).to_le_bytes());
        if let Some(source) = self.emitter_source {
            for (index, word) in [
                1,
                source.route_index,
                source.emitter_id,
                u32::from(self.stable_strip()),
            ]
            .into_iter()
            .enumerate()
            {
                params[48 + index * 4..52 + index * 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        queue.write_buffer(self.params, 0, &params);
        queue.write_buffer(self.surface_material, 0, &self.material_bytes(view));
        if let Some(strip) = &self.strip {
            strip.upload(queue, self.material);
        }
        if let Some(strip) = &self.record_plan_strip {
            strip.upload(queue, self.material);
        }
        *raster_viewport = view.camera.map(ParticleCamera::viewport);
        Ok(())
    }

    // Static geometry, route and sort steps are independent of the camera.
    // Once also serializes first publication across shared callers.
    fn initialize_static(&self, queue: &impl RenderQueue) {
        self.static_upload.call_once(|| {
            if self.owns_input {
                queue.write_buffer(self.route_table, 0, &[0; 16]);
            }
            queue.write_buffer(self.route_params, 0, &self.route_params_bytes());
            queue.write_buffer(self.surface_mesh_vertices, 0, &self.surface_mesh_bytes);
            queue.write_buffer(self.surface_index_buffer, 0, &self.surface_index_bytes);
            if let Some(strip) = &self.strip {
                strip.initialize_static(queue);
            }
            if let Some(strip) = &self.record_plan_strip {
                strip.initialize_static(queue);
            }
            for (index, step) in self.sort_steps.iter().copied().enumerate() {
                let (k, j) = sort_step_at(self.padded_capacity, index);
                let mut bytes = [0; 16];
                bytes[..4].copy_from_slice(&k.to_le_bytes());
                bytes[4..8].copy_from_slice(&j.to_le_bytes());
                bytes[8..12].copy_from_slice(&u32::from(self.stable_strip()).to_le_bytes());
                queue.write_buffer(step, 0, &bytes);
            }
        });
    }

    /// Changes this emitter's material. The next set_view upload publishes it.
    pub fn set_material(
        &mut self,
        material: ParticleSurfaceMaterial,
    ) -> Result<(), ParticleRendererGpuError> {
        if !material.validate() {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        self.material = material;
        Ok(())
    }

    /// Configures GPU Ribbon/Beam arc-length UVs and discontinuity policy.
    /// The next set_view/set_inputs publishes the settings. Other renderer
    /// kinds reject this call; no input ABI or CPU topology gathering is needed.
    pub fn set_strip_settings(
        &mut self,
        settings: ParticleStripSettings,
    ) -> Result<(), ParticleRendererGpuError> {
        if !settings.validate() {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        if self.strip.is_none() {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        self.material.strip = settings;
        Ok(())
    }

    fn material_bytes(&self, view: ParticleCullView) -> [u8; 64] {
        let mut bytes = surface_material_bytes(self.route, view);
        bytes[48..52].copy_from_slice(&self.draw_first_instance().to_le_bytes());
        let tessellated = self
            .strip
            .as_ref()
            .is_some_and(|strip| strip.tessellation.is_some());
        bytes[60..64].copy_from_slice(
            &(u32::from(self.stable_strip()) | (u32::from(tessellated) << 1)).to_le_bytes(),
        );
        for (offset, values) in [
            (0, &self.material.scale[..]),
            (8, core::slice::from_ref(&self.material.softness)),
            (
                12,
                core::slice::from_ref(&self.material.depth_fade_distance),
            ),
            (16, &self.material.tint[..]),
            (32, core::slice::from_ref(&self.material.ribbon_width)),
        ] {
            for (index, value) in values.iter().enumerate() {
                bytes[offset + index * 4..offset + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn draw_first_instance(&self) -> u32 {
        if self.emitter_source.is_some() {
            0
        } else {
            self.route.first_instance()
        }
    }

    fn route_params_bytes(&self) -> [u8; 48] {
        let mut bytes = self.route.params_bytes();
        if self.emitter_source.is_some() {
            bytes[4..8].copy_from_slice(&self.capacity.to_le_bytes());
        }
        // Match the material's instance-index subtraction for both raster paths.
        bytes[12..16].copy_from_slice(&self.draw_first_instance().to_le_bytes());
        bytes
    }

    fn stable_strip(&self) -> bool {
        matches!(self.route.kind(), RendererKind::Ribbon | RendererKind::Beam)
    }

    /// Replaces the source geometry consumed by the Mesh surface route.
    ///
    /// The route's index count is fixed when the stage is created. Geometry
    /// replacement therefore updates existing buffers and does not rebuild
    /// the renderer graph resources.
    pub fn set_mesh_geometry(
        &mut self,
        queue: &impl RenderQueue,
        vertices: &[ParticleMeshVertex],
        indices: &[u32],
    ) -> Result<(), ParticleRendererGpuError> {
        if self.route.kind() != RendererKind::Mesh {
            return Err(ParticleRendererGpuError::MeshGeometryUnsupported);
        }
        let expected_indices = surface_index_count(self.route) as usize;
        let vertex_capacity = self_index_capacity(self.route) as usize;
        if indices.len() != expected_indices || vertices.len() > vertex_capacity {
            return Err(ParticleRendererGpuError::MeshGeometryTooLarge);
        }
        if vertices.is_empty()
            || vertices.iter().copied().any(|vertex| !vertex.validate())
            || indices
                .iter()
                .any(|index| *index as usize >= vertices.len())
        {
            return Err(ParticleRendererGpuError::InvalidMeshGeometry);
        }

        let mut vertex_bytes =
            Vec::with_capacity(vertex_capacity * SURFACE_MESH_VERTEX_STRIDE as usize);
        for vertex in vertices {
            for value in [
                vertex.position[0],
                vertex.position[1],
                vertex.position[2],
                0.0,
                vertex.uv[0],
                vertex.uv[1],
                0.0,
                0.0,
            ] {
                vertex_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        vertex_bytes.resize(vertex_capacity * SURFACE_MESH_VERTEX_STRIDE as usize, 0);

        let mut index_bytes = Vec::with_capacity(indices.len() * 4);
        for index in indices {
            index_bytes.extend_from_slice(&index.to_le_bytes());
        }
        self.initialize_static(queue);
        queue.write_buffer(self.surface_mesh_vertices, 0, &vertex_bytes);
        queue.write_buffer(self.surface_index_buffer, 0, &index_bytes);
        self.surface_mesh_bytes = vertex_bytes;
        self.surface_index_bytes = index_bytes;
        Ok(())
    }

    /// Appends the reset, cull, explicit sort, compaction, and renderer-args
    /// passes to an existing graph.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        let input = import_buffer(
            graph,
            "particle-render-input",
            self.input,
            u64::from(self.input_capacity) * PARTICLE_RENDER_INPUT_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        self.append_to_graph_with_input_handle(graph, input, None, None, true)
    }

    /// Appends the renderer graph using a bridge-authored renderer-input
    /// handle. The handle must resolve to this stage's [`Self::input_buffer`]
    /// physical buffer; the method exists to preserve the graph dependency
    /// from a GPU simulation bridge without introducing a CPU upload.
    pub fn append_to_graph_with_input(
        &self,
        graph: &mut RenderGraph,
        input: BufferHandle,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        self.append_to_graph_with_input_handle(graph, input, None, None, true)
    }

    /// Appends only the renderer matrix's GPU cull, sort, compact and indirect
    /// argument passes. No surface target or raster pass is declared. This is
    /// the embedding entry point used by RenderApp when the host owns the HDR
    /// frame target and records its own material/raster consumer.
    pub fn append_compute_to_graph_with_input(
        &self,
        graph: &mut RenderGraph,
        input: BufferHandle,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        self.append_to_graph_with_input_handle(graph, input, None, None, false)
    }

    /// Appends renderer compute passes while consuming a GPU-written input
    /// count. This is the GPU-authoritative path used after simulation bridge
    /// gather; the cull pass reads the count before scanning the input buffer.
    /// Both handles must resolve to this stage's physical input and input-count
    /// buffers: its persistent bind groups reference those exact buffers.
    pub fn append_compute_to_graph_with_input_count(
        &self,
        graph: &mut RenderGraph,
        input: BufferHandle,
        input_count: BufferHandle,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        self.append_to_graph_with_input_handle(graph, input, Some(input_count), None, false)
    }

    /// Consumes the bridge's final input/count/route-table SSA versions. Reuse
    /// these same handles for every emitter; do not reimport producer buffers.
    /// Handles must refer to the physical buffers supplied to new_emitter.
    pub fn append_compute_to_graph_with_emitter_routes(
        &self,
        graph: &mut RenderGraph,
        input: BufferHandle,
        input_count: BufferHandle,
        emitter_route_table: BufferHandle,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        self.append_to_graph_with_input_handle(
            graph,
            input,
            Some(input_count),
            Some(emitter_route_table),
            false,
        )
    }

    /// Appends the surface raster consumer into caller-owned HDR and depth
    /// attachments. Color must be single-sampled Rgba16Float and depth must be
    /// Depth32Float, using the surface's linear reversed-Z near/far mapping.
    /// Color must be the SSA version initialized by an earlier graph pass;
    /// freshly imported version zero is cleared by the graph on first write.
    /// The pass alpha-blends color and tests GreaterEqual without writing depth;
    /// only color advances its SSA version. Existing attachments are loaded.
    /// Depth must support TEXTURE_BINDING as well as RENDER_ATTACHMENT; its
    /// graph-owned binding lives until the recorded graph is released.
    pub fn append_raster_to_graph(
        &self,
        graph: &mut RenderGraph,
        outputs: &ParticleRendererGraphOutputs,
        color: TextureHandle,
        depth: TextureHandle,
    ) -> Result<(TextureHandle, TextureHandle), CompileError> {
        let mesh_vertices = outputs.strip_geometry.unwrap_or_else(|| {
            import_buffer(
                graph,
                "particle-render-main-mesh-vertices",
                self.surface_mesh_vertices,
                u64::from(self_index_capacity(self.route).max(3)) * SURFACE_MESH_VERTEX_STRIDE,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                BufferState::storage_read(PipelineStages::VERTEX_SHADER),
            )
        });
        let index_buffer = import_buffer(
            graph,
            "particle-render-main-index-buffer",
            self.surface_index_buffer,
            u64::from(surface_index_count(self.route).max(1)) * 4,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            BufferState::index(),
        );
        let material = import_buffer(
            graph,
            "particle-render-main-material",
            self.surface_material,
            SURFACE_MATERIAL_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let camera = self.import_camera(graph);
        let pipeline = self.hdr_surface_pipeline;
        let bind_group = self.surface_bind_group;
        let depth_bind_group = graph.create_texture_bind_group(
            self.depth_bind_group_layout,
            vec![(0, depth)],
            Some(String::from("particle-renderer-scene-depth")),
        );
        let index_id = self.surface_index_buffer;
        let route = self.route;
        let viewport = *self
            .raster_viewport
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let visible = outputs.visible_indices;
        let input = outputs.input;
        let sprite_args = outputs.sprite_args;
        let indexed_args = outputs.indexed_args;
        let input_count = outputs.input_count;
        let control = outputs.control;
        let mut color_after = color;
        let mut depth_after = depth;
        graph.add_raster_pass(
            format!("particle-render-main-raster-{:?}", route.kind()),
            PassFlags::SIDE_EFFECT,
            |builder| {
                color_after = builder.color_attachment(color_after);
                depth_after = builder.read_depth_attachment(depth_after);
                builder.read_texture(depth_after);
                builder.read_storage_buffer(input);
                builder.read_storage_buffer(visible);
                if let Some(strip_vertices) = outputs.strip_vertices {
                    builder.read_storage_buffer(strip_vertices);
                }
                builder.read_storage_buffer(input_count);
                builder.read_storage_buffer(control);
                builder.read_storage_buffer(mesh_vertices);
                builder.read_uniform(material);
                builder.read_uniform(camera);
                builder.read_index(index_buffer);
                match route.kind() {
                    RendererKind::Sprite => builder.read_indirect(sprite_args),
                    RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
                        builder.read_indirect(indexed_args)
                    }
                    RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                        unreachable!()
                    }
                };
                move |context| {
                    let depth_group = context.texture_bind_group(depth_bind_group);
                    context
                        .set_render_pipeline(pipeline)
                        .set_render_bind_group(0, bind_group, Vec::new())
                        .set_render_bind_group(1, depth_group, Vec::new())
                        .set_index_buffer(index_id, IndexFormat::Uint32, 0, None);
                    apply_viewport(context, viewport);
                    match route.kind() {
                        RendererKind::Sprite => {
                            context.draw_indirect(sprite_args, 0);
                        }
                        RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
                            context.draw_indexed_indirect(indexed_args, 0);
                        }
                        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                            unreachable!()
                        }
                    }
                }
            },
        );
        Ok((color_after, depth_after))
    }

    fn append_to_graph_with_input_handle(
        &self,
        graph: &mut RenderGraph,
        input: BufferHandle,
        external_input_count: Option<BufferHandle>,
        external_route_table: Option<BufferHandle>,
        include_raster: bool,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        let camera = self.import_camera(graph);
        let frustum = import_buffer(
            graph,
            "particle-render-frustum",
            self.frustum,
            96,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let params = import_buffer(
            graph,
            "particle-render-params",
            self.params,
            64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let input_count = match external_input_count {
            Some(handle) => handle,
            None => import_buffer(
                graph,
                "particle-render-input-count",
                self.input_count,
                4,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
            ),
        };
        let route_table = external_route_table.unwrap_or_else(|| {
            import_buffer(
                graph,
                "particle-render-emitter-routes",
                self.route_table,
                self.emitter_source
                    .map_or(16, |source| 16 + u64::from(source.route_capacity) * 16),
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
            )
        });
        let control = import_buffer(
            graph,
            "particle-render-control",
            self.control,
            16,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let sort_a = import_buffer(
            graph,
            "particle-render-sort-a",
            self.sort_a,
            u64::from(self.padded_capacity) * 8,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let sort_b = import_buffer(
            graph,
            "particle-render-sort-b",
            self.sort_b,
            u64::from(self.padded_capacity) * 8,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let visible = import_buffer(
            graph,
            "particle-render-visible",
            self.visible_indices,
            u64::from(self.capacity) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let sprite_args = import_buffer(
            graph,
            "particle-render-sprite-args",
            self.sprite_args,
            16,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let indexed_args = import_buffer(
            graph,
            "particle-render-indexed-args",
            self.indexed_args,
            20,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let route_params = import_buffer(
            graph,
            "particle-render-route-params",
            self.route_params,
            48,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let reset_pipeline = self.reset_pipeline;
        let reset_bind_group = self.cull_bind_group;
        let (control_reset, sprite_reset, indexed_reset) = {
            let mut control_reset = control;
            let mut sprite_reset = sprite_args;
            let mut indexed_reset = indexed_args;
            graph.add_compute_pass(
                "particle-renderer-reset",
                PassFlags::SIDE_EFFECT,
                |builder| {
                    builder.read_uniform(params);
                    builder.read_storage_buffer(input);
                    builder.read_storage_buffer(input_count);
                    builder.read_storage_buffer(route_table);
                    control_reset = builder.write_storage_buffer(control);
                    sprite_reset = builder.write_storage_buffer(sprite_args);
                    indexed_reset = builder.write_storage_buffer(indexed_args);
                    move |context| {
                        context
                            .set_compute_pipeline(reset_pipeline)
                            .set_compute_bind_group(0, reset_bind_group, Vec::new())
                            .dispatch(1, 1, 1);
                    }
                },
            );
            (control_reset, sprite_reset, indexed_reset)
        };

        let seed_pipeline = self.seed_pipeline;
        let seed_bind_group = self.cull_bind_group;
        let seed_groups = self.padded_capacity.div_ceil(64);
        let sort_a_seed = {
            let mut output = sort_a;
            graph.add_compute_pass(
                "particle-renderer-seed",
                PassFlags::SIDE_EFFECT,
                |builder| {
                    output = builder.write_storage_buffer(sort_a);
                    move |context| {
                        context
                            .set_compute_pipeline(seed_pipeline)
                            .set_compute_bind_group(0, seed_bind_group, Vec::new())
                            .dispatch(seed_groups, 1, 1);
                    }
                },
            );
            output
        };

        let cull_pipeline = self.cull_pipeline;
        let cull_bind_group = self.cull_bind_group;
        let cull_groups = self.capacity.div_ceil(64);
        let (control_cull, sort_a_cull) = {
            let mut control_cull = control_reset;
            let mut sort_a_cull = sort_a_seed;
            graph.add_compute_pass(
                "particle-renderer-cull",
                PassFlags::SIDE_EFFECT,
                |builder| {
                    builder.read_uniform(frustum);
                    builder.read_uniform(params);
                    builder.read_uniform(camera);
                    builder.read_storage_buffer(input);
                    builder.read_storage_buffer(input_count);
                    builder.read_storage_buffer(route_table);
                    control_cull = builder.write_storage_buffer(control_reset);
                    sort_a_cull = builder.write_storage_buffer(sort_a_seed);
                    move |context| {
                        context
                            .set_compute_pipeline(cull_pipeline)
                            .set_compute_bind_group(0, cull_bind_group, Vec::new())
                            .dispatch(cull_groups, 1, 1);
                    }
                },
            );
            (control_cull, sort_a_cull)
        };

        let mut sort_a_current = sort_a_cull;
        let mut sort_b_current = sort_b;
        if self.route.sort.is_explicit_sort() || self.stable_strip() {
            for (index, bind_group) in self.sort_bind_groups.iter().copied().enumerate() {
                let (source, destination) = if index % 2 == 0 {
                    (sort_a_current, sort_b_current)
                } else {
                    (sort_b_current, sort_a_current)
                };
                let mut output = destination;
                let sort_pipeline = self.sort_pipeline;
                let sort_groups = self.padded_capacity.div_ceil(64);
                let step = import_buffer(
                    graph,
                    &format!("particle-render-sort-step-{index}"),
                    self.sort_steps[index],
                    16,
                    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    BufferState::copy_dst(),
                );
                graph.add_compute_pass(
                    format!("particle-renderer-sort-{index}"),
                    PassFlags::SIDE_EFFECT,
                    |builder| {
                        builder.read_storage_buffer(source);
                        builder.read_storage_buffer(input);
                        builder.read_uniform(step);
                        output = builder.write_storage_buffer(destination);
                        move |context| {
                            context
                                .set_compute_pipeline(sort_pipeline)
                                .set_compute_bind_group(0, bind_group, Vec::new())
                                .dispatch(sort_groups, 1, 1);
                        }
                    },
                );
                if index % 2 == 0 {
                    sort_b_current = output;
                } else {
                    sort_a_current = output;
                }
            }
        }
        let sorted = if self.route.sort.is_explicit_sort() || self.stable_strip() {
            if self.sort_bind_groups.len() % 2 == 0 {
                sort_a_current
            } else {
                sort_b_current
            }
        } else {
            sort_a_cull
        };

        let compact_pipeline = self.compact_pipeline;
        let compact_bind_group = self.compact_bind_group;
        let compact_groups = self.capacity.div_ceil(64);
        let mut visible_output = visible;
        graph.add_compute_pass(
            "particle-renderer-compact",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(sorted);
                builder.read_uniform(params);
                builder.read_storage_buffer(control_cull);
                visible_output = builder.write_storage_buffer(visible);
                move |context| {
                    context
                        .set_compute_pipeline(compact_pipeline)
                        .set_compute_bind_group(0, compact_bind_group, Vec::new())
                        .dispatch(compact_groups, 1, 1);
                }
            },
        );

        let strip_vertices = self
            .strip
            .as_ref()
            .map(|strip| strip.append(graph, input, input_count, visible_output, control_cull));

        let args_pipeline = self.args_pipeline;
        let args_bind_group = self.args_bind_group;
        let mut sprite_output = sprite_reset;
        let mut indexed_output = indexed_reset;
        graph.add_compute_pass(
            "particle-renderer-emit-args",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(control_cull);
                sprite_output = builder.write_storage_buffer(sprite_reset);
                builder.read_uniform(route_params);
                indexed_output = builder.write_storage_buffer(indexed_reset);
                move |context| {
                    context
                        .set_compute_pipeline(args_pipeline)
                        .set_compute_bind_group(0, args_bind_group, Vec::new())
                        .dispatch(1, 1, 1);
                }
            },
        );
        let strip_geometry = self
            .strip
            .as_ref()
            .and_then(|strip| strip.tessellation.as_ref())
            .map(|tessellation| {
                let (geometry, args) = tessellation.append(
                    graph,
                    input,
                    visible_output,
                    control_cull,
                    strip_vertices.expect("tessellation requires strip prefixes"),
                    camera,
                    indexed_output,
                );
                indexed_output = args;
                geometry
            });
        if include_raster {
            let mesh_vertices = strip_geometry.unwrap_or_else(|| {
                import_buffer(
                    graph,
                    "particle-render-mesh-vertices",
                    self.surface_mesh_vertices,
                    u64::from(self_index_capacity(self.route).max(3)) * SURFACE_MESH_VERTEX_STRIDE,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    BufferState::storage_read(PipelineStages::VERTEX_SHADER),
                )
            });
            let index_buffer = import_buffer(
                graph,
                "particle-render-index-buffer",
                self.surface_index_buffer,
                u64::from(surface_index_count(self.route).max(1)) * 4,
                BufferUsages::INDEX | BufferUsages::COPY_DST,
                BufferState::index(),
            );
            let material = import_buffer(
                graph,
                "particle-render-material",
                self.surface_material,
                SURFACE_MATERIAL_BYTES,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                BufferState::copy_dst(),
            );
            let target = graph.import_texture(
                "particle-renderer-surface-target",
                TextureDesc::color(TextureFormat::Rgba8Unorm).with_size(SizeClass::absolute_2d(
                    RENDERER_MATRIX_TARGET_SIZE,
                    RENDERER_MATRIX_TARGET_SIZE,
                )),
                ImportedTexture {
                    texture: self.surface_target,
                    default_view: self.surface_target_view,
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::color_target(),
                    mip_levels: 1,
                    array_layers: 1,
                },
            );
            let surface_pipeline = self.surface_pipeline;
            let surface_bind_group = self.surface_bind_group;
            let surface_index_buffer = self.surface_index_buffer;
            let renderer_route = self.route;
            let viewport = *self
                .raster_viewport
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let visible_for_raster = visible_output;
            let indexed_for_raster = indexed_output;
            let input_for_raster = input;
            graph.add_raster_pass(
                format!("particle-renderer-raster-{:?}", renderer_route.kind()),
                PassFlags::SIDE_EFFECT,
                |builder| {
                    builder.read_storage_buffer(input_for_raster);
                    builder.read_storage_buffer(visible_for_raster);
                    if let Some(strip_vertices) = strip_vertices {
                        builder.read_storage_buffer(strip_vertices);
                    }
                    builder.read_storage_buffer(input_count);
                    builder.read_storage_buffer(control_cull);
                    builder.read_storage_buffer(mesh_vertices);
                    builder.read_uniform(material);
                    builder.read_uniform(camera);
                    builder.read_index(index_buffer);
                    match renderer_route.kind() {
                        RendererKind::Sprite => builder.read_indirect(sprite_output),
                        RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
                            builder.read_indirect(indexed_for_raster)
                        }
                        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                            unreachable!()
                        }
                    };
                    builder.color_attachment(target);
                    move |context| {
                        context
                            .set_render_pipeline(surface_pipeline)
                            .set_render_bind_group(0, surface_bind_group, Vec::new())
                            .set_index_buffer(surface_index_buffer, IndexFormat::Uint32, 0, None);
                        apply_viewport(context, viewport);
                        match renderer_route.kind() {
                            RendererKind::Sprite => {
                                context.draw_indirect(sprite_output, 0);
                            }
                            RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
                                context.draw_indexed_indirect(indexed_for_raster, 0);
                            }
                            RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                                unreachable!()
                            }
                        }
                    }
                },
            );
        }
        Ok(ParticleRendererGraphOutputs {
            input,
            input_count,
            visible_indices: visible_output,
            strip_vertices,
            strip_geometry,
            sprite_args: sprite_output,
            indexed_args: indexed_output,
            control: control_cull,
        })
    }

    /// Builds a graph containing only this stage's compute passes.
    #[must_use]
    pub fn graph_frame(&self) -> RenderGraph {
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        self.append_to_graph(&mut graph)
            .expect("particle renderer graph admission");
        graph
    }

    /// Executes this stage's graph and submits its command buffer.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        self.graph_frame().execute(device, queue)
    }

    /// The local work layout derived from the architecture sort/cull contract.
    /// For budgeted emitters this describes work_capacity, not shared capacity.
    #[must_use]
    pub const fn layout(&self) -> SortCullBufferLayout {
        self.layout
    }

    /// The renderer route used by this stage.
    #[must_use]
    pub const fn route(&self) -> ParticleRendererRoute {
        self.route
    }

    /// Capacity of the shared source buffer (also the work capacity for legacy
    /// constructors). Kept separate from per-emitter budgets for bridge owners.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.input_capacity
    }

    /// Maximum number of particles this stage can cull/compact/sort.
    #[must_use]
    pub const fn work_capacity(&self) -> u32 {
        self.capacity
    }

    /// The persistent renderer-input buffer that a GPU simulation bridge may
    /// write before this stage consumes its graph handle.
    #[must_use]
    pub const fn input_buffer(&self) -> BufferId {
        self.input
    }

    /// The GPU input-count buffer consumed by the cull pass.
    #[must_use]
    pub const fn input_count_buffer(&self) -> BufferId {
        self.input_count
    }

    /// The GPU-authored compact visible-index buffer.
    #[must_use]
    pub const fn visible_indices(&self) -> BufferId {
        self.visible_indices
    }

    /// GPU-generated 16-byte records in visible-rank order: `(u: f32,
    /// head_rank_plus_one: u32, incoming_edge_valid: u32, record_valid: u32)`.
    /// U is XYZ arc length times UV density, reset at every break and saturated
    /// at 1e30 for extreme finite input. Tail records are invalid every frame.
    /// Exposed for diagnostics/readback, never read back by the renderer itself.
    #[must_use]
    pub fn strip_vertices(&self) -> Option<BufferId> {
        self.strip.as_ref().map(StripGpu::output)
    }

    /// GPU-generated 48-byte subedges: start XYZ/half-width, end XYZ/half-width,
    /// then U-start/U-end/fixture-depth-start/fixture-depth-end. Only the prefix
    /// selected by indexed_args.instance_count is drawn; every tail is cleared.
    pub fn strip_tessellated_edges(&self) -> Option<BufferId> {
        self.strip
            .as_ref()?
            .tessellation
            .as_ref()
            .map(|tessellation| tessellation.geometry)
    }

    /// Diagnostic per-source-rank prefix records `(end:u32, count:u32,
    /// arc:f32, head:u32)`, 16 bytes each. Never read back by production code.
    pub fn strip_tessellation_prefixes(&self) -> Option<BufferId> {
        self.strip
            .as_ref()?
            .tessellation
            .as_ref()
            .map(TessellationGpu::prefixes)
    }

    /// The GPU-authored non-indexed Sprite indirect argument buffer.
    #[must_use]
    pub const fn sprite_args(&self) -> BufferId {
        self.sprite_args
    }

    /// The GPU-authored indexed Mesh/Ribbon/Beam indirect argument buffer.
    #[must_use]
    pub const fn indexed_args(&self) -> BufferId {
        self.indexed_args
    }

    /// The GPU-authored visible counter buffer.
    #[must_use]
    pub const fn control(&self) -> BufferId {
        self.control
    }

    /// The offscreen target written by the renderer-specific raster pass.
    #[must_use]
    pub const fn raster_target(&self) -> rhi::TextureId {
        self.surface_target
    }

    /// The final sorted item buffer, or the cull output when sorting is disabled.
    #[must_use]
    pub const fn sorted_output(&self) -> BufferId {
        self.sorted_output
    }

    /// Destroys every persistent resource owned by the stage.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        device.destroy_bind_group(self.surface_bind_group);
        if let Some(strip) = &self.strip {
            strip.destroy(device);
        } else {
            device.destroy_buffer(self.strip_vertices);
        }
        device.destroy_bind_group(self.args_bind_group);
        device.destroy_compute_pipeline(self.args_pipeline);
        device.destroy_pipeline_layout(self.args_pipeline_layout);
        device.destroy_bind_group_layout(self.args_layout);
        device.destroy_bind_group(self.compact_bind_group);
        device.destroy_compute_pipeline(self.compact_pipeline);
        device.destroy_pipeline_layout(self.compact_pipeline_layout);
        device.destroy_bind_group_layout(self.compact_layout);
        for &bind_group in &self.sort_bind_groups {
            device.destroy_bind_group(bind_group);
        }
        for &step in &self.sort_steps {
            device.destroy_buffer(step);
        }
        device.destroy_compute_pipeline(self.sort_pipeline);
        device.destroy_pipeline_layout(self.sort_pipeline_layout);
        device.destroy_bind_group_layout(self.sort_layout);
        device.destroy_bind_group(self.cull_bind_group);
        for pipeline in [self.reset_pipeline, self.seed_pipeline, self.cull_pipeline] {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_pipeline_layout(self.cull_pipeline_layout);
        device.destroy_bind_group_layout(self.cull_layout);
        device.destroy_shader_module(self.cull_shader);
        device.destroy_shader_module(self.sort_shader);
        device.destroy_shader_module(self.compact_shader);
        device.destroy_shader_module(self.args_shader);
        for buffer in [
            self.frustum,
            self.camera,
            self.params,
            self.control,
            self.sort_a,
            self.sort_b,
            self.visible_indices,
            self.sprite_args,
            self.indexed_args,
            self.route_params,
            self.surface_mesh_vertices,
            self.surface_index_buffer,
            self.surface_material,
        ] {
            device.destroy_buffer(buffer);
        }
        if self.owns_input {
            device.destroy_buffer(self.input);
            device.destroy_buffer(self.input_count);
            device.destroy_buffer(self.route_table);
        }
        device.destroy_render_pipeline(self.surface_pipeline);
        device.destroy_render_pipeline(self.hdr_surface_pipeline);
        device.destroy_pipeline_layout(self.hdr_surface_pipeline_layout);
        device.destroy_bind_group_layout(self.depth_bind_group_layout);
        device.destroy_pipeline_layout(self.surface_pipeline_layout);
        device.destroy_bind_group_layout(self.surface_bind_group_layout);
        device.destroy_shader_module(self.surface_shader);
        device.destroy_texture_view(self.surface_target_view);
        device.destroy_texture(self.surface_target);
    }

    fn import_camera(&self, graph: &mut RenderGraph) -> BufferHandle {
        import_buffer(
            graph,
            "particle-render-camera",
            self.camera,
            CAMERA_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        )
    }
}

fn make_compute<D: RenderDevice + ?Sized>(
    device: &D,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    entry_point: &'static str,
) -> ComputePipelineId {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(format!("particle-renderer-{entry_point}")),
        layout: Some(layout),
        module: shader,
        entry_point: entry_point.into(),
    })
}

fn self_index_capacity(route: ParticleRendererRoute) -> u32 {
    match route.geometry {
        RendererGeometry::Sprite { .. } => SPRITE_QUAD_VERTEX_COUNT,
        RendererGeometry::Mesh { index_count, .. } => index_count.max(3),
        RendererGeometry::Ribbon {
            segments_per_chain, ..
        }
        | RendererGeometry::Beam {
            segments_per_chain, ..
        } => segments_per_chain.saturating_mul(4).max(4),
    }
}

fn surface_index_count(route: ParticleRendererRoute) -> u32 {
    match route.geometry {
        RendererGeometry::Sprite { .. } => 6,
        _ => route.index_count(),
    }
}

fn surface_mesh_vertex_bytes(route: ParticleRendererRoute) -> Vec<u8> {
    let count = self_index_capacity(route).max(3);
    let triangle = [[-0.48_f32, -0.40_f32], [0.48, -0.40], [0.0, 0.52]];
    let mut bytes = Vec::with_capacity(count as usize * SURFACE_MESH_VERTEX_STRIDE as usize);
    for index in 0..count {
        let point = triangle[(index % 3) as usize];
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
    }
    bytes
}

fn surface_index_bytes(route: ParticleRendererRoute) -> Vec<u8> {
    let count = surface_index_count(route);
    let mut bytes = Vec::with_capacity(count as usize * 4);
    for index in 0..count {
        let vertex = match route.geometry {
            RendererGeometry::Sprite { .. } => [0_u32, 1, 2, 0, 2, 3][(index % 6) as usize],
            RendererGeometry::Mesh { .. } => index,
            RendererGeometry::Ribbon { .. } | RendererGeometry::Beam { .. } => {
                let local = index % 6;
                let segment = index / 6;
                // strip_vertex interleaves start/end sides, unlike sprite corners.
                segment * 4 + [0_u32, 1, 2, 2, 1, 3][local as usize]
            }
        };
        bytes.extend_from_slice(&vertex.to_le_bytes());
    }
    bytes
}

fn surface_material_bytes(
    route: ParticleRendererRoute,
    view: ParticleCullView,
) -> [u8; SURFACE_MATERIAL_BYTES as usize] {
    let kind_tint: [f32; 4] = match route.kind() {
        RendererKind::Sprite => [1.0, 0.72, 0.20, 1.0],
        RendererKind::Mesh => [0.20, 0.82, 0.38, 1.0],
        RendererKind::Ribbon => [0.20, 0.58, 1.0, 1.0],
        RendererKind::Beam => [1.0, 0.24, 0.12, 1.0],
        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => [1.0; 4],
    };
    let words = [
        1.0_f32.to_bits(),
        1.0_f32.to_bits(),
        1.0_f32.to_bits(),
        0.0_f32.to_bits(),
        kind_tint[0].to_bits(),
        kind_tint[1].to_bits(),
        kind_tint[2].to_bits(),
        kind_tint[3].to_bits(),
        0.08_f32.to_bits(),
        0.025_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        route.first_instance(),
        view.near_depth.to_bits(),
        view.far_depth.to_bits(),
        0.0_f32.to_bits(),
    ];
    let mut bytes = [0; SURFACE_MATERIAL_BYTES as usize];
    for (index, word) in words.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn make_surface_pipeline<D: RenderDevice + ?Sized>(
    device: &D,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    kind: RendererKind,
) -> RenderPipelineId {
    make_surface_pipeline_for_target(
        device,
        layout,
        shader,
        kind,
        TextureFormat::Rgba8Unorm,
        None,
        "fs",
    )
}

fn make_surface_pipeline_for_target<D: RenderDevice + ?Sized>(
    device: &D,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    kind: RendererKind,
    color_format: TextureFormat,
    depth_stencil: Option<DepthStencilState>,
    fragment_entry: &'static str,
) -> RenderPipelineId {
    let entry = match kind {
        RendererKind::Sprite => "vs_sprite",
        RendererKind::Mesh => "vs_mesh",
        RendererKind::Ribbon => "vs_ribbon",
        RendererKind::Beam => "vs_beam",
        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => unreachable!(),
    };
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(format!("particle-renderer-{kind:?}")),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: entry.into(),
            buffers: Vec::new(),
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..PrimitiveState::default()
        },
        depth_stencil,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: shader,
            entry_point: fragment_entry.into(),
            targets: vec![Some(ColorTargetState::alpha_blended(color_format))],
        }),
    })
}

fn sort_step_at(padded_capacity: u32, index: usize) -> (u32, u32) {
    let mut remaining = index;
    let mut k = 2u32;
    while k <= padded_capacity {
        let mut j = k / 2;
        while j != 0 {
            if remaining == 0 {
                return (k, j);
            }
            remaining -= 1;
            if j == 1 {
                break;
            }
            j /= 2;
        }
        if k > padded_capacity / 2 {
            break;
        }
        k *= 2;
    }
    (padded_capacity, 1)
}

fn import_buffer(
    graph: &mut RenderGraph,
    name: &str,
    buffer: BufferId,
    size: u64,
    usage: BufferUsages,
    entry_state: BufferState,
) -> BufferHandle {
    graph.import_buffer(
        name,
        BufferDesc::new(size, usage),
        ImportedBuffer {
            buffer,
            entry_state,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::particle::indirect_draw::RendererGeometry;
    use prism_render_driver::Instance as _;
    use prism_render_driver_null::NullInstance;

    #[derive(Default)]
    struct UploadQueue(Mutex<Vec<(BufferId, u64, Vec<u8>)>>);

    impl RenderQueue for UploadQueue {
        fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
            self.0.lock().unwrap().push((buffer, offset, data.to_vec()));
        }

        fn write_texture(&self, _: rhi::TextureWrite, _: &[u8], _: rhi::ImageDataLayout) {
            panic!("particle view preparation must not upload textures");
        }

        fn submit(&self, _: &[rhi::CommandBuffer]) {
            panic!("particle view preparation must not submit work");
        }
    }

    #[test]
    fn view_updates_do_not_reupload_static_geometry_or_sort_steps() {
        let instance = NullInstance::new();
        let adapter = instance.request_adapter(&Default::default()).unwrap();
        let (device, _) = instance
            .create_device(&adapter, &Default::default())
            .unwrap();
        let mut stage = ParticleRendererGpu::new(&device, 8, route(RendererKind::Mesh)).unwrap();
        let queue = UploadQueue::default();
        // Custom geometry can precede the first view; initialization must never
        // overwrite it later with the built-in fixture mesh.
        let vertices = [ParticleMeshVertex::new([0.25, -0.5, 0.75], [0.2, 0.8]); 36];
        let indices = [0; 36];
        stage
            .set_mesh_geometry(&queue, &vertices, &indices)
            .unwrap();
        queue.0.lock().unwrap().clear();
        for _ in 0..3 {
            stage
                .set_view(&queue, ParticleCullView::unrestricted())
                .unwrap();
        }
        let writes = queue.0.lock().unwrap();
        assert_eq!(
            writes.len(),
            12,
            "four dynamic uniforms per view, no geometry/sort uploads"
        );
        for (buffer, offset, _) in writes.iter() {
            assert!([
                stage.frustum,
                stage.camera,
                stage.params,
                stage.surface_material
            ]
            .contains(buffer));
            assert_eq!(*offset, 0);
        }
        drop(writes);
        let before = queue.0.lock().unwrap().len();
        assert_eq!(
            stage.set_view(
                &queue,
                ParticleCullView {
                    far_depth: 0.0,
                    ..ParticleCullView::unrestricted()
                }
            ),
            Err(ParticleRendererGpuError::InvalidView)
        );
        assert_eq!(
            queue.0.lock().unwrap().len(),
            before,
            "reject invalid view before any write"
        );
        stage.destroy(&device);
    }

    #[test]
    fn native_view_records_matching_viewport_and_scissor() {
        let instance = NullInstance::new();
        let adapter = instance.request_adapter(&Default::default()).unwrap();
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .unwrap();
        let stage = ParticleRendererGpu::new(&device, 8, route(RendererKind::Sprite)).unwrap();
        let camera = ParticleCamera::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [12, 8, 64, 32],
        )
        .unwrap();
        stage
            .set_view(
                &queue,
                ParticleCullView::from_camera(camera, 0.0, 100.0, true).unwrap(),
            )
            .unwrap();
        let recorded = stage.execute(&device, &queue).unwrap();
        let mut found_viewport = false;
        let mut found_scissor = false;
        for pass in &recorded.commands().passes {
            if let rhi::Pass::Render { commands, .. } = pass {
                for command in commands {
                    match command {
                        rhi::RenderCommand::SetViewport(viewport) => {
                            assert_eq!(
                                (viewport.x, viewport.y, viewport.width, viewport.height),
                                (12.0, 8.0, 64.0, 32.0)
                            );
                            found_viewport = true;
                        }
                        rhi::RenderCommand::SetScissor(scissor) => {
                            assert_eq!(
                                (scissor.x, scissor.y, scissor.width, scissor.height),
                                (12, 8, 64, 32)
                            );
                            found_scissor = true;
                        }
                        _ => {}
                    }
                }
            }
        }
        assert!(found_viewport && found_scissor);
        recorded.release(&device);
        stage.destroy(&device);
    }

    #[test]
    fn render_input_abi_preserves_prefix_and_serializes_explicit_ribbon_birth() {
        assert_eq!(
            core::mem::size_of::<ParticleRenderInput>() as u64,
            PARTICLE_RENDER_INPUT_BYTES
        );
        assert_eq!(core::mem::offset_of!(ParticleRenderInput, center_radius), 0);
        assert_eq!(core::mem::offset_of!(ParticleRenderInput, depth), 16);
        assert_eq!(
            core::mem::offset_of!(ParticleRenderInput, particle_index),
            20
        );
        assert_eq!(core::mem::offset_of!(ParticleRenderInput, flags), 24);
        assert_eq!(core::mem::offset_of!(ParticleRenderInput, emitter_id), 28);
        assert_eq!(core::mem::offset_of!(ParticleRenderInput, ribbon_id), 32);
        assert_eq!(
            core::mem::offset_of!(ParticleRenderInput, birth_sequence),
            36
        );
        assert_eq!(
            core::mem::offset_of!(ParticleRenderInput, topology_flags),
            44
        );
        let original = ParticleRenderInput::new([1.0, 2.0, 3.0], 4.0, 0.5, 97);
        assert_eq!(original.topology_flags, 0, "pool identity is not topology");
        let mut sample = original.with_ribbon_topology(7, 0x0123_4567_89ab_cdef);
        sample.emitter_id = 42;
        let mut bytes = Vec::new();
        sample.append_bytes(&mut bytes);
        assert_eq!(bytes.len() as u64, PARTICLE_RENDER_INPUT_BYTES);
        let expected = [
            1f32.to_bits(),
            2f32.to_bits(),
            3f32.to_bits(),
            4f32.to_bits(),
            0.5f32.to_bits(),
            97,
            0,
            42,
            7,
            0x89ab_cdef,
            0x0123_4567,
            1,
        ];
        assert_eq!(
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(sample.validate());
        sample.topology_flags = 2;
        assert!(!sample.validate(), "reserved topology flags must stay zero");
    }

    fn route(kind: RendererKind) -> ParticleRendererRoute {
        let geometry = match kind {
            RendererKind::Sprite => RendererGeometry::Sprite {
                alive_count: 16,
                first_instance: 2,
            },
            RendererKind::Mesh => RendererGeometry::Mesh {
                alive_count: 16,
                index_count: 36,
                first_instance: 2,
            },
            RendererKind::Ribbon => RendererGeometry::Ribbon {
                chain_count: 4,
                segments_per_chain: 8,
                first_instance: 2,
            },
            RendererKind::Beam => RendererGeometry::Beam {
                chain_count: 4,
                segments_per_chain: 8,
                first_instance: 2,
            },
            RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                panic!("surface test route")
            }
        };
        ParticleRendererRoute::new(kind, SortStrategy::ViewDepthBitonic, geometry)
            .expect("valid route")
    }

    #[test]
    fn route_matrix_accepts_the_four_surface_geometry_contracts() {
        for kind in [
            RendererKind::Sprite,
            RendererKind::Mesh,
            RendererKind::Ribbon,
            RendererKind::Beam,
        ] {
            assert_eq!(route(kind).kind(), kind);
        }
        assert_eq!(route(RendererKind::Ribbon).index_count(), 48);
        assert_eq!(route(RendererKind::Beam).index_count(), 48);
    }

    #[test]
    fn soft_depth_material_preserves_uniform_layout_and_rejects_invalid_distances() {
        let instance = NullInstance::new();
        let adapter = instance.request_adapter(&Default::default()).unwrap();
        let (device, _) = instance
            .create_device(&adapter, &Default::default())
            .unwrap();
        let baseline = device.live_resource_count();
        let renderer_route = route(RendererKind::Sprite);
        let mut renderer = ParticleRendererGpu::new(&device, 8, renderer_route).unwrap();
        assert_eq!(renderer.material.depth_fade_distance, 0.0);
        let material = ParticleSurfaceMaterial {
            scale: [0.5, 2.0],
            softness: 0.25,
            depth_fade_distance: 3.0,
            tint: [4.0, 0.5, -0.25, 0.75],
            ribbon_width: 0.125,
            strip: ParticleStripSettings::default(),
        };
        renderer.set_material(material).unwrap();
        let view = ParticleCullView {
            near_depth: 2.0,
            far_depth: 20.0,
            ..ParticleCullView::unrestricted()
        };
        let bytes = renderer.material_bytes(view);
        assert_eq!(bytes.len(), 64);
        for (offset, expected) in [
            (0, 0.5_f32),
            (4, 2.0),
            (8, 0.25),
            (12, 3.0),
            (16, 4.0),
            (20, 0.5),
            (24, -0.25),
            (28, 0.75),
            (32, 0.125),
            (52, 2.0),
            (56, 20.0),
        ] {
            assert_eq!(&bytes[offset..offset + 4], &expected.to_le_bytes());
        }
        assert_eq!(
            &bytes[48..52],
            &renderer_route.first_instance().to_le_bytes()
        );
        let allocated = device.live_resource_count();
        for distance in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                renderer.set_material(ParticleSurfaceMaterial {
                    depth_fade_distance: distance,
                    ..material
                }),
                Err(ParticleRendererGpuError::InvalidInput)
            ));
            assert_eq!(renderer.material, material, "invalid edits must be atomic");
            assert_eq!(device.live_resource_count(), allocated);
        }
        renderer.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn view_and_material_preserve_the_depth_interval_and_reserved_count() {
        let view = ParticleCullView {
            near_depth: 2.0,
            far_depth: 20.0,
            ..ParticleCullView::unrestricted()
        };
        let params = view.params_bytes(7);
        assert_eq!(&params[4..8], &0u32.to_le_bytes());
        let material = surface_material_bytes(route(RendererKind::Sprite), view);
        assert_eq!(&material[48..52], &2u32.to_le_bytes());
        assert_eq!(&material[52..56], &2f32.to_le_bytes());
        assert_eq!(&material[56..60], &20f32.to_le_bytes());
        assert!(!ParticleCullView {
            near_depth: -f32::MAX,
            far_depth: f32::MAX,
            ..view
        }
        .validate());
    }

    #[test]
    fn emitter_local_indirect_and_material_bases_are_zero_with_nonzero_metadata() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, _) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let owner =
            ParticleRendererGpu::new(&device, 8, route(RendererKind::Sprite)).expect("owner");
        let table = device.create_buffer(&BufferDescriptor::new(32, BufferUsages::STORAGE));
        for kind in [
            RendererKind::Sprite,
            RendererKind::Mesh,
            RendererKind::Ribbon,
            RendererKind::Beam,
        ] {
            let stage = ParticleRendererGpu::new_emitter_with_budget(
                &device,
                8,
                4,
                route(kind),
                ParticleEmitterGpuSource {
                    input: owner.input_buffer(),
                    input_count: owner.input_count_buffer(),
                    route_table: table,
                    route_capacity: 1,
                    route_index: 0,
                    emitter_id: 17,
                },
            )
            .expect("emitter");
            assert_eq!(stage.route().first_instance(), 2);
            let args = stage.route_params_bytes();
            assert_eq!(&args[4..8], &4u32.to_le_bytes());
            assert_eq!(&args[12..16], &0u32.to_le_bytes());
            let material = stage.material_bytes(ParticleCullView::unrestricted());
            assert_eq!(&material[48..52], &0u32.to_le_bytes());
            stage.destroy(&device);
        }
        assert_eq!(
            owner.draw_first_instance(),
            2,
            "standalone ABI is unchanged"
        );
        owner.destroy(&device);
        device.destroy_buffer(table);
    }

    #[test]
    fn emitter_budget_bounds_work_buffers_and_dispatch_not_shared_input() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let owner =
            ParticleRendererGpu::new(&device, 257, route(RendererKind::Sprite)).expect("owner");
        let table = device.create_buffer(&BufferDescriptor::new(32, BufferUsages::STORAGE));
        let source = ParticleEmitterGpuSource {
            input: owner.input_buffer(),
            input_count: owner.input_count_buffer(),
            route_table: table,
            route_capacity: 1,
            route_index: 0,
            emitter_id: 17,
        };
        let baseline = device.live_resource_count();
        for budget in [0, 258] {
            assert!(matches!(
                ParticleRendererGpu::new_emitter_with_budget(
                    &device,
                    257,
                    budget,
                    route(RendererKind::Sprite),
                    source
                ),
                Err(ParticleRendererGpuError::InvalidWorkCapacity)
            ));
            assert_eq!(device.live_resource_count(), baseline);
        }
        let registry = [ParticleEmitterRendererConfig {
            emitter_id: 17,
            route_index: 0,
            renderer: route(RendererKind::Sprite),
            material: ParticleSurfaceMaterial::for_route(route(RendererKind::Sprite)),
        }];
        assert!(matches!(
            ParticleEmitterRendererGpu::new_with_budgets(&device, 257, source, &registry, &[]),
            Err(ParticleRendererGpuError::InvalidWorkCapacity)
        ));
        assert_eq!(device.live_resource_count(), baseline);
        let stage = ParticleRendererGpu::new_emitter_with_budget(
            &device,
            257,
            65,
            route(RendererKind::Sprite),
            source,
        )
        .expect("budgeted emitter");
        assert_eq!(stage.capacity(), 257);
        assert_eq!(stage.work_capacity(), 65);
        assert_eq!(stage.padded_capacity, 128);
        assert_eq!(stage.layout().visible_index_bytes, 65 * 4);
        assert_eq!(stage.sort_steps.len(), 28);
        stage
            .set_view(&queue, ParticleCullView::unrestricted())
            .expect("view");
        let recorded = stage.execute(&device, &queue).expect("budget graph");
        let mut work_passes = 0;
        for pass in &recorded.commands().passes {
            if let prism_render_driver::Pass::Compute { label, commands } = pass {
                let label = label.as_deref().unwrap_or("");
                if label == "particle-renderer-seed"
                    || label == "particle-renderer-cull"
                    || label == "particle-renderer-compact"
                    || label.starts_with("particle-renderer-sort-")
                {
                    assert!(
                        commands.iter().any(|command| matches!(
                            command,
                            rhi::ComputeCommand::Dispatch { x: 2, y: 1, z: 1 }
                        )),
                        "{label} must use budget, not global capacity"
                    );
                    work_passes += 1;
                }
            }
        }
        assert_eq!(work_passes, 31);
        recorded.release(&device);
        stage.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline);
        owner.destroy(&device);
        device.destroy_buffer(table);
    }

    #[test]
    fn emitter_registry_rejects_over_input_budgets_before_allocating_any_stage() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, _) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let renderer_route = route(RendererKind::Sprite);
        let owner = ParticleRendererGpu::new(&device, 8, renderer_route).expect("owner");
        let table = device.create_buffer(&BufferDescriptor::new(48, BufferUsages::STORAGE));
        let source = ParticleEmitterGpuSource {
            input: owner.input_buffer(),
            input_count: owner.input_count_buffer(),
            route_table: table,
            route_capacity: 2,
            route_index: 0,
            emitter_id: 0,
        };
        let registry =
            [(42, 0), (7, 1)].map(|(emitter_id, route_index)| ParticleEmitterRendererConfig {
                emitter_id,
                route_index,
                renderer: renderer_route,
                material: ParticleSurfaceMaterial::for_route(renderer_route),
            });
        let baseline = device.live_resource_count();
        for budgets in [[9, 1], [1, 9], [1, u32::MAX], [0, 1], [1, 0]] {
            assert!(
                matches!(
                    ParticleEmitterRendererGpu::new_with_budgets(
                        &device, 8, source, &registry, &budgets
                    ),
                    Err(ParticleRendererGpuError::InvalidWorkCapacity)
                ),
                "invalid budgets {budgets:?} must be rejected"
            );
            assert_eq!(
                device.live_resource_count(),
                baseline,
                "all budgets must be checked before allocating even the first emitter"
            );
        }
        let renderer =
            ParticleEmitterRendererGpu::new_with_budgets(&device, 8, source, &registry, &[1, 8])
                .expect("each budget may equal input capacity; their sum is not an ABI limit");
        assert_eq!(renderer.emitters()[1].work_capacity(), 8);
        renderer.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline);
        owner.destroy(&device);
        device.destroy_buffer(table);
    }

    #[test]
    fn emitter_registry_fans_out_shared_routes_without_aliasing_owned_work() {
        let instance = NullInstance::new();
        let adapter = instance.request_adapter(&Default::default()).unwrap();
        let (device, _) = instance
            .create_device(&adapter, &Default::default())
            .unwrap();
        let sprite = route(RendererKind::Sprite);
        let owner = ParticleRendererGpu::new(&device, 8, sprite).unwrap();
        let table = device.create_buffer(&BufferDescriptor::new(48, BufferUsages::STORAGE));
        let source = ParticleEmitterGpuSource {
            input: owner.input_buffer(),
            input_count: owner.input_count_buffer(),
            route_table: table,
            route_capacity: 2,
            route_index: u32::MAX,
            emitter_id: u32::MAX,
        };
        let config = ParticleEmitterRendererConfig {
            emitter_id: 42,
            route_index: 1,
            renderer: sprite,
            material: ParticleSurfaceMaterial::for_route(sprite),
        };
        let baseline = device.live_resource_count();
        for invalid in [
            ParticleEmitterRendererConfig {
                route_index: 2,
                ..config
            },
            ParticleEmitterRendererConfig {
                emitter_id: 7,
                ..config
            },
            ParticleEmitterRendererConfig {
                route_index: 0,
                ..config
            },
            ParticleEmitterRendererConfig {
                material: ParticleSurfaceMaterial {
                    softness: f32::NAN,
                    ..config.material
                },
                ..config
            },
        ] {
            assert!(matches!(
                ParticleEmitterRendererGpu::new_with_budgets(
                    &device,
                    8,
                    source,
                    &[config, invalid],
                    &[4, 4]
                ),
                Err(ParticleRendererGpuError::InvalidRouteTable)
            ));
            assert_eq!(
                device.live_resource_count(),
                baseline,
                "validate the entire registry before allocation"
            );
        }
        // Three consumers exceed route_capacity, and route ordinals deliberately
        // differ from draw ordinals. Repeated emitter/route pairs are valid.
        let mesh = route(RendererKind::Mesh);
        let registry = [
            config,
            ParticleEmitterRendererConfig {
                renderer: mesh,
                material: ParticleSurfaceMaterial {
                    tint: [2.0, 0.25, 0.0, 1.0],
                    ..config.material
                },
                ..config
            },
            ParticleEmitterRendererConfig {
                emitter_id: 7,
                route_index: 0,
                ..config
            },
        ];
        let stages =
            ParticleEmitterRendererGpu::new_with_budgets(&device, 8, source, &registry, &[4, 4, 8])
                .unwrap();
        for (index, stage) in stages.emitters().iter().enumerate() {
            let borrowed = stage.emitter_source.unwrap();
            assert_eq!(borrowed.route_index, registry[index].route_index);
            assert_eq!(borrowed.emitter_id, registry[index].emitter_id);
            assert_eq!(stage.input_buffer(), source.input);
            assert_eq!(stage.input_count_buffer(), source.input_count);
            assert_eq!(stage.route_table, table);
            assert_eq!(stage.material, registry[index].material);
            for prior in &stages.emitters()[..index] {
                assert_ne!(stage.control, prior.control);
                assert_ne!(stage.visible_indices, prior.visible_indices);
                assert_ne!(stage.surface_material, prior.surface_material);
                assert_ne!(stage.indexed_args, prior.indexed_args);
            }
        }
        stages.destroy(&device);
        assert_eq!(
            device.live_resource_count(),
            baseline,
            "consumers never destroy borrowed buffers"
        );
        owner.destroy(&device);
        device.destroy_buffer(table);
    }

    #[test]
    fn emitter_registry_preserves_disjoint_work_budgets_and_borrowed_input() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let renderer_route = route(RendererKind::Sprite);
        let owner = ParticleRendererGpu::new(&device, 257, renderer_route).expect("owner");
        let table = device.create_buffer(&BufferDescriptor::new(64, BufferUsages::STORAGE));
        let source = ParticleEmitterGpuSource {
            input: owner.input_buffer(),
            input_count: owner.input_count_buffer(),
            route_table: table,
            route_capacity: 3,
            // Registry construction must override both fields, not reuse them.
            route_index: 2,
            emitter_id: u32::MAX,
        };
        let registry = [(42, 0), (7, 1), (99, 2)].map(|(emitter_id, route_index)| {
            ParticleEmitterRendererConfig {
                emitter_id,
                route_index,
                renderer: renderer_route,
                material: ParticleSurfaceMaterial::for_route(renderer_route),
            }
        });
        let baseline = device.live_resource_count();
        // Unequal budgets cross both workgroup and sort-padding boundaries.
        // They partition one input pool: 1 + 65 + 191 == 257.
        let budgets = [1, 65, 191];
        let renderer =
            ParticleEmitterRendererGpu::new_with_budgets(&device, 257, source, &registry, &budgets)
                .expect("independent emitter budgets");
        let mut owned_work_buffers = Vec::new();
        for (index, stage) in renderer.emitters().iter().enumerate() {
            let budget = budgets[index];
            let padded = budget.next_power_of_two();
            assert_eq!(stage.capacity(), 257);
            assert_eq!(stage.work_capacity(), budget);
            assert_eq!(stage.padded_capacity, padded);
            assert_eq!(stage.layout().visible_index_bytes, u64::from(budget) * 4);
            assert_eq!(stage.input_buffer(), source.input);
            assert_eq!(stage.input_count_buffer(), source.input_count);
            assert_eq!(stage.route_table, table);
            assert!(!stage.owns_input);
            let emitter_source = stage.emitter_source.expect("registered emitter");
            assert_eq!(emitter_source.route_index, index as u32);
            assert_eq!(emitter_source.emitter_id, registry[index].emitter_id);
            for buffer in [
                stage.control,
                stage.sort_a,
                stage.sort_b,
                stage.visible_indices,
                stage.sprite_args,
                stage.indexed_args,
                stage.params,
                stage.route_params,
                stage.surface_material,
            ] {
                assert!(![source.input, source.input_count, table].contains(&buffer));
                assert!(
                    !owned_work_buffers.contains(&buffer),
                    "emitter work must not alias"
                );
                owned_work_buffers.push(buffer);
            }
            let levels = padded.trailing_zeros();
            assert_eq!(stage.sort_steps.len() as u32, levels * (levels + 1) / 2);
            stage
                .set_view(&queue, ParticleCullView::unrestricted())
                .expect("view");
            let recorded = stage.execute(&device, &queue).expect("budget graph");
            let mut work_passes = 0;
            for pass in &recorded.commands().passes {
                if let prism_render_driver::Pass::Compute { label, commands } = pass {
                    let label = label.as_deref().unwrap_or("");
                    let groups = if label == "particle-renderer-cull"
                        || label == "particle-renderer-compact"
                    {
                        budget.div_ceil(64)
                    } else if label == "particle-renderer-seed"
                        || label.starts_with("particle-renderer-sort-")
                    {
                        padded.div_ceil(64)
                    } else {
                        continue;
                    };
                    let dispatches: Vec<_> = commands
                        .iter()
                        .filter_map(|command| match command {
                            rhi::ComputeCommand::Dispatch { x, y, z } => Some((*x, *y, *z)),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(dispatches, [(groups, 1, 1)], "emitter {index}: {label}");
                    work_passes += 1;
                }
            }
            assert_eq!(work_passes, 3 + stage.sort_steps.len());
            recorded.release(&device);
        }
        renderer.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline);
        owner
            .set_inputs(&queue, &[], ParticleCullView::unrestricted())
            .expect("borrowed producer survives renderer destruction");
        owner.destroy(&device);
        device.destroy_buffer(table);
    }

    #[test]
    fn strip_quad_indices_cover_the_whole_quad_with_consistent_winding() {
        // Match strip_vertex's bit layout: start-/start+/end-/end+.
        let corners = [[0.0_f32, -1.0], [0.0, 1.0], [1.0, -1.0], [1.0, 1.0]];
        for kind in [RendererKind::Ribbon, RendererKind::Beam] {
            let indices = surface_index_bytes(route(kind));
            for (segment, quad) in indices.chunks_exact(24).enumerate() {
                let local: Vec<_> = quad
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()) - segment as u32 * 4)
                    .collect();
                let mut signed_area = 0.0;
                for triangle in local.chunks_exact(3) {
                    let [a, b, c] = [
                        corners[triangle[0] as usize],
                        corners[triangle[1] as usize],
                        corners[triangle[2] as usize],
                    ];
                    let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                    assert!(twice_area < 0.0, "strip triangles must share winding");
                    signed_area += twice_area * 0.5;
                }
                assert_eq!(signed_area, -2.0, "both halves cover exactly one quad");
            }
        }
    }

    #[test]
    fn invalid_particle_records_are_not_reported_as_invalid_views() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let stage = ParticleRendererGpu::new(&device, 4, route(RendererKind::Sprite))
            .expect("null stage resources");
        assert_eq!(
            stage.set_inputs(
                &queue,
                &[ParticleRenderInput::new([0.0; 3], -1.0, 0.5, 97)],
                ParticleCullView::unrestricted(),
            ),
            Err(ParticleRendererGpuError::InvalidInput)
        );
        assert_eq!(
            stage.set_view(
                &queue,
                ParticleCullView {
                    far_depth: 0.0,
                    ..ParticleCullView::unrestricted()
                }
            ),
            Err(ParticleRendererGpuError::InvalidView)
        );
        stage.destroy(&device);
    }

    #[test]
    fn null_rhi_records_the_complete_sort_cull_graph() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let stage = ParticleRendererGpu::new(&device, 7, route(RendererKind::Sprite))
            .expect("null stage resources");
        stage
            .set_inputs(
                &queue,
                &[ParticleRenderInput::new([0.0, 0.0, 0.0], 0.5, 0.2, 9)],
                ParticleCullView::unrestricted(),
            )
            .expect("valid input");
        let recorded = stage
            .execute(&device, &queue)
            .expect("graph executes on null RHI");
        assert!(recorded
            .commands()
            .passes
            .iter()
            .any(|pass| matches!(pass, prism_render_driver::Pass::Compute { .. })));
        stage.destroy(&device);
    }

    #[test]
    fn mesh_route_accepts_custom_geometry_and_rejects_invalid_geometry() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let mut stage = ParticleRendererGpu::new(
            &device,
            4,
            ParticleRendererRoute::new(
                RendererKind::Mesh,
                SortStrategy::ViewDepthBitonic,
                RendererGeometry::Mesh {
                    alive_count: 1,
                    index_count: 6,
                    first_instance: 0,
                },
            )
            .expect("mesh route"),
        )
        .expect("mesh stage");

        let vertices = [
            ParticleMeshVertex::new([-0.5, -0.5, -0.2], [0.0, 0.0]),
            ParticleMeshVertex::new([0.5, -0.5, 0.2], [1.0, 0.0]),
            ParticleMeshVertex::new([0.5, 0.5, 0.2], [1.0, 1.0]),
            ParticleMeshVertex::new([-0.5, 0.5, -0.2], [0.0, 1.0]),
        ];
        stage
            .set_mesh_geometry(&queue, &vertices, &[0, 1, 2, 0, 2, 3])
            .expect("quad geometry is accepted");
        assert_eq!(
            stage.set_mesh_geometry(&queue, &vertices, &[0, 1, 2]),
            Err(ParticleRendererGpuError::MeshGeometryTooLarge)
        );
        assert_eq!(
            stage.set_mesh_geometry(&queue, &vertices, &[0, 1, 9, 0, 2, 3]),
            Err(ParticleRendererGpuError::InvalidMeshGeometry)
        );
        stage.destroy(&device);
    }
}
