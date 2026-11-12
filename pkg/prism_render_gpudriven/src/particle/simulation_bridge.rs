//! GPU-only handoff from the Ember simulation pool to the particle renderer.
//!
//! The bridge is deliberately a graph contract, not a CPU extraction layer.
//! It consumes the SSA handles published by [`GpuSimulationStage`] and writes
//! the renderer's 48-byte `ParticleRenderInput` ABI into the persistent input
//! buffer owned by [`ParticleRendererGpu`]. No particle record is read back and
//! no simulation pool is uploaded again between the simulation and renderer
//! passes. Per-emitter packed positions are intentionally not ordered: the
//! scatter pass reserves slots with atomics. `particle_index` is the original
//! simulation-pool index, not a birth-order or Ribbon topology key. Explicit
//! RibbonId/birth metadata is forwarded from the 64-byte simulation record;
//! unavailable topology stays invalid, never inferred from pool/scatter order.

use alloc::{format, string::String, vec::Vec};

use super::renderer_matrix::{
    ParticleRendererGpu, ParticleRendererRoute, PARTICLE_RENDER_INPUT_BYTES,
};
use crate::gpu::{bind_buffer, buffer_entry};
use crate::gpu_simulation::{
    GpuSimulationGraphOutputs, GpuSimulationStage, CONTROL_WORDS, GPU_EVENT_STRIDE,
    GPU_PARTICLE_STRIDE,
};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferState, BufferUsages,
    ComputePipelineDescriptor, ComputePipelineId, PipelineLayoutDescriptor, PipelineLayoutId,
    PipelineStages, RenderDevice, RenderQueue, ShaderModuleDescriptor, ShaderModuleId,
    ShaderStages,
};
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ExecuteContext, ImportedBuffer, PassFlags, PassKind,
    RenderGraph,
};

/// The bridge output ABI consumed by the renderer matrix.
pub const PARTICLE_RENDER_INPUT_STRIDE: u64 = PARTICLE_RENDER_INPUT_BYTES;
/// One event flag word stored per simulation-pool slot.
pub const PARTICLE_EVENT_FLAG_STRIDE: u64 = 4;
/// Bytes in one GPU emitter route record.
pub const PARTICLE_EMITTER_ROUTE_STRIDE: u64 = 16;
/// Header bytes preceding the fixed-capacity route records.
pub const PARTICLE_EMITTER_ROUTE_HEADER_BYTES: u64 = 16;
/// The bridge writes this flag when an active index violates the lifetime ABI.
pub const PARTICLE_BRIDGE_FLAG_INVALID_LIFETIME: u32 = 1;
/// The bridge writes this flag when the particle produced at least one event.
pub const PARTICLE_BRIDGE_FLAG_EVENT: u32 = 2;
const BRIDGE_WORKGROUP_SIZE: u32 = 64;
const BRIDGE_STORAGE_BUFFER_BINDINGS: u32 = 11;
const VALIDATION_WORDS: u64 = 4;
const VALIDATION_BYTES: u64 = VALIDATION_WORDS * 4;

/// One registered emitter and its renderer plan.
///
/// `active_start` and `particle_count` are retained as CPU diagnostics only.
/// They are never uploaded as authoritative ranges: every bridge execution
/// clears and rewrites the corresponding GPU route record from the compacted
/// simulation output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleEmitterRenderRoute {
    /// Stable emitter identifier supplied by the caller.
    pub emitter_id: u32,
    /// Non-authoritative CPU diagnostic for the previous/extracted range.
    pub active_start: u32,
    /// Non-authoritative CPU diagnostic for the previous/extracted count.
    pub particle_count: u32,
    /// Renderer geometry and sorting contract for this segment.
    pub renderer: ParticleRendererRoute,
}

/// Errors detected before a bridge graph pass is admitted or a bridge buffer
/// is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleSimulationBridgeError {
    /// The simulation pool capacity must be nonzero.
    ZeroCapacity,
    /// The event pool capacity must be nonzero.
    ZeroEventCapacity,
    /// A capacity multiplication overflowed the ABI footprint.
    CapacityOverflow,
    /// A fixed ABI stride did not match the bridge contract.
    InvalidStride {
        /// Name of the ABI record.
        resource: &'static str,
        /// Required byte stride.
        expected: u64,
        /// Caller-provided byte stride.
        actual: u64,
    },
    /// One bridge resource exceeds the device storage-buffer limit.
    BufferTooLarge {
        /// Resource name used in diagnostics.
        resource: &'static str,
        /// Requested bytes.
        requested: u64,
        /// Device limit.
        limit: u64,
    },
    /// A binding or compute-workgroup device limit is too small.
    BindingLimitTooSmall {
        /// Required binding count.
        requested: u32,
        /// Device limit.
        limit: u32,
    },
    /// The compute stage cannot expose all eleven bridge storage buffers.
    StorageBufferStageLimitTooSmall {
        /// Required storage-buffer bindings across the entire compute stage.
        requested: u32,
        /// Enabled logical-device limit, not the adapter's possible maximum.
        limit: u32,
    },
    /// A bridge dispatch exceeds the device's workgroup limit.
    DispatchTooLarge {
        /// Requested X workgroups.
        workgroups: u32,
        /// Device limit.
        limit: u32,
    },
    /// The emitter registry cannot exceed the fixed simulation capacity.
    TooManyEmitterRoutes {
        /// Number of registered emitters.
        requested: usize,
        /// Fixed route-table capacity.
        capacity: u32,
    },
    /// The renderer and simulation must share the same output capacity.
    RendererCapacityMismatch {
        /// Simulation pool capacity.
        simulation: u32,
        /// Renderer input capacity.
        renderer: u32,
    },
    /// Emitter identifiers must be unique in one bridge plan.
    DuplicateEmitterId { emitter_id: u32 },
    /// A live simulation record violates the age/lifetime ABI.
    InvalidLifetime,
}

/// A physical bridge resource and its pass-local access direction.
///
/// Fabric import/admission uses this manifest to authenticate the exact
/// `BufferId` and byte range before a recorder is invoked. The bridge never
/// allocates or resolves graph handles while producing this manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParticleBridgePhysicalBuffer {
    /// Stable semantic role of the physical buffer.
    pub role: ParticleBridgeBufferRole,
    /// Exact physical buffer referenced by the bridge bind group.
    pub buffer: BufferId,
    /// Byte range required by the bridge ABI.
    pub bytes: u64,
    /// Access required by the phase exposing this resource.
    pub access: ParticleBridgeBufferAccess,
}

/// Physical buffers consumed or produced by a bridge phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleBridgeBufferRole {
    /// Simulation particle pool.
    SimulationParticles,
    /// Compacted simulation active-index list.
    SimulationActiveIndices,
    /// Compacted simulation event records.
    SimulationCompactEvents,
    /// Simulation control words.
    SimulationControl,
    /// Per-pool-slot event flags.
    EventFlags,
    /// Renderer ABI input records.
    RendererInput,
    /// GPU-authored renderer input count.
    RendererInputCount,
    /// Validation counters.
    Validation,
    /// Emitter route table.
    EmitterRouteTable,
    /// Route prefix-scan scratch.
    RouteScratch,
    /// Active route indices cached by count-routes.
    ActiveRouteIndices,
}

/// The access direction a bridge phase requires for a physical buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleBridgeBufferAccess {
    /// The phase reads the buffer.
    Read,
    /// The phase writes the buffer.
    Write,
}

/// One executable compute phase in the bridge record plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParticleBridgeRecordPhase {
    kind: ParticleBridgeRecordPhaseKind,
    pipeline: ComputePipelineId,
    bind_group: BindGroupId,
    dispatch: [u32; 3],
    resources: Vec<ParticleBridgePhysicalBuffer>,
}

impl ParticleBridgeRecordPhase {
    /// Stable phase identity, including the hierarchy level where applicable.
    #[must_use]
    pub const fn kind(&self) -> ParticleBridgeRecordPhaseKind {
        self.kind
    }

    /// Compute pipeline selected for this phase.
    #[must_use]
    pub const fn pipeline(&self) -> ComputePipelineId {
        self.pipeline
    }

    /// Bind group selected for this phase.
    #[must_use]
    pub const fn bind_group(&self) -> BindGroupId {
        self.bind_group
    }

    /// Dispatch dimensions recorded for this phase.
    #[must_use]
    pub const fn dispatch(&self) -> [u32; 3] {
        self.dispatch
    }

    /// Exact physical buffers and byte ranges required by this phase.
    #[must_use]
    pub fn resources(&self) -> &[ParticleBridgePhysicalBuffer] {
        &self.resources
    }
}

/// Stable phase identity matching the bridge's existing graph lowering order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParticleBridgeRecordPhaseKind {
    /// Clears bridge outputs and route-scan state.
    Clear,
    /// Marks compacted simulation events for gather.
    MarkEvents,
    /// Counts active particles per registered emitter.
    CountRoutes,
    /// Writes route prefix metadata and renderer input count.
    PrefixRoutes,
    /// Scans one route-prefix hierarchy level.
    PrefixSums(u32),
    /// Propagates one route-prefix hierarchy level.
    AddOffsets(u32),
    /// Finalizes route starts/counts after the hierarchy scan.
    FinalizeRoutes,
    /// Scatters simulation records into the renderer input ABI.
    Scatter,
}

/// Recording error returned before any command is emitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleBridgeRecordError {
    /// The caller selected a phase that is not present in this plan.
    InvalidPhase { index: usize },
    /// The plan can only record into a compute pass.
    WrongPassKind(PassKind),
}

/// Render Fabric-facing, graph-free command plan for one bridge instance.
///
/// The plan contains only persistent physical IDs, byte ranges, pipeline state,
/// route metadata and dispatch dimensions. It does not create, compile, import,
/// or prepare a [`RenderGraph`]. Call [`Self::record_phase`] from the selected
/// Fabric recorder after the host graph has authenticated the manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSimulationBridgeRecordPlan {
    capacity: u32,
    event_capacity: u32,
    route_capacity: u32,
    emitter_routes: Vec<ParticleEmitterRenderRoute>,
    phases: Vec<ParticleBridgeRecordPhase>,
}

impl ParticleSimulationBridgeRecordPlan {
    /// Simulation pool capacity covered by this plan.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Event pool capacity covered by this plan.
    #[must_use]
    pub const fn event_capacity(&self) -> u32 {
        self.event_capacity
    }

    /// Allocated route-table record capacity.
    #[must_use]
    pub const fn route_capacity(&self) -> u32 {
        self.route_capacity
    }

    /// Current route metadata in the exact route-table order.
    #[must_use]
    pub fn emitter_routes(&self) -> &[ParticleEmitterRenderRoute] {
        &self.emitter_routes
    }

    /// Ordered executable phases. The order is the dependency order used by
    /// the former graph lowering, including every hierarchical scan phase.
    #[must_use]
    pub fn phases(&self) -> &[ParticleBridgeRecordPhase] {
        &self.phases
    }

    /// Records one selected phase into an existing compute pass.
    ///
    /// This method emits only the concrete pipeline, bind group and dispatch;
    /// it never creates a graph pass or touches resource ownership.
    pub fn record_phase(
        &self,
        index: usize,
        context: &mut ExecuteContext<'_>,
    ) -> Result<(), ParticleBridgeRecordError> {
        if context.kind() != PassKind::Compute {
            return Err(ParticleBridgeRecordError::WrongPassKind(context.kind()));
        }
        let phase = self
            .phases
            .get(index)
            .ok_or(ParticleBridgeRecordError::InvalidPhase { index })?;
        context
            .set_compute_pipeline(phase.pipeline)
            .set_compute_bind_group(0, phase.bind_group, Vec::new())
            .dispatch(
                phase.dispatch[0],
                phase.dispatch[1],
                phase.dispatch[2],
            );
        Ok(())
    }

    /// Records one selected phase through a Render Fabric runtime context.
    ///
    /// The runtime context remains responsible for graph-declared accesses;
    /// this seam supplies only the command recording portion.
    pub fn record_runtime_phase(
        &self,
        index: usize,
        context: &mut prism_render_architecture::render_fabric::RuntimePassContext<'_, '_>,
    ) -> Result<(), ParticleBridgeRecordError> {
        self.record_phase(index, context.commands())
    }
}

/// Graph handles authored by the bridge.
#[derive(Clone, Debug)]
pub struct ParticleSimulationBridgeGraphOutputs {
    /// The renderer input handle written by the GPU scatter pass.
    pub renderer_input: BufferHandle,
    /// GPU-authored live record count consumed by renderer culling.
    pub renderer_input_count: BufferHandle,
    /// Per-pool-slot event flags consumed by the gather pass.
    pub event_flags: BufferHandle,
    /// Four saturating GPU diagnostic counters: invalid records, unknown
    /// emitters, truncated active entries, and simulation event overflow.
    /// Event flags are conservatively suppressed on event overflow because
    /// exact compacted channel lengths are not published in that case.
    pub validation: BufferHandle,
    /// The compact event handle consumed by the event-mark pass and retained
    /// here so downstream graph consumers can prove the binding contract.
    pub compact_events: BufferHandle,
    /// GPU-authored route table: four header u32s (registered count, three
    /// reserved zeros), followed by 16-byte records of four u32s:
    /// `(active_start, particle_count, emitter_id, route_index)`.
    /// Record order and `route_index` match [`Self::emitter_routes`]. Ranges
    /// address `renderer_input`, not the simulation active-index buffer.
    /// All values use the little-endian GPU buffer ABI; starts/counts are
    /// authoritative for the frame, including zero-count emitters. Input
    /// records inside a segment have unspecified physical order; consumers
    /// needing stable topology must consume explicit RibbonId/birth metadata,
    /// not pool indices or scatter order. The simulation's explicit topology
    /// is forwarded unchanged; missing metadata has flags zero (no strip).
    pub emitter_route_table: BufferHandle,
    /// CPU emitter registry and renderer metadata. Its range fields are not
    /// authoritative; consume `emitter_route_table` for frame ranges.
    pub emitter_routes: Vec<ParticleEmitterRenderRoute>,
}

/// Persistent GPU resources for the simulation-to-renderer handoff.
pub struct ParticleSimulationRendererBridge {
    capacity: u32,
    event_capacity: u32,
    simulation_particles: BufferId,
    simulation_active_indices: BufferId,
    simulation_compact_events: BufferId,
    simulation_control: BufferId,
    renderer_input: BufferId,
    event_flags: BufferId,
    validation: BufferId,
    renderer_input_count: BufferId,
    shader: ShaderModuleId,
    layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    clear_pipeline: ComputePipelineId,
    mark_events_pipeline: ComputePipelineId,
    count_routes_pipeline: ComputePipelineId,
    prefix_routes_pipeline: ComputePipelineId,
    scan_pipelines: Vec<ComputePipelineId>,
    add_offset_pipelines: Vec<ComputePipelineId>,
    finalize_routes_pipeline: ComputePipelineId,
    gather_pipeline: ComputePipelineId,
    bind_group: BindGroupId,
    route_table: BufferId,
    route_scratch: BufferId,
    route_scratch_words: u32,
    active_route_indices: BufferId,
    route_capacity: u32,
    emitter_routes: Vec<ParticleEmitterRenderRoute>,
}

impl ParticleSimulationRendererBridge {
    /// Validates the immutable ABI and unique emitter registry. CPU range
    /// fields deliberately do not participate in validation or GPU routing.
    /// Route-table capacity is exactly the simulation pool capacity `C`, with
    /// one 16-byte record per registered emitter and a 16-byte header. Thus
    /// `0 <= registered_count <= C`; duplicate IDs and excess routes are
    /// rejected before any resource allocation or queue write.
    pub fn validate_contract(
        capacity: u32,
        event_capacity: u32,
        simulation_particle_stride: u64,
        renderer_input_stride: u64,
        event_stride: u64,
        emitter_routes: &[ParticleEmitterRenderRoute],
    ) -> Result<(), ParticleSimulationBridgeError> {
        if capacity == 0 {
            return Err(ParticleSimulationBridgeError::ZeroCapacity);
        }
        if event_capacity == 0 {
            return Err(ParticleSimulationBridgeError::ZeroEventCapacity);
        }
        for (resource, expected, actual) in [
            (
                "simulation particle",
                GPU_PARTICLE_STRIDE,
                simulation_particle_stride,
            ),
            (
                "renderer input",
                PARTICLE_RENDER_INPUT_STRIDE,
                renderer_input_stride,
            ),
            ("compact event", GPU_EVENT_STRIDE, event_stride),
        ] {
            if expected != actual {
                return Err(ParticleSimulationBridgeError::InvalidStride {
                    resource,
                    expected,
                    actual,
                });
            }
        }

        if emitter_routes.len() > capacity as usize {
            return Err(ParticleSimulationBridgeError::TooManyEmitterRoutes {
                requested: emitter_routes.len(),
                capacity,
            });
        }
        let mut emitter_ids: Vec<_> = emitter_routes
            .iter()
            .map(|route| route.emitter_id)
            .collect();
        emitter_ids.sort_unstable();
        for pair in emitter_ids.windows(2) {
            if pair[0] == pair[1] {
                return Err(ParticleSimulationBridgeError::DuplicateEmitterId {
                    emitter_id: pair[0],
                });
            }
        }
        Ok(())
    }

    /// Validates one active simulation record's lifetime semantics.
    ///
    /// The simulation ABI treats zero lifetime as immortal. A positive
    /// lifetime requires `0 <= age < lifetime`; non-finite values are never
    /// renderer-visible.
    pub fn validate_lifetime(
        alive: bool,
        age: f32,
        lifetime: f32,
    ) -> Result<(), ParticleSimulationBridgeError> {
        if !alive
            || !age.is_finite()
            || age < 0.0
            || !lifetime.is_finite()
            || lifetime < 0.0
            || (lifetime > 0.0 && age >= lifetime)
        {
            return Err(ParticleSimulationBridgeError::InvalidLifetime);
        }
        Ok(())
    }

    /// Creates the bridge and binds it to a renderer's persistent input
    /// buffer. The simulation stage remains the owner of its four input
    /// buffers; this bridge only references them through the stage's resource
    /// accessors and never destroys them. Call [`Self::upload_routes`] or
    /// [`Self::set_emitter_routes`] before the first submission: newly created
    /// GPU buffers contain an empty registry until that explicit queue upload.
    pub fn new<D: RenderDevice + ?Sized>(
        device: &D,
        simulation: &GpuSimulationStage,
        renderer: &ParticleRendererGpu,
        emitter_routes: &[ParticleEmitterRenderRoute],
    ) -> Result<Self, ParticleSimulationBridgeError> {
        let capacity = simulation.capacity();
        let event_capacity = simulation.event_capacity();
        Self::validate_contract(
            capacity,
            event_capacity,
            GPU_PARTICLE_STRIDE,
            PARTICLE_RENDER_INPUT_STRIDE,
            GPU_EVENT_STRIDE,
            emitter_routes,
        )?;
        if renderer.capacity() != capacity {
            return Err(ParticleSimulationBridgeError::RendererCapacityMismatch {
                simulation: capacity,
                renderer: renderer.capacity(),
            });
        }

        let limits = device.capabilities().limits;
        // Group binding counts do not imply a per-stage storage budget. Check
        // the enabled device limit before allocating any resource or pipeline.
        if limits.max_storage_buffers_per_shader_stage < BRIDGE_STORAGE_BUFFER_BINDINGS {
            return Err(
                ParticleSimulationBridgeError::StorageBufferStageLimitTooSmall {
                    requested: BRIDGE_STORAGE_BUFFER_BINDINGS,
                    limit: limits.max_storage_buffers_per_shader_stage,
                },
            );
        }
        if limits.max_bind_groups < 1 {
            return Err(ParticleSimulationBridgeError::BindingLimitTooSmall {
                requested: 1,
                limit: limits.max_bind_groups,
            });
        }
        if limits.max_bindings_per_bind_group < BRIDGE_STORAGE_BUFFER_BINDINGS {
            return Err(ParticleSimulationBridgeError::BindingLimitTooSmall {
                requested: BRIDGE_STORAGE_BUFFER_BINDINGS,
                limit: limits.max_bindings_per_bind_group,
            });
        }
        if limits.max_compute_invocations_per_workgroup < BRIDGE_WORKGROUP_SIZE {
            return Err(ParticleSimulationBridgeError::BindingLimitTooSmall {
                requested: BRIDGE_WORKGROUP_SIZE,
                limit: limits.max_compute_invocations_per_workgroup,
            });
        }
        let event_words = event_capacity
            .checked_mul(4)
            .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?;
        let scan_counts = scan_level_counts(capacity);
        let route_scratch_words = scan_counts.iter().try_fold(
            capacity
                .checked_mul(3)
                .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            |words, count| {
                words
                    .checked_add(*count)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)
            },
        )?;
        let route_table_size = PARTICLE_EMITTER_ROUTE_HEADER_BYTES
            .checked_add(
                u64::from(capacity)
                    .checked_mul(PARTICLE_EMITTER_ROUTE_STRIDE)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            )
            .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?;
        for (resource, size) in [
            (
                "simulation particles",
                u64::from(capacity)
                    .checked_mul(GPU_PARTICLE_STRIDE)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            ),
            (
                "active indices",
                u64::from(capacity)
                    .checked_mul(4)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            ),
            (
                "compact events",
                u64::from(event_words)
                    .checked_mul(GPU_EVENT_STRIDE)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            ),
            (
                "event flags",
                u64::from(capacity)
                    .checked_mul(PARTICLE_EVENT_FLAG_STRIDE)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            ),
            (
                "renderer input",
                u64::from(capacity)
                    .checked_mul(PARTICLE_RENDER_INPUT_STRIDE)
                    .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
            ),
            ("emitter route table", route_table_size),
            ("emitter route scratch", u64::from(route_scratch_words) * 4),
            ("active route indices", u64::from(capacity) * 4),
        ] {
            let limit = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            if size > limit {
                return Err(ParticleSimulationBridgeError::BufferTooLarge {
                    resource,
                    requested: size,
                    limit,
                });
            }
        }
        let capacity_groups = capacity.div_ceil(BRIDGE_WORKGROUP_SIZE);
        if capacity_groups > limits.max_compute_workgroups_per_dimension {
            return Err(ParticleSimulationBridgeError::DispatchTooLarge {
                workgroups: capacity_groups,
                limit: limits.max_compute_workgroups_per_dimension,
            });
        }
        let event_groups = event_words.div_ceil(BRIDGE_WORKGROUP_SIZE);
        if event_groups > limits.max_compute_workgroups_per_dimension {
            return Err(ParticleSimulationBridgeError::DispatchTooLarge {
                workgroups: event_groups,
                limit: limits.max_compute_workgroups_per_dimension,
            });
        }

        let event_flags = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * PARTICLE_EVENT_FLAG_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let validation = device.create_buffer(&BufferDescriptor::new(
            VALIDATION_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let route_table = device.create_buffer(&BufferDescriptor::new(
            route_table_size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        ));
        let route_scratch = device.create_buffer(&BufferDescriptor::new(
            u64::from(route_scratch_words) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let active_route_indices = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * 4,
            BufferUsages::STORAGE,
        ));
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some(String::from("particle-simulation-renderer-bridge")),
            source: prism_render_driver::ShaderSource::Wgsl(bridge_shader_source(capacity).into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-simulation-renderer-bridge-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    3,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    4,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    5,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    6,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    7,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    8,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    9,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    10,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-simulation-renderer-bridge-pl")),
            bind_group_layouts: vec![layout],
            push_constant_ranges: Vec::new(),
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-simulation-renderer-bridge-bg")),
            layout: Some(layout),
            entries: vec![
                bind_buffer(0, simulation.particles()),
                bind_buffer(1, simulation.active_indices()),
                bind_buffer(2, simulation.compact_events()),
                bind_buffer(3, simulation.control()),
                bind_buffer(4, event_flags),
                bind_buffer(5, renderer.input_buffer()),
                bind_buffer(6, validation),
                bind_buffer(7, route_table),
                bind_buffer(8, renderer.input_count_buffer()),
                bind_buffer(9, route_scratch),
                bind_buffer(10, active_route_indices),
            ],
        });
        let clear_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from("particle-simulation-renderer-bridge-clear")),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "clear".into(),
        });
        let mark_events_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from(
                "particle-simulation-renderer-bridge-mark-events",
            )),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "mark_events".into(),
        });
        let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from("particle-simulation-renderer-bridge-gather")),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "scatter".into(),
        });
        let create_pipeline = |entry: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(format!("particle-simulation-renderer-bridge-{entry}")),
                layout: Some(pipeline_layout),
                module: shader,
                entry_point: String::from(entry).into(),
            })
        };
        let count_routes_pipeline = create_pipeline("count_routes");
        let prefix_routes_pipeline = create_pipeline("prefix_routes");
        let finalize_routes_pipeline = create_pipeline("finalize_routes");
        let scan_pipelines = (0..scan_counts.len())
            .map(|level| create_pipeline(&format!("prefix_sums_{level}")))
            .collect();
        let add_offset_pipelines = (0..scan_counts.len())
            .map(|level| create_pipeline(&format!("add_offsets_{level}")))
            .collect();
        Ok(Self {
            capacity,
            event_capacity,
            simulation_particles: simulation.particles(),
            simulation_active_indices: simulation.active_indices(),
            simulation_compact_events: simulation.compact_events(),
            simulation_control: simulation.control(),
            renderer_input: renderer.input_buffer(),
            event_flags,
            validation,
            renderer_input_count: renderer.input_count_buffer(),
            shader,
            layout,
            pipeline_layout,
            clear_pipeline,
            mark_events_pipeline,
            count_routes_pipeline,
            prefix_routes_pipeline,
            scan_pipelines,
            add_offset_pipelines,
            finalize_routes_pipeline,
            gather_pipeline,
            bind_group,
            route_table,
            route_scratch,
            route_scratch_words,
            active_route_indices,
            route_capacity: capacity,
            emitter_routes: emitter_routes.to_vec(),
        })
    }

    /// Replaces the emitter registry consumed by GPU classification.
    ///
    /// The allocation remains fixed for the simulation capacity, while the
    /// header carries the registered route count. CPU starts/counts are never
    /// uploaded. A sorted emitter-id lookup retains the caller's original
    /// route ordinals, permitting O(log emitters) classification per particle.
    /// Queue updates must precede graph construction/submission for that frame.
    /// One bridge instance owns mutable scratch and output buffers: do not
    /// overlap independent executions of the same instance. Keep each graph's
    /// recorded resources alive until its GPU work completes.
    pub fn set_emitter_routes(
        &mut self,
        queue: &impl RenderQueue,
        emitter_routes: &[ParticleEmitterRenderRoute],
    ) -> Result<(), ParticleSimulationBridgeError> {
        Self::validate_contract(
            self.capacity,
            self.event_capacity,
            GPU_PARTICLE_STRIDE,
            PARTICLE_RENDER_INPUT_STRIDE,
            GPU_EVENT_STRIDE,
            emitter_routes,
        )?;
        let route_bytes = usize::try_from(
            PARTICLE_EMITTER_ROUTE_HEADER_BYTES
                .checked_add(
                    u64::from(self.route_capacity)
                        .checked_mul(PARTICLE_EMITTER_ROUTE_STRIDE)
                        .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
                )
                .ok_or(ParticleSimulationBridgeError::CapacityOverflow)?,
        )
        .map_err(|_| ParticleSimulationBridgeError::CapacityOverflow)?;
        let mut bytes = vec![
            0;
            PARTICLE_EMITTER_ROUTE_HEADER_BYTES as usize
                + emitter_routes.len() * PARTICLE_EMITTER_ROUTE_STRIDE as usize
        ];
        bytes[0..4].copy_from_slice(&(emitter_routes.len() as u32).to_le_bytes());
        for (route_index, route) in emitter_routes.iter().enumerate() {
            let offset = PARTICLE_EMITTER_ROUTE_HEADER_BYTES as usize
                + route_index * PARTICLE_EMITTER_ROUTE_STRIDE as usize;
            bytes[offset + 8..offset + 12].copy_from_slice(&route.emitter_id.to_le_bytes());
            bytes[offset + 12..offset + 16].copy_from_slice(&(route_index as u32).to_le_bytes());
        }
        debug_assert!(bytes.len() <= route_bytes);
        let mut lookup: Vec<_> = emitter_routes
            .iter()
            .enumerate()
            .map(|(index, route)| (route.emitter_id, index as u32))
            .collect();
        lookup.sort_unstable_by_key(|entry| entry.0);
        let mut lookup_bytes = Vec::with_capacity(lookup.len() * 8);
        for (emitter_id, route_index) in lookup {
            lookup_bytes.extend_from_slice(&emitter_id.to_le_bytes());
            lookup_bytes.extend_from_slice(&route_index.to_le_bytes());
        }
        queue.write_buffer(self.route_table, 0, &bytes);
        if !lookup_bytes.is_empty() {
            queue.write_buffer(
                self.route_scratch,
                u64::from(self.capacity) * 4,
                &lookup_bytes,
            );
        }
        self.emitter_routes.clear();
        self.emitter_routes.extend_from_slice(emitter_routes);
        Ok(())
    }

    /// Re-uploads the currently selected routes after a queue/device reset.
    pub fn upload_routes(
        &mut self,
        queue: &impl RenderQueue,
    ) -> Result<(), ParticleSimulationBridgeError> {
        let routes = self.emitter_routes.clone();
        self.set_emitter_routes(queue, &routes)
    }

    /// Builds the graph-free command plan consumed by a native Render Fabric
    /// recorder. The plan is tied to this bridge's persistent bind group and
    /// therefore cannot be pointed at a different simulation generation.
    #[must_use]
    pub fn record_plan(&self) -> ParticleSimulationBridgeRecordPlan {
        let capacity_groups = self.capacity.div_ceil(BRIDGE_WORKGROUP_SIZE);
        let event_groups = self
            .event_capacity
            .saturating_mul(4)
            .div_ceil(BRIDGE_WORKGROUP_SIZE);
        let route_groups = (self.emitter_routes.len() as u32)
            .div_ceil(BRIDGE_WORKGROUP_SIZE)
            .max(1);
        let mut phases = Vec::with_capacity(8 + scan_level_counts(self.emitter_routes.len() as u32).len() * 2);
        phases.push(ParticleBridgeRecordPhase {
            kind: ParticleBridgeRecordPhaseKind::Clear,
            pipeline: self.clear_pipeline,
            bind_group: self.bind_group,
            dispatch: [capacity_groups, 1, 1],
            resources: self.phase_resources(&[
                (ParticleBridgeBufferRole::EventFlags, ParticleBridgeBufferAccess::Write),
                (ParticleBridgeBufferRole::RendererInput, ParticleBridgeBufferAccess::Write),
                (
                    ParticleBridgeBufferRole::RendererInputCount,
                    ParticleBridgeBufferAccess::Write,
                ),
                (ParticleBridgeBufferRole::Validation, ParticleBridgeBufferAccess::Write),
                (
                    ParticleBridgeBufferRole::EmitterRouteTable,
                    ParticleBridgeBufferAccess::Write,
                ),
                (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Write),
            ]),
        });
        phases.push(ParticleBridgeRecordPhase {
            kind: ParticleBridgeRecordPhaseKind::MarkEvents,
            pipeline: self.mark_events_pipeline,
            bind_group: self.bind_group,
            dispatch: [event_groups, 1, 1],
            resources: self.phase_resources(&[
                (
                    ParticleBridgeBufferRole::SimulationCompactEvents,
                    ParticleBridgeBufferAccess::Read,
                ),
                (
                    ParticleBridgeBufferRole::SimulationControl,
                    ParticleBridgeBufferAccess::Read,
                ),
                (ParticleBridgeBufferRole::EventFlags, ParticleBridgeBufferAccess::Write),
            ]),
        });
        phases.push(ParticleBridgeRecordPhase {
            kind: ParticleBridgeRecordPhaseKind::CountRoutes,
            pipeline: self.count_routes_pipeline,
            bind_group: self.bind_group,
            dispatch: [capacity_groups, 1, 1],
            resources: self.phase_resources(&[
                (
                    ParticleBridgeBufferRole::SimulationParticles,
                    ParticleBridgeBufferAccess::Read,
                ),
                (
                    ParticleBridgeBufferRole::SimulationActiveIndices,
                    ParticleBridgeBufferAccess::Read,
                ),
                (
                    ParticleBridgeBufferRole::SimulationControl,
                    ParticleBridgeBufferAccess::Read,
                ),
                (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Read),
                (
                    ParticleBridgeBufferRole::EmitterRouteTable,
                    ParticleBridgeBufferAccess::Write,
                ),
                (
                    ParticleBridgeBufferRole::ActiveRouteIndices,
                    ParticleBridgeBufferAccess::Write,
                ),
                (ParticleBridgeBufferRole::Validation, ParticleBridgeBufferAccess::Write),
            ]),
        });
        phases.push(ParticleBridgeRecordPhase {
            kind: ParticleBridgeRecordPhaseKind::PrefixRoutes,
            pipeline: self.prefix_routes_pipeline,
            bind_group: self.bind_group,
            dispatch: [route_groups, 1, 1],
            resources: self.phase_resources(&[
                (
                    ParticleBridgeBufferRole::EmitterRouteTable,
                    ParticleBridgeBufferAccess::Write,
                ),
                (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Write),
                (
                    ParticleBridgeBufferRole::RendererInputCount,
                    ParticleBridgeBufferAccess::Write,
                ),
            ]),
        });

        let active_scan_counts = scan_level_counts(self.emitter_routes.len() as u32);
        for (level, count) in active_scan_counts.iter().copied().enumerate() {
            phases.push(ParticleBridgeRecordPhase {
                kind: ParticleBridgeRecordPhaseKind::PrefixSums(level as u32),
                pipeline: self.scan_pipelines[level],
                bind_group: self.bind_group,
                dispatch: [count.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1],
                resources: self.phase_resources(&[
                    (
                        ParticleBridgeBufferRole::EmitterRouteTable,
                        ParticleBridgeBufferAccess::Read,
                    ),
                    (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Write),
                    (
                        ParticleBridgeBufferRole::RendererInputCount,
                        ParticleBridgeBufferAccess::Write,
                    ),
                ]),
            });
        }
        for level in (0..active_scan_counts.len().saturating_sub(1)).rev() {
            let count = active_scan_counts[level];
            phases.push(ParticleBridgeRecordPhase {
                kind: ParticleBridgeRecordPhaseKind::AddOffsets(level as u32),
                pipeline: self.add_offset_pipelines[level],
                bind_group: self.bind_group,
                dispatch: [count.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1],
                resources: self.phase_resources(&[
                    (
                        ParticleBridgeBufferRole::EmitterRouteTable,
                        ParticleBridgeBufferAccess::Read,
                    ),
                    (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Write),
                ]),
            });
        }
        phases.push(ParticleBridgeRecordPhase {
            kind: ParticleBridgeRecordPhaseKind::FinalizeRoutes,
            pipeline: self.finalize_routes_pipeline,
            bind_group: self.bind_group,
            dispatch: [route_groups, 1, 1],
            resources: self.phase_resources(&[
                (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Read),
                (
                    ParticleBridgeBufferRole::EmitterRouteTable,
                    ParticleBridgeBufferAccess::Write,
                ),
            ]),
        });
        phases.push(ParticleBridgeRecordPhase {
            kind: ParticleBridgeRecordPhaseKind::Scatter,
            pipeline: self.gather_pipeline,
            bind_group: self.bind_group,
            dispatch: [capacity_groups, 1, 1],
            resources: self.phase_resources(&[
                (
                    ParticleBridgeBufferRole::SimulationParticles,
                    ParticleBridgeBufferAccess::Read,
                ),
                (
                    ParticleBridgeBufferRole::SimulationActiveIndices,
                    ParticleBridgeBufferAccess::Read,
                ),
                (
                    ParticleBridgeBufferRole::SimulationControl,
                    ParticleBridgeBufferAccess::Read,
                ),
                (ParticleBridgeBufferRole::EventFlags, ParticleBridgeBufferAccess::Read),
                (
                    ParticleBridgeBufferRole::EmitterRouteTable,
                    ParticleBridgeBufferAccess::Read,
                ),
                (
                    ParticleBridgeBufferRole::ActiveRouteIndices,
                    ParticleBridgeBufferAccess::Read,
                ),
                (ParticleBridgeBufferRole::RouteScratch, ParticleBridgeBufferAccess::Write),
                (ParticleBridgeBufferRole::RendererInput, ParticleBridgeBufferAccess::Write),
            ]),
        });
        ParticleSimulationBridgeRecordPlan {
            capacity: self.capacity,
            event_capacity: self.event_capacity,
            route_capacity: self.route_capacity,
            emitter_routes: self.emitter_routes.clone(),
            phases,
        }
    }

    fn phase_resources(
        &self,
        resources: &[(ParticleBridgeBufferRole, ParticleBridgeBufferAccess)],
    ) -> Vec<ParticleBridgePhysicalBuffer> {
        resources
            .iter()
            .map(|&(role, access)| self.physical_buffer(role, access))
            .collect()
    }

    fn physical_buffer(
        &self,
        role: ParticleBridgeBufferRole,
        access: ParticleBridgeBufferAccess,
    ) -> ParticleBridgePhysicalBuffer {
        let (buffer, bytes) = match role {
            ParticleBridgeBufferRole::SimulationParticles => (
                self.simulation_particles,
                u64::from(self.capacity) * GPU_PARTICLE_STRIDE,
            ),
            ParticleBridgeBufferRole::SimulationActiveIndices => {
                (self.simulation_active_indices, u64::from(self.capacity) * 4)
            }
            ParticleBridgeBufferRole::SimulationCompactEvents => (
                self.simulation_compact_events,
                u64::from(self.event_capacity)
                    * 4
                    * GPU_EVENT_STRIDE,
            ),
            ParticleBridgeBufferRole::SimulationControl => (
                self.simulation_control,
                u64::from(CONTROL_WORDS) * 4,
            ),
            ParticleBridgeBufferRole::EventFlags => (
                self.event_flags,
                u64::from(self.capacity) * PARTICLE_EVENT_FLAG_STRIDE,
            ),
            ParticleBridgeBufferRole::RendererInput => (
                self.renderer_input,
                u64::from(self.capacity) * PARTICLE_RENDER_INPUT_STRIDE,
            ),
            ParticleBridgeBufferRole::RendererInputCount => (self.renderer_input_count, 4),
            ParticleBridgeBufferRole::Validation => (self.validation, VALIDATION_BYTES),
            ParticleBridgeBufferRole::EmitterRouteTable => {
                (self.route_table, self.emitter_route_table_bytes())
            }
            ParticleBridgeBufferRole::RouteScratch => {
                (self.route_scratch, u64::from(self.route_scratch_words) * 4)
            }
            ParticleBridgeBufferRole::ActiveRouteIndices => {
                (self.active_route_indices, u64::from(self.capacity) * 4)
            }
        };
        ParticleBridgePhysicalBuffer {
            role,
            buffer,
            bytes,
            access,
        }
    }

    /// Appends the GPU-only handoff after the simulation graph outputs.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
        simulation: &GpuSimulationGraphOutputs,
    ) -> Result<ParticleSimulationBridgeGraphOutputs, CompileError> {
        let event_flags = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-event-flags",
            self.event_flags,
            u64::from(self.capacity) * PARTICLE_EVENT_FLAG_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let renderer_input = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-renderer-input",
            self.renderer_input,
            u64::from(self.capacity) * PARTICLE_RENDER_INPUT_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let renderer_input_count = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-input-count",
            self.renderer_input_count,
            4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let validation = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-validation",
            self.validation,
            VALIDATION_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let route_table = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-route-table",
            self.route_table,
            PARTICLE_EMITTER_ROUTE_HEADER_BYTES
                .checked_add(
                    u64::from(self.route_capacity)
                        .checked_mul(PARTICLE_EMITTER_ROUTE_STRIDE)
                        .ok_or(CompileError::Capacity {
                            kind: "buffer",
                            count: usize::MAX,
                            limit: usize::MAX,
                        })?,
                )
                .ok_or(CompileError::Capacity {
                    kind: "buffer",
                    count: usize::MAX,
                    limit: usize::MAX,
                })?,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let route_scratch = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-route-scratch",
            self.route_scratch,
            u64::from(self.route_scratch_words) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let active_route_indices = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-active-routes",
            self.active_route_indices,
            u64::from(self.capacity) * 4,
            BufferUsages::STORAGE,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );

        let clear_pipeline = self.clear_pipeline;
        let clear_bind_group = self.bind_group;
        let capacity = self.capacity;
        let mut flags_after_clear = event_flags;
        let mut input_after_clear = renderer_input;
        let mut input_count_after_clear = renderer_input_count;
        let mut validation_after_clear = validation;
        let mut routes_after_clear = route_table;
        let mut scratch_after_clear = route_scratch;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-clear",
            PassFlags::SIDE_EFFECT,
            |builder| {
                flags_after_clear = builder.write_storage_buffer(event_flags);
                input_after_clear = builder.write_storage_buffer(renderer_input);
                input_count_after_clear = builder.write_storage_buffer(renderer_input_count);
                validation_after_clear = builder.write_storage_buffer(validation);
                routes_after_clear = builder.write_storage_buffer(route_table);
                scratch_after_clear = builder.write_storage_buffer(route_scratch);
                move |context| {
                    context
                        .set_compute_pipeline(clear_pipeline)
                        .set_compute_bind_group(0, clear_bind_group, Vec::new())
                        .dispatch(capacity.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1);
                }
            },
        )?;

        let mark_pipeline = self.mark_events_pipeline;
        let mark_bind_group = self.bind_group;
        let event_capacity = self.event_capacity;
        let mut flags_after_events = flags_after_clear;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-mark-events",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(simulation.compact_events);
                builder.read_storage_buffer(simulation.control);
                flags_after_events = builder.write_storage_buffer(flags_after_clear);
                move |context| {
                    context
                        .set_compute_pipeline(mark_pipeline)
                        .set_compute_bind_group(0, mark_bind_group, Vec::new())
                        .dispatch(
                            event_capacity
                                .saturating_mul(4)
                                .div_ceil(BRIDGE_WORKGROUP_SIZE),
                            1,
                            1,
                        );
                }
            },
        )?;

        let count_pipeline = self.count_routes_pipeline;
        let bind_group = self.bind_group;
        let mut routes_after_count = routes_after_clear;
        let mut cache_after_count = active_route_indices;
        let mut validation_after_count = validation_after_clear;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-count-routes",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(simulation.particles);
                builder.read_storage_buffer(simulation.active_indices);
                builder.read_storage_buffer(simulation.control);
                builder.read_storage_buffer(scratch_after_clear);
                routes_after_count = builder.write_storage_buffer(routes_after_clear);
                cache_after_count = builder.write_storage_buffer(active_route_indices);
                validation_after_count = builder.write_storage_buffer(validation_after_clear);
                move |context| {
                    context
                        .set_compute_pipeline(count_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(capacity.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1);
                }
            },
        )?;

        let route_count = self.emitter_routes.len() as u32;
        let route_groups = route_count.div_ceil(BRIDGE_WORKGROUP_SIZE).max(1);
        let prefix_pipeline = self.prefix_routes_pipeline;
        let mut routes_after_prefix = routes_after_count;
        let mut scratch_after_prefix = scratch_after_clear;
        let mut count_after_prefix = input_count_after_clear;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-prefix-routes",
            PassFlags::SIDE_EFFECT,
            |builder| {
                routes_after_prefix = builder.write_storage_buffer(routes_after_count);
                scratch_after_prefix = builder.write_storage_buffer(scratch_after_clear);
                count_after_prefix = builder.write_storage_buffer(input_count_after_clear);
                move |context| {
                    context
                        .set_compute_pipeline(prefix_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(route_groups, 1, 1);
                }
            },
        )?;
        let active_scan_counts = scan_level_counts(route_count);
        for (level, count) in active_scan_counts.iter().copied().enumerate() {
            let pipeline = self.scan_pipelines[level];
            let previous_scratch = scratch_after_prefix;
            let previous_count = count_after_prefix;
            graph.try_add_compute_pass(
                format!("particle-simulation-renderer-bridge-prefix-sums-{level}"),
                PassFlags::SIDE_EFFECT,
                |builder| {
                    builder.read_storage_buffer(routes_after_prefix);
                    scratch_after_prefix = builder.write_storage_buffer(previous_scratch);
                    count_after_prefix = builder.write_storage_buffer(previous_count);
                    move |context| {
                        context
                            .set_compute_pipeline(pipeline)
                            .set_compute_bind_group(0, bind_group, Vec::new())
                            .dispatch(count.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1);
                    }
                },
            )?;
        }
        for level in (0..active_scan_counts.len().saturating_sub(1)).rev() {
            let pipeline = self.add_offset_pipelines[level];
            let count = active_scan_counts[level];
            let previous_scratch = scratch_after_prefix;
            graph.try_add_compute_pass(
                format!("particle-simulation-renderer-bridge-add-offsets-{level}"),
                PassFlags::SIDE_EFFECT,
                |builder| {
                    builder.read_storage_buffer(routes_after_prefix);
                    scratch_after_prefix = builder.write_storage_buffer(previous_scratch);
                    move |context| {
                        context
                            .set_compute_pipeline(pipeline)
                            .set_compute_bind_group(0, bind_group, Vec::new())
                            .dispatch(count.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1);
                    }
                },
            )?;
        }
        let finalize_pipeline = self.finalize_routes_pipeline;
        let routes_before_finalize = routes_after_prefix;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-finalize-routes",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(scratch_after_prefix);
                routes_after_prefix = builder.write_storage_buffer(routes_before_finalize);
                move |context| {
                    context
                        .set_compute_pipeline(finalize_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(route_groups, 1, 1);
                }
            },
        )?;
        let gather_pipeline = self.gather_pipeline;
        let mut input_after_gather = input_after_clear;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-scatter",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(simulation.particles);
                builder.read_storage_buffer(simulation.active_indices);
                builder.read_storage_buffer(simulation.control);
                builder.read_storage_buffer(flags_after_events);
                builder.read_storage_buffer(routes_after_prefix);
                builder.read_storage_buffer(cache_after_count);
                builder.write_storage_buffer(scratch_after_prefix);
                input_after_gather = builder.write_storage_buffer(input_after_clear);
                move |context| {
                    context
                        .set_compute_pipeline(gather_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(capacity.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1);
                }
            },
        )?;

        Ok(ParticleSimulationBridgeGraphOutputs {
            renderer_input: input_after_gather,
            renderer_input_count: count_after_prefix,
            event_flags: flags_after_events,
            validation: validation_after_count,
            compact_events: simulation.compact_events,
            emitter_route_table: routes_after_prefix,
            emitter_routes: self.emitter_routes.clone(),
        })
    }

    /// Returns the bridge's fixed pool capacity.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the simulation event capacity consumed by the mark pass.
    #[must_use]
    pub const fn event_capacity(&self) -> u32 {
        self.event_capacity
    }

    /// Exact simulation particle buffer bound at bridge binding 0.
    #[must_use]
    pub const fn simulation_particles_buffer(&self) -> BufferId {
        self.simulation_particles
    }

    /// Exact simulation active-index buffer bound at bridge binding 1.
    #[must_use]
    pub const fn simulation_active_indices_buffer(&self) -> BufferId {
        self.simulation_active_indices
    }

    /// Exact compact-event buffer bound at bridge binding 2.
    #[must_use]
    pub const fn simulation_compact_events_buffer(&self) -> BufferId {
        self.simulation_compact_events
    }

    /// Exact simulation control buffer bound at bridge binding 3.
    #[must_use]
    pub const fn simulation_control_buffer(&self) -> BufferId {
        self.simulation_control
    }

    /// Exact per-slot event-flag buffer bound at bridge binding 4.
    #[must_use]
    pub const fn event_flags_buffer(&self) -> BufferId {
        self.event_flags
    }

    /// Exact renderer-input buffer bound at bridge binding 5.
    #[must_use]
    pub const fn renderer_input_buffer(&self) -> BufferId {
        self.renderer_input
    }

    /// Exact validation-counter buffer bound at bridge binding 6.
    #[must_use]
    pub const fn validation_buffer(&self) -> BufferId {
        self.validation
    }

    /// Exact renderer input-count buffer bound at bridge binding 8.
    #[must_use]
    pub const fn renderer_input_count_buffer(&self) -> BufferId {
        self.renderer_input_count
    }

    /// Exact route-scratch buffer bound at bridge binding 9.
    #[must_use]
    pub const fn route_scratch_buffer(&self) -> BufferId {
        self.route_scratch
    }

    /// Exact active-route-index buffer bound at bridge binding 10.
    #[must_use]
    pub const fn active_route_indices_buffer(&self) -> BufferId {
        self.active_route_indices
    }

    /// Exact byte size of the route table, including its 16-byte header.
    #[must_use]
    pub const fn emitter_route_table_bytes(&self) -> u64 {
        PARTICLE_EMITTER_ROUTE_HEADER_BYTES
            + (self.route_capacity as u64) * PARTICLE_EMITTER_ROUTE_STRIDE
    }

    /// Exact byte size of the route prefix scratch buffer.
    #[must_use]
    pub const fn route_scratch_bytes(&self) -> u64 {
        (self.route_scratch_words as u64) * 4
    }

    /// Returns the persistent physical route-table buffer for renderer-owned
    /// bind groups. The bridge remains its sole owner and must outlive every
    /// bind group/graph that references it. Layout: 16-byte header followed by
    /// `emitter_route_capacity()` records of 16 bytes each.
    #[must_use]
    pub const fn emitter_route_table_buffer(&self) -> BufferId {
        self.route_table
    }

    /// Returns the allocated route-record capacity, not the current number of
    /// registered emitters. The GPU header's first u32 supplies that count.
    #[must_use]
    pub const fn emitter_route_capacity(&self) -> u32 {
        self.route_capacity
    }

    /// Returns the configured emitter routes without exposing GPU ownership.
    #[must_use]
    pub fn emitter_routes(&self) -> &[ParticleEmitterRenderRoute] {
        &self.emitter_routes
    }

    /// Releases bridge-owned resources. Call exactly once, after every graph
    /// and downstream consumer referencing these resources has finished its
    /// GPU work. The simulation and renderer stages remain the owners of
    /// their persistent buffers.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        device.destroy_bind_group(self.bind_group);
        for pipeline in [
            self.clear_pipeline,
            self.mark_events_pipeline,
            self.count_routes_pipeline,
            self.prefix_routes_pipeline,
            self.finalize_routes_pipeline,
            self.gather_pipeline,
        ] {
            device.destroy_compute_pipeline(pipeline);
        }
        for pipeline in self.scan_pipelines.iter().chain(&self.add_offset_pipelines) {
            device.destroy_compute_pipeline(*pipeline);
        }
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.layout);
        device.destroy_shader_module(self.shader);
        device.destroy_buffer(self.event_flags);
        device.destroy_buffer(self.validation);
        device.destroy_buffer(self.route_table);
        device.destroy_buffer(self.route_scratch);
        device.destroy_buffer(self.active_route_indices);
    }
}

fn scan_level_counts(mut count: u32) -> Vec<u32> {
    let mut levels = Vec::new();
    while count > BRIDGE_WORKGROUP_SIZE {
        count = count.div_ceil(BRIDGE_WORKGROUP_SIZE);
        levels.push(count);
    }
    levels
}

// Only the small, device-capacity-dependent set of hierarchy entry points is
// generated. The scan algorithms and ABI remain in the checked-in WGSL.
fn bridge_shader_source(capacity: u32) -> String {
    let mut source = String::from(include_str!("../shaders/particle_simulation_bridge.wgsl"));
    for level in 0..scan_level_counts(capacity).len() {
        source.push_str(&format!(
            "\n@compute @workgroup_size(64) fn prefix_sums_{level}(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{ scan_sums({level}u, group.x, lane); }}\n\
             @compute @workgroup_size(64) fn add_offsets_{level}(@builtin(global_invocation_id) gid: vec3<u32>) {{ add_offsets({level}u, gid.x); }}\n"
        ));
    }
    source
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
    use crate::{GpuParticle, GpuSimulationParams, ParticleCullView, ParticleRendererGpu};
    use prism_render_architecture::particle::indirect_draw::RendererGeometry;
    use prism_render_architecture::particle::renderers::RendererKind;
    use prism_render_driver as rhi;
    use prism_render_driver::{DeviceDescriptor, Instance as _, RequestAdapterOptions};
    use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

    const RENDER_INPUT_WORDS: usize = (PARTICLE_RENDER_INPUT_STRIDE / 4) as usize;

    fn null_device_with_limits(limits: rhi::Limits) -> (NullDevice, NullQueue) {
        (
            NullDevice::new_with_capabilities(rhi::DeviceCapabilities {
                backend: rhi::Backend::Noop,
                features: rhi::Features::NONE,
                limits,
            }),
            NullQueue::new(),
        )
    }

    fn route() -> ParticleEmitterRenderRoute {
        ParticleEmitterRenderRoute {
            emitter_id: 7,
            active_start: 0,
            particle_count: 1,
            renderer: ParticleRendererRoute::new(
                RendererKind::Sprite,
                prism_render_architecture::particle::SortStrategy::ViewDepthBitonic,
                RendererGeometry::Sprite {
                    alive_count: 1,
                    first_instance: 0,
                },
            )
            .expect("valid sprite route"),
        }
    }

    #[test]
    fn bridge_storage_stage_preflight_rejects_before_allocating_and_accepts_eleven() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("adapter");
        let (baseline_device, _queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("device");
        assert_eq!(
            baseline_device
                .capabilities()
                .limits
                .max_storage_buffers_per_shader_stage,
            8
        );
        let (device, _) = null_device_with_limits(rhi::Limits {
            max_storage_buffers_per_shader_stage: 11,
            ..rhi::Limits::desktop()
        });
        let baseline = device.live_resource_count();
        let simulation = GpuSimulationStage::new(&device, 8, 4).expect("simulation resources");
        let renderer = ParticleRendererGpu::new(&device, 8, route().renderer).expect("renderer");
        let before_bridge = device.live_resource_count();
        for limit in [0, 8, 10] {
            let (limited, _) = null_device_with_limits(rhi::Limits {
                max_storage_buffers_per_shader_stage: limit,
                ..rhi::Limits::desktop()
            });
            assert!(matches!(
                ParticleSimulationRendererBridge::new(&limited, &simulation, &renderer, &[route()]),
                Err(ParticleSimulationBridgeError::StorageBufferStageLimitTooSmall {
                    requested: 11,
                    limit: actual,
                }) if actual == limit
            ));
            assert_eq!(limited.live_resource_count(), 0);
            assert_eq!(device.live_resource_count(), before_bridge);
        }
        let bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[route()])
                .expect("exactly eleven storage bindings are sufficient");
        bridge.destroy(&device);
        assert_eq!(device.live_resource_count(), before_bridge);
        renderer.destroy(&device);
        simulation.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline);
    }

    #[test]
    fn null_rhi_records_simulation_bridge_bindings_and_route_contract() {
        let (device, queue) = null_device_with_limits(rhi::Limits::desktop());
        let baseline_resources = device.live_resource_count();
        let simulation = GpuSimulationStage::new(&device, 8, 4).expect("simulation resources");
        let renderer =
            ParticleRendererGpu::new(&device, 8, route().renderer).expect("renderer resources");
        renderer
            .set_view(&queue, ParticleCullView::unrestricted())
            .expect("view upload");
        let bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[route()])
                .expect("bridge resources");
        let second_route = ParticleEmitterRenderRoute {
            emitter_id: 8,
            active_start: 1,
            particle_count: 2,
            renderer: route().renderer,
        };
        let mut bridge = bridge;
        assert_eq!(bridge.emitter_route_table_buffer(), bridge.route_table);
        assert_eq!(bridge.emitter_route_capacity(), 8);
        bridge
            .set_emitter_routes(&queue, &[route(), second_route])
            .expect("dynamic route update");
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let simulation_outputs = simulation
            .append_to_graph(&mut graph)
            .expect("simulation graph");
        let bridge_outputs = bridge
            .append_to_graph(&mut graph, &simulation_outputs)
            .expect("bridge graph");
        renderer
            .append_compute_to_graph_with_input_count(
                &mut graph,
                bridge_outputs.renderer_input,
                bridge_outputs.renderer_input_count,
            )
            .expect("renderer graph");
        let plan = graph.compile().expect("combined graph compiles");
        assert!(plan.declared_pass_count >= 8);
        assert!(plan.barrier_count() >= 7);
        assert_eq!(bridge_outputs.emitter_routes, vec![route(), second_route]);
        assert_eq!(
            bridge_outputs.compact_events,
            simulation_outputs.compact_events
        );
        bridge.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline_resources);
    }

    #[test]
    fn bridge_generated_scan_shaders_validate_at_hierarchy_boundaries() {
        assert_eq!(scan_level_counts(0), vec![]);
        assert_eq!(scan_level_counts(64), vec![]);
        assert_eq!(scan_level_counts(65), vec![2]);
        assert_eq!(scan_level_counts(4096), vec![64]);
        assert_eq!(scan_level_counts(4097), vec![65, 2]);
        for capacity in [1, 64, 65, 4096, 4097, 262145] {
            let source = bridge_shader_source(capacity);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
            let mut storage_bindings: Vec<_> = module
                .global_variables
                .iter()
                .filter_map(|(_, variable)| {
                    if matches!(variable.space, naga::AddressSpace::Storage { .. }) {
                        variable
                            .binding
                            .as_ref()
                            .map(|binding| (binding.group, binding.binding))
                    } else {
                        None
                    }
                })
                .collect();
            storage_bindings.sort_unstable();
            assert_eq!(
                storage_bindings,
                (0..BRIDGE_STORAGE_BUFFER_BINDINGS)
                    .map(|binding| (0, binding))
                    .collect::<Vec<_>>()
            );
            assert!(module
                .entry_points
                .iter()
                .all(|entry| entry.workgroup_size == [BRIDGE_WORKGROUP_SIZE, 1, 1]));
            let input_type = module
                .types
                .iter()
                .find_map(|(_, ty)| {
                    (ty.name.as_deref() == Some("ParticleRenderInput")).then_some(ty)
                })
                .expect("bridge output ABI type");
            let naga::TypeInner::Struct { members, span } = &input_type.inner else {
                panic!("renderer input must be a struct");
            };
            assert_eq!(u64::from(*span), PARTICLE_RENDER_INPUT_STRIDE);
            assert_eq!(
                members
                    .iter()
                    .map(|member| member.offset)
                    .collect::<Vec<_>>(),
                vec![0, 16, 20, 24, 28, 32, 36, 40, 44]
            );
            assert_eq!(
                core::mem::size_of::<super::super::renderer_matrix::ParticleRenderInput>() as u64,
                PARTICLE_RENDER_INPUT_STRIDE
            );
            let source_type = module
                .types
                .iter()
                .find_map(|(_, ty)| (ty.name.as_deref() == Some("Particle")).then_some(ty))
                .expect("simulation input ABI type");
            let naga::TypeInner::Struct { members, span } = &source_type.inner else {
                panic!("simulation particle must be a struct");
            };
            assert_eq!(u64::from(*span), GPU_PARTICLE_STRIDE);
            assert_eq!(
                members
                    .iter()
                    .map(|member| member.offset)
                    .collect::<Vec<_>>(),
                [0, 16, 32, 36, 40, 44, 48, 52, 56, 60]
            );
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("capacity {capacity}: {}", error.emit_to_string(&source))
            });
        }
    }

    #[test]
    fn bridge_releases_multilevel_scan_resources_and_rejects_capacity_mismatch() {
        let (device, queue) = null_device_with_limits(rhi::Limits::desktop());
        let baseline = device.live_resource_count();
        let simulation = GpuSimulationStage::new(&device, 4097, 4).expect("simulation resources");
        let renderer = ParticleRendererGpu::new(&device, 4097, route().renderer).expect("renderer");
        let before_bridge = device.live_resource_count();
        let mut bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[])
                .expect("bridge");
        bridge.upload_routes(&queue).expect("empty registry");
        assert_eq!(bridge.scan_pipelines.len(), 2);
        assert_eq!(bridge.add_offset_pipelines.len(), 2);
        bridge.destroy(&device);
        assert_eq!(device.live_resource_count(), before_bridge);
        let small_renderer =
            ParticleRendererGpu::new(&device, 8, route().renderer).expect("small renderer");
        let before_failure = device.live_resource_count();
        assert!(matches!(
            ParticleSimulationRendererBridge::new(&device, &simulation, &small_renderer, &[]),
            Err(ParticleSimulationBridgeError::RendererCapacityMismatch {
                simulation: 4097,
                renderer: 8
            })
        ));
        assert_eq!(device.live_resource_count(), before_failure);
        small_renderer.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
        assert_eq!(device.live_resource_count(), baseline);
    }

    fn read_u32(bytes: &[u8], word: usize) -> u32 {
        u32::from_le_bytes(bytes[word * 4..word * 4 + 4].try_into().expect("u32"))
    }

    fn assert_no_fabricated_topology(bytes: &[u8]) {
        assert_eq!(bytes.len() % PARTICLE_RENDER_INPUT_STRIDE as usize, 0);
        for record in bytes.chunks_exact(PARTICLE_RENDER_INPUT_STRIDE as usize) {
            assert_eq!(
                &record[32..48],
                &[0; 16],
                "imported source did not author RibbonId/birth"
            );
        }
    }

    // Test-only imports permit malformed GPU counts/indices without simulation
    // passes resetting them before the bridge can validate them.
    fn resident_outputs(
        graph: &mut RenderGraph,
        simulation: &GpuSimulationStage,
    ) -> GpuSimulationGraphOutputs {
        let state = BufferState::storage_read(PipelineStages::COMPUTE_SHADER);
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        GpuSimulationGraphOutputs {
            particles: import_buffer(
                graph,
                "test-particles",
                simulation.particles(),
                u64::from(simulation.capacity()) * GPU_PARTICLE_STRIDE,
                usage,
                state,
            ),
            active_indices: import_buffer(
                graph,
                "test-active",
                simulation.active_indices(),
                u64::from(simulation.capacity()) * 4,
                usage,
                state,
            ),
            compact_events: import_buffer(
                graph,
                "test-events",
                simulation.compact_events(),
                u64::from(simulation.event_capacity()) * 4 * GPU_EVENT_STRIDE,
                usage,
                state,
            ),
            control: import_buffer(
                graph,
                "test-control",
                simulation.control(),
                u64::from(crate::gpu_simulation::CONTROL_WORDS) * 4,
                usage,
                state,
            ),
        }
    }

    #[test]
    fn physical_gpu_bridge_segments_unknown_empty_overflow_and_multilevel_prefix() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) if !require_hardware_gpu() => return,
            Err(HardwareDeviceError::NonHardware(_)) if !require_hardware_gpu() => return,
            Err(error) => panic!("physical GPU required for bridge segmentation: {error}"),
        };
        for capacity in [8u32, 65, 4097] {
            let simulation =
                GpuSimulationStage::new(&device, capacity, 4).expect("simulation resources");
            let renderer =
                ParticleRendererGpu::new(&device, capacity, route().renderer).expect("renderer");
            // Reverse registration order; repeated even emitters interleave
            // with registered but empty odd emitters. CPU ranges are nonsense.
            let routes: Vec<_> = (0..capacity)
                .rev()
                .map(|emitter_id| ParticleEmitterRenderRoute {
                    emitter_id,
                    active_start: u32::MAX,
                    particle_count: 0,
                    ..route()
                })
                .collect();
            let mut particles: Vec<_> = (0..capacity)
                .map(|index| {
                    GpuParticle::live(
                        [index as f32, 0.0, 0.0],
                        [0.0; 3],
                        2.0,
                        0.2,
                        index,
                        if index == capacity - 1 {
                            u32::MAX
                        } else {
                            index % (capacity / 2) * 2
                        },
                    )
                })
                .collect();
            particles[0].age = particles[0].lifetime;
            particles[1].position[0] = f32::NAN;
            particles[2].radius = f32::INFINITY;
            simulation.initialize_particles(&queue, &particles);
            let mut active: Vec<u32> = (0..capacity).collect();
            active[(capacity - 2) as usize] = u32::MAX;
            let bytes: Vec<_> = active
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect();
            queue.write_buffer(simulation.active_indices(), 0, &bytes);
            let mut bridge =
                ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &routes)
                    .expect("bridge");
            bridge.upload_routes(&queue).expect("registry");
            let mut controls = [0u32; crate::gpu_simulation::CONTROL_WORDS as usize];
            controls[2] = u32::MAX;
            controls[7] = u32::MAX;
            let bytes: Vec<_> = controls
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            queue.write_buffer(simulation.control(), 0, &bytes);
            let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
            let inputs = resident_outputs(&mut graph, &simulation);
            let outputs = bridge
                .append_to_graph(&mut graph, &inputs)
                .expect("bridge graph");
            graph
                .try_add_compute_pass("test-route-consumer", PassFlags::SIDE_EFFECT, |builder| {
                    builder.read_storage_buffer(outputs.emitter_route_table);
                    builder.read_storage_buffer(outputs.renderer_input);
                    builder.read_storage_buffer(outputs.renderer_input_count);
                    move |_| {}
                })
                .expect("consumer");
            let recorded = graph.execute(&device, &queue).expect("GPU segmentation");
            let table = queue
                .read_buffer(bridge.route_table, 16 + u64::from(capacity) * 16)
                .expect("route table readback");
            let rendered = queue
                .read_buffer(
                    renderer.input_buffer(),
                    u64::from(capacity) * PARTICLE_RENDER_INPUT_STRIDE,
                )
                .expect("renderer input readback");
            assert_no_fabricated_topology(&rendered);
            assert_eq!(
                read_u32(
                    &queue
                        .read_buffer(renderer.input_count_buffer(), 4)
                        .expect("renderer input count readback"),
                    0,
                ),
                capacity - 5
            );
            assert_eq!(read_u32(&table, 0), capacity);
            let mut prefix = 0u32;
            let mut observed_by_route = Vec::new();
            for (ordinal, route) in routes.iter().enumerate() {
                let expected = (3..capacity - 2)
                    .filter(|index| index % (capacity / 2) * 2 == route.emitter_id)
                    .count() as u32;
                let word = 4 + ordinal * 4;
                assert_eq!(
                    read_u32(&table, word),
                    prefix,
                    "capacity {capacity}, route {ordinal}"
                );
                assert_eq!(read_u32(&table, word + 1), expected);
                assert_eq!(read_u32(&table, word + 2), route.emitter_id);
                assert_eq!(read_u32(&table, word + 3), ordinal as u32);
                let mut observed_route = Vec::new();
                for slot in prefix..prefix + expected {
                    let input_word = slot as usize * RENDER_INPUT_WORDS;
                    assert_eq!(read_u32(&rendered, input_word + 7), route.emitter_id);
                    assert_eq!(read_u32(&rendered, input_word + 6), 0);
                    observed_route.push(read_u32(&rendered, input_word + 5));
                }
                observed_by_route.push((route.emitter_id, observed_route));
                prefix += expected;
            }
            // Atomic scatter has intentionally unspecified physical order.
            // Assert exact per-emitter membership, not strip connectivity.
            // Missing RibbonId/birth metadata must not fall back to pool order.
            for (emitter_id, mut observed_route) in observed_by_route {
                observed_route.sort_unstable();
                let expected: Vec<_> = (3..capacity - 2)
                    .filter(|index| index % (capacity / 2) * 2 == emitter_id)
                    .collect();
                assert_eq!(
                    observed_route, expected,
                    "capacity {capacity}, emitter {emitter_id}"
                );
            }
            for slot in prefix..capacity {
                assert_eq!(
                    read_u32(&rendered, slot as usize * RENDER_INPUT_WORDS + 6),
                    PARTICLE_BRIDGE_FLAG_INVALID_LIFETIME
                );
            }
            let diagnostics = queue
                .read_buffer(bridge.validation, VALIDATION_BYTES)
                .expect("bridge diagnostics readback");
            assert_eq!(
                (0..4)
                    .map(|word| read_u32(&diagnostics, word))
                    .collect::<Vec<_>>(),
                vec![4, 1, u32::MAX - capacity, u32::MAX]
            );
            recorded.release(&device);

            // Empty live set resets every segment; removing the registry also
            // clears old ranges and rejects all still-live unknown emitters.
            for empty_registry in [false, true] {
                controls.fill(0);
                if empty_registry {
                    bridge
                        .set_emitter_routes(&queue, &[])
                        .expect("remove registry");
                    controls[2] = capacity;
                }
                let bytes: Vec<_> = controls
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                queue.write_buffer(simulation.control(), 0, &bytes);
                let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
                let inputs = resident_outputs(&mut graph, &simulation);
                bridge
                    .append_to_graph(&mut graph, &inputs)
                    .expect("empty bridge graph");
                let recorded = graph
                    .execute(&device, &queue)
                    .expect("empty GPU segmentation");
                assert_eq!(
                    read_u32(
                        &queue
                            .read_buffer(renderer.input_count_buffer(), 4)
                            .expect("renderer input count readback"),
                        0,
                    ),
                    0
                );
                let table = queue
                    .read_buffer(bridge.route_table, 16 + u64::from(capacity) * 16)
                    .expect("route table readback");
                assert_eq!(
                    read_u32(&table, 0),
                    if empty_registry { 0 } else { capacity }
                );
                for ordinal in 0..capacity as usize {
                    assert_eq!(read_u32(&table, 4 + ordinal * 4), 0);
                    assert_eq!(read_u32(&table, 5 + ordinal * 4), 0);
                }
                let rendered = queue
                    .read_buffer(
                        renderer.input_buffer(),
                        u64::from(capacity) * PARTICLE_RENDER_INPUT_STRIDE,
                    )
                    .expect("renderer input readback");
                assert_no_fabricated_topology(&rendered);
                assert!((0..capacity as usize).all(|slot| read_u32(
                    &rendered,
                    slot * RENDER_INPUT_WORDS + 6
                )
                    == PARTICLE_BRIDGE_FLAG_INVALID_LIFETIME));
                let diagnostics = queue
                    .read_buffer(bridge.validation, VALIDATION_BYTES)
                    .expect("bridge diagnostics readback");
                assert_eq!(
                    (0..4)
                        .map(|word| read_u32(&diagnostics, word))
                        .collect::<Vec<_>>(),
                    if empty_registry {
                        vec![4, capacity - 4, 0, 0]
                    } else {
                        vec![0; 4]
                    }
                );
                recorded.release(&device);
            }
            bridge.destroy(&device);
            renderer.destroy(&device);
            simulation.destroy(&device);
        }
    }

    #[test]
    fn physical_gpu_bridge_scans_65_routes_at_4097_capacity_across_registry_changes() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) if !require_hardware_gpu() => return,
            Err(HardwareDeviceError::NonHardware(_)) if !require_hardware_gpu() => return,
            Err(error) => panic!("physical GPU required for registry transitions: {error}"),
        };
        assert_eq!(
            device
                .capabilities()
                .limits
                .max_storage_buffers_per_shader_stage,
            device
                .wgpu_device()
                .limits()
                .max_storage_buffers_per_shader_stage
        );
        const CAPACITY: u32 = 4097;
        let simulation =
            GpuSimulationStage::new(&device, CAPACITY, 4).expect("simulation resources");
        let renderer =
            ParticleRendererGpu::new(&device, CAPACITY, route().renderer).expect("renderer");
        let particles: Vec<_> = (0..CAPACITY)
            .map(|index| {
                GpuParticle::live(
                    [index as f32, 0.0, 0.0],
                    [0.0; 3],
                    2.0,
                    0.2,
                    // Deliberately distinct from the stable pool index.
                    CAPACITY - index,
                    if index == CAPACITY - 1 {
                        u32::MAX
                    } else {
                        index * 17 % 65
                    },
                )
            })
            .collect();
        simulation.initialize_particles(&queue, &particles);
        let active_bytes: Vec<_> = (0..CAPACITY).rev().flat_map(u32::to_le_bytes).collect();
        queue.write_buffer(simulation.active_indices(), 0, &active_bytes);
        let mut bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[])
                .expect("bridge");
        let persistent_buffers = (
            bridge.route_table,
            bridge.route_scratch,
            bridge.active_route_indices,
        );
        let cases: Vec<(Vec<u32>, u32)> = vec![
            ((0..65).rev().collect(), CAPACITY),
            ((0..64).collect(), CAPACITY),
            (vec![17], CAPACITY),
            (vec![], CAPACITY),
            ((0..65).map(|index| (index + 23) % 65).collect(), CAPACITY),
            ((0..CAPACITY).rev().collect(), CAPACITY),
            ((0..CAPACITY).collect(), 0),
            ((0..65).rev().collect(), 0),
            ((32..97).rev().collect(), CAPACITY),
            ((32..97).collect(), CAPACITY),
            ((0..65).collect(), CAPACITY),
        ];
        for (frame, (ids, live_count)) in cases.into_iter().enumerate() {
            let routes: Vec<_> = ids
                .iter()
                .copied()
                .map(|emitter_id| ParticleEmitterRenderRoute {
                    emitter_id,
                    active_start: u32::MAX,
                    particle_count: u32::MAX,
                    ..route()
                })
                .collect();
            bridge
                .set_emitter_routes(&queue, &routes)
                .expect("registry update");
            // Rejected updates must leave both CPU registry and GPU lookup
            // untouched; this execution still verifies the accepted registry.
            if frame == 4 {
                let mut duplicate = routes.clone();
                duplicate.push(routes[0]);
                assert!(matches!(
                    bridge.set_emitter_routes(&queue, &duplicate),
                    Err(ParticleSimulationBridgeError::DuplicateEmitterId { .. })
                ));
                assert_eq!(bridge.emitter_routes, routes);
            }
            if routes.len() == CAPACITY as usize && live_count != 0 {
                let mut oversized = routes.clone();
                oversized.push(ParticleEmitterRenderRoute {
                    emitter_id: u32::MAX,
                    ..route()
                });
                assert!(matches!(
                    bridge.set_emitter_routes(&queue, &oversized),
                    Err(ParticleSimulationBridgeError::TooManyEmitterRoutes {
                        capacity: CAPACITY,
                        ..
                    })
                ));
                assert_eq!(bridge.emitter_routes, routes);
            }
            assert_eq!(
                (
                    bridge.route_table,
                    bridge.route_scratch,
                    bridge.active_route_indices
                ),
                persistent_buffers
            );
            let mut controls = [0u32; crate::gpu_simulation::CONTROL_WORDS as usize];
            controls[2] = live_count;
            let control_bytes: Vec<_> = controls.into_iter().flat_map(u32::to_le_bytes).collect();
            queue.write_buffer(simulation.control(), 0, &control_bytes);
            // Prove every 48B record is rewritten, including cleared tails
            // and topology flags, rather than relying on allocation zeroing.
            queue.write_buffer(
                renderer.input_buffer(),
                0,
                &vec![0xff; CAPACITY as usize * PARTICLE_RENDER_INPUT_STRIDE as usize],
            );
            let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
            let inputs = resident_outputs(&mut graph, &simulation);
            bridge
                .append_to_graph(&mut graph, &inputs)
                .expect("registry graph");
            let recorded = graph
                .execute(&device, &queue)
                .expect("GPU registry transition");
            let table = queue
                .read_buffer(bridge.route_table, 16 + u64::from(CAPACITY) * 16)
                .expect("route table readback");
            let rendered = queue
                .read_buffer(
                    renderer.input_buffer(),
                    u64::from(CAPACITY) * PARTICLE_RENDER_INPUT_STRIDE,
                )
                .expect("renderer input readback");
            assert_no_fabricated_topology(&rendered);
            assert_eq!(read_u32(&table, 0), routes.len() as u32, "frame {frame}");
            assert!((1..4).all(|word| read_u32(&table, word) == 0));
            let mut prefix = 0;
            for (ordinal, route) in routes.iter().enumerate() {
                let expected: Vec<_> = (0..CAPACITY)
                    .filter(|&index| {
                        live_count != 0 && particles[index as usize].emitter_id == route.emitter_id
                    })
                    .collect();
                let word = 4 + ordinal * 4;
                assert_eq!(
                    read_u32(&table, word),
                    prefix,
                    "frame {frame}, route {ordinal}"
                );
                assert_eq!(read_u32(&table, word + 1), expected.len() as u32);
                assert_eq!(read_u32(&table, word + 2), route.emitter_id);
                assert_eq!(read_u32(&table, word + 3), ordinal as u32);
                let end = prefix + expected.len() as u32;
                assert!(end <= CAPACITY);
                let mut observed = Vec::new();
                for slot in prefix..end {
                    let input = slot as usize * RENDER_INPUT_WORDS;
                    assert_eq!(read_u32(&rendered, input + 7), route.emitter_id);
                    assert_eq!(read_u32(&rendered, input + 6), 0);
                    observed.push(read_u32(&rendered, input + 5));
                }
                observed.sort_unstable();
                assert_eq!(
                    observed, expected,
                    "frame {frame}, emitter {}",
                    route.emitter_id
                );
                prefix = end;
            }
            // Shrinking must reset old ranges, not leave consumable stale counts.
            for ordinal in routes.len()..CAPACITY as usize {
                assert_eq!(read_u32(&table, 4 + ordinal * 4), 0);
                assert_eq!(read_u32(&table, 5 + ordinal * 4), 0);
            }
            for slot in prefix..CAPACITY {
                assert_eq!(
                    read_u32(&rendered, slot as usize * RENDER_INPUT_WORDS + 6),
                    PARTICLE_BRIDGE_FLAG_INVALID_LIFETIME
                );
            }
            assert_eq!(
                read_u32(
                    &queue
                        .read_buffer(renderer.input_count_buffer(), 4)
                        .expect("renderer input count readback"),
                    0,
                ),
                prefix
            );
            let diagnostics = queue
                .read_buffer(bridge.validation, VALIDATION_BYTES)
                .expect("bridge diagnostics readback");
            assert_eq!(
                (0..4)
                    .map(|word| read_u32(&diagnostics, word))
                    .collect::<Vec<_>>(),
                vec![0, live_count - prefix, 0, 0],
                "frame {frame}"
            );
            recorded.release(&device);
        }
        bridge.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
    }

    #[test]
    fn physical_gpu_bridge_real_event_payload_overflow_on_off_and_recovery() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) if !require_hardware_gpu() => return,
            Err(HardwareDeviceError::NonHardware(_)) if !require_hardware_gpu() => return,
            Err(error) => panic!("physical GPU required for event overflow comparison: {error}"),
        };
        let simulation = GpuSimulationStage::new(&device, 8, 4).expect("simulation resources");
        let renderer = ParticleRendererGpu::new(&device, 8, route().renderer).expect("renderer");
        let particles: Vec<_> = (0..8)
            .map(|index| GpuParticle::live([index as f32, 0.0, 0.0], [0.0; 3], 2.0, 0.2, index, 7))
            .collect();
        simulation.initialize_particles(&queue, &particles);
        let mut bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[route()])
                .expect("bridge");
        bridge.upload_routes(&queue).expect("registry");
        // Real simulation emission: four fits, eight overflows the source
        // pool but retains four valid compacted records. Positive flags must
        // recover afterwards; disabling events must ignore stale payloads.
        for (enabled, threshold, attempted, overflow) in [
            (true, 4.0, 4, 0),
            (true, 0.0, 8, 4),
            (true, 4.0, 4, 0),
            (false, 0.0, 0, 0),
        ] {
            simulation.upload_inputs(
                &queue,
                &[],
                &[],
                GpuSimulationParams {
                    dt: 0.0,
                    condition_enabled: enabled,
                    condition_x: threshold,
                    ..GpuSimulationParams::default()
                },
            );
            let simulation_output = simulation.run(&queue).expect("GPU simulation readback");
            assert_eq!(simulation_output.control[0], attempted);
            assert_eq!(simulation_output.control[2], 8);
            assert_eq!(&simulation_output.control[3..6], &[0; 3]);
            assert_eq!(simulation_output.control[6], attempted);
            assert_eq!(simulation_output.control[7], overflow);
            assert_eq!(simulation_output.control[8], attempted.min(4));
            assert_eq!(
                simulation_output.compact_events.len(),
                attempted.min(4) as usize
            );
            let mut event_indices = Vec::new();
            for event in &simulation_output.compact_events {
                assert_eq!(event.kind, 3);
                assert!(event.particle_index < 8);
                assert_eq!(event.payload, [event.particle_index as f32, 0.0, 0.0, 0.0]);
                event_indices.push(event.particle_index);
            }
            event_indices.sort_unstable();
            event_indices.dedup();
            assert_eq!(event_indices.len(), attempted.min(4) as usize);
            if attempted == 4 {
                assert_eq!(event_indices, vec![4, 5, 6, 7]);
            }
            let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
            let inputs = resident_outputs(&mut graph, &simulation);
            bridge
                .append_to_graph(&mut graph, &inputs)
                .expect("event bridge graph");
            let recorded = graph.execute(&device, &queue).expect("GPU event handoff");
            let flags = queue
                .read_buffer(bridge.event_flags, 8 * PARTICLE_EVENT_FLAG_STRIDE)
                .expect("event flags readback");
            let expected_flag = |index: u32| {
                if overflow == 0 && event_indices.contains(&index) {
                    PARTICLE_BRIDGE_FLAG_EVENT
                } else {
                    0
                }
            };
            for index in 0..8 {
                assert_eq!(read_u32(&flags, index), expected_flag(index as u32));
            }
            let rendered = queue
                .read_buffer(renderer.input_buffer(), 8 * PARTICLE_RENDER_INPUT_STRIDE)
                .expect("renderer input readback");
            assert_no_fabricated_topology(&rendered);
            let mut observed = Vec::new();
            for slot in 0..8 {
                let index = read_u32(&rendered, slot * RENDER_INPUT_WORDS + 5);
                assert!(index < 8);
                assert_eq!(
                    read_u32(&rendered, slot * RENDER_INPUT_WORDS + 6),
                    expected_flag(index)
                );
                assert_eq!(read_u32(&rendered, slot * RENDER_INPUT_WORDS + 7), 7);
                observed.push(index);
            }
            observed.sort_unstable();
            assert_eq!(observed, (0..8).collect::<Vec<_>>());
            assert_eq!(
                read_u32(
                    &queue
                        .read_buffer(renderer.input_count_buffer(), 4)
                        .expect("renderer input count readback"),
                    0,
                ),
                8
            );
            let table = queue
                .read_buffer(bridge.route_table, 32)
                .expect("route table readback");
            assert_eq!(
                (0..8)
                    .map(|word| read_u32(&table, word))
                    .collect::<Vec<_>>(),
                vec![1, 0, 0, 0, 0, 8, 7, 0]
            );
            let diagnostics = queue
                .read_buffer(bridge.validation, VALIDATION_BYTES)
                .expect("bridge diagnostics readback");
            assert_eq!(
                (0..4)
                    .map(|word| read_u32(&diagnostics, word))
                    .collect::<Vec<_>>(),
                vec![0, 0, 0, overflow]
            );
            recorded.release(&device);
        }
        bridge.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
    }

    #[test]
    fn bridge_validates_registry_and_abi_but_cpu_diagnostic_ranges_are_not_authoritative() {
        let route = route();
        assert_eq!(
            ParticleSimulationRendererBridge::validate_contract(
                8,
                4,
                GPU_PARTICLE_STRIDE - 4,
                PARTICLE_RENDER_INPUT_STRIDE,
                GPU_EVENT_STRIDE,
                &[route],
            ),
            Err(ParticleSimulationBridgeError::InvalidStride {
                resource: "simulation particle",
                expected: GPU_PARTICLE_STRIDE,
                actual: GPU_PARTICLE_STRIDE - 4,
            })
        );
        let mut diagnostic_range = route;
        diagnostic_range.active_start = u32::MAX;
        diagnostic_range.particle_count = u32::MAX;
        assert_eq!(
            ParticleSimulationRendererBridge::validate_contract(
                8,
                4,
                GPU_PARTICLE_STRIDE,
                PARTICLE_RENDER_INPUT_STRIDE,
                GPU_EVENT_STRIDE,
                &[diagnostic_range],
            ),
            Ok(())
        );
        diagnostic_range.particle_count = 0;
        assert!(ParticleSimulationRendererBridge::validate_contract(
            8,
            4,
            GPU_PARTICLE_STRIDE,
            PARTICLE_RENDER_INPUT_STRIDE,
            GPU_EVENT_STRIDE,
            &[diagnostic_range],
        )
        .is_ok());
        let full_registry: Vec<_> = (0..8)
            .map(|emitter_id| ParticleEmitterRenderRoute {
                emitter_id,
                ..route
            })
            .collect();
        assert!(ParticleSimulationRendererBridge::validate_contract(
            8,
            4,
            GPU_PARTICLE_STRIDE,
            PARTICLE_RENDER_INPUT_STRIDE,
            GPU_EVENT_STRIDE,
            &full_registry,
        )
        .is_ok());
        assert_eq!(
            ParticleSimulationRendererBridge::validate_contract(
                8,
                4,
                GPU_PARTICLE_STRIDE,
                PARTICLE_RENDER_INPUT_STRIDE,
                GPU_EVENT_STRIDE,
                &[route, diagnostic_range],
            ),
            Err(ParticleSimulationBridgeError::DuplicateEmitterId { emitter_id: 7 })
        );
        assert_eq!(
            ParticleSimulationRendererBridge::validate_contract(
                1,
                4,
                GPU_PARTICLE_STRIDE,
                PARTICLE_RENDER_INPUT_STRIDE,
                GPU_EVENT_STRIDE,
                &[route, diagnostic_range],
            ),
            Err(ParticleSimulationBridgeError::TooManyEmitterRoutes {
                requested: 2,
                capacity: 1
            })
        );
        assert!(ParticleSimulationRendererBridge::validate_lifetime(true, 0.5, 1.0).is_ok());
        assert!(ParticleSimulationRendererBridge::validate_lifetime(true, 1.0, 1.0).is_err());
        assert!(ParticleSimulationRendererBridge::validate_lifetime(true, 0.0, 0.0).is_ok());
    }

    #[test]
    fn physical_gpu_bridge_can_feed_renderer_without_cpu_particle_upload() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) if !require_hardware_gpu() => return,
            Err(HardwareDeviceError::NonHardware(_)) if !require_hardware_gpu() => return,
            Err(error) => panic!("physical GPU required for simulation bridge test: {error}"),
        };
        let simulation = GpuSimulationStage::new(&device, 8, 4).expect("simulation resources");
        let renderer =
            ParticleRendererGpu::new(&device, 8, route().renderer).expect("renderer resources");
        simulation.initialize_particles(
            &queue,
            &[GpuParticle::live(
                [0.0, 0.0, 0.0],
                [0.0; 3],
                2.0,
                0.2,
                42,
                7,
            )],
        );
        simulation.upload_inputs(
            &queue,
            &[],
            &[],
            GpuSimulationParams {
                dt: 0.0,
                ..GpuSimulationParams::default()
            },
        );
        renderer
            .set_view(&queue, ParticleCullView::unrestricted())
            .expect("view upload without input upload");
        let mut bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[route()])
                .expect("bridge resources");
        bridge.upload_routes(&queue).expect("registry upload");
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let simulation_outputs = simulation
            .append_to_graph(&mut graph)
            .expect("simulation graph");
        let bridge_outputs = bridge
            .append_to_graph(&mut graph, &simulation_outputs)
            .expect("bridge graph");
        renderer
            .append_to_graph_with_input(&mut graph, bridge_outputs.renderer_input)
            .expect("renderer graph");
        let recorded = graph.execute(&device, &queue).expect("GPU graph executes");
        let pixels = queue
            .read_texture_rgba8(
                renderer.raster_target(),
                crate::particle::renderer_matrix::RENDERER_MATRIX_TARGET_SIZE,
                crate::particle::renderer_matrix::RENDERER_MATRIX_TARGET_SIZE,
            )
            .expect("renderer texture readback");
        assert!(crate::count_lit_pixels(&pixels) > 0);
        recorded.release(&device);
        bridge.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
    }
}
