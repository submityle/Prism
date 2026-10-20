//! GPU particle simulation contract and compute implementation.
//!
//! This module is deliberately independent from the legacy `particle.rs`
//! renderer. It mirrors the device-independent particle contracts for force
//! entries, analytic colliders, lifetime, event records, and iteration-domain
//! dispatches in one fixed-capacity compute stage.

use alloc::{string::String, vec, vec::Vec};

use prism_render_architecture::particle::{
    collision::{Capsule, Collider, SdfSample, Sphere},
    forces::{Falloff, Force, ForceEntry},
    sort_cull::{Aabb, Plane},
    stages::{dispatch_domain_size, workgroup_count, StageCounts},
    IntegratorKind, Vec3,
};
use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, CommandBuffer, CommandEncoder,
    ComputeCommand, ComputePipelineDescriptor, ComputePipelineId, PipelineLayoutDescriptor,
    PipelineLayoutId, RenderDevice, RenderQueue, ShaderModuleDescriptor, ShaderModuleId,
    ShaderStages, PipelineStages,
};

use crate::gpu::{bind_buffer, buffer_entry};
use crate::readback::ReadBuffer;
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ImportedBuffer, PassFlags, RenderGraph,
};

/// Workgroup width shared by every particle simulation kernel.
pub const SIMULATION_WORKGROUP_SIZE: u32 = 64;
/// Bytes in one particle storage record.
pub const GPU_PARTICLE_STRIDE: u64 = 48;
/// Bytes in one encoded force entry.
pub const GPU_FORCE_STRIDE: u64 = 112;
/// Bytes in one encoded collider entry.
pub const GPU_COLLIDER_STRIDE: u64 = 96;
/// Bytes in one event record, matching `event_pass_buffers`.
pub const GPU_EVENT_STRIDE: u64 = 32;
/// Number of event channels defined by `EventKind`.
pub const EVENT_CHANNEL_COUNT: u32 = 4;
/// Number of atomic control words.
pub const CONTROL_WORDS: u32 = 12;
const PARAMS_BYTES: u64 = 80;
const MAX_FORCE_ENTRIES: u32 = 64;
const MAX_COLLIDERS: u32 = 32;

const FORCE_GRAVITY: u32 = 0;
const FORCE_LINEAR_DRAG: u32 = 1;
const FORCE_QUADRATIC_DRAG: u32 = 2;
const FORCE_WIND: u32 = 3;
const FORCE_VORTEX: u32 = 4;
const FORCE_POINT: u32 = 5;
const FORCE_LINE: u32 = 6;
const FORCE_RADIAL: u32 = 7;
const FORCE_ORBITAL: u32 = 8;
const FORCE_WELL: u32 = 9;
const FORCE_TURBULENCE: u32 = 10;
const FORCE_SPRING: u32 = 11;

const COLLIDER_PLANE: u32 = 0;
const COLLIDER_SPHERE: u32 = 1;
const COLLIDER_BOX: u32 = 2;
const COLLIDER_CAPSULE: u32 = 3;
const COLLIDER_SDF: u32 = 4;

/// One particle record crossing the simulation ABI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuParticle {
    /// World-space position.
    pub position: [f32; 3],
    /// Seconds lived so far.
    pub age: f32,
    /// World-space velocity.
    pub velocity: [f32; 3],
    /// Total lifetime; non-positive is immortal.
    pub lifetime: f32,
    /// Collision radius.
    pub radius: f32,
    /// Whether the pool slot is live.
    pub alive: bool,
    /// Stable particle identifier.
    pub particle_id: u32,
}

impl GpuParticle {
    /// Creates a live particle.
    #[must_use]
    pub const fn live(
        position: [f32; 3],
        velocity: [f32; 3],
        lifetime: f32,
        radius: f32,
        particle_id: u32,
    ) -> Self {
        Self {
            position,
            age: 0.0,
            velocity,
            lifetime,
            radius,
            alive: true,
            particle_id,
        }
    }

    /// Creates a free pool slot.
    #[must_use]
    pub const fn dead(particle_id: u32) -> Self {
        Self {
            position: [0.0; 3],
            age: 0.0,
            velocity: [0.0; 3],
            lifetime: 0.0,
            radius: 0.0,
            alive: false,
            particle_id,
        }
    }
}

/// Parameters for one simulation frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSimulationParams {
    /// Simulation timestep.
    pub dt: f32,
    /// Normal restitution, clamped by the shader to `[0, 1]`.
    pub restitution: f32,
    /// Coulomb friction, clamped by the shader to `[0, 1]`.
    pub friction: f32,
    /// Number of dead slots to claim in the spawn stage.
    pub spawn_count: u32,
    /// Spawn position.
    pub spawn_position: [f32; 3],
    /// Spawn velocity.
    pub spawn_velocity: [f32; 3],
    /// Lifetime assigned to spawned particles.
    pub spawn_lifetime: f32,
    /// Optional condition-event threshold on world X.
    pub condition_x: f32,
    /// Enables `OnCondition` when position.x is at or beyond `condition_x`.
    pub condition_enabled: bool,
    /// Integrator used by the simulation kernel.
    pub integrator: IntegratorKind,
}

impl Default for GpuSimulationParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            restitution: 0.5,
            friction: 0.0,
            spawn_count: 0,
            spawn_position: [0.0; 3],
            spawn_velocity: [0.0; 3],
            spawn_lifetime: 1.0,
            condition_x: 0.0,
            condition_enabled: false,
            integrator: IntegratorKind::SemiImplicitEuler,
        }
    }
}

/// Compute stage selected by the particle graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuSimulationStageKind {
    /// Clears counters and event cursors.
    Reset,
    /// Claims dead slots and raises `OnSpawn` events.
    Spawn,
    /// Applies forces, lifetime and deepest-contact collision response.
    Simulate,
    /// Appends live pool indices into a compact active list.
    Compact,
    /// Scatters source events into fixed per-kind compact runs.
    EventScatter,
}

impl GpuSimulationStageKind {
    const fn entry_point(self) -> &'static str {
        match self {
            Self::Reset => "reset",
            Self::Spawn => "spawn",
            Self::Simulate => "simulate",
            Self::Compact => "compact",
            Self::EventScatter => "event_scatter",
        }
    }

    const fn domain(self) -> prism_render_architecture::particle::IterationDomain {
        use prism_render_architecture::particle::IterationDomain;
        match self {
            Self::Reset => IterationDomain::Custom(1),
            Self::Spawn | Self::Simulate | Self::Compact => IterationDomain::PerParticle,
            Self::EventScatter => IterationDomain::PerEvent,
        }
    }
}

/// A host-recorded stage dispatch derived from the CPU stage contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuStageDispatch {
    /// Stage entry point.
    pub kind: GpuSimulationStageKind,
    /// CPU contract domain.
    pub domain: prism_render_architecture::particle::IterationDomain,
    /// Invocation upper bound supplied to the shader.
    pub invocations: u32,
    /// Number of 64-wide workgroups recorded.
    pub workgroups: u32,
}

/// Builds a deterministic dispatch plan from the architecture's stage counts.
#[must_use]
pub fn gpu_stage_dispatch_plan(counts: StageCounts, pool_capacity: u32) -> Vec<GpuStageDispatch> {
    let stages = [
        (GpuSimulationStageKind::Reset, 1),
        (GpuSimulationStageKind::Spawn, pool_capacity),
        (GpuSimulationStageKind::Simulate, pool_capacity),
        (GpuSimulationStageKind::Compact, pool_capacity),
        (
            GpuSimulationStageKind::EventScatter,
            dispatch_domain_size(
                prism_render_architecture::particle::IterationDomain::PerEvent,
                counts,
            ),
        ),
    ];
    stages
        .into_iter()
        .map(|(kind, invocations)| GpuStageDispatch {
            kind,
            domain: kind.domain(),
            invocations,
            workgroups: workgroup_count(invocations, SIMULATION_WORKGROUP_SIZE).max(1),
        })
        .collect()
}

/// The GPU-authored results used by verification and diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuSimulationOutput {
    /// Particle pool after the frame.
    pub particles: Vec<GpuParticle>,
    /// GPU-compacted live pool indices.
    pub active_indices: Vec<u32>,
    /// Raw event records in atomic append order.
    pub source_events: Vec<GpuEventRecord>,
    /// Events compacted into four fixed channel runs.
    pub compact_events: Vec<GpuEventRecord>,
    /// Raw control words; word 0 is source count, word 2 is live count.
    pub control: [u32; CONTROL_WORDS as usize],
}

/// SSA handles published by one GPU-authoritative simulation frame.
#[derive(Clone, Copy, Debug)]
pub struct GpuSimulationGraphOutputs {
    /// Particle pool after spawn and simulation.
    pub particles: BufferHandle,
    /// Compacted live pool indices.
    pub active_indices: BufferHandle,
    /// Event records grouped by event channel.
    pub compact_events: BufferHandle,
    /// Counters including live and event counts.
    pub control: BufferHandle,
}

/// The 32-byte event ABI shared with `EventScatterBuffer`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuEventRecord {
    /// Position in xyz and originator age in w.
    pub payload: [f32; 4],
    /// Event kind as `EventKind`'s stable ordinal.
    pub kind: u32,
    /// Originating pool slot.
    pub particle_index: u32,
}

/// GPU particle simulation stage. All simulation state is device-owned.
pub struct GpuSimulationStage {
    capacity: u32,
    event_capacity: u32,
    channel_capacity: u32,
    params: BufferId,
    particles: BufferId,
    forces: BufferId,
    colliders: BufferId,
    control: BufferId,
    source_events: BufferId,
    compact_events: BufferId,
    event_cursors: BufferId,
    channel_offsets: BufferId,
    active_indices: BufferId,
    bind_group_layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    bind_group: BindGroupId,
    shader: ShaderModuleId,
    pipelines: [ComputePipelineId; 5],
}

impl GpuSimulationStage {
    /// Allocates a fixed-capacity stage and its complete compute ABI.
    #[must_use]
    pub fn new(device: &impl RenderDevice, capacity: u32, event_capacity: u32) -> Self {
        assert!(capacity > 0, "a GPU particle pool needs capacity");
        assert!(event_capacity > 0, "a GPU event pool needs capacity");
        let channel_capacity = event_capacity;
        let storage_rw = BufferUsages::STORAGE
            .union(BufferUsages::COPY_DST)
            .union(BufferUsages::COPY_SRC);
        let storage_ro = BufferUsages::STORAGE.union(BufferUsages::COPY_DST);
        let params = device.create_buffer(&BufferDescriptor::new(
            PARAMS_BYTES,
            BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
        ));
        let particles = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * GPU_PARTICLE_STRIDE,
            storage_rw,
        ));
        let forces = device.create_buffer(&BufferDescriptor::new(
            u64::from(MAX_FORCE_ENTRIES) * GPU_FORCE_STRIDE,
            storage_ro,
        ));
        let colliders = device.create_buffer(&BufferDescriptor::new(
            u64::from(MAX_COLLIDERS) * GPU_COLLIDER_STRIDE,
            storage_ro,
        ));
        let control = device.create_buffer(&BufferDescriptor::new(
            u64::from(CONTROL_WORDS) * 4,
            storage_rw,
        ));
        let source_events = device.create_buffer(&BufferDescriptor::new(
            u64::from(event_capacity) * GPU_EVENT_STRIDE,
            storage_rw,
        ));
        let compact_events = device.create_buffer(&BufferDescriptor::new(
            u64::from(channel_capacity) * u64::from(EVENT_CHANNEL_COUNT) * GPU_EVENT_STRIDE,
            storage_rw,
        ));
        let event_cursors = device.create_buffer(&BufferDescriptor::new(
            u64::from(EVENT_CHANNEL_COUNT) * 4,
            storage_rw,
        ));
        let channel_offsets = device.create_buffer(&BufferDescriptor::new(
            u64::from(EVENT_CHANNEL_COUNT) * 4,
            storage_ro,
        ));
        let active_indices =
            device.create_buffer(&BufferDescriptor::new(u64::from(capacity) * 4, storage_rw));

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/particle_sim_stages.wgsl"
        )));
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-simulation-bgl")),
            entries: vec![
                buffer_entry(0, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
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
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    9,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-simulation-pl")),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-simulation-bg")),
            layout: Some(bind_group_layout),
            entries: vec![
                bind_buffer(0, params),
                bind_buffer(1, particles),
                bind_buffer(2, forces),
                bind_buffer(3, colliders),
                bind_buffer(4, control),
                bind_buffer(5, source_events),
                bind_buffer(6, compact_events),
                bind_buffer(7, event_cursors),
                bind_buffer(8, channel_offsets),
                bind_buffer(9, active_indices),
            ],
        });
        let pipelines = [
            GpuSimulationStageKind::Reset,
            GpuSimulationStageKind::Spawn,
            GpuSimulationStageKind::Simulate,
            GpuSimulationStageKind::Compact,
            GpuSimulationStageKind::EventScatter,
        ]
        .map(|kind| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(format!("particle-simulation-{}", kind.entry_point())),
                layout: Some(pipeline_layout),
                module: shader,
                entry_point: kind.entry_point().into(),
            })
        });
        Self {
            capacity,
            event_capacity,
            channel_capacity,
            params,
            particles,
            forces,
            colliders,
            control,
            source_events,
            compact_events,
            event_cursors,
            channel_offsets,
            active_indices,
            bind_group_layout,
            pipeline_layout,
            bind_group,
            shader,
            pipelines,
        }
    }

    /// Initializes the persistent GPU particle pool once after creation or
    /// device recovery. Subsequent frames must not call this method: the
    /// simulation kernels own the pool between frames.
    pub fn initialize_particles(&self, queue: &impl RenderQueue, particles: &[GpuParticle]) {
        assert!(particles.len() <= self.capacity as usize);
        let mut particle_bytes = vec![0; self.capacity as usize * GPU_PARTICLE_STRIDE as usize];
        for (index, particle) in particles.iter().enumerate() {
            encode_particle(
                &mut particle_bytes[index * GPU_PARTICLE_STRIDE as usize
                    ..(index + 1) * GPU_PARTICLE_STRIDE as usize],
                *particle,
            );
        }
        queue.write_buffer(self.particles, 0, &particle_bytes);
    }

    /// Uploads only the per-frame simulation inputs. The particle pool is
    /// intentionally omitted so GPU-authoritative state remains resident
    /// across frames and is never advanced by a competing CPU snapshot path.
    pub fn upload_inputs(
        &self,
        queue: &impl RenderQueue,
        forces: &[ForceEntry],
        colliders: &[Collider],
        params: GpuSimulationParams,
    ) {
        assert!(
            forces.len() <= MAX_FORCE_ENTRIES as usize,
            "too many force entries for GPU ABI"
        );
        assert!(
            colliders.len() <= MAX_COLLIDERS as usize,
            "too many colliders for GPU ABI"
        );
        let force_bytes = encode_force_buffer(forces);
        queue.write_buffer(self.forces, 0, &force_bytes);
        let collider_bytes = encode_collider_buffer(colliders);
        queue.write_buffer(self.colliders, 0, &collider_bytes);
        queue.write_buffer(
            self.params,
            0,
            &encode_params(
                self.capacity,
                forces.len() as u32,
                colliders.len() as u32,
                self.event_capacity,
                params,
            ),
        );
        let offsets = [
            0,
            self.channel_capacity,
            self.channel_capacity * 2,
            self.channel_capacity * 3,
        ];
        queue.write_buffer(self.channel_offsets, 0, &u32_bytes(&offsets));
    }

    /// Returns the fixed GPU stage plan used by `encode`.
    #[must_use]
    pub fn dispatch_plan(&self) -> Vec<GpuStageDispatch> {
        gpu_stage_dispatch_plan(
            StageCounts {
                live_particles: self.capacity,
                events: self.event_capacity,
                ..StageCounts::default()
            },
            self.capacity,
        )
    }

    /// Records all five compute stages in dependency order.
    #[must_use]
    pub fn encode(&self) -> CommandBuffer {
        let mut encoder = CommandEncoder::labeled("particle-gpu-simulation");
        for dispatch in self.dispatch_plan() {
            encoder.push_compute_pass(
                Some(format!("particle-{}", dispatch.kind.entry_point())),
                vec![
                    ComputeCommand::SetPipeline(self.pipeline(dispatch.kind)),
                    ComputeCommand::SetBindGroup {
                        index: 0,
                        bind_group: self.bind_group,
                        dynamic_offsets: Vec::new(),
                    },
                    ComputeCommand::Dispatch {
                        x: dispatch.workgroups,
                        y: 1,
                        z: 1,
                    },
                ],
            );
        }
        encoder.finish()
    }

    /// Appends the complete GPU-authoritative simulation to an existing frame
    /// graph. The five passes are deliberately declared with SSA buffer
    /// versions so the graph compiler owns the reset → spawn → simulate →
    /// compact → event-scatter ordering and the corresponding barriers.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
    ) -> Result<GpuSimulationGraphOutputs, CompileError> {
        let params = import_buffer(
            graph,
            "particle-simulation-params",
            self.params,
            PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            rhi::BufferState::copy_dst(),
        );
        let particles = import_buffer(
            graph,
            "particle-simulation-particles",
            self.particles,
            u64::from(self.capacity) * GPU_PARTICLE_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            rhi::BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let forces = import_buffer(
            graph,
            "particle-simulation-forces",
            self.forces,
            u64::from(MAX_FORCE_ENTRIES) * GPU_FORCE_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            rhi::BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        let colliders = import_buffer(
            graph,
            "particle-simulation-colliders",
            self.colliders,
            u64::from(MAX_COLLIDERS) * GPU_COLLIDER_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            rhi::BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        let control = import_buffer(
            graph,
            "particle-simulation-control",
            self.control,
            u64::from(CONTROL_WORDS) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            rhi::BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let source_events = import_buffer(
            graph,
            "particle-simulation-source-events",
            self.source_events,
            u64::from(self.event_capacity) * GPU_EVENT_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            rhi::BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let compact_events = import_buffer(
            graph,
            "particle-simulation-compact-events",
            self.compact_events,
            u64::from(self.channel_capacity)
                * u64::from(EVENT_CHANNEL_COUNT)
                * GPU_EVENT_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            rhi::BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let event_cursors = import_buffer(
            graph,
            "particle-simulation-event-cursors",
            self.event_cursors,
            u64::from(EVENT_CHANNEL_COUNT) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            rhi::BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let channel_offsets = import_buffer(
            graph,
            "particle-simulation-channel-offsets",
            self.channel_offsets,
            u64::from(EVENT_CHANNEL_COUNT) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            rhi::BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        let active_indices = import_buffer(
            graph,
            "particle-simulation-active-indices",
            self.active_indices,
            u64::from(self.capacity) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            rhi::BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );

        let bind_group = self.bind_group;
        let reset_pipeline = self.pipeline(GpuSimulationStageKind::Reset);
        let mut control_after_reset = control;
        let mut cursors_after_reset = event_cursors;
        graph.try_add_compute_pass("particle-simulation-reset", PassFlags::SIDE_EFFECT, |builder| {
            control_after_reset = builder.write_storage_buffer(control);
            cursors_after_reset = builder.write_storage_buffer(event_cursors);
            move |ctx| {
                ctx.set_compute_pipeline(reset_pipeline)
                    .set_compute_bind_group(0, bind_group, Vec::new())
                    .dispatch(1, 1, 1);
            }
        })?;

        let dispatch_groups = self.capacity.div_ceil(SIMULATION_WORKGROUP_SIZE);
        let spawn_pipeline = self.pipeline(GpuSimulationStageKind::Spawn);
        let mut particles_after_spawn = particles;
        let mut control_after_spawn = control_after_reset;
        let mut source_after_spawn = source_events;
        graph.try_add_compute_pass("particle-simulation-spawn", PassFlags::SIDE_EFFECT, |builder| {
            builder.read_uniform(params);
            builder.read_storage_buffer(forces);
            builder.read_storage_buffer(colliders);
            particles_after_spawn = builder.write_storage_buffer(particles);
            control_after_spawn = builder.write_storage_buffer(control_after_reset);
            source_after_spawn = builder.write_storage_buffer(source_events);
            move |ctx| {
                ctx.set_compute_pipeline(spawn_pipeline)
                    .set_compute_bind_group(0, bind_group, Vec::new())
                    .dispatch(dispatch_groups, 1, 1);
            }
        })?;

        let simulate_pipeline = self.pipeline(GpuSimulationStageKind::Simulate);
        let mut particles_after_simulate = particles_after_spawn;
        let mut control_after_simulate = control_after_spawn;
        let mut source_after_simulate = source_after_spawn;
        graph.try_add_compute_pass(
            "particle-simulation-simulate",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_uniform(params);
                builder.read_storage_buffer(forces);
                builder.read_storage_buffer(colliders);
                particles_after_simulate = builder.write_storage_buffer(particles_after_spawn);
                control_after_simulate = builder.write_storage_buffer(control_after_spawn);
                source_after_simulate = builder.write_storage_buffer(source_after_spawn);
                move |ctx| {
                    ctx.set_compute_pipeline(simulate_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(dispatch_groups, 1, 1);
                }
            },
        )?;

        let compact_pipeline = self.pipeline(GpuSimulationStageKind::Compact);
        let mut control_after_compact = control_after_simulate;
        let mut active_after_compact = active_indices;
        graph.try_add_compute_pass(
            "particle-simulation-compact",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(particles_after_simulate);
                active_after_compact = builder.write_storage_buffer(active_indices);
                control_after_compact = builder.write_storage_buffer(control_after_simulate);
                move |ctx| {
                    ctx.set_compute_pipeline(compact_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(dispatch_groups, 1, 1);
                }
            },
        )?;

        let scatter_pipeline = self.pipeline(GpuSimulationStageKind::EventScatter);
        let event_groups = self.event_capacity.div_ceil(SIMULATION_WORKGROUP_SIZE);
        graph.try_add_compute_pass(
            "particle-simulation-event-scatter",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_uniform(params);
                builder.read_storage_buffer(source_after_simulate);
                builder.read_storage_buffer(channel_offsets);
                builder.read_storage_buffer(control_after_compact);
                builder.read_storage_buffer(cursors_after_reset);
                builder.write_storage_buffer(compact_events);
                builder.write_storage_buffer(cursors_after_reset);
                builder.write_storage_buffer(control_after_compact);
                move |ctx| {
                    ctx.set_compute_pipeline(scatter_pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(event_groups, 1, 1);
                }
            },
        )?;
        Ok(GpuSimulationGraphOutputs {
            particles: particles_after_simulate,
            active_indices: active_after_compact,
            compact_events,
            control: control_after_compact,
        })
    }

    /// Submits and reads the GPU-authored state for verification.
    #[must_use]
    pub fn run(&self, queue: &(impl RenderQueue + ReadBuffer)) -> GpuSimulationOutput {
        queue.submit(&[self.encode()]);
        let control_bytes = queue.read_buffer(self.control, u64::from(CONTROL_WORDS) * 4);
        let control = read_u32_array::<12>(&control_bytes);
        let live_count = control[2].min(self.capacity) as u64;
        let source_count = control[0].min(self.event_capacity) as u64;
        let cursors = read_u32_array::<4>(
            &queue.read_buffer(self.event_cursors, u64::from(EVENT_CHANNEL_COUNT) * 4),
        );
        let compact_count = control[8].min(self.channel_capacity * EVENT_CHANNEL_COUNT) as usize;
        let compact_bytes = queue.read_buffer(
            self.compact_events,
            u64::from(self.channel_capacity) * u64::from(EVENT_CHANNEL_COUNT) * GPU_EVENT_STRIDE,
        );
        let mut compact_events = Vec::with_capacity(compact_count);
        for (channel, cursor) in cursors.into_iter().enumerate() {
            let count = cursor.min(self.channel_capacity) as usize;
            let start = channel * self.channel_capacity as usize * GPU_EVENT_STRIDE as usize;
            let end = start + count * GPU_EVENT_STRIDE as usize;
            compact_events.extend(decode_events(&compact_bytes[start..end]));
        }
        debug_assert_eq!(compact_events.len(), compact_count);
        GpuSimulationOutput {
            particles: decode_particles(&queue.read_buffer(
                self.particles,
                u64::from(self.capacity) * GPU_PARTICLE_STRIDE,
            )),
            active_indices: read_u32s(&queue.read_buffer(self.active_indices, live_count * 4)),
            source_events: decode_events(
                &queue.read_buffer(self.source_events, source_count * GPU_EVENT_STRIDE),
            ),
            compact_events,
            control,
        }
    }

    fn pipeline(&self, kind: GpuSimulationStageKind) -> ComputePipelineId {
        match kind {
            GpuSimulationStageKind::Reset => self.pipelines[0],
            GpuSimulationStageKind::Spawn => self.pipelines[1],
            GpuSimulationStageKind::Simulate => self.pipelines[2],
            GpuSimulationStageKind::Compact => self.pipelines[3],
            GpuSimulationStageKind::EventScatter => self.pipelines[4],
        }
    }

    /// GPU particle pool buffer.
    #[must_use]
    pub const fn particles(&self) -> BufferId {
        self.particles
    }
    /// GPU source event buffer.
    #[must_use]
    pub const fn source_events(&self) -> BufferId {
        self.source_events
    }
    /// GPU compact event buffer.
    #[must_use]
    pub const fn compact_events(&self) -> BufferId {
        self.compact_events
    }
    /// GPU atomic control buffer.
    #[must_use]
    pub const fn control(&self) -> BufferId {
        self.control
    }

    /// Releases all resources owned by the stage.
    pub fn destroy(&self, device: &impl RenderDevice) {
        for buffer in [
            self.params,
            self.particles,
            self.forces,
            self.colliders,
            self.control,
            self.source_events,
            self.compact_events,
            self.event_cursors,
            self.channel_offsets,
            self.active_indices,
        ] {
            device.destroy_buffer(buffer);
        }
        for pipeline in self.pipelines {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_bind_group(self.bind_group);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.bind_group_layout);
        device.destroy_shader_module(self.shader);
    }
}

fn import_buffer(
    graph: &mut RenderGraph,
    name: &str,
    buffer: BufferId,
    size: u64,
    usage: BufferUsages,
    entry_state: rhi::BufferState,
) -> BufferHandle {
    graph.import_buffer(
        name,
        BufferDesc::new(size, usage),
        ImportedBuffer { buffer, entry_state },
    )
}

fn pipeline_force_kind(force: Force) -> u32 {
    match force {
        Force::Gravity { .. } => FORCE_GRAVITY,
        Force::LinearDrag { .. } => FORCE_LINEAR_DRAG,
        Force::QuadraticDrag { .. } => FORCE_QUADRATIC_DRAG,
        Force::Wind { .. } => FORCE_WIND,
        Force::Vortex { .. } => FORCE_VORTEX,
        Force::PointAttractor { .. } => FORCE_POINT,
        Force::LineAttractor { .. } => FORCE_LINE,
        Force::Radial { .. } => FORCE_RADIAL,
        Force::Orbital { .. } => FORCE_ORBITAL,
        Force::GravityWell { .. } => FORCE_WELL,
        Force::Turbulence { .. } => FORCE_TURBULENCE,
        Force::SpringDamper { .. } => FORCE_SPRING,
    }
}

fn falloff_id(falloff: Falloff) -> u32 {
    match falloff {
        Falloff::Constant => 0,
        Falloff::Linear => 1,
        Falloff::Smoothstep => 2,
        Falloff::InverseSquare => 3,
    }
}

fn encode_force_buffer(entries: &[ForceEntry]) -> Vec<u8> {
    let mut out = vec![0; MAX_FORCE_ENTRIES as usize * GPU_FORCE_STRIDE as usize];
    for (entry_index, entry) in entries.iter().enumerate() {
        let record = &mut out[entry_index * GPU_FORCE_STRIDE as usize
            ..(entry_index + 1) * GPU_FORCE_STRIDE as usize];
        let (kind, falloff, data) = force_data(entry.force);
        put_u32(record, 0, kind);
        put_u32(record, 4, u32::from(entry.enabled));
        put_u32(record, 8, falloff);
        put_f32(record, 12, entry.gate_radius);
        for (component, value) in [
            entry.gate_center.x,
            entry.gate_center.y,
            entry.gate_center.z,
            entry.weight,
        ]
        .into_iter()
        .enumerate()
        {
            put_f32(record, 16 + component * 4, value);
        }
        for (index, values) in data.into_iter().enumerate() {
            for (component, value) in values.into_iter().enumerate() {
                put_f32(record, 32 + index * 16 + component * 4, value);
            }
        }
    }
    out
}

fn force_data(force: Force) -> (u32, u32, [[f32; 4]; 5]) {
    let mut data = [[0.0; 4]; 5];
    let mut falloff = 0;
    match force {
        Force::Gravity { acceleration } => data[0] = v4(acceleration, 0.0),
        Force::LinearDrag { coefficient, dt } => data[0] = [coefficient, dt, 0.0, 0.0],
        Force::QuadraticDrag { coefficient } => data[0][0] = coefficient,
        Force::Wind {
            velocity,
            coefficient,
        } => data[0] = v4(velocity, coefficient),
        Force::Vortex {
            center,
            axis,
            strength,
        } => {
            data[0] = v4(center, 0.0);
            data[1] = v4(axis, strength);
        }
        Force::PointAttractor {
            point,
            strength,
            radius,
            softening,
            falloff: f,
        }
        | Force::Radial {
            center: point,
            strength,
            radius,
            softening,
            falloff: f,
        } => {
            data[0] = v4(point, 0.0);
            data[1] = [strength, radius, softening, 0.0];
            falloff = falloff_id(f);
        }
        Force::LineAttractor {
            point,
            direction,
            strength,
            radius,
            softening,
            falloff: f,
        } => {
            data[0] = v4(point, 0.0);
            data[1] = v4(direction, 0.0);
            data[2] = [strength, radius, softening, 0.0];
            falloff = falloff_id(f);
        }
        Force::Orbital {
            center,
            axis,
            target_radius,
            tangential_strength,
            radial_stiffness,
        } => {
            data[0] = v4(center, 0.0);
            data[1] = v4(axis, 0.0);
            data[2] = [target_radius, tangential_strength, radial_stiffness, 0.0];
        }
        Force::GravityWell {
            center,
            strength,
            softening,
        } => {
            data[0] = v4(center, 0.0);
            data[1] = [strength, softening, 0.0, 0.0];
        }
        Force::Turbulence {
            seed,
            frequency,
            amplitude,
            octaves,
            frequency_multiplier,
            amplitude_multiplier,
            epsilon,
        } => {
            data[0] = [
                frequency,
                amplitude,
                frequency_multiplier,
                amplitude_multiplier,
            ];
            data[1] = [f32::from_bits(seed), octaves as f32, epsilon, 0.0];
        }
        Force::SpringDamper {
            anchor,
            stiffness,
            damping,
        } => {
            data[0] = v4(anchor, 0.0);
            data[1] = [stiffness, damping, 0.0, 0.0];
        }
    }
    (pipeline_force_kind(force), falloff, data)
}

fn encode_collider_buffer(colliders: &[Collider]) -> Vec<u8> {
    let mut out = vec![0; MAX_COLLIDERS as usize * GPU_COLLIDER_STRIDE as usize];
    for (collider_index, collider) in colliders.iter().enumerate() {
        let record = &mut out[collider_index * GPU_COLLIDER_STRIDE as usize
            ..(collider_index + 1) * GPU_COLLIDER_STRIDE as usize];
        let (kind, data) = collider_data(*collider);
        put_u32(record, 0, kind);
        for (index, values) in data.into_iter().enumerate() {
            for (component, value) in values.into_iter().enumerate() {
                put_f32(record, 16 + index * 16 + component * 4, value);
            }
        }
    }
    out
}

fn collider_data(collider: Collider) -> (u32, [[f32; 4]; 5]) {
    let mut data = [[0.0; 4]; 5];
    match collider {
        Collider::HalfSpace(Plane { normal, d }) => data[0] = v4(normal, d),
        Collider::Sphere(Sphere { center, radius }) => data[0] = v4(center, radius),
        Collider::Box(Aabb { min, max }) => {
            data[0] = v4(min, 0.0);
            data[1] = v4(max, 0.0);
        }
        Collider::Capsule(Capsule { a, b, radius }) => {
            data[0] = v4(a, 0.0);
            data[1] = v4(b, radius);
        }
        Collider::Sdf(SdfSample { distance, gradient }) => data[0] = v4(gradient, distance),
    }
    let kind = match collider {
        Collider::HalfSpace(_) => COLLIDER_PLANE,
        Collider::Sphere(_) => COLLIDER_SPHERE,
        Collider::Box(_) => COLLIDER_BOX,
        Collider::Capsule(_) => COLLIDER_CAPSULE,
        Collider::Sdf(_) => COLLIDER_SDF,
    };
    (kind, data)
}

fn encode_params(
    capacity: u32,
    force_count: u32,
    collider_count: u32,
    event_capacity: u32,
    params: GpuSimulationParams,
) -> Vec<u8> {
    let mut out = vec![0; PARAMS_BYTES as usize];
    put_u32(&mut out, 0, capacity);
    put_u32(&mut out, 4, force_count);
    put_u32(&mut out, 8, collider_count);
    put_u32(&mut out, 12, event_capacity);
    put_f32(&mut out, 16, params.dt);
    put_f32(&mut out, 20, params.restitution);
    put_f32(&mut out, 24, params.friction);
    put_f32(&mut out, 28, params.spawn_lifetime);
    for (index, value) in params.spawn_position.into_iter().enumerate() {
        put_f32(&mut out, 32 + index * 4, value);
    }
    for (index, value) in params.spawn_velocity.into_iter().enumerate() {
        put_f32(&mut out, 48 + index * 4, value);
    }
    put_u32(&mut out, 64, params.spawn_count);
    put_u32(&mut out, 68, u32::from(params.condition_enabled));
    put_u32(&mut out, 72, integrator_id(params.integrator));
    put_f32(&mut out, 76, params.condition_x);
    out
}

fn integrator_id(kind: IntegratorKind) -> u32 {
    match kind {
        IntegratorKind::SemiImplicitEuler => 0,
        IntegratorKind::Verlet => 1,
        IntegratorKind::Rk2 => 2,
    }
}

fn encode_particle(dst: &mut [u8], particle: GpuParticle) {
    for (index, value) in particle.position.into_iter().enumerate() {
        put_f32(dst, index * 4, value);
    }
    put_f32(dst, 12, particle.age);
    for (index, value) in particle.velocity.into_iter().enumerate() {
        put_f32(dst, 16 + index * 4, value);
    }
    put_f32(dst, 28, particle.lifetime);
    put_f32(dst, 32, particle.radius);
    put_u32(dst, 36, u32::from(particle.alive));
    put_u32(dst, 40, particle.particle_id);
}

fn decode_particles(bytes: &[u8]) -> Vec<GpuParticle> {
    bytes
        .chunks_exact(48)
        .map(|chunk| GpuParticle {
            position: [f32_at(chunk, 0), f32_at(chunk, 4), f32_at(chunk, 8)],
            age: f32_at(chunk, 12),
            velocity: [f32_at(chunk, 16), f32_at(chunk, 20), f32_at(chunk, 24)],
            lifetime: f32_at(chunk, 28),
            radius: f32_at(chunk, 32),
            alive: u32_at(&chunk[36..]) != 0,
            particle_id: u32_at(&chunk[40..]),
        })
        .collect()
}

fn decode_events(bytes: &[u8]) -> Vec<GpuEventRecord> {
    bytes
        .chunks_exact(32)
        .map(|chunk| GpuEventRecord {
            payload: [
                f32_at(chunk, 0),
                f32_at(chunk, 4),
                f32_at(chunk, 8),
                f32_at(chunk, 12),
            ],
            kind: u32_at(&chunk[16..]),
            particle_index: u32_at(&chunk[20..]),
        })
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(u32_at).collect()
}
fn read_u32_array<const N: usize>(bytes: &[u8]) -> [u32; N] {
    core::array::from_fn(|i| u32_at(&bytes[i * 4..i * 4 + 4]))
}
fn v4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}
fn put_u32(dst: &mut [u8], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_f32(dst: &mut [u8], offset: usize, value: f32) {
    put_u32(dst, offset, value.to_bits());
}
fn u32_at(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().expect("u32 ABI"))
}
fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(u32_at(&bytes[offset..offset + 4]))
}

#[cfg(test)]
#[path = "gpu_simulation_tests.rs"]
mod tests;
