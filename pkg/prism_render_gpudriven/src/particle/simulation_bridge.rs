//! GPU-only handoff from the Ember simulation pool to the particle renderer.
//!
//! The bridge is deliberately a graph contract, not a CPU extraction layer.
//! It consumes the SSA handles published by [`GpuSimulationStage`] and writes
//! the renderer's 32-byte `ParticleRenderInput` ABI into the persistent input
//! buffer owned by [`ParticleRendererGpu`]. No particle record is read back and
//! no simulation pool is uploaded again between the simulation and renderer
//! passes.

use alloc::{string::String, vec::Vec};

use super::renderer_matrix::{ParticleRendererGpu, ParticleRendererRoute};
use crate::gpu::{bind_buffer, buffer_entry};
use crate::gpu_simulation::{
    GpuSimulationGraphOutputs, GpuSimulationStage, GPU_EVENT_STRIDE, GPU_PARTICLE_STRIDE,
};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferState, BufferUsages,
    ComputePipelineDescriptor, ComputePipelineId, PipelineLayoutDescriptor, PipelineLayoutId,
    PipelineStages, RenderDevice, ShaderModuleDescriptor, ShaderModuleId, ShaderStages,
};
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ImportedBuffer, PassFlags, RenderGraph,
};

/// The bridge output ABI consumed by the renderer matrix.
pub const PARTICLE_RENDER_INPUT_STRIDE: u64 = 32;
/// One event flag word stored per simulation-pool slot.
pub const PARTICLE_EVENT_FLAG_STRIDE: u64 = 4;
/// The bridge writes this flag when an active index violates the lifetime ABI.
pub const PARTICLE_BRIDGE_FLAG_INVALID_LIFETIME: u32 = 1;
/// The bridge writes this flag when the particle produced at least one event.
pub const PARTICLE_BRIDGE_FLAG_EVENT: u32 = 2;
const BRIDGE_WORKGROUP_SIZE: u32 = 64;
const VALIDATION_BYTES: u64 = 4;

/// A contiguous segment of the simulation active-index list routed to one
/// renderer plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleEmitterRenderRoute {
    /// Stable emitter identifier supplied by the caller.
    pub emitter_id: u32,
    /// First element in the GPU active-index list and bridge output buffer.
    pub active_start: u32,
    /// Number of active particles in this emitter segment.
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
    /// The bridge requires at least one bind group and seven bindings.
    BindingLimitTooSmall {
        /// Required binding count.
        requested: u32,
        /// Device limit.
        limit: u32,
    },
    /// A bridge dispatch exceeds the device's workgroup limit.
    DispatchTooLarge {
        /// Requested X workgroups.
        workgroups: u32,
        /// Device limit.
        limit: u32,
    },
    /// A route has no particles and therefore cannot describe a render batch.
    EmptyEmitterRange { emitter_id: u32 },
    /// A route extends outside the fixed active-index/output capacity.
    InvalidEmitterRange {
        /// Emitter identifier.
        emitter_id: u32,
        /// First active-list element.
        active_start: u32,
        /// Requested element count.
        particle_count: u32,
        /// Fixed simulation capacity.
        capacity: u32,
    },
    /// Routes must form a packed prefix so a route never aliases another one.
    NonContiguousEmitterRanges {
        /// End of the preceding route.
        expected_start: u32,
        /// Start supplied by the next route.
        actual_start: u32,
    },
    /// Emitter identifiers must be unique in one bridge plan.
    DuplicateEmitterId { emitter_id: u32 },
    /// A live simulation record violates the age/lifetime ABI.
    InvalidLifetime,
}

/// Graph handles authored by the bridge.
#[derive(Clone, Debug)]
pub struct ParticleSimulationBridgeGraphOutputs {
    /// The renderer input handle written by the GPU gather pass.
    pub renderer_input: BufferHandle,
    /// Per-pool-slot event flags consumed by the gather pass.
    pub event_flags: BufferHandle,
    /// GPU validation counter; nonzero means an active record violated the
    /// lifetime contract.
    pub validation: BufferHandle,
    /// The compact event handle consumed by the event-mark pass and retained
    /// here so downstream graph consumers can prove the binding contract.
    pub compact_events: BufferHandle,
    /// Per-emitter routing metadata for downstream renderer submissions.
    pub emitter_routes: Vec<ParticleEmitterRenderRoute>,
}

/// Persistent GPU resources for the simulation-to-renderer handoff.
pub struct ParticleSimulationRendererBridge {
    capacity: u32,
    event_capacity: u32,
    renderer_input: BufferId,
    event_flags: BufferId,
    validation: BufferId,
    shader: ShaderModuleId,
    layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    clear_pipeline: ComputePipelineId,
    mark_events_pipeline: ComputePipelineId,
    gather_pipeline: ComputePipelineId,
    bind_group: BindGroupId,
    emitter_routes: Vec<ParticleEmitterRenderRoute>,
}

impl ParticleSimulationRendererBridge {
    /// Validates the immutable ABI and packed emitter-route contract.
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

        let mut expected_start = 0;
        for (route_index, route) in emitter_routes.iter().copied().enumerate() {
            if route.particle_count == 0 {
                return Err(ParticleSimulationBridgeError::EmptyEmitterRange {
                    emitter_id: route.emitter_id,
                });
            }
            if emitter_routes[..route_index]
                .iter()
                .any(|previous| previous.emitter_id == route.emitter_id)
            {
                return Err(ParticleSimulationBridgeError::DuplicateEmitterId {
                    emitter_id: route.emitter_id,
                });
            }
            if route.active_start != expected_start {
                return Err(ParticleSimulationBridgeError::NonContiguousEmitterRanges {
                    expected_start,
                    actual_start: route.active_start,
                });
            }
            let end = route.active_start.checked_add(route.particle_count).ok_or(
                ParticleSimulationBridgeError::InvalidEmitterRange {
                    emitter_id: route.emitter_id,
                    active_start: route.active_start,
                    particle_count: route.particle_count,
                    capacity,
                },
            )?;
            if end > capacity {
                return Err(ParticleSimulationBridgeError::InvalidEmitterRange {
                    emitter_id: route.emitter_id,
                    active_start: route.active_start,
                    particle_count: route.particle_count,
                    capacity,
                });
            }
            expected_start = end;
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
    /// accessors and never destroys them.
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
            return Err(ParticleSimulationBridgeError::InvalidEmitterRange {
                emitter_id: u32::MAX,
                active_start: 0,
                particle_count: renderer.capacity(),
                capacity,
            });
        }

        let limits = device.capabilities().limits;
        if limits.max_bind_groups < 1 {
            return Err(ParticleSimulationBridgeError::BindingLimitTooSmall {
                requested: 1,
                limit: limits.max_bind_groups,
            });
        }
        if limits.max_bindings_per_bind_group < 7 {
            return Err(ParticleSimulationBridgeError::BindingLimitTooSmall {
                requested: 7,
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
        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_simulation_bridge.wgsl"
        )));
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
            entry_point: "gather".into(),
        });
        Ok(Self {
            capacity,
            event_capacity,
            renderer_input: renderer.input_buffer(),
            event_flags,
            validation,
            shader,
            layout,
            pipeline_layout,
            clear_pipeline,
            mark_events_pipeline,
            gather_pipeline,
            bind_group,
            emitter_routes: emitter_routes.to_vec(),
        })
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
        let validation = import_buffer(
            graph,
            "particle-simulation-renderer-bridge-validation",
            self.validation,
            VALIDATION_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );

        let clear_pipeline = self.clear_pipeline;
        let clear_bind_group = self.bind_group;
        let capacity = self.capacity;
        let mut flags_after_clear = event_flags;
        let mut input_after_clear = renderer_input;
        let mut validation_after_clear = validation;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-clear",
            PassFlags::SIDE_EFFECT,
            |builder| {
                flags_after_clear = builder.write_storage_buffer(event_flags);
                input_after_clear = builder.write_storage_buffer(renderer_input);
                validation_after_clear = builder.write_storage_buffer(validation);
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

        let gather_pipeline = self.gather_pipeline;
        let gather_bind_group = self.bind_group;
        let mut input_after_gather = input_after_clear;
        let mut validation_after_gather = validation_after_clear;
        graph.try_add_compute_pass(
            "particle-simulation-renderer-bridge-gather",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(simulation.particles);
                builder.read_storage_buffer(simulation.active_indices);
                builder.read_storage_buffer(simulation.control);
                builder.read_storage_buffer(flags_after_events);
                input_after_gather = builder.write_storage_buffer(input_after_clear);
                validation_after_gather = builder.write_storage_buffer(validation_after_clear);
                move |context| {
                    context
                        .set_compute_pipeline(gather_pipeline)
                        .set_compute_bind_group(0, gather_bind_group, Vec::new())
                        .dispatch(capacity.div_ceil(BRIDGE_WORKGROUP_SIZE), 1, 1);
                }
            },
        )?;

        Ok(ParticleSimulationBridgeGraphOutputs {
            renderer_input: input_after_gather,
            event_flags: flags_after_events,
            validation: validation_after_gather,
            compact_events: simulation.compact_events,
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

    /// Returns the configured emitter routes without exposing GPU ownership.
    #[must_use]
    pub fn emitter_routes(&self) -> &[ParticleEmitterRenderRoute] {
        &self.emitter_routes
    }

    /// Releases bridge-owned resources. The simulation and renderer stages
    /// remain the owners of their persistent buffers.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        device.destroy_bind_group(self.bind_group);
        for pipeline in [
            self.clear_pipeline,
            self.mark_events_pipeline,
            self.gather_pipeline,
        ] {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.layout);
        device.destroy_shader_module(self.shader);
        device.destroy_buffer(self.event_flags);
        device.destroy_buffer(self.validation);
    }
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
    use prism_render_driver_null::NullInstance;
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

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
    fn null_rhi_records_simulation_bridge_bindings_and_route_contract() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device");
        let simulation = GpuSimulationStage::new(&device, 8, 4);
        let renderer =
            ParticleRendererGpu::new(&device, 8, route().renderer).expect("renderer resources");
        renderer
            .set_view(&queue, ParticleCullView::unrestricted())
            .expect("view upload");
        let bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[route()])
                .expect("bridge resources");
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
        let plan = graph.compile().expect("combined graph compiles");
        assert!(plan.declared_pass_count >= 8);
        assert!(plan.barrier_count() >= 7);
        assert_eq!(bridge_outputs.emitter_routes, vec![route()]);
        assert_eq!(
            bridge_outputs.compact_events,
            simulation_outputs.compact_events
        );
        bridge.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
    }

    #[test]
    fn bridge_rejects_stride_capacity_range_and_lifetime_violations() {
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
        let mut invalid_range = route;
        invalid_range.active_start = 2;
        assert!(matches!(
            ParticleSimulationRendererBridge::validate_contract(
                8,
                4,
                GPU_PARTICLE_STRIDE,
                PARTICLE_RENDER_INPUT_STRIDE,
                GPU_EVENT_STRIDE,
                &[invalid_range],
            ),
            Err(ParticleSimulationBridgeError::NonContiguousEmitterRanges { .. })
        ));
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
        let simulation = GpuSimulationStage::new(&device, 8, 4);
        let renderer =
            ParticleRendererGpu::new(&device, 8, route().renderer).expect("renderer resources");
        simulation.initialize_particles(
            &queue,
            &[GpuParticle::live([0.0, 0.0, 0.0], [0.0; 3], 2.0, 0.2, 42)],
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
        let bridge =
            ParticleSimulationRendererBridge::new(&device, &simulation, &renderer, &[route()])
                .expect("bridge resources");
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
        let pixels = queue.read_texture_rgba8(
            renderer.raster_target(),
            crate::particle::renderer_matrix::RENDERER_MATRIX_TARGET_SIZE,
            crate::particle::renderer_matrix::RENDERER_MATRIX_TARGET_SIZE,
        );
        assert!(crate::count_lit_pixels(&pixels) > 0);
        recorded.release(&device);
        bridge.destroy(&device);
        renderer.destroy(&device);
        simulation.destroy(&device);
    }
}
