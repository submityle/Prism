//! GPU particle-pool vertical slice.
//!
//! The stage owns a fixed particle pool and runs the complete frame on the
//! device: a reset pass clears the GPU counters, the simulation pass performs
//! spawn/update/kill and appends live pool indices, a finalize pass copies the
//! GPU active count into indexed-indirect arguments, and a raster pass consumes
//! those arguments without a CPU-side particle count.

use alloc::string::String;
use alloc::vec::Vec;

use prism_render_driver as rhi;
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ImportedBuffer, ImportedTexture, PassFlags,
    RecordedGraph, RenderGraph, SizeClass, TextureDesc,
};
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferState, BufferUsages, ColorTargetState,
    ComputePipelineDescriptor, ComputePipelineId, FragmentState, IndexFormat, MultisampleState,
    PipelineLayoutDescriptor, PipelineLayoutId, PipelineStages, PrimitiveState, PrimitiveTopology,
    RenderDevice, RenderPipelineDescriptor, RenderPipelineId, RenderQueue, ShaderModuleDescriptor,
    ShaderModuleId, ShaderStages, TextureFormat, TextureId, TextureState, TextureUsages,
    TextureViewId, VertexState,
};

use crate::encoding::IndirectArgs;
use crate::gpu::{bind_buffer, buffer_entry};

/// Number of bytes in one GPU particle record.
pub const PARTICLE_STRIDE: u64 = 32;
/// Number of bytes in the simulation uniform.
pub const PARTICLE_PARAMS_BYTES: u64 = 64;
/// Number of bytes in the four-word atomic control block.
pub const PARTICLE_CONTROL_BYTES: u64 = 16;
/// One indexed quad uses four indices.
pub const PARTICLE_INDEX_COUNT: u32 = 4;
const WORKGROUP_SIZE: u32 = 64;

/// One pool element. `position[3]` stores remaining life and the velocity
/// vector is stored in the second four-word block.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// World-space position.
    pub position: [f32; 3],
    /// Remaining lifetime in seconds; zero means the pool slot is dead.
    pub life: f32,
    /// World-space velocity.
    pub velocity: [f32; 3],
    /// Reserved padding, kept zero by the reference and upload encoder.
    pub padding: f32,
}

impl Particle {
    /// Creates a live particle.
    #[must_use]
    pub const fn live(position: [f32; 3], velocity: [f32; 3], life: f32) -> Self {
        Self {
            position,
            life,
            velocity,
            padding: 0.0,
        }
    }

    /// Creates a dead pool slot.
    #[must_use]
    pub const fn dead() -> Self {
        Self {
            position: [0.0; 3],
            life: 0.0,
            velocity: [0.0; 3],
            padding: 0.0,
        }
    }
}

/// CPU-side simulation parameters mirrored by `particle.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleParams {
    /// Simulation timestep in seconds.
    pub dt: f32,
    /// Number of spawn requests presented to this frame.
    pub spawn_count: u32,
    /// Lifetime assigned to successful spawns.
    pub spawn_life: f32,
    /// Acceleration applied to live particles.
    pub gravity: [f32; 3],
    /// Position assigned to successful spawns.
    pub spawn_position: [f32; 3],
    /// Velocity assigned to successful spawns.
    pub spawn_velocity: [f32; 3],
}

impl Default for ParticleParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            spawn_count: 0,
            spawn_life: 1.0,
            gravity: [0.0, -9.8, 0.0],
            spawn_position: [0.0; 3],
            spawn_velocity: [0.0, 2.0, 0.0],
        }
    }
}

impl ParticleParams {
    fn to_bytes(self, capacity: u32) -> [u8; PARTICLE_PARAMS_BYTES as usize] {
        let words = [
            capacity,
            self.spawn_count,
            self.dt.to_bits(),
            self.spawn_life.to_bits(),
            self.gravity[0].to_bits(),
            self.gravity[1].to_bits(),
            self.gravity[2].to_bits(),
            0,
            self.spawn_position[0].to_bits(),
            self.spawn_position[1].to_bits(),
            self.spawn_position[2].to_bits(),
            0,
            self.spawn_velocity[0].to_bits(),
            self.spawn_velocity[1].to_bits(),
            self.spawn_velocity[2].to_bits(),
            0,
        ];
        let mut bytes = [0u8; PARTICLE_PARAMS_BYTES as usize];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// CPU reference result for one simulation step.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleReferenceResult {
    /// Pool state after update, kill and deterministic lowest-slot spawning.
    pub particles: Vec<Particle>,
    /// Live pool indices in ascending order.
    pub active_indices: Vec<u32>,
}

/// Runs the reference simulation for one pool step.
#[must_use]
pub fn simulate_cpu(pool: &[Particle], params: ParticleParams) -> ParticleReferenceResult {
    let mut particles = pool.to_vec();
    let mut active_indices = Vec::with_capacity(pool.len());
    let mut spawn_left = params.spawn_count;
    for (index, particle) in particles.iter_mut().enumerate() {
        if particle.life > 0.0 {
            particle.life = (particle.life - params.dt).max(0.0);
            if particle.life > 0.0 {
                for axis in 0..3 {
                    particle.velocity[axis] += params.gravity[axis] * params.dt;
                    particle.position[axis] += particle.velocity[axis] * params.dt;
                }
            }
        }
        if particle.life <= 0.0 && spawn_left != 0 {
            *particle = Particle::live(
                params.spawn_position,
                params.spawn_velocity,
                params.spawn_life,
            );
            spawn_left -= 1;
        }
        if particle.life > 0.0 {
            active_indices.push(index as u32);
        }
    }
    ParticleReferenceResult {
        particles,
        active_indices,
    }
}

/// Packs particles into the exact storage layout consumed by the shader.
#[must_use]
pub fn particle_storage_bytes(particles: &[Particle]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(particles.len() * PARTICLE_STRIDE as usize);
    for particle in particles {
        for word in particle.position {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&particle.life.to_le_bytes());
        for word in particle.velocity {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&particle.padding.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{particle_storage_bytes, simulate_cpu, Particle, ParticleParams, PARTICLE_STRIDE};

    #[test]
    fn particle_storage_layout_contains_life_at_the_gpu_abi_offset() {
        let particle = Particle::live([1.25, -2.0, 3.5], [4.0, 5.25, -6.0], 7.75);
        let bytes = particle_storage_bytes(&[particle]);
        assert_eq!(bytes.len(), PARTICLE_STRIDE as usize);
        assert_eq!(
            f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            particle.life
        );
        assert_eq!(
            f32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            particle.velocity[0]
        );
        assert_eq!(f32::from_le_bytes(bytes[28..32].try_into().unwrap()), 0.0);
    }

    #[test]
    fn cpu_reference_updates_kills_and_spawns_into_lowest_free_slots() {
        let pool = [
            Particle::live([1.0, 0.0, 0.0], [0.0; 3], 0.1),
            Particle::dead(),
            Particle::live([2.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0),
            Particle::dead(),
        ];
        let params = ParticleParams {
            dt: 0.25,
            spawn_count: 8,
            spawn_life: 1.0,
            gravity: [0.0, -2.0, 0.0],
            spawn_position: [-1.0, 3.0, 0.0],
            spawn_velocity: [2.0, 0.0, 0.0],
        };
        let result = simulate_cpu(&pool, params);
        assert_eq!(result.active_indices, [0, 1, 2, 3]);
        assert_eq!(
            result.particles[0],
            Particle::live([-1.0, 3.0, 0.0], [2.0, 0.0, 0.0], 1.0)
        );
        assert_eq!(
            result.particles[1],
            Particle::live([-1.0, 3.0, 0.0], [2.0, 0.0, 0.0], 1.0)
        );
        assert_eq!(result.particles[2].velocity, [0.0, 0.5, 0.0]);
        assert_eq!(result.particles[2].position, [2.0, 0.125, 0.0]);
        assert_eq!(
            result.particles[3],
            Particle::live([-1.0, 3.0, 0.0], [2.0, 0.0, 0.0], 1.0)
        );
    }
}

/// CPU-side particle renderer that records a GPU-authored indexed indirect
/// draw. The `render_graph` method is the normal frame path.
pub struct ParticleStage {
    capacity: u32,
    resolution: u32,
    pool: BufferId,
    active_indices: BufferId,
    control: BufferId,
    params: BufferId,
    draw_args: BufferId,
    frame: BufferId,
    index_buffer: BufferId,
    target: TextureId,
    target_view: TextureViewId,
    compute_bind_group_layout: BindGroupLayoutId,
    compute_pipeline_layout: PipelineLayoutId,
    render_bind_group_layout: BindGroupLayoutId,
    render_pipeline_layout: PipelineLayoutId,
    compute_pipeline: ComputePipelineId,
    spawn_pipeline: ComputePipelineId,
    reset_pipeline: ComputePipelineId,
    finalize_pipeline: ComputePipelineId,
    render_pipeline: RenderPipelineId,
    shader: ShaderModuleId,
    render_shader: ShaderModuleId,
    compute_bind_group: BindGroupId,
    render_bind_group: BindGroupId,
}

impl ParticleStage {
    /// Creates a fixed-capacity GPU particle pool and offscreen target.
    #[must_use]
    pub fn new(device: &impl RenderDevice, capacity: u32, resolution: u32) -> Self {
        assert!(capacity > 0, "a particle pool needs at least one slot");
        assert!(
            resolution > 0,
            "the particle target needs non-zero resolution"
        );
        let pool = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("particle-pool")),
            size: u64::from(capacity) * PARTICLE_STRIDE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_indices = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("particle-active-indices")),
            size: u64::from(capacity) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let control = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("particle-control")),
            size: PARTICLE_CONTROL_BYTES,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&BufferDescriptor::new(
            PARTICLE_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let draw_args = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("particle-draw-args")),
            size: IndirectArgs::BYTES,
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let frame = device.create_buffer(&BufferDescriptor::new(
            48,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("particle-quad-index")),
            size: 8,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(resolution, resolution),
            TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let target_view =
            device.create_texture_view(target, &rhi::TextureViewDescriptor::default());

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/particle.wgsl"
        )));
        let render_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            include_str!("shaders/particle_draw.wgsl"),
        ));
        let compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(String::from("particle-bgl")),
                entries: vec![
                    buffer_entry(0, BufferBindingType::Uniform, ShaderStages::COMPUTE),
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
                    buffer_entry(
                        3,
                        BufferBindingType::Storage { read_only: false },
                        ShaderStages::COMPUTE,
                    ),
                    buffer_entry(
                        4,
                        BufferBindingType::Storage { read_only: false },
                        ShaderStages::COMPUTE,
                    ),
                ],
            });
        let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-pl")),
            bind_group_layouts: vec![compute_bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from("particle-simulate")),
            layout: Some(compute_pipeline_layout),
            module: shader,
            entry_point: "simulate".into(),
        });
        let spawn_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from("particle-spawn")),
            layout: Some(compute_pipeline_layout),
            module: shader,
            entry_point: "spawn".into(),
        });
        let reset_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from("particle-reset")),
            layout: Some(compute_pipeline_layout),
            module: shader,
            entry_point: "reset".into(),
        });
        let finalize_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(String::from("particle-finalize")),
            layout: Some(compute_pipeline_layout),
            module: shader,
            entry_point: "finalize".into(),
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(String::from("particle-render-bgl")),
                entries: vec![
                    buffer_entry(0, BufferBindingType::Uniform, ShaderStages::VERTEX),
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
                ],
            });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-render-pl")),
            bind_group_layouts: vec![render_bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(String::from("particle-draw")),
            layout: Some(render_pipeline_layout),
            vertex: VertexState {
                module: render_shader,
                entry_point: "vs".into(),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(IndexFormat::Uint16),
                cull_mode: None,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: render_shader,
                entry_point: "fs".into(),
                targets: vec![Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm))],
            }),
        });
        let compute_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-bg")),
            layout: Some(compute_bind_group_layout),
            entries: vec![
                bind_buffer(0, params),
                bind_buffer(1, pool),
                bind_buffer(2, active_indices),
                bind_buffer(3, control),
                bind_buffer(4, draw_args),
            ],
        });

        let render_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-render-bg")),
            layout: Some(render_bind_group_layout),
            entries: vec![
                bind_buffer(0, frame),
                bind_buffer(1, pool),
                bind_buffer(2, active_indices),
            ],
        });

        Self {
            capacity,
            resolution,
            pool,
            active_indices,
            control,
            params,
            draw_args,
            frame,
            index_buffer,
            target,
            target_view,
            compute_bind_group_layout,
            compute_pipeline_layout,
            render_bind_group_layout,
            render_pipeline_layout,
            compute_pipeline,
            spawn_pipeline,
            reset_pipeline,
            finalize_pipeline,
            render_pipeline,
            shader,
            render_shader,
            compute_bind_group,
            render_bind_group,
        }
    }

    /// Uploads the initial pool, simulation parameters, camera rectangle and
    /// the four indices of the shared particle quad.
    pub fn upload(
        &self,
        queue: &impl RenderQueue,
        pool: &[Particle],
        params: ParticleParams,
        world_min: [f32; 2],
        world_size: [f32; 2],
        size: f32,
    ) {
        assert!(
            pool.len() <= self.capacity as usize,
            "particle upload exceeds capacity"
        );
        let mut frame_bytes = Vec::with_capacity(48);
        for value in [
            self.resolution as f32,
            self.resolution as f32,
            world_min[0],
            world_min[1],
            world_size[0],
            world_size[1],
            size,
            0.0,
            0.15,
            0.55,
            1.0,
            1.0,
        ] {
            frame_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut full_pool = vec![Particle::dead(); self.capacity as usize];
        full_pool[..pool.len()].copy_from_slice(pool);
        queue.write_buffer(self.pool, 0, &particle_storage_bytes(&full_pool));
        queue.write_buffer(self.params, 0, &params.to_bytes(self.capacity));
        queue.write_buffer(self.frame, 0, &frame_bytes);
        queue.write_buffer(self.index_buffer, 0, &[0, 0, 1, 0, 2, 0, 3, 0]);
    }

    /// Builds the reset, simulation, deterministic spawn, finalize and
    /// GPU-indirect draw graph.
    #[must_use]
    pub fn graph_frame(&self) -> RenderGraph {
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(self.resolution, self.resolution));
        let pool = import_buffer(
            &mut graph,
            "particle-pool",
            self.pool,
            u64::from(self.capacity) * PARTICLE_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let active_indices = import_buffer(
            &mut graph,
            "particle-active-indices",
            self.active_indices,
            u64::from(self.capacity) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let control = import_buffer(
            &mut graph,
            "particle-control",
            self.control,
            PARTICLE_CONTROL_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let params = import_buffer(
            &mut graph,
            "particle-params",
            self.params,
            PARTICLE_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let args = import_buffer(
            &mut graph,
            "particle-draw-args",
            self.draw_args,
            IndirectArgs::BYTES,
            BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let frame = import_buffer(
            &mut graph,
            "particle-frame",
            self.frame,
            48,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let index_buffer = import_buffer(
            &mut graph,
            "particle-index",
            self.index_buffer,
            8,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let target = graph.import_texture(
            "particle-target",
            TextureDesc::color(TextureFormat::Rgba8Unorm)
                .with_size(SizeClass::absolute_2d(self.resolution, self.resolution))
                .with_usage(TextureUsages::COPY_SRC),
            ImportedTexture {
                texture: self.target,
                default_view: self.target_view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::copy_src(),
                mip_levels: 1,
                array_layers: 1,
            },
        );

        let reset_pipeline = self.reset_pipeline;
        let reset_bind_group = self.compute_bind_group;
        graph.add_compute_pass("particle-reset", PassFlags::EMPTY, move |b| {
            let control = b.write_storage_buffer(control);
            let args = b.write_storage_buffer(args);
            b.blackboard_mut().set(ParticleReset { control, args });
            move |ctx| {
                ctx.set_compute_pipeline(reset_pipeline)
                    .set_compute_bind_group(0, reset_bind_group, Vec::new())
                    .dispatch(1, 1, 1);
            }
        });

        let compute_pipeline = self.compute_pipeline;
        let groups = self.capacity.div_ceil(WORKGROUP_SIZE);
        let compute_bind_group = self.compute_bind_group;
        graph.add_compute_pass("particle-simulate", PassFlags::EMPTY, move |b| {
            b.read_uniform(params);
            let reset = *b
                .blackboard()
                .get::<ParticleReset>()
                .expect("particle reset published its resources");
            let pool = b.write_storage_buffer(pool);
            let active_indices = b.write_storage_buffer(active_indices);
            let control = b.write_storage_buffer(reset.control);
            b.blackboard_mut().set(ParticleSimulated {
                pool,
                active_indices,
                control,
            });
            move |ctx| {
                ctx.set_compute_pipeline(compute_pipeline)
                    .set_compute_bind_group(0, compute_bind_group, Vec::new())
                    .dispatch(groups, 1, 1);
            }
        });

        let spawn_pipeline = self.spawn_pipeline;
        let spawn_bind_group = self.compute_bind_group;
        let spawn_groups = self.capacity.div_ceil(WORKGROUP_SIZE);
        graph.add_compute_pass("particle-spawn", PassFlags::EMPTY, move |b| {
            b.read_uniform(params);
            let simulated = *b
                .blackboard()
                .get::<ParticleSimulated>()
                .expect("particle simulation published its resources");
            let pool = b.write_storage_buffer(simulated.pool);
            let active_indices = b.write_storage_buffer(simulated.active_indices);
            let control = b.write_storage_buffer(simulated.control);
            b.blackboard_mut().set(ParticleSimulated {
                pool,
                active_indices,
                control,
            });
            move |ctx| {
                ctx.set_compute_pipeline(spawn_pipeline)
                    .set_compute_bind_group(0, spawn_bind_group, Vec::new())
                    .dispatch(spawn_groups, 1, 1);
            }
        });

        let finalize_pipeline = self.finalize_pipeline;
        let finalize_bind_group = self.compute_bind_group;
        graph.add_compute_pass("particle-finalize", PassFlags::EMPTY, move |b| {
            b.read_uniform(params);
            let simulated = *b
                .blackboard()
                .get::<ParticleSimulated>()
                .expect("particle simulation published its resources");
            b.read_storage_buffer(simulated.control);
            let reset_args = b
                .blackboard()
                .get::<ParticleReset>()
                .expect("particle reset published its resources")
                .args;
            let args = b.write_storage_buffer(reset_args);
            b.blackboard_mut().set(ParticleFinalized { args });
            move |ctx| {
                ctx.set_compute_pipeline(finalize_pipeline)
                    .set_compute_bind_group(0, finalize_bind_group, Vec::new())
                    .dispatch(1, 1, 1);
            }
        });

        let render_pipeline = self.render_pipeline;
        let render_bind_group = self.render_bind_group;
        let index_buffer_id = self.index_buffer;
        graph.add_raster_pass("particle-draw", PassFlags::EMPTY, move |b| {
            b.read_uniform(frame);
            let simulated = *b
                .blackboard()
                .get::<ParticleSimulated>()
                .expect("particle simulation published its resources");
            b.read_storage_buffer(simulated.pool);
            b.read_storage_buffer(simulated.active_indices);
            b.read_index(index_buffer);
            let finalized = *b
                .blackboard()
                .get::<ParticleFinalized>()
                .expect("particle finalization published its arguments");
            b.read_indirect(finalized.args);
            b.color_attachment(target);
            move |ctx| {
                ctx.set_render_pipeline(render_pipeline)
                    .set_render_bind_group(0, render_bind_group, Vec::new())
                    .set_index_buffer(index_buffer_id, IndexFormat::Uint16, 0)
                    .draw_indexed_indirect(finalized.args, 0);
            }
        });
        graph
    }

    /// Declares and executes the complete particle frame.
    pub fn render_graph<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        self.graph_frame().execute(device, queue)
    }

    /// GPU particle storage buffer, exposed for verification and inspection.
    #[must_use]
    pub const fn pool(&self) -> BufferId {
        self.pool
    }

    /// GPU active-index append buffer.
    #[must_use]
    pub const fn active_indices(&self) -> BufferId {
        self.active_indices
    }

    /// GPU atomic control block; word zero is the active count.
    #[must_use]
    pub const fn control(&self) -> BufferId {
        self.control
    }

    /// GPU-authored indexed-indirect draw arguments.
    #[must_use]
    pub const fn draw_args(&self) -> BufferId {
        self.draw_args
    }

    /// Render target written by the indirect particle draw.
    #[must_use]
    pub const fn target(&self) -> TextureId {
        self.target
    }

    /// Releases every resource owned by the stage.
    pub fn destroy(&self, device: &impl RenderDevice) {
        device.destroy_render_pipeline(self.render_pipeline);
        device.destroy_compute_pipeline(self.compute_pipeline);
        device.destroy_compute_pipeline(self.spawn_pipeline);
        device.destroy_compute_pipeline(self.reset_pipeline);
        device.destroy_compute_pipeline(self.finalize_pipeline);
        device.destroy_pipeline_layout(self.compute_pipeline_layout);
        device.destroy_bind_group_layout(self.compute_bind_group_layout);
        device.destroy_pipeline_layout(self.render_pipeline_layout);
        device.destroy_bind_group_layout(self.render_bind_group_layout);
        device.destroy_shader_module(self.shader);
        device.destroy_shader_module(self.render_shader);
        device.destroy_bind_group(self.compute_bind_group);
        device.destroy_bind_group(self.render_bind_group);
        device.destroy_buffer(self.pool);
        device.destroy_buffer(self.active_indices);
        device.destroy_buffer(self.control);
        device.destroy_buffer(self.params);
        device.destroy_buffer(self.draw_args);
        device.destroy_buffer(self.frame);
        device.destroy_buffer(self.index_buffer);
        device.destroy_texture_view(self.target_view);
        device.destroy_texture(self.target);
    }
}

#[derive(Clone, Copy)]
struct ParticleReset {
    control: BufferHandle,
    args: BufferHandle,
}

#[derive(Clone, Copy)]
struct ParticleSimulated {
    pool: BufferHandle,
    active_indices: BufferHandle,
    control: BufferHandle,
}

#[derive(Clone, Copy)]
struct ParticleFinalized {
    args: BufferHandle,
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
