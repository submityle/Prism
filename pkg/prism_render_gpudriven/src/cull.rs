//! The GPU compute cull stage: upload a batch of instance AABBs and a frustum,
//! dispatch one pass, and read the GPU-authored survivors back.

use alloc::vec::Vec;

use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, CommandBuffer, CommandEncoder,
    ComputeCommand, ComputePipelineDescriptor, ComputePipelineId, PipelineLayoutDescriptor,
    PipelineLayoutId, RenderDevice, RenderQueue, ShaderModuleDescriptor, ShaderModuleId,
    ShaderStages,
};

use crate::encoding::{
    frustum_uniform_bytes, instance_storage_bytes, params_uniform_bytes, IndirectArgs, AABB_STRIDE,
    FRUSTUM_UNIFORM_BYTES, PARAMS_BYTES,
};
use crate::frustum::{Aabb, Frustum};
use crate::gpu::{bind_buffer, buffer_entry};
use crate::readback::ReadBuffer;

/// The workgroup size mirrored by `shaders/cull.wgsl`.
const WORKGROUP_SIZE: u32 = 64;

/// The per-draw constants the cull pass bakes into the indirect arguments it
/// authors, and into each packed multi-draw record.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CullParams {
    /// The geometry's index count (how many indices one instance draws).
    pub index_count: u32,
    /// The first index to fetch.
    pub first_index: u32,
    /// The value added to every index before fetching.
    pub base_vertex: i32,
}

/// What the GPU produced, read back to the CPU for verification and for the
/// caller to act on.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CullOutput {
    /// The GPU-authored indirect arguments: `instance_count` is the surviving
    /// count the CPU never computed.
    pub args: IndirectArgs,
    /// The survivors, in the order the GPU's atomic counter assigned them.
    pub indices: Vec<u32>,
    /// The per-survivor packed records for the multi-draw-indirect path.
    pub multi_args: Vec<IndirectArgs>,
}

impl CullOutput {
    /// How many instances survived the cull.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.args.instance_count
    }
}

/// A GPU frustum-cull stage over a fixed-capacity instance pool.
///
/// The stage owns every buffer it needs, including the GPU-authored visible
/// list and indirect arguments; [`CullStage::visible`],
/// [`CullStage::draw_args`] and [`CullStage::multi_args`] expose the handles the
/// draw stage binds and consumes. The visible list is created with
/// the storage and copy-source usages the draw stage and readback need.
pub struct CullStage {
    capacity: u32,
    params: CullParams,
    frustum: BufferId,
    instances: BufferId,
    visible: BufferId,
    draw_args: BufferId,
    multi_args: BufferId,
    cull_params: BufferId,
    bind_group_layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: ComputePipelineId,
    reset_pipeline: ComputePipelineId,
    shader: ShaderModuleId,
    bind_group: BindGroupId,
}

impl CullStage {
    /// Creates the stage's buffers, bind group and compute pipeline for a pool
    /// of at most `capacity` instances.
    ///
    /// # Panics
    ///
    /// Panics when `capacity` is zero: a zero-sized buffer is not a valid GPU
    /// resource and the stage would have nothing to cull.
    #[must_use]
    pub fn new(device: &impl RenderDevice, capacity: u32, params: CullParams) -> Self {
        assert!(capacity > 0, "a cull stage needs a non-empty instance pool");

        let uniform = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        // The visible list is written by this stage's compute pass, read back
        // for verification, and read by the draw stage's vertex shader.
        let visible_usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let indirect = BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC;

        let frustum = device.create_buffer(&BufferDescriptor::new(FRUSTUM_UNIFORM_BYTES, uniform));
        let instances = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * AABB_STRIDE,
            storage,
        ));
        let visible = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * 4,
            visible_usage,
        ));
        let draw_args = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-draw-args".into()),
            size: IndirectArgs::BYTES,
            usage: indirect | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let multi_args = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * IndirectArgs::BYTES,
            indirect,
        ));
        let cull_params = device.create_buffer(&BufferDescriptor::new(PARAMS_BYTES, uniform));

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/cull.wgsl"
        )));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gpudriven-cull-bgl".into()),
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
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gpudriven-cull-pl".into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("gpudriven-cull".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "cull".into(),
        });
        let reset_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("gpudriven-cull-reset".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "reset".into(),
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gpudriven-cull-bg".into()),
            layout: Some(bind_group_layout),
            entries: vec![
                bind_buffer(0, frustum),
                bind_buffer(1, instances),
                bind_buffer(2, visible),
                bind_buffer(3, draw_args),
                bind_buffer(4, cull_params),
                bind_buffer(5, multi_args),
            ],
        });

        Self {
            capacity,
            params,
            frustum,
            instances,
            visible,
            draw_args,
            multi_args,
            cull_params,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            reset_pipeline,
            shader,
            bind_group,
        }
    }

    /// Uploads the frustum, the instance AABBs and the draw constants, and
    /// resets the GPU's append cursor to zero.
    ///
    /// # Panics
    ///
    /// Panics when `instances` exceeds the stage's capacity.
    pub fn set_scene(&self, queue: &impl RenderQueue, frustum: &Frustum, instances: &[Aabb]) {
        assert!(
            instances.len() <= self.capacity as usize,
            "uploaded {} instances into a pool of {}",
            instances.len(),
            self.capacity
        );
        queue.write_buffer(self.frustum, 0, &frustum_uniform_bytes(frustum));
        queue.write_buffer(self.instances, 0, &instance_storage_bytes(instances));
        queue.write_buffer(
            self.cull_params,
            0,
            &params_uniform_bytes(
                instances.len() as u32,
                self.params.index_count,
                self.params.first_index,
                self.params.base_vertex,
            ),
        );
        queue.write_buffer(
            self.draw_args,
            0,
            &IndirectArgs::new(
                self.params.index_count,
                0,
                self.params.first_index,
                self.params.base_vertex,
                0,
            )
            .to_bytes(),
        );
    }

    /// Records the cull frame into an encoder a caller owns: a one-invocation
    /// pass that resets the append cursor, then the pass that classifies every
    /// candidate instance.
    ///
    /// The reset is part of the recording so the frame is self-contained and
    /// repeatable: submitting the same command buffer twice must produce the
    /// same survivor count. Without it the atomic cursor would accumulate across
    /// submissions, the draw's instance count would grow without bound, and the
    /// cull would write past the end of the visible list.
    ///
    /// Taking an existing encoder lets a caller put the cull and the draw it
    /// feeds into a single command buffer, which is one commit rather than two.
    pub fn record(&self, encoder: &mut CommandEncoder) {
        encoder.push_compute_pass(
            Some("cull-reset".into()),
            vec![
                ComputeCommand::SetPipeline(self.reset_pipeline),
                ComputeCommand::SetBindGroup {
                    index: 0,
                    bind_group: self.bind_group,
                    dynamic_offsets: Vec::new(),
                },
                ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
            ],
        );
        encoder.push_compute_pass(
            Some("cull".into()),
            vec![
                ComputeCommand::SetPipeline(self.pipeline),
                ComputeCommand::SetBindGroup {
                    index: 0,
                    bind_group: self.bind_group,
                    dynamic_offsets: Vec::new(),
                },
                ComputeCommand::Dispatch {
                    x: self.capacity.div_ceil(WORKGROUP_SIZE),
                    y: 1,
                    z: 1,
                },
            ],
        );
    }

    /// Records the cull frame into a command buffer of its own.
    #[must_use]
    pub fn encode(&self) -> CommandBuffer {
        let mut encoder = CommandEncoder::labeled("gpudriven-cull");
        self.record(&mut encoder);
        encoder.finish()
    }

    /// Submits the cull pass and reads the GPU-authored result back.
    ///
    /// The readback blocks until the GPU has finished, which is exactly what a
    /// verification or a decision point needs; a production caller would
    /// instead hand [`Self::draw_args`] straight to the draw stage and never
    /// touch the CPU.
    #[must_use]
    pub fn run(&self, queue: &(impl RenderQueue + ReadBuffer)) -> CullOutput {
        queue.submit(&[self.encode()]);
        let args =
            IndirectArgs::from_bytes(&queue.read_buffer(self.draw_args, IndirectArgs::BYTES));
        let count = args.instance_count;
        let indices = read_u32s(&queue.read_buffer(self.visible, u64::from(count) * 4));
        let multi_args = queue
            .read_buffer(self.multi_args, u64::from(count) * IndirectArgs::BYTES)
            .chunks_exact(IndirectArgs::BYTES as usize)
            .map(IndirectArgs::from_bytes)
            .collect();
        CullOutput {
            args,
            indices,
            multi_args,
        }
    }

    /// The GPU-authored visible-index list: one `u32` survivor id per element.
    #[must_use]
    pub const fn visible(&self) -> BufferId {
        self.visible
    }

    /// The GPU-authored indirect arguments for the single-draw path.
    #[must_use]
    pub const fn draw_args(&self) -> BufferId {
        self.draw_args
    }

    /// The GPU-authored packed records for the multi-draw path.
    #[must_use]
    pub const fn multi_args(&self) -> BufferId {
        self.multi_args
    }

    /// The instance-pool capacity the buffers were sized for.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Releases every resource the stage owns.
    pub fn destroy(&self, device: &impl RenderDevice) {
        device.destroy_buffer(self.frustum);
        device.destroy_buffer(self.instances);
        device.destroy_buffer(self.visible);
        device.destroy_buffer(self.draw_args);
        device.destroy_buffer(self.multi_args);
        device.destroy_buffer(self.cull_params);
        device.destroy_compute_pipeline(self.pipeline);
        device.destroy_compute_pipeline(self.reset_pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.bind_group_layout);
        device.destroy_shader_module(self.shader);
        device.destroy_bind_group(self.bind_group);
    }
}

/// Decodes a little-endian `u32` array, ignoring a trailing partial word.
fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}
