//! The naive per-object submission baseline the GPU-driven path is measured
//! against.
//!
//! Where [`crate::scene_draw::SceneDrawStage`] emits one indirect draw and lets
//! the GPU resolve everything, this stage does what a conventional renderer
//! does: the CPU walks the visible list and records one vertex-buffer rebind and
//! one `DrawIndexed` per object, with each object's world-space quad already
//! baked into its own vertex block. The frame's cost therefore scales with the
//! object count on the CPU, which is exactly the cost the GPU-driven path
//! removes.
//!
//! It shares [`crate::camera::FrameUniform`], [`crate::quad`] and
//! [`crate::instances::Instance`] with the GPU-driven path, so the two render
//! identical images and only the submission differs.

use alloc::string::String;
use alloc::vec::Vec;

use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, ColorTargetState, CommandBuffer,
    CommandEncoder, FragmentState, MultisampleState, PipelineLayoutDescriptor, PipelineLayoutId,
    PrimitiveState, PrimitiveTopology, RenderCommand, RenderDevice, RenderPipelineDescriptor,
    RenderPipelineId, RenderQueue, ShaderModuleDescriptor, ShaderModuleId, ShaderStages, TextureId,
    TextureUsages, TextureViewId, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::camera::{FrameUniform, FRAME_UNIFORM_BYTES};
use crate::gpu::{bind_buffer, buffer_entry, push_black_pass, quad_prefix};
use crate::instances::Instance;
use crate::quad::{corners, QUAD_STRIP_INDICES, QUAD_VERTICES};

/// The byte stride of one baked vertex: `position.xy` then `color.rgb`.
pub const NAIVE_VERTEX_BYTES: u64 = 20;

/// A per-object draw stage: one `DrawIndexed` per submitted instance.
pub struct NaiveStage {
    resolution: u32,
    capacity: u32,
    target: TextureId,
    target_view: TextureViewId,
    vertex_buffer: BufferId,
    index_buffer: BufferId,
    frame_uniform: BufferId,
    bind_group_layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: RenderPipelineId,
    shader: ShaderModuleId,
    bind_group: BindGroupId,
}

impl NaiveStage {
    /// Creates the offscreen target, the shared vertex/index buffers and the
    /// pipeline that draws one quad per instance.
    ///
    /// # Panics
    ///
    /// Panics when `resolution` or `capacity` is zero.
    #[must_use]
    pub fn new(device: &impl RenderDevice, resolution: u32, capacity: u32) -> Self {
        assert!(resolution > 0, "the baseline draw needs a non-empty target");
        assert!(
            capacity > 0,
            "the baseline draw needs a non-empty instance pool"
        );

        let target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(resolution, resolution),
            rhi::TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let target_view =
            device.create_texture_view(target, &rhi::TextureViewDescriptor::default());

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-naive-vertices".into()),
            size: u64::from(capacity) * u64::from(QUAD_VERTICES) * NAIVE_VERTEX_BYTES,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-naive-index".into()),
            size: u64::from(QUAD_VERTICES) * 2,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_uniform = device.create_buffer(&BufferDescriptor::new(
            FRAME_UNIFORM_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/naive.wgsl"
        )));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gpudriven-naive-bgl".into()),
            entries: vec![buffer_entry(
                0,
                BufferBindingType::Uniform,
                ShaderStages::VERTEX,
            )],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gpudriven-naive-pl".into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(String::from("gpudriven-naive")),
            layout: Some(pipeline_layout),
            vertex: VertexState {
                module: shader,
                entry_point: "vs".into(),
                buffers: vec![VertexBufferLayout::packed(
                    VertexStepMode::Vertex,
                    0,
                    &[VertexFormat::Float32x2, VertexFormat::Float32x3],
                )],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(rhi::IndexFormat::Uint16),
                cull_mode: None,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: shader,
                entry_point: "fs".into(),
                targets: vec![Some(ColorTargetState::opaque(
                    rhi::TextureFormat::Rgba8Unorm,
                ))],
            }),
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gpudriven-naive-bg".into()),
            layout: Some(bind_group_layout),
            entries: vec![bind_buffer(0, frame_uniform)],
        });

        Self {
            resolution,
            capacity,
            target,
            target_view,
            vertex_buffer,
            index_buffer,
            frame_uniform,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            shader,
            bind_group,
        }
    }

    /// Uploads the camera uniform, the shared quad indices and the CPU-baked
    /// vertex block of every instance.
    ///
    /// Baking the transform into the vertex buffer is the baseline's defining
    /// cost: every visible object needs its own vertices on the CPU, and every
    /// draw then rebinds a slice of them.
    ///
    /// # Panics
    ///
    /// Panics when `instances` exceeds the stage's capacity.
    pub fn upload(&self, queue: &impl RenderQueue, frame: &FrameUniform, instances: &[Instance]) {
        assert!(
            instances.len() <= self.capacity as usize,
            "uploaded {} instances into a pool of {}",
            instances.len(),
            self.capacity
        );
        queue.write_buffer(self.frame_uniform, 0, &frame.to_bytes());
        queue.write_buffer(
            self.index_buffer,
            0,
            &QUAD_STRIP_INDICES
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
        );

        let mut vertices = Vec::with_capacity(instances.len() * QUAD_VERTICES as usize);
        for instance in instances {
            for corner in corners([instance.center[0], instance.center[1]]) {
                vertices.extend_from_slice(&corner[0].to_le_bytes());
                vertices.extend_from_slice(&corner[1].to_le_bytes());
                for channel in instance.color {
                    vertices.extend_from_slice(&channel.to_le_bytes());
                }
            }
        }
        queue.write_buffer(self.vertex_buffer, 0, &vertices);
    }

    /// Records one `DrawIndexed` per visible instance into an encoder a caller
    /// owns.
    ///
    /// Every draw rebinds the vertex buffer at that instance's block, so the
    /// command stream grows by two commands per object — the CPU cost the
    /// GPU-driven path removes.
    ///
    /// # Panics
    ///
    /// Panics when the visible list names an instance outside the stage's
    /// capacity.
    pub fn record(&self, encoder: &mut CommandEncoder, visible: &[u32]) {
        assert!(
            visible.iter().all(|&object| object < self.capacity),
            "the visible list names an instance outside the stage's capacity"
        );
        let mut commands = quad_prefix(self.pipeline, self.bind_group, self.index_buffer);
        let block_bytes = u64::from(QUAD_VERTICES) * NAIVE_VERTEX_BYTES;
        for &object in visible {
            commands.push(RenderCommand::SetVertexBuffer {
                slot: 0,
                buffer: self.vertex_buffer,
                offset: u64::from(object) * block_bytes,
            });
            commands.push(RenderCommand::DrawIndexed {
                indices: 0..QUAD_VERTICES,
                base_vertex: 0,
                instances: 0..1,
            });
        }
        push_black_pass(encoder, "gpudriven-naive", self.target_view, commands);
    }

    /// Records the frame into a command buffer of its own.
    #[must_use]
    pub fn encode(&self, visible: &[u32]) -> CommandBuffer {
        let mut encoder = CommandEncoder::labeled("gpudriven-naive");
        self.record(&mut encoder, visible);
        encoder.finish()
    }

    /// The offscreen target the objects are rendered into.
    #[must_use]
    pub const fn target(&self) -> TextureId {
        self.target
    }

    /// The target's side length in pixels.
    #[must_use]
    pub const fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Releases every resource the stage owns.
    pub fn destroy(&self, device: &impl RenderDevice) {
        device.destroy_render_pipeline(self.pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.bind_group_layout);
        device.destroy_shader_module(self.shader);
        device.destroy_bind_group(self.bind_group);
        device.destroy_buffer(self.vertex_buffer);
        device.destroy_buffer(self.index_buffer);
        device.destroy_buffer(self.frame_uniform);
        device.destroy_texture_view(self.target_view);
        device.destroy_texture(self.target);
    }
}
