//! The GPU-driven scene draw: rasterize the culled survivors from GPU-authored
//! indirect arguments, with each survivor's placement and color read from the
//! scene's instance storage.
//!
//! This is the M2 slice's "frame": a compute cull (see
//! [`crate::cull::CullStage`]) writes the surviving instance ids and the
//! indirect-draw arguments, and this stage consumes both without the CPU ever
//! learning how many instances survived. The draw renders into an offscreen
//! `Rgba8Unorm` target so a readback can check exactly which instance painted
//! which pixel.

use alloc::string::String;
use alloc::vec::Vec;

use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, ColorTargetState, CommandBuffer,
    CommandEncoder, FragmentState, MultisampleState, PipelineLayoutDescriptor, PipelineLayoutId,
    PrimitiveState, PrimitiveTopology, RenderCommand, RenderDevice, RenderPipelineDescriptor,
    RenderPipelineId, RenderQueue, ShaderModuleDescriptor, ShaderModuleId, ShaderStages, TextureId,
    TextureUsages, TextureViewId, VertexState,
};

use crate::camera::{FrameUniform, FRAME_UNIFORM_BYTES};
use crate::draw::INDICES_PER_INSTANCE;
use crate::gpu::{bind_buffer, buffer_entry, push_black_pass, quad_prefix};
use crate::instances::{instance_storage_bytes, Instance, INSTANCE_STRIDE};
use crate::quad::QUAD_VERTICES;

/// A GPU-driven draw stage over an offscreen target.
///
/// The stage binds the cull pass's visible-index list and the scene's instance
/// storage, then records indirect draws from GPU-authored argument buffers: the
/// single-draw path takes the surviving count straight from the GPU's record,
/// and the multi-draw path takes it from the caller, who echoes the count the
/// GPU already confirmed.
pub struct SceneDrawStage {
    resolution: u32,
    capacity: u32,
    visible: BufferId,
    target: TextureId,
    target_view: TextureViewId,
    index_buffer: BufferId,
    instances: BufferId,
    frame_uniform: BufferId,
    bind_group_layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: RenderPipelineId,
    multi_pipeline: RenderPipelineId,
    shader: ShaderModuleId,
    bind_group: BindGroupId,
}

impl SceneDrawStage {
    /// Creates the offscreen target, index table, instance storage, pipelines
    /// and bind group that rasterize the survivors listed in `visible`.
    ///
    /// `visible` is the cull stage's output: one `u32` survivor id per element,
    /// bound here as a read-only storage buffer. `capacity` is the instance pool
    /// the visible list and the scene storage were sized for.
    ///
    /// # Panics
    ///
    /// Panics when `resolution` or `capacity` is zero, or when `capacity` is
    /// too large for 16-bit index values.
    #[must_use]
    pub fn new(
        device: &impl RenderDevice,
        resolution: u32,
        visible: BufferId,
        capacity: u32,
    ) -> Self {
        assert!(resolution > 0, "the scene draw needs a non-empty target");
        assert!(
            capacity > 0,
            "the scene draw needs a non-empty instance pool"
        );
        assert!(
            capacity * INDICES_PER_INSTANCE <= u32::from(u16::MAX),
            "the index table must fit 16-bit index values"
        );
        let extent = rhi::Extent3d::new_2d(resolution, resolution);

        let target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            extent,
            rhi::TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let target_view =
            device.create_texture_view(target, &rhi::TextureViewDescriptor::default());

        // One block of `INDICES_PER_INSTANCE` indices per instance slot, whose
        // values are that block's own global positions. A draw that starts at a
        // slot's block recovers the slot as `vertex_index / INDICES_PER_INSTANCE`,
        // which is how the multi-draw path names its survivor.
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-scene-index".into()),
            size: u64::from(capacity) * u64::from(QUAD_VERTICES) * 2,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instances = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-scene-instances".into()),
            size: u64::from(capacity) * INSTANCE_STRIDE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_uniform = device.create_buffer(&BufferDescriptor::new(
            FRAME_UNIFORM_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/scene.wgsl"
        )));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gpudriven-scene-bgl".into()),
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gpudriven-scene-pl".into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });

        let pipeline = make_pipeline(device, pipeline_layout, shader, "vs");
        let multi_pipeline = make_pipeline(device, pipeline_layout, shader, "vs_per_draw");

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gpudriven-scene-bg".into()),
            layout: Some(bind_group_layout),
            entries: vec![
                bind_buffer(0, frame_uniform),
                bind_buffer(1, visible),
                bind_buffer(2, instances),
            ],
        });

        Self {
            resolution,
            capacity,
            visible,
            target,
            target_view,
            index_buffer,
            instances,
            frame_uniform,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            multi_pipeline,
            shader,
            bind_group,
        }
    }

    /// Uploads the index table, the camera uniform and the scene's instance
    /// records. Call once per scene (or per frame when the scene animates)
    /// before recording a draw.
    pub fn upload(&self, queue: &impl RenderQueue, frame: &FrameUniform, instances: &[Instance]) {
        let mut index_table = Vec::with_capacity(self.capacity as usize * QUAD_VERTICES as usize);
        for slot in 0..self.capacity {
            for corner in 0..QUAD_VERTICES {
                index_table.push((slot * QUAD_VERTICES + corner) as u16);
            }
        }
        let index_bytes: Vec<u8> = index_table.iter().flat_map(|i| i.to_le_bytes()).collect();
        queue.write_buffer(self.index_buffer, 0, &index_bytes);
        queue.write_buffer(self.frame_uniform, 0, &frame.to_bytes());
        queue.write_buffer(self.instances, 0, &instance_storage_bytes(instances));
    }

    /// Records one indexed indirect draw from `args_buffer` into an encoder a
    /// caller owns, consuming the arguments the cull pass wrote.
    pub fn record_indexed_indirect(
        &self,
        encoder: &mut CommandEncoder,
        args_buffer: BufferId,
        offset: u64,
    ) {
        self.record(
            encoder,
            self.pipeline,
            vec![RenderCommand::DrawIndexedIndirect {
                buffer: args_buffer,
                offset,
            }],
        );
    }

    /// Records a render pass that issues one indexed indirect draw from
    /// `args_buffer`, consuming the arguments the cull pass wrote.
    #[must_use]
    pub fn encode_indexed_indirect(&self, args_buffer: BufferId, offset: u64) -> CommandBuffer {
        let mut encoder = CommandEncoder::labeled("gpudriven-scene");
        self.record_indexed_indirect(&mut encoder, args_buffer, offset);
        encoder.finish()
    }

    /// Records `count` indexed indirect draws from the packed array at
    /// `args_buffer` into an encoder a caller owns.
    ///
    /// Each record is expected to address its own index block (the cull pass
    /// sets `first_index` to the survivor's block base), which is how a per-draw
    /// record selects its survivor without `first_instance`.
    pub fn record_multi_indirect(
        &self,
        encoder: &mut CommandEncoder,
        args_buffer: BufferId,
        offset: u64,
        count: u32,
        stride: u32,
    ) {
        self.record(
            encoder,
            self.multi_pipeline,
            vec![RenderCommand::MultiDrawIndirect {
                buffer: args_buffer,
                offset,
                count,
                stride,
            }],
        );
    }

    /// Records a render pass that issues `count` indexed indirect draws from the
    /// packed array at `args_buffer`.
    #[must_use]
    pub fn encode_multi_indirect(
        &self,
        args_buffer: BufferId,
        offset: u64,
        count: u32,
        stride: u32,
    ) -> CommandBuffer {
        let mut encoder = CommandEncoder::labeled("gpudriven-scene");
        self.record_multi_indirect(&mut encoder, args_buffer, offset, count, stride);
        encoder.finish()
    }

    /// The offscreen target the survivors are rendered into.
    #[must_use]
    pub const fn target(&self) -> TextureId {
        self.target
    }

    /// The target's side length in pixels.
    #[must_use]
    pub const fn resolution(&self) -> u32 {
        self.resolution
    }

    /// The visible-index list this stage rasterizes, as bound for the shader.
    #[must_use]
    pub const fn visible(&self) -> BufferId {
        self.visible
    }

    /// Releases every resource the stage owns.
    pub fn destroy(&self, device: &impl RenderDevice) {
        device.destroy_render_pipeline(self.pipeline);
        device.destroy_render_pipeline(self.multi_pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.bind_group_layout);
        device.destroy_shader_module(self.shader);
        device.destroy_bind_group(self.bind_group);
        device.destroy_buffer(self.index_buffer);
        device.destroy_buffer(self.instances);
        device.destroy_buffer(self.frame_uniform);
        device.destroy_texture_view(self.target_view);
        device.destroy_texture(self.target);
    }

    /// Records a raster pass into the target with the given draw commands.
    fn record(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: RenderPipelineId,
        draws: Vec<RenderCommand>,
    ) {
        let mut commands = quad_prefix(pipeline, self.bind_group, self.index_buffer);
        commands.extend(draws);
        push_black_pass(encoder, "gpudriven-scene", self.target_view, commands);
    }
}

/// Creates a quad pipeline from one of the module's vertex entry points.
///
/// Corners come from the vertex index and placement from the instance storage,
/// so no vertex buffer is bound — only the shared index buffer, which is what
/// makes the indirect draws indexed.
fn make_pipeline(
    device: &impl RenderDevice,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    entry_point: &'static str,
) -> RenderPipelineId {
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(String::from("gpudriven-scene")),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: entry_point.into(),
            buffers: Vec::new(),
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            // A strip topology must declare the index width it restarts on, and
            // it must match the bound index buffer.
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
    })
}
