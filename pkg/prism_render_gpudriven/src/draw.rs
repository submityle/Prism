//! The GPU-driven draw stage: consume the cull pass's GPU-authored visible list
//! and indirect arguments, and rasterize one quad per survivor.

use alloc::vec::Vec;

use prism_render_driver as rhi;
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, ColorTargetState, CommandBuffer,
    FragmentState, IndexFormat, MultisampleState, PipelineLayoutDescriptor, PipelineLayoutId,
    PrimitiveState, PrimitiveTopology, RenderCommand, RenderDevice, RenderPipelineDescriptor,
    RenderPipelineId, RenderQueue, ShaderModuleDescriptor, ShaderModuleId, ShaderStages, TextureId,
    TextureViewId, VertexState,
};

use crate::gpu::{bind_buffer, black_pass, buffer_entry, quad_prefix};

/// The quad's index count: the number of indices one instance draws, and the
/// size of one instance's block in the shared index buffer.
///
/// The multi-draw path depends on it: a per-draw record's `first_index` is that
/// slot's block base, so the shader recovers the slot as
/// `vertex_index / INDICES_PER_INSTANCE`.
pub const INDICES_PER_INSTANCE: u32 = 4;

/// The side of one instance's square screen cell, in pixels. The draw paints a
/// [`QUAD_PIXELS_PER_SIDE`]-pixel quad centred in each cell.
pub const CELL_PIXELS: u32 = 8;

/// The side of the rasterized quad, in pixels: half a cell.
pub const QUAD_PIXELS_PER_SIDE: u32 = CELL_PIXELS / 2;

/// The exact pixel coverage of one drawn survivor: the test asserts the
/// rendered coverage is this times the surviving count.
pub const QUAD_PIXELS: usize = (QUAD_PIXELS_PER_SIDE * QUAD_PIXELS_PER_SIDE) as usize;

/// The uniform mirroring `DrawParams` in `shaders/draw.wgsl`.
const DRAW_PARAMS_BYTES: u64 = 32;

/// A GPU-driven draw stage over an offscreen target.
///
/// The stage binds the cull pass's visible-index list and then records indirect
/// draws from GPU-authored argument buffers: the single-draw path takes the
/// surviving count straight from the GPU's record, and the multi-draw path
/// takes it from the caller, who echoes the count the GPU already confirmed.
pub struct DrawStage {
    resolution: u32,
    cells_per_row: u32,
    capacity: u32,
    visible: BufferId,
    target: TextureId,
    target_view: TextureViewId,
    index_buffer: BufferId,
    params: BufferId,
    bind_group_layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: RenderPipelineId,
    multi_pipeline: RenderPipelineId,
    shader: ShaderModuleId,
    bind_group: BindGroupId,
}

impl DrawStage {
    /// Creates the offscreen target, index buffer, pipelines and bind group
    /// that rasterize the survivors listed in `visible`.
    ///
    /// `visible` is the cull stage's output: one `u32` survivor id per element,
    /// bound here as a read-only storage buffer. `capacity` is the instance
    /// pool the visible list was sized for; it sets the index-table size.
    ///
    /// `resolution` is the square target's side in pixels and must be a
    /// multiple of [`CELL_PIXELS`]. Survivor ids must tile into fewer cells than
    /// the grid provides (`(resolution / CELL_PIXELS)^2`), which the default
    /// scene satisfies.
    ///
    /// # Panics
    ///
    /// Panics when `resolution` is zero or not a multiple of [`CELL_PIXELS`],
    /// or when `capacity` is too large for 16-bit index values.
    #[must_use]
    pub fn new(
        device: &impl RenderDevice,
        resolution: u32,
        visible: BufferId,
        capacity: u32,
    ) -> Self {
        assert!(resolution > 0, "the draw stage needs a non-empty target");
        assert_eq!(
            resolution % CELL_PIXELS,
            0,
            "the target side must divide evenly into {CELL_PIXELS}-pixel cells"
        );
        assert!(
            capacity * INDICES_PER_INSTANCE <= u32::from(u16::MAX),
            "the index table must fit 16-bit index values"
        );
        let cells_per_row = resolution / CELL_PIXELS;
        let extent = rhi::Extent3d::new_2d(resolution, resolution);

        let target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            extent,
            rhi::TextureFormat::Rgba8Unorm,
            rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
        ));
        let target_view =
            device.create_texture_view(target, &rhi::TextureViewDescriptor::default());

        // One block of `INDICES_PER_INSTANCE` indices per instance slot. The
        // block's values are its own global positions, so a draw that starts at
        // a slot's block recovers that slot as `vertex_index /
        // INDICES_PER_INSTANCE`; the first block doubles as the single draw's
        // corner table.
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-index".into()),
            size: u64::from(capacity) * u64::from(INDICES_PER_INSTANCE) * 2,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = device.create_buffer(&BufferDescriptor::new(
            DRAW_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/draw.wgsl"
        )));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gpudriven-draw-bgl".into()),
            entries: vec![
                buffer_entry(0, BufferBindingType::Uniform, ShaderStages::VERTEX),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::VERTEX,
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gpudriven-draw-pl".into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });

        let pipeline = make_pipeline(device, pipeline_layout, shader, "vs");
        let multi_pipeline = make_pipeline(device, pipeline_layout, shader, "vs_per_draw");

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gpudriven-draw-bg".into()),
            layout: Some(bind_group_layout),
            entries: vec![bind_buffer(0, params), bind_buffer(1, visible)],
        });

        Self {
            resolution,
            cells_per_row,
            capacity,
            visible,
            target,
            target_view,
            index_buffer,
            params,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            multi_pipeline,
            shader,
            bind_group,
        }
    }

    /// Uploads the index table and the draw parameters. Call once before
    /// recording a draw.
    pub fn upload(&self, queue: &impl RenderQueue) {
        let mut indices =
            Vec::with_capacity((self.capacity as usize) * INDICES_PER_INSTANCE as usize);
        for slot in 0..self.capacity {
            for corner in 0..INDICES_PER_INSTANCE {
                let value = slot * INDICES_PER_INSTANCE + corner;
                indices.push(value as u16);
            }
        }
        let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        queue.write_buffer(self.index_buffer, 0, &index_bytes);
        queue.write_buffer(self.params, 0, &self.params_bytes());
    }

    /// Records a render pass that issues one indexed indirect draw from
    /// `args_buffer`, consuming the arguments the cull pass wrote.
    #[must_use]
    pub fn encode_indexed_indirect(&self, args_buffer: BufferId, offset: u64) -> CommandBuffer {
        self.encode(
            self.pipeline,
            vec![RenderCommand::DrawIndexedIndirect {
                buffer: args_buffer,
                offset,
            }],
        )
    }

    /// Records a render pass that issues `count` indexed indirect draws from
    /// the packed array at `args_buffer`.
    ///
    /// Each record is expected to address its own index block (the cull pass
    /// sets `first_index` to the survivor's block base), which is how a
    /// per-draw record selects its survivor without `first_instance`.
    #[must_use]
    pub fn encode_multi_indirect(
        &self,
        args_buffer: BufferId,
        offset: u64,
        count: u32,
        stride: u32,
    ) -> CommandBuffer {
        self.encode(
            self.multi_pipeline,
            vec![RenderCommand::MultiDrawIndirect {
                buffer: args_buffer,
                offset,
                count,
                stride,
            }],
        )
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

    /// The exact rendered coverage, in pixels, that `survivors` survivors
    /// produce.
    #[must_use]
    pub const fn expected_coverage(survivors: usize) -> usize {
        survivors * QUAD_PIXELS
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
        device.destroy_buffer(self.params);
        device.destroy_texture_view(self.target_view);
        device.destroy_texture(self.target);
    }

    /// Records a raster pass into the target with the given draw commands.
    fn encode(&self, pipeline: RenderPipelineId, draws: Vec<RenderCommand>) -> CommandBuffer {
        let mut commands = quad_prefix(pipeline, self.bind_group, self.index_buffer);
        commands.extend(draws);
        // Clear so each pass starts from a known black frame and the coverage
        // measured is exactly this pass's output.
        black_pass("gpudriven-draw", self.target_view, commands)
    }

    /// The draw parameter uniform, packed as `DrawParams` expects.
    fn params_bytes(&self) -> [u8; DRAW_PARAMS_BYTES as usize] {
        let mut bytes = [0u8; DRAW_PARAMS_BYTES as usize];
        let resolution = self.resolution as f32;
        bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
        bytes[4..8].copy_from_slice(&resolution.to_le_bytes());
        bytes[8..12].copy_from_slice(&(CELL_PIXELS as f32).to_le_bytes());
        bytes[12..16].copy_from_slice(&self.cells_per_row.to_le_bytes());
        bytes[16..20].copy_from_slice(&INDICES_PER_INSTANCE.to_le_bytes());
        bytes
    }
}

/// Creates a quad pipeline from one of the module's vertex entry points.
///
/// The quad's corners come from the vertex index and its position from the
/// visible list, so no vertex buffer is bound — only the shared index buffer,
/// which is what makes the indirect draws indexed.
fn make_pipeline(
    device: &impl RenderDevice,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    entry_point: &'static str,
) -> RenderPipelineId {
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("gpudriven-draw".into()),
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
            strip_index_format: Some(IndexFormat::Uint16),
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
