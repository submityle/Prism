//! Real render-pass consumer for a raw CLST v3 cluster payload.
//!
//! The vertex shader fetches the local index, reconstructs the quantized
//! position, and decodes one authored v3 attribute directly from the payload.
//! No CPU vertex buffer or attribute repacking is involved.  The public method
//! is intentionally small: page-table/cut code chooses the payload and the
//! material path chooses the attribute semantic; this module owns only the
//! byte-level draw ABI.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    decode_cluster_payload, ClusterPayloadError, ClusterVertexAttribute,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, FragmentState, LoadOp, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor, VertexState,
};

use crate::context::GpuContext;

/// Errors returned before a raw-payload draw is submitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClusterVertexPullError {
    /// The shared CLST codec rejected the payload.
    Payload(ClusterPayloadError),
    /// The requested semantic is not present in the v3 footer.
    MissingAttribute,
    /// The framebuffer dimensions or payload length cannot be represented by
    /// the small render ABI.
    InvalidDimensions,
}

impl From<ClusterPayloadError> for ClusterVertexPullError {
    fn from(error: ClusterPayloadError) -> Self {
        Self::Payload(error)
    }
}

impl core::fmt::Display for ClusterVertexPullError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Payload(error) => write!(f, "invalid CLST payload: {error}"),
            Self::MissingAttribute => f.write_str("requested CLST attribute is missing"),
            Self::InvalidDimensions => f.write_str("CLST vertex-pull dimensions are invalid"),
        }
    }
}

/// One reusable raw-payload vertex-pulling render pipeline.
pub struct GpuClusterVertexPull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline's module remains valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: RenderPipeline,
}

impl GpuClusterVertexPull {
    /// Creates a pipeline that draws a cluster without binding vertex buffers.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_clst_vertex_pull"),
            source: ShaderSource::Wgsl(include_str!("../shaders/clst_vertex_pull.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_clst_vertex_pull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_clst_vertex_pull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("prism_clst_vertex_pull_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            module,
            layout,
            pipeline,
        }
    }

    /// Draws one CLST v3 cluster and reads back its RGBA8 render target.
    ///
    /// `attribute_semantic` identifies the v3 attribute decoded by the
    /// fragment input.  This method validates the complete payload first, then
    /// finds the exact footer data offset and passes only that offset/format to
    /// the shader.  The returned pixels are row-major.
    pub fn draw(
        &self,
        ctx: &GpuContext,
        bounds: SceneBounds,
        payload: &[u8],
        width: u32,
        height: u32,
        attribute_semantic: &str,
    ) -> Result<Vec<[u8; 4]>, ClusterVertexPullError> {
        if width == 0 || height == 0 || payload.len() < 4 || !payload.len().is_multiple_of(4) {
            return Err(ClusterVertexPullError::InvalidDimensions);
        }
        let decoded = decode_cluster_payload(bounds, payload)?;
        let attribute = decoded
            .attributes
            .iter()
            .find(|attribute| attribute.semantic == attribute_semantic)
            .ok_or(ClusterVertexPullError::MissingAttribute)?;
        let (attribute_offset, attribute_format) = attribute_data_offset(
            payload,
            decoded.positions.len(),
            decoded.indices.len() / 3,
            attribute,
        )?;
        let params = Params {
            vertex_count: decoded.positions.len() as u32,
            index_count: decoded.indices.len() as u32,
            attribute_offset: attribute_offset as u32,
            attribute_format,
            attribute_stride: attribute.format.size() as u32,
            _padding: [0; 3],
        };
        let device = ctx.device();
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_clst_vertex_pull_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let payload_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_clst_vertex_pull_payload"),
            contents: payload,
            usage: BufferUsages::STORAGE,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_clst_vertex_pull_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: payload_buffer.as_entire_binding(),
                },
            ],
        });
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("prism_clst_vertex_pull_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bytes_per_row = width * 4;
        let padded_row = bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let output_bytes = u64::from(padded_row) * u64::from(height);
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("prism_clst_vertex_pull_readback"),
            size: output_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_clst_vertex_pull_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("prism_clst_vertex_pull_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..decoded.indices.len() as u32, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue().submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ctx.wait();
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .expect("CLST vertex-pull readback should be mapped after device wait");
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in mapped
            .chunks_exact(padded_row as usize)
            .take(height as usize)
        {
            pixels.extend(
                row[..bytes_per_row as usize]
                    .chunks_exact(4)
                    .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]]),
            );
        }
        drop(mapped);
        staging.unmap();
        Ok(pixels)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    vertex_count: u32,
    index_count: u32,
    attribute_offset: u32,
    attribute_format: u32,
    attribute_stride: u32,
    _padding: [u32; 3],
}

fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn attribute_data_offset(
    bytes: &[u8],
    vertex_count: usize,
    triangle_count: usize,
    requested: &ClusterVertexAttribute,
) -> Result<(usize, u32), ClusterVertexPullError> {
    let geometry_end = 34usize
        .checked_add(vertex_count * 6)
        .and_then(|end| end.checked_add(triangle_count * 6))
        .ok_or(ClusterVertexPullError::InvalidDimensions)?;
    let mut cursor = (geometry_end + 3) & !3;
    let position_end = cursor
        .checked_add(8)
        .ok_or(ClusterVertexPullError::InvalidDimensions)?;
    if position_end > bytes.len() {
        return Err(ClusterVertexPullError::InvalidDimensions);
    }
    let count = u16::from_le_bytes([bytes[cursor + 4], bytes[cursor + 5]]) as usize;
    cursor = position_end;
    for _ in 0..count {
        let header_end = cursor
            .checked_add(12)
            .ok_or(ClusterVertexPullError::InvalidDimensions)?;
        if header_end > bytes.len() {
            return Err(ClusterVertexPullError::InvalidDimensions);
        }
        let name_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        let format_tag = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]) as u32;
        let location = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
        let data_len =
            u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
        cursor = header_end;
        let name_end = cursor
            .checked_add(name_len)
            .ok_or(ClusterVertexPullError::InvalidDimensions)?;
        if name_end > bytes.len() {
            return Err(ClusterVertexPullError::InvalidDimensions);
        }
        let name = core::str::from_utf8(&bytes[cursor..name_end]).unwrap_or("");
        cursor = (name_end + 3) & !3;
        let data_end = cursor
            .checked_add(data_len)
            .ok_or(ClusterVertexPullError::InvalidDimensions)?;
        if data_end > bytes.len() {
            return Err(ClusterVertexPullError::InvalidDimensions);
        }
        if name == requested.semantic && location == requested.shader_location {
            return Ok((cursor, format_tag));
        }
        cursor = data_end;
    }
    Err(ClusterVertexPullError::MissingAttribute)
}
