//! Real-device `wgpu` compute software rasterizer twin.
//!
//! [`GpuSoftwareRaster`] compiles `shaders/vis_buffer_software_raster.wesl`
//! once and exposes [`GpuSoftwareRaster::rasterize`], which uploads a triangle
//! list, dispatches one thread per (pixel, triangle) so overlapping triangles
//! composite through genuine `atomicMax` contention, and reads the resulting
//! per-texel reversed-Z depth keys back. The keys match, texel-for-texel, the
//! high 32 bits of the CPU golden [`VisBuffer`](
//! prism_render_architecture::virtual_geometry::VisBuffer) produced by
//! [`rasterize_triangle`](
//! prism_render_architecture::virtual_geometry::rasterize_triangle).
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization and
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::ScreenVertex;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters shared by every thread. Layout matches `Params` in
/// `shaders/vis_buffer_software_raster.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    tri_count: u32,
    cull_back: u32,
}

/// One triangle in y-down pixel space with reversed-Z depth per vertex. Nine
/// tightly packed `f32`s matching `Tri` in the shader (`36`-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTri {
    x0: f32,
    y0: f32,
    d0: f32,
    x1: f32,
    y1: f32,
    d1: f32,
    x2: f32,
    y2: f32,
    d2: f32,
}

impl GpuTri {
    /// Flattens a CPU-golden triangle into the packed upload layout verbatim.
    fn from_vertices(v: [ScreenVertex; 3]) -> GpuTri {
        GpuTri {
            x0: v[0].pos[0],
            y0: v[0].pos[1],
            d0: v[0].depth,
            x1: v[1].pos[0],
            y1: v[1].pos[1],
            d1: v[1].depth,
            x2: v[2].pos[0],
            y2: v[2].pos[1],
            d2: v[2].depth,
        }
    }
}

/// Errors returned by [`GpuSoftwareRaster::rasterize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RasterError {
    /// The framebuffer had a zero dimension, which allocates no pixels.
    EmptyFramebuffer {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
}

impl core::fmt::Display for RasterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RasterError::EmptyFramebuffer { width, height } => {
                write!(f, "framebuffer has a zero dimension: {width}x{height}")
            }
        }
    }
}

impl core::error::Error for RasterError {}

/// A compiled, reusable `GPU` software-raster pipeline.
pub struct GpuSoftwareRaster {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftwareRaster {
    /// Compiles the rasterizer kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftwareRaster {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vis_buffer_software_raster"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/vis_buffer_software_raster.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vis_buffer_raster_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vis_buffer_raster_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vis_buffer_raster_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("raster"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftwareRaster {
            module,
            layout,
            pipeline,
        }
    }

    /// Rasterizes `triangles` into a `width` x `height` depth-key buffer.
    ///
    /// Returns the per-texel reversed-Z depth keys in row-major order, matching
    /// the high 32 bits of the CPU golden [`VisBuffer`](
    /// prism_render_architecture::virtual_geometry::VisBuffer). Every triangle
    /// is composited by nearest depth (largest key) exactly as the CPU
    /// reference does; `cull_back` skips back-facing triangles under their
    /// original winding, matching [`rasterize_triangle`](
    /// prism_render_architecture::virtual_geometry::rasterize_triangle).
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::EmptyFramebuffer`] when `width` or `height` is
    /// zero, which would allocate no pixels.
    pub fn rasterize(
        &self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
        triangles: &[[ScreenVertex; 3]],
        cull_back: bool,
    ) -> Result<Vec<u32>, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::EmptyFramebuffer { width, height });
        }
        let device = ctx.device();

        let tri_count = triangles.len() as u32;
        let params = Params {
            width,
            height,
            tri_count,
            cull_back: u32::from(cull_back),
        };

        let gpu_tris: Vec<GpuTri> = triangles
            .iter()
            .map(|&tri| GpuTri::from_vertices(tri))
            .collect();
        // A storage buffer must never be zero-sized, so fall back to a single
        // padded triangle that the shader ignores (tri_count stays 0).
        let padded_tri = [GpuTri::zeroed()];
        let tri_upload: &[GpuTri] = if gpu_tris.is_empty() {
            &padded_tri
        } else {
            &gpu_tris
        };

        let pixel_count = (width as u64) * (height as u64);
        let depth_bytes = pixel_count * 4;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_raster_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let tri_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_raster_tris"),
            contents: bytemuck::cast_slice(tri_upload),
            usage: BufferUsages::STORAGE,
        });
        // `wgpu` zero-initialises buffers, so the depth buffer starts cleared to
        // `0` (the farthest reversed-Z key), matching `VisBuffer::new`.
        let depth_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_buffer_raster_depth"),
            size: depth_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let depth_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_buffer_raster_depth_stage"),
            size: depth_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vis_buffer_raster_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: tri_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: depth_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vis_buffer_raster_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vis_buffer_raster_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = width.div_ceil(8);
            let gy = height.div_ceil(8);
            let gz = tri_count.max(1);
            pass.dispatch_workgroups(gx, gy, gz);
        }
        encoder.copy_buffer_to_buffer(&depth_buf, 0, &depth_stage, 0, depth_bytes);
        ctx.queue().submit([encoder.finish()]);

        depth_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = depth_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let keys = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        depth_stage.unmap();
        Ok(keys)
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
