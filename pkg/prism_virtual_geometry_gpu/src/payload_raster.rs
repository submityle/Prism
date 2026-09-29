//! Full 64-bit vis-buffer `wgpu` compute twin (`(depth << 32) | payload`).
//!
//! [`GpuPayloadRaster`] is the payload-carrying companion to
//! [`GpuSoftwareRaster`](crate::GpuSoftwareRaster). Where the depth-only twin
//! composites just the 32-bit reversed-Z key on a portable `atomic<u32>`
//! buffer, this variant reproduces the shipping meshlet rasterizer's actual
//! visibility word: a single `atomic<u64>` per texel holding
//! `(depth_bits << 32) | payload`, resolved with one 64-bit `atomicMax`. The
//! nearest surface (largest reversed-Z depth in the high bits) wins and its
//! low bits carry that surface's winning payload for free.
//!
//! Because it needs a 64-bit atomic maximum, [`GpuPayloadRaster::new`] returns
//! [`None`] unless [`GpuContext::supports_u64_atomics`] is `true`, so callers
//! skip on a backend without the feature (Metal on Apple silicon *does* expose
//! it; the WebGPU baseline and some mobile drivers do not).
//!
//! The result is validated texel-for-texel against the CPU golden
//! [`VisBuffer`](prism_render_architecture::virtual_geometry::VisBuffer)
//! produced by
//! [`rasterize_triangle`](prism_render_architecture::virtual_geometry::rasterize_triangle):
//! the covered set is bit-exact, the winning payload is bit-exact, and the
//! depth field matches within one unit in the last place (the same
//! reassociation tolerance the depth-only twin documents).
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
use crate::raster::RasterError;

/// Uniform parameters shared by every thread. Layout matches `Params` in
/// `shaders/vis_buffer_payload_raster.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    tri_count: u32,
    cull_back: u32,
}

/// One triangle with its vis-buffer payload. Nine tightly packed `f32`s for the
/// three vertices followed by the `u32` payload, matching `Tri` in the shader
/// (`40`-byte stride, no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriPayload {
    x0: f32,
    y0: f32,
    d0: f32,
    x1: f32,
    y1: f32,
    d1: f32,
    x2: f32,
    y2: f32,
    d2: f32,
    payload: u32,
}

impl GpuTriPayload {
    /// Flattens a CPU-golden triangle plus its payload into the upload layout.
    fn from_vertices(v: [ScreenVertex; 3], payload: u32) -> GpuTriPayload {
        GpuTriPayload {
            x0: v[0].pos[0],
            y0: v[0].pos[1],
            d0: v[0].depth,
            x1: v[1].pos[0],
            y1: v[1].pos[1],
            d1: v[1].depth,
            x2: v[2].pos[0],
            y2: v[2].pos[1],
            d2: v[2].depth,
            payload,
        }
    }
}

/// A compiled, reusable 64-bit vis-buffer compute pipeline.
pub struct GpuPayloadRaster {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPayloadRaster {
    /// Compiles the 64-bit payload kernel on `ctx`.
    ///
    /// Returns [`None`] when `ctx` was not created with the 64-bit atomic
    /// features (see [`GpuContext::supports_u64_atomics`]); the kernel would
    /// fail to compile on such a device, so callers skip instead.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Option<GpuPayloadRaster> {
        if !ctx.supports_u64_atomics() {
            return None;
        }
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vis_buffer_payload_raster"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/vis_buffer_payload_raster.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vis_buffer_payload_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vis_buffer_payload_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vis_buffer_payload_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("raster"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(GpuPayloadRaster {
            module,
            layout,
            pipeline,
        })
    }

    /// Rasterizes `triangles` into a `width` x `height` packed vis-buffer.
    ///
    /// `payloads[i]` is written into the low 32 bits of every texel triangle
    /// `i` wins, exactly as [`rasterize_triangle`](
    /// prism_render_architecture::virtual_geometry::rasterize_triangle) writes
    /// its `payload`. The returned words are row-major and directly comparable
    /// to the CPU golden [`VisBuffer::pixels`](
    /// prism_render_architecture::virtual_geometry::VisBuffer::pixels): decode
    /// them with [`vis_depth`](
    /// prism_render_architecture::virtual_geometry::vis_depth) and [`vis_payload`](
    /// prism_render_architecture::virtual_geometry::vis_payload).
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::EmptyFramebuffer`] when `width` or `height` is
    /// zero, and [`RasterError::PayloadCountMismatch`] when `payloads` does not
    /// have exactly one entry per triangle.
    pub fn rasterize(
        &self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
        triangles: &[[ScreenVertex; 3]],
        payloads: &[u32],
        cull_back: bool,
    ) -> Result<Vec<u64>, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::EmptyFramebuffer { width, height });
        }
        if payloads.len() != triangles.len() {
            return Err(RasterError::PayloadCountMismatch {
                triangles: triangles.len(),
                payloads: payloads.len(),
            });
        }
        let device = ctx.device();

        let tri_count = triangles.len() as u32;
        let params = Params {
            width,
            height,
            tri_count,
            cull_back: u32::from(cull_back),
        };

        let gpu_tris: Vec<GpuTriPayload> = triangles
            .iter()
            .zip(payloads.iter())
            .map(|(&tri, &payload)| GpuTriPayload::from_vertices(tri, payload))
            .collect();
        // A storage buffer must never be zero-sized, so fall back to a single
        // padded triangle the shader ignores (`tri_count` stays 0).
        let padded_tri = [GpuTriPayload::zeroed()];
        let tri_upload: &[GpuTriPayload] = if gpu_tris.is_empty() {
            &padded_tri
        } else {
            &gpu_tris
        };

        let pixel_count = (width as u64) * (height as u64);
        let vis_bytes = pixel_count * 8;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_payload_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let tri_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_payload_tris"),
            contents: bytemuck::cast_slice(tri_upload),
            usage: BufferUsages::STORAGE,
        });
        // `wgpu` zero-initialises buffers, so the vis-buffer starts cleared to
        // `0` (farthest reversed-Z, zero payload), matching `VisBuffer::new`.
        let vis_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_buffer_payload_vis"),
            size: vis_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vis_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_buffer_payload_vis_stage"),
            size: vis_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vis_buffer_payload_bind_group"),
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
                    resource: vis_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vis_buffer_payload_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vis_buffer_payload_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = width.div_ceil(8);
            let gy = height.div_ceil(8);
            let gz = tri_count.max(1);
            pass.dispatch_workgroups(gx, gy, gz);
        }
        encoder.copy_buffer_to_buffer(&vis_buf, 0, &vis_stage, 0, vis_bytes);
        ctx.queue().submit([encoder.finish()]);

        vis_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = vis_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let words = bytemuck::cast_slice::<u8, u64>(&view).to_vec();
        drop(view);
        vis_stage.unmap();
        Ok(words)
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
