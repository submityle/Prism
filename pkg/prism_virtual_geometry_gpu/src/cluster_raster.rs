//! Cluster-granularity 64-bit vis-buffer `wgpu` compute twin.
//!
//! [`GpuClusterRaster`] is the *indexed cluster* companion to
//! [`GpuPayloadRaster`](crate::GpuPayloadRaster): where the payload twin
//! receives already-flattened triangles with an explicit per-triangle payload,
//! this variant reproduces the shipping per-cluster dispatch. It uploads one
//! shared vertex array plus a list of triangle index triples and a single
//! `cluster_id`, then dispatches one thread per (pixel, triangle). Each
//! triangle fetches its three vertices through the index list, synthesizes its
//! own payload as
//! [`pack_cluster_triangle`](prism_render_architecture::virtual_geometry::pack_cluster_triangle)`(cluster_id, triangle_id)`
//! and composites through a single 64-bit `atomicMax`, exactly like the CPU
//! golden
//! [`rasterize_cluster`](prism_render_architecture::virtual_geometry::rasterize_cluster).
//!
//! The result is validated texel-for-texel against that reference: the covered
//! set is bit-exact, the winning `(cluster_id, triangle_id)` payload is
//! bit-exact, and the depth field matches within one unit in the last place
//! (the same reassociation tolerance the other vis-buffer twins document). The
//! two failure modes unique to the cluster path are exercised directly - an
//! index that points past the vertex array drops its triangle, and triangles
//! beyond
//! [`MAX_CLUSTER_TRIANGLES`](prism_render_architecture::virtual_geometry::MAX_CLUSTER_TRIANGLES)
//! are dropped so triangle ids never alias the 7-bit payload field.
//!
//! Because it needs a 64-bit atomic maximum, [`GpuClusterRaster::new`] returns
//! [`None`] unless [`GpuContext::supports_u64_atomics`] is `true`, so callers
//! skip on a backend without the feature (Metal on Apple silicon exposes it;
//! the WebGPU baseline and some mobile drivers do not).
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization and
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{ScreenVertex, MAX_CLUSTER_TRIANGLES};
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
/// `shaders/vis_buffer_cluster_raster.wesl` (padded to a 16-byte multiple).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    tri_count: u32,
    vert_count: u32,
    cluster_id: u32,
    cull_back: u32,
    pad0: u32,
    pad1: u32,
}

/// One screen-space vertex in the upload layout: pixel-space position (y-down)
/// and reversed-Z depth. Three tightly packed `f32`s matching `Vert` in the
/// shader (`12`-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVert {
    x: f32,
    y: f32,
    d: f32,
}

impl GpuVert {
    /// Flattens a CPU-golden screen vertex into the packed upload layout.
    fn from_vertex(v: ScreenVertex) -> GpuVert {
        GpuVert {
            x: v.pos[0],
            y: v.pos[1],
            d: v.depth,
        }
    }
}

/// One triangle's vertex indices into the vertex array. Three tightly packed
/// `u32`s matching `Idx` in the shader (`12`-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuIdx {
    i0: u32,
    i1: u32,
    i2: u32,
}

/// A compiled, reusable cluster-granularity 64-bit vis-buffer compute pipeline.
pub struct GpuClusterRaster {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClusterRaster {
    /// Compiles the cluster kernel on `ctx`.
    ///
    /// Returns [`None`] when `ctx` was not created with the 64-bit atomic
    /// features (see [`GpuContext::supports_u64_atomics`]); the kernel would
    /// fail to compile on such a device, so callers skip instead.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Option<GpuClusterRaster> {
        if !ctx.supports_u64_atomics() {
            return None;
        }
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vis_buffer_cluster_raster"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/vis_buffer_cluster_raster.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vis_buffer_cluster_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vis_buffer_cluster_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vis_buffer_cluster_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("raster"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(GpuClusterRaster {
            module,
            layout,
            pipeline,
        })
    }

    /// Rasterizes an indexed cluster into a `width` x `height` packed
    /// vis-buffer.
    ///
    /// `triangles[i]` is a triple of indices into `vertices`; triangle `i`
    /// writes the payload
    /// [`pack_cluster_triangle`](prism_render_architecture::virtual_geometry::pack_cluster_triangle)`(cluster_id, i)`,
    /// exactly as
    /// [`rasterize_cluster`](prism_render_architecture::virtual_geometry::rasterize_cluster)
    /// does. Only the first
    /// [`MAX_CLUSTER_TRIANGLES`](prism_render_architecture::virtual_geometry::MAX_CLUSTER_TRIANGLES)
    /// triangles are rasterized; any beyond that are dropped so triangle ids
    /// never alias the 7-bit payload field, matching the reference. A triple
    /// that indexes past `vertices` is skipped rather than sampling out of
    /// range. The returned words are row-major and directly comparable to the
    /// CPU golden
    /// [`VisBuffer::pixels`](prism_render_architecture::virtual_geometry::VisBuffer::pixels).
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::EmptyFramebuffer`] when `width` or `height` is
    /// zero.
    pub fn rasterize(
        &self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
        vertices: &[ScreenVertex],
        triangles: &[[u32; 3]],
        cluster_id: u32,
        cull_back: bool,
    ) -> Result<Vec<u64>, RasterError> {
        if width == 0 || height == 0 {
            return Err(RasterError::EmptyFramebuffer { width, height });
        }
        let device = ctx.device();

        // Clamp to the per-cluster cap exactly like the CPU reference so
        // triangle ids never alias the 7-bit payload field.
        let capped = triangles.len().min(MAX_CLUSTER_TRIANGLES);
        let tri_count = u32::try_from(capped).expect("capped triangle count fits in u32");
        let vert_count = u32::try_from(vertices.len()).expect("vertex count fits in u32");

        let params = Params {
            width,
            height,
            tri_count,
            vert_count,
            cluster_id,
            cull_back: u32::from(cull_back),
            pad0: 0,
            pad1: 0,
        };

        let gpu_verts: Vec<GpuVert> = vertices.iter().map(|&v| GpuVert::from_vertex(v)).collect();
        let gpu_idx: Vec<GpuIdx> = triangles[..capped]
            .iter()
            .map(|&[i0, i1, i2]| GpuIdx { i0, i1, i2 })
            .collect();

        // A storage buffer must never be zero-sized, so fall back to a single
        // padded element the shader ignores (the relevant count stays 0).
        let padded_vert = [GpuVert::zeroed()];
        let vert_upload: &[GpuVert] = if gpu_verts.is_empty() {
            &padded_vert
        } else {
            &gpu_verts
        };
        let padded_idx = [GpuIdx::zeroed()];
        let idx_upload: &[GpuIdx] = if gpu_idx.is_empty() {
            &padded_idx
        } else {
            &gpu_idx
        };

        let pixel_count = u64::from(width) * u64::from(height);
        let vis_bytes = pixel_count * 8;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_cluster_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let vert_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_cluster_verts"),
            contents: bytemuck::cast_slice(vert_upload),
            usage: BufferUsages::STORAGE,
        });
        let idx_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_buffer_cluster_indices"),
            contents: bytemuck::cast_slice(idx_upload),
            usage: BufferUsages::STORAGE,
        });
        // `wgpu` zero-initialises buffers, so the vis-buffer starts cleared to
        // `0` (farthest reversed-Z, zero payload), matching `VisBuffer::new`.
        let vis_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_buffer_cluster_vis"),
            size: vis_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vis_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_buffer_cluster_vis_stage"),
            size: vis_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vis_buffer_cluster_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: vert_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: idx_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: vis_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vis_buffer_cluster_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vis_buffer_cluster_pass"),
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
