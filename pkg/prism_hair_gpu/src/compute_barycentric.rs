//! `wgpu` compute twin of Prism's planar-projection barycentric solve
//! ([`compute_barycentric`](prism_render_architecture::hair::follicle_bind::compute_barycentric)),
//! the first step of binding a hair root to a scalp triangle.
//!
//! Attaching a follicle to its scalp face begins by projecting the root onto the
//! plane of its triangle and solving for the barycentric weights `(u, v, w)`
//! that locate it there (Ericson's method). This kernel broadcasts one shared
//! triangle to a whole batch of query points, one thread per point, and returns
//! that point's raw (un-sanitised) barycentric weights — the array-in/array-out
//! form the follicle-binding stage consumes when attaching many roots to one
//! scalp face.
//!
//! # A distinct sibling of the other triangle twins
//!
//! The [`closest_point_triangle`](crate::closest_point_triangle) twin clamps a
//! query into the triangle's Voronoi region (vertex / edge / interior test) and
//! returns the nearest *point*; this kernel instead solves the planar-projection
//! linear system and returns the (possibly out-of-triangle) *weights* without
//! clamping — a different algorithm and a different output. The
//! [`follicle_bind`](crate::follicle_bind) root-transfer twin consumes
//! already-sanitised weights to rebuild a world position; this is the earlier
//! solve that *produces* the raw weights it would later sanitise.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairComputeBarycentric::eval`] takes a batch of query points and one
//! shared [`TriangleFrame`] (only its vertex positions are used), and returns
//! the barycentric weights of each point in input order. The point index is the
//! invocation id (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the point count early-return.
//!
//! # Degenerate inputs
//!
//! A zero-area (or non-finite determinant) triangle falls back to the triangle
//! centroid `(1/3, 1/3, 1/3)` exactly as the golden does, rather than dividing by
//! zero. Because the triangle is shared across the batch, that fallback is taken
//! for the whole batch or none of it. An empty batch yields an empty vector
//! without a dispatch — storage buffers cannot be zero-sized.
//!
//! # Correctness model
//!
//! The solve is a dot-product / determinant chain plus a single reciprocal and
//! multiply-adds a `GPU` may fuse, perturbing the low mantissa bits by a few
//! `ULP`. The parity test therefore asserts a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) per component rather than a raw bit compare.
//!
//! # Portability
//!
//! The kernel uses only `dot`, compare, `abs`, a reciprocal and multiply/add in
//! the portable core-`WGSL` subset — no `exp`, `pow`, `sqrt` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Ericson "Real-Time Collision Detection" barycentric solve plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::follicle_bind::{
    compute_barycentric, Barycentric, TriangleFrame,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

extern crate alloc;

/// Uniform parameters for one solve dispatch. Layout matches `Params` in
/// `shaders/compute_barycentric.wesl`: the one shared triangle (three vertex
/// positions as `16`-byte `vec4` slots, xyz used) followed by the point count in
/// a final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pos0: [f32; 4],
    pos1: [f32; 4],
    pos2: [f32; 4],
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-point barycentric-solve pipeline.
pub struct GpuHairComputeBarycentric {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairComputeBarycentric {
    /// Compiles the per-point barycentric-solve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (`dot`, compare,
    /// `abs`, a reciprocal and multiply/add), so no optional device feature is
    /// required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairComputeBarycentric {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_compute_barycentric"),
            source: ShaderSource::Wgsl(include_str!("../shaders/compute_barycentric.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_compute_barycentric_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_compute_barycentric_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_compute_barycentric_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairComputeBarycentric {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves the barycentric weights of every point against the one shared
    /// triangle, returning one [`Barycentric`] per point in input order.
    ///
    /// The weights for point `i` equal the `CPU` golden
    /// [`compute_barycentric`](prism_render_architecture::hair::follicle_bind::compute_barycentric)
    /// of the same inputs within an fma tolerance (the only departures are a
    /// possibly-fused multiply-add and a possibly-differently-rounded
    /// reciprocal). An empty batch yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        points: &[[f32; 3]],
        tri: TriangleFrame,
    ) -> Vec<Barycentric> {
        let point_count = points.len();
        if point_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let p = tri.positions;
        let uniforms = Params {
            pos0: [p[0][0], p[0][1], p[0][2], 0.0],
            pos1: [p[1][0], p[1][1], p[1][2], 0.0],
            pos2: [p[2][0], p[2][1], p[2][2], 0.0],
            point_count: point_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Points are uploaded one vec4 (16 bytes) each; xyz is the position.
        let gpu_points: Vec<[f32; 4]> = points.iter().map(|q| [q[0], q[1], q[2], 0.0]).collect();

        // Output is one vec4 (16 bytes) per point; xyz holds (u, v, w).
        let out_bytes = (point_count as u64) * ((size_of::<f32>() * 4) as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_compute_barycentric_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_compute_barycentric_points"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_compute_barycentric_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_compute_barycentric_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_compute_barycentric_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_compute_barycentric_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_compute_barycentric_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (point_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        raw.chunks_exact(4)
            .map(|c| Barycentric {
                u: c[0],
                v: c[1],
                w: c[2],
            })
            .collect()
    }
}

/// The `CPU` golden barycentric solve for one point, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_compute_barycentric(point: [f32; 3], tri: TriangleFrame) -> Barycentric {
    compute_barycentric(point, tri)
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
