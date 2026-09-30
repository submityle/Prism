//! `wgpu` compute twin of Prism's closest-point-on-triangle primitive
//! ([`closest_point_on_triangle`](prism_render_architecture::hair::binding::closest_point_on_triangle)).
//!
//! The strand-root binder brute-force scans every scalp triangle for the one
//! whose closest surface point is nearest a root, so the closest-point test is
//! the inner kernel of `bind_roots` (twinned by [`crate::root_bind`]); it is
//! also a stand-alone geometry primitive the collision, `SDF` and resample
//! stages lean on. This crate is that primitive's isolated twin: one thread per
//! `(p, a, b, c)` query returns the point on triangle `abc` closest to `p` plus
//! its barycentric weights, so a passing real-device parity test is direct
//! evidence each of the seven Voronoi regions ports bit-faithfully — coverage
//! the aggregate nearest-triangle scan in `root_bind` cannot isolate, since its
//! output only reports the winning face's binding, not which region each
//! candidate resolved to.
//!
//! # Region structure
//!
//! This is the standard Voronoi-region test from Ericson's Real-Time Collision
//! Detection: three vertex regions (`A`/`B`/`C`), three edge regions
//! (`AB`/`AC`/`BC`) and the interior face region, decided in that fixed order by
//! sign tests on the six edge dot products and the barycentric numerators. The
//! branch order and every comparison match the reference exactly, so both sides
//! pick the same region for the same query.
//!
//! # Correctness model
//!
//! The vertex regions return a triangle corner and a canonical barycentric basis
//! (`[1,0,0]` etc.) with no arithmetic, so they are bit-exact. The edge and
//! interior regions divide precomputed dot-product combinations, which a `GPU`
//! may evaluate with a fused multiply-add the scalar reference cannot, so the
//! closest point and interpolated weights can differ by a few low-mantissa
//! `ULP`; the parity test keeps queries clear of region boundaries (so the
//! branch pick is unambiguous) and asserts the point within `1e-4` and each
//! weight within `abs < 1e-4` or `rel < 1e-3`, while vertex-region hits are
//! asserted exactly.
//!
//! # Portability
//!
//! The kernel uses only subtraction, `dot`, multiply, compare and one divide in
//! the portable core-`WGSL` subset — no `exp`, `pow`, atomics or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard closest-point-on-triangle (Ericson Voronoi-region test)
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::binding::closest_point_on_triangle;
use prism_render_architecture::hair::interpolation::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One closest-point query: the free point and the triangle it is tested
/// against.
#[derive(Clone, Copy, Debug)]
pub struct ClosestPointQuery {
    /// The free point whose nearest point on the triangle is sought.
    pub p: Vec3,
    /// First triangle corner.
    pub a: Vec3,
    /// Second triangle corner.
    pub b: Vec3,
    /// Third triangle corner.
    pub c: Vec3,
}

/// One resolved closest point plus its barycentric weights on the queried
/// triangle.
#[derive(Clone, Copy, Debug)]
pub struct ClosestPointResult {
    /// The point on the triangle closest to the query point.
    pub point: Vec3,
    /// Barycentric weights `[wa, wb, wc]` of `point` on the triangle.
    pub bary: [f32; 3],
}

/// Uniform query count, padded to `16` bytes to match `Params` in
/// `shaders/closest_point_triangle.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    query_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query in the shader upload layout: four `xyzw` corners (`w` unused),
/// matching `Query` in `shaders/closest_point_triangle.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    p: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
}

/// One result in the shader upload layout: closest point and barycentric
/// weights, each `xyz` in a `vec4` (`w` unused), matching `Result` in
/// `shaders/closest_point_triangle.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    point: [f32; 4],
    bary: [f32; 4],
}

/// A compiled, reusable closest-point-on-triangle pipeline.
pub struct GpuHairClosestPointTriangle {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairClosestPointTriangle {
    /// Compiles the closest-point kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairClosestPointTriangle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_closest_point_triangle"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/closest_point_triangle.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_closest_point_triangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_closest_point_triangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_closest_point_triangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairClosestPointTriangle {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the closest point (and its barycentric weights) on each query's
    /// triangle to its query point, one result per query in input order.
    ///
    /// The result for query `q` equals the `CPU` golden
    /// [`closest_point_on_triangle`](prism_render_architecture::hair::binding::closest_point_on_triangle)
    /// to within the tolerance documented on this module (bit-exact for
    /// vertex-region hits). An empty query slice yields an empty vector without
    /// a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ClosestPointQuery]) -> Vec<ClosestPointResult> {
        let query_count = queries.len();
        if query_count == 0 {
            return Vec::new();
        }

        let uploads: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                p: [q.p.x, q.p.y, q.p.z, 0.0],
                a: [q.a.x, q.a.y, q.a.z, 0.0],
                b: [q.b.x, q.b.y, q.b.z, 0.0],
                c: [q.c.x, q.c.y, q.c.z, 0.0],
            })
            .collect();

        let device = ctx.device();
        let uniforms = Params {
            query_count: query_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = (query_count as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_closest_point_triangle_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_closest_point_triangle_queries"),
            contents: bytemuck::cast_slice(&uploads),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_closest_point_triangle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_closest_point_triangle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_closest_point_triangle_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_closest_point_triangle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_closest_point_triangle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (query_count as u32).div_ceil(64);
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        raw.into_iter()
            .map(|r| ClosestPointResult {
                point: Vec3::new(r.point[0], r.point[1], r.point[2]),
                bary: [r.bary[0], r.bary[1], r.bary[2]],
            })
            .collect()
    }
}

/// Runs the golden closest-point test directly; a thin re-export so the parity
/// test can name one reference path.
#[must_use]
pub fn reference_closest_point(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    closest_point_on_triangle(p, a, b, c)
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
