//! `wgpu` compute twin of the finite segment vs. triangle intersection contract
//! ([`segment_triangle_intersect`](prism_render_architecture::particle::segment_triangle_intersect),
//! particle design §14).
//!
//! The `CPU` golden
//! [`segment_triangle_intersect`](prism_render_architecture::particle::segment_triangle_intersect)
//! owns the analytic "does this *bounded* segment pierce this face, and where"
//! math: a segment-clamped `Moller-Trumbore` solve. A particle moving from
//! `start` to `end` sweeps a finite segment, and deciding whether that step
//! crosses a triangle is the `Moller-Trumbore` test with the ray parameter `t`
//! clamped to `[0, 1]`. Solving the `3x3` system once yields the two free
//! `barycentric` weights, the third weight `w`, the segment parameter `t`, and
//! the hit point in a single division.
//! [`GpuSegmentTriangleIntersect`] is the on-device twin: one thread per query
//! reproduces the reference answer, so a passing real-device parity test is
//! direct evidence the ported kernel solves the same geometry and classifies the
//! same degenerate cases the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced: the
//! [`Hit`](prism_render_architecture::particle::segment_triangle_intersect::Hit)
//! crossing point, the three `barycentric` weights (`u` for `v1`, `v` for `v2`,
//! `w = 1 - u - v` for `v0`) and the segment parameter `t`, or a miss. The
//! reference's miss branches are mirrored guard for guard: a degenerate
//! (zero-area) triangle whose doubled area is below `AREA_EPS` reports a miss; a
//! segment parallel to or lying in the triangle plane (the solve determinant
//! below `DET_EPS` in magnitude, which also catches a zero-length segment)
//! reports a miss; a crossing whose `barycentric` weights fall outside `[0, 1]`
//! beyond the `EDGE_EPS` slack reports a miss; and a crossing whose `t` falls
//! outside `[0, 1]` beyond the slack reports a miss. The hit presence is routed
//! through a `u32` flag so the kernel never writes an exact `f32` sentinel.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `dot`,
//! `cross`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! `sqrt` (the solve is purely rational), and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! guarded reciprocal, so `CPU` and `GPU` evaluate the same closed form in the
//! same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields while pinning
//! the hit-presence classification exactly, tight enough to catch a genuinely
//! wrong port (a dropped guard, a swapped coefficient, a wrong clamp) yet loose
//! enough to admit legal fused multiply-add contraction. Every fixture is kept
//! well away from the degenerate regions and the inclusion boundaries so `CPU`
//! and `GPU` stay on the same side of every branch.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`segment_triangle_intersect`](prism_render_architecture::particle::segment_triangle_intersect);
//! standard segment-clamped `Moller-Trumbore` triangle test plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::segment_triangle_intersect::{Hit, Segment, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Bit flag set when the segment pierces the triangle at some `t` in `[0, 1]`
/// (the reference `intersect` returning [`Some`]). A direct `f32` equality is
/// forbidden, so the kernel emits an integer flag rather than a sentinel float.
const FLAG_HIT: u32 = 1;

/// The portable core-`WGSL` segment-triangle kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`intersect`](prism_render_architecture::particle::segment_triangle_intersect::intersect)
/// guard for guard; see the module documentation for the algorithm.
const SEGMENT_TRIANGLE_INTERSECT_WGSL: &str = r#"
// Segment vs triangle twin: one thread per query runs a segment-clamped
// Moller-Trumbore solve, rejecting a zero-area triangle, a parallel/coplanar or
// zero-length segment, an out-of-triangle barycentric pair and an out-of-segment
// t, then writes the hit point, the three barycentric weights, the segment
// parameter t and a u32 hit flag. It mirrors the CPU golden
// `particle::segment_triangle_intersect` guard for guard, uses only the portable
// core-WGSL subset (abs/dot/cross and + - * /, no sqrt), and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::segment_triangle_intersect;
// no third-party engine source or derived code.

// Magnitude below which the solve determinant is treated as zero, so a segment
// parallel to (or inside) the triangle plane falls back to a miss.
const DET_EPS: f32 = 1.0e-8;
// Magnitude below which a triangle's doubled area marks it degenerate (a point
// or a sliver), so it reports a miss instead of an ill-conditioned solve.
const AREA_EPS: f32 = 1.0e-6;
// Slack on the barycentric and segment-parameter inclusion tests so a hit on an
// edge, a vertex or an endpoint is accepted despite round-off.
const EDGE_EPS: f32 = 1.0e-5;

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 80-byte std430 stride matching the host `GpuQuery`: the segment
// start and end plus the three triangle vertices, each padded to a vec4 so the
// storage array needs no manual vec3 alignment arithmetic.
struct Query {
    seg_start: vec3<f32>,
    pad0: f32,
    seg_end: vec3<f32>,
    pad1: f32,
    v0: vec3<f32>,
    pad2: f32,
    v1: vec3<f32>,
    pad3: f32,
    v2: vec3<f32>,
    pad4: f32,
}

// One result. 48-byte std430 stride matching the host `GpuResult`: the hit point
// in a vec4 slot, then the four scalar fields (u, v, w, t), then the hit flag
// with three pad words.
struct Result {
    point: vec3<f32>,
    pad0: f32,
    u: f32,
    v: f32,
    w: f32,
    t: f32,
    flags: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Default to a miss; every reject branch writes this and returns.
    var out: Result;
    out.point = vec3<f32>(0.0, 0.0, 0.0);
    out.pad0 = 0.0;
    out.u = 0.0;
    out.v = 0.0;
    out.w = 0.0;
    out.t = 0.0;
    out.flags = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    out.pad3 = 0u;

    let edge1 = q.v1 - q.v0;
    let edge2 = q.v2 - q.v0;

    // Reject a degenerate (zero-area) triangle before dividing.
    let face_normal = cross(edge1, edge2);
    if (dot(face_normal, face_normal) < AREA_EPS * AREA_EPS) {
        results[idx] = out;
        return;
    }

    let dir = q.seg_end - q.seg_start;
    let pvec = cross(dir, edge2);
    let det = dot(edge1, pvec);

    // A near-zero determinant means the segment is parallel to (or lies in) the
    // triangle plane; also catches a zero-length segment (a null direction).
    if (abs(det) < DET_EPS) {
        results[idx] = out;
        return;
    }

    let inv_det = 1.0 / det;
    let tvec = q.seg_start - q.v0;

    let u = dot(tvec, pvec) * inv_det;
    if (u < -EDGE_EPS || u > 1.0 + EDGE_EPS) {
        results[idx] = out;
        return;
    }

    let qvec = cross(tvec, edge1);
    let v = dot(dir, qvec) * inv_det;
    if (v < -EDGE_EPS || u + v > 1.0 + EDGE_EPS) {
        results[idx] = out;
        return;
    }

    let t = dot(edge2, qvec) * inv_det;
    if (t < -EDGE_EPS || t > 1.0 + EDGE_EPS) {
        results[idx] = out;
        return;
    }

    let point = q.seg_start + dir * t;
    let w = 1.0 - u - v;
    out.point = point;
    out.u = u;
    out.v = v;
    out.w = w;
    out.t = t;
    out.flags = 1u;
    results[idx] = out;
}
"#;

/// One segment-triangle query: a finite [`Segment`] tested against the triangle
/// `v0`, `v1`, `v2`, the same inputs the reference
/// [`intersect`](prism_render_architecture::particle::segment_triangle_intersect::intersect)
/// consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentTriangleQuery {
    /// The finite segment being swept against the face (`t = 0` at `start`,
    /// `t = 1` at `end`).
    pub segment: Segment,
    /// First triangle vertex (weighted by `w = 1 - u - v`).
    pub v0: Vec3,
    /// Second triangle vertex (weighted by `u`).
    pub v1: Vec3,
    /// Third triangle vertex (weighted by `v`).
    pub v2: Vec3,
}

impl SegmentTriangleQuery {
    /// Builds a query from a segment and the triangle's three vertices.
    #[must_use]
    pub const fn new(segment: Segment, v0: Vec3, v1: Vec3, v2: Vec3) -> SegmentTriangleQuery {
        SegmentTriangleQuery {
            segment,
            v0,
            v1,
            v2,
        }
    }
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(seg_start.xyz, pad)`, `(seg_end.xyz, pad)`, `(v0.xyz, pad)`,
/// `(v1.xyz, pad)` and `(v2.xyz, pad)` — `80` bytes, each `vec3` on its
/// `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Segment start point.
    seg_start: [f32; 3],
    /// Padding lane after the segment start.
    pad0: f32,
    /// Segment end point.
    seg_end: [f32; 3],
    /// Padding lane after the segment end.
    pad1: f32,
    /// First triangle vertex.
    v0: [f32; 3],
    /// Padding lane after the first vertex.
    pad2: f32,
    /// Second triangle vertex.
    v1: [f32; 3],
    /// Padding lane after the second vertex.
    pad3: f32,
    /// Third triangle vertex.
    v2: [f32; 3],
    /// Padding lane after the third vertex.
    pad4: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &SegmentTriangleQuery) -> GpuQuery {
        GpuQuery {
            seg_start: [
                query.segment.start.x,
                query.segment.start.y,
                query.segment.start.z,
            ],
            pad0: 0.0,
            seg_end: [
                query.segment.end.x,
                query.segment.end.y,
                query.segment.end.z,
            ],
            pad1: 0.0,
            v0: [query.v0.x, query.v0.y, query.v0.z],
            pad2: 0.0,
            v1: [query.v1.x, query.v1.y, query.v1.z],
            pad3: 0.0,
            v2: [query.v2.x, query.v2.y, query.v2.z],
            pad4: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a `vec4` slot for the hit point,
/// then the four scalars `(u, v, w, t)`, then the hit flag with three pad words
/// — `48` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// World-space hit point (valid only when the hit flag is set).
    point: [f32; 3],
    /// Padding lane after the hit point.
    pad0: f32,
    /// `barycentric` weight of the second vertex `v1`.
    u: f32,
    /// `barycentric` weight of the third vertex `v2`.
    v: f32,
    /// `barycentric` weight of the first vertex `v0`, equal to `1 - u - v`.
    w: f32,
    /// Fraction along the segment of the crossing.
    t: f32,
    /// Hit-presence flag (`FLAG_HIT` when the segment pierces the triangle).
    flags: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Padding word.
    pad3: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable segment-triangle compute pipeline.
pub struct GpuSegmentTriangleIntersect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegmentTriangleIntersect {
    /// Compiles the segment-triangle kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegmentTriangleIntersect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect"),
            source: ShaderSource::Wgsl(SEGMENT_TRIANGLE_INTERSECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegmentTriangleIntersect {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one `Option<Hit>` per input, in
    /// order, mirroring the reference
    /// [`intersect`](prism_render_architecture::particle::segment_triangle_intersect::intersect)
    /// contract.
    ///
    /// Each result equals the reference answer to within the tolerance
    /// documented on this module, with the hit-presence classification pinned
    /// exactly. An empty input returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SegmentTriangleQuery]) -> Vec<Option<Hit>> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_segment_triangle_intersect_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_segment_triangle_intersect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the reference `Option<Hit>` shape,
/// unpacking the hit flag into [`Some`] / [`None`].
fn decode_result(raw: &GpuResult) -> Option<Hit> {
    if raw.flags & FLAG_HIT == 0 {
        return None;
    }
    Some(Hit {
        point: Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
        u: raw.u,
        v: raw.v,
        w: raw.w,
        t: raw.t,
    })
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
