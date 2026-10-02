//! `wgpu` compute twin of the 3D point-to-triangle nearest-point geometry
//! contract
//! ([`point_triangle_closest_3d`](prism_render_architecture::particle::point_triangle_closest_3d),
//! particle design §8.2, §10, §14).
//!
//! The `CPU` golden
//! [`point_triangle_closest_3d`](prism_render_architecture::particle::point_triangle_closest_3d)
//! owns the pure proximity math between a point and a filled 3D triangle. Given
//! a query point `p` and a triangle `(a, b, c)` it returns the nearest point on
//! the closed, filled triangle, the barycentric weights of that point, and the
//! squared distance between them, all re-derived from the closed-form
//! Voronoi-region solution in Christer Ericson's *Real-Time Collision Detection*
//! §5.1.5 (`ClosestPtPointTriangle`). [`GpuPointTriangleClosest3d`] is the
//! on-device twin: one thread per query reproduces that answer, so a passing
//! real-device parity test is direct evidence the ported kernel solves the same
//! geometry and classifies the same seven regions and the same degenerate case
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced: the nearest
//! point on the triangle, its barycentric weights `(u, v, w)`, and the squared
//! distance from the query point. The reference's region analysis is mirrored
//! branch for branch — the three vertex regions (`a`, `b`, `c`), the three edge
//! regions (`ab`, `ac`, `bc`), the interior face, and the up-front degenerate
//! (collinear / zero-area) guard that falls back to the minimum over the three
//! edge segments.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `cross`, `dot`, `+ - * /` — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no `sqrt` (the twinned result reports a squared
//! distance, never a length), and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, tight enough
//! to catch a genuinely wrong port (a dropped region, a swapped coefficient, a
//! wrong clamp) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`point_triangle_closest_3d`](prism_render_architecture::particle::point_triangle_closest_3d);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::point_triangle_closest_3d::{
    ClosestPointOnTriangle, Vec3,
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` point-to-triangle nearest-point kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`point_triangle_closest_3d`](prism_render_architecture::particle::point_triangle_closest_3d)
/// region for region; see the module documentation for the algorithm.
const POINT_TRIANGLE_CLOSEST_3D_WGSL: &str = r#"
// 3D point-to-triangle nearest-point twin: one thread per query reproduces the
// nearest point on the closed, filled triangle, its barycentric weights and the
// squared distance to the query point. It mirrors the CPU golden
// `particle::point_triangle_closest_3d` region for region (Ericson §5.1.5),
// uses only the portable core-WGSL subset (min/max/clamp/abs/cross/dot and
// + - * /), needs no sqrt (the reported distance is squared) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::point_triangle_closest_3d; no third-party engine source or derived
// code.

// Squared triangle-normal length below which the triangle is treated as
// degenerate (collinear / zero-area); the normal length is twice the area, so
// this is a squared-area threshold. Matches the reference `AREA_EPS_SQ`.
const AREA_EPS_SQ: f32 = 1.0e-14;

// Magnitude below which a division denominator (an edge squared length or an
// edge parameter denominator) is treated as zero, collapsing the parameter to
// the region's start vertex instead of producing NaN / inf. Matches the
// reference `DENOM_EPS`.
const DENOM_EPS: f32 = 1.0e-20;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point; a pad lane follows.
    point: vec3<f32>,
    pad0: f32,
    // Triangle corner a; a pad lane follows.
    a: vec3<f32>,
    pad1: f32,
    // Triangle corner b; a pad lane follows.
    b: vec3<f32>,
    pad2: f32,
    // Triangle corner c; a pad lane follows.
    c: vec3<f32>,
    pad3: f32,
}

struct Result {
    // Nearest point on the triangle, with the squared distance in the w lane.
    point: vec3<f32>,
    dist_sq: f32,
    // Barycentric weights (u, v, w) of the nearest point, with a pad lane.
    bary: vec3<f32>,
    pad0: f32,
}

// A resolved nearest-point answer, carried out of the branch analysis before it
// is packed into the storage Result.
struct Hit {
    point: vec3<f32>,
    bary: vec3<f32>,
    dist_sq: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Builds a Hit from a known nearest point and its barycentric weights, filling
// in the squared distance to the query point. Mirrors the reference `finish`.
fn make_hit(p: vec3<f32>, point: vec3<f32>, u: f32, v: f32, w: f32) -> Hit {
    var h: Hit;
    h.point = point;
    h.bary = vec3<f32>(u, v, w);
    let delta = p - point;
    h.dist_sq = dot(delta, delta);
    return h;
}

// Closest point on the segment [a, b] to p, returning the point in xyz and the
// clamped parameter t in [0, 1] in w. A zero-length segment collapses to a.
// Mirrors the reference `closest_on_segment`.
fn closest_on_segment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    let ab = b - a;
    let denom = dot(ab, ab);
    if (denom <= DENOM_EPS) {
        return vec4<f32>(a, 0.0);
    }
    let t = clamp(dot(ab, p - a) / denom, 0.0, 1.0);
    let point = a + ab * t;
    return vec4<f32>(point, t);
}

// Degenerate (collinear / zero-area) fallback: the triangle has no interior, so
// the nearest point is the closest of the three edge segments. Barycentric
// weights are taken from the winning edge. Mirrors `closest_on_degenerate`.
fn closest_on_degenerate(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Hit {
    let r_ab = closest_on_segment(p, a, b);
    let r_bc = closest_on_segment(p, b, c);
    let r_ca = closest_on_segment(p, c, a);

    let q_ab = r_ab.xyz;
    let q_bc = r_bc.xyz;
    let q_ca = r_ca.xyz;
    let t_ab = r_ab.w;
    let t_bc = r_bc.w;
    let t_ca = r_ca.w;

    let e_ab = p - q_ab;
    let e_bc = p - q_bc;
    let e_ca = p - q_ca;
    let d_ab = dot(e_ab, e_ab);
    let d_bc = dot(e_bc, e_bc);
    let d_ca = dot(e_ca, e_ca);

    // The AB-first, then BC, then CA tie order matches the reference exactly.
    if (d_ab <= d_bc && d_ab <= d_ca) {
        return make_hit(p, q_ab, 1.0 - t_ab, t_ab, 0.0);
    }
    if (d_bc <= d_ca) {
        return make_hit(p, q_bc, 0.0, 1.0 - t_bc, t_bc);
    }
    // CA runs from c to a: weight on a is t_ca, weight on c is 1 - t_ca.
    return make_hit(p, q_ca, t_ca, 0.0, 1.0 - t_ca);
}

// The full seven-region Voronoi analysis, mirroring `closest_point_on_triangle`
// branch for branch.
fn solve_one(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Hit {
    let ab = b - a;
    let ac = c - a;

    // Degenerate guard: the normal length squared is (2*area)^2.
    let normal = cross(ab, ac);
    if (dot(normal, normal) <= AREA_EPS_SQ) {
        return closest_on_degenerate(p, a, b, c);
    }

    // Vertex region outside A.
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return make_hit(p, a, 1.0, 0.0, 0.0);
    }

    // Vertex region outside B.
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return make_hit(p, b, 0.0, 1.0, 0.0);
    }

    // Edge region of AB: projection of P onto AB.
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let denom = d1 - d3;
        var v: f32 = 0.0;
        if (denom > DENOM_EPS) {
            v = d1 / denom;
        }
        let point = a + ab * v;
        return make_hit(p, point, 1.0 - v, v, 0.0);
    }

    // Vertex region outside C.
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return make_hit(p, c, 0.0, 0.0, 1.0);
    }

    // Edge region of AC: projection of P onto AC.
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let denom = d2 - d6;
        var w: f32 = 0.0;
        if (denom > DENOM_EPS) {
            w = d2 / denom;
        }
        let point = a + ac * w;
        return make_hit(p, point, 1.0 - w, 0.0, w);
    }

    // Edge region of BC: projection of P onto BC.
    let va = d3 * d6 - d5 * d4;
    let bc_num = d4 - d3;
    let bc_den2 = d5 - d6;
    if (va <= 0.0 && bc_num >= 0.0 && bc_den2 >= 0.0) {
        let denom = bc_num + bc_den2;
        var w: f32 = 0.0;
        if (denom > DENOM_EPS) {
            w = bc_num / denom;
        }
        let point = b + (c - b) * w;
        return make_hit(p, point, 0.0, 1.0 - w, w);
    }

    // Interior face region.
    let sum = va + vb + vc;
    if (sum <= DENOM_EPS) {
        // Should not happen for a non-degenerate triangle, but guard the divide.
        return closest_on_degenerate(p, a, b, c);
    }
    let denom = 1.0 / sum;
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    let point = a + ab * v + ac * w;
    return make_hit(p, point, u, v, w);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let h = solve_one(q.point, q.a, q.b, q.c);

    var out: Result;
    out.point = h.point;
    out.dist_sq = h.dist_sq;
    out.bary = h.bary;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// One point-to-triangle nearest-point query: a query point plus the three
/// corners of the triangle, exactly the inputs the reference
/// `closest_point_on_triangle` consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTriangleQuery {
    /// The query point projected onto the closed, filled triangle.
    pub point: Vec3,
    /// Triangle corner `a`.
    pub a: Vec3,
    /// Triangle corner `b`.
    pub b: Vec3,
    /// Triangle corner `c`.
    pub c: Vec3,
}

impl PointTriangleQuery {
    /// Builds a query from a point and the three triangle corners.
    #[must_use]
    pub const fn new(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> PointTriangleQuery {
        PointTriangleQuery { point, a, b, c }
    }
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(point.xyz, pad)`, `(a.xyz, pad)`, `(b.xyz, pad)` and `(c.xyz, pad)` — `64`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point.
    point: [f32; 3],
    /// Padding lane after the query point.
    pad0: f32,
    /// Triangle corner `a`.
    a: [f32; 3],
    /// Padding lane after corner `a`.
    pad1: f32,
    /// Triangle corner `b`.
    b: [f32; 3],
    /// Padding lane after corner `b`.
    pad2: f32,
    /// Triangle corner `c`.
    c: [f32; 3],
    /// Padding lane after corner `c`.
    pad3: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &PointTriangleQuery) -> GpuQuery {
        GpuQuery {
            point: [query.point.x, query.point.y, query.point.z],
            pad0: 0.0,
            a: [query.a.x, query.a.y, query.a.z],
            pad1: 0.0,
            b: [query.b.x, query.b.y, query.b.z],
            pad2: 0.0,
            c: [query.c.x, query.c.y, query.c.z],
            pad3: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a `vec4` slot
/// `(point.xyz, dist_sq)` followed by a `vec4` slot `(bary.xyz, pad)` — `32`
/// bytes matching the `WGSL` `Result` struct and the reference `CLOSEST_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Nearest point on the triangle.
    point: [f32; 3],
    /// Squared distance from the query point to the nearest point.
    dist_sq: f32,
    /// Barycentric weights `(u, v, w)` of the nearest point.
    bary: [f32; 3],
    /// Padding lane after the barycentric weights.
    pad0: f32,
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

/// A compiled, reusable point-to-triangle nearest-point compute pipeline.
pub struct GpuPointTriangleClosest3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPointTriangleClosest3d {
    /// Compiles the point-to-triangle nearest-point kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPointTriangleClosest3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d"),
            source: ShaderSource::Wgsl(POINT_TRIANGLE_CLOSEST_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPointTriangleClosest3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`ClosestPointOnTriangle`]
    /// per input, in order.
    ///
    /// Each result equals the reference answer
    /// (`closest_point_on_triangle`) to within the tolerance documented on this
    /// module. An empty input returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[PointTriangleQuery],
    ) -> Vec<ClosestPointOnTriangle> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_output"),
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
            label: Some("prism_volumetric_point_triangle_closest_3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_bind_group"),
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
            label: Some("prism_volumetric_point_triangle_closest_3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_point_triangle_closest_3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_point_triangle_closest_3d_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`ClosestPointOnTriangle`].
fn decode_result(raw: &GpuResult) -> ClosestPointOnTriangle {
    ClosestPointOnTriangle {
        point: Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
        bary: [raw.bary[0], raw.bary[1], raw.bary[2]],
        distance_squared: raw.dist_sq,
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
