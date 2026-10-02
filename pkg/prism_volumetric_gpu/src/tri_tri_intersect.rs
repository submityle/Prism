//! `wgpu` compute twin of the Möller triangle-versus-triangle boolean
//! intersection contract
//! ([`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect),
//! particle design §14).
//!
//! The `CPU` golden
//! [`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect::tri_tri_intersect)
//! answers one yes/no question per query: do two closed (filled) triangles in
//! 3D share at least one point? It is Möller's 1997 interval-overlap test — each
//! triangle's vertices are scored by their signed distance to the *other*
//! triangle's supporting plane, an early reject fires when either triangle lies
//! strictly on one side, and otherwise both triangles are projected onto the
//! planes' intersection line to form two scalar intervals whose overlap decides
//! the predicate; a coplanar pair drops to a 2D edge-crossing-plus-containment
//! fallback. [`GpuTriTriIntersect`] is the on-device twin: one thread per
//! `(triangle, triangle)` pair reproduces that boolean, so a passing real-device
//! parity test is direct evidence the ported kernel folds the same branches, the
//! same `EPS` guards and the same coplanar fallback the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced branch for
//! branch: the kernel builds the second triangle's plane normal `n2` and rejects
//! a degenerate (near zero-area) triangle whose `length_squared` is below
//! [`AREA_EPS_SQ`](prism_render_architecture::particle::tri_tri_intersect::AREA_EPS_SQ),
//! snaps each signed distance to zero inside the
//! [`EPS`](prism_render_architecture::particle::tri_tri_intersect::EPS) band,
//! early-rejects when all three of a triangle's distances share a strict sign,
//! takes the coplanar fallback when every first-triangle distance vanishes, and
//! otherwise projects both triangles onto the largest-magnitude component of
//! `cross(n1, n2)` to build the two intervals and tests them for overlap with a
//! single `EPS`-slack comparison. The coplanar fallback mirrors the reference's
//! axis-plane projection, nine edge-pair segment crossings and two containment
//! probes.
//!
//! # Degenerate and touching cases
//!
//! A triangle collapsed to a segment or a point has a vanishing plane normal and
//! is reported as non-intersecting on both sides, since its supporting plane —
//! and therefore the whole interval test — is undefined. Shared vertices, shared
//! edges and coplanar touches leave a zero gap, which the `EPS` slack keeps on
//! the overlap side; the twin applies the identical slack, so both report `true`
//! there. Winding order is irrelevant on both sides because flipping it only
//! negates a plane normal, which cancels out of every sign test.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `dot`, `cross` and `+ - * /` — with no `sqrt`, `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no rounding and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The lone-vertex branch convention
//! guarantees every interval division's denominator is bounded away from zero,
//! so no division guard beyond the branch selection is needed, matching the
//! reference `isect`.
//!
//! # Correctness model
//!
//! The output is a pure boolean, so parity is an exact `==` on every element
//! with no tolerance. The reference compares `f32` signed distances, orientation
//! determinants and projected coordinates against `EPS` rather than `==`, so a
//! `GPU` fusing a multiply-add perturbs a quantity by a few units in the last
//! place but cannot flip a decision as long as every fixture stays clearly on
//! one side of each comparison — which the parity fixtures ensure by
//! construction. The boolean is therefore reproduced exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect);
//! Möller's interval-overlap triangle-triangle test plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
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

/// The portable core-`WGSL` triangle-versus-triangle intersection kernel,
/// embedded inline so the twin ships as a single source file. The single entry
/// point `solve` mirrors the `CPU` golden
/// [`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect::tri_tri_intersect)
/// branch for branch; see the module documentation for the algorithm.
const TRI_TRI_INTERSECT_WGSL: &str = r#"
// Triangle-versus-triangle boolean intersection twin: one thread per
// (triangle, triangle) pair runs Möller's interval-overlap test (with the
// coplanar 2D fallback) and writes one u32 (1 = intersect, 0 = disjoint). It
// mirrors the CPU golden `particle::tri_tri_intersect` branch for branch, uses
// only the portable core-WGSL subset (min/max/abs/dot/cross and + - * /), needs
// no sqrt and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: twinned from this repository's particle::tri_tri_intersect; no
// third-party engine source or derived code.

// Magnitude below which a signed distance, an orientation determinant, or a
// projected coordinate difference is treated as zero. Every sign test compares
// against this epsilon instead of using == / != on an f32. Matches the
// reference EPS.
const EPS: f32 = 1.0e-6;

// Squared triangle-normal length below which a triangle is degenerate
// (collinear / zero-area) and reported as non-intersecting, because its
// supporting plane is undefined. Matches the reference AREA_EPS_SQ.
const AREA_EPS_SQ: f32 = 1.0e-12;

struct Params {
    // Number of (triangle, triangle) pairs in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First triangle corners a0/a1/a2, each padded to a vec4 lane.
    a0: vec3<f32>,
    pad0: f32,
    a1: vec3<f32>,
    pad1: f32,
    a2: vec3<f32>,
    pad2: f32,
    // Second triangle corners b0/b1/b2, each padded to a vec4 lane.
    b0: vec3<f32>,
    pad3: f32,
    b1: vec3<f32>,
    pad4: f32,
    b2: vec3<f32>,
    pad5: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// Snaps a signed distance to exactly zero when within EPS, so the subsequent
// sign products and the coplanarity test are decided against the epsilon rather
// than against raw round-off. Mirrors the reference `snap`.
fn snap(value: f32) -> f32 {
    if (abs(value) < EPS) {
        return 0.0;
    }
    return value;
}

// Returns component 0 => x, 1 => y, anything else => z, without dynamic vector
// indexing. Mirrors the reference `Vec3::component`.
fn comp(v: vec3<f32>, index: u32) -> f32 {
    if (index == 0u) {
        return v.x;
    }
    if (index == 1u) {
        return v.y;
    }
    return v.z;
}

// Index of the largest-magnitude component of `dir` (0 => x, 1 => y, 2 => z),
// chosen without a range loop. Mirrors the reference `largest_axis`.
fn largest_axis(dir: vec3<f32>) -> u32 {
    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);
    if (ax > ay) {
        if (ax > az) {
            return 0u;
        }
        return 2u;
    }
    if (ay > az) {
        return 1u;
    }
    return 2u;
}

// Interpolates the two points where the edges leaving vertex v0 cross the
// opposite plane. The lone-vertex convention guarantees d0 - d1 and d0 - d2 are
// bounded away from zero, so neither division approaches a singularity. Mirrors
// the reference `isect`; returns the unsorted pair (i0, i1).
fn isect(v0: f32, v1: f32, v2: f32, d0: f32, d1: f32, d2: f32) -> vec2<f32> {
    let i0 = v0 + (v1 - v0) * d0 / (d0 - d1);
    let i1 = v0 + (v2 - v0) * d0 / (d0 - d2);
    return vec2<f32>(i0, i1);
}

// Projects a triangle onto the plane-intersection line and returns its scalar
// interval (unsorted). Mirrors the reference `compute_interval` branch order.
fn compute_interval(p0: f32, p1: f32, p2: f32, d0: f32, d1: f32, d2: f32) -> vec2<f32> {
    let d0d1 = d0 * d1;
    let d0d2 = d0 * d2;
    if (d0d1 > 0.0) {
        return isect(p2, p0, p1, d2, d0, d1);
    }
    if (d0d2 > 0.0) {
        return isect(p1, p0, p2, d1, d0, d2);
    }
    if (d1 * d2 > 0.0 || abs(d0) > EPS) {
        return isect(p0, p1, p2, d0, d1, d2);
    }
    if (abs(d1) > EPS) {
        return isect(p1, p0, p2, d1, d0, d2);
    }
    return isect(p2, p0, p1, d2, d0, d1);
}

// 2D orientation determinant of (a, b, c): positive counter-clockwise, negative
// clockwise, near zero when collinear. Mirrors the reference `orient`.
fn orient(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Whether the collinear point p lies within the EPS-expanded bounding box of
// segment a-b. Mirrors the reference `on_seg`.
fn on_seg(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> bool {
    let minx = min(a.x, b.x) - EPS;
    let maxx = max(a.x, b.x) + EPS;
    let miny = min(a.y, b.y) - EPS;
    let maxy = max(a.y, b.y) + EPS;
    return p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
}

// Whether the closed 2D segments p1-p2 and p3-p4 intersect, including
// collinear-overlap and endpoint-touching cases decided against EPS. Mirrors the
// reference `seg_seg_2d`.
fn seg_seg_2d(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, p4: vec2<f32>) -> bool {
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);

    let straddle_a = (d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS);
    let straddle_b = (d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS);
    if (straddle_a && straddle_b) {
        return true;
    }
    if (abs(d1) <= EPS && on_seg(p3, p4, p1)) {
        return true;
    }
    if (abs(d2) <= EPS && on_seg(p3, p4, p2)) {
        return true;
    }
    if (abs(d3) <= EPS && on_seg(p1, p2, p3)) {
        return true;
    }
    if (abs(d4) <= EPS && on_seg(p1, p2, p4)) {
        return true;
    }
    return false;
}

// Whether the point p lies inside the closed 2D triangle a, b, c (boundary
// included), regardless of winding. Mirrors the reference `point_in_tri_2d`.
fn point_in_tri_2d(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let d1 = orient(a, b, p);
    let d2 = orient(b, c, p);
    let d3 = orient(c, a, p);
    let has_neg = d1 < -EPS || d2 < -EPS || d3 < -EPS;
    let has_pos = d1 > EPS || d2 > EPS || d3 > EPS;
    return !(has_neg && has_pos);
}

// Coplanar fallback: projects both triangles onto the axis plane that best
// preserves area (dropping the largest normal component) and tests the 2D
// triangles for overlap via edge crossings plus containment. Mirrors the
// reference `coplanar_tri_tri`.
fn coplanar_tri_tri(
    n: vec3<f32>,
    t1a: vec3<f32>,
    t1b: vec3<f32>,
    t1c: vec3<f32>,
    t2a: vec3<f32>,
    t2b: vec3<f32>,
    t2c: vec3<f32>,
) -> bool {
    let ax = abs(n.x);
    let ay = abs(n.y);
    let az = abs(n.z);
    var i0: u32 = 0u;
    var i1: u32 = 2u;
    if (ax > ay) {
        if (ax > az) {
            i0 = 1u;
            i1 = 2u;
        } else {
            i0 = 0u;
            i1 = 1u;
        }
    } else if (az > ay) {
        i0 = 0u;
        i1 = 1u;
    } else {
        i0 = 0u;
        i1 = 2u;
    }

    var t = array<vec2<f32>, 3>(
        vec2<f32>(comp(t1a, i0), comp(t1a, i1)),
        vec2<f32>(comp(t1b, i0), comp(t1b, i1)),
        vec2<f32>(comp(t1c, i0), comp(t1c, i1)),
    );
    var u = array<vec2<f32>, 3>(
        vec2<f32>(comp(t2a, i0), comp(t2a, i1)),
        vec2<f32>(comp(t2b, i0), comp(t2b, i1)),
        vec2<f32>(comp(t2c, i0), comp(t2c, i1)),
    );

    // Fixed edge index pairs (0,1), (1,2), (2,0), matching the reference EDGES.
    var ea = array<u32, 3>(0u, 1u, 2u);
    var eb = array<u32, 3>(1u, 2u, 0u);
    for (var e: u32 = 0u; e < 3u; e = e + 1u) {
        for (var f: u32 = 0u; f < 3u; f = f + 1u) {
            if (seg_seg_2d(t[ea[e]], t[eb[e]], u[ea[f]], u[eb[f]])) {
                return true;
            }
        }
    }

    return point_in_tri_2d(t[0], u[0], u[1], u[2]) || point_in_tri_2d(u[0], t[0], t[1], t[2]);
}

// Returns true when the two closed triangles share at least one point. Mirrors
// the reference `tri_tri_intersect` guard for guard.
fn tri_tri(
    a0: vec3<f32>,
    a1: vec3<f32>,
    a2: vec3<f32>,
    b0: vec3<f32>,
    b1: vec3<f32>,
    b2: vec3<f32>,
) -> bool {
    // Supporting plane of t2: normal n2 and offset d2 with n2 . x + d2 = 0.
    let n2 = cross(b1 - b0, b2 - b0);
    if (dot(n2, n2) < AREA_EPS_SQ) {
        return false;
    }
    let d2 = -dot(n2, b0);

    // Signed distances of t1's vertices to t2's plane.
    let dv0 = snap(dot(n2, a0) + d2);
    let dv1 = snap(dot(n2, a1) + d2);
    let dv2 = snap(dot(n2, a2) + d2);
    let dv0dv1 = dv0 * dv1;
    let dv0dv2 = dv0 * dv2;
    if (dv0dv1 > 0.0 && dv0dv2 > 0.0) {
        // All of t1 lies strictly on one side of t2's plane.
        return false;
    }

    // Supporting plane of t1.
    let n1 = cross(a1 - a0, a2 - a0);
    if (dot(n1, n1) < AREA_EPS_SQ) {
        return false;
    }
    let d1 = -dot(n1, a0);

    // Signed distances of t2's vertices to t1's plane.
    let du0 = snap(dot(n1, b0) + d1);
    let du1 = snap(dot(n1, b1) + d1);
    let du2 = snap(dot(n1, b2) + d1);
    let du0du1 = du0 * du1;
    let du0du2 = du0 * du2;
    if (du0du1 > 0.0 && du0du2 > 0.0) {
        // All of t2 lies strictly on one side of t1's plane.
        return false;
    }

    // If every t1 vertex lies in t2's plane the triangles are coplanar.
    if (abs(dv0) < EPS && abs(dv1) < EPS && abs(dv2) < EPS) {
        return coplanar_tri_tri(n2, a0, a1, a2, b0, b1, b2);
    }

    // Direction of the plane-intersection line, and the axis onto which
    // projection loses the least precision (largest absolute component).
    let dir = cross(n1, n2);
    let index = largest_axis(dir);

    let iv1 = compute_interval(
        comp(a0, index),
        comp(a1, index),
        comp(a2, index),
        dv0,
        dv1,
        dv2,
    );
    let iv2 = compute_interval(
        comp(b0, index),
        comp(b1, index),
        comp(b2, index),
        du0,
        du1,
        du2,
    );

    let lo1 = min(iv1.x, iv1.y);
    let hi1 = max(iv1.x, iv1.y);
    let lo2 = min(iv2.x, iv2.y);
    let hi2 = max(iv2.x, iv2.y);

    // Intervals overlap (including a shared endpoint) unless one ends strictly
    // before the other begins.
    return !(hi1 < lo2 - EPS || hi2 < lo1 - EPS);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var hit: u32 = 0u;
    if (tri_tri(q.a0, q.a1, q.a2, q.b0, q.b1, q.b2)) {
        hit = 1u;
    }
    results[idx] = hit;
}
"#;

/// One triangle-versus-triangle intersection query: the two triangles' corners,
/// exactly the inputs the reference
/// [`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect::tri_tri_intersect)
/// consumes. Winding order is irrelevant to the predicate on both sides. Derives
/// only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriTriQuery {
    /// The first triangle's corners `[a0, a1, a2]`, each a 3D point.
    pub tri_a: [[f32; 3]; 3],
    /// The second triangle's corners `[b0, b1, b2]`, each a 3D point.
    pub tri_b: [[f32; 3]; 3],
}

impl TriTriQuery {
    /// Builds a query from the two triangles' corner triples.
    #[must_use]
    pub const fn new(tri_a: [[f32; 3]; 3], tri_b: [[f32; 3]; 3]) -> TriTriQuery {
        TriTriQuery { tri_a, tri_b }
    }
}

/// `repr(C)` `std430` layout of one packed query: six `vec4` slots holding each
/// triangle corner `(xyz, pad)` — `96` bytes, each `vec3` on its `16`-byte
/// aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First triangle corner `0`.
    a0: [f32; 3],
    /// Padding lane after corner `a0`.
    pad0: f32,
    /// First triangle corner `1`.
    a1: [f32; 3],
    /// Padding lane after corner `a1`.
    pad1: f32,
    /// First triangle corner `2`.
    a2: [f32; 3],
    /// Padding lane after corner `a2`.
    pad2: f32,
    /// Second triangle corner `0`.
    b0: [f32; 3],
    /// Padding lane after corner `b0`.
    pad3: f32,
    /// Second triangle corner `1`.
    b1: [f32; 3],
    /// Padding lane after corner `b1`.
    pad4: f32,
    /// Second triangle corner `2`.
    b2: [f32; 3],
    /// Padding lane after corner `b2`.
    pad5: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &TriTriQuery) -> GpuQuery {
        GpuQuery {
            a0: query.tri_a[0],
            pad0: 0.0,
            a1: query.tri_a[1],
            pad1: 0.0,
            a2: query.tri_a[2],
            pad2: 0.0,
            b0: query.tri_b[0],
            pad3: 0.0,
            b1: query.tri_b[1],
            pad4: 0.0,
            b2: query.tri_b[2],
            pad5: 0.0,
        }
    }
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

/// A compiled, reusable triangle-versus-triangle intersection compute pipeline.
pub struct GpuTriTriIntersect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTriTriIntersect {
    /// Compiles the triangle-versus-triangle intersection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: twinned from this repository's `tri_tri_intersect`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTriTriIntersect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect"),
            source: ShaderSource::Wgsl(TRI_TRI_INTERSECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTriTriIntersect {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one intersection boolean per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect::tri_tri_intersect)
    /// exactly: `true` when the two closed triangles share at least one point,
    /// `false` when they are disjoint or either triangle is degenerate. An empty
    /// input returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: twinned from this repository's `tri_tri_intersect`.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TriTriQuery]) -> Vec<bool> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<u32>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_output"),
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
            label: Some("prism_volumetric_tri_tri_intersect_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_bind_group"),
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
            label: Some("prism_volumetric_tri_tri_intersect_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_tri_tri_intersect_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tri_tri_intersect_pass"),
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
        let raw = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(|&flag| flag != 0).collect()
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
