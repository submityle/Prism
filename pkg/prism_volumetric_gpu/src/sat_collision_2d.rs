//! `wgpu` compute twin of the 2D convex-polygon Separating Axis Theorem
//! (`SAT`) overlap + minimum-translation-vector (`MTV`) golden
//! ([`sat_collision_2d`](prism_render_architecture::particle::sat_collision_2d),
//! particle design §8.2, §10).
//!
//! The `CPU` golden
//! [`sat_collision_2d`](prism_render_architecture::particle::sat_collision_2d)
//! owns the small, verifiable collision contract several particle stages share:
//! given two convex `CCW` (counter-clockwise) polygons it decides overlap with
//! the Separating Axis Theorem
//! ([`overlaps`](prism_render_architecture::particle::sat_collision_2d::overlaps))
//! and, when they interpenetrate, reports the minimum translation vector that
//! pushes them apart
//! ([`mtv`](prism_render_architecture::particle::sat_collision_2d::mtv),
//! packaged as the reused
//! [`Mtv`](prism_render_architecture::particle::sat_collision_2d::Mtv)).
//! [`GpuSatCollision2d`] is the on-device twin: one thread solves one pair, so a
//! passing real-device parity test is direct evidence the ported kernel folds
//! the same per-edge projections, the same interval-overlap test and the same
//! smallest-overlap axis selection the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each `(a, b)` pair of fixed-capacity convex polygons (up to four
//! vertices each, so an oriented box or a triangle fits) the kernel reproduces
//! the whole reference solve: it forms every edge's outward normal as a
//! candidate separating axis
//! ([`Vec2::perp`](prism_render_architecture::particle::sat_collision_2d::Vec2::perp)
//! then
//! [`Vec2::normalize_or_zero`](prism_render_architecture::particle::sat_collision_2d::Vec2::normalize_or_zero)),
//! skips degenerate (zero-length) edges exactly as the reference does, projects
//! both polygons to a `[min, max]` interval, measures the signed interval
//! overlap, declares the pair disjoint on the first axis whose gap exceeds
//! [`SAT_EPS`](prism_render_architecture::particle::sat_collision_2d::SAT_EPS),
//! and otherwise keeps the smallest-overlap axis, clamps a touching contact to a
//! non-negative depth, and orients the axis from `a`'s centroid toward `b`'s —
//! yielding the same overlap `bool` and the same [`Mtv`] the reference returns.
//!
//! # Correctness model
//!
//! The overlap verdict is a discrete classification built from `f32` magnitude
//! comparisons against
//! [`SAT_EPS`](prism_render_architecture::particle::sat_collision_2d::SAT_EPS),
//! so for pairs clear of the touching boundary the `CPU` and `GPU` agree exactly
//! and the parity test asserts an exact `==` on the `bool` and on the presence
//! of the [`Mtv`]. The `MTV` axis and depth thread through multiplies, adds, one
//! guarded division and a single `sqrt` per axis, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the axis and depth.
//!
//! # Degenerate inputs
//!
//! A zero-length edge would normalize to a `NaN` axis; the kernel checks the
//! edge length against
//! [`SAT_EPS`](prism_render_architecture::particle::sat_collision_2d::SAT_EPS)
//! and skips the axis, matching the reference
//! [`Vec2::normalize_or_zero`](prism_render_architecture::particle::sat_collision_2d::Vec2::normalize_or_zero)
//! guard. A polygon with fewer than two vertices has no separating face and is
//! reported as non-overlapping with no [`Mtv`], mirroring the reference. An
//! empty pair batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `dot`, `+ - * /` and one `sqrt` to normalize an axis — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Each thread
//! performs a fixed, bounded sweep over at most eight candidate axes, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sat_collision_2d`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sat_collision_2d::{Mtv, Vec2};
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

/// Maximum vertex count per polygon the fixed-capacity layout carries: four, so
/// an oriented box or a triangle fits a single lane.
const MAX_VERTS: usize = 4;

/// Discrete overlap code written by the kernel for a colliding pair: matches the
/// host `== 1` decode in [`decode_result`]. A direct `f32` equality is
/// forbidden, so the kernel emits an integer flag rather than a sentinel float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` 2D `SAT` overlap + `MTV` kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`sat_collision_2d`](prism_render_architecture::particle::sat_collision_2d)
/// axis for axis; see the module documentation for the algorithm.
const SAT_COLLISION_2D_WGSL: &str = r#"
// 2D SAT overlap + MTV twin: one thread per polygon pair projects both shapes
// onto every edge normal, declares the pair disjoint on the first separating
// axis, and otherwise keeps the smallest-overlap axis as the minimum
// translation vector (oriented from a's centroid toward b's). It mirrors the
// CPU golden `particle::sat_collision_2d` axis for axis, uses only the portable
// core-WGSL subset (abs/min/max/dot, + - * / and one sqrt per axis), needs no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. Each thread sweeps at most eight candidate axes, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::sat_collision_2d；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a projection gap, an edge length or a penetration depth
// is treated as zero. Matches the reference `SAT_EPS`; the compare rule used
// instead of an f32 `==`.
const SAT_EPS: f32 = 1.0e-6;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Polygon a vertices 0 and 1 packed as (x0, y0, x1, y1).
    a01: vec4<f32>,
    // Polygon a vertices 2 and 3 packed as (x2, y2, x3, y3).
    a23: vec4<f32>,
    // Polygon b vertices 0 and 1 packed as (x0, y0, x1, y1).
    b01: vec4<f32>,
    // Polygon b vertices 2 and 3 packed as (x2, y2, x3, y3).
    b23: vec4<f32>,
    // Valid vertex counts (2..=4) of a and b, then two pad lanes.
    count_a: u32,
    count_b: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // MTV axis, meaningful only when `hit` is 1.
    axis: vec2<f32>,
    // MTV penetration depth, meaningful only when `hit` is 1.
    depth: f32,
    // Overlap flag: 1 when the polygons overlap or touch, 0 otherwise.
    hit: u32,
}

// Running state threaded through the per-polygon edge sweeps: whether any valid
// axis was seen, whether some axis separated the pair, whether a best axis was
// recorded, and that best (smallest) overlap axis and depth.
struct Acc {
    found: u32,
    separated: u32,
    has_axis: u32,
    best_depth: f32,
    best_axis: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns vertex `i` (0..3) of a polygon packed into two vec4 lanes.
fn vget(v01: vec4<f32>, v23: vec4<f32>, i: u32) -> vec2<f32> {
    if (i == 0u) {
        return v01.xy;
    }
    if (i == 1u) {
        return v01.zw;
    }
    if (i == 2u) {
        return v23.xy;
    }
    return v23.zw;
}

// Projects the first `count` vertices onto `axis`, returning the [min, max]
// scalar interval of the dot products; mirrors the reference `project`.
fn project(v01: vec4<f32>, v23: vec4<f32>, count: u32, axis: vec2<f32>) -> vec2<f32> {
    var lo = dot(v01.xy, axis);
    var hi = lo;
    if (count > 1u) {
        let d = dot(v01.zw, axis);
        lo = min(lo, d);
        hi = max(hi, d);
    }
    if (count > 2u) {
        let d = dot(v23.xy, axis);
        lo = min(lo, d);
        hi = max(hi, d);
    }
    if (count > 3u) {
        let d = dot(v23.zw, axis);
        lo = min(lo, d);
        hi = max(hi, d);
    }
    return vec2<f32>(lo, hi);
}

// Arithmetic mean of the first `count` vertices, used only to orient the MTV
// axis from a toward b; mirrors the reference `centroid`.
fn centroid(v01: vec4<f32>, v23: vec4<f32>, count: u32) -> vec2<f32> {
    var sum = v01.xy;
    if (count > 1u) {
        sum = sum + v01.zw;
    }
    if (count > 2u) {
        sum = sum + v23.xy;
    }
    if (count > 3u) {
        sum = sum + v23.zw;
    }
    return sum / f32(count);
}

// Sweeps the edges of the polygon whose vertices are `e01`/`e23` (count
// `ecount`), updating the running state against the fixed pair a/b. Degenerate
// (zero-length) edges are skipped exactly as the reference `edge_axes` does.
fn scan(
    acc: Acc,
    e01: vec4<f32>,
    e23: vec4<f32>,
    ecount: u32,
    a01: vec4<f32>,
    a23: vec4<f32>,
    ca: u32,
    b01: vec4<f32>,
    b23: vec4<f32>,
    cb: u32,
) -> Acc {
    var r = acc;
    for (var i = 0u; i < ecount; i = i + 1u) {
        let p = vget(e01, e23, i);
        let qv = vget(e01, e23, (i + 1u) % ecount);
        let edge = qv - p;
        let nrm = vec2<f32>(-edge.y, edge.x);
        let len = sqrt(dot(nrm, nrm));
        if (len > SAT_EPS) {
            r.has_axis = 1u;
            let axis = nrm / len;
            let pa = project(a01, a23, ca, axis);
            let pb = project(b01, b23, cb, axis);
            // overlap = min(max_a, max_b) - max(min_a, min_b)
            let upper = min(pa.y, pb.y);
            let lower = max(pa.x, pb.x);
            let overlap = upper - lower;
            if (overlap < -SAT_EPS) {
                r.separated = 1u;
            } else {
                // Clamp a tiny negative (touching) overlap to a non-negative depth.
                let depth = max(overlap, 0.0);
                if (r.found == 0u || depth < r.best_depth - SAT_EPS) {
                    r.best_depth = depth;
                    r.best_axis = axis;
                    r.found = 1u;
                }
            }
        }
    }
    return r;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.axis = vec2<f32>(0.0, 0.0);
    out.depth = 0.0;
    out.hit = 0u;

    // A polygon with fewer than two vertices has no separating face: no overlap.
    if (q.count_a < 2u || q.count_b < 2u) {
        results[idx] = out;
        return;
    }

    var acc: Acc;
    acc.found = 0u;
    acc.separated = 0u;
    acc.has_axis = 0u;
    acc.best_depth = 0.0;
    acc.best_axis = vec2<f32>(0.0, 0.0);

    // Candidate axes are a's edge normals first, then b's, matching the
    // reference ordering so the smallest-overlap tie-break is identical.
    acc = scan(acc, q.a01, q.a23, q.count_a, q.a01, q.a23, q.count_a, q.b01, q.b23, q.count_b);
    acc = scan(acc, q.b01, q.b23, q.count_b, q.a01, q.a23, q.count_a, q.b01, q.b23, q.count_b);

    // A separating axis, or no valid axis at all, means no overlap.
    if (acc.separated == 1u || acc.has_axis == 0u) {
        results[idx] = out;
        return;
    }

    // No separating axis: overlap confirmed. Orient the minimum-overlap axis
    // from a's centroid toward b's, matching the reference `mtv`.
    let dir = centroid(q.b01, q.b23, q.count_b) - centroid(q.a01, q.a23, q.count_a);
    var oriented = acc.best_axis;
    if (dot(acc.best_axis, dir) < 0.0) {
        oriented = vec2<f32>(-oriented.x, -oriented.y);
    }

    out.axis = oriented;
    out.depth = acc.best_depth;
    out.hit = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SAT_COLLISION_2D_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pair, matching the `WGSL` `Query` struct.
/// Each polygon's four vertices are packed into two `vec4` lanes so every field
/// stays `16`-byte aligned on device without manual `vec2` stride arithmetic.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Polygon `a` vertices `0` and `1` as `(x0, y0, x1, y1)`.
    a01: [f32; 4],
    /// Polygon `a` vertices `2` and `3` as `(x2, y2, x3, y3)`.
    a23: [f32; 4],
    /// Polygon `b` vertices `0` and `1` as `(x0, y0, x1, y1)`.
    b01: [f32; 4],
    /// Polygon `b` vertices `2` and `3` as `(x2, y2, x3, y3)`.
    b23: [f32; 4],
    /// Valid vertex count of polygon `a`.
    count_a: u32,
    /// Valid vertex count of polygon `b`.
    count_b: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `MTV` axis (zero when the pair is disjoint).
    axis: [f32; 2],
    /// `MTV` penetration depth (zero when the pair is disjoint).
    depth: f32,
    /// `1` when the polygons overlap or touch, `0` otherwise.
    hit: u32,
}

/// One collision query: two convex `CCW` polygons of up to [`MAX_VERTS`]
/// vertices each, with the valid vertex count of each.
///
/// Only the first `count_a` / `count_b` vertices are read; trailing slots are
/// ignored, so a triangle (`count = 3`) and an oriented box (`count = 4`) share
/// one layout. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SatCollision2dQuery {
    /// Polygon `a` vertices, `CCW`; only the first `count_a` are used.
    pub poly_a: [[f32; 2]; MAX_VERTS],
    /// Valid vertex count of polygon `a` (`2..=4` for a real face set).
    pub count_a: u32,
    /// Polygon `b` vertices, `CCW`; only the first `count_b` are used.
    pub poly_b: [[f32; 2]; MAX_VERTS],
    /// Valid vertex count of polygon `b` (`2..=4` for a real face set).
    pub count_b: u32,
}

/// The resolved verdict for one pair, mirroring the reference
/// [`overlaps`](prism_render_architecture::particle::sat_collision_2d::overlaps)
/// and
/// [`mtv`](prism_render_architecture::particle::sat_collision_2d::mtv).
///
/// `hit` is the overlap `bool`; `mtv` carries the reused
/// [`Mtv`](prism_render_architecture::particle::sat_collision_2d::Mtv) when the
/// pair overlaps and [`None`] when a separating axis proves them disjoint.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because the [`Mtv`] holds `f32`
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SatCollision2dResult {
    /// Whether the two polygons overlap or touch.
    pub hit: bool,
    /// The minimum translation vector when overlapping, else [`None`].
    pub mtv: Option<Mtv>,
}

/// Encodes one [`SatCollision2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SatCollision2dQuery) -> GpuQuery {
    let a = q.poly_a;
    let b = q.poly_b;
    GpuQuery {
        a01: [a[0][0], a[0][1], a[1][0], a[1][1]],
        a23: [a[2][0], a[2][1], a[3][0], a[3][1]],
        b01: [b[0][0], b[0][1], b[1][0], b[1][1]],
        b23: [b[2][0], b[2][1], b[3][0], b[3][1]],
        count_a: q.count_a,
        count_b: q.count_b,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SatCollision2dResult`],
/// turning the overlap flag back into an [`Option<Mtv>`].
fn decode_result(raw: &GpuResult) -> SatCollision2dResult {
    let hit = raw.hit == CODE_HIT;
    let mtv = if hit {
        Some(Mtv::new(Vec2::new(raw.axis[0], raw.axis[1]), raw.depth))
    } else {
        None
    };
    SatCollision2dResult { hit, mtv }
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

/// A compiled, reusable 2D `SAT` overlap + `MTV` compute pipeline, twinning the
/// `CPU` golden
/// [`sat_collision_2d`](prism_render_architecture::particle::sat_collision_2d).
pub struct GpuSatCollision2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSatCollision2d {
    /// Compiles the 2D `SAT` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSatCollision2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sat_collision_2d"),
            source: ShaderSource::Wgsl(SAT_COLLISION_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSatCollision2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pair in `queries` and returns one [`SatCollision2dResult`]
    /// per input, in order.
    ///
    /// The overlap flag equals the reference exactly for pairs clear of the
    /// touching boundary; the `MTV` axis and depth match to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SatCollision2dQuery],
    ) -> Vec<SatCollision2dResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_bind_group"),
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
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sat_collision_2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sat_collision_2d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
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
