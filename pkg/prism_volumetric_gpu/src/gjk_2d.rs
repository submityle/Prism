//! `wgpu` compute twin of the 2D convex-polygon Boolean intersection
//! `Gilbert-Johnson-Keerthi` (`GJK`) golden
//! ([`gjk_2d`](prism_render_architecture::particle::gjk_2d), particle design
//! §8.2, §10).
//!
//! The `CPU` golden
//! [`gjk_2d`](prism_render_architecture::particle::gjk_2d) owns the small,
//! verifiable overlap contract several particle stages share: given two convex
//! 2D point sets expressed as vertex rings it answers *do they share any point*
//! with the `GJK` support-mapping simplex search over their Minkowski
//! difference
//! ([`intersects`](prism_render_architecture::particle::gjk_2d::intersects),
//! built on
//! [`support`](prism_render_architecture::particle::gjk_2d::support) and
//! [`minkowski_support`](prism_render_architecture::particle::gjk_2d::minkowski_support)).
//! [`GpuGjk2d`] is the on-device twin: one thread evolves one pair's simplex, so
//! a passing real-device parity test is direct evidence the ported kernel folds
//! the same support projections, the same point/line/triangle simplex evolution
//! and the same separating-direction test the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! For each `(a, b)` pair of fixed-capacity convex polygons (up to
//! [`MAX_VERTS`] vertices each, the host passing the real counts) the kernel
//! reproduces the whole reference search control flow for control flow: it seeds
//! the simplex with the Minkowski support along `+x`, then on each iteration it
//! short-circuits to *intersect* when the search direction collapses onto the
//! origin (squared length at or below
//! [`DIR_EPS_SQ`](prism_render_architecture::particle::gjk_2d)), probes a fresh
//! Minkowski support, declares the pair *disjoint* when that probe fails to pass
//! the origin along `dir` (signed projection below `-SEP_EPS * |dir|`), reports
//! *intersect* when the probe duplicates an existing simplex vertex (squared
//! gap at or below `DUP_EPS_SQ`), and otherwise folds the probe into the simplex
//! and evolves it. The point, line and triangle cases pick the next direction
//! from the triple-product edge normals exactly as the reference `do_simplex`,
//! and the fixed [`MAX_ITERS`] bound makes termination deterministic.
//!
//! # Correctness model
//!
//! The overlap verdict is a discrete classification built from `f32` magnitude
//! and sign comparisons against the reference's `SEP_EPS`, `DIR_EPS_SQ` and
//! `DUP_EPS_SQ` bands, never from an `f32` `==`. For pairs clear of the contact
//! boundary the `CPU` and `GPU` fold the identical sequence of support
//! projections and simplex trims, so they agree exactly and the parity test
//! asserts an exact `==` on the overlap `bool`. The intermediate dot and
//! triple-product arithmetic threads through multiplies, adds and one `sqrt`
//! per iteration, so a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate; the fixtures are therefore placed clearly separated or clearly
//! overlapping (far from a grazing contact) where a `ULP`-scale perturbation
//! cannot flip the verdict.
//!
//! # Degenerate inputs
//!
//! A single vertex encodes a point and two vertices a segment, both handled as
//! degenerate convex sets exactly as the reference. A polygon with zero
//! vertices has no support point and is reported as non-intersecting, mirroring
//! the reference empty-input guard. An empty pair batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `min`, `max`,
//! `+ - * /` and one `sqrt` to length-normalize the separation tolerance — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Each thread sweeps at most [`MAX_ITERS`] simplex steps, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_2d`；textbook
//! 2D `GJK` convex-overlap simplex search plus `wgpu` compute dispatch；无第三方引擎
//! 源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::gjk_2d::{intersects, Vec2};
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
///
/// Provenance: 本仓 `prism_volumetric_gpu` 内核约定。
const WORKGROUP_SIZE: u32 = 64;

/// Maximum vertex count per polygon the fixed-capacity `std430` layout carries.
/// Matches the module task bound `MAX_VERTS_A` / `MAX_VERTS_B`; the host passes
/// the real counts and only the first `count` slots are read.
///
/// Provenance: 本模块 `gjk_2d` 任务约定（`MAX_VERTS_A = 16`，`MAX_VERTS_B = 16`）。
pub const MAX_VERTS: usize = 16;

/// Discrete intersection code written by the kernel for an overlapping pair:
/// matches the host `== 1` decode in [`decode_result`]. A direct `f32` equality
/// is forbidden, so the kernel emits an integer flag rather than a sentinel
/// float.
///
/// Provenance: 本模块 `gjk_2d` 布尔分类码约定。
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` 2D `GJK` convex-overlap kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`gjk_2d`](prism_render_architecture::particle::gjk_2d) step for step; see
/// the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_2d`。
const GJK_2D_WGSL: &str = r#"
// 2D GJK convex-overlap twin: one thread per polygon pair evolves a point ->
// line -> triangle simplex over the Minkowski difference A - B, probing it only
// through the support mapping and declaring the pair disjoint on the first
// separating direction, or intersecting when the direction collapses onto the
// origin, a probe duplicates a simplex vertex, or a triangle encloses the
// origin. It mirrors the CPU golden `particle::gjk_2d` step for step, uses only
// the portable core-WGSL subset (dot/min/max, + - * / and one sqrt), needs no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. Each thread sweeps at most MAX_ITERS simplex steps, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gjk_2d；无第三方引擎
// 源码或衍生代码。

// Relative tolerance on the signed support-plane distance classifying a
// separating direction. Matches the reference `SEP_EPS`.
const SEP_EPS: f32 = 1.0e-5;

// Squared-length threshold below which the search direction has collapsed onto
// the origin. Matches the reference `DIR_EPS_SQ`.
const DIR_EPS_SQ: f32 = 1.0e-10;

// Squared-distance threshold below which a probe duplicates a simplex vertex.
// Matches the reference `DUP_EPS_SQ`.
const DUP_EPS_SQ: f32 = 1.0e-10;

// Hard cap on simplex iterations. Matches the reference `MAX_ITERS`.
const MAX_ITERS: u32 = 64u;

// Maximum vertices per polygon in the fixed-capacity layout.
const MAX_VERTS: u32 = 16u;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One pair. The two vertex rings are fixed-capacity std430 arrays of vec2
// (8-byte stride), with the valid counts and two pad lanes matching the host
// `GpuQuery`.
struct Query {
    verts_a: array<vec2<f32>, 16>,
    verts_b: array<vec2<f32>, 16>,
    count_a: u32,
    count_b: u32,
    pad0: u32,
    pad1: u32,
}

// One result. The intersection flag as 0u/1u plus three pad words, a 16-byte
// std430 stride matching the host `GpuResult`.
struct Result {
    intersects: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The running simplex: up to three support points and the valid count.
struct Simplex {
    p: array<vec2<f32>, 3>,
    n: u32,
}

// The result of one `do_simplex` step: whether the triangle encloses the origin
// (the pair overlaps), the trimmed simplex and the next search direction.
struct Step {
    done: u32,
    simplex: Simplex,
    dir: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The vector triple product `(a x b) x c = b (a . c) - a (b . c)`, used to
// build a simplex-edge normal pointing toward the origin. Mirrors the reference
// `triple`.
fn triple(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    return b * dot(a, c) - a * dot(b, c);
}

// Returns the vertex of the first `count` entries farthest along `dir` (the
// support point of the convex set). Ties keep the earliest vertex via a strict
// `>`, matching the reference `support` for a deterministic search.
fn support(verts: array<vec2<f32>, 16>, count: u32, dir: vec2<f32>) -> vec2<f32> {
    var best = verts[0];
    var best_dot = dot(best, dir);
    for (var i = 1u; i < count; i = i + 1u) {
        let projected = dot(verts[i], dir);
        if (projected > best_dot) {
            best_dot = projected;
            best = verts[i];
        }
    }
    return best;
}

// The support point of the Minkowski difference `a - b` along `dir`, i.e.
// `support(a, dir) - support(b, -dir)`. Mirrors the reference `minkowski_support`.
fn minkowski_support(
    va: array<vec2<f32>, 16>,
    ca: u32,
    vb: array<vec2<f32>, 16>,
    cb: u32,
    dir: vec2<f32>,
) -> vec2<f32> {
    return support(va, ca, dir) - support(vb, cb, -dir);
}

// Evolves the current simplex toward the origin. Returns `done = 1u` when the
// simplex is a triangle enclosing the origin (overlap); otherwise trims the
// simplex to the feature closest to the origin and writes the next search
// direction. Mirrors the reference `do_simplex` branch for branch.
fn do_simplex(s: Simplex, dir: vec2<f32>) -> Step {
    var out: Step;
    out.done = 0u;
    out.simplex = s;
    out.dir = dir;

    if (s.n == 3u) {
        let a = s.p[2];
        let b = s.p[1];
        let c = s.p[0];
        let ab = b - a;
        let ac = c - a;
        let ao = -a;
        let ab_perp = triple(ac, ab, ab);
        let ac_perp = triple(ab, ac, ac);
        if (dot(ab_perp, ao) > 0.0) {
            var ns: Simplex;
            ns.p[0] = b;
            ns.p[1] = a;
            ns.n = 2u;
            out.simplex = ns;
            out.dir = ab_perp;
            return out;
        }
        if (dot(ac_perp, ao) > 0.0) {
            var ns: Simplex;
            ns.p[0] = c;
            ns.p[1] = a;
            ns.n = 2u;
            out.simplex = ns;
            out.dir = ac_perp;
            return out;
        }
        out.done = 1u;
        return out;
    }

    // Line case: two points, `a` newest.
    let a = s.p[1];
    let b = s.p[0];
    let ab = b - a;
    let ao = -a;
    if (dot(ab, ao) > 0.0) {
        out.dir = triple(ab, ao, ab);
    } else {
        var ns: Simplex;
        ns.p[0] = a;
        ns.n = 1u;
        out.simplex = ns;
        out.dir = ao;
    }
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var res: Result;
    res.intersects = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;

    let ca = queries[idx].count_a;
    let cb = queries[idx].count_b;
    // Empty input never intersects, mirroring the reference guard.
    if (ca == 0u || cb == 0u) {
        results[idx] = res;
        return;
    }

    var va = queries[idx].verts_a;
    var vb = queries[idx].verts_b;

    var dir = vec2<f32>(1.0, 0.0);
    let first = minkowski_support(va, ca, vb, cb, dir);
    var simplex: Simplex;
    simplex.p[0] = first;
    simplex.n = 1u;
    dir = -first;

    // Default verdict: converged within the iteration bound without finding a
    // separating direction, which the reference reports as intersecting.
    var verdict = 1u;
    for (var iter = 0u; iter < MAX_ITERS; iter = iter + 1u) {
        if (dot(dir, dir) <= DIR_EPS_SQ) {
            // The direction collapsed onto the origin: the sets touch.
            verdict = 1u;
            break;
        }

        let probe = minkowski_support(va, ca, vb, cb, dir);
        let dir_len = sqrt(dot(dir, dir));
        let projection = dot(probe, dir);
        if (projection < -SEP_EPS * dir_len) {
            // The farthest point of `a - b` along `dir` fails to reach the
            // origin: `dir` is a separating direction.
            verdict = 0u;
            break;
        }

        var duplicate = 0u;
        for (var j = 0u; j < simplex.n; j = j + 1u) {
            let d = simplex.p[j] - probe;
            if (dot(d, d) <= DUP_EPS_SQ) {
                duplicate = 1u;
            }
        }
        if (duplicate == 1u) {
            // No new vertex can be added; a strictly separated pair would have
            // returned above, so the origin is at or inside the simplex.
            verdict = 1u;
            break;
        }

        simplex.p[simplex.n] = probe;
        simplex.n = simplex.n + 1u;
        let step = do_simplex(simplex, dir);
        simplex = step.simplex;
        dir = step.dir;
        if (step.done == 1u) {
            verdict = 1u;
            break;
        }
    }

    res.intersects = verdict;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`GJK_2D_WGSL`].
///
/// Provenance: 本模块 `gjk_2d` 的 `std430`/`std140` 布局镜像。
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
/// Each polygon's vertices are a fixed-capacity `[[f32; 2]; MAX_VERTS]` array
/// (`8`-byte stride, matching the device `array<vec2<f32>, 16>`), followed by
/// the valid counts and two pad lanes.
///
/// Provenance: 本模块 `gjk_2d` 的 `std430` 上传布局。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Polygon `a` vertices; only the first `count_a` are read.
    verts_a: [[f32; 2]; MAX_VERTS],
    /// Polygon `b` vertices; only the first `count_b` are read.
    verts_b: [[f32; 2]; MAX_VERTS],
    /// Valid vertex count of polygon `a`.
    count_a: u32,
    /// Valid vertex count of polygon `b`.
    count_b: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the intersection flag plus three pad words.
///
/// Provenance: 本模块 `gjk_2d` 的 `std430` 回读布局。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Intersection flag: `1` when the polygons overlap or touch, `0` otherwise.
    intersects: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One intersection query: two convex polygons of up to [`MAX_VERTS`] vertices
/// each, with the valid vertex count of each.
///
/// Only the first `count_a` / `count_b` vertices are read; trailing slots are
/// ignored, so a point (`count = 1`), a segment (`count = 2`) and a polygon
/// share one layout. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` geometry.
///
/// Provenance: 本模块 `gjk_2d` 新建的查询类型。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGjk2dQuery {
    /// Polygon `a` vertices; only the first `count_a` are used.
    pub poly_a: [[f32; 2]; MAX_VERTS],
    /// Valid vertex count of polygon `a` (`0..=MAX_VERTS`).
    pub count_a: u32,
    /// Polygon `b` vertices; only the first `count_b` are used.
    pub poly_b: [[f32; 2]; MAX_VERTS],
    /// Valid vertex count of polygon `b` (`0..=MAX_VERTS`).
    pub count_b: u32,
}

/// The resolved verdict for one pair, mirroring the reference
/// [`intersects`](prism_render_architecture::particle::gjk_2d::intersects).
///
/// `intersects` carries the overlap flag as a `0`/`1` `u32` (`1` when the two
/// convex sets overlap or touch, matching the `CODE_HIT` encoding), so a
/// boolean verdict stays an exact integer compare rather than an `f32` test.
/// Derives [`Eq`] because the single field is an integer code.
///
/// Provenance: 本模块 `gjk_2d` 新建的结果类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuGjk2dResult {
    /// Overlap flag: `1` when the polygons overlap or touch, `0` otherwise.
    pub intersects: u32,
}

/// Encodes one [`GpuGjk2dQuery`] into its `std430` [`GpuQuery`] slot.
///
/// Provenance: 本模块 `gjk_2d` 的上传打包。
fn encode_query(q: &GpuGjk2dQuery) -> GpuQuery {
    GpuQuery {
        verts_a: q.poly_a,
        verts_b: q.poly_b,
        count_a: q.count_a,
        count_b: q.count_b,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuGjk2dResult`].
///
/// Provenance: 本模块 `gjk_2d` 的回读解包。
fn decode_result(raw: &GpuResult) -> GpuGjk2dResult {
    GpuGjk2dResult {
        intersects: u32::from(raw.intersects == CODE_HIT),
    }
}

/// The `CPU` golden verdict for one query, slicing each polygon to its valid
/// count and dispatching to the reference
/// [`intersects`](prism_render_architecture::particle::gjk_2d::intersects) so
/// callers (and the parity test) can pin the twin lane for lane. Returns `1`
/// when the convex sets overlap or touch and `0` otherwise, matching the
/// [`GpuGjk2dResult`] encoding.
///
/// Provenance: 调用本仓 `prism_render_architecture::particle::gjk_2d::intersects`。
#[must_use]
pub fn cpu_reference(query: &GpuGjk2dQuery) -> u32 {
    let a = alloc_vec(&query.poly_a, query.count_a);
    let b = alloc_vec(&query.poly_b, query.count_b);
    u32::from(intersects(&a, &b))
}

/// Collects the first `count` fixed-capacity slots into reference [`Vec2`]
/// values for the golden call.
///
/// Provenance: 本模块 `gjk_2d` 的金标准输入适配。
fn alloc_vec(verts: &[[f32; 2]; MAX_VERTS], count: u32) -> Vec<Vec2> {
    verts[..count as usize]
        .iter()
        .map(|p| Vec2::new(p[0], p[1]))
        .collect()
}

/// Builds a compute-visible buffer binding layout entry.
///
/// Provenance: 本仓 `prism_volumetric_gpu` 绑定布局约定。
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

/// A compiled, reusable 2D `GJK` convex-overlap compute pipeline, twinning the
/// `CPU` golden
/// [`gjk_2d`](prism_render_architecture::particle::gjk_2d).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_2d`。
pub struct GpuGjk2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGjk2d {
    /// Compiles the 2D `GJK` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 本模块 `gjk_2d` 的管线构建。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGjk2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gjk_2d"),
            source: ShaderSource::Wgsl(GJK_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gjk_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gjk_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gjk_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGjk2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pair in `queries` and returns one [`GpuGjk2dResult`] per
    /// input, in order.
    ///
    /// The overlap flag equals the reference exactly for pairs clear of the
    /// contact boundary. An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 本模块 `gjk_2d` 的分发与回读。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[GpuGjk2dQuery]) -> Vec<GpuGjk2dResult> {
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
            label: Some("prism_volumetric_gjk_2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gjk_2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gjk_2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gjk_2d_bind_group"),
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
            label: Some("prism_volumetric_gjk_2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gjk_2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gjk_2d_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
