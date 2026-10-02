//! `wgpu` compute twin of the 3D convex-polytope Boolean intersection
//! `Gilbert-Johnson-Keerthi` (`GJK`) golden
//! ([`gjk_3d`](prism_render_architecture::particle::gjk_3d), particle design
//! §8.2, §10, §13).
//!
//! The `CPU` golden
//! [`gjk_3d`](prism_render_architecture::particle::gjk_3d) owns the small,
//! verifiable overlap contract several particle stages share: given two convex
//! 3D point sets expressed as convex-hull vertices it answers *do they share
//! any point* with the `GJK` support-mapping simplex search over their
//! Minkowski difference `A - B`
//! ([`intersect`](prism_render_architecture::particle::gjk_3d::intersect),
//! built on
//! [`support`](prism_render_architecture::particle::gjk_3d::support) and the
//! point/line/triangle/tetrahedron
//! [`Simplex`](prism_render_architecture::particle::gjk_3d::Simplex)
//! evolution). [`GpuGjk3d`] is the on-device twin: one thread evolves one
//! pair's simplex, so a passing real-device parity test is direct evidence the
//! ported kernel folds the same support projections, the same
//! point/line/triangle/tetrahedron simplex evolution and the same
//! separating-direction test the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each `(a, b)` pair of fixed-capacity convex polytopes (up to
//! [`MAX_VERTS`] vertices each, the host passing the real counts) the kernel
//! reproduces the whole reference search control flow for control flow: it seeds
//! the simplex with the Minkowski support along `+x`, short-circuits to
//! *intersect* when the seed direction already sits on the origin (squared
//! length at or below
//! [`DIR_EPS_SQ`](prism_render_architecture::particle::gjk_3d::DIR_EPS_SQ)),
//! then on each iteration probes a fresh Minkowski support, declares the pair
//! *disjoint* when that probe fails to pass the origin along `dir` (projection
//! strictly below `0`), reports *intersect* when the probe duplicates an
//! existing simplex vertex (squared gap at or below
//! [`DUP_EPS_SQ`](prism_render_architecture::particle::gjk_3d::DUP_EPS_SQ)),
//! and otherwise folds the probe into the simplex and evolves it. The line,
//! triangle and tetrahedron cases pick the next direction from the
//! triple-product edge normals and cross-product face normals exactly as the
//! reference `do_simplex`, and the fixed
//! [`MAX_ITERS`](prism_render_architecture::particle::gjk_3d::MAX_ITERS) bound
//! makes termination deterministic.
//!
//! # Correctness model
//!
//! The overlap verdict is a discrete classification built from `f32` dot/cross
//! sign comparisons and squared-magnitude comparisons against the reference's
//! `DIR_EPS_SQ` and `DUP_EPS_SQ` bands, never from an `f32` `==`. For pairs
//! clear of the contact boundary the `CPU` and `GPU` fold the identical
//! sequence of support projections and simplex trims, so they agree exactly and
//! the parity test asserts an exact `==` on the overlap `bool`. The
//! intermediate dot and cross arithmetic threads through multiplies and adds, so
//! a `GPU` may fuse a multiply-add the scalar reference leaves separate; the
//! fixtures are therefore placed clearly separated or clearly overlapping (far
//! from a grazing contact) where a `ULP`-scale perturbation cannot flip the
//! verdict.
//!
//! # Degenerate inputs
//!
//! A single vertex encodes a point and two vertices a segment, both handled as
//! degenerate convex sets exactly as the reference. A polytope with zero
//! vertices has no support point and is reported as non-intersecting, mirroring
//! the reference empty-input guard. An empty pair batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `cross`,
//! `min`, `max` and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, not even a `sqrt` (the Boolean `GJK` needs
//! none) and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Each thread sweeps at most [`MAX_VERTS`]-bounded
//! iterations, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_3d`；textbook
//! 3D `GJK` convex-overlap simplex search plus `wgpu` compute dispatch；无第三方引擎
//! 源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::gjk_3d::intersect;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
///
/// Provenance: 本仓 `prism_volumetric_gpu` 的 workgroup 尺寸约定。
const WORKGROUP_SIZE: u32 = 64;

/// Fixed per-polytope vertex capacity in the `std430` layout.
///
/// Matches the module task bound `MAX_VERTS_A` / `MAX_VERTS_B`; the host passes
/// the real counts and only the first `count` slots are read.
///
/// Provenance: 本模块 `gjk_3d` 任务约定（`MAX_VERTS_A = 16`，`MAX_VERTS_B = 16`）。
pub const MAX_VERTS: usize = 16;

/// Discrete intersection code written by the kernel for an overlapping pair:
/// matches the host `== 1` decode in [`decode_result`]. A direct `f32` equality
/// is forbidden, so the kernel emits an integer flag rather than a sentinel
/// float.
///
/// Provenance: 本模块 `gjk_3d` 布尔分类码约定。
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` 3D `GJK` convex-overlap kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`gjk_3d`](prism_render_architecture::particle::gjk_3d) step for step; see
/// the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_3d`。
const GJK_3D_WGSL: &str = r#"
// 3D GJK convex-overlap twin: one thread per polytope pair evolves a point ->
// line -> triangle -> tetrahedron simplex over the Minkowski difference A - B,
// probing it only through the support mapping and declaring the pair disjoint on
// the first separating direction, or intersecting when the direction collapses
// onto the origin, a probe duplicates a simplex vertex, or a tetrahedron
// encloses the origin. It mirrors the CPU golden `particle::gjk_3d` branch for
// branch, uses only the portable core-WGSL subset (dot/cross/min/max and
// + - * /), needs no transcendental call and not even a sqrt, and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. Each thread
// sweeps at most MAX_ITERS simplex steps, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gjk_3d；无第三方引擎
// 源码或衍生代码。

// Squared-length threshold below which the search direction has collapsed onto
// the origin (a boundary touch). Matches the reference `DIR_EPS_SQ`.
const DIR_EPS_SQ: f32 = 1.0e-10;

// Squared-distance threshold below which a probe duplicates a simplex vertex.
// Matches the reference `DUP_EPS_SQ`.
const DUP_EPS_SQ: f32 = 1.0e-10;

// Hard cap on simplex iterations. Matches the reference `MAX_ITERS`.
const MAX_ITERS: u32 = 32u;

// Maximum vertices per polytope in the fixed-capacity layout.
const MAX_VERTS: u32 = 16u;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One pair. The two vertex sets are fixed-capacity std430 arrays of vec4 (the
// xyz lane carries the vertex, the w lane is pad so the array stride is the
// 16-byte std430 vec3 stride without manual alignment arithmetic), with the
// valid counts and two pad lanes matching the host `GpuQuery`.
struct Query {
    verts_a: array<vec4<f32>, 16>,
    verts_b: array<vec4<f32>, 16>,
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

// The running simplex: up to four support points stored newest-first (index 0
// is the most recently added vertex `a`) and the valid count.
struct Simplex {
    p: array<vec3<f32>, 4>,
    n: u32,
}

// The result of one simplex-evolution step: whether a tetrahedron encloses the
// origin (the pair overlaps), the trimmed simplex and the next search direction.
struct Step {
    done: u32,
    simplex: Simplex,
    dir: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The vector triple product `(a x b) x c`, which lies in the plane of `a` and
// `b`. GJK uses it to build an edge normal perpendicular to the edge and
// pointing toward the origin. Mirrors the reference `triple`.
fn triple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return cross(cross(a, b), c);
}

// Returns the vertex of the first `count` entries farthest along `dir` (the
// support point of the convex set). Ties keep the earliest vertex via a strict
// `>`, matching the reference `support_point` for a deterministic search.
fn support_point(verts: array<vec4<f32>, 16>, count: u32, dir: vec3<f32>) -> vec3<f32> {
    var best = verts[0].xyz;
    var best_dot = dot(best, dir);
    for (var i = 1u; i < count; i = i + 1u) {
        let v = verts[i].xyz;
        let projected = dot(v, dir);
        if (projected > best_dot) {
            best_dot = projected;
            best = v;
        }
    }
    return best;
}

// The support point of the Minkowski difference `a - b` along `dir`, i.e.
// `support_point(a, dir) - support_point(b, -dir)`. Mirrors the reference
// `support`.
fn minkowski_support(
    va: array<vec4<f32>, 16>,
    ca: u32,
    vb: array<vec4<f32>, 16>,
    cb: u32,
    dir: vec3<f32>,
) -> vec3<f32> {
    return support_point(va, ca, dir) - support_point(vb, cb, -dir);
}

// The one-vertex-to-line evolution. With segment `[a, b]` (a newest), if the
// origin projects onto the edge the whole edge is kept and `dir` becomes the
// edge normal toward the origin; otherwise the simplex collapses to `a` and
// `dir` points from `a` at the origin. Mirrors the reference `line_case`.
fn line_case(s: Simplex, dir: vec3<f32>) -> Step {
    var out: Step;
    out.done = 0u;
    out.simplex = s;
    out.dir = dir;

    let a = s.p[0];
    let b = s.p[1];
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

// The triangle evolution. With `[a, b, c]` (a newest) it classifies the origin
// against the two free edges `ab` and `ac` and the two face half-spaces of the
// triangle normal `abc`, trimming to the closest edge (delegating to
// `line_case`) or orienting the surviving face toward the origin. Mirrors the
// reference `triangle_case` branch for branch.
fn triangle_case(s: Simplex, dir: vec3<f32>) -> Step {
    var out: Step;
    out.done = 0u;
    out.simplex = s;
    out.dir = dir;

    let a = s.p[0];
    let b = s.p[1];
    let c = s.p[2];
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = cross(ab, ac);

    if (dot(cross(abc, ac), ao) > 0.0) {
        if (dot(ac, ao) > 0.0) {
            var ns: Simplex;
            ns.p[0] = a;
            ns.p[1] = c;
            ns.n = 2u;
            out.simplex = ns;
            out.dir = triple(ac, ao, ac);
        } else {
            var ns: Simplex;
            ns.p[0] = a;
            ns.p[1] = b;
            ns.n = 2u;
            return line_case(ns, out.dir);
        }
    } else if (dot(cross(ab, abc), ao) > 0.0) {
        var ns: Simplex;
        ns.p[0] = a;
        ns.p[1] = b;
        ns.n = 2u;
        return line_case(ns, out.dir);
    } else if (dot(abc, ao) > 0.0) {
        out.dir = abc;
    } else {
        var ns: Simplex;
        ns.p[0] = a;
        ns.p[1] = c;
        ns.p[2] = b;
        ns.n = 3u;
        out.simplex = ns;
        out.dir = -abc;
    }
    return out;
}

// The tetrahedron evolution. With `[a, b, c, d]` (a newest) it tests the three
// faces incident to `a` (`abc`, `acd`, `adb`). If the origin lies outside one
// face the simplex drops to that triangle (delegating to `triangle_case`); if
// the origin is inside all three the tetrahedron encloses it and the shapes
// overlap, so it returns `done = 1u`. Mirrors the reference `tetra_case`.
fn tetra_case(s: Simplex, dir: vec3<f32>) -> Step {
    var out: Step;
    out.done = 0u;
    out.simplex = s;
    out.dir = dir;

    let a = s.p[0];
    let b = s.p[1];
    let c = s.p[2];
    let d = s.p[3];
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;
    let abc = cross(ab, ac);
    let acd = cross(ac, ad);
    let adb = cross(ad, ab);

    if (dot(abc, ao) > 0.0) {
        var ns: Simplex;
        ns.p[0] = a;
        ns.p[1] = b;
        ns.p[2] = c;
        ns.n = 3u;
        return triangle_case(ns, out.dir);
    }
    if (dot(acd, ao) > 0.0) {
        var ns: Simplex;
        ns.p[0] = a;
        ns.p[1] = c;
        ns.p[2] = d;
        ns.n = 3u;
        return triangle_case(ns, out.dir);
    }
    if (dot(adb, ao) > 0.0) {
        var ns: Simplex;
        ns.p[0] = a;
        ns.p[1] = d;
        ns.p[2] = b;
        ns.n = 3u;
        return triangle_case(ns, out.dir);
    }
    out.done = 1u;
    return out;
}

// Evolves the simplex one step toward the origin, dispatching on the current
// vertex count. Returns `done = 1u` only when a tetrahedron encloses the origin;
// otherwise it trims the simplex to the surviving feature and writes the next
// search direction. Mirrors the reference `do_simplex`.
fn do_simplex(s: Simplex, dir: vec3<f32>) -> Step {
    if (s.n == 2u) {
        return line_case(s, dir);
    }
    if (s.n == 3u) {
        return triangle_case(s, dir);
    }
    if (s.n == 4u) {
        return tetra_case(s, dir);
    }
    var out: Step;
    out.done = 0u;
    out.simplex = s;
    out.dir = dir;
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

    let initial_dir = vec3<f32>(1.0, 0.0, 0.0);
    let first = minkowski_support(va, ca, vb, cb, initial_dir);
    var simplex: Simplex;
    simplex.p[0] = first;
    simplex.n = 1u;
    var dir = -first;

    // The first support point is the origin: 0 is inside `A - B`.
    if (dot(dir, dir) <= DIR_EPS_SQ) {
        res.intersects = 1u;
        results[idx] = res;
        return;
    }

    // Default verdict: reached the iteration cap while still bracketing the
    // origin, which the reference reports as intersecting.
    var verdict = 1u;
    for (var iter = 0u; iter < MAX_ITERS; iter = iter + 1u) {
        let probe = minkowski_support(va, ca, vb, cb, dir);
        if (dot(probe, dir) < 0.0) {
            // The farthest point of `a - b` along `dir` still lies on the near
            // side of the origin: a separating direction exists.
            verdict = 0u;
            break;
        }

        var duplicate = 0u;
        for (var j = 0u; j < simplex.n; j = j + 1u) {
            let diff = simplex.p[j] - probe;
            if (dot(diff, diff) <= DUP_EPS_SQ) {
                duplicate = 1u;
            }
        }
        if (duplicate == 1u) {
            // The search cannot expand past a point it already holds: the origin
            // is on or inside the current feature.
            verdict = 1u;
            break;
        }

        // push_front: insert `probe` as the new vertex `a`, shifting the older
        // vertices back by one slot.
        for (var i = simplex.n; i > 0u; i = i - 1u) {
            simplex.p[i] = simplex.p[i - 1u];
        }
        simplex.p[0] = probe;
        simplex.n = simplex.n + 1u;

        let step = do_simplex(simplex, dir);
        simplex = step.simplex;
        dir = step.dir;
        if (step.done == 1u) {
            verdict = 1u;
            break;
        }
        if (dot(dir, dir) <= DIR_EPS_SQ) {
            // The surviving feature passes through the origin (a boundary
            // touch); no further direction can separate the shapes.
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
/// [`GJK_3D_WGSL`].
///
/// Provenance: 本模块 `gjk_3d` 的 `std430`/`std140` 布局镜像。
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
/// Each polytope's vertices are a fixed-capacity `[[f32; 4]; MAX_VERTS]` array
/// (`16`-byte stride, matching the device `array<vec4<f32>, 16>`; the `w` lane
/// pads each vertex to the `std430` `vec3` stride), followed by the valid counts
/// and two pad lanes.
///
/// Provenance: 本模块 `gjk_3d` 的 `std430` 上传布局。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Polytope `a` vertices in `xyz` (the `w` lane is pad); only the first
    /// `count_a` are read.
    verts_a: [[f32; 4]; MAX_VERTS],
    /// Polytope `b` vertices in `xyz` (the `w` lane is pad); only the first
    /// `count_b` are read.
    verts_b: [[f32; 4]; MAX_VERTS],
    /// Valid vertex count of polytope `a`.
    count_a: u32,
    /// Valid vertex count of polytope `b`.
    count_b: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the intersection flag plus three pad words.
///
/// Provenance: 本模块 `gjk_3d` 的 `std430` 回读布局。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Intersection flag: `1` when the polytopes overlap or touch, `0` otherwise.
    intersects: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One intersection query: two convex polytopes of up to [`MAX_VERTS`] vertices
/// each, with the valid vertex count of each.
///
/// Only the first `count_a` / `count_b` vertices are read; trailing slots are
/// ignored, so a point (`count = 1`), a segment (`count = 2`) and a polytope
/// share one layout. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` geometry.
///
/// Provenance: 本模块 `gjk_3d` 新建的查询类型。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGjk3dQuery {
    /// Polytope `a` vertices; only the first `count_a` are used.
    pub poly_a: [[f32; 3]; MAX_VERTS],
    /// Valid vertex count of polytope `a` (`0..=MAX_VERTS`).
    pub count_a: u32,
    /// Polytope `b` vertices; only the first `count_b` are used.
    pub poly_b: [[f32; 3]; MAX_VERTS],
    /// Valid vertex count of polytope `b` (`0..=MAX_VERTS`).
    pub count_b: u32,
}

/// The resolved verdict for one pair, mirroring the reference
/// [`intersect`](prism_render_architecture::particle::gjk_3d::intersect).
///
/// `intersects` carries the overlap flag as a `0`/`1` `u32` (`1` when the two
/// convex sets overlap or touch, matching the `CODE_HIT` encoding), so a
/// boolean verdict stays an exact integer compare rather than an `f32` test.
/// Derives [`Eq`] because the single field is an integer code.
///
/// Provenance: 本模块 `gjk_3d` 新建的结果类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuGjk3dResult {
    /// Overlap flag: `1` when the polytopes overlap or touch, `0` otherwise.
    pub intersects: u32,
}

/// Encodes one [`GpuGjk3dQuery`] into its `std430` [`GpuQuery`] slot, padding
/// each `[f32; 3]` vertex to a `[f32; 4]` lane and the unused fixed-capacity
/// slots with zeros.
///
/// Provenance: 本模块 `gjk_3d` 的上传打包。
fn encode_query(q: &GpuGjk3dQuery) -> GpuQuery {
    let mut verts_a = [[0.0_f32; 4]; MAX_VERTS];
    let mut verts_b = [[0.0_f32; 4]; MAX_VERTS];
    for i in 0..MAX_VERTS {
        let a = q.poly_a[i];
        let b = q.poly_b[i];
        verts_a[i] = [a[0], a[1], a[2], 0.0];
        verts_b[i] = [b[0], b[1], b[2], 0.0];
    }
    GpuQuery {
        verts_a,
        verts_b,
        count_a: q.count_a,
        count_b: q.count_b,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuGjk3dResult`].
///
/// Provenance: 本模块 `gjk_3d` 的回读解包。
fn decode_result(raw: &GpuResult) -> GpuGjk3dResult {
    GpuGjk3dResult {
        intersects: u32::from(raw.intersects == CODE_HIT),
    }
}

/// The `CPU` golden verdict for one query, slicing each polytope to its valid
/// count and dispatching to the reference
/// [`intersect`](prism_render_architecture::particle::gjk_3d::intersect) so
/// callers (and the parity test) can pin the twin lane for lane. Returns `1`
/// when the convex sets overlap or touch and `0` otherwise, matching the
/// [`GpuGjk3dResult`] encoding.
///
/// Provenance: 调用本仓 `prism_render_architecture::particle::gjk_3d::intersect`。
#[must_use]
pub fn cpu_reference(query: &GpuGjk3dQuery) -> u32 {
    let a = collect_verts(&query.poly_a, query.count_a);
    let b = collect_verts(&query.poly_b, query.count_b);
    u32::from(intersect(&a, &b))
}

/// Collects the first `count` fixed-capacity slots into a `[f32; 3]` vector for
/// the golden call.
///
/// Provenance: 本模块 `gjk_3d` 的金标准输入适配。
fn collect_verts(verts: &[[f32; 3]; MAX_VERTS], count: u32) -> Vec<[f32; 3]> {
    verts[..count as usize].to_vec()
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

/// A compiled, reusable 3D `GJK` convex-overlap compute pipeline, twinning the
/// `CPU` golden
/// [`gjk_3d`](prism_render_architecture::particle::gjk_3d).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_3d`。
pub struct GpuGjk3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGjk3d {
    /// Compiles the 3D `GJK` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 本模块 `gjk_3d` 的管线构建。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGjk3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gjk_3d"),
            source: ShaderSource::Wgsl(GJK_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gjk_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gjk_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gjk_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGjk3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pair in `queries` and returns one [`GpuGjk3dResult`] per
    /// input, in order.
    ///
    /// The overlap flag equals the reference exactly for pairs clear of the
    /// contact boundary. An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 本模块 `gjk_3d` 的分发与回读。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[GpuGjk3dQuery]) -> Vec<GpuGjk3dResult> {
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
            label: Some("prism_volumetric_gjk_3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gjk_3d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gjk_3d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gjk_3d_bind_group"),
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
            label: Some("prism_volumetric_gjk_3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gjk_3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gjk_3d_pass"),
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
