//! `wgpu` compute twin of the 3D Expanding-Polytope-Algorithm (`EPA`)
//! penetration golden
//! ([`epa_penetration_3d`](prism_render_architecture::particle::epa_penetration_3d),
//! particle design §7, §10 collision resolution).
//!
//! The `CPU` golden
//! [`epa_penetration_3d`](prism_render_architecture::particle::epa_penetration_3d)
//! owns the small, verifiable contract that turns two overlapping convex vertex
//! clouds into a *penetration depth* and a *contact normal*. It runs the classic
//! two-stage pipeline: a `Gilbert-Johnson-Keerthi` (`GJK`) simplex search
//! ([`gjk_tetrahedron`](prism_render_architecture::particle::epa_penetration_3d::gjk_tetrahedron))
//! evolves a simplex over the Minkowski difference `A - B` until it encloses the
//! origin and hands `EPA`
//! ([`epa_from_tetrahedron`](prism_render_architecture::particle::epa_penetration_3d::epa_from_tetrahedron))
//! a seed tetrahedron, then `EPA` grows that tetrahedron into a polytope hugging
//! the Minkowski boundary, repeatedly taking the face closest to the origin,
//! probing a fresh support point along its outward normal, and carving the
//! polytope open along the visible-face *horizon* until the closest face stops
//! advancing. [`penetration`](prism_render_architecture::particle::epa_penetration_3d::penetration)
//! ties the two stages together.
//!
//! [`GpuEpaPenetration3d`] is the on-device twin: one thread resolves one pair's
//! penetration, so a passing real-device parity test is direct evidence the
//! ported kernel folds the same support projections, the same point/line/
//! triangle/tetrahedron `GJK` simplex evolution and the same face-and-horizon
//! `EPA` refinement the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each `(a, b)` pair of fixed-capacity convex clouds (up to [`MAX_VERTS`]
//! vertices each, the host passing the real counts) the kernel reproduces the
//! whole reference control flow step for step: it seeds the `GJK` simplex with
//! the Minkowski support along `+x`, folds the point/line/triangle/tetrahedron
//! cases from the same triple-product edge normals, and declares the pair
//! disjoint on the first separating support. On overlap it seeds `EPA` with the
//! four tetrahedron vertices and the four seed faces built against the fixed
//! interior centroid, then on each refinement iteration it selects the
//! minimum-`dist` face, probes a support point along its `normal`, converges
//! when the probe advances less than [`EPA_TOLERANCE`] past that face, and
//! otherwise toggles the directed-edge horizon of every visible face and
//! stitches the new vertex onto each horizon edge. The constants
//! [`EPS`], [`EPA_TOLERANCE`], [`VISIBILITY_EPS`], [`EPA_MAX_ITERS`],
//! [`FACE_CAP`] and [`GJK_MAX_ITERS`] mirror the reference exactly.
//!
//! # Correctness model
//!
//! The overlap verdict is a discrete classification (`has_penetration` as a
//! `0`/`1` flag) built from `f32` magnitude and sign comparisons against the
//! reference bands, never from an `f32` `==`, so the parity test asserts an
//! exact `==` on that flag. For pairs clear of a contact tie the `CPU` and
//! `GPU` fold the identical sequence of support probes and face selections, so
//! the surviving closest face agrees and the `normal`/`depth` match within the
//! continuous-field tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), the
//! `depth` already pinned into the [`EPA_TOLERANCE`] convergence band. A `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, so the fixtures
//! are placed with a clear dominant overlap axis where a `ULP`-scale
//! perturbation cannot flip the chosen face.
//!
//! # Degenerate inputs
//!
//! A convex cloud with zero vertices has no support point and reports no
//! penetration, mirroring the reference empty-input guard. An empty pair batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `cross`,
//! `min`, `max`, `abs`, `+ - * /`, `bitcast` for an infinity sentinel and one
//! `sqrt` per face normalize — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`,
//! no inverse trigonometry and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. Each thread sweeps at most [`GJK_MAX_ITERS`]
//! simplex steps and [`EPA_MAX_ITERS`] refinement steps, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::epa_penetration_3d`；
//! textbook 3D `GJK` + `EPA` penetration solver plus `wgpu` compute dispatch；无第三方
//! 引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::epa_penetration_3d::penetration;
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

/// Maximum vertex count per convex cloud the fixed-capacity `std430` layout
/// carries. Matches the module task bound; the host passes the real counts and
/// only the first `count` slots are read.
///
/// Provenance: 本模块 `epa_penetration_3d` 任务约定（`MAX_VERTS = 16`）。
pub const MAX_VERTS: usize = 16;

/// Discrete penetration code written by the kernel for an overlapping pair,
/// matching the host `== 1` decode in [`decode_result`]. A direct `f32` equality
/// is forbidden, so the kernel emits an integer flag rather than a sentinel
/// float.
///
/// Provenance: 本模块 `epa_penetration_3d` 布尔分类码约定。
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` 3D `GJK` + `EPA` penetration kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`epa_penetration_3d`](prism_render_architecture::particle::epa_penetration_3d)
/// step for step; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::epa_penetration_3d`。
const EPA_PENETRATION_3D_WGSL: &str = r#"
// 3D GJK + EPA penetration twin: one thread per convex pair evolves a GJK
// simplex over the Minkowski difference A - B to a seed tetrahedron, then grows
// an EPA polytope that hugs the Minkowski boundary, each round taking the face
// closest to the origin, probing a support point along its outward normal, and
// carving the visible-face horizon until the closest face stops advancing. It
// mirrors the CPU golden `particle::epa_penetration_3d` step for step, uses only
// the portable core-WGSL subset (dot/cross/min/max/abs, + - * /, bitcast and one
// sqrt per face normalize), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. Each thread sweeps
// at most GJK_MAX_ITERS + EPA_MAX_ITERS steps, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::epa_penetration_3d；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a squared length or signed distance is treated as zero.
// Matches the reference `EPS`.
const EPS: f32 = 1.0e-6;

// EPA growth threshold: a support point advancing less than this past the
// closest face declares convergence. Matches the reference `EPA_TOLERANCE`.
const EPA_TOLERANCE: f32 = 1.0e-4;

// Visibility slack: a face is "seen" by a new support point only when the point
// lies strictly more than this beyond the face plane. Matches the reference
// `VISIBILITY_EPS`.
const VISIBILITY_EPS: f32 = 1.0e-6;

// Hard cap on EPA refinement iterations. Matches the reference `EPA_MAX_ITERS`.
const EPA_MAX_ITERS: u32 = 64u;

// Hard cap on the polytope face count, mirroring the reference `FACE_CAP`.
const FACE_CAP: u32 = 128u;

// Hard cap on GJK simplex-evolution iterations. Matches the reference
// `GJK_MAX_ITERS`.
const GJK_MAX_ITERS: u32 = 64u;

// Maximum vertices per convex cloud in the fixed-capacity layout.
const MAX_VERTS: u32 = 16u;

// Storage capacity for the growing polytope vertex list: four seed vertices plus
// at most one appended support point per EPA iteration.
const VERT_CAP: u32 = 72u;

// Storage capacity for the polytope face list. A margin above FACE_CAP so a
// single rebuild can momentarily exceed the cap before the reference break
// check fires.
const FACE_STORAGE: u32 = 160u;

// Storage capacity for the directed-edge horizon toggle buffer.
const HORIZON_CAP: u32 = 128u;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One pair. Each cloud is a fixed-capacity std430 array of vec4 (16-byte stride;
// only `.xyz` is used) so the array stride needs no manual padding, followed by
// the valid counts and two pad lanes matching the host `GpuQuery`.
struct Query {
    verts_a: array<vec4<f32>, 16>,
    verts_b: array<vec4<f32>, 16>,
    count_a: u32,
    count_b: u32,
    pad0: u32,
    pad1: u32,
}

// One result. The penetration flag as 0u/1u with three pad words, then the
// outward unit normal and the penetration depth, a 32-byte std430 stride
// matching the host `GpuResult`.
struct Result {
    has_penetration: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    normal: vec3<f32>,
    depth: f32,
}

// The result of normalizing a vector: `ok = 0u` when the input is shorter than
// EPS and has no well-defined direction.
struct NormRes {
    ok: u32,
    v: vec3<f32>,
}

// A GJK simplex: up to four support points, newest at the highest index, and the
// valid count.
struct Simplex {
    p: array<vec3<f32>, 4>,
    n: u32,
}

// One `do_simplex` step: `done = 1u` when the simplex is a tetrahedron enclosing
// the origin, plus the trimmed simplex and the next search direction.
struct GjkStep {
    simplex: Simplex,
    dir: vec3<f32>,
    done: u32,
}

// The GJK seed-search outcome: `found = 1u` with the four tetrahedron vertices
// when the bodies overlap.
struct GjkOut {
    found: u32,
    t0: vec3<f32>,
    t1: vec3<f32>,
    t2: vec3<f32>,
    t3: vec3<f32>,
}

// A polytope face: three vertex indices wound so `normal` points away from the
// interior, `dist` the signed distance from the origin to the face plane, and
// `valid = 0u` for a degenerate (zero-area) triangle.
struct Face {
    i0: u32,
    i1: u32,
    i2: u32,
    normal: vec3<f32>,
    dist: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Positive infinity built with bitcast, avoiding any transcendental or literal
// overflow. Used as the running minimum-distance sentinel.
fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Unit vector along `a`, or ok = 0u when `a` is shorter than EPS. The kernel's
// only use of sqrt. Mirrors the reference `v_normalize`.
fn v_normalize(a: vec3<f32>) -> NormRes {
    var out: NormRes;
    let len = sqrt(dot(a, a));
    if (len <= EPS) {
        out.ok = 0u;
        out.v = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        out.ok = 1u;
        out.v = a * (1.0 / len);
    }
    return out;
}

// A vector perpendicular to `a`, chosen from the coordinate axis least aligned
// with `a`. Mirrors the reference `any_perpendicular`.
fn any_perpendicular(a: vec3<f32>) -> vec3<f32> {
    let ax = abs(a.x);
    let ay = abs(a.y);
    let az = abs(a.z);
    var axis: vec3<f32>;
    if (ax <= ay && ax <= az) {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    } else if (ay <= az) {
        axis = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        axis = vec3<f32>(0.0, 0.0, 1.0);
    }
    return cross(a, axis);
}

// Farthest vertex of the first `count` entries along `dir`. Ties keep the
// earliest vertex via a strict `>`, matching the reference `support`.
fn support(verts: array<vec4<f32>, 16>, count: u32, dir: vec3<f32>) -> vec3<f32> {
    var best = verts[0].xyz;
    var best_dot = dot(best, dir);
    for (var i = 1u; i < count; i = i + 1u) {
        let p = verts[i].xyz;
        let d = dot(p, dir);
        if (d > best_dot) {
            best_dot = d;
            best = p;
        }
    }
    return best;
}

// Support point of the Minkowski difference `a - b` along `dir`, i.e.
// `support(a, dir) - support(b, -dir)`. Mirrors the reference
// `minkowski_support`.
fn minkowski_support(
    va: array<vec4<f32>, 16>,
    ca: u32,
    vb: array<vec4<f32>, 16>,
    cb: u32,
    dir: vec3<f32>,
) -> vec3<f32> {
    return support(va, ca, dir) - support(vb, cb, -dir);
}

// Builds a face on points `vi`, `vj`, `vk` (original indices `i`, `j`, `k`)
// whose unit normal points away from the interior reference, flipping the
// winding and normal when the raw normal points inward. `valid = 0u` for a
// degenerate triangle. Mirrors the reference `build_face`.
fn build_face(
    interior: vec3<f32>,
    vi: vec3<f32>,
    vj: vec3<f32>,
    vk: vec3<f32>,
    i: u32,
    j: u32,
    k: u32,
) -> Face {
    var out: Face;
    let raw = cross(vj - vi, vk - vi);
    let n = v_normalize(raw);
    if (n.ok == 0u) {
        out.valid = 0u;
        return out;
    }
    var o0 = i;
    var o1 = j;
    var o2 = k;
    var normal = n.v;
    if (dot(n.v, interior - vi) > 0.0) {
        o0 = i;
        o1 = k;
        o2 = j;
        normal = -n.v;
    }
    out.i0 = o0;
    out.i1 = o1;
    out.i2 = o2;
    out.normal = normal;
    out.dist = dot(normal, vi);
    out.valid = 1u;
    return out;
}

// Shared edge-`ab` reduction used by two branches of the triangle case. Mirrors
// the reference `triangle_edge_ab`.
fn triangle_edge_ab(a: vec3<f32>, b: vec3<f32>, ab: vec3<f32>, ao: vec3<f32>) -> GjkStep {
    var out: GjkStep;
    out.done = 0u;
    if (dot(ab, ao) > 0.0) {
        out.simplex.p[0] = b;
        out.simplex.p[1] = a;
        out.simplex.n = 2u;
        out.dir = cross(cross(ab, ao), ab);
    } else {
        out.simplex.p[0] = a;
        out.simplex.n = 1u;
        out.dir = ao;
    }
    return out;
}

// GJK line case: reduce a 2-point simplex toward the origin. Mirrors the
// reference `line_case`.
fn line_case(s: Simplex) -> GjkStep {
    var out: GjkStep;
    out.done = 0u;
    out.simplex = s;
    let a = s.p[1];
    let b = s.p[0];
    let ab = b - a;
    let ao = -a;
    if (dot(ab, ao) > 0.0) {
        let perp = cross(cross(ab, ao), ab);
        if (dot(perp, perp) <= EPS) {
            out.dir = any_perpendicular(ab);
        } else {
            out.dir = perp;
        }
    } else {
        out.simplex.p[0] = a;
        out.simplex.n = 1u;
        out.dir = ao;
    }
    return out;
}

// GJK triangle case: reduce a 3-point simplex toward the origin. Mirrors the
// reference `triangle_case` branch for branch.
fn triangle_case(s: Simplex) -> GjkStep {
    var out: GjkStep;
    out.done = 0u;
    out.simplex = s;
    let a = s.p[2];
    let b = s.p[1];
    let c = s.p[0];
    let ao = -a;
    let ab = b - a;
    let ac = c - a;
    let abc = cross(ab, ac);
    if (dot(cross(abc, ac), ao) > 0.0) {
        if (dot(ac, ao) > 0.0) {
            out.simplex.p[0] = c;
            out.simplex.p[1] = a;
            out.simplex.n = 2u;
            out.dir = cross(cross(ac, ao), ac);
        } else {
            out = triangle_edge_ab(a, b, ab, ao);
        }
    } else if (dot(cross(ab, abc), ao) > 0.0) {
        out = triangle_edge_ab(a, b, ab, ao);
    } else if (dot(abc, ao) > 0.0) {
        out.dir = abc;
    } else {
        out.simplex.p[0] = b;
        out.simplex.p[1] = c;
        out.simplex.p[2] = a;
        out.simplex.n = 3u;
        out.dir = -abc;
    }
    return out;
}

// GJK tetrahedron case: confirm the origin is enclosed or drop the one face it
// lies outside and recurse into the triangle case. Mirrors the reference
// `tetra_case`.
fn tetra_case(s: Simplex) -> GjkStep {
    var out: GjkStep;
    out.done = 0u;
    out.simplex = s;
    let a = s.p[3];
    let b = s.p[2];
    let c = s.p[1];
    let d = s.p[0];
    let ao = -a;
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    var abc = cross(ab, ac);
    var acd = cross(ac, ad);
    var adb = cross(ad, ab);
    if (dot(abc, ad) > 0.0) {
        abc = -abc;
    }
    if (dot(acd, ab) > 0.0) {
        acd = -acd;
    }
    if (dot(adb, ac) > 0.0) {
        adb = -adb;
    }
    if (dot(abc, ao) > 0.0) {
        var tri: Simplex;
        tri.p[0] = c;
        tri.p[1] = b;
        tri.p[2] = a;
        tri.n = 3u;
        out = triangle_case(tri);
    } else if (dot(acd, ao) > 0.0) {
        var tri: Simplex;
        tri.p[0] = d;
        tri.p[1] = c;
        tri.p[2] = a;
        tri.n = 3u;
        out = triangle_case(tri);
    } else if (dot(adb, ao) > 0.0) {
        var tri: Simplex;
        tri.p[0] = b;
        tri.p[1] = d;
        tri.p[2] = a;
        tri.n = 3u;
        out = triangle_case(tri);
    } else {
        out.done = 1u;
    }
    return out;
}

// Evolves a GJK simplex one step. Mirrors the reference `do_simplex`.
fn do_simplex(s: Simplex) -> GjkStep {
    if (s.n == 2u) {
        return line_case(s);
    }
    if (s.n == 3u) {
        return triangle_case(s);
    }
    if (s.n == 4u) {
        return tetra_case(s);
    }
    var out: GjkStep;
    out.simplex = s;
    out.dir = vec3<f32>(0.0, 0.0, 0.0);
    out.done = 0u;
    return out;
}

// Runs GJK on the Minkowski difference, returning the four vertices of a seed
// tetrahedron enclosing the origin on overlap. Mirrors the reference
// `gjk_tetrahedron`.
fn gjk_tetrahedron(
    va: array<vec4<f32>, 16>,
    ca: u32,
    vb: array<vec4<f32>, 16>,
    cb: u32,
) -> GjkOut {
    var out: GjkOut;
    out.found = 0u;
    var dir = vec3<f32>(1.0, 0.0, 0.0);
    let first = minkowski_support(va, ca, vb, cb, dir);
    var simplex: Simplex;
    simplex.p[0] = first;
    simplex.n = 1u;
    dir = -first;
    for (var iter = 0u; iter < GJK_MAX_ITERS; iter = iter + 1u) {
        if (dot(dir, dir) <= EPS) {
            dir = vec3<f32>(1.0, 0.0, 0.0);
        }
        let p = minkowski_support(va, ca, vb, cb, dir);
        if (dot(p, dir) < 0.0) {
            out.found = 0u;
            return out;
        }
        simplex.p[simplex.n] = p;
        simplex.n = simplex.n + 1u;
        let step = do_simplex(simplex);
        simplex = step.simplex;
        dir = step.dir;
        if (step.done == 1u) {
            out.found = 1u;
            out.t0 = simplex.p[0];
            out.t1 = simplex.p[1];
            out.t2 = simplex.p[2];
            out.t3 = simplex.p[3];
            return out;
        }
    }
    out.found = 0u;
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var res: Result;
    res.has_penetration = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    res.normal = vec3<f32>(0.0, 0.0, 0.0);
    res.depth = 0.0;

    let ca = queries[idx].count_a;
    let cb = queries[idx].count_b;
    // Empty input never penetrates, mirroring the reference guard.
    if (ca == 0u || cb == 0u) {
        results[idx] = res;
        return;
    }

    var va = queries[idx].verts_a;
    var vb = queries[idx].verts_b;

    // Stage 1: GJK seed tetrahedron.
    let seed = gjk_tetrahedron(va, ca, vb, cb);
    if (seed.found == 0u) {
        results[idx] = res;
        return;
    }

    // Stage 2: EPA refinement from the seed tetrahedron.
    var verts: array<vec3<f32>, 72>;
    verts[0] = seed.t0;
    verts[1] = seed.t1;
    verts[2] = seed.t2;
    verts[3] = seed.t3;
    var vcount = 4u;
    let centroid = (seed.t0 + seed.t1 + seed.t2 + seed.t3) * 0.25;

    var faces: array<Face, 160>;
    var fcount = 0u;
    // Seed faces from the four tetrahedron combinations.
    let ci0 = array<u32, 4>(0u, 0u, 0u, 1u);
    let ci1 = array<u32, 4>(1u, 1u, 2u, 2u);
    let ci2 = array<u32, 4>(2u, 3u, 3u, 3u);
    for (var c = 0u; c < 4u; c = c + 1u) {
        let i = ci0[c];
        let j = ci1[c];
        let k = ci2[c];
        let f = build_face(centroid, verts[i], verts[j], verts[k], i, j, k);
        if (f.valid == 1u) {
            faces[fcount] = f;
            fcount = fcount + 1u;
        }
    }
    // No non-degenerate seed face means no penetration can be reported.
    if (fcount == 0u) {
        results[idx] = res;
        return;
    }

    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var best_depth = inf();
    for (var it = 0u; it < EPA_MAX_ITERS; it = it + 1u) {
        // Select the face closest to the origin (earliest on a tie).
        var fidx = 0u;
        var min_dist = inf();
        for (var i = 0u; i < fcount; i = i + 1u) {
            if (faces[i].dist < min_dist) {
                min_dist = faces[i].dist;
                fidx = i;
            }
        }
        let normal = faces[fidx].normal;
        best_normal = normal;
        best_depth = min_dist;
        let p = minkowski_support(va, ca, vb, cb, normal);
        let advanced = dot(normal, p) - min_dist;
        if (advanced < EPA_TOLERANCE) {
            res.has_penetration = 1u;
            res.normal = normal;
            res.depth = min_dist;
            results[idx] = res;
            return;
        }

        // Append the fresh support point.
        let p_idx = vcount;
        verts[p_idx] = p;
        vcount = vcount + 1u;

        // Toggle the directed-edge horizon over visible faces and compact the
        // kept (non-visible) faces in place.
        var hx: array<u32, 128>;
        var hy: array<u32, 128>;
        var hcount = 0u;
        var kcount = 0u;
        for (var fi = 0u; fi < fcount; fi = fi + 1u) {
            let face = faces[fi];
            let visible = (dot(face.normal, p) - face.dist) > VISIBILITY_EPS;
            if (visible) {
                let ex = array<u32, 3>(face.i0, face.i1, face.i2);
                let ey = array<u32, 3>(face.i1, face.i2, face.i0);
                for (var e = 0u; e < 3u; e = e + 1u) {
                    let x = ex[e];
                    let y = ey[e];
                    var found = 0u;
                    var pos = 0u;
                    for (var hh = 0u; hh < hcount; hh = hh + 1u) {
                        if (hx[hh] == y && hy[hh] == x) {
                            found = 1u;
                            pos = hh;
                            break;
                        }
                    }
                    if (found == 1u) {
                        // Set-toggle: a shared edge cancels. Swap-remove keeps the
                        // same final edge set as the reference order-preserving
                        // remove, since the rebuild treats the edges as a set.
                        hcount = hcount - 1u;
                        hx[pos] = hx[hcount];
                        hy[pos] = hy[hcount];
                    } else {
                        if (hcount < HORIZON_CAP) {
                            hx[hcount] = x;
                            hy[hcount] = y;
                            hcount = hcount + 1u;
                        }
                    }
                }
            } else {
                faces[kcount] = face;
                kcount = kcount + 1u;
            }
        }
        fcount = kcount;
        if (hcount == 0u) {
            break;
        }

        // Stitch the new vertex onto every horizon edge.
        var overflow = 0u;
        for (var h = 0u; h < hcount; h = h + 1u) {
            let i = hx[h];
            let j = hy[h];
            let f = build_face(centroid, verts[i], verts[j], verts[p_idx], i, j, p_idx);
            if (f.valid == 1u) {
                if (fcount >= FACE_STORAGE) {
                    overflow = 1u;
                    break;
                }
                faces[fcount] = f;
                fcount = fcount + 1u;
            }
        }
        if (overflow == 1u) {
            break;
        }
        if (fcount == 0u || fcount > FACE_CAP) {
            break;
        }
    }

    // Iteration cap, empty polytope or face-cap exit: report the best face seen.
    res.has_penetration = 1u;
    res.normal = best_normal;
    res.depth = best_depth;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`EPA_PENETRATION_3D_WGSL`].
///
/// Provenance: 本模块 `epa_penetration_3d` 的 `std430`/`std140` 布局镜像。
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
/// Each cloud's vertices are a fixed-capacity `[[f32; 4]; MAX_VERTS]` array
/// (`16`-byte stride, matching the device `array<vec4<f32>, 16>`; only the first
/// three lanes carry the point), followed by the valid counts and two pad lanes.
///
/// Provenance: 本模块 `epa_penetration_3d` 的 `std430` 上传布局。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Cloud `a` vertices as `(x, y, z, _pad)`; only the first `count_a` read.
    verts_a: [[f32; 4]; MAX_VERTS],
    /// Cloud `b` vertices as `(x, y, z, _pad)`; only the first `count_b` read.
    verts_b: [[f32; 4]; MAX_VERTS],
    /// Valid vertex count of cloud `a`.
    count_a: u32,
    /// Valid vertex count of cloud `b`.
    count_b: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the penetration flag plus three pad words, then the outward unit
/// normal (`vec3` on a `16`-byte boundary) and the penetration depth.
///
/// Provenance: 本模块 `epa_penetration_3d` 的 `std430` 回读布局。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Penetration flag: `1` when the clouds overlap, `0` otherwise.
    has_penetration: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Outward unit contact normal pointing from body `b` toward body `a`.
    normal: [f32; 3],
    /// Penetration depth along `normal`; never negative.
    depth: f32,
}

/// One penetration query: two convex clouds of up to [`MAX_VERTS`] vertices
/// each, with the valid vertex count of each.
///
/// Only the first `count_a` / `count_b` vertices are read; trailing slots are
/// ignored. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// geometry.
///
/// Provenance: 本模块 `epa_penetration_3d` 新建的查询类型。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpaPenetration3dQuery {
    /// Cloud `a` vertices; only the first `count_a` are used.
    pub verts_a: [[f32; 3]; MAX_VERTS],
    /// Valid vertex count of cloud `a` (`0..=MAX_VERTS`).
    pub count_a: u32,
    /// Cloud `b` vertices; only the first `count_b` are used.
    pub verts_b: [[f32; 3]; MAX_VERTS],
    /// Valid vertex count of cloud `b` (`0..=MAX_VERTS`).
    pub count_b: u32,
}

/// The resolved contact for one pair, mirroring the reference
/// [`Penetration`](prism_render_architecture::particle::epa_penetration_3d::Penetration)
/// wrapped with an overlap flag.
///
/// `has_penetration` carries the overlap verdict as a `0`/`1` `u32` (`1` when
/// the clouds overlap, matching the `CODE_HIT` encoding), so the boolean verdict
/// stays an exact integer compare rather than an `f32` test. `normal` is the
/// outward unit contact normal and `depth` the penetration distance along it;
/// both are zero when `has_penetration` is `0`. Derives only [`PartialEq`]
/// because `normal` and `depth` are `f32`.
///
/// Provenance: 本模块 `epa_penetration_3d` 新建的结果类型。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuEpa {
    /// Overlap flag: `1` when the clouds penetrate, `0` otherwise.
    pub has_penetration: u32,
    /// Outward unit contact normal pointing from body `b` toward body `a`.
    pub normal: [f32; 3],
    /// Penetration depth along `normal`; never negative.
    pub depth: f32,
}

/// Encodes one [`EpaPenetration3dQuery`] into its `std430` [`GpuQuery`] slot,
/// widening each `[f32; 3]` point to a `[f32; 4]` lane with a zero pad.
///
/// Provenance: 本模块 `epa_penetration_3d` 的上传打包。
fn encode_query(q: &EpaPenetration3dQuery) -> GpuQuery {
    let mut verts_a = [[0.0_f32; 4]; MAX_VERTS];
    let mut verts_b = [[0.0_f32; 4]; MAX_VERTS];
    for (slot, p) in verts_a.iter_mut().zip(q.verts_a.iter()) {
        slot[0] = p[0];
        slot[1] = p[1];
        slot[2] = p[2];
    }
    for (slot, p) in verts_b.iter_mut().zip(q.verts_b.iter()) {
        slot[0] = p[0];
        slot[1] = p[1];
        slot[2] = p[2];
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

/// Decodes one packed [`GpuResult`] into the public [`GpuEpa`].
///
/// Provenance: 本模块 `epa_penetration_3d` 的回读解包。
fn decode_result(raw: &GpuResult) -> GpuEpa {
    GpuEpa {
        has_penetration: u32::from(raw.has_penetration == CODE_HIT),
        normal: raw.normal,
        depth: raw.depth,
    }
}

/// The `CPU` golden contact for one query, slicing each cloud to its valid count
/// and dispatching to the reference
/// [`penetration`](prism_render_architecture::particle::epa_penetration_3d::penetration)
/// so callers (and the parity test) can pin the twin lane for lane. Returns a
/// [`GpuEpa`] with `has_penetration = 1` and the resolved `normal`/`depth` on
/// overlap, or an all-zero result when the clouds are disjoint.
///
/// Provenance: 调用本仓 `prism_render_architecture::particle::epa_penetration_3d::penetration`。
#[must_use]
pub fn cpu_reference(query: &EpaPenetration3dQuery) -> GpuEpa {
    let a = alloc_vec(&query.verts_a, query.count_a);
    let b = alloc_vec(&query.verts_b, query.count_b);
    match penetration(&a, &b) {
        Some(pen) => GpuEpa {
            has_penetration: 1,
            normal: pen.normal,
            depth: pen.depth,
        },
        None => GpuEpa {
            has_penetration: 0,
            normal: [0.0, 0.0, 0.0],
            depth: 0.0,
        },
    }
}

/// Collects the first `count` fixed-capacity slots into reference `[f32; 3]`
/// points for the golden call.
///
/// Provenance: 本模块 `epa_penetration_3d` 的金标准输入适配。
fn alloc_vec(verts: &[[f32; 3]; MAX_VERTS], count: u32) -> Vec<[f32; 3]> {
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

/// A compiled, reusable 3D `GJK` + `EPA` penetration compute pipeline, twinning
/// the `CPU` golden
/// [`epa_penetration_3d`](prism_render_architecture::particle::epa_penetration_3d).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::epa_penetration_3d`。
pub struct GpuEpaPenetration3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEpaPenetration3d {
    /// Compiles the 3D `GJK` + `EPA` penetration kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 本模块 `epa_penetration_3d` 的管线构建。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEpaPenetration3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d"),
            source: ShaderSource::Wgsl(EPA_PENETRATION_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEpaPenetration3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pair in `queries` and returns one [`GpuEpa`] per input, in
    /// order.
    ///
    /// The `has_penetration` flag equals the reference exactly, and the
    /// `normal`/`depth` match within the continuous-field tolerance for pairs
    /// clear of a contact tie. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 本模块 `epa_penetration_3d` 的分发与回读。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[EpaPenetration3dQuery]) -> Vec<GpuEpa> {
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
            label: Some("prism_volumetric_epa_penetration_3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_bind_group"),
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
            label: Some("prism_volumetric_epa_penetration_3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_epa_penetration_3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_epa_penetration_3d_pass"),
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
