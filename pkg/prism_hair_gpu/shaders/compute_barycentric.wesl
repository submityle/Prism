// Planar-projection barycentric-coordinate twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::follicle_bind::compute_barycentric`. Binding
// a hair root to a scalp triangle starts by projecting the root onto the plane
// of its triangle and solving for the barycentric weights (u, v, w) that locate
// it there (Ericson's method). This kernel broadcasts one shared triangle to a
// whole batch of query points, one thread per point, and emits that point's
// (raw, un-sanitised) barycentric weights — the array-in/array-out form the
// follicle-binding stage consumes when attaching many roots to one scalp face.
//
// It is a distinct sibling of the `closest_point_on_triangle` twin: that one
// clamps the query into the triangle's region (vertex/edge/interior Voronoi
// test) and returns the nearest *point*; this one solves the planar projection
// linear system and returns the (possibly out-of-triangle) *weights* without
// clamping — a different algorithm, a different output. It also differs from the
// `follicle_bind` root-transfer twin, which consumes already-sanitised weights
// to reconstruct a world position; this is the earlier solve that produces the
// raw weights.
//
// Buffer layout: `params` carries the one shared triangle (three vertex
// positions as `vec4`, xyz used) plus the point count; `points[i]` is query `i`
// (xyz in a `vec4`, w unused); `out[i]` receives its barycentric weights
// (u, v, w in a `vec4`, w-lane unused).
//
// Degenerate (zero-area, or non-finite determinant) triangles fall back to the
// triangle centroid (1/3, 1/3, 1/3) exactly as the golden does, rather than
// dividing by zero, so a bad triangle never panics. Because the triangle is
// shared, that branch is taken for the whole batch or none of it. An empty batch
// dispatches nothing.
//
// The solve is a dot-product/determinant chain plus a single reciprocal and
// multiply-adds a GPU may fuse, so the twin is matched against a tolerance
// (abs_diff < 1e-4 or rel_diff < 1e-3) rather than bit-for-bit. The kernel uses
// only `dot`, compare, `abs`, a reciprocal and multiply/add in the portable
// core-WGSL subset - no `exp`, `pow`, `sqrt` or optional device feature - so the
// twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: Ericson "Real-Time Collision Detection" barycentric solve plus
// wgpu compute dispatch; no Unreal Engine source or derived code.

// Zero-area / non-finite determinant guard, mirroring
// `follicle_bind::DEGENERATE_EPS`.
const DEGENERATE_EPS: f32 = 1.0e-12;
// Largest finite `f32`; used with the self-equality test to reject NaN/inf the
// way the golden's `is_finite` checks do.
const F32_MAX: f32 = 3.4028235e38;

struct Params {
    // Shared triangle vertex positions (xyz used, w padding).
    pos0: vec4<f32>,
    pos1: vec4<f32>,
    pos2: vec4<f32>,
    // Number of query points, bounding the dispatch.
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per query, one point to project onto the triangle (xyz in a vec4, w unused).
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
// Per query, the barycentric weights (u, v, w in a vec4, w-lane unused).
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

// `f32::is_finite` port: rejects NaN (self-inequality) and ±inf (magnitude
// above the largest finite value).
fn is_finite_scalar(x: f32) -> bool {
    return x == x && abs(x) <= F32_MAX;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let a = params.pos0.xyz;
    let b = params.pos1.xyz;
    let c = params.pos2.xyz;
    let p = points[i].xyz;

    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;

    // Degenerate / non-finite determinant -> triangle centroid, matching the
    // golden's `Barycentric::CENTROID` fallback (no divide by zero).
    let third = 1.0 / 3.0;
    var ou = third;
    var ov = third;
    var ow = third;
    if (abs(denom) > DEGENERATE_EPS && is_finite_scalar(denom)) {
        let inv = 1.0 / denom;
        let v = (d11 * d20 - d01 * d21) * inv;
        let w = (d00 * d21 - d01 * d20) * inv;
        ou = 1.0 - v - w;
        ov = v;
        ow = w;
    }

    out[i] = vec4<f32>(ou, ov, ow, 0.0);
}
