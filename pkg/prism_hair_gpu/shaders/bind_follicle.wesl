// Follicle-to-scalp binding twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::follicle_bind::bind_follicle`. Binding a hair
// root to a scalp triangle captures two things: the (sanitised, on-surface)
// barycentric position of the root within the triangle, and the signed height of
// the root above the surface measured along the interpolated surface normal — so
// a root that floats slightly off the skin stays at that height after the scalp
// deforms. This kernel broadcasts one shared triangle (vertex positions and
// per-vertex normals) to a whole batch of roots, one thread per root, and emits
// that root's binding (u, v, w, normal_offset).
//
// It composes three stages the sibling twins each isolate: the raw planar
// barycentric solve (as in the `compute_barycentric` twin), followed by the
// `sanitized` clamp/renormalise that forces the weights on-surface, followed by
// a surface-point and interpolated-normal reconstruction to project the signed
// offset. Unlike the `compute_barycentric` twin it returns *sanitised* weights
// (non-negative, summing to one), and unlike the `follicle_bind` root-transfer
// twin — which consumes a finished binding to rebuild a world position — this is
// the earlier capture step that *produces* the binding.
//
// Buffer layout: `params` carries the one shared triangle (three vertex
// positions and three per-vertex normals as `vec4`, xyz used) plus the root
// count; `roots[i]` is query root `i` (xyz in a `vec4`, w unused); `out[i]`
// receives its binding (sanitised u, v, w in xyz and the signed normal offset in
// the w lane).
//
// Degenerate (zero-area / non-finite determinant) triangles make the raw solve
// fall back to the centroid; an all-non-positive (or non-finite) sanitised
// weight set likewise collapses to the centroid; a zero-length interpolated
// normal falls back to +Z; a non-finite offset is clamped to zero — each exactly
// as the golden does, so a bad triangle or root never panics. An empty batch
// dispatches nothing.
//
// The solve, barycentric folds and offset projection are dot/determinant chains
// plus a reciprocal, a normalising `sqrt`-reciprocal and multiply-adds a GPU may
// fuse, so the twin is matched against a tolerance (abs_diff < 1e-4 or
// rel_diff < 1e-3) rather than bit-for-bit. The kernel uses only `dot`, compare,
// `abs`, `sqrt`, reciprocal and multiply/add in the portable core-WGSL subset —
// no `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: Ericson "Real-Time Collision Detection" barycentric solve plus
// standard mesh-skinning attachment; no Unreal Engine source or derived code.

// Zero-area / non-finite / zero-length guard, mirroring
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
    // Shared triangle per-vertex normals (xyz used, w padding).
    nrm0: vec4<f32>,
    nrm1: vec4<f32>,
    nrm2: vec4<f32>,
    // Number of query roots, bounding the dispatch.
    root_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per query, one root to bind to the triangle (xyz in a vec4, w unused).
@group(0) @binding(1) var<storage, read> roots: array<vec4<f32>>;
// Per query, the binding: sanitised (u, v, w) in xyz, signed normal offset in w.
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

// `f32::is_finite` port: rejects NaN (self-inequality) and ±inf (magnitude
// above the largest finite value).
fn is_finite_scalar(x: f32) -> bool {
    return x == x && abs(x) <= F32_MAX;
}

// `normalize_or`: unit vector, or `fallback` when near zero-length / non-finite.
fn normalize_or(a: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(a, a));
    if (len > DEGENERATE_EPS && is_finite_scalar(len)) {
        return a * (1.0 / len);
    }
    return fallback;
}

// `Barycentric::sanitized`: clamp negatives/non-finite to zero and renormalise,
// or fall back to the centroid when the clamped set is (near) all-zero.
fn clamp0(x: f32) -> f32 {
    if (is_finite_scalar(x) && x > 0.0) {
        return x;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.root_count) {
        return;
    }

    let a = params.pos0.xyz;
    let b = params.pos1.xyz;
    let c = params.pos2.xyz;
    let root = roots[i].xyz;

    // Raw planar-projection barycentric solve (Ericson), centroid on degeneracy.
    let v0 = b - a;
    let v1 = c - a;
    let v2 = root - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;

    let third = 1.0 / 3.0;
    var ru = third;
    var rv = third;
    var rw = third;
    if (abs(denom) > DEGENERATE_EPS && is_finite_scalar(denom)) {
        let inv = 1.0 / denom;
        let vv = (d11 * d20 - d01 * d21) * inv;
        let ww = (d00 * d21 - d01 * d20) * inv;
        ru = 1.0 - vv - ww;
        rv = vv;
        rw = ww;
    }

    // Sanitise to an on-surface weighting (non-negative, sums to one).
    var su = clamp0(ru);
    var sv = clamp0(rv);
    var sw = clamp0(rw);
    let sum = su + sv + sw;
    if (sum > DEGENERATE_EPS) {
        let inv_sum = 1.0 / sum;
        su = su * inv_sum;
        sv = sv * inv_sum;
        sw = sw * inv_sum;
    } else {
        su = third;
        sv = third;
        sw = third;
    }

    // Surface point and interpolated normal under the sanitised weights.
    let surface = params.pos0.xyz * su + params.pos1.xyz * sv + params.pos2.xyz * sw;
    let raw_normal = params.nrm0.xyz * su + params.nrm1.xyz * sv + params.nrm2.xyz * sw;
    let normal = normalize_or(raw_normal, vec3<f32>(0.0, 0.0, 1.0));

    // Signed height of the root above the surface along the normal.
    let offset = dot(root - surface, normal);
    var normal_offset = offset;
    if (!is_finite_scalar(offset)) {
        normal_offset = 0.0;
    }

    out[i] = vec4<f32>(su, sv, sw, normal_offset);
}
