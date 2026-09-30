// Signed-distance-field (SDF) body-collision push-out twin: evaluates on-device
// the same analytic union-field projection the CPU golden `push_out_of_field`
// computes, one thread per query point.
//
// Analytic sphere/capsule proxies keep hair off the round parts of a body, but
// a jaw line, a collarbone or a prop against the scalp needs a tighter fit.
// Production strand solvers (AMD TressFX 4) add an SDF collider: the body is a
// field that is negative inside the surface and positive outside, and any
// particle that lands inside is pushed back along the field gradient to the
// zero isosurface. The CPU golden for that projection is `push_out_of_field`
// (in `prism_render_architecture::hair::sdf_collision`); this kernel mirrors it
// so a passing real-device parity test is direct evidence the ported kernel
// resolves the same positions as the reference, not merely that it compiles.
//
// The union is a slice of primitives (sphere, capsule, half-space, box); the
// union distance is the minimum over them, and each pass steps a still-inside
// point along the central-difference gradient of that union field by the
// current penetration depth, landing it on the nearest zero isosurface.
// Repeating settles points caught where primitives overlap. A point on a
// zero-gradient spot escapes along +Y by the penetration depth, the same
// deterministic, NaN-free fallback the golden uses.
//
// It uses only the portable core-WGSL subset (sqrt via length, min/max/clamp,
// dot, multiply/add) - no exp, no pow, no optional feature - so this twin runs
// unmodified on Metal, Vulkan and DX12. It is not bit-exact with the scalar CPU
// reference: a GPU may fuse a multiply-add or reassociate the union reduction,
// perturbing the low mantissa bits, so the parity test asserts a per-component
// tolerance rather than exact equality.
//
// Provenance: standard analytic SDF union (Quilez box/capsule distance) plus
// gradient push-out; no Unreal Engine source or derived code.

struct Params {
    // Number of query points in `points_in` / `points_out`.
    point_count: u32,
    // Number of primitives in `kinds` / `prims`.
    prim_count: u32,
    // Push-out relaxation passes.
    iterations: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-primitive kind tag: 0 sphere, 1 capsule, 2 half-space, 3 box.
@group(0) @binding(1) var<storage, read> kinds: array<u32>;
// Interleaved per-primitive parameters, stride 7 f32:
//   v0 = [0..2]  sphere center / capsule a / half-space normal / box center
//   s0 = [3]     sphere radius / capsule radius / half-space offset
//   v1 = [4..6]  capsule b / box half extents
struct Prim {
    v0: vec3<f32>,
    s0: f32,
    v1: vec3<f32>,
}
@group(0) @binding(2) var<storage, read> prims: array<f32>;
// Query points in, stride 3 f32 (xyz).
@group(0) @binding(3) var<storage, read> points_in: array<f32>;
// Resolved points out, stride 3 f32 (xyz).
@group(0) @binding(4) var<storage, read_write> points_out: array<f32>;

// Matches the dynamics solver's zero-length epsilon and the golden constants.
const EPS_LEN_SQ: f32 = 1.0e-24;
const GRAD_STEP: f32 = 1.0e-3;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn load_prim(i: u32) -> Prim {
    let b = 7u * i;
    var p: Prim;
    p.v0 = vec3<f32>(prims[b], prims[b + 1u], prims[b + 2u]);
    p.s0 = prims[b + 3u];
    p.v1 = vec3<f32>(prims[b + 4u], prims[b + 5u], prims[b + 6u]);
    return p;
}

// Closest point on segment a..b to p; a zero-length segment collapses to a.
fn closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if (len_sq <= EPS_LEN_SQ) {
        return a;
    }
    let t = clamp(dot(p - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Exact signed distance to an axis-aligned box (Quilez form).
fn box_signed_distance(center: vec3<f32>, half_extents: vec3<f32>, p: vec3<f32>) -> f32 {
    if (half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0) {
        return inf();
    }
    let rel = p - center;
    let dx = abs(rel.x) - half_extents.x;
    let dy = abs(rel.y) - half_extents.y;
    let dz = abs(rel.z) - half_extents.z;
    let ox = max(dx, 0.0);
    let oy = max(dy, 0.0);
    let oz = max(dz, 0.0);
    let outside = sqrt(ox * ox + oy * oy + oz * oz);
    let inside = min(max(dx, max(dy, dz)), 0.0);
    return outside + inside;
}

// Signed distance from p to primitive i (negative inside, +inf if inert).
fn prim_signed_distance(i: u32, p: vec3<f32>) -> f32 {
    let kind = kinds[i];
    let prim = load_prim(i);
    if (kind == 0u) {
        // Sphere: v0 = center, s0 = radius.
        if (prim.s0 <= 0.0) {
            return inf();
        }
        return length(p - prim.v0) - prim.s0;
    } else if (kind == 1u) {
        // Capsule: v0 = a, v1 = b, s0 = radius.
        if (prim.s0 <= 0.0) {
            return inf();
        }
        let closest = closest_point_on_segment(prim.v0, prim.v1, p);
        return length(p - closest) - prim.s0;
    } else if (kind == 2u) {
        // Half-space: v0 = normal, s0 = offset.
        let len_sq = dot(prim.v0, prim.v0);
        if (len_sq <= EPS_LEN_SQ) {
            return inf();
        }
        let inv_len = 1.0 / sqrt(len_sq);
        return dot(p, prim.v0) * inv_len - prim.s0;
    }
    // Box: v0 = center, v1 = half extents.
    return box_signed_distance(prim.v0, prim.v1, p);
}

// Union signed distance: minimum over all primitives (+inf for an empty union).
fn union_signed_distance(p: vec3<f32>) -> f32 {
    var d = inf();
    for (var i: u32 = 0u; i < params.prim_count; i = i + 1u) {
        let di = prim_signed_distance(i, p);
        if (di < d) {
            d = di;
        }
    }
    return d;
}

// Outward unit gradient of the union field by central differences, or zero when
// the field is locally flat (so the caller can fall back to a fixed escape).
fn union_gradient(p: vec3<f32>) -> vec3<f32> {
    let gx = union_signed_distance(vec3<f32>(p.x + GRAD_STEP, p.y, p.z))
        - union_signed_distance(vec3<f32>(p.x - GRAD_STEP, p.y, p.z));
    let gy = union_signed_distance(vec3<f32>(p.x, p.y + GRAD_STEP, p.z))
        - union_signed_distance(vec3<f32>(p.x, p.y - GRAD_STEP, p.z));
    let gz = union_signed_distance(vec3<f32>(p.x, p.y, p.z + GRAD_STEP))
        - union_signed_distance(vec3<f32>(p.x, p.y, p.z - GRAD_STEP));
    let g = vec3<f32>(gx, gy, gz);
    let len_sq = dot(g, g);
    if (len_sq > EPS_LEN_SQ) {
        return g * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Pushes p out of the union field until outside every primitive or the pass cap
// is spent. A NaN cannot arise for finite inputs, so the golden's `d.is_nan()`
// break is not reachable here; the `d >= 0` break matches the reference.
fn push_out(point: vec3<f32>) -> vec3<f32> {
    var p = point;
    for (var it: u32 = 0u; it < params.iterations; it = it + 1u) {
        let d = union_signed_distance(p);
        if (d >= 0.0) {
            break;
        }
        let grad = union_gradient(p);
        if (dot(grad, grad) <= EPS_LEN_SQ) {
            // Locally flat field: escape along a fixed axis by the depth.
            p = p + vec3<f32>(0.0, -d, 0.0);
            break;
        }
        p = p + grad * (-d);
    }
    return p;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.point_count) {
        return;
    }
    let b = 3u * idx;
    let p = vec3<f32>(points_in[b], points_in[b + 1u], points_in[b + 2u]);
    let out = push_out(p);
    points_out[b] = out.x;
    points_out[b + 1u] = out.y;
    points_out[b + 2u] = out.z;
}
