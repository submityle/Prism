// Strand body-collider projection twin: evaluates on-device the same analytic
// push-out the CPU golden `Collider::push_out` computes, one thread per query.
//
// TressFX-class strand solvers keep hair off the body by projecting each free
// particle out of a small set of analytic collider proxies (spheres and
// capsules fitted to head/neck/shoulders) after every constraint sweep
// (design 6.2). The CPU golden for that projection is `Collider::push_out` (in
// `prism_render_architecture::hair::collision`); this kernel mirrors it
// exactly:
//
//   sphere:  push_out_sphere(center, radius, point)
//   capsule: push_out_sphere(closest_point_on_segment(a, b, point), radius, point)
//
// Every guard is reproduced bit-for-bit: a non-positive radius is inert, a
// point already outside is untouched, a point coincident with the center
// escapes along +Y, and the segment projection parameter is clamped to [0, 1].
// It uses only the portable core-WGSL subset (sqrt/min/max/clamp/dot/mul/add) -
// no exp, no pow, no optional feature - so this twin runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard analytic sphere/capsule collider push-out; no Unreal
// Engine source or derived code.

// Matches `collision::EPS_LEN_SQ` (and the dynamics solver's epsilon): vectors
// shorter than the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;

// Collider kind discriminants, matching the CPU upload encoding.
const KIND_SPHERE: u32 = 0u;
const KIND_CAPSULE: u32 = 1u;

struct Params {
    // Number of valid entries in `queries` / points in `values`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One collider-projection query. `48`-byte stride matching the `bytemuck`
// upload struct `CollisionQuery`.
struct Query {
    px: f32,
    py: f32,
    pz: f32,
    kind: u32,
    ax: f32,
    ay: f32,
    az: f32,
    radius: f32,
    bx: f32,
    by: f32,
    bz: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero,
// mirroring `Vec3::normalize_or_zero` term order (`v * (1 / sqrt(len_sq))`).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Pushes `point` out to the surface of the sphere `(center, radius)`, mirroring
// `collision::push_out_sphere`. A non-positive radius or an already-exterior
// point is returned unchanged; a point coincident with the center escapes along
// `+Y`.
fn push_out_sphere(center: vec3<f32>, radius: f32, point: vec3<f32>) -> vec3<f32> {
    if (radius <= 0.0) {
        return point;
    }
    let delta = point - center;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= radius * radius) {
        return point;
    }
    if (dist_sq <= EPS_LEN_SQ) {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = normalize_or_zero(delta);
    return center + dir * radius;
}

// Point on segment `a`..`b` closest to `point`, mirroring
// `collision::closest_point_on_segment`. A zero-length segment returns `a`.
fn closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if (len_sq <= EPS_LEN_SQ) {
        return a;
    }
    let t = clamp(dot(point - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Projects `point` out of the collider `q` encodes, mirroring
// `Collider::push_out`.
fn push_out(q: Query) -> vec3<f32> {
    let point = vec3<f32>(q.px, q.py, q.pz);
    if (q.kind == KIND_CAPSULE) {
        let a = vec3<f32>(q.ax, q.ay, q.az);
        let b = vec3<f32>(q.bx, q.by, q.bz);
        let closest = closest_point_on_segment(a, b, point);
        return push_out_sphere(closest, q.radius, point);
    }
    let center = vec3<f32>(q.ax, q.ay, q.az);
    return push_out_sphere(center, q.radius, point);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let out = push_out(queries[i]);
    let base = 3u * i;
    values[base] = out.x;
    values[base + 1u] = out.y;
    values[base + 2u] = out.z;
}
