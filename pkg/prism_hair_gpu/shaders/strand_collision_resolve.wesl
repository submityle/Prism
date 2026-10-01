// Strand body-collision resolve twin: evaluates on-device the same in-place
// collider sweep the CPU golden `resolve_strand_collisions` performs, one thread
// per guide particle folding a per-dispatch shared collider array in order.
//
// TressFX-class strand solvers keep hair off the body by projecting every free
// particle out of a small set of analytic collider proxies (spheres and
// capsules fitted to head/neck/shoulders) after each constraint sweep
// (design 6.2). The CPU golden `resolve_strand_collisions` (in
// `prism_render_architecture::hair::collision`) visits each particle and, for a
// free particle, applies every collider in order via `Collider::push_out`; the
// last collider to push wins for that particle. A pinned particle
// (`inverse_mass <= 0`, the skinned root) is never moved, and an empty collider
// set is a no-op.
//
// This differs from the sibling `collision.wesl` twin, which runs one thread per
// (point, collider) query with no fold. Here each thread owns one particle and
// sequentially folds the shared colliders, reproducing the real per-pass body
// solve: `pos_0 = position; pos_{c+1} = colliders[c].push_out(pos_c)`.
//
// The per-collider push-out is reproduced bit-for-bit from the golden: a
// non-positive radius is inert, an already-exterior point is untouched, a point
// coincident with a sphere center escapes along +Y, and the capsule segment
// projection parameter is clamped to [0, 1]. The kernel uses only the portable
// core-WGSL subset (sqrt/min/max/clamp/dot/mul/add) - no exp, no pow, no
// optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard analytic sphere/capsule collider push-out plus an
// in-order fold; no Unreal Engine source or derived code.

// Matches `collision::EPS_LEN_SQ` (and the dynamics solver's epsilon): vectors
// shorter than the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;

// Collider kind discriminants, matching the CPU upload encoding.
const KIND_SPHERE: u32 = 0u;
const KIND_CAPSULE: u32 = 1u;

struct Params {
    // Number of particles in `particles` / points of 3 f32 in `values`.
    particle_count: u32,
    // Number of shared colliders folded per particle.
    collider_count: u32,
    pad0: u32,
    pad1: u32,
}

// One shared analytic collider. `32`-byte stride matching the `bytemuck` upload
// struct `GpuCollider`. For a sphere `center_a.xyz` is the center and
// `center_a.w` the radius, with `b_kind.w == 0`. For a capsule `center_a.xyz`
// is endpoint `a`, `b_kind.xyz` endpoint `b`, `center_a.w` the radius and
// `b_kind.w == 1`.
struct Collider {
    center_a: vec4<f32>,
    b_kind: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> colliders: array<Collider>;
// Each particle is (position.xyz, inverse_mass).
@group(0) @binding(2) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;

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

// Projects `point` out of the collider `col` encodes, mirroring
// `Collider::push_out`.
fn push_out(col: Collider, point: vec3<f32>) -> vec3<f32> {
    let radius = col.center_a.w;
    let kind = u32(col.b_kind.w);
    if (kind == KIND_CAPSULE) {
        let a = col.center_a.xyz;
        let b = col.b_kind.xyz;
        let closest = closest_point_on_segment(a, b, point);
        return push_out_sphere(closest, radius, point);
    }
    let center = col.center_a.xyz;
    return push_out_sphere(center, radius, point);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let p = particles[i];
    let base = 3u * i;
    var pos = vec3<f32>(p.x, p.y, p.z);
    // Pinned particles (inverse_mass <= 0) are never moved.
    if (p.w <= 0.0) {
        values[base] = pos.x;
        values[base + 1u] = pos.y;
        values[base + 2u] = pos.z;
        return;
    }
    // Fold every collider in order; the last to push wins for this particle.
    for (var c = 0u; c < params.collider_count; c = c + 1u) {
        pos = push_out(colliders[c], pos);
    }
    values[base] = pos.x;
    values[base + 1u] = pos.y;
    values[base + 2u] = pos.z;
}
