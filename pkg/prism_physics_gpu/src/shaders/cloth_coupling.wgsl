// Two-way rigid coupling between cloth particles and a rigid proxy, the
// real-device twin of `prism_physics_core`'s `resolve_two_way_coupling`.
//
// One-way body collision only pushes particles out of a proxy; this pass closes
// the loop so a light prop resting on cloth dents it and the cloth pushes back.
// The authoritative rigid-body integrator is a separate concern, so this kernel
// contributes only the contact half: a mass-weighted split of each push-out
// between the particle and the body, plus the Newton reaction impulse the body
// accrues.
//
// The host drives one dispatch per body (bodies are processed in slice order,
// each seeing the particle positions after the previous body's corrections,
// exactly as the sequential golden does). Within a dispatch the pass is Jacobi
// and *per-particle independent*: thread `i` reads `positions[i]`, writes back
// its own mass-weighted share of the push-out, and emits its `body_delta` and
// `impulse` contributions to per-particle output slots. The host then sums
// those contributions in index order (matching the golden's accumulation
// order), translates the body once, and accumulates its reaction impulse.
// Keeping the per-body reduction on the host preserves the golden's exact
// summation order, so parity holds within a tight relative tolerance (the only
// divergence is a few ULP in `inverseSqrt`/division inside the projection).
//
// Provenance: the inverse-mass-weighted contact split and the Newton reaction
// impulse are textbook position-based-dynamics / rigid-body contact mechanics.
// No Unreal Engine source or derived code.

const EPS_LEN_SQ: f32 = 1.0e-12;

const COLLIDER_SPHERE: u32 = 0u;
const COLLIDER_CAPSULE: u32 = 1u;
const COLLIDER_HALF_SPACE: u32 = 2u;
const COLLIDER_OBB: u32 = 3u;

// kind: COLLIDER_* discriminant.
// radius: sphere/capsule radius, or half-space offset.
// p0: sphere center / capsule endpoint 0 / half-space normal / box center
//     (w unused).
// p1: capsule endpoint 1 / box half-extents (unused for sphere/half-space).
// p2: box orientation quaternion (x, y, z, w); unused for other primitives.
struct Collider {
    kind: u32,
    radius: f32,
    pad0: u32,
    pad1: u32,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
};

struct Params {
    // The body's current pose for this dispatch.
    collider: Collider,
    // Number of addressable particles (positions length).
    particle_count: u32,
    // The body's (already non-negative) inverse mass.
    w_body: f32,
    // The substep.
    dt: f32,
    // Padding to a 16-byte word.
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> body_delta: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> impulse: array<vec4<f32>>;

// Projects `pos` out to the surface of the sphere `(center, radius)`.
//
// A non-positive radius leaves the point untouched. When `pos` coincides with
// `center` there is no defined radial direction, so the point is nudged out
// along `+Y`, a fixed, deterministic, non-`NaN` fallback.
fn project_out_of_sphere(pos: vec3<f32>, center: vec3<f32>, radius: f32) -> vec3<f32> {
    if (radius <= 0.0) {
        return pos;
    }
    let delta = pos - center;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= radius * radius) {
        return pos;
    }
    if (dist_sq <= EPS_LEN_SQ) {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = delta * inverseSqrt(dist_sq);
    return center + dir * radius;
}

// Returns the point on segment `p0`..`p1` closest to `pos`, clamped to the
// segment so the capsule caps are hemispheres. A zero-length segment
// degenerates to `p0`.
fn closest_point_on_segment(p0: vec3<f32>, p1: vec3<f32>, pos: vec3<f32>) -> vec3<f32> {
    let axis = p1 - p0;
    let len_sq = dot(axis, axis);
    if (len_sq <= EPS_LEN_SQ) {
        return p0;
    }
    let t = clamp(dot(pos - p0, axis) / len_sq, 0.0, 1.0);
    return p0 + axis * t;
}

// Projects `pos` onto the half-space plane `normal.dot(x) == offset` when it
// lies on the infeasible side, otherwise returns `pos`. The normal need not be
// unit length; a (near) zero normal leaves the point untouched.
fn project_out_of_half_space(pos: vec3<f32>, normal: vec3<f32>, offset: f32) -> vec3<f32> {
    let len_sq = dot(normal, normal);
    if (len_sq <= EPS_LEN_SQ) {
        return pos;
    }
    let signed = dot(normal, pos) - offset;
    if (signed >= 0.0) {
        return pos;
    }
    let t = -signed / len_sq;
    return pos + normal * t;
}

// Dispatches one collider's projection by discriminant.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let w = q.w;
    return v * (w * w - dot(b, b)) + b * (2.0 * dot(v, b)) + cross(b, v) * (2.0 * w);
}

fn quat_rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return quat_rotate(vec4<f32>(-q.xyz, q.w), v);
}

// Projects `pos` out to the nearest face of the oriented box when strictly
// inside, otherwise returns `pos`. Mirrors `project_out_of_obb` in
// prism_physics_core.
fn project_out_of_obb(
    pos: vec3<f32>,
    center: vec3<f32>,
    orientation: vec4<f32>,
    half_extents: vec3<f32>,
) -> vec3<f32> {
    if (half_extents.x <= 0.0 && half_extents.y <= 0.0 && half_extents.z <= 0.0) {
        return pos;
    }
    let local = quat_rotate_inv(orientation, pos - center);
    let a = abs(local);
    if (a.x >= half_extents.x || a.y >= half_extents.y || a.z >= half_extents.z) {
        return pos;
    }
    let pen = half_extents - a;
    var local_out = local;
    if (pen.x <= pen.y && pen.x <= pen.z) {
        if (local.x >= 0.0) {
            local_out.x = half_extents.x;
        } else {
            local_out.x = -half_extents.x;
        }
    } else if (pen.y <= pen.z) {
        if (local.y >= 0.0) {
            local_out.y = half_extents.y;
        } else {
            local_out.y = -half_extents.y;
        }
    } else {
        if (local.z >= 0.0) {
            local_out.z = half_extents.z;
        } else {
            local_out.z = -half_extents.z;
        }
    }
    return center + quat_rotate(orientation, local_out);
}

fn project_collider(c: Collider, pos: vec3<f32>) -> vec3<f32> {
    if (c.kind == COLLIDER_SPHERE) {
        return project_out_of_sphere(pos, c.p0.xyz, c.radius);
    }
    if (c.kind == COLLIDER_CAPSULE) {
        let closest = closest_point_on_segment(c.p0.xyz, c.p1.xyz, pos);
        return project_out_of_sphere(pos, closest, c.radius);
    }
    if (c.kind == COLLIDER_OBB) {
        return project_out_of_obb(pos, c.p0.xyz, c.p2, c.p1.xyz);
    }
    // COLLIDER_HALF_SPACE
    return project_out_of_half_space(pos, c.p0.xyz, c.radius);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let pos = positions[i].xyz;
    // Default: no contribution (no contact / degenerate / dt <= 0).
    var particle_delta = vec3<f32>(0.0, 0.0, 0.0);
    var bd = vec3<f32>(0.0, 0.0, 0.0);
    var im = vec3<f32>(0.0, 0.0, 0.0);

    let w_particle = max(inv_mass[i], 0.0);
    let w_body = max(params.w_body, 0.0);
    let w_sum = w_particle + w_body;
    if (params.dt > 0.0 && w_sum > 0.0) {
        let correction = project_collider(params.collider, pos) - pos;
        if (dot(correction, correction) > EPS_LEN_SQ) {
            // Mass-weighted split: the lighter side moves more.
            particle_delta = correction * (w_particle / w_sum);
            bd = correction * (-(w_body / w_sum));
            // Newton reaction on the body (momentum), opposing the push.
            im = correction * (-(1.0 / (w_sum * params.dt)));
        }
    }

    positions[i] = vec4<f32>(pos + particle_delta, positions[i].w);
    body_delta[i] = vec4<f32>(bd, 0.0);
    impulse[i] = vec4<f32>(im, 0.0);
}
