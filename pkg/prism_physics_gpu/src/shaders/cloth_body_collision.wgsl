// Body-proxy collision and per-particle backstops, the real-device twin of
// `prism_physics_core`'s `resolve_body_collisions_with_friction` and
// `resolve_backstops`.
//
// A garment is simulated against a cheap proxy of the animated body: a small
// set of analytic colliders (sphere, capsule, half-space) plus optional
// per-particle backstop planes. Both tiers are position-level projections run
// after the internal constraints: a particle inside a collider is pushed to its
// surface (and its tangential slide rubbed with Coulomb friction), and a
// particle that has sunk behind its backstop is pushed back onto the limiting
// plane.
//
// This pass is *per-particle independent*: one thread owns one particle and
// walks every collider in slice order (last-collider-to-push wins), exactly
// mirroring the sequential golden's per-particle write set, so there is no
// colouring and no cross-thread contention. The only float divergence from the
// CPU is a few ULP in `normalize`/`inverseSqrt`, so parity is checked within a
// tight relative tolerance.
//
// Provenance: the analytic body-proxy projections and the one-sided backstop
// are standard position-based collision techniques; the tangential-friction
// projection is Macklin et al. (2014), "Unified Particle Physics for Real-Time
// Applications". No Unreal Engine source or derived code.

const EPS_LEN_SQ: f32 = 1.0e-12;
const EPS_FRICTION: f32 = 1.0e-12;

const COLLIDER_SPHERE: u32 = 0u;
const COLLIDER_CAPSULE: u32 = 1u;
const COLLIDER_HALF_SPACE: u32 = 2u;
const COLLIDER_OBB: u32 = 3u;
const COLLIDER_CONVEX: u32 = 4u;

const WORKGROUP: u32 = 64u;

struct Params {
    // Number of addressable particles (positions length).
    particle_count: u32,
    // Number of body colliders in the slice.
    collider_count: u32,
    // Number of per-particle backstop planes.
    backstop_count: u32,
    // Combined Coulomb friction coefficient (already clamped to 0..=1 by host).
    mu: f32,
};

// kind: COLLIDER_* discriminant.
// radius: sphere/capsule radius, or half-space offset.
// p0: sphere center / capsule endpoint 0 / half-space normal / box center
//     (w unused).
// p1: capsule endpoint 1 / box half-extents (unused for sphere/half-space).
// p2: box orientation quaternion (x, y, z, w); unused for the other primitives.
struct Collider {
    kind: u32,
    radius: f32,
    pad0: u32,
    pad1: u32,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
};

// origin: xyz anchor point, w max behind-plane distance.
// normal: xyz outward plane normal (need not be unit; w unused).
struct Backstop {
    origin: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(3) var<storage, read> prev_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> colliders: array<Collider>;
@group(0) @binding(5) var<storage, read> backstops: array<Backstop>;
// One face plane of a convex-hull body collider: xyz = outward unit normal,
// w = plane offset so the face is `normal.dot(x) == w`. A COLLIDER_CONVEX
// record indexes a contiguous run here via `pad0` (plane_offset) and `pad1`
// (plane_count); see `GpuConvexPlane` / `pack_body_scene` on the host.
struct ConvexPlane {
    plane: vec4<f32>,
};

@group(0) @binding(6) var<storage, read> convex_planes: array<ConvexPlane>;

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

// Rotates `v` by the unit quaternion `q` (xyzw). Scalar Hamilton form:
// v' = v*(w*w - dot(b,b)) + b*(2*dot(v,b)) + cross(b,v)*(2*w), b = q.xyz.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let w = q.w;
    return v * (w * w - dot(b, b)) + b * (2.0 * dot(v, b)) + cross(b, v) * (2.0 * w);
}

// Rotates `v` by the inverse (conjugate, negated xyz) of the unit quaternion
// `q`, i.e. maps world -> box-local.
fn quat_rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return quat_rotate(vec4<f32>(-q.xyz, q.w), v);
}

// Projects `pos` out to the nearest face of the oriented box
// `(center, orientation, half_extents)` when it lies strictly inside, otherwise
// returns `pos` unchanged. Mirrors `project_out_of_obb` in prism_physics_core.
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

// Projects `pos` out to the nearest face of the convex solid whose face
// planes occupy `convex_planes[plane_offset .. plane_offset + plane_count]`
// when it lies strictly inside every face, otherwise returns `pos`. This is
// the real-device twin of `ConvexProxy::project_out` in prism_physics_core:
// a point interior to a convex polytope is closest to its least-penetrating
// face, and any face with a non-negative signed distance proves the point is
// outside the solid. Degenerate (near-zero-normal) faces never constrain the
// solid, and an empty run is inert.
fn project_out_of_convex(pos: vec3<f32>, plane_offset: u32, plane_count: u32) -> vec3<f32> {
    var best_pen = 0.0;
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var found = false;
    for (var i = 0u; i < plane_count; i = i + 1u) {
        let plane = convex_planes[plane_offset + i].plane;
        let normal = plane.xyz;
        if (dot(normal, normal) <= EPS_LEN_SQ) {
            continue;
        }
        let signed = dot(normal, pos) - plane.w;
        if (signed >= 0.0) {
            // Outside (or exactly on) this face => outside the convex solid.
            return pos;
        }
        let pen = -signed;
        if (!found || pen < best_pen) {
            best_pen = pen;
            best_normal = normal;
            found = true;
        }
    }
    if (!found) {
        return pos;
    }
    // Unit face normals => stepping `best_pen` lands on the shallowest face.
    return pos + best_normal * best_pen;
}

// Dispatches one collider's projection by discriminant.
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
    if (c.kind == COLLIDER_CONVEX) {
        // `pad0`/`pad1` carry the convex plane-run offset/count.
        return project_out_of_convex(pos, c.pad0, c.pad1);
    }
    // COLLIDER_HALF_SPACE
    return project_out_of_half_space(pos, c.p0.xyz, c.radius);
}

// Position-level Coulomb friction against a contact whose outward unit `normal`
// and normal-correction magnitude `normal_push` are known. The tangential slide
// over the frame is cancelled inside the static cone and otherwise shrunk by
// exactly `mu * normal_push`, direction preserved. A non-positive `mu`,
// non-positive push, or (near) zero tangential slide is a no-op.
fn apply_coulomb_friction(
    pos: vec3<f32>,
    prev: vec3<f32>,
    normal: vec3<f32>,
    normal_push: f32,
    mu: f32,
) -> vec3<f32> {
    if (mu <= 0.0 || normal_push <= 0.0) {
        return pos;
    }
    let delta = pos - prev;
    let normal_amount = dot(delta, normal);
    let tangent = delta - normal * normal_amount;
    let tan_len_sq = dot(tangent, tangent);
    if (tan_len_sq <= EPS_FRICTION) {
        return pos;
    }
    let tan_len = sqrt(tan_len_sq);
    let scale = min(mu * normal_push / tan_len, 1.0);
    return pos - tangent * scale;
}

// Clamps `pos` to the front side of `backstop`. When the signed distance drops
// below `-distance` the point is pushed forward along the unit normal onto the
// limiting plane; otherwise it is returned unchanged. A (near) zero normal is
// inert.
fn apply_backstop(pos: vec3<f32>, b: Backstop) -> vec3<f32> {
    let normal = b.normal.xyz;
    let len_sq = dot(normal, normal);
    if (len_sq <= EPS_LEN_SQ) {
        return pos;
    }
    let n = normal * inverseSqrt(len_sq);
    let s = dot(n, pos - b.origin.xyz);
    let min_s = -b.origin.w;
    if (s < min_s) {
        return pos + n * (min_s - s);
    }
    return pos;
}

@compute @workgroup_size(64)
fn body_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.particle_count) {
        return;
    }
    let w = inv_mass[v];
    if (w <= 0.0) {
        return;
    }
    var pos = positions[v].xyz;
    let prev = prev_positions[v].xyz;
    for (var c = 0u; c < params.collider_count; c = c + 1u) {
        let before = pos;
        let projected = project_collider(colliders[c], before);
        let correction = projected - before;
        let push_sq = dot(correction, correction);
        if (push_sq <= EPS_LEN_SQ) {
            // Already outside this collider: no contact, no friction.
            pos = projected;
            continue;
        }
        let push = sqrt(push_sq);
        let n = correction * (1.0 / push);
        pos = apply_coulomb_friction(projected, prev, n, push, params.mu);
    }
    positions[v] = vec4<f32>(pos, positions[v].w);
}

@compute @workgroup_size(64)
fn backstop_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    // The golden runs over the shorter of the position and backstop lengths.
    if (v >= params.particle_count || v >= params.backstop_count) {
        return;
    }
    let w = inv_mass[v];
    if (w <= 0.0) {
        return;
    }
    let pos = positions[v].xyz;
    let out = apply_backstop(pos, backstops[v]);
    positions[v] = vec4<f32>(out, positions[v].w);
}
