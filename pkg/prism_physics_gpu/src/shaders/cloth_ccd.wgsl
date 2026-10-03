// Continuous-collision (tunnelling) sweep for cloth particles against rigid
// body proxies, the real-device twin of `prism_physics_core`'s `resolve_ccd`.
//
// Each free particle's motion `prev -> curr` is swept against every collider;
// the earliest valid time of impact (TOI) wins. On a hit the particle is placed
// on the collider surface plus a skin along the outward normal, its inbound
// normal velocity is reflected by restitution, and its tangential slide is
// damped by position-level Coulomb friction (Macklin et al. 2014).
//
// The pass is *per-particle independent*: thread `i` reads a read-only previous
// position, walks every collider in slice order, and writes only its own
// `positions[i]` / `velocities[i]`, exactly the sequential golden's per-particle
// write set. The only float divergence from the CPU is a few ULP in
// `sqrt`/division, so parity is verified within a tight tolerance.
//
// Provenance: closed-form swept primitive tests and the XPBD tangential
// friction projection are standard, publicly documented techniques. No Unreal
// Engine source or derived code.

const EPS_LEN_SQ: f32 = 1.0e-12;
const EPS_COEF: f32 = 1.0e-12;
const EPS_FRICTION: f32 = 1.0e-12;
// Sentinel for "no impact in [0, 1]"; any valid TOI is <= 1, so min()-style
// reduction and the final `best <= 1.0` gate both work without an Option.
const NO_HIT: f32 = 1.0e30;

const COLLIDER_SPHERE: u32 = 0u;
const COLLIDER_CAPSULE: u32 = 1u;
const COLLIDER_HALF_SPACE: u32 = 2u;
const COLLIDER_OBB: u32 = 3u;
const COLLIDER_CONVEX: u32 = 4u;

struct Collider {
    kind: u32,
    radius: f32,
    pad0: u32,
    pad1: u32,
    // sphere center / capsule endpoint 0 / half-space normal / box center.
    p0: vec4<f32>,
    // capsule endpoint 1 / box half-extents.
    p1: vec4<f32>,
    // box orientation quaternion (x, y, z, w); unused for other primitives.
    p2: vec4<f32>,
};

struct Params {
    // Number of addressable particles (positions length).
    particle_count: u32,
    // Number of body colliders in the slice.
    collider_count: u32,
    // Sanitized skin: distance along the outward normal to place a hit particle.
    skin: f32,
    // Sanitized restitution in 0..=1.
    restitution: f32,
    // Sanitized Coulomb friction coefficient in 0..=1.
    mu: f32,
    // `1 / dt`, or 0 when |dt| is (near) zero (velocity reflection disabled).
    inv_dt: f32,
    // Padding to a 16-byte word.
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prev_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(5) var<storage, read> colliders: array<Collider>;

// One oriented face of a convex hull: `plane.xyz` is the outward unit
// normal, `plane.w` the plane offset. A convex collider's `pad0`/`pad1`
// carry the [offset, count) run of its faces inside this shared pool
// (see `pack_body_scene` on the host).
struct ConvexPlane {
    plane: vec4<f32>,
};
@group(0) @binding(6) var<storage, read> convex_planes: array<ConvexPlane>;

// Earliest root in [0, 1] of `a*t^2 + b*t + c <= 0` for non-negative `a`, or
// `NO_HIT`. A start value `c <= 0` is already inside and reports `t == 0`.
fn first_entry_time(a: f32, b: f32, c: f32) -> f32 {
    if (c <= 0.0) {
        return 0.0;
    }
    if (a <= EPS_COEF) {
        // Linear: b*t + c <= 0 with c > 0 needs b < 0.
        if (b >= -EPS_COEF) {
            return NO_HIT;
        }
        let t = -c / b;
        if (t <= 1.0) {
            return max(t, 0.0);
        }
        return NO_HIT;
    }
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return NO_HIT;
    }
    let root = sqrt(disc);
    let t = ((-b) - root) / (2.0 * a);
    if (t >= 0.0 && t <= 1.0) {
        return t;
    }
    return NO_HIT;
}

// Earliest TOI of the swept point `prev -> curr` into the sphere, or `NO_HIT`.
fn sphere_toi(prev: vec3<f32>, curr: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    if (radius <= 0.0) {
        return NO_HIT;
    }
    let m = curr - prev;
    let e = prev - center;
    let a = dot(m, m);
    let b = 2.0 * dot(e, m);
    let c = dot(e, e) - radius * radius;
    return first_entry_time(a, b, c);
}

// Earliest TOI of the swept point crossing into the infeasible half-space side,
// or `NO_HIT`.
fn half_space_toi(prev: vec3<f32>, curr: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    if (dot(normal, normal) <= EPS_LEN_SQ) {
        return NO_HIT;
    }
    let s0 = dot(normal, prev) - offset;
    if (s0 <= 0.0) {
        return 0.0;
    }
    let ds = dot(normal, curr - prev);
    if (ds >= -EPS_COEF) {
        return NO_HIT;
    }
    let t = -s0 / ds;
    if (t <= 1.0) {
        return max(t, 0.0);
    }
    return NO_HIT;
}

// Earliest TOI into the infinite cylinder about the segment axis, restricted to
// the axis slab `0..=len`, or `NO_HIT`.
fn cylinder_slab_toi(
    prev: vec3<f32>,
    curr: vec3<f32>,
    p0: vec3<f32>,
    axis: vec3<f32>,
    radius: f32,
) -> f32 {
    let len = length(axis);
    if (len <= EPS_COEF) {
        return NO_HIT;
    }
    let u = axis * (1.0 / len);
    let e0 = prev - p0;
    let m = curr - prev;
    let mu_m = dot(m, u);
    let e0u = dot(e0, u);

    // Radial interval where perpendicular distance <= radius.
    let a = dot(m, m) - mu_m * mu_m;
    let b = 2.0 * (dot(e0, m) - e0u * mu_m);
    let c = dot(e0, e0) - e0u * e0u - radius * radius;
    var rad_lo: f32;
    var rad_hi: f32;
    if (a > EPS_COEF) {
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) {
            return NO_HIT;
        }
        let root = sqrt(disc);
        rad_lo = ((-b) - root) / (2.0 * a);
        rad_hi = ((-b) + root) / (2.0 * a);
    } else if (c <= 0.0) {
        rad_lo = -NO_HIT;
        rad_hi = NO_HIT;
    } else {
        return NO_HIT;
    }

    // Axial interval where the projection lies in [0, len].
    var ax_lo: f32;
    var ax_hi: f32;
    if (abs(mu_m) > EPS_COEF) {
        let t_at_zero = -e0u / mu_m;
        let t_at_len = (len - e0u) / mu_m;
        ax_lo = min(t_at_zero, t_at_len);
        ax_hi = max(t_at_zero, t_at_len);
    } else if (e0u >= 0.0 && e0u <= len) {
        ax_lo = -NO_HIT;
        ax_hi = NO_HIT;
    } else {
        return NO_HIT;
    }

    let lo = max(max(rad_lo, ax_lo), 0.0);
    let hi = min(min(rad_hi, ax_hi), 1.0);
    if (lo <= hi) {
        return lo;
    }
    return NO_HIT;
}

// Earliest TOI into the capsule (segment inflated by radius), or `NO_HIT`.
fn capsule_toi(
    prev: vec3<f32>,
    curr: vec3<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    radius: f32,
) -> f32 {
    if (radius <= 0.0) {
        return NO_HIT;
    }
    let axis = p1 - p0;
    if (dot(axis, axis) <= EPS_LEN_SQ) {
        return sphere_toi(prev, curr, p0, radius);
    }
    var best = cylinder_slab_toi(prev, curr, p0, axis, radius);
    best = min(best, sphere_toi(prev, curr, p0, radius));
    best = min(best, sphere_toi(prev, curr, p1, radius));
    return best;
}

// Dispatches the swept TOI test by collider discriminant.
// Rotates `v` by the unit quaternion `q` (xyzw). Scalar Hamilton form.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let w = q.w;
    return v * (w * w - dot(b, b)) + b * (2.0 * dot(v, b)) + cross(b, v) * (2.0 * w);
}

// Rotates `v` by the inverse (conjugate) of the unit quaternion `q`
// (world -> box-local).
fn quat_rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return quat_rotate(vec4<f32>(-q.xyz, q.w), v);
}

// Earliest time in [0, 1] at which the segment `prev -> curr` enters the
// oriented box `(center, orientation, half_extents)`, or `NO_HIT`. Standard
// three-slab ray/box clip in the box-local frame. Mirrors `obb_toi` in
// prism_physics_core: any non-positive half extent makes the box inert.
fn obb_toi(
    prev: vec3<f32>,
    curr: vec3<f32>,
    center: vec3<f32>,
    orientation: vec4<f32>,
    half_extents: vec3<f32>,
) -> f32 {
    if (half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0) {
        return NO_HIT;
    }
    let p = quat_rotate_inv(orientation, prev - center);
    let d = quat_rotate_inv(orientation, curr - prev);
    var t_enter = -3.4028235e38;
    var t_exit = 3.4028235e38;
    for (var axis = 0; axis < 3; axis = axis + 1) {
        let he = half_extents[axis];
        let pa = p[axis];
        let da = d[axis];
        if (abs(da) <= EPS_COEF) {
            // Parallel to this slab: a start outside the slab can never enter.
            if (pa < -he || pa > he) {
                return NO_HIT;
            }
            continue;
        }
        let inv_d = 1.0 / da;
        let t1 = (-he - pa) * inv_d;
        let t2 = (he - pa) * inv_d;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        t_enter = max(t_enter, t_near);
        t_exit = min(t_exit, t_far);
        if (t_enter > t_exit) {
            return NO_HIT;
        }
    }
    if (t_exit < 0.0 || t_enter < 0.0 || t_enter > 1.0) {
        return NO_HIT;
    }
    return t_enter;
}

// Earliest time in [0, 1] at which the segment `prev -> curr` enters the
// convex solid (intersection of half-spaces), or `NO_HIT`. Standard slab
// clip: `t_enter` is the latest entry across approached faces, `t_exit` the
// earliest exit across receding faces; a segment starting inside yields a
// negative entry and is rejected (left to the discrete projection). Mirrors
// `ConvexProxy::segment_toi` in prism_physics_core.
fn convex_toi(prev: vec3<f32>, curr: vec3<f32>, plane_offset: u32, plane_count: u32) -> f32 {
    if (plane_count == 0u) {
        return NO_HIT;
    }
    let dir = curr - prev;
    var t_enter = -3.4028235e38;
    var t_exit = 3.4028235e38;
    for (var i = 0u; i < plane_count; i = i + 1u) {
        let plane = convex_planes[plane_offset + i].plane;
        let normal = plane.xyz;
        if (dot(normal, normal) <= EPS_LEN_SQ) {
            continue;
        }
        // f(t) = normal.(prev + t*dir) - offset; inside is f <= 0.
        let num = dot(normal, prev) - plane.w;
        let rate = dot(normal, dir);
        if (abs(rate) <= EPS_LEN_SQ) {
            // Parallel to this face: a start outside it can never enter.
            if (num > 0.0) {
                return NO_HIT;
            }
            continue;
        }
        let t = -num / rate;
        if (rate > 0.0) {
            t_exit = min(t_exit, t);
        } else {
            t_enter = max(t_enter, t);
        }
        if (t_enter > t_exit) {
            return NO_HIT;
        }
    }
    if (t_exit < 0.0 || t_enter < 0.0 || t_enter > 1.0) {
        return NO_HIT;
    }
    return t_enter;
}

// Projects `pos` out to the least-penetrating face of the convex solid when
// strictly inside, otherwise returns `pos`. Mirrors `ConvexProxy::project_out`
// (a non-negative signed distance on any face proves the point is outside).
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
    return pos + best_normal * best_pen;
}

// Outward unit normal of the convex face the surface point `surf` lies on
// (the face with the largest signed distance), or the zero vector for a
// degenerate hull. Mirrors `ConvexProxy::face_normal`.
fn convex_face_normal(surf: vec3<f32>, plane_offset: u32, plane_count: u32) -> vec3<f32> {
    var best_signed = -3.4028235e38;
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < plane_count; i = i + 1u) {
        let plane = convex_planes[plane_offset + i].plane;
        let normal = plane.xyz;
        if (dot(normal, normal) <= EPS_LEN_SQ) {
            continue;
        }
        let signed = dot(normal, surf) - plane.w;
        if (signed > best_signed) {
            best_signed = signed;
            best_normal = normal;
        }
    }
    let len_sq = dot(best_normal, best_normal);
    if (len_sq <= EPS_LEN_SQ) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return best_normal * inverseSqrt(len_sq);
}

fn collider_toi(c: Collider, prev: vec3<f32>, curr: vec3<f32>) -> f32 {
    if (c.kind == COLLIDER_SPHERE) {
        return sphere_toi(prev, curr, c.p0.xyz, c.radius);
    }
    if (c.kind == COLLIDER_CAPSULE) {
        return capsule_toi(prev, curr, c.p0.xyz, c.p1.xyz, c.radius);
    }
    if (c.kind == COLLIDER_OBB) {
        return obb_toi(prev, curr, c.p0.xyz, c.p2, c.p1.xyz);
    }
    if (c.kind == COLLIDER_CONVEX) {
        return convex_toi(prev, curr, c.pad0, c.pad1);
    }
    return half_space_toi(prev, curr, c.p0.xyz, c.radius);
}

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

fn closest_point_on_segment(p0: vec3<f32>, p1: vec3<f32>, pos: vec3<f32>) -> vec3<f32> {
    let axis = p1 - p0;
    let len_sq = dot(axis, axis);
    if (len_sq <= EPS_LEN_SQ) {
        return p0;
    }
    let t = clamp(dot(pos - p0, axis) / len_sq, 0.0, 1.0);
    return p0 + axis * t;
}

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

// Outward normal of the oriented box face the surface point `surf` lies on, or
// the zero vector when the box is degenerate. Mirrors `obb_face_normal` in
// prism_physics_core: the face is the local axis whose `|local| / he` ratio is
// largest among positive-extent axes.
fn obb_face_normal(
    center: vec3<f32>,
    orientation: vec4<f32>,
    half_extents: vec3<f32>,
    surf: vec3<f32>,
) -> vec3<f32> {
    let local = quat_rotate_inv(orientation, surf - center);
    var best_axis = -1;
    var best_ratio = -3.4028235e38;
    for (var axis = 0; axis < 3; axis = axis + 1) {
        let he = half_extents[axis];
        if (he <= 0.0) {
            continue;
        }
        let ratio = abs(local[axis]) / he;
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_axis = axis;
        }
    }
    if (best_axis < 0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    var local_normal = vec3<f32>(0.0, 0.0, 0.0);
    if (local[best_axis] >= 0.0) {
        local_normal[best_axis] = 1.0;
    } else {
        local_normal[best_axis] = -1.0;
    }
    return quat_rotate(orientation, local_normal);
}

// Projects `pos` onto the collider surface (the nearest feasible point).
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
        return project_out_of_convex(pos, c.pad0, c.pad1);
    }
    return project_out_of_half_space(pos, c.p0.xyz, c.radius);
}

// Unit outward normal of `c` at surface point `surf`, or the zero vector when
// the collider is degenerate and no direction is defined.
fn outward_normal(c: Collider, surf: vec3<f32>) -> vec3<f32> {
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if (c.kind == COLLIDER_SPHERE) {
        n = surf - c.p0.xyz;
    } else if (c.kind == COLLIDER_CAPSULE) {
        let closest = closest_point_on_segment(c.p0.xyz, c.p1.xyz, surf);
        n = surf - closest;
    } else if (c.kind == COLLIDER_OBB) {
        n = obb_face_normal(c.p0.xyz, c.p2, c.p1.xyz, surf);
    } else if (c.kind == COLLIDER_CONVEX) {
        n = convex_face_normal(surf, c.pad0, c.pad1);
    } else {
        n = c.p0.xyz;
    }
    let len_sq = dot(n, n);
    if (len_sq <= EPS_LEN_SQ) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return n * inverseSqrt(len_sq);
}

// Position-level Coulomb friction: cancels the tangential slide inside the
// static cone and shrinks it by `mu * normal_push` otherwise. `normal` is unit.
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    // Pinned particle: never moved.
    if (inv_mass[i] <= 0.0) {
        return;
    }
    let prev = prev_positions[i].xyz;
    let curr = positions[i].xyz;
    let motion = curr - prev;
    if (dot(motion, motion) <= EPS_LEN_SQ) {
        return;
    }

    // Earliest hit across all colliders (slice order breaks ties toward the
    // first, matching the golden's strict `t < best` update).
    var best_t = NO_HIT;
    var best_index = 0u;
    for (var k = 0u; k < params.collider_count; k = k + 1u) {
        let t = collider_toi(colliders[k], prev, curr);
        if (t < best_t) {
            best_t = t;
            best_index = k;
        }
    }
    if (best_t > 1.0) {
        return;
    }

    let best = colliders[best_index];
    let contact = prev + motion * best_t;
    let surface = project_collider(best, contact);
    let n = outward_normal(best, surface);
    if (dot(n, n) <= EPS_LEN_SQ) {
        // No defined normal: snap to the surface only.
        positions[i] = vec4<f32>(surface, positions[i].w);
        return;
    }

    let placed_pre = surface + n * params.skin;
    // Reflect the inbound normal velocity by restitution.
    let v = (placed_pre - prev) * params.inv_dt;
    let vn = dot(v, n);
    if (vn < 0.0) {
        velocities[i] = vec4<f32>(v - n * ((1.0 + params.restitution) * vn), velocities[i].w);
    }
    // Damp the tangential slide; push-out depth is the friction normal size.
    let push = dot(placed_pre - curr, n);
    let placed = apply_coulomb_friction(placed_pre, prev, n, push, params.mu);
    positions[i] = vec4<f32>(placed, positions[i].w);
}
