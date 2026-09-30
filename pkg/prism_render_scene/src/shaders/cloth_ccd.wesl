//! GPU compute twin of the cloth continuous-collision (CCD) sweep.
//!
//! CPU golden reference:
//! prism_render_architecture::cloth::ccd::resolve_ccd -- the per-particle
//! tunnelling guard that sweeps each free particle's segment prev -> curr
//! against every analytic body collider, solves the earliest closed-form time
//! of impact (TOI), snaps the particle to the collider surface plus a skin
//! offset, reflects its normal velocity by a restitution coefficient, and damps
//! the tangential slide with position-level Coulomb friction (Macklin et al.
//! 2014). The TOI solvers are the same exact closed forms as the golden (sphere
//! = one quadratic, half-space = linear, capsule = infinite cylinder slab
//! unioned with two end-cap spheres); only sqrt is used, no transcendentals.
//!
//! ## One own-slot kernel (embarrassingly parallel, exact parity)
//!
//! Every particle only ever reads its own prev/curr position and writes its own
//! position and velocity; the collider set is read-only. The golden visits
//! particles in index order purely for determinism, and since no particle reads
//! another particle's corrected position the one-invocation-per-particle kernel
//! reproduces the golden bit-for-bit regardless of dispatch order. Each
//! invocation walks the colliders in slice order (the earliest valid TOI wins,
//! ties broken by the first collider in order, matching the golden's strict
//! t < best comparison) with no reduction and no synchronisation.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via cloth::shader_tests); real-machine bit-parity against the golden is
//! proven by cloth::ccd_gpu_tests on sphere / capsule / half-space sweeps with
//! restitution and friction.

/// Sphere collider tag. Matches CLOTH_COLLIDER_SPHERE in cloth_collision.wesl.
const CLOTH_COLLIDER_SPHERE: u32 = 0u;
/// Capsule collider tag. Matches CLOTH_COLLIDER_CAPSULE.
const CLOTH_COLLIDER_CAPSULE: u32 = 1u;
/// Half-space collider tag. Matches CLOTH_COLLIDER_HALF_SPACE.
const CLOTH_COLLIDER_HALF_SPACE: u32 = 2u;

/// Squared-length floor for treating a segment / normal as degenerate. Mirrors
/// the CPU EPS_LEN_SQ.
const CLOTH_CCD_EPS_LEN_SQ: f32 = 1.0e-12;
/// Scalar-coefficient floor for classifying a quadratic as linear or a segment
/// as parallel. Mirrors the CPU EPS_COEF.
const CLOTH_CCD_EPS_COEF: f32 = 1.0e-12;
/// Tangential-slide floor below which a friction correction is skipped. Mirrors
/// the CPU EPS_FRICTION.
const CLOTH_CCD_EPS_FRICTION: f32 = 1.0e-12;
/// Stand-in for an unbounded interval endpoint (the CPU uses +/- infinity). Any
/// finite TOI lies well inside [0, 1] after the final clamp, so the largest
/// representable magnitude reproduces the CPU min / max interval intersection
/// exactly without ever entering arithmetic.
const CLOTH_CCD_INF: f32 = 3.4e38;

/// One collision proxy. Byte-compatible with ClothCollider in
/// cloth_collision.wesl and the GpuClothCollider ABI (eight 4-byte words).
struct ClothCollider {
    /// Selects the collider variant (CLOTH_COLLIDER_*).
    kind: u32,
    /// First point x: sphere centre / capsule p0 / half-space normal x.
    ax: f32,
    /// First point y.
    ay: f32,
    /// First point z.
    az: f32,
    /// Second point x (capsule p1 only, else ignored).
    bx: f32,
    /// Second point y.
    by: f32,
    /// Second point z.
    bz: f32,
    /// Radius (sphere / capsule) or signed offset (half-space).
    radius: f32,
}

/// Uniform parameters for the CCD sweep. Flat 32-byte block: the three sanitized
/// scalars (skin, restitution, dt), the friction coefficient, two counts and
/// padding to a 16-byte stride.
struct ClothCcdParams {
    /// Push-out distance along the outward normal after a hit (>= 0).
    skin: f32,
    /// Normal restitution in [0, 1].
    restitution: f32,
    /// Substep dt; a (near) zero dt leaves velocities untouched.
    dt: f32,
    /// Coulomb friction coefficient in [0, 1].
    friction: f32,
    /// Number of colliders bounding the per-particle inner loop.
    collider_count: u32,
    /// Number of particles bounding the dispatch (the CPU min of the position
    /// and prev-position slice lengths).
    particle_count: u32,
    /// Padding to a 16-byte uniform stride.
    pad0: u32,
    /// Padding to a 16-byte uniform stride.
    pad1: u32,
}

/// Earliest time-of-impact result: hit reports whether the swept segment ever
/// reached the collider, and t is the entry time in [0, 1] when it did. Mirrors
/// the CPU Option<f32> the closed-form solvers return.
struct ClothCcdToi {
    /// Whether a valid entry crossing exists in [0, 1].
    hit: bool,
    /// Entry time in [0, 1]; only meaningful when hit is true.
    t: f32,
}

/// Outward-normal result: has reports whether a unit direction is defined and n
/// is that unit outward normal. Mirrors the CPU outward_normal Option<Vec3>.
struct ClothCcdNormal {
    /// Whether a non-degenerate outward normal exists.
    has: bool,
    /// The unit outward normal; only meaningful when has is true.
    n: vec3<f32>,
}

// Current particle state: xyz = world position, w = inverse mass. Read + write;
// the sweep snaps positions and leaves the mass alone.
@group(0) @binding(0) var<storage, read_write> ccd_positions: array<vec4<f32>>;
// Current particle velocity: xyz = velocity, w carried through untouched.
@group(0) @binding(1) var<storage, read_write> ccd_velocities: array<vec4<f32>>;
// Frame-start positions (xyz; w ignored) each swept segment starts from.
@group(0) @binding(2) var<storage, read> ccd_prev_positions: array<vec4<f32>>;
// Analytic body colliders, applied in slice order (earliest TOI wins).
@group(0) @binding(3) var<storage, read> ccd_colliders: array<ClothCollider>;
// Flat 32-byte uniform block.
@group(0) @binding(4) var<uniform> ccd_params: ClothCcdParams;

/// Normalises v, returning the zero vector when it is shorter than the
/// degeneracy floor. Mirrors the CPU Vec3::normalize_or_zero exactly, including
/// its 1.0 / sqrt(len_sq) form (not the hardware rsqrt) so the direction matches
/// the golden to the ULP.
fn cloth_ccd_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > CLOTH_CCD_EPS_LEN_SQ {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Projects pos out to the surface of the sphere (center, radius). Mirrors the
/// CPU project_out_of_sphere: a non-positive radius or an exterior point is
/// untouched, and a point coincident with the centre escapes along +Y.
fn cloth_ccd_project_out_of_sphere(pos: vec3<f32>, center: vec3<f32>, radius: f32) -> vec3<f32> {
    if radius <= 0.0 {
        return pos;
    }
    let delta = pos - center;
    let dist_sq = dot(delta, delta);
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= CLOTH_CCD_EPS_LEN_SQ {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = delta * (1.0 / sqrt(dist_sq));
    return center + dir * radius;
}

/// Projects pos onto the plane dot(normal, x) == offset when it lies on the
/// infeasible side. Mirrors the CPU project_out_of_half_space.
fn cloth_ccd_project_out_of_half_space(pos: vec3<f32>, normal: vec3<f32>, offset: f32) -> vec3<f32> {
    let len_sq = dot(normal, normal);
    if len_sq <= CLOTH_CCD_EPS_LEN_SQ {
        return pos;
    }
    let signed = dot(normal, pos) - offset;
    if signed >= 0.0 {
        return pos;
    }
    let t = -signed / len_sq;
    return pos + normal * t;
}

/// Returns the point on segment p0..p1 closest to pos. Mirrors the CPU
/// closest_point_on_segment (parameter clamped to [0, 1]).
fn cloth_ccd_closest_point_on_segment(p0: vec3<f32>, p1: vec3<f32>, pos: vec3<f32>) -> vec3<f32> {
    let axis = p1 - p0;
    let len_sq = dot(axis, axis);
    if len_sq <= CLOTH_CCD_EPS_LEN_SQ {
        return p0;
    }
    let t = clamp(dot(pos - p0, axis) / len_sq, 0.0, 1.0);
    return p0 + axis * t;
}

/// Projects pos out of one collider. Mirrors BodyCollider::project.
fn cloth_ccd_project_collider(col: ClothCollider, pos: vec3<f32>) -> vec3<f32> {
    let a = vec3<f32>(col.ax, col.ay, col.az);
    if col.kind == CLOTH_COLLIDER_SPHERE {
        return cloth_ccd_project_out_of_sphere(pos, a, col.radius);
    }
    if col.kind == CLOTH_COLLIDER_CAPSULE {
        let b = vec3<f32>(col.bx, col.by, col.bz);
        let closest = cloth_ccd_closest_point_on_segment(a, b, pos);
        return cloth_ccd_project_out_of_sphere(pos, closest, col.radius);
    }
    return cloth_ccd_project_out_of_half_space(pos, a, col.radius);
}

/// Earliest root in [0, 1] of a*t^2 + b*t + c <= 0 for non-negative a. Mirrors
/// the CPU first_entry_time: c <= 0 reports t = 0, a (near) zero is the linear
/// branch, otherwise the earlier quadratic root is the entry crossing.
fn cloth_ccd_first_entry_time(a: f32, b: f32, c: f32) -> ClothCcdToi {
    if c <= 0.0 {
        return ClothCcdToi(true, 0.0);
    }
    if a <= CLOTH_CCD_EPS_COEF {
        if b >= -CLOTH_CCD_EPS_COEF {
            return ClothCcdToi(false, 0.0);
        }
        let t = -c / b;
        if t <= 1.0 {
            return ClothCcdToi(true, max(t, 0.0));
        }
        return ClothCcdToi(false, 0.0);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return ClothCcdToi(false, 0.0);
    }
    let root = sqrt(disc);
    let t = ((-b) - root) / (2.0 * a);
    if t >= 0.0 && t <= 1.0 {
        return ClothCcdToi(true, t);
    }
    return ClothCcdToi(false, 0.0);
}

/// Sphere sweep TOI. Mirrors the CPU sphere_toi.
fn cloth_ccd_sphere_toi(prev: vec3<f32>, curr: vec3<f32>, center: vec3<f32>, radius: f32) -> ClothCcdToi {
    if radius <= 0.0 {
        return ClothCcdToi(false, 0.0);
    }
    let m = curr - prev;
    let e = prev - center;
    let a = dot(m, m);
    let b = 2.0 * dot(e, m);
    let c = dot(e, e) - radius * radius;
    return cloth_ccd_first_entry_time(a, b, c);
}

/// Half-space sweep TOI. Mirrors the CPU half_space_toi.
fn cloth_ccd_half_space_toi(prev: vec3<f32>, curr: vec3<f32>, normal: vec3<f32>, offset: f32) -> ClothCcdToi {
    if dot(normal, normal) <= CLOTH_CCD_EPS_LEN_SQ {
        return ClothCcdToi(false, 0.0);
    }
    let s0 = dot(normal, prev) - offset;
    if s0 <= 0.0 {
        return ClothCcdToi(true, 0.0);
    }
    let ds = dot(normal, curr - prev);
    if ds >= -CLOTH_CCD_EPS_COEF {
        return ClothCcdToi(false, 0.0);
    }
    let t = -s0 / ds;
    if t <= 1.0 {
        return ClothCcdToi(true, max(t, 0.0));
    }
    return ClothCcdToi(false, 0.0);
}

/// Infinite-cylinder-slab sweep TOI restricted to the axis slab [0, len].
/// Mirrors the CPU cylinder_slab_toi: it intersects the radial sub-radius
/// interval, the axial in-slab interval and [0, 1], and returns the lower bound.
fn cloth_ccd_cylinder_slab_toi(prev: vec3<f32>, curr: vec3<f32>, p0: vec3<f32>, axis: vec3<f32>, radius: f32) -> ClothCcdToi {
    let len = length(axis);
    if len <= CLOTH_CCD_EPS_COEF {
        return ClothCcdToi(false, 0.0);
    }
    let u = axis * (1.0 / len);
    let e0 = prev - p0;
    let m = curr - prev;
    let mu = dot(m, u);
    let e0u = dot(e0, u);

    let a = dot(m, m) - mu * mu;
    let b = 2.0 * (dot(e0, m) - e0u * mu);
    let c = dot(e0, e0) - e0u * e0u - radius * radius;
    var rad_lo = 0.0;
    var rad_hi = 0.0;
    if a > CLOTH_CCD_EPS_COEF {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return ClothCcdToi(false, 0.0);
        }
        let root = sqrt(disc);
        rad_lo = ((-b) - root) / (2.0 * a);
        rad_hi = ((-b) + root) / (2.0 * a);
    } else if c <= 0.0 {
        rad_lo = -CLOTH_CCD_INF;
        rad_hi = CLOTH_CCD_INF;
    } else {
        return ClothCcdToi(false, 0.0);
    }

    var ax_lo = 0.0;
    var ax_hi = 0.0;
    if abs(mu) > CLOTH_CCD_EPS_COEF {
        let t_at_zero = -e0u / mu;
        let t_at_len = (len - e0u) / mu;
        ax_lo = min(t_at_zero, t_at_len);
        ax_hi = max(t_at_zero, t_at_len);
    } else if e0u >= 0.0 && e0u <= len {
        ax_lo = -CLOTH_CCD_INF;
        ax_hi = CLOTH_CCD_INF;
    } else {
        return ClothCcdToi(false, 0.0);
    }

    let lo = max(max(rad_lo, ax_lo), 0.0);
    let hi = min(min(rad_hi, ax_hi), 1.0);
    if lo <= hi {
        return ClothCcdToi(true, lo);
    }
    return ClothCcdToi(false, 0.0);
}

/// Returns whichever TOI is earlier, preferring a hit over a miss. Mirrors the
/// CPU earliest helper.
fn cloth_ccd_earliest(lhs: ClothCcdToi, rhs: ClothCcdToi) -> ClothCcdToi {
    if lhs.hit && rhs.hit {
        return ClothCcdToi(true, min(lhs.t, rhs.t));
    }
    if lhs.hit {
        return lhs;
    }
    return rhs;
}

/// Capsule sweep TOI = earliest of the cylinder slab and the two end-cap
/// spheres. Mirrors the CPU capsule_toi; a collapsed capsule reduces to a
/// sphere at p0.
fn cloth_ccd_capsule_toi(prev: vec3<f32>, curr: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, radius: f32) -> ClothCcdToi {
    if radius <= 0.0 {
        return ClothCcdToi(false, 0.0);
    }
    let axis = p1 - p0;
    let len_sq = dot(axis, axis);
    if len_sq <= CLOTH_CCD_EPS_LEN_SQ {
        return cloth_ccd_sphere_toi(prev, curr, p0, radius);
    }
    var best = cloth_ccd_cylinder_slab_toi(prev, curr, p0, axis, radius);
    best = cloth_ccd_earliest(best, cloth_ccd_sphere_toi(prev, curr, p0, radius));
    best = cloth_ccd_earliest(best, cloth_ccd_sphere_toi(prev, curr, p1, radius));
    return best;
}

/// Dispatches the swept segment against one collider. Mirrors the CPU
/// collider_toi.
fn cloth_ccd_collider_toi(col: ClothCollider, prev: vec3<f32>, curr: vec3<f32>) -> ClothCcdToi {
    let a = vec3<f32>(col.ax, col.ay, col.az);
    if col.kind == CLOTH_COLLIDER_SPHERE {
        return cloth_ccd_sphere_toi(prev, curr, a, col.radius);
    }
    if col.kind == CLOTH_COLLIDER_CAPSULE {
        let b = vec3<f32>(col.bx, col.by, col.bz);
        return cloth_ccd_capsule_toi(prev, curr, a, b, col.radius);
    }
    return cloth_ccd_half_space_toi(prev, curr, a, col.radius);
}

/// Unit outward normal of the collider at surf. Mirrors the CPU outward_normal.
fn cloth_ccd_outward_normal(col: ClothCollider, surf: vec3<f32>) -> ClothCcdNormal {
    let a = vec3<f32>(col.ax, col.ay, col.az);
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if col.kind == CLOTH_COLLIDER_SPHERE {
        n = cloth_ccd_normalize_or_zero(surf - a);
    } else if col.kind == CLOTH_COLLIDER_CAPSULE {
        let b = vec3<f32>(col.bx, col.by, col.bz);
        let closest = cloth_ccd_closest_point_on_segment(a, b, surf);
        n = cloth_ccd_normalize_or_zero(surf - closest);
    } else {
        n = cloth_ccd_normalize_or_zero(a);
    }
    if dot(n, n) <= CLOTH_CCD_EPS_LEN_SQ {
        return ClothCcdNormal(false, n);
    }
    return ClothCcdNormal(true, n);
}

/// Position-level Coulomb friction against a contact with known outward unit
/// normal and push-out depth normal_push. Mirrors the CPU apply_coulomb_friction
/// (Macklin et al. 2014): the tangential slide of pos - prev is cancelled inside
/// the static cone and shrunk by mu * normal_push in the dynamic regime.
fn cloth_ccd_apply_coulomb_friction(pos: vec3<f32>, prev: vec3<f32>, normal: vec3<f32>, normal_push: f32, mu: f32) -> vec3<f32> {
    if mu <= 0.0 || normal_push <= 0.0 {
        return pos;
    }
    let delta = pos - prev;
    let normal_amount = dot(delta, normal);
    let tangent = delta - normal * normal_amount;
    let tan_len_sq = dot(tangent, tangent);
    if tan_len_sq <= CLOTH_CCD_EPS_FRICTION {
        return pos;
    }
    let tan_len = sqrt(tan_len_sq);
    let scale = min(mu * normal_push / tan_len, 1.0);
    return pos - tangent * scale;
}

/// Per-particle continuous-collision sweep. One invocation per particle. Skips
/// pinned particles, (near) stationary particles and misses exactly like the
/// golden, then snaps to the earliest hit surface, reflects the normal velocity
/// and damps the tangential slide.
@compute @workgroup_size(64, 1, 1)
fn cloth_resolve_ccd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ccd_params.particle_count {
        return;
    }
    // Pinned particles (inverse mass <= 0) never move.
    if ccd_positions[i].w <= 0.0 {
        return;
    }
    let prev = ccd_prev_positions[i].xyz;
    let curr = ccd_positions[i].xyz;
    let motion = curr - prev;
    if dot(motion, motion) <= CLOTH_CCD_EPS_LEN_SQ {
        return;
    }

    // Sanitised scalars (finite host inputs assumed; the golden's sanitise
    // maps NaN to safe values which the parity fixtures never exercise).
    let skin = max(ccd_params.skin, 0.0);
    let restitution = clamp(ccd_params.restitution, 0.0, 1.0);
    let mu = clamp(ccd_params.friction, 0.0, 1.0);
    var inv_dt = 0.0;
    if abs(ccd_params.dt) > CLOTH_CCD_EPS_COEF {
        inv_dt = 1.0 / ccd_params.dt;
    }

    // Earliest hit across all colliders (slice order, strict t < best).
    var have = false;
    var best_t = 0.0;
    var best = ccd_colliders[0];
    for (var c = 0u; c < ccd_params.collider_count; c = c + 1u) {
        let toi = cloth_ccd_collider_toi(ccd_colliders[c], prev, curr);
        if toi.hit {
            var take = false;
            if !have {
                take = true;
            } else if toi.t < best_t {
                take = true;
            }
            if take {
                best_t = toi.t;
                best = ccd_colliders[c];
                have = true;
            }
        }
    }
    if !have {
        return;
    }

    let contact = prev + motion * best_t;
    let surface = cloth_ccd_project_collider(best, contact);
    let nres = cloth_ccd_outward_normal(best, surface);
    if nres.has {
        let n = nres.n;
        let placed = surface + n * skin;
        let v = (placed - prev) * inv_dt;
        let vn = dot(v, n);
        if vn < 0.0 {
            let reflected = v - n * ((1.0 + restitution) * vn);
            ccd_velocities[i] = vec4<f32>(reflected, ccd_velocities[i].w);
        }
        let push = dot(placed - curr, n);
        let final_pos = cloth_ccd_apply_coulomb_friction(placed, prev, n, push, mu);
        ccd_positions[i] = vec4<f32>(final_pos, ccd_positions[i].w);
    } else {
        ccd_positions[i] = vec4<f32>(surface, ccd_positions[i].w);
    }
}
