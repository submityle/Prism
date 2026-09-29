//! GPU signed-distance-field (SDF) body-collision post-pass for guide strands.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::sdf_collision::resolve_sdf_collisions`.
//! The analytic sphere/capsule proxies (`hair_sim.wesl` per-substep pass) are
//! the cheap always-on base, but a jaw line, a collarbone or a prop against the
//! scalp needs a tighter fit than a round primitive gives. Production strand
//! solvers (`AMD` `TressFX` 4) add an SDF collider for exactly this: the body is
//! sampled as a field that is negative inside the surface and positive outside,
//! and any particle that lands inside is pushed back along the field gradient to
//! the zero isosurface (design §6.2, the heavier tier layered on top of the
//! analytic proxies). This shader runs as a per-frame post-pass *after* the
//! XPBD solve, exactly like the CPU `resolve_sdf_collisions`.
//!
//! One invocation owns one particle of the flat guide pool. The body SDF is a
//! union of analytic primitives (half-space planes, axis-aligned boxes, spheres
//! and capsules); the union distance is the minimum over the primitives and a
//! point is relaxed out along the numeric gradient of that union field until it
//! is outside every primitive or the iteration cap is reached. Everything is
//! bit-faithful to the CPU golden: the same particles and primitives produce
//! identical positions. Pinned particles (skinned roots, `inverse_mass <= 0`)
//! never move, an empty primitive set or zero iterations is a no-op, and a
//! particle on a zero-gradient spot escapes along a fixed `+Y` axis rather than
//! producing a NaN direction. A `GPU` is now available, so this shader is
//! compile- and type-checked through the same render-world `ShaderCache` /
//! `wesl` pipeline the runtime uses (see `shader_tests`), with on-device
//! dispatch/perf calibration still awaiting real hardware.

/// Squared-length epsilon: vectors shorter than its square root are treated as
/// zero-length. Mirrors the CPU `sdf_collision::EPS_LEN_SQ`.
const HAIR_SDF_EPS_LEN_SQ: f32 = 1.0e-24;

/// Central-difference step of the union-field gradient, in world units. Mirrors
/// the CPU `sdf_collision::GRAD_STEP`.
const HAIR_SDF_GRAD_STEP: f32 = 1.0e-3;

/// Finite stand-in for `f32::INFINITY` used as the union-min identity and as an
/// inert primitive's distance. Any real primitive reports a finite distance
/// everywhere, so this is only ever selected when every primitive is inert, in
/// which case the point is treated as outside (identical to the CPU path, which
/// breaks on `d >= 0` before ever differencing the field).
const HAIR_SDF_FAR: f32 = 1.0e30;

/// Primitive kind tags. Mirror the CPU `SdfPrimitive` enum discriminants.
const HAIR_SDF_SPHERE: u32 = 0u;
const HAIR_SDF_CAPSULE: u32 = 1u;
const HAIR_SDF_HALF_SPACE: u32 = 2u;
const HAIR_SDF_BOX: u32 = 3u;

/// One primitive of the body SDF union. Packs the CPU `SdfPrimitive` variants
/// into two `vec4`s plus a kind tag on a clean 48-byte stride:
/// - `Sphere`   : `p0.xyz` = center, `p0.w` = radius.
/// - `Capsule`  : `p0.xyz` = endpoint `a`, `p0.w` = radius, `p1.xyz` = endpoint `b`.
/// - `HalfSpace`: `p0.xyz` = outward normal, `p0.w` = plane offset.
/// - `Box`      : `p0.xyz` = center, `p1.xyz` = positive half extents.
struct HairSdfPrimitive {
    p0: vec4<f32>,
    p1: vec4<f32>,
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Dispatch parameters. `iterations` is the push-out relaxation budget
/// (CPU `DEFAULT_SDF_ITERATIONS` = 4), `primitive_count` bounds the union loop
/// and `particle_count` bounds the dispatch.
struct HairSdfParams {
    iterations: u32,
    primitive_count: u32,
    particle_count: u32,
}

// Particle state: `xyz` = world position, `w` = inverse mass (0 = pinned).
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// The primitives whose union defines the body surface.
@group(0) @binding(1) var<storage, read> primitives: array<HairSdfPrimitive>;

var<immediate> params: HairSdfParams;

// Manual `sqrt(dot(v, v))` so the twin matches the CPU `Vec3::length` exactly
// (rather than trusting the `length` builtin's lowering per backend).
fn hair_sdf_length(v: vec3<f32>) -> f32 {
    return sqrt(dot(v, v));
}

// Mirror of CPU `Vec3::normalize_or_zero`: scale by the reciprocal length, or
// return the zero vector when the input is (near) zero-length.
fn hair_sdf_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > HAIR_SDF_EPS_LEN_SQ {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Returns the point on segment `a`..`b` closest to `point`; a zero-length
// segment collapses to `a`. Mirrors CPU `sdf_collision::closest_point_on_segment`.
fn hair_sdf_closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if len_sq <= HAIR_SDF_EPS_LEN_SQ {
        return a;
    }
    let t = clamp(dot(point - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Exact signed distance to an axis-aligned box (Quilez form). Mirrors CPU
// `sdf_collision::box_signed_distance`: outside distance is the length of the
// positive part of `|p - center| - half_extents`, inside distance is the
// least-penetrating (largest, clamped non-positive) interior component.
fn hair_sdf_box_distance(center: vec3<f32>, half_extents: vec3<f32>, point: vec3<f32>) -> f32 {
    if half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0 {
        return HAIR_SDF_FAR;
    }
    let rel = point - center;
    let dx = abs(rel.x) - half_extents.x;
    let dy = abs(rel.y) - half_extents.y;
    let dz = abs(rel.z) - half_extents.z;
    let ox = max(dx, 0.0);
    let oy = max(dy, 0.0);
    let oz = max(dz, 0.0);
    let outside = sqrt(ox * ox + oy * oy + oz * oz);
    let inside = min(max(max(dx, dy), dz), 0.0);
    return outside + inside;
}

// Signed distance from `point` to one primitive: negative inside the solid
// region, positive outside. An inert primitive (degenerate radius/extent/
// normal) reports `HAIR_SDF_FAR`. Mirrors CPU `SdfPrimitive::signed_distance`.
fn hair_sdf_primitive_distance(primitive: HairSdfPrimitive, point: vec3<f32>) -> f32 {
    if primitive.kind == HAIR_SDF_SPHERE {
        let radius = primitive.p0.w;
        if radius <= 0.0 {
            return HAIR_SDF_FAR;
        }
        return hair_sdf_length(point - primitive.p0.xyz) - radius;
    }
    if primitive.kind == HAIR_SDF_CAPSULE {
        let radius = primitive.p0.w;
        if radius <= 0.0 {
            return HAIR_SDF_FAR;
        }
        let closest = hair_sdf_closest_point_on_segment(primitive.p0.xyz, primitive.p1.xyz, point);
        return hair_sdf_length(point - closest) - radius;
    }
    if primitive.kind == HAIR_SDF_HALF_SPACE {
        let normal = primitive.p0.xyz;
        let len_sq = dot(normal, normal);
        if len_sq <= HAIR_SDF_EPS_LEN_SQ {
            return HAIR_SDF_FAR;
        }
        let inv_len = 1.0 / sqrt(len_sq);
        return dot(point, normal) * inv_len - primitive.p0.w;
    }
    // Box.
    return hair_sdf_box_distance(primitive.p0.xyz, primitive.p1.xyz, point);
}

// Union signed distance: the minimum over all primitives (an empty union is
// `HAIR_SDF_FAR`, i.e. everywhere outside). Mirrors CPU `union_signed_distance`.
fn hair_sdf_union_distance(point: vec3<f32>) -> f32 {
    var d = HAIR_SDF_FAR;
    for (var i = 0u; i < params.primitive_count; i = i + 1u) {
        let di = hair_sdf_primitive_distance(primitives[i], point);
        if di < d {
            d = di;
        }
    }
    return d;
}

// Outward unit gradient of the union field at `point`, by central differences,
// zero when the field is locally flat. Mirrors CPU `union_gradient`.
fn hair_sdf_union_gradient(point: vec3<f32>) -> vec3<f32> {
    let gx = hair_sdf_union_distance(vec3<f32>(point.x + HAIR_SDF_GRAD_STEP, point.y, point.z))
        - hair_sdf_union_distance(vec3<f32>(point.x - HAIR_SDF_GRAD_STEP, point.y, point.z));
    let gy = hair_sdf_union_distance(vec3<f32>(point.x, point.y + HAIR_SDF_GRAD_STEP, point.z))
        - hair_sdf_union_distance(vec3<f32>(point.x, point.y - HAIR_SDF_GRAD_STEP, point.z));
    let gz = hair_sdf_union_distance(vec3<f32>(point.x, point.y, point.z + HAIR_SDF_GRAD_STEP))
        - hair_sdf_union_distance(vec3<f32>(point.x, point.y, point.z - HAIR_SDF_GRAD_STEP));
    return hair_sdf_normalize_or_zero(vec3<f32>(gx, gy, gz));
}

@compute @workgroup_size(64)
fn resolve_sdf_collisions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.particle_count {
        return;
    }
    if params.primitive_count == 0u || params.iterations == 0u {
        return;
    }
    // Pinned skinned roots never move (mirrors `is_pinned`).
    if positions[index].w <= 0.0 {
        return;
    }

    var p = positions[index].xyz;
    for (var it = 0u; it < params.iterations; it = it + 1u) {
        let d = hair_sdf_union_distance(p);
        // Outside every primitive, or a NaN field (self-inequality) — stop.
        if d >= 0.0 || d != d {
            break;
        }
        let grad = hair_sdf_union_gradient(p);
        if dot(grad, grad) <= HAIR_SDF_EPS_LEN_SQ {
            // Locally flat field: escape along a fixed axis by the depth.
            p = p + vec3<f32>(0.0, -d, 0.0);
            break;
        }
        p = p + grad * (-d);
    }
    positions[index] = vec4<f32>(p, positions[index].w);
}
