//! `wgpu` compute twin of the particle-versus-environment collision contract
//! ([`collision`](prism_render_architecture::particle::collision), particle
//! design §10, §13).
//!
//! The `CPU` golden
//! [`collision`](prism_render_architecture::particle::collision) owns the
//! deterministic per-particle collision layer: the signed-distance primitives
//! ([`sd_sphere`](prism_render_architecture::particle::collision::sd_sphere),
//! [`sd_capsule`](prism_render_architecture::particle::collision::sd_capsule),
//! [`sd_aabb`](prism_render_architecture::particle::collision::sd_aabb)), the
//! closest-point projections
//! ([`closest_point_on_segment`](prism_render_architecture::particle::collision::closest_point_on_segment),
//! [`closest_point_sphere`](prism_render_architecture::particle::collision::closest_point_sphere),
//! [`closest_point_capsule`](prism_render_architecture::particle::collision::closest_point_capsule),
//! [`closest_point_aabb`](prism_render_architecture::particle::collision::closest_point_aabb)),
//! the contact predicates
//! ([`collide_half_space`](prism_render_architecture::particle::collision::collide_half_space),
//! [`collide_sphere`](prism_render_architecture::particle::collision::collide_sphere),
//! [`collide_capsule`](prism_render_architecture::particle::collision::collide_capsule),
//! [`collide_aabb`](prism_render_architecture::particle::collision::collide_aabb),
//! [`collide_sdf`](prism_render_architecture::particle::collision::collide_sdf)),
//! the impulse response
//! ([`resolve_contact`](prism_render_architecture::particle::collision::resolve_contact),
//! [`resolve_particle_collision`](prism_render_architecture::particle::collision::resolve_particle_collision))
//! and the continuous (`TOI`) sweeps
//! ([`swept_sphere_vs_plane`](prism_render_architecture::particle::collision::swept_sphere_vs_plane),
//! [`swept_sphere_vs_sphere`](prism_render_architecture::particle::collision::swept_sphere_vs_sphere)).
//! [`GpuCollision`] is the on-device twin: one thread solves one
//! [`CollisionQuery`] and writes one [`CollisionResult`], so a passing
//! real-device parity test is direct evidence the ported kernel folds the same
//! distances, normals, penetrations and impulses the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each query selects one golden routine by a `u32` tag and the kernel
//! reproduces it branch for branch: the three signed-distance primitives, the
//! four closest-point projections, the five contact predicates (each returning
//! a [`Contact`](prism_render_architecture::particle::collision::Contact) of
//! outward normal, non-negative penetration and surface point), the single and
//! batch impulse solves (positional push-out, approaching-only normal
//! restitution and the `Coulomb` friction cone, with the batch taking the
//! deepest penetrating contact) and the two continuous sweeps (their
//! [`Option`] result re-mapped to a `u32` hit flag plus a `f32` time of
//! impact, mirroring the `ray_aabb` flag convention).
//!
//! # What is deliberately not twinned
//!
//! The enum-dispatch front door
//! [`collide`](prism_render_architecture::particle::collision::collide) is only
//! a `match` over the per-primitive predicates already twinned here, so it is
//! kept on the host rather than duplicated. The screen-space depth-buffer path
//! ([`depth_buffer_hit`](prism_render_architecture::particle::collision::depth_buffer_hit)
//! and
//! [`resolve_depth_collision`](prism_render_architecture::particle::collision::resolve_depth_collision))
//! needs a depth-buffer sample and reconstructed scene normal that live in
//! render state, not in this pure numeric layer, so it stays on the host too.
//!
//! # No transcendental math
//!
//! Every routine is polynomial plus at most one `sqrt`: the lengths behind the
//! projections and penetrations, and the single `sqrt` of the quadratic
//! time-of-impact discriminant. The kernel calls no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no inverse trigonometry and no `smoothstep` or `round`;
//! the box interior normal replaces the reference `signum` with an explicit
//! sign branch, and the degenerate guards compare squared lengths against a
//! floor rather than using an `f32` equality.
//!
//! # Correctness model
//!
//! The dispatch tag is an integer classification, so the kernel runs exactly
//! the branch the host requested. The continuous entries thread through
//! multiplies, adds, guarded divisions and at most one `sqrt`, so `CPU` and
//! `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous quantity and an *exact* match on the
//! discrete hit flags, whose fixtures are placed clear of every contact
//! boundary so the boolean verdict is unambiguous.
//!
//! # Degenerate inputs
//!
//! A zero-length segment, a query at a sphere or capsule axis, a zero-length
//! field gradient and a stationary or surface-parallel sweep all hit the same
//! squared-length and `EPS` guards the reference uses: the normal falls back to
//! a fixed up axis and the sweep reports a miss, so no branch ever divides by a
//! near-zero denominator or yields a `NaN`. An empty query batch short-circuits
//! on the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! only loop is the fixed batch over at most [`MAX_COLLIDERS`] colliders, so
//! each thread performs a bounded sequence of arithmetic and the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::collision`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;
use prism_render_architecture::particle::collision::{
    closest_point_aabb, closest_point_capsule, closest_point_on_segment, closest_point_sphere,
    collide_aabb, collide_capsule, collide_half_space, collide_sdf, collide_sphere,
    resolve_contact, resolve_particle_collision, sd_aabb, sd_capsule, sd_sphere,
    swept_sphere_vs_plane, swept_sphere_vs_sphere, Capsule, Collider, CollisionResponse, Contact,
    ParticleState, ResponseParams, SdfSample, Sphere,
};
use prism_render_architecture::particle::sort_cull::{Aabb, Plane};
use prism_render_architecture::particle::Vec3;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of colliders a single
/// [`CollisionQuery::ResolveParticleCollision`] batch carries on device.
///
/// The batch solve runs a fixed loop over this many slots so the kernel stays
/// bounded and terminating; a host query with more colliders is truncated to
/// the first [`MAX_COLLIDERS`] when encoded.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::collision`；无第三方引擎源码或衍生代码。
pub const MAX_COLLIDERS: usize = 8;

/// Discrete hit code the kernel writes for a contact / sweep that connects:
/// decoded on the host with `== CODE_HIT`. A direct `f32` equality is
/// forbidden, so the hit verdict travels as an integer flag.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` collision kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden [`collision`](prism_render_architecture::particle::collision) branch
/// for branch; see the module documentation for the algorithm.
const COLLISION_WGSL: &str = r#"
// Particle collision twin: one thread per query runs the routine its `tag`
// selects, reproducing the CPU golden `particle::collision` branch for branch.
// It uses only the portable core-WGSL subset (abs/min/max/clamp/sqrt and
// + - * / plus unsigned index math), needs no transcendental call and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. The only
// loop is the fixed batch over at most MAX_COLLIDERS colliders, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::collision；无第三方引擎
// 源码或衍生代码。

// Scalar tolerance guarding near-degenerate denominators (a parallel or
// stationary sweep); matches the reference `EPS`. Used instead of an f32 `==`.
const EPS: f32 = 1.0e-6;

// Squared-length floor below which a direction is treated as the zero vector;
// matches the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Routine tags; the host casts its query selector straight to these codes.
const TAG_SD_SPHERE: u32 = 0u;
const TAG_SD_CAPSULE: u32 = 1u;
const TAG_SD_AABB: u32 = 2u;
const TAG_CLOSEST_SEGMENT: u32 = 3u;
const TAG_CLOSEST_SPHERE: u32 = 4u;
const TAG_CLOSEST_CAPSULE: u32 = 5u;
const TAG_CLOSEST_AABB: u32 = 6u;
const TAG_COLLIDE_HALF_SPACE: u32 = 7u;
const TAG_COLLIDE_SPHERE: u32 = 8u;
const TAG_COLLIDE_CAPSULE: u32 = 9u;
const TAG_COLLIDE_AABB: u32 = 10u;
const TAG_COLLIDE_SDF: u32 = 11u;
const TAG_RESOLVE_CONTACT: u32 = 12u;
const TAG_RESOLVE_PARTICLE: u32 = 13u;
const TAG_SWEPT_PLANE: u32 = 14u;
const TAG_SWEPT_SPHERE: u32 = 15u;

// Collider kinds inside a resolve-particle batch slot.
const COLLIDER_HALF_SPACE: u32 = 0u;
const COLLIDER_SPHERE: u32 = 1u;
const COLLIDER_BOX: u32 = 2u;
const COLLIDER_CAPSULE: u32 = 3u;
const COLLIDER_SDF: u32 = 4u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One collider slot inside a batch query. `v0`/`v1` carry the primitive data by
// kind: HalfSpace uses v0 = (normal.xyz, d); Sphere uses v0 = (center.xyz,
// radius); Box uses v0 = min.xyz and v1 = max.xyz; Capsule uses v0 = (a.xyz,
// radius) and v1 = b.xyz; Sdf uses v0 = (gradient.xyz, distance).
struct ColliderSlot {
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    v0: vec4<f32>,
    v1: vec4<f32>,
}

// One query. The vec4 lanes carry every routine's typed inputs; a given tag
// reads only the lanes it needs. The std430 field order matches the host
// `GpuQuery` exactly.
struct Query {
    tag: u32,
    contact_hit: u32,
    collider_count: u32,
    pad0: u32,
    point: vec4<f32>,
    seg_a: vec4<f32>,
    seg_b: vec4<f32>,
    sphere: vec4<f32>,
    cap_a: vec4<f32>,
    cap_b: vec4<f32>,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    plane: vec4<f32>,
    sdf: vec4<f32>,
    state_pos: vec4<f32>,
    state_vel: vec4<f32>,
    params: vec4<f32>,
    contact_normal: vec4<f32>,
    contact_point: vec4<f32>,
    swept_p0: vec4<f32>,
    swept_p1: vec4<f32>,
    swept_sphere: vec4<f32>,
    colliders: array<ColliderSlot, 8>,
}

// One result. `scalar.x` holds a signed distance or time of impact; `vector.xyz`
// holds a closest point or a resolved position; `normal.xyz` / `normal.w` hold a
// contact normal and penetration; `point.xyz` holds a contact point; `vel.xyz`
// holds a resolved velocity. `hit` is the discrete contact / sweep flag.
struct Result {
    hit: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    scalar: vec4<f32>,
    vector: vec4<f32>,
    normal: vec4<f32>,
    point: vec4<f32>,
    vel: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// A resolved contact mirroring the reference `Contact`.
struct ContactW {
    hit: bool,
    normal: vec3<f32>,
    penetration: f32,
    point: vec3<f32>,
}

// A resolved impulse response mirroring the reference `CollisionResponse`.
struct ResponseW {
    new_pos: vec3<f32>,
    new_vel: vec3<f32>,
    hit: bool,
}

// Inner product of two 3-vectors, hand-rolled to stay in the explicit core set.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Squared length of a 3-vector; mirrors the reference `length_squared`.
fn len_sq3(v: vec3<f32>) -> f32 {
    return dot3(v, v);
}

// Euclidean length of a 3-vector; mirrors the reference `length`.
fn len3(v: vec3<f32>) -> f32 {
    return sqrt(len_sq3(v));
}

// Unit vector along `v`, or zero when `v` is numerically zero; mirrors the
// reference `normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = len_sq3(v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Unit vector along `v`, falling back to `fallback` when `v` is numerically
// zero; mirrors the reference `safe_normal`.
fn safe_normal(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    if (len_sq3(v) > EPS_LEN_SQ) {
        return normalize_or_zero(v);
    }
    return fallback;
}

// Explicit sign branch replacing the reference `f32::signum`: positive and
// positive-zero map to +1, strictly negative maps to -1. The fixtures keep the
// interior point off the exact box center, where a signed-zero component could
// otherwise differ from the host.
fn signum(x: f32) -> f32 {
    if (x < 0.0) {
        return -1.0;
    }
    return 1.0;
}

// Closest point to `p` on the segment `a..b`; mirrors
// `closest_point_on_segment`.
fn closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let denom = len_sq3(ab);
    if (denom <= EPS_LEN_SQ) {
        return a;
    }
    let t = clamp(dot3(p - a, ab) / denom, 0.0, 1.0);
    return a + ab * t;
}

// Signed distance from `p` to a sphere surface; mirrors `sd_sphere`.
fn sd_sphere(center: vec3<f32>, radius: f32, p: vec3<f32>) -> f32 {
    return len3(p - center) - radius;
}

// Closest point on a sphere surface to `p`; mirrors `closest_point_sphere`.
fn closest_point_sphere(center: vec3<f32>, radius: f32, p: vec3<f32>) -> vec3<f32> {
    let dir = p - center;
    if (len_sq3(dir) <= EPS_LEN_SQ) {
        return center;
    }
    return center + normalize_or_zero(dir) * radius;
}

// Signed distance from `p` to a capsule surface; mirrors `sd_capsule`.
fn sd_capsule(a: vec3<f32>, b: vec3<f32>, radius: f32, p: vec3<f32>) -> f32 {
    let closest = closest_point_on_segment(a, b, p);
    return len3(p - closest) - radius;
}

// Closest point on a capsule surface to `p`; mirrors `closest_point_capsule`.
fn closest_point_capsule(a: vec3<f32>, b: vec3<f32>, radius: f32, p: vec3<f32>) -> vec3<f32> {
    let axis = closest_point_on_segment(a, b, p);
    let dir = p - axis;
    if (len_sq3(dir) <= EPS_LEN_SQ) {
        return axis;
    }
    return axis + normalize_or_zero(dir) * radius;
}

// Closest point to `p` inside or on a box; mirrors `closest_point_aabb`.
fn closest_point_aabb(bmin: vec3<f32>, bmax: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        clamp(p.x, bmin.x, bmax.x),
        clamp(p.y, bmin.y, bmax.y),
        clamp(p.z, bmin.z, bmax.z),
    );
}

// Signed distance from `p` to a box; mirrors `sd_aabb`.
fn sd_aabb(bmin: vec3<f32>, bmax: vec3<f32>, p: vec3<f32>) -> f32 {
    let c = (bmin + bmax) * 0.5;
    let e = (bmax - bmin) * 0.5;
    let q = vec3<f32>(
        abs(p.x - c.x) - e.x,
        abs(p.y - c.y) - e.y,
        abs(p.z - c.z) - e.z,
    );
    let outside = len3(max(q, vec3<f32>(0.0, 0.0, 0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside;
}

// Outward face normal for a point inside a box; mirrors `aabb_interior_normal`.
fn aabb_interior_normal(bmin: vec3<f32>, bmax: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let c = (bmin + bmax) * 0.5;
    let e = (bmax - bmin) * 0.5;
    let dx = abs(p.x - c.x) - e.x;
    let dy = abs(p.y - c.y) - e.y;
    let dz = abs(p.z - c.z) - e.z;
    if (dx >= dy && dx >= dz) {
        return vec3<f32>(signum(p.x - c.x), 0.0, 0.0);
    }
    if (dy >= dz) {
        return vec3<f32>(0.0, signum(p.y - c.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, signum(p.z - c.z));
}

fn contact_miss() -> ContactW {
    var c: ContactW;
    c.hit = false;
    c.normal = vec3<f32>(0.0, 0.0, 0.0);
    c.penetration = 0.0;
    c.point = vec3<f32>(0.0, 0.0, 0.0);
    return c;
}

fn make_contact(normal: vec3<f32>, penetration: f32, point: vec3<f32>) -> ContactW {
    var c: ContactW;
    c.hit = true;
    c.normal = normal;
    c.penetration = penetration;
    c.point = point;
    return c;
}

// Contact of a particle sphere against a half-space; mirrors
// `collide_half_space`.
fn collide_half_space(normal: vec3<f32>, d: f32, pos: vec3<f32>, radius: f32) -> ContactW {
    let sd = dot3(normal, pos) + d;
    let penetration = radius - sd;
    if (penetration <= 0.0) {
        return contact_miss();
    }
    let n = safe_normal(normal, vec3<f32>(0.0, 1.0, 0.0));
    let surface = pos - n * sd;
    return make_contact(n, penetration, surface);
}

// Contact of a particle sphere against a solid sphere; mirrors `collide_sphere`.
fn collide_sphere(center: vec3<f32>, radius_s: f32, pos: vec3<f32>, radius: f32) -> ContactW {
    let sd = sd_sphere(center, radius_s, pos);
    let penetration = radius - sd;
    if (penetration <= 0.0) {
        return contact_miss();
    }
    let n = safe_normal(pos - center, vec3<f32>(0.0, 1.0, 0.0));
    let surface = center + n * radius_s;
    return make_contact(n, penetration, surface);
}

// Contact of a particle sphere against a solid capsule; mirrors
// `collide_capsule`.
fn collide_capsule(a: vec3<f32>, b: vec3<f32>, radius_c: f32, pos: vec3<f32>, radius: f32) -> ContactW {
    let axis = closest_point_on_segment(a, b, pos);
    let sd = len3(pos - axis) - radius_c;
    let penetration = radius - sd;
    if (penetration <= 0.0) {
        return contact_miss();
    }
    let n = safe_normal(pos - axis, vec3<f32>(0.0, 1.0, 0.0));
    let surface = axis + n * radius_c;
    return make_contact(n, penetration, surface);
}

// Contact of a particle sphere against a solid box; mirrors `collide_aabb`.
fn collide_aabb(bmin: vec3<f32>, bmax: vec3<f32>, pos: vec3<f32>, radius: f32) -> ContactW {
    let closest = closest_point_aabb(bmin, bmax, pos);
    let outward = pos - closest;
    if (len_sq3(outward) > EPS_LEN_SQ) {
        let dist = len3(outward);
        let penetration = radius - dist;
        if (penetration <= 0.0) {
            return contact_miss();
        }
        let n = normalize_or_zero(outward);
        return make_contact(n, penetration, closest);
    }
    let sd = sd_aabb(bmin, bmax, pos);
    let penetration = radius - sd;
    let n = safe_normal(aabb_interior_normal(bmin, bmax, pos), vec3<f32>(0.0, 1.0, 0.0));
    let surface = closest_point_aabb(bmin, bmax, pos + n * radius);
    return make_contact(n, penetration, surface);
}

// Contact of a particle sphere against a signed-distance sample; mirrors
// `collide_sdf`.
fn collide_sdf(gradient: vec3<f32>, distance: f32, pos: vec3<f32>, radius: f32) -> ContactW {
    let penetration = radius - distance;
    if (penetration <= 0.0) {
        return contact_miss();
    }
    let n = safe_normal(gradient, vec3<f32>(0.0, 1.0, 0.0));
    let surface = pos - n * distance;
    return make_contact(n, penetration, surface);
}

// Dispatches one collider slot to its predicate; mirrors the reference
// `collide` match over the `Collider` kinds.
fn collide_slot(slot: ColliderSlot, pos: vec3<f32>, radius: f32) -> ContactW {
    if (slot.kind == COLLIDER_HALF_SPACE) {
        return collide_half_space(slot.v0.xyz, slot.v0.w, pos, radius);
    }
    if (slot.kind == COLLIDER_SPHERE) {
        return collide_sphere(slot.v0.xyz, slot.v0.w, pos, radius);
    }
    if (slot.kind == COLLIDER_BOX) {
        return collide_aabb(slot.v0.xyz, slot.v1.xyz, pos, radius);
    }
    if (slot.kind == COLLIDER_CAPSULE) {
        return collide_capsule(slot.v0.xyz, slot.v1.xyz, slot.v0.w, pos, radius);
    }
    return collide_sdf(slot.v0.xyz, slot.v0.w, pos, radius);
}

// Applies a contact to a particle; mirrors `resolve_contact`.
fn resolve_contact(pos: vec3<f32>, vel: vec3<f32>, c: ContactW, restitution: f32, friction: f32) -> ResponseW {
    var out: ResponseW;
    if (!c.hit) {
        out.new_pos = pos;
        out.new_vel = vel;
        out.hit = false;
        return out;
    }
    let normal = c.normal;
    let new_pos = pos + normal * max(c.penetration, 0.0);

    let vn = dot3(vel, normal);
    let vt_vec = vel - normal * vn;
    let vt_speed = len3(vt_vec);

    var vn_after = vn;
    if (vn < 0.0) {
        vn_after = -restitution * vn;
    }

    let normal_impulse = abs(vn_after - vn);
    let friction_delta = friction * normal_impulse;
    let new_vt_speed = max(vt_speed - friction_delta, 0.0);
    let vt_dir = normalize_or_zero(vt_vec);

    out.new_pos = new_pos;
    out.new_vel = normal * vn_after + vt_dir * new_vt_speed;
    out.hit = true;
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var res: Result;
    res.hit = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    res.scalar = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.vector = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.point = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.vel = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let pos = q.state_pos.xyz;
    let radius = q.state_pos.w;
    let vel = q.state_vel.xyz;

    if (q.tag == TAG_SD_SPHERE) {
        res.scalar.x = sd_sphere(q.sphere.xyz, q.sphere.w, q.point.xyz);
    } else if (q.tag == TAG_SD_CAPSULE) {
        res.scalar.x = sd_capsule(q.cap_a.xyz, q.cap_b.xyz, q.cap_a.w, q.point.xyz);
    } else if (q.tag == TAG_SD_AABB) {
        res.scalar.x = sd_aabb(q.box_min.xyz, q.box_max.xyz, q.point.xyz);
    } else if (q.tag == TAG_CLOSEST_SEGMENT) {
        res.vector = vec4<f32>(closest_point_on_segment(q.seg_a.xyz, q.seg_b.xyz, q.point.xyz), 0.0);
    } else if (q.tag == TAG_CLOSEST_SPHERE) {
        res.vector = vec4<f32>(closest_point_sphere(q.sphere.xyz, q.sphere.w, q.point.xyz), 0.0);
    } else if (q.tag == TAG_CLOSEST_CAPSULE) {
        res.vector = vec4<f32>(closest_point_capsule(q.cap_a.xyz, q.cap_b.xyz, q.cap_a.w, q.point.xyz), 0.0);
    } else if (q.tag == TAG_CLOSEST_AABB) {
        res.vector = vec4<f32>(closest_point_aabb(q.box_min.xyz, q.box_max.xyz, q.point.xyz), 0.0);
    } else if (q.tag == TAG_COLLIDE_HALF_SPACE) {
        let c = collide_half_space(q.plane.xyz, q.plane.w, pos, radius);
        write_contact(&res, c);
    } else if (q.tag == TAG_COLLIDE_SPHERE) {
        let c = collide_sphere(q.sphere.xyz, q.sphere.w, pos, radius);
        write_contact(&res, c);
    } else if (q.tag == TAG_COLLIDE_CAPSULE) {
        let c = collide_capsule(q.cap_a.xyz, q.cap_b.xyz, q.cap_a.w, pos, radius);
        write_contact(&res, c);
    } else if (q.tag == TAG_COLLIDE_AABB) {
        let c = collide_aabb(q.box_min.xyz, q.box_max.xyz, pos, radius);
        write_contact(&res, c);
    } else if (q.tag == TAG_COLLIDE_SDF) {
        let c = collide_sdf(q.sdf.xyz, q.sdf.w, pos, radius);
        write_contact(&res, c);
    } else if (q.tag == TAG_RESOLVE_CONTACT) {
        var c: ContactW;
        c.hit = q.contact_hit == 1u;
        c.normal = q.contact_normal.xyz;
        c.penetration = q.contact_normal.w;
        c.point = q.contact_point.xyz;
        let r = resolve_contact(pos, vel, c, q.params.x, q.params.y);
        write_response(&res, r);
    } else if (q.tag == TAG_RESOLVE_PARTICLE) {
        var deepest = contact_miss();
        for (var i = 0u; i < q.collider_count; i = i + 1u) {
            let c = collide_slot(q.colliders[i], pos, radius);
            if (c.hit && c.penetration > deepest.penetration) {
                deepest = c;
            }
        }
        let r = resolve_contact(pos, vel, deepest, q.params.x, q.params.y);
        write_response(&res, r);
    } else if (q.tag == TAG_SWEPT_PLANE) {
        let rad = q.swept_p0.w;
        let d0 = dot3(q.plane.xyz, q.swept_p0.xyz) + q.plane.w;
        let d1 = dot3(q.plane.xyz, q.swept_p1.xyz) + q.plane.w;
        let denom = d1 - d0;
        if (abs(denom) >= EPS && d0 >= rad - EPS) {
            let t = (rad - d0) / denom;
            if (t >= 0.0 && t <= 1.0) {
                res.hit = 1u;
                res.scalar.x = t;
            }
        }
    } else if (q.tag == TAG_SWEPT_SPHERE) {
        let rad = q.swept_p0.w;
        let center = q.swept_sphere.xyz;
        let tr = q.swept_sphere.w;
        let combined = rad + tr;
        let rel = q.swept_p0.xyz - center;
        let disp = q.swept_p1.xyz - q.swept_p0.xyz;
        let cc = len_sq3(rel) - combined * combined;
        if (cc <= 0.0) {
            res.hit = 1u;
            res.scalar.x = 0.0;
        } else {
            let a = len_sq3(disp);
            if (a >= EPS) {
                let b = 2.0 * dot3(rel, disp);
                let disc = b * b - 4.0 * a * cc;
                if (disc >= 0.0) {
                    let t = (-b - sqrt(disc)) / (2.0 * a);
                    if (t >= 0.0 && t <= 1.0) {
                        res.hit = 1u;
                        res.scalar.x = t;
                    }
                }
            }
        }
    }

    results[idx] = res;
}

// Stores a resolved contact into the result lanes.
fn write_contact(res: ptr<function, Result>, c: ContactW) {
    if (c.hit) {
        (*res).hit = 1u;
    } else {
        (*res).hit = 0u;
    }
    (*res).normal = vec4<f32>(c.normal, c.penetration);
    (*res).point = vec4<f32>(c.point, 0.0);
}

// Stores a resolved impulse response into the result lanes.
fn write_response(res: ptr<function, Result>, r: ResponseW) {
    if (r.hit) {
        (*res).hit = 1u;
    } else {
        (*res).hit = 0u;
    }
    (*res).vector = vec4<f32>(r.new_pos, 0.0);
    (*res).vel = vec4<f32>(r.new_vel, 0.0);
}
"#;

/// One query for the collision twin: a tagged union selecting which golden
/// routine to run with its typed inputs.
///
/// Each variant twins exactly one `CPU` golden entry point. The golden contract
/// types ([`Sphere`], [`Capsule`], [`Aabb`], [`Plane`], [`SdfSample`],
/// [`ParticleState`], [`Contact`], [`ResponseParams`] and [`Collider`]) are
/// reused directly rather than re-declared.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::collision`；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub enum CollisionQuery {
    /// Signed distance to a sphere, twinning
    /// [`sd_sphere`](prism_render_architecture::particle::collision::sd_sphere).
    SdSphere {
        /// The sphere collider.
        sphere: Sphere,
        /// Query point.
        point: Vec3,
    },
    /// Signed distance to a capsule, twinning
    /// [`sd_capsule`](prism_render_architecture::particle::collision::sd_capsule).
    SdCapsule {
        /// The capsule collider.
        capsule: Capsule,
        /// Query point.
        point: Vec3,
    },
    /// Signed distance to a box, twinning
    /// [`sd_aabb`](prism_render_architecture::particle::collision::sd_aabb).
    SdAabb {
        /// The box collider.
        box_: Aabb,
        /// Query point.
        point: Vec3,
    },
    /// Closest point on a segment, twinning
    /// [`closest_point_on_segment`](prism_render_architecture::particle::collision::closest_point_on_segment).
    ClosestPointOnSegment {
        /// First segment endpoint.
        a: Vec3,
        /// Second segment endpoint.
        b: Vec3,
        /// Query point.
        point: Vec3,
    },
    /// Closest point on a sphere surface, twinning
    /// [`closest_point_sphere`](prism_render_architecture::particle::collision::closest_point_sphere).
    ClosestPointSphere {
        /// The sphere collider.
        sphere: Sphere,
        /// Query point.
        point: Vec3,
    },
    /// Closest point on a capsule surface, twinning
    /// [`closest_point_capsule`](prism_render_architecture::particle::collision::closest_point_capsule).
    ClosestPointCapsule {
        /// The capsule collider.
        capsule: Capsule,
        /// Query point.
        point: Vec3,
    },
    /// Closest point inside or on a box, twinning
    /// [`closest_point_aabb`](prism_render_architecture::particle::collision::closest_point_aabb).
    ClosestPointAabb {
        /// The box collider.
        box_: Aabb,
        /// Query point.
        point: Vec3,
    },
    /// Half-space contact predicate, twinning
    /// [`collide_half_space`](prism_render_architecture::particle::collision::collide_half_space).
    CollideHalfSpace {
        /// The half-space plane (normal points into free space).
        plane: Plane,
        /// The particle state.
        state: ParticleState,
    },
    /// Sphere contact predicate, twinning
    /// [`collide_sphere`](prism_render_architecture::particle::collision::collide_sphere).
    CollideSphere {
        /// The sphere collider.
        sphere: Sphere,
        /// The particle state.
        state: ParticleState,
    },
    /// Capsule contact predicate, twinning
    /// [`collide_capsule`](prism_render_architecture::particle::collision::collide_capsule).
    CollideCapsule {
        /// The capsule collider.
        capsule: Capsule,
        /// The particle state.
        state: ParticleState,
    },
    /// Box contact predicate, twinning
    /// [`collide_aabb`](prism_render_architecture::particle::collision::collide_aabb).
    CollideAabb {
        /// The box collider.
        box_: Aabb,
        /// The particle state.
        state: ParticleState,
    },
    /// Signed-distance-field contact predicate, twinning
    /// [`collide_sdf`](prism_render_architecture::particle::collision::collide_sdf).
    CollideSdf {
        /// The signed-distance sample at the particle center.
        sample: SdfSample,
        /// The particle state.
        state: ParticleState,
    },
    /// Single-contact impulse solve, twinning
    /// [`resolve_contact`](prism_render_architecture::particle::collision::resolve_contact).
    ResolveContact {
        /// The particle state.
        state: ParticleState,
        /// The contact to resolve.
        contact: Contact,
        /// Restitution and friction coefficients.
        params: ResponseParams,
    },
    /// Deepest-contact batch impulse solve, twinning
    /// [`resolve_particle_collision`](prism_render_architecture::particle::collision::resolve_particle_collision).
    /// The batch carries at most [`MAX_COLLIDERS`] colliders on device; extra
    /// colliders are truncated when the query is encoded.
    ResolveParticleCollision {
        /// The particle state.
        state: ParticleState,
        /// The colliders scanned for the deepest penetration.
        colliders: Vec<Collider>,
        /// Restitution and friction coefficients.
        params: ResponseParams,
    },
    /// Continuous sphere-versus-plane sweep, twinning
    /// [`swept_sphere_vs_plane`](prism_render_architecture::particle::collision::swept_sphere_vs_plane).
    SweptSphereVsPlane {
        /// The half-space plane.
        plane: Plane,
        /// Sweep start center.
        p0: Vec3,
        /// Sweep end center.
        p1: Vec3,
        /// Particle sphere radius.
        radius: f32,
    },
    /// Continuous sphere-versus-sphere sweep, twinning
    /// [`swept_sphere_vs_sphere`](prism_render_architecture::particle::collision::swept_sphere_vs_sphere).
    SweptSphereVsSphere {
        /// Sweep start center.
        p0: Vec3,
        /// Sweep end center.
        p1: Vec3,
        /// Particle sphere radius.
        radius: f32,
        /// The static target sphere.
        target: Sphere,
    },
}

impl CollisionQuery {
    /// Returns the `u32` tag the kernel branches on for this routine.
    #[must_use]
    const fn tag(&self) -> u32 {
        match self {
            CollisionQuery::SdSphere { .. } => 0,
            CollisionQuery::SdCapsule { .. } => 1,
            CollisionQuery::SdAabb { .. } => 2,
            CollisionQuery::ClosestPointOnSegment { .. } => 3,
            CollisionQuery::ClosestPointSphere { .. } => 4,
            CollisionQuery::ClosestPointCapsule { .. } => 5,
            CollisionQuery::ClosestPointAabb { .. } => 6,
            CollisionQuery::CollideHalfSpace { .. } => 7,
            CollisionQuery::CollideSphere { .. } => 8,
            CollisionQuery::CollideCapsule { .. } => 9,
            CollisionQuery::CollideAabb { .. } => 10,
            CollisionQuery::CollideSdf { .. } => 11,
            CollisionQuery::ResolveContact { .. } => 12,
            CollisionQuery::ResolveParticleCollision { .. } => 13,
            CollisionQuery::SweptSphereVsPlane { .. } => 14,
            CollisionQuery::SweptSphereVsSphere { .. } => 15,
        }
    }
}

/// One resolved answer for a single query: a tagged union whose variant matches
/// the routine the corresponding [`CollisionQuery`] selected.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::collision`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CollisionResult {
    /// A scalar result (a signed distance from `sd_*`).
    Scalar(f32),
    /// A `3`-vector result (a closest point from `closest_point_*`).
    Vector(Vec3),
    /// A contact-predicate result (`collide_*`).
    Contact(Contact),
    /// An impulse-response result (`resolve_contact` or
    /// `resolve_particle_collision`).
    Response(CollisionResponse),
    /// A continuous-sweep result (`swept_*`): the hit flag and, when hit, the
    /// time of impact in `0..=1`.
    Swept {
        /// `true` when the sweep touches within the step.
        hit: bool,
        /// Time of impact in `0..=1`; meaningful only when `hit`.
        toi: f32,
    },
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::collision`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn cpu_reference(query: &CollisionQuery) -> CollisionResult {
    match query {
        CollisionQuery::SdSphere { sphere, point } => {
            CollisionResult::Scalar(sd_sphere(*sphere, *point))
        }
        CollisionQuery::SdCapsule { capsule, point } => {
            CollisionResult::Scalar(sd_capsule(*capsule, *point))
        }
        CollisionQuery::SdAabb { box_, point } => CollisionResult::Scalar(sd_aabb(*box_, *point)),
        CollisionQuery::ClosestPointOnSegment { a, b, point } => {
            CollisionResult::Vector(closest_point_on_segment(*a, *b, *point))
        }
        CollisionQuery::ClosestPointSphere { sphere, point } => {
            CollisionResult::Vector(closest_point_sphere(*sphere, *point))
        }
        CollisionQuery::ClosestPointCapsule { capsule, point } => {
            CollisionResult::Vector(closest_point_capsule(*capsule, *point))
        }
        CollisionQuery::ClosestPointAabb { box_, point } => {
            CollisionResult::Vector(closest_point_aabb(*box_, *point))
        }
        CollisionQuery::CollideHalfSpace { plane, state } => {
            CollisionResult::Contact(collide_half_space(*plane, *state))
        }
        CollisionQuery::CollideSphere { sphere, state } => {
            CollisionResult::Contact(collide_sphere(*sphere, *state))
        }
        CollisionQuery::CollideCapsule { capsule, state } => {
            CollisionResult::Contact(collide_capsule(*capsule, *state))
        }
        CollisionQuery::CollideAabb { box_, state } => {
            CollisionResult::Contact(collide_aabb(*box_, *state))
        }
        CollisionQuery::CollideSdf { sample, state } => {
            CollisionResult::Contact(collide_sdf(*sample, *state))
        }
        CollisionQuery::ResolveContact {
            state,
            contact,
            params,
        } => CollisionResult::Response(resolve_contact(*state, *contact, *params)),
        CollisionQuery::ResolveParticleCollision {
            state,
            colliders,
            params,
        } => CollisionResult::Response(resolve_particle_collision(*state, colliders, *params)),
        CollisionQuery::SweptSphereVsPlane {
            plane,
            p0,
            p1,
            radius,
        } => match swept_sphere_vs_plane(*plane, *p0, *p1, *radius) {
            Some(toi) => CollisionResult::Swept { hit: true, toi },
            None => CollisionResult::Swept {
                hit: false,
                toi: 0.0,
            },
        },
        CollisionQuery::SweptSphereVsSphere {
            p0,
            p1,
            radius,
            target,
        } => match swept_sphere_vs_sphere(*p0, *p1, *radius, *target) {
            Some(toi) => CollisionResult::Swept { hit: true, toi },
            None => CollisionResult::Swept {
                hit: false,
                toi: 0.0,
            },
        },
    }
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`COLLISION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one collider slot, matching the `WGSL`
/// `ColliderSlot` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCollider {
    /// Collider kind matching the `WGSL` `COLLIDER_*` codes.
    kind: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Primary data lane (see the `WGSL` `ColliderSlot` doc for the per-kind
    /// packing).
    v0: [f32; 4],
    /// Secondary data lane for the box and capsule kinds.
    v1: [f32; 4],
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every geometric lane is a padded `vec4` so each slot stays `16`-byte aligned
/// on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Routine selector matching the `WGSL` `TAG_*` codes.
    tag: u32,
    /// Input contact hit flag (`1` = hit) for the `resolve_contact` routine.
    contact_hit: u32,
    /// Number of valid colliders for the batch routine.
    collider_count: u32,
    /// Padding word.
    pad0: u32,
    /// Query point (`xyz`; `w` pad).
    point: [f32; 4],
    /// Segment endpoint `a` (`xyz`; `w` pad).
    seg_a: [f32; 4],
    /// Segment endpoint `b` (`xyz`; `w` pad).
    seg_b: [f32; 4],
    /// Sphere `center` (`xyz`) and `radius` (`w`).
    sphere: [f32; 4],
    /// Capsule endpoint `a` (`xyz`) and `radius` (`w`).
    cap_a: [f32; 4],
    /// Capsule endpoint `b` (`xyz`; `w` pad).
    cap_b: [f32; 4],
    /// Box `min` corner (`xyz`; `w` pad).
    box_min: [f32; 4],
    /// Box `max` corner (`xyz`; `w` pad).
    box_max: [f32; 4],
    /// Plane `normal` (`xyz`) and offset `d` (`w`).
    plane: [f32; 4],
    /// Signed-distance `gradient` (`xyz`) and `distance` (`w`).
    sdf: [f32; 4],
    /// Particle `pos` (`xyz`) and collision `radius` (`w`).
    state_pos: [f32; 4],
    /// Particle `vel` (`xyz`; `w` pad).
    state_vel: [f32; 4],
    /// Response `restitution` (`x`) and `friction` (`y`); `zw` pad.
    params: [f32; 4],
    /// Input contact `normal` (`xyz`) and `penetration` (`w`).
    contact_normal: [f32; 4],
    /// Input contact `point` (`xyz`; `w` pad).
    contact_point: [f32; 4],
    /// Sweep start center (`xyz`) and sweep `radius` (`w`).
    swept_p0: [f32; 4],
    /// Sweep end center (`xyz`; `w` pad).
    swept_p1: [f32; 4],
    /// Swept target sphere `center` (`xyz`) and `radius` (`w`).
    swept_sphere: [f32; 4],
    /// Fixed collider batch; only the first `collider_count` slots are valid.
    colliders: [GpuCollider; MAX_COLLIDERS],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Discrete contact / sweep hit flag (`1` = hit).
    hit: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Scalar lane: signed distance or time of impact in `x`.
    scalar: [f32; 4],
    /// Vector lane: closest point or resolved position in `xyz`.
    vector: [f32; 4],
    /// Contact `normal` (`xyz`) and `penetration` (`w`).
    normal: [f32; 4],
    /// Contact `point` (`xyz`; `w` pad).
    point: [f32; 4],
    /// Resolved velocity (`xyz`; `w` pad).
    vel: [f32; 4],
}

/// Packs a `Vec3` and a trailing scalar into a padded `vec4` lane.
fn v4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

/// Encodes one [`Collider`] into its `std430` [`GpuCollider`] slot.
fn encode_collider(c: &Collider) -> GpuCollider {
    let mut g = GpuCollider::zeroed();
    match c {
        Collider::HalfSpace(plane) => {
            g.kind = 0;
            g.v0 = v4(plane.normal, plane.d);
        }
        Collider::Sphere(sphere) => {
            g.kind = 1;
            g.v0 = v4(sphere.center, sphere.radius);
        }
        Collider::Box(box_) => {
            g.kind = 2;
            g.v0 = v4(box_.min, 0.0);
            g.v1 = v4(box_.max, 0.0);
        }
        Collider::Capsule(capsule) => {
            g.kind = 3;
            g.v0 = v4(capsule.a, capsule.radius);
            g.v1 = v4(capsule.b, 0.0);
        }
        Collider::Sdf(sample) => {
            g.kind = 4;
            g.v0 = v4(sample.gradient, sample.distance);
        }
    }
    g
}

/// Encodes one [`CollisionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CollisionQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    g.tag = q.tag();
    match q {
        CollisionQuery::SdSphere { sphere, point }
        | CollisionQuery::ClosestPointSphere { sphere, point } => {
            g.sphere = v4(sphere.center, sphere.radius);
            g.point = v4(*point, 0.0);
        }
        CollisionQuery::SdCapsule { capsule, point }
        | CollisionQuery::ClosestPointCapsule { capsule, point } => {
            g.cap_a = v4(capsule.a, capsule.radius);
            g.cap_b = v4(capsule.b, 0.0);
            g.point = v4(*point, 0.0);
        }
        CollisionQuery::SdAabb { box_, point }
        | CollisionQuery::ClosestPointAabb { box_, point } => {
            g.box_min = v4(box_.min, 0.0);
            g.box_max = v4(box_.max, 0.0);
            g.point = v4(*point, 0.0);
        }
        CollisionQuery::ClosestPointOnSegment { a, b, point } => {
            g.seg_a = v4(*a, 0.0);
            g.seg_b = v4(*b, 0.0);
            g.point = v4(*point, 0.0);
        }
        CollisionQuery::CollideHalfSpace { plane, state } => {
            g.plane = v4(plane.normal, plane.d);
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
        }
        CollisionQuery::CollideSphere { sphere, state } => {
            g.sphere = v4(sphere.center, sphere.radius);
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
        }
        CollisionQuery::CollideCapsule { capsule, state } => {
            g.cap_a = v4(capsule.a, capsule.radius);
            g.cap_b = v4(capsule.b, 0.0);
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
        }
        CollisionQuery::CollideAabb { box_, state } => {
            g.box_min = v4(box_.min, 0.0);
            g.box_max = v4(box_.max, 0.0);
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
        }
        CollisionQuery::CollideSdf { sample, state } => {
            g.sdf = v4(sample.gradient, sample.distance);
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
        }
        CollisionQuery::ResolveContact {
            state,
            contact,
            params,
        } => {
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
            g.params = [params.restitution, params.friction, 0.0, 0.0];
            g.contact_hit = u32::from(contact.hit);
            g.contact_normal = v4(contact.normal, contact.penetration);
            g.contact_point = v4(contact.point, 0.0);
        }
        CollisionQuery::ResolveParticleCollision {
            state,
            colliders,
            params,
        } => {
            g.state_pos = v4(state.pos, state.radius);
            g.state_vel = v4(state.vel, 0.0);
            g.params = [params.restitution, params.friction, 0.0, 0.0];
            let n = colliders.len().min(MAX_COLLIDERS);
            g.collider_count = n as u32;
            for (slot, collider) in g.colliders.iter_mut().zip(colliders.iter()).take(n) {
                *slot = encode_collider(collider);
            }
        }
        CollisionQuery::SweptSphereVsPlane {
            plane,
            p0,
            p1,
            radius,
        } => {
            g.plane = v4(plane.normal, plane.d);
            g.swept_p0 = v4(*p0, *radius);
            g.swept_p1 = v4(*p1, 0.0);
        }
        CollisionQuery::SweptSphereVsSphere {
            p0,
            p1,
            radius,
            target,
        } => {
            g.swept_p0 = v4(*p0, *radius);
            g.swept_p1 = v4(*p1, 0.0);
            g.swept_sphere = v4(target.center, target.radius);
        }
    }
    g
}

/// Decodes one packed [`GpuResult`] into the public [`CollisionResult`],
/// selecting the variant from the query's routine.
fn decode_result(q: &CollisionQuery, raw: &GpuResult) -> CollisionResult {
    match q {
        CollisionQuery::SdSphere { .. }
        | CollisionQuery::SdCapsule { .. }
        | CollisionQuery::SdAabb { .. } => CollisionResult::Scalar(raw.scalar[0]),
        CollisionQuery::ClosestPointOnSegment { .. }
        | CollisionQuery::ClosestPointSphere { .. }
        | CollisionQuery::ClosestPointCapsule { .. }
        | CollisionQuery::ClosestPointAabb { .. } => {
            CollisionResult::Vector(Vec3::new(raw.vector[0], raw.vector[1], raw.vector[2]))
        }
        CollisionQuery::CollideHalfSpace { .. }
        | CollisionQuery::CollideSphere { .. }
        | CollisionQuery::CollideCapsule { .. }
        | CollisionQuery::CollideAabb { .. }
        | CollisionQuery::CollideSdf { .. } => CollisionResult::Contact(if raw.hit == CODE_HIT {
            Contact::hit(
                Vec3::new(raw.normal[0], raw.normal[1], raw.normal[2]),
                raw.normal[3],
                Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
            )
        } else {
            Contact::miss()
        }),
        CollisionQuery::ResolveContact { .. } | CollisionQuery::ResolveParticleCollision { .. } => {
            CollisionResult::Response(CollisionResponse {
                new_pos: Vec3::new(raw.vector[0], raw.vector[1], raw.vector[2]),
                new_vel: Vec3::new(raw.vel[0], raw.vel[1], raw.vel[2]),
                hit: raw.hit == CODE_HIT,
            })
        }
        CollisionQuery::SweptSphereVsPlane { .. } | CollisionQuery::SweptSphereVsSphere { .. } => {
            CollisionResult::Swept {
                hit: raw.hit == CODE_HIT,
                toi: raw.scalar[0],
            }
        }
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compiled, reusable collision compute pipeline, twinning the `CPU` golden
/// [`collision`](prism_render_architecture::particle::collision).
pub struct GpuCollision {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCollision {
    /// Compiles the collision kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCollision {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_collision"),
            source: ShaderSource::Wgsl(COLLISION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_collision_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_collision_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_collision_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCollision {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`CollisionResult`] per
    /// input, in order.
    ///
    /// The result variant matches the routine each query selected, matching the
    /// reference to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[CollisionQuery]) -> Vec<CollisionResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_collision_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_collision_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_collision_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_collision_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_collision_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_collision_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_collision_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
