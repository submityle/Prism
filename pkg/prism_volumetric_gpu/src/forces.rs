//! `wgpu` compute twin of the built-in force library
//! ([`forces`](prism_render_architecture::particle::forces), particle design
//! §8.2, §29).
//!
//! The `CPU` golden
//! [`forces`](prism_render_architecture::particle::forces) owns the deterministic
//! world-space physics fields a particle feels: point / line attractors, radial
//! blasts, orbital motion, softened gravity wells, quadratic drag, layered
//! curl-noise turbulence, and spring-damper anchors. Every one of those is a
//! pure function of a particle `position` and `velocity`, and the module is
//! *deliberately* transcendental-free (it calls no `sin`, `cos`, `acos`, `exp`
//! or `pow`) so a `GPU` kernel can reproduce it. [`GpuForces`] is that on-device
//! twin: one thread evaluates one single-force query, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same
//! accelerations and takes the same degenerate branches the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! A single [`Force::evaluate`](prism_render_architecture::particle::forces::Force::evaluate)
//! acceleration per query, selected by a `u32` classification code
//! ([`FORCE_POINT_ATTRACTOR`] … [`FORCE_SPRING_DAMPER`]), covering the ten
//! physics forces the reference exposes as free functions
//! ([`point_attractor`](prism_render_architecture::particle::forces::point_attractor),
//! [`line_attractor`](prism_render_architecture::particle::forces::line_attractor),
//! [`radial_force`](prism_render_architecture::particle::forces::radial_force),
//! [`explosion`](prism_render_architecture::particle::forces::explosion),
//! [`implosion`](prism_render_architecture::particle::forces::implosion),
//! [`orbital_force`](prism_render_architecture::particle::forces::orbital_force),
//! [`gravity_well`](prism_render_architecture::particle::forces::gravity_well),
//! [`quadratic_drag`](prism_render_architecture::particle::forces::quadratic_drag),
//! [`turbulence`](prism_render_architecture::particle::forces::turbulence) and
//! [`spring_damper`](prism_render_architecture::particle::forces::spring_damper)),
//! each shaped by the same four [`Falloff`](prism_render_architecture::particle::forces::Falloff)
//! curves (coded [`FALLOFF_CONSTANT`] … [`FALLOFF_INVERSE_SQUARE`]). The
//! heterogeneous [`ForceField`](prism_render_architecture::particle::forces::ForceField)
//! accumulator is intentionally *not* twinned: multi-entry summation (with
//! per-entry weight, enable and gate) is a host-side composition over these
//! single-force evaluations, so the device only needs the per-force kernel.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `floor`, `sqrt`, `+ - * /`, `bitcast` and unsigned bit
//! arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` or inverse
//! trigonometry and no optional device feature, mirroring the reference's own
//! determinism rule. The curl-noise hash lattice is reproduced with the same
//! integer bit-mixer the `CPU` reference uses, and the quintic `fade` and
//! `smoothstep` falloff are already spelled as polynomials, so the twin runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides over `f32`, so `CPU` and `GPU` evaluate the same closed form. They
//! are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The integer hash feeding the noise lattice *is* bit-exact (`u32`
//! arithmetic wraps identically), but it is consumed as an `f32` sample and
//! then differenced, so the final acceleration is a continuous quantity. The
//! parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the acceleration channels, tight enough to catch a
//! genuinely wrong port (a dropped branch, a swapped sign, a wrong coefficient)
//! yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`forces`](prism_render_architecture::particle::forces); no third-party
//! engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Classification code for [`Falloff::Constant`](prism_render_architecture::particle::forces::Falloff::Constant):
/// a flat response that is full strength everywhere inside the radius.
pub const FALLOFF_CONSTANT: u32 = 0;
/// Classification code for [`Falloff::Linear`](prism_render_architecture::particle::forces::Falloff::Linear):
/// the ramp `1 - t` reaching zero at the radius.
pub const FALLOFF_LINEAR: u32 = 1;
/// Classification code for [`Falloff::Smoothstep`](prism_render_architecture::particle::forces::Falloff::Smoothstep):
/// the smoothstep ramp with zero slope at both ends.
pub const FALLOFF_SMOOTHSTEP: u32 = 2;
/// Classification code for [`Falloff::InverseSquare`](prism_render_architecture::particle::forces::Falloff::InverseSquare):
/// the windowed inverse-square `1 / (1 + t * t)`.
pub const FALLOFF_INVERSE_SQUARE: u32 = 3;

/// Classification code selecting the
/// [`point_attractor`](prism_render_architecture::particle::forces::point_attractor)
/// force: a shaped, softened pull toward `center`.
pub const FORCE_POINT_ATTRACTOR: u32 = 0;
/// Classification code selecting the
/// [`line_attractor`](prism_render_architecture::particle::forces::line_attractor)
/// force: a shaped pull toward the closest point on a line through `center`
/// along `direction`.
pub const FORCE_LINE_ATTRACTOR: u32 = 1;
/// Classification code selecting the
/// [`radial_force`](prism_render_architecture::particle::forces::radial_force)
/// force: a signed radial blast about `center`.
pub const FORCE_RADIAL: u32 = 2;
/// Classification code selecting the
/// [`explosion`](prism_render_architecture::particle::forces::explosion)
/// wrapper: an outward blast using `strength.abs()`.
pub const FORCE_EXPLOSION: u32 = 3;
/// Classification code selecting the
/// [`implosion`](prism_render_architecture::particle::forces::implosion)
/// wrapper: an inward collapse using `-strength.abs()`.
pub const FORCE_IMPLOSION: u32 = 4;
/// Classification code selecting the
/// [`orbital_force`](prism_render_architecture::particle::forces::orbital_force)
/// force: tangential drive plus a radial spring about the `direction` axis.
pub const FORCE_ORBITAL: u32 = 5;
/// Classification code selecting the
/// [`gravity_well`](prism_render_architecture::particle::forces::gravity_well)
/// force: softened inverse-square gravity toward `center`.
pub const FORCE_GRAVITY_WELL: u32 = 6;
/// Classification code selecting the
/// [`quadratic_drag`](prism_render_architecture::particle::forces::quadratic_drag)
/// force: `-coefficient * |velocity| * velocity`.
pub const FORCE_QUADRATIC_DRAG: u32 = 7;
/// Classification code selecting the
/// [`turbulence`](prism_render_architecture::particle::forces::turbulence)
/// force: a bounded sum of divergence-free curl-noise octaves.
pub const FORCE_TURBULENCE: u32 = 8;
/// Classification code selecting the
/// [`spring_damper`](prism_render_architecture::particle::forces::spring_damper)
/// force: a Hooke spring toward `center` plus viscous damping.
pub const FORCE_SPRING_DAMPER: u32 = 9;

/// The portable core-`WGSL` force kernel, embedded inline so the twin ships as a
/// single source file. The single entry point `evaluate` mirrors the `CPU`
/// golden [`forces`](prism_render_architecture::particle::forces) branch for
/// branch; see the module documentation for the algorithm.
const FORCES_WGSL: &str = r#"
// Force-library twin: one thread per query reproduces a single Force::evaluate
// acceleration, branching on a u32 force code. It mirrors the CPU golden
// particle::forces branch for branch, using only the portable core-WGSL subset
// (min/max/clamp/abs/floor/sqrt, +-*/, bitcast and unsigned bit arithmetic) so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::forces; no third-party
// engine source or derived code.

// Scalar compare / softening epsilon, matching the reference `EPS`.
const EPS: f32 = 1.0e-6;
// Squared-length floor for normalization, matching the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Upper bound on the turbulence octave loop so WGSL has a static trip count;
// fixtures keep `octaves` at or below this value so the twin matches exactly.
const MAX_OCTAVES: u32 = 16u;

const FALLOFF_CONSTANT: u32 = 0u;
const FALLOFF_LINEAR: u32 = 1u;
const FALLOFF_SMOOTHSTEP: u32 = 2u;
const FALLOFF_INVERSE_SQUARE: u32 = 3u;

const FORCE_POINT_ATTRACTOR: u32 = 0u;
const FORCE_LINE_ATTRACTOR: u32 = 1u;
const FORCE_RADIAL: u32 = 2u;
const FORCE_EXPLOSION: u32 = 3u;
const FORCE_IMPLOSION: u32 = 4u;
const FORCE_ORBITAL: u32 = 5u;
const FORCE_GRAVITY_WELL: u32 = 6u;
const FORCE_QUADRATIC_DRAG: u32 = 7u;
const FORCE_TURBULENCE: u32 = 8u;
const FORCE_SPRING_DAMPER: u32 = 9u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle position; a pad lane follows.
    position: vec3<f32>,
    pad0: f32,
    // Particle velocity; a pad lane follows.
    velocity: vec3<f32>,
    pad1: f32,
    // Center point: the attractor point, blast center, well center, orbit
    // center, line point, or spring anchor; a pad lane follows.
    center: vec3<f32>,
    pad2: f32,
    // Direction: the line direction or the orbit axis; a pad lane follows.
    direction: vec3<f32>,
    pad3: f32,
    // Force classification code.
    force_code: u32,
    // Falloff classification code.
    falloff_code: u32,
    // Turbulence base noise seed.
    seed: u32,
    // Turbulence octave count.
    octaves: u32,
    // Signed pull / blast / well strength.
    strength: f32,
    // Influence radius (<= EPS means unbounded).
    radius: f32,
    // Singularity softening length.
    softening: f32,
    // Quadratic drag coefficient.
    coefficient: f32,
    // Orbital target radius.
    target_radius: f32,
    // Orbital along-orbit acceleration.
    tangential_strength: f32,
    // Orbital radial spring stiffness.
    radial_stiffness: f32,
    // Spring stiffness.
    stiffness: f32,
    // Spring viscous damping.
    damping: f32,
    // Turbulence base frequency.
    frequency: f32,
    // Turbulence base amplitude.
    amplitude: f32,
    // Turbulence per-layer frequency multiplier.
    frequency_multiplier: f32,
    // Turbulence per-layer amplitude multiplier.
    amplitude_multiplier: f32,
    // Turbulence central-difference step.
    epsilon: f32,
    // Tail padding so the struct stride is a multiple of 16 bytes.
    tail0: u32,
    tail1: u32,
}

struct Result {
    // Evaluated acceleration; a pad lane follows.
    acceleration: vec3<f32>,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Scalar linear interpolation, matching the reference `lerp`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Unit vector along `v`, or zero when `v` is numerically zero, matching the
// reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Falloff::shape: maps a normalized distance ratio to a multiplier.
fn falloff_shape(code: u32, ratio: f32) -> f32 {
    let t = clamp(ratio, 0.0, 1.0);
    if (code == FALLOFF_LINEAR) {
        return 1.0 - t;
    }
    if (code == FALLOFF_SMOOTHSTEP) {
        let s = 1.0 - t;
        return s * s * (3.0 - 2.0 * s);
    }
    if (code == FALLOFF_INVERSE_SQUARE) {
        return 1.0 / (1.0 + t * t);
    }
    // FALLOFF_CONSTANT and any unknown code: flat full strength.
    return 1.0;
}

// softened_direction: xyz is the softened unit direction from origin to needle,
// w is the true (un-softened) distance. Mirrors the reference helper.
fn softened_direction(origin: vec3<f32>, needle: vec3<f32>, softening: f32) -> vec4<f32> {
    let to = needle - origin;
    let dist_sq = dot(to, to);
    let soft = max(softening, 0.0);
    let denom = dist_sq + soft * soft + EPS * EPS;
    let inv = 1.0 / sqrt(denom);
    return vec4<f32>(to * inv, sqrt(dist_sq));
}

// outside_radius: a positive radius rejects points strictly beyond it.
fn outside_radius(dist: f32, radius: f32) -> bool {
    return radius > EPS && dist > radius;
}

// point_attractor: shaped, softened pull toward `needle`.
fn point_attractor(
    position: vec3<f32>,
    needle: vec3<f32>,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff_code: u32,
) -> vec3<f32> {
    let sd = softened_direction(position, needle, softening);
    let dir = sd.xyz;
    let dist = sd.w;
    if (outside_radius(dist, radius)) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    var ratio = 0.0;
    if (radius > EPS) {
        ratio = dist / radius;
    }
    return dir * (strength * falloff_shape(falloff_code, ratio));
}

// line_attractor: shaped pull toward the closest point on an infinite line.
fn line_attractor(
    position: vec3<f32>,
    line_point: vec3<f32>,
    line_direction: vec3<f32>,
    strength: f32,
    radius: f32,
    softening: f32,
    falloff_code: u32,
) -> vec3<f32> {
    let axis = normalize_or_zero(line_direction);
    if (dot(axis, axis) <= EPS * EPS) {
        return point_attractor(position, line_point, strength, radius, softening, falloff_code);
    }
    let offset = position - line_point;
    let along = axis * dot(offset, axis);
    let closest = line_point + along;
    return point_attractor(position, closest, strength, radius, softening, falloff_code);
}

// gravity_well: softened inverse-square acceleration toward `center`.
fn gravity_well(position: vec3<f32>, center: vec3<f32>, strength: f32, softening: f32) -> vec3<f32> {
    let to = center - position;
    let dist_sq = dot(to, to);
    let soft = max(softening, 0.0);
    let denom = dist_sq + soft * soft + EPS * EPS;
    let dir = normalize_or_zero(to);
    return dir * (strength / denom);
}

// orbital_force: tangential drive plus a radial spring holding `target_radius`.
fn orbital_force(
    position: vec3<f32>,
    center: vec3<f32>,
    axis_in: vec3<f32>,
    target_radius: f32,
    tangential_strength: f32,
    radial_stiffness: f32,
) -> vec3<f32> {
    let unit_axis = normalize_or_zero(axis_in);
    if (dot(unit_axis, unit_axis) <= EPS * EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let radial = position - center;
    let axial = unit_axis * dot(radial, unit_axis);
    let perp = radial - axial;
    let dist_sq = dot(perp, perp);
    if (dist_sq <= EPS * EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let dist = sqrt(dist_sq);
    let perp_unit = normalize_or_zero(perp);
    let tangent = normalize_or_zero(cross(unit_axis, perp_unit));
    let tangential = tangent * tangential_strength;
    let radial_error = dist - target_radius;
    let correction = perp_unit * (-radial_stiffness * radial_error);
    return tangential + correction;
}

// quadratic_drag: -coefficient * |velocity| * velocity.
fn quadratic_drag(velocity: vec3<f32>, coefficient: f32) -> vec3<f32> {
    if (coefficient <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let speed = sqrt(dot(velocity, velocity));
    if (speed <= EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return velocity * (-coefficient * speed);
}

// spring_damper: Hooke spring toward `anchor` plus viscous damping.
fn spring_damper(
    position: vec3<f32>,
    velocity: vec3<f32>,
    anchor: vec3<f32>,
    stiffness: f32,
    damping: f32,
) -> vec3<f32> {
    let restoring = (anchor - position) * stiffness;
    let resistance = velocity * (-damping);
    return restoring + resistance;
}

// Integer bit-mixer (Wang/murmur finalizer), matching the reference `mix32`.
fn mix32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Stateless hash RNG, matching the reference `hash_rng`.
fn hash_rng(id: u32, seed: u32, stream: u32, frame: u32) -> u32 {
    var h = mix32(seed ^ 0x9e3779b9u);
    h = mix32(h ^ (id * 0x85ebca6bu));
    h = mix32(h ^ (stream * 0xc2b2ae35u));
    return mix32(h ^ (frame * 0x27d4eb2fu));
}

// Maps a hash to [0, 1) via the high 24 bits, matching the reference `unit_f32`.
fn unit_f32(hash: u32) -> f32 {
    return f32(hash >> 8u) / 16777216.0;
}

// Quintic fade 6t^5 - 15t^4 + 10t^3, matching the reference `fade`.
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Signed lattice value in [-1, 1], matching the reference `lattice_value`.
// Negative lattice coordinates are reinterpreted as u32 exactly like Rust's
// `as u32` 2's-complement wrap via `bitcast`.
fn lattice_value(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let h = hash_rng(bitcast<u32>(ix), seed, bitcast<u32>(iy), bitcast<u32>(iz));
    return unit_f32(h) * 2.0 - 1.0;
}

// Scalar value noise in [-1, 1], matching the reference `value_noise`.
fn value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let x0 = floor(p.x);
    let y0 = floor(p.y);
    let z0 = floor(p.z);
    let ix = i32(x0);
    let iy = i32(y0);
    let iz = i32(z0);
    let u = fade(p.x - x0);
    let v = fade(p.y - y0);
    let w = fade(p.z - z0);
    let c000 = lattice_value(ix, iy, iz, seed);
    let c100 = lattice_value(ix + 1, iy, iz, seed);
    let c010 = lattice_value(ix, iy + 1, iz, seed);
    let c110 = lattice_value(ix + 1, iy + 1, iz, seed);
    let c001 = lattice_value(ix, iy, iz + 1, seed);
    let c101 = lattice_value(ix + 1, iy, iz + 1, seed);
    let c011 = lattice_value(ix, iy + 1, iz + 1, seed);
    let c111 = lattice_value(ix + 1, iy + 1, iz + 1, seed);
    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    let y0v = lerp(x00, x10, v);
    let y1v = lerp(x01, x11, v);
    return lerp(y0v, y1v, w);
}

// Divergence-free curl noise, matching the reference `curl_noise`.
fn curl_noise(position: vec3<f32>, seed: u32, epsilon: f32) -> vec3<f32> {
    var eps = 1.0e-3;
    if (epsilon > 0.0) {
        eps = epsilon;
    }
    let inv = 1.0 / (2.0 * eps);
    let s1 = seed;
    let s2 = seed + 0x10000001u;
    let s3 = seed + 0x20000002u;
    let dx = vec3<f32>(eps, 0.0, 0.0);
    let dy = vec3<f32>(0.0, eps, 0.0);
    let dz = vec3<f32>(0.0, 0.0, eps);
    let dp3_dy = (value_noise(position + dy, s3) - value_noise(position - dy, s3)) * inv;
    let dp2_dz = (value_noise(position + dz, s2) - value_noise(position - dz, s2)) * inv;
    let dp1_dz = (value_noise(position + dz, s1) - value_noise(position - dz, s1)) * inv;
    let dp3_dx = (value_noise(position + dx, s3) - value_noise(position - dx, s3)) * inv;
    let dp2_dx = (value_noise(position + dx, s2) - value_noise(position - dx, s2)) * inv;
    let dp1_dy = (value_noise(position + dy, s1) - value_noise(position - dy, s1)) * inv;
    return vec3<f32>(dp3_dy - dp2_dz, dp1_dz - dp3_dx, dp2_dx - dp1_dy);
}

// turbulence: bounded sum of curl-noise octaves, matching the reference.
fn turbulence(
    position: vec3<f32>,
    seed: u32,
    frequency: f32,
    amplitude: f32,
    octaves: u32,
    frequency_multiplier: f32,
    amplitude_multiplier: f32,
    epsilon: f32,
) -> vec3<f32> {
    var total = vec3<f32>(0.0, 0.0, 0.0);
    var freq = frequency;
    var amp = amplitude;
    var octave = 0u;
    loop {
        if (octave >= octaves) {
            break;
        }
        if (octave >= MAX_OCTAVES) {
            break;
        }
        let sample_pos = position * freq;
        let layer_seed = seed + (octave * 0x01000193u);
        let value = curl_noise(sample_pos, layer_seed, epsilon);
        total = total + value * amp;
        freq = freq * frequency_multiplier;
        amp = amp * amplitude_multiplier;
        octave = octave + 1u;
    }
    return total;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let position = q.position;
    let velocity = q.velocity;
    let center = q.center;
    let direction = q.direction;
    let code = q.force_code;

    var accel = vec3<f32>(0.0, 0.0, 0.0);
    if (code == FORCE_POINT_ATTRACTOR) {
        accel = point_attractor(position, center, q.strength, q.radius, q.softening, q.falloff_code);
    } else if (code == FORCE_LINE_ATTRACTOR) {
        accel = line_attractor(
            position, center, direction, q.strength, q.radius, q.softening, q.falloff_code,
        );
    } else if (code == FORCE_RADIAL) {
        // radial_force = point_attractor with the pull reversed.
        accel = point_attractor(position, center, -q.strength, q.radius, q.softening, q.falloff_code);
    } else if (code == FORCE_EXPLOSION) {
        // explosion = radial_force(strength.abs()) = point_attractor(-|strength|).
        accel = point_attractor(
            position, center, -abs(q.strength), q.radius, q.softening, q.falloff_code,
        );
    } else if (code == FORCE_IMPLOSION) {
        // implosion = radial_force(-strength.abs()) = point_attractor(+|strength|).
        accel = point_attractor(
            position, center, abs(q.strength), q.radius, q.softening, q.falloff_code,
        );
    } else if (code == FORCE_ORBITAL) {
        accel = orbital_force(
            position, center, direction, q.target_radius, q.tangential_strength, q.radial_stiffness,
        );
    } else if (code == FORCE_GRAVITY_WELL) {
        accel = gravity_well(position, center, q.strength, q.softening);
    } else if (code == FORCE_QUADRATIC_DRAG) {
        accel = quadratic_drag(velocity, q.coefficient);
    } else if (code == FORCE_TURBULENCE) {
        accel = turbulence(
            position,
            q.seed,
            q.frequency,
            q.amplitude,
            q.octaves,
            q.frequency_multiplier,
            q.amplitude_multiplier,
            q.epsilon,
        );
    } else if (code == FORCE_SPRING_DAMPER) {
        accel = spring_damper(position, velocity, center, q.stiffness, q.damping);
    }

    var out: Result;
    out.acceleration = accel;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// One single-force evaluation query: a particle `position` and `velocity`, a
/// force classification code, and every parameter the ten twinned forces
/// consume.
///
/// `center` plays the role of the attractor point, blast center, well center,
/// orbit center, line point, or spring anchor, depending on `force_code`.
/// `direction` is the line direction (for
/// [`FORCE_LINE_ATTRACTOR`]) or the orbit axis (for [`FORCE_ORBITAL`]).
/// Parameters a given force does not read are ignored by that force, exactly as
/// the reference free functions ignore unrelated arguments.
///
/// Provenance: twinned from this repository's
/// [`forces`](prism_render_architecture::particle::forces); no third-party
/// engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuForcesQuery {
    /// The particle position sampled by every position-dependent force.
    pub position: Vec3,
    /// The particle velocity consumed by the drag and spring-damper forces.
    pub velocity: Vec3,
    /// The force center / point / line point / anchor, by `force_code`.
    pub center: Vec3,
    /// The line direction or orbit axis, by `force_code`.
    pub direction: Vec3,
    /// The force classification code, one of [`FORCE_POINT_ATTRACTOR`] …
    /// [`FORCE_SPRING_DAMPER`].
    pub force_code: u32,
    /// The falloff classification code, one of [`FALLOFF_CONSTANT`] …
    /// [`FALLOFF_INVERSE_SQUARE`].
    pub falloff_code: u32,
    /// The turbulence base noise seed.
    pub seed: u32,
    /// The turbulence octave count (kept at or below `16`).
    pub octaves: u32,
    /// The signed pull / blast / well strength.
    pub strength: f32,
    /// The influence radius (`<= 1e-6` means unbounded).
    pub radius: f32,
    /// The singularity softening length.
    pub softening: f32,
    /// The quadratic-drag coefficient.
    pub coefficient: f32,
    /// The orbital target radius held by the radial spring.
    pub target_radius: f32,
    /// The orbital along-orbit (tangential) acceleration.
    pub tangential_strength: f32,
    /// The orbital radial spring stiffness.
    pub radial_stiffness: f32,
    /// The spring-damper stiffness.
    pub stiffness: f32,
    /// The spring-damper viscous damping.
    pub damping: f32,
    /// The turbulence base sampling frequency.
    pub frequency: f32,
    /// The turbulence base layer amplitude.
    pub amplitude: f32,
    /// The turbulence per-layer frequency multiplier.
    pub frequency_multiplier: f32,
    /// The turbulence per-layer amplitude multiplier.
    pub amplitude_multiplier: f32,
    /// The turbulence central-difference step for the curl.
    pub epsilon: f32,
}

/// The resolved acceleration for one query, matching the acceleration the
/// reference [`Force::evaluate`](prism_render_architecture::particle::forces::Force::evaluate)
/// returns for the same force.
///
/// Provenance: twinned from this repository's
/// [`forces`](prism_render_architecture::particle::forces); no third-party
/// engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuForcesResult {
    /// The evaluated world-space acceleration.
    pub acceleration: Vec3,
}

/// `repr(C)` `std430` layout of one packed query: four `16`-byte-aligned
/// `vec3` slots (position, velocity, center, direction) followed by the flat
/// scalar block, padded with two tail words so the array stride is `144` bytes,
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle position.
    position: [f32; 3],
    /// Padding lane after the position.
    pad0: f32,
    /// Particle velocity.
    velocity: [f32; 3],
    /// Padding lane after the velocity.
    pad1: f32,
    /// Force center / point / line point / anchor.
    center: [f32; 3],
    /// Padding lane after the center.
    pad2: f32,
    /// Line direction or orbit axis.
    direction: [f32; 3],
    /// Padding lane after the direction.
    pad3: f32,
    /// Force classification code.
    force_code: u32,
    /// Falloff classification code.
    falloff_code: u32,
    /// Turbulence base noise seed.
    seed: u32,
    /// Turbulence octave count.
    octaves: u32,
    /// Signed pull / blast / well strength.
    strength: f32,
    /// Influence radius.
    radius: f32,
    /// Singularity softening length.
    softening: f32,
    /// Quadratic drag coefficient.
    coefficient: f32,
    /// Orbital target radius.
    target_radius: f32,
    /// Orbital tangential acceleration.
    tangential_strength: f32,
    /// Orbital radial spring stiffness.
    radial_stiffness: f32,
    /// Spring stiffness.
    stiffness: f32,
    /// Spring viscous damping.
    damping: f32,
    /// Turbulence base frequency.
    frequency: f32,
    /// Turbulence base amplitude.
    amplitude: f32,
    /// Turbulence per-layer frequency multiplier.
    frequency_multiplier: f32,
    /// Turbulence per-layer amplitude multiplier.
    amplitude_multiplier: f32,
    /// Turbulence central-difference step.
    epsilon: f32,
    /// Tail padding word.
    tail0: u32,
    /// Tail padding word.
    tail1: u32,
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &GpuForcesQuery) -> GpuQuery {
        GpuQuery {
            position: [query.position.x, query.position.y, query.position.z],
            pad0: 0.0,
            velocity: [query.velocity.x, query.velocity.y, query.velocity.z],
            pad1: 0.0,
            center: [query.center.x, query.center.y, query.center.z],
            pad2: 0.0,
            direction: [query.direction.x, query.direction.y, query.direction.z],
            pad3: 0.0,
            force_code: query.force_code,
            falloff_code: query.falloff_code,
            seed: query.seed,
            octaves: query.octaves,
            strength: query.strength,
            radius: query.radius,
            softening: query.softening,
            coefficient: query.coefficient,
            target_radius: query.target_radius,
            tangential_strength: query.tangential_strength,
            radial_stiffness: query.radial_stiffness,
            stiffness: query.stiffness,
            damping: query.damping,
            frequency: query.frequency,
            amplitude: query.amplitude,
            frequency_multiplier: query.frequency_multiplier,
            amplitude_multiplier: query.amplitude_multiplier,
            epsilon: query.epsilon,
            tail0: 0,
            tail1: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a single `16`-byte-aligned `vec3`
/// acceleration slot with one pad lane, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Evaluated acceleration.
    acceleration: [f32; 3],
    /// Padding lane after the acceleration.
    pad0: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable force-evaluation compute pipeline.
pub struct GpuForces {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuForces {
    /// Compiles the force kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuForces {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_forces"),
            source: ShaderSource::Wgsl(FORCES_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_forces_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_forces_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_forces_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuForces {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one [`GpuForcesResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`Force::evaluate`](prism_render_architecture::particle::forces::Force::evaluate)
    /// acceleration for the same force to within the tolerance documented on
    /// this module. An empty input returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuForcesQuery]) -> Vec<GpuForcesResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_forces_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_forces_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_forces_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_forces_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_forces_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_forces_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_forces_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuForcesResult`].
fn decode_result(raw: &GpuResult) -> GpuForcesResult {
    GpuForcesResult {
        acceleration: Vec3::new(
            raw.acceleration[0],
            raw.acceleration[1],
            raw.acceleration[2],
        ),
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
