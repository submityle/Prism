//! `wgpu` compute twin of the ray-traced particle-collision and lit-sampling
//! golden
//! ([`raytrace`](prism_render_architecture::particle::raytrace), design §22).
//!
//! The `CPU` golden owns the per-particle *ray-trace resolve* math: the mirror
//! [`reflect`](prism_render_architecture::particle::raytrace::reflect) of a
//! velocity about a surface normal, the bounce-plus-friction collision response
//! [`apply_collision`](prism_render_architecture::particle::raytrace::apply_collision)
//! (which flips the normal for a back-face hit via
//! [`RayHit::oriented_normal`](prism_render_architecture::particle::raytrace::RayHit)
//! and delegates to
//! [`resolve_bounce`](prism_render_architecture::particle::raytrace::resolve_bounce)),
//! the hardware/quality collision-method chooser
//! [`choose_collision_method`](prism_render_architecture::particle::raytrace::choose_collision_method)
//! and the ray-traced lighting evaluation
//! [`evaluate_lit_sample`](prism_render_architecture::particle::raytrace::evaluate_lit_sample).
//!
//! [`GpuRaytrace`] is the on-device twin: one thread resolves one particle,
//! reproducing every one of those answers, so a passing real-device parity test
//! is direct evidence the ported kernel folds the same hand-rolled vector
//! algebra, the same zero-guards and the same discrete classification the
//! reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Per query the kernel reproduces: the mirror `reflected` velocity
//! (`v - 2 (v·n) n`); the `bounced` velocity after the full restitution and
//! friction response, including the back-face normal flip; the chosen
//! `method_code` and its `uses_ray_tracing` flag from the hardware capabilities
//! and the requested quality tier; and the lit sample's clamped `visibility`,
//! its `irradiance` (`visibility * max(N·L, 0) * radiance`, isotropic `N·L = 1`
//! on a zero normal) and the gated `GI` `bounce` colour.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `dot`, `+ - * /` and one `sqrt` inside each robust normalize — with no
//! `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it runs
//! unmodified on Metal, Vulkan and DX12. The golden is likewise transcendental
//! free (only `sqrt`), so the two evaluate the same closed form.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable chain of guarded vector operations, so
//! `CPU` and `GPU` evaluate the same algebra in the same associativity. They are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the continuous vectors and scalars yet an *exact*
//! match on the discrete `method_code`, `uses_ray_tracing` flag and the
//! separating-versus-approaching branch, which are driven by the same guard
//! bands the reference uses so a query placed clear of a boundary folds the
//! identical verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! [`raytrace`](prism_render_architecture::particle::raytrace); hand-rolled
//! particle-collision and lit-sampling algebra plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::raytrace::{
    apply_collision, choose_collision_method, evaluate_lit_sample, reflect, CollisionMethod,
    CollisionQuality, CollisionResponse, HardwareCaps, LitSampleRequest, RayHit,
};
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

/// Quality-tier code for the best-available path, mirroring
/// [`CollisionQuality::High`](prism_render_architecture::particle::raytrace::CollisionQuality).
pub const QUALITY_HIGH: u32 = 0;
/// Quality-tier code for the balanced path, mirroring
/// [`CollisionQuality::Medium`](prism_render_architecture::particle::raytrace::CollisionQuality).
pub const QUALITY_MEDIUM: u32 = 1;
/// Quality-tier code for the cheapest path, mirroring
/// [`CollisionQuality::Low`](prism_render_architecture::particle::raytrace::CollisionQuality).
pub const QUALITY_LOW: u32 = 2;

/// Collision-method code for hardware ray tracing, mirroring
/// [`CollisionMethod::Raytrace`](prism_render_architecture::particle::raytrace::CollisionMethod).
pub const METHOD_RAYTRACE: u32 = 0;
/// Collision-method code for the signed-distance-field query, mirroring
/// [`CollisionMethod::Sdf`](prism_render_architecture::particle::raytrace::CollisionMethod).
pub const METHOD_SDF: u32 = 1;
/// Collision-method code for the depth-buffer fallback, mirroring
/// [`CollisionMethod::DepthBuffer`](prism_render_architecture::particle::raytrace::CollisionMethod).
pub const METHOD_DEPTH_BUFFER: u32 = 2;

/// The portable core-`WGSL` ray-trace-resolve kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`raytrace`](prism_render_architecture::particle::raytrace) functions; see
/// the module documentation for the algorithm.
const RAYTRACE_WGSL: &str = r#"
// Per-particle ray-trace-resolve twin: one thread per query reproduces the
// mirror reflection, the restitution-plus-friction bounce response (with the
// back-face normal flip), the hardware/quality collision-method choice and the
// ray-traced lit sample. It mirrors the CPU golden particle::raytrace function
// for function.
//
// Portability: only the core subset (clamp, min, max, dot, + - * / and one
// sqrt) is used; no sin/cos/exp/log/pow/tan and no optional device feature, so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::raytrace; no third-party
// engine source or derived code.

// Squared-length floor guarding every normalize divide and classifying a (near)
// zero vector, matching the reference EPS_LEN_SQ. A direct f32 ==/!= is
// forbidden, so the degenerate tests compare the squared length against this
// floor instead of exact zero.
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle velocity fed to reflect and the bounce response.
    velocity: vec3<f32>,
    pad0: f32,
    // Geometric hit normal; reflect uses it as-is, the bounce orients it.
    hit_normal: vec3<f32>,
    pad1: f32,
    // Lit-sample surface normal; a zero vector marks an isotropic sample.
    light_normal: vec3<f32>,
    pad2: f32,
    // Direction toward the light for the lit sample.
    to_light: vec3<f32>,
    pad3: f32,
    // Incoming light radiance (colour times intensity).
    light_radiance: vec3<f32>,
    pad4: f32,
    // Gathered GI bounce colour, passed through only when sample_gi is set.
    raw_bounce: vec3<f32>,
    pad5: f32,
    // Coefficient of restitution for the bounce (clamped to 0..=1).
    restitution: f32,
    // Tangential friction for the bounce (clamped to 0..=1).
    friction: f32,
    // Raw shadow-ray visibility for the lit sample (clamped to 0..=1).
    raw_visibility: f32,
    pad6: f32,
    // 1 when the ray struck a back face (the geometric normal is flipped).
    back_face: u32,
    // 1 when hardware ray tracing is available.
    ray_tracing: u32,
    // 1 when a scene SDF volume is available.
    sdf_volume: u32,
    // 1 when the lit sample should gather a GI bounce colour.
    sample_gi: u32,
    // Requested quality tier: 0 High, 1 Medium, 2 Low.
    quality: u32,
    pad7: u32,
    pad8: u32,
    pad9: u32,
}

struct Result {
    // Mirror reflection of the velocity about the raw hit normal.
    reflected: vec3<f32>,
    pad0: f32,
    // Velocity after the restitution-plus-friction bounce response.
    bounced: vec3<f32>,
    pad1: f32,
    // Approximate incident irradiance of the lit sample.
    irradiance: vec3<f32>,
    pad2: f32,
    // GI bounce colour of the lit sample (zero when sample_gi is false).
    bounce: vec3<f32>,
    pad3: f32,
    // Clamped shadow-ray visibility.
    visibility: f32,
    pad4: f32,
    pad5: f32,
    pad6: f32,
    // Chosen collision-method code: 0 Raytrace, 1 Sdf, 2 DepthBuffer.
    method_code: u32,
    // 1 when the chosen method uses ray-tracing hardware.
    uses_ray_tracing: u32,
    pad7: u32,
    pad8: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns the unit vector along v, or the zero vector when v is (numerically)
// zero, mirroring the reference normalize_or_zero: a squared length at or below
// EPS_LEN_SQ yields zero so the divide never produces a NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Mirror reflection v - 2 (v·n) n about a robustly normalized normal, mirroring
// the reference reflect: a (near) zero normal is a no-op returning v.
fn reflect_velocity(velocity: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let n = normalize_or_zero(normal);
    if (dot(n, n) <= EPS_LEN_SQ) {
        return velocity;
    }
    let vn = dot(velocity, n);
    return velocity - n * (2.0 * vn);
}

// Restitution-plus-friction bounce about a robustly normalized normal,
// mirroring the reference resolve_bounce: a (near) zero normal or a separating
// velocity (v·n >= 0) is a no-op, otherwise the normal component is reversed and
// scaled by the clamped restitution while the tangential component is damped by
// the clamped friction.
fn resolve_bounce(
    velocity: vec3<f32>,
    normal: vec3<f32>,
    restitution: f32,
    friction: f32,
) -> vec3<f32> {
    let n = normalize_or_zero(normal);
    if (dot(n, n) <= EPS_LEN_SQ) {
        return velocity;
    }
    let vn_scalar = dot(velocity, n);
    if (vn_scalar >= 0.0) {
        return velocity;
    }
    let v_n = n * vn_scalar;
    let v_t = velocity - v_n;
    let e = clamp(restitution, 0.0, 1.0);
    let f = clamp(friction, 0.0, 1.0);
    return v_t * (1.0 - f) - v_n * e;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // reflect: about the raw geometric normal.
    let reflected = reflect_velocity(q.velocity, q.hit_normal);

    // apply_collision: flip the normal for a back-face hit, then bounce.
    var oriented = q.hit_normal;
    if (q.back_face != 0u) {
        oriented = q.hit_normal * -1.0;
    }
    let bounced = resolve_bounce(q.velocity, oriented, q.restitution, q.friction);

    // choose_collision_method: the quality ladder degrading to a runnable path.
    // 0 High / 1 Medium / 2 Low -> 0 Raytrace / 1 Sdf / 2 DepthBuffer.
    var method_code: u32 = 2u;
    if (q.quality == 0u) {
        if (q.ray_tracing != 0u) {
            method_code = 0u;
        } else if (q.sdf_volume != 0u) {
            method_code = 1u;
        } else {
            method_code = 2u;
        }
    } else if (q.quality == 1u) {
        if (q.sdf_volume != 0u) {
            method_code = 1u;
        } else {
            method_code = 2u;
        }
    } else {
        method_code = 2u;
    }
    var uses_ray_tracing: u32 = 0u;
    if (method_code == 0u) {
        uses_ray_tracing = 1u;
    }

    // evaluate_lit_sample: clamp visibility, isotropic N·L on a zero normal.
    let visibility = clamp(q.raw_visibility, 0.0, 1.0);
    let n = normalize_or_zero(q.light_normal);
    let l = normalize_or_zero(q.to_light);
    var n_dot_l: f32 = 1.0;
    if (dot(n, n) > EPS_LEN_SQ) {
        n_dot_l = max(dot(n, l), 0.0);
    }
    let irradiance = q.light_radiance * (visibility * n_dot_l);
    var bounce = vec3<f32>(0.0, 0.0, 0.0);
    if (q.sample_gi != 0u) {
        bounce = q.raw_bounce;
    }

    var out: Result;
    out.reflected = reflected;
    out.pad0 = 0.0;
    out.bounced = bounced;
    out.pad1 = 0.0;
    out.irradiance = irradiance;
    out.pad2 = 0.0;
    out.bounce = bounce;
    out.pad3 = 0.0;
    out.visibility = visibility;
    out.pad4 = 0.0;
    out.pad5 = 0.0;
    out.pad6 = 0.0;
    out.method_code = method_code;
    out.uses_ray_tracing = uses_ray_tracing;
    out.pad7 = 0u;
    out.pad8 = 0u;
    results[idx] = out;
}
"#;

/// One per-particle ray-trace-resolve query: the collision inputs, the hardware
/// and quality selectors and the lit-sample inputs the reference's twinned
/// functions consume.
///
/// `velocity` and `hit_normal` drive both the mirror
/// [`reflect`](prism_render_architecture::particle::raytrace::reflect) and the
/// bounce
/// [`apply_collision`](prism_render_architecture::particle::raytrace::apply_collision),
/// with `back_face` requesting the back-face normal flip and `restitution` /
/// `friction` the response coefficients (both clamped to `0..=1`). `ray_tracing`
/// / `sdf_volume` are the hardware capabilities and `quality` the requested tier
/// (`QUALITY_HIGH` / `QUALITY_MEDIUM` / `QUALITY_LOW`) fed to
/// [`choose_collision_method`](prism_render_architecture::particle::raytrace::choose_collision_method).
/// `light_normal` / `to_light` / `light_radiance` / `raw_visibility` /
/// `raw_bounce` / `sample_gi` feed
/// [`evaluate_lit_sample`](prism_render_architecture::particle::raytrace::evaluate_lit_sample).
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRaytraceQuery {
    /// Particle velocity fed to the reflection and the bounce response.
    pub velocity: [f32; 3],
    /// Geometric hit normal; `reflect` uses it directly, the bounce orients it.
    pub hit_normal: [f32; 3],
    /// Lit-sample surface normal; a zero vector marks an isotropic sample.
    pub light_normal: [f32; 3],
    /// Direction toward the light for the lit sample.
    pub to_light: [f32; 3],
    /// Incoming light radiance (colour times intensity).
    pub light_radiance: [f32; 3],
    /// Gathered `GI` bounce colour, passed through only when `sample_gi`.
    pub raw_bounce: [f32; 3],
    /// Coefficient of restitution for the bounce (clamped to `0..=1`).
    pub restitution: f32,
    /// Tangential friction for the bounce (clamped to `0..=1`).
    pub friction: f32,
    /// Raw shadow-ray visibility for the lit sample (clamped to `0..=1`).
    pub raw_visibility: f32,
    /// Whether the ray struck a back face (the geometric normal is flipped).
    pub back_face: bool,
    /// Whether hardware ray tracing is available.
    pub ray_tracing: bool,
    /// Whether a scene `SDF` volume is available.
    pub sdf_volume: bool,
    /// Whether the lit sample should gather a `GI` bounce colour.
    pub sample_gi: bool,
    /// Requested quality tier (`QUALITY_HIGH` / `QUALITY_MEDIUM` /
    /// `QUALITY_LOW`).
    pub quality: u32,
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its twinned functions.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRaytraceResult {
    /// Mirror reflection of the velocity, matching
    /// [`reflect`](prism_render_architecture::particle::raytrace::reflect).
    pub reflected: [f32; 3],
    /// Velocity after the bounce response, matching
    /// [`apply_collision`](prism_render_architecture::particle::raytrace::apply_collision).
    pub bounced: [f32; 3],
    /// Chosen collision-method code (`METHOD_RAYTRACE` / `METHOD_SDF` /
    /// `METHOD_DEPTH_BUFFER`), matching
    /// [`choose_collision_method`](prism_render_architecture::particle::raytrace::choose_collision_method).
    pub method_code: u32,
    /// Whether the chosen method uses ray-tracing hardware, matching
    /// [`CollisionMethod::uses_ray_tracing`](prism_render_architecture::particle::raytrace::CollisionMethod).
    pub uses_ray_tracing: bool,
    /// Clamped shadow-ray visibility, matching
    /// [`evaluate_lit_sample`](prism_render_architecture::particle::raytrace::evaluate_lit_sample).
    pub visibility: f32,
    /// Approximate incident irradiance of the lit sample.
    pub irradiance: [f32; 3],
    /// `GI` bounce colour of the lit sample (zero when `sample_gi` is false).
    pub bounce: [f32; 3],
}

/// Maps one golden
/// [`CollisionMethod`](prism_render_architecture::particle::raytrace::CollisionMethod)
/// to its wire code.
fn method_code(method: CollisionMethod) -> u32 {
    match method {
        CollisionMethod::Raytrace => METHOD_RAYTRACE,
        CollisionMethod::Sdf => METHOD_SDF,
        CollisionMethod::DepthBuffer => METHOD_DEPTH_BUFFER,
    }
}

/// Maps a wire quality code to the golden
/// [`CollisionQuality`](prism_render_architecture::particle::raytrace::CollisionQuality);
/// any code other than `QUALITY_HIGH` / `QUALITY_MEDIUM` folds to the cheapest
/// tier, matching the kernel's `else` arm.
fn quality_from_code(code: u32) -> CollisionQuality {
    match code {
        QUALITY_HIGH => CollisionQuality::High,
        QUALITY_MEDIUM => CollisionQuality::Medium,
        _ => CollisionQuality::Low,
    }
}

/// Builds a hand-rolled [`Vec3`] from a packed component triple.
fn to_vec3(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Flattens a hand-rolled [`Vec3`] back to a packed component triple.
fn from_vec3(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// The `CPU` golden verdict for one query, composing the reference entry points
/// so callers (and the parity test) can pin the twin field for field.
///
/// Evaluates
/// [`reflect`](prism_render_architecture::particle::raytrace::reflect),
/// [`apply_collision`](prism_render_architecture::particle::raytrace::apply_collision),
/// [`choose_collision_method`](prism_render_architecture::particle::raytrace::choose_collision_method)
/// and
/// [`evaluate_lit_sample`](prism_render_architecture::particle::raytrace::evaluate_lit_sample)
/// on the query's inputs.
#[must_use]
pub fn cpu_reference(query: &GpuRaytraceQuery) -> GpuRaytraceResult {
    let velocity = to_vec3(query.velocity);
    let hit_normal = to_vec3(query.hit_normal);
    let response = CollisionResponse {
        restitution: query.restitution,
        friction: query.friction,
    };
    let hit = RayHit {
        distance: 0.0,
        normal: hit_normal,
        material_slot: 0,
        back_face: query.back_face,
    };
    let reflected = reflect(velocity, hit_normal);
    let bounced = apply_collision(velocity, hit, response);

    let caps = HardwareCaps {
        ray_tracing: query.ray_tracing,
        sdf_volume: query.sdf_volume,
    };
    let method = choose_collision_method(caps, quality_from_code(query.quality));

    let request = LitSampleRequest {
        position: Vec3::ZERO,
        normal: to_vec3(query.light_normal),
        to_light: to_vec3(query.to_light),
        light_radiance: to_vec3(query.light_radiance),
        sample_gi: query.sample_gi,
    };
    let lit = evaluate_lit_sample(request, query.raw_visibility, to_vec3(query.raw_bounce));

    GpuRaytraceResult {
        reflected: from_vec3(reflected),
        bounced: from_vec3(bounced),
        method_code: method_code(method),
        uses_ray_tracing: method.uses_ray_tracing(),
        visibility: lit.visibility,
        irradiance: from_vec3(lit.irradiance),
        bounce: from_vec3(lit.bounce),
    }
}

/// `repr(C)` `std430` layout of one packed query: six `vec3` slots (each on its
/// own `16`-byte-aligned slot with a trailing pad lane) for the vectors, a
/// four-scalar slot holding `(restitution, friction, raw_visibility, pad)` and
/// two four-`u32` slots holding the four `bool` flags and the quality code —
/// `144` bytes, exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle velocity.
    velocity: [f32; 3],
    /// Padding lane after the velocity.
    pad0: f32,
    /// Geometric hit normal.
    hit_normal: [f32; 3],
    /// Padding lane after the hit normal.
    pad1: f32,
    /// Lit-sample surface normal.
    light_normal: [f32; 3],
    /// Padding lane after the lit normal.
    pad2: f32,
    /// Direction toward the light.
    to_light: [f32; 3],
    /// Padding lane after the light direction.
    pad3: f32,
    /// Incoming light radiance.
    light_radiance: [f32; 3],
    /// Padding lane after the radiance.
    pad4: f32,
    /// Gathered `GI` bounce colour.
    raw_bounce: [f32; 3],
    /// Padding lane after the bounce colour.
    pad5: f32,
    /// Coefficient of restitution.
    restitution: f32,
    /// Tangential friction.
    friction: f32,
    /// Raw shadow-ray visibility.
    raw_visibility: f32,
    /// Padding lane.
    pad6: f32,
    /// Back-face flag (`1` = back face).
    back_face: u32,
    /// Hardware ray-tracing flag (`1` = available).
    ray_tracing: u32,
    /// Scene `SDF` volume flag (`1` = available).
    sdf_volume: u32,
    /// `GI` gather flag (`1` = gather bounce).
    sample_gi: u32,
    /// Quality-tier code.
    quality: u32,
    /// Padding word.
    pad7: u32,
    /// Padding word.
    pad8: u32,
    /// Padding word.
    pad9: u32,
}

impl GpuQuery {
    /// Packs one [`GpuRaytraceQuery`] into its `std430` image.
    fn new(query: &GpuRaytraceQuery) -> GpuQuery {
        GpuQuery {
            velocity: query.velocity,
            pad0: 0.0,
            hit_normal: query.hit_normal,
            pad1: 0.0,
            light_normal: query.light_normal,
            pad2: 0.0,
            to_light: query.to_light,
            pad3: 0.0,
            light_radiance: query.light_radiance,
            pad4: 0.0,
            raw_bounce: query.raw_bounce,
            pad5: 0.0,
            restitution: query.restitution,
            friction: query.friction,
            raw_visibility: query.raw_visibility,
            pad6: 0.0,
            back_face: u32::from(query.back_face),
            ray_tracing: u32::from(query.ray_tracing),
            sdf_volume: u32::from(query.sdf_volume),
            sample_gi: u32::from(query.sample_gi),
            quality: query.quality,
            pad7: 0,
            pad8: 0,
            pad9: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four `vec3` slots (each with a
/// trailing pad lane) for the reflection, bounce, irradiance and `GI` bounce, a
/// four-scalar slot holding `(visibility, pad, pad, pad)` and a four-`u32` slot
/// holding `(method_code, uses_ray_tracing, pad, pad)` — `96` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Mirror reflection direction.
    reflected: [f32; 3],
    /// Padding lane after the reflection.
    pad0: f32,
    /// Bounced velocity.
    bounced: [f32; 3],
    /// Padding lane after the bounce.
    pad1: f32,
    /// Incident irradiance.
    irradiance: [f32; 3],
    /// Padding lane after the irradiance.
    pad2: f32,
    /// `GI` bounce colour.
    bounce: [f32; 3],
    /// Padding lane after the bounce colour.
    pad3: f32,
    /// Clamped shadow-ray visibility.
    visibility: f32,
    /// Padding lane.
    pad4: f32,
    /// Padding lane.
    pad5: f32,
    /// Padding lane.
    pad6: f32,
    /// Chosen collision-method code.
    method_code: u32,
    /// Ray-tracing-usage flag (`1` = uses ray tracing).
    uses_ray_tracing: u32,
    /// Padding word.
    pad7: u32,
    /// Padding word.
    pad8: u32,
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

/// A compiled, reusable ray-trace-resolve compute pipeline.
pub struct GpuRaytrace {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRaytrace {
    /// Compiles the ray-trace-resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRaytrace {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_raytrace"),
            source: ShaderSource::Wgsl(RAYTRACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_raytrace_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_raytrace_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_raytrace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRaytrace {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`GpuRaytraceResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers (`reflect`, `apply_collision`,
    /// `choose_collision_method` and `evaluate_lit_sample`) to within the
    /// tolerance documented on this module, with the `method_code` and
    /// `uses_ray_tracing` classification matching exactly. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuRaytraceQuery]) -> Vec<GpuRaytraceResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_raytrace_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_raytrace_output"),
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
            label: Some("prism_volumetric_raytrace_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_raytrace_bind_group"),
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
            label: Some("prism_volumetric_raytrace_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_raytrace_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_raytrace_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuRaytraceResult`].
fn decode_result(raw: &GpuResult) -> GpuRaytraceResult {
    GpuRaytraceResult {
        reflected: raw.reflected,
        bounced: raw.bounced,
        method_code: raw.method_code,
        uses_ray_tracing: raw.uses_ray_tracing != 0,
        visibility: raw.visibility,
        irradiance: raw.irradiance,
        bounce: raw.bounce,
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
