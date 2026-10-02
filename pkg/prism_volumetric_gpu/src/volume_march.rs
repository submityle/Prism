//! `wgpu` compute twin of the §20 volumetric ray-march integration primitives
//! ([`volume_march`](prism_render_architecture::particle::volume_march), design
//! §20).
//!
//! The `CPU` golden
//! [`volume_march`](prism_render_architecture::particle::volume_march) owns the
//! *stepping / compositing* half of the volumetric renderer: the slab-method
//! ray/box clip
//! [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab),
//! the per-ray start-offset dither
//! [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset),
//! and the front-to-back transmittance integral inside
//! [`march`](prism_render_architecture::particle::volume_march::march).
//! [`GpuVolumeMarch`] is the on-device twin that runs one thread per query and
//! reproduces every lane, so a passing real-device parity test is direct
//! evidence the ported kernels fold the same guarded divisions, the same
//! integer avalanche hash and the same algebraic opacity composite the
//! reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Three disjoint kernels, one per [`VolumeMarchQuery`] variant:
//!
//! - [`VolumeMarchQuery::Slab`] reproduces
//!   [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab)
//!   guard for guard: a direction component whose magnitude is below [`EPS`] is
//!   *parallel* and divides nothing (a miss when the origin lies outside that
//!   slab), every other axis forms the reciprocal `1 / d` and folds its
//!   `[t_near, t_far]` into the running `[t_enter, t_exit]` with `min`/`max`, and
//!   the chord is reported only when `t_enter <= t_exit` and `t_exit >= 0`
//!   (compared with `<=`/`>=`, never with `==`).
//! - [`VolumeMarchQuery::Jitter`] reproduces
//!   [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset):
//!   the integer lattice avalanche
//!   [`hash_lattice`](prism_render_architecture::particle::noise) keyed on
//!   `ray_index` and `seed`, its top 24 bits mapped into `[0, 1)` and scaled by
//!   `step_size`. The integer path is bit-exact; only the final `f32` scale is
//!   compared within tolerance.
//! - [`VolumeMarchQuery::Integrate`] reproduces the transmittance-integration
//!   core of
//!   [`march`](prism_render_architecture::particle::volume_march::march) over a
//!   *host-presampled* density array: per sample `sigma = density ·
//!   density_scale · extinction`, `tau = (sigma · step_size).max(0)`, `alpha =
//!   tau.clamp(0, 1)`, `transmittance *= 1 - alpha`, early-out once
//!   `transmittance < cutoff`, and a final clamp — the exact loop body of
//!   `march` with a uniform step.
//!
//! # What stays on the host (deliberately not twinned)
//!
//! The closure-driven density and lighting sampling that
//! [`march`](prism_render_architecture::particle::volume_march::march) and
//! [`march_scattered`](prism_render_architecture::particle::volume_march::march_scattered)
//! evaluate per step cannot cross to a fixed compute kernel: the sampler is an
//! arbitrary host closure (typically the `froxel` density field and the six-way
//! luminance rig). This twin therefore keeps the *sampling* on the host — the
//! caller presamples density into the [`VolumeMarchQuery::Integrate`] array —
//! and ports only the *integration* algorithm to the `GPU`. The multi-light
//! front-to-back radiance accumulation of
//! [`march_scattered`](prism_render_architecture::particle::volume_march::march_scattered)
//! is likewise left on the host for the same closure reason.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `+ - * /`, integer multiply / shift / xor and one `bitcast` to
//! materialize the `IEEE`-754 infinity sentinel that mirrors the reference's
//! `f32::INFINITY` running bounds — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `sqrt` or optional device feature. They run unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of guarded divisions, an
//! integer avalanche or an algebraic composite, so `CPU` and `GPU` evaluate the
//! same closed form in the same associativity. They are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place. The parity test asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` chord,
//! offset and transmittance values yet an *exact* match on the discrete hit
//! flag and step count, which the kernels route through the same [`EPS`]
//! magnitude band and the same integer arithmetic the reference uses.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::volume_march`；
//! standard slab-method ray/`AABB` intersection, `PCG`-style integer lattice
//! hash and algebraic (`exp`-free) front-to-back transmittance compositing plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sort_cull::Aabb;
use prism_render_architecture::particle::volume_march::{
    jittered_start_offset, ray_aabb_slab, Ray,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete hit code written by the slab kernel for a lane whose chord is
/// visible: matches the host `== 1` decode in [`decode_slab`]. A direct `f32`
/// equality is forbidden, so the kernel emits an integer flag rather than a
/// sentinel float.
const CODE_HIT: u32 = 1;

/// Magnitude floor guarding the parallel-slab `1 / d` division and every `f32`
/// step span, matching the reference
/// [`EPS`](prism_render_architecture::particle::volume_march::EPS). A direct
/// `f32` `==`/`!=` is forbidden, so the parallel and degenerate-step tests
/// compare magnitudes against this floor instead of exact zero.
pub const EPS: f32 = 1.0e-6;

/// Fixed upper bound on the host-presampled density array an
/// [`VolumeMarchQuery::Integrate`] query may carry, matching the
/// `MAX_DENSITY_SAMPLES` constant inside [`VOLUME_MARCH_INTEGRATE_WGSL`].
///
/// Samples beyond this bound are dropped by both the host reference and the
/// kernel, so the two agree lane for lane. Provenance: fixed-capacity
/// `std430` block sizing for `volume_march`.
pub const MAX_DENSITY_SAMPLES: usize = 256;

/// The portable core-`WGSL` ray/`AABB` slab kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab).
const VOLUME_MARCH_SLAB_WGSL: &str = r#"
// Ray vs AABB slab-method twin: one thread per (ray, box) query folds the three
// per-axis slab intervals into the ordered chord [t_enter, t_exit], then writes
// a hit flag plus the chord. A direction component below EPS is parallel and
// divides nothing (a miss when the origin lies outside that slab); the chord is
// reported only when t_enter <= t_exit and t_exit >= 0, matching the CPU golden
// `volume_march::ray_aabb_slab` guard for guard. Uses only abs/min/max, + - * /
// and one bitcast for the infinity sentinel, so it needs no optional feature.
//
// Provenance: standard slab-method ray/AABB intersection; no third-party engine
// source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    origin: vec4<f32>,
    dir: vec4<f32>,
    bmin: vec4<f32>,
    bmax: vec4<f32>,
}

struct Result {
    hit: u32,
    t_enter: f32,
    t_exit: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const EPS: f32 = 1.0e-6;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let origin = queries[idx].origin.xyz;
    let dir = queries[idx].dir.xyz;
    let bmin = queries[idx].bmin.xyz;
    let bmax = queries[idx].bmax.xyz;

    // IEEE-754 +/-infinity sentinels, mirroring the reference's f32::INFINITY
    // running bounds exactly; built by bitcast because WGSL has no inf literal.
    let pos_inf = bitcast<f32>(0x7f800000u);
    let neg_inf = bitcast<f32>(0xff800000u);
    var t_enter = neg_inf;
    var t_exit = pos_inf;
    var missed = false;

    // Axis x.
    if (abs(dir.x) < EPS) {
        if (origin.x < bmin.x || origin.x > bmax.x) {
            missed = true;
        }
    } else {
        let inv = 1.0 / dir.x;
        let a = (bmin.x - origin.x) * inv;
        let b = (bmax.x - origin.x) * inv;
        t_enter = max(t_enter, min(a, b));
        t_exit = min(t_exit, max(a, b));
    }
    // Axis y.
    if (abs(dir.y) < EPS) {
        if (origin.y < bmin.y || origin.y > bmax.y) {
            missed = true;
        }
    } else {
        let inv = 1.0 / dir.y;
        let a = (bmin.y - origin.y) * inv;
        let b = (bmax.y - origin.y) * inv;
        t_enter = max(t_enter, min(a, b));
        t_exit = min(t_exit, max(a, b));
    }
    // Axis z.
    if (abs(dir.z) < EPS) {
        if (origin.z < bmin.z || origin.z > bmax.z) {
            missed = true;
        }
    } else {
        let inv = 1.0 / dir.z;
        let a = (bmin.z - origin.z) * inv;
        let b = (bmax.z - origin.z) * inv;
        t_enter = max(t_enter, min(a, b));
        t_exit = min(t_exit, max(a, b));
    }

    var res: Result;
    res.pad0 = 0u;
    let hit = (!missed) && (t_enter <= t_exit) && (t_exit >= 0.0);
    if (hit) {
        res.hit = 1u;
        res.t_enter = t_enter;
        res.t_exit = t_exit;
    } else {
        res.hit = 0u;
        res.t_enter = 0.0;
        res.t_exit = 0.0;
    }
    results[idx] = res;
}
"#;

/// The portable core-`WGSL` start-offset dither kernel, embedded inline. Mirrors
/// the `CPU` golden
/// [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset)
/// and its integer lattice hash
/// [`hash_lattice`](prism_render_architecture::particle::noise).
const VOLUME_MARCH_JITTER_WGSL: &str = r#"
// Per-ray start-offset dither twin: one thread per query folds `ray_index` and
// `seed` through the PCG-style integer lattice avalanche the reference uses,
// maps the top 24 hash bits into [0, 1) and scales by step_size. The integer
// path is bit-exact; only the final f32 scale carries ULP slack. Uses only
// integer multiply / shift / xor and one f32 multiply, so it needs no optional
// feature.
//
// Provenance: PCG-style integer lattice hash plus a 24-bit mantissa map; no
// third-party engine source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    ray_index: u32,
    seed: u32,
    step_size: f32,
    pad0: u32,
}

struct Result {
    offset: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Scale that maps a 24-bit hash mantissa into [0, 1) without a runtime divide,
// equal to 1 / 2^24, matching the reference HASH_UNIT_SCALE.
const HASH_UNIT_SCALE: f32 = 1.0 / 16777216.0;

// One 32-bit left-rotate by 15, matching the reference `rotate_left(15)`.
fn rotl15(x: u32) -> u32 {
    return (x << 15u) | (x >> 17u);
}

// One folding step: xor-in a multiplied input word, then rotate and multiply to
// spread the bits before the next word is folded.
fn mix_h(h: u32, v: u32) -> u32 {
    var x = h ^ (v * 0x9E3779B1u);
    x = rotl15(x) * 0x85EBCA6Bu;
    return x;
}

// Final avalanche applied once after all inputs are folded.
fn finalize_h(h: u32) -> u32 {
    var x = h ^ (h >> 16u);
    x = x * 0x7FEB352Du;
    x = x ^ (x >> 15u);
    x = x * 0x846CA68Bu;
    x = x ^ (x >> 16u);
    return x;
}

// hash_lattice(ray_index as i32, seed as i32, 0, seed): the signed casts are
// two's-complement no-ops at the bit level, so the u32 words are folded as-is.
fn hash_lattice(ray_index: u32, seed: u32) -> u32 {
    var h = seed ^ 0x811C9DC5u;
    h = mix_h(h, ray_index);
    h = mix_h(h, seed);
    h = mix_h(h, 0u);
    return finalize_h(h);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let span = max(queries[idx].step_size, 0.0);
    let h = hash_lattice(queries[idx].ray_index, queries[idx].seed);
    let unit = f32(h >> 8u) * HASH_UNIT_SCALE;

    var res: Result;
    res.offset = unit * span;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    results[idx] = res;
}
"#;

/// The portable core-`WGSL` transmittance-integration kernel, embedded inline.
/// Mirrors the loop body of the `CPU` golden
/// [`march`](prism_render_architecture::particle::volume_march::march) over a
/// host-presampled density array.
const VOLUME_MARCH_INTEGRATE_WGSL: &str = r#"
// Front-to-back transmittance integration twin: one thread per query walks the
// host-presampled density array and composites the algebraic (exp-free) step
// opacity exactly as the reference `march` loop body does — sigma = density *
// density_scale * extinction, tau = (sigma * step).max(0), alpha = clamp(tau),
// transmittance *= 1 - alpha, early-out on transmittance < cutoff, final clamp.
// Uses only min/max/clamp and + - *, so it needs no optional feature.
//
// Provenance: algebraic front-to-back transmittance compositing; no
// third-party engine source or derived code.

const MAX_DENSITY_SAMPLES: u32 = 256u;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    step_size: f32,
    density_scale: f32,
    extinction: f32,
    cutoff: f32,
    sample_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    densities: array<f32, 256>,
}

struct Result {
    transmittance: f32,
    optical_depth: f32,
    steps_taken: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const EPS: f32 = 1.0e-6;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let step = max(queries[idx].step_size, EPS);
    let density_scale = queries[idx].density_scale;
    let extinction = queries[idx].extinction;
    let cutoff = clamp(queries[idx].cutoff, 0.0, 1.0);
    var sample_count = queries[idx].sample_count;
    if (sample_count > MAX_DENSITY_SAMPLES) {
        sample_count = MAX_DENSITY_SAMPLES;
    }

    var transmittance = 1.0;
    var optical_depth = 0.0;
    var steps = 0u;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let density = max(queries[idx].densities[i], 0.0);
        let sigma = density * density_scale * extinction;
        let tau = max(sigma * step, 0.0);
        optical_depth = optical_depth + tau;
        let alpha = clamp(tau, 0.0, 1.0);
        transmittance = transmittance * (1.0 - alpha);
        steps = steps + 1u;
        if (transmittance < cutoff) {
            break;
        }
    }

    var res: Result;
    res.transmittance = clamp(transmittance, 0.0, 1.0);
    res.optical_depth = optical_depth;
    res.steps_taken = steps;
    res.pad0 = 0u;
    results[idx] = res;
}
"#;

/// One volumetric ray-march query, tagged by which §20 primitive it exercises.
///
/// The three variants map to the three disjoint kernels of [`GpuVolumeMarch`]:
/// a slab ray/box clip, a per-ray start-offset dither and a transmittance
/// integration over a host-presampled density array. Holds `f32` geometry, so
/// it derives only [`Clone`], [`Debug`] and [`PartialEq`] (no [`Eq`]/[`Hash`]).
/// Provenance: query tagging for `volume_march`.
#[derive(Clone, Debug, PartialEq)]
pub enum VolumeMarchQuery {
    /// A ray vs `AABB` slab clip, mirroring
    /// [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab).
    Slab {
        /// The ray whose infinite line is clipped against `aabb`.
        ray: Ray,
        /// The axis-aligned box the ray is clipped against.
        aabb: Aabb,
    },
    /// A per-ray start-offset dither, mirroring
    /// [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset).
    Jitter {
        /// The ray index keying the lattice hash.
        ray_index: u32,
        /// The dither seed keying the lattice hash.
        seed: u32,
        /// The nominal march step the offset lands inside `[0, step_size)`.
        step_size: f32,
    },
    /// A front-to-back transmittance integration over host-presampled density,
    /// mirroring the loop body of
    /// [`march`](prism_render_architecture::particle::volume_march::march).
    Integrate {
        /// Host-presampled per-step density, at most [`MAX_DENSITY_SAMPLES`]
        /// entries (extra entries are dropped on both sides).
        densities: Vec<f32>,
        /// Uniform world-space step length applied to every sample.
        step_size: f32,
        /// Multiplier applied to the sampled density before extinction.
        density_scale: f32,
        /// Extinction coefficient scaling `density` into `sigma`.
        extinction: f32,
        /// Transmittance below which the integration early-terminates.
        cutoff: f32,
    },
}

/// The resolved verdict for one [`VolumeMarchQuery`], the host-side mirror of a
/// kernel result lane.
///
/// Carries `f32` parameters, so it derives [`Clone`], [`Copy`], [`Debug`] and
/// [`PartialEq`] (no [`Eq`]/[`Hash`]). Provenance: result decoding for
/// `volume_march`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VolumeMarchResult {
    /// The slab-clip verdict: whether the chord is visible and, when `hit` is
    /// `true`, its ordered endpoints.
    Slab {
        /// Whether the forward ray's clipped chord is visible.
        hit: bool,
        /// The chord entry parameter; meaningful only when `hit`.
        t_enter: f32,
        /// The chord exit parameter; meaningful only when `hit`.
        t_exit: f32,
    },
    /// The dither verdict: the start offset in `[0, step_size)`.
    Jitter {
        /// The per-ray start offset.
        offset: f32,
    },
    /// The integration verdict: surviving transmittance, accumulated optical
    /// depth and the number of composited steps.
    Integrate {
        /// Surviving light fraction in `0..=1`.
        transmittance: f32,
        /// Accumulated optical thickness `Σ sigma·step`.
        optical_depth: f32,
        /// Number of composited steps.
        steps_taken: u32,
    },
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// The slab and dither arms call the golden
/// [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab)
/// and
/// [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset)
/// directly; the integration arm replays the
/// [`march`](prism_render_architecture::particle::volume_march::march) loop body
/// over the presampled density array through [`integrate_transmittance`].
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::volume_march`.
#[must_use]
pub fn cpu_reference(query: &VolumeMarchQuery) -> VolumeMarchResult {
    match query {
        VolumeMarchQuery::Slab { ray, aabb } => match ray_aabb_slab(*ray, *aabb) {
            Some((t_enter, t_exit)) => VolumeMarchResult::Slab {
                hit: true,
                t_enter,
                t_exit,
            },
            None => VolumeMarchResult::Slab {
                hit: false,
                t_enter: 0.0,
                t_exit: 0.0,
            },
        },
        VolumeMarchQuery::Jitter {
            ray_index,
            seed,
            step_size,
        } => VolumeMarchResult::Jitter {
            offset: jittered_start_offset(*ray_index, *seed, *step_size),
        },
        VolumeMarchQuery::Integrate {
            densities,
            step_size,
            density_scale,
            extinction,
            cutoff,
        } => {
            let (transmittance, optical_depth, steps_taken) = integrate_transmittance(
                densities,
                *step_size,
                *density_scale,
                *extinction,
                *cutoff,
            );
            VolumeMarchResult::Integrate {
                transmittance,
                optical_depth,
                steps_taken,
            }
        }
    }
}

/// Replays the front-to-back transmittance integration of
/// [`march`](prism_render_architecture::particle::volume_march::march) over a
/// host-presampled density array with a uniform step.
///
/// Returns the surviving `transmittance`, the accumulated `optical_depth` and
/// the `steps_taken`, matching the reference loop body: `sigma = density ·
/// density_scale · extinction`, `tau = (sigma · step).max(0)`, `alpha =
/// tau.clamp(0, 1)`, `transmittance *= 1 - alpha`, early-out once
/// `transmittance < cutoff`, final clamp. Samples beyond [`MAX_DENSITY_SAMPLES`]
/// are dropped so the host and the kernel agree lane for lane. Provenance:
/// 孪生自本仓 `prism_render_architecture::particle::volume_march::march`.
#[must_use]
fn integrate_transmittance(
    densities: &[f32],
    step_size: f32,
    density_scale: f32,
    extinction: f32,
    cutoff: f32,
) -> (f32, f32, u32) {
    let step = step_size.max(EPS);
    let cutoff = cutoff.clamp(0.0, 1.0);
    let count = densities.len().min(MAX_DENSITY_SAMPLES);
    let mut transmittance = 1.0f32;
    let mut optical_depth = 0.0f32;
    let mut steps_taken = 0u32;
    for &d in &densities[..count] {
        let density = d.max(0.0);
        let sigma = density * density_scale * extinction;
        let tau = (sigma * step).max(0.0);
        optical_depth += tau;
        let alpha = tau.clamp(0.0, 1.0);
        transmittance *= 1.0 - alpha;
        steps_taken += 1;
        if transmittance < cutoff {
            break;
        }
    }
    (transmittance.clamp(0.0, 1.0), optical_depth, steps_taken)
}

/// Uniform dispatch parameters shared by all three kernels. `repr(C)` `std430`
/// layout matching each `Params` struct: the lane count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid lanes in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One slab query as uploaded. `64`-byte `std430` stride matching `Query` in
/// [`VOLUME_MARCH_SLAB_WGSL`]: the ray origin and direction and the box corners,
/// each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSlabQuery {
    /// Ray origin in `xyz`; the `w` lane is unused padding.
    origin: [f32; 4],
    /// Ray direction in `xyz`; the `w` lane is unused padding.
    dir: [f32; 4],
    /// Box `min` corner in `xyz`; the `w` lane is unused padding.
    bmin: [f32; 4],
    /// Box `max` corner in `xyz`; the `w` lane is unused padding.
    bmax: [f32; 4],
}

impl GpuSlabQuery {
    /// Packs a slab query into the `std430` upload layout.
    fn new(ray: Ray, aabb: Aabb) -> GpuSlabQuery {
        let o = ray.origin;
        let d = ray.direction;
        let lo = aabb.min;
        let hi = aabb.max;
        GpuSlabQuery {
            origin: [o.x, o.y, o.z, 0.0],
            dir: [d.x, d.y, d.z, 0.0],
            bmin: [lo.x, lo.y, lo.z, 0.0],
            bmax: [hi.x, hi.y, hi.z, 0.0],
        }
    }
}

/// One slab result as read back. `16`-byte `std430` stride matching `Result` in
/// [`VOLUME_MARCH_SLAB_WGSL`]: the hit flag, the ordered chord and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSlabResult {
    /// Chord-visible flag (`1` = hit).
    hit: u32,
    /// Chord entry parameter.
    t_enter: f32,
    /// Chord exit parameter.
    t_exit: f32,
    /// Padding word.
    pad0: u32,
}

/// One dither query as uploaded. `16`-byte `std430` stride matching `Query` in
/// [`VOLUME_MARCH_JITTER_WGSL`]: the ray index, seed, step size and one pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuJitterQuery {
    /// Ray index keying the lattice hash.
    ray_index: u32,
    /// Dither seed keying the lattice hash.
    seed: u32,
    /// Nominal march step scaling the offset.
    step_size: f32,
    /// Padding word.
    pad0: u32,
}

/// One dither result as read back. `16`-byte `std430` stride matching `Result`
/// in [`VOLUME_MARCH_JITTER_WGSL`]: the offset and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuJitterResult {
    /// The per-ray start offset.
    offset: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One integration query as uploaded. `std430` layout matching `Query` in
/// [`VOLUME_MARCH_INTEGRATE_WGSL`]: a `32`-byte scalar header followed by the
/// fixed-capacity [`MAX_DENSITY_SAMPLES`] density block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuIntegrateQuery {
    /// Uniform world-space step length.
    step_size: f32,
    /// Density multiplier applied before extinction.
    density_scale: f32,
    /// Extinction coefficient scaling `density` into `sigma`.
    extinction: f32,
    /// Transmittance early-out threshold.
    cutoff: f32,
    /// Number of live density samples, at most [`MAX_DENSITY_SAMPLES`].
    sample_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Fixed-capacity density block; only the first `sample_count` are live.
    densities: [f32; MAX_DENSITY_SAMPLES],
}

impl GpuIntegrateQuery {
    /// Packs an integration query into the `std430` upload layout, truncating
    /// the presampled density array to [`MAX_DENSITY_SAMPLES`].
    fn new(
        densities: &[f32],
        step_size: f32,
        density_scale: f32,
        extinction: f32,
        cutoff: f32,
    ) -> GpuIntegrateQuery {
        let count = densities.len().min(MAX_DENSITY_SAMPLES);
        let mut block = [0.0f32; MAX_DENSITY_SAMPLES];
        block[..count].copy_from_slice(&densities[..count]);
        GpuIntegrateQuery {
            step_size,
            density_scale,
            extinction,
            cutoff,
            sample_count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            densities: block,
        }
    }
}

/// One integration result as read back. `16`-byte `std430` stride matching
/// `Result` in [`VOLUME_MARCH_INTEGRATE_WGSL`]: transmittance, optical depth,
/// step count and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuIntegrateResult {
    /// Surviving light fraction in `0..=1`.
    transmittance: f32,
    /// Accumulated optical thickness.
    optical_depth: f32,
    /// Number of composited steps.
    steps_taken: u32,
    /// Padding word.
    pad0: u32,
}

/// Maps one slab kernel lane back to the host [`VolumeMarchResult`].
fn decode_slab(raw: &GpuSlabResult) -> VolumeMarchResult {
    VolumeMarchResult::Slab {
        hit: raw.hit == CODE_HIT,
        t_enter: raw.t_enter,
        t_exit: raw.t_exit,
    }
}

/// Maps one dither kernel lane back to the host [`VolumeMarchResult`].
fn decode_jitter(raw: &GpuJitterResult) -> VolumeMarchResult {
    VolumeMarchResult::Jitter { offset: raw.offset }
}

/// Maps one integration kernel lane back to the host [`VolumeMarchResult`].
fn decode_integrate(raw: &GpuIntegrateResult) -> VolumeMarchResult {
    VolumeMarchResult::Integrate {
        transmittance: raw.transmittance,
        optical_depth: raw.optical_depth,
        steps_taken: raw.steps_taken,
    }
}

/// A compiled, reusable volumetric ray-march twin bundling the three §20
/// kernels (slab clip, start-offset dither, transmittance integration).
pub struct GpuVolumeMarch {
    #[expect(
        dead_code,
        reason = "kept alive so the slab pipeline it produced stays valid"
    )]
    module_slab: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the dither pipeline it produced stays valid"
    )]
    module_jitter: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the integration pipeline it produced stays valid"
    )]
    module_integrate: ShaderModule,
    layout: BindGroupLayout,
    slab_pipeline: ComputePipeline,
    jitter_pipeline: ComputePipeline,
    integrate_pipeline: ComputePipeline,
}

impl GpuVolumeMarch {
    /// Compiles the three volumetric ray-march kernels on `ctx`.
    ///
    /// All three share one bind-group layout (uniform params, read-only query
    /// storage, read-write result storage) and use only the portable
    /// core-`WGSL` subset, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVolumeMarch {
        let device = ctx.device();
        let module_slab = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_volume_march_slab"),
            source: ShaderSource::Wgsl(VOLUME_MARCH_SLAB_WGSL.into()),
        });
        let module_jitter = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_volume_march_jitter"),
            source: ShaderSource::Wgsl(VOLUME_MARCH_JITTER_WGSL.into()),
        });
        let module_integrate = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_volume_march_integrate"),
            source: ShaderSource::Wgsl(VOLUME_MARCH_INTEGRATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_volume_march_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_volume_march_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let slab_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_volume_march_slab_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_slab,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let jitter_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_volume_march_jitter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_jitter,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let integrate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_volume_march_integrate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_integrate,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVolumeMarch {
            module_slab,
            module_jitter,
            module_integrate,
            layout,
            slab_pipeline,
            jitter_pipeline,
            integrate_pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`VolumeMarchResult`] per
    /// query in input order.
    ///
    /// Queries are partitioned by variant and dispatched to the matching kernel
    /// (empty sub-batches are skipped, since a storage buffer cannot be
    /// zero-sized), then the per-kernel results are scattered back into input
    /// order. The returned result for a slab query mirrors
    /// [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab),
    /// a dither query
    /// [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset),
    /// and an integration query the loop body of
    /// [`march`](prism_render_architecture::particle::volume_march::march).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[VolumeMarchQuery]) -> Vec<VolumeMarchResult> {
        if queries.is_empty() {
            return Vec::new();
        }

        let mut slab_idx: Vec<usize> = Vec::new();
        let mut slab_q: Vec<GpuSlabQuery> = Vec::new();
        let mut jitter_idx: Vec<usize> = Vec::new();
        let mut jitter_q: Vec<GpuJitterQuery> = Vec::new();
        let mut integrate_idx: Vec<usize> = Vec::new();
        let mut integrate_q: Vec<GpuIntegrateQuery> = Vec::new();

        for (i, query) in queries.iter().enumerate() {
            match query {
                VolumeMarchQuery::Slab { ray, aabb } => {
                    slab_idx.push(i);
                    slab_q.push(GpuSlabQuery::new(*ray, *aabb));
                }
                VolumeMarchQuery::Jitter {
                    ray_index,
                    seed,
                    step_size,
                } => {
                    jitter_idx.push(i);
                    jitter_q.push(GpuJitterQuery {
                        ray_index: *ray_index,
                        seed: *seed,
                        step_size: *step_size,
                        pad0: 0,
                    });
                }
                VolumeMarchQuery::Integrate {
                    densities,
                    step_size,
                    density_scale,
                    extinction,
                    cutoff,
                } => {
                    integrate_idx.push(i);
                    integrate_q.push(GpuIntegrateQuery::new(
                        densities,
                        *step_size,
                        *density_scale,
                        *extinction,
                        *cutoff,
                    ));
                }
            }
        }

        let mut results: Vec<Option<VolumeMarchResult>> =
            (0..queries.len()).map(|_| None).collect();

        if !slab_q.is_empty() {
            let raw: Vec<GpuSlabResult> =
                run_kernel(ctx, &self.slab_pipeline, &self.layout, &slab_q);
            for (lane, &i) in slab_idx.iter().enumerate() {
                results[i] = Some(decode_slab(&raw[lane]));
            }
        }
        if !jitter_q.is_empty() {
            let raw: Vec<GpuJitterResult> =
                run_kernel(ctx, &self.jitter_pipeline, &self.layout, &jitter_q);
            for (lane, &i) in jitter_idx.iter().enumerate() {
                results[i] = Some(decode_jitter(&raw[lane]));
            }
        }
        if !integrate_q.is_empty() {
            let raw: Vec<GpuIntegrateResult> =
                run_kernel(ctx, &self.integrate_pipeline, &self.layout, &integrate_q);
            for (lane, &i) in integrate_idx.iter().enumerate() {
                results[i] = Some(decode_integrate(&raw[lane]));
            }
        }

        results
            .into_iter()
            .map(|r| r.expect("every query lane should be filled by its kernel"))
            .collect()
    }
}

/// Runs one compute dispatch over `gpu_queries` and reads back one `R` per
/// query.
///
/// Uploads the lane count as `GpuParams`, binds the query and result storage
/// buffers, dispatches `div_ceil(count, WORKGROUP_SIZE)` workgroups of
/// [`WORKGROUP_SIZE`] threads and maps the results back. The caller guarantees
/// `gpu_queries` is non-empty, since a storage buffer cannot be zero-sized.
fn run_kernel<Q, R>(
    ctx: &GpuContext,
    pipeline: &ComputePipeline,
    layout: &BindGroupLayout,
    gpu_queries: &[Q],
) -> Vec<R>
where
    Q: Pod,
    R: Pod,
{
    let device = ctx.device();
    let count = gpu_queries.len();

    let gpu_params = GpuParams {
        count: count as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let out_bytes = (count * size_of::<R>()) as u64;

    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("prism_volumetric_volume_march_params"),
        contents: bytemuck::bytes_of(&gpu_params),
        usage: BufferUsages::UNIFORM,
    });
    let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("prism_volumetric_volume_march_queries"),
        contents: bytemuck::cast_slice(gpu_queries),
        usage: BufferUsages::STORAGE,
    });
    let results_buf = device.create_buffer(&BufferDescriptor {
        label: Some("prism_volumetric_volume_march_results"),
        size: out_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let results_stage = device.create_buffer(&BufferDescriptor {
        label: Some("prism_volumetric_volume_march_results_stage"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("prism_volumetric_volume_march_bind_group"),
        layout,
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
                resource: results_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism_volumetric_volume_march_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_volumetric_volume_march_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One thread per query, flattened to a 1-D dispatch.
        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
    ctx.queue().submit([encoder.finish()]);

    results_stage.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();
    let view = results_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let raw = bytemuck::cast_slice::<u8, R>(&view).to_vec();
    drop(view);
    results_stage.unmap();
    debug_assert_eq!(raw.len(), count);

    raw
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
