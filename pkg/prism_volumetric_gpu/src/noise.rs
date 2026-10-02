//! `wgpu` compute twin of the particle procedural-noise field
//! ([`noise`](prism_render_architecture::particle::noise), particle design §8,
//! §10).
//!
//! The `CPU` golden
//! [`noise`](prism_render_architecture::particle::noise) owns the analytic,
//! simulation-free turbulence stack: a stateless integer lattice hash
//! ([`hash_lattice`](prism_render_architecture::particle::noise::hash_lattice)),
//! the twelve-edge gradient it selects
//! ([`lattice_gradient`](prism_render_architecture::particle::noise::lattice_gradient)),
//! trilinear value noise
//! ([`value_noise_3d`](prism_render_architecture::particle::noise::value_noise_3d)),
//! `Perlin`-style gradient noise
//! ([`gradient_noise_3d`](prism_render_architecture::particle::noise::gradient_noise_3d)),
//! the fractal-Brownian-motion sum
//! ([`fbm`](prism_render_architecture::particle::noise::fbm)), and the analytic
//! divergence-free curl flow
//! ([`curl_noise_3d`](prism_render_architecture::particle::noise::curl_noise_3d),
//! [`curl_noise_fbm`](prism_render_architecture::particle::noise::curl_noise_fbm))
//! wrapped as a turbulence force
//! ([`turbulence_force`](prism_render_architecture::particle::noise::turbulence_force)).
//! [`GpuNoise`] is the on-device twin: one thread evaluates one query and
//! reproduces every field, so a passing real-device parity test is direct
//! evidence the ported kernel hashes the same lattice and sums the same octaves
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each [`NoiseQuery`] the kernel reports, at one `(pos, seed)` with a
//! supplied [`GpuFbmParams`] and [`GpuTurbulenceParams`]: the integer
//! `hash_lattice` of an explicit cell, that cell's `lattice_gradient`, the
//! `value_noise_3d` and `gradient_noise_3d` scalars at `pos`, the `fbm` sum, the
//! single-octave `curl_noise_3d`, the multi-octave `curl_noise_fbm`, and the
//! `turbulence_force`.
//!
//! # Correctness model
//!
//! All randomness flows through the integer hash: `WGSL` unsigned `+` / `*` wrap
//! on overflow exactly like Rust's `wrapping_add` / `wrapping_mul`, and signed
//! cell coordinates reach the hash through a `bitcast` that mirrors the
//! reference's two's-complement `as u32`, so the integer lattice work and the
//! `hash_lattice` output are bit-identical and compared with an exact `==`. The
//! float fields thread through the quintic fade, trilinear blends, the
//! amplitude-normalised octave sum and the central-difference curl, so `CPU`
//! and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous quantity, tight enough to catch a
//! dropped octave, a swapped gradient or a wrong stencil yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `abs`,
//! `dot`, `+ - * /`, unsigned bit arithmetic and `bitcast` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`,
//! no `round`, no `sqrt` and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The quintic fade is written in `Horner`
//! form (multiply / add only) rather than `smoothstep`, and the `fbm` octave
//! loop is bounded by the compile-time constant `MAX_OCTAVES` with an
//! `if i >= octaves { break; }` guard, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::noise`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` procedural-noise kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`noise`](prism_render_architecture::particle::noise) function for function;
/// see the module documentation for the algorithm.
const NOISE_WGSL: &str = r#"
// Procedural-noise twin: one thread per query reproduces the integer lattice
// hash, the selected gradient, value and gradient noise, the fBm sum and the
// analytic curl-noise turbulence of the CPU golden. The integer hash is
// bit-identical; the float fields match under the tolerance documented on the
// Rust module.

// Small positive guard against a (near) zero fBm amplitude-normalisation sum.
const EPS: f32 = 1.0e-9;
// Central-difference half-step for the analytic curl of the vector potential.
const CURL_EPS: f32 = 1.0e-2;
// Odd-integer salt for the second vector-potential component.
const SEED_SALT_Y: u32 = 0x9E3779B9u;
// Odd-integer salt for the third vector-potential component.
const SEED_SALT_Z: u32 = 0x85EBCA6Bu;
// Per-octave seed increment so successive fBm octaves decorrelate.
const SEED_STEP: u32 = 0x165667B1u;
// Scale that turns a 24-bit hash mantissa into [0, 1).
const INV_2POW24: f32 = 0.000000059604644775390625;
// Compile-time upper bound on the fBm octave loop; the real count is clamped by
// an `if i >= octaves { break; }` guard so the kernel always terminates.
const MAX_OCTAVES: u32 = 16u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sample position for the continuous noise fields; a pad lane follows.
    pos: vec3<f32>,
    pad0: f32,
    // Integer lattice cell for hash_lattice / lattice_gradient; a pad lane
    // follows.
    cell: vec3<i32>,
    pad1: i32,
    // Base seed, then the fBm octave count, lacunarity and gain.
    seed: u32,
    fbm_octaves: u32,
    fbm_lacunarity: f32,
    fbm_gain: f32,
    // Turbulence seed, then its fBm octave count, lacunarity and gain.
    turb_seed: u32,
    turb_fbm_octaves: u32,
    turb_fbm_lacunarity: f32,
    turb_fbm_gain: f32,
    // Turbulence frequency and amplitude, then two pad words.
    turb_frequency: f32,
    turb_amplitude: f32,
    pad2: u32,
    pad3: u32,
}

struct Result {
    // hash_lattice (bit-exact), then value, gradient and fBm scalars.
    hash: u32,
    value_noise: f32,
    gradient_noise: f32,
    fbm_value: f32,
    // lattice_gradient; a pad lane follows.
    lattice_gradient: vec3<f32>,
    pad0: f32,
    // curl_noise_3d (single octave); a pad lane follows.
    curl3: vec3<f32>,
    pad1: f32,
    // curl_noise_fbm (query octaves); a pad lane follows.
    curl_fbm: vec3<f32>,
    pad2: f32,
    // turbulence_force; a pad lane follows.
    turbulence: vec3<f32>,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// One folding step of the hash: xor-in a multiplied input word, then rotate and
// multiply to spread the bits before the next word is folded.
fn hash_mix(h0: u32, v: u32) -> u32 {
    var h = h0;
    h = h ^ (v * 0x9E3779B1u);
    h = ((h << 15u) | (h >> 17u)) * 0x85EBCA6Bu;
    return h;
}

// Final avalanche applied once after all inputs are folded.
fn finalize_hash(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return h;
}

// Stateless integer hash of a lattice cell and seed (the noise RNG). Negative
// coordinates reach the hash through a bitcast, mirroring the reference's
// two's-complement `as u32`.
fn hash_lattice(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    var h = seed ^ 0x811C9DC5u;
    h = hash_mix(h, bitcast<u32>(i));
    h = hash_mix(h, bitcast<u32>(j));
    h = hash_mix(h, bitcast<u32>(k));
    return finalize_hash(h);
}

// Maps the low bits of a hash to one of the twelve edge gradients, reproducing
// Perlin's improved-noise grad selection as an explicit vector.
fn grad_select(h: u32) -> vec3<f32> {
    let hh = h & 15u;
    var ux = 0.0;
    var uy = 0.0;
    if (hh < 8u) {
        ux = 1.0;
        uy = 0.0;
    } else {
        ux = 0.0;
        uy = 1.0;
    }
    var vx = 0.0;
    var vy = 0.0;
    var vz = 0.0;
    if (hh < 4u) {
        vx = 0.0;
        vy = 1.0;
        vz = 0.0;
    } else if (hh == 12u || hh == 14u) {
        vx = 1.0;
        vy = 0.0;
        vz = 0.0;
    } else {
        vx = 0.0;
        vy = 0.0;
        vz = 1.0;
    }
    var su = 1.0;
    if ((hh & 1u) != 0u) {
        su = -1.0;
    }
    var sv = 1.0;
    if ((hh & 2u) != 0u) {
        sv = -1.0;
    }
    return vec3<f32>(su * ux + sv * vx, su * uy + sv * vy, sv * vz);
}

// The quintic fade 6t^5 - 15t^4 + 10t^3 in Horner form (multiply / add only).
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Linear interpolation a + t * (b - a).
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

// Signed cell value in [-1, 1) hashed from a lattice cell, for value noise.
fn cell_value(i: i32, j: i32, k: i32, seed: u32) -> f32 {
    let h = hash_lattice(i, j, k, seed);
    let unit = f32(h >> 8u) * INV_2POW24;
    return unit * 2.0 - 1.0;
}

// Three-dimensional value noise sampled at pos with the given seed.
fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let fx = floor(pos.x);
    let fy = floor(pos.y);
    let fz = floor(pos.z);
    let xi = i32(fx);
    let yi = i32(fy);
    let zi = i32(fz);
    let xf = pos.x - fx;
    let yf = pos.y - fy;
    let zf = pos.z - fz;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let c000 = cell_value(xi, yi, zi, seed);
    let c100 = cell_value(xi + 1, yi, zi, seed);
    let c010 = cell_value(xi, yi + 1, zi, seed);
    let c110 = cell_value(xi + 1, yi + 1, zi, seed);
    let c001 = cell_value(xi, yi, zi + 1, seed);
    let c101 = cell_value(xi + 1, yi, zi + 1, seed);
    let c011 = cell_value(xi, yi + 1, zi + 1, seed);
    let c111 = cell_value(xi + 1, yi + 1, zi + 1, seed);

    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    return lerp(y0, y1, w);
}

// The gradient contribution of one lattice corner: its hashed gradient dotted
// with the displacement from the corner to the sample point.
fn corner_grad(i: i32, j: i32, k: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    return dot(grad_select(hash_lattice(i, j, k, seed)), vec3<f32>(dx, dy, dz));
}

// Three-dimensional Perlin-style gradient noise sampled at pos.
fn gradient_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let fx = floor(pos.x);
    let fy = floor(pos.y);
    let fz = floor(pos.z);
    let xi = i32(fx);
    let yi = i32(fy);
    let zi = i32(fz);
    let xf = pos.x - fx;
    let yf = pos.y - fy;
    let zf = pos.z - fz;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let g000 = corner_grad(xi, yi, zi, xf, yf, zf, seed);
    let g100 = corner_grad(xi + 1, yi, zi, xf - 1.0, yf, zf, seed);
    let g010 = corner_grad(xi, yi + 1, zi, xf, yf - 1.0, zf, seed);
    let g110 = corner_grad(xi + 1, yi + 1, zi, xf - 1.0, yf - 1.0, zf, seed);
    let g001 = corner_grad(xi, yi, zi + 1, xf, yf, zf - 1.0, seed);
    let g101 = corner_grad(xi + 1, yi, zi + 1, xf - 1.0, yf, zf - 1.0, seed);
    let g011 = corner_grad(xi, yi + 1, zi + 1, xf, yf - 1.0, zf - 1.0, seed);
    let g111 = corner_grad(xi + 1, yi + 1, zi + 1, xf - 1.0, yf - 1.0, zf - 1.0, seed);

    let x00 = lerp(g000, g100, u);
    let x10 = lerp(g010, g110, u);
    let x01 = lerp(g001, g101, u);
    let x11 = lerp(g011, g111, u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    return lerp(y0, y1, w);
}

// Amplitude-normalised sum of gradient-noise octaves sampled at pos. The loop is
// bounded by MAX_OCTAVES and stops at the requested octave count; zero octaves
// (or a fully collapsed amplitude) yield 0.0 rather than a NaN.
fn fbm(pos: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32, seed: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    var octave_seed = seed;

    for (var i = 0u; i < MAX_OCTAVES; i = i + 1u) {
        if (i >= octaves) {
            break;
        }
        sum = sum + gradient_noise_3d(pos * freq, octave_seed) * amp;
        norm = norm + amp;
        freq = freq * lacunarity;
        amp = amp * gain;
        octave_seed = octave_seed + SEED_STEP;
    }

    if (abs(norm) > EPS) {
        return sum / norm;
    }
    return 0.0;
}

// The three-component vector potential whose curl becomes the noise flow; the
// seed is salted per channel so the channels are statistically independent.
fn potential(pos: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32, seed: u32) -> vec3<f32> {
    return vec3<f32>(
        fbm(pos, octaves, lacunarity, gain, seed),
        fbm(pos, octaves, lacunarity, gain, seed ^ SEED_SALT_Y),
        fbm(pos, octaves, lacunarity, gain, seed ^ SEED_SALT_Z),
    );
}

// Analytic curl of the vector potential by central differences with half-step
// CURL_EPS.
fn curl_of_potential(pos: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32, seed: u32) -> vec3<f32> {
    let e = CURL_EPS;
    let inv = 1.0 / (2.0 * e);

    let px = potential(pos + vec3<f32>(e, 0.0, 0.0), octaves, lacunarity, gain, seed);
    let mx = potential(pos - vec3<f32>(e, 0.0, 0.0), octaves, lacunarity, gain, seed);
    let py = potential(pos + vec3<f32>(0.0, e, 0.0), octaves, lacunarity, gain, seed);
    let my = potential(pos - vec3<f32>(0.0, e, 0.0), octaves, lacunarity, gain, seed);
    let pz = potential(pos + vec3<f32>(0.0, 0.0, e), octaves, lacunarity, gain, seed);
    let mz = potential(pos - vec3<f32>(0.0, 0.0, e), octaves, lacunarity, gain, seed);

    let dpsi_dx = (px - mx) * inv;
    let dpsi_dy = (py - my) * inv;
    let dpsi_dz = (pz - mz) * inv;

    return vec3<f32>(
        dpsi_dy.z - dpsi_dz.y,
        dpsi_dz.x - dpsi_dx.z,
        dpsi_dx.y - dpsi_dy.x,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let pos = q.pos;
    let seed = q.seed;

    // hash_lattice / lattice_gradient at the explicit integer cell.
    let hash = hash_lattice(q.cell.x, q.cell.y, q.cell.z, seed);
    let grad = grad_select(hash);

    // Scalar value and gradient noise, and the fBm sum, at pos.
    let value_noise = value_noise_3d(pos, seed);
    let gradient_noise = gradient_noise_3d(pos, seed);
    let fbm_value = fbm(pos, q.fbm_octaves, q.fbm_lacunarity, q.fbm_gain, seed);

    // Single-octave curl noise reuses the canonical SINGLE_OCTAVE structure.
    let curl3 = curl_of_potential(pos, 1u, 2.0, 0.5, seed);
    // Multi-octave curl noise uses the query fBm structure.
    let curl_fbm = curl_of_potential(pos, q.fbm_octaves, q.fbm_lacunarity, q.fbm_gain, seed);

    // Turbulence force: scale pos by frequency, curl the turbulence fBm, scale
    // by amplitude.
    let sample = pos * q.turb_frequency;
    let turb = curl_of_potential(
        sample,
        q.turb_fbm_octaves,
        q.turb_fbm_lacunarity,
        q.turb_fbm_gain,
        q.turb_seed,
    ) * q.turb_amplitude;

    var out: Result;
    out.hash = hash;
    out.value_noise = value_noise;
    out.gradient_noise = gradient_noise;
    out.fbm_value = fbm_value;
    out.lattice_gradient = grad;
    out.pad0 = 0.0;
    out.curl3 = curl3;
    out.pad1 = 0.0;
    out.curl_fbm = curl_fbm;
    out.pad2 = 0.0;
    out.turbulence = turb;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// Fractional-Brownian-motion parameters: how many octaves of noise are summed
/// and how frequency and amplitude evolve between them.
///
/// Host-side `Pod` replica of the reference
/// [`FbmParams`](prism_render_architecture::particle::noise::FbmParams):
/// `octaves` is the summed octave count (zero yields a flat field),
/// `lacunarity` is the per-octave frequency multiplier (canonically `2.0`) and
/// `gain` is the per-octave amplitude multiplier (canonically `0.5`). It carries
/// an `f32`, so it derives `PartialEq` only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuFbmParams {
    /// Number of noise octaves summed. Zero yields a flat (zero) field.
    pub octaves: u32,
    /// Per-octave frequency multiplier (`> 1` adds finer detail each octave).
    pub lacunarity: f32,
    /// Per-octave amplitude multiplier (`< 1` fades finer octaves out).
    pub gain: f32,
}

impl GpuFbmParams {
    /// The cinematic default: four octaves, `lacunarity` `2.0`, `gain` `0.5`.
    pub const DEFAULT: Self = Self {
        octaves: 4,
        lacunarity: 2.0,
        gain: 0.5,
    };

    /// A single octave: `fbm` then reduces to plain `gradient_noise_3d`.
    pub const SINGLE_OCTAVE: Self = Self {
        octaves: 1,
        lacunarity: 2.0,
        gain: 0.5,
    };

    /// Builds parameters from explicit octave count, `lacunarity` and `gain`.
    #[must_use]
    pub const fn new(octaves: u32, lacunarity: f32, gain: f32) -> GpuFbmParams {
        GpuFbmParams {
            octaves,
            lacunarity,
            gain,
        }
    }
}

impl Default for GpuFbmParams {
    fn default() -> GpuFbmParams {
        GpuFbmParams::DEFAULT
    }
}

/// Controls for turning curl noise into an applied turbulence force.
///
/// Host-side replica of the reference
/// [`TurbulenceParams`](prism_render_architecture::particle::noise::TurbulenceParams):
/// `frequency` scales the sample position, `amplitude` scales the resulting
/// force, `fbm` selects the octave structure of the vector potential and `seed`
/// re-rolls the field. It carries `f32` fields, so it derives `PartialEq` only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTurbulenceParams {
    /// Spatial frequency: larger values shrink the swirls.
    pub frequency: f32,
    /// Force magnitude scale applied to the divergence-free field.
    pub amplitude: f32,
    /// Octave structure of the vector potential.
    pub fbm: GpuFbmParams,
    /// Hash seed selecting the field instance.
    pub seed: u32,
}

impl GpuTurbulenceParams {
    /// A reasonable cinematic default: unit frequency and amplitude with the
    /// default `fBm` octave structure.
    pub const DEFAULT: Self = Self {
        frequency: 1.0,
        amplitude: 1.0,
        fbm: GpuFbmParams::DEFAULT,
        seed: 0x00C0_FFEE,
    };
}

impl Default for GpuTurbulenceParams {
    fn default() -> GpuTurbulenceParams {
        GpuTurbulenceParams::DEFAULT
    }
}

/// One procedural-noise query: a sample position, an integer lattice cell, a
/// base seed, and the `fBm` / turbulence parameters the kernel evaluates.
///
/// The `pos` drives `value_noise_3d`, `gradient_noise_3d`, `fbm`, `curl_noise_3d`
/// and `curl_noise_fbm`; the `cell` is the integer argument of `hash_lattice`
/// and `lattice_gradient`; `turbulence` drives `turbulence_force`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseQuery {
    /// Sample position for the continuous noise fields.
    pub pos: [f32; 3],
    /// Integer lattice cell for `hash_lattice` and `lattice_gradient`.
    pub cell: [i32; 3],
    /// Base hash seed shared by the scalar / vector noise fields.
    pub seed: u32,
    /// Octave structure for `fbm` and `curl_noise_fbm`.
    pub fbm: GpuFbmParams,
    /// Parameters for `turbulence_force`.
    pub turbulence: GpuTurbulenceParams,
}

impl NoiseQuery {
    /// Builds a query from its position, cell, seed and parameters.
    #[must_use]
    pub const fn new(
        pos: [f32; 3],
        cell: [i32; 3],
        seed: u32,
        fbm: GpuFbmParams,
        turbulence: GpuTurbulenceParams,
    ) -> NoiseQuery {
        NoiseQuery {
            pos,
            cell,
            seed,
            fbm,
            turbulence,
        }
    }
}

/// The resolved answer for one query, mirroring every field the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseResult {
    /// `hash_lattice` of the query cell and seed (bit-exact).
    pub hash: u32,
    /// `lattice_gradient` of the query cell and seed.
    pub lattice_gradient: [f32; 3],
    /// `value_noise_3d` at the query position.
    pub value_noise: f32,
    /// `gradient_noise_3d` at the query position.
    pub gradient_noise: f32,
    /// `fbm` at the query position with the query `fBm` parameters.
    pub fbm: f32,
    /// `curl_noise_3d` (single octave) at the query position.
    pub curl_noise_3d: [f32; 3],
    /// `curl_noise_fbm` at the query position with the query `fBm` parameters.
    pub curl_noise_fbm: [f32; 3],
    /// `turbulence_force` at the query position with the turbulence parameters.
    pub turbulence_force: [f32; 3],
}

/// `repr(C)` `std430` layout of one packed query: five `16`-byte slots holding
/// `(pos.xyz, pad)`, `(cell.xyz, pad)`, `(seed, fbm_octaves, fbm_lacunarity,
/// fbm_gain)`, `(turb_seed, turb_fbm_octaves, turb_fbm_lacunarity,
/// turb_fbm_gain)` and `(turb_frequency, turb_amplitude, pad, pad)` — `80`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Sample position.
    pos: [f32; 3],
    /// Padding lane after the position.
    pad0: f32,
    /// Integer lattice cell.
    cell: [i32; 3],
    /// Padding lane after the cell.
    pad1: i32,
    /// Base hash seed.
    seed: u32,
    /// `fBm` octave count.
    fbm_octaves: u32,
    /// `fBm` lacunarity.
    fbm_lacunarity: f32,
    /// `fBm` gain.
    fbm_gain: f32,
    /// Turbulence hash seed.
    turb_seed: u32,
    /// Turbulence `fBm` octave count.
    turb_fbm_octaves: u32,
    /// Turbulence `fBm` lacunarity.
    turb_fbm_lacunarity: f32,
    /// Turbulence `fBm` gain.
    turb_fbm_gain: f32,
    /// Turbulence frequency.
    turb_frequency: f32,
    /// Turbulence amplitude.
    turb_amplitude: f32,
    /// Padding word.
    pad2: u32,
    /// Padding word.
    pad3: u32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &NoiseQuery) -> GpuQuery {
        GpuQuery {
            pos: query.pos,
            pad0: 0.0,
            cell: query.cell,
            pad1: 0,
            seed: query.seed,
            fbm_octaves: query.fbm.octaves,
            fbm_lacunarity: query.fbm.lacunarity,
            fbm_gain: query.fbm.gain,
            turb_seed: query.turbulence.seed,
            turb_fbm_octaves: query.turbulence.fbm.octaves,
            turb_fbm_lacunarity: query.turbulence.fbm.lacunarity,
            turb_fbm_gain: query.turbulence.fbm.gain,
            turb_frequency: query.turbulence.frequency,
            turb_amplitude: query.turbulence.amplitude,
            pad2: 0,
            pad3: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(hash, value_noise, gradient_noise, fbm)` then four `16`-byte `vec3` slots
/// for the lattice gradient, single-octave curl, `fBm` curl and turbulence
/// force — `80` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `hash_lattice` output.
    hash: u32,
    /// `value_noise_3d` output.
    value_noise: f32,
    /// `gradient_noise_3d` output.
    gradient_noise: f32,
    /// `fbm` output.
    fbm_value: f32,
    /// `lattice_gradient` output.
    lattice_gradient: [f32; 3],
    /// Padding lane after the lattice gradient.
    pad0: f32,
    /// `curl_noise_3d` output.
    curl3: [f32; 3],
    /// Padding lane after the single-octave curl.
    pad1: f32,
    /// `curl_noise_fbm` output.
    curl_fbm: [f32; 3],
    /// Padding lane after the `fBm` curl.
    pad2: f32,
    /// `turbulence_force` output.
    turbulence: [f32; 3],
    /// Padding lane after the turbulence force.
    pad3: f32,
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

/// Decodes one packed [`GpuResult`] into the public [`NoiseResult`].
fn decode_result(raw: &GpuResult) -> NoiseResult {
    NoiseResult {
        hash: raw.hash,
        lattice_gradient: raw.lattice_gradient,
        value_noise: raw.value_noise,
        gradient_noise: raw.gradient_noise,
        fbm: raw.fbm_value,
        curl_noise_3d: raw.curl3,
        curl_noise_fbm: raw.curl_fbm,
        turbulence_force: raw.turbulence,
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

/// A compiled, reusable procedural-noise compute pipeline.
pub struct GpuNoise {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuNoise {
    /// Compiles the procedural-noise kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNoise {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_noise"),
            source: ShaderSource::Wgsl(NOISE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_noise_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_noise_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_noise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuNoise {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one [`NoiseResult`] per input,
    /// in order.
    ///
    /// The `hash` field equals the reference `hash_lattice` exactly; the
    /// continuous fields match the reference to within the tolerance documented
    /// on this module. An empty input returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[NoiseQuery]) -> Vec<NoiseResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_noise_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_noise_output"),
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
            label: Some("prism_volumetric_noise_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_noise_bind_group"),
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
            label: Some("prism_volumetric_noise_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_noise_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_noise_pass"),
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
