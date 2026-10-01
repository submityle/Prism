//! `wgpu` compute twin of the screen-space heat-distortion `UV` offset golden
//! ([`heat_distortion`](prism_render_architecture::particle::heat_distortion),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`heat_distortion`](prism_render_architecture::particle::heat_distortion)
//! owns the deterministic post-process heat-haze model a compositor layers over
//! hot emitters: a particle contributes a small screen-space `UV` perturbation
//! that bends the `HDR` scene color sampled behind it. It exposes four
//! independent pieces: a normal-driven base offset
//! ([`uv_offset`](prism_render_architecture::particle::heat_distortion::uv_offset),
//! `normal_xy * strength * scale`), a rational-polynomial distance falloff
//! ([`distance_falloff`](prism_render_architecture::particle::heat_distortion::HeatParams::distance_falloff),
//! `1 / (1 + k1 d + k2 d^2)`), a hash value-noise rolling phase perturbation
//! ([`rolling_offset`](prism_render_architecture::particle::heat_distortion::rolling_offset)),
//! and the two clamps
//! ([`clamp_offset`](prism_render_architecture::particle::heat_distortion::clamp_offset)
//! and
//! [`apply_to_uv`](prism_render_architecture::particle::heat_distortion::apply_to_uv))
//! that bound the magnitude and keep the distorted `UV` inside the sampled
//! `[0, 1]` domain. The combined
//! [`evaluate`](prism_render_architecture::particle::heat_distortion::HeatParams::evaluate)
//! chains the base offset, falloff, magnitude clamp and domain clamp into a
//! `HeatSample`.
//!
//! [`GpuHeatDistortion`] is the on-device twin: one thread per sample
//! reproduces the same closed form branch for branch, so a passing real-device
//! parity test is direct evidence the ported kernel evaluates the same model,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Per sample the kernel reproduces the full
//! [`evaluate`](prism_render_architecture::particle::heat_distortion::HeatParams::evaluate)
//! output (the clamped `offset` and the `distorted_uv`), which internally
//! exercises
//! [`uv_offset`](prism_render_architecture::particle::heat_distortion::uv_offset),
//! [`distance_falloff`](prism_render_architecture::particle::heat_distortion::HeatParams::distance_falloff),
//! [`clamp_offset`](prism_render_architecture::particle::heat_distortion::clamp_offset)
//! and
//! [`apply_to_uv`](prism_render_architecture::particle::heat_distortion::apply_to_uv),
//! plus the standalone
//! [`rolling_offset`](prism_render_architecture::particle::heat_distortion::rolling_offset)
//! shimmer, whose hash value-noise field (`hash_cell`, `cell_value`, `fade`,
//! `lerp` and `value_noise_1d`) is reproduced bit for bit.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `floor`, `+ - * /`, unsigned shift / xor / multiply and one
//! `bitcast` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! The integer hash is pure unsigned-integer `FNV`-style mixing (seed xor the
//! `FNV` basis, one `rotate`/multiply fold, a final `xorshift`-multiply
//! avalanche), and `WGSL` unsigned integers wrap on overflow exactly like
//! Rust's `wrapping_mul` / `^` / `>>`, so the lattice `hash` and therefore the
//! `cell_value`, `fade`, `lerp` and `value_noise_1d` results are bit-identical
//! between host and device. The surrounding continuous algebra (the base
//! offset, the rational falloff, the clamps) is a fixed, non-reorderable
//! sequence of multiplies, adds and divides, so `CPU` and `GPU` evaluate the
//! same closed form in the same associativity. They are not bit-exact there: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the `f32` fields, tight enough to catch a genuinely
//! wrong port (a dropped branch, a swapped coefficient, a wrong clamp or a
//! mis-twinned hash constant) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
//! integer-hash value-noise shimmer plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::heat_distortion::{
    rolling_offset, HeatParams, HeatSample,
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` heat-distortion kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`heat_distortion`](prism_render_architecture::particle::heat_distortion)
/// branch for branch; see the module documentation for the algorithm.
const HEAT_DISTORTION_WGSL: &str = r#"
// Heat-distortion twin: one thread per sample reproduces the clamped UV offset
// and the distorted UV from the combined evaluate(), plus the standalone
// rolling_offset shimmer. It mirrors the CPU golden particle::heat_distortion
// branch for branch, uses only the portable core-WGSL subset (clamp/min/max/
// floor, + - * /, unsigned shift/xor/multiply and one bitcast) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::heat_distortion; no
// third-party engine source or derived code.

// Scale that turns a 24-bit hash mantissa into the half-open range [0, 1);
// matches the reference INV_2POW24.
const INV_2POW24: f32 = 1.0 / 16777216.0;

// Odd-integer salt seeding the horizontal rolling-noise channel; matches
// ROLL_SEED_X.
const ROLL_SEED_X: u32 = 0x68e31da4u;
// Odd-integer salt seeding the vertical rolling-noise channel; matches
// ROLL_SEED_Y.
const ROLL_SEED_Y: u32 = 0xb5439c13u;
// Smallest denominator allowed in the distance falloff; matches
// MIN_FALLOFF_DENOM.
const MIN_FALLOFF_DENOM: f32 = 1.0e-3;

// FNV-1a offset basis xored into the seed; matches `hash_cell`.
const HASH_BASIS: u32 = 0x811c9dc5u;
// Multiplier xored into the folded input word; matches `mix`.
const MIX_MUL_A: u32 = 0x9e3779b1u;
// Post-rotate multiplier of the mixing fold; matches `mix`.
const MIX_MUL_B: u32 = 0x85ebca6bu;
// First avalanche multiplier; matches `finalize`.
const FIN_MUL_A: u32 = 0x7feb352du;
// Second avalanche multiplier; matches `finalize`.
const FIN_MUL_B: u32 = 0x846ca68bu;

struct Params {
    // Number of samples in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // View-space normal xy driving the base offset.
    nx: f32,
    ny: f32,
    // Per-emitter refraction strength and global distortion scale.
    strength: f32,
    scale: f32,
    // Linear and quadratic distance-falloff coefficients.
    k1: f32,
    k2: f32,
    // Maximum absolute UV offset per component after falloff.
    max_offset: f32,
    // Distance from the emitter/camera for the falloff.
    distance: f32,
    // Sampled scene UV.
    uvx: f32,
    uvy: f32,
    // Base UV driving the rolling shimmer.
    buvx: f32,
    buvy: f32,
    // Rolling phase and frequency.
    phase: f32,
    freq: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Clamped UV offset, distorted UV and the rolling shimmer, with two pad
    // lanes filling the second vec4 slot.
    offx: f32,
    offy: f32,
    duvx: f32,
    duvy: f32,
    rollx: f32,
    rolly: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// One folding step of the hash: xor-in a multiplied input word, then rotate
// left 15 and multiply, mirroring the reference `mix`. WGSL unsigned shift and
// multiply wrap exactly like Rust's `wrapping_mul` / rotate_left.
fn hash_mix(h_in: u32, word: u32) -> u32 {
    var h = h_in ^ (word * MIX_MUL_A);
    let rotated = (h << 15u) | (h >> 17u);
    return rotated * MIX_MUL_B;
}

// Final avalanche applied once after the word is folded, mirroring the
// reference `finalize`.
fn hash_finalize(h_in: u32) -> u32 {
    var h = h_in;
    h = h ^ (h >> 16u);
    h = h * FIN_MUL_A;
    h = h ^ (h >> 15u);
    h = h * FIN_MUL_B;
    h = h ^ (h >> 16u);
    return h;
}

// Stateless integer hash of a 1D lattice cell and seed, mirroring `hash_cell`.
// Negative cells address the whole signed lattice through the two's-complement
// reinterpret of the i32 cell index.
fn hash_cell(i: i32, seed: u32) -> u32 {
    var h = seed ^ HASH_BASIS;
    h = hash_mix(h, bitcast<u32>(i));
    return hash_finalize(h);
}

// The reproducible scalar value of a lattice cell in [-1, 1), mirroring
// `cell_value`. `(hash >> 8)` is a 24-bit integer, exactly representable in
// f32, so the conversion matches the reference `as f32`.
fn cell_value(i: i32, seed: u32) -> f32 {
    let h = hash_cell(i, seed);
    let unit = f32(h >> 8u) * INV_2POW24;
    return unit * 2.0 - 1.0;
}

// The multiply-only smoothstep fade `t * t * (3 - 2 t)`, mirroring `fade`.
fn fade(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Linear interpolation `a + (b - a) * t`, mirroring `lerp`.
fn lerp_f(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Smoothstep-faded 1D value noise in [-1, 1], mirroring `value_noise_1d`. The
// floor split uses `floor` then an exact integer cast, like `floor_split`.
fn value_noise_1d(t: f32, seed: u32) -> f32 {
    let fl = floor(t);
    let i = i32(fl);
    let f = t - fl;
    let c0 = cell_value(i, seed);
    let c1 = cell_value(i + 1, seed);
    return lerp_f(c0, c1, fade(f));
}

// The base screen-space UV offset from a surface normal, mirroring `uv_offset`.
fn uv_offset(nx: f32, ny: f32, strength: f32, scale: f32) -> vec2<f32> {
    let k = strength * scale;
    return vec2<f32>(nx * k, ny * k);
}

// The animated rolling perturbation sampled from the hash value-noise field,
// mirroring `rolling_offset`.
fn rolling_offset(buvx: f32, buvy: f32, phase: f32, freq: f32) -> vec2<f32> {
    let tx = buvx * freq + phase;
    let ty = buvy * freq + phase;
    return vec2<f32>(
        value_noise_1d(tx, ROLL_SEED_X),
        value_noise_1d(ty, ROLL_SEED_Y)
    );
}

// Clamps each component to [-max_offset, max_offset]; a negative bound collapses
// to zero. Mirrors `clamp_offset`.
fn clamp_offset(offset: vec2<f32>, max_offset: f32) -> vec2<f32> {
    let m = max(max_offset, 0.0);
    return vec2<f32>(clamp(offset.x, -m, m), clamp(offset.y, -m, m));
}

// Applies an offset to a UV and clamps into [0, 1], mirroring `apply_to_uv`.
fn apply_to_uv(uv: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        clamp(uv.x + offset.x, 0.0, 1.0),
        clamp(uv.y + offset.y, 0.0, 1.0)
    );
}

// The rational-polynomial distance falloff in (0, 1], mirroring
// `distance_falloff`: `1 / (1 + k1 d + k2 d^2)` on the clamped distance, with
// the denominator floored and the result clamped to [0, 1].
fn distance_falloff(d: f32, k1: f32, k2: f32) -> f32 {
    let dd = max(d, 0.0);
    let denom = max(1.0 + k1 * dd + k2 * dd * dd, MIN_FALLOFF_DENOM);
    return clamp(1.0 / denom, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // evaluate(): base normal offset, distance falloff, magnitude clamp and
    // UV-domain clamp, mirroring HeatParams::evaluate.
    let base = uv_offset(q.nx, q.ny, q.strength, q.scale);
    let fall = distance_falloff(q.distance, q.k1, q.k2);
    let scaled = vec2<f32>(base.x * fall, base.y * fall);
    let offset = clamp_offset(scaled, q.max_offset);
    let uv = vec2<f32>(q.uvx, q.uvy);
    let distorted = apply_to_uv(uv, offset);

    // The standalone rolling shimmer.
    let roll = rolling_offset(q.buvx, q.buvy, q.phase, q.freq);

    var out: Result;
    out.offx = offset.x;
    out.offy = offset.y;
    out.duvx = distorted.x;
    out.duvy = distorted.y;
    out.rollx = roll.x;
    out.rolly = roll.y;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One heat-distortion sample: the `HeatParams` scalars plus the per-particle
/// inputs the golden
/// [`evaluate`](prism_render_architecture::particle::heat_distortion::HeatParams::evaluate)
/// and
/// [`rolling_offset`](prism_render_architecture::particle::heat_distortion::rolling_offset)
/// consume.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatQuery {
    /// View-space normal `xy` driving the base offset.
    pub normal_xy: [f32; 2],
    /// Per-emitter refraction strength multiplying the normal offset.
    pub strength: f32,
    /// Global screen-space displacement scale (in `UV` units).
    pub distortion_scale: f32,
    /// Linear distance-falloff coefficient `k1`.
    pub falloff_k1: f32,
    /// Quadratic distance-falloff coefficient `k2`.
    pub falloff_k2: f32,
    /// Maximum absolute `UV` offset per component after falloff.
    pub max_offset: f32,
    /// Distance for the rational falloff.
    pub distance: f32,
    /// Sampled scene `UV`.
    pub uv: [f32; 2],
    /// Base `UV` driving the rolling shimmer.
    pub base_uv: [f32; 2],
    /// Rolling phase advancing the shimmer.
    pub phase: f32,
    /// Rolling frequency scaling the `base_uv`.
    pub freq: f32,
}

impl HeatQuery {
    /// Builds a sample from its fields.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
    /// no third-party engine source or derived code.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the golden HeatParams scalars plus the per-sample inputs"
    )]
    pub const fn new(
        normal_xy: [f32; 2],
        strength: f32,
        distortion_scale: f32,
        falloff_k1: f32,
        falloff_k2: f32,
        max_offset: f32,
        distance: f32,
        uv: [f32; 2],
        base_uv: [f32; 2],
        phase: f32,
        freq: f32,
    ) -> HeatQuery {
        HeatQuery {
            normal_xy,
            strength,
            distortion_scale,
            falloff_k1,
            falloff_k2,
            max_offset,
            distance,
            uv,
            base_uv,
            phase,
            freq,
        }
    }
}

/// The resolved answer for one sample, mirroring the golden
/// [`evaluate`](prism_render_architecture::particle::heat_distortion::HeatParams::evaluate)
/// output and the standalone
/// [`rolling_offset`](prism_render_architecture::particle::heat_distortion::rolling_offset).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatResult {
    /// The clamped screen-space `UV` offset the compositor adds.
    pub offset: [f32; 2],
    /// The distorted `UV`, clamped into the `[0, 1]` sampled domain.
    pub distorted_uv: [f32; 2],
    /// The rolling value-noise shimmer per channel, in `[-1, 1]`.
    pub roll: [f32; 2],
}

/// Evaluates the `CPU` golden for one sample: the combined
/// [`evaluate`](prism_render_architecture::particle::heat_distortion::HeatParams::evaluate)
/// result plus the standalone
/// [`rolling_offset`](prism_render_architecture::particle::heat_distortion::rolling_offset).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &HeatQuery) -> HeatResult {
    let params = HeatParams::new(
        query.strength,
        query.distortion_scale,
        query.falloff_k1,
        query.falloff_k2,
        query.max_offset,
    );
    let sample: HeatSample = params.evaluate(query.normal_xy, query.distance, query.uv);
    let roll = rolling_offset(query.base_uv, query.phase, query.freq);
    HeatResult {
        offset: sample.offset,
        distorted_uv: sample.distorted_uv,
        roll,
    }
}

/// `repr(C)` `std430` layout of one packed sample: four `vec4` slots holding the
/// normal, the `HeatParams` scalars, the distance, the sampled `UV`, the base
/// `UV` and the rolling phase/frequency — `64` bytes matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Normal `x`.
    nx: f32,
    /// Normal `y`.
    ny: f32,
    /// Refraction strength.
    strength: f32,
    /// Distortion scale.
    scale: f32,
    /// Linear falloff coefficient.
    k1: f32,
    /// Quadratic falloff coefficient.
    k2: f32,
    /// Maximum absolute offset.
    max_offset: f32,
    /// Falloff distance.
    distance: f32,
    /// Sampled `UV` `x`.
    uvx: f32,
    /// Sampled `UV` `y`.
    uvy: f32,
    /// Base `UV` `x`.
    buvx: f32,
    /// Base `UV` `y`.
    buvy: f32,
    /// Rolling phase.
    phase: f32,
    /// Rolling frequency.
    freq: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one sample into its `std430` image.
    fn new(query: &HeatQuery) -> GpuQuery {
        GpuQuery {
            nx: query.normal_xy[0],
            ny: query.normal_xy[1],
            strength: query.strength,
            scale: query.distortion_scale,
            k1: query.falloff_k1,
            k2: query.falloff_k2,
            max_offset: query.max_offset,
            distance: query.distance,
            uvx: query.uv[0],
            uvy: query.uv[1],
            buvx: query.base_uv[0],
            buvy: query.base_uv[1],
            phase: query.phase,
            freq: query.freq,
            pad0: 0.0,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(offset.xy, distorted_uv.xy)` and `(roll.xy, pad, pad)` — `32` bytes
/// matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped offset `x`.
    offx: f32,
    /// Clamped offset `y`.
    offy: f32,
    /// Distorted `UV` `x`.
    duvx: f32,
    /// Distorted `UV` `y`.
    duvy: f32,
    /// Rolling shimmer `x`.
    rollx: f32,
    /// Rolling shimmer `y`.
    rolly: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of samples in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable heat-distortion compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
/// no third-party engine source or derived code.
pub struct GpuHeatDistortion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHeatDistortion {
    /// Compiles the heat-distortion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHeatDistortion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_heat_distortion"),
            source: ShaderSource::Wgsl(HEAT_DISTORTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_heat_distortion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_heat_distortion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_heat_distortion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHeatDistortion {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every sample on-device and returns one [`HeatResult`] per input,
    /// in order.
    ///
    /// Each result equals the reference
    /// [`evaluate`](prism_render_architecture::particle::heat_distortion::HeatParams::evaluate)
    /// and
    /// [`rolling_offset`](prism_render_architecture::particle::heat_distortion::rolling_offset)
    /// answers to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[HeatQuery]) -> Vec<HeatResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_heat_distortion_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_heat_distortion_output"),
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
            label: Some("prism_volumetric_heat_distortion_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_heat_distortion_bind_group"),
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
            label: Some("prism_volumetric_heat_distortion_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_heat_distortion_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_heat_distortion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`HeatResult`].
fn decode_result(raw: &GpuResult) -> HeatResult {
    HeatResult {
        offset: [raw.offx, raw.offy],
        distorted_uv: [raw.duvx, raw.duvy],
        roll: [raw.rollx, raw.rolly],
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
